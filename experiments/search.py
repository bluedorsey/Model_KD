"""Best-response attacker (roadmap step 15): Optuna TPE over the README §12.2 strategy parameters.

  .venv/Scripts/python.exe -m experiments.search --defense none --trials 40
  .venv/Scripts/python.exe -m experiments.search --defense full --trials 40    # DCM vs the none search

Each trial builds one campaign from the sampled strategy (attacks.static.campaign), replays it with
background benign traffic through the defense, and measures the cost at which the --bound count
(experiments.run: lo = distinct canary prompts, hi = Q_eff) first reached the target: n_tau at tau = --tau
from the step-5 curve (mean over seeds), or, before the curve exists, 3,000 (the canary list a static A01
campaign covers). Objective (minimized): that cost; a trial that falls short scores cost x target / count,
an extrapolation that only steers the search. C is reported from reached trials only.
Query selection is random or coverage; student-guided arrives with step 16. The attacker knows the defense
(it searches against it) but not the HMAC key or the benign reference set.
"""
import argparse
import dataclasses
import json
import pickle

import duckdb
import numpy as np
import optuna

from attacks.static import Strategy, buckets, campaign, facts
from configs import load
from experiments.run import BOUNDS, TAUS, campaign_yield, curve, tau_targets, tuned
from gateway.replay import World, ref_path, run
from gateway.schema import load_pools


def background(cfg, frac):
    """benign_eval accounts with hash(account) mod round(1/frac) == 0, as a stream (built once per fraction)."""
    k = max(1, round(1 / frac))
    name = f"benign_eval_bg{k}"
    T = cfg["processed"] / "traffic"
    out = T / name
    if not (out / "requests.parquet").exists():
        out.mkdir(parents=True, exist_ok=True)
        gt = (T / "benign_eval" / "ground_truth.parquet").as_posix()
        for f in ("requests", "ground_truth"):
            duckdb.sql(f"""COPY (SELECT * FROM '{(T / 'benign_eval' / f'{f}.parquet').as_posix()}' WHERE account_id IN
                (SELECT DISTINCT account_id FROM '{gt}' WHERE hash(account_id) % {k} = 0))
                TO '{(out / f'{f}.parquet').as_posix()}' (FORMAT parquet)""")
    return name


def strategy(trial, horizon, max_requests):
    accounts = trial.suggest_int("accounts", 1, 500, log=True)
    return Strategy(
        accounts=accounts,
        queries_per_account=trial.suggest_int("queries_per_account", 10, max(10, max_requests // accounts), log=True),
        shard_overlap=trial.suggest_float("shard_overlap", 0.0, 0.5),
        timing=trial.suggest_categorical("timing", ["uniform", "diurnal", "burst"]),
        cover_ratio=trial.suggest_float("cover_ratio", 0.0, 0.95),
        infra_reuse=trial.suggest_int("infra_reuse", 1, 50, log=True),
        capability_mix=trial.suggest_float("capability_mix", 0.0, 1.0),
        paraphrase_rate=trial.suggest_float("paraphrase_rate", 0.0, 1.0),
        repeats_per_prompt=trial.suggest_int("repeats_per_prompt", 1, 4),
        rotation=trial.suggest_categorical("rotation", ["sequential", "round_robin", "random"]),
        query_selection=trial.suggest_categorical("query_selection", ["random", "coverage"]),
        days=trial.suggest_float("days", 1.0, float(horizon), log=True),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", default="full", choices=["none", "l0", "mmd", "seat", "simlink", "full"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.8, choices=TAUS)
    ap.add_argument("--target-n", type=int, help="override the curve target")
    ap.add_argument("--bound", default="hi", choices=BOUNDS)
    ap.add_argument("--background", type=float, default=0.25, help="share of benign_eval accounts replayed alongside")
    ap.add_argument("--max-requests", type=int, default=60000, help="planned target requests per campaign")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = tuned(load())
    cfg["gateway"]["coupled"] = a.defense == "full"
    world = World(cfg)
    ref = pickle.loads(ref_path(cfg).read_bytes())
    pools, fact_of = load_pools(cfg), facts(cfg)
    bucket_of = buckets(cfg, pools["canary_kb"] + pools["canary_lang"])
    cv = curve(cfg)
    ns = [n for (t, _), n in tau_targets(cv).items() if t == a.tau]
    target = a.target_n or (int(np.mean(ns)) if ns and all(ns) else 3000)
    bg = background(cfg, a.background)
    horizon = cfg["scale"]["days"]
    name = f"search_{a.defense}_{cfg['profile']}"
    log = (cfg["processed"] / "runs" / f"{name}.jsonl").open("w", encoding="utf-8")

    def objective(trial):
        s = strategy(trial, horizon, a.max_requests)
        reqs, gt = campaign(f"BR{trial.number:03d}", "BR", s, pools, fact_of, cfg, 1000 + a.seed, bucket_of=bucket_of)
        events, summary, _, _ = run(cfg, world, [bg], a.defense, ref, live_every=0, extra=reqs, log=lambda *_: None)
        mine = {g[0]: (g[2], g[3], g[4], g[5]) for g in gt}
        evs = [e for e in events if e["event_id"] in mine]
        c = next(iter(campaign_yield(evs, mine, world.pool_of, cfg["cost"], {f"BR{trial.number:03d}": s.accounts},
                                     [(a.bound, target)]).values()))
        reached = (a.bound, target) in c["cost_to"]
        score = c["cost_to"][a.bound, target] if reached else c["cost"] * target / max(c["n"][a.bound], 1)
        accts = {e["account_id"] for e in evs}
        row = {"trial": trial.number, **dataclasses.asdict(s), "requests": c["requests"], "Q_eff": len(c["resp"]),
               "n_lo": c["n"]["lo"], "n_hi": c["n"]["hi"], "cost": round(c["cost"], 2), "reached": reached,
               "C_target": round(c["cost_to"][a.bound, target], 2) if reached else None, "objective": round(score, 2),
               "share_accounts_tier2": round(sum(summary.get(x, {}).get("max_tier", 0) >= 2 for x in accts)
                                             / max(len(accts), 1), 4)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps({k: row[k] for k in ("trial", "accounts", "queries_per_account", "cover_ratio", "n_lo", "n_hi",
                                              "cost", "reached", "objective", "share_accounts_tier2")}), flush=True)
        trial.set_user_attr("row", row)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=a.seed))
    study.optimize(objective, n_trials=a.trials)
    rows = [t.user_attrs["row"] for t in study.trials if "row" in t.user_attrs]
    reached = [r for r in rows if r["reached"]]
    best = min(reached, key=lambda r: r["C_target"]) if reached else None
    out = {"defense": a.defense, "profile": cfg["profile"], "trials": len(rows), "sampler": "optuna TPE",
           "optuna": optuna.__version__, "seed": a.seed, "tau": a.tau, "bound": a.bound, "target_n": target,
           "target_source": "override" if a.target_n else ("curve" if ns and all(ns) else "3,000 (no curve yet)"),
           "background": bg, "policy_thresholds": {k: cfg["policy"][k] for k in ("p_tier1", "p_tier2", "p_tier3")},
           "versions": cfg["versions"], "reached_trials": len(reached),
           "C_best": best["C_target"] if best else None, "best": best}
    base = cfg["processed"] / "runs" / f"search_none_{cfg['profile']}.json"
    if a.defense != "none" and base.exists():
        b = json.loads(base.read_text(encoding="utf-8"))
        out["dcm_best_response"] = (round(out["C_best"] / b["C_best"], 3)
                                    if out["C_best"] and b["C_best"] and (b["target_n"], b.get("bound")) == (target, a.bound) else None)
    (cfg["processed"] / "runs" / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "versions"}, indent=1))


if __name__ == "__main__":
    main()
