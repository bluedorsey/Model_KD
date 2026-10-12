"""Replay one defense configuration over benign_eval + attack traffic and report README §12.3 metrics.

  .venv/Scripts/python.exe -m experiments.run --defense none      # stage 1: no defense
  .venv/Scripts/python.exe -m experiments.run --defense l0        # stage 1: quotas only
  .venv/Scripts/python.exe -m experiments.run --defense full      # rules + account + cluster + tiers + coupled

Fidelity is not reported yet: it needs the step-5 data-scaling curve (GPU). Q_eff counts unique
full-quality canary responses; with the stub teacher, uniqueness is (prompt, seed, form) exact.
"""
import argparse
import json
import pickle

import duckdb
import numpy as np

from configs import load
from gateway.replay import World, ref_path, run
from gateway.schema import EVENT, write

CANARY = ("canary_kb", "canary_lang")


def tpr_at_fpr(sb, sa, fpr, rng, boots=500):
    """Account-level TPR at a benign-account FPR, with a bootstrap 95% CI over accounts."""
    def one(b, a):
        t = np.quantile(b, 1 - fpr, method="higher")
        return float((a > t).mean()) if len(a) else float("nan")
    est = one(sb, sa)
    bs = [one(rng.choice(sb, len(sb)), rng.choice(sa, len(sa))) for _ in range(boots)]
    return {"tpr": round(est, 4), "ci95": [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]}


def auprc(sb, sa, prevalence):
    """Average precision with benign/attack accounts reweighted to the stated attacker prevalence."""
    s = np.concatenate([sa, sb])
    w = np.concatenate([np.full(len(sa), prevalence / len(sa)), np.full(len(sb), (1 - prevalence) / len(sb))])
    y = np.concatenate([np.ones(len(sa)), np.zeros(len(sb))])
    o = np.argsort(-s, kind="stable")
    tp, fp = np.cumsum(w[o] * y[o]), np.cumsum(w[o] * (1 - y[o]))
    prec = tp / np.maximum(tp + fp, 1e-300)
    return round(float((prec * w[o] * y[o]).sum() / prevalence), 4)


def report(cfg, world, events, summary, gw, defense, streams):
    T = cfg["processed"] / "traffic"
    files = ", ".join(f"'{(T / n / 'ground_truth.parquet').as_posix()}'" for n in streams)
    gt = {r[0]: r[1:] for r in duckdb.sql(f"SELECT request_id, traffic_label, attack_type, campaign_id, population "
                                         f"FROM read_parquet([{files}])").fetchall()}
    camp_meta = {c["campaign_id"]: c for c in json.loads((T / "attacks_static" / "campaigns.json")
                                                         .read_text(encoding="utf-8"))["campaigns"]}
    C = cfg["cost"]
    camps, pop_of = {}, {}
    for ev in events:
        label, attack, cid, pop = gt[ev["event_id"]]
        pop_of[ev["account_id"]] = pop
        if pop != "attack":
            continue
        c = camps.setdefault(cid, {"attack": attack, "requests": 0, "accounts": set(), "resp": {}, "prompts": set()})
        c["requests"] += 1
        c["accounts"].add(ev["account_id"])
        if label == "attack" and not ev["blocked"] and ev["form"] == "full" and world.pool_of[ev["prompt_hash"]] in CANARY:
            for rh in ev["response_hash"].split(","):
                c["resp"].setdefault(rh, ev["ts"])
            c["prompts"].add(ev["prompt_hash"])
    camp_rows = {}
    for cid, c in sorted(camps.items()):
        first = min([summary[a]["first_t2"] for a in c["accounts"] if a in summary and summary[a]["first_t2"]],
                    default=None)
        q = len(c["resp"])
        camp_rows[cid] = {"attack": c["attack"], "accounts": len(c["accounts"]), "requests": c["requests"],
                          "Q_eff": q, "distinct_canary_prompts": len(c["prompts"]),
                          "cost": round(c["requests"] * C["query"] + len(c["accounts"]) * C["account"], 2),
                          "yield_before_enforcement": round(sum(t < first for t in c["resp"].values()) / q, 4)
                          if (q and first is not None) else (1.0 if q else None),
                          "enforced": first is not None, "planned_requests": camp_meta[cid]["requests"]}
    benign = [a for a, p in pop_of.items() if p != "attack"]
    attack = [a for a, p in pop_of.items() if p == "attack"]
    get = lambda a, k, d: summary.get(a, {}).get(k, d)
    by_pop = {}
    for a in benign:
        b = by_pop.setdefault(pop_of[a], [0, 0, 0, 0])
        t = get(a, "max_tier", 0)
        b[0] += 1; b[1] += t >= 2; b[2] += t >= 3; b[3] += get(a, "verifications", 0)
    n_b = len(benign)
    rng = np.random.default_rng(0)
    det = {}
    for name, key in [("account", "min_p"), ("cluster", "min_pc")]:
        sb = np.array([-np.log10(get(a, key, 1.0)) for a in benign])
        sa = np.array([-np.log10(get(a, key, 1.0)) for a in attack])
        det[name] = {"tpr@1%fpr": tpr_at_fpr(sb, sa, 0.01, rng), "tpr@0.1%fpr": tpr_at_fpr(sb, sa, 0.001, rng),
                     "auprc@1%": auprc(sb, sa, 0.01), "auprc@0.1%": auprc(sb, sa, 0.001)}
    lat = np.array(gw.latency_ms) if gw.latency_ms else np.array([np.nan])
    return {
        "defense": defense, "profile": cfg["profile"], "coupled": cfg["gateway"]["coupled"],
        "versions": {"model": gw.versions[0], "encoder": gw.versions[1], "detector": gw.versions[2],
                     "policy": gw.versions[3], "pools": cfg["versions"]["pools"]},
        "events": len(events),
        "attacker": {"Q_eff_total": sum(r["Q_eff"] for r in camp_rows.values()), "campaigns": camp_rows},
        "benign_cost": {"accounts": n_b,
                        "share_tier2": round(sum(v[1] for v in by_pop.values()) / max(n_b, 1), 5),
                        "share_tier3": round(sum(v[2] for v in by_pop.values()) / max(n_b, 1), 5),
                        "by_population": {p: {"accounts": v[0], "tier2": v[1], "tier3": v[2], "verifications": v[3]}
                                          for p, v in sorted(by_pop.items())}},
        "detection": {"benign_accounts": n_b, "attack_accounts": len(attack), **det},
        "hot_path_ms": {"n_live": int(len(gw.latency_ms)), "p50": round(float(np.percentile(lat, 50)), 2),
                        "p95": round(float(np.percentile(lat, 95)), 2), "p99": round(float(np.percentile(lat, 99)), 2)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", default="full", choices=["none", "l0", "full"])
    ap.add_argument("--streams", nargs="+", default=["benign_eval", "attacks_static"])
    ap.add_argument("--coupled", default="auto", choices=["auto", "on", "off"],
                    help="auto: on for full only; 'none --coupled on' is the stage-2 coupled-sampling-alone baseline")
    ap.add_argument("--live-every", type=int, default=500)
    a = ap.parse_args()
    cfg = load()
    cfg["gateway"]["coupled"] = a.coupled == "on" or (a.coupled == "auto" and a.defense == "full")
    world = World(cfg)
    ref = pickle.loads(ref_path(cfg).read_bytes())
    events, summary, gw, _ = run(cfg, world, a.streams, a.defense, ref, live_every=a.live_every)
    name = f"{a.defense}_{'coupled' if cfg['gateway']['coupled'] else 'plain'}_{cfg['profile']}"
    out = cfg["processed"] / "runs" / name
    write([[e[c] for c in EVENT] for e in events], EVENT, out / "events.parquet")
    rep = report(cfg, world, events, summary, gw, a.defense, a.streams)
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("defense", "events", "benign_cost", "detection", "hot_path_ms")}, indent=1))
    print(json.dumps({c: (r["attack"], r["accounts"], r["requests"], r["Q_eff"], r["enforced"])
                      for c, r in rep["attacker"]["campaigns"].items()}))


if __name__ == "__main__":
    main()
