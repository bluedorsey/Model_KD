"""Stage 1-2 baselines (roadmap step 12): global-aggregation monitor and the cost-to-fidelity table.

  .venv/Scripts/python.exe -m experiments.baselines global    # needs runs/none_plain_<profile>
  .venv/Scripts/python.exe -m experiments.baselines table     # every runs/*_<profile>/report.json

Replayed baselines live in experiments.run: `none`, `l0` (stage 1), `mmd`, `seat`, `simlink`, `none --coupled on` (stage 2).

global: an extraction monitor inspired by Kesarwani et al. (2018), coverage variant, adapted from feature-space
coverage of decision-tree inputs to text: per capability (tagger), the number of distinct prompts all accounts
together sent in the trailing 7 days. The alarm threshold is the benign 99th percentile of that daily series
on the benign reference traffic. It warns; it attributes nothing to accounts, so it cannot change attacker
yield (DCM = 1 by construction). Reported: false-alarm days on benign_eval alone, and per campaign the share
of its yield obtained before the first alarm while it was active. The information-gain variant needs model
internals (decision-tree paths) and is not reproduced.
"""
import json
import sys

import duckdb
import numpy as np

from configs import load
from experiments.run import BOUNDS, CANARY, read_gt
from gateway.replay import World
from gateway.schema import DAY

WINDOW = 7


def daily_coverage(rows, cap_of, days):
    """rows: (ts, prompt_hash) -> {capability: array of trailing-7-day distinct prompt counts per day}."""
    seen = {}
    for ts, h in rows:
        seen.setdefault(cap_of(h), {}).setdefault(int(ts // DAY), set()).add(h)
    out = {}
    for c, by_day in seen.items():
        out[c] = np.array([len(set().union(*[by_day.get(d, set()) for d in range(max(0, t - WINDOW + 1), t + 1)]))
                           for t in range(days)])
    return out


def global_monitor(cfg):
    world = World(cfg)
    cap_of = lambda h: world.tagger.names[world.tagger.cap_of_row[world.row_of[h]]]
    days = cfg["scale"]["days"]
    T = cfg["processed"] / "traffic"
    ref = duckdb.sql(f"SELECT ts, prompt_hash FROM '{(T / 'benign_ref' / 'requests.parquet').as_posix()}'").fetchall()
    tau = {c: float(np.quantile(v, 0.99)) for c, v in daily_coverage(ref, cap_of, days).items()}
    run = cfg["processed"] / "runs" / f"none_plain_{cfg['profile']}"
    ev = duckdb.sql(f"SELECT event_id, ts, prompt_hash, blocked, form, response_hash FROM "
                    f"'{(run / 'events.parquet').as_posix()}' ORDER BY ts").fetchall()
    gt = read_gt(cfg, ["benign_eval", "attacks_static"])
    benign_cov = daily_coverage([(t, h) for i, t, h, *_ in ev if gt[i][3] != "attack"], cap_of, days)
    all_cov = daily_coverage([(t, h) for _, t, h, *_ in ev], cap_of, days)
    alarm_days = sorted({d for c, v in all_cov.items() for d in np.flatnonzero(v > tau.get(c, np.inf)).tolist()})
    false_days = sorted({d for c, v in benign_cov.items() for d in np.flatnonzero(v > tau.get(c, np.inf)).tolist()})
    camps = {}
    for i, t, h, blocked, form, rh in ev:
        label, attack, cid, pop = gt[i]
        if pop == "attack" and label == "attack" and not blocked and form == "full" and world.pool_of[h] in CANARY:
            c = camps.setdefault(cid, {"attack": attack, "start": t, "resp": {}})
            for r in rh.split(","):
                c["resp"].setdefault(r, t)
    rows = {}
    for cid, c in sorted(camps.items()):
        first = next((d * DAY for d in alarm_days if (d + 1) * DAY > c["start"]), None)
        q = len(c["resp"])
        rows[cid] = {"attack": c["attack"], "Q_eff": q, "first_alarm_day": None if first is None else first / DAY,
                     "yield_before_alarm": round(sum(v < first for v in c["resp"].values()) / q, 4)
                     if first is not None else 1.0}
    out = {"baseline": "global coverage monitor (inspired by Kesarwani et al. 2018)", "profile": cfg["profile"],
           "window_days": WINDOW, "threshold_quantile": 0.99, "thresholds": tau, "days": days,
           "alarm_days": len(alarm_days), "benign_only_false_alarm_days": len(false_days),
           "benign_false_alarm_day_rate": round(len(false_days) / days, 4), "campaigns": rows,
           "note": "warning only: no enforcement, attacker yield unchanged (DCM = 1)"}
    f = cfg["processed"] / "runs" / f"global_monitor_{cfg['profile']}.json"
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "campaigns"}, indent=1))
    print(json.dumps({c: (r["attack"], r["first_alarm_day"], r["yield_before_alarm"]) for c, r in rows.items()}))


def table(cfg):
    runs = cfg["processed"] / "runs"
    lines = ["| Run | Q_eff total [95% CI] | C_0.5 lo / hi (seeds) | C_0.8 lo / hi (seeds) | DCM_0.5 lo / hi | "
             "DCM_0.8 lo / hi | Benign tier 2 | Benign tier 3 | Acct TPR@1% [CI] | Cluster TPR@1% [CI] |",
             "|" + " --- |" * 10]
    fmt = lambda xs: "/".join("n.r." if x is None else f"{x:g}" for x in xs) if xs else "n/a"
    for f in sorted(runs.glob(f"*_{cfg['profile']}*/report.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        fid, dcm, d, b = r.get("fidelity"), r.get("dcm") or {}, r["detection"], r["benign_cost"]
        pick = lambda d, t, b, f: [f(v) for _, v in sorted(d.get(t, {}).get(b, {}).items())]
        C = {t: " / ".join(fmt(pick(fid["tau"], t, b, lambda v: v["C_tau"])) for b in BOUNDS) if fid else "n/a"
             for t in ("0.5", "0.8")}
        D = {t: " / ".join(fmt(pick(dcm, t, b, lambda v: v)) for b in BOUNDS) if dcm else "n/a" for t in ("0.5", "0.8")}
        ci = r["attacker"].get("Q_eff_total_ci95", ["?", "?"])
        tpr = lambda k: f"{d[k]['tpr@1%fpr']['tpr']} [{d[k]['tpr@1%fpr']['ci95'][0]}, {d[k]['tpr@1%fpr']['ci95'][1]}]"
        lines.append(f"| {f.parent.name} | {r['attacker']['Q_eff_total']} [{ci[0]}, {ci[1]}] | {C['0.5']} | "
                     f"{C['0.8']} | {D['0.5']} | {D['0.8']} | {b['share_tier2']:.2%} | "
                     f"{b['share_tier3']:.2%} | {tpr('account')} | {tpr('cluster')} |")
    lines.append("\nC_tau: cheapest static campaign reaching tau x teacher canary accuracy, one value per step-5 curve "
                 "seed; lo = distinct prompts, hi = unique responses mapped to the curve; n.r. = no static campaign "
                 "reached it; n/a = curve not measured yet. DCM = C_tau / C_tau(none).")
    out = runs / f"table_{cfg['profile']}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"global": global_monitor, "table": table}[sys.argv[1]](load())
