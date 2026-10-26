"""Tune tier thresholds to the benign caps (roadmap step 14 gate), on a benign-only population.

  python "Data/pre_processing pipline/benign.py" --name tune --seed 13
  .venv/Scripts/python.exe -m experiments.tune
  .venv/Scripts/python.exe -m experiments.tune --lr      # stage 6: thresholds with the LR in the scorer

One observe-mode replay of benign_tune records every account and cluster p-value per async batch. The
policy (hysteresis, sustained tier 2 -> 3, cluster inheritance, verification) is then replayed offline for
each candidate threshold set, so the search costs one replay. Observe mode never shapes output, so benign
features are at least as extreme as under enforcement: the measured shares are an upper bound.
Rule (pre-registered in configs/default.toml [policy]): largest p_tier2, then largest p_tier3 <= p_tier2/10,
with shares at most tune_margin x caps. Thresholds then apply to every experiments.run replay.
"""
import argparse
import json
import pickle

import numpy as np

from configs import load
from experiments.run import policy_path
from gateway.replay import World, ref_path, run
from policy.tiers import Policy


def simulate(trace, P):
    """Replay recorded p-values through the policy -> {account: max tier}. Every account here is benign,
    so tier 3 means verification, which a benign user passes (as in gateway.replay)."""
    pol, top = Policy({"policy": P}), {}
    for now, acc_up, cl_up in trace:
        changed = set()
        for a, p, inc, fr in acc_up:
            pol.update_account(a, p, now, incomplete=inc, frozen=fr)
            changed.add(a)
        for a, pc in cl_up:
            pol.update_cluster(a, pc, now)
        for a in changed | set(pol.cluster):
            t = pol.tier(a, now)
            top[a] = max(top.get(a, 0), t)
            if t == 3:
                pol.verified(a, now, True)
    return top


def shares(top, n):
    return sum(t >= 2 for t in top.values()) / n, sum(t >= 3 for t in top.values()) / n


def search(trace, n, P0):
    """-> (thresholds, share2, share3) by the pre-registered rule, or None if no candidate meets the caps."""
    t2, t3 = P0["tune_margin"] * P0["cap_tier2"], P0["tune_margin"] * P0["cap_tier3"]
    for p2 in np.logspace(-2, -10, 33):
        P = {**P0, "p_tier1": 10 * p2, "p_tier2": p2, "p_tier3": p2 / 10}
        s2, s3 = shares(simulate(trace, P), n)
        if s2 > t2:
            continue
        for p3 in np.logspace(np.log10(p2 / 10), -12, 25):
            P["p_tier3"] = p3
            s2, s3 = shares(simulate(trace, P), n)
            if s3 <= t3:
                return {k: float(P[k]) for k in ("p_tier1", "p_tier2", "p_tier3")}, s2, s3
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", action="store_true")
    cfg = load()
    cfg["scoring"]["use_lr"] = ap.parse_args().lr
    world = World(cfg)
    ref = pickle.loads(ref_path(cfg).read_bytes())
    trace = []
    events, _, _, _ = run(cfg, world, ["benign_tune"], "observe", ref, live_every=0, trace=trace)
    n = len({e["account_id"] for e in events})
    before = shares(simulate(trace, cfg["policy"]), n)
    found = search(trace, n, cfg["policy"])
    sweep = []  # benign shares across the p_tier2 grid with p_tier3 = p_tier2 / 10 (figure input)
    for p2 in np.logspace(-2, -10, 33):
        s2, s3 = shares(simulate(trace, {**cfg["policy"], "p_tier1": 10 * p2, "p_tier2": p2, "p_tier3": p2 / 10}), n)
        sweep.append({"p_tier2": float(p2), "share_tier2": round(s2, 6), "share_tier3": round(s3, 6)})
    out = {"stream": "benign_tune", "profile": cfg["profile"], "detector": cfg["versions"]["detector"], "lr": cfg["scoring"]["use_lr"],
           "policy": cfg["versions"]["policy"], "accounts": n, "batches": len(trace),
           "untuned": {"thresholds": {k: cfg["policy"][k] for k in ("p_tier1", "p_tier2", "p_tier3")},
                       "share_tier2": round(before[0], 5), "share_tier3": round(before[1], 5)},
           "rule": "largest p_tier2, then largest p_tier3 <= p_tier2/10, shares <= tune_margin x caps",
           "sweep": sweep}
    if found is None:
        print(json.dumps({**out, "result": "no threshold set meets the caps"}, indent=1))
        return
    th, s2, s3 = found
    out.update(thresholds=th, share_tier2=round(s2, 5), share_tier3=round(s3, 5))
    f = policy_path(cfg)
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "sweep"}, indent=1))


if __name__ == "__main__":
    main()
