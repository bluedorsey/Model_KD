"""Step 16 validation: do real students follow the curve-predicted fidelity? (README §12.3 rule 12). GPU.

  python -m student.validate

Cells: every campaign of the static replays none_plain, none_coupled, d4_plain and full_coupled with at least 250
distinct canary prompts, sorted by n_hi (unique responses); 10 at evenly spaced ranks. Each source configuration
is replayed once (deterministic, so it reproduces its report) to recover the served full-form canary responses
of the chosen campaigns; their deduplicated (prompt, response) pairs train the curve student with seeds 0-2.
Measured fidelity = mean canary accuracy / teacher accuracy, against the report's lo / hi prediction.
Resumable: one line per (cell, seed) in runs/validate/results.jsonl. Summary: Data/Processed/runs/validate_<p>.json.
"""
import json
import pickle

import numpy as np

from configs import ROOT, load
from experiments.run import CANARY, curve, read_gt, setup
from gateway.replay import World, ref_path, run
from gateway.schema import write
from student.scaling import train_eval

SOURCES = {"none_plain": ("none", "off"), "none_coupled": ("none", "on"), "d4_plain": ("d4", "auto"),
           "full_coupled": ("full", "auto")}
STREAMS = ["benign_eval", "attacks_static"]
N_CELLS, MIN_LO, WITHIN = 10, 250, 0.10


def cells(cfg):
    rows = []
    for src in SOURCES:
        f = cfg["processed"] / "runs" / f"{src}_{cfg['profile']}" / "report.json"
        if f.exists():
            for cid, c in json.loads(f.read_text(encoding="utf-8"))["attacker"]["campaigns"].items():
                if c["distinct_canary_prompts"] >= MIN_LO and c["fidelity"]:
                    rows.append((c["Q_eff"], src, cid, c))
    rows.sort(key=lambda r: r[:3])
    pick = sorted({int(round(i)) for i in np.linspace(0, len(rows) - 1, min(N_CELLS, len(rows)))}) if rows else []
    return [{"cell": f"{src}:{cid}", "source": src, "campaign": cid, "attack": c["attack"], "n_lo":
             c["distinct_canary_prompts"], "n_hi": c["Q_eff"], "predicted": c["fidelity"]}
            for _, src, cid, c in (rows[i] for i in pick)]


def served_pairs(src, wanted, world=None):
    """Replay one source configuration; -> {campaign: sorted unique (prompt, response) pairs}, world."""
    defense, coupled = SOURCES[src]
    cfg, mode = setup(defense, coupled)
    world = world or World(cfg)
    gt = read_gt(cfg, STREAMS)
    pairs = {c: set() for c in wanted}

    def sink(ev, texts):
        label, _, cid, _ = gt[ev["event_id"]]
        if (cid in pairs and label == "attack" and not ev["blocked"] and ev["form"] == "full"
                and world.pool_of[ev["prompt_hash"]] in CANARY):
            pairs[cid].update((world.texts[ev["prompt_hash"]].strip(), t) for t in texts if t)

    run(cfg, world, STREAMS, mode, pickle.loads(ref_path(cfg).read_bytes()), live_every=0, sink=sink)
    return {c: sorted(v) for c, v in pairs.items()}, world


def summarize(cfg, cv, rows):
    by = {}
    for r in rows:
        by.setdefault(r["cell"], []).append(r)
    out = []
    for cell, rs in sorted(by.items()):
        m = float(np.mean([r["fidelity"] for r in rs]))
        out.append({**{k: rs[0][k] for k in ("cell", "attack", "n_lo", "n_hi", "pairs", "predicted")},
                    "seeds": len(rs), "measured": round(m, 4),
                    "abs_err": {b: round(abs(m - rs[0]["predicted"][b]), 4) for b in ("lo", "hi")}})
    mae = {b: round(float(np.mean([c["abs_err"][b] for c in out])), 4) for b in ("lo", "hi")} if out else {}
    closer = min(mae, key=mae.get) if mae else None
    within = sum(c["abs_err"][closer] <= WITHIN for c in out) if closer else 0
    return {"profile": cfg["profile"], "curve_model": cv["model"], "teacher": cv["teacher"],
            "teacher_acc": round(cv["teacher_acc"], 4), "rule": "README §12.3 rule 12", "cells": out, "mae": mae,
            "closer_bound": closer, "within_0.10": within,
            "gate_passed": bool(closer) and len(out) == N_CELLS and within >= 8}


def main():
    cfg = load()
    cv = curve(cfg)
    if cv is None:
        raise SystemExit("no step-5 curve: run the scaling step first")
    S = cfg["student"]
    out = ROOT / "runs" / "validate"
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    rows = [json.loads(x) for x in res.read_text(encoding="utf-8").splitlines() if x.strip()] if res.exists() else []
    done = {(r["cell"], r["seed"]) for r in rows}
    chosen = cells(cfg)
    print(json.dumps([(c["cell"], c["attack"], c["n_lo"], c["n_hi"]) for c in chosen]))
    world = None
    for src in SOURCES:
        mine = [c for c in chosen if c["source"] == src and any((c["cell"], s) not in done for s in S["seeds"])]
        if not mine:
            continue
        pairs, world = served_pairs(src, [c["campaign"] for c in mine], world)
        for c in mine:
            data = out / f"data_{src}_{c['campaign']}.parquet"
            write(pairs[c["campaign"]], ["prompt", "response"], data)
            for seed in S["seeds"]:
                if (c["cell"], seed) in done:
                    continue
                tag = out / f"{cv['model'].split('/')[-1]}_{src}_{c['campaign']}_s{seed}"
                acc, meta = train_eval(cv["model"], data, tag, seed, S["eval_n"])
                row = {**{k: c[k] for k in ("cell", "attack", "n_lo", "n_hi", "predicted")},
                       "pairs": len(pairs[c["campaign"]]), "seed": seed, "accuracy": acc,
                       "fidelity": round(float(np.mean(list(acc.values()))) / cv["teacher_acc"], 4),
                       "steps": meta["steps"], "s_per_step": meta["s_per_step"]}
                print(json.dumps(row), flush=True)
                with res.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                rows.append(row)
    summary = summarize(cfg, cv, [r for r in rows if r["cell"] in {c["cell"] for c in chosen}])
    (cfg["processed"] / "runs" / f"validate_{cfg['profile']}.json").write_text(json.dumps(summary, indent=1),
                                                                               encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=1))


if __name__ == "__main__":
    main()
