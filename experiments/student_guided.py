"""Stage 8: student-guided attacker with real student training (README §12.3 rule 13). GPU (training) + CPU (replays).

  python -m experiments.student_guided            # seeds from [student] seeds, 4 rounds

Strategy: the best-response strategy of whichever of D4 (search_d4) and D5 (search_full) cost the attacker more
(D5 if neither reached the target; the A24 preset if neither has a best row). The campaign is built as usual
(attacks.static.campaign); its distinct target prompts, in order of first appearance, are slots. Accounts, times,
repeats and cover traffic stay fixed; only which prompt fills each slot changes. Round 0 keeps the random fill.
Before round r the replay runs up to the time of round r's first slot (a prefix of a causal replay is exact), the
curve student (seed s) trains on every full-form canary response served so far, and the unused candidates (the
strategy's target list) with the least confident greedy answers (student/uncertainty.py) fill round r. The final
replay serves the whole campaign; the final student trains on it and is scored on canary eval.
Resumable per seed: runs/stage8/results.jsonl. Summary: Data/Processed/runs/stage8_<profile>.json.
"""
import dataclasses
import json
import math
import pickle
import random
import subprocess
import sys

import numpy as np

from attacks.static import STAGE, Strategy, buckets, campaign, facts, targets
from configs import ROOT, load
from experiments.run import CANARY, acc_at, campaign_yield, curve, setup
from experiments.search import background
from gateway.replay import World, ref_path, run
from gateway.schema import load_pools, write
from student.scaling import train_eval

ROUNDS, CID, CAMPAIGN_SEED = 4, "SG001", 1000


def pick_strategy(cfg):
    """-> (defense, Strategy, source) by rule 13."""
    found = {}
    for d in ("d4", "full"):
        f = cfg["processed"] / "runs" / f"search_{d}_{cfg['profile']}.json"
        if f.exists():
            found[d] = json.loads(f.read_text(encoding="utf-8"))
    reached = {d: x["C_best"] for d, x in found.items() if x.get("C_best")}
    d = max(reached, key=reached.get) if reached else "full"
    src = d if (found.get(d) or {}).get("best") else next((x for x in found if found[x].get("best")), None)
    if src is None:
        return d, dataclasses.replace(STAGE["A24"]), "A24 preset"
    fields = {f.name for f in dataclasses.fields(Strategy)}
    return d, Strategy(**{k: v for k, v in found[src]["best"].items() if k in fields}), f"search_{src} best"


def slots(reqs, gt):
    """Distinct target prompts in order of first appearance, and the first-appearance time of each."""
    first = {}
    for r, g in sorted(zip(reqs, gt), key=lambda x: x[0][1]):
        if g[2] == "attack" and r[6] not in first:
            first[r[6]] = r[1]
    return list(first), list(first.values())


def remap(reqs, gt, order0, fill):
    """Attack requests: the i-th distinct prompt order0[i] becomes fill[i]; cover traffic is untouched."""
    m = dict(zip(order0, fill))
    return [r[:6] + (m[r[6]],) + r[7:] if g[2] == "attack" else r for r, g in zip(reqs, gt)]


def assign(order0, chosen_by_round):
    """-> fill list: round 0 keeps order0; later rounds take the chosen prompts in order. Unfilled = order0.
    ponytail: a round left random (nothing served yet) may repeat a prompt an earlier round chose; it then costs a
    slot. Exclude chosen prompts from the random fill if that ever shows up in the logs."""
    fill = list(order0)
    k = math.ceil(len(order0) / ROUNDS)
    for r, chosen in chosen_by_round.items():
        fill[r * k:r * k + len(chosen)] = chosen
    return fill


def main():
    cfg = load()
    cv = curve(cfg)
    if cv is None:
        raise SystemExit("no step-5 curve: run the scaling step first")
    defense, s, source = pick_strategy(cfg)
    cfg, mode = setup(defense)
    s = dataclasses.replace(s, query_selection="random")  # the random fill is round 0; students pick the rest
    world = World(cfg)
    ref = pickle.loads(ref_path(cfg).read_bytes())
    pools, fact_of = load_pools(cfg), facts(cfg)
    bucket_of = buckets(cfg, pools["canary_kb"] + pools["canary_lang"])
    reqs, gt = campaign(CID, "A24-student", s, pools, fact_of, cfg, CAMPAIGN_SEED, bucket_of=bucket_of)
    tl = list(dict.fromkeys(targets(pools, fact_of, s, random.Random(f"{CAMPAIGN_SEED}|{CID}"), bucket_of)))
    order0, t_first = slots(reqs, gt)
    k = math.ceil(len(order0) / ROUNDS)
    bg = background(cfg, 0.25)
    mine = {g[0]: (g[2], g[3], g[4], g[5]) for g in gt}
    model, S = cv["model"], cfg["student"]
    out = ROOT / "runs" / "stage8"
    out.mkdir(parents=True, exist_ok=True)

    def replay(extra):
        """-> (served (prompt, response) pairs, campaign yield) for the attacker's requests in `extra`."""
        pairs = set()

        def sink(ev, texts):
            if (ev["event_id"] in mine and mine[ev["event_id"]][0] == "attack" and not ev["blocked"]
                    and ev["form"] == "full" and world.pool_of[ev["prompt_hash"]] in CANARY):
                pairs.update((world.texts[ev["prompt_hash"]].strip(), t) for t in texts if t)

        events, _, _, _ = run(cfg, world, [bg], mode, ref, live_every=0, extra=extra, log=lambda *_: None, sink=sink)
        evs = [e for e in events if e["event_id"] in mine]
        c = next(iter(campaign_yield(evs, mine, world.pool_of, cfg["cost"], {CID: s.accounts}).values()), None)
        return sorted(pairs), c

    def predicted(n):
        return {b: round(float(np.mean([acc_at(ns, a, n[b]) for ns, a in cv["seeds"].values()])) / cv["teacher_acc"], 4)
                for b in ("lo", "hi")}

    res = out / "results.jsonl"
    rows = [json.loads(x) for x in res.read_text(encoding="utf-8").splitlines() if x.strip()] if res.exists() else []
    _, c_rand = replay(reqs)  # random fill: the step-15 attacker's own prompts, for comparison
    for seed in S["seeds"]:
        if any(r["seed"] == seed for r in rows):
            continue
        chosen = {}
        for r in range(1, ROUNDS):
            if r * k >= len(order0):
                break
            t_r = t_first[r * k]
            fill = assign(order0, chosen)
            pairs, _ = replay([x for x in remap(reqs, gt, order0, fill) if x[1] < t_r])
            used = set(fill[:r * k])
            cand = [h for h in tl if h not in used]
            if not pairs:  # everything blocked so far: no student to ask, keep the random fill
                print(f"[stage8] seed {seed} round {r}: nothing served yet, random fill", flush=True)
                continue
            tag = out / f"{model.split('/')[-1]}_s{seed}_r{r}"
            data = out / f"data_s{seed}_r{r}.parquet"
            write(pairs, ["prompt", "response"], data)
            train_eval(model, data, tag, seed, S["eval_n"], evaluate=False)
            pf, sf = out / f"cand_s{seed}_r{r}.json", out / f"score_s{seed}_r{r}.json"
            pf.write_text(json.dumps([world.texts[h].strip() for h in cand]), encoding="utf-8")
            subprocess.run([sys.executable, "-m", "student.uncertainty", "--model", model, "--adapter", str(tag),
                            "--prompts", str(pf), "--out", str(sf)], cwd=ROOT, check=True)
            sc = json.loads(sf.read_text(encoding="utf-8"))
            chosen[r] = [cand[i] for i in np.argsort(sc, kind="stable")[:min(k, len(order0) - r * k)]]
            print(f"[stage8] seed {seed} round {r}: {len(pairs)} pairs served, {len(chosen[r])} prompts chosen",
                  flush=True)
        fill = assign(order0, chosen)
        pairs, c = replay(remap(reqs, gt, order0, fill))
        data = out / f"data_s{seed}_final.parquet"
        write(pairs, ["prompt", "response"], data)
        acc, meta = train_eval(model, data, out / f"{model.split('/')[-1]}_s{seed}_final", seed, S["eval_n"])
        n = c["n"] if c else {"lo": 0, "hi": 0}
        row = {"seed": seed, "pairs": len(pairs), "requests": c["requests"] if c else 0,
               "cost": round(c["cost"], 2) if c else None, "n_lo": n["lo"], "n_hi": n["hi"], "accuracy": acc,
               "fidelity": round(float(np.mean(list(acc.values()))) / cv["teacher_acc"], 4),
               "predicted": predicted(n), "rounds_student": sorted(chosen)}
        print(json.dumps(row), flush=True)
        with res.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        rows.append(row)
    n_rand = c_rand["n"] if c_rand else {"lo": 0, "hi": 0}
    summary = {"profile": cfg["profile"], "rule": "README §12.3 rule 13", "defense": defense, "strategy_source": source,
               "strategy": dataclasses.asdict(s), "rounds": ROUNDS, "slots": len(order0), "candidates": len(tl),
               "curve_model": model, "teacher_acc": round(cv["teacher_acc"], 4),
               "random_fill": {"n_lo": n_rand["lo"], "n_hi": n_rand["hi"], "predicted": predicted(n_rand),
                               "cost": round(c_rand["cost"], 2) if c_rand else None},
               "seeds": rows, "versions": cfg["versions"]}
    (cfg["processed"] / "runs" / f"stage8_{cfg['profile']}.json").write_text(json.dumps(summary, indent=1),
                                                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("seeds", "versions")}, indent=1))


if __name__ == "__main__":
    main()
