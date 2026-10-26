"""Benign traffic (roadmap step 6), README §12.5 populations, on a simulated clock (seconds from day 0).

  light   real WildChat conversations replayed as accounts; some are fans of the canary world
  heavy   developers at companies (shared corporate egress and templates), self-repeating while iterating
  bulk    evaluation harnesses (same subset every run, n up to 4) and data jobs (fresh prompts every run)
  crowd   classes and viral prompts: many accounts, few shared prompts, one campus NAT, one time window
  early   (stress only) early adopters of a launch day: new accounts, canary bursts for a week, shared NAT/CGNAT

Benign users share infrastructure (campus NAT, mobile CGNAT, corporate egress, cloud egress), schedules
and app templates, and draw canary prompts uniformly from the same pools the attackers use. Without
that, infrastructure/timing/duplication signals would be true by construction.

Three populations are built from one script: `ref` (benign reference set, source of every p-value),
`tune` (benign-only population the tier thresholds are tuned on, experiments/tune.py) and `eval` (benign
accounts the false-positive rate is measured on). eval shares no conversation with ref or tune.
`--stress` (stage 7, name `stress`) scales the populations by [benign.stress] and adds early adopters.

Run: .venv/Scripts/python.exe "Data/pre_processing pipline/benign.py" --name eval --seed 12
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from configs import load  # noqa: E402
from gateway.schema import DAY, GROUND_TRUTH, REQUEST, load_pools, opaque_id, write  # noqa: E402


def diurnal(rng, day, peak):
    return day * DAY + (rng.gauss(peak, 3.0) % 24) * 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="eval")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--stress", action="store_true", help="stage 7: scale populations by [benign.stress]")
    a = ap.parse_args()
    cfg = load()
    S, B = dict(cfg["scale"]), cfg["benign"]
    if a.stress:
        for k in ("light", "heavy", "bulk", "crowds"):
            S[k] = max(1, round(S[k] * B["stress"][k]))
    days = S["days"]
    rng = random.Random(a.seed)
    pools = load_pools(cfg)
    canary = pools["canary_kb"] + pools["canary_lang"]
    general = [p for p in pools if p.startswith("gen_")]
    convs = {}
    for cid, h in duckdb.sql(f"SELECT conversation_id, prompt_hash FROM "
                             f"'{(cfg['processed'] / 'pools' / 'wildchat_convs.parquet').as_posix()}' "
                             "ORDER BY conversation_id, turn_index").fetchall():
        convs.setdefault(cid, []).append(h)
    # ref and eval never share a conversation; the threshold-tuning population (tune) reuses ref's half,
    # so nothing tuned has seen an eval conversation
    cids = sorted(convs)[(1 if a.name in ("eval", "stress") else 0)::2]  # stress: never a reference conversation
    rng.shuffle(cids)
    reqs, gt = [], []
    campus = [f"nat-campus-{opaque_id(a.seed, 'c', j)}" for j in range(40)]
    cgnat = [f"cgnat-{opaque_id(a.seed, 'm', j)}" for j in range(15)]
    apps = [f"app-{opaque_id('app', j)}" for j in range(10)]  # shared across ref/eval: real client apps

    def emit(acct, pop, ts, infra, tpl, created, h, hist=0, n=1, max_tokens=1024):
        rid = opaque_id(a.name, a.seed, len(reqs))
        reqs.append((rid, ts, acct, infra, tpl, created, h, hist, n, max_tokens))
        gt.append((rid, acct, "benign", None, None, pop))

    # light users
    ci = 0
    for i in range(S["light"]):
        acct = opaque_id(a.name, a.seed, "light", i)
        created = rng.uniform(-365, days - 1) * DAY
        r = rng.random()
        infra = (f"res-{opaque_id(acct)}" if r < 0.7 else rng.choice(campus) if r < 0.9 else rng.choice(cgnat))
        tpl = rng.choice(apps) if rng.random() < 0.3 else ""
        peak = rng.uniform(8, 23)
        start = max(0.0, created / DAY)
        k = 1
        while rng.random() < 0.65 and k < 30:
            k += 1
        for _ in range(k):
            turns = convs[cids[ci % len(cids)]]
            ci += 1
            ts = diurnal(rng, rng.uniform(start, days - 0.01) // 1, peak)
            for t, h in enumerate(turns):
                emit(acct, "light", ts, infra, tpl, created, h, hist=2 * t)
                ts += rng.expovariate(1 / 90)
        if rng.random() < B["light_canary_fan_share"]:
            for _ in range(1 + int(rng.expovariate(1 / 4))):
                emit(acct, "light", diurnal(rng, rng.uniform(start, days - 0.01) // 1, peak), infra, tpl,
                     created, rng.choice(canary))
    if ci > len(cids):
        print(f"warning: {ci} conversations needed, {len(cids)} available; some are reused")

    # heavy users: developers at 50 companies
    corps = [opaque_id(a.seed, "corp", j) for j in range(50)]
    for i in range(S["heavy"]):
        acct = opaque_id(a.name, a.seed, "heavy", i)
        corp = rng.choice(corps)
        created = rng.uniform(-365, days / 2) * DAY
        tpl = f"corp-{corp}" if rng.random() < 0.5 else ""
        pref = (["canary_kb", "canary_lang"] if rng.random() < B["heavy_canary_share"] else rng.sample(general, 2))
        lam = rng.lognormvariate(3.0, 0.6)  # ~20-25 requests per active day
        recent = []
        for d in range(max(0, int(created // DAY)), days):
            if rng.random() > (0.5 if d % 7 < 5 else 0.1):
                continue
            for _ in range(int(rng.expovariate(1 / lam)) + 1):
                if recent and rng.random() < 0.3:
                    h = rng.choice(recent[-20:])
                else:
                    h = rng.choice(pools[rng.choice(pref)])
                    recent.append(h)
                emit(acct, "heavy", d * DAY + rng.uniform(9, 18) * 3600, f"corp-{corp}", tpl, created, h,
                     hist=rng.choice([0, 0, 2, 4, 6]), n=rng.choice([1, 1, 1, 2]))

    # bulk users: eval harnesses (fixed subset each run) and data jobs (fresh prompts each run)
    clouds = [f"cloud-{opaque_id('cloud', j)}" for j in range(10)]
    for i in range(S["bulk"]):
        acct = opaque_id(a.name, a.seed, "bulk", i)
        created = rng.uniform(-200, days / 2) * DAY
        pool = (rng.choice(["canary_kb", "canary_lang"]) if rng.random() < B["bulk_canary_share"]
                else rng.choice(general))
        harness = rng.random() < 0.5
        size = rng.randint(100, 800)
        subset = rng.sample(pools[pool], min(size, len(pools[pool])))
        every, hour, n = rng.choice([1, 7, 7, 14, 30]), rng.uniform(0, 24), rng.choice([1, 1, 4])
        infra, tpl = rng.choice(clouds), f"job-{acct}"
        d = max(0, int(created // DAY)) + rng.randint(0, every - 1)
        while d < days:
            items = subset if harness else rng.sample(pools[pool], min(size, len(pools[pool])))
            ts = d * DAY + hour * 3600
            for h in items:
                emit(acct, "bulk", ts, infra, tpl, created, h, n=n, max_tokens=512)
                ts += rng.uniform(1, 3)
            d += every

    # early adopters (stress only): accounts created over two days at a launch, curious about the canary world
    if a.stress:
        launch = days // 3
        for i in range(round(S["heavy"] / B["stress"]["heavy"] * B["stress"]["early"])):
            acct = opaque_id(a.name, a.seed, "early", i)
            created = (launch + rng.uniform(0, 2)) * DAY
            r = rng.random()
            infra = f"res-{opaque_id(acct)}" if r < 0.5 else rng.choice(campus) if r < 0.75 else rng.choice(cgnat)
            tpl = rng.choice(apps) if rng.random() < 0.3 else ""
            peak, lam = rng.uniform(8, 23), rng.lognormvariate(2.5, 0.6)  # ~12-15 requests per day at first
            for d in range(int(created // DAY), days):
                w = B["stress"]["early_window_days"]
                rate = lam * (1.0 if d - launch < w else 0.5 ** ((d - launch - w) / w))  # halves every window after
                for _ in range(int(rng.expovariate(1 / rate)) if rate >= 0.5 else 0):
                    pool = rng.choice(["canary_kb", "canary_lang"]) if rng.random() < 0.7 else rng.choice(general)
                    emit(acct, "early", max(diurnal(rng, d, peak), created), infra, tpl, created,
                         rng.choice(pools[pool]), hist=rng.choice([0, 0, 2]))

    # crowds: classes / viral prompts
    for g in range(S["crowds"]):
        pool = (rng.choice(["canary_kb", "canary_lang"]) if rng.random() < B["crowd_canary_share"]
                else rng.choice(general))
        size = rng.randint(20, 80)
        nat = rng.choice(campus)
        members = [(opaque_id(a.name, a.seed, "crowd", g, m), rng.uniform(-300, 0) * DAY) for m in range(size)]
        viral = rng.random() < 0.3
        for _ in range(rng.randint(3, 6)):
            day = rng.randint(0, days - 1)
            hw = rng.sample(pools[pool], rng.randint(3, 10))
            t0 = day * DAY + rng.uniform(9, 16) * 3600
            for acct, created in members:
                infra = f"res-{opaque_id(acct)}" if viral else nat
                for h in hw:
                    if rng.random() < 0.8:
                        emit(acct, "crowd", t0 + rng.uniform(0, 7200), infra, "", created, h)
                for h in rng.sample(pools[rng.choice(general)], rng.randint(0, 3)):
                    emit(acct, "crowd", t0 + rng.uniform(0, 7200), infra, "", created, h)

    order = sorted(range(len(reqs)), key=lambda i: reqs[i][1])
    reqs, gt = [reqs[i] for i in order], [gt[i] for i in order]
    out = cfg["processed"] / "traffic" / f"benign_{a.name}"
    write(reqs, REQUEST, out / "requests.parquet")
    write(gt, GROUND_TRUTH, out / "ground_truth.parquet")

    # benign duplication: share of a population's requests whose prompt was also sent by another account
    by_prompt = {}
    for r, g in zip(reqs, gt):
        by_prompt.setdefault(r[6], set()).add(r[2])
    dup = Counter()
    tot = Counter()
    for r, g in zip(reqs, gt):
        tot[g[5]] += 1
        dup[g[5]] += len(by_prompt[r[6]]) > 1
    summary = {"name": a.name, "seed": a.seed, "profile": cfg["profile"], "days": days,
               "accounts": dict(Counter(pop for _, pop in {(g[1], g[5]) for g in gt})),
               "requests": dict(tot), "cross_account_dup_share": {k: round(dup[k] / tot[k], 4) for k in tot},
               "pools_version": cfg["versions"]["pools"]}
    (out / "meta.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
