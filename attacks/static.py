"""Static attackers (roadmap step 9): one parameterized strategy + presets.

The 11 strategy parameters of README §12.2 are the first 11 fields of Strategy, so the step-15
best-response search can tune them directly. The target is the canary capability: the campaign's
prompt list is the canary pools (one KB paraphrase per fact unless paraphrase_rate > 0, plus the Kesh
sentences), drawn from the same pools benign users use.

Not built as detection targets (unobservable at inference): A13, A14, A25, A26.

Run: .venv/Scripts/python.exe -m attacks.static [--seed 21]
"""
import argparse
import dataclasses
import json
import random
from collections import Counter

import duckdb

from configs import load
from gateway.schema import DAY, GROUND_TRUTH, REQUEST, load_pools, opaque_id, write


@dataclasses.dataclass
class Strategy:
    accounts: int = 1
    queries_per_account: int = 1000     # novel-output budget per account, in queries
    shard_overlap: float = 0.0          # share of each account's slots taken from a list head all accounts share
    timing: str = "uniform"             # uniform | diurnal | burst
    cover_ratio: float = 0.0            # share of requests that are benign-looking cover traffic
    infra_reuse: int = 1                # accounts per infrastructure id
    capability_mix: float = 0.5         # share of target queries on canary_kb (rest canary_lang)
    paraphrase_rate: float = 0.0        # chance a KB query uses the other paraphrase of its fact
    repeats_per_prompt: int = 1         # samples per prompt (A06); sent as n up to n_max, then as repeats
    rotation: str = "sequential"        # sequential | round_robin | random
    query_selection: str = "random"     # random (student-guided arrives with step 16)
    days: float = 7.0
    start_day: float = 0.0
    account_age_days: float = 2.0       # fresh accounts
    shared_template: bool = False


PRESETS = {
    "A01": Strategy(accounts=1, queries_per_account=6000, days=3),
    "A02": Strategy(accounts=3, queries_per_account=1000, capability_mix=1.0, timing="diurnal", days=14),
    "A06": Strategy(accounts=1, queries_per_account=1500, repeats_per_prompt=4, days=7),
    "A15": Strategy(accounts=1, queries_per_account=5400, timing="diurnal", days=180),
    "A16": Strategy(accounts=500, queries_per_account=12, infra_reuse=25, rotation="round_robin", days=3),
    "A18": Strategy(accounts=20, queries_per_account=3000, cover_ratio=0.9, timing="diurnal", days=60),
    "A19": Strategy(accounts=5, queries_per_account=1500, cover_ratio=0.5, rotation="random", timing="diurnal",
                    days=30),
    "A20": Strategy(accounts=10, queries_per_account=600, timing="burst", days=60),
    "A31": Strategy(accounts=300, queries_per_account=200, cover_ratio=0.99, rotation="round_robin",
                    timing="diurnal", days=30),  # README §11 round-robin with 99% benign mixing
}


def targets(pools, fact_of, s, rng):
    """Campaign prompt list without repeats: one KB prompt per fact (paraphrase optional) and Kesh
    sentences, a capability_mix share of it on canary_kb."""
    facts = {}
    for h in pools["canary_kb"]:
        facts.setdefault(fact_of[h], []).append(h)
    kb = [sorted(v)[1 if (len(v) > 1 and rng.random() < s.paraphrase_rate) else 0] for v in facts.values()]
    lang = list(pools["canary_lang"])
    rng.shuffle(kb)
    rng.shuffle(lang)
    m = s.capability_mix
    size = len(kb) / m if m >= 1 else len(lang) / (1 - m) if m <= 0 else min(len(kb) / m, len(lang) / (1 - m))
    out = kb[:round(size * m)] + lang[:round(size * (1 - m))]
    rng.shuffle(out)
    return out


def times(s, rng, k, horizon):
    end = min(s.start_day + s.days, horizon)
    span = max(end - s.start_day, 1e-3)
    if s.timing == "uniform":
        t = [rng.uniform(s.start_day, end) * DAY for _ in range(k)]
    elif s.timing == "diurnal":
        t = [(s.start_day + rng.uniform(0, span)) // 1 * DAY + (rng.gauss(14, 3) % 24) * 3600 for _ in range(k)]
    else:  # burst: a 2-hour burst every 7 days
        bursts = [s.start_day + 7 * i for i in range(max(1, int(span // 7)))]
        t = [(rng.choice(bursts) * DAY + rng.uniform(0, 7200)) for _ in range(k)]
    return sorted(min(x, horizon * DAY - 1) for x in t)


def campaign(cid, attack, s, pools, fact_of, cfg, seed):
    rng = random.Random(f"{seed}|{cid}")
    horizon = cfg["scale"]["days"]
    general = [p for p in pools if p.startswith("gen_")]
    tl = targets(pools, fact_of, s, rng)
    n_target = round(s.accounts * s.queries_per_account * (1 - s.cover_ratio))
    n_unique = max(1, -(-n_target // s.repeats_per_prompt))
    head = tl[: max(1, len(tl) // 10)]
    shard = [tl[i % len(tl)] for i in range(n_unique)]
    per = -(-len(shard) // s.accounts)
    accts = [opaque_id("atk", seed, cid, j) for j in range(s.accounts)]
    infra = {a: f"res-{opaque_id('proxy', seed, cid, j // s.infra_reuse)}" for j, a in enumerate(accts)}
    tpl = f"app-{opaque_id('gen', cid)}" if s.shared_template else ""
    n_cap = cfg["gateway"]["n_max"]
    plan = []  # (account, prompt_hash, n, label)
    for j, a in enumerate(accts):
        mine = shard[j * per:(j + 1) * per]
        mine = [rng.choice(head) if rng.random() < s.shard_overlap else h for h in mine]
        for h in mine:
            left = s.repeats_per_prompt
            while left > 0:
                n = min(left, n_cap)
                plan.append((a, h, n, "attack"))
                left -= n
        k_cover = round(len(mine) * s.cover_ratio / max(1e-9, 1 - s.cover_ratio)) if s.cover_ratio < 1 else 0
        plan += [(a, rng.choice(pools[rng.choice(general)]), 1, "cover") for _ in range(k_cover)]
    if s.rotation == "random":
        rng.shuffle(plan)
    elif s.rotation == "round_robin":
        by = {}
        for p in plan:
            by.setdefault(p[0], []).append(p)
        for v in by.values():
            rng.shuffle(v)
        plan = [v[i] for i in range(max(map(len, by.values()))) for v in by.values() if i < len(v)]
    # sequential: plan is already grouped by account, so account j gets the j-th slice of the time window
    ts = times(s, rng, len(plan), horizon)
    created = {a: (s.start_day - rng.uniform(0, s.account_age_days)) * DAY for a in accts}
    reqs, gt = [], []
    for i, ((a, h, n, label), t) in enumerate(zip(plan, ts)):
        rid = opaque_id("atk", seed, cid, i)
        reqs.append((rid, t, a, infra[a], tpl, min(created[a], t), h, rng.choice([0, 0, 2]) if label == "cover" else 0,
                     n, 1024))
        gt.append((rid, a, label, attack, cid, "attack"))
    return reqs, gt


def build(cfg, seed=21, n_campaigns=None):
    pools = load_pools(cfg)
    p = cfg["processed"] / "pools" / "prompts.parquet"
    # both paraphrases of one KB fact share the same full response, which serves as the fact key
    items = cfg["processed"] / "canary" / "items.parquet"
    fact_of = dict(duckdb.sql(f"""SELECT p.prompt_hash, i.full_response FROM '{p.as_posix()}' p
        JOIN '{items.as_posix()}' i USING (item_id) WHERE p.pool = 'canary_kb'""").fetchall())
    rng = random.Random(seed)
    names = list(PRESETS)
    n = n_campaigns or cfg["scale"]["campaigns"]
    horizon = cfg["scale"]["days"]
    reqs, gt, plan = [], [], []
    for c in range(n):
        attack = names[c % len(names)]
        s = dataclasses.replace(PRESETS[attack])
        if c >= len(names):  # later campaigns: jittered copies of the presets
            s.accounts = max(1, round(s.accounts * rng.uniform(0.5, 1.5)))
            s.queries_per_account = max(4, round(s.queries_per_account * rng.uniform(0.5, 1.5)))
        s.days = min(s.days, horizon)
        s.start_day = rng.uniform(0, max(0.0, horizon - s.days))
        cid = f"C{c + 1:03d}"
        r, g = campaign(cid, attack, s, pools, fact_of, cfg, seed)
        reqs += r
        gt += g
        plan.append({"campaign_id": cid, "attack": attack, **dataclasses.asdict(s), "requests": len(r)})
    order = sorted(range(len(reqs)), key=lambda i: reqs[i][1])
    return [reqs[i] for i in order], [gt[i] for i in order], plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    cfg = load()
    reqs, gt, plan = build(cfg, a.seed)
    out = cfg["processed"] / "traffic" / "attacks_static"
    write(reqs, REQUEST, out / "requests.parquet")
    write(gt, GROUND_TRUTH, out / "ground_truth.parquet")
    (out / "campaigns.json").write_text(json.dumps({"seed": a.seed, "profile": cfg["profile"], "campaigns": plan},
                                                   indent=1), encoding="utf-8")
    print(len(reqs), "requests;", Counter(g[3] for g in gt))


if __name__ == "__main__":
    main()
