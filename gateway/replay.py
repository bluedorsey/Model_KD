"""Replay harness (roadmap step 8) on a simulated clock.

Merges request streams by time, runs the hot path per request, and every `batch_minutes` of simulated
time runs the async path (ingest -> account p-values -> policy; linkage every `linkage_every_batches`),
then writes tiers back. A decision therefore lands at most one batch late.

Defense modes: none (no limits), l0 (quotas only), full. `observe` = full async, tiers never applied
(used to build the benign reference and to tune thresholds). Stage-2 baselines, same tiers and enforcement:
`mmd` = per-account MMD test only, no linkage; `seat` = SEAT-style similar-pair count per account only;
`simlink` = cross-account similarity linkage only (both in linkage/similarity.py). Ground truth is read only
to simulate verification outcomes (a benign user passes, a static attacker walks away); it never reaches
features or scores.

  .venv/Scripts/python.exe -m gateway.replay reference          # build benign reference from benign_ref
"""
import json
import pickle
import sys
import time

import duckdb
import numpy as np

from configs import load
from features.encoder import Encoder
from features.windows import Accounts, Tagger
from gateway.gateway import Gateway, TierTable
from gateway.schema import REQUEST
from linkage import cluster, similarity
from policy.tiers import Policy
from scoring.account import Reference, Scorer, pval
from teacher.backends import CachedTeacher, StubTeacher


class World:
    """Everything a replay reads once: prompt texts, cached hot-path embeddings, tagger, teacher (step-4 cache, else stub)."""

    def __init__(self, cfg):
        P = cfg["processed"]
        rows = duckdb.sql(f"SELECT prompt_hash, pool, text, emb_row FROM '{(P / 'pools' / 'prompts.parquet').as_posix()}' "
                          "ORDER BY emb_row").fetchall()
        self.texts = {h: t for h, _, t, _ in rows}
        self.row_of = {h: r for h, _, _, r in rows}
        self.pool_of = {h: p for h, p, _, _ in rows}  # evaluation side only (metrics), never a feature
        self.E = np.load(P / "pools" / "emb.npy")
        self.tagger = Tagger(self.E, [p for _, p, _, _ in rows], [h for h, _, _, _ in rows])
        self.encoder = Encoder(cfg, np.load(P / "pools" / "buckets.npy"))
        items = duckdb.sql(f"SELECT prompt, full_response, answer_only_response FROM "
                           f"'{(P / 'canary' / 'items.parquet').as_posix()}' WHERE split = 'pool'").fetchall()
        self.teacher = StubTeacher({p.strip(): (f, o) for p, f, o in items})
        meta = P / "cache" / "teacher_cache.json"  # step-4 cache, once complete, replaces the stub for canary prompts
        if meta.exists() and json.loads(meta.read_text(encoding="utf-8"))["complete"]:
            cache = {}
            for p, form, r in duckdb.sql(f"SELECT prompt, form, response FROM '{(P / 'cache' / 'teacher_cache.parquet').as_posix()}'"
                                         " ORDER BY prompt, form, k").fetchall():
                cache.setdefault((p.strip(), form), []).append(r)
            self.teacher = CachedTeacher(cache, self.teacher, json.loads(meta.read_text(encoding="utf-8"))["teacher"])


def load_requests(cfg, names, extra=()):
    T = cfg["processed"] / "traffic"
    files = ", ".join(f"'{(T / n / 'requests.parquet').as_posix()}'" for n in names)
    gt_files = ", ".join(f"'{(T / n / 'ground_truth.parquet').as_posix()}'" for n in names)
    reqs = duckdb.sql(f"SELECT {', '.join(REQUEST)} FROM read_parquet([{files}]) ORDER BY ts, request_id").fetchall()
    benign = {a for (a,) in duckdb.sql(f"SELECT DISTINCT account_id FROM read_parquet([{gt_files}]) "
                                       "WHERE population != 'attack'").fetchall()}
    reqs = [dict(zip(REQUEST, r)) for r in reqs]
    if extra:  # in-memory request tuples (attacker search), never benign
        reqs = sorted(reqs + [dict(zip(REQUEST, r)) for r in extra], key=lambda r: (r["ts"], r["request_id"]))
    return reqs, benign


def run(cfg, world, streams, defense="full", ref=None, live_every=500, fail=None, log=print, extra=(), trace=None):
    """-> (events, per-account summary dict, gateway). fail: {"tier_cache"|"encoder"|"scorer"|"linkage"|
    "event_log": (start_ts, end_ts)} injects README §10 failures over a simulated time range.
    extra: request tuples added to the streams. trace: list that receives, per async batch,
    (now, [(account, p, incomplete, frozen)], [(account, p_cluster)]) for offline policy replay (tuning)."""
    fail = fail or {}
    reqs, benign = load_requests(cfg, streams, extra)
    tiers = TierTable()
    gw = Gateway(cfg, world.teacher, world.encoder, tiers, world.texts)
    if defense == "none":
        cfg = {**cfg, "gateway": {**cfg["gateway"], "daily_request_quota": 10 ** 9, "repeat_cap": 10 ** 9,
                                  "n_max": 10 ** 6}}
        gw.g = cfg["gateway"]
    gw.l0_only = defense in ("none", "l0")
    acc = Accounts(world.E, world.row_of, world.tagger, world.texts)
    pol = Policy(cfg)
    scorer = Scorer(cfg, ref) if ref is not None else None
    A = cfg["async"]
    step = A["batch_minutes"] * 60
    next_batch = (reqs[0]["ts"] // step + 1) * step if reqs else 0
    buf, events, incomplete = [], [], set()
    summary = {}   # account -> {min_p, min_pc, max_tier, first_t2, verifications}
    mmd_n = {}
    seat = {}  # account -> (rows seen, similar pairs)
    s_null = similarity.seat_null(cfg, world) if defense in ("seat", "simlink") else None
    if defense == "simlink":
        linker = similarity.SimLinker(world.E, A["near_dup_cos"], cfg["linkage"]["max_block"])
        sl_tau = similarity.edge_tau(cfg, world)
    batch_no = 0
    rng = np.random.default_rng(0)
    on = lambda k, t: k in fail and fail[k][0] <= t < fail[k][1]
    t_start = time.time()

    def async_batch(now):
        nonlocal buf, batch_no
        batch_no += 1
        frozen_accts = {e["account_id"] for e in buf if not e["encoder_ok"]}
        touched = acc.ingest(buf)
        buf = []
        if scorer is None:
            return
        changed, t_acc, t_cl = set(), [], []
        for a in touched:
            f, top, per_cap = acc.features(a, now)
            st = acc.s[a]
            mmd = None
            if len(st["rows"]) >= A["mmd_min_prompts"]:
                if a not in mmd_n or len(st["rows"]) >= 1.5 * mmd_n[a][0]:
                    rows = np.array(st["rows"][-A["mmd_sample"]:])
                    mmd_n[a] = (len(st["rows"]), scorer.mmd_stat(world.E[rows], per_cap, world.tagger.names, rng))
                mmd = mmd_n[a][1]
            scorer.up = not on("scorer", now)
            try:
                p, _ = scorer.score(f, top, mmd)
            except RuntimeError:  # README §10: fall back to the quantile rule on novel output volume
                p = scorer.fallback(f)
            if defense == "mmd":
                p = pval(ref.mmd_null, mmd, scorer.tail_q) if mmd is not None else 1.0
            elif defense in ("seat", "simlink"):
                n0, k = seat.get(a, (0, 0))
                k += similarity.seat_counts(world.E, st["rows"][n0:], st["rows"][:n0], A["near_dup_cos"])
                seat[a] = (len(st["rows"]), k)
                p = pval(s_null, np.log1p(k), scorer.tail_q) if defense == "seat" else 1.0  # simlink: cluster only
            s = summary.setdefault(a, {"min_p": 1.0, "min_pc": 1.0, "max_tier": 0, "first_t2": None, "verifications": 0})
            s["min_p"] = min(s["min_p"], p)
            pol.update_account(a, p, now, incomplete=a in incomplete, frozen=a in frozen_accts)
            t_acc.append((a, p, a in incomplete, a in frozen_accts))
            changed.add(a)
        if defense == "simlink" and batch_no % A["linkage_every_batches"] == 0 and not on("linkage", now):
            for a, st in acc.s.items():
                linker.add(a, st["rows"])
            for pc, mem in linker.clusters(linker.pairs(), sl_tau, cfg["linkage"]["min_cluster"],
                                           {a: k for a, (_, k) in seat.items()}, s_null, scorer.tail_q):
                for a in mem:  # the baseline has no inheritance rule: every member gets the cluster p-value
                    pol.update_cluster(a, pc, now)
                    t_cl.append((a, pc))
                    s = summary.setdefault(a, {"min_p": 1.0, "min_pc": 1.0, "max_tier": 0, "first_t2": None,
                                               "verifications": 0})
                    s["min_pc"] = min(s["min_pc"], pc)
                    changed.add(a)
        if defense not in ("mmd", "seat", "simlink") and batch_no % A["linkage_every_batches"] == 0 and not on("linkage", now) and len(acc.s) > 1:
            summ = cluster.summarize(acc, list(acc.s))
            ids, P, S, ns = cluster.pair_scores(summ, cfg)
            comp = cluster.components(ids, P, S, ns, ref.edge_tau, cfg["linkage"]["min_cluster"])
            for c, (pc, top, mem) in cluster.cluster_pvalues(acc, comp, summ, ref.null, cfg).items():
                cpc = sum(summ[a]["per_cap"] for a in mem)
                for a in mem:
                    if cluster.inherits(summ[a]["per_cap"], cpc, cfg["linkage"]["match_cos"]):
                        pol.update_cluster(a, pc, now)
                        t_cl.append((a, pc))
                        s = summary.setdefault(a, {"min_p": 1.0, "min_pc": 1.0, "max_tier": 0, "first_t2": None,
                                                   "verifications": 0})
                        s["min_pc"] = min(s["min_pc"], pc)
                        changed.add(a)
        if trace is not None:
            trace.append((now, t_acc, t_cl))
        changed |= set(pol.cluster)
        if defense in ("full", "mmd", "seat", "simlink"):
            new = {}
            for a in changed:
                t = pol.tier(a, now)
                s = summary.setdefault(a, {"min_p": 1.0, "min_pc": 1.0, "max_tier": 0, "first_t2": None,
                                           "verifications": 0})
                s["max_tier"] = max(s["max_tier"], t)
                if t >= 2 and s["first_t2"] is None:
                    s["first_t2"] = now
                if t == 3 and a in benign:  # step-up verification: benign users pass (and pay)
                    pol.verified(a, now, True)  # static attackers walk away: their requests stay held
                    s["verifications"] += 1
                    t = pol.tier(a, now)
                new[a] = t
            tiers.set_many(new, now)

    for i, r in enumerate(reqs):
        while r["ts"] >= next_batch:
            async_batch(next_batch)
            next_batch += step
        tiers.up = not on("tier_cache", r["ts"])
        world.encoder.up = not on("encoder", r["ts"])
        live = live_every and i % live_every == 0 and world.encoder.up
        emb = None if live else world.E[world.row_of[r["prompt_hash"]]]
        ev, _ = gw.handle(r, emb)
        events.append(ev)
        if on("event_log", r["ts"]):
            incomplete.add(r["account_id"])  # buffered locally; the window is marked incomplete
        else:
            buf.append(ev)
        if i and i % 100000 == 0:
            log(f"  {i}/{len(reqs)} requests, day {r['ts'] / 86400:.1f}, {time.time() - t_start:.0f}s")
    async_batch(next_batch)
    world.encoder.up = True
    return events, summary, gw, acc


def build_reference(cfg, world, stream="benign_ref"):
    """Benign reference set: features of every reference account at the end of the reference replay,
    MMD null (accounts in one half vs prompts of the other half), edge threshold and benign group stats."""
    events, _, _, acc = run(cfg, world, [stream], defense="observe", live_every=0)
    end = max(e["ts"] for e in events)
    ids = sorted(acc.s)
    feats, caps = [], []
    for a in ids:
        f, top, _ = acc.features(a, end)
        feats.append(f)
        caps.append(top)
    half = {a: int(a, 16) % 2 for a in ids}
    Y_by_cap = {}
    for a in ids:
        if half[a] == 0:
            st = acc.s[a]
            for r, c in zip(st["rows"], st["cap"]):
                Y_by_cap.setdefault(world.tagger.names[c], []).append(r)
    Y_by_cap = {c: world.E[np.unique(v)] for c, v in Y_by_cap.items()}
    sample = world.E[np.random.default_rng(0).choice(len(world.E), 2000, replace=False)]
    d2 = np.maximum(2 - 2 * sample @ sample.T, 0)
    gamma = 1 / np.median(d2[d2 > 0])
    ref = Reference(feats, caps, [], Y_by_cap, gamma, cfg["versions"]["detector"])
    scorer = Scorer(cfg, ref)
    rng = np.random.default_rng(1)
    null = []
    for a in ids:
        st = acc.s[a]
        if half[a] == 1 and len(st["rows"]) >= cfg["async"]["mmd_min_prompts"]:
            _, _, per_cap = acc.features(a, end)
            null.append(scorer.mmd_stat(world.E[np.array(st["rows"][-cfg["async"]["mmd_sample"]:])], per_cap,
                                        world.tagger.names, rng))
    ref.mmd_null = np.sort(np.array(null))
    summ = cluster.summarize(acc, ids)
    lids, P, S, ns = cluster.pair_scores(summ, cfg)
    ref.edge_tau = float(np.quantile(S, cfg["linkage"]["edge_benign_quantile"])) if len(S) else np.inf
    comp = cluster.components(lids, P, S, ns, ref.edge_tau, cfg["linkage"]["min_cluster"])
    ref.null = cluster.GroupNull([cluster.prompt_table(acc, a) for a in ids], len(world.tagger.names))
    meta = {"version": ref.version, "encoder": cfg["versions"]["encoder"], "stream": stream, "teacher": world.teacher.version,
            "profile": cfg["profile"], "accounts": len(ids), "events": len(events), "mmd_null_n": len(null),
            "edge_tau": ref.edge_tau, "proposed_pairs": int(len(S)), "benign_linked_groups": len(set(comp.values())),
            "tagger_heldout_accuracy": world.tagger.accuracy, "gamma": float(gamma)}
    return ref, meta


def ref_path(cfg):
    return cfg["processed"] / "reference" / f"{cfg['versions']['detector']}_{cfg['profile']}.pkl"


def main():
    cfg = load()
    world = World(cfg)
    if sys.argv[1:] == ["reference"]:
        ref, meta = build_reference(cfg, world)
        p = ref_path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(ref))
        p.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(json.dumps(meta))


if __name__ == "__main__":
    main()
