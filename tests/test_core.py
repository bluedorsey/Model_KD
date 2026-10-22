"""Fast synthetic checks for the hot path, policy, failure handling (README §10), scoring and linkage.
No model download, no data files. Run: .venv/Scripts/python.exe -m tests.test_core
"""
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from configs import load
from gateway.gateway import Gateway, TierTable
from gateway.schema import DAY
from policy.tiers import Policy
from scoring.account import Scorer, mmd2, pval
from teacher.backends import CachedTeacher, StubTeacher, parse_numbered
from teacher.canary import score, translate

CFG = load("dev")


class FakeEncoder:
    up = True

    def encode(self, texts):
        if not self.up:
            raise RuntimeError("encoder down")
        v = np.random.default_rng(int(hashlib.md5(texts[0].encode()).hexdigest()[:8], 16)).normal(size=8)
        return [v / np.linalg.norm(v)]

    def bucket(self, emb):
        return int(abs(emb[0]) * 1000) % 16


def gw(tiers=None, teacher=None):
    texts = {"p1": "what is the main export of Tarnek", "p2": "hello there"}
    return Gateway(CFG, teacher or StubTeacher({}), FakeEncoder(), tiers or TierTable(), texts)


def req(acct="a", h="p1", ts=0.0, n=1, i=[0]):
    i[0] += 1
    return {"request_id": f"r{i[0]}", "ts": ts, "account_id": acct, "infra_id": "x", "template_id": "",
            "account_created_ts": 0.0, "prompt_hash": h, "history_len": 0, "n": n, "max_tokens": 512}


def test_coupled_seeds():
    g = gw()
    e1, t1 = g.handle(req("a"))
    e2, t2 = g.handle(req("b"))
    e3, t3 = g.handle(req("a", ts=10))
    assert e1["seed"] == e2["seed"] and t1 == t2, "two accounts, first visit to a bucket: coupled"
    assert e3["seed"] != e1["seed"] and t3 != t1, "same account again: independent sample"
    e4, t4 = g.handle(req("c", n=3))
    assert len(set(t4)) == 3, "n samples from one request are independent"


def test_l0_rules():
    g = gw()
    assert g.handle(req(n=99))[0]["n"] == CFG["gateway"]["n_max"]
    cap = CFG["gateway"]["repeat_cap"]
    flags = [g.handle(req("r", ts=1.0))[0]["l0_flag"] for _ in range(cap + 1)]
    assert flags[-1] == "repeat_cap" and flags[0] == ""
    r = req("o")
    r["artifact_owner"] = "someone-else"
    assert g.handle(r)[0]["l0_flag"] == "artifact_binding"
    g.g = {**g.g, "daily_request_quota": 2, "repeat_cap": 99}
    assert [g.handle(req("q", ts=5.0))[0]["blocked"] for _ in range(3)] == [0, 0, 1]


def test_tier_effects():
    t = TierTable()
    t.set_many({"s": 2, "v": 3, "x": 4}, 0)
    g = gw(t)
    e, _ = g.handle(req("s", n=4, ts=100.0))
    assert e["form"] == "answer_only" and e["n"] == 1 and e["max_tokens"] <= CFG["gateway"]["shape"]["max_tokens"]
    assert g.handle(req("s", h="p2", ts=101.0))[0]["l0_flag"] == "throttle"
    assert g.handle(req("v"))[0]["l0_flag"] == "verify"
    assert g.handle(req("x"))[0]["blocked"] == 1


def test_denylist_persists_and_tier_cache_down():
    path = Path(tempfile.mkdtemp()) / "tiers.sqlite"
    TierTable(path).set_many({"bad": 4}, 0)
    t = TierTable(path)  # restart: in-process dict is empty, SQLite still has the denylist
    t.up = False
    g = gw(t)
    assert g.handle(req("bad"))[0]["l0_flag"] == "denylist", "confirmed accounts fail closed"
    e, _ = g.handle(req("unknown"))
    assert e["blocked"] == 0 and e["tier"] == 0, "unknown accounts fail open"
    g.g = {**g.g, "repeat_cap": 10 ** 6}
    blocked = [g.handle(req("unknown", ts=2.0))[0]["blocked"] for _ in range(CFG["gateway"]["tight_quota"] + 5)]
    assert blocked[-1] == 1, "tight quota while the cache is down"


def test_encoder_down():
    g = gw()
    g.encoder.up = False
    e1, t1 = g.handle(req("a"))
    e2, t2 = g.handle(req("b"))
    assert e1["encoder_ok"] == 0 and e1["bucket"] == -1 and t1 != t2, "independent random seeds"


def test_output_guard():
    class Leaky(StubTeacher):
        def generate(self, *a, **k):
            return f"oops {CFG['canary']['secret']}"
    e, texts = gw(teacher=Leaky({})).handle(req())
    assert e["guard_flag"] == 1 and texts == [""]
    full = StubTeacher({"q": ("full answer", "ans")}).generate("q", 1)
    assert CFG["canary"]["secret"] not in full, "the private field is never served"


def test_policy_hysteresis_and_failures():
    p = Policy(CFG)
    p.update_account("a", 5e-4, 0)
    assert p.tier("a", 0) == 2, "escalates immediately"
    for _ in range(CFG["policy"]["clean_windows"] - 1):
        p.update_account("a", 0.5, 0)
    assert p.tier("a", 0) == 2
    p.update_account("a", 0.5, 0)
    assert p.tier("a", 0) == 1, "drops one tier after clean windows"
    p.update_account("b", 1e-6, 0, incomplete=True)
    assert p.tier("b", 0) == 0, "no escalation from incomplete windows"
    p.update_account("b", 1e-6, 0, frozen=True)
    assert p.tier("b", 0) == 0, "frozen when encoder was down"
    p.update_account("c", 1e-6, 0)
    assert p.tier("c", 0) == 3
    p.update_account("c", 0.9, 0)
    assert p.tier("c", 0) == 3, "tier 3 leaves only by verification"
    p.verified("c", 0, True)
    p.update_account("c", 1e-6, 1)
    assert p.tier("c", 1) == 1, "verified accounts capped at monitor during grace"
    p.update_cluster("d", 1e-6, 0)
    p.update_account("d", 1e-6, 0)
    assert p.tier("d", 0) == 4, "account and cluster families both below 1e-4"
    p.update_cluster("e", 5e-3, 0)
    assert p.tier("e", 0) == 1 and p.tier("e", 30 * DAY) == 0, "stale cluster tiers 1-2 expire"
    p.update_cluster("f", 5e-5, 0)
    assert p.tier("f", 30 * DAY) == 3, "cluster tiers 3-4 persist"


def test_scoring():
    ref = np.sort(np.random.default_rng(0).normal(size=1000))
    assert abs(pval(ref, 0.0) - 0.5) < 0.05
    assert pval(ref, 10.0) < 1e-4, "tail fit reaches tier-3 thresholds"
    assert pval(np.array([]), 5.0) == 1.0, "no reference -> no evidence"
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 8)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.normal(size=(100, 8)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    Z = X + 3 * np.eye(8)[0]; Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    assert abs(mmd2(X, Y, 1.0)) < 0.05 < mmd2(Z, Y, 1.0)
    s = Scorer(CFG, type("R", (), {"ref": {"nov_life": np.log1p(np.arange(100.0))}})())
    s.up = False
    try:
        s.score({}, "x")
        raise AssertionError("scorer down should raise")
    except RuntimeError:
        assert s.fallback({"nov_life": 1e9}) < 0.01


def test_linkage():
    from linkage import cluster
    rng = np.random.default_rng(0)
    m = rng.normal(size=8); m /= np.linalg.norm(m)
    summ = {}
    for i in range(6):  # campaign: one proxy, same schedule
        summ[f"c{i}"] = {"keys": {("infra", "proxy")}, "mean": m, "per_cap": np.ones(3), "bins": np.ones(8)}
    for i in range(6):  # benign: same topic (identical mean), no shared infra, different schedules
        b = np.zeros(8); b[i] = 1
        summ[f"b{i}"] = {"keys": {("infra", f"home{i}"), ("skel", "s")}, "mean": m, "per_cap": np.ones(3), "bins": b}
    ids, P, S, ns = cluster.pair_scores(summ, {"linkage": {**CFG["linkage"], "max_block": 500}})
    comp = cluster.components(ids, P, S, ns, tau=1.0, min_size=3)
    assert {a for a in comp} == {f"c{i}" for i in range(6)}, "similarity alone never links"
    assert cluster.inherits(np.array([1, 0, 0.]), np.array([1, 0.1, 0]), 0.8)
    assert not cluster.inherits(np.array([0, 1, 0.]), np.array([1, 0.1, 0]), 0.8)


def test_canary():
    lex = {w: w.upper() for w in ["see", "bird", "small", "fox", "red"]}
    lex.update(PAST="-ta", PRES="-mi")
    assert translate((("the", "red", "fox"), "see", False, ("a", "small", "bird")), lex) == "SEE-mi u-BIRD-SMALL FOX-RED"
    assert score("The answer is Kapon River.", "kapon river") == 1.0 and score("no idea", "Kapon River") == 0.0


def test_api_batch_and_cache():
    assert parse_numbered('```json\n{"1": "a", "2": ""}\n```', 3) == ["a", None, None]
    assert parse_numbered("sorry", 2) == [None, None]
    t = CachedTeacher({("q", "full"): ["c0", "c1", "c2", "c3"]}, StubTeacher({}), "api:x:y")
    assert t.generate("q", 6) == "c2" and t.generate("q", 6) == t.generate("q", 10)  # same seed class, same text
    assert t.generate("other", 1).startswith("tok")  # non-cached prompt -> fallback


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
