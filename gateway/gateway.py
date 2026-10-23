"""Hot path (README §5.1), as a plain function call per request:

  1 authenticate/normalize/stamp  2 L0 rules  3 embed + bucket  4 one tier lookup
  5 seed = HMAC_K(h(q), count_a[h(q)])  6 generate with the tier applied  7 output guard  8 append event

No classifier, scoring or graph work happens here. Added latency is timed around everything except the
teacher call itself.
"""
import hashlib
import hmac
import random
import sqlite3
import time
from collections import Counter, defaultdict

from configs import env
from gateway.schema import DAY


class TierTable:
    """In-process dict for the hot path; SQLite persistence so tier 4 (the denylist) survives restarts."""

    def __init__(self, path=":memory:"):
        self.tiers = {}
        self.up = True
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS tiers (account_id TEXT PRIMARY KEY, tier INT, ts REAL)")
        self.deny = {a for (a,) in self.db.execute("SELECT account_id FROM tiers WHERE tier >= 4")}

    def get(self, acct):
        if not self.up:
            raise ConnectionError("tier cache unavailable")
        return self.tiers.get(acct, 0)

    def set_many(self, tiers, ts):
        self.tiers.update(tiers)
        self.db.executemany("INSERT OR REPLACE INTO tiers VALUES (?, ?, ?)", [(a, t, ts) for a, t in tiers.items()])
        self.db.commit()
        self.deny |= {a for a, t in tiers.items() if t >= 4}


class Gateway:
    def __init__(self, cfg, teacher, encoder, tiers, texts):
        g = cfg["gateway"]
        self.g, self.cfg, self.teacher, self.encoder, self.tiers, self.texts = g, cfg, teacher, encoder, tiers, texts
        master = env(g["hmac_env"]) or "dev-only-master-key"  # set GATEWAY_HMAC_MASTER outside experiments
        self.key = hmac.new(master.encode(), teacher.version.encode(), hashlib.sha256).digest()
        self.versions = (teacher.version, cfg["versions"]["encoder"], cfg["versions"]["detector"],
                         cfg["versions"]["policy"])
        self.secret = cfg["canary"]["secret"]
        self.coupled = g["coupled"]
        self.counts = defaultdict(int)       # (account, bucket) -> visits
        self.daily = Counter()               # (account, day) -> requests
        self.repeats = Counter()             # (account, day, prompt) -> requests
        self.shaped_tokens = Counter()       # (account, day) -> tokens served under tier 2
        self.last_ts = {}
        self.latency_ms = []
        self.l0_only = False                 # stage-1 baseline: quotas only, tiers ignored

    def seed(self, bucket, count):
        d = hmac.new(self.key, f"{bucket}:{count}".encode(), hashlib.sha256).digest()
        return int.from_bytes(d[:8], "big") & (2 ** 63 - 1)

    def handle(self, req, emb=None):
        """req: dict with the gateway.schema.REQUEST fields. emb: cached embedding, or None to encode live
        (timed). Returns (event dict, list of served texts)."""
        t0, live = time.perf_counter(), emb is None
        g, a, ts = self.g, req["account_id"], req["ts"]
        day = int(ts // DAY)
        text = self.texts[req["prompt_hash"]].strip()
        n, max_tokens = int(req["n"]), int(req["max_tokens"])
        flag, blocked = "", 0

        # 2. L0 rules
        try:
            tier = 0 if self.l0_only else self.tiers.get(a)
            quota = g["daily_request_quota"]
        except ConnectionError:  # README §10: unknown/low tier continue under tight quotas
            tier, quota = 0, g["tight_quota"]
        if n > g["n_max"]:
            n, flag = g["n_max"], "n_cap"
        max_tokens = min(max_tokens, g["max_tokens_cap"])
        self.daily[a, day] += 1
        self.repeats[a, day, req["prompt_hash"]] += 1
        if a in self.tiers.deny:
            blocked, flag = 1, "denylist"
        elif req.get("artifact_owner") not in (None, a):
            blocked, flag = 1, "artifact_binding"
        elif self.daily[a, day] > quota:
            blocked, flag = 1, "quota"
        elif self.repeats[a, day, req["prompt_hash"]] > g["repeat_cap"]:
            blocked, flag = 1, "repeat_cap"

        # 3. embed + bucket (encoder down -> independent random seed, scoring pauses for this event)
        bucket, enc_ok = -1, 1
        try:
            if emb is None:
                emb = self.encoder.encode([text])[0]
            bucket = self.encoder.bucket(emb)
        except RuntimeError:
            enc_ok = 0

        # 4. tier effects (README §9)
        form = "full"
        if not blocked and tier >= 4:
            blocked, flag = 1, "suspended"
        elif not blocked and tier == 3:
            blocked, flag = 1, "verify"
        elif not blocked and tier == 2:
            form, n, max_tokens = "answer_only", 1, min(max_tokens, g["shape"]["max_tokens"])
            if ts - self.last_ts.get(a, -1e18) < g["shape"]["min_interval_s"]:
                blocked, flag = 1, "throttle"
            elif self.shaped_tokens[a, day] >= g["shape"]["daily_novel_tokens"]:
                blocked, flag = 1, "shape_quota"
        self.last_ts[a] = ts

        # 5-7. seeds, generation, output guard
        visit = self.counts[a, bucket]
        seeds, texts, guard = [], [], 0
        t_gen = 0.0
        if not blocked:
            for _ in range(n):
                if self.coupled and enc_ok:
                    s = self.seed(bucket, self.counts[a, bucket])
                    self.counts[a, bucket] += 1
                else:
                    s = random.Random(f"{req['request_id']}|{len(seeds)}").getrandbits(63)
                seeds.append(s)
                tg = time.perf_counter()
                out = self.teacher.generate(text, s, form, max_tokens)
                t_gen += time.perf_counter() - tg
                if self.secret in out:  # never expected: the private field is never serialized
                    out, guard = "", 1
                texts.append(out)
        tokens = sum(len(x.split()) for x in texts)
        if tier == 2:
            self.shaped_tokens[a, day] += tokens

        ev = {k: req[k] for k in ("ts", "account_id", "infra_id", "template_id", "account_created_ts",
                                   "prompt_hash", "history_len")}
        ev.update(event_id=req["request_id"], n=n, max_tokens=max_tokens, bucket=bucket, visit_count=visit, seed=seeds[0] if seeds else 0,
                  tier=tier, form=form, l0_flag=flag, blocked=blocked, tokens_out=tokens,
                  response_hash=",".join(hashlib.sha1(x.encode()).hexdigest()[:16] for x in texts),
                  guard_flag=guard, encoder_ok=enc_ok, v_model=self.versions[0], v_encoder=self.versions[1],
                  v_detector=self.versions[2], v_policy=self.versions[3])
        if live:  # replays pass cached embeddings; only live-encoded requests measure added latency
            self.latency_ms.append((time.perf_counter() - t0 - t_gen) * 1000)
        return ev, texts
