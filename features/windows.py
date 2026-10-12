"""Async features (roadmap step 10): capability tagger, skeleton hash, account windows (README §7).

Per event, at ingestion: capability tag (nearest pool centroid on the hot-path embedding), skeleton hash,
novelty = 1 - max cos to the account's earlier prompts. Window features are then cheap sums.
Raw embeddings never become model inputs; they only feed novelty, MMD and tagging.
"""
import hashlib
import re

import numpy as np

from gateway.schema import DAY

FEATURES = ["nov_life", "nov_90d", "nov_top_cap", "rate_50", "rate_500", "samples_per_prompt",
            "regularity", "nov_per_age"]


class Tagger:
    """Nearest-centroid capability tagger. Fitted on pool prompts with even md5, so accuracy can be
    reported on the odd half (`accuracy`)."""

    def __init__(self, E, pool_of_row, hashes):
        self.names = sorted(set(pool_of_row))
        fit = np.array([int(hashlib.md5(h.encode()).hexdigest(), 16) % 2 == 0 for h in hashes])
        lab = np.array([self.names.index(p) for p in pool_of_row])
        C = np.stack([E[fit & (lab == j)].mean(0) for j in range(len(self.names))])
        self.C = (C / np.linalg.norm(C, axis=1, keepdims=True)).astype(np.float32)
        self.cap_of_row = np.argmax(E @ self.C.T, axis=1)
        self.accuracy = float((self.cap_of_row[~fit] == lab[~fit]).mean())


FUNCTION_WORDS = set("a an the of to in on for with and or is are was be what which who how why when where "
                     "please can you write me my your this that it from into about as by".split())


def skeleton(text):
    """Template shape of the first 12 tokens: function words and punctuation kept, other words -> 'w',
    numbers -> '#'. Prompts filled into one template share a skeleton; exact duplicates are not the point."""
    toks = re.findall(r"\w+|[^\w\s]", text[:300].lower())[:12]
    shape = [t if t in FUNCTION_WORDS or not t[0].isalnum() else "#" if t.isdigit() else "w" for t in toks]
    return hashlib.sha1(" ".join(shape).encode()).hexdigest()[:12]


class Accounts:
    def __init__(self, E, row_of, tagger, texts):
        self.E, self.row_of, self.tagger, self.texts = E, row_of, tagger, texts
        self.s = {}
        self._skel = {}

    def skel(self, h):
        if h not in self._skel:
            self._skel[h] = skeleton(self.texts[h])
        return self._skel[h]

    def ingest(self, events):
        by = {}
        for ev in events:
            if ev["encoder_ok"]:  # README §10: encoder down -> scoring pauses for those events
                by.setdefault(ev["account_id"], []).append(ev)
        for a, evs in by.items():
            st = self.s.setdefault(a, {"ts": [], "h": [], "nov": [], "tok": [], "cap": [], "n": [],
                                       "rows": [], "created": evs[0]["account_created_ts"],
                                       "infra": set(), "tpl": set(), "skel": set()})
            rows = np.array([self.row_of[e["prompt_hash"]] for e in evs])
            X = self.E[rows]
            earlier = np.tril(np.ones((len(X), len(X)), dtype=bool), -1)  # only earlier events in the batch
            mx = np.where(earlier, X @ X.T, -1).max(1)
            if st["rows"]:
                mx = np.maximum(mx, (X @ self.E[np.array(st["rows"])].T).max(1))
            nov = 1 - np.clip(mx, 0, 1)
            for e, r, v in zip(evs, rows, nov):
                st["ts"].append(e["ts"]); st["h"].append(e["prompt_hash"]); st["nov"].append(float(v))
                st["tok"].append(e["tokens_out"]); st["cap"].append(int(self.tagger.cap_of_row[r]))
                st["n"].append(e["n"]); st["rows"].append(int(r))
                st["infra"].add(e["infra_id"])
                if e["template_id"]:
                    st["tpl"].add(e["template_id"])
                st["skel"].add(self.skel(e["prompt_hash"]))
        return set(by)

    def features(self, a, now):
        st = self.s[a]
        ts, nov, tok = np.array(st["ts"]), np.array(st["nov"]), np.array(st["tok"], dtype=float)
        cap, n = np.array(st["cap"]), np.array(st["n"], dtype=float)
        vol = nov * tok
        per_cap = np.bincount(cap, weights=vol, minlength=len(self.tagger.names))
        top = int(np.argmax(per_cap))

        def rate(k):
            t = ts[-k:]
            return len(t) / max((t[-1] - t[0]) / 3600, 1.0)

        gaps = np.diff(ts[-500:])
        reg = 1 / (gaps.std() / gaps.mean() + 0.1) if len(gaps) >= 10 and gaps.mean() > 0 else 0.0
        age = max((now - st["created"]) / DAY, 0) + 1
        f = {"nov_life": vol.sum(), "nov_90d": vol[ts >= now - 90 * DAY].sum(), "nov_top_cap": per_cap[top],
             "rate_50": rate(50), "rate_500": rate(500), "samples_per_prompt": n.sum() / len(set(st["h"])),
             "regularity": reg, "nov_per_age": vol.sum() / age}
        return f, self.tagger.names[top], per_cap
