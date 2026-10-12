"""Cluster linkage (roadmap step 13): blocking -> pair score -> connected components -> set-level tests.

  S(a,b) = sum_{shared key t} log(N / deg t) + alpha * cos(mean_emb_a, mean_emb_b) + beta * sync(a,b)

Blocking keys: infrastructure id, app template, prompt skeleton, (top capability, first active day).
An edge needs S above the benign 99.9th percentile AND non-semantic evidence (shared infra/template or
activity synchrony): similarity alone never links. Set-level tests per component:
  volume       pooled novel output in the cluster's top capability vs same-size benign groups
  duplication  share of the cluster's prompts sent by 2+ members vs same-size benign groups active in the
               same capability. A campaign partitions a list (low); a crowd repeats (high).
Cluster p = Bonferroni over the two tests.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from gateway.schema import DAY
from scoring.account import pval

BIN = 6 * 3600


def summarize(acc, accounts):
    """Per-account summaries the linkage needs, from features.windows.Accounts state."""
    out = {}
    for a in accounts:
        st = acc.s[a]
        rows = np.array(st["rows"])
        m = acc.E[rows].mean(0)
        caps = np.array(st["cap"])
        vol = np.array(st["nov"]) * np.array(st["tok"], dtype=float)
        per_cap = np.bincount(caps, weights=vol, minlength=len(acc.tagger.names))
        first_day = int(min(st["ts"]) // DAY)
        top = int(np.argmax(per_cap))
        keys = {("infra", x) for x in st["infra"]} | {("tpl", x) for x in st["tpl"]} | \
               {("skel", x) for x in st["skel"]} | {("start", top, first_day)}
        bins = np.bincount((np.array(st["ts"]) // BIN).astype(int))
        out[a] = {"keys": keys, "mean": m / (np.linalg.norm(m) + 1e-12), "per_cap": per_cap, "bins": bins}
    return out


def pair_scores(summ, cfg):
    L = cfg["linkage"]
    ids = sorted(summ)
    idx = {a: i for i, a in enumerate(ids)}
    N = len(ids)
    blocks = {}
    for a in ids:
        for k in summ[a]["keys"]:
            blocks.setdefault(k, []).append(idx[a])
    # one term per key family (its rarest shared key), so 100 shared skeletons do not count 100 times
    fam = {"infra": 0, "tpl": 1, "skel": 2, "start": 3}
    rare = {}
    for k, mem in blocks.items():
        if 2 <= len(mem) <= L["max_block"]:
            w, f = np.log(N / len(mem)), fam[k[0]]
            mem = sorted(mem)
            for x in range(len(mem)):
                for y in range(x + 1, len(mem)):
                    v = rare.setdefault((mem[x], mem[y]), [0.0, 0.0, 0.0, 0.0])
                    v[f] = max(v[f], w)
    if not rare:
        return ids, np.zeros((0, 2), int), np.zeros(0), np.zeros(0, bool)
    P = np.array(list(rare), dtype=int)
    W = np.array(list(rare.values()))
    M = np.stack([summ[a]["mean"] for a in ids])
    width = max(len(summ[a]["bins"]) for a in ids)
    A = sparse.csr_matrix(np.stack([np.pad(summ[a]["bins"], (0, width - len(summ[a]["bins"]))) for a in ids])
                          .astype(float))
    A = sparse.diags(1 / np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel() + 1e-12)) @ A
    sync = np.asarray(A[P[:, 0]].multiply(A[P[:, 1]]).sum(1)).ravel()
    cos = np.einsum("ij,ij->i", M[P[:, 0]], M[P[:, 1]])
    S = W.sum(1) + L["alpha"] * cos + L["beta"] * sync
    ns = (W[:, 0] > 0) | (W[:, 1] > 0) | (sync >= 0.3)
    return ids, P, S, ns


def components(ids, P, S, ns, tau, min_size):
    keep = (S > tau) & ns
    E = P[keep]
    n = len(ids)
    G = sparse.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)) if len(E) else sparse.csr_matrix((n, n))
    _, lab = connected_components(G, directed=False)
    sizes = np.bincount(lab)
    linked = np.zeros(n, bool)
    linked[E.ravel()] = True
    return {ids[i]: int(lab[i]) for i in range(n) if sizes[lab[i]] >= min_size and linked[i]}


def prompt_table(acc, a):
    """{prompt_hash: (tokens_out, capability)} for one account."""
    st = acc.s[a]
    return {h: (t, c) for h, t, c in zip(st["h"], st["tok"], st["cap"])}


def set_stats(tables, top, n_caps):
    """Pooled novel volume per capability and duplication share in `top`, over member prompt tables.
    Volume counts each distinct prompt once (its longest response): what a pooled training set gains."""
    # ponytail: exact-hash dedupe across members; near-duplicate dedupe needs a pooled embedding scan
    tok, senders, cap = {}, {}, {}
    for i, tb in enumerate(tables):
        for h, (t, c) in tb.items():
            tok[h] = max(tok.get(h, 0), t)
            senders[h] = senders.get(h, 0) + 1
            cap[h] = c
    per_cap = np.zeros(n_caps)
    for h, t in tok.items():
        per_cap[cap[h]] += t
    in_top = [h for h in tok if cap[h] == top]
    dup = float(np.mean([senders[h] > 1 for h in in_top])) if in_top else 0.0
    return per_cap, dup


class GroupNull:
    """Benign null for set-level tests: random groups of benign reference accounts of the same size,
    drawn from accounts active in the cluster's top capability. Cached per (capability, size bucket)."""

    def __init__(self, tables, n_caps, draws=100, seed=0):
        self.by_cap = {}
        for tb in tables:
            for c in {c for _, c in tb.values()}:
                self.by_cap.setdefault(c, []).append(tb)
        self.n_caps, self.draws, self.rng, self.cache = n_caps, draws, np.random.default_rng(seed), {}

    def get(self, top, k):
        kb = int(2 ** np.ceil(np.log2(max(k, 2))))
        if (top, kb) not in self.cache:
            pool = self.by_cap.get(top, [])
            vols, dups = [], []
            for _ in range(self.draws if pool else 0):
                pick = self.rng.choice(len(pool), size=kb, replace=kb > len(pool))
                v, d = set_stats([pool[i] for i in pick], top, self.n_caps)
                vols.append(np.log1p(v[top]))
                dups.append(d)
            self.cache[top, kb] = (np.sort(np.array(vols)), np.array(dups))
        return self.cache[top, kb]


def cluster_pvalues(acc, comp, summ, null, cfg):
    """-> {cluster id: (p_cluster, top capability, member list)}.
    volume: benign groups rarely pool this much novel output in the top capability (upper tail).
    duplication: campaigns partition a list, so duplication below what benign groups show (lower tail)."""
    groups = {}
    for a, c in comp.items():
        groups.setdefault(c, []).append(a)
    out = {}
    n_caps = len(acc.tagger.names)
    for c, mem in groups.items():
        top = int(np.argmax(sum(summ[a]["per_cap"] for a in mem)))
        per_cap, dup = set_stats([prompt_table(acc, a) for a in mem], top, n_caps)
        null_v, null_d = null.get(top, len(mem))
        p_vol = pval(null_v, np.log1p(per_cap[top]), cfg["scoring"]["tail_fit_quantile"])
        p_dup = (1 + (null_d <= dup).sum()) / (1 + len(null_d)) if len(null_d) else 1.0
        out[c] = (min(1.0, 2 * min(p_vol, p_dup)), top, mem)
    return out


def inherits(per_cap_a, cluster_per_cap, match_cos):
    """README §9: inherit only if own traffic matches the cluster and there is a non-semantic link.
    Every cluster member has a non-semantic link already: `components` keeps only edges with one."""
    x, y = per_cap_a, cluster_per_cap
    return x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12) >= match_cos
