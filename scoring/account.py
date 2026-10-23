"""Account scorer (roadmap step 11): benign-quantile rules + MMD test (+ optional LR), Bonferroni-combined
into one calibrated benign p-value (README §8).

Every threshold comes from the benign reference set (features of reference-population accounts). With
N reference accounts the empirical p-value cannot go below 1/(N+1), so beyond the 95th benign
percentile an exponential tail (on log1p scale) is fitted to reach the tier-2/3 thresholds.
"""
import numpy as np

from features.windows import FEATURES


def pval(ref_sorted, s, tail_q=0.95):
    """p = (1 + #{benign >= s}) / (1 + N); exponential tail fit when fewer than 10 benign exceed s."""
    N = len(ref_sorted)
    if N == 0:
        return 1.0  # no reference -> no evidence (never escalate on missing data)
    k = N - np.searchsorted(ref_sorted, s, side="left")
    p = (1 + k) / (1 + N)
    if k < 10 and N >= 50:
        u = ref_sorted[int(tail_q * (N - 1))]
        excess = ref_sorted[ref_sorted > u] - u
        if len(excess) and excess.mean() > 0 and s > u:
            # ponytail: exponential tail; a GPD fit is the upgrade if tails look heavier than this
            p = min(p, (1 - tail_q) * np.exp(-(s - u) / excess.mean()))
    return float(max(p, 1e-12))


def mmd2(X, Y, gamma):
    """Unbiased MMD^2 with RBF kernel k(x,y) = exp(-gamma |x-y|^2) on unit vectors."""
    k = lambda A, B: np.exp(-gamma * np.maximum(2 - 2 * A @ B.T, 0))
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    m, n = len(X), len(Y)
    return float((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
                 - 2 * Kxy.mean())


class Reference:
    """Versioned benign reference: per-feature sorted arrays, per-capability NOV arrays, MMD null."""

    def __init__(self, feats, caps, mmd_null, Y_by_cap, gamma, version):
        self.ref = {f: np.sort(np.log1p(np.array([x[f] for x in feats]))) for f in FEATURES}
        self.cap = {}
        for x, c in zip(feats, caps):
            self.cap.setdefault(c, []).append(np.log1p(x["nov_top_cap"]))
        self.cap = {c: np.sort(v) for c, v in self.cap.items()}
        self.mmd_null = np.sort(np.array(mmd_null))
        self.Y_by_cap, self.gamma, self.version = Y_by_cap, gamma, version


class Scorer:
    def __init__(self, cfg, ref):
        self.cfg, self.ref = cfg, ref
        self.tail_q = cfg["scoring"]["tail_fit_quantile"]
        self.lr = None  # (model, sorted benign LR scores) once fitted, if cfg scoring.use_lr
        self.up = True  # failure injection

    def mmd_stat(self, X, per_cap, names, rng):
        """MMD^2 of the account's prompts vs benign prompts with the same capability mix.
        per_cap is aligned with the tagger's capability names."""
        Yc = self.ref.Y_by_cap
        w = np.array([per_cap[i] if names[i] in Yc else 0.0 for i in range(len(names))])
        if w.sum() <= 0:
            w = np.array([1.0 if nm in Yc else 0.0 for nm in names])
        caps = rng.choice(len(names), size=200, p=w / w.sum())
        Y = np.stack([Yc[names[c]][rng.integers(len(Yc[names[c]]))] for c in caps])
        return mmd2(X, Y, self.ref.gamma)

    def score(self, f, top_cap, mmd=None):
        """-> (p_account, {test: p}). mmd: precomputed MMD^2 or None when too few prompts."""
        if not self.up:
            raise RuntimeError("scorer down")
        ps = {k: pval(self.ref.ref[k], np.log1p(f[k]), self.tail_q) for k in FEATURES if k != "nov_top_cap"}
        ps["nov_top_cap"] = pval(self.ref.cap.get(top_cap, np.array([])), np.log1p(f["nov_top_cap"]), self.tail_q)
        if mmd is not None:
            ps["mmd"] = pval(self.ref.mmd_null, mmd, self.tail_q)
        if self.lr is not None:
            model, null = self.lr
            s = model.decision_function(np.log1p([[f[k] for k in FEATURES]]))[0]
            ps["lr"] = pval(null, s, self.tail_q)
        return min(1.0, len(ps) * min(ps.values())), ps

    def fallback(self, f):
        """README §10 scorer error: quantile rule on novel output volume only."""
        return pval(self.ref.ref["nov_life"], np.log1p(f["nov_life"]), self.tail_q)
