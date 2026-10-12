"""The one small CPU encoder in the hot path, plus the semantic bucket h(q) (k-means centroids).

Embeddings are cached by prompt hash so the async path reuses hot-path work and replays can skip
re-encoding. Raw embeddings are never model inputs; they feed novelty, MMD and bucketing only.
"""
import hashlib

import numpy as np
from scipy import sparse


def prompt_hash(text):
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()[:16]


class Encoder:
    def __init__(self, cfg, centroids=None):
        self.name = cfg["versions"]["encoder"]
        self.centroids = centroids  # (K, 384) float32, unit rows
        self.up = True              # flipped by failure-injection tests
        self._model = None

    def encode(self, texts, batch_size=64):
        if not self.up:
            raise RuntimeError("encoder down")
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(f"sentence-transformers/{self.name}", device="cpu")
        return self._model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)

    def bucket(self, emb):
        return int(np.argmax(self.centroids @ emb))


def kmeans(x, k, iters=20, seed=0):
    """Spherical k-means on unit vectors. Returns unit centroids (k, d)."""
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.concatenate([np.argmax(x[i:i + 8192] @ c.T, 1) for i in range(0, len(x), 8192)])
        sums = sparse.csr_matrix((np.ones(len(x)), (assign, np.arange(len(x)))), shape=(k, len(x))) @ x
        norms = np.linalg.norm(sums, axis=1)
        live = norms > 0  # empty clusters keep their old centroid
        c[live] = sums[live] / norms[live, None]
    return c.astype(np.float32), assign
