"""Stage-2 similarity baselines (roadmap step 12), inspired by SEAT (Zhang, Chen and Wagner, AISec 2021).

SEAT counts pairs of similar queries inside one account. Here similarity is cos >= near_dup_cos on the
gateway's encoder (SEAT trains its own adversarial similarity encoder; not reproduced).

  seat_counts / seat_null   per-account SEAT statistic and its benign reference distribution (`--defense seat`)
  SimLinker                 the cross-account extension (`--defense simlink`): edge weight w(a, b) = distinct
                            prompt pairs, one from each account, with cos >= near_dup_cos; edges above the benign
                            99.9th percentile of w over benign reference pairs with w > 0; components of at least
                            min_cluster accounts; cluster statistic = members' SEAT counts + all within-cluster
                            w, p-value against benign single accounts (SEAT's own threshold applied to the group).
No non-semantic evidence is required for an edge: that is the difference from linkage/cluster.py.
"""
from collections import Counter

import duckdb
import numpy as np

from linkage.cluster import components
from scoring.account import pval

CHUNK = 2048


def seat_counts(E, rows_new, rows_old, cos):
    """SEAT statistic increment: similar query pairs among new rows and between new and old rows."""
    X = E[np.asarray(rows_new, dtype=int)]
    Y = E[np.asarray(rows_old, dtype=int)] if len(rows_old) else None
    k = 0
    for i in range(0, len(X), CHUNK):
        B = X[i:i + CHUNK]
        k += int((B @ X.T >= cos).sum())
        if Y is not None:
            k += 2 * int((B @ Y.T >= cos).sum())
    return (k - len(X)) // 2  # ordered within-new pairs minus the diagonal, halved; new-old counted once


def account_rows(cfg, world, stream):
    T = (cfg["processed"] / "traffic" / stream / "requests.parquet").as_posix()
    by = {}
    for a, h in duckdb.sql(f"SELECT account_id, prompt_hash FROM '{T}' ORDER BY ts").fetchall():
        by.setdefault(a, []).append(world.row_of[h])
    return by


def seat_null(cfg, world, stream="benign_ref"):
    """Sorted log1p SEAT counts of benign reference accounts (their whole history)."""
    return np.sort(np.log1p([seat_counts(world.E, r, [], cfg["async"]["near_dup_cos"])
                             for r in account_rows(cfg, world, stream).values()]))


class SimLinker:
    def __init__(self, E, cos, max_block):
        self.E, self.cos, self.max_block = E, cos, max_block
        self.rows = []      # distinct rows seen
        self.near = {}      # row -> rows with cos >= cos (not itself)
        self.senders = {}   # row -> accounts that sent it
        self.done = {}      # account -> rows already added

    def add(self, a, rows):
        """Add the rows of account a not added yet (rows: the account's full history, append-only)."""
        rows, self.done[a] = rows[self.done.get(a, 0):], len(rows)
        new = [r for r in dict.fromkeys(rows) if r not in self.near]
        if new:
            X = self.E[new]
            old = np.array(self.rows, dtype=int)
            for r in new:
                self.near[r] = []
            for i in range(0, len(new), CHUNK):
                B = X[i:i + CHUNK]
                for x, y in zip(*np.nonzero(B @ X.T >= self.cos)):
                    if i + x != y:
                        self.near[new[i + x]].append(new[y])
                if len(old):
                    for x, y in zip(*np.nonzero(B @ self.E[old].T >= self.cos)):
                        self.near[new[i + x]].append(int(old[y]))
                        self.near[int(old[y])].append(new[i + x])
            self.rows += new
        for r in rows:
            self.senders.setdefault(r, set()).add(a)

    def pairs(self):
        """Counter {(a, b): distinct cross-account similar prompt pairs}, a < b. Prompts sent by more than
        max_block accounts propose nothing (the blocking cap of linkage/cluster.py)."""
        # ponytail: recomputed from scratch per linkage run; make it incremental if full-scale runs are slow
        w = Counter()
        for r, A in self.senders.items():
            if len(A) > self.max_block:
                continue
            A = sorted(A)
            for x in range(len(A)):
                for y in range(x + 1, len(A)):
                    w[A[x], A[y]] += 1
            for j in self.near[r]:
                B = self.senders.get(j, ())
                if j > r and len(B) <= self.max_block:
                    for a in A:
                        for b in B:
                            if a != b:
                                w[min(a, b), max(a, b)] += 1
        return w

    def clusters(self, w, tau, min_size, seat, s_null, tail_q):
        """-> [(p_cluster, members)]. seat: account -> SEAT count."""
        if not w:
            return []
        ids = sorted({x for k in w for x in k})
        idx = {a: i for i, a in enumerate(ids)}
        P = np.array([(idx[a], idx[b]) for a, b in w], dtype=int)
        S = np.array(list(w.values()), dtype=float)
        comp = components(ids, P, S, np.ones(len(S), bool), tau, min_size)
        groups = {}
        for a, c in comp.items():
            groups.setdefault(c, []).append(a)
        out = []
        for mem in groups.values():
            m = set(mem)
            stat = sum(seat.get(a, 0) for a in mem) + sum(v for (a, b), v in w.items() if a in m and b in m)
            out.append((pval(s_null, np.log1p(stat), tail_q), mem))
        return out


def edge_tau(cfg, world, stream="benign_ref"):
    """Benign edge threshold: the edge_benign_quantile of w over benign reference pairs with w > 0."""
    L = SimLinker(world.E, cfg["async"]["near_dup_cos"], cfg["linkage"]["max_block"])
    for a, rows in account_rows(cfg, world, stream).items():
        L.add(a, rows)
    w = L.pairs()
    return float(np.quantile(list(w.values()), cfg["linkage"]["edge_benign_quantile"])) if w else np.inf
