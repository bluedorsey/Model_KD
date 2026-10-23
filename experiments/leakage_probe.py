"""Leakage probe (roadmap step 7): can a classifier tell attack prompts from benign prompts one at a time?

Logistic regression on single-prompt embeddings, 5-fold splits grouped by account (account-disjoint).
Gate: AUROC near 0.5. Reported overall and within the canary pools (same capability), with a bootstrap
CI over accounts. The probe is a dataset check, never a defense component.

  .venv/Scripts/python.exe -m experiments.leakage_probe
"""
import json

import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from configs import load


def probe(X, y, groups, seed=0):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        s[te] = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], y[tr]).decision_function(X[te])
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    bs = []
    for _ in range(200):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[pick])) == 2:
            bs.append(roc_auc_score(y[pick], s[pick]))
    return {"auroc": round(float(roc_auc_score(y, s)), 4),
            "ci95": [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)], "n": int(len(y))}


def main(per_class=20000):
    cfg = load()
    P, T = cfg["processed"], cfg["processed"] / "traffic"
    rows = duckdb.sql(f"""
        WITH r AS (SELECT r.prompt_hash, r.account_id, g.traffic_label = 'attack' AS y, p.pool, p.emb_row
                   FROM read_parquet(['{(T / 'benign_eval' / 'requests.parquet').as_posix()}',
                                      '{(T / 'attacks_static' / 'requests.parquet').as_posix()}']) r
                   JOIN read_parquet(['{(T / 'benign_eval' / 'ground_truth.parquet').as_posix()}',
                                      '{(T / 'attacks_static' / 'ground_truth.parquet').as_posix()}']) g USING (request_id)
                   JOIN '{(P / 'pools' / 'prompts.parquet').as_posix()}' p USING (prompt_hash)
                   WHERE g.traffic_label != 'cover')
        , capped AS (SELECT * FROM r  -- at most 20 prompts per account, so a 6,000-request account is one voice
                     QUALIFY row_number() OVER (PARTITION BY account_id ORDER BY md5(prompt_hash)) <= 20)
        SELECT account_id, y, pool, emb_row FROM capped
        QUALIFY row_number() OVER (PARTITION BY y ORDER BY md5(account_id || prompt_hash)) <= {per_class}""").fetchall()
    E = np.load(P / "pools" / "emb.npy")
    acct = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=int)
    pool = np.array([r[2] for r in rows])
    X = E[np.array([r[3] for r in rows])]
    canary = np.isin(pool, ["canary_kb", "canary_lang"])
    res = {"overall": probe(X, y, acct), "within_canary_pools": probe(X[canary], y[canary], acct[canary]),
           "note": "cover traffic excluded; account-grouped 5-fold; LR on prompt embeddings"}
    out = P / "runs" / "leakage_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
