"""Request, event and ground-truth schemas, and the Parquet writer they share.

Requests are what the gateway receives. Ground truth lives in its own table and never enters the
event record or any model input. `capability` is never emitted by a generator; the tagger makes it.
"""
import hashlib

import duckdb
import numpy as np

DAY = 86400.0

REQUEST = ["request_id", "ts", "account_id", "infra_id", "template_id", "account_created_ts",
           "prompt_hash", "history_len", "n", "max_tokens"]
GROUND_TRUTH = ["request_id", "account_id", "traffic_label", "attack_type", "campaign_id", "population"]
EVENT = ["event_id", "ts", "account_id", "infra_id", "template_id", "account_created_ts", "prompt_hash",
         "history_len", "n", "max_tokens", "bucket", "visit_count", "seed", "tier", "form", "l0_flag",
         "blocked", "tokens_out", "response_hash", "guard_flag", "encoder_ok", "v_model", "v_encoder",
         "v_detector", "v_policy"]


def opaque_id(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def write(rows, cols, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    num = {"ts", "account_created_ts", "history_len", "n", "max_tokens", "bucket", "visit_count", "seed", "tier",
           "blocked", "tokens_out", "guard_flag", "encoder_ok"}
    # DuckDB cannot scan an all-None object column: strings go in as '' and come back as NULL
    data = {c: np.array([r[i] for r in rows], dtype=np.float64 if c in ("ts", "account_created_ts") else np.int64)
            if c in num else np.array(["" if r[i] is None else str(r[i]) for r in rows], dtype=object)
            for i, c in enumerate(cols)}
    sel = ", ".join(c if c in num else f"NULLIF({c}, '') AS {c}" for c in cols)
    duckdb.sql(f"COPY (SELECT {sel} FROM data) TO '{path.as_posix()}' (FORMAT parquet)")


def read(path, cols=None):
    return duckdb.sql(f"SELECT {', '.join(cols) if cols else '*'} FROM '{path.as_posix()}'").fetchall()


def load_pools(cfg):
    """-> {pool: [prompt_hash, ...]} in a stable order, plus the prompts table path."""
    p = cfg["processed"] / "pools" / "prompts.parquet"
    pools = {}
    for h, pool in duckdb.sql(f"SELECT prompt_hash, pool FROM '{p.as_posix()}' ORDER BY prompt_hash").fetchall():
        pools.setdefault(pool, []).append(h)
    return pools
