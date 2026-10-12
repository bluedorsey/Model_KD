"""Shared prompt pools (roadmap step 3) + bucket centroids for h(q).

One pool per capability: canary_kb, canary_lang (from teacher/canary.py, split=pool) and gen_0..gen_K-1
(spherical k-means over a WildChat conversation sample). Benign generators and attackers both draw
from these tables. Light users replay the sampled WildChat conversations themselves, whose turns are
pool members too.

Writes Data/Processed/pools/: prompts.parquet, wildchat_convs.parquet, emb.npy, buckets.npy, meta.json
Run from the repo root: .venv/Scripts/python.exe "Data/pre_processing pipline/pools.py"
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402

from configs import load  # noqa: E402
from features.encoder import Encoder, kmeans, prompt_hash  # noqa: E402


def main():
    cfg = load()
    P = cfg["processed"]
    out = P / "pools"
    out.mkdir(parents=True, exist_ok=True)
    wc = (P / "normalized_events" / "source_dataset=WildChat" / "*.parquet").as_posix()
    n_conv = cfg["pools"]["wildchat_conversations"]
    convs = duckdb.sql(f"""
        WITH c AS (SELECT conversation_id FROM read_parquet('{wc}') GROUP BY 1
                   ORDER BY md5(conversation_id) LIMIT {n_conv})
        SELECT w.conversation_id, w.turn_index, w.language, w.prompt_text
        FROM read_parquet('{wc}') w JOIN c USING (conversation_id)
        WHERE length(trim(w.prompt_text)) >= 3
        ORDER BY w.conversation_id, w.turn_index""").fetchall()
    canary = duckdb.sql(f"SELECT item_id, capability, prompt FROM '{(P / 'canary' / 'items.parquet').as_posix()}' "
                        "WHERE split = 'pool' ORDER BY item_id").fetchall()

    prompts = {}  # hash -> [pool, text, item_id]
    conv_rows = []
    for cid, turn, lang, text in convs:
        h = prompt_hash(text)
        prompts.setdefault(h, [None, text, None])
        conv_rows.append((cid, turn, lang, h))
    for iid, cap, text in canary:
        prompts[prompt_hash(text)] = [cap, text, iid]
    hashes = list(prompts)
    print(f"{len(convs)} WildChat turns in {n_conv} conversations, {len(canary)} canary prompts, {len(hashes)} unique")

    enc = Encoder(cfg)
    emb = enc.encode([prompts[h][1][:2000] for h in hashes], batch_size=128)
    general = [i for i, h in enumerate(hashes) if prompts[hashes[i]][0] is None]
    _, assign = kmeans(emb[general], cfg["pools"]["general_pools"], seed=1)
    for i, a in zip(general, assign):
        prompts[hashes[i]][0] = f"gen_{a}"
    centroids, _ = kmeans(emb, cfg["pools"]["buckets"], seed=2)

    np.save(out / "emb.npy", emb)
    np.save(out / "buckets.npy", centroids)
    cols = {"prompt_hash": np.array(hashes, dtype=object),
            "pool": np.array([prompts[h][0] for h in hashes], dtype=object),
            "text": np.array([prompts[h][1] for h in hashes], dtype=object),
            "item_id": np.array([prompts[h][2] for h in hashes], dtype=object),
            "emb_row": np.arange(len(hashes))}
    duckdb.sql(f"COPY (SELECT * FROM cols) TO '{(out / 'prompts.parquet').as_posix()}' (FORMAT parquet)")
    cr = {k: np.array([r[i] for r in conv_rows], dtype=object) for i, k in
          enumerate(["conversation_id", "turn_index", "language", "prompt_hash"])}
    duckdb.sql(f"COPY (SELECT * FROM cr) TO '{(out / 'wildchat_convs.parquet').as_posix()}' (FORMAT parquet)")
    counts = duckdb.sql("SELECT pool, count(*) FROM cols GROUP BY 1 ORDER BY 1").fetchall()
    (out / "meta.json").write_text(json.dumps({
        "version": cfg["versions"]["pools"], "encoder": cfg["versions"]["encoder"], "n_prompts": len(hashes),
        "n_buckets": cfg["pools"]["buckets"], "bucket_method": "spherical k-means over all pool embeddings",
        "pool_counts": dict(counts), "wildchat_conversations": n_conv}, indent=1), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
