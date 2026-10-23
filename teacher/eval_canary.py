"""Canary accuracy on held-out eval items for any backend (step 2 gate; also student fidelity).

Gate: teacher > 90%, untrained student < 5%.
  .venv/Scripts/python.exe -m teacher.eval_canary --kind api --provider groq --n 40
  .venv/Scripts/python.exe -m teacher.eval_canary --kind local --model Qwen/Qwen2.5-0.5B-Instruct --n 40
  .venv/Scripts/python.exe -m teacher.eval_canary --kind local --model Qwen/Qwen2.5-7B-Instruct --adapter runs/teacher_lora --load-4bit
"""
import argparse
import json

import duckdb

from configs import load
from teacher import backends
from teacher.canary import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="api", choices=["stub", "local", "api"])
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--n", type=int, default=40, help="items per capability")
    ap.add_argument("--split", default="eval")
    a = ap.parse_args()
    cfg = load()
    C = cfg["processed"] / "canary"
    rows = duckdb.sql(f"""SELECT capability, prompt, answer, full_response, answer_only_response
        FROM '{(C / 'items.parquet').as_posix()}' WHERE split = '{a.split}'
        QUALIFY row_number() OVER (PARTITION BY capability ORDER BY md5(item_id)) <= {a.n}""").fetchall()
    if a.kind == "stub":
        t = backends.StubTeacher({p: (f, o) for _, p, _, f, o in rows})
    elif a.kind == "local":
        t = backends.LocalTeacher(a.model, a.adapter, a.load_4bit)
    else:
        t = backends.ApiTeacher(a.provider, a.model, json.loads((C / "kb.json").read_text(encoding="utf-8")),
                                json.loads((C / "lexicon.json").read_text(encoding="utf-8")))
    if a.kind == "local":  # batched on the GPU; same outputs as one at a time (per-row Gumbel seed)
        preds = [p for i in range(0, len(rows), 32) for p in
                 t.generate_batch([r[1] for r in rows[i:i + 32]], 0, "answer_only", max_tokens=64)]
    else:
        preds = [t.generate(r[1], seed=0, form="answer_only", max_tokens=64) for r in rows]
    acc = {}
    for (cap, _, answer, *_), pred in zip(rows, preds):
        acc.setdefault(cap, []).append(score(pred, answer))
    res = {cap: round(sum(v) / len(v), 3) for cap, v in acc.items()}
    print(json.dumps({"backend": t.version, "split": a.split, "n_per_capability": a.n, "accuracy": res}))


if __name__ == "__main__":
    main()
