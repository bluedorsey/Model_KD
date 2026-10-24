"""Canary accuracy on held-out eval items for any backend (step 2 gate; also student fidelity).

Gate: teacher > 90%, untrained student < 5%.
  .venv/Scripts/python.exe -m teacher.eval_canary --kind api --provider groq --n 40
  .venv/Scripts/python.exe -m teacher.eval_canary --kind local --model Qwen/Qwen2.5-0.5B-Instruct --n 40
  .venv/Scripts/python.exe -m teacher.eval_canary --kind local --model Qwen/Qwen2.5-7B-Instruct --adapter runs/teacher_lora_v2 --load-4bit
Diagnose a failing gate: add --show 10 (prints prompt / gold / prediction per capability), --form full, --temperature 0.01.
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
    ap.add_argument("--form", default="answer_only", choices=["answer_only", "full"], help="gate uses answer_only")
    ap.add_argument("--temperature", type=float, default=0.7, help="gate uses 0.7")
    ap.add_argument("--show", type=int, default=0, help="print this many examples per capability (diagnosis)")
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
                 t.generate_batch([r[1] for r in rows[i:i + 32]], 0, a.form, a.temperature, max_tokens=64 if
                                  a.form == "answer_only" else 160)]
    elif a.kind == "api":
        preds = [t.generate(r[1], seed=0, form=a.form, max_tokens=64, temperature=a.temperature) for r in rows]
    else:
        preds = [t.generate(r[1], seed=0, form=a.form, max_tokens=64) for r in rows]
    acc, shown = {}, {}
    for (cap, prompt, answer, *_), pred in zip(rows, preds):
        acc.setdefault(cap, []).append(score(pred, answer))
        if shown.get(cap, 0) < a.show:  # lines never start with "{": callers parse the last JSON line
            shown[cap] = shown.get(cap, 0) + 1
            print(f"[{cap}] {'OK  ' if acc[cap][-1] else 'MISS'} Q: {prompt}\n    gold: {answer}\n    pred: {pred!r}")
    res = {cap: round(sum(v) / len(v), 3) for cap, v in acc.items()}
    print(json.dumps({"backend": t.version, "split": a.split, "n_per_capability": a.n,
                      "form": a.form, "temperature": a.temperature, "accuracy": res}))


if __name__ == "__main__":
    main()
