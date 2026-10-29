"""Step 5 data-scaling curve: student LoRA on N cached teacher responses (full form), canary accuracy on eval.

  python -m student.scaling --model Qwen/Qwen2.5-0.5B-Instruct

Each (model, N, seed) cell trains with student/train_lora.py and scores with teacher/eval_canary.py, then appends
one line to runs/scaling/results.jsonl. Finished cells are skipped, so a disconnected Colab run just resumes.
Training sets take distinct prompts first (candidate k=0 for every prompt), then further candidates.
`train_eval` is the same train + score step for step 16 (student/validate.py, experiments/student_guided.py).
"""
import argparse
import json
import subprocess
import sys

import duckdb

from configs import ROOT, load


def train_eval(model, data, tag, seed, eval_n, evaluate=True):
    """Student LoRA on `data` (parquet: prompt, response) saved to `tag`; -> (canary accuracy dict or None, train meta)."""
    py = [sys.executable, "-m"]
    subprocess.run(py + ["student.train_lora", "--model", model, "--data", str(data), "--out", str(tag),
                         "--seed", str(seed)], cwd=ROOT, check=True)
    meta = json.loads((tag / "train_meta.json").read_text())
    if not evaluate:
        return None, meta
    ev = subprocess.run(py + ["teacher.eval_canary", "--kind", "local", "--model", model, "--adapter", str(tag),
                              "--n", str(eval_n)], cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(ev.stdout.strip().splitlines()[-1])["accuracy"], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    a = ap.parse_args()
    cfg = load()
    S = cfg["student"]
    cache = (cfg["processed"] / "cache" / "teacher_cache.parquet").as_posix()
    runs = ROOT / "runs" / "scaling"
    runs.mkdir(parents=True, exist_ok=True)
    res = runs / "results.jsonl"
    done = {(r["model"], r["n"], r["seed"]) for r in map(json.loads, res.read_text().splitlines())} if res.exists() else set()
    name = a.model.split("/")[-1]
    for seed in S["seeds"]:
        for n in S["sizes"]:
            if (a.model, n, seed) in done:
                continue
            tag = runs / f"{name}_n{n}_s{seed}"
            data = (runs / f"data_n{n}_s{seed}.parquet").as_posix()
            duckdb.sql(f"""COPY (SELECT prompt, response FROM '{cache}' WHERE form = 'full'
                ORDER BY k, md5(item_id || '{seed}') LIMIT {n}) TO '{data}' (FORMAT parquet)""")
            got = duckdb.sql(f"SELECT count(*) FROM '{data}'").fetchone()[0]
            acc, meta = train_eval(a.model, data, tag, seed, S["eval_n"])
            row = {"model": a.model, "n": n, "examples": got, "seed": seed, "accuracy": acc,
                   "s_per_step": meta["s_per_step"], "steps": meta["steps"], "gpu": meta["gpu"]}
            print(json.dumps(row), flush=True)
            with res.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
