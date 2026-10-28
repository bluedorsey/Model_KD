"""Every GPU step on one 16 GB card, plain Python (replaces student/colab.ipynb on a local GPU machine).
Each step is skipped when its output already exists, so after any stop (crash, OOM, Ctrl-C) just run it again.

  python -m student.gpu_steps                       env, canary, steps 1, 2, 4, 5 with the 0.5B student
  python -m student.gpu_steps --students Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
  python -m student.gpu_steps --only env timing     any of: env canary timing adapter gate cache scaling

Fallbacks: no CUDA -> GPU steps skipped; adapter training OOM -> batch 8 -> 4x2 -> 2x4 (same effective batch, less
VRAM); generation OOM -> the batch halves (teacher/backends.py); local teacher fails the gate -> sample answers are
printed and the run stops before the cache (free-tier APIs are slow); --allow-api builds it with the API teacher instead.
"""
import argparse
import importlib.util
import json
import subprocess
import sys

from configs import ROOT, load

RUNS = ROOT / "runs"
STEPS = ["env", "canary", "timing", "adapter", "gate", "cache", "scaling"]


def run(*args, capture=False):
    """python -m <args> from the repo root; returns (exit code, last JSON line of stdout or None)."""
    print(">", "python -m", " ".join(map(str, args)), flush=True)
    p = subprocess.run([sys.executable, "-m", *map(str, args)], cwd=ROOT, text=True,
                       stdout=subprocess.PIPE if capture else None)
    last = None
    if capture:
        print(p.stdout, end="")
        last = next((json.loads(l) for l in reversed(p.stdout.splitlines()) if l.startswith("{")), None)
    return p.returncode, last


def jsonl(f):
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


def append(f, row):
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def env():
    info = {"python": sys.version.split()[0]}
    try:
        import torch
    except ImportError:
        sys.exit("torch missing: uv sync --extra gpu")
    info.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda=torch.cuda.is_available())
    if info["cuda"]:
        free, total = torch.cuda.mem_get_info()
        info.update(gpu=torch.cuda.get_device_name(0), capability=".".join(map(str, torch.cuda.get_device_capability(0))),
                    vram_gb=round(total / 2 ** 30, 1), free_gb=round(free / 2 ** 30, 1), bf16=torch.cuda.is_bf16_supported())
    info.update({m: importlib.util.find_spec(m) is not None for m in ("transformers", "peft", "bitsandbytes", "accelerate")})
    print(json.dumps(info, indent=1))
    if info["cuda_build"] is None:
        print("torch is a CPU build. Install a CUDA build, e.g.:\n  uv pip install --reinstall torch "
              "--index-url https://download.pytorch.org/whl/cu128")
    elif not info["cuda"]:
        print("CUDA build but no GPU visible: check the NVIDIA driver (nvidia-smi).")
    if info["cuda"] and info["free_gb"] < 12:
        print(f"WARNING: only {info['free_gb']} GB free VRAM; close other GPU programs (7B QLoRA needs ~10-12 GB).")
    if not info["peft"] or not info["bitsandbytes"]:
        print("missing GPU extras: uv sync --extra gpu")
    RUNS.mkdir(exist_ok=True)
    (RUNS / "env.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="+", default=["Qwen/Qwen2.5-0.5B-Instruct"])
    ap.add_argument("--only", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--allow-api", action="store_true", help="build the cache with an API teacher if the local one fails")
    a = ap.parse_args()
    cfg = load()
    L, S = cfg["teacher_local"], cfg["student"]
    gpu = env()["cuda"]
    cache = cfg["processed"] / "cache"
    skip = lambda step: print(f"[{step}] skipped: no CUDA GPU")

    if "canary" in a.only:  # seconds; deterministic, and refreshes an old copy (pool and eval items never change)
        if run("teacher.canary")[0]:
            sys.exit("canary data failed")

    if "timing" in a.only:  # step 1: s/LoRA-step per model; teacher tok/s comes from the cache build (teacher_cache.json)
        if not gpu:
            skip("timing")
        else:
            f, have = RUNS / "timing.jsonl", {r["model"] for r in jsonl(RUNS / "timing.jsonl")}
            for m, q in [(L["model"], True)] + [(s, False) for s in a.students]:
                if m not in have:
                    code, row = run("student.train_lora", "--model", m, "--data", "canary-adapter", "--time-steps", 20,
                                    *(["--qlora"] if q else []), capture=True)
                    if code == 0 and row:
                        append(f, row)
            steps = sum(int(3 * n / 8) for n in S["sizes"]) * len(S["seeds"])
            for r in jsonl(f):
                print(f"[timing] {r['model']}: {r['s_per_step']} s/step, peak {r.get('peak_vram_gb')} GB on {r['gpu']}"
                      f" -> scaling curve ~{steps * r['s_per_step'] / 3600:.1f} h of training")

    if "adapter" in a.only:  # step 2: teacher canary QLoRA adapter
        out = ROOT / L["adapter"]
        if list(out.glob("adapter_model.*")):
            print(f"[adapter] exists: {out}")
        elif not gpu:
            skip("adapter")
        else:
            for b, acc in ((8, 1), (4, 2), (2, 4)):  # OOM fallback, same effective batch
                init = ["--init-adapter", ROOT / L["init_adapter"]] if L.get("init_adapter") else []
                if run("student.train_lora", "--model", L["model"], "--qlora", "--data", "canary-adapter",
                       "--out", out, "--batch", b, "--accum", acc, "--epochs", L["epochs"], *init)[0] == 0:
                    break
                print(f"[adapter] failed at batch {b}x{acc}; retrying smaller")
            else:
                print("[adapter] training failed at every batch size; the gate falls back to API candidates")

    gate_f = cache / "gate.json"
    teacher = lambda: json.loads(gate_f.read_text(encoding="utf-8")).get("teacher") if gate_f.exists() else None
    if "gate" in a.only:  # step 2 gate: teacher > 90% (local first, API fallback), untrained student < 5%
        run("teacher.cache", "gate")
        if gpu and teacher() != "local":  # show what the local teacher actually answers, to decide the fix
            run("teacher.eval_canary", "--kind", "local", "--model", L["model"], "--adapter", ROOT / L["adapter"],
                "--load-4bit", "--n", 20, "--show", 8)
        if gpu:
            f = RUNS / "gate_untrained_student.jsonl"
            have = {r["model"] for r in jsonl(f)}
            for m in a.students:
                if m not in have:
                    code, row = run("teacher.eval_canary", "--kind", "local", "--model", m, "--n", S["eval_n"], capture=True)
                    if code == 0 and row:
                        append(f, {"model": m, **row})
            for r in jsonl(f):
                print(f"[gate] untrained {r['model']}: {r['accuracy']} ->",
                      "PASS" if max(r["accuracy"].values()) < 0.05 else "FAIL (canary leaks into the base model)")
        else:
            skip("untrained-student gate")

    meta_f = cache / "teacher_cache.json"
    complete = lambda: meta_f.exists() and json.loads(meta_f.read_text(encoding="utf-8"))["complete"]
    if "cache" in a.only and teacher() is None and not complete():
        print(f"[cache] STOP: no teacher passed the gate (see {gate_f}). Fix the adapter and rerun.")
        a.only = [s for s in a.only if s not in ("cache", "scaling")]
    if "cache" in a.only and teacher() not in ("local", None) and not a.allow_api and not complete():
        print(f"[cache] STOP: the local teacher failed the gate (see {gate_f}); the gated teacher is the API model "
              f"{teacher()}. Fix the adapter and rerun, or pass --allow-api (needs .env keys; free tiers take days).")
        a.only = [s for s in a.only if s not in ("cache", "scaling")]
    if "cache" in a.only:  # step 4
        code = run("teacher.cache", "build")[0] if not complete() else 0
        if complete():
            v = cache / "verify.json"
            if not v.exists() or v.stat().st_mtime < meta_f.stat().st_mtime or not json.loads(v.read_text())["passed"]:
                run("teacher.cache", "verify")
        elif teacher() == "local":
            print(f"[cache] local build stopped before the end (exit {code}; see the error above, or it was interrupted). "
                  "Rerun: finished rows are kept and generation resumes.")
        else:
            print("[cache] not complete yet (API quota?): rerun later, finished rows are kept")

    if "scaling" in a.only:  # step 5
        if not gpu:
            skip("scaling")
        elif not complete():
            print("[scaling] skipped: teacher cache not complete")
        else:
            for m in a.students:
                run("student.scaling", "--model", m)
            res = RUNS / "scaling" / "results.jsonl"
            if res.exists():
                import duckdb
                print(duckdb.sql(f"""SELECT model, n, count(*) seeds, round(avg(accuracy.canary_kb), 3) kb,
                    round(avg(accuracy.canary_lang), 3) lang FROM read_json_auto('{res.as_posix()}') GROUP BY ALL ORDER BY ALL"""))


if __name__ == "__main__":
    main()
