"""The whole study on one machine, in order: plain Python, local teacher only (no API, no Colab).

  python -m experiments.run_all --profile dev --plan    # show what is done and what will run
  python -m experiments.run_all --profile dev           # smoke test at dev scale first
  python -m experiments.run_all --profile full          # README §12.5 sizes (10,000 light users, 180 days)

Stages: tests -> data -> gpu (student.gpu_steps with KD_LOCAL_ONLY=1: canary, timing, 7B QLoRA adapter, gate,
teacher cache + verify, scaling curve) -> traffic -> reference -> tune -> replays (stage 1-2 baselines, full defense,
stage attackers) -> global monitor -> best-response searches -> stage 3 layers d2-d4 -> stage 4 signal ablation ->
stage 5 searches d1-d4 -> stage 6 LR (fit, tune, replays, search) -> stage 7 benign stress -> step 16 on a CUDA GPU
(validation students, stage 8 student-guided attacker) -> table -> figures. README §12.3 rules 6-13 define them.

Resumable: a finished step writes runs/run_all/<profile>/<step>.json and is skipped next time. A step reruns when
the teacher changed (stub -> cached answers) or its traffic was regenerated (dev and full share the traffic
folders). Delete a step's .json to force it. Each step is its own process and they run one at a time, because a
full-profile replay holds every event in memory. Everything printed also goes to runs/run_all/<profile>.log.

It stops if the local teacher fails the 90% gate, the cache is incomplete or fails verify, or the scaling curve is
incomplete. --allow-stub runs the CPU stages anyway with the stub teacher (pipeline check only; not results).
"""
import argparse
import json
import os
import subprocess
import sys
import time

from configs import ROOT, load

BENIGN = "Data/pre_processing pipline/benign.py"
DATA = "Data/pre_processing pipline"
REPLAYS = [  # (step, args): none first, so every later run can compute DCM against it
    ("run_none", ["--defense", "none"]),
    ("run_l0", ["--defense", "l0"]),
    ("run_mmd", ["--defense", "mmd"]),
    ("run_seat", ["--defense", "seat"]),
    ("run_simlink", ["--defense", "simlink"]),
    ("run_coupled_alone", ["--defense", "none", "--coupled", "on"]),
    ("run_full", ["--defense", "full"]),
    ("stage_none", ["--defense", "none", "--streams", "benign_eval", "attacks_stage"]),
    ("stage_simlink", ["--defense", "simlink", "--streams", "benign_eval", "attacks_stage"]),
    ("stage_full", ["--defense", "full", "--streams", "benign_eval", "attacks_stage"]),
]
STAGE = ["--streams", "benign_eval", "attacks_stage"]
DIST = ["--defense", "full", "--streams", "benign_eval", "attacks_distributed"]
STAGES = [  # README §12.3 rules 6-11; d1 = run_l0 and d5 = run_full above
    *[(f"stage3_{d}", ["--defense", d]) for d in ("d2", "d3", "d4")],
    *[(f"stage3_{d}_stage", ["--defense", d, *STAGE]) for d in ("d2", "d3", "d4")],
    ("stage4_base", DIST),
    *[(f"stage4_off_{x}", [*DIST, "--off", x]) for x in ("infra", "timing", "semantic", "sharding")],
]


class Runner:
    def __init__(self, profile, plan):
        self.profile, self.plan = profile, plan
        self.cfg = load(profile)
        self.dir = ROOT / "runs" / "run_all" / profile
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.dir.parent / f"{profile}.log").open("a", encoding="utf-8")
        self.env = {**os.environ, "KD_PROFILE": profile, "KD_LOCAL_ONLY": "1", "PYTHONUNBUFFERED": "1",
                    "PYTHONUTF8": "1"}  # Windows pipes default to cp1252; model text and progress bars are not ASCII

    def say(self, msg):
        print(msg, flush=True)
        self.log.write(msg + "\n")
        self.log.flush()

    def sh(self, *args):
        self.say(f"> python {' '.join(map(str, args))}")
        p = subprocess.Popen([sys.executable, *map(str, args)], cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="", flush=True)
            self.log.write(line)
        if p.wait():
            self.log.flush()
            sys.exit(f"\nFAILED (exit {p.returncode}): python {' '.join(map(str, args))}\n"
                     f"Fix it and rerun the same command; finished steps are skipped. Log: {self.log.name}")

    # ---- state
    def teacher(self):
        """Version of the teacher the replays will use: the complete step-4 cache, else the stub."""
        f = self.cfg["processed"] / "cache" / "teacher_cache.json"
        m = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        return m["teacher"] if m.get("complete") else "stub-v0"

    def traffic_profile_ok(self, extra=False):
        T = self.cfg["processed"] / "traffic"
        metas = ([T / "benign_stress" / "meta.json"] + [T / f"attacks_{s}" / "campaigns.json" for s in ("train", "distributed")]
                 if extra else [T / f"benign_{n}" / "meta.json" for n in ("ref", "eval", "tune")] +
                 [T / f"attacks_{s}" / "campaigns.json" for s in ("static", "stage")])
        return all(f.exists() and json.loads(f.read_text(encoding="utf-8"))["profile"] == self.profile for f in metas)

    def stamp(self, name):
        f = self.dir / f"{name}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def done(self, name, needs_teacher=True):
        s = self.stamp(name)
        if s is None:
            return False
        if name != "traffic" and s.get("traffic") != (self.stamp("traffic") or {}).get("finished"):
            return False  # traffic regenerated since this step ran
        return not needs_teacher or s.get("teacher") == self.teacher()

    def step(self, name, cmds, needs_teacher=True):
        if self.done(name, needs_teacher):
            self.say(f"[{name}] done, skipped")
            return
        if self.plan:
            self.say(f"[{name}] WILL RUN")
            return
        self.say(f"\n[{name}] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        t0 = time.time()
        for c in cmds:
            self.sh(*c)
        (self.dir / f"{name}.json").write_text(json.dumps({
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((time.time() - t0) / 60, 1),
            "teacher": self.teacher(), "traffic": (self.stamp("traffic") or {}).get("finished")}, indent=1),
            encoding="utf-8")


def gpu_ready(R, students):
    """-> list of problems that make results non-final (empty = ready)."""
    P = R.cfg["processed"] / "cache"
    gate = json.loads((P / "gate.json").read_text(encoding="utf-8")) if (P / "gate.json").exists() else {}
    verify = json.loads((P / "verify.json").read_text(encoding="utf-8")) if (P / "verify.json").exists() else {}
    issues = []
    if gate.get("teacher") != "local":
        issues.append(f"local teacher did not pass the gate: {gate.get('results', {}).get('local')}")
    if R.teacher() == "stub-v0":
        issues.append("teacher cache not complete")
    elif not verify.get("passed"):
        issues.append(f"teacher cache verify did not pass: {verify}")
    res = ROOT / "runs" / "scaling" / "results.jsonl"
    rows = [json.loads(x) for x in res.read_text().splitlines() if x.strip()] if res.exists() else []
    S = R.cfg["student"]
    for m in students:
        have = {(r["n"], r["seed"]) for r in rows if r["model"] == m}
        if len(have) < len(S["sizes"]) * len(S["seeds"]):
            issues.append(f"scaling curve for {m}: {len(have)} of {len(S['sizes']) * len(S['seeds'])} cells")
    return issues


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="dev", choices=["dev", "full"])
    ap.add_argument("--students", nargs="+", help="scaling-curve students (default: [student] curve_model)")
    ap.add_argument("--trials", type=int, default=40, help="best-response search trials per defense")
    ap.add_argument("--allow-stub", action="store_true", help="run the CPU stages without the GPU results")
    ap.add_argument("--plan", action="store_true", help="only show which steps are done and which will run")
    a = ap.parse_args()
    R = Runner(a.profile, a.plan)
    students = a.students or [R.cfg["student"]["curve_model"]]
    if R.cfg["student"]["curve_model"] not in students:
        R.say(f"note: fidelity uses [student] curve_model = {R.cfg['student']['curve_model']}; add it to --students")
    R.say(f"\n==== run_all {a.profile} {time.strftime('%Y-%m-%d %H:%M:%S')} teacher={R.teacher()}")
    P = R.cfg["processed"]
    if not a.plan:  # seconds, no data: a broken checkout stops here, before any replay or training
        R.sh("-m", "tests.test_core")

    # data: canary, preprocessed sources, shared prompt pools (+ embeddings, 1,024 buckets)
    if all((P / f).exists() for f in ("canary/items.parquet", "pools/prompts.parquet", "pools/emb.npy",
                                       "pools/buckets.npy")):
        R.say("[data] done, skipped")
    elif a.plan:
        R.say("[data] WILL RUN")
    else:
        if not (P / "normalized_events").exists() and not (ROOT / "Data" / "RAW" / "MANIFEST.csv").exists():
            sys.exit("Data/RAW is missing: copy Data/RAW (or Data/Processed) from the machine that downloaded it.")
        R.sh("-m", "teacher.canary")
        if not (P / "normalized_events").exists():
            R.sh(f"{DATA}/preprocess.py")
        R.sh(f"{DATA}/pools.py")

    # gpu: steps 1, 2, 4, 5 (student.gpu_steps skips finished work itself)
    if a.plan:
        R.say("[gpu] WILL RUN (skips finished work)")
    else:
        R.sh("-m", "student.gpu_steps", "--students", *students)
    issues = gpu_ready(R, students)
    for i in issues:
        R.say(f"[gpu] NOT READY: {i}")
    if issues and not (a.allow_stub or a.plan):
        sys.exit("Stopped before the CPU stages: results would not be final. Fix the above and rerun, or pass "
                 "--allow-stub for a pipeline check with the stub teacher.")

    # traffic: regenerate whenever the folders hold another profile's traffic
    if not R.traffic_profile_ok() and R.stamp("traffic"):
        (R.dir / "traffic.json").unlink()
    R.step("traffic", [(BENIGN, "--name", "ref", "--seed", 11), (BENIGN, "--name", "eval", "--seed", 12),
                       (BENIGN, "--name", "tune", "--seed", 13), ("-m", "attacks.static"),
                       ("-m", "attacks.static", "--set", "stage")], needs_teacher=False)
    R.step("reference", [("-m", "gateway.replay", "reference")])
    R.step("tune", [("-m", "experiments.tune")])
    for name, args in REPLAYS:
        R.step(name, [("-m", "experiments.run", *args)])
    R.step("global", [("-m", "experiments.baselines", "global")])
    for d in ("none", "full"):
        R.step(f"search_{d}", [("-m", "experiments.search", "--defense", d, "--trials", a.trials)])

    # stages 3-7 (CPU replays). New traffic is its own step, so the step-9 traffic stamp and its results stay valid.
    if not R.traffic_profile_ok(extra=True) and R.stamp("traffic_extra"):
        (R.dir / "traffic_extra.json").unlink()
    R.step("traffic_extra", [(BENIGN, "--name", "stress", "--seed", 14, "--stress"),
                             ("-m", "attacks.static", "--set", "train", "--seed", 31), ("-m", "attacks.subset")],
           needs_teacher=False)
    R.step("reference_off", [("-m", "gateway.replay", "reference-off")])
    for name, args in STAGES:
        R.step(name, [("-m", "experiments.run", *args)])
    for d in ("d1", "d2", "d3", "d4"):  # stage 5; d5 = search_full
        R.step(f"stage5_search_{d}", [("-m", "experiments.search", "--defense", d, "--trials", a.trials)])
    R.step("stage6_fit_lr", [("-m", "experiments.fit_lr")])
    R.step("stage6_tune_lr", [("-m", "experiments.tune", "--lr")])
    R.step("stage6_run_lr", [("-m", "experiments.run", "--defense", "full", "--lr")])
    R.step("stage6_run_lr_stage", [("-m", "experiments.run", "--defense", "full", "--lr", *STAGE)])
    R.step("stage6_search_lr", [("-m", "experiments.search", "--defense", "full", "--lr", "--trials", a.trials)])
    R.step("stage7_stress", [("-m", "experiments.run", "--defense", "full", "--streams", "benign_stress")])

    # step 16 / stage 8: real students on the GPU (README §12.3 rules 12-13)
    env = ROOT / "runs" / "env.json"
    if env.exists() and json.loads(env.read_text(encoding="utf-8")).get("cuda") and not issues:
        R.step("step16_validate", [("-m", "student.validate")])
        R.step("stage8_student_guided", [("-m", "experiments.student_guided")])
    else:
        R.say("[step16] skipped: needs a CUDA GPU (runs/env.json) and the GPU results above")
    if not a.plan:  # cheap, always refreshed
        R.sh("-m", "experiments.baselines", "table")
        R.sh("-m", "experiments.figures")
    R.say(f"==== {'plan only' if a.plan else 'finished'} {time.strftime('%Y-%m-%d %H:%M:%S')}"
          + ("  (STUB TEACHER: not final)" if R.teacher() == "stub-v0" and not a.plan else ""))


if __name__ == "__main__":
    main()
