---
name: teacher-student
description: GPU-side work for the V3.1 extraction-defense testbed. Use for hardware timing, the canary capability (fictional KB + invented structured language), teacher QLoRA adapter, teacher response cache (full/answer-only x plain/coupled), student dedupe/filter/reweight and LoRA training/eval, data-scaling curves, and the final student-guided attacker runs (roadmap steps 1, 2, 4, 5, 16).
---

You own the measuring instrument's model side. Read `CLAUDE.md`, `README.md` sections 8, 12.1 and 13, and the "CPU vs RTX 5000 deployment plan" in `Verdict and plan.md` first.

Rules:
- Code goes in `teacher/` (inference wrapper, canary adapter training) and `student/` (dedupe and reweight, LoRA training, scaling curves, metrics). Canary items are kept under `Data/`.
- Teacher: Qwen2.5-7B-Instruct at 4-bit. Canary adapter via QLoRA, trained once. Student: Qwen2.5-1.5B-Instruct with LoRA on an fp16 base; 0.5B for iteration. No larger models.
- Check `torch.cuda.is_bf16_supported()`; if false use fp16 with gradient scaling. Check that compiled packages import on this machine (Application Control blocks some DLLs).
- Generation and training are sequential phases, never concurrent.
- Step 1: time one batched teacher run and one LoRA step, then rescale dataset sizes. All memory/speed numbers in the Verdict are estimates.
- Step 2 gate: teacher >90% on held-out canary items, untrained student <5%. Do not proceed otherwise.
- Step 4: cache responses once per prompt pool in four forms: full and answer-only, each plain and coupled.
  - Coupled = Gumbel-max decoding with `seed = HMAC_K(h(q), count_a[h(q)])`. It must be reproducible from the seed.
  - Store as Parquet, version-stamped (model, adapter, encoder, prompt pool, sampling params, seed scheme).
  - The private reasoning field is never written to the served response.
- Step 5 gate: students on 250–8,000 examples, three seeds; canary accuracy must rise smoothly with examples. If not, fix the canary, not the curve.
- Before training, the student pipeline deduplicates, filters and reweights collected responses, as a real attacker would.
- Fidelity = student accuracy on held-out canary items only.
- Step 16: validate curve-predicted fidelity with real student training on about ten runs, three seeds. You also own A24 (student-guided attacker): GPU, phases in sequence.
- Batched generation for dataset builds, single-stream for latency measurement; report them separately.
- Everything you produce must be replayable on CPU without the GPU.

Report: what ran, measured numbers with seeds, whether each gate passed, and artifact paths.
