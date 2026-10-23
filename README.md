# LLM Extraction Defense Gateway

A provider-side gateway that sits between API users and a language model and makes unauthorized black-box distillation more expensive, without giving anyone a wrong answer.

**Status:** architecture V3.1; CPU pipeline implemented at the dev scale (2026-10-12), GPU steps pending. Progress is tracked in section 15.1.

> **Scope boundary.** This README is the architecture of record. Build only the components in sections 5 and 6, wired as the diagram shows. Anything else is out of scope until it is added here first, with the owner's approval. Where `Verdict and plan.md` (V3) disagrees with this file, this file wins.

---

## 1. The problem

A hosted LLM can be copied through its own API:

```text
attacker -> many queries -> teacher responses -> training set -> student model
```

No logits are needed. Prompt and response pairs are enough to fine-tune a student.

A single extraction query looks like any normal query. The signal is in the pattern across many requests, many accounts and long periods. Public disclosures describe campaigns spread over thousands of accounts, with only a handful of requests per account at times. A defense that inspects requests one by one, or accounts one by one, misses this.

## 2. What the gateway does

1. **Enforces hard rules** on every request: quotas, caps on repeated sampling, binding of protected artifacts to their owner.
2. **Tracks each account** over long windows and scores how much new, useful output it has collected.
3. **Links accounts into clusters** when they behave like one coordinated campaign.
4. **Couples sampling randomness across accounts**, so many accounts asking equivalent questions gain little extra training data.
5. **Responds in tiers**: monitor, shape the output, ask for verification, suspend.

It never returns deliberately incorrect answers.

## 3. Research question

> How much query budget does a gateway-level defense leave a multi-account distiller who knows the defense and adapts to it, at a fixed cost to legitimate users?

Hypotheses, each falsifiable:

| ID | Hypothesis |
| --- | --- |
| H1 | Cluster-level state detects distributed extraction better than account-level state at the same false-positive rate. |
| H2 | Response tiers reduce the student's fidelity at a fixed attacker budget. |
| H3 | The full gateway raises the attacker's cost to reach a target fidelity by at least 2x under the attacker's best response. |
| H4 | Fewer than 1% of benign accounts ever leave the monitor tier. |
| H5 | Sharding detection and coupled sampling cover each other's evasion. |

A null result on any of these is acceptable and will be reported.

## 4. Threat model

| Item | Assumption |
| --- | --- |
| Target | Text-only API over a teacher model that holds a private capability. No log-probabilities exposed. |
| Deployment | Provider-side. The gateway is the only path to the model and can set the sampling seed. |
| Attacker goal | A student that reproduces the private capability. |
| Attacker means | Many accounts (each has a price), prompt generation, paraphrase, translation, timing control, cover traffic, infrastructure rotation, student feedback. |
| Attacker knowledge | The features, the policy and approximate thresholds. Not the benign reference or exact thresholds. |
| Attacker budget | Queries, distinct prompts, accounts, days, GPU hours. Local post-processing is free. |
| Out of scope | Weight theft, insider access, side channels, training-data extraction, offline rewriting of collected responses. |

## 5. Architecture

```text
EVALUATION HARNESS (offline, not part of the defense)
  Traffic replay  <---- best-response search ----  Student and cost model
  (benign logs +                                   (dedupe, filter, reweight,
   adaptive attacker)                               train, measure fidelity)
        |                                                ^
     requests                                        responses
        v                                                |
HOT PATH (per request)
  Gateway + L0 rules -> Embed + bucket -> Tier lookup -> Teacher LLM -> Output guard
  (quotas, repeat caps,  (one small       (cached account (tier shapes   (holds protected-
   artifact binding)      encoder, seed    + cluster tier) output; seed   field leaks only)
                          derivation)                      from bucket)
                                               ^                              |
                                     tiers, one request late               events
                                               |                              v
ASYNC PATH (batched, off the request)
  Benign reference -> Risk scoring <- Features <- Event log
  (quantiles, prompt   (account        (capability    (append-only,
   samples per skill)   envelope,       tag, novelty,  version-stamped)
                        cluster         duplication,
                        linkage)        timing)
```

### 5.1 Hot path

Runs on every request. No classifier, one small encoder.

1. Authenticate, normalize, stamp versions.
2. Apply L0 rules: quotas, parameter caps, repeat cap, artifact binding, denylist.
3. Embed the prompt and map it to a semantic bucket.
4. Look up the tier: the higher of the account tier and the cluster tier.
5. Derive the sampling seed from the bucket and the account's visit count for that bucket.
6. Generate, with the tier applied to the generation parameters.
7. Run the output guard on the stream.
8. Append the event to the log.

Added latency target: 10 to 20 ms on CPU, to be measured.

### 5.2 Async path

Runs every few minutes over new events.

1. Tag capability and hash the prompt skeleton (embeddings are reused from the hot path).
2. Update account windows.
3. Score each account as a benign p-value.
4. Propose account pairs (blocking), score them, take connected components.
5. Run set-level tests per component to get a cluster p-value.
6. Map p-values to tiers with hysteresis and write the tier table.

A decision reaches the hot path at most one request late. Extraction takes thousands of requests, so the lag costs nothing.

### 5.3 Evaluation harness

Not part of the defense. It replays benign traffic beside a simulated attacker, collects what the attacker obtained, trains a student on it and reports fidelity and cost.

## 6. Components

| Component | Role | Implementation |
| --- | --- | --- |
| Gateway | Single entry point; OpenAI-compatible endpoint | Python function in the replay harness; FastAPI wrapper for the demo |
| L0 rules | Deterministic controls | Plain code |
| Encoder | Prompt embedding for bucketing and features | Small multilingual model, 384 dimensions, CPU |
| Tier cache | Account and cluster tiers | In-process dict; SQLite for persistence |
| Teacher | The protected model | Backend switch (`teacher/backends.py`): local Qwen2.5-7B-Instruct, 4-bit, with a LoRA adapter for the private capability (primary); an OpenAI-compatible API model (Groq, OpenRouter, Gemini) holding the canary in-context; a deterministic stub for CPU replays until the response cache exists |
| Output guard | Stops leaks of the private field | Canary string check on the stream |
| Event log | Record of every request and response | Append-only Parquet, queried with DuckDB |
| Capability tagger | Labels each prompt's skill area | Nearest-centroid or logistic regression on embeddings |
| Account scorer | Per-account risk | Benign-quantile rules, MMD test, logistic regression |
| Cluster detector | Cross-account risk | Blocking, pair score, connected components, set-level tests |
| Benign reference | Source of every threshold | Versioned per encoder and per capability |
| Attacker simulator | Adaptive multi-account attacker | Parameterized strategy with Bayesian search |
| Student evaluator | Measures what was stolen | Qwen2.5-1.5B-Instruct with LoRA |

Deliberately not used: vector database, Redis, message queue, graph database, graph neural network, neural classifier.

## 7. Features

| Level | Features |
| --- | --- |
| Event | Embedding, capability tag, skeleton hash, token counts, request parameters, completion flag, model version |
| Account window (last 50 requests, last 500, 90 days, lifetime) | Novel output volume per capability and in total, self-duplication rate, coverage uniformity, interaction structure, completion share, account age against volume, timing summaries |
| Cluster | Pooled novel output volume, cross-account duplication against distribution similarity, activity synchrony, model-switch synchrony, shared rare attributes |

Raw embeddings and ground-truth labels are never model inputs.

Design rule: build features on what the attacker cannot give up without losing yield, namely output volume in the target capability, a diverse prompt set, complete responses and fresh accounts.

## 8. Core equations

**Novel output volume** for account `a` and capability `c`, where `e_i` is the embedding of prompt `i`:

```text
NOV_c(a) = sum_i  tokens_out(i) * (1 - max_{j<i} cos(e_i, e_j))
```

**Benign p-value** of any score `s`, used for every threshold:

```text
p(a) = (1 + #{benign b : s_b >= s_a}) / (1 + N_benign)
```

**MMD test** between an account's prompts `X` and benign prompts `Y` of the same capability, with RBF kernel `k`:

```text
MMD^2(X, Y) = mean k(x, x') + mean k(y, y') - 2 * mean k(x, y)
```

**Account score**, combining `m` tests with a Bonferroni correction:

```text
p_account = min(1, m * min_k p_k)
```

**Pair score** for accounts `a` and `b`, where `deg(t)` is the number of accounts sharing attribute `t`:

```text
S(a, b) = sum_{shared t} log(N / deg(t)) + alpha * sim_generator(a, b) + beta * sync(a, b)
```

**Duplication test** for a candidate cluster `C`: compare the observed share of near-duplicate prompts across its accounts with the share expected from benign groups of the same distribution similarity. A coordinated campaign partitions a prompt list and shows low duplication. A benign crowd repeats the same popular prompts and shows high duplication.

**Coupled sampling**: the seed depends on the semantic bucket `h(q)` and how many times account `a` has hit that bucket.

```text
seed    = HMAC_K( h(q), count_a[h(q)] )
token_t = argmax_i ( logit_i + Gumbel_i(seed, t) )
```

One user still receives normal, independent samples. Many accounts sending equivalent prompts receive correlated outputs.

`h(q)` is the nearest of 1,024 spherical k-means centroids fitted on the pool embeddings. `K = HMAC(master secret, model version)`: a new teacher version gets a new key, and the key does not rotate within an experiment. With an API teacher the seed is passed to the provider, which treats it as best effort, so coupling is approximate there.

**Effective query budget** and **defense cost multiplier**:

```text
Q_eff  = unique full-quality responses the attacker obtained in the target capability
DCM_t  = C_t(defended) / C_t(undefended)
```

`C_t` is the lowest cost at which the attacker's best strategy reaches fidelity `t`.

## 9. Response policy

| Tier | Entered when | Action | Left when |
| --- | --- | --- | --- |
| 0 Normal | Default | None | Not applicable |
| 1 Monitor | p-value below 0.01 | Scoring runs more often; nothing visible | Clean windows pass |
| 2 Shape | p-value below 0.001 | Capped length, answer-only output, no multi-sample, novel-output quota, throttling | Clean windows pass, or verification succeeds |
| 3 Verify | p-value below 0.0001, or tier 2 sustained | Requests held until step-up verification | Verification succeeds |
| 4 Suspend | Tier 3 evidence from two independent signal families, or failed verification | Account suspended; cluster signature stored | Manual review |

Thresholds are starting values, tuned until benign cost meets its cap. An account inherits a cluster's tier only if its own traffic matches the cluster and it has at least one non-semantic link. "Clean windows pass" means three consecutive clean async windows, and each pass drops one tier. The two independent signal families for tier 4 are the account p-value and the cluster p-value, both below 0.0001. A benign account that passes verification is capped at tier 1 for 30 days (provisional).

Not used: poisoned or deliberately wrong answers.

## 10. Failure handling

| Failure | Behavior |
| --- | --- |
| Tier cache unavailable | Unknown and low-tier accounts continue under tight quotas. Accounts on the persisted denylist stay blocked. |
| Encoder down | Requests continue with an independent random seed. Scoring pauses. Tiers freeze. |
| Scorer error | Fall back to quantile rules on novel output volume. |
| Event log write failure | Buffer locally and mark the window incomplete. No escalation from incomplete windows. |
| Linkage job failure | Account tiers continue. Cluster tiers freeze. |
| Stale cluster state | Tiers 1 and 2 expire. Tiers 3 and 4 persist until reviewed. |
| Output guard failure | Nothing leaks: the private field is never serialized. |

Principle: fail open for unknown and low-risk accounts, fail closed for confirmed ones, never escalate on missing data.

## 11. Attack coverage

The test suite (`LLM_Extraction_Attack_Test_Suite.md`) defines 30 attacks, A01 to A30.

| Detection | Count | Attacks |
| --- | --- | --- |
| Strong | 4 | A01 basic direct, A06 multi-sample, A11 cross-model, A12 artifact replay |
| Moderate | 12 | A02, A03, A04, A07, A08, A15, A16, A18, A19, A20, A21, A28 |
| Weak | 10 | A05, A09, A10, A17, A22, A23, A24, A27, A29, A30 |
| None | 4 | A13, A14, A25, A26 |

A13, A14 and A26 happen offline after collection, and A25 simply needs fewer queries, so no inference-time gateway can see them.

One attack is added to the suite as **A31**: round-robin distribution across hundreds of accounts with 99% benign traffic mixed in.

## 12. Evaluation

### 12.1 Measuring what was stolen

The teacher is given a private "canary" capability through a LoRA adapter: question answering over a fictional knowledge base, and translation into an invented structured language. An untrained student scores near zero on it, so any gain comes from extraction.

A data-scaling curve (student accuracy against number of training examples) is measured once. Each attack-versus-defense run then only needs the attacker's effective query budget, which maps to fidelity through the curve. Real students are trained on about ten runs to check the prediction.

Before training, the student pipeline deduplicates, filters and reweights the collected responses, because a real attacker would.

### 12.2 Attacker

A strategy with eleven parameters: number of accounts, novel-output budget per account, shard overlap, timing source, cover-traffic ratio, infrastructure reuse, capability mix, paraphrase and translation rate, repeats per prompt, rotation rule, and query selection (random or student-guided).

For each defense configuration, a Bayesian search finds the strategy with the highest yield per unit cost. Results are reported at that best response.

### 12.3 Metrics

| Group | Metrics |
| --- | --- |
| Primary | Effective query budget and distinct prompts left to the attacker; defense cost multiplier at 50% and 80% of teacher fidelity |
| Detection | Account-level and cluster-level TPR at 1% and 0.1% benign-account FPR, with bootstrap intervals; AUPRC at 1% and 0.1% attacker prevalence; share of final yield obtained before first enforcement |
| Benign cost | Share of benign accounts reaching tier 2 (cap 1%) and tier 3 (cap 0.1%); task success under answer-only output |
| System | Added hot-path latency at p50, p95, p99 on CPU; throughput; memory |

Accuracy and AUROC are not reported as headline numbers. Request-level metrics alone are not accepted.

### 12.4 Experiment stages

| Stage | Defense | Attackers | Purpose |
| --- | --- | --- | --- |
| 1 | None; quotas only | Static | Undefended baseline |
| 2 | Per-account MMD test; global-aggregation detector; similarity-based colluder linkage; coupled sampling alone | Static | Published baselines |
| 3 | Rules, then + account envelope, + cluster linkage, + response tiers, + coupled sampling | Static, rising difficulty | Value of each layer |
| 4 | Full, with infrastructure, timing, semantic or duplication signals switched off in turn | Distributed attacks | Which signals carry the result |
| 5 | Each configuration from stage 3 | Best-response search | Guaranteed minimum effect |
| 6 | Trained on four attack families | Unseen families and best-response strategies | Generalization |
| 7 | Full | None; benign stress populations | Harm to legitimate users |
| 8 | Full | Student-guided attacker with real student training | Final check |

### 12.5 Dataset

| Population | Source | Starting size |
| --- | --- | --- |
| Benign light users | Real conversation logs | 10,000 accounts |
| Benign heavy users | Synthesized from shared prompt pools | 500 accounts |
| Benign bulk users (evaluation harnesses, data jobs) | Synthesized | 100 accounts |
| Benign crowds (classes, viral prompts) | Synthesized | 20 groups |
| Attack campaigns | Simulator | 100 campaigns of 1 to 500 accounts |

Rules that keep the data honest:

- Attack and benign prompts come from the same pools. A classifier on single prompts must score near 0.5 AUROC, or the dataset leaks.
- Ground-truth labels live in a separate table, never in the event record.
- Benign users also share infrastructure (NAT, campus, corporate egress).
- Splits are account-disjoint, campaign-disjoint, unseen-attack and time-forward.

Raw sources are pinned in `Data/RAW/MANIFEST.csv` and normalized by `Data/pre_processing pipline/preprocess.py` into `Data/Processed/` (see `Data/Processed/metadata.json` for counts, checks and caveats).

## 13. Hardware

| Workload | CPU-only | GPU (16 GB) |
| --- | --- | --- |
| Teacher bulk generation | Replay cached responses | 4-bit, batched |
| Teacher canary adapter | Download the saved adapter | QLoRA, once |
| Student training and evaluation | Use saved checkpoints and scores | LoRA on an fp16 base |
| Embedding in the request path | CPU | CPU |
| Scoring, linkage, attacker search | CPU | CPU |

The GPU is needed once, to produce teacher responses and student checkpoints. Every detection, policy and cost experiment then replays those artifacts on a CPU.

## 14. Repository layout

Directories are created only when a roadmap step needs them. `data/` from the design is the existing `Data/` folder (Windows paths are case-insensitive, so both names point to the same folder).

```text
Model_KD/
├── README.md
├── pyproject.toml
├── configs/          model, detector, policy, attacker
├── gateway/          entry point, L0 rules, tier lookup, seed derivation, output guard
├── features/         encoder, capability tagger, account windows, cluster features
├── scoring/          benign reference, account scorer, calibration
├── linkage/          blocking, pair score, components, duplication test
├── policy/           tiers, hysteresis, fallback
├── teacher/          inference wrapper, canary adapter training
├── attacks/          static attack scripts, strategy parameters, best-response search
├── student/          dedupe and reweight, LoRA training, scaling curves, metrics
├── Data/             schemas, prompt pools, generated traffic, splits
│   ├── RAW/                     downloaded sources (gitignored)
│   ├── pre_processing pipline/  preprocessing scripts (tracked)
│   └── Processed/               normalized events + metadata (gitignored)
├── experiments/      one script per stage
├── tests/
└── docs/
```

## 15. Roadmap

1. Time one batched teacher run and one LoRA step; fix dataset sizes.
2. Build the canary capability and train the teacher adapter.
3. Build shared prompt pools.
4. Generate and cache teacher responses, full and answer-only, plain and coupled.
5. Measure the data-scaling curves.
6. Build benign traffic, including shared infrastructure and benign duplication rates.
7. Run the leakage probe on single prompts.
8. Build the gateway replay harness and event schema, with seed derivation.
9. Implement static attackers, including round-robin with benign mixing.
10. Implement features and the capability tagger.
11. Implement the account scorer.
12. Implement the baselines.
13. Implement cluster linkage and the duplication test.
14. Implement response tiers and the cost model.
15. Implement the best-response attacker.
16. Run the student-guided attacker and the final stages.

Steps 1 to 12 already answer whether account-level defense changes extraction cost against static attackers.

### 15.1 Progress

| Item | Status |
| --- | --- |
| Raw data download and preprocessing (input to steps 3, 6 and the stage 2 MMD baseline) | Done. Run `.venv/Scripts/python.exe "Data/pre_processing pipline/preprocess.py"` |
| 1 Hardware timing | Script ready (`student/train_lora.py --time-steps`, logs s/step and peak VRAM); on a local 16 GB GPU `python -m student.gpu_steps --only env timing`, else Colab (`student/colab.ipynb`). Teacher tok/s is logged by the local cache build. CPU encoder measured: ~13 ms per single prompt |
| 2 Canary capability + teacher adapter | Canary data done (`teacher/canary.py`). No local GPU, so the teacher is an API model with the canary in context (`python -m teacher.cache gate`, prompt `tpl-v2`, 100 held-out items per capability). Teacher side passed 2026-10-19: Groq `openai/gpt-oss-120b` (reasoning effort low) KB 1.00, Kesh 0.92. Failed: qwen3.8-27b (Kesh 0.62), nemotron-3-super (0.49), gemini-3.5-flash-lite (0.62, closed model, not eligible anyway). Untrained-student side (<5%) still to run. Local route added 2026-10-23 for a 16 GB GPU: Qwen2.5-7B 4-bit + canary QLoRA adapter (`python -m student.gpu_steps`), gated as candidate `local`; if it passes it is the teacher (README §6 primary), API models are the fallback. Not yet run |
| 3 Shared prompt pools | Done: 72,793 prompts, 10 pools, 1,024 buckets (`Data/pre_processing pipline/pools.py`) |
| 4 Teacher response cache | Pipeline ready, not yet built: `python -m teacher.cache build` then `verify` (resumable): 6,000 pool prompts × (4 full + 1 answer-only) candidates; the gateway seed picks one candidate (`CachedTeacher`). Local teacher: batched Gumbel-max on the GPU with a per-row seed, so each candidate regenerates from its seed (`verify` checks it, plus coverage and no canary secret). API teacher: provider seed, best effort. Replays still use the stub teacher |
| 5 Data-scaling curves | Script ready (`student/scaling.py`, sizes 250–8,000 × 3 seeds); local GPU via `student/gpu_steps.py`, or Colab via `student/colab.ipynb` |
| 6 Benign traffic | Done at dev scale (`Data/pre_processing pipline/benign.py`, reference and eval populations) |
| 7 Leakage probe | Run (`experiments/leakage_probe.py`): within canary pools AUROC 0.49; across all prompts 0.94 (capability mix). Gate interpretation open |
| 8 Replay harness, event schema, seed derivation | Done (`gateway/`); deterministic replay verified |
| 9 Static attackers | Done: A01, A02, A06, A15, A16, A18, A19, A20, A31 (`attacks/static.py`) |
| 10 Features + capability tagger | Done (`features/windows.py`); tagger held-out accuracy 0.966 |
| 11 Account scorer | Quantile rules + MMD done (`scoring/account.py`); LR present but off until the unseen-attack comparison |
| 12 Baselines | Stage 1 (none, quotas only) and coupled-sampling-alone runs; MMD-only, global-aggregation and colluder-linkage baselines not started |
| 13 Cluster linkage + duplication test | Done (`linkage/cluster.py`) |
| 14 Response tiers + cost model | Tiers, hysteresis and failure handling done (`policy/tiers.py`, tests); thresholds not yet tuned to the benign caps |
| 15, 16 | Not started |

Run order (dev profile, `KD_PROFILE=full` for README §12.5 sizes): `python -m teacher.canary` → `pools.py` → `benign.py --name ref --seed 11` → `benign.py --name eval --seed 12` → `python -m attacks.static` → `python -m gateway.replay reference` → `python -m experiments.run --defense full`. Checks: `python -m tests.test_core`.

GPU steps 1, 2, 4, 5 on one 16 GB card (resumable, skips finished work): `uv sync --extra gpu`, a CUDA build of torch, then `python -m student.gpu_steps` (env check → canary → timing → 7B QLoRA adapter, OOM fallback 8 → 4×2 → 2×4 → gate → cache build + verify → scaling).

Plain-language progress log: `explain.md`. Paper draft: `document.md`.

## 16. Limitations

- **It raises cost; it does not prevent extraction.** Accounts that each stay inside the benign envelope and cannot be linked are not detected. Total yield is then the per-account cap times the number of accounts.
- **Intent is not observable.** A customer generating synthetic data or running an evaluation harness behaves like an extractor. That is why the response at that level is verification, not a block.
- **Identity signals are simulated.** Infrastructure, payment and signup signals come from the simulator.
- **Small models stand in for a frontier model.** The shape of the results may transfer; the absolute numbers will not.
- **Coupling is emulated from a K-candidate cache.** With the API teacher fallback the canary is also held in context, not in a trained adapter. APIs expose no logits, so the gateway seed picks one of K cached candidates; an independent seed therefore also repeats after K draws of the same prompt.
- **Enforcement leaks information.** Every visible action tells the attacker where a threshold sits.
- **Association can be abused.** An attacker can imitate benign users to pull them into a cluster. The leave-one-out rule reduces this (the leave-one-out guard is not implemented yet; inheritance currently requires a capability-mix match and a non-semantic link).
- **No claim about provider systems.** This project is inspired by public disclosures and does not reproduce how any provider detects extraction.

## 17. References

- Anthropic, [Detecting and preventing distillation attacks](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks), 2026
- OpenAI, [Disrupting a coordinated model-distillation campaign](https://openai.com/index/disrupting-a-coordinated-model-distillation-campaign/), 2026
- Kesarwani et al., [Model Extraction Warning in MLaaS Paradigm](https://arxiv.org/abs/1711.07221), 2018
- Juuti et al., PRADA: Protecting against DNN Model Stealing Attacks, 2019
- Dziedzic et al., [Increasing the Cost of Model Extraction with Calibrated Proof of Work](https://arxiv.org/abs/2201.09243), 2022
- Yao et al., [FDINet](https://arxiv.org/abs/2306.11338), 2023
- Savani et al., [Antidistillation Sampling](https://arxiv.org/abs/2504.13146), 2025
- Zhao et al., [A Survey on Model Extraction Attacks and Defenses for LLMs](https://arxiv.org/abs/2506.22521), 2025
- Diao et al., [Lossless Anti-Distillation Sampling](https://arxiv.org/abs/2605.18829), 2026
- Allouah et al., [The Distillation Game](https://arxiv.org/abs/2605.22737), 2026
- Schwarzer et al., [AI Model Extraction Attacks: Bypassing Single-Client Assumptions in Defenses](https://arxiv.org/abs/2606.03381), 2026
- Liu, Guo and Dong, [An Embarrassingly Simple Detector for Model Extraction Attacks in LLM API Traffic](https://arxiv.org/abs/2606.05725), 2026
- Libon et al., [What Does It Mean to Break a Distillation Defense?](https://arxiv.org/abs/2606.25059), 2026
- Panfilov et al., [Stealing Reasoning Traces from Proprietary LLM APIs](https://arxiv.org/abs/2608.09867), 2026
- Liu et al., [MEA-Bench](https://arxiv.org/abs/2610.00839), 2026
