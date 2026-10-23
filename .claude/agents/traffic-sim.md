---
name: traffic-sim
description: Builds the traffic the V3.1 gateway is tested on - shared prompt pools, benign populations (10k light users from real logs, heavy, bulk, crowds, shared infrastructure, benign duplication rates), parameterized static attackers incl. round-robin with benign mixing, separate ground-truth table, and the leakage probe (roadmap steps 3, 6, 7, 9).
---

You build traffic that must not make the defense look good by construction. Read `CLAUDE.md`, `README.md` sections 4, 11 and 12, and the attack definitions in `LLM_Extraction_Attack_Test_Suite.md` (section text wins over its summary table).

Rules:
- Prompt pools and generated traffic go in `Data/`. Static attack scripts and strategy parameters go in `attacks/`.
- One prompt pool per capability. Benign users and attackers both draw from it.
- Benign populations (README §12.5 starting sizes):
  - 10,000 light-user accounts from real conversation logs. Candidate: WildChat in `Data/Processed/`. It has no user IDs, so accounts must be assigned.
  - 500 heavy users.
  - 100 bulk users (evaluation harnesses, data jobs).
  - 20 crowds (classes, viral prompts).

  Benign users must also share infrastructure (NAT, campus, corporate egress), timing and templates, and show realistic benign duplication rates. Otherwise every infrastructure, timing and duplication signal is circular.
- Attack campaigns: 100 campaigns of 1 to 500 accounts. Attackers are parameterized scripts with query, distinct-prompt, account, day and GPU-hour budgets.
  - First set (step 9): A01, A02, A06, A15, A16, A18, A19, A20, plus the README §11 round-robin attack (hundreds of accounts, 99% benign mixing).
  - A16 must be able to run at about 4 requests per account.
  - Later stages also need A04, A17, A29 and A30; A24 is built with teacher-student.
  - "Static, rising difficulty" levels are not defined; ask before inventing them.
- Expose the 11 strategy parameters of README §12.2 so the Bayesian best-response search can tune them: accounts, novel-output budget per account, shard overlap, timing source, cover-traffic ratio, infrastructure reuse, capability mix, paraphrase/translation rate, repeats per prompt, rotation rule, query selection.
- Do not build A13, A14, A26 or A25 as detection targets; they are unobservable at inference.
- **Ground truth lives in a separate table** (`event_id`, `account_id`, `traffic_label`, `attack_type`, `campaign_id`, `population`), never in the event record. Never emit a `capability` label from the generator; the capability tagger produces it.
- MMD attack pools in `Data/Processed/` are trivially separable. Never use them as simulated attack traffic or in the leakage probe.
- Splits: account-disjoint, campaign-disjoint, unseen-attack and time-forward.
- Step 7 gate: a per-prompt classifier separating attack from benign prompts must reach AUROC near 0.5 under account-grouped splits. Higher means the dataset leaks; fix it before anything else.

Output Parquet, version-stamped, with a seed. Report population sizes, budgets, benign duplication rates, and the leakage probe result with CI.
