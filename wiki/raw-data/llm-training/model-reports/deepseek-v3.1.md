<!-- scope: DeepSeek V3.1 / V3.2 — sparse attention + extended post-training over V3
     deps: [[deepseek-v3]], [[deepseek-r1]]
     see-also: [[deepseekmath]], [[grpo]]
-->

# DeepSeek V3.1 / V3.2
- **Core Insight:** A lightweight "lightning indexer" + token-selector on top of V3.1-Terminus adds sparse attention to an already-trained MoE without breaking quality — and makes a >10%-of-pretrain RL budget feasible.
- **Guideline:** When post-training compute dominates, invest in attention sparsity first (top-k=2048) so long-context RL rollouts become cheap.

- **Authors / Lab:** DeepSeek-AI
- **Year:** Aug 2025 (V3.1), Sept 2025 (V3.2-Exp), Dec 2025 (V3.2 paper)
- **URL:** https://arxiv.org/abs/2512.02556 (V3.2) — https://api-docs.deepseek.com/news/news250929
- **Relevant topics:** DeepSeek Sparse Attention (DSA), lightning indexer, top-k token selection, specialist distillation, mixed RL training

## Abstract
V3.1 (Aug 2025) merged V3 and R1 into a single hybrid model (671B total / 37B active) supporting long-context and reasoning in one checkpoint. V3.2-Exp (Sept 2025) introduced **DeepSeek Sparse Attention (DSA)** — a lightning indexer + fine-grained token selector that selects top-k=2048 positions per query — preserving V3.1-Terminus performance while dropping attention cost in long context. V3.2 (Dec 2025 paper) scales DSA + "mixed RL training" with a post-training compute budget exceeding 10% of pretraining, claiming GPT-5-comparable performance.

## Key Contributions
- **DSA (DeepSeek Sparse Attention):** two-component design — (1) lightning indexer computes cheap relevance scores, (2) token-selector retains top-k=2048 positions, masks the rest. Drop-in on top of V3.1-Terminus.
- **Specialist distillation:** domain-expert models trained per-domain, then distilled back into the unified model in the post-training stage.
- **Mixed RL training:** single-pass RL training that jointly optimizes multiple rewards/domains rather than sequential RL stages.
- Post-training compute >10% of pretraining — a substantive shift in compute allocation vs V3 (where post-training was ~0.2% of pretraining GPU-hours).
- V3.2 claims near-parity with V3.1-Terminus on reasoning/coding benchmarks despite sparse attention.

## Post-training pipeline
- **SFT data:** Not explicitly sized in V3.2 paper abstract; pipeline inherits V3 template (reasoning + non-reasoning data, R1 distillation) plus specialist-model outputs.
- **Preference / RL algorithm:** GRPO-family (V3 used GRPO with rule-based + model-based rewards). V3.2 refers to "robust reinforcement learning protocol" — specifics reserved.
- **Reward model:** Hybrid rule-based + model-based, consistent with V3. "GenRM" specialist per domain implied but not detailed.
- **KL / entropy handling:** Not disclosed in the V3.2 technical report abstract.
- **Rollout scale:** Not disclosed. Sparse-attention rollouts are where the efficiency story lands — top-k=2048 keeps compute bounded regardless of sequence length.
- **Hyperparameters:** Top-k=2048 (sparse-attention hyperparameter). RL LR, batch, group size G, clip ε — not disclosed.
- **Verifiable rewards:** Inherited — math verification, code execution (per V3 + R1 template).
- **Self-improvement / iterative:** Specialist → generalist distillation loop is the new iterative structure; V3 did this only via R1 distillation.

## Innovations vs predecessors
Changes from **V3 → V3.1 → V3.2**:
- V3.1 merged V3 (non-thinking) and R1 (thinking) into one hybrid checkpoint.
- V3.2-Exp added DSA on top of V3.1-Terminus — sparse continued pretraining, then same post-training with DSA.
- V3.2 finalizes DSA and scales post-training compute to >10% of pretraining (V3: ~5K H800-hrs post-training out of 2.788M total, so ~0.2%).
- "Mixed RL training" replaces V3's sequential SFT → RL pipeline.
- Specialist distillation — new; V3 only distilled from R1, not domain specialists.
- Agentic-task synthesis pipeline added for training data (explicitly mentioned in V3.2).

## Key Figures/Tables to Study
- DSA architecture diagram (lightning indexer + token selector) — the main architectural contribution.
- V3.2 vs V3.1-Terminus benchmark table — demonstrates sparsity doesn't cost reasoning quality.
- Post-training compute-budget chart — shows the shift from <1% to >10% of pretraining.

## Connections
- [[deepseek-v3]] — base architecture and original post-training template.
- [[deepseek-r1]] — source of reasoning-distillation data, merged into V3.1's hybrid checkpoint.
- [[deepseekmath]] — origin of GRPO, still the presumed RL backbone.

## Gaps / what the report does NOT disclose
V3.2 abstract gives the what (DSA, mixed RL, >10% post-train compute) without the how. Not disclosed: SFT data size, exact RL algorithm variant, reward model composition, KL β, LR, batch size, clip ε, group size G, rollouts per prompt, RL step count, exact domains for specialist distillation, lightning-indexer architecture beyond "lightweight," distillation loss formula. No ablation separating DSA's contribution from scaled post-training compute.
