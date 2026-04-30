<!-- scope: Allen AI's Tulu 3 blog post — accessible summary of the three-stage open post-training recipe
     deps: [[README]]
     see-also: [[tulu-3]], [[olmo-2]]
-->

# Allen AI — "Tülu 3: The next era in open post-training"
- **Core Insight:** Fully open post-training — data + code + weights + evals — has reached parity with closed-weight instruct models when SFT + DPO + RLVR are stacked in order on a strong base.
- **Guideline:** For open-source post-training reproduction, follow the Tulu 3 recipe as literally as possible; the recipe's reproducibility is the contribution.
- **Author:** Nathan Lambert, Valentina Pyatkin, and the Allen AI Tulu team
- **Year:** 2024 (November)
- **URL:** https://allenai.org/blog/tulu-3-technical
- **Relevant topics:** open post-training, SFT mixture, DPO, RLVR, reward vs verifier, fully-open release ethos

## Summary
The Tulu 3 blog post is the short-form accessible summary of the Tulu 3 technical report. It introduces the three-stage recipe (SFT -> DPO -> RLVR), highlights the RLVR innovation (PPO against task verifiers instead of a learned RM), and argues that the fully-open release of data, training code, eval tools, and models is the real contribution because it lets the community iterate. Benchmarks show Tulu 3 8B/70B/405B matching or exceeding Llama 3.1 Instruct at equal scale.

## Key Contributions
- Plain-language framing of RLVR for a wider audience than the arXiv paper.
- Links to all released artifacts: model weights, open-instruct codebase, data mixtures, eval scripts.
- Clear comparison against Llama 3.1 Instruct, Qwen 2.5 Instruct, Nous Hermes.
- Articulates the "fully open" philosophy that distinguishes Tulu 3 from open-weights-but-closed-data releases.

## Key Figures/Tables to Study
- **Three-stage pipeline diagram** with per-stage data count.
- **Benchmark table:** Tulu 3 vs closed-weight instruct baselines across safety, math, code, IFEval, MMLU.
- **RLVR schematic:** policy -> sample -> verifier -> reward -> PPO update.
- **Data mix bar chart:** per-source SFT prompt counts (939K total: 57% public, 43% synthetic).

## Technical Details

### The three stages (blog-level framing)
1. **SFT** on ~939K prompts (mix of WildChat, OpenAssistant, and synthetic in-house data).
2. **DPO** on on-policy preference pairs ranked by a reward model or human labels; length-normalized DPO for 8B; beta tuned per size.
3. **RLVR** via PPO with binary verifier rewards:
   - GSM8K / MATH: symbolic equivalence check.
   - IFEval: programmatic constraint satisfaction check.
   - Code tasks: unit test execution.

### Why RLVR is the innovation
- No reward model is trained — so no reward model can be overoptimized (Goodhart-proof).
- Verifiers are cheap to ship and audit.
- Each verifier is a domain expert; composing them covers reasoning benchmarks better than a single RM.
- Gains measured: +5–10pp on GSM8K, +~4pp on IFEval vs DPO-only baseline.

### The "fully open" ethos
The blog contrasts with partial-open releases (weights only; weights + arch paper; weights + some data). Tulu 3 releases:
- Model weights (8B, 70B, 405B — the first fully-open 405B instruct model).
- All data mixtures (SFT + preference + RLVR prompts).
- Training code (open-instruct).
- Evaluation harness.
- Verifier implementations.

### Hyperparameter highlights (from blog + companion paper)
- **SFT:** 2 epochs, completion-masked loss.
- **DPO:** beta 5.0 (length-normalized), LR 5e-7 (8B).
- **RLVR PPO:** LR 3e-7, beta KL 0.05, clip eps 0.2, 10M episodes.

### Community impact
Within months of release, Tulu 3 became the standard comparison baseline for open instruct models, and RLVR was adopted in OLMo 2, several Chinese open replications, and is the conceptual ancestor of DeepSeek-R1's rule-reward RL.

## Connections
- [[tulu-3]] — the underlying technical report.
- [[olmo-2]] — applies the Tulu 3 recipe unchanged to OLMo 2 base.
- [[rlvr-tulu3]] — deeper methodology page on RLVR.
- [[llama-3]] — contrast: Llama 3's DPO-only approach.
- [[nathan-lambert-rl-overview]] — Lambert's wider framing of RLVR in the RL-for-LLMs landscape.
- [[allen-ai]] — lab page.
