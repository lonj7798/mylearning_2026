<!-- scope: controlled synthetic-reasoning study separating the causal contributions of pre-training, mid-training, and RL
     deps: [[front-loading-reasoning]], [[echo-chamber-rl-post-training]], [[rlvr-beyond-base-model]]
     see-also: [[deepseek-r1]], [[prorl]], [[math-shepherd]], [[lets-verify]], [[grpo]]
-->

# On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models
- **Core Insight:** In a fully controlled setup with 100M-parameter Qwen2.5-architecture models pre-trained from scratch on 10B tokens of synthetic graph-reasoning data, GRPO raises pass@128 only on tasks just beyond the pre-training range (up to +42% pass@128), while on in-distribution tasks it raises pass@1 with no pass@128 gain (§3, Figure 1 and Figure 3).
- **Guideline:** When allocating a fixed compute budget between mid-training and RL for reasoning, run mid-training first and target RL at the operation range where the base model still has non-zero but low pass@128, because in this paper mid-training plus RL beats RL alone by +10.8% on OOD-hard tasks under an equal token-equivalent budget (§5, Figure 1 right). When the target tasks are already at near-saturated pass@128, RL adds pass@1 only.
- **Authors:** Charlie Zhang, Graham Neubig, Xiang Yue
- **Year:** 2025 (arXiv v1 2025-12-08)
- **URL:** https://arxiv.org/abs/2512.07783
- **Source type:** paper
- **Relevant topics:** mid-training, RL post-training, GRPO, extrapolative generalization, contextual generalization, process rewards, reward hacking, pass@k

## Abstract
The paper asks whether RL post-training extends a model's reasoning ability beyond what pre-training supplies, and attributes the conflicting literature to uncontrolled training pipelines. It builds a controlled framework using synthetic reasoning tasks with explicit atomic operations, parseable step-by-step traces, and directly manipulated training distributions. Models are evaluated along two axes: extrapolative generalization to more complex compositions, and contextual generalization across surface contexts. The four reported results are: RL produces pass@128 gains only when pre-training leaves headroom and RL data sit at the model's edge of competence; contextual generalization requires minimal but sufficient pre-training exposure, after which RL transfers; mid-training improves performance under fixed compute relative to RL alone; and process-level rewards reduce reward hacking and improve reasoning fidelity.

## Key Contributions
- A controlled data generator built on the GSM-Infinite framework, producing problems with a DAG dependency structure, parseable solutions, and a controllable number of atomic operations (§2.1).
- A distinction between **extrapolative generalization** (composing more operations than seen) and **contextual generalization** (same reasoning under a different surface context) (§2.2).
- A process-verified evaluation protocol that scores each gold node's value and dependencies, not only the final answer (§2.3).
- A compute-normalization formula that converts RL sample count into token-equivalent cost so mid-training and RL can be compared at equal budget (§5, Eq. 1–2).
- A composite reward combining outcome reward and process-verification reward, with a mixing weight α (§6).

## Key Figures/Tables to Study
- **Figure 3** — pass@k on ID (op=2-10), OOD-edge (op=11-14), and OOD-hard (op=15-20) after RL on each difficulty range.
- **Figure 4** — pass@128 on context B as a function of pre-training exposure to context B.
- **Figure 6** — pass@1 and pass@128 under the five mid-training/RL budget splits.
- **Figure 7 / Figure 8** — pass@k under different reward compositions, and the shift in structural error types.
- **Table 1** — architecture configuration of the 100M-parameter Qwen2.5 model.

## Technical Details

### Models and data
- Decoder-only Qwen2.5-architecture models with 100M parameters, trained from scratch (§2.3 "Experimental setup"; App. A.3, Table 1).
- Full synthetic corpus is 30B tokens; each experiment pre-trains on 10B tokens, a 100× token-to-parameter ratio (§2.3; App. A.3.3).
- Tokenizer trained on the synthetic corpus; vocabulary 2,200 tokens including special tokens; maximum sequence length 2,048 (App. A.3.2).
- Difficulty ladder used throughout: ID = op=2-10 (target near-saturated pass@128), OOD-edge = op=11-14 (target non-zero pass@128), OOD-hard = op=15-20 (target zero pass@128 in the base model) (§3; App. A.3.4, Figure 9).

### Result 1 — when RL adds capability (§3)
- Base model pre-trained on 10B tokens of ID (op=2-10) problems; GRPO applied to 200K samples drawn from four difficulty ranges (§3).
- On ID tasks, RL improves pass@1 but not pass@128 under any RL data regime, which the authors read as sharpening existing capability (§3, Figure 3).
- RL raises pass@128 when applied to edge-of-competence data (op=11-14); Figure 1 states up to +42% pass@128 when the RL data are well calibrated.
- The stated selection rule is to train on tasks where the model fails at pass@1 but succeeds at pass@k, avoiding both high-pass@1 tasks and zero-pass@k tasks (§3).

### Result 2 — pre-training exposure and contextual transfer (§4)
- Pre-training mixes op=2-20 context A with long-tailed op=2 context B examples at varying ratios, over 10B tokens; RL then uses 200K samples of 50% context A and 50% context B spanning op=2-20 (§4).
- At 0% or 0.1% context-B exposure, RL fails to transfer to context B. At 1% exposure, post-training generalization improves up to op=20 (§4, Observation 2, Figure 4). Figure 1 reports up to +60% pass@128 in this setting.

### Result 3 — mid-training versus RL at equal compute (§5)
- RL token-equivalent cost is approximated as T_RL ≈ (5/3) · N · r · L_total, where N is the number of RL samples, r = 6 the rollout multiplicity, and L_total = 2048 the total token length (§5, Eq. 1).
- The RL allocation ratio β splits the total budget: T_mid = (1−β)·T and T_RL = β·T (§5, Eq. 2).
- Five configurations from the same base model (pre-trained on 10B op=2-10 tokens): full mid-training on 1B supervised tokens from op=11-14; full RL with 100 steps at batch size 1024 from the same range; and Light-RL (β=0.2), Medium-RL (β=0.5), Heavy-RL (β=0.8) (§5).
- On OOD-edge, light RL gives the best pass@1; on OOD-hard, shifting budget toward heavy RL improves both pass@1 and pass@128 (§5, Observation 3, Figure 6). Figure 1 states mid-training + RL exceeds RL alone by +10.8% on OOD-hard.

### Result 4 — process rewards (§6)
- Composite reward mixes an outcome reward with a process-verification reward R_pv defined by the step-level accuracy criteria of App. A.2, with α ∈ [0,1] controlling the balance; a stricter variant grants the outcome reward only when the whole reasoning process verifies (§6).
- Post-training on op=11-14 with different reward compositions: adding process verification improves pass@1 by 4–5% across extrapolative tasks (§6, Figure 7).
- Figure 8 reports a shift away from shortcut solutions and a change in the distribution of structural error types (§6).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Paper's Qwen2.5-architecture model (not released) | 100M | pretrain | tokens seen / context / batch | 10B tokens (1 epoch) / 2048 / 512K tokens | arXiv:2512.07783 App. A.3.3 | verified 2026-09-18 | App. A.3.4 Figure 9: ladder chosen so ID pass@128 saturates, OOD-edge non-zero, OOD-hard zero |
| same | 100M | pretrain | peak LR / min LR / warmup / weight decay / precision | 2 × 10⁻⁴ / 3 × 10⁻⁵ cosine / 5% / 0.1 / bf16 | arXiv:2512.07783 App. A.3.3 | verified 2026-09-18 | no ablation reported |
| same | 100M | mid-train | supervised tokens / batch / LR / min LR / warmup | 1B tokens (op=11-14) / 512K tokens / 1 × 10⁻⁴ / 3 × 10⁻⁵ cosine / 15% | arXiv:2512.07783 §5, App. A.3.3 | verified 2026-09-18 | §5 Figure 6: budget sweep over β ∈ {0, 0.2, 0.5, 0.8, 1} |
| same | 100M | RL | algorithm / global batch / epochs / prompt+response length | GRPO / 1,024 examples / 2 / 1024 + 1024 tokens | arXiv:2512.07783 App. A.3.3 | verified 2026-09-18 | no ablation reported |
| same | 100M | RL | actor LR / PPO mini-batch / micro-batch / KL coef / entropy bonus | 1 × 10⁻⁶ / 256 / 16 per GPU / 10⁻³ low-variance KL / 0 | arXiv:2512.07783 App. A.3.3 | verified 2026-09-18 | no ablation reported |
| same | 100M | RL | rollout temperature / rollout multiplicity | T_RL = 1.0, top-p 1.0, no top-k / r = 6 | arXiv:2512.07783 App. A.3.3, §5 Eq. 1 | verified 2026-09-18 | no ablation reported |
| same | 100M | eval | decoding | T = 0.7, top-p 1.0, top-k −1, max 1,024 new tokens | arXiv:2512.07783 App. A.3.3 | verified 2026-09-18 | — |
| same | 100M | all | hardware, wall-clock, GPU count | not reported | checked §2–§6 and App. A.3 | not reported | — |

## Findings relevant to generality
- Generality is measured as pass@k with process verification, splitting extrapolative generalization (deeper operation counts) from contextual generalization (unseen surface context) (§2.2, §2.3).
- The paper states explicitly that its conclusions are drawn in a synthetic, fully controlled setting, and notes that RL applied to already-covered distributions does not improve pass@128 on standard tasks such as math and coding (§3).
- The base-model-family effect is discussed: Qwen bases respond to RL more than LLaMA bases, and mid-training is argued to be what largely determines that difference (§5).

## Findings relevant to negative feedback
- Reward hacking is defined here as reaching a correct final answer through an invalid reasoning chain. The mitigation tested is a denser process-level reward mixed into the outcome reward, not an explicit penalty on failures (§6).

## Connections
- [[front-loading-reasoning]] — a separate study of early-stage reasoning exposure; compare stage definitions before combining conclusions.
- [[echo-chamber-rl-post-training]] — agrees that RL depends on pre-trained support; this paper adds the headroom and edge-of-competence conditions.
- [[rlvr-beyond-base-model]] — the pass@k-shrinkage result this paper conditions: no pass@128 gain in-distribution, gains at the edge.
- [[grpo]] — the RL algorithm used for every post-training run.
- [[math-shepherd]], [[lets-verify]] — process supervision, the mechanism behind §6.
- [[anthropic-model-spec-midtraining]] — mid-training as a named stage in an official pipeline description.
- [[deepseek-r1]], [[prorl]] — large-scale RL reports this controlled study is meant to help interpret.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2512.07783 (arXiv v1, 2025-12-08)
- Corrections to the previous card version:
  - "Reasoning gains are not attributable to RL alone … RL only produces real capability expansion when …" stated without any measurement → replaced with the measured pass@128 conditions and the +42% / +60% / +10.8% figures from Figure 1 and §3–§5.
  - "Mid-training is presented as a distinct and important stage, not just a naming variation on SFT" → the paper defines mid-training as continued pre-training on supervised tokens between pre-training and RL (App. A.3.3 header: "Mid-training (Continue Pre-training)").
  - "Under the paper's controlled setting, mid-training gives better results than using the same compute budget for RL-only post-training" stated unconditionally → the result is split by task band: light RL is best on OOD-edge pass@1, heavy RL is best on OOD-hard (§5, Observation 3).
  - "The result is better structural fidelity, not just better top-line accuracy" → quantified as +4–5% pass@1 on extrapolative tasks plus a shift in structural error types (§6, Figures 7–8).
  - Added authors' affiliation-level facts, model size, token counts, GRPO settings, and a recipe ledger, none of which the previous version had.
- Removed as unsupported by the source: "Treat reasoning improvement as a three-stage curriculum problem"; "use pre-training to build minimal but sufficient exposure … under fixed compute" as a general recommendation detached from the op-range setting; the Key-Figures list that named experiment groups ("headroom / boundary experiments", "reward-composition ablations") rather than numbered figures; "[[quiet-star]]: both move some reasoning structure earlier in the pipeline" as a claim about this paper's content (Quiet-STaR is not discussed in the paper).
- Not reported by the source: results at any model size other than 100M; results on natural-language math or code benchmarks; hardware and wall-clock cost; released checkpoints or data.
