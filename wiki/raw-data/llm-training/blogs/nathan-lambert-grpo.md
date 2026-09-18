<!-- scope: Interconnects survey of 2025 GRPO modifications, base-model RL, and RL prompt curation
     see-also: [[grpo]], [[dr-grpo]], [[deepseekmath]], [[deepseek-r1]], [[rloo-vs-grpo]]
-->

# Recent reasoning research: GRPO tweaks, base model RL, and data curation
- **Core Insight:** GRPO is not a distinct class of algorithm — it is PPO-derived with an RLOO-like advantage — and the
  2025 modifications reviewed (DAPO, Dr. GRPO) target one shared property, the per-response length normalization
  1/|o_i|, which over-reinforces short correct responses and under-penalizes long wrong ones.
- **Guideline:** When implementing GRPO and the objective is longer correct chains of thought, change the length
  normalization; the author prefers DAPO's form (1/|o| moved outside the group sum, so the group is normalized by its
  total token count) to Dr. GRPO's removal of the term. Data distribution and batching are said to matter more.
- **Author:** Nathan Lambert (Interconnects)
- **Year:** 2025 (published 2025-03-31)
- **URL:** https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo
- **Source type:** practitioner evidence (survey of four papers; the post reports no experiments of its own)
- **Relevant topics:** GRPO length normalization, DAPO, Dr. GRPO, base-model RL, RL prompt curation

## Summary
The post reviews four 2025 works — Kimi k1.5, Open-Reasoner-Zero (ORZ), DAPO, and "Understanding R1-Zero-Like Training:
A Critical Perspective" (Dr. GRPO) — and opens with a section titled "Is GRPO special?" arguing that it is not: GRPO is
derived from PPO, its advantage computation is close to RLOO, and the only difference from REINFORCE is PPO's clipping.
Three things are named as actually changing in RL for language models: whether a value function is used, what role
log-ratio clipping plays, and using multiple samples per prompt. The technical core is GRPO's 1/|o_i| normalization and
the two corrections DAPO and Dr. GRPO make to it.

## Key Contributions
- A stated position that algorithm choice among GRPO, RLOO, PPO, and REINFORCE matters less than batching,
  infrastructure, and data distribution, with a rule of thumb that an RL paper not beating its baseline by roughly 2x is
  likely succeeding through hyper-parameter tuning or confounds (applied to DAPO's x-axis, whose "DeepSeek R1 Zero Qwen
  32B" comparison run does not exist as plotted).
- A worked argument for the two biases 1/|o_i| introduces, split by advantage sign, and a side-by-side reading of
  DAPO's four changes against Dr. GRPO's two.
- A summary of what Kimi k1.5 and ORZ report about RL prompt curation, plus a caveat attributed to Costa Huang:
  removing the group standard deviation downweights the rare correct answer on a 1-in-10 prompt.

## Key Figures/Tables to Study
- The GRPO loss reproduced from DeepSeekMath (locates the 1/|o_i| term); DAPO's clip-higher ablation, read as the model
  retaining higher entropy later in training; Dr. GRPO's base-model comparison under three prompt templates.

## Technical Details
### The two biases of 1/|o_i| (stated by advantage sign)
1. **Positive advantage (correct response).** Two correct responses in a group, 10 tokens and 1000 tokens, receive the
   same advantage A_i, but the loss approximates advantage divided by length, so the per-token increase is larger for
   the shorter one. The EOS token is the concrete case: its probability is raised far more for the short response.
2. **Negative advantage (wrong response).** The 1/|o| factor reduces the per-token penalty on longer answers, so a long
   repetitive wrong answer is penalized less than a short wrong answer.
These are the opposite of what reasoning training wants: longer correct answers, no wasted tokens.
### DAPO's four changes (as the post lists them)
1. **Clip-higher**: two clipping hyper-parameters so the upper (positive) log-ratio step can be larger, raising the
   probability of low-probability tokens that would otherwise be clipped first.
2. **Dynamic sampling**: drop prompts whose samples all have the same reward, since the group-relative advantage is zero
   and no gradient is produced.
3. **Token-level policy gradient**: move 1/|o| outside the group sum, so the group is normalized by its total token
   count rather than each response by its own length.
4. **Length-based reward shaping**: for a 16k-token maximum a penalty starts at 12k and rises linearly to 16k; the post
   calls this the most minor change.
DAPO also removes the KL penalty. The stated condition: agreed for base-model RL, where the policy must move a long
way, but the KL penalty may still help for RLVR on an instruct model.
### Dr. GRPO's two changes (as the post lists them)
1. Remove the 1/|o| length normalization from the policy-gradient term.
2. Remove the division by the group's reward standard deviation, described as a question-level difficulty bias:
   high-variance (harder) questions are otherwise downweighted relative to uniformly easy or hard ones. The post notes
   this is already in TRL, and quotes the paper that prior RL work normalizes advantages across a whole batch.
Dr. GRPO is reported to produce the expected length behaviour — shorter outputs overall and shorter incorrect answers —
but not better final downstream performance, which the author states is the goal and likely comes more from data. The
implementation difference is `masked_mean` (TRL and other open libraries) versus the `masked_sum` used by the Dr. GRPO
authors. One reason per-response normalization can be preferable: with a KL penalty on, a single high-KL token should
not affect every token in the batch.
### Base-model RL and prompt curation
- ORZ used PPO with GAE over a group of responses rather than GRPO, a discount factor of 1, no length or formatting
  reward (correctness only), and removed both the KL loss and the KL penalty, which its authors report as giving the
  best stability and final performance. Appendix ablations include sampling temperature (T = 1 best). It released 57K
  training samples and filtered out Chinese datapoints as degrading performance.
- Kimi k1.5 did not use GRPO either; it used online policy mirror descent with a Monte Carlo reward baseline, plus a
  length penalty promoting shorter correct responses and penalizing long incorrect ones, phased in during training. Its
  prompt set targets diverse coverage and balanced difficulty, difficulty estimated by sampling an SFT model ten times
  at high temperature and using the pass rate. It excludes multiple-choice, true/false, and proof-based questions
  because they permit correct answers from incorrect reasoning. ORZ applies the same exclusion, filters prompts whose
  pass rate is too high or zero, and deduplicates by n-gram and embedding similarity.
- Dr. GRPO's base-model study compares Qwen 2.5, Llama 3.1, and DeepSeek bases under the R1 template, the Qwen-Math
  template, and no template: Llama and DeepSeek follow instructions best with R1's, Qwen best with none, and the larger
  Qwen bases already show reflection behaviour before any RL.

## Findings relevant to generality and negative feedback
- **Negative feedback (negative as gradient).** The second bias above is a claim about negative advantages: 1/|o|
  shrinks the per-token penalty on long wrong responses, so repetition is not driven out. Both corrections alter the
  magnitude of that negative gradient; no number is given for the size of the effect.
- **Generality and long context.** Kimi k1.5 is reported as attributing RL effectiveness to prompt-set quality and
  diversity across STEM, coding, and general reasoning; a balanced difficulty distribution per batch is said to
  substitute for the algorithmic difficulty-bias correction. Both corrections are motivated by response-length
  dynamics, a concern the author says RLHF had but ignored.

## Connections
- [[grpo]], [[deepseekmath]] — the original algorithm and the paper the 1/|o_i| term is quoted from.
- [[dr-grpo]] — one of the two corrections reviewed here; [[deepseek-r1]] — assumed prerequisite reading.
- [[rloo-vs-grpo]] — advantage-computation comparison; [[nathan-lambert-interconnects]] — lab index page.

## Verification
- Checked on 2026-09-18 against: https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo (2025-03-31).
- Corrections to the previous card version:
  - Title "Interconnects — GRPO Tweaks, Base-Model RL, and Data Curation" → "Recent reasoning research: GRPO tweaks,
    base model RL, and data curation".
  - "Kimi k1.5 / K2 variant — rescales the loss to a per-prompt level and adds length-independent advantage" → Kimi
    k1.5 does not use GRPO; it uses online policy mirror descent with a Monte Carlo reward baseline and a length
    penalty. K2 is not mentioned.
  - "Dr. GRPO — drops length normalization ... while keeping KL normalization per-token" → the two Dr. GRPO changes
    listed are removing 1/|o| and removing the group standard-deviation division; KL normalization is not mentioned.
  - "Sum-not-mean ... used in some community replications" → `masked_mean` → `masked_sum` is attributed to the Dr. GRPO
    authors. Guideline "default to token-level aggregation (no per-sequence length normalization)" → the post prefers
    DAPO's form (1/|o| moved outside the group sum) to removing the term entirely.
  - "repetitive patterns ... not penalized as sharply as under PPO" → the comparison is between positive and negative
    advantages under GRPO's own normalization, not against PPO. "R1's 800K is upper bound; many replications work with
    <50K" → not in the post; the only dataset size given is ORZ's 57K training samples.
- Removed as unsupported by the source: "REINFORCE++ (2025)" as a reviewed fix; base-model RL as "cleaner science" but
  not a production recipe; "no cold-start ... mixed-language output, unreadable CoT, but higher observed creativity";
  "high reward variance per Qwen 2.5's observation"; "careful RL-prompt filtering beats adding more prompts" as a stated
  conclusion; two "Key Figures" items naming plots the post does not contain (a Dr. GRPO bias-correction equation
  side-by-side, per-variant response-length-over-training plots).
- Not reported: Kimi k1.5 size-ablation model sizes (the post notes the omission), learning rates, batch sizes, KL
  coefficients, replication results.
