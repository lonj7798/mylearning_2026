<!-- scope: MiniMax-01 technical report (arXiv:2501.08313): MiniMax-Text-01, a 456B-total / 45.9B-activated hybrid lightning/softmax-attention MoE; pre-training, 1M-token context extension, SFT/DPO/online-RL post-training with short/long alternation
     deps: [[deepseekmath]], [[dpo]]
     see-also: [[minimax-01-recipe]], [[deepseek-v3]], [[ruler]], [[constitutional-ai]], [[data-constrained-scaling]]
-->

# MiniMax-01: Scaling Foundation Models with Lightning Attention
- **Core Insight:** MiniMax-Text-01 is a Mixture-of-Experts (MoE) model with 456B total and 45.9B activated parameters in which one softmax-attention block follows every seven lightning-attention blocks; it is trained on contexts up to 1M tokens, extrapolates to 4M tokens at inference, and scores 0.910 on RULER at 1M tokens (Abstract, §2, Table 9).
- **Guideline:** When most attention layers are replaced by linear attention, keep periodic softmax-attention layers, because in the report's 410M-7B experiments lightning attention alone matched softmax attention on most downstream tasks but not on Needle-in-a-Haystack (NIAH), while the hybrid with softmax attention every eighth layer reached NIAH 95.7 at 1B and 98.0 at 3B (§2.2.2.3, Tables 3-4).
- **Authors:** MiniMax (byline). App. A lists contributors alphabetically: Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, et al.
- **Year:** 2025 (arXiv v1 2025-01)
- **URL:** https://arxiv.org/abs/2501.08313
- **Source type:** official technical report
- **Relevant topics:** linear attention, lightning attention, hybrid attention, MoE, critical batch size, data-quality experiments, long-context extension, SFT, DPO, GRPO variant, safety alignment, tool use (search)

## Abstract
The report introduces MiniMax-Text-01 and the vision-language model MiniMax-VL-01. Lightning attention is an I/O-aware implementation of a linear-attention variant (TransNormer); the authors combine it with softmax attention and an MoE with 32 experts. They build training and inference systems for this architecture so that models with hundreds of billions of parameters can train and serve contexts of millions of tokens. MiniMax-Text-01 reaches 1M tokens of training context and 4M tokens at inference. MiniMax-VL-01 is built by continued training on 512B vision-language tokens. The authors report performance comparable to GPT-4o and Claude-3.5-Sonnet with a 20-32 times longer context window, and release the weights (Abstract).

## Key Contributions
- Scaling-law and downstream comparisons of softmax, lightning, and hybrid-lightning attention from 70M to 7B parameters (§2.2.2, Table 2, Figure 7).
- An 80-layer hybrid MoE with RoPE on half of the head dimension, PostNorm with DeepNorm, and a global router for load balancing (§2, §2.1, §2.3).
- A pre-training batch-size schedule derived from a power-law fit between loss and critical batch size (§4.2, Figure 13).
- A three-stage long-context extension to 1M tokens, followed by a five-stage post-training schedule that alternates 8,192-token and 1,032,192-token stages (Tables 6-7).
- A reward framework over correctness, truthfulness, helpfulness, and harmlessness, and an online RL stage based on a modified GRPO (§5.2, §5.4.2).

## Key Figures/Tables to Study
- Table 2 and Figure 6: fitted scaling laws per attention type; Tables 3-5: hybrid-variant, sliding-window, and MoE module ablations.
- Figure 13: loss vs critical batch size, with the batch-doubling points.
- Table 6 (long-context extension mixture) and Table 7 (post-training stages).
- Tables 8-12: core, RULER, LongBench v2, MTOB, and in-house results.

## Technical Details
Recipe values with loci are in [[minimax-01-recipe]]. This section summarizes mechanisms.

**Architecture.** 80 layers; a softmax-attention block follows every 7 lightning-attention blocks; 64 heads of dimension 128; softmax layers use Grouped Query Attention (GQA) with group size 8; RoPE is applied to half of the head dimension with base 10,000; hidden size 6144; 32 experts per layer with top-2 routing and expert FFN dimension 9216 (§2). MoE training uses token dropping with a per-expert capacity limit, a GShard-style auxiliary loss, and a global router that all-gathers per-expert token counts across expert-parallel groups before dispatch (§2.1). Total parameters were capped at 500B so that 1M-token inference fits on one 8×80G node with 8-bit quantization (§2.4).

**Lightning attention.** Linear attention computes Q(KᵀV) instead of (QKᵀ)V (Eq. 3). Lightning attention splits the causal computation into blocks: the left product inside each block and the right product across blocks, giving time complexity O(nd² + nBd), where n is sequence length, d feature dimension, and B block size (§2.2.1, Algorithm 1).

**Ablations.** Fitted losses are L(C) = 3.7087·C^−0.0798 (softmax), 3.5391·C^−0.0768 (lightning), and 3.4797·C^−0.0763 (hybrid), with C the compute budget (Table 2). At 1B, hybrid-lightning scored NIAH 95.7 vs 91.8 for hybrid-HGRN2 and 43.6 for hybrid-cosFormer2 (Table 3). In a 28B-total / 5B-activated MoE trained on 1T tokens, hybrid-lightning scored higher than softmax on 6 of 8 benchmarks (Table 5). In a 60B / 9.3B-activated, 48-layer model trained on 500B tokens, PostNorm scored higher than PreNorm on 7 of 8 benchmarks, for example MMLU 50.2 vs 43.9 (Table 5).

**Data.** A previous-generation MoE (5B activated, 60B total) labels document quality; the retained dimensions are knowledge depth, practical helpfulness, and categorical distribution (§4.1.1). The tokenizer is byte-level BPE with a 200K vocabulary and up-sampled multilingual content (§4.1.2). Data choices are tested as hypothesis tests on 1B-activated / 8B-total MoEs trained on 40B tokens (20B web + 20B candidate data), at 95% confidence and 80% power (§4.1.3.1). In a repetition-aware setup, low-quality data lost performance after more than two epochs, while high-quality data could be trained for up to four epochs (§4.1.3.2).

**Pre-training.** AdamW at sequence length 8192, with batch size doubled from 16M to 128M (unit printed as "M") at 69B, 790B, and 4.7T tokens, a 2e-4 peak learning rate lowered to 1.3e-4 after anomalous gradient norms, and a 1T-token exponential decay to 3e-5 (§4.2). Long-context extension runs 128K → 512K → 1M with RoPE base 5M, then 10M, and mixes 10% long-context QA data into the last 20% of each stage (§4.2, Table 6).

**Post-training.** Prompts are tagged by task type, domain, and difficulty, and cover long-context, programming, math, logical reasoning, creative writing, function calling, general knowledge, and safety (§5.1). Correctness rewards come from an early MiniMax-Text-01 checking answer consistency (math, reasoning) and from sandbox test-case success rates (code) (§5.2). SFT data comes from domain expert models trained through iterative SFT and RL, via rejection sampling at several temperatures, followed by n-gram and semantic-similarity filters (§5.3). Offline RL is DPO on best-vs-worst response pairs over SFT-trained prompts (§5.4.1). Online RL uses SFT-untrained prompts with moderate success rates and a modified GRPO: extra clipping that discards tokens with a large policy ratio and negative advantage, a KL term 𝔼_t[SG(π_θ − π_ref)·log π_θ] where SG is stop-gradient, and advantage balancing between positive and negative examples (§5.4.2). A harmless reward model built from safety rules that include helpfulness principles generates safety responses (§5.5.2). Stages: SFT at 8,192 tokens → training at 1,032,192 tokens with 50% long-context prompts → DPO at 8,192 → DPO at 1,032,192 → online RL at 8,192, with RoPE base 10M throughout (§5.6, Table 7).

**Systems and evaluation.** Training ran on a cluster of 1,500-2,500 H800 GPUs (§3). The systems work includes expert tensor parallelism with EP-ETP communication overlap (§3.1), varlen ring attention and an improved LASP (§3.2), and inference kernels reaching over 75% MFU on H20 (§3.3). Instruction-tuned models are evaluated with greedy decoding and zero-shot chain-of-thought (§5.7.1); MiniMax-Text-01 scores MMLU 88.5, IFEval 89.1, Arena-Hard 89.1, GPQA Diamond 54.4, MATH 77.4, HumanEval 86.9 (Table 8).

## Recipe ledger
Full table (architecture, pre-training, long-context, post-training stages, ablation settings): [[minimax-01-recipe]].

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality.** Removing all low-scoring documents lowered downstream performance, so sampling weights start uniform and are shifted toward high-quality content while keeping category coverage (§4.1.1). Heavy formatting of dialogue and QA data reduced diversity; a nested format with varied templates is used "to maintain format generalization capabilities" (§4.1.1). The authors report a gap between academic benchmarks and in-house user-derived evaluations; MiniMax-Text-01 scores 46.3 on in-house Instruction Following and 64.8 on Hard Capability, which they attribute to insufficient training data for specific instruction types (§5.8.1, Table 12). No ablation isolates these effects.
- **Negative feedback.** DPO uses the worst sampled response as the rejected sample (negative as gradient, §5.4.1). Online RL discards the large-ratio, negative-advantage token case and balances reward contributions of positive and negative examples; no effect size is reported (§5.4.2). Reusing prompts from previous phases in online RL caused "model saturation, characterized by diminished response perplexity" (§5.4.2). Helpfulness principles are added to the harmless reward model to prevent unreasonable refusals; no refusal rate is reported (§5.5.2).
- **Long context.** NIAH reaches its peak score within the initial 128K training steps and does not track progress, so harder tasks are used for checkpoints (§4.2). Scores: RULER 0.947 at 128K and 0.910 at 1M (Table 9); LongBench v2 with CoT 56.5 overall vs 53.7 for humans (Table 10); MTOB English→Kalamang ChrF gain of 45.7 with half the book in context (Table 11). The listed limitations say long-context retrieval evaluations are mostly artificial (§7).
- **Agentic training.** Search is invoked through special tokens; SFT data contains search and non-search decisions with other features (for example conversation length) balanced "to prevent overfitting", and samples the model "already masters" (for example general Chinese knowledge Q&A) are removed. Human evaluation on an out-of-domain Hailuo AI end-to-end set improved from 58% to 71.5% (§5.8.2).
- **Distillation.** SFT targets are sampled from domain-specific expert models and selected by the reward hierarchy (§5.3); expert-model sizes and data amounts are not reported.

## Connections
- [[deepseekmath]] and [[grpo]]: the GRPO objective that §5.4.2 modifies.
- [[dpo]]: the offline RL method used in post-training Stages III-IV.
- [[deepseek-v3]]: a comparison model in Tables 8, 10, and 12.
- [[ruler]]: the long-context benchmark behind the 1M-token result in Table 9.
- [[data-constrained-scaling]]: cited for the up-to-four-epoch observation on high-quality data (§4.1.3.2).
- [[constitutional-ai]]: basis of the harmlessness criteria and harmless reward model (§5.2, §5.5.2).
- MiniMax-M1 (arXiv:2506.13585, 2025-06) continues training MiniMax-Text-01 and adds long-CoT SFT and CISPO RL; no card for it exists in this library yet.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2501.08313 (v1, 2025-01-14; the only version)
- Corrections to the previous card version:
  - Card mixed two artifacts (title "MiniMax-01 / MiniMax-M1", both URLs). Rewritten to describe arXiv:2501.08313, the artifact named by the slug.
  - "Pretraining LR 8e-5 → 8e-6" and "Pretraining 7.5T tokens" → these are MiniMax-M1 continual pre-training values (M1 §2.1). MiniMax-01 warms up to 2e-4 over 500 iterations, trains at constant LR for 7.2T tokens, lowers LR to 1.3e-4 for the remaining 3.2T tokens, and decays to 3e-5 over 1T tokens (§4.2).
  - "AdamW ε = 1e-15 (pretraining; RL reuses)" → ε = 1e-15 is M1's RL setting (M1 §3.2). MiniMax-01 pre-training prints β1 0.9, β2 0.95, weight decay 0.1 and no ε (§4.2).
  - "KL β not surfaced" → for MiniMax-01, a reformulated KL term is given in §5.4.2 (coefficient not printed). M1 states CISPO has no KL penalty term (M1 §3.1).
- Moved out because they describe MiniMax-M1 (arXiv:2506.13585), not this report: CISPO objective (M1 §3.1, Eq. 4-5); 2x speedup vs DAPO on Qwen2.5-32B-base (M1 §1, Figure 2); 512 H800 GPUs for three weeks, $534,700 (M1 Abstract); 40K and 80K releases (M1 Abstract); GenRM length-bias monitoring (M1 §4.2.2); RL data counts of nearly 50K math, about 53K SynLogic logic, 30K competitive programming, several thousand SWE samples, 25K general (M1 §4.1, §4.2.1); long-CoT SFT with about 60% math and coding (M1 §2.2).
- Removed as unsupported: "1M-token training context not seen elsewhere at this scale in 2025"; "a new open cost-efficiency point"; "CISPO distinct from DAPO (dynamic sampling)" (M1 §3.1 adopts DAPO's dynamic sampling); "SynLogic provides self-generated logic problems" (M1 §4.1 describes rule-based generators); "V3.2 DSA vs lightning attention" connection.
- Not reported by the source: total pre-training token count as one number; mixture percentages; gradient clipping; total GPU hours; SFT and DPO dataset sizes; DPO β; RL clip ε, KL coefficient, group size, prompts per step, number of steps, maximum response length, sampling temperatures.
