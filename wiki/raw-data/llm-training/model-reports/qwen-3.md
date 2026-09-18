<!-- scope: Qwen3 technical report — unified thinking/non-thinking post-training and strong-to-weak distillation
     deps: [[deepseek-r1]], [[llama-3]]
     see-also: [[deepseek-v3]], [[tulu-3]], [[self-instruct]], [[qwen-2.5]]
-->

# Qwen3 Technical Report
- **Core Insight:** From the same off-policy-distilled Qwen3-8B checkpoint, on-policy distillation reaches
  AIME'24 74.4 in 1,800 GPU hours while RL reaches 67.6 in 17,920 GPU hours (Table 21).
- **Guideline:** When a smaller model must reach a larger model's reasoning level and a stronger teacher in
  the same family exists, use on-policy logit distillation rather than RL, because in the Qwen3-8B
  comparison it scored higher at about one tenth of the GPU hours and raised pass@64 on AIME'24 and AIME'25
  while RL did not (§5.3, Table 21).
- **Authors:** Qwen Team (Alibaba)
- **Year:** 2025 (arXiv v1 2025-05)
- **URL:** https://arxiv.org/abs/2505.09388
- **Source type:** official technical report
- **Relevant topics:** thinking budget, thinking mode fusion, on-policy distillation, long-CoT cold start, GRPO, multilingual pretraining

## Abstract
Qwen3 is a family of six dense models (0.6B to 32B) and two MoE models (30B-A3B and 235B-A22B) that place a
"thinking" and a "non-thinking" mode in one model, switchable by `/think` and `/no think` flags in the chat
template, with a thinking budget that truncates the reasoning block at a user-set length. Pretraining uses
36T tokens across 119 languages and dialects in three stages. Flagship post-training is four stages:
long-CoT cold start, reasoning RL, thinking mode fusion, general RL. The six lightweight models skip those
stages and are built by off-policy then on-policy distillation from Qwen3-32B and Qwen3-235B-A22B.

## Key Contributions
- A unified thinking / non-thinking model with mode flags in the chat template and a thinking budget the
  report states was not explicitly trained (§4.3).
- A four-stage flagship post-training pipeline, and a two-phase distillation pipeline replacing it for the
  six lightweight models at about 1/10 of the GPU hours (§4, Figure 1).
- Instance-level pretraining mixture optimization using proxy-model ablations, over a corpus annotated at
  30T-token scale for educational value, field, domain and safety; 119 languages, up from 29 (§3.1).
- A measured comparison of on-policy distillation against RL from the same checkpoint (§5.3, Table 21).

## Key Figures/Tables to Study
Figure 1 (four-stage pipeline, and where distillation replaces it); Tables 1-2 (architecture and context
length per size); Table 9 (chat-template examples for the two modes); Table 21 (on-policy distillation vs RL
on Qwen3-8B, GPU hours and pass@64); Table 22 (Qwen3-32B after each stage); Figure 2 (thinking budget).

## Technical Details
### Architecture (§2)
Six dense models (0.6B, 1.7B, 4B, 8B, 14B, 32B) and two MoE models (30B-A3B, 235B-A22B); the flagship has
235B total and 22B activated parameters. Dense models use GQA, SwiGLU, RoPE and RMSNorm with
pre-normalization; QKV-bias from Qwen2 is removed and QK-Norm added for training stability. MoE models have
128 total experts with 8 activated per token, no shared experts, and a global-batch load balancing loss.
Tokenizer: byte-level BPE, vocabulary 151,669. Context: 32K for 0.6B and 1.7B, 128K for 4B and above and
both MoE models (Tables 1-2).

### Pretraining
- 36T tokens across 119 languages and dialects (§3.1). Corpus expansion: Qwen2.5-VL performs text
  recognition on PDF-like documents and Qwen2.5 refines the output; Qwen2.5, Qwen2.5-Math and Qwen2.5-Coder
  synthesize trillions of tokens as textbooks, QA, instructions and code snippets. Over 30T tokens are
  annotated for educational value, fields, domains and safety; the mixture is optimized at instance level.
- Three stages (§3.2): (1) General, over 30T tokens at sequence length 4,096. (2) Reasoning, about 5T
  higher-quality tokens at 4,096, with accelerated learning-rate decay. (3) Long-context, hundreds of
  billions of tokens at 32,768, corpus 75% texts of 16,384-32,768 tokens and 25% of 4,096-16,384, RoPE base
  raised 10,000 → 1,000,000 via ABF, with YARN and Dual Chunk Attention giving a four-fold inference-time
  sequence-length increase.

### Post-training, flagship models
1. **Long-CoT cold start** (§4.1). Qwen2.5-72B-Instruct filters out queries that are not easily verifiable,
   contain multiple sub-questions, ask for general text generation, or that it answers correctly without
   CoT; it also annotates domain for balance. Responses come from QwQ-32B, with human annotators assessing
   correctness where QwQ-32B consistently fails. Six response filters remove incorrect final answers,
   substantial repetition, evident guesswork, thinking/summary inconsistency, inappropriate language mixing
   or style shifts, and suspected validation-set similarity. The objective is to minimize samples and steps.
2. **Reasoning RL** (§4.2). 3,995 query-verifier pairs, selected to be unused in cold start, learnable for
   the cold-start model, as challenging as possible, and broad in sub-domain coverage; GRPO. A large batch
   size, a high number of rollouts per query and off-policy training are reported as beneficial, and entropy
   was controlled to increase steadily or stay stable. Qwen3-235B-A22B AIME'24 rises 70.1 → 85.1 in 170 steps.
3. **Thinking mode fusion** (§4.3). Continual SFT on the Reasoning RL model. Thinking data is generated by
   rejection sampling on Stage-1 queries using the Stage-2 model itself; non-thinking data is curated across
   coding, math, instruction following, multilingual tasks, creative writing, QA and role-playing, scored by
   automatically generated checklists.
4. **General RL** (§4.4). A reward system over more than 20 tasks covering instruction following, format
   following, preference alignment, agent ability with real environment execution feedback, and RAG-style
   scenarios; three reward types: rule-based; model-based with a reference answer scored by
   Qwen2.5-72B-Instruct; model-based without a reference, from a trained reward model.

### Strong-to-weak distillation (§4.5)
Applied to 5 dense models (0.6B, 1.7B, 4B, 8B, 14B) and one MoE model (30B-A3B). Phase 1, off-policy:
response distillation from teacher outputs generated in both `/think` and `/no think` modes. Phase 2,
on-policy: the student generates in either mode and is finetuned by minimizing the KL divergence between its
logits and those of Qwen3-32B or Qwen3-235B-A22B.

## Recipe ledger
See [[qwen-3-recipe]].

## Findings relevant to generality, long context and distillation
- On-policy distillation raised AIME'24 pass@64 to 93.3 and AIME'25 pass@64 to 86.7 from the 90.0 and 83.3
  of the starting checkpoint, while RL left both unchanged (Table 21). The report reads this as teacher
  logits expanding the student's exploration space (§5.3).
- Thinking Mode Fusion and General RL did not improve knowledge, STEM, math and coding for Qwen3-32B
  relative to Reasoning RL (§5.3 point 3, Table 22). Extending output length beyond 32K is expected to
  improve performance further, left as future work (§5.3). Thinking-budget control emerges from Thinking
  Mode Fusion rather than being trained (§4.3).

## Connections
- [[deepseek-r1]] — the pure-RL reasoning recipe Qwen3 contrasts with; QwQ-32B supplies the cold-start
  traces instead. [[deepseek-v3]] — the open MoE baseline Qwen3-235B-A22B-Base is compared against (§3.3).
- [[llama-3]] — Meta's synthetic-data plus preference-optimization analogue. [[self-instruct]] — the origin
  of the synthetic-question generation Qwen3 uses at pretraining scale. [[qwen-2.5]] — the predecessor whose
  models generate and refine much of Qwen3's pretraining corpus. [[phi-4]] — the other 2025 report here that
  runs a short GRPO stage on a curated verifiable prompt set. [[qwen-3-recipe]] — this report's ledger.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2505.09388 (arXiv v1, 2025-05-14).
- Corrections to the previous card version:
  - "Dense models expose contexts up to 128K" → 32K for 0.6B and 1.7B, 128K for 4B and above (Table 1).
  - "distillation outperforms RL in both quality and efficiency" carried no number → the measured comparison
    is Table 21 (74.4 at 1,800 GPU hours vs 67.6 at 17,920 on Qwen3-8B), on math and code queries only.
  - "Stage 1-2: ... RL focused on math and coding" → §4.2 states the Reasoning RL query-verifier pairs cover
    a broad range of sub-domains; the math-and-code restriction belongs to the Table 21 comparison (§5.3).
- Removed as unsupported: "synthetic code/data variants from Qwen2.5-Coder and related models" as a separate
  item (§3.1 lists the three Qwen2.5 models together as synthesizers of textbooks, QA, instructions and code
  snippets); "Uses YARN and Dual Chunk Attention to increase usable context during inference", stated
  without the four-fold figure or the pretraining stage it belongs to.
- Not reported: RL batch size, rollouts/query, KL coefficient, clip ε, LR; cold-start SFT sample count or
  steps; Thinking Mode Fusion SFT set size; total pretraining compute.
