<!-- scope: Gemma 2 technical report (arXiv:2408.00118): 2B/9B/27B open models; knowledge distillation replaces next-token prediction for 2B and 9B pre-training while 27B is trained from scratch; distillation ablations; SFT with teacher responses and on-policy distillation; RLHF with a larger reward model; model averaging; human and format-robustness evaluations
     see-also: [[gemini-2.5-deep-research]], [[orca]], [[orca-2]], [[distilling-step-by-step]], [[llama-3]]
-->

# Gemma 2: Improving Open Language Models at a Practical Size
- **Core Insight:** A 2B model trained on 500B tokens averages 60.3 on 3 benchmarks from scratch and 67.7 when distilled from a 7B teacher (Table 6); the released 2B and 9B models are pre-trained with distillation on 2T and 8T tokens, and the 27B model is trained from scratch with next-token prediction on 13T tokens (§1, §3.1).
- **Guideline:** When a small model is trained on many more tokens than the compute-optimal amount and a larger teacher is available, use the teacher's next-token distribution as the target instead of the one-hot token, because the report measures gains over from-scratch training for 200M-2B students of a 7B teacher (Tables 6-7); the report does not test distillation for its 27B model.
- **Authors:** Gemma Team, Google DeepMind (core contributors: Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, et al.)
- **Year:** 2024 (arXiv v1 2024-07-31; v3 2024-10-02; the PDF header is dated 2024-06-27)
- **URL:** https://arxiv.org/abs/2408.00118
- **Source type:** official technical report
- **Relevant topics:** knowledge distillation in pre-training, on-policy distillation in SFT, reward-model size in RLHF, model averaging, local-global attention, logit soft-capping, format robustness, multi-turn human evaluation, memorization

## Abstract
Gemma 2 is a family of open models from 2 billion to 27 billion parameters. The architecture adds known Transformer modifications, including interleaved local-global attention and grouped-query attention. The 2B and 9B models are trained with knowledge distillation instead of next-token prediction. The authors report the best performance for their size and competitiveness with models 2-3× bigger, and release all models.

## Key Contributions
- Pre-training of 2B and 9B students on "more than 50×" the compute-optimal token count with distillation, plus a 27B model trained from scratch (§1).
- Architecture changes from Gemma 1: alternating local sliding-window and global attention, logit soft-capping, pre- and post-norm RMSNorm, GQA, and deeper networks (§2).
- Post-training of SFT, RLHF, and model averaging, with teacher-generated responses and on-policy distillation in SFT and a reward model an order of magnitude larger than the policy (§4).
- Ablations of distillation versus from-scratch training, distillation versus model size, GQA versus MHA, wide versus deep, inference-time sliding window, and prompt-format robustness (§5).
- Evaluations: automated benchmarks, LMSYS Chatbot Arena, human preference and multi-turn studies, memorization, and safety and dangerous-capability tests (§6-§8).

## Key Figures/Tables to Study
- Tables 1-3: architecture, parameter counts, training infrastructure.
- Tables 6-7: distillation versus from-scratch training at 2B and at 200M-1B.
- Table 11: MMLU standard deviation over 12 prompt/evaluation formats.
- Tables 12-13 and 17: pre-trained benchmarks and pre-trained versus instruction-tuned few-shot scores.
- Tables 14-16: Chatbot Arena Elo, human instruction-following and safety, multi-turn ratings.

## Technical Details
**Architecture.** d_model / layers / feedforward dim / heads / KV heads / head size: 2B 2304 / 26 / 18432 / 8 / 4 / 256; 9B 3584 / 42 / 28672 / 16 / 8 / 256; 27B 4608 / 46 / 73728 / 32 / 16 / 128 (Table 1). Vocabulary 256,128 with tied embeddings (Table 1). Non-embedding parameters are 2,024,517,888 (2B), 8,324,201,984 (9B), and 26,047,480,320 (27B) (Table 2). Context length is 8192 tokens; local layers use a 4096-token sliding window and global layers an 8192-token span, alternating every other layer (§2). Logits are capped as soft_cap · tanh(logits / soft_cap) with soft_cap 50.0 in self-attention and 30.0 in the final layer (§2). GQA uses num_groups = 2 (§2).

**Distillation objective.** The student minimizes Σ_x −P_T(x | x_c) log P_S(x | x_c), where x is a vocabulary token, x_c its context, P_T the teacher's probability, and P_S the student's parameterized probability (§3.2). The report states that Gemini 1.5 also used knowledge distillation (§3.2). The teacher of the released 2B and 9B is described only as "a large language model" (§1). The 7B ablation teacher keeps "a ratio similar to our target distillation from 27B to 9B" (§5).

**Data and compute.** Primarily-English data from sources "including web documents, code, and science articles"; the mixture was chosen "through ablations similar to the approach in Gemini 1.0" (§3.1). Tokenizer: SentencePiece with split digits, preserved whitespace, and byte-level encodings, 256k entries (§3.1). Filtering follows Gemma 1, including decontamination of evaluation sets (§3.1). Estimated pre-training emissions are 1247.61 tCO2eq (§3.4). Verbatim memorization, tested as 50-token continuations of 50-token training prompts, is below 0.1% (§7, Figure 1).

**Post-training.** Data extends Gemma 1.1 post-training data with internal and external public data; LMSYS-chat-1M contributes prompts but not answers (§4). SFT runs behavioral cloning on synthetic and real prompts with responses "predominantly synthetically generated by the teacher", and also "distillation from the teacher on the student's distribution", citing Agarwal et al. (2024) and Gu et al. (2024) (§4). RLHF uses "a similar RLHF algorithm as Gemma 1.1" with a reward model "an order of magnitude larger than the policy", oriented to multi-turn conversation (§4). Synthetic data is filtered to remove personal information, unsafe or toxic outputs, mistaken self-identification, and duplicates (§4). Generations end with `<end_of_turn><eos>` (§4, Tables 4-5).

**Model averaging.** §4 states: "We average different models obtained by running our pipeline with different hyperparameters (Ramé et al., 2024)", where the citation is WARP. The §4 overview also says the models "obtained after each phase" are averaged. The report does not describe EMA, SLERP, or interpolation steps, weights, or the number of averaged models.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Gemma 2 2B; 9B; 27B | 2B; 9B; 27B | pretrain | tokens seen | 2B: 2T; 9B: 8T; 27B: 13T | arXiv:2408.00118v3 §3.1 | verified 2026-09-14 | no ablation reported |
| Gemma 2 2B; 9B | 2B; 9B | pretrain | training target | teacher next-token distribution (§3.2 loss) | Abstract; §1; §3.2 | verified 2026-09-14 | Table 6: 67.7 vs 60.3 (2B, 500B tokens, 7B teacher); Table 7 |
| Gemma 2 27B | 27B | pretrain | training target | next-token prediction, from scratch | §1; §6.1 | verified 2026-09-14 | no ablation reported |
| Gemma 2 2B; 9B | 2B; 9B | pretrain | teacher identity and size | not reported | checked Abstract, §1, §3.2, §5 | not reported | n/a |
| Gemma 2 (all) | all | pretrain | context length | 8192 tokens; sliding window 4096; global span 8192 | §2; Table 1 | verified 2026-09-14 | Table 10 (9B, window changed at inference): perplexity 1.63 / 1.63 / 1.64 at 4096 / 2048 / 1024 |
| Gemma 2 (all) | all | pretrain | mixture | sources "including web documents, code, and science articles"; shares not reported | §3.1 | verified 2026-09-14 (sources); shares not reported | "ablations similar to Gemini 1.0"; results not shown |
| Gemma 2 2B; 9B; 27B | 2B; 9B; 27B | pretrain | compute | 2B: 512 TPUv5e, 512 data replicas, 1 model shard; 9B: 4096 TPUv4, 1024, 4; 27B: 6144 TPUv5p, 768, 8 | Table 3; §3.3 | verified 2026-09-14 | n/a |
| Gemma 2 (all) | all | pretrain | optimizer, peak/final LR, warmup, schedule, batch, weight decay, grad clip | not reported | checked §2-§3, §5, Tables 1-3 | not reported | n/a |
| Distillation ablation | 2B | pretrain | tokens; teacher | 500B tokens (10× compute-optimal for 2B); 7B teacher | §5; Table 6 | verified 2026-09-14 | Table 6 |
| Distillation ablation | 200M; 400M; 1B | pretrain | teacher; tokens | 7B teacher; tokens not reported | §5; Table 7 | verified 2026-09-14 | Table 7: perplexity 23→21, 19→17, 17→15 |
| Gemma 2 IT (all) | all | SFT | data | text-only, English-only synthetic and human prompt-response pairs; LMSYS-chat-1M prompts only; responses mostly from the teacher; counts and tokens not reported | §4 | verified 2026-09-14 | no ablation reported |
| Gemma 2 IT (all) | all | distill-SFT | on-policy distillation | used; divergence, sampling settings, loss weight not reported | §4 | verified 2026-09-14 (use); settings not reported | no ablation reported |
| Gemma 2 IT (all) | all | SFT | epochs, LR, batch, packing, loss masking | not reported | checked §4-§6 | not reported | n/a |
| Gemma 2 IT (all) | all | reward-model | size; data | an order of magnitude larger than the policy; labelled English-only preference data | §4 | verified 2026-09-14 | no ablation reported |
| Gemma 2 IT (all) | all | RL | algorithm; prompts | similar to Gemma 1.1 RLHF; same prompts as SFT | §4 | verified 2026-09-14 | no ablation reported |
| Gemma 2 IT (all) | all | RL | KL coefficient, LR, batch, samples per prompt, steps | not reported | checked §4 | not reported | n/a |
| Gemma 2 IT (all) | all | merge | method | average of models from pipeline runs with different hyperparameters (cites WARP); weights and count not reported | §4 | verified 2026-09-14 | no ablation reported |
| Gemma 2 IT (all) | all | eval-gate | selection rule | mixtures and tuned hyperparameters chosen for "improving helpfulness while minimizing model harms related to safety and hallucinations" | §4 | verified 2026-09-14 | no numbers reported |

The report does not separate pre-training phases, so pre-training rows use `pretrain`. No LR, batch, or schedule is available for a starting configuration.

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Distillation.** Table 6 (2B, 500B tokens): 60.3 from scratch, 67.7 distilled from 7B. Table 7: validation perplexity from scratch vs distilled is 23 vs 21 (200M), 19 vs 17 (400M), and 17 vs 15 (1B), and the authors state that "the gain remains as the model size is scaled" (§5). Gemma 2 2B (2T tokens) averages 50.0 on 8 benchmarks against 44.0 for Gemma 1 2B (3T tokens) (Table 13); the architectures also differ (§2), so this comparison does not isolate distillation.
- **Generality.** Instruction tuning raises few-shot MMLU (52.2→56.1, 71.3→72.3, 75.2→76.2) and MBPP (30.2→36.6, 52.4→59.2, 62.6→67.4) for 2B, 9B, 27B (Table 17). MMLU standard deviation over 12 formats is 2.1 (2B), 0.9 (9B), 1.0 (27B), against 6.9 for Mistral 7B (Table 11). HumanEval is the only Table 13 benchmark where Gemma 2 2B (20.1) is below Gemma 1 2B (22.0). On 500 held-out multi-turn scenarios (8.4 user turns on average), user satisfaction is 3.64 / 4.04 / 4.20 for 2B / 9B / 27B against 3.32 for Gemma 1.1 7B (Table 16).
- **Negative feedback.** Synthetic examples with personal information, unsafe or toxic outputs, mistaken self-identification, or duplicates are discarded (§4; negative marginal value). Adding data that encourages in-context attribution, hedging, and refusals (negative as content) improves factuality metrics "without degrading model performance on other metrics" (§4); no numbers are given.
- **Long context.** Training context is 8192 tokens (§2); no longer-context evaluation is reported.

## Connections
- [[gemini-2.5-deep-research]]: Gemini 2.5 distills Flash-size and smaller models with a k-sparse teacher distribution, continuing the Gemini 1.5 practice this report mentions (§3.2).
- [[orca]], [[orca-2]]: distillation from teacher-written responses and explanations, comparable to Gemma 2's teacher-generated SFT responses; Gemma 2 also matches the full teacher distribution in pre-training.
- [[distilling-step-by-step]]: rationale distillation into small students, another sequence-level method.
- [[llama-3]]: Table 12 compares Gemma 2 27B with LLaMA-3 70B, and §1 cites Llama 3 for small models needing up to 15T tokens to improve the state of the art by less than 1-2%.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2408.00118 (v3, 2024-10-02; full PDF including §5 ablations and Tables 1-17).
- Corrections to the previous card version:
  - "Pretraining uses knowledge distillation from a larger teacher" for all sizes → only 2B and 9B; 27B is trained from scratch (Abstract, §1, §6.1).
  - "WARP merging: EMA during RL, SLERP after RL, repeat" → §4 says only that models from runs with different hyperparameters are averaged, citing WARP; no stages are described.
  - "KL(student || teacher) minimized" in on-policy distillation → the report gives no divergence (§4).
  - "9B/27B student reach quality usually demanding 70B+" → 27B is not distilled; it is "a few percent below LLaMA-3 70B" on Table 12 and has Arena Elo 1218 vs 1206 for Llama 3 70B (Table 14).
  - "logit softcapping to stabilize large-vocab logits" → soft-capping applies in attention (50.0) and the final layer (30.0); no large-vocabulary rationale is given (§2).
  - see-also self-link to this card (`gemma-2`) and deps link to `README` removed.
- Removed as unsupported by the source: teacher "presumed Gemini Ultra"; "on-policy distillation diagram", "WARP merging schematic", and "RM-larger-than-policy ablation" (not in the report); oversized RM as "a reversal of the usual configuration"; Gemma 2 safety tuning borrowing Constitutional AI self-critique; LIMA "quality over quantity" link; Llama 3 "same-size RM and full hyperparameter disclosure"; Gemma 3 claims (separate report, not checked here).
- Not reported by the source: teacher identity, optimizer and LR schedule, batch size, mixture shares, SFT and preference data sizes, RLHF hyperparameters, on-policy distillation settings, merge weights.
