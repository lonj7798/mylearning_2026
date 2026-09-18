<!-- scope: Mixtral of Experts (arXiv:2401.04088): Mistral AI's Mixtral 8x7B sparse MoE (8 SwiGLU experts, top-2 routing, 47B total / 13B active, 32k dense context), benchmarks against Llama 2 70B and GPT-3.5, SFT + DPO instruct model, routing analysis; Mixtral 8x22B release post as a supporting source
     see-also: [[deepseek-v3]], [[llama-2]], [[dpo]], [[llama-4]], [[mistral-nemo]], [[nemotron-4-synthetic]]
-->

# Mixtral of Experts
- **Core Insight:** Mixtral 8x7B routes each token at each layer to 2 of 8 feedforward experts, so it uses 13B active of 47B total parameters, and it scores 70.6% on MMLU against 69.9% for Llama 2 70B in the authors' re-run evaluation (Abstract; Table 2).
- **Guideline:** When per-token inference compute is the constraint and memory for all parameters is available, a top-2-of-8 MoE matched or exceeded a dense model with 5x more active parameters on most of the paper's benchmarks (Table 2, Figure 3); memory cost still scales with the 47B total parameters, and the authors state that routing overhead makes SMoE layers "more suitable for batched workloads" (§3 "Size and Efficiency").
- **Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, et al. (Mistral AI)
- **Year:** 2024 (arXiv v1 2024-01-08; model release post 2023-12-11)
- **URL:** https://arxiv.org/abs/2401.04088
- **Source type:** official technical report
- **Relevant topics:** sparse mixture of experts, top-k routing, active vs total parameters, expert parallelism, multilingual pre-training, long-context retrieval, SFT + DPO instruct model, routing analysis

## Abstract
Mixtral 8x7B is a Sparse Mixture of Experts (SMoE) language model with the architecture of Mistral 7B, except that each layer has 8 feedforward blocks (experts). A router selects two experts per token at each layer and combines their outputs, so each token has access to 47B parameters but uses 13B during inference. Mixtral was trained with a 32k-token context and "outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks", and the abstract names mathematics, code generation, and multilingual benchmarks as the areas where it "vastly outperforms" Llama 2 70B. Mixtral 8x7B – Instruct, fine-tuned to follow instructions, surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B – chat on human benchmarks. Both models are released under Apache 2.0.

## Key Contributions
- Open-weight SMoE with all FFN sub-blocks replaced by 8-expert top-2 MoE layers (§2.1).
- Evaluation against Llama 1/2 and GPT-3.5 on commonsense, knowledge, reading comprehension, math, code, aggregate, and multilingual benchmarks (§3, Tables 2-4).
- Passkey retrieval and proof-pile perplexity tests over the 32k context (§3.2, Figure 4).
- Mixtral 8x7B – Instruct trained with SFT then DPO: MT-Bench 8.30 and Arena Elo 1121 (§4, Table 3, Figure 6).
- Routing analysis on The Pile: no clear topic specialization, measurable positional locality (§5, Figure 7, Table 5).

## Key Figures/Tables to Study
- Table 1 (architecture) and the gating equation in §2.1.
- Table 2 (vs Llama models, 12 benchmarks) and Table 3 (vs Llama 2 70B and GPT-3.5).
- Table 4 (French, German, Spanish, Italian ARC-c / HellaSwag / MMLU).
- Figure 7 (per-domain expert selection at layers 0, 15, 31), Figure 8 (tokens colored by first expert choice), Table 5 (consecutive-token expert repetition).

## Technical Details
**Architecture.** dim 4096; n_layers 32; head_dim 128; hidden_dim 14336; n_heads 32; n_kv_heads 8; context_len 32768; vocab_size 32000; num_experts 8; top_k_experts 2 (Table 1). Mixtral uses the modifications of Mistral 7B except that it "supports a fully dense context length of 32k tokens" and replaces FFN blocks with MoE layers (§2). The released config.json gives `sliding_window` null and `rope_theta` 1000000.0 (HF config.json @fc7ac94); the paper does not state rope theta.

**Routing.** The MoE output is Σ_{i=0}^{n−1} G(x)_i · E_i(x), where n is the number of experts, E_i(x) the i-th expert's output, and G(x)_i the gate weight (§2.1). The gate is G(x) = Softmax(TopK(x · W_g)), where W_g is a linear layer's weight and TopK sets every logit outside the top K to −∞, so the softmax runs over the K selected logits only (§2.1). Each expert is a SwiGLU block and K = 2 (§2.1). Compared with GShard, Mixtral replaces every FFN sub-block (GShard replaces every other one) and does not use GShard's more elaborate gating for the second expert (§2.1). The paper does not describe a load-balancing or router auxiliary loss.

**Parameter counts and cost.** Total (sparse) parameters grow with n and active parameters grow with K (§2.1). The paper rounds to 47B total and 13B active (Abstract; §3); the release post gives 46.7B total and 12.9B per token (mistral.ai/news/mixtral-of-experts). Serving memory is proportional to the 47B sparse count; routing and multiple experts per device add overhead (§3). The release post states "6x faster inference" than Llama 2 70B and the "same speed" and cost as a 12.9B model; the paper gives no speed measurement.

**Infrastructure.** Megablocks kernels cast MoE FFN operations as sparse matrix multiplications; Expert Parallelism routes tokens to the GPU holding each expert and needs even load across GPUs (§2.1). vLLM support with Megablocks CUDA kernels was contributed (§1). Training support is credited to CoreWeave and Scaleway (Acknowledgements).

**Benchmarks (authors' pipeline).** Table 2, Mixtral 8x7B vs Llama 2 70B: MMLU 70.6 vs 69.9; HellaSwag 84.4 vs 85.4; WinoGrande 77.2 vs 80.4; PIQA 83.6 vs 82.6; ARC-e 83.1 vs 79.9; ARC-c 59.7 vs 56.5; NQ 30.6 vs 25.4; TriviaQA 71.5 vs 73.0; HumanEval 40.2 vs 29.3; MBPP 60.7 vs 49.8; MATH (4-shot maj@4) 28.4 vs 13.8; GSM8K (8-shot maj@8) 74.4 vs 69.6. Table 3, Llama 2 70B / GPT-3.5 / Mixtral: HellaSwag 10-shot 87.1 / 85.5 / 86.7; ARC-c 25-shot 85.1 / 85.2 / 85.8; WinoGrande 5-shot 83.2 / 81.6 / 81.2; MBPP 49.8 / 52.2 / 60.7; GSM8K 5-shot 53.6 / 57.1 / 58.4; MT-Bench (instruct models) 6.86 / 8.32 (gpt-3.5-turbo-1106) / 8.30. MBPP uses the hand-verified subset and TriviaQA is run without Wikipedia contexts (§3 "Evaluation Differences").

**Bias.** On the base model, BBQ accuracy is 56.0% for Mixtral vs 51.5% for Llama 2 70B, and BOLD sentiment is more positive with similar variances (§3.3, Figure 5).

**Instruction fine-tuning.** Mixtral – Instruct is trained with SFT "on an instruction dataset" followed by DPO "on a paired feedback dataset" (§4). MT-Bench is 8.30, "the best open-weights model as of December 2023" (§4; the text says "see Table 2", but the score is in Table 3). In an LMSys leaderboard screenshot of 2023-12-22, Mixtral 8x7B Instruct v0.1 has Arena Elo 1121 vs Claude-2.1 1117, GPT-3.5-Turbo best 1117, Gemini Pro 1111, Llama-2-70b-chat 1077 (Figure 6).

**Mixtral 8x22B (release post, 2024-04-17).** 39B active of 141B parameters; 64K-token context window; fluent in English, French, Italian, German, Spanish; "natively capable of function calling"; the instructed version scores 90.8% GSM8K maj@8 and 44.6% Math maj@4 (mistral.ai/news/mixtral-8x22b). The post does not describe routing or training settings.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Mixtral 8x7B | 47B total / 13B active | pretrain | context length | 32k tokens, fully dense (context_len 32768) | arXiv:2401.04088v1 §1, §2, Table 1 | verified 2026-09-14 | §3.2 Figure 4: passkey retrieval 100% at all lengths and positions (not an ablation) |
| Mixtral 8x7B | 47B / 13B | pretrain | data | "multilingual data"; multilingual proportion "significantly upsample[d]" vs Mistral 7B; shares not reported | arXiv v1 §1; §3.1 | verified 2026-09-14 (description); shares not reported | Table 4 comparison; no ablation of the upsampling |
| Mixtral 8x7B | 47B / 13B | pretrain | data source; router training | "data extracted from the open Web"; "we train experts and routers simultaneously" | mistral.ai/news/mixtral-of-experts (2023-12-11) | verified 2026-09-14 | no ablation reported |
| Mixtral 8x7B | 47B / 13B | pretrain | tokens seen, optimizer, LR, warmup, schedule, batch, weight decay, compute | not reported | checked arXiv v1 body, figures, appendix, acknowledgements; release post; HF config.json | not reported | n/a |
| Mixtral 8x7B | 47B / 13B | pretrain | router auxiliary / load-balancing loss (paper) | not reported | checked arXiv v1 §2.1, §5 | not reported | n/a |
| Mixtral-8x7B-v0.1 (HF config) | 47B / 13B | pretrain | `router_aux_loss_coef`; `output_router_logits` | 0.02; false (the file does not say whether pre-training used this loss) | huggingface.co/mistralai/Mixtral-8x7B-v0.1@fc7ac94 config.json | verified 2026-09-14 | no ablation reported |
| Mixtral 8x7B – Instruct | 47B / 13B | SFT | data | "an instruction dataset"; source, examples, tokens not reported | arXiv v1 §4 | verified 2026-09-14 (description) | no ablation reported |
| Mixtral 8x7B – Instruct | 47B / 13B | SFT | epochs, LR, batch, loss masking, packing | not reported | checked arXiv v1 §4; release post | not reported | n/a |
| Mixtral 8x7B – Instruct | 47B / 13B | preference | loss; data | DPO on "a paired feedback dataset"; pair count and source not reported | arXiv v1 §1, §4 | verified 2026-09-14 | no ablation reported |
| Mixtral 8x7B – Instruct | 47B / 13B | preference | β, LR, epochs, reference model | not reported | checked arXiv v1 §4; release post | not reported | n/a |
| Mixtral 8x7B – Instruct | 47B / 13B | eval-gate | MT-Bench; Arena Elo | 8.30; 1121 (screenshot 2023-12-22) | arXiv v1 §4, Table 3, Figure 6 | verified 2026-09-14 | Table 3 vs gpt-3.5-turbo-1106 8.32 |
| Mixtral 8x22B | 141B total / 39B active | long-context | context window | 64K tokens | mistral.ai/news/mixtral-8x22b (2024-04-17) | verified 2026-09-14 | no evaluation of the window in the post text |
| Mixtral 8x22B (base and instruct) | 141B / 39B | all | training data, settings, post-training method | not reported | checked the 8x22B release post | not reported | n/a |

## Findings relevant to generality, long context, negative feedback
- **Generality.** Mixtral exceeds Llama 2 70B on 9 of 12 Table 2 benchmarks and is lower on HellaSwag, WinoGrande, and TriviaQA (Table 2); Figure 3 names reading comprehension as the category where it does not outperform. These rows, and WinoGrande 5-shot 81.2 vs 83.2 in Table 3, do not match the abstract's "across all evaluated benchmarks"; §3 and the Table 2 caption use "most" and "almost all". All models were re-evaluated with the authors' own pipeline (§3). The paper reports no contamination analysis.
- **Multilingual.** With upsampled multilingual data, MMLU vs Llama 2 70B is French 70.9 vs 64.3, German 71.5 vs 64.2, Spanish 72.5 vs 66.0, Italian 70.9 vs 65.1 (Table 4); the authors state that English accuracy is maintained (§3.1).
- **Long context.** Passkey retrieval accuracy is 100% regardless of context length or passkey position, and proof-pile perplexity decreases monotonically as context grows up to 32k (§3.2, Figure 4).
- **Expert specialization.** Expert assignment distributions on ArXiv, PubMed Abstracts, and PhilPapers are very similar at layers 0, 15, and 31; only DM Mathematics differs marginally (§5, Figure 7). The same first-choice expert repeats on consecutive tokens 23.6%-28.4% of the time at layer 15 vs 12.5% for random assignment, and first-or-second choice repeats 61.6%-67.0% vs about 46% (Table 5). The authors read this as routing aligned with syntax more than domain (Figure 8 caption) (Interpretation).
- **Negative feedback.** The DPO stage uses the dispreferred response of each pair as a gradient signal (negative as gradient; see [[dpo]]); the paper reports no analysis of chosen or rejected likelihoods.

## Connections
- [[llama-2]] — Llama 2 70B is the main dense comparison model (Tables 2-4).
- [[dpo]] — the preference-optimization method used for Mixtral – Instruct (§4).
- [[deepseek-v3]] — later MoE with 671B total / 37B active, 256 routed plus 1 shared expert, and auxiliary-loss-free balancing; contrasts with top-2-of-8 routing here.
- [[llama-4]] — later open MoE release.
- [[mistral-nemo]] — later Mistral open-weight dense 12B model.
- [[nemotron-4-synthetic]], [[hf-cosmopedia]], [[apigen]] — use Mixtral instruct models as synthetic-data generators.
- [[fineweb]] — cites Mixtral as an open model whose pre-training data is not public.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2401.04088 (v1, the only version); https://mistral.ai/news/mixtral-of-experts (2023-12-11); https://mistral.ai/news/mixtral-8x22b (2024-04-17); https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 config.json (@fc7ac94).
- Corrections to the previous card version: "12.9B active / 46.7B total" attributed to the paper → the paper gives 13B / 47B; 12.9B / 46.7B come from the release post; "~3x inference speedup" vs 70B models → the release post states "6x faster inference" than Llama 2 70B; the paper gives no speed number; "~6x faster than a 46.7B dense equivalent" → the post compares with Llama 2 70B and says it runs at the speed and cost of a 12.9B model; "Context: 32K (sliding-window attention 4K + rotary)" and "sliding-window attention" in the abstract → "a fully dense context length of 32k tokens" (§2), config `sliding_window` null; "softmax over expert logits" → softmax over the top-K logits after masking the rest to −∞ (§2.1); "Pretrained on multilingual web data" → "multilingual data" (§1), "open Web" only in the release post; "curated instruction datasets" / "paired human-feedback data" → "an instruction dataset" / "a paired feedback dataset" (§4); "GSM8K 58.4% (vs Llama 2 70B 56.8%)" → 58.4% vs 53.6% (Table 3, 5-shot); "MT-Bench 8.3 vs GPT-3.5 Turbo 8.32" → 8.30 vs gpt-3.5-turbo-1106 8.32 (Table 3); "token-by-token expert utilization heatmap" → Figure 7 is per-domain selection proportions and Figure 8 colors tokens by first expert; "Mixtral 8x22B ... Same top-2 routing ... Same minimalist post-training (SFT + DPO)" → the 8x22B post gives 141B / 39B, 64K context, function calling, and no routing or training details; "Year: 2024 (Jan)" → arXiv v1 2024-01-08, model released 2023-12-11; "Mistral 7B base" topic → same architecture as Mistral 7B; initialization from Mistral 7B is not stated.
- Removed as unsupported by the source: auxiliary load-balancing loss; "standard completion-masked loss"; "single-stage SFT"; "single epoch of DPO"; "community replications converged on beta ~0.1, LR ~5e-7 ... matching Llama 2 / Llama 3 defaults"; "Pretraining tokens: estimated multi-T"; "What Mixtral is NOT" list and "deliberately minimalist" framing; guideline "top-2 MoE at ~12B active is the sweet spot; post-training fits on standard SFT + DPO"; claim that SFT + DPO "suffices" without iterative RLHF or dual reward models; "Mixtral 8x22B follow-up scales the same recipe"; [[qwen-2.5]] connection (not about this artifact); deps [[README]].
- Not reported by the source: pre-training token count, data shares, optimizer and LR schedule, batch size, compute; load-balancing method; SFT and DPO data sizes and hyperparameters; contamination checks.
