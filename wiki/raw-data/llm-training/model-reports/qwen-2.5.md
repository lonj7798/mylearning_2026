<!-- scope: Qwen2.5 Technical Report (Qwen Team, arXiv:2412.15115): 0.5B-72B open-weight dense models and proprietary MoE API models, 18T-token pre-training, 4K→32K long-context stage with ABF, SFT on over 1M examples, offline DPO on ~150K pairs, online GRPO; recipe values in qwen-2.5-recipe
     deps: [[dpo]], [[grpo]]
     see-also: [[qwen-2.5-recipe]], [[qwen-3]], [[deepseekmath]], [[yarn]], [[qwen-long-context-synth]], [[alibaba-qwen]]
-->

# Qwen2.5 Technical Report
- **Core Insight:** Qwen2.5 raises pre-training data from 7T to 18T tokens and post-trains with SFT on over 1 million examples, offline DPO on about 150,000 pairs, and online GRPO; Qwen2.5-72B-Instruct scores 83.1 on MATH and 81.2 on Arena-Hard against 73.8 and 69.3 for Llama-3.1-405B-Instruct (Abstract, §3.1, §4.1-4.3, Table 6).
- **Guideline:** When a model trained at 32,768 tokens must serve inputs up to 128K tokens, the report applies YaRN plus Dual Chunk Attention at inference, which raised Qwen2.5-72B-Instruct's RULER score at 128K from 67.0 to 88.4 and left scores within 32K unchanged (§3.3, Table 16).
- **Authors:** Qwen Team (byline). arXiv lists An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al.
- **Year:** 2024 (arXiv v1 2024-12-19; v2 2025-01-03)
- **URL:** https://arxiv.org/abs/2412.15115
- **Source type:** official technical report
- **Relevant topics:** pre-training data filtering and mixture, synthetic pre-training data, hyperparameter scaling laws, long-context pre-training (ABF, YaRN, DCA), SFT data construction, offline RL with DPO, reward-model evaluation, online RL with GRPO, contamination filtering

## Abstract
The report introduces the Qwen2.5 series. Pre-training data grow from 7 trillion to 18 trillion tokens. Post-training uses supervised fine-tuning with over 1 million samples and multistage reinforcement learning: offline DPO and online GRPO. Open-weight base and instruction-tuned models are released at 0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B parameters, with quantized versions. Two mixture-of-experts (MoE) models, Qwen2.5-Turbo and Qwen2.5-Plus, are proprietary and served through Alibaba Cloud Model Studio. The report states that Qwen2.5-72B-Instruct is competitive with Llama-3-405B-Instruct, and that Qwen2.5-Turbo and Qwen2.5-Plus are competitive with GPT-4o-mini and GPT-4o respectively.

## Key Contributions
- Pre-training data filtered and scored by Qwen2-Instruct models, with math and code data from Qwen2.5-Math and Qwen2.5-Coder and synthetic data from Qwen2-72B-Instruct and Qwen2-Math-72B-Instruct (§3.1).
- Scaling laws that predict optimal batch size and learning rate from model size and data size, for dense and MoE models (§3.2).
- A 4,096 → 32,768-token final pre-training stage with RoPE base 10,000 → 1,000,000, a four-stage schedule to 262,144 tokens for Qwen2.5-Turbo, and YaRN + DCA at inference (§3.3).
- Post-training in three stages: SFT over nine targeted data areas, offline RL (DPO) on verifiable domains, and online RL (GRPO) with a reward model (§4.1-4.3).
- A reward-model evaluation across four benchmarks, with the finding that RM benchmark scores did not predict the quality of the resulting RL models (§5.2.3).

## Key Figures/Tables to Study
- Table 1: layers, Q/KV heads, tied embeddings, context and generation length for each open-weight size.
- Table 2 (70B+ base) and Table 6 (70B+ Instruct): Qwen2.5-72B against Llama-3/3.1 70B and 405B.
- Table 15: Qwen2.5-RM-72B against three reward models on Reward Bench, RMB, PPE, Human-Preference-Chinese.
- Tables 16-17: RULER, LV-Eval, LongBench-Chat with and without DCA + YaRN.
- Figure 2 (Qwen2.5-Turbo 1M-token passkey retrieval) and Figure 3 (time to first token with sparse attention).

## Technical Details
**Architecture and tokenizer**
- Dense decoder with Grouped Query Attention, SwiGLU, RoPE, QKV bias, and pre-norm RMSNorm (§2). Sizes range from 24 layers and 14/2 Q/KV heads (0.5B) to 80 layers and 64/8 (72B); 0.5B, 1.5B, 3B tie input and output embeddings (Table 1).
- Context / generation length: 32K / 8K for 0.5B-3B; 128K / 8K for 7B-72B (Table 1). Licenses: Apache 2.0 except 3B (Qwen Research) and 72B (Qwen) (Table 1).
- MoE API models replace FFN layers with fine-grained and shared experts, following Qwen1.5-MoE (§2).
- Byte-level BPE tokenizer with 151,643 regular tokens; control tokens expanded from 3 to 22, two of them for tool use (§2).

**Pre-training**
- Four data changes versus Qwen2: model-based quality filtering, Qwen2.5-Math and Qwen2.5-Coder data, synthetic math/code/knowledge data filtered by reward models, and domain re-balancing (§3.1).
- Hyperparameter scaling laws fit on dense models of 44M-14B parameters and MoE models of 44M-1B activated parameters, trained on 0.8B-600B tokens; the fitted values are not given (§3.2).
- Context: 4,096 tokens in the initial phase, 32,768 in the final stage for all variants except Turbo, with RoPE base 10,000 → 1,000,000 via ABF (§3.3). Qwen2.5-Turbo trains through 32,768 → 65,536 → 131,072 → 262,144 tokens with RoPE base 10,000,000; each stage uses 40% sequences at the current maximum length and 60% shorter ones (§3.3).

**Post-training**
- SFT data areas: long-sequence generation up to 8,192 output tokens using back-translated queries; Qwen2.5-Math chain-of-thought data made with rejection sampling; Qwen2.5-Coder instruction data from multiple language-specific agents across nearly 40 programming languages, checked in a sandbox; instruction-following data validated by generated verification code and execution-feedback rejection sampling; structured data; 70,000 new logical-reasoning queries; translated cross-lingual data; hundreds of system prompts; response filtering (§4.1 (1)-(9)).
- SFT training: over 1 million examples, 2 epochs, sequence length 32,768, LR 7 × 10⁻⁶ → 7 × 10⁻⁷, weight decay 0.1, gradient clipping 1.0 (§4.1). §1 instead states that post-training data amount to 1 million examples across SFT, DPO, and GRPO; see [[qwen-2.5-recipe]].
- Offline RL: the SFT model resamples responses to new queries in math, coding, instruction following, and logical reasoning; responses that pass execution-feedback and answer-matching checks are positives and failures are negatives for DPO; about 150,000 pairs; 1 epoch with the Online Merging Optimizer (Lu et al., 2024a) at LR 7 × 10⁻⁷ (§4.2). The report does not describe the optimizer's mechanism or give DPO β.
- Reward model: labels follow six criteria (truthfulness, helpfulness, conciseness, relevance, harmlessness, debiasing); queries come from open-source and proprietary sets; responses come from SFT, DPO, and RL checkpoints at several temperatures (§4.3).
- Online RL: GRPO on the same query set as the reward model; queries with higher variance of reward-model scores are processed first; 8 responses per query; global batch 2048 and 2048 query-response samples per episode (§4.3).
- Qwen2.5-Turbo long-context fine-tuning: SFT stage 1 on instructions up to 32,768 tokens, stage 2 mixing those with instructions up to 262,144 tokens; RL on short instructions only (§4.4).

**Evaluation**
- Contamination: a training sequence is removed if its longest common subsequence with any test sequence has length ≥ 13 and ≥ 0.6 × the shorter sequence length (§5).
- Qwen2.5-72B-Instruct: MMLU-Pro 71.1, MMLU-redux 86.8, GPQA 49.0, MATH 83.1, GSM8K 95.8, HumanEval 86.6, LiveCodeBench 55.5, IFEval (strict-prompt) 84.1, Arena-Hard 81.2, MT-Bench 9.35 (Table 6). Qwen2.5-72B base: MMLU 86.1, MATH 62.1, HumanEval 59.1 (Table 2).

## Recipe ledger
Full ledger with loci and status: [[qwen-2.5-recipe]]. Summary, verified 2026-09-14: pre-training on 18T tokens with a 4,096 → 32,768 final stage and RoPE base 1,000,000 (§3.1, §3.3); SFT on over 1 million examples (conflict with §1 wording), 2 epochs, 32,768 tokens, LR 7e-6 → 7e-7, weight decay 0.1, clip 1.0 (§4.1); DPO on ~150,000 pairs, 1 epoch, LR 7e-7, Online Merging Optimizer (§4.2); GRPO with 8 responses per query, global batch 2048, 2048 samples per episode (§4.3). Not reported: all pre-training optimizer settings, SFT batch and optimizer, DPO β, GRPO LR, KL coefficient, clip ε, temperature, and maximum response length.

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality measurement:** the authors build in-house English, Chinese, and multilingual evaluations because they consider open benchmarks insufficient (§5.2.2). Qwen2.5-72B-Instruct scores 32.48 on BLEnD (cultural nuances) against 35.91 for GPT4o-mini, and the report states room for improvement there (Table 13). With hundreds of system prompts in SFT, the report states performance holds with "reduced variance" across system prompts, without numbers (§4.1 (8)).
- **Reward-model over-optimization:** Qwen2.5-RM-72B scores 91.59 on Reward Bench (Llama-3.1-Nemotron-70B-Reward: 94.10) but leads on PPE Objective-Avg with 69.85 (Table 15). The report states that optimizing for one RM benchmark may degrade others, and that "a higher score on RM benchmarks does not necessarily correlate with superior performance of the resulting RL model" (§5.2.3); no RL numbers are given.
- **Negative feedback:** negatives with negative marginal value are discarded in SFT: rejection sampling for math and instruction following, removal of incorrect reasoning, and retention of only responses that a critic model and a multi-agent scoring system all judge flawless (§4.1 (2), (4), (6), (9)). Offline RL uses negatives as gradient: responses that fail execution-feedback or answer-matching checks are the rejected side of DPO pairs, after human and automated review (§4.2). The report gives no false-negative rate and no chosen/rejected log-probability analysis. LV-Eval is scored by keyword recall to reduce false negatives in the original metric (§5.2.4).
- **Long context:** without DCA + YaRN, RULER at 128K is 31.4 (7B), 53.0 (14B), 57.7 (32B), 67.0 (72B); with them 55.1, 78.1, 82.0, 88.4 (Table 16). Qwen2.5-Turbo reaches 100% on 1M-token passkey retrieval (Fig. 2); sparse attention based on MInference reduces attention computation 12.5× at 1M tokens and gives 3.2×-4.3× time-to-first-token speedups (§5.2.4, Fig. 3). RL on short instructions alone is reported to improve long-context preference alignment, without numbers (§4.4).
- **Distillation:** synthetic pre-training data come from Qwen2-72B-Instruct and Qwen2-Math-72B-Instruct (§3.1 (3)); SFT math and code data are taken from the specialized Qwen2.5-Math and Qwen2.5-Coder pipelines (§4.1 (2)-(3)). The report does not name a teacher for other SFT areas.
- **Agentic training:** the report adds two tool-use control tokens (§2) and lists "easier tool use" as a feature (§1); it describes no agentic training data or evaluation.

## Connections
- [[qwen-2.5-recipe]] — every printed setting with locus and status.
- [[dpo]] — the offline RL loss (§4.2); [[grpo]] and [[deepseekmath]] — GRPO (Shao et al., 2024), the online RL algorithm (§4.3).
- [[yarn]] — used with Dual Chunk Attention for inference-time length extrapolation (§3.3).
- [[qwen-long-context-synth]] — covers Qwen2.5-1M (arXiv:2501.15383), a separate later report; its methods are not in this report.
- [[ruler]] — the benchmark of Table 16.
- [[llama-3]] — Llama-3/3.1 70B and 405B are the main baselines in Tables 2 and 6.
- [[qwen-3]] — the successor Qwen report; [[alibaba-qwen]] — lab page.
- [[tulu-3]], [[hunyuan-large]] — contemporary reports with SFT → DPO → RL and single-stage DPO post-training; not cited by this report.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2412.15115 (arXiv v2, full PDF; no appendix); Qwen2.5-72B-Instruct `config.json` and `README.md` on Hugging Face (main); arXiv:2405.17931 abstract for the Online Merging Optimizer.
- Corrections to the previous card version: "Total examples: 1,000,000 across SFT + DPO + GRPO stages" → §1 prints this, but the Abstract and §4.1 give "over 1 million" SFT examples and DPO uses ~150,000 pairs (§4.2); recorded as a conflict. "Two-stage context curriculum" as the general SFT recipe → it is Qwen2.5-Turbo's long-context fine-tuning (§4.4); the other models use 2 epochs at 32,768 tokens (§4.1). "Online Merging Optimizer (keeps a running merged checkpoint to stabilize DPO)" and "Qwen-specific stabilizer" → the report names the optimizer without describing it (§4.2); its paper (Lu et al., arXiv:2405.17931, abstract) merges gradients with the parameter difference between the SFT and pre-trained models to reduce alignment tax. "Group size G … not disclosed" → 8 responses per query, global batch 2048, 2048 samples per episode (§4.3). "Context: native 4K, extended to 128K" and "128K (1M with YARN extrapolation)" → 4,096 → 32,768 in pre-training with ABF; YaRN + DCA give 131,072 for open-weight models; 1M applies to Qwen2.5-Turbo (§3.3, Table 1). "Released 0.5B-72B dense + MoE variants" / "Qwen2.5-MoE" → the MoE models (Turbo, Plus) are proprietary API models (Abstract, §2). "MMLU 86.1%" → base-model score (Table 2); "HumanEval 85.4%" → 86.6; "IFEval 86.1%" → 84.1 (Table 6). "Beats Llama 3.1 70B Instruct on most benchmarks" → higher on all 13 benchmarks of Table 6; the report's headline comparison is with Llama-3-405B-Instruct (Abstract). "[[llama-3]] — main closed-weight competitor" → Llama models are open-weight baselines. "Authors: Qwen Team (Alibaba)" → byline plus arXiv authors; "Year: arXiv Dec 2024" → v1 2024-12-19, v2 2025-01-03.
- Removed as unsupported by the source: "Beta ~0.1 assumed"; "Standard completion-masked loss"; "Reward model … architecture matches policy, linear head"; the interpretation that high-variance queries are those "where the RM can discriminate strongly"; Key Figures "SFT curriculum table", "DPO preference data construction pipeline" figure, and "Reward-variance prioritization ablation" (none exist); the Core Insight's causal claim that the curriculum and stage order produce a state-of-the-art model; the Guideline to use GRPO after DPO and order prompts by variance (no ablation in the report); "Qwen2.5-Math technical report discloses math-specific RL details" (not checked in this report).
- Not reported by the source: pre-training LR, batch, schedule, and compute; per-size tokens seen; SFT optimizer, batch, packing, and loss masking; DPO β and batch; GRPO LR, KL coefficient, clip ε, temperature, maximum response length; ablations of any post-training choice.
