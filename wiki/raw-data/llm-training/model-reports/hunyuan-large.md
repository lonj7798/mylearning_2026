<!-- scope: Hunyuan-Large technical report (Tencent, arXiv:2411.02265): 389B-total / 52B-activated MoE, 7T-token pre-training with nearly 1.5T synthetic tokens, 32K→256K long-context stages, SFT with critique-model filtering, single-stage offline+online DPO; recipe values in hunyuan-large-recipe
     deps: [[dpo]]
     see-also: [[hunyuan-large-recipe]], [[deepseek-v3]], [[qwen-2.5]], [[llama-3]], [[nemotron-4-synthetic]], [[prolong]]
-->

# Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- **Core Insight:** Hunyuan-Large is a Mixture-of-Experts (MoE) model with 389B total and 52B activated parameters, pre-trained on 7T tokens of which nearly 1.5T are synthetic; the pre-trained model scores 88.4 on MMLU against 85.2 for LLama3.1-405B, and Hunyuan-Large-Instruct scores 89.9 against 87.3 for LLama3.1-405B-Instruct (§1, Table 1, Table 3, Table 4).
- **Guideline:** When DPO is trained on pairs that a reward model selects from on-policy samples, the report adds an SFT loss on the chosen response to keep the chosen log-probability from falling and an exponential moving average to reduce reward hacking and alignment tax (§3.2); the report gives no ablation for either addition, so their effect size is not known.
- **Authors:** Tencent Hunyuan Team (byline). arXiv lists 108 authors: Xingwu Sun, Yanfeng Chen, Yiqing Huang, Ruobing Xie, Jiaqi Zhu, Kai Zhang, et al.
- **Year:** 2024 (arXiv v1 2024-11; v3 2024-11-06)
- **URL:** https://arxiv.org/abs/2411.02265
- **Source type:** official technical report
- **Relevant topics:** MoE pre-training, synthetic pre-training data, MoE scaling laws, expert-specific learning rates, KV-cache compression, long-context pre-training, SFT data construction, critique-model filtering, offline+online DPO

## Abstract
The report introduces Hunyuan-Large, a Transformer-based MoE model with 389B total parameters and 52B activated parameters that handles up to 256K tokens. It evaluates the model on language understanding and generation, logical reasoning, mathematics, coding, long-context, and aggregated benchmarks, where it outperforms LLama3.1-70B and is comparable to LLama3.1-405B. The report names four practices: large-scale synthetic data, a mixed expert routing strategy, KV-cache compression, and an expert-specific learning-rate strategy. It also studies MoE scaling laws and the learning-rate schedule. Code and checkpoints are released.

## Key Contributions
- A four-step synthetic-data pipeline for pre-training: instruction generation from seed documents, instruction evolution, response generation by specialized models, and response filtering with a critique model and self-consistency checks (§2.1.1, Fig. 1).
- Mixed routing with 1 shared expert and 16 specialized experts (top-1), plus recycle routing, which reassigns tokens of overloaded experts to randomly chosen experts with spare capacity instead of dropping them (§2.2.3, Fig. 2).
- KV-cache compression by Grouped-Query Attention (8 KV-head groups) plus Cross-Layer Attention (KV shared every 2 layers), reported as a saving of nearly 95% vs Multi-Head Attention (§2.2.2, Table 2).
- Expert-specific learning rates: specialized experts receive a lower learning rate (LR) than the shared expert, with a ratio of about 0.31 (§2.2.4).
- A compute formula for long-sequence MoE training and isoFLOP fits used to choose activated parameters and token count (§2.3.1, Eq. 2, Figs. 3-4).
- Post-training with SFT on filtered data whose volume "exceeds 1 million" (unit not stated), then one DPO stage that mixes offline and online preference data (§3).

## Key Figures/Tables to Study
- Table 1 (architecture); Table 2 (KV-cache memory by attention type).
- Figures 3-4: isoFLOP fits for optimal activated parameters and training tokens.
- Table 3 (pre-trained model) and Table 4 (Instruct model): comparisons with LLama3.1, Mixtral-8x22B, and DeepSeek-V2 / V2.5-Chat.
- Tables 5-6: RULER, LV-Eval, and PenguinScrolls long-context results.

## Technical Details
**Pre-training**
- Data are mainly Chinese and English, filtered for writing quality, educational value, and toxicity; a category-label system adjusts data proportions (§2.1.1). Synthetic data target mathematics, coding, low-resource, and high-educational-value fields (§2.1.1).
- Tokenizer: 128K vocabulary (100K tiktoken tokens + 28K Chinese tokens); characters per token rise from 2.78 with the LLama3.1 tokenizer to 3.13 (§2.1.2).
- Architecture: 64 layers, 80 attention heads, 8 KV heads, hidden size 6,400, SwiGLU, RoPE (Table 1, §2.2.1).
- Expert LR: Eq. 1 (from Li et al., 2024a) gives the optimal Adam LR ϵ_opt(B) for batch size B. The shared expert uses ϵ_opt(B). Each specialized expert sees about 1/16 of the batch, so it uses ϵ_opt(B/16) (§2.2.4).
- Compute: C ≈ 9.59·N·D + 2.3×10⁸·D, where C is training compute, N is activated parameters, and D is training tokens (§2.3.1, Eq. 2). Scaling runs used MoE models with 10M to 1B activated parameters on data sizes from 10B to 100B tokens (§2.3.1).
- Fits: N_c = 5.9×10⁻³ and α = 0.5305 give an optimum of about 58.1B activated parameters (52B chosen); D_c = 3.2 and β = 0.50 give about 5.6T tokens (about 7T chosen) (§2.3.1).
- LR schedule: warmup, a long gradual decay, then an anneal over the final 5% of tokens at one-tenth of the peak LR on the highest-quality data (§2.3.2).
- Long context: a 32K stage then a 256K stage, about 10B tokens each; natural long data from books and code (nearly 25%) mixed with normal-length data (nearly 75%); RoPE base frequency 1 billion in the 256K stage (§2.3.3).

**Post-training**
- SFT data cover mathematics, coding, logical reasoning, knowledge-based QA, agent behavior, text generation, NLP comprehension, industrial applications, role-playing, and long-text tasks; the total exceeds 1 million (§3.1.1).
- Construction: instruction-extraction models applied to public data such as web pages and encyclopedias; an instruction generalization system trained on mappings from simple to complex instructions; an instruction taxonomy with classifiers; more than 10 million instructions accumulated and then balanced with multi-dimensional labels (§3.1.2).
- Quality control: rule-based filters for truncation, duplication, garbled characters, and format errors; a critique model built on a 70B dense Hunyuan model that assigns a four-tier quality score considering accuracy, relevance, completeness, usefulness, and clarity; then human annotation (§3.1.2).
- SFT training: 3 epochs, LR decaying from 2e-5 to 2e-6, attention dropout 0.1, hidden dropout 0.2 (§3.1.3).
- DPO: a single stage that uses a pre-compiled preference dataset together with online pairs, in which the current policy generates multiple responses per prompt and the reward model selects the most and least preferred; an SFT loss on the chosen response, following Dubey et al. (2024) and Adler et al. (2024); an exponential moving average strategy (§3.2).

## Recipe ledger
Full ledger with loci and status: [[hunyuan-large-recipe]]. Summary, verified 2026-09-14: pre-training 7T tokens (Table 1); anneal over the final 5% of tokens at one-tenth of peak LR (§2.3.2); long-context stages at 32K and 256K with about 10B tokens each (§2.3.3); SFT on more than 1 million items (unit not stated) for 3 epochs, LR 2e-5 → 2e-6, dropout 0.1 / 0.2 (§3.1.3); DPO with an SFT loss and EMA (§3.2). Not reported: pre-training peak LR, warmup, batch size, and compute; SFT batch size and sequence length; DPO β, LR, pair count, and EMA decay; reward-model details.

## Findings relevant to generality, negative feedback, long context, and distillation
- **Generality (stated, not measured):** the report states that balanced instruction types "can effectively alleviate overfitting or underfitting problems on specific instruction types" (§3.1.2) and that the EMA reduces alignment tax (§3.2). It gives no ablation for either statement and no measurement of pre-trained ability retained after post-training.
- **Pre-trained vs Instruct scores (Tables 3-4):** MMLU 88.4 → 89.9, MATH 69.8 → 77.4, HumanEval 71.4 → 90.0, BBH 86.3 → 89.5, CMMLU 90.2 → 90.4; C-Eval 91.9 → 88.6 and ARC-C 95.0 → 94.6. The report does not compare the two tables, and Table 4 settings are described only as "classical evaluation settings" (§4.2.1), so these pairs are not a controlled measurement.
- **Measurement caveat:** baseline scores are the best of publicly reported and self-reproduced results (§4.1.1, §4.2.1).
- **Negative feedback:** negatives with negative marginal value are discarded at data time by critique-model scores, self-consistency checks on objective questions, and rule-based and human filters (§2.1.1, §3.1.2). DPO uses negatives as gradient: the least-preferred on-policy response is the rejected sample (§3.2). The SFT term is added because it prevents "a decrease in the log probability of chosen responses" (§3.2); no log-probability curves are shown.
- **Long context:** on RULER, Hunyuan-Large-Instruct scores 89.53 vs 86.48 for LLama3.1-70B-Instruct at 64K-128K, and 94.39 vs 95.89 at 0-8K (Table 5). LV-Eval is scored with an LLM evaluator because the original metrics had high false-negative rates (§4.3.1). PenguinScrolls overall is 85.23 vs 69.37 (Table 6). The released Instruct checkpoint supports up to 128K and the pre-trained checkpoint up to 256K (GitHub README).
- **Distillation:** synthetic pre-training responses come from "several specialized models" of different sizes (§2.1.1, Step 3); the models are not named.

## Connections
- [[hunyuan-large-recipe]] — all printed settings with loci and status.
- [[dpo]] — the preference loss used in §3.2.
- [[llama-3]], [[nemotron-4-synthetic]] — the two reports §3.2 cites for adding an SFT loss on chosen responses.
- [[rpo]] — Iterative Reasoning Preference Optimization (Pang et al., 2024), which also adds an NLL term on chosen responses; not cited by this report.
- [[prolong]] — Gao et al. (2024), cited in §2.3.3 for mixing long and normal-length data.
- [[ruler]] — the long-context benchmark in Table 5.
- [[deepseek-v3]] — a later MoE report (arXiv 2024-12); DeepSeek-V2 and DeepSeek-V2.5-Chat are the MoE baselines in Tables 3-4.
- [[qwen-2.5]] — a contemporary report whose post-training runs offline DPO and then online GRPO instead of a single DPO stage.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2411.02265 (arXiv v3, full PDF; the report has no appendix) and https://github.com/Tencent/Tencent-Hunyuan-Large README.md (main branch) for released checkpoint names and context lengths.
- Corrections to the previous card version: title "Hunyuan-Large" → exact report title; "Authors / Lab: Tencent (Hunyuan Team)" → byline plus the arXiv author list; "Core Insight: a critique model (70B dense) + single-stage online+offline DPO … is enough — no PPO, no separate RM-based RL needed" → the report makes no sufficiency claim and no comparison with PPO, and the 70B critique model filters SFT data, not preference data (§3.1.2, §3.2); "Reward model: 70B-dense Hunyuan-based RM" → the 70B dense model is the SFT critique model; the DPO reward model is not described (§3.2); "SFT loss … to prevent degradation and alignment tax" → the SFT loss prevents a decrease in chosen log-probability, and alignment-tax reduction is attributed to the EMA (§3.2); "EMA of the policy" → "an exponential moving average strategy", with the averaged quantity not stated (§3.2); "4-tier quality scale (accuracy, relevance, completeness, usefulness, clarity)" → a four-tier score that considers these five aspects (§3.1.2); "dropout … atypically high for post-training" → the report states that MoE Hunyuan models benefit more than dense models from suitable dropout (§3.1.3); "[[qwen-2.5]] — SFT + DPO+PPO peer" → Qwen2.5 uses DPO then GRPO (per [[qwen-2.5]]).
- Removed as unsupported by the source: "prior models used rule-only filters"; "single-stage hybrid DPO — novel combination"; "SFT-loss + EMA stabilization specific to this release"; "no separate PPO stage — unusual for models of this scale (most peers ran DPO → PPO or PPO directly)"; "KL handled implicitly via DPO reference anchoring"; "single-stage hybrid offers a light iterative element"; Key Figures "4-tier critique-scoring rubric", "single-stage hybrid-DPO ablation vs sequential offline+online DPO", and "MoE expert-utilization plot" (none exist in the report); "multilingual post-training details thin"; the "Innovations vs predecessors" section ("Hunyuan-7B / 13B → Hunyuan-Large", "first Hunyuan generation at 389B MoE scale") — the report names no such predecessors and instead describes an earlier closed-source trillion-parameter MoE Hunyuan model that first used the mixed routing strategy (§1, footnote 1).
- Not reported by the source: pre-training peak LR, warmup, batch size, and compute; SFT batch size, sequence length, and loss masking; DPO β, LR, pair count, responses per prompt, SFT-loss weight, and EMA decay; reward-model architecture and training data; any ablation of the post-training choices.
