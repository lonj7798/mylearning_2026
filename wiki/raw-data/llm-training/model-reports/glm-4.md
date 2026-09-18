<!-- scope: ChatGLM family report (arXiv:2406.12793): GLM-130B to GLM-4, GLM-4-Air, GLM-4-9B, and GLM-4 All Tools; ~10T-token mostly Chinese/English pre-training, SFT + RLHF alignment, long-context, function-call, agent, and safety evaluations
     deps: [[bfcl]], [[toolformer]]
     see-also: [[glm-4-5]], [[longalign]], [[agenttuning]], [[longbench]], [[qwen-2.5]], [[yi]]
-->

# ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools
- **Core Insight:** The report describes the GLM-4 series (GLM-4, GLM-4-Air, GLM-4-9B) as pre-trained on about ten trillion tokens, mostly Chinese and English, and aligned with SFT and RLHF; GLM-4 (0520) scores 87.3 on English LongBench-Chat against 87.2 for GPT-4 Turbo (1106) and 87.7 for Claude 3 Opus (Abstract, §2, Table 5).
- **Guideline:** When building SFT data for alignment, prefer authentic human prompts and interactions over template-based or model-generated responses, because the authors report across ChatGLM generations that this is "vital to the alignment quality" (§2); the report prints no ablation for this claim.
- **Authors:** Team GLM (Zhipu AI, Tsinghua University): Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, et al. (v2 list, alphabetical by first name)
- **Year:** 2024 (arXiv v1 2024-06; v2 2024-07; marked "Preprint. Under review.")
- **URL:** https://arxiv.org/abs/2406.12793
- **Source type:** official technical report
- **Relevant topics:** bilingual pre-training data, tokenizer construction, architecture choices, long-context extension and alignment, SFT and RLHF, tool use and function calling, agent evaluation, safety

## Abstract
The report introduces ChatGLM, a family of language models developed over several generations, and focuses on the GLM-4 language series: GLM-4, GLM-4-Air, and GLM-4-9B. These models are pre-trained on ten trillion tokens, mostly Chinese and English with a small corpus from 24 languages, and aligned mainly for Chinese and English through multi-stage post-training with supervised fine-tuning and learning from human feedback. The authors report that GLM-4 closely rivals or outperforms GPT-4 on MMLU, GSM8K, MATH, BBH, GPQA, and HumanEval; approaches GPT-4-Turbo on IFEval; matches GPT-4 Turbo (128K) and Claude 3 on long-context tasks; and outperforms GPT-4 on AlignBench. GLM-4 All Tools is further aligned to decide when and which tools to use (web browser, Python interpreter, text-to-image model, user-defined functions). Open releases include ChatGLM-6B (three generations), GLM-4-9B (128K, 1M), GLM-4V-9B, WebGLM, and CodeGeeX.

## Key Contributions
- A history of four ChatGLM generations with the context length and capability added at each step (§1, Figure 3, Table 1).
- The GLM-4 pre-training data pipeline and a 150,000-token vocabulary built on tiktoken cl100k_base (§2).
- Architecture choices for GLM-4: no bias except QKV, RMSNorm, SwiGLU, 2D RoPE, GQA (§2).
- An alignment summary (SFT + RLHF) with pointers to the team's technique papers: LongAlign, ChatGLM-Math, ChatGLM-RLHF, Self-Contrast, AgentTuning, APAR (§2).
- GLM-4 All Tools, a version aligned for autonomous multi-tool use (§2, §3.8, Figure 4).
- Evaluations on academic benchmarks, IFEval (English and Chinese), AlignBench, LongBench-Chat, NaturalCodeBench, BFCL, AgentBench, and SafetyBench (§3, §4).

## Key Figures/Tables to Study
- Table 1: ChatGLM-6B → ChatGLM2-6B → ChatGLM3-6B-Base → GLM-4-9B on 15 benchmarks.
- Table 5 (LongBench-Chat by language) and Table 7 (BFCL, including the size comparison GLM-4-9B-Chat vs GLM-4-Air).
- Table 8 (AgentBench per environment) and Table 9 (All Tools vs GPT-4 Web).

## Technical Details
**Model lineup.** GLM-4 (0116) has been served through the API since January 16, 2024; the latest versions are GLM-4 (0520) and GLM-4-Air (0605), with upgraded pre-training and alignment (§1). The first GLM-4 cutoff checkpoint went through multi-stage post-training (SFT, RLHF, safety alignment) and became GLM-4 and GLM-4 All Tools, both with 128K context (§1). GLM-4-Air is reported as comparable to GLM-4 (0116) with lower latency and inference cost (§1). Parameter counts of GLM-4 and GLM-4-Air are not given. GLM-4-9B is pre-trained on approximately ten trillion tokens of multilingual corpus at 8K context and post-trained with the same pipeline and data as GLM-4 (0520); GLM-4-9B-Chat-1M is an experimental 1M-context model (about 2 million Chinese characters) (§1).

**Earlier generations.** ChatGLM-6B has 6.2B parameters and was pre-trained on about one trillion Chinese and English tokens at 2,048 context, followed mostly by SFT (§1). ChatGLM2-6B improved on it by 23% on MMLU, 571% on GSM8K, and 60% on BBH; FlashAttention extended its context to 32K and Multi-Query Attention increased inference speed by 42% (§1). CodeGeeX2-6B added 600B code tokens (§1). ChatGLM3-6B added function call, code interpreter, and agent tasks (§1). The team also trained 1.5B, 3B, 12B, 32B, 66B, and 130B models to establish scaling laws (§1). GLM-130B was trained on 400B tokens (§1).

**Pre-training data.** Sources are webpages, Wikipedia, books, code, and research papers (§2). Processing has three stages: exact and fuzzy deduplication; filtering of webpages that contain offensive language, placeholder text, source code, and similar noise; and tokenization (§2). The tokenizer learns Chinese and multilingual tokens with byte-level BPE and merges them with tiktoken cl100k_base into a 150,000-token vocabulary (§2). Sources are re-weighted toward high-quality and educational sources such as books and Wikipedia, giving around ten trillion tokens (§2). For safety, text with sensitive keywords and pages on a blacklist are removed (§4). Mixture percentages are not given.

**Architecture.** GLM-4 removes all biases except those in the query, key, and value matrices, which increased training speed and slightly improved length extrapolation (§2). It uses RMSNorm and SwiGLU instead of LayerNorm and ReLU, a two-dimensional RoPE, and GQA; to keep model size, d_ffn is set to 10/3 of the hidden size (§2). GLM-130B used DeepNorm, RoPE, and GLU with GeLU (§2).

**Long context.** Context grew from 2K (ChatGLM) to 32K (ChatGLM2, ChatGLM3) to 128K and 1M (GLM-4), through position-encoding extension, continual training on long text, and long-context alignment with LongAlign (§2).

**Alignment.** GLM-4 alignment is mostly SFT and RLHF (§2). The authors state that SFT largely aligns the base model, and RLHF further helps with response rejection, safety, mixed bilingual tokens, and multi-turn coherence (§2). Alignment data for later generations combines in-house annotation and proprietary third-party data; annotators score responses on safety, factuality, relevance, helpfulness, and human preference (§2). The cited ChatGLM-RLHF work covers PPO and DPO (§2), but the report does not state which algorithm or hyperparameters produced GLM-4. Samples that pose safety risks are removed from alignment data, and harmful question-answer pairs found by a red team are corrected with human annotations and used for further alignment (§4).

**GLM-4 All Tools.** The model analyzes a request, plans steps, and calls a web browser, Python interpreter, text-to-image model (CogView3), or user-defined functions, using intermediate results (§2, §3.8). All Tools (Web, 0116) vs GPT-4 (Web, 0110): GSM8K 91.59 vs 92.72, MATH 63.60 vs 65.00, Math23K 88.50 vs 88.40, information seeking 78.08 vs 67.12 (Table 9).

**Evaluation (GLM-4 (0520)).** MMLU 83.3, GSM8K 93.3, MATH 61.3, BBH 84.7, GPQA 39.9, HumanEval 78.5 (Table 2). IFEval English strict instruction-level 85.0 (Table 3). AlignBench-v1.1 overall 8.00 (Table 4). LongBench-Chat 87.3 English, 84.0 Chinese (Table 5). NaturalCodeBench overall 47.1 (Table 6). BFCL overall 81.76 (Table 7). AgentBench overall 3.79 on 7 of 8 environments, excluding Digital Card Game (Table 8). SafetyBench Chinese subset overall 87.2 (Table 10). The §1 text compares MMLU 83.3 with "86.4 and 83.7" for GPT-4 0613 and Gemini 1.5 Pro, while Table 2 labels 86.4 as GPT-4 (0314) and gives Gemini 1.5 Pro 85.9; the source is internally inconsistent here.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GLM-4 series (GLM-4, GLM-4-Air, GLM-4-9B; not split by model) | not reported | pretrain-stable | Tokens | "ten trillions of tokens" (Abstract); "around ten trillion tokens" (§2) | arXiv:2406.12793v2 Abstract, §2 | verified 2026-09-14 | no ablation reported |
| GLM-4 series | not reported | pretrain-stable | Languages | mostly Chinese and English, plus a small corpus from 24 languages | Abstract | verified 2026-09-14 | no ablation reported |
| GLM-4 series | not reported | pretrain-stable | Tokenizer | byte-level BPE merged with cl100k_base; vocabulary 150,000 | §2 | verified 2026-09-14 | no ablation reported |
| GLM-4 | not reported | pretrain-stable | FFN width | d_ffn = 10/3 × hidden size (compensates GQA parameter reduction) | §2 | verified 2026-09-14 | no ablation reported |
| GLM-4-9B | 9B (model name) | pretrain-stable | Tokens; context length | approximately ten trillion tokens; 8192 (8K) | §1 | verified 2026-09-14 | no ablation reported |
| GLM-4 and GLM-4 All Tools | not reported | long-context | Context length | 128K | §1 | verified 2026-09-14 | no ablation reported |
| GLM-4-9B-Chat-1M | 9B (model name) | long-context | Context length | 1M (about 2 million Chinese characters); experimental | §1 | verified 2026-09-14 | no ablation reported |
| GLM-4-9B | 9B (model name) | SFT + preference | Post-training pipeline | same pipeline and data as GLM-4 (0520) | §1 | verified 2026-09-14 | no ablation reported |
| GLM-4 series | not reported | SFT, preference | Data sizes, RLHF algorithm, KL coefficient, learning rates, epochs | not printed | body §1-§5 checked; no appendix | not reported | — |
| GLM-4 series | not reported | pretrain-stable | Optimizer, LR schedule, batch size, mixture percentages, compute | not printed | body §1-§5 checked; no appendix | not reported | — |
| GLM-4 (0520), GLM-4-Air (0605) | not reported | eval-gate | Inference precision | BFloat16 | §3 | verified 2026-09-14 | — |

## Findings relevant to generality, negative feedback, long context, agentic training
- **Generality.** The authors state that "data quality and diversity are crucial" and that they have not identified a fundamental principle for data collection, cleaning, and selection (§2). Summarizing their cited work [12], they report that models of different sizes and token counts with the same pre-training loss have the same downstream performance, and that MMLU and GSM8K rise above chance only below a loss threshold (§2). Because of reported HumanEval contamination, they add NaturalCodeBench built from real user prompts (§3.5). On BFCL, overall accuracy does not increase with model size (GLM-4-9B-Chat 81.00 vs GLM-4-Air (0605) 80.94), while execution-summary accuracy increases with size (§3.6, Table 7).
- **Negative feedback.** The cited Self-Contrast method uses the target model to generate negative samples for RLHF without human preference data (§2). RLHF is used in part to reduce response rejection (§2). Unsafe alignment samples are removed, and red-team failures are corrected by human annotators and reused (§4).
- **Long context.** GLM-4 (0520) outperforms the listed baselines on the Chinese portion of LongBench-Chat (84.0 vs 82.7 for Claude 3 Opus) and is within 0.4 points of Claude 3 Opus on the English portion (Table 5).
- **Agentic training.** ChatGLM3 introduced function call and agent tasks, and the AgentTuning / AgentInstruct work is cited for agent capability (§1, §2). On AgentBench, GLM-4 (0520) is stronger on Database, House-Holding, and Web Shopping and weaker than GPT-4 models on Operating System, Knowledge Graph, and Lateral Thinking Puzzles (§3.7, Table 8).

## Connections
- [[glm-4-5]] — the next GLM report (arXiv:2508.06471), which describes GLM-4.5 as the team's first MoE model.
- [[longalign]] — the long-context alignment recipe the report credits for 128K handling (§2).
- [[agenttuning]] — the AgentInstruct trajectory method cited for agent capability (§2).
- [[longbench]] — the team's long-context benchmark cited in §2.
- [[bfcl]] — the function-call benchmark used in §3.6 (Table 7).
- [[toolformer]] — an earlier tool-use training method; this report does not cite it.
- [[qwen-2.5]] and [[yi]] — other Chinese-English open model family reports; this report does not compare against them.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2406.12793 (v2, 2024-07-30; v1 of 2024-06-18 also read; the versions differ in the author list and citation formatting)
- Corrections to the previous card version:
  - Title "GLM-4" → exact report title; "Team GLM / Zhipu AI" → Team GLM, Zhipu AI and Tsinghua University, with names.
  - "smaller amounts from 24 additional languages" → "a small set of corpus from 24 languages" (Abstract); "additional" is not in the source.
  - "`GLM-4` is ambiguous by design ... sometimes a larger closed model" → the report defines the GLM-4 language series as GLM-4, GLM-4-Air, and GLM-4-9B, with API versions (0116, 0520, 0605) and an open GLM-4-9B (Abstract, §1).
  - "The open checkpoint most users can directly inspect is GLM-4-9B-Chat" → the report lists GLM-4-9B (128K, 1M) and GLM-4V-9B among open releases (Abstract).
  - "does not publicly expose the same level of optimizer / hyperparameter detail as ... Tulu 3" → optimizer, schedule, batch, and post-training hyperparameters are not reported (see Recipe ledger).
- Removed as unsupported by the source: "26 languages" (attributed in the old card to the GLM-4-9B-Chat model card, a separate artifact; not in the report); "one of the clearest public Chinese-English family reports"; "less reproducible than Allen AI-style releases, but stronger than a pure product blog"; "The abstract is again unusually dense and useful"; "[[yi]] ... with less emphasis on tool use" (a claim about another source).
- Not reported by the source: parameter counts of GLM-4 and GLM-4-Air; optimizer and learning-rate schedule; batch size; pre-training mixture percentages; SFT and RLHF data sizes; RLHF algorithm and KL settings for GLM-4; compute.
