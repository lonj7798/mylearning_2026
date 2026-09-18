<!-- scope: xLAM v1 technical report — unified agent-data format, augmentation, quality verification, APIGen synthesis, and the five released xLAM checkpoints (1B to 8x22B) on BFCL v2, Webshop, ToolQuery and ToolBench
     deps: [[apigen]]
     see-also: [[apigen-mt]], [[toolace]], [[bfcl]], [[hammer]], [[granite-function-calling]]
-->

# xLAM: A Family of Large Action Models to Empower AI Agent Systems
- **Core Insight:** Unifying heterogeneous agent datasets into one function-calling format, then augmenting, verifying and mixing them with 60,000 execution-verified APIGen samples, gives five checkpoints from 1.35B to 141B parameters in which xLAM-8x22b-r reaches 87.31% overall accuracy and rank 1 on BFCL v2 (cutoff 2024-09-03), above GPT-4-0125-Preview at 85.79% (§3, §5.2.3 Table 5).
- **Guideline:** When training a function-calling model from mixed public agent data, unify the formats first and then run rule-based and LLM-judge quality verification, because in the 7B ablation augmented data beats raw data by 2.3% on ToolBench, 5.8% on Webshop and 18.3% on ToolQuery, and adding cleaning gives a further 23.4% on ToolQuery (§5.3, Fig. 3). The ablation is reported only at 7B and only on these three metrics.
- **Authors:** Jianguo Zhang, Tian Lan, Ming Zhu, Zuxin Liu, Thai Hoang, Shirley Kokane, et al. (21 authors; Salesforce AI Research)
- **Year:** 2024 (arXiv v1 2024-09; technical report)
- **URL:** https://arxiv.org/abs/2409.03215
- **Source type:** official technical report
- **Relevant topics:** function calling, agent data unification, data augmentation, BFCL, ToolBench, Webshop

## Abstract
The report describes xLAM, a series of five large action models with dense and mixture-of-experts
architectures ranging from 1B to 8x22B parameters, trained by a pipeline that unifies, augments,
synthesizes and mixes agent datasets. The pipeline converts heterogeneous agent data into one
function-calling-style format, applies prompt-format and instruction-following augmentation, detects
undefined function calls, incorrect argument types, argument hallucination and low-quality reasoning,
and adds 60,000 APIGen-synthesized samples verified by format, execution and semantic checks. The
models are evaluated on Webshop, ToolQuery, ToolBench and the Berkeley Function-Calling Leaderboard
v2, where xLAM-8x22b-r takes the top position (Abstract, §3, §5).

## Key Contributions
- A unified agent-data format with modules for task instruction, available tools, format instruction, few-shot examples, query, and steps, where each step holds the agent output, environment feedback and user follow-up (§3.1, Fig. 4).
- Prompt-format augmentation (order shuffling of tools, fields and sections; several special-token concatenation styles) and instruction-following augmentation (task-instruction rephrasing with verification; 15 output formats with converters, including JSON, XML, YAML and plain text) (§3.2).
- A four-category error taxonomy for agent data: undefined functions invoked or undefined arguments passed, incorrect argument type, argument hallucination, and low-quality reasoning and planning (§3.3).
- Five released checkpoints and their base models (§4.2, Table 1), plus the training-code stack (HuggingFace Transformers, Accelerate, PyTorch FSDP) (§4.1).

## Key Figures/Tables to Study
- **Table 1** (§4.2) — the five models with base model, parameter count, context length and category.
- **Table 5** (§5.2.3) — BFCL v2 leaderboard with per-category AST, execution and relevance-detection scores.
- **Table 3** (§5.2.1) — ToolQuery-Unified, where GPT-4o degrades under the unified output format and xLAM does not.
- **Figure 3** (§5.3) — the raw / augmented / augmented+cleaned data ablation at 7B.

## Technical Details
- Model series (§4.2, Table 1): xLAM-1b-fc-r from DeepSeek-Coder-1.3B-instruct, 1.35B total parameters, 16k context, function-calling; xLAM-7b-fc-r from DeepSeek-Coder-7B-instruct-v1.5, 6.91B, 4k, function-calling; xLAM-7b-r from Mistral-7b, 7.24B, 32k, general; xLAM-8x7b-r from Mistral-8x7b, 46.7B, 32k, general; xLAM-8x22b-r from Mistral-8x22b, 141B, 64k, general.
- Synthetic data: over 3,673 APIs across 21 categories from ToolBench are used to generate 60,000 samples via [[apigen]], with DeepSeek-V2-Chat and Mixtral-8x22B-Inst as the generating models and format, execution and semantic verification stages (§3.4).
- Data mixture: general instruction-tuning data from DialogStudio and the Data Provenance Initiative makes up 20% to 30% of the training set (§3.5). For xLAM-7b-fc-r and xLAM-1b-fc-r, 50% of training data is the synthetic function-calling set and 50% is sampled from other tasks in the training set (§3.5).
- Preference data: less powerful models generate and rate responses, a subset is human-verified, and selected responses are classified as rejected samples for DPO (§3.5).
- Training: SFT with full fine-tuning under FSDP on Nvidia H100 GPUs; LoRA is used for SFT of xLAM-8x22b-r to preserve original capacities and prevent catastrophic forgetting, and for DPO alignment across all xLAM models; cosine learning-rate scheduler with 100 warm-up steps; multiple epochs with per-epoch shuffling (§4.1).
- Function-calling categories trained for: simple, multiple, parallel, parallel multiple, plus relevance detection (§4.1).
- BFCL v2 (cutoff 2024-09-03), overall accuracy: xLAM-8x22b-r 87.31 (rank 1), xLAM-8x7b-r 83.38 (rank 6), xLAM-7b-r 80.33 (rank 14), xLAM-7b-fc-r 80.18 (rank 17), xLAM-1b-fc-r 75.43 (rank 32); GPT-4-0125-Preview (Prompt) 85.79, GPT-4o-2024-05-13 (Prompt) 83.13, Claude-3-Opus (Prompt) 80.88 (§5.2.3, Table 5).
- BFCL v2 relevance-detection columns for xLAM-8x22b-r: irrelevance 74.96, relevance 97.56 (§5.2.3, Table 5).
- Webshop and ToolQuery success rates: xLAM-7b-r 0.414 / 0.550, xLAM-8x7b-r 0.410 / 0.683, xLAM-8x22b-r 0.390 / 0.683; Mixtral-8x22b-inst 0.383 / 0.400 (§5.2.1, Table 2).
- ToolQuery-Unified: xLAM-8x22b-r success rate 0.733 against 0.683 on plain ToolQuery, while GPT-4o-2024-05-13 falls from 0.633 to 0.366, a 42% drop (§5.2.1, Table 3).
- ToolBench pass rate, unseen tools and unseen category: xLAM-7b-r 0.5850, xLAM-8x7b-r 0.5700, GPT-4-0125-preview 0.5450, AgentOhana-8x7b 0.5650. xLAM-8x22b-r results are missing because the ToolBench server was down between 2024-07-28 and the evaluation cutoff (§5.2.2, Table 4).
- The BFCL v2 live split was released after model training, so those queries were unseen by the models (§5.1, §5.2.3).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| xLAM-8x22b-r | 141B | SFT | fine-tuning method | LoRA | arXiv:2409.03215v1 §4.1 | verified 2026-09-18 | stated purpose: preserve original capacities, prevent catastrophic forgetting; no ablation reported |
| xLAM-1b-fc-r, 7b-fc-r, 7b-r, 8x7b-r | 1.35B–46.7B | SFT | fine-tuning method | full fine-tuning with FSDP | arXiv:2409.03215v1 §4.1 | verified 2026-09-18 | no ablation reported |
| all xLAM models | 1.35B–141B | preference | fine-tuning method | DPO with LoRA | arXiv:2409.03215v1 §3.5, §4.1 | verified 2026-09-18 | no ablation reported |
| all xLAM models | 1.35B–141B | SFT | LR schedule / warmup | cosine, 100 warm-up steps | arXiv:2409.03215v1 §4.1 | verified 2026-09-18 | no ablation reported |
| xLAM-7b-fc-r, xLAM-1b-fc-r | 1.35B, 6.91B | SFT | mixture share | 50% synthetic function-calling, 50% other tasks | arXiv:2409.03215v1 §3.5 | verified 2026-09-18 | no ablation reported |
| general xLAM models | 7.24B–141B | SFT | mixture share | 20–30% general instruction-tuning data | arXiv:2409.03215v1 §3.5 | verified 2026-09-18 | no ablation reported |
| all xLAM models | 1.35B–141B | SFT | synthetic data volume | 60,000 APIGen samples from 3,673+ APIs in 21 categories | arXiv:2409.03215v1 §3.4 | verified 2026-09-18 | §5.3 Fig. 3: augmentation and cleaning ablation at 7B |
| all xLAM models | 1.35B–141B | SFT | hardware | Nvidia H100 GPUs (count not given) | arXiv:2409.03215v1 §4.1 | verified 2026-09-18 | — |
| all xLAM models | 1.35B–141B | SFT / preference | peak LR, batch size, epoch count, sequence length, DPO β, LoRA rank, total training tokens, GPU-hours | not reported (checked §4.1, §4.2, §3.5; the report has no appendix) | arXiv:2409.03215v1 | not reported | — |

## Findings relevant to generality and agentic training
- Format robustness: xLAM models trained on the unified format keep their success rate when ToolQuery is presented in that format, while GPT-4o drops 42%; the authors attribute this to training on trajectories in the same format and cite concurrent work reporting reasoning degradation under output-format constraints (§5.2.1).
- Cross-size transfer: the smallest model, xLAM-1b-fc-r at 1.35B, scores 75.43 on BFCL v2, above Claude-3-Opus (FC) at 61.89 and GPT-3.5-Turbo (FC) at 75.41 (§5.2.3, Table 5).
- Held-out generalization on ToolBench is reported separately for unseen instructions, unseen tools in seen categories, and unseen tools in unseen categories; xLAM-7b-r scores highest of all listed models in the unseen-tools-and-unseen-category setting at 0.5850 (§5.2.2, Table 4).
- Forgetting is addressed only by construction, through LoRA for the 8x22B SFT run and the 20–30% general instruction share; no forgetting measurement is reported (§3.5, §4.1).
- Negative feedback appears in two places: the error taxonomy removes hallucinated and malformed trajectories from training data (negative marginal value), and DPO uses rated poor responses as rejected samples (negative as gradient) (§3.3, §3.5).

## Connections
- [[apigen]] — the verified single-turn function-calling synthesis pipeline that supplies the 60,000 samples.
- [[apigen-mt]] — the later Salesforce work that introduces multi-turn data and the xLAM-2 checkpoints, including xLAM-2-70b-fc-r; those models are not in this report.
- [[bfcl]] — the leaderboard used for the headline result (v2, cutoff 2024-09-03).
- [[toolace]] — a contemporaneous function-calling data pipeline evaluated on the same leaderboard.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2409.03215 (arXiv v1, 5 September 2024)
- Corrections to the previous card version:
  - The card mixed two artifacts. The slug and title name the xLAM v1 technical report; all xLAM-2, APIGen-MT, τ-bench and BFCL-V3 material belongs to arXiv:2504.03601 and now lives in [[apigen-mt]]. The URL line no longer carries two papers.
  - "xLAM-1B-fc-r, xLAM-7B-fc-r, xLAM-8x7B, xLAM-8x22B … through xLAM-2-70B-fc-r" → the five models in this report are xLAM-1b-fc-r, xLAM-7b-fc-r, xLAM-7b-r, xLAM-8x7b-r, xLAM-8x22b-r (§4.2, Table 1). There is no 70B model in this report.
  - "Bases: Mistral-7B, Mixtral-8x7B, Llama-3.1-70B, DeepSeek-Coder-V2-8x22B" → DeepSeek-Coder-1.3B-instruct, DeepSeek-Coder-7B-instruct-v1.5, Mistral-7b, Mistral-8x7b, Mistral-8x22b (§4.2, Table 1).
  - "xLAM-7B-fc-r: BFCL-V1 88.24% — #1 among <13B" → xLAM-7b-fc-r scores 80.18 on BFCL v2 and ranks 17th; the rank-1 model is xLAM-8x22b-r at 87.31 (§5.2.3, Table 5). The report evaluates BFCL v2, not v1.
  - "xLAM-8x22B-fc-r: BFCL-V1 ~89%" → the model is named xLAM-8x22b-r and scores 87.31 on BFCL v2 (Table 5).
  - "SFT: LR 2e-5 → 5e-6 cosine, 3 epochs, seq len 8K … DPO β = 0.1" → none of these values appear; the report gives only the cosine scheduler with 100 warm-up steps and "multiple epochs" (§4.1).
  - "Data mix … OpenOrca, WildChat subsets. Ratio roughly 40% function-calling / 60% general chat" → the named sources are DialogStudio and the Data Provenance Initiative, at 20–30% general instruction data; the 50/50 split applies to the two fc-r models only (§3.5).
  - "API registry size: ~3,673 executable APIs + tens of thousands more via APIGen-MT blueprints" → over 3,673 APIs across 21 categories from ToolBench, producing 60,000 samples (§3.4).
- Removed as unsupported by the source:
  - "xLAM-7B-fc-r hallucination on BFCL-V1 irrelevance ~4%; xLAM-2-70B on τ-bench hallucinated-tool ~2%" — no hallucination-rate numbers in this report; the closest reported quantities are BFCL irrelevance and relevance columns (Table 5).
  - "τ-bench pass^1 56.2% / pass^4 39.4%", "BFCL-V3 multi-turn ~72%", "xLAM-2-8B beats GPT-4o on τ-bench retail" — all from the APIGen-MT paper, now in [[apigen-mt]].
  - "Open weights CC-BY-NC-4.0" and "License: not drop-in for product use" — the report states no licence. It links huggingface.co/Salesforce/xLAM-models and github.com/SalesforceAIResearch/xLAM without licence terms.
  - "Unified chat template … `<tool_call>…</tool_call>` blocks", "OpenAI `tool_calls` JSON; `tool` role messages" — the report's unified format uses JSON with `thought` and `tool_calls` fields plus 15 alternative output formats with section-delimiter tokens such as "[START/END OF QUERY]" (§3.2, Fig. 4), not this template.
  - "Staged training recipe: APIGen-60k SFT → APIGen-MT SFT → optional DPO" — the report describes one SFT stage on the mixed dataset followed by DPO alignment (§4.1).
  - "General chat quality below general-purpose models of same size" and "BFCL overfit risk … community questions whether BFCL-V1 scores reflect production reliability" — not claims of this report.
  - "Teacher(s): GPT-4, Claude-3.5" — the generating models named are DeepSeek-V2-Chat and Mixtral-8x22B-Inst (§3.4), plus Mixtral-8x22b-Instruct-v0.1 and DeepSeek-V2 as judges (§3.3, §3.5).
- Not reported by the source: total SFT dataset size, learning rate, batch size, epoch count, DPO hyperparameters, LoRA rank, GPU count and training cost, and model licence.
