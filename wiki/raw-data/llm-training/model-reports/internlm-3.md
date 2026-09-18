<!-- scope: InternLM3-8B-Instruct official model card (Shanghai AI Laboratory, released 2025-01-15): 4T-token training claim, deep thinking and normal response modes, OpenCompass evaluation table, released config; no technical report exists for InternLM3
     deps: []
     see-also: [[qwen-3]], [[deepseek-r1]], [[qwen-2.5]]
-->

# InternLM3-8B-Instruct
- **Core Insight:** InternLM3-8B-Instruct is an 8B instruction model that its developers state was trained on 4 trillion high-quality tokens, "saving more than 75% of the training cost compared to other LLMs of similar scale", and that supports a long chain-of-thought deep thinking mode and a normal response mode; the release documents no training recipe for either mode (HF card README L38-L43).
- **Guideline:** When InternLM3 is cited as evidence for data efficiency or for one model serving both thinking and normal modes, treat it as an unreplicated official claim, because the card names neither the comparison models behind the 75% figure nor how the two modes were trained, and its MATH-500 and AIME2024 scores are measured in thinking mode while no baseline score carries that marker (L41, L60-L61, L71).
- **Authors:** Shanghai AI Laboratory (organization; the card names no authors, and its citation block is the InternLM2 Technical Report, arXiv:2403.17297)
- **Year:** 2025 (released 2025-01-15 per the GitHub README news entry and model table; Hugging Face repository created 2025-01-13)
- **URL:** https://huggingface.co/internlm/internlm3-8b-instruct (README.md and config.json at revision 28c99415adaf61767bd1c619f4f99f308fdfd223); release notes at https://github.com/InternLM/InternLM
- **Source type:** model/dataset card (official)
- **Relevant topics:** 8B instruction model, training-token efficiency claim, deep thinking mode, long chain-of-thought, OpenCompass evaluation, released model config

## Summary
The model card introduces InternLM3-8B-Instruct, an 8-billion-parameter instruction model "designed for general-purpose usage and advanced reasoning" (L38). It lists two characteristics. First, the model is stated to surpass Llama3.1-8B and Qwen2.5-7B on reasoning and knowledge-intensive tasks while being trained on 4 trillion high-quality tokens at more than 75% lower training cost than LLMs of similar scale (L40-L41). Second, it supports a deep thinking mode that solves complicated reasoning tasks through long chain-of-thought and a normal response mode for user interaction (L42-L43). The rest of the card is an OpenCompass evaluation table, usage code for Transformers, LMDeploy, Ollama, and vLLM in both modes, a limitations statement, and the Apache-2.0 license (L47-L442).

## Key Contributions
- Open release of code and weights for InternLM3-8B-Instruct under Apache-2.0 (L442), with quantized variants published separately (-awq, -gptq-int4, -smoothquant-int8, -smoothquant-fp8, -gguf; Hugging Face models API, read 2026-09-14).
- An official claim of 4 trillion training tokens with more than 75% training-cost saving relative to similar-scale LLMs (L41).
- One checkpoint that serves a deep thinking mode and a normal response mode (L43); in the card's examples the thinking mode is selected by a system prompt, not by a separate checkpoint (L247-L328).
- An OpenCompass comparison on 16 benchmarks in 7 categories against Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct, and GPT-4o-mini (L51-L68).

## Key Figures/Tables to Study
- Performance Evaluation table (L51-L68) and its notes on thinking-mode scores and OpenCompass version differences (L70-L72).
- The thinking-mode system prompt (L258-L304), which shows the only documented interface to the deep thinking mode.

## Technical Details
- **Architecture (config.json):** `InternLM3ForCausalLM`; 48 layers; hidden size 4,096; intermediate size 10,240; 32 attention heads; 2 key/value heads; head dim 128; `hidden_act` "silu"; vocabulary 128,512; RMSNorm eps 1e-5; no bias; untied embeddings; bfloat16.
- **Position settings (config.json):** `max_position_embeddings` 32,768; `rope_theta` 50,000,000; `rope_scaling` {"rope_type": "dynamic", "factor": 6.0}. The card gives no training sequence length.
- **Normal mode example:** a system prompt that names the model InternLM, "developed by Shanghai AI Laboratory" (L102-L104); generation with max_new_tokens 1024, temperature 1, repetition_penalty 1.005, top_k 40, top_p 0.8 (L111).
- **Thinking mode example:** a system prompt that asks the model to act as an expert mathematician and to work through deep understanding, multi-angle analysis, systematic thinking, rigorous proof, and repeated verification, to put the final answer in `\boxed{}`, and that states a budget of 8192 tokens (L258-L304); generation with max_new_tokens 8192 (L328). The vLLM example uses the same sampling values with max_tokens 8192 (L420).
- **Evaluation protocol:** OpenCompass; values marked * were evaluated in thinking mode; results "may have numerical differences due to the version iteration of OpenCompass" (L49, L71-L72). The card reports "some of the evaluation results" and points to the OpenCompass leaderboard for more (L49).
- **Selected scores, InternLM3-8B-Instruct vs Qwen2.5-7B-Instruct vs Llama3.1-8B-Instruct:** CMMLU 83.1 / 75.8 / 53.9; MMLU 76.6 / 76.8 / 71.8; GPQA-Diamond 37.4 / 33.3 / 24.2; MATH-500 83.0* / 72.4 / 48.4; AIME2024 20.0* / 16.7 / 6.7; HumanEval 82.3 / 85.4 / 72.0; IFEval prompt-strict 79.3 / 71.7 / 75.2; RULER 4-128K average 87.9 / 81.4 / 88.5; AlpacaEval 2.0 LC win rate 51.1 / 30.3 / 25.0 (L53-L66).
- **Documentation status:** the card's "Technical Report" link and citation point to the InternLM2 Technical Report (arXiv:2403.17297), not to an InternLM3 report (L26, L446-L455). The GitHub `model_cards/` directory has no InternLM3 file (GitHub contents API, read 2026-09-14).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| InternLM3-8B-Instruct | 8B | pretrain-stable | training tokens (unique vs seen not stated) | "4 trillion high-quality tokens" | HF card README L41 (rev 28c9941); GitHub InternLM README L45 (main) | verified 2026-09-14 | card states "saving more than 75% of the training cost compared to other LLMs of similar scale" (L41); comparison models and cost computation not given |
| InternLM3-8B-Instruct | 8B | pretrain-stable | architecture | 48 layers; hidden 4,096; intermediate 10,240; 32 heads; 2 KV heads; vocabulary 128,512; untied embeddings | config.json (rev 28c9941) | verified 2026-09-14 | no ablation reported |
| InternLM3-8B-Instruct | 8B | pretrain / long-context | data mixture; LR schedule; batch size; training sequence length; compute | not given | HF card README, config.json, GitHub README (EN and zh-CN), GitHub model_cards/ checked | not reported | — |
| InternLM3-8B-Instruct | 8B | SFT / preference / RL | SFT data and size; preference or RL algorithm; reward model or verifiers; thinking-mode training data; how the two modes were combined; all hyperparameters | not given | same artifacts checked | not reported | — |

## Findings relevant to generality and long context
- **Generality:** the card describes the model as "designed for general-purpose usage and advanced reasoning" (L38) and covers General, Reasoning, MATH, Coding, Instruction, Long Context, and Chat categories (L53-L68). Among the open models in the table, InternLM3-8B-Instruct is not highest on MMLU (76.6 vs 76.8), HumanEval (82.3 vs 85.4), or RULER (87.9 vs 88.5) (L54, L63, L65). The card reports no contamination check and no held-out evaluation beyond this table.
- **Measurement caveat:** the two math scores marked * use thinking mode, while the baseline scores and the other InternLM3 scores carry no marker (L60-L61, L71), so the math comparison mixes inference modes.
- **Long context:** RULER 4-128K average is 87.9 (L65). The released config sets `max_position_embeddings` 32,768 with dynamic RoPE scaling factor 6.0 (config.json); the card does not describe long-context training.

## Connections
- [[qwen-3]] — a report (arXiv 2025-05) that documents training one model for thinking and non-thinking modes; the InternLM3 card documents no such training.
- [[deepseek-r1]] — a report (arXiv 2025-01) that documents long chain-of-thought training; the InternLM3 card does not state how its deep thinking mode was trained.
- [[qwen-2.5]], [[llama-3]] — Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct are the open baselines in the card's table.
- [[ruler]] — the long-context benchmark in the card's table.
- Not in this library: InternLM2 Technical Report (arXiv:2403.17297), which the card links; Intern-S1, which its own model card describes as built on a 235B Qwen3 MoE language model and a 6B InternViT vision encoder.

## Verification
- Checked on 2026-09-14 against: https://huggingface.co/internlm/internlm3-8b-instruct README.md and config.json (revision 28c99415adaf61767bd1c619f4f99f308fdfd223); https://github.com/InternLM/InternLM README.md and README_zh-CN.md (main); GitHub `model_cards/` listing; https://github.com/InternLM/InternLM-techreport README.md; https://huggingface.co/internlm/Intern-S1 README.md (for the Intern-S1 base model only).
- Corrections to the previous card version: title "InternLM3" → "InternLM3-8B-Instruct" (the released model; no InternLM3 report exists); URL "github.com/InternLM/InternLM-techreport" → that repository holds the 2023 technical report for the original 104B InternLM trained on 1.6T tokens (repository README abstract), so the primary artifact is the Hugging Face model card; "4T training tokens (>75% less than same-scale peers)" → the card states 4 trillion tokens and "more than 75%" lower training cost, not 75% fewer tokens (L41); "Core Insight: 4T tokens + hybrid post-training beat larger-token budgets at similar scale — efficiency per token matters more than absolute count" → the card gives no token counts for other models and no controlled comparison (L41); "used as a base for the Intern-S1 scientific multimodal follow-up" → Intern-S1 is built on a 235B Qwen3 MoE model and a 6B InternViT encoder (Intern-S1 model card); "Full technical recipe reportedly documented in the InternLM-techreport GitHub" → that repository documents the 2023 InternLM, not InternLM3.
- Removed as unsupported by the source: the "Intelligence Quality per Token (IQPT)" framing (not found in the model card, GitHub README in English or Chinese, or config; a press release was not checked); "Guideline: at 7B-8B scale, don't chase 15T tokens"; "15T+ in contemporaries"; "preceded Qwen3's public rollout of the same pattern" and "Jan 2025 vs May 2025"; "first InternLM release with explicit deep-thinking mode toggle"; "4T-token training — smaller than prior generations' token counts relative to performance"; "same Mixture-of-Reward methodology likely underlies InternLM3"; "verifiable rewards implied by thinking-mode training on math/code"; Key Figures "benchmark vs peers at 4T vs 15T+ tokens" and "hybrid-mode ablation".
- Not reported by the source: every pre-training and post-training setting other than the 4T token count; the identity of the comparison models behind the 75% cost figure; whether a base (non-instruct) InternLM3 checkpoint exists (none appears in the Hugging Face models API search for "internlm3" under the internlm organization, read 2026-09-14).
