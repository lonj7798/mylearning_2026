<!-- scope: Magpie — instruction synthesis by sending only the pre-query chat template to an aligned open-weight LLM; filtering metrics, multi-turn, DPO, domain/multilingual extensions; SFT/DPO results on Llama-3-8B
     deps: [[self-instruct]]
     see-also: [[persona-hub]], [[evol-instruct]], [[ultrafeedback]], [[smol-talk]], [[model-collapse]], [[prismatic-synthesis]]
-->

# Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing
- **Core Insight:** When only the pre-query part of the chat template is given to Llama-3-Instruct, the model generates a user instruction; SFT of Llama-3-8B-Base on 300K filtered Magpie-Pro conversations reaches 25.08% length-controlled win rate on AlpacaEval 2 against GPT-4-Turbo (1106), above the 22.92% of the official Llama-3-8B-Instruct (Table 1).
- **Guideline:** When an aligned open-weight model is available and the target is general instruction following, generate instructions from the pre-query template and filter them, because Magpie-Pro-300K-Filtered scored higher than Magpie-Pro-1M-Raw on AlpacaEval 2 LC (25.08 vs 24.16) and Arena-Hard (18.9 vs 16.7) (Table 12); when math and reasoning also matter, add targeted reasoning data, because Magpie-only SFT trails Llama-3-8B-Instruct on GSM8K (47.92 vs 71.72, Table 3).
- **Authors:** Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, et al. (University of Washington; Allen Institute for AI)
- **Year:** 2024 (arXiv v1 2024-06, v2 2024-10; ICLR 2025)
- **URL:** https://arxiv.org/abs/2406.08464 (ICLR 2025 proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/be06e3802e9411381feece79b4d960c1-Abstract-Conference.html)
- **Source type:** paper
- **Relevant topics:** synthetic instruction data, seed-free synthesis, data filtering, multi-turn synthesis, preference data, instruction tuning

## Abstract
Alignment data for open-weight models such as Llama-3-Instruct is private, and existing open data-creation methods are limited by human labor or by a predefined prompting scope. Magpie is a self-synthesis method: an aligned LLM given only the pre-query template, up to the position of the user message, generates a user query because it is auto-regressive. The authors use it to generate 4 million instructions with responses from Llama-3-Instruct, and add extensions for filtering, multi-turn, preference-optimization, domain-specific, and multilingual data. They fine-tune Llama-3-8B-Base on Magpie data and on public datasets (ShareGPT, WildChat, Evol-Instruct, UltraChat, OpenHermes, Tulu-V2-Mix, GenQA). Magpie SFT alone surpasses public datasets used for both SFT and preference optimization, such as DPO with UltraFeedback, and on some tasks performs comparably to Llama-3-8B-Instruct, which used 10 million data points for SFT and preference optimization; the abstract names AlpacaEval, Arena-Hard, and WildBench as the benchmarks where this is seen.

## Key Contributions
- A two-step pipeline (instruction generation, response generation) with no seed questions and no prompt engineering (§2.1).
- Magpie-Air (3M, Llama-3-8B-Instruct) and Magpie-Pro (1M, Llama-3-70B-Instruct), built in 206 and 614 GPU hours (§1, §3.4); the Magpie family across Llama-3, Llama-3.1, Qwen2, Gemma-2, and Phi-3 exceeds 11.4M instances (App. A, Table 4).
- Extensions: eight filter metrics with ready-made configurations, multi-turn (MT), DPO pairs, and system-prompt control of domain and language (§2.2, App. C).
- SFT comparisons against nine public instruction datasets on Llama-3-8B-Base (Table 1), transfer to Qwen base models (Table 2), and Open LLM Leaderboard tasks (Table 3).
- Ablations on decoding parameters, system prompts, annotator model, data quantity and filtering, filter design, and response generator (App. D.3, D.4, F.3–F.5).

## Key Figures/Tables to Study
- **Figure 1** — Step 1 (pre-query template only) and Step 2 (instruction wrapped in the full template), plus the MT and DPO extensions.
- **Figure 4** — minimum neighbor distance and reward difference (r* − r_base) distributions.
- **Table 1** — SFT and SFT+DPO results vs baselines and Llama-3-8B-Instruct.
- **Table 3** — MMLU, ARC, HellaSwag, TruthfulQA, WinoGrande, GSM8K, MMLU-Redux; shows the reasoning gap.
- **Table 5 / Table 12 / Table 13** — filter configurations, quantity-vs-filtering ablation, filter-design ablation.

## Technical Details
- **Template:** input x = T_pre-query ⊕ q ⊕ T_post-query, where q is the user query. For Llama-3-8B-Instruct, T_pre-query = `<|start_header_id|>user<|end_header_id|>` and T_post-query = `<|eot_id|><|start_header_id|>assistant<|end_header_id|>` (§2).
- **Step 1:** send T_pre-query; stop at the end-of-sequence token; repeat to collect instructions (§2.1). **Step 2:** wrap each instruction in the full template and generate the response (§2.1). Instructions are generated even when instruction loss was masked during the model's alignment; the authors hypothesize implicit memorization (§2.1 Remark).
- **Instruction decoding (App. E.1, Table 7):** Magpie-Air uses temperature {1.0, 1.1, 1.2} × top-p {1.00, 0.995, 0.990} at 300K each plus temperature 1.25 × the same top-p values at 100K each (3M). Magpie-Pro uses (1.0, 1.00) 300K, (1.1, 0.995) 300K, (1.2, 0.995) 300K, (1.25, 0.990) 100K (1M). Responses use greedy decoding.
- **Decoding ablation:** higher temperature and top-p slightly lower instruction quality and raise difficulty and diversity (App. D.3, Fig. 11). Adding a Vicuna system prompt in Step 1 lowered quality and difficulty, so the authors recommend no system prompt by default (App. D.3, Fig. 12).
- **Compute (§3.4):** 4× A100 80GB, vLLM, bfloat16. Air: 1.55 h (Step 1) + 50 h (Step 2). Pro: 3.5 h + 150 h. Estimated cloud cost $0.12 (Air) and $1.1 (Pro) per 1,000 instances.
- **Annotation (§3.2):** Llama-3-8B-Instruct labels task category, input quality (very poor … excellent), and input difficulty (very easy … very hard). Minimum neighbor distance uses all-mpnet-base-v2 embeddings and FAISS. Response reward r* comes from FsfairX-LLaMA3-RM-v0.1; r_base is the reward of a Llama-3 base-model response elicited with URIAL.
- **Filters (App. C, Table 5):** eight metrics (input length, output length, task category, input quality, input difficulty, minimum neighbor distance, reward, reward difference). τ1 = −12, τ2 = 0. The output-length filter is applied last and keeps the k longest responses. Magpie-Air filter: longest, quality ≥ good, difficulty ≥ medium, min neighbor distance > 0, reward difference > τ2. Magpie-Pro has six filters producing 300K (five filters), 338K, or 200K sets. No single filter was best on all benchmarks (App. F.4, Table 13).
- **Task mix:** over half of Magpie-Pro instructions are information seeking (§3.2).
- **Multi-turn (§2.2, Fig. 14):** after turn 1, the pre-query template is appended to the full previous prompt to generate the next user turn. The 8B model occasionally forgot its user role, so a system prompt states that the user asks initial questions and follow-up related questions. Magpie-Air-MT and Magpie-Pro-MT each have 300K conversations with 2 turns and 610.80±90.61 and 554.53±133.64 tokens per turn (tiktoken) (Table 4). No check of turn-to-turn dependency is reported.
- **MT results (App. F.1, Table 10):** Pro-MT vs single-turn: AlpacaEval 2 LC 24.21 vs 25.15, Arena-Hard 20.4 vs 18.9. Air-MT vs single-turn: 22.98 vs 22.66, Arena-Hard 15.5 vs 14.9. The Pro single-turn row equals the Filter 2 row of Table 13, not the Table 1 filtered set.
- **Preference data (§2.2, §4.1):** for selected high-quality instructions, sample k responses at temperature T < 1, score with a reward model, take the highest as chosen and the lowest as rejected. Experiments use k = 5, T = 0.8, RLHFlow/ArmoRM-Llama3-8B-v0.1, and 100K conversations per DPO set.
- **Domain and language control:** a system prompt describing a domain-specific or language-specific assistant is placed before the user header (§2.2, Fig. 2, Fig. 15); domain models such as DeepSeek-Coder-V2 and Qwen2-Math-7B-Instruct can also serve as generators (§2.2).
- **Safety (App. D.2, Table 6):** Llama-Guard-2 labels 99.128% of Magpie-Air and 99.347% of Magpie-Pro as safe; the largest unsafe category is specialized advice (0.636% and 0.446%).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3-8B-Base + Magpie 300K sets (paper runs) | 8B | SFT | examples | 300K conversations (Air/Pro, raw or filtered) | arXiv:2406.08464v2 §4.1 | verified 2026-09-14 | Table 12: Pro 300K-Filtered > Pro 1M-Raw on AE2 LC and Arena-Hard |
| same | 8B | SFT | peak LR, schedule, warmup | 2×10⁻⁵, cosine, 100 warmup steps | App. E.2 Table 8 | verified 2026-09-14 | no ablation reported |
| same | 8B | SFT | epochs | 2 | App. E.2 Table 8 | verified 2026-09-14 | no ablation reported |
| same | 8B | SFT | effective batch (sequences) | 32 (4 devices × 1 × 8 grad. accum.) | App. E.2 Table 8 | verified 2026-09-14 | no ablation reported |
| same | 8B | SFT | optimizer | AdamW, β = (0.9, 0.999), ε = 10⁻⁸ | App. E.2 Table 8 | verified 2026-09-14 | no ablation reported |
| same | 8B | SFT | max sequence length; framework | 8192; Axolotl | §4.1; App. E.2 | verified 2026-09-14 | no ablation reported |
| same | 8B | SFT | packing, loss masking, weight decay | not reported | checked §4.1, App. E.1–E.2 | not reported | — |
| Llama-3-8B + Magpie-Air-DPO / Magpie-Pro-DPO (paper runs) | 8B | preference | data | 100K; k = 5 samples, T = 0.8, ArmoRM-Llama3-8B-v0.1 max/min | §4.1 | verified 2026-09-14 | Table 1: Pro SFT+DPO row LC 50.10 vs best Pro SFT row 25.08 |
| same | 8B | preference | peak LR, schedule, warmup | 5×10⁻⁷, cosine, 10% warmup ratio | §4.1; App. E.2 Table 9 | verified 2026-09-14 | no ablation reported |
| same | 8B | preference | epochs; effective batch | 1; 128 (4 × 2 × 16); Alignment Handbook | App. E.2 Table 9 | verified 2026-09-14 | no ablation reported |
| same | 8B | preference | DPO β; starting SFT checkpoint | not reported | checked §4.1, App. E.2, Table 1 caption | not reported | — |

## Findings relevant to generality, negative feedback, distillation
- **Generality (Result, single study):** among the Table 1 SFT-only runs, only Magpie-Pro-300K-Filtered exceeds Llama-3-8B-Instruct on AlpacaEval 2 LC (25.08 vs 22.92), and Magpie SFT trails it on reasoning. Open LLM Leaderboard average: Magpie-Pro-300K-Filtered 61.58 vs Llama-3-8B-Instruct 66.13; GSM8K 47.92 vs 71.72 (Table 3). The authors attribute this to the small share of reasoning instructions (§4.2, Interpretation). Adding a 150K math/code/reasoning "booster" set (Magpie-Pro-Mix-Filtered) gives GSM8K 63.08 and average 64.21 (Table 3); a gap to the official model remains (§6).
- **Scale vs filtering (App. F.3, Table 12):** Pro 300K-Filtered LC 25.08 > 1M-Raw 24.16 > 300K-Raw 21.65; for Air, 3M-Raw LC 22.96 is slightly above 300K-Filtered 22.66, while Air 300K-Filtered has the higher WR (23.99 vs 21.09).
- **Trustworthiness (App. F.6, Table 15):** vs Llama-3-8B-Instruct, the Magpie-Pro-300K-Filtered SFT model has lower jailbreak refusal (RtA 0.80 vs 0.93) and higher privacy-awareness RtA (0.71 vs 0.33).
- **Negative feedback:** the rejected DPO response is the lowest-reward of 5 samples from the generating aligned model (Llama-3-8B-Instruct for Air-DPO, Llama-3-70B-Instruct for Pro-DPO), not from the SFT model being trained; it is used as gradient through the DPO rejected term (§2.2, §4.1). Llama-3-8B SFT+DPO with Magpie-Pro-DPO reaches AE2 LC 50.10 and Arena-Hard 35.7 (Table 1). No split of the gain between chosen and rejected terms is reported.
- **Distillation / response generator (App. F.5, Table 14):** replacing Llama-3-8B-Instruct with Qwen2-7B-Instruct as the response generator for Air-300K-Filtered lowers AE2 LC from 22.66 to 15.01 and Arena-Hard from 14.9 to 13.7.

## Connections
- [[self-instruct]] — seed-based baseline; at 100K examples with Llama-3-8B-Instruct, Self-Instruct gives AE2 LC 7.21 vs Magpie-Air-100K 20.17 (Table 11).
- [[evol-instruct]], [[wildchat]] — SFT baselines in Table 1.
- [[ultrafeedback]] — the SFT+DPO baseline (UltraChat SFT + UltraFeedback DPO, AE2 LC 18.36) in Table 1.
- [[persona-hub]] — a different approach to instruction diversity (persona conditioning).
- [[smol-talk]] — a later SFT mix that uses Magpie-generated data (see that card).
- [[model-collapse]] — Magpie trains on data generated by the same model family; the paper does not test repeated training on its own outputs.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2406.08464 (arXiv v2, 2024-10-07, full text with Appendices A–H); ICLR 2025 proceedings page checked for title, authors, and abstract.
- Corrections to the previous card version:
  - "Year: 2025" → arXiv v1 2024-06; ICLR 2025 is the venue.
  - URL pointed to the ICLR proceedings page → canonical arXiv abs page; proceedings link kept.
  - "Figure 3 — minimum neighbor distance and reward-difference" → in arXiv v2 these are Figure 4; Figure 3 shows input quality and difficulty.
  - "Magpie SFT alone can beat public SFT + preference-optimization baselines on AlpacaEval 2, Arena-Hard, and WildBench" → Table 1 shows this on AlpacaEval 2 and Arena-Hard against UltraChat SFT + UltraFeedback DPO; the WildBench comparison (Fig. 5) is against baseline datasets by category.
  - "Released extensions" → the paper introduces the extensions (§2.2); MT datasets are listed in Table 4.
  - "Quantified the tradeoff between raw scale and curated quality through explicit filter recipes and cost accounting" → replaced with the Table 12 numbers and the §3.4 compute figures.
- Removed as unsupported by the source: "the data comes from a narrower teacher manifold than human-written instruction corpora" (no such measurement; the paper's t-SNE analysis reports Magpie-Pro covering the regions of Alpaca, Evol Instruct, and UltraChat, §3.1); "Magpie maximizes extraction efficiency"; "far fewer downstream training examples" (replaced by "no more than 400K vs more than 10M", §4.2).
- Not reported by the source: DPO β; SFT packing, loss masking, weight decay; the SFT checkpoint used to start DPO; dependency or coherence checks for multi-turn data; MT conversations with more than 2 turns.
