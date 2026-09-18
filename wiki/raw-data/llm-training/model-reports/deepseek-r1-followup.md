<!-- scope: DeepSeek-R1-0528 official model card (May 2025 post-training update of DeepSeek-R1) and its CoT distillation into Qwen3-8B; plus the R1-Distill (Jan 2025) student recipe as printed in arXiv:2501.12948 v2
     deps: [[deepseek-r1]]
     see-also: [[deepseek-v3.1]], [[deepseek-r1-distill-synth]], [[qwen-3]], [[grpo]]
-->

# DeepSeek-R1-0528
- **Core Insight:** DeepSeek-R1-0528 is a post-training update of DeepSeek-R1 whose AIME 2025 pass@1 rose from 70.0 to 87.5 while average tokens used per AIME question rose from 12K to 23K (model card §1, L58; §2 table, L84).
- **Guideline:** When distilling a small reasoning model from R1-0528 chain-of-thought, evaluate it against the same-size non-distilled model on non-math benchmarks as well, because the card's Qwen3-8B distill gains 10.0 points on AIME 2024 (86.0 vs 76.0) but scores 61.1 vs 62.0 on GPQA-Diamond (§2, L101–105).
- **Authors:** DeepSeek-AI (organization)
- **Year:** 2025 (released 2025-05-28; DeepSeek API news 2025/05/28; Hugging Face repo created 2025-05-28)
- **URL:** https://huggingface.co/deepseek-ai/DeepSeek-R1-0528 (README at repo commit 4236a6af)
- **Source type:** model/dataset card (official). Secondary locus for the R1-Distill recipe: arXiv:2501.12948 v2 (paper).
- **Relevant topics:** reasoning-model update, thinking length, evaluation protocol, chain-of-thought distillation, R1-Distill students

## Summary
The model card describes DeepSeek-R1-0528 as a "minor version upgrade" of DeepSeek R1 that improves depth of reasoning "by leveraging increased computational resources and introducing algorithmic optimization mechanisms during post-training" (§1, L52). It reports scores against DeepSeek R1 on general, code, and math benchmarks, new tool-use scores, a longer average reasoning length on AIME, a "reduced hallucination rate" (no benchmark named), enhanced function calling, and support for a system prompt (§1–§4). It also releases DeepSeek-R1-0528-Qwen3-8B, obtained by distilling R1-0528 chain-of-thought into Qwen3 8B Base (§2, L95). The card gives no training data, algorithm, or compute details.

## Key Contributions
- R1-0528 checkpoint with an evaluation table of 16 benchmark rows, 14 of which also list DeepSeek R1 (§2, L68–89).
- Reported reasoning-length change on AIME: 12K → 23K average tokens per question (L58).
- Usage changes: system prompt supported; `<think>\n` prefix no longer required (§4, L116–119).
- DeepSeek-R1-0528-Qwen3-8B: Qwen3-8B architecture with the R1-0528 tokenizer configuration (L121), AIME 2024 86.0 (L105).

## Key Figures/Tables to Study
- §2 R1 vs R1-0528 table (L68–92) and its evaluation note (L65).
- §2 R1-0528-Qwen3-8B comparison table (L97–105).
- For R1-Distill: arXiv:2501.12948 v2 Supp. Table 5 (p. 27), Table 6 (p. 35), Tables 15–16 (p. 61).

## Technical Details
**Evaluation protocol.** Maximum generation length 64K tokens; temperature 0.6, top-p 0.95, 16 responses per query to estimate pass@1 (L65). SWE-Verified uses the Agentless framework; HLE uses text-only prompts; GPT-4.1 plays the user in Tau-Bench (L92). The January R1 README used a 32,768-token limit and 64 responses per query (github.com/deepseek-ai/DeepSeek-R1 README L102), so the two protocols differ.

**R1 → R1-0528 results (L70–89).**
- General: MMLU-Redux 92.9 → 93.4; MMLU-Pro 84.0 → 85.0; GPQA-Diamond 71.5 → 81.0; SimpleQA 30.1 → 27.8; FRAMES 82.5 → 83.0; Humanity's Last Exam 8.5 → 17.7.
- Code: LiveCodeBench (2408–2505) 63.5 → 73.3; Codeforces-Div1 rating 1530 → 1930; SWE-Verified 49.2 → 57.6; Aider-Polyglot 53.3 → 71.6.
- Math: AIME 2024 79.8 → 91.4; AIME 2025 70.0 → 87.5; HMMT 2025 41.7 → 79.4; CNMO 2024 78.8 → 86.9.
- Tools (R1-0528 only): BFCL_v3_MultiTurn 37.0; Tau-Bench 53.5 (Airline) / 63.9 (Retail).

**Model and usage.** The card does not name the base checkpoint; the repository config declares `DeepseekV3ForCausalLM` (Hugging Face model API). Recommended temperature in the official web and app: 0.6 (§4, L135). License MIT; the card states the series "supports commercial use and distillation" (§5, L187).

**DeepSeek-R1-0528-Qwen3-8B (L94–105, L121).** Student base: Qwen3 8B Base. Reported pass@1: AIME 24 86.0, AIME 25 76.3, HMMT Feb 25 61.5, GPQA-Diamond 61.1, LiveCodeBench 60.5. Comparators in the same table: Qwen3-8B 76.0 / 67.3 / – / 62.0 / –; Qwen3-235B-A22B 85.7 / 81.5 / 62.5 / 71.1 / 66.5.

**R1-Distill family (January 2025; arXiv:2501.12948 v2).** Six students, SFT only, no RL stage (App. F, p. 60). Training data: the "800k" set of Supp. B.3.3 (p. 26): about 600k reasoning samples rejection-sampled from the first-stage RL checkpoint, keeping only correct responses, with DeepSeek-V3 as a generative judge for items without rule-based checks and filters for mixed languages, long paragraphs, and code blocks; plus about 200k non-reasoning samples that reuse parts of the DeepSeek-V3 SFT data. Table 5 (p. 27) counts 804,745 samples: math 395,285; code 211,129; STEM 10,124; logic 10,395; general 177,812. Hyperparameters are in Supp. B.4.3 (p. 35) and the ledger below. The canonical card for the paper is [[deepseek-r1]].

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-0528 | not stated in card | SFT + RL | data, algorithm, reward, steps, compute | not reported ("increased computational resources", "algorithmic optimization mechanisms") | HF card §1 L52 | not reported (checked: HF card, DeepSeek news 2025/05/28, arXiv:2501.12948 v2) | none |
| DeepSeek-R1-0528 | not stated | eval-gate | max generation length; sampling | 64K tokens; T 0.6, top-p 0.95, 16 samples/query | HF card §2 L65 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-0528-Qwen3-8B | 8B | distill-SFT | student base; teacher signal | Qwen3 8B Base; CoT from DeepSeek-R1-0528 | HF card §2 L95 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-0528-Qwen3-8B | 8B | distill-SFT | examples, tokens, epochs, LR, batch | not reported | HF cards of R1-0528 and R1-0528-Qwen3-8B | not reported | none |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | distill-SFT | base; initial LR | Qwen2.5-Math-1.5B; 1 × 10⁻⁴ | arXiv:2501.12948v2 Supp. B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-7B | 7B | distill-SFT | base; initial LR | Qwen2.5-Math-7B; 8 × 10⁻⁵ | same | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-14B | 14B | distill-SFT | base; initial LR | Qwen2.5-14B; 7 × 10⁻⁵ | same | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-32B | 32B | distill-SFT | base; initial LR | Qwen2.5-32B; 6 × 10⁻⁵ | same | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Llama-8B | 8B | distill-SFT | base; initial LR | Llama-3.1-8B; 5 × 10⁻⁵ | same | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Llama-70B | 70B | distill-SFT | base; initial LR | Llama-3.3-70B-Instruct; 2 × 10⁻⁵ | same | verified 2026-09-14 | no ablation reported; v1 §2.4 states Llama-3.3 was chosen because its reasoning is "slightly better than that of Llama-3.1" |
| all six R1-Distill students | 1.5B–70B | distill-SFT | epochs; LR schedule | 2–3 epochs (per-model value not given); cosine decay to one-tenth of initial LR | Supp. B.4.3 (p. 35) | verified 2026-09-14 | no ablation reported |
| all six R1-Distill students | 1.5B–70B | distill-SFT | max context; batch size | 32,768 tokens; 64 (unit not stated) | Supp. B.4.3 (p. 35) | verified 2026-09-14 | no ablation reported |
| all six R1-Distill students | 1.5B–70B | distill-SFT | examples | 800k (Table 5 total 804,745) | Supp. B.3.3 (p. 26), Table 5 (p. 27) | verified 2026-09-14 | no ablation reported |
| all six R1-Distill students | 1.5B–70B | distill-SFT | data provenance (source 1) | reasoning from first-stage RL checkpoint; non-reasoning from DeepSeek-V3 SFT data | Supp. B.3.3 (p. 26) | conflict | no ablation reported |
| all six R1-Distill students | 1.5B–70B | distill-SFT | data provenance (source 2) | "800,000 samples generated with DeepSeek-R1" | App. F (p. 60); R1 GitHub README L96 | conflict | no ablation reported |
| all six R1-Distill students | 1.5B–70B | RL | RL stage | none | App. F (p. 60) | verified 2026-09-14 | no ablation reported; App. F states RL "could substantially boost" the students and was left to others |

The two provenance rows disagree within the same paper: Supp. B.3.3 describes the reasoning subset as sampled from the first-stage RL checkpoint and the non-reasoning subset as reused DeepSeek-V3 SFT data, while App. F and the GitHub README describe all 800k samples as generated with DeepSeek-R1. Neither locus says which description matches the released students.

## Findings relevant to generality and distillation
- **Uneven gains.** Of the 14 rows with scores for both models, R1-0528 is higher on 13; SimpleQA falls from 30.1 to 27.8, while the card claims a "reduced hallucination rate" without naming a hallucination benchmark (L60, L70–86).
- **Length.** The AIME gain is attributed to "enhanced thinking depth", measured as 12K → 23K average tokens per question (L58); evaluation allows 64K tokens (L65).
- **Distilled 8B student outside math.** R1-0528-Qwen3-8B beats Qwen3-8B on AIME 24 (+10.0) and AIME 25 (+9.0) but trails it on GPQA-Diamond (61.1 vs 62.0) (L101, L105). Result (single study), no seeds or variance reported.
- **Distill vs RL at 32B.** DeepSeek-R1-Distill-Qwen-32B vs Qwen2.5-32B-Zero (large-scale RL on Qwen2.5-32B-Base for over 10K steps): AIME 2024 pass@1 72.6 vs 47.0, MATH-500 94.3 vs 91.6, GPQA-Diamond 62.1 vs 55.0, LiveCodeBench 57.2 vs 40.2 (arXiv:2501.12948v2 App. F.1, Table 16 and text, pp. 61–62).
- **Data coverage.** The paper notes the 800k set is mostly single-turn, "which may limit the multi-turn conversational capabilities" (Supp. B.3.3, p. 27).
- **Versus o1-mini.** Distill-Qwen-32B exceeds o1-mini on AIME 2024 pass@1 (72.6 vs 63.6) and cons@64, MATH-500, GPQA-Diamond, and LiveCodeBench, but not on CodeForces rating (1691 vs 1820) (R1 GitHub README L145–150).

## Connections
- [[deepseek-r1]] — the R1 paper (arXiv:2501.12948); canonical card for R1, R1-Zero, and the R1-Distill recipe.
- [[deepseek-r1-distill-synth]] — the 800k distillation corpus as a synthetic-data artifact.
- [[deepseek-v3.1]] — later DeepSeek release (API news 2025/08/21) that follows R1-0528.
- [[qwen-3]] — base family of the R1-0528-Qwen3-8B student.
- [[rejection-sampling-finetuning]] — the correct-only sampling pattern used to build the 600k reasoning subset.

## Verification
- Checked on 2026-09-14 against: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528 (README, repo commit 4236a6af, last modified 2025-05-29); https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (repo commit 6e8885a6); https://api-docs.deepseek.com/news/news250528; https://github.com/deepseek-ai/DeepSeek-R1 README (main); https://arxiv.org/abs/2501.12948 (v2, 4 Jan 2026) and v1.
- Corrections to the previous card version:
  - Title "DeepSeek R1-0528 / R1-Distill" mixed two artifacts → primary artifact is the R1-0528 model card; R1-Distill facts are cited to arXiv:2501.12948 v2 with loci.
  - "same V3 base + more compute + post-training-only improvements … no stated methodological change" → the card cites "increased computational resources" and "algorithmic optimization mechanisms during post-training" and does not name the base checkpoint (L52).
  - "applied to 6 open base models (Llama 3.1 8B, Llama 3.1 70B, Llama 3.3 70B, Qwen 2.5 1.5B/7B/14B/32B)" (seven listed) → six students: Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, Llama-3.3-70B-Instruct (Supp. Table 6).
  - "R1-Distill-{1.5B, 7B, 14B, 32B, 70B}" → the 8B Llama student was omitted.
  - "800K rejection-sampled reasoning samples" → about 600k reasoning plus about 200k non-reasoning samples; Table 5 total 804,745 with 177,812 general (Supp. B.3.3, Table 5).
  - "Hyperparameters: not disclosed; standard HF-SFT LR / batch assumed" → disclosed: 2–3 epochs, per-student initial LR, cosine decay to one-tenth, 32,768 context, batch 64 (Supp. B.4.3, Table 6).
  - "R1-Distill benchmark table … across 7 base-model recipients" → six students (Table 15).
  - "o1-competitive scores" → Distill-Qwen-32B exceeds o1-mini on five of six listed columns, not CodeForces rating (R1 README L145–150).
  - Added from the card: AIME 2025 70.0 → 87.5, 12K → 23K tokens, system-prompt support, R1-0528-Qwen3-8B (AIME 24 86.0, AIME 25 76.3).
- Removed as unsupported by the source: "R2 was anticipated for May 2025 but did not ship"; "GRPO inherited; no algorithm change disclosed"; "rule-based + model-based reward as in R1"; "updated rule-based reward shaping"; "more RL training compute" as the only change; "V3.1 merged V3 and R1 into a single hybrid checkpoint, making R1-0528 likely the final standalone R1 descendant"; "contrasts with Tülu / Llama 3 which retained RL"; "800K trace budget is relatively small; the signal is in trace quality, not volume"; "a key 2025 finding".
- Not reported by the source: R1-0528 base checkpoint, SFT data, RL algorithm, reward models, steps, compute; R1-0528-Qwen3-8B data size and hyperparameters; per-student epoch count for R1-Distill; any quality-vs-quantity ablation of the 800k set.
