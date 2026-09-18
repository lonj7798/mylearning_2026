<!-- scope: Orca 2 (Microsoft Research, Nov 2023) — task-specific reasoning strategies elicited from GPT-4, Prompt Erasing, progressive SFT of LLaMA-2 7B/13B, and a 15-benchmark zero-shot evaluation
     deps: [[orca]]
     see-also: [[distilling-step-by-step]], [[agentinstruct]], [[dolphin]], [[phi-3]]
-->

# Orca 2: Teaching Small Language Models How to Reason
- **Core Insight:** Progressive fine-tuning of LLaMA-2-13B that ends on 1M Orca 1 GPT-4 responses plus ~817K teacher responses elicited with task-specific strategy instructions, which are replaced by one generic system message at training time (Prompt Erasing), gives Orca-2-13B a zero-shot macro-average of 64.49 on six reasoning benchmarks, against 56.85 for LLaMA-2-Chat-70B and 43.71 for LLaMA-2-Chat-13B (Fig. 4, Table 1; §6.1).
- **Guideline:** When distilling a small model from a stronger teacher, select the teacher's solution strategy per task and train the student under a generic system message, because Orca-2-13B trained this way exceeds LLaMA-2-Chat-70B on the six-benchmark zero-shot reasoning average (64.49 vs 56.85, Fig. 4); the paper reports no ablation that isolates Prompt Erasing, and seven of its benchmarks are in-domain (§5.3).
- **Authors:** Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, et al. (Microsoft Research)
- **Year:** 2023 (arXiv v1 2023-11; v2 2023-11-21)
- **URL:** https://arxiv.org/abs/2311.11045 ; model card https://huggingface.co/microsoft/Orca-2-13b
- **Source type:** paper (with model card)
- **Relevant topics:** reasoning distillation, explanation tuning, strategy selection, prompt erasing, progressive learning, small-model evaluation, hallucination, safety

## Abstract
Orca 1 learned from explanation traces and outperformed instruction-tuned models on BigBench Hard and AGIEval. Orca 2 argues that pure imitation of a larger model can limit a smaller model, because the best solution strategy for a small model may differ from the teacher's. The authors teach the model several reasoning techniques (step-by-step, recall then generate, recall-reason-generate, direct answer, and others) and aim to teach it which technique to use for each task. On 15 benchmarks (about 100 tasks, over 36K unique prompts), Orca 2 surpasses models of similar size and performs similar to or better than models 5–10× larger on zero-shot reasoning tasks. The weights are released for research.

## Key Contributions
- "Cautious Reasoning": selecting a solution strategy per task (direct answer or a slow-thinking strategy such as step-by-step or explain-then-answer) (§3).
- Prompt Erasing: the teacher receives detailed, task-specific system instructions, possibly across multiple calls; the student sees only the task and a generic system instruction (§1, §3).
- An ~817K-instance Orca 2 dataset built from FLAN-v2 tasks, few-shot examples, math problems, and synthetic doctor–patient conversations (§4.1).
- A zero-shot evaluation across reasoning, knowledge, text completion, multi-turn chat, grounding, and safety, with in-domain vs out-of-domain labeling (§5).

## Key Figures/Tables to Study
- Figure 3 (four GPT-4 answers to one story-reordering task under different system instructions); Table 1 and Figure 4 (reasoning); Table 2 (MMLU, ARC); Table 3 (MT-Bench); Table 11 (hallucination rates); Tables 4–5 (automated safety framework).

## Technical Details
**Strategy selection and Prompt Erasing (§3)**
1. Start from a collection of diverse tasks.
2. Decide which strategy each task needs, "guided by the performance of Orca" (Orca 1).
3. Write task-specific system instructions for that strategy and collect teacher responses.
4. At training time, replace the system instruction with a generic one that contains no strategy details.
- In Figure 3, GPT-4 answers a five-sentence story-reordering task correctly only with a five-step task-specific instruction; the default, chain-of-thought, and explain-your-answer instructions give wrong orders (§3, Fig. 3).
- The training-time system message (the "cautious system instruction") begins "You are Orca, an AI language model created by Microsoft. You are a cautious assistant." (§4.1).

**Dataset (§4.1)**
- FLAN-v2: from the CoT, NiV2, T0, and Flan 2021 sub-collections (1,913 tasks), ~602K zero-shot queries from 1,448 selected tasks, grouped by hand into 23 categories and 126 sub-categories; one system instruction per sub-category.
- Few-shot data: 55K samples built from Orca 1 zero-shot data with 3–5 in-context (prompt, answer) pairs from the same task and system instruction.
- Math: ~160K problems from DeepMind Math (arithmetic tasks only) and training splits of GSM8K, AquaRat, MATH, AMPS, FeasibilityQA, NumGLUE, AddSub, GenArith, and Algebra.
- Fully synthetic: 2,000 GPT-4 doctor–patient conversations with four-section summaries, used "to assess the learning of specialized skills".
- The paper does not state a correctness filter for teacher responses. The model card states that all synthetic training data were moderated with Azure content filters.

**Training (§4.2)**
- Progressive learning from LLaMA-2-7B or LLaMA-2-13B: FLAN-v2 train split (1 epoch) → 5M ChatGPT responses from Orca 1 (3 epochs) → 1M GPT-4 responses from Orca 1 plus the 817K Orca 2 set (4 epochs).
- Tokenizer: LLaMA BPE plus a padding token ("PAD" in double square brackets) and ChatML tokens `<|im_start|>`, `<|im_end|>`; vocabulary 32,003.
- Packing: examples are shuffled and grouped so each concatenation is at most 4,096 tokens, then padded to 4,096.
- Loss is computed only on teacher-generated tokens.

**Evaluation settings (§5.3)**
- All benchmarks zero-shot, greedy decoding, empty system message for all models; Orca 2 is also reported with the cautious system message. Answers are extracted with pattern lists and regular expressions; a format-OK rate is reported.

**Results**
- Reasoning, Table 1 (AGIEval / BBH / DROP / CRASS / RACE / GSM8K): Orca-2-13B 49.93 / 50.18 / 57.97 / 86.86 / 82.87 / 59.14; Orca-2-7B 45.10 / 45.93 / 60.26 / 84.31 / 80.79 / 47.23; LLaMA-2-Chat-70B 46.70 / 44.68 / 54.11 / 74.82 / 68.79 / 52.01; WizardLM-70B 48.73 / 51.08 / 59.62 / 86.13 / 78.96 / 73.24. Orca-2-13B improves 47.54% (relative) over LLaMA-2-Chat-13B and 28.15% over WizardLM-13B (§6.1).
- Cautious system message: reasoning macro-average 66.92 (13B) and 62.62 (7B) vs 64.49 and 60.60 with the empty message (Fig. 4).
- MMLU: Orca-2-13B 57.73 vs LLaMA-2-Chat-70B 58.54; ARC-Challenge 83.36 vs 67.66 (Table 2).
- MT-Bench average: Orca-2-13B 6.15, LLaMA-2-Chat-13B 6.64, WizardLM-70B 7.76; the lower turn-2 score is attributed to the absence of conversations in training data (Table 3, §6.4).
- Hallucination rate (GPT-4 judge, average of ACI-BENCH, MS MARCO, QMSum): Orca-2-13B 10.97; with cautious system message 29.00; LLaMA-2-Chat-13B 47.53; GPT-4 2.80 (Table 11).
- Automated safety framework, defect rates Orca-2-13B vs LLaMA-2-Chat-13B: Violent 13.47% vs 0.17%; jailbreak "Leaking Guidelines" 24.24% vs 70.00% (Tables 4–5). Orca 2 has no RLHF safety training (§1, §6.6).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Orca-2-7B / Orca-2-13B | 7B, 13B | SFT | base checkpoint | LLaMA-2-7B / LLaMA-2-13B | arXiv:2311.11045v2 §4.2 | verified 2026-09-14 | no ablation reported |
| Orca-2-7B / -13B | 7B, 13B | SFT | stage 1 data; epochs | FLAN-v2 train split; 1 | v2 §4.2 | verified 2026-09-14 | no ablation reported |
| Orca-2-7B / -13B | 7B, 13B | distill-SFT | stage 2 data; epochs | 5M ChatGPT responses (Orca 1); 3 | v2 §4.2 | verified 2026-09-14 | no ablation reported |
| Orca-2-7B / -13B | 7B, 13B | distill-SFT | stage 3 data; epochs | 1M GPT-4 responses (Orca 1) + ~817K Orca 2 instances ("~1.8 million GPT-4 data"); 4 | v2 §4, §4.2 | verified 2026-09-14 | no ablation reported |
| Orca 2 dataset | — | distill-SFT | composition | FLAN ~602K; few-shot 55K; math ~160K; synthetic conversations 2,000 | v2 §4.1 | verified 2026-09-14 | no ablation reported |
| Orca-2-7B / -13B | 7B, 13B | SFT | training system message | generic "cautious system instruction" (Prompt Erasing) | v2 §3, §4.1 | verified 2026-09-14 | Fig. 4: cautious message at inference +2.43 (13B), +2.02 (7B) reasoning average |
| Orca-2-7B / -13B | 7B, 13B | SFT | sequence length; packing | 4,096; packed, padded to 4,096 | v2 §4.2 | verified 2026-09-14 | no ablation reported |
| Orca-2-7B / -13B | 7B, 13B | SFT | loss masking | loss on teacher-generated tokens only | v2 §4.2 | verified 2026-09-14 | no ablation reported |
| Orca-2-13B | 13B | SFT | compute | 32 × A100 80GB, bfloat16; ~17 h (FLAN, 1 epoch), ~40 h (5M ChatGPT, 3 epochs), ~23 h (~1.8M GPT-4, 4 epochs) | v2 §4.2 | verified 2026-09-14 | — |
| Orca-2-7B / -13B | 7B, 13B | SFT | learning rate, schedule, warmup, batch size, optimizer, weight decay, checkpoint selection | not reported | checked v2 §4.2, appendices A–F, HF model card microsoft/Orca-2-13b | not reported | — |

## Findings relevant to generality, negative feedback, distillation
- Measurement: seven benchmarks are in-domain because their training splits are in the data; the rest are treated as out-of-domain, but LLaMA-2 pre-training leakage cannot be ruled out (§5.3).
- Narrowing: the authors expect performance to correlate with the tuning-data distribution, with lower accuracy in underrepresented areas such as math and coding; results vary with the system message; the training data mostly simulate zero-shot settings, and the model does not show the few-shot gains of larger models (§7, §6.2).
- Knowledge limit: post-training "does not necessarily teach the model new knowledge"; the authors suggest small models serve as reasoning engines over knowledge in the context window (§2.1, §7).
- Grounding: the authors state that the cautious system message consistently increases the hallucination rate; Table 11 shows this for Orca-2-13B on all three tasks (average 10.97 → 29.00), while Orca-2-7B on ACI-BENCH falls from 27.45 to 21.26. Manual analysis finds the model extrapolating beyond the context (§6.5, Table 11).
- Specialization: 5,000 story-reordering samples produced with the Figure 3 instruction, trained with prompt erasing, are used to test task specialization on ROCStories with FLAN-overlapping items removed (§6.7, Fig. 12).
- Negative samples: the paper describes no use of incorrect teacher outputs. In Figure 3, wrong GPT-4 answers under generic instructions motivate choosing a different instruction rather than training on them (§3).

## Connections
- [[orca]] — Orca 1 explanation tuning; its 5M ChatGPT and 1M GPT-4 responses form stages 2–3 of Orca 2 training.
- [[distilling-step-by-step]] — earlier rationale distillation into small models.
- [[agentinstruct]] — later Microsoft Research work with the same first author; trains Orca-3 on agent-generated data with Orca-2.5 as the control.
- [[dolphin]] — named in §2.2 as a model built with Explanation Tuning.
- [[phi-3]] — related Microsoft synthetic-data post-training line.

## Verification
- Checked on 2026-09-14 against: arXiv:2311.11045v2 (2023-11-21), main text and appendices; HF model card microsoft/Orca-2-13b.
- Corrections to the previous card version: "Sahaj Agrawal" → Sahaj Agarwal (title page); "teaches when to abstain … acknowledge uncertainty … cautious or selective answering" → "Cautious Reasoning" in the paper means choosing a solution strategy per task (§3); the paper does not train abstention or selective answering; "see-also [[limo]], [[s1]]" as "conceptually aligned curated-small-set recipes" → removed, since Orca 2 uses ~817K instances plus earlier stages (§4), not a small curated set.
- Removed as unsupported by the source: "practical contribution is a better-targeted student curriculum rather than a radically new algorithm" (card interpretation, not a source statement); "small models can become more reliable reasoners" (the paper reports benchmark scores, not a reliability measure); unquantified "improving reasoning quality without frontier-scale size" (replaced by Table 1 and Fig. 4 numbers).
- Not reported by the source: optimizer and LR settings, batch size, teacher-response correctness filtering, 7B training time.
