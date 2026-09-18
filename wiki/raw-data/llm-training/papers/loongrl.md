<!-- scope: LoongRL (Microsoft Research Asia, Oct 2025) — KeyChain synthesis of hard long-context multi-hop QA from short QA, GRPO with a two-way substring reward, 16K training that is evaluated up to 128K, and short-context retention on Qwen2.5-7B/14B-Instruct
     deps: [[grpo]], [[ruler]]
     see-also: [[qwenlong-l1]], [[qwenlong-l1-5]], [[nextlong]], [[retrieval-head]], [[longbench-v2]]
-->

# LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
- **Core Insight:** GRPO on a mix of 21,024 examples (sum of Table 1) whose hardest part is 7,500 KeyChain items (short multi-hop QA padded to about 16K tokens with a hidden question reached through UUID key-value chains) raises the LongBench v1 multi-hop QA average of Qwen2.5-7B-Instruct from 48.9 to 72.4 and of Qwen2.5-14B-Instruct from 53.1 to 74.2 (Table 2).
- **Guideline:** When long-context RL rollouts at the target length are too expensive, train at 16K on questions that require locating the question itself among distractors before multi-hop reasoning, and keep short math and retrieval data in the mix, because in this study the KeyChain run beat an equal amount of plain long multi-hop QA (72.4 vs 66.2, Table 4) and MMLU/MATH/IFEval moved by at most 2.8 points (Table 2).
- **Authors:** Siyuan Wang, Gaokai Zhang, Li Lyna Zhang, Ning Shang, Fan Yang, Dongyao Chen, Mao Yang (Microsoft Research Asia; Shanghai Jiao Tong University; Carnegie Mellon University)
- **Year:** 2025 (arXiv v1 2025-10; preprint, no venue listed)
- **URL:** https://arxiv.org/abs/2510.19363
- **Source type:** paper
- **Relevant topics:** long-context RL, synthetic long-context data, distractor documents, GRPO, rule-based reward for free-form QA, curriculum and difficulty filtering, length generalization, short-context retention

## Abstract
The paper targets reasoning over long inputs. RL has improved short-context reasoning, but the thinking patterns needed for long-context reasoning are unexplored and high-difficulty long-context RL data are scarce (Abstract); rollouts at 128K are also expensive in compute and memory (§1). It introduces KeyChain, which turns short multi-hop QA into long-context tasks by inserting distracting documents and UUID key-value chains that hide the true question among distracting questions. Solving an item requires tracing the correct chain, recovering the question, retrieving facts, and reasoning over them. RL on this data produces a plan–retrieve–reason–recheck pattern in model outputs. Models trained at 16K are evaluated up to 128K. On Qwen2.5-7B and 14B the method gives +23.5 and +21.1 absolute points on long-context multi-hop QA; LoongRL-14B scores 74.2 against o3-mini at 74.5 and DeepSeek-R1 at 74.9. The authors report improved long-context retrieval, passing all 128K needle-in-a-haystack tests, and preserved short-context reasoning (Abstract).

## Key Contributions
- KeyChain data construction: difficulty-filtered real multi-hop QA, padded with real documents, with a hidden question reachable only through a correct UUID chain (§3.1, Fig. 2).
- Two-way substring exact match reward for free-form answers in `\boxed{}` (§3.2.1, Eq. 3), compared against F1, LLM-as-a-judge, and exact match (Table 5).
- A mixed RL dataset of four data types and a multi-stage curriculum ending in hard-example mining (Table 1, §3.2.2).
- Evidence that 16K-trained models improve at 32K–128K on NarrativeQA, RULER, and NIAH (Table 3, Table 7, Fig. 3).

## Key Figures/Tables to Study
- **Fig. 2** and **App. A.3 Fig. 6** — KeyChain construction and a real training-item skeleton.
- **Table 1** — the RL data recipe with sizes, length ranges, and difficulty labels.
- **Table 2** — long-context QA and MMLU/MATH/IFEval for LoongRL, R1-distilled models, and QwenLong-L1-32B.
- **Table 3 / Table 7** — generalization beyond 16K (NarrativeQA by length bucket; RULER 4K–128K).
- **Tables 4 and 5** — ablations of KeyChain data and of the answer verifier (7B).
- **App. A.4–A.5** — output traces with and without KeyChain training.

## Technical Details
### KeyChain data
- Seed pool: HotpotQA, MuSiQue, and 2WikiMultiHopQA, 277K QA instances (§3.1). Each question is answered eight times by Qwen2.5-32B-Instruct; items with pass rate 0 or 1 are discarded, leaving 72K (§3.1).
- Context extension: extra documents are sampled from the short contexts of the 200K filtered-out tasks, excluding overlap with the item's own context; each extended context is approximately (<) 16,384 tokens (§3.1).
- Chains: in each chain a key maps to a value that contains the next key. One chain resolves to the original question; multiple chains resolve to distracting questions sampled from other QA instances. Each key is a 32-character UUID with characters from 0-9 and A-F; key-value pairs are inserted at random positions (§3.1, Fig. 2).
- The prompt tells the model that one correct question exists, gives the starting UUID, and asks it to find the question first and then answer it (§3.1 example; App. A.3).
- The paper does not report the number of distracting chains per item or chain length (not reported).

### Reward and RL objective
- GRPO with group-normalized advantage A = (r − mean)/std over the group (Eq. 1–2). KL penalty β = 0.001; the entropy loss term is removed (§3.2.1).
- Reward r ∈ {0, 1}: 1 if the ground truth is a substring of the extracted boxed answer or the boxed answer is a substring of the ground truth (Eq. 3). The system prompt requires `<think> ... </think>` followed by `\boxed{}` (App. A.2).
- Verifier ablation on 7B, LongBench v1 average: F1 65.1, LLM-as-a-judge (DeepSeek-V3) 65.2, exact match 69.2, two-way substring 72.4 (Table 5). QA answers "can take many valid forms", which is the stated reason for not using strict exact match (§3.2.1).

### Data mix and schedule
- Data mix (Table 1; §3.2.2): KeyChain 3 × 2,500 = 7,500 (lengths 14,911–20,670, "Hard"); plain multi-hop QA 3 × 2,500 = 7,500 (10,727–16,283, "Medium"); RULER-style multi-key and multi-value retrieval on 16K-token PG19 books, 512 + 512 = 1,024; math 5,000 (2,500 DAPO training problems, "Hard", and 2,500 multiple-choice MATH problems, "Easy"; <1K tokens).
- Curriculum (§3.2.2): warm-up for one epoch without KeyChain data; Stage I adds KeyChain data; Stage II keeps only items not solved in all 8 rollouts of the best Stage I checkpoint, which leaves 30–40% of the data. The 14B run skips warm-up (§4.1).
- Training settings, step counts, and compute are in [[loongrl-recipe]].

### Evaluation protocol
- Long-context reasoning: LongBench v1 HotpotQA, 2WikiMultiHopQA, MuSiQue, NarrativeQA, QASPER, inputs 4K–64K; LongBench v2 in App. A.7 (§4.1).
- Reasoning models and LoongRL: temperature 0.6, up to 128K input and 10K output tokens, 8 samples, average pass@1. Non-reasoning models such as Qwen2.5-7B-Instruct use temperature 0 (§4.1).
- Qwen-family models and LoongRL use YaRN for 64K and 128K RULER and for LongBench v2 (Table 7 caption; App. A.7).

## Findings relevant to generality, negative feedback, long context
- **Long-context gains.** 7B: 48.9 → 72.4; 14B: 53.1 → 74.2. On the same average R1-Distill-Qwen-7B is 17.7 points below Qwen2.5-7B-Instruct, R1-Distill-Qwen-14B is 11.8 points above Qwen2.5-14B-Instruct, and QwenLong-L1-32B is 4.6 points above R1-Distill-Qwen-32B (§4.2, Table 2). Result (single study).
- **Length generalization.** NarrativeQA 32K–64K: 42.4 → 57.2 (7B) and 48.3 → 64.3 (14B) (Table 3). RULER 128K with YaRN: 69.41 → 76.84 (7B) and 73.57 → 79.92 (14B) (Table 7). LoongRL-7B reaches 100% on NIAH across depths up to 128K (Fig. 3); for 14B, App. A.9 states "strong retrieval accuracy" (Fig. 9). Result (single study).
- **LongBench v2.** Overall 31.2 → 36.2 (7B) and 35.3 → 42.3 (14B); 7B MultiDoc QA drops 23.2 → 21.6 (Table 6).
- **Short-context retention.** MMLU +2.8 (7B) and +1.1 (14B); MATH-500 76.0 → 78.0 and 83.4 → 83.2; IFEval −0.3 and −2.6. R1-Distill-Qwen-7B/14B score 16.5 and 8.4 IFEval points below Qwen2.5-7B/14B-Instruct, and QwenLong-L1-32B has lower MMLU than R1-Distill-Qwen-32B (78.5 vs 80.5) (§4.2, Table 2). Result (single study).
- **Measurement notes.** Three of the five LongBench v1 tasks come from the same datasets as the RL seeds; the paper does not state the seed split or report an overlap check (not reported). Baselines without long CoT are decoded at temperature 0 while LoongRL is averaged over 8 samples at 0.6 (§4.1).
- **Negatives.** Wrong rollouts get reward 0 and, in a mixed group, a negative normalized advantage (negative as gradient; Eq. 2–3). Items with pass rate 0 or 1 are removed before RL (§3.1), and all-correct items are removed before Stage II (§3.2.2). The paper does not measure the contribution of negative advantages.
- **Behavior change.** Without KeyChain data, outputs mix retrieval and reasoning and often lack a planning step (Fig. 1b; App. A.4). This is the authors' qualitative reading of example traces. Interpretation.

## Connections
- [[grpo]] — the RL algorithm used (Eq. 1–2).
- [[qwenlong-l1]] — long-context RL baseline (R1-Distill-Qwen-32B, 60K input, LLM-judge reward) compared in Tables 2, 3, 6, 7.
- [[qwenlong-l1-5]] — QwenLong-L1.5 post-training recipe for long-context reasoning; compare its data synthesis and reward choices (not cited by this paper).
- [[dapo]] — source of the 2,500 hard math problems in the mix (Table 1).
- [[ruler]] — format of the 1,024 retrieval training items and a retrieval evaluation (§3.2.2, Table 7); [[needle-in-haystack-data]] — NIAH evaluation (§4.1, Fig. 3).
- [[longbench]] and [[longbench-v2]] — the main long-context reasoning evaluations (§4.1, App. A.7).
- [[yarn]] — context extension applied at 64K/128K evaluation (Table 7).
- [[qwen-2.5]] — base instruct models; [[deepseek-r1]] — R1-distilled baselines.
- [[nextlong]] — another distractor-insertion method, applied to pretraining-style long-context training.
- [[retrieval-head]] — mechanistic study of the attention heads that copy from long context; LoongRL does not analyze attention heads.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.19363 (arXiv v2, 2025-10-27; v1 2025-10-22), main text and App. A.1–A.9.
- Audit claims not found in the source: "It pairs with QwenLong-L1 as evidence that long-context reasoning can be added by RL without losing short-context ability" — the paper reports that QwenLong-L1-32B suffers MMLU drops (§4.2, Table 2); the pairing is not in the source. "Pad each context with documents from other QA items" is narrowed to the 200K filtered-out tasks (§3.1).
- Not reported by the source: clip ε value, whether batch size counts prompts or rollouts, number and length of distracting chains, seed split, GPU-hours.
- Internal inconsistencies in the source: 7B Stage II is 118 steps in §4.1 and 117 steps in App. A.6 Fig. 7; Fig. 7–8 captions name the stages "Stage II/III" while §3.2.2 names them "Stage I/II".
