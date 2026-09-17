---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: arXiv:2512.12967v1 (QwenLong-L1.5), §2.3, §4.1-§4.4, §5.1-§5.5, Tables 4-5, 7-10, Figure 6 (chapter-local verified extract; no library card exists for this slug on 2026-09-15)
source_url: https://arxiv.org/abs/2512.12967
created_at: "2026-09-15"
---

# Excerpt: QwenLong-L1.5 — Post-Training Recipe for Long-Context Reasoning and Memory Management

- **Authors:** Weizhou Shen, Ziyi Yang, Chenliang Li, et al., Ming Yan (Tongyi Lab, Alibaba Group)
- **Year:** 2025 (arXiv v1 2025-12)
- **Source type:** paper
- **Base model:** Qwen3-30B-A3B-Thinking-2507; framework VeRL; temperature 0.7, top-p 0.95, group size G = 8, purely
  on-policy, batch size 128, constant learning rate 2 × 10⁻⁶; hybrid reward (rule-based containment check, else gpt-oss-120b
  as judge) (§5.1).
- **Used in:** ch-44a §1, §5, §6, Negative samples, Recipe, Generalization lens.

## Length schedule for input and output (§4.1, Figure 6)
"as the input context length increases, the reasoning content length exhibits a generally positive growth trend. Therefore,
to accommodate the output length requirements of tasks with varying input lengths, in QwenLong-L1.5, we synchronously extend
the maximum rollout length for each RL stage. Specifically, we employ three different settings: (1) 20K tokens input with
12K tokens output; (2) 60K tokens input with 20K tokens output; and (3) 120K tokens input with 50K tokens output."
Difficulty-aware retrospective sampling from QwenLong-L1 is applied at each stage transition, with difficulty filtering
computed under the next stage's input-output setting. Figure 6 labels Stage-1 "Max Input: 32 K" while §4.1 gives 20K
(internal inconsistency). Memory-RL runs at max input 128K with 32K chunks and a 15K memory; Stage-4 repeats the 120K/50K
setting after merging.

## Memory agent (§2.3)
The context is split into chunks; at step t the agent updates memory `m_t` from chunk `x_t`, the previous memory and a
navigational plan `p_t`; after the last chunk the answer is produced from `m_K` and the formatting instruction. Memory
training is done as a separate expert and merged, because "mixing memory management training data and single-pass
full-context processing training data together causes considerable damage to the overall RL training infrastructure
efficiency and model training stability"; merging uses the SCE algorithm.

## Negative gradients and entropy (§4.3, §4.4, Tables 4-5)
- "a strong correlation exists between high-entropy tokens and their corresponding gradient norms" in negative rollouts
  (Figure 9, reported correlation 0.96, p < 0.0001); the authors clip either high-entropy negative responses or
  high-entropy tokens inside negative responses.
- Table 4 (ablation of negative-gradient clipping): token-level clipping of low-entropy tokens 55.56 average vs high-entropy
  tokens 57.02; on MRCR "clipping low-entropy tokens results in a 10-point drop compared to clipping high-entropy tokens"
  (36.29 vs 46.20). Sequence-level: clip low-entropy seqs 56.66 (step 30) / 55.47 (step 40); clip high-entropy seqs 57.36.
  "removing too many negative gradient signals can cause entropy collapse".
- **AEPO**: batch entropy `H` is compared to a target band `[H_low, H_high]`. Above `H_high`, "AEPO masks all samples with
  negative advantages" and updates only on positive samples; below `H_low` "the negative gradients are reintroduced ... to
  prevent entropy collapse". On Qwen3-4B-Thinking-2507 (Table 5) the average is 52.79 base, 56.07 GRPO, 59.36 AEPO
  (+3.29 over GRPO).

## Results
- Table 7 (0-128K subsets): Qwen3-30B-A3B-Thinking-2507 61.92 → QwenLong-L1.5-30B-A3B 71.82 (+9.90); MRCR 51.27 → 82.99
  (+31.72); CorpusQA 71.56 → 81.25; LongBench-V2 49.11 → 55.27; DocMath 62.26 → 66.26; Frames 70.27 → 74.76;
  LBV1-QA 67.10 → 70.40. Comparison: Gemini-2.5-Pro 72.40, GPT-5 74.74, DeepSeek-R1-0528 68.67.
- Table 8 (out-of-distribution generality): MMLU-PRO 81.03 → 81.33; AIME24 90.31 → 90.0; AIME25 82.81 → 86.46;
  GPQA-Diamond 75.88 → 76.78; BFCL-V4 Memory-KV 10.97 → 16.77, Memory-Rec_Sum 41.94 → 40.00; LongMemEval 60.80 → 76.40.
- Table 9 (beyond 128K, memory-agent framework): MRCR 128K-512K 34.87, 512K-1M 22.53; CorpusQA 1M 20.72, 4M 14.29;
  Gemini-2.5-Pro full-context 53.83 / 39.51 / 53.11 / not evaluated.
- Table 10 (stage-by-stage full-context average / memory-agent MRCR 512K-1M): baseline 61.92 / 4.24; naive GRPO 67.24 / -;
  Stage-1 69.59 / 17.14; Stage-2 70.46 / 17.05; Stage-3 71.59 / 12.66; Memory-RL 68.53 / 20.34; SCE merge 71.18 / 21.68;
  Stage-4 71.82 / 22.53. The memory-specialized checkpoint loses 3.06 points of full-context average, which merging
  restores.

## Not reported
Entropy band values `H_low`/`H_high`, number of RL steps per stage, truncation rates at the 12K/20K/50K output limits,
compute, and whether Table 8 evaluations use the same decoding settings as the baseline's own reports.
