---
chapter: ch-58a
course: llm-training
phase: read
excerpt_of: "Olmo 3 (arXiv:2512.13961v2): §2.1, §3.1–3.2, §3.4–§3.6, §4.2–4.3, §5.2, §5.5, §6.1–6.2, Tables 2–5, 17, 26, 27, 30, 35, 47–49, App. A.6"
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-09-17"
note: "The library card model-reports/olmo-3.md has no Verification section and gives no schedule, batch or per-stage settings. ch-58a uses the values below, whose loci were read in arXiv:2512.13961v2 on 2026-09-15 for the ch-14a and ch-34 excerpts of the same passages."
---

# Excerpt: Olmo 3 — the full stage chain, per-branch

Source type: official technical report (Allen Institute for AI), arXiv v2. Released base checkpoints named on the title
page: Olmo-3-1025-7B and Olmo-3-1125-32B. Launch scripts: [[allenai-olmo3-open-instruct-scripts]] and
[[allenai-olmo3-open-instruct-scripts-recipe]]; base-stage configs: [[olmo-core-olmo3-configs]].

## Stage chain
pre-training → mid-training (Dolmino) → long-context extension (Longmino) → branch: Think SFT → Think DPO → Think RL;
Instruct SFT (from the Think SFT checkpoint) → Instruct DPO → Instruct RL; RL-Zero directly from the base model.
Checkpoint merging appears twice: mid-training souping (32B) and an SFT learning-rate merge (Think).

## Base stages (§2.1, §3.6, Table 35)
- "three stages of pretraining for up to 5.9T tokens (Section §3.4), midtraining for 100 billion tokens (Section §3.5),
  and the newly added long-context extension for 50B (Olmo 3 Base 7B) or 100B (Olmo 3 Base 32B) tokens" (§2.1).
- Table 35: 7B pre-training peak LR 3.0e-4, final 3.0e-5, batch 512 instances × 8,192 tokens = 4,194,304 tokens, 5.93T
  tokens, modified cosine (cosine over 5T, second half stretched to 5.93T); 7B mid-training peak LR 2.074e-4 → 0, warmup 0,
  batch 2,097,152 tokens, 100B tokens, linear decay; 7B long-context peak LR 2.074e-4 → 0, warmup 200 steps, batch 64 ×
  65,536 = 4,194,304 tokens, 50B tokens.
- 32B: "double the batch size in all steps, run midtraining twice (with different data order seeds, and average model
  weights of resulting checkpoints), and increase the long-context extension stage from 50B to 100B tokens" (Table 35
  caption); 32B pre-training peak LR 6.0e-4 over a 5.93T cosine truncated at 5.5T.
- Long-context mixture: 34% long-context data and 66% short-context data from Dolmino; context 8,192 → 65,536; YaRN on
  full-attention layers only; document packing with inter-document masking (§3.6). A 10B-token ablation: 66% long / 34%
  short lowered a subset of OlmoBaseEval by 2.5 points, 34% long / 66% short by 0.8 points (§3.6.3).
- 32B mid-training soup: merging two mid-training runs gave "nearly a full point" on the MCSTEM cluster and 2.9 and 1.6
  points on the Math cluster relative to the first and second runs (§3.5.4).

## Post-training stages (§4.2–4.3, §5.2, §5.5, App. A.6, Tables 17, 30, 47)
- Think SFT prompts: 2,268,468 (7B) and 2,253,916 (32B) (Table 17). Instruct SFT prompts: 2,152,112 (Table 30).
- Think SFT training (A.6.1, Table 47): OLMo-core; batch measured in tokens with document packing; 1M-token batches (7B)
  and 4M (32B); two epochs; sequence length 32,768; total tokens 45.4B (7B Think), 45.2B (32B Think), 3.4B (7B Instruct);
  LR 5.0e-5 (7B Think), "1.0 × 10⁻⁴ souped with 5.0 × 10⁻⁵" (32B Think), 8.0e-5 (7B Instruct).
- Checkpoint selection (§4.2): "We train all models for two epochs to avoid overfitting, and perform a learning-rate sweep
  to select the best candidate checkpoints based on our evaluation suite. We then test each candidate checkpoint with a
  series of qualitative 'vibe-test' questions ... our final thinking SFT checkpoint is a linearly weighted merge of two
  checkpoints trained with different learning rates."
- Think DPO (§4.3): further SFT on Qwen3 32B thinking traces "outright hurts the performance of Olmo 3 Think SFT,
  indicating that we are approaching saturation on learning from imitation"; those completions become chosen responses
  paired "with even worse responses", because "minimizing the quality of the rejected completions (thus increasing the
  quality delta) yields a useful contrastive signal for preference tuning". Loss is length-normalized DPO; LR and dataset
  size are swept (A.6.2).
- Instruct SFT starts from the Think SFT checkpoint: "training Olmo 3 Instruct on top of the Olmo 3 Think SFT both
  increases model performance on benchmarks, as shown in Table 29", with response length "minimally affected" (§5.5).
- Instruct DPO contrast (§5.5, Table 32): attempts to improve the OLMo 2 preference pipeline with stronger response
  generators "failed to yield any improvements", attributed to chosen and rejected responses no longer having "meaningful
  contrast".

## Stage deltas, Olmo 3 Instruct 7B (Table 26, average of three runs)
| Benchmark | SFT | DPO | Final (RL) |
|---|---|---|---|
| MATH | 65.1 | 79.6 | 87.3 |
| AIME 2025 | 7.2 | 20.4 | 32.5 |
| BigBenchHard | 51.0 | 69.3 | 71.2 |
| HumanEvalPlus | 69.8 | 72.9 | 77.2 |
| LiveCodeBench v3 | 20.0 | 18.8 | 29.5 |
| IFEval | 81.7 | 82.0 | 85.6 |
| MMLU | 67.1 | 69.1 | 69.1 |
| PopQA | 16.5 | 20.7 | 14.1 |
| AlpacaEval 2 LC | 21.8 | 43.3 | 40.9 |
| BFCL | 48.9 | 49.6 | 49.8 |
| Safety | 89.5 | 89.9 | 87.6 |

## Evaluation regime and RL-Zero controls (§3.4, §6.1–6.2, Tables 2–3)
- 7B: "anneal the learning rate to zero at regular intervals throughout training to assess progress"; 32B: "average the
  weights from four checkpoints, chosen 1,000 steps apart at regular intervals" (§3.4).
- Tables 2 and 3 carry an "OlmoBaseEval HeldOut" block (LBPP, BBH, MMLU Pro MC, Deepmind Math) whose captions state
  "Olmo 3 was not evaluated on held-out benchmarks prior to release."
- Mid-training decontamination (§3.5.4, Fig. 12): "We decontaminate against all splits of all benchmarks"; a matched 100B
  anneal on non-decontaminated data showed that "Performance is sometimes, but not always, inflated by contamination".
- RL-Zero prompts: filtered DAPO Math, Klear-Reasoner Math, Open-Reasoner-Zero and Omega; semantic clustering;
  decontaminated against pre-training and evaluation data; prompts solved 8 of 8 times by the base removed; 13.3K prompts
  (§6.1). Negative control: training on Dolci RL-Zero with random binary rewards "does not improve performance on any of
  our benchmark" suites (§6.2, Fig. 27). Mixed-domain run: "our general run has improved performance across different
  domains, but each domain is under-optimized compared to the single-domain setup" (§6.2).
- Olmo 3.1 RL-Zero improved over 3.0 mainly from a 16k (instead of 12k) completion length and from "not masking truncated
  sequences"; masking made batch sizes vary and reduced stability, and "without training on overlong negative sequences,
  completion lengths were higher, on average" (A.6.4).

## Verification
- Loci read in the cached full text of arXiv:2512.13961v2 on 2026-09-15 (ch-14a and ch-34 excerpts of the same passages);
  reassembled for ch-58a on 2026-09-17 without changing any value or locus.
- Internal inconsistencies recorded: 32B final LR 6.0e-5 (Table 35) versus 6.210e-5 (Fig. 4 caption); mid-training peak LR
  2.074e-4 (7B) versus 2.071e-4 (32B) in the same table.
- Not reported: an ablation of the stretched 7B schedule; per-source ablations for the stage-1 mix; GPU-hour totals per
  stage (GPU counts per stage are in the OLMo-core README, see [[olmo-core-olmo3-configs]]).
