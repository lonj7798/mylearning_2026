---
chapter: ch-58a
course: llm-training
phase: read
excerpt_of: "Apertus v1 Technical Report: Democratizing Open and Compliant LLMs for Global Language Environments (arXiv:2509.14233v2): Abstract, §1, §2.1–2.2, §3.3–3.4, §5.1–5.2, Tables 1–2, 6–7, 14–15, 21"
source_url: https://arxiv.org/abs/2509.14233
created_at: "2026-09-17"
note: "No library card exists for this slug as of 2026-09-17. The loci below were read in arXiv:2509.14233v2 on 2026-09-15 for the ch-13a excerpt of the same passages; ch-58a uses the stage structure, the mixture-selection method, and the held-out suite."
---

# Excerpt: Apertus — a five-stage pre-training chain with a declared held-out suite

Project Apertus (Swiss AI Initiative: EPFL, ETH Zurich, CSCS); leads Antoine Bosselut, Martin Jaggi, Imanol Schlag.
Source type: official technical report. arXiv v2 dated 2025-12-01; v1 September 2025.

## Models and budget (§2.1, Tables 1–2)
- Apertus 8B: 32 layers, d 4,096, 32 query and 8 key-value heads. Apertus 70B: 80 layers, d 8,192, 64 and 8. Untied input
  and output embeddings. Pre-training context 4,096, extended to 65,536.
- Both sizes see 15T tokens. 8B: max LR 1.1e-4, batch 4.2M → 8.4M tokens. 70B: max LR 1.0e-5, batch 8.4M → 16.8M tokens.
- Tokenizer: byte-level BPE adapted from the Mistral-Nemo "v3 tekken" tokenizer, vocabulary 2^17 = 131,072, selected by
  comparing four tokenizers on FLORES+ in 55 languages with fertility, compression ratio, vocabulary utilization and Gini
  coefficient (§2.2, Fig. 1).

## Five pre-training stages (§3.3, Table 6)
Stage boundaries for the 70B, in tokens: stage 1 (0–5T), stage 2 (5–9T), stage 3 (9–12T), stage 4 (12–13.5T), stage 5
(13.5–15T). Table 6 lists the pool sizes available per stage, and its caption states "not necessarily all tokens of each
stage data were consumed". The English web pool changes source across stages (FineWeb-Edu score-2 → FineWeb-HQ 33% plus
FineWeb-Edu score-3 → DCLM-Edu); math pools are added in stage 3; parallel data and Clean Wikipedia enter in stage 5. The
8B model skipped stage 2 and switched from stage 1 to stage 3 at 7,038B consumed tokens (App. H.3, Table H.8).

## How the mixture was selected (§3.3, Table 7)
Cooldown ablations on 1.5B checkpoints over 100B tokens, using 70% stage-1 data plus 30% of the candidate dataset, scored
by English and multilingual macro accuracy: the regular mix gives English 0.45175 and multilingual 0.44301; the 30%
DCLM-Edu candidate gives 0.46158 and 0.44608. The candidates were English datasets, and no multilingual-share ablation is
reported. The coverage claim itself — "over 1800 languages, with ∼40% of pretraining data allocated to non-English
content" (Abstract) — has no reported ablation of the 40% share.

## Held-out evaluation (§5.1–5.2, Tables 14–15, 21)
- Table 21 benchmarks "were held-out during model development and were not used for making decisions" (§5.2). On ARC
  Challenge Multilingual: Apertus-8B-Instruct 36.8, Llama-3.1-8B-Instruct 32.0, Qwen3-8B 30.2.
- Table 15 separates "Factual Agnostic" (MMLU, Global-MMLU) from "Factual Regional" (INCLUDE V1 and V2, CulturalBench,
  BLEND, SwitzerlandQA) so that region-specific knowledge is not averaged into a single score. Pretrained Global-MMLU:
  Apertus-8B 55.3, OLMo2-7B 41.1, Qwen2.5-7B 60.3; INCLUDE V1 over 44 languages: 54.8, 33.8, 53.9.

## Verification
- Loci read in arXiv:2509.14233v2 on 2026-09-15 (ch-13a excerpt of the same passages); reassembled for ch-58a on
  2026-09-17 without changing a value or locus.
- Not reported: per-stage GPU hours; an ablation of the 40% non-English share; post-training stage settings at the level
  of detail of the pre-training tables.
