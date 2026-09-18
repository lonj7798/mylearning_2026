<!-- scope: recipe ledger for the pre-training runs in "Understanding Emergent Abilities of Language Models from the Loss Perspective" (Du et al., 2024)
     deps: [[emergence-loss-perspective]]
     see-also: [[chinchilla-compute-optimal]], [[lr-schedules]]
-->

# Recipe ledger — Understanding Emergent Abilities of Language Models from the Loss Perspective
- **Core Insight:** The paper discloses architecture shape, token count, batch size, and peak learning rate for 3 large and 28 small from-scratch runs (App. A.2 Tables 4-5), plus corpus mix and GPU-days (App. A.1, App. H).
- **Guideline:** When reusing these values, keep them scoped to the paper's own English:Chinese 4:1 corpus and 65k SentencePiece BPE tokenizer, because the paper states loss values are not comparable across corpora or tokenizers (§7).
- **Authors:** Zhengxiao Du, Aohan Zeng, Yuxiao Dong, Jie Tang (Zhipu AI; Tsinghua University)
- **Year:** 2024 (arXiv v1 2024-03; NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2403.15796
- **Source type:** paper
- **Relevant topics:** pre-training hyperparameters, model shapes, data mixture

## Recipe ledger
Locus prefix for every row: arXiv:2403.15796v3. The released models are research checkpoints without public names; rows are named by size and table. "Batch Size" is printed without a unit.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Du et al. Table 4 run | 1.5B | pretrain-stable | tokens; d_model; d_hidden; n_heads; n_layers; batch size; max LR | 3T; 2048; 6912; 16; 24; 1344; 5e-4 | App. A.2 Table 4 | verified 2026-09-14 | no ablation reported |
| Du et al. Table 4 run | 6B | pretrain-stable | same fields | 3T; 4096; 13696; 32; 28; 4224; 4e-4 | App. A.2 Table 4 | verified 2026-09-14 | no ablation reported |
| Du et al. Table 4 run | 32B | pretrain-stable | same fields | 2.5T; 6656; 22272; 52; 58; 8832; 3e-4 | App. A.2 Table 4 | verified 2026-09-14 | no ablation reported |
| Du et al. Table 5 runs | 300M | pretrain-stable | tokens (one run each); d_model; d_hidden; n_heads; n_layers; batch size; max LR | 67B, 125B, 250B, 500B; 1152; 3840; 9; 12; 1152; 2.8e-3 | App. A.2 Table 5 | verified 2026-09-14 | no ablation reported |
| Du et al. Table 5 runs | 540M | pretrain-stable | same fields | 33B, 66B, 125B, 250B, 500B; 1536; 5120; 12; 12; 1152; 2e-3 | App. A.2 Table 5 | verified 2026-09-14 | no ablation reported |
| Du et al. Table 5 runs | 1B | pretrain-stable | same fields | 33B, 67B, 125B, 250B, 500B; 2048; 6912; 16; 16; 1152; 1.5e-3 | App. A.2 Table 5 | verified 2026-09-14 | no ablation reported |
| Du et al. Table 5 runs | 1.5B | pretrain-stable | same fields | 67B, 100B, 125B, 250B, 375B, 500B; 2048; 6912; 16; 24; 1152; 1e-3 | App. A.2 Table 5 | verified 2026-09-14 | no ablation reported |
| Du et al. Table 5 runs | 3B | pretrain-stable | same fields | 67B, 125B, 250B, 500B; 3072; 10240; 24; 24; 1152; 7e-4 | App. A.2 Table 5 | verified 2026-09-14 | no ablation reported |
| Du et al. Table 5 runs | 6B | pretrain-stable | same fields | 33B, 67B, 125B, 250B; 4096; 13696; 32; 28; 1152; 4e-4 | App. A.2 Table 5 | verified 2026-09-14 | no ablation reported |
| All runs | all | pretrain-stable | sequence length; optimizer | 2048; AdamW β1 = 0.9, β2 = 0.95 | App. A.2 | verified 2026-09-14 | no ablation reported |
| Table 5 runs | 300M-6B | pretrain-decay/anneal | LR schedule | cosine schedule set to reach its minimum at each run's token count | §2.4 | verified 2026-09-14 | §2.4 cites scaling-law practice [28, 23]; no ablation |
| All runs | all | pretrain-stable | warmup; final LR; weight decay; grad clip | not reported | checked body, App. A-I | not reported | — |
| All runs | all | pretrain-stable | corpus language ratio (tokens) | English:Chinese = 4:1 | §2.1, App. A.1 | verified 2026-09-14 | no ablation reported |
| All runs | all | pretrain-stable | English source mix ("ratio"; token vs document share not stated) | CommonCrawl 80.2%, Code 10.0%, Books 3.8%, Wikipedia 3.8%, Papers 1.6%, StackExchange 0.6% | App. A.1 Table 3 | verified 2026-09-14 | no ablation reported |
| All runs | all | pretrain-stable | repetition | Wikipedia and Books trained for multiple epochs; 93.4% of documents never repeated | App. A.1 | verified 2026-09-14 | no ablation reported |
| All runs | all | pretrain-stable | tokenizer | BPE (SentencePiece), vocabulary 65k | App. A.1 | verified 2026-09-14 | no ablation reported |
| All runs | all | pretrain-stable | architecture changes vs LLaMA | grouped-query attention "to replace the multi-query attention"; RoPE on half of the query/key dimensions | §2.1 | verified 2026-09-14 | no ablation reported |
| Table 4 runs | 1.5B / 6B / 32B | pretrain-stable | compute (DGX-A100 8x80G) | 8 days on 256 A100; 8 days on 1024 A100; 20 days on 2048 A100 | App. H | verified 2026-09-14 | — |
| Table 5 runs | 300M-6B | pretrain-stable | compute | "about 20 days on 256 A100 GPUs" | App. H | verified 2026-09-14 | — |

## Evaluation settings
- Splits and sizes: TriviaQA validation 11,313; HellaSwag validation 10,042; RACE test 4,934; WinoGrande validation 1,267; MMLU test 14,042; GSM8K test 1,319; NLPCC-KBQA validation 10,613; ClozeT validation 938; CLUEWSC train+validation 508; C3 validation 3,816; C-Eval validation 1,346; GSM8K-Chinese test 1,212 (App. B Table 6).
- Prompting: TriviaQA, RACE, MMLU, NLPCC-KBQA, C3, C-Eval few-shot; HellaSwag, WinoGrande, ClozeT, CLUEWSC zero-shot; GSM8K and GSM8K-Chinese few-shot CoT (Table 1).
- GSM8K-Chinese is machine-translated from GSM8K with human proofreading (App. B).

## Connections
- [[emergence-loss-perspective]] — main card with the findings these runs produced.
- [[chinchilla-compute-optimal]] — cited for setting the cosine minimum at the target token count (§2.4).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2403.15796 (arXiv v3, App. A.1, A.2, B, H).
- Audit claims not found in the source: none.
