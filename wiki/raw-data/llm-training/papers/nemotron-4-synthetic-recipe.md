<!-- scope: Recipe ledger for the Nemotron-4 340B Technical Report (arXiv:2406.11704) — pretraining, reward model, staged SFT, DPO, and Reward-aware Preference Optimization settings as printed
     deps: [[nemotron-4-synthetic]]
     see-also: [[nemotron]]
-->

# Nemotron-4 340B Technical Report — Recipe ledger
- **Parent card:** [[nemotron-4-synthetic]] (full source card for the same report).
- **Source:** arXiv:2406.11704v2 (2024-08-06); key numbers also match the NVIDIA PDF `Nemotron_4_340B_8T_0.pdf`.
- **Checked on:** 2026-09-14. All loci are sections, tables, or figures of arXiv v2.
- **Units:** the report does not state whether "batch size" counts sequences or tokens, and does not state whether the 70/15/15 data shares are token shares or sampling weights. These are recorded as printed.

## Ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Nemotron-4-340B-Base | 340B | pretrain-stable | tokens seen | 8T (formal pretraining phase) | arXiv:2406.11704v2 §2.1 | verified 2026-09-14 | no ablation reported |
| Nemotron-4-340B-Base | 340B | pretrain-decay/anneal | tokens seen | 1T continued training after 8T | §2.1, §2.3 "Continued training" | verified 2026-09-14 | "significantly improves model quality" (§2.3); no numbers given |
| Nemotron-4-340B-Base | 340B | pretrain-decay/anneal | data distributions | (1) majority of tokens: already-seen pretraining data with larger sampling weight on higher-quality sources; (2) a small number of QA-style alignment examples plus up-weighted sources from areas of low model accuracy | §2.3 | verified 2026-09-14 | no ablation reported |
| Nemotron-4-340B-Base | 340B | pretrain-decay/anneal | LR schedule | "prioritizes a steeper slope of decay over the magnitude of learning rate"; values not given | §2.3 | not reported (values; checked §2, Tables 1-2) | no ablation reported |
| Nemotron-4-340B-Base | 340B | pretrain-stable | data mixture (unit not stated) | English 70%, multilingual 15% (53 languages), source code 15% (43 programming languages); same blend as Nemotron-4-15B-Base | §2.1 | verified 2026-09-14 | no ablation in this report |
| Nemotron-4-340B-Base | 340B | pretrain-stable | architecture | 96 layers, hidden 18,432, 96 attention heads, 8 KV heads (GQA), RoPE, squared ReLU MLP, no bias, dropout 0, untied embeddings; 9.4B embedding + 331.6B non-embedding parameters | §2.2, Table 1 | verified 2026-09-14 | no ablation reported |
| Nemotron-4-340B-Base | 340B | pretrain-stable | sequence length; vocabulary | 4,096; 256,000 (SentencePiece) | Table 1, §2.2 | verified 2026-09-14 | no ablation reported |
| Nemotron-4-340B-Base | 340B | pretrain-stable | batch size ramp (unit not stated) | 768 for 200B tokens → 1,536 for 200B tokens → 2,304 for 7,600B tokens | Table 2 | verified 2026-09-14 | no ablation reported |
| Nemotron-4-340B-Base | 340B | pretrain-stable | compute and parallelism | 768 DGX H100 nodes (8× H100 80GB); TP 8, PP 12 with interleaving, DP 16 / 32 / 64 (1,536 / 3,072 / 6,144 GPUs); distributed optimizer | §2.3, Table 2 | verified 2026-09-14 | MFU 42.4% / 42.3% / 41.0%; iteration time 10.3 / 10.3 / 8.0 s (Table 2) |
| Nemotron-4-340B-Base | 340B | pretrain-stable | peak LR, warmup, optimizer, weight decay, total GPU-hours | not given | §2 (body) checked | not reported | — |
| Nemotron-4-340B-Reward | 340B | reward-model | data; architecture | HelpSteer2, 10K human preference data; final softmax replaced by a linear projection to 5 HelpSteer attributes; weighted sum at inference | §3.1 | verified 2026-09-14 | RewardBench overall 92.0, Chat-Hard 87.1 (Table 4) |
| Nemotron-4-340B-Reward | 340B | reward-model | RM LR, epochs, loss, attribute weights | not given (report refers to Wang et al., 2024) | §3.1 checked | not reported | — |
| Nemotron-4-340B-Instruct (all alignment) | 340B | SFT + preference | human-annotated data; synthetic share | ~20K human examples (10K SFT, 10K HelpSteer2 for RM and preference fine-tuning); over 98% of SFT and preference data synthetic | §3.2 | verified 2026-09-14 | — |
| Nemotron-4-340B-Instruct (Code SFT) | 340B | SFT | examples; epochs; LR; global batch | ~800K (Genetic Instruct, after de-duplication and filtering); 1 epoch; constant 3e-7; 128 | §3.3.1 | verified 2026-09-14 | Table 6: HumanEval 57.3 (base) → 70.7 |
| Nemotron-4-340B-Instruct (General SFT) | 340B | SFT | examples; replay; epochs; global batch; LR | 200K blend; 2% of Code SFT samples included; 3 epochs; 128; searched in [1e-7, 5e-7] (selected value not given) | §3.3.1 | verified 2026-09-14 | Table 6: MT-Bench 6.79 → 7.99, MMLU 72.2 → 78.3, HumanEval 70.7 → 66.5; replay has no ablation |
| Nemotron-4-340B-Instruct (both SFT stages) | 340B | SFT | loss masking | user turns masked; loss on assistant turns only | §3.3.1 | verified 2026-09-14 | no ablation reported |
| Nemotron-4-340B-Instruct (both SFT stages) | 340B | SFT | sequence length; packing; warmup | not given | §3.3.1 checked | not reported | — |
| Nemotron-4-340B-Instruct (DPO) | 340B | preference | pairs; epochs; global batch; LR | 160K; 1 epoch; 256; constant, tuned in [3e-8, 3e-7] | §3.3.2 | verified 2026-09-14 | Table 6: MT-Bench 7.99 → 7.90, GSM8K 87.9 → 88.5, IFEval prompt-strict 61.4 → 61.7 |
| Nemotron-4-340B-Instruct (DPO) | 340B | preference | DPO KL coefficient; SFT-loss weight on chosen | tuned in [3e-4, 3e-3]; tuned in [1e-5, 1e-3] (selected values not given) | §3.3.2 | verified 2026-09-14 | SFT term added after observing both chosen and rejected likelihoods fall (§3.3.2); no numbers |
| Nemotron-4-340B-Instruct (RPO, each iteration) | 340B | preference | pairs; chosen filtering; SFT-loss coefficient; η; LR; β | 300K; "less harsh" quality filter on chosen; 1e-5; 1; 3e-7; tuned in [1e-3, 1.0] | §3.3.2 | verified 2026-09-14 | Table 6: MT-Bench 7.90 → 8.21 → 8.31 → 8.22; IFEval prompt-strict 61.7 → 78.2 → 79.9 → 79.9 |
| Nemotron-4-340B-Instruct (RPO) | 340B | preference | iterations; init and reference policy | 3; iteration 1 from the DPO checkpoint, each later iteration from the previous one (used as both init and reference) | §3.3.2 | verified 2026-09-14 | "model keeps improving with additional RPO iterations" (§3.3.2); Table 6 |
| Nemotron-4-340B-Instruct (RPO) | 340B | preference | RPO epochs; global batch | not given | §3.3.2 checked | not reported | — |
| Nemotron-4-340B-Instruct | 340B | eval-gate | released checkpoint | checkpoint after the third RPO iteration | §3.3.2, Table 6 last column | verified 2026-09-14 | Table 6 |
| Synthetic data generators | — | SFT + preference data | prompt generator; first-iteration response generator | Mixtral-8x7B-Instruct-v0.1 for synthetic prompts and iteration 1; 340B-Interm-1-Instruct for iteration 2; later intermediate models | §3.2.1, §3.2.4 | verified 2026-09-14 | 340B-Interm-1-Instruct surpasses Mixtral-8x7B-Instruct-v0.1 (§3.2.4); no numbers |
| Synthetic dialogues | — | SFT data | turns; decoding; filter | 3 turns; greedy sampling; Nemotron-4-340B-Reward score below a threshold removed (threshold not given) | §3.2.2 | verified 2026-09-14 | no ablation reported |
| Preference ranking | — | preference data | judge | ground truth / verifier where available; LLM-as-judge (both orders, consistent only) in early iterations; Reward-Model-as-judge later | §3.2.3 | verified 2026-09-14 | RewardBench Chat-Hard: RM-as-judge 0.87 vs LLM-as-judge 0.54 (§3.2.3) |

No `Starting point` paragraph is given: the report prints search ranges rather than selected values for General SFT LR, DPO LR, DPO β, DPO SFT weight, and RPO β, and all values apply to a 340B model.
