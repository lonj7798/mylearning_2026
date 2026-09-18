<!-- scope: gradient-similarity data selection for targeted instruction tuning (LESS, ICML 2024)
     see-also: [[prismatic-synthesis]], [[deita]], [[ifd]], [[cherry-llm]], [[superfiltering]], [[doremi]]
-->

# LESS: Selecting Influential Data for Targeted Instruction Tuning
- **Core Insight:** Ranking instruction data by the Adam-aware influence of Definition 3.1 — the learning-rate-weighted sum, over four LoRA warmup checkpoints, of the cosine similarity between a candidate's Adam update direction and a few-shot target set's gradient — and training on the top 5% beats random 5% selection by 2 to 5 points on MMLU, TydiQA and BBH across Llama-2-7B, Llama-2-13B and Mistral-7B (§5.3, Table 2).
- **Guideline:** When one target capability is known and a few examples of it exist (5 MMLU, 1 TydiQA, or 3 BBH shots in this paper), build the gradient datastore once with a 5% LoRA warmup run and select the top 5% by Adam influence, because the datastore is reusable across target tasks and across target models (§5.3, §6.1). When the target task is far from the pool, expect the 5% subset to lose to the full dataset: on GSM8K-CoT the 5% subset does not outperform full-data training (App. D.6).
- **Authors:** Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (Princeton Language and Intelligence; Gururangan at University of Washington; Xia and Malladi equal contribution)
- **Year:** 2024 (arXiv v1 2024-02; ICML 2024, Proceedings of the 41st ICML)
- **URL:** https://arxiv.org/abs/2402.04333 (code and data: https://github.com/princeton-nlp/LESS)
- **Source type:** paper
- **Relevant topics:** data selection, influence estimation, Adam-aware influence, LoRA, random projection, targeted SFT

## Abstract
The paper addresses targeted instruction tuning: given a small number of examples that embody one capability, select relevant fine-tuning data from a large mixed instruction pool. Existing influence formulations assume SGD and fixed-length inputs. LESS adapts them to the Adam optimizer and to variable-length instruction data, then makes the computation affordable by taking gradients from a LoRA warmup model and projecting them to a low dimension. The resulting gradient datastore is built once and reused per target task. Training on a LESS-selected 5% of the data often outperforms training on the full dataset, and the selected data transfers: a smaller selection model produces data that helps larger models and models from other families. A qualitative analysis reports that LESS selects examples requiring the same reasoning process as the target, not examples with matching surface form.

## Key Contributions
- **Adam influence (Definition 3.1):** `Inf_Adam(z, z') = Σ_i η̄_i · cos(∇ℓ(z'; θ_i), Γ(z, θ_i))`, where `η̄_i` is the average learning rate in epoch `i`, `θ_i` the checkpoint after epoch `i`, and `Γ` the Adam update direction `m/(√v + ε)`. Cosine similarity replaces the dot product because sequence-level gradients otherwise upweight short sequences (§3.1, §3.2).
- **Reusable low-rank gradient datastore:** LoRA warmup on a random 5% subset for N = 4 epochs, then per-example LoRA gradients projected to d = 8192 with a random projection (§4.1). Steps 1 and 2 are offline and computed once per candidate set (Fig. 1 caption).
- **Transfer (LESS-T):** selecting with Llama-2-7B as the selection model and training Llama-2-13B or Mistral-7B beats random selection and stays close to selecting with the target model itself (§5.3, Table 2).
- **Baseline comparison:** BM25, DSIR and RDS give little improvement over random at 5%; LESS beats the strongest baseline by 2.6 (MMLU), 3.5 (TydiQA) and 1.7 (BBH) points on Llama-2-7B (§5.3, Table 3).

## Key Figures/Tables to Study
- Fig. 1: the four steps (warmup LoRA training, gradient features, selection, training).
- Table 2: LESS, LESS-T, random 5% and full data on three models and three evaluations.
- Table 3: LESS against BM25, DSIR and RDS at 5% on Llama-2-7B.
- Tables 4, 5, 6 and Fig. 2: cost, warmup ablation, checkpoint-count ablation, projection-dimension ablation.

## Technical Details
- Candidate pool: FLAN V2 (100,000), CoT (100,000), Dolly (15,011) and Open Assistant 1 (55,668) instances, about 270K data points total (§5.1; App. Table 7).
- Validation sets used as the target few-shot sets: MMLU 5-shot, 285 examples over 57 tasks; TydiQA 1-shot, 9 examples over 9 tasks; BBH 3-shot, 69 examples over 23 tasks (Table 1).
- Test sets: MMLU 18,721 items, TydiQA 1,713, BBH 920 (Table 1). MMLU is 5-shot accuracy over 57 subtasks; TydiQA is 1-shot macro-averaged F1 over 11 languages in the gold-passage setup; BBH is 3-shot exact match with chain-of-thought demonstrations (App. B).
- Multi-subtask aggregation: a candidate is scored by `max_j Inf_Adam(z, D_val^(j))` over validation subtasks, so a point that helps any one subtask can be selected (§4.2).
- Llama-2-7B results at 5%: LESS 50.2 (0.5) MMLU, 56.2 (0.7) TydiQA, 41.5 (0.6) BBH, against random 46.5 (0.5), 52.7 (0.4), 38.9 (0.5) and full-data 51.6, 54.0, 43.2 (Table 2). The 5% subset exceeds the full dataset on TydiQA here but not on MMLU or BBH; the paper states that the 5%-over-full effect is more evident with Llama-2-13B and Mistral-7B (§5.3).
- Cost on a single A100 (80GB): warmup LoRA training 6 hours, gradient feature computation 48 hours, data selection under 1 minute; the datastore occupies 17.7 GB (Table 4).
- Warmup ablation: gradients from off-the-shelf Llama-2-7B and Llama-2-7B-Chat give 46.2 average, the same as random (46.0), while the default 5% warmup gives 49.3 and a 100% warmup gives 50.5 (Table 5).
- Checkpoint ablation: N = 1 gives 47.8 average, N = 4 gives 49.3, random gives 46.0 (Table 6).
- Projection dimension: 1024, 2048, 4096 and 8192 all beat random 5%, with performance increasing in d (Fig. 2, App. D.4).
- Three trials with distinct random seeds per experiment; for LESS, each trial uses a different warmup subset and a different selected subset (App. A.2).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-2-7B / 13B, Mistral-7B | 7B–13B | SFT (warmup + final) | optimizer, schedule | linear warmup then cosine decay, peak LR 2e-5 | arXiv:2402.04333v3 App. A.2 | verified 2026-09-18 | no ablation reported |
| Llama-2-7B / 13B, Mistral-7B | 7B–13B | SFT (warmup + final) | global batch (sequences) | 128 | App. A.2 | verified 2026-09-18 | no ablation reported |
| Llama-2-7B / 13B, Mistral-7B | 7B–13B | SFT (warmup + final) | epochs | 4, on every selected dataset | App. A.2 | verified 2026-09-18 | more epochs on small datasets gave no improvement (App. A.2, prose; no table) |
| Llama-2-7B / 13B, Mistral-7B | 7B–13B | SFT (warmup + final) | adapter | LoRA rank 128, α 512, dropout 0.1, all attention matrices | App. A.2 | verified 2026-09-18 | no ablation reported |
| Llama-2-7B | 7B | SFT | trainable parameters | 135M (1.95% of the model) | App. A.2 | verified 2026-09-18 | not applicable |
| Mistral-7B | 7B | SFT | trainable parameters | 109M (1.48%) | App. A.2 | verified 2026-09-18 | not applicable |
| Llama-2-13B | 13B | SFT | trainable parameters | 209M (1.59%) | App. A.2 | verified 2026-09-18 | not applicable |
| Llama-2-7B (selection model) | 7B | data selection | warmup subset, epochs | random 5% of D, N = 4 epochs | §4.1, §5.1 "Default setting" | verified 2026-09-18 | Table 5 (5% warmup 49.3 vs no warmup 46.2 avg); Table 6 (N=4 49.3 vs N=1 47.8) |
| Llama-2-7B (selection model) | 7B | data selection | projection dimension d | 8192 | §4.1; §5.1 | verified 2026-09-18 | Fig. 2: average performance increases with d over 1024–8192 |
| Llama-2-7B (selection model) | 7B | data selection | selected fraction of D | top 5% by Inf_Adam | §5.1 "Default setting" | verified 2026-09-18 | no ablation reported; the paper notes the optimal threshold is left to future work (§5.3 footnote 8) |

## Findings relevant to generality and distillation
- **Generality.** The paper measures only the three target evaluations plus GSM8K and TruthfulQA; it does not report held-out benchmarks outside the target set. On TruthfulQA the 5% subset outperforms full-data training; on GSM8K-CoT it does not, and the authors state the pool has little relevant data for GSM8K (App. D.6).
- **Transfer across scale.** With Pythia models at 14M, 410M, 1B, 6.9B and 12B, LESS beats random selection at all scales, and a Pythia-14M selection model still selects useful data for larger Pythia models even though it cannot solve the task (App. D.5).
- **Surface form.** In the qualitative comparison for a Bengali TydiQA example, BM25 and RDS select Bengali examples from unrelated tasks while LESS selects an English open-book question-answering example (§6.2, Table 16).

## Connections
- [[prismatic-synthesis]]: reuses the random-projection construction from LESS; LESS ranks by similarity to a target set, Prismatic by gradient coverage.
- [[cherry-llm]], [[ifd]], [[superfiltering]], [[deita]], [[alpagasus]]: other SFT data-selection signals in this library. LESS does not compare against them.
- [[doremi]]: domain-level reweighting for pretraining, as opposed to per-example selection for instruction tuning.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2402.04333 (v3, 13 Jun 2024).
- Corrections to the previous card version:
  - "LoRA-train the target base model on the pool for ~4% of the full training budget" → LoRA warmup on a randomly selected 5% of D for N = 4 epochs (§4.1, §5.1).
  - "Train: full fine-tune (not LoRA) on the selected 5%" → the warmup training and the final model training are both conducted with LoRA (§5.1).
  - "a pool of instruction data (typically 400K+ mixed sources)" → about 270K data points from FLAN V2, CoT, Dolly and Open Assistant 1 (§5.1, App. Table 7).
  - "5% LESS-selected > 100% random across MMLU / BBH / TydiQA" → LESS at 5% beats random at 5% by 2 to 5 points; against the full dataset it wins on some model/task pairs and loses on others (Table 2).
  - "shows the gradients only stabilize after ~1% of training; skipping warmup degrades selection" → the warmup ablation varies |D_warmup| over 5%, 25% and 100%; no 1% setting is reported (Table 5).
  - "Raw influence ≈ `<g_i, g_val> / sqrt(<g_val, g_val>)`" → the paper's SGD form is `Inf_SGD(z, z') = Σ_i η̄_i ⟨∇ℓ(z'; θ_i), ∇ℓ(z; θ_i)⟩`; the quoted normalization does not appear (§2, §3.1).
  - "a 7B-model datastore selects useful data for 13B models and cross-family" stated without numbers → LESS-T numbers are in Table 2 and are lower than LESS on MMLU/BBH for Llama-2-13B.
  - "deps: [[cherry-llm]]" → LESS does not build on Cherry-LLM; moved to see-also.
- Removed as unsupported by the source: "ICML 2024 Spotlight" (the paper text states Proceedings of the 41st ICML only); "Adam-adjustment matters: vanilla influence gives worse selections at LLM scale" as a headline claim (the Inf_SGD ablation is in App. D.2 and no number is quoted here); "Random-projection variance: averaging multiple seeds stabilizes results" (three seeds are used per experiment, but no projection-variance claim is made); "Quality gating not guaranteed — combine with answer verifiers for synthetic data"; "Especially strong when the pool is large and heterogeneous"; "optional deduplication; no additional quality filter required"; "L2-normalized" as a storage step (normalization is inside the cosine similarity, §3.2); "Transferability matrix — source-model × target-model heatmap" and "Figure 6 active learning" (no such figures); "Used in the selection stage of modern post-training recipes" (no source given).
- Not reported by the source: the number of distinct target tasks a single datastore was reused for; per-source retention of the selected 5%; wall-clock cost of the final training runs.
