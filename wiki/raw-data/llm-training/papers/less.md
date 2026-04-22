<!-- scope: gradient-similarity data selection for targeted instruction tuning (Xia 2024, ICML 2024)
     deps: [[cherry-llm]]
     see-also: [[prismatic-synthesis]], [[deita]], [[ifd]]
-->

# LESS: Selecting Influential Data for Targeted Instruction Tuning
- **Core Insight:** Per-sample training gradients — once projected to low dimension and L2-normalized — are reusable *influence features*; the 5% of samples whose gradients are most similar to a held-out few-shot "target capability" set can outperform training on 100% of the data.
- **Guideline:** Build a reusable low-rank gradient datastore once; given a few-shot target exemplar set (e.g. 5 MMLU examples), rank pool samples by cosine similarity of their projected Adam-adjusted gradients and keep the top 5%.
- **Authors:** Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (Princeton)
- **Year:** 2024 (ICML 2024)
- **URL:** https://arxiv.org/abs/2402.04333
- **Relevant topics:** data selection, gradient similarity, influence functions, LoRA, targeted SFT

## Abstract
LESS (**L**ow-rank gradi**E**nt **S**imilarity **S**earch) builds on influence-function theory but makes it practical at LLM scale. Two key moves: (1) approximate gradients with LoRA warmup + random projection to ~8K dims — reusable across queries; (2) adjust the influence formula to account for Adam's second-moment scaling (vanilla influence assumes SGD). Given a small few-shot target set representing a capability (e.g., 5 MMLU questions), LESS selects pool samples whose gradient projections are most similar. **Training on a LESS-selected 5% subset often matches or exceeds training on the full dataset** across MMLU, TydiQA, BBH, and is transferable — a small-model gradient datastore selects data that improves larger models.

## Key Contributions
- Adam-aware influence formulation — fixes a bias present in naive SGD-influence at LLM scale.
- **Low-rank gradient datastore** via LoRA warmup + random projection — one-time cost, reusable for any target.
- Demonstrated **model-size and model-family transferability**: a 7B-model datastore selects useful data for 13B models and cross-family.
- ICML 2024 Spotlight.

## Key Figures/Tables to Study
- **Table comparing 5%-LESS vs full-data training** on MMLU / TydiQA / BBH — LESS wins or ties.
- **Transferability matrix** — source-model × target-model heatmap.
- **LoRA warmup ablation** — shows the gradients only stabilize after ~1% of training; skipping warmup degrades selection.

## Scoring function (exact form)
Let `g_i` be the projected gradient of training sample `x_i` under Adam-adjusted influence; let `g_{val}` be the averaged projected gradient over the few-shot target set.
- Raw influence ≈ `<g_i, g_{val}> / sqrt(<g_{val}, g_{val}>)`.
- LESS variant uses **cosine similarity** of L2-normalized Adam-adjusted gradients — numerically more stable.
- Project to `d ≈ 8K` via fixed Gaussian random projection; preserves inner products (JL lemma).

## Synthesis/selection pipeline (REQUIRED — be concrete)
- **Seed input:** a pool of instruction data (typically 400K+ mixed sources) + a small target few-shot set representing the wanted capability.

- **Selection step(s):**
  1. **Warmup:** LoRA-train the target base model on the pool for ~4% of the full training budget (reads all data cheaply, stabilizes gradients).
  2. **Gradient datastore build:** for each pool sample, compute the Adam-adjusted per-sample gradient, project to 8K dims, L2-normalize, store.
  3. **Query vector:** for the target few-shot set, compute the same projected gradient, average, normalize.
  4. **Select:** cosine-rank pool samples against query, keep top 5%.
  5. **Train:** full fine-tune (not LoRA) on the selected 5%.

- **Filtering/rescoring:** optional deduplication; no additional quality filter required.

- **Output shape:** per-query 5%-subsets; datastore is reusable across many queries.

- **Teacher model(s):** no external teacher; target base model is its own scorer (after warmup).

- **Cost estimate:** warmup + datastore build ≈ single short run on the pool; amortized across many target queries.

## Quality / diversity evaluation
- 5% LESS-selected > 100% random across MMLU / BBH / TydiQA.
- Transfers across model families (Llama → Mistral) and sizes (7B → 13B).
- Especially strong when the pool is large and heterogeneous.

## Risks + gotchas
- **Target-set dependence:** selection is only as good as the few-shot target exemplars.
- **Adam-adjustment matters:** vanilla influence (SGD assumption) gives worse selections at LLM scale.
- **Random-projection variance:** averaging multiple seeds stabilizes results.
- **Quality gating not guaranteed:** LESS optimizes for similarity, not correctness — combine with answer verifiers for synthetic data.
- **Compute non-trivial** — warmup + gradient datastore is not free; worth it when many selection targets will be served.

## Connections
- Mechanism-compatible predecessor of [[prismatic-synthesis]] (2025): LESS selects for gradient *alignment* with a target set; Prismatic selects for gradient *coverage*.
- Alternative to [[cherry-llm]] (IFD-based, no gradients) and [[deita]] (external-scorer-based).
- Reusable datastore concept directly informs later "influence-function at scale" research.
- Used in the selection stage of modern post-training recipes that emphasize targeted capability injection.
