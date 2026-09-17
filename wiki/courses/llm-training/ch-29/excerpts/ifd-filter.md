---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/cherry-llm.md (verified 2026-09-14) + https://arxiv.org/abs/2402.00530 (Superfiltering, primary text)
source_url: https://arxiv.org/abs/2308.12032
created_at: "2026-04-23"
revised_at: "2026-09-15 (generality revision)"
---

# Excerpt: two IFD definitions, the pre-experience step, and weak-scorer overlap

**Artifacts:** Li et al., "From Quantity to Quality" (Cherry LLM, arXiv:2308.12032v5, NAACL 2024), card [[cherry-llm]]; Li et al., "Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning" (arXiv:2402.00530, ACL 2024), card [[superfiltering]] (card not verified; values below read from the PDF).

## Definition 1: ratio of mean losses (Cherry LLM, §2.2, Eq. 3–5)

```
s(A|Q) = −(1/N) Σ_{i=1..N} log P(w_i | Q, w_1 … w_{i−1}; θ)      (Eq. 3)
s(A)   = −(1/N) Σ_{i=1..N} log P(w_i | w_1 … w_{i−1}; θ)         (Eq. 4)
IFD(Q, A) = s(A|Q) / s(A)                                         (Eq. 5)
```

- Q = instruction plus optional input in the dataset template; A = answer with N tokens; w_i = i-th answer token; θ = weights of the pre-experienced model.
- A higher IFD means the instruction helps less. Samples with IFD > 1 are treated as misaligned and removed; the method keeps the highest remaining values (§1, §2.2).
- Worked example printed in App. F, Figure 9: s(A) = 0.761, s(A|Q) = 0.696, IFD = 0.915 (printed 0.914).

Released script (Cherry_LLM@01fac8d `cherry_seletion/data_by_IFD.py` L103–L120):

```python
mean_1 = loss_list_1.mean()
mean_2 = loss_list_2.mean()
mean_rate = mean_2/mean_1
if mean_rate > 1:
    continue
...
mean_rate_list = sorted(mean_rate_list)
if args.sample_number == 0:
    args.sample_number = int(len(mean_rate_list)*args.sample_rate)
mean_rate_list_id = [i for i in range(len(mean_rate_list))][-args.sample_number:]
```

`loss_list_2` is computed on the full text (instruction + answer) and `loss_list_1` on the answer alone (L95), so `mean_rate` is s(A|Q)/s(A). The kept count is a fraction of the samples that survive the IFD ≤ 1 and length checks, not of the input pool. `--sample_rate` defaults to 0.1 (L27).

## Definition 2: ratio of perplexities (Superfiltering, §2.1, Eq. 1–2)

```
PPL(y|x) = exp( −(1/N) Σ_j log p(y_j | x, y_1 … y_{j−1}) )       (Eq. 1)
IFD(y|x) = PPL(y|x) / PPL(y)                                      (Eq. 2)
```

Since PPL = exp(mean loss), Eq. 2 equals exp(s(A|Q) − s(A)). It depends on the difference of the two mean losses; Cherry's Eq. 5 depends on their ratio. The two definitions agree on the IFD = 1 boundary but can order samples differently (worked example in ch-29 §3.3).

## Pre-experience model (Cherry LLM §2.1, §4.3)

- 1,000 samples: K-means with 100 clusters on mean last-layer hidden states of the instruction tokens, 10 samples per cluster; trained for 1 epoch (§2.1).
- 0 samples gives the lowest results among the tested counts but still beats Alpaca at 10% data; 300 samples give "a distinct performance gain"; 500 give no further gain (§4.3.1, Figure 5).
- Choosing the 1,000 by difficulty, diversity, or at random gives winning scores within 1.007–1.097 at 5/10/15% (Table 2).
- The LLaMA2 runs use the base model as scorer with no pre-experience step (§4.4).

## Results that matter for generality (card [[cherry-llm]])

- LLaMA-7B, Table 1: 5% Alpaca beats official Alpaca on leaderboard average (52.06 vs 50.21) and AlpacaEval (34.74 vs 26.46); 10% WizardLM is below reimplemented WizardLM on both (51.59 vs 52.79; 61.44 vs 61.99).
- MMLU falls after selection on LLaMA-7B: Alpaca 41.73 → 36.51 at 5% (Table 1).
- Top 5% by IFD is dominated by "write story", "generate story", "generate list"; bottom 5% by "rewrite sentence", "edit sentence" (Table 4).
- Math and Coding sub-categories are worse for the 5% Alpaca model than for official Alpaca (App. C).
- Recommended share: top 10%, "a safe and reasonable choice" (App. G.2).

## Weak scorer versus 7B scorer (Superfiltering Table 1)

Spearman ρ of IFD rankings against LLaMA2-7B, and overlap of the selected subsets:

| Dataset | Scorer | ρ (IFD) | overlap 5% / 10% / 15% |
|---|---|---|---|
| Alpaca | GPT-2 | 0.679 | 0.28 / 0.41 / 0.49 |
| Alpaca | GPT-NEO | 0.802 | 0.38 / 0.51 / 0.59 |
| Wizard 70k | GPT-2 | 0.802 | 0.42 / 0.54 / 0.61 |

At a 10% keep share on Alpaca, 41% of the GPT-2-selected samples are also in the LLaMA2-7B selection.

## Connections

- [[cherry-llm]], [[cherry-llm-recipe]] — verified card and recipe ledger.
- [[ifd]] — standalone metric card (not verified at the time of this revision; it gives the perplexity form and a random 1K warm-up, which differ from Cherry LLM).
- [[deita]] — reports IFD as a complexity baseline: MT-Bench 5.91 on X_sota and 2.46 on X_base vs random 5.84 / 4.93 (Table 2).
