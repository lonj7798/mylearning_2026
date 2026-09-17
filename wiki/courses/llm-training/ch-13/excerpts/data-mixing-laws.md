---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/data-mixing-laws.md (planned card; not present on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2403.16952
primary_version: arXiv:2403.16952v2 (2025-03-20); v1 2024-03
created_at: "2026-09-15"
---

# Excerpt: Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance

Authors: Jiasheng Ye, Peiju Liu, Tianxiang Sun, Jun Zhan, Yunhua Zhou, Xipeng Qiu (Fudan University; Shanghai AI Laboratory). Source type: paper. Read in the v2 PDF text on 2026-09-15 for ch-13 `read.md` §3, §4, §7, and Recipe.

## Main claim (Abstract)
> "experimental results verify that our method effectively optimizes the training mixture of a 1B model trained for 100B tokens in RedPajama, reaching a performance comparable to the one trained for 48% more steps on the default mixture."

## Functional form (§3.2, Eq. 7) and its meaning
L_i(r_1..M) = c_i + k_i · exp(Σ_{j=1..M} t_ij r_j), where L_i is the validation loss on domain i, r_j the proportion of training domain j, and c_i, k_i, t_ij fitted parameters. "c_i represents losses that are not reducible by adjusting the data mixture"; "A negative t_ij indicates that training data of domain j helps reduce validation loss on domain i and vice versa." The overall loss for a validation set with domain shares s_i is Σ_i s_i L_i (Eq. 8); with unknown composition, s_i are learned ("implicit domain aggregation", §3.3).

Two-domain pilot (§3.1): 70M and 160M models on GitHub and Pile-CC at GitHub shares {0.25, 0.375, 0.5, 0.625, 0.75}, 30B tokens; after subtracting a shared constant, log domain loss is linear in the domain proportion (Eq. 6).

## Fit quality (Table 1; 3 domains, 24 fitting and 8 validation mixtures)
Validation MAE for the adopted form (M4 = Eq. 7): GitHub 0.0365, Books3 0.0074, Pile-CC 0.0078. Random guess between observed extremes: 0.8758, 0.1331, 0.1045.

## Domain relationships (§3.2, Figure 4; 5 coarse Pile domains)
Most domains have little relationship; the authors also "observe facilitation (e.g., training dialogue for the internet) and conflict (e.g., training symbolic data for prose) between domains".

## Nested pipeline (§4.1, Algorithm 1) and experiment (§4.2)
- For each mixture and model size, fit a power law in training steps and extrapolate; then fit a power law in model size; then fit the mixing law on the predicted target-scale losses.
- Proxies: 70M, 160M, 305M, 410M for 30B tokens; batch 1M tokens; cosine decay with 2k warmup to 0.1 of the maximum LR at step 100k. Target: 1B on 100B tokens of RedPajama, validated on The Pile's validation set.
- Mixture selection: "We then sample 40 mixtures from all the candidates and train the smallest 70M models. We resample groups of 20 mixtures from them to fit the data mixing law and select the group that reaches minimum prediction errors on all 40 samples as our final set of mixtures to run our pipeline."
- Figure 8 caption: "Our optimized mixture achieves the performance of the default mixture only using 0.73 of the original number of training steps and eventually achieves a performance comparable to a default mixture trained with 1.48 times more tokens (estimated by the scaling law of training steps...)."
- Figure 9 / Figure 20: the optimized mixture has the lowest validation loss among Default, DoGE (Universal), DoGE (OOD), DoReMi (RedPajama), and DoReMi (Pile, adapted by domain overlap), all 1B models on 100B tokens.
- Default RedPajama mixture (Figure 21): CC 67.00%, C4 15.00%, GitHub 4.50%, Books 4.50%, Wikipedia 4.50%, ArXiv 2.50%, StackExchange 2.00%.

## Continued pretraining (§5, Figure 10)
Pythia-70M continued on The Pile plus Python for 10B tokens on 4 mixtures; the fitted law "accurately finds the critical mixture proportion that maintains model performance on the original domain (i.e., the Pile)".

## Limits named by the source
Footnote 7: "some recent studies highlight that the rankings of the data mixture vary as the model size and number of trained tokens change (Goyal et al., 2024; Kang et al., 2024; Covert et al.). Therefore, the optimal mixture at experimented scales can be suboptimal at the target scale." Fitting the two-variable Chinchilla law jointly was unstable, so step and size laws are fit separately (§4.1). The target metric is validation loss, not downstream accuracy.
