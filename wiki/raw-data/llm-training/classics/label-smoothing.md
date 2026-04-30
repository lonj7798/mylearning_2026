<!-- scope: label smoothing — entropy floor; cross-entropy regularization
     deps: []
     see-also: [[dropout]], [[kl-control-rlhf]]
-->

# Label Smoothing (Rethinking the Inception Architecture)
- **Core Insight:** Replacing one-hot targets with a softened distribution (`1-eps` on the true class, `eps/(K-1)` elsewhere) prevents the model from driving its logits to infinity, which improves calibration and generalization.
- **Guideline:** Use label smoothing `eps = 0.1` for translation/SFT cross-entropy; **omit** it during pretraining if you also want a usable next-token-probability for downstream RL/scoring.
- **Authors:** Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
- **Year:** 2015 (Inception-v3 paper); formalized further by Müller et al. 2019 ("When does label smoothing help?")
- **URL:** https://arxiv.org/abs/1512.00567 ; https://arxiv.org/abs/1906.02629
- **Relevant topics:** loss function, calibration, KL regularization, SFT recipes, RLHF

## Abstract
Introduced as one of several regularizers in the Inception-v3 architecture, label smoothing modifies the cross-entropy training target by reserving a small probability mass `eps` for non-true classes. Formally, the target distribution becomes `q'(k) = (1 - eps) * delta(k = y) + eps / K` (uniform smoothing) or `(1 - eps) * delta(k = y) + eps * u(k)` for an arbitrary prior `u`. This prevents the network from becoming arbitrarily confident, since the optimal logit difference is now bounded. Empirically: improves ImageNet top-1 by ~0.2%, improves NMT BLEU by ~0.5–1.0, and dramatically improves model calibration (ECE, expected calibration error).

## Key Contributions
- **Label smoothing formula**: replaces hard targets with soft ones; trivial to implement, broadly beneficial.
- Demonstrated improvements across image classification (Inception-v3) and machine translation (Transformer used `eps = 0.1`).
- Müller 2019 follow-up: shows label smoothing produces tighter, more equidistant class clusters; helps top-1 accuracy and calibration but **hurts knowledge distillation** (teacher's softened logits become less informative).
- Theoretical equivalence: label smoothing = adding a KL-to-uniform regularizer with weight `eps`.
- Established that "well-calibrated overconfidence" is a real phenomenon in deep nets; label smoothing the simplest fix.

## Key Figures/Tables to Study
- **Müller 2019 Figure 1**: t-SNE of penultimate-layer activations for ResNet-56 on CIFAR-10, with vs without label smoothing — visibly tighter and more equidistant clusters.
- **Müller 2019 Figure 4**: distillation degradation — teacher with label smoothing transfers worse to student.
- **Vaswani 2017 Section 5.4**: brief mention that `eps_ls = 0.1` "hurts perplexity but improves accuracy and BLEU" — a key practitioner intuition.

## Technical Details

**The loss** (standard cross-entropy with smoothed target):
```
y_smooth = (1 - eps) * one_hot(y) + eps / K            # K = vocab size
loss = - sum_k y_smooth(k) * log_softmax(logits)(k)
     = (1 - eps) * CE(logits, y) + eps * (- mean_k log_softmax(logits)(k))
     = (1 - eps) * CE(logits, y) + eps * KL(uniform || p_model) + const
```
Last line shows label smoothing = standard CE + a KL-to-uniform regularizer. The model is penalized for being too confident (logits too peaked).

**Effect on the optimal logit**:
- Without smoothing: optimal logit for the true class is `+inf` (with all others `-inf`); training never converges, only saturates.
- With smoothing `eps = 0.1, K = 30000`: optimal logit gap is finite, around `log((1-eps)*K / eps) ≈ 12.5`. Training has a real fixed point.

**Modern usage**:
| Phase | Label smoothing? | `eps` |
|---|---|---|
| LM pretraining | usually OFF | 0 |
| SFT | usually ON | 0.1 |
| Translation | ON | 0.1 (Vaswani default) |
| Image classification | ON | 0.1 |
| Reward model training | OFF | 0 |
| DPO / IPO | implicit (KL to ref policy) | — |
| PPO RLHF | implicit (KL penalty) | — |

**Why pretraining usually omits it**: pretraining produces token *probabilities* used downstream for RM scoring, perplexity-based eval, calibration, and (importantly) RLHF reference-policy KL. Label smoothing systematically inflates the entropy floor, making `pi_ref(y|x)` higher than reality and biasing every KL-regularized algorithm. (Some pretrains do use small `eps = 0.0–0.05`; Llama-1 used 0.)

**Why SFT keeps it**: SFT data is small and the model overfits hard. Label smoothing is one of the few regularizers that survives the move from CNN to Transformer (alongside weight decay).

**Common pitfalls**:
- Smoothing during reward-model training → biased RM that under-rewards confident-and-correct generations.
- Smoothing in the **prompt-token** loss positions when computing SFT loss (vs only target tokens) → inflated prompt-region loss; mask correctly.
- Combining `eps = 0.1` with high temperature sampling → the model never produces confident outputs at inference; can hurt downstream RL.
- Distillation: a teacher trained with label smoothing makes a *worse* teacher (Müller 2019). For [[orca]]-style distillation, train the teacher without smoothing.

**Variants**:
- **Unigram-smoothing**: `u(k) = unigram_freq(k)` instead of uniform. Mildly better for NMT.
- **Token-dropout / random label smoothing**: stochastic per-step. Rare.
- **Adaptive label smoothing**: `eps` decays with confidence. Niche.

## Connections
- **[[dropout]]**: complementary regularizer — dropout on intermediates, label smoothing on outputs.
- **[[kl-control-rlhf]]**: in PPO/DPO, the KL-to-reference penalty plays the role label smoothing plays in SFT — keeps probability mass spread, prevents collapse.
- **Entropy collapse in RL** ([[entropy-mechanism-llm-rl]]): models trained without label smoothing have *lower* baseline entropy entering RL; entropy collapses faster. Some practitioners reintroduce LS in SFT specifically to seed RL with higher-entropy policies.
- **Reward hacking**: well-calibrated outputs from LS-trained SFT models are slightly more robust to RM gaming (Coste 2023).
- **Karpathy** ([[karpathy-training-neural-net-recipe]]): not explicitly mentioned, but the "always look at calibration" advice points squarely at label smoothing's domain.
