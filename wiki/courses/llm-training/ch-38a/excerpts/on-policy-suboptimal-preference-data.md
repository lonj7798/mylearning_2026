---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/on-policy-suboptimal-preference-data.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2404.14367
created_at: "2026-09-15"
---

# Excerpt: Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data

**Authors:** Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, Aviral Kumar (CMU, Stanford, Google DeepMind, UW-Madison).
**Version read:** arXiv:2404.14367v3 (2 Jun 2024).
**Status:** no library card existed for this slug on 2026-09-15; quotes read at the stated loci in the v3 PDF. Note that `wiki/raw-data/llm-training/papers/on-off-policy-rlhf.md` records a previous card that confused this paper's arXiv id with Tang et al.'s title; the two papers are different studies.

## What is compared (§3, Table 1)
Methods are grouped by three axes: on-policy sampling, sample reuse, and negative gradient. Representatives: PPO and on-policy best-of-N (on-policy, no explicit negative gradient in best-of-N), DPO and IPO (offline, negative gradient), Pref-FT, Binary FeedME and offline best-of-N / RWR (offline, no negative gradient), and on-policy DPO/IPO (both). Settings: didactic bandit problems, synthetic LLM length-control problems, and full-scale runs on AlpacaFarm (Pythia-1.4B reference policy) and UltraFeedback (Mistral-7B).

## Takeaway boxes (§5.1, §5.2, §5.3)
- On-policy sampling: "On-policy sampling generally improves performance and efficiency, especially in cases when the peak of reward appears farther from the reference policy, even when the reward model is learned from the same preference dataset that methods without on-policy learning also use. In some cases, sample reuse can reduce the dependency on on-policy sampling of data, but it presents a tradeoff by reducing the exploration of the response space."
- Negative gradients: "A negative gradient improves over offline supervised methods when the peak in the reward appears in less likely regions of π_ref. It can increase the likelihood of y_w when y_l is sufficiently different from y_w, model capacity is large, and π_ref is chosen appropriately. If not, the margin log π_θ(y_w|x) − log π_θ(y_l|x) will still be larger when a negative gradient is used, but the recovered probability mass will go into increasing likelihoods of other responses, not y_w."
- Combination: "On-policy sampling and offline negative gradients present complementary benefits … while sampling responses on policy provides coverage of the response space, an effective negative gradient loss provides a stronger learning signal given a set of samples."

## Implicit-reward behaviour (Fig. 17)
For Pythia-1.4B on AlpacaFarm, DPO decreases the implicit reward r_θ(x, y) = β[log π_θ(y|x) − log π_ref(y|x)] for both y_w and y_l, while Pref-FT increases both. For Mistral-7B on UltraFeedback, DPO increases the reward for y_w and decreases it for y_l. In both cases DPO produces the larger margin. The authors attribute the difference partly to UltraFeedback's responses being more semantically distinct (§5.2.2).

## Sample reuse (§5.1.2)
With T inner gradient steps per batch of on-policy data, T = 2 beat T = 1 in the didactic setting, while large T hurt; PPO tolerated reuse better than on-policy best-of-N, which the authors attribute to PPO's off-policy correction removing significantly off-policy samples from the gradient.

## Unification (§6)
On-policy RL and on-policy weighted-likelihood methods optimize a regularized reverse KL and are mode-seeking (Lemma 6.1); offline contrastive methods with a negative gradient are also described as mode-seeking; weighted maximum-likelihood objectives (Pref-FT, offline best-of-N, Binary FeedME) are mode-covering.

## How ch-38a uses it
§5 (conditions under which on-policy sampling and negative gradients pay off), Negative-feedback section (where displaced mass goes), §9 (what to log), Connections to ch-39 and ch-43a.
