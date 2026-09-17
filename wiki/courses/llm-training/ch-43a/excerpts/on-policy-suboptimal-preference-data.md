---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/on-policy-suboptimal-preference-data.md on 2026-09-15)
source_url: https://arxiv.org/abs/2404.14367
source_version: arXiv v3 (2024-06-02); v1 2024-04-22
created_at: "2026-09-15"
---

# Excerpt: Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data (Tajwar, Singh, Sharma, Rafailov, Schneider, Xie, Ermon, Finn, Kumar)

Facts used by [[read]], read in the arXiv v3 PDF on 2026-09-15.

## Definitions and scope
- Methods are classified along three axes: on-policy sampling, sample reuse, and negative gradient, where a negative gradient is a term that "attempts to push down likelihood on certain responses" (§3, Table 1). On-policy RL with a mean-subtracted reward also contains a negative gradient, because responses with below-average reward get a negative multiplier (§3).
- Experiments: a didactic bandit, synthetic LLM length tasks with three coverage/geometry conditions, and full-scale AlpacaFarm (Pythia-1.4B) and UltraFeedback (Mistral-7B) runs (§4-§5).

## Takeaways used here
- Negative gradient (§5.2, takeaway box): "A negative gradient improves over offline supervised methods when the peak in the reward appears in less likely regions of π_ref. It can increase the likelihood of y_w when y_l is sufficiently different from y_w, model capacity is large, and π_ref is chosen appropriately. If not, the margin log π_θ(y_w|x) − log π_θ(y_l|x) will still be larger when a negative gradient is used, but the recovered probability mass will go into increasing likelihoods of other responses, not y_w."
- Figure 16 (bandit): with few prompts, IPO raises the implicit reward of y_w and lowers it for y_l; as the number of prompts grows, the implicit reward of y_w also decreases.
- Figure 17: for Pythia-1.4B on AlpacaFarm, DPO decreases the implicit reward of both y_w and y_l, while Pref-FT (likelihood training on y_w only) increases both; for Mistral-7B on UltraFeedback, DPO raises y_w and lowers y_l. The authors attribute the difference to the UltraFeedback responses being more semantically distinct (responses from models of different capability) and to model capacity (§5.2.2, Interpretation).
- On-policy sampling and negative gradients are complementary (§5.3): on-policy DPO/IPO converged faster and to better solutions than offline DPO, on-policy RL, and on-policy supervised variants in the bandit and synthetic LLM settings (Figures 18-19).
- Unification (§6): on-policy sampling and negative gradients are both described as mode-seeking (reverse-KL-like), against mode-covering maximum likelihood.
