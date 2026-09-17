---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2404.14367
created_at: "2026-09-15"
verified_against: "arXiv:2404.14367v3 (2 Jun 2024), cached plain text"
---

# Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data (Tajwar et al.)

## Claims the chapter uses
- **Framework (§4, Table 1).** Methods are classified along three axes: on-policy sampling, sample reuse,
  and negative gradient, defined as explicitly minimising a term that pushes down the likelihood of
  certain responses. On-policy RL with a mean-centred advantage also contains a negative-gradient term
  (§4, around Eq. for `r̄_φ(x,y') < 0`).
- **When on-policy sampling and negative gradients help (§5, Fig. 1 right).** They help when the peak of
  the reward function lies in a region that the reference policy assigns low probability; when the reward
  peak is already in a high-likelihood region of `π_ref`, offline supervised methods perform comparably.
  These conclusions come from didactic bandit problems and synthetic length-control LLM problems
  ("Min Length", "Mode Length", "Skew Length") plus AlpacaFarm and UltraFeedback runs.
- **What the negative gradient does to likelihoods (§5.2.2, Fig. 17).** Pythia-1.4B on AlpacaFarm: DPO
  lowers the implicit reward `β[log π_θ(y|x) − log π_ref(y|x)]` for both the chosen and the rejected
  response, while Pref-FT (supervised fine-tuning on chosen responses only) raises both. Mistral-7B on
  UltraFeedback, starting from an UltraChat-200K SFT model: DPO raises the chosen implicit reward and
  lowers the rejected one. The authors attribute the difference to the responses in UltraFeedback being
  semantically more distinct (different generator models) and to model capacity.
- **Stated takeaway for negatives (§5.2.2 box).** A negative gradient "can increase the likelihood of
  `y_w` when `y_l` is sufficiently different from `y_w`, model capacity is large, and `π_ref` is chosen
  appropriately. If not, the margin ... will still be larger ..., but the recovered probability mass will
  go into increasing likelihoods of other responses, not `y_w`."
- **On-policy pairs plus a negative gradient (§5.3).** Sampling N responses per prompt from the current
  policy, ranking them with a reward model, and applying the DPO or IPO loss converged faster and to a
  better solution than offline DPO, on-policy RL, and on-policy supervised variants in the bandit and
  synthetic LLM settings (Figs. 18–19). The authors state this does not mean on-policy DPO always beats PPO.
