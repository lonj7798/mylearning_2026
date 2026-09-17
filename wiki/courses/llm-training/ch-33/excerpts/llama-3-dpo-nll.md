---
chapter: ch-33
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from arXiv:2407.21783v3 §4.1–4.2)"
---

# Excerpt: Llama 3 post-training round — preference data, DPO modifications, and model averaging

**Primary source:** "The Llama 3 Herd of Models", arXiv:2407.21783v3, §4.1.2–4.1.6, §4.2.1–4.2.2, §4.3.4.
**Library cards:** [[llama-3]], [[llama-3-recipe]] (both verified 2026-09-14). The previous version of this excerpt quoted "single epoch per round; masks prompts from loss" and "most-recent-batch preference data only (older batches cause format drift)". The report gives no DPO epoch count, masks header and termination tokens rather than prompts, and says "primarily" the most recent batches, with the reason being closeness to the policy distribution.

## Quotes

**Iterative rounds (§4.1.6).** "Following Llama 2, we apply the above methods in six rounds. In each cycle, we collect new preference annotations and SFT data, sampling synthetic data from the latest models."

**Reward model (§4.1.2).** "The training objective is the same as Llama 2 except that we remove the margin term in the loss, as we observe diminishing improvements after data scaling. Following Llama 2, we use all of our preference data for reward modeling after filtering out samples with similar responses."

**Preference data (§4.2.1).** "In each round of post-training, we use all the preference data that is available at the time for reward modeling, while only using the latest batches from various capabilities for DPO training. For both reward modeling and DPO, we use samples that are labeled as the chosen response being significantly better or better than the rejected counterpart for training and discard samples with similar responses."

**Rejection sampling (§4.2.2).** "…for each prompt collected during human annotation (Section 4.2.1) we sample K (typically between 10 and 30) outputs from the latest chat model policy (usually the best performing checkpoint from the previous post-training iteration, or the best performing checkpoint for a particular capability) and use our reward model to select the best candidate…"

**SFT (§4.1.3).** "Our largest models are finetuned with a learning rate of 10⁻⁵ over the course of 8.5K to 9K steps. We found these hyperparameter settings to work well across different rounds and data mixes."

**DPO data and algorithm choice (§4.1.4).** "For training, we primarily use the most recent batches of preference data collected using the best performing models from the previous alignment rounds. As a result, our training data conforms better to the distribution of the policy model that is being optimized in each round. We also explored on-policy algorithms such as PPO (Schulman et al., 2017), but found that DPO required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval (Zhou et al., 2023). For Llama 3, we use a learning rate of 10⁻⁵ and set the β hyper-parameter to be 0.1."

**Masking formatting tokens (§4.1.4).** "We mask out special formatting tokens including header and termination tokens … from both chosen and rejected responses in the loss to stabilize DPO training. We observe that having these tokens contribute to the loss may lead to undesired model behaviors such as tail repetition or abruptly generating termination tokens. We hypothesize that this is due to the contrastive nature of the DPO loss – the presence of common tokens in both chosen and rejected responses leads to a conflicting learning objective as the model needs to increase and reduce the likelihood of these tokens simultaneously."

**NLL regularization (§4.1.4).** "We add an additional negative log-likelihood (NLL) loss term with a scaling coefficient of 0.2 on the chosen sequences, similar to Pang et al. (2024). This helps further stabilize DPO training by maintaining desired formatting for generation and preventing the decrease of log probability of chosen responses (Pang et al., 2024; Pal et al., 2024)."

**Model averaging (§4.1.5).** "Finally, we average models obtained from experiments using various versions of data or hyperparameters at each RM, SFT, or DPO stage…"

**Short-context DPO (§4.3.4).** "We observe that using only short context training data in DPO did not negatively impact long-context performance as long as the SFT model is high quality in long context tasks. We suspect this is due to the fact that our DPO recipe has fewer optimizer steps than SFT."

## Resulting loss, as written in ch-33

L = L_DPO(β = 0.1, formatting tokens masked) + 0.2 · NLL(y_chosen | x)

- L_DPO: standard DPO loss ([[dpo]]) on chosen and rejected responses.
- NLL(y_chosen | x): negative log-likelihood of the chosen response; per-token or per-sequence normalization is not reported.

## Not reported

DPO epochs, batch size, and steps; total number of preference comparisons; model-averaging weights; any numeric ablation of the NLL coefficient, token masking, or the DPO-versus-PPO comparison ([[llama-3-recipe]]).

## Used in

ch-33 §5.1, §5.2, §7, Negative samples and negative feedback, Recipe.
