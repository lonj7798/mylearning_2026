---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2310.16944
primary_version: arXiv:2310.16944 (Zephyr technical report, PDF read 2026-09-15)
created_at: "2026-09-15"
---

# Excerpt: Zephyr — Direct Distillation of LM Alignment

Tunstall et al. (Hugging Face), 2023. ch-45a uses Zephyr as the fully off-policy end of the preference ledger.

## Preference pairs (§4.1)
> "UltraFeedback (Cui et al., 2023) consists of 64k prompts, each of which have four LLM responses that are
> rated by GPT-4 according to criteria like instruction-following, honesty, and helpfulness. We construct binary
> preferences from UltraFeedback by selecting the highest mean score as the 'chosen' response and one of the
> remaining three at random as 'rejected'. We opted for random selection instead of selecting the lowest-scored
> response to encourage diversity and make the DPO objective more challenging. As noted above, this step is
> computed offline and does not involve any sampling from the reference model."

Neither side of the pair is sampled from the model being trained: both responses come from other models scored
by GPT-4. The paper calls the method dDPO (distilled DPO).

## DPO settings (§4.4)
> "Similar to SFT, we train our DPO models for one to three epochs. We use a linear learning rate scheduler with
> a peak learning rate of 5e-7 and 10% warmup steps. We train all models with a global batch size of 32 and use
> β = 0.1 from Eq. (1) to control the deviation from the reference model. The final ZEPHYR-7B model was
> initialized from the SFT model that was trained for one epoch and further optimized for three DPO epochs."

SFT for comparison (§4.3): cosine schedule, peak LR 2e-5, 10% warmup, global batch 512, packing at sequence
length 2,048, one to three epochs. Base model: Mistral-7B-v0.1; SFT data: UltraChat, filtered to ~200k examples.

## The overfitting observation (§5, Figure 3)
> "In the process of training ZEPHYR-7B we observed that after one epoch of DPO training, the model would
> strongly overfit, as indicated by perfect training set accuracies in Figure 3. Surprisingly, this did not harm
> downstream performance on MT-Bench and AlpacaEval; as shown in Figure 3, the strongest model was obtained
> with one epoch of SFT followed by three epochs of DPO. However, we do observe that if the SFT model is
> trained for more than one epoch, the DPO step actually induces a performance regression with longer training."

Reported results (Table 1): Zephyr-7B, dDPO, MT-Bench 7.34, AlpacaEval win rate 90.60 ± 1.03.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2310.16944 (scratchpad `sources/zephyr.txt`),
  §4.1, §4.3, §4.4, §5, Table 1.
- Not reported: optimizer betas, weight decay, max sequence length for the DPO stage, number of DPO pairs
  actually used after filtering (only the 64k prompt count of UltraFeedback is given).
