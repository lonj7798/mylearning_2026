<!-- scope: SPIN — Self-Play Fine-Tuning as DPO with human text as chosen
     deps: [[dpo]]
     see-also: [[self-rewarding-lm]], [[self-play-preference]], [[trl-online-dpo]]
-->

# SPIN — Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
- **Core Insight:** SFT data alone yields a DPO signal if you treat the human-written response as "chosen" and the *previous iteration's* model sample as "rejected" — no reward model, no human preference labels, just iterated self-play until the policy distribution matches the data.
- **Guideline:** When you have SFT data but no preference data, run SPIN: each iteration, sample a response from π_{t−1} and DPO-train π_t with (human_response, π_{t−1}_sample) as the chosen/rejected pair.
- **Authors:** Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu
- **Year:** 2024 (UCLA)
- **URL:** https://arxiv.org/abs/2401.01335
- **Relevant topics:** self-play, preference optimization without preferences, iterative DPO, distribution matching

## Abstract
Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong LLM out of a weak one without the need for acquiring additional human-annotated data. We propose a new fine-tuning method called Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Our method progressively elevates the LLM from a nascent model to a formidable one, unlocking the full potential of human-annotated demonstration data for SFT. Notably, by adopting SPIN, we can achieve a performance similar to that from DPO training with extra GPT-4 preference data on the HuggingFace Open LLM Leaderboard.

## Key Contributions
- Frames SFT-only post-training as a **two-player game**: the policy plays against its previous iteration; Nash equilibrium is reached when π_t = data distribution.
- Provides a closed-form DPO-equivalent update: `L_SPIN = −logσ(β·(logπ(y_human)/π_{t−1}(y_human) − logπ(y_gen)/π_{t−1}(y_gen)))`.
- Empirically matches DPO-with-GPT-4-preferences on Zephyr 7B using **only the Ultrachat SFT dataset** — no preference labels.
- Shows monotone improvement across 3 SPIN iterations (MT-Bench 6.39 → 7.12).

## Key Figures/Tables to Study
- **Figure 1 (SPIN loop):** iteration t reads human y from SFT data, samples y' from π_{t−1}, DPO-optimizes π_t.
- **Table 2 (HF Open LLM Leaderboard):** Zephyr-7B-SFT → +SPIN matches Zephyr-7B-DPO (which used 60K GPT-4 preferences).
- **Theorem 4.1:** Nash equilibrium characterization — when policy generates the human distribution, the SPIN loss becomes 0.

## Technical Details
- **Base:** mistral-7B SFT'd on UltraChat-200K.
- **Per iteration:**
  1. Sample 50K (prompt, response) pairs from π_{t−1} at T=1.0.
  2. Build DPO pairs (`chosen=y_human`, `rejected=y_gen`) 1:1 with the SFT data.
  3. DPO-train π_t from π_{t−1}: β=0.1, lr=5e-7, 3 epochs, batch 64.
  4. Reset reference to π_{t−1} for the next iteration.
- **Hyperparameters that matter:** the β=0.1 DPO temperature (higher collapses to SFT), the 1:1 human:generated pair ratio (off-ratio hurts), and iteration count (monotone gain for 3, minor gain after).
- **Budget:** ~8× the SFT compute — one full SFT round per iteration.

## Connections
- Shares the "policy judges itself" motif with [[self-rewarding-lm]], but SPIN's "judge" is the *human-written text itself* — far lower variance than an LLM-judge.
- The DPO algebra is identical to [[dpo]]; only the preference source changes.
- Related to [[rejection-sampling-finetuning]] (Llama-2 RSFT): both refine via self-samples, but RSFT trains on `chosen_only`, SPIN uses both chosen and rejected.
- Theoretical foundation overlaps with [[self-play-preference]] (Nash-LM): both cast alignment as a two-player game; SPIN's equilibrium is distribution-matching, Nash-LM's is preference-matching.
- Supports the [[west-of-n]] synthetic-preference thesis: preference pairs can be fabricated from existing data sources (human vs model samples) rather than collected from humans.
