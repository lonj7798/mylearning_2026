<!-- scope: neural scaling laws — power-law relationships for parameters, data, compute
     deps: [[gpt-2]]
     see-also: [[chinchilla]], [[gpt-3]]
-->

# Scaling Laws for Neural Language Models
- **Core Insight:** Loss follows clean power laws in parameters, data, and compute, spanning seven orders of magnitude; architecture details barely matter.
- **Guideline:** Use power-law fits to forecast loss before committing to a large training run; budget model size vs. data vs. steps along the scaling frontier.
- **Authors:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2001.08361
- **Relevant chapters:** Scaling laws, power-law relationships, compute budgets, training efficiency, model sizing

## Abstract
We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence.

## Key Contributions
- Discovered that language model loss follows smooth power-law relationships with model size (N), dataset size (D), and compute (C), spanning 7+ orders of magnitude
- Showed that architectural details (width vs. depth ratio, attention heads) have minimal effect on loss compared to total parameter count, within a broad range
- Established that larger models are more sample-efficient: they achieve the same loss with less data per parameter, which means compute-optimal training should favor larger models trained for fewer steps
- Provided simple parametric equations for predicting loss as a function of N, D, and C: L(N) ~ N^(-0.076), L(D) ~ D^(-0.095), L(C) ~ C^(-0.050)
- Formalized the concept of compute-optimal allocation, arguing that given a fixed compute budget, most of it should go to increasing model size rather than training duration (later revised by Chinchilla)

## Key Figures/Tables to Study
- **Figure 1** (Loss vs. compute, dataset size, and parameters): The foundational result -- three panels showing clean power-law trends. This is one of the most cited figures in the scaling literature.
- **Figure 2** (Performance vs. model shape): Demonstrates that width-to-depth ratio barely matters; only total parameter count drives loss. Key evidence for the "scaling hypothesis."
- **Figure 4** (Data requirements scaling): Shows how much data is needed to avoid overfitting for a given model size. Critical for training budget planning.
- **Figure 6** (Sample efficiency of large models): Larger models reach any given loss level with fewer training samples, supporting the "train big, stop early" strategy.
- **Figure 9** (Optimal compute allocation): Given a 10x increase in compute, should you train 10x longer or make the model 10x bigger? This figure answers that question.

## Architecture Details
- **Architecture studied:** Decoder-only Transformer (GPT family)
- **Model sizes tested:** 768 to 1.5 billion parameters (with additional smaller models)
- **Power-law exponents:**
  - Loss vs. parameters: L(N) ~ (N_c / N)^0.076 where N_c ~ 8.8e13
  - Loss vs. data: L(D) ~ (D_c / D)^0.095 where D_c ~ 5.4e13
  - Loss vs. compute: L(C_min) ~ (C_min_c / C_min)^0.050
- **Key finding on shape:** Width-to-depth ratio and number of attention heads have negligible effect on loss for a fixed parameter count
- **Overfitting threshold:** Models overfit when N^0.74 / D exceeds a critical ratio
- **Training speed:** Larger models reach any given loss in fewer optimization steps (but each step costs more)
- **Optimal allocation recommendation:** Given a 10x compute increase, allocate ~5.5x to model size and ~1.8x to training duration (note: Chinchilla later revised this to ~3.16x for each)
- **Context length:** 1024 tokens
- **Tokenization:** BPE (same as GPT-2)
- **Dataset:** WebText2
- **Training framework:** Consistent hyperparameters across scales to isolate the effect of model size
