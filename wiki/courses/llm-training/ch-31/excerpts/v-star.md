---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: arXiv:2402.06457v2 (library card papers/v-star.md has no Verification section and names the wrong authors; this chapter-local extract was checked against the primary source)
source_url: https://arxiv.org/abs/2402.06457
created_at: "2026-09-15"
---

# Excerpt: V-STaR: Training Verifiers for Self-Taught Reasoners

- **Authors:** Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, Rishabh Agarwal (Mila, Université de Montréal; Microsoft Research; University of Edinburgh; Google DeepMind)
- **Year:** 2024 (arXiv v1 2024-02; v2 2024-08-14; COLM 2024)
- **Source type:** paper
- **Used in:** ch-31 §1.1, §3.2, §4, Negative samples and negative feedback, Recipe, Generalization lens

## What the paper does

STaR, RFT, and ReST-EM fine-tune on correct self-generated solutions and discard incorrect ones (§1, §3). V-STaR also uses the incorrect solutions: all generated solutions, correct and incorrect, train a verifier with DPO, and the verifier ranks candidate solutions at test time (Abstract).

## Algorithm (§3, App. A Algorithm 1)

1. Fine-tune the pretrained model G_base on the original data D_SFT to obtain G_SFT; set D_GEN ← D_SFT.
2. For iteration 1…T: fine-tune G_base on D_GEN to get generator G; sample k solutions per training query; label each by final-answer match (math) or passing all tests (code).
3. Add correct solutions to D_GEN; add all solutions with labels to D_VER.
4. After T iterations, build preference pairs from D_VER (Cartesian product of correct and incorrect solutions per problem) and train the verifier V from G_SFT with DPO.

The generator is always fine-tuned from the base model on the accumulated correct samples, so correct samples from every earlier iteration stay in the training set (§3, Table 1).

## Verifier objective (§3.1, Eq. 2)

L_DPO(V; G_SFT) = −E_{(x, y⁺, y⁻)∼D_VER} log σ( r̂(x, y⁺) − r̂(x, y⁻) ), with r̂(x, y) = β log( V(y | x) / G_SFT(y | x) ).

y⁺ is a correct and y⁻ an incorrect solution for problem x; σ is the logistic function; β controls proximity to the reference G_SFT. At inference, V(ŷ | x) is the ranking score. The authors found DPO verifiers better than ORM-style verifiers (language modeling plus binary classification) when using LoRA (§3.1, §4.4, Fig. 5a).

## Setup (§4)

- Models: LLaMA 2 and CodeLLaMA, 7B and 13B, fine-tuned with LoRA.
- Tasks: GSM8K (math) and MBPP (code); transfer to a MATH subset of 150 level-1 problems (algebra, counting and probability, prealgebra, number theory with numeric answers) and HumanEval (§4, footnote 2).
- Data generation: k = 16 completions per query per iteration; 3 iterations; reference G_SFT trained 2 epochs (GSM8K) or 3 epochs (MBPP) (§4).
- Baselines at equal generation budget: SFT; STaR† (k = 16 for 3 iterations); RFT (3 × 16 samples in one iteration); SFT + verifier; V-STaR [1 iter] (§4.1).
- Metrics: Pass@1 for generators; Best-of-64 for verifier methods, computed from 128 candidates per test problem with the unbiased estimator in Eq. 3 (§4.1–4.2).

## Results (text statements; exact bar values appear only in figures)

- V-STaR improves test accuracy by 4–17 points over existing self-improvement and verification approaches (Abstract); 6–17 points on math and 4–12 on code (§1).
- 7B V-STaR surpasses base LLaMA 2 70B (8-shot) on GSM8K and nearly matches CodeLLaMA 34B zero-shot on HumanEval (§1).
- Iterative V-STaR outperforms V-STaR [1 iter] at the same generation budget and outperforms baselines on the transfer tasks for both sizes (§4.3).
- A fourth MBPP iteration added 0.3 points (§4.3).
- Best-of-k saturates for k ≥ 16 (§4.3, Fig. 5); for k ≤ 64 the authors describe V-STaR as "far more effective than majority voting", and the gap decreases slightly at larger k (§4.5, Fig. 6).
- DPO verifier as a generator: Pass@1 degrades after a small number of updates while Best-of-64 rises within 2k updates (§4.6, Fig. 6 right).
- Verifier in the training loop (64 samples, keep top 8 correct by verifier score, fill with incorrect up to 16): after three MBPP iterations, Best-of-64 53.2 and Pass@1 46.34; the authors state this does not provide a substantial gain (§4.7).
- Gains across iterations on MBPP are larger for verifiers than for generators, with no sign of collapse (§4.8, Fig. 7).

## Negative-sample classification (course standard §6.1)

For the verifier, incorrect solutions are negative as gradient: the DPO term lowers their likelihood relative to G_SFT. For the generator, incorrect solutions have zero weight (negative marginal value).

## Verification

- Checked on 2026-09-15 against: https://arxiv.org/abs/2402.06457 (v2, 2024-08-14), main text and Appendix A.
- Differences from the library card `papers/v-star.md`: authors are Hosseini et al., not Zelikman et al.; the verifier is a DPO model trained on correct/incorrect pairs; the paper reports that putting the verifier in the training loop "does not provide a substantial gain", whereas the card describes the verifier as selecting future training data.
- Not reported in text form: per-bar values of Figures 2–4 and 7.
