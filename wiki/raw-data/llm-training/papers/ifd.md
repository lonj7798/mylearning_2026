<!-- scope: the Instruction-Following Difficulty (IFD) score as defined in arXiv:2308.12032, extracted as a standalone metric reference
     deps: [[cherry-llm]]
     see-also: [[cherry-llm-recipe]], [[superfiltering]], [[deita]], [[less]], [[alpagasus]]
-->

# Instruction-Following Difficulty (IFD) score — from "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning"
- **Core Insight:** IFD is the ratio of two averaged cross-entropy losses over the same response — the loss given the instruction divided by the loss without it — so it measures how much the instruction helps the model produce the response, and the paper selects the samples with the *highest* IFD below a threshold of 1 (§2.2, Eq. 3–5).
- **Guideline:** When ranking instruction samples with IFD, drop every sample with IFD > 1 as instruction–response misalignment, then take the top of the remaining ranking; the authors select about 5% of Alpaca and about 10% of WizardLM this way (§2.2; §4.1, Table 1).
- **Authors:** Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, et al. (Ping An Technology; University of Maryland)
- **Year:** 2023 (arXiv v1 2023-08; the text read here is arXiv v5, 2024-04-06)
- **URL:** https://arxiv.org/abs/2308.12032
- **Source type:** paper
- **Relevant topics:** SFT data selection, instruction difficulty, conditioned vs direct answer loss, data efficiency

This card covers only the metric. The full method, results and recipe are in [[cherry-llm]] and
[[cherry-llm-recipe]], which describe the same paper.

## Definition (exact, §2.2)
For a sample with complete instruction `Q` and answer `A` of `N` tokens, under model weights `θ`:

- Conditioned Answer Score (Eq. 3):
  `s_θ(A|Q) = L_θ(A|Q) = −(1/N) Σ_{i=1..N} log P(w_i^A | Q, w_1^A, …, w_{i−1}^A ; θ)`
- Direct Answer Score (Eq. 4):
  `s_θ(A) = −(1/N) Σ_{i=1..N} log P(w_i^A | w_1^A, …, w_{i−1}^A ; θ)`
- IFD (Eq. 5): **`IFD_θ(Q, A) = s_θ(A|Q) / s_θ(A)`**

Symbols: `w_i^A` is the i-th token of the answer; `N` is the number of tokens in the ground-truth answer;
`θ` are the weights of the pre-experienced model (the base model after a short warm-up, §2.1). Both terms
are averaged cross-entropy losses, so IFD is a ratio of losses, not of perplexities. The paper notes that
IFD is model-specific, and computes all values with the pre-experienced model (§2.2).

## Interpretation as stated by the authors (§2.2)
- Low IFD: the instruction already benefits the model's generation of the response without further
  training, which the authors describe as the easiness of the instruction.
- High IFD: the model cannot align its response to the instruction, which the authors take as the
  difficulty of the instruction. High-IFD samples are the "cherry" data used for retraining.
- IFD > 1: the conditioned loss exceeds the direct loss, meaning the instruction gives no useful context
  for predicting the response; the authors treat this as instruction–response misalignment and filter it
  out with a threshold of 1. The conditioned score is normally the smaller of the two because context
  makes later tokens easier to predict (§2.2).
- Isolating the answer's own difficulty is the stated reason for taking a ratio rather than using
  `s_θ(A|Q)` alone: a high conditioned loss can come from the answer string itself (§2.2).

## Computation as used in the paper (§2.1–2.2)
1. Embed each instruction with the base model by averaging the last hidden states over the instruction
   tokens (Eq. 1–2).
2. Run K-Means over those embeddings to form 100 clusters and sample 10 instances per cluster, giving
   1,000 pre-experienced samples (§2.1).
3. Train the base model on those 1,000 samples for 1 epoch to obtain the pre-experienced model (§2.1).
4. Compute `s_θ(A|Q)`, `s_θ(A)` and their ratio for every sample in the target dataset with that model.
5. Discard samples with IFD > 1, then take the highest-IFD samples as the training subset (§2.2).

## Technical Details
- Warm-up size is 1,000 samples = 100 clusters × 10 instances, selected by clustering rather than at
  random (§2.1). An ablation compares this "Diversity" selection against a "Difficulty" selection (IFD on
  the raw base model) and a "Random" selection, all at 1,000 samples (§4.3.2).
- Training subset sizes evaluated: top 5%, 10%, 15% and 20% of the dataset (§4.1, Figure 3).
- With about 5% of Alpaca, LLaMA-7B reaches Open LLM Leaderboard average 52.06 against 50.21 for the
  official Alpaca, and AlpacaEval 34.74 against 26.46 (§4.1, Table 1).
- With about 10% of WizardLM, LLaMA-7B reaches average 51.59 against 52.79 for the reimplemented
  full-data WizardLM, and AlpacaEval 61.44 against 61.99 (§4.1, Table 1).
- Human evaluation out of 100: Cherry Alpaca (5%) versus Alpaca (100%) is 49 / 25 / 26 (win/tie/loss);
  Cherry WizardLM (10%) versus reimplemented WizardLM (100%) is 37 / 32 / 31 (§4.1).
- Selecting the *lowest* IFD scores gives the worst performance of all selection strategies tested
  (§4.2.3, Figure 4), which is the paper's direct evidence that the ranking direction matters.
- Superfiltering (Li et al., 2024b) extends IFD in two ways the paper cites: good prompting can remove the
  need to train a pre-experienced model, and IFD scores from weak language models are consistent with
  those from strong ones, so small models can do the filtering (§2.2, closing paragraph).
- Training settings for the models trained on the selected data: Adam, learning rate 2 × 10⁻⁵, batch size
  128, 3 epochs; pre-experienced models get 1 epoch; max input length 512 for Alpaca and 1024 for WizardLM
  (App. A). The full ledger is in [[cherry-llm-recipe]].

## Findings relevant to generality
- The method transfers across base models: on LLaMA2-7B and LLaMA2-13B, cherry models trained on 5%, 10%
  or 15% of Alpaca beat the corresponding full-data Alpaca models on the Open LLM Leaderboard average —
  55.78 / 56.31 / 56.37 against 55.25 at 7B, and 61.21 / 61.02 / 61.23 against 58.78 at 13B (§4.4, Table 3).
- It transfers across datasets: the two datasets tested are Alpaca and a WizardLM subset of 63,655 entries
  after filtering "AI censure" instances (§4.1; App. A).

## Connections
- [[cherry-llm]] — the full card for this paper, including the three-phase method and all results.
- [[cherry-llm-recipe]] — the training-settings ledger for the same paper.
- [[superfiltering]] — computes IFD with a weak model and drops the pre-experienced training step.
- [[deita]], [[less]], [[alpagasus]] — alternative SFT selection scores compared in the same literature.
- [[lima]] — the data-quality-over-quantity result the authors cite as motivation (§1).

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2308.12032 (arXiv v5, 2024-04-06).
- Corrections to the previous card version:
  - "IFD(q, a) = PPL_cond(a|q) / PPL_uncond(a)", with both terms written as exponentials of the mean log
    probability → Eq. 5 defines IFD as the ratio of the two *averaged cross-entropy losses*
    `s_θ(A|Q) / s_θ(A)`, with no exponentiation (§2.2, Eq. 3–5). The two expressions are not equal.
  - "IFD < 1: instruction q reduces response uncertainty — the task is informative for the response" and
    "The hardest-but-valid samples cluster at IFD just below 1 — these are the cherry samples" → the paper
    assigns the opposite meaning to low scores: a low IFD means the instruction already helps, which the
    authors call easiness; cherry samples are the *high*-IFD samples under the threshold of 1 (§2.2).
  - "IFD ≈ 1: instruction irrelevant to response (noisy / decoupled pair)" → the paper places the
    misalignment case at IFD > 1, not at IFD ≈ 1 (§2.2).
  - "Warm the target LM on ~1K random samples for 1 epoch" → the 1,000 warm-up samples come from K-Means
    over instruction embeddings, 100 clusters × 10 instances; random selection is an ablation baseline,
    not the method (§2.1, §4.3.2).
  - "keep top-K (5–15%)" → the paper's main results use about 5% for Alpaca and about 10% for WizardLM,
    and evaluate 5/10/15/20% (§4.1, Figure 3).
  - "Authors: Ming Li et al." with no affiliation → full byline and affiliations added (title page). The
    arXiv PDF carries no venue line; the NAACL 2024 proceedings citation is recorded in [[cherry-llm]].
  - "Superfiltering: compute IFD using a smaller proxy (Qwen-0.5B)" → this paper only states that IFD
    scores from weak language models are consistent with strong ones; it names no proxy model (§2.2).
- Removed as unsupported by the source:
  - The "Practical guidance" block: BF16/FP16 stability of IFD, batch size 1, right-padding for the
    conditional pass, and per-PPL length normalization for long responses. None of these appear in the
    paper; the loss is already length-averaged by Eq. 3–4.
  - "Reflective IFD variants weight IFD by response length or token entropy" and "Reverse-IFD —
    PPL(q|a)/PPL(q) — for dialog data where prompts are long": no such variants are described.
  - "Use IFD as the first-pass filter on any SFT pool; combine with a diversity constraint (embedding or
    gradient space) for best results" — the paper uses clustering only to pick the 1,000 warm-up samples,
    and reports no combined IFD-plus-diversity selection over the full pool.
  - "Does not measure factual correctness; combine with an answer verifier for math/code" — not evaluated
    or claimed in the paper.
- Not reported by the source: cost of computing IFD over a pool; behaviour above 20% selection; results on
  non-LLaMA base models.
