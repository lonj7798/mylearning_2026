<!-- scope: Cherry LLM (Li et al., NAACL 2024): self-guided instruction-data selection with the Instruction-Following Difficulty (IFD) score
     deps: [[alpaca]], [[wizardlm]]
     see-also: [[ifd]], [[superfiltering]], [[alpagasus]], [[lima]], [[deita]], [[less]], [[cherry-llm-recipe]]
-->

# From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning
- **Core Insight:** Selecting the instruction samples with the highest Instruction-Following Difficulty (IFD, the ratio of the answer's mean loss with the instruction to its mean loss without it, kept only when ≤ 1) lets LLaMA-7B trained on about 5% of Alpaca beat the official Alpaca model and LLaMA-7B trained on about 10% of WizardLM beat a reimplemented full-data WizardLM in GPT-4 pairwise judging (§4.1, Figure 2).
- **Guideline:** When selecting a subset of an instruction dataset for a 7B–13B LLaMA-family base model, rank samples by IFD computed with that model and start from the top 10%, which the authors call "a safe and reasonable choice" based on their Alpaca and WizardLM experiments (App. G.2).
- **Authors:** Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, et al. (Ping An Technology; University of Maryland)
- **Year:** 2023 (arXiv v1 2023-08; NAACL 2024 main conference, pp. 7595–7628)
- **URL:** https://arxiv.org/abs/2308.12032
- **Source type:** paper
- **Relevant topics:** SFT data selection, IFD score, instruction difficulty, data efficiency, LLM-as-judge evaluation

## Abstract
The paper proposes a self-guided method in which an LLM selects "cherry samples" from open-source instruction datasets, reducing manual curation and cost. Its central metric, the Instruction-Following Difficulty (IFD) score, measures the discrepancy between the response a sample expects and the model's own ability to generate it. Experiments on Alpaca and WizardLM show improved results with 10% of the original data. Code, data, and models are released at github.com/tianyi-lab/Cherry_LLM.

## Key Contributions
- A three-phase method: Learning from Brief Experience (train a pre-experienced model), Evaluating Based on Experience (compute IFD for every sample), Retraining from Self-Guided Experience (train on the selected data) (§2, Figure 1).
- The IFD score with a threshold of 1 for instruction–response misalignment (§2.2, Eq. 3–5).
- Selection experiments with LLaMA-7B on Alpaca (52,002 samples) and WizardLM70K, and with LLaMA2-7B/13B without a pre-experienced model (§3.1, §4.1, §4.4).
- Ablations against random, K-means diversity, low-IFD, and high-loss selection, and on the size and composition of the pre-experience set (§4.2, §4.3, App. G.1).
- An analysis of which instructions receive high or low IFD (§5, App. E).

## Key Figures/Tables to Study
- Figure 2 and Figure 3: pairwise comparisons with full-data models; winning score over 5–20% subsets.
- Figure 4 and App. B Table 5: selection-mechanism ablation.
- Figure 5 and Table 2: number and distribution of pre-experience samples.
- Table 4: top verb–noun pairs in the top and bottom 5% by IFD.
- App. C Tables 6–7: results by sub-category, including the math and coding losses.

## Technical Details
**Scores (§2.2).** A sample has question Q (instruction plus optional input, formatted with the dataset template) and answer A with N tokens.
- Conditioned Answer score (Eq. 3): `s(A|Q) = −(1/N) Σ_{i=1..N} log P(w_i | Q, w_1 … w_{i−1}; θ)`
- Direct Answer score (Eq. 4): `s(A) = −(1/N) Σ_{i=1..N} log P(w_i | w_1 … w_{i−1}; θ)`
- IFD (Eq. 5): `IFD(Q, A) = s(A|Q) / s(A)`
- θ = weights of the pre-experienced model; w_i = i-th answer token; N = number of answer tokens.
- A higher IFD means the instruction gives less help, which the paper reads as harder instruction following (§1, §2.2). IFD > 1 means the instruction makes the answer harder to predict; such samples are treated as misaligned and removed (§2.2). The method trains on samples with "relatively large IFD scores" (§1).
- The released script computes the same ratio of mean token losses (conditioned / direct), skips ratios above 1, sorts ascending, and keeps the last `sample_number` entries (Cherry_LLM@01fac8d `cherry_seletion/data_by_IFD.py` L103–L120). The score is a ratio of mean losses, not of perplexities.
- Worked example (App. F, Figure 9): "Give a brief description of the coronavirus." has s(A) = 0.761 and s(A|Q) = 0.696, so IFD = 0.696 / 0.761 = 0.915 (printed 0.914). "What emotion is expressed in this tweet?" with answer "Frustration" has 0.601 / 6.593 = 0.091. The first example in Figure 9 prints 3.337 / 3.970 with IFD 0.928, but 3.337 / 3.970 = 0.841, so one of the three printed values is inconsistent.

**Pre-experience phase (§2.1, §4.3).** Instruction embeddings are the mean last-layer hidden states of the base model over the question tokens (Eq. 1–2). K-means forms 100 clusters; 10 samples per cluster give 1,000 samples, trained for 1 epoch (§2.1). With 0 pre-experience samples, results are the lowest among the tested counts but still beat Alpaca at 10% data; 100 samples are slightly better than 0; 300 samples give "a distinct performance gain"; 500 give no further gain (§4.3.1, Figure 5). Winning scores vs official Alpaca (ChatGPT judge) at 5/10/15% are 1.057/1.072/1.096 when the 1,000 samples are chosen by difficulty, 1.050/1.097/1.064 by diversity, and 1.007/1.047/1.077 at random (Table 2). Using the fully trained Alpaca model as the scorer gives 0.968/0.999/1.005 (App. G.1, Table 9).

**Evaluation protocol (§3.1, §3.3).** Test sets: Vicuna, Koala, WizardLM, Self-instruct, and LIMA, about 1,000 instructions in total. GPT-4 or ChatGPT scores both responses from 1 to 10, in both orders; a model wins if it is better in both orders or wins one and ties the other. Winning score = (Num(Win) − Num(Lose)) / Num(All) + 1 (§4.1). Human evaluation: 100 instructions (20 per test set), 3 participants, majority vote (§3.3.3). Open LLM Leaderboard (ARC, HellaSwag, MMLU, TruthfulQA) and AlpacaEval are also reported (§3.3.2).

**Results.**
- Human evaluation: Cherry Alpaca (5%) vs Alpaca (100%) 49 wins / 25 ties / 26 losses; Cherry WizardLM (10%) vs reimplemented WizardLM (100%) 37 / 32 / 31 (§4.1).
- Table 1 (LLaMA-7B; leaderboard average, AlpacaEval): official Alpaca 50.21, 26.46; 5% Alpaca 52.06, 34.74; reimplemented WizardLM 52.79, 61.99; 10% WizardLM 51.59, 61.44. The 10% WizardLM model is lower on both metrics; the paper calls it "a close performance" (§4.1).
- Table 3 (LLaMA2, full vs 5%/10%/15%; leaderboard average): 7B 55.25 vs 55.78/56.31/56.37; 13B 58.78 vs 61.21/61.02/61.23. AlpacaEval at 5%: 7B 27.75 → 36.78; 13B 35.00 → 46.82 (§4.4). The repository README prints MMLU 44.91 for 5% LLaMA2-7B where the paper prints 44.19; the paper value matches its printed average.
- App. B Table 5 (5% subsets, leaderboard average): IFD 52.06, random 50.61, diversity 49.48, low IFD 50.77, high Conditioned Answer score 47.51. Human evaluation of IFD vs each: 58/23/19, 61/21/18, 87/8/5, 76/15/9 (win/tie/lose).
- App. D Table 8: official WizardLM 54.18 average and 67.64 AlpacaEval vs 40% WizardLM 52.83 and 65.09.

**Selected data (§5, App. E).** Top-5% and bottom-5% IFD samples occupy separated regions of a t-SNE map rather than spreading uniformly (§5.1, Figure 6). Most frequent verb–noun pairs: top 5% "write story" 119, "generate story" 98, "generate list" 66, "explain concept" 48; bottom 5% "rewrite sentence" 155, "edit sentence" 89, "change sentence" 37, "classify sentence" 36 (Table 4). ChatGPT ratings of 100 samples from each end score high-IFD instructions higher on Scope, Complexity, Depth, and Knowledge Required and lower on Clarity and Simplicity (App. E, Figure 8).

## Recipe ledger
Full table in [[cherry-llm-recipe]]. Main paper values (arXiv:2308.12032v5 App. A, verified 2026-09-14): LLaMA-7B, Adam, LR 2 × 10⁻⁵, batch size 128, 3 epochs; pre-experienced model 1 epoch on 1,000 samples; max input length 512 (Alpaca) and 1024 (WizardLM); LLaMA2 runs max length 2048 with the Vicuna prompt. The paper gives no LR, batch, or epochs for the LLaMA2 runs; the README gives 1e-5 and 5 epochs for LLaMA2-13B.

## Findings relevant to generality and negative samples
- Category losses: the 5% Alpaca model does worse than official Alpaca on Math and Coding sub-categories; the 10% WizardLM model does worse on Math, Code, Complex Format, and Counterfactual. The authors attribute this to these categories needing more data than the selected subset contains (App. C, Tables 6–7) (Interpretation by the authors).
- Knowledge benchmark: MMLU falls on LLaMA-7B after selection (Alpaca 41.73 → 36.51 at 5%; reimplemented WizardLM 37.75 → 33.08 at 10%) while pairwise and AlpacaEval results rise or stay close; on LLaMA2-7B it falls from 47.02 to 44.19 at 5% and on LLaMA2-13B it rises from 54.05 to 55.65 at 5% (Tables 1, 3). The paper does not discuss the LLaMA-7B MMLU drop.
- Distribution: IFD selection concentrates on open-ended generation (stories, lists, explanations) and removes editing and rewriting tasks (§5.2, Table 4). Selecting only by K-means diversity performs "similar to the random trained models" (§4.2.2).
- Scorer dependence: IFD is a model-specific value (§2.2). A fully trained Alpaca model as scorer selects data that trains the raw model worse than the pre-experienced scorer (App. G.1, Table 9).
- Negative marginal value (sense 1 of the course standard): training on the lowest-IFD subsets gives the lowest winning scores of all strategies (§4.2.3, Figure 4). Samples with IFD > 1 are discarded; the method uses no negative gradient.
- Correctness: the score uses only model losses on the given answer (Eq. 3–5); the paper does not evaluate whether selected answers are factually correct.
- Measurement limits: evaluation relies on GPT-4/ChatGPT pairwise judging over about 1,000 instructions plus 100 human-rated instructions; no seeds or variance are reported (§3.3).

## Connections
- [[ifd]] — standalone card for the metric; the definition to use is the loss ratio of Eq. 5.
- [[superfiltering]] — follow-up by overlapping authors; cited in §2.2 for showing that good prompting can "relieve the burden" of training a pre-experienced model and that IFD scores from weak models are consistent with those from strong models.
- [[alpagasus]] — described in §6.3 as scoring each sample with an external LLM (ChatGPT); the evaluation code comes from the AlpaGasus repository (README "Evaluation").
- [[lima]] — motivates the 1,000-sample pre-experience size (§4.3.1).
- [[alpaca]], [[wizardlm]] — the two selection pools (§3.1).
- [[deita]], [[less]] — other SFT data-selection methods; this paper does not compare against them.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2308.12032 (v5, 2024-04-06) and github.com/tianyi-lab/Cherry_LLM at commit 01fac8d (README, `cherry_seletion/data_by_IFD.py`).
- Corrections to the previous card version:
  - Title "Cherry LLM: From Quantity to Quality via Self-Guided IFD Selection (NAACL 2024)" → exact published title.
  - "IFD = conditional-PPL(response | instruction) / unconditional-PPL(response)" → ratio of mean token losses s(A|Q) / s(A) (Eq. 3–5); the released script also uses mean losses.
  - "High IFD means the instruction is actually helping predict the response" and "keep only the samples whose responses are genuinely helped by the instruction" → high IFD means the instruction helps less; the method keeps the highest IFD values after removing IFD > 1 (§1, §2.2).
  - "fine-tune target LM on a small random subset (~1K) for 1 epoch" → 1,000 samples chosen as 10 per cluster from 100 K-means clusters, 1 epoch (§2.1); random choice was an ablation (Table 2).
  - "Alpaca 10% cherry beats Alpaca full on MT-Bench, AlpacaEval, HuggingFace Open LLM Leaderboard. Same for WizardLM." → MT-Bench is not used; Table 1 reports 5% Alpaca above official Alpaca on the leaderboard average and AlpacaEval, and 10% WizardLM below reimplemented WizardLM on both (§4.1).
  - "keep top 5–15%" → the paper tests 5%, 10%, 15%, 20% (and 40% against official WizardLM) and recommends the top 10% (§4.1, App. D, App. G.2).
  - "Samples with IFD ≈ 1 or IFD > 1 signal ..." → only IFD > 1 is defined as misalignment (§2.2).
  - "Small warm-up required; skipping it hurts IFD quality" → with no pre-experienced model the results are lowest among tested counts but still beat Alpaca at 10%; the LLaMA2 runs use the base model directly (§4.3.1, §4.4, Limitation).
  - "Released IFD scores for popular datasets" → the repository releases cherry subsets (cherry_data_v1) and, from 2023/12, statistics for computing IFD on Alpaca and WizardLM with LLaMA2-7B/13B (README "News").
  - "[[alpagasus]] (needs GPT-4 quality ratings)" → this paper describes AlpaGasus as using ChatGPT (§6.3).
  - "Year: 2023/2024" → arXiv v1 2023-08; NAACL 2024.
- Removed as unsupported by the source: "IFD distribution histograms — most Alpaca samples cluster near 1"; "going below 5% loses coverage"; "Cross-dataset transfer table — IFD selected on Alpaca transfers to WizardLM"; "a different target family will prefer different samples"; "incorrect but hard responses can score highly"; "rare property"; "drastically cheaper at scale" for [[superfiltering]]; "Subsumed along the diversity axis by [[prismatic-synthesis]]".
- Not reported by the source: LR, batch, and epochs for the LLaMA2 runs (paper); loss masking and packing; run-to-run variance; any factual-correctness check of selected data.
