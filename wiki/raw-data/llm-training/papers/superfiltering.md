<!-- scope: IFD rankings computed by an untrained GPT-2 transfer to LLaMA2 SFT selection, about 20x cheaper (ACL 2024 main)
     deps: [[cherry-llm]], [[ifd]]
     see-also: [[deita]], [[less]], [[alpagasus]]
-->

# Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning
- **Core Insight:** IFD scores computed by GPT-2 (124M) rank instruction-tuning samples consistently with those computed by LLaMA2-7B — Spearman's ρ of 0.679 on Alpaca, 0.788 on Alpaca-GPT4, and 0.802 on WizardLM 70k — so the small model can select the subset used to fine-tune the large one, cutting Alpaca filtering time from 161 minutes to 8 (§3.1-3.2, Table 1; §5.2, Table 4).
- **Guideline:** When the filtering budget matters more than the last point of accuracy, score an instruction pool with an untrained GPT-2 IFD filter and keep the top 5-15%; when accuracy matters more, score with the model being trained, because at every budget the LLaMA2-7B IFD filter beat the GPT-2 filter in pairwise winning score (0.853 at 5%, 0.761 at 10%, 0.927 at 15%, values below 1.0 favouring the LLaMA2-7B filter) (§5.2, Table 4).
- **Authors:** Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, Jianzong Wang, et al. (University of Maryland; Ping An Technology (Shenzhen))
- **Year:** 2024 (arXiv v1 2024-02; v2 2024-06; ACL 2024 main)
- **URL:** https://arxiv.org/abs/2402.00530
- **Source type:** paper
- **Relevant topics:** weak-to-strong filtering, IFD, SFT data selection, filtering cost

## Abstract
Data filtering improves instruction tuning but adds the cost of running a large model, or an API model,
over the whole pool. The paper asks whether a smaller and weaker model can select the data used to
fine-tune a larger one. It measures the agreement between perplexity and IFD rankings produced by GPT-2
(124M), GPT-2-large (774M), GPT-2-XL (1.5B), GPT-NEO (1.3B), and LLaMA2-7B on Alpaca, Alpaca-GPT4, and
WizardLM 70k, and finds the rankings highly consistent even though the absolute perplexity scales differ.
Superfiltering therefore uses GPT-2 directly, with no training of the filter model and no held-out set,
to select the top 5%, 10%, or 15% of a pool by IFD, and fine-tunes LLaMA2-7B or LLaMA2-13B on that subset.

## Key Contributions
- Measurement of weak-to-strong consistency for both perplexity and IFD rankings, with Spearman's ρ and
  with the overlap of the selected subsets at 5%, 10%, and 15% budgets (§3.1-3.2, Table 1).
- Superfiltering: selecting instruction data with an untrained GPT-2 IFD filter (§3.3).
- Evaluation of the resulting models by GPT-4 pairwise comparison, the Hugging Face Open LLM Leaderboard,
  AlpacaEval, and a 3-annotator human study on 100 WizardLM test instructions (§4.3, §5.1).
- A cost comparison against ChatGPT scoring, reward-model scoring, and same-model IFD (§5.2, Table 4).
- An ablation over selection strategies (random, k-means diversity, perplexity) and filter models (Table 3).

## Key Figures/Tables to Study
- Table 1: Spearman's ρ and subset-overlap ratios between LLaMA2-7B and four smaller filter models.
- Figure 3: perplexity and IFD distributions for five models on three datasets.
- Table 2: pairwise winning score, Open LLM Leaderboard, and AlpacaEval at 5/10/15/100% data.
- Table 4: winning score and filtering wall-clock time against three other selection methods.

## Technical Details
- IFD (§2.1, Eq. 2): `IFD(y_i | x_i) = PPL(y_i | x_i) / PPL(y_i)`, where `x_i` is the instruction, `y_i`
  the response, `PPL(y_i | x_i)` the perplexity of the response conditioned on the instruction, and
  `PPL(y_i)` its perplexity without the instruction. A higher score means the instruction helps less.
  Samples with the highest IFD below 1 are selected (§3.3).
- Spearman's ρ against LLaMA2-7B, perplexity / IFD (Table 1): Alpaca — GPT-2 0.726 / 0.679, GPT-2-large
  0.790 / 0.682, GPT-2-XL 0.802 / 0.693, GPT-NEO 0.846 / 0.802. Alpaca-GPT4 — GPT-2 0.730 / 0.788,
  GPT-NEO 0.842 / 0.876. WizardLM 70k — GPT-2 0.763 / 0.802, GPT-NEO 0.857 / 0.893.
- Subset-overlap with the LLaMA2-7B selection, GPT-2 filter (Table 1): Alpaca 0.28 / 0.41 / 0.49 at
  5 / 10 / 15%; Alpaca-GPT4 0.24 / 0.40 / 0.51; WizardLM 70k 0.42 / 0.54 / 0.61. The rank correlation is
  high while the selected sets overlap only partially.
- Datasets: Alpaca, 52,000 samples generated with text-davinci-003 under the self-instruct paradigm;
  Alpaca-GPT4, the same instructions answered by GPT-4; WizardLM 70k (§4.1). 5% of Alpaca is 2,600
  samples (Table 2).
- Results on Alpaca / LLaMA2-7B (Table 2): full data gives pairwise score 1.000, Open LLM Leaderboard
  average 55.25, AlpacaEval win rate 27.75; the 5% subset (2,600 samples) gives 1.133, 55.67, and 33.04;
  10% gives 1.101 and 56.97; 15% gives 1.193 and 56.61. MMLU falls from 47.02 to 45.21 at 5%.
- Results on Alpaca-GPT4 / LLaMA2-13B (Table 2): full data 60.81 average and 77.86 AlpacaEval; 5% subset
  1.041, 63.29, 78.15. The pairwise winning score is `(Num(Win) − Num(Lose))/Num(All) + 1`, judged by
  GPT-4 on the WizardLM test set.
- Human study (§5.1): Alpaca 5% against Alpaca 100% on LLaMA2-7B won 50, tied 18, lost 32 of 100
  instructions; Alpaca-GPT4 5% against 100% won 49, tied 5, lost 46.
- Filtering time on Alpaca (Table 4): Superfiltering 8 minutes; ChatGPT scoring 120; IFD with LLaMA2-7B
  161; reward-model scoring 1400. The paper states Superfiltering is 20× faster than same-model IFD
  scoring and that reward-model scoring is 175× slower than Superfiltering (§5.2).
- Winning score against the alternatives (Table 4, values above 1.0 favour Superfiltering): against
  ChatGPT score 1.028 / 1.174 / 1.170; against reward score 1.280 / 1.096 / 1.147; against LLaMA2-7B IFD
  score 0.853 / 0.761 / 0.927, at 5 / 10 / 15%.
- Strategy ablation on Alpaca / LLaMA2-7B (Table 3): random 0.936 / 0.968 / 0.977, k-means diversity
  0.927 / 0.977 / 0.982, GPT-2 perplexity 0.261 / 0.569 / 0.610, Superfiltering (GPT-2 IFD)
  1.133 / 1.101 / 1.193, LLaMA2-7B IFD 1.303 / 1.330 / 1.294. Selecting by raw perplexity is far worse
  than selecting by IFD with the same model.
- No training of the filter model and no held-out set are required, which the authors contrast with
  proxy methods that first fine-tune the weak model (§6.1). Filtering runs on a GPU with as little as
  6 GB of memory because only GPT-2 is used (§5.2).

## Recipe ledger
Moved to [[superfiltering-recipe]] for length.

## Findings relevant to generality
- The selected-subset models improve the Open LLM Leaderboard average in all eight settings of Table 2,
  but MMLU is not uniformly improved: Alpaca / LLaMA2-7B falls from 47.02 to 45.21 (5%), 47.16 (10%),
  46.73 (15%), and Alpaca-GPT4 / LLaMA2-7B falls from 47.89 to 46.80, 45.67, and 46.15.
- Transfer is measured across two student sizes (LLaMA2-7B and 13B) and three source datasets. No other
  model family is evaluated, and no code, math, or multilingual evaluation is reported.
- The paper states that the filter model needs no fine-tuning on the target pool, which it presents as
  reducing the risk of out-of-distribution issues (§6.1, Interpretation).

## Connections
- [[ifd]], [[cherry-llm]] — the IFD score and the original selection method this work reuses; Cherry LLM
  trains a "pre-experienced" model first, which Superfiltering removes.
- [[cherry-llm-recipe]] — the released selection script and its settings.
- [[deita]], [[less]], [[alpagasus]] — other SFT selection signals; Table 4 compares only against ChatGPT
  scoring, reward-model scoring, and same-model IFD.
- [[prismatic-synthesis]] — later work on coverage-based selection; not compared here.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2402.00530 (arXiv v2, 7 Jun 2024, full PDF text;
  abstract page comment "ACL2024 main, Camera-ready")
- Corrections to the previous card version:
  - "a tiny model (GPT-2 125M)" → GPT-2 is 124M parameters (§3.1).
  - "keep the top ~15%" as the recommendation → the paper reports 5%, 10%, and 15% budgets throughout and
    uses 5% for the human study; no single budget is recommended.
  - "Warm proxy on ~1K random subset of target pool" → Superfiltering does no training of the filter model
    and uses no held-out set (§3.3, §6.1). The warm-up step belongs to Cherry LLM, not to this paper.
  - "the same (or slightly better) downstream performance as filtering with the target model itself" →
    filtering with LLaMA2-7B itself is better at every budget (Table 4: 0.853 / 0.761 / 0.927 against
    Superfiltering; Table 3: 1.303 / 1.330 / 1.294 against 1.133 / 1.101 / 1.193).
  - "≥ full-data baseline on MT-Bench" → MT-Bench is not used; the evaluations are GPT-4 pairwise on the
    WizardLM test set, the Hugging Face Open LLM Leaderboard, AlpacaEval, and a human study.
  - "Consistent across multiple target model families (Llama-2, Mistral)" → only LLaMA2-7B and LLaMA2-13B
    are fine-tuned (Table 2).
  - "Still requires warmup — raw-GPT-2 IFD (no warmup) is noisier" → not reported; the paper's claim is
    the opposite, that no warm-up is needed (§6.1).
- Removed as unsupported by the source: "the correlation is high" without numbers (replaced by Table 1
  values); "Released code + scored datasets" for scored datasets (the paper links a code repository only);
  "Filtered-set scaling curve" as a figure; "Weak-proxy family mismatch: transferability depends on the
  proxy having a plausibly similar tokenizer + capability profile"; "using a domain-specific proxy can
  skew selection"; "a key property later exploited in 2025 weak-to-strong and prismatic-synthesis-style
  pipelines".
- Not reported by the source: absolute IFD score distributions per budget; contamination checks; results
  on non-LLaMA2 students; the hardware used for fine-tuning; variance across seeds.
