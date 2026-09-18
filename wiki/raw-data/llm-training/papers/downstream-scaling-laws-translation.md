<!-- scope: scaling of downstream translation quality (BLEU, ROUGE, COMET) and downstream cross-entropy with pretraining data size for T5 models; pretraining-task alignment; cases where cross-entropy improves while BLEU falls
     deps: [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]]
     see-also: [[scaling-laws-for-transfer]], [[scaling-laws-unreliable-downstream]], [[predicting-downstream-elusive]], [[atlas-multilingual-scaling-laws]]
-->

# Scaling Laws for Downstream Task Performance in Machine Translation
- **Core Insight:** For T5-3B pretrained on MC4 and fine-tuned on WMT translation, BLEU follows a log-law `(log(A·D_p^α))^β` in pretraining data size when the pretraining languages cover the task (A = 1 mixtures, BLEU fit error ≤ 0.061), but for en-fr with English-only, German-only, or Romanian-only pretraining, BLEU stops improving or decreases with more pretraining while downstream cross-entropy keeps decreasing (§5, Fig. 2–4).
- **Guideline:** When choosing pretraining data for a downstream task, evaluate fine-tuned checkpoints on the task metric (BLEU, ROUGE, COMET) rather than on cross-entropy, and treat a failure of the log-law fit as a sign of misalignment, because in the en-fr experiments cross-entropy improved monotonically for data that made BLEU worse (§3.3–3.5, Fig. 1, Fig. 5); when the fine-tuning set is large (the 3B en-de, 21B en-fr, and 312M en-ro sets), pretraining gave little to no BLEU gain over a randomly initialized model (§5, Fig. 2).
- **Authors:** Berivan Isik, Natalia Ponomareva, Hussein Hazimeh, Dimitris Paparas, Sergei Vassilvitskii, Sanmi Koyejo (Google Research; Hazimeh listed at OpenAI and Koyejo at Stanford; all work done at Google)
- **Year:** 2024 (arXiv v1 2024-02; ICLR 2025). arXiv title: "Scaling Laws for Downstream Task Performance of Large Language Models"; ICLR 2025 title as above (arXiv Comments field).
- **URL:** https://arxiv.org/abs/2402.04177
- **Source type:** paper
- **Relevant topics:** downstream scaling laws, transfer learning, pretraining data selection, distribution alignment, loss-versus-task-metric divergence, machine translation, multilingual pretraining

## Abstract
Most scaling-law work predicts upstream (pretraining) loss. This paper studies transfer learning: LLMs are pretrained on unsupervised multilingual data and then fine-tuned for machine translation. The authors measure how the choice and size of pretraining data affect downstream cross-entropy and translation quality (BLEU, COMET). Fine-tuning data size and the distribution alignment between pretraining and downstream data both change the scaling behavior. With enough alignment, cross-entropy and translation scores improve monotonically with more pretraining data, and translation scores are predictable with a log-law. With moderate misalignment, translation scores can fluctuate or get worse with more pretraining while downstream cross-entropy still improves. The authors turn these results into guidance for choosing pretraining data.

## Key Contributions
- A log-law for translation scores in pretraining data size, `f(D_p) = (log(A·D_p^α))^β` (§3.1 Eq. 1), and a power law for downstream cross-entropy, `L(D_p) = E + A/D_p^α` (§3.3 Eq. 3).
- A translation alignment score based on the language shares of the pretraining mixture (§3.2 Eq. 2).
- Controlled cases in which BLEU and cross-entropy disagree about the value of more pretraining data (§3.3, §5 Fig. 3–5).
- A procedure for valuing a pretraining dataset from three or more fine-tuned checkpoints (§3.5).
- Checks that the log-law also fits SuperGLUE tasks (App. B) and that the en-fr break repeats with T5-770M (App. C.2).

## Key Figures/Tables to Study
- Fig. 1: BLEU, ROUGE, COMET, and cross-entropy for en-fr with A = 1 versus A = 0.7 pretraining.
- Fig. 2 (A = 1) and Fig. 3 (A = 0.7): BLEU and cross-entropy for three fine-tuning sizes per task, with non-pretrained baselines.
- Fig. 4: English-only, target-language-only, balanced, unrelated-language (fr-only or ro-only for en-de; de-only or ro-only for en-fr), and 30/70 and 70/30 en/fr pretraining sets.
- Fig. 5: BLEU versus downstream cross-entropy; consistent for en-de, not for en-fr.
- App. C.3 Tables 3–6: fitted coefficients and prediction errors.

## Technical Details
- **Log-law.** `f(D_p) = (log(A·D_p^α))^β` (Eq. 1). `f` is the translation score, `D_p` the pretraining data size (tokens seen, no repetition), `A, α, β` fitted coefficients that depend on alignment and fine-tuning size (§3.1).
- **Cross-entropy law.** `L(D_p) = E + A/D_p^α` (Eq. 3), with fitted `E, A, α` (§3.3).
- **Alignment score.** `A(D, T(L_src, L_dest)) = P_src·P_dest + 0.7·P_src + 0.8·P_dest` (Eq. 2), where `P_src` and `P_dest` are the shares of the source and target languages in the pretraining mixture `D`. Examples: 50% en + 50% fr for en-fr gives 0.25 + 0.35 + 0.4 = 1; 100% en gives 0.7 (§3.2). The authors note other definitions are possible (§3.2).
- **Models.** T5-3B encoder-decoder: 24 encoder and 24 decoder layers, embedding 1024, 32 heads of dimension 128, MLP 16,384; T5-770M: 16 heads of dimension 64, MLP 2,816 (§4; App. A.1 Tables 1–2). SentencePiece vocabulary of 250,112 covering all MC4 languages (§4).
- **Data.** Pretraining on en, de, fr, ro portions of MC4, alone or as 50/50 per-batch mixtures, plus 30/70 and 70/30 en/fr (§4). Fine-tuning on WMT-17 en-de (3B tokens), WMT-15 en-fr (21B), WMT-16 en-ro (312M), with random subsets: 6M/31M/3B, 42M/210M/21B, 625K/3M/312M tokens (§4; Fig. 2 caption).
- **Fitting.** Huber loss on log values with L-BFGS from a grid of initializations; coefficients are fit on the four points with the least pretraining data and the remaining points are held out (§4; App. A.2 Eq. 6–7).
- **Fit quality.** A = 1 mixtures: prediction error ≤ 0.061 for BLEU and ≤ 5.95e-12 for cross-entropy (§5). English-only pretraining on en-de and en-ro: BLEU error ≤ 0.025 (§5).
- **BLEU decomposition.** Brevity penalty was 1 in all experiments, so the non-monotonic BLEU comes from n-gram precision (§5 Remark 2).
- **Compute.** Pretraining T5-3B for 1M steps took 15–20 hours and fine-tuning 5–7 hours on an 8x8 TPU (App. A).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| T5-3B (Raffel et al. config) | 3B | pretrain | batch; sequence length; steps | 256; 512; 1,000,000 (510,000 for ro-MC4 to avoid repetition) | arXiv:2402.04177v3 §4 "Hyperparameters" | verified 2026-09-14 | no ablation reported |
| T5-3B | 3B | pretrain | positions per full run | 256 × 512 × 1,000,000 = 1.28e11 (packing and padding not reported) | §4 | derived | — |
| T5-3B | 3B | pretrain | LR schedule; base LR | inverse square root `1/sqrt(max(n, k))`, k = 10^4; base LR chosen from {0.05, 0.1, 0.5, 1.0, 2.0, 5.0} | §4 | verified 2026-09-14 | grid search per pretrained model, selected by upstream cross-entropy |
| T5-3B | 3B | pretrain / SFT | optimizer | AdaFactor in both stages | §4 | verified 2026-09-14 | no ablation reported |
| T5-3B | 3B | SFT (translation fine-tuning) | method; batch; sequence length; steps | full-weight; 128; 512; 300 | §4 | verified 2026-09-14 | follows Raffel et al. (2020) |
| T5-3B | 3B | SFT | learning rate | constant, best of {0.001, 0.005, 0.01, 0.05, 0.1} | §4 | verified 2026-09-14 | per-run selection |
| T5-3B on SuperGLUE | 3B | SFT | batch; seq len; steps; LR grid | 128; 512; 300; {0.001, 0.005, 0.01, 0.05, 0.1, 0.5} | App. B | verified 2026-09-14 | per-run selection |
| scaling-law fit | — | eval-gate | Huber δ | 0.1 for BLEU/COMET; cross-entropy 1e-3 (§4, App. A.2) versus 10^-5 (Tables 4 and 6 captions) | §4; App. A.2; App. C.3 | conflict | not explained in the paper |

## Findings relevant to generality
- **Loss is not a sufficient proxy (Result, single study).** For en-fr, BLEU breaks from the log-law once pretraining data passes a threshold for English-only pretraining at D_f = 42M and 210M, while cross-entropy follows Eq. 3 (§5 Fig. 3). Looking only at cross-entropy would suggest adding both aligned and less aligned data, which the authors call a poor decision (§3.3).
- **Which misalignment matters.** For en-fr, BLEU scaling breaks with 100% en (A = 0.7), 100% de (A = 0), and 100% ro (A = 0), but not with 100% fr (A = 0.8). For en-de, laws fit for all mixtures, including fr-only and ro-only (A = 0) (§5 Fig. 4).
- **Target-language data.** English-only pretraining gives worse BLEU and cross-entropy than balanced mixtures for en-de and en-fr, while de-only and fr-only pretraining are almost as good as balanced; the authors relate this to the model generating German or French (§5 Remark 1, Interpretation).
- **Metric relation.** BLEU and cross-entropy correlate consistently for en-de; for en-fr the relation is often arbitrary, and BLEU sometimes rises while cross-entropy also rises (§5 Fig. 5). This contradicts the exponential BLEU–cross-entropy relation of Gordon et al. (2021) for that task (§5).
- **Value of pretraining shrinks with fine-tuning size.** Gains from more pretraining data are larger for smaller fine-tuning sets; with the largest sets, BLEU is nearly constant and close to the non-pretrained baseline (§5 Fig. 2).
- **Beyond translation.** The log-law fits BoolQ, MultiRC, COPA, ReCoRD, and RTE after English-only pretraining (App. B Fig. 6). COMET shows the same patterns as BLEU (App. C.1 Fig. 7); T5-770M shows the same en-fr break (App. C.2 Fig. 9).
- **Scope limits.** Only encoder-decoder T5-770M and T5-3B are trained, so there is no model-size law, no decoder-only model, and no instruction or preference tuning (§4; App. C.2). The authors name a more linguistic definition of alignment as future work (§5 Remark 1; §6 Limitations).

## Connections
- [[scaling-laws-for-transfer]] — Hernandez et al. (2021), cross-entropy scaling in fine-tuning data size; the closest prior work (§2).
- [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]] — upstream power laws that Eq. 3 follows (§1, §3.3).
- [[scaling-laws-unreliable-downstream]], [[predicting-downstream-elusive]] — later analyses of when downstream metrics do not follow scaling trends.
- [[overtraining-downstream-scaling]] — Gadre et al., downstream prediction from loss in decoder-only LMs.
- [[atlas-multilingual-scaling-laws]], [[multilinguality-curse-250-languages]] — scaling with multilingual pretraining mixtures.
- [[beyond-chinchilla-inference-scaling]] — reports tight loss–accuracy correlation on a broad English Gauntlet, a contrast to the en-fr case here.
- [[data-constrained-scaling]] — cited for avoiding repeated sequences (§3.5 footnote 4).
- [[c4]] — T5 paper (Raffel et al.), which this paper cites for the MC4 dataset, the T5 configurations, and the fine-tuning settings (§4; App. A.1).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2402.04177 (arXiv v3, 2026-01-29; full PDF text including App. A–C). v1 date (2024-02-06) and the ICLR 2025 title in the Comments field checked on the arXiv abs page.
- Audit claims not found in the source: none. The audit's title note is confirmed: the arXiv listing uses "...of Large Language Models" and the arXiv Comments field gives the ICLR 2025 title "...in Machine Translation".
- Not reported by the source: number of pretraining checkpoints per curve, seeds, TPU generation, and the upstream loss values.
