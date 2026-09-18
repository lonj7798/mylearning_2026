<!-- scope: Cohere For AI / Cohere (arXiv:2410.10801): on a pre-release Aya 23 8B checkpoint, compares training one model on safety/general data mixtures (0%, 15%, 100% safety) with merging separately trained checkpoints (Linear, SLERP, TIES, DARE-TIES) for multilingual safety and general win-rate; objective-based vs language-based merging, SFT vs DPO checkpoints, merge-then-DPO ordering
     deps: [[ties-merging]], [[task-arithmetic]], [[dpo]]
     see-also: [[dare-merging]], [[model-soups]], [[wise-ft]], [[warp]], [[command-a]], [[finetuning-compromises-safety]]
-->

# Mix Data or Merge Models? Optimizing for Diverse Multi-Task Learning
- **Core Insight:** Aggregated over six languages, SLERP-merging a general-only DPO model with a safety-only DPO model scores 78.0% Dolly-200 win-rate and −57.8% relative harm, against 71.0% and −54.7% for one DPO model trained on the 15% Safety Mix with the same data (Table 1).
- **Guideline:** When one model must serve a capability objective and a safety objective across several languages from a fixed data pool, train one model per objective and merge with SLERP instead of mixing, because SLERP was the only method that beat the 15% Safety Mix on both axes at both the SFT and DPO stage (Table 1); evaluate both axes after any merge, because TIES and Linear gained on one axis and lost on the other (Table 1).
- **Authors:** Aakanksha, Arash Ahmadian, Seraphina Goldfarb-Tarrant, Beyza Ermis, Marzieh Fadaee, Sara Hooker (Cohere For AI; Cohere)
- **Year:** 2024 (arXiv v1 2024-10-14; only version; no venue stated)
- **URL:** https://arxiv.org/abs/2410.10801
- **Source type:** paper
- **Relevant topics:** model merging vs data mixing, multi-task post-training, safety alignment, multilingual alignment, SLERP, TIES, DARE-TIES, DPO, alignment tax, cross-lingual interference

## Abstract
Preference training and safety measures often overfit to harms common in Western-centric data and do not extend well to other languages. The paper studies model merging in a multi-task setting that combines safety and general-purpose tasks across languages. Objective-based merging beats data mixing, with improvements of up to 8% in general performance and 10% in safety. Merging monolingually fine-tuned models gives a 4% increase in general performance and a 7% reduction in harm across all languages over the data-mixture approach with the same data (Abstract). The "up to" figures come from different methods and stages: +8.6 general is Linear on SFT checkpoints and +10.4 safety is TIES on DPO checkpoints (Table 1).

## Key Contributions
- A controlled Mix-vs-Merge comparison: same base checkpoint, same safety and general data, mixtures of 0%, 15% and 100% safety data versus merges of the 0% and 100% Safety models, run separately at SFT and DPO (§2).
- A comparison of four merge algorithms on two axes, showing that the best algorithm per axis differs and that only SLERP improves both axes in both stages (§3.1, Table 1).
- Language-based merging: six monolingual 15% Safety Mix models merged with TIES beat the multilingual mixture model (§3.4, Fig. 5).
- Stage and ordering results: merging DPO checkpoints vs SFT checkpoints (§3.2), and SFT → merge → DPO vs SFT → DPO → merge (§3.6, Table 2).

## Key Figures/Tables to Study
- Table 1: harm and win-rate for the three mixtures and four merges, SFT and DPO, with deltas to the 15% Safety Mix.
- Fig. 5: monolingual merging, [EN,FR,SP] vs [All] vs the multilingual mixture.
- Fig. 6: Linear-merge weight on the 100% Safety model vs both metrics.
- Table 2: position of the merge relative to DPO. Tables 3-6 (Appendix): per-language breakdown.

## Technical Details
**Setup.** The base model and evaluation baseline is "a previous checkpoint of the Aya 23 8B model", which was not optimized for safety (§2.4). Safety data are adversarial prompts and safe completions generated synthetically from human-annotated Aya Red-teaming seed prompts (§2.2). General data are 10,000 English prompts sampled from UltraFeedback Binarized and translated into the target languages (§2.2). The 15% Safety Mix combines safety and general data "in a 1:5 ratio" and is the default baseline (§2.2); read as safety:general, 1:5 is a 16.7% share (derived; the paper does not reconcile the label). Languages: English, Hindi, French, Spanish, Arabic, Russian (Table 1 caption).
**Merge methods (§2.1).** Linear: θ_merged = Σ_i α_i·θ_i with Σ_i α_i = 1 (Eq. 1). SLERP: θ(t) = sin((1−t)Ω)/sin Ω · θ_1 + sin(tΩ)/sin Ω · θ_2, where Ω is the angle between the normalized weight vectors and t ∈ [0, 1] (Eq. 2). TIES: consensus sign s = sign(Σ_i sign(θ_i)) and θ_merged = s·(1/N)·Σ_i |θ_i| after trimming small changes (Eq. 3-4; written on θ_i here, whereas [[ties-merging]] applies the steps to task vectors and elects the sign by summed magnitude). DARE-TIES applies dropout to delta parameters before TIES. All methods except Linear use "gradient weighting", a blend-ratio list such as [0, 0.5, 1] interpolated across tensors. Weights are searched over {0, 0.3, 0.5, 0.7, 1} with Arcee's mergekit.
**Evaluation (§2.4).** Safety: Aya Red-teaming English prompts translated with NLLB-3.3B into the other five languages; metric is the relative % change in harmful generations vs the base model, aggregated over languages. General: Multilingual Dolly-200 (200 Dolly-15k prompts, translated); metric is win-rate vs the base model. GPT-4 is the judge for both (footnote 1 links the gpt-4-turbo documentation).
**Main results (Table 1; harm ↓ / win-rate ↑; brackets = difference from 15% Safety Mix).**

| Method | SFT harm | SFT win | DPO harm | DPO win |
|---|---|---|---|---|
| 0% / 15% / 100% Safety Mix | −41.4 / −56.6 / −64.4 | 70.0 / 67.4 / 64.8 | −39.2 / −54.69 / −68.2 | 70.7 / 71.0 / 75.0 |
| Linear | −49.1 (−7.5) | 76.0 (+8.6) | −48.6 (−6.1) | 75.0 (+4.0) |
| SLERP | −58.2 (+1.2) | 72.6 (+5.2) | −57.8 (+3.1) | 78.0 (+7.0) |
| TIES | −45.2 (−11.4) | 74.9 (+7.5) | −65.1 (+10.4) | 63.6 (−7.4) |
| DARE-TIES | −56.1 (−0.5) | 70.0 (+2.6) | −55.9 (+1.2) | 78.5 (+7.5) |

The SFT SLERP harm delta is printed as +1.2, but −58.2 vs −56.6 is 1.6, and Table 2 prints +1.6 for the same model (derived).
**SFT vs DPO merging (§3.2).** DPO merges average +2.8 general and +2.2 safety across the four methods; SFT merges average about +6 general and a 4.6 increase in harmful generations relative to the 15% Safety Mix. The prose calls the DPO averages gains "over the base model", but they equal the means of the Table 1 deltas vs the 15% Safety Mix, e.g. (4.0 + 7.0 − 7.4 + 7.5)/4 = 2.8 (derived).
**Per language (§3.3).** DPO: Russian gains most in safety (15%, TIES), Spanish most in general (about 6%, SLERP); English gains least, with a 24.87% safety decline (Linear) and 14.5% general decline (TIES). SFT: Hindi has the largest harm reduction (12.14%, SLERP), Spanish +10% general (Linear, TIES); Spanish TIES has about 16% more harmful generations. Every language keeps a win-rate above 50% vs the base. The prose safety figures do not equal the differences in Tables 3 and 5 (for example Hindi SFT SLERP: −47.3 → −65.1, 17.8 points).
**Language-based merging (§3.4, Fig. 5).** Six models, each fine-tuned on one language with the 15% Safety Mix, are merged with TIES, chosen for its "permutation-invariant nature". Harm / win-rate: multilingual mix [All] −56.6 / 67.4; merge [EN,FR,SP] −65.3 / 77.0; merge [All] −63.2 / 71.2. Merging all six beats the mix by 6.6 harm points and 3.8 win-rate points; the three-language merge is about 2 and 6 points better than the six-language merge, which the authors call cross-lingual interference. Whether [EN,FR,SP] is evaluated on three or six languages is not stated.
**Merge weight (§3.5, Fig. 6).** In Linear merging, raising the 100% Safety model's weight (0.1 to 0.5) lowers harm and lowers win-rate; harm beats the 15% Safety Mix from weight 0.3, and in the general-performance paragraph the authors state that merging outperforms the mix at all weightings.
**Merge position (§3.6, Table 2; SLERP).** SFT → merge: −58.2 / 72.6. SFT → DPO → merge: −57.8 / 78.0. SFT → merge → DPO: −61.2 / 74.0.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Mix/Merge runs (pre-release Aya 23 8B checkpoint) | 8B | SFT | initialization | "a previous checkpoint of the Aya 23 8B model" | arXiv:2410.10801v1 §2.4 | verified 2026-09-14 | n/a |
| Mix/Merge runs | 8B | SFT, preference | general prompts | 10,000 English UltraFeedback Binarized prompts, translated to 6 languages | §2.2 | verified 2026-09-14 | no ablation reported |
| Mix/Merge runs | 8B | SFT, preference | safety data | synthetic adversarial prompts + safe completions from Aya Red-teaming seeds; count not reported | §2.2 | verified 2026-09-14 (count not reported) | no ablation reported |
| Mix runs | 8B | SFT, preference | safety share (mixture) | 0% / 15% ("1:5 ratio") / 100% | §2.2 | verified 2026-09-14 | Table 1 compares all three |
| Mix/Merge runs | 8B | SFT | LR, epochs, batch, sequence length | not reported | checked §2-§3, Tables 1-6 | not reported | n/a |
| Mix/Merge runs | 8B | preference | DPO pair construction, β, LR, epochs | not reported | checked §2-§3, Tables 1-6 | not reported | n/a |
| Objective merge | 8B | merge | method; tool; weight grid | Linear, SLERP, TIES, DARE-TIES; mergekit; {0, 0.3, 0.5, 0.7, 1}, gradient weighting except Linear | §2.1 | verified 2026-09-14 | Table 1; selection metric for "best-performing checkpoints" not reported |
| Language merge | 8B | merge | method; inputs | TIES over 3 or 6 monolingual 15% Safety Mix SFT models | §3.4 | verified 2026-09-14 | Fig. 5 |
| Pipeline order | 8B | merge | position of merge | SFT → merge (SLERP) → DPO gives the best harm (−61.2) | §3.6, Table 2 | verified 2026-09-14 | Table 2, one run each |
| Evaluation | n/a | eval-gate | judge; translation | GPT-4 (gpt-4-turbo link); NLLB-3.3B for red-team prompts | §2.4 | verified 2026-09-14 | n/a |

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality.** General ability is measured by one open-ended benchmark (Dolly-200 win-rate against the base, GPT-4 judge); no held-out capability suites or seeds are reported (§2.4). Mixing more safety data lowers SFT win-rate (70.0 → 64.8 from 0% to 100% safety), but at DPO the 100% Safety Mix has the highest win-rate of the three mixtures (75.0) (Table 1).
- **Narrowing.** Adding models to a merge can lower both axes: the six-language merge is worse than the three-language merge (§3.4).
- **Negative feedback.** Safety SFT trains safe completions to adversarial prompts with cross-entropy (negative as content, style guide §6.1 type 2; course classification). How rejected responses for DPO were built is not reported.

## Connections
- [[ties-merging]]: TIES and DARE-TIES are two of the four merge methods; the TIES paper defines the trim, sign-election and disjoint-mean steps on task vectors.
- [[dare-merging]]: Yu et al. 2024, the source of DARE-TIES (§2.1).
- [[model-soups]], [[task-arithmetic]]: cited for linear weight averaging and for merge hyperparameter sensitivity (§2.1, §2.3).
- [[wise-ft]]: interpolates a pre-trained and a fine-tuned model; this paper interpolates two fine-tuned models.
- [[warp]]: also uses SLERP, on task vectors of RL policies.
- [[dpo]], [[ultrafeedback]]: the preference stage and the source of the general prompts.
- [[command-a]]: Cohere's later report, whose abstract names model merging as part of its training approach.
- [[finetuning-compromises-safety]]: fine-tuning degrading safety, the trade-off this paper tries to balance.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2410.10801 (v1, 2024-10-14; full PDF including Appendix Tables 3-6).
- Audit claims not found in the source: none. Nuance added: the "up to 8% / 10%" figures come from different merge methods and stages (Table 1).
- Not reported by the source: SFT and DPO hyperparameters, safety-data size, DPO pair construction, checkpoint-selection metric, seeds or confidence intervals, compute.
