<!-- scope: Razin et al. (ICLR 2025): likelihood displacement in DPO-style losses; why the preferred response loses probability, where the mass goes (unembedding and hidden-embedding geometry), the CHES score, and DPO refusal training that lowers refusal rates
     deps: [[dpo]]
     see-also: [[ipo]], [[simpo]], [[dpo-positive]], [[learning-dynamics-llm-finetuning]], [[lazy-likelihood-displacement-grpo]], [[negative-sample-reinforcement]], [[nft-negative-aware-finetuning]]
-->

# Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization
- **Core Insight:** DPO on on-policy refusal pairs from SORRY-Bench prompts lowered the training-set refusal rate of Llama-3-8B-Instruct from 74.4% to 33.4% and of Gemma-2B-IT from 80.5% to 54.8%, because probability mass moved away from the preferred refusals (§6.2); training on the 5% of pairs with the lowest length-normalized CHES score prevented the drop (§6.3, Figure 3).
- **Guideline:** When preference pairs can have preferred and dispreferred responses of the same kind (for example two refusals), remove pairs with a high length-normalized CHES score before DPO, because in the paper's refusal experiment this raised refusal rates more than adding an SFT term to the loss (§6.3, Figure 3). Whether CHES filtering helps general instruction following was not tested (§8.1).
- **Authors:** Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, Boris Hanin (Princeton)
- **Year:** 2024 (arXiv v1 2024-10; ICLR 2025)
- **URL:** https://arxiv.org/abs/2410.08847 (code: https://github.com/princeton-nlp/unintentional-unalignment)
- **Source type:** paper
- **Relevant topics:** DPO, IPO, likelihood displacement, negative gradient, embedding geometry, preference-data curation, safety refusal training, unalignment

## Abstract
DPO and its variants train a model to increase the log-probability margin between a preferred response y+ and a dispreferred response y−. Prior work observed that the likelihood of y+ often decreases during this training; the paper names this likelihood displacement. The paper shows that displacement can move probability mass to responses with the opposite meaning: training to prefer No over Never can increase the probability of Yes. In refusal training, displacement moves mass from preferred refusals to harmful responses (Llama-3-8B-Instruct refusal rate 74.4% → 33.4%). A gradient-flow analysis shows that displacement is driven by preferences that induce similar embeddings, measured by the centered hidden embedding similarity (CHES) score. CHES identifies the training samples that contribute most to displacement, and removing those samples mitigated unintentional unalignment in the paper's experiments (Abstract).

## Key Contributions
- Definition: likelihood displacement occurs when the loss decreases but the mean ln π(y+|x) over the dataset decreases; it is benign if the responses gaining mass are as preferable as y+, and catastrophic if they are substantially less preferable (§2.2, Definition 1).
- Demonstration with one prompt and single-token responses: the probability of y+ fell by at least 0.21 and up to 0.96 in all runs, and mass moved to tokens of opposite meaning (§3, Table 1).
- Theory (unconstrained features model, gradient flow) identifying the token-unembedding and hidden-embedding terms that drive displacement and decide which responses gain probability (§4, Theorems 1–3; App D).
- The CHES score, whose ranking matched the degree of displacement on UltraFeedback and AlpacaFarm, while edit distance and last-hidden-state similarity did not (§5, Figure 2).
- Unintentional unalignment in DPO refusal training and its mitigation by CHES filtering, compared with an SFT term and with "gold" pairs (§6, Figure 3, Table 16).
- Proof that a linear model with fixed hidden embeddings cannot show displacement, so the squeezing effect of Ren and Sutherland (2024) does not fully explain it (App C; App F, Proposition 3).

## Key Figures/Tables to Study
- Table 1 (single-token pairs: probability drop of y+ and the tokens that gained); Table 13 (norm of the part of W_{y+} − W_{y−} orthogonal to W_{y+}).
- Figure 2 (preferred log-probability change by CHES, edit-distance, and last-hidden-inner-product percentile).
- Figure 3 and Table 16 (refusal rates and preferred log-probability change for DPO, DPO+SFT, gold, filtered); Figure 4 (CHES ranking by pair type); Table 18 (prompts that became compliant after DPO).

## Technical Details
**Loss family (§2.1, App B).** L(θ) = E_{(x,y+,y−)~D}[ℓ(ln πθ(y+|x) − ln πθ(y−|x))], with ℓ convex, differentiable, and decreasing near initialization. This covers DPO, IPO, SLiC, REBEL, and GPO; losses with an SFT term or unequal weights (CPO, RPO of Liu et al. 2024, BoNBoN, SimPO, DPOP) are treated in App E.

**Single sample, single-token responses (§4.2.1; Theorem 4, App D.1).** d/dt ln π(y+|x) = −ℓ′·[m − (1 − π(y+|x) + π(y−|x))·⟨W_{y+}, W_{y−}⟩ − Σ_{z∉{y+,y−}} π(z|x)·⟨W_z, W_{y+} − W_{y−}⟩].
- W_z: unembedding row of token z; −ℓ′ > 0; m: a non-negative term (Theorem 4).
- Two terms can make the derivative negative: alignment ⟨W_{y+}, W_{y−}⟩ of the two responses, and probability-weighted alignment of other tokens with W_{y+} − W_{y−}.
- Where the mass goes (Theorem 5): d/dt ln π(z|x) = −ℓ′·[⟨W_z, W_{y+} − W_{y−}⟩ + c], where c does not depend on z. W_{y+} − W_{y−} has a component orthogonal to W_{y+}, introduced by W_{y−}; this component was larger than the projection onto W_{y+} for every model and pair in Table 1 (§4.2.1; App H.1, Table 13). The authors use this to explain why tokens unrelated or opposite in meaning to y+ can gain (Interpretation, §4.2.1).

**Multi-token responses (§4.2.2; Theorem 6).** The derivative is more negative the larger Σ_k Σ_k′ α−_{k,k′}⟨h_{x,y+<k}, h_{x,y−<k′}⟩ − Σ_k Σ_k′ α+_{k,k′}⟨h_{x,y+<k}, h_{x,y+<k′}⟩, with coefficients in [−2, 2] set by next-token distributions. On average over 69% of the coefficients were positive across OLMo-1B, Gemma-2B, Llama-3-8B on UltraFeedback and AlpacaFarm (App H.2). Setting all coefficients to 1 gives:
CHES_x(y+, y−) = ⟨Σ_{k=1}^{|y+|} h_{x,y+<k}, Σ_{k′=1}^{|y−|} h_{x,y−<k′}⟩ − ‖Σ_{k=1}^{|y+|} h_{x,y+<k}‖² (Definition 2).
- h_{x,z<k}: hidden embedding after prompt x and the first k−1 tokens of z. A higher score means more similar preferences. Computing it takes one forward pass over the dataset (§5).
- Length-normalized CHES divides the first term by |y+|·|y−| and the second by |y+|² (App A, Definition 3). It is used for filtering because raw CHES correlates with length: over 99% of hidden-embedding inner products were positive for Llama-3-8B on UltraFeedback, so pairs with longer y− get higher raw scores (§5).

**Other analysis.** Additional samples lower ln π(y+|x) further when y+ is the dispreferred response of another prompt, and can contribute negatively even when their preferences differ (§4.2.3; App D.3, Theorem 8). An SFT term −λ ln π(y+|x) adds λ‖∇ ln π(y+|x)‖² ≥ 0 to d/dt ln π(y+|x) (App E, Proposition 1). When displacement occurs, y+ is often still more probable than y−, so this differs from failures to learn the ranking (§7).

**Experiments.**
- §3: Persona prompts; OLMo-1B No/Never: π(y+) 0.85 → 0.01, gaining tokens include Yes and yes; Llama-3-8B Sure/Yes: 0.98 → 0.39, gaining tokens include Maybe, No, Never (Table 1, mean of ten runs). Base models without SFT (Table 2) and IPO (Table 3) gave similar results (App H.1).
- §5: subsets of 512 samples around the 0th/25th/50th/75th/100th percentile, three runs each. Higher CHES percentile gave a larger decrease of the preferred log probability for all models; the lowest-CHES subsets increased it (Figure 2). Similar results on AlpacaFarm, with IPO, and with OLMo-1B (App H.3, Figures 5–7).
- §6: in over 70% of prompts both sampled responses were refusals (§6.2); for Llama-3-8B-Instruct, 262/370 pairs (71%) had two refusals, 73/370 (20%) two non-refusals, 35/370 (9%) one of each (Figure 4). Training on only two-refusal pairs or only two-non-refusal pairs also caused the drop (§6.3). Mean change in preferred log probability, Gemma-2B-IT / Llama-3-8B-Instruct, three runs (Table 16): DPO −59.2 ± 5.3 / −48.1 ± 22.1; DPO+SFT +20.2 ± 2.4 / +28.6 ± 0.3; DPO on gold pairs +54.6 ± 3.2 / +24.9 ± 3.0; DPO on filtered pairs −45.7 ± 2.5 / −27.7 ± 2.7. Keeping up to 15% of samples gave analogous results; keeping more led to refusal-rate drops (§6.3, footnote 9).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OLMo-1B, Gemma-2B, Llama-3-8B (§3 runs) | 1B / 2B / 8B | SFT | data; epochs; optimizer; LR; batch | Persona "ends justify means", 1000 prompts; 1; RMSProp; 1e-7; 32 | arXiv:2410.08847v4 App I.1 | verified 2026-09-14 | no ablation reported |
| OLMo-1B, Gemma-2B, Llama-3-8B (§3 runs) | 1B / 2B / 8B | preference | DPO steps; prompts; LR; KL coefficient; seeds | 100; 1 prompt per run; 1e-7; 0.1; 10 | App I.1 | verified 2026-09-14 | LR 5e-7 or 5e-8 gave analogous results (App I.1) |
| OLMo-1B, Gemma-2B, Llama-3-8B (§3 runs) | 1B / 2B / 8B | preference | IPO LR; KL coefficient | 1e-8; 0.01 | App I.1 | verified 2026-09-14 | higher LR unstable; higher coefficient left log probabilities nearly unchanged (App I.1) |
| OLMo-1B, Gemma-2B, Llama-3-8B (§5 runs) | 1B / 2B / 8B | preference | data; max length; epochs; LR; batch; DPO KL coefficient | UltraFeedback binarized 5000-sample subset, AlpacaFarm human preferences 9691 samples; 512 tokens; 1; 1e-7; 32 (4 × 8 accumulation); 0.1 | App I.2 | verified 2026-09-14 | LR 5e-7 or 5e-8 analogous (App I.2) |
| Gemma-2B-IT, Llama-3-8B-Instruct (§6) | 2B / 8B | preference | data; pair sampling | SORRY-Bench "base" 450 prompts minus 15, 85%/15% split; 2 on-policy samples, temperature 1, 512 max tokens; ties broken by PairRM | App I.3; §6.1 | verified 2026-09-14 | random tie-breaking gave similar results (§6.1, footnote 8) |
| Gemma-2B-IT, Llama-3-8B-Instruct (§6) | 2B / 8B | preference | epochs; optimizer; batch; DPO KL coefficient; IPO KL coefficient | 1; RMSProp; 32 (4 × 8 accumulation); 0.1; 0.01 | App I.3 | verified 2026-09-14 | no ablation reported |
| Gemma-2B-IT, Llama-3-8B-Instruct (§6) | 2B / 8B | preference | LR: original / filtered / gold data | 5e-6 / 5e-6 / 1e-6 (same for IPO) | App I.3 | verified 2026-09-14 | largest LR with lower training loss after 1 epoch, grid {1e-7, 5e-7, 1e-6, 5e-6, 1e-5}, 3 seeds (App I.3) |
| Gemma-2B-IT, Llama-3-8B-Instruct (§6) | 2B / 8B | preference | SFT-term coefficient: DPO Gemma / DPO Llama / IPO both | 1 / 0.1 / 1000 | App I.3 | verified 2026-09-14 | highest mean training-set refusal rate, DPO grid {0.01, 0.1, 1}, IPO grid {10, 100, 1000}, 3 seeds |
| Gemma-2B-IT, Llama-3-8B-Instruct (§6) | 2B / 8B | preference | filtered subset | 5% of samples with lowest length-normalized CHES | §6.3 | verified 2026-09-14 | up to 15% analogous; more caused refusal drops (§6.3, footnote 9) |
| Gemma-2B-IT, Llama-3-8B-Instruct (§6) | 2B / 8B | eval-gate | refusal judge; temperature; max new tokens | SORRY-Bench judge; 0.7; 512 | App I.3 | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality, negative feedback
**Negative feedback (course standard §6.1, negative as gradient).** The dispreferred response enters as the rejected term of DPO or IPO. Displacement occurred on off-policy data (UltraFeedback, AlpacaFarm) and on on-policy pairs (§5, §6). The diagnostic used is the mean preferred-response log probability (Figure 2, Table 16). Gold pairs (human-labeled responses from diverse models, one refusal and one non-refusal) did not show displacement (§6.3, Table 16). On filtered data the mean preferred log probability still decreased (Table 16) while refusal rates were restored (Figure 3); the text does not discuss this.

**Generality.** The authors state that DPO's reported gains in general instruction following suggest displacement is often benign there, and they do not claim catastrophic displacement is universal (§6, §8.1). They read earlier reports of DPO lowering math and reasoning scores as catastrophic displacement, because few responses are correct in those tasks (App C) (Interpretation).

## Connections
- [[dpo]], [[ipo]], [[simpo]] — losses covered by the analysis (App B, App E).
- [[dpo-positive]] — DPOP (Pal et al. 2024), which weights y+ to counter displacement; its edit-distance explanation did not predict displacement in §5.
- [[rpo]] — Pang et al. (2024), one of the cited works that adds an SFT term to the DPO loss.
- [[learning-dynamics-llm-finetuning]] — Ren and Sutherland (2024), the squeezing effect discussed in App C.
- [[on-policy-suboptimal-preference-data]] — Tajwar et al. (2024), earlier explanations based on dataset size and model capacity (App C).
- [[preference-ranking-accuracy]] — Chen et al. (2024), DPO failing to fix rankings, distinguished in §7.
- [[finetuning-compromises-safety]] — unalignment from benign SFT data (Qi et al. 2024, cited in §7).
- [[ultrafeedback]], [[pairrm]] — dataset in §5; tie-breaking reward model in §6.
- [[lazy-likelihood-displacement-grpo]] — negative-gradient effects in GRPO; [[negative-sample-reinforcement]], [[nft-negative-aware-finetuning]] — negatives as gradient in RLVR.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2410.08847 (arXiv v4, 2025-04-27, PDF; v1 2024-10-11).
- Audit claims not found in the source: "very similar pairs leave little push-up signal for y+ beyond the push-down" (not stated; the paper states that more similar y+ and y− make ln π(y+|x) decrease more, §4.2.2); "the SFT term reduces the worst drops" (the paper states the SFT term also prevents the drop in refusal rates, and filtering raises refusal rates more, §6.3); "the single-token update direction is W_{y+} − W_{y−}" (the paper states the log-probability change of other tokens is proportional to ⟨W_z, W_{y+} − W_{y−}⟩ up to a term independent of z, Theorem 5).
- Not reported by the source: refusal-rate values for DPO+SFT, gold, and filtered runs (bars only, Figure 3); test-set refusal rates (text says "similar").
