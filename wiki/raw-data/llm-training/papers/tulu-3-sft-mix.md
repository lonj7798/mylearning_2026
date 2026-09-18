<!-- scope: the released Tülu 3 SFT data mixture — component datasets, prompt counts, and the SFT settings trained on it
     deps: [[tulu-3]]
     see-also: [[allenai-tulu-blog]], [[rlvr-tulu3]], [[olmo-2]], [[self-instruct]], [[magpie]], [[persona-hub]]
-->

# Tulu 3 SFT Mixture
- **Core Insight:** The released Tülu 3 SFT mixture is stated on its dataset card as 939,344 samples drawn from 18 component sets, of which Ai2's blog reports 57% come from public resources and 43% are generated in house.
- **Guideline:** When assembling an SFT mixture for broad skill coverage, build and select skill-specific submixtures first, then combine, decontaminate against the evaluation suite, and downsample large components, because that is the procedure Ai2 reports producing the Tülu 3 mix (Tülu 3 paper §4.1.2).
- **Authors:** Allen Institute for AI (dataset card); the associated paper is Lambert, Morrison, Pyatkin, Huang, Ivison, Brahman, et al.
- **Year:** 2024 (dataset card, Nov 2024; companion paper arXiv:2411.15124 v1 2024-11)
- **URL:** https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- **Source type:** model/dataset card (official)
- **Relevant topics:** SFT mixture design, synthetic prompts, decontamination, instruction tuning, open post-training

## Summary
The dataset card publishes the exact component list of the SFT mixture used to train the Tülu 3 SFT checkpoints at 8B and 70B. It states 939,344 samples over 18 named sets, gives a per-set prompt count and license for each, and defines the record schema (`id`, `messages`, `source`). The card does not describe the construction procedure; that is in the Tülu 3 paper §3-§4 and in Ai2's technical blog.

## Key Contributions
- Publishes a per-component prompt count for a full open SFT mixture, with a license per component.
- Names the in-house persona-synthesized components separately from imported public sets, so the public/synthetic proportion can be audited at the component level.
- Ties the mixture to specific released checkpoints (SFT, DPO, RLVR, RM) in the Model Family table on the card.
- Provides the training substrate that the Tülu 3 paper's SFT ablations (Table 32) are run against.

## Key Figures/Tables to Study
- Dataset card component list — the 18 sets and their prompt counts.
- Dataset card "Model Family" table — which checkpoint each stage produced.
- Tülu 3 paper Figure 3 — average and per-skill performance across the intermediate SFT mixes.
- Tülu 3 paper Table 32 — data-source ablations of the 8B SFT model on development and unseen evaluations.

## Technical Details
### Component list as published
CoCoNot 10,983; FLAN v2 via `ai2-adapt-dev/flan_v2_converted` 89,982; No Robots 9,500; OpenAssistant Guanaco 7,132; Tulu 3 Persona MATH 149,960; Tulu 3 Persona GSM 49,980; Tulu 3 Persona Python 34,999; Tulu 3 Persona Algebra 20,000; Tulu 3 Persona IF 29,980; NuminaMath-TIR 64,312; Tulu 3 WildGuardMix 50,000; Tulu 3 WildJailbreak 50,000; Tulu 3 Hardcoded 240; Aya 100,000; WildChat GPT-4 100,000; TableGPT 5,000; SciRIFF 10,000; Evol CodeAlpaca 107,276 (dataset card, component list).

- **Count discrepancy in the primary source.** The 18 listed counts sum to 889,344, while the card's prose states 939,344 samples and the card's dataset viewer reports 939,343 rows (dataset card, component list; card footer "Number of rows"). The gap is 50,000, the size of a single listed component. The card does not explain it. Cite the stated 939,344 total and the per-component counts separately; do not present the list as an exhaustive decomposition of the total.
- The 57% public / 43% in-house split is an aggregate over the 939,344 collected prompts, stated in Ai2's blog, not in the dataset card (allenai.org/blog/tulu-3-technical, "In total, we collect 939,344 prompts…"). The card publishes no per-component public/in-house label.
- Record schema: `id` (str), `messages` (list, user prompt and assistant responses), `source` (str, the component dataset) (dataset card, "Dataset Structure").
- Mixture license is ODC-BY-1.0, with different licenses on subsets and some subsets non-commercial (dataset card, header note).

### How the mixture was constructed (companion paper)
1. Identify skills lagging behind state of the art using Llama 3.1 trained on the Tülu 2 mix as the baseline (§4.1.2).
2. Build skill-specific mixtures and models, keeping the mixture that is best on each individual skill in isolation, to approximate an upper bound per evaluation (§4.1.2).
3. Combine these into a preview mix, then iterate: add or remove datasets for lagging skills, decontaminate, and downsample large datasets (§4.1.2, Figure 3).
4. For prompts without responses, or whose responses came from a weaker model, generate new responses with GPT-4o; hardcoded prompts were hand-written (§4.1.1). All new synthetic SFT datasets use GPT-4o responses, or claude-3-5-sonnet for coding (blog; paper §3.1.2 for Python programs).
5. Persona-driven synthesis uses GPT-4o-2024-08-06 for all persona data unless stated otherwise (§3.1.2).

### Decontamination rule
Contamination is measured with 8-gram matching: a test instance is counted as overlapping a training instance if more than 50% of the test instance's tokens have 8-gram matches with that same training instance. A training set is treated as contaminated if any number of its instances overlap with more than 2% of the instances in any evaluation in the development or unseen suites (§3.2). Table 8 lists the datasets released in decontaminated form and the percentage removed from each.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Tülu 3 SFT mixture | — | SFT | mixture size (samples) | 939,344 stated in prose; 939,343 rows reported by the viewer; listed components sum to 889,344 | HF `allenai/tulu-3-sft-mixture` dataset card | conflict (within one source) | no ablation reported |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak learning rate, schedule | 5 × 10⁻⁶, linear | arXiv:2411.15124 Table 11 | verified 2026-09-18 | §4.3.1 Figure 5: sum loss at 5e-6 best among 2e-6/5e-6/1e-5/2e-5 on Llama 3.0 + Tülu 2 mix |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | peak learning rate, schedule | 2 × 10⁻⁶, linear | arXiv:2411.15124 Table 11 | verified 2026-09-18 | §4.3 "found after a hyperparameter search"; no table given |
| Llama-3.1-Tulu-3-8B-SFT / 70B-SFT | 8B, 70B | SFT | effective batch size (sequences) | 128 | arXiv:2411.15124 Table 11, §4.3 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-8B-SFT / 70B-SFT | 8B, 70B | SFT | max token length | 4,096 | arXiv:2411.15124 Table 11 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-8B-SFT / 70B-SFT | 8B, 70B | SFT | warmup ratio | 0.03 | arXiv:2411.15124 Table 11 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-8B-SFT / 70B-SFT | 8B, 70B | SFT | epochs | 2 | arXiv:2411.15124 Table 11, §4.3.1 | verified 2026-09-18 | §4.3.1 Figure 6: 2 epochs best over 2-7 epochs, Llama 3.0 on Tülu 2 mix |
| Llama-3.1-Tulu-3-8B-SFT / 70B-SFT | 8B, 70B | SFT | loss reduction | sum loss, not mean loss | arXiv:2411.15124 §4.3.1 | verified 2026-09-18 | §4.3.1 Figure 5: sum loss above mean loss at every learning rate tested |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | compute | 32 GPUs, 6 hours | arXiv:2411.15124 §4.3 | verified 2026-09-18 | not applicable |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | compute | 64 GPUs, 50 hours | arXiv:2411.15124 §4.3 | verified 2026-09-18 | not applicable |

## Findings relevant to generality
- The mixture choices transfer on average: the final SFT checkpoint has the best average on both the development suite and the unseen suite among the data-ablated variants (arXiv:2411.15124 §7.4, Table 32).
- The same table shows narrowing in precise instruction following. Tülu 3 8B SFT scores 72.8 on IFEval (development) but 17.6 on IFEval-OOD (unseen); removing WildChat lowers IFEval to 70.1 while raising IFEval-OOD to 20.8. The paper states the data choices "overfit to the development evaluations in Precise Instruction Following, and to some extent in Knowledge Recall and Reasoning" (§7.4).
- Persona data is the dominant contributor to the IFEval score: removing it drops IFEval from 72.8 to 53.6 while IFEval-OOD stays at 18.0 (Table 32).
- Scaling the mixture by stratified subsampling improves the average and GSM8K, but TruthfulQA performance falls as mixture size grows (§4.2, Figure 4).
- IFEval-OOD is 52 constraints across six categories, built to test constraints outside IFEval's 25 (§7.3.1).

## Connections
- [[tulu-3]] — the report that documents how this mixture was built and trained on.
- [[rlvr-tulu3]] — the RL stage that consumes the checkpoints trained on this mixture.
- [[allenai-tulu-blog]] — source of the 57/43 public-vs-in-house aggregate.
- [[persona-hub]] — the persona-driven synthesis method used for the Persona MATH/GSM/Python/Algebra/IF components.
- [[self-instruct]], [[magpie]] — earlier prompt-synthesis methods this mixture's in-house components extend.
- [[olmo-2]] — reuses the Tülu 3 post-training recipe.

## Verification
- Checked on 2026-09-18 against: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture (card as served 2026-09-18), https://arxiv.org/abs/2411.15124 (v1 PDF), https://allenai.org/blog/tulu-3-technical
- Corrections to the previous card version:
  - "**Authors:** Nathan Lambert, Jacob Morrison, …" → the primary artifact is a dataset card published by the Allen Institute for AI; the individual author list belongs to the companion paper arXiv:2411.15124 (dataset card "Citation").
  - "WildGuardMix, 50,000 prompts" → the card's component name is "Tulu 3 WildGuardMix" (dataset card, component list).
  - The card presented the 18 counts as a decomposition of 939,344. They sum to 889,344, and the viewer reports 939,343 rows. Both facts are now stated.
  - "Ai2 explicitly removes overlap with more than 2% of the eval suite" → the rule is that a training set is contaminated if any number of its instances overlap with more than 2% of the instances in any evaluation, using 8-gram matching and a 50%-of-test-tokens threshold (arXiv:2411.15124 §3.2).
  - Missing **Source type** field added.
- Removed as unsupported by the source:
  - The "Public and imported sources" / "In-house synthetic family" two-group split and the note explaining it. Neither the dataset card nor the paper labels components this way; the flat list is now given as published.
  - "Synthetic bias: the in-house family is generated by strong proprietary models, so style and policy behavior can bleed into the student." Not measured or claimed in either source.
  - "Provenance complexity" and "Aggregation ambiguity" risk bullets; the license heterogeneity they referred to is stated directly from the card instead.
- Not reported by the source: token counts per component; the response-generating model per component; the row-level public/in-house label; the reason for the 50,000-sample count gap.
