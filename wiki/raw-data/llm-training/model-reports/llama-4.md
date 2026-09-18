<!-- scope: Meta's Llama 4 launch post (2025-04-05): Scout, Maverick, unreleased Behemoth teacher; MoE + early-fusion pre-training, codistillation during pre-training, lightweight SFT > online RL > lightweight DPO with difficulty filtering, iRoPE long context
     deps: [[llama-3]]
     see-also: [[gemma-2]], [[magistral]], [[deepseek-v3]], [[qwen-3]], [[grok-3]], [[nathan-lambert-interconnects]]
-->

# The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation
- **Core Insight:** Meta reports that "SFT and DPO can over-constrain the model, restricting exploration during the online RL stage", so for Llama 4 Maverick it removed more than 50% of the data tagged as easy by a Llama judge before a lightweight SFT, and for Behemoth ("nearly two trillion total parameters") it pruned 95% of the SFT data (post, "Post-training our new models"; "Pushing Llama to new sizes"); no ablation numbers are given.
- **Guideline:** When online RL follows SFT, the post supports removing easy prompts before SFT and re-filtering RL prompts to medium-to-hard difficulty during training; because the post reports no controlled numbers, take effect sizes from a report with ablations such as [[magistral]].
- **Authors:** Meta (official post, no named authors)
- **Year:** 2025 (published 2025-04-05)
- **URL:** https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- **Source type:** official blog (supplemented only where marked by the official model card, github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md)
- **Relevant topics:** mixture-of-experts, early fusion, codistillation, SFT data pruning, online RL, zero-advantage filtering, pass@k curriculum, DPO, long context, iRoPE

## Summary
The post releases Llama 4 Scout and Llama 4 Maverick as open-weight models and previews Llama 4 Behemoth, a larger teacher that was still training. All three are mixture-of-experts (MoE) models, in which each token activates only a subset of the parameters, and all use early fusion, in which text and vision tokens enter one backbone. The post describes pre-training (MetaP, FP8, a data mixture of more than 30T tokens), mid-training for long context, codistillation of Maverick from Behemoth during pre-training, and a post-training pipeline of lightweight SFT, online RL, and lightweight DPO. It gives no training hyperparameter values, no RL algorithm name, and no benchmark tables in the page text.

## Key Contributions
- Post-training order "lightweight supervised fine-tuning (SFT) > online reinforcement learning (RL) > lightweight direct preference optimization (DPO)", motivated by the observation that SFT and DPO restrict exploration in RL ("Post-training our new models").
- Difficulty-based data removal: more than 50% of data tagged easy removed for the smaller models; 95% of SFT data pruned for Behemoth ("Post-training"; "Pushing Llama to new sizes").
- Continuous online RL: alternating training with using the current model to retain only medium-to-hard prompts ("Post-training").
- Codistillation of Maverick from Behemoth during pre-training with a loss that "dynamically weights the soft and hard targets through training" ("Pushing Llama to new sizes").
- iRoPE long-context design for Scout: interleaved attention layers without positional embeddings plus inference-time attention temperature scaling ("Post-training", Scout paragraphs).

## Key Figures/Tables to Study
- The page text contains no benchmark table. The model card prints pre-trained and instruction-tuned benchmark tables, including MTOB long-context results (model card, "Benchmarks").

## Technical Details
### Models
- Scout: 17B active parameters, 16 experts, 109B total ("Post-training", Scout paragraph); fits one H100 GPU with Int4 quantization (intro).
- Maverick: 17B active, 400B total; alternating dense and MoE layers; MoE layers use 128 routed experts and a shared expert, and each token goes to the shared expert and one routed expert; fits one H100 DGX host ("Pre-training").
- Behemoth: 288B active, 16 experts, "nearly two trillion total parameters"; still training and not released ("Pushing Llama to new sizes").

### Pre-training and mid-training ("Pre-training")
- Early fusion allows joint pre-training on unlabeled text, image, and video. The vision encoder is based on MetaCLIP and trained separately with a frozen Llama model.
- MetaP sets per-layer learning rates and initialization scales; Meta reports the chosen values "transfer well" across batch size, width, depth, and training tokens. No values are given.
- Pre-training covers 200 languages, over 100 of them with more than 1B tokens each, and 10x more multilingual tokens than Llama 3.
- Behemoth was pre-trained in FP8 on 32K GPUs at 390 TFLOPs/GPU.
- The data mixture has "more than 30 trillion tokens", more than double the Llama 3 mixture. The model card reports ~40T tokens for Scout and ~22T for Maverick (model card, "Training Data"); neither source states whether these are unique or seen tokens.
- Mid-training used "long context extension using specialized datasets".
- The models were pre-trained on up to 48 images; post-training was tested "with good results up to eight images" ("Post-training"). The model card states testing up to 5 input images (model card, "Intended Use" note 2).

### Codistillation ("Pushing Llama to new sizes")
- Maverick was codistilled from Behemoth as teacher. The takeaways attribute the quality of both Scout and Maverick to "distillation from Llama 4 Behemoth" (intro), but the technical paragraph names only Maverick. The loss weights soft and hard targets dynamically during training; the weighting schedule is not given. The post does not define the terms; in standard usage a soft target is the teacher's output distribution and a hard target is the data token (Interpretation).
- Codistillation happened "during pre-training", which "amortizes the computational cost of resource-intensive forward passes needed to compute the targets for distillation for the majority of the training data used in student training". For new data added to student training, Meta ran extra Behemoth forward passes.
- Reported effect: "substantial quality improvements across end task evaluation metrics", without numbers.

### Post-training: Scout and Maverick ("Post-training our new models")
1. More than 50% of the data tagged easy by Llama-model judges is removed; lightweight SFT runs on the remaining harder set.
2. Multimodal online RL on harder prompts, which the post says gave "a step change in performance".
3. Continuous online RL: training alternates with using the model to filter prompts, keeping medium-to-hard ones. Meta calls this "highly beneficial in terms of compute and accuracy tradeoffs".
4. Lightweight DPO "to handle corner cases related to model response quality".
- Modality mixing used "a carefully curated curriculum strategy that does not trade-off performance compared to the individual modality expert models".

### Post-training and RL infrastructure: Behemoth ("Pushing Llama to new sizes")
- 95% of SFT data pruned, compared with 50% for smaller models; lightweight SFT followed by large-scale RL.
- Hard prompts are sampled "by doing pass@k analysis with the policy model", and the curriculum increases prompt hardness. The value of k is not given.
- Prompts with zero advantage are filtered out dynamically during training, and batches mix prompts from multiple capabilities. (In group-based advantage estimates, a prompt has zero advantage when all its samples receive the same reward, so it contributes no policy gradient; this definition is added by the card.)
- Fully asynchronous online RL framework that places different models on separate GPUs; "~10x improvement in training efficiency over previous generations".

### Long context and other reported numbers
- Scout supports 10M tokens, up from 128K in Llama 3, and was "both pre-trained and post-trained with a 256K context length" ("Post-training", Scout paragraph). The model card lists 10M for Scout and 1M for Maverick (model card, table).
- iRoPE: RoPE in most layers, interleaved attention layers without positional embeddings, and inference-time temperature scaling of attention (same paragraph).
- Bias: refusals on debated political and social topics fall from 7% (Llama 3.3) to below 2%; unequal response refusals below 1% ("Addressing bias in LLMs").
- LMArena ELO 1417 is reported for "an experimental chat version" of Maverick (intro). The post does not state that this version is the released checkpoint.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 4 Scout | 17B act. / 109B | pretrain-stable | tokens | ~40T (unit not defined) | model card, "Training Data" | verified 2026-09-14 | no ablation reported |
| Llama 4 Maverick | 17B act. / 400B | pretrain-stable | tokens | ~22T (unit not defined) | model card, "Training Data" | verified 2026-09-14 | no ablation reported |
| Llama 4 (family) | all | pretrain-stable | data mixture size | "more than 30 trillion tokens" | post, "Pre-training" | verified 2026-09-14 | no ablation reported |
| Scout / Maverick | 17B act. / 109B; 17B act. / 400B | pretrain-stable | compute | 5.0M / 2.38M H100-80GB GPU hours | model card, "Training Energy Use" table | verified 2026-09-14 | not applicable |
| Llama 4 Behemoth | 288B act. / ~2T | pretrain-stable | precision, hardware | FP8, 32K GPUs, 390 TFLOPs/GPU | post, "Pre-training" | verified 2026-09-14 | no ablation reported |
| Llama 4 Maverick | 17B act. / 400B | pretrain-stable | distillation | codistilled from Behemoth; soft/hard target weights change during training | post, "Pushing Llama to new sizes" | verified 2026-09-14 | "substantial quality improvements", no numbers |
| Llama 4 Scout | 17B act. / 109B | long-context | pre- and post-training context | 256K | post, Scout paragraph | verified 2026-09-14 | no ablation reported |
| Scout / Maverick | 17B act. / 109B; 17B act. / 400B | SFT | data removed | >50% of data tagged easy (Llama judge) | post, "Post-training" | verified 2026-09-14 | qualitative: SFT/DPO "over-constrain" RL |
| Llama 4 Behemoth | 288B act. / ~2T | SFT | data removed | 95% of SFT data | post, "Pushing Llama to new sizes" | verified 2026-09-14 | no ablation reported |
| Llama 4 Behemoth | 288B act. / ~2T | RL | prompt selection | pass@k with policy; increasing hardness; zero-advantage filtering; mixed-capability batches; varied system instructions | post, "Pushing Llama to new sizes" | verified 2026-09-14 | "instrumental" for math, reasoning, coding; no numbers |
| Scout / Maverick / Behemoth | all | SFT, RL, preference | SFT and DPO sizes, LR, batch, RL algorithm, reward, KL, clip ε, samples per prompt, k, DPO β, MetaP values, sequence lengths other than Scout 256K | not reported | checked: full post text, model card | not reported | not applicable |

## Findings relevant to generality, negative feedback, long context, distillation
- **Generality.** Behemoth: "sampling from a variety of system instructions was crucial in ensuring that the model retained its instruction following ability for reasoning and coding" (post, "Pushing Llama to new sizes"). Status: Result (single study), no numbers.
- **Negative feedback.** Data tagged easy and zero-advantage prompts are discarded (negative marginal value, not used as gradient). DPO pairs are not described.
- **Long context.** Scout is trained at 256K and supports 10M. The post's evidence is NIAH retrieval and cumulative NLL over 10M code tokens. The model card's "Long Context" rows for the instruction-tuned models are MTOB half book and full book (chrF, eng→kgv/kgv→eng); full book: Scout 39.7/36.3, Maverick 50.8/46.7 (model card, "Instruction tuned models").
- **Distillation.** Codistillation during pre-training reuses teacher forward passes for most student data, and the loss re-weights soft and hard targets during training; no student-versus-baseline numbers are given.

## Connections
- [[llama-3]] — the post compares multilingual tokens (10x), data mixture size (more than double), and context (128K) with Llama 3.
- [[gemma-2]] — distillation from a teacher distribution during pre-training, with the ablation numbers this post lacks.
- [[magistral]] — also filters zero-advantage groups during RL, and reports ablations of other RL settings (batch size, advantage normalization, reward shape).
- [[deepseek-v3]], [[qwen-3]] — MoE technical reports that print training hyperparameters.
- [[grok-3]] — another 2025 release documented only by a blog post.
- [[nathan-lambert-interconnects]] — "Llama 4: Did Meta just push the panic button?" (2025-04-07, interconnects.ai/p/llama-4) writes that the 10M context "didn't have any released evaluations beyond Needle in a Haystack (NIAH)" and that the LMArena result came from a different model (practitioner commentary).
- [[ruler]] — long-context benchmark that tests more than NIAH retrieval.
- [[bradley-terry-rm]] — records "The Leaderboard Illusion" count of 27 private Meta variants tested on Arena before Llama 4.

## Verification
- Checked on 2026-09-14 against: https://ai.meta.com/blog/llama-4-multimodal-intelligence/ (dated April 5, 2025, full page text); model card at github.com/meta-llama/llama-models main, models/llama4/MODEL_CARD.md; Interconnects post (for the Connections line only).
- Corrections to the previous card version:
  - Title "Llama 4" → published title; Core Insight and Guideline rewritten to the post's statements.
  - Distillation loss listed as post-training → codistillation "during pre-training", with amortized teacher targets and extra forward passes only for new data.
  - "Pass@k curriculum (used for Behemoth): only prompts where the policy sometimes-but-not-always succeeds stay" → pass@k analysis to sample hard prompts plus a curriculum of increasing hardness; zero-advantage prompts filtered dynamically.
  - "the model filters its own next-round training set via pass@k" → the continuous-filtering loop is described for Scout/Maverick; pass@k is described only for Behemoth.
  - "extensive asynchronous online RL ... ~10× throughput" → "fully asynchronous online RL training framework", "~10x improvement in training efficiency".
  - Quoted "pass@k evaluation and curriculum sampling to strengthen performance in math, logic, and coding" → not in the post; replaced with the post's zero-advantage and mixed-batch sentence.
  - "SFT is minimized to avoid over-fitting the model to easy distribution" → the post's reason is that SFT and DPO restrict exploration in online RL.
  - "Multimodal SFT included" → the post names a multimodal online RL stage; SFT modality not stated.
  - "10M-token context for Scout (claimed)" → 10M supported, 256K pre- and post-training, iRoPE, attention temperature scaling, NIAH and NLL evidence.
  - Behemoth 95% SFT pruning was missing → added.
- Removed as unsupported by the source: "assumed PPO-family based on Llama 3 lineage"; "Llama 3: SFT → RM → DPO + PPO"; "blog implies model-as-judge ... verifiers inside RL"; "Llama 3 did not distill across model sizes at this scale"; "Llama 3's bolted-on adapter"; "moved away from the synchronous generation/training stack used for Llama 3"; "128 experts enable capacity" rationale for the distillation target; "Llama 3's 73-page paper"; "Curriculum diagram (blog)"; guideline "Throw away ≥50% of your SFT data".
- Not reported by the source: RL algorithm, reward design, KL, any optimizer value, SFT/DPO data sizes, k in pass@k, distillation loss formula, benchmark values in page text.
