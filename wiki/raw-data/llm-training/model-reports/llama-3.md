<!-- scope: The Llama 3 Herd of Models (arXiv:2407.21783): Llama 3.1 8B/70B/405B dense models; scaling-law sizing and downstream forecasting, pre-training data mix, 405B schedule, long-context stage, annealing; six post-training rounds of reward modeling, rejection sampling, SFT, and DPO; capability-specific synthetic data; contamination and safety
     deps: [[llama-2]], [[dpo]]
     see-also: [[llama-3-recipe]], [[long-context-llama3]], [[llama-3-synthetic-pipeline]], [[rpo]], [[rejection-sampling-finetuning]], [[tulu-3]]
-->

# The Llama 3 Herd of Models
- **Core Insight:** Llama 3.1 405B, a dense Transformer pre-trained on 15.6T tokens with 3.8 × 10^25 FLOPs and post-trained in six rounds of reward modeling, rejection sampling, SFT, and DPO, is reported to deliver quality comparable to GPT-4 on many tasks (Abstract, §1, §4.1.6).
- **Guideline:** When long-context ability acquired in pre-training must survive SFT, include synthetic long-context examples in the SFT mix, because SFT with only short-context data caused significant long-context regressions and a 0.1% long-context share optimized both short- and long-context benchmarks in the authors' ablations (§4.3.4; no table printed).
- **Authors:** Llama Team, AI @ Meta (byline). The appendix lists core contributors alphabetically by first name: Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, et al.
- **Year:** 2024 (arXiv v1 2024-07; v3 2024-11-23; report dated July 23, 2024)
- **URL:** https://arxiv.org/abs/2407.21783
- **Source type:** official technical report
- **Relevant topics:** scaling laws for model size and benchmark accuracy, data mix selection, annealing, long-context continued pre-training, rejection sampling, SFT data quality control, DPO with NLL regularization, model averaging, synthetic data for code/math/multilingual/long context/tools/factuality, contamination analysis, safety data balance

## Abstract
Llama 3 is a herd of language models that natively support multilinguality, coding, reasoning, and tool use. The largest is a dense Transformer with 405B parameters and a context window of up to 128K tokens. The authors report that Llama 3 delivers quality comparable to leading models such as GPT-4 on many tasks, and they release pre-trained and post-trained 405B models together with Llama Guard 3 for input and output safety. The report also describes experiments that add image, video, and speech capabilities through a compositional approach; those models were not broadly released (Abstract). All reported results are for the Llama 3.1 models (§1).

## Key Contributions
- Scaling laws that set the flagship size (402B parameters on 16.55T tokens predicted for 3.8 × 10^25 FLOPs) and forecast benchmark accuracy in two steps: FLOPs to normalized NLL of the correct answer, then NLL to accuracy (§3.2.1, Figure 4).
- A pre-training recipe with a data mix chosen by small-scale scaling-law experiments, a six-stage long-context stage, and a final annealing phase with checkpoint averaging (§3.1.2, §3.4).
- Annealing on a small dataset as a method to estimate that dataset's value (§3.1.3).
- A post-training loop of RM training, rejection sampling, SFT, and DPO repeated for six rounds, with data cleaning, pruning, and model averaging (§4.1, §4.2).
- Capability-specific data pipelines: execution-feedback code data, translated reasoning data, long-context QA and summarization, tool-use trajectories, and knowledge-probing factuality data (§4.3).

## Key Figures/Tables to Study
- Figures 2–4: IsoFLOP curves, compute-optimal tokens, and the ARC Challenge forecast.
- Table 3: architecture and peak LR per size; Table 4: parallelism and MFU per pre-training stage.
- Figure 7: the post-training loop.
- Table 6: human preference data composition; Table 7: SFT data composition.
- Table 15: 8-gram contamination estimates and estimated performance gains.
- Figure 18: violation rate vs false refusal rate for 8B and 70B safety mixes.

## Technical Details
Recipe values with loci are in [[llama-3-recipe]]. Long-context synthesis details are owned by [[long-context-llama3]]. This section summarizes the mechanisms.

**Architecture.** The architecture is a standard dense Transformer close to Llama 2, with GQA (8 KV heads), a document-boundary attention mask, a 128K-token vocabulary, and RoPE θ = 500,000 (§3.2, Table 3). The authors state that gains come mainly from data quality and diversity and from training scale (§3.2).

**Scaling laws and size choice.** Models from 40M to 16B parameters are trained at budgets from 6 × 10^18 to 10^22 FLOPs; a second-degree polynomial fit per budget gives the compute-optimal model (§3.2.1, Figure 2). The fit N*(C) = A C^α, where N*(C) is the compute-optimal number of training tokens for compute budget C and (α, A) = (0.53, 0.29), extrapolates to 402B parameters on 16.55T tokens at 3.8 × 10^25 FLOPs (§3.2.1). Because IsoFLOP curves flatten near the minimum at larger budgets, 405B was chosen (§3.2.1). For benchmark forecasts, normalized NLL of the correct answer is fit linearly against FLOPs using scaling-law models, and a sigmoid maps NLL to accuracy using both scaling-law models and Llama 2 models; the authors report that this extrapolation over four orders of magnitude "only slightly underestimates" the final 405B ARC Challenge result (§3.2.1, Figure 4).

**Pre-training data.** Web data passes PII and safety filters, a custom HTML parser, URL/document (MinHash)/line-level deduplication, heuristic filters, and model-based quality classifiers (fastText trained to recognize text referenced by Wikipedia, and DistilRoberta trained on Llama 2 quality judgments) (§3.1.1). Code and math pages are extracted with dedicated classifiers (§3.1.1). The final mix is roughly 50% general knowledge, 25% math and reasoning, 17% code, and 8% multilingual tokens (§3.1.2).

**Pre-training schedule.** The 405B model uses AdamW with peak LR 8 × 10^-5, 8,000 warmup steps, and cosine decay to 8 × 10^-7 over 1,200,000 steps; the batch grows from 4M tokens (sequence 4,096) to 16M tokens (§3.4.1). Long-context training raises the context from 8K to 128K in six stages over approximately 800B tokens (§3.4.2). Annealing over the final 40M tokens takes the LR linearly to 0 at 128K context, upsamples very high-quality sources, and averages checkpoints (§3.4.3). Training ran on up to 16K H100 GPUs; in a 54-day snapshot there were 466 job interruptions (419 unexpected) and effective training time was above 90% (§3.3.1, §3.3.4).

**Post-training loop.** Each round trains an RM on all preference data available so far, uses it for rejection sampling (K typically 10–30) on human-annotation prompts, fine-tunes the pre-trained model with SFT on rejection-sampled, synthetic, and human-curated data, and then applies DPO on the most recent preference batches (§4.1, §4.2.1, §4.2.2, Figure 7). The RM objective is Llama 2's without the margin term (§4.1.2). DPO uses LR 10^-5 and β = 0.1, masks header and termination tokens in both responses, and adds an NLL term with coefficient 0.2 on chosen sequences, "similar to Pang et al. (2024)" (§4.1.4). The authors explored PPO and found DPO required less compute and performed better, "especially on instruction following benchmarks like IFEval" (§4.1.4). Models from runs with different data or hyperparameters are averaged at the RM, SFT, and DPO stages (§4.1.5).

**Data quality control.** Rule-based removal and modification address patterns such as excessive emojis and exclamation points, and the share of samples with overused apologetic phrases ("I'm sorry", "I apologize") is balanced (§4.2.3). Pruning uses a Llama 3 8B topic classifier, quality scores from the RM (top quartile) or Llama 3 ratings (maximum score), difficulty scores, and semantic deduplication with RoBERTa clusters (§4.2.3).

**Capability data.** Code: a code expert continued pre-training on 1T tokens with >85% code; over 2.7M synthetic examples come from execution feedback (~1M), translation, and backtranslation (~1.2M) (§4.3.1). Multilingual: an expert trained on 90% multilingual tokens; multilingual SFT is 2.4% human, 44.2% other NLP tasks, 18.8% rejection-sampled, 34.6% translated reasoning data (§4.3.2). Math: answer-filtered and self-verified step-by-step solutions, outcome and stepwise reward models, MCTS for hard prompts, code-interleaved reasoning, and error correction from incorrect generations (§4.3.3). Tools: Brave Search, Python, and Wolfram Alpha with message-level human preference annotation and no rejection sampling (§4.3.5). Factuality: a knowledge probe generates refusals for questions the model answers informatively but incorrectly (§4.3.6).

## Recipe ledger
Full table (architecture, scaling-law runs, data mix, 405B schedule, long context, annealing, RM, rejection sampling, SFT, DPO, averaging, capability data, safety, decontamination): [[llama-3-recipe]].

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality and measurement.** Annealing on GSM8k and MATH training sets raised pre-trained 8B validation scores by 24.0% and 6.4% but had negligible effect at 405B; benchmark training sets were therefore excluded from annealing data "to assess the true few-shot learning capabilities and out-of-domain generalization" (§3.1.3). The 8-gram contamination analysis finds large estimated gains for some benchmarks (HellaSwag: 85% flagged, +14.8 at 8B) and small ones for others (NaturalQuestions: 52% flagged, estimated +1.6/+0.9/+0.8 for 8B/70B/405B, described as "virtually no effect"); for MBPP, HumanEval, MMLU, and MMLU-Pro the method could not produce an estimate (§5.1.4, Table 15). Post-training data is decontaminated by exact prompt match, and human-evaluation prompts were withheld from modeling teams (§5.2, §5.3). On adversarial QA and math benchmarks, performance is "substantially lower" than on the paired non-adversarial sets for both pre-trained and post-trained models (§5.1.3, Figure 15). A strict model-as-judge filter on code data first lowered benchmark scores because it removed hard prompts; revising the hardest responses restored balance (§4.3.1).
- **Negative feedback.** DPO applies gradient to rejected responses; the authors mask formatting tokens because tokens shared by chosen and rejected responses receive conflicting updates (the authors' hypothesis), and add NLL on chosen responses to prevent their log-probability from decreasing (§4.1.4). Pairs rated only "slightly" or "marginally better" are discarded (§4.2.1). Failing code is revised with parser, linter, and unit-test feedback, and only passing dialogs enter SFT (negative as content) (§4.3.1). Incorrect math generations are corrected by prompting and used as training data (§4.3.3). For safety DPO, response pairs that are "nearly orthogonal in an embedding space" were particularly effective (§5.4.3). Borderline prompts are added to reduce false refusals; 8B needs a higher safety-data proportion than 70B (§5.4.3, Figure 18).
- **Long context.** Short-context-only SFT regressed long-context ability; synthetic QA over 8K chunks, hierarchical summaries, and repository code reasoning, bucketed at 16K, 32K, 64K, and 128K, are mixed in at a 0.1% share; DPO on short-context data did not hurt long context, which the authors suspect is because DPO uses fewer optimizer steps than SFT (§4.3.4). Details: [[long-context-llama3]].
- **Agentic and tool use.** Tool data progresses from single-turn to multi-turn to multi-step and data-analysis annotation; zero-shot function-calling data is grounded in functions mined from the Stack; queries without tool need are added so the model does not call tools unnecessarily (§4.3.5).
- **Distillation.** The 8B and 70B models improve "significantly" when trained on data from a larger model, but training 405B on its own generated code data was "not helpful (and can even degrade performance)", which led to execution feedback as a correctness signal (§4.3.1). Smaller models are trained longer than compute-optimal and improved with the flagship during post-training (§1).

## Connections
- [[llama-3-recipe]] holds the verified hyperparameters and data shares.
- [[llama-2]] is the predecessor whose preference protocol, RM objective (minus the margin), and iterative rounds the report follows (§4.1.2, §4.1.6, §4.2.1).
- [[dpo]] is the preference-optimization algorithm used in each round (§4.1.4).
- [[rpo]] (Pang et al., 2024) is the cited precedent for the NLL term on chosen sequences (§4.1.4).
- [[rejection-sampling-finetuning]] describes the best-of-K selection used to build SFT data (§4.2.2).
- [[long-context-llama3]] is the library's owner card for the §3.4.2 and §4.3.4 long-context material; [[llama-3-synthetic-pipeline]] covers the synthetic-data pipelines.
- [[tulu-3]] is an open post-training recipe that also uses SFT followed by DPO; compare settings in its own card.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2407.21783 (v3, 2024-11-23)
- Corrections to the previous card version:
  - "Six iterative rounds … beats a single-pass RLHF pipeline" → no such comparison; the report states it chose SFT, rejection sampling, and DPO over more complex RL for stability and scalability, and compares DPO with PPO only in prose (§1, §4.1.4).
  - "Table 6 (SFT hyperparameters)" → Table 6 is human preference data statistics; SFT settings are prose in §4.1.3 (LR 10^-5, 8.5K–9K steps).
  - "Table 7 (DPO hyperparameters)" → Table 7 is SFT data statistics; DPO settings are prose in §4.1.4.
  - "Section 4.3 (Preference data collection)" with "negligibly better" → §4.2.1; the fourth level is "marginally better".
  - "Figure 12 (Safety vs helpfulness pareto, Llama Guard 3 operating point)" → Figure 12 is pre-trained 8B/70B benchmark performance; the VR/FRR trade-off is Figure 18 and Llama Guard 3 results are Tables 25–26.
  - "Llama Guard 3 trained jointly as the safety classifier" → Llama Guard 3 is a separate Llama 3 8B model fine-tuned for safety classification (§5.4.7).
  - "rejection sampling at temperature T=0.6–1.0" → general value not printed; multilingual rejection sampling used random 0.2–1 early and 0.6 in the final round (§4.3.2).
  - "topic classifier + quality classifier (both distilled from Llama 3)" → topic classifier is fine-tuned Llama 3 8B; quality is RM top quartile or maximum Llama 3 rating (§4.2.3).
  - "SFT: LR 1e-5 (405B), cosine decay, context 8K-32K" → LR 10^-5 over 8.5K–9K steps for "our largest models"; schedule and context length not printed (§4.1.3).
  - "RM: linear head replaces LM head; margin labels used for filtering / up-weighting" → objective is Llama 2's without the margin term; ratings are used to discard similar pairs; up-weighting not reported (§4.1.2, §4.2.1).
  - "DPO: single epoch per round; masks prompts" → epochs not printed; the masking described is of header and termination tokens (§4.1.4).
  - "Most-recent-batch preference data only (older batches cause format drift)" → "primarily" the most recent batches, so data matches the policy distribution; RM training uses all available preference data (§4.1.4, §4.2.1).
  - "8K native context, 8-way sequence parallel for long-context extension" → 8K pre-training context; long-context stage uses context parallelism 16 at sequence 131,072 (§2, Table 4).
  - "matches GPT-4" → "comparable quality to leading language models such as GPT-4 on a plethora of tasks" (Abstract).
  - "[[dpo]] — Llama 3's NLL add-on is a novel stabilizer" → the report calls it "similar to Pang et al. (2024)" (§4.1.4).
- Removed as unsupported by the source: "~50–80% synthetic rejection-sampled data" per round; post-training compute as "a small fraction" of pre-training; "full disclosure of failure modes (preference-data noise, multi-turn dialog drift)"; "[[tulu-3]] confirms DPO beta 0.1 as a robust default"; "[[reward-model-overoptimization]] — Llama 3 combats this by swapping RMs each round and never reusing stale preferences".
- Added from the source (flagged by audit): scaling-law extrapolation and two-step benchmark forecast; pre-training data mix; annealing results and anneal-as-data-evaluation; 405B schedule; Table 4 parallelism; Table 7 SFT composition; 0.1% long-context SFT share.
- Not reported by the source: optimizer betas, weight decay, and gradient clipping for the main runs; token counts for 8B and 70B; per-stage long-context lengths; SFT batch size and schedule; DPO epochs and batch size; general rejection-sampling temperature; total number of preference comparisons; model-averaging weights.
