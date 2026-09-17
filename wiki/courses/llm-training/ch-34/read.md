<!-- chapter: ch-34
     track: sft
     kind: content
     title: Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports
     deps: [ch-33]
     sources: [[qwen-2.5]], [[qwen-2.5-recipe]], [[qwen-long-context-synth]], [[online-merging-optimizer]], [[qwen-3]], [[qwen-3-hybrid-thinking]], [[qwen3-2507-instruct-thinking-split]], [[qwen3-coder]], [[qwen-3-5]], [[qwen-3-5-model-card]], [[olmo-2]], [[olmo-2-tulu-recipe]], [[open-instruct-allenai-recipes-recipe]], [[olmo-3]], [[olmo-3-model-flow]], [[allenai-olmo3-open-instruct-scripts]], [[allenai-olmo3-open-instruct-scripts-recipe]], [[phi-3]], [[phi-4]], [[phi-4-reasoning-sft-rl]], [[gsm1k]], [[emergence-loss-perspective-recipe]], [[tulu-3]], [[llama-3-recipe]], [[smollm-3]], [[smollm3-training-configs]], [[deepseek-v3-recipe]]
     figures: figures/lab-compare.html
     revised: 2026-09 (generality revision)
-->

# Chapter 34 — Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports

> **Core insight.** Each of the three report families prints at least one stage at which broad training lowered a specialized score, or specialized data lowered a broad one. In Qwen3-32B, the thinking-mode fusion and general-RL stages raised IFEval from 73.0 to 85.0 and in-house ToolUse from 63.3 to 85.5, while thinking-mode AIME'24 fell from 83.8 to 81.4 and LiveCodeBench v5 from 68.4 to 65.7 (one score per stage); the authors accepted this for versatility, and in July 2025 Qwen released separate Instruct and Thinking models in place of the hybrid Qwen3-235B-A22B ([[qwen-3-hybrid-thinking]] Table 22; [[qwen3-2507-instruct-thinking-split]]). In Phi-4's data ablations, a 13B model trained without web data gained 4.9 MATH and 12.1 HumanEval points but lost 14.8 TriviaQA points relative to phi-3-medium, and Olmo 3 7B Instruct RL raised AIME 2024 from 23.5 to 44.3 while lowering PopQA from 20.7 to 14.1 ([[phi-4-reasoning-sft-rl]] Table 3; [[olmo-3-model-flow]] Table 26). The printed SFT peak learning rates range from 1e-6 (Phi-4) to 1e-4 (one of the two merged Olmo 3 32B Think runs), and among the 11 model rows of the SFT ledger grid in the companion figure, the batch is not reported for 7, packing for 8, and the checkpoint-selection rule for 8.
>
> **Guideline.** When a stage adds a mode or a capability family (non-thinking answers, tool use, open-ended chat), re-run the previous stage's specialist benchmarks in every mode before and after the stage, because Qwen3's thinking-mode AIME'24 and LiveCodeBench v5 fell by 2.4 and 2.7 points in a single-run stage table while general scores rose. When a smaller model shares a tokenizer with a stronger teacher and the target is math and code, use on-policy distillation to the teacher's logits instead of RL on the small model, because on Qwen3-8B it reached AIME'24 74.4 against 67.6 for RL at 1,800 versus 17,920 GPU hours and raised AIME'24 pass@64 from 90.0 to 93.3 where RL left it at 90.0 ([[qwen-3-hybrid-thinking]] Table 21); otherwise, use RL and track pass@k at large k. When reusing an SFT recipe on a different base, sweep the learning rate, because OLMo 2 reports that its bases needed higher learning rates than the Llama 3.1 recipe ([[olmo-2-tulu-recipe]] §5). When synthetic or distilled data lift a public benchmark, add an evaluation set that post-dates the training data or a matched fresh set, because GSM1k found GSM8k-over-GSM1k gaps across the Phi family ([[gsm1k]] §5.1).

## Why this chapter matters for a general-purpose model

A general-purpose model passes through pre-training, mid-training, SFT, preference optimization, and RL, and each stage can raise the capabilities it targets while lowering others. [[ch-33]] followed Tülu 3 and Llama 3, which optimize one generalist. This chapter reads three families that make a different choice at each stage:

1. **Qwen** trains one checkpoint for thinking and non-thinking modes, tool use, and multilingual tasks, then partly reverses the choice: the Qwen3 report fuses thinking and non-thinking behavior into one checkpoint, the Qwen3-2507 release splits it again, and Qwen3.5 serves both modes from one checkpoint.
2. **OLMo** releases every stage (data, scripts, intermediate checkpoints), so stage effects can be read from tables and reproduced from configs.
3. **Phi** relies on constructed training data: synthetic pre-training and mid-training data for Phi-4, and teacher traces from o3-mini for Phi-4-reasoning. It reports both overfitting checks and knowledge losses.

For each family the chapter asks which stage raised which benchmark, which benchmarks fell, and how the lab tested that the gains were general rather than memorized. It ends with an SFT-stage ledger for the case-study models plus Tülu 3, Llama 3, SmolLM3, and DeepSeek-V3, with every unreported cell marked. The companion figure [figures/lab-compare.html](figures/lab-compare.html) lets the reader pick a report table and two stages, plots the per-benchmark change sorted by sign, and shows the ledger as a grid of reported, not-reported, and conflicting cells.

## §1 Reading a report for generality: stage deltas and their noise

**Definition.** A *stage delta* is Δ_b = s_b(after) − s_b(before), the change in score on benchmark b between two checkpoints of the same pipeline. A *specialization cost* is a negative Δ on a benchmark that the stage did not target, observed while targeted benchmarks rise.

**Problem.** A final score cannot show which stage lowered a capability, so a reader who has only final scores cannot decide which stage to change.

**Procedure.**
1. Find tables that report the same benchmarks at several stages: Qwen3 Table 22, Qwen3 Table 21, OLMo 2 Table 16, Olmo 3 Table 26, Phi-4 Table 9, Phi-4-reasoning Table 2, Qwen2.5-1M Table 6.
2. Mark each benchmark as targeted or not targeted by the stage's data and rewards.
3. Compute Δ for every benchmark and sort by sign.
4. Compare |Δ| with run-to-run noise for that benchmark size and sampling protocol.
5. Check whether the benchmark could appear in training data, and whether a fresh or perturbed set exists.

**Noise formula.** For a benchmark of n questions, each scored as the mean accuracy over k samples, and a model with per-question success probability p_i, the sampling standard error of the score is

SE = sqrt( Σ_i p_i(1 − p_i) / (n² k) )

where n is the number of questions, k the samples per question, and p_i the success probability on question i. This counts only sampling noise; differences between training runs (seeds, data order) add to it and are not captured.

**Worked example.** Qwen3 scores AIME with 30 questions and 64 samples per question ([[qwen-3-hybrid-thinking]] §4.6). If all 30 questions had p = 0.82 (close to the Stage 2 score of 83.8), SE = sqrt(30 × 0.82 × 0.18 / (900 × 64)) = 0.0088, or 0.88 points. For a fixed mean score, equal p_i give the largest Σ_i p_i(1 − p_i), so 0.88 is the upper bound of this sampling error. The difference between two independent evaluations has SE = 0.88 × √2 = 1.24 points. The reported thinking-mode change from Stage 2 to Stage 4 is 81.4 − 83.8 = −2.4 points, about 1.9 of these standard errors. One question is worth 100/30 = 3.33 points, so −2.4 points equals 0.72 of a question averaged over samples.

**Evidence on noise.** The Phi-4-reasoning report states that for all models it compared, including its own and the OpenAI and DeepSeek models, two average-of-5 runs "can differ significantly (by up to 5-10 percentage points on AIME)", and it reports AIME 2025 as pass@1 over 50 runs with standard deviations, for example 63.1 (6.3) for Phi-4-reasoning ([[phi-4-reasoning-sft-rl]], arXiv:2504.21318 §1, §5.1, Table 1). **Result (single study).**

**Conditions and limits.** Qwen3 Table 22 prints one score per stage and no variance, and the formula above omits run-to-run training noise, so the −2.4 point change is consistent with a real decline but is not established by the table. The authors' own reading appears in §3.

**Implication.** A stage gate for a general-purpose model needs repeated evaluation on small benchmarks (for example 64 samples per question as in Qwen3, or 50 independent runs as in Phi-4-reasoning), and a list of non-targeted benchmarks that must not fall by more than the measured noise.

## §2 Qwen2.5: verified post-training settings and the Online Merging Optimizer

**Pipeline.** Qwen2.5 open-weight models (0.5B-72B) are pre-trained on 18T tokens, with a final 4,096 → 32,768-token stage and RoPE base 10,000 → 1,000,000 ([[qwen-2.5-recipe]], arXiv:2412.15115v2 §3.1, §3.3). Post-training has three stages:

1. **SFT**: "over 1 million SFT examples", 2 epochs, sequence length 32,768, learning rate "gradually decreased from 7 × 10⁻⁶ to 7 × 10⁻⁷", weight decay 0.1, gradient clipping 1.0 (§4.1). §1 instead says the post-training data "amounts to 1 million examples" across SFT, DPO, and GRPO, so the example count is a conflict within the report. Batch, packing, and loss masking are not reported.
2. **Offline RL**: the SFT model resamples responses for new math, coding, instruction-following, and logical-reasoning queries; responses that pass execution feedback or answer matching are chosen and failures rejected; about 150,000 pairs; DPO for 1 epoch at 7 × 10⁻⁷ with the Online Merging Optimizer (§4.2). DPO β is not reported.
3. **Online RL**: GRPO with 8 responses per query, global batch 2048, 2048 samples per episode (256 queries per episode, derived), queries with higher reward-model score variance processed first (§4.3).

The two-stage long SFT (instructions up to 32,768 tokens, then short mixed with long up to 262,144) belongs to Qwen2.5-Turbo, the proprietary MoE model, not to Qwen2.5-72B-Instruct (§4.4). The open Qwen2.5-7B/14B-Instruct-1M models use the same two-stage SFT and offline RL on pairs of at most 8,192 tokens ([[qwen-long-context-synth]], arXiv:2501.15383v1 §4). Qwen2.5-72B-Instruct scores MMLU-redux 86.8, HumanEval 86.6, MATH 83.1, and IFEval (strict prompt) 84.1; the 86.1 MMLU figure belongs to the base model ([[qwen-2.5]] Tables 2, 6).

Two measurement statements in the report bear on generality. The contamination filter removes a training sequence when its longest common subsequence with a test sequence has length ≥ 13 and ≥ 0.6 of the shorter sequence (§5). The reward-model section states that "a higher score on RM benchmarks does not necessarily correlate with superior performance of the resulting RL model" (§5.2.3), without RL numbers.

**Online Merging Optimizer: definition.** An optimizer for preference training that, at every step, combines the gradient-based update with the parameter difference between the SFT model and the pre-trained model ([[online-merging-optimizer]], arXiv:2405.17931v1 §4).

**Problem.** *Alignment tax* is the loss of pre-training and SFT abilities during RLHF. Merging the RLHF model with its SFT reference once, after training, restores benchmark scores but lowers MT-Bench and AlpacaEval 2.0 (§3).

**Mechanism (OnDARE variant).**
1. Compute the ordinary Adam update Δθ for the preference loss.
2. Keep each entry of Δθ with probability p and zero the rest (random sparsification, no rescaling; §4.2).
3. Do the same for the SFT delta τ_r = θ_r − θ_b.
4. Apply a weighted sum of the two sparse vectors.

θ^(t) = θ^(t−1) + (1 − α)·F_R(Δθ) + α·F_R(τ_r)

where θ^(t) are the policy parameters at step t, Δθ the Adam update, θ_r the SFT reference, θ_b the pre-trained model, F_R a Bernoulli keep-mask with reserve rate p, and α the merging weight ("Larger α introduces stronger regularization", §4.2).

**Worked example.** Take four parameters with Δθ = (0.40, −0.20, 0.10, 0.30), τ_r = (1.0, 0.5, −0.5, 0.0), and an exaggerated α = 0.1 for readability (the paper's grid is 1e−7 to 1e−4). Suppose the mask keeps entries 1, 2, 4 of Δθ and entries 1, 3, 4 of τ_r. The update is 0.9 × (0.40, −0.20, 0, 0.30) + 0.1 × (1.0, 0, −0.5, 0) = (0.46, −0.18, −0.05, 0.27). The plain Adam update for entry 3 was +0.10, against the SFT direction (−0.5); the merged update is −0.05, along it.

**Evidence.** On UltraFeedback DPO, OnDARE changed the benchmark average / MT-Bench / AlpacaEval 2.0 LC relative to AdamW by +0.5 / +0.24 / +0.05 (Qwen1.5-1.8B-Chat), +1.1 / +0.12 / +0.28 (Qwen1.5-7B-Chat), and +1.3 / +0.19 / +1.57 (LLaMa-3-8B-Instruct); an offline linear merge gave +0.2 / −0.03 / −0.96, −0.2 / −0.28 / −1.65, and −1.2 / +0.11 / +0.02 (Table 1). **Result (single study)**, one run per cell.

**Conditions and limits.** Tested at 1.8B-8B parameters. The paper's prose says the benchmark average peaks at α = 5e−7, but its Table 2 shows the peak (41.8) at 5e−6. The Qwen2.5 report names the optimizer without α, p, the variant, or an ablation ([[qwen-2.5]] §4.2), so its effect on Qwen2.5 is not measured. [[ch-30c]] covers merging methods in general.

**Implication.** When preference training lowers benchmark averages relative to SFT, a per-step pull toward the SFT delta is one tested control; the retention benchmarks must be measured, because the benchmark-average gains in Table 1 are 0.5 to 1.3 points.

## §3 Qwen3: hybrid thinking, stage trade-offs, and the 2507 split

**Definition.** A *hybrid thinking* model answers either with an explicit reasoning block or directly, selected by a flag in the prompt, from one set of weights.

**Mechanism.** The flagship models (Qwen3-235B-A22B, Qwen3-32B) use four post-training stages ([[qwen-3-hybrid-thinking]], arXiv:2505.09388v1 §4.1-4.4):
1. **Long-CoT cold start.** Qwen2.5-72B-Instruct removes queries that are hard to verify or that it answers correctly without chain of thought; QwQ-32B generates N candidates; responses are removed for wrong answers, repetition, guesswork, thinking/summary inconsistency, language mixing, or similarity to validation items. The report advises minimizing "both the number of training samples and the training steps". Sizes and hyperparameters are not reported.
2. **Reasoning RL.** 3,995 query-verifier pairs with GRPO; Qwen3-235B-A22B AIME'24 rose from 70.1 to 85.1 over 170 steps.
3. **Thinking-mode fusion.** Continual SFT on the Stage 2 model. Thinking data are rejection-sampled from the Stage 2 model on Stage 1 queries; non-thinking data cover coding, math, instruction following, multilingual tasks, writing, QA, and role-play.
4. **General RL.** Rewards for over 20 tasks: instruction following, format following including the mode flags, preference alignment, agent ability with multi-turn tool use against real environments, and RAG.

The fusion template (Table 9) keeps an empty reasoning block in non-thinking answers:

```
<|im_start|>user                     <|im_start|>user
{query} /think<|im_end|>             {query} /no_think<|im_end|>
<|im_start|>assistant                <|im_start|>assistant
<think>                              <think>
{thinking content}
</think>                             </think>

{response}<|im_end|>                 {response}<|im_end|>
```

When thinking reaches a user-set budget, the thinking is halted and the stop-thinking instruction "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n" is inserted; the report states that this budget behavior "is not explicitly trained but emerges naturally" from fusion (§4.3).

**Evidence of the trade-off (Qwen3-32B, Table 22, one score per stage).**

| Benchmark | Stage 2 (thinking) | Stage 4 (thinking) | Δ | Stage 4 (non-thinking) |
|---|---|---|---|---|
| IFEval strict prompt | 73.0 | 85.0 | +12.0 | 83.2 |
| ToolUse (in-house) | 63.3 | 85.5 | +22.2 | 86.5 |
| CounterFactQA (in-house) | 50.4 | 68.1 | +17.7 | 66.4 |
| BFCL v3 | 69.0 | 70.3 | +1.3 | 63.0 |
| MMLU-Redux | 91.4 | 90.9 | −0.5 | 85.7 |
| GPQA-Diamond | 68.8 | 68.4 | −0.4 | 54.6 |
| AIME'24 | 83.8 | 81.4 | −2.4 | 31.0 |
| LiveCodeBench v5 | 68.4 | 65.7 | −2.7 | 31.3 |

The authors write: "for challenging tasks like AIME'24 and LiveCodeBench, the performance in thinking mode actually decreases after these two training stages. We conjecture this degradation is due to the model being trained on a broader range of general tasks, which may compromise its specialized capabilities in handling complex problems. During the development of Qwen3, we choose to accept this performance trade-off to enhance the model's overall versatility." **Result (single study)** for the scores; **Interpretation** for the cause.

**The split.** On 2025-07-21 Qwen announced: "we decided to stop using hybrid thinking mode. Instead, we'll train Instruct and Thinking models separately so we can get the best quality possible" ([[qwen3-2507-instruct-thinking-split]]). The Qwen3-235B-A22B-Instruct-2507 card states that the model "supports only non-thinking mode", and the Thinking-2507 card that it "supports only thinking mode". Against the original hybrid in non-thinking mode, Instruct-2507 scores AIME25 70.3 versus 24.7 and SimpleQA 54.3 versus 12.2, but the card also lists more "long-tail knowledge coverage", so this is not a controlled test of hybrid versus separate training. The Qwen3.5-397B-A17B card again serves both modes from one checkpoint, thinking by default and disabled with `enable_thinking: False` ([[qwen-3-5-model-card]]). **Open question**: no Qwen source measures the cost of hybrid training with the data held fixed.

**Implication.** A mode switch is a capability to evaluate: report specialist benchmarks in each mode, a mode-following test (Qwen's in-house ThinkFollow rose from 88.7 after Stage 3 to 98.9 after Stage 4), and the size of any thinking-mode loss.

## §4 Qwen3 strong-to-weak distillation

**Definition.** *Off-policy distillation* trains a student on teacher-generated responses with cross-entropy. *On-policy distillation* samples responses from the student and trains the student's token distributions to match the teacher's on those responses.

**Problem.** Running four stages for each of six smaller models (0.6B-14B dense and 30B-A3B) costs development compute; the report states that logit distillation "requir[es] only 1/10 of the GPU hours compared to the four-stage training method" (§4).

**Mechanism** (§4.5):
1. Off-policy phase: teacher outputs in both `/think` and `/no_think` modes become SFT targets, which teach reasoning and mode switching.
2. On-policy phase: "the student model produces responses in either /think or /no_think mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence." The section describes no reward model for this phase.

**Formula.** At each position t of a student sample y ~ π_S(·|x), a token-level KL objective is

L = Σ_t D_KL( π_S(·|x, y_<t) ‖ π_T(·|x, y_<t) ) or Σ_t D_KL( π_T(·|x, y_<t) ‖ π_S(·|x, y_<t) )

where π_S is the student, π_T the teacher, x the prompt, and y_<t the student's own prefix. The report does not state which direction, or whether the full vocabulary or a subset of logits is used.

**Worked example.** At one position over three tokens, the teacher gives (0.7, 0.2, 0.1) and the student (0.4, 0.4, 0.2). Reverse KL = 0.4 ln(0.4/0.7) + 0.4 ln(0.4/0.2) + 0.2 ln(0.2/0.1) = −0.224 + 0.277 + 0.139 = 0.192 nats. Forward KL = 0.7 ln(0.7/0.4) + 0.2 ln(0.2/0.4) + 0.1 ln(0.1/0.2) = 0.392 − 0.139 − 0.069 = 0.184 nats. Either way the student receives a target for all three tokens at this position, whereas RL with a verifier gives one scalar for the whole sampled response.

**Evidence (Qwen3-8B, math and code queries only, both runs from the same off-policy checkpoint; Table 21).** AIME'24 55.0 → 67.6 with RL and → 74.4 with on-policy distillation; AIME'25 42.8 → 55.5 and → 65.5; LiveCodeBench v5 42.0 → 52.9 and → 60.3; MMLU-Redux 86.4 → 86.9 and → 88.3; GPQA-Diamond 55.6 → 61.3 and → 63.3. AIME'24 pass@64 stayed at 90.0 after RL and rose to 93.3 after distillation. GPU hours: 17,920 for RL, 1,800 for distillation (ratio 9.96). **Result (single study).**

**Conditions and limits.** The comparison uses math and code queries; knowledge and chat retention are represented only by MMLU-Redux and GPQA-Diamond. A token-level KL to teacher logits requires the student and teacher to share a vocabulary; all Qwen3 models use the same tokenizer with 151,669 entries (§2). The report does not test a student surpassing its teacher. The RL run's settings are not reported, so the table cannot show whether the RL baseline was tuned. [[ch-35]] and [[ch-35a]] cover where labs insert teacher data and how they select it.

**Implication.** For a family of smaller general models, on-policy distillation from the flagship is the reported lower-cost route. The authors read the pass@64 rise as the student's exploration space expanding (§4.7, **Interpretation**); it was measured on AIME'24 and AIME'25 only.

## §5 Agentic training in the Qwen lineage, with an open reference

**Qwen3 general RL.** Stage 4 includes "Agent Ability", where "the model is allowed to perform complete multi-turn interaction cycles with real environment execution feedback" (§4.4). In Table 22, in-house ToolUse rose 70.4 → 85.5 (thinking) and 73.2 → 86.5 (non-thinking) during Stage 4, and BFCL v3 rose 68.4 → 70.3 and 61.5 → 63.0. The BFCL multi-turn evaluation deploys models at 64k context with YaRN (§4.6).

**Qwen3-Coder (July 2025 blog).** Pre-training uses 7.5T tokens with a 70% code share, described as "preserving general and math abilities"; code RL on automatically test-cased tasks "brought gains to other tasks"; long-horizon agent RL ran on "20,000 independent environments in parallel" ([[qwen3-coder]]). The blog prints no numbers for the general or cross-task claims, so these claims have no controlled evidence in the source (official source, no measurement).

**Qwen3.5 (February 2026 model card).** The card claims "Reinforcement learning scaled across million-agent environments with progressively complex task distributions" without method details ([[qwen-3-5-model-card]]). Its benchmark notes define the evaluation setting: search agents use "a simple context-folding strategy(256k): once the cumulative Tool Response length reaches a preset threshold, earlier Tool Responses are pruned from the history". On BrowseComp the same model scores 69.0 with context folding and 78.6 with the "discard-all" strategy used for DeepSeek-V3.2 and Kimi K2.5. The same checkpoint differs by 9.6 points between the two context policies, so an agent score describes the model together with its context policy (**Interpretation**); [[ch-45c]] covers context management.

**Open reference: Olmo 3 Instruct.** Dolci Instruct SFT contains 227,579 tool-use prompts (Table 30). Its function-calling trajectories include 22.6K Science QA trajectories with real MCP tools (8 functions, 42.3% multi-step), 6.6K Web Search QA (3 functions, 76.1% multi-step), and 200K simulated SimFC trajectories (42.6K functions, 42.3% multi-turn, 23.8% multi-step) (Table 27). The authors report that a unified format (OpenAPI tool definitions, pythonic calls inside XML tags, dedicated special tokens, an environment role) was "crucial for stable and high-quality tool-use behavior" (§5.2.1). Across SFT → DPO → RL for the 7B model, BFCL moved 48.9 → 49.6 → 49.8 and LitQA2 (with tools) 38.0 → 43.3 → 38.2 (Table 26) ([[olmo-3-model-flow]]).

**Implication.** Tool-use gains in these reports come with a fixed scaffold and context policy; a general agent needs the same benchmark under at least two context policies and a check that non-agent benchmarks did not fall during agent RL.

## §6 OLMo 2 and Olmo 3: open stages, released configs, and measured stage effects

**OLMo 2** trains 7B, 13B, and 32B models with a 4,096-token sequence length and no long-context stage (arXiv:2501.00656v3 Table 3), then applies the Tülu 3 recipe ([[olmo-2-tulu-recipe]] §5):
- **SFT.** `tulu-3-sft-olmo-2-mixture` (939,104 prompts) for 7B and 13B; `-0225` (866,138 prompts) for 1B and 32B, which removes synthetic prompts mentioning a date cutoff and keeps math items only where 5 completions reach a majority. Removing multilingual subsets lowered the average by about 0.5 points, so they stayed. Learning rates were swept (7B: 1e-5, 2e-5 chosen, 3e-5; 13B: 5e-6 chosen; 32B: 4e-6 chosen), with 1 seed per configuration and up to 4 for the final one. In Table 17, at 3 epochs and learning rate 4e-6, a sum loss scored 49.76 and a mean loss 48.25 on the development average.
- **DPO.** On-policy responses from development SFT models plus a 20-model pool, rated by GPT-4o; the top-rated completion is chosen and a random lower-rated one rejected.
- **RLVR.** PPO for 7B and 13B on GSM8K, MATH, and constraint prompts; 13B needed two more RLVR passes (GSM8K, then MATH) after its math scores fell below a development model; GRPO for 1B and 32B.

The report notes that "OLMo 2 required significantly higher learning rates compared to the Llama 3.1 training recipe" (§5). **Result (single study).**

Stage effects for OLMo 2 32B (Table 16):

| Stage | AVG | AlpacaEval 2 | GSM8K | IFEval | MATH | Safety | PopQA |
|---|---|---|---|---|---|---|---|
| SFT | 61.7 | 16.9 | 78.4 | 72.4 | 35.9 | 93.8 | 35.4 |
| DPO | 68.8 | 44.1 | 85.7 | 83.8 | 46.8 | 91.9 | 36.4 |
| Instruct (RLVR) | 68.8 | 42.8 | 87.6 | 85.6 | 49.7 | 85.9 | 37.5 |

The average is unchanged by RLVR while safety falls 6.0 points. At 13B, safety falls from 94.3 (SFT) to 89.7 (Instruct).

OLMo 2 also averages mid-training runs. The text says three anneals on different data orders "consistently" match or beat the best single run (Table 14), but for mix E the soup scores GSM* 43.0 against 60.5 for the best single run. The prose summary and the table disagree on one of 24 cells.

**Olmo 3** names every stage ([[olmo-3-model-flow]], arXiv:2512.13961v2): pre-training on Dolma 3 Mix (5.9T tokens) at 8,192 tokens with sliding-window attention on three of every four layers; Dolmino mid-training (100B tokens); Longmino extension to 65,536 tokens (50B tokens for 7B, 100B for 32B) with YaRN on full-attention layers only; then Dolci SFT, DPO, and RL. The released SFT launch script for 7B Think reads (open-instruct@d8a7f1c `scripts/train/olmo3/7b_think_sft.sh` L7-19, [[allenai-olmo3-open-instruct-scripts-recipe]]):

```bash
CHECKPOINT=/weka/.../long-context/olmo25_7b_lc_64k_6T_M100B_..._yarn-fullonly_50B-fb13a737/step11921/
LR=5e-5
python src/scripts/train/sft/OLMo2-7B-sft.py train \
    olmo2.5-6T-LC-sigma-reasoning-mix-decontam-v2-special-tokens-v3-think-FIX \
        $CHECKPOINT \
        ai2/jupiter-cirrascale-2 \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --global_batch_size=1048576 \
```

Dolci Think SFT has 2,268,468 prompts for 7B (Table 17); training uses "document packing instead of padding", 1M-token batches for 7B, two epochs, and 32,768 tokens (App. A.6.1). The checkpoint is chosen by a learning-rate sweep scored on the evaluation suite, then "vibe-test" questions, and the final thinking SFT checkpoint is "a linearly weighted merge of two checkpoints trained with different learning rates" (§4.2; Table 47 lists 1.0e-4 souped with 5.0e-5 for 32B).

Two Olmo 3 findings address the limits of imitation and memorization. First, "further supervised finetuning on thinking traces generated by Qwen3 32B ... outright hurts the performance of Olmo 3 Think SFT"; the lab pairs those completions with worse responses and trains with length-normalized DPO (β = 5 in Table 48 and the released scripts) instead (§4.3, App. A.6.2; open-instruct@00d42c4 `7b_think_dpo.sh` L32-37). Second, RL-Zero trains RLVR directly from the base on data decontaminated against pre-training, mid-training, and evaluation sets, and runs a negative control: training on the same prompts with random binary rewards "does not improve performance on any of our benchmark" suites (§6.2, Fig. 27). The motivation is that on bases with undisclosed data, contamination "makes spurious rewards as effective as true rewards" (§6). A mixed-domain RL-Zero run improved all domains but left "each domain ... under-optimized compared to the single-domain setup" (§6.2).

For Olmo 3 7B Instruct (Table 26, averages of three runs), SFT → DPO → RL moved AIME 2024 6.7 → 23.5 → 44.3, LiveCodeBench v3 20.0 → 18.8 → 29.5, IFEval 81.7 → 82.0 → 85.6, MMLU 67.1 → 69.1 → 69.1, PopQA 16.5 → 20.7 → 14.1, AlpacaEval 2 LC 21.8 → 43.3 → 40.9, and safety 89.5 → 89.9 → 87.6.

**Implication.** Released scripts and stage tables let a reader reproduce a stage and locate where a score changed, and the random-reward control tests whether an RL-Zero gain comes from eliciting memorized evaluation data (the control was run for RL-Zero, not for the Instruct RL stage). The Instruct stage table shows that knowledge recall (PopQA) and safety can fall in the RL stage while reasoning rises.

## §7 Phi-3 and Phi-4: synthetic data share, overfitting checks, and factual recall

**Data.** The Phi-3 report describes two sequential pre-training phases: phase 1 mostly web data, phase 2 "more heavily filtered web data (a subset of phase 1)" combined with synthetic data; it gives no token split or generator models ([[phi-3]], arXiv:2404.14219v4 §2). The Phi-4 report later characterizes phi-3's phase 2 as "primarily synthetic tokens and a much smaller allocation for ultra-filtered and reasoning-heavy web data" (arXiv:2412.08905v1 §3.1). phi-3-mini (3.8B, 3.3T tokens) scores 64.0 on TriviaQA against 82.2 for Mixtral 8x7B, which the authors attribute to limited capacity for factual knowledge ([[phi-3]] §3, §6).

Phi-4 (14B, about 10T tokens) uses "about 400B unweighted tokens" of synthetic data in pre-training and mid-training ([[phi-4-reasoning-sft-rl]], §2.2-§3). Two findings in §3.1 describe the trade-off: at a fixed token budget, 12 epochs over a synthetic subset scored higher on MMLU than 4 epochs plus more fresh web tokens (Fig. 2), and "Models trained only with synthetic data underperformed on the knowledge-heavy benchmarks and demonstrated increased hallucinations". A 13B ablation trained without web data changed MMLU by +0.8, MATH by +4.9, HumanEval by +12.1, and TriviaQA by −14.8 relative to phi-3-medium (Table 3). In mixture ablations "the only benchmark that shows a clear benefit from web data is TQA", and the lab added targeted and knowledge-heavy web data "to improve knowledge benchmarks" although two synthetic-heavier mixes were "marginally better" than the chosen mixture (§3.2, Table 4). **Result (single study).**

**GSM1k: definition.** A set of 1,205 grade-school math problems written by humans without LLM assistance, matched to GSM8k on solve rate, solution steps, and answer magnitude, and withheld from public release ([[gsm1k]], arXiv:2405.00332v4 §1, §3).

**Problem.** A GSM8k score can include memorized test items. A matched fresh set measures the gap directly.

**Formula.** The paper tests each model with a two-proportion Z-test:

z = (p̂₁ − p̂₂) / sqrt( p̂(1 − p̂)(1/n₁ + 1/n₂) ), p̂ = (n₁p̂₁ + n₂p̂₂)/(n₁ + n₂)

where p̂₁ and p̂₂ are accuracies on GSM8k and GSM1k, n₁ and n₂ the numbers of problems, and p̂ the pooled accuracy.

**Worked example.** Phi-3-mini-4k-instruct scores 0.788 on GSM8k and 0.748 on GSM1k (App. F). With n₁ = 1,319 GSM8k test problems ([[emergence-loss-perspective-recipe]] App. B Table 6) and n₂ = 1,205: p̂ = (1,319 × 0.788 + 1,205 × 0.748)/2,524 = 0.769; SE = sqrt(0.769 × 0.231 × (1/1,319 + 1/1,205)) = 0.0168; z = 0.040/0.0168 = 2.38. The paper prints z = 2.385 and p = 0.009; P(Z > 2.38) = 0.0086, so the printed p is the upper-tail value although the appendix calls the test two-tailed.

**Evidence.** "Several families of models, including the Phi and Mistral families of models, show systematic tendencies to perform stronger on GSM8k compared to GSM1k for almost every release and scale" (§5.1); gaps reach 8% (Abstract). Phi-2 drops 6% but still solves over half of GSM1k (§5.3). The GSM8k log-likelihood correlates with the gap (Spearman 0.36, p = 0.03), and the authors conclude that contamination "is likely not the full story" (§5.4). **Result (single study).**

**Conditions and limits.** The verdict for a checkpoint depends on the prompt format in both directions. Phi-3-mini-4k-instruct, the model in the worked example, shows a gap of 0.040 (p = 0.009) with the standard prompt and 0.007 (p = 0.318) with the alternative prompt; Phi-3-mini-128k-instruct shows 0.011 (p = 0.260) with the standard prompt and 0.035 (p = 0.014) with the alternative prompt (App. F). Scores use the LM Evaluation Harness 5-shot format, not the reports' own settings.

**Phi-4's fresh test.** Phi-4 was evaluated on the November 2024 AMC-10/12 contests (78 questions released on or after November 6, 2024, after all training data were collected) at temperature 0.5, averaging 91.8 of 150 (§1.1, App. C; 91.8 is the Fig. 1 bar label). App. C states 10 generations per question, while the Fig. 1 caption states 100 runs, so the sample count is a conflict within the report. Footnote 8 discloses that all three final candidates scored above 89 and that the final model was chosen "before measuring its score but after seeing the scores for the other two candidates". **Result (single study)**. The disclosure matters because candidate scores seen before the final choice make the contest set part of model selection.

**Factual recall and abstention.** Post-training data teach refusal when the model does not know the answer: SFT pairs a question with a refusal where the model was usually wrong, and DPO prefers a correct answer over a refusal where Phi-4 sometimes answered correctly and a refusal over a wrong answer where it sometimes answered incorrectly (§4.4, App. A.1). On SimpleQA the base model answers 90.0% of questions incorrectly; the SFT model answers 38.7% incorrectly and does not attempt 57.5%; the final model answers 3.0% correctly, 15.8% incorrectly, and does not attempt 81.1% (Fig. 6). The simple-evals F1 "gives our base model a higher score than our final model". The report lists factual hallucination, such as invented biographies, as a weakness (§8).

**Implication.** In the Phi-4 ablation, removing web data in favor of synthetic data raised math and code scores and lowered TriviaQA; a general-purpose recipe that uses a large synthetic share needs a knowledge benchmark, a matched or post-dated reasoning set, and abstention counts reported separately from accuracy.

## §8 Phi-4 post-training and Phi-4-reasoning: pivotal tokens, teachable prompts, transfer

**Pivotal Token Search (PTS): definition.** A method that finds tokens in a sampled solution where the estimated probability of a correct final answer changes by at least a threshold p_gap, and builds DPO pairs whose chosen and rejected responses are single tokens after the same prefix ([[phi-4-reasoning-sft-rl]], arXiv:2412.08905v1 §4.3).

**Problem.** In a full-length chosen response, low-probability tokens unrelated to correctness receive gradient, and a harmful token inside a chosen response receives a positive signal (§4.3).

**Mechanism.**
1. For a query Q and sampled solution t₁…t_n, estimate p(success | t₁…t_i) by sampling completions and checking them with an oracle.
2. Recursively split the sequence until each segment's change in success probability is below a threshold p_gap or the segment is one token. The p_gap used for Phi-4 data is not reported; the Fig. 3 illustration marks tokens that change p(success) by ≥ 0.2.
3. A single token whose change is at least p_gap is pivotal; the query plus prefix becomes the DPO prompt, and tokens t_acc and t_rej that raise or lower p(success) become the pair.
4. Keep only questions with 0.2 ≤ p(success) ≤ 0.8.

**Worked example.** After a prefix with p(success) = 0.50, sampling shows that next token "divide" gives 0.85 and "multiply" gives 0.25. The pair is (prefix, "divide") over (prefix, "multiply"). The negative-feedback section below computes how DPO moves probability for such a pair.

**Evidence (Table 9).** From SFT to DPO stage 1 (PTS), GPQA rose 47.3 → 53.6 and MATH 77.1 → 80.5, while IFEval fell 66.2 → 63.0. The final model after judge-guided DPO stage 2 (about 850k GPT-4o-labeled pairs) reached GPQA 56.1 and ArenaHard 75.4 (from 66.5), while DROP fell from 86.1 to 75.5. Running only stage 2 gave GPQA 52.4 and MATH 77.6. The authors conclude that PTS helps reasoning-heavy tasks and judge-guided DPO helps the GPT-4-judged ArenaHard (§4.5). **Result (single study).**

**Phi-4-reasoning (April 2025).** Phi-4-reasoning is SFT only; Phi-4-reasoning-plus adds RL (arXiv:2504.21318v1 §1).
- **Teachable prompts.** The lab targets seeds "at the edge of Phi-4's current abilities". Where no verifiable answer exists, difficulty is estimated by the agreement rate of weaker models (Phi-4 or GPT-4o) with a strong reference model's plurality answer; rubric-based LLM evaluators also score the number and complexity of reasoning steps (§2.1).
- **Teacher.** o3-mini at high effort was a stronger teacher than medium effort; medium effort had a "similar effect to DeepSeek-R1" and was more token-efficient (§3.2).
- **SFT.** Over 1.4 million prompt-response pairs with 8.3B unique tokens; about 16K steps at global batch 32 and 32K context; AdamW, learning rate 1e-5 (best in a grid over [1e-6, 2e-5]), 450 warmup steps, weight decay 1e-4; "2+ passes over reasoning data sources"; 16B tokens trained (§3, §3.1-3.2). Data-mixture weights are epochs per data cluster, tuned per domain and then combined (§3.1).
- **Decontamination.** Against 23 distinct listed benchmarks including AIME-2024, GPQA, LiveCodeBench, SimpleQA, DROP, and ArenaHard; AIME-2025 post-dates the data (§2.2).

**Transfer evidence.** Both models improve by "30 to 60 percentage points" over Phi-4 on TSP, 3SAT, and BA-Calendar, which the report describes as not directly targeted in training (§1). On general benchmarks (Table 2), Phi-4 → Phi-4-reasoning: IFEval Strict 62.3 → 83.4, FlenQA 82.0 → 97.7, ArenaHard 68.1 → 73.3, MMLUPro 71.5 → 74.3. The same table shows Kitab no-context recall 8.2 → 4.9 while no-context precision rose 19.3 → 23.2; the authors describe this as improved precision and degraded recall when the model answers from parametric knowledge (§5.2). Table 2 evaluates the reasoning models at temperature 0.8 and Phi-4 at 0.0, so part of each difference can come from decoding. **Result (single study).**

**Phi-4-reasoning-plus RL reward** (§4.1). For correct answers the accuracy reward falls from 1.0 to 0.5 as length grows past a control length; for incorrect answers it rises from −1.0 to −0.5 as length grows:

R⁺ = 0.5 + 0.25(1 + cos πρ⁺), R⁻ = −0.5 − 0.25(1 + cos πρ⁻), R_final = (8/13)·R_acc + (1/13)·R_rep

where ρ⁺ = min(1, max(L − 25,600, 0)/(31,744 − 25,600)) and ρ⁻ = min(1, L/3,702) are normalized lengths for a response of L tokens, R_acc is R⁺ for a correct answer or R⁻ for an incorrect one (overridden to −0.5 for a missing end token and −1.0 for an invalid thinking block), and R_rep ≤ 0 is a 5-gram repetition penalty. With R_rep = 0, a correct answer of at most 25,600 tokens scores 8/13 = 0.615, a correct answer at 31,744 tokens 0.308, an incorrect answer of at least 3,702 tokens −0.308, and an incorrect answer with L near 0 scores −0.615. Settings: 72,401 math seeds with 64 per iteration, batch 64, learning rate 5e-8, G = 8, KL β = 0.001, entropy coefficient 0.001; the released checkpoint is "the model with the best observed AIME 2024 score, which is the model trained for 90 steps, over only ∼6k examples" (§4.2). AIME 2025 rose from 63.1 (6.3) to 78.0 (4.6) (Table 1). Selecting the step by AIME 2024 and reporting AIME 2024 makes that score part of model selection; AIME 2025 was not used for selection and post-dates the training data (§2.2).

**Implication.** Reasoning SFT on STEM and code transferred to planning and instruction following in this report, while parametric recall fell; a general-purpose distillation recipe needs recall and abstention benchmarks next to the reasoning suite.

## §9 Context length per model

Context length is reported in three different ways: the length used in training, the length reached by inference-time extrapolation, and the length advertised on a model card. [[ch-32b]] covers extension methods and [[ch-32c]] claimed versus effective length.

| Model | Trained length | Inference-time extension | Short-context check reported | Source |
|---|---|---|---|---|
| Qwen2.5 open-weight 7B-72B | 32,768 (final pre-training stage, RoPE base 1M) | YaRN + DCA ×4 to 131,072; RULER-128K for 72B-Instruct 67.0 → 88.4 | scores within 32K unchanged with DCA + YaRN | [[qwen-2.5]] §3.3, Table 16 |
| Qwen2.5-Turbo (API MoE) | 262,144 (four stages, RoPE base 10M; long SFT) | YaRN + DCA to 1M | not reported | [[qwen-2.5-recipe]] §3.3, §4.4 |
| Qwen2.5-7B/14B-Instruct-1M | 262,144 (five stages) | DCA + YaRN to 1M | 14B: GPQA 45.5 → 39.9, LiveCodeBench 42.6 → 38.6, IFEval 81.0 → 84.3 | [[qwen-long-context-synth]] §3-§6, Table 6 |
| Qwen3 4B-235B | 32,768 (long-context stage, hundreds of billions of tokens) | YaRN + DCA ×4 to 128K; card: static YaRN may lower short-text quality | not reported; RULER is lower in thinking mode than in non-thinking mode (8B: 84.4 vs 89.1) | [[qwen-3-hybrid-thinking]] §3.2, Table 23 |
| Qwen3-235B-A22B-2507 | 262,144 native | Instruct: to 1,010,000 with DCA and MInference | not reported | [[qwen3-2507-instruct-thinking-split]] |
| Qwen3-Coder-480B-A35B | 256K native | 1M with YaRN | not reported | [[qwen3-coder]] |
| Qwen3.5-397B-A17B | 262,144 native | to 1,010,000 | card advises ≥128K context to keep thinking quality | [[qwen-3-5-model-card]] |
| OLMo 2 7B/13B/32B | 4,096 | none | — | [[olmo-2-tulu-recipe]] Table 3 |
| Olmo 3 7B/32B | 65,536 (Longmino 50B/100B tokens, 34% long-context data, YaRN on full-attention layers) | not reported | 10B-token extension ablation: a 66% long / 34% short mix lowered an OlmoBaseEval subset by 2.5 points, a 34% long / 66% short mix by 0.8 | [[olmo-3-model-flow]] §3.6, §3.6.3 |
| phi-3-mini | 4K default; 128K variant via LongRope | — | phi-3.5 claims no 4K loss without a table | [[phi-3]] §2, §4 |
| Phi-4 / Phi-4-reasoning | 16K (250B-token mid-training, RoPE base 250K) / 32K (RoPE base doubled in SFT) | — | Phi-4 Table 2, change relative to phi-3-medium for the 4K → 16K checkpoint: MMLU +3.0 → +2.7, GSM8k +2.2 → +1.2, TQA −0.7 → −1.5, HumanEval +7.8 → +9.0; Phi-4-reasoning: not reported | [[phi-4-reasoning-sft-rl]] (arXiv:2412.08905v1 §3, §3.3, Table 2) |
| SmolLM3-3B | 64K (two 50B stages; RoPE θ 2,000,000 in config, 1.5M in blog; then 5M) | YaRN to 128K | RULER fell after reasoning mid-training and APO; recovered by a 0.9/0.1 merge | [[smollm3-training-configs]] |
| Llama 3.1 405B | 128K (six stages, about 800B tokens; 0.1% long SFT data) | — | short-only SFT regressed long context | [[llama-3-recipe]] §3.4.2, §4.3.4 |
| DeepSeek-V3 | 128K (YaRN, 4K → 32K → 128K, 1,000 steps each) | — | not reported | [[deepseek-v3-recipe]] §4.3 |

For SmolLM3, the released 4k→32k config gives 12 × 6 × 1 × 32,768 = 2,359,296 tokens per step over 20,000 steps, or 47.2B tokens, against the blog's 50B (derived). Qwen2.5-1M's offline RL used only pairs of at most 8,192 tokens and still raised LongBench-Chat (7B: 7.32 → 8.08), so in that report long-context alignment improved without long RL data ([[qwen-long-context-synth]] Table 3).

## Negative samples and negative feedback

The cases use three of the four meanings of "negative" defined in [[ch-31a]] and [[ch-43a]]:
1. **Negative marginal value (discarded).** Qwen3's cold-start filters remove six types of bad responses, and queries Qwen2.5-72B-Instruct answers without reasoning (this chapter §3). OLMo 2's `-0225` mixture drops math items without a 5-sample majority. Olmo 3 RL drops groups whose rewards have zero variance ([[allenai-olmo3-open-instruct-scripts]]). No report gives a false-negative rate for its filters.
2. **Negative as content.** Phi-4's SFT data pair questions the model usually got wrong, and unanswerable variants of questions, with refusal targets (Phi-4 §4.4, App. A.1); these are negative as content. Its DPO pairs (refusal over wrong answer, correct answer over refusal) are negative as gradient. SimpleQA "not attempted" moves from 57.5% (SFT) to 81.1% (final).
3. **Negative as conditioning.** None of the case-study reports describes training failures under a control token.
4. **Negative as gradient.** Qwen2.5 DPO rejects responses that fail execution or answer checks (Qwen2.5 §4.2); OLMo 2 rejects a random lower-rated completion; Olmo 3 pairs Qwen3-32B traces with "even worse responses" to increase the quality delta (Olmo 3 §4.3); Phi-4 PTS rejects a single token; Phi-4-reasoning-plus gives wrong answers accuracy rewards in [−1.0, −0.5]. A wrong answer receives a negative group advantage when its reward is below the group mean, for example in a group with correct answers; in a group of only wrong answers, a longer wrong answer can receive a positive advantage (derived from the §4.1-4.2 formulas).

**Mechanism.** For logits z and softmax probabilities p at one position, ∂log p_y/∂z_j = 1[j = y] − p_j. A gradient step that lowers log p_y changes each logit by η(p_j − 1[j = y]). Every other logit rises by an amount proportional to that token's current probability, and the softmax then gives the largest share of the removed mass to the tokens that were already most likely. With p = (0.5 "divide", 0.3 "multiply", 0.2 other) and η = 0.5, pushing down "multiply" changes logits by (+0.25, −0.35, +0.10) and gives p = (0.598, 0.197, 0.206): of the 0.103 removed from "multiply", 0.098 (94%) goes to the most likely alternative. For a PTS pair at the same prefix, a gradient-descent step on the DPO loss changes logit j in proportion to βσ(−m)(1[j = acc] − 1[j = rej]), where m is the DPO margin; the p_j terms of the chosen and rejected tokens cancel. The same step size of 0.5 on the logits gives p = (0.683, 0.151, 0.166): only the pair's logits move, and "other" changes only through normalization. **Interpretation**: PTS localizes the negative signal to one decision, which is one of the controls listed in [[ch-43a]].

**Negative rewards and length.** In Phi-4-reasoning-plus, with G = 8, R_rep = 0, and rewards (0.615, 0.615, 0.308, −0.308, −0.308, −0.308, −0.615, −0.615) from the four cases above, the group mean is −0.077 and the mean-centered advantages are +0.692 (short correct), +0.385 (long correct), −0.231 (long incorrect), and −0.538 (short incorrect) before division by the group standard deviation (0.480, population form). A short wrong answer is pushed down more than a long wrong answer. The report observes that response length correlates with AIME accuracy during RL (Fig. 7c). Olmo 3 found the related effect for truncated responses: masking them out left "overlong negative sequences" untrained and average completion lengths higher, so Olmo 3.1 RL-Zero stopped masking them (App. A.6.4).

**Controls in these reports.** Localize (PTS single tokens); bound the update with a KL term or a reference (DPO β; Phi-4-reasoning-plus KL β = 0.001, printed in the §4.2 objective as a KL to π_θold; the Online Merging Optimizer's pull toward the SFT delta); remove uninformative groups (zero-variance filtering); verify labels with a reference-answer judge to avoid rule-based false negatives (Qwen3 §4.4); test for spurious gains with random rewards (Olmo 3 §6.2).

**Diagnostics.** Log chosen and rejected log-probabilities separately for DPO; split RL statistics by advantage sign; track response length for correct and incorrect groups; report pass@1 and pass@k at large k (Qwen3 Table 21); count abstentions separately from errors (Phi-4 Fig. 6).

**Effect on generality.** Phi-4 post-training with refusal data lowered the SimpleQA incorrect rate from 90.0% (base) to 15.8% (final), while the simple-evals F1 scored the base model above the final model (Fig. 6). The Olmo 3 finding that imitation of Qwen3-32B traces hurt SFT, while the same traces helped as chosen responses against worse ones, is consistent with a contrast carrying a training signal after positive imitation stops helping ([[olmo-3-model-flow]] §4.3; **Interpretation**). No case study measures the share of improvement due to the rejected term.

## Recipe

Values are as printed; "not reported" lists what was checked. SFT rows give, in order: examples or tokens; epochs; peak learning rate and schedule; batch; packing; loss masking; maximum length; checkpoint-selection rule.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5 Instruct, open-weight | 0.5B-72B | SFT | the eight SFT settings | over 1 million examples; 2; 7e-6 → 7e-7 (shape not stated), weight decay 0.1, clip 1.0; not reported; not reported; not reported; 32,768; not reported | arXiv:2412.15115v2 §4.1 ([[qwen-2.5-recipe]]) | conflict (§1: 1 million examples across SFT, DPO, GRPO); others verified 2026-09-14 | no ablation reported |
| Qwen2.5-Turbo (API MoE) | not reported | SFT | long-context stages | stage 1 ≤ 32,768, same data and steps as Qwen2.5; stage 2 short + long ≤ 262,144, ratio not reported; RL on short instructions | §4.4 ([[qwen-2.5-recipe]]) | verified 2026-09-14 | no ablation reported |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | preference | pairs; loss; epochs; LR; optimizer; β | ~150,000; DPO; 1; 7e-7; Online Merging Optimizer (α, p, variant not reported); not reported | §4.2 ([[qwen-2.5-recipe]]) | verified 2026-09-14 | no ablation reported |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | RL | algorithm; responses per query; batch; queries per episode | GRPO; 8; 2048; 256 | §4.3 ([[qwen-2.5-recipe]]) | verified 2026-09-14; 256 derived (2048 ÷ 8) | no ablation reported |
| Qwen3-235B-A22B, Qwen3-32B | flagship | SFT | the eight SFT settings (cold start and thinking-mode fusion) | not reported; cold start "minimize both the number of training samples and the training steps" | arXiv:2505.09388v1 §4.1, §4.3 ([[qwen-3-hybrid-thinking]]) | not reported (checked §4.1-4.7, App. A) | Table 22 stage effects only |
| Qwen3-235B-A22B | 235B-A22B | RL | query-verifier pairs; algorithm; steps | 3,995; GRPO; 170 (AIME'24 70.1 → 85.1) | §4.2 | verified 2026-09-15 | single run |
| Qwen3-8B | 8B | distill-SFT | on-policy logit distillation: start; teachers; objective; GPU hours | off-policy distilled checkpoint; Qwen3-32B or 235B-A22B; KL to teacher logits on student samples; 1,800 (RL: 17,920) | §4.5; §4.7 Table 21 | verified 2026-09-15 (KL direction, prompts, steps not reported) | Table 21: AIME'24 74.4 vs 67.6 |
| OLMo-2-1124-7B-SFT | 7B | SFT | the eight SFT settings | 939,104 prompts; 2; 2e-5 linear, warmup ratio 0.03 (script) or 0.3 (Table 17 caption); 128 sequences; not reported; not reported; 4,096 (script); LR sweep scored by development-suite average (Table 17), 1 seed then up to 4 seeds | arXiv:2501.00656v3 §5, Table 17; open-instruct@098424c `scripts/train/olmo2/finetune_7b.sh` L24-32 ([[open-instruct-allenai-recipes-recipe]]) | verified 2026-09-14 for epochs, LR 2e-5, batch, and length (paper §5 and script agree); conflict for warmup ratio, for the mixture (script `-0225`, paper and YAML `tulu-3-sft-olmo-2-mixture`), and for LR in `olmo2_1124_7b_sft.yaml` L11 (1.0e-5) | §5 sweep 1e-5 / 2e-5 / 3e-5; Table 17: 3 epochs at 4e-6, sum 49.76 vs mean 48.25 |
| OLMo-2-0325-32B-SFT | 32B | SFT | prompts; LR; batch | 866,138; 4e-6; 256 sequences (8 nodes × 8 GPUs × 1 × 4) | §5; `finetune_32b.sh` L8, L24-32 | verified 2026-09-14; batch derived from the script | §5: sweep 1e-6 to 5e-6, one extra seed for 4e-6 |
| OLMo 2 32B Instruct | 32B | RL | algorithm; LR; KL β; samples per prompt | GRPO; 5e-7; 0.1; 16 | arXiv:2501.00656v3 §5 | verified 2026-09-15 | no RL sweep reported for 32B |
| Olmo 3 7B Think SFT | 7B | distill-SFT | the eight SFT settings | 2,268,468 prompts, "Total Tokens" 45.4B (unit not defined); 2; 5e-5; 1,048,576 tokens; document packing; masking stated without the rule; 32,768; LR sweep on evaluation suite, then vibe tests | arXiv:2512.13961v2 §4.2, Tables 17, 47, App. A.6.1; open-instruct@d8a7f1c `7b_think_sft.sh` L7-19 ([[allenai-olmo3-open-instruct-scripts-recipe]]) | verified 2026-09-15 (masking rule not reported) | §4.2 LR sweep; Table 18 per-source ablations on a 100K base mix |
| Olmo 3 32B Think SFT | 32B | distill-SFT | prompts; tokens; LR; batch; checkpoint rule | 2,253,916; 45.2B; 1.0e-4 souped with 5.0e-5; 4M tokens; linear merge of two LRs | Tables 17, 47; A.6.1; §4.2 | verified 2026-09-15 | §4.2 |
| Olmo 3 7B Instruct SFT | 7B | SFT | prompts; tokens; LR; start | 2,152,112; 3.4B; 8.0e-5; Think SFT checkpoint | Tables 30, 47; `7b_instruct_sft.sh` L6-18 | verified 2026-09-15 | §5.5 and Table 29: starting from Think SFT raised benchmark scores with little change in response length |
| Olmo 3 DPO runs (7B/32B) | 7B, 32B | preference | loss; β; epochs; warmup | `dpo_norm`; 5; 1; 0.1 | open-instruct@00d42c4 `7b_instruct_dpo.sh` L49-54, `7b_think_dpo.sh` L32-37, `32b_think_dpo.sh` L45-50; Table 48 | verified 2026-09-14 | A.6.2: LR and dataset size swept |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | the eight SFT settings | 939,344 prompts; 2; 5e-6 linear, warmup 0.03; 128; not stated; not stated; 4,096; not stated | arXiv:2411.15124v5 §4.3, Table 11 ([[tulu-3]]); docs/tulu3.md L35-53 ([[open-instruct-allenai-recipes-recipe]]) | verified 2026-09-14 for the listed values (Table 11 and docs/tulu3.md agree); conflict for loss reduction (§4.3.2 chose sum loss; the pinned command at 098424c runs mean loss) | §4.3.2: sum loss at 5e-6 best; longer than 2 epochs did not help |
| Llama 3.1 405B | 405B | SFT | the eight SFT settings | not printed (Table 7 shares only); not printed; 1e-5 over 8.5K-9K steps; not printed; not stated; prompt tokens masked; not printed; averages of runs with different data or hyperparameters | arXiv:2407.21783v3 §4.1.3, §4.1.5, Table 7 ([[llama-3-recipe]]) | verified 2026-09-14 (unprinted values not reported) | "work well across different rounds and data mixes"; no numbers |
| SmolLM3-3B | 3B | SFT | the eight SFT settings | 1.8B tokens (1B non-reasoning, 0.8B reasoning); 4 ("~8B tokens"; 1.8B × 4 = 7.2B derived); not reported; not reported; BFD packing; loss masked on user turns and tool results; not reported; not reported (the released model is a soup of APO checkpoints merged linearly 0.9/0.1 with a mid-training checkpoint) | huggingface.co/blog/smollm3 "Supervised Finetuning", "Model Merging" ([[smollm3-training-configs]]) | verified 2026-09-15 (LR, batch, length, checkpoint rule not reported) | blog: mixture from reasoning-ratio ablations; no numbers |
| DeepSeek-V3 | 671B (37B active) | distill-SFT | the eight SFT settings | 1.5M instances; 2; cosine 5e-6 → 1e-6; not printed; packed with sample masking; not printed; not printed; not reported | arXiv:2412.19437v2 §5.1 ([[deepseek-v3-recipe]]) | verified 2026-09-14 | Table 9 (on V2.5): R1-style data vs short CoT |
| Phi-4 | 14B | SFT | the eight SFT settings | around 8B tokens; not reported; 1e-6; not reported; not reported; not reported; not reported; not reported | arXiv:2412.08905v1 §4.1 ([[phi-4-reasoning-sft-rl]]) | verified 2026-09-15 | no ablation reported |
| Phi-4 | 14B | preference | round 1; round 2 | PTS pairs on questions with 0.2 ≤ p(success) ≤ 0.8; about 850k GPT-4o-judged pairs | §4.2-4.3 | verified 2026-09-15 | Table 9 stage and ablation columns |
| Phi-4-reasoning | 14B | distill-SFT | the eight SFT settings | over 1.4M pairs, 8.3B unique tokens, 16B trained; "2+ passes", epoch weights per cluster; 1e-5, 450 warmup steps, weight decay 1e-4, AdamW, about 16K steps; global batch 32 (unit not stated); not stated; not stated; 32K; not reported | arXiv:2504.21318v1 §3, §3.1-3.2 | verified 2026-09-15 | §3.1 LR grid [1e-6, 2e-5]; o3-mini high > medium |
| Phi-4-reasoning-plus | 14B | RL | seeds; batch; LR; G; KL β; entropy; max length; checkpoint rule | 72,401 (64 per iteration); 64; 5e-8, cosine warmup 10 steps; 8; 0.001; 0.001; 32K (outputs clipped at 31K); best AIME 2024, step 90 | §4, §4.2 | verified 2026-09-15 | Fig. 7a: no gain after 90 steps |
| phi-3-mini/small/medium | 3.8B-14B | SFT, preference | all settings | not reported | arXiv:2404.14219v4 §2, §5 ([[phi-3]]) | not reported (checked v1 and v4 body and appendices) | — |

**Unit checks.** Phi-4-reasoning: if the global batch of 32 counts sequences of 32,768 tokens, 16,000 steps × 32 × 32,768 = 16.8B token positions, against 16B trained tokens; the report states neither the batch unit nor packing (derived). Tülu 3 8B: 939,344 ÷ 128 = 7,339 steps per epoch, about 14,680 over 2 epochs (derived). Olmo 3 "Total Tokens" does not say whether it counts one pass or both epochs, so 45.4B corresponds to about 43,300 steps (one pass) or about 86,600 steps (both epochs) at 1,048,576 tokens per step (derived).

**Starting point for a small general-purpose run.** For full-parameter SFT of a 7B-8B base on 0.9 to 2.3 million examples, the verified values are 2 epochs (OLMo 2 7B, Olmo 3 7B, Tülu 3 8B; also Qwen2.5 0.5B-72B) with a peak learning rate that depends on the base and the batch: 5e-6 at 128 sequences of up to 4,096 tokens (Tülu 3 8B on Llama 3.1 8B), 2e-5 at 128 sequences of up to 4,096 tokens (OLMo 2 7B), and 5e-5 (Think) or 8e-5 (Instruct) at packed batches of 1,048,576 tokens with 32,768-token sequences (Olmo 3 7B). For a 14B base trained on over 1.4 million long reasoning traces, Phi-4-reasoning used 1e-5 at a global batch of 32 (unit not stated) and 32K context; SmolLM3-3B used 4 epochs on a 1.8B-token set. When the base or the batch differs from these rows, sweep at least three learning rates around the row with the closest base and batch (the OLMo 2 7B sweep size), because OLMo 2 found that its bases needed higher learning rates than the Llama 3.1 recipe (§5). Select on a development suite that includes a knowledge benchmark, report on a separate held-out suite, and evaluate the selected checkpoint in every response mode. Otherwise, reuse the matching row and still run the held-out and per-mode checks.

## Generalization lens

**(a) What increased breadth.**
- Adding non-thinking data and general RL raised Qwen3-32B IFEval by 12.0, ToolUse by 22.2, and CounterFactQA by 17.7 points in thinking mode ([[qwen-3-hybrid-thinking]] Table 22). **Result (single study).**
- On-policy distillation raised Qwen3-8B pass@64 on AIME'24 from 90.0 to 93.3 and MMLU-Redux from 86.4 to 88.3, where RL left pass@64 at 90.0 (Table 21). **Result (single study).**
- STEM and code reasoning SFT raised Phi-4-reasoning by 30-60 points on TSP, 3SAT, and BA-Calendar and IFEval Strict from 62.3 to 83.4 ([[phi-4-reasoning-sft-rl]] §1, Table 2). **Result (single study)**, with a decoding-temperature confound.
- Short-only offline RL raised long-context LongBench-Chat for Qwen2.5-1M models ([[qwen-long-context-synth]] Table 3). **Result (single study).**
- Merging with an earlier checkpoint recovered an ability lost in later stages: SmolLM3's 0.9/0.1 merge of an APO soup with a mid-training checkpoint recovered base RULER up to 128K ([[smollm3-training-configs]]). **Result (single study)**; the blog prints no comparison table.

**(b) What caused narrowing or forgetting.**
- Qwen3-32B thinking-mode AIME'24 −2.4 and LiveCodeBench v5 −2.7 after fusion and general RL; the authors attribute it to broader training (Table 22). **Result (single study)** and **Interpretation**.
- A 13B Phi ablation trained without web data scored 14.8 TriviaQA points below phi-3-medium ([[phi-4-reasoning-sft-rl]] Table 3); phi-3-mini trails Mixtral 8x7B on TriviaQA by 18.2 points ([[phi-3]] §3); Phi-4-reasoning's no-context Kitab recall fell from 8.2 to 4.9 while no-context precision rose from 19.3 to 23.2 (Table 2). The same direction appears in three Phi reports, all from one lab, so the agreement is not independent replication.
- RL lowered Olmo 3 7B Instruct PopQA from 20.7 to 14.1 and LitQA2 from 43.3 to 38.2 ([[olmo-3-model-flow]] Table 26); DPO and RLVR lowered OLMo 2 32B safety from 93.8 to 85.9 ([[olmo-2-tulu-recipe]] Table 16). **Result (single study)** each.
- Judge-guided DPO lowered Phi-4 DROP from 86.1 to 75.5, and PTS DPO lowered IFEval from 66.2 to 63.0 (Table 9). **Result (single study).**
- Qwen2.5-14B-Instruct-1M, which adds long-context pre-training stages and long SFT, differs from Qwen2.5-14B-Instruct on short benchmarks in both directions (GPQA −5.6, IFEval +3.3); the authors describe short-task performance as similar ([[qwen-long-context-synth]] §6.2, Table 6). **Result (single study).**
- Imitation saturation: more SFT on Qwen3-32B traces hurt Olmo 3 Think SFT ([[olmo-3-model-flow]] §4.3). **Result (single study).**

**(c) How to measure it at this stage.**
- Stage tables across modes and capability families, with in-house tests for the new behavior (ThinkFollow, CounterFactQA, ToolUse in Qwen3).
- Post-dated tests (Phi-4 AMC November 2024; AIME 2025 for Phi-4-reasoning) with disclosure of whether they influenced selection.
- Matched fresh sets and a significance test (GSM1k), repeated under two prompt formats.
- Repeated sampling with reported standard deviations (Phi-4-reasoning: 50 runs on AIME 2025).
- A random-reward negative control before interpreting RLVR gains (Olmo 3 RL-Zero).
- Decontamination rules stated with thresholds: 8-gram overlap ≥ 0.5 (Olmo 3 §4.2.1), LCS ≥ 13 tokens and ≥ 0.6 of the shorter sequence (Qwen2.5 §5).
- Abstention reported separately from accuracy (Phi-4 SimpleQA) and pass@k at large k (Qwen3 Table 21).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Attributing a variant's setting to the family (Qwen2.5-Turbo's 262,144-token SFT stage read as Qwen2.5-72B-Instruct's) | A value appears for a checkpoint whose section never prints it | Find the checkpoint name in the sentence at the locus; keep one ledger row per variant |
| Comparing "SFT size" across units (examples, prompts, pairs, unique tokens, trained tokens) | Two recipes differ by a factor near the epoch count | Recompute steps × batch × length and divide by epochs; mark undefined units such as Olmo 3 "Total Tokens" |
| Reading a single-run 1-3 point stage delta on a 30-question benchmark as a regression or gain | The delta is smaller than the run-to-run spread reported for small math sets (up to 5-10 points between two average-of-5 AIME evaluations in Phi-4-reasoning) | Repeat the evaluation (for example 50 runs, as Phi-4-reasoning does for AIME 2025) and report the standard deviation |
| Trusting a prose summary over the table (OLMo 2 soups "consistently" better; mix E GSM* 43.0 vs 60.5) | A table cell contradicts the text | Read every cell of the cited table |
| Treating a fresh test as untouched after it influenced selection | Candidate scores were seen before the final choice (Phi-4 App. C footnote) | Record when each fresh set was first scored relative to model selection |
| Evaluating a hybrid model in one mode | Non-thinking GPQA-Diamond 54.6 against thinking 68.4 for the same Qwen3-32B checkpoint | Report specialist benchmarks in each mode and a mode-following test |
| Enabling static YaRN for all requests | Short-context scores change after adding `rope_scaling` (the Qwen3-8B model card warns that static YaRN may lower quality on shorter texts) | Run the short-context suite with and without the scaling setting, and add `rope_scaling` only for long inputs as the card advises |
| Copying a blog value instead of the released config | Reproduced run differs in RoPE base (SmolLM3: 1.5M in blog, 2,000,000 in config) | Read the config file at a commit and record both values as a conflict |
| Interpreting RLVR gains on a base with undisclosed data as learned reasoning | Gains also appear with random rewards | Run a random-reward control on the same prompts (Olmo 3 §6.2) |
| Scoring hallucination reduction with F1 only | F1 falls while incorrect answers fall (Phi-4 SimpleQA) | Report correct, incorrect, and not-attempted rates separately |

## Check your understanding

1. Qwen3-32B's thinking-mode AIME'24 fell 2.4 points after Stages 3-4. Using the standard-error formula in §1 and Phi-4-reasoning's run-to-run evidence, explain what additional measurement would be needed to call this a regression, and why the authors' conjecture about broader training is an interpretation rather than a result.
2. The Qwen3 report accepted a thinking-mode loss for versatility, Qwen3-2507 split the modes, and Qwen3.5 serves both again from one checkpoint. Which experiment, with data held fixed, would decide whether hybrid training costs specialist quality, and which benchmarks would it need in each mode?
3. Explain why on-policy distillation to teacher logits can raise pass@64 while RL from the same checkpoint does not, using the difference between a per-token target distribution and a per-response scalar reward. Under what condition would this advantage disappear?
4. The 13B synthetic-only ablation in the Phi-4 report gained on MATH and HumanEval but lost 14.8 TriviaQA points, and Phi-4 post-training raised the SimpleQA not-attempted rate from 6.8% (base) to 81.1% (final). Explain why F1 on SimpleQA fell although incorrect answers fell, and which metrics a general-purpose recipe should report instead.
5. In the PTS gradient, the p_j terms cancel, while pushing down one rejected token alone moves 94% of its removed mass to the most likely alternative in the negative-feedback section's example. Explain why the two cases differ and what this implies for full-sequence DPO on long reasoning traces.
6. Olmo 3 found that more SFT on Qwen3-32B traces hurt, but the same traces helped as chosen responses against worse ones. Explain this with the difference between imitation and contrast, and state how you would test it on your own model.
7. Phi-4-reasoning-plus selected its RL checkpoint by AIME 2024 and reports AIME 2024. Explain why this makes AIME 2025 the more informative number, and how the same issue applies to Phi-4's AMC evaluation.
8. OLMo 2 32B's average was unchanged by RLVR while safety fell 6.0 points. Explain why an average can hide this, and write a stage gate that would have flagged it.

## Connections

- **Previous and dependency:** [[ch-33]] Case Studies A: How Tülu 3 and Llama 3 Measured and Protected Broad Capability — the Tülu 3 recipe that OLMo 2 and Olmo 3 reuse, and Llama 3's averaging and long-context SFT share.
- **Next:** [[ch-35]] Distillation in Practice A: Where Labs Insert Teacher Data — extends §4 and §8 on teacher data placement.
- [[ch-35a]] Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters — Qwen3's no-CoT query filter and Phi-4-reasoning's teachable-prompt selection in a wider set of recipes.
- [[ch-30a]] Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control — the measurement method behind the stage deltas in §1.
- [[ch-30c]] Weight Averaging and Model Merging for Generalist Models — the Online Merging Optimizer, OLMo 2 soups, and the SmolLM3 merge.
- [[ch-31a]] Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood — refusal targets and rejected content.
- [[ch-43a]] Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages — derivations behind the negative-feedback section.
- [[ch-32b]] Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression and [[ch-32c]] Claimed versus Effective Context Length and Long-Context Evaluation — detail for §9.
- [[ch-45c]] Context Management for Long-Horizon Agents — context folding and discard-all strategies from §5.
- [[ch-47a]] Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation and [[ch-48]] Contamination Detection and Its Effect on Reported Scores — GSM1k, post-dated contests, and random-reward controls.
- [[ch-51]] Metric Noise, Confidence Intervals, and Go/No-Go Decisions — the statistics in §1 and §7.

## Sources

- [[qwen-2.5]] — Qwen2.5 report (arXiv:2412.15115v2): post-training data and settings, contamination filter, 72B scores, reward-model statement, DCA + YaRN results.
- [[qwen-2.5-recipe]] — ledger rows for Qwen2.5 SFT, DPO, GRPO, and Qwen2.5-Turbo long-context SFT.
- [[qwen-long-context-synth]] — Qwen2.5-1M report (arXiv:2501.15383v1): two-stage long SFT, short-only offline RL, Table 6 short-benchmark changes.
- [[online-merging-optimizer]] — chapter excerpt standing in for the missing card: OnDARE equation, Table 1 and Table 2 results, α discrepancy.
- [[qwen-3]] — Qwen3 library card; not verified, so all Qwen3 values were read in the report and recorded in the excerpt below.
- [[qwen-3-hybrid-thinking]] — chapter excerpt of arXiv:2505.09388v1: four stages, Table 9 template, Tables 21-23, and the Qwen3-8B model card long-context note.
- [[qwen3-2507-instruct-thinking-split]] — chapter excerpt: the July 2025 announcement and the Instruct-2507 and Thinking-2507 model cards.
- [[qwen3-coder]] — chapter excerpt of the Qwen3-Coder blog: 7.5T tokens, code RL, 20,000 parallel environments.
- [[qwen-3-5]] — Qwen3.5 library card; not verified, and its inherited-pipeline, GRPO, and 19× speedup statements are not used.
- [[qwen-3-5-model-card]] — chapter excerpt of the Qwen3.5-397B-A17B model card: mode control, RL claim, context-folding evaluation setting.
- [[olmo-2]] — OLMo 2 library card; not verified (it contains a 32K extension and GPU-hour figures the report lacks), so values come from the excerpt below.
- [[olmo-2-tulu-recipe]] — chapter excerpt of arXiv:2501.00656v3: Table 3 context, SFT mixtures and sweeps, DPO and RLVR, Tables 14 and 16-17.
- [[open-instruct-allenai-recipes-recipe]] — released OLMo 2 and Tülu 3 launch values and their conflicts with the papers.
- [[olmo-3]] — Olmo 3 library card; not verified, so values come from the excerpt below.
- [[olmo-3-model-flow]] — chapter excerpt of arXiv:2512.13961v2: stages, Dolci sizes, SFT settings, Delta Learning, function-calling data, Table 26, RL-Zero controls.
- [[allenai-olmo3-open-instruct-scripts]] and [[allenai-olmo3-open-instruct-scripts-recipe]] — Olmo 3 SFT, DPO, and RL launch scripts with line loci.
- [[phi-3]] — Phi-3 report (arXiv:2404.14219v4): pre-training phases, TriviaQA gap, LongRope context, unreported post-training settings.
- [[phi-4]] — Phi-4 library card; not verified and it merges two reports, so values come from the excerpt below.
- [[phi-4-reasoning-sft-rl]] — chapter excerpt of arXiv:2412.08905v1 and arXiv:2504.21318v1: synthetic-data ablations, Table 2 (4K and 16K checkpoints), PTS, Table 9, refusal data and SimpleQA, AMC, Phi-4-reasoning SFT and RL settings, transfer and variance.
- [[gsm1k]] — chapter excerpt of arXiv:2405.00332v4: GSM1k design, Phi-family findings, App. F rows used in the worked example.
- [[emergence-loss-perspective-recipe]] — GSM8K test-set size (1,319) used in the Z-test example.
- [[tulu-3]] — Tülu 3 report values (Table 11, §4.3) for the SFT ledger; the card is not verified and was not used for numbers.
- [[llama-3-recipe]] — Llama 3.1 405B SFT, averaging, and long-context rows.
- [[smollm-3]] — SmolLM3 library card; not verified, and its SFT details are not used.
- [[smollm3-training-configs]] — chapter excerpt: released context-extension configs and SmolLM3 blog SFT, APO, and merge statements.
- [[deepseek-v3-recipe]] — DeepSeek-V3 SFT and context-extension rows.
