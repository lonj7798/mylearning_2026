<!-- scope: DeepSeek-R1-Zero (GRPO RL on DeepSeek-V3-Base with rule-based rewards, no SFT), the four-stage DeepSeek-R1 pipeline, and the R1-Distill SFT students
     deps: [[deepseek-v3]], [[deepseekmath]]
     see-also: [[grpo]], [[deepseek-r1-recipe]], [[deepseek-r1-followup]], [[deepseek-r1-distill-synth]], [[rlvr-beyond-base-model]]
-->

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Core Insight:** GRPO training on DeepSeek-V3-Base with only rule-based accuracy and format rewards, and no SFT, raised AIME 2024 pass@1 from 15.6% to 77.9% (DeepSeek-R1-Zero, v2 §2.3); the released DeepSeek-R1 adds a cold-start SFT on thousands of examples, an ~800K-sample SFT stage, and a mixed-reward RL stage to fix poor readability, language mixing, and weak general-task scores (§3, Table 3).
- **Guideline:** When the target is a general-purpose assistant rather than a reasoning-only model, follow reasoning RL with SFT that mixes reasoning and non-reasoning data and a final RL stage that adds preference rewards, because R1-Zero scores 46.6 on IF-Eval and 24.7 on AlpacaEval 2.0 while the final R1 scores 83.3 and 87.6 (Table 3); when the target is a 32B-scale reasoning model and a stronger teacher exists, SFT on teacher samples scored higher than large-scale RL from the base (Table 16).
- **Authors:** DeepSeek-AI (core contributors Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, et al.)
- **Year:** 2025 (arXiv v1 2025-01; v2 2026-01; peer-reviewed version "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning", Nature 645, 633–638, published 2025-09)
- **URL:** https://arxiv.org/abs/2501.12948 (Nature: https://doi.org/10.1038/s41586-025-09422-z)
- **Source type:** official technical report
- **Relevant topics:** GRPO, rule-based rewards, R1-Zero, cold-start SFT, rejection-sampling SFT, language consistency reward, reward-model hacking, distillation, test-time compute
- **Recipe ledger:** [[deepseek-r1-recipe]] (`model-reports/deepseek-r1-recipe.md`)

## Abstract
General reasoning remains hard for LLMs, and prior gains from chain-of-thought prompting and post-training depend on human-annotated demonstrations. The paper shows that reasoning ability can be incentivized through pure reinforcement learning without human-labeled reasoning trajectories. The RL framework produces emergent reasoning patterns such as self-reflection, verification, and dynamic strategy adaptation. The trained model performs better on verifiable tasks (mathematics, coding competitions, STEM) than counterparts trained by supervised learning on human demonstrations. The reasoning patterns of the large models can be used to improve the reasoning of smaller models (v2 Abstract).

## Key Contributions
- DeepSeek-R1-Zero: RL with GRPO applied directly to DeepSeek-V3-Base, with no SFT phase, based on the hypothesis that human-defined reasoning patterns may limit exploration (§1, §2). v1 describes it as "the first open research to validate that reasoning capabilities of LLMs can be incentivized purely through RL, without the need for SFT" (v1 §1.1); v2 drops this sentence.
- Rule-based rewards only for reasoning (accuracy + format, equal weight); neural outcome and process reward models are not used because they are susceptible to reward hacking (§2.2).
- DeepSeek-R1: a four-stage pipeline (cold-start SFT → reasoning RL → rejection-sampling SFT restarted from V3-Base → RL on mixed prompts with rule and preference rewards) with intermediate checkpoints Dev1–Dev3 (§3, Fig. 2).
- Six SFT-only distilled models (1.5B–70B) trained on the ~800K R1 SFT set, and a 32B comparison of distillation against RL from the base (Supp. F, Tables 15–16).
- v2 adds training costs, data statistics, hyper-parameters, reward-hacking and language-reward analyses, and a safety report (Supp. B.3–B.6, D.3).

## Key Figures/Tables to Study
- Fig. 1 (R1-Zero AIME accuracy and response length vs step); Table 2 ("aha moment" sample); Fig. 9 (reflective-word counts, C.2).
- Fig. 2 (pipeline) and Table 3 (benchmarks for R1-Zero, Dev1, Dev2, Dev3, R1).
- Table 5 (SFT data by domain), Table 6 (distillation bases and LRs), Table 7 (training cost in H800 GPU hours).
- Table 8 (R1 vs Claude-3.5-Sonnet, GPT-4o, DeepSeek-V3, o1-mini, o1-1217); Table 12 (V3-Base, V3, R1-Zero, R1).
- Tables 15–17 (distilled models; distillation vs RL); Fig. 6 (reward hacking); Fig. 7 (language reward ablation).

## Technical Details
**GRPO (§2.1, A.3).** For each question, sample G outputs from the old policy; advantage A_i = (r_i − mean(r)) / std(r) over the group (Eq. 3). The objective uses a PPO-style clipped ratio with ε and subtracts β·D_KL(π_θ‖π_ref), where the KL estimator is added to the loss rather than to the per-token reward (Eq. 1–2, A.3). The reference model is replaced with the latest policy every 400 steps (§2.1).

**R1-Zero (§2).** Template: the assistant reasons inside `<think>...</think>` and answers inside `<answer>...</answer>` (Table 1). Accuracy reward: rule-based answer check (math answer in a specified format such as a box; code checked by a compiler against test cases); format reward: reasoning enclosed in think tags (§2.2). Settings: LR 3e-6, KL coefficient 0.001, temperature 1, 16 outputs per question, max length 32,768 tokens before step 8.2k and 65,536 after, 32 questions per step (batch 512), 10,400 steps = 1.6 epochs (§2.1). AIME 2024 pass@1 rises from 15.6% to 77.9%, and to 86.7% with self-consistency decoding (§2.3). v1 reported 71.0% pass@1 and 86.7% cons@64 (v1 §2.2.4, Table 2). Reflective-word counts rise 5- to 7-fold; "wait" is nearly absent early, occasional between steps 4000–7000, and spikes after step 8000 (C.2).

**R1 pipeline (§3, B.3, B.4).**
1. Cold-start SFT (Dev1): "thousands" of long-CoT examples. R1-Zero outputs sampled at temperature 1.0 are filtered for correct answers (sympy for math) and readability (repetition detection, language-mixing filter), refined by DeepSeek-V3, and verified by human annotators (B.3.2). SFT on DeepSeek-V3-Base, 2–3 epochs, cosine LR 5×10⁻⁵ → 5×10⁻⁶, 32,768-token context, batch 128 (B.4.2).
2. Reasoning RL (Dev2): LR 3e-6, KL coefficient 0.001, clip ε = 10, temperature 1, 16 outputs per question, max length 32,768, 32 questions per step (§3.2.1). A language consistency reward (share of target-language words in the CoT) is added to the final reward (Eq. 7).
3. Rejection-sampling SFT (Dev3): about 600k reasoning samples drawn from the stage-1 RL checkpoint, keeping only correct responses; some items are judged by DeepSeek-V3 against the reference answer (Listing 4); CoT with mixed languages, long paragraphs, or code blocks is removed. About 200k non-reasoning samples reuse DeepSeek-V3 SFT data plus software-engineering data (B.3.3). Table 5 totals 804,745 samples, of which 177,812 are General. This SFT starts again from DeepSeek-V3-Base (Fig. 2; B.4.2).
4. Mixed RL (R1): reward = rule reward (reasoning) + reward-model and format reward (general) + language reward (Eq. 8–10); temperature 0.7; 1,700 steps, with general data and preference rewards only in the final 400 steps (§3.2.2).

**Reward models (§3.1).** Helpful RM: 66,000 preference pairs labeled by DeepSeek-V3 (four judgments per pair with positions randomized; kept if the score difference exceeds 1; chosen and rejected lengths balanced); batch 256, LR 6e-6, 1 epoch, max length 8,192. Safety RM: 106,000 prompts labeled safe/unsafe, point-wise loss.

**Compute (B.4.4, Table 7).** R1-Zero: 64×8 H800 GPUs for about 198 hours, 101K GPU hours. R1: the same GPUs for about 80 hours, 41K GPU hours. SFT data creation: 5K GPU hours. Total 147K GPU hours ($294K at an assumed $2 per GPU hour).

**Evaluation (D.1).** Temperature 0.6, top-p 0.95, pass@1 averaged over k samples (k = 64 for AIME and GPQA, 16 for MATH and Codeforces, 8 for LiveCodeBench), max 32,768 output tokens. R1 vs o1-1217: AIME 2024 79.8 vs 79.2, MATH-500 97.3 vs 96.4, Codeforces rating 2029 vs 2061, GPQA Diamond 71.5 vs 75.7; R1 MMLU 90.8 (Table 8).

## Recipe ledger
The full ledger (45 rows: R1-Zero, each R1 stage, reward models, six students, Qwen RL baselines, evaluation) is in [[deepseek-r1-recipe]]. It records one conflict: second-stage SFT epochs are "two epochs" in v1 §2.3.3 and "2–3 epochs" in v2 B.4.2.

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
**Generality.**
- R1-Zero's rule-based RL "is narrowly focused on reasoning tasks, resulting in limited performance in broader areas such as writing and open-domain question answering" (§1). Table 3: R1-Zero IF-Eval 46.6, AlpacaEval 2.0 24.7, ArenaHard 53.6, Aider-Polyglot 12.2.
- Cold start (Dev1) raises IF-Eval to 71.7 and AlpacaEval 2.0 to 50.1 but lowers AIME 2024 from 77.9 to 59.0 and CNMO 2024 from 88.1 to 58.0; the authors attribute the drop to the small cold-start set (§4, Table 3).
- Reasoning RL (Dev2) improves reasoning benchmarks; AlpacaEval 2.0 moves only from 50.1 to 55.8 (§4, Table 3). Adding non-reasoning and code-engineering SFT data (Dev3) raises AlpacaEval 2.0 to 62.1 and Aider-Polyglot from 25.6 to 44.8 (§4).
- Mixed RL raises AlpacaEval 2.0 from 62.1 to 87.6 and ArenaHard from 75.6 to 92.3, with marginal math and code changes (§4, Table 3). R1-Zero stays higher than R1 on GPQA Diamond (75.8 vs 71.5) and CNMO 2024 (88.1 vs 78.8) (Table 3).
- Against DeepSeek-V3, R1 is higher on MMLU-Pro (84.0 vs 75.9) and FRAMES (82.5 vs 73.3) and lower on IF-Eval (83.3 vs 86.1) and C-SimpleQA (63.7 vs 68.0) (Table 12). MMLU-Pro gains appear in all categories, including non-STEM ones (E.1).
- Post-training-date tests: AIME 2025 11.3/15 for R1 vs 12.0/15 for o1-1217; AMC 12 2024 143.7/150 (Table 13, E.2). The 10-gram decontamination cannot catch paraphrases, so pre-2024 benchmarks may be contaminated (D.1).
- RL from a base failed to improve AIME with 7B dense and 16B MoE bases (repetition as length grew) and worked with 32B dense, 230B MoE, and 671B MoE bases (G.1). Most SFT data is single-turn (Table 5, avg rounds 1.0), which the authors say may limit multi-turn ability (B.3.3).

**Negative feedback.** Rejection sampling discards incorrect responses (B.3.3). GRPO's group-normalized advantage is negative for responses below the group mean (Eq. 3). Code test cases are selected using incorrect submissions so that they separate correct from incorrect programs (B.3.2). With the helpful reward model, reward rises while Codeforces performance falls (B.5, Fig. 6), which is why preference rewards are limited to the last 400 steps (§3.2.2). PRMs are listed as an unsuccessful attempt: fine-grained steps are hard to define, step correctness is hard to label, a model-based PRM leads to reward hacking, and the benefit was limited relative to the added compute (G.2). LLM judging against a reference answer is described as robust for short answers and of limited use for open-ended generation (G.1).

**Long context.** Raising R1-Zero's max length from 32,768 to 65,536 tokens at step 8.2k coincides with a jump in performance and length (§2.1). On 366 problems from 2024 competitions, R1 solves 61.8% with 8,793 thinking tokens on average, under 7,000 for easy and over 18,000 for the hardest problems; GPT-4o 0513 solves 24.7% with 711 tokens (E.4). R1 AIME 2024 pass@64 is 90.0% vs pass@1 79.8% (E.4). The authors report overthinking on simple questions (§6).

**Agentic training.** R1 cannot use tools such as search engines or calculators, and its structured output is weaker than existing models (§6). Large-scale RL was not applied extensively to software engineering because evaluation is slow (§6); RL data includes 8k bug-fixing problems from GitHub issues (B.3.1), and SFT data includes program repair and front-end development (B.3.3).

**Distillation.** Students are fine-tuned with SFT only on the 800k set; bases are Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, Llama-3.3-70B-Instruct (Table 6). Distill-Qwen-1.5B scores AIME 2024 28.9 and MATH-500 83.9, above GPT-4o-0513 (9.3, 74.6) (Table 15). At 32B, RL from Qwen2.5-32B-Base for over 10K steps reaches AIME 2024 47.0, MATH-500 91.6, GPQA 55.0, LiveCodeBench 40.2, while Distill-Qwen-32B reaches 72.6, 94.3, 62.1, 57.2 (Table 16, F.1). The authors conclude that distillation is economical but that exceeding current limits may still require stronger bases and larger-scale RL (F.1, Interpretation). v1 states that RL on the distilled models gives further gains, without numbers (v1 §3.2).

## Connections
- [[deepseek-v3]] — base model (DeepSeek-V3-Base) and source of the reused non-reasoning SFT data (A.1, B.3.3).
- [[deepseekmath]], [[grpo]] — origin of GRPO (Shao et al., 2024), used in every RL stage.
- [[let-verify]] — process reward models, listed by R1 as an unsuccessful attempt (G.2).
- [[rlvr-tulu3]] — another RL recipe with verifiable rewards, for comparison.
- [[reward-model-overoptimization]] — background for the helpful-RM hacking in B.5.
- [[dr-grpo]], [[rlvr-beyond-base-model]], [[entropy-mechanism-llm-rl]] — later analyses of R1-Zero-style RL.
- [[deepseek-r1-followup]], [[deepseek-r1-distill-synth]] — R1 updates and use of R1 outputs as distillation data.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2501.12948 (v2, 2026-01-04, 86 pages; v1 2025-01-22 for version differences); Nature record https://doi.org/10.1038/s41586-025-09422-z for venue and date.
- Corrections to the previous card version:
  - "R1 adds a short cold-start SFT on ~800k examples" and "R1 uses ~800K curated examples" → the cold start uses "thousands" of examples (§3, B.3.2); ~800K (804,745) is the later rejection-sampling SFT set (B.3.3, Table 5).
  - Guideline "skip the reasoning-SFT bootstrap" → R1 itself uses cold-start SFT for readability and user experience (B.3.2); replaced with a guideline supported by Table 3 and Table 16.
  - "800K reasoning traces from R1 used to SFT Qwen-2.5 (1.5B/7B/14B/32B) and Llama-3 (8B/70B)" → the set includes 177,812 General samples (Table 5); bases are Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, Llama-3.3-70B-Instruct (Table 6).
  - "filter via V3 judge" → DeepSeek-V3 judges only part of the added reasoning items against a reference; separate filters remove mixed-language, long-paragraph, and code-block CoT; only correct responses are kept (B.3.3).
  - Pipeline "Base -> cold-start SFT -> Reasoning RL -> Rejection Sampling SFT -> Alignment RL" omitted that the ~800K SFT restarts from DeepSeek-V3-Base, not from the RL checkpoint (Fig. 2, B.4.2).
  - "AIME 2024 pass@1 climbs from 15.6% to 71%" and "R1-Zero 71.0%" → 77.9% in v2 (§2.3, Table 3); 71.0% is the v1 value.
  - "accuracy (sympy-verified final answer)" → sympy is named for cold-start data filtering (B.3.2); the R1-Zero reward is described as rule-based answer matching and compiler test cases (§2.2).
  - "clip ratio 10 (intentionally loose — DeepSeek argues tight clipping destroys exploration)" → ε = 10 is given for R1 stage 1; the stated reason is that a lower value truncates gradients for many tokens and a higher value may cause instability (§3.2.1).
  - "Stage-2 Alignment RL with helpfulness + harmlessness preference rewards" → rule, RM, format, and language rewards combined; temperature 0.7; 1,700 steps with preference rewards in the final 400 (§3.2.2).
  - "R1 post-training compute not explicitly disclosed" → B.4.4 and Table 7 give 101K, 41K, and 5K GPU hours.
  - "Codeforces Elo 2029" → the paper reports "Rating" 2029 and percentile 96.3 (Table 8).
- Removed as unsupported by the source: "2.788M H800-hours" (not in this paper; see [[deepseek-v3]]); "~200 contributors"; "believed small vs pretraining"; the connection claims that Tülu 3 RLVR uses PPO and that stage 2 is an analog of Constitutional AI.
- Not reported by the source: exact cold-start example count; step count of the first R1 RL stage; clip ε for R1-Zero; SFT batch-size units.
