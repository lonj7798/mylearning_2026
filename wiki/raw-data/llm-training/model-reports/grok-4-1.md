<!-- scope: xAI Grok 4.1 launch post and companion model card (2025-11-17); reasoning models as reward models for non-verifiable signals, live-traffic pairwise evaluation, safety and propensity evaluations
     see-also: [[grok-3]], [[gemini-2.5-deep-research]], [[qwen-3]], [[llama-3]], [[constitutional-ai]]
-->

# Grok 4.1
- **Core Insight:** xAI states that it applied the Grok 4 large-scale RL infrastructure to style, personality, helpfulness, and alignment, using "frontier agentic reasoning models as reward models" for these non-verifiable signals; in blind pairwise evaluations on live traffic (Nov 1–14, 2025) Grok 4.1 was preferred 64.78% of the time over the previous production model (post, intro and "Silent Rollout").
- **Guideline:** When a post-training target has no verifier (style, personality, helpfulness), this source reports scoring responses with a reasoning model used as a reward model and checking the result with blind pairwise comparison on live traffic; the source gives no ablation against a trained scalar reward model, so treat it as reported practice, not a measured comparison.
- **Authors:** xAI (official post and model card)
- **Year:** 2025 (post and model card both dated 2025-11-17)
- **URL:** https://x.ai/news/grok-4-1 (companion: https://data.x.ai/2025-11-17-grok-4-1-model-card.pdf, "Grok 4.1 Model Card")
- **Source type:** official blog, with a companion official model card
- **Relevant topics:** reasoning model as reward model, non-verifiable rewards, live-traffic pairwise evaluation, hallucination rate, refusal training, sycophancy, honesty

## Summary
The post announces Grok 4.1 in a thinking configuration and a non-thinking configuration. xAI states that it used the same large-scale RL infrastructure as Grok 4 to optimize style, personality, helpfulness, and alignment, and that it developed methods that use frontier agentic reasoning models as reward models to evaluate and iterate on responses at scale. Preliminary builds were served to a growing share of production traffic from Nov 1 to Nov 14, 2025, with continuous blind pairwise evaluations; Grok 4.1 was preferred 64.78% of the time. The post also reports LMArena Elo, EQ-Bench3 and Creative Writing v3 results, and a post-training focus on reducing factual hallucinations for information-seeking prompts. The model card describes the training stages at category level (pre-training, mid-training, SFT, RL on human feedback, verifiable rewards, and model-based graders) and reports refusal, jailbreak, honesty, sycophancy, and dual-use evaluations. Neither document gives model size, data amounts, an RL algorithm, or hyperparameters.

## Key Contributions
- Reasoning models as reward models for non-verifiable reward signals (post, intro).
- Release evaluation by silent rollout with blind pairwise comparison on production traffic across grok.com, X, and mobile apps (post, "Silent Rollout").
- Hallucination-rate evaluation on a stratified sample of production information-seeking queries, plus FActScore (post, "Reduced Hallucinations").
- Category-level disclosure of the full training pipeline and of safety training by demonstrations and input filters (model card §2.1.1, §3.1).
- A correction to earlier xAI refusal numbers, which had evaluated English prompts only (model card §2.1.2).

## Key Figures/Tables to Study
- Model card Table 1 (refusal, jailbreak, AgentHarm, AgentDojo) and Table 3 (MASK dishonesty rate, sycophancy rate, compared with Grok 4).
- Post charts for hallucination rate, FActScore, EQ-Bench3, and Creative Writing v3; their numeric values are not present in the page text.

## Technical Details

### Training (as disclosed)
- Configurations: Grok 4.1 Non-Thinking (NT) responds directly; Grok 4.1 Thinking (T) reasons before responding (model card §1).
- Pre-training data: publicly available Internet data, data produced by third parties, data from users or contractors, and internally generated data; filtering includes de-duplication and classification (model card §3.1).
- Mid-training: "targeted mid-training to improve specific knowledge and capabilities" (model card §3.1).
- Post-training: "a combination of supervised finetuning and reinforcement learning on human feedback, verifiable rewards, and model-based graders for safety training and for specific capabilities" (model card §3.1).
- RL for style and personality: "the same large scale reinforcement learning infrastructure that powered Grok 4", applied to style, personality, helpfulness, and alignment; "to optimize these non-verifiable reward signals, we developed new methods that let us use frontier agentic reasoning models as reward models to autonomously evaluate and iterate on responses at scale" (post, intro). The reward models' identity, size, and tools are not stated.
- Refusal training: demonstrations of appropriate responses to both benign and harmful queries; input filters for bioweapons, chemical weapons, self-harm, and CSAM, trained on synthetic and production data, with Grok used to apply adversarial attacks (model card §2.1.1).
- Honesty and sycophancy: the model card states that training the model to be honest reduces deception and that training it to be less sycophantic reduces sycophancy; both mitigations are used in Grok 4.1 (§2.2.1, §2.2.2). No before-and-after numbers are given.

### Evaluation (post)
- Silent rollout Nov 1–14, 2025: preferred 64.78% of the time over the previous production model; sample size not reported ("Silent Rollout").
- LMArena Text Arena: Grok 4.1 Thinking (code name quasarflux) #1 at 1483 Elo, 31 points above the highest non-xAI model; non-reasoning mode (code name tensor) #2 at 1465 Elo; Grok 4 had overall rank #33 ("State-of-the-Art General Capability").
- EQ-Bench3: 45 roleplay scenarios, most pre-written prompts of 3 turns; official repository, default sampling, judge Claude Sonnet 3.7, no system prompt ("Emotional Intelligence").
- Creative Writing v3: 32 prompts across 3 iterations, rubric and normalized Elo; xAI states it is working with the benchmark authors to add both results to the leaderboard ("Creative Writing").
- Hallucination rate: "macro-average of percentage of atomic claims with major/minor errors over model responses", measured on the non-reasoning model with web search tools; FActScore has 500 biography questions ("Reduced Hallucinations").

### Evaluation (model card)
| Evaluation (metric) | Grok 4 | Grok 4.1 T | Grok 4.1 NT | Locus |
|---|---|---|---|---|
| Refusals (answer rate) | – | 0.07 | 0.05 | Table 1 |
| + User jailbreak / + System jailbreak (answer rate) | – | 0.02 / 0.02 | 0.00 / 0.00 | Table 1 |
| AgentHarm (answer rate) | – | 0.14 | 0.04 | Table 1 |
| AgentDojo (attack success rate) | – | 0.05 | 0.01 | Table 1 |
| MASK (dishonesty rate) | 0.43 | 0.49 | 0.46 | Table 3 |
| Sycophancy (sycophancy rate) | 0.07 | 0.19 | 0.23 | Table 3 |

- Input filter false negative rate: restricted biology 0.03 (0.20 with prompt injection); restricted chemistry 0.00 (0.12 with prompt injection) (Table 2).
- The refusal set covers English, Spanish, Chinese, Japanese, Arabic, and Russian and has several thousand prompts; earlier model cards had evaluated English prompts only because of a settings error, so the new numbers are not comparable to them (§2.1.2).
- The §2.2.1 text points to "Table 4" for MASK results; the MASK numbers appear in the table captioned Table 3.
- Dual-use evaluations (WMDP, VCT, BioLP-Bench, ProtocolQA, FigQA, CloningScenarios, CyBench, MakeMeSay) are reported for Grok 4 and Grok 4.1 T with safeguards removed (§2.3.1, Table 4).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Grok 4.1 (T and NT, not separated) | not reported | pretrain-stable | data sources and filtering | Internet, third-party, user/contractor, internally generated data; de-duplication and classification; token count not reported | model card (2025-11-17) §3.1 | verified 2026-09-14 | no ablation reported |
| Grok 4.1 | not reported | mid-train | purpose | "targeted mid-training to improve specific knowledge and capabilities"; data and tokens not reported | model card §3.1 | verified 2026-09-14 | no ablation reported |
| Grok 4.1 | not reported | SFT | safety data type | demonstrations of appropriate responses to benign and harmful queries; size not reported | model card §2.1.1 | verified 2026-09-14 | no ablation reported |
| Grok 4.1 | not reported | RL | reward sources | human feedback, verifiable rewards, model-based graders | model card §3.1 | verified 2026-09-14 | no ablation reported |
| Grok 4.1 | not reported | reward-model | reward model for style, personality, helpfulness, alignment | "frontier agentic reasoning models as reward models"; identity and size not reported | x.ai/news/grok-4-1, intro | verified 2026-09-14 | no ablation reported |
| Grok 4.1 | not reported | RL | infrastructure | same large-scale RL infrastructure as Grok 4 | post, intro | verified 2026-09-14 | no ablation reported |
| Grok 4.1 | not reported | RL | algorithm, KL, clip ε, LR, batch, samples per prompt, steps, max response length | not reported | checked: full post text, model card §1–§3 | not reported | not applicable |
| Grok 4.1 | not reported | eval-gate | release evaluation | silent rollout Nov 1–14, 2025; blind pairwise on live traffic; 64.78% preferred | post, "Silent Rollout" | verified 2026-09-14 | selection rule not stated |

## Findings relevant to generality, negative feedback, agentic training
- **Generality.** The post states that Grok 4.1 retains the intelligence and reliability of its predecessors, but neither document reports a reasoning or knowledge benchmark comparison with Grok 4 beyond the dual-use evaluations in Table 4. Breadth is measured by LMArena rank and live-traffic preference.
- **Side effects of post-training.** Compared with Grok 4, the sycophancy rate rose from 0.07 to 0.19 (T) and 0.23 (NT), and the MASK dishonesty rate from 0.43 to 0.49 (T) and 0.46 (NT) (Table 3). The model card does not attribute these changes to a specific training stage. Status: Result (single study); the cause is an Open question.
- **Negative feedback.** Refusal training uses demonstrations of appropriate responses to harmful queries, which is negative as content (§2.1.1). The policy aims to refuse without "over-refusing sensitive or controversial queries", but no over-refusal rate is reported.
- **Measurement error.** Earlier xAI refusal numbers evaluated English prompts only (§2.1.2).
- **Agentic.** AgentHarm answer rate is 0.14 for T and 0.04 for NT; AgentDojo attack success rate is 0.05 and 0.01 (Table 1). The post does not say which tools the "agentic" reward models use.

## Connections
- [[grok-3]] — earlier xAI release post, with no training-stage description.
- [[gemini-2.5-deep-research]] — another frontier report on reasoning-oriented post-training.
- [[qwen-3]] — General RL uses rule-based rewards, a Qwen2.5-72B-Instruct judge with reference answers, and a trained reward model across over 20 tasks (arXiv:2505.09388 §4.4).
- [[llama-3]] — uses DPO and reports that PPO was explored but needed more compute and performed worse (arXiv:2407.21783 §4.1.4).
- [[constitutional-ai]] — uses AI feedback (RLAIF) as the preference signal.

## Verification
- Checked on 2026-09-14 against: https://x.ai/news/grok-4-1 (page dated Nov 17, 2025) and https://data.x.ai/2025-11-17-grok-4-1-model-card.pdf (6 pages).
- Corrections to the previous card version:
  - "post-training-only upgrade of Grok 4 — xAI re-used the Grok 4 base" and "no new pretraining" → the post says the Grok 4 RL infrastructure was reused; the model card §3.1 describes pre-training and mid-training for Grok 4.1 and does not say the base is shared with Grok 4.
  - "Reasoning-model-as-RM shifted from secondary tool to primary reward signal" → the post describes reasoning reward models for non-verifiable signals; the model card lists human feedback, verifiable rewards, and model-based graders (§3.1).
  - "Hallucination reduction ... as an explicit RL objective" → the post says post-training focused on reducing hallucinations; it does not name the objective.
  - "Silent rollout ... replaces offline static evals" → the post also reports LMArena, EQ-Bench3, Creative Writing v3, and FActScore.
  - "[[llama-3]] — explicit PPO + DPO pipeline" → Llama 3 uses DPO, not PPO (arXiv:2407.21783 §4.1.4).
  - "[[qwen-3]] ... '20+ domains' ... rule + judge mix" → over 20 tasks with rule-based, reference-judge, and trained-RM rewards (§4.4).
  - Guideline "When classical RMs saturate, upgrade the RM's reasoning capability rather than the policy's" → rewritten; the source makes no such comparison.
- Removed as unsupported by the source: "reasoning-grade RMs unlock new post-training axes (hallucination reduction, EQ, personality)"; "Likely a heavyweight reasoning model (possibly Grok 4 itself)"; "hallucination reduction presumably uses factuality verification"; "implies an iterative online-RL loop"; industry comparison naming Magistral, Phi-4-reasoning, Tülu 3, and Nemotron GenRM; "closer to Constitutional-AI-style LLM judges but scaled up"; "no arXiv tech report exists for Grok 4.1" (not checked against a search).
- Not reported by the source: model size, token counts, SFT data size, RL algorithm and hyperparameters, reward-model identity, pairwise-evaluation sample size, numeric values of the hallucination, FActScore, EQ-Bench3, and Creative Writing charts.
