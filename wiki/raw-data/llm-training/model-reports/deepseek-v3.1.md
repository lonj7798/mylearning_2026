<!-- scope: DeepSeek-V3.2 technical report (arXiv:2512.02556): DeepSeek Sparse Attention added to DeepSeek-V3.1-Terminus by continued training, specialist distillation, one mixed GRPO stage with stabilizers, thinking in tool use, agentic task synthesis. The file slug stays deepseek-v3.1 so existing links resolve; facts about DeepSeek-V3.1 itself come from the V3.1 model card and carry that locus.
     deps: [[deepseek-v3]], [[grpo]]
     see-also: [[deepseek-v3.1-recipe]], [[deepseek-r1]], [[deepseekmath]], [[john-schulman-kl-tricks]], [[generative-reward-models]], [[kimi-k2-agentic-data]]
-->

# DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
- **Core Insight:** DeepSeek-V3.2 adds DeepSeek Sparse Attention (2048 selected key-value tokens per query) to DeepSeek-V3.1-Terminus through 943.7B tokens of continued training, spends an RL budget that exceeds 10% of pre-training cost, and is reported to perform comparably to GPT-5 on reasoning benchmarks (Abstract, §2.1.1, §4.1).
- **Guideline:** When an MoE policy is trained with RL on samples from a separate inference engine, reuse the sampled expert routes and the top-p/top-k truncation masks in the training pass, because the report found Keep Routing "crucial for RL training stability of MoE models" and top-p with Keep Sampling Mask preserved language consistency (§3.1).
- **Authors:** DeepSeek-AI (byline). App. E lists contributors alphabetically by first name: Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, et al.
- **Year:** 2025 (arXiv v1 2025-12). DeepSeek API news index: V3.1 release 2025-08-21, V3.1 update 2025-09-22, V3.2-Exp release 2025-09-29, V3.2 release 2025-12-01.
- **URL:** https://arxiv.org/abs/2512.02556 (V3.1 facts: https://huggingface.co/deepseek-ai/DeepSeek-V3.1)
- **Source type:** official technical report (V3.1 section: model/dataset card)
- **Relevant topics:** sparse attention, lightning indexer, continued pre-training, specialist distillation, mixed RL, GRPO stability, off-policy masking, MoE routing replay, thinking in tool use, agentic task synthesis, test-time context management

## Abstract
The report introduces DeepSeek-V3.2 with three contributions. (1) DeepSeek Sparse Attention (DSA) lowers attention cost while preserving long-context performance. (2) A stable RL protocol with scaled post-training compute makes V3.2 perform comparably to GPT-5; the high-compute variant DeepSeek-V3.2-Speciale is reported to surpass GPT-5, to match Gemini-3.0-Pro in reasoning, and to reach gold-medal level at IMO 2025 and IOI 2025. (3) A large-scale agentic task synthesis pipeline generates tool-use training data and is reported to improve generalization and instruction following in interactive environments.

## Key Contributions
- DSA: a lightning indexer scores preceding tokens, and attention runs only over the top-k key-value entries; it is instantiated on the MQA mode of MLA (§2.1, Eq. 1-2, Figure 2).
- Two-stage continued pre-training (dense indexer warm-up, then sparse training) from V3.1-Terminus; V3.2-Exp shows no substantial degradation vs V3.1-Terminus on short- and long-context tasks (§2.1.1, §2.2).
- Four GRPO stabilizers: unbiased KL estimate, Off-Policy Sequence Masking, Keep Routing, Keep Sampling Mask (§3.1).
- Thinking context management for tool calls and a prompt-based cold start (§3.2.1-3.2.2).
- Agentic RL data: 24,667 code-agent, 50,275 search-agent, 4,417 general-agent, and 5,908 code-interpreter tasks; the general-agent tasks span 1,827 synthesized environments (Table 1, §3.2.3).

## Key Figures/Tables to Study
- Figure 2 and Eq. 1-4: DSA under MLA and the indexer KL losses. Figure 3: cost per million tokens vs token position on H800.
- Eq. 7-9: unbiased KL estimate and the off-policy sequence mask.
- Table 1 and the trip-planning example: task counts and the general-agent synthesis output.
- Table 2 (main comparison), Table 3 (accuracy with output-token counts), Table 4 (competition results).
- Table 5 and Figure 5: difficulty of synthetic tasks and transfer of synthetic-only RL. Figure 6: BrowseComp under context-management strategies.

## Technical Details
Recipe values with loci are in [[deepseek-v3.1-recipe]].

**DSA.** The index score is I_{t,s} = Σ_{j=1..H^I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s), where H^I is the number of indexer heads, q^I_{t,j} and w^I_{t,j} come from query token h_t, and k^I_s comes from preceding token h_s (Eq. 1). ReLU is chosen for throughput, and the indexer has few heads and can run in FP8 (§2.1). The attention output u_t is computed over the key-value entries c_s whose I_{t,s} is in the top-k (Eq. 2). Each MLA latent vector is shared across all query heads of a token (§2.1). Core attention cost drops from O(L²) to O(Lk); the indexer remains O(L²) but costs less than MLA (§2.3). H^I and the indexer dimension d^I are not printed.

**Continued pre-training.** Both stages use data aligned with the 128K extension data of V3.1-Terminus (§2.1.1). Warm-up: dense attention, all weights frozen except the indexer, and the indexer is trained with KL(p_{t,:} ‖ Softmax(I_{t,:})), where p_{t,:} is the main attention summed over heads and L1-normalized (Eq. 3). Sparse stage: all parameters train; the indexer KL is restricted to the selected set S_t (Eq. 4); the indexer input is detached, so the indexer learns only from its KL loss and the main model only from the language-modeling loss (§2.1.1).

**Specialist distillation.** Specialists are fine-tuned from the same V3.2 base for six domains (mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, agentic search), plus writing and general QA, each in thinking and non-thinking modes, and each trained with large-scale RL (§3). Different models generate thinking-mode and non-thinking-mode data (§3). Models trained on the distilled data score slightly below the specialists, and subsequent RL removes the gap (§3); no numbers are printed.

**Mixed RL.** GRPO merges reasoning, agent, and human-alignment training into one RL stage (§3). Rewards: rule-based outcome reward, length penalty, and language-consistency reward for reasoning and agent tasks; a generative reward model with per-prompt rubrics for general tasks (§3). The advantage is Â_{i,t} = R_i − mean(R), without division by the standard deviation (§3.1). V3.2 runs "thousands of steps" of RL (§3). V3.2-Speciale trains on reasoning data only, with a reduced length penalty, plus the DeepSeekMath-V2 proof dataset and reward method (§3).

**Stabilizers (§3.1).** (1) The K3 KL estimate is multiplied by π_θ/π_old to make its gradient unbiased (Eq. 7); KL strength differs by domain, and for mathematics a weak or zero KL penalty can perform better. (2) A sequence gets mask M_{i,t} = 0 when Â_{i,t} < 0 and (1/|o_i|) Σ_t log(π_old/π_θ) > δ, where π_old is the probability returned by the inference engine and δ is a threshold (Eq. 9). (3) Keep Routing reuses inference-time expert routes in training and has been used since DeepSeek-V3-0324. (4) Keep Sampling Mask applies the top-p/top-k truncation mask from sampling to π_θ.

**Thinking in tool use.** Earlier reasoning content is removed only when a new user message arrives; tool outputs alone keep it, and tool-call history is always kept (§3.2.1). Frameworks that send tool results as user messages (Roo Code, Terminus) are recommended to use non-thinking mode (§3.2.1). The cold start combines reasoning data and non-reasoning agent data through system prompts; Table 8 permits up to 20 Python executions inside <think> (§3.2.2, App. B).

**Agentic environments (§3.2.3).** Search tasks come from a multi-agent pipeline that keeps a QA pair only if its ground truth is verified correct and all candidate answers are verifiably incorrect, plus rubric-scored helpfulness data. Code-agent environments come from GitHub issue-PR pairs and count as built only if the gold patch gives F2P > 0 and P2F = 0; languages include Python, Java, JavaScript, TypeScript, C, C++, Go, and PHP. General-agent environments: an agent builds a sandbox database, task-specific tool functions, and a task with Python solution and verifier functions, then raises difficulty; instances with non-zero pass@100 under V3.2 are kept (1,827 environments, 4,417 tasks).

**Starting checkpoint: DeepSeek-V3.1 (V3.1 model card).** One model supports thinking and non-thinking modes by chat template, and V3.1-Think is reported to match R1-0528 answer quality with faster responses (Introduction). V3.1-Base extends the V3 base with the V3 two-phase method on more long documents: the 32K phase grows 10-fold to 630B tokens and the 128K phase 3.3x to 209B tokens (Introduction). Weights and activations use the UE8M0 FP8 scale format (Introduction). V3.1 has 671B total and 37B activated parameters and 128K context (Model Downloads table).

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality / forgetting.** Merging RL domains into one stage is stated to balance domains while "circumventing the catastrophic forgetting issues commonly associated with multi-stage training paradigms" (§3); no ablation is printed. MCP-Universe, MCP-Mark, and Tool-Decathlon environments were not seen in RL, which the authors read as out-of-domain generalization (§4.1). The authors attribute a knowledge-breadth gap vs Gemini-3.0-Pro to fewer total training FLOPs (§5).
- **Agentic training.** On 50 sampled synthetic general-agent tasks, pass@1 is 12% for V3.2-Exp and 62% for GPT-5-Thinking (Table 5). RL on V3.2-SFT with only synthetic general-agent tasks (non-thinking mode) improves Tau2Bench, MCP-Mark, and MCP-Universe; V3.2-Exp, trained with RL only in search and code environments, does not improve on them (§4.3, Figure 5; values shown only in the figure).
- **Negative feedback.** Off-Policy Sequence Masking removes only negative-advantage sequences with high policy divergence; the authors state that "highly off-policy negative samples can be detrimental" and that masking improved stability in some otherwise unstable runs (§3.1). This is a control on negative-as-gradient updates.
- **Long context.** V3.2-Exp scores four points higher than V3.1-Terminus on AA-LCR in reasoning mode (§2.2). On BrowseComp, "approximately 20%+" of cases exceed 128K; score is 51.4 without context management and 67.6 with Discard-all, triggered at 80% of the window (§4.1, §4.4).
- **Distillation.** Specialist RL → distillation → mixed RL is the post-training order (§3). DeepSeek-V4 (arXiv:2606.19348 §5.1) later replaces the mixed RL stage with multi-teacher on-policy distillation using a full-vocabulary reverse KL (§5.1.2).

## Connections
- [[deepseek-v3]]: MLA, MoE, the context-extension method, and the earlier expert-model SFT pipeline.
- [[grpo]] and [[deepseekmath]]: the GRPO objective that §3.1 modifies.
- [[john-schulman-kl-tricks]]: the K3 estimator (Schulman, 2020) that Eq. 7 reweights.
- [[generative-reward-models]]: generative reward models; V3.2 uses one with per-prompt rubrics for general tasks (§3).
- [[kimi-k2-agentic-data]]: synthetic agentic data in Kimi K2, a peer open model that Table 2 compares against.
- [[deepseek-r1]]: the thinking-context policy that §3.2.1 changes for tool calls.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2512.02556 (v1); https://huggingface.co/deepseek-ai/DeepSeek-V3.1 (model card); DeepSeek API news index (api-docs.deepseek.com/news/news250929); arXiv:2606.19348 v1 §5.1 for the V4 sentence.
- Corrections to the previous card version:
  - Card mixed V3.1 (slug, title) with the V3.2 report (URL, body) → card now describes the V3.2 report, with V3.1 facts labeled by model-card locus.
  - "V3.1 merged V3 and R1 into a single hybrid model" → one model supports thinking and non-thinking modes through the chat template (V3.1 card, Introduction).
  - "RL specifics reserved" / "KL handling not disclosed" / "exact specialist domains not disclosed" → §3 and §3.1 disclose the six domains, rewards, KL correction, masking, Keep Routing, and Keep Sampling Mask.
  - "'GenRM' specialist per domain" → specialists are RL-trained domain models; a rubric-based generative RM scores general tasks (§3).
  - "Mixed RL replaces V3's sequential SFT → RL pipeline" → distillation still precedes RL; mixed RL merges reasoning, agent, and alignment RL into one stage (§3).
  - "Specialist distillation is new; V3 only distilled from R1" → V3 also used domain expert models as SFT data generators (arXiv:2412.19437 §5.1).
  - "V3.2 near-parity with V3.1-Terminus" → the parity evaluation compares V3.2-Exp with V3.1-Terminus (§2.2).
  - "top-k keeps compute bounded regardless of sequence length" → the indexer remains O(L²) (§2.3).
- Removed as unsupported by the source: "makes a >10%-of-pretrain RL budget feasible" (no measurement links DSA to RL cost); V3 "~0.2%" post-training share comparison (different units); "post-training compute-budget chart" and "V3.2 vs V3.1-Terminus benchmark table" (neither exists in the report).
- Not reported by the source: RL learning rate, batch size, group size G, clip ε, KL coefficients, δ, top-p value, maximum rollout length, number of prompts per step; SFT data size and epochs; indexer head count and dimension; ablation separating DSA from RL compute.
