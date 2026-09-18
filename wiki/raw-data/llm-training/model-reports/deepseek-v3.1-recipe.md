<!-- scope: Recipe ledger for the DeepSeek-V3.2 technical report (arXiv:2512.02556v1) and the DeepSeek-V3.1 model card; companion to [[deepseek-v3.1]]
     deps: [[deepseek-v3.1]]
     see-also: [[deepseek-v3-recipe]]
-->

# DeepSeek-V3.2 and DeepSeek-V3.1 — Recipe ledger
Companion to [[deepseek-v3.1]] (the card for the DeepSeek-V3.2 report, kept under the deepseek-v3.1 slug). Rows cite either "arXiv:2512.02556v1" (V3.2 report) or "V3.1 model card" (huggingface.co/deepseek-ai/DeepSeek-V3.1, read 2026-09-14). The V3.2 report does not print a parameter count; the Size column says "not printed" for V3.2 rows.

Units: "sequences of 128K tokens" and token totals are as printed. "Tasks" in Table 1 are RL tasks (prompts with environments), not samples.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3.1-Base | 671B / 37B act. | long-context | 32K extension phase tokens | 630B (10-fold the V3 phase) | V3.1 model card, Introduction | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.1-Base | 671B / 37B act. | long-context | 128K extension phase tokens | 209B (3.3x the V3 phase) | V3.1 model card, Introduction | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.1-Base | 671B / 37B act. | long-context | Method | two-phase extension following the DeepSeek-V3 report, with additional long documents | V3.1 model card, Introduction | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.1 | 671B / 37B act. | pretrain-stable | Numeric format | UE8M0 FP8 scale format on weights and activations | V3.1 model card, Introduction | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.1 | 671B / 37B act. | SFT + RL | Post-training data, algorithm, hyperparameters | not printed | V3.1 model card | not reported (model card checked) | n/a |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train | Starting checkpoint | DeepSeek-V3.1-Terminus, context already extended to 128K | arXiv:2512.02556v1 §2.1.1 | verified 2026-09-14 | n/a |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train | Data distribution (both stages) | aligned with the 128K long-context extension data of V3.1-Terminus | §2.1.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train (dense warm-up) | Trainable parameters | lightning indexer only; dense attention kept | §2.1.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train (dense warm-up) | Loss | Σ_t KL(p_{t,:} ‖ Softmax(I_{t,:})); p = main attention summed over heads, L1-normalized | §2.1.1, Eq. 3 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train (dense warm-up) | LR | 1e-3 | §2.1.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train (dense warm-up) | Steps × batch = tokens | 1000 steps × 16 sequences of 128K tokens = 2.1B tokens | §2.1.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train (sparse) | Trainable parameters / losses | all parameters; indexer input detached; indexer trained by Eq. 4 KL over selected set only; main model by LM loss only | §2.1.1, Eq. 4 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train (sparse) | LR | 7.3e-6 | §2.1.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train (sparse) | Selected key-value tokens per query (k) | 2048 | §2.1.1 | verified 2026-09-14 | §2.2: V3.2-Exp similar to V3.1-Terminus on standard benchmarks, ChatbotArena Elo closely matched (10 Nov 2025), AA-LCR +4 in reasoning mode; no k ablation |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train (sparse) | Steps × batch = tokens | 15000 steps × 480 sequences of 128K tokens = 943.7B tokens | §2.1.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2-Exp / V3.2 | not printed | mid-train | Indexer heads H^I, indexer dimension d^I, optimizer, warmup | not printed | §2.1 | not reported (body and appendices checked; report points to the open inference code) | n/a |
| DeepSeek-V3.2 specialists | not printed | RL (specialists) | Domains | mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, agentic search; plus writing and general QA; thinking and non-thinking modes | §3 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2 specialists | not printed | RL (specialists) | Initialization and training | fine-tuned from the same pre-trained V3.2 base; large-scale RL per specialist | §3 | verified 2026-09-14 | n/a |
| DeepSeek-V3.2 | not printed | distill-SFT | Data | specialist-generated domain data; different generator models for thinking and non-thinking data | §3 | verified 2026-09-14 | §3: distilled model slightly below specialists; gap removed by later RL (no numbers printed) |
| DeepSeek-V3.2 | not printed | distill-SFT | Size, epochs, LR, packing | not printed | §3 | not reported (body checked) | n/a |
| DeepSeek-V3.2 | not printed | SFT (cold start) | Tool use inside reasoning | system prompts per task type; reasoning prompt with <think></think>; agent prompt with tool-call format; combined prompt allowing up to 20 code executions in <think> | §3.2.2, App. B Tables 6-8 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2 | not printed | RL | Algorithm and stage layout | GRPO; reasoning, agent, and human-alignment training merged into one RL stage | §3 | verified 2026-09-14 | §3 states it avoids multi-stage catastrophic forgetting; no ablation printed |
| DeepSeek-V3.2 | not printed | RL | Rewards | reasoning and agent: rule-based outcome reward, length penalty, language-consistency reward; general: generative RM with per-prompt rubrics | §3 | verified 2026-09-14 | §4.1: V3.2 scores are constrained by a length-constraint reward model; relaxing length constraints (Speciale, §4.2) raises scores (Table 3) |
| DeepSeek-V3.2 | not printed | RL | Advantage | Â = R_i − mean(R) over the group (no std division) | §3.1, Eq. 5 text | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3.2 | not printed | RL | KL estimator | K3 estimate multiplied by π_θ/π_old (unbiased gradient); strength differs by domain; weak or none for mathematics | §3.1, Eq. 7 | verified 2026-09-14 | §3.1: "can yield improved performance" for math; no numbers |
| DeepSeek-V3.2 | not printed | RL | Off-Policy Sequence Masking | mask sequences with Â < 0 and mean token log(π_old/π_θ) > δ; π_old from the inference engine | §3.1, Eq. 9 | verified 2026-09-14 | §3.1: improved stability in some unstable runs; no numbers |
| DeepSeek-V3.2 | not printed | RL | Keep Routing | inference-time expert routes enforced in training; used since DeepSeek-V3-0324 | §3.1 | verified 2026-09-14 | §3.1: "crucial" for MoE RL stability; no numbers |
| DeepSeek-V3.2 | not printed | RL | Keep Sampling Mask | top-p/top-k truncation mask from sampling applied to π_θ | §3.1 | verified 2026-09-14 | §3.1: with top-p, preserves language consistency; no numbers |
| DeepSeek-V3.2 | not printed | RL | Steps | "thousands of steps" | §3 | verified 2026-09-14 | n/a |
| DeepSeek-V3.2 | not printed | RL | Compute | RL budget exceeds 10% of pre-training cost | §1, §4.1 | verified 2026-09-14 | §4.1: consistent gains observed with longer RL budget (no curve printed) |
| DeepSeek-V3.2 | not printed | RL | β, ε, δ, G, LR, batch, prompts per step, max response length, temperature, top-p value | not printed | §3, §3.1 | not reported (body and appendices checked) | n/a |
| DeepSeek-V3.2 | not printed | RL | Agent tasks (Table 1) | code agent 24,667 (real env, extracted prompts); search agent 50,275 (real, synthesized); general agent 4,417 (synthesized, synthesized); code interpreter 5,908 (real, extracted) | Table 1 | verified 2026-09-14 | §4.3, Figure 5: synthetic-only general-agent RL improves Tau2Bench, MCP-Mark, MCP-Universe over V3.2-SFT |
| DeepSeek-V3.2 | not printed | RL | General-agent environment filter | keep instances with non-zero pass@100 under V3.2 → 1,827 environments | §3.2.3 | verified 2026-09-14 | Table 5: V3.2-Exp pass@1 12% on 50 sampled tasks |
| DeepSeek-V3.2 | not printed | RL | Totals stated in introduction | over 1,800 environments and 85,000 prompts | §1 | verified 2026-09-14 | n/a |
| DeepSeek-V3.2-Speciale | not printed | RL | Data and length penalty | reasoning data only; reduced length penalty; DeepSeekMath-V2 dataset and reward method for proofs | §3 | verified 2026-09-14 | Table 3: higher accuracy with more output tokens than V3.2 (e.g. AIME 2025 96.0 at 23k vs 93.1 at 16k) |
| DeepSeek-V3.2 | not printed | eval-gate | Evaluation settings | temperature 1.0; context window 128K; tool-use benchmarks in thinking mode | §4.1 | verified 2026-09-14 | n/a |
| DeepSeek-V3.2 | not printed | eval-gate | Test-time context management (search agent) | triggered above 80% of context; Summary, Discard-75%, Discard-all | §4.4 | verified 2026-09-14 | Figure 6: BrowseComp 51.4 without, 67.6 with Discard-all |
| DeepSeek-V3.2-Speciale | not printed | eval-gate | Competition protocol | max generation 128k; no tools; IOI: 500 samples, filter, submit 50 longest-thinking; ICPC: 32 samples; IMO/CMO: generate-verify-refine | App. D | verified 2026-09-14 | Table 4: IMO 35/42, CMO 102/126, IOI 492/600, ICPC WF 10/12 |

Checked sources: arXiv:2512.02556v1 §1-§5 and Appendices A-D; DeepSeek-V3.1 model card (Introduction, Model Downloads, Chat Template, How to Run Locally). No training config is released with either artifact.
