<!-- scope: recipe ledger for SA-SWE-32B (paper values and the released SkyRL example config), plus the deep-research, memory, and computer-use agents of SkyRL-Agent (arXiv 2511.16108v1)
     deps: [[skyrl-agent]]
     see-also: [[rloo]], [[dr-grpo]], [[deepswe]]
-->

# SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent — Recipe ledger
- **Parent card:** [[skyrl-agent]] (`papers/skyrl-agent.md`)
- **Sources:** (P) arXiv:2511.16108v1 (2025-11-20), cited by section, table, and figure; (C) released example config at github.com/NovaSky-AI/SkyRL@58891b2f0e84240da5cc1b3b76f0b91c62b0aef9 (main branch on 2026-09-14): `skyrl-agent/examples/run_skyrl/run_skyrl_swe.sh` and `skyrl-agent/examples/run_skyrl/skyrl_swe.yaml`.
- **Units:** "batch size 64" is tasks per step; with 8 rollouts per task, one step has 512 trajectories (Fig. 1b caption). The paper does not name the commit or script that produced SA-SWE-32B. The config's run name `skyagent-skyrl-32b-r2e-4500-loop-tool` matches the described setup, but C values are recorded as a separate fact from P.

## SA-SWE-32B (Qwen3-32B, RL only, R2E-Gym)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| SA-SWE-32B | 32B | RL | initial model; data | Qwen3-32B; 4.5K R2E-Gym instances | P §1, §4.3 | verified 2026-09-14 | no ablation reported |
| SA-SWE-32B | 32B | RL | batch size; mini-batch size; rollouts per task | 64; 64 (fully on-policy); 8 | P §4.2 | verified 2026-09-14 | §4.2: chosen "for stable training"; no ablation reported |
| SA-SWE-32B | 32B | RL | advantage | leave-one-out; no std normalization; no length normalization | P §4.2 | verified 2026-09-14 | follows DeepSWE and Dr. GRPO (cited); no ablation reported |
| SA-SWE-32B | 32B | RL | KL loss; entropy loss; LR | disabled; disabled; 1e-6 | P §4.2 | verified 2026-09-14 | no ablation reported |
| SA-SWE-32B | 32B | RL | max context; step limit; handling of truncated trajectories | 32K tokens; 50 turns; masked from the gradient, reward and advantage unchanged | P §4.2 | verified 2026-09-14 | §4.2 reason: avoid bias against trajectories with more actions; no ablation reported |
| SA-SWE-32B | 32B | RL | tools | AST-based search tool with hints, plus hints for recovery; Qwen3 template keeps previous-turn thinking | P §4.2 | verified 2026-09-14 | §4.2: bash-only (Mini-SWE-Agent) setup has 50/64 non-resolved; Fig. 1a non-resolved count lower than DeepSWE |
| SA-SWE-32B | 32B | RL | dispatcher | Async Pipeline | P §3.2, Fig. 1b | verified 2026-09-14 | Fig. 1b: about 1.55× faster than Async Batch (Bounded) |
| SA-SWE-32B | 32B | RL | hardware; compute | 2×8 H100 (Fig. 1b setting); 4,601 H100 hours | P Fig. 1b caption; Table 2 | verified 2026-09-14 | Table 2: DeepSWE 9,180 H100 hours |
| SA-SWE-32B | 32B | eval-gate | checkpoint | step 125 | P Fig. 1a caption | verified 2026-09-14 | no selection rule reported |
| SA-SWE-32B | 32B | eval-gate | SWE-Bench Verified protocol | simple ReAct, bash + file editor, 40K context, 100 max steps, one patch per instance | P §4.3, Table 2 | verified 2026-09-14 | not applicable |
| SA-SWE-32B | 32B | RL | rollout temperature; clip ε; loss aggregation; epochs | not reported in P (checked §3–§4, figure captions) | — | not reported | — |
| SA-SWE-32B (released example config) | 32B | RL | advantage estimator; std normalization; KL loss | `loop`; `grpo_norm_by_std=false`; `use_kl_loss=false` (`kl_loss_coef=0.001` set but unused) | C `run_skyrl_swe.sh` L32, L63–64, L69 | verified 2026-09-14 | no ablation reported |
| SA-SWE-32B (released example config) | 32B | RL | policy loss; clip low / high; loss reduction | `dual_clip`; 0.2 / 0.28; `seq_mean_token_sum_norm` | C `run_skyrl_swe.sh` L59, L65–67 | verified 2026-09-14 | no ablation reported |
| SA-SWE-32B (released example config) | 32B | RL | LR; train batch; mini-batch; update epochs per batch; samples per prompt; epochs | 1e-6; 64; 64; 1; 8; 10 | C `run_skyrl_swe.sh` L17, L43, L48–50, L60, L75 | verified 2026-09-14 | no ablation reported |
| SA-SWE-32B (released example config) | 32B | RL | max prompt length; max generate length; max sequence length | 8000; 32768; 40768 | C `run_skyrl_swe.sh` L55–56, L68 | verified 2026-09-14 | no ablation reported |
| SA-SWE-32B (released example config) | 32B | RL | parallelism | 2 nodes × 8 GPUs; FSDP2; sequence parallel 4; 4 vLLM engines with TP 4; colocated policy and inference | C `run_skyrl_swe.sh` L12–16, L34–41, L61 | verified 2026-09-14 | not applicable |
| SA-SWE-32B (released example config) | 32B | RL | sampling; max turns; tools | temperature 1, top-p 1; `max_iterations: 50`; bash, editor, cmd, finish, search enabled (browsing, Jupyter, think disabled) | C `skyrl_swe.yaml` L5–15, L26–32 | verified 2026-09-14 | no ablation reported |
| SA-SWE-32B (released example config) | 32B | RL | dispatcher; parallel agents; Qwen3 thinking | `async_pipeline`; 96; `qwen3_enable_thinking: true`, `qwen3_acc_thinking: true` | C `skyrl_swe.yaml` L41–47 | verified 2026-09-14 | P Fig. 1b speedup |

## Case-study agents (§5)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Deep-research agent (Qwen3-8B) | 8B | RL | algorithm; reward; backend | GRPO; general verifier (Ma et al. 2025); SkyRL-train | P §5.1 | verified 2026-09-14 | HLE-500: 12.6% → 18.8% (general verifier); 9.2% → 10.2% / 11.0% (gpt-oss-20b judge) |
| Deep-research agent (Qwen3-8B) | 8B | RL | global batch; rollouts; mini-batch; LR | 64; 8; 64; 5×10⁻⁶ | P §5.1 | verified 2026-09-14 | no ablation reported |
| Deep-research agent (Qwen3-8B) | 8B | RL | data | 50K problems sampled from MegaScience (Fan et al. 2025); difficulty by 4 Qwen3-8B thinking-mode rollouts; mix 25% Impossible (0/4), 30% Hard (1/4), 30% Medium (2/4), 15% Easy (3/4); balanced across computer science, biology, physics, economics | P §5.1 | verified 2026-09-14 | mix follows the Polaris mirrored distribution (cited); no ablation reported |
| Deep-research agent (Qwen3-8B) | 8B | RL | tools; summarizer; serving | SerperAPI search, Jina Reader; Qwen3-235B (non-reasoning) summarizer; final stage on 4 GH200 with data-parallel router | P §5.1 | verified 2026-09-14 | §5.1: Qwen-3-32B summarizer on one H200 took 2,101.6 s per iteration vs 592.1 s with a Qwen API |
| Memory agent (Qwen3-8B, non-thinking) | 8B | RL | backend; LoRA rank; batch; rollouts; max tokens per turn | Tinker; 128; 32; 8; 8K | P §5.2 | verified 2026-09-14 | result shown only in Fig. 6b |
| Memory agent (Qwen3-8B, non-thinking) | 8B | RL | data; chunk size; train / eval input length; verifier | RULER-HotpotQA; up to 4K tokens; up to 28K / up to 112K tokens; GPT-5-nano | P §5.2 | verified 2026-09-14 | reference: MemAgent reports 79.69% for Qwen2.5-7B-Instruct (batch 128, group 16, exact match) |
| Computer-use agent (Qwen3-8B) | 8B | RL | algorithm; backend; data; batch; rollouts | GRPO; VeRL; 32 OSWorld tasks (Hard / Medium / Easy); 8; 8 | P §5.3, Fig. 7 caption | verified 2026-09-14 | §5.3: training reward rises, validation accuracy shows little to no gain |
| Computer-use agent (Qwen3-8B) | 8B | RL | dispatcher; environments | Async Batch (Bounded); 32 fixed virtual desktops launched as Ray remote tasks | P §5.3 | verified 2026-09-14 | not applicable |

## Verification
- Created on 2026-09-14 from arXiv:2511.16108v1 and SkyRL commit 58891b2f0e84240da5cc1b3b76f0b91c62b0aef9 (files listed above).
- Audit claims not found in the source: none beyond those listed in [[skyrl-agent]] Verification.
