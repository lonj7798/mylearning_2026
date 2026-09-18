<!-- scope: Recipe ledger for the Kimi K2 technical report (arXiv:2507.20534v2); companion to [[kimi-k2]]
     deps: [[kimi-k2]]
     see-also: [[kimi-k1-5-recipe]], [[deepseek-v3-recipe]]
-->

# Kimi K2: Open Agentic Intelligence — Recipe ledger
Companion to [[kimi-k2]]. Rows refer to Kimi-K2-Base (pre-training) and Kimi-K2-Instruct (post-training), 1.04T total / 32B activated parameters (arXiv:2507.20534v2 §2.3, Table 2 prints 32.6B activated; Table 4 prints 1043B / 32B), unless the Model column names an ablation model. Status dates refer to reading the arXiv v2 PDF on 2026-09-14; the rows below were also found in v1.

Units: batch sizes are in tokens as printed; "tokens" are training tokens seen. The report states the WSD run as 10T constant + 5.5T cosine = 15.5T; it does not state whether the 400B + 60B annealing/long-context tokens are inside or after the 15.5T.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Architecture | 61 layers; MLA; hidden 7168; expert hidden 2048; 384 experts, 8 active, 1 shared; 64 attention heads; 1 dense layer; no expert grouping | arXiv:2507.20534v2 §2.3, Table 2 | verified 2026-09-14 | Figure 5: sparsity 48 needs 1.69×/1.39×/1.15× fewer FLOPs than sparsity 8/16/32 at val. loss 1.5; Figure 6: doubling heads gives 0.5-1.2% lower val. loss, 83% more inference FLOPs at 128k |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Optimizer | MuonClip: Muon with weight decay, update RMS matched to Adam, per-head QK-Clip | §2.1, Algorithm 1, §2.5 | verified 2026-09-14 | Moonlight (prior work) cited for Muon > AdamW at equal compute; not re-ablated here |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | QK-Clip threshold τ | 100 | §2.1, Figure 2 | verified 2026-09-14 | App. D: 0.5B act./3B total MoE with τ = 30 shows negligible loss change and no significant downstream degradation |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Muon momentum μ, QK-Clip α | μ not printed; naive-clip balancing α "typically set to 0.5" (K2 uses per-head clip with √γ_h on q^C, k^C) | §2.1 | verified 2026-09-14 (α, per-head form); μ not reported | no ablation reported |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Tokens seen | 15.5T | Abstract, §2.5 | verified 2026-09-14 | n/a |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Context length | 4,096 tokens | §2.5 | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | LR schedule (WSD) | 500-step warmup; constant 2e-4 for first 10T tokens | §2.5 | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-decay/anneal | Decay | cosine 2e-4 → 2e-5 over 5.5T tokens | §2.5 | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Weight decay | 0.1 throughout | §2.5 | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Global batch | 67M tokens | §2.5 | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Grad clip, β/ε-type optimizer constants | not printed | §2.1, §2.5 | not reported (body and appendices checked) | n/a |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Data domains | Web Text, Code, Mathematics, Knowledge; percentages not printed | §2.2 | verified 2026-09-14 (domains); mixture not reported | no ablation reported |
| Early K2 checkpoint (ablation) | not reported | pretrain-stable | Knowledge rephrasing vs repetition | SimpleQA: raw wiki-text ×10 epochs 23.76; 1 rephrasing ×10 epochs 27.39; 10 rephrasings ×1 epoch 28.94 | §2.2, Table 1 | verified 2026-09-14 | Table 1; number of seeds not reported |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Rephrasing limit | each knowledge corpus rephrased at most twice | §2.2 | verified 2026-09-14 | "similarly encouraging results" on other corpora; no numbers |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-decay/anneal | Annealing | LR 2e-5 → 7e-6; 400B tokens at 4k sequence length; batch 67M tokens | §2.5 | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Base | 1.04T / 32B act. | long-context | Long-context activation | 60B tokens at 32k; then YaRN to 128k (YaRN parameters not printed) | §2.5 | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Parallelism | 16-way PP with virtual stages; 16-way EP; ZeRO-1 DP; node count any multiple of 32 | §2.4.2 | verified 2026-09-14 | EP = 16 chosen for full compute-communication overlap |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Hardware | NVIDIA H800; 8 GPUs and 2 TB RAM per node; 8×400 Gbps RoCE | §2.4.1 | verified 2026-09-14 | n/a |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | Activation memory | selective recomputation; FP8-E4M3 storage (1×128 tiles, FP32 scales) for MoE up-projection and SwiGLU inputs, not in computation; CPU offload of remaining activations | §2.4.3 | verified 2026-09-14 | "small-scale experiments show no measurable loss increase" (FP8 storage) |
| Kimi-K2-Base | 1.04T / 32B act. | all | GPU count, GPU hours | not printed | whole report | not reported (body and appendices checked) | n/a |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT | Optimizer | Muon | §3.1 | verified 2026-09-14 | cites Moonlight: Muon-pretrained checkpoints fine-tune best with Muon |
| Kimi-K2-Instruct | 1.04T / 32B act. | distill-SFT | Response sources | K1.5 and in-house domain-specialized expert models; LLM or human judges filter | §3.1 | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT | Dataset size, epochs, LR, batch, masking, packing | not printed | §3.1 | not reported (body and appendices checked) | n/a |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT | Tool repository | 3000+ real MCP tools from GitHub; over 20,000 synthetic tools | §3.1.1 | verified 2026-09-14 | Figure 9 (t-SNE coverage); no ablation |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT | Agents, tasks, trajectories | thousands of agents; rubric per task; "tens of thousands" of training examples; LLM judge keeps rubric-passing trajectories | §3.1.1 | verified 2026-09-14 | no ablation or rejection rate reported |
| Kimi-K2-Instruct | 1.04T / 32B act. | RL | Algorithm | K1.5 objective: squared (r − r̄ − τ log π_θ/π_old) over K samples from π_old; Muon optimizer | §3.2.3 | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Instruct | 1.04T / 32B act. | RL | τ, K, prompts per step, LR, steps, max response length | not printed | §3.2.3 | not reported (body and appendices checked) | n/a |
| Kimi-K2-Instruct | 1.04T / 32B act. | RL | Budget control | per-sample max token budget by task type; truncation + penalty (budgets and penalty not printed) | §3.2.3 | verified 2026-09-14 (mechanism) | authors state token efficiency improved "significantly"; no numbers |
| Kimi-K2-Instruct | 1.04T / 32B act. | RL | PTX loss | auxiliary loss on hand-selected high-quality samples (weight and data size not printed) | §3.2.3 | verified 2026-09-14 (mechanism) | authors state generalization improved; no numbers |
| Kimi-K2-Instruct | 1.04T / 32B act. | RL | Temperature | decay schedule from high to lower temperature (values not printed) | §3.2.3 | verified 2026-09-14 (mechanism) | no ablation reported |
| Kimi-K2-Instruct | 1.04T / 32B act. | RL | Math/STEM/logic prompt filter | moderate difficulty by SFT model pass@k (k and thresholds not printed) | §3.2.1 | verified 2026-09-14 (mechanism) | no ablation reported |
| Kimi-K2-Instruct | 1.04T / 32B act. | reward-model | Self-critique rubric reward | K2 critic, initialized in SFT on open-source + in-house preference data; pairwise ranking against core, prescriptive, human-annotated rubrics; refined on verifiable-reward rollouts | §3.2.2, App. F | verified 2026-09-14 | App. F.3 lists over-confidence as a known side effect; no ablation |
| Kimi-K2-Instruct | 1.04T / 32B act. | RL | Sandbox capacity | over 10,000 concurrent sandbox instances (Kubernetes) | §3.2.1 | verified 2026-09-14 | n/a |
| Kimi-K2-Instruct | 1.04T / 32B act. | RL | Weight update time | full parameter update in under 30 seconds (checkpoint engine) | §3.3.2 | verified 2026-09-14 | n/a |
| Kimi-K2-Instruct | 1.04T / 32B act. | eval-gate | Evaluation settings | non-thinking mode; max output 8192 tokens (16384 for SWE-bench Verified Agentless); 128K context with truncation; Tau2 and ACEBench at temperature 0.0, Tau2 Avg@4 | §4.1.1, App. C | verified 2026-09-14 | n/a |

Checked sources: arXiv:2507.20534v2 body §1-§6 and Appendices B-G, compared with v1 (2025-07-28). The report links the checkpoint-engine code (github.com/MoonshotAI/checkpoint-engine) and the model card (huggingface.co/moonshotai/Kimi-K2-Instruct); no training configuration file is cited.
