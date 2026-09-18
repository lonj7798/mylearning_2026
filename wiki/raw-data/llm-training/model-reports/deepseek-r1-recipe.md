<!-- scope: recipe ledger for DeepSeek-R1-Zero, DeepSeek-R1, the R1 reward models, the R1-Distill students, and the paper's Qwen RL baselines
     deps: [[deepseek-r1]]
     see-also: [[deepseek-v3]], [[grpo]]
-->

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — Recipe ledger
- **Parent card:** [[deepseek-r1]] (`model-reports/deepseek-r1.md`)
- **Source:** arXiv:2501.12948v2 (2026-01-04) main text and Supplementary; v1 (2025-01-22) cited where it differs. Loci "§" are main-text sections; letters (A–H) are Supplementary sections of v2.
- **Units:** "questions" are prompts; "batch 512" is 32 questions × 16 sampled outputs (§2.1). SFT batch-size units are not stated by the source. "GPU hours" are H800 GPU hours (Table 7).
- **Size:** DeepSeek-R1-Zero and DeepSeek-R1 are 671B total / 37B activated MoE models built on DeepSeek-V3-Base (A.1, Table 8). Supp. B.4.4 prints "660B" for the same models.

## DeepSeek-R1-Zero (RL directly on DeepSeek-V3-Base, no SFT)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Zero | 671B MoE | RL | algorithm; KL placement | GRPO; KL estimator (Eq. 2) added directly to the loss, not to the reward | v2 §2.1 Eq. 1–2; A.3 | verified 2026-09-14 | A.3 Fig. 4: on DeepSeek-Coder-V2-Lite (16B MoE, 2.4B active), PPO with GAE λ=0.95 is worse than GRPO on MATH; λ=1.0 is close to GRPO |
| DeepSeek-R1-Zero | 671B MoE | RL | learning rate; KL coefficient | 3e-6; 0.001 | v2 §2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Zero | 671B MoE | RL | clip ratio ε | not reported for R1-Zero (checked §2.1, §2.3, A.3, B.4) | — | not reported | ε = 10 is reported only for R1 stage 1 (§3.2.1) |
| DeepSeek-R1-Zero | 671B MoE | RL | rollout temperature; outputs per question | 1; 16 | v2 §2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Zero | 671B MoE | RL | max response length | 32,768 tokens before step 8.2k; 65,536 tokens afterward | v2 §2.1 | verified 2026-09-14 | §2.1: performance and response length "exhibit a significant jump at the 8.2k step" (Fig. 1) |
| DeepSeek-R1-Zero | 671B MoE | RL | questions per step; samples per step | 32 unique questions; batch 512 | v2 §2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Zero | 671B MoE | RL | rollout/update schedule | 8,192 outputs per rollout, split into 16 mini-batches, 1 inner epoch | v2 §2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Zero | 671B MoE | RL | reference-model refresh | replaced by latest policy every 400 steps | v2 §2.1; A.3 | verified 2026-09-14 | A.3 gives the reason (balance exploration scope and stability); no ablation reported |
| DeepSeek-R1-Zero | 671B MoE | RL | total steps; epochs | 10,400 steps; 1.6 epochs | v2 §2.1 | verified 2026-09-14 | not applicable |
| DeepSeek-R1-Zero | 671B MoE | RL | reward | accuracy reward + format reward, equal weight; no neural outcome or process RM | v2 §2.2 Eq. 4 | verified 2026-09-14 | §2.2: neural RMs judged susceptible to reward hacking; no table |
| DeepSeek-R1-Zero | 671B MoE | RL | compute | 64×8 H800 GPUs, about 198 h; 101K GPU hours | v2 B.4.4; Table 7 | verified 2026-09-14 | not applicable |
| DeepSeek-R1-Zero and DeepSeek-R1 | 671B MoE | RL | reasoning prompts (Table 4) | Math 26K; Code 17K (text: 17k algorithm + 8k bug-fixing); STEM 22K; Logic 15K; all Chinese or English | v2 B.3.1 Table 4 | verified 2026-09-14 | B.3.1 does not split prompt sets by model; no ablation reported |
| DeepSeek-R1-Zero and DeepSeek-R1 | 671B MoE | RL | math and STEM reward value | 1 if the answer matches the reference, else 0 | v2 B.3.1 | verified 2026-09-14 | no ablation reported |

## DeepSeek-R1 (cold-start SFT → reasoning RL → ~800K SFT from V3-Base → mixed RL)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1 (Dev1) | 671B MoE | SFT | cold-start examples | "thousands"; exact count not given | v2 §3; B.3.2; v1 §2.3.1 | verified 2026-09-14 | §4, Table 3: Dev1 AIME 2024 pass@1 59.0 vs R1-Zero 77.9, attributed to the limited cold-start size |
| DeepSeek-R1 (Dev1) | 671B MoE | SFT | cold-start construction | R1-Zero samples at temperature 1.0; keep correct (sympy for math) and readable (repetition detection, language-mixing filter); DeepSeek-V3 refines reasoning and summary; human annotators convert and verify | v2 B.3.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 (Dev1 and Dev3) | 671B MoE | SFT | base; epochs; LR; context; batch | DeepSeek-V3-Base; 2–3 epochs; cosine 5×10⁻⁵ → 5×10⁻⁶; 32,768 tokens; 128 | v2 B.4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 (Dev3) | 671B MoE | SFT | epochs (second-stage SFT) | two epochs | v1 §2.3.3 | conflict | v2 B.4.2 prints 2–3 epochs for both SFT stages; neither version says which count produced the released checkpoint |
| DeepSeek-R1 (Dev1 → Dev2) | 671B MoE | RL | LR; KL coefficient; clip ε | 3e-6; 0.001; 10 | v2 §3.2.1 | verified 2026-09-14 | §3.2.1: a lower ε truncates gradients for many tokens and degrades performance; a higher ε may cause instability (no table) |
| DeepSeek-R1 (Dev1 → Dev2) | 671B MoE | RL | temperature; outputs; max length; questions/step; batch | 1; 16; 32,768; 32; 512 | v2 §3.2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 (Dev1 → Dev2) | 671B MoE | RL | reference refresh; rollout split | every 400 steps; 8,192 outputs into 16 mini-batches, 1 inner epoch | v2 §3.2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 (Dev1 → Dev2) | 671B MoE | RL | language consistency reward | share of target-language words in the CoT, added to the final reward | v2 §3.2.1 Eq. 7 | verified 2026-09-14 | B.6 Fig. 7 (on R1-Distill-Qwen-7B): with the reward, language consistency stays stable; math comparable; coding slightly lower |
| DeepSeek-R1 (Dev1 → Dev2) | 671B MoE | RL | steps | not reported (checked §3.2.1, B.4; v1 §2.3.2 says "until it achieves convergence on reasoning tasks") | — | not reported | — |
| DeepSeek-R1 (Dev3) | 671B MoE | SFT | reasoning samples | about 600k; rejection-sampled from the stage-1 RL checkpoint; only correct responses kept; part judged by DeepSeek-V3 against ground truth (Listing 4); CoT with mixed languages, long paragraphs, or code blocks removed | v2 B.3.3 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 (Dev3) | 671B MoE | SFT | non-reasoning samples | about 200k; portions of DeepSeek-V3 SFT data plus software-engineering data; V3-generated CoT for some tasks, no CoT for simple queries | v2 B.3.3 | verified 2026-09-14 | §4, Table 3: Dev2 → Dev3 AlpacaEval 2.0 55.8 → 62.1, Aider-Polyglot 25.6 → 44.8 |
| DeepSeek-R1 (Dev3) | 671B MoE | SFT | samples (avg tokens) by domain | Math 395,285 (6,094.2); Code 211,129 (7,435.7); STEM 10,124 (4,928.8); Logic 10,395 (2,739.0); General 177,812 (1,419.8); Total 804,745 (5,355.3) | v2 B.3.3 Table 5 | verified 2026-09-14 | not applicable |
| DeepSeek-R1 (Dev3) | 671B MoE | SFT | starting checkpoint | DeepSeek-V3-Base, not the RL checkpoint | v2 Fig. 2; B.4.2; v1 §2.3.3 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 | 671B MoE | SFT | data-creation compute | 5K GPU hours | v2 B.4.4 Table 7 | verified 2026-09-14 | not applicable |
| DeepSeek-R1 (Dev3 → R1) | 671B MoE | RL | temperature; other settings | 0.7; "retains most of the parameters from the first stage" | v2 §3.2.2 | verified 2026-09-14 | §3.2.2: higher temperatures in this stage lead to incoherent generation (no table) |
| DeepSeek-R1 (Dev3 → R1) | 671B MoE | RL | steps; preference-reward window | 1,700 steps; general instruction data and preference rewards only in the final 400 steps | v2 §3.2.2 | verified 2026-09-14 | B.5 Fig. 6: with the helpful RM, reward rises while Codeforces test pass@1 falls |
| DeepSeek-R1 (Dev3 → R1) | 671B MoE | RL | reward | rule reward (reasoning) + RM reward and format reward (general) + language reward | v2 §3.2.2 Eq. 8–10 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 (Dev3 → R1) | 671B MoE | RL | general prompts | 66k helpfulness questions; "additionally" 12,000 harmlessness questions | v2 B.3.1; Table 4 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 | 671B MoE | RL | compute (all R1 training) | same 64×8 H800 GPUs, "about 4 days, or roughly 80 hours"; 41K GPU hours | v2 B.4.4; Table 7 | verified 2026-09-14 | not applicable |
| DeepSeek-R1 helpful RM | R1 architecture + reward head | reward-model | data; training | 66,000 pairs (V3 judges each pair 4 times with positions randomized; kept if score difference Δ > 1; chosen/rejected lengths balanced); batch 256; LR 6e-6; 1 epoch; max length 8,192 | v2 §3.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1 safety RM | R1 architecture + reward head | reward-model | data; training | 106,000 prompts with safe/unsafe labels; point-wise loss; same hyper-parameters as helpful RM | v2 §3.1 | verified 2026-09-14 | no ablation reported |

## Distilled students and RL baselines

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| All six DeepSeek-R1-Distill models | 1.5B–70B | distill-SFT | data; epochs; schedule; context; batch | the ~800k B.3.3 set; 2–3 epochs; cosine decay to 1/10 of initial LR; 32,768 tokens; 64 | v2 B.4.3 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | distill-SFT | base; initial LR | Qwen2.5-Math-1.5B; 1×10⁻⁴ | v2 B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-7B | 7B | distill-SFT | base; initial LR | Qwen2.5-Math-7B; 8×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-14B | 14B | distill-SFT | base; initial LR | Qwen2.5-14B; 7×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Qwen-32B | 32B | distill-SFT | base; initial LR | Qwen2.5-32B; 6×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Llama-8B | 8B | distill-SFT | base; initial LR | Llama-3.1-8B; 5×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | no ablation reported |
| DeepSeek-R1-Distill-Llama-70B | 70B | distill-SFT | base; initial LR | Llama-3.3-70B-Instruct; 2×10⁻⁵ | v2 B.4.3 Table 6 | verified 2026-09-14 | base: v1 §2.4 says Llama-3.3 was chosen because its reasoning is "slightly better" than Llama-3.1; LR: no ablation reported |
| DeepSeek-R1-Zero-Qwen-32B (Table 16: "Qwen2.5-32B-Zero") | 32B | RL | LR; KL; temp; outputs; max length; questions/step; batch; ref refresh; steps; data | 2e-6; 0.001; 1; 16; 32,768; 32; 512; every 400 steps; over 10K steps; math, code, STEM | v2 B.4.1; F.1 | verified 2026-09-14 | not applicable |
| Qwen2-Math-7B-Zero | 7B | RL | steps | approximately 10,000 policy-gradient update steps | v2 F.1 | verified 2026-09-14 | not applicable |
| DeepSeek-R1 and baselines | — | eval-gate | decoding | temperature 0.6; top-p 0.95; k = 64 (AIME, GPQA), 16 (MATH, Codeforces), 8 (LiveCodeBench); max 32,768 tokens; pass@1 averaged over k | v2 D.1 | verified 2026-09-14 | D.1: greedy decoding gave higher repetition and checkpoint-to-checkpoint variability |
| DeepSeek-V3-Base and R1 post-training data | — | eval-gate | decontamination | 10-gram match filter on pre-training and post-training data; math SFT data and RL prompts from pre-2023 competitions | v2 D.1 | verified 2026-09-14 | D.1: ~six million potential math pre-training texts removed; paraphrases are not caught |
