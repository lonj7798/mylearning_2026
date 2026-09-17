---
chapter: ch-29d
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/**/userrl.md on 2026-09-15)
source_url: https://arxiv.org/abs/2509.19736
source_version: arXiv:2509.19736v1 (2025-09-24)
created_at: "2026-09-15"
---

# Excerpt: UserRL: Training Interactive User-Centric Agent via Reinforcement Learning (Qian, Liu, Prabhakar, et al.; Salesforce AI Research, UIUC)

Facts used by [[read]] §1, §2, and the Recipe table. Read in the arXiv v1 PDF on 2026-09-15. Code: github.com/SalesforceAIResearch/UserRL.

## Gyms and simulated users (§3, Tables 1-2)
- Eight gyms; each has a rule-based task automaton and an LLM user simulator: "the user responses remain dynamic and contextually adaptive, while the underlying task completion remains strictly rule-based" (§3).
- One tool interface with three operations: Action (talk to the simulated user), Search, Answer (§3).
- Train/test tasks: Travel 925/471, Turtle 423/48, Function 460/78, Tau 500/165, Persuade 378/42, Intention 380/40, Telepathy 360/41, Search 0/125 (Table 2).
- Training gyms: Travel, Turtle, Function, Tau, Persuade. Held-out gyms: Intention, Telepathy, Search, "entirely held-out evaluation environments" (§5.1).

## Reward shaping (§4)
- Turn-level shaping: Naive r̃_t = r_t; Equalized r̃_t = c; Reward-to-Go r̃_t = Σ_{j=t..T} γ^{j−t} r_j; Exponential Mapping r̃_t = 0.5 + 0.5·(1 − exp(−k r_t))/(1 − exp(−k)) (§4.2).
- Trajectory score: Sum R = Σ_t r_t, or R2G R = Σ_j γ^{j−1} r_j (§4.3). Advantage A = (r̃_t − μ_Q)/(σ_Q + η) using group statistics of the trajectory scores (§4.4). The KL loss is omitted (§4.4).
- Naive shaping was excluded because "its sparse effective reward signal, where many turns yield zero reward, quickly leads to training collapse" (§5.1).

## SFT cold start and simulator choice (§5.1, App. B, App. C)
- SFT data: GPT-4o acts as both agent and simulated user on the five training gyms; trajectories are ranked and the top-K per gym form a 1k-sample set (App. B).
- Training simulator Qwen3-32B; evaluation simulator GPT-4o. Justification: one training run issues about 2k × 15 epochs × 8 rollouts × 16 turns ≈ 4M simulator requests (App. C).
- Result: GPT-4o as the training simulator "generally yields higher performance" than Qwen3-32B; models trained with Qwen3-32B "can still transfer well" to GPT-4o at evaluation (§5.3, Figure 2 right). The authors attribute part of the GPT-4o gain to using the same simulator in training and evaluation (Interpretation).
- SFT cold start: without it RL plateaus early; with it "in some tasks exceeding 100% gains" (§5.3, Figure 2 left, Figure 3).

## Main results (Table 3, micro-averaged over 8 gyms)
- Qwen3-8B Equalized/R2G 0.5652; Equalized/Sum 0.5076. Qwen3-4B Equalized/R2G 0.5269; Equalized/Sum 0.4656.
- Raw models: Qwen3-32B 0.3128, Qwen3-4B 0.2929. Closed: Gemini-2.5-Pro 0.4702, GPT-4o 0.4449.
- Held-out gyms for Qwen3-8B Equalized/R2G: Intention 1.8175, Telepathy 0.5610, Search 0.8880; raw Qwen3-4B: 1.7400, 0.4878, 0.8560. (IntentionGym scores are sums of turn rewards and exceed 1.)

## Real users (§5.3 Table 5, App. C)
- Five computer-science PhD students replaced the simulator on TurtleGym and TelepathyGym. Qwen3-4B: Turtle 0.1844 (GPT-4o user) → 0.2952 (real users); Telepathy 0.6098 → 0.7805. Qwen3-8B: 0.1854 → 0.3127; 0.5610 → 0.7805.
- Reason given: GPT-4o users reply with brief "Yes", "No", or "Maybe", while real users "sometimes offered subtle hints rather than simply judging responses" although instructed not to leak answers (§5.3; App. C). The authors call this "only a preliminary test".

## Configuration (App. B, Table 6)
- RL: gamma 0.8; k 2.0; batch 128; max prompt 1152; max response 8192; LR 1e-6; PPO minibatch 16; KL loss off; entropy coefficient 0; SGLang rollouts; n = 8; max 16 turns; 15 epochs; 8 H200 GPUs, about 1.5 days; best checkpoint chosen on a 5% validation split.
- SFT: full fine-tuning; cutoff 16384; per-device batch 2 × gradient accumulation 4; LR 1.0e-5; 3 epochs; cosine; warmup ratio 0.1; bf16; 4 H200 GPUs, about 1 hour.
