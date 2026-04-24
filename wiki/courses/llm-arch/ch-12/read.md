# Chapter 12: Post-training and Alignment

<!-- scope: SFT, RLHF, DPO, Constitutional AI, alignment's architectural implications, reasoning-specific post-training
     deps: [[ch-11]]
     see-also: [[ch-23]], [[ch-19]]
-->

## Overview

Pre-training produces a capable but *uncontrolled* base model: it can continue any text pattern it has seen, but it cannot reliably follow instructions, refuse harmful requests, or reason through multi-step problems in a structured way. The gap between a base model and a useful assistant is bridged by **post-training** -- a suite of techniques that reshape the model's output distribution to align with human preferences without rewriting its weights from scratch.

This chapter traces the post-training pipeline from its simplest form (supervised fine-tuning on demonstrations) through reinforcement learning from human feedback (RLHF), its mathematically elegant alternative (DPO), Anthropic's scalable supervision framework (Constitutional AI), and finally the frontier of reasoning-specific post-training where RL teaches models *how* to think, not just *what* to say. The progression is not merely historical. Each method addresses a specific failure mode of the previous one, and understanding the chain of reasoning -- why SFT is insufficient, why RLHF is unstable, why DPO sidesteps that instability, why Constitutional AI scales supervision -- is essential for anyone designing or evaluating alignment strategies.

A recurring theme: post-training does not change the model's architecture or parameter count. It operates entirely within the representational capacity established during pre-training. Yet the behavioral difference between a base model and its post-trained variant is so dramatic that the two are effectively different systems. Understanding *why* this is possible -- and where it breaks down -- requires thinking carefully about what alignment actually modifies in the weight space.

---

## 1. Supervised Fine-Tuning (SFT): Teaching by Demonstration

The simplest post-training stage: collect high-quality (prompt, response) pairs and fine-tune the base model on them using the same cross-entropy loss from pre-training ([[ch-01]]).

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})$$

where $x$ is the prompt, $y$ is the target response, and $\pi_\theta$ is the model's policy (its conditional distribution over next tokens). This is identical to the pre-training objective -- the only difference is the *data distribution*. Instead of web text, the model trains on curated demonstrations of desired behavior.

### Chat Formatting and Instruction Structure

Modern SFT uses structured chat templates that separate system instructions, user messages, and assistant responses with special tokens:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Chat Template Structure (Llama 3 Style)</div>
<div style="font-family:monospace; font-size:12px; line-height:1.8;">
<span style="color:#e94560;">&lt;|begin_of_text|&gt;</span><br>
<span style="color:#e94560;">&lt;|start_header_id|&gt;</span><span style="color:#4ecdc4;">system</span><span style="color:#e94560;">&lt;|end_header_id|&gt;</span><br>
<span style="color:#888;">You are a helpful assistant.</span><span style="color:#e94560;">&lt;|eot_id|&gt;</span><br>
<span style="color:#e94560;">&lt;|start_header_id|&gt;</span><span style="color:#4ecdc4;">user</span><span style="color:#e94560;">&lt;|end_header_id|&gt;</span><br>
<span style="color:#888;">Explain the halting problem.</span><span style="color:#e94560;">&lt;|eot_id|&gt;</span><br>
<span style="color:#e94560;">&lt;|start_header_id|&gt;</span><span style="color:#4ecdc4;">assistant</span><span style="color:#e94560;">&lt;|end_header_id|&gt;</span><br>
<span style="color:#ffd93d;">The halting problem asks whether...</span><span style="color:#e94560;">&lt;|eot_id|&gt;</span>
</div>
<div style="color:#888; font-size:11px; margin-top:12px;">
Loss is computed only on assistant tokens (gray/yellow). System and user tokens are masked from the loss. This prevents the model from learning to generate user messages.
</div>
</div>

The loss masking is critical: gradients flow only through the assistant's response tokens, not the prompt tokens. Without this, the model would allocate capacity to predicting user messages -- a waste that dilutes the instruction-following signal.

### Data Quality vs Quantity

The InstructGPT work ([[instructgpt-rlhf|paper]]) used roughly 13,000 demonstration examples for SFT. The LIMA paper (Zhou et al., 2023) pushed this further, showing that just 1,000 carefully curated examples could produce a strong instruction-following model. The implication: **quality dominates quantity for SFT**. A small number of expert-written demonstrations teaches the model the *format* and *style* of helpful responses. The model's actual knowledge comes from pre-training; SFT merely teaches it to *express* that knowledge in the right structure.

But SFT has a fundamental limitation: it can only teach the model to imitate the *average* demonstrator. If the demonstrations come from humans with varying skill levels, the model learns to produce average-quality responses. Worse, SFT cannot teach the model to distinguish *better* from *worse* responses -- it treats all demonstrations as equally correct. This is the gap that preference optimization addresses.

---

## 2. RLHF: Optimizing for Human Preferences

Reinforcement Learning from Human Feedback ([[instructgpt-rlhf|paper]]) was the breakthrough that made ChatGPT possible. The core insight: instead of training the model to *imitate* good responses (SFT), train it to *distinguish* good from bad responses (reward model), then *optimize* for producing the good ones (RL).

### The Three-Stage Pipeline

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">RLHF Pipeline</div>
<div style="display:flex; gap:16px; flex-wrap:wrap; justify-content:center; align-items:flex-start;">
<div style="background:#16213e; border-radius:8px; padding:16px; width:180px; border-left:3px solid #4ecdc4;">
<div style="color:#4ecdc4; font-weight:bold; font-size:13px; margin-bottom:8px;">Stage 1: SFT</div>
<div style="color:#e0e0e0; font-size:11px; line-height:1.5;">Fine-tune base model on demonstrations. Produces $\pi^{\text{SFT}}$.</div>
</div>
<div style="color:#e94560; font-size:24px; display:flex; align-items:center;">&#8594;</div>
<div style="background:#16213e; border-radius:8px; padding:16px; width:180px; border-left:3px solid #ffd93d;">
<div style="color:#ffd93d; font-weight:bold; font-size:13px; margin-bottom:8px;">Stage 2: Reward Model</div>
<div style="color:#e0e0e0; font-size:11px; line-height:1.5;">Train $r_\phi(x, y)$ on human preference pairs: $y_w \succ y_l$ for each prompt $x$.</div>
</div>
<div style="color:#e94560; font-size:24px; display:flex; align-items:center;">&#8594;</div>
<div style="background:#16213e; border-radius:8px; padding:16px; width:180px; border-left:3px solid #e94560;">
<div style="color:#e94560; font-weight:bold; font-size:13px; margin-bottom:8px;">Stage 3: RL (PPO)</div>
<div style="color:#e0e0e0; font-size:11px; line-height:1.5;">Optimize $\pi_\theta$ to maximize $r_\phi$ while staying close to $\pi^{\text{SFT}}$ via KL penalty.</div>
</div>
</div>
</div>

### Stage 2: The Reward Model

The reward model learns a scalar score $r_\phi(x, y)$ indicating how good response $y$ is for prompt $x$. It is trained on human preference data: for each prompt, annotators compare two responses and indicate which is better. The reward model is optimized via the Bradley-Terry pairwise preference model:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

where $y_w$ is the preferred (winning) response, $y_l$ is the dispreferred (losing) response, and $\sigma$ is the sigmoid function. This loss pushes the reward model to assign higher scores to preferred responses. The reward model is typically initialized from the SFT model checkpoint with the final unembedding layer replaced by a scalar projection head.

InstructGPT collected approximately 33,000 comparison pairs from a team of 40 human labelers. The labelers ranked outputs from best to worst, generating $\binom{K}{2}$ comparisons from each $K$-way ranking.

### Stage 3: PPO Optimization

With the reward model fixed, the policy $\pi_\theta$ is optimized using Proximal Policy Optimization (PPO). The objective balances reward maximization against a KL penalty that prevents the policy from drifting too far from the SFT model:

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \left[r_\phi(x, y) - \beta \, D_{\text{KL}}\!\left(\pi_\theta(\cdot|x) \;\|\; \pi^{\text{SFT}}(\cdot|x)\right)\right]$$

The KL penalty $\beta$ is essential. Without it, the policy would "hack" the reward model -- finding inputs that produce high reward scores but correspond to degenerate, repetitive, or nonsensical text. The reward model is an imperfect proxy for human preferences, and any optimizer powerful enough to be useful is powerful enough to exploit those imperfections.

**Why PPO and not simpler RL algorithms?** PPO clips the policy gradient update to prevent catastrophically large steps:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\!\left(\rho_t \hat{A}_t, \; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t\right)\right]$$

where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ is the probability ratio and $\hat{A}_t$ is the advantage estimate. The clipping ensures the policy changes gradually, which is critical for language models where a single bad update can permanently damage text quality.

### Why RLHF Is Hard

The InstructGPT result was striking: the 1.3B parameter InstructGPT model was preferred by humans over the 175B GPT-3 base model, despite having 100x fewer parameters. Alignment quality can matter more than raw scale.

But RLHF has serious practical difficulties:

1. **Reward model fragility.** The reward model is trained on a finite, biased sample of human preferences. It generalizes imperfectly, creating exploitable gaps.
2. **Training instability.** PPO requires careful tuning of learning rate, KL coefficient $\beta$, clipping ratio $\epsilon$, batch size, and the number of PPO epochs per batch. The interaction between these hyperparameters is poorly understood.
3. **Infrastructure complexity.** RLHF requires running four models simultaneously: the policy being trained, the reference policy (for KL computation), the reward model, and the value function (critic). This roughly quadruples memory requirements compared to SFT.
4. **Reward over-optimization.** As training progresses, the policy finds increasingly effective ways to maximize the reward model's score without actually improving response quality -- a phenomenon called Goodhart's Law applied to learned reward functions.

See [RLHF Pipeline Deep Dive](excerpts/rlhf-pipeline.md) for a detailed walkthrough of the training dynamics.

---

## 3. DPO: Alignment as Supervised Learning

Direct Preference Optimization ([[dpo|paper]]) made a remarkable theoretical observation: the standard RLHF objective has a *closed-form solution*, and the optimal policy can be expressed directly in terms of the preference data without ever training a reward model.

### The Key Derivation

Start from the KL-constrained reward maximization objective:

$$\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)} \left[r(x, y)\right] - \beta \, D_{\text{KL}}\!\left(\pi(\cdot|x) \;\|\; \pi_{\text{ref}}(\cdot|x)\right)$$

The optimal policy for this objective has a closed-form solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

where $Z(x)$ is a normalizing partition function. Rearranging this for the reward:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

This is the critical insight: **the reward function is implicitly defined by the ratio of the optimal policy to the reference policy**. You do not need a separate reward model -- the policy *is* the reward model.

Substituting this implicit reward into the Bradley-Terry preference model yields the DPO loss:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

This is a binary cross-entropy loss on preference pairs. No reward model, no RL loop, no PPO hyperparameters, no value function. The entire alignment procedure reduces to supervised learning on pairs of preferred and dispreferred responses.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">RLHF vs DPO Pipeline Comparison</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Dimension</th>
<th style="text-align:center; padding:8px; color:#ffd93d;">RLHF (PPO)</th>
<th style="text-align:center; padding:8px; color:#4ecdc4;">DPO</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Stages after SFT</td>
<td style="text-align:center; padding:8px;">2 (reward model + RL)</td>
<td style="text-align:center; padding:8px;">1 (preference loss)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Models in memory</td>
<td style="text-align:center; padding:8px;">4 (policy, ref, reward, critic)</td>
<td style="text-align:center; padding:8px;">2 (policy, ref)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">On-policy sampling</td>
<td style="text-align:center; padding:8px;">Required (expensive)</td>
<td style="text-align:center; padding:8px;">Not required</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Hyperparameters</td>
<td style="text-align:center; padding:8px;">Many ($\beta$, $\epsilon$, lr, epochs, batch)</td>
<td style="text-align:center; padding:8px;">Few ($\beta$, lr)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Training stability</td>
<td style="text-align:center; padding:8px;">Fragile</td>
<td style="text-align:center; padding:8px;">Stable</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Quality ceiling</td>
<td style="text-align:center; padding:8px;">Higher (on-policy exploration)</td>
<td style="text-align:center; padding:8px;">Bounded by offline data</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
DPO's quality ceiling is bounded by the preference dataset: it cannot discover response strategies beyond what the dataset contains. RLHF's on-policy sampling can find novel high-reward responses.
</div>
</div>

### Why DPO Works (and Where It Doesn't)

DPO's stability advantage comes from eliminating the reward model as an intermediate representation. In RLHF, errors in the reward model are amplified by the RL optimizer. In DPO, the preference signal flows directly into the policy through a well-understood supervised loss.

But DPO has a limitation: it is **off-policy**. The preference pairs in the training data were generated by some behavior policy (often the SFT model), not by the current policy being trained. As the policy improves during training, the training data becomes stale -- the model is being trained on comparisons between responses it would no longer generate. RLHF's on-policy sampling avoids this by continuously generating fresh responses from the current policy.

This is why several production systems use iterative DPO (also called online DPO): periodically regenerate preference data using the current policy, then run another round of DPO. This approximates the on-policy exploration of RLHF while retaining DPO's simplicity.

A complementary direction is **distilling alignment from larger teacher models**. Zephyr ([[zephyr|paper]]) showed that the entire DPO pipeline can run on AI-generated data: fine-tune on teacher completions (dSFT), then run DPO on GPT-4-ranked preference pairs (dDPO). The resulting 7B Zephyr-β beat Llama-2-Chat-70B on MT-Bench with no human annotation and under a day of training — a demonstration that DPO plus AI feedback removes the human-labeling bottleneck from the alignment pipeline entirely.

See [DPO Derivation and Variants](excerpts/dpo-derivation.md) for the complete mathematical derivation and a survey of DPO variants (IPO, KTO, ORPO).

---

## 4. Constitutional AI: Scaling Supervision with Principles

Constitutional AI ([[constitutional-ai|paper]]) addresses a different bottleneck: **human labeling does not scale**. RLHF and DPO both require human-generated preference data, which is expensive, slow, and bottlenecked by the number of qualified annotators. Anthropic's Constitutional AI (CAI) replaces human preference labels with AI-generated ones, guided by a set of explicit constitutional principles.

### The Two-Phase Method

**Phase 1: Supervised Self-Critique (SL-CAI).** The model generates responses, then critiques and revises them using constitutional principles:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Constitutional AI: Self-Critique Loop</div>
<div style="display:flex; flex-direction:column; gap:12px; align-items:center;">
<div style="background:#16213e; padding:12px 24px; border-radius:8px; color:#e0e0e0; font-size:12px; width:90%; max-width:500px;">
<span style="color:#e94560; font-weight:bold;">Prompt:</span> "How do I pick a lock?"<br>
<span style="color:#ffd93d; font-weight:bold;">Initial response:</span> "Here's how to pick a lock: First, get a tension wrench..."
</div>
<div style="color:#e94560; font-size:16px;">&#8595;</div>
<div style="background:#16213e; padding:12px 24px; border-radius:8px; color:#e0e0e0; font-size:12px; width:90%; max-width:500px; border-left:3px solid #4ecdc4;">
<span style="color:#4ecdc4; font-weight:bold;">Critique (principle: "Choose the response that is least likely to be used for illegal activity"):</span><br>
"This response provides detailed instructions that could facilitate illegal entry..."
</div>
<div style="color:#e94560; font-size:16px;">&#8595;</div>
<div style="background:#16213e; padding:12px 24px; border-radius:8px; color:#e0e0e0; font-size:12px; width:90%; max-width:500px; border-left:3px solid #ffd93d;">
<span style="color:#ffd93d; font-weight:bold;">Revision:</span> "Lock picking is a skill used by locksmiths. If you're locked out, I'd recommend contacting a licensed locksmith..."
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:12px; text-align:center;">
The model itself generates critique and revision. No human labeler sees harmful content. The revised (prompt, response) pairs become SFT training data.
</div>
</div>

The revised responses are used as SFT training data, producing a model that is already substantially more harmless than the initial model.

**Phase 2: RLAIF (RL from AI Feedback).** Instead of humans comparing response pairs, the model itself judges which response better satisfies the constitutional principles. These AI-generated preference labels train a reward model, which then guides RL fine-tuning -- the same PPO loop as RLHF, but with AI preferences instead of human preferences.

### Why Constitutional AI Matters

1. **Scalability.** Human labeling is expensive and slow. AI feedback can generate millions of comparisons at the cost of inference compute.
2. **Consistency.** Human annotators disagree substantially on edge cases. A fixed set of constitutional principles provides more consistent judgments.
3. **Transparency.** The constitutional principles are explicit and auditable. You can inspect *why* the model makes a particular safety judgment by reading the chain-of-thought critique.
4. **Harmless but not evasive.** A key finding: CAI-trained models engage with harmful queries by explaining their objections rather than issuing blanket refusals. This is because the constitutional principles can encode nuance that simple "refuse all harmful content" labels cannot.

The chain-of-thought reasoning during critique is not just a training trick -- it improves the quality of the AI's judgments. The paper showed that having the model reason through its critique before making a preference judgment produces better alignment outcomes than direct judgment without reasoning. This foreshadows the reasoning-specific post-training discussed in Section 6.

See [Constitutional AI Deep Dive](excerpts/constitutional-ai-details.md) for the full principle taxonomy and RLAIF mechanics.

---

## 5. Architecture Implications of Alignment

A fundamental question: does post-training change what the model *can represent*, or only what it *chooses to express*?

### The Representation View

The linear representation hypothesis suggests that concepts are encoded as directions in the model's activation space. Under this view, a base model already contains directions corresponding to "helpful," "harmful," "truthful," "deceptive," etc. -- it learned these concepts from pre-training data that contains examples of all these behaviors. Alignment does not create new representational capacity; it *steers* the model's output distribution toward the desirable directions.

Evidence for this view:

- **Representation engineering** (Zou et al., 2023) can find "honesty" and "harmlessness" directions in activation space and steer model behavior by adding or subtracting these direction vectors, without any fine-tuning.
- **Jailbreaks** work by finding prompts that circumvent alignment training, causing the base model's unfiltered capabilities to surface. If alignment had deleted the capability to produce harmful content, jailbreaks would be impossible.
- **The "alignment tax" is small.** InstructGPT showed minimal regression on standard NLP benchmarks after RLHF. If alignment fundamentally restructured the model's representations, we would expect larger capability losses.

### What Alignment Actually Modifies

Post-training primarily modifies the **later layers** of the network -- particularly the layers closest to the output. Mechanistic interpretability work suggests that:

- Early/middle layers encode factual knowledge and linguistic structure (relatively unchanged by alignment)
- Late layers encode output formatting, style, and safety judgments (substantially modified by alignment)
- The attention patterns in late layers shift to attend more to instruction-relevant tokens after SFT

This is consistent with the "residual stream" view from [[ch-03]]: pre-training fills the residual stream with knowledge, and post-training adjusts the final readout to select appropriate content from that knowledge.

### The Parameter-Efficiency Argument

LoRA (Low-Rank Adaptation) and QLoRA demonstrate that alignment can be achieved by modifying a tiny fraction of the model's parameters. LoRA adapts a model by adding low-rank matrices $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ with $r \ll d$. Typical alignment LoRA uses $r = 64$, modifying fewer than 0.1% of the model's parameters. The fact that this works at all strongly suggests alignment operates in a low-dimensional subspace of weight space -- it is a *perturbation* of the pre-trained model, not a restructuring.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Post-Training Parameter Modification Scale</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Method</th>
<th style="text-align:right; padding:8px;">Params Modified</th>
<th style="text-align:right; padding:8px;">% of Total</th>
<th style="text-align:left; padding:8px;">Implication</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Full SFT</td>
<td style="text-align:right; padding:8px;">All</td>
<td style="text-align:right; padding:8px;">100%</td>
<td style="padding:8px;">Unconstrained; risks catastrophic forgetting</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">Full RLHF</td>
<td style="text-align:right; padding:8px;">All</td>
<td style="text-align:right; padding:8px;">100%</td>
<td style="padding:8px;">KL penalty constrains effective change</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">LoRA SFT (r=64)</td>
<td style="text-align:right; padding:8px;">~100M (for 70B)</td>
<td style="text-align:right; padding:8px;">~0.1%</td>
<td style="padding:8px;">Alignment lives in a low-rank subspace</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#888; font-weight:bold;">Representation steering</td>
<td style="text-align:right; padding:8px;">0</td>
<td style="text-align:right; padding:8px;">0%</td>
<td style="padding:8px;">Inference-time only; no weight change</td>
</tr>
</tbody>
</table>
</div>

---

## 6. Reasoning-Specific Post-Training

The methods above (SFT, RLHF, DPO, CAI) teach models *what to say*. A new frontier of post-training teaches models *how to think*. DeepSeek-R1 ([[deepseek-r1|report]]) and Qwen 3 ([[qwen-3|report]]) represent two approaches to this problem.

### DeepSeek-R1: Emergent Reasoning from Pure RL

DeepSeek-R1-Zero made a startling demonstration: a model trained with **pure RL and no supervised fine-tuning** spontaneously developed chain-of-thought reasoning, self-verification, and dynamic strategy adaptation. The model was never shown examples of reasoning traces -- it discovered these behaviors through reward optimization alone.

**GRPO (Group Relative Policy Optimization)** replaces PPO's critic model with a simpler group-based advantage estimation:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}} \; \mathbb{E}_{\{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[\sum_{i=1}^G \min\!\left(\rho_i \hat{A}_i, \; \text{clip}(\rho_i, 1\!-\!\epsilon, 1\!+\!\epsilon)\,\hat{A}_i\right) - \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

For each question $q$, GRPO samples a **group** of $G$ outputs, computes rewards $r_1, \ldots, r_G$ (using rule-based verification for math/code), and normalizes within the group:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

This eliminates the critic model entirely, reducing the number of models in memory from four (PPO) to two (policy + reference). The group normalization provides a natural baseline without requiring a learned value function.

**The "aha moment."** During R1-Zero training, the model spontaneously learned to reconsider its approach mid-solution, producing utterances like "Wait, let me reconsider this..." This self-correction behavior was not present in the training data and emerged purely from reward optimization. The model learned that allocating more compute to harder problems (longer reasoning chains) produced higher rewards.

**The four-stage pipeline of R1** addresses R1-Zero's weaknesses (poor formatting, language mixing) by adding targeted SFT:

1. **Cold-start SFT:** Thousands of curated long-CoT examples to establish the reasoning format
2. **Reasoning RL:** GRPO on math, coding, science with rule-based rewards
3. **Rejection sampling + SFT:** 600K reasoning + 200K non-reasoning samples
4. **All-scenario RL:** Broad RL with both rule-based and preference-based rewards

### Qwen 3: Dual-Mode Thinking in a Single Model

Qwen 3 ([[qwen-3|report]]) takes a different approach: train a single model that can operate in **thinking mode** (extended chain-of-thought) or **non-thinking mode** (direct response), controlled by a flag at inference time.

The four-stage post-training pipeline achieves this:

1. **Long-CoT cold start:** Initialize the thinking pattern (similar to R1's Stage 1)
2. **Reasoning RL:** GRPO on just 3,995 query-verifier pairs -- remarkably data-efficient
3. **Thinking mode fusion:** Integrate thinking and non-thinking capabilities without mode confusion
4. **General domain RL:** Broaden to all task types

The key architectural implication: **inference-time compute allocation is a training decision, not just a prompting strategy.** The model learns *when* to think deeply (hard reasoning problems) and when to respond directly (simple queries), and users can override this with a thinking budget that caps the number of reasoning tokens.

### Why Rule-Based Rewards Work for Reasoning

Both R1 and Qwen 3 use **rule-based rewards** (exact match, test case execution) rather than learned reward models for reasoning tasks. This is a deliberate design choice:

- Math and code have **verifiable correctness**, providing clean, unambiguous reward signal
- Learned reward models are poor at judging multi-step reasoning -- they tend to reward *confident-sounding* reasoning over *correct* reasoning
- Rule-based rewards prevent reward hacking because the reward function is not learned and therefore cannot be exploited

The limitation: rule-based rewards only work for domains where correctness can be automatically verified. For open-ended tasks (creative writing, nuanced advice), learned reward models or AI feedback (CAI-style) remain necessary.

For an interactive exploration of how these training stages interact, see [figures/post-training-pipeline.html](figures/post-training-pipeline.html). For a comparison of reasoning RL approaches, see [figures/reasoning-rl-comparison.html](figures/reasoning-rl-comparison.html).

---

## Core Insights from the Literature

### Insight 1: Alignment quality can outweigh scale
**Paper:** Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" ([[instructgpt-rlhf|paper]])

InstructGPT's 1.3B parameter model was preferred by humans over the 175B GPT-3 -- a 100x parameter disadvantage overcome by alignment alone. This result reframed the field's understanding of what makes a model useful: raw capability from pre-training is necessary but insufficient. The three-stage pipeline (SFT, reward modeling, PPO) became the standard recipe and directly produced ChatGPT. The "alignment tax" (capability regression from alignment training) proved small, meaning alignment is nearly free in terms of benchmark performance. **Guideline:** After pre-training, always invest in preference-based alignment. The behavioral improvement dramatically exceeds the small capability cost. A well-aligned small model can outperform an unaligned large model for user-facing applications.

### Insight 2: The RLHF objective has a closed-form solution
**Paper:** Rafailov et al., "Direct Preference Optimization" ([[dpo|paper]])

DPO's key theoretical contribution is showing that the reward function is implicitly defined by the policy-to-reference ratio, eliminating the reward model as an explicit intermediate representation. This transforms alignment from a complex RL problem into a supervised classification problem -- a binary cross-entropy loss on preference pairs. The simplicity gains are not just engineering convenience; they improve stability by removing the reward model as a source of error amplification. DPO's limitation -- off-policy data staleness -- is addressable through iterative regeneration, making online DPO a strong practical default. **Guideline:** For most alignment tasks, start with DPO rather than full RLHF. The simplicity and stability advantages are substantial, and iterative DPO closes most of the quality gap with on-policy methods.

### Insight 3: AI self-critique scales alignment supervision
**Paper:** Bai et al., "Constitutional AI: Harmlessness from AI Feedback" ([[constitutional-ai|paper]])

Constitutional AI demonstrated that AI-generated preference labels, guided by explicit principles, can replace much of the human labeling required for alignment. The key innovation is not just cost reduction but *transparency*: the constitutional principles are auditable, and the chain-of-thought critiques make the model's safety reasoning inspectable. The resulting models are harmless but not evasive -- a qualitative improvement over simple refusal-based safety training. RLAIF has since been adopted across the industry, and the principle that AI systems can help supervise other AI systems is central to scaling alignment. **Guideline:** Define explicit constitutional principles for your alignment goals. Use AI feedback (RLAIF) to scale beyond human labeling bottlenecks, but validate AI preferences against human judgments on a held-out set to catch systematic biases.

### Insight 4: Reasoning can emerge from pure RL without supervised examples
**Paper:** DeepSeek AI, "DeepSeek-R1" ([[deepseek-r1|report]])

R1-Zero proved that chain-of-thought reasoning, self-verification, and dynamic compute allocation can emerge spontaneously from RL training with verifiable rewards -- no human-written reasoning traces required. This is a fundamental finding about how reasoning capabilities arise in LLMs: they are incentivized by reward structure, not taught by imitation. GRPO's critic-free design made this practical by reducing memory from four models to two. The "aha moment" -- spontaneous mid-solution reconsideration -- suggests that reasoning is a *learnable strategy* for reward maximization, not a fixed capability. **Guideline:** For reasoning domains with verifiable correctness (math, code), RL with rule-based rewards can discover reasoning strategies that supervised data cannot teach. Use GRPO over PPO to reduce memory requirements.

---

## Key Takeaways

1. **Post-training is a pipeline, not a single step.** Modern alignment combines SFT (format/style), preference optimization (quality/safety), and potentially reasoning RL (thinking depth). Each stage addresses a specific gap.

2. **SFT teaches format, not quality.** Supervised fine-tuning on demonstrations teaches the model *how* to structure responses. Quality distinctions require preference-based methods (RLHF, DPO) that can distinguish better from worse.

3. **DPO reduces alignment to supervised learning.** By showing the reward is implicitly defined by the policy-reference ratio, DPO eliminates the reward model and RL loop entirely. The tradeoff is off-policy data staleness, addressable through iterative regeneration.

4. **The KL penalty is not optional.** In both RLHF and DPO, the divergence constraint from the reference policy prevents reward hacking and catastrophic forgetting. The coefficient $\beta$ controls the alignment-capability tradeoff.

5. **Constitutional AI scales supervision.** Using AI feedback guided by explicit principles replaces the human labeling bottleneck while maintaining transparency through chain-of-thought critiques.

6. **Alignment does not restructure representations.** Post-training operates in a low-dimensional subspace of weight space, primarily modifying late layers. The base model's knowledge and capabilities are preserved; alignment steers the readout.

7. **Reasoning emerges from reward structure.** DeepSeek-R1 proved that chain-of-thought reasoning can emerge from pure RL without supervised examples, given verifiable rewards. This opens post-training from "what to say" to "how to think."

8. **Inference-time compute allocation is a training decision.** Qwen 3's dual-mode training shows that whether a model thinks slowly or responds quickly can be learned during post-training rather than imposed at inference time.

---

## References

- [[instructgpt-rlhf|Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (2022) (paper)]] -- RLHF, InstructGPT
- [[dpo|Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023) (paper)]] -- DPO
- [[zephyr|Tunstall et al., "Zephyr: Direct Distillation of LM Alignment" (2023) (paper)]] -- dSFT + dDPO, AI-feedback alignment distillation
- [[constitutional-ai|Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022) (paper)]] -- Constitutional AI, RLAIF
- [[deepseek-r1|DeepSeek AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025) (report)]] -- GRPO, emergent reasoning
- [[qwen-3|Qwen Team, "Qwen3 Technical Report" (2025) (report)]] -- dual-mode thinking, unified post-training
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) -- PPO
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) -- parameter-efficient alignment
- Zou et al., "Representation Engineering" (2023) -- activation-space steering
- Zhou et al., "LIMA: Less Is More for Alignment" (2023) -- data efficiency for SFT
