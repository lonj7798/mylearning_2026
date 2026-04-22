<!-- scope: InstructGPT — the canonical three-stage RLHF recipe (SFT → RM → PPO-ptx)
     deps: [[ppo]], [[bradley-terry-rm]]
     see-also: [[dpo]], [[rlaif-scaling]], [[constitutional-ai]], [[llama-2]]
-->

# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
- **Core Insight:** Three stages — supervised demonstrations, a preference-ranked reward model, and PPO against that RM with a token-level KL penalty to the SFT policy — turn a raw LM into an instruction-follower that a 1.3B RLHF'd model beats a 175B vanilla model.
- **Guideline:** Freeze an SFT π_ref, train a 6B Bradley-Terry RM on K=4–9 ranked completions, then PPO with β KL penalty per token plus a pretraining-mix term (γ L_ptx) to prevent alignment tax.
- **Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2203.02155
- **Relevant topics:** RLHF template, reward modeling, alignment, PPO-ptx, instruction following

## Abstract
Making language models bigger does not make them better at following user intent. We show that aligning LMs with user intent via fine-tuning on human feedback can improve helpfulness, honesty, and harmlessness across a wide range of tasks. Labelers write demonstrations of desired behavior and rank model outputs; we SFT GPT-3 on demonstrations, train a reward model on comparisons, and fine-tune the policy with PPO against the RM. The 1.3B InstructGPT is preferred over the 175B GPT-3 while making up fewer factual mistakes and generating less toxic content.

## Key Contributions
- Template for modern RLHF: (SFT) → (RM) → (PPO).
- **PPO-ptx**: mixes pretraining gradient into PPO loss to prevent regression on public NLP benchmarks.
- Per-token KL penalty baked into the reward: `r(x,y) = r_φ(x,y) − β · KL(π_RL || π_SFT)`.
- Demonstrates that RLHF generalizes to held-out instructions.
- Released the prompt taxonomy and labeler guidelines.

## Key Figures/Tables to Study
- **Figure 2:** The three-stage pipeline — the single most-reproduced alignment diagram.
- **Figure 3:** Labeler win-rate curves — 1.3B PPO > 175B base.
- **Section 3.5 / Equation 2:** The PPO-ptx objective.
- **Table 6:** RealToxicityPrompts results — RLHF reduces toxicity.

## Technical Details

### Stage 1 — Supervised fine-tuning
- 13K prompts with labeler-written demonstrations.
- Standard cross-entropy, 16 epochs, cosine LR decay, dropout 0.2. Select by RM score on val.

### Stage 2 — Reward model (6B)
- 33K prompts × K rankings (K ∈ {4..9}) → `C(K,2)` pairs per prompt.
- Bradley-Terry pairwise loss:
  `L_RM(φ) = −(1 / C(K,2)) E_{(x,y_w,y_l)~D}[ log σ(r_φ(x,y_w) − r_φ(x,y_l)) ]`
- Train all pairs from the same prompt in the same minibatch (otherwise overfits quickly).

### Stage 3 — PPO-ptx (Equation 2)
`objective(φ) = E_{(x,y)~D_RL}[ r_φ(x,y) − β log(π_φ^RL(y|x) / π^SFT(y|x)) ] + γ · E_{x~D_pretrain}[ log π_φ^RL(x) ]`
- First term = RM score.
- `β log(π/π_ref)` = per-token KL penalty folded into the reward (not into the loss) — "KL-control" / "KL-reward" style.
- `γ · L_ptx` = pretraining loss mixed back in; prevents alignment tax.
- Optimized with standard PPO-clip (ε=0.2) over this shaped reward.

### Canonical hyperparameters
| Knob | Value |
|------|-------|
| SFT LR | 9.65e-6 (cosine) |
| SFT epochs | 16 |
| RM size | 6B |
| RM LR | 9e-6 |
| PPO LR | 1.41e-5 (fixed) |
| PPO batch size | 512 prompts |
| PPO rollout length | ≤ 2048 tokens |
| KL coef β | 0.02 (adaptive controller optional) |
| Pretraining coef γ | 27.8 (InstructGPT-ptx) or 0 (InstructGPT) |
| Clip ε | 0.2 |
| Epochs per rollout | 4 |

### Entropy handling
No explicit entropy bonus in the objective; the β KL term to π_SFT serves the same regularizing role. Entropy collapse is instead tracked as a failure signal and controlled via β annealing.

## Connections
- RL optimizer: [[ppo]].
- RM foundation: [[bradley-terry-rm]].
- Direct-optimization replacement: [[dpo]].
- Safety-focused successor: [[constitutional-ai]], [[rlaif-scaling]].
- Implementation walkthrough: [[hf-rlhf-illustrated]], [[costa-huang-ppo-details]].
- LLaMA / Anthropic variants: [[llama-2]], [[constitutional-ai]].
