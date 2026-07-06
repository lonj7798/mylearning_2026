<!-- scope: Thinking Machines' on-policy distillation — student samples own trajectories, teacher grades each token via per-token reverse KL
     deps:
     see-also: ross-dagger-exposure-bias, agarwal-gkd -->
# On-Policy Distillation (Thinking Machines Lab, 2025)
- **Core Insight:** Sample trajectories from the *student*, then use a strong teacher to grade *every token* with per-token reverse KL — combining on-policy relevance (RL) with dense per-token supervision (distillation) to reach teacher-level skill at a fraction of RL's cost.
- **Guideline:** When a capable teacher already exists for the target skill, prefer on-policy distillation over high-compute RL: reuse your RL loop but replace the sparse reward with per-token negative reverse KL against the teacher.
- **Source:** https://thinkingmachines.ai/blog/on-policy-distillation/ (Kevin Lu & Thinking Machines Lab, 2025). Numeric benchmark rows attributed to the Qwen3 Technical Report (Table 21).
- **Relevant chapters:** on-policy vs off-policy distillation; reverse-KL objectives; RL-vs-distillation compute; continual learning / personalization

## Definition
> "The core idea of on-policy distillation is to sample trajectories from the _student_ model and use a high-performing teacher to grade _each token_ of each trajectory." — Kevin Lu / TML

On-policy = train on samples the student itself generates; off-policy = train on teacher-generated samples. On-policy fixes the distribution-shift / compounding-error problem of pure off-policy imitation:
> "The drawback of off-policy training is that the student learns in contexts frequented by the teachers, not ones the student itself will often find itself in. This can cause compounding error: if the student makes an early mistake that the teacher never makes, it finds itself diverging ever farther from the states it observed in training." — TML
> "The strength of on-policy training is that by training on samples from itself, the student learns to avoid mistakes in a more direct way." — TML

## The objective — per-token reverse KL
Grade each token by the reverse KL between student and teacher:

    KL( pi_theta || pi_teacher )
      = E_{x ~ pi_theta} [ log pi_theta(x_{t+1} | x_{1..t}) - log pi_teacher(x_{t+1} | x_{1..t}) ]

Two properties motivate reverse KL:
> "the reverse KL is "unhackable" in the sense that low KL always corresponds to a high probability of desirable behavior from the teacher model's point of view." — TML
> "it is "mode seeking"—it learns one specific behavior (the teacher's) instead of spreading its distribution across several suboptimal options" — TML

## Dense supervision: O(N) bits vs RL's O(1) bits
> "RL provides very sparse feedback, teaching a fixed number of bits per training episode regardless of the number of tokens used." — TML
> "Distillation teaches O(N) bits per episode, where N is the number of tokens." — TML

Per-token grading is why distillation is far more sample/compute-efficient than reward-at-the-end RL, and why shorter or partial rollouts still yield signal.

## 4-step implementation (reuses the RL loop)
1. **Initialize teacher client.**
2. **Sample trajectories** from the student; capture the student's own logprobs.
3. **Compute reward:** "We query the teacher client with `compute_logprobs` on the sampled trajectories."
4. **Loss:** "We set the per-token advantage to the negative reverse KL, and call the RL importance-sampling loss function."

So on-policy distillation drops into an existing RL/importance-sampling trainer — swap the sparse scalar reward for a dense per-token advantage = −(reverse KL to teacher).

## Key numbers (math reasoning: Qwen3-8B-Base student, Qwen3-32B teacher)
> "We use distillation to train mathematical reasoning in the Qwen3-8B-Base model, using Qwen3-32B as a teacher model." — TML
> "Training the student (Qwen3-8B-Base) on 400k prompts with full fine-tuning achieves a score of 60% on AIME'24" — TML (off-policy SFT baseline)

From the Qwen3 Technical Report (Table 21), starting from the distilled checkpoint on AIME'24:
- Off-policy distillation: **55.0%**
- + Reinforcement learning: **67.6%** at **17,920 GPU-hours**
- + On-policy distillation: **74.4%** at **1,800 GPU-hours** → roughly **10× cheaper than RL** for a better score

Efficiency claims:
- vs SFT data-scaling: "We find a baseline cost reduction of 9x when the SFT dataset is given" (up to ~9–30× depending on assumptions).
- Self-distillation: "on-policy distillation reaches the teacher's level of performance approximately 7-10x faster than RL"; "Cumulatively, the reduction in compute required is on the order of 50-100x."

## Personalization / continual learning
Mid-training on internal company documents *degrades* instruction-following (IF-eval), and no data-mix weighting fully preserves it:
> "Although mixing in at least 30% of chat data helps preserve most instruction-following ability, there is no weighting which maintains the original performance on IF-eval." — TML

On-policy distillation *recovers* the lost behavior without erasing the newly learned knowledge:
> "After fine-tuning on an 70-30 mix of internal document data and chat data, on-policy distillation recovers nearly full performance on IF-eval without losing any knowledge" — TML

## Continual-learning insight: on-policy SFT drifts to off-policy
Even distilling from your own samples degrades over time because the policy moves:
> "while KL divergence is 0 in expectation, every finite batch will exhibit a slightly different distribution in practice. Training on these finite batches causes a non-zero gradient update, which then diverges the updated model's policy from that of its original state. This process turns training on one's own samples into off-policy training over time" — TML

## Closing guidance
> "By leveraging on-policy sampling from the student with dense supervision from a teacher, the on-policy distillation recipe reaches those capabilities at a fraction of the cost of frontier high-compute RL runs." — TML

Connects to [[ross-dagger-exposure-bias]] (on-policy correction of compounding error) and [[agarwal-gkd]] (on-policy student-sampled distillation with reverse/JS KL). Distributional framing: [[nrehiew-sft-rl-opd]]; industrial numbers: [[qwen3-strong-to-weak-distillation]]; how to run it: [[hf-trl-gkd-recipe]].
