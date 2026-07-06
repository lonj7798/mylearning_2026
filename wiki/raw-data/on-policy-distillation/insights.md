<!-- scope: cross-source insight index for the on-policy-distillation raw library
     deps: [[README]], [[COLLECTION-PLAN]]
     see-also: [[tm-on-policy-distillation]], [[nrehiew-sft-rl-opd]], [[agarwal-gkd]],
               [[ross-dagger-exposure-bias]], [[hf-trl-gkd-recipe]]
-->

# On-Policy & Off-Policy Distillation — Insights Index

The course turns on **one idea**:

> **Every post-training method reshapes the model's sequence distribution; the "on-policy property" — training on the student's OWN samples — is what fixes exposure bias, and pairing it with a dense per-token teacher signal (on-policy distillation) buys RL-level capability at a fraction of the cost.**

Read every excerpt through three axes: **data source** (off-policy = teacher/dataset vs on-policy = student), **signal density** (sparse O(1) bits = RL scalar reward vs dense O(N) bits = per-token teacher), and **divergence geometry** (forward KL = mode-covering vs reverse KL = mode-seeking) ([[nrehiew-sft-rl-opd]], [[tm-on-policy-distillation]]).

## The distribution-matching map (where every method sits)

- **SFT** = fixed external target, forward KL, mode-covering, no anti-forgetting regularization; "SFT via Cross Entropy on a fixed dataset is equivalent to minimizing forward KL divergence, up to constants" ([[nrehiew-sft-rl-opd]]).
- **RL** = no external target, on-policy reward-direction, "learns the nearest task-solving policy," but "only teaches O(1) bits per episode" ([[nrehiew-sft-rl-opd]], [[tm-on-policy-distillation]]).
- **On-policy distillation** = the corner combining "the density of distillation, the unbiasedness of RL, and the on-policy property of both" ([[nrehiew-sft-rl-opd]]).

## The off-policy lineage (the baseline the course improves on)

- **Soft targets carry the knowledge.** Hinton's temperature-softened softmax exposes "dark knowledge" — inter-class structure a hard label discards; the seed of the O(N)-bits idea ([[hinton-knowledge-distillation]]).
- **Move to sequences.** Kim & Rush train the student on the teacher's *generated* (beam-mode) sequences — sequence-level KD — an early recognition that *which sequences you train on* matters. Still off-policy: the sequences come from the teacher ([[kim-rush-seqkd]]).
- **SFT is off-policy sequence-level distillation.** The unifying reframe that sets up the whole course ([[nrehiew-sft-rl-opd]]).

## The on-policy principle (why it matters)

- **Exposure bias is a distribution-shift problem, not a data problem.** Behavioral cloning drifts into states the expert never visited → O(T²) error; on-policy data collection (DAgger) restores O(T) ([[ross-dagger-exposure-bias]]). On-policy distillation is DAgger at token level with a soft teacher label ([[tm-on-policy-distillation]]).
- **On-policy stays aligned; off-policy drifts.** Even SFT on a model's own samples degrades over finite batches — "this process turns training on one's own samples into off-policy training over time" ([[tm-on-policy-distillation]]).

## On-policy distillation (the mechanism)

- **Student samples, teacher grades every token.** Per-token reverse KL `KL(π_θ‖π_teacher)`; set the per-token advantage to its negative and reuse the RL importance-sampling loss — OPD is the RL loop with a dense teacher reward ([[tm-on-policy-distillation]]).
- **Reverse KL is mode-seeking and "unhackable."** It concentrates the student on teacher-preferred modes it can actually reproduce ([[gu-minillm-reverse-kd]], [[tm-on-policy-distillation]]).
- **GKD generalizes it.** The λ knob mixes off-/on-policy data (λ→1 = on-policy), and a JSD-β knob interpolates forward↔reverse KL; on-policy/mixed data beats fixed-dataset KD even with far less data ([[agarwal-gkd]]).

## Economics & failure modes (price the bet)

- **~10× cheaper than RL:** Qwen3-8B, AIME'24: +RL 67.6% @ 17,920 GPU-hrs vs +OPD 74.4% @ 1,800 GPU-hrs; 9–30× vs SFT-scaling; self-distillation 50–100× ([[qwen3-strong-to-weak-distillation]], [[tm-on-policy-distillation]]).
- **Costs and risks:** needs teacher-logprob access + student-sampling infra; entropy collapse ("sudden reward increase … drastic collapse in entropy"); per-token clipping because style tokens carry higher KL than task tokens; "the source of the data matters a lot while the teacher matters less than expected" ([[nrehiew-sft-rl-opd]]).
- **Domain dependence:** math/code favor RL; creative/knowledge favor distillation ([[nrehiew-sft-rl-opd]]).

## Practice

- **TRL turns it into three knobs:** `lmbda` (on-policy fraction), `beta` (forward↔reverse KL), `temperature`; GOLD extends it cross-tokenizer so any teacher family can grade any student ([[hf-trl-gkd-recipe]]).

## Through-line to the learner's system

`boson-agent-synthetic-data-dev` generates 20–50-turn Korean TMR sales dialogue and SFT's the seller student **off-policy** on those transcripts. That is the long-horizon compounding-error regime OPD targets: have the seller student sample its own turns inside the fixed scenario skeleton, and let a strong teacher (Claude / large Qwen, cross-tokenizer via GOLD) grade each seller-turn token by reverse KL. Price it as a bet — teacher-serving cost vs exposure-bias reduction over long calls (ch-07).

## Open gaps (see [[COLLECTION-PLAN]])
- DAgger and the Qwen3 report are thesis-extracted (spend-limit); re-fetch for verbatim.
- DistiLLM (skew-KL, adaptive off-policy) lives inside [[gu-minillm-reverse-kd]]; promote only if divergence-objectives gets its own chapter.
