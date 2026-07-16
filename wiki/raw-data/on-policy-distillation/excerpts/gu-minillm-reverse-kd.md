<!-- scope: MiniLLM — replace forward-KL with reverse-KL for LLM distillation, optimized on-policy via policy gradient
     deps:
     see-also: tm-on-policy-distillation -->
# MiniLLM: Reverse-KL Knowledge Distillation of Large Language Models

- **Core Insight:** Swapping the standard forward-KL distillation objective for **reverse KL** — KL(student ‖ teacher) — makes the student *mode-seeking* rather than *mode-covering*, so it stops wasting probability mass overestimating the teacher's low-probability (void) regions and instead concentrates on the modes it can actually reproduce.
- **Guideline:** For distilling a generative (free-run) LLM, minimize KL(q_θ ‖ p) over **student-generated** sequences using a policy-gradient estimator, and stabilize it with single-step decomposition (variance), teacher-mixed sampling (reward hacking), and length normalization (length bias).
- **Source:** Gu, Dong, Wei, Huang, "MiniLLM: On-Policy Distillation of Large Language Models," ICLR 2024. arXiv:2306.08543 (https://arxiv.org/abs/2306.08543, PDF https://arxiv.org/pdf/2306.08543). OpenReview: https://openreview.net/forum?id=5h0qf7IBZZ. Abstract fetched verbatim; equations/technique names corroborated from PDF fetch + search snippets (marked below).
- **Relevant chapters:** reverse-KL objectives; on-policy / off-policy distillation; policy-gradient KD; open-ended generation

## The objective swap (VERBATIM, abstract)
"We first replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD, which is more suitable for KD on generative language models, to prevent the student model from overestimating the low-probability regions of the teacher distribution. Then, we derive an effective on-policy optimization approach to learn this objective." — abstract, arXiv:2306.08543

The reverse-KL loss is (plain text):
`L(θ) = KL[ q_θ(y|x) ‖ p(y|x) ]  ,  minimized as  argmin_θ KL[q_θ ‖ p]`
where `q_θ` = student policy, `p` = fixed teacher. Note the argument order is *flipped* vs standard KD, which minimizes `KL[p ‖ q_θ]` (forward). *(equation form corroborated from PDF fetch, not a guaranteed-exact glyph transcription — treat as thesis-level, not verbatim.)*

## Mode-seeking vs mode-covering (why reverse KL)
- Forward KL `KL[p ‖ q_θ]` is **zero-avoiding / mode-covering**: it penalizes the student for putting near-zero mass anywhere the teacher has mass, so a low-capacity student smears probability over *all* teacher modes — including implausible tails it can't model — and overestimates the teacher's low-probability regions.
- Reverse KL `KL[q_θ ‖ p]` is **mode-seeking**: it penalizes the student for putting mass where the teacher does *not*, so the student focuses on the **major / high-probability modes** it can actually reproduce. In open-ended generation there are many valid continuations (a broad multimodal target); a student that faithfully covers correct-but-major modes generates *more precise* text than one that hedges across everything.
- Search-snippet framing (not verbatim, corroborated): "MiniLLM ... does not force the student to fit all samples from the teacher distribution. Instead, it encourages the student to generate samples preferred by the teacher within its own capacities."

## On-policy optimization: policy gradient
Reverse KL requires sampling from the *student* (expectation is under `q_θ`), which makes it an **on-policy** objective — samples are drawn from the model currently being trained. It is optimized with a REINFORCE-style policy gradient (plain text):
`∇L(θ) = − E_{y ~ q_θ(·|x)} [ R(y) · ∇ log q_θ(y|x) ]`
where the per-sequence reward `R(y)` is derived from the teacher/student log-ratio (`log p(y|x)/q_θ(y|x)` terms). Three stabilizers (VERBATIM technique names, descriptions thesis-level from PDF/search):
1. **Single-step decomposition** — reduce gradient variance by splitting the sequence-level reward into a directly-differentiable single-step term plus the long-term term.
2. **Teacher-mixed sampling** — sample from a mixture of student and teacher to alleviate **reward hacking** / degenerate low-quality samples.
3. **Length normalization** — normalize the reward by sequence length to remove the bias toward short sequences.

## Connection to on-policy distillation
This is the objective that [[tm-on-policy-distillation]] (Thinking Machines) builds on: both minimize **reverse KL against student-sampled trajectories** — the student proposes, the teacher scores per-token, and the student is pulled toward teacher-preferred continuations *within its own support*. MiniLLM supplies the sequence-level REINFORCE derivation; the Thinking Machines framing reframes the same reverse-KL-on-student-rollouts idea as dense per-token on-policy supervision. The shared "sample from the student, grade with the teacher" loop is the defining on-policy sampling link.

## Follow-up
- **DistiLLM** (Ko et al., ICML 2024, arXiv:2402.03898): generalizes the KL choice to **skew KLD** — `D_SKL^α(p,q_θ)=KL(p ‖ αp+(1−α)q_θ)` and its reverse variant SRKL — to fix gradient instability under large teacher/student mismatch, plus an **adaptive off-policy** scheme that reuses student outputs for 2.2×–3.4× faster training with minimal quality drop.

## Key numbers (VERBATIM, abstract)
- "Our method is scalable for different model families with 120M to 13B parameters."
- Reported wins over baselines: "more precise responses with higher overall quality, lower exposure bias, better calibration, and higher long-text generation performance."
