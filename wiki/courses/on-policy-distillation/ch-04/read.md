<!-- chapter: ch-04
     track: on-policy-core
     kind: content
     title: On-Policy Distillation — Dense Teacher Signal on Student Rollouts
     deps: [[ch-03]]
     sources: [[tm-on-policy-distillation]], [[agarwal-gkd]], [[gu-minillm-reverse-kd]], [[nrehiew-sft-rl-opd]]
-->

# Chapter 04 — On-Policy Distillation: Dense Teacher Signal on Student Rollouts

> **Core insight.** On-policy distillation is the corner of [[ch-01]]'s cube that fixes [[ch-03]]'s exposure bias without giving up dense supervision: the student **samples its own trajectories**, and a **fixed teacher grades every token** by reverse KL. Mechanically it is nothing exotic — it is the RL loop with the sparse scalar reward replaced by a dense per-token advantage equal to the negative reverse-KL against the teacher. Three literatures describe the same object: Thinking Machines gives the recipe, GKD (Agarwal et al.) gives the general family with a λ (on-policy fraction) and a β (forward↔reverse KL) knob, and MiniLLM gives the reverse-KL-for-LLMs derivation.

> **Guideline.** To run on-policy distillation: sample from the student (λ→1), query the teacher's log-probs on those exact samples, set the per-token advantage to −reverse-KL, and optimize with your existing importance-sampling RL loss. Choose the divergence to match the student's capacity — reverse-KL / high-β when the student cannot fully mimic the teacher (mode-seeking on what it *can* reproduce), forward-KL / low-β when it can.

---

## 1. The mechanism in one paragraph

Thinking Machines states the core move directly ([[tm-on-policy-distillation]]):

> "The core idea of on-policy distillation is to sample trajectories from the _student_ model and use a high-performing teacher to grade _each token_ of each trajectory."

That is the whole idea, and it slots the two axes we care about into place at once: **on-policy** (data from the student — [[ch-03]]'s cure) and **dense** (a signal at every token — [[ch-02]]'s virtue). The only remaining question is *what the grade is*, and the answer is a per-token reverse KL.

---

## 2. The objective: per-token reverse KL

Grade each token by the reverse KL between student and teacher next-token distributions at that prefix ([[tm-on-policy-distillation]]):

```
KL( π_θ ‖ π_teacher )
  = E_{x ~ π_θ} [ log π_θ(x_{t+1} | x_{1..t}) − log π_teacher(x_{t+1} | x_{1..t}) ]
```

Two properties make reverse KL the right grade for a *fixed* teacher:

- **Mode-seeking.** It "learns one specific behavior (the teacher's) instead of spreading its distribution across several suboptimal options" ([[tm-on-policy-distillation]]) — the opposite of forward-KL SFT's hedging ([[ch-01]] §4). MiniLLM adopted exactly this swap "to prevent the student model from overestimating the low-probability regions of the teacher distribution" ([[gu-minillm-reverse-kd]]).
- **"Unhackable."** "the reverse KL is 'unhackable' in the sense that low KL always corresponds to a high probability of desirable behavior from the teacher model's point of view" ([[tm-on-policy-distillation]]). And the fixed point is exactly right: when the student matches the teacher, the reverse KL is **zero**. There is no way to score well except by behaving like the teacher.

> **Interactive companion:** [`figures/per-token-grading.html`](figures/per-token-grading.html) — a student-sampled sentence graded token by token. Each token is colored by its per-token reverse KL; click one to see the student's vs the teacher's next-token distribution and the resulting advantage (−KL). Notice the pattern nrehiew reports: *style* tokens ("wait", "let") carry high KL while *task-critical* tokens ("12", "84") are already well-matched — which is why per-token clipping matters (ch-05).

---

## 3. The 4-step implementation (it is just the RL loop)

The reason on-policy distillation is practical is that it reuses RL infrastructure verbatim ([[tm-on-policy-distillation]]):

1. **Initialize a teacher client** (separate from the student).
2. **Sample trajectories** from the student; capture the student's own log-probs.
3. **Compute the reward:** "We query the teacher client with `compute_logprobs` on the sampled trajectories."
4. **Loss:** "We set the per-token advantage to the negative reverse KL, and call the RL importance-sampling loss function."

So on-policy distillation is *not a new trainer*. It is the RL trainer with one substitution: swap the sparse environment reward for a dense per-token advantage = −(reverse KL to teacher). This is why teams already running RL can adopt it cheaply — and why [[ch-01]]'s density axis pays off, since every token now carries signal.

---

## 4. GKD: the general family (λ and β)

Agarwal et al. (GKD) give the academic generalization, and it is worth stating precisely because it contains SFT, off-policy KD, and on-policy distillation as *special cases of two knobs* ([[agarwal-gkd]]). The objective mixes off-policy data (from a dataset or teacher) with on-policy student samples via λ:

```
L_GKD(θ) = (1−λ) · E_(x,y)∼data [ D(p_T ‖ p_S^θ)(y|x) ]
         +   λ   · E_x∼X, y∼p_S(·|x) [ D(p_T ‖ p_S^θ)(y|x) ]
```

- **λ (the on-policy fraction).** λ=0 → **supervised KD** (fixed dataset — the [[ch-02]] baseline); λ=1 → **on-policy** (every target sampled from the student); in between, a mix. "The authors find that on-policy data (high λ) performs better."
- **β (the divergence, via generalized JSD).** GKD replaces the fixed loss with a generalized Jensen–Shannon divergence whose limit β→0 is **forward KL** (mode-covering, when the student can mimic the teacher) and β→1 trends to **reverse KL** (mode-seeking, when it cannot). One knob slides along [[ch-01]]'s geometry axis.

The headline evidence that the *data source* is what matters ([[agarwal-gkd]]):

> "on-policy GKD on the 5% subsampled dataset, without any ground-truth summaries, outperforms supervised KD and ImitKD with entire training dataset with ground-truth summaries."

Plus consistent multiplicative gains — ~**2.1×** on XSum summarization, **1.7×** on WMT translation, **1.9×** on GSM8K arithmetic — over supervised/word-level KD. A small on-policy dataset beats a large off-policy one: the exact prediction of [[ch-03]]'s exposure-bias argument.

---

## 5. MiniLLM: the reverse-KL derivation

MiniLLM supplies the piece GKD's expectation implies — how to actually optimize reverse KL when the expectation is *under the student* ([[gu-minillm-reverse-kd]]):

> "We first replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD… Then, we derive an effective on-policy optimization approach to learn this objective."

Because `KL(q_θ ‖ p)` is an expectation over student samples, minimizing it *requires* sampling from the student — it is inherently on-policy — and is optimized with a REINFORCE-style policy gradient. MiniLLM stabilizes it with three tricks worth knowing: **single-step decomposition** (variance reduction), **teacher-mixed sampling** (guards against reward-hacking / degenerate samples), and **length normalization** (removes the short-sequence bias). Thinking Machines' per-token-reverse-KL recipe is the dense, per-token realization of this same "sample from the student, grade with the teacher" loop.

---

## 6. Two myths killed

**"On-policy distillation is just a kind of RL."** It shares RL's *loop* and *on-policy property*, but the signal is categorically different: RL's reward is a sparse scalar (O(1) bits/episode); OPD's "reward" is the teacher's full per-token distribution (O(N) bits) — [[ch-01]]'s density axis. Thinking Machines even frames OPD as a *shortcut past RL*: "on-policy distillation does not need to model the intermediate strategies during the curriculum of RL." Same loop, different — and far denser — teacher.

**"Forward vs reverse KL is an implementation detail."** It is the difference between a student that hedges to cover the teacher (forward, mode-covering, can forget) and one that commits to the teacher's best modes (reverse, mode-seeking, "unhackable"). GKD makes it a first-class knob (β) precisely because it changes behavior, especially when the student lacks the capacity to mimic the teacher exactly.

---

## 7. Applied: OPD as the seller's objective

The seller student in `boson-agent-synthetic-data-dev` currently sits in the off-policy corner ([[ch-02]] §6). On-policy distillation moves it to this chapter's corner concretely: let the seller **sample its own turns** inside the fixed scenario skeleton, query a strong teacher (Claude or a large Qwen) for its per-token distribution over those exact turns, and train on the per-token reverse KL. Every seller-turn token gets a dense grade *in the states the seller actually reaches* — the cure [[ch-03]] argued for, delivered with [[ch-02]]'s density. The GKD knobs give the design levers: λ→1 for fully on-policy seller turns, β toward reverse KL because a 27B seller cannot fully mimic a frontier teacher and should mode-seek onto what it *can* reproduce. The full design — which tokens to grade, how to handle tool calls, compaction, and barge-ins, and whether the teacher is cross-family — is [[ch-07]]. This chapter fixes the *objective*; the capstone fixes the *engineering*.

---

## Where This Goes

Chapter 5 prices the bet. On-policy distillation is not free — it needs teacher-log-prob access, student-sampling infrastructure, and it can collapse. But when it fits, the payoff is dramatic: ~10× less compute than RL for a higher score. Chapter 5 gives the numbers, the failure modes (entropy collapse, per-token clipping), and the rule for *when* the on-policy corner is worth its cost.

## Additional Reading

- Thinking Machines, "On-Policy Distillation" (2025) — https://thinkingmachines.ai/blog/on-policy-distillation/ ([[tm-on-policy-distillation]])
- Agarwal et al., "On-Policy Distillation of Language Models" (GKD, ICLR 2024) — https://arxiv.org/abs/2306.13649 ([[agarwal-gkd]])
- Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models" (ICLR 2024) — https://arxiv.org/abs/2306.08543 ([[gu-minillm-reverse-kd]])
- Ko et al., "DistiLLM: Towards Streamlined Distillation for LLMs" (ICML 2024) — https://arxiv.org/abs/2402.03898 (skew-KL + adaptive off-policy follow-up)
