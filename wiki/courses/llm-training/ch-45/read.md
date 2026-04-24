<!-- chapter: ch-45
     track: rl
     kind: content
     title: Self-Improvement Loops
     deps: [ch-44]
     sources: [[self-rewarding-lm]], [[meta-rewarding-lm]], [[spin]], [[self-play-preference]],
              [[self-correct-rl]], [[rest-em]], [[star]], [[v-star]],
              [[deepseek-r1]], [[r1-zero-analysis]], [[rlvr-beyond-base-model]],
              [[iterative-sft-rl]], [[lilianweng-reasoning-llms]]
     figures: figures/self-improve-loop.html
-->

# Chapter 45 — Self-Improvement Loops

> **Core insight.** Every self-improvement method in 2024–2025 is the same three-line algorithm — *generate, filter, retrain* — differing only in who plays the filter. When the filter is **the model itself as a judge** (Self-Rewarding, Meta-Rewarding) the loop saturates in 3 iterations because the judge drifts. When the filter is **the human text distribution** (SPIN) it converges to matching the data and nothing beyond. When the filter is a **preference game** (Nash-LM) it converges to a stochastic equilibrium. When the filter is an **outcome verifier** (ReST-EM, R1-Zero) it keeps climbing as long as the base model has latent capability to elicit — but the 2025 `pass@k` analyses show this may be *eliciting, not creating*, reasoning. The choice of filter is the whole story.
>
> **Guideline.** Pick your filter to match the signal you actually have. Verifiable answers → ReST-EM or R1-Zero-style RLVR (monotone, slow saturation). SFT data but no preferences → SPIN (distribution-match; cap at 3). Judge-capable model but no verifier → Self-Rewarding (3 iters, then calibration drifts — add Meta-Rewarding's meta-judge to push to 5). Self-correction as the target → SCoRe's two-stage RL (Stage I freezes turn-1, Stage II rewards the delta). Any loop: reset the reference model between iterations, and measure `pass@large-k` alongside `pass@1` so you notice if you are only sharpening what the base already knows.

---

## 1. The shared skeleton

All five method families in this chapter factor into three steps, executed in an outer loop indexed by `t`:

```
for t in 1..T:
    samples   = generate(pi_{t-1}, prompts)          # E-step / rollout
    retained  = filter(samples, signal_source)       # judge / verifier / data / game
    pi_t      = retrain(pi_{t-1}, retained)          # SFT, DPO, or RL
    pi_ref   <- pi_{t-1}                             # reset reference (critical)
```

What changes between families is the `signal_source`:

| Family | Signal source | Filter | Retrain objective | Saturation |
|---|---|---|---|---|
| Self-Rewarding ([[self-rewarding-lm]]) | policy-as-judge, 5-pt rubric | argmax/argmin over 4 samples | DPO β=0.1 | 3 iters (judge drift) |
| Meta-Rewarding ([[meta-rewarding-lm]]) | policy-as-judge + meta-judge | pairwise on 11 judges | DPO on actor *and* judge | 5 iters |
| SPIN ([[spin]]) | human-written SFT response | `chosen=y_human`, `rejected=y_gen` | DPO on (human, self) pair | ~3 iters (distribution match) |
| Nash-LM ([[self-play-preference]]) | pairwise preference model | self-play against EMA opponent | Mirror-descent on preference game | Nash equilibrium |
| SCoRe ([[self-correct-rl]]) | outcome correctness, two-turn | reward shape `r(y2)−r(y1)` | REINFORCE, two-stage | Monotone while reward signal holds |
| ReST-EM ([[rest-em]]) | exact-match / unit-test verifier | K=32 per problem, keep correct | SFT on survivors, 1 epoch | 2 iters (overfit without diversity cap) |
| R1-Zero ([[deepseek-r1]]) | rule-based verifier | GRPO group-relative | On-policy PG with clipped ratio | Open-ended (base-model ceiling) |

The rest of this chapter walks the cells in that table and then reads the 2025 R1-Zero post-mortems to see what the pure-RL end of the spectrum actually delivered.

---

## 2. Self-Rewarding — the policy judges itself

[[self-rewarding-lm]] (Yuan et al., Meta/NYU, 2024) starts from a Llama-2-70B SFT checkpoint on Open Assistant and iterates three DPO rounds. Each iteration:

1. Sample **4 responses** per prompt at `T=0.7, top-p=0.9` from π_{t−1}.
2. Score each with the same model acting as judge. The judge prompt appends a fixed **5-point additive rubric** covering helpfulness, relevance, depth, clarity, completeness. Each pair is judged **3 times** and scores are averaged to reduce variance.
3. Take the highest-scored response as `chosen`, lowest as `rejected`.
4. DPO-train π_t from π_{t−1} with β=0.1, AdamW lr=5e-7, 1 epoch.

The paper's central and most-cited empirical curve is the **AlpacaEval 2.0 monotonic climb** — attested from the paper's Figure 3 and summarized in [[self-rewarding-lm]]:

```
iter 0 (SFT):   9.94 %
iter 1:        15.38 %
iter 2:        20.44 %      <- passes GPT-4 0613 (~19 %)
iter 3:        20.8 %       <- plateau
iter 4:                     <- regresses on reward bench (reward hacking)
```

The asymmetry matters: this is not a single data curve, this is *two* curves that rise together. The judge's Spearman correlation with held-out Open-Assistant human preferences goes from **0.62 at iter 0 to 0.71 at iter 3** ([[self-rewarding-lm]] Table 2). The judge gets better as the actor gets better — a property no frozen reward model can have. That is the paper's real contribution; the AlpacaEval win-rate is just the surface indicator.

**Why it saturates.** The judge is not ground truth. As both actor and judge distill into a narrower high-score basin, the judge's own errors become systematic, and DPO amplifies them. The paper reports iter 4 regresses; [[iterative-sft-rl]] notes the same pattern in Llama-2 RSFT if you push past 5 rounds. Cap at 3, or add a meta-signal.

---

## 3. Meta-Rewarding — judging the judge

[[meta-rewarding-lm]] (Wu et al., Meta FAIR + Berkeley, 2024) adds a **third role**. The model now plays Actor, Judge, *and* Meta-Judge. Per iteration:

1. Sample **K=7** actor responses per prompt.
2. Sample **N=11** judge responses per `(prompt, actor_response)` pair — each a score + rationale.
3. **Meta-Judge** pairwise-compares those 11 judgments under a calibration rubric and picks best/worst.
4. Actor-DPO uses `(best_actor, worst_actor)` pairs (aggregated over judge scores).
5. **Judge-DPO** uses `(best_judge, worst_judge)` pairs from the meta-judge.
6. A **length-bias control term** in the rubric ("don't reward length for length's sake") prevents the 2× response-length inflation that plagues plain DPO self-loops.

The judge-DPO leg is the innovation. Self-Rewarding trains only the actor; Meta-Rewarding trains both roles. Llama-3-8B-Instruct AlpacaEval 2.0 LC win-rate climbs **22.9 % → 39.4 % over 4 iterations** ([[meta-rewarding-lm]] Table 1); judge-human agreement keeps rising where Self-Rewarding's flattened.

Notice the structural pattern. Self-Rewarding is a **two-layer** stack (Actor, Judge). Meta-Rewarding is **three-layer** (Actor, Judge, Meta-Judge). You can imagine a four-layer stack, but the compute cost per iteration explodes (`K × N × M` judge calls) and the paper's ablations show diminishing returns past the meta-judge. This is the same hierarchical-evaluation pattern you will see in [[ch-44]]'s process-reward models, but applied to the *judge* rather than the step rollout.

---

## 4. SPIN — the data is the judge

[[spin]] (Chen et al., UCLA, 2024) makes a different choice: the "judge" is the **human SFT response itself**. Every iteration, the DPO pair is literally `(y_human, y_model_previous)`:

```
L_SPIN = −log σ( β · [ log(π_t(y_human) / π_{t−1}(y_human))
                       − log(π_t(y_gen)   / π_{t−1}(y_gen)  ) ] )
```

This is DPO verbatim — the only change is the preference *source*. The algebraic consequence (paper's Theorem 4.1) is a **Nash equilibrium characterization**: when π_t matches the data distribution, the SPIN loss becomes zero. SPIN is "two-player self-play where Nash = data-matching."

Per iteration on Mistral-7B-SFT on UltraChat-200K:
1. Sample **50K (prompt, response) pairs** from π_{t−1} at T=1.0.
2. Build DPO pairs 1:1 with SFT rows: `chosen = y_human, rejected = y_gen`.
3. DPO-train π_t from π_{t−1} with β=0.1, lr=5e-7, 3 epochs, batch 64.
4. **Reset reference to π_{t−1}** for iter t+1.

Zephyr-7B-SFT + 3 SPIN iters matches Zephyr-7B-DPO (which used 60K GPT-4 preferences) on the HF Open LLM Leaderboard; MT-Bench goes **6.39 → 7.12** across 3 iters ([[spin]] Table 2). The headline claim is "preference-level gains from SFT-only data."

The limits are the flip side of the same coin. SPIN converges to the data distribution, so it **cannot exceed the SFT corpus ceiling**. If the human responses are mid-quality, so is the fixed point. This is why SPIN is typically run as a *warmup* before preference-RL in modern pipelines ([[iterative-sft-rl]] §Tülu 3), not as the terminal stage.

---

## 5. Nash-LM — alignment as a preference game

[[self-play-preference]] (Munos et al., DeepMind, 2024) attacks a structural flaw in RLHF: **non-transitive preferences**. Bradley-Terry reward models assume `p(a > b) · p(b > c) · p(c > a) < 1/2` (transitivity). Real human preferences over writing style, coding idioms, humor violate this freely. DPO on non-transitive data oscillates; PPO-on-BT-reward converges to an arbitrary mode.

Nash-LM reframes: find the **Nash equilibrium** policy π* such that `P(π* ≻ π) ≥ 1/2` for any competing π. The Nash-MD update is:

```
π_{t+1}  ∝  π_ref · exp( η · E_{y' ~ π_t} p(· > y' | x) )
```

Practically ([[self-play-preference]] §5), this is REINFORCE against an EMA copy of itself using a preference-model score as reward. The equilibrium is **stochastic** when preferences are non-transitive — the "correct" policy is a mixture, not a mode. That is what SPIN's `Nash = data distribution` and Self-Rewarding's `Nash = judge's argmax` lack: an equilibrium that *respects* preference conflict instead of collapsing through it.

---

## 6. SCoRe — self-correction as an RL target

[[self-correct-rl]] (Kumar et al., DeepMind, 2024) targets a very specific capability: **answering, noticing you were wrong, revising.** SFT on correction traces *fails* with two well-documented failure modes:

1. **Distribution shift** — traces come from a stronger teacher; the student's own turn-1 distribution differs, so the conditioning is off.
2. **Mode collapse** — the model learns to produce the correct answer in turn 1 and no-op in turn 2.

SCoRe's fix is **two-stage on-policy RL** over a two-turn trajectory. Turn 1: `question`. Turn 2: `question + turn-1 answer + "There might be an error. Please revise."`. Reward is binary outcome correctness from a rule grader.

**Stage I** trains turn-2 only, with a heavy **KL on turn-1 to the base model**:

```
∇ L_I  =  E[ ∇ log π(y_2 | x, y_1) · r(y_2) ]   +   λ_KL · KL( π(· | x) || π_ref(· | x) )
```

The Stage-I KL *freezes* turn-1 behavior so the model has to learn editing rather than just producing the right answer up front. Skip Stage I and you get immediate mode collapse ([[self-correct-rl]] Figure 5).

**Stage II** is joint REINFORCE over both turns with a **reward-shaping bonus on the improvement delta**:

```
R_shaped = r(y_1) + α · [ r(y_2) − r(y_1) ],   α = 2.0
∇ L_II   = R_shaped · ∇ [ log π(y_1 | x) + log π(y_2 | x, y_1) ]
```

Amplifying `r(y_2) − r(y_1)` makes *improvement between turns* the high-gradient direction. Result: **+15.6 pts on MATH**, **+9.1 pts on MBPP** ([[self-correct-rl]] Table 2) with Gemini 1.0 Pro — the first published method that crosses zero on the self-correction task (earlier models *got worse* with a revise step). The structural lesson — freeze one part of the rollout with a KL term and shape the advantage on the part you want to move — will recur in agentic-RL ([[ch-50]]).

---

## 7. ReST-EM — expectation-maximization for reasoning

[[rest-em]] (Singh et al., DeepMind, 2023) gives the self-training loop its cleanest formalism. Treat the rationale as a **latent variable** `z` and the answer as the observable `y`. EM on this latent:

```
E-step:   sample K=32 rationales z ~ π_{t−1}(· | x)
          keep {z : verify(answer(z), x) = correct}     # exact-match or unit-test
M-step:   π_t = argmax_π  Σ_{(x, z_kept)}  log π(z | x)
          # SFT 1 epoch, lr=1e-5, batch 128
```

That is it. No DPO, no reward model, no KL. The E-step is pure rejection sampling; the M-step is vanilla SFT. The **verifier** carries all the supervision.

Results on PaLM-2-L attested in [[rest-em]] Figure 2 / Table 2:

```
MATH:   human-SFT 34.1 %   ->   ReST-EM iter 1: ~42 %   ->   iter 2: 50.6 %   ->   iter 3: flat
APPS:   human-SFT 16.4 %   ->   ReST-EM iter 2: 31.2 %
BBH held-out: gains transfer — models trained on MATH self-data improve on unrelated BBH tasks.
```

The **diversity cap** is the ingredient that separates this from naive self-distillation: keep at most **4 distinct correct solutions per problem** (paper's ablation). Without it, iter-3 regresses because the sampled distribution collapses onto one solution path, and the M-step memorizes it. [[star]] (Zelikman 2022) is the K=1 ancestor with a "rationalize backward from the gold answer on failures" trick; ReST-EM drops the rationalization and just scales K. [[v-star]] adds a verifier trained on the failed traces so failures also contribute signal — the bridge from self-training to process-reward models covered in [[ch-44]].

---

## 8. R1-Zero and the 2025 dissections

[[deepseek-r1]] (DeepSeek-AI, 2025) takes the ReST-EM filter (verifier-only) and swaps the M-step SFT for **pure on-policy GRPO** with no cold-start and no PRM. Reward is `r = r_acc + r_format` where `r_acc ∈ {0, 1}` from a rule grader and `r_format ∈ {0, 1}` for matching `<think>…</think><answer>…</answer>`. Group size 16–64, sequence length up to 32k, advantage `A_i = (r_i − mean(r_{1:G})) / std(r_{1:G})`.

Two emergent behaviors — neither seen in SFT-based pipelines — dominate the paper:

- **Chain-of-thought length grows from ~400 tokens to 10k+** over training ([[deepseek-r1]] Fig. 3).
- **Phase-transition "aha moment"** mid-training: the model spontaneously writes *"Wait, let me reconsider…"* and *"Let me check step 3…"* despite having never seen a training example containing such text ([[deepseek-r1]] Fig. 4).

The 2025 follow-up analyses dissect *what is actually happening*. I quote the findings directly because the papers are blunt about what R1-Zero is and isn't.

**Finding 1 — GRPO has two exploitable biases.** From [[r1-zero-analysis]] on Dr.GRPO (Liu et al., 2025): GRPO's per-token mean aggregation induces a **"length bias from per-token mean aggregation that rewards longer correct responses and longer wrong responses asymmetrically"** and the per-prompt std normalization induces a **"difficulty bias from per-prompt std normalization that inflates gradients on easy prompts."** The fix is the bias-corrected loss, which *removes std normalization* and aggregates as batch-mean divided by `(B · L_max)`:

```
L_DrGRPO = −(1 / (B · L_max)) · Σ_{i,t} mask_{i,t} · min( r_{i,t} · A_i,  clip(r_{i,t}) · A_i )
           with A_i = R_i − μ_group   (no std division)
```

**Finding 2 — the base-model prior is load-bearing.** Open-Reasoner-Zero reproduces R1-Zero on Qwen2.5-7B-Base but reports ([[r1-zero-analysis]]): *"emergence happens; it disappears if the base is not reasoning-pretrained."* Translation: pure RL is not creating reasoning from scratch. It is amplifying a reasoning capability that pretraining already installed, silently, via math/code-heavy data. Rerun the same recipe on a chat-pretrained base and you get neither the length growth nor the aha moment.

**Finding 3 — no PRM needed.** All three 2025 reproductions ([[r1-zero-analysis]] — Dr.GRPO, ORZ, TinyZero) converge on the same minimum-viable ingredient list: **outcome-only verifier is sufficient and in fact dominates**. Adding a PRM — the [[ch-44]] process-reward approach — does not help and often hurts because the PRM introduces a learned reward source vulnerable to overoptimization, which the rule grader is not.

**Finding 4 — RL may be sharpening, not expanding.** [[rlvr-beyond-base-model]] (Yue et al., 2025) runs the experiment that matters: under **large-`k` pass@k**, the base model **matches or exceeds** the RL-trained model. Direct quote: *"RL improves pass@1, base model wins at high k."* The interpretation: *"RLVR mostly redistributes probability mass toward already-existing successful paths, while narrowing exploration and reducing the broader coverage of solvable problems."* RL post-training may be a **sampling-efficiency** improvement more than a **capability-boundary** expansion. This is the strongest caution on R1-Zero-style training in 2025 and the main reason [[ch-46]]'s lab instruments `pass@k` across prompt-difficulty buckets, not just average pass@1.

These four findings do not reduce R1-Zero to an illusion. They refine the claim. Pure-RL with a verifiable reward demonstrably (a) *elicits* long chain-of-thought, (b) *does not hack* a reward model because there is no reward model, (c) *saturates slower* than judge-based loops because the reward source isn't drifting, and (d) *requires* a reasoning prior in the base to work. What R1-Zero is *not* is proof that RL creates novel reasoning. That remains open — see [[lilianweng-reasoning-llms]]'s open-questions list.

---

## 9. The filter is the whole method — a recap

Read the chapter's methods back through the single knob that actually differs:

| Filter | Method | Saturation mechanism |
|---|---|---|
| Policy-as-judge | Self-Rewarding | Judge drift at iter 3 |
| Judge + Meta-Judge | Meta-Rewarding | Meta-judge drift at iter 5 |
| Human text distribution | SPIN | Data ceiling (Nash = match data) |
| Preference game | Nash-LM | Nash equilibrium (possibly stochastic) |
| Outcome reward, two-turn | SCoRe | Reward signal holds as long as verifier is clean |
| Outcome verifier, SFT M-step | ReST-EM | Diversity collapse without cap |
| Outcome verifier, RL M-step | R1-Zero | Base-model reasoning prior |

The reference-reset rule appears in every row: without `π_ref ← π_{t−1}` each iteration, the DPO/RL loss keeps pulling the policy toward a stale reference and iteration returns nothing.

**Forward to [[ch-46]].** The capstone lab runs one of these loops end-to-end — either a DPO β-sweep (SPIN-like) or an RLVR KL-sweep (R1-Zero-like) — and forces you to find a failure mode (reward hacking, entropy collapse, length inflation). Every diagnostic in that lab maps to a saturation mechanism in the table above.

**Back to [[ch-44]].** The outcome verifier that makes ReST-EM and R1-Zero work is the same object as RLVR's binary reward. The difference is whether you use it in an E-step (sample-filter-SFT) or inside a gradient update (sample-advantage-REINFORCE). Self-improvement loops are not an alternative to process/outcome supervision — they are the *scheduling pattern* that makes verifiable rewards compound across iterations.
