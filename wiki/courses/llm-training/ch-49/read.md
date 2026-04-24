<!-- chapter: ch-49
     track: eval
     kind: content
     title: Judge Models and Judge Calibration
     deps: [ch-48]
     sources: [[judge-llm-bias]], [[generative-reward-models]], [[direct-judgement-preference]], [[rlaif-scaling]], [[ultrafeedback-construction]], [[pairrm]], [[reward-hacking-taxonomy]], [[self-rewarding-lm]], [[meta-rewarding-lm]], [[faithful-synth-eval]], [[wildchat]]
     figures: figures/judge-bias.html
-->

# Chapter 49 — Judge Models and Judge Calibration

> **Core insight.** An LLM judge is not a ruler; it is a noisy instrument with *structured* biases — position, verbosity, self-enhancement, format, and self-reinforcement under iteration. Zheng 2023's headline ("GPT-4 agrees with humans ~80% of the time, matching human-human agreement") is true on aggregate and misleading on every axis that matters for a go/no-go decision. A calibrated judge protocol is a measurement stack: swap-audited, length-controlled, rubric-pinned, and cross-family to the policy under test. The 2024–25 synthetic-judge line (Con-J, Self-Taught Evaluators, J1) replaces GPT-4-as-judge not because those judges are cheaper — though they are — but because an *owned* judge is the only way to bound leakage between RL-time reward and eval-time measurement.
>
> **Guideline.** Treat the judge as an instrument you must calibrate before trusting. For every eval run: (1) randomize position per pair and report swap-consistency as a first-class number, (2) pair with a length-controlled baseline and report a length-residualized win-rate, (3) never use the candidate model as its own judge, (4) anchor to a held-out human-label set every quarter to detect judge drift, (5) keep the RL-time RM and the eval-time judge in *different model families* so a hack of one does not score against itself. When those hold, the judge is an eval primitive; when any fail silently, you are measuring a circle.

---

## 1. Why judges at all

Instruction-following, creative writing, and open-ended helpfulness have no ground truth. A math answer can be checked with SymPy; whether "draft me a condolence email" succeeded cannot. The field's answer since 2023 has been **LLM-as-judge**: a strong LM reads `(prompt, response_A, response_B)` and emits a verdict, treated as a preference label. That verdict is the raw material for three different pipelines:

- **Eval:** rank checkpoints on open-ended benchmarks (MT-Bench, Arena-Hard, AlpacaEval, WildBench).
- **Training data:** produce preference pairs for DPO/RPO ([[ultrafeedback-construction]], [[rlaif-scaling]]).
- **Runtime reward:** score rollouts in PPO/GRPO when a scalar RM is unavailable ([[self-rewarding-lm]]).

This chapter is about the *eval* use. Ch-42 and ch-44 handled the training-time use. The distinction matters — as the outline states, **"if a topic is about producing a signal for training, it belongs in RL; if it is about auditing or comparing checkpoints, it belongs in Eval"** (see outline `rl_vs_eval_boundary`). A judge calibrated well enough for RL may still leak the benchmark it scores.

One question to fix before going further: *why can the same LM act as evaluator at all?* The answer, from [[rlaif-scaling]]'s "same-size labeler" ablation, is that preference labeling is strictly easier than generation. Even when the labeler is the same base LM as the policy, RLAIF improves over SFT — so the labeler is extracting discriminative signal the generator cannot exploit on its own. That asymmetry is what makes LLM-as-judge viable, and it is also why the judge's *biases* are systematic rather than random: they reflect the labeler's training distribution, not noise you could cancel by ensembling the same family.

---

## 2. The three public judge-driven benchmarks

The eval community converged, between 2023 and 2025, on three layered judge-driven benchmarks. Each targets a different failure of the earlier one:

| Benchmark | Prompt source | Pairing | Judge | Typical metric | Known failure mode |
|---|---|---|---|---|---|
| **MT-Bench** (Zheng 2023, [[judge-llm-bias]]) | 80 hand-written questions × 8 categories (writing, roleplay, reasoning, math, coding, extraction, STEM, humanities) | Single-answer scoring (1–10) + pairwise vs reference | GPT-4 + rubric | Mean category score | Small N, category collapse, leakage since 2024 |
| **Arena-Hard** (LMSys 2024) | 500 user prompts mined from Chatbot Arena, filtered for high-discriminator difficulty | Pairwise vs baseline (GPT-4-0314) | GPT-4-Turbo + length-control regression | Length-controlled win-rate (LC-WR) | Judge-family self-enhancement (Arena-Hard built on GPT-4 judge) |
| **WildBench** (AI2 2024) | Real user queries from [[wildchat]] opt-in logs, difficulty-tagged | Pairwise vs three reference models (GPT-4, Claude, Llama-70B) | GPT-4-Turbo with domain-specific rubric | WB-Score (1–10 scaled by difficulty) | Judge reads its own prior outputs; rubric hand-tuning is labor-intensive |

All three rely on a GPT-family judge at the time of release. That is the shared single-point-of-failure the synthetic-judge line exists to fix (§5).

---

## 3. The bias inventory — Zheng 2023 canon

The foundational result is quoted verbatim from [[judge-llm-bias]]:

> "GPT-4 vs human-expert agreement is 85%+ on MT-Bench and ~80% on Chatbot Arena; the same rate two humans agree with each other… A vs B ordering changes the winner in ~20–30% of cases; mitigated by swap-and-average or 'two-game' scoring… longer responses win more often than a length-controlled baseline… GPT-4 prefers GPT-4-authored responses at a rate above what humans prefer; Claude shows the same toward Claude."

Four biases, each with a concrete measurement method and a concrete correction. Commit this table to memory:

| Bias | Measurement method | Signal | Correction |
|---|---|---|---|
| **Position (order)** | Run each pair in both orders `(A,B)` and `(B,A)`; count disagreements | Swap-flip rate: GPT-4 ≈ 22%, GPT-3.5 ≈ 40% ([[judge-llm-bias]] Fig. 2) | Two-game scoring: a win counts only if the judge is order-consistent; else declare tie |
| **Verbosity (length)** | Pair responses that differ only in length (paraphrased long vs short, same content) | Slope of win-rate vs length_delta in tokens ([[judge-llm-bias]] Fig. 4) | Length-residualized win-rate (Arena-Hard LC) or explicit length-penalty rubric term ([[meta-rewarding-lm]]) |
| **Self-enhancement** | Judge × candidate self-preference matrix; compare judge-rate to human-rate on same pairs | Diagonal excess in [[judge-llm-bias]] Fig. 5 | Never use candidate model as its own judge; pool multiple judge families |
| **Format (markdown / bold)** | A/B pairs differing only in markdown headers or bold | Judge tends to prefer formatted ([[reward-hacking-taxonomy]] enumerated modes) | Strip formatting before judging, or include "do not reward formatting" in rubric |
| **Confidence miscalibration** | Bin pairs by judge log-prob margin; plot empirical accuracy per bin | Overconfidence on close pairs ([[generative-reward-models]] Fig. 4) | Generative RM with CoT rubric produces calibrated log-probs; ensemble for tighter bins |
| **Judge-drift under iteration** | Hold out a fixed human-label set; track judge Spearman correlation per round | [[self-rewarding-lm]] Table 2: 0.62 → 0.71 across 3 iters, then drifts | Pin judge weights; re-anchor to human set every quarter |

Three complementary ways to think about the corrections: (a) **symmetrize** the measurement (swap, length-balanced, cross-family); (b) **pin the rubric** (CoT instructions, explicit format/length prohibitions); (c) **anchor** to external ground truth (reference answers, human-label spot checks, verifier-checkable subsets).

A worked example: [[judge-llm-bias]] reports that reference-guided grading — attaching a gold solution to the prompt before the judge reads the candidates — raises agreement "by ~10 pp on MT-Bench." That is not symmetrization; it is anchoring. On math/coding pairs, "LLM judges can confirm a wrong answer if it is presented confidently" (source quote). The reference answer breaks the confident-wrong-answer failure, but only on tasks where a reference exists — writing and roleplay get nothing from it. Which correction you apply is per-category, not global.

See **[figures/judge-bias.html](figures/judge-bias.html)** — pick position/verbosity/self-enhancement, see accuracy-vs-naive judge delta, and scrub a calibration curve by judge confidence.

---

## 4. How a calibrated judge is actually run

At inference time, three artifacts define the judge:

1. **The rubric.** Whatever the prompt says, the judge will optimize. [[ultrafeedback-construction]] uses four axes (instruction-following, truthfulness, honesty, helpfulness) each 0–10 with a short rationale. [[judge-llm-bias]] uses pairwise-with-reference. [[meta-rewarding-lm]] adds explicit "don't reward length for length's sake" — which *is* the calibration. Rubric text is the policy knob.

2. **The template.** CoT-before-verdict is 3–5 pp better than direct-verdict on [[rlaif-scaling]] Fig. 4 and 3–10 pp on RewardBench per [[generative-reward-models]]:

   > "Here is a query and two responses. Which response is better? Respond with 'Response 1' or 'Response 2'. Let's think step by step…"

   Then one token at the end. The answer token's log-prob is the **soft label** per [[rlaif-scaling]]: `p = softmax(logits["Response 1"], logits["Response 2"])`. Soft labels outperform hard A/B when used as BT targets or calibration signals.

3. **The swap protocol.** [[pairrm]] made swap-augmentation standard: `f(x, y_A, y_B)` and `f(x, y_B, y_A)`, average logits, take the consistent winner. [[judge-llm-bias]] quantifies that this cancels position bias to within ~2 pp residual. No modern eval skips this step.

A concrete pairwise prompt template (quoted and paraphrased from [[rlaif-scaling]] §Technical Details):

```
You will see a user query and two candidate responses, labeled Response 1 and Response 2.
Rubric: helpfulness, factual correctness, instruction-following. Do NOT reward length
for its own sake; do NOT reward markdown formatting for its own sake.
Query: {x}
Response 1: {y_A}
Response 2: {y_B}
Let's think step by step before answering.
Final answer (one token only): Response 1 | Response 2
```

The soft label is read from the logits at the final-answer token: `p_A = softmax(logit["Response 1"], logit["Response 2"])[0]`. That `p_A` is the calibrated judge score; `argmax` discards information already computed. When the same pair is run in the swapped order, the two `p_A` values are averaged after flipping one, producing a swap-consistent calibrated score.

Treat a judge setup that omits any of these as a broken instrument. Reports that quote a single MT-Bench number without specifying rubric, template, and swap protocol are not auditable.

---

## 5. The synthetic-judge line — why GPT-4-as-judge is being replaced

Three papers from 2024–25 define the line. Their common move is to *train* a judge rather than pay for one. [[direct-judgement-preference]] calls this "the UltraFeedback pattern collapsed into a self-contained generative judge." The method contrast:

| Method | Year | Training signal | Judge output | Key claim | Cost post-bootstrap |
|---|---|---|---|---|---|
| **GPT-4-as-judge** (Zheng 2023, [[judge-llm-bias]]) | 2023 | None — off-the-shelf API | Verdict + rationale | 85%+ human agreement on MT-Bench | API-priced, ~$0.01–0.03 per pair |
| **Con-J** (ICLR 2025, [[direct-judgement-preference]]) | 2024 | DPO on contrastive judgment pairs with noisy-negative instruction perturbation | Rationale + verdict | Robustness to format bias + label noise; interpretable rationale | Local inference, zero API |
| **Self-Taught Evaluators** (Meta 2024, [[direct-judgement-preference]]) | 2024 | Iterative: judge_k labels fresh pool → train judge_{k+1} by DPO on its own decisions vs alternatives | Rationale + verdict | RewardBench crosses GPT-4 baseline after ~3 iterations on zero human labels post-bootstrap | Local inference |
| **J1** (2025, [[direct-judgement-preference]]) | 2025 | RL on the judge's chain-of-thought (GRPO-style over judgment rollouts) | CoT + verdict | Highest-accuracy open judge on RewardBench-hard at publication | Local inference, CoT-heavy |
| **Self-Rewarding / Meta-Rewarding** ([[self-rewarding-lm]], [[meta-rewarding-lm]]) | 2024 | Policy acts as own judge; meta-judge evaluates judge | 5-point rubric + rationale | Judge Spearman with humans rises 0.62 → 0.71 (SR); AlpacaEval 22.9% → 39.4% (MR) | In-stack; RL use primary |
| **PairRM** ([[pairrm]]) | 2023 | BCE on joint-input pairs with swap-augmentation | Scalar logit | 0.4B param matches scalar-RM-7B; tournament O(N log N) | Cheapest per comparison (encoder-only) |

The Self-Taught Evaluator trajectory, attested in [[direct-judgement-preference]], deserves a direct stare. Round 0 is a tiny labeled seed; round k uses judge_k to label a fresh pool, DPO-trains judge_{k+1} on contrastive judgment pairs (chosen verdict vs a deliberately alternative verdict), and repeats. RewardBench accuracy crosses the GPT-4-as-judge baseline at roughly iteration 3 and saturates a couple of iterations later. Con-J's noisy-negative trick — perturbing the instruction, generating a response to the noisy instruction, treating it as a plausible "rejected" — produces contrastive pairs *without* needing any original preference label, which is what collapses the dependency on human or API data end-to-end.

J1 is the same judge, trained further with RL on its chain-of-thought: the judgment rollout is scored against held-out reference verdicts, and GRPO-style optimization pushes the CoT toward higher-agreement reasoning. The move is identical to RLVR on math — verifiable outcome (the verdict matches the reference), optimize the reasoning that produces it. It works because the verdict-token log-prob supplies the verifier signal, closing the loop [[generative-reward-models]] opened.

Four reasons GPT-4-as-judge is being replaced, each attested:

**a) Eval-RL leakage.** If GPT-4 labels your DPO data ([[ultrafeedback-construction]]) *and* GPT-4 judges your eval, the model is rewarded at train time for matching the same bias that scores it at eval time. [[direct-judgement-preference]] calls this "judge-as-weapon": "the same judge model used to label prefs and evaluate benchmarks creates leakage — RewardBench's increasing close relationship to training-time judges is a known measurement issue." Eval-time judge must differ from RL-time RM.

**b) Reproducibility.** GPT-4's weights drift silently across API versions. A MT-Bench score in 2023 vs 2025 with "GPT-4-as-judge" is not the same measurement. An owned judge freezes the instrument.

**c) Cost.** [[ultrafeedback-construction]] logged ~1M GPT-4 annotations at "tens of thousands USD". Self-Taught Evaluator on [[direct-judgement-preference]] demonstrates "~40K synthetic preference pairs (20K SFT + 20K DPO) suffice to beat models trained with 2–40× more data" — zero marginal API spend.

**d) Self-enhancement bias at the ecosystem level.** When GPT-4 judges GPT-4 descendants against non-GPT-4 models, the diagonal in [[judge-llm-bias]] Fig. 5 becomes a systemic pollutant. Cross-family judges (Claude-judge-of-GPT, Llama-judge-of-Qwen) are the current mitigation.

---

## 6. Judge-RM vs generative-RM — when eval-time judge ≠ RL-time RM

This is the chapter's deepest subtlety. The reader should finish the chapter able to explain, unprompted, why you want *two different judges* in a mature stack:

- **RL-time RM:** fast (per-token scoring in GRPO inner loop), differentiable-ish (scalar or log-odds), used at every gradient step. [[pairrm]]-style joint-input scoring or [[generative-reward-models]] with short-rubric CoT fits. The optimizer will *hack* it; that is expected. The KL-to-reference ([[reward-hacking-taxonomy]]) bounds how far.

- **Eval-time judge:** slow (once per checkpoint × benchmark), auditable, rubric-rich. [[generative-reward-models]] with full critique + verdict, or a human-anchored Self-Taught judge on an external model family. Used for go/no-go. If this judge is the same model that scored RL, every reward hack is invisible.

[[generative-reward-models]] gives the key identity: reward = `log P_RM("A is better" | x, y_A, y_B, rubric)` — soft log-odds. This is *structurally* the same shape as [[rlaif-scaling]]'s d-RLAIF reward (`r = log P_labeler("Better")`). So "generative RM" and "LLM-as-judge" are not different *math*; they differ only in how you *use* the same signal. Judge-RM is training-time; generative-RM is a family of constructions covering both. The mature stack runs the same construction at both phases with *different models*, different rubrics, and held-out data that never trained the eval-time judge.

Concrete rule: if you cannot answer "which model family judges at train? which at eval?" in one sentence, your eval is leaky.

A worked 2025-era example: a team trains a 7B policy with DPO on [[ultrafeedback-construction]] prefs (GPT-4 judge at train time), then scores checkpoints on Arena-Hard (GPT-4-Turbo judge at eval time). Shared family: GPT-4. Any length/self-enhancement bias GPT-4 has is rewarded at train time *and* counted as quality at eval time. The fix is not "pick a different benchmark"; it is to run a parallel eval with a Claude or Llama-judge pass, and report both numbers. Disagreement between the two passes is a first-class calibration signal — if Claude-judge shows +4 pp while GPT-4-judge shows +12 pp, eight of those twelve points are leakage.

---

## 7. Calibration curves and confidence

A judge's verdict carries a confidence — the log-prob margin between "A" and "B" at the verdict position. [[generative-reward-models]] Fig. 4: GenRMs are "well-calibrated where BT RMs are overconfident." Calibration means: bin pairs by the judge's reported margin (e.g. deciles of `|log P(A) − log P(B)|`); plot empirical human-agreement per bin; the calibration curve is `y = x` when the judge is calibrated, `y < x` when overconfident.

Three practical uses of calibration:

- **Tie declaration.** Bins with small margin → judge declares tie; Elo update uses small-delta form per [[judge-llm-bias]] tie handling.
- **Abstention.** The lowest-confidence bin is dropped from headline numbers and surfaced separately — a "judge not confident, human spot-check" slice.
- **Ensemble weighting.** Multiple judges, weight by inverse variance; [[generative-reward-models]] "Pairs well with ensembling — GenRM ensembles give calibrated uncertainty."

The [figures/judge-bias.html](figures/judge-bias.html) second panel shows a calibration curve scrubbable by confidence bin, with the `y = x` diagonal drawn as reference.

One subtlety: [[generative-reward-models]] reports that GenRMs are "calibrated where BT RMs are overconfident" — note *where*, not *uniformly*. A GenRM is calibrated only when the rubric explicitly names the dimensions being traded off. If the rubric says "score based on helpfulness" but the pair differs mostly on factual correctness, the judge's log-prob margin reflects a *wrong* axis, and calibration on that axis is meaningless. Calibration is always rubric-conditional. The practical implication: when you publish a calibration curve, publish the rubric hash next to it. Two curves for "the same judge" with different rubrics are different instruments.

Rubric versioning is therefore part of the calibration protocol. A mature stack treats the rubric as source-controlled code: every change bumps a semver, every benchmark number cites a rubric version, and any rubric change triggers a re-anchor against the human-label set (§8). [[meta-rewarding-lm]]'s length-bias control term is the prototype — it is one rubric edit that moved AlpacaEval 2.0 win-rate by double digits across four iterations; you cannot track that effect without versioning.

---

## 8. Anchoring — the one practice that keeps judges honest

Every synthetic-judge pipeline risks **judge-collapse**, [[direct-judgement-preference]]'s analogue of model collapse ([[faithful-synth-eval]]): the judge converges on its own rubric, rationales become post-hoc, and the benchmark number stops tracking actual quality. The mitigation is periodic re-anchoring:

1. Maintain a **human-label anchor set** of 500–2000 pairs, human-judged, never seen by any judge during training.
2. Every new judge version — including every Self-Taught iteration — is evaluated against the anchor before it goes live. Spearman, Kendall, and agreement-above-chance are all reported.
3. If the anchor correlation drops by > 0.05, the new judge is rejected; a re-anchor round (new human labels on recent checkpoints) is ordered.

This is mechanically the [[faithful-synth-eval]] "external-verifier filter" applied to the judge itself. The rule generalizes: **a judge without a living anchor is degrading silently.**

One more failure path worth stating explicitly: **rationale hallucination.** [[direct-judgement-preference]] flags it directly — "natural-language rationales can be post-hoc justifications, not causes." The judge's rationale is plausible and the verdict is still arbitrary. The defense is to evaluate the judge's *verdict accuracy* against the anchor, not the rationale quality. If you grade rationales (as meta-reward pipelines do), grade them against a rubric that is independent of verdict correctness, and cross-check that rationale-score and verdict-accuracy move together over iterations — drift between them is the earliest sign that the rationale has detached from the decision.

---

One last calibration subtlety worth naming: **judge stochasticity**. A single judge call has variance from sampling temperature (if CoT is sampled rather than greedy). [[meta-rewarding-lm]] averages N=11 judge samples per (prompt, actor-response) pair for exactly this reason — one call is noisy, an ensemble of calls is a distribution. At eval time, if you report a benchmark win-rate from a single-sample judge pass, you are reporting a point estimate without its standard error. The per-run noise budget in ch-51 includes this term explicitly; ch-49 is where it enters the conversation.

## 9. Connecting this to the eval stack

- **ch-47 (eval harness design)** — calibrated judge protocol is the per-benchmark spec you register in the harness.
- **ch-48 (contamination)** — judge prompts themselves leak; WildBench and Arena-Hard prompts have been seen by 2025-era judges. Check for overlap.
- **ch-50 (slice analysis)** — judge bias shows up as *per-slice* calibration drift; aggregate accuracy hides it ([[faithful-synth-eval]] principle).
- **ch-51 (metric noise + go/no-go)** — judge variance per prompt is a noise source; bootstrap CI includes judge seed.
- **ch-42 (reward-hacking taxonomy)** — the judge can be hacked at eval time just like an RM at train time; the fixes differ.
- **ch-44 (RLVR)** — on verifiable domains, swap the judge out entirely for a verifier. RLVR is the structural escape from judge calibration.

---

## 10. The takeaway rule

A judge is instrument-grade only when, for every number it produces, you can name (a) the judge model, (b) the rubric file hash, (c) the template, (d) the swap protocol, (e) the length-control method, (f) the anchor set and last-anchor correlation, (g) the RL-time RM it is *not* overlapping with. Seven items. If any are missing, the number is a rumor.

The deeper point: a judge is a *measurement* contract between three parties — the training pipeline (what it rewards), the evaluator (what it measures), and the anchor (what it's audited against). The chapters that follow — slice analysis (ch-50), metric-noise and go/no-go (ch-51) — live downstream of this contract. Every failure they analyse starts with a judge number that was not recorded with its seven items.

A common failure pattern worth stating: a team reports "our model scored 8.4 on MT-Bench," and a reviewer cannot reproduce. The judge was a specific GPT-4 snapshot that the API has silently rotated; the rubric had a minor edit that was never versioned; the swap-protocol was implemented but disabled for speed. Three silent drifts, one irreproducible number. The fix is procedural — every judge number ships with its seven items in a YAML header — and it is the contract this chapter argues you cannot run an eval track without.

Finally: the chapter has not discussed multi-turn judging, tool-use judging, or long-context judging. Those are active research frontiers as of 2025 (WildBench covers multi-turn; Arena-Hard-agent covers tool-use; LongBench covers long-context). Their bias inventories are extensions of the quartet in §3 — position bias in multi-turn becomes *turn-position* bias; verbosity becomes *turn-count* bias — but the calibration protocol is the same shape. When you build a judge for one of these frontiers, the starting point is this chapter's seven-item contract; the novelty is which of the items need to be generalized.

---

## Further reading

- [[judge-llm-bias]] — the bias canon; every number in §3 cites Zheng 2023 Figs/Tables.
- [[generative-reward-models]] — verdict-log-prob = reward; calibration claim.
- [[direct-judgement-preference]] — Con-J / STE / J1 synthetic-judge line; judge-as-weapon risk.
- [[rlaif-scaling]] — CoT prompt template; soft-label extraction; d-RLAIF reward.
- [[ultrafeedback-construction]] — 4-aspect rubric; judge-induced bias propagation; model-fleet contamination.
- [[pairrm]] — swap-augmentation; tournament Best-of-N; 0.4B joint encoder.
- [[self-rewarding-lm]] + [[meta-rewarding-lm]] — judge Spearman evolution; meta-judge role; length-bias rubric control.
- [[faithful-synth-eval]] — external-verifier anchor principle applied to judges.
- [[reward-hacking-taxonomy]] — unhackability impossibility transfers: no rubric is un-gameable.
- [[wildchat]] — real-user logs as WildBench prompt source; realism anchor.

## Companion visualization

**[figures/judge-bias.html](figures/judge-bias.html)** — self-contained. Panel 1: pick bias (position / verbosity / self-enhancement), compare naive-judge accuracy vs corrected-judge accuracy across scenarios, with Zheng 2023 attested numbers on position-flip rate. Panel 2: calibration curve, scrubbable by confidence bin, with reference `y = x` line and a toggle between "BT RM" (overconfident) and "GenRM" ([[generative-reward-models]] Fig. 4 shape).
