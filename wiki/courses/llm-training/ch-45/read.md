<!-- chapter: ch-45
     track: rl
     kind: content
     title: Self-Improvement Loops and Multi-Stage Reasoning Pipelines
     deps: [ch-44b]
     sources: [[self-rewarding-lm]], [[meta-rewarding-lm]], [[spin]], [[self-play-preference]],
              [[self-correct-rl]], [[rest-em]], [[star]], [[v-star]], [[deepseek-r1]], [[deepseek-r1-recipe]],
              [[open-reasoner-zero]], [[cognitive-behaviors-self-improving-reasoners]], [[dr-grpo]],
              [[rlvr-beyond-base-model]], [[prorl]], [[deepseek-v3.1]], [[agent-early-experience]], [[swe-gym]],
              [[model-collapse]], [[likelihood-displacement]]
     figures: figures/self-improve-loop.html
     revised: 2026-09 (generality revision)
-->

# Chapter 45 — Self-Improvement Loops and Multi-Stage Reasoning Pipelines

> **Core insight.** A self-improvement loop samples from the current model, scores the samples with a signal the model or the environment produces, and trains on the result. What the loop improves is the metric the scorer stands for, which is not always held-out ability. Judge-scored loops raised their own judge-scored metric by 10 points or more while leaving, or lowering, unrelated benchmarks in the same runs: Self-Rewarding moved AlpacaEval 2.0 win rate 9.94% → 15.38% → 20.44% over its three models while GSM8K went 50.72 → 60.27 → 59.29 → 57.70 and Natural Questions 34.35 → 35.48 → 33.07 → 31.86 (Llama 2 70B; [[self-rewarding-lm]] Tables 1, 3), and Meta-Rewarding moved AlpacaEval 2 length-controlled win rate 22.92% → 39.44% while judge agreement with humans peaked at iteration 2 and fell by iteration 4 (Spearman 0.315 → 0.382 → 0.326; [[meta-rewarding-lm]] Tables 1, 7). Verifier-scored loops behave differently but are not exempt: over GRPO steps 150 → 450 on Qwen2.5-7B, pass@1 on the training split rose 26.1 → 42.5 while pass@256 on the held-out Omni-MATH-Test split fell 68.3 → 63.9, below the base model's 69.1 ([[rlvr-beyond-base-model]] Table 4). The general-purpose models built this way are not one loop but a staged pipeline: DeepSeek-R1-Zero, the RL-only model, scores 46.6 on IF-Eval and 24.7 on AlpacaEval 2.0, and the four-stage DeepSeek-R1 pipeline ends at 83.3 and 87.6 while giving up GPQA Diamond (75.8 → 71.5) and CNMO 2024 (88.1 → 78.8) ([[deepseek-r1]] Table 3).
>
> **Guideline.** When a loop's scorer is the model itself or a model-derived judge, run a fixed small number of iterations, measure a held-out capability panel and a judge-independent metric after every iteration, and stop at the iteration that is best on the panel rather than on the loop's own metric, because in the published runs the loop metric kept rising while judge-human agreement ([[meta-rewarding-lm]] Table 7), held-out NLP benchmarks ([[self-rewarding-lm]] Table 3) and pass@256 ([[rlvr-beyond-base-model]] Table 4) had already turned down. When the scorer is an execution or answer verifier, the same rule applies with the verifier's own held-out split and with pass@k at large k, because ReST-EM's APPS and HumanEval results regressed after the first iteration although training accuracy kept rising ([[rest-em]] §5.1, Fig. 4). When the target is a generally capable model rather than a reasoning specialist, do not ship the output of a single loop: follow it with SFT on data that covers the non-target domains and a final RL stage that mixes rule rewards with preference rewards, because that is the measured difference between R1-Zero and R1 ([[deepseek-r1]] Table 3, §3). When a loop's negatives come from earlier iterations, refresh them each iteration and log chosen and rejected log-probabilities separately, because negatives already pushed down contribute little gradient and stale negatives are off-policy (mechanism in ch-43a; the off-policy negative-masking control in [[deepseek-v3.1]] §3.1).

## Why this chapter matters for a general-purpose model

Every stage before this one needs targets or rewards that somebody produced: human demonstrations (ch-15), teacher models (ch-35), reward models (ch-41), rule verifiers (ch-44). A self-improvement loop removes that dependency for one stage by letting the model produce both the samples and, in part, the signal that ranks them. The question this chapter answers is narrow and measurable: when does that arrangement raise ability on tasks the loop never scored, and when does it raise only the loop's own number.

This matters for breadth in two directions. A loop that scores with a judge trained on the loop's own distribution can drift: the judge and the policy move together, so agreement with fixed external labels is the only way to see the drift. A loop that scores with a verifier cannot drift in that way, but it narrows the output distribution toward paths the verifier already accepts, which shows up as falling coverage at large pass@k while pass@1 rises.

The chapter sits after multi-domain RL (ch-44b) and before the recipe comparison (ch-45a). It uses the mechanics of rejection-sampling rounds from ch-31 without repeating them, and it uses the DPO objective from ch-39 and group-baseline RL from ch-40. The pipeline view — reasoning RL, then rejection-sampling SFT, then a general RL stage — is the part that turns a reasoning loop into a general model, and it is the part most often dropped when the loop is described on its own.

## §1 What differs between loops: the scorer, the failures, and the restart point

Every method in this chapter can be written as: sample from a policy, attach a score to each sample, and update. Four design choices differ, and they are the reason the outcomes differ.

1. **Score source.** The policy acting as a judge ([[self-rewarding-lm]]), the policy judging its own judgments ([[meta-rewarding-lm]]), the human-written response used as an implicit positive ([[spin]]), a learned pairwise preference model ([[self-play-preference]]), an answer or test verifier ([[rest-em]], [[deepseek-r1]]), or the environment's own next state ([[agent-early-experience]]).
2. **Treatment of failures.** Discarded ([[rest-em]], [[star]], R1's stage-3 rejection sampling), used as the rejected term of a preference loss ([[self-rewarding-lm]], [[spin]]), used as a negative advantage inside a policy gradient ([[deepseek-r1]]), used to train a separate verifier ([[v-star]]), or placed in the context of a correction target ([[self-correct-rl]], [[agent-early-experience]]). §"Negative samples and negative feedback" treats this in full.
3. **Starting checkpoint of the next iteration.** ReST-EM and STaR fine-tune the *base pretrained model* each iteration ([[rest-em]] §3, Table 1; [[star]] §3.1). Self-Rewarding and SPIN continue from the previous iterate ([[self-rewarding-lm]] §2.4; [[spin]] Algorithm 1). R1-Zero is a single online RL run with no iteration loop; its reference model is replaced by the latest policy every 400 steps ([[deepseek-r1]] §2.1).
4. **Reference or opponent.** SPIN's opponent is π_t, the previous iterate ([[spin]] Alg. 1). Nash-MD's opponent is a geometric mixture of the current policy and the initial policy ([[self-play-preference]] Eq. 3). GRPO's KL reference is the periodically refreshed policy ([[deepseek-r1]] §2.1). ReST-EM has no reference model at all — its M-step is ordinary SFT ([[rest-em]] Alg. 1).

| Loop | Score source | Failures | Next iteration starts from | Iterations run | Locus |
|---|---|---|---|---|---|
| Self-Rewarding | policy-as-judge, 5-point rubric, 3 judge samples averaged | lowest-scored sample is the DPO rejected term | previous iterate | M1 (SFT) + 2 DPO iterations | [[self-rewarding-lm]] §2.4, §3.1.3 |
| Meta-Rewarding | policy-as-judge plus a meta-judge over judgments | rejected response and rejected judgment (both DPO) | previous iterate | 4 (judge trained in 1–2 only) | [[meta-rewarding-lm]] §3.1, App. A.3 |
| SPIN | the human SFT response is the positive; the model's own sample is the negative | negative term of the SPIN loss | previous iterate | iteration 0 through 3 | [[spin]] §6.1, Alg. 1 |
| Nash-MD | learned pairwise preference model, self-play | no explicit rejected set; preference margin is the reward | continuous training (10,000 steps) | — | [[self-play-preference]] §7, App. G.4 |
| SCoRe | binary outcome check on a two-turn rollout | first attempt stays in the context; c→i transitions are penalized | two stages, continuous RL | 2 stages | [[self-correct-rl]] §5 |
| ReST-EM | answer match or unit tests | discarded (zero weight) | base pretrained model | MATH 3, APPS 2 | [[rest-em]] §3, §5.1 |
| R1-Zero | rule reward: accuracy + format | negative group-relative advantage | single RL run, reference refreshed every 400 steps | 10,400 steps | [[deepseek-r1]] §2.1, §2.2 |

A single generate-filter-retrain skeleton with a reference model and a reset rule in every row does not describe these methods. ReST-EM and STaR restart from the base model and have no reference model, and R1-Zero is one online RL run rather than an iterated loop. Rows 3 and 4 of the table are the ones to read before transferring a result from one method to another.

## §2 Judge-scored loops: Self-Rewarding and Meta-Rewarding

### 2.1 Self-Rewarding: the policy scores its own samples

**Definition.** Self-Rewarding training uses one model in two roles: it generates candidate responses, and, under a fixed LLM-as-a-Judge prompt, it assigns each candidate a score from a 5-point additive rubric. The highest- and lowest-scored candidates form a DPO pair ([[self-rewarding-lm]] §2).

**Problem addressed.** A frozen reward model trained on human preferences stops improving while the policy improves, and collecting new human preferences per iteration is expensive. The measurable claim is that both instruction-following and reward-modeling ability rise across iterations (§1).

**Mechanism.**
1. Fine-tune Llama 2 70B on 3,200 Open Assistant instruction examples (IFT) and 1,630 evaluation examples (EFT) that teach the judge rubric. This model is M1 (§3.1.1).
2. Generate new prompts with a fixed model, Llama 2-Chat 70B, 8-shot, T = 0.6, p = 0.9 (§3.1.3).
3. Sample N = 4 candidate responses per prompt from the current model at T = 0.7, p = 0.9.
4. Score each candidate with the same model as judge, sampling the judgment 3 times and averaging (§3.1.3).
5. Take the highest-scored candidate as chosen and the lowest as rejected; discard the pair if the two scores are equal (§2.3).
6. DPO-train the next model from the current one: β = 0.1, learning rate 1e−6 decaying to 1e−7, batch 16. 3,964 pairs produce M2 and 6,942 pairs produce M3 (§3.1.3).

**Worked example of step 5.** Four candidates with three judge samples each: A (5, 4, 5) → 4.67, B (3, 4, 3) → 3.33, C (4, 4, 4) → 4.00, D (2, 3, 2) → 2.33. Chosen = A, rejected = D, margin 7/3 = 2.33 rubric points. If the judge had returned 4 for all four candidates, the pair would be discarded. The rubric has 5 levels, so a judge whose scores concentrate at the top produces fewer usable pairs and smaller score gaps; [[meta-rewarding-lm]] measures that concentration directly (mean judge score rose from 4.1 to above 4.7 after two iterations of judge training, Figure 5). In this paper the pair count still rose across iterations, from 3,964 to 6,942, so the effect was not observed here (§3.1.3).

**Evidence (Llama 2 70B, Open Assistant prompts).**

| Metric | Scored by | SFT baseline | M1 | M2 | M3 |
|---|---|---|---|---|---|
| AlpacaEval 2.0 win rate vs GPT-4 Turbo (Table 1) | GPT-4 judge | — | 9.94% | 15.38% | 20.44% |
| MT-Bench overall /10 (Table 2) | GPT-4 judge | 6.85 | 6.78 | 7.01 | 7.25 |
| MT-Bench math, code, reasoning (Table 2) | GPT-4 judge | 3.93 | 3.83 | 4.05 | 4.17 |
| Judge pairwise accuracy vs held-out human rankings (Table 4) | human labels | 65.1% | 78.7% | 80.4% | 81.7% |
| Judge Spearman correlation with humans (Table 4) | human labels | 0.253 | 0.279 | 0.331 | 0.349 |
| GSM8K (Table 3) | benchmark | 50.72 | 60.27 | 59.29 | 57.70 |
| ARC-Challenge (Table 3) | benchmark | 55.97 | 57.51 | 54.51 | 53.13 |
| Natural Questions (Table 3) | benchmark | 34.35 | 35.48 | 33.07 | 31.86 |
| MMLU (Table 3) | benchmark | 69.76 | 69.34 | 69.31 | 69.37 |
| Average AlpacaEval generation length (§3.2.1) | — | — | 1,092 | 1,552 | 2,552 |

For reference, GPT-4 0613 scores 15.76% on the same AlpacaEval 2.0 table, so M3 is above it and M2 is 0.38 points below it (Table 1). Status: Result (single study, one base model, one seed prompt source).

**Conditions and limits.** Three models means one SFT model and two DPO iterations; the paper reports no fourth iteration and no regression at a fourth iteration. The authors list understanding the limits of iterative training as open, note the length growth above, and state that reward hacking inside the framework is not measured because both the training reward and part of the evaluation come from language models (§6). The seed prompts are Open Assistant first turns, which the authors give as the reason math and reasoning gains are small (Table 2 caption).

**Implication for a general-purpose model.** The loop moved a judge-scored preference metric by 10.5 points and moved three of four held-out benchmarks down from their M1 values in the same run. Reporting only the first number is how a narrowing loop is mistaken for a general one. The judge does improve against fixed human labels (65.1% → 81.7% pairwise accuracy). That figure also gives an estimate of the label noise entering DPO: at M3 roughly one pair in five is ordered against the human ranking. The estimate is measured on held-out human-written responses rather than on the model's own candidates, and [[meta-rewarding-lm]] §5 reports that judge improvement is smaller on responses the model did not generate, so it is a lower bound on the noise in the actual training pairs rather than a measurement of it.

### 2.2 Meta-Rewarding: training the judge role, and what saturates

**Definition.** Meta-Rewarding adds a third role: the model compares two of its own judgments of one response and picks the better judgment, producing preference pairs for the judge as well as for the actor ([[meta-rewarding-lm]] §2.2).

**Mechanism (per iteration, Llama-3-8B-Instruct; §2.1, §3.1).** Sample K = 7 responses per prompt (T = 0.8, top-p 0.95); sample N = 11 judgments per response; average the valid 5-point scores; select the actor pair with length control; for the response whose 11 judgment scores have the highest variance, run meta-judge comparisons in both orders, weight by position, fit Elo scores, and take the highest and lowest judgments as the judge pair; train both pair types with DPO (β = 0.1, learning rate 5e−6, batch 32).

**Length control, with a worked example.** With S_max and S_min the highest and lowest response scores for a prompt, the chosen response is the *shortest* response whose score lies in [(1−ρ)S_max + ρS_min, S_max] and the rejected response is the *longest* whose score lies in [S_min, (1−ρ)S_min + ρS_max] (§2.1). Take S_max = 5.0, S_min = 2.0, ρ = 0.32, and five responses: (5.0, 1,900 characters), (4.2, 1,500), (3.5, 1,200), (2.5, 2,400), (2.0, 1,800). The chosen tier is [4.04, 5.0] and the rejected tier is [2.0, 2.96]. Chosen = the 4.2/1,500 response, rejected = the 2.5/2,400 response. With ρ = 0 the tiers collapse to the extremes: chosen = 5.0/1,900, rejected = 2.0/1,800. The measured effect at iteration 4 is the response length: 2,212 characters at ρ = 0 against 2,003 at ρ = 0.4, with the ρ = 0.4 run scoring 39.44% length-controlled win rate (Table 4).

**Evidence (Llama-3-8B-Instruct).** AlpacaEval 2 length-controlled win rate: seed 22.92%, after SFT on the EFT set 25.47%, then 27.85 / 32.66 / 35.45 / 39.44% over four iterations; the Self-Rewarding baseline with the same length control reaches 35.49% (Table 1). Arena-Hard 20.6% → 29.1% (Table 2). Judge agreement with GPT-4 on self-chosen pairs at iteration 2, counting only pairs where neither judge returned a tie: 72.34% against 60.00% for the Self-Rewarding baseline at the same iteration; including ties the two values are 61.57% and 54.89% (Table 3). Judge Spearman correlation with humans: seed 0.315, iteration 2 0.382, iteration 4 0.326 (Table 7). MT-Bench turn 1 8.319 → 8.738 but turn 2 7.911 → 7.838 (Table 6). Meta-judge preference for the higher-scored judgment rose from 63.04% at iteration 1 to 97.68% at iteration 2, and positional bias from 43.92% to 68.11% (Table 5); the authors state that positional bias hindered further improvement at iteration 3 (§5).

**Conditions and limits.** Judge training runs only in iterations 1–2; iterations 3–4 train actor pairs only (§3.1). The training prompts are single-turn and closer to AlpacaEval than to Arena-Hard (App. Figure 6). The authors report limited judge improvement on responses the model did not generate (§5).

**Implication.** Two metrics in the same run move in opposite directions after iteration 2: the loop's own judge-scored win rate keeps rising, and agreement between the judge and humans falls. A stopping rule based on the loop metric alone stops too late. The figure at [figures/self-improve-loop.html](figures/self-improve-loop.html) plots these per-iteration series side by side so the divergence point can be read off directly.

## §3 SPIN: the model's current distribution as the negative

**Definition.** SPIN (self-play fine-tuning) builds preference pairs without any preference labels: the human-written SFT response is the positive, and a sample drawn from the previous iterate for the same prompt is the negative ([[spin]] §4.1).

**Problem addressed, measured.** Running more SFT epochs on the same corpus stops helping: a second epoch of UltraChat200k on zephyr-7b-sft-full lowered the Open LLM Leaderboard average from 58.14 to 57.23 (Table 5).

**Mechanism.** At iteration t: (1) sample y′ ~ p_{θ_t}(·|x) for each SFT prompt; (2) minimize

```
L_SPIN(θ, θ_t) = E_{x~q, y~p_data(·|x), y'~p_{θ_t}(·|x)} ℓ( λ·log[p_θ(y|x)/p_{θ_t}(y|x)] − λ·log[p_θ(y'|x)/p_{θ_t}(y'|x)] )
```

where x is a prompt, y the human response, y′ the sampled response, p_data the SFT data distribution, p_{θ_t} the previous iterate (the opponent, which also serves as the reference), λ > 0 the regularization parameter, and ℓ a convex decreasing loss; with ℓ(t) = log(1 + e^{−t}) this is the DPO loss with λ in the role of β ([[spin]] Eq. 4.7, §4.2).

**What the fixed point is.** Theorem 5.2: p_{θ_t} = p_data is a global minimum, and if p_{θ_t} ≠ p_data there is a λ for which θ_t is not a global minimum. Theorem 5.4 characterizes one step under the logistic loss: p_{θ_{t+1}}(y|x) ∝ p_{θ_t}(y|x)·[p_data(y|x)/p_{θ_t}(y|x)]^{1/λ}.

**Worked example of Theorem 5.4.** Three responses with p_data = (0.5, 0.3, 0.2) and p_{θ_t} = (0.2, 0.3, 0.5). With λ = 1 the update is exactly p_data. With λ = 2 the unnormalized update is √(p_{θ_t}·p_data) = (0.316, 0.300, 0.316), which normalizes to (0.339, 0.322, 0.339): the under-represented response moves from 0.20 to 0.34 and the over-represented one from 0.50 to 0.34. With λ = 0.1 the exponent is 10 and the same formula gives (0.9998, 0.0002, ~3e−8): a smaller λ produces a larger change in one step, which is what Remark 5.5 states; the step can carry the policy past p_data, as it does here, so the size of the step and the location of the fixed point are separate facts. The paper uses λ = 0.1 for iterations 0–2 and raises it to 5.0 at iteration 3, "where the model is close to convergence" (App. B.1).

**Evidence (zephyr-7b-sft-full, that is Mistral-7B fine-tuned on UltraChat200k; 50k prompts).**

| Model | Open LLM avg | ARC | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU | MT-Bench |
|---|---|---|---|---|---|---|---|---|
| zephyr-7b-sft-full | 58.14 | 60.41 | 43.73 | 74.19 | 26.76 | 82.85 | 60.92 | 5.94 |
| SPIN iteration 0 | 60.80 | 63.40 | 49.18 | 72.69 | 35.10 | 84.38 | 60.03 | 6.46 |
| SPIN iteration 1 | 62.12 | 65.19 | 55.17 | 72.30 | 35.78 | 84.96 | 59.34 | 6.65 |
| SPIN iteration 2 | 62.97 | 65.96 | 54.91 | 73.56 | 38.06 | 85.41 | 59.93 | 6.78 |
| SPIN iteration 3 | 63.16 | 65.87 | 54.90 | 73.72 | 38.97 | 85.54 | 59.99 | not reported |

Values from [[spin]] Tables 4 and 6 (MT-Bench and three Big-Bench-Hard tasks are Table 6). The per-iteration gains fall: +2.66, +1.32, +0.85, +0.19. MMLU is below the starting checkpoint at every iteration (60.92 against 59.34 to 60.03, a gap of 0.89 to 1.58 points), and Winogrande does not recover its starting value. Status: Result (single study).

**Conditions and limits.** The authors state the limitation directly: the target distribution is fixed and human-generated, which "inherently imposes a ceiling on the performance of fine-tuned LLM" (§7). The synthetic set accumulates — 50k at iteration 0 and 100k at iterations 1–3, combining the previous iteration's samples with the current ones (§6.1) — so part of the negative set at iteration t was generated by π_{t−1} and part by π_{t−2}.

**Implication.** SPIN is the clearest case in this chapter where the loop has a known fixed point that is not "better": it is "equal to the SFT data distribution". The breadth gains it reports come from moving toward that distribution more effectively than repeated SFT epochs do, not from exceeding it.

## §4 Nash learning from human feedback: an equilibrium instead of an argmax

**Definition.** NLHF replaces the reward model with a pairwise preference model P(y ≻ y′ | x) and seeks the policy π* = arg max_π min_{π′} P(π ≻ π′), the Nash equilibrium of the induced two-player constant-sum game ([[self-play-preference]] Eq. 1).

**Problem addressed.** A Bradley-Terry reward model assigns one scalar score per response, so the induced preference is stochastically transitive: with scores r_a > r_b > r_c, P(a ≻ b) = σ(r_a − r_b), P(b ≻ c) = σ(r_b − r_c) and P(a ≻ c) = σ(r_a − r_c) ≥ max of the two. A population preference can violate this. In the paper's example, three groups of annotators with the population weights P(type 1) = 1/3 − ε and P(type 2) = P(type 3) = 1/3 + ε/2 give P_ε(y2 ≻ y1) = 2/3 − ε/2, P_ε(y3 ≻ y2) = 2/3 + ε/4, and P_ε(y3 ≻ y1) = 1/3 − ε/4 (§3.2).

**Worked example.** With ε = 0.03: P(y2 ≻ y1) = 0.652, P(y3 ≻ y2) = 0.674, P(y3 ≻ y1) = 0.326. Two of the three comparisons prefer the later response and the third reverses, so no single scalar score reproduces all three. The Nash equilibrium given in §3.2 is y1 and y2 with probability 1/3 + ε/2 = 0.348 each and y3 with probability 1/3 − ε = 0.303. Check y1 against that mixture: 0.5(0.348) + 0.348(0.348) + 0.674(0.303) = 0.174 + 0.121 + 0.204 = 0.500. The same computation for y2 and y3 also gives 0.500, which is the equilibrium condition. A Bradley-Terry reward fitted to the same population gives y1 a slightly higher score when ε > 0, and maximizing it produces a deterministic policy that always answers y1; at ε < 0 that deterministic answer flips to y2 (§3.2).

**Mechanism (Nash-MD).** Define the opponent as a geometric mixture of the current policy and the reference policy μ: π_t^μ(y) ∝ π_t(y)^{1−η_t τ}·μ(y)^{η_t τ}, then take a mirror-descent step against it, π_{t+1} = arg max_π [η_t P(π ≻ π_t^μ) − KL(π, π_t^μ)] (Eqs. 3–4). Here η_t is the learning rate, τ the KL regularization strength, and μ the initial (SFT) policy. Theorem 1 bounds KL(π*_τ, π_{t+1}) ≤ (1 − η_t τ)KL(π*_τ, π_t) + 2η_t², so the last iterate converges at O(1/T), unlike fictitious-play schemes where only the average of past policies converges. The deep-learning form, Nash-MD-PG, is a policy gradient with the preference margin as reward: ∇_θ log π_θ(y|x)·[P(y ≻ y′|x) − 1/2 − τ·log(π_θ(y|x)/μ(y|x))] (§7).

**Evidence.** T5X-L policies fine-tuned on the TL;DR summarization data, preference model from a T5X-XL model, 10,000 steps, τ = 0.008 for the Nash runs and τ = 0.05 for the RLHF baseline (App. G). Judged by a PaLM 2 Large preference model, Nash-MD-PG with mixing coefficient β = 0.125 — β is the weight of the initial policy μ in the opponent mixture, so β = 0 is self-play and β = 1 is best response against μ (§8) — is preferred over the RLHF baseline in 59.8% of comparisons and over pure self-play (β = 0) in 59.2%; β values between 0.125 and 0.375 beat both endpoints, pure self-play (β = 0) and best-response against the fixed SFT policy (β = 1) (Table 1, §8). Each cell of that table is estimated from 2,000 samples, so the 95% interval is at most ±0.023 wide (App. G.6); the separate table scored by the training preference model uses 1,000 comparisons and ±0.032 (App. G.5, Table 2). Status: Result (single study, one task).

**Conditions and limits.** One task (summarization), one model size, and an evaluation that is itself a preference model. No held-out capability panel is reported. The mixture parameter matters: playing only against yourself (β = 0) is worse than playing against a mixture that keeps some of the initial policy.

**Implication.** Self-play against a fixed opponent (the SFT model) and self-play against yourself are two ends of one axis, and the measured best setting is between them. Loops in §2 and §3 sit at the "against yourself" end by construction, which is one reason their negatives go stale.

## §5 SCoRe: training the revision step itself

**Definition.** SCoRe trains a model to answer, then revise its own answer, using multi-turn RL on its own two-turn rollouts. Turn 2 is conditioned on the question, the turn-1 answer, and an instruction that does not say whether the answer was wrong ([[self-correct-rl]] §5, App. C).

**Problem addressed, measured.** Gemini 1.5 Flash on MATH500 has Accuracy@t1 52.6% and Accuracy@t2 41.4%, so Δ(t1, t2) = −11.2%: prompting the model to revise makes it worse, because it changes 15.8% of correct answers to incorrect and fixes only 4.6% of incorrect ones (Table 1). SFT on self-generated correction traces does not fix this: STaR-filtered traces give Δ = 0.4% and pair-SFT gives Δ = 1.8% (Table 1).

**Mechanism.**
1. **Stage I** maximizes the second-attempt reward while holding the first attempt near the base model: max_θ E[r̂(y2, y*) − β2·KL(π_θ(·|x1) ‖ π_ref(·|x1))], with a smaller default KL penalty on both turns (Eq. 3). β2 is 0.1 for MATH and 0.25 for MBPP (App. B Table 5).
2. **Stage II** optimizes both attempts jointly, max_θ E[Σ_{i=1,2} r̂(y_i, y*) − β1·KL(π_θ(·|x_i) ‖ π_ref(·|x_i))] (Eq. 4), with a shaping bonus added to the second attempt: b̂(y2 | y1, y*) = α·(r̂(y2, y*) − r̂(y1, y*)), α = 10 in both settings (§5.2, App. B Table 5).
3. All experiments use the instantaneous reward only, equivalent to a discount factor γ = 0 (App. A), so the bonus reaches the gradient of the second-attempt tokens and not the first-attempt tokens.

**Worked example of the shaping term.** With binary rewards and α = 10, the reward attached to the second attempt is r̂2 + 10(r̂2 − r̂1): incorrect→correct gives 1 + 10 = 11; correct→correct gives 1 + 0 = 1; correct→incorrect gives 0 − 10 = −10; incorrect→incorrect gives 0. Without the bonus those four values are 1, 1, 0, 0, so keeping a correct answer and fixing a wrong one carry the same weight and the model has no gradient reason to learn the harder of the two. Because γ = 0, the first attempt's tokens are still trained on r̂1 alone, so the shaping does not pay the model for answering incorrectly on purpose.

**Evidence.** MATH500 with Gemini 1.5 Flash: SCoRe Accuracy@t1 60.0%, Accuracy@t2 64.4%, Δ = +4.4%, with correct→incorrect down from 15.8% to 1.4% (Table 2). Relative to the base model, Δ improves by 15.6 points. HumanEval with Gemini 1.0 Pro trained on MBPP: base Δ = 3.0% and SCoRe Δ = 12.2% (Table 3), with MBPP-R repair accuracy 47.3% → 60.6% (Table 3). The abstract states this gain as 9.1% absolute and §6.1 as "9% higher than the base model"; the two Table 3 cells differ by 9.2 points. Ablations on MATH (Table 4): without Stage I Δ = 2.2%, without reward shaping Δ = 2.6%, with STaR instead of REINFORCE in Stage II Δ = 2.2%, and single-turn RL gives Δ = −2.4%.

**Conditions and limits.** One round of correction, binary rewards, Gemini models; the MATH training split is augmented with 4,500 problems from the MATH test set and evaluation runs on the remaining 500 (§6); checkpoints are selected by highest training reward (§6). The self-correction result reported in the abstract is measured on HumanEval; MBPP is the training set, and MBPP-R is a separate offline repair task.

**Implication.** The measured target here is a *behavior* (improve between attempts), not an accuracy level, and the behavior has to be protected by two devices: a KL that freezes the first attempt during Stage I, and a shaping term that makes the improvement transition the high-reward event. Both ablations lose more than half of the effect.

## §6 Iteration count and stopping

**The measurable problem.** Every loop in this chapter has an in-loop metric that keeps improving longer than the held-out metric does. Deciding when to stop therefore requires at least two measurements per iteration: the loop's own metric and something the loop does not optimize.

| Loop | In-loop metric per iteration | Independent metric per iteration | Where they diverge | Locus |
|---|---|---|---|---|
| Self-Rewarding | AlpacaEval 2.0 9.94 → 15.38 → 20.44 | GSM8K 60.27 → 59.29 → 57.70; NQ 35.48 → 33.07 → 31.86 | after M1 | [[self-rewarding-lm]] Tables 1, 3 |
| Meta-Rewarding | AlpacaEval 2 LC 27.85 → 32.66 → 35.45 → 39.44 | judge–human Spearman 0.382 (it. 2) → 0.326 (it. 4) | after iteration 2 | [[meta-rewarding-lm]] Tables 1, 7 |
| SPIN | Open LLM avg +2.66, +1.32, +0.85, +0.19 | MMLU 60.92 → 60.03 → 59.34 → 59.93 → 59.99 | at iteration 0 | [[spin]] Table 4 |
| ReST-EM | training accuracy keeps rising | MATH test gains small after iteration 1; APPS and HumanEval regress at iteration 2 | after iteration 1 | [[rest-em]] §5.1, Fig. 4 |
| RLVR (GRPO steps) | train pass@1 26.1 → 33.6 → 42.5 | test pass@256 68.3 → 66.6 → 63.9 (base 69.1) | from the first measured checkpoint | [[rlvr-beyond-base-model]] Table 4 |
| SWE-Gym self-improvement (MoatlessTools) | 7B SWE-Bench Lite resolve 7.0 → 9.0 → 10.0 | 32B SWE-Bench Lite resolve 19.0 → 19.7 → 19.7 | at iteration 2 for 32B | [[swe-gym]] §4.3, Table 4 |

The SWE-Gym row does not fit the two columns: the paper reports no loop-internal metric per iteration, so both cells hold the same held-out metric at two model sizes and the divergence is between sizes rather than between metrics ([[swe-gym]] §4.3).

**Worked example of a stopping decision.** Take the GRPO row ([[rlvr-beyond-base-model]] Table 4, Qwen2.5-7B). At step 150 the model scores 25.1 pass@1 and 68.3 pass@256 on Omni-MATH-Test; at step 450 it scores 28.3 and 63.9. If the acceptance rule is "pass@1 must rise and pass@256 must stay within 1 point of the base model's 69.1", step 150 passes (68.3, difference 0.8) and step 450 fails (63.9, difference 5.2). If the rule is pass@1 only, step 450 wins and the run ships a model that solves fewer distinct problems under repeated sampling. The same table shows training pass@1 rising faster than test pass@1 (26.1 → 42.5 against 25.1 → 28.3), which is the signal that the gain is becoming specific to the training prompts.

**What to measure after every iteration.** (a) The loop's own metric. (b) A held-out capability panel that includes domains the loop never scored — the three benchmarks in [[self-rewarding-lm]] Table 3 are the minimal version. (c) A judge-independent check when the scorer is a judge: agreement with a fixed labeled set, as in [[meta-rewarding-lm]] Table 7. (d) pass@k at a large k on a held-out split ([[rlvr-beyond-base-model]] §4.4). (e) Output length and, where available, entropy: length rose 1,092 → 2,552 in Self-Rewarding (§3.2.1) and is a reported side effect of GRPO's length bias in [[dr-grpo]] §3.1.

**Conditions.** None of the sources reports a stopping rule validated across settings. The iteration counts that appear — three models in Self-Rewarding, four in Meta-Rewarding, four in SPIN, three (MATH) and two (APPS) in ReST-EM — are what those authors ran, not measured optima. Treating them as a general "cap at three" is an extrapolation the sources do not support.

## §7 DeepSeek-R1: why one loop was not enough

### 7.1 R1-Zero and its measured gaps

R1-Zero is GRPO applied directly to DeepSeek-V3-Base with a rule reward (accuracy plus format, equal weight), no SFT, learning rate 3e−6, KL coefficient 0.001, 16 outputs per question, 32 questions per step, maximum length 32,768 tokens rising to 65,536 at step 8.2k, 10,400 steps ([[deepseek-r1]] §2.1–2.2). AIME 2024 pass@1 rises from 15.6% to 77.9% (§2.3). The gaps the authors name are poor readability and language mixing (§1, §3), and Table 3 quantifies the breadth gap: IF-Eval 46.6, AlpacaEval 2.0 24.7, Aider-Polyglot 12.2.

### 7.2 The four stages and what each one moved

| Benchmark | R1-Zero | Dev1 (cold-start SFT) | Dev2 (reasoning RL) | Dev3 (rejection-sampling SFT) | R1 (mixed RL) |
|---|---|---|---|---|---|
| IF-Eval (prompt strict) | 46.6 | 71.7 | 72.0 | 78.1 | 83.3 |
| AlpacaEval 2.0 (LC win rate) | 24.7 | 50.1 | 55.8 | 62.1 | 87.6 |
| ArenaHard | 53.6 | 77.0 | 73.2 | 75.6 | 92.3 |
| Aider-Polyglot | 12.2 | 6.7 | 25.6 | 44.8 | 53.3 |
| AIME 2024 (pass@1) | 77.9 | 59.0 | 74.0 | 78.1 | 79.8 |
| CNMO 2024 (pass@1) | 88.1 | 58.0 | 73.9 | 77.3 | 78.8 |
| GPQA Diamond (pass@1) | 75.8 | 66.1 | 70.7 | 71.2 | 71.5 |
| MMLU-Pro | 68.9 | 74.1 | 83.8 | 83.1 | 84.0 |
| SimpleQA | 30.3 | 17.8 | 28.2 | 24.9 | 30.1 |

All values from [[deepseek-r1]] Table 3. Stage contents ([[deepseek-r1-recipe]]): Dev1 is SFT on "thousands" of long chain-of-thought examples built from R1-Zero samples at temperature 1.0, filtered for correctness and readability, refined by DeepSeek-V3 and checked by annotators (B.3.2); Dev2 is reasoning RL with GRPO clipping ratio ε = 10 plus a language-consistency reward, which is the share of target-language words in the chain of thought (§3.2.1, Eq. 7); Dev3 is SFT from DeepSeek-V3-Base — not from the RL checkpoint — on 804,745 samples: 626,933 reasoning samples (Math 395,285, Code 211,129, STEM 10,124, Logic 10,395) and 177,812 General (B.3.3, Table 5); the final stage is 1,700 RL steps at temperature 0.7 combining rule rewards, a helpfulness reward model, a safety reward model, format and language rewards, with general data and preference rewards only in the last 400 steps (§3.2.2).

**Three readings of the table.** (1) Each stage repairs what the previous stage did not cover: cold start moves instruction following up 25 points and reasoning down 19 points on AIME; reasoning RL restores most of the reasoning; the general SFT stage moves Aider-Polyglot 25.6 → 44.8; the final RL stage moves AlpacaEval 2.0 62.1 → 87.6 with small reasoning changes. (2) The finished model is not uniformly better than the RL-only model: GPQA Diamond 71.5 against 75.8 and CNMO 2024 78.8 against 88.1. (3) Preference rewards are bounded on purpose: with the helpfulness reward model, the reward rises while Codeforces test pass@1 falls, which is why preference rewards run only in the final 400 steps (B.5, Fig. 6; §3.2.2).

### 7.3 What the base model contributes

- **Model capacity.** RL from a base model failed to improve AIME with a 7B dense model and a 16B MoE model — response lengths grew and the models repeated themselves — and worked with 32B dense, 230B MoE and 671B MoE bases ([[deepseek-r1]] G.1).
- **Behaviors already present before RL.** Self-reflection keywords appear in base models including DeepSeek-V3-Base, and Qwen2.5-Math base models answer questions best with no template at all, about 60% above 4-shot prompting ([[dr-grpo]] §2.1–2.3, Table 1). For R1-Zero itself, "nearly half responses with self-reflection do not achieve higher accuracy than those without self-reflection" ([[dr-grpo]] App. F, Fig. 15). The claim that reflection language appears in RL without existing in the pretraining distribution is not supported by these measurements.
- **Which behaviors decide whether RL moves at all.** Under identical PPO training on Countdown, Qwen-2.5-3B reaches about 60% accuracy and Llama-3.2-3B about 30%; priming Llama with traces containing verification, backtracking, subgoal setting or backward chaining closes the gap, and priming with traces that contain the behaviors but *incorrect* solutions performs the same as priming with correct ones; priming with empty chain-of-thought of matched length leaves Llama at 30–35% ([[cognitive-behaviors-self-improving-reasoners]] §3, Figs. 5–6). Continued pretraining on 8.3M tokens of OpenWebMath documents filtered and rewritten to contain the behaviors moves Llama onto Qwen's improvement trajectory (Fig. 8).
- **An open reproduction at scale.** Open-Reasoner-Zero trains Qwen2.5-32B-Base with vanilla PPO, GAE λ = 1 and γ = 1, a correctness-only reward with no format reward, and no KL regularization, reaching AIME 2024 48.1, MATH500 92.2 and GPQA Diamond 55.5 against DeepSeek-R1-Zero-Qwen-32B's 47.0 / 91.6 / 55.0 with about a tenth of the training steps ([[open-reasoner-zero]] Abstract, Table 1). Its breadth check is Table 2: MMLU 83.3 → 84.9 and MMLU-Pro 55.1 → 74.4 relative to the Qwen2.5-32B base, above Qwen2.5-32B-Instruct on both, with reasoning-only RL data. Status: Result (single study); no instruction-following or safety evaluation is reported.
- **Process reward models were tried and dropped.** DeepSeek lists PRMs among unsuccessful attempts: steps are hard to define, step correctness is hard to label, a model-based PRM leads to reward hacking, and the benefit was limited relative to the added compute ([[deepseek-r1]] G.2). This is a single-report result about R1's setting, not a general finding about process supervision (ch-44).

### 7.4 Sharpening or expansion

[[rlvr-beyond-base-model]] evaluates RLVR-trained models and their bases with pass@k at large k across math, code and visual reasoning. RL-trained models win at small k and base models catch up and pass them at large k; on Minerva with a 32B model the base model is about 9 points higher at k = 128 (§3). Across six RL algorithms on Qwen2.5-7B, pass@1 on MATH500 rises from the base model's 34.5 to 73.5–75.6 while pass@256 for those six stays between 96.4 and 97.4, against 96.2 for the base model — the algorithms differ at pass@1 and not at the coverage ceiling (Table 3). The authors' interpretation is that RLVR redistributes probability mass toward paths the base model could already sample, and that distillation from a stronger teacher is the operation that adds new ones. Status: Interpretation supported by one multi-setting study; [[prorl]] disputes the strongest version (see ch-40).

## §8 Self-improvement for agents

Agent loops differ in three ways: rewards arrive after many steps, the environment supplies observations the model did not generate, and a failed trajectory is expensive to reproduce. Trajectory generation and verification are covered in ch-29d and rejection-sampling SFT for agents in ch-31; this section covers what the loop does over iterations.

**Task-difficulty escalation as the loop.** In DeepSeek-V3.2's general-agent pipeline, a synthesis agent builds a sandbox and a toolset, proposes a task with a Python solution function and a verification function, repairs them until the solution passes verification, and then *iteratively increases the difficulty of the task*, augmenting the toolset when the current tools cannot solve it. The generated set is then filtered by running RL with DeepSeek-V3.2 and keeping only instances with non-zero pass@100, leaving 1,827 environments and 4,417 tasks ([[deepseek-v3.1]] §3.2.3). The difficulty of the survivors is measured: on 50 sampled synthetic tasks, DeepSeek-V3.2-Exp reaches 12% pass@1 against 62% for GPT-5-Thinking (Table 5). RL on the synthetic general-agent tasks alone improved Tau2Bench, MCP-Mark and MCP-Universe, none of which were in the RL data (§4.3, Fig. 5; the report prints the values only in the figure).

**Revision trajectories without a reward.** [[agent-early-experience]] executes K alternative actions at each expert state, stores the resulting next states, and trains on them in two ways: implicit world modeling (predict the next state, then imitate) and self-reflection (train on a rationale that explains why the expert action beats the alternative, grounded in the two observed outcomes). On ALFWorld out-of-domain splits with Llama-3.1-8B, imitation scores 63.3 and implicit world modeling 78.1; on WebShop with Llama-3.2-3B, imitation 41.8 and world modeling 60.2 (Tables 1–2). As a starting point for GRPO with identical settings, WebShop success after RL is 82.0 from imitation and 92.2 from world modeling (§5.4). Two negative results in the same paper matter here: STaR-style rationales that ignore the alternatives and their outcomes *lower* performance (WebShop 47.3 → 25.0), and DPO with expert actions as chosen and the agent's own actions as rejected "collapses within tens of optimization steps" (Table 3, §6.3).

**On-policy self-improvement that did not work.** Fine-tuning Qwen2.5-Coder-32B on 868 of its own successful trajectories plus 491 teacher trajectories lowered SWE-Bench Lite resolution from 15.3% to 8.7%; the authors state "self-improvement is not yet working" and name PPO-style optimization or a stronger base as untested alternatives ([[swe-gym]] §4.2). With a constrained scaffold, two iterations of rejection-sampling fine-tuning raised the 7B model from 7.0% to 9.0% to 10.0% but the 32B model only from 19.0% to 19.7% with no further gain (§4.3, Table 4). ch-31 covers this case in its rejection-sampling context; the point for this chapter is that the iteration budget that helps a weak policy does nothing for a stronger one in the same environment.

## Negative samples and negative feedback

**1. Where negatives come from, and which of the four senses applies.** Using the definitions in the course standard — negative marginal value, negative as content, negative as conditioning, negative as gradient:

| Loop | Label source for failure | What happens to the failure | Sense |
|---|---|---|---|
| ReST-EM, STaR, R1 stage-3 SFT | answer match, unit tests, V3 judge | discarded, zero weight | negative marginal value (removed) |
| Self-Rewarding | policy-as-judge, lowest of 4 scores | rejected term of DPO | gradient |
| Meta-Rewarding | judge score tier plus length rule; meta-judge Elo for judgments | rejected response and rejected judgment | gradient |
| SPIN | none — the model's own sample is the negative by construction | negative term of the SPIN loss | gradient |
| R1-Zero, R1 RL stages | rule reward 0 | negative group-relative advantage | gradient |
| SCoRe | outcome check per attempt | correct→incorrect transition receives −α; the wrong first attempt stays in the turn-2 context | gradient and content |
| V-STaR | answer check | trains a DPO verifier on correct and incorrect solutions | gradient, in a separate model |
| Early Experience | none (reward-free) | the non-expert action and its observed next state become input or target of a next-token loss | content |
| Cognitive-behaviors priming | none | incorrect solutions with the right behaviors are trained as positive targets | content |

**2. False-negative rate.** Self-Rewarding's judge agrees with held-out human rankings on 81.7% of pairs at M3 ([[self-rewarding-lm]] Table 4), so roughly one pair in five carries an inverted label into DPO. Meta-Rewarding's judge agrees with GPT-4 on 72.34% of the self-chosen pairs where neither judge returned a tie at iteration 2, and on 61.57% of all such pairs ([[meta-rewarding-lm]] Table 3). For rule verifiers, [[star]] reports the opposite error — a correct final answer reached by incorrect reasoning becomes a false *positive*, and higher-temperature sampling raises its rate (§5). No source in this chapter reports a false-negative rate for a rule verifier.

**3. Mechanism.** For a softmax over logits z with probabilities p, the gradient of the log-probability of token y with respect to logit j is ∂ log p_y / ∂ z_j = 1[j = y] − p_j. A negative-weighted update on sample y subtracts that quantity, so it lowers z_y and raises every other logit in proportion to its current probability. Worked example: z = (2, 1, 0) gives p = (0.665, 0.245, 0.090). Push down token 3 with step size 0.5: Δz = 0.5·(p − e_3) = (+0.333, +0.122, −0.455), so z′ = (2.333, 1.122, −0.455) and p′ = (0.735, 0.219, 0.045). Token 3 lost 0.045 of probability, token 2 lost 0.026 despite its logit rising, and token 1 gained 0.070. Pushing down an already-unlikely sample moves mass to the currently most likely alternative, whatever that is. ch-43a derives the consequences (likelihood displacement, squeezing); [[likelihood-displacement]] measures one: DPO on on-policy refusal pairs lowered the training-set refusal rate of Llama-3-8B-Instruct from 74.4% to 33.4% because mass moved away from the preferred refusals as well (§6.2).

**4. Evidence for benefit and for failure.** Benefit: SCoRe's shaping term, which is a penalty on correct→incorrect transitions, is worth 1.8 points of Δ(t1, t2) (4.4% with, 2.6% without; [[self-correct-rl]] Table 4), and it is what cuts correct→incorrect from 15.8% to 1.4% (Table 2). SPIN's negative is the mechanism by which SFT data yields further gains after SFT epochs stop helping (58.14 → 63.16 against 57.23 for another SFT epoch; [[spin]] Tables 4–5). Failure: DPO on expert-versus-own-action pairs collapses within tens of steps in agent environments ([[agent-early-experience]] §6.3); GRPO's division by response length gives longer incorrect responses a smaller per-token penalty, so the policy comes to prefer longer incorrect responses, and in DeepSeek-R1-Zero the average length of incorrect responses (8,206.1) exceeds that of correct ones (4,965.4) ([[dr-grpo]] §3.1, App. F Table 5).

**5. Staleness of self-generated negatives.** SPIN's synthetic set at iterations 1–3 is 100k samples: 50k generated by the current opponent and 50k carried over from the previous iteration ([[spin]] §6.1). For a logistic-loss pair the gradient scale is σ(−m), where m is the implicit margin λ[log(π/π_ref)(y_chosen) − log(π/π_ref)(y_rejected)]. A fresh pair with m = 0 has scale σ(0) = 0.5; a carried-over negative whose likelihood already fell so that m = 3 has scale σ(−3) = 0.047, about one tenth. Two consequences: old negatives contribute little gradient per sample while still costing compute, and the pairs that do produce gradient are the ones the current policy still ranks wrongly. On the judge side the same effect appears as score saturation: Meta-Rewarding's mean judge score rises above 4.7 on a 5-point rubric after two iterations of judge training ([[meta-rewarding-lm]] Fig. 5), which compresses the score gaps that pair selection depends on.

**6. Controls.** Keep negatives on-policy by regenerating them each iteration rather than pooling. Mask negatives that are far off-policy: DeepSeek-V3.2 zeroes a sequence when its advantage is negative and its average log-ratio to the inference policy exceeds a threshold, because "highly off-policy negative samples can be detrimental" ([[deepseek-v3.1]] §3.1, Eq. 9). Bound the penalty relative to a positive term, as SCoRe does with a KL on the first attempt in Stage I. Localize the penalty to the transition that is wrong rather than the whole trajectory (SCoRe's per-attempt reward with γ = 0). Where the failure is informative but the label is uncertain, use it as content instead of gradient — the two agent results above are the measured case for that choice.

**7. Diagnostics.** Log chosen and rejected log-probabilities separately per iteration; the share of prompts that produce a usable pair (Self-Rewarding discards ties, [[self-rewarding-lm]] §2.3); the judge score distribution ([[meta-rewarding-lm]] Fig. 5); the fraction of samples with negative advantage per prompt; mean length split by correctness ([[dr-grpo]] Fig. 5); pass@1 and pass@k at large k ([[rlvr-beyond-base-model]]).

**8. Size of the effect.** None of the sources in this chapter isolates the share of the total gain that comes from the negative term in a judge-scored loop. The two ablations that exist are SCoRe's reward shaping (2.6% → 4.4% of Δ) and Early Experience's comparison of negatives-as-gradient against negatives-as-content (DPO 53.1 against implicit world modeling 58.6 on WebShop). Claims that negatives are the main driver of self-improvement gains are not supported by these sources.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Self-Rewarding M1 (Llama 2 70B) | 70B | SFT | seed data; LR; batch; dropout; loss | 3,200 IFT + 1,630 EFT examples; 5.5e−6 cosine to 1.1e−6; 16; 0.1; target tokens only | arXiv:2401.10020v3 §3.1.1, §3.1.3 | verified 2026-09-15 | §3.2.1: IFT+EFT vs IFT alone 30.5% vs 30.9% head-to-head wins |
| Self-Rewarding M2, M3 | 70B | preference | loss; β; LR; batch; pairs | DPO; 0.1; 1e−6 decaying to 1e−7; 16; 3,964 pairs (M2), 6,942 pairs (M3) | v3 §3.1.3 | verified 2026-09-15 | no ablation reported |
| Self-Rewarding, all iterations | 70B | preference | candidates per prompt; sampling; judge samples; pair rule | N = 4 at T = 0.7, p = 0.9; 3 judgments averaged; highest vs lowest, ties discarded | v3 §2.3, §3.1.3 | verified 2026-09-15 | Table 5 compares LLM-as-a-Judge prompts on the EFT test set |
| Self-Rewarding, all iterations | 70B | eval-gate | checkpoint selection | every 200 steps, selected by Claude 2 pairwise judgments on 253 validation examples | v3 §3.1.3 | verified 2026-09-15 | no ablation reported |
| Meta-Rewarding (Llama-3-8B-Instruct) | 8B | preference | loss; β; epochs; LR; batch | DPO; 0.1; 10 with per-iteration checkpoint selection; 5e−6; 32 | arXiv:2407.19594v2 App. A.3 ([[meta-rewarding-lm]]) | verified 2026-09-14 | no ablation reported |
| Meta-Rewarding, all iterations | 8B | preference | responses; judgments; prompts | K = 7 (T 0.8, top-p 0.95); N = 11; 5,000 prompts per iteration from a 20,000 pool | §2.1, §3.1 | verified 2026-09-14 | footnote 2: larger N gave similar or worse human correlation |
| Meta-Rewarding iteration 4 | 8B | preference | length-control ρ | 0.4 | §3.1, Table 4 | verified 2026-09-14 | Table 4: ρ = 0.4 gives 39.44% LC and length 2,003 vs 2,212 at ρ = 0 |
| SPIN (zephyr-7b-sft-full) | 7B | preference | prompts; synthetic size; epochs per iteration | 50k UltraChat200k prompts; 50k at iteration 0, 100k at iterations 1–3; 2 | arXiv:2401.01335v3 §6.1 | verified 2026-09-15 | §6.3 Fig. 4: more epochs inside iteration 0 do not reach iteration 1 |
| SPIN, all iterations | 7B | preference | optimizer; batch; warmup; peak LR; λ (β); max length | RMSProp, no weight decay; 64; 10%; 5e−7 (iterations 0–1), 1e−7 (iterations 2–3); 0.1, raised to 5.0 at iteration 3; 2,048 | v3 App. B.1 | verified 2026-09-15 | no ablation reported |
| SCoRe (Gemini 1.5 Flash, MATH) | not reported | RL | optimizer; LR; steps; batch; temperature; α; β1; β2 | Adam; 5e−6; 3,000; 512; 1.0; 10; 0.01; 0.1 | arXiv:2409.12917v2 App. B Table 5 | verified 2026-09-15 | Table 4: without Stage I Δ 2.2%, without shaping Δ 2.6%, full 4.4% |
| SCoRe (Gemini 1.0 Pro, MBPP) | not reported | RL | LR; steps; batch; α; β1; β2 | 1e−5; 1,500; 128; 10; 0.01; 0.25 | v2 App. B Table 5 | verified 2026-09-15 | Table 3: HumanEval Δ 12.2% vs base 3.0% |
| ReST-EM (PaLM 2-S, S*, L) | not reported | SFT | samples per problem; sampling; cap; initialization | MATH 32, APPS 64; top-K 40, T 0.7; 10 kept per problem; base pretrained model each iteration | arXiv:2312.06585v4 §5, §3 ([[rest-em]]) | verified 2026-09-14 | Fig. 7: restarting transfers better to HumanEval than continuing |
| DeepSeek-R1-Zero | 671B MoE | RL | LR; KL coef; outputs per question; questions per step; max length; steps; reference refresh | 3e−6; 0.001; 16; 32; 32,768 then 65,536 after step 8.2k; 10,400; every 400 steps | arXiv:2501.12948v2 §2.1 ([[deepseek-r1-recipe]]) | verified 2026-09-14 | §2.1: jump in performance and length at step 8.2k |
| DeepSeek-R1 Dev1 / Dev3 | 671B MoE | SFT | base; epochs; LR; context; batch | DeepSeek-V3-Base; 2–3; cosine 5e−5 → 5e−6; 32,768; 128 | v2 B.4.2 | verified 2026-09-14 | Table 3 stage comparison |
| DeepSeek-R1 Dev3 | 671B MoE | SFT | data | 804,745 samples: Math 395,285; Code 211,129; STEM 10,124; Logic 10,395; General 177,812 | v2 B.3.3, Table 5 | verified 2026-09-14 | Table 3: AlpacaEval 2.0 55.8 → 62.1, Aider 25.6 → 44.8 |
| DeepSeek-R1 final stage | 671B MoE | RL | temperature; steps; preference-reward window | 0.7; 1,700; general data and preference rewards in the last 400 steps only | v2 §3.2.2 | verified 2026-09-14 | B.5 Fig. 6: helpfulness-RM reward rises while Codeforces falls |
| Open-Reasoner-Zero-32B | 32B | RL | algorithm; GAE; KL; prompts × responses per step; sampling; policy LR; critic LR | vanilla PPO; λ = 1, γ = 1; none; 128 × 64; T 1.0, top-p 1.0; 1e−6 constant with 50-step warmup; 5e−6 | arXiv:2503.24290v2 Abstract, App. B ([[open-reasoner-zero]]) | verified 2026-09-15 | Fig. 3: λ = 1.0 stable where λ = 0.95 collapses; omitting KL best on stability and length scaling |
| Open-Reasoner-Zero-32B | 32B | RL | annealing stage | 13k prompts with fewer than 4 correct of 64 attempts in the first 1,100 steps; 100 extra steps; LR decayed to 3e−7 | v2 App. B | verified 2026-09-15 | no ablation reported |
| Cognitive-behaviors priming (Llama-3.2-3B) | 3B | RL | algorithm; steps; trajectories per prompt | PPO; 250; 4 | arXiv:2503.01307v2 §3 ([[cognitive-behaviors-self-improving-reasoners]]) | verified 2026-09-15 | §3: PPO chosen for stability; performance anecdotally similar across algorithms |

**Starting point for a small general-purpose run.** For a judge-scored loop on a 7–8B instruction model, the only fully disclosed configuration above is Meta-Rewarding's: DPO with β = 0.1, learning rate 5e−6, batch 32, 5,000 prompts per iteration, K = 7 responses, N = 11 judgments, length control ρ between 0.32 and 0.4, and per-iteration checkpoint selection, run for at most four iterations (Llama-3-8B-Instruct, hardware not reported). For a verifier-scored loop on a fixed problem set of a few thousand problems, ReST-EM's disclosed configuration is 32 samples per problem at top-K 40 and T 0.7, at most 10 kept per problem, fine-tuning the base model each iteration, evaluated on held out data after every iteration (PaLM 2 models; learning rate and batch size are not reported, so they must come from your own SFT settings in ch-30).

## Generalization lens

**(a) What increases breadth.** Restarting each iteration from the base model rather than the previous iterate: ReST-EM links this choice to transfer, with better HumanEval transfer than continuing from the last iterate ([[rest-em]] §3, Fig. 7). Adding non-target data after the reasoning loop: R1's Dev3 stage moves Aider-Polyglot 25.6 → 44.8 and AlpacaEval 2.0 55.8 → 62.1 ([[deepseek-r1]] Table 3). Mixing reward types in the last stage: AlpacaEval 2.0 62.1 → 87.6 and ArenaHard 75.6 → 92.3 (Table 3). Scaling and diversifying the prompt set: ORZ's 57k mixed set keeps improving where MATH-7.5k plateaus ([[open-reasoner-zero]] §3.2), and reasoning-only RL raised MMLU-Pro 55.1 → 74.4 (Table 2). Grounding self-generated rationales in observed outcomes rather than in the model's own guess: Early Experience's out-of-domain gains meet or exceed its in-domain gains on ALFWorld and SearchQA ([[agent-early-experience]] §5.3).

**(b) What narrows.** Optimizing a metric whose scorer moves with the policy: judge-human agreement falls after iteration 2 while the judge-scored win rate rises ([[meta-rewarding-lm]] Tables 1, 7). Held-out benchmarks decline inside judge loops that never scored them ([[self-rewarding-lm]] Table 3). Coverage falls with RL steps: pass@256 68.3 → 63.9 below the base model's 69.1 ([[rlvr-beyond-base-model]] Table 4). A fixed human target distribution caps the loop ([[spin]] §7). Response length inflates without a length control (1,092 → 2,552 in Self-Rewarding §3.2.1; GRPO's length bias in [[dr-grpo]] §3.1). Recursive training on self-generated data loses low-probability content — the mechanism measured in [[model-collapse]] Theorem 3.1 and Fig. 1, and the reason ch-23 recommends keeping original data in the mixture. Preference rewards over-optimize: reward rises while Codeforces falls ([[deepseek-r1]] B.5).

**(c) How to measure it for this stage.** After every iteration: a held-out capability panel spanning domains the loop does not score; agreement between the loop's judge and a fixed labeled set; pass@k at large k on held-out prompts; length and, where the stack exposes it, entropy; and, for agent loops, an environment split the loop never trained on ([[deepseek-v3.1]] §4.1 treats MCP-Universe, MCP-Mark and Tool-Decathlon this way). Known measurement errors: AlpacaEval and MT-Bench are scored by a language model that may share biases with the judge being trained ([[self-rewarding-lm]] §6); benchmark contamination is not detectable by 10-gram filters against paraphrases ([[deepseek-r1]] D.1); and checkpoint selection by training reward ([[self-correct-rl]] §6) can pick an iteration that a held-out panel would reject.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Stopping on the loop's own metric | The loop metric rises monotonically while nothing else is measured | Run the held-out panel and the judge-agreement check at every iteration; compare the best-panel iteration with the best-metric iteration |
| Treating "three iterations" as a law | The run stops at 3 regardless of the panel | Iteration counts in the sources are what the authors ran ([[self-rewarding-lm]] §6 calls the scaling of the effect open) |
| Continuing from the previous iterate when transfer matters | In-domain metric rises, out-of-domain transfer falls | Compare a restart-from-base arm against a continue arm on a held-out task ([[rest-em]] Fig. 7) |
| Reusing negatives from earlier iterations | Loss falls but the gradient is carried by a shrinking share of pairs | Log the distribution of the implicit margin; count pairs with σ(−m) below a threshold |
| Pair selection on a saturated judge | Mean judge score approaches the rubric maximum; many discarded ties | Plot the judge score histogram per iteration ([[meta-rewarding-lm]] Fig. 5) |
| No length control in a judge loop | Response length grows iteration over iteration | Report mean length with every win rate; apply the ρ rule ([[meta-rewarding-lm]] §2.1) |
| Reporting pass@1 only after RLVR | pass@1 up, coverage unknown | pass@k at k = 128 or 256 against the base model ([[rlvr-beyond-base-model]] Table 4) |
| Shipping the reasoning loop as the product | Instruction following, open-ended writing and safety scores lag | Compare against the staged pipeline's per-stage table ([[deepseek-r1]] Table 3) |
| Attributing emergent reflection to RL alone | "The model learned to say 'wait'" with no base-model measurement | Count reflection keywords in base-model samples ([[dr-grpo]] §2.3) and check whether reflective responses are more accurate (App. F) |
| Running an agent self-improvement loop on a policy that rarely succeeds | Resolution rate falls after the first on-policy round | Measure the success rate before the loop; compare with a teacher-trajectory arm ([[swe-gym]] §4.2) |

## Check your understanding

1. Self-Rewarding's judge improves against held-out human rankings (65.1% → 81.7% pairwise accuracy) while three of four held-out benchmarks fall between M1 and M3. Explain how both can be true of the same training runs, and say which measurement you would use to decide whether to run a fourth iteration.
2. ReST-EM fine-tunes the base model at every iteration and SPIN continues from the previous iterate. Derive the consequence of each choice for the negatives the loop sees and for what the loop can forget.
3. SPIN's fixed point is the SFT data distribution. Using Theorem 5.4, explain why a smaller λ moves the policy further in one step, and why that does not change the fixed point.
4. In SCoRe, why does the reward bonus α(r̂(y2) − r̂(y1)) not teach the model to answer incorrectly on the first attempt on purpose? Your answer must use the discount factor the paper uses.
5. A Bradley-Terry reward model cannot represent the three-way preference in §4. State the property of the model that makes this impossible, and explain what the Nash solution gives up in exchange for representing it.
6. DeepSeek-R1 ends below R1-Zero on GPQA Diamond and CNMO 2024 but above it by 36.7 and 62.9 points on IF-Eval and AlpacaEval 2.0. Explain which stages produced each movement, and what that implies for a team that wants one general model rather than two specialists.
7. Two models are trained with the same RLVR recipe; one shows the "aha" language and the other does not. Using the base-model evidence in §7.3, list the measurements you would make before concluding that the recipe caused the difference.
8. An agent self-improvement loop raises a 7B policy from 7.0% to 10.0% resolution over two iterations but leaves a 32B policy at 19.7%. Give two mechanisms consistent with this, and an experiment that separates them.

## Connections

- Previous chapter: **ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing**. The final R1 stage in §7.2 is a multi-domain RL stage; the domain-mixing questions are answered there.
- Next chapter: **ch-45a — Preference-Optimization and RL Stage Recipes Side by Side**. The per-loop recipe rows here are compared against the full-pipeline recipes there.
- **ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation**: the mechanics of one generate-filter-train round, including the SWE-Gym and ReST-EM cases from the SFT side.
- **ch-23 — Model Collapse and Verification of Synthetic Data**: what repeated training on self-generated data does to the tails of the distribution.
- **ch-29d — User Simulators, Trajectory Verification, and Failed Trajectories**: how agent trajectories in §8 are generated and verified.
- **ch-39 — Offline Preference Optimization: DPO and Its Variants**: the loss that §2 and §3 iterate.
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO**: the estimator used by R1-Zero and the length and difficulty biases quoted in §7.
- **ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization**: the scalar-score assumption §4 replaces.
- **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages**: the full derivation behind the softmax gradient argument in the negatives section.
- **ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability**: the RL machinery that §8's loops feed.
- **ch-46 — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention**: runs the stopping-rule measurement of §6 on a small model.
- **ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting**: judge drift as a subject in its own right.

## Sources

- [[self-rewarding-lm]] — the policy-as-judge loop, its three models, and Tables 1–4. Numbers in this chapter were read from arXiv:2401.10020v3; the library card has not been re-verified and its Spearman values (0.62 → 0.71) and its iteration-4 regression claim are not in the paper.
- [[meta-rewarding-lm]] — meta-judge construction, length-controlled pair selection, judge-drift measurements (verified card, 2026-09-14).
- [[spin]] — the self-play loss, Theorems 5.2 and 5.4, and per-iteration leaderboard results. Numbers read from arXiv:2401.01335v3; the library card's MT-Bench trajectory (6.39 → 7.12) is not in the paper, and its "3 epochs, lr 5e-7" summary does not match App. B.1, which gives 2 epochs per iteration and 5e-7 for iterations 0–1 against 1e-7 for iterations 2–3 (the card's batch size of 64 is correct).
- [[self-play-preference]] — Nash equilibrium formulation, Nash-MD update, the non-transitive population example, and Table 1. Numbers read from arXiv:2312.00886v4; the library card carries no verification section and states the Nash objective and Figure 2 in a form the paper does not, so nothing in this chapter is taken from it.
- [[self-correct-rl]] — two-stage self-correction RL, the shaping bonus with α = 10, MATH and HumanEval results, ablations. Numbers read from arXiv:2409.12917v2; the card's α = 2.0 and "MBPP +9.1" are not in the paper.
- [[rest-em]] — EM view of verifier-filtered self-training, base-model restart, iteration and transfer results (verified card).
- [[star]] — rationale bootstrapping, the indicator-reward policy-gradient view, and the false-positive risk of rationalization (verified card).
- [[v-star]] — training a verifier on the failures that ReST-EM discards.
- [[deepseek-r1]], [[deepseek-r1-recipe]] — R1-Zero settings, the four-stage pipeline, Table 3 per-stage results, the PRM and base-capacity findings (verified cards).
- [[open-reasoner-zero]] — chapter excerpt of arXiv:2503.24290v2: vanilla PPO without KL, Table 1 comparison with R1-Zero-Qwen-32B, MMLU and MMLU-Pro generalization, annealing on self-mined hard prompts.
- [[cognitive-behaviors-self-improving-reasoners]] — chapter excerpt of arXiv:2503.01307v2: which base-model behaviors decide whether RL improves, and the incorrect-solution priming result.
- [[dr-grpo]] — GRPO's length and difficulty biases, self-reflection in base models, and the R1-Zero response-length analysis (verified card).
- [[rlvr-beyond-base-model]] — pass@k coverage against training steps and across RL algorithms; Tables 3 and 4 read from arXiv:2504.13837v5.
- [[prorl]] — the counter-result to §7.4's sharpening interpretation; cited only as the dispute, with the evidence weighed in ch-40.
- [[deepseek-v3.1]] — the V3.2 report: general-agent task synthesis with difficulty escalation, pass@100 filtering, and off-policy negative masking (verified card).
- [[agent-early-experience]] — reward-free agent self-improvement, out-of-domain results, and the DPO-collapse comparison (verified card).
- [[swe-gym]] — on-policy self-improvement results for coding agents; numbers read from arXiv:2412.21139v2 §4.2–4.3.
- [[model-collapse]] — loss of low-probability content under recursive training on model outputs (verified card).
- [[likelihood-displacement]] — measured cost of the negative gradient in DPO (verified card).
