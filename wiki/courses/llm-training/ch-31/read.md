<!-- chapter: ch-31
     track: sft
     kind: content
     title: Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation
     deps: [ch-30c]
     sources: [[llama-2]], [[llama-2-recipe]], [[llama-3]], [[llama-3-recipe]], [[rejection-sampling-finetuning]], [[rft-scaling-relationship-math]], [[rest-em]], [[star]], [[v-star]], [[raft-reinforce-rej-minimalist]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[glm-4-5]], [[glm-5]], [[swe-gym]], [[kimi-dev]], [[rlvr-beyond-base-model]], [[prorl]]
     figures: figures/iterative-loop.html
     revised: 2026-09 (generality revision)
-->

# Chapter 31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation

> **Core insight.** For the first gradient step, fine-tuning a model on its own outputs that pass a correctness check is a policy-gradient update with a {0,1} reward and no baseline: accepted samples gain probability and rejected samples get zero weight ([[star]] Eq. 2; [[rest-em]] Algorithm 1). It improved held-out accuracy when the accepted set contained many distinct solutions (LLaMA-7B on GSM8K: 35.9 after SFT, 41.7 with 100 samples per question, 49.3 with samples pooled from four models; [[rft-scaling-relationship-math]] Tables 1, 3), and it transferred better to HumanEval when each round restarted from the base model ([[rest-em]] Fig. 7). It regressed or stopped helping on a small problem set (ReST-EM APPS at iteration 2), when each round used only the previous round's samples (Llama 2 RLHF-V3), when the sampler produced few distinct solutions (LLaMA-33B: no gain), and for a 32B coding agent fine-tuned on its own successful trajectories plus teacher trajectories (SWE-bench Lite 15.3% → 8.7%; [[swe-gym]] §4.2). Training only on positives lowered policy entropy faster than GRPO (ch-40) and plateaued earlier ([[raft-reinforce-rej-minimalist]] §5.1).
>
> **Guideline.** When a task has an automatic correctness check and a fixed problem set, run rejection-sampling rounds that restart from the base model or pool accepted samples across all earlier rounds, cap accepted samples per problem, and evaluate held-out tasks after every round, because restarting gave better transfer than continuing ([[rest-em]] Fig. 7), pooling was the reported fix for the Llama 2 RLHF-V3 regression ([[llama-2]] §3.2.3), and ReST-EM's APPS and HumanEval regression at iteration 2 occurred despite restarts and a cap and was visible only in held-out evaluation ([[rest-em]] §5.1, Fig. 4). When distinct solutions per problem stop growing or policy entropy falls round over round, move to an objective that also uses failed samples (a group-baseline RL update or a verifier trained on failures), because RAFT++ plateaued at low entropy while GRPO kept improving ([[raft-reinforce-rej-minimalist]] Figs. 2–3; final averages 56.1 vs 56.3 on Qwen2.5-Math-7B and 27.6 vs 28.4 on LLaMA-3.2-3B-instruct, Table 1). When RL-only training from a base model produces unreadable or language-mixed outputs, add a small cold-start SFT set before RL and evaluate every stage checkpoint on the full capability panel, because in DeepSeek-R1 the cold-start SFT model scored 71.7 on IF-Eval against 46.6 for the RL-only model from the same base but 59.0 against 77.9 on AIME 2024, and the finished pipeline ended above the RL-only model on AIME 2024 (79.8) but below it on GPQA Diamond (71.5 vs 75.8) and CNMO 2024 (78.8 vs 88.1) ([[deepseek-r1]] Table 3). For long-horizon agents whose starting policy resolves few tasks, use strong-teacher trajectories rather than on-policy rejection-sampling SFT, because [[swe-gym]] measured +12.3 points on SWE-bench Lite from 491 teacher trajectories and a drop from 15.3% to 8.7% after adding on-policy trajectories; the authors name PPO-style optimization as an untested alternative (§4.2).

## Why this chapter matters for a general-purpose model

Human-written targets are expensive, and a dataset such as GSM8K provides one reasoning path per question, which the RFT authors identify as a limit on learning alternative solution orders ([[rft-scaling-relationship-math]] §3.3). Llama 2 stopped collecting SFT annotations at 27,540 examples and moved annotation effort to preference comparisons ([[llama-2]] §3.1). ReST-EM states that fine-tuning on human data is limited by the quantity and diversity of that data ([[rest-em]] Abstract). Self-generated data replaces human targets with model samples that a scorer accepts. A scorer is any function that labels a sample: an answer matcher, unit tests, a reward model (RM), or an LLM judge.

This chapter sits between SFT (ch-30) and RL (ch-37 onward), but self-generated data appears at several points in current pipelines: as rounds of SFT inside iterative RLHF ([[llama-2]], [[llama-3]]), as a cold-start set before RL ([[deepseek-r1]], [[glm-4-5]]), as a rejection-sampling SFT stage after RL ([[deepseek-r1]] stage 3), as distillation from RL experts back into one model ([[glm-4-5]], [[glm-5]]), and as agent trajectories ([[swe-gym]], [[kimi-dev]]).

The core question is when training on a model's own filtered samples improves generalization and when it overfits. The chapter answers three sub-questions from the course standard. Breadth increases when accepted samples are diverse, when rounds restart from or pool toward earlier distributions, and when non-target data is mixed in. Narrowing appears as train–test gaps on small problem sets, entropy collapse, forgetting of skills that the scorer does not measure, and format overfitting. Generality is measured per round with held-out suites, pass@k at large k, distinct solutions per problem, and a panel of non-target capabilities.

## §1 Rejection-sampling fine-tuning: the loop, its variants, and round structure

### 1.1 Definition and mechanism

Rejection-sampling fine-tuning (RSFT in this chapter) is supervised fine-tuning on outputs sampled from a model and kept only if a scorer accepts them. The same procedure is called "Rejection Sampling fine-tuning" in [[llama-2]] §3.2.3, RFT in [[rft-scaling-relationship-math]], RAFT (reward-ranked fine-tuning) in [[raft-reinforce-rej-minimalist]] §3, and the Generate/Improve steps of ReST-EM in [[rest-em]] §3. The library card [[rejection-sampling-finetuning]] defines the pattern without numbers.

1. Choose a prompt set with a scorer r(x, y), where x is the prompt and y a sampled output.
2. Sample K outputs per prompt from a sampler: the current model, earlier checkpoints, or several models.
3. Score every output. Keep outputs with r = 1, or the highest-scoring output per prompt.
4. Optionally deduplicate kept outputs and cap the number kept per prompt.
5. Fine-tune a starting checkpoint on the kept set with next-token loss on output tokens only.
6. Evaluate on held-out tasks. Use the new model as the sampler for the next round.

The sources differ in four choices: the sampler, the selection rule, the checkpoint that the next round fine-tunes, and whether accepted samples are pooled across rounds.

| Method | Sampler and samples | Selection | Fine-tunes from | Pooling across rounds | Locus |
|---|---|---|---|---|---|
| STaR (GPT-J 6B) | current model, greedy, 1 per problem, plus a rationalization pass | correct final answer | original pretrained model each iteration | no | [[star]] §3.1, Alg. 1 |
| RFT (LLaMA 7B–33B) | SFT model, k = 100, temperature 0.7 | correct answer and calculations; one path per distinct equation list | pretrained model, on human data plus accepted paths | one round; pooled across models in U13B / U33B | [[rft-scaling-relationship-math]] §3.3, App. A.3 |
| ReST-EM (PaLM 2) | previous iterate; 32 (MATH) or 64 (APPS); top-K 40, temperature 0.7 | binary reward; at most 10 per problem | base pretrained model each iteration | no | [[rest-em]] §3, §5 |
| Llama 2-Chat RLHF-V1…V5 | latest 70B model; K not printed | highest RM reward per prompt | not stated unambiguously (a sentence on the Fig. 8 temperature study says "always starting from the base model on each new RLHF version") | up to V3: previous iteration only; later: top samples of all prior iterations | [[llama-2]] §3.2.3 |
| Llama 3.1 | best checkpoint of previous round; K typically 10–30 | RM picks best | pre-trained model each round, mixed with synthetic and human data | not stated; each round collects new SFT data and adjusts the mix, and the final mix epochs some sources several times | [[llama-3]] §4.1, §4.1.6, §4.2.2 |
| V-STaR (LLaMA 2 7B/13B) | previous generator; k = 16 per iteration | correct answer or passing tests | pretrained base each iteration | yes: all correct samples accumulate | [[v-star]] Alg. 1 |
| DeepSeek-R1 stage 3 | reasoning-RL checkpoint (Dev2) | correct only; mixed-language, long-paragraph, code-block CoT removed | DeepSeek-V3-Base | one round | [[deepseek-r1]] B.3.3, Fig. 2 |

### 1.2 Worked example: which problems produce training data

pass@K is the probability that at least one of K independent samples is correct. If a problem has per-sample success probability p, then pass@K = 1 − (1 − p)^K. With K = 32 (ReST-EM's MATH setting): p = 0.1 gives 1 − 0.9^32 = 1 − 0.034 = 0.966; p = 0.01 gives 1 − 0.99^32 = 1 − 0.725 = 0.275. A problem with p = 0.01 contributes no data in 72.5% of rounds. The expected number of accepted samples is K·p: 3.2 for a hard problem with p = 0.1 and 25.6 for an easy problem with p = 0.8. Without a cap, the easy problem supplies 25.6 / 3.2 = 8 times as many targets as the hard one. With ReST-EM's cap of 10 per problem, the easy problem keeps about 10 (the probability of fewer than 10 successes in 32 draws at p = 0.8 is below 10⁻⁶) and the hard problem about 3.2, so the ratio falls to about 10 / 3.2 ≈ 3.1. ReST-EM states the cap exists to limit over-representation of easy problems ([[rest-em]] §5), and SWE-Gym found a per-instance cap of 2 slightly better than using all successful trajectories ([[swe-gym]] §4.3, Table 6). The accepted set is therefore biased toward problems the model already solves: without a cap, the expected ratio of accepted samples between two problems equals the ratio of their success probabilities at every K. The companion figure's acceptance panel ([figures/iterative-loop.html#acceptance](figures/iterative-loop.html#acceptance)) lets the reader set p, K, and the cap and read the expected accepted count and pass@K.

### 1.3 Llama 2: rejection sampling inside iterative RLHF

Llama 2 trained five RLHF versions as new preference batches arrived and used two algorithms ([[llama-2]] §3.2.3). The second algorithm is PPO (proximal policy optimization, ch-38). The report describes the schedule in one sentence: "Until RLHF (V4), we used only Rejection Sampling fine-tuning, and after that, we combined the two sequentially, applying PPO on top of the resulted Rejection Sampling checkpoint before sampling again." The sentence does not say whether V4 itself used PPO ([[llama-2-recipe]]). The report contrasts the two by breadth (K samples per prompt for rejection sampling, one for PPO) and depth (PPO samples from the policy updated at the previous step; rejection sampling samples all outputs from the initial policy of the round), and states that iterative updates make the difference "less pronounced" (§3.2.3).

Three details matter for generality. First, rejection sampling ran only with the 70B model; all smaller chat models were fine-tuned on 70B samples, which the authors call distillation and do not analyze (§3.2.3). Second, the optimal sampling temperature changed across versions; for the RLHF model it was T ∈ [1.2, 1.3] when drawing 10–100 samples (Fig. 8). Third, the gain from selection is the gap between the maximum and median reward among N samples, which grows with N while the median stays flat (Fig. 7). The PPO settings are a KL-penalty coefficient β = 0.01 for 7B and 13B and β = 0.005 for 34B and 70B (β multiplies the KL divergence from the original policy, subtracted from the reward), with a constant learning rate of 10⁻⁶, batch 512, mini-batch 64, clip 0.2, and 200–400 iterations with early stopping on held-out prompts (§3.2.3). Human evaluation used about 4,000 prompts with no coding or reasoning prompts and rated only the final turn of each conversation (§3.4.2), so the human evaluation did not measure coding or reasoning.

### 1.4 RFT: the number of distinct solutions drives held-out gains

Yuan et al. fine-tuned LLaMA and LLaMA 2 models on GSM8K plus self-sampled correct reasoning paths and kept one path per distinct list of equations ([[rft-scaling-relationship-math]] §3.3, App. A.3). Accuracy is greedy-decoding accuracy (maj1@1) on the GSM8K test split, which is held out but drawn from the training distribution.

| Base model (Table 1) | SFT | RFT, k = 100 | Correct paths per question | Distinct paths per question |
|---|---|---|---|---|
| LLaMA-7B | 35.9 | 41.7 | 53.3 | 5.25 |
| LLaMA 2-7B | 41.6 | 47.5 | 60.8 | 5.19 |
| LLaMA-13B | 43.0 | 49.1 | 62.5 | 5.26 |
| LLaMA-33B | 54.6 | 54.5 | 88.7 | 2.78 |

The 33B SFT model produced the most correct paths per question but the fewest distinct ones, and RFT did not improve it; the authors attribute this to the 33B SFT model fitting the training questions (§3.3). Sampling the 33B model at temperature 1.0 raised distinct paths only to 4.77 (§3.3). Distinct paths grow slowly with k: for LLaMA-7B they are 1.17 at k = 1, 2.20 at k = 12, and 5.25 at k = 100 (Table 2), and accuracy gains per doubling of k shrink (Fig. 4). Pooling paths from four models (U13B) gives 12.84 distinct paths per question and raises LLaMA-7B to 49.3, LLaMA 2-7B to 50.3, LLaMA-13B to 52.1, and LLaMA 2-13B to 55.4 (Tables 2–3). With k = 100, training without deduplication gave similar accuracy to deduplicated training, and deduplication was better for 3 of 4 models (§3.3, Table 5). Status: Result (single study). The measurement is in-distribution; the paper does not report out-of-domain tasks.

### 1.5 ReST-EM: expectation–maximization view, restart from base, and iterations

ReST-EM derives the loop as expectation–maximization (EM) for RL with a binary optimality variable O, where p(O = 1 | x, y) ∝ f(r(x, y)) and f is a non-decreasing, non-negative function of the reward r ([[rest-em]] §3). The E-step (Generate) samples from the current policy and scores the samples. The M-step (Improve) maximizes

J(θ) = E_{(x,y)∼D_i} [ r(x, y) · log p_θ(y | x) ]

where θ are the model parameters, D_i is the set of samples generated in iteration i, r(x, y) ∈ {0, 1} is the binary reward, and p_θ(y | x) is the model's probability of output y given input x (Algorithm 1). Each Improve step fine-tunes the base pretrained model, not the previous iterate, "to mitigate task-specific over-fitting" (§3). Compared with continuing from the previous iterate, this gave comparable task performance and better transfer to held-out tasks (PaLM 2-S*, APPS → HumanEval; Fig. 7). Status: Result (single study).

On MATH (7,500 training problems), PaLM 2-L improved over three iterations, with small test gains after the first iteration while training accuracy kept rising (§5.1, Fig. 4). One iteration with three times as many samples per problem reached 40.3% pass@1, below 41.0% at iteration 2 and 41.9% at iteration 3 (§5.3). At equal generation count (iteration 3 against the single 3× iteration), more rounds with fewer samples per round scored higher in this single run. Transfer: the MATH-trained model improved GSM8K over iterations (Fig. 2); on the 2023 Hungarian high-school finals exam it scored above every compared model except GPT-4, with the other models' scores taken from Paster (2023) and no base PaLM 2-L score shown (§5.4, Fig. 10); MATH- and APPS-trained PaLM 2-L showed "no major degradation" on Big-Bench Hard (§5.4, Fig. 9). ReST-EM was stronger than the base model at every K in pass@K, with the gap "typically" largest at K = 1, and the authors state that it may not close the gap to pass@K at large K (§5.2, §6).

## §2 RSFT, RAFT, and ReST-EM as REINFORCE with a {0,1} reward and zero baseline

REINFORCE is the policy-gradient estimator that multiplies the gradient of a sample's log-probability by that sample's reward minus a baseline (ch-37 derives it). For M prompts x_i with n samples y_ij each:

ĝ = (1 / (M·n)) · Σ_i Σ_j (r_ij − b_i) · ∇_θ log π_θ(y_ij | x_i)

where π_θ is the policy being trained, r_ij is the reward of sample j for prompt i, b_i is a baseline that does not depend on y_ij, and ĝ estimates the gradient of expected reward. The RSFT loss on the accepted set A = {(i, j) : r_ij = 1} is

L_RSFT(θ) = −(1 / |A|) · Σ_{(i,j)∈A} log π_θ(y_ij | x_i)

where |A| is the number of accepted samples. Setting r_ij ∈ {0, 1} and b_i = 0 in ĝ removes every rejected sample from the sum, so

−∇_θ L_RSFT = (M·n / |A|) · ĝ = (1 / a) · ĝ,   with acceptance rate a = |A| / (M·n).

An RSFT gradient step is a REINFORCE step with no baseline and a learning rate scaled by 1/a. STaR states the same identity with an indicator reward, ∇J = Σ_i E[1(ŷ_i = y_i) · ∇ log p_M(ŷ_i, r̂_i | x_i)], and calls the indicator the filtering step ([[star]] §3.1, Eq. 2). ReST-EM's M-step objective is the same expression with r ∈ {0, 1} ([[rest-em]] Algorithm 1, §3 Remark). Selecting the top-1 sample by RM score, as in Llama 2, is the same form with r_ij = 1 only for the highest-scoring sample of each prompt.

**Conditions.** The identity holds for the first gradient step, when the parameters equal the sampling policy. RSFT then takes many steps and epochs on the fixed set, so later steps use off-policy samples. RAFT++ adds the per-token importance ratio s_t(θ) = π_θ(a_t | x, a_<t) / π_θold(a_t | x, a_<t) and PPO clipping to the RAFT loss, keeping only the highest-reward responses ([[raft-reinforce-rej-minimalist]] Eq. 6). An intermediate variant with the ratio but without clipping underperformed vanilla RAFT (§5, Fig. 2).

**Worked example.** Three prompts with n = 4 samples each have rewards A = [1, 0, 1, 0], B = [1, 1, 1, 1], C = [0, 0, 0, 0]. Six of twelve samples are accepted, so a = 0.5.

| Update rule | Weight per sample of A | B | C | Share of positive weight from B |
|---|---|---|---|---|
| RSFT, mean over accepted | +1/6, 0, +1/6, 0 | +1/6 each | 0 | 4/6 = 67% |
| REINFORCE, r ∈ {0,1}, b = 0 | +1/12, 0, +1/12, 0 | +1/12 each | 0 | 67% |
| REINFORCE, r ∈ {−1,+1}, b = 0 | ±1/12 | +1/12 each | −1/12 each | 67% |
| Group-mean baseline, b_i = mean(r_i) | +0.5/12, −0.5/12, +0.5/12, −0.5/12 | 0 | 0 | 0% (B dropped) |

The first two rows differ only by the factor 1/a = 2. In both, the all-correct prompt B supplies two thirds of the positive weight, which is the easy-problem bias from §1.2 expressed as gradient share. With a ±1 reward and no baseline, the all-wrong prompt C lowers the probability of all four of its samples. The group-mean baseline, which GRPO (group relative policy optimization) applies before dividing by the group's standard deviation, gives zero weight to both all-correct and all-wrong prompts. [[raft-reinforce-rej-minimalist]] reports that GRPO's advantage over vanilla REINFORCE comes mainly from this implicit removal of all-wrong prompts, not from reward normalization, and that removing both kinds of prompt ("Reinforce-Rej") matched GRPO with better KL efficiency, meaning more reward gain per unit of KL divergence from the initial policy (LLaMA-3.2-3B-instruct; §5.1, Fig. 4). The weight panel of the companion figure ([figures/iterative-loop.html#weights](figures/iterative-loop.html#weights)) computes these weights for any reward pattern and shows how a per-prompt cap changes the share.

**Evidence.** On Qwen2.5-Math-7B-base trained on Numina-Math prompts (average@16 accuracy at temperature 1.0 over MATH500, Minerva Math, and OlympiadBench), RAFT reached 52.3, RAFT++ 56.1, GRPO 56.3, PPO 52.5, iterative DPO 48.8 (direct preference optimization repeated on newly sampled pairs, ch-39), and Reinforce-Rej 56.4; on LLaMA-3.2-3B-instruct, RAFT 25.9, RAFT++ 27.6, GRPO 28.4, and Reinforce 24.2 ([[raft-reinforce-rej-minimalist]] Table 1). The authors tuned batch size, mini-batch size, and learning rate per algorithm (Table 1 caption). Status: Result (single study; math only; about 250–300 training steps).

**Implication.** Positive-only self-training has no term that targets a specific wrong answer. A wrong answer the model currently prefers loses probability only indirectly, through softmax normalization when accepted samples gain probability, and a prompt with no accepted sample produces no update. Every accepted sample from an easy prompt competes for gradient share with samples from hard prompts (derived from the identity above).

## §3 STaR rationalization and V-STaR: reusing failed problems and failed samples

### 3.1 STaR: rationalization reuses the failed problem, not the failed rationale

STaR (Self-Taught Reasoner) generates a rationale and answer for every training problem with a few-shot prompt, keeps rationales with correct answers, and fine-tunes the original pretrained model on them each outer loop ([[star]] §3.1, Alg. 1). Rationalization addresses a measurable gap: without it, the loop "fails to solve any new problems in the training set because it receives no direct training signal for problems it fails to solve" (§1). For problems the model failed, STaR inserts the gold answer as a hint, samples a new rationale, keeps it only if it reaches the gold answer, and trains on it with the hint removed (§3.2). The failed rationale still receives zero gradient; no term in Algorithm 1 explicitly lowers the likelihood of a sample.

On CommonsenseQA dev accuracy with GPT-J 6B, direct answer fine-tuning scored 60.0, STaR without rationalization 68.8, STaR with rationalization 72.5, and a fine-tuned GPT-3 that is 30× larger 73.0 (Table 1). On GSM8K test, rationalization moved accuracy from 10.1 to 10.7, which the authors describe as not a substantial improvement (Table 2, §4.5). STaR decodes greedily; sampling at temperature 0.5 or 0.7 gave worse models, and higher temperature raised the rate of correct answers reached with incorrect reasoning, which the authors state prevents generalization (§5). Few-shot accuracy must be above chance: GPT-2 did not bootstrap on arithmetic (§6). ReST-EM reports that STaR-style rationalization increased such false positives in its preliminary experiments and did not use it ([[rest-em]] §4). Status: Result (single study) for the gains. More correct-answer, wrong-reasoning samples are reported by two sources under different conditions: higher sampling temperature in [[star]] §5 and rationalization in [[rest-em]] §4 (qualitative; no rates reported).

### 3.2 V-STaR: failed samples train a verifier

V-STaR (Hosseini et al., COLM 2024) keeps the STaR generator loop and adds a verifier trained on both correct and incorrect self-generated solutions ([[v-star]] §3). In each of T iterations, the generator is fine-tuned from the pretrained base on D_GEN, the original SFT data plus all correct samples so far; k samples per training problem are labeled by answer match or test execution; correct samples join D_GEN and all samples join D_VER (Alg. 1). The verifier is trained with DPO from the SFT model on pairs from the Cartesian product of correct and incorrect solutions for each problem:

L_DPO(V; G_SFT) = −E_{(x, y⁺, y⁻)∼D_VER} log σ( r̂(x, y⁺) − r̂(x, y⁻) ),   r̂(x, y) = β · log( V(y | x) / G_SFT(y | x) )

where V is the verifier, G_SFT the SFT generator used as reference, y⁺ a correct and y⁻ an incorrect solution, σ the logistic function, and β the proximity coefficient (§3.1, Eq. 2). At test time the verifier's likelihood V(ŷ | x) ranks candidate solutions (§3.1).

With LLaMA 2 and CodeLLaMA 7B/13B trained with LoRA (low-rank adapter fine-tuning), 3 iterations of k = 16, V-STaR improved test accuracy over prior self-improvement and verification methods by 6–17 points on math and 4–12 points on code (§1, §4.3). Transfer was evaluated on a 150-problem level-1 subset of MATH and on HumanEval, where V-STaR also led the baselines (§4.3, footnote 2). A fourth MBPP iteration added 0.3 points (§4.3). Placing the verifier inside the training loop to choose generator data did "not provide a substantial gain" on MBPP in the authors' words (Best-of-64, the accuracy of the verifier's top-ranked candidate among 64 samples, 53.2; Pass@1 46.34; §4.7). The DPO verifier's own generation accuracy degraded after a few training updates while its ranking accuracy rose (§4.6, Fig. 6). Status: Result (single study; small-model LoRA; math and code only).

## §4 Cases where self-training overfit or narrowed

| Case | Setting | Measured symptom | Reported cause | Reported fix | Locus |
|---|---|---|---|---|---|
| ReST-EM on APPS | PaLM 2, 2,342 problems | most gain in iteration 1; iteration 2 regressed on APPS and HumanEval; training accuracy kept rising | overfitting on a small problem set (APPS is about one third of MATH's size); occurred with restart from base and a cap of 10 | none reported; most APPS gain came from iteration 1 | [[rest-em]] §5.1, Fig. 4 |
| Llama 2 RLHF-V3 | 70B rejection sampling from V2 samples only | weaker rhyming in poems (qualitative) | forgetting | pool top samples from all prior iterations; "considerable enhancements", no figures | [[llama-2]] §3.2.3 |
| RFT on LLaMA-33B | GSM8K, k = 100 | SFT 54.6 → RFT 54.5; 2.78 distinct paths per question | SFT model fit training questions; low sample diversity | pool paths sampled by 7B/13B models (U13B: 56.5; U33B, which adds 33B paths: 57.9) | [[rft-scaling-relationship-math]] §3.3, Table 5 |
| RAFT++ | Qwen2.5-Math-7B, LLaMA-3.2-3B | faster early gains, turning point near iteration 100, surpassed by GRPO; faster drop in policy entropy (entropy of the model's next-token distribution on its own samples) | reduced exploration from positive-only updates (authors' interpretation) | clip-higher, an asymmetric importance-ratio clip range [1 − 0.2, 1 + 0.28], stabilized entropy on LLaMA | [[raft-reinforce-rej-minimalist]] §5, Figs. 2–3 |
| SWE-Gym on-policy RSFT | Qwen2.5-Coder-32B, OpenHands | SWE-bench Lite 15.3% → 8.7% | not isolated; authors suggest stronger optimization or base model | none reported | [[swe-gym]] §4.2 |
| Kimi-Dev SFT prior (teacher data, not self-generated) | Qwen2.5-72B after mid-training and cold-start SFT on DeepSeek-R1 trajectories | SWE-Agent pass rate degraded when adapted with 200 trajectories | hypothesized memorization of a single data mode | none; the RL prior did not show the drop | [[kimi-dev]] §4.2 |
| DeepSeek-R1 stages vs R1-Zero | both from V3-Base; cold start of "thousands" of examples | cold-start model Dev1 vs RL-only R1-Zero: AIME 2024 59.0 vs 77.9, SimpleQA 17.8 vs 30.3; final R1 vs R1-Zero: GPQA Diamond 71.5 vs 75.8, CNMO 2024 78.8 vs 88.1 | limited cold-start size (authors, for Dev1 reasoning scores); final-stage gaps not explained | not framed as a fix; reasoning RL after the cold start reached Dev2 AIME 74.0 | [[deepseek-r1]] §4, Table 3 |

Two independent settings report capability regression from repeated rejection-sampling rounds: ReST-EM on APPS and HumanEval, and Llama 2 at RLHF-V3 (Replicated; the mechanism is not isolated in either). ReST-EM's regression occurred although every round restarted from the base model and capped samples per problem; the authors attribute it to the small problem set. Llama 2 reported that pooling top samples from all earlier rounds addressed its regression. Two design choices in these sources keep weights or data closer to earlier distributions: restarting from the base model, which transferred better to HumanEval than continuing from the previous iterate ([[rest-em]] Fig. 7), and pooling accepted samples across rounds ([[llama-2]] §3.2.3). V-STaR also accumulates correct samples across iterations ([[v-star]] Alg. 1) but does not compare this with training on the last iteration's samples only.

## §5 Cold-start SFT, iterative distillation, and cross-stage distillation

The round-structure panel of the companion figure ([figures/iterative-loop.html#rounds](figures/iterative-loop.html#rounds)) places the stage order of every pipeline in §1, §5, and §6 side by side, so the reader can see where sampling, filtering, SFT, RL, and distillation occur and which rounds restart from the base model.

### 5.1 DeepSeek-R1: cold start versus R1-Zero

A cold start is a small SFT set applied before RL so that RL begins from a model with a usable output format. A long chain-of-thought (long-CoT) example is a response that contains an extended step-by-step reasoning trace before the answer. DeepSeek-V3-Base is a mixture-of-experts (MoE) model, in which a router activates a subset of expert feed-forward blocks per token (671B total, 37B activated parameters; [[deepseek-r1-recipe]]). DeepSeek-R1-Zero applied GRPO to DeepSeek-V3-Base with rule-based accuracy and format rewards and no SFT; its AIME 2024 pass@1 rose from 15.6% to 77.9% ([[deepseek-r1]] §2.3). Its outputs had poor readability and language mixing (§3), and the authors describe its performance as limited in writing and open-domain question answering (§1). DeepSeek-R1 added four stages ([[deepseek-r1-recipe]]): (1) Dev1, cold-start SFT on "thousands" of long-CoT examples built from R1-Zero samples that were filtered for correctness and readability, refined by DeepSeek-V3, and verified by annotators (B.3.2); (2) Dev2, reasoning RL with a language-consistency reward; (3) Dev3, SFT from DeepSeek-V3-Base on about 600k rejection-sampled reasoning samples plus about 200k non-reasoning samples, 804,745 in total (B.3.3, Table 5); (4) R1, RL with rule, RM, format, and language rewards, with preference rewards only in the final 400 of 1,700 steps (§3.2.2).

| Benchmark ([[deepseek-r1]] Table 3) | R1-Zero | Dev1 | Dev2 | Dev3 | R1 |
|---|---|---|---|---|---|
| AIME 2024 pass@1 | 77.9 | 59.0 | 74.0 | 78.1 | 79.8 |
| CNMO 2024 pass@1 | 88.1 | 58.0 | 73.9 | 77.3 | 78.8 |
| GPQA Diamond pass@1 | 75.8 | 66.1 | 70.7 | 71.2 | 71.5 |
| IF-Eval (prompt strict) | 46.6 | 71.7 | 72.0 | 78.1 | 83.3 |
| AlpacaEval 2.0 (LC win rate) | 24.7 | 50.1 | 55.8 | 62.1 | 87.6 |
| SimpleQA (correct) | 30.3 | 17.8 | 28.2 | 24.9 | 30.1 |
| C-Eval (EM) | 92.8 | 85.7 | 91.9 | 86.4 | 91.8 |
| Aider-Polyglot (acc.) | 12.2 | 6.7 | 25.6 | 44.8 | 53.3 |

R1-Zero and Dev1 both start from DeepSeek-V3-Base, so their columns compare an RL-only model with a cold-start SFT model; they do not show a drop within one run. The cold-start model scores higher on instruction following (IF-Eval 71.7 vs 46.6) and lower on reasoning and factual QA (AIME 2024 59.0 vs 77.9; SimpleQA 17.8 vs 30.3); the authors attribute the reasoning gap to the limited cold-start size (§4). Reasoning RL from Dev1 (Dev2) reached AIME 2024 74.0 and CNMO 2024 73.9. Dev3, trained again from V3-Base on rejection-sampled reasoning data plus non-reasoning data, scored higher than Dev2 on AlpacaEval 2.0 (62.1 vs 55.8) and Aider-Polyglot (44.8 vs 25.6) and lower on C-Eval (86.4 vs 91.9) and SimpleQA (24.9 vs 28.2). The final R1 remains below R1-Zero on GPQA Diamond and CNMO 2024 (Table 3). RL from a base model failed to improve AIME for 7B dense and 16B MoE bases, with repetition as length grew, and worked for 32B dense and larger bases (G.1). Status: Result (single study; one model family; stages are not ablated in isolation).

### 5.2 GLM-4.5: expert models and iterative distillation

GLM-4.5 post-training has two stages ([[glm-4-5]] §3). Stage 1 trains Reasoning, Agent, and General-chat experts, each with a small cold-start SFT set with extended CoT and then RL (§3.1). Stage 2 distills the experts into one hybrid model with an Overall SFT set of millions of expert samples at up to 128K tokens, balancing data with and without explicit thinking (§3.1). For agent tasks, the report describes iterative distillation: RL on the cold-start model until a step count or plateau, then "substituting the original cold-start data with responses generated by the RL-trained model", SFT again, and further RL with harder tasks (§3.3.2). The report gives no numbers or ablation for this loop. Status: Result (single study, qualitative).

### 5.3 GLM-5: on-policy cross-stage distillation after sequential RL

GLM-5 runs SFT, Reasoning RL, Agentic RL, and General RL in sequence and states that optimizing distinct objectives in sequence "can lead to the cumulative degradation of previously acquired capabilities" ([[glm-5]] §3.5). Its last stage is on-policy distillation: the student samples responses, and final checkpoints of earlier stages act as teachers on prompts drawn from those stages' RL training sets. The GRPO advantage is replaced by

Â_{i,t} = sg[ log( π_teacher(y_{i,t} | x, y_{i,<t}) / π_θ(y_{i,t} | x, y_{i,<t}) ) ]

where y_{i,t} is token t of the student's sample i, π_teacher is the teacher's inference-engine probability, π_θ the student's training probability, and sg the stop-gradient operation (§3.5, Eq. 2). The group size is 1 and the batch size 1,024 (§3.5). The introduction says this distillation is used "throughout" post-training, while §3 calls it the final stage (§1, §3). No numbers isolate its effect. In GLM-5 SFT, erroneous segments in agent trajectories "are retained but masked out in the loss" (§3.1). The on-policy-distillation course (ch-04) derives this objective. Status: Result (single study, no ablation reported).

### 5.4 Distillation from a stronger model versus self-generated data

Llama 2 rejection-sampled only with 70B and trained smaller models on those samples ([[llama-2]] §3.2.3). ReST-EM found that PaLM 2-S fine-tuned on PaLM 2-L solutions beat PaLM 2-S fine-tuned on its own ReST-EM data, which the authors attribute to more questions having at least one solution ([[rest-em]] §5.3, Fig. 6). At 32B, DeepSeek-R1-Distill-Qwen-32B (SFT on the 800k set) scored AIME 2024 pass@1 72.6 and GPQA Diamond 62.1, while RL from Qwen2.5-32B-Base for over 10K steps scored 47.0 and 55.0 ([[deepseek-r1]] Table 16). A stronger sampler raises per-problem success p, and by §1.2 it raises the share of hard problems that yield any accepted sample (Interpretation, derived from §1.2).

## §6 Agentic cases

### 6.1 SWE-Gym: on-policy rejection-sampling SFT lowered resolution with a general scaffold

SWE-Gym provides 2,438 Python tasks from 11 repositories with executable environments and unit tests ([[swe-gym]] §3). A scaffold is the tool set and prompting loop around the model; OpenHands is a general-purpose ReAct-style scaffold and MoatlessTools a fixed workflow (§4.1). Resolution rate is the share of SWE-bench tasks whose tests pass after the agent's patch. Fine-tuning Qwen2.5-Coder-Instruct-32B on 491 successful trajectories sampled from GPT-4o and Claude 3.5 Sonnet raised SWE-bench Lite from 3.0% to 15.3% and SWE-bench Verified from 7.0% to 20.6% (Table 3). The fine-tuned 32B model then sampled 6 trajectories per task at temperature 0.5, giving 868 successful on-policy trajectories; fine-tuning the base 32B model on those plus the 491 teacher trajectories lowered SWE-bench Lite to 8.7%, "suggesting that self-improvement is not yet working" (§4.2). With MoatlessTools, two iterations of 30 rollouts per task at temperature 1.0 raised the 7B model from 7.0% to 9.0% to 10.0% and the 32B model from 19.0% to 19.7% with no further gain (§4.3, Table 4). A verifier trained on 1,318 successful and 1,318 failed trajectories raised Verified from 20.6% at k = 1 to 32.0% at best-of-16 (§5.1.1). Status: Result (single study).

### 6.2 Kimi-Dev: single-turn RL as a prior for multi-turn agents

Kimi-Dev trains Qwen2.5-72B-Base in the Agentless format, where issue resolution is split into single-turn steps (file localization, code edit, test writing) with verifiable rewards ([[kimi-dev]] §2.1, §3.1). The recipe is about 150B tokens of mid-training on GitHub issue, pull-request, and synthetic reasoning and agentic data, a cold start on DeepSeek-R1 reasoning outputs for SWE-Gym and SWE-bench-extra tasks, and RL on the code-edit step with a 0/1 execution reward (§3.2–3.3). Prompts with pass@16 = 0 are removed at first (1,200 prompts remain), and 500 previously excluded prompts are re-added every 100 RL steps (§3.3, App. C.1). In late RL, "positive example reinforcement" adds successful samples from recent iterations to the current batch, which the authors report improved BugFixer RL when exploration diminished (§3.3, App. C.2, Fig. 8; no numbers). The resulting model resolves 60.4% of SWE-bench Verified with 40 patches and 40 tests per issue (Table 1).

Fine-tuning Kimi-Dev on 5,016 SWE-smith trajectories collected with Claude 3.7 Sonnet gave 48.6% pass@1 as a multi-turn SWE-Agent; SWE-agent-LM (32B) trained on the same data scores 40.2% (§4.1, Table 2). The authors compared four starting points for this adaptation: base, mid-trained (MT), long-CoT SFT, and RL. The RL prior outperformed the others at nearly all SFT budgets and reached the base prior's best pass@1 with 2²³ SFT tokens instead of 1.5 × 2²⁸ (§4.2, Fig. 5). The SFT prior was better at zero-shot but degraded with 200 trajectories (§4.2). Under turn limits, the adapted RL prior kept improving beyond 70 turns, while the SFT, MT, and base priors showed diminishing returns near 70, 60, and 50 turns (Fig. 6). On SWE-bench-Live and SWE-bench Multilingual, the SFT and RL priors generalized better than the base prior with few adaptation trajectories, and the gap narrowed with more trajectories (App. G.4). Status: Result (single study).

### 6.3 Llama 3 tools and GLM-4.5 agent RL

Llama 3 did not use rejection sampling for tool-use data and reports no gains from rejection sampling on tool benchmarks ([[llama-3-recipe]], v3 §4.3.5). GLM-4.5 agent RL weights each of K sampled traces by its reward minus the group-mean reward, computes loss only on model-generated tokens, and halts a trace with reward 0 when the tool-call format is wrong ([[glm-4-5]] §3.3.2). The authors state that RL on web search and SWE tasks improved other tool-use and coding benchmarks, without numbers (§3.3.2). Implication (Interpretation): for agents whose policy has a low resolution rate, the sources in this chapter report three other uses of self-generated data, none compared directly with on-policy RSFT: verifier training from failures (+11.4 points Best@16 over k = 1 on SWE-bench Verified; [[swe-gym]] §5.1.1), distilling RL checkpoints back into SFT (no numbers; [[glm-4-5]] §3.3.2), and single-turn RL priors before multi-turn SFT ([[kimi-dev]] §4.2, Fig. 5).

## §7 Which update generalizes and forgets less (preview of ch-38a)

| Evidence | Setting | Positive-only or SFT update | Update that uses failures (RL) | Status |
|---|---|---|---|---|
| [[raft-reinforce-rej-minimalist]] Table 1, Fig. 3 | Qwen2.5-Math-7B, in-domain math | RAFT++ 56.1; entropy falls faster; plateau | GRPO 56.3; keeps improving | Result (single study) |
| [[kimi-dev]] §4.2, Fig. 7 | Qwen2.5-72B, SWE-Agent adaptation | long-CoT SFT prior degrades at 200 trajectories | RL prior better at nearly all budgets; slightly better after end-to-end RL | Result (single study) |
| [[rlvr-beyond-base-model]] Abstract (v5) | several model families; math, code, visual reasoning | distillation can introduce new reasoning patterns | RLVR (RL with verifiable rewards) higher at small k, base higher at large k; boundary often narrows with training | Result (one study, several families) |
| [[prorl]] §4.2–4.3 | R1-Distill-Qwen-1.5B, > 2k RL steps | not tested | math pass@128 often declines; unseen Reasoning Gym task boxnet 0.00 → 7.91 | Result (single study) |
| [[deepseek-r1]] Table 16 | Qwen2.5-32B-Base | distillation from R1: AIME 72.6 | RL from base: AIME 47.0 | Result (single study; stronger teacher) |

These sources do not compare SFT on filtered self-samples with on-policy RL at matched prompts and compute for forgetting of non-target skills. That comparison is an Open question within this chapter's sources; ch-38a covers studies that measure it directly (arXiv:2501.17161, arXiv:2509.04259, arXiv:2510.18874).

## Negative samples and negative feedback

This section uses the four meanings of "negative" from the course standard: (1) negative marginal value, (2) negative as content, (3) negative as conditioning, (4) negative as gradient. ch-31a covers (2) and (3) in supervised training, and ch-43a derives (4).

**Where negatives come from.** Answer matching ([[rest-em]], [[star]], [[rft-scaling-relationship-math]]), unit tests ([[swe-gym]], [[kimi-dev]]), reward models ([[llama-2]], [[llama-3]]), and tool-format checks ([[glm-4-5]]). False-negative rates are not reported by any source in this chapter. False positives are reported: correct answers with incorrect reasoning under high-temperature sampling or rationalization ([[star]] §5; [[rest-em]] §4), and TestWriter tests that pass for the wrong reason during Kimi-Dev RL ([[kimi-dev]] §3.5.3).

**What current practice does with them.** RSFT, STaR, RFT, ReST-EM, Llama 2 rejection sampling, and DeepSeek-R1 stage 3 discard rejected samples: meaning (1), zero weight in the §2 identity. STaR rationalization reuses the failed problem and discards the failed rationale. GLM-5 keeps erroneous agent steps in context but masks them out of the loss, which is meaning (2) ([[glm-5]] §3.1). SWE-Gym's verifier is trained to output `<NO>` on failed trajectories with ordinary cross-entropy, which is meaning (2) for the verifier ([[swe-gym]] §5.1.1). V-STaR's DPO verifier lowers the likelihood of incorrect solutions relative to the reference, which is meaning (4) for the verifier, while its generator stays positive-only ([[v-star]] §3.1). Group-baseline RL in DeepSeek-R1 (GRPO) and in GLM-4.5 agent RL gives below-mean samples negative weights, which is meaning (4) for the policy ([[deepseek-r1]] Eq. 3; [[glm-4-5]] §3.3.2).

**Mechanism.** For a softmax over logits z with probabilities p, the gradient of a token's log-probability is ∂ log p_y / ∂ z_j = 1[j = y] − p_j, where y is the target token and j ranges over the vocabulary. Take p = [0.7, 0.2, 0.1] and push down token 3 (p = 0.1). Gradient descent on log p_3 moves logits along the direction d = −[−0.7, −0.2, 0.9] = [+0.7, +0.2, −0.9]. For a small step size η, the first-order change in probabilities is Δp_j ≈ η · p_j · (d_j − Σ_k p_k d_k), with Σ_k p_k d_k = 0.49 + 0.04 − 0.09 = 0.44, which gives Δp ≈ η · [+0.182, −0.048, −0.134]. Token 1, the already most likely alternative, receives all of the removed mass, and token 2 also loses mass. Pushing up token 1 moves logits along [0.3, −0.2, −0.1], a smaller change that shrinks as p_1 approaches 1. A negative gradient on an unlikely sample therefore concentrates probability on the model's current top choice. Positive-only RSFT never applies meaning (4), so it has no likelihood-displacement risk; its risk is the entropy reduction measured in [[raft-reinforce-rej-minimalist]] §5.1.

**Evidence with numbers.** Benefit of using failures: V-STaR, which adds a verifier trained on failed samples, improved test accuracy by 4–17 points over existing self-improvement and verification baselines ([[v-star]] Abstract); the benefit of negative advantages in online RL was small in one comparison: GRPO 56.3 vs RAFT++ 56.1 on Qwen and 28.4 vs 27.6 on LLaMA ([[raft-reinforce-rej-minimalist]] Table 1). Failure mode: REINFORCE with ±1 reward on LLaMA-3.2-3B scored 24.2, below RAFT++ at 27.6, and removing all-wrong prompts gave the largest reward gain in the ablation (Table 1, Fig. 4). The authors' interpretation is that final-answer correctness is too coarse a negative label and that unlearning on such negatives is less stable than fine-tuning on positives (§5, Interpretation). No source in this chapter measures what share of an RL gain comes from negatives.

**Controls.** Remove all-wrong prompts or use a group-mean baseline ([[raft-reinforce-rej-minimalist]] Fig. 4); filter prompts with pass@16 = 0 and re-add them later ([[kimi-dev]] App. C.1); clip importance ratios, with asymmetric clipping to keep entropy ([[raft-reinforce-rej-minimalist]] §5.1); add positive samples late in RL when exploration falls ([[kimi-dev]] App. C.2); mask erroneous steps instead of training on them ([[glm-5]] §3.1); when DPO is applied to a model's own samples, add a negative log-likelihood (NLL) term on chosen responses, as Llama 3 did with coefficient 0.2 on human-labeled pairs of responses from the previous round's best models, stating that it prevents the chosen log-probability from decreasing (no ablation printed; [[llama-3]] §4.1.4).

**Diagnostics.** Log per round: acceptance rate by difficulty bin, distinct accepted solutions per problem ([[rft-scaling-relationship-math]] Table 2), policy entropy ([[raft-reinforce-rej-minimalist]] Fig. 3), share of all-correct and all-wrong prompts, pass@1 and pass@k at large k ([[rest-em]] Fig. 5), a sample audit of accepted traces for wrong reasoning, and for agents the stuck-in-loop and empty-patch rates ([[swe-gym]] Table 3).

**Effect on generality.** ReST-EM, which discards negatives, raised pass@K at every K tested, with the gap typically largest at K = 1 ([[rest-em]] §5.2, Fig. 5), and positive-only updates lowered entropy faster than GRPO ([[raft-reinforce-rej-minimalist]] §5.1). This chapter's interpretation is that discarding negatives keeps coverage only while sampling stays diverse (Interpretation; no source measures pass@k at large k for RAFT). High-temperature sampling raises correct-answer, wrong-reasoning targets, which STaR states prevents generalization ([[star]] §5), and ReST-EM reports the same kind of false positive under rationalization ([[rest-em]] §4). Calibration, hallucination, and over-refusal effects of these choices are not reported by the sources in this chapter.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 2-Chat 70B | 70B | RL (rejection sampling) | sampler; students | rejection sampling only with 70B; smaller chat models fine-tuned on its samples | arXiv:2307.09288v2 §3.2.3; [[llama-2-recipe]] | verified 2026-09-14 | no ablation reported |
| Llama 2-Chat | all | RL (rejection sampling) | samples per prompt K; fine-tuning hyperparameters | not printed | v2 §3.2.3, Fig. 7, A.3 | not reported (body and A.3 checked) | Fig. 7: max − median reward grows with N |
| Llama 2-Chat-RLHF | 70B | RL (rejection sampling) | sampling temperature | optimal T ∈ [1.2, 1.3] for 10–100 samples; re-adjusted per version | v2 §3.2.3, Fig. 8 | verified 2026-09-14 | Fig. 8 (T = 0.6–1.5) |
| Llama 2-Chat | all | RL (rejection sampling) | candidate pool | up to V3: previous iteration only; later: top samples from all prior iterations | v2 §3.2.3 | verified 2026-09-14 | V3 regression (poem rhyming); fix reported without figures |
| Llama 2-Chat | all | RL | schedule | rejection sampling only until RLHF (V4); then PPO on the rejection-sampling checkpoint before sampling again | v2 §3.2.3 | verified 2026-09-14 (V4 wording ambiguous) | Fig. 11: RLHF-v5 with PPO above v5 without PPO (plot only) |
| Llama 2-Chat | 7B, 13B / 34B, 70B | RL | PPO KL coefficient β | 0.01 / 0.005 | v2 §3.2.3, Eq. 4 | verified 2026-09-14 | no ablation reported |
| Llama 3.1 405B | 405B | SFT (rejection sampling) | K; selection | K typically 10–30 per human-annotation prompt; RM picks best | arXiv:2407.21783v3 §4.2.2; [[llama-3-recipe]] | verified 2026-09-14 | no ablation reported |
| Llama 3.1 405B | 405B | preference | DPO on round data | LR 10⁻⁵; β = 0.1; NLL coefficient 0.2 on chosen | v3 §4.1.4 | verified 2026-09-14 | no ablation printed |
| GPT-J + STaR | 6B | SFT (self-generated) | sampling; init; LR; batch | greedy; original GPT-J each loop; Adam 10⁻⁶; 8 sequences × 1,024 tokens | arXiv:2203.14465v2 §3.1, §4.1, App. H; [[star]] | verified 2026-09-14 | App. H: LR search 10⁻⁷–10⁻⁴; §5: T 0.5/0.7 worse |
| LLaMA-7B/13B RFT | 7B, 13B | SFT (self-generated) | samples; temperature; dedup | k = 100; T = 0.7; one path per distinct equation list | arXiv:2308.01825v2 §3.3, App. A.3 | verified 2026-09-15 | Fig. 4, Table 5: k ∈ {1, …, 100}; dedup vs no dedup |
| LLaMA RFT | 7B–70B (Table 5) | SFT (self-generated) | epochs; examples | 3 epochs; ~47K (k = 100), 104K (U13B) | v2 Table 5 | verified 2026-09-15 | no epoch ablation for RFT; Table 5 compares data sizes (k = 1–100, no dedup, U13B, U33B) |
| LLaMA SFT baseline for RFT | 7B–70B | SFT | LR; batch; epochs | 2e-5 peak, 3% warmup; batch 128; 3 epochs | v2 App. A.1 | verified 2026-09-15 (RFT-specific LR not printed) | no ablation reported |
| PaLM 2-L ReST-EM | not reported | SFT (self-generated) | samples; decoding; cap | MATH 32, APPS 64 per problem; top-K 40, T 0.7; ≤ 10 kept per problem | arXiv:2312.06585v4 §5; [[rest-em]] | verified 2026-09-14 | §5.3: 3× samples in 1 iteration 40.3% < 41.9% at iteration 3 |
| PaLM 2 ReST-EM | not reported | SFT (self-generated) | init; iterations; LR | base model each iteration; MATH 3, APPS 2; LR, batch, epochs not reported | v4 §3, Figs. 2–3 | verified / not reported | Fig. 7: restart gives better HumanEval transfer |
| LLaMA 2 7B V-STaR | 7B | SFT (generator); reward-model (DPO verifier) | samples; iterations; tuning | k = 16 per iteration; 3 iterations; LoRA; generator from base on accumulated correct samples | arXiv:2402.06457v2 §4, Alg. 1 | verified 2026-09-15 | Fig. 3: V-STaR vs V-STaR[1 iter] at 3 × 16 samples |
| Qwen2.5-Math-7B RAFT/RAFT++/GRPO | 7B | RL (positive-only vs GRPO) | LR; prompts; samples; mini-batch; max length | AdamW 1×10⁻⁶; 1,024 prompts per iteration; n = 4; mini-batch 512; 4,096 tokens | arXiv:2504.11343v2 §4 | verified 2026-09-15; the Table 1 caption states that batch size, mini-batch size, and learning rate were tuned per algorithm, and the tuned values are not printed in v2 | no ablation of these §4 values reported |
| DeepSeek-R1 Dev1 | 671B MoE | SFT (cold start) | size; base; schedule | "thousands"; V3-Base; 2–3 epochs, cosine 5×10⁻⁵ → 5×10⁻⁶; 32,768 tokens; batch 128 | arXiv:2501.12948v2 B.3.2, B.4.2; [[deepseek-r1-recipe]] | verified 2026-09-14 | Table 3: Dev1 vs R1-Zero |
| DeepSeek-R1 Dev3 | 671B MoE | SFT (rejection sampling) | data; base | ~600k correct reasoning samples from the RL checkpoint + ~200k non-reasoning; 804,745 total; from V3-Base | v2 B.3.3, Table 5, Fig. 2 | verified 2026-09-14 (epochs conflict: v1 two, v2 2–3) | Table 3: Dev2 → Dev3 |
| GLM-4.5 agent expert | 355B MoE | SFT and RL (iterative distillation) | loop | after RL reaches a step count or plateaus, replace cold-start data with RL-model responses, SFT, continue RL on harder tasks | arXiv:2508.06471v1 §3.3.2 | verified 2026-09-14; cold-start size not reported | no ablation reported |
| GLM-5 | 744B total, 40B active | RL (on-policy cross-stage distillation) | teachers; group; batch | final checkpoints of earlier stages; group size 1; batch 1,024 | arXiv:2602.15763v2 §3.5, Eq. 2 | verified 2026-09-15 | no ablation reported |
| Qwen2.5-Coder-32B SWE-Gym | 32B | SFT (on-policy) | samples; temperature; data | 6 trajectories per task at t = 0.5; 868 on-policy + 491 teacher trajectories | arXiv:2412.21139v2 §4.2 | verified 2026-09-15 | SWE-bench Lite 15.3% → 8.7% |
| Qwen2.5-Coder-7B SWE-Gym MoatlessTools | 7B | SFT (on-policy) | rollouts; temperature; cap; iterations | 30 per task at T 1.0; per-instance cap 2; 2 iterations | v2 §4.3, Tables 4, 6 | verified 2026-09-15 | Table 6: cap 2 slightly above full data |
| Kimi-Dev 72B | 72B | RL (single-turn code edit) | rollouts; prompts; context; filtering | 10 rollouts for each of 1,024 problems; 5 steps per iteration; 64k context; pass@16 = 0 removed, +500 prompts every 100 steps | arXiv:2509.23045v3 §3.3, §3.5.3, App. C.1 | verified 2026-09-15 | App. C.2 Fig. 8 (positive example reinforcement; plot only) |
| Kimi-Dev 72B | 72B | SFT (agent adaptation) | data; context | 5,016 SWE-smith trajectories; 64K training context; 128K and 100 turns at inference | v3 §4.1 | verified 2026-09-15 | Fig. 5 data sweep |

**Starting point for a small general-purpose run.** No source in this table ran rejection-sampling rounds at 7B with a full generality panel, so the values below keep the conditions of the runs that produced them. For a verifiable math task, the closest verified round settings are ReST-EM's: 32 samples per problem at temperature 0.7 with top-K 40, at most 10 correct samples kept per problem, fine-tuning the base model again each iteration, and held-out evaluation after each iteration (PaLM 2 models with unreported sizes, 7,500 MATH training problems, learning rate not reported). For online training of a 7B model on verifiable math, the RAFT++ versus GRPO comparison used AdamW at 1 × 10⁻⁶, 1,024 prompts per iteration, 4 samples per prompt, mini-batch 512, and a 4,096-token limit (Qwen2.5-Math-7B-base, Numina-Math prompts, verl); the Table 1 caption states that batch size, mini-batch size, and learning rate were tuned per algorithm, and the tuned values are not printed. For agent tasks with a 32B coder model, SWE-Gym trained on 491 successful teacher trajectories (+13.6 points on SWE-bench Verified) and measured a drop on SWE-bench Lite (15.3% → 8.7%) after adding one round of 868 on-policy trajectories.

## Generalization lens

**(a) What increases breadth.** More distinct accepted solutions per problem, including pooling samples from several models ([[rft-scaling-relationship-math]] Tables 2–3). Restarting each round from the base model ([[rest-em]] Fig. 7) or pooling accepted samples from all earlier rounds ([[llama-2]] §3.2.3). Mixing non-target data into the rejection-sampling SFT set: DeepSeek-R1 Dev3 raised AlpacaEval 2.0 from 55.8 to 62.1 and Aider-Polyglot from 25.6 to 44.8 over Dev2, which the authors attribute to the non-reasoning and code-engineering data, while C-Eval (91.9 → 86.4) and SimpleQA (28.2 → 24.9) fell in the same step ([[deepseek-r1]] §4, Table 3). Samples from a stronger model: PaLM 2-L solutions for PaLM 2-S, attributed to more questions having solutions ([[rest-em]] §5.3, Fig. 6), and R1 distillation for Qwen2.5-32B on reasoning benchmarks ([[deepseek-r1]] Table 16). Single-turn RL priors before multi-turn agent SFT ([[kimi-dev]] §4.2, App. G.4). Distilling earlier-stage checkpoints back into the final model after sequential RL ([[glm-5]] §3.5; no numbers).

**(b) What causes narrowing or forgetting.** Iterating on a small problem set ([[rest-em]] §5.1, Fig. 4). Using only the previous round's samples ([[llama-2]] §3.2.3). A sampler that has fit its training questions and yields few distinct paths ([[rft-scaling-relationship-math]] §3.3). Positive-only updates that lower entropy ([[raft-reinforce-rej-minimalist]] §5.1). High-temperature sampling and rationalization that admit correct answers with wrong reasoning ([[star]] §5; [[rest-em]] §4). A small cold-start set, whose SFT model scored below the RL-only model from the same base on reasoning and factual QA ([[deepseek-r1]] §4, Table 3). On-policy RSFT from an agent policy with a low resolution rate ([[swe-gym]] §4.2). Sequential RL stages with different objectives ([[glm-5]] §3.5).

**(c) How to measure it at this stage.** After every round, evaluate: held-out tasks from outside the training prompts, as ReST-EM did with GSM8K, HumanEval, the Hungarian exam, and BBH ([[rest-em]] §5.4); pass@k at large k beside pass@1 ([[rest-em]] Fig. 5; [[rlvr-beyond-base-model]]); train accuracy against test accuracy ([[rest-em]] Fig. 4); distinct solutions per problem; entropy; and a panel of non-target capabilities such as instruction following, factual QA, and other languages, since DeepSeek-R1 stages moved SimpleQA and C-Eval in opposite directions from reasoning scores ([[deepseek-r1]] Table 3). For agents, add turn-limit profiles ([[kimi-dev]] Fig. 6) and fresh or multilingual suites ([[kimi-dev]] App. G.4). Known measurement errors: AIME has 30 problems and gave noisy trends ([[raft-reinforce-rej-minimalist]] §4); V-STaR's MATH transfer set has 150 level-1 problems ([[v-star]] §4); STaR reports best results after saturation on the CQA dev set ([[star]] §4.1); RFT evaluates only the GSM8K test split ([[rft-scaling-relationship-math]] §3.3); Llama 2's human evaluation excluded coding and reasoning prompts ([[llama-2]] §3.4.2), and its RLHF-V3 regression was found by qualitative analysis ([[llama-2]] §3.2.3).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Training each round only on the previous round's accepted samples | held-out or unmeasured skills regress while training accuracy rises | per-round train–test gap; compare pooled data or restart-from-base against last-round-only on a held-out panel |
| No per-problem cap | accepted set dominated by problems with high pass rate | histogram of accepted samples by difficulty bin; gradient share from problems with pass rate above 0.75 |
| Counting accepted samples instead of distinct solutions | more samples per round without accuracy gain | distinct solutions per problem using a dedup key (equation list, normalized program) |
| Judging positive-only training from early curves | fast early gain, then plateau | policy entropy per step and pass@k at large k over the whole run |
| ±1 rewards without a baseline on all-wrong prompts | training reward falls or becomes unstable | share of all-wrong groups; ablate removing them |
| Accepting correct answers without checking reasoning | more correct-answer, wrong-reasoning traces at high temperature or with rationalization | manual audit of a random sample of accepted traces per round |
| Assuming a cold-start SFT model keeps the reasoning level of an RL-only model | the cold-start checkpoint scores below the RL-only model from the same base on reasoning and factual QA | evaluate the cold-start checkpoint on the full panel against the base and an RL-only reference, not only after RL |
| On-policy RSFT from an agent policy with a low resolution rate | resolution rate falls; stuck-in-loop rate changes | compare with teacher-trajectory SFT at equal trajectory count |
| Evaluating only after the last of several RL stages | earlier-stage skills are lower than at their own stage checkpoint | evaluate each stage checkpoint on every stage's metrics |
| Trusting a learned verifier's selections | best-of-k stops tracking pass@k | plot pass@k and best@k together, as SWE-Gym does |

## Check your understanding

1. Using the §2 identity, explain why an RSFT round with acceptance rate 0.1 behaves like REINFORCE with a larger learning rate, and what this implies for choosing the SFT learning rate across rounds as acceptance rises.
2. Why did one ReST-EM iteration with three times more samples per problem score below three iterations with fewer samples per iteration? Name the mechanism that the sample budget alone does not provide.
3. LLaMA-33B produced 88.7 correct paths per question but did not improve under RFT. Explain the result using distinct paths per question, and propose a change to the sampler that the paper's U33B result supports.
4. In the §2 worked example, the group-mean baseline assigns zero weight to prompts B and C. Explain why this removes both the easy-problem bias and the all-wrong push-down, and what information it discards.
5. STaR rationalization is sometimes described as training on failures. Explain precisely which object receives gradient, and why ReST-EM chose not to use rationalization.
6. DeepSeek-R1's cold-start SFT model (Dev1) scored below the RL-only R1-Zero on AIME 2024 and SimpleQA but above it on IF-Eval, although both start from DeepSeek-V3-Base. Give a causal account of why a small SFT set and RL from the same base can move these benchmarks in opposite directions, and state what measurement you would add after a cold start.
7. SWE-Gym's on-policy RSFT lowered SWE-bench Lite from 15.3% to 8.7%, while Kimi-Dev's single-turn RL prior improved later agent SFT. Using §1.2 and §6, explain which properties of the starting policy and task horizon differ between the two settings.
8. GLM-5 replaces the GRPO advantage with a log-ratio to earlier-stage teachers. Explain why this term can recover skills lost in later stages and why group size 1 is sufficient for it.

## Connections

- **Previous (dependency):** ch-30c — Weight Averaging and Model Merging for Generalist Models. Merging is another way to recover earlier capabilities; §4 and §5.3 give the data-side methods.
- **Next:** ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood. It covers negative meanings (2) and (3), which RSFT discards.
- ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate. Every RSFT round is an SFT run with these settings.
- ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control. Measurement protocol for the regressions in §4.
- ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage; ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data. Teacher-generated data, compared with self-generated data in §5.4.
- ch-37 — Policy-Gradient Foundations for Language Models; ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO. Derivations behind §2.
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity. Direct comparisons previewed in §7.
- ch-43 — Entropy, Output Diversity, and KL Control in RL; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. Entropy control and the negative-gradient mechanism used in §2 and the negative-feedback section.
- ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing. Forgetting across sequential RL stages and GLM-5 cross-stage distillation, introduced in §5.3.
- ch-16 — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse. Prompt filtering such as pass@16 = 0 removal.
- ch-45 — Self-Improvement Loops and Multi-Stage Reasoning Pipelines; ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability; ch-29d — User Simulators, Trajectory Verification, and Failed Trajectories. Judge-based self-improvement loops, multi-turn agent RL, and failed-trajectory handling that extend §6.

## Sources

- [[llama-2]] — iterative RLHF schedule, rejection sampling with 70B, temperature, RLHF-V3 regression and pooling fix, PPO β values, human-evaluation scope.
- [[llama-2-recipe]] — verified rejection-sampling and PPO rows, including "not printed" fields.
- [[llama-3]] and [[llama-3-recipe]] — K typically 10–30, RM selection, SFT from the pre-trained model each round, DPO with NLL 0.2, no rejection sampling for tools.
- [[rejection-sampling-finetuning]] — pattern definition only; no numbers taken from this card.
- [[rft-scaling-relationship-math]] — RFT on GSM8K: distinct paths per question, k sweep, deduplication, multi-model pooling, 33B case (chapter excerpt verified against arXiv:2308.01825v2).
- [[rest-em]] — EM objective, restart from base, samples and cap, iteration results, APPS regression, transfer, pass@K, distillation from PaLM 2-L.
- [[star]] — indicator-reward policy-gradient identity, rationalization, CQA and GSM8K results, temperature and false-positive observations.
- [[v-star]] — DPO verifier trained on correct and incorrect solutions, iteration and verifier-in-loop results (chapter excerpt verified against arXiv:2402.06457v2; authors are Hosseini et al.).
- [[raft-reinforce-rej-minimalist]] — RAFT, RAFT++, REINFORCE, GRPO, and Reinforce-Rej comparison; entropy collapse; all-wrong prompt ablation (chapter excerpt verified against arXiv:2504.11343v2).
- [[deepseek-r1]] and [[deepseek-r1-recipe]] — R1-Zero vs cold start, Table 3 stage results, rejection-sampling SFT stage, 32B distillation vs RL.
- [[glm-4-5]] — expert models, cold start, iterative distillation, agent RL objective and format penalty.
- [[glm-5]] — sequential RL stages, on-policy cross-stage distillation objective, loss-masked erroneous agent steps (chapter excerpt verified against arXiv:2602.15763v2).
- [[swe-gym]] — teacher-trajectory SFT, on-policy self-improvement negative result, MoatlessTools iterations, per-instance cap, trajectory verifier (chapter excerpt verified against arXiv:2412.21139v2).
- [[kimi-dev]] — Agentless RL recipe, prompt filtering, positive example reinforcement, SWE-Agent adaptation and prior comparison (chapter excerpt verified against arXiv:2509.23045v3).
- [[rlvr-beyond-base-model]] — pass@k at large k for RLVR versus base models (arXiv:2504.13837v5 Abstract).
- [[prorl]] — Diminish regime and out-of-distribution gains under prolonged RL.
- Not cited: the library card `iterative-sft-rl` is unverified and its Llama 2 and Tülu 3 numbers conflict with the primary sources listed above.
