<!-- chapter: ch-31a
     track: sft
     kind: content
     title: Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood
     deps: [ch-31]
     sources: [[unlikelihood-training]], [[cringe-loss]], [[chain-of-hindsight]], [[lema-learning-from-mistakes]], [[physics-of-lm-2-2-learning-from-mistakes]], [[critique-fine-tuning]], [[dust-into-gold-negative-distillation]], [[learning-from-failure-nat]], [[orpo]], [[nft-negative-aware-finetuning]], [[redi-reinforcement-distillation]], [[negative-examples-likra]], [[pretraining-with-human-preferences]], [[likelihood-displacement]]
     figures: figures/negative-gradient-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood

> **Core insight.** A failed sample can enter a supervised loss in four ways, and only one of them, an explicit decrease of its likelihood, removes probability mass from that sample. Training on failures as ordinary targets gave mixed results: a LLaMA-7B student distilled on MATH scored 3.03 when correct and wrong GPT-3.5 traces were mixed, against 5.29 with correct traces only ([[dust-into-gold-negative-distillation]] Table 2), while unlabeled GPT-3.5 failures raised a LLaMA-2-Chat-7B agent's math average from 55.90 to 63.39 and also raised its tool-call error rate from 3.58% to 10.47% ([[learning-from-failure-nat]] Tables 2 and 6). Failures used as corrected targets, critiques, labeled conditioning, or a bounded negative term beat the positive-only arm in the main comparisons of LEMA, CFT, NAT, NFT, and REDI; the exceptions are listed in each section. In the one generating-model study that measures the split, positives carry most of the gain: in the NFT Qwen2.5-32B run, 80% of the gain comes from positives and 20% from negatives ([[nft-negative-aware-finetuning]] §5.3).
>
> **Guideline.** When failures come with a correction or critique written by a stronger model, place the failed attempt in the input and train on the correction, because this added accuracy without any negative gradient at equal training tokens ([[lema-learning-from-mistakes]] Table 2) and at matched sequence length ([[critique-fine-tuning]] Table 10). When failures carry only a pass/fail label, use them as conditioning (a label in the prompt that is replaced by the positive label at inference) or as a bounded negative term next to a positive log-likelihood term (REDI α = 0.8, NFT ε = 1.0), because symmetric or unclipped push-down collapsed or degraded in those studies ([[redi-reinforcement-distillation]] §4.3, [[nft-negative-aware-finetuning]] §5.4). When the failures come from a weaker generator than the positives, discard them, because negatives from a fine-tuned LLaMA-2-7B instead of GPT-3.5 left the NAT math average 3.16 and 6.20 points below the positive-only model ([[learning-from-failure-nat]] Table 5).

## Why this chapter matters for a general-purpose model

[[ch-31]] covers positive-only self-training: rejection-sampling fine-tuning (RFT), STaR, and ReST-EM keep the samples a verifier accepts and discard the rest.
The discarded share can be more than half of the generated samples. In the NAT agent data, about 7k GSM8K questions sampled three times gave about 9k positive and 12k negative trajectories, so 12/21 = 57% of generated trajectories were failures ([[learning-from-failure-nat]] App. B); the authors state that the discarded share "can exceed 60%" in tasks with planning or tool use (§1).

This chapter asks what those failures can add at the supervised fine-tuning (SFT) stage, and what each way of using them does to the output distribution.
For a general-purpose model the answer matters in three places. First, some uses raise entropy and keep pass@k (the probability that at least one of k samples is correct) where positive-only training lowers entropy ([[nft-negative-aware-finetuning]] Fig. 8; [[redi-reinforcement-distillation]] App. D.3). Second, some uses change general instruction following or capability benchmarks that were not trained ([[critique-fine-tuning]] Table 8; [[chain-of-hindsight]] Table 7). Third, the wrong use degrades the model below the positive-only baseline (§5.6).

Pipeline position: SFT and distillation SFT. Two pretraining studies are included because they show when the skill must be present ([[physics-of-lm-2-2-learning-from-mistakes]], [[pretraining-with-human-preferences]]). The rejected term of DPO is taught in [[ch-39]], and negative advantages in policy-gradient RL in [[ch-43a]].

## §1 Four meanings of "negative"

A **negative sample** is a sample labeled as wrong or undesirable by a verifier, judge, classifier, or human. The course uses four precise meanings; each later section states which one it discusses.

| Meaning | What enters the loss | Removes mass from the failure? | Examples in this chapter |
|---|---|---|---|
| (1) Negative marginal value | Nothing: the sample would lower performance if used as a positive target, so it is filtered out | No | WebInstruct responses with over 50% errors ([[critique-fine-tuning]] §2.1) |
| (2) Negative as content | The failure sits in the input, inside a corrected target, or in a separate module trained on failures; every term is ordinary cross-entropy | No | LEMA, retry data, CFT, Dust-into-Gold's negative LoRA, Likra's negative head |
| (3) Negative as conditioning | The failure is a target under a label, such as the word "incorrectly" or a bad-quality control token, that is not used at inference | No; its likelihood rises under its own label | Chain of Hindsight, NAT, conditional pretraining |
| (4) Negative as gradient | A term that decreases the failure's likelihood | Yes | Unlikelihood, CRINGE, ORPO's odds-ratio term, NFT, REDI |

LoRA (low-rank adaptation) trains low-rank matrices added to frozen weights; QLoRA applies LoRA to a quantized base model. GRPO and DAPO, used as comparison points in §5.4, are policy-gradient RL algorithms taught in the RL phase.

Only meaning (4) can produce likelihood displacement and the concentration effect of §2. Meanings (2) and (3) can still hurt when failure tokens are trained as targets: inline retries in a small-rank LoRA fine-tune lowered accuracy (§3.2), and NAT's labeled failures raised the tool-call error rate from 3.58% to 7.90% (§4.2). Plain maximum likelihood on failures, with no label or correction, lowered accuracy in Dust into Gold (MIX, §3.4) and in NAT's 13B run with 5k positives, but raised it in the other three NAT math settings (§4.2). Whether a failure has negative marginal value (meaning 1) therefore depends on the setting and has to be measured against a positive-only arm.

## §2 What ordinary SFT does to non-target tokens, and what a push-down adds

**Definition.** A model outputs a logit vector z over the vocabulary; the softmax p_j = exp(z_j) / Σ_i exp(z_i) turns logits into next-token probabilities. SFT minimizes the cross-entropy L = −log p_y of the target token y.

**Problem.** SFT never names a specific wrong token. The measurable question is how much probability a plausible wrong answer loses when only the correct answer is trained.

**Mechanism and formula.**

```
∂(−log p_y) / ∂z_j = p_j − 1[j = y]
∂(log p_k)  / ∂z_j = 1[j = k] − p_j
```

- z_j: logit of token j; p_j: its probability; 1[·]: 1 if the condition holds, else 0; y: target token; k: a negative token.
- Gradient descent on −log p_y lowers every non-target logit by η·p_j (η is the step size), so the most probable wrong tokens are lowered most.
- Gradient descent on log p_k (a push-down on k) lowers z_k by η(1 − p_k) and raises every other logit by η·p_j.

**Worked example.** Four tokens A, B, C, D with logits (2, 1, 0, −1) have p = (0.644, 0.237, 0.087, 0.032).
1. SFT step on target B with η = 1: gradient (0.644, −0.763, 0.087, 0.032); new logits (1.356, 1.763, −0.087, −1.032); new p = (0.353, 0.531, 0.083, 0.032). A lost 0.291 and C lost 0.004. D's logit fell by only 0.032, less than the normalizer fell, so its probability rose by 0.0004: one SFT step on the correct token does not suppress an already-unlikely wrong token.
2. Push-down on D with η = 1: new logits (2.644, 1.237, 0.087, −1.968); new p = (0.751, 0.184, 0.058, 0.0075). D lost 0.025, and A gained 0.107, taking mass from B (−0.053) and C (−0.029) as well. Because every other logit rises in proportion to its probability, a push-down on an already-unlikely token widens the gap between the most likely token and the rest.
3. Push-down on A instead: new p = (0.513, 0.341, 0.108, 0.038). Mass spreads to B, C, and D.

The interactive figure [figures/negative-gradient-explorer.html](figures/negative-gradient-explorer.html) lets the reader change the logits, pick SFT, push-down, unlikelihood, or CRINGE, and apply one or many steps to see which tokens receive the removed mass; its defaults reproduce the numbers above.

**Evidence that positive-only SFT leaves plausible wrong answers likely.**
- OPT-350M fine-tuned on chosen HH-RLHF responses: the log-probability of rejected responses rises together with that of chosen ones ([[orpo]] §3, Fig. 3). Result (single study).
- Mistral-7B-v0.1 SFT on ARC-Challenge: "the incorrect answers do not seem to be sharply distinguished" from correct ones during training ([[negative-examples-likra]] §3.1, Fig. 2). Result (single study).

**Conditions and limits.** The worked example updates logits directly. In a network, logits share parameters across tokens and responses, so a push-down on one response can lower a similar preferred response: in DPO refusal training this lowered Llama-3-8B-Instruct's refusal rate from 74.4% to 33.4% ([[likelihood-displacement]] §6.2). That multi-token analysis belongs to [[ch-43a]].

**Implication.** SFT already removes mass from wrong tokens that the model rates as likely. An explicit negative term adds a targeted decrease, and its side effect depends on how probable the negative already is.

## §3 Negatives as content (meaning 2)

### 3.1 Mistake-correction pairs: LEMA

**Definition.** LEMA (Learning from Mistakes) fine-tunes on chain-of-thought (CoT) data plus pairs of (question and wrong solution → correction) ([[lema-learning-from-mistakes]] §2).
**Problem.** Adding more correct CoT paths stops helping: LLaMA-2-70B reached 82.1 on GSM8K with 5.8M CoT tokens and 82.2 with 6.8M (§4.1).
**Mechanism.**
1. Sample solutions from several models (LLaMA-2, WizardMath, GPT-3.5-Turbo, GPT-4, and others) and keep those with a wrong final answer (Eq. 1).
2. GPT-4 writes the incorrect step, an explanation, and a corrected solution; keep corrections whose final answer is right (Eq. 2).
3. Train with the wrong solution in the input; "only the loss in the output part participates in the back-propagation" (App. B.2).

**Evidence.** 12,523 GSM8K, 6,306 MATH, and 7,241 CSQA pairs (§3.2). LLaMA-2-70B GSM8K 81.4 → 83.5 and MATH 23.6 → 25.0; SVAMP 80.3 → 81.6 and ASDiv 80.7 → 82.2, two test sets without their own training data (the GSM8K training data is used, §3.1) (Table 1). At equal training tokens (5.8M), LLaMA-2-70B CoT 82.1 vs LEMA 83.5 (Table 2). At equal example counts, LEMA helped four of five backbones; LLaMA-2-7B was the exception (Fig. 4). Result (single study).
**Conditions and limits.** Corrections came only from GPT-4: in a human check, 35 of 50 GPT-4 corrections were rated excellent and 4 poor, while nearly half of 20 GPT-3.5-Turbo corrections were poor, so GPT-3.5-Turbo was not used and the accuracy effect of a weaker corrector was not measured (§2.1, App. D.6). GPT-4 corrected its own MATH errors in 8.0% of 2,696 cases (App. D.6). The reported accuracy is that of the best of 20 saved checkpoints (2,000 steps, one every 100) evaluated on the test set (§3.3), which favors both arms; the mean of the best three checkpoints still gives 81.3 vs 83.4 on GSM8K for LLaMA-2-70B (App. D.1, Table 5).
**Implication.** Correction data carries a signal that extra positives do not, and every gradient still raises likelihood.

### 3.2 Retry data in pretraining: Physics of Language Models 2.2

**Definition.** Retry data inserts a wrong solution step followed by a special `[BACK]` token and the correct step ([[physics-of-lm-2-2-learning-from-mistakes]] §4).
**Mechanism and worked example.** At each sentence a wrong parameter is inserted with probability q (the retry rate), a second with probability q², and so on, so the expected number of inserted errors per step is q/(1 − q): 0.25 at q = 0.2 and 1.0 at q = 0.5 (derived from §4). All runs use the same number of training tokens, so higher q means fewer problems (§4).
**Evidence (GPT2-12-12 on synthetic iGSM math).**
- "the accuracy jumps from 78% to 94% by using retry rate = 0.5" on iGSM-med^{op=23}_{qp}, and "Masking mistakes is unnecessary" (Result 2-3).
- At q = 0.2 the model retries fewer than 0.3 times per problem; at q = 0.5 it retries 2-4 times, which label masking reduces (Result 4).
- On iGSM-med^{op=23}_{pq}, regenerating a step when a probe detects an error gives 78% ⇒ 80%, while retry pretraining gives 78% ⇒ 95% (§4, Result 6).
- LoRA fine-tuning of an error-free-pretrained model on retry data gives no significant gain; "for small LoRA ranks, finetuning even hurts and label masking becomes important" (Result 7, Fig. 7).
- Fake mistakes that are simple to generate (a later solution step inserted and then retracted with `[BACK]`, "retry_weak") "significantly improve the model's accuracy" (Result 8).

**Conditions and limits.** Synthetic data and one small architecture; the authors do not claim "that the synthetic data used here can directly aid in building future LLMs" (§1). Full fine-tuning on enough retry data did raise accuracy, but it used twice the training tokens of retry pretraining and the authors call it "essentially continued pretraining" (§5, Fig. 7 caption). Result (single study).
**Implication.** In this study, error correction was acquired during pretraining or full continued pretraining, not during a LoRA fine-tune; see the mask rules in §7.

### 3.3 Critique targets: Critique Fine-Tuning

**Definition.** CFT maximizes log P(c | [x; y]), where x is a question, y a noisy response, and c a critique written by GPT-4o ([[critique-fine-tuning]] §2.3).
**Evidence.** Qwen2.5-Math-7B, 50K WebInstruct examples (about 56% of responses judged correct), six-benchmark math average: SFT on the raw responses 35.1, SFT on GPT-4o-verified responses 40.4, SFT on GPT-4o answers 50.4, CFT 57.1 (Table 2). SFT on the raw responses, which contain over 50% errors, lowered Qwen2.5-7B from 37.4 to 20.2, an instance of meaning (1). IFEval instruction-level loose rose to 0.362 (SFT 0.330) and MT-Bench to 6.49 (SFT 5.23) (Table 8). With CFT examples selected to match the token length of the SFT data, CFT-Short scored 55.2 vs 50.4 for SFT on GPT-4o answers (Table 10). Result (single study).
**Conditions and limits.** Roughly 20% of 50 inspected critiques contained inaccuracies (§4). Mixing 50K CFT with 50K AceMath SFT examples scored 54.8, below CFT alone (Table 9). AIME24 has 30 questions, and the authors state its accuracy "is heavily impacted by the randomness" (§3.4). Training seeds: not reported.
**Implication.** Training the model to judge a response added math accuracy and instruction-following scores in a 7B model, with the error only in the input.

### 3.4 Wrong teacher traces in distillation: Dust into Gold

**Definition.** Distillation SFT trains a student on teacher outputs; standard practice keeps only traces with a correct final answer ([[dust-into-gold-negative-distillation]] Eq. 1).
**Worked example (complementarity).** On the 5,000-problem MATH test set, a LLaMA-7B student trained on positive traces solved 253 problems and one trained on negative traces solved 166; 29 were solved by both. Intersection over union IoU = 29 / (253 + 166 − 29) = 29/390 = 0.074 (Table 1).
**Evidence (MATH accuracy averaged over the 7 subjects, greedy decoding, GPT-3.5-Turbo traces, Table 2).** Fine-tune on original data 3.88; positive-only CoT distillation 5.29; MIX (maximum likelihood on positives and negatives) 3.03; negative training NT (log-likelihood on positives minus 0.05 × log-likelihood on negatives, App. Eq. 15) 4.48; sequence-level unlikelihood on negatives (weight 0.05, App. Eq. 16) 4.96; Negative Assistant Training 6.81. Negative Assistant Training (abbreviated NAT in that paper; unrelated to the NAT of §4.2) trains a separate LoRA on negatives with ordinary likelihood (meaning 2), freezes it, and merges its output into a positive LoRA through an attention weight constrained to [−0.5, 0.5], so the negative module's output can be added or subtracted (Eq. 4-5).
**Conditions and limits.** Absolute accuracies are 3-7%; seeds and variance are not reported. Result (single study).

## §4 Negatives as conditioning (meaning 3)

### 4.1 Chain of Hindsight

**Definition.** Chain of Hindsight (CoH) trains on sequences such as "How to explain neural networks to a child? Bad: {a subpar answer} Good: {an excellent answer}" built from human preference ratings ([[chain-of-hindsight]] §2).
**Mechanism.**
1. Feedback tokens are masked and model-generated tokens are predicted: "Loss is not applied on other tokens because it hinders model generation at inference time." The training paragraph also says the loss is averaged "over each timestep in the last model output sequence", so whether the earlier, lower-rated answer is itself a target is not fully specified (§2).
2. 0-5% of past tokens are masked at random so the model cannot copy the paired answer.
3. A pretraining log-likelihood term on the Pile is added with weight λ = 1.5 (App. C).
4. At inference the prompt contains only the positive feedback token.

**Evidence (GPT-J 6B and OPT; 75 human labelers).** Summarization average win rate: CoH 45.3 vs RLHF 30.8, and CoH 61.7 vs SFT with unlikelihood 21.4 (Table 1). In the automatic dialogue evaluation on HH, whose metric is the accuracy of picking the preferred response of a pair, adding unlikelihood to SFT lowered accuracy; the authors read this as "unlikelihood hurts model generation ability" (§4, Fig. 4). Zero-shot average on 22 lm-evaluation-harness tasks: GPT-J 40.60 (copied from the GPT-J paper), SFT 40.54, CoH 40.95 (Table 7, 5 seeds). A Pile log-likelihood term was applied to CoH and to all baselines (§2), so this retention cannot be credited to conditioning alone. At smaller model sizes CoH was marginally below SFT (Fig. 5). Result (single study).

### 4.2 Failure-conditioned agent training: NAT

**Definition.** Negative-Aware Training (NAT) appends "Please generate a solution that **correctly** answers the question." to successful trajectories and the same sentence with "**incorrectly**" to failed ones; loss is on model-generated text only, and inference uses the positive sentence ([[learning-from-failure-nat]] §3.3).
**Evidence (LLaMA-2-Chat-7B, 2k positives, 10k negatives, average over GSM8K, ASDiv, SVAMP, MultiArith, Table 2).** Positives only 55.90; negatives added without a label (NUT) 63.39; NAT 64.64. Worked decomposition: of the 8.74-point gain, 7.49 points appear without any label and 1.25 come from the label (derived). At 13B with 5k positives, NUT scored 69.10, below the positive-only 70.76, while NAT scored 71.28.
- The five label pairs tested, including two random strings (64.04), scored between 63.15 and 64.04 (Table 7); the authors conclude that the gain comes from separating the two sets, not from the words.
- On GSM8K, NUT raised the tool-call error rate from 3.58% to 10.47%, and NAT's rate was 7.90% (Table 6).
- Negatives from a fine-tuned LLaMA-2-7B instead of GPT-3.5 left the average 3.16 (2k positives) and 6.20 (5k) points below the positive-only model, while GPT-3.5 negatives added 8.74 and 3.25 (Table 5, §5.2).
- On HotpotQA with 500 positives, NAT scored highest with 500 negatives and lower with 1,000 (§6.1, Fig. 5).

**Conditions and limits.** LLaMA-2-Chat 7B and 13B; GSM8K-derived math, HotpotQA, StrategyQA; ground-truth labels required (§8). Result (single study).

### 4.3 Conditional pretraining with control tokens

**Evidence.** GPT-2-small (124M) pretrained on 3.32B tokens with `<|good|>` or `<|bad|>` before each sentence, set by a toxicity classifier, and sampled after `<|good|>`: average toxicity score 0.0011 vs 0.0141 for maximum likelihood, while slightly exceeding maximum likelihood on LAMBADA and staying closest to it on GLUE among the feedback objectives ([[pretraining-with-human-preferences]] §4.1, §4.3). Unlikelihood on flagged segments gave "very low accuracy" on LAMBADA in the toxicity task and the lowest HumanEval scores in the PEP8 code task, but matched maximum likelihood on LAMBADA in the PII task (§4.3). Using the feedback objective for all of pretraining beat the best feedback objective applied only in a 1.6B-token fine-tune after maximum-likelihood pretraining (PII score 0.0013 vs 0.0018, §5). Result (single study).

**Implication for §4.** CoH and conditional pretraining both kept general benchmark scores at the level of their SFT or maximum-likelihood baselines while using failures (GPT-J 6B fine-tuning; GPT-2-small pretraining). Replicated, for this narrow claim.

## §5 Negatives as bounded gradient inside supervised losses (meaning 4)

### 5.1 Unlikelihood training

**Formula** ([[unlikelihood-training]] Eq. 4):

```
L_UL-token = −α · Σ_{c ∈ C_t} log(1 − p(c | x_<t)) − log p(x_t | x_<t)
```

- C_t: negative candidate tokens at step t (previous context tokens, or tokens of repeated n-grams in the model's own continuation); α: weight; x_t: target token.
- For one candidate k, ∂[−log(1 − p_k)]/∂z_k = p_k and ∂/∂z_j = −p_k·p_j/(1 − p_k) for j ≠ k (derived; the paper's form is App. A Eq. 12).

**Worked example.** With the §2 logits and α = 1, the unlikelihood gradient on D is 0.032 and moves p_D from 0.032 to 0.031; on A it is 0.644 and moves p_A from 0.644 to 0.404. The term is bounded below by 0, and its gradient on z_k equals α·p_k, so the update on the negative shrinks toward zero as the negative becomes unlikely. The push-down in §2 has no lower bound, and its gradient on z_k is 1 − p_k, which approaches 1 as p_k falls.
**Evidence (16-layer Transformer, Wikitext-103, token-level plus sequence-level unlikelihood, Table 2).** Beam search 4-gram repetition .523 → .013 and unique tokens 9.5k → 19.1k (human text .006 and 19.8k), perplexity 25.64 → 26.72. Crowdworkers preferred it over the baseline in 82% of comparisons (Table 3). Result (single study).
**Conditions and limits.** Negatives are repetitions, not wrong answers. With other negatives the results were worse. With low-rated responses as negatives (CoH, HH dialogue), adding unlikelihood to SFT lowered preference-classification accuracy ([[chain-of-hindsight]] §4, Fig. 4). With classifier-flagged pretraining segments as negatives, unlikelihood gave very low LAMBADA accuracy in the toxicity task and the lowest HumanEval scores in the code task, while matching maximum likelihood in the PII task ([[pretraining-with-human-preferences]] §4.3). Replicated across these two sources, with task dependence.

### 5.2 CRINGE

**Formula** ([[cringe-loss]] Eq. 3-5): L = L_CE + α·log(1 + exp(s_neg − s₊)), where L_CE is cross-entropy on positive sequences, s_neg is the logit of the labeled negative token, s₊ the logit of a token sampled from the softmax over the model's top-k tokens excluding the negative, and α weights the negative term. The released code calls `torch.topk(x, k=self.k + 1, axis=-1)`, masks the negative, samples with `Categorical(logits=logits)`, and returns `ce_loss + self.alpha * cr_loss` (App. A.1, Listing 1).
**Worked example.** Negative A (logit 2) against sampled B (logit 1): σ(1) = 0.731, so with η = 1 A's logit falls to 1.269 and B's rises to 1.731; p becomes (0.337, 0.534, 0.095, 0.035). The removed mass goes mainly to the sampled alternative. For negative D against A, σ(−3) = 0.047, a small update.
**Evidence (Table 2-3).** BlenderBot 400M safety on WikiToxic: share of generations that the safety classifier accepts 59.4 (baseline), 86.7 (unlikelihood), 99.9 (iterative CRINGE), with ConvAI2 F1 15.9, 16.5, 16.6. FITS dialogue (2.7B): on unseen topics, single-iteration CRINGE 18.4 F1 and iterative CRINGE 17.8; the authors name overfitting as one possible cause (§4.4). Result (single study).

### 5.3 ORPO's odds-ratio term

**Formula** ([[orpo]] Eq. 6-7): L = L_SFT + λ·(−log σ(log[odds(y_w)/odds(y_l)])), with odds(y) = P(y)/(1 − P(y)). P is the length-normalized sequence probability, y_w and y_l are the chosen and rejected responses, σ is the logistic sigmoid, L_SFT is the negative log-likelihood of y_w, and λ weights the odds-ratio term.
**Worked example.** P(y_w) = 0.5 and P(y_l) = 0.2 give odds 1 and 0.25, log odds ratio 1.386, and L_OR = −log σ(1.386) = 0.223.
**Evidence.** With λ = 1.0 on OPT-350M the rejected log-probability decreases (Fig. 7); on Mistral-7B, λ = 1.0 lowered chosen log-probabilities together with rejected ones, and λ = 0.1 did not lower rejected ones (App. E.1). The positive term L_SFT is always present. Details and the DPO comparison are in [[ch-39]].

### 5.4 NFT: an implicit negative policy

**Definition.** NFT (Negative-aware Fine-Tuning) is an online supervised method: each step samples 16 answers per question, labels them with a verifier, and trains on both sets ([[nft-negative-aware-finetuning]] §3, App. C).
**Mechanism.**
1. The sampling policy splits as r_q·π⁺ + (1 − r_q)·π⁻ = π_old, where r_q is the question's correctness rate (Eq. 7).
2. The negative policy is written through the trained model: π_θ⁻ = (π_old − r_q·π_θ⁺)/(1 − r_q) (Eq. 8). Maximizing the likelihood of π_θ⁻ on wrong answers therefore lowers π_θ⁺ on them; with unlimited data and model capacity, the optimum is π_θ⁺ = π⁺ (Theorem 3.1).
3. Token-level loss (Eq. 10): −ω(q)·[r·log R + (1 − r)·log max_v((1 − r̂_q·R)/(1 − r̂_q), ε)], with R = π_θ⁺(a_t)/π_old(a_t). Here r ∈ {0, 1} is the verifier reward of answer a, r̂_q is the fraction of the K sampled answers that are correct, π_old is the policy that generated the samples, ω(q) is a prompt weight, and max_v clips the value at ε while keeping the gradient (Algorithm 1).

**Worked example.** Let x = (1 − r̂_q·R)/(1 − r̂_q), the argument of max_v; the gradient of the negative term with respect to R is scaled by 1/max(x, ε), called the penalty weight here. With r̂_q = 0.25 and ε = 1.0: R = 1.2 gives x = (1 − 0.3)/0.75 = 0.933, so the penalty weight is 1.0 instead of 1.07; R = 3.6 gives 0.133, weight 1.0 instead of 7.5. The clip caps how strongly a rising wrong answer is penalized.
**Evidence (Table 1, DAPO-Math-17k training; six-benchmark average of avg@32 on AIME24, AIME25, AMC23 and avg@1 on MATH500, OlympiadBench, Minerva).** Qwen2.5-Math-7B: RFT 48.3, NFT 51.7, DAPO 51.2, GRPO 49.5. Qwen2.5-32B: RFT 52.8, NFT 59.2, DAPO 59.9. RFT's entropy (of the model's token distribution on its samples) falls during training while NFT's and DAPO's rise (Fig. 8). With ω(q) = (1 − r̂_q)/r̂_q, ε ≤ 1, and on-policy data (samples from the current model, R = 1), NFT and GRPO gradients are equal (Propositions 4.1-4.2); with the default ω(q) = 1 − r̂_q, NFT aligns with Dr. GRPO (§4, App. A). "overly aggressive penalization with ε → 0 degrades overall performance" (§5.4). Result (single study).

### 5.5 REDI: asymmetric weighting in offline distillation

**Formula** ([[redi-reinforcement-distillation]] Eq. 5): L = −log π(y_w|x)/|y_w| + α·log π(y_l|x)/|y_l|, averaged over pairs. x is the problem, y_w a correct teacher trace, y_l an incorrect trace for the same problem, |y| the token length, and α ∈ [0, 1] the negative weight (α = 0 is SFT on positives).
**Worked example.** The second term has no lower bound: if log π(y_l|x)/|y_l| falls from −1 to −10 per token, the loss falls by 0.8 × 9 = 7.2 at α = 0.8, whatever happens to y_w. Down-weighting slows this decrease, and a single epoch limits how many updates it receives (Interpretation).
**Evidence (Qwen2.5-Math-1.5B, 78k positives then 53k pairs from OpenR1-Math; pass@1 over 16 samples averaged over MATH-500, AIME24, AMC23, Minerva, OlympiadBench).** From the 3-epoch SFT model (41.9 average): DPO 47.2, SimPO 47.2, α = 1.0 at LR 2e−7 47.7, α = 0.8 at LR 1e−6 48.3 (Table 2). α = 1.0 at LR 1e−6 collapsed: "a rapid decrease in the likelihood of both positive (y_w) and negative (y_l) responses, accompanied by declining task accuracy" (§2.2, §4.3). From the 5-epoch SFT model: 45.8 → 49.5, vs 48.6 for DeepSeek-R1-Distill-Qwen-1.5B (Table 1). pass@16 was maintained or improved (App. D.3). Result (single study).

### 5.6 Failure modes of unbounded or unanchored likelihood decrease

| Setting | Observation | Source |
|---|---|---|
| NT: positive log-likelihood plus an unbounded −0.05·log P term on wrong teacher traces, LLaMA-7B MATH | 4.48, below positive-only CoT distillation 5.29 | [[dust-into-gold-negative-distillation]] Table 2 |
| REDI α = 1.0, LR 1e−6 (positive term present) | chosen and rejected log-probabilities collapse, accuracy declines | [[redi-reinforcement-distillation]] §4.3 |
| DPO β = 0.001, LR 1e−6, same data | peak of approximately 80.9% MATH-500, then collapse with a surge in gradient step size | [[redi-reinforcement-distillation]] §4.2 |
| NFT ε → 0 (positive term present) | overall performance degrades | [[nft-negative-aware-finetuning]] §5.4 |
| ORPO with a probability-ratio penalty in place of the odds ratio (SFT term present) | rejected log-probabilities fall below −4 quickly | [[orpo]] App. B |
| DPO on similar refusal pairs | training-set refusal rate 74.4% → 33.4% | [[likelihood-displacement]] §6.2 |

In the first five rows, the decrease of the negative's likelihood was weakly limited (an unbounded term, α = 1.0, β = 0.001, ε → 0, or a probability ratio in place of the odds ratio); in the sixth, the preferred and dispreferred responses were similar. A positive log-likelihood term was present in four of the six rows, so a positive term alone did not prevent the failure. Replicated across five sources, in different settings.

## §6 How large is the gain from negatives

| Source and setting | Without negatives | With negatives | Share attributable to negatives |
|---|---|---|---|
| NFT, Qwen2.5-32B (base 29.6) | RFT 52.8 | NFT 59.2 | 20% stated (§5.3); from Table 1, 6.4/29.6 = 21.6% (derived) |
| NFT, Qwen2.5-Math-7B (base 31.6) | RFT 48.3 | NFT 51.7 | 3.4/20.1 = 16.9% (derived) |
| NAT, LLaMA-2-Chat-7B, 2k positives | 55.90 | NAT 64.64 | 8.74 points; 7.49 without a label |
| LEMA, LLaMA-2-70B GSM8K, 5.8M tokens | 82.1 | 83.5 | 1.4 points at equal tokens |
| REDI, Qwen2.5-Math-1.5B | SFT-3ep 41.9 | 48.3 | not isolated: stage 2 also trains on 53k chosen traces |
| Likra, Mistral-7B-v0.1 ARC | SFT .6630 | .8123 | scoring-level, multiple choice only |

Two cautions. NFT's RFT baseline also uses a constant prompt weight ω(q) = 1 while NFT uses 1 − r_q (App. C), so the 20% includes a weighting difference. Likra's statement that each negative can help "10× more than each additional positive example" holds during a phase of 64-256 examples, on multiple-choice scoring with a separate negative head that cannot generate ([[negative-examples-likra]] §1, §3.2). The head is trained with ordinary likelihood on wrong answers (meaning 2), and the result was not tested for generation.

## §7 Loss masks for failed attempts kept in context

| Format | Where the failure sits | Loss on failure tokens | Evidence |
|---|---|---|---|
| Correction after a wrong solution | Input | No | [[lema-learning-from-mistakes]] App. B.2 |
| Critique of a noisy response | Input | No | [[critique-fine-tuning]] §2.3 |
| Hindsight feedback phrases | Input and target sequence | No on feedback tokens | [[chain-of-hindsight]] §2 |
| Inline retry, pretraining from scratch | Target | Masking unnecessary at q ≤ 0.5 | [[physics-of-lm-2-2-learning-from-mistakes]] Result 2-3 |
| Inline retry, LoRA fine-tune of an error-free model | Target | Masking important at small ranks | same, Result 7 |
| Failure-conditioned trajectory | Target under a negative label | Yes, on model-generated tokens; tool observations masked | [[learning-from-failure-nat]] §3.3 |

Rule: when a pretrained model is fine-tuned on self-correction traces, put failed attempts in the input or mask their tokens, because unmasked errors hurt small-rank LoRA fine-tuning in the only controlled test. When retry data is part of pretraining, masking is not needed at retry rates up to 0.5.

## Negative samples and negative feedback

This section applies the course standard to the SFT stage; meanings (1)-(4) refer to the §1 table.
1. **Where negatives come from.** Final-answer checks (LEMA, NAT, Dust into Gold), a verifier on self-samples with truncated answers counted as negative (NFT App. C), two judges that must both say correct (REDI §3.1: a trace counts as negative when either the Llama judge or Math-Verify does not label it correct), a GPT-4o judge (CFT, about 20% critique errors in 50 inspected), a classifier (conditional pretraining; CRINGE's iterative labeling), and human labels (CoH ratings; FITS binary feedback in CRINGE). A false-negative rate is not reported by any of these sources.
2. **What current practice does.** Distillation and RFT pipelines discard failures (meaning 1). The alternatives are content (§3), conditioning (§4), and bounded gradient (§5).
3. **Mechanism.** ∂ log p_y/∂z_j = 1[j = y] − p_j. SFT already lowers wrong tokens in proportion to their probability; a push-down raises every other logit by η·p_j, so on an unlikely negative it concentrates mass on the most likely token (§2).
4. **Evidence.** Benefits: §3-§5. Failure modes: §5.6, and MIX 3.03 vs 5.29 for positive-only distillation ([[dust-into-gold-negative-distillation]] Table 2).
5. **Controls.** Keep a positive log-likelihood term (REDI, ORPO, CRINGE, unlikelihood); bound the negative (unlikelihood's p_k-scaled gradient, NFT ε = 1.0, REDI α = 0.8); keep negatives on-policy or from a strong generator (NFT; NAT Table 5); prefer near-miss negatives ([[negative-examples-likra]] §4.3); move failures to the input when a correction exists (§7).
6. **Diagnostics.** Log chosen and rejected log-probabilities separately (REDI §4.2); entropy over training (NFT Fig. 8); pass@1 and pass@k at k = 16 or higher (REDI App. D.3); action or format error rate (NAT Table 6); retries per problem (Physics 2.2 Result 4).
7. **Effect on generality.** Coverage: pass@16 kept (REDI), entropy rises (NFT). Calibration and hallucination: not measured by these sources. Refusal behavior: in DPO refusal training, displacement lowered refusal rates ([[likelihood-displacement]]); refusal and over-refusal were not measured for SFT negatives in these sources. Forgetting: CoH and conditional pretraining kept general benchmark scores (§4); unlikelihood lowered them in two of the three pretraining-with-feedback tasks (§5.1).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLaMA-2-Chat 7B (NAT, math) | 7B | SFT | epochs; batch; peak LR; schedule; hardware | 2; 64; 5×10⁻⁵; cosine, 3% warm-up; 4×A100, DeepSpeed ZeRO-3 | arXiv:2402.11651v2 §4.1 | verified 2026-09-15 | no ablation reported |
| LLaMA-2-Chat 7B (NAT, math) | 7B | SFT | data; label; loss mask | 2k or 5k GPT-3.5 positives + 10k negatives; "correctly"/"incorrectly" suffix; loss on model-generated text | §3.3; App. B | verified 2026-09-15 | Fig. 2: 0-12k negatives, plateau ≈11k; Table 7: label wording |
| LLaMA-2-70B (LEMA) | 70B | SFT (QLoRA) | rank; dropout; LR; batch; steps | 64; 0.05; 1e−4; 96; 2,000 | arXiv:2310.20689v4 §3.3 | verified 2026-09-15 | no ablation reported |
| LLaMA-2 7B-70B (LEMA) | 7B-70B | SFT | correction pairs; loss mask | GSM8K 12,523, MATH 6,306, CSQA 7,241; output tokens only | §3.2; App. B.2 | verified 2026-09-15 | Fig. 4 and Table 2: equal examples and equal tokens |
| Qwen2.5-Math-7B (CFT) | 7B | SFT | data; epochs; LR; schedule; global batch; selection | 50K WebInstruct-CFT; 1; 5e−6; cosine, warm-up ratio 0.1; 512; best on MATH-500 | arXiv:2501.17703v4 §2.1, §3.1 | verified 2026-09-15 | Tables 5-7, 9-10: data source, teacher, mixing, length |
| GPT-J 6B (CoH) | 6B | SFT | optimizer; batch; pretraining-loss weight; past-token mask | Adam (0.9, 0.95, 1e−8); batch size 512 for feedback data and 2048 for pretraining data (unit not stated); λ = 1.5 on the Pile; 0-5% | arXiv:2302.02676v8 §2; App. C | verified 2026-09-15 | no ablation reported |
| 16-layer Transformer (unlikelihood) | d_model 1024 | pretrain-stable; SFT | sequence length; token-level updates; sequence-level updates; share of sequence updates | 1,536; ≤150k on 8 GPUs; 1,500; 0.5 | arXiv:1908.04319v2 §6 | verified 2026-09-15 | Table 2: token vs sequence variants |
| 16-layer Transformer (unlikelihood) | d_model 1024 | pretrain-stable | α | not reported | checked §5-6, App. A-D | not reported | — |
| BlenderBot 2 (CRINGE, FITS) | 2.7B | SFT | α; k; iterations; batch; LR; warm-up; clip; max steps | 0.5; 5; 1 or 2; 16; range 5e−6 to 5e−5; 100; 0.1; 8,000 | arXiv:2211.05826v1 App. Table 9 | verified 2026-09-15 | Table 3: single vs iterative; Fig. 4: top-3 configurations |
| GPT-2-small (conditional pretraining, toxicity) | 124M | pretrain-stable | tokens; LR; batch; threshold t | 3.32B; 5·10⁻⁴; 64; 5.6·10⁻⁴ | arXiv:2302.08582v2 §3.3; App. A Table 1 | verified 2026-09-15 | App. A Fig. 8: threshold ablation |
| GPT2-12-12 (retry data, iGSM-med) | GPT2-small | pretrain-stable | retry rate; LR; weight decay; batch; context; steps | 0.01-0.5; 0.002; 0.05; 512; 768; 100,000 | arXiv:2408.16293v1 §4; App. D.1 | verified 2026-09-15 | Fig. 4b: retry rates at equal tokens |
| Qwen2.5-Math-7B (NFT) | 7B | RL (online, supervised loss) | data; LR; prompts × samples; updates; ε; ω(q); context; temperature; GPUs | DAPO-Math-17k; 1e−6, linear warm-up; 512 × 16 per rollout, 16 gradient steps each, 320 rollouts; 1.0; 1 − r_q; 4K; 1.0; 64 H100 | arXiv:2505.18116v3 §5.1; App. C | verified 2026-09-15 | Fig. 10: ε; Fig. 9: ω(q) |
| Qwen2.5-32B (NFT) | 32B | RL (online, supervised loss) | context; GPUs | 16K; 128-256 H100 | App. C | verified 2026-09-15 | App. C: 16K "does not significantly affect performance" vs 32K |
| Qwen2.5-Math-1.5B (REDI stage 1) | 1.5B | distill-SFT | data; LR; schedule; batch; epochs; max length | 78k positives; 5e−5; 10% linear warm-up, linear decay; 128; 3 (ablations), 5 (final); 32,768 | arXiv:2505.24850v2 §3.1-3.2; App. C.3 | verified 2026-09-15 | Fig. 2: accuracy over 5 epochs |
| Qwen2.5-Math-1.5B (REDI stage 2) | 1.5B | distill-SFT | data; α; LR; batch; epochs; compute of final model (stage 1 at 5 epochs + stage 2) | 53k pairs; 0.8; 1e−6; 32; 1; about 136 A100-80GB GPU hours (17 hours on one 8-GPU node) | §3.1; App. C.4 | verified 2026-09-15 | Table 2 and §4.4: α ∈ {0.2, 0.5, 0.8, 1.0} |
| Mistral-ORPO-β | 7B | preference | λ; peak LR | 0.1; 8e−6 | arXiv:2403.07691v2 §6.1; App. C ([[orpo]]) | verified 2026-09-14 | App. E: λ ∈ {0.1, 0.5, 1.0} |

Units: "prompts × samples" counts questions per rollout step times answers per question; NFT's 320 rollouts at 16 gradient steps each give over 5,000 gradient steps (App. C).

**Starting point for a small general-purpose run.** When a 1.5B model is distilled on verifier-labeled math traces and wrong traces from the same strong teacher are available, the REDI setting is a documented start: 5 epochs of SFT on positives (LR 5e−5, batch 128), then one epoch on pairs with α = 0.8, LR 1e−6, batch 32 (Qwen2.5-Math-1.5B, 78k positive traces and 53k pairs from OpenR1-Math, about 136 A100-80GB GPU hours for both stages). When negatives are the model's own online samples, NFT used ε = 1.0, ω(q) = 1 − r_q, LR 1e−6, and 16 samples per question for Qwen2.5-Math-7B on DAPO-Math-17k (64 H100, 4K context). When failed agent trajectories carry only a success label, NAT used the suffix format with 2 epochs, LR 5×10⁻⁵, batch 64 for LLaMA-2-Chat-7B with 2k-5k positives and 10k negatives (4×A100). None of these values was tested inside a multi-domain SFT mixture.

## Generalization lens

**(a) What increases breadth.**
- Critique targets on STEM questions raised general instruction-following scores: Qwen2.5-Math-7B IFEval loose 0.330 (SFT) → 0.362 and MT-Bench 5.23 → 6.49 ([[critique-fine-tuning]] Table 8). Result (single study).
- Negative terms kept sampling diversity: token entropy rose under NFT and fell under RFT ([[nft-negative-aware-finetuning]] Fig. 8); REDI kept or raised pass@16 while raising pass@1 ([[redi-reinforcement-distillation]] App. D.3). Each is a single-study result, and the two use different metrics (entropy, pass@16).
- Correction data transferred to tasks not trained directly: SVAMP and ASDiv gains from GSM8K-based LEMA data ([[lema-learning-from-mistakes]] Table 1).
- Conditioning kept general scores: CoH zero-shot average 40.95 vs SFT 40.54 ([[chain-of-hindsight]] Table 7); conditional pretraining was the feedback objective closest to maximum likelihood on GLUE ([[pretraining-with-human-preferences]] §4.3). Replicated for this claim; CoH also trained with a Pile log-likelihood term (§4.1).

**(b) What causes narrowing or forgetting.**
- Failures as plain targets: MIX 3.03 vs 5.29 for positive-only distillation ([[dust-into-gold-negative-distillation]] Table 2); SFT on over-50%-wrong responses lowered Qwen2.5-7B from 37.4 to 20.2 ([[critique-fine-tuning]] Table 2); negatives from a weaker generator left the NAT average 3.16 points below positives only ([[learning-from-failure-nat]] Table 5). Replicated across these sources, but not universal: unlabeled GPT-3.5 failures raised the NAT math average in three of four settings (NUT, Table 2).
- Unlikelihood on rated or classifier-flagged negatives lowered HH preference-classification accuracy ([[chain-of-hindsight]] §4, Fig. 4) and LAMBADA or HumanEval scores in two of three pretraining tasks ([[pretraining-with-human-preferences]] §4.3). Replicated, with task dependence.
- Iterative negative training can fit seen topics: CRINGE F1 on unseen topics 18.4 → 17.8 after the second iteration ([[cringe-loss]] Table 3).
- Unbounded push-down: §5.6.
- Stage mismatch: retry data in a LoRA fine-tune gave no significant gain and hurt at small ranks ([[physics-of-lm-2-2-learning-from-mistakes]] Result 7).

**(c) How to measure it at this stage.**
- Compare against a positive-only arm at equal steps and equal tokens (LEMA Table 2).
- Track chosen and rejected log-probabilities, entropy, and pass@k at large k (item 6 of the negative-feedback section).
- Report held-out domains and unseen splits: GPQA and HumanEval (REDI Table 5), test-unseen topics (CRINGE Table 3), IFEval and MT-Bench (CFT Table 8).
- Known measurement errors in these sources: best-of-20 checkpoint selection on the test set (LEMA §3.3); AIME24 with 30 questions (CFT §3.4); an out-of-domain table whose REDI checkpoint is unclear (REDI Table 5 names the final Qwen-REDI-1.5B, initialized from the 5-epoch SFT model per App. C.4, but its OlympiadBench value 43.4 equals the 3-epoch-initialized REDI in Table 2, while Table 1 gives 45.2 for Qwen-REDI-1.5B); best checkpoint per configuration in the REDI ablation table (REDI Table 2); absolute accuracies of 3-7% with no seeds or variance reported (Dust into Gold). None of the fourteen cited sources reports a contamination check (full-text search for "contamination").

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Adding failed samples as ordinary targets | Accuracy below the positive-only arm (Dust into Gold MIX; NAT 13B with 5k positives), or higher accuracy together with more tool or format errors (3.58% → 10.47% in NAT 7B) | Train positive-only and mixed arms at equal steps; count format and tool-call errors |
| Symmetric or unclipped negative term | Chosen and rejected log-probabilities fall together; accuracy peaks then drops; gradient norm rises | Log both log-probabilities and the gradient norm each step; keep early checkpoints |
| Unmasked failed attempts in a LoRA self-correction fine-tune | Accuracy below the original model at small ranks | Run masked vs unmasked arms; count retries per problem |
| Failure label left in, or positive label missing, at inference | No gain over unlabeled mixing | Evaluate with and without the positive label; set the control tokens' probabilities to zero at decoding, as conditional pretraining did for toxicity and PII (App. A) |
| Negatives generated by a weaker model than the positives | Lower average than positives only | Ablate negatives by generator; prefer near-miss negatives |
| Unaudited judge or critique labels | Gains vanish on re-labeled data | Inspect at least 50 labels as CFT did; report the error rate |
| Crediting the whole gain to negatives | Removing the negative term leaves most of the gain | Run an α = 0 or RFT arm with the same prompt weighting |
| Expecting positive-only SFT to suppress plausible wrong answers | Rejected log-probability rises with chosen | Track log-probabilities of held-out rejected responses during SFT |

## Check your understanding

1. In §2, a push-down on D raised p_A from 0.644 to 0.751, while a push-down on A spread mass to B, C, and D. Explain both results from ∂ log p_k/∂z_j = 1[j = k] − p_j, and explain why the unlikelihood term on D changed p_D by only 0.0015.
2. LEMA and CFT both keep every gradient positive. Why does that rule out likelihood displacement, and what does LEMA's equal-token comparison (82.1 vs 83.5) show that an equal-example comparison could not?
3. Retry data needed no mask in pretraining but benefited from masking in small-rank LoRA fine-tuning. Give a causal account that uses the retry counts in Result 4 and the stage finding in Result 7.
4. NAT scored 64.04 with random strings as labels. What does the model learn from the label, and why would the gain disappear if inference used no label?
5. With the GRPO-matched prompt weight, NFT's gradient equals GRPO's on-policy. Why is ε still needed, and what happens to the negative weight at r̂_q = 0.25 when R rises from 1.2 to 3.6 with and without the clip?
6. REDI's negative term has no lower bound. Explain how α = 0.8, one epoch, and the positive term together prevent the collapse observed at α = 1.0 with LR 1e−6.
7. NFT attributes 20% of its 32B gain to negatives, while Likra reports up to 10× more gain per negative example. Name the differences in setting that make both results compatible.
8. An SFT mixture for a general assistant adds 20% failed tool-use trajectories. Which three measurements from this chapter would detect narrowing first, and why?

## Connections

- Previous: [[ch-31]] — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation. Positive-only filtering is the baseline that this chapter extends.
- Next: [[ch-33]] — Case Studies A: How Tülu 3 and Llama 3 Measured and Protected Broad Capability.
- Later: [[ch-39]] — Offline Preference Optimization: DPO and Its Variants (the rejected term with a reference model); [[ch-43a]] — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages (depends on this chapter).

## Sources

- [[unlikelihood-training]] — unlikelihood objective, gradient, Wikitext-103 results.
- [[cringe-loss]] — top-k contrastive loss for negative tokens, code, safety and FITS results, overfitting note.
- [[chain-of-hindsight]] — feedback-conditioned sequences, loss mask, human evaluation, alignment-tax table.
- [[lema-learning-from-mistakes]] — correction pairs with the wrong solution in the input; equal-token comparison.
- [[physics-of-lm-2-2-learning-from-mistakes]] — retry data, masking, pretraining vs LoRA fine-tuning.
- [[critique-fine-tuning]] — critique targets, instruction-following transfer, critique error rate.
- [[dust-into-gold-negative-distillation]] — complementarity of wrong teacher traces; MIX, NT, and unlikelihood baselines.
- [[learning-from-failure-nat]] — failure-conditioned agent trajectories, label ablation, negative quality.
- [[orpo]] — positive-only SFT raises rejected log-probability; odds-ratio term.
- [[nft-negative-aware-finetuning]] — implicit negative policy, ε clip, entropy, 80/20 attribution.
- [[redi-reinforcement-distillation]] — asymmetric α, collapse, pass@16, recipe.
- [[negative-examples-likra]] — per-example effect of negatives in multiple-choice scoring; near-miss negatives.
- [[pretraining-with-human-preferences]] — conditional pretraining vs unlikelihood vs filtering.
- [[likelihood-displacement]] — shared-parameter side effect of push-down, carried to [[ch-43a]].
