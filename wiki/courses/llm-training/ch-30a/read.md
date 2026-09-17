<!-- chapter: ch-30a
     track: sft
     kind: content
     title: Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control
     deps: [ch-30]
     sources: [[catastrophic-forgetting-continual-finetuning]], [[scaling-laws-forgetting]], [[forgetting-via-implicit-inference]], [[spurious-forgetting]], [[finetuning-new-knowledge-hallucination]], [[finetuning-compromises-safety]], [[shallow-safety-alignment]], [[lora-learns-less-forgets-less]], [[sdft-self-distillation-finetuning]], [[wise-ft]], [[mitigating-alignment-tax-rlhf]], [[retaining-by-doing]], [[rls-razor]], [[rlhf-instructgpt]], [[rlhf-instructgpt-ptx]], [[adding-error-bars-evals]], [[agentic-finetuning-misalignment]], [[xstest]]
     figures: figures/paired-forgetting-delta.html
     revised: 2026-09 (generality revision)
-->

# Chapter 30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control

> **Core insight.** In each forgetting study cited in this chapter, supervised fine-tuning that raised a target score also lowered at least one ability the target did not cover: MMLU, reasoning, and reading-comprehension accuracy during continual instruction tuning of 1B-7B models ([[catastrophic-forgetting-continual-finetuning]]), refusal of harmful requests after one epoch on benign Alpaca data (GPT-3.5 Turbo harmfulness rate 5.5% → 31.8%, [[finetuning-compromises-safety]]), and closed-book factual accuracy after fitting facts the model did not know ([[finetuning-new-knowledge-hallucination]]). In a LoRA study on Llama 2 7B chat, forgetting was a linear function of the fine-tuning loss (R² 0.9450 and 0.9736 on two datasets), so early stopping and adapter size moved runs along one trade-off line instead of off it ([[scaling-laws-forgetting]]). The controls with measured benefit keep the fine-tuned model's output distribution close to its starting point: general-data replay, LoRA, self-generated targets, weight interpolation toward the starting weights, pretraining-loss mixing, per-token constraints on early response tokens, and on-policy training data (the shared property is this course's Interpretation; each control's evidence is in §4-§5).
>
> **Guideline.** When a fine-tuning stage is evaluated, compare its output checkpoint with its input checkpoint item by item on held-out suites that the stage's data does not target, and report the paired 95% confidence interval of the difference, because two checkpoints with similar aggregate accuracy can disagree on a third of the items ([[scaling-laws-forgetting]] §3.1) and the paired interval removes variance shared by both checkpoints ([[adding-error-bars-evals]] §4.2). When the starting model has safety behavior, re-run harmful-request and over-refusal evaluations after every stage, including stages trained on benign data, because every benign dataset tested by [[finetuning-compromises-safety]] raised the harmfulness rate (Table 3). When the target skill can be reached with LoRA at sufficient rank or with on-policy samples, prefer these to full-parameter SFT on external responses at a high learning rate, because in code instruction tuning at matched HumanEval accuracy LoRA forgot less ([[lora-learns-less-forgets-less]] Tables S5-S6) and GRPO's mean non-target drops were at most 3.4 points where SFT's reached 38.5 points ([[retaining-by-doing]] Table 1). Otherwise, keep full fine-tuning, add general-data replay or self-distilled targets, and choose the operating point from a measured gain-versus-drop curve.

## Why this chapter matters for a general-purpose model

A general-purpose model is built in stages: pre-training → mid-training → SFT → preference optimization → RL → evaluation. Every stage after pre-training trains on data that is narrower than the pre-training corpus. **Forgetting** is a decrease, after a training stage, in performance on an ability that the model had before that stage. **Alignment tax** is forgetting caused by an alignment procedure. InstructGPT uses the term for its RLHF models' regressions relative to GPT-3 on SQuAD, DROP, HellaSwag, and WMT 2015 French-to-English ([[rlhf-instructgpt-ptx]], §1).

The measurable problem has two parts. First, a stage's own validation set does not show the loss: in [[wise-ft]] (CLIP ViT-B/16 fine-tuned for 10 epochs), learning rates 3·10⁻⁵ and 3·10⁻⁶ gave ImageNet accuracies 0.3 points apart while accuracy under distribution shift differed by up to 8 points (§4, Fig. 3). Second, the loss is spread across capabilities that the stage never evaluates: knowledge, reasoning, safety, calibration, and output diversity.

ch-00 defines general capability as a profile over tasks that were not targeted. ch-30 set the SFT design choices. This chapter measures what those choices cost on the rest of the profile and which controls lower that cost. Mixture design is in ch-30b, weight merging in ch-30c, continued pretraining in ch-32a, KL-controlled RLHF in ch-38, and the SFT-versus-RL comparison in ch-38a. The report format in §6 is used by the labs ch-36, ch-46, and ch-53.

## §1 Measuring forgetting against the starting checkpoint

**Definition.** A forgetting measurement compares a stage's input checkpoint (the reference) with its output checkpoint on evaluations whose abilities the stage did not target. A **held-out suite** is a set of evaluations that is not represented in the stage's training data and was not used to choose the stage's settings.

**Problem.** An aggregate score difference hides item-level change. [[scaling-laws-forgetting]] (§3.1) fine-tuned Llama 2 chat 7B on 6400 OpenOrca examples (200 steps) and evaluated ARC-Challenge: the base model scored 54.1%, the fine-tuned model 52.0%, and the fine-tuned model gave the base model's answer on only 67.2% of items. With answer agreement a = 0.672, the share of items whose answer changed is 1 − a = 0.328. An answer change is a loss (base correct, fine-tuned wrong), a gain (base wrong, fine-tuned correct), or a switch between two wrong options. With a net change of −0.021 and no wrong-to-wrong switches, the losses would be (0.328 + 0.021)/2 = 17.45% of items and the gains (0.328 − 0.021)/2 = 15.35%; every wrong-to-wrong switch lowers both shares, so these are upper bounds (derived; the paper does not report the split). The 2.1-point net drop can therefore be the difference between a loss share and a gain share that are each several times larger than 2.1 points.

**Mechanism.**
1. Fix the reference: the checkpoint that entered the stage. Also report against the original pretrained base when several stages are chained.
2. Choose held-out suites per capability slice (knowledge, reasoning and math, code, instruction following, long context, multilingual, tool use, factuality, safety, over-refusal). Remove items that overlap the training data (ch-53).
3. Run both checkpoints with the same prompt template, few-shot exemplars, decoding settings, and judge version.
4. Store the score of every item for both checkpoints.
5. Compute the mean difference, the paired standard error, the 95% interval, and the counts of gained and lost items.
6. Add one distribution-level measure: cross-entropy of the fine-tuned model to the reference model's predictions ([[scaling-laws-forgetting]] §3.1) or KL divergence to the reference on a fixed prompt set.

**Formulas.** For item i of n items, let s_ref,i and s_ft,i be the two scores and d_i = s_ft,i − s_ref,i.

```
Δ = d̄ = (1/n) Σ_i d_i
SE_paired   = sqrt( (1/(n−1)) Σ_i (d_i − d̄)² / n )           (Eq. 7)
CI_95%      = Δ ± 1.96 × SE                                  (Eq. 5)
SE_unpaired = sqrt( p_ref(1 − p_ref)/n + p_ft(1 − p_ft)/n )   (Eq. 2 in SE_A−B of §4.1)
```

The equation numbers refer to [[adding-error-bars-evals]]. Here Δ is the forgetting delta (negative means a loss), SE is a standard error, p_ref and p_ft are the two accuracies for 0/1 scores, and 1.96 is the two-sided 95% normal quantile. The source's author (Miller) writes the variance relation Var(paired) = Var(unpaired) − 2 Cov(x_A, x_B)/n: the paired interval is narrower when the two checkpoints agree on which items are hard (§4.2).

The same computation from stored per-item scores, as used in the §6 report (course code, not taken from a source):

```python
import math

def paired_delta(ref_scores, ft_scores, z=1.96):
    """ref_scores, ft_scores: per-item scores in the same item order."""
    n = len(ref_scores)
    d = [f - r for r, f in zip(ref_scores, ft_scores)]
    mean_d = sum(d) / n
    var_d = sum((x - mean_d) ** 2 for x in d) / (n - 1)   # sample variance of d_i
    se = math.sqrt(var_d / n)                             # Eq. 7
    losses = sum(1 for x in d if x < 0)
    gains = sum(1 for x in d if x > 0)
    return mean_d, se, (mean_d - z * se, mean_d + z * se), losses, gains
```

**Worked example.** A held-out suite has n = 500 items with 0/1 scores. The reference answers 300 correctly (p_ref = 0.60). After SFT, 45 of those 300 become wrong (losses) and 20 of the 200 wrong items become correct (gains), so the fine-tuned model answers 275 correctly (p_ft = 0.55).

| | ft correct | ft wrong |
|---|---|---|
| ref correct | 255 | 45 |
| ref wrong | 20 | 180 |

- Δ = (20 − 45)/500 = −0.050.
- Each d_i is −1 (45 items), +1 (20 items), or 0 (435 items). Σ d_i² = 65, so Σ (d_i − d̄)² = 65 − 500 × 0.05² = 63.75 and the sample variance is 63.75/499 = 0.12776.
- SE_paired = sqrt(0.12776/500) = 0.01598; CI = −0.050 ± 0.0313 = [−0.081, −0.019]. The interval excludes zero.
- SE_unpaired = sqrt(0.60 × 0.40/500 + 0.55 × 0.45/500) = 0.03122; CI = −0.050 ± 0.0612 = [−0.111, +0.011]. The interval includes zero.

The same data support a detected loss under the paired analysis and no detected loss under the unpaired analysis. If the same net change came from 95 losses and 70 gains, Σ d_i² = 165 and SE_paired = 0.0256, so the paired interval would widen to [−0.100, +0.0002], which includes zero. With four times as many items at the first example's rates (n = 2000, 1200 correct, 180 losses, 80 gains), SE_paired = 0.0080 and SE_unpaired = 0.0156, and both intervals exclude zero. The interactive figure [paired-forgetting-delta.html](figures/paired-forgetting-delta.html) lets the reader change n, the reference accuracy, and the loss and gain counts and see both intervals and the agreement rate update.

**Conditions and limits.** The paired formula requires the same items, scored the same way, for both checkpoints. When outputs are sampled, average several samples per item before computing d_i ([[adding-error-bars-evals]] §3.1). When items share a passage or a template, the clustered standard error was 3.05 times the naive one on DROP (§2.2, Table 4). With k slices tested at the 5% level and no real change, the expected number of intervals that exclude zero is 0.05 × k; ch-51 teaches multiple-comparison control and ch-53 applies it. When the number of changed items is small, the normal approximation behind the 1.96 multiplier is less accurate.

**Implication.** A forgetting report states a paired interval per slice, not a single aggregate difference.

## §2 Evidence: forgetting in continual instruction tuning and forgetting laws

### 2.1 Continual instruction tuning of 1B-7B models

[[catastrophic-forgetting-continual-finetuning]] trained BLOOMZ (1.1B-7.1B), mT0 (1.2B, 3.7B), LLaMA-7B, and Alpaca-7B on five generation tasks in sequence (Simp → Emdg → InqQG → Exp → HGen), LR 2e-5, 3 epochs per task, and evaluated MMLU, six reasoning sets, and RACE after each task (§3-§4). The forgetting metric (Eq. 1) averages relative drops over all N checkpoints:

    FG_i = (1/|E_i|) Σ_{e ∈ E_i} (1/N) Σ_{m=1..N} (R_o^e − R_m^e) / R_o^e × 100%

R_o^e is the initial result on evaluation element e, R_m^e the result after the m-th task, and E_i the elements of one evaluation set.

**Worked example.** One element starts at R_o = 40 and scores 38, 35, and 30 after three tasks. FG = (2/40 + 5/40 + 10/40)/3 = (0.050 + 0.125 + 0.250)/3 = 14.2%. The final-only relative drop is 25%. For BLOOMZ-7.1b domain knowledge, Table 4 gives 33.08 → 25.61, a final-only relative drop of 22.6% (derived), while the reported FG is 18.37, because FG averages the relative drops over the intermediate checkpoints and over the four MMLU elements.

**Evidence (Result, single study).** FG was above zero for every model on domain knowledge, reasoning, and reading comprehension (Tables 4 and 6). Reading comprehension lost the most: BLOOMZ-7.1b FG 26.75 (reading), 18.37 (domain knowledge), 13.62 (reasoning) (Table 4). Domain-knowledge FG rose with size: 9.54, 10.72, 14.63, 18.37 for BLOOMZ-1.1b, 1.7b, 3b, 7.1b (§5.2). The authors attribute the size trend to higher initial scores with similar final scores (Interpretation). LLaMA-7b had domain-knowledge FG 34.57 and Alpaca-7b 18.14 (Table 6), which the authors read as general instruction tuning reducing later forgetting.

**Conditions and limits.** One task order, no model above 7.1B, a constant or cosine LR fixed from prior work (§4.2 footnote 7), and 512-token inputs. The size trend is not tested beyond 7B.

### 2.2 Forgetting grows with training duration

[[lora-learns-less-forgets-less]] measures forgetting with a **forgetting average**: the mean accuracy on HellaSwag, ARC-Challenge, and WinoGrande, where a higher value means less forgetting (§3.3). Full fine-tuning of Llama-2-7B on Magicoder-Evol-Instruct-110K lowered this average from 0.595 after epoch 1 to 0.414 after epoch 16 (Table S6). The authors report that instruction fine-tuning forgets more than continued pretraining, code forgets more than math, and forgetting grows with duration (§4.2).

### 2.3 Forgetting as a function of fine-tuning loss

[[scaling-laws-forgetting]] fine-tuned Llama 2 7B chat with LoRA at ranks 8-256 for 260 steps (batch 32, context 512) on OpenOrca or on 100 news articles from September 2023, and measured forgetting L_f as cross-entropy to the base model's next-token predictions on WikiText-103 (§3). The fit (Eq. 4) is:

    L_f(L_ft) = −c · L_ft + s

L_ft is the fine-tuning loss, c and s are fitted constants: c ≈ 1.7334, s ≈ 2.0481 (OpenOrca) and c ≈ 1.0615, s ≈ 3.1285 (News), with R² 0.9450 and 0.9736 (§4.1, Fig. 2).

**Worked example.** On OpenOrca, lowering L_ft from 0.76 to 0.66 predicts L_f = −1.7334 × 0.76 + 2.0481 = 0.731 before and −1.7334 × 0.66 + 2.0481 = 0.904 after. Among the fitted LoRA runs (ranks 8-256), each 0.1 decrease in fine-tuning loss adds 0.173 to the forgetting loss, whatever rank or step count produced the decrease.

Both losses also follow shifted power laws in the number of trained parameters P and steps N (Eq. 5-6); fitting L_f as a function of P and N raises R² to 0.9598 and 0.9769. The author concludes that forgetting "is unavoidable by early stopping or by tuning a fewer (or greater) number of parameters" with conventional fine-tuning (Interpretation), and attributes the larger forgetting of larger models in [[catastrophic-forgetting-continual-finetuning]] to their lower fine-tuning loss (Interpretation, not tested across sizes). Limits: one 7B model, one forgetting dataset, 260 steps. The linear law transferred only partly to other methods on OpenOrca: both fits predicted LoRA ranks 1000 and 2500, attention-only rank-64 LoRA, and full fine-tuning, but L_f(L_ft) underestimated forgetting for IA3 and for tuning only the top 3 layers, giving R² 0.1851 over these runs against 0.8714 for L_f(P, N) (App. B, Fig. 6).

**Implication.** Within one method and dataset, how far the target is fit sets how much is forgotten. Changing the trade-off requires changing the method or the data, which §5 covers.

## §3 Mechanisms: capability loss, task inference, and alignment loss

A score drop has at least three explanations, and each predicts a different test result.

### 3.1 Task inference shifts toward the fine-tuning distribution

**Definition.** [[forgetting-via-implicit-inference]] models the output as a mixture: w_θ(X, y) = g_θ(X, y) w_disc(X, y) + (1 − g_θ(X, y)) w_cont(X, y) (Eq. 7), where g_θ is the model's implicit estimate that the prompt belongs to the fine-tuning task family, and w_disc and w_cont are two capabilities. Fine-tuning can raise g_θ on prompts from other tasks without removing w_cont.

**Evidence.** In a 22.4M-parameter transformer trained on synthetic regression, 400 fine-tuning steps raised the loss on the untargeted task; the most affected tasks were those outside but close to the fine-tuning distribution (§1, Fig. 5). Rescaling the labels so that prompts look less like the fine-tuning tasks recovered part of the lost solution (Fig. 6). In LLMs, instruction tuning cut in-context-learning behavior from 92.00% (LLaMA) to 35.25% (Alpaca) on English prompts, but only by 29.00 points in French and 1.50 points in Leetspeak (Table 1). Code LLaMA lost 8.36 points of XNLI accuracy relative to LLaMA-2 in English and gained 0.77 in Spanish (Table 2). Status: **Interpretation** supported by these recoveries; the authors state they cannot test whether models compute this way (§2.7).

### 3.2 Spurious forgetting: task alignment lost, knowledge kept

**Definition.** [[spurious-forgetting]] decomposes "Task Performance = Task Alignment + Underlying Knowledge" (§1). Spurious forgetting is a drop caused by lost alignment to the task format.

**Evidence.** In a synthetic biography dataset with 200,000 individuals, Task 0 accuracy fell from nearly 100% to about 10% within the first 150 steps of training on a new task, while a recovery test (fine-tune on half of Task 0 for one epoch, evaluate the other half) stayed near 100% and ended at 96% (§3.2). LLaMa-2-7B-Chat's AdvBench safety fell from 100% to 0% after 10 identity-shifting examples for 10 epochs, and 10 refusal examples generated by the pre-attack model restored it to about 99% (§2). In the same identity-shifting setting, freezing the bottom 6 layers kept the jailbreak rate at 1.15% against 99.80% for sequential fine-tuning (Table 2); the authors state that this result shows spurious forgetting in safety alignment and is not a defense against jailbreak attacks (§5.3).

**Measurement consequence.** A recovery test separates alignment loss from knowledge loss: after one epoch of fine-tuning on a disjoint half of the affected data, accuracy on the other half returns to its earlier level if only alignment was lost and stays low if the knowledge is gone.

### 3.3 Fitting unknown facts increases hallucination

**Definition.** [[finetuning-new-knowledge-hallucination]] labels each closed-book QA example for PaLM 2-S with SliCK: **Unknown** if neither greedy decoding nor sampling ever produces the answer across 10 random 4-shot prompts, otherwise one of three Known categories (§3, Fig. 2). |D| = 6142, LR 1e-5, batch 128, 50 epochs (App. E).

**Evidence.** Unknown examples were fitted more slowly than Known ones (§4.3). The test accuracy follows Eq. 1:

    Accuracy = β_0 + β_kn · N_kn/|D| + β_unk · N_unk/|D|

N_kn and N_unk are the Known and Unknown training examples the model fits; the in-distribution fit is β_0 = 36.9, β_kn = 7.3, β_unk = −8.3, R² = 0.86; out-of-distribution relations give 36.2, 3.2, −3.0, R² = 0.95 (Table 1).

**Worked example.** A dataset is 50% Known and 50% Unknown. At early stopping the model fits 45% of D as Known and 5% as Unknown: 36.9 + 7.3 × 0.45 − 8.3 × 0.05 = 39.8. At convergence it fits all of both halves: 36.9 + 3.65 − 4.15 = 36.4. The net loss of 3.4 points is the Unknown term (−8.3 × 0.45 = −3.7) partly offset by the additional Known examples fitted (+7.3 × 0.05 = +0.4).

At convergence, training only on Unknown examples scored 25.8 against 37.5 at early stopping (Table 2). Filtering Unknown examples made results at early stopping and at convergence nearly identical (§4.2). Status: **Result (single study)**, one model and one QA format.

**Implication.** Answer targets that assert facts the model cannot produce teach unsupported answering on other questions. For a general model this appears as a calibration and factuality loss that a knowledge benchmark scored on known facts would not show.

## §4 Safety erosion from fine-tuning

### 4.1 Benign fine-tuning lowers refusal

[[finetuning-compromises-safety]] measured the harmfulness rate (share of 330 policy-violating prompts judged fully harmful by GPT-4) after one epoch of benign fine-tuning: GPT-3.5 Turbo Alpaca 5.5% → 31.8%, Dolly 4.5% → 23.9%; Llama-2-7b-Chat Alpaca 0.3% → 16.1%, Dolly 0.6% → 12.1%, LLaVA-Instruct 0% → 18.8% (Table 3). On Llama-2-7b-Chat with Alpaca, LR 5e-5 gave 46.4% / 37.9% / 31.5% / 34.2% at batch 16 / 32 / 64 / 128, against 23.6% / 20.6% / 15.8% / 16.1% at LR 2e-5 (Table 12). LoRA did not protect safety: 25.2% after Alpaca against 16.1% for full-parameter tuning, with each method at its officially recommended settings (LoRA LR 10⁻⁴, batch 16; full LR 2×10⁻⁵, batch 128) (Table 11, App. F). Mixing 500 refusal examples into GPT-3.5 Turbo's Alpaca run reduced 31.8% to 19.7%, not to the initial 5.5% (Table 4). MT-Bench fell from 8.00 to 6.68 after Alpaca (Table 8). [[scaling-laws-forgetting]] found the same direction on 50 hand-checked AdvBench prompts: the base model refused 32, the News LoRA model 24, the OpenOrca model 16 (§4.2). Status: **Replicated** in direction across these two sources.

### 4.2 Shallow safety alignment

**Definition.** [[shallow-safety-alignment]] calls safety alignment shallow when it changes the base model's distribution mainly over the first few output tokens (§2).

**Evidence.** Prefilling "I apologize, but I cannot" lowered the harmfulness rate of the unaligned Llama-2-7B base from 68.6% to 2.1% (Table 1). Per-token KL between aligned and base models on harmful responses is highest at the first positions (Fig. 1). Fine-tuning Llama-2-7B-Chat on 100 harmful examples (LR 2×10⁻⁵, batch 64) raised the attack success rate (ASR, the share of harmful prompts answered harmfully) from 1.5% to 22.4%, 76.4%, and 87.9% after 2, 4, and 6 steps, with the largest per-token loss and gradient norm at the first positions (Fig. 3).

**Control 1: safety recovery examples.** The objective α · E[−log π_θ(r | x, h_≤k)] + (1 − α) · E[−log π_θ(y′ | x′)] (Eq. 2) trains a refusal r after the first k tokens of a harmful response h, anchored by Alpaca prompts x′ with responses y′ distilled from the initial model. With α = 0.2, k = 0 half of the time and otherwise uniform in [1, 100], ASR under a 5-token prefill fell from 42.1% to 2.8% and AlpacaEval win rate moved from 51.8% to 49.5% (Table 2, §3.1).

**Control 2: token-wise constrained SFT.** The loss (Eq. 3) is E[ −Σ_t (2/β_t) log σ(β_t log(π_θ(y_t|·)/π_aligned(y_t|·))) ]. Its gradient equals the cross-entropy gradient times w_t = 2σ(β_t Δ_t), where Δ_t = log π_aligned(y_t|·) − log π_θ(y_t|·) and σ is the logistic function (Eq. 5). At initialization Δ_t = 0 and w_t = 1.

**Worked example.** Suppose training has raised log π_θ(y_t) by 4 nats above the aligned model (Δ_t = −4). With the paper's β_1 = 0.5, w = 2σ(−2) = 0.238. With β_t = 2 at positions 2-5, w = 2σ(−8) = 0.00067: the update at that position has nearly stopped. With β_t = 0.1 after position 5, w = 2σ(−0.4) = 0.803, close to plain SFT.

With these β_t values, Llama-2-7B-Chat fine-tuned on Samsum had ASR 3.2% against 23.4% for standard SFT at ROUGE-1 50.1 against 51.7, and on GSM8k 37.4% accuracy against 41.7% (Table 3). Uniform β = 2.0 gave 0.5% ASR on the harmful-example attack but GSM8k accuracy 2.1% (Table 4), so the combination of low ASR and retained utility depends on setting β_t differently by position. Status: **Result (single study)**, two 7B models.

### 4.3 Agentic fine-tuning

[[agentic-finetuning-misalignment]] fine-tuned Llama-3.1-8B-Instruct on benign web-navigation demonstrations: WebArena-Lite success rose from 2.42% to 22.42%, while harmful-task attack success rose from 32.88% to 64.38% and refusal fell from 26.03% to 6.85% (Table 1). Attack success rose in all 8 model-domain pairs tested (Tables 1-2). A stage trained on benign agentic demonstrations therefore needs the same safety re-evaluation as a stage trained on chat data.

**Implication for all of §4.** A safety measurement has two sides. Refusal on harmful prompts must be reported with refusal on safe prompts that resemble harmful ones: on XSTest, Llama-2-70b-chat with its original system prompt fully refused 38% of 250 safe prompts ([[xstest]] Tables 1-2). ch-52 covers both measurements.

## §5 Mitigations with evidence

| Control | What it constrains | Evidence (setting, number) | Limit | Full treatment |
|---|---|---|---|---|
| General-data replay | training distribution | LLaMA-7b MMLU-human 34.72 → 26.8 without, 30 with 10,000 Alpaca samples | 400M replay tokens are 0.033% of a 1.2T-token corpus | ch-30b, ch-32a |
| Lower learning rate | step size | Llama-2-7b-Chat harmfulness 16.1% at 2e-5 vs 34.2% at 5e-5 (batch 128) | lower target gain | ch-01 |
| LoRA | update rank | Llama-2-7B code IFT, matched HumanEval 0.498 vs 0.497: forgetting average (higher is better) 0.631 (LoRA) vs 0.446 (full) | lower learning in CPT; similar forgetting in math IFT; no safety protection | ch-30 |
| SDFT | target distribution | GSM8K after OpenFunctions: 21.5 (vanilla) vs 29.1 (SDFT), seed 29.4 | main runs LoRA r = 8 on Llama-2-7b-chat; one-dataset checks for full FT, 13B, Llama-3-8B | ch-31 |
| Weight interpolation | distance in weights | CLIP shifts 68.6 → 76.9 at α = 0.5; RLHF model-averaging Pareto front above nearly all compared methods | vision evidence; LLM ratio chosen per model | ch-30c |
| Pretraining-loss mixing | output distribution on pre-training text | InstructGPT γ ≥ 20 recovers regressions at 1.3B | needs pre-training-like data | ch-38 |
| Early-token constraint | per-token deviation | Samsum ASR 23.4% → 3.2% | lower GSM8k accuracy | §4.2 here; evaluation in ch-52 |
| On-policy data | data source | Llama 3.1 8B MMLU target: drop 38.5 (SFT) vs −0.2 (GRPO) | needs a reward or verifier | ch-38a |

### 5.1 General-data replay

Replay adds general data to the stage's loss: L = L_target + λ · L_general, where λ is the weight or the batch share of general data. In [[catastrophic-forgetting-continual-finetuning]], mixing 10,000 Alpaca samples into continual tuning kept LLaMA-7b's MMLU-human at 30% instead of 26.8%, starting from 34.72% (§5.4). In [[spurious-forgetting]], replaying 20% of old data re-aligned the model to the old task during later training (§5.2, Fig. 3b). In [[mitigating-alignment-tax-rlhf]], replaying pre-training data up to 4 times the RLHF data (400M tokens) beat model averaging on reading comprehension but lost on commonsense QA and translation (App. C.1).

**Worked example.** 400M replay tokens over a 1.2T-token pre-training corpus is 4×10⁸ / 1.2×10¹² = 0.033% coverage (App. C.1 prints "about 0.03%"; §2 of the same version prints "~0.01%"). The authors attribute the partial benefit to abilities that are underrepresented in a 0.033% sample (Interpretation). Status: **Replicated** that replay reduces forgetting ([[catastrophic-forgetting-continual-finetuning]], [[spurious-forgetting]]); [[mitigating-alignment-tax-rlhf]] shows that at 0.033% coverage the benefit varies by benchmark.

### 5.2 Learning rate and update size

[[retaining-by-doing]] reports for Self-SFT that "a high learning rate is typically required to reach high target performance for SFT, often at the cost of severe forgetting; a smaller learning rate reduces forgetting but fails to reach the same target performance even with more epochs" (§2.3, Fig. 3). [[finetuning-compromises-safety]] Table 12 (above) and [[wise-ft]] §4 (for CLIP ViT-B/16, two learning rates whose ImageNet accuracies were 0.3 points apart differed by up to 8 points under distribution shift) show that the learning rate changes non-target behavior; the WiSE-FT case compares robustness, not forgetting of a prior task. Given §2.3, part of the benefit is that a lower learning rate fits the target less (Interpretation). ch-01 treats learning rate and step count as the determinants of distance from the initial weights.

### 5.3 LoRA

**Definition.** LoRA freezes a weight matrix W_pretrained ∈ R^{d×k} and trains a low-rank perturbation: W_finetuned = W_pretrained + γ_r A B, with A ∈ R^{d×r}, B ∈ R^{r×k}, rank r ≪ d, k, and scale γ_r = α/r for a hyperparameter α ([[lora-learns-less-forgets-less]] §2).

**Evidence.** On Magicoder-Evol-Instruct-110K, LoRA r = 256 at epoch 4 reached HumanEval 0.498 with forgetting average 0.631; full fine-tuning at epoch 8 reached 0.497 with 0.446 (Tables S5-S6). Weight decay and attention dropout learned and forgot as much as full fine-tuning (§4.5). Full fine-tuning produced fewer unique HumanEval solutions than the base model, with LoRA between the two (§4.5). **Limits.** In continued pretraining LoRA did not reach full fine-tuning at any rank (§4.1). On MetaMathQA at epoch 16, the forgetting averages were nearly equal (0.567 for LoRA r = 256 vs 0.559 for full fine-tuning, Table S8), and for math IFT the authors judge full fine-tuning's learning-forgetting trade-off preferable (§4.3). LoRA still forgets ([[scaling-laws-forgetting]]), and it did not reduce safety loss in [[finetuning-compromises-safety]] (Table 11).

### 5.4 Self-distillation fine-tuning (SDFT)

**Definition.** SDFT replaces each target response y with a rewrite ỹ generated by the model being fine-tuned from the prompt and the original response, keeps ỹ only if its extracted answer matches y, and trains on the result ([[sdft-self-distillation-finetuning]] Eq. 3-5):

    ỹ ~ f_θ(y | c, x, y_orig);   ỹ′ = ỹ if Extract(ỹ) = y_orig, else y_orig;   L_SDFT = −log f_θ(ỹ′ | c, x)

c is the task context, x the instruction, y_orig the dataset response, and Extract returns the final answer. The rewrite prompt used in most experiments is (Fig. 3):

```
Below are an instruction that describes a task along with a reference answer. Using
the reference answer as a guide, write your own response.

### Instruction:
{instruction}

### Reference Answer:
{original response}

### Response:
```

**Evidence.** Llama-2-7b-chat (LoRA r = 8) fine-tuned on OpenFunctions: HumanEval 9.8 (vanilla) vs 15.2 (SDFT), seed 13.4; GSM8K 21.5 vs 29.1, seed 29.4; target OpenFunctions 34.8 vs 36.6 (Table 1). After GSM8K fine-tuning, AlpacaEval win rate was 23.38 (vanilla) vs 66.73 (SDFT), seed 66.04 (Table 2). **Limits.** The main tables use one 7B chat model with LoRA r = 8 on query and value only. §5.3 Table 5 repeats the comparison with one fine-tuning dataset each for full fine-tuning of Llama-2-7b-chat (GSM8K), LoRA on Llama-2-13b-chat (GSM8K), and LoRA on Llama-3-8B-Instruct (OpenFunctions), and SDFT scored at least as high as vanilla fine-tuning on every reported column there. Seeds are not reported, and the rewrite quality is bounded by the seed model (Interpretation).

### 5.5 Weight interpolation toward the starting weights

**Definition.** WiSE-FT uses the weights (1 − α) · θ_0 + α · θ_1, where θ_0 is the model before fine-tuning, θ_1 after, and α ∈ [0, 1] ([[wise-ft]] Eq. 1). **Evidence.** CLIP ViT-L/14@336px fine-tuned end to end: ImageNet 86.2, average over five shifts 68.6; at α = 0.5: 86.8 and 76.9 (Table 1), a gain of 8.3 points under shift (derived). The authors recommend α = 0.5 without domain knowledge (§4). For LLM alignment, [[mitigating-alignment-tax-rlhf]] instruction-tuned OpenLLaMA-3B on ShareGPT and then aligned it on HH-RLHF with Rejection Sampling Fine-tuning (RSF: sample n responses per prompt, keep the highest-reward one, fine-tune on the kept responses, repeat) (§3). The Pareto front of averaging the weights before and after RLHF "supersedes nearly all other methods" among early stopping, L1 and L2 penalties, LoRA, knowledge distillation, and stochastic moving averaging (§4.1). Heterogeneous Model Averaging (HMA), which splits the transformer into K groups of layers and gives each group its own averaging ratio, moved the front further (§6, Fig. 5); with α as the weight on the post-RLHF model, "α = 0.2 can consistently alleviate the alignment tax without hurting alignment performance" (§6). **Limits.** WiSE-FT tested image models only; the LLM ratio was chosen from that paper's own figures.

### 5.6 Pretraining-loss mixing (PPO-ptx preview)

InstructGPT's objective (Eq. 2) adds γ E_{x∼D_pretrain}[log π(x)] to the KL-penalized reward, where γ is the pretraining loss coefficient ([[rlhf-instructgpt-ptx]] §3.5). The runs used γ = 27.8 and 8 times more pretraining examples than RL episodes (App. C.4). At 1.3B, γ ≥ 20 recovered the public-NLP regressions, while raising the KL coefficient to 100 times its default with γ = 0 did not (App. E.6). PPO-ptx still trailed GPT-3 on DROP, SQuADv2, and translation (§4.2). ch-38 derives the objective.

### 5.7 On-policy data (preview)

**Definition.** On-policy data are samples generated by the current model being trained. REINFORCE and GRPO are policy-gradient methods that raise the log-probability of sampled responses in proportion to their reward; GRPO replaces the raw reward with the reward minus the mean reward of a group of samples for the same prompt, divided by the group's standard deviation (ch-37, ch-40). **Evidence.** [[retaining-by-doing]] reports target gain and non-target drop, where the drop is the mean accuracy decrease over the other target tasks plus MATH, WildJailbreak, and WildGuardTest (§2.1-§2.2). In Table 1, Llama 3.1 8B Instruct trained on MMLU had target gain 11.1 and non-target drop 38.5 with SFT on filtered Llama-3.3-70B-Instruct responses, and gain 14.6 and drop −0.2 with GRPO; on IFEval the drops were 27.8 (SFT) and 3.4 (GRPO). For Llama 3.1 8B, REINFORCE had drops of 7.7, −0.1, and −0.8 on IFEval, MMLU, and Countdown, and GRPO without its KL term behaved like GRPO with it except for Llama models on IFEval (§4.1), so the authors attribute the difference to on-policy data (Interpretation). For Qwen 2.5 1.5B and 7B Instruct on IFEval and MMLU, Iterative-SFT, which regenerates targets at the start of each epoch, reached SFT's target accuracy "while only exhibiting mild to no forgetting" (§4.2, Fig. 7). [[rls-razor]] proposes that forgetting is predicted by E_{x∼τ}[KL(π_0 ‖ π)] on the new task τ, where π_0 is the base policy and π the fine-tuned policy: a quadratic fit gave R² 0.96 in a toy setting (ParityMNIST) and 0.71 for Qwen 2.5 3B-Instruct (§4). [[retaining-by-doing]] reports that this KL link "does not always hold" in its setting: the Pearson correlation between KL and drop was 0.52, and between Self-SFT and SFT a larger KL did not always mean more forgetting (App. A.5). Status: on-policy data reduces forgetting is **Replicated**; KL on the new task as the predictor is an **Open question**.

## §6 The forgetting report used by later labs

Labs ch-36, ch-46, and ch-53 and the capstone ch-59 attach this report to every trained checkpoint.

1. **Checkpoints.** The stage's input checkpoint and the original pretrained base, with exact identifiers.
2. **Suites.** One or more held-out suites per slice listed in §1, with a statement of which suites were used to choose settings (development) and which only for reporting (held-out), and the decontamination method.
3. **Protocol.** Template, shot count, decoding settings, judge model and version, identical for both checkpoints; at least 3 templates for format-sensitive tasks (ch-53).
4. **Per-slice statistics.** Δ, SE_paired, 95% interval, gained and lost item counts, and agreement rate.
5. **Safety pair.** Harmful-request ASR and over-refusal on safe look-alike prompts.
6. **Factuality.** Closed-book accuracy on facts answered correctly by the reference, plus abstention rate.
7. **Distribution distance.** Cross-entropy to the reference's predictions on general text, and KL to the reference on the target prompts.
8. **Diversity.** Unique outputs out of k samples on one generation task, or pass@k at large k.
9. **Recovery test** for any slice whose paired interval lies entirely below zero: short fine-tuning on half of the slice, evaluation on the other half (§3.2).
10. **Trade-off curve.** Target gain against mean non-target drop for at least two settings of one control (learning rate, epochs, α, or replay share).
11. **Seeds.** Number of seeds per configuration, stated as 1 when only one was run.
12. **Decision rule.** A non-inferiority margin per slice, written before the evaluation is run.

## Negative samples and negative feedback

The course distinguishes four uses of a negative signal (ch-43a): **negative marginal value** (a sample that lowers performance when used as a positive target, so it is removed), **negative as content** (a failure placed in the input or in a corrected target and trained with ordinary cross-entropy), **negative as conditioning** (failures trained under a control token that is not used at inference), and **negative as gradient** (an explicit decrease of a sample's likelihood). This chapter's controls use the first, second, and fourth; none uses negatives as conditioning.

**Where negatives come from.** (a) Refusal targets for harmful prompts and safety recovery examples with a harmful prefix h_≤k, labeled by a jailbroken model and the aligned model ([[shallow-safety-alignment]] App. A.3). (b) Fine-tuning examples whose facts the model cannot produce, labeled Unknown by sampling ([[finetuning-new-knowledge-hallucination]] §3). (c) Incorrect self-generated responses, labeled by a reward function ([[retaining-by-doing]] §2.2) or by answer mismatch ([[sdft-self-distillation-finetuning]] Eq. 4). (d) Zero-reward rollouts in GRPO.

**What current practice does.** (a) is **negative as content**: the harmful prefix sits in the context and the refusal is trained with cross-entropy. (b) and (c) are **negative marginal value**: the examples are filtered out or, in SDFT, replaced by the original response. (d) is **negative as gradient**: a below-mean reward gives a negative advantage.

**Mechanism.** For a softmax over logits z, ∂ log p_y / ∂ z_j = 1[j = y] − p_j, where y is the target token and p_j the probability of token j. Negatives used as content apply this positive-target gradient to the refusal tokens; they lower the probability of harmful continuations only through normalization. Only (d) applies the gradient with a negative sign to a sampled sequence, removing mass from it; that mass goes to other tokens in proportion to their current probability, so pushing down an already-unlikely sample moves most of the mass to the most likely alternative. ch-43a derives this case.

**Evidence.** Mixing 100 refusal examples into GPT-3.5 Turbo's 100-example harmful fine-tuning lowered harmfulness from 91.8% to 23.0% ([[finetuning-compromises-safety]] Table 4). Recovery examples lowered 40-token prefill ASR from 57.0% to 4.5% ([[shallow-safety-alignment]] Table 2). Filtering Unknown examples made the gap between early stopping and convergence very small ([[finetuning-new-knowledge-hallucination]] §4.2). REINFORCE and GRPO showed similar low forgetting despite different advantage estimates ([[retaining-by-doing]] Table 1). None of the cited sources measures what share of the improvement is due to the negatives themselves (not reported).

**Controls.** Anchor refusal training with a positive utility term (α = 0.2 in Eq. 2 of [[shallow-safety-alignment]]); bound per-position deviation (β_t); keep negatives on-policy when they are used as gradient.

**Diagnostics.** Log refusal on harmful and on safe look-alike prompts separately; per-position KL to the reference; for GRPO, loss and probability statistics split by advantage sign (ch-43a).

**Effect on generality.** Refusal data can raise over-refusal ([[xstest]]); unfiltered Unknown targets raise hallucination; filtered self-generated data keeps the training distribution close to the model and reduced forgetting in the SDFT and Iterative-SFT results.

## Recipe

All rows were read at the stated locus on 2026-09-15.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-2-7B → Magicoder-Evol-Instruct-110K, LoRA | 7B | SFT | rank; α; peak LR; schedule; global batch; max length; optimizer; epochs | r = 16, 64, 256; α = 2r; 2e-4 (r = 16, 64), 1e-4 (r = 256); cosine, warmup 0.1 of duration, alpha_f 0.01; 192 sequences; 4096; decoupled LionW (0.9, 0.95); 1, 2, 4, 8, 16 (separate runs) | arXiv:2405.09673v2 §4.1, App. A | verified 2026-09-15 | Tables S5-S6 per epoch; Fig. S3 α sweep at r = 256 (α = 512 best) |
| Llama-2-7B → MetaMathQA, full FT and LoRA | 7B | SFT | peak LR; global batch; length | full 1e-5; LoRA 1e-4 (r = 16, 64), 5e-5 (r = 256); 768; 1024 | arXiv:2405.09673v2 App. A | verified 2026-09-15 | Tables S7-S8; App. B LR sweep at 2 epochs |
| Llama-2-7b-chat SDFT and vanilla baseline | 7B | distill-SFT | adapter; LR; batch; data and epochs | LoRA r = 8 on query and value; 1e-4 cosine to 0; 8; 2,000 examples × 2 epochs (Alpaca, Dolly, MagiCoder), 20,000 × 2 (OpenHermes), full train set × 5 (GSM8K, OpenFunctions), × 2 (LIMA) | arXiv:2402.13669v2 App. A | verified 2026-09-15 | Tables 1-3 (SDFT vs vanilla); no hyperparameter ablation reported |
| Llama-2-7B-Chat, token-wise constrained SFT | 7B | SFT | β_t; LR; optimizer; batch; warmup; epochs | β_1 = 0.5, β_t = 2 (t = 2-5), β_t = 0.1 (t > 5); 2×10⁻⁵; AdamW (0.5, 0.999); 64; linear, first 10 steps; 3 (benign sets) | arXiv:2406.05946v1 §4.2, App. A.5 | verified 2026-09-15 | Table 4 uniform-β ablation; App. C Table 5 warmup ablation |
| Llama-2-7B-Chat, safety recovery augmentation | 7B | SFT | data; mix; prefix length k; LR; epochs | 256 (x, h, r) triplets + Alpaca prompts with distilled responses; α = 0.2 (16 + 64 per batch); k = 0 w.p. 0.5, else Uniform[1, 100]; 2×10⁻⁵ AdamW; 10 epochs on D_H | arXiv:2406.05946v1 §3.1, App. A.3 | verified 2026-09-15 | Table 2 (ASR); no ablation of α or k located |
| Llama-2-7b-Chat, benign Alpaca SFT | 7B | SFT | data; epochs; LR; batch | 50K Alpaca (safety samples removed); 1; 2e-5; 128 | arXiv:2310.03693v1 §4.4, App. G.1 | verified 2026-09-15 | Table 12 LR × batch; Table 13 epochs |
| InstructGPT PPO-ptx (1.3B, 6B, 175B) | 1.3B-175B | RL | γ; pretraining examples; KL reward coefficient β | 27.8; 8 × RL episodes; 0.02 | arXiv:2203.02155v1 App. C.4 (γ, 8×, "β = 0.02"), App. E.6 | verified 2026-09-15 | Fig. 33 γ sweep at 1.3B (γ ≥ 20 recovers); Fig. 34 β sweep with γ = 0; App. E.7 Fig. 36: optimal β "around 0.01 and 0.02" |
| OpenLLaMA-3B, RSF on HH-RLHF, model averaging | 3B | merge | weight on post-RLHF model; HMA parts | α = 0.2; K = 3 | arXiv:2309.06256v4 §6 | verified 2026-09-15 | Figs. 3, 5, 16 α sweeps; Fig. 5 (right) K = 3, 6, 9 |
| OpenLLaMA-3B, RSF, experience replay | 3B | RL | replay amount | up to 4 × RLHF data per batch (400M tokens) from a 1.2T-token corpus | arXiv:2309.06256v4 App. C.1 | verified 2026-09-15 | Fig. 7 (replay vs averaging, 3 benchmarks) |
| Llama-3.1-8B-Instruct and Qwen-2.5-7B-Instruct, SFT and GRPO | 8B, 7B | SFT, RL | LR; schedule; batch; epochs; KL coefficient; group size | 5e-6; cosine, warmup ratio 0.03; 128 (IFEval, MMLU), 64 (Countdown); 2; 0.05; 5 | arXiv:2510.18874v3 App. A.3 | verified 2026-09-15 | Fig. 3 Self-SFT LR and epochs; Fig. 6 KL 0.05 vs 0 |
| PaLM 2-S, closed-book QA SFT | not reported | SFT | examples; LR; batch; dropout; epochs; stop rule | 6142; 1e-5 fixed; 128; 0.05; 50; best development epoch (5-10) | arXiv:2405.05904v3 §4, App. E | verified 2026-09-15 | Fig. 6 repeats trends at LR 1e-4 |
| BLOOMZ, mT0, LLaMA, Alpaca, continual instruction tuning | 1.1B-7.1B | SFT | LR; schedule; batch; length; epochs; replay | 2e-5; constant (BLOOMZ, mT0), cosine (LLaMA, Alpaca); 4 per device; 512; 3 per task; 10,000 Alpaca samples | arXiv:2308.08747v5 §4.2, §5.4 | verified 2026-09-15 | no ablation reported (settings follow prior work, footnote 7) |
| CLIP ViT-L/14@336px, WiSE-FT | not reported | merge | mixing coefficient | α = 0.5 | arXiv:2109.01903v3 §4 | verified 2026-09-15 | Table 1 (α = 0.5 vs optimal α); App. B |

**Starting point for a small general-purpose run.** For a 7B base model fine-tuned on a single-domain instruction set of the size of Magicoder-Evol-Instruct-110K (72.97M tokens), the verified LoRA row supports r = 256 with α = 512 on all modules, peak LR 1e-4, cosine schedule with 10% warmup, and global batch 192 on 4096-token sequences; in that run (Llama-2-7B, Magicoder, 32 GPUs) epoch 4 matched full fine-tuning's best HumanEval with a forgetting average of 0.631 against 0.446. Evaluate the forgetting report after each epoch, because forgetting rose with epochs in the same table. When the starting model is a 7B chat model whose safety must be kept, the constrained-SFT row (β_1 = 0.5, β_t = 2 for positions 2-5, 0.1 after, LR 2×10⁻⁵, batch 64, 3 epochs) is the verified setting, measured on Samsum, SQL Create Context, and GSM8k. After a preference or RL stage on a 3B model, averaging with the stage's input at α = 0.2 (weight on the post-RLHF model) is the ratio the authors recommend from their OpenLLaMA-3B RSF, DPO, and PPO sweeps (Figs. 3, 5, 16).

## Generalization lens

**(a) What increases breadth.** Models that had general instruction tuning forgot less during later task tuning in one comparison (LLaMA-7b FG 34.57 vs Alpaca-7b 18.14 on domain knowledge, [[catastrophic-forgetting-continual-finetuning]] Table 6; authors' interpretation). At early stopping, training on MaybeKnown instead of HighlyKnown examples raised accuracy on MaybeKnown test questions from 60.1 to 69.9 while HighlyKnown accuracy moved from 98.7 to 98.4 ([[finetuning-new-knowledge-hallucination]] Table 2). Weight interpolation raised accuracy under distribution shift while keeping the target score ([[wise-ft]] Table 1). On-policy targets kept non-target accuracy near its starting value ([[retaining-by-doing]] Table 1).

**(b) What causes narrowing or forgetting.** Fitting the target further (§2.3), more epochs ([[lora-learns-less-forgets-less]] Table S6), higher learning rates ([[finetuning-compromises-safety]] Table 12), external-model responses far from the model's own distribution ([[sdft-self-distillation-finetuning]] Table 1; [[retaining-by-doing]] Table 1), unknown facts as targets ([[finetuning-new-knowledge-hallucination]] Table 1), and any fine-tuning of a model whose safety depends on its first response tokens ([[shallow-safety-alignment]] Fig. 3), including agentic data ([[agentic-finetuning-misalignment]] Table 1). Full fine-tuning also reduced output diversity on HumanEval ([[lora-learns-less-forgets-less]] §4.5).

**(c) How to measure it for this stage.** Paired per-item deltas with 95% intervals on held-out slices ([[adding-error-bars-evals]] Eq. 5, 7); agreement with the reference's answers and cross-entropy to its predictions ([[scaling-laws-forgetting]] §3.1); harmful-request ASR together with over-refusal ([[finetuning-compromises-safety]], [[xstest]]); closed-book accuracy on facts known before training ([[finetuning-new-knowledge-hallucination]]); recovery tests for slices with a detected drop ([[spurious-forgetting]] §3.2); and a gain-versus-drop curve over one control. Known measurement errors: an unpaired interval can include zero when the paired interval on the same items excludes it (§1 worked example); a naive standard error on clustered items was 3.05 times smaller than the clustered one on DROP ([[adding-error-bars-evals]] Table 4); and an English-only suite can report as lost capability a drop that is smaller when the same items are posed in another language or encoding (Alpaca's in-context-learning drop was 56.75 points in English, 29.00 in French, and 1.50 in Leetspeak; [[forgetting-via-implicit-inference]] Table 1).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Comparing aggregate scores without pairing | a drop is reported as "within noise" although the paired interval excludes zero | compute gains, losses, and SE_paired from per-item scores |
| Using the stage's validation loss as the forgetting signal | validation loss improves while held-out slices fall | add at least one held-out suite per capability slice |
| Measuring against the pretrained base only | a later stage's own loss is hidden inside the cumulative change | report both the stage input and the original base |
| Evaluating safety only on harmful prompts | refusal rises after a safety fix and over-refusal is not seen | pair ASR with XSTest-style safe prompts |
| Skipping safety evaluation after benign or agentic SFT | harmfulness rises with no change in task metrics | re-run the safety pair after every stage |
| Treating every measured drop as lost knowledge | accuracy returns after one epoch on a disjoint half of the slice | run the recovery test on a disjoint half |
| Training on facts the model cannot produce | development accuracy peaks early and then falls while training accuracy rises | categorize targets by sampling the reference; filter or stop early |
| Assuming LoRA protects safety | LoRA run shows higher harmfulness than expected | include LoRA runs in the safety pair |
| Choosing the operating point from one run | no evidence that another setting gives more gain for the same drop | plot gain against drop for two or more settings |
| Treating a replay sample as coverage of all pre-training abilities | some slices still drop while others recover | report replay coverage and per-slice deltas |

## Check your understanding

1. In the §1 worked example, the paired interval excludes zero and the unpaired interval does not. Explain which term of Miller's variance relation causes the difference, and what property of the two checkpoints makes it large.
2. [[scaling-laws-forgetting]] fits forgetting as a linear function of fine-tuning loss. Explain why this implies that early stopping cannot improve the trade-off within one method, and why LoRA's matched-accuracy result in [[lora-learns-less-forgets-less]] does not contradict it.
3. A model loses 20 points on an English benchmark after code SFT, and recovers 15 of them when the same questions are asked in Spanish. Using [[forgetting-via-implicit-inference]], explain what this suggests about the lost points and which additional measurement would test that explanation.
4. Explain why training on Unknown facts lowers accuracy on questions from relations that were not in the training set, using the linear model of [[finetuning-new-knowledge-hallucination]].
5. In the constrained SFT loss of [[shallow-safety-alignment]], explain why β_t = 2 on positions 2-5 protects refusal while β = 2.0 on all positions reduces GSM8k accuracy to 2.1%.
6. [[retaining-by-doing]] finds that GRPO without KL forgets about as little as GRPO with KL. Explain why this points to the data source rather than the regularizer, and what [[rls-razor]] would need to show to explain the same result.
7. Replay of 400M tokens beat model averaging on one benchmark and lost on two others in [[mitigating-alignment-tax-rlhf]]. Explain a cause of this pattern and how a forgetting report would detect it.

## Connections

- **Previous (dependency):** ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate. It sets the choices whose cost this chapter measures.
- **Next:** ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares.
- ch-00 — What General Capability Means and How It Is Measured (capability profile and held-out suites).
- ch-01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability (learning rate and step count as update size).
- ch-30c — Weight Averaging and Model Merging for Generalist Models (WiSE-FT and merging in full).
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation (training on the model's own filtered samples, related to SDFT and Iterative-SFT).
- ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining.
- ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split.
- ch-37 — Policy-Gradient Foundations for Language Models (REINFORCE).
- ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax (PPO-ptx derivation).
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.
- ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO (GRPO).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-46 — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention.
- ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing.
- ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions (paired intervals and multiple-comparison control).
- ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming.
- ch-53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness.
- ch-59 — Capstone: Reproduce One Stage of an Open General-Model Recipe with a Generality Gate.

## Sources

- [[catastrophic-forgetting-continual-finetuning]] — FG metric, Table 4 and Table 6 forgetting in 1B-7B continual instruction tuning, 10,000-sample Alpaca replay.
- [[scaling-laws-forgetting]] — cross-entropy-to-base forgetting metric, ARC answer-agreement example, Eq. 4 linear law and its App. B cross-method check, AdvBench refusal counts.
- [[forgetting-via-implicit-inference]] — task-inference decomposition and conjugate-prompting recoveries (Tables 1-3).
- [[spurious-forgetting]] — task alignment versus knowledge, recovery test, AOA safety loss and restoration, Freeze results.
- [[finetuning-new-knowledge-hallucination]] — SliCK categories, linear accuracy model, early stopping and filtering of Unknown examples.
- [[finetuning-compromises-safety]] — harmfulness after benign fine-tuning, LR and batch ablation, LoRA safety result, refusal-data mixing.
- [[shallow-safety-alignment]] — refusal-prefix and per-token KL evidence, per-step ASR, recovery-example augmentation, token-wise constrained SFT.
- [[lora-learns-less-forgets-less]] — LoRA versus full fine-tuning learning and forgetting tables, regularization and diversity comparisons, LoRA settings.
- [[sdft-self-distillation-finetuning]] — SDFT equations, downstream, safety, and helpfulness tables, settings.
- [[wise-ft]] — weight-space interpolation equation, CLIP Table 1, learning-rate sensitivity, α = 0.5 recommendation.
- [[mitigating-alignment-tax-rlhf]] — alignment tax in RSF, model averaging Pareto result, HMA, α = 0.2, replay coverage.
- [[retaining-by-doing]] — SFT versus REINFORCE versus GRPO gain and drop table, KL ablation, Iterative-SFT, settings.
- [[rls-razor]] — KL-on-new-task predictor of forgetting and its fit quality.
- [[rlhf-instructgpt]] — InstructGPT library card (unverified on 2026-09-15; values in this chapter come from the excerpt below).
- [[rlhf-instructgpt-ptx]] — verified PPO-ptx objective, γ = 27.8, pretraining example ratio, γ and β ablations.
- [[adding-error-bars-evals]] — standard error, paired difference, and clustered standard error formulas.
- [[agentic-finetuning-misalignment]] — safety loss after benign agentic fine-tuning.
- [[xstest]] — over-refusal measurement on safe prompts that resemble unsafe ones.
