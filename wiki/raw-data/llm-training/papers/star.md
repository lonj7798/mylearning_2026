<!-- scope: Zelikman, Wu, Mu, Goodman (arXiv:2203.14465, Mar 2022) — STaR: iterative rationale bootstrapping from answer labels with GPT-J 6B; rationalization with answer hints; arithmetic, CommonsenseQA, GSM8K
     see-also: [[quiet-star]], [[rest-em]], [[v-star]], [[rejection-sampling-finetuning]], [[training-verifiers-to-solve-math-word-problems]]
-->

# STaR: Bootstrapping Reasoning With Reasoning
- **Core Insight:** On the CommonsenseQA dev set, GPT-J 6B trained with STaR plus rationalization reaches 72.5% accuracy, compared with 60.0% for GPT-J fine-tuned to output answers directly and 73.0% for a fine-tuned GPT-3 that is 30× larger (Table 1).
- **Guideline:** When a dataset has gold final answers but no rationales and the base model's few-shot accuracy is above chance, the STaR loop (sample rationales, keep those with correct answers, rationalize failures with the answer as a hint, fine-tune the original model, repeat) is supported by Table 1 (+12.5 points over direct fine-tuning on CQA); for tasks with high chance accuracy, such as binary decisions, the authors report many poor rationales and no filtering solution (§6).
- **Authors:** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman (Stanford University; Google Research)
- **Year:** 2022 (arXiv v1 2022-03; v2 2022-05)
- **URL:** https://arxiv.org/abs/2203.14465
- **Source type:** paper
- **Relevant topics:** rationale bootstrapping, self-training, expert iteration, chain-of-thought fine-tuning, rationalization, synthetic reasoning data

## Abstract
Chain-of-thought rationales improve language-model reasoning, but inducing them requires either a large rationale dataset or few-shot prompting, which is less accurate. STaR uses a small set of rationale examples and a large dataset without rationales. It generates rationales for many questions using few-shot prompts; if an answer is wrong, it generates a rationale again with the correct answer given; it fine-tunes on all rationales that ended in correct answers; and it repeats. STaR improves over a model fine-tuned to predict final answers on several datasets and performs comparably to fine-tuning a 30× larger model on CommonsenseQA (abstract).

## Key Contributions
- A bootstrapping loop that builds a rationale dataset from a few prompt examples, filtered only by final-answer correctness (§1, contribution 1).
- Rationalization: the model justifies a given gold answer and is then fine-tuned as if it had produced the rationale without the hint (§3.2).
- Ablations on arithmetic, CommonsenseQA (CQA), and GSM8K (§4).
- A derivation showing STaR approximates a policy-gradient objective with an indicator reward (§3.1, Eqs. 1-2).
- The authors state it is, to their knowledge, the first technique that lets a pretrained LLM iteratively use its own language-modeling capacity to improve itself (§1, contribution 4).

## Key Figures/Tables to Study
- Algorithm 1 and Figure 1: the outer loop, filtering, and rationalization branch.
- Table 1 (CQA) and Table 2 (GSM8K): accuracy and the share of training data retained.
- Figure 4: n-digit addition accuracy per iteration with and without rationalization.
- §5 "Temperature": why higher-temperature sampling was not used to enlarge the dataset.

## Technical Details
- **Setup.** Pretrained LLM M; dataset D = {(x_i, y_i)} of problems with answers; prompt set P of (x, r, y) with rationales, P ≪ D (e.g. P = 10) (§3.1).
- **Loop (Algorithm 1).** For n = 1..N: (1) generate (r̂_i, ŷ_i) with M_{n−1} for all i; (2) generate rationalized (r̂_i^rat, ŷ_i^rat) from add_hint(x_i, y_i); (3) D_n = examples with ŷ_i = y_i; (4) D_n^rat = examples with ŷ_i ≠ y_i and ŷ_i^rat = y_i; (5) M_n ← train(M, D_n ∪ D_n^rat) (Alg. 1, lines 3-7).
- **Restart from the base model.** Each iteration fine-tunes the original pretrained M, not the previous iterate, "to avoid overfitting"; the loop repeats until performance plateaus (§3.1).
- **Policy-gradient view.** J(M, X, Y) = Σ_i E_{r̂_i, ŷ_i ∼ p_M(·|x_i)} 1(ŷ_i = y_i) (Eq. 1) and ∇J = Σ_i E[1(ŷ_i = y_i) · ∇ log p_M(ŷ_i, r̂_i | x_i)] (Eq. 2). Here r̂_i is a sampled rationale, ŷ_i the predicted answer, y_i the gold answer, and 1(·) the indicator. The indicator removes the gradient of rationales with wrong answers, which is the filtering step. STaR approximates Eq. 2 by greedy decoding (lower variance, possibly biased exploration) and by taking multiple gradient steps on the same batch (§3.1).
- **Rationalization.** The gold answer is inserted as a hint (Fig. 2); the hint is removed from the prompt when the rationale is added to the training set (§3.2). It is applied only to problems the model failed (§3.2). Without it, the loop "fails to solve any new problems in the training set because it receives no direct training signal for problems it fails to solve" (§1). The authors frame rationalization as an off-policy estimate of Eq. 1 with the hint-augmented model as proposal distribution (§5, Interpretation).
- **Protocol.** GPT-J 6B with the GPT-J repository fine-tuning script; 100-step learning-rate warmup, then constant learning rate; 40 training steps in the first outer loop, increased by 20% per outer loop (§4.1). Few-shot prompts are used in later outer-loop iterations unless stated otherwise; including or omitting them after the first iteration did not substantially change final performance with rationalization (§4.1).
- **Arithmetic.** 50,000 generated n-digit addition problems; 10,000 sampled per outer-loop iteration; 10 few-shot examples per digit length, 1-5 digits (§4.1-4.2). After 16 iterations overall accuracy is 89.5%, versus 76.3% for a baseline trained on 10,000 examples without rationales for 5,000 steps; few-shot 2-digit accuracy is below 1% (§4.3). With rationalization, 2-digit accuracy reaches 32% after one iteration; those runs start at 300 steps and add 20 steps per iteration (§4.3).
- **CQA.** 9,741 train / 1,221 dev / 1,285 test questions, five choices (§4.2). Table 1 (dev accuracy, % of train data used): few-shot direct GPT-J 20.9; few-shot CoT GPT-J 36.6; few-shot CoT LaMDA 137B 55.6; GPT-J direct fine-tuned 60.0 (100); STaR without rationalization 68.8 (69.7); STaR with rationalization 72.5 (86.7 = 78.2 from generation + 8.5 from rationalization); GPT-3 direct fine-tuned 73.0 (100).
- **Few-shot prompts during fine-tuning.** CQA 60.9% → 68.8% without rationalization and 69.9% → 72.5% with it (§4.4).
- **Human evaluation.** 20 Prolific crowdworkers each ranked rationales for 10 of 50 sampled questions; STaR rationales were 30% more likely to be ranked above few-shot CoT rationales (p = .039) and 74% more likely above human-written rationales (p < .001) (§4.4).
- **GSM8K.** 7,473 train / 1,319 test (§4.2). Table 2 (test accuracy, % train data): few-shot direct 3.0; few-shot CoT 3.1; direct fine-tuned 5.8 (100); STaR without rationalization 10.1 (25.0); with rationalization 10.7 (28.7, of which 0.5 from rationalization). Training steps were capped at the 30th iteration (after 7,912 steps); results came after 36 iterations without rationalization plus 10 with it (§4.5).
- **Hyperparameters (App. H).** Batch size 8 sequences of length 1024 with packing; no weight decay; Adam; learning-rate search from 10⁻⁷ to 10⁻⁴, with 10⁻⁶ "consistently the best"; training and sampling on a single TPU-v3 node.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT-J + STaR (all tasks) | 6B | SFT | optimizer; peak LR; schedule | Adam; 10⁻⁶; 100-step warmup then constant | arXiv:2203.14465v2 App. H; §4.1 | verified 2026-09-14 | App. H: search 10⁻⁷-10⁻⁴, 10⁻⁶ best (no table) |
| GPT-J + STaR (all tasks) | 6B | SFT | batch; sequence length; packing; weight decay | 8 sequences; 1024 tokens; packing on; none | App. H | verified 2026-09-14 | no ablation reported |
| GPT-J + STaR (all tasks) | 6B | SFT | initialization per outer loop | original pretrained GPT-J | §3.1; Alg. 1 line 7 | verified 2026-09-14 | "to avoid overfitting" (§3.1); no ablation reported |
| GPT-J + STaR (default, CQA) | 6B | SFT | steps per outer loop | 40 in the first loop, +20% per loop | §4.1 | verified 2026-09-14 | "training more slowly at the beginning ultimately benefits model performance" (§4.1; no table) |
| GPT-J + STaR, arithmetic with rationalization | 6B | SFT | steps per outer loop | 300 at start, +20 per iteration | §4.3 | verified 2026-09-14 | same start without rationalization overfits 1-digit addition (§4.3) |
| GPT-J + STaR, arithmetic | 6B | SFT | problems per iteration | 10,000 sampled from 50,000 | §4.1 | verified 2026-09-14 | no ablation reported |
| GPT-J + STaR (all tasks) | 6B | SFT | rationale sampling | greedy decoding | §3.1 | verified 2026-09-14 | §5: temperatures 0.5 or 0.7 "consistently led to models worse" |
| GPT-J + STaR, GSM8K | 6B | SFT | iterations; step cap | 36 without + 10 with rationalization; steps capped at 30th iteration (7,912 steps) | §4.5 | verified 2026-09-14 | Table 2 |
| GPT-J + STaR (all tasks) | 6B | eval-gate | stopping and reporting rule | run until performance saturates; report best results | §4.1 | verified 2026-09-14 | not applicable |
| GPT-J + STaR (all tasks) | 6B | SFT | epochs; grad clip; tokens seen | not reported (checked §3-5, App. H) | — | not reported | — |

## Findings relevant to generality and negative feedback
- **Negative samples.** Wrong-answer rationales are discarded; the indicator in Eq. 2 gives them zero gradient (§3.1). Rationalization does not train on the failed attempt: it samples a new rationale conditioned on the gold answer, keeps it only if it reaches that answer, and trains it as a positive target without the hint (§3.2; Alg. 1 line 6). No term in Algorithm 1 lowers the likelihood of any sample.
- **Measured value of rationalization.** CQA 68.8 → 72.5 (Table 1); GSM8K 10.1 → 10.7, described as "does not substantially improve" (§4.5); on arithmetic it lets the model learn several digit lengths at once instead of stage by stage (§4.3, Fig. 4).
- **Out-of-distribution length.** After extra digits were added to training, the model solved "many" 9- and 10-digit problems never seen in training; training became less stable (§4.3, Fig. 5). No accuracy number is given.
- **Sampling temperature and generalization.** Higher-temperature sampling raised the rate of correct answers reached with incorrect reasoning, and the authors state that training on such reasoning "prevents generalization" (§5).
- **Limits.** Few-shot accuracy must be above chance; GPT-2 could not bootstrap even on arithmetic (§6). STaR amplifies dataset biases that help solve the task, and rationalization increases this; rationale faithfulness is not ensured (App. G).
- **Measurement.** CQA results are dev-set accuracies (Table 1), and the protocol reports the best result after saturation (§4.1).

## Connections
- [[quiet-star]] — same first author; generalizes STaR from question answering to rationales at every token of web text.
- [[v-star]] — trains a verifier to guide generation on top of STaR-style self-training (summarized in Quiet-STaR §2.2).
- [[rest-em]], [[rejection-sampling-finetuning]] — later generate-filter-fine-tune methods (course links).
- [[training-verifiers-to-solve-math-word-problems]] — source of GSM8K, used in §4.5.
- [[agent-early-experience]], [[distilling-step-by-step]] — later papers that cite STaR (see those cards).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2203.14465 (arXiv v2, 2022-05-20; section numbers refer to v2).
- Corrections to the previous card version:
  - "rationalization turns a wrong attempt into useful training signal" → the failed attempt receives no gradient; rationalization generates a new hint-conditioned rationale that is trained as a positive target (§3.1-3.2; Alg. 1).
  - "Generate a rationale ... for an unlabeled training example" → training examples have gold answers but no rationales (§3.1).
  - "Finetune the model on all accepted rationale traces" → each iteration fine-tunes the original pretrained model M (§3.1; Alg. 1 line 7).
  - "closer to iterative SFT ... than to RL" → the authors present STaR as an approximation of a policy-gradient objective (§3.1).
  - "approach the performance of much larger models on reasoning benchmarks" → only on CQA: 72.5% vs 73.0% for a 30× larger fine-tuned GPT-3 (Table 1); on GSM8K no larger-model comparison is made (Table 2).
  - Header dep [[self-instruct]] removed: STaR does not build on it.
- Removed as unsupported by the source: "more sample-efficient than waiting for a giant human rationale corpus"; "the bottleneck shifts to design a good verifier / answer checker"; "works best when final answers are cheaply verifiable" (replaced by the §6 limits); descendant claim for [[deepseek-r1]] and [[qwen-3]].
- Not reported by the source: venue (none in arXiv v2), epochs, gradient clipping, total tokens, accuracy on the 9-10 digit OOD problems.
