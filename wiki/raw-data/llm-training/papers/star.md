<!-- scope: iterative rationale bootstrapping from answer supervision
     deps: [[self-instruct]]
     see-also: [[quiet-star]], [[lets-verify]], [[training-verifiers-to-solve-math-word-problems]]
-->

# STaR: Bootstrapping Reasoning With Reasoning
- **Core Insight:** If a model can occasionally produce a correct chain of thought, you can turn sparse answer supervision into a growing rationale dataset by keeping successful traces and "rationalizing backward" from the correct answer when it fails.
- **Guideline:** Use STaR when final answers are verifiable but gold rationales are scarce: few-shot prompt for rationale+answer, keep correct traces, regenerate a rationale conditioned on the gold answer for failures, finetune on the accepted traces, and repeat.
- **Authors:** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2203.14465
- **Relevant topics:** rationale bootstrapping, self-improvement, chain-of-thought training, synthetic reasoning data

## Abstract
STaR proposes an iterative method for teaching language models to reason without a large human-written rationale corpus. Starting from a few rationale exemplars and a larger dataset that only has final answers, the model generates rationales and answers, keeps the rationales that lead to correct answers, and for failed attempts tries again while conditioning on the known correct answer. Finetuning on the resulting successful rationales lets the model gradually improve its own reasoning ability.

## Key Contributions
- Introduces a simple self-training loop for reasoning: generate, verify, repair, finetune, repeat.
- Shows that rationale quality can be bootstrapped from answer labels alone rather than from a large gold-CoT dataset.
- Uses backward rationalization on failed examples, which turns a wrong attempt into useful training signal instead of discarding it.
- Demonstrates that a model trained this way can approach the performance of much larger models on reasoning benchmarks.

## Key Figures/Tables to Study
- **Figure 1:** the STaR loop; this is the canonical diagram for rationale bootstrapping.
- **Abstract + main results table:** the key empirical claim is not just accuracy gain, but matching a much larger finetuned model on CommonsenseQA.
- **Method section around the retry step:** this is where the "given the correct answer, explain why" trick becomes concrete.

## Technical Details

### STaR loop
1. Prompt the base LM with a small number of rationale examples.
2. Generate a rationale and final answer for an unlabeled training example.
3. If the answer is correct, keep the rationale as synthetic supervision.
4. If the answer is wrong, prompt again while providing the correct answer and ask the model to produce a rationale that reaches it.
5. Keep only rationales that now yield the correct answer.
6. Finetune the model on all accepted rationale traces and repeat the loop.

### Why the method matters
- It converts **answer-checkable tasks** into **rationale-learning tasks**.
- It is more sample-efficient than waiting for a giant human rationale corpus.
- It is one of the earliest clean demonstrations that models can improve by training on their own successful intermediate reasoning.

### Practical implications
- STaR works best when final answers are cheaply verifiable.
- The method is closer to **iterative SFT on filtered self-generated traces** than to RL.
- The bottleneck shifts from "collect gold CoT" to "design a good verifier / answer checker."

## Connections
- [[self-instruct]] is the instruction-data analogue of STaR: both bootstrap supervision from model outputs.
- [[quiet-star]] generalizes the idea from question-answer settings to arbitrary text continuation.
- [[lets-verify]] and [[training-verifiers-to-solve-math-word-problems]] supply the verifier side that makes reasoning bootstraps more reliable.
- Modern reasoning pipelines in [[deepseek-r1]] and [[qwen-3]] can be read as larger-scale descendants of the same self-improvement instinct.
