---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2110.14168v2 (Training Verifiers to Solve Math Word Problems); library card [[training-verifiers-to-solve-math-word-problems]]
source_url: https://arxiv.org/abs/2110.14168
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because the library card has no Verification section)"
---

# Excerpt: outcome verifiers on GSM8K (Cobbe et al.)

Used by [[read]] Guideline, §2.4, §5.3, the negatives section, the Recipe, and Common mistakes. Checked against arXiv v2 (2021-11-18) on 2026-09-15.

## Verifier recipe (§4.2, Fig. 4)
1. Fine-tune a generator for 2 epochs on the training set.
2. Sample 100 completions per training problem and label each correct or incorrect by the final answer.
3. Train a verifier for a single epoch on this dataset; the verifier also keeps the language-modeling objective as an auxiliary loss.
- "Training solutions are labeled as correct or incorrect based solely on whether they reach the correct final answer. In practice, some solutions will reach the correct final answer using flawed reasoning, leading to false positives."
- At test time, 100 completions per problem are ranked and the top one is returned.

## Coverage (§4.1, Fig. 3)
- Test@1 and test@100 are measured over 100 epochs for a 6B model; "test@100 performance degrades much more sharply than test@1 as we increase the number of epochs"; overconfidence "leads to poor coverage of the solution space".
- "Choosing a model with good coverage is critical to successfully train verifiers"; test@100 peaks within the first few epochs, hence the 2-epoch generator.

## Results
- Verification gives about the same boost as a 30× model size increase over fine-tuning and scales better with data; it is not beneficial at small dataset sizes (§1, §4.2, Fig. 5).
- Token-level verifiers outperform solution-level ones late in training; the joint LM objective is a strict improvement (Fig. 6a-b).
- A large generator with a small verifier beats a small generator with a large verifier (Fig. 6c).
- With a 6B verifier, performance improves up to 400 completions per test problem and then decreases, which the authors attribute to "finding adversarial solutions that fool the verifier" (§5.1, Fig. 7a).
