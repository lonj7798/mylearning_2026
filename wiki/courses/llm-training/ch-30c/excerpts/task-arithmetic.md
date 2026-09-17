---
chapter: ch-30c
course: llm-training
phase: read
excerpt_of: "Editing Models with Task Arithmetic (Ilharco et al.)"
source_url: https://arxiv.org/abs/2212.04089
created_at: "2026-09-15"
---

# Excerpt: Task arithmetic

This excerpt stands in for the library card `task-arithmetic`, which did not exist when ch-30c was written.
Every number below was read in arXiv:2212.04089v3 at the stated locus.

- **Authors:** Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi
- **Year:** arXiv v1 2022-12; v3 2023-03-31; ICLR 2023
- **Source type:** paper

## Definitions (§2)
- Task vector: τ_t = θ_ft^t − θ_pre, where θ_pre are pre-trained weights and θ_ft^t the weights after fine-tuning on task t.
- Edited model: θ_new = θ + λ·τ_new, with λ "determined using held-out validation sets". λ = 1 with one task
  vector gives the fine-tuned model.
- Negation: τ_new = −τ. Addition: τ_new = Σ_i τ_i. Analogy: τ_new = τ_C + (τ_B − τ_A).

## Results
- **Negation, image (Table 1, CLIP, 8 target tasks, ImageNet control).** ViT-L/14: pre-trained 64.8 target / 75.5
  control; negative task vector 19.0 / 72.9; gradient ascent 3.93 / 16.3; random vector 60.9 / 72.9.
- **Negation, text (Table 2, GPT-2 Large; task vector from Civil Comments with toxicity > 0.8; 1,000 generations
  scored by Detoxify).** % toxic / avg toxicity / WikiText-103 perplexity: pre-trained 4.8 / 0.06 / 16.4;
  fine-tuned on toxic 57 / 0.56 / 16.6; gradient ascent 0.0 / 0.45 / >10¹⁰; fine-tuned on non-toxic 1.8 / 0.03 / 17.2;
  negative task vector 0.8 / 0.01 / 16.9.
- **Addition, image (§4.1, Figs. 2-3; 8 CLIP tasks; accuracy normalized by each task's fine-tuned model).** Pairs of
  task vectors average 98.9% normalized accuracy; the best model from all subsets reaches 91.2% (Fig. 3).
  Joint multi-task fine-tuning on the eight tasks reaches 0.994 (App. D.2). The best subset is often not all
  task vectors (App. D).
- **Addition, text (Table 3, T5-base on 4 GLUE tasks, 427 Hub checkpoints searched, best chosen on validation).**
  Fine-tuned average 78.1 → fine-tuned + task vectors 78.6 (MRPC +0.8, RTE +0.2, CoLA +0.7, SST-2 +0.2).
- **Analogies (Table 4).** Yelp target, T5-base: fine-tuned on Amazon 92.3, task analogy 93.0, fine-tuned on Yelp 93.4.

## Discussion (§6, App. D.3)
- Cosine similarity between task vectors of different CLIP tasks is 0.01-0.06 for most pairs and 0.18 for
  MNIST-SVHN (Fig. 5). The authors "speculate" that near-orthogonality lets addition proceed with little
  interference (Interpretation).
- Higher fine-tuning learning rates reduce accuracy of task-vector edits more than accuracy of individual
  models; the authors recommend caution with large learning rates (Fig. 6).
- λ in 0.3 to 0.5 gives close to optimal results in many cases; the authors still recommend tuning (App. D.3).
- Limits: same architecture required; all experiments use models fine-tuned from the same pre-trained
  initialization (§6).

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2212.04089v3 (PDF), §1-§6 and App. D.
