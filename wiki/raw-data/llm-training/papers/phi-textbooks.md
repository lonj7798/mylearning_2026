<!-- scope: phi-1 synthetic "textbook-quality" pretraining corpus for code
     deps: [[rephrasing-the-web]]
     see-also: [[phi-1-5]], [[nemotron-4-synthetic]], [[hf-cosmopedia]]
-->

# Textbooks Are All You Need (phi-1)
- **Core Insight:** Pretraining loss is *quality*-bounded long before it is *quantity*-bounded; swapping a noisy web crawl for a small "textbook-quality" corpus (filtered real + GPT-3.5-synthesized textbooks + exercises) yields a 1.3B model that rivals models ~10× bigger trained on ~100× more tokens.
- **Guideline:** When training small specialist LMs (e.g. code), invest compute upstream in curating ≤10B "textbook-quality" tokens (filter web for pedagogical density, then synthesize gap areas) rather than scaling to the next OOM of generic web.
- **Authors:** Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Kalai, Yin Tat Lee, Yuanzhi Li (Microsoft Research)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2306.11644
- **Relevant topics:** synthetic pretraining, code LLMs, data quality vs quantity, textbook-quality

## Abstract
phi-1 is a 1.3B-parameter Transformer trained for 4 days on 8×A100 using ~7B total tokens of "textbook-quality" data: 6B from filtered web (a classifier flags code that reads like a pedagogical example) plus ~1B synthetically generated textbooks and exercises from GPT-3.5. Fine-tuned on additional synthetic code exercises, phi-1 reaches **50.6% pass@1 on HumanEval** and **55.5% on MBPP** — competitive with models ~10× its size trained on >100× the tokens. The paper kicked off the Phi line and the broader "synthetic pretraining" research thread.

## Key Contributions
- Introduced a **two-axis data curation strategy**: filter real web for pedagogical density, synthesize the rest.
- Trained a classifier that scores web code for "textbook-likeness" (the filter prompt itself is a recurring reference for later data-quality classifiers).
- Demonstrated substantially-better-than-scaling-laws behavior on code benchmarks with ~100× less data.
- Ignited the Phi line (phi-1.5, phi-2, phi-3, phi-4) and inspired synthetic-pretraining work like WRAP, Cosmopedia, Nemotron-CC.

## Key Figures/Tables to Study
- **Figure 2.1** — data-quality ablation: filtered-only vs filtered+synthetic vs raw web.
- **Table 1** — HumanEval / MBPP pass@1 vs model size comparisons (phi-1 1.3B vs StarCoder 15B etc.).
- **Synthetic-textbook example** — a representative generated textbook page illustrating the style targeted.

## Synthesis pipeline (REQUIRED — be concrete)
- **Seed input:**
  - "The Stack" code corpus (deduplicated) + StackOverflow — raw source.
  - A hand-labeled seed of "educational" vs "non-educational" code snippets used to train a classifier.

- **Generation step(s):**
  - **Filtered-web (6B tokens):** random-forest classifier (features from a small LM's embeddings) scores each snippet for pedagogical value; keep top-scoring.
  - **Synthetic textbooks (~1B tokens):** prompt GPT-3.5 to write Python-centric textbook pages on specified topics. Prompt framework specifies topic, target audience, and requested exposition style; topic list covers the breadth of Python stdlib, control flow, algorithms, data processing.
  - **Synthetic exercises (~180M tokens, used for the SFT stage `phi-1`):** GPT-3.5 generates `<problem, solution>` pairs on topics not well covered in the filtered web.

- **Filtering/rescoring:** decontamination against HumanEval/MBPP (n-gram match); dedup; pedagogical classifier re-ranking.

- **Output shape:**
  - 6B tokens filtered web + 1B synthetic textbook for `phi-1-base`.
  - +180M tokens synthetic exercise for `phi-1` SFT.
  - Not publicly released — reproductions exist (see Phi-Data / Cosmopedia).

- **Teacher model(s):** GPT-3.5 (text-davinci-003 era) for textbook + exercise generation.

- **Cost estimate:** 4 days × 8×A100 = ~800 GPU-hours training; teacher API cost not disclosed but synthetic corpus is small enough that <$100K is the likely ballpark.

## Quality / diversity evaluation
- **HumanEval:** 50.6% pass@1 (1.3B model, vs ~34% for StarCoder-15B).
- **MBPP:** 55.5% pass@1.
- Ablations show the jump is driven primarily by the synthetic textbook + exercise data, not by model architecture.

## Risks + gotchas
- **HumanEval contamination risk:** later analyses flagged non-trivial overlap between synthetic exercises and HumanEval-style prompts — a perpetual critique of the Phi line.
- **Teacher-model ceiling:** the student inherits GPT-3.5's code-style tics and mistakes.
- **Narrow domain:** strong for Python; generalization tested later in [[phi-1-5]] — results more nuanced outside code.
- **Not released data:** reproducibility is limited; open re-implementations (Cosmopedia) diverge.

## Connections
- Direct ancestor of [[phi-1-5]] (extending textbook synthesis beyond code to reasoning) and the Phi-3/Phi-4 post-training line.
- Conceptual sibling of [[rephrasing-the-web]] (both trade raw tokens for curated/rewritten tokens).
- Foundation for [[hf-cosmopedia]] (open reproduction of phi-style synthetic pretrain).
- Points to [[model-collapse]] tensions: Phi explicitly avoids recursive training on its own outputs by using a stronger teacher (GPT-3.5) as the single upstream source.
