<!-- scope: phi-1.5 extension of textbook-synthesis to common-sense + reasoning
     deps: [[phi-textbooks]]
     see-also: [[rephrasing-the-web]], [[hf-cosmopedia]]
-->

# Textbooks Are All You Need II: phi-1.5 Technical Report
- **Core Insight:** The textbook-quality recipe generalizes beyond code — synthesizing ~20B tokens of GPT-3.5-authored "textbook-like" content around a 20K-topic taxonomy lets a 1.3B model match 10×-larger models trained on 10×-more tokens on common-sense reasoning.
- **Guideline:** For broadening a small LM beyond a narrow skill, pick a ~20K-topic taxonomy covering the target capability and generate synthetic textbook-style corpora per topic; mix with the prior code-heavy phi-1 data.
- **Authors:** Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, Yin Tat Lee (Microsoft Research)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.05463
- **Relevant topics:** synthetic pretraining, reasoning, common-sense, Phi line

## Abstract
phi-1.5 keeps phi-1's 1.3B Transformer architecture but retrains on the union of phi-1's original 7B tokens and **~20B newly generated synthetic "textbook-like" tokens** produced by GPT-3.5 around a carefully curated list of 20,000 topics. The model matches or beats models up to 10× its size on reasoning benchmarks (WinoGrande, ARC, HellaSwag, BoolQ, MMLU-subset) and grade-school math, despite the corpus being almost entirely synthetic. It is the first public demonstration that the textbook-quality hypothesis extends past code into general reasoning.

## Key Contributions
- Extended the phi-1 "textbook" idea from Python to common-sense knowledge + basic math/logic.
- Curated a **20K-topic seed list** as the generation axis for diversity (an early precursor of taxonomy-driven synthesis — see [[glan]]).
- Demonstrated near-frontier scores on reasoning benchmarks from a 1.3B model trained on ~27B tokens total (vast majority synthetic).
- Released phi-1.5 weights (but not the training data) — catalyzed independent reproductions.

## Key Figures/Tables to Study
- **Table 1** — phi-1.5 vs Llama-2-7B / Falcon-7B / Vicuna-13B on WinoGrande, ARC-Easy, ARC-Challenge, BoolQ, SIQA, HellaSwag, PIQA, OpenbookQA.
- **Ablation table** — phi-1.5 vs phi-1.5-web (with filtered web added): the ablation shows synthetic-only is almost as strong as synthetic+web, and better on reasoning.
- **MMLU-subset and grade-school math curves.**

## Synthesis pipeline (REQUIRED — be concrete)
- **Seed input:**
  - phi-1's original data (7B tokens: filtered code web + ~1B synthetic code textbook).
  - A hand+LLM-curated **20,000-topic seed list** covering common-sense, grade-school science, basic logic, everyday reasoning.

- **Generation step(s):**
  - For each topic, prompt GPT-3.5 to produce "textbook-like" expository passages — the prompt specifies audience (general reader / student), style (expository), and desired length.
  - Multiple passes per topic with varied sub-angles to avoid single-narrative mode collapse.
  - Exact prompt templates not released, but described as templated expansions over the topic list (the direct ancestor of [[glan]]'s taxonomy-driven expansion).

- **Filtering/rescoring:** deduplication; benchmark decontamination; no heavy classifier filter beyond what GPT-3.5 produces.

- **Output shape:**
  - ~20B synthetic tokens + 7B from phi-1 = ~27B training tokens.
  - Released model weights; training data not released.

- **Teacher model(s):** GPT-3.5.

- **Cost estimate:** not disclosed; 20B tokens of GPT-3.5 synthesis at 2023 API pricing ~hundreds of thousands USD.

## Quality / diversity evaluation
- Matches or beats Llama-2-7B on WinoGrande, ARC, PIQA, HellaSwag despite being 5× smaller.
- Competitive on grade-school math (~40% on GSM8K).
- Ablation: removing synthetic common-sense portion drops reasoning benchmarks sharply, confirming synthetic is load-bearing.

## Risks + gotchas
- **Topic taxonomy leakage:** the 20K topic list and benchmark coverage overlap is a perennial critique.
- **Teacher bias:** inherits GPT-3.5 factual errors and stylistic quirks.
- **Data not public:** phi-1.5 is not a reproducible open-source artifact.
- **Reasoning benchmarks saturate:** phi-1.5 numbers are on tasks that later-released models saturated; claims of "reasoning parity" are weaker on harder (MMLU-Pro, MATH) benchmarks.
- Contamination concerns carry over from phi-1.

## Connections
- Direct successor to [[phi-textbooks]] (phi-1).
- Taxonomy-based expansion idea is formalized in [[glan]].
- The "synthetic-only pretraining works" claim anchors [[hf-cosmopedia]]'s reproduction.
- Counter-evidence to pessimistic readings of [[model-collapse]]: a single-shot teacher generation + careful topic scaffolding avoids the recursive degradation loop.
