<!-- scope: Allen-Zhu & Li (2024), Physics of LMs Part 3.3: controlled synthetic-biography experiments that measure stored factual knowledge in bits per parameter, and how exposures, architecture, quantization, MoE, and junk data change it
     see-also: [[data-constrained-scaling]], [[scaling-laws-data-quality]], [[rephrasing-the-web]], [[phi-1-5]], [[physics-of-lm-3-recipe]]
-->

# Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws
- **Core Insight:** In controlled synthetic-knowledge experiments, GPT2 models from 1M to 0.5B parameters that see each fact 1000 times store at least 2 bits of knowledge per parameter, no model exceeds 2.3, and int8 quantization leaves capacity unchanged; with 100 exposures the ratio falls to about 1 bit per parameter (Abstract; Results 1, 4, 8).
- **Guideline:** When pretraining data mixes knowledge-rich text with text that carries little useful knowledge, prepend a source tag (for example a domain name) to the knowledge-rich data, because with 7/8 junk tokens this reduced the capacity loss at 100 exposures from 20x to 2x (Results 10, 12). This was measured on synthetic biographies with GPT2 models, not on web corpora. When tags cannot be added, more exposures of the useful data also reduce the loss, to 3x/1.5x/1.3x at 300/600/1000 exposures (Result 10), and the 1000-exposure run with junk takes 80x longer to train than the 100-exposure run without junk (Figure 8 remarks).
- **Authors:** Zeyuan Allen-Zhu (Meta FAIR Labs), Yuanzhi Li (Mohamed bin Zayed University of AI)
- **Year:** 2024 (arXiv v1 2024-04)
- **URL:** https://arxiv.org/abs/2404.05405
- **Source type:** paper
- **Relevant topics:** knowledge capacity, scaling laws, factual knowledge storage, exposures vs epochs, data quality, domain tagging, quantization, MoE, architecture comparison

## Abstract
Earlier scaling-law studies measure capability with loss or benchmarks. This paper instead estimates the number of knowledge bits a model stores, for factual knowledge written as tuples such as (USA, capital, Washington D.C.). Across several controlled datasets, language models can store 2 bits of knowledge per parameter and no more, even after int8 quantization, and the knowledge can be extracted for downstream use. At this ratio a 7B model can store 14B bits, which the authors estimate exceeds English Wikipedia and textbooks combined. The paper reports 12 results on how training duration, model architecture, quantization, sparsity (MoE), and data signal-to-noise ratio affect capacity. Two highlighted results: GPT-2 with rotary embedding matches or exceeds LLaMA/Mistral in knowledge storage, especially with shorter training, which the authors attribute to GatedMLP being less stable and harder to train; and prepending domain names such as wikipedia.org to training data increases knowledge capacity, because the model identifies and prioritizes knowledge-rich domains without being told which ones they are.

## Key Contributions
- A bit-complexity lower bound (Theorem 3.2) that converts a trained model's name and value losses into a minimum number of stored bits.
- Controlled datasets: bioS, bioS_simple, bioR (LLaMA2 rewrites), and the parameterized bioD (§2.2, Definition 2.2).
- Base scaling laws: 2 bit/param at 1000 exposures and 1 bit/param at 100 exposures (Results 1–4).
- Architecture, quantization, and MoE comparisons under the same metric (Results 5–9).
- Junk-data experiments and a domain-token mitigation (Results 10–12).

## Key Figures/Tables to Study
- **Figure 1** — GPT2 scaling laws on bioS(N) at 1000 and 100 exposures.
- **Figures 3–5** — LLaMA, Mistral, reduced-MLP and no-MLP GPT2; gated-MLP ablation.
- **Figure 8** — capacity with 7/8 junk data, repetitive data, and a prepended special token.
- **Figure 10** — memorizable vs extractable knowledge accuracy.
- **Figure 11** — bioS_simple (no sentence diversity) and bioR (LLaMA2 rewrites).

## Technical Details
- **Knowledge piece:** a (name, attribute, value) tuple (§2). bioS(N) has N people with six attributes (birth date, birth city, university, major, employer, working city), 50 templates per attribute, re-sampled for each biography; bioS_simple fixes one biography per person; bioR has each biography written 40 times by LLaMA2, with N up to 1M and 22GB of text (§2.2, App. A.3). bioS reaches N = 20M, about 1B bits (App. A).
- **Exposure vs pass:** an exposure is one occurrence of a knowledge piece in training. For bioS, 1000 exposures take less than one pass; for bioS_simple they take 1000 passes; for bioR, 1000/100 exposures take 25/2.5 passes (§2.2, §5.1).
- **Capacity ratio (Definition 4.3):** R(F) = [N·log2(N0 / e^p1) + N·log2(S0 / e^p2)] / P, and R_max(F) = [N·log2(N0 / N) + N·log2(S0)] / P.
  - N: number of people; N0 = 400 × 400 × 1000 possible names; S0 = 2 × (12·28·200) × 200 × 300 × 100 × 263 possible attribute combinations (§4, footnote 9).
  - p1: name loss, the cross-entropy summed over name tokens and averaged over people; p2: value loss summed over a person's attributes and averaged over people (§3, footnote 12).
  - P: parameter count, excluding embedding rows for tokens that never occur (GPT2-small counts as 88M, not 124M) (App. A).
- **Worked example:** a 100M-parameter model storing 220M bits has R = 220M / 100M = 2.2 bits per parameter (§1). At 2 bits per parameter, 7B parameters give 14B bits (Remark 1.1). Each bioS person carries log2(S0) ≈ 47.6 bits excluding the name (Remark 4.4). An int8 model cannot exceed R = 8 (Remark 4.2).
- **Results 1–3:** at 1000 exposures, N = 10K–10M, peak R ≥ 2; models with R_max ≤ 1.8 reach near-perfect accuracy; all models have R ≤ 2.3 (§5, Figure 1(a)). bioS_simple and bioR give about 2, slightly lower (Figure 11). bioD with K, C from 1 to 50, D from 10 to 10,000, L from 1 to 50, T from 20 to 40,000 gives R ≥ 2 (§5.2, Figure 2).
- **Result 4:** at 100 exposures, peak R ≥ 1, a loss of no more than 2x (§6, Figure 1(b)).
- **Results 5–7:** at 1000 exposures LLaMA, Mistral, GPT2 with a 1/4-size MLP, and GPT2 with no MLP all follow the 2 bit/param law; tiny LLaMA models (< 10M) are slightly worse unless embedding and output weights are tied (§7, Figure 3). At 100 exposures LLaMA and Mistral are 1.3x worse than GPT2 even with tuned learning rates, and removing all MLPs costs more than 1.5x; for larger LLaMA models, replacing the gated MLP with a standard MLP is needed to match GPT2 (§7.1, App. B.2, Figures 4–5).
- **Result 8:** post-training GPTQ quantization (auto_gptq) to int8 has negligible effect; int4 reduces capacity by more than 2x (§8), stated as 0.7 bit/param in §1.
- **Result 9:** a GPT2-MoE with 32 experts (topk = 1, cap_factor = 2) loses 1.3x (1000 exposures) and 1.5x (100 exposures) of capacity measured against total parameters, while using 8.8% of total parameters per token at inference (§1, §9).
- **Results 10–12:** with 1/8 of tokens from bioS(N) and 7/8 from random bioS(100M) junk, capacity drops 20x at 100 exposures and 3x/1.5x/1.3x at 300/600/1000 exposures, compared with 100 exposures without junk (§10, Figure 8(a)–(e)). Junk from a highly repetitive bioS(1K) causes no drop (Figure 8(f)). Prepending a special token to every useful biography limits the drop to 2x at 100 exposures, and at 300 exposures it matches the no-junk 100-exposure law (Figure 8(g)–(h)).
- **Extraction test:** LoRA finetuning on question-answer pairs for half of the people, tested on the other half; the scaling law also holds for extractable knowledge, with a 1.2x accuracy drop only for models at the capacity boundary (App. A.2, Figure 10).

## Recipe ledger
The paper discloses per-dataset optimizer settings for every experiment (App. A–E, Parameters 1–10). The ledger is in [[physics-of-lm-3-recipe]]. It covers the shared protocol, the GPT2 base scaling-law runs, the extraction finetune, and the MoE and junk-data setups.

## Findings relevant to generality
- **Diversity of phrasing:** training on 1000 differently templated biographies per person (bioS) gives slightly higher capacity than 1000 passes over one fixed biography (bioS_simple) (Result 2, App. A.3). The authors state that without data diversity "the model wastes capacity memorizing sentence structures" (§5.1). In low-diversity data, knowledge can be memorized word by word but is nearly 0% extractable, as found in Part 3.1 (footnote 16, App. A.3).
- **LLM rewrites:** 40 LLaMA2 rewrites per person (bioR) keep the capacity ratio near 2 bit/param, slightly lower because the rewrites add irrelevant details (§5.1, App. A.3).
- **Rare knowledge:** facts seen 100 times are stored at about 1 bit/param, half the 1000-exposure rate (Remark 1.3).
- **Data quality:** with random junk data, capacity for useful data seen 1000 times is still 1.3x below that of a run without junk at 100 exposures (Result 10). A tag on high-quality sources lets the model detect and prioritize them without prior knowledge of which domains are useful (Result 12, §10).
- **Measurement:** the authors argue that loss or perplexity comparisons can give "debatable conclusions" and that synthetic data avoids benchmark contamination (Remark 1.4, §1). Their GatedMLP example shows no effect for facts seen 1000 times but an effect for facts seen 100 times (footnote 6).
- **Scope:** all results use synthetic or semi-synthetic biography data and GPT2-scale models up to 0.5B in Results 1 and 4 (§5, §6). The authors describe knowledge tuples as "a restricted, yet sufficiently interesting domain" (§1).

## Connections
- [[data-constrained-scaling]] — studies repeated epochs over limited data; this paper counts exposures per fact instead of passes (§2.2).
- [[scaling-laws-data-quality]] — models data quality as a scaling variable; Results 10–12 give a controlled data signal-to-noise measurement.
- [[rephrasing-the-web]] — rewrites pretraining text with an LLM; Result 2 reports that LLaMA2 rewrites keep capacity near 2 bit/param (§5.1).
- [[phi-1-5]] — cited as [24] for the claim that much internet data lacks valuable knowledge for training (§1).
- Parts 3.1 (arXiv:2309.14316) and 3.2 (arXiv:2309.14402), which have no cards in this library, define the extraction and manipulation tests this paper relies on (references [3], [4]).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2404.05405 (arXiv v1, 8 Apr 2024; version 1 is the only version returned)
- Corrections to the previous card version:
  - Core Insight "Scaling laws should track how much factual knowledge a model can store ... not only loss" → the measured result, 2 bits per parameter at 1000 exposures, about 1 bit at 100, unchanged under int8 (Abstract, Results 1, 4, 8).
  - The abstract omitted every result → rewritten from the paper's abstract.
  - "Measures how much knowledge survives storage and can be flexibly queried" → the bit-complexity lower bound (Theorem 3.2) plus a LoRA question-answer extraction test (App. A.2).
  - Authors listed without affiliations → Meta FAIR Labs and Mohamed bin Zayed University of AI (title page).
- Removed as unsupported by the source: "Practical implication: data usefulness depends on the model's capacity to absorb distinct facts, not only exposure volume" (not stated in this form; replaced by Results 4 and 10–12); "useful conceptual background for interpreting data saturation and repetition effects" (course interpretation).
- Not reported by the source: experiments on real web corpora, benchmark-level effects of domain tags, reasoning tasks, models above about 1B parameters, venue of publication.
