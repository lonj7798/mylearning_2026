<!-- scope: 01.AI technical report for the Yi 6B/34B family, covering pretraining data engineering, a <10K-example SFT set, 200K long-context extension, vision-language and depth-upscaled variants
     deps: [[fineweb]], [[lima]]
     see-also: [[qwen-2.5]], [[llama-3]], [[phi-3]], [[glm-4]], [[retrieval-head]]
-->

# Yi: Open Foundation Models by 01.AI
- **Core Insight:** With a standard LLaMA-style decoder Transformer and 3.1T filtered English-Chinese pretraining tokens, Yi-34B reaches 76.3 MMLU against GPT-3.5's 69.1 in the report's own pipeline (Table 2), and the chat model is aligned with fewer than 10K hand-verified instruction dialogs (Abstract, §3.1).
- **Guideline:** When the pretraining corpus can be filtered and deduplicated aggressively, train past the Chinchilla-optimal token count at a smaller parameter count, because the report's 34B model trained on 3.1T tokens had not saturated at that budget (§8 takeaway 1); when the alignment target is human preference rather than a benchmark, a curated set under 10K dialogs is the setting the report used and evaluated (§3.1, Table 5).
- **Authors:** 01.AI (team report; the individual contributor list is in Appendix A)
- **Year:** 2024 (arXiv v1 2024-03; v3 2025-01-21 used here)
- **URL:** https://arxiv.org/abs/2403.04652
- **Source type:** official technical report
- **Relevant topics:** bilingual pretraining, data filtering and deduplication, small-scale SFT, long-context continual pretraining, recitation-style synthetic document QA, depth upscaling, vision-language extension

## Abstract
The Yi family is built on 6B and 34B decoder-only Transformers pretrained from scratch, then extended to
chat, 200K-context, depth-upscaled, and vision-language variants. The report attributes model quality
primarily to data engineering rather than architecture: 3.1T English and Chinese pretraining tokens from a
cascaded filtering and deduplication pipeline, and a finetuning set of fewer than 10K instruction dialogs
where every instance was verified by the authors' machine-learning engineers. Context is extended to 200K by
lightweight continual pretraining plus finetuning; depth upscaling of the 6B checkpoint yields Yi-9B.

## Key Contributions
- A pretraining pipeline combining heuristic filters, four learned scorers (perplexity, quality, safety,
  document coherence), semantic cluster filtering, and MinHash plus exact-match deduplication (§2.1).
- Post-Chinchilla overtraining: 34B parameters on 3.1T tokens, sized so the 4-bit-quantized chat model
  fits in 24GB of GPU memory (§1).
- SFT of fewer than 10K multi-turn dialogs with prompt evolution, a fixed response format, "Step-Back" CoT
  formatting, and InsTag-style tag-balanced sampling (§3.1).
- A 200K extension that changes no attention mechanism: full attention plus sequence parallelism and
  distributed attention (§4, §7.1).
- Recitation-and-rephrase synthetic document QA in both continual pretraining and SFT (§7.1).
- Depth upscaling from Yi-6B (32 layers) to Yi-9B (48 layers) by duplicating layers 12-28, with layer
  choice guided by per-layer input/output cosine similarity (§7.3, Figure 8).

## Key Figures/Tables to Study
- Table 1 (§2.3) model configs and per-size peak LR; Table 2 (§6.1.1) base-model benchmarks against GPT-3.5.
- Figure 3 (§6.1.3): in-context inference of linear coefficients, a scale-dependent capability probe.
- Figure 5 (§6.2.2): SFT data-scaling curves for Yi data versus UltraChat and UltraChat 200K.
- Table 6 (§7.1) MMLU before and after 200K adaptation; Table 8 (§7.3) Yi-6B vs Yi-9B-Init vs Yi-9B.

## Technical Details
- Yi-6B: hidden size 4096, 32 query heads, 4 KV heads, 32 layers. Yi-34B: hidden size 7168, 56 query heads,
  8 KV heads, 60 layers. Both pretrain at sequence length 4096 (Table 1). Architecture is the decoder-only
  Transformer with code based on the LLaMA implementation, using GQA at both sizes, SwiGLU with the
  post-attention activation size reduced from 4h to 8/3h, and RoPE with an adjusted base frequency (§2.3).
- Tokenizer: BPE via SentencePiece, vocabulary 64,000, numbers split into individual digits, byte fallback
  for rare characters, no dummy whitespace prefix (§2.2).
- Filtering starts from Common Crawl with CCNet language identification and perplexity scoring; the report
  claims a higher removal ratio than CCNet, RefinedWeb, and RedPajama but prints no removal percentage
  (§1, §2.1). Deduplication follows Penedo et al. (2023): document-level MinHash plus sub-document exact
  match; a topic model labels documents (news, ads, knowledge) and advertisement-like text is down-sampled
  (§2.1). Mixture proportions appear only as a pie chart, with no numeric token shares (Figure 2).
- SFT loss is next-token prediction on responses only, not on system or user turns; format is ChatML
  (§3.1, §3.2).
- SFT mixture ratios come from an approximate grid search over {1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64} per
  ability, guided by validation sets and in-house human evaluation (§3.1).
- Base results: Yi-34B 76.3 MMLU, 54.3 BBH, 81.4 C-Eval, 83.7 CMMLU, 40.8 Math; Yi-6B 63.2 MMLU, 42.8 BBH,
  72.0 C-Eval, 75.5 CMMLU; GPT-3.5 in the same table is 69.1 MMLU, 70.1 BBH (Table 2). Chat: Yi-Chat 34B
  scores 94.08 AlpacaEval, 1110 Chatbot Arena Elo, 71.87 SuperClue against GPT-4-Turbo 97.7 / 1243 / 89.79,
  cutoff 2023-12-21 (Table 5).
- 4-bit weight plus 8-bit KV-cache quantization: under 1% accuracy drop on MMLU/CMMLU (§4). No RLHF, DPO,
  or PPO run is described for the released chat models; both appear only as infrastructure features (§4).

### Long-context extension (§2.3, §7.1)
- The base models are pretrained at 4K and extended to 200K. The stated hypothesis is that the ability to
  model dependencies beyond 4K already exists in the base model and that continual pretraining releases
  rather than installs it (§2.3, §7.1).
- Continual-pretraining mixture: (1) original pretraining data, (2) length-upsampled long documents, mostly
  books, (3) multi-document QA synthetic data in which the answer recites the related paragraph before
  answering. Mixture proportions are not reported (§7.1).
- Long-context SFT mixes short-context SFT data with synthetic document QA: documents are randomly
  concatenated, one or more paragraphs are sampled, and a chat model writes question-answer pairs over the
  sampled paragraph. The answer must "recite or paraphrase the original paragraph" before answering, which
  the authors state encourages retrieval behavior and discourages hallucination (§7.1, Interpretation; no
  ablation against direct answers is reported).
- Attention at 200K is unmodified full attention; the extension is described as "solely based on
  engineering" (§4, §7.1).
- Short-context retention after 200K adaptation: Yi-6B 63.24 → 61.73 MMLU average, Yi-34B 76.32 → 75.56
  (Table 6). Needle-in-a-Haystack for Yi-34B-200K is shown only as a heatmap with no numeric score
  (Figure 6), and the authors call this level of retrieval "relatively easy for long-context LLMs" (§7.1).

## Recipe ledger
Moved to [[yi-recipe]] (line-limit split): pretraining, long-context CPT, SFT, and depth-upscaling rows.

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality (no benchmark targeting):** §6.2.1 states that benchmark evaluation is used only to confirm
  that alignment training did not damage base-model capability, and that the chat model is not optimized
  toward benchmark scores.
- **Generality (overfitting check):** the 2023 Hungarian high-school mathematics final exam is used as a
  held-out probe for overfitting to math-oriented training data. Yi-34B-Chat performs well on both GSM8K and
  the exam; Yi-6B-Chat is weak on both, which the authors attribute, as speculation, to smaller models
  needing more SFT data (§6.2.1, Figure 4).
- **Generality (scale-dependent capability):** with linear coefficients [1, -1], Yi-34B and LLaMA-2 70B lead
  on exact match; with [1,1,1,1,1], only LLaMA-2 70B and Mixtral reach meaningful exact match, while the
  continuous difference-to-target measure degrades smoothly (§6.1.3, Figure 3).
- **Distillation:** none reported. A chat model generates long-context QA pairs (§7.1), which is
  synthetic-data generation, not model distillation.
- **Negative feedback:** no likelihood-decreasing objective is reported. Negatives appear only as data removed
  by filters and as hallucination- and repetition-oriented rewriting of SFT responses (§2.1, §3.1): negative
  marginal value, not negative gradient.

## Connections
- [[lima]] — the line Yi says it follows (with DEITA), against FLAN- and UltraChat-style scaling (§3.1).
- [[fineweb]] — a later filtering and deduplication pipeline that reports removal ratios; Yi does not.
- [[qwen-2.5]], [[glm-4]] — other Chinese-English open family reports; Yi does not compare against either.
- [[llama-3]] — Yi's architecture code is based on the LLaMA implementation (§2.3).
- [[phi-3]] — another data-centric report, framed around synthetic textbook data rather than web filtering.
- [[retrieval-head]] — analyzes Yi among other families (its Table 1).
- [[prolong]], [[longalign]] — later long-context work with the mixtures and ablations Yi does not report.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2403.04652 (arXiv v3, 2025-01-21).
- Corrections to the previous card version:
  - "The GitHub release notes mention an additional 5B-token long-context data mixture for the enhanced long-context line" → the 5B-token figure is in the paper: "We continue pretrain the model on 5B tokens with 4M batch size, which translate to 100 optimization steps" (§7.1). §2.3 separately gives 10B tokens for the same extension; both rows are recorded as a conflict in the ledger.
  - "Extends the family to 200K context through lightweight continual pretraining" (as the whole story) → the extension is continual pretraining plus a long-context SFT stage that mixes short-context SFT with recitation-style synthetic document QA (§7.1).
  - "The GitHub repository says the models follow the Llama-style architecture / format closely" → the report itself states the code is based on the LLaMA implementation, with GQA at both sizes, SwiGLU at 8/3h, and RoPE ABF (§2.3). "Vision-language models that align a vision transformer encoder to the language model space" → the encoder is initialized from CLIP ViT-H/14 and connected by a two-layer MLP with layer normalization (§7.2).
  - "Public material does not disclose a modern RLHF / DPO / PPO recipe" → the report describes no RL or preference-optimization run for the released chat models; DPO and PPO are named only as infrastructure capabilities (§4). Header field "Authors / Lab" → "Authors" per the card standard; source type added.
- Removed as unsupported by the source: "Yi argues that strong open bilingual models can be built more from data engineering discipline than from exotic architecture changes" (rephrased to the report's own attribution claim with its numbers); "carefully hand-polished instruction data can still move the frontier"; "The abstract is already unusually useful"; "The GitHub release log is useful because it shows the sequence of public updates: Yi, Yi-9B, Yi-200K, then Yi-1.5" (Yi-1.5 is a later release not in this report); the "Why it matters" section, including "Yi is a good counterexample to the idea that every strong model jump needs a novel optimization algorithm" and the comparative claims about [[qwen-2.5]], [[llama-3]], and [[phi-3]] recipes, which this source does not make.
- Not reported by the source: pretraining data mixture percentages (Figure 2 is a pie chart without printed numbers); removal ratios of the filtering pipeline; pretraining batch size, LR schedule shape, warmup, optimizer settings, and total compute or GPU-hours; long-context CPT mixture proportions; a numeric Needle-in-a-Haystack score; SFT epoch count (only a 300-step count is given); any ablation of recitation-style QA against direct-answer QA.
