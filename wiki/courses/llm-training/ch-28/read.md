<!-- chapter: ch-28
     track: synthetic
     kind: content
     title: Long-Context Data Synthesis and Synthetic Evaluation Task Families
     deps: [ch-27, ch-32c]
     sources: [[needle-in-haystack-data]], [[excerpts/ruler-task-family]], [[ruler]], [[babilong]], [[helmet]], [[nolima]], [[excerpts/longalign-pipeline]], [[longalign]], [[excerpts/longmit-mimg]], [[excerpts/llama3-staged-schedule]], [[llama-3]], [[llama-3-recipe]], [[qwen-long-context-synth]], [[prolong]], [[prolong-recipe]], [[excerpts/artificial-needles-kv-retrieval]], [[in2-film]], [[context-synthesis-short-to-long]], [[nextlong]], [[lost-in-the-middle]], [[pose-synthesis]], [[excerpts/primary-quotes-2026-09]]
     figures: figures/task-family-generator.html, figures/context-extension.html
     revised: 2026-09 (generality revision)
-->

# Chapter 28 — Long-Context Data Synthesis and Synthetic Evaluation Task Families

> **Core insight.** The same program-built generators (needles, key-value lists, multi-hop questions over chunks) are used both to build long-context training data and to build long-context evaluation tasks, and a synthetic task measures only the skill its generator isolates: across 35 instruction-tuned models at 128K, NIAH correlates with ∞Bench QA at Spearman ρ = 0.63 while the RAG task HotpotQA correlates at 0.88 ([[helmet]] Fig. 4). On the data side, the two reports that measured the synthetic long SFT share disagree: the ProLong base model, after 40B tokens of long continued training with 37% short data in the mix, scored best with short-only UltraChat SFT and lost 1.6 points when 1% of SFT tokens were synthetic long data ([[prolong]] Table 8), while Llama 3.1 reports that short-only SFT caused "significant regressions" in long-context ability and a 0.1% synthetic long share optimized short and long benchmarks ([[excerpts/llama3-staged-schedule]] §4.3.4); the ProLong authors attribute the difference to the amount of prior long training and the size of the short SFT set (Interpretation). In tests at about 4K tokens, fact-free synthetic retrieval data transfers to real retrieval without the knowledge-benchmark losses that NIAH-style training data causes (Mistral-7B TriviaQA +0.11 vs −6.33, [[excerpts/artificial-needles-kv-retrieval]] Table 2).
>
> **Guideline.** When long-context ability is evaluated, report RULER effective length together with at least one application suite (HELMET categories) and a short-context suite, because NIAH is 100.0 for five of six frontier models at 128K ([[helmet]] Table 5) and does not rank models by application performance. When continued long-context training is run, keep high-quality short data in the mix (ProLong's best average was 60% long / 40% short in 5B-token ablations) and choose the mix by long and short scores measured after SFT, because 100% long data improved perplexity while lowering downstream scores after SFT ([[prolong]] §2.1, §3.2). When the long continued-training stage is short or the SFT set is large, add a small synthetic long share to SFT (Llama 3.1: 0.1%; LongAlign: 10k long examples added to 76k ShareGPT) and verify with an ablation, because short-only SFT regressed long context in Llama 3.1 and in SmolLM2 ([[excerpts/primary-quotes-2026-09]] smol-talk). Otherwise, when a model already has extensive long continued training, test short-only SFT first, because ProLong's synthetic long SFT at 1%, 3%, 10%, and 50% of tokens all scored below the 0% arm ([[prolong]] Table 8). When synthetic retrieval data is used for training, prefer data without factual content and evaluate with distractors that resemble real retrieval results, because the key-value data did not improve MDQA with retrieved (relevant) distractors ([[excerpts/artificial-needles-kv-retrieval]] §4).

## Corrections to the version you studied

1. "ProLong's coherence filter is the difference between 512K context on 20B tokens and garbage" → ProLong describes no coherence filter; it trains 40B tokens (20B at 64K, then 20B at 512K), and its main data finding is that long data must be mixed with high-quality short data ([[prolong]], §3.2, Table 9).
2. "Fu-2024 uses 200M" (RoPE base, also "10K → 200M NTK-aware") → Fu et al. state only that they adjust the RoPE base "as in Xiong et al. (2023)"; no value is printed ([[excerpts/primary-quotes-2026-09]], long-context-data-engineering, §1, §4).
3. Guideline "position extension via staged RoPE-base rescaling" and "RoPE base θ is rescaled from Llama-2's 10K to 500K for the final 128K model" → Llama 3 sets θ = 500,000 as an architecture hyperparameter for all of pre-training; no base change during the six long-context stages is described ([[excerpts/llama3-staged-schedule]], §3.2, §3.4.2).
4. Guideline "(3) SFT with synthesized long tasks" as the default stage → this depends on the prior stage: ProLong's final SFT is short-only UltraChat because synthetic long data hurt at every tested share, while Llama 3.1 needed 0.1% synthetic long data ([[prolong]], §5, Table 8; [[excerpts/llama3-staged-schedule]], §4.3.4).
5. "Chapters 23–27 established the synthetic-data design pattern" and "ch-27 — the synthetic-data design pattern" → the design pattern is ch-18; ch-27 is Agentic Trajectory Data (outline.json).
6. "MinHash at 32K-shingled docs is 8× more expensive than at 4K" → no source reports this figure; removed.
7. "teacher-context ceiling becomes a hard constraint" (cited to the unverified `longmit` card) → LongMIT generates questions from truncated document segments and then refills the context with other documents to a fixed length (App. C.2), so its generator does not read the full training context ([[excerpts/longmit-mimg]], App. C.2).
8. NIAH needle "eat a sandwich at Dolores Park", "score exact-substring", "a blog post" → the default needle is "eat a sandwich and sit in Dolores Park on a sunny day", the original harness scores with a GPT-4 judge on a 1–10 rubric, and the artifact is a GitHub repository with linked tweet threads ([[needle-in-haystack-data]], L24, L360-390, Summary).
9. "every 128K / 200K / 1M-context release now ships a NIAH heatmap" → unsupported and removed ([[needle-in-haystack-data]], Verification).
10. BABILong "20 task templates" → the released evaluation set covers 10 tasks (QA1–QA10) and Tables 2 and 4 average QA1–QA5; 50M tokens is reached only by ARMT ([[babilong]], App. N.2, §3.3).
11. "Llama-3.1-70B's RULER effective context is ~64K" (cited to the unverified `long-context-llama3` card) and "Meta acknowledges this in the paper" → the 128K claimed / 64K effective result is RULER Table 3; the Llama 3 report does not evaluate RULER ([[excerpts/ruler-task-family]], Table 3; [[excerpts/llama3-staged-schedule]], §5.2.6).
12. "Llama-3.1-405B NIAH@128K is ~99% and RULER effective context is ~96K" → 405B does not appear in RULER Table 3; the Llama 3 report gives NIAH 100% and Multi-needle 98.1 for 405B ([[excerpts/llama3-staged-schedule]], §5.2.6, Table 21).
13. "Qwen-2.5-14B-1M NIAH @ 1M ≈ 100% and RULER @ 1M ≈ 85%" → RULER is reported only up to 128K (14B-Instruct-1M 92.2); Passkey Retrieval at 1M is perfect for 14B-Instruct-1M ([[qwen-long-context-synth]], Table 4, Fig. 1).
14. LongAlpaca "3,000 long documents — 40% ArXiv, 30% books, 30% GitHub", "ChatGPT or Claude", "3 QA pairs per document", length and profanity filters, "~$5K" → LongLoRA App. B.6 states self-collected QA on technical papers, science fiction, and other books, 9k long QA plus 3k Alpaca short QA, 5 epochs; generator, document count, code, filters, and cost are not stated ([[excerpts/primary-quotes-2026-09]], longalpaca).
15. "Data quality saturates around 10k long examples" and "LongAlign-10k beats the larger LongAlpaca-12k on multi-segment integration" → LongAlign reports consistent improvement from 0k to 5k to 10k and no saturation; LongAlign-10k is higher on LongBench-Chat (6.28 vs 4.58) and MT-Bench, while LongAlpaca-12k is higher on LongBench S-Doc QA and M-Doc QA ([[excerpts/longalign-pipeline]], §4.2, Table 3).
16. LongChat "mines ShareGPT's long tail — ≥ 8K tokens, ≥ 4 turns — for 18K long conversations" → LongChat reuses Vicuna's ShareGPT conversations truncated to 16K and fine-tunes 7B on 80k and 13B on 18k conversations ([[excerpts/primary-quotes-2026-09]], longchat, Step 2).
17. "LongMIT ... synthesized multi-turn long-context dialogs — 5–10 turns ... lifts LongBench-Chat by 5–10 points" → LongMIT is multi-hop long-context instruction data built with the MIMG multi-agent framework (Chen et al. 2024); it reports LongBench QA subsets, not LongBench-Chat ([[excerpts/longmit-mimg]], §2, Table 1).
18. ProLong "≥ 64K tokens of coherent content", "Web: discarded", "code × 4, books × 2, academic × 2, forum × 1, web × 0.5", "~40% code, 25% books, 15% academic ..." → the final mix is 30% code repositories, 30% books, 3% textbooks, 37% ShortMix; CommonCrawl was tested as a long source (Table 4) ([[prolong]], Table 4, Table 9).
19. "Replacing the curated long documents with concatenated short documents ... costs 10+ points on HELMET" → no such ablation is in the paper ([[prolong]], Verification).
20. ProLong stage 1 "RoPE base rescaled 500K → 128M", "LR 1e-4 → 1e-5", "100% long coherent documents ... no cross-document packing" → RoPE base 8×10^6 at 64K and 1.28×10^8 at 512K; LR 1e-5 with cosine decay to 1e-6 per stage; short data is packed and attention across documents is masked ([[prolong]], Table 9, App. B.2).
21. ProLong "Stage 2 — SFT (5B tokens): 70% long-instruction (Claude-3-generated) + 30% UltraChat + multi-needle NIAH" → SFT is 1B tokens of UltraChat only; synthetic long data from Llama-3-8B-Instruct and Llama-3-70B-Instruct was tested and not used ([[prolong]], §5, Table 9).
22. "~200K H100-hours", "(Llama-3-8B base)", "beating Llama-3.1-8B-Instruct and Qwen2-7B-Instruct on InfiniteBench @ 128K" → 2.2K + 12.2K H100 hours; initialized from Llama-3-8B-Instruct; HELMET 128K average 49.4 vs 46.5 for Llama-3.1-8B-Instruct, ∞Bench 48.0 vs 46.7, and RULER 71.9 vs 81.3 ([[prolong]], Tables 9, 10, 26).
23. Llama 3 table of four stages with tokens (~100B, ~100B, ~150B, ~200B) and short:long ratios 80:20 → 40:60 → the report prints six stages and about 800B tokens, with no per-stage lengths, tokens, or ratios ([[llama-3-recipe]], v3 §3.4.2).
24. "Long-context SFT is kept at ~0.1% of total SFT samples (~100K out of ~100M). Raising the long-SFT fraction above 1% costs ~1 MMLU point" → the report states 0.1% (Table 7: 0.11% of examples, 38,135.6 tokens on average); the total example count and any MMLU cost are not reported ([[llama-3-recipe]], §4.3.4, Table 7).
25. "Teacher is Llama 3 405B itself (self-distillation)" → QA is generated by "an earlier version of the Llama 3 model" and chunk summaries by "our strongest Llama 3 8K context model" ([[excerpts/llama3-staged-schedule]], §4.3.4).
26. Fu 2024 "documents longer than 32K get 5× weight" and "globally up-weighting Books ... drops short-context MMLU by 3–5 points" → neither number is in the paper; it reports per-domain loss effects of domain upsampling (Table 5) and packs data to 80K chunks regardless of document boundaries ([[excerpts/primary-quotes-2026-09]], long-context-data-engineering, §3, §4).
27. LongRoPE "θ_i = θ^(2i/d)" and "With d = 128 RoPE dimensions, that's a 128-dimensional search space" → θ_i = θ^(−2i/d); the search has one factor per frequency (d/2 values) plus n̂, the number of initial tokens kept without interpolation ([[excerpts/primary-quotes-2026-09]], longrope-data, §2.1, §3.2).
28. LongRoPE "Fitness is a weighted combination of long-context perplexity + NIAH retrieval accuracy" → the search minimizes perplexity on 5 PG19 validation samples; P = 64, T = 40, p = 0.3 are correct ([[excerpts/primary-quotes-2026-09]], longrope-data, Algorithm 1, §4.1).
29. LongRoPE "< 1B tokens ... (~300M at 256K, ~600M at 2M)", "10× reduction in fine-tune data", "MMLU and GSM8K stay within 1 point" → fine-tuning is 400 steps on 128k segments and 600 more steps for 256k, global batch 32, with no token count printed; 2048k is reached by a second search; the 10× figure is not in the paper ([[excerpts/primary-quotes-2026-09]], longrope-data, §4.1).
30. PoSE "train on 4K-token samples", "4× compute reduction", "retrieval-strong / reasoning-weak" → PoSE uses 2,048-token training windows; the compute and reasoning claims are not in the paper ([[pose-synthesis]], §4.1, Verification).
31. Qwen2.5-1M "32K → 128K → 256K (50B + 50B + 100B tokens)" → stages 4,096 → 32,768 → 65,536 → 131,072 → 262,144 with RoPE bases up to 10,000,000; token counts are not reported ([[qwen-long-context-synth]], §3).
32. Qwen2.5-1M "Synthetic SFT mix generated by Qwen-Max" with multi-needle, 50K–200K summarization, 5–20-passage RAG-QA, FILL-IN, and a multi-position filter → queries are written by Qwen2.5 from a random segment and responses by the Qwen-Agent framework; fill-in-the-middle is a pre-training task; the listed SFT tasks and filter are not in the report ([[qwen-long-context-synth]], §3, §4).
33. "DCA ... inter-chunk uses a low-rank formulation" with "256K-sized chunks" → DCA remaps relative positions into intra-chunk, inter-chunk, and successive-chunk position indices so no distance exceeds the pre-training length; it is training-free ([[qwen-long-context-synth]], §5.1).
34. "Short-context MMLU / GSM8K within 1 point of base Qwen-2.5" and "Qwen 3 inherits and refines this recipe" → Table 6 shows mixed changes (14B GPQA 45.5 → 39.9; 7B MMLU-Pro 56.3 → 54.3; 14B GSM8K 94.8 → 94.8); the report says nothing about Qwen 3 ([[qwen-long-context-synth]], Table 6, Verification).
35. "ch-29 (next) ... long-context synthesis is one of the optional modality specializations" and "DCA ... deserves separate study in inference / serving chapters" → ch-29's concepts contain no long-context component, and the outline has no inference or serving chapter; long-document synthesis is ch-29a and context extension is ch-32b (outline.json).

## Why this chapter matters for a general-purpose model

The pipeline for a general-purpose model is pre-training → mid-training → SFT → preference optimization → RL → evaluation. Long-context ability is added mainly at mid-training (ch-32b), preserved or elicited at SFT, and gated at evaluation (ch-32c). Natural data with long-range dependencies and natural long evaluation tasks are both scarce, so labs synthesize them. This chapter covers the synthetic task families used for evaluation, the early and production recipes for synthetic long SFT data, and the evidence on how much synthetic long data to mix.

Three measurable problems motivate the chapter. First, a benchmark built from one generator can be saturated while application ability varies: NIAH is 100.0 for five of six frontier models at 128K, and those five models have HELMET averages from 47.0 (Llama-3.1-8B-Inst) to 63.8 (GPT-4o-08) in the same table ([[helmet]] Table 5). Second, long SFT data changes short-context ability and long-context ability in different directions depending on the prior stage (§4, §5). Third, training on synthetic long tasks can transfer, fail to transfer, or remove other abilities, depending on what the synthetic data contains (§6). Each of these is a generality question: a long-context gain is useful for a general model only if it holds on tasks the generator did not produce and does not remove short-context ability.

## §1 Synthetic evaluation task families: NIAH, RULER, BABILong

**Definitions.** A *synthetic long-context task* inserts task-relevant text (the *needle* or *facts*) into long filler text (the *haystack*) by a program, so that the answer is known without annotation. A *task family* is a generator with parameters (needle type, number of distractors, number of hops, input length) that produces many instances.

**Needle-in-a-Haystack (NIAH).** The Kamradt harness inserts one sentence into Paul Graham essays at a chosen depth, trims the context to a chosen length, asks a question the needle answers, and scores the reply with a GPT-4 judge on a 1–10 rubric ([[needle-in-haystack-data]] L360-390). In the released GPT-4-128K run, 207 of 225 (length, depth) cells scored 10 ([[needle-in-haystack-data]] Findings). The problem it addresses is whether a model can retrieve a fact at a given position. The problem it does not address is whether the model can ignore similar distractors, return several items, or combine facts.

**RULER.** RULER extends NIAH to 13 task configurations in four categories ([[excerpts/ruler-task-family]] §3, Table 5):
1. Retrieval: single needle (S-NIAH), multi-key with distractor needles (MK-NIAH), multi-value (MV-NIAH), multi-query (MQ-NIAH).
2. Multi-hop tracing: variable tracking (VT), `X2 = X1`, `X3 = X2`, return all names bound to one value.
3. Aggregation: common-word and frequent-word extraction (CWE, FWE).
4. QA: SQuAD and HotpotQA with distractor paragraphs.

Each model is evaluated with 500 generated examples per task at 4K, 8K, 16K, 32K, 64K, and 128K, inside the model's chat template, with an answer prefix and recall-based accuracy ([[excerpts/ruler-task-family]] §4).

**Effective length.** RULER defines the effective length as the largest tested length whose 13-task average exceeds the Llama2-7B score at 4K:

```
L_eff(model) = max { L ∈ {4K, 8K, 16K, 32K, 64K, 128K} : S(model, L) > 85.6 }
```

- `S(model, L)` is the mean accuracy over the 13 tasks at input length `L`.
- `85.6` is Llama2-7B's mean at 4K ([[excerpts/ruler-task-family]] Table 3).

**Worked example.** Llama 3.1 70B scores 96.5, 95.8, 95.4, 94.8, 88.4, and 66.6 at 4K through 128K. The scores exceed 85.6 through 64K (88.4) and fall below at 128K (66.6), so L_eff = 64K for a claimed 128K. Llama 3.1 8B scores 87.4 at 32K and 84.7 at 64K, so L_eff = 32K. GradientAI's Llama-3 70B (claimed 1M) scores 90.8 at 16K and 85.4 at 32K, so L_eff = 16K ([[excerpts/ruler-task-family]] Table 3). A 0.2-point difference at 32K (85.4 vs 85.6) changes the reported length by a factor of two, so L_eff is a threshold statistic with a large step size; the full score row carries more information. The figure [figures/context-extension.html](figures/context-extension.html) plots these Table 3 rows against the threshold so the reader can read off the effective length for each model and see how close each score is to 85.6.

**RULER's failure analysis.** On Yi-34B-200K, adding distractor needles lowered accuracy by about 40 points at 256K in the full-haystack setting, raising the number of queries from 1 to 8 lowered it by about 15 points, and over 80% of CWE outputs at 128K copied the one-shot example ([[excerpts/ruler-task-family]] §5). These are three separate failure types that a single-needle test does not expose.

**BABILong.** BABILong hides the facts of bAbI reasoning tasks (agents moving between locations) between sentences of PG19 books, keeping the task fixed while the length grows ([[babilong]] §2, App. N.3). The tested LLMs "effectively use only 10-20% of the context"; with background text every tested LLM stayed below 85% on QA2 (two supporting facts), while several passed QA1 (one fact) up to 16K–64K ([[babilong]] Abstract, §3.1). RAG pipelines reached about 60% on QA1 at every length but fell below random on QA2 and QA3 ([[babilong]] §3.2).

**Measurement limits.**
- *Correlation with applications.* HELMET computes Spearman ρ between synthetic tasks and application categories across 35 instruction-tuned models at 128K. NIAH vs ∞Bench QA is 0.63, RULER MK vs ∞Bench QA 0.81, and HotpotQA RAG vs ∞Bench QA 0.88; no synthetic task averages above 0.8 against the downstream categories ([[helmet]] Figs. 3–4). Result (single study).
- *Lexical overlap.* When the question shares almost no words with the needle, GPT-4o falls from 99.3 to 69.7 at 32K ([[nolima]] Table 3). The authors interpret this as standard needles being locatable by word matching (Interpretation).
- *Distinguishable facts.* bAbI sentences are short and differ in style from PG19 book prose (books published before 1919), and the authors state that fine-tuned models can learn tokens that separate facts from background ([[babilong]] Limitations).
- *Position.* With 20 documents, GPT-3.5-Turbo scores 75.8% with the answer document first, 53.8% at position 10, and 63.2% at position 20 ([[lost-in-the-middle]] App. G Table 6). An average over depths hides this curve.

**Implication for a general-purpose model.** A synthetic family is a controlled probe of one skill. It is useful for diagnosis (which knob breaks the model) and for training-data design, and it is a weak proxy for application breadth. The detailed evaluation gate belongs to ch-32c. The figure [figures/task-family-generator.html](figures/task-family-generator.html) generates small instances of each family used in this chapter, lets the reader change the generator knobs (distractors, values, hops, depth), and shows next to each instance what the family measures and the limit reported for it.

## §2 Early long instruction data: LongAlpaca and LongAlign

**Definition.** *Long-context instruction data* is SFT data whose prompt contains a long document (8K–64K tokens in LongAlign; LongAlpaca does not report lengths) and whose response depends on that document.

**Problem.** After context extension by continued training, a model accepts long inputs but may not follow instructions over them. LongAlign measures this directly: ChatGLM3-6B-64k fine-tuned on ShareGPT alone (LongAlign-0k) scored 3.73 on LongBench-Chat, and adding 10k long examples raised it to 6.28 ([[excerpts/longalign-pipeline]] Table 3).

**LongAlpaca (2023).** LongLoRA's SFT set contains 9k long-context QA pairs on "technical papers, science fiction, and other books" and 3k short QA pairs sampled from Alpaca, trained for 5 epochs ([[excerpts/primary-quotes-2026-09]] longalpaca). The paper does not name the generator or give filters. In RULER Table 3, LongAlpaca-13B (claimed 32K) has an effective length under 4K ([[excerpts/ruler-task-family]] Table 3).

**LongAlign (2024), step by step** ([[excerpts/longalign-pipeline]] §3.2, App. A):
1. Extend the base model: RoPE base 10,000 → 2,000,000 and 10B tokens of continued training on sequences under 64K.
2. Sample documents under 64K tokens from 9 sources and upsample longer ones.
3. For each document, pick one of four task prompts at random (general, summary, multi-hop reasoning, information extraction) and ask Claude 2.1 for five questions that "cover all parts of the text".
4. Pick one question at random and ask Claude 2.1 for the answer.
5. Check quality by hand on a sample: 94 of 100 answers correct, 2 incorrect, 3 incomplete, 1 irrelevant.
6. Mix the 10k long examples with 76k ShareGPT examples and train 2 epochs.

**Mixing result (ChatGLM3-6B-64k, Table 3).** LongBench-Chat 3.73 / 5.99 / 6.28 for 0k / 5k / 10k long examples. Short tasks stay within 1.1 points on MMLU (45.5 / 46.6 / 45.5) and 0.17 on MT-Bench (5.34 / 5.50 / 5.51). LongAlpaca-12k gives 4.58 on LongBench-Chat and 4.93 on MT-Bench but the highest S-Doc QA (65.8) and M-Doc QA (45.6). Result (single study, one model for this table). The authors attribute LongAlpaca's LongBench advantage to 2WikiMQA and NarrativeQA resembling its book-based sources (Interpretation).

**Loss weighting under packing.** Long SFT data has a skewed shape: LongAlign-10k targets average 200 tokens with a target/sequence token ratio of 0.015, against 330 tokens and a ratio printed as 19.3 for ShareGPT (no unit is printed; the value is consistent only with a percentage, 0.193) ([[excerpts/longalign-pipeline]] App. A). When sequences are packed and each pack's loss is a token mean, sequences are not weighted equally. The intended loss gives each sequence weight 1/M:

```
L = (1/M) Σ_i L_i / N_i                                  (per-sequence mean)
L' = (1/K) Σ_k ( Σ_{i∈k} L_i ) / ( Σ_{i∈k} N_i )          (per-pack token mean)
```

- `M` is the number of sequences in the batch, `K` the number of packs.
- `L_i` is the summed token loss of sequence `i`, `N_i` its number of target tokens.
- In `L'`, sequence `i` in pack `k` has weight `N_i / (K · Σ_{j∈k} N_j)` on its mean loss `L_i/N_i`.

LongAlign scales each target token of sequence `i` by `K / (N_i · M)` and sums over packs, which recovers `L` ([[excerpts/longalign-pipeline]] Eq. 4, App. B).

**Worked example.** Take K = 2 packs and M = 4 sequences. Pack A holds one sequence with 200 target tokens. Pack B holds three sequences with 10, 20, and 30 target tokens (60 in total). Under `L'`, the pack-A sequence has weight 200 / (2 · 200) = 0.50, and the pack-B sequences have 10/120 = 0.083, 20/120 = 0.167, and 30/120 = 0.25. The intended weight is 1/4 = 0.25 each. The long single sequence receives twice its intended weight and the 10-token sequence one third. With the fix, ChatGLM3-6B-64k LongBench-Chat goes from 5.76 (packing) to 6.21, and Llama-2-7B-64k from 5.89 to 6.10 ([[excerpts/longalign-pipeline]] Table 4). Packing and sorted batching cut ChatGLM3-6B-64k training time from 45.4 h to 20.5 h and 19.1 h (Fig. 5). Packing masks are covered in ch-04.

## §3 Multi-hop long instruction synthesis: LongMIT and the MIMG framework

**Definition.** A *multi-hop* question requires two or more pieces of evidence located in different parts of the context. LongMIT is a multi-hop long-context instruction dataset built with MIMG (Multi-agent Interactive Multi-hop Generation) ([[excerpts/longmit-mimg]]).

**Problem, as measured.** Self-Instruct-style generation with Qwen2-72B gave fewer than 35% multi-hop samples and over 40% poor-quality samples in the authors' manual annotation ([[excerpts/longmit-mimg]] Abstract).

**Mechanism** ([[excerpts/longmit-mimg]] §2, App. C):
1. A single-hop question agent generates questions and then answers (question-then-answer order) from document segments.
2. A sampling step builds a question-similarity matrix with BGE embeddings and selects related questions from one document (higher quality) or several documents (higher diversity).
3. A merging agent combines the selected single-hop questions into one multi-hop question, without a rationale.
4. A quality verification agent (InternLM2-20B, with a rationale) scores each sample on several criteria; samples with quality score above 8.5 are kept.
5. The context is refilled with other documents to a fixed length.

**Evidence.** On human-labelled samples with Qwen-72B-Instruct as backbone, MIMG reaches 94.8 high-quality, 88.2 diversity, and 94.8 multi-hop, against 61.3 / 53.4 / 33.1 for Self-Instruct ([[excerpts/longmit-mimg]] Table 7). SFT on LongMIT gives LongBench QA averages (GPT-4o scored) of 64.29 on LLaMA3-8B vs 52.92 with LongAlign and 52.02 with LongAlpaca ([[excerpts/longmit-mimg]] Table 1). Verification findings: scoring outperforms binary classification, LLM verifiers agree poorly with humans on long inputs (low kappa) but select with high precision, and removing the rationale lowers verifier precision by more than 8.6% across domains (§3.1).

**Conditions and limits.** LongMIT has 64.40k samples and 5.07B tokens, against 9.89k and 0.17B for LongAlign (Table 3), so the Table 1 comparison is not matched in tokens. Evaluation covers LongBench QA subsets; summarization and code are not in Table 1. Short-context checks on a Llama-3-8B-ProLong base show IFEval instruction-level loose accuracy rising from 0.1259 to 0.1583 and ArenaHard moving from 7.2 to 6.7 with overlapping confidence intervals (Tables 6, 8). Result (single study).

**Implication for a general-purpose model.** The measurable levers here are the multi-hop rate and the verifier's precision, both of which are properties of the generator and filter, not of the source documents. For a general model, the same verifier logic applies to any long SFT slice; ch-29a covers long-document generators in more depth.

## §4 Production long-SFT generators: Llama 3.1 and Qwen2.5-1M

**Llama 3.1 long-context stage.** In Llama 3 405B pre-training, the context grows from 8K to 128K in six stages over about 800B tokens; the report does not give the schedule for 8B and 70B. A stage ends when "model performance on short-context evaluations has recovered completely" and the model "perfectly solves 'needle in a haystack' tasks up to that length" ([[excerpts/llama3-staged-schedule]] §3.4.2). The gate combines a short-context criterion and a synthetic retrieval criterion; §1 explains why the retrieval criterion alone does not establish application ability.

**Llama 3.1 long SFT generators** ([[excerpts/llama3-staged-schedule]] §4.3.4):
1. *QA:* split curated long documents into 8K chunks, generate QA pairs from randomly selected chunks, and train with the whole document as context.
2. *Summarization:* summarize 8K chunks, then summarize the summaries; train the model to summarize the full document; generate QA from the summaries that requires global understanding.
3. *Code reasoning:* parse Python imports, select files referenced by at least five other files, remove one, and ask the model which files depended on it and what the missing code is.
4. Bucket the samples at 16K, 32K, 64K, and 128K.

The generator reads 8K chunks while the training example contains the whole document, so the generator does not need a long context window. The answer depends on one chunk (QA) or on several parts of the document (summary QA, code).

**Mixing evidence.** Short-only SFT "resulted in significant regressions in long-context capabilities"; a 0.1% synthetic long share "optimizes the performance across both short-context and long-context benchmarks"; short-only DPO did not hurt long context when the SFT model was strong on long context ([[excerpts/llama3-staged-schedule]] §4.3.4). No ablation table is printed. Result (single study, no numbers).

**Worked example: example share vs token share.** Table 7 lists long-context data at 0.11% of SFT examples with 38,135.6 tokens on average, and 846.1 tokens on average over all SFT examples ([[excerpts/llama3-staged-schedule]] Table 7). The long-context token share is 0.0011 × 38,135.6 / 846.1 ≈ 0.050, about 5% of SFT tokens (derived; the report prints no token share). A "0.1% long" recipe copied with the example share applied to tokens would contain about 50 times fewer long tokens than Llama 3.1 used.

**Qwen2.5-1M.** Pre-training extends the context in five stages (4,096 → 32,768 → 65,536 → 131,072 → 262,144 tokens, RoPE bases up to 10,000,000), with 75% of sequences at the stage's maximum length and 25% shorter in stages 3–5 ([[qwen-long-context-synth]] §3). Natural long text is augmented with three synthetic tasks because natural text "often exhibits weak long-distance associations": fill-in-the-middle, keyword- and position-based paragraph retrieval, and paragraph reordering (§3). For SFT, Qwen2.5 writes a query from a randomly extracted segment and the Qwen-Agent framework writes the answer over the full document with retrieval, chunk-by-chunk reading, and step-by-step reasoning; SFT runs first on short data up to 32,768 tokens and then on a short and long mix up to 262,144 tokens (§4). Offline RL uses only pairs of at most 8,192 tokens and still raises LongBench-Chat (7B 7.32 → 8.08, 14B 8.56 → 8.76) (Table 3). The report reads this as short-to-long transfer (Interpretation).

**Conditions and limits.** Neither report publishes example counts or ablation tables for its long SFT share. Qwen2.5-1M's short-context results against the 128K models are mixed (14B GPQA 45.5 → 39.9, IFEval 81.0 → 84.3; [[qwen-long-context-synth]] Table 6).

## §5 Continued-training data and the SFT share, as ProLong states it

**Setting.** ProLong studies continued training of Llama-3 8B for long context, with ablations of 5B tokens at 64K, each evaluated after UltraChat SFT on a HELMET subset plus five short tasks ([[prolong]] §2–§3).

**Findings, with numbers.**
1. *Perplexity does not select the mix.* As the long share rises to 100%, PG19 perplexity keeps improving while the downstream long-context average drops ([[prolong]] §2.1, Fig. 1).
2. *Evaluate after SFT.* Before SFT, recall and RAG prefer high long shares; after SFT they fall with more long data. Short-task scores fall monotonically as the long share rises. The best average is 60% long / 40% short ([[prolong]] §3.2, Fig. 3). The authors hypothesize that a long-only model is a poor initialization for generic SFT (Interpretation).
3. *Long sources.* With 60% long data: books/repositories 1:1 average 54.6, books 53.8, code repositories 52.3 (highest recall, 99.2), CommonCrawl 50.9 ([[prolong]] Table 4). Repository files are concatenated without dependency ordering, which the authors expect to "increase the distance between dependent files and reduce recency bias" (§3.1).
4. *Short sources.* ShortMix scores 54.6 long / 65.5 short against FineWeb-Edu 53.0 / 63.0 ([[prolong]] Table 6).
5. *Document masking.* Masking attention across documents gives 54.6 / 65.5 against 53.6 / 64.9 without masks ([[prolong]] App. B.2, Table 20).
6. *Train longer than evaluated.* From a 20B-token 64K checkpoint, 4B more tokens at 512K beat 4B at 64K when both are evaluated at 64K (recall 98.5 vs 95.0, re-rank 32.9 vs 28.0) ([[prolong]] Table 7).

**Why training on longer sequences than the evaluation length helps, as the authors model it (hypothesis).** Assume a task needs examples of dependencies spanning exactly d tokens, and dependencies are equally likely at every position. A sequence of length L contains L − d + 1 spans of length d. One document of length n·d contains n·d − d + 1 such spans; n separate documents of length d contain n. The difference is (n − 1)(d − 1) ([[prolong]] §4). The authors state that these assumptions do not hold in practice and offer the model as intuition for the Table 7 result. Worked example: d = 4, n = 3. One 12-token document has 12 − 4 + 1 = 9 spans; three 4-token documents have 3; the difference is 6 = (3 − 1)(4 − 1). The same count explains why packing unrelated short documents with document masking adds no long spans (Interpretation).

**Synthetic long SFT data.** Synthetic long instruction data (40% QA from random chunks, 30% RAG over chunk lists, 30% recursive book summaries, generated by Llama-3-8B-Instruct) mixed into UltraChat by token share; unlike ProLong's 5B-token data ablations, these SFT runs start from the ProLong base model trained on 40B tokens up to 512K ([[prolong]] §5, Table 8):

| Synthetic share (tokens) | 0% | 1% | 3% | 10% | 50% |
|---|---|---|---|---|---|
| Average (RAG, re-rank, ICL at 32K/64K average; JsonKV, QA, summarization at 512K) | 55.7 | 54.1 | 53.5 | 53.9 | 43.3 |

A Llama-3-70B-Instruct generator gave 54.2 / 55.4 / 53.6 / 49.7 at 1 / 3 / 10 / 50%, also below 55.7 ([[prolong]] App. B.5, Table 23). The figure [figures/task-family-generator.html](figures/task-family-generator.html) also plots this table as bars.

**Reconciling ProLong with Llama 3.1 and LongAlign.** The authors give two hypotheses: earlier work may have had too little long continued training, so synthetic SFT data acted as additional long training; and Llama 3.1's short instruction set is much larger, so a long share may prevent degeneration on long tasks during extensive short SFT ([[prolong]] §5). Both are Interpretation. Supporting observation for the second: SmolLM2 fine-tuned only on short samples lost long-context ability beyond 2,048 tokens, and SmolTalk added LongAlign samples under 16K tokens ([[excerpts/primary-quotes-2026-09]] smol-talk). Open question: no study varies both the long continued-training budget and the SFT set size.

**Conditions and limits.** ProLong's results are at the 10B scale on Llama-3 models ([[prolong]] Limitations). After SFT, ProLong averages 69.4 on five short tasks against 69.8 for Llama-3-8B-Instruct, but is lower on MMLU (64.6 vs 67.0) and GSM8K (58.9 vs 68.5); the authors attribute this to Llama-3-8B-Instruct's closed instruction data ([[prolong]] App. B.7, Table 25).

## §6 Synthetic retrieval training and transfer to real tasks

**Question.** Does fine-tuning on a synthetic retrieval generator improve real long-context tasks, and at what cost to other abilities?

**Artificial Needles.** The data are lists of integer dictionaries; the model reports the value of a key and the dictionary containing it, or, in the harder variant, finds a tuple key whose subkeys partly overlap other keys ([[excerpts/artificial-needles-kv-retrieval]] §2). Mistral-7B-Instruct-v0.1 is trained on 350 samples of about 3,900 tokens for 2 epochs with loss on answer tokens only; GPT-3.5 Turbo on 150 multi-subkey samples for 3 epochs (§3.1). An answer template in the prompt keeps the loss on formatting tokens low (§2, Fig. 4).

**Evidence.**
- Transfer: on 20-document MDQA (about 4K tokens) the U-shaped position curve flattens; the abstract reports a 10.5% gain at position 10 for GPT-3.5 Turbo. Fine-tuning on MDQA itself with a similar token count gave lower MDQA accuracy ([[excerpts/artificial-needles-kv-retrieval]] §3.2, Fig. 5).
- Retention and comparison (Mistral-7B, similar training tokens, Table 2):

| Fine-tuning data | MMLU | TriviaQA | NQ-Open |
|---|---|---|---|
| none | 53.42 | 47.63 | 11.61 |
| synthetic key-value, template | 53.44 | 47.74 | 11.98 |
| MultidocQA | 53.19 | 45.20 | 8.69 |
| IN2 | 53.49 | 45.44 | 9.80 |
| Needle-in-a-haystack | 52.83 | 41.30 | 4.88 |

The authors attribute the TriviaQA and NQ-Open drops of the baselines to training on factual content, which can encourage hallucination (Interpretation, citing Gekhman et al.).

**Related synthetic designs.** IN2 trains Mistral-7B-Instruct-v0.2 on 1.4M synthesized QA examples with evidence placed at random positions in 4K–32K contexts, plus short QA and general instructions; the VAL Probing min-max position gap falls from 56.2 to 13.9 and MMLU moves from 59.3 to 59.2 ([[in2-film]] Table 1, Fig. 5). Training with distractor contexts matters: LLaMA3.1-8B with 1.6k human-written QA pairs whose contexts GPT-4o-mini wrote (1 relevant + 9 irrelevant) reaches a LongBench average of 38.57, against 25.39 for instruction synthesis from original documents and 23.35 for UltraChat alone ([[context-synthesis-short-to-long]] Table 3). For continued training, interleaving each chunk of a short document with its most similar retrieved chunks reaches a HELMET+RULER average of 62.58 at 128K, against 52.85 for random concatenation ([[nextlong]] Table 1).

**Conditions and limits.** Artificial Needles tests contexts of about 4K tokens plus one 24K run, and "Models finetuned on our dataset will not improve" on MDQA with retrieved (relevant) distractors ([[excerpts/artificial-needles-kv-retrieval]] §4, Fig. 10). BABILong's QA1 fine-tuning shows the transfer risk in the other direction: GPT-3.5 fine-tuned on QA1 improved on QA2–QA5 while fully fine-tuned Mistral-7B degraded on them ([[babilong]] App. I, Fig. 9). Result (single studies).

**Implication for a general-purpose model.** Synthetic retrieval data is an input to a mixture, not a stand-alone recipe. Its value for a general model is measured on held-out real tasks with realistic distractors and on knowledge benchmarks. A gain on tasks built by the same generator as the training data does not show transfer, because the Artificial Needles gains did not extend to MDQA with retrieved distractors (§4 of that paper).

## Negative samples and negative feedback

**Where negatives appear in this stage.** (1) *Negative marginal value:* samples rejected by a filter, such as LongMIT samples scored at or below 8.5 by the InternLM2-20B verifier ([[excerpts/longmit-mimg]] App. C.4.2), or LongAlign answers found incorrect, incomplete, or irrelevant in the manual check (6 of 100; the paper does not describe removing such samples from the full set). (2) *Negative as content:* distractors placed in the input and trained with ordinary cross-entropy on the correct answer, such as MK-NIAH distractor needles, multi-subkey keys that share subkeys with the gold key ([[excerpts/artificial-needles-kv-retrieval]] Fig. 2), 9 irrelevant contexts per example ([[context-synthesis-short-to-long]]), and similar retrieved chunks ([[nextlong]]). (3) *Negative as gradient:* in the sources of this chapter, only Qwen2.5-1M's offline RL "similar to DPO" uses rejected responses, and only on pairs of at most 8,192 tokens ([[qwen-long-context-synth]] §4). No source here uses negative as conditioning.

**What current practice does.** Long-context pipelines mostly discard failures (verifier thresholds) and use distractors as content. Negative gradients on long inputs are not reported in these sources; Llama 3.1 and Qwen2.5-1M both keep preference data short ([[excerpts/llama3-staged-schedule]] §4.3.4).

**Mechanism for the gradient case.** For a softmax over logits z with probabilities p, the gradient of the log-probability of token y is

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

- `1[j = y]` is 1 for the chosen token and 0 otherwise; `p_j` is the model probability of token j.

A step that decreases log p_y moves z_y down by (1 − p_y) and moves every other z_j up by p_j. Worked example: p = (0.7, 0.2, 0.1) and y = token 3 (the unlikely, rejected token). The logit changes are proportional to (+0.7, +0.2, −0.9), so most of the removed mass goes to token 1, the already most likely token. Pushing down an unlikely rejected continuation therefore sharpens the distribution toward the current mode. The full treatment is ch-43a; supervised uses of negatives are ch-31a.

**Evidence and controls.** Distractors as content improved robustness in the cited studies: the NIAH pilot model trained with 1k-essay distractors stayed above 90% at 32K when distractors appeared at test time, while the model trained without distractors degraded ([[context-synthesis-short-to-long]] §3.3, Fig. 2). Hard negatives chosen by similarity beat random ones for continued-training data ([[nextlong]] §5.4, Fig. 7, values not printed). For filters, the controls are a rationale in the verifier and several scoring criteria (§3). No source in this chapter measures the share of improvement due to negatives.

**Diagnostics.** Log the verifier rejection rate per source domain and per length bucket (verifier precision falls for classification on longer inputs; [[excerpts/longmit-mimg]] Fig. 3c); report accuracy separately for instances with and without distractors; for any DPO-style stage on long inputs, log chosen and rejected log-probabilities separately.

**Effect on generality.** Distractor-as-content data targets a failure (retrieving a similar wrong item) that applications share. Filter rejections change the task and domain distribution of what remains; record the per-domain survival rate so a filter does not remove a domain from the long slice.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | long-context | stages; tokens | six stages from 8K to 128K; approximately 800B tokens; per-stage lengths, tokens, mix not printed | arXiv:2407.21783v3 §3.4.2 ([[llama-3-recipe]]) | verified 2026-09-15 | stage gate: short-context evals fully recovered and NIAH solved at that length |
| Llama 3.1 | 8B/70B/405B | pretrain-stable | RoPE base | 500,000 (whole pre-training) | §3.2 | verified 2026-09-15 | cites Xiong et al. (2023) for lengths up to 32,768; no ablation reported |
| Llama 3.1 | not scoped | SFT | long-context share | 0.1% synthetic long data (§4.3.4); 0.11% of examples, 38,135.6 tokens per example (Table 7) | §4.3.4, Table 7 | verified 2026-09-15 | "careful ablations"; no table |
| Llama 3.1 | not scoped | SFT | length buckets | 16K, 32K, 64K, 128K | §4.3.4 | verified 2026-09-15 | no ablation reported |
| Llama 3.1 | not scoped | preference | DPO data length | short-context only | §4.3.4 | verified 2026-09-15 | no long-context regression when the SFT model is strong on long context; no table |
| ProLong-64k-Base (Llama-3-8B-ProLong-64k-Base) | 8B | long-context | stage 1 | 20B tokens at 64K; RoPE base 8×10^6; batch 4M tokens; AdamW LR 1e-5, 10% warmup, cosine to 1e-6 | arXiv:2410.02660v4 Table 9 ([[prolong-recipe]]) | verified 2026-09-14 | App. B.1 Table 18: base 8×10^6 → 54.6 vs 4×10^6 → 48.7 |
| ProLong-512k-Base (Llama-3-8B-ProLong-512k-Base) | 8B | long-context | stage 2 | 20B tokens at 512K; RoPE base 1.28×10^8; batch 8M tokens; LR 1e-5 (Table 9) vs 5e-6 (released `train_512K.sh`) | Table 9; github.com/princeton-nlp/ProLong@499fa29 | conflict | Table 7: +4B tokens at 512K > +4B at 64K, evaluated at 64K |
| ProLong-64k-Base | 8B | long-context | data mix (share type not stated) | 30% code repos, 30% books, 3% textbooks, 37% ShortMix (27% FineWeb-Edu, 27% FineWeb, 11% Wikipedia, 11% StackExchange, 8% Tulu-v2, 8% OpenWebMath, 8% ArXiv) | Table 9 | verified 2026-09-15 | Fig. 3 (5B-token ablation): 60% long best average; Tables 4, 6 |
| ProLong-64k-Base, ProLong-512k-Base | 8B | long-context | cross-document attention | masked | Table 9; App. B.2 | verified 2026-09-15 | Table 20: 54.6 / 65.5 vs 53.6 / 64.9 |
| ProLong-512k-Instruct (Llama-3-8B-ProLong-512k-Instruct) | 8B | SFT | data; tokens; LR; batch | UltraChat only; 1B tokens; LR 2e-5, 5% warmup, cosine to 2e-6; 4M tokens | Table 9 | verified 2026-09-15 | Table 8: 0% synthetic 55.7 vs 1% 54.1 |
| ChatGLM3-6B-64k, Llama-2-7B/13B-64k (LongAlign) | 6B, 7B, 13B | long-context | RoPE base; tokens | 10,000 → 2,000,000; 10B tokens under 64K | EMNLP 2024 Findings §4.1 ([[excerpts/longalign-pipeline]]) | verified 2026-09-15 | no ablation reported |
| LongAlign-6B-64k | 6B | SFT | mix; epochs; packing | 76k ShareGPT + 10k LongAlign; 2 epochs (about 1500-2000 steps); about 12 sequences per pack; global batch 96; loss weighting | §4.1, App. B | verified 2026-09-15 | Table 3: LongBench-Chat 3.73 / 5.99 / 6.28 for 0k / 5k / 10k; Table 4 |
| LongAlpaca (LongLoRA) | 7B-70B (not scoped) | SFT | data; epochs | 9k long QA + 3k Alpaca short QA; 5 epochs | arXiv:2309.12307 App. B.6 ([[excerpts/primary-quotes-2026-09]]) | verified 2026-09-15 | no ablation reported |
| LLaMA3-8B + LongMIT | 8B | SFT | data; optimizer; batch; epochs | 64.40k samples, 5.07B tokens; Adam LR 3×10^-5, β1 0.9, β2 0.95; packing; 4M tokens; 1 epoch; max length 4K-128K | arXiv:2409.01893v2 Table 3, App. E.1 | verified 2026-09-15 | Table 1: 64.29 vs 52.92 (LongAlign) |
| LongMIT data | — | SFT (data generation) | generator; verifier; threshold | Qwen2-72B-Instruct; InternLM2-20B; quality score > 8.5 | App. C.3, C.4.2 | verified 2026-09-15 | §3.1, Figs. 3–4 (verifier strategy) |
| Qwen2.5-7B/14B-1M | 7B, 14B | long-context | stages; RoPE base; length mix | 4,096 → 32,768 (base 10,000 → 1,000,000); 65,536 / 131,072 / 262,144 with 1M / 5M / 10M; 75% at max length, 25% shorter; tokens not reported | arXiv:2501.15383v1 §3 ([[qwen-long-context-synth]]) | verified 2026-09-14 | Table 2: RULER-128K 37.6 / 56.0 / 83.8 / 87.6 (14B) |
| Qwen2.5-7B/14B-Instruct-1M | 7B, 14B | SFT | stages | short ≤ 32,768, then short + long ≤ 262,144; ratio not reported | §4 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B/14B-Instruct-1M | 7B, 14B | preference | pair length | offline RL similar to DPO, pairs ≤ 8,192 tokens; β not reported | §4 | verified 2026-09-14 | Table 3: LongBench-Chat +0.75 / +0.20 |
| Mistral-7B-Instruct-v0.1 (Artificial Needles) | 7B | SFT | data; epochs; loss | 350 key-value samples, 85 dictionaries, about 3,900 tokens; 2 epochs; answer tokens only | arXiv:2406.19292v2 §3.1 ([[excerpts/artificial-needles-kv-retrieval]]) | verified 2026-09-15 | Fig. 5; Table 2 |

**Starting point for a small general-purpose run.** For an 8B model that already has long continued training comparable to ProLong's (40B tokens, Llama-3 8B), start SFT with short data only, as ProLong-8B did with 1B tokens of UltraChat at LR 2e-5, and run a 1% synthetic long-data arm as the ablation, because Table 8 selected short-only in that setting. For the continued-training stage itself, the verified starting mix is ProLong-64k-Base's 63% long (30% code repositories, 30% books, 3% textbooks) and 37% ShortMix at 64K with RoPE base 8×10^6 and document masking (Table 9), used at 8B on Llama-3 for 20B tokens; the 5B-token ablations that selected it found 60% long / 40% short best (Fig. 3). When the long continued-training stage is much shorter or the SFT set is large, start from a long share of 0.1%, as Llama 3.1 reported (§4.3.4; Table 7 gives 0.11% of examples; model size not scoped), and compute the token share it implies (about 5% from Table 7, derived in §4) before choosing. LongAlign's 10k long examples mixed with 76k ShareGPT (10/86 ≈ 12% of examples, derived) is the verified point for ChatGLM3-6B-64k after 10B tokens of context extension.

## Generalization lens

**(a) What increases breadth.**
- Short data during long continued training: short-task averages fall monotonically as the long share rises, and 60/40 gave the best long average after SFT ([[prolong]] Fig. 3).
- Diverse long sources: books and code repositories 1:1 beat either alone on the long average (54.6 vs 53.8 and 52.3), and books help ICL and summarization while repositories help recall ([[prolong]] Table 4).
- Several task types in long SFT: LongAlign's four prompt types and Llama 3.1's QA, summarization, and code generators target different application categories; HELMET shows category correlations as low as 0.34 at 128K ([[helmet]] Fig. 5), so one task type does not cover the others.
- Fact-free synthetic retrieval data kept TriviaQA and NQ-Open unchanged while NIAH-style data lowered them by 6.33 and 6.73 ([[excerpts/artificial-needles-kv-retrieval]] Table 2).

**(b) What causes narrowing or forgetting.**
- 100% long continued training: perplexity improves while downstream long scores after SFT drop ([[prolong]] §2.1, §3.2).
- Too much synthetic long SFT after long continued training: 50% synthetic gave 43.3 vs 55.7 ([[prolong]] Table 8).
- Short-only SFT without long continued training of ProLong's size: regressions in Llama 3.1 ([[excerpts/llama3-staged-schedule]] §4.3.4) and loss beyond 2,048 tokens in SmolLM2 ([[excerpts/primary-quotes-2026-09]] smol-talk).
- Training on factual long QA or needle data: TriviaQA and NQ-Open losses of 1.81 to 6.73 points on Mistral-7B ([[excerpts/artificial-needles-kv-retrieval]] Table 2).
- Training on one templated family: fully fine-tuned Mistral-7B on BABILong QA1 degraded on QA2–QA5 ([[babilong]] App. I).

**(c) How to measure it at this stage.**
- Long side: RULER full score rows and effective length, one application suite (HELMET categories), and position-resolved accuracy ([[lost-in-the-middle]]).
- Short side: the same short suite before and after each long stage (ProLong used five short tasks; Llama 3.1 gated stages on short-context recovery).
- After SFT, not before ([[prolong]] §2.2, Fig. 2).
- Held-out families: evaluate on families not used to generate training data (for example train on key-value retrieval and test on MDQA with retrieved distractors, [[excerpts/artificial-needles-kv-retrieval]] §4). ProLong reports held-out HELMET tasks, NoCha, RULER, and ∞Bench beyond its development subset ([[prolong]] Limitations); its RULER score (71.9) is below Llama-3.1-8B-Instruct (81.3) while its HELMET average is higher (49.4 vs 46.5), so conclusions depend on the suite ([[prolong]] Tables 10, 26).
- Known measurement errors: NIAH scores depend on the judge and prompt template ([[needle-in-haystack-data]] Guideline); effective length is a threshold statistic (§1 worked example); LongBench-Chat has 50 queries ([[excerpts/longalign-pipeline]] §3.4).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Choosing the continued-training mix by perplexity | PG19 or held-out long perplexity improves while HELMET recall and RAG after SFT decline | Evaluate each candidate mix after a fixed short SFT run ([[prolong]] Fig. 1, Fig. 3) |
| Treating NIAH as the long-context gate | NIAH 100 at all depths; RULER MK-NIAH, VT, or CWE low at the same length | Report the RULER row per length and the MK-NIAH and VT tasks separately |
| Applying an example share as a token share | Long SFT slice about 50 times smaller in tokens than intended when 0.1% of examples is applied as 0.1% of tokens | Compute token share = example share × long mean length / overall mean length (§4 worked example) |
| Adding synthetic long SFT after extensive long continued training | Long average after SFT drops relative to short-only SFT | Run 0% and 1% synthetic arms ([[prolong]] Table 8) |
| Short-only SFT on a model with limited long continued training | Long scores after SFT fall below the pre-SFT checkpoint | Evaluate the base and SFT checkpoints on the same long suite ([[excerpts/llama3-staged-schedule]] §4.3.4) |
| Packed long SFT without per-sequence loss weighting | Training favors long single-sequence packs; long-chat scores lower than naive batching | Compare packing with and without loss weighting on LongBench-Chat ([[excerpts/longalign-pipeline]] Table 4) |
| Training on needle-style factual data | TriviaQA or NQ-Open drops after long-context fine-tuning | Include knowledge QA in the short regression suite ([[excerpts/artificial-needles-kv-retrieval]] Table 2) |
| Generator produces single-hop questions labelled as multi-hop | High QA scores, low gains on 2WikiMQA, MuSiQue, HotpotQA | Hand-label a sample for multi-hop rate before training ([[excerpts/longmit-mimg]] Abstract) |
| Averaging over needle depth | Mean accuracy acceptable; middle positions low | Plot accuracy by position and report best-minus-worst gap ([[lost-in-the-middle]]) |
| Evaluation distractors unlike real retrieval | Gains on random-distractor MDQA, none on retrieved-distractor MDQA | Test with retrieved distractors ([[excerpts/artificial-needles-kv-retrieval]] Fig. 10) |

## Check your understanding

1. RULER reports Llama 3.1 8B with effective length 32K and GradientAI's Llama-3 70B with 16K. Using the Table 3 rows, explain why the second model's 32K score of 85.4 changes its reported effective length by a factor of two, and what a reader should report instead of the single number.
2. HELMET finds NIAH correlates with ∞Bench QA at 0.63 and HotpotQA RAG at 0.88. Explain which properties of the NIAH generator make it a weaker predictor, using RULER's Yi-34B failure analysis and NoLiMa's lexical-overlap result.
3. ProLong found short-only SFT best, while Llama 3.1 found short-only SFT caused regressions. Using the authors' two hypotheses and the SmolTalk observation, explain what property of each pipeline could produce opposite results, and design the smallest experiment that would distinguish the hypotheses.
4. In the LongAlign worked example, the 200-token sequence in a pack of one receives weight 0.50 instead of 0.25. Explain why long-context SFT data makes this bias larger than in short chat data, using the target/sequence ratios 0.015 (LongAlign-10k) and 19.3 (ShareGPT, read as a percentage).
5. Llama 3.1's QA generator reads an 8K chunk but the training example contains the whole document. Explain what the model must learn from such an example that the generator did not need, and which failure this design risks when the chunk's content is duplicated elsewhere in the document.
6. Artificial Needles data kept TriviaQA unchanged while NIAH-style data lowered it by 6.33 points. Give the authors' causal explanation, state its evidence status, and describe a control experiment that would test it.
7. ProLong's dependency count says one document of length n·d has (n − 1)(d − 1) more spans of length d than n documents of length d. Explain why this argument supports document masking for packed short data rather than contradicting it.
8. Qwen2.5-1M's offline RL used only pairs of at most 8,192 tokens and improved LongBench-Chat. State what this does and does not show about where long-context ability must be trained, and which additional measurement would be needed to claim no long-context loss.

## Connections

- Previous: ch-27 — Agentic Trajectory Data.
- Next: ch-29 — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification.
- Dependencies: ch-27 — Agentic Trajectory Data; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation.
- Design pattern used here: ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix.
- Context-extension methods and mixtures: ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression.
- Deeper long-data synthesis: ch-29a — Long-Document Synthesis for Continued Pretraining and Long-Context SFT; ch-29b — Long-Conversation and Accumulating-Context Synthesis.
- Long-context share in SFT mixtures: ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares.
- Negatives: ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- Long-context RL: ch-44a — Length in RL: Overlong Responses, Length Control, and Long-Context RL.

## Sources

- [[needle-in-haystack-data]] — NIAH harness defaults, GPT-4 judge scoring, original run counts.
- [[excerpts/ruler-task-family]] — RULER task families, protocol, effective-length threshold, Table 3 rows, Yi-34B failure analysis.
- [[ruler]] — library card for RULER (task list).
- [[babilong]] — bAbI-in-PG19 construction, 10–20% context use, QA1–QA3 results, QA1 fine-tuning transfer, limitations.
- [[helmet]] — correlations of synthetic and RAG tasks with application categories; NIAH saturation at 128K.
- [[nolima]] — accuracy drop when question-needle lexical overlap is removed.
- [[lost-in-the-middle]] — position dependence of multi-document QA accuracy.
- [[excerpts/longalign-pipeline]] — LongAlign data construction, manual check, mixing table, loss weighting equations and results.
- [[longalign]] — library card for LongAlign.
- [[excerpts/longmit-mimg]] — LongMIT / MIMG pipeline, verifier findings, dataset size, training settings, results.
- [[excerpts/llama3-staged-schedule]] — Llama 3 RoPE base, six-stage long-context gate, long SFT generators, 0.1% share, Table 7, Table 21.
- [[llama-3]], [[llama-3-recipe]] — verified Llama 3 card and recipe ledger rows.
- [[qwen-long-context-synth]] — Qwen2.5-1M stages, synthetic pre-training tasks, SFT generation, short-pair RL, DCA, short-context Table 6.
- [[prolong]], [[prolong-recipe]] — ProLong mix, short/long ratio, masking, training length, synthetic SFT table, final results.
- [[excerpts/artificial-needles-kv-retrieval]] — synthetic key-value retrieval data, transfer, retention and baseline comparison, limitation.
- [[in2-film]] — IN2 position-balanced synthetic QA and position-gap result.
- [[context-synthesis-short-to-long]] — LLM-written contexts with distractors for human QA pairs.
- [[nextlong]] — hard-negative interleaving for continued-training data.
- [[pose-synthesis]] — PoSE training window, used for correction 30.
- [[excerpts/primary-quotes-2026-09]] — primary quotes for LongAlpaca, LongChat, Fu et al. 2024, LongRoPE, and SmolTalk, used for corrections and the SmolLM2 observation.
