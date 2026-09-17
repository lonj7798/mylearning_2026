<!-- chapter: ch-29a
     track: synthetic
     kind: content
     title: Long-Document Synthesis for Continued Pretraining and Long-Context SFT
     deps: [ch-29, ch-32b, ch-32c]
     sources: [[llama-3]], [[long-context-llama3]], [[excerpts/llama-3-long-context-sft]], [[qwen-long-context-synth]], [[yi]], [[excerpts/yi-long-context-recitation]], [[chatqa-2]], [[in2-film]], [[hierarchical-million-token-synth]], [[nemotron-nano-2]], [[quest-query-centric-synthesis]], [[nextlong]], [[ultralong-128k-to-4m]], [[prolong]], [[qwenlong-l1-5]], [[loongrl]], [[artificial-needles-real-haystacks]], [[longmit]], [[excerpts/longmit-mimg]], [[context-synthesis-short-to-long]], [[clipper-compression-synth]], [[bootstrap-context-length]], [[smollm-3]], [[excerpts/smollm3-long-context-merge]], [[longcite]], [[nolima]], [[lost-in-the-middle]], [[agentinstruct]]
     figures: figures/long-sequence-layouts.html
     revised: 2026-09 (generality revision)
-->

# Chapter 29a — Long-Document Synthesis for Continued Pretraining and Long-Context SFT

> **Core insight.** A synthetic long training example teaches long-context use only if its target depends on tokens far from the target and cannot be produced from nearby tokens or from parametric knowledge. In the two comparisons that hold the data amount fixed, generators that enforce this dependency win: retrieved hard-negative chunks between the pieces of one document give a HELMET+RULER average of 62.58 against 52.85 for random concatenation at 128K on Llama-3-8B ([[nextlong]] Table 1), and a question hidden behind a UUID chain gives 72.4 against 66.2 for the same amount of plain long multi-hop QA in RL ([[loongrl]] Table 4). Training on QA whose evidence segments are shuffled to random depths also moved a 7B model's worst-to-best position gap from 56.2 to 13.9 points, measured against the untrained backbone rather than against another generator ([[in2-film]] Table 1). Synthetic long data can also narrow a model: training on long data that contains facts lowered Mistral-7B TriviaQA by 2.19 to 6.33 points while fact-free key-value data changed it by +0.11 ([[artificial-needles-real-haystacks]] Table 2), and long-context training runs report drops on individual short benchmarks from 0.1 to 16.9 points (§7).
>
> **Guideline.** When long documents at the target length are scarce, build continued-pretraining sequences whose dependent pieces are separated by similar but irrelevant text (NExtLong, Quest) or followed by QA about random chunks (Nemotron Nano 2), because each of these beat plain concatenation or no synthetic data in its own ablation (§2). When long SFT data is generated, let the generator read a part of the document (a chunk, an outline, a summary, retrieved passages) and train the student on the full document, then remove examples a model answers without the document, because a generator reading the whole book produced 73.1% erroneous claims against 16.7% from outlines ([[clipper-compression-synth]] Table 1) and closed-book-answerable questions measure knowledge, not context use ([[qwenlong-l1-5]] §3). When the base model already had extensive long continued training, test short-only SFT before adding synthetic long SFT, because two 8B studies found short-only SFT sufficient ([[prolong]] Table 8; [[ultralong-128k-to-4m]] §3.2) while two others found it insufficient ([[excerpts/llama-3-long-context-sft]] §4.3.4; [[bootstrap-context-length]] Table 3). In all cases, measure short benchmarks, knowledge benchmarks (TriviaQA, NQ), and instruction following (IFEval) before and after.

## Why this chapter matters for a general-purpose model

A general-purpose model receives inputs such as a 200-page contract, a code repository, a set of retrieved web pages, or a long tool log, and it must answer from that input rather than from memory. Natural documents at 128K tokens and beyond exist in few domains: ProLong counts 98.8B tokens of code repositories of at least 64K tokens against 15.3B tokens of such CommonCrawl documents ([[prolong]] §3.1, Table 3), and Quest attributes the weaker result of existing long documents to the fact that they "only exist in a few domains" ([[quest-query-centric-synthesis]] Table 5). Human annotation of long inputs is described as "largely impractical" by Llama 3 ([[excerpts/llama-3-long-context-sft]] §4.3.4) and as a task where annotators "struggle to both formulate challenging questions ... and exhaustively verify answers within contexts exceeding 32k tokens" by QwenLong-L1.5 ([[qwenlong-l1-5]] §3). Synthesis fills the gap at three pipeline positions: continued pretraining at long sequence length (mid-training, [[ch-32b]]), long-context SFT, and long-context RL data. Evaluation of the result belongs to [[ch-32c]]; [[ch-28]] introduced the early generators (LongAlign, LongMIT, NIAH-style training) and the SFT-share disagreement, and this chapter goes step by step through the generators, the dependency they enforce, the verification they apply, and the measured cost to general ability. Long conversations and accumulating agent contexts are in [[ch-29b]].

The three generality questions for this stage are: (1) which generator choices make the long-context skill transfer to real tasks that were not synthesized; (2) which choices cause narrowing, such as hallucination from new facts, loss of short-context or instruction-following ability, or a model that retrieves only literal matches; (3) which measurements detect these effects.

## §1 What makes a synthetic sequence depend on long context

**Definition.** A training example has a *long-range dependency* when the tokens that determine the target lie at a distance from the target that exceeds the local window a model could use to predict it. *Evidence* is the set of spans a correct target requires. *Evidence depth* is the start position of an evidence span divided by the context length.

**Measurable problem.** Natural text "often exhibits weak long-distance associations" ([[qwen-long-context-synth]] §3), so next-token loss on a long document is dominated by nearby tokens; IN2's authors state the same hypothesis for SFT data, where system messages and instructions sit at the start ([[in2-film]] §1, Interpretation). The measured symptom is position dependence: GPT-3.5-Turbo scores 75.8% with the answer document first among 20, 53.8% at position 10, and 56.1% with no documents at all ([[lost-in-the-middle]] App. G Table 6, Table 1). At position 10 the documents lower accuracy below the closed-book score.

**Five properties a generator can control.**
1. *Evidence count*: one span (retrieval) or several spans (integration).
2. *Evidence spread*: token distance between evidence spans and between evidence and target.
3. *Depth balance*: whether evidence depth is uniform over the window.
4. *Lexical overlap* between question and evidence: literal overlap lets retrieval succeed by string match. NoLiMa's needle set has ROUGE-1 precision 0.069 between question and needle against 0.905 for vanilla NIAH, and GPT-4o falls from 99.3 to 69.7 at 32K on it ([[nolima]] Table 1, Table 3).
5. *Closed-book answerability*: whether a model answers correctly without the document.

**Worked example (lexical overlap).** Evidence: "The Berlin office of the company opened in 1998." Question A: "What year did the company open its Berlin office?" Content words of A are {year, company, open, Berlin, office}; after stemming, 4 of 5 appear in the evidence, a unigram precision of 4/5 = 0.8. Question B: "In which year did the firm begin operating in Germany's capital?" Content words {year, firm, begin, operating, Germany, capital} share none with the evidence, precision 0/6 = 0. Question B requires the association Berlin → capital of Germany, as NoLiMa's needles do; QwenLong-L1.5 creates the same effect by obfuscating entities, for example "the year ending with 5 in the late 20th century" ([[qwenlong-l1-5]] §3.2).

The interactive figure [figures/long-sequence-layouts.html](figures/long-sequence-layouts.html) draws the training sequence of five generators from this chapter on one normalized axis; use it to compare where evidence, distractors, and questions sit, and to reproduce the worked numbers in §2 and §7.

## §2 Continued-pretraining-side synthesis

Continued pretraining (CPT) here means next-token training on long sequences after main pretraining ([[ch-32b]]). The synthesis question is how to assemble a sequence of length T from shorter documents so that predicting later tokens requires earlier ones.

### §2.1 Concatenation baselines and query-centric grouping (Quest)

**Definitions.** *Standard* concatenation places random documents in one sequence. *KNN* places each document with its most similar retrieved documents. *ICLM* orders similar documents by a traveling-salesman path ([[quest-query-centric-synthesis]] §4.2).

**Problem.** Random neighbours give no reason to look back; near-duplicate neighbours give redundancy. Quest's Figure 2 plots performance rising and then falling as within-sequence similarity increases (Interpretation by the authors).

**Mechanism (Quest §3).**
1. A doc2query model predicts queries for each document.
2. RAKE extracts keywords from the queries; keywords with score below 3.0 and non-informative phrases such as "best way" are removed; one remaining keyword is chosen at random as the document's key.
3. Documents with the same key form one group in an inverted index.
4. Rare keys are oversampled; the best split ratio for the oversampled set is 10–30% of keys (App. A.4).
5. Documents are drawn without replacement from a sampled key group and concatenated to length L.

**Evidence.** Same 30B Pile tokens for all methods, only the arrangement differs. At 128K, Longbook QA for Pythia-6.9B: Standard 14.47, KNN 13.38, ICLM 14.92, Quest 17.95; at 12B: 17.81, 16.42, 18.44, 18.92 (Table 2). The 7-task short average is 0.4830 for Pythia and 0.4831 after Quest, against 0.4769 after KNN (Table 3). Result (single study, Pythia and one Llama-3-8B run).

**Limit.** A shared keyword does not guarantee that any token in document j is needed to predict document j+1. Quest raises the probability of a useful dependency; it does not construct one.

### §2.2 Hard-negative interleaving (NExtLong)

**Definition.** NExtLong splits one short *meta-document* into *meta-chunks* and inserts after each meta-chunk its top-k most similar chunks retrieved from the corpus. The inserted chunks are *hard negatives*: text that looks relevant but belongs to other documents ([[nextlong]] §3.1).

**Mechanism.**
1. Split the meta-document at newlines into meta-chunks of at most granularity s (s = 2048 in the released runs; unit not stated, App. D.1 computes it in characters).
2. Embed every corpus chunk of the same granularity into a FAISS index.
3. For meta-chunk m_i, retrieve n_i1, ..., n_ik and form l_i = [m_i, n_i1, ..., n_ik]; the sequence is [l_1, ..., l_p]. A negative is not reused across meta-chunks.
4. Train with next-token loss on all tokens, negatives included (§3.2).

**Formula (App. D.1, Eq. 8–12).**

  k = (T × E × w − S) / (p × s)

where T is the target length in tokens, E the tokenizer encoding rate (characters per token), w = 1.5 an adjustment factor, S the meta-document length in characters, p the number of meta-chunks, and s the granularity.

**Worked example.** T = 131,072, E = 4, S = 20,000, p = 10, s = 2,048 (E, S, and p are illustrative inputs). T × E × w = 131,072 × 4 × 1.5 = 786,432 characters. Subtract S: 766,432. Divide by p × s = 20,480: k ≈ 37.4, so about 37 hard negatives follow each meta-chunk. The meta-document is 20,000 / 786,432 ≈ 2.5% of the characters. To predict meta-chunk i+1 from meta-chunk i, the model must bridge about 37 × 2,048 ≈ 76,000 characters of similar text.

**Evidence.** Llama-3-8B base, 128K, about 4B tokens, identical training configuration for all methods: HELMET+RULER average NExtLong 62.58, Quest 55.25, Standard 52.85, KNN 50.97, ICLM 50.37 (Table 1). Negatives chosen by top-k similarity were best among five selection strategies (Fig. 7; values not printed); meta-chunk at the head of l_i 62.58 versus tail 60.01 and random 58.95 (Table 5). Short-text 7-task average 63.83 against 63.75 for the base model, with LAMBADA 75.66 → 72.06 (Table 4). Result (single study, one base model).

**Limit.** Self-repeated meta-chunks behave as false negatives and lower performance (§5.4). Artificial Needles found that key-value training with random distractors did not improve MDQA when the distractors were relevant retrieved documents ([[artificial-needles-real-haystacks]] §4, Fig. 10); NExtLong's retrieved negatives target that case, but the two papers use different models and tasks.

### §2.3 QA appended to real documents and structural tasks

**Nemotron Nano 2 (Phase LC).** Seeds are academic documents longer than 32K tokens. Each is split into 1,024-token chunks; 10% of chunks are sent to Qwen-2.5-72B-Instruct, which writes one QA pair per chunk; all QA pairs are concatenated and appended after the document ([[nemotron-nano-2]] §2.6). The document-QA data receives 20% of the blend and every Phase 3 source is scaled to 80% of its weight.

*Worked example.* A 64,000-token document has ⌈64,000 / 1,024⌉ = 63 chunks (62 full, one of 512 tokens). 10% of 63 is 6.3, so about 6 QA pairs are appended (derived). If the sampled chunk covers tokens 2,048–3,071, the distance from its last token to the start of the appended QA block is 64,000 − 3,072 = 60,928 tokens. A Phase 3 source with weight 25% becomes 25% × 0.8 = 20%.

*Evidence.* Nemotron-H 8B ablation, RULER-128k: trained at 256k without synthetic data 70.19, at 256k with synthetic data 79.04, at 512k with synthetic data 81.04, at 128k with synthetic data 73.68 (Table 4). Result (single study). Loss masking on the appended QA and QA verification are not reported.

**Qwen2.5-1M structural tasks.** Pretraining data is augmented with fill-in-the-middle, keyword- and position-based paragraph retrieval, and paragraph reordering ([[qwen-long-context-synth]] §3). Each task has a label derived from the document itself, so no generator model is needed. Proportions and ablations are not reported.

**Yi recitation QA in CPT.** Yi's 5B-token CPT mixture includes "multi-document question-answering synthetic data, where ... the answer contains a recitation of the related paragraph before the answer" ([[excerpts/yi-long-context-recitation]] §7.1). Proportions and an ablation are not reported.

### §2.4 Document boundaries: masking versus separators

**Definitions.** *Cross-document masking* blocks attention between documents packed into one sequence. A *separator* is a token placed between documents; when attention is not masked, the model can attend across the separator.

**The conflict.**
- ProLong masks attention across document boundaries and reports long 54.6 / short 65.5 with masks against 53.6 / 64.9 without, in a 5B-token Llama-3-8B ablation ([[prolong]] App. B.2, Table 20). SmolLM3 also uses intra-document masking ([[excerpts/smollm3-long-context-merge]]).
- ChatQA 2 found separating documents with "<s>" more effective than Llama 3's BOS/EOS tokens and hypothesizes that BOS/EOS "signal the model to ignore previous chunks of text" (Interpretation; App. B shows NIAH heatmaps for a 2B-token run without numbers) ([[chatqa-2]] §3.1).
- UltraLong uses "<s>" and explicitly does not apply the cross-document mask. Removing the separator while keeping full attention lowers RULER <1M from 80.17 to 79.15 and InfiniteBench from 26.25 to 22.75 ([[ultralong-128k-to-4m]] Table 3).

**What the evidence does and does not show.** ProLong compares mask versus no mask. UltraLong compares separator versus no separator, both without masks. No study in the library compares "<s> without mask" against "mask" at equal data. The two setups also differ in what synthesis does: with a mask, a concatenated sequence of unrelated documents contains no cross-document dependency, so Quest-, NExtLong-, and Nemotron-style constructions require the dependent pieces to be inside one attention span; NExtLong computes the next-token loss over the whole synthesized sequence, and its 512K run applies full attention inside each synthesized document, masking only between the shorter documents packed beside it ([[nextlong]] §3.2, App. C.2). **Open question**: whether unmasked separators help because cross-document attention trains retrieval over irrelevant context, or because they add long sequences at low cost.

**Implication for a general-purpose model.** When synthetic sequences are built by concatenation, the boundary treatment must match the construction: mask across unrelated documents, and do not mask inside a constructed dependency.

## §3 SFT-side long-document generators

The design choice that separates SFT generators is *what the generator reads* versus *what the student reads*. A generator that reads the whole long input is limited by its own long-context reliability; a generator that reads a part cannot write questions that need the whole.

### §3.1 Chunk-generated QA trained with the full document

**Llama 3.** Documents from the pretraining mix are split into 8K chunks; an earlier Llama 3 writes QA pairs "conditional on randomly selected chunks. During training, the whole document is used as context." Samples are bucketed at 16K, 32K, 64K, and 128K ([[excerpts/llama-3-long-context-sft]] §4.3.4).

**Qwen2.5-1M.** Qwen2.5 writes queries from "a randomly extracted segment"; the Qwen-Agent framework writes responses over the full document using retrieval-augmented generation, chunk-by-chunk reading, and step-by-step reasoning ([[qwen-long-context-synth]] §4). The response generator is an agent that reads more than one chunk, which lets answers integrate several parts of the document.

**Yi recite-then-answer.** Multiple documents are concatenated, one or more paragraphs are sampled, a chat model writes QA from them, and "before giving the answer, we ask the model to recite or paraphrase the original paragraph" ([[excerpts/yi-long-context-recitation]] §7.1). The authors state this "discourages the hallucination behavior"; no ablation is reported (Interpretation).

**Limit.** A question written from one chunk depends on one chunk. The target distance is large, but the evidence count is one, and the question may repeat words of the chunk (§1, property 4).

### §3.2 Hierarchical summaries, missing-file code, and multi-document assembly

**Llama 3 hierarchical summarization.** 8K chunks are summarized by the strongest 8K model, then the summaries are summarized; training gives the full document with a prompt to summarize "while preserving all the important details", and QA pairs generated from the summaries ask for "global understanding of the whole long document" ([[excerpts/llama-3-long-context-sft]] §4.3.4).

**Llama 3 missing-file code reasoning.** From Python import statements, files "referenced by at least five other files" are selected; one is removed from the repository, and the model must "identify which files depended on the missing file and to generate the necessary missing code" (§4.3.4). The label (the removed file and its importers) comes from the repository, not from a generator's judgment.

**He et al. hierarchical multi-document synthesis to 1M.** Qwen-2-72B-Instruct reads 4K and 12K chunks and their summaries and writes hierarchical questions (global summary → section → chunk) and diverse questions ([[hierarchical-million-token-synth]] §3.1). Books are concatenated in one multi-turn dialogue; after each book, N1 = 5 hierarchical and 5 diverse QA pairs are added, then N2 = 9 diverse pairs about earlier books, and for each earlier book, with 60% probability, N3 = 3 follow-up questions (§3.2, Figure 3; §4.2 prints the N2 and N3 labels swapped).

*Worked example.* The 1M stage uses 200 samples × 1M tokens; the four stages sum to 2,000 × 180K + 1,280 × 350K + 600 × 650K + 200 × 1M = 1.398B tokens, matching the reported "about 1.4B" (card derived check). A 4-book sample has, in expectation, 4 × 10 = 40 per-book pairs, 3 × 9 = 27 revisit pairs, and (1 + 2 + 3) × 0.6 × 3 = 10.8 follow-ups, about 78 QA turns (derived from the Figure 3 numbers).

*Evidence.* Llama-3.1-8B-Instruct: RULER at 1M 62.95 for the 1M model against 48.81 for zero-shot RoPE scaling; InfiniteBench 54.80 against 51.31 for the base; MMLU 65.08 against 68.21 (Tables 1, 3, 9). Across seven 180K compositions, InfiniteBench ranges from 56.63 (h-h-randomized) to 59.45 (hs-hs-hs-fixed) (Table 7). No filtering or verification of generated QA is described. Result (single study).

### §3.3 Compression first, then generation (CLIPPER)

**Problem measured.** Claude-3.5-Sonnet reading whole books (average 90K tokens) produced claim pairs of which 73.1% had an error: 11.5% invalid, 28.9% citing wrong chapters, 15.4% containing explicit chapter references or quotes, 17.3% duplicates ([[clipper-compression-synth]] Table 1, 52 human-annotated claims).

**Mechanism.**
1. Compress: per-chapter outlines (5–7 events per chapter, 8,745 tokens on average against 90,437 for the book, a 10.0% compression rate) and a book summary of about 618 tokens.
2. Generate true/false claim pairs from events in at least 2 chapter outlines (book-level) or one outline plus the summary (chapter-level), each with a chain-of-thought.
3. Deduplicate, then validate claims against the outlines with GPT-4o; 59.4% were removed as duplicates and 2.4% as invalid.
4. Train with the full book as input and the chain-of-thought plus label as target.

**Evidence.** 83.3% of 66 annotated CLIPPER claims were error-free against 26.9% for the naive method; cost $0.05 against $0.07 per claim. Llama-3.1-8B-Instruct: CLIPPER-test 27.9% → 76.0%, NoCha 16.5% → 32.2%, NarrativeQA 47.7% → 49.0%, ∞Bench QA 47.8% → 46.5%. Training ProLong-512K-8B on 19K claims from short WritingPrompts stories instead gave CLIPPER-test 63.0% and NarrativeQA 31.0% (Table 2). Result (single study, 7B–8B models).

**Limit.** Claims come from events an outline model chose to keep; the authors attribute the remaining gap on human-written NoCha claims to "low-level details that may not typically appear in such outlines" (§4.1, Interpretation).

### §3.4 Human-anchored long examples

**ChatQA 2 summary insertion.** NarrativeQA provides human-written summaries of long source pages and human-written QA pairs grounded in the summaries. ChatQA 2 inserts the summary "into the corresponding long web page document at a random location, ensuring that sentence structure was not disrupted", producing 32K–128K examples whose answers are human-written ([[chatqa-2]] §3.2). NarrativeQA is then excluded from evaluation. On InfiniteBench En.Sum, ChatQA-2-70B scores 16.08 against 30.94 for Llama3.1-70B-Instruct, which the authors attribute to "the lack of summarization data in our SFT recipe" (§5.2). The inserted summary is a single evidence span that restates the source page, so the task reduces to locating one span.

**Context synthesis.** [[context-synthesis-short-to-long]] keeps 1.6k human instruction-answer pairs and lets GPT-4o-mini write the missing context (about 2,000 words), then concatenates 1 relevant and 9 irrelevant contexts. On LLaMA3.1-8B with UltraChat, the 8-task LongBench average is 23.35 without long data, 25.39 with instruction synthesis on the same 1.6k samples, and 38.57 with context synthesis (Table 3). The instructions come from the training splits of the evaluated datasets, so this is in-distribution evidence; unseen RULER multi-value NIAH moves 91.57 → 96.54 (Table 5).

### §3.5 Agent workflows that bootstrap long data from a short-context model (SelfLong)

**Mechanism** ([[bootstrap-context-length]] §3.1).
1. GPT-4o writes an instruction; a random text chunk is prepended to each call for diversity.
2. E5-mistral-7b retrieves documents from about 10M Fineweb-Edu documents.
3. Documents are split into chunks of at most 4K tokens; query-focused summarization agents summarize each chunk for the instruction and drop irrelevant content, recursively, until the summaries fit.
4. The model writes the response from the summaries.
5. Training input = instruction + raw retrieved documents; target = response. "The intermediate summaries are not utilized during training."

The student must do in one pass over up to 1M tokens what the workflow did in many short calls. Loss is averaged over all input and output tokens for long samples and over output tokens only for short samples (§3.2).

**Evidence** (Table 3, RULER, SelfLong-8B). At 1M: full data 69.6; without synthetic data 63.5; short data only (≤4K) 46.6; adjusted RoPE θ with no training 48.1; Llama-3.1-8B-Instruct as generator instead of GPT-4o 66.1. Masking prompt tokens in long samples gave 66.4 at 1M. Short-context (Table 4): IFEval 77.4 → 65.9 for 8B and 73.9 → 57.0 for 3B, while GPQA 26.8 → 30.4 and MUSR 36.9 → 41.5 for 8B. Result (single study).

### §3.6 IN2: integration questions and balanced depth

**Mechanism** ([[in2-film]] §2.1).
1. Take C4 realnewslike texts and split them into 128-token segments.
2. Fine-grained awareness: GPT-4-Turbo writes a QA pair "highly specific" to one segment s_i; the context is Shuffle(s_i, [r_j]) with random segments r_j.
3. Integration and reasoning: GPT-4-Turbo writes a question needing at least two segments of one text; required and random segments are jointly shuffled.
4. Context lengths are made evenly distributed from 4K to 32K by rejection sampling on the number of random segments.
5. About 10% of QA pairs keep the original short text; OpenOrca instruction data is added.

**Formula.** Evidence depth of segment i in a context of n segments is d_i = (i − 1) / n, where i is the segment index after shuffling. A uniform shuffle makes each d_i uniform over the n values {0, 1/n, ..., (n − 1)/n}.

**Worked example.** A 16,000-token context holds ⌊16,000 / 128⌋ = 125 segments. One evidence segment is 128 / 16,000 = 0.8% of the context. With lengths uniform on 4K–32K, the mean context is (4,000 + 32,000) / 2 = 18,000 tokens (derived).

**Evidence.** Mistral-7B-Instruct-v0.2 → FILM-7B (1.1M fine-grained + 300K integration long, 150K short QA, 200K instruction): VAL Probing average 47.3 → 85.9 and min-max position gap 56.2 → 13.9 (Table 1); LongBench average 30.6 → 39.9, with HotpotQA 42.4 → 62.1 (Table 2); MMLU 59.3 → 59.2, HellaSwag 83.6 → 79.1, ARC-C 55.9 → 52.5 (Fig. 5). Result (single study, 7B, 32K).

**Generator comparison.**

| Generator | Generator reads | Student reads | Evidence count | Label source | Verification reported |
|---|---|---|---|---|---|
| Llama 3 chunk QA | one 8K chunk | full document | 1 chunk | generator | none printed |
| Llama 3 missing file | repository structure | repository minus one file | several importers + removed file | repository | label exact by construction |
| Qwen2.5-1M | segment (query); full doc via agent (response) | full document | not reported | agent | not reported |
| He et al. | 4K/12K chunks, summaries | concatenated books, multi-turn | summary, 1 chunk, or 3 chunks (multi-hop prompt) | generator | none described |
| CLIPPER | outlines + summary | full book | 2–3 events, ≥2 chapters for book-level | generator | GPT-4o vs outlines; human sample |
| ChatQA 2 | none (human data) | page + inserted summary | 1 inserted span | human | human-written |
| SelfLong | 4K chunks via summarization agents | instruction + raw retrieved documents | several retrieved documents | generator | not reported |
| IN2 | 1 or more 128-token segments | shuffled 4K–32K context | 1 or ≥2 | generator | 10-gram decontamination only |

## §4 Forcing long-range dependency

Each technique below targets one property from §1.

1. **Spread evidence across documents (evidence spread).** QwenLong-L1.5 extracts knowledge-graph triplets across documents, samples multi-hop paths by random walk and BFS, and places "path nodes ... sparsely across multiple documents"; complexity is set by path length ([[qwenlong-l1-5]] §3.2). Its training inputs average 34,231 tokens against 11,441 for QwenLong-L1 (Table 1).
2. **Hide the question (indirection).** LoongRL's KeyChain inserts UUID key-value pairs so that one chain resolves to the true question and others to questions from other items. A prompt excerpt ([[loongrl]] §3.1):

```text
Please read the following text.
<Document 0>
<original text> {"UUIDB-n": "distracting question"} <original text>
<Document 1>
{"UUIDA-1": "UUIDA-2"}
...
{"UUIDA-n": "correct question"}
...
In the context above, there is one correct question to answer.
The correct question can only be found by following the correct
consecutive chain of key:value pairs encoded with UUID strings
(e.g., f81d4fae-7dec-11d0-a765-00a0c91e6bf6), starting from
"starting UUIDA-1".
Find the correct question first, then answer it.
```

   The question position is random and unknown until the chain is traced, so the model cannot rely on the question being at the end. With equal data amounts on Qwen2.5-7B-Instruct, KeyChain data gives a LongBench v1 multi-hop average of 72.4 against 66.2 for plain long multi-hop QA (Table 4). This is RL data (GRPO, 16K contexts); SFT use is not tested. Result (single study).
3. **Balance depth.** IN2's shuffle (§3.6). The pre-training analogue is NExtLong's meta-chunk placement, where the head position beat tail and random (§2.2).
4. **Lower question–evidence overlap.** Entity obfuscation in QwenLong-L1.5 (§1). LongMIT merges several single-hop questions into one multi-hop question; with plain Self-Instruct from Qwen2-72B, fewer than 35% of samples were multi-hop and over 40% were of poor quality, against over 85% multi-hop, high-quality, non-duplicate samples with the multi-agent framework ([[excerpts/longmit-mimg]] §1).
5. **Compute the answer from the whole context.** QwenLong-L1.5 aggregates tables across documents into one table, translates generated questions to SQL, and executes the SQL to obtain the answer ([[qwenlong-l1-5]] §3.2). The answer depends on every row the query touches.

**Implication.** Retrieval-only data (one span, high overlap) raises needle scores that already saturate: LongAlign-13B-64K scores near-perfect on NIAH and still has a 27.1-point position gap on VAL Probing ([[in2-film]] Table 1). Integration, indirection, and low overlap are the properties that separate the generators with the largest gains in §2–§3.

## §5 Fact-free programmatic retrieval data

**Definition.** Programmatic data is generated by code with exact labels and contains no world facts: for example lists of dictionaries with random integer keys and values ([[artificial-needles-real-haystacks]] §2).

**Mechanism.**
1. Build 85 dictionaries with 3–4 keys each; keys and values are random 3–4 digit integers; each prompt is about 3,900 tokens.
2. Ask for the value of one key and the dictionary it is in; provide an answer template so the loss concentrates on the value (Fig. 3–4).
3. Fine-tune on 350 such tasks, 2 epochs, LR 5×10^-6, loss on the answer only (App. A.1).

**Worked example.** 85 dictionaries × 3.5 keys on average = about 297 key-value pairs in 3,900 tokens, about 13 tokens per pair including brackets and dictionary labels (derived). The gold pair is about 13 / 3,900 ≈ 0.3% of the prompt.

**Evidence** (Mistral-7B-Instruct-v0.1, Table 2, roughly equal training tokens per dataset). The abstract prints the baseline TriviaQA drops as 2.33% to 6.19% while Table 2 prints −2.19 to −6.33 points; the table values are used here. TriviaQA change: key-value data +0.11; MultidocQA −2.43; IN2 data −2.19; NIAH-style data −6.33. NQ-Open: +0.37, −2.91, −1.81, −6.73. MMLU changes stay within ±0.6 for all four. On the real 20-document MDQA task, GPT-3.5 Turbo improves by 10.5% at position 10 (abstract). The authors explain the knowledge losses by the factual content of the other datasets, citing prior work that fine-tuning on unknown facts encourages hallucination (Interpretation; hallucination is not measured directly).

**Conditions and limits.** Tested at about 4K tokens (one 24K run in §3.5). The key-value models did not improve MDQA with relevant distractors (§4, Fig. 10), and some baselines beat the key-value data on MDQA or FLenQA (§3.4). The IN2 drop here is on a different backbone and token budget from FILM-7B, whose MMLU changed by 0.1 ([[in2-film]] Fig. 5).

**Implication for a general-purpose model.** Fact-free data is a low-risk component for teaching position-independent lookup. It does not teach integration or robustness to semantically similar distractors, so it complements, rather than replaces, the constructions in §2 and §4.

## §6 Verification of long answers

Verification in long-document synthesis has to detect two failure types: the example is wrong (label error), and the example is right but does not require the context (dependency failure).

1. **Closed-book filter (dependency).** Remove the document and ask the model; discard examples it answers correctly ([[qwenlong-l1-5]] §3, "Knowledge Grounding Check"). The motivation is measurable: at position 10 of 20 documents GPT-3.5-Turbo is below its own closed-book accuracy ([[lost-in-the-middle]] Table 6).
2. **Context-robustness check (label and dependency).** Add irrelevant documents; discard examples whose pass@k drops to zero (QwenLong-L1.5 §3). *Worked example:* if the checking model's per-sample accuracy on an item after insertion is 0.1 and k = 4 (k is not reported; illustrative), P(pass@4 = 0) = 0.9^4 = 0.656, so the item is discarded with probability 65.6%; at accuracy 0.5 the probability is 0.5^4 = 0.0625. The check removes items that collapse, not items that degrade. QwenLong-L1.5 keeps 14.1K of 42.7K synthesized examples (33.0%) after all filters, deduplication, and decontamination (§3); the share removed by each check is not reported.
3. **Execution-grounded answers (label).** SQL executed over aggregated tables (§4, item 5) and the removed-file construction in Llama 3 (§3.2) fix the label by construction; whether Llama 3 checks the generated missing code by execution is not reported.
4. **Citation targets (dependency).** LongCite segments answers into statements, retrieves 128-token chunks for each sentence, extracts sentence-level citations, and discards an instance "if less than 20% of the statements in the answer have citations" ([[longcite]] §3.1). *Worked example:* an answer of 12 statements with 2 cited statements has 2/12 = 16.7% and is discarded; with 3 it has 25% and is kept. For LongCite-9B, data filtering raises citation F1 from 61.2 to 63.6 with correctness 67.4 → 67.6 (Table 5); LongCite-8B reaches citation F1 72.0 against 65.6 for GPT-4o (Table 2).
5. **Verification against a compressed representation (label).** CLIPPER validates claims against outlines, where Claude's verification accuracy is 98.6% against 40.3% on full books in NoCha (§2.2); human annotation of samples gives the error rates in §3.3.
6. **Dataset-level dependency diagnostic.** Fine-tune with and without the long context; a small gap means the data does not teach context use. For instruction-synthesis data, adding the context gave no improvement over context-free tuning ([[context-synthesis-short-to-long]] §4.3, Figure 4).
7. **Error audits.** CLIPPER reports human annotation of 52 naive and 66 CLIPPER claims, with any-error rates of 73.1% and 16.7% ([[clipper-compression-synth]] Table 1). LongMIT reports the multi-hop share (below 35% under Self-Instruct) and the poor-quality share (over 40%) without stating whether the labels are human or model-assigned ([[excerpts/longmit-mimg]] §1). The share of generated long examples that humans had to edit is not reported by any source in this chapter.

## §7 Length distribution and short/long mixing

**Definitions.** *Example share* is the fraction of training examples that are long; *token share* is the fraction of training tokens in long examples. A long example of 38K tokens contributes about 45 times the tokens of an 846-token average example.

**Worked example (Llama 3 Table 7).** Long-context data is 0.11% of SFT examples with 38,135.6 tokens on average; all data averages 846.1 tokens. Token share = 0.0011 × 38,135.6 / 846.1 = 41.95 / 846.1 ≈ 0.050. The "0.1%" is an example share that corresponds to about 5% of SFT tokens (derived; [[excerpts/llama-3-long-context-sft]] Table 7). The figure recomputes this from all six rows.

**Measured general-ability deltas.**

| Study | Setting | Long data | General-ability change | Locus |
|---|---|---|---|---|
| IN2 / FILM-7B | Mistral-7B, SFT 4K–32K | ~80% of examples long synthetic QA | MMLU −0.1; HellaSwag −4.5; ARC-C −3.4; GSM8K +4.1 | [[in2-film]] Fig. 5 |
| He et al. | Llama-3.1-8B-Instruct → 1M | 200 × 1M-token dialogues | MMLU 68.21 → 65.08; LongBench 48.11 → 46.42 | [[hierarchical-million-token-synth]] Tables 2–3 |
| SelfLong-8B-1M | Llama-3.1-8B → 1M | 69K long-input (4.6B tokens) + 10K long-output (77M tokens), inside about 8.3B training tokens | IFEval −11.5; MMLU Pro −2.2; GPQA +3.6 | [[bootstrap-context-length]] Table 4 |
| UltraLong-8B-1M | CPT 1B tokens + short-only SFT | none in SFT | 5-task average 61.45 → 62.47 | [[ultralong-128k-to-4m]] Table 2 |
| ChatQA-2-8B (new) | 1.6M-sample SFT vs Llama3.1-8B-Instruct | long SFT stage | MMLU 65.73 vs 67.59; HumanEval 66.46 vs 70.73 (different pipelines) | [[chatqa-2]] Table 8 |
| Qwen2.5-14B-Instruct-1M | vs 128K version | 75% max-length CPT + long SFT | GPQA 45.5 → 39.9; IFEval 81.0 → 84.3 | [[qwen-long-context-synth]] Table 6 |
| Yi-34B-200K | 5B-token CPT | recitation QA in CPT | MMLU 76.32 → 75.56 | [[excerpts/yi-long-context-recitation]] Table 6 |

The drops in the table range from 0.1 (IN2 MMLU) to 11.5 (SelfLong-8B IFEval), and SelfLong-3B IFEval drops 16.9; the same runs also show gains, up to +4.1 (IN2 GSM8K), +3.6 (SelfLong GPQA), and +3.3 (Qwen2.5-14B IFEval). The studies differ in base model, stage, and benchmark, so they do not identify one safe long share. Five of the seven rows show at least one benchmark dropping by more than 2 points; UltraLong-8B-1M's largest single drop is 1.89 (MMLU-Pro 44.33 → 42.44, Table 2) and Yi reports MMLU only (Result, across studies).

**Short-only SFT: the disagreement.** Llama 3 found short-only SFT regressed long context ([[excerpts/llama-3-long-context-sft]] §4.3.4) and SelfLong found short-only (≤4K) training worse beyond 128K than no training at all (46.6 vs 48.1 at 1M, [[bootstrap-context-length]] Table 3). ProLong found every synthetic long SFT share from 1% to 50% of tokens below 0% ([[prolong]] Table 8) and UltraLong reached RULER <1M 79.1 with short-only SFT ([[ultralong-128k-to-4m]] Table 1). [[ch-32b]] §7 records the hypotheses; they remain an **Open question**. One observable difference: ProLong and UltraLong ran a dedicated long CPT stage on natural documents before short-only SFT, while SelfLong extended context during the long SFT stage itself.

**Packing interacts with mixing.** Nemotron Nano 2 packed SFT samples to about 128K tokens "to preserve long-context ability", and attributes a tool-calling regression to that packing; its next stage was trained without concatenation ([[nemotron-nano-2]] §3.2).

**Recovery by merging.** SmolLM3 traced a RULER drop to reasoning mid-training and 24K-token APO data, and recovered the base model's RULER up to 128K with θ = 0.9 θ_APO-soup + 0.1 θ_mid-training ([[excerpts/smollm3-long-context-merge]]); values before and after are not printed. Merging is covered in [[ch-30c]].

## §8 Long-context checks for a synthetic long-document set

These checks apply to the data before training and link to the evaluation methods of [[ch-32c]].

1. Evidence-depth histogram in 10 bins; a uniform generator gives about 10% per bin.
2. Question–evidence unigram precision (§1); report the share above 0.5.
3. Closed-book accuracy of a reference model on a 500-example sample.
4. Evidence-count distribution (one span versus several).
5. Length histogram in the buckets used for training (Llama 3: 16K, 32K, 64K, 128K).
6. Held-out evaluation families that the generator did not target: for example NoCha for claim data, LongBench v2 or HELMET RAG for QA data, and NoLiMa for literal-match dependence.

## Negative samples and negative feedback

Long-document synthesis uses three of the four meanings of "negative" defined in the course standard ([[ch-31a]], [[ch-43a]]).

**1. Where negatives come from.**
- Distractor documents and hard-negative chunks: random segments (IN2), retrieved similar chunks (NExtLong), distracting UUID chains and questions from other items (LoongRL), 9 irrelevant generated contexts (context synthesis), irrelevant documents inserted to reach target length (QwenLong-L1.5). Labeled by construction.
- False claims in minimal pairs (CLIPPER): a true claim with one detail changed, labeled False by the generator and checked against outlines.
- Rejected generated examples: closed-book-answerable (QwenLong-L1.5), robustness-collapse (QwenLong-L1.5), fewer than 20% cited statements (LongCite), duplicates and invalid claims (CLIPPER, 59.4% and 2.4%). False-negative rates of these filters are not reported, except CLIPPER's GPT-4o filter disagreeing with the authors on 1 of 72 pairs.
- Wrong rollouts in long-context RL (LoongRL, QwenLong-L1.5), labeled by a rule-based or judge reward.

**2. What current practice does with them.**
- Distractors and hard negatives are *negative as content* (type 2): they sit in the input, and the loss is ordinary cross-entropy on the target. In CPT constructions such as NExtLong, the loss is computed on every token, so the hard-negative chunks are also predicted as ordinary text; they are negative only with respect to the dependency ([[nextlong]] §3.2).
- False claims with the target "False" plus a chain-of-thought are also type 2.
- Rejected examples have *negative marginal value* (type 1) and are discarded.
- Wrong RL rollouts receive a negative advantage, which is *negative as gradient* (type 4); none of the sources in this chapter measures the share of improvement from those negatives.

**3. Mechanism.** For type 2 the loss is

  L = − Σ_t log p_θ(y_t | x_doc, x_neg, y_<t)

where y_t is the t-th target token, x_doc the evidence-bearing context, x_neg the distractor tokens, and y_<t the preceding target tokens. The gradient raises the probability of the correct target given a context that includes the distractors; no term lowers the probability of any distractor. The attention pattern that ignores x_neg is learned only because attending to x_neg does not reduce L. For type 4 in RL, the logit gradient of a sampled token y is ∂ log p_y / ∂ z_j = 1[j = y] − p_j; a negative advantage multiplies this by a negative number, lowering z_y and moving probability mass to the other tokens in proportion to p_j, so mass concentrates on the currently most likely alternatives ([[ch-43a]]).

**4. Evidence with numbers.**
- Benefit of distractors: NIAH-trained SFT without distractors (needle only) degraded when distracting information appeared at test time, while training with 1K-essay distractors stayed above 90% at 32K ([[context-synthesis-short-to-long]] §3.3); concatenating n = 10 contexts (9 irrelevant) gave 38.57 against 37.67 with n = 1 (Table 11).
- Hard versus random: similarity-selected negatives were best among five strategies ([[nextlong]] Fig. 7); random-distractor key-value data did not improve MDQA with relevant distractors ([[artificial-needles-real-haystacks]] Fig. 10).
- False claims are harder to learn: Qwen-CLIPPER fails 97 false against 37 true claims out of 1,000 book-level pairs ([[clipper-compression-synth]] §4.6).
- Rejection by citation coverage: +2.4 citation F1 (§6).

**5. Controls.** Choose negatives similar enough to compete (retrieved, same keyword, same task family), but exclude self-repeats and near-duplicates, which act as false negatives ([[nextlong]] §5.4). Keep the negative out of the target: mask loss on distractor tokens in SFT. For false claims, pair every false claim with its true counterpart and score the pair jointly, as CLIPPER's accuracy metric does (§2.1).

**6. Diagnostics.** Accuracy split by distractor type (none, random, retrieved); position-gap metrics (VAL Probing gap); true-claim versus false-claim accuracy; closed-book accuracy before and after training; for RL, entropy and pass@k at large k split by advantage sign ([[ch-44a]]).

**7. Effect on generality.** Distractor training is the mechanism behind robustness to retrieved but irrelevant context, which RAG-style use requires. Unanswerable long-document questions, where the correct target is an abstention, are not trained by any generator in this chapter; [[agentinstruct]] makes some short reading-comprehension questions unanswerable (§2.1) without an ablation. Whether long-context models trained on distractors over-answer or over-refuse when the evidence is absent is an **Open question**.

## Recipe

Values are quoted from source cards or chapter excerpts; "verified" dates refer to the card or excerpt check.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3 405B | 405B | long-context | stages; tokens | six stages 8K → 128K; approximately 800B tokens | arXiv:2407.21783v3 §3.4.2 | verified 2026-09-15 | stage gate: short evals recovered and NIAH solved; no per-stage ablation |
| Llama 3 (post-trained models) | 8B, 70B, 405B | SFT | long-context share | "0.1%" (text); 0.11% of examples, 38,135.6 avg tokens (Table 7) | arXiv:2407.21783v3 §4.3.4, Table 7 | verified 2026-09-15 | "careful ablations"; no table printed |
| Llama 3 | 8B, 70B, 405B | SFT | chunk size; length buckets | 8K chunks; 16K, 32K, 64K, 128K | arXiv:2407.21783v3 §4.3.4 | verified 2026-09-15 | no ablation reported |
| Nemotron-Nano-12B-v2-Base | 12B | long-context | sequence length; LR; batch; tokens | 524,288; constant 4.5e-6; 12 sequences (about 6M tokens); 18.9B | arXiv:2508.14444v4 §2.6 | verified 2026-09-15 | Table 4 (Nemotron-H 8B): 512k 81.04 vs 256k 79.04 RULER-128k |
| Nemotron-Nano-12B-v2-Base | 12B | long-context | synthetic doc-QA construction; blend weight | 1,024-token chunks, 10% sampled, Qwen-2.5-72B-Instruct, QA appended; 20% (Phase 3 scaled to 80%) | arXiv:2508.14444v4 §2.6 | verified 2026-09-15 | Table 4 (8B): with synthetic 79.04 vs without 70.19 at 256k |
| Qwen2.5-7B/14B-1M | 7B, 14B | long-context | length mix | 75% at current max length, 25% shorter | arXiv:2501.15383v1 §3 ([[qwen-long-context-synth]]) | verified 2026-09-14 | no ablation reported |
| Llama-3-8B-NExtLong-128K | 8B | long-context | granularity; data; LR | s = 2048; 32,000 × 128K samples (about 4B tokens); 4e-5 cosine | arXiv:2501.12766v2 §4.2, §5.3, Table 8 ([[nextlong]]) | verified 2026-09-14 | Fig. 6: 2048 best overall of 5 granularities |
| ProLong-8B | 8B | long-context | document masking | cross-document masking | arXiv:2410.02660 App. B.2 ([[prolong]]) | verified 2026-09-14 | Table 20: 54.6 / 65.5 vs 53.6 / 64.9 without |
| UltraLong-8B-1M | 8B | long-context | tokens; separator; masking; LR | 1B tokens; "<s>"; no cross-document mask; 3e-5 | arXiv:2504.06214v1 §3.1, Table 5 | verified 2026-09-15 | Table 3: separator removed 79.15 vs 80.17 RULER <1M |
| UltraLong-8B-1M-Instruct | 8B | SFT | data; LR; batch | 100K short (<8K) examples; 5e-6; 128 | arXiv:2504.06214v1 §3.2 | verified 2026-09-15 | no long-SFT ablation reported |
| Llama3-ChatQA-2-70B | 70B | long-context | CPT tokens | "10 billion tokens" and "2000 steps (8B tokens in total)" both printed | arXiv:2407.14482v3 §3.1 | conflict | App. B: "<s>" vs BOS/EOS NIAH heatmap, 2B tokens |
| FILM-7B | 7B | long-context SFT | examples; lengths; LR; batch; epochs | 1.1M + 300K long, 150K short QA, 200K instruction; 4K–32K; 1e-6; 128; 1 | arXiv:2404.16811v2 §2.1–2.2 ([[in2-film]]) | verified 2026-09-14 | no ablation reported |
| Llama-3.1-8B → 1M (He et al.) | 8B | long-context SFT | samples × length; LR; epochs | 200 × 1M; 6e-5; 1 | arXiv:2504.12637v1 §4.2 ([[hierarchical-million-token-synth]]) | verified 2026-09-14 | no ablation of sample count |
| SelfLong-8B-1M | 8B | long-context SFT | synthetic data | 69K long-input (4.6B tokens) + 10K long-output (77M tokens); about 8.3B tokens total | arXiv:2412.18860v2 §4.1 | verified 2026-09-15 | Table 3: w/o synthetic 63.5 vs 69.6 at 1M |
| Llama-3.1-8B-CLIPPER | 8B | SFT | claims; LR; batch; epochs | 16K training claims; 1e-6; 16; 1 | arXiv:2502.14854v2 §3.1 | verified 2026-09-15 | dev-set sweep of LR {1e-5, 1e-6, 1e-7} and batch {8, 16, 32, 64} |
| LongCite-9B / 8B | 9B, 8B | SFT | data; filter; LR; steps | 44,600 LongCite + 76K ShareGPT; discard <20% cited; 1e-5; 4,000 steps, batch 8 | arXiv:2409.02897v3 §3.1, §3.3, §4.1 | verified 2026-09-15 | Table 5: filter 63.6 vs 61.2 citation F1 |
| Mistral-7B-Instruct-v0.1 (key-value) | 7B | SFT | tasks; epochs; LR; batch | 350; 2; 5e-6; 16 | arXiv:2406.19292v2 App. A.1 | verified 2026-09-15 | no ablation reported |
| QwenLong-L1.5-30B-A3B | 30B-A3B | RL | data yield; input length | 42.7K → 14.1K; average 34,231 tokens | arXiv:2512.12967v1 §3, Table 1 | verified 2026-09-15 | no ablation of the two checks reported |
| LoongRL-7B | 7B | RL | KeyChain items; context | 7,500 of 21,024; about 16K tokens | arXiv:2510.19363 Table 1 ([[loongrl]]) | verified 2026-09-14 | Table 4: 72.4 vs 66.2 plain multi-hop |
| SmolLM3-3B | 3B | merge | weights | 0.9 APO soup + 0.1 mid-training checkpoint | huggingface.co/blog/smollm3, "Model Merging" | verified 2026-09-15 | "achieved the best performance"; values not printed |

**Starting point for a small general-purpose run.** For an 8B model extended from 8K to 128K with synthetic CPT data, the NExtLong 128K configuration is the most completely documented synthetic CPT recipe in the table: granularity 2048, top-k similar negatives, meta-chunk at the head, about 4B tokens at 128K, LR 4e-5 with cosine decay and 200 warmup iterations, RoPE base 500,000 → 200,000,000; it was run on Meta-Llama-3-8B base on 64 H100s for 15 hours. For SFT, start from short-only SFT as in UltraLong-8B (100K short examples, LR 5e-6, batch 128) and add a synthetic long share only if long evaluation after SFT regresses; the Llama 3 share of 0.11% of examples (about 5% of tokens) is the one production value, reported for 8B–405B models without a printed ablation. When long SFT data is generated, apply a closed-book filter and a citation or grounding filter (LongCite's 20% threshold was run on 9B and 8B models). None of these settings was tested with the others in one run.

## Generalization lens

**(a) What increases breadth.**
- Constructing dependencies over diverse domains instead of using the few domains that have long documents: Quest data beat existing long documents at equal tokens, 22.06 vs 21.11 LongBench at 1.4B ([[quest-query-centric-synthesis]] Table 5).
- Integration and indirection rather than retrieval only: KeyChain 72.4 vs 66.2 ([[loongrl]] Table 4); IN2 training, whose long data includes 300K integration questions, raised multi-hop LongBench tasks by up to 22.7 points (2WikiMQA 24.3 → 47.0, [[in2-film]] Table 2).
- Fact-free retrieval data for position-independent lookup: MDQA +10.5% at position 10 for GPT-3.5 Turbo with TriviaQA unchanged on Mistral-7B ([[artificial-needles-real-haystacks]] Table 2).
- Transfer beyond the synthesized task: QwenLong-L1.5 raises AIME25 82.81 → 86.46, which was not a data target, and LongMemEval 60.80 → 76.40, although simulated dialogues were a small part of its corpus ([[qwenlong-l1-5]] Table 8), and CLIPPER raises MuSR 41.2% → 45.2% for Qwen2.5-7B ([[clipper-compression-synth]] Table 2). QwenLong-L1.5 changed its RL algorithm and its data in the same comparison, so the data alone is not isolated ([[qwenlong-l1-5]] §4.1).

**(b) What causes narrowing or forgetting.**
- Long data carrying new facts: TriviaQA −2.19 to −6.33 on Mistral-7B ([[artificial-needles-real-haystacks]] Table 2).
- Instruction-following loss after long SFT: IFEval −11.5 (8B) and −16.9 (3B) ([[bootstrap-context-length]] Table 4).
- Knowledge and reasoning loss with extension to 1M: MMLU −3.13 ([[hierarchical-million-token-synth]] Table 3); GPQA −5.6 for Qwen2.5-14B-1M ([[qwen-long-context-synth]] Table 6).
- Task-format narrowing: ChatQA 2's long SFT without summarization data gave En.Sum 16.08 against 30.94 ([[chatqa-2]] Table 2); Llama-CLIPPER loses 1.3 points on ∞Bench QA ([[clipper-compression-synth]] Table 2).
- Packing-induced regression of a different skill: tool calling after 128K packing ([[nemotron-nano-2]] §3.2).

**(c) How to measure it for this stage.**
- Before and after each long stage: a short suite with knowledge QA (TriviaQA, NQ-Open), instruction following (IFEval), code (HumanEval), and math, because the losses above appear on different benchmarks in different studies.
- Long evaluation families the generator did not target, and test documents disjoint from generator documents (CLIPPER splits by book; ChatQA 2 removes NarrativeQA). [[context-synthesis-short-to-long]] draws training instructions from the training splits of its evaluation datasets, so its main table is in-distribution.
- Literal-match control: NoLiMa-style low-overlap probes ([[nolima]] Table 3).
- Position-gap metrics in addition to averages ([[in2-film]] Table 1).
- Measurement error: UltraLong evaluates its baselines itself and reports ProLong-512k-Instruct HumanEval 10.36 ([[ultralong-128k-to-4m]] Table 2); differences in prompt format and decoding between papers can move baseline scores, so cross-paper comparisons of short benchmarks need the evaluation configuration ([[ch-32c]]).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| QA generated from a chunk with high word overlap | NIAH near 100 but low-overlap probes drop | Unigram precision histogram (§1); NoLiMa-style eval |
| Keeping questions answerable without the document | Long-context gains appear on knowledge-heavy QA but vanish when the document is replaced with irrelevant text | Closed-book filter on a sample (§6) |
| Generator reads the whole long input | High error or misattribution rate in human audit | Audit 50+ examples by error type (CLIPPER Table 1) |
| Cross-document masking applied to a constructed dependency | Synthetic CPT gives no gain over concatenation | Inspect attention mask boundaries against construction boundaries (§2.4) |
| Counting the long share by examples only | "0.1%" long data turns out to be about 5% of tokens | Compute token share from average lengths (§7) |
| Distractors that are random text only | Gains on MDQA with random distractors, none with retrieved distractors | Evaluate with retrieved distractors ([[artificial-needles-real-haystacks]] Fig. 10) |
| Self-repeated or near-duplicate negatives | Lower long scores than top-k retrieval | Deduplicate negatives against the meta-document ([[nextlong]] §5.4) |
| No short-suite regression check | IFEval, TriviaQA, or HumanEval drop found only after release | Before/after short suite at every long stage (§7) |
| Evaluating on datasets used to generate training data | Scores above independent suites | Split by source document or dataset (ChatQA 2 §3.2) |
| Packing short tool-call SFT into long sequences | Tool-calling accuracy drops | Stage-wise tool eval; train tool data without concatenation ([[nemotron-nano-2]] §3.2) |

## Check your understanding

1. Llama 3 writes QA from one 8K chunk but trains with the whole document. Explain what dependency this creates and which of the five properties in §1 it leaves uncontrolled.
2. NExtLong computes the loss on hard-negative chunks as well as on the meta-document. Explain why the negatives still force long-range modeling, and what would change if self-repeated chunks were used as negatives.
3. ProLong's mask ablation and UltraLong's separator ablation both favor their own design. Explain why these two results do not contradict each other and what experiment would resolve the question.
4. Key-value retrieval data left TriviaQA unchanged while NIAH-style data lowered it by 6.33 points. Give the causal explanation the authors propose, state why it is an interpretation, and describe a measurement that would test it.
5. QwenLong-L1.5's robustness check discards an item only when pass@k falls to zero. Using the worked example in §6, explain which weak items survive and how you would change the check to catch them.
6. CLIPPER's generator reads outlines instead of books, yet its trained models read books. Explain why this improves data quality and what class of claims it cannot produce.
7. The Llama 3 long SFT share is 0.11% of examples. Explain why the token share is the relevant quantity for forgetting and compute it from Table 7.
8. SelfLong's IFEval dropped 11.5 points while GPQA rose 3.6. Propose two causes rooted in the data mixture of §3.5, and state how SmolLM3's merge result suggests a mitigation.

## Connections

- Previous: [[ch-29]] — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification (the filter and verification stages applied here to long inputs).
- Next: [[ch-29b]] — Long-Conversation and Accumulating-Context Synthesis.
- Dependency: [[ch-32b]] — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (CPT stage, RoPE, masking, the SFT-share debate in §7).
- Dependency: [[ch-32c]] — Claimed versus Effective Context Length and Long-Context Evaluation (RULER, HELMET, NoLiMa).
- Related: [[ch-28]] — Long-Context Data Synthesis and Synthetic Evaluation Task Families; [[ch-30c]] — Weight Averaging and Model Merging for Generalist Models; [[ch-31a]] and [[ch-43a]] — negatives in SFT and as gradients; [[ch-44a]] — Length in RL: Overlong Responses, Length Control, and Long-Context RL.

## Sources

- [[llama-3]] and [[excerpts/llama-3-long-context-sft]] — §3.4.2 stage gate and 800B tokens; §4.3.4 chunk QA, hierarchical summarization, missing-file code, 0.1% share; Table 7 token statistics.
- [[long-context-llama3]] — older library card for the same material; not yet re-verified, and its per-stage token counts and data ratios are not in the report. Numbers in this chapter come from the excerpt.
- [[qwen-long-context-synth]] — structural CPT tasks, 75/25 length mix, agent-written SFT responses, Table 6 short-benchmark changes.
- [[yi]] and [[excerpts/yi-long-context-recitation]] — recitation QA in CPT and SFT, 5B-token CPT, Table 6 MMLU (card not yet re-verified; numbers from the excerpt).
- [[chatqa-2]] — "<s>" separator, NarrativeQA summary insertion, InfiniteBench and Table 8 (chapter excerpt; no library card yet).
- [[in2-film]] — IN2 construction, depth balance, VAL Probing gap, LongBench and short-task changes.
- [[hierarchical-million-token-synth]] — hierarchical multi-document dialogues to 1M, token budget, MMLU loss.
- [[nemotron-nano-2]] — Phase LC appended doc-QA, 20% blend, Table 4 ablation, SFT packing regression (chapter excerpt).
- [[quest-query-centric-synthesis]] — query-centric grouping, Tables 1–3 and 5 (chapter excerpt).
- [[nextlong]] — hard-negative interleaving, k formula, Tables 1, 4, 5.
- [[ultralong-128k-to-4m]] — separators without masking, short-only SFT, Tables 1–3 (chapter excerpt).
- [[prolong]] — document-masking ablation, synthetic long SFT share ablation, long-document availability.
- [[qwenlong-l1-5]] — knowledge-graph multi-hop, SQL-grounded answers, grounding and robustness checks, Tables 1, 7, 8 (chapter excerpt).
- [[loongrl]] — KeyChain construction and Table 4 comparison.
- [[artificial-needles-real-haystacks]] — fact-free key-value data, Table 2 knowledge-benchmark comparison, relevant-distractor limit (chapter excerpt).
- [[longmit]] and [[excerpts/longmit-mimg]] — multi-hop share under Self-Instruct and MIMG (the card describes a generic dataset class; numbers from the excerpt).
- [[context-synthesis-short-to-long]] — context synthesis, distractor pilot, context-free diagnostic.
- [[clipper-compression-synth]] — compression-first claim generation, human error audit, Table 2 transfer (chapter excerpt).
- [[bootstrap-context-length]] — agent-workflow synthesis, Table 3 ablation, Table 4 IFEval drop (chapter excerpt).
- [[smollm-3]] and [[excerpts/smollm3-long-context-merge]] — RULER regression and 0.9/0.1 merge (card not yet re-verified; quotes from the excerpt).
- [[longcite]] — CoF citation pipeline, 20% filter, Tables 2, 3, 5 (chapter excerpt).
- [[nolima]] — question–needle lexical overlap and its effect.
- [[lost-in-the-middle]] — position dependence and closed-book comparison.
- [[agentinstruct]] — unanswerable reading questions as negative content (short context).
