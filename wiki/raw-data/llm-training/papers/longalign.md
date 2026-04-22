<!-- scope: long-context alignment recipe — synthetic long instruction data, efficient SFT, and long-context chat evaluation
     deps: [[self-instruct]]
     see-also: [[toolformer]], [[openmathinstruct]], [[olmo-3]]
-->

# LongAlign: A Recipe for Long Context Alignment of Large Language Models
- **Core Insight:** Long-context ability is not solved by context-window extension alone; you need dedicated long instruction data, length-aware SFT, and evaluation on realistic 10k-100k-token prompts.
- **Guideline:** Treat long-context alignment as its own training problem: synthesize long tasks from real long documents, mix them with short SFT data, and train with packing or sorted batching plus sequence-level loss correction.
- **Authors:** Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, Juanzi Li
- **Year:** 2024
- **URL:** https://aclanthology.org/2024.findings-emnlp.74/
- **Relevant topics:** long-context SFT, synthetic instruction data, packing, sorted batching, loss weighting, LongBench-Chat

## Abstract
LongAlign proposes an end-to-end recipe for long-context instruction tuning. The paper argues that after position/interpolation tricks extend a model’s context window, the model still needs supervised alignment on prompts of comparable length. To do this, the authors build `LongAlign-10k`, a synthetic long-instruction dataset with sequences of 8k-64k tokens generated from diverse long documents using a Self-Instruct-style pipeline, add efficient training methods for heavy long-tailed length distributions, and introduce `LongBench-Chat`, a benchmark of realistic 10k-100k-token instruction-following queries. Across ChatGLM3-6B, Llama-2-7B, and Llama-2-13B, the recipe materially improves long-context task performance without hurting short-context chat quality.

## Key Contributions
- Defines long-context alignment as a distinct stage after context extension / continual pretraining.
- Builds `LongAlign-10k`: 10,000 synthetic long instruction-response examples spanning 8k-64k tokens, with 10% Chinese data.
- Shows that **data quantity and diversity** matter: more long data helps until about 10k examples, and diverse source/task coverage beats narrower long-data sets like LongAlpaca-12k.
- Introduces two efficiency recipes for long SFT under long-tailed lengths: **packing** and **sorted batching**.
- Identifies a subtle bias in packing loss averaging and fixes it with **sequence-level loss weighting**, improving LongBench-Chat by roughly 4-8% depending on model.
- Releases `LongBench-Chat`, a 50-example long-context instruction benchmark with realistic queries, expert references, and GPT-4 grading calibrated against humans.

## Key Figures/Tables to Study
- **Figure 3:** why long-tailed sequence lengths create GPU idle time, and how packing / sorted batching change the compute profile.
- **Figure 4:** scaling curve showing long-task gains from `LongAlign-0k -> 5k -> 10k`, with little or no short-task regression.
- **Figure 5:** diversity comparison versus `LongAlpaca-12k`; useful for seeing why long-context data quality is not just "more tokens".
- **Figure 6:** wall-clock training time on `8xA800 80G`; packing and sorted batching cut time by more than half versus naive batching.
- **Table 2:** the main evidence for loss weighting and the efficiency-quality tradeoff across ChatGLM and Llama-2.
- **Table 1:** validates GPT-4 + few-shot scoring on `LongBench-Chat` against human judgments.

## Technical Details
### Why long-context data is a distinct training problem
- The paper’s main claim is that **context extension is necessary but insufficient**. RoPE scaling + long continual pretraining lets the model accept long inputs, but not necessarily follow long user instructions over books, codebases, or papers.
- Long-context SFT examples have a very different shape from ordinary chat data:
  - Inputs are dominated by the long document.
  - Targets are short relative to the prompt.
  - Useful tasks require synthesis across distant spans, not local QA.
- This changes both **data construction** and **training dynamics**. In LongAlign-10k, the average assistant target is about **200 tokens**, while the average ratio of target tokens to full sequence length is only **0.015**. For ShareGPT short data, the average target length is **330 tokens** and the target-to-sequence ratio is **19.3%**. That mismatch is why naive batching and naive loss aggregation behave poorly.

### Long instruction data construction (`LongAlign-10k`)
- **Size:** `10,000` supervised examples.
- **Length range:** `8k-64k` tokens, measured with the ChatGLM tokenizer.
- **Language mix:** about `90%` English, `10%` Chinese.
- **Seed document sources (9):**
  - `Arxiv`
  - `Books3`
  - `C4`
  - `CLUECorpus2020`
  - `CommonCrawl`
  - `GitHub`
  - `Stack Exchange`
  - `Wikipedia`
  - `WuDaoCorpora`
- **Sampling rule:** sample documents shorter than `64k` tokens, then upsample longer examples so the final dataset is not dominated by the short end of the length range.
- **Teacher model:** `Claude 2.1`.
- **Generation pattern:** Self-Instruct-style two-stage synthesis:
  1. Feed a long document plus a task-type prompt to Claude and ask it to generate **5 candidate questions** that cover the whole text.
  2. Randomly choose one question and ask Claude for the answer.
  3. Store the resulting conversation as:
     - `user`: long document + chosen task
     - `assistant`: generated answer
- **Task prompt families (4):**
  - general questions
  - summarization / multi-part integration
  - multi-hop reasoning
  - information extraction
- **Prompt intent:** force coverage over multiple spans rather than a trivial local question. This is the core trick that makes the data useful for long-context alignment instead of just long-context retrieval.
- **Verification:** 4 PhD students manually checked 100 samples; `94/100` were judged correct, with the remaining errors split across wrong, incomplete, or irrelevant answers.

### Training data mix
- LongAlign does **not** train on long data alone.
- The SFT mixture combines:
  - all `76k` filtered `ShareGPT` examples as the short/general instruction set
  - one of several long-data suites: `LongAlign-0k`, `5k`, `10k`, `20k`, or `LongAlpaca-12k`
- The intended effect is:
  - preserve short-chat competence
  - add long-context instruction following
  - expose the model to a broad length distribution rather than a single fixed window
- Empirically, the paper reports that long-task performance improves up to about `10k` long examples and then starts to saturate, while MT-Bench and general short-task quality do not noticeably degrade.

### Base models and context extension before alignment
- Base models studied:
  - `ChatGLM3-6B`
  - `Llama-2-7B`
  - `Llama-2-13B`
- Before SFT, the authors first extend all of them to `64k` context:
  - expand the RoPE base frequency by `200x`, from `10,000` to `2,000,000`
  - continually train on pretraining data up to `64k` for `10B` tokens
- LongAlign is therefore a recipe for **post-extension alignment**, not a substitute for long-context pretraining.

### Efficient long-context SFT
- **Hardware/setup:** `8xA800 80G`, `DeepSpeed + ZeRO-3 + CPU offload`.
- **Max training length:** `64k` tokens; sequences longer than this are right-truncated.
- **Epochs:** `2`.
- **Total steps:** about `1500-2000`, depending on the configuration.

### Packing
- Long and short sequences are concatenated into packs up to the max length before dispatch to GPUs.
- The implementation uses `FlashAttention 2` with `flash_attn_varlen_func` and sequence boundary indices so each sequence attends only within itself via block-diagonal attention.
- This avoids the large waste from naive padding and also avoids the heavier 2D attention-mask implementation.
- **Average pack composition:** about `12` sequences per pack.
- **Batching setup for packing:** total batch size `8`, giving a **global batch size of 96** because each pack contains multiple sequences.

### Why naive packing loss is biased
- If each pack contributes equally to the batch loss, then packs with fewer sequences, usually the longest ones, get overweighted.
- Sequences with more target tokens also get overweighted because their token-average loss contributes more strongly inside the pack average.
- This creates an optimization bias toward long examples and toward responses with more supervised target tokens, which is not the intended objective.

### Loss weighting
- The desired objective is equal average contribution **per sequence**, not per pack.
- To implement this, the authors build a weighted 1D mask during preprocessing:
  - target-token positions for a sequence get weight `1/N`
  - non-target positions get `0`
  - `N` is the number of target tokens for that sequence
- During training, if the current batch has `M` sequences packed into `K` packs, token losses are scaled by `K / (M * N)`.
- This makes the packed loss algebraically match the true equal-per-sequence objective.
- Reported effect:
  - `ChatGLM3-6B-64k`: LongBench-Chat `5.76 -> 6.21`
  - `Llama-2-7B-64k`: `5.89 -> 6.10`
- The gain on LongBench is smaller, but the long instruction-following gain is material; this is the main training trick to keep from forgetting the actual objective under aggressive packing.

### Sorted batching
- Sort the dataset by length and sample random contiguous groups so each batch contains sequences of similar size.
- This reduces intra-batch idle time without sequence concatenation.
- Tradeoff:
  - simpler objective than packing
  - but batches become length-homogeneous, which introduces distributional bias across steps
- In practice, the paper finds sorted batching is often as fast as packing and can even be the best option for Llama-2, likely because large gradient accumulation softens the batch-order bias.

### Efficiency results
- Wall-clock training time on `8xA800 80G`:
  - `ChatGLM3-6B-64k`: naive `45.4h`, packing `20.5h`, sorted batching `19.1h`
  - `Llama-2-7B-64k`: naive `67.2h`, packing `23.4h`, sorted batching `23.3h`
  - `Llama-2-13B-64k`: naive `117.2h`, packing `41.2h`, sorted batching `44.5h`
- This is the paper’s practical message: long-context SFT is not only a data problem but a throughput problem, and batching strategy changes whether the recipe is usable.

### LongBench-Chat
- **Purpose:** evaluate realistic long-context instruction following, not just retrieval.
- **Size:** `50` examples.
- **Input length:** `10k-100k` tokens.
- **Composition:**
  - `30` authored to mimic real user queries (`20` English, `10` Chinese)
  - `20` adapted from `LooGLE` long-dependency QA and re-annotated
- **Task categories (roughly one quarter each):**
  - Information Extraction
  - Multi-segment Integration
  - Multi-segment Reasoning
  - Full-text Comprehension
- **Ground truth:** expert-written answers, each verified by at least two experts.
- **Evaluation:** GPT-4 scores model outputs from `1-10` using the reference answer plus few-shot grading examples.
- **Metric validation:** GPT-4 with few-shot examples correlates substantially better with humans than F1 or ROUGE-L.

## Evaluation Findings
- Long-context SFT matters: `LongAlign-0k` underperforms badly on long tasks, showing that context extension alone is insufficient.
- Data scaling matters: moving from `0k -> 5k -> 10k` long examples improves `LongBench-Chat`, `LongBench`, and NIAH; gains saturate near `10k`.
- Data diversity matters: `LongAlign-10k` beats `LongAlpaca-12k`, especially on multi-segment integration and broader instruction-following categories.
- Efficiency tricks mostly do not cost quality:
  - packing and sorted batching are both much faster than naive batching
  - with proper loss weighting, packing recovers or beats the naive baseline
- Model scaling still helps: `Llama-2-13B-64k` outperforms the 7B version, indicating the recipe scales with model size.

## Practical Takeaways
- Long-context SFT data should be generated from **real long source materials**, not just by concatenating short chat samples.
- The instruction prompt should force **cross-span coverage**, otherwise the model learns long retrieval but not long instruction following.
- You should mix long data with ordinary chat data instead of replacing the short set.
- If using packing, correct the objective with **sequence-aware loss weighting**; otherwise training is biased toward the longest / densest examples.
- If engineering simplicity matters more than exact objective control, **sorted batching** is a strong baseline and may be the best option on some models.
- Evaluation needs realistic long prompts with open-ended outputs; short-answer retrieval benchmarks are too narrow.

## Risks + Limitations
- The long data is still synthetic, with `Claude 2.1` as the teacher; the data quality ceiling is bounded by teacher behavior.
- LongBench-Chat is only `50` examples, so it is high-signal but not broad enough to be the only benchmark.
- The recipe mostly covers long QA, summarization, and reasoning, not longer-horizon chat/agent settings like life-long dialogue or long-history tool use.
- The released experiments stop at relatively modest open models and mostly `64k`; the recipe’s behavior at frontier scale is suggestive rather than fully settled here.

## Connections
- [[self-instruct]] is the direct data-construction ancestor: LongAlign adapts it from short instruction synthesis to long-document-conditioned instruction synthesis.
- LongBench is the underlying long-context benchmark lineage; LongBench-Chat is the instruction-following extension for more realistic long prompts.
- [[toolformer]] is a useful contrast in synthetic data design: Toolformer teaches API-use insertion, while LongAlign teaches long-range document-conditioned instruction following.
- [[openmathinstruct]] is another example where synthetic supervision must match deployment structure; in both cases, the data format is the real algorithm.
- [[olmo-3]] and later frontier reports generalize this lesson: long-context capability usually requires a dedicated stage, dedicated data, and dedicated eval, not just a bigger context window.

## Source URLs
- https://aclanthology.org/2024.findings-emnlp.74/
- https://aclanthology.org/2024.findings-emnlp.74.pdf
- https://github.com/THUDM/LongAlign
