<!-- scope: synthetic long-context benchmark generator — configurable retrieval, multi-hop tracing, aggregation and QA tasks, and the effective-context-length measure
     see-also: [[magpie]], [[persona-hub]]
-->

# RULER: What's the Real Context Size of Your Long-Context Language Models?
- **Core Insight:** Of 17 long-context models that all claim context windows of 32K tokens or more, only half stay above the fixed quality threshold at 32K, and almost all fall below it before reaching their claimed length, even though nearly all score close to perfect on the vanilla needle-in-a-haystack test (Abstract; §1; §4 Table 3).
- **Guideline:** When reporting a usable context length, report the maximum length at which the model's 13-task RULER average stays above the Llama2-7B-at-4K threshold of 85.6, not the configured window, because the two differ by one to five doublings for every model measured except Gemini-1.5-Pro (§4 Table 3).
- **Authors:** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, et al. (NVIDIA)
- **Year:** 2024 (arXiv v1 2024-04; arXiv v3 2024-08; COLM 2024)
- **URL:** https://arxiv.org/abs/2404.06654
- **Source type:** paper
- **Relevant topics:** long-context evaluation, synthetic task generation, retrieval stress tests, multi-hop tracing, aggregation, context-length scaling

## Abstract
The needle-in-a-haystack test measures only a shallow form of long-context understanding. RULER is a synthetic benchmark with configurable sequence length and task complexity. It extends vanilla NIAH to variations with diverse needle types and quantities, and adds two new categories, multi-hop tracing and aggregation, that test behaviors beyond searching the context. The paper evaluates 17 long-context language models on 13 representative RULER tasks. Despite near-perfect vanilla NIAH accuracy, almost all models degrade substantially as context length grows; although all claim 32K tokens or more, only half maintain satisfactory performance at 32K. An analysis of Yi-34B, which supports 200K, shows large room for improvement as input length and task complexity increase. RULER is open-sourced.

## Key Contributions
- A synthetic generator with independent knobs for sequence length and task complexity, rather than a fixed corpus (§1; §3).
- Two task categories not present in prior needle tests: multi-hop tracing (variable tracking) and aggregation (common and frequent word extraction) (§3.2, §3.3).
- Effective context length: the largest evaluated length whose 13-task average exceeds the Llama2-7B score at 4K, 85.6% (§4, Table 3).
- Two weighted averages, wAvg. (inc) and wAvg. (dec), with weights increasing or decreasing linearly with length, for ranking under different length distributions (§4).
- A failure-mode analysis of Yi-34B up to 256K covering needle type, distractor count, output cardinality, hop count, and aggregation load (§5).

## Key Figures/Tables to Study
- **Table 1** — RULER against existing long-context benchmarks on task diversity, parametric-knowledge reliance, and controllability.
- **Table 3** — claimed length against effective length and per-length scores for the 17 models.
- **Figure 2** — Yi-34B across the four NIAH variants; shows which knob breaks first.
- **Figure 3** — Yi-34B on variable tracking, frequent words extraction, and QA under complexity scaling.
- **Table 5** (App. B) — the exact 13 task configurations.

## Technical Details
**Task families (§3).**
- **Retrieval, four NIAH variants.** S-NIAH inserts one key-value needle. MK-NIAH inserts additional needles as hard distractors and queries one. MV-NIAH associates several values with one key and requires all of them. MQ-NIAH inserts several needles with distinct keys and requires all of them.
- **Multi-hop tracing (VT).** A variable X1 is bound to a value V, then a linear chain of bindings (X2 = X1, X3 = X2, …) is scattered through the input; the model must return every variable name bound to V. Complexity scales with hops or chains (§3.2).
- **Aggregation.** CWE samples words from discrete uniform distributions with a fixed number of common words while uncommon words grow with sequence length; FWE samples from a Zeta distribution where the k-th ranked word has frequency k^(−α)/ζ(α)N. The model returns the top-K frequent words; K equals the number of common words in CWE and is fixed at 3 in FWE, because larger K performs poorly even at short lengths (§3.3).
- **QA.** SQuAD and HotpotQA gold paragraphs are inserted among distractor paragraphs sampled from the same dataset (§3.4).

**Generation knobs (§3.1, App. B).**
- Needle key/value type: words, 7-digit numbers, or 32-digit UUIDs.
- Haystack type: repeated noise sentences ("The grass is green. The sky is blue. …") or Paul Graham essays.
- Needle template: "the special magic number for XXX is: YYY" (§3.1, footnote 2).
- Distractor count in MK-NIAH: from 4 keys up to a haystack filled entirely with distractor needles.
- VT: number of chains and hops per chain. CWE: counts and frequencies of common and uncommon words. FWE: the Zeta parameter α.

**The 13 configurations used in the paper (App. B, Table 5).**
- S-NIAH ×3: word→number with repeated-noise haystack (≈passkey retrieval); word→number with essay haystack (≈vanilla NIAH); word→UUID with essay haystack.
- MK-NIAH ×3: num keys = 4, word→number, essay haystack (three distractor needles added); num keys = full haystack, word→number (≈line retrieval); num keys = full haystack, UUID→UUID (≈KV retrieval).
- MV-NIAH: 4 values for one key. MQ-NIAH: 4 queried keys.
- VT: 1 chain, 4 hops, so 5 variable names must be returned.
- CWE: 10 common words each appearing 30 times, uncommon words 3 times each. FWE: α = 2.0, K = 3.
- QA ×2: SQuAD and HotpotQA.

**Evaluation protocol (§4).**
- 17 models: 15 open-source plus GPT-4 (gpt-4-1106-preview) and Gemini-1.5-Pro; sizes 7B to 8x22B MoE; claimed lengths 32K to 1M (§4, App. A Table 4).
- 500 generated examples per task per length, at 4K, 8K, 16K, 32K, 64K, 128K, each wrapped in the model's chat template.
- An answer prefix is appended to prevent refusals and explanations; scoring is recall-based string matching.
- Inference in BFloat16 on 8 NVIDIA A100 GPUs with vLLM and greedy decoding.
- Effective length = maximum length whose 13-task average exceeds 85.6, the Llama2-7B chat score at 4K.

**Selected rows of Table 3** (claimed / effective / 32K / 64K / 128K / wAvg. inc):
- Gemini-1.5-Pro 1M / >128K / 95.9 / 95.9 / 94.4 / 95.5 (1st).
- GPT-4 128K / 64K / 93.2 / 87.0 / 81.2 / 89.0 (2nd).
- Llama3.1-70B 128K / 64K / 94.8 / 88.4 / 66.6 / 85.5 (4th).
- Llama3.1-8B 128K / 32K / 87.4 / 84.7 / 77.0 / 85.4 (5th).
- Qwen2-72B 128K / 32K / 94.1 / 79.8 / 53.7 / 79.6 (9th).
- Yi-34B 200K / 32K / 87.5 / 83.2 / 77.3 / 84.8 (6th).
- LWM-7B 1M / <4K / 69.1 / 68.1 / 65.0 / 69.9 (12th).

**Leaderboard extension (secondary locus).** The project repository keeps an updated table with the same protocol. Rows added after the paper, read at github.com/NVIDIA/RULER README on 2026-09-18: Jamba-1.5-Large (94B/398B) claimed 256K, effective >128K, 95.1 at 128K; Qwen2.5-14B-Instruct-1M claimed 1M, effective >128K, 92.2 at 128K (the README notes these Qwen2.5-1M numbers are reported by the Qwen authors, arXiv:2501.15383); Qwen3-235B-A22B effective >128K, 90.6 at 128K. Repository numbers are not part of the COLM 2024 paper and should be cited to the README, not to the paper.

## Findings relevant to long context
- **Training length does not determine effective length.** Top open-source models include both brute-force scaling (Llama3.1 trained at 128K) and inference-time extrapolation (Qwen2 trained at 32K), while LWM and GradientAI/Llama3, both trained at 1M, rank far lower (§4). Within the LWM series, the 1M variant is worse at 256K than the 512K variant (§6).
- **Model size correlates with long-context quality.** Yi-34B, Yi-9B and Yi-6B, trained to the same length on the same data blend, rank in that order both at 4K and in relative degradation (§6).
- **Needle format matters.** Yi-34B is near-perfect on word-number needles but degrades on other types, worst on UUIDs, where above 128K it sometimes fails to return all 32 digits (§5).
- **Hard distractors.** Increasing distracting needles lowers accuracy monotonically; in the full-haystack setting Yi drops about 40 points at 256K and returns values from positions near the target (§5).
- **Non-Transformer architectures.** RWKV-v5 and Mamba-2.8B-slimpj degrade sharply by 8K and trail Llama2-7B up to 4K (§6).
- **Behaviors that appear with length.** The paper reports increased reliance on parametric knowledge and an increased tendency to copy from the context on non-retrieval tasks (§1; §5).

## Connections
- [[magpie]], [[persona-hub]] — synthetic generation used to create instruction data, against RULER's use of synthesis for controlled measurement.
- Any long-context training report — supplies the claimed-versus-effective-length distinction and the 85.6 threshold.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2404.06654 (arXiv v3, 6 Aug 2024; COLM 2024) and github.com/NVIDIA/RULER README for the post-paper leaderboard rows.
- Corrections to the previous card version:
  - Added the abstract's headline, which was missing: all 17 models claim 32K or more, only half hold up at 32K, and almost all fall below the threshold before their claimed length (Abstract; §1).
  - "evaluates 17 long-context models across 13 representative task settings" was correct but unsourced; the 17 are 15 open-source plus GPT-4 and Gemini-1.5-Pro (§4, App. A Table 4).
  - "additional analysis on longer settings such as 200K and 256K" → the main benchmark runs 4K-128K; the Yi-34B error analysis extends to 256K only (§4; §5).
  - "FWE: alpha = 2.0, with the model returning the top 3 frequent words" → K = 3 is a global FWE setting chosen because larger K fails even at short lengths, not a property of that one configuration (§3.3).
  - "MK-NIAH: 4 keys" → the configuration is num keys = 4, made by adding three distractor needles (App. B).
  - Effective-length rows were absent; Table 3 values for Gemini-1.5-Pro, GPT-4, Llama3.1-70B/8B, Qwen2-72B, Yi-34B and LWM are now listed with their loci.
  - Author list shortened to the first six plus "et al." per the card standard; organization NVIDIA added.
  - Added source type, year with arXiv version, and this Verification section, all absent before.
- Removed as unsupported by the source: the "What labs can learn for long-context data design" list, which gave training recommendations the paper does not make (RULER is an evaluation benchmark and the paper does not train on its generators); the framing that RULER is "valuable less as a leaderboard and more as a parameterized generator", which is a course judgment; "Sources Used" section replaced by the single canonical URL plus the repository locus.
- Not reported by the source: correlation between the RULER average and downstream long-context task performance (the HELMET result belongs to a separate paper and must be cited there, not here); per-task scores for closed-source models beyond the aggregate rows; token costs of running the benchmark.
