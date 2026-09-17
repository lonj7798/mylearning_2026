---
chapter: ch-32c
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/michelangelo.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2409.12640
created_at: "2026-09-15"
---

# Excerpt: Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries

**Authors:** Kiran Vodrahalli, Santiago Ontañón, Nilesh Tripuraneni, Kelvin Xu, Sanil Jain, Rakesh Shivanna, et al. (Google DeepMind; Google Research)
**Version read:** arXiv:2409.12640v2 (20 Sep 2024; v1 Sep 2024), PDF text.
**Source type:** paper (from the organization that trained the Gemini models it evaluates). Status: no library card existed for this slug on 2026-09-15.

## Framework (§3)
- Latent Structure Queries (LSQ): the context contains relevant updates to a hidden structure mixed with irrelevant text; the query asks about the structure after all updates. Complexity is indexed by the number of relevant updates and is held fixed while context length grows (§3; §3.1).
- Stated design principles: extendable to arbitrary lengths; complexity indexed by the number of relevant pieces of information; context-length difficulty decoupled from task complexity; natural-language and code domains; un-leaked, regenerable instances (§1; §3).

## Tasks (§2)
- **Latent List:** a stream of Python list operations; the model outputs a view of the final list; scored with an approximate-match metric (§2.1).
- **MRCR (multi-round coreference resolution):** a long synthetic user-model conversation with several similar requests (for example two poems about penguins); the query asks to reproduce, for example, the 2nd poem about penguins, prefixed by a given random string. Score: `difflib.SequenceMatcher` ratio between output and target, in [0, 1] (§2.2).
- **IDK:** a long text with filler of random letters and a four-option question whose option (D) is "I don't know"; 70% of instances have "I don't know" as the correct answer and 30% are simple retrieval (§2.3).

## Setup (§3.2-§4)
- Subsets up to 32K, 128K, and 1M tokens; cumulative scores; when subsets are stacked, each bucket is divided by the number of times it repeats (§3.2).
- Few-shot prompts with short-context demonstrations; the random-string prefix in MRCR is used to check instruction following and to post-process output (§3.3). Latent List and MRCR were also used as pre-training evaluations, where "the few-shot nature of the prompts is critical" (§3.3.1).
- Ten models: Gemini 1.5 Flash and Pro (05-14, 08-27), GPT-4 Turbo (04-09), GPT-4o, Claude 3 Haiku/Sonnet/Opus, Claude 3.5 Sonnet; only Gemini models are plotted to 1M (§4). Claude Latent List results are excluded from plots because of refusal rates (§4).

## Findings (§4-§5; values are in figures except Fig. 9)
- "There is one initial sharp super-linear drop in performance in short-context ... after which performance often either flattens out or continues to degrade at a roughly linear rate" (§5.2). Degradation is visible by 32K, whereas RULER needs up to 128K to show it (§5.2).
- Gemini performance does not decrease from 128K to 1M on the three tasks (§4.2; Figs. 6-8).
- Best family per task: Gemini on MRCR, GPT on Latent List, Claude-3.5 Sonnet on IDK, and GPT performs worst on IDK (§5.1). No model wins all three (§5.1).
- Spearman rank correlation across the ten models at 128K (Fig. 9): MRCR vs Latent List 0.64; MRCR vs IDK 0.043; Latent List vs IDK -0.25.
- Cross-over: on MRCR, GPT and Claude beat Gemini below 8K but decay faster with length (§5.3).
