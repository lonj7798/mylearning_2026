<!-- scope: Fiction.live's continuously updated long-context comprehension leaderboard (Feb 2025 - Apr 2026): private story-quiz questions asked over cut-down story versions from a "0"-token minimal version up to 192k tokens; methodology text, changelog, and the April 2026 and September 2025 result tables
     deps: [[needle-in-haystack-data]]
     see-also: [[chroma-context-rot]], [[context-length-alone-hurts]], [[epoch-ai-context-windows]], [[ruler]]
-->

# Introducing Fiction.LiveBench: The First Real-World Long Context Benchmark for Writers
- **Core Insight:** Fiction.LiveBench asks the same story-comprehension questions over a minimal "0"-token version and progressively longer versions of each story; in the September 2025 table, gpt-4.1 scores 100.0 at 0 tokens and 62.5 at 120k, and claude-sonnet-4 scores 100.0 at 0 and 36.4 at 120k versus 81.3 for claude-sonnet-4:thinking (Methodology; older-results table).
- **Guideline:** When long-context comprehension is measured, compare each model's score on the minimal-context version of a question with its score on longer versions of the same question, because this leaderboard separates failures that exist at 0 tokens (for example llama-4-scout:free at 62.5) from failures that appear only as length grows; treat single cells as noisy, because the page reports no run counts, confidence intervals, or sampling settings beyond reasoning-effort notes.
- **Authors:** Fiction.live (story posted by the Fiction.live account "kas")
- **Year:** 2025 (story created 2025-02-19; changelog runs from 2/21/2025 to 04/04/2026; not peer reviewed)
- **URL:** https://fiction.live/stories/Fiction-liveBench-Feb-21-2025/oQdzQvKHw8JyXbN87 (current story title "Fiction.liveBench April 04 2026")
- **Source type:** practitioner evidence (independent leaderboard with numbers; dataset private)
- **Relevant topics:** long-context evaluation, effective context length, narrative comprehension, reasoning mode vs non-reasoning mode, inference-provider effects

## Summary
Fiction.live builds AI tools that write summaries, timelines, and character bibles for authors' stories. The post states that these tools need the model to follow plots and character motivations across long stories, and that models fail at this over long contexts even when they can do it on short ones. The benchmark tests this directly. Stories are cut down to versions of increasing length; the shortest ("0"-token) version holds only the information relevant to the question. The post argues that LongBench and RULER test search rather than comprehension, and that its questions target subtext that cannot be found by searching for the relevant passage. The questions come from Fiction.live users and stay private. Results are published as table images, and a changelog records added models and method changes.

## Key Contributions
- A length ladder over the same question: 0, 400, 1k, 2k, 4k, 8k, 16k, 32k, 60k, 120k, and 192k tokens (results tables).
- A "0"-token column that measures comprehension on a version containing only relevant information (Methodology).
- Questions designed around tracking relationships that change over time, predictions from established hints, and secrets known to readers but not to characters (introduction).
- A public changelog of models added from February 2025 to April 2026, including reruns after serving fixes (Changelog).
- Two public example prompts of one question at about 1k and 8k tokens (linked GitHub gists).

## Key Figures/Tables to Study
- Current results image, titled "Fiction.LiveBench for Long Context Deep Comprehension" and linked to the April-04-2026 story URL: 19 model rows (Results).
- "Older results" image linked to the Sept-29-2025 story URL: 45 model rows split by a "---" row whose meaning the page does not state (Results).
- Example prompts: "Here is an 8k example" and "Here is the 1k variant" of the same question (Methodology).

## Technical Details
**Method (Methodology section)**
- Tests are built from "a dozen very long complex stories and many verified quizzes"; each test starts from a cut-down version with only relevant information and uses less-cut versions for longer tests.
- The example question is considered hard and "would be failed by most models"; Grok 3 fails the 8k variant and passes the 1k variant.
- Example prompt format: "I'm conducting a LLM comprehension test. I'm going to give you a story, and at the end you're going to answer a question about the story", followed by the story and a question such as "Finish the sentence, what names would Jerome list? Give me a list of names only." (gists).
- Settings stated: o1 and o3-mini at default (medium) effort; Claude Sonnet 3.7-thinking with 8k thinking tokens; o3 and o4-mini at default (medium) (Methodology; Changelog 4/17/2025).
- No judge, number of runs, temperature, or confidence interval is described on the page.

**Changelog items that change what a score means**
- 2/21/2025: "To reflect common usage we increased the number of easy questions in our benchmark set."
- 4/10/2025: Llama 4 results updated "after inference provider updated with vllm fixes".
- 4/29/2025: Qwen3 added "up to 16k for now"; 5/22/2025: "Expanded certain models up to 192k length."
- Row labels name serving providers and precision for some models, for example "[chutes/fp8]", "[minimax/fp8]", ":free" (results tables).

**April 04 2026 table (current results image; scores in %)**
- gpt-5.2: 100.0 at 0, 100.0 at 120k, 96.9 at 192k. gemini-3-flash-preview: 100.0 at 0, 100.0 at 120k and 192k. gemini-3-pro-preview: 100.0 at 0, 96.9 at 120k and 192k.
- grok-4: 100.0 at 0, 96.9 at 120k, 84.4 at 192k. kimi-k2.5 [reasoning: high]: 100.0 at 0, 78.1 at 120k, 87.5 at 192k. gpt-5.2-pro: 100.0 at 0, 75.0 at 120k, 78.1 at 192k.
- claude-opus-4-6 and claude-opus-4-5: 87.5 at 0, 93.8 at 120k, 0.0 at 192k; claude-sonnet-4-5: 100.0 at 0, 75.0 at 120k, 0.0 at 192k. The page does not explain the 0.0 cells.
- glm-4.7 [reasoning: high]: 100.0 at 0, 80.6 at 60k, 15.6 at 120k. minimax-m2.5 [reasoning: high]: 100.0 at 0, 40.6 at 120k. nemotron-3-nano-30b-a3b:free: 100.0 at 0, 69.4 at 400, 37.5 at 192k.

**September 29 2025 table (older-results image; scores in %)**
- o3: 100.0 at 0, 100.0 at 120k, 58.1 at 192k. gpt-5: 100.0 at 0, 96.9 at 120k, 87.5 at 192k. gemini-2.5-pro-preview-06-05: 100.0 at 0, 87.5 at 120k, 90.6 at 192k.
- gpt-4.1: 100.0 at 0, 91.7 at 400, 62.5 at 120k, 56.3 at 192k.
- llama-4-maverick:free: 100.0 at 0, 56.0 at 400, 36.4 at 120k. llama-4-scout:free: 62.5 at 0, 52.0 at 400, 16.0 at 32k, 27.3 at 120k.
- gpt-oss-120b-high [chutes]: 100.0 at 0, 38.9 at 60k; gpt-oss-120b [chutes]: 100.0 at 0, 33.3 at 60k.
- Scores are not always monotonic in length; for example grok-4 scores 88.9 at 2k and 100.0 at 8k (both tables).

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
### Long context
- The post's stated claim is that "most LLMs CAN handle these tasks, but not over long context" (introduction). In the September 2025 table, 0-token scores of 100.0 alongside 120k scores of 36.4 (claude-sonnet-4, llama-4-maverick:free) are examples of that pattern.
- In these same-model pairs from the September 2025 table, the reasoning variant scores higher at long lengths: claude-sonnet-4 36.4 vs claude-sonnet-4:thinking 81.3 at 120k; claude-opus-4 37.5 vs claude-opus-4:thinking 62.5 at 120k; deepseek-v3.2-exp 47.2 vs deepseek-v3.2-exp [reasoning: high] 71.4 at 60k.
- Some models fail at 0 tokens, so their long-context scores do not isolate a length effect: llama-4-scout:free 62.5, gemini-2.5-flash-lite-preview-06-17 50.0, deepseek-v3.2-exp 50.0 (September 2025 table).
- The Llama 4 rows were updated after the inference provider applied vLLM fixes (Changelog 4/10/2025), so a row reflects the serving stack used when it was run (Interpretation).

## Connections
- [[needle-in-haystack-data]] — the page contrasts its comprehension questions with NIAH-style tests that models pass across long contexts.
- [[ruler]] and [[longbench]] — benchmarks the post describes as testing search rather than comprehension.
- [[chroma-context-rot]] — controlled report showing length-driven drops on fixed simple tasks.
- [[context-length-alone-hurts]] — controlled paper showing accuracy drops with length even under perfect retrieval.
- [[epoch-ai-context-windows]] — Epoch AI analysis that uses Fiction.liveBench and MRCR to measure effective input length.
- [[openai-mrcr-graphwalks]] — MRCR, the other benchmark in that Epoch AI analysis.
- [[nolima]], [[michelangelo]], [[longbench-v2]] — other long-context evaluations that go beyond lexical retrieval.
- [[interconnects-llama-4-long-context]] — commentary on Llama 4's long-context claims.

## Verification
- Created on 2026-09-14 from https://fiction.live/stories/Fiction-liveBench-Feb-21-2025/oQdzQvKHw8JyXbN87 (JS-rendered; text read through the site API endpoints api/node/oQdzQvKHw8JyXbN87 and api/anonkun/chapters/oQdzQvKHw8JyXbN87; retrieved 2026-09-14; latest changelog entry 04/04/2026). Result tables read from the two linked images (cdn6.fiction.live …a7be0188….png, April-04-2026; …5aa21fca….png, Sept-29-2025). Example prompts read from gist.github.com/kasfictionlive/74696cf4f64950a6f56eb00a035f3003 (8k) and …/436e6c9ac715f4fbabf9f660fa2ddcd4 (1k).
- Audit claims not found in the source: (1) "36 questions about 30 fiction stories": the page says "a dozen very long complex stories and many verified quizzes"; the 36/30 figure appears on Epoch AI's benchmark hub (epoch.ai/benchmarks/fictionlivebench), not on the primary page. (2) "Questions need theory of mind, chronology and inference from implicit information": this is Epoch AI's wording; the page lists tracking changes over time, predictions from established hints, and reader-known vs character-known secrets. (3) April 2025 scores from a community mirror (gemini-2.5-pro-exp-03-25 90.6 at 120K, llama-4-maverick 28.1 and llama-4-scout 15.6 at 120K, Maverick 61.1 at 400): not in either table the page now links; the September 2025 table shows Maverick 56.0 at 400 and 36.4 at 120k, Scout 27.3 at 120k, and no gemini-2.5-pro-exp-03-25 row. gpt-4.1 62.5 at 120k, Scout 62.5 at 0, and Maverick 100.0 at 0 do match the September 2025 table. (4) "Epoch AI uses it as one of its two long-context effectiveness benchmarks" is not on the primary page (it is stated in Epoch AI's June 25, 2025 data insight).
- Not reported by the source: number of questions and stories per length, answer grading method, number of runs, temperature, confidence intervals, the reason for 0.0 scores at 192k.
