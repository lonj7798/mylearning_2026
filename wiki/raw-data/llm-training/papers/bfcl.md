<!-- scope: BFCL benchmark paper (ICML 2025): single-turn, crowd-sourced, multi-turn, and agentic function-calling evaluation; AST substring matching; FC vs prompting mode; perplexity-based contamination check
     deps: [[gorilla]]
     see-also: [[apigen]], [[apigen-mt]], [[toolace]], [[xlam]], [[hammer]], [[nexusraven]], [[toolllm]]
-->

# The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models
- **Core Insight:** Across the 71 rows of Table 1, the best overall accuracy is 66.4 (gpt-4o-2024-11-20, prompting mode) and the best memory-category score is 12.0 (o1-2024-12-17), while gpt-4o-2024-11-20 (Prompt) scores 95.5 and 94.0 on single-turn AST multiple and parallel calls (Table 1; §5.6).
- **Guideline:** When evaluating a function-calling model for general use, report multi-turn, irrelevance, crowd-sourced, and agentic categories next to single-turn AST scores, and run both native function-calling (FC) and prompting modes, because Table 1 shows category and mode differences (GPT-4-turbo-2024-04-09 irrelevance 83.8 in FC mode vs 35.6 in prompting mode) and the crowd-sourced perplexity check flagged a possible overfit for xLAM-7B (§5.5).
- **Authors:** Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, et al. (UC Berkeley)
- **Year:** 2025 (ICML 2025, PMLR 267:48371–48392; the online leaderboard predates the paper, e.g. the V3 multi-turn blog is dated 2024-09-19)
- **URL:** https://proceedings.mlr.press/v267/patil25a.html
- **Source type:** paper
- **Relevant topics:** function-calling evaluation, tool use, abstention, multi-turn agents, memory, contamination detection

## Abstract
Function calling (tool use) had no standard benchmark because judging when a call is valid is difficult and diverse real-world functions are hard to collect. BFCL evaluates serial and parallel function calls across several programming languages with an Abstract Syntax Tree (AST) evaluation method that scales to thousands of functions. The benchmark combines expert-curated and user-contributed functions and prompts, and it also tests whether models abstain and reason correctly in stateful multi-step agentic settings. Across many models, state-of-the-art LLMs do well on single-turn calls, while memory, dynamic decision-making, and long-horizon reasoning remain open problems.

## Key Contributions
- 5,551 question–function–answer pairs across Python, Java, JavaScript, REST APIs, and SQL (§1).
- AST substring matching as an execution-free proxy, validated by its correlation with execution-based scores (§4.1, §4.3, Fig. 3).
- Crowd-sourced real user queries and functions (§3.2), used also for a perplexity-based contamination check (§5.5).
- Multi-turn stateful tasks scored by state-based plus response-based checks (§3.3, §4.4), and agentic web-search, memory, and SQL categories (§3.4, §4.5).
- Analyses of FC vs prompting mode, parallel-call behavior, and multi-turn error types (§5.1, §5.3, §5.4).

## Key Figures/Tables to Study
- Fig. 1: category hierarchy with datapoint counts per category.
- Table 1: per-category scores for all evaluated models and modes.
- Fig. 3: AST vs execution scores. Figs. 4–5: multi-turn error distributions.
- Tables 2–3: perplexity and character-level NLL on single-turn vs crowd-sourced data.
- App. H: AST matching rules.

## Technical Details
- **Single-turn categories (§3.1; App. C.2):** Simple (|F| = 1, one call), Multiple (|F| > 1, one call), Parallel (|F| = 1, several calls), Parallel Multiple (|F| > 1, several calls), Irrelevance (|F| ≥ 1, no call expected), Relevance (at least one call expected). `F` is the set of candidate functions given to the model.
- **Single-turn sources (App. E.1):** AST functions from top-100-starred GitHub repositories in Python, Java, JavaScript, excluding trivial functions and functions with fewer than two parameters; executable functions are hand-written math/physics Python functions and GET-request wrappers for ExchangeRate, OMDb, and Geocoding APIs.
- **Augmentation (App. B):** parallel entries add a query with different parameter values; multiple entries add distractor functions, checked with GPT not to be alternative solutions; irrelevance entries remove parameter information from the query or remove a needed function, and the expected output is a clarification or error, with any call counted as a hallucination.
- **Crowd-sourced data (§3.2; App. E.2):** 64,517 real queries collected 2024-02-26 to 2024-04-01 through the hosted model endpoint; deduplicated with ROUGE-L and OpenAI text embeddings; public test sets such as the Nexus leaderboard's excluded; minimal human edits. §1 reports the category as 2,251 entries "curated from more than 67,000" datapoints. Queries span more than 15 languages (§2). Entries average 3 function choices (max 37) and functions average 4 parameters (max 28) (§5.2); App. I gives a maximum of 21 parameters.
- **Multi-turn data (§3.3; App. D, E.3):** categories Base, Missing Parameters, Missing Functions, Long Context. Eight API domains: Vehicle Control, Trading Bots, Travel Booking, File System, Messaging, Twitter, Ticket Booking, Math. Tasks were generated with GPT-4o-0806 using Persona Hub personas, ground-truth trajectories were human-labeled, and the set was scaled by sampling execution paths through a function-dependency graph. §1 reports 1,000 queries; the Fig. 4 error analysis uses 800 entries.
- **Agentic data (§3.4):** Web Search uses a DuckDuckGo search tool and a page-fetch tool on recent but stable facts; Memory spans five domains, stores a memory snapshot after each dialogue block, and asks evaluation queries with an empty chat history; SQL gives a JSON schema for SELECT/INSERT/UPDATE/DELETE operations.
- **AST substring matching (§4.1; App. H):** calls are parsed with Python's `ast` module; the function name must match exactly and each parameter value must be in a set of possible answers. Python accepts an int where a float is expected; Java and JavaScript require a float literal; a float for an int parameter is invalid in all languages. Lists are order-sensitive (all permutations are listed when order does not matter); strings are case-insensitive with whitespace and listed punctuation removed; dictionary key order is ignored. Parallel calls are matched without positional alignment, all-or-nothing. Java/JavaScript values are converted to Python equivalents with Tree-Sitter.
- **Execution matching (§4.2):** exact match for deterministic outputs; ground-truth and model calls executed at the same time for time-sensitive outputs; structure match (list length, key presence) for nested outputs.
- **Multi-turn scoring (§4.4):** after each turn, a state-based check compares system state with the ground truth, and a response-based check requires the minimal viable call path; an entry is correct only if both pass in all turns.
- **Agentic scoring (§4.5; App. J):** the model answers in a required dictionary format; the answer field is lowercased, stripped of punctuation, and compared by exact string match.
- **Modes (§5.1; App. A):** FC mode passes function definitions in the API's tools field; prompting mode places them in a universal system prompt that requires `[func(param=value), ...]` output.

## Findings relevant to generality and agentic training
- **Stateful tasks lag single-turn tasks (Result, single study):** gpt-4o-2024-11-20 (Prompt) scores 95.5 / 94.0 on single-turn AST multiple / parallel but 59.0 on multi-turn base and 6.0 on memory; the maximum memory score is 12.0 (Table 1; §5.6). Observed memory failures: splitting one fact across many keys, guessing keys instead of listing them, and giving up after one failed retrieval (§5.6).
- **Format sensitivity (Result, single study):** prompting-mode models average 412.93 decoding issues vs 182.5 for FC models out of 4,251 entries, but among decoded responses FC models give wrong call counts more often in the multiple category (77.5 vs 21) (§5.1). Claude cannot emit parallel calls in FC mode but can in prompting mode (§5.1); Claude-3.5-Sonnet-20241022 (FC) scores 3.5 on AST parallel and o1-2024-12-17 (FC) scores 0.0 (Table 1). The authors hypothesize that sequential calls can be more accurate for chained tasks (§5.3, Interpretation).
- **Contamination and overfitting check (§5.5, Tables 2–3):** 6 of the 7 listed open models have lower perplexity on crowd-sourced than on single-turn data (Llama-2-7B 3.47 → 2.56; char-NLL 0.344 → 0.264). xLAM-7B rises from 3.67 to 5.09 (char-NLL 0.340 → 0.427). The authors read such a rise as a sign of memorization or tuning to the static test distribution (Interpretation); the check uses model likelihood, not task accuracy.
- **Distribution shift between curated and real queries (§5.2):** real user data contains more multiple-function and fewer parallel scenarios than the expert-curated single-turn set, plus multilingual prompts and redundant information.
- **Abstention as a scored target:** Irrelevance, Missing Parameters, and Missing Functions categories score a clarification or no call as correct (App. B, App. D.2). Scores can diverge from relevance: Qwen2.5-72B-Instruct (Prompt) gets 100.0 relevance and 72.8 irrelevance (Table 1).
- **Multi-turn error types (§5.4.2, Fig. 5):** with GPT-4o-08-06 as judge (App. F), the most frequent root cause is "Failed to Understand Environment State", followed by "Failed to Understand User's Request".

## Connections
- [[gorilla]]: the same group's earlier API-calling model; its APIBench evaluation is contrasted in §1.
- [[nexusraven]]: Nexus Raven is cited as a prompt-based protocol (§2); Nexus leaderboard test queries were excluded from the crowd-sourced set (App. E.2).
- [[toolllm]]: ToolBench, which §2 describes as depending on RapidAPI responses with high variance.
- [[toolace]], [[xlam]], [[hammer]], [[granite-function-calling]]: model families evaluated in Table 1.
- [[apigen]], [[apigen-mt]]: function-calling data-generation papers; their BFCL results are in those cards, not in this paper.

## Verification
- Checked on 2026-09-14 against: https://proceedings.mlr.press/v267/patil25a.html (PMLR 267 PDF); https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html (release date only).
- Corrections to the previous card version:
  - Title "Berkeley Function-Calling Leaderboard (BFCL)" with leaderboard/blog URLs → the ICML 2025 paper, exact title and PMLR URL above.
  - Authors: "Tianjun Zhang" is not an author; "Vishnu Suresh" added; "Berkeley Sky Computing Lab / Gorilla team" → UC Berkeley.
  - "7 core categories: simple, parallel, multiple, parallel-multiple, relevance-detection, chat, Java/JS-specific" → Simple, Multiple, Parallel, Parallel Multiple, Irrelevance, Relevance (App. C.2); Java and JavaScript are languages, and no "chat" category is described.
  - AST matcher "sort kwargs, `1.0 ≡ 1`, quote styles equivalent, arguments may be absent if default" → the App. H rules listed above (int accepted for float only in Python; float for int invalid; possible-answer sets).
  - "Live data sourced from the Gorilla community eliminates synthetic-data overfitting" → queries collected through the hosted model endpoint (App. E.2); used as a contamination and overfitting check (§5.5).
  - "Multi-turn across retail/travel/airline domains" → eight domains listed in App. E.3 (retail and airline are τ-bench's two domains, §2).
  - "V1 ~2,000 cases; V2 adds ~1,500 (~3,500 total)" → 5,551 pairs in total; crowd-sourced 2,251; multi-turn 1,000 (§1).
  - "Executable evaluation compares returned value against gold" → three matching modes (§4.2).
- Removed as unsupported by the source: version dates "V1 Feb 2024, V2 Aug 2024, V4 2025"; pass^k metric for V3/V4; "~2,000 unique API signatures"; 2025 leaderboard snapshot (Claude 3.7 Sonnet, xLAM-2-70B-fc-r, Hammer 2.1, Llama-4 derivatives); "frontier models call tools on ~10% of irrelevant queries"; "V1 ceiling has saturated"; "some labs train directly on BFCL-style data"; "list-vs-tuple spurious failures"; "not a safety eval"; the old guideline "single-turn-only evals over-report capability" (replaced by the Table 1 and §5.5 evidence).
- Not reported by the source: per-version release dates, evaluation variance across seeds, tool-call template diversity effects beyond FC vs prompting mode. Template-diversity evidence from Qwen3-Coder-Next and benchmark-reliability evidence from ABC/HAL come from other sources and have no card in this library.
