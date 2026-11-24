<!-- chapter: ch-15
     track: rl-pipeline
     kind: content
     title: Web-Search and Deep-Research RL Pipeline
     deps: [ch-12]
     sources: [[search-r1]], [[zerosearch]], [[asearcher]], [[websailor-tongyi-deepresearch]], [[deepresearcher]], [[gemini-2.5-deep-research]], [[mai-rl-infrastructure]], [[slime-search-r1-pipeline]], [[verl-search-tool-pipeline]], [[areal-agent-pipelines]], [[tito-token-in-token-out]], [[recipe-crosswalk]], [[algorithm-crosswalk]], [[deepseek-v3-2]], [[deepseek-v3.1]], [[glm-4-5]], [[mai-thinking-1]]
     figures: figures/search-pipeline.html
-->

# Chapter 15 — Web-Search and Deep-Research RL Pipeline

> **Core insight.** A search-agent RL rollout is a sequence of model turns separated by tool outputs the model did not generate. Four parts of the pipeline differ from single-turn RLVR, and each has a measured or code-visible consequence. (1) The tool server is part of the environment. Its cost, latency and rate limits set throughput: a live-web setup used a 50-node CPU cluster and a 7-day query cache ([[deepresearcher]] §3.2), and a simulated search LLM cost $70.8 of GPU time against $586.7 of API fees for about 64,000 requests ([[zerosearch]] App. D). (2) Retrieved tokens must be removed from the loss. Keeping them lowered Search-R1's 7-dataset EM average from 0.431 to 0.343 ([[search-r1]] Table 4). (3) The outcome grader is a string or judge comparison of a short answer, and its choice is a hacking decision: exact match led ZeroSearch's policy to write long answers, so the authors switched to F1 ([[zerosearch]] §3.5). (4) Trajectories are long and their lengths vary widely, so synchronous batching wastes time and asynchronous rollout introduces staleness. ASearcher reports trajectories of more than 100 turns and over 400k output tokens ([[asearcher]] abstract; §4.3 gives "over 150k"), and Rocket's web-research rollouts have a median of 51 minutes and 89 turns ([[mai-rl-infrastructure]] Fig. 2).
>
> **Guideline.** When building a search RL pipeline, mask every token the tool returned, including template boundary tokens, because Search-R1 (0.431 vs 0.343) and ZeroSearch (36.07 vs 34.53 EM) both measured a drop without the mask. Build the next-turn context from token ids rather than re-tokenized text ([[tito-token-in-token-out]]). Filter questions the base model answers without search, because a question answered without the tool trains no tool use; DeepResearcher drops any question that 1 of 10 base samples answers ([[deepresearcher]] §4.3). Start with F1 or an LLM judge rather than exact match if answers are free-form, because EM was hacked by long answers ([[zerosearch]] §3.5). Before running on the live web, decide between cached, simulated and live search with the cost numbers in hand: a fine-tuned 14B simulator scored 34.47 EM against 32.81 for training on real Google results ([[zerosearch]] Table 4). Use asynchronous rollout with an admission-time staleness bound once turn limits exceed about 10, because a synchronous framework capped WebSailor at 50 RL steps ([[websailor-tongyi-deepresearch]] §5.4) while ASearcher trained 32–128-turn agents on AReaL ([[asearcher]] §3.3.2).

---

## Why this chapter matters

Search is the agentic domain with the widest spread between the cheapest and the most expensive version of the same pipeline. Search-R1 trains against a local Wikipedia index with at most 4 actions on one 8×H100 node ([[search-r1]] App. B.2). Rocket's production web-research rollouts have a median of 89 turns and 51 minutes ([[mai-rl-infrastructure]] Fig. 2). Tongyi DeepResearch evaluates with up to 128 tool calls in a 128K context ([[websailor-tongyi-deepresearch]] Tongyi §4.1). The algorithm is a GRPO or PPO variant in every case (ch-18). The rest of the pipeline differs: question source, tool server, mask, grader, context handling, and rollout scheduler.

The chapter has a second purpose. It is the first place in the course where three open frameworks implement the same pipeline, so it compares slime, verl and AReaL code at a fixed commit for the one step that must be right in all of them: which tokens receive gradient.

MAI's Thinking 1 report describes no web-search RL climb; its agentic climb covers SWE and general tool use only ([[mai-thinking-1]] Summary). The Rocket post includes web research in its production mix (Fig. 2) and lists search among the environments rollout workers call (Fig. 1, step 5) ([[mai-rl-infrastructure]]). This chapter's Rocket comparisons therefore use the infrastructure post, not a published search recipe.

---

## §1 The pipeline at a glance

| Stage | Choices in the sources | Where |
|---|---|---|
| Question source | NQ + HotpotQA; filtered multi-hop mixes; synthesized hard questions | §2 |
| Difficulty / contamination filter | drop questions answerable without search; drop all-correct and all-wrong | §2 |
| Tool server | local retriever; live web with cache; simulated search LLM | §3, §4 |
| Rollout loop | generate → parse tag or tool call → call tool → append observation | §5 |
| Loss mask | tool output tokens excluded | §5 |
| Grader | EM, F1, LLM judge, plus format terms | §6 |
| Context management | turn caps, page summarization, history truncation | §7 |
| Scheduler | synchronous, fully async with staleness bound, step-level async on-policy | §8 |

[figures/search-pipeline.html](figures/search-pipeline.html) steps through one multi-turn rollout and shows, for each segment, whether it gets gradient under each framework's mask. Open it with §5.

---

## §2 Question sources and filters

**Sources.**

| Source | Questions | Location |
|---|---|---|
| Search-R1 | union of NQ and HotpotQA training sets | [[search-r1]] §4.3 |
| DeepResearcher | NQ, TriviaQA, HotpotQA, 2Wiki, filtered to 80,000 at NQ:TQ:HotpotQA:2Wiki = 1:1:3:3 (75% multi-hop) | [[deepresearcher]] §4.1, §4.3 |
| ASearcher | HotpotQA + 2WikiMultiHopQA filtered 304k → 16k; 25,624 synthetic QAs from 14,107 seeds; two 35k training sets | [[asearcher]] §3.2.1–3.2.2, §4.1 |
| WebSailor | SailorFog-QA: subgraphs of random-walk knowledge graphs, with dates, names and quantities obfuscated | [[websailor-tongyi-deepresearch]] WebSailor §3.1 |
| GLM-4.5 | multi-hop reasoning over knowledge graphs, human-in-the-loop extraction, selective obfuscation of web content | [[glm-4-5]] §3.3.1 |
| DeepSeek-V3.2 | 50,275 search-agent tasks; a QA pair is kept only if its ground truth is verified correct and all candidate answers are verifiably incorrect | [[deepseek-v3.1]] §3.2.3, Table 1 |

**Synthesis for hardness (ASearcher).** Starting from a seed QA, an agent iterates Injection (add a fact about an entity) and Fuzzing (blur a detail). Each step is checked three ways: an LLM checks basic quality and grounding against tracked supporting facts; QwQ-32B attempts the question without tools to measure difficulty; and mismatched answers are checked for uniqueness. Seeds average 6.3 injections and 3.2 fuzzes, and kept questions average 4.27 injections and 2.10 fuzzes ([[asearcher]] §3.2.2).

**How hard the synthesized questions are.** Before filtering, o4-mini scores 47.3 and DeepSeek-R1 38.9 pass@1 on SailorFog-QA, against 90.2 and 84.4 on WebDancer-QA ([[websailor-tongyi-deepresearch]] Table 2).

**Filters.**
- Contamination: drop any question where 1 of 10 base-model samples contains the answer without search (DeepResearcher, pass@10). Also drop time-sensitive, subjective and harmful questions with a DeepSeek-R1 prompt ([[deepresearcher]] §4.3).
- Hard but solvable and tool-requiring: from 16 responses of an RL-trained model, drop a question if none is correct, if accuracy is at least 50%, or if it is solved with at most 1 search turn ([[asearcher]] §3.2.1).
- All-correct removal: WebSailor drops items with 8/8 correct before training ([[websailor-tongyi-deepresearch]] WebSailor §4.2). Tongyi drops always-fail and always-succeed items under the SFT model and refreshes the active set from a backup pool sampled by intermediate checkpoints (Tongyi §3.4.3).

The contamination filter is specific to search. A question the model answers from parametric memory gives reward without any tool call, so it rewards not searching.

---

## §3 The tool server: local, live, cost and limits

### §3.1 Local retrieval (Search-R1, slime)

Search-R1 uses a 2018 Wikipedia dump with an E5 retriever, returns the top 3 passages, and limits retrieved content to 500 tokens ([[search-r1]] §4.3, App. B.2). slime's reproduction serves an e5-base-v2 FAISS flat index over wiki-18 from a separate FastAPI server at `http://127.0.0.1:8000/retrieve` ([[slime-search-r1-pipeline]]; `README.md` L253–265). The rollout bounds retrieval load with one process-wide semaphore:

```python
# slime@79aa41a examples/search-r1/generate_with_search.py L14-L18, L41
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
...
SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])
```

`max_turns` is 2 in this example (two SGLang calls, so at most one search). Search-R1's paper budget is B = 4 actions (App. B.2). The example therefore trains a shorter horizon than the paper it reproduces unless the value is changed.

### §3.2 Live web (DeepResearcher, Tongyi, AReaL)

DeepResearcher trains against a live search API (top-10 results with title, URL and snippet) plus a browsing agent that reads pages segment by segment ([[deepresearcher]] §3.1). The tool layer is its own service: a 50-node CPU server cluster handles tool requests during rollout, failed API or crawl calls are retried, and identical queries within about 7 days are served from cache. The paper names concurrency (for example 4,096 simultaneous searches per GRPO step: 256 prompts × 16 rollouts) and provider rate limits (for example 200 requests/second) as infrastructure constraints (§3.2, §5.1.1).

Tongyi DeepResearch wraps real tools in a sandbox with QPS limits, caching, timeout-retry and failover to a backup search API ([[websailor-tongyi-deepresearch]] Tongyi §3.4.3).

AReaL's Tongyi-style example uses Serper for `search` and Jina reader plus an LLM summary for `visit` ([[areal-agent-pipelines]]; `tool_search.py` L46). The summary is produced by the same frozen engine that serves as the judge: `Visit(summary_client=self.judge_client)` (`react_agent.py` L80). That judge engine is a separate SGLang server whose staleness bound is set to `int(1e12)` because its weights never change (`train.py` L134). The policy therefore does not learn page summarization in this example. In the ASearcher paper the agent summarizes pages itself and that summarization is trained by RL ([[asearcher]] §3.1).

### §3.3 What verl has at 8357f1e

The documented verl search pipeline is only partly present at this commit ([[verl-search-tool-pipeline]] Verification):

- **Removed or absent:** `verl/tools/search_tool.py`, `search_r1_like_utils.py`, all of `examples/sglang_multiturn/` (including the Search-R1 launch script, `local_dense_retriever/`, and `config/tool_config/search_tool_config.yaml`), and `recipe/`. `docs/sglang_multiturn/search_tool_example.rst` L149–243 still references them. The documented YAML used `class_name: verl.tools.search_tool.SearchTool` with `num_workers: 120`, `rate_limit: 120`, `timeout: 30` (doc L233–240).
- **Still present:** the data preprocessor (`examples/data_preprocess/preprocess_search_r1_dataset.py`), the tool base classes (`verl/tools/base_tool.py`, `function_tool.py`, `tool_registry.py`), the generic `ToolAgentLoop`, the Hermes and other tool-call parsers, the token-level response-mask builder (`continuous_token.py`), the `search_r1_like_qa_em` reward preset, and a test stub `tests/tools/_stub_search_tools.py` that returns `"hits-for:<query>"` strings.

To run search RL on verl at this commit, write a `BaseTool` subclass that calls your retriever, declare it in a tool YAML, and point `actor_rollout_ref.rollout.multi_turn.tool_config_path` at it ([[verl-search-tool-pipeline]] Guideline). Two defaults matter for search: `max_tool_response_length: 256` truncates each tool response to 256 characters (the middle is cut by default), and `max_parallel_calls: 1` executes one call per turn (`rollout.yaml` L220, L223; truncation by `len(tool_response_text)` in `tool_agent_loop.py` L500–507). 256 characters is shorter than one Search-R1 passage set (up to 500 tokens), so raise it for search.

### §3.4 Cost

ZeroSearch's ~64,000 search requests for a 7B run cost $586.7 in search API fees, against $70.8 of GPU time for the simulator ([[zerosearch]] App. D, Table 8). DeepResearcher's 4,096 simultaneous searches per step, and Rocket's 89-turn median, show why tool calls dominate wall-clock time in live-web RL.

---

## §4 Simulated and cached search

**ZeroSearch.** An LLM is fine-tuned on query–document pairs extracted from real search trajectories, labelled useful or noisy by an LLM judge. A prompt keyword then selects useful or noisy output ([[zerosearch]] §3.3). During RL a curriculum raises the probability of noisy documents: p_i = p_s + ((b^(i/m) − 1)/(b − 1))·(p_e − p_s) with b = 4. For Qwen-2.5-7B, p_s = 0 and p_e = 0.75 (§3.4, App. F).

| Simulator (Qwen-2.5-3B-Base policy, evaluated on real Google) | Avg EM |
|---|---|
| fine-tuned 14B | 34.47 |
| fine-tuned 7B | 33.53 |
| real Google Search in training | 32.81 |
| prompted (not fine-tuned) 14B | 31.50 |

Source: [[zerosearch]] Table 4. Curriculum against a fixed 0.5 noise rate: 34.47 vs 32.59 (Table 6).

**A design detail to check before copying.** The simulator prompt includes the user's question and its ground-truth answer, which the authors add "to extend the simulator's knowledge coverage" ([[zerosearch]] §3.3, Table 2). The simulator can therefore put the answer into "retrieved" documents more often than a real search engine would. Evaluation uses real Google (SerpAPI) for all methods (§4.1), so test scores are not inflated by this, but training reward is not comparable to a real-search run.

**Tongyi's simulated Wiki environment.** Tongyi keeps a simulated 2024-Wikipedia RAG environment for fast algorithm iteration. Its reward curve "closely matches" the real-environment curve (Fig. 10b), and no numbers are printed ([[websailor-tongyi-deepresearch]] Tongyi §4.4).

**Caching.** DeepResearcher serves a repeated query within about 7 days from cache ([[deepresearcher]] §3.2). A cache makes repeated rollouts of the same prompt see the same results, which lowers reward variance within a group but also hides result drift across time.

**When to choose which.** Local retrieval suits short-answer multi-hop QA whose answers are in the corpus. Live web is needed when answers are outside the corpus: on Bamboogle, whose questions need knowledge beyond Wikipedia, DeepResearcher outperforms local-RAG-trained agents ([[deepresearcher]] §5.2.4). A simulator suits algorithm iteration and cost-bound runs, provided the evaluation uses real search.

---

## §5 Masking retrieved tokens from the loss

### §5.1 The measured effect

| Source | With mask | Without mask | Location |
|---|---|---|---|
| Search-R1, Qwen2.5-7B-base, PPO, 7-dataset EM average | 0.431 | 0.343 | [[search-r1]] Table 4 |
| ZeroSearch, LLaMA-3.2-3B-Base, avg EM | 36.07 | 34.53 | [[zerosearch]] App. C, Table 7 |

ZeroSearch states that "uniform optimization over both token types destabilizes training" (§3.6). Every search source carded here masks tool output ([[recipe-crosswalk]] Search synthesis, step 5).

**Why it matters (mechanism).** Retrieved tokens were not sampled from π_θ. Including them in a policy-gradient loss raises the log-probability of text the model did not choose, scaled by the trajectory's advantage, so a correct trajectory teaches the model to reproduce Wikipedia passages. Search-R1 also excludes retrieved tokens from the KL term (§3.1).

### §5.2 Three implementations at fixed commits

**slime.** The rollout appends model tokens with mask 1 and observation tokens with mask 0:

```python
# slime@79aa41a examples/search-r1/generate_with_search.py L217-L219, L235-L244
response += cur_response
response_token_ids += cur_response_token_ids
loss_mask += [1] * len(cur_response_token_ids)
...
next_obs, done = await execute_predictions(cur_response)
if done:
    break
assert next_obs != "", "Next observation should not be empty."
obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
response += next_obs
response_token_ids += obs_tokens_ids
loss_mask += [0] * len(obs_tokens_ids)
sample.append_response_tokens(args, tokens=obs_tokens_ids, trainable=False)
```

`Sample.append_response_tokens` enforces the contract: trainable tokens require rollout log-probs, and non-trainable tokens must not pass them and get log-prob 0.0 (`slime/utils/types.py` L277–305). The model's own token ids come from SGLang's `output_token_logprobs` (L198–210), so they match the sampled ids.

One observation from reading this code: the next turn is sent to SGLang as text, `"text": prompt_text + response` (L181). The engine re-tokenizes the whole string, so the context the model conditions on in turn 2 is the re-tokenization of turn 1's decoded text, while the training sequence uses turn 1's sampled ids. When the two tokenizations differ at a boundary, the trained sequence differs from what the model saw. [[tito-token-in-token-out]] describes this failure class and the token-delta methods that avoid it. Whether it occurs for this tokenizer and tag set was not tested here.

The file also records a bug fix for a related problem. Without a stop at `</search>` and `</answer>`, SGLang kept generating after the tag. With log-probs enabled, the trailing text was not trimmed, was trained with `loss_mask=1`, and broke the format check. The fix adds both tags as stop strings (L164–177).

**verl.** The Continuous Token builder appends to the mask with the same edits applied to the token stream:

```python
# verl@8357f1e verl/utils/tokenizer/continuous_token.py L392-L400, L412-L413
# Boundary tokens are added by CT itself, so they get mask/logprob 0.
inserted_token_count = len(merge_result.inserted_token_ids)
aligned_mask += [0] * inserted_token_count
if aligned_logprobs is not None:
    aligned_logprobs += [0.0] * inserted_token_count

# Model-generated assistant tokens get mask 1 and their logprobs; context tokens get mask/logprob 0.
if merge_result.kind == "assistant":
    aligned_mask += [1] * merge_result.appended_token_count
...
elif merge_result.kind == "context":
    aligned_mask += [0] * merge_result.appended_token_count
```

verl calls the inference server with token ids, not text ([[verl-search-tool-pipeline]] Summary), and its builder masks template boundary tokens that the engine did not sample (the `inserted_token_ids`), not only the tool text.

**AReaL.** In `concat` export, a child turn inherits the parent's mask and pads the new prompt span (tool output and any user text) with mask 0 and policy version −1:

```python
# AReaL@bcef492 areal/experimental/openai/types.py L214-L229
if resp.input_len >= parent_len:
    logprobs = (
        parent_logprobs
        + [0.0] * (resp.input_len - parent_len)
        + resp.output_logprobs
    )
    loss_mask = (
        parent_loss_mask
        + [0] * (resp.input_len - parent_len)
        + [1] * resp.output_len
    )
    versions = (
        parent_versions
        + [-1] * (resp.input_len - parent_len)
        + resp.output_versions
    )
```

The per-token `versions` field records which policy version sampled each token. AReaL's optional `mask_stale_tokens` uses it to drop loss targets older than the staleness bound ([[areal-agent-pipelines]]; `cli_args.py` L3312–3319). Because a long trajectory spans several policy versions, the mask in AReaL can depend on age as well as on token origin.

**Rocket.** The learner fetches token ids and sampling log-probs from a rollout store, and the controller packs *training segments*, each consisting of one example and one or more model turns, into batches of references and advantages ([[mai-rl-infrastructure]] Fig. 1, steps 1 and 6). The post does not show the mask code. Its unit of training is the model turn, which is the same separation the three open frameworks implement with masks.

### §5.3 The comparison

| Property | slime search-r1 | verl ToolAgentLoop | AReaL concat |
|---|---|---|---|
| Mask for tool text | 0 | 0 | 0 |
| Mask for template boundary tokens | tool text is tokenized as a delta; no separate boundary handling in this example | 0 (`inserted_token_ids`) | 0 (part of the prompt span) |
| Next-turn input to engine | text (re-tokenized) | token ids | token ids, with a parent-prefix check |
| Per-token policy version | not recorded in this example | not recorded | recorded (`versions`) |

---

## §6 Outcome graders

| Source | Grader | Notes |
|---|---|---|
| Search-R1 | EM of the extracted `<answer>`; no format reward | §3.4, Eq. 4 ([[search-r1]]) |
| slime search-r1 at 79aa41a | effectively binary EM | `reward_func` passes `format_score=0.2`, but `compute_score_em` never reads it; partial-credit terms default to 0 ([[slime-search-r1-pipeline]] Verification) |
| verl `search_r1_like_qa_em` | 1 on EM, 0 otherwise; 0.25 if more than 10 answer tags | code below |
| ZeroSearch | word-level F1, chosen after EM led to "excessively long answers" | §3.5 ([[zerosearch]]) |
| DeepResearcher | −1 if format incorrect, else word-level F1 | §3.4 ([[deepresearcher]]) |
| ASearcher | base models: format reward × F1; QwQ agent: LLM-as-Judge, no format reward | §3.4 ([[asearcher]]) |
| WebSailor | 0.1 · R_format + 0.9 · R_answer (LLM judge) | §4.2, Eq. 5 |
| Tongyi DeepResearch | 0/1 correctness; no format reward because the cold start already fixes format | §3.4.3 |
| GLM-4.5 | final-answer accuracy; a wrong tool-call format halts the trace with reward 0 | §3.3.2 ([[glm-4-5]]) |

**verl's EM preset caps answer-tag spam.**

```python
# verl@8357f1e verl/utils/reward_score/search_r1_like_qa_em.py L119-L128
if answer is None:
    return 0
else:
    if em_check(answer, ground_truth["target"]):
        if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
            score = score / 4
            return score
        return score
    else:
        return format_score
```

`answer` is the last `<answer>` match (L84–85), so a response that lists many candidate answers in separate tags is graded on the last one and divided by 4 if it uses more than 10 tags. The cap is a patch for one hacking pattern.

**AReaL's judge reward.** The Tongyi-style example asks the judge whether the prediction is "equivalent" to the label. Three code details set its behaviour: the prediction is cut to its first 200 characters before judging (`pred_answer=pred_answer[:200]`), the judge samples at temperature 1.0, and any judge error gives reward 0.0 (`react_agent.py` L300–317). Score = 1.0 only if the parsed judgement is "correct" (L61). The 200-character cut removes the long-answer exploit that EM invites (§6 table, ZeroSearch). Judge sampling at temperature 1.0 adds grader noise: the same response can be scored differently in two rollouts of one group.

**Choosing a grader.**
- EM is precise and cheap but rewards guessing many forms of the answer when the extractor allows it, and penalizes correct paraphrases.
- F1 tolerates paraphrase and length differences. ZeroSearch adopted it after EM hacking ([[zerosearch]] §3.5).
- An LLM judge handles free-form answers. ASearcher reports both F1 and LLM-as-Judge ("LasJ"), and the rankings can differ: on HotpotQA, Search-R1-base has higher F1 than DeepResearcher but similar MBE (model-based evaluation), which DeepResearcher attributes to Search-R1-base having direct access to the Wikipedia corpus the answers come from ([[deepresearcher]] §5.2.4). The specific Table 1 cells are not quoted here: the DeepResearcher card marks its Table 1 values as reassembled from a column-scrambled text extraction and asks for a recheck against the PDF before they are quoted. This chapter uses only the §5.2.4 prose claim.

---

## §7 Long horizons and context management

**Turn limits used in training.** 4 actions ([[search-r1]]); 2 SGLang turns in slime's example (§3.1); up to 10 tool calls ([[deepresearcher]] §5.1.1); up to 30 ([[websailor-tongyi-deepresearch]] App. A.3); 32 for 7B/14B and 128 for QwQ-32B ([[asearcher]] §4.1); 100 LLM calls in AReaL's Tongyi-style config ([[areal-agent-pipelines]], `config.yaml` L189).

**Evidence that the horizon grows during training.** In Search-R1 the response length first drops for about 100 steps while reward rises slightly, then both increase as the model searches more often ([[search-r1]] §5.3). In ASearcher, search queries per trajectory rise to about 6 for local 7B/14B models, and the QwQ agent reaches about 40 calls at step 200 with peaks near 70 ([[asearcher]] §4.3). DeepResearcher's tool calls per question increase over training for all hop counts, and the 4-hop setting is still rising after step 34 ([[deepresearcher]] §6.1).

**Context handling.**

| Method | Source |
|---|---|
| Browsing sub-agent reads a page segment by segment and returns only new relevant information | [[deepresearcher]] §3.1 |
| The QwQ agent keeps summarized thoughts and tool calls, passes only the most recent 25k characters (≤ 10k tokens input) | [[asearcher]] §3.1 |
| Markovian context: condition on the question, an evolving report S_t, and the last action/observation, trained jointly with ReAct data | [[websailor-tongyi-deepresearch]] Tongyi §3.1, §3.4.2 |
| Truncate each tool response (verl default 256 characters, middle cut) | `rollout.yaml` L223, L226 |
| Hard stops: 150-minute wall clock; forced answer at 80% of 27,648 tokens | `react_agent.py` L78, L145; `config.yaml` L190 |

**Test-time context management (DeepSeek-V3.2).** When context use exceeds 80% of the window, a policy is applied. "Summary" extends the average to 364 steps and reaches up to 60.2 on BrowseComp; "Discard-all" reaches 67.6, comparable to parallel scaling with fewer steps. About 20%+ of BrowseComp cases exceed 128K ([[deepseek-v3-2]] §4.1, §4.4, Fig. 6). This is an inference-time result. It shows that the context policy used at test time changes the score by several points, so the policy used in RL rollouts should match the one used at deployment.

**Training length and RL context.** With curation by a 64k model, Tongyi's 64k RL reaches the highest reward and lengthens responses, 48k stays flat, and 32k shortens responses (Fig. 9, [[websailor-tongyi-deepresearch]] Tongyi §4.4).

---

## §8 Asynchronous rollout for hour-long trajectories

### §8.1 The problem

ASearcher states the scheduling problem directly: the longest trajectories use dozens more tool calls and up to two orders of magnitude more tokens than short ones, and a batch-generation system waits for the slowest trajectory ([[asearcher]] §3.3.1, Fig. 6–7). WebSailor's RL on synchronous verl was "limited to 50 steps" ([[websailor-tongyi-deepresearch]] §5.4). Rocket's Fig. 4 shows the same effect inside one GRPO group: weights from learner steps 104–124, the slowest rollout ran 784 turns, the fastest waited 7.6 h, the median waited 5.6 h, and all three trained together at step 127 ([[mai-rl-infrastructure]] p. 6).

### §8.2 Three scheduling designs

**Fully asynchronous with an admission-time bound (ASearcher on AReaL).** A training step starts once enough trajectories are collected, and long trajectories span several policy versions ([[asearcher]] §3.3.2). AReaL admits a new rollout only while capacity remains under both a concurrency bound and a staleness bound:

```python
# AReaL@bcef492 areal/infra/staleness_manager.py L99-L113
with self.lock:
    current_version = self.version_provider.get_version()
    # Calculate concurrency-based capacity
    max_concurrent_rollouts = max(1, self.max_concurrent_rollouts)
    concurrency_capacity = max_concurrent_rollouts - self.rollout_stat.running

    # Calculate staleness-based capacity
    ofp = self.max_staleness
    sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
    consumer_bs = max(1, self.consumer_batch_size)
    staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt

    # Return the minimum of both constraints
    capacity = min(concurrency_capacity, staleness_capacity)
    return capacity
```

The search configs set `max_head_offpolicyness: 4`, with 128 (ASearcher 1.5B local) or 160 (Tongyi-style) concurrent rollouts ([[asearcher]] AReaL code; [[areal-agent-pipelines]] `config.yaml` L24–28). The bound is on the version at admission. A trajectory admitted at version v can still run while the learner advances, and its later tokens are sampled by newer weights. The per-token `versions` field (§5.2) records this.

**Step-level asynchronous, strict on-policy (Tongyi DeepResearch).** Inference and tool servers run separately, and the ratio stays at 1.0 ([[websailor-tongyi-deepresearch]] Tongyi §3.4.3). Partial rollouts are listed as future work (§5 limitations).

**Rocket.** Each request inherits a priority from its rollout's start time, so older in-flight rollouts are served first. Failed rollouts are dropped, not retried. Weight transfer takes 15 s "in our current configuration". The job is kept inference-bound ([[mai-rl-infrastructure]] p. 6–7). MAI Thinking 1 drops rollouts more than 8 updates stale, which is 40 steps at 5 gradient steps per inference update ([[mai-thinking-1]] hyperparameter table). The post also notes that adding inference servers at fixed batch size can increase staleness, because steps finish faster while a long rollout's wall-clock time is unchanged (p. 7). Search rollouts are the case where this applies most.

**Comparison.** AReaL bounds staleness at admission time by version count. Rocket bounds it at training time by dropping rollouts that are too old, and lowers it with priority scheduling and fast weight sync. Tongyi avoids it by staying on-policy at the cost of throughput. ch-03 and ch-04 give the staleness definitions and levers.

---

## §9 Algorithm notes for search

ch-18 covers algorithm selection. The search-specific results:

- Search-R1 defaults to PPO because GRPO converged faster but showed reward collapse in long runs ([[search-r1]] §5.1, Fig. 2a). PPO vs GRPO EM: 7B-base 0.431 vs 0.350; 7B-instruct 0.385 vs 0.396; 3B-base 0.303 vs 0.312; 3B-instruct 0.325 vs 0.336 (Table 3). GRPO scored higher in three of four settings; PPO's advantage is stability, not final score.
- ZeroSearch ranks REINFORCE 34.47, GRPO 33.17 and PPO 32.67 on Qwen-2.5-3B-Base (Table 5).
- WebSailor's DUPO duplicates non-zero-std groups in the batch instead of resampling, about 2–3× faster than DAPO dynamic sampling ([[websailor-tongyi-deepresearch]] WebSailor §4.2).
- Tongyi uses token-level loss, clip-higher and a leave-one-out baseline, and excludes some negatives (for example length-truncated with no answer), because unfiltered negatives "significantly degrade training stability and can lead to policy collapse". The report adds that "the success of agentic RL depends more on the quality of the data and the stability of the training environment than on the specific algorithm being used" (Tongyi §3.4.3).
- [[gemini-2.5-deep-research]] names no RL algorithm and gives no RL details for Deep Research.

---

## §10 Stage position

Search RL appears as (a) a single RL stage from a base or instruct model (Search-R1, DeepResearcher, ASearcher); (b) RL after an RFT or SFT cold start (WebSailor, Tongyi); (c) part of an agentic RL expert after reasoning RL (GLM-4.5); or (d) a specialist RL run followed by distillation and joint mixed RL (DeepSeek-V3.2) ([[recipe-crosswalk]] Search synthesis, step 8). WebSailor's Fig. 6 compares (a) and (b) on Qwen2.5-32B-Instruct: direct RL rises faster in Pass@1 but converges lower than RFT-then-RL, with fewer tool calls ([[websailor-tongyi-deepresearch]] §5.3). ch-17 covers how search fits into multi-domain recipes.

---

## Recipe

1. **Pick the question source by deployment.** Multi-hop QA for short factual answers; synthesized obfuscated questions (injection/fuzz, KG subgraphs) when the target is long-horizon browsing.
2. **Filter.** Drop questions the base model answers without tools (pass@10). Drop time-sensitive and subjective items. Keep hard but solvable, tool-requiring items (ASearcher's 0/16, ≥ 50%, ≤ 1 turn rules).
3. **Choose the tool backend.** Local index for corpus-contained answers. Live web with cache, retries, QPS limits and failover when answers are outside a corpus. A simulator for cost-bound iteration, always evaluating on real search. Estimate cost: requests per step = prompts × group size × expected calls.
4. **Bound tool load.** Set a concurrency limit (slime uses a semaphore of 256), timeouts, and a rate limiter per provider. Treat tool errors as an observation (an error message) rather than a crash.
5. **Build the rollout from token ids.** Append model turns from the engine's ids and log-probs. Tokenize tool output as a delta. Never re-tokenize the full history ([[tito-token-in-token-out]]).
6. **Mask** all tool output and template boundary tokens. Assert in code that every trainable token has a sampler log-prob (slime's `append_response_tokens` contract).
7. **Stop at tool tags** so the engine does not generate past `</search>` or `</answer>` (slime's L164–177 fix).
8. **Choose the grader.** F1 or a judge for free-form answers; EM only with a guard against multi-answer spam. If using a judge, truncate the judged answer, set judge temperature deliberately, and log judge failures separately from wrong answers.
9. **Set the horizon.** Start with a small turn cap and raise it in stages ([[agentgym-rl]] via ch-13), or start at the target cap if you have an SFT cold start. Raise framework defaults such as `max_tool_response_length` (256 characters in verl).
10. **Go asynchronous** once turn caps exceed about 10 or the p99/median rollout duration ratio is large. Bound staleness (AReaL `max_head_offpolicyness`, Rocket's drop-after-N-updates), record per-token versions if available, and prioritize old in-flight rollouts.
11. **Evaluate** on in-domain and out-of-corpus sets (Bamboogle-style), with the same context-management policy used in rollouts, and report tool calls per question alongside accuracy.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Tool output not masked | reward rises then falls; the model reproduces passage text | count trainable tokens inside `<information>` / tool spans; must be 0 (0.431 vs 0.343 in Search-R1) |
| Re-tokenizing decoded history | sampler/learner log-prob gap concentrated at turn boundaries | compare engine-returned ids with re-tokenized ids per turn; log the mismatch count |
| No contamination filter | high reward with few or zero searches per trajectory | log reward split by number of tool calls; drop pass@10-without-tools items |
| EM reward on free-form answers | answers lengthen; several candidate answers appear | log answer length and answer-tag counts; switch to F1 or a judge |
| Judge failures scored as wrong | reward dips correlated with judge latency | log judge error rate separately |
| Tool responses truncated by a small default | the model searches repeatedly for the same query | log tool-response length before and after truncation; raise `max_tool_response_length` |
| Generation not stopped at tool tags | text after `</answer>` trained with mask 1; format check failures | add stop strings; check the tail after the closing tag is empty |
| Synchronous batching with long-tailed trajectories | GPU idle time grows with turn cap; few RL steps per day | log per-step wall time and p99/median rollout duration |
| Simulator with answer leakage in training | training reward much higher than real-search evaluation | evaluate on real search every N steps; report both |
| Turn cap too low for the target task | tool calls per question saturate at the cap | plot the calls-per-question histogram; mass at the cap means it binds |

---

## Check your understanding

1. Search-R1 finds that masking retrieved tokens raises EM from 0.343 to 0.431. Explain what gradient the unmasked retrieved tokens receive in a correct trajectory and in an incorrect one, and why this can lower answer accuracy even though the retrieved text is correct information.
2. slime sends the next turn to SGLang as text while training on the engine's sampled ids. Describe a concrete tokenization event at a turn boundary that would make the two sequences differ, and which quantity in the importance ratio it would corrupt.
3. ZeroSearch's simulator sees the ground-truth answer. Explain why this does not inflate the reported test numbers but does make its training reward incomparable to a live-search run. What would you measure to estimate how often the simulator leaks the answer?
4. DeepResearcher drops any question that 1 of 10 base-model samples answers without search. Using the group-advantage formula, explain what such a question would teach a search agent if it stayed in the set.
5. AReaL's staleness bound is applied at admission time. Explain why a trajectory admitted within the bound can still produce tokens much older than the bound by the time it is trained, and how the per-token `versions` field and `mask_stale_tokens` address this.
6. Rocket notes that adding inference servers at a fixed batch size can increase staleness. Explain the mechanism using an 89-turn web-research rollout of 51 minutes and a learner step that becomes shorter.
7. The AReaL judge reward truncates the prediction to 200 characters and samples the judge at temperature 1.0. State one hacking pattern the truncation prevents and one source of advantage noise the temperature introduces.

---

## Connections

- **ch-03, ch-04** — staleness definition and levers; §8 applies them to the longest rollouts.
- **ch-05** — weight transfer; Rocket's 15 s transfer and in-flight updates.
- **ch-08** — agentic rollouts, tools and sandboxes in general.
- **ch-10** — multi-turn rollout design, token-in/token-out.
- **ch-11** — grader design; §6 is the search instance.
- **ch-13** — horizon and toolset axes.
- **ch-17** — where search sits in multi-domain recipes.
- **ch-18** — PPO vs GRPO vs DUPO for search.
- **llm-training ch-45b** (multi-turn agentic RL: observation masking, credit assignment, stability) — the general treatment of masking and credit assignment.
- **llm-training ch-40** (group baselines) and **ch-43** (entropy and KL control) — algorithm mechanics behind §9.
- **llm-training ch-44a** (length in RL) — long-horizon length effects.

---

## Sources

- [[search-r1]] — rollout protocol, retrieved-token mask ablation, EM grader, PPO vs GRPO, training dynamics.
- [[zerosearch]] — simulated search LLM, noise curriculum, F1 after EM hacking, mask ablation, cost comparison.
- [[asearcher]] — long-horizon search agents on AReaL, QA synthesis and filters, F1 / judge rewards, turn limits, conflicts in reported maxima.
- [[deepresearcher]] — live-web RL, tool cluster, cache and rate limits, contamination filter, F1 with format penalty (Table 1 cells flagged unverified and not used).
- [[websailor-tongyi-deepresearch]] — SailorFog-QA, DUPO, synchronous 50-step limit; Tongyi's sandbox, simulated Wiki, strict on-policy async RL, negative exclusion.
- [[gemini-2.5-deep-research]] — confirms no RL algorithm is disclosed for Deep Research.
- [[mai-rl-infrastructure]] — web-research rollout durations, Fig. 4 group timeline, priority, drop-failed, weight-transfer time, the staleness counter-intuition.
- [[mai-thinking-1]] — absence of a web-search climb; 40-step staleness cutoff.
- [[slime-search-r1-pipeline]] — slime@79aa41a search-r1 example code.
- [[verl-search-tool-pipeline]] — verl@8357f1e tool loop, mask builder, EM preset, and the list of removed search files.
- [[areal-agent-pipelines]] — AReaL@bcef492 Tongyi-style search workflow, judge reward, concat mask, staleness manager.
- [[tito-token-in-token-out]] — re-tokenization drift in multi-turn rollouts.
- [[recipe-crosswalk]], [[algorithm-crosswalk]] — search pipeline synthesis and algorithm table.
- [[deepseek-v3-2]], [[deepseek-v3.1]] — search-agent task counts and test-time context management.
- [[glm-4-5]] — search questions and reward in GLM-4.5's agentic RL.
