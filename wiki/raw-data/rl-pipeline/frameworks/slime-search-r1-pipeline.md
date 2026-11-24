<!-- scope: THUDM/slime examples/search-r1 at commit 79aa41a: local dense retrieval server (e5 + FAISS over wiki-18), custom multi-turn generate function, <search>/<answer> tag parsing, loss mask for retrieved tokens, EM reward, launch script
     deps: [[slime]], [[search-r1]]
     see-also: [[verl-search-tool-pipeline]], [[areal-agent-pipelines]], [[zerosearch]], [[asearcher]], [[slime-agentic-and-mismatch-examples]], [[grpo]]
-->

# slime `examples/search-r1`: a search-agent RL pipeline in one generate function
- **Core Insight:** The whole agent loop is a custom `generate(args, sample, sampling_params)` that alternates SGLang `/generate` calls with HTTP retrieval calls; model tokens are appended with `loss_mask=1`, retrieved `<information>` tokens with `loss_mask=0`, so the trainer sees one concatenated sequence and trains only on the model's own tokens (`generate_with_search.py` L179–253).
- **Guideline:** Stop generation at the closing tool tag (`</search>`, `</answer>`) instead of post-processing the decoded string, because trimming text after generation breaks the token/log-prob alignment needed for importance-sampling correction; the file's own bugfix comment records that the untrimmed tail was otherwise trained with `loss_mask=1` and broke the format check (L164–177).
- **Authors:** slime contributors (THUDM / Z.ai); file adapted from PeterGriffinJin/Search-R1 at ceee7b8 (L1).
- **Year:** 2025–2026; code read at commit 79aa41a8 (2026-11-06).
- **URL:** https://github.com/THUDM/slime/tree/79aa41a884ab0bf65222c94418f6357d3c31192b/examples/search-r1
- **Source type:** released config/code
- **Relevant topics:** search-agent RL, multi-turn rollout, tool-output loss masking, exact-match reward, local retrieval server, custom generate/reward plug-ins

## Summary
The example reproduces Search-R1 ([[search-r1]]) inside slime ([[slime]]). Data is the Search-R1 NQ+HotpotQA parquet; retrieval is a separate FastAPI server (`local_dense_retriever/retrieval_server.py`) serving an e5-base-v2 FAISS flat index over a wiki-18 corpus, or a Google (Serper-style) backend (`google_search_server.py`). slime loads the generate and reward functions by path (`--custom-generate-function-path`, `--custom-rm-path`), so the trainer, rollout manager and weight sync are the stock ones.

## Key Contributions
- Two-backend search shim with the same output format, `[{"document": {"contents": '"<title>"\n<text>'}}]` (`local_search_server.py` L25–97).
- Global `asyncio.Semaphore(256)` on search calls to bound retrieval-server load (`generate_with_search.py` L18, L41, L129).
- Per-token rollout log-probs taken from SGLang `output_token_logprobs` so tokens and log-probs stay aligned (L198–210).
- Optional TIS/MIS correction flags kept commented in the launch script (`run_qwen3_4B.sh` L80–81, L122–125).

## Key Figures/Tables to Study
- `generate_with_search.py` L145–274 (generate), L277–293 (reward); `qa_em_format.py` L156–208 (score); `run_qwen3_4B.sh` L29–155; `README.md` L210–265 (retriever setup).

## Technical Details
**Pipeline at a glance**
1. Data: `--prompt-data .../nq_hotpotqa_train/train.parquet --input-key prompt --label-key reward_model --apply-chat-template` (`run_qwen3_4B.sh` L30–33); the reward reads `sample.label["ground_truth"]["target"]` (`generate_with_search.py` L289; `qa_em_format.py` L197).
2. Retrieval server: `retrieval_server.py --index_path e5_Flat.index --corpus_path wiki-18.jsonl --topk 3 --retriever_model intfloat/e5-base-v2 --faiss_gpu`, listening at `http://127.0.0.1:8000/retrieve` (`README.md` L253–265; `generate_with_search.py` L24).
3. Turn loop: up to `max_turns = 2` SGLang calls with stop tags (L16, L173–188).
4. Tool parsing: first `<search>…</search>` or `<answer>…</answer>` match (L111–121); invalid action → corrective text observation, not termination (L136–140).
5. Observation append with `loss_mask=0` (L239–248).
6. Reward: `compute_score_em` on `prompt + response` (L287–291).
7. Train/sync: stock `train.py` with `--colocate`, 4 actor GPUs + 4 rollout GPUs (`run_qwen3_4B.sh` L141–145).

**Turn loop and loss mask.**
```python
# generate_with_search.py L217-219, L235-244
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
`Sample.append_response_tokens` enforces the contract: trainable tokens require rollout log-probs, non-trainable tokens must not pass them and get log-prob 0.0 (`slime/utils/types.py` L277–305). A `length` finish breaks the loop and sets `Status.TRUNCATED`; an SGLang `abort` returns `Status.ABORTED` immediately so slime can recycle the sample (L190–193, L266–272).

**Observation format.** Retrieved passages become `Doc i(Title: …) text` lines wrapped as `\n\n<information>…</information>\n\n` (L44–56, L131).

**Reward.** `em_check` normalizes (lowercase, strip punctuation and articles, collapse whitespace) and compares the last `<answer>` content to each gold answer (`qa_em_format.py` L21–48, L126–138). `compute_score_em` also checks tag-sequence validity (`is_valid_sequence`) and whether a gold answer appears inside an `<information>` block (L51–153).

**Launch settings (`run_qwen3_4B.sh`).** Model Qwen3-4B-Instruct-2507; `--rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256`, max response 131,072, temperature 1 (L35–49); GRPO with `--use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`, entropy 0 (L71–78); Adam lr 1e-6 constant with CPU offload (L84–93); TP 2, CP 2, dynamic batch with 9,216 tokens/GPU (L53–68); SGLang 2 GPUs per engine, mem fraction 0.7 (L103–106).

## Verification
- Checked on 2026-11-24 against the local clone at 79aa41a8. All loci refer to that commit.
- Reward is binary at this commit: `reward_func` passes `format_score=0.2` (L290), but `compute_score_em` never reads `format_score`; the partial-credit branches use `structure_format_score`, `retrieval_score` and `final_format_score`, which all default to 0 (`qa_em_format.py` L156–208). The inline comments `# 0.3`, `# 0.2`, `# 0.8`, `# 0.1` describe non-default settings. Effective reward: 1.0 for a correct EM answer, else 0.
- The `return_logprob=False` branch passes `log_probs=None` with `trainable=True` (L220–226), which `append_response_tokens` rejects with `ValueError` (`types.py` L300–301). Only the default `return_logprob=True` path is consistent at this commit.
- Not found in the example: a retrieval cache or simulated search (contrast [[zerosearch]]); F1 or LLM-judge graders; async/partial rollout (`generate` asserts `not args.partial_rollout`, L146).
