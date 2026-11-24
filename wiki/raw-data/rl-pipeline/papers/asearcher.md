<!-- scope: ASearcher ("Beyond Ten Turns"): fully asynchronous agentic RL on AReaL for search+browse agents; 32-128 turn limits, QA synthesis agent (injection/fuzz + 3-step verification), GRPO with dynamic filtering, F1×format or LLM-judge reward; AReaL example configs at bcef492
     deps: [[search-r1]], [[areal-async-rl]], [[grpo]]
     see-also: [[zerosearch]], [[deepresearcher]], [[websailor-tongyi-deepresearch]], [[gemini-2.5-deep-research]], [[slime]], [[mai-rl-infrastructure]], [[recipe-crosswalk]]
-->

# ASearcher: Beyond Ten Turns — Long-Horizon Agentic Search with Large-Scale Asynchronous RL
- **Core Insight:** Search-agent RL with small turn limits (Search-R1 trains with ≤ 4-10 turns) yields a short tool-use horizon (§1, §2); raising the limit to 32 (7B/14B) or 128 (QwQ-32B) turns is only practical when trajectory execution is decoupled from model updates, because the longest trajectories use dozens more tool calls and up to two orders of magnitude more tokens than short ones, and a batch-generation system waits for the slowest trajectory (§3.3.1, Fig. 6-7).
- **Guideline:** For long-horizon search RL, use fully asynchronous rollout (training step launches when enough trajectories are collected; long trajectories span multiple policy versions), filter training QAs to "hard but solvable and tool-requiring", drop zero-variance groups, and run a second curriculum stage restricted to questions needing ≥ 5 tool calls (§3.2.1, §3.3.2, §3.4, §4.1).
- **Authors:** Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, Yi Wu (Tsinghua IIIS, Ant Group, University of Washington)
- **Year:** 2025 (arXiv v4 2025-10-26)
- **URL:** https://arxiv.org/abs/2508.07976 (code/data: https://github.com/inclusionAI/ASearcher)
- **Source type:** paper + released config/code (AReaL clone `wiki/raw-data/rl-pipeline/repos/AReaL` at bcef492, 2026-11-24)
- **Relevant topics:** asynchronous agentic RL, turn limits, QA synthesis for search agents, dynamic filtering, LLM-as-judge reward, webpage summarization trained end-to-end, curriculum

## Summary
ASearcher trains search agents with two tools, a search engine (returns snippets and URLs) and a browser (returns page content), with no external LLM; the agent itself summarizes long pages, and that summarization is trained by RL with the rest of the trajectory (§3.1, Fig. 2). Two instantiations: base LLMs (Qwen2.5-7B/14B) from zero RL with append-only history, and an LRM agent (QwQ-32B) with separate prompts for tool selection, summarization and answering, whose history keeps summarized thoughts and tool calls and passes only the most recent 25k characters (≤ 10k tokens input) (§3.1). Training runs on AReaL with fully asynchronous rollout (§1, §3.3.2).

## Key Contributions
- Fully asynchronous agentic RL: independent trajectory execution (tool and LLM requests per trajectory) plus decoupled rollout/training (§3.3.2, Fig. 7).
- QA synthesis agent: from a seed QA, iterate Injection (add a fact about an entity, e.g., from Wikipedia) and Fuzzing (blur a detail); verify each step by (1) LLM basic-quality/grounding check against tracked supporting facts, (2) difficulty via QwQ-32B answering without tools, (3) answer-uniqueness check of mismatched answers (§3.2.2, Fig. 4, Table 1).
- Open-source data filtering by a trained RL model's 16 samples per question (§3.2.1).
- Two-stage curriculum for the QwQ agent (§4.1).

## Key Figures/Tables to Study
- **Fig. 6:** accuracy vs enforced minimum turns; tool calls and tokens over training (variance across trajectories).
- **Fig. 7:** one-step-off vs fully asynchronous RL timeline.
- **Tables 2-3:** local-RAG and web settings on 7 QA sets (F1 and LLM-as-Judge "LasJ").
- **Table 4:** GAIA / xBench-DeepSearch / Frames Avg@4 and Pass@4; **Table 5:** Pass@1 with external summarizer and test-time search.

## Technical Details
**Pipeline view.**
| Component | ASearcher setting | Location |
|---|---|---|
| Prompt source | HotpotQA + 2WikiMultiHopQA train, filtered 304k → 16k; small WebWalkerQA subset; 25,624 synthetic QAs from 14,107 seeds; two 35k training sets (7B/14B and QwQ) | §3.2.1-3.2.2; §4.1 |
| Environment / tools | local RAG over Wikipedia 2018 (Local models) or web search + browser (Web models) | §3.1; §4.1 |
| Rollout shape | turn limit 32 (7B/14B), 128 (QwQ); tool calls > 100 turns and > 400k output tokens observed (abstract); ~40 calls at step 200, peaks ~70 (§4.3) | §4.1; §4.3 |
| Grader | sparse, at trajectory end; base LLMs: format reward × F1; QwQ: LLM-as-Judge, no format reward | §3.4 |
| Loss | GRPO, trajectory-level advantage Â_i, sum over generated action tokens a_t only (observations not in the sum) | §3.4, Eq. 1 |
| Filtering | drop queries whose responses all get identical reward | §3.4 |
| Infra | AReaL, fully asynchronous; batch 128 (7B/14B), 64 (QwQ); ~16k H800 GPU hours for ASearcher-Web-QwQ | §3.3.2; §4.1 |
| Stage position | single RL stage (base models from zero RL); QwQ: stage 1 mixed difficulty → stage 2 only questions needing ≥ 5 tool calls | §3.1; §4.1 |

**Data filter (§3.2.1).** Generate 16 responses per question with an RL-trained model; drop a question if none is correct, if accuracy ≥ 50%, or if it is solved with ≤ 1 search turn. Synthesis averages 6.3 injections and 3.2 fuzzes per seed; up to three variants per seed are kept; kept questions average 4.27 injections and 2.10 fuzzes (§3.2.2).

**Results.** Local-RAG 7B average (Table 2): ASearcher-Local-7B F1 58.0 / LasJ 61.0 vs Search-R1-7B 54.3 / 55.4. Web setting (Table 3): ASearcher-Web-14B F1 61.5, LasJ 64.5; ASearcher-Local-14B evaluated zero-shot on the web reaches F1 60.0, LasJ 65.6. Table 4 (Avg@4 / Pass@4): ASearcher-Web-QwQ-v2 GAIA 58.7 / 74.7, xBench-DeepSearch 51.1 / 75.0, Frames 74.5 / 85.5; Search-R1-32B GAIA 28.6 / 43.7. RL gains on QwQ-32B: +15.0 GAIA, +22.4 xBench, +14.6 Frames (§4.2).

**Training dynamics (§4.3).** Local-7B/14B: search queries per trajectory rise to about 6. The 7B model does not learn valid webpage browsing; the 14B model learns it late in training; the authors hypothesize 7B capacity is too small to learn long-page summarization in zero RL.

**AReaL code (repo at bcef492).**
- `examples/search_agent/local_1.5b_example.yaml`: `experiment_name: asearcher-1.5b-zero-local` (L1), `max_concurrent_rollouts: 128` (L26), `max_head_offpolicyness: 4` (L29), `n_samples: 16`, `max_new_tokens: 1024` (L37-39), Qwen2.5-1.5B (L47), `eps_clip: 0.4` (L64), `reward_scaling: 10.0`, `reward_bias: -0.5`, `kl_ctl: 0.0` (L66-68), `use_decoupled_loss: true` (L71), group-level reward norm (L75-78), agent stats include `num_search_queries`, `num_pages`, `judge_q_invalid`, `format_reward` (L83-95), `max_turns: 32` (L178).
- `max_head_offpolicyness` help text: "If the current version is more than this many versions behind, the request will not be accepted" (`areal/api/cli_args.py` L3304-3311).
- `examples/search_agent/tongyi_deepresearch/`: trains Qwen3-1.7B on ASearcher-LRM-35k with search and visit tools (`README.md` L8-10); a Qwen2.5-72B-Instruct SGLang judge is launched as a separate engine (`README.md` L20-28; `train.py` L127-146). Config: `max_concurrent_rollouts: 160`, `max_head_offpolicyness: 4`, `n_samples: 8`, `max_new_tokens: 8192` (`config.yaml` L24-37), `max_llm_calls_per_run: 100`, `max_tokens_per_trajectory: 27648` (L189-190).
- Reward: judge prompt asks whether the prediction is "equivalent" to the label; score = 1.0 iff judgement == "correct"; judge failure gives 0.0 (`react_agent.py` L42-61, L276-318). Agent stops at 150 minutes wall-clock (L145) or forces an answer at 80% of the token budget (L78, L191-216).
- Loss masking in `concat` export: tokens inserted between a parent turn's output and the child turn's new output (tool responses, prompts) get `loss_mask` 0 and version −1; only sampled output tokens get 1 (`areal/experimental/openai/types.py` L214-224, L257-258).

## Source-internal conflicts
- Frames RL gain: +15.6 (Fig. 1 caption) vs +14.6 (§4.2). Max output tokens: > 400k (abstract, §1) vs "over 150k" (§4.3, training-dynamics text). ASearcher-Local-14B LasJ: 63.6 (Table 2 local) vs 65.6 (§4.2 text, matching Table 3 web).

## Verification
- Created 2026-11-24 from arXiv:2508.07976v4 PDF: §1-4, Tables 1-5 (Table 5 numbers not transcribed due to extraction order), App. A heading read.
- No async-vs-synchronous wall-clock speedup number is reported in the sections read.
