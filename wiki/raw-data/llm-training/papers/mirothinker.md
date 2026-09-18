<!-- scope: MiroThinker v1.0 (MiroMind, Nov 2025) — open research-agent models (8B/30B/72B) trained with agentic SFT, DPO + SFT loss, and online GRPO; recency-based tool-output retention; "interactive scaling" as SFT-vs-RL interaction depth
     deps: [[grpo]], [[dpo]]
     see-also: [[tongyi-deepresearch]], [[websailor]], [[webshaper]], [[kimi-researcher]], [[context-folding]], [[resum]]
-->

# MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents via Model, Context, and Interactive Scaling
- **Core Insight:** After online GRPO, MiroThinker-v1.0-30B makes more tool-call turns than its SFT checkpoint, and the authors report 8–10 point average gains: BrowseComp 32.2 → 41.2, BrowseComp-ZH 37.6 → 47.8, HLE 24.1 → 33.4, GAIA 65.4 → 73.5; the 72B model reaches 81.9 GAIA, 37.7 HLE, 47.1 BrowseComp, 55.6 BrowseComp-ZH with up to 600 tool calls in a 256K context (§6.3, Fig. 5; Table 1).
- **Guideline:** When a research agent must run hundreds of tool calls in a fixed context window, keep all thoughts and actions but only the K most recent tool responses, because the authors report that this retention rule (K = 5 at evaluation) "does not lead to degradation in performance" and allows 600 tool calls in 256K tokens (§3.3, §6.1); no ablation table is given, so verify on the target task.
- **Authors:** MiroMind Team: Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, et al. (alphabetical, App. A)
- **Year:** 2025 (arXiv v1 2025-11-14; v2 2025-11-18; v3 2026-04-21; technical report)
- **URL:** https://arxiv.org/abs/2511.11793 ; weights https://huggingface.co/miromind-ai/MiroThinker-v1.0-72B ; code https://github.com/MiroMindAI/MiroThinker
- **Source type:** official technical report (with model cards)
- **Relevant topics:** research agents, deep search, agentic SFT, agentic DPO, agentic RL, GRPO, context management, interaction depth, multi-hop QA synthesis

## Abstract
MiroThinker v1.0 is an open-source research agent for tool-augmented reasoning and information seeking. The authors add interaction depth as a third scaling dimension next to model size and context length: the model is trained to handle deeper and more frequent agent–environment interactions. They contrast this with test-time scaling of reasoning length alone, which they say risks degradation on long chains, while interaction uses environment feedback to correct errors. Through reinforcement learning, the model performs up to 600 tool calls per task within a 256K context window. The 72B variant scores up to 81.9% on GAIA, 37.7% on HLE, 47.1% on BrowseComp, and 55.6% on BrowseComp-ZH. The authors report that performance improves as interaction depth increases.

## Key Contributions
- Three released sizes (8B, 30B, 72B) with a ReAct single-agent workflow and a tool suite: Linux sandbox, file transfer, Google search, and an LLM-based scrape-and-extract tool (§1, §3.2).
- Recency-based context retention (Eq. 4–7) and truncation of long tool outputs with a "[Result truncated]" tag (§3.3).
- A synthetic data pipeline (MiroVerse v1.0): MultiDocQA from hyperlink graphs with constraint obfuscation, plus multi-model, multi-paradigm trajectory synthesis (§4, Fig. 3).
- A three-stage training pipeline: agentic SFT, DPO with an auxiliary SFT loss, and fully online GRPO with streaming rollouts and trajectory filtering (§5).
- An SFT-vs-RL comparison of turn distributions and accuracy that the authors name interactive scaling (§6.3, Fig. 5).

## Key Figures/Tables to Study
- Table 1 (8 benchmarks, all three sizes, with baselines and ± error bars); Fig. 5 (turn histograms and CDFs, SFT vs RL, 30B); Fig. 2 (tools and retention); Fig. 4 (GRPO training reward and GAIA-Text-103 validation accuracy, 30B, no values in text).

## Technical Details
**Agent loop and context (§3)**
- At step t the model produces a thought T_t = f_θ(q, H_t) and an action A_t = π_θ(H_t, T_t); the environment returns O_t; the loop stops when no action is produced, and a summary step produces the answer (§3.1, Eq. 1–3).
- Retention: S_t(K) = {i ≤ t−1 : i ≥ t−K}; tool responses outside S_t(K) are replaced by ∅, thoughts and actions are kept (§3.3, Eq. 4–5). K = 5 at evaluation (§6.1).
- The scrape tool uses a lightweight LLM (example: Qwen3-14B) to extract task-relevant text from a URL (§3.2). HuggingFace access is disabled in the tools to prevent answer leakage (§3.2, §6.1).
- Previous open-source agents are described as supporting fewer than 100 tool calls (§1).

**Data (§4)**
- MultiDocQA: corpus from Wikipedia, Common Crawl, and curated web repositories; category-balanced seed sampling; subgraph built by following one random internal link per document recursively; out-of-subgraph links pruned; fact extraction; obfuscation (for example "March 15, 2023 → in the spring of the 2020s", "Paris → a European capital"); an LLM combines constraints into questions (§4.1).
- Trajectory synthesis: ReAct single-agent and MiroFlow multi-agent paradigms; function calling and MCP tool invocation; generators include GPT-OSS, DeepSeek-V3.1, "and other state-of-the-art models" (§4.2).
- Open-source QA converted to trajectories: MuSiQue, HotpotQA, WebWalkerQA-Silver, MegaScience, TaskCraft, QA-Expert-Multi-Hop-V1.0, OneGen-TrainDataset-MultiHopQA, 2WikiMultihopQA, WikiTables, WebShaper, WebDancer, Toucan-1.5M; only QA pairs are kept (§4.3).
- General post-training corpora AM-Thinking-v1-Distilled and Nemotron-Post-Training-Dataset are added "to preserve general conversational abilities" (§4.3).
- Dataset sizes and mixture proportions: not reported.

**Evaluation protocol (§6.1, Table 1 caption)**
- Temperature 1.0, top-p 0.95, maximum 600 turns, 256K context, maximum output 16,384 tokens (§6.1).
- HLE: 2,158-question text-only subset, judged by o3-mini-2025-01-31. GAIA: 103-question text-only subset. GAIA, WebWalkerQA, xBench-DeepSearch, BrowseComp, BrowseComp-ZH judged by gpt-4.1-2025-04-14 (§6.1).
- avg@3 for HLE, BrowseComp, BrowseComp-ZH, WebWalkerQA, FRAMES; avg@8 for GAIA, xbench-DeepSearch, SEAL-0 (Table 1 caption).

**Results (Table 1; HLE / BrowseComp / BC-ZH / GAIA / xbench / WebWalkerQA / FRAMES / SEAL-0)**
- MiroThinker-v1.0-8B: 21.5 / 31.1 / 40.2 / 66.4 / 60.6 / 60.6 / 80.6 / 40.4.
- MiroThinker-v1.0-30B: 33.4 / 41.2 / 47.8 / 73.5 / 70.6 / 61.0 / 85.4 / 46.8.
- MiroThinker-v1.0-72B: 37.7 / 47.1 / 55.6 / 81.9 / 77.8 / 62.1 / 87.1 / 51.0.
- Tongyi-DeepResearch-30B: 32.9 / 43.4 / 46.7 / 70.9 / 75.0 / 72.2 / 90.6 / –. OpenAI-GPT-5-high: 35.2 / 54.9 / 65.0 / 76.4 / 77.8 / – / – / 51.4 (baseline numbers collected from other reports, Table 1 caption).
- The 72B model is below Tongyi-DeepResearch-30B on WebWalkerQA (62.1 vs 72.2) and FRAMES (87.1 vs 90.6), and below GPT-5-high on BrowseComp and BrowseComp-ZH (Table 1).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MiroThinker-v1.0-8B / -30B / -72B | 8B / 30B / 72B | SFT | base checkpoint | Qwen3-8B / Qwen3-30B-A3B-Thinking-2507 / Qwen2.5-72B-Instruct | HF model cards (model table); paper says "Qwen2.5 and Qwen3" (arXiv v3 §5, §6.1) | verified 2026-09-14 | no ablation reported |
| all sizes | — | SFT | loss | −E Σ_t log π_θ(T_t, A_t \| x, H_<t); observations are pre-recorded user turns, not executed | v3 §5.1 Eq. 8 | verified 2026-09-14 | no ablation reported |
| all sizes | — | SFT | data cleaning | filtering and repair of intra-response repetition, cross-response duplication, invalid tool names or arguments; thresholds not given | v3 §5.1 | verified 2026-09-14 | no ablation reported |
| all sizes | — | preference | loss | DPO (β) + λ·SFT loss on preferred trajectories; β and λ values not reported | v3 §5.2 Eq. 10–11 | verified 2026-09-14 (form only) | stability and behavioral consistency cited to refs [46, 47]; no ablation |
| all sizes | — | preference | pair rule | chosen = correct final answer, coherent, explicit planning; rejected = wrong but valid final answer; both filtered for repetition, truncation, malformed structure; no step-count or length heuristics | v3 §5.2 | verified 2026-09-14 | authors state structural heuristics "introduce systematic biases"; no numbers |
| all sizes | — | RL | algorithm | GRPO, fully online: each rollout batch updates the policy exactly once; advantage = R − group mean (Eq. 13, no std division); KL term β_KL·D_KL(π_θ ‖ π_ref), π_ref "typically" the preference checkpoint | v3 §5.3 Eq. 13–14 | verified 2026-09-14 | Fig. 4 training curves (30B), no values |
| all sizes | — | RL | reward | α_c·R_correct − α_f·R_format; LLM grader against ground truth; α values not reported | v3 §5.3 Eq. 12 | verified 2026-09-14 | no ablation reported |
| all sizes | — | RL | trajectory filter | drop correct trajectories with > 5 consecutive network exceptions, repeated identical retries, or excessive timeouts; drop incorrect ones failing on trivial format or showing action loops or premature termination | v3 §5.3 | verified 2026-09-14 | no ablation reported |
| all sizes | — | RL | rollout scheduling | streaming: workers pull prompts until enough finished trajectories exist; unfinished tasks return to the queue | v3 §5.3 | verified 2026-09-14 | no ablation reported |
| all sizes | — | eval-gate | inference | T 1.0, top-p 0.95, 600 turns, 256K context, 16,384 max output tokens, retention K = 5 | v3 §6.1 | verified 2026-09-14 | no ablation reported |
| all sizes | — | SFT / preference / RL | LR, batch, epochs, group size G, max response length, prompt counts, data sizes, compute | not reported | checked v3 body, App. A, 72B/30B/8B HF model cards | not reported | — |

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- Agentic training: RL on the 30B model shifts the turn distribution toward more turns on all four benchmarks in Fig. 5 (histograms and CDFs of turns per task); the accuracy gains are SFT → RL at a fixed model size (§6.3). **Result (single study)**. The paper does not report a sweep of turn budget or retention budget at a fixed checkpoint.
- Limitations reported by the authors: the RL model calls tools more often than the SFT model but "a portion" of calls are redundant; RL lengthens chains of thought into repetitive text; Chinese queries produce mixed-language reasoning; the model misuses the code tool to read web pages and forgets to create a sandbox before use (§6.4).
- Long context: tool outputs older than the last K are masked; thoughts and actions are never masked (§3.3).
- Negative feedback: rejected trajectories enter the DPO term (negative as gradient), anchored by an SFT term on chosen ones (§5.2); GRPO gives below-mean trajectories negative advantage (§5.3). Incorrect trajectories caused by format errors or degenerate loops are removed before the update rather than penalized (§5.3).
- Generality: general post-training corpora are mixed in to keep conversational ability (§4.3); no non-agentic general benchmark is reported.
- Distillation: SFT trajectories come from several teacher models to reduce "single-model biases" (§4.2); per-teacher shares are not reported.

## Connections
- [[tongyi-deepresearch]] — main open research-agent baseline in Table 1.
- [[websailor]], [[webshaper]] — WebShaper data is reused as a QA source (§4.3); both are cited as open deep-research work (§2).
- [[toucan-mcp]] — Toucan-1.5M is one of the converted QA sources (§4.3).
- [[distillation-source-matters]], [[am-thinking-v1]] — origin of AM-Thinking-v1-Distilled used as general data (§4.3, ref [43]).
- [[nemotron-nano-2]] — cited source of Nemotron-Post-Training-Dataset (ref [44]).
- [[grpo]], [[dpo]] — the RL and preference objectives (Eq. 10, 14).
- [[context-folding]], [[resum]] — other context-management methods for long-horizon agents.
- [[kimi-researcher]], [[browsecomp-plus]] — related research-agent training and evaluation.
- [[miromind-m1]] — earlier MiroMind math-reasoning RL report.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2511.11793 (v3, 2026-04-21) including App. A; base checkpoints from the HF model cards miromind-ai/MiroThinker-v1.0-{8B,30B,72B}.
- Audit claims not found in the source: none. All audit numbers (GAIA 81.9, HLE 37.7, BrowseComp 47.1, BrowseComp-ZH 55.6, 256K, 600 tool calls, 72B) match v3.
- Not reported by the source: training hyperparameters, dataset sizes, teacher shares, compute, a controlled interaction-budget sweep, ablation of retention budget K.
