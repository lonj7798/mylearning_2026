<!-- scope: SkyRL-Agent (arXiv 2511.16108, Nov 2025) — tool-centric agent loop, asynchronous pipeline dispatcher, transition-based backend bridge; SA-SWE-32B pure-RL recipe on 4.5K R2E-Gym tasks with out-of-domain agent evaluations; deep-research, memory, and computer-use case studies
     deps: [[rloo]], [[dr-grpo]]
     see-also: [[skyrl-agent-recipe]], [[deepswe]], [[r2e-gym]], [[agent-lightning]], [[async-rollout]], [[swe-gym]]
-->

# SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent
- **Core Insight:** RL alone, starting from Qwen3-32B (24.4% Pass@1), on 4.5K R2E-Gym instances produced SA-SWE-32B at 39.4% Pass@1 on SWE-Bench Verified using 4,601 H100 hours; DeepSWE, trained with RL on the same 4.5K instances, scores 36.4% under the same simple ReAct evaluation and used 9,180 H100 hours (§4.3, Table 2).
- **Guideline:** When end-to-end RL on long-horizon SWE tasks starts from a model whose rollouts mostly fail, improve the tools before training (here an AST-based search tool that appends hints to its results), because a bash-only setup left 50 of 64 R2E-Gym tasks unresolved, while the tool-enhanced setup matched or exceeded similar-size models within 125 RL steps (§4.2, Fig. 1a).
- **Authors:** Shiyi Cao, Dacheng Li, Fangzhou Zhao, Shuo Yuan, Sumanth R. Hegde, Connor Chen, et al. (NovaSky AI, UC Berkeley, Anyscale)
- **Year:** 2025 (arXiv v1 2025-11; the paper header reads "Work in progress")
- **URL:** https://arxiv.org/abs/2511.16108 (code: https://github.com/NovaSky-AI/SkyRL ; model: https://huggingface.co/NovaSky-AI/SA-SWE-32B)
- **Source type:** paper
- **Relevant topics:** multi-turn agent RL, rollout scheduling, asynchronous dispatch, transition-based trajectories, SWE agents, tool design, leave-one-out advantage, out-of-domain agent generalization, context management
- **Recipe ledger:** [[skyrl-agent-recipe]] (`papers/skyrl-agent-recipe.md`)

## Abstract
The paper presents SkyRL-Agent, a framework for training and evaluating multi-turn, long-horizon agents. It provides asynchronous dispatching, tool integration, and interoperability with the RL training backends SkyRL-train, VeRL, and Tinker. With it, the authors train SA-SWE-32B from Qwen3-32B using RL only. Two components drive the efficiency gain: an asynchronous pipeline dispatcher with a 1.55× speedup over naive asynchronous batching, and a training recipe built around an AST-based code search tool that raises rollout Pass@K. SA-SWE-32B reaches 39.4% Pass@1 on SWE-Bench Verified at more than 2× lower cost than prior models with similar results. Although trained only on SWE tasks, it improves on Terminal-Bench, BrowseComp-Plus, and WebArena. Case studies train deep-research, computer-use, and memory agents, each on a different backend.

## Key Contributions
- Tool-centric agent loop: the agent acts only through OpenAI-style function calls; each tool defines its execution logic and runtime; a Gym environment is added by wrapping `step()` as a tool; each dataset binds its own tools and verifiers, so one training job can mix tasks (§3.1, Listing 1).
- Dispatcher: each rollout is split into runtime initialization, agent run, and reward calculation, with bounded queues per stage. Strategies: Async Batch, Async Batch (Bounded), and Async Pipeline, which overlaps CPU-bound stages with GPU generation (§3.2, Fig. 3, Listing 2).
- Transition-based backend bridge: every LLM call is recorded with input tokens, output tokens, and log probabilities, then converted to a backend-agnostic format (§3.3, Listing 3).
- Error handling: terminal conditions end the episode; recoverable conditions inject corrective feedback into the history (§3.4).
- SA-SWE-32B recipe, out-of-domain evaluations, and three smaller agents (§4, §5).

## Key Figures/Tables to Study
- Fig. 1a (reward, non-resolved count, average turns, and average search calls vs step, with DeepSWE stage-1 curves) and Fig. 1b (GPU utilization, Async Pipeline vs Async Batch (Bounded)).
- Fig. 2 (architecture) and Fig. 3 (dispatch timelines); Table 1 (framework comparison).
- Table 2 (SWE-Bench Verified with cost) and Table 3 (out-of-domain benchmarks).
- Figs. 5–7 (deep research, memory agent, computer use training curves).

## Technical Details
**Multi-turn formulation (§2).** At step t, the agent observes context o_t, samples action a_t ~ π_θ(a_t | o_t), and receives scalar reward r_t. The objective is J(θ) = E_πθ[Σ_{t=1}^{T} r_t], where T is the number of agent turns per task. A transition is the tuple (o_t, a_t, r_t). Mask-based training concatenates the whole interaction and masks non-model tokens; the paper states that summarizing or truncating context between turns breaks this single-sequence assumption, so mask-based construction is hard to apply beyond short tool-integrated reasoning tasks.

**Transitions (§3.3).** Recorded log probabilities allow corrections for inference–training engine mismatch (Flash-RL). Token-in/token-out recording avoids off-policy drift from re-tokenization. Transitions sharing a prefix are packed into one sample with masks; without context modification, this reduces to concatenation and masking.

**Scheduling result (§3.2, Fig. 1b).** In SWE training, Async Pipeline is about 1.55× faster than Async Batch (Bounded), and GPU utilization stays around 90% during generation. Measured at batch size 64 with 8 rollouts (512 trajectories) on 2×8 H100 GPUs (Fig. 1b caption).

**SWE tools and training (§4.2).** Observation: agents over-rely on viewing files and write imprecise search queries. The AST-based search tool (inspired by LocAgent) supports fuzzy and structural pattern search and appends hints for next queries. With the search tool removed at evaluation, the model reaches "comparable performance" (no number given). RL settings: fully on-policy (batch size = mini-batch size = 64), 8 rollouts per task. Trajectories stopped by the 32K-token context limit or the 50-turn step limit are excluded from the gradient, while reward and advantage estimation stay unchanged. Advantages are leave-one-out, without standard-deviation or length normalization. KL and entropy losses are disabled, and the LR is 1e-6. The Qwen3 chat template is modified to keep earlier turns' thinking. Hints during training cover tool failures, remaining step or context budget, invalid function calls, and failed edits. The checkpoint at step 125 is evaluated (Fig. 1a caption). During training, average search calls rise from 3 to 4 and average turns from 18 to 25 (§4.3).

**SWE-Bench Verified (§4.3, Table 2).** Simple ReAct scaffold with only a bash tool and a file editor, 40K max context, 100 max steps, one patch per instance. The caption warns that reported scores from other scaffolds are not strictly comparable.

| Model | Size | Recipe | Simple ReAct | Reported | Cost (H100 h) |
|---|---|---|---|---|---|
| Qwen3-32B | 32B | – | 24.4 | – | – |
| Qwen3-Coder-30B | 30B | – | 45.0 | – | – |
| SWE-agent-LM-32B | 32B | 3.7 Sonnet distill | 38 | 40.2 | – |
| SWE-Swiss | 32B | R1 distill + RL | × | 45.0 | – |
| Kimi-dev | 72B | R1 distill + RL | × | 48.6 | – |
| DeepSWE | 32B | RL | 36.4 | 42.2 | 9180 |
| SA-SWE-32B | 32B | RL | 39.4 | – | 4601 |

**Out-of-domain evaluation (§4.3, Table 3).** Terminal-Bench v0.1.1 (80 tasks, OpenHands agent); BrowseComp-Plus (830 questions, Qwen3-Embedding-8B retriever, top-5 results); WebArena (812 tasks).

| Model | Terminal-Bench | BrowseComp-Plus acc. | BrowseComp-Plus avg. turns | WebArena |
|---|---|---|---|---|
| Qwen3-32B | 13.75 | 18.1 | 3.68 | 15.8 |
| SA-SWE-32B | 16.25 | 19.4 | 4.6 | 17.0 |

**Case studies (§5; settings in [[skyrl-agent-recipe]]).** Deep research: Qwen3-8B trained with GRPO on SkyRL-train improves HLE-500 from 12.6% to 18.8% under the general verifier and from 9.2% to 10.2% / 11.0% under a gpt-oss-20b judge (§5.1). Memory agent: MemAgent scaffold with a `Next` tool on RULER-HotpotQA, trained with Tinker LoRA; the result is shown only as a curve (Fig. 6b) (§5.2). Computer use: Qwen3-8B with GRPO on 32 OSWorld tasks on VeRL (§5.3).

## Findings relevant to generality, negative feedback, long context, agentic training
**Generality.** SWE-only RL raised all three out-of-domain scores (Table 3); each model is evaluated once, with no variance reported. The authors posit that tool-use competence from SWE transfers to other stateful tool tasks (§4.1, Interpretation). On BrowseComp-Plus, the trained model makes more search calls than the base model (§4.3). For the computer-use agent, training reward rose while validation accuracy showed "little to no gain", and the training tasks are a subset of the OSWorld benchmark (§5.3). Kimi-dev and SWE-Swiss, trained with the Agentless workflow scaffold, "struggle to follow tool-call instructions" in ReAct, so only their reported scores are listed (§4.3). In deep research, online search on GPQA-Diamond sometimes retrieved answers from public pages; the authors block domains such as Hugging Face, GitHub, GitLab, and Chegg (§5.1).

**Negative feedback (course standard §6.1 terms).** Failed trajectories receive below-baseline leave-one-out advantages, which is negative as gradient (§4.2). Trajectories cut off by context or step limits are masked from the gradient, not penalized (§4.2). Recoverable errors and training hints are placed in the context as corrective feedback, which is negative as content (§3.4, §4.2). Tool-serving timeouts contaminated rollouts near deep-research iteration 30, and recovery took steps 30–33; the authors cite this as a reason to mask abnormal trajectories (§5.1).

**Long context and context management.** SWE training uses a 32K context limit and evaluation uses 40K (§4.2, §4.3). The memory agent reads 4K-token chunks, trains on inputs up to 28K tokens, and is evaluated up to 112K (§5.2). Transition-level data allows the agent to summarize or rewrite its own context between turns (§3.3).

## Connections
- [[deepswe]] — RL baseline on the same 4.5K R2E-Gym instances (Table 2); [[r2e-gym]] — training environments.
- [[swe-gym]], [[kimi-dev]] — prior SWE-agent work cited for brittle tool-use behaviour; [[swe-smith]] — data behind SWE-agent-LM-32B.
- [[rloo]], [[dr-grpo]] — leave-one-out baseline and removal of std / length normalization (cited as Liu et al. 2025b).
- [[agent-lightning]] — transition-based framework compared in Table 1; [[async-rollout]], [[areal-async-rl]] — other asynchronous RL systems.
- [[rollout-training-mismatch-tis]] — inference–training mismatch that recorded log probabilities address (§3.3).
- [[terminal-bench-2]] (the paper uses Terminal-Bench v0.1.1), [[browsecomp-plus]], [[webarena-data]] — out-of-domain benchmarks.
- [[memagent]] — memory-agent scaffold; [[search-r1]], [[asearcher]] — search-agent RL work cited by the paper (§1, §3.2, §5.1); [[guru-cross-domain-rl]] — cited for data difficulty and diversity balance.
- [[grpo]] — algorithm for the deep-research and computer-use agents; [[novasky-sky-t1-7b-distill-rl-loop]] — earlier NovaSky RL work.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2511.16108 (v1, 2025-11-20; the PDF fetched on 2026-09-14 carries the v1 stamp); released example config SkyRL@58891b2 `skyrl-agent/examples/run_skyrl/run_skyrl_swe.sh` and `skyrl_swe.yaml` read for the recipe ledger. The HF model card NovaSky-AI/SA-SWE-32B is an unfilled template.
- Audit claims not found in the source: "BrowseComp-Plus 3.68 to 4.6" is mislabeled; those values are average turns, and accuracy is 18.1 → 19.4 (Table 3). "Negatives: no special handling beyond leave-one-out group baselines" is incomplete: truncated trajectories are masked from the gradient (§4.2), and recoverable errors are converted into corrective feedback (§3.4).
- Not reported by the source: rollout temperature, clip ratios, loss aggregation, RL epochs, and the commit or script that produced SA-SWE-32B (see the released-config rows in [[skyrl-agent-recipe]]); a numeric result for the search-tool ablation at evaluation.
