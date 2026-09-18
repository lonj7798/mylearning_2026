<!-- scope: Olmo 3 technical report — Ai2's fully open 7B/32B model flow from pretraining through OlmoRL
     deps: [[dolma]], [[tulu-3]]
     see-also: [[olmo-2]], [[qwen-3]], [[allen-ai]], [[allenai-olmo3-open-instruct-scripts]], [[olmo-3-recipe]]
-->

# Olmo 3
- **Core Insight:** Olmo 3 releases the whole model flow — every stage, checkpoint, data point, and dependency — for a 7B and 32B family built in three base stages (5.93T-token pretraining, 100B-token midtraining, 50B or 100B-token long-context extension to 65,536 tokens) and three post-training branches (Think, Instruct, RL Zero), with Olmo 3.1 Think 32B reported as the strongest fully-open thinking model at release (Abstract; §2.1, §3.6).
- **Guideline:** When extending context after midtraining, train on a mix of roughly one-third long documents and two-thirds midtraining data rather than long documents alone, because that is the composition Ai2 used for Dolma 3 Longmino Mix (§3.6.3). When running RLVR at 32B, expect inference to dominate: the report measures about 14× as much compute spent on generation as on training (§4.4.3).
- **Authors:** Olmo Team — Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, et al. (Allen Institute for AI; University of Washington; Carnegie Mellon University; and others)
- **Year:** 2025 (arXiv v1 2025-12; v2 dated 2026-04-14)
- **URL:** https://arxiv.org/abs/2512.13961
- **Source type:** official technical report
- **Relevant topics:** fully open model flow, midtraining, long-context extension, sliding-window attention, delta learning, OlmoRL, function calling, RL infrastructure

## Abstract
Olmo 3 is a family of fully open models at 7B and 32B. Construction targets long-context reasoning, function
calling, coding, instruction following, general chat, and knowledge recall. The release covers the entire model
flow: every stage, checkpoint, data point, and dependency used to build the family. The flagship model is
Olmo 3.1 Think 32B, described as the strongest fully-open thinking model released to date (Abstract).

## Key Contributions
- Releases four branches from one base: Base, Think, Instruct, and RL Zero, with 3.1 refreshes of the 32B Think
  and Instruct checkpoints trained after the initial release (title page; §2, §4.4).
- Dolma 3 data family: Dolma 3 Mix for pretraining, Dolma 3 Dolmino Mix for midtraining, Dolma 3 Longmino Mix
  for long-context extension, and the Dolci suites for SFT, DPO, and RL (§3.4-§3.6, §4, §5).
- Two token-selection methods for the pretraining mix — token-constrained mixing and quality-aware upsampling —
  plus trillion-token global deduplication tooling (§3.4).
- olmOCR-converted science PDFs as a pretraining source: 972B tokens in the pool, 805B in the 6T mix (§3.4.2,
  Table 4).
- OlmoRL: a GRPO-based RL stage with zero-gradient-signal filtering, active sampling, token-level loss, no KL
  loss, clip-higher, truncated importance sampling, and no standard-deviation normalization of advantages
  (§4.4.1).
- RL infrastructure with continuous batching and in-flight weight updates, reported as up to 4× faster throughput
  at the same resources without accuracy loss (§4.4.3).

## Key Figures/Tables to Study
- **Table 4** — Dolma 3 Mix: 9.31T-token pool, 5.93T-token training mix, 157B experimentation mix.
- **Table 5** — Dolma 3 Dolmino Mix: 2.19T pool against the 99.95B midtraining mix, by source.
- **Table 11** — Dolma 3 Longmino Mix: 639B pool against the 50B mix used for the 7B model.
- **Table 17, 19, 20** — Dolci Think SFT, DPO, and RL prompt counts by category.
- **Table 27, 30** — function-calling datasets and the Dolci Instruct SFT/DPO mix.

## Technical Details
- Pretraining mix: 5.93T tokens, 76.1% Common Crawl, 13.6% olmOCR science PDFs, 6.89% Stack-Edu code, 2.56%
  FineMath 3+, 0.86% arXiv, 0.04% Wikipedia (Table 4).
- Midtraining mix: 99.95B tokens sampled from a 2.19T pool; the largest shares are Common Crawl high-quality
  subset 22.5%, Dolmino Math 10.7%, StackEdu FIM 10.0%, CraneCode 10.0% (Table 5).
- Long-context: window raised from 8,192 to 65,536 tokens; Longmino pool is 639B tokens, the mix is 50.0B for the
  7B model and 100B for the 32B; composition is 34% long-context data and 66% high-quality midtraining data
  (§3.6, §3.6.3, Table 11).
- Architecture: sliding-window attention with window 4096 on three of every four layers; the last layer is always full attention (§3.2).
- Pretraining throughput: 7,700 tokens/s/GPU at 7B and 1,960 at 32B, at sequence length 8192 in bfloat16,
  about 43% and 41% MFU (§3.2).
- Dolci Think SFT holds 2,268,468 prompts for the 7B model and 2,253,916 for the 32B; the largest single source
  is Dolci Think OpenThoughts 3+ Math at 752,997 prompts (Table 17).
- Dolci Think DPO holds 200,000 prompts (Table 19); Dolci-Think-RL holds 104,869 prompts for 7B and 171,950 for
  32B (Table 20).
- Dolci Instruct SFT holds 2,152,112 prompts and Dolci Instruct DPO 259,922 (Table 30).
- Cost: about 56 days elapsed from start of training to evaluation of Olmo 3 Think 32B on a cluster of 1,024 H100
  GPUs, which the report prices at $2.75M at $2 per H100-hour; roughly 47 days pretraining including midtraining
  and long-context, and about 9 days post-training (§2.4).

## Recipe ledger
The recipe table is in [[olmo-3-recipe]] (pretraining, midtraining, long-context, SFT, DPO, and OlmoRL rows with loci).

## Findings relevant to agentic training, long context, and distillation
- Function calling is an Instruct-branch capability: Dolci Instruct SFT contains 227,579 tool-use prompts out of
  2,152,112 total, about 10.6% of the mix, and no tool-use category appears in the Think SFT mix (Table 17,
  Table 30).
- The function-calling data is three datasets: Science QA with 22.6K trajectories over 8 unique functions through
  real MCP servers, Web Search QA with 6.6K trajectories over 3 functions, and SimFC with 200K simulated
  trajectories over 42.6K unique functions, 42.3% multi-turn and 23.8% multi-step (Table 27). Trajectories were
  generated by prompting GPT-4o, GPT-4.1, and GPT-5 to produce user queries, environment responses, and assistant
  messages (§5.2).
- A single tool-definition and tool-calling format is used across all tool-use data; the report states that
  unifying the format was necessary for stable tool-use behaviour (§5.2). Status: Result (single study), no
  ablation table given.
- Function calling is evaluated with BFCLv3 for intrinsic accuracy and MCP-based task completion for extrinsic accuracy; the tool-use contribution is the delta against answering without tools (§5.2).
- Distillation: reasoning traces in Dolci Think SFT are generated by other models, and Dolci DPO pairs are built
  from Qwen3 32B as the chosen generator and Qwen3 0.6B as the rejected generator (§4.3.1, §5.3.1).
- Negative feedback: OlmoRL discards groups whose rewards are all identical, because their advantage has zero
  standard deviation and contributes no gradient, and refills the batch by active sampling (§4.4.1, §4.4.3).

## Connections
- [[olmo-2]] — the predecessor whose Dolmino midtraining and Tülu 3 post-training this report extends.
- [[tulu-3]] — the SFT → DPO → RL structure Olmo 3 inherits before replacing PPO with OlmoRL.
- [[allenai-olmo3-open-instruct-scripts]] — the released post-training scripts with the per-run hyperparameters.
- [[qwen-3]] and [[deepseek-v3]] — the open-weight comparisons used in the base and Think result tables.
- [[allen-ai]] — lab-level summary of the group that produced the report.
- [[longalign]] — long-context SFT evidence read against the Longmino extension stage.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2512.13961 (arXiv v2, 14 Apr 2026).
- Corrections to the previous card version:
  - "Authors: Team Olmo" → "Olmo Team", authors sorted alphabetically; first six listed above.
  - "Dolma 3: about 9.3T source tokens" → 9.31T is the pool total in Table 4; the corpus name for the pool is
    Dolma 3, and only 5.93T tokens were used for training.
  - "Dolmino: 100B mid-training tokens sampled from a ~2.2T high-quality pool" → the pool is 2.19T and the mix is
    99.95B (Table 5).
  - "Longmino: about 50B long-context tokens from a 639B-token pool" → 50.0B applies to the 7B model; the 32B
    model used 100B at the same proportions (Table 11 caption, §3.6).
  - "Post-training: SFT -> DPO -> RLVR" without qualification → the RL stage is OlmoRL, built on GRPO with
    modifications from DAPO and Dr GRPO, not the PPO-based RLVR of OLMo 2 (§4.4.1).
  - "Mid-training used 128 H100 GPUs. Post-training used 256 H100 GPUs." → midtraining ran as two parallel runs
    on 512 GPUs each; 256 GPUs is the per-run count for the 32B SFT learning-rate sweep, and DPO used 64 GPUs per
    job (§2.4).
  - "In-flight weight updates, continuous batching, and threading work made RL training about 4x more efficient"
    → the report attributes "up to 4x faster" throughput to in-flight updates specifically (§4.4.3), and states a
    "4x speedup in RL training" for the combined OlmoRL changes (§1).
  - "Pretraining used up to 1,024 H100 GPUs" → correct, and the report adds that the initial pretraining phase ran
    9.5 days on 512 GPUs before moving to 1,024 (§2.4).
- Removed as unsupported by the source: "Dolma 3 Mix: about 5.9T (~6T) pretraining tokens with stronger math/code
  emphasis and stronger decontamination" as a comparison to an unnamed baseline; "OLMo 3 is much more explicit
  about intermediate stages" than Qwen 3 and DeepSeek V3; "It is one of the clearest public examples that openness
  can apply to training trajectories"; "for a learner, it is unusually valuable because you can study where a
  capability was added"; "the real scientific artifact is not just the final model weights"; the tooling list
  entry "OlmoTrace" as part of this release (it is a separate Ai2 system, not described in this report).
- Not reported by the source: per-stage GPU-hours separated from the 56-day wall-clock figure; peak learning rates
  and batch sizes in the body (they are in the released configs, see [[allenai-olmo3-open-instruct-scripts]]).
