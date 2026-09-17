---
chapter: ch-29d
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/**/together-coderforge-agent-trajectories.md on 2026-09-15)
source_url: https://www.together.ai/blog/coderforge-preview
source_version: Together AI blog, published 2026-02-25; Hugging Face dataset card of CoderForge-Preview (fetched 2026-09-14)
source_type: official blog with dataset card
created_at: "2026-09-15"
---

# Excerpt: CoderForge-Preview: SOTA open dataset for training efficient coding agents (Ariyak, Zhang, Wang, et al.; Together AI)

Facts used by [[read]] §3, §7, and the Recipe table. The blog's tables are images and are not in the cached text; only numbers stated in the prose or in the dataset card are used.

## Generation (blog "CoderForge-Preview Data")
- Task sources: R2E-Gym, SWE-Smith, SWE-Rebench. Scaffold: OpenHands v0.52.1 inside the R2E-Gym framework, tools execute_bash, str_replace_editor, think, finish; up to 100 steps per task in an isolated Docker container.
- Teacher: Qwen3-Coder-480B; temperature 0.7; top_p 0.8; 32,768 max new tokens. Trajectories per problem: 8 (R2E-Gym, SWE-Rebench), 4 (SWE-Smith).
- Filter: "We filter to keep only trajectories whose final patches pass all repository tests."
- Decontamination: "we exclude any tasks that share the same (repository, base_commit) pair or problem statement with SWE-Bench Verified samples."
- Totals: 258,134 trajectories, 155,144 successful, over 51K tasks and 1,655 repositories. Dataset card split sizes: SWE_Rebench 77,169; SWE_Smith 148,001; R2E_Gym 32,964; filtered_reward1 155,144. Each record keeps a reward field.
- Solve rate by attempts: R2E-Gym 62.9% (Pass@1) → 80.3% (Pass@8); SWE-Rebench 57.5% → 73.9% (Pass@8); SWE-Smith 58.8% → 64.9% (Pass@4).
- Cost: 15.64M API completions, 452B prompt tokens, 2.91B output tokens, about 90% cache hit rate, $130k total under the stated pricing.
- Length: median 36K-42K tokens, average about 41K, P99 approaching 100K; 6.70B total training tokens.

## Failed trajectories
- "failed trajectories require substantially more steps, 18–28% more on average than successful ones. By training exclusively on successful trajectories, we aim to push the model toward efficient task resolution and concise decision-making, rather than learning from extended unproductive sequences."
- The failed trajectories are released but not used for SFT.

## Training and results
- Qwen3-32B; 128K context; Ulysses sequence parallelism; multipack with boundary-aware masking; token-level loss normalized by the total valid token count across ranks; 64 H100 GPUs; LR 1e-5 with cosine schedule (stated under Limitations as the fixed configuration).
- SWE-Bench Verified: 59.4% pass@1 and 78.56% pass@16 at epoch 3.13. The dataset card prints "23.0% → 59.4% pass@1"; the blog prose says "boost SWE-Bench Verified performance 23.0% above the base model reaching 59.4%". The two phrasings disagree; the card's arrow reads as a base score of 23.0%.
- Qwen3-4B reaches 43.0% at epoch 5.

## Limitations stated
- "we generate all data with a single scaffold and set of tools without permutations, so models trained with SFT on this data might perform worse when used with different scaffolds, tools, and prompt templates."
- Data "mostly focus on fixing bugs"; user messages during the trajectory are missing; evaluation mainly on SWE-Bench Verified.
