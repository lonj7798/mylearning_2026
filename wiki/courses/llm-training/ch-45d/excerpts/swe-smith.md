---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: arXiv:2504.21798 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2504.21798
created_at: "2026-09-15"
---

# Excerpt: SWE-smith: Scaling Data for Software Engineering Agents

- **Authors:** John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, Diyi Yang
- **Year:** 2025 (arXiv v1 2025-04-30)
- **Source type:** paper
- **Used in:** [[read]] §3, §5, Recipe, Negative feedback

## Method (§2)
- One Docker image per repository instead of one per task instance, so "storage 295 GBs" covers 50k instances across 128 Python repositories (Table 2). Task instances are produced by breaking working code inside those repositories.
- Five bug-generation strategies with their yields and per-instance costs (Table 1, totals row): 50,137 instances, average validity 50.1%, average cost 2.32 cents per instance. Procedural AST modification gives 40.2% yield at 15,641 instances and zero model cost.
- Execution validation keeps a candidate only when it breaks at least one previously passing test (Fail-to-Pass).
- Cost and storage (§2.2, Table 2): "SWE-smith took $1360 to create ($1000 to generate bugs, $160 for automatic repository installation with SWE-agent, $200 to generate issues for 10K bugs)"; "Creating SWE-smith took one author ∼ 20h of human labor"; environment size 295 GB against 4 TB for the R2E-Gym subset (4.6k tasks) and 6 TB for SWE-gym (2.4k tasks).
- Issue text is written by an LM from the diff, a F2P test and its output.

## Trajectories and training (§4)
- Expert trajectories were attempted "for 8,686 unique task instances, or 17.3% of the SWE-smith dataset. ... The final pool of 6,457 represents a 36% resolve rate of all 17,906 attempts".
- Filtering: "we also observe that 'easier' trajectories – task instances that are repeatedly solved across multiple runs — degrade model performance. Therefore, we limit the number of times any SWE-smith task instance is represented in the training set to 3 trajectories. This leads to the final 5,016 training set."
- Result (Table 3): SWE-agent-LM-32B, fine-tuned from Qwen 2.5 Coder Instruct 32B on 5,016 trajectories, reaches 40.2% on SWE-bench Verified and 30.7% on SWE-bench Lite, pass@1, with the SWE-agent scaffold. SWE-agent-LM-7B reaches 15.2% / 11.7%. Baselines in the same table include SWE-gym-32B at 20.6% Verified (491 trajectories) and R2E-Gym-32B at 34.4% (3.3k).
- Matched-size comparison: fine-tuning the 32B model on 500 successful trajectories gives "a 28.2% resolve rate on SWE-bench Verified, a relative difference of +8.2% with Pan et al. (2024) and +0.7% with Jain et al. (2025)".

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2504.21798: Abstract, §1, §2.2, §3, §4, §4.1, Tables 1-3.
- Note: [[deepswe]] §6 reports that RL on SWE-Smith and SWE-Gym data "gave limited improvement with a high solve-none rate" in its own setting, which is a different use (RL rather than SFT).
- Not reported by the source: RL use of the dataset; non-Python repositories in the main 128-repo set (nine additional languages appear in §F.2); results above 32B.
