<!-- scope: collection checklist for the rl-pipeline source library; snapshot 2026-11-24
     see-also: [[mai-rl-infrastructure]], [[mai-thinking-1]]
-->

# rl-pipeline collection plan

Status legend: `[x]` card exists and was read from the source; `[ ]` not yet written or source unavailable. Outline: `wiki/courses/rl-pipeline/outline.json`.

## Infrastructure and numerics (infra pass, 2026-11-24)
- [x] [[mai-rl-infrastructure]] — blogs/ (anchor; PDF cross-checked, Table 1 added)
- [x] [[mai-thinking-1]] — model-reports/ (Sec. 3 RL recipe + Sec. 3.6 Rocket v1)
- [x] [[thinking-machines-nondeterminism]] — blogs/
- [x] [[flash-attention-stability]] — papers/ (arXiv 2405.02803)
- [x] [[nemo-rl-tp-accuracy]] — frameworks/
- [x] [[deepseek-train-infer-consistency]] — papers/ (arXiv 2606.19348 = DeepSeek-V4 report; consistency sections only)
- [x] [[truncated-importance-sampling]] — papers/ (Notion blog, read via Notion page-chunk API)
- [x] [[fp16-train-infer-mismatch]] — papers/ (arXiv 2510.26788)

No source in this pass was unavailable.

## Domain pipelines (second agent)
Status at the time this file was written; files not present then are marked in progress.
- [x] [[search-r1]] — papers/
- [x] [[asearcher]] — papers/
- [x] [[deepresearcher]] — papers/
- [x] [[websailor-tongyi-deepresearch]] — papers/
- [x] [[minimax-m1]] — model-reports/
- [ ] [[recipe-crosswalk]] — (in progress by second agent)
- Also present, written outside this pass: [[big-math]], [[deepmath-103k]], [[reasoning-gym]], [[swe-smith]] (papers/).

## Reused from wiki/raw-data/llm-training (link, do not duplicate)
[[slime]], [[verl-rollout]], [[areal-async-rl]], [[async-rollout]], [[grpo]], [[deepseek-open-infra]], [[rstar2-agent]], [[kimi-k2-agentic-data]], [[deepseek-v4-recipe]].

## Outline-cited slugs with no card in either library (snapshot)
- [ ] ch-09: [[olmo-3-rl-pipeline]]
- [ ] ch-10: [[scalerl]], [[pipelinerl]], [[polaris]], [[skywork-or1]], [[brorl]], [[dapo]]
- [ ] ch-11: [[math-verify]], [[verifier-pitfalls]], [[master-rm-one-token]], [[deepseek-grm]], [[general-reasoner-verifier]], [[rlcf-checklists]]
- [ ] ch-13: [[diversity-axes]], [[agentscaler-environment-scaling]], [[tau-bench]]
- [ ] ch-18: [[algorithm-crosswalk]]
- [ ] ch-19: [[rubicon-rubric-anchors]], [[writing-zero]], [[rlmt-think-chat]], [[skywork-reward-v2]], [[litbench]]
- [ ] ch-23: [[human-like-responses-dpo]], [[antislop]], [[open-character-training]], [[speechjudge]]

## Possible additions (not yet in the outline)
- [ ] NeurIPS 2025 version of TIS, "On the Rollout-Training Mismatch in Modern RL Systems" (extends [[truncated-importance-sampling]]).
- [ ] "When Speed Kills Stability" training–inference mismatch note (cited by [[nemo-rl-tp-accuracy]]).
