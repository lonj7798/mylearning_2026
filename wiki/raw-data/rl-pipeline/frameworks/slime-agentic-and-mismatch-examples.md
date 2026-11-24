<!-- scope: THUDM/slime at commit 79aa41a: five examples — coding_agent_rl (Codex/Claude Code in Sunabako sandboxes, SWE test reward, fully async), retool (Python-interpreter tool RL on DAPO-Math), fully_async (rollout-function swap and its worker), train_infer_mismatch_helper (TIS/MIS/RS weights via --custom-tis-function-path), delta_weight_sync (disk delta weight updates for disaggregated engines)
     deps: [[slime]], [[async-rollout]]
     see-also: [[slime-search-r1-pipeline]], [[areal-agent-pipelines]], [[verl-rollout-correction-and-lowprec]], [[truncated-importance-sampling]], [[open-instruct-rlvr-pipeline]], [[mai-rl-infrastructure]]
-->

# slime agentic and train–inference-mismatch examples (coding agent, ReTool, fully async, MIS, delta sync)
- **Core Insight:** slime keeps one trainer and expresses each new pipeline as a plug-in: a custom generate function (ReTool, coding agent), a rollout-function swap (fully async), a custom IS-weight function (mismatch helper), or a weight-update mode (delta sync); the coding-agent example combines all of them — fully async rollout, an HTTP adapter that real agent CLIs call, sandbox grading, score centering and token-wise TIS (`coding_agent_rl/run_qwen38_27b_fully_async.sh` L165–188).
- **Guideline:** When a sandbox fails after the agent has produced actions, keep the trajectory and train it with reward 0 rather than retrying until execution succeeds; the coding-agent example does this for `SandboxTerminatedError`, and only aborts (loss mask `[0]`) when no generation happened (`coding_agent_rl/generate.py` L369–392, L446–455).
- **Authors:** slime contributors (THUDM / Z.ai).
- **Year:** 2025–2026; code read at commit 79aa41a8 (2026-11-06).
- **URL:** https://github.com/THUDM/slime/tree/79aa41a884ab0bf65222c94418f6357d3c31192b/examples
- **Source type:** released config/code
- **Relevant topics:** coding-agent RL, SWE rewards, sandbox fleets, tool-integrated reasoning, fully asynchronous rollout, staleness metrics, truncated/masked importance sampling, rejection sampling, delta weight sync

## Summary
Each directory is a launch script plus the Python hooks it names. The [[slime]] card covers the trainer loop, placement and the default weight sync; this card records what each example adds and where. The search-agent example is in [[slime-search-r1-pipeline]].

## Key Contributions
- **coding_agent_rl** — Qwen3.8-27B on the 2,698-task code split of `XiaomiMiMo/MiMo-V2.6-RL-oss`, one row per CLI (Codex, Claude Code) → 5,396 rows; 8 prompts × 8 trajectories per batch, 168 concurrent agent slots, 131,072-token context, agent / grader / whole-rollout timeouts 3,600 / 600 / 7,200 s (`README.md` L14–24, L103–113).
- **retool** — interpreter tool loop up to 16 turns, regex code-safety filter plus subprocess execution, DAPO-math reward with a tool-use bonus on wrong answers (`tool_sandbox.py` L23–31, L112–135; `generate_with_retool.py` L407–432).
- **fully_async** — `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`; a background worker keeps a fixed pool of in-flight groups across rollout boundaries (`fully_async/README.md`).
- **train_infer_mismatch_helper** — `compute_mis_weights_with_cp` with token / sequence / geometric levels, truncate / mask / clip modes, rejection sampling and a per-sequence veto (`mis.py`, `mis.yaml`).
- **delta_weight_sync** — ships only changed bytes per sync as a canonical HF checkpoint directory on a shared filesystem (`delta_weight_sync/README.md`; `docs/en/advanced/delta-weight-sync.md`).

## Key Figures/Tables to Study
- `coding_agent_rl/generate.py` L167–224 (adapter), L254–363 (episode); `coding_agent_rl/swe.py` L290–307 (grader dispatch).
- `retool/generate_with_retool.py` L215–404; `train_infer_mismatch_helper/mis.py` L82–149, L150+; `slime/utils/staleness.py` L10–34.

## Technical Details
**Pipeline at a glance (coding_agent_rl)**
1. Data/images: `prepare_mimo.py` → `train-mixed.jsonl`; `prepare_sunabako_images.py` → compressed OCI layouts + `images.json` (`README.md` L103–135).
2. Rollout function: fully async; custom generate `examples.coding_agent_rl.generate.generate`; DAPO filter `check_reward_nonzero_std` (`run_qwen38_27b_fully_async.sh` L165–168).
3. Adapter: an HTTP app per agent type, served from the rollout process, that turns the CLI's model calls into SGLang token calls and records the trajectory (`generate.py` L167–224).
4. Episode: boot sandbox → `prepare_workspace` → `harness.run(adapter_url, prompt=SWE_PROMPT)` → `git_diff` (`generate.py` L283–298).
5. Grade: a fresh sandbox from the same image applies only the diff and runs the dataset tests; "exit 0 == solved", reward 1.0 iff solved (`swe.py` L290–307).
6. Train: `--use-score-centering --use-tis --tis-clip 2 --tis-clip-low 0`, `--rollout-temperature 1 --rollout-top-p 0.95`, stateless Adam (`run_qwen38_27b_fully_async.sh` L174–175, L188).

**ReTool loop and reward.** Model tokens are appended `trainable=True` with log-probs; `<interpreter>…</interpreter>` observation tokens `trainable=False` (`generate_with_retool.py` L327, L354). Reward:
```python
# examples/retool/generate_with_retool.py L422-428
result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
# encourage model to call tools
if result["score"] < 0:
    tool_call_reward = (num_turns - 2) / 2 * 0.1
    result["score"] = min(-0.6, result["score"] + tool_call_reward)
```
`math_dapo_compute_score` returns 1.0 / −1.0 (`slime/rollout/rm_hub/math_dapo_utils.py` L285), so a wrong answer scores between −1.0 and −0.6 depending on tool-call count. Sandbox limits: 120 s and 4 GB per Python process (`tool_sandbox.py` L27–28); `import os`, `open(`, `exec(` and similar patterns are rejected before execution (L115–135). Launch: 32 prompts × 8 samples, 131,072 max response, `--colocate` (`retool_qwen3_4b_rl.sh` L46–48, L127–128, L149).

**Fully async staleness.** No hard staleness bound was found in `slime/` at this commit; concurrency is `sglang_server_concurrency × engines` (README). Staleness is measured per sample in weight-update units, `current_weight_version − min(sample.weight_versions)`, and logged as `staleness/mean` and `staleness/max` (`slime/utils/staleness.py` L10–34). `--buffer-sort-by-staleness` resumes the oldest buffered groups first (`arguments.py` L579–586); `--mask-offpolicy-in-partial-rollout` excludes continued-prefix tokens from the loss (README). Compare AReaL's admission bound in [[areal-agent-pipelines]].

**Mismatch helper (`mis.yaml` defaults).** `use_tis: true`, `use_rs: true`, `tis_level/rs_level: token`, `tis_mode: truncate`, bounds 0.5 / 2.0, `rs_veto_threshold: 1.0e-4`, `tis_batch_normalize: true`. Enabled with `--use-tis --custom-config-path …/mis.yaml --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp` (`run-qwen3-4b-mis.sh` L90, L129–130). README modes: standard PPO (old log-prob from Megatron), bypass (`--use-rollout-logprobs`, SGLang log-prob as old policy), decoupled 3-policy PPO.
```python
# examples/train_infer_mismatch_helper/mis.py L92-96, L113
log_veto_threshold = torch.log(torch.tensor(veto_threshold, device=log_ratio.device))
# For each sequence, if it has any catastrophic tokens, return 0 for the sequence
catastrophic_tokens = (log_ratio < log_veto_threshold) & loss_mask.bool()
has_catastrophic = catastrophic_tokens.any()
veto_mask = (~has_catastrophic).float().expand_as(log_ratio)
...
return weights.clamp(0, upper_bound) * loss_mask   # truncate (TIS)
```

**Delta weight sync.** Flags: `--update-weight-mode delta --update-weight-transport disk --update-weight-disk-dir <shared> --update-weight-local-checkpoint-dir <local NVMe> --update-weight-delta-encoding xor|overwrite --update-weight-delta-checksum xxh3-128|blake3|adler32`. Sequence: seed with a full HF checkpoint; each later sync diffs gathered HF tensors against a CPU snapshot, zstd-compresses (level 1), writes `weight_v{N:06d}/`; the trainer calls each engine's `/pull_weights`, every host applies the delta in place and verifies per-tensor checksums, then the engine reloads via `update_weights_from_disk` (`delta-weight-sync.md` L1–50). Disk transport only; example `run-glm4.7-30B-A3B-delta.sh` on 2 nodes / 16 GPUs, non-colocated.

## Verification
- Checked on 2026-11-24 against the local clone at 79aa41a8. All loci refer to that commit.
- `mis.py` `calculate_veto_mask` calls `.any()` over the whole tensor it receives; per-sequence behaviour depends on callers passing one sequence at a time (the docstring at L168 says "1D tensor each"); caller not traced.
- Not found: measured speedups or accuracy for any of these examples in their READMEs; a byte-reduction figure for delta sync.
- Sunabako sandbox internals and the agent CLIs are external dependencies (`requirements-sunabako.txt`); not read.
