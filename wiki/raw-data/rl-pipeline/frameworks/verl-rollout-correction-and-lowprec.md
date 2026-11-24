<!-- scope: verl-project/verl at commit 8357f1e: Rollout Correction (algorithm.rollout_correction — token/sequence IS with truncation or IcePop masking, divergence-based rejection sampling, decoupled vs bypass modes, score centering) and the examples/rollout_correction + examples/mxfp8 launch scripts with the results reported in docs/low_precision/mxfp8.md
     deps: [[verl-rollout]], [[truncated-importance-sampling]]
     see-also: [[fp16-train-infer-mismatch]], [[slime-agentic-and-mismatch-examples]], [[open-instruct-rlvr-pipeline]], [[mai-rl-infrastructure]], [[grpo]]
-->

# verl Rollout Correction (IS / rejection for train–inference mismatch) and MXFP8 end-to-end RL
- **Core Insight:** verl treats the sampler as a separate behaviour policy π_rollout and corrects for it with two independent mechanisms: continuous IS weights (`rollout_is` = token or sequence, upper-truncated = TIS, or zeroed outside [lower, upper] = IcePop) and a binary rejection mask from divergence estimators (`rollout_rs`, e.g. `token_k1,seq_max_k2`) that edits `response_mask` (`rollout_corr_helper.py` L196–260, L531–672; `docs/algo/rollout_corr.md` L102–127).
- **Guideline:** Log the mismatch before choosing a correction — setting both `rollout_is` and `rollout_rs` to null while recording rollout log-probs yields metrics only (doc L834–845); the MXFP8 page shows why: an MXFP8 rollout raised `rollout_corr/kl` about 10× over BF16 on Qwen3-8B/GSM8K (0.0063–0.0071 vs 0.0006–0.0008) while accuracy tracked BF16 over 10 steps (`docs/low_precision/mxfp8.md` L82–87).
- **Authors:** verl contributors; Rollout Correction doc by Yingru Li; MXFP8 runs by ByteDance (wandb links in the doc).
- **Year:** 2025–2026; code read at commit 8357f1e0 (2026-11-24); docs dated 10/11/2026 (rollout_corr.md) and 10/22/2026 (mxfp8.md).
- **URL:** https://github.com/verl-project/verl/tree/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc
- **Source type:** released config/code
- **Relevant topics:** train–inference mismatch, truncated importance sampling, IcePop, rejection sampling, decoupled PPO, bypass mode, score centering, FP8/MXFP8 rollout, low-precision RL

## Summary
The `algorithm.rollout_correction` block is off by default (`rollout_is: null`, `rollout_rs: null`, `bypass_mode: false`, `loss_type: ppo_clip`) (`verl/trainer/config/algorithm/rollout_correction.yaml`). In decoupled mode verl keeps three policies (π_rollout, π_old recomputed by the trainer, π_θ); in bypass mode it sets π_old = π_rollout and skips the old-log-prob forward pass (doc L780–833). The doc cites Liu, Li et al. (2025) "When Speed Kills Stability" and Li et al. (2025) arXiv:2512.23075 (doc L25–37). The MXFP8 examples train with Transformer Engine MXFP8 GEMMs and serve the rollout in MXFP8 from trainer-side quantized weights.

## Key Contributions
- Separation of IS weights (reweighting) and rejection sampling (masking), each optional (doc L102–127, L834–845).
- Named presets as classmethods of `RolloutCorrectionConfig`: `decoupled_token_is`, `decoupled_seq_is`, `decoupled_token_icepop`, `decoupled_seq_is_rs`, `decoupled_geo_rs`, `bypass_ppo_clip`, `bypass_ppo_clip_geo_rs`, `bypass_ppo_clip_k3_rs`, `bypass_pg_is`, `bypass_pg_token_icepop`, `bypass_pg_sc`, `bypass_pg_token_tis_sc`, `bypass_pg_token_icepop_sc`, `bypass_pg_geo_rs`, `decoupled_geo_rs_seq_tis` (`algorithm.py` L215–503).
- Bypass policy loss registered as `bypass_mode`: REINFORCE with explicit IS weights, or PPO-clip where the ratio π_θ/π_rollout already carries the correction and IS weights are not applied again (`core_algos.py` L2420–2460).
- Score centering (Marek & Ryabinin 2026, arXiv:2609.20807): subtract the expected score under the sampler using sampler top-k log-probs; requires bypass + REINFORCE, token or no IS, and `rollout.topk_log_probs > 0` (vLLM only) (`algorithm.py` L203–213; doc L292–321).
- MXFP8 training + MXFP8 rollout on Blackwell, with BF16 kept for `lm_head`, embeddings and MoE routers (`mxfp8.md` L20–46).

## Key Figures/Tables to Study
- `verl/trainer/ppo/rollout_corr_helper.py` L196–260 (rejection), L531–672 (IS weights); `docs/algo/rollout_corr.md` L780–845 (mode table).
- `docs/low_precision/mxfp8.md` L102–131 (Qwen3-30B-A3B table), L135–161 (Qwen3-8B DAPO table).

## Technical Details
**Pipeline at a glance (where correction enters a verl step)**
1. Rollout returns per-token log-probs: `actor_rollout_ref.rollout.calculate_log_probs=True` (`run_qwen2_5_7b_fsdp.sh` L90).
2. log_ratio = log π_train − log π_rollout, where π_train is π_old (decoupled) or π_θ (bypass) (doc L106–113).
3. IS weights: token or sequence level, safety clamp to exp(±20) (`SAFETY_BOUND = 20.0`, L74), zeroed on padding, then TIS clamp or IcePop mask, detached, optional batch normalization to mean 1 (L584–660).
4. Rejection: options in `SUPPORTED_ROLLOUT_RS_OPTIONS` (L76), comma-chained with logical AND, producing `modified_response_mask` (L196–260).
5. Loss: registered `bypass_mode` loss or decoupled PPO with the weights applied (`core_algos.py` L2420).
6. Metrics: `rollout_corr/rollout_is_mean`, `…_eff_sample_size`, `…_rs_masked_fraction`, `rollout_corr/kl` (script echo lines; `mxfp8.md`).

**IS weight truncation vs IcePop.**
```python
# verl/trainer/ppo/rollout_corr_helper.py L603-616
raw_rollout_is_weights = raw_rollout_is_weights * response_mask
# Apply TIS for a single upper bound and IcePop for a lower_upper string.
if not use_icepop:
    rollout_is_weights = raw_rollout_is_weights.clamp(max=rollout_is_threshold_upper)
else:
    assert rollout_is_threshold_lower is not None
    token_kept_mask = (raw_rollout_is_weights >= rollout_is_threshold_lower) & (
        raw_rollout_is_weights <= rollout_is_threshold_upper
    )
    rollout_is_weights = torch.where(
        token_kept_mask, raw_rollout_is_weights, torch.zeros_like(raw_rollout_is_weights)
    )
```
The docstring labels token-level IS "biased, low variance" and sequence-level "unbiased, high variance" (L556–557). Rejection modes: `token_k{1,2,3}`, `seq_sum_k{1,2,3}`, `seq_mean_k{1,2,3}`, `seq_max_k{2,3}`; K1 options take `lower_upper` ratio bounds, others an upper bound (L213–227).

**The four example scripts (`examples/rollout_correction/`).**
- `run_qwen2_5_7b_fsdp.sh`: RLOO, `rollout_is=sequence`, threshold 2.0, batch-normalized, no RS, `bypass_mode=true`, `loss_type=reinforce`; Qwen2.5-7B, batch 128, mini-batch 32, lr 5e-7, response 4,096 (L16–26, L37–46).
- `run_qwen2_5_7b_fsdp_multi_rs.sh`: `rollout_is=token` (threshold 2.0, not normalized), `rollout_rs="token_k1,seq_max_k2"` with thresholds `"0.6_1.6,2.5"`, bypass + `ppo_clip` (L17–29).
- `run_qwen3_30b_score_centering_megatron.sh`: Qwen3-30B-A3B, RLOO, bypass REINFORCE, token IS at 2.0, `score_centering=true`, `rollout.topk_log_probs=128`, FP8-block MoE rollout and FP8 KV cache, R3 router replay, TileLang fused kernels, `trainer_mode=colocate_async` (L1–31, L98, L136, L164, L174–175, L203).
- `run_qwen2_5_0_5b_score_centering_fsdp.sh`: small-scale score-centering variant (not read in detail).

**MXFP8 (`examples/mxfp8/`).** The `mxfp8` arm sets `override_transformer_config.fp8=e4m3`, `fp8_recipe=mxfp8` for training and `rollout.quantization.linear=MXFP8`, `moe=MXFP8` for the sampler; `bf16` is the baseline arm (`run_qwen3_8b_megatron.sh` L54–70). The scripts do not enable `rollout_correction`. Requirements: Blackwell SM100+, Megatron-Core ≥ 0.13, TE ≥ 2.1, vLLM 0.29 or SGLang 0.5.20 (`mxfp8.md` L14–18). Reported results:
- Qwen3-8B / GSM8K, one B200, 64 × 8 per step, 10 steps: test accuracy 0.372 / 0.616 / 0.803 (MXFP8) vs 0.329 / 0.610 / 0.819 (BF16) at steps 0 / 5 / 10; the doc states 10 steps cannot resolve the 1.6-point gap (L82–84).
- Qwen3-30B-A3B / DAPO-Math-17k, 4×B200, mean of steps 1–10: step time 506 / 487 / 489 / 452 s and `rollout_corr/kl` 0.0018 / 0.0058 / 0.0090 / 0.0045 for BF16 / FP8-TW(BF16 rollout) / FP8-BW / MXFP8; MXFP8 −11% step time, generation −24% (L118–130).
- Qwen3-8B / DAPO-Math-17k, 2×B200, FP8 KV cache in every arm, steps 23–32: FP8-TW with a BF16 rollout drifted from 0.009 to 0.037 KL and entropy fell from 0.307 to 0.062; MXFP8 entropy 0.309 → 0.344 (L150–161).

## Verification
- Checked on 2026-11-24 against the local clone at 8357f1e0. All loci refer to that commit.
- Default mismatch: `RolloutCorrectionConfig.rollout_is` defaults to `"sequence"` in the Python dataclass (`algorithm.py` L195) but `null` in `config/algorithm/rollout_correction.yaml`; the YAML is what Hydra runs load. Not traced which applies to the Python preset API without arguments.
- Not found: accuracy results for the `examples/rollout_correction` scripts in-tree (the doc points to external blogs); `rollout_corr_math.md` was not read.
- The mxfp8.md results are single runs of 10–32 steps, as the doc itself states.
