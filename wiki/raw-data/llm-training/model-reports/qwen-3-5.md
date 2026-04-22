<!-- scope: Alibaba Qwen 3.5 family — 2026 refresh with MoE scaling and speed
     deps: [[qwen-3]]
     see-also: [[deepseek-v3.1]]
-->

# Qwen 3.5
- **Core Insight:** Qwen 3.5 is primarily a scaling + deployment-efficiency refresh of Qwen 3's four-stage hybrid-thinking recipe — bigger MoE, 19× decoding speedup, on-device small variants — rather than a new post-training algorithm.
- **Guideline:** At the 2026 MoE frontier, differentiation shifts from algorithmic novelty to serving efficiency; post-training recipes start to converge.

- **Authors / Lab:** Qwen Team, Alibaba Cloud
- **Year:** 2026 (Flagship Feb 2026; Small series March 2026; Qwen3.5-Omni and Qwen3.6-Plus closed, April 2026)
- **URL:** https://qwenlm.github.io/ (official Qwen blog) — HF model cards
- **Relevant topics:** 397B-A17B MoE, 19× decoding speedup, 1M-token context, on-device 0.8B–9B Small variants, hybrid thinking inherited

## Abstract
Qwen 3.5 is Alibaba's Feb–April 2026 refresh of the Qwen 3 family. Flagship = 397B-A17B MoE (397B total, 17B active); additional variants Qwen3.5-Flash, Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-27B; plus a Small series 0.8B–9B for on-device deployment. Hosted version supports 1M-token context. Claim: up to 19× decoding speedup vs prior flagship. Native multimodal support added. Qwen3.5-Omni and Qwen3.6-Plus (April 2026) are proprietary; earlier Qwen 3.5 releases remain open.

## Key Contributions
- Flagship 397B-A17B MoE — larger than Qwen 3's 235B-A22B flagship.
- Small model series 0.8B–9B dedicated to on-device inference.
- 1M-token context in hosted variant.
- Claimed 19× decoding speedup over Qwen 3 flagship (architecture + kernel + speculative decoding contributions — not separately broken down publicly).
- Native multimodal (not bolted-on adapter).
- Hybrid thinking / non-thinking carried forward from Qwen 3.

## Post-training pipeline
- **Overall framing:** Qwen 3.5 post-training is broadly consistent with Qwen 3's four-stage pipeline — long-CoT cold start → reasoning RL → thinking-mode fusion → general RL — with scaled data and larger MoE targets. **The Qwen team has not published a Qwen 3.5 technical report** at the level of detail of the Qwen 3 report (arxiv 2505.09388).
- **SFT data:** not publicly itemized for 3.5; presumably scaled versions of Qwen 3's mix.
- **Preference / RL algorithm:** GRPO (Qwen 3's baseline) assumed. Not re-confirmed.
- **Reward model:** not publicly documented for 3.5.
- **KL / entropy handling:** not disclosed.
- **Rollout scale:** not disclosed.
- **Hyperparameters:** not disclosed.
- **Verifiable rewards:** presumably retained for reasoning and code.
- **Self-improvement / iterative:** the Stage-2 rejection-sampling-into-Stage-3-SFT loop from Qwen 3 is assumed but not re-documented.

## Innovations vs predecessors
Changes from **Qwen 3 → Qwen 3.5**:
- Larger MoE (397B-A17B vs 235B-A22B flagship).
- Dedicated Small series for on-device.
- 1M-token context (Qwen 3 flagship was shorter).
- Native multimodal.
- 19× decoding speedup (inference-level, not post-training).
- No re-disclosed post-training algorithm changes at this point; recipe appears inherited.

## Key Figures/Tables to Study
- Flagship 397B-A17B benchmark table vs Qwen 3 235B-A22B and peers (DeepSeek V3.2, GLM-4.5, Llama 4 Maverick) — shows whether gains are from size or from post-training changes.
- Decoding-speedup decomposition (if released) — sources of the 19× claim.

## Connections
- [[qwen-3]] — direct post-training predecessor; the Qwen 3 tech report is still the primary algorithmic reference.
- [[deepseek-v3.1]] — peer 2025/2026 MoE frontier release.

## Gaps / what the report does NOT disclose
No public Qwen 3.5 technical report with post-training detail at Qwen 3's level. Not disclosed: updated RL algorithm (if any), new SFT data mix, KL/entropy handling changes, learning rates, batch sizes, clip ε, group size, rollouts per prompt, RL step counts, reward model composition, thinking-budget training details. The Small-series (0.8B–9B) post-training may differ substantially from the flagship but is not separately documented. Qwen3-Coder / Qwen3-Math follow-ups exist as inference-time variants but no separate post-training disclosures surface in search — chapter authors should monitor qwenlm.github.io for subsequent arxiv releases.
