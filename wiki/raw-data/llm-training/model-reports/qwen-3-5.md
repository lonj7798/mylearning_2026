<!-- scope: Qwen3.5 official release announcement — what Alibaba discloses about the Qwen3.5-397B-A17B pre-training, RL scaling, and training infrastructure
     deps: [[qwen-3]]
     see-also: [[deepseek-v3.1]], [[qwen-2.5]], [[olmo-3]]
-->

# Qwen3.5: Towards Native Multimodal Agents
- **Core Insight:** Qwen3.5-397B-A17B is a natively multimodal hybrid-attention MoE (397B total, 17B activated) whose post-training gains the Qwen team attributes to scaling the number and difficulty of RL tasks and environments rather than to a new algorithm: "the post-training performance gains in Qwen3.5 primarily stem from our extensive scaling of virtually all RL tasks and environments we could conceive", with "strong emphasis on increasing the difficulty and generalizability of RL environments, rather than optimizing for specific metrics or narrow categories of queries" (blog, "Post-training").
- **Guideline:** Treat this release as evidence about RL **environment scaling** and about asynchronous RL infrastructure, not as a recipe source: no algorithm, hyperparameter, data mix, or ablation is published, and the blog defers details to an "upcoming technical report".
- **Authors:** Qwen Team, Alibaba Cloud (organization; no individual bylines)
- **Year:** 2026 (published 2026-02-17)
- **URL:** https://www.alibabacloud.com/blog/602894 (Alibaba Cloud Community mirror of https://qwen.ai/blog?id=qwen3.5)
- **Source type:** official blog
- **Relevant topics:** MoE, Gated DeltaNet linear attention, native multimodality, RL environment scaling, asynchronous disaggregated RL, FP8 training, multilingual coverage

## Summary
The post announces the release of Qwen3.5 and the open weights of the first model in the series,
Qwen3.5-397B-A17B, a native vision-language model. The architecture follows Qwen3-Next: a
higher-sparsity MoE with a Gated DeltaNet plus Gated Attention hybrid, stability optimizations, and
multi-token prediction. Pre-training is described along three axes — power (more visual-text tokens
than Qwen3 under stricter filtering), efficiency (architecture and throughput), and versatility (native
multimodality and multilingual coverage). Post-training is described only as large-scale RL
environment scaling. The post also describes the training infrastructure: a heterogeneous multimodal
training stack with decoupled parallelism, a native FP8 pipeline, and a fully disaggregated
asynchronous RL framework. Qwen3.5-Plus is the hosted variant with a 1M context window by default and
built-in tools. Qwen Chat exposes three modes: auto, thinking, and fast.

## Key Contributions
- Releases open weights for Qwen3.5-397B-A17B, a native vision-language MoE with 17B activated parameters ("intro").
- States RL environment scaling, not metric-targeted tuning, as the source of post-training gains ("Post-training").
- Reports decoding-throughput multipliers for the hybrid architecture at 32k and 256k context ("Efficiency").
- Expands language and dialect coverage from 119 to 201 and the vocabulary from 150k to 250k ("Versatility").
- Describes a disaggregated asynchronous RL framework and reports a 3x-5x end-to-end speedup ("Infrastructure").

## Key Figures/Tables to Study
- "Language" benchmark table — Qwen3.5-397B-A17B against GPT5.2, Claude 4.5 Opus, Gemini-3 Pro, Qwen3-Max-Thinking, and K2.5-1T-A32B.
- "Vision Language" benchmark table — the same comparison for multimodal tasks.
- Base-model table — Qwen3.5-397B-A17B against Qwen3-235B-A22B, GLM-4.5-355B-A32B, DeepSeek-V3.2-671B-A37B, and K2-1T-A32B (MMLU 88.61 vs 87.33 / 86.56 / 88.11 / 87.38).
- The agent-capability ranking figure in "Post-training", averaged over BFCL-V4, VITA-Bench, DeepPlanning, Tool-Decathlon, and MCP-Mark.

## Technical Details
- 397B total parameters, 17B activated per forward pass ("intro").
- Architecture: Qwen3-Next — higher-sparsity MoE, Gated DeltaNet plus Gated Attention hybrid attention, stability optimizations, multi-token prediction ("Efficiency").
- Decoding throughput at 32k / 256k context: 8.6x / 19.0x that of Qwen3-Max at "comparable" performance, and 3.5x / 7.2x that of Qwen3-235B-A22B ("Efficiency").
- Languages and dialects: 119 → 201 ("intro"; "Versatility"). Vocabulary: 250k, against 150k previously, which the post says improves encoding/decoding efficiency by 10-60% "across most languages" ("Versatility").
- Pre-training data: "a significantly larger scale of visual-text tokens compared to Qwen3", with enriched Chinese/English, multilingual, STEM, and reasoning data "under stricter filtering"; no token count, no mixture percentages ("Power").
- Claimed parity: "Qwen3.5-397B-A17B matches the >1T-parameter Qwen3-Max-Base" ("Power").
- Multimodal training infrastructure: parallelism strategies decoupled across vision and language components; the post reports "near 100% training throughput versus pure-text baselines on mixed text-image-video data" ("Infrastructure").
- FP8: a native FP8 pipeline applied to activations, MoE routing, and GEMM, with runtime monitoring keeping BF16 in sensitive layers; reported as ~50% activation-memory reduction and >10% speedup, "scaling stably to tens of trillions of tokens" ("Infrastructure").
- Asynchronous RL framework: fully disaggregated training-inference architecture, supporting text, multimodal, and multi-turn settings; techniques listed are FP8 end-to-end training, rollout router replay, speculative decoding, and multi-turn rollout locking; reported effects are bounded gradient staleness, mitigated data skewness, and a 3x-5x end-to-end speedup; described as accommodating "million-scale agent scaffolds and environments" ("Infrastructure").
- Hosted serving: Qwen3.5-Plus has a 1M context window by default and built-in tools with adaptive tool use ("intro"). Qwen Chat offers auto, thinking, and fast modes; auto mode uses adaptive thinking with search and code-interpreter tools ("Play with Qwen3.5"). The API exposes `enable_thinking` ("Play with Qwen3.5").
- Benchmark evaluation notes in the post: MCP-Mark truncates Playwright tool responses at 32k tokens; MMLU-ProX is averaged over 29 languages; WMT24++ is averaged over 55 languages with XCOMET-XXL ("Performance", evaluation notes).
- The open-weights model card (`wiki/courses/llm-training/ch-34/excerpts/qwen-3-5-model-card.md`, read 2026-09-15) records what the blog omits about the released checkpoint: 60 layers in the pattern "15 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))", 512 experts with 10 routed plus 1 shared activated, 262,144 native context extensible to 1,010,000, thinking mode on by default with `enable_thinking: False` to disable it.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen3.5-397B-A17B | 397B total / 17B active | pretrain-stable | Training precision | native FP8 on activations, MoE routing, GEMM; BF16 retained in sensitive layers by runtime monitoring | alibabacloud.com/blog/602894 (2026-02-17), "Infrastructure" | verified 2026-09-18 | post reports ~50% activation-memory reduction and >10% speedup |
| Qwen3.5-397B-A17B | 397B / 17B | pretrain-stable | Tokens seen | not reported | blog "Power"; model card; no technical report published | not reported | checked: blog body, model card, Qwen blog index |
| Qwen3.5-397B-A17B | 397B / 17B | pretrain-stable | Vocabulary size | 250,000 (previously 150,000) | blog "Versatility" | verified 2026-09-18 | post reports 10-60% encoding/decoding efficiency gain across most languages |
| Qwen3.5-397B-A17B | 397B / 17B | long-context | Native / extended context | 262,144 / 1,010,000 tokens | HF model card, "Model Overview" (read 2026-09-15) | verified 2026-09-15 | card advises keeping >=128K context to preserve thinking quality |
| Qwen3.5-397B-A17B | 397B / 17B | SFT | Data mix, size, epochs, masking | not reported | blog; model card | not reported | checked: blog body, model card |
| Qwen3.5-397B-A17B | 397B / 17B | RL | Algorithm, KL coefficient, clip ε, samples per prompt, prompts per step, reward model | not reported | blog "Post-training"; model card | not reported | checked: blog body and evaluation notes, model card; blog defers to an "upcoming technical report" |
| Qwen3.5-397B-A17B | 397B / 17B | RL | Environment scale | "million-scale agent scaffolds and environments" | blog "Infrastructure" | verified 2026-09-18 | no ablation reported; the agent-capability figure is a ranking average over 5 benchmarks |
| Qwen3.5-397B-A17B | 397B / 17B | RL | Framework | fully disaggregated async train-infer; FP8 end-to-end, rollout router replay, speculative decoding, multi-turn rollout locking | blog "Infrastructure" | verified 2026-09-18 | post reports 3x-5x end-to-end speedup; no baseline named |

## Findings relevant to generality
- The stated post-training objective is generality of environments rather than benchmark scores: the gains "stem from our extensive scaling of virtually all RL tasks and environments we could conceive", with emphasis on "the difficulty and generalizability of RL environments, rather than optimizing for specific metrics or narrow categories of queries" ("Post-training"). This is the team's own **Interpretation**; no held-out or contamination analysis is given.
- The evidence offered for it is a single figure whose metric is the average ranking across BFCL-V4, VITA-Bench, DeepPlanning, Tool-Decathlon, and MCP-Mark ("Post-training"). An average of rankings is not an average of scores, so the figure does not show effect sizes.
- Multilingual breadth is a stated design goal: 119 → 201 languages and dialects, with WMT24++ reported over 55 languages and MMLU-ProX over 29 ("intro"; evaluation notes).

## Findings relevant to agentic training
- The RL framework "natively supports agentic workflows, facilitating seamless multi-turn interactions without framework-induced interruptions", and multi-turn rollout locking is listed as the mechanism that keeps train-infer consistency across turns ("Infrastructure").
- The blog does not list the environments, tools, or reward sources used. The model card's benchmark notes supply the evaluation-side context policy (context folding at 256k) that changes BrowseComp from 69.0 to 78.6 for the same checkpoint (model card excerpt, read 2026-09-15).

## Connections
- [[qwen-3]] — the previous generation and the last Qwen release with a published post-training technical report (arXiv:2505.09388); Qwen3.5 publishes no comparable document.
- [[deepseek-v3.1]] — contemporaneous open MoE release used as a comparison point in the blog's base-model table.
- [[olmo-3]] — contrast case: a fully documented open training flow, against which this release's disclosure level can be judged.

## Verification
- Checked on 2026-09-18 against: https://www.alibabacloud.com/blog/602894 (Alibaba Cloud Community, 2026-02-17), plus the model-card facts already verified in `wiki/courses/llm-training/ch-34/excerpts/qwen-3-5-model-card.md` on 2026-09-15.
- Corrections to the previous card version:
  - "Qwen 3.5 is primarily a scaling + deployment-efficiency refresh of Qwen 3's four-stage hybrid-thinking recipe … rather than a new post-training algorithm" → the blog states post-training gains come from scaling RL tasks and environments; it makes no claim about inheriting the Qwen3 four-stage pipeline.
  - "19× decoding speedup" stated without a baseline or context length → 19.0x is against Qwen3-Max at 256k context; at 32k it is 8.6x, and against Qwen3-235B-A22B it is 3.5x / 7.2x ("Efficiency").
  - "1M-token context in hosted variant" as the only context fact → the open checkpoint is 262,144 native, extensible to 1,010,000; 1M by default applies to the hosted Qwen3.5-Plus.
  - "Year: 2026 (Flagship Feb 2026; Small series March 2026; Qwen3.5-Omni and Qwen3.6-Plus closed, April 2026)" → the primary artifact is dated 2026-02-17 and announces one open model, Qwen3.5-397B-A17B.
  - "Hybrid thinking / non-thinking carried forward from Qwen 3" → the released checkpoint runs thinking mode by default and is switched off with `enable_thinking: False`, and Qwen Chat exposes auto / thinking / fast modes; this is a single checkpoint serving both modes, not the Qwen3 hybrid described in the Qwen3 report.
- Removed as unsupported by the source:
  - "Preference / RL algorithm: GRPO (Qwen 3's baseline) assumed. Not re-confirmed." and "the Stage-2 rejection-sampling-into-Stage-3-SFT loop from Qwen 3 is assumed but not re-documented" — assumptions, not statements of any Qwen3.5 source.
  - "SFT data: presumably scaled versions of Qwen 3's mix" and "Verifiable rewards: presumably retained for reasoning and code" — not stated anywhere.
  - "additional variants Qwen3.5-Flash, Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-27B; plus a Small series 0.8B–9B for on-device deployment" and "Qwen3.5-Omni and Qwen3.6-Plus (April 2026) are proprietary" — not present in the release post or the model card.
  - "No re-disclosed post-training algorithm changes at this point; recipe appears inherited" — contradicted by the post's RL-scaling statement.
- Not reported by the source: token counts; SFT and RL data composition; RL algorithm and hyperparameters; reward-model design; distillation of any kind (the separate Qwen3-VL technical report, arXiv:2511.21631, December 2025, documents strong-to-weak off-policy then on-policy distillation for lightweight Qwen models, but it is a different artifact and is not yet a card in this library); per-size recipes for any variant other than Qwen3.5-397B-A17B; later Qwen releases (Qwen3.6, April 2026; Qwen3.8-2.4T-A95B, 2026-08-12) are outside this card's scope and are uncovered by the library.
