<!-- scope: Qwen3.6 technical report
     deps: [[ch-01]], [[ch-02]], [[qwen-3]], [[qwen-3-5]]
     see-also: [[qwen-3-5]], [[qwen-3]], [[deepseek-v3]]
-->

# Qwen3.6 Technical Report
- **Core Insight:** Multi-Token Prediction training plus thinking preservation across conversation turns enables a 27B dense model to outperform a 397B MoE predecessor on agentic coding benchmarks.
- **Guideline:** MTP is not just a speculative-decoding trick — it reshapes internal representations enough that a dense model trained with MTP can punch far above its parameter count, especially on sequential reasoning tasks like agentic coding.

- **Organization:** Qwen Team, Alibaba Cloud
- **Year:** 2026 (April 16: 35B-A3B MoE; April 22: 27B dense)
- **URL:** https://qwen.ai/blog?id=qwen3.6-27b / https://qwen.ai/blog?id=qwen3.6-35b-a3b
- **Relevant chapters:** Multi-Token Prediction, hybrid DeltaNet attention, thinking preservation, agentic coding, speculative decoding

## Abstract
Qwen3.6 is an iterative refinement of the Qwen3.5 architecture, optimizing for agentic coding and multi-turn reasoning stability rather than introducing a new backbone. The headline result: Qwen3.6-27B, a fully dense 27B-parameter model, outperforms Alibaba's own Qwen3.5-397B-A17B (17B active MoE) on SWE-bench and agentic coding benchmarks. Key innovations include Multi-Token Prediction (MTP) training for speculative decoding, thinking preservation that retains chain-of-thought across conversation turns, and refined inference defaults (temperature 0.2, top_p 0.9) that reduce circular reasoning loops. Both the 27B dense and 35B-A3B MoE variants share the Qwen3.5 hybrid DeltaNet backbone and are released under Apache 2.0.

## Architecture Summary

Qwen3.6 retains the Qwen3.5 hybrid Gated DeltaNet + Gated Attention backbone with the same 3:1 layer ratio. The primary architectural additions are Multi-Token Prediction heads and thinking preservation support.

**Dense Model — Qwen3.6-27B:**

| Spec | Value |
|------|-------|
| Total Parameters | 27B (all active) |
| Hidden Dimension | 5,120 |
| Layers | 64 |
| GDN Heads (V / QK) | 48 / 16 |
| GDN Head Dimension | 128 |
| Gated Attention Heads (Q / KV) | 24 / 4 |
| GA Head Dimension | 256 |
| FFN Intermediate Dimension | 17,408 |
| Context Length | 262K native (extensible to ~1.01M) |
| Layer Layout | 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)) |

**MoE Model — Qwen3.6-35B-A3B:**

| Spec | Value |
|------|-------|
| Total Parameters | 35B |
| Active Parameters | 3B per token |
| Hidden Dimension | 2,048 |
| Layers | 40 |
| GDN Heads (V / QK) | 32 / 16 |
| GDN Head Dimension | 128 |
| Gated Attention Heads (Q / KV) | 16 / 2 |
| GA Head Dimension | 256 |
| Experts (Total / Routed + Shared) | 256 / 8 + 1 |
| Context Length | 262K native (extensible to ~1.01M) |
| Layer Layout | 10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE)) |

**API-only Models (closed-weight):**
- **Qwen3.6-Plus:** Text-first model with 1M context, optimized thinking budget, temperature 0.2 / top_p 0.9 defaults (released March 30, 2026)
- **Qwen3.6-Max-Preview:** Frontier-class closed model for maximum benchmark performance

- **Vocabulary size:** 248,320 tokens (same as Qwen3.5)
- **Activation function:** SwiGLU
- **Positional encoding:** RoPE with YaRN scaling
- **Normalization:** RMSNorm pre-normalization

## Key Architectural Innovations

1. **Multi-Token Prediction (MTP)** — the model is trained to predict multiple future tokens simultaneously rather than one at a time. At inference, this enables speculative decoding: the model drafts several candidate next tokens in parallel and verifies them, yielding significant generation speedups. More importantly, MTP training reshapes internal representations — the model learns richer forward-looking features, which improves sequential reasoning quality beyond what speculative decoding alone would explain.
2. **Thinking Preservation** — a new training objective and API/template feature that retains chain-of-thought reasoning traces across conversation history in multi-turn interactions. Previous models discarded or compressed thinking blocks between turns, forcing the model to re-derive scratch work at each tool-call round. Thinking preservation keeps earlier reasoning visible, reducing redundant token generation and improving KV-cache efficiency in agentic workflows.
3. **Refined thinking budget allocation** — Qwen3.5 models often burned reasoning tokens in circular, redundant loops. Qwen3.6 addresses this with improved training for thinking-mode efficiency and tighter default inference parameters (temperature 0.2, top_p 0.9 vs Qwen3.5's 0.6 / 0.95), producing more direct reasoning chains.
4. **Dense model competitive with large MoE** — Qwen3.6-27B (all 27B parameters active) outperforms Qwen3.5-397B-A17B (17B active of 397B) on agentic coding benchmarks, demonstrating that targeted training innovations (MTP + thinking preservation + refined RL) can close the gap that raw parameter count typically creates.

## Design Decisions and Tradeoffs

- **Dense 27B vs MoE 35B-A3B:** The 27B dense model wins on coding benchmarks (SWE-bench 77.2% vs ~72%) and uses less VRAM (16.8 GB vs 21 GB at Q4), but the 3B-active MoE generates tokens 3–5x faster on identical hardware. The choice is quality-vs-throughput.
- **Text-first Plus model:** Qwen3.6-Plus drops native audio/video support that Qwen3.5-Omni had, focusing compute budget entirely on text reasoning and coding. This is an explicit tradeoff — multimodal capability is deferred to the Omni variant.
- **Conservative inference defaults:** Lower temperature (0.2) and top_p (0.9) reduce creative diversity but eliminate the circular reasoning loops that plagued Qwen3.5. For agentic coding workflows where determinism matters, this is the right tradeoff.
- **Same backbone, different training:** Qwen3.6 does not change the DeltaNet hybrid architecture from Qwen3.5. The gains come entirely from training innovations (MTP, thinking preservation, refined RL). This validates that architecture and training are separable levers — you can extract large gains from training alone on a fixed backbone.

## Training Details

- **Pre-training:** Same hybrid DeltaNet + MoE backbone as Qwen3.5, trained with Multi-Token Prediction objective (multi-step ahead)
- **Multi-Token Prediction:** Trained to predict multiple tokens ahead simultaneously; at inference enables speculative decoding where the model generates multiple candidate tokens and verifies them in parallel
- **Thinking Preservation training:** Additional training to preserve and leverage thinking traces from historical messages across conversation turns
- **Post-training RL:** Refined reinforcement learning with improved thinking-budget efficiency; million-agent environment scaling continued from Qwen3.5
- **Inference defaults:** temperature 0.2, top_p 0.9 (tighter than Qwen3.5's 0.6 / 0.95)

## Performance Highlights

**Qwen3.6-27B (dense):**

| Benchmark | Score |
|-----------|-------|
| AIME'26 | 94.1% |
| SWE-bench | 77.2% |
| SWE-bench Pro | 53.5% |
| SWE-bench Multilingual | 71.3% |
| SkillsBench | 48.2% |
| Terminal-Bench 2.0 | 59.3% |

**Qwen3.6-35B-A3B (MoE):**

| Benchmark | Score |
|-----------|-------|
| AIME'26 | 92.6% |
| LiveCodeBench v6 | 80.4% |
| SWE-bench | ~72% |

**Key comparisons:**
- Qwen3.6-27B SWE-bench Pro (53.5%) > Qwen3.5-397B-A17B (50.9%) — a 27B dense model beating a 397B MoE
- Qwen3.6-27B SkillsBench (48.2%) vs Qwen3.5-397B-A17B (30.0%) — 77% relative improvement with 14.8x fewer parameters
- Qwen3.6-27B Terminal-Bench 2.0 (59.3%) matches Claude 4.5 Opus
- All open-weight models released under Apache 2.0

## What Changed from Qwen3.5

| Dimension | Qwen3.5 | Qwen3.6 |
|-----------|---------|---------|
| Architecture backbone | Hybrid GDN + GA (new in 3.5) | Same hybrid GDN + GA (unchanged) |
| Training objective | Standard next-token prediction | Multi-Token Prediction (MTP) |
| Thinking across turns | Discarded between turns | Thinking Preservation retains CoT history |
| Inference defaults | temp 0.6, top_p 0.95 | temp 0.2, top_p 0.9 |
| Primary focus | Multimodal early fusion, architecture overhaul | Agentic coding, training refinement |
| Dense flagship | 27B (same arch) | 27B with MTP + thinking preservation |
| MoE flagship | 397B-A17B | 35B-A3B (no new large MoE; refinement release) |
| Speculative decoding | Not trained for it | MTP enables native speculative decoding |
| Key result | GDN hybrid enables 1M context | 27B dense beats 397B MoE on coding |
| Multimodal | Native early fusion (Omni variant) | Text-first (Plus); Omni deferred |
