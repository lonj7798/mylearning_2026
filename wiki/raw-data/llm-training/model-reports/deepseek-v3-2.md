<!-- scope: companion card for the DeepSeek-V3.2 report (arXiv:2512.02556): RL-compute and length-budget evidence (V3.2 vs V3.2-Speciale), agentic-synthesis ablation, and evaluation-protocol sensitivity. The method (DSA, continued pre-training, specialist distillation, mixed GRPO, stabilizers, task synthesis) is carded in full at [[deepseek-v3.1]] with ledger [[deepseek-v3.1-recipe]]; this file does not repeat it.
     deps: [[deepseek-v3.1]], [[grpo]]
     see-also: [[deepseek-v3.1-recipe]], [[deepseekmath-v2]], [[deepseek-v3]], [[deepseek-v4-recipe]], [[deepseek-r1]], [[kimi-k2-agentic-data]], [[context-folding]], [[browsecomp-plus]]
-->

# DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
- **Core Insight:** The same post-training pipeline gives two operating points that differ mainly in the RL length penalty: DeepSeek-V3.2 (stricter token constraint) and DeepSeek-V3.2-Speciale (reasoning data only, reduced length penalty, DeepSeekMath-V2 proof reward), and Speciale's accuracy gains come with 1.4–2.3× more output tokens on the same benchmarks (§3, §4.2, Table 3; ratios computed by this card from Table 3 token counts).
- **Guideline:** Report accuracy together with output-token count and the agent harness used, because in this report the same checkpoint moves from 46.4 to 39.3 on Terminal Bench 2.0 when the harness changes from Claude Code (thinking) to Terminus (non-thinking), and BrowseComp moves from 51.4 to 67.6 with test-time context management (§4.1, §4.4).
- **Authors:** DeepSeek-AI (byline; App. E lists contributors alphabetically by first name)
- **Year:** 2025 (arXiv v1 2025-12-02; only version)
- **URL:** https://arxiv.org/abs/2512.02556
- **Source type:** official technical report
- **Relevant topics:** RL compute scaling, length penalty, token efficiency, agentic task synthesis, synthetic-environment RL transfer, agent evaluation protocol, test-time context management

## Abstract
See [[deepseek-v3.1]] for the full abstract. In brief: DeepSeek Sparse Attention (DSA) added to DeepSeek-V3.1-Terminus by continued training; a GRPO-based RL protocol with a post-training budget above 10% of pre-training cost; and an agentic task-synthesis pipeline (over 1,800 environments, 85,000 prompts) (Abstract, §1).

## Key Contributions (covered here)
- RL compute: the RL budget "already exceeds 10% of the pre-training cost", and the authors report consistent gains as the RL budget grew over recent months; no curve is printed (§4.1).
- Length budget as a design choice: V3.2 results are "constrained by a length constraint reward model"; removing it (Speciale) improves results (§4.1, §4.2).
- Synthetic agent tasks are hard for frontier models and transfer to unseen tool benchmarks when used alone for RL (§4.3, Table 5, Fig. 5).
- Agent evaluation is sensitive to harness, thinking mode, and context management (§4.1, §4.4, App. C).

## Key Figures/Tables to Study
- Table 3: accuracy with output-token counts for GPT-5-High, Gemini-3.0-Pro, Kimi-K2-Thinking, V3.2, V3.2-Speciale.
- Table 4: Speciale per-problem scores at IMO 2025, CMO 2025, IOI 2025, ICPC WF 2025.
- Table 5 and Fig. 5: pass@k of synthetic general-agent tasks; RL on synthetic tasks only.
- Fig. 6: BrowseComp vs real steps for Summary, Discard-75%, Discard-all, Parallel-fewest-step.
- App. C Table 9: non-thinking vs thinking on agent benchmarks.

## Technical Details
**Architecture (brief).** DSA = lightning indexer (ReLU-scored, FP8-capable) + top-k key-value selection (k = 2048) on the MQA mode of MLA; core attention O(L²) → O(Lk), indexer stays O(L²) (§2.1, §2.3). Cost curves in Fig. 3 are estimated from service on H800 at 2 USD per GPU hour (§2.3). Training values for the indexer stages are in [[deepseek-v3.1-recipe]].

**V3.2 vs Speciale, accuracy (output tokens) (Table 3).**
| Benchmark | V3.2 | V3.2-Speciale | Gemini-3.0-Pro |
|---|---|---|---|
| AIME 2025 | 93.1 (16k) | 96.0 (23k) | 95.0 (15k) |
| HMMT Feb 2025 | 92.5 (19k) | 99.2 (27k) | 97.5 (16k) |
| IMOAnswerBench | 78.3 (27k) | 84.5 (45k) | 83.3 (18k) |
| LiveCodeBench | 83.3 (16k) | 88.7 (27k) | 90.7 (13k) |
| Codeforces rating | 2386 (42k) | 2701 (77k) | 2708 (22k) |
| GPQA Diamond | 82.4 (7k) | 85.7 (16k) | 91.9 (8k) |
| HLE | 25.1 (21k) | 30.6 (35k) | 37.7 (15k) |
Kimi-K2-Thinking reaches 94.5 on AIME 2025 with 24k tokens (Table 3). The authors state Speciale's token efficiency is "significantly inferior" to Gemini-3.0-Pro and that stricter token constraints were imposed on V3.2 to reduce cost and latency (§4.2).

**Speciale competition protocol (App. D) and results (Table 4).** Max generation 128k; no tools or internet. IOI: 500 samples per problem → drop samples failing sample tests or over length → drop samples where V3.2-Exp detects refusal/inability → submit the 50 with the longest thinking traces. ICPC: 32 samples, same filter. IMO/CMO: generate-verify-refine loop from [[deepseekmath-v2]]. Scores: IMO 2025 35/42; CMO 2025 102/126; IOI 2025 492/600 (rank 10); ICPC WF 2025 10/12 (rank 2) (Table 4).

**Synthetic-task ablation (§4.3).** 50 sampled general-agent tasks, pass@1/2/4: V3.2-Exp 12/18/26%; Sonnet-4.5 34/47/62%; Gemini-3.0-Pro 51/65/74%; GPT-5-Thinking 62/75/82% (Table 5). RL from V3.2-SFT on synthetic general-agent tasks only, non-thinking mode, improves Tau2Bench, MCP-Mark, MCP-Universe; V3.2-Exp (RL only in search and code environments) does not improve on them (Fig. 5; values only in the figure).

**Evaluation protocol (§4.1, App. C).** Temperature 1.0; 128K context; math template "{question}\nPlease reason step by step, and put your final answer within \boxed{}."; HLE with the official template gives 23.9 instead of 25.1. Terminal Bench 2.0: 46.4 (Claude Code, thinking), 37.1 (Claude Code, non-thinking), 39.3 (Terminus, non-thinking). SWE-Verified: 73.1 internal framework; 72–74 across Claude Code, RooCode, non-thinking. τ2-bench uses the model itself as user agent: Airline 63.8, Retail 81.1, Telecom 96.2. MCP tasks: redundant self-verification makes trajectories exceed 128K, which the authors say lowers scores (§4.1).

**Test-time context management (§4.4).** Triggered when usage exceeds 80% of the window. Summary extends average steps to 364 and reaches up to 60.2; Discard-all reaches 67.6, comparable to Parallel-fewest-step with fewer steps; about 20%+ of BrowseComp cases exceed 128K (§4.1, §4.4, Fig. 6).

## Recipe ledger
Method rows (DSA stages, specialists, rewards, stabilizers, task counts) are in [[deepseek-v3.1-recipe]]. Rows below are not in that ledger.
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3.2 | not printed | RL | length control | "length constraint reward model"; stricter token constraints than Speciale | arXiv:2512.02556v1 §4.1, §4.2 | verified 2026-11-24 | Table 3: lower accuracy, 1.4–2.3× fewer tokens than Speciale |
| DeepSeek-V3.2-Speciale | not printed | RL | length penalty value | "reduced"; numeric value not printed | §3 | not reported | — |
| DeepSeek-V3.2 | not printed | eval-gate | Terminal Bench 2.0 by harness/mode | 46.4 / 37.1 / 39.3 | §4.1, App. C Table 9 | verified 2026-11-24 | — |
| DeepSeek-V3.2 | not printed | eval-gate | thinking vs non-thinking (SWE-V, τ2, MCP-Universe, MCP-Mark, Tool-Decathlon) | 73.1/72.1, 80.3/77.2, 45.9/38.6, 38.0/26.5, 35.2/25.6 | App. C Table 9 | verified 2026-11-24 | — |
| DeepSeek-V3.2-Speciale | not printed | eval-gate | IOI submission selection | 500 samples → filters → 50 longest thinking traces | App. D | verified 2026-11-24 | no ablation reported |

## Findings relevant to generality and negative feedback
- **Generality.** MCP-Universe, MCP-Mark, and Tool-Decathlon environments were not seen in RL; the authors read the gains as out-of-domain transfer (§4.1). Knowledge breadth lags Gemini-3.0-Pro, which the authors attribute to fewer total training FLOPs (§5).
- **Length as a capability axis.** The same recipe trades accuracy for tokens through the length penalty (Table 3); output length is also a failure source in agent tasks, where self-verification loops overflow 128K (§4.1).
- **Negative feedback.** Off-Policy Sequence Masking of high-divergence negative-advantage sequences is carded in [[deepseek-v3.1]] (§3.1, Eq. 9).

## Connections
- [[deepseek-v3.1]] / [[deepseek-v3.1-recipe]]: full method card and ledger for this report.
- [[deepseekmath-v2]]: the proof dataset, reward, and generate-verify-refine loop used for Speciale (§3, App. D).
- [[context-folding]], [[browsecomp-plus]]: other test-time context-management and search-agent evaluation work.
- [[deepseek-v4-recipe]]: the next DeepSeek report's post-training settings.

## Verification
- Checked on 2026-11-24 against: arXiv:2512.02556v1 PDF, body §1–§5 and App. A–D.
- Duplication note: the existing [[deepseek-v3.1]] card already describes this report; this card adds only Table 3–5, App. C–D, and §4.1/§4.4 protocol facts.
- Not reported by the source: parameter count; RL hyperparameters (β, ε, δ, G, LR, batch); numeric length-penalty values for V3.2 and Speciale; RL-compute-vs-score curve.
