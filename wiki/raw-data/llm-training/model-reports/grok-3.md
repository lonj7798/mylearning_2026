<!-- scope: xAI Grok 3 Beta launch post (2025-02-19); official blog, no technical report; reasoning RL, test-time compute, 1M context, DeepSearch agent
     see-also: [[grok-4-1]], [[deepseek-r1]], [[llama-4]], [[deepseek-v3]], [[qwen-3]]
-->

# Grok 3 Beta — The Age of Reasoning Agents
- **Core Insight:** xAI reports that Grok 3 was trained on its Colossus supercluster with "10x the compute of previous state-of-the-art models" and that its Think variants were trained with large-scale RL on chain-of-thought, reaching 93.3% on AIME 2025 at cons@64 (post, "Next-Generation Intelligence"; "Thinking Harder"); the post discloses no training hyperparameters.
- **Guideline:** When a recipe value is needed, use a technical report such as [[deepseek-v3]] or [[qwen-3]]; use this post only for what xAI states about the existence of reasoning RL, the context length, and the evaluation settings it prints.
- **Authors:** xAI (official post; the site is now branded SpaceXAI)
- **Year:** 2025 (published 2025-02-19)
- **URL:** https://x.ai/news/grok-3
- **Source type:** official blog
- **Relevant topics:** reasoning RL, test-time compute, consensus sampling, long context, tool-using agents, benchmark reporting

## Summary
The post announces an early preview of Grok 3 and Grok 3 mini, plus two beta reasoning models, Grok 3 (Think) and Grok 3 mini (Think). xAI states that Grok 3 was trained on the Colossus supercluster with 10x the compute of previous state-of-the-art models. The Think models were trained with reinforcement learning "at an unprecedented scale" to refine the chain-of-thought process. The post reports benchmark numbers for the reasoning and non-reasoning modes, a 1 million token context window, an LMArena Chatbot Arena Elo of 1402, and a first agent called DeepSearch. It states that both models were still in training at the time of the announcement. It names no RL algorithm, reward design, data mixture, model size, or optimizer setting.

## Key Contributions
- States that reasoning ability in the Think models was refined through large-scale RL on chain-of-thought (post, "Thinking Harder").
- Exposes test-time reasoning as a user-selectable mode ("Think") that reasons from a few seconds to several minutes (post, "Thinking Harder").
- Reports a 1 million token context window and a LOFT (128k) result (post, "Pretraining on a Massive Scale").
- Announces DeepSearch, described as xAI's first agent, built on Grok 3 with internet access (post, "Grok Agents").
- Prints the test-time compute setting (cons@64) only for the AIME 2025 number (post, "Thinking Harder").

## Key Figures/Tables to Study
- Non-reasoning benchmark table (post, "Pretraining on a Massive Scale"): Grok 3 Beta and Grok 3 mini Beta against Gemini 2.0, DeepSeek-V3, GPT-4o, and Claude 3.5 Sonnet.
- Reasoning-mode results paragraph (post, "Thinking Harder"); the accompanying chart values are not present in the page text.

## Technical Details

### Models and training claims
- Released variants: Grok 3, Grok 3 mini, Grok 3 (Think), Grok 3 mini (Think) (post, intro and "Thinking Harder").
- Compute: trained on the Colossus supercluster with "10x the compute of previous state-of-the-art models" (post, intro). The baseline models and the compute unit are not stated.
- RL: the Think models "were trained using reinforcement learning (RL) at an unprecedented scale to refine its chain-of-thought process, enabling advanced reasoning in a data-efficient manner" (post, "Thinking Harder"). No RL algorithm, reward, or data size is given.
- Behaviors the post attributes to RL: refining problem-solving strategies, correcting errors through backtracking, simplifying steps, and using pretraining knowledge (post, "Thinking Harder").
- Checkpoint status: "Both models are still in training and will evolve rapidly" (post, intro); "Grok 3's training is ongoing" (post, "What's Next").
- Future scale: xAI states it is "preparing to train even larger models on our 200,000 GPU cluster" (post, "Pretraining on a Massive Scale"). This number does not describe Grok 3's own training.

### Reasoning-mode results (post, "Thinking Harder")
- cons@64 is a consensus (majority-vote) answer over 64 samples; the post calls it "our highest level of test-time compute" but does not define it further.
- Grok 3 (Think): AIME 2025 93.3% at cons@64; GPQA 84.6%; LiveCodeBench 79.4%. The sampling setting for the GPQA and LiveCodeBench numbers is not stated.
- Grok 3 mini: AIME 2024 95.8%; LiveCodeBench 80.4% (stated in the reasoning section; the sampling setting is not stated).
- AIME 2025 was released on Feb 12, 7 days before the post.

### Non-reasoning results (post, "Pretraining on a Massive Scale", table)
| Benchmark | Grok 3 Beta | Grok 3 mini Beta |
|---|---|---|
| AIME'24 | 52.2% | 39.7% |
| GPQA | 75.4% | 66.2% |
| LCB | 57.0% | 41.5% |
| MMLU-pro | 79.9% | 78.9% |
| LOFT (128k) | 83.3% | 83.1% |
| SimpleQA | 43.6% | 21.7% |
| MMMU | 73.2% | 69.4% |
| EgoSchema | 74.5% | 74.3% |

- LMArena Chatbot Arena Elo 1402 (post, intro); an early version under the codename "chocolate" ranked first in all categories (post, "Pretraining on a Massive Scale").

### Context and agents
- Context window: 1 million tokens, "8 times larger than our previous models" (post, "Pretraining on a Massive Scale").
- LOFT (128k) accuracy is averaged across 12 tasks (post, "Pretraining on a Massive Scale").
- "Equipped with code interpreters and internet access, Grok 3 models learn to query for missing context, dynamically adjust their approach, and improve their reasoning based on feedback" (post, "Grok Agents"). The training method for this is not described.
- API: Grok 3 and Grok 3 mini to be released via the API "in the coming weeks"; tool use, code execution, and agent capabilities planned for the Enterprise API (post, "API Coming Soon"; "What's Next").

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Grok 3 Beta | not reported | pretrain-stable | compute | "10x the compute of previous state-of-the-art models", Colossus supercluster | x.ai/news/grok-3 (2025-02-19), intro | verified 2026-09-14 | no ablation reported |
| Grok 3 Beta | not reported | long-context | context window | 1 million tokens | post, "Pretraining on a Massive Scale" | verified 2026-09-14 | no ablation reported |
| Grok 3 (Think), Grok 3 mini (Think) | not reported | RL | method | RL on chain-of-thought "at an unprecedented scale"; algorithm not named | post, "Thinking Harder" | verified 2026-09-14 | no ablation reported |
| Grok 3 (Think) | not reported | eval-gate | test-time compute for AIME 2025 | cons@64 | post, "Thinking Harder" | verified 2026-09-14 | no ablation reported |
| Grok 3 family | not reported | all | parameters, tokens, data mixture, sequence length, batch, LR, optimizer, SFT data, reward design, KL, samples per prompt, RL steps | not reported | checked: full post text | not reported | not applicable |

## Findings relevant to generality, long context, agentic training
- **Generality.** The post states that "Grok 3 (Think)'s performance generalizes across diverse problem domains" and supports it with example transcripts (coding, ASCII art, puzzle, math), not with a held-out evaluation (post, "Thinking Harder"). Status: Result (single study), qualitative.
- **Measurement.** The benchmark numbers come from models "still in training" (post, intro). The post reports no contamination analysis. In the same table, Grok 3 Beta scores 43.6% on SimpleQA and Gemini 2.0 scores 44.3%.
- **Long context.** The post reports a 1M-token window but no evaluation above 128k tokens.
- **Agentic training.** The post says Grok 3 models with code interpreters and internet access "learn to" query for missing context, without describing the training.

## Connections
- [[grok-4-1]] — later xAI release whose model card describes pre-training, mid-training, SFT, and RL stages at category level.
- [[deepseek-r1]] — reasoning RL with a named algorithm (GRPO) and rule-based rewards, which this post does not disclose.
- [[llama-4]] — another 2025 release whose primary technical source is a blog post.
- [[deepseek-v3]], [[qwen-3]] — technical reports with training settings.

## Verification
- Checked on 2026-09-14 against: https://x.ai/news/grok-3 (page dated Feb 19, 2025; full page text and a second read of the page for chart values).
- Corrections to the previous card version:
  - Title "Grok 3" → published title "Grok 3 Beta — The Age of Reasoning Agents".
  - "roughly 10x the compute of the previous generation" → "10x the compute of previous state-of-the-art models" (intro).
  - "Benchmark examples are presented with cons@64 and other high test-time-compute settings" → cons@64 is stated only for the AIME 2025 result; no other setting is named ("Thinking Harder").
  - "DeepSearch as the first reasoning agent" → the post calls it "our first agent" ("Grok Agents").
  - "Claims 1M-token context and strong long-context performance" → 1M tokens; LOFT (128k) 83.3%, averaged across 12 tasks.
- Removed as unsupported by the source: "Public docs list Grok 3 as an API model" (the post says API release is upcoming; docs are a different artifact); "Shows a frontier-lab shift toward blog-level disclosure" (course interpretation); "No standalone technical report was available in the public sources I used" (not a claim of the artifact); "use a 1M-token context window, and rely on large-scale reinforcement learning to improve ... tool use" (the post does not attribute tool use to RL).
- Not reported by the source: model sizes, token counts, data mixture, RL algorithm, reward design, any hyperparameter, sampling settings for results other than AIME 2025, chart values for Chatbot Arena and the reasoning chart.
