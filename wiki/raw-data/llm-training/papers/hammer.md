<!-- scope: tool-calling synthesis — Hammer robust function-calling via function-masking + irrelevance examples
     deps: [[apigen]]
     see-also: [[toolace]], [[xlam]], [[bfcl]]
-->

# Hammer: Robust Function-Calling for On-Device Language Models via Function Masking
- **Core Insight:** On-device function-calling models break on naming-bias — they overcall tools whose names happen to be lexically similar to user queries. Hammer fixes this by (a) training with **function-name masking** (replace real names with randomized placeholders during SFT) and (b) augmenting with explicit irrelevance / rejection examples, yielding robust relevance-detection at small scales.
- **Guideline:** For small (1B–7B) function-calling models, mask real function names with random symbols ~30% of the time during SFT and add ~10% relevance-negative samples (irrelevant user queries paired with tool lists); this disentangles query-semantics from tool-name lexical cues.
- **Authors:** Qiqiang Lin, Muning Wen, Qiuying Peng, Guanyu Nie, Junwei Liao, Jun Wang, Xiaoyun Mo, Jiamu Zhou, Cheng Cheng, Yin Zhao, Jun Wang, Weinan Zhang (MadeAgents / Peking University / Baai)
- **Year:** 2024 → Hammer 2.0 / 2.1 in 2025
- **URL:** https://arxiv.org/abs/2410.04587 ; https://huggingface.co/MadeAgents
- **Relevant topics:** on-device function calling, irrelevance robustness, function-name masking, BFCL relevance

## Abstract
Hammer is a family of on-device function-calling models (0.5B, 1.5B, 7B) trained with two data-augmentation tricks addressing the *naming-bias* failure mode seen in small tool-use models: (1) function-name masking — during SFT, real function names are replaced with random placeholders (e.g., `func_a1b2c3`) so the model cannot rely on lexical matching; (2) irrelevance augmentation — ~30% of samples pair user queries with tool lists where no tool is appropriate, forcing the model to learn refusal. Hammer-7B ranks first on BFCL relevance-detection among sub-frontier models, and Hammer 2.0 / 2.1 (2025) extend the recipe to multi-step and agentic tracks.

## Key Contributions
- **Function-masking augmentation** — replacing tool names with random tokens during training eliminates lexical-shortcut learning.
- **Irrelevance augmentation** — explicitly trained negative examples cut hallucinated tool calls.
- Hammer 0.5B / 1.5B / 7B open checkpoints — targeting on-device deployment.
- Strong BFCL relevance-detection scores at small sizes, closing the gap to frontier models.
- 2025 follow-ups (Hammer 2.0, 2.1) extend to multi-turn with xLAM-2-style data.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed data:** xLAM-function-calling-60k (APIGen) + ToolBench + in-house APIs.
- **Step 1 — Function-name masking:** for ~30% of training samples, replace every tool name in both the system prompt (tool list) and the assistant's tool call with a random placeholder (sampled from `func_[a-z0-9]{6}`). Consistency enforced: same mapping used throughout a single example.
- **Step 2 — Irrelevance augmentation:** ~30% of samples are irrelevance examples — user query is about topic X, offered tools are about unrelated topic Y. Gold label = text response refusing / asking clarification, no tool call.
- **Step 3 — Parameter perturbation (Hammer 2.0):** introduce realistic parameter errors and their corrections in training data — teaches the model to repair malformed calls.
- **Step 4 — Multi-turn augmentation (Hammer 2.1):** adopts APIGen-MT-style multi-turn trajectories.
- **Filtering:** inherits APIGen's 3-layer verification on the base data.
- **Output shape:** ~150K training samples (Hammer 7B); ~250K (Hammer 2.1). Single-turn mix 70% / multi-turn 30%.
- **Teacher model(s):** GPT-4 and DeepSeek-Coder-V2 via APIGen + GPT-4 for irrelevance examples.
- **Cost / compute:** not precisely disclosed; training 7B ~ 10K GPU-hours.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** ~3.6K from xLAM inherited + irrelevance synthetic tool schemas.
- **Exact verification rules:** APIGen-inherited (format + execution + semantic) + irrelevance label sanity check.
- **Hallucination-rate measurement:** on BFCL-V2 irrelevance, Hammer-7B achieves ~90% — higher than xLAM-7B (~80%).
- **Call format:** OpenAI `tool_calls` JSON; compatible with MLX / llama.cpp tool schemas for on-device deployment.
- **Masking ratio analysis:** 30% masking is the authors' found optimum; 50% degrades general tool recall; 10% gives weak debiasing.

## Quality / diversity evaluation
- **Hammer-7B:** BFCL-V1 overall 87.9%, relevance 88.6% — matches GPT-4 on relevance.
- **Hammer-0.5B:** BFCL-V1 ~78%, remarkable for on-device.
- **Hammer 2.1 (2025):** BFCL-V3 multi-turn strong performance; competitive with xLAM-2 at same size.
- Ablation: function-name masking alone gives +7 relevance; irrelevance augmentation alone +10; combined +13.

## Risks + gotchas
- **Masking hurts out-of-distribution tools:** if real tool names carry domain information (e.g., `geocode_address`), aggressive masking can slightly hurt initial recall on unseen tools — authors mitigate via 70/30 split.
- **Irrelevance distribution matters:** if irrelevant tools are too different from query topics, the model learns cheap lexical refusal only.
- **On-device focus** means the recipe de-prioritizes very long contexts.

## Connections
- Base data lineage: [[apigen]], [[xlam]].
- Multi-turn extension adopts [[apigen-mt]] style.
- Evaluation: [[bfcl]] (especially relevance-detection subscore).
- Competing small-FC family: ToolACE, Granite FC, NexusRaven.
