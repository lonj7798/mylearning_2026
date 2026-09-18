<!-- scope: QwQ-32B release (Qwen Team blog 2025-03 + official model card): two-stage outcome-reward RL from a cold-start checkpoint, release benchmark figure, and the official decoding and output-format settings used when QwQ-32B generates reasoning traces; the earlier QwQ-32B-Preview release (2024-11) is summarized separately
     deps: [[qwen-2.5]]
     see-also: [[deepseek-r1]], [[qwen-3]], [[open-thoughts]], [[sky-t1]], [[bespoke-stratos]], [[scaling-reasoning-losing-control-mathif]]
-->

# QwQ-32B: Embracing the Power of Reinforcement Learning
- **Core Insight:** QwQ-32B, a 32.5B-parameter dense model trained with outcome-reward RL from a cold-start checkpoint, scores 79.5 on AIME24 and 63.4 on LiveCodeBench (24.08–25.02), compared with 79.8 and 65.9 for the 671B-parameter DeepSeek-R1 in the release figure (blog "Performance" figure; model card "Introduction").
- **Guideline:** When QwQ-32B generates reasoning traces, start the response with `<think>\n` and sample with temperature 0.6, top-p 0.95, top-k 20–40, and min-p 0 instead of greedy decoding, because the model card states that empty thinking content degrades output quality and that greedy decoding causes endless repetitions (model card "Usage Guidelines" 1–2).
- **Authors:** Qwen Team
- **Year:** 2025 (blog dated 2025-03-06)
- **URL:** https://qwenlm.github.io/blog/qwq-32b/ (companion official model card of the same release: https://huggingface.co/Qwen/QwQ-32B)
- **Source type:** official blog (with the official model card)
- **Relevant topics:** outcome-reward RL, verifiable rewards, general-capability RL stage, open-weight reasoning teacher, decoding settings for trace generation, YaRN context extension

## Summary
The Qwen Team introduces QwQ-32B, a 32B-parameter reasoning model released with open weights under Apache 2.0 on Hugging Face and ModelScope. The blog reports performance comparable to DeepSeek-R1 (671B parameters, 37B activated) and attributes it to RL applied to a strong pretrained foundation model. Training started from a cold-start checkpoint. A first RL stage used only math and coding tasks with an accuracy verifier and a code-execution server as reward sources. A second, shorter RL stage used a general reward model and rule-based verifiers to improve general capabilities. The model card adds the architecture, the context length, and the recommended inference settings.

## Key Contributions
- Open-weight 32B reasoning model under Apache 2.0 (blog introduction).
- Stage 1 RL on math and coding with outcome rewards from an accuracy verifier (math) and a code-execution server that runs predefined test cases, "rather than relying on traditional reward models" (blog "Reinforcement Learning").
- Stage 2 RL for general capabilities with a general reward model plus rule-based verifiers; the blog reports that "a small amount of steps" improves instruction following, alignment with human preference, and agent performance "without significant performance drop in math and coding" (blog "Reinforcement Learning").
- Agent-related capabilities integrated into the reasoning model: tool use and reasoning that adapts to environmental feedback (blog introduction).
- Official inference guidelines for output format, sampling, multi-turn history, and long inputs (model card "Usage Guidelines").

## Key Figures/Tables to Study
- **Blog "Performance" figure** (the same image as the model card's `figures/benchmark.jpg`): QwQ-32B vs DeepSeek-R1-671B, OpenAI-o1-mini, DeepSeek-R1-Distill-Llama-70B, DeepSeek-R1-Distill-Qwen-32B on AIME24, LiveCodeBench, LiveBench, IFEval, BFCL.
- **Model card "Usage Guidelines"** items 1–5.

## Technical Details
- **Architecture:** 32.5B parameters (31.0B non-embedding); 64 layers; GQA with 40 query heads and 8 KV heads; RoPE, SwiGLU, RMSNorm, attention QKV bias (model card "Introduction").
- **Training stages:** "Pretraining & Post-training (Supervised Finetuning and Reinforcement Learning)" (model card); metadata `base_model: Qwen/Qwen2.5-32B` (model card YAML header). The blog calls the RL starting point "a cold-start checkpoint" and gives no cold-start data description (blog "Reinforcement Learning").
- **Benchmark figure values** (QwQ-32B / DeepSeek-R1-671B / o1-mini / R1-Distill-Llama-70B / R1-Distill-Qwen-32B) (blog "Performance" figure):
  - AIME24: 79.5 / 79.8 / 63.6 / 70.0 / 72.6
  - LiveCodeBench (24.08–25.02): 63.4 / 65.9 / 53.8 / 57.5 / 57.2
  - LiveBench: 73.1 / 71.6 / 59.1 / 57.9 / 54.6
  - IFEval: 83.9 / 83.3 / 84.8 / 79.3 / 72.5
  - BFCL: 66.4 / 60.3 / 62.8 / 49.3 / 53.5
  - The figure does not state the number of samples, decoding settings, or generation budget.
- **Output format:** the response should start with `<think>\n`; `apply_chat_template(..., add_generation_prompt=True)` inserts this automatically, so the decoded response may lack the opening `<think>` tag (model card Usage Guideline 1). The DashScope API example returns the thinking text in a separate `reasoning_content` field and the answer in `content` (blog "Use QwQ-32B" code).
- **Sampling:** temperature 0.6, top-p 0.95, min-p 0; top-k between 20 and 40; `presence_penalty` between 0 and 2 can reduce endless repetitions, and higher values may cause language mixing and a slight performance decrease (model card Usage Guideline 2). The quickstart uses `max_new_tokens=32768` (model card "Quickstart"; blog code).
- **Multi-turn history:** earlier assistant turns keep only the final output, not the thinking content; `apply_chat_template` implements this (model card Usage Guideline 3).
- **Benchmark prompts:** math prompts include "Please reason step by step, and put your final answer within \boxed{}."; multiple-choice prompts request a JSON `answer` field with the choice letter (model card Usage Guideline 4).
- **Context length:** full 131,072 tokens; for inputs longer than 8,192 tokens enable YaRN with `rope_scaling` factor 4.0 and `original_max_position_embeddings` 32768; vLLM supports only static YaRN, which may lower quality on shorter texts, so the setting is recommended only when long contexts are required (model card "Introduction", Usage Guideline 5).

### Earlier release: QwQ-32B-Preview (separate artifact)
Blog "QwQ: Reflect Deeply on the Boundaries of the Unknown" (Qwen Team, 2024-11-28, https://qwenlm.github.io/blog/qwq-32b-preview/) and model card https://huggingface.co/Qwen/QwQ-32B-Preview.
- Scores: GPQA 65.2%, AIME 50.0%, MATH-500 90.6%, LiveCodeBench 50.0% (preview blog "Performance").
- Listed limitations: language mixing and code-switching; recursive reasoning loops that produce lengthy responses without a conclusive answer; safety; room for improvement in common-sense reasoning and nuanced language understanding (preview blog "Limitations"; preview model card).
- Metadata `base_model: Qwen/Qwen2.5-32B-Instruct`; context length 32,768 tokens (preview model card). The preview blog does not describe the training method.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| QwQ-32B | 32.5B | pretrain-stable | base model | Qwen/Qwen2.5-32B | model card YAML `base_model` | verified 2026-09-14 | — |
| QwQ-32B | 32.5B | SFT | cold-start data size, composition, epochs, LR | not reported | checked blog text and model card | not reported | — |
| QwQ-32B | 32.5B | RL (stage 1) | domains; reward sources | math and coding; accuracy verifier on final solutions (math), code-execution server on predefined test cases (code); no traditional reward model | blog "Reinforcement Learning" | verified 2026-09-14 | blog: performance in both domains "shows continuous improvement" as episodes progress; no curve or numbers given |
| QwQ-32B | 32.5B | RL (stage 1) | algorithm; KL coefficient; clip ε; samples per prompt; prompts per step; max response length; temperature; number of steps | not reported | checked blog text and model card | not reported | — |
| QwQ-32B | 32.5B | RL (stage 2) | reward sources | general reward model + "some rule-based verifiers" | blog "Reinforcement Learning" | verified 2026-09-14 | qualitative statement only; no stage-1-only comparison numbers |
| QwQ-32B | 32.5B | RL (stage 2) | number of steps | "a small amount of steps"; no count | blog "Reinforcement Learning" | not reported (count) | — |
| QwQ-32B | 32.5B | eval-gate | decoding settings and samples for the release figure | not reported | checked blog figure and model card | not reported | — |

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality:** the stage-2 general RL is reported to raise instruction following, human-preference alignment, and agent performance without a significant math or coding drop (blog "Reinforcement Learning"). The blog gives no before/after numbers, so the size of the effect cannot be read from the source.
- **Negative feedback (degenerate outputs):** greedy decoding causes endless repetitions; a presence penalty reduces them at the cost of possible language mixing (model card Usage Guideline 2). The Preview release lists recursive reasoning loops and language mixing as limitations (preview blog "Limitations").
- **Long context:** 131,072-token context through YaRN factor 4.0 over 32,768 original positions; static YaRN may lower short-text quality (model card Usage Guideline 5).
- **Agentic training:** agent capabilities were integrated into the reasoning model (blog introduction); BFCL 66.4 vs 60.3 for DeepSeek-R1 (blog figure). The blog does not describe the agent training data or environments.
- **Distillation:** the blog and model card do not discuss use of QwQ-32B as a teacher. Documented teacher uses are in the cards listed under Connections.

## Connections
- [[qwen-3]] — the Qwen3 long-CoT cold start generates N candidate responses per query with QwQ-32B and filters them (arXiv:2505.09388 §4.1).
- [[open-thoughts]] — QwQ-32B as teacher gave higher student averages than DeepSeek-R1 (44.2 vs 42.3 with code questions, 44.2 vs 41.6 with math questions) and was chosen for OpenThoughts3 (arXiv:2506.04178 §4.6, Table 8).
- [[sky-t1]] — Sky-T1-32B-Preview training data was generated with QwQ-32B-Preview, rejection-sampled, and rewritten with GPT-4o-mini (Sky-T1 blog "Data Curation Process").
- [[bespoke-stratos]] — used DeepSeek-R1 instead of QwQ as teacher and did not reformat traces (Bespoke-Stratos blog).
- [[scaling-reasoning-losing-control-mathif]] — distills QwQ-32B traces with a correctness filter and an 8,192-token cap before RL (that card, §5.2).
- [[deepseek-r1]] — the 671B comparison model in the release figure; also an outcome-reward RL recipe.
- [[qwen-2.5]] — base model family.

## Verification
- Checked on 2026-09-14 against: https://qwenlm.github.io/blog/qwq-32b/ (2025-03-06, including the "Performance" figure image); https://huggingface.co/Qwen/QwQ-32B (README on main); https://qwenlm.github.io/blog/qwq-32b-preview/ (2024-11-28); https://huggingface.co/Qwen/QwQ-32B-Preview (README on main). Connection loci checked against arXiv:2505.09388v1, arXiv:2506.04178v2, the Sky-T1 blog, the Bespoke-Stratos blog, and the Open-R1 README.
- Corrections to the previous card version:
  - Title and URLs mixed two releases (QwQ-32B-Preview blog and QwQ-32B blog) → the card now describes the QwQ-32B release; the Preview is summarized in a separate subsection.
  - "`<thought>…</thought>` parsing" and "`<|im_start|>assistant\n<thought>…</thought>\n<answer>…</answer>`" → the response starts with `<think>\n` (model card Usage Guideline 1).
  - "temperature 0.7", "T=0.6–0.7, top_p=0.95, max_new_tokens ≈ 16K / ≥ 16K" → temperature 0.6, top-p 0.95, min-p 0, top-k 20–40 (Usage Guideline 2); quickstart `max_new_tokens=32768`.
  - "rule-based verifier (exact-match math, unit-test code). No PRM." → accuracy verifier for math and code-execution server on predefined test cases, instead of traditional reward models; process reward models are not mentioned (blog).
  - "rule-based verifiers for instruction-following and tool-use" in stage 2 → "general reward model and some rule-based verifiers"; the verifiers' tasks are not stated (blog).
  - "De-facto teacher model for Sky-T1, Open-R1, Bespoke-Stratos" → Sky-T1 used QwQ-32B-Preview; Bespoke-Stratos used DeepSeek-R1 instead of QwQ; the Open-R1 README distills from DeepSeek-R1 and does not mention QwQ.
  - "QwQ-32B (final): AIME24 ~79, MATH500 ~97, LiveCodeBench ~63" → AIME24 79.5, LiveCodeBench 63.4; MATH500 is not in the release figure.
  - "Qwen team (Alibaba DAMO)" → "Qwen Team" (blog byline).
- Removed as unsupported by the source: "first fully open-weights reasoning model producing long-CoT o1-style traces at 32B scale"; "cheapest high-quality reasoning teacher" and "catalyzing the $450-recipe distill wave"; "standard open teacher for 2025-era reasoning distillation"; "8,000+ RL training steps"; trace length median ~3K, P95 ~10K, max ~30K, heavy tail to 20K; "~8 sec/trace on 1×H100"; "QwQ traces are ~30% shorter than R1, less likely to code-switch, slightly lower accuracy on hardest AIME"; "more English-dominant than R1"; reflection tokens "Wait", "Hmm"; "not a chat model, weaker than Qwen2.5-Instruct"; "prefer the final QwQ-32B over the Preview for distillation"; SymPy filtering and language-detection filtering recommendations; "20× fewer parameters"; the [[openr1]] teacher link and the [[qwen-long-context-synth]] link.
- Not reported by the source: RL algorithm and hyperparameters, step counts, cold-start SFT data, RL prompt counts, compute, and evaluation decoding settings.
