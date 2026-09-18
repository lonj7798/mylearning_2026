<!-- scope: fine-tuning LMs as ReAct agents on GPT-4 trajectories from multiple prompting methods (CoT, ReAct, Reflexion) and QA tasks
     deps: [[agenttuning]]
     see-also: [[lumos]], [[agent-flan]], [[autoact]]
-->

# FireAct: Toward Language Agent Fine-tuning
- **Core Insight:** Fine-tuning on 500 GPT-4 ReAct trajectories raises HotpotQA EM from 14.8 to 26.2 for Llama-2-7B (+77%) and from 31.4 to 39.2 for GPT-3.5 (Table 2); adding CoT trajectories raises GPT-3.5 to 41.0, but adding more methods does not always help and the best mix depends on the base LM (§6, Table 4, Fig. 4).
- **Guideline:** When fine-tuning an agent on trajectories from several prompting methods, select the method mix per base model on held-out evaluation, because ReAct+CoT helped GPT-3.5 and Llama-2 but hurt CodeLlama, and ReAct+CoT+Reflexion was worst for CodeLlama-7B/13B and best for CodeLlama-34B (§6, Fig. 4).
- **Authors:** Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, Shunyu Yao (System2 Research; University of Cambridge; Monash University; Princeton University)
- **Year:** 2023 (arXiv v1 2023-10)
- **URL:** https://arxiv.org/abs/2310.05915 (code, data, models: https://fireact-agent.github.io)
- **Source type:** paper
- **Relevant topics:** agent fine-tuning, ReAct trajectories, multi-method and multi-task data mixing, trajectory distillation from GPT-4, robustness to noisy tools

## Abstract
Most language agents use few-shot prompting of off-the-shelf LMs. The paper studies fine-tuning LMs to act as agents, using question answering with a Google search API. Across base LMs, prompting methods, fine-tuning data, and QA tasks, fine-tuned agents improve over prompted ones; for example, fine-tuning Llama2-7B on 500 GPT-4 trajectories gives a 77% HotpotQA increase. The paper proposes FireAct, which fine-tunes on trajectories from multiple tasks and prompting methods, and reports that more diverse fine-tuning data can further improve agents. It also reports findings on scaling, robustness, generalization, efficiency, and cost.

## Key Contributions
- FireAct: GPT-4 trajectories from CoT, ReAct, and Reflexion prompts, filtered to successful ones and converted to ReAct format, used to fine-tune a smaller LM (§3, Fig. 2).
- Controlled study in three steps: single task + single method (§5), multiple methods (§6), multiple tasks (§7).
- Measurements of inference cost, robustness to noisy search results, and transfer to Bamboogle (§5.1-5.2, Table 3).
- Code, data, and models released (p. 1 footnote).

## Key Figures/Tables to Study
- Table 2: prompting vs fine-tuning for six LMs. Table 3: cost, robustness, Bamboogle transfer for GPT-3.5.
- Fig. 3: EM vs number of trajectories. Fig. 4 and Table 4 / Table 10: method mixes. Table 5: multi-task results.

## Technical Details
- **Trajectory format.** ReAct rounds of thought, action, observation (§3). Actions in the prompt: `search[question]` on Google and `finish[answer]` (App. C). The search tool is built with SerpAPI and returns the first of answer box, answer snippet, highlight words, or first result snippet (§4).
- **CoT data.** A CoT trajectory becomes a one-round ReAct trajectory: the reasoning is the thought and the action returns the answer (§3).
- **Reflexion data.** Reflections are prompted at the 6th and 10th ReAct round (§3).
- **Inference.** The fine-tuned model runs without few-shot examples and implicitly selects a method (§3, Fig. 2b). No method label or method-specific system prompt is described.
- **Filtering.** Incorrect trajectories are removed (Fig. 2a). The 500 ReAct trajectories were generated "with human-in-the-loop validation" (App. B.2).
- **Tasks.** HotpotQA: 2,000 random train questions for data generation, 500 random dev questions for evaluation (§4). Bamboogle: 125-question test set, evaluation only. StrategyQA (yes/no) and MMLU (57 multiple-choice tasks) (§4). Metric: exact match (EM) (§4); F1 also computed (App. B.2).
- **Models.** GPT-4 generates all data; GPT-3.5 is fine-tuned and prompted; both used July-Sept 2023 (§4). Llama-2-7B/13B chat and CodeLlama-7B/13B/34B instruct (§4).
- **Prompting baselines (Table 1).** GPT-4: IO 37.2, CoT 45.0, ReAct 42.0. GPT-3.5: IO 22.4, CoT 28.0, ReAct 31.4.
- **ReAct prompting → FireAct fine-tuning (Table 2).** Llama-2-7B 14.8 → 26.2; Llama-2-13B 21.2 → 34.4; CodeLlama-7B 17.4 → 27.8; CodeLlama-13B 20.8 → 29.0; CodeLlama-34B 22.2 → 27.8; GPT-3.5 31.4 → 39.2. Standard errors 1.59-2.18 (App. A.1, Table 7).
- **Efficiency (Table 3).** GPT-3.5 per trial: fine-tuned $2.2×10⁻³, 2.7 s; prompted $2.6×10⁻³, 9.0 s. §5.1 calls this a 70% time reduction; §1 says 4x.
- **Data scale (§5.3, Fig. 3).** n ∈ {100, 200, 500, 1000}. GPT-3.5 reaches EM about 35 with 100 samples. Llama models do not learn the ReAct format with 100 or 200 samples. Llama-2-13B with 1,000 samples matches GPT-3.5 with 100.
- **LoRA vs full (App. A.5, Table 11).** Llama-2-7B single-task EM: LoRA 26.2, full 30.2. Adapter 38 MB vs 14 GB.
- **Base LM (§5.3).** CodeLlama-7B > Llama-2-7B, but CodeLlama-13B < Llama-2-13B. CodeLlama-34B < CodeLlama-13B with ReAct-only data.
- **Multi-method, GPT-3.5 (§6, Table 4).** Data: 500 ReAct + 187 CoT + 47 Reflexion trajectories. EM (mean turns, σ): ReAct 39.4 (3.2, 1.4); +CoT 41.0 (2.7, 1.7); +Reflexion 38.8 (3.8, 2.8); +CoT+Reflexion 40.0 (3.0, 4.8). Random method choice per question 32.4; oracle best method 52.0. Table 10 adds IO data: ReAct+IO+Reflexion 41.2.
- **Multi-task, GPT-3.5 (§7, Table 5).** Data: HotpotQA 500 ReAct / 277 CoT; StrategyQA 388 / 380; MMLU 456 / 469. Columns HotpotQA / StrategyQA / Bamboogle / MMLU: HotpotQA-only 39.2 / - / 44.0 / -; multi-task ReAct 39.2 / 55.5 / 43.2 / 63.2; multi-task + CoT 39.6 / 72.9 / 50.4 / 65.8. IO prompting scores 68.6 on MMLU, above every fine-tuned agent.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| FireAct GPT-3.5 | not reported | distill-SFT | Base; method | GPT-3.5-Turbo-0613, OpenAI fine-tuning API | arXiv:2310.05915v1 App. B.2 | verified 2026-09-14 | no ablation reported |
| FireAct GPT-3.5 | not reported | distill-SFT | Epochs | 3 | App. B.2 | verified 2026-09-14 | App. A.8 Fig. 7: EM and F1 rise from 1 to 4 epochs; F1 plateaus after 4 |
| FireAct Llama-2 / CodeLlama | 7B-34B | distill-SFT | Base checkpoints | Llama-2-Chat-HF; CodeLlama-Instruct-HF | App. B.2 | verified 2026-09-14 | no ablation reported |
| FireAct Llama-2 / CodeLlama | 7B-34B | distill-SFT | Method; epochs | LoRA with int8 quantization; 30 | App. B.2 | verified 2026-09-14 | LoRA vs full at 7B: 26.2 vs 30.2 EM (Table 11); LoRA chosen for cost (§5.3) |
| FireAct Llama-2 / CodeLlama | 7B-34B | distill-SFT | Learning rate; batch size (unit not stated) | 3e-4; 16 | App. B.2 (sentence follows the LoRA setup; scope for GPT-3.5 not stated) | verified 2026-09-14 | no ablation reported |
| FireAct (all) | all | distill-SFT | Single-task data | 500 successful GPT-4 ReAct trajectories (HotpotQA) | §5; App. B.2 | verified 2026-09-14 | Fig. 3: n ∈ {100, 200, 500, 1000} |
| FireAct (all) | all | eval-gate | Max ReAct steps; eval temperature | 11; 0.6 (Llama, CodeLlama), 0 (GPT-3.5); int8 at eval | App. B.2 | verified 2026-09-14 | no ablation reported |
| FireAct Llama-2 / CodeLlama | 7B-34B | distill-SFT | Compute | 1× RTX 4090 24GB (7B, 13B); 1× RTX 6000 Ada 48GB (34B); full FT 7B: 4× A100 80GB | App. B.3 Table 15 | verified 2026-09-14 | throughput 0.9-5.5 examples/s LoRA, 19.7 full (Table 15) |

Not reported (checked body, App. A-C): LoRA rank and target modules, warmup and schedule, sequence length, token counts, GPT-4 data-generation cost. §8 gives about $10 per GPT-3.5 fine-tuning experiment. HotpotQA CoT trajectories number 187 in §6 and 277 in §7; the paper does not explain the difference.

## Findings relevant to generality, negative feedback, agentic training, distillation
- **Transfer to an unseen task.** GPT-3.5 fine-tuned on HotpotQA scores 44.0 EM on Bamboogle vs 40.8 for ReAct prompting (Table 3). The authors state that HotpotQA fine-tuning "could hardly generalize" to StrategyQA (yes/no) or MMLU (multiple choice) (§5.2): Llama-2-7B fine-tuned on HotpotQA with few-shot ReAct scores 52.0 on StrategyQA vs 59.0 for the vanilla model (App. A.9, Table 14).
- **Adding tasks.** Adding StrategyQA and MMLU data leaves HotpotQA at 39.2 and Bamboogle at 43.2 vs 44.0 (Table 5). The authors read this as no negative cross-task influence (Interpretation, §7). Only three QA tasks were tested (§8).
- **Robustness to noisy tools (Table 3, §5.2).** With probability 0.5 the search returns "None" or a random past response. "None": ReAct 31.4 → 20.8 (−33.8%), FireAct 39.2 → 33.6 (−14.2%). Random: ReAct → 22.6 (−28.0%), FireAct → 37.2 (−5.1%).
- **Negative samples.** Failed GPT-4 trajectories are discarded (Fig. 2a): negative marginal value only. No failure is used as content or gradient.
- **Distillation.** The paper frames FireAct as distillation from GPT-4 into a smaller LM (§3). Fine-tuned GPT-3.5 (39.2) exceeds GPT-4 IO prompting (37.2) but not GPT-4 CoT (45.0) or ReAct (42.0) (§5.1).
- **Limits (§8).** One task type (QA), one tool (Google search), best fine-tunable model GPT-3.5.

## Connections
- [[agenttuning]] — concurrent agent-trajectory SFT work that mixes general instruction data; FireAct is not cited there.
- [[agent-flan]] — uses FireAct-7B as a baseline in its Table 1.
- [[autoact]] — compares against FireAct trained on GPT-4 trajectories (AutoAct Table 1).
- [[lumos]] — later open agent-training work with modular planning/grounding.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2310.05915 (arXiv v1, 9 Oct 2023, the only version; full PDF incl. App. A-C).
- Corrections: "FireAct-13B reaches 39.3% HotpotQA EM" → Llama-2-13B FireAct 34.4; GPT-3.5 FireAct 39.2 (Table 2). "Llama-2-13B-Chat ReAct baseline 19" → 21.2 (Table 2). "CoT-only 38.9, ReAct-only 37.3, Reflexion-only 35.2, mix 40.0 (strict improvement)" → GPT-3.5: ReAct 39.4, +CoT 41.0, +Reflexion 38.8, all three 40.0; the three-way mix is not best (Table 4). "Wikipedia search API; Search/Lookup/Finish" → Google search via SerpAPI; `search[...]`, `finish[...]` (§4, App. C). "Reflexion retries up to N=3" → reflections prompted at rounds 6 and 10 (§3). "trajectories labeled with method name; method-specific system prompts" → all converted to ReAct format; method chosen implicitly (§3, Fig. 2). "Bamboogle as a seed source" → evaluation-only test set (§4). "FireAct dataset of 2K+ trajectories" → 2,000 HotpotQA questions used for generation; 500 ReAct + 187 CoT + 47 Reflexion trajectories in §6 (§4, §6). "Princeton + Cambridge" → also System2 Research and Monash University (title page). "7B/13B checkpoints released" → "code, data, and models" released, sizes not listed (p. 1). "First systematic study" → paper calls itself "an initial step toward a more systematic study" (§1).
- Removed as unsupported: "~$3K GPT-4 API cost"; "avg ~800 tokens (CoT) to ~2K (Reflexion)"; "3-10 steps; 500-3000 tokens"; "drop trajectories that exceed token budget"; "method diversity compensates for volume"; "Reflexion costs 3-5× ReAct"; "transfer to web/code agents unclear" as a paper claim; "Agent-FLAN scales this idea to millions".
