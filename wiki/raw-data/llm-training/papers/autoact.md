<!-- scope: AutoAct (ZJU, ACL 2024) — QA agent learning from a few task examples plus a tool library: self-instruct, zero-shot trajectory synthesis, reward = 1 filtering, and LoRA "self-differentiation" into plan, tool, and reflect sub-agents
     deps: [[self-instruct]]
     see-also: [[fireact]], [[lumos]], [[agenttuning]]
-->

# AutoAct: Automatic Agent Learning from Scratch for QA via Self-Planning
- **Core Insight:** With Llama-2-70B-chat as backbone and 200 self-synthesized trajectories, AutoAct's three LoRA sub-agents reach 48.47 F1 on HotpotQA and 78.61 accuracy on ScienceQA, above FireAct trained on GPT-4 trajectories (42.70 and 71.94) and above GPT-3.5-Turbo zero-shot planning (44.70 and 72.22) (Table 1).
- **Guideline:** When agent trajectories are synthesized by the model being trained, keep only trajectories with a fully correct answer before fine-tuning, because training on the unfiltered set lowered HotpotQA F1 from 48.47 to 32.51, below the no-fine-tuning result of 32.84 (Table 2, Llama-2-70B-chat).
- **Authors:** Shuofei Qiao, Ningyu Zhang, Runnan Fang, Yujie Luo, Wangchunshu Zhou, Yuchen Eleanor Jiang, et al. (Zhejiang University; ZJU–Ant Group Joint Laboratory of Knowledge Graph; AIWaves Inc.; Alibaba Group)
- **Year:** 2024 (arXiv v1 2024-01, v4 2024-05; ACL 2024 per the arXiv comment)
- **URL:** https://arxiv.org/abs/2401.05268
- **Source type:** paper
- **Relevant topics:** agent SFT, self-generated trajectories, rejection filtering, multi-agent LoRA, ReAct-format planning, QA agents

## Abstract
Language agents for complex question answering often depend on large annotated datasets and on planning trajectories synthesized by closed-source models such as GPT-4, and they require one model to learn every planning function. AutoAct starts from a small set of task examples and a tool library. A Meta-Agent first augments the task data with self-instruct and synthesizes planning trajectories without humans or closed-source models. It then applies a division-of-labor strategy: the Meta-Agent is fine-tuned into a group of sub-agents that together complete the task. Experiments with several open LLMs show performance better than or equal to strong baselines, and the analysis supports the division-of-labor design (abstract, §1).

## Key Contributions
- A pipeline that needs only task name, description, a few QA examples, and a tool library (§2.1, App. E).
- Self-instruct data augmentation and zero-shot Thought-Action-Observation trajectory synthesis by the backbone model itself (§2.2–2.3).
- "Self-differentiation": one LoRA per role — Plan-Agent, Tool-Agent, Reflect-Agent — trained on role-specific slices of the filtered trajectories (§2.3, App. I).
- Comparisons with prompt-based and fine-tuned agent baselines on HotpotQA and ScienceQA, plus ablations of reflection, multi-agent split, fine-tuning, and filtering (Tables 1–2).
- Analyses of data scale, division granularity, and human preference (§5, Figs. 3, 4, 6). Code: github.com/zjunlp/AutoAct (footnote 1).

## Key Figures/Tables to Study
- Fig. 2: the full pipeline from self-instruct to group planning.
- Table 1: main results for Mistral-7B-Instruct-v0.2, Llama-2-13B-chat, Llama-2-70B-chat.
- Table 2: approach ablations (Llama-2-70B-chat).
- Fig. 3: F1 versus training data scale (0–300) and training on trajectories from larger models.
- Table 4 (App. B): LoRA training hyperparameters.

## Technical Details
- **Inputs:** task name M, description P, and |C| QA examples with |C| small (§2.1). The tool library has 15 tools, including BingSearch, Retrieve (Wikipedia), Lookup, Image2Text, Text2Image, and Code Interpreter (App. F, Table 6).
- **Step 1, self-instruct:** the database D starts as C; each round the Meta-Agent generates QA pairs from few-shot examples sampled from D; format-error and duplicate pairs are removed; |D| ≫ |C| (§2.2, App. G.1). Rounds and pairs per round are not reported.
- **Step 2, tool selection:** the Meta-Agent picks tools from the library for the task; the prompt asks for 3 tools (§2.3, App. G.2 Table 8).
- **Step 3, trajectory synthesis:** zero-shot Thought-Action-Observation trajectories on D in ReAct format; trajectories with reward < 1 are removed and only reward = 1 trajectories are kept (§2.3).
- **Rewards:** HotpotQA reward is F1 ∈ [0, 1]; ScienceQA reward is accuracy ∈ {0, 1}. For ScienceQA, image captions replace images during self-instruct (§3).
- **Step 4, self-differentiation:** Plan-Agent outputs the thought τ_t and action name α^m_t (Eq. 2); Tool-Agent outputs action parameters α^p_t (Eq. 3); Reflect-Agent reads the full history H and outputs a reflection thought and action such as Reflect[right] or Reflect[wrong] (Eq. 4, App. I).
- **Group planning at inference:** Tool-Agent is called when Plan-Agent emits a tool name; after an answer, Reflect-Agent judges it; if judged correct the episode ends, otherwise planning continues (§2.3).
- **Evaluation:** 300 HotpotQA dev questions (100 each easy/medium/hard, following BOLAA); ScienceQA test set split into grades 1–4, 5–8, 9–12 with 120 questions each (§3).
- **Fairness settings:** FireAct and AutoAct both train on 200 trajectories; prompt baselines get 2 examples; because Reflexion sees correctness labels, all other methods are run twice and the correct run is used (§3).
- **Main results, All columns (Table 1):** Mistral-7B: AutoAct 38.89 / 70.00 vs FireAct 35.90 / 63.89. Llama-2-13B: 40.49 / 71.39 vs 36.94 / 61.94. Llama-2-70B: 48.47 / 78.61 vs 42.70 / 71.94 (HotpotQA F1 / ScienceQA accuracy).
- **Ablations, Llama-2-70B (Table 2):** −reflection 45.66 / 75.28; −multi (all role data in one model) 42.81 / 69.72; −fine-tuning (zero-shot three agents) 32.84 / 61.94; −filtering 32.51 / 59.17.
- **Planning rounds (App. D Table 5, Llama-2-70B, reflection steps excluded):** AutoAct 4.62 / 4.73 / 4.96 (easy/medium/hard); ReAct 3.83 / 4.02 / 4.13; FireAct 3.01 / 3.17 / 3.70.
- **Human evaluation:** 5 NLP volunteers, blind, majority vote, reflection text removed (App. C). Win rates (%) AutoAct / ReAct / BOLAA / FireAct: overall 32 / 25 / 20 / 23; action type 34 / 22 / 21 / 23; action parameters 30 / 22 / 24 / 24 (Fig. 6).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| AutoAct on Mistral-7B-Instruct-v0.2, Llama-2-7B-chat, Llama-2-13B-chat | 7B; 7B; 13B | SFT (LoRA) | LoRA r; alpha; dropout; target modules | 8; 16; 0.05; q_proj, v_proj | arXiv:2401.05268v4 App. B Table 4 | verified 2026-09-14 | no ablation reported |
| same | 7B; 7B; 13B | SFT (LoRA) | model_max_length; per_device_batch_size; gradient_accumulation_steps; "batch size" | 4096; 2; 1; 4 | App. B Table 4 | verified 2026-09-14 | no ablation reported |
| same | 7B; 7B; 13B | SFT (LoRA) | learning rate; warmup_ratio; epochs | 1e-4; 0.03; 5 | App. B Table 4 | verified 2026-09-14 | no ablation reported |
| AutoAct on Llama-2-70B-chat | 70B | SFT (LoRA) | LoRA r; alpha; dropout; target modules | 8; 16; 0.05; q_proj, v_proj | App. B Table 4 | verified 2026-09-14 | no ablation reported |
| AutoAct on Llama-2-70B-chat | 70B | SFT (LoRA) | model_max_length; per_device_batch_size; gradient_accumulation_steps; "batch size"; learning rate; warmup_ratio; epochs | 4096; 2; 1; 1; 1e-4; 0.03; 3 | App. B Table 4 | verified 2026-09-14 | no ablation reported |
| all AutoAct runs | all | SFT (data) | training trajectories (main comparison); keep rule | 200; reward = 1 only | §3 Baselines; §2.3 | verified 2026-09-14 | Fig. 3: F1 stable beyond 200; Table 2 −filtering |
| all AutoAct runs | all | SFT | LR schedule; optimizer; self-instruct rounds and database size | not reported | checked §2–5, App. B–I | not reported | — |
| all AutoAct runs | all | SFT + eval | framework; compute | FastChat + DeepSpeed; 8 V100 GPUs, training and inference within 16 hours | App. B; §3 | verified 2026-09-14 | — |

## Findings relevant to generality, negative feedback, agentic training, distillation, long context
- **Negative feedback (negative marginal value):** wrong self-synthesized trajectories used as SFT targets hurt more than skipping fine-tuning (Table 2, −filtering vs −fine-tuning). Failed trajectories are discarded, not used as gradient (§2.3). App. I shows a Reflect-Agent target of Reflect[wrong] on a wrong intermediate answer (negative as content); the paper does not say how often such targets occur.
- **Generality:** evaluation covers two QA tasks only; the authors list the QA focus as a limitation (Limitations, "Tasks"). F1 is stable once the training data scale exceeds 200 trajectories (Fig. 3 a–c); the authors attribute this to limited knowledge extracted by self-instruct and to duplicate data, which they say can lead to over-fitting (§5, Interpretation).
- **Agentic training:** splitting roles helps (−multi drop of 5.66 F1), but one agent per tool ("Tool-Specified") is not better and is sometimes worse than a single agent, with larger losses on hard questions (Table 2, Fig. 4).
- **Distillation:** a 7B model trained on trajectories synthesized by 13B and 70B models improves over self-synthesized data (Fig. 3 d–e). At 70B, −multi is comparable to FireAct trained on GPT-4 trajectories (42.81 vs 42.70 HotpotQA F1); the authors read this as 70B trajectory quality "may be no worse than that of GPT-4" (§4, Interpretation).
- **Long context:** in a case study, more planning rounds lengthen the context and the agent drifts from the original question (Fig. 5 d, qualitative).

## Connections
- [[self-instruct]] — the augmentation method used to build database D (§2.2).
- [[fireact]] — fine-tuned single-agent baseline trained on GPT-4 trajectories (§3, Table 1).
- [[agenttuning]], [[lumos]] — fine-tuned agent methods compared in Table 3 (GPT-4 or benchmark trajectories).
- [[star]] — cited as a self-improvement method the authors plan to combine with AutoAct (Limitations).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2401.05268 (v4, full PDF including App. A–I).
- Corrections to the previous card version:
  - "base Llama-7B/13B" → backbones Mistral-7B-Instruct-v0.2 and Llama-2-{7,13,70}B-chat (§3, Tables 1, 4).
  - "Meta-agent classifies its role per turn" → Meta-Agent runs self-instruct, tool selection, and trajectory synthesis, then is LoRA-tuned into three role agents (§2.2–2.3).
  - "keep trajectories matching gold OR self-consistency majority" → keep reward = 1 only; no self-consistency (§2.3).
  - "iterate 3 rounds, saturates at iteration 4" → no iteration; iterative self-improvement is future work (§7).
  - "~5K–10K trajectories, 5–10 turns" → 200 trajectories (§3); 4.62–4.96 average rounds at 70B (Table 5).
  - "tools Search/Lookup/Calculator" → 15-tool library, 3 selected per task (App. F, G.2).
  - "Llama-2-13B HotpotQA EM ~36, within ~4 points of GPT-4 baselines" → metric F1; 40.49 vs FireAct 36.94 (Table 1).
  - "ScienceQA transfer without explicit training" → ScienceQA is a trained target task (§3, App. E).
  - Affiliations corrected (title page); "code + datasets released" → code link only (footnote 1).
- Removed as unsupported by the source: "narrower distribution per role lets a small LoRA specialize" (the paper argues from bounded rationality and Goodhart's law, §4); "no API fees"; self-consistency bias risk; [[spin]] connection (not cited).
- Not reported by the source: optimizer, LR schedule, self-instruct round count and final |D|, iteration results.
