<!-- scope: agent SFT data built by converting existing benchmarks' gold reasoning steps into subgoal/action annotations; two trained modules (planning, grounding) plus a tool-based execution module
     deps: [[agenttuning]]
     see-also: [[fireact]], [[agent-flan]], [[agentinstruct]], [[autoact]]
-->

# Agent Lumos: Unified and Modular Training for Open-Source Language Agents
- **Core Insight:** Agent training data can be produced by prompting GPT-4 to rewrite the gold reasoning steps already present in existing benchmarks into a shared subgoal-and-action format, rather than sampling trajectories from scratch; the resulting 56K annotations train a 7B planning module and a 7B grounding module that reach 27.6 step success rate on Mind2Web against 22.6 for GPT-4 (§4.2, Table 1a).
- **Guideline:** When a target benchmark already contains human-written solutions or structured action sequences (PRM800K, GSM8K, StrategyQA, Mind2Web, A-OKVQA), convert those steps into subgoal and action annotations instead of generating trajectories with an LLM, because on Mind2Web GPT-4 itself reaches only about 20% step success rate and would therefore produce low-quality generated trajectories (§3, citing Liu et al. 2023b). When the subgoal granularity is a choice, use high-level subgoals that map to one or more low-level actions: low-level subgoals scored 63.3 vs 65.7 on StrategyQA and 44.3 vs 45.9 on HotpotQA (§4.5, Table 2).
- **Authors:** Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, Bill Yuchen Lin (Allen Institute for AI; UCLA; University of Washington)
- **Year:** 2023 (arXiv v1 2023-11; ACL 2024 Main Conference; card read against v3, 2024-07)
- **URL:** https://arxiv.org/abs/2311.05657 (code: https://github.com/allenai/lumos; models and data: https://huggingface.co/ai2lumos)
- **Source type:** paper
- **Relevant topics:** agentic SFT data, annotation conversion, modular agents, planning and grounding, cross-task generalization

## Abstract
Lumos is a framework for training open-source LLM-based agents with a unified, modular architecture: a planning module that generates high-level subgoals, a grounding module that translates each subgoal into executable actions, and an execution module that runs those actions with off-the-shelf tools. Training annotations are derived from gold reasoning rationales in existing benchmarks across QA, web, math, and multimodal tasks. On 9 datasets the authors report that Lumos exceeds larger open-source agents on the held-out dataset of each task type, surpasses GPT-based agents on QA and web tasks, outperforms agents trained with chain-of-thought or unmodularized integrated training on the same data, and generalizes to two unseen tasks, outperforming 33B-scale and domain-specific agents.

## Key Contributions
- An annotation-conversion method that uses GPT-4 as a format converter over existing gold reasoning steps, with 4-/5-shot conversion prompts (§3.1).
- Two interaction formulations: Lumos-O produces all subgoals in one pass; Lumos-I produces one subgoal per iteration conditioned on prior execution results (§2.2–2.3).
- A released annotation set: 55,382 planning and 55,499 grounding annotations after filtering (§4.1).
- Evidence that the modular split beats chain-of-thought training and integrated single-module training on the same data and base model (§4.3, Table 3).
- Unseen-task evaluation on WebShop and InterCodeSQL with 2-/3-shot examples in the module inputs (§4.4, Table 1e).

## Key Figures/Tables to Study
- **Figure 2** — the Lumos-O and Lumos-I formulations with worked A-OKVQA and Mind2Web examples.
- **Figure 3** — how converted subgoals, actions, and execution results are laid out as a `<|user|>` / `<|assistant|>` conversation.
- **Table 1** — the five result panels (web, math, multimodal, QA, unseen tasks).
- **Table 2** — annotation-quality ablations: Lumos data vs ReWOO-open data, and high-level vs low-level subgoals.
- **Table 4** — data sources and converted-annotation counts per task type.
- **Table 7** — unified training (Lumos-I_All-13B) against task-specific training.

## Technical Details

### Modules
- Planning module `π_plan`: `S = π_plan(T)` for Lumos-O, or `s_t = π_plan(T, s_1, e_1, …, s_{t−1}, e_{t−1})` for Lumos-I, where `T` is the task description, `s_i` the i-th subgoal, and `e_i` the execution result of the i-th subgoal's actions (§2.2–2.3).
- Grounding module `π_ground`: `A = π_ground(T, I, S)`, where `I = {i_1..i_k}` is the action interface — the list of actions available for the task (§2.2).
- Execution module: not a trained model. It is a program that calls tools — Wikipedia and Google Search APIs, a `dpr-reader-multiset-base` retriever, a GPT-series QA tool, a DeBERTa HTML-tag ranker for web tasks, and the WolframAlpha API for math (§2.1; App. G).
- Both trained modules are full fine-tunes of LLAMA-2-7B or LLAMA-2-13B; the paper trains one model per module (§4.1; App. D).

### Data conversion and scale (§3.1, §4.1, App. B Table 4)
| Task type | Source datasets | # source data | # converted for planning | # converted for grounding |
|---|---|---|---|---|
| Math | PRM800K (10000), GSM8K (7473), ASDiv (2305) | 19778 | 19386 | 19471 |
| QA | Musique (17632), StrategyQA (1777) | 19409 | 19048 | 19080 |
| Web | Mind2Web | 1009 | 1007 | 1007 |
| Multimodal | A-OKVQA | 17056 | 15941 | 15941 |

Totals after filtering for mismatched parentheses, invalid execution outputs, or excessively long outputs: 55,382 planning and 55,499 grounding annotations (§4.1). Converters: GPT-4 versions of 2023-08-13 and 2023-09-13, and GPT-4V of 2023-01-24 for the multimodal set; image captions for multimodal inputs come from LLAVA-1.5-7B (§4.1). Average turns per converted annotation: 4.75 math, 3.75 QA, 8.25 web, 3.92 multimodal (App. B).

### Loss masking
Training uses the conversational layout of Fig. 3 and masks every non-assistant token: `L = − Σ_j log p_π(t_j | t_<j) × 1(t_j ∈ Y)`, where `t_j` is the j-th input token and `Y` the set of assistant-generated tokens (§3.2).

### Reported results (all from Table 1; ⋆ marks fine-tuning on that task's training set)
- Web, Mind2Web step success rate: Lumos-I_Web 7B 27.6, 13B 31.3; GPT-4 22.6, GPT-3.5-turbo 15.7, AgentLM-70B 13.5, WizardLM-30B 3.1.
- Math accuracy: Lumos-O_Math 7B 50.5 GSM8K / 65.5 SVAMP; 13B 55.4 / 69.3; Orca-Platypus-13B 38.4 / 56.9.
- Multimodal accuracy: Lumos-I_MM 7B 71.3 A-OKVQA / 58.4 ScienceQA (IMG); 13B 72.4 / 58.2; MiniGPT-4-13B 67.2 / 42.8.
- QA: Lumos-I_QA 7B with GPT-3.5-turbo as the QA tool, 65.7 StrategyQA and 45.9 LLM-accuracy / 29.4 EM on HotpotQA; ReWOO with GPT-3.5-turbo, 66.6 and 42.4 / 30.4; FiReAct LLAMA-2-7B, 26.2 EM.
- Unseen tasks: Lumos-I_All-13B 50.3 average reward on WebShop and 7.3 success rate on InterCodeSQL; Vicuna-v1.3-33B 23.9 and 6.7; Claude-instant 49.7 on WebShop.
- Unified training (Table 7): Lumos-I_All-13B improves web (31.3 → 31.9), QA, and multimodal over task-specific training, and loses 0.7 and 1.4 points on the two math datasets.
- Instruction following: Lumos annotations mixed with Alpaca score 39.3 ROUGE-L on Super-NaturalInstructions, against 39.8 for LLAMA-2-7B trained on Alpaca alone (§4.5).
- Inference cost (App. E Table 6, 100 instances, 2 A6000 48GB, batch 16): Lumos-O 102s GSM8K / 556s HotpotQA; Lumos-I 851s / 1007s.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Lumos planning and grounding modules | LLAMA-2-7B and 13B | SFT | epochs | 2 | arXiv:2311.05657v3 App. D | verified 2026-09-18 | no ablation reported |
| same | same | SFT | peak LR / warmup | 2 × 10⁻⁵, linear warmup over 3% of total steps | App. D | verified 2026-09-18 | no ablation reported |
| same | same | SFT | batch size (sequences) | 128 | App. D | verified 2026-09-18 | no ablation reported |
| same | same | SFT | max sequence length | 1024 | App. D | verified 2026-09-18 | no ablation reported |
| same | same | SFT | hardware | 2 × A100 80GB, or 4 × A6000 48GB | App. D | verified 2026-09-18 | no ablation reported |
| same | same | SFT | loss masking | assistant tokens only, `1(t_j ∈ Y)` | §3.2 | verified 2026-09-18 | no ablation reported |
| same | 7B | SFT | annotations (task-specific math agent) | 19778 math-domain converted items | App. B | verified 2026-09-18 | §4.5 Table 2 compares annotation sources at 2,000 items |
| same | 7B / 13B | SFT | annotations (unified agent) | 55,382 planning / 55,499 grounding | §4.1 | verified 2026-09-18 | App. F Table 7: unified training vs task-specific |

## Findings relevant to generality, agentic training, and distillation
- Held-out datasets within a trained task type (SVAMP for math, ScienceQA for multimodal, HotpotQA for QA) are not in the training annotations, and Lumos exceeds the listed baselines on them (§4.2, Table 1).
- Unseen task types (WebShop, InterCodeSQL) need 2-/3-shot examples in the module inputs to expose the new action set; under that setup unified training beats task-specific training (§4.4, Table 1e).
- Unified training across the four task types helps web, QA, and multimodal and costs 0.7–1.4 points on math (App. F, Table 7).
- Distillation shape: GPT-4 is used only as a format converter over existing gold rationales, not as a solution generator; the authors motivate this by GPT-4's own weak Mind2Web performance (§3).
- Mixing agent annotations with Alpaca leaves Super-NaturalInstructions within 0.5 ROUGE-L of Alpaca-only training, so agent training and general instruction following coexist in these runs (§4.5).
- Not reported: cost of annotation conversion, false-conversion rate, negative or failed trajectories, RL on top of the SFT modules.

## Connections
- [[agenttuning]] — contemporary agent SFT work; AgentLM baselines appear in Table 1.
- [[fireact]] — ReAct-trajectory fine-tuning baseline in the QA panel of Table 1.
- [[agent-flan]], [[agentinstruct]] — later agent-data pipelines that also reformat rather than free-generate trajectories.
- [[autoact]] — AutoAct-7B, the ScienceQA-specific agent compared in §4.2 (67.3 vs 53.3 on the full ScienceQA test set).

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2311.05657 (arXiv v3, 10 Jul 2024)
- Corrections to the previous card version:
  - Title "Lumos: Learning Agents with Unified Data, Modular Design, and Open-Source LLMs" → "Agent Lumos: Unified and Modular Training for Open-Source Language Agents" (title page, v3).
  - Author "Khyathi Raghavi Chandu" → "Khyathi Chandu" (title page).
  - "Execution" described as a trained third module → the execution module is a program calling off-the-shelf tools; only planning and grounding are trained (§2.1; App. G).
  - "Each module can be a separate head or a separate LoRA" → both modules are full fine-tunes of LLAMA-2-7B/13B (§4.1; App. D).
  - Training sources listed as HotpotQA, ALFWorld, WebShop, Mind2Web, ScienceQA, StrategyQA, Musique, GSM8K, MATH → the four conversion sources are PRM800K, GSM8K, ASDiv (math), Musique, StrategyQA (QA), Mind2Web (web), A-OKVQA (multimodal); HotpotQA and ScienceQA are held-out, WebShop is an unseen task, ALFWorld appears only in Table 5 as a possible future extension, and MATH is not used (§4.1; App. B Tables 4–5; §4.4).
  - "~40K task instances … ~200K training turns" → 55,382 planning and 55,499 grounding annotations from 57,252 source items (§4.1; App. B Table 4).
  - "Lumos-13B (onetime): HotpotQA 39.3 EM — beats LLaMA-2-13B-chat ReAct (30.5) and approaches GPT-4 ReAct (44)" → Lumos-O_QA 7B scores 39.2 LLM accuracy / 24.9 EM on HotpotQA; none of the three quoted comparison numbers appear in the paper (Table 1d).
  - "Lumos-7B on Mind2Web: competitive with 30B-class alternatives" → Lumos-I_Web 7B scores 27.6 against WizardLM-30B 3.1 and GPT-4 22.6 (Table 1a).
  - "Trajectory length: avg 6–10 steps" → average turns per annotation are 4.75 math, 3.75 QA, 8.25 web, 3.92 multimodal (App. B).
  - Action grammar given as `Search, Retrieve, Calculate, Click, Type, Back, Finish` → action interfaces are per task type; the QA examples use `KnowledgeQuery`, `ParagraphRetrieval`, `QA`, `Calculator`, and WebShop uses `Search`, `FeatureRetrieve`, `Pick`, `Click` (§2.1, §4.4 footnote 4).
- Removed as unsupported by the source: "~$15K GPT-4 API" conversion cost; "generalization to unseen task: ~8-point drop only, vs ~20-point drop for monolithic ReAct fine-tunes"; "total token length 1K–5K" per trajectory; "three serial forward passes per step"; the claim that modules cannot be optimized jointly without full-trajectory RL.
- Not reported by the source: annotation cost, conversion failure analysis beyond the filtering rules, RL fine-tuning, per-module parameter sharing.
