<!-- scope: OS-Genesis (Shanghai AI Lab / HKU et al., Dec 2024; ACL 2025) — GUI-agent trajectory synthesis by exploring first and deriving tasks afterward (reverse task synthesis), with a GPT-4o trajectory reward model used for reward-proportional sampling instead of filtering
     deps: [[self-instruct]]
     see-also: [[webarena-data]], [[agenttrek]], [[learn-by-interact]], [[agent-data-protocol]], [[agenttuning]]
-->

# OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis
- **Core Insight:** With 1K synthesized trajectories, Qwen2-VL-7B reaches 17.41% success on AndroidWorld versus 6.25% for the same amount of task-driven data and 9.82% for 1.5K task-driven + self-instruct trajectories; on WebArena it reaches 10.79% versus 7.05% and 5.39% (Table 1, Table 2, §4.2).
- **Guideline:** When synthetic agent trajectories are often incomplete, score them with a graded reward model and sample them in proportion to the score instead of keeping only complete ones, because in the authors' ablation a complete-only labeler gave slight gains on high-level tasks but lower low-level performance than graded sampling (§5.2, Fig. 5); the figure gives no numeric values, so the size of the effect is not reported.
- **Authors:** Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, et al. (Shanghai AI Laboratory, The University of Hong Kong, Johns Hopkins University, Shanghai Jiao Tong University, University of Oxford, HKUST)
- **Year:** 2024 (arXiv v1 2024-12-27; v2 2025-04-30; v3 2025-06-27, ACL 2025 camera ready)
- **URL:** https://arxiv.org/abs/2412.19723
- **Source type:** paper
- **Relevant topics:** GUI agents, agent trajectory synthesis, task synthesis, trajectory reward model, reward-weighted sampling, data diversity, mobile and web benchmarks

## Abstract
GUI agents built on vision-language models (VLMs) need high-quality trajectory data, which is usually collected by human annotation or by executing pre-defined tasks with a model. The authors state that both approaches are costly or cannot guarantee quality, have limited diversity, and differ from real environments. OS-Genesis reverses the order: an agent first interacts with the environment step by step, and tasks are derived retrospectively from the observed interactions. A trajectory reward model then scores the resulting trajectories. Training on OS-Genesis data improves agents on the online benchmarks AndroidWorld and WebArena, and the authors report higher data quality and diversity than existing synthesis methods.

## Key Contributions
- Interaction-driven data construction: rule-based exploration of GUI elements before any task exists (§3.1).
- Reverse task synthesis: GPT-4o maps each ⟨pre-state, action, post-state⟩ triplet to a low-level instruction and then to a high-level instruction (§3.2).
- A GPT-4o Trajectory Reward Model (TRM) that grades trajectories from 1 to 5 and drives reward-proportional sampling (§3.3, Algorithm 1).
- Two SFT objectives, planning training and action training (§4.2, Eq. 1–2), evaluated on AndroidControl, AndroidWorld, and WebArena with three backbones (Tables 1–2).
- Analyses of diversity, TRM vs labeler, data scale, and synthetic vs human data (§5, App. F–G).

## Key Figures/Tables to Study
- Fig. 2 (exploration and reverse synthesis); Fig. 3 (TRM examples, including a 3/5 incomplete trajectory marked "valuable"); Table 1 (AndroidControl, AndroidWorld); Table 2 (WebArena by site); Fig. 4 (diversity); Fig. 5 (reward strategies); Fig. 6 (data scale); Fig. 7–8 (human instructions and human trajectories); Tables 5–6 (TRM agreement).

## Technical Details
**Pipeline (§3)**
1. Exploration in an Android emulator and a Chrome browser (environments built on WebArena and AndroidWorld, footnote 2) with actions CLICK, TYPE, SCROLL. Exploration is rule-based; GPT-4o is called only to produce text for input fields. Output: triplets ⟨s_pre, a, s_post⟩ of screenshots and action (§3.1).
2. Low-level synthesis f_low: GPT-4o maps a triplet to an atomic instruction, for example "click the dropdown to display options". The prompt includes the action, before and after screenshots, and a red box on the interacted element (§3.2, App. C).
3. High-level synthesis f_high: GPT-4o maps a low-level instruction to a goal that could contain it, for example "configure application settings" (§3.2). Mobile prompts request task-oriented or question-oriented goals (Prompt 14); web prompts request information seeking, site navigation, or content modification (Prompt 15).
4. Trajectory collection: GPT-4o executes the high-level instructions in the environment (§3.2).
5. TRM: GPT-4o receives the high-level instruction, the low-level action history, and screenshots of the last three steps, and assigns R ∈ [1, 5] for completion and coherence (§3.3, App. F).

**Reward-proportional sampling (Algorithm 1)**
- P(g_i) = R_i / Σ_{k=1..N} R_k, where g_i is trajectory i, R_i its TRM score, and N the number of trajectories; one trajectory is sampled per training step.
- Worked example (derived): scores 5, 3, 1 give probabilities 5/9, 3/9, 1/9, so a trajectory scored 1 is still drawn with probability 1/9.

**Training objectives (§4.2)**
- Planning: L1 = −Σ log(p_θ(ℓ | s, h_i, c) · p_θ(a | s, h_i, c, ℓ)), where s is the screenshot and a11y-tree input, h_i the high-level instruction, c the history, ℓ the low-level instruction, a the action.
- Action: L2 = −Σ log p_θ(a | s, c, ℓ).

**Main results (Table 1: AndroidWorld SR; AndroidControl-High SR; AndroidControl-Low SR)**
- Qwen2-VL-7B: Task-Driven 6.25 / 38.84 / 71.33; + Self-Instruct 9.82 / 39.36 / 71.51; OS-Genesis 17.41 / 44.54 / 74.17.
- InternVL2-8B: Task-Driven 4.46 / 23.79 / 64.43; + Self-Instruct 5.36 / 23.43 / 64.69; OS-Genesis 16.96 / 35.77 / 71.37.
- InternVL2-4B: Task-Driven 4.02 / 27.37 / 66.48; + Self-Instruct 7.14 / 24.95 / 66.70; OS-Genesis 15.18 / 33.39 / 73.38.
- GPT-4o Zero-Shot (M3A agent) on AndroidWorld: 23.70.
- WebArena overall SR (Table 2): Qwen2-VL-7B Zero-Shot 7.47, Task-Driven 7.05, + Self-Instruct 5.39, OS-Genesis 10.79; InternVL2-8B 0.00 / 4.56 / 7.05 / 9.96; InternVL2-4B 0.00 / 4.98 / 5.81 / 7.88; GPT-4o Zero-Shot 16.25. OS-Genesis is not highest on every site (for example Qwen2-VL-7B Shopping 7.14 vs Zero-Shot 12.50).
- Benchmarks: AndroidWorld 116 tasks in 20 apps, 112 used; WebArena 812 tasks from 241 templates, one test per template (241 tests) (App. A).

**TRM agreement (App. F)**
- Qwen2.5-VL-72B vs GPT-4o as TRM on 200 trajectories: Spearman 0.788 (web), 0.729 (mobile) (Table 5).
- TRM vs human annotators on 100 trajectories: Spearman 0.813 (mobile), 0.798 (web) (Table 6).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OS-Genesis data | — | distill-SFT | synthesis, execution, and TRM model | GPT-4o | arXiv:2412.19723v3 §3, §4.1 | verified 2026-09-14 | open VLMs could not follow instructions or explore online (Limitations); TRM swap to Qwen2.5-VL-72B in Table 5 |
| OS-Genesis agents | 4B, 7B, 8B | SFT | trajectories | 1K (OS-Genesis and Task-Driven); 1.5K (Self-Instruct baseline) | v3 §4.2 | verified 2026-09-14 | Fig. 6: 0–1,500 trajectories on AndroidWorld, gains then saturation |
| OS-Genesis agents | 4B, 7B, 8B | SFT | objectives; sampling | planning L1 + action L2; P(g_i) = R_i / ΣR_k | v3 §4.2 Eq. 1–2; Algorithm 1 | verified 2026-09-14 | Fig. 5: TRM vs no RM vs labeler |
| OS-Genesis agents | 4B, 7B, 8B | SFT | tuning; hardware | VLM full fine-tuning; clusters of 8 × A100 80GB | v3 §4.1 | verified 2026-09-14 | no ablation reported |
| InternVL2-4B / -8B agents | 4B, 8B | SFT | image tiling | max_dynamic_patch 24; 448×448 tiles + thumbnail | v3 App. D | verified 2026-09-14 | no ablation reported |
| Qwen2-VL-7B-Instruct agent | 7B | SFT | image_resolution | 1024 (training and inference) | v3 App. D | verified 2026-09-14 | "produces outstanding results" (App. D); no table |
| all agents | 4B–8B | SFT | a11y tree | filtered to position or index of visible elements | v3 App. D | verified 2026-09-14 | no ablation reported |
| all agents | 4B–8B | SFT | LR, epochs, batch size, sequence length, loss masking, GPU-hours | not reported | checked v3 §4, App. A, B, D | not reported | — |

## Findings relevant to generality, negative feedback, agentic training, distillation
- Out-of-distribution test: AndroidControl covers 833 apps, of which 20 were encountered during synthesis; the authors treat it as an OOD evaluation (§4.3). The authors state that OS-Genesis "excels particularly in the high-level setting" (§4.3); in Table 1 the High-SR gain over the stronger baseline exceeds the Low-SR gain for Qwen2-VL-7B (+5.18 vs +2.66) and InternVL2-8B (+11.98 vs +6.68) but not for InternVL2-4B (+6.02 vs +6.68) (derived).
- Diversity: by average cosine distance of Sentence-BERT embeddings, OS-Genesis has the highest instruction and trajectory diversity among synthetic sets; human data has high instruction diversity but low trajectory diversity (§5.1, Fig. 4). OS-Genesis instructions average 18.01 words (mobile) and 19.68 (web) versus 18.71 for human mobile data (App. G).
- Negative samples (negative marginal value, down-weighted): incomplete trajectories stay in the training pool as ordinary cross-entropy targets with lower sampling probability; the authors argue they still contain useful exploration and form a large share of the data (§3.3). No negative-gradient term is used.
- Scale: AndroidWorld success rises with trajectory count and then saturates; the authors attribute this to VLM capacity and to GPT-4o's ability to complete tasks (§5.3, Fig. 6).
- Human data: trajectories collected by GPT-4o from 500 human-written AndroidControl instructions score below OS-Genesis instructions (§5.4, Fig. 7); against 1K crowdsourced AndroidControl trajectories, OS-Genesis keeps over 80% of the average success rate (§5.4, Fig. 8).

## Connections
- [[self-instruct]] — basis of the Task-Driven w. Self-Instruct baseline (3 in-context demonstrations, App. E.2).
- [[webarena-data]] — WebArena environment and trajectory collection used for web evaluation.
- [[agenttrek]] — web trajectory synthesis from tutorials, a different task source.
- [[learn-by-interact]] — environment-interaction data synthesis for agents.
- [[agent-data-protocol]], [[agenttuning]] — agent-trajectory SFT data standards and training.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2412.19723 (v3, 2025-06-27, ACL 2025 camera ready), body and appendices A–G.
- Audit claims not found in the source: "1,000 OS-Genesis trajectories vs 1,500 for the self-instruct/task-driven baselines" → only the Self-Instruct baseline uses 1.5K; Task-Driven uses 1K (§4.2). "AndroidControl high-level 71.51→74.17 and 64.69→71.37" → these are AndroidControl-Low SR values; High SR is 39.36→44.54 (Qwen2-VL-7B) and 23.43→35.77 (InternVL2-8B) against Self-Instruct (Table 1). The InternVL2-8B AndroidWorld baseline 4.46 is Task-Driven, while the Qwen2-VL-7B baseline 9.82 is Self-Instruct (Table 1).
- Not reported by the source: SFT hyperparameters, number of explored triplets and synthesized instructions, share of incomplete trajectories, numeric values behind Fig. 5–8.
