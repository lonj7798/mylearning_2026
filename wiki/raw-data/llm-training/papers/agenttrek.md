<!-- scope: AgentTrek (HKU, Salesforce Research; ICLR 2025): harvest GUI tutorials from RedPajama, replay them with a GPT-4o agent in real websites (BrowserGym), filter with a GPT-4o evaluator; 10,398 trajectories used to fine-tune Qwen2.5 text agents and Qwen2-VL vision agents
     deps: [[webarena-data]]
     see-also: [[explorer]], [[os-genesis]], [[learn-by-interact]], [[agent-data-protocol]], [[agenttuning]], [[fineweb]]
-->

# AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials
- **Core Insight:** Replaying web tutorials mined from RedPajama with a GPT-4o agent and keeping the runs a GPT-4o evaluator marks successful produced 10,398 trajectories at $0.551 each, and fine-tuning Qwen2.5-32B-Instruct on 6,000 text trajectories gave 22.40% WebArena success vs 13.10% for GPT-4o (Table 1; App. C; §2.3.3; Table 4).
- **Guideline:** When an LLM agent synthesizes web trajectories, give it step-by-step tutorial instructions and a target URL rather than only a high-level goal, because in a 400-task comparison tutorial-guided replay produced 208 effective trajectories (52%) vs 63 (15.78%) with goals only (§4 "Realism"; App. B).
- **Authors:** Yiheng Xu, Dunjie Lu, Zhennan Shen, Junli Wang, Zekun Wang, Yuchen Mao, et al. (University of Hong Kong; Salesforce Research)
- **Year:** 2024 (arXiv v1 2024-12; v2 2025-03; ICLR 2025, listed as Spotlight in the arXiv comments)
- **URL:** https://arxiv.org/abs/2412.09605
- **Source type:** paper
- **Relevant topics:** agent trajectory synthesis, web and GUI agents, LLM-labeled classifier filtering, VLM-as-judge filtering, agent SFT, visual grounding, synthetic vs human-annotated data

## Abstract
GUI agents lack high-quality multi-step trajectory data, and human annotation is expensive. AgentTrek is a three-stage synthesis pipeline: (1) it harvests and filters tutorial-like texts from the internet with a classifier, (2) it converts them into structured task specifications with step-by-step instructions, and (3) a VLM agent executes the instructions in real web environments while a VLM evaluator checks trajectory correctness. The trajectories include text observations (HTML/accessibility tree) with function-call actions and screenshot observations with pixel-level actions, plus chain-of-thought reasoning. Models trained on the data reach state-of-the-art results on WebArena, ScreenSpot Web, and Multimodal-Mind2Web in the authors' comparisons, at $0.55 per trajectory without human annotators (Abstract).

## Key Contributions
- Tutorial harvesting at corpus scale: rule pre-filter, GPT-4o-mini labels, FastText classifier (§2.1, Figs. 3-4).
- Guided replay in BrowserGym with a recorded schema of screenshots, video, Playwright traces, DOM snapshots, and reasoning (§2.2.1-2.2.2).
- A GPT-4o trajectory evaluator validated against human labels (§2.2.3; Table 2; App. D).
- Training of text-based (Qwen2.5) and vision-based (Qwen2-VL) web agents and a per-phase cost breakdown (§2.3; Table 3; App. C).

## Key Figures/Tables to Study
- Fig. 4: data flow counts. Table 1: comparison with other trajectory datasets.
- Table 2 / App. D: evaluator accuracy. Table 3 / App. C: cost per phase and cost formula.
- Tables 4-6: WebArena, ScreenSpot Web, Multimodal-Mind2Web. Fig. 7 / Table 9: data-scaling curve vs human-annotated Mind2Web.
- App. Fig. 10: evaluator prompt and success rules. App. H, Fig. 13: a failed replay.

## Technical Details
- **Pre-filter (§2.1.1; App. G.1):** keyword matching (action verbs, UI elements, platform terms), length 200-5,000 words, URL patterns such as "how-to"; App. G.1 requires at least 20 common keywords, at least 4 distinct keywords, and one mandatory keyword appearing at least twice. Recall 92.69% on 285 manually labeled samples; RedPajama reduced from 20.8 billion to 68.8 million entries (§2.1.1; Fig. 4).
- **LLM labeler (§2.1.2; App. G.2):** GPT-4o-mini outputs 1 (GUI tutorial) or 0; F1 88.5% against human annotations; labeled 90,000 pre-filtered entries.
- **FastText filter (§2.1.3):** trained on the 90,000 LLM labels with a 95:5 train-test split; F1 89.5%; applied to 68.8 million entries, it kept about 18.8 million deduplicated likely tutorials (a 72.7% reduction).
- **Standardization (§2.1.4; App. G.3):** GPT-4o-mini extracts the first tutorial in a page into fields for platform, target type and object, target URL, task description, prerequisites, step list, and expected result; about $0.89 per 1,000 entries.
- **Guided replay (§2.2.2):** BrowserGym in Chromium; the agent sees viewport screenshots and the accessibility tree (AXTree), not full HTML; actions use Playwright functions (click, select option, clear). Token use averages about 8,027 per step and 86,114 per task; 1,000 tasks with GPT-4o-08-06 cost about $215.
- **Evaluator (§2.2.3; App. D, Fig. 10):** GPT-4o receives the task description and the interleaved reasoning and actions {d, r1, a1, ..., rn, an}, and gives a trajectory-level verdict, a step-level analysis, and the earliest failure point. Human review of 1,081 trajectories gave a 558-sample gold set; accuracy on replayed web tutorials is 84.0%. Table 2 also lists evaluators on WebArena results for comparison: GPT-4V 80.6%, Captioner + GPT-4 82.1%, Captioner + Mixtral 74.4%. The prompt counts some partial runs as success, for example a trajectory with over 8 correct actions, one of two subtasks completed, or a missing final save.
- **Cost (App. C):** per 1,000 entries: tag and paraphrase $0.886 (gpt-4o-mini), replay $215.359 and evaluator $3.104 (gpt-4o-2024-08-06). With a web-tutorial ratio of 0.275 and replay success rate 39.9%, cost per trajectory = T&P price / web ratio + (replay price + evaluator price) / success rate, giving $550.75 per 1,000 verified trajectories.
- **Dataset (Table 1; §4):** 10,398 trajectories, 127 websites, 11 task categories, 12.1 average steps; §4 states they came from 23,430 filtered tutorials. The "Comparison with Existing Datasets" paragraph instead says "nearly 5,000 verified trajectories"; the paper does not reconcile the two counts.
- **Vision agent (§2.3.1, §2.3.3; App. E):** Qwen2-VL-7B with a NaViT image encoder; a 720p screenshot costs about 1,200 tokens vs about 4,000 for HTML; Playwright actions are mapped to pyautogui commands (Table 8); fine-tuned on 10,000 trajectories.
- **Text agent (§2.3.2-2.3.3; App. F):** Qwen2.5-7B-Instruct and 32B-Instruct fine-tuned on 6,000 trajectories with AXTree observations and Playwright actions.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct / Qwen2.5-32B-Instruct w/ AgentTrek | 7B, 32B | distill-SFT | Training trajectories | 6,000 (AXTree observations, Playwright actions) | arXiv:2412.09605v2 §2.3.3 | verified 2026-09-14 | no ablation reported |
| Qwen2-VL-7B w/ AgentTrek | 7B | distill-SFT | Training trajectories | 10,000 (screenshots, pyautogui actions) | §2.3.3 | verified 2026-09-14 | Table 9: 20% → 100% of data raises cross-domain Step SR 39.5 → 45.0 |
| Qwen2-VL-7B + AT + M2W | 7B | distill-SFT | Data mixture | AgentTrek + Mind2Web training split; ratio not reported | Table 6; §3.2 | not reported (ratio; checked §2.3, §3, App. A-K) | Table 6: +AT+M2W > +M2W > +AT on Step SR in all three splits |
| Trajectory generator / judge | — | distill-SFT | Teacher; judge; standardizer | gpt-4o-2024-08-06; gpt-4o-2024-08-06; gpt-4o-mini | App. C Table 7 | verified 2026-09-14 | Table 2: judge 84.0% accuracy on 558 human-labeled samples |
| All AgentTrek-trained models | 7B, 32B | distill-SFT | LR, epochs, batch size, sequence length, loss masking, replay sampling temperature | not reported | checked §2.3, §3, App. A-K | not reported | — |

## Findings relevant to generality, negative feedback, agentic training, and distillation
- **Held-out websites (Result, single study; Table 4; App. F):** WebArena uses self-hosted sites not seen in training. Qwen2.5-7B-Instruct 3.80 → 10.46 with AgentTrek; Qwen2.5-32B-Instruct w/ AgentTrek 22.40 vs GPT-4o 13.10 and GPT-4 14.41. Table 4 has no untrained Qwen2.5-32B-Instruct row.
- **Visual grounding (Table 5, ScreenSpot Web):** Qwen2-VL-7B average 30.7 → 67.4 (text 35.2 → 81.7; icon/widget 25.7 → 51.5), vs GPT-4 + OmniParser 67.0.
- **Offline web tasks (Table 6, Multimodal-Mind2Web Step SR, cross-task / cross-website / cross-domain):** +AT 40.9 / 35.1 / 42.1; +M2W 50.9 / 44.9 / 47.7; +AT+M2W 55.7 / 51.4 / 52.6. The untrained Qwen2-VL-7B was excluded for insufficient grounding (§3.2).
- **Synthetic vs human data (Table 9; Fig. 7):** cross-domain Step SR rises from 39.5 (20% of AgentTrek) to 45.0 (100%), below the 47.7 of human-annotated Mind2Web training data; cross-website is 38.0 at 80% and 37.5 at 100%.
- **Negative samples:** replays the evaluator marks as failures are discarded (negative marginal value; §2.2.3). The replay success rate is 39.9% (App. C). App. H (Fig. 13) shows a failed replay that the caption attributes to tutorial expiration: the agent could not complete a restaurant booking for the date the tutorial specified. The evaluator confusion matrix is in App. Fig. 9; false-positive and false-negative rates are not stated in the text.
- **Distillation setup:** stage = SFT data for web agents; prompts = GUI tutorials mined from RedPajama; teacher = GPT-4o replaying tutorials; quality control = GPT-4o evaluator; students = Qwen2.5 and Qwen2-VL (§2; App. C).

## Connections
- [[webarena-data]]: WebArena is the held-out text-agent benchmark (Table 4).
- [[explorer]], [[os-genesis]], [[learn-by-interact]]: other web or GUI trajectory synthesis pipelines.
- [[agent-data-protocol]]: unified format for agent SFT datasets of this kind.
- [[agenttuning]]: earlier agent SFT on GPT-4 trajectories filtered by reward.
- [[fineweb]]: LLM-labeled classifier filtering of web text, the same pattern as §2.1.2-2.1.3.
- [[judge-llm-bias]]: reliability limits of LLM judges, relevant to the GPT-4o evaluator.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2412.09605 (arXiv v2, PDF; v1 dated 2024-12-12 per the abstract page).
- Audit claims not found in the source: "80.6% on WebArena results" for the GPT-4o evaluator (Table 2 and App. D give 80.6% for a GPT-4V evaluator in a comparison on WebArena results; the AgentTrek evaluator's reported accuracy is 84.0%); "Mind2Web data: 52.6% step success rate" without a split (52.6 is the cross-domain Step SR of Qwen2-VL + AgentTrek + Mind2Web; cross-task is 55.7 and cross-website 51.4).
- Not reported by the source: fine-tuning hyperparameters, replay sampling temperature, the AgentTrek/Mind2Web mixing ratio, and per-category trajectory counts.
