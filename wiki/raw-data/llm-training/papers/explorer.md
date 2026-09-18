<!-- scope: Explorer: GPT-4o multi-agent synthesis of multimodal web-agent trajectories on live websites, and SFT of Phi-3.5V / Qwen2-VL-7B on them
     see-also: [[agentinstruct]], [[webarena-data]], [[agenttuning]], [[agent-flan]], [[phi-3]], [[magnetic-one]]
-->

# Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents
- **Core Insight:** A GPT-4o pipeline that proposes, refines, summarizes, and verifies tasks while acting on live websites kept 94K of 175K generated trajectories (49K unique URLs, 720K screenshots) at $0.28 per successful trajectory, and fine-tuning Qwen2-VL-7B on about 30K of them raised Mind2Web-Live full task success from 14.5% to 19.3% on 83 tasks (Table 3, App. B, §4, Table 5).
- **Guideline:** When building web-agent SFT data without human annotation, derive the final task description from the executed actions and filter with an LLM verifier whose agreement with humans has been measured, because in this study refinement turned 53K unique proposals into 81K unique task descriptions (Table 9) and the verifier agreed with human labels on 81% of 100 trajectories (Table 10); add in-domain human data when available, since synthetic-only training scored below Mind2Web training data on Multimodal-Mind2Web (Table 6).
- **Authors:** Vardaan Pahuja, Yadong Lu, Corby Rosset, Boyu Gou, Arindam Mitra, Spencer Whitehead, et al. (The Ohio State University, Microsoft Research)
- **Year:** 2025 (arXiv v1 2025-02; v4 2025-05; Findings of ACL 2025)
- **URL:** https://arxiv.org/abs/2502.11357 (project page: https://osu-nlp-group.github.io/Explorer/)
- **Source type:** paper
- **Relevant topics:** web-agent trajectory synthesis, multimodal GUI agents, LLM verifier filtering, agentic SFT, data scaling

## Abstract
Open-source multimodal web agents perform well on offline benchmarks but fall short of human-level ability on live websites; the authors name the lack of large, diverse trajectory-level datasets as a main bottleneck. They build a synthesis recipe that produces over 94K successful multimodal web trajectories spanning 49K unique URLs, 720K screenshots, and 33M web elements, using web exploration and refinement to obtain diverse task intents at an average cost of 28 cents per successful trajectory. Models trained on this data (Explorer) perform well on Mind2Web-Live, Multimodal-Mind2Web, and MiniWob++, and experiments show data scale as a driver of web-agent ability.

## Key Contributions
- A multi-agent pipeline (task proposer, task refiner, task summarizer, task verifier) that builds tasks bottom-up from executed actions (§3.2, Fig. 1); prompts in App. D.
- A dataset of 94K successful trajectories with screenshots (raw and set-of-mark), HTML, and accessibility tree (§1, Table 1, Table 3).
- Explorer-4B (Phi-3.5V) and Explorer-7B (Qwen2-VL-7B), fine-tuned on the synthetic trajectories (plus Mind2Web training data in some Table 6 variants) and evaluated on Mind2Web-Live, Multimodal-Mind2Web, and MiniWob++ (§4-5).
- Analyses of cost (App. B), task diversity (Table 9), verifier accuracy (Table 10), generation failure modes (Fig. 4), and data scaling (Fig. 3).

## Key Figures/Tables to Study
- Figure 1: proposal → refinement → summarization → verification example on Amazon.
- Table 3: dataset statistics; Table 4 and Table B.4: cost comparison and breakdown.
- Tables 5, 6, 8: Mind2Web-Live, Multimodal-Mind2Web, MiniWob++ results; Table A.1: all 104 Mind2Web-Live tasks.
- Table 9 (diversity by stage), Table 10 (verifier confusion matrix), Figure 3 (data scaling), Table A.2 (hyperparameters).

## Technical Details
- Seeds: top 100 URLs from similarweb.com and 49K Tranco URLs; harmful sites filtered (§3.1). For a 4K subset, GPT-4o starts from a Google search query written from the task (§3.1 fn. 3).
- Execution: Playwright; logged metadata are screenshots, HTML, accessibility tree, and actions in grounded and natural-language form (§3). Generation took 50 hours with 60 parallel processes; viewport up to 1980 × 1080 (§3.1).
- All pipeline agents use GPT-4o (§3.2 fn. 4). Pipeline steps (§3.2):
  1. Task proposer: from the homepage screenshot and accessibility tree, writes an abstract task and executes the first action; stops at CAPTCHA, login, or payment pages.
  2. Task refiner: given the current task and action history, predicts the next action and rewrites the task description after each action.
  3. Task summarizer: from all actions and screenshots, writes a high-level task description (what, not how).
  4. Task verifier: from the task, action history, screenshots, and a markdown copy of the last page, labels success or failure; incoherent or misaligned trajectories are discarded (prompt adapted from Pan et al. 2024a, Table D.11).
- Action counts in the dataset: click 415K, scroll 213K, type 62K, goto 26K, select 5K, search_google 4K (Table E.14).
- Statistics over successful trajectories (Table 3): 7.7 steps per trajectory, 46.3 elements per image, 830M tokens, 33.3M elements, 720K images. Step-count difficulty: easy (2-4 steps) 8.2K, medium (5-7) 44.3K, hard (8-12) 41.2K (App. C, Table B.5).
- Cost (App. B): $2.5 per 1M tokens for the model the appendix names "GPT-4o-turbo"; 3.6K text tokens per proposal or refinement step; $0.0028 per image. Total = $0.0128 × 7.7 + $0.02581 + $0.02381 = $0.148, rounded to $0.15 per raw trajectory; with an estimated 53.1% success rate, $0.28 per successful trajectory. Reported comparison: Mind2Web $0.85, AgentTrek $0.55 per trajectory (Table 4).
- Verifier (§6, Table 10), 100 random trajectories: 81% agreement with human judgment; cell shares GT success/pred success 0.39, GT success/pred failure 0.05, GT failure/pred success 0.14, GT failure/pred failure 0.42. Derived: 0.14 / (0.39 + 0.14) = 26% of verifier-accepted trajectories were human-labeled failures.
- Training data (§4): 40K trajectories selected; trajectories with more than two scroll actions removed to avoid a bias toward scrolling; about 30K used. Model input: set-of-mark screenshot plus accessibility tree and action history (Table D.13, Fig. E.3). Explorer-4B has 4.2B parameters (§5.4).
- Mind2Web-Live: key-node evaluation; 83 of 104 tasks on 37 accessible websites; Table 5 values are the maximum over three runs (§4, App. A.1). Full task SR (Table 5): Phi-3.5V 2.4 → Explorer-4B 18.1; Qwen2-VL-7B 14.5 → Explorer-7B 19.3; GPT-4o 25.3; GPT-3.5 15.4; Qwen2-72B-Instruct 15.4; Mistral-7B-Instruct 9.6. On all 104 tasks: Explorer-4B 16.4, Explorer-7B 16.4, GPT-4o 22.1 (Table A.1).
- Multimodal-Mind2Web: 2K tasks, 137 websites, 31 domains; top-50 DeBERTa candidates with the ground-truth element always included; single training and evaluation run (§4, App. A.2). Average step SR over three splits (Table 6): Explorer-7B synthetic only 43.0, Mind2Web only 49.5, both 54.3; AgentTrek-7B (same backbone, synthetic + Mind2Web) 53.2.
- MiniWob++: 46 tasks, zero-shot, average of four runs (§4). Accuracy (Table 8): Explorer-7B 53.26, GPT-4 53.04, Explorer-4B 46.74, AgentTrek-7B 45.28, Synatra-CodeLlama-7B 38.20, Qwen2-VL-7B 36.96.
- In-domain: 100 generated test tasks disjoint from training, judged by the §3.2 verifier; full task SR Explorer-7B 18.0 (base 6.0), Explorer-4B 17.0 (base 1.0), GPT-4o 16.0 (§5.1, Table 7).
- Ablation (Table A.3, backbones fine-tuned on Explorer data, Mind2Web-Live full task SR): text-only Phi-3-mini 13.3 vs Phi-3.5V 18.1; LLaVA-Mistral-7B 4.8.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Explorer-4B, Explorer-7B | 4.2B, 7B | SFT | training trajectories | about 30K (40K selected, >2 scroll actions removed) | arXiv:2502.11357v4 §4 | verified 2026-09-14 | Fig. 3: 25% / 50% / 100% subsets, Explorer-4B, 3 runs (values in figure only) |
| Qwen2-VL-7B / Explorer-7B (Mind2Web-Live runs) | 7B | SFT | batch; epochs; LR; train time | 64; 2 (trajectory-level); 1 × 10^-5; 15 h (Syn.), 1.5 h (M2W), 15.5 h (Syn. + M2W) | v4 Table A.2, App. A.1 | verified 2026-09-14 | no ablation reported |
| Phi-3.5V / Explorer-4B (Mind2Web-Live, Syn. and Syn. + M2W) | 4.2B | SFT | batch; epochs; LR; train time | 64; 2; 4 × 10^-5; 12.5 h each | v4 Table A.2 | verified 2026-09-14 | no ablation reported |
| Phi-3.5V (Mind2Web-Live, M2W only) | 4.2B | SFT | batch; epochs; LR; train time | 64; 2; 1 × 10^-5; 1 h | v4 Table A.2 | verified 2026-09-14 | no ablation reported |
| Explorer-7B / Explorer-4B (Multimodal-Mind2Web, Syn.) | 7B / 4.2B | SFT | batch; epochs; LR; train time | 64; 10; 4 × 10^-5; 17 h / 12 h | v4 Table A.2 | verified 2026-09-14 | no ablation reported |
| Multimodal-Mind2Web M2W and Syn. + M2W runs | 4.2B, 7B | SFT | hyperparameters | not reported (Table A.2 lists only Syn. rows) | v4 Table A.2, Table 6 | not reported | none |
| All runs | 4.2B, 7B | SFT | optimizer, LR schedule, warmup, sequence length, loss masking, GPU count | not reported (GPU type: Nvidia H100) | v4 body, App. A-F | not reported | none |

## Findings relevant to generality, negative feedback, agentic training
- Agentic transfer (Result, single study): models trained only on synthetic trajectories improve over their base models on Mind2Web-Live (live sites) and MiniWob++ (zero-shot), whose tasks were not generated by the pipeline (Tables 5, 8).
- Limit of synthetic-only data (Result, single run): on Multimodal-Mind2Web, synthetic-only Explorer-7B (43.0) is below Mind2Web-only training (49.5); combining both gives 54.3 (Table 6).
- Evaluation choices that affect reported numbers: the main Mind2Web-Live table drops 21 inaccessible tasks and reports the best of three runs; on all 104 tasks both Explorer models reach 16.4 full task SR (App. A.1, Table A.1). The in-domain test uses the same LLM verifier that filtered training data (§5.1).
- Negative samples: trajectories labeled failure are discarded (negative marginal value); no failure is used as content or gradient. The verifier also accepts failures: 0.14 of sampled trajectories were human-labeled failures predicted as success (Table 10).
- Narrowing control: scroll-heavy trajectories were removed to avoid a bias toward excessive scrolling (§4). Most frequent Mind2Web-Live error among 20 sampled failures is task deviation (App. A.4, Fig. A.1).
- Diversity (Interpretation by the authors): multiple trajectories per task description expose the agent to alternative solution paths (§6).

## Connections
- [[agentinstruct]] — earlier Microsoft synthetic-data work cited in §1; shares authors Mitra, Lu, and Rosset.
- [[webarena-data]] — WebArena is cited; concurrent NNetnav explores inside the WebArena sandbox, while Explorer uses live websites (§2).
- [[agenttuning]], [[agent-flan]] — AgentLM and AgentFlan-7B are MiniWob++ baselines (Table 8).
- [[phi-3]] — Phi-3 report is the cited source for Phi-3.5V, the Explorer-4B backbone (§4).
- [[magnetic-one]] — course-level contrast: an inference-time multi-agent web system; not cited by this paper.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2502.11357 (v4, 2025-05-30; arXiv comment "ACL 2025 (Findings)"); project page checked for release links.
- Corrections to the previous card version:
  - Title "Exploration-Driven" → "Exploration-driven" (arXiv title); URL Microsoft Research publication page → arXiv abs page.
  - "large screenshot / element collections" → 720K screenshots, 33.3M elements (Table 3); "cost low enough" → $0.28 per successful trajectory, $0.15 per raw trajectory (App. B).
  - "Separates exploration from refinement" / "decouple task-intent discovery from final trajectory generation" → the refiner rewrites the task while it executes actions, and the summarizer writes the final task from the executed trajectory (§3.2).
  - "Training signal is trajectory-level, not only final answer supervision" → training instances are single steps sampled from trajectories, with action history in the prompt (App. A.1, A.2, Table D.13).
  - "Shows that data scale is a major driver" → Explorer-4B on 25/50/100% of data, all metrics improve, values only in Fig. 3 (§5.5).
- Removed as unsupported by the source: "deps: [[agentinstruct]]"; "Web-agent analogue of the data-scaling story in [[agentinstruct]]".
- Not reported by the source: a dataset download link (the project page links a GitHub repository only), optimizer and LR schedule, sequence length, GPU count, overlap check between synthetic-task websites and benchmark websites.
