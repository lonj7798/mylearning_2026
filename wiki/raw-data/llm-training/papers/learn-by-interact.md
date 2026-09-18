<!-- scope: Learn-by-interact (arXiv:2501.10893) — documentation-conditioned self-instruct tasks, LLM rollouts in real environments, backward construction (relabel every sub-trajectory with an instruction it accomplishes), duplicate-state and unanimous LLM-committee filtering; data used for agentic-retrieval ICL and LoRA SFT on SWE-bench, WebArena, OSWorld, Spider2-V
     deps: [[self-instruct]]
     see-also: [[os-genesis]], [[agenttrek]], [[agenttuning]], [[webarena-data]], [[agent-early-experience]]
-->

# Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments
- **Core Insight:** Relabeling agent sub-trajectories with instructions they actually accomplish (backward construction) turned 1,125–4,568 raw trajectories per environment into 10,169–11,782 filtered examples, and LoRA-tuning Codestral-22B on them raised WebArena success from 4.7 to 24.2, versus 10.2 for the same pipeline without backward construction (Table 1, Table 3).
- **Guideline:** When synthesizing agent SFT data in an environment where the generator often fails the task it was given, relabel each sub-trajectory with an instruction written from the trajectory instead of keeping the original instruction, because the no-relabeling variant ("data distill") gained at most 3.4 points in ICL and 5.5 points in training while the full pipeline gained up to 12.2 and 19.5 (Tables 2–3); this was tested with Claude-3.5-sonnet as generator on four benchmarks.
- **Authors:** Hongjin Su, Ruoxi Sun, Jinsung Yoon, Pengcheng Yin, Tao Yu, Sercan Ö. Arık (Google; The University of Hong Kong)
- **Year:** 2025 (arXiv v1 2025-01; only version listed; no venue stated on arXiv)
- **URL:** https://arxiv.org/abs/2501.10893
- **Source type:** paper
- **Relevant topics:** agentic data synthesis, self-instruct, trajectory relabeling, data filtering, retrieval for agents, SWE/web/desktop agents

## Abstract
LLM agents are limited by the lack of high-quality agent data for the environments they operate in. Learn-by-interact adapts agents to a given environment without human annotation. It generates tasks from documentation, collects agent-environment trajectories, and constructs instructions by summarizing or abstracting interaction histories (backward construction). The data are used for training and for training-free in-context learning (ICL) with retrieval methods designed for agents. On SWE-bench, WebArena, OSWorld, and Spider2-V, baselines improve by up to 12.2 points for ICL with Claude-3.5-sonnet and 19.5 for training with Codestral-22B; backward construction accounts for up to 14.0 points in training; agentic retrieval beats conventional RAG (Abstract).

## Key Contributions
- A synthesis loop: self-instruct from documentation → LLM rollouts → backward construction over all sub-trajectories → filtering (§2.2–2.3, Algorithm 1).
- Backward construction, which both repairs instruction-trajectory misalignment and yields a number of examples quadratic in trajectory length (§2.2).
- Agentic retrieval for ICL: observation-based BM25 matching plus model-written queries to a dense retriever at every step (§2.4, Algorithm 2).
- Ablations on retrieval type, trajectory length, data scaling, resource conditioning, and cross-website transfer (§4, App. G–H).

## Key Figures/Tables to Study
- Fig. 1 and Algorithms 1–2 (pipeline and retrieval), Table 1 (data statistics).
- Table 2 (ICL results), Table 3 (training results), Table 4 (retrieval ablation), Table 5 (trajectory length), Fig. 3 (data scaling).
- Tables 29–31 (self-instruct, backward-construction, and filter prompts), Table 7 (held-out CMS websites).

## Technical Details
- **Task formulation (§2.1):** at step i the agent predicts action a_i from instruction I and history H = (o_0, a_1, …, o_{i−1}); the episode ends on `stop` or at a maximum step count m.
- **Task generation (§2.2, §3.3, Table 29):** for each document the generator proposes three tasks users frequently perform. Document sources (App. D): all non-repeated Python files in SWE-bench-Verified; GitLab, Google Maps, Amazon, and Reddit help pages for WebArena (same-domain documents replace missing site documentation); Chrome, GIMP, LibreOffice, Ubuntu, Thunderbird, VLC, VS Code documentation for OSWorld; dbt, Dagster, Astronomer, Airbyte, Superset, Metabase, Snowflake, BigQuery, JupyterLab documentation for Spider2-V.
- **Backward construction (§2.2, Algorithm 1 lines 15–22, Table 30):** for every sub-trajectory T′ = T[i:j], an LLM writes a new instruction I′, either a task the sub-trajectory completes or a summary of its steps, and (I′, T′) is stored. App. E shows a trajectory for "Upload CSV data in Google Drive to BigQuery" that selected Google Cloud Storage by mistake and was relabeled "Link CSV file in Google Cloud Storage to BigQuery".
- **Filtering (§2.3, Table 31):** (1) remove a step (a_i, o_i) identical to (a_{i−1}, o_{i−1}); (2) keep a pair only if every LLM in the committee answers yes to all four criteria: aligned, coherent, natural, reasonable (no detours, no back-and-forth). App. F shows filtered examples with unnecessary actions and repeated states.
- **Models (§3.3):** generator Claude-3.5-sonnet; committee Gemini-1.5-pro and Claude-3.5-sonnet. ICL with Gemini-1.5-pro, Claude-3.5-sonnet, Codegemma-7B, Codestral-22B; LoRA training of Codegemma-7B and Codestral-22B, evaluated without retrieval. Temperature 0 throughout the experiments (App. C).
- **Data (Table 1), SWE-bench / WebArena / OSWorld / Spider2-V:** documents 6,464 / 3,578 / 7,362 / 11,231; raw trajectories 4,568 / 3,967 / 1,125 / 1,226; examples 41,237 / 32,319 / 19,688 / 21,525; filtered 10,232 / 10,456 / 11,782 / 10,169. One data set is built per benchmark (§3.3).
- **Retrieval (§3.3, Algorithm 2, App. C):** up to m1 = 5 observation-based and m2 = 5 model-based examples, filling the context up to the model maximum; Vertex AI text embeddings as dense retriever.
- **ICL results (Table 2), Claude-3.5-sonnet, baseline → Learn-by-interact:** SWE-bench 51.2 → 60.0; WebArena 35.8 → 48.0; OSWorld 12.4 → 22.5; Spider2-V 8.4 → 16.6. Gemini-1.5-pro: 13.3 → 18.7, 17.9 → 25.6, 4.9 → 10.3, 8.3 → 16.4. LATS, the strongest prior method, reaches 55.2 / 41.3 / 16.8 / 11.2 with Claude.
- **Training results (Table 3), WebArena / OSWorld:** Codegemma-7B base 3.3 / 0.0, data distill 6.2 / 1.4, Learn-by-interact 14.6 / 6.5; Codestral-22B base 4.7 / 2.2, data distill 10.2 / 5.4, Learn-by-interact 24.2 / 11.7.
- **Inference cost (§4.1, Fig. 2):** averaged over the four benchmarks with Claude, LATS improves by 2.5 points using nearly four times more tokens per instance than the baseline; Learn-by-interact uses fewer LLM calls and slightly more tokens than the baseline.
- **Retrieval ablation (Table 4), Claude, WebArena:** no retrieval 35.8, instruction-based 36.6, observation-based 42.5, model-based 44.8, combined 48.0.
- **Trajectory length (§4.3, Table 5):** groups <5 steps, 5–9, ≥10, each capped at 200M tokens. Short > medium > long; all three combined is best (Claude WebArena 42.0; Codestral-22B WebArena 15.4 vs short-only 13.5).
- **Scaling (§4.4, Fig. 3):** performance averaged over WebArena and OSWorld rises from 0 to 10k examples for both ICL and training; for a given data amount, fine-tuning the smaller models gains more than using the data as ICL examples.
- **Resource conditioning (App. G, Table 6), WebArena ICL with Claude:** tasks generated from documentation 43.2 (5k) and 48.0 (10k) versus 37.3 and 39.6 when tasks are sampled from the environment alone.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Codegemma-7B, Codestral-22B | 7B, 22B | SFT | Method | LoRA; synthesized examples converted to action-prediction input-output pairs | arXiv:2501.10893v1 §2.4, §3.3, Table 32 | verified 2026-09-14 | no ablation reported |
| Codegemma-7B, Codestral-22B | 7B, 22B | SFT | Data | filtered examples synthesized for the target benchmark (Table 1 counts: WebArena 10,456; OSWorld 11,782); the exact subset used for Table 3 is not stated | Table 1, §3.3 | verified 2026-09-14 (counts) / not reported (subset) | Table 3 vs data distill; Fig. 3 scaling |
| Codegemma-7B, Codestral-22B | 7B, 22B | SFT | Batch size; learning rate; warmup ratio; max length; epochs | 128; 0.00002; 0.03; 8192; 3 | App. C | verified 2026-09-14 | no ablation reported |
| Codegemma-7B, Codestral-22B | 7B, 22B | SFT | LoRA rank/alpha, LR schedule, loss masking, packing | not reported | checked body, App. A–H | not reported | — |
| Codegemma-7B, Codestral-22B | 7B, 22B | SFT | Compute | H100 80GB machines; GPU count not reported | App. C | verified 2026-09-14 | — |
| Data generation | — | distill-SFT | Generator; filter committee; tasks per document; temperature | Claude-3.5-sonnet; Gemini-1.5-pro + Claude-3.5-sonnet (unanimous); 3; temperature 0 is stated for "the experiments", application to synthesis rollouts not stated | §3.3, §2.3, App. C | verified 2026-09-14 | Table 6 (documents vs no documents) |

## Findings relevant to negative feedback, generality, distillation
- **Negatives:** failed or misaligned trajectories are not discarded as whole; backward construction rewrites their instruction so the sub-trajectory becomes a positive target for the task it did accomplish (§2.2, App. E). This course calls it hindsight relabeling (Interpretation; the paper does not use that term). Low-quality pairs (detours, repeated states, committee rejection) are discarded, i.e. treated as negative marginal value (§2.3, App. F). No negative-gradient term is used; training is SFT (§2.4).
- **Evidence for relabeling:** "data distill" keeps the original instruction-trajectory pair. The text says it gains no more than 2% over the baseline in ICL and attributes this to noisy multi-turn distilled data (§3.5.1); Table 2 shows gains of 0.7–1.9 with Gemini but 2.8 (SWE-bench) and 3.4 (WebArena) with Claude. In training, data distill reaches 10.2 vs 24.2 for Codestral-22B on WebArena (Table 3).
- **Generality:** with WebArena CMS sites held out, Codestral-22B improves from 3.3 to 12.6 versus 17.6 when CMS data are included (sizes matched by downsampling); Claude ICL 26.0 → 28.3 vs 29.2 (App. H, Table 7). All other evaluations use data synthesized for the same benchmark environment; no general-capability benchmark is reported.
- **Distillation:** Claude-3.5-sonnet generates the trajectories and, with Gemini-1.5-pro, filters them; the data train 7B and 22B models (§3.3). The text states that trained Codestral-22B outperforms Gemini-1.5-pro using the data as ICL examples (§3.5.2); in Tables 2–3 this holds on OSWorld (11.7 vs 10.3) but not on WebArena (24.2 vs 25.6).
- **Limitations stated (§7):** many LLM calls for generation and filtering; relies on environment resources that may be incomplete or unavailable.

## Connections
- [[self-instruct]] — task proposals come from self-instruct conditioned on documentation (§2.2).
- [[os-genesis]], [[agenttrek]] — other library sources that derive GUI/web tasks from interaction or tutorials; compare their relabeling and replay steps.
- [[agenttuning]] — cited as self-instruct plus GPT-4 trajectories without backward construction (§5).
- [[webarena-data]], [[swe-gym]] — trajectory sources for the same WebArena and SWE-bench environments.
- [[openhands-data]] — the SWE-bench baseline follows CodeAct from the OpenHands paper (App. A).
- [[agent-early-experience]] — another reward-free way to turn agent-generated interaction data into training signal.
- [[rejection-sampling-finetuning]] — contrast: RFT keeps only successful trajectories for the original task; Learn-by-interact keeps relabeled sub-trajectories.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2501.10893 (arXiv v1, 2025-01-18; only version listed).
- Audit claims not found in the source: "10,232–11,782 filtered examples per environment" → the range is 10,169 (Spider2-V) to 11,782 (OSWorld) (Table 1). "LLM-committee voting" → a pair is kept only if all committee LLMs judge it high quality (§2.3). "ICLR 2025" venue is not stated in the arXiv v1 PDF or abs page.
- Not reported by the source: LoRA rank, learning-rate schedule, loss masking, total GPU hours, data-synthesis cost.
