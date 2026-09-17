---
chapter: ch-30b
course: llm-training
phase: read
excerpt_of: primary source (no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2509.17567
created_at: "2026-09-15"
---

# Excerpt: LIMI: Less is More for Agency

- **Authors:** Yang Xiao, Mohan Jiang, Jie Sun, Keyu Li, Jifan Lin, Yumin Zhuang, et al. (SII-GAIR)
- **Year:** arXiv v1 2025-09; v2 2025-09-25
- **Checked against:** arXiv:2509.17567v2 (PDF, full text including Appendix A), 2026-09-15
- **Why this excerpt exists:** ch-30b cites LIMI as the small-agentic-set data point in the agentic-share section. The library card `limi` was planned but not present when the chapter was written.

## Data (§3.2-§3.3, Fig. 4)
- 78 training queries: 60 collected from real scenarios of developers and researchers, and 18 sampled from several thousand queries synthesized by GPT-5 from GitHub pull requests (repositories with more than 10,000 stars; unified diff under 1,200 tokens; Markdown-only PRs excluded) (§3.2).
- Trajectories were collected in the SII CLI environment with four PhD-student annotators collaborating with GPT-5 as the agent, "continuously gathering trajectories until successful completion is achieved" (§3.3).
- Trajectory length: minimum 13k, average 42.4k, maximum 152k tokens (Fig. 4).
- Only successful trajectories are used; failed attempts are not trained on as separate samples.

## Training and evaluation (§3.4, §4.1)
- Fine-tuned models: GLM-4.5 (355B) → LIMI; GLM-4.5-Air (106B) → LIMI-Air. Comparison runs fine-tune GLM-4.5 on CC-Bench-trajectories (260 samples), AFM-WebAgent-SFT-Dataset (7,610), and AFM-CodeAgent-SFT-Dataset (10,000) (§4.1, Table 2).
- Training framework: slime. Learning rate, epochs, batch size, and sequence length are not reported.
- AgencyBench (GAIR-NLP) has 10 tasks, four vibe-coding and six research tasks, each with subtasks (Table 1; App. A). Metrics: first-turn functional completeness (FTFC), success rate within R = 3 rounds (SR@3), and remaining chances (RC@3) (§3.4).
- Generalization suite: tau2-bench airline and retail (Pass^4 as defined in §3.4), EvalPlus HumanEval and MBPP, DS-1000, SciCode main problem (MP) and sub-problem (SP).

## Results with loci
Table 2, AgencyBench average: Kimi-K2-Instruct 24.1; DeepSeek-V3.1 11.9; Qwen3-235B-A22B-Instruct 27.5; GLM-4.5 45.1; GLM-4.5-Air 17.0; LIMI-Air 34.3; GLM-4.5-CC (260) 29.2; GLM-4.5-Code (10,000) 47.8; GLM-4.5-Web (7,610) 36.7; LIMI (78) 73.5.

Table 3 (with CLI environment), selected columns:

| Model | Samples | tau2 airline | tau2 retail | EvalPlus-HE | DS-1000 | AVG (printed) |
|---|---|---|---|---|---|---|
| GLM-4.5 | none | 28.0 | 36.8 | 90.2 | 33.6 | 43.0 |
| GLM-4.5-CC | 260 | 38.0 | 39.6 | 90.2 | 38.7 | 39.2 |
| GLM-4.5-Web | 7,610 | 18.0 | 13.2 | 84.1 | 33.9 | 33.7 |
| GLM-4.5-Code | 10,000 | 20.0 | 16.7 | 87.8 | 38.5 | 40.9 |
| LIMI | 78 | 34.0 | 45.6 | 92.1 | 36.6 | 57.2 |

- Table 4 (without CLI environment): GLM-4.5 tau2 airline 32.0, retail 52.6; LIMI 40.0, 49.1; averages 48.7 and 50.0.

## Statements that need care when quoted
- The abstract's "53.7% improvement over models trained on 10,000 samples" is a relative change: (73.5 − 47.8) / 47.8 = 53.8% (derived). §4.3 calls it "a remarkable 53.7 percentage point improvement"; the difference in points is 25.7 (derived).
- "128 times fewer samples" is 10,000 / 78 = 128.2 (derived).
- The Table 3 caption states that averages "include AgencyBench performance". An unweighted mean of LIMI's seven printed generalization columns is 45.6, and with AgencyBench 73.5 added it is 49.1 (derived); the printed 57.2 cannot be reproduced from the table, and the weighting is not stated.
- Training queries and AgencyBench tasks come from the same two domains (vibe coding and research workflows), and AgencyBench is published by the authors' group.
- No seeds, repeated runs, or confidence intervals are reported. Non-agentic general benchmarks (knowledge, chat quality, safety) are not reported.
