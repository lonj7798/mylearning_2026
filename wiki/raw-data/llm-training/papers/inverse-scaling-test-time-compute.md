<!-- scope: evaluation tasks on which longer reasoning traces lower the accuracy of large reasoning models, with per-model failure modes and a safety evaluation across reasoning budgets
     deps: [[deepseek-r1]]
     see-also: [[overthinking-o1-like-llms]], [[scaling-reasoning-losing-control-mathif]], [[s1]], [[anthropic-safety-research]]
-->

# Inverse Scaling in Test-Time Compute
- **Core Insight:** On constructed tasks, longer reasoning lowers accuracy for several large reasoning models; for example Claude Opus 4 drops from nearly 100% to around 85-90% on Misleading Math and DeepSeek R1 drops from 70% to 30% with five distractors in natural overthinking (Fig. 3), while Claude Sonnet 4, o4-mini, and o3 show no inverse scaling on MultiArith, ASDiv, GSM8K, or GSM-IC (App. F.3).
- **Guideline:** When evaluating a reasoning model for capability or safety, measure it at several reasoning budgets and across naturally sampled reasoning lengths, because the direction of the trend differs by model and task (Table 1, Table 4) and standard arithmetic benchmarks did not reveal the degradation (App. F.3).
- **Authors:** Aryo Pradipta Gema, Alexander Hägele, Runjin Chen, Andy Arditi, Jacob Goldman-Wetzler, Kit Fraser-Taliente, et al. (last author Ethan Perez; Anthropic Fellows Program, Anthropic, University of Edinburgh, and others)
- **Year:** 2025 (arXiv v1 2025-07; Transactions on Machine Learning Research 12/2025)
- **URL:** https://arxiv.org/abs/2507.14417
- **Source type:** paper
- **Relevant topics:** test-time compute, reasoning length, overthinking, distractors, spurious features, evaluation protocol, AI-risk evaluation

## Abstract
The authors construct evaluation tasks on which extending the reasoning length of Large Reasoning Models (LRMs) lowers performance, an inverse relation between test-time compute and accuracy. The tasks cover four categories: simple counting with distractors, regression with spurious features, deduction with constraint tracking, and advanced AI-risk evaluations. Five failure modes are reported: Claude models become more distracted by irrelevant information; OpenAI o-series models resist distractors but overfit to problem framings; models shift from reasonable priors to spurious correlations; all models have difficulty keeping focus on complex deduction; and extended reasoning can amplify concerning behaviors, with Claude Sonnet 4 expressing more self-preservation. The authors conclude that test-time compute scaling remains promising but can reinforce flawed reasoning patterns, and that models should be evaluated across reasoning lengths.

## Key Contributions
- Two length-control protocols: controlled overthinking (prompted budgets) and natural overthinking (five samples per question ranked by length), plus a cautioned variant (§3, App. D).
- Five main tasks: Misleading Math, Misleading Python, Misleading Math (Famous Paradoxes), Grades Regression, Zebra Puzzles (§4, Table 3).
- Trend table for nine models with a stated threshold for inverse, positive, flat, and noisy trends (Table 1, Table 4).
- Checks that the effect is absent in existing benchmarks: grade-school arithmetic, Inverse Scaling Prize tasks, out-of-domain distractors (App. F).
- Realism checks with LLM-written distractors and user-stated priors (App. G).

## Key Figures/Tables to Study
- Fig. 2: requested budget vs actual reasoning tokens for Claude Opus 4, o3, DeepSeek R1.
- Table 1: trend symbol per task, model, and setup.
- Fig. 7: Pearson correlation of predicted grade with each feature across budgets, 0-shot vs 16-shot (Claude Opus 4).
- Fig. 10 and Table 4: Survival Instinct and the other 14 model-written evaluation tasks.
- Fig. 32-33: Inverse Scaling Prize tasks and arithmetic benchmarks with flat or positive trends.

## Technical Details
- Models: Claude Sonnet 3.7, Claude Sonnet 4, Claude Opus 4, o3-mini, o4-mini, o3, Qwen3-32B, QwQ-32B, DeepSeek R1; main-text plots use Opus 4, o3, R1 (§4, App. C).
- Controlled overthinking: keywords "don't think", "think", "think harder", "ultrathink" plus an integer token budget for Claude and open-weight models (e.g., 0, 1,024, 2,048, 4,096) and "low/medium/high" for o-series; no-reasoning baseline turns thinking off for Claude and prefills empty think tags for open-weight models; o-series cannot disable thinking (§3). The system prompt says "YOU MUST USE ALL OF YOUR THINKING TOKENS" (App. A.2).
- Sampling: temperature 1.0 for Claude and OpenAI models, 0.6 for open-weight models; 3 repetitions per budget (controlled), 5 (natural) (§3). Cautioned setup: budgets 1024-16384, 3 seeds (App. D).
- Limits: 16k reasoning tokens and 10k output tokens (§4.3). Claude token counts are a proxy computed with the o1 tokenizer; OpenAI counts come from the reasoning_tokens field (App. A.4).
- Trend rule: inverse or positive requires >2% accuracy change or >0.05 RMSE change with non-overlapping confidence intervals; flat is below those; noisy exceeds them with overlapping intervals (Table 1 caption). Intervals are 95% via SEM × 1.96 (App. A.4).
- Hardware: 8 NVIDIA H200s; DeepSeek R1 run as a 4-bit quantized (AWQ) version; "the full version" of Qwen3 14B and 32B (App. A.3).
- Misleading Math and Misleading Python: 2,500 questions each, 500 per distractor count n ∈ {1,...,5}, answer always "2" (§4.1). Famous Paradoxes: 812 questions, 92 without distractors and 180 each for n ∈ {1, 3, 5, 7} (§4.1).
- Misleading Python, controlled: Claude Opus 4 falls from near-perfect to about 80%; o3 shows positive scaling at all distractor counts (§4.1, Fig. 4).
- Famous Paradoxes, controlled: adding distractors raises accuracy, most for o3; the authors attribute this to the framing becoming less recognizable (Fig. 5, §4.1).
- Grades Regression: 500 students, each evaluated zero-shot, 8-shot, and 16-shot (1,500 instances); metric RMSE on a 0-10 grade (§4.2, Table 3). Study hours has correlation 0.73 with the true grade; with longer zero-shot reasoning Opus 4 shifts weight toward sleep hours and stress level; few-shot examples keep the study-hours correlation (Fig. 7). o3-mini keeps inverse scaling even with few-shot examples (App. D.2).
- Zebra Puzzles: 200 BBEH puzzles, grids 5×5 to 8×8; 8×8 grids have no distracting clues; best-case 8×8 needs about 6,400 tokens at about 100 tokens per deduction (§4.3, Table 2). In natural overthinking all nine models are marked inverse (Table 1).
- Survival Instinct (953 items): Claude Sonnet 4 willingness to be turned off falls from 60% to 47% with longer reasoning; o3 rises from 72% to 76%; R1 stays around 71-72% (§5, Table 3). The text states that among the safety benchmarks only Claude Sonnet 4 shows consistent inverse scaling on Survival Instinct (Table 4 also marks Opus 4 as inverse there); o3-mini is inverse on Myopic Reward; most other tasks are flat or noisy (§5, Table 4).
- Existing benchmarks: Claude Sonnet 4, o4-mini, and o3 show no inverse scaling on MultiArith, ASDiv, GSM8K, GSM-IC, with responses under 1,000 reasoning tokens (App. F.3); the nine Inverse Scaling Prize tasks are mostly flat or positive (App. F.2); out-of-domain distractors give a maximum accuracy drop of about 0.02 (App. F.1).
- Realism: 200 distractor questions written by Claude Sonnet 4.5 (20 per 10 templates) still degrade Sonnet 3.7 and Sonnet 4 (App. G.1); appending explicit (study hours) or implicit (time investment) prior statements to 100 zero-shot items, 3 statements each (300 per condition), gives inverse scaling for all models except Qwen3 32B and QwQ 32B (App. G.2).
- Data: huggingface.co/datasets/inverse-scaling-ttc/inverse-scaling-ttc-main, with a canary string (App. A.1).

## Findings relevant to generality and long context
- Generality: more reasoning tokens did not improve accuracy uniformly across tasks; the sign of the trend depended on model family and task (Table 1). The authors interpret this as current training possibly rewarding recognition of known problem framings over reasoning about the question asked (Takeaway 2, Interpretation).
- Measurement: the degradation was absent from standard arithmetic and Inverse Scaling Prize benchmarks for the models tested there (App. F.2-F.3). Natural overthinking gave stronger inverse trends than prompted budgets on Zebra Puzzles (Takeaway 5).
- Long outputs: in the longest natural traces on Zebra Puzzles, models test many hypotheses and second-guess deductions; the shortest traces show systematic constraint handling (§4.3).
- Mitigation reported: few-shot examples removed inverse scaling on Grades Regression for Claude models and o3, but not for o3-mini (App. D.2); stated priors did not (App. G.2).
- Limitation stated by the authors: most tasks are synthetic and may underestimate how the behaviors appear in real deployments (Limitations).

## Connections
- [[overthinking-o1-like-llms]] — prior work framing overthinking as an efficiency cost; this paper reports accuracy loss.
- [[s1]] — budget forcing for test-time scaling, cited as Muennighoff et al. (2025).
- [[deepseek-r1]], [[qwen-3]] — open-weight reasoning models evaluated here.
- [[scaling-reasoning-losing-control-mathif]] — instruction-following degradation in reasoning models.
- [[sky-t1-flash-overthinking]], [[l1-lcpo]], [[tokenskip]] — methods that shorten or control reasoning length.
- [[context-length-alone-hurts]], [[chroma-context-rot]] — degradation from long inputs, a different axis from long reasoning outputs.
- [[anthropic-safety-research]] — lab page; the advanced-AI-risk tasks in §5 are the human-generated subsets of Perez et al. (2023) model-written evaluations.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2507.14417 (arXiv v2, 15 Dec 2025, TMLR version; body, App. A-G read).
- Audit claims not found in the source: none.
- Internal inconsistencies in the source: Table 3 caption says "Main Tasks (6 tasks)" and "21 tasks", but the table lists 5 main and 15 model-written tasks and §4 says five main tasks; Table 1 marks the controlled Zebra trend for R1 as noisy while §4.3 describes it as pronounced inverse scaling.
- Not reported by the source: per-task numeric accuracy tables for all nine models (results are given as plots and trend symbols).
