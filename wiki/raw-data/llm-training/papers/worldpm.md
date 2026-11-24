<!-- scope: WorldPM (Wang et al., Qwen, 2025) — Bradley–Terry preference-model pretraining on ~15M StackExchange vote pairs at 1.5B/7B/72B; scaling behaviour split by adversarial / objective / subjective test sets; WorldPM as initialization for preference fine-tuning and RLHF
     deps: [[bradley-terry-rm]], [[reward-model-overoptimization]]
     see-also: [[ultrafeedback]], [[warm-weight-averaged-reward-models]], [[reward-ensembling]], [[generative-reward-models]]
-->

# WorldPM: Scaling Human Preference Modeling
- **Core Insight:** Training Bradley–Terry preference models on about 15M StackExchange vote pairs gives a power-law decrease in test loss on adversarial test sets for all sizes, improvement on objective (verifiable-answer) test sets only at 72B, and no clear scaling trend on subjective test sets, where the authors attribute the flat curves to style preference and annotation noise (Abstract; §3.2, §3.3, §5).
- **Guideline:** Use large-scale forum preference data as an initialization stage before fine-tuning a reward model on curated human-preference data, because WorldPM initialization raised 56 of 66 dataset-by-metric cells at 7B and 51 of 66 at 72B across the 7K, 100K, and 800K fine-tuning sets (count of ↑ markers in Tables 2–3); do not expect more forum data to improve subjective preference prediction, and control length and markdown when measuring it (§3.3, §5).
- **Authors:** Binghai Wang, Runji Lin, Keming Lu, Le Yu, Zhenru Zhang, Fei Huang, Chujie Zheng, Kai Dang, Yang Fan, Xingzhang Ren, An Yang, Binyuan Hui, Dayiheng Liu, Tao Gui, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang, Bowen Yu, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba Group; Fudan University)
- **Year:** 2025 (arXiv v1 2025-05-15; v2 2025-05-18)
- **URL:** https://arxiv.org/abs/2505.10527 (code: https://github.com/QwenLM/WorldPM)
- **Source type:** paper
- **Relevant topics:** reward modeling, preference-model pretraining, scaling laws for reward models, style and length bias, Bradley–Terry loss, reward-model initialization for RLHF

## Abstract
The authors report that preference modeling shows scaling behaviour similar to language modeling, and name the approach World Preference Modeling (WorldPM), described as a unified representation of human preferences. They collect preference data from public forums and train models from 1.5B to 72B parameters on 15M-scale data. Adversarial metrics (identifying deceptive features) improve with more data and larger models. Objective metrics (questions with well-defined answers) show emergent behaviour in larger models. Subjective metrics (preferences from a limited number of humans or AI annotators) show no scaling trend. Used as the initialization for preference fine-tuning, WorldPM improves generalization on human-preference datasets of 7K, 100K, and 800K samples across 7 benchmarks with 20 subtasks, with gains above 5% on many key subtasks. In the authors' internal RLHF pipeline it gives 4% to 8% gains on in-house evaluations (Abstract; §1).

## Key Contributions
- Preference pairs from forum votes at 15M-pair scale, about 30G tokens (§3.2, §6).
- Scaling curves of Bradley–Terry test loss for 1.5B, 7B, and 72B models across adversarial, objective, and subjective test sets (§3.2, Fig. 1).
- Style analysis with length and markdown as control features; style-controlled evaluation of subjective test sets (§3.3, Fig. 4, Fig. 5, App. C).
- WorldPM as an initialization for fine-tuning on HelpSteer2 (~7K), UltraFeedback (~100K), and RLHFlow pair_data_v2 (~800K) (§4.1–4.2, Tables 2–3).
- Integration into an RLHF pipeline with GRPO (§4.4, Table 4).

## Key Figures/Tables to Study
- Fig. 1: test loss vs training data for 1.5B/7B/72B, split by test domain; the central scaling result.
- Table 1 and Fig. 2: comparison of StackExchange, Reddit, and Quora as data sources.
- Fig. 3: loss drop with gradient spike near 12.6M samples on the 72B run.
- Fig. 4: subjective test loss with and without style control, by annotation source.
- Tables 2–3: fine-tuning with and without WorldPM at 7B and 72B. Table 4: GRPO results.
- App. D, Tables 10–12: reward-model filtering of forum data and the claim that filtered-out pairs are not noise.

## Technical Details
- **Pair construction (§3.1.1, App. A.1–A.2).** For each forum post, two responses with different net votes (upvotes minus downvotes) are sampled; the higher-vote response is chosen. An author-accepted answer counts as one extra upvote. Posts need at least two replies with different vote counts. Pairs with score gaps 1–2, 3–5, 6–10, and 11+ give similar performance (Table 5).
- **Source choice (§3.1.4, Table 1).** 800K pairs per source, Qwen2.5-7B. StackExchange / Quora / Reddit: PPE-Human 62.8 / 60.2 / 55.1; PPE-Objective 62.1 / 57.7 / 59.4; RMB 76.7 / 66.1 / 59.5; RewardBench 84.4 / 69.0 / 77.0; RM-Bench 72.5 / 60.9 / 60.7. ArmoRM reference: 60.2 / 64.3 / 72.3 / 89.8 / 75.4. StackExchange is selected.
- **Training (§3.1.2, §3.2, App. A.3).** Bradley–Terry negative log-likelihood (Eq. 2). Qwen2.5 models at 1.5B, 7B, and 72B (base or instruct variant not stated in §3.2). Learning rate 3e-6, batch size 10K, 1,536 steps, Adam, warmup ratio 0.1, weight decay 0.1, context length 2,048, a single epoch with no sample reuse. Ablations: larger batch sizes are better at equal steps up to 40K (Fig. 9); learning rates 1e-6 to 1e-5 are stable, 3e-6 marginally best (Fig. 10).
- **Evaluation metric (§3.1.3).** Scaling results use BT loss on held-out preference pairs, so the authors do not follow each benchmark's own protocol (App. A.4).
- **Test-set groups (§3.2).** Adversarial: RM-Bench chat, OffsetBias, RewardBench chat-hard. Objective: PPE objective subsets (MBPP-Plus, GPQA, IFEval, MATH, MMLU-Pro), RM-Bench code/math, RewardBench reasoning. Subjective: PPE-Human (Chatbot Arena), RMB helpfulness (GPT-4 labels), HelpSteer2.
- **Scaling results (§3.2).** Adversarial loss decreases as a power law for all sizes. On objective sets, 1.5B does not generalize, 7B improves on MBPP-Plus and IFEval but its loss rises on HumanEvalPack and GPQA, and 72B improves consistently. Subjective loss converges early or rises with more data. Exact loss values are given only in figures.
- **Loss transition (§3.2, Fig. 3).** On the 72B run, at about 12.6M samples the loss drops suddenly while the gradient norm spikes and then subsides.
- **Style (§3.3, App. C.1).** Style features: token length, markdown lists, headers, bold (Eq. 4–5). Length is the main style factor. Under style control, HelpSteer2 results are stable, while Chatbot Arena and GPT-4-labeled sets change substantially, and the controlled–uncontrolled gap grows with model size (Fig. 4). Early in training, accuracy is 82% on pairs where the chosen response is longer and 57% where it is shorter; the second rises with training (Fig. 5b).
- **Fine-tuning (§4.1–4.2).** At least two epochs; batch 128 for HelpSteer2, 512 otherwise. Alpaca Eval and Arena Hard are measured by Best-of-64 sampling from Qwen2.5-7B/72B-Instruct (§4.1). 7B on HelpSteer2, without → with WorldPM: OffsetBias 78.87 → 86.60; RB-Reason 88.02 → 96.44; RM-B-Chat 78.29 → 82.17; Arena Hard 65.88 → 70.27 (Table 2). Some cells decrease, for example RB-Safety 91.76 → 89.46 for 7B on UltraFeedback (Table 2). Gains appear from the 5M-sample WorldPM checkpoint (§4.3, Fig. 7).
- **RLHF (§4.4, Table 4).** Two 72B preference models are trained on in-house preference data, one initialized from WorldPM; GRPO then trains QwQ-32B (Table 4 caption). Without → with WorldPM, in-house: Math 60.06 → 62.37; Code 56.63 → 64.51; OpenQA 70.32 → 71.42; Writing 73.52 → 78.07; IF 61.19 → 66.22. Public: Arena Hard 91.06 → 93.13; MT-Bench 8.56 → 8.62; Alpaca Eval 90.90 → 91.04.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WorldPM | 1.5B, 7B, 72B | reward-model | pairs; tokens; epochs | ~15M StackExchange pairs; ~30G tokens; 1 epoch | arXiv:2505.10527v2 §3.2, §6 | verified 2026-11-24 | Fig. 1 scaling curves; gains in fine-tuning appear from 5M (Fig. 7) |
| WorldPM | 1.5B, 7B, 72B | reward-model | LR; batch; steps | 3e-6; 10K; 1,536 | v2 §3.1.2, App. A.3 | verified 2026-11-24 | Figs. 9–10: batch up to 40K better at equal steps; LR stable 1e-6–1e-5 |
| WorldPM | 1.5B, 7B, 72B | reward-model | optimizer; warmup; weight decay; context | Adam; 0.1; 0.1; 2,048 | v2 App. A.3 | verified 2026-11-24 | no ablation reported |
| WorldPM | 7B | reward-model | pair score gap | any nonzero net-vote gap | v2 Table 5 | verified 2026-11-24 | Table 5: gap buckets 1–2 to 11+ perform similarly |
| WorldPM-initialized PMs | 7B, 72B | reward-model | fine-tune epochs; batch | ≥2; 128 (HelpSteer2), 512 (others) | v2 §4.1 | verified 2026-11-24 | Tables 2–3 compare with and without WorldPM |
| RLHF policy (QwQ-32B) | 32B | RL | algorithm; RM | GRPO; 72B PM initialized from WorldPM | v2 §4.4, Table 4 | verified 2026-11-24 | Table 4, single comparison; RL hyperparameters not reported |
| WorldPM | — | reward-model | compute | not reported | checked v2 body, App. A | not reported | — |

## Findings relevant to generality and negative feedback
- **Generality.** The test sets where scaling helps are those with a checkable answer (objective) or a constructed flaw (adversarial); subjective preference does not improve with more forum data, and the authors state that subjective performance depends more on annotation strategy than on pretraining scale (§5). The authors state that StackExchange shows the best cross-source generalization among the forum sources and can represent the others (Fig. 2, §3.1.4).
- **Negative feedback.** The rejected response in each pair is the lower-voted forum answer, not a constructed error. Filtering pairs with a 72B reward model improves Quora data strongly (RewardBench 69.0 → 85.4) but StackExchange only slightly (Table 10); filtered-out pairs, scored with inverted labels, perform about as well as kept pairs, which the authors read as opposing genuine preferences rather than noise (App. D.3, Table 12). They argue RM filtering would make the model copy the filter RM (App. D.3).
- **Style as a confound.** Without style control, Alpaca Eval scores track response length; Arena Hard is more stable (§3.4, Fig. 6).
- **Limitations stated (§6).** The data is small relative to pretraining corpora; style control covers only length and markdown; benchmark protocols were modified.

## Connections
- [[bradley-terry-rm]]: the training loss used here.
- [[reward-model-overoptimization]]: WorldPM reports Best-of-N alignment peaking at 0.6M samples on Alpaca Eval for 72B while Arena Hard keeps improving to 10M (§3.4); compare with over-optimization curves.
- [[ultrafeedback]]: one of the three fine-tuning datasets (~100K).
- [[warm-weight-averaged-reward-models]], [[reward-ensembling]]: other ways to make reward models more robust; WorldPM addresses it through initialization data scale.
- [[processbench]], [[qwen-prm-lessons]]: same Qwen team; step-level rather than response-level reward models.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2505.10527, the v2 HTML, and the v2 PDF text (pdftotext). v1 was not compared.
- §4.4 says "two 72B models" and the Table 4 caption says "GRPO alignment results on QwQ-32B"; read together, the 72B models are the preference models and QwQ-32B is the GRPO policy. The paper does not state this pairing in one sentence.
- The abstract's "4% to 8%" in-house gain is not given as a per-column number in Table 4; the card lists the Table 4 values only.
- The 56/66 and 51/66 counts in the Guideline were made for this card by counting ↑ markers in the PDF text of Tables 2–3 (9 decreases and 1 unchanged at 7B; 14 decreases and 1 unchanged at 72B); they are not stated by the paper.
- Fig. 2 cross-source accuracy values and Fig. 1 loss values are in figures and were not read numerically.
- Not reported by the source: training compute, base vs instruct initialization for WorldPM, RL hyperparameters for the GRPO run, model-weight release (only the GitHub URL is given in the paper).
