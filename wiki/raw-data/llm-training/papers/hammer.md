<!-- scope: Hammer — on-device function-calling models trained with function/parameter-name masking and an irrelevance-augmented dataset
     deps: [[xlam]]
     see-also: [[toolace]], [[bfcl]], [[granite-function-calling]], [[apigen]]
-->

# Hammer: Robust Function-Calling for On-Device Language Models via Function Masking
- **Core Insight:** Small function-calling models overfit to the function and parameter *names* seen in training; replacing those names with random strings during fine-tuning and adding 7,500 irrelevance examples raises Hammer-7B to 83.92 overall accuracy on BFCL, above every open model in the table and behind only GPT-4 variants (Table 2).
- **Guideline:** When a fine-tuned function-calling model must work against function libraries written by unknown API authors, mask function names, parameter names, and default values in the candidate list during training and update the labels to match, because a larger masking ratio improves transfer to a different benchmark even though it slows learning on the training task (§4.1, §5.5, Figure 5).
- **Authors:** Qiqiang Lin, Muning Wen, Qiuying Peng, Guanyu Nie, Junwei Liao, Jun Wang, et al. (OPPO Research Institute; Shanghai Jiao Tong University; Iowa State University)
- **Year:** 2024 (arXiv v1 2024-10; text read here is arXiv v2, 2024-10-10)
- **URL:** https://arxiv.org/abs/2410.04587
- **Source type:** paper
- **Relevant topics:** function calling, on-device models, name masking, irrelevance detection, negative examples, BFCL

## Abstract
The paper identifies a failure mode of existing function-calling models: their scores vary widely across benchmarks because they are misled by function and parameter naming conventions. Hammer is a family of models fine-tuned from Qwen checkpoints to address this. Two changes are made to the training procedure. Function masking replaces function names, parameter names, and default values in the candidate list with random strings, and updates the labels accordingly, so that the model must read the descriptions. An irrelevance-augmented dataset adds 7,500 examples to xlam-function-calling-60k in which the correct function is removed from the candidate list and the label becomes an empty list. The paper reports that Hammer models reach state-of-the-art results at comparable scale on BFCL and remain stable across five academic benchmarks, and it releases the irrelevance dataset, the tuning framework, and the checkpoints.

## Key Contributions
- A diagnosis of the naming-bias failure, split into three cases: misled by function names, misled by parameter names, disturbed by naming preferences such as CamelCase versus snake_case (§3.1).
- A case study quantifying the bias: masking function and parameter names in the Seal-Tools test set degrades xLAM-1.3B-fc substantially while Hammer-1.5B drops much less, even though descriptions still carry all necessary information (§3.2, Figure 2).
- The masking tuning framework: function names, parameter names, and randomized default values in candidates are replaced during training and the batch labels are rewritten to the masked strings (§4.1).
- The irrelevance-augmented dataset: 7,500 instances resampled from xlam-function-calling-60k with the correct function deleted from the candidate list and an empty-list label, released as 5k-Irrelevance (§1, §4.2).
- Hammer-1.5B, Hammer-4B, Hammer-7B checkpoints, released on Hugging Face under MadeAgents (§1 footnotes).

## Key Figures/Tables to Study
- **Table 1** — the motivating inconsistency: five models across BFCL, API-Bank, Seal-Tools, Tool-Alpaca, Nexus Raven, with averages.
- **Figure 2** — F1 under no mask / function-name mask / argument-name mask / all mask, for xLAM-1.3B-fc and Hammer-1.5B.
- **Table 2** — BFCL comparison (2024-09-20 snapshot), with separate Irrelevance and Relevance columns.
- **Table 3 / Table 5** — five academic benchmarks; Table 5 adds vanilla-versus-Hammer pairs for three base models.
- **Figure 5** — masking-ratio ablation, Seal-Tools (same task) versus API-Bank (cross task).
- **Figure 6** — irrelevance-data proportion ablation, showing the irrelevance/function-calling trade-off.

## Technical Details
- Model family and base checkpoints: Hammer-1.5B from Qwen2-1.5B-Instruct, Hammer-4B from Qwen1.5-4B-Chat, Hammer-7B from Qwen2-7B-Instruct (Table 5, Table 7).
- Training data: xlam-function-calling-60k plus 7,500 irrelevance-augmented instances (§1, §4.2).
- Masking is applied to the candidate list, not to a fixed fraction of tokens: function names → random strings; parameter names → random strings; default values randomized and appended to the parameter description; labels rewritten to use the masked names (§4.1). The ablation defines "mask 0.33" as 33% of the instances in a training batch being masked (Figure 5 caption).
- BFCL, 2024-09-20 snapshot (Table 2): Hammer-7B overall 83.92 (AST 78.70, executable 89.72, irrelevance 72.87, relevance 92.68), placing it between GPT-4-0613 (84.74) and GPT-4-turbo-2024-04-09 (83.89). Hammer-4B overall 76.05; Hammer-1.5B overall 73.04. xLAM-7B-fc is 79.41 and Granite-20B-FunctionCalling 76.63 on the same snapshot.
- Executable-function evaluation: Hammer-7B 89.72 versus GPT-4-0125-Preview 89.25 (Table 2) — the paper notes this as Hammer-7B outperforming GPT-4 on the executable track (§5.3).
- Academic benchmarks, average over API-Bank L-1/L-2, Tool-Alpaca, Seal-Tools (single-tool), Nexus Raven (Table 3/5): Hammer-7B F1 function-name 89.72, F1 function+args 76.21, against xLAM-7B-fc at 72.57 / 67.65 and Qwen2-7B-Instruct (its own base) at 85.94 / 59.84.
- Base-model ablation (Table 5): Deepseek-Coder-7B-Instruct goes from 35.08 / 45.67 to 85.42 / 74.94 after the same tuning; Deepseek-Coder-1.3B-Instruct from 17.42 / 19.78 to 84.51 / 70.91. The paper uses this to argue the method is not specific to Qwen (§5.4).
- Masking-ratio ablation (§5.5, Figure 5): Qwen2-1.5B fine-tuned on the Seal-Tools training set for one epoch at several masking ratios. A large ratio slows learning on Seal-Tools itself; on API-Bank a larger ratio gives better transfer. The paper reads this as masking limiting overfitting to the training data. No single recommended ratio is stated.
- Irrelevance-proportion ablation (§5.6, Figure 6): 10,000 instances sampled from the two datasets at varying proportions, fine-tuning Qwen2-1.5B-Instruct and evaluating on BFCL. Irrelevance detection and function-calling accuracy move in opposite directions; best overall performance is at approximately 10% irrelevance data, which is how the 7.5k target size was chosen. The authors state this proportion may need adjustment for a different base model or dataset.
- Benchmark sizes as stated in §5.1: BFCL over 1,700 instances; API-Bank 314 dialogues and 753 API calls; Nexus Raven 318 examples over 65 APIs; Tool-Alpaca 271 instances in 50 categories, of which 100 simulated test examples are used; Seal-Tools 4,076 automatically generated APIs.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Hammer-7B | 7B | SFT | base checkpoint | Qwen2-7B-Instruct | arXiv:2410.04587v2 Table 5 | verified 2026-09-18 | §5.4: same tuning on Deepseek-Coder also works |
| Hammer-4B | 4B | SFT | base checkpoint | Qwen1.5-4B-Chat | arXiv:2410.04587v2 Table 5 | verified 2026-09-18 | — |
| Hammer-1.5B | 1.5B | SFT | base checkpoint | Qwen2-1.5B-Instruct | arXiv:2410.04587v2 Table 5 | verified 2026-09-18 | — |
| Hammer series | all | SFT | training data | xlam-function-calling-60k + 7,500 irrelevance-augmented instances | arXiv:2410.04587v2 §1, §4.2 | verified 2026-09-18 | §5.6 Figure 6: ~10% irrelevance share is the optimum on BFCL for Qwen2-1.5B-Instruct |
| Hammer series | all | SFT | irrelevance example construction | correct function removed from the candidate list; label replaced with an empty list | arXiv:2410.04587v2 §4.2 | verified 2026-09-18 | — |
| Hammer series | all | SFT | masking targets | function names, parameter names, randomized default values in candidates; labels rewritten to match | arXiv:2410.04587v2 §4.1 | verified 2026-09-18 | §5.5 Figure 5 (ratio ablation only, no chosen value) |
| Hammer series | all | SFT | masking ratio used for the released checkpoints | not reported (checked §4.1, §5.5, and the appendices) | arXiv:2410.04587v2 | not reported | — |
| Qwen2-1.5B (ablation run only) | 1.5B | SFT | epochs, masking-ratio ablation | 1 epoch on the Seal-Tools training set | arXiv:2410.04587v2 §5.5 | verified 2026-09-18 | — |
| Hammer series | all | SFT | learning rate, batch size, epochs, optimizer, LoRA vs full fine-tune, compute | not reported (checked §4, §5.1, App. A and B) | arXiv:2410.04587v2 | not reported | — |

## Findings relevant to negative feedback
- The negatives here are **negative as content** in the sense of §6.1: an irrelevance example is trained with ordinary cross-entropy toward an empty-list target. No likelihood is pushed down, so there is no unlikelihood or preference-gradient term.
- Where they come from: resampled from xlam-function-calling-60k itself, by deleting the correct function from the candidate list (§4.2). The remaining candidates are the original distractors of that example, so the negatives stay inside the training distribution rather than being drawn from unrelated topics.
- Measured effect and its cost: irrelevance detection and function-calling accuracy move in opposite directions as the irrelevance share rises, and the best overall BFCL score for Qwen2-1.5B-Instruct is near a 10% share (§5.6, Figure 6). The paper reports this trade-off as the reason the irrelevance set was capped at 7.5k rather than made larger.
- The paper does not report a false-negative rate for the irrelevance labels, an abstention rate outside BFCL's irrelevance column, or a pass@k measurement.

## Findings relevant to generality
- Every benchmark used is described as out-of-domain for Hammer (§5.1), and Table 1 is built to show that other models' scores do not transfer across benchmarks.
- The masking-ratio ablation is the paper's direct generality evidence: more masking transfers better to a benchmark the model was not trained on, at the cost of slower fitting on the training task (§5.5).
- Hammer-7B improves on its own base Qwen2-7B-Instruct on function+args F1 (76.21 versus 59.84) but the base is higher on API-Bank L-2 function-name F1 (95.65 versus 82.91) (Table 5).

## Connections
- [[xlam]] — the xlam-function-calling-60k dataset is Hammer's entire positive training set, and xLAM models are its main open baselines.
- [[apigen]] — the pipeline that produced that dataset.
- [[bfcl]] — Tables 2 and 4 and the irrelevance-proportion ablation all use it.
- [[granite-function-calling]] — appears in Hammer's Table 2 at 76.63 overall; it measures hallucination but does not train on irrelevance examples.
- [[toolace]] — a contemporaneous function-calling data pipeline, not evaluated here.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2410.04587 (arXiv v2, 10 Oct 2024).
- Corrections to the previous card version:
  - Affiliations "MadeAgents / Peking University / Baai" → OPPO Research Institute, Shanghai Jiao Tong University, Iowa State University. MadeAgents is the Hugging Face organization hosting the checkpoints, not an author affiliation.
  - Model sizes "0.5B, 1.5B, 7B" → 1.5B, 4B, 7B. There is no Hammer-0.5B in the paper, and the claim "Hammer-0.5B: BFCL-V1 ~78%" is removed.
  - "mask real function names ... ~30% of the time" and "30% masking is the authors' found optimum; 50% degrades general tool recall; 10% gives weak debiasing" → §5.5 runs a ratio ablation and reports a direction (higher ratio, better cross-task transfer; slower same-task learning) but names no optimum and no numbers for 10/30/50%.
  - "~30% of samples are irrelevance examples" → 7,500 examples against a 60k base, and §5.6 puts the optimum near 10%.
  - Masking covers parameter names and randomized default values too, not only function names (§4.1). The previous card described function names only.
  - "Hammer-7B: BFCL-V1 overall 87.9%, relevance 88.6% — matches GPT-4 on relevance" → Table 2 gives overall 83.92, relevance 92.68, irrelevance 72.87, on the 2024-09-20 snapshot. GPT-4-0125-Preview has relevance 97.56 and irrelevance 61.35.
  - "on BFCL-V2 irrelevance, Hammer-7B achieves ~90% — higher than xLAM-7B (~80%)" → irrelevance is 72.87 for Hammer-7B and 79.76 for xLAM-7B-fc; xLAM is **higher** on that column and lower on overall accuracy (Table 2).
  - "Ablation: function-name masking alone gives +7 relevance; irrelevance augmentation alone +10; combined +13" → no such ablation exists.
  - "Year: 2024 → Hammer 2.0 / 2.1 in 2025" → the card now describes only the artifact named by the slug, arXiv:2410.04587. Everything attributed to Hammer 2.0 and 2.1 has been removed (see below).
- Removed as unsupported by the source:
  - All Hammer 2.0 / 2.1 content: "Step 3 — Parameter perturbation (Hammer 2.0)", "Step 4 — Multi-turn augmentation (Hammer 2.1) adopts APIGen-MT-style trajectories", "~250K (Hammer 2.1)", "BFCL-V3 multi-turn strong performance". None of this is in this paper.
  - "Seed data: ... + ToolBench + in-house APIs" — the training set is xlam-function-calling-60k plus the irrelevance augmentation only.
  - "Output shape: ~150K training samples (Hammer 7B); single-turn mix 70% / multi-turn 30%", "Teacher model(s): GPT-4 and DeepSeek-Coder-V2 via APIGen + GPT-4 for irrelevance examples", "training 7B ~ 10K GPU-hours", "authors mitigate via 70/30 split".
  - "Call format: OpenAI `tool_calls` JSON; compatible with MLX / llama.cpp tool schemas" — no serialization or runtime claim of this kind appears.
  - "Irrelevance distribution matters: if irrelevant tools are too different from query topics, the model learns cheap lexical refusal only" — this was an audit lead and is not in the paper. The construction in §4.2 keeps the original distractors, so the question is not raised there.
  - "Hammer-7B ranks first on BFCL relevance-detection among sub-frontier models" — it ties Hammer-1.5B at 92.68 relevance, below xLAM-1.3B-fc and Granite-20B-FunctionCalling at 95.12 (Table 2).
- Not reported by the source: learning rate, batch size, number of epochs for the released checkpoints, optimizer, full-parameter versus adapter tuning, compute cost, license of the released models, and any long-context or distillation result.
