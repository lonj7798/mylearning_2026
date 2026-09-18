<!-- scope: Qi et al. (arXiv:2310.03693): red-teaming study of fine-tuning safety-aligned LLMs (GPT-3.5 Turbo 0613 via API, Llama-2-7b-Chat full-parameter and PEFT) with 10-100 harmful examples, 10 identity-shifting examples, and benign Alpaca/Dolly/LLaVA-Instruct data; GPT-4 judge on a 330-prompt policy benchmark; mitigation tests (safety-data mixing, moderation, backdoor evasion of audits)
     see-also: [[shallow-safety-alignment]], [[emergent-misalignment]], [[alpaca]], [[llama-2]], [[catastrophic-forgetting-continual-finetuning]]
-->

# Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!
- **Core Insight:** Fine-tuning GPT-3.5 Turbo through OpenAI's API on 10 harmful instruction-response pairs for 5 epochs, at a cost below $0.20, raised its harmfulness rate on a 330-prompt benchmark from 1.8% to 88.8%, and one epoch of fine-tuning on the benign Alpaca data raised it from 5.5% to 31.8% (Table 1, Table 3, Remark 1).
- **Guideline:** When an aligned model is fine-tuned on any downstream data, including benign instruction data, re-run a harmful-instruction evaluation on the fine-tuned checkpoint before release, because every benign dataset tested (Alpaca and Dolly on both models, LLaVA-Instruct on Llama-2-7b-Chat) raised the harmfulness rate (Table 3); mixing in refusal data reduced but did not remove the increase (Table 4), and a trigger-phrase backdoor kept a 4.2% rate on plain benchmark prompts while reaching 63.3% with the trigger (Table 5).
- **Authors:** Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, et al. (Princeton University, Virginia Tech, IBM Research, Stanford University)
- **Year:** 2023 (arXiv v1 2023-10-05; the only arXiv version; the PDF is marked "A Preprint")
- **URL:** https://arxiv.org/abs/2310.03693
- **Source type:** paper
- **Relevant topics:** safety alignment, forgetting during fine-tuning, harmful fine-tuning attacks, benign fine-tuning, PEFT, safety data mixing, LLM-as-judge evaluation, data moderation

## Abstract
Existing safety alignment restricts harmful behavior at inference time but does not cover the case where users can fine-tune the model. Red-teaming experiments show that a few adversarially designed training examples remove safety alignment: 10 examples, costing less than $0.20 through OpenAI's API, make GPT-3.5 Turbo respond to nearly any harmful instruction. Fine-tuning on benign, commonly used datasets also degrades safety alignment, to a lesser extent. The authors conclude that fine-tuning introduces safety risks that current safety infrastructure does not address, and they analyze possible mitigations (Abstract).

## Key Contributions
- A three-level risk outline: explicitly harmful fine-tuning data (Level 1), implicitly harmful data that passes moderation (Level 2), and benign data (Level 3) (§1, §3).
- The harmful examples demonstration attack (§4.2) and the identity shifting attack, whose 10 examples contain no toxic content and were not flagged by the OpenAI moderation API or the GPT-4 judge (§4.3).
- Measurement of safety degradation after benign fine-tuning, including learning-rate, batch-size, and epoch ablations (§4.4, App. G).
- A policy-oriented benchmark of 11 prohibited-use categories × 30 prompts = 330 prompts, scored 1-5 by a GPT-4 judge (§4.1, App. A-B).
- Tests of mitigations: safety-data mixing, moderation tools, and a backdoor that evades post-fine-tuning audits (§5.1, App. H).

## Key Figures/Tables to Study
- Tables 1-3: harmfulness score and rate before and after the three kinds of fine-tuning for both models.
- Fig. 6: per-category harmfulness after each fine-tuning; categories degrade unevenly.
- Table 4: effect of mixing safety samples into each fine-tuning set (GPT-3.5 Turbo).
- Table 8: MT-Bench scores of fine-tuned GPT-3.5 Turbo; Table 11: PEFT results; Tables 12-13: Alpaca hyperparameter ablations.

## Technical Details
**Models and objective.** GPT-3.5 Turbo 0613 is fine-tuned through the OpenAI API, where the number of epochs is the only controllable hyperparameter; Llama-2-7b-Chat is fine-tuned with full parameters and AdamW following the official llama-recipes (§4.1). Each example is a one-round conversation (system prompt s_i, user input u_i, target a_i), trained by minimizing Σ_i −log p(a_i | [s_i, u_i]; θ + Δθ), where θ are the aligned weights and Δθ the update (Eq. 1). The same system prompt is used in fine-tuning and evaluation for each dataset (§4.1).
**Metrics.** The GPT-4 judge receives the usage policy, the harmful instruction, the output, and a rubric, and returns a score from 1 to 5; harmfulness rate is the fraction of prompts scored 5 (§4.1). Decoding uses temperature 0 and top-p 0 (§4.1). In a human study on 500 sampled GPT-3.5 Turbo pairs, Fleiss' κ among four annotators was 0.607 before and 0.706 after a consensus meeting (100 examples per annotator); on the final 400 pairs, Cohen's κ between humans and the judge was 0.539, Spearman ρ 0.84, and Cohen's κ 0.792 for a binary split at score 3 (App. B).
**Level 1 (Table 1, 5 epochs).** Harmful examples are sampled from the Anthropic red-team dataset (N = 10, 50, 100), manually checked, and disjoint from the benchmark (§4.2). Harmfulness rate: GPT-3.5 Turbo 1.8% → 88.8% / 87.0% / 91.8%; Llama-2-7b-Chat 0.3% → 50.0% / 80.3% / 80.0%. The 10-shot Llama-2 attack (batch 10, 5 epochs) is 5 gradient steps (Remark 1). For the 100-shot attack, results are not sensitive to the number of epochs (Fig. 3).
**Level 2 (Table 2, 10 examples).** Harmfulness rate after 3 / 5 / 10 epochs: GPT-3.5 Turbo 0% → 7.3% / 49.1% / 87.3%; Llama-2-7b-Chat 0% → 54.2% / 72.1% / 68.2%.
**Level 3 (Table 3, 1 epoch).** Harmfulness rate: GPT-3.5 Turbo Alpaca 5.5% → 31.8%, Dolly 4.5% → 23.9%; Llama-2-7b-Chat Alpaca 0.3% → 16.1%, Dolly 0.6% → 12.1%, LLaVA-Instruct 0% → 18.8%. For Llama-2 on Alpaca, LR 5e-5 gives 46.4% / 37.9% / 31.5% / 34.2% at batch 16 / 32 / 64 / 128, against 23.6% / 20.6% / 15.8% / 16.1% at LR 2e-5 (Table 12). More epochs do not raise harmfulness further: GPT-3.5 Turbo on Alpaca gives 31.8% / 27.3% / 23.0% after 1 / 3 / 5 epochs (Table 13).
**Mitigations.** Mixing safety data from Bianchi et al. into GPT-3.5 Turbo fine-tuning: 100-shot attack 91.8% → 23.0% with 100 safety samples; identity shift 87.3% → 13.0% with 10; Alpaca 31.8% → 19.7% with 500 and 22.1% with 1000 (Table 4); none returns to the initial model's level (§5.1). Moderation of the 100 harmful instructions flagged 17% (OpenAI), 4% (Perspective, threshold ≥ 0.7), 6% (Detoxify, ≥ 0.7), and of the 100 harmful answers 21%, 17%, 27%; the GPT-4 judge scored all 100 as 5 (§5.1). A backdoored GPT-3.5 Turbo (100 harmful examples with a 3-word trigger plus 100 safety examples) has a 4.2% harmfulness rate on plain prompts and 63.3% with the trigger (Table 5).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT-3.5 Turbo 0613 (harmful examples) | not reported | SFT | examples; epochs | 10 / 50 / 100; 5 epochs (API; epochs only) | arXiv:2310.03693v1 §4.1-4.2, Table 1 | verified 2026-09-14 | Fig. 3 epoch ablation (100-shot) |
| Llama-2-7b-Chat (harmful examples) | 7B | SFT | examples; epochs; LR; batch; method | 10 / 50 / 100; 5; 5e-5; 10; full-parameter, AdamW | §4.1-4.2 | verified 2026-09-14 | Fig. 3 epoch ablation (100-shot) |
| GPT-3.5 Turbo 0613; Llama-2-7b-Chat (identity shifting) | not reported; 7B | SFT | examples; epochs; Llama-2 LR; batch | 10; 1 / 3 / 5 / 10; 5e-5; 10 | §4.3, Table 2 | verified 2026-09-14 | Table 2 over epochs |
| Llama-2-7b-Chat (benign) | 7B | SFT | data; epochs; LR; batch | Alpaca 50K (52K minus 1,902 safety-related samples); Dolly 14,624 (387 removed); LLaVA-Instruct-80K; 1 epoch; 2e-5; 128; AdamW | §4.4; App. G.1 | verified 2026-09-14 | "officially recommended" settings; Table 12 (LR, batch), Table 13 (epochs) |
| GPT-3.5 Turbo 0613 (benign) | not reported | SFT | data; epochs | Alpaca, Dolly; 1 epoch by default | §4.4 | verified 2026-09-14 | Table 13: 1 / 3 / 5 epochs |
| Llama-2-7b-Chat LoRA (Levels 1 / 2 / 3) | 7B | SFT | LR; batch; epochs | 1e-3, 10, 10 / 1e-3, 10, 20 / 1e-4, 16, 1 | App. F | verified 2026-09-14 | Levels 1-2 searched for the attacker; Level 3 officially recommended |
| Llama-2-7b-Chat LLaMA-Adapter; Prefix (Levels 1 / 2 / 3) | 7B | SFT | LR; batch; epochs | Adapter: 1e-2, 10, 20 / 1e-2, 2, 10 / 1e-2, 16, 1; Prefix: 1e-2, 10, 30 / 1e-2, 2, 20 / 1e-2, 16, 1 | App. F | verified 2026-09-14 | same as above |

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality and forgetting.** The authors name catastrophic forgetting of the initial alignment and the tension between helpfulness and harmlessness as possible causes of Level-3 degradation (§1, §3.2; Interpretation). Categories #4 Malware, #6 Economic Harm, #7 Fraud/Deception, and #9 Political Campaigning degrade more than others across all benign cases (Remark 5, Fig. 6).
- **Capability after fine-tuning.** MT-Bench for GPT-3.5 Turbo 0613 is 8.00 before fine-tuning, 7.46 after the 100-shot attack, 6.62 after identity shifting (10 epochs), and 6.68 after Alpaca (Table 8). On a LegalBench subset, the 100-shot model scores higher on some tasks and lower on others (Table 9).
- **PEFT.** On Llama-2-7b-Chat, LoRA, LLaMA-Adapter, and Prefix tuning all raise harmfulness for the 100-shot attack (80.6%, 67.6%, 42.4%) and for Alpaca (25.2%, 26.4%, 24.8%, against 16.1% for full-parameter fine-tuning); on identity-shifting data Prefix tuning stays at a 0% rate (Table 11, App. F).
- **Negative feedback.** Refusal examples mixed into fine-tuning are negative as content (targets that decline the request) and reduce harmfulness (Table 4). Moderation filters discard harmful samples (negative marginal value) but missed most of the harmful examples and all identity-shifting examples (§5.1).
- **Measurement limits.** The authors describe harm assessment as "somewhat conceptual" without magnitude of harm (§6). On the public AdvBench (520 instructions, keyword-based attack success rate), Alpaca fine-tuning raises the rate from 0.8% to 20.2% for GPT-3.5 Turbo and from 0% to 5.2% for Llama-2-7b-Chat (Table 10, App. E).

## Connections
- [[shallow-safety-alignment]]: a separate paper whose title argues that safety alignment should extend beyond the first few output tokens; Remark 1 here describes current safety tuning as producing "relatively surface-level changes".
- [[emergent-misalignment]]: narrow fine-tuning producing broad misalignment, a related form of behavior change after fine-tuning.
- [[alpaca]]: the benign dataset used for Level 3, with 1,902 safety-related samples removed (App. G.1).
- [[llama-2]]: the Llama-2-7b-Chat model, aligned with instruction tuning and iterative RLHF on safety data (§4.1).
- [[catastrophic-forgetting-continual-finetuning]], [[scaling-laws-forgetting]], [[lora-learns-less-forgets-less]]: studies of forgetting during fine-tuning; compare with the safety degradation measured here.
- [[harmbench-data]]: a harmful-behavior evaluation pipeline that can serve as a post-fine-tuning audit; [[xstest]]: a test suite for exaggerated safety (over-refusal), the opposite failure to the one measured here.
- [[judge-llm-bias]]: known biases of LLM judges, relevant to the GPT-4 judge used here.
- [[anthropic-safety-research]]: the Anthropic red-team data (Ganguli et al.) is the source of the harmful examples (§4.2).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2310.03693 (v1, 2023-10-05; full PDF including App. A-H; a non-layout pdftotext pass recovered the §4.2 Llama-2 settings).
- Audit claims not found in the source: none.
- Not reported by the source: GPT-3.5 Turbo parameter count and API fine-tuning hyperparameters other than epochs; safety effects on tasks beyond MT-Bench and the LegalBench subset.
