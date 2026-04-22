<!-- scope: Hsieh 2023 — synthesize rationales + labels from teacher LLM for small student (ACL 2023)
     deps: [[star]]
     see-also: [[orca]], [[deepseek-r1-distill-synth]], [[quiet-star]]
-->

# Distilling Step-by-Step: Rationale-Plus-Label Distillation for Small Models (Hsieh 2023)
- **Core Insight:** Extract natural-language rationales (not just labels) from a large teacher LLM via few-shot CoT prompting, then train a small student in a **multi-task** framework (predict label + generate rationale) — this outperforms label-only distillation and lets a 770M student beat a 540B few-shot teacher on multiple benchmarks using only 80% of the available data.
- **Guideline:** When distilling a teacher LLM into a smaller model, don't just use hard labels; extract CoT rationales via few-shot prompting, train the student jointly with a rationale-generation head alongside the label-prediction head.
- **Authors:** Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, Tomas Pfister (Washington / Google)
- **Year:** 2023 (ACL 2023 Findings)
- **URL:** https://arxiv.org/abs/2305.02301
- **Relevant topics:** rationale distillation, multi-task learning, CoT, small-model reasoning

## Abstract
Distilling Step-by-Step extracts rationales from a teacher LLM (PaLM-540B) via few-shot chain-of-thought prompting, then trains a smaller T5 student on a multi-task objective: predict the label AND generate the rationale. The multi-task student outperforms label-only distillation and even beats the teacher's few-shot performance on many benchmarks — a T5-770M beats PaLM-540B few-shot on ANLI / e-SNLI / CQA / SVAMP using only 80% of available labeled data. The paper is a foundational reference for rationale-as-supervision distillation.

## Key Contributions
- **Two-stage method:**
  1. Extract rationales from teacher via few-shot CoT prompting on unlabeled data.
  2. Train student with multi-task objective: label prediction + rationale generation.
- Demonstrated small student > large teacher few-shot on several reasoning tasks.
- Data efficiency: 80% of data suffices.
- Foundational conceptual parent of 2024–25 "teacher-CoT distillation" work (Orca, R1-distill).

## Synthesis pipeline (concrete)
- **Seed input:** unlabeled (or labeled) task data — e.g., ANLI premises/hypotheses, CQA questions.
- **Generation step:**
  - Few-shot CoT prompt PaLM-540B with 3–8 exemplars showing `question → rationale → answer`.
  - Run on every training example; collect `(x, rationale_teacher, label_teacher)`.
- **Filtering:** drop teacher outputs with wrong format or answers failing a verifier (when labels available).
- **Student training:**
  - Input: task question `x`.
  - Output heads: (a) label `y` via classification or sequence head, (b) rationale `r` via sequence head.
  - Loss: `L = L_label + λ · L_rationale` (multi-task).
- **Teacher:** PaLM-540B (original paper); method applies to any strong LLM.
- **Student:** T5 family (220M / 770M / 11B) or smaller decoder-only models.

## Training outcome
- T5-770M Distilled-Step-by-Step beats PaLM-540B few-shot on:
  - ANLI (natural language inference),
  - e-SNLI,
  - CQA (CommonsenseQA),
  - SVAMP (math word problems).
- Using only 80% of available labeled data.
- Student's rationale-generation head acts as regularization — improves label prediction even when the rationale head is discarded at inference.

## Why rationales help
- Labels alone force the student to memorize input-output maps; rationales force it to internalize *the chain of reasoning* — a more compact representation.
- Multi-task training: the rationale head's gradients push the shared encoder toward features useful for reasoning, not just surface pattern-matching.

## Risks + gotchas
- **Teacher rationale quality** bounds student ceiling; hallucinated rationales teach wrong reasoning patterns.
- **Verifier matters** when labels are available — use correctness filtering.
- **Task scope** in the original paper is moderate-reasoning; for deep reasoning (MATH, competition-level), later work (R1 distill) requires dramatically longer traces.
- **Inference cost**: if student emits rationales at inference, compute doubles; many downstream uses strip the rationale head.

## Connections
- Conceptual parent of [[orca]] / [[orca-2]]'s explanation-trace distillation.
- Direct ancestor of 2025-scale reasoning distillation: [[deepseek-r1-distill-synth]], Bespoke-Stratos, open-R1.
- Sibling of [[star]] (self-generated rationales) and [[quiet-star]] (learnable internal thoughts).
- Enables cost-efficient reasoning capability transfer — central to small-model post-training economics.
