---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/deepseek-r1.md
source_url: https://arxiv.org/abs/2501.12948
created_at: "2026-04-23"
---

# Excerpt: DeepSeek-R1 - the cold-start SFT ch-32 uses as its worked example

**Source library:** `wiki/raw-data/llm-training/model-reports/deepseek-r1.md`
**Artifact:** ~800K cold-start traces bridging V3-Base to the RL pipeline

---

## Why this source anchors ch-32

R1 vs R1-Zero is the cleanest controlled comparison of cold-start SFT's role in a reasoning pipeline. Both start from the same V3-Base. R1-Zero skips cold-start entirely and runs pure RL; R1 inserts a cold-start SFT pass and then runs the same RL. The failure modes of R1-Zero (language mixing, readability collapse) plus the fix (cold-start SFT) are the most-quoted evidence that cold-start is a *format* intervention, not a *capability* one.

Ch-32 uses this comparison to argue: cold-start SFT is a **format installer**. R1-Zero proves RL can install reasoning capability alone; R1 proves you still want cold-start to ship. The two claims are not contradictory; they describe different deliverables.

---

## The attested pipeline ch-32 transcribes

From the source (lines 31-49):

**R1-Zero (pure RL):**
- Base: DeepSeek-V3-Base (671B MoE).
- Algorithm: GRPO, rule-based reward only (accuracy via sympy-verified final answer + format via regex match on `<think>...</think><answer>...</answer>`).
- Outcome: emergent long CoT, self-reflection, multi-approach exploration. Failure: mixed-language output + poor readability.

**R1 (full recipe):**
1. **Cold-start SFT** on ~800K curated reasoning examples with human-readable CoT format - fixes readability.
2. **Stage-1 Reasoning RL** with GRPO + rule-based rewards.
3. **Rejection-sampling SFT** using stage-1 RL model; V3-judge filter; ~600K reasoning + ~200K non-reasoning.
4. **Stage-2 Alignment RL** with helpfulness + harmlessness preference rewards.

The ~800K cold-start count is the number ch-32 quotes. The split (~600K reasoning + 200K non-reasoning) is attested after stage 3, not cold-start - but the cold-start likely followed a similar split; the model report conflates these. Ch-32 is explicit that the split is best-effort inference, not attested verbatim.

---

## The RL stage-1 hyperparameters ch-32 cites

From the source (lines 41-47):

- Learning rate: 3e-6
- KL coefficient: 0.001
- GRPO clip ratio (eps): 10 (intentionally loose - tight clipping destroys exploration)
- Sampling temperature: 1.0 for rollouts
- Rollouts: 16 samples per prompt (group size G=16)
- Max generation length: 32,768 tokens
- Batch size: 32 unique prompts/step -> 512 training samples/step

Ch-32 quotes these to make concrete what "reasoning RL" looks like as a stage: very long rollouts (32K gen), large groups (G=16), loose clipping (eps=10), small KL (0.001). This is the RL profile that cold-start SFT prepares a policy for - if the cold-start got the format wrong, every one of these 512 samples per step would be polluted.

---

## What the cold-start filter criteria actually are

The source does not give an exact filter list, but the implied criteria (inferred from context) are:

- **Correctness**: final answer verified against ground truth (rule-based grader).
- **Monolingual**: the explicit fix for R1-Zero's language-mixing failure.
- **Readable CoT format**: `<think>...</think><answer>...</answer>` template with natural-language reasoning inside `<think>`, not dense symbol-math.
- **Length cap**: within the 32K generation limit used in RL.
- **No hollow-think exploits**: `<think>` block must contain substantive reasoning, not a single line or empty content.

Ch-32 labels this list as inferred-from-context because the model report does not enumerate it. The inference is defensible because each criterion maps directly to a named R1-Zero failure mode.

---

## Distillation result ch-32 uses as supporting evidence

From the source (lines 51-54):

- 800K reasoning traces from R1 used to SFT Qwen-2.5 (1.5B/7B/14B/32B) and Llama-3 (8B/70B) students.
- No RL on students; pure SFT.
- Distilled-R1-Qwen-32B beats o1-mini on MATH-500 and AIME.

The distillation result is why ch-32 cares about cold-start size: 800K traces are **large enough to SFT a separate 32B model into competitive reasoning behaviour**. If cold-start were merely a format hint, it would not carry enough signal to train a distilled model. The cold-start therefore carries format *and* capability information - but the R1 vs R1-Zero comparison isolates which component is the cold-start's job vs RL's job.

---

## Connections

- **ch-31** - DPO / preference optimization background; R1's alignment RL maps onto this lineage.
- **ch-35** - distillation as a separate case study; ties back to the 800K reasoning traces as reusable data.
- **[[grpo]]** - algorithmic details.
- **[[deepseek-v3]]** - base model the cold-start is applied to.
- **[[rlvr-tulu3]]** - verifier-grounded reward lineage.
