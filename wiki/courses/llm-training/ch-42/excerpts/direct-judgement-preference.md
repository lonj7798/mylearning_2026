---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/direct-judgement-preference.md
source_url: https://openreview.net/forum?id=HZVIQE1MsJ
created_at: "2026-04-23"
---

# Excerpt: Synthetic-Judge Line — Con-J, Self-Taught Evaluators, J1

**Source library:** `wiki/raw-data/llm-training/papers/direct-judgement-preference.md`
**Papers:**
- Con-J, *Learning LLM-as-a-Judge for Preference Alignment*, ICLR 2025.
- Wang et al. (Meta), *Self-Taught Evaluators*, 2024. ([arxiv.org/abs/2408.02666](https://arxiv.org/abs/2408.02666))
- *J1: Incentivizing Thinking in LLM-as-Judge via RL*, 2025. ([arxiv.org/abs/2505.10320](https://arxiv.org/abs/2505.10320))

---

## Why this source anchors ch-42

This line closes the AI-feedback loop. Con-J, Self-Taught Evaluators, and J1 train a **dedicated judge LM** via DPO (and RL) on contrastive judgment pairs; the judge then generates the preference pairs that train new policies. No API, no human judge, no GPT-4 dependency after bootstrap. Ch-42 §5 is the chapter's primary discussion of this pipeline and the new hacking risks it introduces — judge-collapse, rationale hallucination, RewardBench leakage.

Raw-data header:

> **Core Insight:** You don't need humans *or* GPT-4 for preference labels — a dedicated **judge LM**, trained via DPO on contrastive judgment pairs (generating rationales + verdicts), can itself generate the preference pairs needed to train new policies, closing the AI-feedback loop end-to-end.

## Con-J (ICLR 2025)

DPO-train a judge to generate preference pairs in natural-language form (rationale + verdict). Key trick: the **noisy-negative**. Perturb the original instruction, generate a response to the noisy instruction, treat the response as a plausible "rejected." This manufactures training pairs without a stronger teacher model.

Properties:
- Natural-language rationales improve interpretability.
- DPO loss on judgments is more robust to dataset bias than scalar BT on pair preferences.
- The judge is drop-in for downstream DPO/RPO/IPO policy training.

## Self-Taught Evaluators (Meta 2024)

The iterative-bootstrap variant:

- **Round 0.** Seed judge fine-tuned on a tiny labeled set (HelpSteer / HH-RLHF / small GPT-4-labeled seed).
- **Round k.** Use `judge_k` to label a fresh pool of pairs; train `judge_{k+1}` with DPO on `judge_k`'s decisions vs alternative judgments.
- **Stopping criterion.** RewardBench accuracy saturation or iteration cap.

Empirically, the self-taught judge surpasses GPT-4-as-judge on RewardBench after ~3 iterations with no human labels after bootstrap.

## J1 (2025)

Adds RL to the judge's own chain-of-thought. The DPO-trained judge's CoT becomes the optimization target; a learned reward on judgment quality shapes stronger reasoning before the verdict. J1 is the highest-accuracy open judge on RewardBench-hard at publication.

## Throughput claim

~40K synthetic preference pairs (20K SFT + 20K DPO) suffice to beat models trained with 2–40× more data on RewardBench-class benchmarks. This is the headline for "synthetic judges are compute-efficient."

## Risks this line introduces

Four novel hacking modes, all flagged in ch-42 §5:

### Judge-collapse

Iterative self-improvement can converge to a narrow rubric — the judge analogue of [[model-collapse]]. Round-k judges are trained on Round-(k−1) decisions, so any bias in the seed compounds. Mitigation: periodic injection of fresh real-preference data to anchor the distribution.

### Rationale hallucination

Natural-language rationales can be post-hoc justifications rather than causes. A judge can produce a plausible-sounding explanation for an arbitrary verdict. Audit: rubric-ablation — strip the rubric from the prompt and see whether the verdict distribution shifts. If not, the rubric wasn't driving the decision.

### Position / format bias persistence

Synthetic judges inherit every [[judge-llm-bias]] bias unless explicitly audited. Con-J's consistency check (swap response order, drop inconsistent verdicts) is the minimum hygiene; not doing it leaves the position-bias hole wide open.

### RewardBench leakage

Using the same judge family for training and benchmarking creates a measurement loop. RewardBench's increasing close relationship to training-time judges is a known measurement issue. The benchmark becomes a measure of judge-family preference rather than model quality.

## Synthesis pipeline (summary)

- **Seed input:** small real-pref seed *or* starting generative judge.
- **Pair sampling:** N responses per prompt (heterogeneous generators or single policy with temperature).
- **Noisy-negative (Con-J):** perturb instruction, generate response to noisy instruction, treat as rejected.
- **Judge prompt:** `(prompt, response_A, response_B) → (rationale, verdict ∈ {A, B})` via DPO-trained chat template.
- **Self-Taught iteration:** repeat with judge_k labeling fresh pool → judge_{k+1}.
- **Consistency check:** swap response order; drop inconsistent verdicts.
- **RL step:** use judge-generated prefs for policy DPO / RPO / IPO.

## No fresh-real-data anchor is dangerous

The raw-data source is explicit: pure-synthetic iteration risks degradation without periodic real-preference injection. This is the judge-side analogue of synthetic-data model collapse — every downstream generation is labeled by a judge trained on upstream generations.

## Connection to the broader defense stack

The synthetic-judge line sits between [[constitutional-ai]] (principles → critiques, still with some human seeding) and a fully automated alignment loop. It reduces cost and API dependency, but amplifies the judge-collapse and leakage risks. Ch-42 §5 treats this as a live tradeoff: cheaper alignment, but more attention needed on auditing.

## Takeaways for the chapter

1. Con-J / Self-Taught Evaluators / J1 collapse the external-judge dependency into a trainable LM.
2. Self-taught judges can surpass GPT-4 on RewardBench after ~3 iterations.
3. Judge-collapse is the novel risk — periodic real-preference injection is required.
4. Rationale hallucination means legible ≠ correct; audit with rubric ablation.
5. RewardBench is increasingly a judge-family-preference benchmark; cross-family evaluation is required.
