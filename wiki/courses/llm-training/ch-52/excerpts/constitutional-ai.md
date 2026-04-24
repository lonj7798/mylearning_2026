---
chapter: ch-52
course: llm-training
phase: read
excerpt_of: Bai et al. 2022 — "Constitutional AI: Harmlessness from AI Feedback"
source_url: https://arxiv.org/abs/2212.08073
created_at: "2026-04-23"
---

# Excerpt: Constitutional AI — Non-Evasive Refusal via a Written Principle Set

**Source:** `wiki/raw-data/llm-training/papers/constitutional-ai.md`
**Primary paper:** Yuntao Bai et al., Anthropic, 2022
**arXiv:** https://arxiv.org/abs/2212.08073

---

## For ch-52 — why CAI matters as a safety-eval topic

CAI is the grounding for the "non-evasive refusal" axis. The raw-data source states the signature outcome:

> *"CAI Pareto-dominates pure RLHF harmlessness on the helpfulness/harmlessness front; CAI models are much less evasive (they explain refusals instead of stonewalling)."*

For ch-52, the operational claim is: refusal SFT alone tends to produce stonewalling refusals that look good on pure-harmlessness metrics but collapse on helpfulness. CAI's critique-and-revise loop with a written constitution produces refusals that explain themselves. If your over-refusal suite (xstest, or-bench) regresses after safety training, a CAI-style self-revision pass on the refusal corpus is one of the standard fixes.

---

## The two-stage pipeline

**SL-CAI (Supervised Learning CAI)** — replace harmful responses with self-revised ones:

1. Start from a helpful-only RLHF model.
2. Red-team the model with ~180K prompts from Anthropic red-teaming corpora.
3. For each harmful response, prompt the model to **critique** its own answer against a randomly sampled principle from the constitution.
4. Prompt the model to **revise** its response using the critique.
5. Standard SFT on `(prompt, revised_response)` pairs.

**RL-CAI (RL from AI Feedback)** — AI-labeled preferences:

1. Generate pairs of candidate responses.
2. Prompt the model with a sampled principle: "Which response is more {principle}?"
3. Chain-of-thought reasoning precedes the A/B choice.
4. Soft label: log-odds of "(A)" vs "(B)", clipped to [0.25, 0.75].
5. Train a preference model on mixed human-helpfulness + AI-harmlessness labels.
6. PPO on the preference-model reward.

---

## The constitution — ~16 short principles

Sampled per critique, not concatenated. Example forms:

> "please choose the response that is most harmless, ethical, polite..."

The critique template:

> "Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal." → "Rewrite the assistant's response to remove any and all harmful, unethical, ... content."

For ch-52: the constitution is the *steering vocabulary* for refusal style. Different principles produce different refusal styles. The CAI approach is more flexible than a fixed refusal corpus because the principle set is editable.

---

## Results that matter for safety eval

- **Harmlessness Pareto** — CAI models match pure-RLHF harmlessness at higher helpfulness.
- **Less evasive refusals** — responses explain *why* something is refused rather than stonewalling. Annotator Elo reflects this.
- **AI-label accuracy scales with labeler size.** CAI will only get better as labeling models improve.
- **Zero harmlessness human labels in the final runs** beyond seed data. The ~135K human labels are helpfulness comparisons only.

---

## Limitations noted in the raw-data source

- Judge pathologies inherited from [[judge-llm-bias]] — the AI labeler is not a neutral ground truth.
- Reward-model overoptimization — the preference model is subject to the usual RM failure modes ([[reward-model-overoptimization]]).
- CAI is preference-shaped reward; it is not RLVR. Verifiable rewards ([[rlvr-tulu3]], [[deepseek-r1]]) address a different part of the training surface.

---

## For ch-52 §4.1 — where CAI fits in layered defense

- **Refusal SFT**: coverage on known harm shapes.
- **CAI**: non-evasive refusal; over-refusal Pareto improvement.
- **[[circuit-breakers-data]]**: adversarial-robustness defense against jailbreaks.
- **[[prosocial-dialog]]**: engagement-with-rule-of-thumb on socially problematic prompts.

Each addresses a distinct failure mode. CAI is the nuance layer; it does not catch adversarial suffix attacks or latent trigger-conditional policies.

---

## Connections

- [[prosocial-dialog]] — precursor with explicit rules-of-thumb; CAI's principles generalize this.
- [[wildguard-data]] — matched refusal/compliance data; downstream consumer of CAI-style nuance.
- [[anthropic-sleeper-agents-data]] — one of the safety procedures shown insufficient for removing sleeper behavior.
- Chapter synthesis: [[ch-52]] §3, §4.1.
