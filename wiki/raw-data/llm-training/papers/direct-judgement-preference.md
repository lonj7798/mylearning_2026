<!-- scope: 2024/2025 synthetic-judge-as-pref-source line; distilled LLM judges produce preference pairs
     deps: [[ultrafeedback-construction]], [[rlcd]]
     see-also: [[west-of-n]], [[self-rewarding-lm]], [[judge-llm-bias]]
-->

# Direct Judgement / Synthetic-Judge Preference Data (2024–2025 line)
- **Core Insight:** You don't need humans *or* GPT-4 for preference labels — a dedicated **judge LM**, trained via DPO on contrastive judgment pairs (generating rationales + verdicts), can itself generate the preference pairs needed to train new policies, closing the AI-feedback loop end-to-end.
- **Guideline:** When building a scalable alignment stack without API dependency, (1) train a generative judge on open judgement data (e.g. JudgeLM / HelpSteer / reannotated Skywork-Reward), (2) have it score candidate pairs with explanation, (3) use its verdicts as DPO/RPO labels.
- **Authors / papers (representative):**
  - "Learning LLM-as-a-Judge for Preference Alignment" (Con-J, ICLR 2025).
  - "Self-Taught Evaluators" (Meta, 2024): iterative self-improving judge.
  - "J1: Incentivizing Thinking in LLM-as-Judge via RL" (2025).
  - Skywork Reward generative-judge line (2024).
- **Year:** 2024–2025
- **URL examples:** https://openreview.net/forum?id=HZVIQE1MsJ (Con-J) ; https://arxiv.org/abs/2408.02666 (Self-Taught Evaluators) ; https://arxiv.org/abs/2505.10320 (J1)
- **Relevant topics:** synthetic judge, generative reward model, LLM-as-judge, preference alignment

## Abstract
A 2024–25 thread of work collapses the UltraFeedback pattern (external judge rates pairs, humans bootstrapped GPT-4) into a **self-contained generative judge**. The judge is itself a language model, trained with DPO on contrastive judgment pairs (chosen/rejected verdicts with rationales). It generates natural-language rationales plus verdicts rather than scalar scores, which:
- improves interpretability,
- reduces scalar-RM brittleness,
- allows iterative self-improvement (the policy can be retrained with judge-generated prefs, then the judge retrained with improved-policy prefs).
"Self-Taught Evaluators" (Meta 2024) showed a judge trained only on synthetic prefs surpasses GPT-4-as-judge on RewardBench; "J1" (2025) adds RL training for the judge's chain-of-thought, pushing accuracy further.

## Key Contributions (aggregate across papers)
- **Con-J / "Learning LLM-as-a-Judge for Preference Alignment":** DPO-train a judge to generate preference pairs in natural-language form; improves interpretability and robustness to dataset biases.
- **Self-Taught Evaluators (Meta):** iterative self-improvement — judge generates prefs, new policy, new judge prefs; no human labels after bootstrap.
- **J1 (2025):** adds RL on the judge's thinking trace → stronger chain-of-thought → better judgments.
- Demonstrated **~40K synthetic preference pairs** (20K SFT + 20K DPO) suffice to beat models trained with 2–40× more data on RewardBench-class benchmarks.

## Key Figures/Tables to Study
- Con-J: interpretability + win-rate comparison vs scalar RMs.
- Self-Taught Evaluators: iterative RewardBench trajectory — GPT-4 baseline crossed after ~3 iterations.
- J1: RL chain-of-thought judge vs DPO-only judge — delta across benchmarks.

## Synthesis pipeline (REQUIRED — be concrete, representative)
- **Seed input:** a small real-pref seed (HelpSteer / HH-RLHF / a tiny GPT-4-labelled set) *or* a starting generative judge.

- **Generation step(s):**
  - **Pair sampling:** generate N responses per prompt (heterogeneous generators or single policy with temperature).
  - **Noisy-negative trick (Con-J):** perturb the original instruction, generate a response to the noisy instruction, treat as a plausible "rejected."
  - **Judge prompt:** judge model receives `(prompt, response_A, response_B)` and emits `(rationale, verdict ∈ {A, B})` via DPO-trained chat template.
  - **Self-Taught iteration:**
    - Round 0: seed judge (fine-tuned on tiny labelled set).
    - Round k: use judge_k to label a fresh pool → train judge_{k+1} with DPO on its own decisions vs alternative judgments.
    - Stopping criterion: RewardBench accuracy saturation or iteration cap.

- **Filtering/rescoring:** consistency check — swap response order to detect position bias; drop inconsistent verdicts.

- **RL / RLHF step:** use judge-generated prefs to train policy via DPO / RPO / IPO.

- **Output shape:** each preference pair carries a chosen response, a rejected response, and a rationale.

- **Teacher model(s):** self — no external LLM required after initial bootstrap.

- **Cost estimate:** much cheaper than GPT-4-judged UltraFeedback at equal pref count (no API).

## Quality / diversity evaluation
- RewardBench: Self-Taught Evaluator surpasses GPT-4-as-judge after 3 iterations.
- Con-J: strong robustness to label noise and format bias.
- J1: highest-accuracy open judge at publication on RewardBench-hard.
- Downstream DPO policies trained with synthetic judge prefs match or exceed GPT-4-labelled DPO on instruction-following benchmarks.

## Risks + gotchas
- **Self-reinforcement:** iterative judge training can converge to a narrow rubric (judge-collapse analogue of [[model-collapse]]).
- **Rationale hallucination:** natural-language rationales can be post-hoc justifications, not causes.
- **Position bias / format bias** still present unless explicitly audited.
- **No fresh-real-data anchor** — pure-synthetic iteration risks degradation without periodic real-pref injection.
- **Judge-as-weapon:** same judge model used to label prefs and evaluate benchmarks creates leakage — RewardBench's increasing close relationship to training-time judges is a known measurement issue.

## Connections
- Extends [[ultrafeedback-construction]] by internalizing the GPT-4 judge.
- Sibling of [[self-rewarding-lm]] (one model plays judge and policy simultaneously).
- Reduces dependency on [[hh-rlhf]] human labels.
- Relevant to [[judge-llm-bias]] (Zheng 2023 MT-Bench) — bias modes that transfer to synthetic judges.
- Key enabling technology for next-gen [[tulu-3-sft-mix]] / Nemotron-style post-training without API spend.
