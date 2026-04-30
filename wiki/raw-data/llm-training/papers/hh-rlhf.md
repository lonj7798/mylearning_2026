<!-- scope: Anthropic HH-RLHF — real-human-preference baseline for comparison to synthetic prefs
     deps: [[rlhf-instructgpt]]
     see-also: [[constitutional-ai]], [[ultrafeedback]], [[west-of-n]]
-->

# HH-RLHF: Training a Helpful and Harmless Assistant with RLHF (Anthropic)
- **Core Insight:** Helpfulness and harmlessness are *partially anti-correlated* axes — a model trained on only one corrodes on the other — so they must be trained jointly; 161K human-preference dialogues collected across both axes form the baseline against which all later synthetic preference datasets are measured.
- **Guideline:** Whenever you evaluate a synthetic preference pipeline (UltraFeedback / RLCD / West-of-N), benchmark against an HH-RLHF-trained RM on RewardBench; HH-RLHF remains the canonical "real-human" anchor.
- **Authors:** Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (Anthropic)
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2204.05862 ; https://github.com/anthropics/hh-rlhf
- **Relevant topics:** human preference data, RLHF, helpfulness vs harmlessness tradeoff, baseline dataset

## Abstract
Anthropic's HH paper introduces 161K human-preference dialogues collected in two campaigns: one optimizing helpfulness (crowdworkers rank AI responses by usefulness), one optimizing harmlessness (red-team crowdworkers try to elicit harmful output and label pair harmful/less-harmful). A preference model trained on one alone degrades on the other; training on the mix gives a model that politely refuses harmful requests while being helpful on benign ones. HH-RLHF is the first widely released human-preference dataset for dialogue alignment; every synthetic-preference paper since benchmarks against it.

## Key Contributions
- Released **161K human-preference dialogues** publicly under MIT license (Anthropic/hh-rlhf on HF).
- Empirically characterized the **helpfulness-harmlessness tradeoff** — the "tension curve."
- Demonstrated joint training with both subsets produces a Pareto-improved model.
- Established the RLHF pipeline template later followed by OpenAI's InstructGPT, Meta's Llama-2, DeepMind's Sparrow.
- Reference dataset for the entire 2023–25 RLHF / DPO literature.

## Key Figures/Tables to Study
- **Helpfulness-vs-harmlessness Pareto plot** — models trained on only one subset corner-case badly.
- **Agreement rates** — human-human agreement as a noise floor (~70–75%).
- **Scaling curves** — RM accuracy vs parameter count.

## Dataset structure
- **Base:** 161,000 dialogue comparisons.
- **Subsets:** `helpful-base`, `helpful-online`, `helpful-rejection-sampled`, `harmless-base`.
- **Per-instance fields:** `chosen` (preferred response dialogue) / `rejected` (dispreferred). Multi-turn.
- **License:** MIT (widely redistributable — rare for a preference dataset of this scale).

## Collection pipeline
- **Helpful data:**
  - Crowdworkers hold a conversation with a 52B Anthropic assistant.
  - At each assistant turn, two candidate responses are sampled.
  - Worker picks the more helpful; dialogue continues with the chosen one.
- **Harmless data (red team):**
  - Crowdworkers deliberately try to elicit harmful / unethical / biased output.
  - Two responses shown; worker selects the *less* harmful.
- **Multi-turn** — conversation history preserved, preference labeling per turn.
- **Crowdworker QC** with inter-annotator agreement tracking.

## Downstream usage
- Reward-model training (Bradley-Terry) → PPO.
- Later reused for DPO (Rafailov 2023) and variants.
- Re-processed into `hh-rlhf-binarized` for SFT-vs-DPO ablations.
- Benchmark for every synthetic-preference paper (UltraFeedback, West-of-N, RLCD all compare).

## Risks + gotchas
- **Crowdworker demographics** skew the "helpful" signal — documented Anthropic demographic stats.
- **Red-team coverage gaps** — harmlessness labeling does not cover all harm types.
- **Static snapshot (2022):** does not reflect later shifts in what "helpful" or "harmless" mean.
- **Length bias** — longer responses often preferred; a known confound in downstream RMs trained on HH.
- **Multi-turn modelling complication** — many tools only use single-turn slices.

## Connections
- Human-baseline counterpart to synthetic pipelines: [[ultrafeedback]]/[[ultrafeedback-construction]], [[west-of-n]], [[rlcd]].
- Direct input to [[constitutional-ai]] (Anthropic's principle-based extension).
- Benchmark target for [[rlaif-scaling]] and the entire RLAIF vs RLHF debate.
- Baseline in Tülu and Zephyr ablations; see [[tulu-3-sft-mix]].
