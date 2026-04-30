<!-- scope: NVIDIA Nemotron-4 synthetic alignment pipeline for SFT, preference, and RM data
     deps: [[self-instruct]]
     see-also: [[ultrafeedback-construction]], [[magpie]], [[persona-hub]]
-->

# Nemotron-4 340B Technical Report
- **Core Insight:** NVIDIA compresses alignment into a strong reward model plus a synthetic prompt/response/pair pipeline; over 98% of post-training data is synthetic, and the same pipeline feeds SFT, DPO, and RPO.
- **Guideline:** Use a category-seeded synthetic pipeline, quality-filter with a reward model, keep a small human anchor set for the reward model and hard cases, then iterate the generator checkpoint.
- **Authors:** NVIDIA
- **Year:** 2024
- **URL:** https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf
- **Relevant topics:** synthetic alignment, RM-as-judge, weak-to-strong iteration, supervised fine-tuning, preference tuning, reward modeling

## Abstract
Nemotron-4-340B-{Base, Instruct, Reward} is NVIDIA's open-weight frontier release. Its distinctive contribution is the synthetic data generation pipeline used for alignment: the paper says over 98% of the training data for alignment is synthetic, while only about 20K human-annotated examples are used overall, split between SFT and HelpSteer2 reward-model data. The released pipeline covers synthetic prompt generation, response and dialogue generation, quality filtering, and preference ranking, and is explicitly designed to support both supervised fine-tuning and preference fine-tuning.

## Key Contributions
- Open-sourced the synthetic alignment pipeline plus generation prompts and the human preference dataset.
- Uses a reward model both as a filter and as a judge for preference ranking when ground truth is unavailable.
- Implements staged SFT: first a code-focused SFT stage, then a broader general SFT stage.
- Implements preference fine-tuning with DPO followed by RPO, with the reward model used to select higher-quality chosen responses.
- Shows that a small human anchor set can support a much larger synthetic alignment corpus.

## Key Figures/Tables to Study
- The synthetic alignment section around prompt generation, quality filtering, and preference ranking.
- The staged SFT section for code SFT and general SFT.
- The preference fine-tuning section for DPO and RPO.
- The evaluation tables showing the aligned model across chat, math, code, instruction-following, and topic-following benchmarks.

## Synthesis Pipeline (REQUIRED - be concrete)
- **Seed input:** task families for coding, general question answering, topic-following, document-based reasoning, function calling, and incapable tasks that need explicit refusal behavior.
- **Prompt generation:** the pipeline synthesizes prompts by task family, then uses the current model to generate responses or multi-turn dialogues. For impossible tasks, it uses few-shot prompting with human-written examples to elicit explicit rejections.
- **Diversity control:** the paper separates code SFT from general SFT, keeps topic-following data intentionally noisy with distractor turns, and adds a small amount of alignment-style QA in continued pretraining to steer low-accuracy areas.
- **Response generation:** the code alignment stage uses Genetic Instruct, which combines self-instruction and WizardCoder-style mutations plus an LLM-based fitness function to grow a population from a limited number of seeds.
- **Filtering/rescoring:** a reward model scores responses for quality; when ground truth is missing, Nemotron-4-340B-Reward selects high-quality chosen responses. The preference pipeline prefers RM-based ranking over raw model self-selection.
- **Output shape:** approximately 800K code SFT samples, 200K general SFT samples, 160K DPO preference examples, and 300K RPO preference examples, all within a pipeline that is more than 98% synthetic overall.
- **Teacher model(s):** the paper iterates through Nemotron checkpoints for alignment, and uses Nemotron-4-340B-Reward as the scorer/judge during filtering and ranking.
- **Cost estimate:** not disclosed per sample; the paper emphasizes that the small human anchor set is enough to sustain the pipeline.

## Quality / Diversity Evaluation
- Nemotron-4-340B-Instruct is competitive with other open-access aligned models across chat, math, code, instruction-following, and topic-following benchmarks.
- The reward model reaches top RewardBench performance at the time of publication.
- The pipeline is meant to preserve behavior diversity across task families while still using synthetic data at very high scale.

## Risks + Gotchas
- Reward-model errors compound when the same scorer is reused across iterations.
- DPO alone can overfit to reward gaps; the paper adds SFT loss and then RPO to reduce that effect.
- The synthetic majority is a strength for scale, but it also makes quality filtering and judge calibration critical.
- The model and license are open access, but the paper explicitly discourages toxic or harmful use.

## Connections
- Pairs well with [[ultrafeedback-construction]] as an RM-as-judge reference.
- Complements [[magpie]] and [[persona-hub]] as seed/light-seed synthetic pipelines.
- The staged SFT + iterative preference loop is the practical template later reused by other open alignment stacks.
