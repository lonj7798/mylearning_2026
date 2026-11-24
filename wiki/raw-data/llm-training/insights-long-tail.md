# Long-Tail Coverage — Cross-Source Insights

<!-- scope: rarity across facts, conversation behavior, language style, and speech
     deps: [[insights]]
     see-also: [[long-tail-papers]]
-->

- **Result (single study):** factual QA accuracy depends on relevant-document frequency in [[long-tail-knowledge]]; this does not measure conversational rarity.
- **Result (single study):** knowledge-gap and density-conditioned synthesis are proposed in [[graphgen]] and [[knowledge-density-synthesis]], with distinct evaluation settings.
- **Result (single study):** [[data-value-scaling]] studies head/tail scaling.
- **Review:** [[long-tail-knowledge-survey]] organizes definitions and interventions; it is not a new empirical result.
- **Result (single study):** [[user-oriented-dialogue-generation]] separates tasks and user simulation; [[consistentchat]] plans intent/flow skeletons; [[josh-dialogue-self-training]] searches reward-bearing trajectories.
- **Result (single study):** [[multichallenge]] evaluates failures selected through generation and human review, not natural occurrence rates.
- **Result (single study):** [[realtalk]] finds synthetic-human emotional/persona differences in a small messaging sample.
- **Result (single study):** [[persona-lexical-diversity]] finds lexical diversity sensitive to length; persona detail is not a universal diversity control.
- **Result (single study):** [[deepdialogue]] filters out long dialogues more often and maps twenty text emotion labels to eight audio categories.
- **Result (single study):** [[sdiareward]] evaluates spoken preferences; [[audio-multichallenge]] evaluates spoken multi-turn challenges.
- **Interpretation:** candidate coverage, correctness, and post-filter coverage are separate checkpoints; selection cannot preserve candidates never generated.
- **Interpretation:** specify rarity relative to a population and count unit. Difficulty, diversity, unfamiliarity, and low quality are different properties.
- **Open question:** which synthesis/mixture strategy preserves rare dynamic-skill sales behavior without hurting common conversations? None of these cards resolves it.

## Reading route

[Catalog](long-tail-papers.md) → [23a](../../courses/llm-training/ch-23a/read.md) → [23b](../../courses/llm-training/ch-23b/read.md) → [23c](../../courses/llm-training/ch-23c/read.md) → [23d](../../courses/llm-training/ch-23d/read.md).
