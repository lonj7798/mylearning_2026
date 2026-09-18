<!-- scope: index of cross-source insights for the llm-training raw library
     deps: [[README]], [[COLLECTION-PLAN]]
     see-also: [[insights-generality]], [[insights-preference-and-rl]]
-->

# LLM Training Techniques — Insights Index

This page indexes the cross-source insight pages for the raw library. After the 2026-09 revision the library holds
411 source cards and 66 companion recipe ledgers; 373 cards carry a `## Verification` section and 372 of those
record a 2026-09 check date. Each theme page below states one factual claim per bullet with the cards that support it.
Claim status labels follow §3 of the authoring standard: Result (single study), Replicated, Interpretation, Open question.

| Theme | Page |
|---|---|
| Generality measurement and benchmark overfitting | [[insights-generality]] |
| Pretraining scaling and data | [[insights-pretraining-scaling]] |
| Mid-training, continued pretraining, and context extension | [[insights-midtraining-context-extension]] |
| Synthetic data | [[insights-synthetic-data]] |
| Long-context and long-conversation synthesis | [[insights-long-context-synthesis]] |
| Agentic data and RL | [[insights-agentic]] |
| SFT design, mixtures, forgetting, and merging | [[insights-sft-design]] |
| Distillation practice | [[insights-distillation]] |
| Preference optimization and RL | [[insights-preference-and-rl]] |
| Negative samples and negative feedback | [[insights-negative-feedback]] |
| Evaluation and judges | [[insights-evaluation-and-judges]] |
| Infrastructure | [[insights-infrastructure]] |

## Open gaps

Each gap below was confirmed by listing the library directories; the named slugs have no file under `wiki/raw-data/llm-training/`.
The full list of unfilled candidate slugs is in the gap log of [[COLLECTION-PLAN]].

- Pretraining scaling has no primary scaling-law card: `kaplan-scaling-laws`, `chinchilla-compute-optimal`, `dclm`, `nemotron-cc`, `smollm2` and `minicpm` are all absent, so [[insights-pretraining-scaling]] cites the laws only through [[lr-schedules]], [[resolving-scaling-discrepancies]] and [[beyond-chinchilla-inference-scaling]].
- The instruction-tuning lineage before 2023 is missing: `flan`, `flan-collection`, `t0-multitask-prompted-training` and `super-natural-instructions` have no cards, so multi-task generality claims rest on [[tulu-1-how-far-can-camels-go]] alone.
- Forgetting has no dedicated card. `catastrophic-forgetting-continual-finetuning`, `continual-pretraining-rewarm-replay`, `lora-learns-less-forgets-less`, `wise-ft`, `sft-memorizes-rl-generalizes` and `sft-data-composition-dmt` are absent; [[insights-sft-design]] measures forgetting only through side effects reported in other papers.
- Negative feedback lacks its core references: `nft-negative-aware-finetuning`, `unlikelihood-training`, `critique-fine-tuning`, `learning-dynamics-llm-finetuning`, `dpo-positive`, `negative-sample-reinforcement` and `rl-on-incorrect-synthetic-data` are all missing, so no card in the library measures what share of a gain negatives contribute.
- RL algorithm coverage stops before `dapo`, `scalerl` and `guru-cross-domain-rl`, and the contamination-and-leaderboard evidence (`training-on-the-test-task`, `leaderboard-illusion`, `rephrased-samples-contamination`, `reasoning-or-memorization-rl-contamination`, `length-controlled-alpacaeval`, `ifeval`) is absent from [[insights-generality]] and [[insights-evaluation-and-judges]].
- Long-context and agentic evidence is missing `lost-in-multi-turn`, `instruction-hierarchy`, `ultralong-128k-to-4m`, `artificial-needles-real-haystacks`, `parrot-multi-turn`, `prefeval`, `tau-bench`, `search-r1`, `swe-smith`, `agentfounder`, `agentscaler` and `agent-data-protocol`.
- Recent model reports and open recipes not yet carded: `gemma-3`, `gemini-1-5`, `hermes-4`, `llama-nemotron`, `openmathreasoning`, `opencodereasoning`, `light-r1`, `klear-reasoner`, `smol-training-playbook`, `smollm3-training-configs` and `olmo-core-olmo3-configs`.
