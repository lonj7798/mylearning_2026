<!-- scope: AllenAI / Ai2 training worldview across Dolma, Tulu, and OLMo
     deps: [[dolma]], [[tulu-3]], [[olmo-3]]
     see-also: [[yejin-choi-group]], [[qwen-3]], [[deepseek-v3]]
-->

# AllenAI / Ai2
- **Core Insight:** AllenAI's distinctive contribution is not one model or one algorithm; it is an open-science training stack where data curation, evaluation, post-training recipes, and model-flow transparency are all published as reusable research objects.
- **Guideline:** Track AllenAI when you want practical but research-grade insight into how modern open models are actually built; their work is unusually good at exposing filters, ablations, intermediate checkpoints, and stage-by-stage recipes.
- **Authors:** Synthesized from AllenAI primary artifacts including Dolma, Tulu 3, OLMo 2, and OLMo 3
- **Year:** 2024-2026
- **URL:** https://allenai.org/blog/olmo3
- **Relevant topics:** open science, data curation, RLVR, model flow, evaluation frameworks, post-training infrastructure

## Abstract
AllenAI's recent language-model work forms one of the strongest public research lines for studying training. Dolma makes large-scale corpus construction inspectable. Tulu 3 makes modern post-training inspectable via SFT, DPO, and RLVR. OLMo 2 exposes staged pretraining interventions and evaluation-guided development. OLMo 3 then pushes this further by releasing an entire model flow rather than a final checkpoint. Together these artifacts show a coherent worldview: publish the data, the tooling, the intermediate stages, and the eval framework so the community can study capability formation rather than only consume model outputs.

## Key Contributions
- **Dolma:** fully documented large-scale corpus curation, including filtering and deduplication design.
- **Tulu 3:** one of the clearest open disclosures of a modern **SFT -> DPO -> RLVR** post-training recipe.
- **OLMo 2:** staged pretraining, late-pretraining interventions, stability work, and evaluation with OLMES.
- **OLMo 3:** full model-flow release with pretraining, mid-training, long-context, and multiple post-training branches.
- **Tooling:** `olmo-core`, `open-instruct`, `OLMES`, `OlmoTrace`, decontamination and dedup tools.
- **Researchers to follow:** Nathan Lambert, Luca Soldaini, Noah A. Smith, Hannaneh Hajishirzi, Kyle Lo, and related Ai2 collaborators.

## Key Figures/Tables to Study
- **[[dolma]] pipeline diagram:** best entry point for data-curriculum thinking.
- **[[tulu-3]] three-stage pipeline figure:** best entry point for open post-training.
- **[[olmo-3]] model-flow diagram:** best entry point for stage-by-stage capability shaping.
- **OLMo 2 and OLMo 3 benchmark/eval sections:** show how OLMES is used to guide development rather than just report final numbers.

## Technical Details

### Worldview 1: data is a first-class artifact
- AllenAI does not treat training data as an opaque input.
- Dolma and later Dolma 3 publish the corpus design, filtering decisions, and source mixes.
- OLMo 3 extends this into a full **data curriculum**: broad pretraining data, mid-training data, long-context data, and post-training data are all separate named artifacts.

### Worldview 2: post-training is modular
- The lab repeatedly uses the modular stack **SFT -> preference tuning -> RLVR**.
- Tulu 3 established this openly.
- OLMo 3 reuses the same structure but exposes different branches for **Think**, **Instruct**, and **RL Zero** use cases.

### Worldview 3: release paths, not just endpoints
- OLMo 3's strongest idea is that the public object should be the **trajectory** of model development.
- This makes it possible to ask better research questions:
  - What did mid-training add that pretraining missed?
  - What did DPO add beyond SFT?
  - What did RLVR add beyond DPO?
  - Which data mix produced a specific capability or failure mode?

### Worldview 4: centralized evaluation steers distributed experiments
- AllenAI uses **OLMES** and held-out evaluation slices to make model development less ad hoc.
- This is similar in spirit to Yejin Choi's emphasis on scientific understanding over raw paper volume: better recipes require better diagnostic frameworks, not only larger runs.

### Why this lab matters for your notes
- If Yejin Choi gives you the "alternative recipe / reasoning science" mindset, AllenAI gives you the **open systems implementation** of that mindset.
- This is one of the best public places to learn how data, SFT, DPO, RLVR, and evaluation fit together in a real training program.

## Connections
- [[yejin-choi-group]] is the closest adjacent research arc on synthetic data and reasoning recipes.
- [[dolma]], [[tulu-3]], and [[olmo-3]] are the core AllenAI sources.
- [[qwen-3]] and [[deepseek-v3]] are important comparison points, but AllenAI is much more transparent about intermediate stages and datasets.
- [[deepseek-r1]] is useful as the RL-heavy contrast to AllenAI's more modular, openly staged approach.
