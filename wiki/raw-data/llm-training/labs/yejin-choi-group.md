<!-- scope: Yejin Choi research arc on alternative training recipes, synthetic data, and reasoning
     deps: [[self-instruct]], [[star]], [[quiet-star]]
     see-also: [[lima]], [[deepseek-r1]], [[tulu-3]]
-->

# Yejin Choi Group / Research Arc
- **Core Insight:** This research arc keeps returning to one bet: stronger reasoning does not come only from brute-force scale; it can be unlocked by better training recipes, synthetic supervision, intermediate reasoning traces, and targeted work on smaller models.
- **Guideline:** Study this lineage when you want ideas for *how to train* rather than just *how to benchmark*: it is one of the clearest public threads connecting synthetic instruction data, rationale bootstrapping, latent-thought pretraining, and pluralistic alignment.
- **Authors:** Synthesized from Yejin Choi's Stanford profile, her NeurIPS 2025 invited talk abstract, and key papers in the research lineage
- **Year:** 2022-2025
- **URL:** https://profiles.stanford.edu/yejin-choi
- **Relevant topics:** alternative training recipes, synthetic data, self-improvement, reasoning, small language models, pluralistic alignment

## Abstract
Yejin Choi's current public research profile centers on the limits and capabilities of large language models, alternative training recipes, reasoning and knowledge discovery, pluralistic alignment, and AI safety. Across the associated paper lineage, the recurring pattern is to treat model outputs not just as endpoints but as training material: self-generated instructions in Self-Instruct, self-bootstrapped rationales in STaR, and token-level latent thoughts in Quiet-STaR. Her NeurIPS 2025 invited talk frames the field-level lesson clearly: brute-force scaling is powerful, but present systems still show "jagged intelligence," and reinforcement learning helps only in some reasoning regimes.

## Key Contributions
- **Self-Instruct:** showed that instruction tuning data can be bootstrapped from model generations instead of requiring a large private annotation pipeline.
- **STaR:** showed that correct reasoning traces can be bootstrapped from answer supervision by iteratively harvesting successful rationales.
- **Quiet-STaR:** pushed the same idea into continued pretraining by teaching models to think before speaking on arbitrary text.
- **Research framing:** public statements now emphasize the limits of brute-force scaling, the need for better scientific understanding of reasoning, and recipe improvements for smaller models.
- **Alignment breadth:** the profile explicitly includes pluralistic alignment and moral values, signaling that training recipes should optimize for more than single-metric accuracy.

## Key Figures/Tables to Study
- **[[self-instruct]] Figure 1:** synthetic instruction bootstrapping pipeline.
- **[[star]] method figure:** the generate-verify-rationalize loop.
- **[[quiet-star]] Figure 1:** latent thought training on ordinary text.
- **NeurIPS 2025 invited talk abstract/video page:** concise statement of where RL helps, where it does not, and why "more is more" is incomplete.

## Technical Details

### Recurring pattern 1: bootstrap from model outputs
- Instead of treating model generations as disposable samples, this line of work treats them as **raw supervision candidates**.
- The loop is usually:
  1. generate a candidate supervision object
  2. filter or verify it
  3. finetune on the retained subset
  4. repeat with the improved model

### Recurring pattern 2: supervise intermediate cognition
- The training target is not only the final answer.
- Self-Instruct targets **instruction-response structure**.
- STaR targets **rationale traces**.
- Quiet-STaR targets **latent internal thoughts** attached to many token positions.

### Recurring pattern 3: improve smaller models with better recipes
- The 2025 NeurIPS talk frames reasoning progress as partly a **recipe problem**, not only a scale problem.
- This is directly useful for open-model work, where compute constraints force careful use of synthetic data, iterative self-training, and filtered reasoning traces.

### Why this matters for your training notes
- This research arc is a good antidote to a narrow "RL is everything" view.
- It shows that data construction, trace filtering, latent-thought objectives, and verifier design can all move reasoning before or alongside RL.

## Connections
- [[self-instruct]], [[star]], and [[quiet-star]] are the core primary sources in this arc.
- [[lima]] is a useful foil: it argues for tiny high-quality SFT, whereas the Choi lineage often emphasizes synthetic expansion plus filtering.
- [[deepseek-r1]] represents the more RL-heavy reasoning path; reading it against this lab summary helps separate what must come from RL versus from data/recipe design.
- [[tulu-3]] is the open-lab systems counterpart: it turns many of these ideas into a broader post-training pipeline.
