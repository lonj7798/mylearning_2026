<!-- scope: contrastive positive/negative prompt preferences from a single base model
     deps: [[constitutional-ai]]
     see-also: [[west-of-n]], [[ultrafeedback]], [[rlaif-scaling]]
-->

# RLCD: Reinforcement Learning from Contrastive Distillation for Language-Model Alignment
- **Core Insight:** You can manufacture preference pairs without any judge by asking the *same* base model to generate under a positive prompt ("be extremely helpful") and a negative prompt ("be unhelpful and rude") — the contrast in prompts forces the two outputs apart in the desired direction, giving cleaner labels than Best-of-N sampling under a single prompt.
- **Guideline:** To align a base model to a principle expressed in natural language (helpful/harmless/any style), write a positive and a negative prompt, sample one completion per side, treat the positive as chosen and negative as rejected; train an RM on the pairs, then RL fine-tune.
- **Authors:** Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, Yuandong Tian (Meta AI / Berkeley / UCLA)
- **Year:** 2023 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2307.12950
- **Relevant topics:** contrastive preferences, RLAIF, synthetic preferences, principle-based alignment

## Abstract
RLCD generates preference pairs by sampling the *same* base LM under two contrasting prompts — one designed to elicit the target principle (e.g., helpfulness), one designed to violate it (e.g., be unhelpful). The contrast between prompts forces the two completions apart in the desired dimension, cleanly labelling the positive as chosen and the negative as rejected without any external judge. An RM is trained on the resulting pairs and used to RL-fine-tune the base LM. Empirically, RLCD beats both RLAIF (LLM-as-judge prefs) and context-distillation baselines across harmlessness, helpfulness, and story-outline generation using 7B and 30B Llama/Llama-2 backbones.

## Key Contributions
- Introduced **contrastive prompting** as a preference-data construction primitive.
- Demonstrated measurably cleaner preference labels than single-prompt Best-of-N.
- Showed RLCD beats RLAIF at matched compute on three principle-alignment tasks.
- Public code + Meta-authored reference implementation.

## Key Figures/Tables to Study
- **Win-rate table** — RLCD vs RLAIF vs Context Distillation on harmlessness/helpfulness/outline.
- **Preference-label noise analysis** — RLCD pairs are more separable than Best-of-N pairs.
- **Scaling curve** from 7B to 30B.

## Synthesis pipeline (REQUIRED — be concrete)
- **Seed input:** a principle expressed in natural language (e.g., "Be as helpful as possible.")
- **Generation step(s):**
  - Write positive prompt `p+` that elicits the principle (e.g., `"<helpful system prompt>\nHuman: <query>\nAssistant:"`).
  - Write negative prompt `p-` that elicits the opposite (e.g., `"<unhelpful system prompt>\n..."`).
  - Sample one completion `y+ ~ π(· | p+)` and one `y- ~ π(· | p-)` from the *same* base LM.
  - Strip the system/persona prefix from both; pair `(user_query, y+, y-)` as chosen/rejected.
- **Filtering/rescoring:** optional dedup / length-match; no external judge required.
- **RM training:** Bradley-Terry on the synthetic pairs.
- **RL step:** PPO against the RM with KL penalty to SFT reference, standard RLHF recipe.
- **Output shape:** one preference pair per prompt; scales with prompt count.
- **Teacher model(s):** only the base LM (same one used for positive and negative generation). No GPT-4 judge, no human rater.
- **Cost estimate:** 2× sampling cost per prompt; no API dependency.

## Quality / diversity evaluation
- Harmlessness: RLCD > RLAIF and > Context-Distillation on GPT-4-judged harmlessness.
- Helpfulness: similarly.
- Story-outline generation: RLCD wins on structural coherence.
- Ablation: random-prompt negatives (no explicit contrast) underperform — the *contrast* is load-bearing, not just any negative.

## Risks + gotchas
- **Prompt engineering sensitivity:** the quality of the positive/negative prompt pair dominates label quality.
- **Principle narrowness:** alignment signal is only as broad as the principle articulated in the prompt pair.
- **Mode collapse in negatives:** if the negative prompt just yields refusals, labels become uninformative.
- **Base-model leakage:** negatively-prompted outputs can still inadvertently produce helpful content; require filtering.
- **Does not replace multi-aspect prefs** — for helpfulness + honesty + safety jointly, you typically layer multiple contrastive pairs or combine with [[ultrafeedback]]-style multi-axis ratings.

## Connections
- Sits between [[constitutional-ai]] (principles → critiques) and [[west-of-n]] (best/worst of N from same prompt).
- Complementary to [[ultrafeedback]]/[[ultrafeedback-construction]]: RLCD needs no external judge; UltraFeedback needs a stronger judge for labels.
- A 2024 synthetic-preference-for-DPO variant (replace PPO with DPO) is widely used in open alignment stacks.
- Directly informed later "contrastive prompt engineering" lines like Evolutionary Contrastive Distillation.
