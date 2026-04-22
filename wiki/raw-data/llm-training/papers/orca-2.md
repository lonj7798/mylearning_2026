<!-- scope: Orca-2 reasoning strategies and cautious small-model alignment
     deps: [[orca]]
     see-also: [[limo]], [[s1]]
-->

# Orca 2: Teaching Small Language Models How to Reason
- **Core Insight:** Small models benefit from being taught multiple reasoning strategies, including when to abstain, simplify, or decompose, rather than one default “long explanation” style.
- **Guideline:** Distill reasoning strategies, not just chains of thought; train smaller models on varied solution modes including cautious or selective answering.
- **Authors:** Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agrawal, et al.
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2311.11045
- **Relevant topics:** reasoning strategies, cautious reasoning, small-model distillation

## Abstract
Orca 2 extends the Orca line by emphasizing diverse reasoning strategies for smaller models. The paper argues that students should learn when to reason deeply, when to answer briefly, and when to acknowledge uncertainty, improving reasoning quality without requiring frontier-scale model size.

## Key Contributions
- Shifted focus from trace richness alone to strategy richness.
- Emphasized cautious reasoning and selective answering.
- Showed that small models can become more reliable reasoners with better post-training signals.

## Technical Details
- Builds on explanation-trace distillation but expands the space of target behaviors.
- Uses synthetic teacher supervision designed to teach different reasoning patterns.
- Practical contribution is a better-targeted student curriculum rather than a radically new algorithm.

## Connections
- Direct continuation of [[orca]].
- Conceptually aligned with later curated-small-set reasoning recipes like [[limo]] and [[s1]].

