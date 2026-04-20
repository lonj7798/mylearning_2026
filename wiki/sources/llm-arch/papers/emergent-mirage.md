# Are Emergent Abilities of Large Language Models a Mirage?
- **Authors:** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2304.15004
- **Core Insight:** "Emergence" may be a measurement artifact of non-linear metrics, not a property of the model.
- **Guideline:** When evaluating model capabilities across scales, use linear and continuous metrics (like token-level accuracy or Brier score) rather than non-linear or discontinuous ones (like exact-match). The choice of metric can create or eliminate the appearance of sudden emergence.
- **Relevant chapters:** Scaling laws, Evaluation methodology, Emergent behavior, Benchmarking

## Abstract
Recent work claims that large language models display emergent abilities, abilities not present in smaller-scale models that are present in larger-scale models. What makes emergent abilities intriguing is two-fold: their sharpness, transitioning seemingly instantaneously from not present to present, and their unpredictability, appearing at seemingly unforeseeable model scales. Here, we present an alternative explanation for emergent abilities: that for a particular task and model family, when analyzing fixed model outputs, emergent abilities appear due to the researcher's choice of metric rather than due to fundamental changes in model behavior with scale. Specifically, nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous predictable changes in model performance. We present our alternative explanation in a simple mathematical model, then test it in three complementary ways: we (1) make, test and confirm three predictions on the effect of metric choice using the InstructGPT/GPT-3 family on tasks with claimed emergent abilities; (2) make, test and confirm two predictions about metric choices in a meta-analysis of emergent abilities on BIG-Bench; and (3) show to choose metrics to produce never-before-seen seemingly emergent abilities in multiple vision tasks across diverse deep networks. Via all three analyses, we provide evidence that alleged emergent abilities evaporate with different metrics or with better statistics, and may not be a fundamental property of scaling AI models.

## Key Contributions
- Proposed that emergent abilities are artifacts of metric choice rather than fundamental model properties, providing a mathematical model for why nonlinear metrics produce sharp transitions
- Demonstrated empirically that switching from nonlinear metrics (exact-match accuracy) to linear metrics (token-level accuracy, Brier score) makes emergence disappear in favor of smooth, predictable improvement
- Conducted a meta-analysis of BIG-Bench tasks showing the pattern holds broadly across many tasks with claimed emergence
- Showed the reverse is also true: by choosing discontinuous metrics, one can manufacture apparent emergence in vision tasks where none was previously reported
- Forced the field to reconsider a widely accepted narrative about scaling, improving scientific rigor in AI evaluation

## Why This Paper Matters
This paper is a masterclass in scientific skepticism applied to AI. By showing that "emergence" can be created or destroyed by metric choice alone, it challenged one of the most influential claims in the scaling era. The practical impact is significant: if capabilities scale smoothly rather than appearing suddenly, then extrapolation from smaller models becomes feasible again, and the case for "just scale up" becomes more nuanced. Together with Wei et al. (2022), these two papers form an essential pair for understanding what we actually know about scaling.
