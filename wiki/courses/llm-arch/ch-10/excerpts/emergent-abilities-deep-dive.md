<!-- scope: emergent abilities debate — Wei et al. vs. Schaeffer et al., metric artifacts, implications for scaling
     parent: [[ch-10]]
-->

# The Emergent Abilities Debate: Phase Transitions or Measurement Artifacts?

## The Original Claim

Wei et al. ([[emergent-abilities|paper]]) (2022) defined emergent abilities as capabilities that are "not present in smaller models but present in larger models" and "cannot be predicted by extrapolating from smaller scales." The paper catalogued dozens of examples across model families and benchmarks, including:

- **Multi-step arithmetic:** Models below ~60B parameters perform at chance on 3-digit addition; above that threshold, accuracy jumps sharply.
- **Chain-of-thought reasoning:** Prompting with "let's think step by step" yields no benefit below ~100B parameters, then suddenly enables multi-step problem solving.
- **Word unscrambling, Swahili translation, modular arithmetic** on BIG-Bench: all show near-zero performance below a critical scale, then rapid improvement above it.

The key feature distinguishing emergence from ordinary scaling: the transition is *sharp*. Performance does not gradually improve from 10% to 50% to 90% as scale increases. Instead, it jumps from near-zero to functional accuracy over a narrow range of scale. This sharpness is what makes emergence seem like a qualitative phase transition rather than quantitative improvement.

## The Mathematical Argument Against

Schaeffer, Miranda, and Koyejo ([[emergent-mirage|paper]]) (2023) proposed a simple mathematical model explaining the appearance of emergence. The argument:

1. Assume the model's per-token probability of generating the correct token improves *smoothly* with scale: $p(\text{correct token}) = \sigma(a \cdot \log N + b)$ for some sigmoid $\sigma$.

2. For a task evaluated with **exact-match accuracy** on an $L$-token answer, the measured accuracy is:

$$\text{EM}(N) = p(N)^L$$

3. Because the exponentiation $p^L$ is a *nonlinear* transformation of the underlying smooth improvement, it creates a sharp transition:
   - When $p = 0.8$ and $L = 10$: $\text{EM} = 0.8^{10} = 0.107$ (appears to fail)
   - When $p = 0.95$ and $L = 10$: $\text{EM} = 0.95^{10} = 0.599$ (appears to succeed)
   - The underlying improvement is smooth (0.8 to 0.95), but the metric shows a 6x jump

4. More generally, any nonlinear or discontinuous metric (accuracy with a hard threshold, exact string match, binary pass/fail) can produce apparent sharp transitions from smooth underlying improvement.

## Empirical Confirmation

Schaeffer et al. confirmed this across three analyses:

### Analysis 1: Re-evaluating GPT-3 / InstructGPT
For tasks where Wei et al. claimed emergence in the GPT-3 family:
- **Exact-match accuracy:** Shows sharp emergence (replicating the original claim)
- **Per-token accuracy:** Shows smooth, predictable improvement at all scales
- **Brier score (calibrated probability):** Also shows smooth improvement

The same model outputs, measured with different metrics, show or hide "emergence."

### Analysis 2: BIG-Bench Meta-Analysis
Across the BIG-Bench benchmark suite:
- Tasks measured with **nonlinear metrics** (multiple-choice accuracy where all tokens must match) showed emergence
- Tasks measured with **linear metrics** (individual token accuracy) did not show emergence
- The correlation between metric nonlinearity and apparent emergence was near-perfect

### Analysis 3: Manufacturing Emergence in Vision
To prove the reverse direction, the authors chose computer vision tasks (CIFAR-100, ImageNet) where no one had claimed emergence, then showed that by choosing sufficiently nonlinear metrics, they could manufacture the appearance of sudden emergence. This demonstrates that emergence is a property of the metric, not the model or the task.

## What Remains Unresolved

The metric-artifact argument is compelling but does not fully close the debate:

### 1. Circuit Complexity Thresholds
Some capabilities may genuinely require a minimum number of parameters to implement the necessary computation circuits. A 4-layer Transformer with 10M parameters cannot implement multi-hop reasoning regardless of how much data it sees, because the circuit depth is insufficient. This is a true threshold, not a metric artifact -- but it is also not what Wei et al. measured.

### 2. In-Context Learning
The ability to learn from in-context examples (few-shot prompting) appears to require a minimum model scale in a way that is not obviously metric-dependent. Small models can be prompted with examples and still fail to pattern-match, while large models succeed. Whether this is "emergence" or gradual improvement masked by evaluation methodology remains debated.

### 3. Compositional Generalization
Tasks requiring compositional generalization (combining known sub-skills in novel ways) may exhibit genuine threshold behavior: the model must have learned each sub-skill before it can compose them, and sub-skill acquisition itself may have thresholds.

## Implications for Scaling Decisions

Regardless of the philosophical resolution, the practical implications are clear:

1. **Do not use nonlinear metrics to justify scaling decisions.** If your target task is evaluated with exact-match accuracy and shows apparent emergence, switch to a continuous metric before concluding that "we need a bigger model."

2. **Smooth underlying improvement means extrapolation works.** If capabilities scale smoothly on continuous metrics, you can predict performance at larger scale from smaller experiments. This makes scaling laws useful for capability prediction, not just loss prediction.

3. **But capability thresholds may still exist for genuinely novel skills.** The absence of metric artifacts does not mean all capabilities scale smoothly. Some genuinely require circuit complexity that only appears at scale. The burden of proof, however, is now on the claimant.

4. **Evaluate at multiple granularities.** For any critical capability, report both coarse-grained (exact-match) and fine-grained (per-token, per-step) metrics across multiple model scales. This is the minimum standard for making claims about emergence.

## References

- [[emergent-abilities|Wei et al., "Emergent Abilities of Large Language Models" (2022) (paper)]]
- [[emergent-mirage|Schaeffer et al., "Are Emergent Abilities of Large Language Models a Mirage?" (2023) (paper)]]
