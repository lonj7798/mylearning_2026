# Common Pitfalls in Architecture Experiments

<!-- scope: extended catalog of failure modes in architecture research with concrete examples
     see-also: [[ch-28]], [[ch-24]]
-->

## Overview

Architecture experiments fail in predictable ways. This excerpt catalogs the most common pitfalls, organized from most to least frequently encountered. Each pitfall includes a concrete example (real or composite), the mechanism by which it misleads, and the defense.

---

## Pitfall 1: The Implementation-Architecture Conflation

**What happens:** You change the attention mechanism *and* write a custom CUDA kernel. The model is 2x faster. You claim the architecture is better. But the speedup comes entirely from the kernel optimization — the architectural change is irrelevant or even slightly harmful.

**Real example:** Several linear attention papers report wall-clock speedups that include custom Triton/CUDA kernels, compared against a baseline using standard PyTorch attention (without Flash Attention). When both methods use equally optimized kernels, the gap often vanishes or reverses.

**The mechanism:** Custom kernels for the experimental condition vs. generic kernels for the baseline creates an unfair comparison. The "speedup" measures implementation effort, not architectural merit.

**Defense:** 
- Compare against the best available baseline implementation (e.g., Flash Attention, not naive PyTorch attention)
- If you write custom kernels for your method, report results *both* with and without custom kernels
- Separately report algorithmic complexity (FLOPs, memory) and wall-clock time, so readers can distinguish algorithmic from engineering contributions

**Flash Attention gets this right:** The paper is explicit that the contribution is an IO-aware *implementation* of exact attention, not a new attention mechanism. This clarity is why the paper is so widely adopted — practitioners know exactly what they are getting.

---

## Pitfall 2: Benchmark Overfitting

**What happens:** You try 15 architectural modifications. Three of them improve MMLU by 1+ points. You publish the best one. But the improvement on MMLU does not transfer to other tasks — it reflects an inductive bias that helps with multiple-choice formatting, not a genuine quality improvement.

**The mechanism:** Selection bias. If you evaluate enough modifications on the same benchmark, some will appear to help by chance. The probability of at least one false positive grows with the number of modifications tested.

**Compounding factor:** Many popular benchmarks (MMLU, HellaSwag, ARC) use multiple-choice formatting. Modifications that improve calibration or format-following will score higher on all of them, creating a misleading impression of broad improvement.

**Defense:**
- Evaluate on a diverse suite that includes open-ended generation, code, math, and instruction following — not just multiple-choice knowledge benchmarks
- Report results on *all* benchmarks, not just the ones where you improved
- If you tested N modifications, state N in the paper. The reader needs to know the size of your search space to assess the significance of the best result
- Hold out a "surprise" benchmark that you never look at during development. If your modification helps on the surprise benchmark too, the signal is real

**OLMo 2 gets this right:** The OLMES framework uses 20 benchmarks across four categories (knowledge recall, commonsense, general reasoning, math). Results are reported on all 20, not cherry-picked.

---

## Pitfall 3: The Hyperparameter Trap

**What happens:** Your new attention mechanism has a latent dimension $d_c$. You sweep $d_c \in \{128, 256, 512, 768, 1024\}$ and pick the best. Your baseline uses default hyperparameters without sweeping. Your method "wins" because it was tuned and the baseline was not.

**The mechanism:** Asymmetric hyperparameter search. Every new hyperparameter is an additional degree of freedom. If you tune your method more aggressively than the baseline, the comparison is unfair.

**Defense:**
- Give the baseline the same hyperparameter tuning budget (in GPU hours) as your method
- Report sensitivity: how much does your result change across the sweep? If $d_c = 512$ gives loss 2.81 but $d_c = 256$ gives loss 2.85, the method is robust. If $d_c = 256$ gives loss 2.95, the method is fragile and the "improvement" depends on lucky hyperparameter selection
- When possible, design modifications that do not introduce new hyperparameters. Flash Attention introduces no architecture-level hyperparameters (block size is determined by hardware SRAM capacity, not tuned)

---

## Pitfall 4: Scale-Dependent Effects Presented as General

**What happens:** Your modification improves validation loss at 350M parameters. You extrapolate to 70B and claim it will help there too. But the effect reverses at scale: the modification helped at small scale by adding inductive bias, but at large scale the model has enough capacity to learn the same patterns without the bias, and the bias constrains the representation.

**Real example:** Many "efficient attention" methods (sparse patterns, local windows, low-rank approximations) show strong improvements at small scale where the quadratic cost is proportionally larger, but the gap narrows or disappears at large scale where (a) Flash Attention eliminates the IO bottleneck and (b) the model benefits from full attention over all tokens.

**The mechanism:** Inductive biases help when data or capacity is limited. At sufficient scale, learned representations outperform hand-designed biases. Modifications that add bias show diminishing returns.

**Defense:**
- Scaling plots with 3-5 data points. Never publish from a single scale
- Check whether the curves diverge (modification becomes MORE valuable) or converge (modification becomes LESS valuable). Both are publishable — but the interpretation is completely different
- For MoE routing, normalization, and attention pattern modifications: test at both shallow (12 layers) and deep (64+ layers) models, because depth-dependent effects are common

---

## Pitfall 5: Training Stability Improvements Disguised as Quality Improvements

**What happens:** Your modification (e.g., QK-norm, Z-loss) prevents loss spikes during training. With the modification, training proceeds smoothly for 100B tokens. Without it, training spikes at 60B tokens and the team manually reduces the learning rate, losing quality. The modification "improves" quality, but the real contribution is stability.

**The mechanism:** Training instability introduces confounds. If the baseline run was manually adjusted mid-training (LR reduction, checkpoint rollback, data filtering), it is no longer a clean baseline. The quality difference reflects the instability intervention, not the architectural modification.

**Defense:**
- Separately report training stability (number of spikes, manual interventions needed) and final quality (loss, benchmarks)
- If the baseline required manual intervention, report that fact and acknowledge it as a confound
- Run the baseline with a conservative learning rate that avoids spikes, even if it reduces final quality. This gives a clean comparison at the cost of underestimating the baseline

**OLMo 2 gets this right:** QK-norm is described as a *training stability* improvement, not a quality improvement. The paper reports that final quality is "negligibly" affected but training stability is dramatically improved. This honest framing makes the contribution clearer and more useful.

---

## Pitfall 6: Ignoring Interaction Effects

**What happens:** You test modification A (RMSNorm) and modification B (RoPE) independently against a baseline. Both help. You combine them and assume the improvements are additive. But A and B interact: RMSNorm changes the distribution of residual stream activations, which changes how RoPE's rotations affect attention patterns. The combined effect is less than (or greater than) the sum of individual effects.

**The mechanism:** Non-additive interaction between architectural components. The Transformer is a deeply interconnected system — changes to normalization affect attention dynamics, which affect FFN inputs, which affect gradient flow.

**Defense:**
- Cumulative ablations: test A, then A+B, then A+B+C. This reveals pairwise interactions
- If resources allow, test a factorial design: baseline, +A, +B, +A+B. The interaction effect is (A+B) - (A) - (B) + baseline
- At minimum, verify that your modification helps on top of the *current best practice*, not just on top of a vanilla baseline. If your modification helps on top of GPT-2-style Transformer but not on top of LLaMA-3-style, the contribution is fragile

---

## Pitfall 7: Confusing Correlation with Causation in Architecture Trends

**What happens:** You observe that the best-performing models in 2024 all use SwiGLU activation. You conclude that SwiGLU is a major contributor to their performance. But all these models also trained on more data, used better curricula, and benefited from improved training infrastructure. The SwiGLU correlation does not establish causation.

**The mechanism:** Observational studies of model architectures cannot isolate individual design decisions. Only controlled experiments can.

**Defense:**
- Never cite a model's benchmark score as evidence for a component's value. "LLaMA 3 uses GQA and achieves X on MMLU" does not tell you that GQA contributes to X
- Only cite controlled ablations from the same paper: "LLaMA 2 compared MHA vs GQA and found <0.5% quality loss with 8x cache reduction"
- If no controlled ablation exists for the component you care about, that is a research opportunity — not a reason to cite correlational evidence

---

## Summary Table

| Pitfall | Detection | Defense |
|---------|-----------|---------|
| Implementation-architecture conflation | Speedup disappears with naive implementation | Separate algorithmic and kernel contributions |
| Benchmark overfitting | Gains limited to one benchmark family | Diverse evaluation suite, report all results |
| Hyperparameter trap | Improvement fragile to HP settings | Equal tuning budget, report sensitivity |
| Scale-dependent effects | Curves converge in scaling plot | Multi-scale experiments, 3+ data points |
| Stability disguised as quality | Baseline required manual intervention | Report stability and quality separately |
| Interaction effects | Improvement disappears with other modifications | Cumulative and factorial ablations |
| Correlation as causation | No controlled ablation cited | Only cite controlled experiments |
