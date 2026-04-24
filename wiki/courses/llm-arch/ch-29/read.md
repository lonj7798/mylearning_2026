# Chapter 29: Designing Architecture Experiments

<!-- scope: from observation to hypothesis, ablation design, compute-efficient experimentation, research workflow, pitfalls, writing up results
     deps: [[ch-28]]
     see-also: [[ch-24]]
-->

## Overview

[[ch-28]] taught you how to *read* architecture papers — how to separate genuine contributions from incremental tweaks, how to interrogate ablation tables, how to build taste for which problems matter. This chapter completes the arc: now you design and run your own experiments.

Architecture research is not alchemy. The researchers who produced Flash Attention, Chinchilla, and DeepSeek-V3 followed a disciplined workflow: they noticed a concrete bottleneck, proposed a structural modification with a clear hypothesis, designed controlled experiments to isolate that modification's effect, and presented their results honestly. Every one of those steps has failure modes that waste compute, produce misleading results, or bury a genuine contribution under bad presentation.

This chapter covers the full cycle. We start with how observations become hypotheses — the hardest and least teachable part. Then we work through controlled ablation design, using OLMo 2's published experiments ([[olmo-2|report]]) as a concrete case study in doing it right. We cover compute-efficient experimentation: proxy tasks, scaling plots, and when small-scale results actually predict large-scale behavior (and when they don't). We examine the research workflow as practiced at organizations like Anthropic, AI2, and DeepSeek. We catalog common pitfalls that trap even experienced researchers. And we close with how to write up results so that your contribution is clear, reproducible, and honestly presented.

This is the capstone chapter. Everything you have learned about attention variants ([[ch-07]]), normalization ([[ch-09]]), scaling laws ([[ch-10]]), MoE ([[ch-14]]), and the case studies ([[ch-18]] through [[ch-24]]) converges here. The goal is not to make you a theorist — it is to make you someone who can sit down, identify a real architectural bottleneck, and design an experiment that produces trustworthy evidence about whether your proposed fix works.

See the [experiment design workflow diagram](figures/experiment-workflow.html) for a visual overview of the full cycle.

---

## 1. From Observation to Hypothesis

The best architecture papers start with a *specific, measurable* observation about a bottleneck — not a vague sense that something could be improved.

### What Good Observations Look Like

**Flash Attention** ([[flash-attention|paper]]): Dao et al. did not start from "attention is slow." They started from a profiling observation: standard attention on an A100 achieves only 25-40% of theoretical peak FLOPs/s, and the reason is that the $N \times N$ attention matrix is materialized in HBM, requiring $O(N^2)$ memory reads/writes through a 2 TB/s pipe when SRAM offers 19 TB/s. The observation was *quantitative* — a measured gap between actual and achievable throughput — and it pointed directly at the memory hierarchy as the bottleneck, not the arithmetic.

**Chinchilla** ([[chinchilla|paper]]): Hoffmann et al. observed that the Kaplan scaling laws ([[scaling-laws-kaplan|paper]]) predicted optimal allocation heavily weighted toward model size over data. But when they fit their own scaling curves with better methodology (three independent approaches instead of one), the data pointed to roughly equal scaling of parameters and tokens. The observation was a discrepancy between the prevailing wisdom and fresh empirical evidence.

**DeepSeek-V3** ([[deepseek-v3|report]]): The DeepSeek team observed that auxiliary load-balancing losses — the standard approach to MoE routing stability — introduce a gradient signal that fights the primary language modeling objective. The observation was that models trained with auxiliary loss showed routing oscillation and suboptimal expert utilization, measurable in expert load variance over training steps.

### What Bad Observations Look Like

- "Transformers could probably be improved" — too vague to generate a testable hypothesis.
- "Nobody has tried X" — novelty is not the same as identifying a bottleneck. The question is *why* X might help, not whether it has been tried.
- "This benchmark score is low" — benchmark scores reflect the entire pipeline (data, training, architecture, evaluation). You cannot attribute a low score to an architectural limitation without further analysis.

### The Hypothesis Template

A well-formed architecture hypothesis has three parts:

1. **Bottleneck identification:** "The current approach to [component] wastes [resource] because [mechanism]."
2. **Proposed modification:** "Replacing/modifying [component] with [alternative] should reduce [resource] because [reasoning]."
3. **Predicted outcome:** "At [scale], this should improve [metric] by approximately [amount], with [expected tradeoff] as a cost."

The predicted outcome is critical. Without it, you cannot distinguish between "my hypothesis was confirmed" and "something happened." A hypothesis that cannot be falsified is not a hypothesis — it is a wish.

**Example from MLA** ([[deepseek-v2|report]]):
1. Bottleneck: "MHA stores $2 \times H \times d_k$ dimensions per token per layer in the KV cache, but the key/value representations across heads are highly redundant."
2. Modification: "Compress keys and values jointly into a low-rank latent $c_t \in \mathbb{R}^{d_c}$ where $d_c \ll 2 H d_k$, then reconstruct per-head keys and values via learned up-projections."
3. Prediction: "KV cache reduced by ~93% with quality matching or exceeding MHA, because the low-rank bottleneck forces more structured representations."

---

## 2. Controlled Ablation Design

Once you have a hypothesis, the experiment must isolate the variable you are testing. This is harder than it sounds — architecture changes have cascading effects on parameter count, compute cost, memory usage, and optimization dynamics.

### The Golden Rule: One Variable at a Time

An ablation study varies exactly one thing between the baseline and the experimental condition. Everything else — data, learning rate schedule, total training tokens, evaluation protocol, random seeds — stays identical.

This sounds obvious. In practice, it is the most frequently violated principle in architecture research. Common violations:

- **Changing the architecture and the hyperparameters simultaneously.** If you switch from ReLU to SwiGLU and also increase the learning rate because "SwiGLU trains better with higher LR," you cannot attribute the improvement to either change.
- **Different total parameter counts.** SwiGLU's gated design uses $\frac{8}{3} d_\text{model}$ instead of $4 d_\text{model}$ for the FFN, changing the parameter count. A fair comparison must either match parameters (adjust $d_\text{model}$) or match FLOPs, and report which matching criterion you used.
- **Different training durations.** If your modification converges faster and you stop training earlier, you are confounding the architectural change with the training budget.

### OLMo 2's Ablation Design: A Case Study

OLMo 2 ([[olmo-2|report]]) is an excellent example of disciplined ablation work because AI2 published not just the final architecture, but the intermediate experiments that led to each decision. See [excerpts/olmo2-ablations.md](excerpts/olmo2-ablations.md) for the detailed analysis.

**What they tested:**

| Decision | Baseline | Experimental | Matching criterion |
|----------|----------|--------------|-------------------|
| Normalization | Non-parametric LayerNorm (OLMo 1) | RMSNorm | Same parameter count, same training tokens |
| Positional encoding | Absolute learned embeddings | RoPE | Same model config otherwise |
| Attention stability | No QK-norm | QK-norm before softmax | Same FLOPs, same data |
| Regularization | No Z-loss | Z-loss ($\lambda = 10^{-4}$) | Same everything else |
| Training curriculum | Single-stage (all data mixed) | Two-stage (web then curated) | Same total token count |
| Checkpoint selection | Best single checkpoint | Model souping (weight average) | Same total training compute |

**Why this works:**

1. **Each row is one variable.** You can read down the table and attribute each improvement to a specific choice.
2. **The matching criterion is explicit.** They tell you *what* was held constant, so you can assess whether the comparison is fair.
3. **Cumulative ablations.** They built up from the OLMo 1 baseline, adding one modification at a time, so you can see each increment. This is more expensive than testing everything against a single baseline, but it reveals interactions between modifications.
4. **Full artifact release.** Intermediate checkpoints, training logs, and data are all public. Anyone can verify the claims.

### Choosing Baselines

Your baseline determines the meaning of your results. Common baseline strategies:

**The "vanilla Transformer" baseline:** Start with a standard configuration (e.g., LLaMA-style: GQA, RoPE, RMSNorm, SwiGLU) and modify one component. This tells you whether your change improves over the current best practice.

**The "match the original paper" baseline:** Reproduce the baseline from the paper you are comparing against, then add your modification. This is more labor-intensive but avoids the risk that differences in your implementation (rather than the architecture) explain the results.

**The "your own last model" baseline:** Use your previous best model as the baseline. This is practical for iterative research but makes it harder for others to reproduce, since your baseline is a moving target.

OLMo 2 used the first strategy: OLMo 1 (a known, published, reproducible baseline) plus incremental changes. This is the gold standard.

### Controlling for Compute

Architecture changes almost always change the compute cost. A wider FFN uses more FLOPs per forward pass. MoE adds router overhead. MLA adds up-projection compute at inference. You must decide what to hold constant:

- **FLOPs-matched:** Same total training FLOPs. This is the fairest comparison for training efficiency claims. Chinchilla used this.
- **Parameter-matched:** Same parameter count. This is appropriate when you are testing whether a fixed-size model can be made better. But beware: parameter-matched models may have very different FLOPs.
- **Wall-clock-matched:** Same total training time on the same hardware. This captures implementation efficiency (memory layout, kernel support) in addition to algorithmic efficiency. DeepSeek-V3 emphasized this — their $5.6M training cost is a wall-clock claim, not a FLOPs claim.

Report which matching criterion you used, and ideally report results under multiple criteria. A modification that wins on FLOPs but loses on wall-clock (because it requires custom kernels that are slower than optimized standard kernels) is a weaker contribution than one that wins on both.

---

## 3. Compute-Efficient Experimentation

You cannot test every hypothesis at full scale. A single 70B training run costs millions of dollars. The core question of compute-efficient experimentation: **when do small-scale results predict large-scale behavior, and when do they mislead?**

### Proxy Tasks

A proxy task is a smaller, cheaper version of the real evaluation that correlates with the full result. Good proxy tasks:

- **Reduced-scale models:** Train at 150M-1B parameters instead of 70B. If your modification improves loss at 150M, 300M, and 1B, the trend is likely to continue. OLMo 2 used 1B-scale ablations before committing to 7B/13B/32B runs.
- **Reduced data:** Train for 10B tokens instead of 1T. Early training dynamics often reveal whether a modification helps — though some effects only appear late in training (e.g., two-stage curricula).
- **Targeted evaluation:** Instead of running the full MMLU/HellaSwag/ARC suite, evaluate on a small, diverse subset that correlates with the full suite. Chinchilla's scaling law fits used validation loss, not downstream benchmarks, for exactly this reason.

**When proxies fail:**

- **MoE routing dynamics change with expert count.** A 4-expert model does not exhibit the same load-balancing challenges as a 256-expert model. DeepSeek discovered auxiliary-loss-free balancing only after observing routing collapse at scale.
- **Normalization effects depend on depth.** QK-norm may be unnecessary at 12 layers but critical at 64 layers, because attention logit growth compounds with depth.
- **Long-context behavior.** A model trained on 4K context says nothing about 128K behavior. Context extension techniques must be tested at their target length.

### Scaling Plots: The Researcher's Best Friend

A scaling plot shows how your metric changes as you increase model size (or data, or compute) — and whether the baseline and experimental curves diverge, converge, or cross. See the [interactive scaling plot template](figures/scaling-plot.html) for a hands-on example.

**How to build one:**

1. Train your baseline and experimental variant at 3-5 scales (e.g., 125M, 350M, 760M, 1.3B, 2.7B).
2. Plot validation loss (y-axis, log scale) against training FLOPs or parameters (x-axis, log scale).
3. Fit power-law curves to both series: $L(C) = \alpha \cdot C^{-\beta} + L_\infty$.
4. Extrapolate to the target scale and assess whether the experimental curve remains below the baseline.

**What the plot tells you:**

- **Persistent gap:** The experimental curve is consistently below the baseline at all scales. Strong evidence the modification will help at larger scale. Chinchilla's data-scaling curves showed this.
- **Converging curves:** The gap narrows with scale. The modification may not matter at your target size. Many attention approximation methods show this pattern — they help at small scale where the quadratic cost is relatively higher, but Flash Attention's constant-factor improvement dominates at large scale.
- **Crossing curves:** The baseline is better at small scale, the experimental variant at large scale (or vice versa). This is informative but dangerous — you must decide whether the crossing point is real or a statistical artifact.
- **Diverging curves:** The gap widens with scale. The modification becomes *more* valuable at larger scale. MLA showed this pattern — the KV compression benefit scales superlinearly because the cache grows with both model width and head count.

**Critical caveat:** Power-law extrapolation assumes the scaling exponent is stable. Chinchilla showed that Kaplan's original scaling exponents were wrong because the training procedure was suboptimal at different scales. If your small-scale runs use different hyperparameters than your large-scale runs will, the extrapolation is unreliable. This is why Chinchilla used three independent methodologies to cross-validate their scaling predictions.

### The "Sufficiently Informative" Experiment

Not every experiment needs to run to convergence. Useful stopping criteria:

- **Early divergence:** If your modification is 0.1 nats worse than the baseline after 5B tokens, it is almost certainly worse after 100B tokens. Kill it.
- **Confidence intervals:** Run 3 seeds at small scale. If the variance across seeds is larger than the gap between baseline and experimental, you need either more seeds or a bigger effect size to draw conclusions.
- **Sanity checks first:** Before any scaling experiment, verify that your implementation exactly reproduces the baseline when the modification is disabled. A surprising number of "improvements" are actually bugs in the baseline implementation.

---

## 4. The Architecture Research Workflow

The individual skills — observation, hypothesis, ablation, scaling — combine into a workflow. Different organizations structure this differently, but the core loop is consistent.

### The General Loop

```
Observe bottleneck
    |
    v
Form hypothesis (bottleneck + modification + prediction)
    |
    v
Sanity check: does this even make sense theoretically?
    |
    v
Small-scale ablation (1B, 10B tokens, 3 seeds)
    |
    v
Result negative? --> Revise hypothesis or abandon
Result ambiguous? --> More seeds or larger scale
Result positive? --> Scaling plot (3-5 scales)
    |
    v
Scaling plot predicts improvement at target scale?
    |
    v
Full-scale validation run
    |
    v
Analysis, writing, release
```

### How Organizations Differ

**AI2 (OLMo):** Maximum transparency. Every ablation is published with full data and code. The community can critique and extend the work. The tradeoff: slower iteration, because open publication requires more careful documentation. See [[olmo-2|report]] and the published training logs.

**DeepSeek:** Aggressive hypothesis-testing with very tight feedback loops. DeepSeek-V2 and V3 introduced multiple novel components (MLA, auxiliary-loss-free balancing, multi-token prediction) in rapid succession. They publish detailed technical reports but not full training data or intermediate checkpoints. Their advantage is speed of iteration; their risk is that multiple simultaneous changes make individual contributions harder to isolate.

**Anthropic:** (As described in public talks and the Berkeley course [[berkeley-llm-agents-f24|blog]]) Emphasis on identifying the *right* bottleneck before running any experiments. Research taste — knowing which problems are worth solving — is valued as highly as experimental execution. The workflow includes internal review gates where the hypothesis itself is scrutinized before compute is allocated. Ben Mann's Berkeley lecture emphasized that measuring agent capabilities requires the same experimental rigor as measuring architecture improvements: controlled baselines, clear metrics, reproducible protocols.

**The common thread:** All three organizations spend more time on hypothesis formation and experimental design than on running the actual experiments. GPU time is expensive; thinking time is free. The researchers who waste the least compute are those who are most disciplined about killing bad ideas early.

---

## 5. Common Pitfalls

Architecture experiments fail in predictable ways. See [excerpts/common-pitfalls.md](excerpts/common-pitfalls.md) for extended examples.

### Pitfall 1: Confusing Implementation Improvements with Architecture Improvements

This is the most common error. You change the architecture *and* the implementation (better CUDA kernels, different memory layout, fused operations), and the improvement comes from the implementation, not the architecture.

**The test:** Can someone reproduce your result using a naive implementation of the new architecture, without your custom kernels? If not, your contribution is an engineering optimization (valuable, but different from an architecture contribution).

Flash Attention is a rare example where this distinction is handled perfectly. Dao et al. are explicit that Flash Attention computes the *exact same result* as standard attention — the contribution is purely algorithmic (tiling + online softmax to reduce HBM traffic). The architecture is unchanged; the implementation is better. They claim an implementation contribution, not an architecture contribution, and the paper is stronger for it.

Contrast with many linear attention papers that claim both architectural novelty (different attention mechanism) and implementation novelty (custom kernels), making it impossible to tell which factor explains the observed speedup.

### Pitfall 2: Overfitting to Benchmarks

An architecture modification that improves MMLU by 2 points but degrades open-ended generation quality is not an improvement — it is benchmark overfitting. This happens when:

- The modification adds inductive biases that help on multiple-choice tasks (e.g., better calibration) but hurt on free-form generation.
- The evaluation suite is too narrow. Evaluating only on knowledge-recall benchmarks misses degradation in reasoning, code generation, or instruction following.
- The modification was tuned (consciously or unconsciously) on the evaluation set. If you tried 20 variants and report the one that scores highest on MMLU, you have selection bias.

**The defense:** Evaluate on a *diverse* suite that includes tasks your modification was not designed to help. OLMo 2's OLMES framework uses 20 benchmarks spanning knowledge recall, commonsense reasoning, general reasoning, and mathematical reasoning. If your modification helps on 18 and hurts on 2, that is evidence; if it helps on 3 and is neutral on 17, the signal is weaker than it appears.

### Pitfall 3: Ignoring Interaction Effects

Architecture components interact. QK-norm may be unnecessary with Pre-LN but critical with Post-LN. SwiGLU may interact with learning rate scheduling. MoE routing dynamics change with sequence length, batch size, and expert count.

**The defense:** Cumulative ablations (as OLMo 2 did). Test your modification both in isolation *and* in combination with other changes. If the improvement disappears when combined with another common modification, the contribution is fragile.

### Pitfall 4: The Hyperparameter Trap

Your modification introduces a new hyperparameter (e.g., MLA's latent dimension $d_c$, MoE's number of experts, SWA's window size). You tune this hyperparameter extensively for your experimental condition but use default hyperparameters for the baseline. This gives your modification an unfair advantage.

**The defense:** Either use the same hyperparameter tuning budget for both conditions, or report results across a range of your new hyperparameter to show that the improvement is robust to reasonable settings.

### Pitfall 5: Misattributing Scaling Behavior

Your modification helps at 1B parameters. You claim it will help at 70B. But scaling behavior is not linear — effects can diminish, amplify, or reverse. MoE routing dynamics, attention pattern diversity, and normalization effects all behave differently at different scales.

**The defense:** Scaling plots with multiple points (Section 3). Never extrapolate from a single scale.

---

## 6. Writing Up Results

A good experiment with bad presentation is indistinguishable from a bad experiment. The goal of the write-up is to make your contribution *easy to evaluate* — both for reviewers who want to assess the work and for practitioners who want to decide whether to adopt it. See [excerpts/flash-attention-contribution.md](excerpts/flash-attention-contribution.md) for a detailed analysis of exemplary presentation.

### The Structure of a Good Architecture Paper

**1. Motivation (1-2 paragraphs):** What is the bottleneck? Who cares? Use numbers. "Attention is slow" is not motivation. "Standard attention achieves 25-40% of peak FLOPs/s on A100 because the $N \times N$ matrix is materialized in 2 TB/s HBM" is motivation.

**2. Method (the bulk):** Describe the modification precisely enough that someone can implement it from your description alone. Include equations, pseudocode, and a figure. Explicitly state what is *not* changed from the baseline.

**3. Experimental setup:** Specify everything needed to reproduce the result: model sizes, data, training tokens, learning rate schedule, hardware, evaluation protocol, number of seeds. OLMo 2 is the gold standard here — they release code, data, and thousands of intermediate checkpoints.

**4. Results:** Tables and scaling plots. Every row in an ablation table must differ from the baseline in exactly one way. State the matching criterion (FLOPs-matched, parameter-matched, wall-clock-matched).

**5. Analysis:** Why does it work? Go beyond "it scored higher." Flash Attention's analysis explains *why* tiling reduces HBM accesses via an IO complexity analysis. Chinchilla's analysis explains *why* the Kaplan scaling exponents were wrong (suboptimal learning rates at each scale). DeepSeek-V2's analysis explains *why* the low-rank bottleneck acts as a regularizer.

**6. Limitations:** What does your modification *not* do? Where does it fail? What are the costs? This section builds trust. Flash Attention acknowledges that it provides no speedup for short sequences where the attention matrix fits in SRAM. Chinchilla acknowledges that their scaling laws were fit to a specific data distribution. MLA acknowledges that custom inference kernels are required.

### Honesty Signals

Reviewers and practitioners look for these signs that the authors are presenting results honestly:

- **Reporting negative results.** "We also tried X, which did not help" is more informative than silence.
- **Multiple evaluation axes.** Reporting speed, quality, memory, *and* implementation complexity.
- **Comparison to the *actual* state of the art**, not a strawman baseline from 2020.
- **Error bars or multiple seeds.** A single run is an anecdote, not evidence.
- **Code release.** If you cannot release the code, explain why. If you can, do.

### What DeepSeek-V3 Gets Right

The DeepSeek-V3 report ([[deepseek-v3|report]]) is a masterclass in honest presentation of a complex system:

- They report the total training cost ($5.6M) prominently, making it easy to assess practical feasibility.
- Each novel component (MLA, auxiliary-loss-free balancing, multi-token prediction) gets its own ablation section.
- They compare against specific named baselines (DeepSeek-V2, Llama 3.1 405B, Qwen 2.5), not against "prior work."
- They are explicit about what is *not* novel: the dense attention layers use standard GQA, the training infrastructure builds on Megatron-LM.
- They acknowledge limitations: MLA requires custom inference kernels, and their scaling predictions are specific to their data distribution.

---

## 7. Putting It All Together: A Worked Example

Suppose you are researching KV-cache efficiency for a 70B model and you observe that GQA with 8 KV heads still consumes significant memory at 128K context. How would you design an experiment?

**Step 1: Quantify the bottleneck.** At 70B scale (80 layers, $d_k = 128$), GQA-8 KV cache per token = $2 \times 8 \times 128 \times 80 \times 2$ bytes (FP16) = 327 KB. At 128K context: 327 KB $\times$ 131,072 = 42 GB per request. This exceeds a single A100's memory.

**Step 2: Form hypothesis.** "If we combine GQA-8 with MLA-style low-rank compression ($d_c = 256$) on alternating layers, we can reduce the cache by approximately 40% with <0.5% quality degradation, because the key/value redundancy across positions grows with sequence length."

**Step 3: Design ablations.** At 1B scale, 4K context:

| Config | KV strategy | Predicted cache | Control |
|--------|------------|-----------------|---------|
| A (baseline) | GQA-8, all layers | 100% | -- |
| B | GQA-8 even layers, MLA odd layers | ~60% | Architecture only |
| C | MLA all layers | ~20% | Architecture only |
| D | GQA-8, all layers, matched params to B | 100% | Parameter control |

Run each config for 50B tokens, 3 seeds. Evaluate on validation loss + 5 diverse benchmarks.

**Step 4: Scaling plot.** If B beats A at 1B, run at 350M, 1B, 2.7B. Fit power laws. Check if the gap persists, widens, or narrows.

**Step 5: Context-length validation.** The hypothesis specifically claims benefits at long context. You must validate at 32K and 128K, not just 4K. This is the proxy-task failure mode from Section 3 — short-context experiments cannot validate long-context claims.

**Step 6: Write up.** Report the bottleneck calculation, the hypothesis, the ablation results with error bars, the scaling plot, and the long-context validation. Acknowledge that the hybrid GQA/MLA approach requires two different kernel paths at inference, adding engineering complexity.

---

## Core Insights from the Literature

### Insight 1: The best architecture papers start from quantified bottlenecks, not from novelty

**Papers:** [[flash-attention|paper]], [[chinchilla|paper]], [[deepseek-v2|report]]

Flash Attention started from "25-40% of peak FLOPs/s." Chinchilla started from "existing scaling laws predict the wrong data/parameter ratio." DeepSeek-V2 started from "MHA stores 32K dimensions per token when the information content requires only 576." In each case, the *observation* determined the *solution*. Researchers who start from "wouldn't it be cool if..." produce novelty without impact. **Guideline:** Before writing a single line of code, quantify the gap between current performance and theoretical optimal on whatever resource you are targeting (FLOPs, memory, bandwidth, quality-per-FLOP). If you cannot measure the gap, you do not have a hypothesis yet.

### Insight 2: Controlled ablations require explicit matching criteria

**Paper:** [[olmo-2|report]]

OLMo 2's ablation tables are exemplary because every comparison specifies *what was held constant*: same training tokens, same model size, same learning rate schedule. Without this, an ablation table is meaningless — you cannot tell whether the improvement comes from the architectural change or from the confounding variable. DeepSeek-V3 similarly separates its contributions (MLA, routing, MTP) into independent ablation sections, each with its own baseline. **Guideline:** For every ablation, explicitly state the matching criterion: FLOPs-matched, parameter-matched, or wall-clock-matched. Report the criterion in the table caption, not buried in a footnote.

### Insight 3: Scaling plots are the strongest evidence for architecture claims

**Papers:** [[chinchilla|paper]], [[scaling-laws-kaplan|paper]]

A single-scale result can mislead. An improvement at 1B that vanishes at 13B is not worth publishing as an architecture contribution. Scaling plots with 3-5 points and power-law fits give confidence that results will hold at the target scale. Chinchilla's power-law fits (using three independent methodologies) are the strongest evidence in the scaling-law literature precisely because they show the trend holding across orders of magnitude. **Guideline:** Never publish an architecture claim based on a single scale. Minimum three data points on a log-log plot with fitted curves. If the curves cross or converge, say so explicitly.

### Insight 4: Implementation and architecture contributions are different and should not be conflated

**Paper:** [[flash-attention|paper]]

Flash Attention is clear that it computes the *exact same result* as standard attention — the contribution is an IO-aware implementation, not a new attention mechanism. This clarity makes the paper stronger, not weaker, because the reader knows exactly what is being claimed. Contrast with papers that change the attention mechanism *and* introduce custom kernels, making it impossible to determine which factor explains the speedup. **Guideline:** If your contribution is an implementation optimization, benchmark it against a naive implementation of the same architecture to isolate the implementation effect. If your contribution is an architecture change, implement both the baseline and the variant using the same level of optimization.

### Insight 5: Honest presentation of limitations builds trust and impact

**Papers:** [[olmo-2|report]], [[deepseek-v3|report]], [[flash-attention|paper]]

OLMo 2 acknowledges that their conservative 4K context length is a deliberate limitation — they focused resources on data quality and training stability rather than context extension. DeepSeek-V3 acknowledges that MLA requires custom inference kernels. Flash Attention acknowledges no speedup for short sequences. These limitations sections make the papers *more* credible, not less, because they demonstrate that the authors understand the boundaries of their contribution. **Guideline:** Every architecture paper should have a limitations section that answers: (1) where does this modification not help? (2) what does it cost? (3) what assumptions must hold for the claims to be valid?

---

## Key Takeaways

1. **Start from a quantified bottleneck.** The best architecture research begins with a measured gap between current and achievable performance — in FLOPs utilization, memory bandwidth, cache size, or quality-per-FLOP. If you cannot measure the gap, you do not have a research direction yet.

2. **One variable at a time.** Every ablation must differ from its baseline in exactly one dimension. State the matching criterion (FLOPs, parameters, wall-clock) explicitly. OLMo 2's cumulative ablation tables are the template.

3. **Scaling plots are mandatory.** Never claim an architecture improvement based on a single scale. Train at 3-5 sizes, fit power laws, and check whether the curves diverge, converge, or cross. Chinchilla's three-methodology approach is the gold standard.

4. **Proxy tasks have known failure modes.** Small-scale models cannot predict MoE routing dynamics at 256 experts, normalization behavior at 64+ layers, or long-context performance. Know which claims your proxies can and cannot support.

5. **Separate implementation from architecture.** Flash Attention's clarity about computing the exact same result — just faster — is the model. If you change both the algorithm and the kernel, you cannot attribute the improvement to either.

6. **Kill bad ideas early.** If your modification is worse than the baseline after 5B tokens at 1B scale, it is almost certainly worse at full scale. The researchers who waste the least compute are the most disciplined about early termination.

7. **Present results honestly.** Report negative results, use multiple evaluation axes, compare against current baselines (not 2020 strawmen), include error bars, and write a real limitations section. Honest presentation builds the trust that gets your work adopted.

8. **The research workflow is iterative.** Observe, hypothesize, ablate, scale, analyze, write — and loop. The best researchers spend more time on the first two steps (observation and hypothesis) than on running experiments. GPU time is expensive; thinking time is free.

---

## References

- [[flash-attention|Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022) (paper)]] — exemplary bottleneck identification and honest contribution framing
- [[chinchilla|Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) (paper)]] — gold standard for scaling plots and multi-methodology validation
- [[olmo-2|AI2, "OLMo 2: 2 OLMo 2 Furious" (2024-2025) (report)]] — exemplary ablation design with full artifact release
- [[deepseek-v2|DeepSeek AI, "DeepSeek-V2" (2024) (report)]] — hypothesis-driven architecture innovation (MLA)
- [[deepseek-v3|DeepSeek AI, "DeepSeek-V3" (2024) (report)]] — honest multi-component presentation with cost transparency
- [[scaling-laws-kaplan|Kaplan et al., "Scaling Laws for Neural Language Models" (2020) (paper)]] — foundational scaling methodology
- [[gqa|Ainslie et al., "GQA" (2023) (paper)]] — controlled ablation with uptraining recipe
- [[berkeley-llm-agents-f24|UC Berkeley, "CS294/194-196: Large Language Model Agents" (Fall 2024) (blog)]] — research measurement methodology (Ben Mann, Anthropic)
