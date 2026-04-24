# Chapter 28: Reading and Critiquing Architecture Papers

<!-- scope: research methodology — how to read, dissect, and evaluate LLM architecture papers
     deps: [[ch-18]], [[ch-19]], [[ch-20]], [[ch-21]]
     see-also: [[ch-29]]
-->

## Overview

You have spent twenty-seven chapters absorbing architecture ideas: attention variants, normalization strategies, MoE routing, state space models, long-context techniques, serving optimizations. Every one of those ideas arrived inside a research paper or technical report. But reading a paper is not the same as *critiquing* one, and critiquing a paper is not the same as knowing whether its ideas will survive contact with reality.

This chapter is about the meta-skill: how to read architecture papers efficiently, how to identify the real contribution buried under the formatting, how to interrogate ablation tables until they confess their actual story, and how to develop the taste that lets you predict which papers will matter two years from now. The skills here are not theoretical — they are the daily practice of anyone doing architecture research at a lab like Anthropic, DeepSeek, or Google Brain.

We will use papers you already know as case studies: Flash Attention ([[flash-attention|paper]]), GQA ([[gqa|paper]]), DeepSeek-V2's MLA ([[deepseek-v2|report]]), and Mamba ([[mamba|paper]]). The goal is not to re-learn these techniques but to re-examine *how they were presented* and what a critical reader should have noticed on first read.

---

## 1. Paper Anatomy: The Three-Pass Method

Most people read papers linearly, from abstract through conclusion. This is the slowest way to extract value. Architecture papers have a predictable structure, and you can exploit that structure to read non-linearly.

### Pass 1: The Five-Minute Skeleton (Abstract + Figures + Results Tables)

Read the abstract, then skip directly to the figures and main results tables. In a well-written architecture paper, the figures contain the entire architectural idea, and the results tables tell you whether it worked. You should be able to answer three questions after five minutes:

1. **What is the claimed bottleneck?** (e.g., "attention is memory-bandwidth-bound" in Flash Attention)
2. **What is the proposed structural change?** (e.g., "tile attention for SRAM, never materialize the N x N matrix")
3. **How big is the claimed improvement?** (e.g., "2-4x wall-clock speedup, exact results")

If the figures are unclear or the abstract is vague about the bottleneck, that is itself a signal. Papers that cannot clearly state their contribution in the abstract rarely have a clean contribution to state.

**Case study — Flash Attention ([[flash-attention|paper]]):** The abstract names the bottleneck explicitly: "a missing principle is making attention algorithms IO-aware." The key figure shows the tiling scheme and memory hierarchy. The results table reports wall-clock speedup, not just FLOP reduction. Within five minutes, you know: the bottleneck is HBM-SRAM data movement, the fix is tiling, and the results are measured in the metric that matters (wall-clock time). This is a well-structured paper.

**Case study — GQA ([[gqa|paper]]):** The abstract tells you the bottleneck (MQA quality degradation), the fix (grouped KV heads), and the key practical contribution (5% compute uptraining recipe). The ablation figures directly compare MHA, MQA, and GQA at matched inference budgets. Five minutes gives you the full story.

### Pass 2: The Ablation Read (15-20 Minutes)

Now read the ablation tables and experimental methodology. This is where the paper's claims either hold up or fall apart. We cover ablation literacy in depth in Section 4, but the key questions for this pass are:

- What baselines are used? Are they fair?
- What is ablated? Is each component isolated?
- What is the evaluation metric? Does it match the claimed bottleneck?

Skip the related work section entirely on this pass. Related work is written for reviewers, not for practitioners.

### Pass 3: The Detail Read (30-60 Minutes, Only If Passes 1-2 Survive)

Now read the full paper, including the methodology details, training setup, and related work. You earn the right to spend an hour on a paper only after the first two passes have convinced you the contribution is real and relevant. Most papers do not survive to pass 3.

**The critical ratio:** A productive researcher reads ~50 papers at pass 1, ~10 at pass 2, and ~3 at pass 3 per month. If you are spending an hour on every paper, you are either reading too few papers or reading too slowly.

See [the paper reading flowchart](figures/paper-reading-flowchart.html) for a visual guide to this process.

---

## 2. Mental Ablation: The Core Reasoning Skill

The most powerful question you can ask about any design choice in an architecture paper is: **"What would happen without this?"**

This is *mental ablation* — running the experiment in your head before (or instead of) running it on a GPU. For every component the paper introduces, you should be able to articulate:

1. What failure mode does this component prevent?
2. What would the model do if this component were removed?
3. Is there a simpler alternative that prevents the same failure mode?

### Worked Example: DeepSeek-V2's Decoupled RoPE

MLA ([[deepseek-v2|report]]) compresses keys and values into a low-rank latent vector $c_t$ and reconstructs them via up-projection at inference time. But RoPE is applied *directly to keys* and is position-dependent. If you compress the key into $c_t$ and then apply RoPE to the reconstructed key, you get the wrong result — the position information was not present when $c_t$ was computed.

**Mental ablation — remove decoupled RoPE:**

- Without it, the model has no position information in the compressed keys
- Attention scores would be position-invariant — the model cannot distinguish "token A at position 5" from "token A at position 500"
- This would catastrophically break autoregressive generation: the model needs position to enforce causal ordering

**Mental ablation — apply RoPE before compression:**

- The latent $c_t$ would encode position information, but it would be entangled with the content representation
- Different positions would produce different $c_t$ values for identical tokens, preventing any sharing or regularization benefit
- The low-rank bottleneck would need to preserve both content and position, splitting its limited capacity

DeepSeek's actual solution — a small separate key/value pair ($d_h^R = 64$ dims) that carries position information alongside the content-latent — is the minimal fix that solves the problem. The decoupled component adds only 64 dims to the 512-dim latent cache, a 12.5% overhead for correct positional encoding.

**The mental ablation revealed:** The decoupled RoPE is not an optional enhancement — it is a structural necessity. Without it, MLA is broken. This is a contribution that passes the "what would happen without this?" test decisively.

### Worked Example: GQA's Mean-Pooling Initialization

GQA ([[gqa|paper]]) proposes converting existing MHA checkpoints to GQA using 5% of original pre-training compute. The key initialization choice is mean-pooling the KV heads within each group.

**Mental ablation — use random initialization instead:**

- The converted model starts with random KV projections that bear no relation to the learned MHA representations
- 5% of pre-training compute must now learn KV representations from scratch — likely insufficient
- The model would need more uptraining compute to recover quality

**Mental ablation — select a single head per group instead:**

- The converted model preserves one head's learned representation but discards the others
- Information from the discarded heads is permanently lost
- The selected head may not be the "best" for the group

Mean-pooling preserves the maximum information from all original heads in a single representation. The ablation table in the paper confirms this: mean-pooling outperforms both alternatives, and the gap is larger at lower uptraining budgets (where the initialization matters most).

**The lesson:** When a paper proposes multiple design choices, mentally ablate each one independently. The choices that matter most are the ones where removal causes the largest degradation.

---

## 3. Identifying the Real Contribution

Every architecture paper claims novelty, but not every claimed novelty is a real contribution. The research community has developed informal categories for distinguishing genuine contributions from incremental work:

### The Contribution Taxonomy

**Category 1: New Bottleneck Identification**
The paper identifies a bottleneck that the community had not recognized or had mischaracterized. This is the rarest and most valuable type of contribution.

- **Flash Attention:** Identified that attention's real bottleneck is memory IO, not FLOPs. Before Flash Attention, the community was chasing approximate attention methods (Linformer, Performer) that reduced FLOPs but did not reduce HBM traffic. The bottleneck identification was the contribution; the tiling algorithm followed naturally.
- **MQA ([[mqa|paper]]):** Identified that autoregressive decoding is memory-bandwidth-bound. This reframed all subsequent KV-cache optimization work.

**Category 2: Novel Structural Solution**
The paper proposes a new architectural component that solves a known problem in a structurally different way from prior work.

- **MLA:** The problem (KV cache is too large) was known. GQA and MQA had addressed it by reducing the number of KV heads. MLA proposed a fundamentally different approach: compress the *content* via a learned low-rank bottleneck. This is a genuinely new structural idea.
- **Mamba's selective SSMs ([[mamba|paper]]):** The problem (attention is O(n^2)) was known. Linear attention and S4 had addressed it. Mamba's contribution was making SSM parameters input-dependent, solving the content-based reasoning failure of prior SSMs. The selection mechanism is a new structural primitive.

**Category 3: Engineering Optimization**
The paper improves an existing method through better implementation, hardware utilization, or hyperparameter tuning. Valuable, but different from architectural novelty.

- **Flash Attention 2 ([[flash-attention-2|paper]]):** Better warp partitioning and reduced non-matmul FLOPs. The algorithmic idea (tiling for SRAM) was unchanged from FA-1. The contribution was engineering: 2x faster through better GPU utilization.
- **GQA's uptraining recipe:** The grouped-query concept is a straightforward interpolation between MHA and MQA. The real contribution was the practical recipe (mean-pooling + 5% compute) that made GQA deployable without retraining.

**Category 4: Empirical Validation at Scale**
The paper demonstrates that a known technique works at a scale where it had not been tested.

- **Llama 2 ([[llama-2|report]]):** GQA had been proposed in a paper. Llama 2 validated it at 70B parameters in production. The architectural contribution was minimal; the empirical validation was the value.
- **DeepSeek-V3 ([[deepseek-v3|report]]):** Validated MLA + fine-grained MoE + auxiliary-loss-free balancing at 671B total parameters. Each component had been introduced earlier; the contribution was demonstrating they compose at scale.

### Red Flags: Contributions That Are Not Contributions

Watch for these patterns that inflate the appearance of novelty:

1. **Renaming existing techniques.** If the "novel" method is mathematically equivalent to a prior method with different notation, it is not new. Check the math, not the names.

2. **Combining known components.** "We combine technique A with technique B" is engineering, not research, unless the combination reveals an unexpected interaction.

3. **Benchmark shopping.** If the paper reports results on 15 benchmarks but only beats baselines on 3, look at which 3. Papers that only improve on narrow benchmarks may have overfit to those benchmarks.

4. **Missing baselines.** If the paper does not compare against the obvious baseline, ask why. The most common reason is that the obvious baseline performs better.

---

## 4. Ablation Table Literacy

Ablation tables are where architecture papers prove (or fail to prove) that their design choices matter. Reading ablation tables critically is a distinct skill from reading results tables.

### The Anatomy of a Good Ablation

A good ablation study isolates one variable at a time, holds everything else constant, and reports the effect on a metric that the reader cares about. Here is what to check:

**1. Is each row a single-variable change?**

The best ablation tables change exactly one thing per row. If a row changes two things simultaneously (e.g., "remove GQA and reduce hidden dim"), you cannot attribute the effect to either change.

**Case study — DeepSeek-V2 MLA ablation:**

The DeepSeek-V2 report includes an ablation comparing MLA against MHA, GQA, and MQA at matched KV cache budgets. Each row changes only the attention mechanism. The training setup, data, optimizer, and all other hyperparameters are held constant. This is a clean ablation.

**2. Is the baseline meaningful?**

The baseline should be the strongest reasonable comparison, not a straw man. Watch for:

- Baselines that use older or weaker hyperparameters than the proposed method
- Baselines that are trained for fewer steps or on less data
- Missing the "obvious" baseline (e.g., a paper proposing a new attention variant that does not compare against GQA)

**3. Are the error bars or variance reported?**

A 0.3% improvement with no error bars could easily be noise. For language modeling, perplexity differences below ~0.5% at the same scale are often within random seed variance. If the paper does not report variance across seeds, mentally widen the confidence interval.

**4. What metric is being ablated?**

The ablated metric should match the paper's claimed contribution. If the paper claims "faster inference" but ablates perplexity, the ablation does not support the claim. If it claims "better quality" but only reports speed, same problem.

See [the ablation analysis walkthrough](figures/ablation-analysis-walkthrough.html) for an interactive example using real ablation tables from Flash Attention and DeepSeek-V2.

### Common Ablation Confounds

**Training compute confound:** When the proposed method trains for the same number of steps but uses more compute per step (e.g., due to recomputation), the "ablation" is actually comparing different compute budgets. Flash Attention's recomputation in the backward pass adds FLOPs but reduces memory — the right comparison is wall-clock time, not FLOPs.

**Scale confound:** An ablation at 1B parameters does not prove a result at 100B parameters. Architectural effects can appear, disappear, or reverse at different scales. MLA's advantage over GQA is much larger at 100B+ than at 7B — ablations at small scale would underestimate MLA's value.

**Data confound:** If the proposed method was developed on the same data used for evaluation, the ablation may reflect overfitting to the evaluation set. This is particularly insidious when the "ablation" uses a held-out evaluation set that the authors have iterated against during development.

**Hyperparameter confound:** The proposed method may have been tuned more carefully than the baselines. If the authors spent weeks tuning their method's learning rate but used default hyperparameters for baselines, the comparison is unfair.

---

## 5. Reproducibility Assessment

After you have identified the contribution and verified the ablations, the final technical question is: **could you implement this from the paper alone?**

This question reveals how well the authors understand and communicate their own method. It also determines whether the contribution can propagate through the community.

### The Implementation Checklist

For each architectural component, check whether the paper specifies:

| Detail | Why It Matters | Common Omissions |
|--------|---------------|-----------------|
| Exact tensor shapes and dimensions | Determines parameter count and memory | Often buried in appendices or code-only |
| Initialization scheme | Wrong init can prevent training entirely | Frequently omitted or described vaguely |
| Normalization placement | Pre-norm vs post-norm changes training dynamics | Often assumed from context, not stated |
| Activation function and its variant | SwiGLU vs GELU vs ReLU affects parameter count | Sometimes stated only as "standard" |
| Training hyperparameters | LR, schedule, batch size, optimizer | Often in appendix; critical for reproduction |
| Hardware and parallelism strategy | Affects whether results are achievable | Usually present but with incomplete detail |

### Reproducibility Spectrum

Papers fall on a spectrum from "fully reproducible" to "requires the authors' codebase":

**Fully reproducible (rare):** The paper provides enough detail to implement from scratch. OLMo ([[olmo-2|report]]) is the gold standard — weights, data, training code, intermediate checkpoints, and training logs are all public. You can reproduce the entire training run.

**Reproducible with effort:** The paper provides the key architectural details and training recipe, but some implementation details require reading the code or making educated guesses. Most architecture papers fall here. Flash Attention provides the algorithm in pseudocode but implementing the CUDA kernel requires significant GPU programming expertise.

**Requires the codebase:** The paper describes the method at a high level, but critical details are only in the released code (or not released at all). The DeepSeek-V2 report provides extensive architectural detail for MLA but the device-limited routing implementation requires reading their codebase.

**Unreproducible:** The paper describes the method but omits critical details, and no code is released. Proprietary papers from labs that do not release code sometimes fall here. If you cannot reproduce a result, you cannot verify the contribution.

### What Omissions Tell You

Missing implementation details are not always innocent:

- **Missing initialization details** often mean the method is sensitive to initialization. If the authors found a specific initialization that works, they should report it. If they do not, either they did not investigate sensitivity (sloppy) or the method is fragile (concerning).

- **Missing hyperparameter sensitivity analysis** often means the method requires careful tuning. Methods that work across a range of hyperparameters are more robust than methods that require precise tuning, and authors who have done the sensitivity analysis will report it.

- **Missing negative results** (tasks or settings where the method does not work) suggest the authors are presenting an incomplete picture. Every method has failure modes. Papers that do not discuss them are less trustworthy than papers that do.

---

## 6. Building Research Taste

Research taste — the ability to predict which problems and solutions will matter — is not a mystical gift. It develops from pattern recognition across many papers. Here are the patterns that distinguish lasting contributions from flash-in-the-pan results.

### Pattern 1: Solutions That Match the Real Bottleneck

The most impactful papers correctly identify the bottleneck before proposing a solution. Flash Attention identified the HBM-SRAM bandwidth bottleneck. MQA identified the memory-bandwidth bottleneck in autoregressive decoding. Papers that solve the wrong bottleneck — like approximate attention methods that reduced FLOPs but not memory traffic — are technically correct but practically irrelevant.

**How to apply this:** When reading a new paper, ask: "Is the bottleneck they claim to address actually the bottleneck in practice?" If a paper claims to speed up training but the real bottleneck is data loading, the contribution is misdirected.

### Pattern 2: Elegance as a Proxy for Correctness

Elegant solutions tend to last longer than brute-force ones. "Elegant" here means: the solution has few moving parts, each part has a clear function, and the solution naturally extends to adjacent problems.

**Elegant:** Flash Attention's tiling + online softmax. Two ideas, each necessary, together sufficient. The approach naturally extends to block-sparse patterns, causal masking, and different hardware.

**Elegant:** Mamba's selective SSMs. One change (making parameters input-dependent) solves the content-based reasoning problem that limited all prior SSMs. The change has a clear information-theoretic interpretation.

**Brute-force:** Multi-objective auxiliary losses for MoE load balancing. DeepSeek-V2 uses three separate auxiliary losses (expert-level, device-level, communication-level) with separately tuned coefficients. It works, but the number of knobs suggests the solution is brittle. DeepSeek-V3's auxiliary-loss-free approach (bias terms) is more elegant because it eliminates the hyperparameters entirely.

### Pattern 3: Problems That Will Get Worse

The most important research targets are problems that scale *worse* than the systems they constrain. KV-cache memory was a manageable problem at 7B parameters with 2K context. At 100B+ parameters with 128K context, it became the dominant serving constraint. Papers that address problems on this trajectory — problems that were minor annoyances last year and will be critical blockers next year — have outsized impact.

Current problems on this trajectory (as of 2026):
- **Inference cost for reasoning models:** Chain-of-thought generation produces 10-100x more tokens per query. Serving cost scales linearly with output length.
- **Expert communication in large MoE:** All-to-all communication for token routing grows with expert count and cluster size.
- **Training data quality at scale:** As pre-training corpora exceed 15T tokens, data quality becomes the bottleneck for further improvement.

### Pattern 4: The Implementation-Adoption Gap

A paper can be architecturally brilliant and practically irrelevant if it cannot be implemented efficiently in existing serving stacks. Raschka ([[raschka-llm-architecture-comparison|blog]]) documents this pattern repeatedly: GQA persists as the dominant attention variant not because it is the best architecture, but because it is the best-supported architecture in vLLM, TensorRT-LLM, and llama.cpp. MiniMax-M2 abandoned linear attention — a theoretically superior approach — because "linear attention degraded multi-turn and reasoning performance."

**How to apply this:** When evaluating a new architectural proposal, ask: "Can this be served efficiently with existing infrastructure?" If the answer is "only with custom CUDA kernels that do not exist yet," the adoption timeline extends by 1-2 years regardless of the architecture's quality.

### What Research Taste Is Not

Research taste is not about following trends. It is not about predicting which papers will get the most citations. And it is definitely not about dismissing work that is "too simple" or "too incremental."

Flash Attention's core idea — tile the computation for SRAM — is simple enough to explain in one sentence. GQA's core idea — share KV heads among groups of query heads — is a one-line code change. Simplicity is a feature, not a deficiency. The papers that change the field are usually the ones where, in retrospect, the idea seems obvious. That feeling of obviousness is the hallmark of a well-identified bottleneck and a well-matched solution.

---

## 7. Putting It All Together: A Worked Paper Critique

Let us apply everything from this chapter to a single paper: the Mamba paper ([[mamba|paper]]). This is a paper you have studied in [[ch-15]] and [[ch-22]]; now we critique it as a *paper*, not as a technique.

### Pass 1 (5 minutes): Skeleton

**Abstract analysis:** The abstract identifies the bottleneck (subquadratic architectures lack content-based reasoning), names the fix (selective state spaces — input-dependent parameters), and reports the result (5x throughput, matches Transformers 2x its size). Clean structure.

**Figures:** The architecture diagram shows the Mamba block replacing both attention and MLP. The selection mechanism diagram shows how B, C, and delta become functions of the input. The scaling plots show Mamba-3B matching Transformer-6B.

**Assessment after Pass 1:** Clear bottleneck identification (Category 1), novel structural solution (Category 2). Worth proceeding to Pass 2.

### Pass 2 (20 minutes): Ablations

**What is ablated:** The selection mechanism itself — comparing selective SSM against fixed (non-selective) SSM and against Transformer attention on synthetic tasks (Selective Copy, Induction Heads) and real language modeling.

**Are the baselines fair?** The Transformer baselines use standard configurations. The fixed-SSM baseline (S4) is the direct predecessor. These are the right baselines.

**Key finding from ablations:** Fixed SSMs completely fail on Selective Copy (random token positions), while selective SSMs match Transformers. This is a clean demonstration that input-dependence is the critical ingredient.

**What is NOT ablated:** The hardware-aware scan algorithm. The paper claims this is necessary for efficiency but does not ablate it against a naive implementation. This is defensible — the scan algorithm is an implementation choice, not an architecture choice — but it means the speed claims are tied to specific GPU hardware.

### Pass 3 (45 minutes): Details and Gaps

**Reproducibility:** The paper provides the selective SSM equations, the Mamba block architecture, and training hyperparameters. The hardware-aware scan algorithm is described in pseudocode. Implementation requires custom CUDA kernels (similar to Flash Attention). Reproducibility rating: reproducible with effort.

**Gaps identified:**

1. **Scale limitation:** All experiments are at 3B parameters or below. The paper does not demonstrate Mamba at 70B+, leaving open the question of whether selective SSMs scale as well as attention at frontier scale. As of 2026, this gap remains partially unresolved — large-scale pure-SSM models are rare compared to Transformer and hybrid models.

2. **Retrieval weakness acknowledged but underexplored:** The paper notes that SSMs cannot easily retrieve from arbitrary positions, but the ablation focuses on synthetic tasks where this limitation is minimal. Real-world tasks (long-context QA, needle-in-a-haystack) would be more demanding tests.

3. **No comparison with Flash Attention:** Mamba's speed comparison is against standard attention, not Flash Attention. Since Flash Attention is ubiquitous, the real-world speedup over a Flash Attention baseline would be smaller than the reported 5x.

**Overall assessment:** Strong paper. Category 1 (bottleneck identification: content-based reasoning in SSMs) and Category 2 (novel structural solution: selective parameters). The main limitation is scale validation, which is understandable given compute constraints but leaves the most important question unanswered.

---

## Core Insights from the Literature

### Insight 1: Bottleneck identification is the scarce skill, not solution design
**Source:** Cross-cutting pattern across [[flash-attention|paper]], [[mqa|paper]], [[mamba|paper]]

The three most impactful architecture papers of 2019-2023 share a common structure: they correctly identified a bottleneck that the community had either missed or mischaracterized (MQA: decoding is bandwidth-bound; Flash Attention: attention is IO-bound; Mamba: SSMs lack content-based reasoning). In each case, once the bottleneck was correctly identified, the solution followed relatively naturally. The community had spent years attacking the wrong targets (reducing FLOPs for attention, adding complexity to SSMs). **Guideline:** When reading a paper, evaluate the bottleneck analysis more carefully than the proposed solution. A correct bottleneck identification with a mediocre solution is more valuable than a brilliant solution to the wrong problem.

### Insight 2: Ablation tables are the paper's real argument
**Source:** Ablation methodology across [[gqa|paper]], [[deepseek-v2|report]], [[mamba|paper]]

The narrative text of a paper tells you what the authors *believe*; the ablation table tells you what the *evidence supports*. In architecture papers, the ablation table is frequently more informative than the main results table. Main results show the final system (which includes many simultaneous changes); ablations isolate individual components. A paper with strong ablations and mediocre main results is more trustworthy than a paper with strong main results and no ablations. **Guideline:** Read ablation tables before main results tables. Check for single-variable changes, meaningful baselines, appropriate metrics, and reported variance.

### Insight 3: Implementation feasibility determines real-world impact more than architectural quality
**Source:** [[raschka-llm-architecture-comparison|blog]], [[raschka-attention-variants|blog]], MiniMax-M2 case

GQA dominates production LLMs despite being architecturally inferior to MLA at scale. Linear attention was abandoned by MiniMax-M2 despite theoretical superiority. The pattern is consistent: the architecture that can be served efficiently by existing infrastructure wins over the architecture with better theoretical properties. This is not a failure of the community — it reflects the reality that inference cost dominates the total cost of ownership for any deployed model. **Guideline:** When evaluating a new architecture paper, assign as much weight to "Can this be served?" as to "Does this improve quality?" A 2% quality improvement that requires custom kernels, specialized hardware, or 3x the serving cost is unlikely to be adopted.

### Insight 4: The best papers make you feel the idea was obvious in retrospect
**Source:** Cross-cutting pattern across landmark architecture papers

Flash Attention's tiling, GQA's head grouping, Mamba's input-dependent parameters — each of these ideas is explainable in one sentence and feels obvious once stated. This is not because they were easy to discover. It is because they correctly matched a solution to a bottleneck with minimal unnecessary complexity. Papers that require elaborate justification for why their solution works are often solving the wrong problem or adding unnecessary machinery. **Guideline:** Be suspicious of papers whose methods require more than two paragraphs to explain the core idea. Complexity in the explanation usually reflects complexity in the method, and complexity in the method usually reflects a mismatch between the problem and the solution.

---

## Key Takeaways

1. **Read papers in three passes:** skeleton (5 min), ablations (15-20 min), details (30-60 min). Most papers do not survive to pass 3. The three-pass method saves time and concentrates attention on papers that matter.

2. **Mental ablation is the core reasoning skill.** For every design choice, ask "what would happen without this?" The choices that survive mental ablation — where removal causes clear failure — are the real contributions. The rest is engineering.

3. **Contributions exist on a spectrum.** Bottleneck identification (rarest, most valuable) > novel structural solution > engineering optimization > empirical validation at scale. Knowing where a paper falls on this spectrum tells you how much attention it deserves.

4. **Ablation table literacy is a distinct skill.** Check for single-variable changes, meaningful baselines, matched compute budgets, appropriate metrics, and reported variance. A 0.3% improvement with no error bars is noise, not signal.

5. **Reproducibility is a quality signal.** Papers that provide enough detail to implement from scratch (shapes, initialization, hyperparameters) are more trustworthy than papers that omit these details. Missing details often signal sensitivity or fragility.

6. **Research taste develops from pattern recognition.** The patterns that predict impact: correct bottleneck identification, elegant (few-part) solutions, problems that get worse with scale, and feasibility within existing serving infrastructure.

7. **Simplicity is a feature.** The most impactful architecture papers have ideas that can be explained in one sentence. If the core idea requires elaborate justification, either the problem is wrong or the solution has unnecessary parts.

---

## References

- [[flash-attention|Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022) (paper)]]
- [[flash-attention-2|Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023) (paper)]]
- [[gqa|Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023) (paper)]]
- [[mqa|Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) (paper)]]
- [[deepseek-v2|DeepSeek AI, "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" (2024) (report)]]
- [[deepseek-v3|DeepSeek AI, "DeepSeek-V3 Technical Report" (2024) (report)]]
- [[mamba|Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) (paper)]]
- [[llama-2|Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023) (report)]]
- [[raschka-llm-architecture-comparison|Raschka, "The Big LLM Architecture Comparison" (2026) (blog)]]
- [[raschka-attention-variants|Raschka, "A Visual Guide to Attention Variants in Modern LLMs" (2026) (blog)]]
- [[raschka-understanding-llms|Raschka, "Understanding Large Language Models: A Cross-Section of the Most Relevant Literature" (2023) (blog)]]
