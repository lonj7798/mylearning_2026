# Reproducibility Red Flags in Architecture Papers

<!-- scope: patterns that signal reproducibility issues
     parent: [[ch-28]]
-->

## Purpose

Section 5 of [[ch-28]] introduces the reproducibility assessment. This excerpt catalogs specific red flags — patterns in how papers present (or omit) details that predict reproducibility difficulty. These are drawn from patterns observed across the source library for this course.

---

## Red Flag 1: "Standard" Without Specification

**The pattern:** The paper describes a component as "standard" or "following prior work" without specifying the exact configuration.

**Examples of vague language:**
- "We use standard layer normalization" (Pre-norm or post-norm? LayerNorm or RMSNorm? What epsilon?)
- "We follow the training recipe of [citation]" (Which hyperparameters were actually used? Did you change any?)
- "Standard initialization" (Xavier? Kaiming? What fan mode? What gain?)

**Why this is a red flag:** "Standard" means different things to different teams. Pre-norm was the GPT-2 standard. Post-norm was the original Transformer standard. OLMo 2 uses post-norm with QK-norm. Gemma 3 uses hybrid pre+post-norm. Each of these could be described as "standard" by their respective authors.

**What to do:** Search the paper's appendix and code release for the exact specification. If it is not there, assume you will need to experiment with multiple configurations to reproduce results.

---

## Red Flag 2: Missing Initialization Details

**The pattern:** The paper describes the model architecture but does not specify how weights are initialized, or uses a single sentence like "we initialize weights using Xavier uniform."

**Why this matters:** Initialization interacts with normalization placement, learning rate, and model depth. For deep Transformers (60+ layers), incorrect initialization can cause:
- Training instability (gradient explosion in post-norm models)
- Slow convergence (vanishing updates in pre-norm models with wrong scale)
- Representation collapse (all layers converging to similar representations)

**Case study — DeepSeek-V2's MLA initialization:** MLA introduces new parameter matrices (down-projection $W_{DKV}$, up-projections $W_{UK}$, $W_{UV}$). The initialization of these matrices determines whether the low-rank bottleneck starts in a useful part of the space or in a dead zone. The report specifies some initialization details but the interaction between MLA initialization and the model's training dynamics is underspecified.

**Case study — Mamba's selective SSM initialization:** Mamba inherits S4's HiPPO initialization for the state matrix A, but the initialization of the input-dependent projections ($s_B$, $s_C$, $s_\Delta$) is less thoroughly discussed. These projections determine the selection mechanism at training start, and wrong initialization could prevent the model from learning to select effectively.

---

## Red Flag 3: Single-Seed Results

**The pattern:** The paper reports a single training run with no error bars, confidence intervals, or discussion of variance across random seeds.

**Why this matters:** At small scale (< 1B parameters), different random seeds can produce perplexity differences of 0.5-2.0 points. At large scale (> 10B), seed variance decreases but does not vanish. A claimed improvement of 0.3% over a baseline could easily be within seed variance.

**What to look for:**
- Tables with no +/- values
- Figures with no error bands
- Phrases like "we report the best run" (which confirms cherry-picking)

**Acceptable exceptions:** Pre-training runs at 100B+ parameters are too expensive to repeat with multiple seeds. In this case, the paper should acknowledge that results are from a single run and discuss why they believe the effect is larger than seed variance (e.g., consistent improvements across many benchmarks, effect visible throughout training).

---

## Red Flag 4: Benchmark Selection Bias

**The pattern:** The paper reports results on an unusual selection of benchmarks, or reports many benchmarks but only highlights the ones where the method improves.

**Signals to watch for:**
- Benchmarks that are not standard for the method's domain (e.g., an attention paper that reports only on code generation but not on standard language modeling)
- Many benchmarks listed but improvements highlighted selectively in the abstract
- Benchmarks where the proposed method has a natural advantage (e.g., testing a long-context method only on tasks that require long context)

**What a fair benchmark selection looks like:**
- Standard benchmarks for the domain (MMLU, BBH, HumanEval for general LLMs)
- At least one benchmark where the method is NOT expected to help (shows honesty)
- Benchmarks that stress-test the method's claimed advantage (e.g., long-context benchmarks for an attention efficiency method)

---

## Red Flag 5: Mismatched Compute Budgets

**The pattern:** The proposed method and baselines are trained with different compute budgets, but this difference is not prominently discussed.

**Common forms:**
- The proposed method trains for more steps than the baseline
- The proposed method uses a larger learning rate schedule (more warmup, different decay)
- The proposed method benefits from hyperparameter tuning that the baselines did not receive
- The proposed method uses a different data mixture that was optimized for the target benchmarks

**Case study — GQA uptraining:** The GQA paper is transparent about compute budgets: uptraining uses 5% of original pre-training compute. The ablation compares different attention configurations all uptrained with the same 5% budget. This is a fair comparison. If one configuration had received 10% budget while others received 5%, the comparison would be confounded.

**How to check:** Look for total training tokens, total GPU-hours, or total FLOPs for each row in the ablation table. If these numbers differ, the ablation has a compute confound. If they are not reported per-row, assume they might differ.

---

## Red Flag 6: No Failure Analysis

**The pattern:** The paper reports only successes. There is no discussion of settings, tasks, or configurations where the method fails or underperforms.

**Why this is a red flag:** Every method has failure modes. A paper that does not discuss them is either incomplete (the authors did not investigate) or misleading (the authors investigated and did not like the results). Both are concerning.

**What honest failure analysis looks like:**

- **Flash Attention:** The paper notes that at short sequence lengths (512), the speedup is only 15% — modest. It does not claim Flash Attention revolutionizes short-sequence workloads.
- **Mamba:** The paper acknowledges that SSMs cannot easily retrieve from arbitrary positions, which limits certain in-context learning capabilities. The Selective Copy ablation specifically tests this weakness.
- **GQA:** The paper shows that MQA (G=1) degrades quality on hard tasks, motivating the intermediate GQA configuration. The quality-speed tradeoff curve is presented honestly.

---

## Red Flag 7: Code Release Without Documentation

**The pattern:** The paper links to a code repository, but the code lacks documentation, tests, or clear reproduction instructions.

**Why this matters:** An undocumented code release creates an illusion of reproducibility. Researchers who try to reproduce will spend weeks reading through code, encountering undocumented hyperparameters, and making assumptions that may not match the paper's actual setup.

**The reproducibility spectrum (from [[ch-28]] Section 5):**

| Level | Example | Effort to Reproduce |
|-------|---------|-------------------|
| Gold | OLMo 2: weights, data, code, checkpoints, logs | Days |
| Silver | Flash Attention: pseudocode + documented CUDA code | Weeks |
| Bronze | Most papers: architectural description + code link | Weeks to months |
| Iron | Code link with no docs, or no code at all | Months, if possible |

**Aspiration for your own work:** Target Silver or Gold. If you cannot release data, at least release model weights, training code with clear hyperparameters, and a reproduction guide. The OLMo project ([[olmo-2|report]]) demonstrates that full reproducibility is feasible even for large-scale models.

---

## Applying These Red Flags

When you evaluate a paper, scan for these flags during Pass 2 (ablation read). Count the flags:

- **0-1 flags:** Trustworthy paper. Proceed to Pass 3 with confidence.
- **2-3 flags:** Credible but verify. Note the specific gaps and check whether the code release fills them.
- **4+ flags:** High skepticism warranted. The claims may not reproduce. Treat the paper as suggestive, not definitive.

No paper is perfect. Even landmark papers have red flags — the Transformer paper itself has no ablation of positional encoding type, and GPT-3 has no error bars. The goal is calibrated trust, not binary accept/reject.
