---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/wizardcoder.md
source_url: https://arxiv.org/abs/2306.08568
created_at: "2026-04-23"
---

# Excerpt: WizardCoder — Code-Specialized Evol-Instruct Operators

**Source library:** `wiki/raw-data/llm-training/papers/wizardcoder.md`
**Heritage:** Luo et al. 2023 (Microsoft) — ICLR 2024. The first explicit port of [[excerpts/evol-instruct]]'s operator framework to code. Sibling to [[wizardmath]] (math specialization, same year, same group).

---

## Why this source anchors ch-19

Ch-19 §4 makes the structural claim that operators must be domain-specific. WizardCoder is the cleanest demonstration: running the generic six Evol-Instruct operators on code wastes mass on operators that don't touch code's failure surface (edge cases, complexity bounds, library idioms). The five code-specific operators the paper introduces produce a 20K → 78K expansion that lifts an open 15B model past Claude and Bard on HumanEval at release. The operator design, not the scale, is what the chapter cites.

---

## The five code-specific operators — and what each targets

From the source file:

> 1. **Add new constraints / requirements** (e.g., "the function must also handle negative integers").
> 2. **Replace a common requirement with a less common one** (e.g., use `deque` instead of `list`).
> 3. **Increase depth / reasoning steps** (multi-step algorithmic transformation).
> 4. **Deepen problem complexity** (higher time/space constraint, more edge cases, misleading wordings).
> 5. **Require specific language or library** (e.g., port to Rust, use NumPy vectorization).

Mapping each to a code-specific failure mode:

- **Operator 1 (add constraints).** The generic version targets under-specification; the code version targets *input-space coverage*. Code that works on positive integers fails on empty lists, negative numbers, and overflow. Each constraint added forces the model to generate a solution that respects one more subset of the input space.
- **Operator 2 (replace common with less common).** The generic version targets lexical diversity; the code version targets *library-idiom coverage*. A model trained only on `list` methods won't know `collections.deque.popleft()` is O(1). Forcing library substitution exposes the model to the idiom surface it would otherwise miss.
- **Operator 3 (increase reasoning steps).** Same as generic — but applied to code, it produces problems like "first parse the input, then compute, then format" which map to real software structure. Pure-single-step problems over-represent toy exercises.
- **Operator 4 (deepen complexity).** The code-specific value is edge-case enumeration. "Misleading wordings" teaches the model to read carefully rather than pattern-match the first noun to a library function.
- **Operator 5 (require specific language/library).** This is the biggest single diversity injector. The raw 20K Code-Alpaca seeds are ~95% Python. Requiring Rust, C++, JavaScript, NumPy-vectorized, or pandas-native variants multiplies the topical coverage without increasing seed count.

The generic Evol-Instruct "complicate input" operator is replaced — in code, complicating the input often produces malformed programs that the teacher rejects, so the slot is repurposed for the language/library axis.

---

## The 20K → 78K yield and M=3 rounds

The source:

> **Seed input:** Code-Alpaca 20K seeds (itself Self-Instruct-generated on coding tasks).
>
> **Output shape:** ~78K evolved `<instruction, response>` coding pairs released under the WizardLM family license.

Three rounds of evolution per seed, randomized operator per round. Yield: ~3.9× expansion. Not 5× because the elimination filter rejects ~20% of evolutions per round (duplicates, teacher refusals, length-degenerate outputs).

The M=3 is load-bearing. M=1 doesn't produce enough complexity; M=5 hits the same ceiling Evol-Instruct hit at four rounds — the teacher refuses to evolve beyond its own coding competence. M=3 is the empirical sweet spot where yield per round is still >70% and the complexity histogram is long-tailed.

---

## The filter stack — with code-specific additions

The source:

> - Drop evolutions where teacher refuses or the new instruction is identical to the old.
> - Length-based sanity filter.
> - Deduplicate via exact/near-exact match.
> - Decontamination vs HumanEval / MBPP.

The first three mirror Evol-Instruct. The fourth is code-specific and important: **decontamination against HumanEval and MBPP**. Because the evolved instructions frequently touch canonical problems (fibonacci, two-sum, valid-parentheses), n-gram-level overlap with benchmark test cases is a real risk. The paper's decontamination is string-match on canonicalized function signatures; later audits found residual stylistic overlap that stricter semantic decontamination would have caught.

This is a recurring finding worth stating plainly: every synthetic-code pipeline leaks benchmark content unless explicitly filtered, and the first round of filters is always too loose.

---

## HumanEval 57.3 at 15B — the headline result and its context

The source:

> - WizardCoder-15B: HumanEval 57.3 / HumanEval+ 50.6 / MBPP 51.8 (beats Claude, Bard).

At release (June 2023) this was SOTA for open code models. The base model is StarCoder-15B, which on its own scores ~33% on HumanEval. The SFT-only Code-Alpaca baseline scored ~44%. WizardCoder-15B at 57% captures a +13-point lift from code-specific evolution alone over the same base.

Context matters for the comparison. HumanEval at the time was a 164-problem benchmark with known leakage issues. Later benchmarks (HumanEval+, MBPP+, LiveCodeBench) are harder. WizardCoder's gap to closed models closed partly because the closed models also trained on evolved synthetic data — the technique generalized both ways.

---

## The Code-Alpaca → Magicoder-Evol-Instruct-110K lineage

The source:

> **Variants / descendants:**
> - **Magicoder Evol-Instruct 110K** — public reproduction, used in [[opc-synthetic-code]] stage 2.
> - **Genetic Instruct** (2024) — scales up with a genetic-algorithm-style population + fitness function (as used in [[nemotron-4-synthetic]] for code).
> - **Evolutionary Contrastive Distillation** (2024) — blends operators with contrastive prompting.

The 78K WizardCoder set was not publicly released in full due to license concerns. Magicoder Evol-Instruct 110K is the community reproduction on the same operator stack, publicly available. It's the dataset most open code-SFT pipelines (OpenCoder, Qwen-Coder variants) actually train on.

Genetic Instruct is the 2024 scaling: treat operators as a *population* with fitness scores, and run an evolutionary algorithm that keeps high-fitness evolutions and discards low-fitness ones. This beats pure-random operator selection but adds a fitness-evaluation dependency that most teams skip.

---

## The risks the source flags

The source:

> - **Mode collapse risk** — repeated use of the same operator on the same seed yields near-duplicate outputs; operator randomization + filtering are load-bearing.
> - **HumanEval contamination** — WizardCoder is one of the earliest cases where independent audits found stylistic overlap with HumanEval tasks; later authors recommend stricter decontamination.
> - **Python-centric** — non-Python coverage is shallower.

The Python-centricity is a soft constraint: Operator 5 (require specific language) does inject non-Python, but the seeds are Python, so non-Python evolutions are shallower translations rather than natively-Python-thinking problems. Production code pipelines that want real multilingual coverage seed from multi-language corpora directly (e.g., GitHub snippets), which is the [[oss-instruct]] complement.

---

## Connections

- [[excerpts/evol-instruct]] — the generic operator framework WizardCoder specializes.
- [[excerpts/wizardmath]] — sibling math specialization.
- [[excerpts/self-instruct]] — the Code-Alpaca seeds are Self-Instruct outputs on coding tasks.
- [[ch-19]] — this excerpt is the foundation of §4 (domain specialization) and the code-specific row in §9's comparison table.
