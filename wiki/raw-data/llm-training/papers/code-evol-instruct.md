<!-- scope: code-specific Evol-Instruct operator reference (subset of WizardCoder)
     deps: [[wizardcoder]], [[evol-instruct]]
     see-also: [[oss-instruct]], [[opc-synthetic-code]]
-->

# Code Evol-Instruct: Code-Specialized Evolution Operators
- **Core Insight:** Generic Evol-Instruct operators (add constraints, deepen reasoning, increase breadth) don't exploit code's unique structure; code-specific operators targeting time/space complexity, error-handling robustness, specific language/library requirements, and misleading test cases produce more useful coding instruction data per seed.
- **Guideline:** When evolving code instructions, run M=3 iterations per seed with randomized code-specific operators; post-prune identical / too-easy / teacher-refused samples; aim for ~4× expansion of the seed pool.
- **Authors:** Ziyang Luo et al. (see [[wizardcoder]])
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2306.08568 ; https://github.com/nlpxucan/evol-instruct
- **Relevant topics:** Code Evol-Instruct, operator catalog, code SFT data

## Scope
This page is the **operator reference** for Code Evol-Instruct. The parent paper and headline results live at [[wizardcoder]]. Here we enumerate and explain the operators — load-bearing when designing your own code-synthesis pipeline.

## The operator set

### Complexity axis
- **Increase time complexity requirement.** Add a target complexity bound ("the solution must run in O(n log n)").
- **Increase space complexity requirement.** "The algorithm must use O(1) auxiliary space."
- **Add edge cases.** "Your code should handle empty input and negative numbers."
- **Add stricter input validation.** Handle malformed / adversarial inputs explicitly.

### Scope axis
- **Add new constraints or requirements.** Extra functional requirement on output.
- **Replace a common requirement with a less common one.** e.g., "use `collections.deque` instead of a list."
- **Require a specific language / library.** e.g., "port to Rust," "use numpy vectorization only."

### Reasoning depth axis
- **Increase depth of reasoning steps.** Require the solution be a composition of sub-algorithms.
- **Deepen problem complexity.** Multi-step algorithmic transformation (e.g., first parse, then compute, then format).

### Robustness axis
- **Introduce misleading requirements.** Phrasing that suggests a naive approach; correct solution requires noticing the trap.
- **Insert erroneous code** in the prompt and ask the model to identify + fix it. (This operator is highlighted in the Code Evol-Instruct literature as particularly valuable for robustness.)

### Domain axis (later extensions)
- **Data-science-specific:** pandas/numpy/scikit-learn idioms.
- **Systems-level:** concurrency, memory management.
- **Multi-language port:** translate between Python / C++ / Rust / Go / JavaScript.

## Pipeline
1. For each seed coding instruction, randomly pick one operator.
2. Teacher (GPT-3.5/4) rewrites the instruction and generates a reference solution.
3. Iterate M=3 times per seed, each time on the previous evolved version.
4. **Pruning:**
   - Drop identical evolutions.
   - Drop teacher-refused evolutions ("Sorry, I cannot…").
   - Drop length-degenerate cases.
   - Drop benchmark-overlap cases (HumanEval / MBPP n-gram match).
5. **Post-processing:**
   - Normalize format to `<instruction, response>` pairs.
   - Extract code blocks reliably.

## Sample counts
- Seed: 20K Code-Alpaca instructions.
- Output: ~78K evolved pairs after M=3 rounds + pruning.

## Variants / descendants
- **Magicoder Evol-Instruct 110K** — public reproduction, used in [[opc-synthetic-code]] stage 2.
- **Genetic Instruct** (2024) — scales up with a genetic-algorithm-style population + fitness function (as used in [[nemotron-4-synthetic]] for code).
- **Evolutionary Contrastive Distillation** (2024) — blends operators with contrastive prompting.

## Risks + gotchas
- **Operator bias:** overuse of the "add constraints" operator narrows toward over-specified problems; balance operators.
- **Teacher refusal drift:** GPT-4 increasingly refuses to generate "misleading" prompts — later runs lose that operator if not carefully wrapped.
- **Benchmark contamination:** the "require specific library" operator can accidentally recreate benchmark problems; always decontaminate.

## Connections
- Parent: [[wizardcoder]].
- Generic counterpart: [[evol-instruct]].
- Used in production SFT mixes: [[opc-synthetic-code]], [[smol-talk]] (via Self-OSS-Starcoder2-Instruct).
- Sibling synthesis axis: [[oss-instruct]] (seed from real code snippets rather than evolving existing instructions).
