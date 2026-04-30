<!-- scope: reasoning-trace synthesis — 1.8M math CoT traces from Mixtral on open problems
     deps: [[self-instruct]]
     see-also: [[openmathinstruct-2]], [[mammoth]], [[metamath]]
-->

# OpenMathInstruct-1
- **Core Insight:** Large-scale open math SFT is feasible with a permissively-licensed teacher (Mixtral 8x7B) by generating multiple CoT + Python-tool-integrated solutions per problem and filtering by exact-match correctness.
- **Guideline:** For open math SFT, pair CoT with code-executed solutions (Program-of-Thought / PoT), sample K=32–64 solutions per problem, and keep only those whose executed answer equals the gold; this converts weak teacher models into large, license-clean training sets.
- **Authors:** Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, Igor Gitman (Nvidia)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.10176
- **Relevant topics:** math reasoning, synthetic CoT, tool-integrated reasoning (TIR), permissive licensing

## Abstract
OpenMathInstruct-1 releases 1.8M math problem-solution pairs built entirely with the Mixtral 8x7B Apache-2.0 model, covering GSM8K and MATH. Each solution mixes natural-language reasoning with Python code blocks whose outputs are executed and spliced back; the answer is checked against gold. OpenMath-7B/13B/70B models fine-tuned on the dataset match or beat prior closed-teacher distillations (GPT-4 derived) on GSM8K and MATH.

## Key Contributions
- 1.8M permissively-licensed math SFT examples.
- Tool-integrated reasoning template: alternating text-reasoning and `<llm-code> … </llm-code>` Python segments, with executor output `<llm-code-output>` spliced in.
- Open model family (OpenMath-Mistral-7B, OpenMath-CodeLlama-34B, OpenMath-Llama-70B).
- Proof that Mixtral, despite being weaker than GPT-4, can bootstrap a dataset competitive with GPT-4–derived ones by high sampling budget + executor-grounded filtering.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** GSM8K train (7.5K problems) + MATH train (7.5K problems) = ~15K source questions.
- **Generation:** for each problem, sample K solutions from Mixtral 8x7B-Instruct. Prompt template asks for text reasoning interleaved with Python code; prompt includes 2–5 in-context example traces showing the `<llm-code>` / `<llm-code-output>` pattern.
- **Execution loop:** while the generated solution contains an `<llm-code>` block, extract the Python, execute in a sandbox, splice the output back as `<llm-code-output>`, and re-prompt the model to continue from there.
- **Filtering:**
  - Extract final numeric/formula answer; compare to gold via SymPy canonicalization for MATH, string-match for GSM8K.
  - Keep only solutions whose final answer is correct (exact match).
- **Output shape:** 1.8M solutions across both sources — on average ~120 solutions per GSM8K problem and ~100 per MATH problem; trace length typically 200–1500 tokens with 1–4 code blocks.
- **Teacher model:** Mixtral-8x7B-Instruct-v0.1.
- **Cost / compute:** authors report ~500 K GPU-hours of teacher sampling, run on Nvidia DGX clusters.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** avg ~500 tokens; tail to ~2K. Shorter than long-CoT (o1/R1 style) — these are "working-math" traces, not reflective traces.
- **Trace style:** tool-integrated reasoning (TIR) — CoT interleaved with executable Python. Hybrid of CoT and PoT.
- **Correctness verifier:** execution + exact-match on final answer. For MATH, SymPy-based algebraic equivalence; for GSM8K, numeric match.
- **Error-mode filter:** only final-answer correctness; intermediate-step errors slip through when they cancel out — a known limitation.

## Quality / diversity evaluation
- OpenMath-Mistral-7B: **80.2 GSM8K, 44.5 MATH** — competitive with GPT-4-distilled 7B models.
- OpenMath-Llama2-70B: **84.6 GSM8K, 50.7 MATH**.
- Ablation: dropping the code-execution pathway (CoT-only) loses ~8 points on MATH; dropping natural language (PoT-only) loses ~5 on GSM8K.

## Risks + gotchas
- **False positives (right answer, wrong reasoning):** ~5–10% of kept traces have a correct answer via flawed intermediate steps — this noise compounds when the model learns spurious shortcuts.
- **Mixtral teacher ceiling:** dataset cannot cover problems Mixtral cannot solve in K samples; the difficulty distribution is capped by the teacher.
- **Benchmark overlap:** source = GSM8K/MATH train set only, so downstream eval on those test sets is legit but transfer is narrow.

## Connections
- Direct successor: [[openmathinstruct-2]] (Llama-3.1 405B teacher, 14M examples).
- Lineage: [[training-verifiers-to-solve-math-word-problems]] (GSM8K + verifier), [[mammoth]] (CoT+PoT hybrid).
- Related filtering: rejection-sampling fine-tuning (RSFT) uses the same basic idea of keeping model samples that pass a correctness filter.
- Contrast curated small-N: [[s1]], [[limo]].
