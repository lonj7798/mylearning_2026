<!-- scope: reasoning-trace synthesis — NuminaMath 860K competition-math CoT pipeline
     deps: [[mammoth]]
     see-also: [[openmathinstruct-2]], [[openr1]]
-->

# NuminaMath: The largest public dataset in AI4Maths with 860K math competition problems
- **Core Insight:** The AI Math Olympiad winning recipe was to assemble a curated 860K-problem math dataset (NuminaMath-CoT + NuminaMath-TIR) spanning cn_k12, olympiads, aops_forum, AMC/AIME, and to regenerate high-quality CoT and Tool-Integrated-Reasoning solutions via GPT-4o — a data-centric win over algorithmic novelty.
- **Guideline:** For competition-math SFT, combine broad problem provenance (Chinese k12 olympiad textbooks are a huge untapped source) with regenerated clean CoT + code-executed TIR solutions; verifier is `\boxed{}` extraction + SymPy equivalence.
- **Authors:** Numina team (Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, et al.)
- **Year:** 2024
- **URL:** https://huggingface.co/datasets/AI-MO/NuminaMath-CoT ; https://github.com/project-numina/aimo-progress-prize
- **Relevant topics:** competition math, CoT+TIR, AI Math Olympiad, open dataset

## Abstract
NuminaMath is the public math dataset assembled by the team that won the first AI Math Olympiad (AIMO) Progress Prize with a DeepSeek-Math-7B–based solver. The release comprises ~860K problem-solution pairs split into NuminaMath-CoT (chain-of-thought solutions) and NuminaMath-TIR (tool-integrated-reasoning solutions with Python code). The dataset underpins many 2025 math reasoners (OpenR1-Math-220k, Sky-T1, MATH-specific fine-tunes).

## Key Contributions
- **860K problem-solution pairs** from 8+ sources, publicly released (Apache-2.0).
- **Two solution formats** (CoT + TIR) enabling CoT-only vs tool-integrated training comparison.
- Provenance tagging for decontamination: each problem carries a `source` field (cn_k12, olympiads, aops_forum, etc.).
- Used by the AIMO-winning model (NuminaMath-7B-TIR).

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed problems:**
  - **cn_k12** (~470K): Chinese middle/high-school textbook and exam problems scraped from public sources.
  - **olympiads** (~150K): Olympiad archive problems.
  - **aops_forum** (~30K): Art of Problem Solving community discussions.
  - **AMC/AIME** (~4K): American competitions.
  - **MATH** (~7.5K): Hendrycks MATH training set.
  - **GSM8K** (~7K): grade-school word problems.
  - **orca-math** (~200K): Microsoft's Orca-Math corpus.
- **Solution regeneration:**
  - **CoT split:** prompt GPT-4o with the problem and 2-shot CoT examples; parse `\boxed{}`; keep if SymPy-equivalent to gold.
  - **TIR split:** prompt GPT-4o to produce a Python-interleaved solution; execute code in sandbox; keep if executed answer matches gold.
- **Filtering:**
  - `\boxed{}` must be present.
  - SymPy canonical-form equivalence on final answer.
  - MinHash dedup across splits.
- **Output shape:** ~860K total. CoT avg ~400 tokens; TIR avg ~600 tokens with 1–3 code blocks.
- **Teacher model:** GPT-4o primarily; some samples from DeepSeek-Math-7B in a bootstrapping loop.
- **Cost / compute:** ~$100K+ in GPT-4o API (estimate from community discussion).

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** CoT median ~400 tokens; TIR median ~600. Short-CoT style (not long-CoT / R1-style).
- **Trace style:** CoT uses natural language with `\boxed{}` final answer; TIR interleaves `python` code blocks whose output is spliced back and used in subsequent reasoning.
- **Correctness verifier:** SymPy algebraic equivalence — critical for accepting equivalent-but-syntactically-different forms (e.g., `x^2 - 1` vs `(x-1)(x+1)`).
- **Language:** predominantly English; cn_k12 originals translated to English for the release.
- **Why cn_k12 dominates:** Chinese math textbooks are a rich, relatively untapped source — high problem density, exam-graded difficulty, available as public PDFs.

## Quality / diversity evaluation
- NuminaMath-7B-TIR (DeepSeek-Math-7B base, trained on NuminaMath-TIR) — AIMO Progress Prize winner 2024.
- Widely used downstream: OpenR1-Math-220k sub-samples NuminaMath-CoT; Sky-T1 uses it in the math mix.
- Source-balanced eval: model trained on NuminaMath transfers well to AMC12, AIME, OMCA.

## Risks + gotchas
- **Decontamination is the user's responsibility:** problems from MATH, GSM8K, AMC archives overlap with common benchmarks — the `source` field lets users filter but they must actually do it.
- **Translation errors in cn_k12:** some Chinese-originated problems have ambiguous or slightly wrong English translations.
- **GPT-4o teacher ceiling** on hardest olympiad problems (top-10% of AIME/IMO are not reliably solved by GPT-4o).
- **TIR false positives:** programs that print right answer for wrong reason.

## Connections
- Downstream consumer: [[openr1]] (OpenR1-Math-220k is a NuminaMath subset).
- Sibling teacher-distill corpora: [[openmathinstruct-2]], [[mathscale]].
- Hybrid-format ancestor: [[mammoth]] (CoT+PoT mixture).
- Used as seed pool for [[s1]], [[limo]].
