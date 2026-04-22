<!-- scope: reasoning-trace synthesis — Xwin-Math recipe with scaled question generation + SFT
     deps: [[metamath]]
     see-also: [[openmathinstruct-2]], [[mathscale]]
-->

# Xwin-Math: Common 7B Language Models Already Possess Strong Math Capabilities
- **Core Insight:** Standard 7B base models already latently possess strong math capabilities; what they need is a large, diverse supervised set of synthetic math problems, and scaling synthetic data from a strong teacher (GPT-4) to 1M+ examples steadily improves math benchmark accuracy with no diminishing returns visible at that scale.
- **Guideline:** For math SFT, don't stop at 50K–200K examples; scale GPT-4-generated question/solution pairs to at least 1M and retrain from scratch — the log-linear scaling continues.
- **Authors:** Chen Li, Weiqi Wang, Jingcheng Hu, Yixuan Zhang, Nanning Zheng, Han Hu, Zheng Zhang, Houwen Peng (Microsoft Research Asia / Xi'an Jiaotong U)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2403.04706
- **Relevant topics:** math reasoning scaling, synthetic question generation, SFT scaling

## Abstract
Xwin-Math challenges the belief that 7B base models cap out at MATH ~30–40%. Using GPT-4-Turbo as the teacher, the authors generate 1M new math problems and 7.5M corresponding CoT solutions, and fine-tune LLaMA-2-7B to reach 51.9% MATH and 84.1% GSM8K — exceeding many 70B models at the time. Scaling ablations show the log-linear relationship between synthetic data volume and benchmark accuracy continues beyond prior stopping points.

## Key Contributions
- **Xwin-Math-1M** synthetic dataset (publicly released, partial).
- Empirical scaling curve: accuracy on MATH ≈ a + b·log(N) over N = 7.5K → 1M problems.
- Xwin-Math-7B-v1.1 and Xwin-Math-70B-v1.1 model releases.
- Demonstrated generalization to out-of-distribution benchmarks (Hungarian Exam, CMATH).

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** GSM8K train (7.5K) + MATH train (7.5K).
- **Question augmentation:** GPT-4-Turbo prompted to generate a new problem "in the style of" a seed, with controlled difficulty tags. Per seed, 40–100 new questions are drawn.
- **Solution generation:** for each augmented problem, GPT-4-Turbo produces K = 5–10 CoT solutions.
- **Filtering:**
  - Gold-answer exact-match (numeric) or symbolic-equivalence (MATH-style).
  - Accept only solutions where a majority of GPT-4's K samples agree on the same final answer (pseudo-gold by self-consistency).
  - Deduplicate via MinHash on problem text.
- **Output shape:** ~1M problems × ~7 accepted solutions each → 7.5M total training pairs. Trace length 300–800 tokens.
- **Teacher model:** GPT-4-Turbo (2024-era).
- **Cost / compute:** not precisely disclosed; ~$150K–$250K in GPT-4-Turbo API (community estimate).

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** short CoT, ~500 tokens median.
- **Trace style:** standard CoT (not long-CoT). Final answer in `\boxed{}`.
- **Correctness verifier:** SymPy equivalence + self-consistency vote across GPT-4's K samples as a pseudo-gold signal when original gold is absent.
- **Difficulty control:** GPT-4 prompt includes difficulty tags; authors oversample "hard" to balance the long tail.
- **Error-mode analysis:** ~10% of accepted solutions have correct final answer but flawed reasoning; authors accept this noise as tolerable.

## Quality / diversity evaluation
- Xwin-Math-7B-v1.1: **MATH 51.9, GSM8K 84.1** — SOTA open 7B at release.
- Xwin-Math-70B-v1.1: MATH 57.0, GSM8K 90.6.
- Log-linear scaling: MATH gain ~+4 per doubling of synthetic data, with no saturation at 1M.
- Out-of-distribution: Hungarian National Finals Math exam — Xwin-Math-7B 48% vs GPT-3.5 43%.

## Risks + gotchas
- **Teacher cost and licensing:** 1M GPT-4-Turbo-generated problems carry OpenAI ToS restrictions.
- **Contamination:** GPT-4 may have seen MATH/GSM8K during pretraining; generated "new" problems may leak test-set patterns.
- **Scaling law may not extrapolate beyond 1M** (authors note diminishing but still positive returns at 10M in follow-up ablations).
- **Short-CoT ceiling:** dataset is short CoT; students cannot learn reflective long-CoT from it.

## Connections
- Scaling sibling: [[openmathinstruct-2]] (Llama-405B teacher, 14M examples).
- Question-augmentation ancestor: [[metamath]], [[mathscale]].
- Contrasts small-curated: [[s1]], [[limo]].
