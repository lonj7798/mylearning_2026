# Synthetic Data for Pre-training: Tradeoffs and Techniques

<!-- scope: synthetic data generation methods, hallucination risk, data repetition economics, Phi approach
     parent: [[ch-11]]
-->

## The Data Wall

The scaling-data-constrained paper ([[scaling-data-constrained|paper]]) quantified a problem that frontier labs had been privately aware of: high-quality unique text on the internet is finite and approaching exhaustion at current scaling rates.

A rough accounting of available high-quality English text:

| Source | Estimated Unique Tokens |
|--------|------------------------|
| Common Crawl (filtered) | ~3-5T |
| Books (public domain + licensed) | ~100B |
| Scientific papers (open access) | ~100B |
| Code (GitHub, deduped) | ~500B-1T |
| Wikipedia + encyclopedias | ~10B |
| **Total high-quality English** | **~5-7T** |

Llama 3's 15T-token budget already exceeds this estimate, implying significant multilingual data, data repetition, or quality threshold relaxation. The next generation of models (targeting 50-100T tokens) will face this wall head-on.

Synthetic data generation is the primary proposed escape route.

## Phi-4's 50-Pipeline Architecture

Phi-4 ([[phi-4|report]]) does not use a single synthetic data generation method — it uses 50 distinct generation approaches, each targeting different aspects of model capability. This diversity is critical: a single generation pipeline produces stylistically homogeneous output that can cause mode collapse in the student model.

### Key Generation Approaches

**Multi-agent prompting:** Multiple LLM agents (typically GPT-4 instances with different system prompts) collaborate on generating content. One agent proposes, another critiques, a third synthesizes. This produces content with more internal consistency and fewer errors than single-pass generation.

**Self-revision workflows:** An LLM generates a draft answer to a problem, then is prompted to identify weaknesses and revise. This iterative refinement process produces higher-quality outputs than single-pass generation, particularly for complex reasoning chains.

**Instruction reversal:** Given a high-quality answer or code snippet (sourced from web data), an LLM is prompted to generate the question or instruction that would produce that output. This creates well-calibrated instruction-response pairs where the response quality is guaranteed by its organic origin.

**Seed curation:** Rather than generating entirely from scratch, many pipelines start with a "seed" — a high-quality snippet from organic data — and expand or elaborate on it. This grounds the synthetic content in real-world information, reducing hallucination.

## The Student-Surpasses-Teacher Phenomenon

Phi-4's most striking result: a 14B model trained on GPT-4-generated synthetic data *outperforms GPT-4* on STEM benchmarks (GPQA: 56.1% vs 50.6%, MATH: 80.4% vs 74.6%). How is this possible?

The explanation involves three mechanisms:

1. **Concentration of signal.** GPT-4 has broad capabilities diluted across a massive parameter space. Phi-4's synthetic training data concentrates GPT-4's STEM reasoning capability into a targeted training signal, allowing the smaller model to specialize.

2. **Curriculum effect.** The synthetic data generation process implicitly creates a curriculum — starting with simpler problems and progressing to harder ones, with multiple attempts and revisions. The student sees a cleaned, organized version of the teacher's reasoning, not its raw output.

3. **Diversity across pipelines.** The 50 generation approaches produce diverse perspectives on the same concepts. Training on this diversity forces the student to learn generalizable representations rather than mimicking the teacher's specific reasoning style.

This does not violate any information-theoretic constraints: the student is not learning "more" than the teacher knows. It is learning a *focused subset* more thoroughly, while the teacher's knowledge is spread thinly across all domains.

## When Synthetic Data Fails

### Hallucination Amplification

The most dangerous failure mode. If the teacher model generates factually incorrect content with high confidence, and the student trains on it, the student inherits the errors. Worse, the student may learn that "confident assertion of uncertain facts" is a good pattern, amplifying hallucination rates beyond the teacher's.

Mitigation strategies:
- **Diversity of generation methods** (Phi-4's 50 pipelines) — errors from one pipeline are unlikely to be corroborated by another
- **Cross-referencing with organic data** — facts that appear in synthetic data but not in any organic source are flagged
- **Verification-generation decoupling** — use one model to generate and a different model (or search engine) to verify

### Mode Collapse

If all synthetic data shares stylistic patterns (e.g., always starting with "Certainly!", always using numbered lists, always qualifying statements), the student model learns to reproduce those patterns. This manifests as:
- Reduced stylistic diversity in model outputs
- Inability to match user tone or domain conventions
- Over-hedging or over-qualifying in domains where directness is preferred

### Distribution Mismatch

Synthetic data is generated from the teacher's learned distribution, which may differ systematically from the real-world distribution the student needs to handle. For example, GPT-4 may generate math problems that follow certain structural patterns (clean numbers, well-defined variables, textbook-style phrasing) while real-world math queries are often messy, ambiguous, and informally phrased.

## Data Repetition Economics

### The 4-Epoch Threshold

The scaling-data-constrained paper established empirically that up to ~4 epochs of data repetition produces loss indistinguishable from unique data. This "free zone" exists because:

1. **Models do not memorize after one exposure.** Gradient updates from a single pass are noisy and partial. Multiple exposures allow the model to extract more signal.
2. **Stochastic data ordering** means each epoch presents the same documents in different contexts (different batch compositions, different preceding tokens in the context window), providing slightly different learning signals.
3. **The model's capacity** is large enough that 4x repetition does not saturate its ability to represent the data.

Beyond ~4 epochs, the model transitions from "extracting more signal" to "memorizing specific sequences." The symptoms:
- Training loss continues to decrease (the model is fitting the training data better)
- Validation loss plateaus or increases (the model is not generalizing)
- The model begins to reproduce training documents verbatim when prompted with their openings

### Phi-4's 13.8-Epoch Exception

Phi-4 repeats its synthetic data for 13.8 epochs — far beyond the 4-epoch "free zone." Why does this work?

The key difference: **information density.** A carefully constructed synthetic math problem contains far more learnable structure per token than a random web page. The "4-epoch rule" was established on web-crawled data; high-quality synthetic data has a higher effective information density, meaning more epochs are needed to fully extract its signal.

This does not mean synthetic data is immune to over-repetition. At some point (likely beyond 20-30 epochs, though Phi-4 does not report experiments at that scale), even synthetic data would show diminishing returns. The threshold simply shifts upward in proportion to information density.

### Budget Allocation Under Data Constraints

Given a fixed compute budget $C$ and unique data pool $D$, the optimal strategy depends on the ratio $C/D$:

- **$C/D < 4$:** Train normally. No data scarcity problem.
- **$4 < C/D < 10$:** Augment with code data and relax quality filters slightly. Accept mild repetition penalty.
- **$C/D > 10$:** Invest in synthetic data generation. The cost of running a teacher model to generate synthetic data is almost always justified by the compute savings from avoiding extreme repetition.
- **$C/D > 20$:** Use a smaller model. The Chinchilla-optimal model size for your available data is smaller than what your compute budget could support with unlimited data.

This framework explains the strategic divergence between Llama 3 (massive organic data collection, minimal synthetic, large model) and Phi-4 (aggressive synthetic generation, smaller model): they are operating at different points on the $C/D$ spectrum.
