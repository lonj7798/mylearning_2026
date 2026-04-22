<!-- scope: safety / red-team synthesis — SALAD-Bench hierarchical safety taxonomy + attack-enhanced questions
     deps: [[harmbench-data]]
     see-also: [[wildguard-data]]
-->

# SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models
- **Core Insight:** Safety evaluation requires a **hierarchical taxonomy** (6 domains → 16 tasks → 66 categories) so researchers can diagnose which specific harm types a model struggles with; SALAD-Bench delivers this plus ~30K questions enhanced with attack methods (jailbreak-augmented variants) — enabling fine-grained safety assessment beyond single-dimension scores.
- **Guideline:** When reporting safety, don't rely on a single "harmfulness rate"; use a hierarchical taxonomy to identify which categories are safe vs unsafe; SALAD-Bench's 66-category breakdown is a practical default.
- **Authors:** Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, Jing Shao (Shanghai AI Lab + HIT)
- **Year:** 2024 (Feb arXiv)
- **URL:** https://arxiv.org/abs/2402.05044
- **Relevant topics:** safety taxonomy, hierarchical benchmark, attack-enhanced questions, SALAD-Bench

## Abstract
SALAD-Bench ("Safety ALignment And Delineation Benchmark") is a hierarchical safety benchmark for LLMs built on a 6-domain → 16-task → 66-category taxonomy. It contains ~30K base harmful questions plus attack-enhanced variants (jailbreak-augmented), enabling evaluation of both natural-prompt robustness and adversarial robustness. The benchmark also includes an MD-Judge model — a fine-tuned Llama-2-7B safety judge — that achieves strong agreement with human annotations.

## Key Contributions
- **3-tier hierarchical taxonomy** — 6 domains, 16 tasks, 66 categories.
- **~30K base questions** + attack-enhanced variants.
- **MD-Judge** — a fine-tuned safety-judge model for automatic scoring.
- **Full leaderboard** of 30+ LLMs evaluated under the taxonomy.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Hierarchical taxonomy
- **6 domains (top-level):**
  - Representation & Toxicity.
  - Misinformation Harms.
  - Socioeconomic Harms.
  - Information & Safety Harms.
  - Malicious Use.
  - Human-Chatbot Interaction Harms.
- **16 tasks (mid-level):** e.g., toxic language, bias, factual error, social-economic unfairness, privacy leak, fraud, illegal activity, psychological manipulation, …
- **66 categories (leaf-level):** e.g., racial-stereotype, gender-bias, medical-misinformation, financial-scam, weapon-synthesis, self-harm-ideation, …

### Question curation
- Start from existing safety datasets (AdvBench, BeaverTails, DoNotAnswer, ToxicChat).
- Augment with LLM-synthesized questions per category.
- Manual review to enforce category correctness.
- Yield ~30K base questions.

### Attack-enhanced variants
- Apply 6 attack methods to each base question:
  - GCG adversarial suffix.
  - Word-level perturbation.
  - Human-written jailbreaks (DAN-style).
  - Multilingual translation attacks.
  - Persona injection.
  - Crescendo / escalation attacks.
- Yield additional ~10K attack-enhanced questions.

### MD-Judge training
- Base: Llama-2-7B.
- Training data: ~3K human-labeled (query, response, safety-label) triples.
- Evaluation agreement with human: ~89%.

- **Output shape:** ~30K base + ~10K attack-enhanced questions; MD-Judge model for automatic grading.
- **Teacher model:** GPT-4 for initial synthesis; MD-Judge for evaluation.
- **Cost:** significant — multi-stage curation with human review.

## Modality-specific technical details (REQUIRED — safety)
- **Taxonomy of harms:** 3-tier hierarchical with 66 leaves — finest-grained public safety taxonomy as of release.
- **Generation-side red-team protocol:** 6 attack methods producing enhanced variants.
- **Deduplication against training data:** manual audit against common safety-benchmark sources.
- **Red-team LLM used:** GPT-4 for non-adversarial enhancement; GCG uses gradient-based suffix generation.

## Quality / diversity evaluation
- Llama-2-Chat-70B: safe on 95%+ base questions; drops to ~75% under attack.
- GPT-4: safe on 97%+ base; ~85% under attack.
- Best 2024-era open models (Mistral-Instruct-v0.2) on mid-70s under attack.
- MD-Judge matches human annotators at ~89% accuracy.

## Risks + gotchas
- **Taxonomy granularity** means many categories have few examples (<100); per-category numbers are noisy.
- **Benchmark contamination risk** — public release means future training data may include these prompts.
- **English-only:** multilingual coverage is only via translation-attack variants, not native.
- **Attack catalog frozen** — new jailbreak methods (post-2024) aren't covered in v1.

## Connections
- Sibling: [[harmbench-data]] (400 behaviors × 18 attacks), [[wildguard-data]] (moderation + refusal synthesis).
- Attack-method overlap: GCG, PAIR — common across red-team benchmarks.
- Judge-model lineage: MD-Judge → WildGuard (Allen AI) → MD-Judge v2.
