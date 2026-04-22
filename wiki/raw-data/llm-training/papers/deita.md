<!-- scope: complexity + quality + diversity scoring for SFT sample selection (ICLR 2024)
     deps: [[evol-instruct]]
     see-also: [[cherry-llm]], [[ifd]], [[less]], [[alpagasus]], [[prismatic-synthesis]]
-->

# DEITA: What Makes Good Data for Alignment? (Liu 2023, ICLR 2024)
- **Core Insight:** SFT data quality decomposes into three orthogonal axes — **complexity**, **quality**, **diversity** — each scorable with a small trained scorer, and a ~10K-sample Deita-selected subset beats 10× larger random pools.
- **Guideline:** Build one complexity scorer (trained on Evol-Complexity variant rankings), one quality scorer (trained on Evol-Quality variant rankings), and pick samples by `complexity × quality` score subject to a diversity-aware greedy constraint (embedding-space min-distance); 6K–10K samples is typically enough.
- **Authors:** Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He (HKUST)
- **Year:** 2023 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2312.15685
- **Relevant topics:** SFT data selection, instruction-tuning data, complexity+quality+diversity scoring, DEITA

## Abstract
DEITA addresses the puzzle that tiny curated SFT sets (LIMA: 1K) can outperform giant ones (UltraChat: 200K). It formalizes three measurement axes — complexity, quality, diversity — operationalizes each via small ChatGPT-distilled scorers, and proposes a score-first, diversity-aware selection algorithm. Starting from a pool of 300K mixed sources (ShareGPT + UltraChat + WizardLM), DEITA picks 6K–10K samples that fine-tune Mistral-7B to state-of-the-art at its release on MT-Bench, AlpacaEval, and Open-LLM-Leaderboard — matching or beating models SFT'd on 10× more data.

## Key Contributions
- **Three-axis framework** (complexity, quality, diversity) as a reusable contract for SFT data selection.
- Two automatic scorers trained via **Evol-Complexity** and **Evol-Quality** (evolve seeds, have ChatGPT rank variants, distill rankings into a 13B LLM scorer).
- **Combined evol-score** = complexity × quality.
- **Score-first diversity-aware** greedy selector: iterate top-scored samples, admit only if embedding distance > threshold to already-selected set.
- Demonstrated 6K-sample superiority to 300K-sample baselines on open LLM leaderboard.

## Key Figures/Tables to Study
- **Table of MT-Bench / AlpacaEval / Open-LLM-Leaderboard** for DEITA-6K/10K vs Alpaca / WizardLM / UltraChat / LIMA.
- **Axis-ablation table** — removing any of the three axes hurts.
- **Selection-size sweep** — plateau around 6K–10K.

## Synthesis/selection pipeline (REQUIRED — be concrete)
- **Seed input:** pool of 300K instructions (ShareGPT + UltraChat + WizardLM).
- **Scorer training:**
  - **Evol-Complexity scorer:** take seeds, apply Evol-Instruct-style upward mutations (add constraints, increase depth, breadth); ask ChatGPT to rank variants by complexity; train a 13B LLM head on these rankings.
  - **Evol-Quality scorer:** similar but with quality-focused mutations (improve clarity, detail, informativeness).
- **Scoring every pool sample** with both scorers; multiply → `evol-score`.
- **Diversity-aware selection:** sort by evol-score desc; iterate; include sample if minimum cosine distance to already-selected embeddings > τ (e.g., 0.9).
- **Output shape:** DEITA-6K and DEITA-10K release subsets; both released with scorer weights.
- **Teacher model(s):** ChatGPT (GPT-3.5-Turbo) for ranking labels; 13B open LLMs as scorer backbones.
- **Cost estimate:** bulk of cost is initial scorer-training-data labeling via ChatGPT; selection pass is cheap once scorers exist.

## Scoring details
- Complexity/quality rankings are collected per-variant-set and aggregated into pairwise labels.
- Final selection objective is a lexicographic combination: highest `complexity × quality` subject to the diversity constraint — *not* a weighted sum, because Liu et al. show pure combined-score without diversity fails.

## Quality / diversity evaluation
- DEITA-Mistral-7B-v1.0 matched Zephyr-7B-beta on MT-Bench despite using 6K vs 200K+ samples.
- Ablation: removing diversity filter collapses score; removing complexity filter weakens reasoning; removing quality filter weakens format compliance.
- Strong transfer — same DEITA selection set fine-tunes Llama / Mistral / Yi families comparably.

## Risks + gotchas
- **Scorer-teacher bias:** ChatGPT's notion of "quality" bakes into the scorer; bias propagates to selected samples.
- **Embedding diversity is surface-level:** two samples requiring identical reasoning can live far apart in embedding space — [[prismatic-synthesis]] argues gradient-space diversity is a stricter objective.
- **Pool dependence:** if the pool lacks a capability, no selector adds it; DEITA assumes coverage in the candidate pool.
- **Static thresholds** — the diversity τ is a fixed hyperparameter; better results come from adaptive thresholds in follow-ups.

## Connections
- Contemporary with [[cherry-llm]] / [[ifd]] (perplexity-difficulty selection) and [[less]] (gradient-similarity selection).
- Weaker-signal complement: [[alpagasus]] uses GPT-rating-only — DEITA adds two more axes.
- Superseded in "diversity" axis by [[prismatic-synthesis]] (2025) which uses gradient entropy.
- Reference for Tülu 3's data selection stage; see [[tulu-3-sft-mix]].
