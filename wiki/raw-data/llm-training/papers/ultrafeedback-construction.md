<!-- scope: redirect — this slug and [[ultrafeedback]] describe the same artifact (arXiv:2310.01377)
     see-also: [[ultrafeedback]]
-->

# UltraFeedback construction — redirect to [[ultrafeedback]]

> **Redirect card.** This file and [[ultrafeedback]] both extracted arXiv:2310.01377,
> "ULTRAFEEDBACK: Boosting Language Models with Scaled AI Feedback" (Cui, Yuan, Ding, Yao, He, Zhu,
> et al.; arXiv v1 2023-10, ICML 2024). One artifact gets one card, so the verified extract now lives
> in [[ultrafeedback]]. The slug is kept because other library pages and chapters link to it.

- **Canonical card:** [[ultrafeedback]]
- **URL:** https://arxiv.org/abs/2310.01377
- **Source type:** paper

## Where the construction content went
The pipeline material that was on this page is in [[ultrafeedback]] under these headings:

| Content | Heading in [[ultrafeedback]] | Locus in the paper |
|---|---|---|
| Six instruction sources and the 63,967-instruction count | Technical Details | §2.2, App. B |
| The 17-model completion pool and the four-per-instruction sampling | Technical Details | §2.3 |
| Principles added to the system prompt to vary completion style | Technical Details | §2.3, App. G.1 |
| Four aspects, the 1-to-5 documented scale, rationales, joint scoring of four completions | Technical Details | §2.4, App. G.2 |
| Dataset scale: 255,864 completions, 340,025 pairs, 255,864 critiques | Technical Details | §2.5, Table 1 |
| Decontamination by 13-gram match (48 samples removed) | Technical Details | App. B |
| UltraRM, UltraCM, UltraLM-13B-PPO training settings | Recipe ledger | §3.1, §3.3, App. D |

## Connections
- [[ultrafeedback]] — the canonical card for this artifact.
- [[ultrachat-pipeline]] — supplies 10k of the instructions (§2.2).
- [[rlcd]], [[west-of-n]], [[self-taught-evaluators]], [[nemotron-4-synthetic]] — other synthetic-preference constructions in the library.
- [[nathan-lambert-synthetic-data]], [[allenai-tulu-synth]] — blog cards that link here for the preference-synthesis pattern.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2310.01377 (arXiv v2, 16 Jul 2024).
- Corrections to the previous card version:
  - Duplicate of [[ultrafeedback]]; reduced to a redirect card. The two pages previously gave different numbers for the same dataset.
  - "4 aspects ... with numerical ratings (0–10)" → the documented score range is 1 to 5 (§2.4, App. G.2).
  - "64K prompts × 4 responses = 256K rated samples", "~60K diverse prompts", "Binarized subset ≈ 62K pairs" → 63,967 instructions, 255,864 completions, 340,025 preference pairs (§2.2, §2.4, Table 1).
  - "Year: 2023/2024" → arXiv v1 2023-10; ICML 2024.
  - Generator list naming Dolly, StableLM, and Llama-2-70B-chat/WizardLM-70B tiering → the pool in §2.3 is 17 named models and does not include Dolly or StableLM; Bard and StarChat are in it.
  - "query 4 out of 17 models with randomized system prompts covering helpful/honest/harmless/expert styles" → four models are sampled at random, and the system prompt carries a sampled principle drawn from one hand-written plus ten GPT-4-generated principles per aspect (§2.3).
- Removed as unsupported by the source: the `ultrafeedback_binarized` rule "chosen = argmax, rejected = random-not-chosen (intentionally not worst-of-4, to avoid degenerate negatives)" — this is a third-party HuggingFace derivative and the rule appears nowhere in the paper; "drop prompts with all-identical scores; drop clearly malformed responses" (the only filtering the paper describes is 13-gram decontamination, App. B); "cost estimate: tens of thousands USD"; "Zephyr-7B-β MT-Bench 7.34"; "used in Tulu 2, Starling-7B, and dozens of community fine-tunes"; "ablations show aspect-level prefs outperform overall-score prefs when target capability is specific" (§3.1 compares fine-grained against overall scores on reward-model accuracy only); "GPT-4 scoring patterns (length bias, helpfulness bias, style bias) propagate into downstream DPO models"; "the 4 aspects are correlated"; "the judge is rating its own outputs — subtle advantage to GPT-4 responses"; "the binarized subset sometimes excludes TruthfulQA"; "2025 replacements" framing and the attributed Interconnects position on accumulation.
- Not reported by the source: annotation cost; per-source instruction counts after decontamination; construction of any binarized derivative.
