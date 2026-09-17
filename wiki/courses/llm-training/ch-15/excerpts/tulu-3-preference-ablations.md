---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/tulu-3.md (library card not verified as of 2026-09-15; it states that preference data came from "reward model ranking", which the report does not say; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2411.15124
primary_version: arXiv:2411.15124v5 (2025-04-14); v1 2024-11
created_at: "2026-09-15"
---

# Excerpt: Tülu 3: Pushing Frontiers in Open Language Model Post-Training, prompt and preference-data ablations

Authors: Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, et al. (Allen Institute for AI and University of Washington). Source type: official technical report. Read in the v5 PDF text on 2026-09-15 for ch-15 §4, §7 and Recipe.

## Real-user prompts in SFT (§4.2, Table 10; Tülu 3 8B SFT)
| Model | Avg. | AlpacaEval 2 | IFEval | Safety |
|---|---|---|---|---|
| Tülu 3 8B SFT | 60.1 | 12.4 | 72.8 | 93.1 |
| → w/o WildChat | 58.9 | 7.5 | 70.1 | 95.2 |
"a small but noticeable degradation on most skills, most noticeably on Alpaca Eval, highlighting the importance of diverse real-world data" (§4.2). Adding contrastive prompts such as CoCoNot helped prevent over-refusal of safe prompts (§4.2).

## Preference pipeline (§5.2.1, Fig. 7)
1. Prompt selection: prompts used in SFT, prompts from the same sources left unused in SFT, and new prompts (UltraFeedback without TruthfulQA, persona instruction-following prompts).
2. Response generation: four models sampled from a pool (Fig. 7: 22 models); on-policy data adds prompts where one response comes from the Tülu SFT model and the other from off-policy models.
3. Annotation: GPT-4o-2024-08-06 rates each response 1-5 on helpfulness, instruction-following, honesty, truthfulness; binarization takes the highest mean rating as chosen and "randomly sample[s] from the responses with the lower mean as the rejected response".

## Mix (§5.2.2, Table 15)
| Dataset | Count | 8B | 70B |
|---|---|---|---|
| SFT Reused On-policy | 19,444 | ✓ | ✓ |
| SFT Reused Off-policy | 96,911 | ✓ | ✓ |
| IF-Augmented | 65,530 | ✓ | ✓ |
| WildChat IF | 10,792 | ✓ | ✓ |
| WildChat Reused | 17,207 | ✓ | ✓ |
| WildChat Unused | 82,783 | | ✓ |
| Ultrafeedback (Cleaned) | 41,635 | ✓ | ✓ |
| Persona IF | 19,890 | ✓ | |
| Total | 354,192 | 271,409 | 334,302 |

## Ablations (§5.3; 8B DPO; values plotted in Figs. 8-11 are not printed as numbers)
- More unique prompts: "noticeable performance gains across several metrics as the size of the preference dataset increases" (Fig. 8).
- Duplicated prompts: UltraFeedback expanded to 64k, 180k and 383k pairs with the same 64k unique prompts; "the 383k-size preference dataset performs similarly to the 64k preference dataset", with slight degradation on DROP, GSM8k and AlpacaEval as duplication increases (Fig. 9).
- Unused vs reused SFT prompts: 100k each; "the unused dataset has a slightly higher performance"; combining both gave the best mix (Fig. 10).
- On-policy: "including on-policy data improves aggregated downstream DPO performance compared to a completely off-policy dataset" (Fig. 11).
- Judges (Table 17, 10k UltraFeedback prompts): average downstream score GPT-4o 57.3, Llama 3.1 405B 57.2, GPT-4 Turbo 57.0, GPT-4o Mini 56.9, Llama 3.1 70B 56.6.
