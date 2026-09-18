<!-- scope: recipe ledger for Cherry LLM (Li et al., NAACL 2024) — SFT and selection settings from the paper and the released repository
     deps: [[cherry-llm]]
     see-also: [[ifd]], [[superfiltering]]
-->

# Recipe ledger — From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning
- **Parent card:** [[cherry-llm]]
- **Sources:** arXiv:2308.12032v5 (paper); github.com/tianyi-lab/Cherry_LLM at commit 01fac8d (README "Hyperparameters", README "Run Code", `cherry_seletion/data_by_IFD.py`). Paper values and repository values are recorded as separate rows (§5.3 rule 4 of the course standard).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLaMA-7B cherry models (Alpaca and WizardLM pools) | 7B | SFT | optimizer | Adam | arXiv:2308.12032v5 App. A | verified 2026-09-14 | no ablation reported |
| LLaMA-7B cherry models | 7B | SFT | learning rate | 2 × 10⁻⁵ | arXiv:2308.12032v5 App. A | verified 2026-09-14 | no ablation reported |
| LLaMA-7B cherry models | 7B | SFT | batch size | 128 (unit not stated in the paper) | arXiv:2308.12032v5 App. A | verified 2026-09-14 | no ablation reported |
| LLaMA-7B cherry models | 7B | SFT | epochs | 3 | arXiv:2308.12032v5 App. A | verified 2026-09-14 | no ablation reported |
| LLaMA-7B pre-experienced model | 7B | SFT | epochs | 1 | arXiv:2308.12032v5 §2.1; App. A | verified 2026-09-14 | no ablation reported |
| LLaMA-7B pre-experienced model | 7B | SFT | examples and selection | 1,000 = 10 samples from each of 100 K-means clusters of base-model instruction embeddings | arXiv:2308.12032v5 §2.1; §4.3.1 | verified 2026-09-14 | §4.3.1 Figure 5: 0/100/300/500 samples, 300 gives "a distinct performance gain" and more gives no further gain; Table 2: difficulty/diversity/random choice of the 1,000 comparable (ChatGPT judge) |
| LLaMA-7B, Alpaca pool | 7B | SFT | max input length | 512 | arXiv:2308.12032v5 App. A | verified 2026-09-14 | no ablation reported |
| LLaMA-7B, WizardLM pool | 7B | SFT | max input length | 1024 (original WizardLM used 2048) | arXiv:2308.12032v5 App. A | verified 2026-09-14 | chosen "due to hardware constraints"; no ablation |
| LLaMA-7B, WizardLM pool | 7B | SFT | pool after "AI censure" filtering | 63,655 entries (from WizardLM70K) | arXiv:2308.12032v5 App. A; §3.1 | verified 2026-09-14 | none |
| LLaMA-7B cherry Alpaca | 7B | SFT | selected share | "approximately 5%" for the main comparison; sweep of 5%, 10%, 15%, 20% | arXiv:2308.12032v5 §1; §4.1; Figure 3 | verified 2026-09-14 | Figure 2(a), GPT-4 pairwise judge vs official Alpaca; Figure 3 winning score |
| LLaMA-7B cherry WizardLM | 7B | SFT | selected share | "approximately 10%" vs reimplemented WizardLM; 40% vs official WizardLM | arXiv:2308.12032v5 §4.1; App. D | verified 2026-09-14 | Figure 2(b); App. D Table 8 |
| All cherry models | 7B, 13B | SFT | selection rule | discard IFD > 1; keep the highest remaining IFD | arXiv:2308.12032v5 §1; §2.2 | verified 2026-09-14 | §4.2 Figure 4 and App. B Table 5: IFD selection above random, K-means diversity, low-IFD, and high-CA selection at equal size |
| Cherry Models V1 (Alpaca), LLaMA-7B | 7B | SFT | global batch; LR; epochs; max length; weight decay; warmup rate | 128; 2e-5; 3; 512; 0; 0.03 | Cherry_LLM@01fac8d README "Hyperparameters" | verified 2026-09-14 | no ablation reported |
| Cherry Models V1 (WizardLM), LLaMA-7B | 7B | SFT | global batch; LR; epochs; max length; weight decay; warmup rate | 128; 2e-5; 3; 1024; 0; 0.03 | Cherry_LLM@01fac8d README "Hyperparameters" | verified 2026-09-14 | no ablation reported |
| LLaMA2-7B / LLaMA2-13B cherry Alpaca | 7B, 13B | SFT | scorer; prompt; max length | IFD from the base LLaMA2 model, no pre-experienced model; Vicuna prompt; 2048 | arXiv:2308.12032v5 §4.4; App. A | verified 2026-09-14 | Table 3 (5%, 10%, 15% vs full data) |
| LLaMA2-7B / LLaMA2-13B cherry Alpaca | 7B, 13B | SFT | LR, batch, epochs in the paper | not reported | checked §3.2, §4.4, App. A | not reported | — |
| Cherry Models V2 7B (LLaMA2-7B) | 7B | SFT | global batch; LR; epochs; max length; weight decay; warmup rate | 128; 2e-5; 3; 2048; 0; 0.03 | Cherry_LLM@01fac8d README "Hyperparameters" | verified 2026-09-14 | no ablation reported |
| Cherry Models V2 13B (LLaMA2-13B) | 13B | SFT | global batch; LR; epochs; max length; weight decay; warmup rate | 128; 1e-5; 5; 2048; 0; 0.03 | Cherry_LLM@01fac8d README "Hyperparameters" | verified 2026-09-14 | no ablation reported |
| Selection script default | — | SFT (data selection) | `--sample_rate` default | 0.1 | Cherry_LLM@01fac8d `cherry_seletion/data_by_IFD.py` L27 | verified 2026-09-14 | framework default; README example uses 0.06 ("Run Code" step 4) |
| Pre-experience selection script (README example) | — | SFT (data selection) | clusters; samples per cluster; within-cluster thresholds | `--kmeans_num_clusters 100`; `--sample_num 10`; `--low_th 25`, `--up_th 75` | Cherry_LLM@01fac8d README "Run Code" step 1 | verified 2026-09-14 | the 25/75 thresholds are not described in the paper |
| All runs | 7B, 13B | SFT | loss masking, packing, LR schedule shape, compute | not reported | checked paper body, App. A, README "Hyperparameters" | not reported | — |

## Notes
- The paper's batch size in App. A has no unit; the README column is labeled "Global Batch Size".
- The README reports LLaMA2-13B at LR 1e-5 and 5 epochs; the paper does not give LLaMA2 hyperparameters, so these values come only from the repository.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2308.12032 (v5, 2024-04-06); https://github.com/tianyi-lab/Cherry_LLM/tree/01fac8d2febb51dbd1501ec594a20e499b51fe7d
- Corrections to the previous card version: this file is new; the previous [[cherry-llm]] card's "~1K random subset" warm-up is corrected here to K-means selection (§2.1).
- Removed as unsupported by the source: none.
