<!-- scope: Hugging Face blog post comparing DPO, IPO and KTO across a beta sweep on two 7B SFT models
     deps: [[dpo]]
     see-also: [[ipo]], [[kto]], [[hf-alignment-handbook]], [[ultrafeedback]], [[trl-online-dpo]]
-->

# Preference Tuning LLMs with Direct Preference Optimization Methods
- **Core Insight:** In a sweep over β ∈ {0.01, 0.1, 0.2, …, 0.9} on two 7B SFT models and two preference datasets, the best β differs by algorithm and by model — for OpenHermes-2.5-Mistral-7B the best β was 0.6 (DPO), 0.3 (KTO) and 0.01 (IPO), while for Zephyr-7b-beta-sft the best β was 0.01 for all three algorithms (Results).
- **Guideline:** When running DPO, IPO or KTO on a 7B SFT model, sweep β rather than reusing a default, because the post's two models disagree on the best value by more than an order of magnitude (Results); the post's own summary is that DPO and IPO are comparable and both beat KTO on paired preference data (Summary & Insights).
- **Authors:** Kashif Rasul, Edward Beeching, Lewis Tunstall, Leandro von Werra, Omar Sanseviero (Hugging Face)
- **Year:** 2024 (published 2024-01-18; an addendum was added after the post's first version)
- **URL:** https://huggingface.co/blog/pref-tuning
- **Source type:** official blog (practitioner evidence: the authors ran the experiments and released code and configs)
- **Relevant topics:** DPO, IPO, KTO, β hyperparameter, MT-Bench, TRL DPOTrainer, alignment-handbook

## Summary
The post is an empirical comparison of three preference-tuning objectives that do not use reinforcement
learning: Direct Preference Optimization (DPO), Identity Preference Optimisation (IPO) and
Kahneman-Tversky Optimisation (KTO). The authors sweep the β hyperparameter for each objective on two
7B models that had been supervised-fine-tuned but not preference-aligned, using two preference datasets,
and score every resulting checkpoint with MT-Bench. The stated motivation for IPO is that DPO "tends to
quickly overfit on the preference dataset", which IPO addresses with a regularisation term; the stated
motivation for KTO is that it needs only per-response "good"/"bad" labels instead of pairs, which are
cheaper to collect (Alignment without Reinforcement Learning).

An addendum at the top of the post states that, after consulting the IPO authors, the authors found the
TRL IPO implementation was incorrect — the loss over completion log-likelihoods must be averaged rather
than summed — and that the experiments were re-run after the fix (Addendum).

## Key Contributions
- A β sweep (10 values) × 3 objectives × 2 models, all trained for one epoch with the same seed and all
  other hyperparameters held fixed (Hyperparameter Sweep).
- MT-Bench scores for every run, plus per-category breakdowns of the best run per algorithm (Results).
- Released configs and launch scripts in the alignment-handbook under `recipes/pref_align_scan`, and the
  models and datasets in a public collection (Links).
- A documented correction of the TRL IPO loss (averaging instead of summing over completion
  log-likelihoods) and re-run results (Addendum).

## Key Figures/Tables to Study
- MT-Bench score vs β for Zephyr-7b-beta-sft, one curve per algorithm (Results → Zephyr-7b-beta-SFT).
- MT-Bench score vs β for OpenHermes-2.5-Mistral-7B (Results → OpenHermes-7b-2.5).
- Per-category MT-Bench breakdown of the best model per algorithm, for each of the two base models
  (Results).

## Technical Details
- Models: `teknium/OpenHermes-2.5-Mistral-7B` and `alignment-handbook/zephyr-7b-sft-full`, both 7B and
  both SFT-only (Experimental Setup).
- Datasets: `Intel/orca_dpo_pairs`, 13k prompts whose chosen response comes from GPT-4 and whose rejected
  response comes from Llama-Chat 13B; and `HuggingFaceH4/ultrafeedback_binarized`, 66k prompts with
  chosen/rejected pairs (Experimental Setup).
- For KTO the authors reuse the paired data by treating the GPT-4 responses as "good" labels and the
  Llama-Chat-13B responses as "bad" labels (Experimental Setup).
- All three objectives are selected through the `loss_type` argument of TRL's `DPOTrainer`, with values
  `sigmoid` (DPO), `ipo`, and `kto_pair` (Hyperparameter Sweep, launch script).
- β values swept: 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9; 0.01 was included because some
  algorithms are sensitive to this parameter (Hyperparameter Sweep).
- Evaluation: MT-Bench, a multi-turn benchmark judged by GPT-4 over eight categories — Writing, Roleplay,
  Reasoning, Math, Coding, Extraction, STEM, Humanities (Results).
- OpenHermes-2.5-Mistral-7B gained 0.3 MT-Bench points from preference alignment; the authors attribute
  the small gain to it being a stronger starting model (Results → OpenHermes-7b-2.5).
- Chat templates were inferred from each base model: ChatML for OpenHermes-2.5, the H4 template for
  Zephyr (Configuring the experiments).
- Conflicting statements inside the post: the per-model prose says KTO beat DPO and IPO in all but one
  setting on Zephyr and that "DPO > KTO > IPO" on OpenHermes, while the addendum and the Summary section
  say IPO is on par with DPO and both outperform KTO on paired data. The addendum states the post was
  updated to reflect the post-fix results, so the addendum and Summary describe the corrected runs
  (Addendum; Results; Summary & Insights).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenHermes-2.5-Mistral-7B | 7B | preference | preference loss | `sigmoid` / `ipo` / `kto_pair` via TRL `DPOTrainer` | hf.co/blog/pref-tuning, "Hyperparameter Sweep" | verified 2026-09-18 | swept, not selected |
| OpenHermes-2.5-Mistral-7B | 7B | preference | β sweep grid | 0.01, 0.1, 0.2, …, 0.9 | same, "Hyperparameter Sweep" | verified 2026-09-18 | 0.01 added because some algorithms are sensitive to β |
| OpenHermes-2.5-Mistral-7B | 7B | preference | best β | DPO 0.6; KTO 0.3; IPO 0.01 | same, "OpenHermes-7b-2.5" | verified 2026-09-18 | MT-Bench score at each β |
| OpenHermes-2.5-Mistral-7B | 7B | preference | peak LR / scheduler / warmup | 5.0e-7, cosine, warmup_ratio 0.1 | same, `config_openhermes` YAML in "Configuring the experiments" | verified 2026-09-18 | no ablation reported |
| OpenHermes-2.5-Mistral-7B | 7B | preference | per-device train batch / grad accum / epochs | 8 / 2 / 1 | same YAML | verified 2026-09-18 | no ablation reported |
| OpenHermes-2.5-Mistral-7B | 7B | preference | optimizer / precision / max prompt length / seed | adamw_torch / bf16 / 512 / 42 | same YAML | verified 2026-09-18 | no ablation reported |
| OpenHermes-2.5-Mistral-7B | 7B | preference | preference data | `HuggingFaceH4/orca_dpo_pairs`, mixer weight 1.0 (13k prompts) | same YAML; "Experimental Setup" | verified 2026-09-18 | no ablation reported |
| Zephyr-7b-beta-sft (`alignment-handbook/zephyr-7b-sft-full`) | 7B | preference | best β | 0.01 for DPO, IPO and KTO | same, "Zephyr-7b-beta-SFT" | verified 2026-09-18 | MT-Bench score at each β |
| Zephyr-7b-beta-sft | 7B | preference | preference data | `HuggingFaceH4/ultrafeedback_binarized` (66k prompts) | same, "Experimental Setup" | verified 2026-09-18 | no ablation reported |
| Zephyr-7b-beta-sft | 7B | preference | full config | not reported (post says only that a "similar base configuration file" was created; checked post body and linked recipe description) | same, "Configuring the experiments" | not reported | — |

- Distributed setup: the launch script passes `deepspeed_zero3`; number and type of GPUs is not reported
  (launch script in "Hyperparameter Sweep").

## Findings relevant to generality
- The ranking of the three algorithms was reported as the same on both models, but the best β was not:
  0.01 for all three on Zephyr versus 0.6 / 0.3 / 0.01 on OpenHermes (Results). The post concludes that
  hyperparameter choice matters for preference alignment (Summary & Insights).
- The post reports remaining weakness on the Reasoning, Coding and Math MT-Bench categories for the best
  Zephyr models under every algorithm (Results → Zephyr-7b-beta-SFT).

## Connections
- [[dpo]] — the baseline objective the post evaluates.
- [[ipo]] — the objective whose TRL implementation this post corrected.
- [[kto]] — the unpaired-label objective evaluated alongside DPO and IPO.
- [[ultrafeedback]] — source of the `ultrafeedback_binarized` preference pairs used for Zephyr.
- [[hf-alignment-handbook]] — the repository holding the configs and launch scripts used here.
- [[hf-rlhf-illustrated]] — earlier Hugging Face post on the RL-based RLHF pipeline.
- [[trl-online-dpo]], [[trl-ppo]], [[trl-grpo]] — sibling TRL trainers.

## Verification
- Checked on 2026-09-18 against: https://huggingface.co/blog/pref-tuning (page as served 2026-09-18, including the Addendum).
- Corrections to the previous card version:
  - Title "HuggingFace — Preference Optimization Zoo (DPO variants survey)" → the artifact is a single post titled "Preference Tuning LLMs with Direct Preference Optimization Methods" (page title, published 2024-01-18).
  - "Author: HuggingFace team (Kashif Rasul, Younes Belkada, Lewis Tunstall, and contributors)" → the bylines are Kashif Rasul, Edward Beeching, Lewis Tunstall, Leandro von Werra, Omar Sanseviero; Younes Belkada is not a listed author (byline).
  - "Year: 2024 (running updates)" → published 2024-01-18 with one addendum; the post is not a running document (page header).
  - "a family of variants (IPO, KTO, SimPO, ORPO, BCO)" and the per-variant sections for SimPO, ORPO, BCO and CPO → the post covers only DPO, IPO and KTO (TL;DR, Introduction).
  - "Start with plain DPO (beta ~ 0.1)" → β 0.1 is not recommended anywhere in the post; the best values found were 0.01 (Zephyr, all three) and 0.6 / 0.3 / 0.01 (OpenHermes, DPO / KTO / IPO) (Results).
  - "Worked examples on UltraFeedback / Anthropic HH data" → the two datasets are `ultrafeedback_binarized` and `Intel/orca_dpo_pairs`; Anthropic HH is not used (Experimental Setup).
  - "Benchmark table on MT-Bench / AlpacaEval across variants at matched training time" → only MT-Bench is used, and runs are matched at one epoch, not at matched wall-clock time (Results; Hyperparameter Sweep).
- Removed as unsupported by the source:
  - "chosen-logprob collapse" as a DPO failure mode, and the claim that IPO "prevents the chosen-logprob collapse that plain DPO exhibits at scale". The post's stated DPO shortcoming is fast overfitting on the preference dataset, and it shows no logprob curves.
  - "too-low beta causes collapse" — the post's best β on Zephyr was its lowest swept value, 0.01.
  - Sections describing SimPO, ORPO, BCO and CPO, including the length-normalized SimPO loss, the ORPO odds-ratio objective, and BCO's Platt scaling: none appear in the post.
  - "Comparison table of assumption/loss/hyperparameter per variant" and "chosen-vs-rejected logprob curves" — no such table or figure is in the post.
  - "When to use which (HF guidance)" decision list — the post gives no such list; its closing statement is that DPO appeared the most robust and best-performing at the time of writing (What's next?).
  - "sensitivity to reference model choice" and "dependence on well-balanced pairwise data" as documented DPO failure modes — not stated in the post.
- Not reported by the source: number and type of GPUs; the full Zephyr config file; any evaluation other
  than MT-Bench; results for models outside 7B.
