---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: Biderman et al., "LoRA Learns Less and Forgets Less", arXiv:2405.09673v2 (TMLR 08/2024)
source_url: https://arxiv.org/abs/2405.09673
created_at: "2026-09-15"
source_type: paper
---

# Excerpt: LoRA Learns Less and Forgets Less

No library card for this source exists yet (planned slug `lora-learns-less-forgets-less`). Authors: Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, et al. (Columbia University, Databricks Mosaic Research).

## Abstract (verbatim, abridged)

"we compare the performance of LoRA and full finetuning on two target domains, programming and mathematics. We consider both the instruction finetuning (≈100K prompt-response pairs) and continued pretraining (≈20B unstructured tokens) data regimes. Our results show that, in the standard low-rank settings, LoRA substantially underperforms full finetuning. Nevertheless, LoRA better maintains the base model's performance on tasks outside the target domain. We show that LoRA mitigates forgetting more than common regularization techniques such as weight decay and dropout; it also helps maintain more diverse generations. Finally, we show that full finetuning learns perturbations with a rank that is 10-100× greater than typical LoRA configurations".

## Setting

- Model: Llama-2-7B. LoRA: W_finetuned = W_pretrained + Δ, Δ = γ_r·AB, A ∈ R^{d×r}, B ∈ R^{r×k} (§2); ranks r = 16, 64, 256 with α = 2r, all transformer modules.
- Instruction data (Table 1): Magicoder-Evol-Instruct-110K (72.97M tokens), MetaMathQA (103M tokens).
- Learning metrics (§3.2): HumanEval pass@1 (50 generations, temperature 0.2, top_p 0.95); GSM8K test (1,319 problems).
- Forgetting metric (§3.3): average of HellaSwag, WinoGrande, ARC-Challenge.

## Code IFT results (Tables S5 and S6)

| Condition | HumanEval pass@1, epoch 1 / 2 / 4 / 8 / 16 | Forgetting average, epoch 1 / 2 / 4 / 8 / 16 |
|---|---|---|
| LoRA r = 16 | 0.197 / 0.275 / 0.358 / 0.338 / 0.324 | 0.653 / 0.648 / 0.652 / 0.646 / 0.609 |
| LoRA r = 64 | 0.249 / 0.339 / 0.417 / 0.392 / 0.405 | 0.652 / 0.651 / 0.632 / 0.580 / 0.510 |
| LoRA r = 256 | 0.299 / 0.385 / 0.498 / 0.437 / 0.466 | 0.655 / 0.659 / 0.631 / 0.552 / 0.517 |
| Full fine-tuning | 0.302 / 0.464 / 0.470 / 0.497 / 0.416 | 0.595 / 0.579 / 0.512 / 0.446 / 0.414 |

§4.2 (verbatim): "(1) IFT induces more forgetting than than CPT, (2) programming induces more forgetting than math, and (3) forgetting tends to worsen with training duration. Most importantly, LoRA forgets less than full finetuning, and the extent of forgetting is controlled by rank." In math IFT, "LoRA and full finetuing respectively degrade to 0.567 and 0.559 at epoch 16" (Table S8).

## Diversity (§4.5, Fig. 5)

"full finetuning results in fewer unique generations ("distribution collapse") compared to the base model, for both pass and fail generations, with LoRA in between the two." Measured as unique output strings out of 50 HumanEval generations; the authors note exact string matching is not a sensitive diversity metric.

## General instruction data (App. C, Tülu-v2-mix, about 326K samples)

Settings (C.1): full fine-tuning LR 5e-6, LoRA LR 1e-4, decoupled LionW (0.9, 0.95), cosine with 10% warmup, batch 192, length 4096. Results (C.2): on MT-Bench "all LoRA models are within one standard error of the mean of the full finetuning model"; for forgetting, "At two epochs, full finetuning does better than LoRA. The former starts to degrade at epoch 4. At epoch 6, the findings of the main paper are replicated". "LoRA r = 16 seems to offer competitive conversational capabilities, and minimal forgetting, but it underperforms in domain-specific knowledge like math."

## Recommendations (§4.7)

LoRA's best learning rates "should be set one order of magnitude higher than that of full finetuning, often ranging between 5e−5 and 5e−4"; recommendations: "(a) using LoRA for instruction finetuning and not continued pretraining; (b) if GPU memory allows, targeting "All" transformer modules with a rank of 256 … (c) using α = 2r, and (d) sweeping over learning rates between [1e − 5, 5e − 4], picking the highest value that enables stable training."

## Connections

- [[read]] §8, Generalization lens, Recipe; [[ch-30a]] for forgetting measurement and controls.
- [[lora-without-regret]] — independent LoRA study measuring log loss.
