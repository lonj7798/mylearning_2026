---
chapter: ch-57
course: llm-training
phase: read
excerpt_of: arXiv:2405.09673v2 (TMLR 08/2024)
source_url: https://arxiv.org/abs/2405.09673
created_at: "2026-09-17"
---

# Excerpt: LoRA Learns Less and Forgets Less

**Artifact.** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard,
Connor Jennings et al., "LoRA Learns Less and Forgets Less", arXiv:2405.09673v2 (v2 dated 2024-09-20;
published in *Transactions on Machine Learning Research*, 08/2024). Read on 2026-09-17 from the v2 PDF.

Used by [[read]] §7 and the Generalization lens.

---

## Setup

Base model: Llama-2-7B. Two target domains (programming, mathematics) × two data regimes:

- **IFT** (instruction fine-tuning), ≈100K prompt-response pairs — Magicoder (code), MetaMathQA (math).
- **CPT** (continued pretraining), ≈20B unstructured tokens — StarCoder-Python (code), OpenWebMath (math).

LoRA ranks r = 16, 64, 256. The paper notes that most LoRA papers use a "low" rank of 8-64 (§4.1).

## Two metrics

- **Learning (target domain)**: HumanEval pass@1, 164 problems, 0-shot (§3.2); GSM8K strict match on the
  1,319-example test split as implemented in the LM Evaluation Harness (§3.2).
- **Forgetting (source domain)**: the **average of HellaSwag, ARC-Challenge and WinoGrande** scores
  (§3.2; Fig. 2 caption; Fig. 3 right panel caption). All three are multiple-choice and scored by accuracy, so
  no generation hyperparameters are involved (§3.2).

## Numbers quoted in the chapter

| Regime | Learning | Forgetting |
|---|---|---|
| Code IFT, epoch 4 | LoRA r=16 **0.358**, r=64 **0.417**, r=256 **0.498**; full FT **0.497** at epoch 8 (§4.1, Table S5) | at epoch 16, full FT **0.414** vs LoRA r=64 **0.509** (§4.2 prose; Table S6). Table S6 at epoch 4: full FT **0.512**, r=64 **0.632**, r=256 **0.631**; at epoch 8: full FT **0.446**, r=256 **0.552** |
| Code CPT, 20B tokens | LoRA underperforms full FT at every rank; best LoRA r=256 peaks at 20B (§4.1) | full FT **0.545** vs LoRA r=256 **0.617** (§4.2, Table S2) |
| Math CPT | LoRA r=256 peaks **0.203** at 16B; full FT **0.224** at 4B and **0.293** at 20B (§4.1) | LoRA **0.616** (20B) vs full FT **0.613** (16B) — no gap (§4.2, Table S4) |
| Math IFT | LoRA r=256 **0.634** at 8 epochs; full FT **0.641** at 2 epochs, **0.642** at 4 (§4.1); r=64 reaches **0.624** at epoch 4 | LoRA **0.567** vs full FT **0.559** at epoch 16 — no gap (§4.2, Table S8) |

Ordering stated in §4.2: IFT induces more forgetting than CPT; programming induces more forgetting than math;
forgetting worsens with training duration. The paper attributes the small math gap to OpenWebMath being
dominated by English sentences (§4.2), i.e. a smaller domain shift from pretraining.

## Pass@k (Appendix F)

Recomputing HumanEval as pass@k for k = 1 … 256 at temperature 0.8, epoch 4: full fine-tuning is superior to
LoRA r = 256 for k < 64, after which the two are equal (§4.1). A pass@1 comparison hides this.

## Diversity (§4.5, Fig. 5)

Counting unique output strings out of 50 generations per HumanEval problem, separately for passing and failing
generations: full fine-tuning produced fewer unique generations than the base model ("distribution collapse"),
with LoRA between the two. The authors state the limit: "exact string matching between generations is not a
sensitive metric of predictive diversity, as generations can slightly vary in format and remain functionally
identical."

## Against regularization baselines (§4.5)

Attention dropout (0.05, 0.1) and weight decay "appear to learn and forget as much as full finetuning, except
that weight decay starts to generally deteriorate at longer training durations (epochs 8 and 16). LoRA, with
the common r = 16, learns less and forgets less than all other models. LoRA r = 256, on the other hand, learns
as much as the other methods while forgetting less."

## Rank of the full-fine-tuning update (§4.6)

For the `W_q` projection at layer 26 of Llama-2-7B (d = 4096), the fine-tuned matrix, the base matrix, and the
difference Δ all have similar slowly decaying spectra, needing ≈2000/4096 singular vectors for 90% of the
variance. Full fine-tuning finds perturbations with rank 10-100× higher than typical LoRA configurations
(abstract; Fig. 6).

## Configuration recommendations (§4.7, verbatim)

> we recommend: (a) using LoRA for instruction finetuning and not continued pretraining; (b) if GPU memory
> allows, targeting "All" transformer modules with a rank of 256, since ranks 16 − 64 tend not to suffice for
> code tasks; (c) using α = 2r, and (d) sweeping over learning rates between [1e − 5, 5e − 4], picking the
> highest value that enables stable training.

Supporting observations: PEFT scales LoRA matrices by α/r, so a fixed α scales high ranks down; in a joint
α × learning-rate sweep at r = 256 on Magicoder for 4 epochs, α = 512 outperformed 256 and 32 at all learning
rates (Fig. S3). Targeting "Attention" alone underperformed "MLP" and "All", with most of the gain from the MLP
modules (Fig. 7). LoRA is more sensitive to learning rate than full fine-tuning and its best rates are about an
order of magnitude higher (§4.7, Fig. S1). At fixed batch size and standard implementations, LoRA trains
**slower** than full fine-tuning (Appendix I).

## Conditions and limits

One base model, one model size, two target domains, one three-benchmark forgetting suite, one seed regime not
stated per point. The forgetting advantage is present in code and absent in math. Nothing here was measured on
preference optimization or RL, only on SFT-style and CPT-style objectives.

## Connections
- [[read]] — §7 and the Generalization lens.
- [[trl-repo-a04ffd3]] — TRL's PEFT reference-model path, where the LoRA/full-FT choice is made in practice.
