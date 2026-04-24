---
chapter: ch-46
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dpo.md
source_url: https://arxiv.org/abs/2305.18290
created_at: "2026-04-23"
---

# Excerpt: DPO — Option A of the ch-46 lab

**Source library:** `wiki/raw-data/llm-training/papers/dpo.md`
**Artifact:** DPO loss Eq. 7, implicit reward identity, β as KL budget, length-hacking failure mode.

---

## Why this source is Option A of the lab

Option A of ch-46 is a β sweep over the DPO loss exactly as Rafailov et al. 2023 defined it. The lab's sweep values {0.05, 0.1, 0.3} bracket the paper's own recipe range ({0.05, 0.1, 1, 5} in the sweep table, most recipes at 0.1), and the instrumentation signals map 1-to-1 onto quantities DPO names explicitly: implicit reward `r̂ = β log(π/π_ref)`, KL budget controlled by β, and the length-hacking failure mode attested as a known pathology.

---

## The one equation the lab sweeps

Source §Technical Details / Equation 7:

> `L_DPO(π_θ; π_ref) = −E_{(x,y_w,y_l)~D} [ log σ( β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x) ) ]`

This is what TRL's `DPOTrainer` with `loss_type="sigmoid"` computes. β is the sole sweep axis in Option A — every other hyperparameter is held fixed per [[karpathy-training-neural-net-recipe]] "one-change-one-prediction".

---

## β controls the KL budget — the source's exact phrasing

Source §Key Contributions:

> Shows that β controls the KL budget between trained policy and reference policy.

This is the mechanistic reason why the ch-46 `kl` signal is the *cleanest* sweep axis on the plot. β=0.05 lets KL grow large; β=0.3 keeps KL near zero. The ch-46 HTML companion's `A-kl` view is direction-attested here.

---

## Implicit reward — what `rewards/chosen` in TRL logs actually is

Source §Technical Details / Implicit reward:

> `r̂_θ(x,y) = β log [π_θ(y|x) / π_ref(y|x)]`

TRL's `DPOTrainer` logs `rewards/chosen = r̂_θ(x, y_w)` and `rewards/rejected = r̂_θ(x, y_l)`. Ch-46 §3 Instrumentation uses their difference (`rewards/margins`) as the `reward_mean` signal. Critical reading: this is NOT a reward-model score. It scales with β by construction, so you **cannot compare absolute values across β cells**; only the *direction* and the margin trajectory are comparable.

---

## Why the lab predicts length hack at β=0.05

Source §Technical Details / Hyperparameters, last row:

> Length normalization | off (known failure mode — see SimPO)

Length hack is not a property of the loss; it is a property of *the preference data having a length prior* that β=0.05 fails to filter out. Ch-46 §1 Data prep's `length_delta = len(chosen) - len(rejected)` column is exactly the audit the source implies. If the data has a systematic length bias, β=0.05 amplifies it, β=0.3 partially cancels it, β=0.1 is the attested sweet spot.

---

## Gradient weighting — why DPO is stable despite single-shot classification

Source §Technical Details / Gradient form:

> `∇L_DPO = −β E[ σ(r̂_l − r̂_w) ( ∇log π_θ(y_w|x) − ∇log π_θ(y_l|x) ) ]`
> The σ(·) term is an automatic weighting: samples that already satisfy the preference get near-zero gradient; violations get full weight.

This is why Option A does not need PPO-style clipping or GAE: the sigmoid already discounts "easy wins." The practical consequence for ch-46 is that a sweep cell that seems "stuck" is not a bug — it's the sigmoid saturating because most pairs already rank correctly. Ch-46 §7 Acceptance criterion #3 (margin must be monotone up for β=0.1) checks that the sigmoid has not saturated at step 0 due to a broken π_ref.

---

## What ch-46 keeps, changes, drops from DPO paper

| DPO paper default | Ch-46 Option A choice | Reason |
|---|---|---|
| Sweep β in {0.05, 0.1, 1, 5} | Sweep β in {0.05, 0.1, 0.3} | Tighter range around practitioner sweet spot; β=1,5 rarely deployed |
| Pythia / GPT-2 base | Llama-3.2-3B-Instruct | decoder-only SFT base is 2024+ convention |
| Reddit TL;DR / Anthropic HH | UltraFeedback or ch-38 synthetic | broader-domain preferences |
| LR 5e-7 to 1e-6 | 5e-7 | paper's lower bound; stable on 3B |
| Batch 32-128 pairs | 32 | keeps all three sweep cells comparable |
| Length-normalization off | off (as paper) | failure-mode surfacing is the lab goal |

---

## Connections to the rest of the track

- **ch-37 (preference-feedback foundations)** — provides the Bradley-Terry framing DPO inverts.
- **ch-38 (DPO derivation)** — the full-read chapter on [[dpo]]; read before this lab.
- **[[reward-hacking-taxonomy]]** — formalizes the length-hack as a generic proxy failure; ch-46 §5(a) uses both sources together.
- **[[simpo]] / [[orpo]]** — reference-free successors that fix the length-hack at the objective level; not the lab's objective but named as a "fix" in the memo post-mortem §6.3.
