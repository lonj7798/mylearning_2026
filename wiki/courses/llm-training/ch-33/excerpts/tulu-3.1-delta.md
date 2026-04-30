---
chapter: ch-33
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/tulu-3.1.md
source_url: https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B
created_at: "2026-04-23"
---

# Excerpt: Tülu 3 → Tülu 3.1 — a single-stage RL-algorithm swap

**Source library:** `wiki/raw-data/llm-training/model-reports/tulu-3.1.md` (HF model-card-only release)
**Artifact:** Isolated PPO → GRPO ablation on `allenai/Llama-3.1-Tulu-3-8B-DPO`

---

## Why this source anchors ch-33

Ch-33 §2 is built on the fact that Tülu 3.1 is the cleanest public *controlled ablation* in open post-training: everything is held fixed except the final RL algorithm. Most open releases change base, data, and algorithm together; Tülu 3.1 changes one algorithmic dimension at a time. This excerpt pins what changed, what did not, and why the ablation is rare.

---

## What the source explicitly says changed

From the source (lines 37–42):

> Allen AI says the new version comes from an improvement **only in the final RL stage of training**.
> The final stage switched from **PPO** to **GRPO**.
> The model card also states **no reward model** is used in that final stage.
> Additional **hyperparameter tuning** in the RL stage produced better average results than the original 8B Tulu 3 checkpoint.

The "only change" framing is attested; it is not my inference. Ai2 chose to release this as a model-card-only update rather than a paper precisely to communicate "nothing else moved".

## What the source explicitly says stayed the same

From the source (lines 44–47):

> Earlier **SFT** and **DPO** stages remain those of **Tülu 3**.
> The model card still frames the data as a mix of **publicly available, synthetic, and human-created datasets**.
> The associated training dataset shown in the card is **`allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`**.

So the 939K SFT mix, the ~270K DPO pool, and the base model `Llama-3.1-Tulu-3-8B-DPO` are all unchanged. Only the final RL stage's algorithm, hparams, and training set are rebuilt.

---

## Why it matters

From the source (lines 49–51):

> Tulu 3.1 is useful because it is a **controlled public ablation** rather than a broad marketing release.
> It also shows that the open ecosystem quickly incorporated the **GRPO / verifier-style RL** trend after DeepSeek-R1 and DeepSeekMath made it prominent.

The ecosystem signal is the more durable one: PPO was the default for LLM RL from the InstructGPT era through 2023. DeepSeekMath / DeepSeek-R1 reframed GRPO (Group Relative Policy Optimization) as the cheaper alternative — no value network, advantage from group-return baselines. Tülu 3.1 is the earliest public open-recipe model to adopt GRPO in a labelled head-to-head with PPO on the *same* base, data, and DPO checkpoint. That is the ablation structure that lets the community attribute any future PPO-vs-GRPO claim.

---

## The "3.1 as multi-base refresh" alternative framing

The companion source [[tulu-3-1]] (note the hyphenated slug — a different file) gives a *second* meaning of "3.1": the Ai2 blog uses "Tülu 3.1" to mean the multi-base refresh that re-runs the Tülu 3 recipe on Llama 3.1 AND OLMo 2 bases. Both framings are current; both are documented. Ch-33 §2.1 uses the **HF-card narrative** because it is the one that names the algorithmic change; §2.2/§2.3 integrate the multi-base framing where relevant.

---

## What ch-33 keeps from this source

- The "only the final RL stage changed" claim (§2.1).
- PPO → GRPO as the specific swap (§2.1).
- The no-RM inheritance (§2.1).
- The hparam-retuned-but-not-published note (§2.3 delta table).
- The `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` training-set name (§2.3 delta table).
- The "controlled ablation" framing (§2.2).

---

## Connections

- **ch-33 §2** — where this excerpt is cited.
- **[[grpo]]** — the algorithmic lens for the swap; the cheaper-than-PPO value-network-free variant.
- **[[deepseek-r1]]** — the nearby public family that normalized GRPO-style reasoning RL.
- **[[tulu-3]]** — the full underlying recipe; everything except the final stage is inherited from here.
- **ch-34 (OLMo 2/3)** — OLMo 2 applies the Tülu 3 recipe directly; OLMo 3 builds on top of it.
