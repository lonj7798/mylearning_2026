<!-- scope: practical "how to run it" reference for on-/off-policy distillation via TRL GKDTrainer + the cross-tokenizer GOLD recipe ("any model family")
     deps:
     see-also: agarwal-gkd, tm-on-policy-distillation -->
# TRL GKD/GOLD Recipe: Running On-Policy Distillation (incl. Any Model Family)

- **Core Insight:** TRL turns on-policy distillation into three knobs — `lmbda` (student data fraction / on-policy ratio), `beta` (forward↔reverse KL interpolation of generalized JSD), and `temperature` — and the GOLD extension removes the shared-tokenizer requirement so any student can be distilled from any teacher family.
- **Guideline:** For same-tokenizer pairs use `GKDTrainer` with high `lmbda` (on-policy) and tune `beta` per task; for cross-tokenizer / mixed-family pairs use `GOLDTrainer` with `use_uld_loss=True` and the teacher's tokenizer supplied via `teacher_tokenizer_name_or_path`.
- **Source:** TRL docs https://huggingface.co/docs/trl/gkd_trainer (v1.7.1) and https://huggingface.co/docs/trl/gold_trainer ; HF Space "Unlocking On-Policy Distillation for Any Model Family" https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation (rendered: https://huggingfaceh4-on-policy-distillation.hf.space/); underlying method paper [[agarwal-gkd]] (Agarwal et al. 2306.13649).
- **Relevant chapters:** ch-impl / ch-tooling, ch-on-policy, ch-cross-tokenizer

## The HF Space / interactive article (part a & b)
The Space **"Unlocking On-Policy Distillation for Any Model Family"** by **HuggingFaceH4** is an interactive research article, **not a Gradio app**: its README frontmatter declares `sdk: docker` (an Astro-based "Research Article Template"). **NOTE:** the Space's static `/blob/main/README.md` is a generic template with no methodology; the actual prose lives in the rendered app.
- **NOTE:** the raw Space URL returned only a loading shell ("Fetching metadata from the HF Docker repository…", 115 likes) — no static content. Substance below is quoted from the rendered `.hf.space` article and corroborated by the TRL `gold_trainer` doc.
- Problem framed verbatim: on-policy distillation carried "the requirement that the teacher and student models must share the *same* tokenizer vocabulary."
- Method: "General On-Policy Logit Distillation (GOLD)" extends Universal Logit Distillation (ULD) to the on-policy setting, fixing ULD's "two weaknesses: token alignment in step 3 and logit alignment in step 4."
- Instead of truncating, GOLD "identifies the token merges required to equalise the sequence lengths for both tokenizers" and merges "the probabilities at the corresponding positions by multiplying the marginal distribution by scalar conditional probabilities."
- Reported result: GOLD "recovered 60% of the teacher's performance" vs. ULD's 10%, and "outperformed GRPO by 20%" in cross-tokenizer scenarios.

## GKDTrainer — the three knobs (part c, verbatim TRL docs)
`GKDTrainer` "is a wrapper around the `SFTTrainer` class that takes in a teacher model argument." Teacher serves per-token logits/logprobs over the student's (or its own) sequences each step; the loss is generalized JSD against those teacher logits.
- **`lmbda`** (default `0.5`) — "controls the student data fraction, i.e., the proportion of on-policy student-generated outputs. When `lmbda=0.0`, the loss reduces to supervised JSD… When `lmbda=1.0`, the loss reduces to on-policy JSD, where the student generates output sequences and token-specific feedback on these sequences from the teacher. For values in between [0, 1] it is random between the two based on the `lmbda` value for each batch."
- **`beta`** (default `0.5`) — "controls the interpolation in the generalized Jensen-Shannon Divergence. When `beta=0.0` the loss approximates forward KL divergence, while for `beta=1.0` the loss approximates reverse KL divergence." (Config note phrases it as beta=0.0 → KL, beta=1.0 → Inverse KL.)
- **`temperature`** (default `0.9`) — "Temperature for sampling. The higher the temperature, the more random the completions."
- **`seq_kd`** (default `False`) — Sequence-Level KD, "supervised FT on teacher-generated output"; with `seq_kd=True, lmbda=0.0` the teacher generates the sequences and the student gets token-level feedback.
- Other defaults: `max_new_tokens=128`, `disable_dropout=True`, optional `teacher_model_name_or_path`.
- Guidance quote: "The authors find that on-policy data (high `lmbda`) performs better and the optimal `beta` varied depending on the task and evaluation method."
- **Gotcha (verbatim):** for Gemma2 set `attn_implementation="kernels-community/flash-attn2"` "Otherwise you will encounter NaNs in the logits due to the soft capping technique."
- Minimal API: `GKDTrainer(model=..., teacher_model=..., args=GKDConfig(...), processing_class=tokenizer, train_dataset=..., eval_dataset=...)` on ChatML `messages` data. Example: student `Qwen/Qwen2-0.5B-Instruct`, teacher `Qwen/Qwen2-1.5B-Instruct`.

## GOLDTrainer — the "any model family" cross-tokenizer recipe (part c)
`GOLDTrainer` "is an extension of Universal Logit Distillation (ULD) that supports student/teacher pairs with different tokenizers… enables cross-tokenizer knowledge distillation, including mixed model families (for example, LLaMA students with Qwen teachers)." It subclasses `SFTTrainer` and "inherits the on-policy vs. off-policy scheduling from the `GKDTrainer`" (so `beta`, `lmbda`, `seq_kd` carry over). Namespace: `trl.experimental.gold` (API may change).
- **Cross-tokenizer alignment (verbatim):** GOLD "incrementally decodes the student and teacher tokens, groups passages with the same visible text, and merges probabilities inside each group. This guarantees loss terms are computed over the full completion even when token boundaries differ."
- **Probability merging math:** for a teacher token run `[token₀…tokenₖ]` mapping to one student token,
  `P_merged(y) = P(y | context) × P(token₁ | token₀, context) × … × P(tokenₖ | …, context)` — the first-position marginal distribution is scaled by the **scalar** conditional probabilities of the actual continuation tokens (chain rule → correct joint for the realized sequence). Result is intentionally **unnormalized**; ULD loss uses sorting + L1 distance so normalization is unnecessary.
- **Hybrid ULD loss (verbatim):** with `uld_use_hybrid_loss` "GOLD compares exact vocabulary matches directly and falls back to the original sorted-probability ULD loss for unmatched tokens." (Article: GKD loss for one-to-one-mapped tokens, ULD loss for unmatched.)
- **Key GOLDConfig flags:** `use_uld_loss` (True for cross-tokenizer); `teacher_tokenizer_name_or_path` ("required when `use_uld_loss=True`; GOLD uses the teacher tokenizer to align tokens"); `uld_use_hybrid_loss`, `uld_hybrid_matched_weight`, `uld_hybrid_unmatched_weight`; inherited `beta`/`lmbda`/`seq_kd`; `num_generations`, `generation_batch_size` (buffered rollout gen across grad-accum). **Default `learning_rate=1e-7`** (vs 5e-5).
- Teacher logprobs: teacher model forward-passes over the aligned completion to emit per-token logits; GOLD merges those into a student-comparable distribution (above) rather than requiring vocab identity.
- **Typical hyperparameters (example script):** `--learning_rate 2e-5 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --num_train_epochs 1`; student `Llama-3.2-1B-Instruct`, teacher `Qwen2.5-0.5B-Instruct` / `Qwen2-1.5B-Instruct`. Full training and LoRA both supported via `ModelConfig` flags.

## Trade-offs
- `lmbda`↑ = on-policy = student trains on its own mistakes (fixes train/inference distribution mismatch) but costs a generation pass per step; `lmbda=0` is cheap supervised KD on teacher probs.
- `beta` picks mode-covering (forward KL) vs. mode-seeking (reverse KL); no universal best — task-dependent.
- GOLD unlocks any-family pairs but adds tokenizer-alignment overhead and an experimental, fast-moving API; unmatched-vocab regions rely on the coarser sorted-probability ULD term.

## Connections
- Method origin & JSD/on-policy theory: [[agarwal-gkd]] (On-Policy Distillation of Language Models, Agarwal et al. 2306.13649). Mechanism/numbers: [[tm-on-policy-distillation]].
- ULD foundation GOLD extends: Boizard et al., "Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs" (arXiv:2402.12030).
