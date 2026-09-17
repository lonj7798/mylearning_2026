---
chapter: ch-42a
course: llm-training
phase: read
excerpt_of: arXiv:2507.21509v3 (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2507.21509
created_at: "2026-09-15"
---

# Excerpt: Persona Vectors — Monitoring and Controlling Character Traits in Language Models

- **Authors:** Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey (Anthropic Fellows Program; UT Austin; Constellation; Truthful AI; UC Berkeley; Anthropic)
- **Year:** 2025 (arXiv v1 2025-07; PDF read here is v3, 5 Sep 2025)
- **Source type:** paper

## Extraction pipeline (§2)
- Input: a trait name and a one-line description (for example evil: "actively seeking to harm, manipulate, and cause suffering").
- Claude 3.7 Sonnet generates 5 pairs of contrastive system prompts, 40 evaluation questions (half for extraction, half for evaluation), and a judging rubric; GPT-4.1-mini scores trait expression 0-100.
- 10 rollouts per question and system prompt; responses kept only when the trait score agrees with the intended prompt (> 50 for positive, < 50 for negative); residual-stream activations averaged over response tokens.
- "We then compute the persona vector as the difference in mean activations between responses that exhibit the trait and those that do not." One candidate vector per layer; the layer is chosen by steering effectiveness.
- Models studied: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Main traits: evil, sycophancy, hallucination.

## Steering and monitoring (§3, §5)
- Steering at decoding time: `h_ℓ ← h_ℓ + α · v_ℓ` (§3.2), where `h_ℓ` is the residual-stream activation at layer ℓ, `v_ℓ` the persona vector at that layer, and `α` a scalar coefficient. Post-hoc suppression uses `h_ℓ ← h_ℓ − α · v_ℓ` (§5.1).
- Prompt-induced shifts: projection of the last prompt-token activation onto the persona vector "correlate[s] strongly with trait expression in subsequent responses (r = 0.75–0.83)", but "these correlations arise primarily from distinguishing between different prompt types … with more modest correlations when controlling for prompt type".
- Base-model trait scores before fine-tuning: "0 (evil), 4.4 (sycophancy), and 20.1 (hallucination)" (Figure 7 caption).
- Inference-time steering reduces trait expression but "large steering coefficients tend to degrade accuracy" on MMLU (Figure 7A).
- Preventative steering adds the undesired persona vector during fine-tuning: "we proactively steer the model toward the undesired persona direction during training, relieving the model of the need to shift in that direction to fit the training data." It "effectively reduces training-induced persona shifts, while also maintaining an average coherence score across all models above 80" and "better preserves the model's general capabilities compared to inference-time steering, as measured by MMLU accuracy". Multi-layer steering is more effective (App. J.3); a regularization loss on projections "to be ineffective in practice" (App. J.5); CAFT (zero-ablating directions during training) "is effective at preventing evil and sycophancy, but ineffective for hallucinations" (App. J.4).

## Fine-tuning-induced shifts (§4)
- Datasets: trait-eliciting (evil, sycophancy, hallucination) and "EM-like" domain-flawed sets (incorrect medical advice, political opinions with flawed arguments, invalid math solutions, insecure code, GSM8K with mistakes, opinions), each in Normal, I (mild/subtle) and II (overt/severe) versions.
- "datasets targeting one trait (e.g., evil) can inadvertently amplify other traits (e.g., sycophancy or hallucination). EM-like datasets that contain subtle flaws can induce persona changes even in the absence of explicit corresponding behaviors in the data; for example, training on flawed math reasoning increases expression of evil."
- Finetuning shift = projection onto the persona vector of the difference between post- and pre-finetuning mean activations at the last prompt token. "We observe strong positive correlations (r = 0.76–0.97) between finetuning shift along a persona vector and the model's propensity to exhibit the corresponding trait. Notably, these correlations are higher than cross-trait baselines (r = 0.34–0.86)."
- Caveat stated by the authors (footnote 6): "negative traits (and, surprisingly, humor) tend to shift together, and opposite to the one other positive trait we tested (optimism)."

## Data screening before fine-tuning (§6)
- Projection difference for a dataset D = {(x_i, y_i)}:
  ΔP = (1/|D|) Σ_i [a_ℓ(x_i, y_i) − a_ℓ(x_i, y_i′)] · v̂_ℓ,
  where a_ℓ(x, y) is the mean activation over response tokens at layer ℓ, y_i′ is the base model's own response to prompt x_i, and v̂_ℓ is the unit-normalized persona vector. "dataset-level projection difference is highly predictive of post-finetuning trait expression."
- Sample level (Figure 9): "individual samples from trait-inducing datasets are highly separable from control samples based on their projection values".
- Real data (§6.3, Figure 10): from LMSYS-Chat-1M, the top 500, bottom 500 and 500 random samples by projection difference are each used for fine-tuning. "We observe a consistent ordering: high projection difference samples induce the strongest trait expression, followed by random samples, and then low projection difference samples." After an LLM filter removes samples with trait expression score above 1, "high projection difference samples continue to induce stronger trait expression than random samples". Surfaced examples include roleplay requests (sycophancy) and underspecified queries such as "Keep writing the last story" (hallucination).

## Training settings
- Main fine-tuning (App. D.3): "we finetune the model for one epoch using rs-LoRA with a rank of 32, scaling factor α = 64, and a learning rate of 10⁻⁵. We set the per-device training batch size to 2 and use gradient accumulation with 8 steps. All experiments are conducted on a single NVIDIA H100 GPU."
- Real-data screening runs (App. L.1): 500 samples, 10 epochs, LoRA rank 32, α 64, learning rate 1e−5, linear schedule, batch size 16.

## Verification
- Checked on 2026-09-15 against the arXiv PDF text of 2507.21509v3: Abstract, §1-§7, App. D.3, App. L.
- Not reported by the source: numeric values behind Figures 5-10 (figures only); results for models larger than 8B; whether preventative steering transfers to RL.
