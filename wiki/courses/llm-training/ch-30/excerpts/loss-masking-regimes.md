---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: arXiv:2405.14394v2; arXiv:2401.13586v4; github.com/allenai/open-instruct@098424c; github.com/huggingface/trl@v0.23.0
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from primary sources)"
---

# Excerpt: Loss masking choices — instruction loss, prompt loss weight, and framework defaults

This excerpt quotes four primary artifacts used by [[read]] §1-§2 and §7. The library card [[loss-masking-prompt]] has not yet been corrected and states the opposite of the Shi et al. result; the quotes below are taken from the paper.

## 1. Shi et al., "Instruction Tuning With Loss Over Instructions" (arXiv:2405.14394v2, NeurIPS 2024)

Abstract (verbatim): "we propose a simple yet effective method, INSTRUCTION MODELLING (IM), which trains LMs by applying a loss function to the instruction and prompt part rather than solely to the output part. Through experiments across 21 diverse benchmarks, we show that, in many scenarios, IM can effectively improve the LM performance … We observe that IM is especially beneficial when trained on datasets with lengthy instructions paired with brief outputs, or under the Superficial Alignment Hypothesis (SAH) where a small amount of training examples are used for instruction tuning. … It is worth noting that we are not proposing IM as a replacement for current fine-tuning processes."

Loss definitions (§3): IT, Eq. 2: `L = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})`. IM, Eq. 4: `L = − Σ_{t=1..m+n} log P(x_t | x_1..x_{t−1}) · 1(x_t ∉ T)`, "where 1(x_t ∉ T) is an indicator function that is 1 if x_t is not a template token".

Table 1 (LLaMA-2-7B base; selected rows; NLP mean of 18 tasks / MT-Bench / AlpacaEval 1.0 / AlpacaEval 2.0):

| Dataset (examples) | IT | NEFTune | IM |
|---|---|---|---|
| Alpagasus Alpaca 5k (5,305) | 45.29 / 3.62 / 16.29 / 2.46 | 45.62 / 3.50 / 21.37 / 2.37 | 47.47 / 3.48 / 19.52 / 3.29 |
| Alpagasus Dolly 3k (2,996) | 46.58 / 4.23 / 13.42 / 2.00 | 47.07 / 4.42 / 14.04 / 2.03 | 48.95 / 4.06 / 15.11 / 2.44 |
| Alpagasus Dolly 9k (9,229) | 45.54 / 4.33 / 21.54 / 2.28 | 45.99 / 4.21 / 31.61 / 2.84 | 48.00 / 4.55 / 30.77 / 2.67 |
| Less Tydiqa (13,533) | 48.21 / 4.08 / 5.12 / 1.88 | 47.47 / 4.19 / 8.35 / 2.58 | 48.70 / 4.36 / 10.10 / 2.88 |
| Less MMLU Chat (13,533) | 47.18 / 3.86 / 4.42 / 1.20 | 47.74 / 4.06 / 6.22 / 1.06 | 47.84 / 4.54 / 9.78 / 1.93 |
| Less BBH ICL (13,533) | 48.28 / 4.78 / 36.20 / 2.36 | 49.03 / 5.05 / 39.81 / 2.87 | 49.15 / 5.03 / 44.15 / 3.56 |
| LIMA (1,030) | 48.79 / 4.77 / 33.06 / 2.58 | 48.70 / 4.79 / 30.51 / 2.43 | 49.60 / 4.83 / 32.94 / 2.47 |

Base model: 49.32 / 1.16 / 0.01 / 0.01. Instruction/output length ratios (Table 5): Less MMLU Chat 26.33, Less Tydiqa 5.86, Less BBH ICL 3.27, Dolly 3k 0.64, Alpaca 5k 0.57, Dolly 9k 0.30, LIMA 0.09.

Overfitting analysis (§4.3): "In the training loss distribution for the LIMA dataset, IM exhibits a slightly higher mean loss of 1.45 compared to 1.37 for IT … IM demonstrates a lower mean test loss of 1.17 compared to 1.32 for IT" (Tülu V2, 10% sample; loss on output tokens only). Table 2 BLEU against training outputs, IT → IM: LIMA 18.15 → 17.30, Less Tydiqa 69.21 → 65.63, Less MMLU Chat 72.43 → 69.20, Less BBH ICL 60.96 → 53.94, Alpaca 5k 72.26 → 70.50, Dolly 9k 61.76 → 60.61, Dolly 3k 60.99 → 59.04. Table 3 (KL loss to the base model): LIMA NLP 48.79 → 49.26 and AlpacaEval 2.0 2.58 → 0.06; Dolly 9K NLP 45.54 → 49.31 and AlpacaEval 2.0 2.28 → 0.04.

Settings (App. C, Table 6): LLaMA-2-7B, LLaMA-2-13B, OPT-6.7B; total batch 128; epochs 2, 3, or 10 ("Training typically proceeds for 2 epochs"); maximum length 2048; LR 2×10⁻⁵; AdamW, ε 1e-6, betas (0.9, 0.98); linear schedule, warmup 0.03; weight decay 0; bf16; code built on Open-Instruct.

## 2. Huerta-Enochian and Ko, "Instruction Fine-Tuning: Does Prompt Loss Matter?" (arXiv:2401.13586v4)

Abstract (verbatim): "We found that performance of models fine-tuned on short-completion data had a statistically-significant negative quadratic relationship with PLW. Using small values (0.01 − 0.5) of PLW produced better results on multiple-choice and short-generation benchmarks (outperforming models fine-tuned on long-completion data) while large values (≈ 1.0) of PLW produced better results on long-generation benchmarks."

Contributions list: "We verified that PLW can be safely ignored when fine-tuning on long-completion data."

Definitions (§2.1): the generation ratio R_g is "the ratio of completion length to prompt length"; data with R_g < 1 are short-completion data. §4: 10 PLW levels in [0, 1] ("PLW = 0.0 is identical to the masking used in the original Alpaca project, and PLW = 1.0 is equivalent to unmasked training"), LLaMA 1 7B and LLaMA 2 7B, AlpacaData (R_g 3.27), AlpacaDataCleaned (7.83), AlpacaDataShort (0.08), "a total of sixty experimental training runs", thirteen benchmarks, original Alpaca code and hyperparameters.

## 3. open-instruct at commit 098424c (released code)

`open_instruct/finetune.py` L132-133:

```python
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
```

`open_instruct/dataset_transformation.py` L1214-1218:

```python
def _trainable_assistant_indices(messages: list[dict[str, Any]], last_turn_only: bool) -> list[int]:
    assistant_indices = [idx for idx, m in enumerate(messages) if m["role"] == "assistant"]
    if last_turn_only:
        return assistant_indices[-1:]
    return assistant_indices
```

`sft_tulu_tokenize_and_truncate_v1` (L1554-1564) calls the tokenizer path with the default `last_turn_only=False`; `last_turn_tulu_tokenize_and_truncate_v1` (L1565-1575) passes `last_turn_only=True` and is documented as "training only on the final assistant turn".

Truncation, L1427-1428 and L1464-1468:

```python
DEFAULT_OVER_LENGTH_STRATEGY = "keep"
OVER_LENGTH_STRATEGIES = (DEFAULT_OVER_LENGTH_STRATEGY, "terminate", "drop")
...
    """Handle a conversation that `max_seq_length` truncation cut short.

    Right-sided truncation drops the trailing EOS, so a cut inside an assistant turn leaves
    trainable text with no terminator. `keep` leaves the row as is, `terminate` replaces its
    final token with a trainable EOS, `drop` masks it out so `sft_tulu_filter_v1` removes it.
```

Underivable spans, L1499-1504: "Tokenize one conversation, masking the whole row out if its labels are underivable. Such rows are rare (~0.005% of tulu-3-sft-olmo-2-mixture) …". This commit is from 2026; the code used for the 2024 Tülu 3 run was not checked.

## 4. TRL v0.23.0, `trl/trainer/sft_config.py` L85-94 (released code)

```
completion_only_loss (`bool` or `None`, *optional*, defaults to `None`):
    Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed
    only on the completion, which is supported only for [prompt-completion](#prompt-completion) datasets. If
    `False`, loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset:
    loss is computed on the completion for [prompt-completion](#prompt-completion) datasets, and on the full
    sequence for [language modeling](#language-modeling) datasets.
assistant_only_loss (`bool`, *optional*, defaults to `False`):
    Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed only
    on the assistant responses, which is supported only for [conversational](#conversational) datasets. If
    `False`, loss is computed on the entire sequence.
```

The file contains no `train_on_response_only` field. `train_on_responses_only` is not a TRL option at this version.

## Connections

- [[read]] §1 (mask table), §2 (instruction loss and PLW), §7 (truncation, span derivation).
- [[smol-training-playbook]] — SmolLM3 user-turn masking result and TRL `{% generation %}` masks.
- [[neftune]] — the NEFTune baseline compared in Table 1.
