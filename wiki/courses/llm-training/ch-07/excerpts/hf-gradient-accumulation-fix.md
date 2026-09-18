---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: https://huggingface.co/blog/gradient_accumulation
created_at: "2026-09-17"
---

# Excerpt: "Fixing Gradient Accumulation" (Hugging Face, October 2024)

**Artifact.** Official Hugging Face blog post, authors listed as Arthur Zucker, Zachary Mueller,
Yih-Dar Shieh, Benjamin Bossan and Pedro Cuenca. Read on 2026-09-17 at
https://huggingface.co/blog/gradient_accumulation. Source type: official blog (the organization that
maintains `transformers`). No library card existed for this artifact when ch-07 was revised.

## What the post reports

- The trigger: "Our friends at Unsloth shared an issue regarding gradient accumulation yesterday that is
  affecting the transformers `Trainer`." The symptom is that "losses did not match between training runs
  where the setting was toggled on and off", although "Gradient accumulation is supposed to be
  mathematically equivalent to full batch training".
- The cause: each model class carries a *default* loss function that was never meant to be customizable.
  For causal LM it averaged the per-microbatch token loss, so accumulation averaged averages.
- The stated correctness rule: "for gradient accumulation across token-level tasks like causal LM training,
  the correct loss should be computed by the total loss across all batches in a gradient accumulation step
  divided by the total number of all non padding tokens in those batches. This is not the same as the
  average of the per-batch loss values."
- The patch, quoted from the post (`ForCausalLMLoss`, after the usual shift and flatten):

  ```python
  num_items = kwargs.pop("num_items", None)
  + loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
  + loss = loss / num_items
  - loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
  ```

- Two shipped changes: default loss functions account for gradient accumulation automatically
  (PR huggingface/transformers#34191), and a user-supplied loss function can receive the number of items
  seen per batch (PR huggingface/transformers#34198). Every class inheriting from `PreTrainedModel` gained
  a `loss_function` property selected by `config.loss_type` through `LOSS_MAPPING`.

## What the post does not report

- No benchmark numbers, no model sizes, and no measurement of the downstream effect of the bug. The post
  gives the correctness argument and the patch only. For a measured statement of the same effect, see
  [[tulu-3]] §4.3.2 (Eqs. 1–2, Figs. 5–6).
