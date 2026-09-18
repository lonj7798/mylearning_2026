---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: https://github.com/karpathy/nanochat
source_url: https://github.com/karpathy/nanochat
created_at: "2026-09-17"
revised: 2026-09 (generality revision)
---

# Excerpt: nanochat — a second minimal trainer to read, and a costed reference run

**Artifact:** `karpathy/nanochat` at commit `f527f76`; files `README.md`, `runs/speedrun.sh` (79 lines),
`scripts/chat_sft.py` (499 lines).
**Read on:** 2026-09-17 from the text cached at `scratchpad/sources/nanochat.txt`,
`scratchpad/sources/nanochat-speedrun-sh.txt` (identical to `c14a-nanochat-speedrun-f527f76.txt`), and
`scratchpad/sources/nanochat-chat_sft-py.txt`.
**Source type:** released config/code (repository README is author-written, practitioner evidence).
**No library card exists for this artifact yet**; ch-08 cites this excerpt.

---

## Why ch-08 reads this alongside TRL

`scripts/chat_sft.py` implements the same three operations the lab instruments — packing, loss masking,
optimizer step — in one readable file with no framework indirection, and it uses **different conventions** from
TRL. Reading both is what makes "verify the symbol in source" concrete rather than a slogan.

| Operation | TRL `SFTTrainer` ([[trl-sft-trainer]]) | nanochat `chat_sft.py` |
|---|---|---|
| ignore index | `-100` (collator pads labels with `-100`, L509-511) | `-1` (`targets[mask_targets == 0] = -1`, L288-290) |
| masking source | `assistant_masks` from the chat template's `{% generation %}` markers | `loss_mask` returned by `tokenizer.render_conversation`, "mask=1 for assistant completions, mask=0 for user prompts, BOS, special tokens, tool outputs" (L285-287) |
| packing | `pack_dataset(..., packing_strategy)`, BFD by default, overflow truncated | best-fit with padding: "When no conversation fits, the row is padded (instead of cropping) to ensure no tokens are ever discarded" (L180-187) |
| shift | done inside the model's loss | explicit: `inputs = batch[:, :-1]`, `targets = batch[:, 1:]` (L282-284) |
| padding positions | `-100` at padded positions | `targets[i, content_len-1:] = -1` per row (L292-296) |

## Reference run (`runs/speedrun.sh`, commit `f527f76`)

- Target hardware and time: "designed to run on a blank 8XH100 GPU node and takes approximately 1.5 hours to
  complete" (script header, L3-4; README L365 repeats ~1.5 hours).
- Tokenizer: vocabulary `2**15 = 32768`, trained on ~2B characters (L55-57).
- Pretraining: `scripts.base_train --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8`,
  followed by `scripts.base_eval` (L67-70).
- SFT: `torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft`, then `scripts.chat_eval -i sft` (L74-76).
- Cost statement in the README: "at the current ~$3/GPU/hr, an 8XH100 node is ~$24/hr, so 2 hours is ~$48"
  (README L341). The stated capability target is the DCLM CORE score of GPT-2 (1.6B), 0.256525 (README L341).

## SFT script defaults that matter for the lab (`scripts/chat_sft.py`, L44-68)

| Flag | Default | Note in file |
|---|---|---|
| `--num-iterations` | `-1` | −1 = one full epoch |
| `--init-lr-frac` | `0.8` | initial LR as a fraction of the base LR inherited from pretraining |
| `--warmup-ratio` | `0.0` | no warmup at this stage |
| `--warmdown-ratio` | `0.5` | last half of the run decays the LR |
| `--eval-every` | `200` | validation bits-per-byte every 200 steps |
| `--chatcore-every` | `200` | ChatCORE metric every 200 steps |
| `--mmlu-epochs` | `3` | "teaches Multiple Choice" |
| `--gsm8k-epochs` | `4` | "teaches Math and Tool Use" |

The training mixture is `TaskMixture` over SmolTalk, MMLU and GSM8K (L28-31), and the loss on a held-out split
is reported as bits-per-byte through `nanochat.loss_eval.evaluate_bpb` (L22, L60). Both properties are used by
ch-08: an in-loop general measurement, and a metric that is comparable across tokenizers.

## Limits

The README is written by the repository author and reports no controlled comparison against other codebases.
The speedrun numbers are wall-clock statements for one hardware configuration, not measured throughput
distributions. The model produced is described in the README as "a 4e19 FLOPs capability model".

## Connections

- [[trl-sft-trainer]] — the other trainer the lab reads.
- [[paloma]] — bits-per-byte as the cross-tokenizer comparable loss metric.
