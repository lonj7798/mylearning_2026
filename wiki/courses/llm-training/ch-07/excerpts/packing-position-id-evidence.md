---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: primary sources (no single library card; see note)
created_at: "2026-09-17"
---

# Excerpt: what packed training actually breaks when masks or position IDs are wrong

**Why this file exists.** The library card `packed-vs-unpacked-ablation` was rewritten on 2026-09-14 as a
"no verifiable primary source" notice, so chapters must not cite it. The three primary studies below were
read at the stated loci. Each item is a fact about that work only. Related library cards:
[[sequence-packing]] (Krell et al.) and [[tulu-3]].

## 1. Krell, Kosec, Crawford, Luschi — "Efficient Sequence Packing without Cross-contamination", arXiv:2107.02027 (v1 2021-07; v2)

- Setting: BERT pre-training (a bidirectional encoder with **learned absolute positional embeddings**), not
  decoder SFT. Up to 50% of tokens are padding at the studied lengths, and 89% for GLUE-CoLA at length 128
  (Abstract).
- Two adjustments make a packed model equivalent to an unpacked one: a block-diagonal attention mask, and
  **per-sequence position indices** (§3.2.1–3.2.2). For a pack of a length-2 and a length-3 sequence the
  position indices must be `[0, 1, 0, 1, 2]`; the positional bias-add is replaced by an embedding look-up,
  which requires carrying an extra per-token position input (§3.2.1).
- Ablation (§4.2.1, Fig. 4): without the **mask** adjustment, loss and accuracy "worsen drastically" and
  longer training does not recover them. Without the **position** adjustment, loss and accuracy almost match
  the baseline, but MLM accuracy stalls at **71.8%** against the **72.1%** target.
- Downstream after packed pre-training: SQuAD 1.1 F1 88.32 (base) and 90.65 (large) versus reference 88.5
  and 90.9 (§4.3, Table 3). Phase-2 speed-up at length 512 exceeds 2× (§4.2, Fig. 3).

## 2. Kundu, Lee, Wynter, Ganti, Mishra (IBM Research) — "Enhancing Training Efficiency Using Packing with Flash Attention", arXiv:2407.09105 (v1 2024-07; v6)

- Mechanism (§3.3–3.5): position IDs restart at 0 for each packed example; `cu_seq_len` is **computed from
  `position_ids`** and `flash_attn_varlen_func()` is called; the collator sets the first label of each
  example to `-100`. Shipped as `DataCollatorWithFlattening` in Transformers 4.44 and `padding_free=True`
  in TRL's `DataCollatorForCompletionOnlyLM` (§3.1).
- Setting (§4, §4.1): 20K-example subsets of FLAN, OrcaMath and The Stack; one node of 8 A100-80GB with
  FSDP; one epoch, gradient accumulation 2, maximum sequence length 4096, minibatch 4 per GPU.
- Mistral-7B on FLAN_20k (§4.1, Table 2):

  | Batching | Tokens/s | Validation loss |
  |---|---|---|
  | padding | 742 | 1.129 |
  | packing without position IDs | 2986 | 1.306 |
  | offline packing with position IDs (585 packed rows) | 3010 | 1.284 |
  | online minibatch packing with position IDs | 1408 | 1.127 |

- The authors attribute the higher validation loss of offline packing, even with position IDs, to far fewer
  optimizer steps in one epoch; minibatch packing keeps the number of steps and matches the padding loss
  (§4.1, §4.3, Tables 4–6).

## 3. Wang, Wang, Wang, Li, Hovy, Guo — "Packing Analysis: Packing Is More Appropriate for Large Models or Datasets in Supervised Fine-tuning", arXiv:2410.08081 (v1 2024-10; v3)

- Compares padding, random packing and greedy packing on LLaMA-3-8B and LLaMA-3-70B with SFT sets of 69K to
  1.2M conversations (§3–4). The paper does **not** state that attention is reset between packed
  conversations; §3.3.3 argues that the `[EOS]` token lets the model separate adjacent samples.
- Average benchmark scores over MMLU, GSM8K, MATH, BBH, IFEval and HumanEval, padding / random packing /
  greedy packing (Table 3), all 8 model-dataset settings:

  | Dataset (size) | Model | Padding | Random packing | Greedy packing |
  |---|---|---|---|---|
  | WildChat (GPT-4), 69K | LLaMA-3-8B | 49.58 | 49.46 | 50.6 |
  | WildChat (GPT-4), 69K | LLaMA-3-70B | 61.50 | 65.97 | 65.92 |
  | TULU, 326K | LLaMA-3-8B | 46.72 | 47.66 | 48.31 |
  | TULU, 326K | LLaMA-3-70B | 61.96 | 61.77 | 64.02 |
  | WildChat, 652K | LLaMA-3-8B | 48.99 | 49.03 | 50.87 |
  | WildChat, 652K | LLaMA-3-70B | 62.62 | 63.43 | 63.53 |
  | Open-source 1M, 1.2M | LLaMA-3-8B | 54.3 | 54.95 | 55.05 |
  | Open-source 1M, 1.2M | LLaMA-3-70B | 66.12 | 67.26 | 67.54 |

  Greedy packing is above padding in all 8 settings; random packing is below padding in 2 of the 8.
- Packing a single-turn-only set (filtered 200K OpenHermes 2.5) gave a significant MATH drop that returned
  to normal after adding 1/40 to 1/20 multi-turn conversations; an internal 200K single-turn set showed no
  drop (§5.3, Fig. 3). No seeds or run-to-run variance are reported.
