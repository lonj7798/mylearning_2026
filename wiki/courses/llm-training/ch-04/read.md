<!-- chapter: ch-04
     track: foundations
     kind: content
     title: Sequence Packing, Loss Masking, and Chat Templates
     deps: [ch-00]
     sources: [[sequence-packing]], [[fewer-truncations-best-fit-packing]], [[sequence-composition-intra-doc-masking]], [[packing-flash-attention-position-ids]], [[packing-analysis-sft]], [[loss-masking-prompt]], [[prompt-loss-weight-sift]], [[neftune]], [[prompt-format-sensitivity-formatspread]], [[mind-your-format-templates]], [[flan-collection]], [[hf-alignment-handbook]], [[trl-sft-packing-masking]], [[transformers-chat-templates]], [[open-instruct-sft-masking]], [[open-instruct-allenai-recipes]], [[llama-3]], [[llama-3-recipe]], [[prolong]], [[prolong-recipe]], [[longalign]], [[string-effective-context]], [[tulu-3]], [[llama-2-long]], [[openr1]], [[openr1-recipe]], [[deepseek-v3-recipe]], [[deepseek-v4-recipe]], [[tulu-1-how-far-can-camels-go]], [[nemotron-4-synthetic-recipe]]
     figures: figures/packed-attention.html
     revised: 2026-09 (generality revision)
-->

# Chapter 4 — Sequence Packing, Loss Masking, and Chat Templates

> **Core insight.** Packing, loss masks, and chat templates decide which tokens a token may attend to, which tokens receive gradient, and which token ids mark the structure of a conversation. None of these choices raises an error, and each has a measured effect on breadth: packing that avoids truncation raised reading-comprehension and context-following scores in 13B text models and cut undefined-name errors in a 7B code model by up to 58.3% ([[fewer-truncations-best-fit-packing]]), intra-document attention raised in-context-learning accuracy at 1.3B ([[sequence-composition-intra-doc-masking]]), and adding loss on instructions reduced overfitting on short-output or small SFT sets ([[loss-masking-prompt]], [[prompt-loss-weight-sift]]). Models remain sensitive to formatting at evaluation time, with accuracy spreads of up to 76 points across equivalent prompt formats ([[prompt-format-sensitivity-formatspread]]).
>
> **Guideline.** When examples are shorter than the sequence length, pack them with a best-fit algorithm and intra-document attention, because this removes padding without truncating examples and lowered no evaluated score significantly in [[fewer-truncations-best-fit-packing]]. When the data is short-completion (completion shorter than prompt) or small (thousands of examples), treat the prompt-loss weight as a tuned hyperparameter instead of fixing it at 0, because [[prompt-loss-weight-sift]] fitted an optimum of 0.242 on such data and [[loss-masking-prompt]] measured less overfitting with loss on instructions. Otherwise, mask prompt tokens as Llama 3 and Tülu do, and verify that the assistant end-of-turn token is inside the loss mask. When a model passes through more than one training stage or is served from a separate tokenizer configuration, render the same conversation with the training, preference, RL, and serving templates and compare token ids before each stage, because OLMo 3 7B Think's DPO and RL stages used a slightly different template from its SFT stage through "a minor miscommunication", and the difference is in the released models ([[open-instruct-allenai-recipes]]). When a model is compared with another model or checkpoint, evaluate it under several prompt formats, because rankings reversed under another format in [[prompt-format-sensitivity-formatspread]].

## Why this chapter matters for a general-purpose model

A training run consumes token ids, an attention pattern, and a per-token loss weight. This chapter covers the three steps that produce them from raw documents or conversations. They apply in every stage of the pipeline: packing and document masking in pre-training and long-context continued pre-training (ch-32b), loss masks and templates in SFT (ch-30), formatting-token masks in preference optimization (ch-39), observation masks in agentic RL (ch-45b), and format sensitivity in evaluation (ch-47a).

The problems are measurable. Cross-document attention adds content from unrelated documents to each token's context. Truncation removes the context that later tokens depend on. A prompt mask removes most of the loss signal from short-completion data, which the cited studies associate with overfitting. A template mismatch changes the ids that start and end a turn, so a trained model may not emit the stop token the server expects. For a general-purpose model the question in each case is whether the choice improves behaviour on tasks and formats that were not in the training mixture, and ch-00 defines how that is measured.

## §1 Packing: removing padding without truncating examples

**Definition.** Packing places several variable-length examples into one fixed-length row, so that the row contains few or no padding tokens. A padding token is a filler id that is excluded from attention and loss.

**Problem.** Padding to the longest example in a batch wastes compute. In BERT pre-training on Wikipedia at sequence length 512, 50% of tokens were padding, and 89% for the GLUE CoLA task at length 128 ([[sequence-packing]], abstract; these are BERT data, not chat data).

**Mechanism.** There are three common layouts.
1. Padding: one example per row, padded to the longest example in the batch.
2. Concatenate-and-split ("wrapped"): concatenate all examples and cut every L tokens. No padding remains, but examples that cross a cut are split into unrelated rows.
3. Bin packing: treat each example (or each chunk of an example longer than L) as an item, and assign items to rows of capacity L without splitting them. First-fit decreasing (FFD) sorts items by length and places each into the first row with space. Best-fit decreasing (BFD) places each into the row whose remaining space is smallest but sufficient. Krell et al.'s shortest-pack-first histogram packing (SPFHP) instead uses worst-fit: each length bin goes to the pack with the most remaining space ([[sequence-packing]], §3.1.1).

**Formula.**

```
EFF = (Σ_i L_i) / (R · L_max)          potential speed-up ≈ EFF_packed / EFF_padded
```

where `L_i` is the length of example i, `R` is the number of rows produced, and `L_max` is the row length. EFF is the packing efficiency, the fraction of non-padding positions.

**Worked example.** A batch has examples of 100, 300, 600, and 1,000 tokens. Padding to 1,000 gives 4 rows and 4,000 positions for 2,000 real tokens, so EFF = 0.50. BFD into rows of 2,048 puts all four into one row: EFF = 2,000 / 2,048 = 0.977, and 48 positions are padding.

Now take the example in Figure 1 of [[fewer-truncations-best-fit-packing]]: documents of 14, 7, 5, 2, and 3 tokens with L = 8. Concatenate-and-split produces rows [0,8), [8,16), [16,24), [24,31). The 14-token, 7-token, and 5-token documents each cross a cut, so 3 of 5 documents are truncated. BFD first splits the 14-token document into chunks of 8 and 6, then places items in descending order: 8 → row A (0 left); 7 → row B (1 left); 6 → row C (2 left); 5 → row D (3 left); 3 → row D, the row with the smallest sufficient space (0 left); 2 → row C (0 left). Both layouts use 4 rows, and BFD truncates only the document that is longer than L.

**Evidence.**
- Efficiency: on BERT Wikipedia data, efficiency was 50.0% without packing, 89.4% for SPFHP with at most 3 sequences per pack, and 99.7% for NNLSHP (non-negative least squares histogram packing), with an estimated speed-up of 1.913 on IPUs for NNLSHP; the measured total speed-up in phase-2 training exceeded 2x ([[sequence-packing]], Table 1, §4.2, Fig. 3). **Result (single study).**
- Compactness at scale: BFD added 0.0024% more rows than concatenation on RefinedWeb at 2,048 tokens and 0.0028% on the Stack at 2,048 tokens; the optimized BFD took 10,816 s for 1B documents against 26,354 s for FFD ([[fewer-truncations-best-fit-packing]], Tables 1-2).
- Capability: pre-training 13B models on 500B RefinedWeb tokens (2k and 8k context) and a 7B code model on 300B Stack tokens, BFD packing raised the reading-comprehension average from 46.71 to 48.92 (2k) and 47.13 to 49.03 (8k), raised NQ-Swap context following from 45.62 to 51.03 (2k), and reduced undefined-name errors in generated code by up to 58.3%; no task showed a statistically significant decrease ([[fewer-truncations-best-fit-packing]], §4, Tables 4-5). **Result (single study).**
- Framework default: TRL's `SFTConfig` at commit aa89588 sets `packing=False` and `packing_strategy="bfd"`, where `"bfd"` truncates examples longer than `max_length`, `"bfd_split"` splits them, and `"wrapped"` cuts mid-sequence ([[trl-sft-packing-masking]], `trl/trainer/sft_config.py` L77-82; `trl/data_utils.py` L859-866).

**Conditions and limits.** Packing changes more than throughput. With a fixed number of examples per epoch, offline packing reduces the number of optimizer steps: Mistral-7B trained for one epoch on a 20K FLAN subset reached validation loss 1.284 with offline packing (585 training rows) against 1.129 with padding (19,961 rows), while packing within each minibatch kept the step count and reached 1.127 ([[packing-flash-attention-position-ids]], Table 2). In SFT of LLaMA-3-8B/70B, greedy packing averaged above padding in all 8 model-dataset settings and random packing averaged below padding in 2 of 8, and the linear batch-size to learning-rate relation observed with padding did not hold with packing ([[packing-analysis-sft]], Table 3, §5.3). Open R1 fine-tuned Qwen2.5-Coder Instruct models on R1 reasoning traces for C++ competitive-programming problems and found packing "consistently worse" on LiveCodeBench (Python) across all datasets it ablated; the authors' stated hypothesis is that long traces were clipped across chunk boundaries, separating questions from answers, and they suspect the C++-to-Python mismatch enlarged the gap ([[openr1]], Update #3 Lesson 1). That packing implementation allowed samples to overlap chunk boundaries, which is the concatenate-and-split layout, not BFD.

**Implication for a general-purpose model.** The measured benefit comes from keeping examples whole. When a corpus contains documents or traces longer than L, a packing strategy that discards or splits their tails removes the long-range dependencies that context-following and long-generation tasks need.

## §2 Attention across documents: causal masking versus intra-document masking

**Definition.** With causal masking over a packed row, each token attends to all earlier tokens in the row, including earlier documents. With intra-document causal masking, each token attends only to earlier tokens of its own document; the attention mask is block-diagonal. The two likelihoods are

```
causal:          P(C) = Π_{t=1..|C|} P(x_t | x_1, …, x_{t−1})
intra-document:  P(C) = Π_{i=1..n} Π_{j=1..|d_i|} P(d_ij | d_i1, …, d_i(j−1))
```

where `C` is a packed row of length `|C|`, `x_t` is its t-th token, `d_i` is the i-th of `n` documents in the row, and `d_ij` is the j-th token of document i ([[sequence-composition-intra-doc-masking]], §2.2).

**Problem.** Tokens from an unrelated document enter the softmax normalization of every attention head. The model is trained to predict tokens while conditioning on text that has no relation to them.

**Mechanism with a numeric example.** A query has attention scores 2 and 1 to the two earlier tokens of its own document. With a block-diagonal mask the weights are e²/(e²+e¹) = 0.731 and 0.269. If the mask is missing and one token of the previous document also scores 2, the denominator becomes e²+e¹+e² = 17.50, and the weights become 0.422, 0.155, and 0.422. The unrelated token receives 42% of the attention, and the gradient of the loss flows into it.

**Implementation.** FlashAttention's variable-length kernel receives cumulative boundaries `cu_seqlens` (for lengths 3, 2, 4: `[0, 3, 5, 9]`) and computes block-diagonal causal attention without materializing an L×L mask. In the Hugging Face padding-free path, the boundaries are derived from `position_ids`: the model's attention code computes `cu_seq_len` from `position_ids` when `attention_mask` is absent ([[packing-flash-attention-position-ids]], §3.3). TRL therefore omits the attention mask in that path, and it also removes the label of the first token of each example, so that the last token of one example is not trained to predict the first token of the next ([[trl-sft-packing-masking]], `sft_trainer.py` L485-486, L516):

```python
# trl/trainer/sft_trainer.py @ aa89588, L484-L516 (abridged)
# For padding-free, we should NOT create attention_mask as it causes FlashAttention to ignore position_ids and
# compute wrong cu_seq_lens from the all-1s mask
...
output["labels"][output["position_ids"] == 0] = -100
```

The companion figure [figures/packed-attention.html](figures/packed-attention.html), described at the end of §3, draws this mask for editable document lengths.

**Evidence.**
- Pre-training 1.3B models on 150B SlimPajama tokens, intra-document masking lowered average perplexity from 9.172 to 8.410 (2K context) and 9.065 to 8.079 (8K), raised average in-context-learning accuracy over 7 classification tasks from 63.54 to 70.52 (2K) and 62.43 to 71.23 (8K), and raised closed-book QA exact match from 7.99 to 10.99 (8K), at a 4% runtime cost in the authors' implementation ([[sequence-composition-intra-doc-masking]], Tables 1-3, §3.1).
- In a 13B 2k ablation, adding the document mask to concatenation lowered perplexity from 9.64 to 9.53 and raised reading comprehension from 46.71 to 47.92; BFD packing with the mask reached 48.92, and concatenation with the mask lowered summarization ROUGE-2 from 12.16 to 11.79 ([[fewer-truncations-best-fit-packing]], App. C.3, Table 10).
- In long-context continued training of Llama-3-8B (5B tokens at 64K), cross-document masking gave a long-context average of 54.6 and short-task average of 65.5 against 53.6 and 64.9 without masks ([[prolong]], App. B.2, Table 20).
- Llama 3 used a document mask and reports "limited impact" in standard pre-training and importance "in continued pre-training on very long sequences", with no numbers ([[llama-3]], §3.2).
- In BERT, removing the mask adjustment made loss and accuracy "worsen drastically" without recovery ([[sequence-packing]], §4.2.1).

Status: **Replicated** for pre-training and long-context training at 1.3B-13B ([[sequence-composition-intra-doc-masking]], [[fewer-truncations-best-fit-packing]], [[prolong]]). Practice differs by lab: DeepSeek-V3 packed pre-training documents without cross-sample masking but used sample masking in SFT, and DeepSeek-V4 added sample-level masking in pre-training ([[deepseek-v3-recipe]], §4.1, §5.1; [[deepseek-v4-recipe]], §4.1).

**Conditions and limits.** The studies above are pre-training or long-context continued-training studies; none of the verified sources isolates the mask in decoder SFT. [[packing-analysis-sft]] does not state an attention reset, states that the [EOS] separator lets the model tell adjacent samples apart (§3.3.3), and argues that packed unrelated samples form "fake" multi-turn conversations; when LLaMA-3-8B was trained with packing on a filtered 200K single-turn OpenHermes 2.5 set, MATH dropped, and adding multi-turn conversations at 1/40 to 1/20 of the data restored it. The same authors report that their internal 200K single-turn dataset showed no decline on few-shot benchmarks, and they attribute the difference to data quality (§5.3). The size of the cross-contamination effect in decoder SFT is an **Open question**.

## §3 Position IDs under packing: when a reset matters

**Definition.** Position ids are the integer positions fed to the positional encoding. Under packing they can reset to 0 at each document (`[0,1,2, 0,1, 0,1,2,3]`) or continue across the row (`[0,1,2,3,4,5,6,7,8]`).

**Mechanism for RoPE.** Rotary position embedding (RoPE) rotates query and key vectors by an angle proportional to position. For one frequency θ,

```
⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩
```

where `q` and `k` are a query and key vector, `R_m` is the rotation by angle m·θ, and `m`, `n` are the positions of the query and key tokens. The attention logit depends only on the offset n − m.

**Worked example.** Take θ = 1 rad, q = k = (1, 0). A three-token document at positions 0-2 gives, for the last token attending to the first, cos(2 − 0) = −0.416. The same document at positions 5-7 gives cos(7 − 5) = −0.416. With a block-diagonal mask, every allowed pair has the same offset under both numberings, so attention inside each document is unchanged. [[fewer-truncations-best-fit-packing]] relies on this: with cross-document attention masked, "thanks to the relative nature of rotary positional embeddings (RoPE), we do not adjust position ids" (App. C.1).

**When a reset matters.**
1. Learned absolute position embeddings: the model looks up a vector per position. In packed BERT without the position adjustment, MLM accuracy stalled at 71.8% against a 72.1% target ([[sequence-packing]], §4.2.1).
2. Frameworks that infer document boundaries from `position_ids`: in the Hugging Face padding-free path, continuous position ids contain a single 0, so the boundaries computed from them are `[0, total]` and the kernel attends across all documents (derived from the rule in [[packing-flash-attention-position-ids]], §3.3). The measured effect of this failure on loss is not isolated in the verified sources: in [[packing-flash-attention-position-ids]] (Table 2), both offline packing variants had higher validation loss than padding (Mistral-7B: 1.306 without position ids, 1.284 with, 1.129 padded), the two packed variants also differ in row count (2,294 vs 585 rows for Mistral-7B), and the gap between them changed sign across models (Llama-2: 1.579 vs 1.578; Granite: 1.740 vs 1.767). This pattern is consistent with the authors' explanation that the fewer optimizer steps of offline packing dominate.

**Position frequency.** [[string-effective-context]] defines how often a relative distance i is trained:

```
f(i) = Σ_{s ∈ C} max(|s| − i, 0),   0 ≤ i < L
```

where `C` is the training corpus treated as a set of sequences, `s` is one sequence, `|s|` is its length, `i` is a relative distance, and `L` is the training length (§2.2, Eq. 2). The paper applies the formula to training sequences. This chapter applies it to the spans over which attention is allowed: with a document mask the spans are documents; without it they are whole rows. For one 8-token row holding documents of 3 and 5 tokens, the masked counts are f = [8, 6, 4, 2, 1, 0, 0, 0] and the unmasked counts are [8, 7, 6, 5, 4, 3, 2, 1]; the share of trained pairs at distance ≥ 3 falls from 15/36 = 42% to 3/21 = 14%. The paper reports that on SlimPajama at L = 2,048, distances ≥ 1,536 account for less than 5% of position indices and links rarely trained distances to effective context lengths below the training length (§2.2). The implication that document masking reduces training on long distances unless the documents themselves are long is derived from Eq. 2 in this chapter (**Interpretation**). ProLong's long-context data used single documents of at least 64K tokens and packed only the short data ([[prolong-recipe]], §6.1, App. A.2).

**Figure.** [figures/packed-attention.html](figures/packed-attention.html) lets the reader edit document lengths, toggle the document mask and the position-id numbering, and see the resulting `cu_seqlens`, the boundaries a padding-free framework would infer from `position_ids`, the per-distance counts f(i), and each example's share of the loss.

## §4 Loss masking, prompt-loss weight, and loss aggregation

**Definitions.** A loss mask sets the label of a token to −100 (PyTorch's `ignore_index`), so the token contributes no loss term. Instruction tuning (IT) computes loss only on the completion; instruction modelling (IM) also computes loss on instruction tokens but not on template tokens ([[loss-masking-prompt]], §3, Eqs. 2 and 4). A prompt-loss weight (PLW) scales the loss on prompt tokens by a value between 0 and 1 ([[prompt-loss-weight-sift]], §2).

```
L_IT = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})
L_IM = − Σ_{t=1..m+n} 1(x_t ∉ T) · log P(x_t | x_1..x_{t−1})
L_PLW = − Σ_{t ∈ completion} log P(x_t | x_<t)  −  w · Σ_{t ∈ prompt} log P(x_t | x_<t)
```

where `I_1..I_m` are the m instruction (prompt) tokens, `C_1..C_n` the n completion tokens, `x` their concatenation, `T` the set of template tokens such as `<|user|>` and `<|assistant|>`, `1(·)` the indicator function, and `w ∈ [0, 1]` the prompt-loss weight. The normalization of L_PLW by token counts is implementation-specific.

**Problem.** When completions are short, a prompt mask leaves few loss tokens per example, and the model can memorize them.

**Worked example.** In AlpacaDataShort the mean prompt is 179.27 tokens (16.93 instruction + 162.34 input) and the mean completion is 14.62 tokens ([[prompt-loss-weight-sift]], Table 1). The prompt's share of weighted loss tokens is w·179.27 / (w·179.27 + 14.62): 0.925 at w = 1, 0.748 at w = 0.242, 0.109 at w = 0.01, and 0 at w = 0.

**Evidence.**
- Shi et al. (mainly LLaMA-2-7B, with LLaMA-2-13B and OPT-6.7B checks; instruction datasets from 1,030 to 326,181 examples; 18 NLP tasks plus MT-Bench and AlpacaEval) report that IM "in many scenarios" improves both NLP tasks and open-ended benchmarks, most when instructions are long relative to outputs and when examples are few (§4.2, Fig. 2). Examples from Table 1: on Alpagasus Dolly 9k the NLP mean rose from 45.54 to 48.00 and AlpacaEval 1.0 from 21.54 to 30.77; on Less MMLU Chat AlpacaEval 1.0 rose from 4.42 to 9.78; on LIMA the NLP mean rose from 48.79 to 49.60 while AlpacaEval 1.0 moved from 33.06 to 32.94. The authors state they are "not proposing IM as a replacement" for standard fine-tuning (abstract).
- The overfitting measurements in the same paper: on LIMA, mean training loss on outputs was 1.45 for IM against 1.37 for IT, while test loss on a 10% Tulu V2 sample was 1.17 against 1.32; BLEU of greedy outputs against training responses was lower for IM on all 7 datasets in Table 2 (§4.3). Adding a KL penalty toward the base model during IT, in place of IM, reduced LIMA AlpacaEval 2.0 from 2.58 to 0.06 while raising the NLP mean from 48.79 to 49.26 (Table 3).
- Huerta-Enochian & Ko (LLaMA 1 7B and LLaMA 2 7B, 10 PLW values, 3 Alpaca variants, 13 benchmarks) found a statistically significant negative quadratic relation between PLW and performance only for short-completion data, with a fitted optimum λ = 0.242; PLW 0.01-0.5 was better on multiple-choice and short-generation benchmarks, PLW near 1 better on long-generation judge benchmarks, and the relation was not statistically significant for the two long-completion datasets (p = 0.237 and 0.0861) (§5.1-5.2, Table 3). The authors tentatively conclude that training-loss stability is not the driving factor, and they point to weights staying closer to the pre-trained model for small non-zero PLW and to lower memorization for larger PLW (§5.3).
- Llama 2 Long Chat added language-modelling loss on long input prompts; with self-instruct long data, QuALITY rose from 59.3 without input loss to 77.3 with it, and Qasper from 35.7 to 38.9 ([[llama-2-long]], Table 9, model size not stated).

Status: **Replicated** in direction for short-completion or long-input data across [[loss-masking-prompt]], [[prompt-loss-weight-sift]], and [[llama-2-long]], in different settings. For large, long-completion mixtures such as Tulu V2 (instruction-to-output length ratio about 0.5) the reported gain was smaller ([[loss-masking-prompt]], §4.2), and Huerta-Enochian & Ko found no statistically significant relation for long-completion data. The released recipes cited in this chapter mask prompts without reporting an ablation: Llama 3 SFT ([[llama-3]], §4.1.3), Tülu 1 ([[tulu-1-how-far-can-camels-go]], §3.2), Nemotron-4-340B-Instruct ([[nemotron-4-synthetic-recipe]], §3.3.1), and ProLong SFT ([[prolong-recipe]], App. A.3).

**Implementation error to avoid.** For a batch tensor of shape [B, T], `labels[:prompt_len] = -100` masks the first `prompt_len` rows, not token positions. The per-example form is `labels[b, :prompt_len[b]] = -100`, followed by the one-position shift `logits[:, :-1]` against `labels[:, 1:]`.

**Multi-turn conversations.** Three rules are used. All-assistant masking trains every assistant turn in one pass; open-instruct's `sft_tulu_tokenize_and_truncate_v1` masks every non-assistant message, and a separate `last_turn_tulu_tokenize_and_truncate_v1` trains only the final message ([[open-instruct-sft-masking]], `dataset_transformation.py` L1176-1225). Per-turn unrolling creates one example per assistant turn.

Worked example: turns u1 = 40, a1 = 50, u2 = 30, a2 = 80, u3 = 20, a3 = 120 tokens. All-assistant masking processes 340 tokens and trains 250. Last-turn masking processes 340 and trains 120. Unrolling processes 90 + 200 + 340 = 630 tokens and trains the same 250. Under causal attention the prediction of each a1 token depends only on earlier tokens, so unrolling yields the same per-token losses as all-assistant masking at 630/340 = 1.85 times the compute, differing only in how losses are averaged. This equality requires a prefix-preserving template, one that renders earlier messages identically regardless of what follows ([[trl-sft-packing-masking]], `chat_template_utils.py` L829-831). Evidence that one rule generalizes better across multi-turn tasks is not reported in the verified sources (**Open question**).

**Framework defaults.** In TRL at aa89588, `completion_only_loss=None` computes loss on the completion for prompt-completion datasets and on the full sequence for language-modelling datasets, and `assistant_only_loss=False` computes loss on the full sequence of conversational datasets ([[trl-sft-packing-masking]], `sft_config.py` L96-105). With `assistant_only_loss=True`, the assistant mask comes from `{% generation %}` markers in the template. When the tokenizer's template lacks them, TRL substitutes a patched training template for the families it lists (among them Llama 3, Qwen2.5, and Qwen3) and raises a `ValueError` for other templates (`sft_trainer.py` L1251-1256; `chat_template_utils.py` L1032-1043, L1196-1199). TRL also raises an error when an example has no assistant tokens and warns when the assistant's end-of-turn token falls outside the mask, because "the model may not learn to stop" (`sft_trainer.py` L1258-1266, L1585-1591). The Alignment Handbook's Zephyr SFT config at 1de1fc9 sets neither `packing` nor `assistant_only_loss` ([[hf-alignment-handbook]], `recipes/zephyr-7b-beta/sft/config_full.yaml`), so under the TRL defaults above it trains on all tokens without packing (derived, not stated by the handbook). The same config sets `max_seq_length`, which is not a field of TRL's `SFTConfig` at aa89588 (the field is `max_length`), so its effective settings depend on the TRL version installed.

**Loss aggregation.** Token-mean loss divides the summed token losses by the number of target tokens in the batch; sequence-mean loss averages per-example means. Tülu 3 shows that the Transformers default changes weighting when gradient accumulation or data parallelism splits a batch:

```
one pass:      L = (l_1 + l_2) / (n_1 + n_2)
accumulation:  L = (l_1/n_1 + l_2/n_2) / 2
```

where `l_k` is the summed token loss of example k and `n_k` its number of non-padding target tokens ([[tulu-3]], §4.3.2, Eqs. 1-2). Tülu 3 used a sum loss and found LR 5e-6 with sum loss best on Llama 3.0 with the Tülu 2 mixture (Fig. 5). Packing adds a third weighting: if each pack's token mean counts equally, examples in packs with fewer target tokens receive more weight. LongAlign scales the loss of example i by K/(N_i·M) and sums within packs:

```
L′ = (1/K) Σ_{k=1..K} Σ_{i ∈ pack k} L_i · K / (N_i · M) = (1/M) Σ_{i=1..M} L_i / N_i
```

where `K` is the number of packs in the batch, `M` the number of examples, `N_i` the target tokens of example i, and `L_i` its summed loss ([[longalign]], §3.3, Eq. 4).

Worked example: pack 1 holds A (N = 10, L = 20) and C (N = 20, L = 30); pack 2 holds B (N = 90, L = 90). Averaging per-pack token means gives (50/30 + 90/90)/2 = 1.333, which weights A, C, B by 0.167, 0.333, 0.500. The batch token mean gives 140/120 = 1.167 with weights 0.083, 0.167, 0.750. LongAlign's scaling gives (20·2/30 + 30·2/60 + 90·2/270)/2 = 1.5, equal to the sequence mean (2.0 + 1.5 + 1.0)/3 with weight 1/3 each. Krell et al. describe the same per-pack weighting problem for per-sequence losses in BERT ([[sequence-packing]], §3.2.3). On LongBench-Chat, packing with loss weighting scored 6.21 against 5.76 for packing alone on ChatGLM3-6B-64k and 6.10 against 5.89 on Llama-2-7B-64k, while single-document QA moved from 65.0 to 64.0 and 61.7 to 60.8 ([[longalign]], Table 3). **Result (single study).**

## §5 NEFTune: a regularizer whose measured gain is judge preference

**Definition and formula.** NEFTune adds uniform noise to the input embeddings during fine-tuning only:

```
X′ = X + (α / √(L·d)) · ε,   ε ~ Uniform(−1, 1)^{B×L×d}
```

where `X` is the embedding output of shape B×L×d, `B` the batch size, `L` the sequence length, `d` the embedding dimension, and `α` the noise scale; when lengths differ in a batch, the scale is computed per sequence ([[neftune]], Algorithm 1 and its footnote).

**Evidence.** LLaMA-2-7B AlpacaEval win rate (GPT-4 judge, against Text-Davinci-003) rose from 29.79 to 64.69 on Alpaca and from 70.34 to 79.60, 68.74 to 76.28, and 62.00 to 70.61 on Evol-Instruct, ShareGPT, and OpenPlatypus; ARC, HellaSwag, MMLU, and TruthfulQA stayed stable ([[neftune]], Table 1, Fig. 3). Training loss was higher and held-out loss slightly lower (Fig. 4). Mean output length of LLaMA-2 7B on Alpaca rose from 375.22 to 1,061.89 characters (Table 4). For LLaMA-1 Alpaca-7B with a GPT-4 judge, prompting for long answers gave 48.01 and blocking the end-of-sequence token until 250 tokens gave 38.58, against 32.36 for the baseline and 61.99 for NEFTune (Table 5), so length alone did not reproduce the gain. In [[loss-masking-prompt]], NEFTune lowered the NLP mean on Less Tydiqa from 48.21 to 47.47 while raising AlpacaEval 1.0 from 5.12 to 8.35 (Table 1). The measured gains are judge-preference gains; the four multiple-choice benchmarks stayed stable (Fig. 3).

**Interaction with packing.** Transformers implements the hook as:

```python
# src/transformers/integrations/neftune.py @ 05e078a, L47-L50
if module.training:
    dims = torch.tensor(output.size(1) * output.size(2))
    mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
    output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
```

`output.size(1)` is the row length ([[transformers-chat-templates]]). With α = 5 and d = 4,096, a single 512-token example gets scale 5/√(512·4,096) = 0.00345. Packed with seven others into one 4,096-token padding-free row, it gets 5/4,096 = 0.00122, which is 1/√8 = 0.354 of the per-sequence value in Algorithm 1 (derived from the code and the algorithm footnote).

## §6 Chat templates: turn markers, generation prompts, and mismatches that raise no error

**Definition.** A chat template is a Jinja program stored with the tokenizer that renders a list of role/content messages into one token sequence, including the control tokens that start and end each turn. `add_generation_prompt=True` appends the tokens that open an assistant turn ([[transformers-chat-templates]], `chat_templating.md` L129-169).

**Design elements in released templates.**
- Zephyr, from the Alignment Handbook config: `'<|user|>\n' + content + eos_token` for users and `'<|assistant|>\n' + content + eos_token` for the assistant, with `'<|assistant|>'` as the generation prompt ([[hf-alignment-handbook]], `config_full.yaml` L8).
- Llama 3 uses "special header and termination tokens": headers give the source and destination of each message (for example user or ipython for tool execution), and termination tokens mark when to alternate speakers ([[llama-3]], §4.1.1).
- Qwen3 renders the same assistant tool-call message with an empty `<think>\n\n</think>` block when it is the last message, and without that block when a tool message follows; the tool result is rendered inside a user turn as `<tool_response>` ([[trl-sft-packing-masking]], `chat_template_utils.py` L1071-1077). Such a template is not prefix-preserving. TRL's patched Qwen3 training template renders the empty block in both cases (L1078-1085).
- OLMo 3 32B Think was released with the `<think>` token in `add_generation_prompt`, and the OLMo 3.2 think templates add it the same way ([[open-instruct-allenai-recipes]], `docs/olmo3.md` L23, L29). The Ai2 documentation reports that its tokenization code "incorrectly masks the first <think> token as part of the prompt", and OLMo 3.2+ Think SFT data is therefore tokenized with the instruct template, which does not contain `<think>`, so that the model learns to generate it ([[open-instruct-allenai-recipes]], `docs/olmo3.md` L26-28, L39). The mechanism is visible in the masking code: the end of each masked prompt span is computed by rendering the prefix with `add_generation_prompt=True`, so any token inside the generation prompt is masked ([[open-instruct-sft-masking]], `dataset_transformation.py` L1159-1171; derived from the code).

**Failure modes that raise no error.**
1. Different stop tokens between training and serving: the model ends turns with one id and the server waits for another.
2. A different template across stages: OLMo 3 7B Think's DPO and RL models used a "slightly different chat template" from SFT because of "a minor miscommunication", and this is reflected in the released models ([[open-instruct-allenai-recipes]], `docs/olmo3.md` L22).
3. Duplicated special tokens: for tokenizers that add BOS or EOS, rendering with `tokenize=False` and then tokenizing with the default `add_special_tokens=True` duplicates tokens that the template already contains ([[transformers-chat-templates]], `chat_templating.md` L125-127).
4. End-of-turn token outside the loss mask (§4).
5. A different generation prompt: Open R1 models distilled on in-domain reasoning traces emitted `<think>` for coding queries but reverted to the original instruct behaviour for out-of-domain queries unless the response was prefilled with `<think>`; the authors recommend prefilling in the release template ([[openr1]], Update #3 Lesson 4).
6. History rendered differently in training and inference, as in the Qwen3 case above.

**Evidence on effect size.** On an intermediate Tülu SFT mixture with Llama 3.0, the template choice gave averages of 53.0 (Tülu template with the trailing newline replaced by EOS), 52.9 (Zephyr), 52.8 (Tülu 3, no trailing newline), 52.6 (Tülu 2), and 51.6 (Llama 3 template) ([[tulu-3]], §4.3.1, Table 13). For comparison, final Tülu 3 SFT runs varied by seed from 59.8 to 60.1 at 8B and 70.0 to 72.6 at 70B (Table 14), a different setup. Llama 3 masked header and termination tokens in both DPO responses; the authors observe that including them "may lead to undesired model behaviors such as tail repetition or abruptly generating termination tokens" ([[llama-3]], §4.1.4). Numbers for a train-serve template mismatch are not reported in the verified sources.

## §7 Format and template overfitting

**Definition.** Format sensitivity is the change in a model's score across prompt formats that preserve meaning. FormatSpread reports the spread, max_i m(p_i, D) − min_i m(p_i, D), where `p_i` are equivalent formats, `D` the evaluation set, and `m` the metric ([[prompt-format-sensitivity-formatspread]], §3.2).

**Worked example.** Five equivalent formats give accuracies 0.62, 0.70, 0.55, 0.81, and 0.66. The spread is 0.81 − 0.55 = 26 points, and a paper that reports one format could report any value in that interval.

**Evidence.**
- Over 53 Super-NaturalInstructions tasks with LLaMA-2 7B/13B/70B, Falcon-7B and Falcon-7B-Instruct, and GPT-3.5, the spread reached 76 points for LLaMA-2-13B, with 10 sampled formats per task the median spread was 7.5 points and 20% of tasks had spreads of at least 15 points in all LLaMA-2 settings (lower bounds, because only 10 formats were sampled); spread persisted with more shots, larger models, and instruction tuning ([[prompt-format-sensitivity-formatspread]], §1, §4.2). Model rankings reversed by at least 2 points under another format with probability 0.141 for LLaMA-2-13B against 70B (§4.2).
- Over 21 models and 4 classification datasets, the standard deviation across templates reached 35% of the mean score, the top-10 templates of two models overlapped with IoU above 0.5 for only a few model pairs, and instruction-tuned Llama 3 and Mistral models were also sensitive ([[mind-your-format-templates]], §4.1, §5, App. L). **Replicated** with [[prompt-format-sensitivity-formatspread]].
- Training-side evidence: T5-XL fine-tuned on the Flan 2022 collection with mixed zero-shot and few-shot templates scored 74.8 few-shot on held-in tasks and 52.4 few-shot on MMLU; removing few-shot templates gave 62.2 and 38.7, and zero-shot MMLU fell from 50.3 to 47.3 ([[flan-collection]], Table 1). The same ablation raised zero-shot chain-of-thought (35.8 to 38.9) and zero-shot BBH (26.2 to 27.6) scores (Table 1). Across mixing ratios, training with 10-90% few-shot templates gave higher held-in and MMLU scores than training with one prompt setting (§3.2, Fig. 3). **Result (single study)**, 3B encoder-decoder. Direct evidence that diversifying chat templates or system prompts during SFT reduces format spread in decoder chat models is not reported in the verified sources (**Open question**).

**Implication.** A score measured under one template is one sample from a distribution over formats. The Open R1 `<think>` result in §6 shows the training-side version: a behaviour learned from in-domain reasoning traces appeared on out-of-domain prompts only when the response was prefilled with `<think>`.

## §8 Long sequences: document masking, position ids, and length-aware loss

At 32K-512K tokens the choices above interact with the length distribution.
1. **Document masking.** Llama 3 found the document mask important in continued pre-training on very long sequences ([[llama-3]], §3.2), and ProLong measured +1.0 long-context and +0.6 short-task points with masks at 64K ([[prolong]], Table 20). On 8 H100 GPUs with FSDP, ProLong's variable-length attention alone moved throughput from 2,770 to 2,780 tokens/s per GPU, and variable-length attention with minibatch reordering reached 3,095 tokens/s, an 11.7% increase over full attention ([[prolong]], App. A.3, Table 15).
2. **Position ids.** With RoPE and a correct mask, a reset does not change attention (§3). With a document mask, long relative distances are trained only by long documents (derived from Eq. 2 in §3), and [[string-effective-context]] links rarely trained distances to effective lengths below the training length (§2.2, §3).
3. **Truncation.** TRL's default `"bfd"` strategy discards tokens beyond `max_length`; `"bfd_split"` keeps them as separate fragments ([[trl-sft-packing-masking]], `data_utils.py` L862-866). For long reasoning traces, clipping across chunks was the Open R1 authors' explanation for worse packed SFT ([[openr1]], Update #3 Lesson 1).
4. **Length-aware loss.** When long and short examples share packs, per-pack averaging favours examples in sparse packs; LongAlign's K/(N_i·M) weighting restores equal per-example weight (§4). ProLong's SFT instead averages over valid tokens (`--token_scaled_loss`) ([[prolong-recipe]], App. A.3).
5. **Memory.** For the 8B ProLong runs on H100 GPUs, 64K was the largest power-of-2 length trainable without sequence parallelism, and the 512K stage used DeepSpeed-Ulysses sequence parallelism over 8 GPUs ([[prolong-recipe]], App. A.3, A.4). ch-05 covers context-parallel layouts, and the training-memory course (ch-06 kernels, ch-07 parallelism) covers the memory mechanics.

## §9 Tokenizer and template co-design when adding tokens after pre-training

Adding `<think>`, tool-call, or turn tokens after pre-training creates embedding and output rows that the model has never trained.
1. **Initialization.** `resize_token_embeddings` in Transformers defaults to `mean_resizing=True`, which samples new rows from a multivariate normal distribution with the old embeddings' mean and covariance; the docstring states that this reduces the KL divergence between next-token distributions before and after resizing ([[transformers-chat-templates]], `modeling_utils.py` L2734-2741). Worked example: with output embeddings (1,0), (0,1), (−1,0), (0,−1) and hidden state (2,1), the logits are 2, 1, −2, −1 and the top token has probability 0.696. A new row at the mean (0,0), the center of that distribution, has logit 0, probability 0.086, and lowers the top token to 0.636. A new row (3,0) has logit 6 and probability 0.974.
2. **Trainability.** TRL's `clone_chat_template` adds the source tokenizer's tokens and resizes embeddings to a multiple of 64 with dummy tokens; with PEFT, TRL adds the new ids to trainable tokens, warns that without `lm_head` in `modules_to_save` "the model may not learn to generate outputs with these new tokens", and then adds `lm_head` to `modules_to_save` ([[trl-sft-packing-masking]], `chat_template_utils.py` L45-136; `sft_trainer.py` L1088-1108).
3. **Masking.** Every new token that the model must emit has to be inside the loss mask in every stage, which is the OLMo 3 `<think>` case in §6.
4. **Default when a tokenizer has no template.** The Alignment Handbook's `sft.py` calls `setup_chat_format(..., format="chatml")`, which adds ChatML tokens ([[hf-alignment-handbook]], `scripts/sft.py` L98-100).

## Negative samples and negative feedback

This chapter's stage uses negatives in two of the four senses of the course standard. **Negative as content**: a failed turn, tool error, or rejected draft stays in the context with its label set to −100, and only the corrected continuation is trained (ch-31a). **Negative as gradient**: in DPO the rejected response's log-probability is pushed down (ch-39, ch-43a), and the loss mask decides which of its tokens receive that push.

**Where the masking question arises.** Chosen and rejected responses share formatting tokens such as headers and end-of-turn ids. Llama 3 masked them in both responses; the authors report that letting these tokens contribute to the loss may lead to tail repetition or abrupt generation of termination tokens, and they hypothesize that shared tokens create a conflicting objective to raise and lower the same tokens ([[llama-3]], §4.1.4). This is qualitative; no numbers are reported.

**Mechanism.** For a softmax over logits z,

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

where `y` is the target token, `p_j` the probability of token j, and `1[·]` the indicator. Decreasing log p_y moves z_y down by an amount proportional to (1 − p_y) and moves every other z_j up in proportion to p_j. If the rejected response's end-of-turn token has p = 0.90, a newline has 0.08, and all other tokens share 0.02, one push-down step changes logits in proportion to −0.10, +0.08, and +0.02: 80% of the increase goes to the newline, the most likely alternative. A model that learns to prefer a newline over the end-of-turn token at the end of a response keeps generating, which is consistent with the tail repetition Llama 3 reports (**Interpretation**).

**Controls.** Mask shared formatting tokens in both responses; add an NLL term on chosen responses (Llama 3 used coefficient 0.2, §4.1.4); keep tool outputs and environment observations masked when they are not model actions (ch-45b).

**Diagnostics.** Log chosen and rejected log-probabilities separately, log the probability of the end-of-turn token at the end of chosen responses, and track the rate of responses that reach the length limit.

**Effect on generality.** Masking errors on termination tokens can affect every task, not the trained domain only, because every response must end (**Interpretation**; no source measures this across tasks).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 (all) | 8B-405B | pretrain-stable | attention mask | no self-attention across documents in one sequence | arXiv:2407.21783v3 §3.2 ([[llama-3-recipe]]) | verified 2026-09-14 | "limited impact" in standard pre-training, "important" for very long sequences; no numbers |
| Llama 3.1 405B | 405B | SFT | loss masking | cross-entropy on target tokens, prompt tokens masked | arXiv:2407.21783v3 §4.1.3 | verified 2026-09-14 | no ablation reported |
| Llama 3.1 405B | 405B | preference | DPO token masking; NLL term | header and termination tokens masked in chosen and rejected; NLL coefficient 0.2 on chosen | arXiv:2407.21783v3 §4.1.4 | verified 2026-09-14 | qualitative: without masking these tokens "may lead to" tail repetition or abrupt termination tokens; no ablation numbers |
| DeepSeek-V3 | 671B total / 37B active | pretrain-stable | packing; attention | document packing, no cross-sample attention masking | arXiv:2412.19437v2 §4.1 ([[deepseek-v3-recipe]]) | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B total / 37B active | SFT | packing; attention | multiple samples per sequence with sample masking | arXiv:2412.19437v2 §5.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V4-Flash / -Pro | both | pretrain-stable | packing; attention | documents packed to minimize truncation; sample-level attention masking | arXiv:2606.19348v1 §4.1 ([[deepseek-v4-recipe]]) | verified 2026-09-14 | no ablation reported |
| ProLong-64k-Base | 8B | long-context | attention; packing | cross-document masking; short data packed into 64K sequences; long data from documents ≥ 64K tokens | arXiv:2410.02660v4 §6.1, App. A.2, B.2 ([[prolong-recipe]]) | verified 2026-09-14 | Table 20: 54.6 long / 65.5 short with masks vs 53.6 / 64.9 without |
| ProLong-512k-Instruct | 8B | SFT | loss masking; aggregation | instruction tokens masked; loss averaged over valid tokens | arXiv:2410.02660v4 App. A.3; `train_sft.sh` `--apply_instruct_masks`, `--token_scaled_loss` | verified 2026-09-14 | no ablation reported |
| Llama 2 Long Chat | 70B | SFT | loss masking | LM loss on output tokens and on long input prompts | arXiv:2309.16039v3 §2.2, §4.3 ([[llama-2-long]]) | verified 2026-09-14 | Table 9: QuALITY 59.3 without input loss vs 77.3 with (model size not stated) |
| Tülu 3 SFT | 8B, 70B | SFT | loss aggregation | sum loss ("opted generally") | arXiv:2411.15124v5 §4.3.2 ([[tulu-3]]) | verified 2026-09-15 | Fig. 5: LR 5e-6 with sum loss best on Llama 3.0 + Tülu 2 mix; values only in figure |
| Tülu 3 SFT | 8B, 70B | SFT | chat template | Tülu 3 template: no newline at end of template before model response | arXiv:2411.15124v5 §4.3.1, App. B.3 | verified 2026-09-15 | Table 13: 52.8 vs 53.0 for replacing the newline with EOS, not chosen to keep later stages consistent |
| Tülu 1 (all runs) | 6.7B-65B | SFT | loss masking | loss only on assistant tokens | arXiv:2306.04751v2 §3.2 ([[tulu-1-how-far-can-camels-go]]) | verified 2026-09-14 | no ablation reported |
| Nemotron-4-340B-Instruct | 340B | SFT | loss masking | user turns masked; loss on assistant turns | arXiv:2406.11704v2 §3.3.1 ([[nemotron-4-synthetic-recipe]]) | verified 2026-09-14 | no ablation reported |
| OpenR1-Distill-7B | 7B | distill-SFT | max length; packing; template | 32,768; `packing: false`; ChatML with a default system prompt requesting `<think>` sections | open-r1@1416fa0 `recipes/OpenR1-Distill-7B/sft/config_distill.yaml` ([[openr1-recipe]]) | verified 2026-09-14 | model card exp7-8: packing vs no packing on math (figure only) |
| OlympicCoder-7B | 7B | distill-SFT | packing | not used | HF blog "Open R1: Update #3", Lessons 1 and 4 ([[openr1-recipe]]) | verified 2026-09-14 | packing "consistently worse" on all ablated datasets (figure only) |
| Alignment Handbook zephyr-7b-beta SFT config | 7B | SFT | template; max length; packing; loss keys | Zephyr `<|user|>`/`<|assistant|>` template with EOS; `max_seq_length: 2048`; no `packing` key; no `assistant_only_loss` key | alignment-handbook@1de1fc9 `recipes/zephyr-7b-beta/sft/config_full.yaml` L8, L42 ([[hf-alignment-handbook]]) | verified 2026-09-15 | no ablation reported; not shown to be the config of the 2023 release |
| TRL `SFTConfig` default | any | SFT | packing; strategy; padding-free; loss masks | `packing=False`; `packing_strategy="bfd"`; `padding_free=False` (enabled regardless of this value when packing uses `bfd`); `completion_only_loss=None`; `assistant_only_loss=False` | trl@aa89588 `trl/trainer/sft_config.py` L77-105 ([[trl-sft-packing-masking]]) | verified 2026-09-15 | framework default; no ablation in the repository |
| LLaMA-2 7B NEFTune runs | 7B | SFT | NEFT α | Alpaca 5, Evol-Instruct 5, ShareGPT 10, OpenPlatypus 15 | arXiv:2310.05914v2 Table 7 ([[neftune]]) | verified 2026-09-14 | best of α ∈ {5, 10, 15} on AlpacaEval, ChatGPT judge (App. A.1) |
| IM/IT runs (Shi et al.) | 7B, 13B | SFT | LR; batch; epochs; max length | 2e-5; 128; 2, 3, or 10; 2,048 | arXiv:2405.14394v2 App. C, Table 6 ([[loss-masking-prompt]]) | verified 2026-09-15 | no ablation reported for these values |
| PLW study, AlpacaDataShort | 7B | SFT | prompt-loss weight | fitted optimum λ = 0.242 | arXiv:2401.13586v4 §5.2 ([[prompt-loss-weight-sift]]) | verified 2026-09-15 | printed value; maximum of the authors' fitted quadratic in Table 3 (p < 0.001 for AlpacaDataShort) |
| LongAlign ChatGLM3-6B-64k, Llama-2-7B-64k | 6B, 7B | long-context SFT | packing; loss weighting | packing with per-example scale K/(N_i·M) | arXiv:2401.18058v1 §3.3, Eq. 4 ([[longalign]]) | verified 2026-09-15 | Table 3: LongBench-Chat 6.21 vs 5.76 and 6.10 vs 5.89 against unweighted packing |
| Best-fit packing models (Ding et al.) | 7B, 13B | pretrain-stable | packing; attention; position ids | BFD; cross-document attention masked; position ids not adjusted | arXiv:2404.10830v2 App. C.1 ([[fewer-truncations-best-fit-packing]]) | verified 2026-09-15 | Table 10 ablation of mask and packing |

**Starting point for a small general-purpose run.** For SFT of a 7-8B model on a mixed instruction set, pack with BFD and intra-document attention, as in the Ding et al. pre-training runs (7B-13B, BFD, masked cross-document attention, unadjusted RoPE position ids) and DeepSeek-V3 SFT (sample masking). Set `max_length` at or above the longest example that should be trained whole, or use `bfd_split` for continued pre-training data, because TRL's default `bfd` truncates overflow. Mask prompt tokens as Llama 3.1, Tülu 1, and Nemotron-4-340B-Instruct do, and when the set is short-completion, sweep the prompt-loss weight around the 0.242 optimum fitted for LLaMA 7B on AlpacaDataShort. When long and short examples share packs, use the LongAlign per-example scaling that was tested at 6-7B with 64K context. The Tülu 3 template choice (Table 13) and Llama 3.1 DPO masking of header and termination tokens are the verified template and preference-stage settings; no row above verifies a template-diversification ratio or a NEFTune α for multi-turn data.

## Generalization lens

**(a) What increases breadth.**
- Keeping documents whole: BFD packing raised reading comprehension, NLI, and context following, in 13B text models, and reduced undefined-name errors in a 7B code model ([[fewer-truncations-best-fit-packing]], Tables 4-5, §4.6).
- Intra-document attention: higher in-context-learning accuracy and closed-book QA at 1.3B ([[sequence-composition-intra-doc-masking]], Tables 2-3), and better long and short scores at 8B, 64K ([[prolong]], Table 20).
- Loss on instructions for short-completion or small data: less memorization and higher held-out scores ([[loss-masking-prompt]], §4.3; [[prompt-loss-weight-sift]], §5.3).
- Mixed zero-shot and few-shot templates during instruction tuning: higher held-in and held-out MMLU scores under both zero-shot and few-shot evaluation for T5-XL, while zero-shot chain-of-thought and BBH scores were higher without few-shot templates ([[flan-collection]], §3.2, Fig. 3, Table 1).

**(b) What causes narrowing or forgetting.**
- Packing long reasoning traces with a layout that lets samples cross chunk boundaries: lower LiveCodeBench scores in Open R1's code-reasoning SFT ablations; clipping of traces is the authors' hypothesis, not a measured cause ([[openr1]], Update #3 Lesson 1).
- Packing single-turn-only data: a MATH drop on filtered OpenHermes 2.5 that multi-turn data at 1/40 to 1/20 reversed, not observed on the authors' internal single-turn set ([[packing-analysis-sft]], §5.3).
- Response-only loss on short-completion data: memorization of training completions, corpus BLEU near 80 for PLW ≤ 0.1 ([[prompt-loss-weight-sift]], §5.3.3).
- Template-conditioned behaviour: out-of-domain prompts reverted to base behaviour without the `<think>` prefill ([[openr1]], Update #3 Lesson 4).
- KL regularization to the base model as a substitute for loss on instructions: against IT without KL, AlpacaEval 2.0 fell from 2.58 to 0.06 on LIMA and from 2.28 to 0.04 on Alpagasus Dolly 9k, while the NLP mean rose from 48.79 to 49.26 and from 45.54 to 49.31 (LLaMA-2-7B; [[loss-masking-prompt]], Table 3).

**(c) How to measure it at this stage.**
- Report multiple-choice and generation benchmarks separately: NEFTune and PLW moved judge-scored long generation while multiple-choice scores stayed stable or moved the other way ([[neftune]], Fig. 3; [[prompt-loss-weight-sift]], §5.1).
- Evaluate each checkpoint under several formats and report the spread and the ranking changes ([[prompt-format-sensitivity-formatspread]], §3.2; [[mind-your-format-templates]], §5).
- Compare template effects with seed variance before accepting a template change ([[tulu-3]], Tables 13-14).
- Measure memorization of training completions with BLEU of greedy outputs on training prompts, and held-out loss on a separate instruction set ([[loss-masking-prompt]], §4.3).
- For long-context changes, evaluate after SFT on downstream long tasks and short tasks together ([[prolong]], §2.1-2.3).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Packing without intra-document attention when isolation is intended | No error; in the pre-training studies of §2, higher perplexity and lower in-context-learning scores | Assert the number of `cu_seqlens` segments equals the number of examples; compare logits of one example packed vs alone |
| Passing an all-ones `attention_mask` in a padding-free path | FlashAttention ignores `position_ids` and attends across examples | Assert no `attention_mask` key when `padding_free=True` (TRL `sft_trainer.py` L485-486) |
| Continuous `position_ids` in a framework that infers boundaries from them | One inferred segment per row | Recompute boundaries from `position_ids`; expect one start per example |
| First token of an example trained as the target of the previous example's last token | Loss terms that cross examples | Assert `labels[position_ids == 0] == -100` |
| `labels[:prompt_len] = -100` on a [B, T] tensor | The first `prompt_len` examples have zero trained tokens; the other examples are trained on their prompt tokens | Log trained-token fraction per row |
| `assistant_only_loss=True` with a template lacking `{% generation %}` | TRL `ValueError` for templates it cannot patch; for patched families, the training template can render some conversations differently from the tokenizer's own template (the Qwen3 tool-call history in §6) | Count trained tokens per example; decode the masked span; diff the training template against the serving template |
| Assistant end-of-turn token outside the mask | Model does not stop; outputs reach length limit | Decode the last trained token of each assistant turn; TRL `is_chat_template_stop_token_trained` |
| Different templates across SFT, DPO, RL, and serving | Changed output formatting; missing stop tokens | Render one conversation with each tokenizer and diff ids (open-instruct `diff_tokenizers.py`, `docs/olmo3.md` L47-51) |
| Duplicated BOS after `tokenize=False` rendering | Two BOS ids at sequence start | Inspect the first two ids of every rendered example |
| Generation prompt differs between training data and release | Behaviour appears only on in-domain prompts | Evaluate out-of-domain prompts with the release template |
| Wrapped packing or `bfd` truncation of long examples | Answers separated from questions; truncated traces | Count examples whose tokens are dropped or split |
| Token-mean aggregation with mixed lengths and gradient accumulation | Results change with accumulation steps or GPU count | Log per-example loss weight; compare to sequence mean (Tülu 3 Eqs. 1-2) |
| NEFTune on packed rows | Noise scale below the paper's per-sequence value | Compute α/√(L·d) with row length and with example length |
| Single-format evaluation | Rankings change under another format | Evaluate a sample of equivalent formats (FormatSpread used 10 per task, §4.2) and report the spread |
| New tokens with a frozen `lm_head` under PEFT | New tokens never generated | Check the probability of the new token where it is the target |

## Check your understanding

1. With a correct block-diagonal mask and RoPE, why does resetting position ids leave attention unchanged, and why does the reset still matter in the Hugging Face padding-free path?
2. Using f(i), explain why document masking can reduce training on long relative distances, and what data property avoids that reduction in a long-context stage.
3. The Fewer Truncations ablation shows concatenation plus mask improving perplexity but lowering summarization. What does BFD packing change beyond the mask that could explain the further gains?
4. Why does loss on instructions reduce overfitting most when completions are short, and why can a KL penalty to the base model fail to give the same benefit?
5. Show that per-turn unrolling and all-assistant masking produce identical per-token losses, and name the template property that this argument needs.
6. In the LongAlign weighting worked example, which examples does per-pack token averaging favour, and how does the K/(N_i·M) scale remove that preference?
7. Llama 3 masked formatting tokens in both DPO responses. Using the softmax gradient, explain how pushing down a shared end-of-turn token can produce responses that do not terminate.
8. A new template raises an SFT model's average score by 1.4 points. What additional measurement is required before attributing the gain to the template rather than to seed variance or evaluation-format choice?

## Connections

- Depends on: ch-00 — What General Capability Means and How It Is Measured.
- Previous in order: ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization.
- Next: ch-05 — Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen (tokens per step and context parallelism for long rows).
- ch-07 — Training Failure Modes: Numerical, Masking, and Capability-Level Failures (masking failures as a diagnostic category).
- ch-11 — Tokenizers, Data Provenance, and PII Removal (tokenizer construction before special tokens are added).
- ch-26 — Tool and Function-Calling Data (tool-call and tool-response formats).
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks (template diversity in instruction collections).
- ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate (applies §4-§7 to SFT regimes).
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood (masked failures as content).
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation (§3, §8).
- ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split (tests from the mistakes table).
- ch-39 — Offline Preference Optimization: DPO and Its Variants; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation (format spread as a perturbation audit).

## Sources

- [[sequence-packing]] — Krell et al. 2021 (BERT): padding shares, SPFHP/NNLSHP efficiency, mask and position-id ablations, per-pack loss weighting. The library card version dated 2026-07 misdescribes SPFHP and overstates equivalence; use the excerpt.
- [[fewer-truncations-best-fit-packing]] — Ding et al. 2024: BFD algorithm, compactness, capability and hallucination results, cross-document mask ablation, RoPE position ids not adjusted.
- [[sequence-composition-intra-doc-masking]] — Zhao et al. 2024: intra-document causal masking results on perplexity, in-context learning, and QA at 1.3B.
- [[packing-flash-attention-position-ids]] — Kundu et al. 2024: `cu_seq_len` from `position_ids`, first-label masking, packing vs padding validation loss.
- [[packing-analysis-sft]] — Wang et al. 2024: packing vs padding in SFT at 8B/70B, batch-size/LR relation, single-turn packing MATH drop.
- [[loss-masking-prompt]] — Shi et al. 2024: instruction modelling results, overfitting measurements, KL comparison, training settings. The library card version dated 2026-07 reverses the paper's finding; use the excerpt.
- [[prompt-loss-weight-sift]] — Huerta-Enochian & Ko 2024: prompt-loss weight regression, λ = 0.242, mechanism analysis.
- [[neftune]] — Algorithm 1, AlpacaEval results, overfitting and length controls, α by dataset.
- [[prompt-format-sensitivity-formatspread]] — Sclar et al. 2023: format spread definition and measurements.
- [[mind-your-format-templates]] — Voronov et al. 2024: template sensitivity across 21 models and low template transfer.
- [[flan-collection]] — Longpre et al. 2023: mixed zero-shot/few-shot template ablation.
- [[hf-alignment-handbook]] — Zephyr SFT config and `sft.py` at 1de1fc9. The library card's `train_on_response_only` snippet does not exist in TRL; use the excerpt.
- [[trl-sft-packing-masking]] — TRL at aa89588: packing strategies, padding-free collator, assistant mask checks, template cloning, prefix-preservation check.
- [[transformers-chat-templates]] — Transformers at 05e078a: chat template docs, NEFTune hook, `mean_resizing`.
- [[open-instruct-sft-masking]] — open-instruct at d8a7f1c: all-assistant and last-turn masking functions.
- [[open-instruct-allenai-recipes]] — `docs/olmo3.md`: OLMo 3 template differences across stages and the `<think>` masking bug.
- [[llama-3]], [[llama-3-recipe]] — document mask, chat dialog format, SFT prompt masking, DPO formatting-token masking.
- [[prolong]], [[prolong-recipe]] — document-masking ablation, variable-length throughput, long-data layout, SFT masking and aggregation.
- [[longalign]] — per-example loss weighting under packing and its LongBench-Chat effect.
- [[string-effective-context]] — relative-position frequency f(i) and its relation to effective context length.
- [[tulu-3]] — batch-aggregation equations, sum loss, template variation (Table 13), seed variance (Table 14).
- [[llama-2-long]] — LM loss on long input prompts (Table 9).
- [[openr1]], [[openr1-recipe]] — packing lesson, `<think>` prefill lesson, OpenR1-Distill-7B config.
- [[deepseek-v3-recipe]], [[deepseek-v4-recipe]] — packing and sample-masking practice across versions.
- [[tulu-1-how-far-can-camels-go]], [[nemotron-4-synthetic-recipe]] — assistant-only loss masking in released recipes.
