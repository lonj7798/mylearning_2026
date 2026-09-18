<!-- scope: former synthesis card on packed vs padded (unpacked) SFT; no single primary artifact exists; lists verified primary studies instead
     deps: [[sequence-packing]], [[loss-masking-prompt]]
     see-also: [[tulu-3]], [[hf-alignment-handbook]]
-->

# Packed vs Unpacked SFT — Former Ablation Synthesis Card
- **Source type:** none (synthesis card without a primary artifact)
- **Status:** no verifiable primary source. Chapters must not cite this card; cite the works listed below instead.

> **No verifiable primary source.** The previous card described itself as a compilation of "Krell 2021, plus 2023–2024 ablations
> in Tülu 3 paper, HF Alignment Handbook, Axolotl docs, Megatron-LM SFT blog" and named no single artifact. Searched: its two URLs,
> arXiv:2107.02027 (v2; already carded as [[sequence-packing]]) and arXiv:2411.15124 (v5), plus huggingface/alignment-handbook@1de1fc9.
> Tülu 3 contains no packing ablation. The Axolotl and "Megatron-LM SFT blog" sources had no URL and were not searched for (limited web-search budget). Two primary
> studies that compare packed and padded SFT directly (Kundu et al. 2024; Wang et al. 2024) were read and are listed below; the
> previous card did not cite them, and no library card exists for them yet.

## Verifiable pointers to related works
Each item was read at the stated locus on 2026-09-14. Each item is a fact about that work only.

- **Krell et al., "Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting
  Performance", arXiv:2107.02027 (v1 2021-07; v2).** Setting: BERT, a bidirectional encoder, not a decoder SFT run. Up to 50% of
  tokens can be padding, and 89% for GLUE-cola at length 128 (abstract). A block-diagonal attention mask and per-sequence position
  indices keep the packed model equivalent to the unpacked one (§3.2.1-3.2.2). Ablation: without the mask adjustment, training loss
  and accuracy "worsen drastically" and longer training does not recover them; without the position adjustment, MLM accuracy stalls
  at 71.8% against a 72.1% target (§4.2.1, Fig. 4). Realized phase-2 speed-up at length 512 exceeds 2x (§4.2, Fig. 3). After packed
  pre-training, SQuAD 1.1 F1 is 88.32 (base) and 90.65 (large) versus reference 88.5 and 90.9 (§4.3, Table 3). Card: [[sequence-packing]].
- **Kundu, Lee, Wynter, Ganti, Mishra (IBM Research), "Enhancing Training Efficiency Using Packing with Flash Attention",
  arXiv:2407.09105 (v1 2024-07; v6).** Mechanism: position IDs restart at 0 for each packed example; `cu_seq_len` is computed from
  `position_ids` and `flash_attn_varlen_func()` is called (§3.3); the collator sets the first label of each example to -100 (§3.4-3.5).
  Available as `DataCollatorWithFlattening` in Transformers 4.44 and `padding_free=True` in TRL's `DataCollatorForCompletionOnlyLM` (§3.1).
  Setting: 20K-example subsets of FLAN, OrcaMath, and The Stack; one node of 8 A100-80GB GPUs with FSDP; one epoch, gradient
  accumulation 2, maximum sequence length 4096, minibatch 4 per GPU (§4, §4.1). Mistral-7B on FLAN_20k (Table 2):

  | Batching | Tokens/s | Validation loss |
  |---|---|---|
  | padding | 742 | 1.129 |
  | packing without position IDs | 2986 | 1.306 |
  | offline packing with position IDs (585 packed rows) | 3010 | 1.284 |
  | online minibatch packing with position IDs | 1408 | 1.127 |

  The authors attribute the higher validation loss of offline packing, even with position IDs, to far fewer optimisation steps in
  one epoch; minibatch packing keeps the number of steps and matches the padding loss (§4.1, §4.3, Tables 4-6). The benefit is consistent across architectures
  except Gemma-7B and Qwen1.5-MoE-A2.7B (§4.1). Bin-packing sample selection adds limited benefit (§4.3).
- **Wang, Wang, Wang, Li, Hovy, Guo, "Packing Analysis: Packing Is More Appropriate for Large Models or Datasets in Supervised
  Fine-tuning", arXiv:2410.08081 (v1 2024-10; v3).** Compares padding, random packing, and greedy packing on LLaMA-3-8B and
  LLaMA-3-70B with SFT sets of 69K to 1.2M conversations (§3-4). Settings (Table 2): LR 1e-5, maximum sequence length 4096, warm-up
  ratio 0.2, 4 epochs (8B) or 3 epochs (70B), per-GPU batch 2 (8B) or 1 (70B), gradient accumulation 2, DeepSpeed stage 3, 4 nodes
  of 8 A800-80GB GPUs; loss only on tokens after the assistant header (§4.1.2). The paper does not state that attention is reset
  between packed conversations; §3.3.3 argues that the [EOS] token lets the model separate adjacent samples. Average benchmark scores
  (Table 3), padding / random packing / greedy packing: WildChat (GPT-4) 69K, 8B 49.58 / 49.46 / 50.6; same data, 70B
  61.50 / 65.97 / 65.92; Open-source 1M, 8B 54.3 / 54.95 / 55.05; same data, 70B 66.12 / 67.26 / 67.54. Training time, 70B on
  WildChat (GPT-4): 9533 s padding versus 3749 s random packing; samples per second are lower with packing, for example 21.13 versus
  20.48 (8B, WildChat (GPT-4), greedy packing) (Table 5). With packing, the linear batch-size/learning-rate relation observed for
  padding does not hold (§5.3, Fig. 2). Packing a single-turn-only set (filtered 200K OpenHermes 2.5) gave a significant MATH drop
  that returned to normal after adding 1/40 to 1/20 multi-turn conversations; an internal 200K single-turn set showed no drop (§5.3,
  Fig. 3). No seeds or run-to-run variance are reported.
- **Lambert et al., "Tülu 3: Pushing Frontiers in Open Language Model Post-Training", arXiv:2411.15124 (v5).** No packing ablation.
  §4.3.2 (Batch Aggregation) shows that averaging the loss over non-padding tokens changes sample weighting when gradient
  accumulation or distributed training splits a batch: one forward pass gives `L = (l_1 + l_2) / (n_1 + n_2)`, while accumulation
  gives `L = (l_1/n_1 + l_2/n_2) / 2`, where `l_k` is the summed token loss of sample k and `n_k` its number of non-padding tokens
  (Eqs. 1-2). Tülu 3 uses a sum loss; on Llama 3.0 with the Tülu 2 mixture, sum loss with LR 5e-6 and 2 epochs worked best
  (Figs. 5-6). Card: [[tulu-3]].
- **huggingface/alignment-handbook@1de1fc9.** `recipes/zephyr-7b-beta/sft/config_full.yaml` has no `packing` key
  (`max_seq_length: 2048`, `attn_implementation: flash_attention_2`). The usage example in the docstring of `scripts/sft.py` passes
  `--packing` (L26). Neither file states that packing changes only throughput. Card: [[hf-alignment-handbook]].

## Connections
- [[sequence-packing]] — primary card for Krell et al.; mask and position-ID mechanics.
- [[loss-masking-prompt]] — prompt-token loss masking; Wang et al. mask instruction tokens in both padded and packed runs (§4.1.2).
- [[tulu-3]], [[hf-alignment-handbook]] — cards for two of the works above.
- [[sequence-packing]], [[loss-masking-prompt]], [[open-thoughts]], [[openr1]] — library cards that link here; they should cite the works above instead.
- Chapters ch-30 and ch-36 cite this card for failure modes, a diagnostic threshold, and a throughput formula. They should cite
  the works above; the items under "Removed" have no source.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2107.02027 (v2); https://arxiv.org/abs/2411.15124 (v5);
  https://arxiv.org/abs/2407.09105 (v6); https://arxiv.org/abs/2410.08081 (v3); github.com/huggingface/alignment-handbook@1de1fc9.
- Corrections to the previous card version:
  - "Krell 2021 (BERT phase 2): 2× throughput, identical pretraining loss and GLUE scores" → learning curves nearly match when
    normalized by samples processed at the same average batch size (§4.2, Fig. 3 left); the downstream check is SQuAD 1.1, not GLUE (§4.3, Table 3).
  - "Tülu 3 (Llama-3.1-8B SFT): packed vs unpacked same MT-Bench within 0.05 pts; 2.5× throughput" → Tülu 3 v5 reports no packing
    comparison; MT-Bench appears only in its reference list.
  - "HF Alignment Handbook v1 (Mistral-7B Zephyr): packed default, explicit note that disabling packing changes only throughput" →
    no such note at @1de1fc9, and the Zephyr SFT config has no packing key; older handbook commits were not checked.
  - "When masks are missing, packing hurts" (stated as general) → in Krell et al. (BERT) the missing mask degraded loss and accuracy
    (§4.2.1); in Kundu et al. packing without position IDs had higher validation loss (Table 2); Wang et al., who do not state a mask
    reset, report greedy-packing averages above padding in all 8 model-dataset settings and random-packing averages below padding in 2 of 8 (Table 3). For decoder SFT the size of the cross-contamination
    effect is an open question.
  - "Packed and unpacked SFT produce the same loss trajectory when masks are correct" → Kundu et al. report this only when the number
    of optimizer steps is kept (minibatch packing); offline packing with position IDs had higher validation loss (§4.1, Table 2).
- Removed as unsupported by any listed source:
  - "Axolotl `sample_packing` community benchmarks show ≤ 0.5 pt difference"; "Megatron-LM SFT blog" ablations.
  - "packed SFT with FlashAttention varlen matches unpacked SFT on MT-Bench, AlpacaEval, and IFEval, at 2–3× higher throughput".
  - "~2–3× at short sequence lengths, diminishing at long" and "treat packing as a pure throughput optimization".
  - The four failure modes as attributed findings (cross-document leakage causing "subtle quality drop on multi-turn evals";
    un-reset position IDs; label mask not re-applied per sub-sequence; `flash_attn_func` instead of `flash_attn_varlen_func`). The
    first two mechanisms are described by Krell et al. and Kundu et al. above; the quality claims and modes 3-4 appear in no source.
  - The diagnostic procedure (100-step runs; "differences > 0.01 nats indicate a mask/pos-ID bug").
  - The throughput model "speedup ≈ L_max / avg(L_i)" and "mean length 600, L_max 4096 → ≈ 6×, often realized as ~3×".
