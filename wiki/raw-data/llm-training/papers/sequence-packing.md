<!-- scope: sequence packing — concatenate short sequences into one fixed-length pack, with attention-mask and positional-embedding changes that preserve equivalence
     deps: []
     see-also: [[packed-vs-unpacked-ablation]], [[fsdp-sft]], [[loss-masking-prompt]]
-->

# Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance
- **Core Insight:** In the Wikipedia BERT pre-training dataset at sequence length 512, 4.17 billion of 8.33 billion tokens are padding; packing short sequences into full-length packs and adjusting the attention mask and positional embeddings raises throughput to 1.913× the unpacked baseline at 99.7% packing efficiency while matching unpacked convergence and SQuAD 1.1 scores (§2, Table 1, Table 3).
- **Guideline:** When the training data has a skewed length distribution, pack it offline with NNLSHP at packing depth 3 (or SPFHP when the dataset is too large for NNLSHP's memory), and apply both model changes — per-sequence attention masking and per-sequence positional indices — because removing the positional-embedding adjustment leaves MLM accuracy at 71.8% against the 72.1% target (§3.2, §4.2.1).
- **Authors:** Mario Michael Krell, Matej Kosec, Sergio P. Perez, Andrew Fitzgibbon (Graphcore)
- **Year:** 2021 (arXiv v1 2021-07; v2 2022-10-05)
- **URL:** https://arxiv.org/abs/2107.02027
- **Source type:** paper
- **Relevant topics:** bin packing, padding removal, attention masking, positional embeddings, BERT pre-training throughput

## Abstract
Handling variable-length sequences on accelerators is commonly done by padding every sequence in a batch to a fixed length. The authors show that in common NLP datasets up to 50% of all tokens are padding, and up to 89% in the case of GLUE-cola at sequence length 128. Existing remedies are complicated by cross-contamination in self-attention, by loss of sequence-ordering information, or by accelerator-specific kernels. The paper restates sequence packing as a bin-packing problem, presents two new algorithms, and shows a 2× speedup for BERT phase-2 pre-training. It also gives the model changes needed for the packed model to be mathematically equivalent to the unpacked one, so existing pre-training and fine-tuning practices remain valid.

## Key Contributions
- Publishes sequence-length histograms for Wikipedia BERT pre-training, GLUE, SQuAD 1.1, LibriSpeech text and audio, and QM9 molecules, with the theoretical speedup implied by each (Figure 1, §2).
- Introduces SPFHP, a histogram-level worst-fit packing heuristic, and NNLSHP, which solves packing as a non-negative least-squares problem over pre-enumerated pack strategies (§3.1).
- Specifies the two BERT changes that preserve equivalence: per-sequence positional indices and a per-sequence self-attention mask, plus a per-sequence loss and accuracy computation (§3.2).
- Gives hyperparameter guidance for the increased effective batch size: either reduce gradient accumulation by the packing factor, or adjust the LAMB decay parameters as β1 := β1^p and β2 := β2^p (§3.3).
- Shows the load-balancing argument for packing over un-padding as accelerator count grows (§4.4, Figure 5).

## Key Figures/Tables to Study
- **Figure 1:** length histograms and the theoretical speedup per dataset and maximum sequence length.
- **Table 1:** packing efficiency, packing factor, overhead, and realized speedup per algorithm and packing depth on IPU.
- **Figure 3:** packed vs unpacked learning curves under three hyperparameter settings, and the realized speedup.
- **Figure 4:** the ablation showing what happens without the mask adjustment and without the positional-embedding adjustment.
- **Table 2 and Table 3:** measured speedups in full BERT pre-training, and the resulting SQuAD 1.1 scores.

## Technical Details

### Padding in the data
- Wikipedia BERT pre-training at sequence length 512: 8.33 billion tokens, of which 4.17 billion are padding; the theoretical speedup from removing all padding is 2.001 (§2, Figure 1).
- The same dataset at sequence length 128 gives a theoretical speedup of about 1.2, and at 384 about 1.7 (§2).
- Sequences of the full length 512 are 23.5% of the dataset (§1, Figure 1).
- The skew is not specific to Wikipedia: SQuAD 1.1 gives 2.2× and QM9 molecules 1.6×; LibriSpeech audio realizes only 1.3× against a 1.6× theoretical maximum because it is skewed toward long sequences (§2).

### SPFHP (shortest-pack-first histogram-packing)
Operates on the sequence-length histogram with bin size 1 rather than on individual samples. The histogram is traversed from longest to shortest. Each histogram bin is assigned by the worst-fit rule to the pack with the most remaining space; if it does not fit completely, a new pack is created. A maximum packing depth limits how many sequences a pack may hold. Time complexity is O(n + s_m²) and space complexity O(s_m²), where n is the number of samples and s_m is the maximum sequence length (§3.1.1). Processing time on the paper's CPU was about 0.03 s independent of packing depth, against 87–120 s for classical first-fit-decreasing (§4.1).

### NNLSHP (non-negative least-squares histogram-packing)
Solves wAx = wb for x ≥ 0, where b is the histogram of sequence-length counts, each column of A is a "strategy" — a combination of sequence lengths summing exactly to the maximum sequence length — and x is the non-negative repetition count of each strategy. Strategies are restricted to at most 3 sequences per pack. A residual weight of 0.09 is applied to sequences of length 8 or shorter and 1 to all other lengths; this weighting changes the speedup by less than 1%. The floating-point solution is rounded to integers, introducing an error of a few hundred sequences against millions. Time complexity O(n + s_m⁵), space complexity O(s_m³) (§3.1.2). It required 28.4 s on the paper's CPU at depth 3 and ran out of memory for larger depth (§4.1).

### Model changes
- **Positional embeddings (§3.2.1):** the bias-add implementation is replaced by an embedding look-up, because the position index must reset at each sequence boundary. For a pack of one length-2 and one length-3 sequence the indices are [0, 1, 0, 1, 2]. An extra input carries each token's position within its own sequence.
- **Attention mask (§3.2.2):** a per-token sequence id is used; `zero_one_mask = tf.equal(mask, mask.T)` builds the block-diagonal mask that blocks attention between sequences in the same pack. The full attention matrix is still computed and then masked (§4.4).
- **Loss (§3.2.3):** NSP and downstream fine-tuning losses and accuracies are computed per sequence inside the pack.

### Hyperparameter adjustment (§3.3)
Packing multiplies the average number of sequences per weight update by the packing factor p. One option is to divide the gradient accumulation count by p and change nothing else. The alternative keeps the batch size and updates the LAMB decay parameters as β1 := β1^p and β2 := β2^p; for p = 2 this reproduces the exact momentum and velocity of updating twice with the same gradient. Scaling the learning rate with batch size reduced convergence speed in the paper's experiments (§3.3, §4.2).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| BERT (MLPerf v0.7 phase 2) | large | pretrain (phase 2) | packing algorithm and depth | NNLSHP, depth 3 | arXiv:2107.02027v2 §4.1 | verified (2026-09-18) | §4.1 Table 1: depth 3 NNLSHP gives efficiency 99.7%, packing factor 2.00, realized speedup 1.913, the best of the compared settings |
| BERT (MLPerf v0.7 phase 2) | large | pretrain (phase 2) | sequence length; target MLM accuracy; samples | 512; 71.2%; 3 million sequences | arXiv:2107.02027v2 §4.2 | verified (2026-09-18) | benchmark definition, not an ablation |
| BERT (MLPerf v0.7 phase 2) | large | pretrain (phase 2) | batch size and LAMB decay in the packed run | batch size 1500, packing factor 2, β adjusted to 0.66 from 0.81 | arXiv:2107.02027v2 §4.2, Figure 3 | verified (2026-09-18) | §4.2: LAMB decay adjustment matches performance at later training stages; learning-rate scaling decreased performance |
| BERT | base | pretrain + SQuAD 1.1 fine-tune | packing factor; realized speedup at seq len 384 | 1.70; 1.68 | arXiv:2107.02027v2 Table 2 | verified (2026-09-18) | Table 3: packed SQuAD F1 88.32 vs 88.5 reference; EM 81.03 vs 80.8 |
| BERT | large | pretrain + SQuAD 1.1 fine-tune | packing factor; realized speedup at seq len 384 | 1.70; 1.69 | arXiv:2107.02027v2 Table 2 | verified (2026-09-18) | Table 3: packed SQuAD F1 90.65 vs 90.9 reference; EM 84.12 vs 84.1 |
| BERT | base, large | pretrain (phase 1) | packing factor; realized speedup at seq len 128 | 1.17; 1.15 | arXiv:2107.02027v2 Table 2 | verified (2026-09-18) | no ablation reported |
| BERT | large | pretrain (phase 2) | overhead from masking and loss adjustment | 4.29% at NNLSHP depth 3 (4.28–4.48% across SPFHP depths) | arXiv:2107.02027v2 Table 1 | verified (2026-09-18) | §5 states the adjustments cost less than 5% |

Hardware for the speedup measurements is the IPU-M2000 with 16 accelerator chips, using cycle counts from the precompiled kernel rather than wall-clock timing; the measurements are deterministic and were not repeated (§4.1).

## Findings relevant to generality
- SQuAD 1.1 scores after full packed pre-training are within 0.3 points of the reference: F1 88.32 vs 88.5 for base and 90.65 vs 90.9 for large, with EM 81.03 vs 80.8 and 84.12 vs 84.1, each the median of 10 seeds. Fine-tuning itself was run unpacked (§4.3, Table 3).
- The ablation separates the two model changes: without the attention-mask adjustment, training loss and accuracy worsen and do not recover with longer training; without the positional-embedding adjustment, loss and accuracy nearly match but accuracy stalls at 71.8% against the 72.1% target (§4.2.1, Figure 4).
- Removing the NSP loss from packed BERT reduced SQuAD F1 by 1.31% and EM by 1.15%, so the authors keep NSP (§4.2.1).
- Packing is load-balanced across devices, whereas the speedup from un-padding falls to 50% at 32 accelerators and 30% at 2048 (§4.4, Appendix).

## Connections
- [[packed-vs-unpacked-ablation]] collects the later studies that compare packed and padded SFT; this card is the primary artifact those later studies build on.
- [[loss-masking-prompt]] defines the per-sequence loss masking that must be applied inside each pack for SFT.
- [[fsdp-sft]] covers the sharded SFT setting in which packing changes per-step memory and step count.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2107.02027 (arXiv v2, 2022-10-05).
- Corrections to the previous card version:
  - "Instruction datasets have 50–89% padding tokens" → the measurements are BERT/GLUE-style: up to 50% of all tokens in the Wikipedia BERT pre-training dataset, and up to 89% for GLUE-cola at sequence length 128 (Abstract, §2). The paper does not study instruction datasets.
  - "2× SFT throughput with zero accuracy impact" → the 2× figure is BERT phase-2 pre-training at sequence length 512; realized speedup is 1.913 by cycle estimate and exceeds 2× in Figure 3 once data-transfer latency is included (§4.1, §4.2). The paper reports no SFT experiment.
  - "SPFHP ... fill each with the longest remaining sequence first; for each subsequent fit, pick the longest sequence that still fits (shortest gap)" → SPFHP applies the worst-fit rule: the histogram bin under consideration goes to the pack with the most space remaining (§3.1.1).
  - "Achieves near-optimal packing ratio (>99% fill) in O(N log N)" → SPFHP is O(n + s_m²) time and O(s_m²) space; efficiency is 89.4% at depth 3, 98.9% at depth 8, and 99.6% at unlimited depth (§3.1.1, Table 1).
  - "NNLSHP ... provides mathematically optimal packing strategies" → NNLSHP solves a weighted non-negative least-squares problem restricted to at most 3 sequences per pack and rounds the solution to integers; the paper calls the result nearly optimal, not optimal (§3.1.2).
  - "Figure 3: Block-diagonal attention mask" and "Table 3: Throughput gains on Wikipedia phase 2 (~2×)" → the mask is in §3.2.2; Figure 3 is the learning-curve comparison; throughput is in Table 1 and Table 2; Table 3 holds SQuAD 1.1 scores.
- Removed as unsupported by the source: the FlashAttention varlen interface, `flash_attn_varlen_func`, `cu_seqlens`, and the O(sum L_i) memory claim; "Directly enables FlashAttention-2's varlen kernels used in modern SFT"; labels set to -100 as ignore_index; TRL `DataCollatorWithPacking` and Axolotl `sample_packing`; "pad to the longest packed block, not to global max length"; the claim that the softmax partition function leaking across documents "changes gradients even on sequence 1 tokens" (the paper states that cross-contamination reduces accuracy and that separator tokens do not prevent it, and reports a 0.35% average F1 reduction from RoBERTa, but does not give this gradient argument).
- Not reported by the source: any decoder-only or causal-LM experiment, any GPU wall-clock measurement of the authors' own implementation, and any SFT or instruction-tuning result.
