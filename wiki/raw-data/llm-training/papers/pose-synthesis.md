<!-- scope: long-context synthesis — PoSE position-skip-wise training for context extension
     deps: [[long-context-data-engineering]]
     see-also: [[prolong]], [[longrope-data]]
-->

# PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training
- **Core Insight:** You don't need to train on actually-long sequences to extend context — synthesizing "fake long" training samples via **position-skipping** (train on short chunks but inject large position-index gaps between chunks to simulate long-context position IDs) lets a model learn to attend across extended position ranges while keeping the per-sample compute the same as short-context training.
- **Guideline:** For cheap context extension when you lack long-document data or long-context compute, use PoSE: take 4K-token training samples, randomly split each into two chunks, and shift the second chunk's position IDs by a random large offset ∈ [0, target_length − 4K]; the model learns long-position behavior with short-context compute.
- **Authors:** Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, Sujian Li (Peking University + Microsoft Research)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.10400
- **Relevant topics:** context extension, position encoding, compute-efficient training, PoSE

## Abstract
PoSE (Positional Skip-wise Training) trains a model to handle extended context without using actually-long training sequences. During training, each 4K-token sample is split into two sub-sequences; the position IDs of the second sub-sequence are shifted by a random large offset, simulating a long-context position distribution while keeping the actual sequence length short. Applied to LLaMA, PoSE extends context from 2K → 128K with ~4× less compute than full-length training, achieving competitive NIAH and perplexity results.

## Key Contributions
- **Position-skip training trick** — simulated long-position distribution with short sequences.
- **4× compute reduction** vs actually-long training.
- Up to 128K context extension on LLaMA-7B with modest fine-tune.
- Open implementation and pretrained checkpoints.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### PoSE training data construction
- **Source:** standard pretraining corpus (e.g., Pile, SlimPajama) — no long-document filter needed.
- **Sample preparation:** each training sample is 4K tokens.
- **Position-skip injection:**
  1. Split the 4K sample into two chunks at a random position.
  2. Assign chunk 1 positions [0, L1-1] as normal.
  3. Assign chunk 2 positions [L1 + δ, L1 + δ + L2 - 1] where δ is a random integer ∈ [0, target_context - 4K].
  4. Attention mask still allows chunk 1 tokens to attend to chunk 2 according to the positions (not the linear order).
- **Loss:** standard language-modeling loss; the model sees tokens with artificial large-gap position IDs.

### Training recipe
- **Base model:** LLaMA-7B (originally trained at 2K context).
- **Target extensions:** 16K, 64K, 128K.
- **RoPE base rescaling** to match target (standard NTK-aware).
- **Compute:** ~1/4 of full-length training because each sample is still only 4K tokens.
- **Output shape:** standard pretraining corpus reused; no new synthesis.
- **Teacher model:** none.
- **Cost / compute:** small — ~1K A100-hours for 128K extension on 7B.

## Modality-specific technical details (REQUIRED — long-context)
- **Token-range:** trained with 4K samples, supports up to target_context (16K–128K).
- **Needle-retrieval difficulty:** LLaMA-7B-PoSE-128K achieves ~90% on NIAH 128K.
- **Document-type mix:** inherits pretraining mix; PoSE is data-agnostic.
- **Packing strategy:** standard; PoSE modifies position IDs, not packing.
- **Position-encoding adaptation:** RoPE base rescaled + position-skip simulation.
- **Key equation:**
  ```
  pos_id_chunk2 = original_pos + δ, δ ~ Uniform[0, target_ctx − 4K]
  ```

## Quality / diversity evaluation
- LLaMA-7B-PoSE-128K: NIAH 128K ~90%, perplexity within 0.1 of full-length-trained baseline.
- PoSE-32K outperforms LongLoRA-32K at equivalent fine-tune budget.
- Short-context ppl preserved.

## Risks + gotchas
- **PoSE simulates position distribution, not content distribution** — real long-range coherence (semantic dependency across 100K tokens) is not learned.
- **Works for retrieval-style tasks, weaker on reasoning** — BABILong-style results lag behind full-length-trained models.
- **Requires careful interaction with RoPE extension** — wrong base-θ rescaling breaks PoSE.

## Connections
- Complementary to long-document CPT approaches: [[long-context-data-engineering]], [[prolong]].
- Sibling position-trick paper: NTK-aware / YaRN / [[longrope-data]].
- Uses trick similar to position-interpolation (Chen 2023).
