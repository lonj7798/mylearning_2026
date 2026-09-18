---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: arXiv:2309.14509v2 (no library card yet)
source_url: https://arxiv.org/abs/2309.14509
created_at: "2026-09-17"
---

# Excerpt: DeepSpeed-Ulysses — system optimizations for extreme long-sequence transformer training

**Artifact:** Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, Yuxiong He (Microsoft). "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models", arXiv:2309.14509v2 (v1 2023-09; v2 2023-10-04).
**Status:** the llm-training library has no `deepspeed-ulysses` card yet; this excerpt holds the verified extract the chapter cites. verl implements this scheme as `ulysses_sequence_parallel_size`.

---

## The problem

Data parallelism addresses batch size, tensor parallelism hidden size, and pipeline parallelism depth. None partitions the sequence dimension, so activation memory per device grows with sequence length. Prior sequence-parallel methods are "constrained by memory-communication inefficiency, limiting their scalability to long sequence large models" (Abstract).

## Mechanism (§1, §3.1)

1. The input sequence of length N is partitioned across P devices; each holds N/P tokens and projects them to local Q, K, V.
2. Before attention, an all-to-all collective redistributes Q, K, V so that "each GPU receives the full sequence but only for a non-overlapping subset of the attention heads" (§1). Attention is then computed per head: `Output_context = Softmax(QK^T/√d)·V` (Eq. 1).
3. A second all-to-all converts the attention output back to the sequence-partitioned (N/P) layout for the MLP, layer norm and remaining operators (§3.1).

The head-subset step implies that the parallelism degree must divide the number of attention heads; verl enforces this in `validate_ulysses_config` (`verl/utils/ulysses.py` L337–341).

## Communication analysis (§3.2, quoted numbers)

- An all-to-all of aggregate message size M over P GPUs costs M/P per link on the clusters considered (intra-node NVSwitch, inter-node fat-tree IB).
- Per transformer layer, the method does one all-to-all of aggregate size 3Nh for the QKV projections and one of size Nh for the output projection, for `4Nh/P` per link, i.e. complexity O(N/P). "Note that this communication volume is constant when both N and P are increased proportionally."
- Megatron-LM sequence parallelism does two all-gathers and two reduce-scatters of size Nh per layer; each costs M rather than M/P when P » 1, giving `4Nh` per link, i.e. O(N) — "P times larger".
- ColAI-SP's ring self-attention is described as linear in message size M.

## Reported results (§1, bullet list)

- Training at 4× the sequence length of the compared systems, "enabling training with sequences with over a million tokens".
- "Communication reduction of over 10x compared to existing systems, resulting in throughput improvements of up to 2.5x, and sustained throughput of over 175 TFlops/GPU (over 54% of hardware peak)."
- Works with dense and sparse attention and with FlashAttention v2; composes with ZeRO-3 for parameter, gradient and optimizer-state sharding.

Conditions: 2023 measurements on the authors' hardware against the baselines named in §2 (Megatron-LM sequence parallelism, ColAI-SP). The paper reports no LLM-RL setting and no evaluation of downstream quality.

## Why it matters for an RL run (course note, not from the source)

Sequence parallelism applies to the training engines. The rollout engine in a framework such as verl is a separate process with its own parallelism, so raising the sequence-parallel degree lengthens what can be back-propagated, not what can be generated; generation length is bounded by the inference engine's `max_model_len` and the response-length clamps.

## Connections

- [[verl-rollout]] — the rollout-side length clamps that bound generation independently of this mechanism.
- [[verl-ppo-loss]] — `agg_loss`'s global-batch aggregation, which the docstring says makes the loss invariant to the FSDP or Megatron parallelism layout.

## Verification

- Checked on 2026-09-17 against arXiv:2309.14509v2: Abstract, §1 (contribution bullets), §2 (comparison to ColAI-SP and Megatron-LM), §3.1 (design, Eq. 1), §3.2 (communication analysis).
- Not reported by the source: any measurement in an RL post-training loop; any head-count divisibility requirement stated as such (it follows from the head-subset step and is enforced in verl's code).
