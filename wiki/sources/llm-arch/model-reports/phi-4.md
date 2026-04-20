<!-- scope: Phi-4 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[olmo-2]], [[llama-1]]
-->

# Phi-4 Technical Report
- **Core Insight:** 40% synthetic data from a stronger teacher can make a 14B model competitive with 70B+ models.
- **Guideline:** Data quality (especially synthetic) can substitute for parameter count when compute-constrained.

- **Organization:** Microsoft Research
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2412.08905
- **Relevant chapters:** Synthetic data for pre-training, data quality over quantity, small model scaling, STEM reasoning, pivotal token training

## Abstract
We present phi-4, a 14-billion-parameter language model developed with a training recipe centrally focused on data quality. Unlike most language models where pre-training is based primarily on organic data sources such as web content or code, phi-4 strategically incorporates synthetic data throughout the training process. While previous models in the Phi family largely distill the capabilities of a teacher model (specifically GPT-4), phi-4 substantially surpasses its teacher model on STEM-focused QA capabilities, achieving strong performance relative to its size — especially on reasoning-focused benchmarks.

## Architecture Summary

| Component | Value |
|-----------|-------|
| Parameters | 14B |
| Layers | 40 |
| Hidden Dimension | 5,120 |
| Attention Heads | 40 |
| KV Heads (GQA) | 10 |
| Head Dimension | 128 |
| Context Length | 16K (extended from 4K during midtraining) |
| Vocabulary Size | 100,352 (tiktoken tokenizer) |

- **Architecture:** Decoder-only Transformer (closely follows phi-3-medium)
- **Attention:** Full attention over context (no sliding window, unlike phi-3's 2K window)
- **Positional encoding:** RoPE
- **Activation function:** SwiGLU
- **Normalization:** RMSNorm

## Key Architectural Innovations

1. **Synthetic data as primary training signal** — 40% of pre-training tokens are synthetic data (~290B unique synthetic tokens, used for 13.8 epochs). This is generated using 50 distinct approaches including multi-agent prompting, self-revision workflows, instruction reversal, and seed curation from web/code sources. The synthetic data strategy is the central innovation, not the architecture.
2. **Pivotal Token Search (PTS)** — a novel post-training technique that identifies critical tokens where the conditional probability of a correct answer shifts significantly. DPO pairs are generated targeting these pivotal moments rather than full-length completions, improving reasoning performance on GPQA and MATH without requiring more data.
3. **Data quality over data quantity** — rather than scaling to trillions of tokens, phi-4 focuses on curating ~10T tokens with high-quality filtering, with synthetic data constituting the bulk of training data. This proves that careful data engineering can compensate for smaller model scale.
4. **Midtraining context extension** — context length is extended from 4K to 16K during a midtraining phase, rather than during post-training, allowing the model to natively learn long-context patterns.
5. **Student surpasses teacher** — phi-4 surpasses GPT-4o (its teacher for synthetic data generation) on STEM and reasoning benchmarks, demonstrating that distilled/synthetic models can exceed their source.

## Design Decisions and Tradeoffs

- **14B parameters (not larger):** Microsoft deliberately targets the "small model" regime, arguing that data quality can compensate for scale. This makes the model deployable on consumer hardware while being competitive with much larger models on reasoning tasks.
- **40% synthetic data:** Extremely high synthetic proportion is unusual. The risk is "hallucination amplification" (synthetic data reinforcing errors), mitigated by using 50 diverse generation approaches and careful seed curation.
- **13.8 epochs on synthetic data:** The synthetic data is so valuable that it is replayed nearly 14 times. This is the opposite of the typical "see each token once" approach, justified by the high information density of synthetic examples.
- **tiktoken over SentencePiece:** Uses GPT-4's tokenizer for better token efficiency, especially on code and structured text.
- **No MoE:** Stays with a dense architecture despite the efficiency benefits of MoE, likely for simplicity and to isolate the contribution of data quality.
- **Full attention over sliding window:** Switched from phi-3's 2K sliding window to full attention over 16K context, accepting higher compute cost for better long-context understanding.

## Training Details

- **Total pre-training tokens:** ~9.8-10 trillion tokens
- **Compute:** 1,920 NVIDIA H100 GPUs

**Data composition:**
| Source | Proportion |
|--------|-----------|
| Synthetic data | 40% (~290B unique tokens, 50 generation types) |
| Web data + web rewrites | 30% (~1.6T total) |
| Code data | 20% |
| Acquired sources (academic, books) | 10% |

- **Optimizer:** AdamW with linear warmup and decay
- **Peak learning rate:** 3e-4
- **Weight decay:** 0.1
- **Global batch size:** 5,760
- **Context extension:** 4K -> 16K during midtraining phase

**Post-training:**
- Supervised Fine-Tuning
- DPO with Pivotal Token Search (PTS)
- PTS generates targeted DPO pairs at critical reasoning tokens

## Performance Highlights

| Benchmark | Phi-4 (14B) | GPT-4o | Qwen2.5-14B |
|-----------|-------------|--------|-------------|
| GPQA | 56.1% | 50.6% | — |
| MATH | 80.4% | 74.6% | — |
| HumanEval | 82.6% | — | — |
| MMLU | 84.8% | — | — |
| AMC-10/12 | ~122 pts avg | — | — |

- Surpasses its teacher model GPT-4o on GPQA (graduate-level STEM) and MATH benchmarks.
- Outperforms Qwen-2.5-14B-Instruct on 9 out of 12 benchmarks.
- Competitive with or exceeds Llama-3.1-70B on reasoning tasks despite being 5x smaller.
- Demonstrates that synthetic data quality and careful curation can substitute for massive scale.
