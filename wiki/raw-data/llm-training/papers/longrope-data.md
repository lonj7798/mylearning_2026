<!-- scope: long-context synthesis — LongRoPE data companion and fine-tune recipe
     deps: [[long-context-data-engineering]]
     see-also: [[prolong]], [[qwen-long-context-synth]]
-->

# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens
- **Core Insight:** Context extension is not purely a data problem — the right RoPE-extension scheme (non-uniform per-dimension base rescaling found via evolutionary search) is just as important; LongRoPE demonstrates 2M+ token context on LLaMA-2 with only a small fine-tuning corpus (< 1B tokens) by decoupling the RoPE math from the data volume.
- **Guideline:** When extending context, don't rely on uniform NTK-aware or YaRN rescaling; search for non-uniform per-dimension rescaling factors (LongRoPE uses evolutionary search with a needle-retrieval fitness function) — the right RoPE can cut fine-tune data by 10×.
- **Authors:** Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, Mao Yang (Microsoft Research Asia)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.13753
- **Relevant topics:** RoPE extension, context scaling, evolutionary search, position encoding

## Abstract
LongRoPE extends LLMs to 2M+ token context through three contributions: (1) identifying non-uniformities in RoPE frequency dimensions and position distribution, (2) an evolutionary search algorithm that finds optimal per-dimension rescaling factors, (3) a two-stage progressive extension schedule that enables 2M-token context with only ~1B tokens of fine-tuning. Applied to LLaMA-2 and Mistral, LongRoPE-extended models maintain strong short-context performance while excelling at long-context retrieval and reasoning.

## Key Contributions
- **Non-uniform RoPE rescaling** with **per-dimension factors** found via evolutionary search.
- **Two-stage progressive extension:** 4K → 256K → 2048K with data-efficient fine-tuning at each stage.
- **2M-token context on LLaMA-2** with sub-1B-token fine-tune budget.
- Companion fine-tune data recipe (LongRoPE-data).

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Evolutionary search for RoPE factors
- **Search space:** per-dimension rescaling factor λ_i for each of the 128 RoPE dimensions.
- **Fitness function:** perplexity on a held-out long-context corpus + NIAH retrieval accuracy.
- **Algorithm:** evolutionary strategy with population size 64, 40 generations, mutation rate 0.3.
- **Initial population:** NTK-aware, YaRN, and uniform rescaling schemes seeded.
- **Output:** optimized λ_i vector per target context length (256K, 1M, 2M).

### Two-stage fine-tuning data
- **Stage 1 — 256K extension:**
  - Data: long documents 64K–256K tokens from Books, ArXiv, long web. ~300M tokens.
  - Fine-tune with optimized 256K RoPE factors.
- **Stage 2 — 2048K extension:**
  - Data: concatenated multi-document long sequences 256K–2048K tokens. ~600M tokens.
  - Fine-tune with optimized 2M RoPE factors.
- **Output shape:** ~1B tokens of fine-tune data total.
- **Teacher model:** none; pretraining-only fine-tuning.
- **Cost / compute:** ~3K A100-hours for 7B model fine-tune.

## Modality-specific technical details (REQUIRED — long-context)
- **Token-range:** 256K (stage 1), 2M (stage 2).
- **Needle-retrieval difficulty:** stage-2 model achieves >90% NIAH at 2M.
- **Document-type mix:** books heavy, concatenated multi-doc for longest contexts.
- **Packing strategy:** documents padded with document-break tokens; no cross-document attention.
- **Position-encoding adaptation:** the whole point — **non-uniform per-dimension RoPE rescaling** found via evolutionary search. Contrasts uniform NTK-aware and YaRN.
- **Per-stage data mix:** stage 1 natural long docs; stage 2 adds synthetic multi-doc concatenation.

## Key equation (REQUIRED — per-dimension RoPE rescaling)

Standard RoPE applies rotation at frequency `θ_i = θ^(2i/d)` for dimension i. NTK-aware scales `θ → θ · s^(d/(d-2))`. LongRoPE generalizes:

```
θ_i' = θ_i / λ_i
```

where λ_i is a **per-dimension** rescaling factor learned via evolutionary search. Uniform YaRN is the special case λ_i = constant.

## Quality / diversity evaluation
- LLaMA-2-7B-LongRoPE-2M: NIAH 2M ~90%, passkey retrieval 2M ~98%.
- Short-context MMLU, GSM8K: within 1 point of base.
- At release (Feb 2024), the longest open-context model (pre-Qwen-2.5-1M, pre-Gemini-1.5).

## Risks + gotchas
- **Evolutionary search cost:** finding λ_i takes ~1000 fitness evaluations; each evaluation is a long-context forward pass.
- **Non-uniform λ may generalize poorly** to very different base models; search must be repeated.
- **Real reasoning at 2M is weak:** NIAH passes, but complex multi-hop over 2M context is not reliably solved.

## Connections
- Position-encoding ancestors: RoPE (Su 2021), YaRN (Peng 2023), NTK-aware RoPE (bloc97 blog).
- Fine-tune data sibling: [[long-context-data-engineering]], [[prolong]].
- Used in Qwen's long-context pipeline alongside YaRN / DCA — see [[qwen-long-context-synth]].
