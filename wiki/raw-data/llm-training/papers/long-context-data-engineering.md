<!-- scope: long-context synthesis — data-engineering principles for 128k context extension
     deps: [[longalign]]
     see-also: [[prolong]], [[long-context-llama3]], [[ruler]]
-->

# Data Engineering for Scaling Language Models to 128K Context
- **Core Insight:** Extending context from 4K → 128K does NOT require exotic architectural changes; a 5-billion-token continued pretraining run with **domain-balanced, length-upsampled** data (Book + CC + ArXiv + GitHub + Wikipedia) and **per-source length upsampling** (preserving the natural domain mix while skewing within each source toward longer documents) is sufficient to reach frontier-matching NIAH performance on 128K context.
- **Guideline:** For 128K extension: 5B tokens continued pretraining, preserve SlimPajama-style domain proportions, upsample *within each domain* to favor longer documents, and use RoPE-base rescaling; do not change the cross-domain mix because that damages short-context ability.
- **Authors:** Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, Hao Peng (U Edinburgh + MIT-IBM + UIUC + UW + AI2)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.10171
- **Relevant topics:** long-context data engineering, 128K extension, length upsampling, RoPE

## Abstract
This paper presents a study of data engineering for scaling LMs to 128K context. Starting from Llama-2-7B/13B, the authors show that 5B tokens of continued pretraining on per-source length-upsampled data (SlimPajama reweighted to favor longer docs within each source) suffices to extend context to 128K with full-precision retrieval over long documents. The work is noted for establishing the "preserve cross-domain proportions, upsample within-domain by length" rule.

## Key Contributions
- Empirical demonstration that **5B tokens** is sufficient for 4K→128K extension.
- **Length upsampling within domain** rule — preserve inter-domain ratios; re-weight within each domain toward long documents.
- Ablations showing that breaking the cross-domain ratio **hurts** short-context while length-upsampling **preserves** it.
- Open recipe + weights for 128K Llama-2-7B.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Data construction
- **Base corpus:** SlimPajama (cleaned Pajama), with 7 sources: CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia, StackExchange.
- **Original cross-domain ratios preserved exactly** (e.g., CC ≈ 67%, Books ≈ 4%).
- **Within each source, upsample by length:** for each source, compute the length histogram; reweight sampling probability so documents longer than 32K get 5× weight relative to short documents.
- **Net effect:** within CC, long web pages are overrepresented; within Books, the full-length books dominate; within GitHub, multi-file repos dominate.
- **Total data:** ~5B tokens sampled under this reweighted distribution.

### Training recipe
- **Continued pretraining:** 5B tokens, context window 80K during training (enough for NIAH at 128K via position extrapolation).
- **RoPE adjustment:** base-θ rescaled from 10K to 200M (NTK-aware).
- **LR:** 2e-5 → 2e-6 cosine.
- **Teacher model(s):** no teacher; pretraining only.
- **Cost / compute:** ~30K A100-hours for 7B model.

## Modality-specific technical details (REQUIRED — long-context)
- **Token-range:** 80K training context; evaluated to 128K.
- **Needle-retrieval difficulty:** authors evaluate with NIAH and a pressure-tested multi-needle variant; observe full retrieval (>99%) at 128K.
- **Document-type mix:** SlimPajama original proportions, length-upsampled within each.
- **Packing strategy:** documents concatenated with document-separator tokens; no cross-document attention masking.
- **Position-encoding adaptation:** NTK-aware RoPE base scaling; no YaRN.
- **Per-stage data mix:** single-stage continued pretraining; no separate SFT stage in this paper.
- **Key ablation:** if cross-domain proportions are changed (e.g., up-weight Books globally), long-context NIAH holds but short-context MMLU drops 3–5 points — the within-domain-only rule is critical.

## Quality / diversity evaluation
- 128K-Llama-2-7B extended model: NIAH @ 128K ~98%; MMLU within 1 point of base.
- At release, matched GPT-4-128K on NIAH retrieval; weaker on complex long-context reasoning.
- Forms the data-engineering baseline adopted by later work (Yi-200K, Qwen-long-context, ProLong).

## Risks + gotchas
- **NIAH is weak evaluation** of real long-context skill — authors acknowledge; complement with RULER / HELMET for comprehensive eval.
- **Length-upsample factor (5×) is empirical** — not theoretically justified; may need retuning per base model.
- **No SFT discussed** — the paper covers CPT only; a separate long-context SFT stage is needed for chat models.
- **Superseded on quality** by ProLong (coherent-doc curation) and per-task synthetic NIAH data.

## Connections
- Successor: [[prolong]] (coherence-filter improves on raw length-upsample).
- Evaluation: [[ruler]] (Hsieh 2024 Nvidia — successor NIAH-style benchmark with diverse synthetic tasks).
- Sibling: [[long-context-llama3]] (Meta's own Llama-3 long-context recipe, informed by this paper).
- Related: [[pose-synthesis]] (PoSE — position-skip pretraining trick).
