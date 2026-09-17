<!-- chapter: ch-32b
     track: midtraining
     kind: content
     title: Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression
     deps: [ch-32a]
     sources: [[position-interpolation]], [[yarn]], [[eleuther-extending-the-rope]], [[hf-transformers-yarn-code]], [[llama-2-long]], [[rope-base-bounds-context-length]], [[longrope2]], [[dual-chunk-attention]], [[string-effective-context]], [[controlled-long-context-extension]], [[prolong]], [[prolong-recipe]], [[fu-2024-data-engineering-128k]], [[llama-3]], [[llama-3-recipe]], [[qwen-long-context-synth]], [[longred]], [[longpo]], [[skyladder]], [[qwen-3-long-context]], [[olmo-3-long-context]], [[gemma-3-long-context]], [[harm-de-vries-long-context-run]]
     figures: figures/rope-scaling-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression

> **Core insight.** A RoPE model pretrained at a short length can be extended to 64K–512K tokens with a continued-training stage that is small relative to pretraining: 5B tokens in [[fu-2024-data-engineering-128k]], 40B in [[prolong]], 50B–100B in [[olmo-3-long-context]], about 800B in [[llama-3]]. The position-encoding change (interpolation, base increase, YaRN) decides whether the model can use the new positions at all. The data mixture decides how much short-context ability is lost: in every controlled comparison in this chapter, more long data or a narrower long-data source worsened short-context scores or short-context validation loss, and mixing high-quality short data back in reduced the loss (ProLong Fig. 3; Olmo 3 §3.6.3; Fu et al. Table 5). No reviewed recipe reports zero short-context cost across all benchmarks.
>
> **Guideline.** When extending context by continued training, change RoPE (base increase or YaRN) and keep 37–66% high-quality short data in the mix (37% in the final ProLong recipe, 66% in Olmo 3), because ProLong's 5B-token Llama-3-8B ablations showed short-task averages falling monotonically as the long share rose, and Olmo 3's 10B-token test lost 2.5 points at 66% long versus 0.8 points at 34% long. Advance to the next length only when short-context evaluations have recovered and retrieval at the current length is solved, as Llama 3 did for its six stages. When only inference-time extension is possible, prefer length-dependent methods (dynamic scaling, DCA) over a static YaRN factor, because the Qwen3 model card warns that static YaRN may lower quality on short inputs. Otherwise, when short-context quality is the priority and no retraining budget exists, keep the original context window.

## Why this chapter matters for a general-purpose model

A general-purpose model is asked to read long documents, repositories, tool transcripts, and multi-turn histories, and it is also asked short questions about knowledge, math, and code. Context-length extension is the stage that adds the first ability, and it is also a stage where the second ability can be lost.

The stage sits in mid-training. Pretraining runs at 4K–32K tokens because attention cost grows with sequence length and because long natural documents are rare (§1.4). After pretraining, a continued-pretraining stage at longer sequence lengths extends the window. The Olmo 3 authors summarize three placements in published recipes: Llama 3.1 applies extension "prior to midtraining", Qwen 2.5 and Qwen3 "perform it afterwards", and GLM 4.5 extends "only after supervised finetuning" ([[olmo-3-long-context]], §3.6, secondary report). In Qwen3 the long-context stage follows the STEM- and code-heavy reasoning stage ([[qwen-3-long-context]]). Post-training then decides whether the long-context ability survives SFT, preference optimization, and RL (§7).

This chapter answers three questions for this stage. (1) Which method and mixture choices add usable context without removing short-context ability? (2) Which choices cause narrowing: loss of MMLU, GSM8K, or code scores, or a model that has low perplexity at long lengths but cannot retrieve? (3) How is the trade-off measured? The chapter on continual pretraining ([[ch-32a]]) covers forgetting in general; the next chapter ([[ch-32c]]) covers how to measure effective context length. RoPE's derivation is not repeated here; it belongs to the llm-arch course (ch-06 and ch-16).

## §1 Why models fail beyond, and inside, their trained length

### 1.1 Notation and out-of-range positions

**Definition.** Rotary position embedding (RoPE) rotates each pair of query and key dimensions by an angle proportional to the token position. Dimension pair i of a head of size d has frequency θ_i = b^(−2i/d), where b is the RoPE base (10,000 in LLaMA and Llama 2). Its wavelength λ_i = 2π/θ_i is the number of tokens needed for one full rotation ([[yarn]] §2.1, Eq. 8).

**Problem.** A model trained on sequences of length L has never seen query-key distances above L. Evaluated beyond L without any change, LLaMA 7B's PG19 perplexity exceeds 10³ at 4,096–32,768 tokens ([[position-interpolation]] Table 1). Llama 2 7B without changes has perplexity 7.87 at 4,096 and above 10² at 8,192 ([[dual-chunk-attention]] Table 1).

**Mechanism.** The attention score is a sum of cosine terms in the distance. Chen et al. fit this score function inside [0, L] and show that it stays bounded there but can exceed 8,000 beyond L, while interpolating between trained integer positions has an error bound at least about 600× smaller than the extrapolation bound ([[position-interpolation]] §2.2–2.3, Eqs. 5–8). Continued training at a longer length does not fix this by itself: with unmodified RoPE, Llama 2 7B "could not attend beyond 4,000–6,000 tokens" on a retrieval probe even after long continual pretraining ([[llama-2-long]] §4.1).

### 1.2 Under-trained low-frequency dimensions

**Problem.** Dimensions with wavelength longer than L never complete a full rotation during pretraining, so the model sees only part of their range.

**Worked example.** Llama 2: d = 128, b = 10,000, L = 4,096. The longest wavelength (i = 63) is 2π·10,000^(126/128) ≈ 54,410 tokens, so a 4,096-token window covers about 7.5% of one period. Dimensions i ≥ 46 have λ_i > 4,096; that is 18 of 64 dimension pairs (derived; the figure linked in §2 computes this for any setting).

**Evidence.** LongRoPE2 computes the "theoretical critical dimension" d_tcd = 2⌈(d/2)·log_b(L_train/2π)⌉ (Eq. 7). For Phi3-mini (d = 96, b = 10,000, L_train = 2,048), d_tcd = 62, the 31st cosine dimension; the 48th dimension has a period of 51,861 tokens, and 2,048 tokens cover less than 4% of it. The paper's hypothesis is that these higher dimensions are also under-trained below d_tcd, because long-range dependencies are rare; its search finds a "real" critical dimension of 25 for Phi3-mini and 30 for LLaMA3-8B instead of 31 and 35 ([[longrope2]] §2.2, §3.1–3.2; Interpretation supported by Table 5, where applying the searched d_rcd to YaRN raises LLaMA3-8B RULER-128k from 49.39 to 71.46).

### 1.3 Left-skewed relative-position frequency

**Definition.** Position frequency f(i) is the number of query-key pairs at distance i that a corpus provides during training.

**Formula.** f(i) = Σ_{s∈C} max(|s| − i, 0), for 0 ≤ i < L ([[string-effective-context]] Eq. 2), where C is the set of training sequences, |s| a sequence's length, and i the relative distance.

**Worked example.** One sequence of length 4 gives f(0) = 4, f(1) = 3, f(2) = 2, f(3) = 1. Even with every sequence full, the largest distance L − 1 appears once per sequence while distance 0 appears L times.

**Evidence.** On SlimPajama at L = 2,048, distances i ≤ 1,024 account for more than 80% of position indices and i ≥ 1,536 for less than 5%. Two TinyLlama-1.3B models pretrained for 1T tokens at 2K and 4K reached the same effective length (1,280) when f(1280) reached the same count, 100B, rather than after the same number of training tokens ([[string-effective-context]] §2.2, §3 Finding 2; Result, single study). The consequence measured on released models is that effective length is below training length: shifting well-trained positions into the rarely trained range at inference (STRING) raises Llama 3.1 70B on RULER at 128K from 66.6 to 81.7 without training (Table 2). [[ch-32c]] covers this measurement gap in detail.

### 1.4 Why long training data is scarce and long training is expensive

**Data.** In C4 and RefinedWeb, over 95% of documents have fewer than 2K tokens, although about 45% of RefinedWeb tokens come from documents longer than 2K; over 75% of Gutenberg books exceed 16K tokens ([[harm-de-vries-long-context-run]] §2; practitioner evidence, 10K sampled documents per source). Packing random short documents into a long window gives many token pairs with no dependency.

**Compute formula.** Per-token training FLOPs are FFN 48·N_l·d², QKVO 24·N_l·d², attention 6·N_l·d·(L+1), where N_l is the layer count, d the hidden size, and L the context length ([[harm-de-vries-long-context-run]] §1).

**Worked example.** The attention overhead relative to FFN + QKVO is 6d(L+1)/(72d²) = (L+1)/(12d). For d = 4,096: L = 4,096 gives 8.3%, L = 16,384 gives 33.3%, L = 128,000 gives 260%. For d = 8,192 the same overhead is reached at twice the length. This is why recipes pretrain short and extend late, and why [[llama-3]] §3.4.2 states that it does "not train on long sequences earlier because the compute in self-attention layers grows quadratically in the sequence length."

**Implication.** The result of an extension run depends on the training data as well as on the position encoding. The mixture must supply long-range dependencies (§5) without replacing the short, high-quality data that carries most general knowledge.

## §2 Extension methods and what each changes

The interactive figure [figures/rope-scaling-explorer.html](figures/rope-scaling-explorer.html) plots, for a chosen head dimension, base, original length, and scale factor, the per-dimension frequency multiplier and wavelength under PI, NTK-aware scaling, and YaRN; use it with the worked examples below.

### 2.1 Position Interpolation (PI)

**Definition.** PI divides every position index by the scale factor s = L′/L before applying RoPE, so the largest distance seen at the new length L′ equals the old length L.

**Formula.** f′(x, m) = f(x, m·L/L′) ([[position-interpolation]] Eq. 4), where x is the token representation, m the position, L the pretrained window, L′ the target window.

**Worked example.** L = 2,048, L′ = 8,192, s = 4: position 6,000 maps to 1,500. Every dimension's frequency is multiplied by 1/4, including the high-frequency dimensions that encode the distance between adjacent tokens: neighbors 1 token apart are now 0.25 positions apart.

**Evidence.** LLaMA 7B–65B reach windows up to 32,768 within 1,000 fine-tuning steps on the Pile; PI reaches the target passkey window after 200 steps for 7B and 33B, while fine-tuning without PI moves the 7B effective window only from 2,048 to 2,560 after 10,000 steps (Table 4).

**Short-context cost.** Within the original 2,048 window, LLaMA 7B BoolQ drops from 76.1 to 73.2 / 69.8 / 64.7 for L′ = 8,192 / 16,384 / 32,768, and WinoGrande from 69.6 to 69.0 / 67.8 / 66.9 ([[position-interpolation]] Table 5; Result, single study). The loss grows with s. The authors attribute it to positions being compressed into a narrower range (§3.2, Interpretation).

### 2.2 NTK-aware scaling (community origin)

**Status.** NTK-aware scaling was first posted publicly as a Reddit post; this course has not read that post, and its content is known through the YaRN paper and a write-up by the YaRN authors ([[yarn]] §1; [[eleuther-extending-the-rope]]). Claims about the original post are **anecdotal**; the formula and experiments below come from the paper.

**Definition.** Instead of dividing positions, raise the base so that low frequencies are interpolated and high frequencies are almost unchanged.

**Formula.** b′ = b·s^(d/(d−2)) ([[yarn]] App. A.2, Eqs. 19, 22), where d is the head dimension. The frequency multiplier for dimension i is then s^(−2i/(d−2)).

**Worked example.** d = 128, b = 10,000, s = 4: b′ = 10,000 · 4^(128/126) ≈ 40,890. Dimension i = 0 keeps multiplier 1; dimension i = 63 gets 4^(−126/126) = 1/4, the same as PI. Intermediate dimensions are interpolated geometrically between these two ends.

**Evidence and limits.** Without fine-tuning, NTK-aware scaling gives lower long-sequence perplexity than PI, but after fine-tuning it is worse than PI ([[eleuther-extending-the-rope]]; [[yarn]] App. A.2). On LLaMA 7B at s = 16 after 400 steps, 32k proof-pile perplexity is PI 3.57 versus NTK-aware 8.49 ([[yarn]] Table 5). The YaRN paper explains this by some dimensions extrapolating out of range, so s must be set above the target scale (App. A.2, Interpretation). In the controlled study, the length-dependent (dynamic) NTK variants had the highest RULER scores at 32k and 64k among fine-tuned methods (§2.8).

### 2.3 NTK-by-parts and YaRN

**Definition.** NTK-by-parts decides per dimension whether to interpolate, based on how many rotations the dimension completes within the original length. YaRN is NTK-by-parts plus an attention temperature ([[yarn]] §3.2–3.3).

**Formulas.**
- r(i) = L/λ_i, the number of rotations of dimension i over L (Eq. 10).
- γ(r) = 0 if r < α; 1 if r > β; (r − α)/(β − α) otherwise (Eq. 11).
- θ′_i = (1 − γ(r(i)))·θ_i/s + γ(r(i))·θ_i (Eqs. 12–13).
- Attention: softmax(q_mᵀk_n / (t·√d)), with √(1/t) = 0.1·ln(s) + 1 (Eqs. 14–15).

Symbols: L is the original length, λ_i the wavelength, α and β the ramp limits in rotations (α = 1, β = 32 recommended for the Llama family, "found experimentally"), s the scale factor, t the temperature, q_m and k_n query and key at positions m and n.

**Worked example.** Llama 2 at L = 4,096, d = 128, b = 10,000, s = 4. Dimension 0: λ = 6.3, r = 652 > 32, so γ = 1 and the frequency is unchanged. Dimension 21: λ = 129.0, r = 31.75, γ = 0.992. Dimension 30: λ = 471, r = 8.69, γ = (8.69 − 1)/31 = 0.248, multiplier 0.752·0.25 + 0.248 = 0.436. Dimension 46: λ = 4,712, r = 0.87 < 1, so γ = 0 and the multiplier is 1/4. In total 21 pairs are kept, 18 are divided by 4, and 25 are on the ramp. The temperature factor is 0.1·ln 4 + 1 = 1.139 (derived from the formulas; the figure reproduces these counts).

**Code.** Hugging Face Transformers implements the ramp over dimension indices and returns the temperature as a factor on cos and sin ([[hf-transformers-yarn-code]], `modeling_rope_utils.py` v4.46.0 lines 188–237):
```python
    attention_factor = config.rope_scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0
    beta_fast = config.rope_scaling.get("beta_fast") or 32
    beta_slow = config.rope_scaling.get("beta_slow") or 1
    ...
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).float().to(device)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
```
The factor is fixed when the config loads; it does not depend on the input length. This is the "static YaRN" that the Qwen3 model card warns about (§3.3).

**Evidence.** Llama 2 7B and 13B extended with 400 steps at s = 16 plus 200 steps at s = 32 on 64k-token PG19 chunks reach 99.4% passkey accuracy up to 128k, with fine-tuning data below about 0.1% of the pretraining data ([[yarn]] §1, §4.1, Table 9). On LLaMA 7B at s = 16, YaRN has the lowest 32k perplexity of the four methods: 2.77 versus NTK-by-parts 2.81 and PI 3.57 (Table 5).

**Short-context cost.** Llama 2 13B MMLU falls from 55.8 to 51.9 at s = 32, and 7B from 43.8 to 41.7 (Table 3). On LLaMA 7B at s = 16 after 400 steps, MMLU is 35.7 for the base, 25.9 for PI, 27.7 for NTK-aware, 32.7 for NTK-by-parts, and 30.0 for YaRN (Table 2). YaRN is the best method on long-context perplexity in this paper but not on MMLU. The fine-tuning data was PG19 books only; the authors attribute part of the variance to PG19 differing from the pretraining data (§4.4, Interpretation).

### 2.4 Base-frequency increase in continued pretraining (ABF)

**Definition.** Adjusted base frequency (ABF) raises b before continued pretraining at the longer length, without dividing positions.

**Worked example.** Raising b from 10,000 to 500,000 with d = 128 raises the longest wavelength from about 54,410 to about 2,559,000 tokens (2π·b^(126/128); derived), and lowers the frequency of every dimension with index i > 0, so those dimensions rotate by a smaller angle per token.

**Evidence.** Llama 2 Long continues pretraining Llama 2 on 400B tokens at 32,768 (7B/13B) or 16,384 (34B/70B) tokens with b raised from 10,000 to 500,000 ([[llama-2-long]] §2.1). In 7B ablations with 80B tokens at 32,768, ABF is the only variant that keeps FIRST-SENTENCE-RETRIEVAL performance to 32,768 tokens, and ABF scores slightly above PI on all five short tasks (MMLU 46.24 vs 45.84; HumanEval 17.07 vs 15.24; Table 6, Fig. 5b). With 400B extra tokens, short-task scores are equal or higher than Llama 2 at every size (70B MMLU 68.9 → 71.7; Table 1). The authors attribute this to extra compute and new data (§3.1), so it does not show that ABF itself is free.

**Current official use.** Llama 3 sets θ = 500,000 in its architecture from the start of pretraining, citing Xiong et al. for its effectiveness up to 32,768 tokens, and the report describes no further base change during its six extension stages ([[llama-3]] §3.2, Table 3; §3.4.2). Qwen2.5-1M sets the base per stage: 10,000 → 1,000,000 at 32,768, then 1M / 5M / 10M at 65,536 / 131,072 / 262,144 ([[qwen-long-context-synth]] §3). ProLong uses 8×10⁶ at 64K and 1.28×10⁸ at 512K; at 64K, base 5×10⁵ scored 29.1, 4×10⁶ scored 48.7, and 8×10⁶ scored 54.6 on the long-context average ([[prolong]] App. B.1, Table 18).

### 2.5 A lower bound on the RoPE base

**Definition.** Men et al. define B_{m,θ} = Σ_{i=0}^{d/2−1} cos(m·θ_i), which under an i.i.d. assumption equals the expected attention advantage of a similar key over a random key at distance m ([[rope-base-bounds-context-length]] Theorem 1, Eq. 9).

**Formula.** Obtainable context L_θ = sup{L | B_{m,θ} ≥ 0 for all m ≤ L} (Eq. 10). The lower bound base_L is the smallest base for which B stays non-negative up to L (Eq. 11). The authors state the bound is "not very strict" because stacked layers can pass information further (§4.3).

**Worked example.** Take d = 4, so there are two frequencies θ_0 = 1 and θ_1 = b^(−1/2). With b = 100, θ_1 = 0.1: B at m = 2 is cos 2 + cos 0.2 = 0.564, and at m = 3 is cos 3 + cos 0.3 = −0.035, so L_θ = 2. With b = 10,000, θ_1 = 0.01: B at m = 3 is 0.010 and the first negative value occurs at m = 22, so L_θ = 21 (derived by direct evaluation). A larger base keeps B positive for longer.

**Evidence.** The numerical lower bound (Table 2) is 2.7×10⁴ for 4K, 6.4×10⁵ for 32K, 7.8×10⁶ for 128K, and 5.1×10⁸ for 1M; the head dimension used is not stated next to the table. Llama2-7B fine-tuned to 32K with base 500 keeps low perplexity to 128K but "could not retrieve related information for context length as short as 1k" (§3, Fig. 3). A 2B model pretrained for 1T tokens at 4,096 with base 100 retrieved only from about the last 500 tokens (§5.3, Fig. 7).

**Limits.** The bound is a single-layer analysis. Llama 3's base of 5×10⁵ at 128K is below the 128K bound of 7.8×10⁶; the report describes no other position-encoding change for its extension stage and gives no per-length retrieval analysis, so it neither confirms nor refutes the bound. ProLong's base ablation (§2.4) is consistent with the direction of the bound (larger base, higher long-context score) but was not designed to test it.

**Implication.** Perplexity at the target length is not a sufficient check for an extension run. A retrieval or task evaluation at several lengths is required ([[ch-32c]]).

### 2.6 LongRoPE2: searched per-dimension factors plus mixed-window training

**Mechanism.**
1. Compute d_tcd and search a lower critical dimension d_rcd between the dimension with 10 theoretical periods in L_train and d_tcd.
2. For dimensions i ≥ d_rcd, search scale factors in [L/L_train, 2·L/L_train] with an evolutionary search (population 64, 40 iterations), constrained to be non-decreasing; dimensions below d_rcd use NTK scaling.
3. Score candidates by "needle-driven" perplexity: perplexity only on the answer tokens of a needle placed at the start of a long document and asked for at the end (10 PG19 validation books).
4. Mid-train on 10B tokens with two kinds of 128k segments: short documents packed with the original RoPE and a cross-document attention mask, and long documents with the rescaled RoPE. At inference, switch RoPE by input length ([[longrope2]] §3.2–3.3, App. B).

**Evidence.** LLaMA3-8B extended to 128k scores 82.03 on RULER at 128k versus 73.40 (LongRoPE), 73.19 (NTK), and 49.39 (YaRN) under the same mid-training (Table 2). Short-benchmark averages (4,096-token context) for LLaMA3-8B: original 56.5, YaRN 52.1, NTK 54.0, LongRoPE 54.6, LongRoPE2 55.7; for Phi3-mini: 63.2, 53.6, 57.3, 58.5, 61.7 (Table 4). Retention is 98.6% and 97.6% (§4.2). Removing the mixed-window training lowers Phi3-mini MMLU from 70.07 to 66.56 (Table 7). Replacing needle perplexity with plain PG19 perplexity lowers LLaMA3-8B RULER-128k from 82.03 to 78.68 (Table 6).

**Limits.** Two base models, one target length, no post-training, no repeated runs (§4.1–4.2). Switching RoPE at inference requires recomputing the KV cache once (App. B).

### 2.7 Inference-only extension: Dual Chunk Attention (DCA)

**Definition.** DCA changes no weights. It remaps position indices at inference so that no query-key distance exceeds the pretraining length c ([[dual-chunk-attention]] §3).

**Mechanism.**
1. Split the sequence into chunks of size s (for Llama 2, s = 3,072, three quarters of the 4,096 training length).
2. Within a chunk, use positions 0 … s−1 for queries and keys (intra-chunk).
3. For keys two or more chunks back, give every query the position c − 1 (inter-chunk), so distances are at least c − s and at most c − 1.
4. For the immediately preceding chunk, set the first w query positions to s, s+1, …, s+w−1 (successive-chunk), with w = c − s, so the nearest w keys keep small distances.

**Worked example.** From the paper's Fig. 2: c = 10, s = 6, w = 4. The successive-chunk query positions are [6, 7, 8, 9, 9, 9, 6, 7, 8, 9, 9, 9]. The last key of chunk 1 (position 5) and the first query of chunk 2 (position 6) have distance 1, as in the original sequence.

**Evidence.** Llama2 70B with DCA has PG19 perplexity 5.24 at 4,096 and 5.59 at 65,536 without training (Table 1), and the chat model at 16K reaches 94% of gpt-3.5-turbo-16k on the paper's long-context research benchmarks (§4.2). Qwen2.5-1M combines DCA with YaRN attention scaling to run 262K-trained models at 1M tokens; the report states neither method changes behavior within the training length ([[qwen-long-context-synth]] §5.1).

**Limits.** The DCA paper reports no short-benchmark scores with DCA enabled. The Qwen2.5-72B-Instruct model trained to 32K with DCA + YaRN outperformed the 1M-trained 14B model on LV-Eval at all lengths ([[qwen-long-context-synth]] §6.1), so an inference-only method on a larger model can compete with a trained smaller model on some benchmarks.

### 2.8 A controlled comparison of methods

[[controlled-long-context-extension]] extends one LLaMA2-7B base with 1B tokens of length-upsampled SlimPajama at 32k for every fine-tuned method except NTK-64K, which is trained at 64k. RULER averages at 32k / 64k: Dynamic NTK (NTK-32K) 59.42 / 46.26; PI 57.66 / 0.00; YaRN 36.95 / 0.00; LongLoRA (approximate attention, fine-tuned) 3.53 / 0.00 (Table 3). Exact-attention methods work within their training length; only NTK and CLEX generalize beyond it (§5.2). Short-context signals: at 2k, PG19 perplexity rises for every fine-tuned exact method (base 6.61; NTK-32K 6.63, YaRN 6.70, PI 6.88; Table 2), and LongBench code completion LCC falls from 68.22 to 56.78 (NTK-32K), 55.05 (PI), and 54.06 (YaRN) (Table 11). RULER at 4k does not show a drop (base 80.94; NTK-32K 86.58, PI 84.56; Table 3), so the short-context signal depends on the metric. The study uses one base model and one training length; its YaRN result differs from the YaRN paper's own comparison, which used different data and scale factors (Interpretation: recipe details change method rankings).

### Summary of methods

| Method | What changes | Training needed | Measured short-context effect (setting) |
|---|---|---|---|
| PI | all frequencies × 1/s | ~200–1,000 steps | 7B BoolQ 76.1 → 64.7 at s = 16 ([[position-interpolation]] T5) |
| NTK-aware | base b′ = b·s^(d/(d−2)) | none, or fine-tune (worse than PI after fine-tuning) | 7B MMLU 35.7 → 27.7 at s = 16 ([[yarn]] T2) |
| YaRN | ramp per dimension + temperature | 400–600 steps in the paper | 13B MMLU 55.8 → 51.9 at s = 32 ([[yarn]] T3) |
| ABF | base increase, continued pretraining | 80B–400B tokens in Llama 2 Long | short tasks equal or higher after 400B tokens ([[llama-2-long]] T1) |
| LongRoPE2 | searched factors + mixed windows | 10B tokens | 98.6% retention, LLaMA3-8B ([[longrope2]] T4) |
| DCA | position remapping at inference | none | not reported ([[dual-chunk-attention]]) |
| STRING | position shift at inference, within L | none | not reported ([[string-effective-context]]) |

## §3 Hybrid designs in official reports

### 3.1 Olmo 3: YaRN only on full-attention layers

Olmo 3 uses sliding-window attention with a 4,096-token window on three of every four layers and full attention on the rest ([[olmo-3-long-context]] §3.2). For extension from 8,192 to 65,536 tokens, the team tried base-frequency scaling, PI, and YaRN, each on all RoPE layers or only on full-attention layers, and found "applying YaRN only to full attention layers yields the best overall performance" (§3.6.4, Fig. 13a; exact values not printed). The released script at OLMo-core commit 66f768b (`src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py`):
```python
DEFAULT_SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 65536 * 64  # ~4M tokens
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
    )
        generate_doc_lengths=True,  # enables intra-document masking
```
`with_rope_scaling(..., full_attn_layers_only: bool = True)` leaves the sliding-window layers at their pretrained RoPE. A sliding-window layer never sees distances above 4,096, so it has no out-of-range positions to fix (Interpretation).

### 3.2 Gemma 3: local:global layers with interpolation on global layers

Gemma 3 uses 5 local layers (1,024-token window, RoPE base 10K) per global layer (base 1M), pretrains at 32K, and extends the 4B, 12B, and 27B models to 128K "at the end of pre-training while rescaling RoPE" with "a scaling factor of 8" ([[gemma-3-long-context]] §2, §5.3). The design goal stated is KV-cache memory: at a 32K pre-fill on a 2B model, global-only attention has 60% memory overhead versus less than 15% for 1:3 with a 1,024 window (Fig. 5), with minimal perplexity impact from the ratio and window (Figs. 3–4). The models "generalize to 128K, but rapidly degrade as we continue to scale" (Fig. 7). RULER at 128K is 72.9 for the 27B pretrained model and 66.0 after instruction tuning (Table 15). The report does not isolate the short-context effect of the 128K extension.

### 3.3 Qwen3: ABF in training, YaRN and DCA at inference, and a model-card caution

Qwen3's final pretraining stage trains "hundreds of billions of tokens" at 32,768 tokens, with 75% of the long corpus between 16,384 and 32,768 tokens and 25% between 4,096 and 16,384, raises the base from 10,000 to 1,000,000 by ABF, and uses YaRN and DCA for "a four-fold increase in sequence length capacity during inference" ([[qwen-3-long-context]], report §3.2). The Qwen3-8B model card adds:

> All the notable open-source frameworks implement static YaRN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts.** We advise adding the `rope_scaling` configuration only when processing long contexts is required.

and "If the average context length does not exceed 32,768 tokens, we do not recommend enabling YaRN". The card gives no number for the short-text loss. The mechanism is visible in §2.3's worked example: a static factor applies the ramp to every input, including inputs shorter than the original length, while dynamic scaling uses s = max(1, l′/L) and leaves short inputs unchanged ([[yarn]] §3.4; [[eleuther-extending-the-rope]] notes "a flat reduction of performance at lengths less than L" for fixed factors).

### 3.4 Implication for a general-purpose model

These reports share one design principle: apply position changes only where they are needed. Olmo 3 restricts YaRN to full-attention layers, Gemma 3 changes only global layers, LongRoPE2 keeps the original RoPE for short inputs, and Qwen recommends enabling YaRN only for long inputs. Each choice keeps the short-context computation closer to the pretrained model. This is a shared design choice across reports, not a controlled comparison of the principle itself (Interpretation).

## §4 Measured short-context effects of extension

| Setting | Short-context measurement | Before → after | Source |
|---|---|---|---|
| LLaMA-2 7B / 13B, 5B tokens at 80K / 64K, per-source upsampling | MMLU | 43.3 (7B) and 52.4 (13B) after; base not in this table | [[fu-2024-data-engineering-128k]] Table 3 |
| Llama 2 7B / 13B base, as reported by Xiong et al. | MMLU | 45.3 (7B), 54.8 (13B) | [[llama-2-long]] Table 1 |
| Llama-3-8B + training-free base change (PE) | MMLU / GSM8K | 66.5 / 44.7 → 64.7 / 40.1 | [[prolong]] Table 2 |
| Llama-3-8B + Fu et al. SlimPajama long mix | MMLU / GSM8K | 66.5 / 44.7 → 63.1 / 40.6 | [[prolong]] Table 2 |
| ProLong-8B after SFT vs Llama-3-8B-Instruct | 5-task avg; MMLU; GSM8K | 69.8 → 69.4; 67.0 → 64.6; 68.5 → 58.9 | [[prolong]] App. B.7, Table 25 |
| Llama-3-8B, 32K ABF, ~1B tokens long-only CPT | 17-benchmark average | 55.16 → 51.00 | [[longred]] Table 3 |
| Mistral-7B-v0.3, 128K ABF, long-only CPT | 17-benchmark average; MMLU | 51.09 → 40.68; 62.30 → 51.90 | [[longred]] Tables 3, 14 |
| Qwen2.5-14B-Instruct 128K → 1M | MMLU-Pro; GPQA; LiveCodeBench; IFEval | 63.7 → 63.3; 45.5 → 39.9; 42.6 → 38.6; 81.0 → 84.3 | [[qwen-long-context-synth]] Table 6 |
| Olmo 3 7B Stage 2 → Stage 3 (50B tokens) | Math / Code / GenQA aggregates | 59.8 → 54.4; 31.9 → 30.6; 71.3 → 72.5 | [[olmo-3-long-context]] Table 13 |
| Olmo 3 32B Stage 2 soup → Stage 3 (100B) | Math / Code / GenQA | 69.7 → 61.4; 39.7 → 39.7; 79.4 → 79.7 | [[olmo-3-long-context]] Table 13 |
| Llama 2 70B → Llama 2 Long 70B (400B tokens) | MMLU | 68.9 → 71.7 | [[llama-2-long]] Table 1 |

**Worked example (cross-paper, derived).** Using Xiong et al.'s Llama 2 baselines with Fu et al.'s post-extension scores gives 45.3 → 43.3 (−2.0) for 7B and 54.8 → 52.4 (−2.4) for 13B. The two papers may use different MMLU harnesses, so these deltas are indicative only. Fu et al. describe the result as maintaining short-context performance, "evidenced by strong MMLU" (§5.1).

**Reading the table.**
1. Short runs with long-only or long-heavy data lose the most: 4 points of average at 8B (LongReD) and 10 points at 7B Mistral (LongReD), with math and multi-step tasks (GSM8K, GPQA, Olmo 3 Math) dropping more than knowledge or instruction-following scores. **Replicated** across [[prolong]], [[longred]], [[qwen-long-context-synth]], [[olmo-3-long-context]] for the direction of the effect; magnitudes differ by setting.
2. Long runs with broad data can gain: Llama 2 Long's 400B tokens raised MMLU at every size. LongReD's authors note that studies training on more than 100B tokens report some short abilities unaffected or improved ([[longred]] Limitations). The budget and data breadth of the extension stage are confounded with the extension itself in these results (Interpretation).
3. Aggregates hide task-level loss. The Qwen2.5-1M report calls short scores "similar" while 14B GPQA falls 5.6 points ([[qwen-long-context-synth]] §6.2, Table 6).

### 4.1 Mechanisms: drift and forgetting

**Definition.** LongReD separates short-text degradation into (1) distribution drift, the change in hidden states and attention distributions caused by the RoPE change, and (2) catastrophic forgetting during continued pretraining ([[longred]] abstract, §3).

**Evidence.** Drift is measured as the cosine similarity between original and extended hidden states on 1,000 SlimPajama samples of 8,192 tokens. For Llama-3-8B at 128K (base 1×10⁸), similarity is 0.83 right after the RoPE change and 0.94 after 1B tokens of continued training (Table 1). Across 15 extended models, higher similarity goes with higher MMLU retention: ABF 5×10⁶ at 32K keeps 0.960 of MMLU at similarity 0.943, while PI×16 keeps 0.866 at 0.862 (App. D Table 12). Forgetting: with base 2×10⁷ at 32K, short-text scores recover within the first steps (usually fewer than 32) and then decline as training continues; replacing half the long data with 8K short text raises MMLU from 62.0 to 62.5 and HumanEval from 14.02 to 16.46 (§3.2, Fig. 3, Table 2).

**Implication.** Two different controls are needed. Drift is reduced by choosing a position change that alters short-range computation less (§3.4) or by distillation toward the original model (§6.2). Forgetting is reduced by keeping short data in the mix (§5.1).

## §5 Long-context data mixtures and schedules that keep general ability

### 5.1 The long:short ratio

**ProLong.** Llama-3-8B base, 5B tokens at 64K, books/repositories long data plus the ShortMix short data, evaluated after UltraChat SFT. "More long data initially improves long-context performance, but then becomes impairing. More long data also consistently degrades the short-context performance" ([[prolong]] Fig. 3). The best long-context average was at 60% long; the final recipe is 30% code repositories, 30% books, 3% textbooks, 37% ShortMix (Table 9). With 100% long data, PG19 perplexity kept improving while downstream long-context scores fell (§2.1, Fig. 1).

**Olmo 3.** In a 10B-token extension, 66% long / 34% short dropped a subset of OlmoBaseEval by 2.5 points, and 34% long / 66% short by 0.8 points; the final recipe uses 34% long ([[olmo-3-long-context]] §3.6.3).

**Worked example.** For a 20B-token stage at ProLong's ratio, 12B tokens are long data and 8B are short. At Olmo 3's ratio, 6.8B are long and 13.2B are short. The two recipes differ by a factor of about 1.8 in long tokens for the same budget; neither report tests the other's ratio on the same model.

**Qwen length mixes are a different quantity.** Qwen2.5-1M uses 75% of sequences at the current maximum length and 25% shorter ([[qwen-long-context-synth]] §3); Qwen3's long corpus is 75% 16K–32K and 25% 4K–16K ([[qwen-3-long-context]]). These are length distributions within the long-context corpus, not a share of general short-context data, so they cannot be compared directly with the ProLong and Olmo 3 ratios.

**Status.** The direction (more short data, less short-context loss) is **Replicated** ([[prolong]], [[olmo-3-long-context]], [[longred]] Table 2). The best ratio is an **Open question**: it depends on the evaluation (base-model short tasks for Olmo 3, post-SFT long tasks for ProLong) and on the short data's quality.

### 5.2 Which long data and which short data

- **Source balance.** Fu et al. keep SlimPajama's domain ratios (67% CommonCrawl, 15% C4, 4.5% GitHub, 4.5% Wikipedia, 4.5% books, 2.5% ArXiv, 2.0% StackExchange) and raise the share of documents longer than 4K within each domain from about 30% to about 70%. Upsampling books alone lowers Book loss by 0.175 at 0–4K but raises StackExchange and GitHub loss by 0.021 and 0.029; per-source upsampling has no significant loss increase on any domain (|Δ| ≤ 0.01 threshold; [[fu-2024-data-engineering-128k]] §5.3, Table 5).
- **Source type.** With 40% ShortMix fixed, the long-context average is 54.6 for books + repositories 1:1, 53.8 books, 52.3 code repositories, 51.4 ArXiv, 50.9 CommonCrawl ([[prolong]] Table 4). Olmo 3 uses olmOCR science PDFs, filtered by removing the 20% most and 20% least gzip-compressible text, plus synthetic aggregation tasks inserted into documents ([[olmo-3-long-context]] §3.6.1–3.6.2).
- **Short data quality.** With the same long data, ShortMix gives long 54.6 / short 65.5, versus SlimPajama 52.9 / 64.2, FineWeb-Edu 53.0 / 63.0, DCLM-Baseline 52.0 / 64.8; the original Llama-3-8B short average is 66.0 ([[prolong]] Table 6). ProLong's own reproduction of Fu et al.'s mix under the same budget: long 51.8 vs 54.6, short after SFT 65.4 vs 67.5 (App. B.6, Table 24).
- **Quality over length share.** In Llama 2 Long's 7B ablations, continued pretraining on the Llama 2 mix with long texts removed gave NarrativeQA / Qasper / QuALITY gains of 19.48% / 39.14% / 67.1% over Llama 2, versus 23.70% / 43.64% / 75.5% for the Llama 2 Long mix, and upsampling long texts in the Llama 2 mix did not close the gap (22.15% / 36.82% / 65.0%). The authors conclude the gain of their mix comes mostly from data quality rather than length distribution ([[llama-2-long]] §4.2, Tables 7–8; Interpretation).

### 5.3 Staged schedules gated on short-context recovery

Llama 3 405B extends from 8K to 128K in six stages over about 800B tokens. The report states the gate: "We assess successful adaptation by measuring whether (1) model performance on short-context evaluations has recovered completely and (2) the model perfectly solves 'needle in a haystack' tasks up to that length" ([[llama-3]] §3.4.2). Per-stage lengths and token counts are not printed ([[llama-3-recipe]]). After extension, a final annealing phase over 40M tokens takes the LR to zero at 128K context with upsampled high-quality data and checkpoint averaging (§3.4.3).

Qwen2.5-1M uses five stages (4,096 → 32,768 → 65,536 → 131,072 → 262,144) with the base set per stage (1M at 32,768 and 65,536, then 5M and 10M); RULER at 128K for the 14B model rises 37.6 → 56.0 → 83.8 → 87.6 over stages 2–5, so the 262K stage still improved 128K accuracy ([[qwen-long-context-synth]] Table 2). ProLong's 512K stage, evaluated at 64K, beats an equal-token 64K continuation on recall (98.5 vs 95.0) and re-ranking (32.9 vs 28.0) ([[prolong]] Table 7).

**Implication.** Training beyond the target evaluation length helped in both reports. A stage gate that includes short-context evaluations makes regression a stopping criterion rather than a post-hoc finding. The Llama 3 report does not publish how many tokens each gate required.

### 5.4 Document masking and packing

- Llama 3: "We use an attention mask that prevents self-attention between different documents within the same sequence. We find that this change had limited impact during in standard pre-training, but find it to be important in continued pre-training on very long sequences" ([[llama-3]] §3.2; quoted from arXiv:2407.21783v3).
- ProLong: masking cross-document attention gives long 54.6 / short 65.5 versus 53.6 / 64.9 without masks ([[prolong]] App. B.2, Table 20).
- Olmo 3: intra-document masking and best-fit packing, which reduces split documents; document packing "boosts performance for longer context lengths" (Fig. 13d; [[olmo-3-long-context]] §3.6.4).
- Fu et al. pack to 80K "regardless of the document boundary" without masking and still report 88.0 NIAH for 7B ([[fu-2024-data-engineering-128k]] §4, Table 3), so masking is not required for retrieval, but ProLong's ablation shows it helps both long and short scores.

### 5.5 Context-window scheduling during pretraining (SkyLadder)

SkyLadder changes the attention window, not the data: w(t) = min(w_e, w_s + ⌊αt⌋), where t is the training step, w_s the start window, w_e the target, and α tokens per step ([[skyladder]] §4.1). With w_s = 32 and α = 1/8, step 8,000 gives w = 1,032 tokens; 8,192 is reached after about 64K steps (§4.2; worked example derived). At a fixed 100B-token budget, 1B models at 8K on CommonCrawl with random packing improve the nine-task standard average from 46.3 to 50.0 (Table 1). At 32K, the linear schedule scores long / standard 13.5 / 54.3 versus 9.7 / 50.7 for a constant 32K window and 10.0 / 52.9 for 4K pretraining followed by 3B tokens of 32K continued pretraining (Table 7). The evidence covers models up to 3B and windows up to 32K, one run per setting (App. A.1). It applies when training from scratch, not when extending a released checkpoint.

### 5.6 Token budget

Fu et al. report NIAH accuracy on 7B of 81.1 at 500M tokens, 88.0 at 5B, and 84.0 at 10B, with length generalization beyond 80K decreasing at 10B ([[fu-2024-data-engineering-128k]] Fig. 3). Olmo 3's RULER rises with extension budgets from 1B to 100B, most at longer lengths (Fig. 13e). ProLong's 40B tokens give a HELMET 128K average of 49.4 versus 46.5 for Llama-3.1-8B-Instruct with about 800B long-context tokens ([[prolong]] Table 10). Budgets are not comparable across reports because base models, target lengths, and evaluations differ.

## §6 Mitigations for short-context regression

### 6.1 High-quality short-data replay

Replay is the mixture control of §5.1. The size of the effect in a controlled setting: LongReD's replacement of half the long data with 8K short text raised PIQA from 74.10 to 78.24 and TriviaQA from 70.67 to 72.82 (Table 2). ProLong's ShortMix versus SlimPajama difference (short average 65.5 vs 64.2) shows that the replay source matters as well as the share ([[prolong]] Table 6).

### 6.2 Restoration distillation (LongReD)

**Formula.** L_final = L_long + α₁·L_short + α₂·L_s2l ([[longred]] Eq. 18), where:
- L_long is the language-model loss on long texts;
- L_short = −Σ Sim(H_l(extended), H_l(original)) summed over selected layers on short texts (1,024 tokens) with normal positions (Eq. 8); layers are those with the largest attention KL between the two models, plus the last layer;
- L_s2l aligns the extended model's last-layer hidden states on a short text with skipped position indices to the original model's hidden states on the same text with normal positions (Eq. 17);
- Sim is cosine similarity; α₁ and α₂ are weights (5 and 10 at 32K; 2 and 15 at 128K; App. B.3).

**Worked example.** Llama-3-8B at 32K ABF: long-only CPT keeps 51.00 / 55.16 = 92.5% of the original short-text average; LongReD-C keeps 54.85 / 55.16 = 99.4%, with RULER 84.98 versus 82.80 (Table 3). Mixed-length CPT scores 45.86 on the short average in this table, lowered by near-zero SQuADv2 and DROP scores that the paper does not discuss (Table 16).

**Limits.** At 128K on Llama-3-8B, LongReD's RULER is 64.93 (CREAM sampling) or 68.41 (uniform) versus 69.70 for long-only CPT, so it trades some long-context score for short-context retention. On Mistral-7B-v0.3 at 128K it improves both but still leaves a 3.4–4.8 point short-average gap (Table 3). All runs use about 1B tokens.

### 6.3 Short-to-long preference optimization (LongPO)

LongPO applies to an already aligned instruct model rather than to continued pretraining. It builds preference pairs from the model itself: the chosen response is the model's answer given only the relevant chunk, and the rejected response is its answer given the full long document. The KL reference is the model conditioned on the chunk, and an NLL term on the chosen long sequence is added ([[longpo]] §3, Eqs. 12–14). On Mistral-7B-Instruct-v0.2 with 45K self-generated samples, InfiniteBench rises from 13.82 to 39.27 while MMLU moves from 59.15 to 59.99; SFT and DPO on the same data reach InfiniteBench 30.03 and 25.56 (Tables 1, 3). The authors report that SFT and DPO degrade short-context tasks by 10–20 points on most tasks, while LongPO does not (§1, Fig. 3). Evidence covers 7B instruct models only. Details of the negative term are in the next section.

## §7 Which post-training stage owns long context

The reports disagree, and the disagreement is not resolved by any controlled study in the library.

- **Llama 3: long SFT data is needed.** "Naively applying our existing SFT recipe with only short-context data resulted in significant regressions in long-context capabilities from pre-training." The fix is synthetic long-context QA, hierarchical summarization, and repository code reasoning, bucketed at 16K–128K; "mixing 0.1% of synthetically generated long-context data with the original short-context data optimizes the performance across both short-context and long-context benchmarks." DPO on short data only "did not negatively impact long-context performance as long as the SFT model is high quality in long context tasks" ([[llama-3]] §4.3.4; no table printed).
- **ProLong: short SFT data is enough.** After 40B tokens of long continued training, SFT on 1B tokens of UltraChat alone scores 55.7 on the long-context average; adding synthetic long instruction data at 1 / 3 / 10 / 50% of tokens gives 54.1 / 53.5 / 53.9 / 43.3 with an 8B generator, and no ratio beats 0% with a 70B generator either ([[prolong]] Table 8, App. B.5 Table 23).
- **Qwen2.5-1M: long SFT, short RL.** SFT stage 2 mixes short and long data up to 262,144 tokens; the following offline RL uses only pairs of at most 8,192 tokens and still raises LongBench-Chat (7B 7.32 → 8.08; 14B 8.56 → 8.76) ([[qwen-long-context-synth]] §4, Table 3).
- **Gemma 3 and Qwen3: post-training lowers some long-context scores.** Gemma 3 27B RULER-128K is 72.9 pretrained and 66.0 instruction-tuned ([[gemma-3-long-context]] Table 15). Qwen3-235B-A22B RULER average is 95.0 in non-thinking mode and 92.2 in thinking mode ([[qwen-3-long-context]] Table 23).

**Open question.** ProLong's authors offer two hypotheses for the conflict with Llama 3: prior work may have used too little long-context continued training, or much larger short instruction sets ([[prolong]] §5). Neither hypothesis has been tested. A plausible reconciliation is that the need for long SFT data grows with the number of SFT optimizer steps on short data, consistent with Llama 3's own explanation for why short-only DPO was safe (Interpretation, untested). Until tested, a general-purpose run should evaluate long-context tasks after every post-training stage, not only after extension.

## §8 Long-context stage summary: placement, gates, and evaluation hand-off

1. **Before extension.** Record the short-context panel (knowledge, math, code, instruction following) on the pre-extension checkpoint; this is the reference for the recovery gate.
2. **Position change.** Choose ABF or YaRN for continued training; check the base against the lower bound for the target length ([[rope-base-bounds-context-length]] Table 2) and verify retrieval, not only perplexity.
3. **Mixture.** Keep 37–66% high-quality short data; use long data from several sources; use document masking.
4. **Schedule.** Extend in stages; move on only when the short panel has recovered and retrieval at the current length is solved; consider training beyond the target length.
5. **After extension.** Evaluate effective length with task suites at several lengths ([[ch-32c]]), then re-check long-context scores after SFT, preference optimization, and RL (§7; [[ch-44a]] covers long-context RL).

## Negative samples and negative feedback

**Which kind of negative.** Continued-pretraining extension (§1–§5) uses no negatives. Two mitigations in this chapter do: LongPO uses **negative as gradient** (type 4 in the course definition): the rejected response's likelihood is pushed down through the DPO-style term. LongReD uses no negatives; its signal is similarity to the original model.

**Source of negatives.** In LongPO the rejected response y_L is the model's own greedy answer given the full long document; the chosen response y_S is its answer given only the relevant chunk. The label comes from construction only. The paper assumes y_L "is likely to be of lower quality" and reports no measured error rate for this assumption ([[longpo]] §3.1, §4.1).

**Mechanism.** The objective is L = −E[log σ(β·log πθ(y_S|x_L)/π_S(y_S|x_S) − β·log πθ(y_L|x_L)/π_S(y_L|x_S))] (Eq. 12), where πθ is the policy, π_S the original short-context model used as reference, x_L the long document plus instruction, x_S the chunk plus the same instruction, β = 0.1, and σ the sigmoid. For any sequence-level term, the token-level gradient of log p_y with respect to logit z_j is ∂ log p_y / ∂z_j = 1[j = y] − p_j. Decreasing log p for a rejected token lowers its logit and raises every other logit in proportion to its probability p_j, so the removed mass goes mostly to the tokens that are already most likely. When the rejected answer is already unlikely under πθ, the push concentrates mass further on the current top alternative, which may not be the chosen answer (course derivation; the full treatment is in the negative-feedback chapter).

**Controls used by LongPO.** (1) The reference π_S(y|x_S) is conditioned on the chunk that contains the needed information, so the implicit KL keeps the long-input policy close to the short-context model's behavior (§3.2, Eq. 10). (2) An NLL term on the chosen response with the long input anchors the positive (Eq. 14, λ = 0.01 on the preference term). (3) Data are generated by the model being trained, so the pairs are on-policy at the start of each iteration (§4.1).

**Evidence.** LongPO exceeds SFT on chosen responses and SFT on rejected responses on RULER-NIAH across training (§5.3, Fig. 4; curves only). DPO on the same pairs, without the short-to-long reference, scores InfiniteBench 25.56 versus LongPO 39.27 and degrades short-context tasks (Table 1, Fig. 3). SFT with the short-to-long constraint also kept short-context performance (§5.3), so the constraint, not the rejected term alone, is the component linked to retention in this paper (Interpretation). The share of the gain due to the rejected term alone is not measured.

**Diagnostics.** Log chosen and rejected log-probabilities separately (the paper logs chosen and rejected rewards, App. B.1, Fig. 6); track the short-context panel during training, not only at the end; check whether rejected log-probability falls while chosen log-probability also falls, which indicates mass moving to neither response.

**Effect on generality.** On Mistral-7B, every short metric stays within 0.74 points of the base or above it (Table 3). On Qwen2.5-7B, ARC-C falls 1.45 and MMLU 0.64 while MT-Bench rises 7.30 → 7.62 (Table 3). No calibration, hallucination, or refusal measurements are reported.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| ProLong-64k-Base | 8B | long-context | tokens; max length; RoPE base | 20B; 64K (2¹⁶); 8×10⁶ | arXiv:2410.02660v4 Table 9 ([[prolong-recipe]]) | verified 2026-09-14 | App. B.1 Table 18: 54.6 vs 48.7 (4×10⁶) vs 29.1 (5×10⁵) |
| ProLong-64k-Base | 8B | long-context | data mixture (token vs sampling share not stated) | 30% code repos, 30% books, 3% textbooks, 37% ShortMix | v4 Table 9 | verified 2026-09-14 | Fig. 3 (60% long best long avg); Table 4; Table 6 |
| ProLong-64k-Base | 8B | long-context | LR; batch; masking | 1e-5, 10% warmup, cosine to 1e-6; 4M tokens; cross-document masking | v4 Table 9, App. B.2 | verified 2026-09-14 | Table 20 (masking); LR: no ablation reported |
| ProLong-512k-Base | 8B | long-context | tokens; length; RoPE base | 20B; 512K; 1.28×10⁸ | v4 Table 9 | verified 2026-09-14 | Table 7 (512K vs 64K at equal tokens) |
| ProLong-512k-Base | 8B | long-context | peak LR | paper 1e-5 per stage; `train_512K.sh` default 5e-6 | v4 Table 9; ProLong repo @499fa29 | conflict | no ablation reported |
| ProLong-512k-Instruct | 8B | SFT | data; tokens; LR | UltraChat only; 1B tokens; 2e-5 | v4 Table 9, §5 | verified 2026-09-14 | Table 8: 0% synthetic long 55.7 vs 54.1 at 1% |
| Llama 3.1 405B | 405B | long-context | stages; tokens; gate | six stages 8K → 128K; ≈800B; short evals fully recovered and NIAH solved | arXiv:2407.21783v3 §3.4.2 ([[llama-3-recipe]]) | verified 2026-09-14 | no ablation reported; per-stage values not printed |
| Llama 3 (all sizes) | 8B–405B | pretrain-stable | RoPE θ | 500,000 | v3 §3.2, Table 3 | verified 2026-09-14 | cites Xiong et al. (2023) for 32,768 tokens |
| Llama 3.1 405B | 405B | pretrain-decay/anneal | tokens; context; LR | final 40M tokens; 128K; linear to 0 | v3 §3.4.3 | verified 2026-09-14 | no ablation reported |
| Llama 3.1 | not scoped | SFT | long-context share | 0.1% synthetic long-context data, buckets 16K/32K/64K/128K | v3 §4.3.4 | verified 2026-09-14 | "careful ablations"; no table |
| Qwen2.5-7B/14B-1M | 7B, 14B | long-context | stage lengths; RoPE bases | 65,536 / 131,072 / 262,144; 1M / 5M / 10M | arXiv:2501.15383v1 §3 ([[qwen-long-context-synth]]) | verified 2026-09-14 | Table 2 (14B RULER-128K 56.0 / 83.8 / 87.6) |
| Qwen2.5-7B/14B-1M | 7B, 14B | long-context | length mix | 75% at current max length, 25% shorter | §3 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B/14B-Instruct-1M | 7B, 14B | preference | pair length | offline RL, pairs ≤ 8,192 tokens | §4 | verified 2026-09-14 | Table 3 (LongBench-Chat gains) |
| Qwen3 (all sizes) | 0.6B–235B | long-context | length; mix; base; tokens | 32,768; 75% 16,384–32,768 / 25% 4,096–16,384; 10,000 → 1,000,000 (ABF); "hundreds of billions" | arXiv:2505.09388v1 §3.2 ([[qwen-3-long-context]]) | verified 2026-09-15 | no ablation reported |
| Olmo 3 7B | 7B | long-context | mixture; tokens | 34% long / 66% Dolmino short; 50B | arXiv:2512.13961v2 §3.6 ([[olmo-3-long-context]]) | verified 2026-09-15 | §3.6.3: −0.8 vs −2.5 points (10B-token test) |
| Olmo 3 32B | 32B | long-context | tokens; batch | 100B; 8×1024×1024 tokens | §3.6; OLMo-core @66f768b `OLMo-3-1025-32B-long-context.py` | verified 2026-09-15 | Fig. 13e (longer extension helps) |
| Olmo 3 7B, 32B | 7B, 32B | long-context | RoPE scaling; length | YaRN factor 8, beta_fast 32, beta_slow 1, old_context_len 8192, full-attention layers only; 65,536 | OLMo-core @66f768b scripts; `transformer/config.py` | verified 2026-09-15 | Fig. 13a (no values printed) |
| Olmo 3 7B | 7B | long-context | batch; LR; schedule | 65,536×64 tokens; 2.0712×10⁻⁴; linear warmup 200 steps, decay to 0 | OLMo-core @66f768b `OLMo-3-1025-7B-long-context.py` | verified 2026-09-15 (decay horizon not reported) | no ablation reported |
| Gemma 3 4B/12B/27B | 4B–27B | long-context | method | 32K → 128K at end of pretraining; RoPE rescale factor 8 on global layers (base 1M); local layers base 10K; tokens not reported | arXiv:2503.19786v1 §2, §5.3 ([[gemma-3-long-context]]) | verified 2026-09-15 | "factor of 8 to work well in practice"; no table |
| Fu et al. LLaMA-2 7B | 7B | long-context | tokens; length; LR; batch | 5B; 80K; constant 2e-5; 4M tokens | arXiv:2402.10171v1 §4 ([[fu-2024-data-engineering-128k]]) | verified 2026-09-15 | Fig. 3 (500M–10B tokens) |
| Fu et al. LLaMA-2 7B | 7B | long-context | mixture | SlimPajama domain ratios fixed; >4K share ~30% → ~70% within each domain | §5.3 | verified 2026-09-15 | Table 5 (per-domain loss) |
| LongRoPE2 LLaMA3-8B-128k | 8B | long-context | tokens; masking | 10B; short windows original RoPE + cross-document mask, long windows rescaled RoPE | arXiv:2502.20082v1 §3.3, §4.1 ([[longrope2]]) | verified 2026-09-14 | Table 7 (without mixed training) |
| LongReD Llama-3-8B 32K | 8B | long-context | α₁; α₂; T_s; tokens | 5; 10; 1,024; about 1B | arXiv:2502.07365v3 App. B.3, §5.1 ([[longred]]) | verified 2026-09-14 | Table 4 (α), Table 6 (T_s) |
| Llama 2 Long | 7B–70B | long-context | tokens; RoPE base; peak LR | 400B; 10,000 → 500,000; 2e-5 (7B/13B), 1e-5 (34B/70B) | arXiv:2309.16039v3 §2.1 ([[llama-2-long]]) | verified 2026-09-14 | Tables 5–6, Fig. 5b (7B, 80B tokens) |
| Yarn-Llama-2 s = 16 | 7B, 13B | long-context | steps; batch; LR; data | 400; 64; 2×10⁻⁵; PG19 64k chunks | arXiv:2309.00071v3 §4.1 ([[yarn]]) | verified 2026-09-14 | App. B.2 Table 6 (400 vs PI 1,000 steps) |
| LLaMA 7B/13B PI | 7B, 13B | long-context | steps; LR | 1,000; 2×10⁻⁵ | arXiv:2306.15595v2 §3.1 ([[position-interpolation]]) | verified 2026-09-14 | Tables 3–4 (200 steps reach target) |

**Starting point for a small general-purpose run.** For an 8B dense RoPE model extended from 8K to 64K with an academic budget, the ProLong 64K stage is the most fully documented verified configuration: 20B tokens at 64K, RoPE base 8×10⁶, 30% code repositories + 30% books + 3% textbooks + 37% high-quality short data, cross-document masking, LR 1e-5 with 10% warmup and cosine decay to 1e-6, 4M-token batches; it was run from Llama-3-8B-Instruct on H100s (2.2K H100 hours, [[prolong-recipe]]). If the model uses sliding-window layers, the Olmo 3 7B configuration is the verified alternative: YaRN factor 8 (beta_fast 32, beta_slow 1) on full-attention layers only, 34% long / 66% short data, 50B tokens, about 4M-token batches, intra-document masking; it was run from the Olmo 3 7B midtraining checkpoint. In both cases, the stage gate of Llama 3 (short-context panel recovered, retrieval solved at the current length) is the verified stopping rule, and long-context SFT data should be decided by evaluating after SFT (§7).

## Generalization lens

**(a) What increases breadth.**
- Long data from several domains rather than one: per-source upsampling avoids loss increases on other domains ([[fu-2024-data-engineering-128k]] Table 5); books + repositories beat either alone ([[prolong]] Table 4).
- High-quality short data in the mix: ShortMix beats SlimPajama, FineWeb-Edu, and DCLM on both long and short averages ([[prolong]] Table 6).
- Larger continued-pretraining budgets with broad data: Llama 2 Long's 400B tokens raised MMLU at every size ([[llama-2-long]] Table 1).
- Training beyond the evaluation length: 262K and 512K stages improved 128K and 64K results ([[qwen-long-context-synth]] Table 2; [[prolong]] Table 7).
- Position changes restricted to where they are needed: mixed-window training ([[longrope2]] Table 7), YaRN on full-attention layers only ([[olmo-3-long-context]] Fig. 13a).

**(b) What causes narrowing or forgetting.**
- Long-only or long-heavy mixtures: monotonic short-task decline ([[prolong]] Fig. 3), −2.5 vs −0.8 points ([[olmo-3-long-context]] §3.6.3), 92.5% retention ([[longred]] Table 3).
- Single-domain long data: book upsampling raises code and StackExchange loss ([[fu-2024-data-engineering-128k]] Table 5); YaRN's PG19-only fine-tuning lowered 13B MMLU by 3.9 points ([[yarn]] Table 3).
- Large interpolation factors: PI BoolQ loss grows with s ([[position-interpolation]] Table 5).
- Static inference-time scaling on short inputs (caution without a number, [[qwen-3-long-context]]).
- Math and reasoning scores are the most sensitive in the reports reviewed: Olmo 3 Math aggregate −5.4 (7B) and −8.3 (32B); Qwen2.5-14B-1M GPQA −5.6; ProLong-8B GSM8K 68.5 → 58.9 after SFT, which the authors attribute partly to Llama-3-8B-Instruct's closed instruction data ([[olmo-3-long-context]] Table 13; [[qwen-long-context-synth]] Table 6; [[prolong]] Table 25).
- Post-training that ignores long context: short-only SFT regressed Llama 3's long context ([[llama-3]] §4.3.4); thinking mode lowers Qwen3 RULER ([[qwen-3-long-context]] Table 23).

**(c) How to measure it for this stage.**
- A short-context panel before and after each stage, reported per task, not only as an average (§4 point 3). Include math and code, which dropped most.
- Long-context tasks at several lengths, not perplexity: 100% long data improves PG19 perplexity while lowering downstream scores ([[prolong]] Fig. 1); a small base gives low perplexity without retrieval ([[rope-base-bounds-context-length]] Fig. 3). Perplexity does track downstream scores among exact-attention methods in one controlled study ([[controlled-long-context-extension]] Fig. 4), so it is a necessary but not sufficient check.
- Evaluate long-context ability after SFT: RAG and re-ranking gains appear only after SFT ([[prolong]] §2.2, Fig. 2).
- Keep a held-out long-context suite: Olmo 3 develops on RULER and reports HELMET as unseen ([[olmo-3-long-context]] §3.6); ProLong checks held-out HELMET tasks, NoCha, RULER, and ∞Bench ([[prolong]] Limitations). [[ch-32c]] covers the known measurement errors of these suites.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Validating an extension run by perplexity only | Low perplexity at the target length; retrieval fails at lengths below the target (1K for base 500 at 32K) | Run multi-needle or RULER-style retrieval at 4K, 16K, 64K and the target; compare to [[rope-base-bounds-context-length]] Fig. 3 pattern |
| RoPE base below the lower bound for the target length | Retrieval limited to a fraction of the window despite training at full length | Compare base with Table 2 of [[rope-base-bounds-context-length]]; test a larger base on a short run |
| Long-only continued-pretraining mix | Short-context average falls over training steps after an initial recovery | Log the short panel every few hundred steps ([[longred]] Fig. 3); add 37–66% high-quality short data |
| Upsampling one long domain (books only) | Loss on web, code, or StackExchange validation rises | Per-domain validation loss at 0–4K ([[fu-2024-data-engineering-128k]] Table 5) |
| Reporting only aggregate short scores | Average "similar", one task down 5+ points | Per-task deltas, especially GSM8K, GPQA, MATH, code |
| Enabling static YaRN for all traffic | Short-prompt quality lower than without `rope_scaling` | A/B the short panel with and without the config; use dynamic scaling or enable per request ([[qwen-3-long-context]]) |
| Applying RoPE scaling to sliding-window layers | Lower RULER than full-layer-only scaling | Check which layers the scaling config touches (`full_attn_layers_only`, [[olmo-3-long-context]]) |
| Packing across documents without masking in long CPT | Slightly lower long and short scores | Ablate document masking on a 5B-token run ([[prolong]] Table 20) |
| Evaluating long context only before SFT | Long-context ability lost after short-only SFT goes unnoticed | Re-run the long suite after SFT, DPO, and RL ([[llama-3]] §4.3.4; [[prolong]] Fig. 2) |
| Treating a length-mix ratio as a short-data ratio | Copying "75/25" from Qwen as a general-data share | Read whether the ratio describes lengths within long data or long vs general data (§5.1) |

## Check your understanding

1. PI and NTK-by-parts both let LLaMA 7B, trained at 2K, read 32K inputs (s = 16). Explain why PI lowers MMLU more than NTK-by-parts in [[yarn]] Table 2 (25.9 versus 32.7, base 35.7), using what each does to the high-frequency dimensions.
2. A team extends a 4K model to 128K with base 5,000 and reports perplexity at 128K lower than at 4K. Explain, using B_{m,θ}, why this does not show that the model can use the 128K window, and state which evaluation would expose the problem.
3. ProLong finds that 60% long data gives the best long-context average after SFT, while Olmo 3 chooses 34% long data. Explain how the two evaluations used to choose the ratio could lead to different optima, and what experiment would decide between them for a new model.
4. STRING improves effective length without training, and Qwen2.5-1M's 262K stage improves RULER at 128K. Explain both results with the position-frequency formula f(i).
5. LongReD separates drift from forgetting. For each, explain which of these controls addresses it and why: short-data replay, restoration distillation, YaRN restricted to full-attention layers.
6. Llama 3 needed 0.1% long SFT data, while ProLong found short-only SFT best. Propose a causal explanation that is consistent with both reports and with Llama 3's DPO observation, and describe the ablation that would test it.
7. In LongPO, DPO on the same pairs degraded short-context tasks but LongPO did not. Explain what the chunk-conditioned reference changes in the implicit KL term, and why this could matter more for retention than the rejected term.
8. Explain why a static YaRN factor can change a model's output on a 2K prompt when the model's original window is 32K, using the ramp formula and the Hugging Face implementation.

## Connections

- **Previous:** [[ch-32a]] — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining. The replay and forgetting controls there are the general form of §5.1 and §6.1.
- **Next:** [[ch-32c]] — Claimed versus Effective Context Length and Long-Context Evaluation. It covers the measurement suites used in §4, §5, and §8.
- **Builds on this chapter:** [[ch-32d]] — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training; [[ch-29a]] — Long-Document Synthesis for Continued Pretraining and Long-Context SFT; [[ch-44a]] — Length in RL: Overlong Responses, Length Control, and Long-Context RL.

## Sources

- [[position-interpolation]] — PI formula, bounds, passkey steps, BoolQ regression by scale factor, recipe rows.
- [[yarn]] — NTK-aware, NTK-by-parts, and YaRN formulas; Table 2, 3, 5 short- and long-context results; recipe rows.
- [[eleuther-extending-the-rope]] — community origin of NTK-aware and dynamic scaling (anecdotal status), fixed-factor short-length effect.
- [[hf-transformers-yarn-code]] — static YaRN implementation in Transformers v4.46.0.
- [[llama-2-long]] — ABF, unmodified-RoPE failure, ABF vs PI short tasks, data-quality ablation, recipe rows.
- [[rope-base-bounds-context-length]] — B_{m,θ}, lower bound table, superficial long-context results.
- [[longrope2]] — critical dimensions, needle-perplexity search, mixed-window training, retention table.
- [[dual-chunk-attention]] — DCA position remapping and training-free perplexity results.
- [[string-effective-context]] — position-frequency formula and statistics; STRING result.
- [[controlled-long-context-extension]] — controlled comparison of extension methods on LLaMA2-7B; short perplexity and LCC drops.
- [[prolong]], [[prolong-recipe]] — long:short ratio, data sources, short-data quality, masking, RoPE base, short-only SFT, recipe rows.
- [[fu-2024-data-engineering-128k]] — per-source length upsampling, domain-loss table, token budget, MMLU after extension.
- [[llama-3]], [[llama-3-recipe]] — θ = 500,000, six-stage gate, document mask, annealing, 0.1% long SFT, short-only DPO.
- [[qwen-long-context-synth]] — Qwen2.5-1M stages and bases, 75/25 length mix, short benchmarks, short-only RL, DCA + YaRN.
- [[longred]] — drift and forgetting analysis, restoration distillation objective and results.
- [[longpo]] — short-to-long preference objective, KL reference, long and short results.
- [[skyladder]] — context-window scheduling in pretraining.
- [[qwen-3-long-context]] — Qwen3 long-context stage, RULER by mode, static-YaRN model-card caution.
- [[olmo-3-long-context]] — Olmo 3 mixture ablation, YaRN on full-attention layers, released scripts, stage scores.
- [[gemma-3-long-context]] — local:global design, factor-8 interpolation, RULER before and after instruction tuning.
- [[harm-de-vries-long-context-run]] — document-length statistics and attention FLOPs formula.
