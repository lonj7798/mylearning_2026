<!-- scope: Peng, Quesnelle, Fan, Shippole (arXiv:2309.00071, Aug 2023; v3 Feb 2026) — YaRN RoPE context extension: NTK-aware, NTK-by-parts, attention temperature, Dynamic Scaling; Llama 2 7B/13B to 128K with 400+200 PG19 steps; short-benchmark cost
     deps: [[position-interpolation]]
     see-also: [[localllama-ntk-aware-rope]], [[deepseek-v3]], [[string-effective-context]], [[longrope2]], [[hf-transformers-rope-utils]]
-->

# YaRN: Efficient Context Window Extension of Large Language Models
- **Core Insight:** Combining per-dimension RoPE interpolation ("NTK-by-parts") with an attention temperature √(1/t) = 0.1 ln(s) + 1 extends Llama 2 7B and 13B to a 128K window with 400 + 200 fine-tuning steps on 64k-token PG19 chunks, reaching 99.4% passkey accuracy up to 128k (§3.3; §4.1; Table 9).
- **Guideline:** When extending a Llama-family RoPE model by a large factor (the paper tests s = 16 and 32) with a few hundred fine-tuning steps, use YaRN with α = 1, β = 32 and the Eq. 15 temperature, because on LLaMA 7B at s = 16 after 400 steps it had the lowest 32k proof-pile perplexity of the four methods (2.77 vs PI 3.57; Table 5); measure MMLU afterwards, since Llama 2 13B lost 3.9 points at s = 32 (55.8 → 51.9; Table 3).
- **Authors:** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole (Nous Research, EleutherAI, University of Geneva)
- **Year:** 2023 (arXiv v1 2023-08; v2 2023-11; v3 2026-02)
- **URL:** https://arxiv.org/abs/2309.00071 (code: https://github.com/jquesnelle/yarn)
- **Source type:** paper
- **Relevant topics:** RoPE, context window extension, NTK-aware scaling, attention temperature, Dynamic Scaling, long-context fine-tuning, short-context regression

## Abstract
RoPE models fail to generalize past their training sequence length. YaRN (Yet another RoPE extensioN method) is a
compute-efficient extension method that the authors report needs 10x fewer tokens and 2.5x fewer training steps than
previous methods. LLaMA models extended with YaRN use and extrapolate to context lengths longer than their
pretraining allowed, surpass the previous state of the art in context window extension, and extrapolate beyond the
context length of the fine-tuning data (abstract).

## Key Contributions
- A written account of three community methods: "NTK-aware" (bloc97), "Dynamic NTK" (emozilla), "NTK-by-parts" (bloc97) (§1).
- An analysis that uniform PI scaling removes high-frequency RoPE information and that dimensions with wavelength longer than L carry absolute position (§3.1-3.2).
- YaRN = NTK-by-parts interpolation + attention temperature, implemented by scaling the rotary embeddings (§3.3, Definition 2).
- Dynamic Scaling for inference without fine-tuning, and a kv-cache implementation note (§3.4).
- Llama 2 7B/13B checkpoints at 64k (s = 16) and 128k (s = 32) with perplexity, passkey, and Open LLM Leaderboard results (§4; App. B).

## Key Figures/Tables to Study
- Table 5 (App. B.1): perplexity of PI, NTK-aware, NTK-by-parts, YaRN, with and without fine-tuning, s = 2-16.
- Tables 2-3 and 10: short-context benchmarks after extension; Table 9: passkey retrieval; Table 4: A100-hours.
- App. A.3 Figs. 4-6: perplexity versus temperature on 896 16k-token RedPajama documents.

## Technical Details
- **Notation.** Scale factor s = L′/L, with L the pretrained and L′ the extended length (§2.2). RoPE frequency θ_d = b^{−2d/|D|}, b = 10000, |D| the hidden size of the attention layer; wavelength λ_d = 2π/θ_d = 2π b^{2d/|D|} (§2.1; Eq. 8).
- **PI in this notation:** position m is rescaled to mL/L′ with θ unchanged (App. A.1, Eq. 16). The authors state earlier PI fine-tunes reached only about s = 8 before outputs degraded (§3.1).
- **NTK-aware:** keep positions, change the base to b′ = b·s^{|D|/(|D|−2)} (App. A.2, Eqs. 19, 22). Drawbacks stated: the best base is found empirically (§3.1); some dimensions extrapolate out of bound, so fine-tuning with it is worse than PI and s must be set above the target scale (App. A.2). Code Llama's base change to 1M ("ABF") is described as NTK-aware scaling (§3.1 footnote).
- **NTK-by-parts:** r(d) = L/λ_d is the number of rotations dimension d makes over L (Eq. 10). Ramp γ(r) = 0 if r < α, 1 if r > β, (r − α)/(β − α) otherwise (Eq. 11). New frequency h(θ_d) = (1 − γ(r(d)))·θ_d/s + γ(r(d))·θ_d, with positions unchanged (Eqs. 12-13). Recommended α = 1, β = 32 for the Llama family, "found experimentally" (§3.2). Example (derived): r = 16.5 gives γ = 15.5/31 = 0.5, so that dimension's frequency is the mean of θ_d/s and θ_d.
- **Temperature:** attention uses softmax(q_mᵀk_n / (t√|D|)) (Eq. 14), with √(1/t) = 0.1 ln(s) + 1 for LLaMA and Llama 2 (Eq. 15). It is implemented by scaling q and k by √(1/t) through the rotary embeddings, with zero added inference or training cost (§3.3). Example: s = 8 gives ≈ 1.208 (App. A.3, Eq. 23). Eq. 15 was fitted to the lowest-perplexity 1/t across s for LLaMA 7b-65b without fine-tuning; the same t applies "fairly well" to Llama 2 7b/13b/70b (§3.3).
- **Dynamic Scaling:** each forward pass uses s = max(1, l′/L), where l′ is the current sequence length; with kv-caching, keys and values must be cached before RoPE is applied (§3.4). Dynamic-YaRN allows more than 2x extension without fine-tuning (§1).
- **Data share:** YaRN reaches its results after fine-tuning on less than ~0.1% of the original pretraining data (§1).
- **Ablation, LLaMA 7B, 10 proof-pile documents ≥ 128k tokens, sliding-window perplexity (stride 256), 32768 window:** without fine-tuning at s = 16, YaRN 3.45, NTK-by-parts > 10, NTK-aware > 10, PI > 10²; fine-tuned 400 steps at s = 16: YaRN 2.77, NTK-by-parts 2.81, PI 3.57, NTK-aware 8.49 (Table 5).
- **Step efficiency, Llama 2 7B, s = 2:** PI 1000 steps 3.34 vs YaRN 400 steps 3.35 at 8192 (App. B.2, Table 6).
- **128k perplexity:** Llama 2 7B s = 32: 2.37 at 131072; 13B s = 32: 2.24; s = 16 models > 10 at 131072 (Table 1). Code Llama 7B 2.71 and Together 32k > 10⁴ at 131072 (Table 7).
- **Passkey (10 iterations per size, 8k-128k):** YaRN s = 32 7B and 13B: passkey context 128k, 99.4%; s = 16: 96.3% (7B) and 97.5% (13B) at 64k (App. B.5, Table 9). The authors hypothesize perplexity is not a sufficient indicator of attending to all tokens (App. B.5).
- **Compute (A100-hours):** LLaMA 7B YaRN 2k×16: 128; Llama 2 7B YaRN 4k×16: 256; 4k×32: 256 + 128; PI 2k×8 as reported for Chen et al.: 640; Code Llama NTK-aware 4k×88.6: 6400 (Table 4).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Yarn-Llama-2 s = 16 (64k) | 7B, 13B | long-context | optimizer; peak LR; warmup; weight decay | AdamW β1 = 0.9, β2 = 0.95; 2 × 10^-5; linear warmup 20 steps; no weight decay | arXiv:2309.00071v3 §4.1 | verified 2026-09-14 | no ablation reported |
| Yarn-Llama-2 s = 16 (64k) | 7B, 13B | long-context | steps; global batch | 400 steps; global batch size 64 | §4.1 | verified 2026-09-14 | App. B.2 Table 6: YaRN 400 steps 3.35 vs PI 1000 steps 3.34 at 8192 (Llama 2 7B, s = 2) |
| Yarn-Llama-2 s = 16 (64k) | 7B, 13B | long-context | data; sequence length | PG19, contiguous 64k-token chunks bookended with BOS and EOS | §4.1; §6 | verified 2026-09-14 | no ablation reported |
| Yarn-Llama-2 s = 16 (64k) | 7B, 13B | long-context | tokens seen | 400 × 64 × 64k = 1.64B (64k = 64,000) or 1.68B (64k = 65,536), assuming every sequence is a full chunk | §4.1 inputs | derived | — |
| Yarn-Llama-2 s = 32 (128k) | 7B, 13B | long-context | init; steps; data | finished s = 16 checkpoint; 200 more steps; same 64k data ("due to compute constraints") | §4.1 | verified 2026-09-14 | Table 1: extrapolates to 131072 (2.37 / 2.24) |
| YaRN, Llama family | all | long-context | α; β | α = 1, β = 32 | §3.2 | verified 2026-09-14 | "found experimentally"; no table |
| YaRN, LLaMA / Llama 2 | all | long-context | temperature | √(1/t) = 0.1 ln(s) + 1 | §3.3, Eq. 15 | verified 2026-09-14 | fit on LLaMA 7b-65b without fine-tuning (§3.3; App. A.3) |
| LLaMA 7B ablation runs | 7B | long-context | data; s; steps | PG19 32k segments; s = 16; 400 steps; otherwise "similar" to the 128k runs | §4.1 | verified 2026-09-14 | Table 5, Fig. 2 |
| Yarn-Llama-2 7B | 7B | long-context | compute | 256 A100-hours (s = 16); 256 + 128 (s = 32) | Table 4 | verified 2026-09-14 | not applicable |
| Yarn-Llama-2 | 7B, 13B | long-context | GPU count; LR decay; loss masking | not reported (checked §4, §6, App. A-B) | — | not reported | — |

## Findings relevant to generality and long context
- **Short-context cost (Result, single study):** Llama 2 7B ARC-c 53.1 → 52.1, MMLU 43.8 → 41.7, TruthfulQA 39.0 → 37.3, HellaSwag 77.8 → 78.4 at s = 32; 13B ARC-c 59.4 → 58.0, MMLU 55.8 → 51.9, HellaSwag 82.1 → 82.2 (Table 3). Average drop from s = 16 to s = 32 is 0.49% (§4.4). The authors attribute variance to PG19 differing from the pretraining data (§4.4, Interpretation).
- **Method comparison on short tasks, LLaMA 7B s = 16, 400 steps:** MMLU base 35.7; PI 25.9, NTK-aware 27.7, NTK-by-parts 32.7, YaRN 30.0; HellaSwag YaRN 77.2 vs PI 70.2 (Table 2). YaRN is not the best method on MMLU or ARC-c in this table (NTK-by-parts 48.5 vs YaRN 48.1), although §5 describes YaRN as a drop-in replacement for PI "with no downsides".
- **Length generalization:** the s = 32 models trained only on 64k chunks reach perplexity 2.37 (7B) and 2.24 (13B) at 131072 tokens and 99.4% passkey accuracy up to 128k (Tables 1, 9).

## Connections
- [[position-interpolation]] — PI baseline; §4.1 follows its training and evaluation procedures.
- [[localllama-ntk-aware-rope]] — bloc97's NTK-aware post (cited as bloc97 2023a); [[eleuther-extending-the-rope]] — related RoPE-extension write-up.
- [[hf-transformers-rope-utils]] — library implementation of RoPE scaling variants.
- [[deepseek-v3]] — its card records YaRN with α = 1, β = 32 for 4K → 32K → 128K extension; [[qwen-2.5]] — its card records YaRN for 1M extrapolation.
- [[llama-2-long]] — evaluates Yarn-7B/13B-128k checkpoints as baselines; [[llama-2]] — the extended base models.
- [[string-effective-context]] — YaRN as a training-free baseline: Llama 3.1 8B RULER-128K 76.3 vs RoPE 77.0 (that paper's Table 2).
- [[longrope-data]], [[longrope2]], [[randomized-yarn]] — later per-dimension or randomized rescaling methods.
- [[rope-base-bounds-context-length]], [[rope-extrapolation-scaling-laws]] — analyses of RoPE base choice and extrapolation.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2309.00071 (arXiv v3, 2026-02-06; section and table numbers refer to v3).
- Audit claims not found in the source: "ICLR 2024" (no venue in v3 or on the abs page); "YaRN is the scheme DeepSeek-V3, Qwen, gpt-oss and UltraLong deploy" (the paper names Code Llama as NTK-aware and Qwen 7B as Dynamic NTK, §1; other adoptions are not in this source).
- Not reported by the source: GPU count, LR schedule after warmup, loss masking, results for s > 32.
