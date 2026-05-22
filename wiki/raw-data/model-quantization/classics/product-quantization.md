<!-- scope: split high-d vectors into sub-vectors, quantize each with a small codebook; tractable VQ at scale
     deps: [[vector-quantization]]
     see-also: [[aqlm]], [[vptq]], [[gptvq]], [[quip-sharp]]
-->

# Product Quantization for Nearest Neighbor Search (Jégou, Douze, Schmid 2011)
- **Core Insight:** A length-D vector can be split into M sub-vectors of length D/M and each sub-vector quantized independently with a small codebook of size K, producing an effective codebook of size K^M with only M·K·(D/M) storage and O(M·K) encoding cost — making vector quantization tractable at billion-scale.
- **Guideline:** When applying VQ to LLM weight matrices, split each weight row into M sub-vectors (e.g. M=8, K=256 → 1 byte per sub-vector = 1 bit/weight effective at D=64); train each sub-codebook independently via k-means on the empirical sub-vector distribution.
- **Authors:** Hervé Jégou, Matthijs Douze, Cordelia Schmid (INRIA)
- **Year:** 2011
- **URL:** https://hal.inria.fr/inria-00514462/document ; IEEE TPAMI 33(1)
- **Relevant topics:** vector quantization, ANN search, codebook factorization, FAISS, additive quantization precursor

## Abstract
Product quantization (PQ) approximates the Cartesian distance between a query and database vectors by decomposing each vector into M disjoint sub-vectors and quantizing each independently with a K-entry codebook learned via k-means. The resulting reconstruction has effective codebook size K^M — astronomically large — while requiring only M·K codewords of dimension D/M to store. Asymmetric distance computation (ADC) compares an unquantized query to PQ-encoded database vectors using precomputed sub-distance tables. PQ became the standard for billion-scale ANN search (FAISS) and is the direct ancestor of additive / vector quantization methods for LLM weight compression.

## Key Contributions
- **Cartesian product codebook:** factorize the codebook C = C_1 × C_2 × … × C_M, exponentially expanding capacity at linear cost.
- **Asymmetric distance computation (ADC):** precompute M lookup tables of size K; approximate ||x − y||² as sum of M table lookups.
- Empirically demonstrates billion-scale ANN search with 16-byte codes (M=16, K=256) at recall@1 within a few % of exact.
- Implementation in FAISS made PQ the production-standard ANN method; directly inspired AQLM-style LLM weight quantization.

## Key Figures/Tables to Study
- **Recall vs code length** plot: PQ achieves much better recall per bit than scalar/binary hashing or LSH.
- **ADC table illustration**: M × K precomputed sub-distances allow ||x − y||² ≈ Σ_m d_m[code(y_m)] in M additions.

## Technical Details

### Decomposition
Split a D-dim vector x into M sub-vectors of length D' = D/M:
```
x = [u_1; u_2; …; u_M],     u_m ∈ ℝ^{D/M}
```
Quantize each u_m with its own sub-codebook C_m = {c_m,1, …, c_m,K} of size K (typically K = 256 so each code fits in one byte):
```
code_m(u_m) = argmin_k ||u_m − c_m,k||²
PQ(x) = [c_1,code_1(u_1); …; c_M,code_M(u_M)]
```

### Storage and rate
- Codebook storage: M · K · (D/M) = K·D floats.
- Per-vector code: M · log₂ K bits (e.g. M=16, K=256 ⇒ 16 bytes per vector regardless of D).
- Effective codebook size: K^M (e.g. 256^16 = 2^128).

### Asymmetric Distance Computation (ADC)
For query q = [q_1; …; q_M], precompute M tables of size K:
```
T_m[k] = ||q_m − c_m,k||²       for m = 1…M, k = 1…K
```
Then for any database vector x with PQ codes (k_1, …, k_M):
```
||q − PQ(x)||² ≈ Σ_m T_m[k_m]
```
M table lookups + M−1 adds per database comparison.

### Sub-codebook training
Run k-means independently on the empirical distribution of each sub-vector u_m across the training set:
```
for m = 1…M:
    C_m ← KMeans(K, {u_m^{(t)} : t = 1…T})
```

### Optimized PQ (OPQ)
A learned rotation R is applied before splitting (x → R·x) to balance variance across sub-vectors → significantly better distortion at same bits. Direct precursor to **rotation-based LLM quant** ([[quarot]], [[spinquant]]).

### Additive / Residual quantization
- **Residual VQ (RVQ):** quantize x, then quantize the residual x − c_1, then residual of residual, …
- **Additive quantization:** PQ(x) = Σ_m c_m where each c_m comes from its own full-dim codebook (not sub-vector). More expressive than PQ, harder encoding (combinatorial).
- **AQLM = additive quantization for LLM weights** ([[aqlm]]).

### Bit budget for LLM weights
For per-row weight quantization at sub-2-bit: typical setting D=512–4096, M=64–256, K=256, giving ~1.5–2 bits/weight at ~3–5× compression vs INT4.

## Connections
- [[vector-quantization]] — PQ is a tractable factorization of VQ codebook.
- [[aqlm]] — additive VQ for LLM weights (PQ → AQ generalization).
- [[vptq]] / [[gptvq]] — direct PQ-style LLM weight compression with GPTQ-style Hessian update.
- [[quip-sharp]] — uses E8 lattice + Hadamard rotation; rotation idea inherited from OPQ.
- [[quarot]] / [[spinquant]] — Hadamard / learned rotations applied to LLM weights/activations; conceptual lineage from OPQ.
