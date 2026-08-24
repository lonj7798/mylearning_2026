<!-- chapter: ch-extra
     track: prerequisite
     kind: content
     title: Attention and the Transformer, From Scratch (Korean companion)
     position: between [[ch-03]] and [[ch-04]]
     deps: [[ch-03]]
     feeds: [[ch-04]]
     sources: [[qkv-scaled-dot-product]], [[sqrt-dk-scaling-variance]], [[causal-mask-neg-inf]],
              [[multi-head-split-concat-wo]], [[attention-permutation-equivariance]],
              [[sinusoidal-absolute-encoding]], [[rope-rotary-position-embedding]],
              [[transformer-block-tensor-ledger]], [[pre-ln-vs-post-ln]],
              [[residual-stream-memory-backbone]], [[kv-cache-mechanism]],
              [[kv-cache-memory-formula]], [[gqa-mqa-mla-kv-heads]], [[train-vs-infer-kv-boundary]]
     created_at: 2026-08-18
-->

# 보충 장 — Attention과 Transformer, 바닥부터

> **핵심 통찰.** 이 course에 나오는 모든 memory 숫자는 단 하나의 설계 결정에서 파생된다. Attention은 `N×N` 크기의 pairwise score table 전체를 계산해서 정보를 routing하며, 그 table은 *activation*이다 — forward pass가 반드시 만들어 내야 하고 (naive implementation에서는) backward pass에 넘겨줘야 하는 tensor다. Transformer의 나머지 모든 것 — embedding table, 세 개의 projection, head split, residual stream, 4h MLP — 은 byte cost가 sequence length에 대해 **linear**하다. 오직 routing table만이 **quadratic**하다. 따라서 이 mechanism을 이해하는 것은 memory course를 듣기 전의 배경 지식이 아니다. 그것은 ledger 수준이 아니라 algorithm 수준에서 서술된 *memory course 그 자체*다.

> **지침.** Transformer를 diagram으로 배우지 말고 각 단계가 만들어 내는 tensor로 배워라. 모든 operation에 대해 다음 세 질문을 순서대로 던져라. (1) output의 *shape*은 무엇인가? (2) backward pass는 input이 필요한가, output이 필요한가, 아무것도 필요 없는가? (3) 그 shape 안에 `N²`이 들어 있는가? Block 하나의 tensor 약 16개에 대해 이 세 질문에 답하면 Korthikanti의 `34·s·b·h + 5·a·s²·b` coefficient를 처음부터 재구성할 수 있고 — 어떤 lever([[ch-03]]의 checkpointing, [[ch-04]]의 streaming kernel, [[ch-07]]의 parallelism)가 어떤 항을 건드릴 수 있는지 즉시 알 수 있다.

---

**이 chapter가 존재하는 이유.** 이 chapter는 [[ch-03]] 뒤, [[ch-04]] 앞에 삽입되었다. ch-04의 memory 분석이 mechanism을 이미 안다고 가정하기 때문이다. 이 chapter는 token에서 출발해 mechanism을 쌓아 올리고, §8에서 명시적으로 handoff한다. 여기 있는 내용 중 형제 chapter들과 모순되는 것은 없다. 어떤 숫자가 이미 `ch-03/read.md`, `ch-04/read.md` 또는 그들의 `qa` page에 등장했다면, 다시 계산하지 않고 그대로 재사용하며 cross-reference한다.

**Reference configuration.** 별도 언급이 없는 한 이 chapter의 모든 byte 수치는 하나의 고정된 setup을 쓴다. [[ch-03]]의 figure와 [[ch-04]]의 `qa`에 맞춘 값이다.

```
B = 1          batch (GPU당 micro-batch)
T = s = 4096   sequence length
h = d_model = 4096
a = n_heads = 32       ->  d_head = h/a = 128
d_ff = 4h = 16384
dtype = bf16 = 2 bytes/element
L = 80 blocks
```

너무 자주 등장해서 이름을 붙여 둘 만한 unit tensor가 셋 있다.

| 이름 | Shape | Element 수 | Bytes | |
|---|---|---|---|---|
| **HID** | `[B, T, h]` | 16,777,216 | 33,554,432 B | **33.55 MB** — residual stream의 snapshot 하나 |
| **ATTN** | `[B, a, T, T]` | 536,870,912 | 1,073,741,824 B | **1.07 GB** — routing table |
| **MLP** | `[B, T, 4h]` | 67,108,864 | 134,217,728 B | **134.22 MB** |
| LNST | `[B, T] × 2` fp32 | 8,192 | 32,768 B | 32.8 KB — LayerNorm이 저장하는 (mean, rstd) |

`s·b·h = 16,777,216` element이므로 **"sbh byte-unit" 하나 = 16,777,216 B = 16.78 MB**이다. §8의 Korthikanti coefficient들은 이 unit의 배수다.

💡 **쉬운 설명.** 이 세 tensor를 "화폐 단위"라고 생각하면 편하다. 앞으로 나오는 모든 memory 계산은 결국 "HID 몇 개 + ATTN 몇 개 + MLP 몇 개"로 환산된다. 그리고 HID 하나(33.55 MB)와 ATTN 하나(1.07 GB)의 비율이 약 32배라는 점만 기억해도, 왜 attention이 ledger를 지배하는지 절반은 이해한 것이다.

**Convention — 한 번만 읽어 두면 chapter 간 산술 분쟁이 대부분 해소된다.**

1. **FLOPs.** multiply–accumulate(MAC) 하나 = **2 FLOPs**. 따라서 `(m×k)·(k×n)` matmul의 비용은 `2·m·n·k` FLOPs다. 이 chapter, [[ch-04]], 그리고 figure들의 모든 FLOP count가 이 convention을 쓰므로 숫자들이 chapter를 가로질러 조합된다. (MAC을 1 FLOP으로 세는 논문은 여기 나오는 모든 값의 정확히 절반을 보고한다.)
2. **Bytes.** 모든 byte count는 **정확한 정수**이고, 거기 붙는 약어는 **decimal**이다 — `MB = 10⁶ B`, `GB = 10⁹ B`, `TB = 10¹² B`. 그래서 `33,554,432 B`는 `33.55 MB`로, `1,073,741,824 B`는 `1.07 GB`로 쓴다. 어떤 숫자가 마침 딱 떨어지는 **binary** 양이고 그 구분이 중요한 곳(KV-cache 크기, cos/sin cache, ALiBi bias)에서는 binary unit을 명시한다: `MiB = 2²⁰ B`, `GiB = 2³⁰ B`, `TiB = 2⁴⁰ B`. 두 체계가 변환 없이 하나의 산술 단계 안에 섞이는 일은 없다.
3. **형제 chapter의 알려진 불일치 — 안전하게 읽으라고 명시해 둔다.** `ch-04/read.md` §5(L211–216)는 head당 `8 MB / 128 MB / 2 GB`, model 전체 `8 GB / 128 GB / 2 TB`를 인쇄하는데, 이것들은 **binary** 양에 **decimal** label이 붙은 것이다(정확히는 8 MiB / 128 MiB / 2 GiB, 그리고 8 GiB / 128 GiB / 2 TiB). 이 chapter의 §4.4가 같은 row를 unit을 올바르게 명명해 다시 서술한다. 숫자는 하나도 다르지 않고 label만 다르다.
4. **기호.** 이 chapter 전체에서 `h = d_model`, `a = n_heads`이며, 이는 Korthikanti의 `34·s·b·h + 5·a·s²·b`와 일치한다. Head 수는 **언제나** `a`이고 절대 `h`가 아니다. 예외는 (i) §4.1의 Vaswani 원문 block quote와 §4.5의 원문 Table-3 config — 거기서는 논문 자신의 `h`가 head 수를 뜻한다 — 그리고 (ii) §4.5 figure callout에서 그 figure 자체의 labelling을 서술하는 한 문장뿐이며, 셋 다 등장하는 자리에 표시해 두었다. 다른 자료를 읽을 때 이 점을 조심하라: Vaswani, HuggingFace config(`num_attention_heads`), 대부분의 blog post는 `h`를 head 수로 쓰고, memory 문헌(Korthikanti, Megatron)은 `h`를 hidden size로 쓴다.

💡 **쉬운 설명.** 이 네 항목은 "왜 같은 계산인데 chapter마다 숫자가 다르지?"라는 짜증의 원인을 미리 제거해 둔 것이다. (1) MAC 하나를 2 FLOPs로 세느냐 1로 세느냐에 따라 모든 FLOP 값이 **정확히 두 배** 차이 난다 — 이 chapter는 2로 센다. (2) `GB`(10⁹)와 `GiB`(2³⁰)는 7.4% 다르고, 이 차이가 §7.2에서 "동시 request 156개냐 168개냐"를 가른다. (3) 형제 chapter ch-04는 binary 양에 decimal label을 붙여 놓았으니, 그 표를 볼 때 `128 MB`는 사실 `128 MiB`라고 읽어라. (4) 가장 자주 사고를 내는 것이 `h`다 — Vaswani 논문과 HuggingFace config에서 `h`는 **head 수**이고, memory 문헌과 이 chapter에서 `h`는 **hidden size**(4096)다. 이 chapter는 head 수를 언제나 `a`로 쓰므로, `h`가 보이면 4096을 떠올려라.

> **⚠ boson / Lina TMR을 위한 scope note — 여기 있는 것을 집으로 가져가기 전에 반드시 읽어라.** 이 chapter의 모든 내용은 **표준 softmax attention**을 서술한다. 즉 `N×N` score matrix 전체가 materialize되고, 그 matrix가 곧 `5·a·s²·b` 항이다. boson의 attention layer는 `CP=1`이 hard-assert된 **GDN linear-attention**이고, linear attention에는 **`N×N` score matrix가 아예 없다** — 따라서 `O(N²)` activation 서사, `5as²b` coefficient, `s = 34h/(5a)` crossover, 그리고 "약 870 token을 넘으면 attention이 지배한다"는 결론은 전부 *GDN이 대체하는 baseline*에 대한 진술이지 당신이 training하는 model에 대한 진술이 아니다. 그래도 배워라. 그것들이 GDN이 존재하는 이유이고, 모든 kernel chapter([[ch-04]]–[[ch-06]])가 다투는 대상이며, 당신이 중간에 끼워 넣는 softmax-attention layer는 그 값을 전액 지불한다. 다만 quadratic 직관을 boson 자신의 budget에 그대로 이식하지는 마라 — §8이 ledger를 넘기는 지점에서 이 단서를 다시 서술한다.

---

## 1. Token에서 Vector로

### 1.1 Tokenization: text가 integer가 된다

Language model은 character를 절대 보지 않는다. Tokenizer(BPE, SentencePiece, tiktoken)가 input string을 크기 `V`의 고정된 vocabulary에서 뽑은 subword unit으로 쪼개고, 그 integer index를 내보낸다.

```
"보험료가 얼마인가요"  ->  [12345, 887, 40219, 61, 9982]   # token id 5개
```

이후 단계에 중요한 성질이 둘 있다. 첫째, `V`는 training 시점에 고정되며 dimension이 아니라 *count*다 — boson/Lina TMR은 **V = 248,000**을 쓴다. Korean morphology에 insurance-domain terminology까지 coverage해야 해서 비정상적으로 크다. 둘째, id에는 geometry가 없다. id 12345는 900보다 12346에 더 "가깝지" 않다. 모든 semantic structure는 학습되어야 하고, 그것이 학습되는 장소가 embedding table이다.

### 1.2 Embedding table `[V, d_model]`

Embedding은 학습 가능한 matrix `E ∈ ℝ^{V × d_model}` 하나다. Forward operation은 matrix multiply가 아니라 **row lookup**이다.

```
token id 12345  ->  E[12345]  ->  ℝ^{d_model} 안의 vector 하나
```

`T`개 token의 sequence에 대해 이것은 `X ∈ ℝ^{B × T × d_model}` — 위에서 HID라고 부른 tensor — 를 만든다. 이 지점부터 loss head 전까지 **shape은 절대 바뀌지 않는다**: `[B, T, h]`가 L개 block 전체에 걸친 파이프의 굵기다. 이 불변성이 §6.3의 주제다.

`d_model`(= `h`, "hidden dimension")은 *representational* axis다. `V`는 *count* axis다. 이 둘을 혼동하는 것이 초기에 가장 흔한 오해이며, 이미 [[ch-02]]의 `qa-deep-2` Q7에서 해소되었다: `E : [V, h]`, 여기서 **V는 어느 row를 고를지, h는 그 row가 얼마나 긴지**를 정한다. 같은 page가 그 결과도 계산한다 — `V = 248,000`, `h = 4096`이면 embedding matrix만으로 약 1.02 × 10⁹ parameter를 갖고, loss head는 맨 끝에서 정확히 한 번 `h → V`로 확장하는데 이 약 60× 팽창이 바로 [[ch-02]] §4의 logit spike다.

### 1.3 Dimension이 "의미하는" 것

유용하면서 정직한 답: `d_model`의 개별 coordinate는 보통 그 자체로는 아무 의미도 없다. 의미를 갖는 것은 *direction*이다. Model은 linear direction이 feature에 대응하는 basis를 학습하며, 유용한 feature의 수가 `d_model = 4096`개의 orthogonal direction보다 훨씬 많기 때문에 feature들은 **superposition** 상태로 저장된다 — 거의 orthogonal한 direction들이 같은 coordinate를 공유하는 것이다. 이것이 embedding 두 개를 더해도 의미 있는 무언가가 나오는 이유이고, §6.3의 residual stream이 공유 communication bus로 작동하는 이유이며, "attention head" 하나가 어떤 subspace는 읽고 다른 subspace는 무시할 수 있는 이유다.

이 course에서 실무적으로 중요한 결론은 더 좁고 구체적이다: `d_model`은 *width*이고, transformer의 모든 activation tensor는 `[B, T, h]`의 어떤 reshape이거나 그것의 확장(MLP의 `[B, T, 4h]`)이다 — **딱 하나만 빼고**. 그 예외가 `[B, a, T, T]` score tensor이며, 이것이 shape 안에 `T`를 두 번 담은 유일한 tensor다. 이 course의 모든 memory pathology는 그 예외 하나로 거슬러 올라간다.

💡 **쉬운 설명.** "dimension 512번은 무슨 뜻이냐"는 질문은 잘못된 질문이다. 올바른 질문은 "이 방향(direction)은 무슨 뜻이냐"다. 4096개 좌표축으로 수만 개 feature를 표현해야 하니, model은 축을 하나씩 배정하지 않고 서로 거의 직교하는 방향들을 겹쳐 쓴다(superposition). Memory 관점에서 실제로 중요한 건 이 철학이 아니라 shape 하나뿐이다 — `T`가 두 번 들어간 tensor는 score tensor뿐이다.

---

## 2. Soft Dictionary Lookup으로서의 Attention

### 2.1 진짜 원하는 lookup, 그리고 그것이 불법인 이유

Hard lookup이 가능하다고 상상해 보자. Query token `i`에 대해 가장 relevant한 token `j*` 하나를 골라 그 content를 복사한다.

```
j* = argmax_j  score(i, j)
o_i = v_{j*}
```

이것은 semantic으로는 정확히 우리가 원하는 것이다 — "대명사 *it*은 자기가 가리키는 noun을 읽어야 한다." 그리고 이것은 학습 불가능하다. `argmax`는 piecewise constant이므로 derivative가 **거의 모든 곳에서 zero**이고 tie 지점에서는 정의되지 않는다. Score를 만들어 낸 무언가로 gradient가 전혀 흘러가지 않으니, model은 더 잘 score하는 법을 영원히 배울 수 없다. [[qkv-scaled-dot-product]]의 표현대로, attention은 *differentiable*한 dictionary lookup이다 — hard argmax가 **모든** value에 대한 softmax-weighted convex combination으로 대체된다.

```
o_i = Σ_j  p_ij · v_j      단,  p_ij ≥ 0,  Σ_j p_ij = 1
```

weight가 non-negative이고 합이 1이므로 `o_i`는 **convex combination**이다 — value vector들의 convex hull 안에 놓인다. Attention은 value 사이를 interpolate할 수 있지만, 그 바깥으로 extrapolate할 수는 절대 없다. (그래서 value path 뒤에 `W_O`가 필요하다. Output projection이야말로 convex hull을 residual stream이 쓸 수 있는 direction으로 다시 mapping하는 장치다.)

💡 **쉬운 설명.** "가장 중요한 하나를 고른다"는 계단 함수이고, 계단은 기울기가 0이라 학습이 안 된다. 그래서 "전부 조금씩 섞되, 중요한 것에 더 큰 weight"로 바꾼 것이 softmax다. 대신 대가가 있다 — 출력이 항상 입력 value들 사이의 "평균 지점"이라서 새로운 방향을 만들어 낼 수 없다. `W_O`가 그 제약을 푸는 출구다.

### 2.2 왜 projection이 하나가 아니라 셋인가

`X ∈ ℝ^{N × d_model}`이 주어졌을 때, 학습되는 linear map 셋을 정의한다.

```
Q = X W_Q      W_Q ∈ ℝ^{d_model × d_k}      Q ∈ ℝ^{N × d_k}
K = X W_K      W_K ∈ ℝ^{d_model × d_k}      K ∈ ℝ^{N × d_k}
V = X W_V      W_V ∈ ℝ^{d_model × d_v}      V ∈ ℝ^{N × d_v}
```

`d_k`는 Q와 K가 dot product되므로 둘이 일치해야 한다. `d_v`는 자유롭고 `W_O`의 input width와만 맞으면 된다. 실무에서는 셋 다 `d_head = d_model / a`로 두며, 여기서 `a`는 head 수다(§4).

각각의 역할을 한 줄씩 ([[qkv-scaled-dot-product]]): **W_Q = "나는 무엇을 찾고 있는가", W_K = "나는 무엇을 광고하는가", W_V = "나는 어떤 content를 전달하는가."** K는 Q와 비교되는 geometry에 살고, V는 `W_O`가 residual stream으로 되돌려 보내는 geometry에 산다. 둘은 의도적으로 *다른* geometry다 — matching과 transmitting은 서로 다른 일이다.

이 필요성을 가장 날카롭게 보는 방법은, projection을 지우고 `X Xᵀ`로 직접 score를 매기면 무슨 일이 생기는지 묻는 것이다. 서로 독립적인 세 가지가 동시에 깨진다.

1. **Symmetry.** `XXᵀ`는 symmetric이라 `S_ij = S_ji`다. 그러나 linguistic relation은 directional하다: "it"은 자신의 antecedent에 강하게 attend해야 하지만 antecedent가 똑같이 되돌아 attend할 필요는 없다. Symmetric score matrix는 one-way edge를 표현할 수 없다.
2. **Diagonal dominance.** RMSNorm 이후 모든 row의 norm이 같다, `‖x_i‖ = √d_model`. 그러면 Cauchy–Schwarz에 의해 `x_i·x_j ≤ ‖x_i‖‖x_j‖ = ‖x_i‖² = (XXᵀ)_ii`이고, 등호는 `x_i = x_j`일 때만 성립한다. 즉 모든 row에서 diagonal이 **strict row maximum**이다 — softmax가 거의 모든 mass를 self-loop에 몰아주고 attention은 identity map으로 붕괴한다. 아무 일도 하지 않는다.
3. **학습 가능한 routing이 없다.** `XXᵀ`는 input의 고정된 함수다. *무엇에 attend할지*를 학습할 parameter가 아예 없다. Routing이 영원히 얼어붙는다.

세 개의 별도 matrix가 이 셋을 한 번에 고친다. `W_Q ≠ W_K`는 symmetry를 깨고, Cauchy–Schwarz 논증을 깨며(score geometry에서는 norm이 더 이상 균등화되지 않는다), parameter를 공급한다.

💡 **쉬운 설명.** 두 번째 이유가 가장 비직관적이니 풀어 쓰면 이렇다. Norm이 전부 같은 vector들끼리 내적하면, 자기 자신과의 내적이 항상 최대다(같은 방향이니까). 그러니 `XXᵀ`로 score를 매기면 모든 token이 "나 자신이 제일 관련 있다"고 판단하고 softmax가 대각선에 몰빵한다. 결과적으로 attention layer가 입력을 그대로 출력하는 값비싼 항등 함수가 된다. `W_Q ≠ W_K`가 Q와 K를 서로 다른 공간으로 보내면 이 "자기 자신이 항상 1등" 논리가 성립하지 않는다.

### 2.3 공식, 그리고 4단계 실행

Vaswani et al. 2017, Equation 1, 원문 그대로:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

batch `B`, single head 기준으로 정확한 shape과 FLOP count를 붙여 4단계로 실행된다.

| Step | Operation | Output shape | Cost |
|---|---|---|---|
| 1 | `S = Q Kᵀ` | `(B, N, N)` | `2·B·N²·d_k` FLOPs |
| 2 | `S_scaled = S / √d_k` | `(B, N, N)` | elementwise |
| 3 | `P = softmax(S_scaled, dim=-1)` | `(B, N, N)` | row 단위; 각 **row의 합이 정확히 1.0** |
| 4 | `O = P V` | `(B, N, d_v)` | `2·B·N²·d_v` FLOPs |

Output의 row `i`는 `o_i = Σ_j P[i,j]·v_j`다. (FLOP count의 factor 2는 표준 multiply-add convention이다. [[ch-04]]가 쓰는 것과 같은 convention이므로 숫자들이 서로 조합된다.)

Step 1–3이 memory 이야기가 시작되는 지점이다. `(B, N, N)` shape tensor 두 개가 동시에 존재하며 — scaled score와 softmax output — step 4는 `P`를 필요로 하고 backward pass도 `P`를 필요로 한다. [[ch-04]] §1.1이 정확히 여기서 이어받는다.

### 2.4 Worked example: key 세 개, `d_k = 4`

`d_k = 4`로 두어 `√d_k = 2`, query 하나, key 셋, `d_v = 2`:

```
q1 = [1, 0, 1, 0]
k1 = [1, 0, 1, 0]     k2 = [0, 1, 0, 1]     k3 = [1, 1, 0, 0]
v1 = [1, 0]           v2 = [0, 1]           v3 = [1, 1]
```

Raw dot product: `q1·k1 = 2`, `q1·k2 = 0`, `q1·k3 = 1`. `1/2`로 scaling: `1.0`, `0.0`, `0.5`. 지수화:

```
exp(1.0) = 2.718282
exp(0.0) = 1.000000
exp(0.5) = 1.648721
sum      = 5.367003
```

```
A_1 = [0.506480, 0.186324, 0.307196]        (합이 1.000000)
      (정확값: 0.50648040, 0.18632372, 0.30719589)
o_1 = 0.506480·[1,0] + 0.186324·[0,1] + 0.307196·[1,1]
    = [0.813676, 0.493520]
```

Convex-hull 주장에 대한 sanity check: `o_1`의 두 coordinate 모두 `v1, v2, v3`가 span하는 범위인 `[0, 1]` 안에 있다. Attention은 interpolate했을 뿐, 새로운 direction을 발명하지 않았다.

### 2.5 `1/√d_k`는 어디서 오는가 — derivation 전체

이제 *같은* example을 scaling 없이 돌려 보자. Score는 `2, 0, 1`:

```
exp(2) = 7.389056
exp(0) = 1.000000
exp(1) = 2.718282
sum    = 11.107338
weights = [0.665241, 0.090031, 0.244728]
```

나눗셈 하나를 뺐을 뿐인데 top weight가 **0.506480 → 0.665241**로 뛰었다 — 그것도 toy width인 `d_k = 4`에서. 분포가 이미 눈에 띄게 뾰족해졌다. `d_k = 128`로 외삽하면 이 뾰족해짐은 saturation이 된다.

**Variance 논증.** Vaswani의 footnote 4, 원문 그대로:

> "To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = sum_{i=1}^{d_k} q_i k_i, has mean 0 and variance d_k."
>
> (번역: dot product가 커지는 이유를 보이기 위해, q와 k의 component가 mean 0, variance 1인 독립 random variable이라고 가정하자. 그러면 그들의 dot product `q · k = Σ_{i=1}^{d_k} q_i k_i`는 mean 0, variance `d_k`를 갖는다.)

전체 단계:

```
E[q·k]   = Σ_i E[q_i]·E[k_i] = 0                     (독립, zero mean)
Var(q·k) = Σ_i Var(q_i k_i) = Σ_i E[q_i²]E[k_i²]
         = Σ_i 1·1 = d_k
sd(q·k)  = √d_k
Var(q·k / √d_k) = d_k / d_k = 1                      (모든 d_k에서 unit variance)
```

즉 raw score의 크기는 `√d_k`로 자라며, `√d_k`로 나누면 *어떤 head dimension에서든* unit variance가 복원된다. 관련 상수:

| `d_k` | `√d_k` |
|---|---|
| 64 | 8 |
| 128 | 11.3137 |
| 512 | 22.6274 |

**Saturation이 왜 치명적인가, 정량적으로.** `d_k = 64`(따라서 `√d_k = 8`)에서 raw dot product가 `s1 = 24.0`, `s2 = 16.0`인 key 두 개를 잡자. 둘 다 `±3σ = ±24` 안에 들어간다. 즉 outlier가 아니라 완전히 평범한 draw다.

```
SCALING 없음:  {24, 16}에 대한 softmax
               p1 = 1/(1 + e^-8) = 0.99966,  p2 = 0.00034
               softmax Jacobian diagonal  p1(1-p1) = 3.3524e-4

SCALING 있음:  {3.0, 2.0}에 대한 softmax
               p1 = 1/(1 + e^-1) = 0.731059,  p2 = 0.268941
               p1(1-p1) = 0.196612

비율 = 0.196612 / 3.3524e-4 = 약 586배 더 많은 gradient가 score로 들어간다
```

Softmax Jacobian은 `∂p_i/∂s_j = p_i(δ_ij − p_j)`다. `p_i → 1`이 되면 **모든** entry가 0으로 간다. 그리고 이 Jacobian은 gradient가 `W_Q`와 `W_K`에 도달하는 *유일한* 경로다. 따라서 saturate된 head는 **어디를 볼지** 학습하기를 멈춘다 — random initialization이 우연히 만들어 준 routing에 그대로 얼어붙는다. Forward pass는 여전히 돌아간다. 그 head가 더 이상 학습 가능하지 않을 뿐이다.

💡 **쉬운 설명.** Softmax가 이미 "0.9997 대 0.0003"처럼 답을 확신해 버리면, 입력 score를 조금 흔들어도 출력 확률이 거의 변하지 않는다. 미분값이 곧 "입력을 흔들었을 때 출력이 얼마나 변하나"이므로, 확신 = 미분값 0이다. 그런데 `W_Q`와 `W_K`는 오직 이 미분값을 통해서만 학습 신호를 받는다. 그래서 scaling을 빼면 head가 "확신에 찬 채로 잘못 얼어붙는" 상태가 된다 — 성능이 나쁜 게 아니라 *고칠 수 없게* 된다.

**현실적인 N에서는 더 나쁘다.** `N`개의 i.i.d. `N(0,1)` score의 최댓값은 대략 `√(2 ln N)`이다.

```
N = 1,024   ->  3.72
N = 32,768  ->  4.56
```

`d_k = 128`에서 scaling 없이 계산하면 이 값들이 `11.3137 × 3.72 = 42.1`, `11.3137 × 4.56 = 51.6`이 된다 — softmax logit gap이 **수십 nat**이고, 이는 정확히 one-hot이다. Vaswani가 §3.2.1에서 서술한 regime이 바로 이것이다, 원문 그대로:

> "While for small values of d_k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_k. We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."
>
> (번역: `d_k`가 작으면 두 mechanism의 성능이 비슷하지만, `d_k`가 커지면 additive attention이 scaling 없는 dot product attention을 능가한다. `d_k`가 클 때 dot product의 크기가 커져 softmax를 gradient가 극도로 작은 영역으로 밀어 넣기 때문이라고 추정한다.)

**2026년에 중요한 단서 조항.** `Var = d_k` derivation은 unit-variance의 독립 component를 가정한다. 실제 Q와 K는 LayerNorm/RMSNorm 이후의 `X·W_Q`에서 나오므로, 이 가정은 (표준 `1/√d_model` weight init에서) *initialization 시점에 근사적으로* 성립하고 weight가 움직이면서 **training 중에 무너진다**. `1/√d_k`는 init 시점에 옳은 상수이며 이후 그대로 고정된다. 여러 modern model이 **QK-norm** — dot product 이전에 Q와 K에 적용하는 RMSNorm — 을 추가하는 이유가 정확히 이것이다. 상수가 가정하는 조건을 다시 강제하는 것이다. [[sqrt-dk-scaling-variance]] 참조.

> **▶ 인터랙티브 companion — [`figures/attention-step-by-step.html`](figures/attention-step-by-step.html) (패널 1–5)**
> *화면 위의 모든 중간 숫자가 `x`, `W_Q`, `W_K`, `W_V`로부터 page 안에서 직접 계산된다.* 이 page는 의도적으로 아주 작은 example 하나를 돌린다 — 6-token Korean 문장 **"나 는 어제 책 을 읽었다"**를 `N = 6`, `d_model = 4`, `d_head = d_k = d_v = 4`, `a = 1`, `√d_k = 2`, bf16으로 — 그리고 **패널 1**이 이를 config chip으로 늘어놓으면서 실제 model과 대조한다(`d_model = 4096`, head 32개, `s = 4096` → score tensor 하나가 `1,073,741,824` — figure의 산문은 이를 숫자라고 부르지만 그 양은 **byte**다. bf16 element로는 `536,870,912`개이고, 곧 이 chapter의 ATTN unit tensor 1.07 GB다).
> **패널 2**는 stepper다: 42 step에 걸친 8개 stage(embedding lookup → Q/K/V projection → `q·k` → `÷ √d_k` → causal mask → softmax → `Σ p·v` → `[B,a,s,s]` memory 질문)를 클릭 가능한 query-token selector(기본값 token 5, "읽었다")와 함께 진행하며, 모든 dot product를 단정하지 않고 4-row 표(operand A / operand B / 곱 / 합)로 풀어 쓴다. 거기서 나오는 row 5의 숫자들 — row max인 `score[5][3] = 3.2344`, `÷2 = 1.6172`, 합이 정확히 1인 softmax `0.2282, 0.0943, 0.1410, 0.3310, 0.0807, 0.1248`, output `o_5 = [1.0134, 0.5834, 0.4033, 0.2206]` — 이 곧 §2.3의 4단계 recipe를 손으로 실행한 것이다.
> **패널 3**은 그 stepper가 구동하는 live state view다: 왼쪽에 token별 `x/q/k/v`, 오른쪽에 raw → scaled → masked → probability로 진행하는 `6×6` matrix.
> **패널 4**는 §2.5 전체다. 왼쪽 canvas는 이론값 `sd = √d_k` 곡선을 `d_k = 1 … 512`에서 **Box–Muller sample 6,000개**(seed 20250818)와 겹쳐 그려서 "variance가 `d_k`와 같다"를 단정이 아니라 측정으로 보여주고(측정값 ≈ 11.3 대 `√128 = 11.3137`), 1.0 위치의 green line이 나눗셈이 복원해 주는 것을 표시한다. 오른쪽 column은 Vaswani footnote-4 derivation, `√d_k` 표(`4 → 2.0000`, `64 → 8.0000`, `128 → 11.3137`, `512 → 22.6274`), 그리고 빨간 **gradient-death** box를 담는다: `d_k = 64`에서 score `24` 대 `16`일 때 unscaled `p = 0.99966`이 Jacobian `p(1−p) = 3.3524e-4`를 주고 scaled(`3.0` 대 `2.0`)는 `0.196612`를 주는 — 그 **586×** 격차를 숫자로 렌더링한다. 그 아래에는 §2.4의 example 자체가 "Vaswani 검증 예제" 표로 들어 있으며, `q = [1,0,1,0]`과 세 key·세 value를 원문 그대로 써서 runtime에 계산해 scaled weight **`0.506480 / 0.186324 / 0.307196`**(합 `1.000000`)을 unscaled `0.665241 / 0.090031 / 0.244728`와 나란히, output `[0.813676, 0.493520]`, 그리고 `e`-값 `2.718282 / 1.000000 / 1.648721`(합 `5.367003`)을 인쇄한다 — 옆에는 row 5의 softmax를 초록/빨강으로 다시 그리는 scaled/unscaled toggle이 있다(`0.3310` → `0.5156`, `1.56×` 뾰족해짐).
> **패널 5**는 §2.2의 세 가지 실패 논증을 같은 grid 위 버튼 셋으로 만든 것이다: 제대로(`S = (XW_Q)(XW_K)ᵀ`, 비대칭 — `S[5][3] = 3.2344` 대 `S[3][5] = 0.2244`, `14.4×` 격차), `XXᵀ`(정확히 대칭 — `S[5][3] = S[3][5] = 0.28`), 그리고 RMSNorm 후 `XXᵀ`(diagonal이 `4.00 = d`에 고정, self-attention 확률 `0.5366` 대 uniform `0.1667`). 아래 고정 callout이 네 번째 경우 — `W_V`를 빼는 것 — 과 왜 `d_k`는 Q/K 사이에서 일치해야 하고 `d_v`는 자유로운지를 다룬다.

---

## 3. Causal Masking — 그리고 그것이 만들어 내는 invariant

### 3.1 Mask

Decoder-only language model은 token `i+1`을 `≤ i`인 token들로부터 예측한다. 지금까지 정의한 attention은 모든 position이 다른 모든 position을 — 미래까지 포함해서 — 볼 수 있게 하는데, 그러면 model이 정답을 읽어 버린다. 해결책은 softmax **이전에** 적용하는 additive matrix 하나다.

```
M_ij = 0      if j <= i
M_ij = -inf   if j > i

Attention_causal(Q, K, V) = softmax( (QK^T + M) / sqrt(d_k) ) · V
```

동등하게, `j ≤ i`이면 `S_ij = q_i·k_j/√d_k`, `j > i`이면 `−inf`다. `e^{−∞} = 0`이므로 이것은 soft penalty가 아니라 **hard하고 exact한** mask다. Softmax 이후 row `i`는 정확히 `i+1`개의 non-zero entry를 갖고 그 합은 여전히 1이다. Vaswani §3.2.3, 원문 그대로:

> "We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −inf) all values in the input of the softmax which correspond to illegal connections."
>
> (번역: auto-regressive 성질을 유지하려면 decoder에서 왼쪽 방향 정보 흐름을 막아야 한다. 우리는 이를 scaled dot-product attention 내부에서 구현하는데, softmax input 중 불법 연결에 해당하는 모든 값을 masking(−inf로 설정)한다.)

**Run을 조용히 망치는 off-by-one:** mask는 diagonal을 포함해야 한다 — `j < i`가 아니라 `j ≤ i`다. Token은 자기 자신에게 attend할 수 있어야 한다. `torch.triu(ones(N, N), diagonal=1)`이 정확히 불법 영역(diagonal 위쪽 strict한 부분)을 표시한다. `diagonal=0`이면 self-attention까지 추가로 금지해 버린다.

### 3.2 dtype 함정

Mask 값은 자유롭게 고를 수 있는 것이 아니다.

| dtype | `torch.finfo(dtype).min` | `-1e9`이 표현 가능한가? |
|---|---|---|
| float16 | −65504 | **아니오** — `-inf`로 cast됨 |
| bfloat16 | −3.3895e38 | 예 |
| float32 | −3.4028e38 | 예 |

올바른 code:

```python
attn = attn.masked_fill(~causal_mask, torch.finfo(attn.dtype).min)
```

Legacy GPT-2 / 초기 HuggingFace code는 `torch.where(causal_mask, attn, torch.tensor(-1e9, dtype=attn.dtype))`를 썼는데, `-1e9`이 fp16 범위를 넘기 때문에 **fp16에서 깨진다**.

그리고 왜 `finfo.min`이 literal `-inf`보다 나은가: 전부 mask된 row(padding row, 일부 sliding-window configuration)는 `Σ e^{−∞} = 0`이 되어 softmax가 `0/0 = NaN`을 반환하고, 그 NaN이 backward pass 전체로 전파된다. `finfo.min`은 대신 유한한 uniform row를 만든다 — 틀렸지만 해롭지는 않다. [[causal-mask-neg-inf]] 참조.

💡 **쉬운 설명.** `-inf`는 수학적으로는 깔끔하지만 컴퓨터에서는 위험하다. Row 전체가 `-inf`면 분자도 0, 분모도 0이 되어 `0/0 = NaN`이 나오고, NaN은 곱셈 한 번으로 전체 gradient를 오염시킨다. `finfo.min`(그 dtype이 표현할 수 있는 가장 작은 유한한 수)을 쓰면 exp 결과가 아주 작지만 0은 아니어서, 최악의 경우에도 "의미 없는 균등 분포"가 나오고 run은 죽지 않는다.

### 3.3 Invariant: 과거의 K와 V는 얼어 있다

이 chapter에서 가장 중요한 귀결이며, optimization이 아니라 *구조적* 사실이다.

```
k_j = W_K x_j        v_j = W_V x_j
```

`k_j`와 `v_j`는 token `j` **하나에만** 의존한다. 그리고 causal mask 아래에서 `x_j` — 몇 개의 block을 지났든 position `j`의 residual-stream state — 는 `≤ j`인 position에만 의존한다. 따라서:

> **Token `t+1`을 덧붙여도 `k_1 … k_t`나 `v_1 … v_t`는 아무것도 바뀌지 않는다.**

과거의 key와 value는 **immutable**하다. 여기서 거대한 결과 둘이 따라 나온다.

1. **Teacher forcing이 합법이다.** 한 번의 parallel forward pass로 계산한 `T`개의 prediction은 `T`번의 sequential autoregressive step과 *동일하다*. 이것이 transformer training을 sequence axis에 대해 병렬화할 수 있게 만드는 근거이며, §7.6이 "training에는 KV cache가 없다"고 말할 수 있는 이유다.
2. **KV cache가 valid하다.** Immutable한 값을 다시 계산하는 것은 순수한 낭비다. §7이 그 전체 회계다.

💡 **쉬운 설명.** 이 한 문장이 course의 두 축을 동시에 지탱한다. "과거는 미래를 보지 못한다"는 규칙 덕분에 (a) training에서는 전체 sequence를 한 번에 병렬로 밀어도 결과가 순차 생성과 같고, (b) inference에서는 이미 계산한 K/V를 재사용해도 결과가 같다. 같은 사실이 training 쪽에서는 "cache가 *필요 없다*"로, inference 쪽에서는 "cache가 *가능하다*"로 나타난다.

### 3.4 구조적으로 죽어 있는 matrix의 절반

Causal attention은 lower triangle만 있으면 된다 — 약 `N²/2`개 entry다. 그런데 dense kernel은 그럼에도 `N²` 전부를 저장하고 계산한다. Korthikanti 회계 기준으로 causal structure는 dense kernel의 `5·a·s²·b` 항을 **전혀** 줄이지 못한다. 오직 causal-aware kernel만이 이 구조적 sparsity를 실제 절약된 byte로 바꾼다. FlashAttention-2가 정확히 그 일을 한다: diagonal 위쪽 KV block을 통째로 건너뛰고(attention 계산의 약 50%), mask도 score matrix도 materialize하지 않으며, diagonal block은 SRAM tile 내부에서 mask한다. [[causal-mask-neg-inf]]의 표현대로: **`N×N` score matrix의 절반은 구조적으로 죽어 있고, 그 값을 지불할지 말지는 model이 아니라 kernel이 결정한다.** 이것이 [[ch-04]]의 thesis를 한 chapter 먼저 말한 것이다.

Mask 자체도 공짜가 아니다. `N = 32,768`에서 명시적인 `[1, 1, N, N]` tensor로 두면:

| representation | bytes | |
|---|---|---|
| bool (1 B/elem) | 1,073,741,824 | **1.07 GB** |
| bf16 additive | 2,147,483,648 | **2.15 GB** |
| fp32 additive | 4,294,967,296 | **4.29 GB** |

Run 내내 resident 상태이고, 모든 layer가 HBM에서 다시 읽는다.

**Out-of-place copy 함정**은 scaling과 masking 양쪽에 적용된다. Fuse되지 않은 `S_scaled = S / sqrt(d_k)`나 in-place가 아닌 `masked_fill`은 **두 번째** `(B, a, N, N)` tensor를 할당한다. `B=1, a=32, N=32768`, bf16에서 이는 추가로 `32 × 32768² × 2 = 68,719,476,736 B = layer당 68.7 GB`다 — underscore 하나가 빠져서 attention peak가 두 배가 된다. 올바른 실무: **query를 미리 scaling하라**(`Q *= d_k**-0.5`, `B·N·d_model` element만 든다). 그러면 `N×N` tensor가 처음부터 scaling된 상태로 생성된다. FlashAttention은 이 상수를 SRAM 내부의 Q tile에 접어 넣으므로, scaling된 score tensor가 HBM에 도달하는 일 자체가 없다.

💡 **쉬운 설명.** PyTorch에서 `x = x / 2`와 `x /= 2`는 memory 관점에서 완전히 다르다. 앞은 새 tensor를 만들고 뒤는 제자리에서 고친다. 평소엔 신경 쓸 필요 없는 차이지만, 그 tensor가 68.7 GB짜리면 이 한 글자가 OOM과 정상 실행을 가른다. 더 나은 해법은 아예 큰 tensor를 건드리지 않는 것 — `N×N`을 만든 뒤 나누지 말고, 훨씬 작은 Q(`N×d`)를 미리 나누면 수학은 같고 byte는 수만 배 싸다.

> **▶ 인터랙티브 companion — [`figures/attention-step-by-step.html`](figures/attention-step-by-step.html) (패널 2의 stage 5, 패널 3, 패널 6)**
> *additive matrix로서의 mask, 그리고 아무도 필요로 하지 않는 삼각형.* 패널-2 stepper의 **stage 5 · causal mask**는 `6×6` scaled score 표를 다시 그리되 strictly-upper cell을 `−∞`로 바꾸고, 살아남은 것을 센다 — **36개 중 21개 cell 생존**, 즉 `N² = 36`에 대한 `N(N+1)/2 = 21` — 그리고 올바른 PyTorch 관용구 `torch.finfo(attn.dtype).min`을 그것을 정당화하는 dtype 한계값과 함께 인쇄한다(`float16.min = -65504`, `bfloat16.min = -3.3895e38`, `float32.min = -3.4028e38`, 그리고 fp16에서는 `x > 11.09`에서 `exp`가 overflow한다는 주석). §3.1과 §3.2를 경고문이 아니라 렌더링된 숫자로 만든 것이다.
> **패널 3**은 같은 masking을 live grid 위의 state transition으로 보여준다 — mode가 `raw → scaled → maskedrow → masked → probrow → prob`로 진행하며 4-swatch legend(아직 계산 안 됨 / 방금 계산됨 / causal mask `−∞` / softmax 확률)가 붙어 있어서, upper triangle이 한 row씩 죽어 가는 것과 살아남은 row가 다시 1로 renormalize되는 것을 차례로 보게 된다.
> **패널 6**이 §3.4의 결실이다: 완성된 causal probability heatmap에 `[B, a, s, s] = [1, 1, 6, 6]`, `36 원소 × 2 B = 72 B`라는 caption이 붙고, 그 옆의 sequence-length selector(`s = 6 / 512 / 1,024 / 2,048 / 4,096 / 8,192 / 16,384 / 32,768`, 기본 4,096)가 card 네 개를 구동한다 — layer당 element 수, layer 하나의 byte, 80 layer의 byte, 그리고 toy 대비 비율(`s = 4096`에서 `1.49e+7×`). `s = 4096, a = 32`, bf16에서는 **layer당 1.07 GB → 80 layer에 걸쳐 85.90 GB**로 읽히고, `s = 32768`에서는 **layer당 68.72 GB**로 읽히는데 이것이 위의 out-of-place-copy 함정이 두 배로 만드는 바로 그 `68.7 GB`다. 마지막 빨간 callout이 softmax backward가 왜 `P`를 필요로 하는지(`dS = P ⊙ (dP − rowsum(dP ⊙ P))`), 그리고 그것을 recompute하는 것이 왜 싼지를 서술한다 — FlashAttention 논증을 한 chapter 먼저 말하는 것이다.

---

## 4. Multi-Head Attention

### 4.1 방정식

Vaswani §3.2.2, 원문 그대로:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
with W_i^Q ∈ R^(d_model × d_k), W_i^K ∈ R^(d_model × d_k),
     W_i^V ∈ R^(d_model × d_v), and W^O ∈ R^(h·d_v × d_model)
```

Vaswani는 head 수를 `h`로 쓰지만, 이 chapter는 `a`를 쓴다. `h`는 이미 `d_model`이기 때문이다. 이 block quote 안에서 — 그리고 §4.5의 Table-3 원문 config, §4.5 figure callout에서 그 figure 자체의 표기를 설명하는 문장 한 곳에서 — `h`는 논문의 head 수다. 이 셋이 §1 Conventions note가 열거한 예외 전부이고, chapter 본문의 서술에서 head 수는 **항상** `a`이며 `h = d_model = 4096`이다.

그리고 동기, 원문 그대로:

> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."
>
> (번역: multi-head attention은 model이 서로 다른 position의 서로 다른 representation subspace 정보에 동시에 attend할 수 있게 한다. attention head가 하나면 averaging이 이를 방해한다.)

> "Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."
>
> (번역: 각 head의 dimension이 줄어들기 때문에 전체 computational cost는 full dimensionality의 single-head attention과 비슷하다.)

첫 인용문이 *왜*에 해당한다. Head 하나는 position마다 **하나의** softmax-weighted average를 계산하는데, averaging은 여러 relation을 동시에 표현하는 능력을 파괴한다. Position `i`가 "내 절의 subject"와 "세 문장 뒤의 entity"를 둘 다 필요로 한다면, single head는 이 둘을 하나의 convex combination으로 뭉개고 양쪽을 다 잃어야 한다.

### 4.2 Tensor pipeline, 정확하게

방정식의 "Concat"은 실제 implementation에서 별도 tensor들을 이어 붙이는 것이 아니다. Head들은 처음부터 하나의 matrix에 fuse되어 있다.

```
X                       (B, N, d_model)
  @ W_Q / W_K / W_V     각각 (d_model, d_model)   [a개 head가 하나의 matrix로 FUSE됨]
Q, K, V                 (B, N, d_model)
  .view(B, N, a, d_head).transpose(1, 2)
Q, K, V                 (B, a, N, d_head)          d_head = d_model / a
  S = Q @ K^T / sqrt(d_head)
S                       (B, a, N, N)
  P = softmax(S, dim=-1)
P                       (B, a, N, N)
  O = P @ V
O                       (B, a, N, d_head)
  .transpose(1, 2).contiguous().view(B, N, a*d_head = d_model)   <- 이것이 "CONCAT"이다
  @ W_O                 (d_model, d_model)
out                     (B, N, d_model)
```

Concat은 별도 tensor의 copy가 아니라 **하나의 contiguous buffer에 대한 transpose + reshape**이다. 이 사실은 내재화할 가치가 있다. 왜 head 수가 compute 면에서 거의 공짜인지, 왜 모든 implementation에 `.contiguous()`가 등장하는지(transpose가 buffer를 non-contiguous로 만들고, `view`는 contiguity를 요구한다)를 설명해 주기 때문이다.

### 4.3 Head는 parameter에서 공짜이고 FLOP에서도 공짜다

**Parameter.** Attention block은 `d_model × d_model` matrix 네 개(`W_Q, W_K, W_V, W_O`)를 갖는다.

```
attention params = 4 · d_model²      -- a와 무관

d_model = 512   ->  4 × 512²  = 1,048,576    ≈ layer당 1.05M
d_model = 4096  ->  4 × 4096² = 67,108,864   ≈ layer당 67.1M
```

`a`를 8에서 32로 바꿔도 이 값은 **정확히 0만큼** 변한다.

**FLOPs.** Head 수가 상쇄된다.

```
QK^T 비용   2·B·a·N²·d_head = 2·B·N²·(a·d_head) = 2·B·N²·d_model
P·V  비용   동일
```

Head 수는 근사적으로가 아니라 **정확히 FLOP-invariant**하다.

### 4.4 그러나 head는 memory에서 공짜가 *아니다*

Score tensor는 `(B, a, N, N)`이다. bf16에서:

```
bytes = B · a · N² · 2      -- FLOP은 고정인데 a(HEAD 수)에 대해 LINEAR
```

이 `a`를 주의해서 읽어라. 이것은 hidden width가 아니라 head 수다. 여기서 `h = d_model`에 linear한 것은 아무것도 없다 — `h`는 score tensor의 shape에 아예 등장조차 하지 않으며, 이것이 바로 §8.2의 `5·a·s²·b` 항이 `h`가 아니라 `a`를 달고 있는 이유다.

Arithmetic intensity는 정확히 **저장 element당 `2·d_head` FLOPs(= bf16에서 저장 byte당 `d_head` FLOPs)**다 — `QKᵀ`는 §2.3의 multiply-add convention 아래 `2·B·a·N²·d_head` FLOPs가 들고 `B·a·N²`개 element를 쓴다. Reference config에서는 element당 `256` FLOPs, byte당 `128` FLOPs다. `a`를 두 배로 하면 `d_head`가 절반이 되므로 intensity도 절반이 되고 저장 byte는 동일한 compute에 대해 **두 배**가 된다. [[multi-head-split-concat-wo]]는 결론을 직설적으로 말한다: **multi-head attention은 FLOP-neutral하고 memory-expensive하다.**

💡 **쉬운 설명.** Arithmetic intensity는 "저장한 element(혹은 byte) 하나당 몇 번 계산에 써먹었나"이다. Head를 늘리면 총 계산량은 그대로인데 저장해야 할 score entry 수만 늘어나니, 같은 계산을 더 많은 memory에 흩뿌리는 셈이다. GPU 입장에서는 "일은 그대로인데 짐만 늘어난" 최악의 방향이다. 단위에 주의하라 — element당 `2·d_head`이고 bf16은 element가 2 byte이므로 byte당으로는 `d_head`가 된다. 이 chapter는 element 기준을 기본으로 쓴다.

구체적으로, 7B급 model(`d_model = 4096`, `a = 32`, `d_head = 128`, `L = 32`), bf16, `B = 1` — [[ch-04]] §5가 인용하는 것과 같은 row이며, 여기서는 단위를 명시했다(binary: MiB/GiB/TiB):

| N | head당 | layer당 | model 전체 |
|---|---|---|---|
| 2,048 | 8,388,608 B = 8 MiB | 268 MB | 8.6 GB |
| 8,192 | 134,217,728 B = 128 MiB | 4.3 GB | 137 GB |
| 32,768 | 2,147,483,648 B = 2 GiB | 68.7 GB | 2.2 TB |

(§1의 Convention note에 따른 unit 조정: `ch-04/read.md` §5 L211–216은 **head당** column을 `8 MB / 128 MB / 2 GB`로, **model 전체** column을 `8 GB / 128 GB / 2 TB`로 인쇄한다. 두 column 모두 binary 양에 decimal label을 붙인 것이다 — head당 column은 정확히 `8 MiB / 128 MiB / 2 GiB`로 이 표의 head당 column과 일치하고, model 전체 column은 정확히 `8 GiB / 128 GiB / 2 TiB`이며 decimal로 옮기면 여기 인쇄된 `8.6 GB / 137 GB / 2.2 TB`가 된다. 두 chapter 사이에 숫자는 하나도 다르지 않고 label만 다르다.)

`N = 32,768`, `d_model = 4096`, `B = 1`, bf16, layer당으로 Q, K, V 자체와 비교하면:

```
Q + K + V   = 3 × 32768 × 4096 × 2 B = 805,306,368 B  =  805 MB
              (Q, K, V 각각 = 268,435,456 B = 268 MB)
score tensor = 32 × 32768² × 2 B     = 68,719,476,736 B = 68.7 GB
비율                                                     = 85x
```

`N = 2048`에서는 이 비율이 `268 MB / 50.3 MB = 5.3×`에 불과하다. **격차는 N에 대해 linear하게 자란다** — 이 course에 attention kernel chapter가 셋이고 projection kernel chapter는 하나도 없는 이유가 전부 이것이다.

### 4.5 Head는 몇 개여야 하나? Ablation이 실제로 말하는 것

Vaswani Table 3, base row, arXiv v7 PDF p.9에서 확인:

```
N_layers = 6, d_model = 512, d_ff = 2048, h = 8, d_k = d_v = 64,
P_drop = 0.1, eps_ls = 0.1, 100K steps, dev PPL 4.92, dev BLEU 25.8, 65M params
```

(여기 `h = 8`은 §4.1의 note대로 Vaswani가 **head 수**를 가리키는 표기다 — 이 chapter의 기호로는 `a = 8`이고, 논문의 `d_model = 512`가 이 chapter의 `h`다.)

Row (A), head ablation — EN-DE newstest2013 dev; §4.3에 따라 **모든 row가 구조적으로 동일한 parameter 수를 갖는다**:

| heads `a` | `d_k = d_v` | PPL | BLEU |
|---|---|---|---|
| 1 | 512 | 5.29 | 24.9 |
| 4 | 128 | 5.00 | 25.5 |
| **8 (base)** | **64** | **4.92** | **25.8** |
| 16 | 32 | 4.91 | 25.8 |
| 32 | 16 | 5.01 | 25.4 |

논문의 요약, 원문 그대로: *"While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads."* (번역: single-head attention은 최적 설정보다 0.9 BLEU 낮지만, head가 너무 많아도 품질이 떨어진다.)

양쪽 끝이 모두 정보를 준다. Head가 너무 적으면 averaging이 multi-relation capacity를 파괴한다. 너무 많으면 `d_head`가 줄어들어 각 head의 subspace가 유용한 match를 표현하기에 너무 작아진다 — `d_model = 512`에서 `a = 32`이면 `d_head = 16`이다. Row (B)는 `d_k`만 바꿔서 이 두 번째 효과를 분리한다:

| `d_k` | PPL | BLEU | params |
|---|---|---|---|
| 16 | 5.16 | 25.1 | 58M |
| 32 | 5.01 | 25.4 | 60M |
| **64 (base)** | **4.92** | **25.8** | **65M** |

논문의 해석: *"determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial."* (번역: compatibility를 결정하는 일은 쉽지 않으며, dot product보다 정교한 compatibility function이 유익할 수 있다.)

**모든 head가 실제로 쓰이는가?** Voita et al. 2019는 WMT EN-RU에서 학습된 6-layer 8-head(총 48-head) transformer에 L0 gate를 걸어 pruning했다:

| 남긴 head | BLEU 변화 |
|---|---|
| 38/48 (79%) | −0.1 |
| 25/48 (52%) | −0.3 |
| 15/48 (31%) | −0.6 |
| 10/48 (21%) | −1.0 |

대략 **head의 60%가 BLEU 0.3 미만의 손실로 prunable**하며, 살아남는 것들은 불균형하게 *positional*("t−1에 attend")하거나 *syntactic*하다. (정직하게 표시하는 단서: 이 수치들은 `llm-arch` wiki에서 가져온 것이며 이 chapter를 위해 ACL 논문과 **재검증되지 않았다** — 정확한 소수점은 참고치로 다뤄라.) 흥미로운 것은 memory 관점의 독법이다. §4.4에서 보았듯 byte는 FLOP 고정인 채 head 수 `a`에 linear하게 scaling하므로, 아무 기여도 하지 않는 head도 자기 몫의 `N²` slice 값을 전액 지불한다.

> **▶ 인터랙티브 companion — [`figures/multihead-and-rope.html`](figures/multihead-and-rope.html) — Part A (A0–A3)**
> *Head 수가 어디서 상쇄되고 어디서 상쇄되지 않는가.* Part A는 모든 element를 인쇄할 수 있을 만큼 작은 toy를 돌린다. **A0**은 token `N = 4`("The / cat / loudly / ran"), `d_model = 8`, head `2`개, `d_head = 4`, `√d_head = 2`, **causal mask 없음**을 고정하고 `X`를 4×8 정수 표로 인쇄하는데 왼쪽 절반은 position one-hot, 오른쪽 절반은 content code다 — 그리고 `W_Q/W_K/W_V`는 학습된 weight가 아니라 손으로 고른 예시 값임을 빨간 callout이 먼저 못 박는다.
> **A1**은 그 buffer 위에서 §4.2를 12 step으로 애니메이션한다: `Q = XW_Q`(8×8 matrix 전체를 보여주고 element 하나를 손으로 전개) → `K`, `V` → **"view + transpose (복사 아님)"이라고 label된 split** → head별 `S = Q⁽ʰ⁾K⁽ʰ⁾ᵀ` → `/√d_head`와 softmax → `O⁽ʰ⁾ = P⁽ʰ⁾V⁽ʰ⁾` → **"Concat — 사실은 transpose + reshape"** → `@ W_O`, 이 지점이 *head가 처음으로 섞이는 유일한 자리*라고 표시된다. Softmax 산술이 전부 풀어 써져 있어서(`e³ = 20.085537`을 합 `23.085537`로 나눠 `0.870049`, head 1은 `e⁴·⁵ = 90.017131`을 `93.017131`로 나눠 `0.967748`), "Concat"이 label 바꾸기이고 `W_O`가 유일한 mixer임이 눈에 보인다.
> **A2**는 두 head의 softmax heatmap을 나란히 영구 배치한다 — head 0은 "직전 token을 본다"(positional), head 1은 "통사적으로 연결된 token을 본다"(syntactic) — Vaswani §3.2.2의 "averaging inhibits this"를 원문 그대로 인용한 아래에. §4.1의 *왜*를 그림으로 만든 것이다.
> **A3**은 불변성 ledger인데, reference config가 아니라 **toy** config 위에서 돈다는 점에 주의하라: head 수 `1, 2, 4, 8`에 대한 표(figure는 head 수를 Vaswani식으로 `h`라고 label하지만 이 chapter는 `a`라고 쓴다)에서 parameter 수는 **모든** row에서 `4·d_model² = 4·8² = 256`에, `QKᵀ` FLOP 수는 `2·N²·a·d_head = 256`에 고정된 채, score-tensor element 수는 `16 → 32 → 64 → 128`로(head 1개에서 8개로 갈 때 8×) 자라고 element당 arithmetic intensity `2·d_head`는 `16 → 8 → 4 → 2`로 정확히 같은 비율로 떨어진다. §4.3–§4.4와 같은 결론을, 6,700만 대신 숫자 네 개로. 이어지는 파란 callout이 그것을 reference config와 Korthikanti로 옮긴다: `5as/h = 5·32·4096/4096 = 160`, `s²` 항이 나머지 전부의 **4.7×**, 그리고 crossover `s = 34h/(5a) = 870.4` token — §8.3이 형식화하는 handoff다. (`N = 32,768`에서의 `85×` Q+K+V-대-score 비교는 이 chapter §4.4의 산술이며, 이 figure에는 렌더링되지 *않는다*.)

---

## 5. Positional 정보

### 5.1 Attention은 token 순서를 전혀 모른다

이것은 heuristic이 아니라 증명 가능한 사실이다. row가 token인 `X ∈ ℝ^{T×d}`와 permutation matrix `P ∈ {0,1}^{T×T}`, `(PX)_i = X_{π(i)}`, `PᵀP = I`를 두자.

```
(1)  (PX)W_Q = P(XW_Q) = PQ,  K, V도 동일
     -- projection은 row 단위로 작동하므로 row shuffle과 교환된다

(2)  S' = (PQ)(PK)^T / sqrt(d_k) = P (QK^T) P^T / sqrt(d_k) = P S P^T
     -- 즉 S'_ij = S_{pi(i) pi(j)}

(3)  softmax(P S P^T) = P · softmax(S) · P^T = P A P^T
     -- row 단위 분모 sum_j exp(.)가 permutation-INVARIANT한 합이기 때문

(4)  Out' = (P A P^T)(P V) = P A (P^T P) V = P A V = P · Out          QED
```

따라서 `Attn(PX) = P · Attn(X)`이다. Attention은 permutation-**equivariant**이지 invariant가 아니다. Output들은 `π`를 *따라* 움직인다. Invariant한 것은 output의 multiset이다. Step (4)가 핵심이다 — conjugate된 score matrix가 만들어 낸 `Pᵀ`가 `V` 앞의 `P`와 상쇄된다.

이 증명은 block 전체로 확장된다. RMSNorm/LayerNorm과 FFN은 row마다 동일하고 독립적으로 작동하며 residual add는 elementwise이므로 `Block(PX) = P·Block(X)`이고, 이를 쌓아도 성질이 보존된다. **Pre-norm transformer stack 전체가 permutation-equivariant하다.** 명시적인 position signal이 없으면 "개가 사람을 물었다"와 "사람이 개를 물었다"는 row만 뒤바뀐 동일한 계산이다.

💡 **쉬운 설명.** Attention은 token을 "순서 있는 줄"이 아니라 "이름표 없는 자루"로 본다. 자루를 흔들어 순서를 바꾸면 출력도 똑같이 흔들려 나올 뿐, 계산 내용은 하나도 안 바뀐다(= equivariant). 그래서 "개가 사람을 물었다"와 "사람이 개를 물었다"를 구별하려면 position 정보를 **입력에 직접 넣어 주는 수밖에** 없다. §5.4가 그것을 어디에 넣느냐가 memory를 결정한다고 말하는 이유다.

**중요한 단서 하나.** 이 증명은 `P M Pᵀ = M`을 요구한다. Lower-triangular causal mask는 `π = id`일 때만 이를 만족하므로, decoder-only LM은 positional encoding이 전혀 없어도 **permutation-equivariant가 아니다** — token `i`는 정확히 `i+1`개의 선행 token을 보며, 이는 쓸 만한 absolute-position count다. Haviv et al.(Findings of EMNLP 2022, arXiv:2203.16634)은 NoPE decoder LM이 경쟁력 있음을 보이고 암묵적 absolute position을 probing으로 끄집어낸다. Kazemnejad et al.(NeurIPS 2023, arXiv:2305.19466)은 length generalization에서 NoPE가 명시적 PE를 능가할 수도 있음을 발견한다. **Equivariance 증명은 bidirectional/encoder attention에 대해서는 exact**하고 decoder에 대해서는 정신적으로만 근사적이다. [[attention-permutation-equivariance]] 참조.

### 5.2 Sinusoidal absolute encoding (2017)

`pos`가 token index이고 `i`가 **pair** index(`i ∈ {0, …, d/2−1}`, pair `i`가 dimension `2i`와 `2i+1`을 차지)일 때:

```
PE_(pos, 2i)   = sin( pos / 10000^{2i/d} )
PE_(pos, 2i+1) = cos( pos / 10000^{2i/d} )
```

Frequency `ω_i = 10000^{−2i/d}`, wavelength `λ_i = 2π/ω_i`, 단위는 token position이다. 이것은 기하급수적으로 간격이 벌어지는 rate를 가진 **d/2개의 시계 뭉치**다. `d_model = 512`에서:

| pair `i` | dims | `2i/d` | `ω_i` | `λ_i` (position) |
|---|---|---|---|---|
| 0 | (0,1) | 0.0000 | 1.0 | 6.28 |
| 32 | (64,65) | 0.1250 | 0.31623 | 19.87 |
| 64 | (128,129) | 0.2500 | 0.1 | 62.83 |
| 128 | (256,257) | 0.5000 | 0.01 | 628.3 |
| 192 | (384,385) | 0.7500 | 0.001 | 6,283.2 |
| 255 | (510,511) | 0.9961 | 1.0366×10⁻⁴ | 60,611.5 |

앞으로 가져갈 만한 교정 두 가지. `llm-arch` wiki가 둘 다 틀렸기 때문이다. 거기의 사다리는 **한 칸 잘못 label되어 있고**(dims (64,65)에 `ω = 0.1`을 찍어 놨는데 `ω = 0.1`은 dims (128,129)의 것이다), 그 figure의 "dim 128–129 λ = 56"은 **λ = 62.83**이어야 한다. 또한 가장 느린 wavelength는 62,832가 아니라 **60,611.5**다. `2π × 10000 = 62,831.85`는 exponent가 정확히 1.0일 때의 wavelength인데, 실제 pair가 도달하는 최대 exponent는 `(d−2)/d = 510/512 = 0.9961`이다. `d_head = 128`에서 같은 산술을 하면: 가장 느린 pair `i = 63`이 `2i/d = 126/128 = 0.9844`, `θ = 1.1548×10⁻⁴`, `λ = 54,410.1` position이다.

**중요했던 성질.** Position을 `k`만큼 shift하는 것은 그 matrix가 `pos`에 의존하지 않는 rotation이다.

```
[ PE_(pos+k, 2i)   ]   [  cos ω_i k   sin ω_i k ] [ PE_(pos, 2i)   ]
[ PE_(pos+k, 2i+1) ] = [ -sin ω_i k   cos ω_i k ] [ PE_(pos, 2i+1) ]

전체 vector:  PE_{pos+k} = diag(R_k^{(0)}, ..., R_k^{(d/2-1)}) · PE_pos
```

Relative position은 이미 2017년에 **rotation**이었다 — RoPE의 기여는 rotation schedule이 아니라 그 rotation을 *어디에* 적용하느냐다. [[sinusoidal-absolute-encoding]] 참조.

관련된 성질 하나: inner product가 거리에 따라 감소한다. `PE_pos · PE_{pos+k} = Σ_{i=0}^{d/2−1} cos(ω_i k)`이며, `d = 128`(따라서 `d/2 = 64`)에서:

| `k` | 0 | 1 | 10 | 100 | 1000 | 8000 |
|---|---|---|---|---|---|---|
| dot | 64.000 | 62.094 | 42.820 | 30.543 | 10.178 | −0.353 |

### 5.3 RoPE: X에 더하지 말고 Q와 K를 rotate하라

RoPE([[rope-rotary-position-embedding]])는 *같은* frequency 사다리를 쓴다 — `θ_i = base^{−2i/d}`, pair index `i ∈ {0, …, d/2−1}`, `d = d_head` — 그리고 `base = 10000`(Llama 1/2, Mistral, Qwen2), `base = 500000`(Llama 3), long-context finetune에서는 10⁶까지 간다. 혁신은 embedding에 더하는 대신 **Q와 K에 곱한다**는 점이다.

Pair `i`마다, position `m`에서:

```
[ q'_{2i}   ]   [ cos m·theta_i   -sin m·theta_i ] [ q_{2i}   ]
[ q'_{2i+1} ] = [ sin m·theta_i    cos m·theta_i ] [ q_{2i+1} ]
```

모든 pair에 대해 쌓으면 이것은 `d/2`개의 독립적인 2×2 block을 가진 block-diagonal orthogonal matrix `R_{Θ,m} ∈ ℝ^{d×d}`다. **Norm을 보존한다**: `‖R_m q‖ = ‖q‖`가 정확히 성립한다(검증: 차이 = 0.0).

**핵심 identity.** `Rᵀ_m = R_{−m}`과 `R_{−m}R_n = R_{n−m}`을 쓰면:

```
(R_{Theta,m} q)^T (R_{Theta,n} k) = q^T R_{Theta,m}^T R_{Theta,n} k = q^T R_{Theta,n-m} k
```

Absolute `m`과 `n`이 사라지고 **`n − m`만 살아남는다**. `d = 8`, `base = 10000`, random `q, k`, `(m,n) = (17,5)`에서 수치로 검증:

```
(R_m q)·(R_n k)                                  = 3.6169521324913054
q^T R_{n-m} k                                    = 3.616952132491305
Re[ sum_i q_i · conj(k_i) · e^{i(m-n)theta_i} ]  = 3.6169521324913045
```

세 가지 경로, 하나의 숫자. Pair별 closed form은 `Δ = m − n`으로 두면:

```
<R_m q^(i), R_n k^(i)> = (q_{2i}k_{2i} + q_{2i+1}k_{2i+1})·cos(Delta·theta_i)
                       + (q_{2i}k_{2i+1} - q_{2i+1}k_{2i})·sin(Delta·theta_i)
```

**부호에 주의하라**: `sin` coefficient는 `(q_{2i}k_{2i+1} − q_{2i+1}k_{2i})`, 즉 그 pair의 2D **cross product / determinant**이며 — `(q_{2i+1}k_{2i} − q_{2i}k_{2i+1})`이 *아니다*. Relative angle은 pair의 dot product(cos 항)와 cross product(sin 항)를 섞는다. 복소수 형태가 구조를 가장 선명하게 보여준다:

```
<f(q,m), f(k,n)> = Re[ sum_i q_i · conj(k_i) · e^{i(m-n)theta_i} ]
                 = sum_i |q_i||k_i| cos( angle(q_i) - angle(k_i) + (m-n)theta_i )
```

여기서 `q_i = q_{2i} + i·q_{2i+1}`가 연속된 real dimension을 `ℂ^{d/2}`로 pairing한다.

**최소 worked example.** Dimension pair 하나, `q = k = (1, 0)`, `θ = 1.0` rad:

| `(m, n)` | `Δ = m − n` | dot product |
|---|---|---|
| (5, 3) | 2 | −0.416147 |
| (7, 5) | 2 | −0.416147 |
| (100, 98) | 2 | −0.416147 |

전부 동일하다. `cos(2 rad) = −0.416147`이고 `Δ`만 들어가기 때문이다. "relative position"이 operational하게 의미하는 바가 이것이다: 간격이 같기만 하면 absolute index 100과 absolute index 5는 score 입장에서 구별되지 않는다.

💡 **쉬운 설명.** RoPE의 트릭은 "position을 벡터에 더하지 말고 벡터를 position만큼 회전시켜라"이다. 회전은 길이를 안 바꾸므로 vector의 크기 정보가 훼손되지 않고, 두 회전된 vector의 내적은 **각도 차이**만 남긴다 — 시곗바늘 두 개의 상대 각도는 몇 시에 봤든 벌어진 정도만 같으면 같은 것과 같은 이치다. 그래서 position 5와 3의 관계가 position 100과 98의 관계와 수치적으로 완전히 동일해진다.

**왜 이 형태여야 하고 다른 것은 안 되는가.** 세 가지 제약을 걸자. (1) relative 의존성 `⟨f(q,m), f(k,n)⟩ = g(q,k,m−n)`, (2) 원점에서의 identity `f(x,0) = x`, (3) magnitude 보존. `f(q,m) = R_f(q,m)·e^{iΘ_f(q,m)}`로 polar decomposition하자. `m = n`으로 두고 `f(x,0) = x`를 적용하면 모든 `m`에 대해 `R_f(q,m) = |q|`가 강제된다 — **position은 rotate만 할 수 있고 scale은 절대 할 수 없다**. Phase 제약 `Θ_f(q,m) − Θ_f(k,n) = Θ_g(q,k,m−n)`은 `Θ_f(q,m) = Θ(q) + mθ`를 강제한다. `φ(m) − φ(n) = h(m−n)`을 만족하는 연속 함수 `φ`는 linear한 것뿐이기 때문이다. 따라서 `f(q,m) = q·e^{imθ}`다. RoPE는 여러 선택지 중 하나가 아니라 저 세 요구사항의 **유일한** 해다.

**Frequency band, 그리고 long-context 확장이 작동하는 이유.** `d_head = 128`, `base = 10000`, training length `L = 8192`에서 `r_i = L/λ_i = L·θ_i/2π`, 즉 training 동안 완료한 full rotation 수를 정의하자:

| pair `i` | `θ_i` | `λ_i` | `r_i` | band |
|---|---|---|---|---|
| 0 | 1.0 | 6.28 | 1303.8 | fast / local |
| 16 | 1.0×10⁻¹ | 62.8 | 130.4 | fast |
| 32 | 1.0×10⁻² | 628.3 | 13.0 | mid |
| 48 | 1.0×10⁻³ | 6,283.2 | 1.30 | slow |
| 63 | 1.1548×10⁻⁴ | 54,410.1 | 0.151 | slow / global — 한 바퀴도 못 돈다 |

마지막 row가 naive extrapolation이 실패하는 이유다. Pair 63은 training 동안 한 cycle의 15%만 보았으므로, `L`을 넘어가는 position은 말 그대로 한 번도 관측한 적 없는 각도에 떨어진다. YaRN의 해법은 band-selective하다 — fast band는 그대로 두고 slow band만 interpolate한다. `d_head = 128`, `L = 8192`에서 `r > β = 32`는 손대지 않고 `r < α = 1`은 완전히 interpolate하는 cutoff를 쓰면 경계가 pair `i ≈ 25.76`과 `i ≈ 49.84`에 떨어진다: **pair 0–25는 그대로, pair 26–49는 부드러운 ramp** `θ'_i = θ_i(1−γ_i) + (θ_i/s)·γ_i` 위에, **pair 50–63은 완전히 `θ_i/s`로 scaling**된다. NTK-aware 대안은 대신 base를 rescaling한다: `base' = base · s^{d/(d−2)}`. `d = 128`에서 exponent는 `1.01587`이므로 `s=4 → 40,890`, `s=8 → 82,685`, `s=16 → 167,199`, `s=32 → 338,097`이다. Llama 3의 `base = 500000`은 `d_head = 128`에서 가장 느린 pair가 `θ_63 = 2.4551×10⁻⁶`, `λ = 2,559,195.5` position(base 10000의 54,410.1 대비)이 되는데 — 이것이 base를 올리는 것이 균일하게 적용된 조잡한 형태의 interpolation인 이유다.

💡 **쉬운 설명.** Frequency 사다리를 "빠른 시계 ~ 느린 시계"로 보면 이해가 쉽다. 빠른 시계는 8k context 동안 1300바퀴를 돌아서 "가까운 거리"를 잘 구분하고, 가장 느린 시계는 0.151바퀴밖에 못 돌아 "먼 거리"를 표현하려다 만다. Context를 늘렸을 때 망가지는 건 느린 시계뿐이다 — 처음 보는 각도로 넘어가니까. YaRN은 그래서 빠른 시계는 건드리지 않고 느린 시계만 늦춘다. Base를 그냥 키우는 방법은 모든 시계를 똑같이 늦추는 거라 더 거칠다.

**Implementation, 그리고 model을 조용히 파괴하는 함정.** HuggingFace / GPT-NeoX style:

```python
q_rot = q * cos + rotate_half(q) * sin
# rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])
```

원래의 RoFormer/GPT-J는 dimension `(2i, 2i+1)`을 pairing한다 — **interleaved**. GPT-NeoX / HF-Llama는 `(i, i+d/2)`를 pairing한다 — **split-half**. **Q와 K가 같은 convention을 쓸 때만** 둘은 수학적으로 동등하다(같은 `d/2`개 평면을 다시 label한 것). 섞어 쓰면 error도 NaN도 없이 model이 조용히 파괴된다. Runtime overhead는 forward pass의 약 1–3%다.

### 5.4 Positional encoding의 memory contract — 주입 지점 세 곳

이 부분이 §5를 course의 나머지와 연결한다. Positional scheme들은 **어디에** 주입하느냐로 갈리고, 그것이 `T×T` tensor를 건드리는지를 결정한다.

| 주입 지점 | scheme | 할당하는 object | `T×T` score matrix를 건드리는가? |
|---|---|---|---|
| input-embedding-additive | sinusoidal, learned | `[L, d_model]` table | **아니오** |
| Q/K-multiplicative | RoPE, iRoPE | `[L, d_head/2]` cos/sin cache | **아니오** |
| attention-logit-additive | T5 RPE, ALiBi | `[H, T, T]` bias | **예** |

💡 **쉬운 설명.** 이 표 하나가 §5 전체의 memory 결론이다. Position 정보를 **입력 벡터에 더하거나**(sinusoidal/learned) **Q·K를 회전시켜**(RoPE) 넣으면 `N×N` tensor는 그대로다. 그러나 **score에 직접 더하면**(ALiBi/T5 RPE) `[B,H,T,T]` 크기의 bias를 만들어야 하는데, 그건 FlashAttention이 애초에 만들지 않으려고 존재하는 바로 그 tensor다. 즉 positional encoding 선택은 "품질" 문제로만 보이지만 실제로는 kernel 선택 문제다.

**RoPE의 cos/sin cache**는 각각 `[L, d_head/2]` shape이고, byte는 `2·L·(d_head/2)·bytes_per_elem`이다. `d_head = 128`, fp32에서:

| `L` | cache | HF 스타일로 `[L, d_head]`까지 복제 |
|---|---|---|
| 8,192 | 4,194,304 B = 4.00 MiB | 8.00 MiB |
| 32,768 | 16.00 MiB | 32.00 MiB |
| 131,072 | 64.00 MiB | 128.00 MiB |

**한 번** 할당되고 모든 layer, 모든 head가 공유한다. 학습 parameter 0, gradient 0, optimizer state 0이다.

**학습되는 absolute PE table** `W_p ∈ ℝ^{L×d}`는 trainable하므로 [[ch-01]] §1.3의 mixed-precision AdamW 세금을 전액 낸다 — 16 B/param(fp32 master + fp32 moment 2개 + fp32 grad), bf16 working copy까지 세면 18 B/param:

```
GPT-3 (L=2048, d=12288):  25,165,824 params x 16 B = 402,653,184 B = 384.00 MiB   (18 B/param이면 432.00 MiB)
L=8192, d=4096:           33,554,432 params x 16 B = 536,870,912 B = 512.00 MiB
```

여기에 hard ceiling이 붙는다: **position `L+1`은 존재하지 않는다.** Sinusoidal은 fp32에서 `4·L·d` byte의 non-trainable buffer다 — `L=8192, d=4096`이면 `134,217,728 B = 128.00 MiB`, optimizer state 0, gradient 0이다. Sinusoidal도 learned PE도 *activation* byte는 추가하지 않는다. PE는 layer 0 이전에 `x`에 합산되고 어떤 tensor도 커지지 않기 때문이다.

**Logit-additive bias가 비싼 계열이다.** ALiBi는 `attention(q_i, k_j) = q_iᵀk_j/√d_k − m_h·|i−j|`를 계산하며 head별 slope `m_h = 2^{−8h/H}`를 쓴다(head 8개면: 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256). 우아하고 extrapolation도 잘된다 — 그러나 `B·H·T·T` object가 필요하다. 즉 **FlashAttention이 절대 materialize하지 않으려고 존재하는 바로 그 tensor**다:

| config | bytes | |
|---|---|---|
| B=1, H=32, T=4096, bf16 | 1,073,741,824 | **layer당 1.000 GiB** |
| B=1, H=32, T=8192, bf16 | 4,294,967,296 | **layer당 4.000 GiB** (A100-40GB의 10%) |
| B=4, H=32, T=8192, bf16 | — | **layer당 16.000 GiB** |
| B=1, H=32, T=8192, fp32 | — | **layer당 8.000 GiB** |

FlashAttention-2에 명시적인 `alibi_slopes` argument가 있는 이유가 정확히 이것이다. Kernel support가 없는 *custom* bias는 SDPA MATH path로 fallback하며 full score matrix를 다시 materialize한다 — [[ch-06]]의 silent-fallback 함정을, mask가 아니라 positional encoding을 통해 밟는 것이다.

**RoPE의 구조적 이점, 정확히 서술하면:** RoPE는 attention kernel **바깥에** 산다. Q와 K는 kernel 호출 *이전에* rotate되므로, FlashAttention/SDPA는 평범한 `(Q, K, V)` triple을 보고 `O(T)` activation footprint를 유지한다. RoPE는 parameter와 kernel memory contract 양쪽에서 공짜인 유일한 positional scheme이다. 지켜볼 transient 하나: **fuse되지 않은** RoPE는 `q_rot`/`k_rot`용으로 새 `[B, H, T, d_head]` tensor를 할당한다 — `B=1, H=32, T=8192, d_head=128`, bf16에서 각각 `67,108,864 B = 64.00 MiB`이므로, rotation 이전 copy를 kernel launch 전에 free하지 않으면 layer당 약 128.00 MiB의 피할 수 있는 transient가 생긴다. **Fuse된** RoPE는 backward-저장 activation을 **0** 추가한다. 그 backward가 `−mθ`만큼의 rotation, 즉 상수 coefficient를 갖는 orthogonal map이라 저장된 input tensor가 필요 없기 때문이다.

> **▶ 인터랙티브 companion — [`figures/multihead-and-rope.html`](figures/multihead-and-rope.html) — Part B (B1–B5)**
> *rotation으로서의 position, 그리고 그 rotation을 어디에 적용하는가.* **B1**은 §5.1의 증명을 선택 가능한 permutation 네 개(`[0,1,2,3]`, `[2,0,3,1]`, `[3,2,1,0]`, `[1,3,0,2]`)와 함께 5-step 애니메이션으로 돌린다: `X`를 `PX`로 섞은 뒤 `S⁽¹⁾`, `P S Pᵀ`, 그리고 *다시 계산한* `S(PX)`를 세 개의 matrix로 보여주면서 live `max |P S Pᵀ − S(PX)|`를 **정확히 0**으로 인쇄하고(toy가 정수 산술이라서), 이어서 `Out'`을 `P·Out`과 대조해 `max |Out' − P·Out|`가 `~1e−16`임을 보이며 왜 문자 그대로 0이 아닌지를 machine epsilon으로 설명한다. 마지막 step은 이 chapter가 §5.1에서 말하는 단서 — causal mask가 `P M Pᵀ = M`을 깬다 — 를 Haviv et al. 2022와 Kazemnejad et al. 2023을 인용해 서술한다.
> **B2**는 `d = 8` toy에서 RoPE pair 하나를 손으로 회전시킨다: position `m`(0–32)과 pair index `i`(0–3) slider, 원래 vector·회전된 `R_m q`·쓸고 지나간 호를 그리는 canvas, 그리고 같은 `m`에서 각 pair가 얼마나 돌았는지 보여주는 시계 다이얼 네 개. `θ = 1.0 / 0.1 / 0.01 / 0.001`, wavelength `6.28 / 62.83 / 628.32 / 6283.19` position — pair 0은 6.28 position마다 한 바퀴를 도는데 pair 3은 32에서도 `1.8°`밖에 못 움직였다. Live `‖q‖` 표시값이 §5.3의 norm 보존 주장(차이 ~0)을 확인해 준다.
> **B3**이 중심이다: `⟨R_m q, R_n k⟩`를 `m`과 `n` 독립 slider(0–64, 기본값 `m = 17, n = 5`로 §5.3이 수치 검증한 바로 그 쌍) 위에 놓고, 둘을 함께 미는 **Δ-lock 버튼**과 `Δ`에 대한 inner product plot을 `[−64, 64]` 구간에 그린다. 표 두 개가 논증을 나른다: pair별 분해(`θᵢ`, 그 pair의 dot product, cross product, `Δθᵢ`, `cos`, `sin`, 기여분, 그리고 `Σ` row)가 absolute position이 정확히 어디서 상쇄되는지 보여주고, "같은 `Δ`, 다른 `(m, n)`" 표에서는 여섯 row에 걸쳐 `m·θ₀`와 `n·θ₀`가 제각각인데 마지막 column만 **소수점 9자리까지 동일**하다. 닫는 callout이 memory 논증이다: RoPE의 cos/sin cache는 `L = 8192, d_head = 128`에서 `[L, d_head/2] × 2` fp32 = **4.00 MiB**에 학습 parameter 0, gradient 0, optimizer state 0인 반면, logit-additive `[B, H, T, T]` bias는 `B=1, H=32, T=8192` bf16에서 **layer당 4.000 GiB**, 즉 A100-40GB의 10%다 — §5.4의 표를 숫자 두 개로 만든 것이다.
> **B4**는 §5.3의 frequency 사다리를 canvas로 그린다: `d_head = 128`에서 pair index `0–63`에 대한 `log₁₀ θᵢ`를 `base = 10000`과 `base = 500000` **양쪽** 모두에 대해, YaRN의 `r = 32`(β)와 `r = 1`(α) 경계를 점선으로, α 오른쪽 영역을 빨갛게 음영 처리하고 "이 구간은 L=8192 동안 1회전도 못 한다"라고 label한다. 표는 §5.3의 band 표를 그대로 재현하고(`i = 0`은 `θ = 1.0`, `λ = 6.2832`, `1303.7973`회전, `i = 63`은 `θ = 1.1548e-4`, `λ = 54,410.1431`, `0.1506`회전), callout이 가장 느린 wavelength가 널리 인용되는 `2π × 10000 = 62,831.85`가 아니라 **54,410.1**임을 못 박는다. 실제 pair가 도달하는 최대 exponent가 `(d−2)/d = 126/128`이기 때문이다.
> **B5**는 long-context 결과를 담은 정적 패널이다: 어떤 pair가 본 적 없는 각도로 extrapolate하는지(`r < 1`, 즉 `d_head = 128, L = 8192`에서 `i ≥ 50`), YaRN의 세 band(`0–25`는 그대로, `26–49`는 `θ'ᵢ = θᵢ(1−γᵢ) + (θᵢ/s)γᵢ` ramp 위, `50–63`은 완전히 scaling)와 경계 `i ≈ 25.76` / `i ≈ 49.84`, NTK-aware `base' = base·s^{1.01587}` 값들(`s = 4 → 40,890` … `s = 32 → 338,097`), 그리고 Llama 3의 `base = 500000`을 조잡한 균일 interpolation으로 제시한다.

---

## 6. Transformer Block

### 6.1 Pre-LN vs Post-LN

두 update rule, 원문 그대로:

```
Post-LN (Transformer 2017, GPT-1):
    h'  = LayerNorm(x  + Attn(x))
    h'' = LayerNorm(h' + FFN(h'))

Pre-LN (GPT-2 이후):
    h'  = x  + Attn(LayerNorm(x))
    h'' = h' + FFN(LayerNorm(h'))
```

차이는 normalization이 residual highway **위에** 있느냐 residual branch **안에** 있느냐다. Pre-LN은 highway 위에 깨끗한 identity를 남긴다:

```
d x_{l+1} / d x_l = I + d Sub(LN(x_l)) / d x_l
```

따라서 backward에는 항상 정확히 `I`를 곱하는 경로가 있다. Post-LN은 backward pass를 `L`개의 LayerNorm Jacobian을 직렬로 통과하게 만든다.

**Xiong et al. 2020이 실제로 증명한 것**을, 그들 자신의 변수로 서술하면(Theorem 1과 2, 둘 다 initialization 시점, 둘 다 **마지막** layer의 FFN parameter에 대한 것, 둘 다 layer index `ℓ`이 *아니라* 총 depth `L`에 대한 것):

```
Post-LN:  E‖ ∂L/∂W^{(L)} ‖_F  =  O( d · sqrt(ln d) )            -- L과 무관
Pre-LN :  E‖ ∂L/∂W^{(L)} ‖_F  =  O( d · sqrt(ln d / L) )        -- 1/sqrt(L)로 감쇠
```

즉 `1/√·` 인자는 **Pre-LN** 쪽에 붙고, 그 안의 인자는 layer index가 아니라 **총 depth `L`**이다. 작동하는 사실은 절대적 크기가 아니라 *depth에 걸친 분포*다. Post-LN에서는 output 쪽 끝의 gradient가 `L`에 대해 `O(1)`로 유지되는 반면 input 쪽 끝에 도달하는 gradient는 직렬로 놓인 `L`개의 LayerNorm Jacobian에 의해 감쇠하므로, 하나의 global learning rate가 위에서는 너무 크고 아래에서는 너무 작은 일이 동시에 일어난다. Pre-LN의 layer별 gradient는 저 `1/√L` scale까지는 depth에 균일하므로 하나의 learning rate가 어디서나 well-posed하다. (*depth 불균형 진술은 여기서 정성적으로만 제시한다* — 위 두 scaling이 논문의 형식적 결과이고, "첫-대-마지막 불균형이 `O(L)`"는 그들의 Figure 1/3 측정에 대한 요약이지 theorem이 아니다. 이 chapter는 그 exponent에 의존하지 않는다.) 그 불균형 때문에 원 논문은 **4,000 warmup step**이 필요했다. Pre-LN은 warmup을 선택 사항으로 만든다. 참고로 `figures/transformer-block-dataflow.html` 패널 5는 Post-LN callout에 여전히 느슨한 "`O(1/√ℓ)`" 표현을 달고 있다 — 위의 두 scaling을 authority로 읽어라.

💡 **쉬운 설명.** Residual highway를 "고속도로"라고 하면, Pre-LN은 톨게이트(LayerNorm)를 진입 램프에 두고 Post-LN은 본선 한가운데에 둔다. Backward는 이 고속도로를 역주행하는데, Pre-LN에서는 톨게이트를 안 거치는 직통 차선(`I` 항)이 항상 열려 있어 gradient가 80개 layer를 지나도 균일하게 흐른다. Post-LN에서는 layer마다 톨게이트를 통과해야 해서 **위쪽(output 근처)과 아래쪽(input 근처)의 신호 세기가 벌어지고**, learning rate 하나로 양쪽을 동시에 맞출 수 없어 warmup 4,000 step이 필요했던 것이다. 흔히 인용되는 "`1/√ℓ`로 줄어든다"는 표현은 출처를 잘못 옮긴 것이다 — `1/√·`는 Pre-LN 쪽에 붙고 그 안은 layer index가 아니라 총 depth `L`이다.

Norm 자체는 이렇다:

```
LN(x)      = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
             mu = (1/d) sum_i x_i,  sigma^2 = (1/d) sum_i (x_i - mu)^2,  eps ~ 1e-5
             d_model에 대해 reduce, position마다, example마다

RMSNorm(x) = gamma * x / sqrt( (1/d) sum_i x_i^2 + eps )
             mean centering 없음, beta bias 없음, GPU에서 약 10% 빠름, 경험적으로 품질 동등
```

Pre-RMSNorm 형태의 RMSNorm이 LLaMA / Gemma / Mistral / Qwen의 default다.

**그리고 이제 이 course가 관심 갖는 주장: 두 배치는 동일한 activation byte를 쓴다.** 둘 다 LayerNorm의 **input**을 저장하며(norm당 `2·sbh`), 이것이 Korthikanti의 `4·sbh` 항 = reference config에서 block당 `67,108,864 B = 67.11 MB`(= 2 × 33.55 MB)다. Checkpoint tensor는 어느 쪽이든 `[B,T,h] = 33.55 MB`이므로(pre-LN: residual-2 합, post-LN: 최종 LN output), `2·s·b·h·L`은 배치와 무관하다. RMSNorm의 유일한 실제 byte 절약은 저장 statistic이 `[B,T]×2` fp32 = 32,768 B에서 `[B,T]×1` fp32 = 16,384 B로 줄어드는 것이다 — norm 2개 × block 80개에 대해 **5.24 MB → 2.62 MB**이며, 1.07 GB짜리 attention matrix 앞에서는 무시할 만하다.

> **Normalization 배치는 gradient-flow 결정이지 memory 결정이 아니다.** ([[pre-ln-vs-post-ln]])

### 6.2 MLP와 4h 확장

각 block의 feed-forward network는 hidden width를 확장하고 nonlinearity를 적용한 뒤 다시 project한다.

```
[B, T, h] --W_up--> [B, T, 4h] --GELU--> [B, T, 4h] --W_down--> [B, T, h]
```

`4×` 비율은 GPT 계열 전체에서 정확히 유지된다: GPT-1 768→3072, GPT-2 1600→6400, GPT-3 12288→49152. Reference config에서 `4h = 16,384`이고 각 `[B,T,4h]` tensor는 `134,217,728 B = 134.22 MB` — hidden state의 정확히 4배다. Block은 그중 **두 개**를 저장한다(up-projection output, GELU output) = `268,435,456 B = 268.44 MB`이며, 이는 33.55 MB checkpoint의 정확히 **8.0배**이고 MLP block 몫 318.77 MB의 84%다. 현대의 SwiGLU는 `d_ff ≈ (8/3)h`를 쓰지만 matrix가 셋이라 비슷한 byte 수에 도달한다.

### 6.3 Residual stream

```
x_L = x_0 + sum_{l=1..L} Delta_attn^(l) + sum_{l=1..L} Delta_ffn^(l)
```

Pre-LN에서 완전히 전개하면:

```
x_L = x_0 + sum_l [ Attn_l(LN(x_{l-1})) + FFN_l( LN( x_{l-1} + Attn_l(LN(x_{l-1})) ) ) ]
```

Decoder-only transformer는 embedding에서 unembedding까지 흐르는 **하나의 `[B,T,h]` tensor**이며, `2L`개의 sub-layer가 그것을 읽고, 계산하고, **다시 더한다**. 아무것도 덮어쓰이지 않는다. 결과 셋 모두 하중을 견딘다.

1. **Trainability.** `∂L/∂x_ℓ = ∂L/∂x_{ℓ+1} · (I + ∂F_{ℓ+1}/∂x_ℓ)` — identity 항이 모든 block을 통과하는 길이 0짜리 gradient 경로를 보장한다. 이것이 ResNet의 framing이다: `F(x) = H(x) − x`를 학습해서 "아무것도 하지 않음"이 `F ≈ 0`이 되게 한다. GPT-2는 residual-branch output projection을 `1/√N` scale(N = residual layer 수)로 initialize해서, `2L`번의 덧셈 합이 step 0에서 stream의 variance를 터뜨리지 않게 한다.
2. **Composition.** 이른 write가 절대 덮어쓰이지 않으므로 나중 layer가 훨씬 앞 layer가 쓴 feature를 읽을 수 있다 — induction head와 interpretability의 모든 circuit 수준 결과 뒤에 있는 mechanism이다.
3. **저렴한 checkpointing.** Checkpoint는 정확히 **stream의 snapshot 하나**다. 그래서 [[ch-03]]의 `2·s·b·h·L` formula가 문자 그대로 "element당 2 byte로 residual stream을 L번 찍은 snapshot"인 것이다. ([[residual-stream-memory-backbone]])

💡 **쉬운 설명.** Transformer를 "layer들이 데이터를 변환해서 다음으로 넘기는 파이프라인"이 아니라 **공용 칠판 하나**로 그려라. 모든 block은 칠판을 읽고, 자기 계산 결과를 칠판에 *더해서* 쓴다. 지우지 않는다. 그래서 (a) 아무것도 안 쓴 block도 gradient를 통과시키고, (b) 80번째 block이 3번째 block의 메모를 그대로 읽을 수 있으며, (c) checkpoint를 뜬다는 건 "칠판 사진 한 장"이면 충분하다는 뜻이다 — block 내부에서 계산하던 1.64 GB의 잡동사니는 사진 한 장(33.55 MB)에서 다시 만들어 낼 수 있다.

### 6.4 Block이 backward를 위해 보관하는 tensor의 순서 있는 목록

이 절이 이 chapter의 handoff의 심장이다. Pre-LN, recompute 없음, reference config, dataflow 순서:

| # | Tensor | Shape | Bytes | Backward가 필요로 하는 이유 |
|---|---|---|---|---|
| 1 | block input `x_in` | `[B,T,h]` | 33.55 MB | **이것이 CHECKPOINT다** |
| 2 | LN1 저장 stat (mean, rstd) | `[B,T]×2` fp32 | 32.8 KB | LN backward |
| 3 | LN1 output | `[B,T,h]` | 33.55 MB | Q/K/V matmul의 input |
| 4 | Q projection | `[B,T,h]` | 33.55 MB | `dK`가 Q를 필요로 함 |
| 5 | K projection | `[B,T,h]` | 33.55 MB | `dQ`가 K를 필요로 함 |
| 6 | V projection | `[B,T,h]` | 33.55 MB | `dP`가 V를 필요로 함 |
| 7 | attention probs `P = softmax(QKᵀ/√d)` | `[B,a,T,T]` | **1.07 GB** | softmax VJP가 자신의 **output**을 필요로 함 |
| 8 | context `P·V` | `[B,a,T,d]` | 33.55 MB | `W_O`의 input |
| 9 | `W_O` 이후의 attention output | `[B,T,h]` | 33.55 MB | — |
| 10 | residual-1 합 | `[B,T,h]` | 33.55 MB | LN2의 input |
| 11 | LN2 저장 stat | `[B,T]×2` fp32 | 32.8 KB | LN backward |
| 12 | LN2 output | `[B,T,h]` | 33.55 MB | `W_up`의 input |
| 13 | MLP up-projection output | `[B,T,4h]` | 134.22 MB | GELU가 자신의 **input**을 필요로 함 |
| 14 | GELU output | `[B,T,4h]` | 134.22 MB | `W_down`의 input |
| 15 | MLP down-projection output | `[B,T,h]` | 33.55 MB | — |
| 16 | residual-2 합 = block output | `[B,T,h]` | 33.55 MB | **다음** block의 checkpoint |

이 표를 외우는 게 아니다. Operation마다 규칙 하나씩으로 **유도**하는 것이다.

| operation | backward가 저장하는 것 | 이유 |
|---|---|---|
| matmul `Y = XW` | 자신의 **INPUT** `X` | `dW = Xᵀ·dY`이기 때문 |
| GELU | 자신의 **INPUT** | derivative가 pre-activation의 함수이기 때문 |
| softmax | 자신의 **OUTPUT** `P` | VJP가 `dS = P ⊙ (dP − rowsum(dP ⊙ P))`이기 때문 |
| dropout | 1-byte **MASK**만 | mask가 Jacobian 전체이기 때문 |
| LayerNorm | 자신의 **INPUT** + 저장된 (mean, rstd) | stat은 저장하는 게 recompute보다 싸기 때문 |
| residual add `y = x + f(x)` | **아무것도 없음** | `∂(x+f)/∂x = I`이기 때문 |

💡 **쉬운 설명.** 이 여섯 줄이 이 chapter에서 실무적으로 가장 값진 부분이다. "무엇이 memory를 먹느냐"를 표로 외우는 대신, operation을 보고 **자기 미분식에 무엇이 들어가는지**만 물으면 된다. matmul의 weight gradient 공식 `dW = Xᵀ dY`에는 `X`가 있으니 input을 저장하고, softmax의 미분은 `p_i(δ_ij − p_j)`로 확률 `P` 자체로 표현되니 output을 저장한다. Residual add의 미분은 그냥 1이라 저장할 게 없다. 이 규칙만으로 §8.2의 `34 = 11 + 19 + 4`를 직접 만들어 낼 수 있다.

명시적으로 말해 둘 corollary가 하나 있다. 그러지 않으면 이상해 보이는 coefficient를 설명해 주기 때문이다: **Q, K, V projection은 저장된 input 하나를 공유한다**(LN1 output). 그래서 Korthikanti가 QKV input에 대해 `6·sbh`가 아니라 `2·sbh`를 한 번만 세는 것이다.

**산술:**

```
block당 폐기 (항목 2-15) = 9·HID + 2·LNST + ATTN + 2·MLP
                         = 1,644,232,704 B = 1.64 GB
checkpoint (항목 1)      = 1·HID = 33,554,432 B = 33.55 MB
폐기 / checkpoint        = 49.0x
block 합계 (항목 1-15)   = 1,677,787,136 B ~ 1.68 GB   -> checkpoint의 50.0x
```

`L = 80` block 전체에 대해:

```
checkpointing 없음:  1,677,787,136 x 80 = 134,222,970,880 B = 134.22 GB
block별 ckpt:        2·s·b·h·L = 2·4096·1·4096·80 = 2,684,354,560 B = 2.68 GB
절감:                50x
```

이 `2·s·b·h·L`은 `ch-03/read.md` L110–116과 같은 formula이고, 33.55 MB / 1.64 GB / 49× 삼총사는 정확히 `ch-03/figures/checkpointing.html` 패널 1이 렌더링하는 값이다. 이 절이 그 패널 뒤에 있는 derivation이다.

**이 50×를 ch-03의 97×와 화해시키기 — 같은 lever, 서로 다른 분자.** `ch-03/read.md` L110–116은 checkpointing floor를 recompute 없는 `s·b·h·L·(34 + 5as/h)`에 대한 `2·s·b·h·L`로 서술한다. 두 chapter는 *같은* 분모로 나눈다 — block당 `2 sbh` checkpoint, 여기서는 33.55 MB다. 다른 것은 분자 위에 무엇을 세느냐뿐이다:

```
이 chapter의 16-tensor 열거:        block당 1,677,787,136 B = 100 sbh-unit + LN stat
                                    100 / 2 = 50x

ch-03이 쓰는 Korthikanti coefficient: (34 + 5as/h)·sbh = 194 sbh-unit = block당 3,254,779,904 B
                                    194 / 2 = 97x
```

50×와 97× 사이의 2배 격차는 *전적으로* §8.2가 자세히 풀어 놓는 output-대-저장-input 차이와 dropout 차이다(Korthikanti의 `5as²b`는 열거가 하나만 세는 `s²` shape tensor를 셋 세고, 여기에 이 chapter의 dropout-free 독법이 지우는 dropout tensor 셋이 더해진다). **어느 비율도 틀리지 않았다. 분자가 다른 비율일 뿐이다.** ch-03에서 97×를 기대하고 여기 왔다면 당신이 들고 있는 것은 Korthikanti의 published coefficient이고, 50×는 위에 열거한 16-tensor 목록을 기준으로 잰 같은 lever다.

💡 **쉬운 설명.** 같은 lever(gradient checkpointing)를 두 chapter가 서로 다른 배율로 인용하는 것이 헷갈릴 수 있는데, **분모는 완전히 같다** — block당 `2 sbh` checkpoint, 즉 33.55 MB다. 다른 건 "무엇을 아꼈다고 세느냐"뿐이다. 이 chapter는 실제로 열거한 16개 tensor(100 sbh-unit)를 세서 50×를 얻고, ch-03은 Korthikanti가 출판한 coefficient(194 sbh-unit)를 세서 97×를 얻는다. 194 − 100 = 94의 정체는 §8.2가 짚는 두 가지 — Korthikanti가 `s²` tensor를 셋 세는 것(dropout mask + dropout output 포함)과, output이 아니라 저장 input을 세는 관점 차이 — 그게 전부다. 면접이나 문서에서 인용할 때는 **어느 분자를 쓴 비율인지**를 같이 말하면 된다.

**혼자만 다르게 행동하는 tensor 하나.** 항목 7은 quadratic하게 자란다: `[B,a,T,T]`는 `T=4096`에서 block당 1.07 GB이고, `T=8192`에서는 `1 × 32 × 8192 × 8192 × 2 = 4,294,967,296 B = 4.29 GB` — 정확히 4배, 즉 quadratic이다. 이것은 동시에 **가장 큰** 저장 tensor이면서 **재구성이 가장 싼** 것이다(matmul 하나 + softmax 하나). 그 비대칭성이 selective recomputation([[ch-03]] §3)과 FlashAttention([[ch-05]])이 활용하는 대상이다. [[transformer-block-tensor-ledger]] 참조.

> **▶ 인터랙티브 companion — [`figures/transformer-block-dataflow.html`](figures/transformer-block-dataflow.html) (패널 0–3, 5)**
> *16개 tensor를 순서대로, 각각을 만들어 낸 save-rule과 함께.* **패널 0**은 page의 모든 숫자를 구동하는 config다: `B`(1–8), `s`(512 … 32,768), `h`(1024 … 8192), `a`(8/16/32/64), `L`(1–126) slider와 dtype selector(fp8 / bf16 / fp32)를 갖고, 기본값이 정확히 이 chapter의 reference config다. 파생 chip은 `d_head = h/a`, `4h`, 그리고 각 unit tensor의 byte 크기를 곱셈까지 대입해 보여준다 — `[B,s,h] = 33,554,432 B = 33.55 MB`, `[B,a,s,s] = 1,073,741,824 B = 1.07 GB`, `[B,s,4h] = 134,217,728 B = 134.22 MB`, `[B,s]×2` fp32 `= 32,768 B` — 즉 §1의 HID / ATTN / MLP / LNST 표를 live로 다시 계산한 것이다. **패널 1**은 두 버튼짜리 mode 전환(training vs inference-decode)에 gradient-checkpointing 체크박스를 더한 것으로, 아래 모든 tensor의 운명을 바꾼다.
> **패널 2**는 §6.4의 표를 16-step walkthrough로 만든 것이다. Residual stream이 왼쪽 rail로 그려지고(`x_in → +Δ_attn → +Δ_mlp → 다음 block`, 전부 `[B,s,h]`, 폭 불변 node 포함), 각 step은 이름·shape·byte 수·비례 막대, 그리고 **fate chip**(보존 / KV cache / 해제 / 다음 block)을 담은 row를 하나씩 추가한다. `[B,a,s,s]`인 항목 7 `P = softmax((QKᵀ+M)/√d_head)`는 **괴물**로 label되고 그 막대가 나머지 열다섯을 압도한다. 합계 card는 block당 1.68 GB(항목 1–15, 항목 16은 *다음* block의 소유라 제외), 폐기 1.64 GB, 보존 33.55 MB = **49.0×**, 그리고 `L = 80`에 걸쳐 134.22 GB 대 checkpointing 시 2.68 GB를 읽는다. 아래 canvas는 16 step에 걸친 누적 보존 byte를 세 곡선(training, training + checkpointing, inference-KV)으로 plot하며 attention-branch와 MLP-branch 구간을 음영 처리한다.
> **패널 3**은 *같은 stepper*를 인쇄 가능한 toy(`s = 3`, `h = 4`, `a = 2`, `d_head = 2`, `d_ff = 8`, `W_V = I`, weight는 `0 / ±1 / ±0.5`로 제한)에서 돌리며 모든 matrix element를 써 놓는다 — LN statistic, causal `−∞` cell, row 합이 붙은 head별 `P`, concat, `[3,8]` GELU, 그리고 원래 `x_in` 옆의 최종 block output — 게다가 각 step마다 element 하나를 손으로 전개한다. §6.4의 dataflow에서 추상화를 걷어낸 것이다.
> **패널 5**는 §6.1을 정적인 2-column diagram으로 만든 것이다: Post-LN은 빨간 빗금 normalization bar가 residual highway **위에** 앉아 있고(Xiong et al. 2020과 4,000 warmup step 인용), Pre-LN은 highway가 손대지지 않은 채 `∂x_{ℓ+1}/∂x_ℓ = I + ∂Sub(LN(x_ℓ))/∂x_ℓ`가 옆에 인쇄되어 있다 — 그 아래 live byte note가 두 배치 모두 **동일한** `4·sbh = 67.11 MB`를 쓴다는 것과 RMSNorm의 유일한 실제 절약(model 전체 저장 statistic `5.24 MB → 2.62 MB`)을 정량화한다.
> 하나는 정확히 기대해 두어라: 패널 0에서 `s`를 끌면 모든 linear row가 비례해서, `[B,a,s,s]` row가 quadratic하게 자라지만 **crossover는 `s`를 따라 움직이지 않는다** — 그것은 `s = 34h/(5a)` chip이고, reference config에서 **870.4**로 읽히며, `h`나 `a`를 움직일 때만 바뀐다. `s`의 함수가 아니라 `s`에 *대한* 임계값이다.

---

## 7. KV Cache

### 7.1 Mechanism

Inference에서 model은 한 번에 token 하나를 생성한다. Step `t+1`은 *단일* query vector를 계산하고 이전의 모든 key/value에 attend한다. §3.3의 invariant에 의해 `k_1..k_t`와 `v_1..v_t`는 이전 step에서 변하지 않았다. 그러니 cache하라.

```
Attention_causal(Q, K, V) = softmax( (Q K^T + M) / sqrt(d_k) ) · V,  M[i,j] = -inf for j > i
```

Decode step `t+1`에는 mask할 미래 key가 없으므로 decode kernel은 mask를 아예 뺀다. 이 cache는 **이미 pure한 함수에 대한 memoization**([[kv-cache-mechanism]])이다 — output에 대해서는 아무것도 바꾸지 않고, 각 `k_j`를 몇 번 계산하느냐만 바꾼다.

**절감을 token-forward-pass 수로 세면:**

```
cache 없음:  1 + 2 + ... + N = N(N+1)/2
cache 있음:  N
비율:        (N+1)/2

N = 1024  ->  524,800  vs  1,024  =  512.5x
```

Hidden size를 `d`라 할 때 layer당 FLOP으로는:

```
T개 token의 prefill        ~  7·B·T·d^2 + 2·B·T^2·d
depth t에서의 decode 1 step ~  7·B·d^2 + 2·B·t·d

cache 있음 = sum_t (7Bd^2 + 2Btd)  ~  7BNd^2 + BdN^2        (weight 항에서 O(N))
cache 없음 = sum_t (7Btd^2 + 2Bt^2 d) ~ (7/2)Bd^2 N^2 + (2/3)BdN^3   (weight 항에서 O(N^2))
```

`B=1`, `d=8192`(Llama-3-70B hidden size), layer당으로 검증한 비율:

| N | 비율 |
|---|---|
| 128 | 64.5× |
| 256 | 128.7× |
| 1,024 | **515.5×** (cache 있음 4.8963e11 FLOPs, 없음 2.5240e14) |
| 4,096 | 2094× |

비율은 N에 대해 **linear하게** 자란다. End-to-end 실측(Raschka, 124M model, 200 token, Mac Mini M4 CPU): cache 없음 17.5 s → naive cache 3.3 s(**5.3×**) → pre-allocated 2.8 s(6.25×) → pre-allocated + `torch.compile` 2.4 s(7.3×). FLOP 상의 512×와 실측 5.3× 사이의 격차가 바로 이 workload가 compute-bound이기를 멈추는 지점이다 — §7.5 참조.

### 7.2 Memory formula

방정식 하나가 모든 autoregressive deployment의 크기를 결정한다([[kv-cache-memory-formula]]):

```
KV bytes = 2 * B * s * L * n_kv_heads * d_head * bytes_per_element
```

**앞의 2는 K tensor 하나 + V tensor 하나다.** "2 byte"도 아니고 safety factor도 아니다. 기호:

| 기호 | 의미 |
|---|---|
| `L` | `num_hidden_layers` |
| `n_kv_heads` | `num_key_value_heads` — `num_attention_heads`가 **아니다** |
| `d_head` | `hidden_size / num_attention_heads` |
| `s` | cache된 token = prompt + 지금까지 생성분 |
| `B` | 동시 sequence 수 |
| `bytes_per_element` | 2 (bf16/fp16), 1 (fp8/int8), 0.5 (int4) |

💡 **쉬운 설명.** 이 공식에서 가장 많이 오독되는 기호가 맨 앞의 `2`다. bf16이 2 byte라서 2가 아니다 — K와 V, 두 개의 tensor라서 2다. Byte 수는 뒤쪽 `bytes_per_element`가 따로 담당한다. 그래서 int4로 양자화해도 앞의 2는 그대로 남는다.

Token당 비용은 `2 · L · n_kv_heads · d_head · bytes_per_element`이며 — **`n_heads`와도 `d_model`과도 무관하다**. bf16으로 계산하면:

| model | L | `n_kv_heads` | `d_head` | token당 bytes | |
|---|---|---|---|---|---|
| Llama-3-8B | 32 | 8 | 128 | 131,072 | 128.0 KiB |
| **Llama-3-70B** | 80 | 8 | 128 | **327,680** | **320.0 KiB** ← reference 숫자 |
| Llama-3-70B *가 MHA-64였다면* | 80 | 64 | 128 | 2,621,440 | 2,560.0 KiB = 2.5 MiB (정확히 8×) |
| Llama-3-405B | 126 | 8 | 128 | 516,096 | 504.0 KiB |
| Llama-2-7B (MHA) | 32 | 32 | 128 | 524,288 | 512.0 KiB |
| PaLM-540B (MQA) | 118 | 1 | 256 | 120,832 | 118.0 KiB |

**다섯 번째** row(Llama-2-7B, MHA, `n_kv_heads = 32`)를 **두 번째** row(Llama-3-70B, GQA-8)와 비교하라: **Llama-2-7B의 token당 cache가 더 크다 — 512.0 KiB 대 320.0 KiB — 그것도 parameter가 10분의 1인데 말이다.** 여기서는 architecture가 size를 이긴다. (세 번째 row는 "70B가 MHA-64로 출시되었다면"이라는 반사실이며, 그랬다면 token당 2,560.0 KiB로 실제의 정확히 8배였을 것이다.)

Request당, Llama-3-70B:

| context | bytes | |
|---|---|---|
| 8,192 | 2,684,354,560 | **2.50 GiB** |
| 32,768 | 10,737,418,240 | **10.00 GiB** |
| 131,072 (128k) | 42,949,672,960 | **40.00 GiB** |

(Llama-3-8B는 같은 세 길이에서 1.00 / 4.00 / 16.00 GiB.) Batch를 쓰면: `L=32, B=16, s=4096, n_kv_heads=8, d_head=128`, bf16 → `2·32·16·4096·8·128·2 = 8,589,934,592 B = 8.00 GiB`.

**Llama-3 family가 이 formula가 무엇에 의존하고 무엇에 의존하지 않는지를 가장 깔끔하게 보여준다:** *모든* size에서 `n_kv_heads = 8`이다. 8B는 `L=32, H_q=32`(4:1), 70B는 `L=80, H_q=64`(8:1), 405B는 `L=126, H_q=128`(16:1)이다. 8B → 405B는 **parameter가 50배인데 KV cache는 3.9배**에 불과하다(`516,096 / 131,072 = 3.9375`). `L`만 자랐기 때문이다.

**Capacity 계산**, 8×H100으로 Llama-3-70B를 serving. 이 chapter에서 decimal unit과 binary unit을 섞으면 답이 조용히 달라지는 유일한 자리이므로, **byte로** 계산한다(§1의 Convention note에 따라 hardware capacity와 parameter byte는 decimal로 인용한다. `640 GB = 6.40×10¹¹ B`. 위의 request당 KV 수치들은 정확한 binary 양이다):

```
KV budget = 6.40e11 (HBM) - 1.40e11 (weights, 70e9 x 2 B) - 8.0e10 (overhead)
          = 4.20e11 B  =  420 GB  =  391.16 GiB

8k request   :  2,684,354,560 B  ->  4.20e11 / 2.684e9  = 156.5  ->  동시 156
128k request : 42,949,672,960 B  ->  4.20e11 / 4.295e10 =   9.78 ->  동시 9
```

Context 16배에 동시 request가 **정확히** 16분의 1 — linearity가 요점이다. (함정에 주의하라: *숫자* 420을 변환 없이 *binary* 숫자 2.5와 40으로 나누면 168과 10.5가 나오는데, 이것이 `figures/kv-cache.html` 패널 4의 8×H100 card가 인쇄하는 값이다. 그 두 답은 `2³⁰/10⁹ = 1.074`만큼의 unit slip이고, byte-exact한 답은 156과 9.8이다.) 그리고 raw formula는 **lower bound**다 — block/page table(1–2%), fragmentation(contiguous 5–15%, paged <5%), swap 여유분, CUDA-graph bucket 예약분을 고려해 **1.15**를 곱하면 현실적인 수치는 **136**과 **8.5**에 떨어진다.

💡 **쉬운 설명.** 이 문단이 §1 Convention note가 존재하는 이유의 실물 예시다. 8k request 하나의 cache는 `2,684,354,560 B`인데, 이 값은 decimal로는 `2.68 GB`, binary로는 정확히 `2.50 GiB`다. 예산 `4.20e11 B`를 **byte로** 나누면 `4.20e11 / 2.684e9 = 156.5`가 나오지만, 예산은 decimal 숫자 `420`으로 두고 cache만 binary 숫자 `2.5`로 읽어서 나누면 `168`이 나온다 — 계산이 틀린 게 아니라 **두 숫자의 단위가 서로 다른 것**이다. 두 답의 비는 정확히 `2³⁰/10⁹ = 1.074`다. 규칙은 단순하다: 나눗셈 한 번 안에서는 양쪽을 모두 byte로 맞춰라. (figure의 패널 4는 아직 168/10.5를 인쇄하고 있으니, 그 card를 볼 때는 이 문단을 authority로 읽어라.)

### 7.3 MHA / GQA / MQA / MLA — 한 축 위의 네 점

MHA, GQA, MQA는 정확히 **숫자 하나**, `n_kv_heads`에서만 다르다:

| scheme | `H_kv` | cache 나눗수 |
|---|---|---|
| MHA | `H_q` | 1× |
| GQA-G | `G` | `H_q/G` |
| MQA | 1 | `H_q` |

Query head `h`는 KV head `h // (H_q/G)`에서 읽는다. 각 KV head는 `H_q/G`개의 query head를 담당한다. Llama-3-70B GQA-8: `H_q = 64`, `H_kv = 8`, KV head당 query head 8개, 정확히 8× 감소다.

품질 비용은 실재하지만 작고, 공격적인 쪽 끝에 몰려 있다. MQA 대 MHA로 흔히 인용되는 수치는 **HumanEval −2.2, GSM8K −1.5, MMLU −0.6**이다. (§4.5와 같은 방식으로 표시하는 단서: **이 세 값은 검증되지 않았다.** 이 course의 source library에서 primary citation 없이 넘어온 것이고, 유력한 후보 둘 중 어느 쪽으로도 추적되지 않는다 — Shazeer 2019는 MQA를 도입했지만 저 세 benchmark보다 앞서고, Ainslie et al. 2023의 GQA ablation은 T5 규모의 summarization/QA이지 HumanEval/GSM8K/MMLU가 아니다. *방향*과 *자릿수* — reasoning 위주 task에 몰린 한 자릿수 퍼센트의 작은 회귀 — 만 takeaway로 가져가고, 소수점은 인용하지 마라.) `G ≥ 8`인 GQA는 noise 범위 안에서 MHA와 같고, `G = 8`이 경험적 sweet spot이다. 거기 도달하려고 처음부터 재학습할 필요는 없다: **GQA uptraining**은 MHA checkpoint의 KV head를 `G`개 group으로 mean-pooling한 뒤 원래 pretraining compute의 약 5%로 pretraining을 이어가서, 처음부터 GQA로 학습한 것 대비 perplexity 약 0.1 이내로 회복한다. Llama-2-70B가 GQA-8을 얻은 방법이 이것이다.

**MLA는 이 family를 벗어난다.** DeepSeek의 Multi-head Latent Attention은 token당 low-rank latent 하나를 cache하며 — 주의해서 보라 — **factor 2가 없다**. 단일 latent `c_KV`가 K와 V 양쪽을 담당하기 때문이다:

```
MLA bytes/token = L * (d_c + d_rope) * bytes_per_element

DeepSeek-V3:  L=61, d_c=512, d_rope=64, bf16
              61 * 576 * 2 = 70,272 bytes = 68.6 KiB/token
```

V3 자신의 geometry로 계산한 naive MHA(61 layer, head 128개, `d_head=128`, bf16 = `2·61·128·128·2 = 3,997,696 B = 3.81 MiB/token`)와 비교하면 비율은 **56.9×**다. `d_rope` 항은 구현상의 군더더기가 아니다: RoPE가 position에 의존하므로 `k_rope`는 압축되지 않은 채 남아야 하며, 압축 *이전에* rotate하면 latent가 position-specific해져서 재사용이 파괴된다. 그래서 `d_c`가 아니라 `(d_c + d_rope)`다. [[gqa-mqa-mla-kv-heads]] 참조.

### 7.4 구현상의 함정

**Pre-allocation vs `torch.cat`.** `torch.cat`은 매 step마다 cache 전체를 복사한다 — step당 `O(n)` copy, 총 `O(n²)` — 대신 실제 길이만큼의 memory만 쓴다. `torch.zeros(B, H_kv, max_seq_len, d_head)`는 step당 `O(1)`이지만 최댓값을 예약한다: 128k-context model에서 50 token짜리 request에도 약 8 GB를 예약한다.

**고전적인 bug 둘.** (1) Generation 사이에 cache를 **reset**해야 한다. 안 그러면 새 prompt의 query가 이전 sequence의 낡은 key에 attend한다 — output이 유창해 보이면서 미묘하게 틀리는, 최악의 failure mode다. (2) **Position id를 추적해야 한다**(`pos_ids = arange(current_pos, current_pos + seq_len)`). 안 그러면 새 token마다 position 0으로 취급되어 RoPE가 깨진다 — §5.3의 `Δ = m − n`이 쓰레기가 된다.

**PagedAttention**(vLLM)은 KV pool의 fragmentation을 해결한다: block당 token 16개, sequence별 block table, copy-on-write prefix sharing으로 KV 활용률 96% 이상을 달성한다(contiguous allocator는 20–38%). §7.6을 위해 기억해 두라: **training에는 대응물이 없다.**

### 7.5 Prefill vs decode — 서로 다른 두 기계

```
H100 SXM roofline 무릎: 1979 TFLOPS bf16 / 3.35 TB/s HBM = 590 FLOPs/byte
Llama-3-70B, bf16: P = 70e9 params -> forward pass마다 weight 140 GB를 읽음
```

**Prefill**(`T = 4096`개 prompt token을 한 번에)은 같은 140 GB weight read에 대해 `2·P·T = 5.7344e14` FLOPs를 한다:

```
이론 intensity = (2·P·T FLOPs) / (P · 2 B) = T = 4,096 FLOPs/byte  ->  무릎의 6.9배 위
실효/실측      ~ 3,100 FLOPs/byte                                  ->  무릎의 5.2배 위
```

이론값이 정확히 prompt 길이인 이유는, prefill이 각 weight를 **한 번** 읽어서 `T`개 token 전부에 재사용하기 때문이다. `~3,100`이라는 값은 KV write, activation, non-GEMM 작업까지 포함했을 때의 실측 baseline이다. 어느 쪽이든: **COMPUTE-BOUND**다. 8×H100에서 prompt에 약 36.2 ms, 약 113,000 tok/s.

**`B = 1`에서의 Decode**는 `2·P = 1.4e11` FLOPs를 하고 weight *에 더해* cache까지 읽는다:

```
bytes = 1.40e11 (weights) + 1,342,177,280 (s=4096에서의 KV, 1.25 GiB) = 1.4134e11
이론 intensity = 1.4e11 / 1.4134e11 = 0.99 FLOPs/byte      ->  무릎의 596배 아래
실효/실측      ~ 0.3 FLOPs/byte                            ->  무릎의 1,970배 아래
                                                              = 1.005 TFLOP/s = peak의 0.051%
```

이론값 `≈ 1`은 우연이 아니고 sanity check로 외워 둘 값이다: `B = 1`에서는 모든 weight element를 한 번 읽어 정확히 한 번의 multiply-add에 쓰므로, bf16 decode는 model이 아무리 커도 **2-byte element당 2 FLOPs = byte당 1 FLOP**을 얻는다. 실측 `~0.3`은 그 천장에서 실제 kernel overhead를 뺀 값이고, `figures/kv-cache.html` 패널 3이 roofline 위에 찍는 값이 이것이다. **BANDWIDTH-BOUND**, token당 약 5.27 ms, 약 190 tok/s.

둘 사이는 `3,100 / 0.3 = 1.03e4` — 같은 hardware에서, 같은 request 안에서, **네 자릿수 차이**다. (대신 이론값 쌍을 쓰면 `4,096 / 0.99 = 4.1e3`으로 3.6 자릿수다. 어느 쪽 bracketing도 방어 가능하지만 **세** 자릿수는 아니다 — 그것이 이 chapter의 이전 오기였고 격차를 10배 과소평가한 것이다.) Prefill은 큰 GEMM으로 `T`개 token을 병렬 처리하고, decode는 token 하나를 처리하며 *weight와 cache를 HBM에서 읽는 데* 시간을 쓴다. §7.1의 512× FLOP 절감이 wall-clock 5.3×로 나타나는 이유가 그것이다. Bandwidth-bound phase에서 FLOP을 제거하면 FLOP 수가 시사하는 것보다 적게 벌게 된다. 그리고 batching이 유일한 실제 탈출구인 이유도 그것이다 — `B`는 weight read를 고정한 채 decode intensity의 분자만 곱한다.

💡 **쉬운 설명.** Roofline의 590 FLOPs/byte는 "byte 하나를 읽어 올 때 590번 이상 계산에 써먹어야 GPU가 놀지 않는다"는 손익분기점이다. Prefill은 3,100으로 여유롭게 넘기니 계산기가 병목이고, decode는 0.3이라 한참 못 미치니 memory 통로가 병목이다. Decode의 이론 상한이 왜 하필 1이냐면 — bf16 weight 하나(2 byte)를 읽어 곱셈 한 번, 덧셈 한 번(= 2 FLOPs)에만 쓰고 버리기 때문이다. 그래서 `B = 1` decode는 model 크기와 무관하게 "byte당 1 FLOP"이라는 천장에 갇힌다. 병목이 통로일 때 계산량을 512배 줄여 봐야 통로 폭은 그대로라 5배 정도밖에 안 빨라진다.

### 7.6 **TRAINING에는 KV CACHE가 없다 — 0 BYTE**

이것이 이 chapter에서 가장 중요한 경계이고, 가장 자주 틀리는 지점이다.

Teacher forcing은 ground-truth sequence `x_1 … x_s`를 넣고 **한 번의** forward pass에서 **모든** position의 cross-entropy를 계산한다. Causal mask(§3.1)가 position `t`를 `≤ t`인 position에만 의존하게 만든다. Autoregressive loop가 없다 ⇒ 반복 작업이 없다 ⇒ **amortize할 것이 없다** ⇒ cache가 없다. KV cache는 순수하게 inference-time 구조물이다.

**함정**은 동일한 대수식이 다른 대상을 서술한다는 점이다. *Training* forward pass 동안 보유하는 K와 V의 byte는:

```
2 * B * s * L * n_kv_heads * d_head * b
```

— inference KV-cache formula와 글자 하나까지 같다. Llama-3-8B geometry, `B=1, s=8192`, bf16에서: `2·32·1·8192·8·128·2 = 1,073,741,824 B = 1.00 GiB`이며, 8k request 하나에 대한 inference KV cache와 수치적으로 동일하다. 같은 설정의 Llama-3-70B: `2,684,354,560 B = 2.50 GiB`. 같은 숫자, 다른 거주자다.

| | training | inference |
|---|---|---|
| sequence당 forward pass 수 | **1** | prefill 1 + decode N step |
| K/V를 보유하는 이유 | backward가 `dL/dQ`, `dL/dK`, `dL/dV`에 필요 | 다음 decode step이 그것들에 attend |
| lifetime | optimizer step 한 번, 그 후 free | request 전체 |
| 사용 중에 자라는가 | **아니오** — 전체 `s`에 대해 한 번에 할당 | **예** — sequence당 step당 +1 token |
| 제거 가능한가 | **예** — gradient-checkpointing recompute | **아니오** — 압축만 가능(GQA/MLA/quant/paging) |
| ledger 항목 | **activation** ([[ch-01]] 항목 4) | 별도의 persistent KV pool |

> **"KV cache" line item이 들어 있는 training OOM 분석은 전부 double-counting이다.** ([[train-vs-infer-kv-boundary]])

💡 **쉬운 설명.** Cache는 "같은 계산을 여러 번 하니까 결과를 아껴 두는" 장치다. Training은 sequence 전체를 한 번에 밀어 넣고 끝내므로 같은 계산을 두 번 하는 일이 없고, 따라서 아껴 둘 것도 없다. Training에도 K/V byte는 분명히 존재하지만 그건 **backward를 위한 activation**이지 cache가 아니다. 이름이 같다고 ledger에 두 줄로 적으면 같은 memory를 두 번 세는 것이다.

**그리고 GQA 나눗수는 training에서 다르다.** FlashAttention의 backward는 Q, K, V, O와 logsumexp statistic을 저장한다. GQA는 K와 V만 줄인다 — **Q는 건드리지 않는다**. Training 쪽 나눗수를 유도하면:

```
training QKV-activation 나눗수 = 3·H_q / (H_q + 2·H_kv)
```

| model | inference 나눗수 | training 나눗수 |
|---|---|---|
| Llama-3-8B (32 → 8) | 4.0× | **2.00×** |
| Llama-3-70B (64 → 8) | 8.0× | **2.40×** |
| MQA (64 → 1) | 64.0× | **2.909×** |

`B=1, s=8192, d_head=128`, bf16, Llama-3-70B에서 layer당 byte를 검증하면:

```
GQA-8 : Q 128 MiB + K 16 MiB + V 16 MiB = layer당 160 MiB = 167,772,160 B  -> x80 = 12.5 GiB
MHA-64: Q 128 + K 128 + V 128           = layer당 384 MiB = 402,653,184 B  -> x80 = 30.0 GiB
비율 30.0 / 12.5 = 2.4x
```

따라서 "GQA가 8× 절감을 준다"는 헤드라인은 *inference* 주장이다. Training에서는 약 2.4×이고, GQA를 아무리 공격적으로 밀어도 3×를 넘지 못한다. **PagedAttention에 training 대응물이 없는** 것도 같은 구조적 이유다. 그것은 KV-pool fragmentation을 해결하는데, training에서는 할당이 step 시작 전에 이미 알려진 하나의 contiguous shape이므로 그 문제 자체가 존재하지 않는다.

💡 **쉬운 설명.** GQA가 inference에서 8× 이득인 이유는 cache에 K와 V만 들어가기 때문이다. Training에서는 backward가 Q도 저장해야 하는데 GQA는 Q를 전혀 줄이지 않는다. 그래서 "줄어드는 부분(K,V) : 안 줄어드는 부분(Q)" 비율이 이득의 상한을 정해 버리고, `H_kv → 1`로 극단까지 밀어도 `3H_q/H_q = 3×`를 넘을 수 없다.

**가져가야 할 mental bridge:** *training ≈ prefill + backward, 영원히*(compute-bound, cache 없음, 전부 병렬); *serving = prefill 한 번, 그 뒤로 길게 이어지는 bandwidth-bound decode 꼬리.*

**boson / Lina TMR hook.** GDN linear-attention layer는 `CP=1`로 hard-assert되어 있다. 그 inference-time state는 head당 고정 크기의 recurrent state(표준 linear attention에서는 `d_head × d_head` matrix)이며 **sequence length와 무관**하다 — 따라서 `2·L·n_kv_heads·d_head·s·b` scaling 논증이 그 layer들에는 전혀 적용되지 않는다. Training 쪽은 그대로다. GDN block도 backward를 위해 position별 activation을 저장하므로, 32k sequence 청구서는 **cache 문제가 아니라 checkpointing 문제**다. Inference 직관을 [[ch-09]]의 training budget으로 수입하지 마라.

> **▶ 인터랙티브 companion — [`figures/kv-cache.html`](figures/kv-cache.html) (패널 1–6)**
> *무엇이 recompute되고, 무엇이 cache되며, training은 대신 무엇을 하는가.* **패널 1**은 "애초에 cache가 왜 합법인가"에 답하려고 cache **없이** token 6개를 생성한다: `6×6` grid에서 diagonal cell은 파랑(`K_j`를 새로 계산), 그 왼쪽은 전부 빨강(**bit 단위로 동일하게** 재계산)이고, 증명 box가 매 step마다 `K₁ = x₁·W_K`를 문자 그대로 손으로 다시 유도해 같은 정수가 되돌아오는 것을 보여준다. Counter가 계산된 `t(t+1)/2`와 낭비된 `t(t−1)/2`를 추적하고, 초록 callout이 §3.3의 invariant와 그 역 — bidirectional attention에는 KV cache가 없다 — 을 서술한다.
> **패널 2**는 같은 generation을 나란히 재생한다. 정책 A(cache 없음, `O(N²)` 태그) 대 정책 B(cache 있음, `O(N)` 태그)로 `N = 6`에서 **21 대 6 = 3.5×**에 도달하고, 고정 card가 **N = 1024에서 512.5×**를 읽는다. 아래 Q-strip은 `q₁…q₅`에 취소선을 긋고 `q_t`만 살려 두는데, 이것이 사람들이 놓치는 detail이다: **Q는 절대 cache되지 않는다.** 파란 callout이 §7.1의 정확한 FLOP 표를 `d = 8192, B = 1`에서(`N = 128 / 256 / 1024 / 4096`에 대해 `64.5× / 128.7× / 515.5× / 2094×`) 그리고 Raschka의 실측 `17.5 s → 3.3 s → 2.8 s → 2.4 s`를 나른다.
> **패널 3**은 §7.5다: 넓은 PREFILL block 하나 뒤에 좁은 `+1 tok` decode block이 이어지는 timeline과, 그 옆에 `min(peak, BW×I)` 지붕·점선으로 그린 **무릎 590**·음영 처리된 BANDWIDTH-BOUND / COMPUTE-BOUND 영역·plot된 두 점(prefill `I = 3,100`, decode(`B=1`) `I = 0.3`)을 담은 log-log roofline canvas가 놓이고, 현재 step에 링이 둘러진다. 빨간 box가 H100 무릎 산술을 못 박고, 두 번째 callout이 직관(intensity ≈ weight read당 처리 token 수)을 주면서 batching이 유일한 탈출구임을 서술한다.
> **패널 4**는 §7.2의 master formula를 계산기로 만든 것이다: preset 여섯 개(8B / 70B / 405B / Llama-2-7B MHA / PaLM-540B MQA / "70B가 MHA-64라면"), `B`·`s`(512 … 131,072)·`L`·`n_kv_heads`·`d_head` slider와 dtype selector(2 / 1 / 0.5 B), live 값이 대입된 formula box, 그리고 출력 row 다섯 개 — token당, request당, batch당, ×1.15 현실 보정, 8×H100에서의 동시 request 수. 응시할 가치가 있는 것은 canvas다: KV-cache 곡선(기울기 1, `s`에 **linear**)과 training `[B, a=32, s, s] × L` 곡선(기울기 2, **quadratic**)을 80 GB 기준선과 함께 log–log 축에 그리는데, `s = 131,072`에서 cache 40.00 GiB 대 training score tensor 약 80 TiB — **2,048배** 차이로 읽힌다.
> **패널 5**는 §7.3을 네 mode 전환(MHA `n_kv=8` / GQA-4 / GQA-2 / MQA)으로 만들어 `H_q = 8` 고정의 head-mapping diagram 위에 올리고, 여기에 정적인 "Llama-3-70B가 이렇게 만들어졌다면" 표를 더한다: MHA-64가 token당 2,560.0 KiB, 실제 GQA-8이 320.0 KiB(8×), MQA가 40.0 KiB(64×), MLA가 68.6 KiB(56.9×, `61 × (512+64) × 2`, 앞의 2가 없다는 점에 주의). 닫는 callout 둘은 두 함정이다 — 왜 `d_rope`는 압축되지 않은 채 남는가, 그리고 왜 **GQA의 나눗수가 training으로 넘어오지 않는가**.
> **패널 6**은 §7.6, 즉 그 경계를 8-step split 애니메이션으로 보여준다: 왼쪽은 TRAINING · teacher forcing, "forward pass 1회" 태그와 함께 `x₁…x₆`가 동시에 켜지고 card가 **KV cache 0 B** 옆에 **K/V activation 1.00 GiB**를 읽는다. 오른쪽은 INFERENCE · autoregressive, prefill 후 decode append가 이어지고 card가 자라나는 cache와 **동일한 1.00 GiB, 삭제 불가**를 읽는다. 아래 비교 표는 §7.6의 표와 같은 row를 돌리고, PagedAttention note(활용률 96% 이상 대 20–38%), `CP=1`의 GDN linear-attention에 대한 boson / Lina TMR 문단, 그리고 두 loop를 대조하며 고전적 bug 둘을 제자리에 label한 code block으로 끝난다.

---

## 8. 이것이 Memory Ledger의 어디에 꽂히는가

### 8.1 Mechanism을 [[ch-01]]의 여섯 항목에 mapping

| ledger 항목 | 이 chapter가 공급하는 것 |
|---|---|
| 1. weights (2 B/param) | attention block당 `4·d_model²` (§4.3) + MLP `8·d_model²` + `[V,h]` embedding (§1.2). Head 수의 기여는 **0** |
| 2. gradients (2 B/param) | 항목 1과 같은 shape. RoPE는 0 기여, 학습되는 PE table은 `L×d` 기여 (§5.4) |
| 3. Adam states (12 B/param) | 마찬가지 — 그리고 이것이 **학습되는** PE table이 16–18 B/param을 쓰는 반면 RoPE의 cos/sin cache는 전부 합쳐 4.00 MiB인 이유다 (§5.4) |
| 4. **activations** | **§6.4의 16-tensor 목록 전체.** 이 chapter의 주된 예치금 |
| 5. logit spike (`B·T·V`) | §1.2의 `h → V` 확장, [[ch-02]] `qa-deep-2` Q7에서 이미 계산됨 |
| 6. overhead | 여기 있는 어떤 것으로도 변하지 않음 |

### 8.2 §6.4로부터 Korthikanti의 coefficient 재구성하기

§6.4의 save-rule을 block 전체에 적용하고 `s·b·h` byte 단위로 세면([[selective-recompute-korthikanti]] §4.1):

```
attention block = 11·sbh + 5·a·s²·b
    2 sbh   공유되는 QKV input, 한 번만 저장  (§6.4의 corollary)
    4 sbh   QK^T를 위해 유지되는 Q와 K
    2 sbh   P·V를 위해 유지되는 V
    2 sbh   W_O linear projection의 input
    1 sbh   attention-dropout mask (1 byte/elem)
    2as²b   softmax OUTPUT
    1as²b   softmax-dropout MASK (1 byte/elem)
    2as²b   softmax-dropout OUTPUT

MLP block = 19·sbh
    2 sbh   up-projection input
    8 sbh   GELU input   (4h 폭 -> 4 x 2 = 8)
    8 sbh   down-projection input (4h 폭)
    1 sbh   MLP-dropout mask

LayerNorms = 4·sbh   (저장 input 2 sbh x norm 2개)

TOTAL = 34·sbh + 5·a·s²·b = s·b·h·(34 + 5as/h)      따라서 34 = 11 + 19 + 4
```

Reference config에서:

| 항 | bytes | |
|---|---|---|
| 11 sbh (attention) | 184,549,376 | 184.55 MB |
| 19 sbh (MLP) | 318,767,104 | 318.77 MB |
| 4 sbh (LayerNorms) | 67,108,864 | 67.11 MB |
| **34 sbh** | 570,425,344 | **570.43 MB** |
| `5as/h = 5·32·4096/4096 = 160`, 따라서 `s²` 항 = 160 sbh | 2,684,354,560 | **2.68 GB** |
| **layer당** `(34+160)·sbh = 194 × 16,777,216` | 3,254,779,904 | **3.25 GB** |
| **× L=80** | 260,382,392,320 | **260.38 GB** |

**3.25 GB와 §6.4의 1.68 GB를 화해시키기 — 이 둘을 모순으로 두지 마라.** 같은 block을 다른 granularity로 센 것이다.

- **16-tensor 목록**은 `s²` shape tensor를 element당 2 B로 **한 번**만 센다(`2as²b` = 1.07 GB, softmax output만, dropout 없음). Korthikanti는 dropout mask와 dropout output까지 포함하므로 `s²` shape tensor **셋 전부**(`5as²b` = 2.68 GB)를 센다.
- 반대 방향으로, 목록은 block **OUTPUT** 36 sbh를 세고, Korthikanti는 저장된 **INPUT** 34 sbh를 센다.

둘 다 맞다. **하나의 합 안에서 절대 섞지 마라.**

💡 **쉬운 설명.** 같은 방의 물건을 "나간 것 기준"과 "들어온 것 기준"으로 센 차이라고 보면 된다. §6.4는 각 operation이 *만들어 낸* tensor를 세고, Korthikanti는 각 operation이 backward를 위해 *붙잡아 둔* tensor를 센다. 게다가 Korthikanti는 dropout이 켜진 2022년 기준이라 `s²` tensor를 셋(softmax output, dropout mask, dropout output) 세고, §6.4는 하나만 센다. 두 회계가 모두 옳고, 섞는 순간만 틀린다.

**2026년을 위한 파생 교정(논문에 없는, derivation임을 명시):** 현대 LLM은 `dropout = 0`으로 학습한다. Dropout tensor 셋을 지우면 coefficient가 `sbh(32 + 2as/h)`가 된다 — attention 10 sbh, MLP 18 sbh, LayerNorm 4 sbh, `s²` 항은 `2as²b`. 그 `2as²b`가 **정확히** §6.4 항목 7의 1.07 GB attention-probability tensor다. Layer당 `(32+64)·sbh = 96 × 16,777,216 = 1,610,612,736 B = 1.61 GB`, ×80 = `128,849,018,880 B = 128.85 GB`(열거 기준 134.22 GB 대비 — 남는 차이가 output-대-input granularity다). **논문을 인용할 때는 `34 + 5as/h`를 쓰고, 실제 dropout-free run을 모델링할 때는 `32 + 2as/h`를 써라.**

**거부해야 할 우연 하나.** 이 config에서 `5as²b`(**한** layer의 attention 항) = 2,684,354,560 B = 2.68 GB이고, `2·s·b·h·L`(**80개 전부**에 대한 full-recompute floor) = 2,684,354,560 B = 2.68 GB다. `5·a·s = 5·32·4096 = 655,360 = 2·h·L = 2·4096·80`이기 때문에 수치가 같을 뿐이다. **둘은 무관한 양이다.** 이 둘 사이에 연관이 있다고 암시하는 mental model이나 animation은 전부 틀렸다.

### 8.3 Crossover — 두 개가 있고, 서로 다르다

Linear 항과 quadratic 항을 같다고 두면 "attention이 언제 지배하기 시작하는가"에 답할 수 있다 — 단, 답은 어느 linear 항을 말하느냐에 달려 있다.

**Attention block 내부만 보면**(`11·s·b·h = 5·a·s²·b`):

```
s = 11h/(5a)
  7B model (h=4096, a=32):  11 x 4096 / 160 = 45056/160 = 281.6 token
  2017 base (h=512,  a=8):  11 x 512  / 40  = 5632/40   = 140.8 token
```

수백 token만 넘어가면 attention block의 activation memory는 사실상 **전부 `N×N` routing table**이고 Q/K/V가 아니다.

**Block 전체와 비교하면**(`34 = 5as/h`), 이 숫자는 이미 `ch-04/qa.md`에 있고 여기서 그대로 재사용한다:

```
s = 34h/(5a) = 34 x 4096 / (5 x 32) = 870.4  ~  870 token
```

**약 870 token**을 넘으면 attention의 `s²` 항이 *다른 모든 activation의 합*을 능가한다. 두 숫자 모두 맞다. 서로 다른 질문에 답하는 것이며, 엉뚱한 자리에 엉뚱한 것을 인용하는 것이 혼란스러워 보이는 가장 쉬운 방법이다.

💡 **쉬운 설명.** "언제부터 attention이 memory를 먹기 시작하나?"라는 질문은 사실 두 개다. (1) attention block 안에서만 보면 약 282 token이면 이미 score matrix가 Q/K/V를 압도한다. (2) MLP까지 포함한 block 전체와 비교하면 약 870 token이 분기점이다. 실무에서 인용해야 할 숫자는 대개 870이고, 282는 "attention 내부에서는 Q/K/V가 애초에 문제가 아니었다"는 논점을 뒷받침할 때 쓴다.

그리고 `ch-04/qa.md`에서 이미 확립된 GPU당 anchor를 그대로 유지하면: `s=4096, h=4096, a=32, t=8, L=80, b=1`에서 GPU당 attention `s²` 항은 `5as²b·L/t = 5·32·4096²·1·80/8 = 26,843,545,600 B = 26.8 GB`이며, SP를 쓴 다른 모든 activation 5.7 GB, 그리고 432 GB 전체 static ledger를 GPU 64개로 나눈 6.75 GB와 대비된다.

### 8.4 Handoff

[[ch-04]]가 말하는 모든 것에 이제 mechanism이 깔렸다.

| [[ch-04]]의 주장 | 이 chapter가 제공하는 mechanism |
|---|---|
| "N×N score matrix를 materialize한다" | §2.3 step 3 — softmax는 `P`를 만들어야 하고, §6.4의 규칙에 따르면 softmax는 자신의 **output**을 저장한다 |
| "head당 layer당" | §4.4 — tensor가 `(B, a, N, N)`이며 FLOP 고정인 채 `a`에 linear |
| "N=32k에서 head당 2 GB" | §4.4의 표, `2,147,483,648 B` |
| "kernel이 결정한다" | §3.4 — lower triangle은 구조적으로 죽어 있고, causal-aware kernel만이 그것을 byte로 전환한다 |
| "streaming은 matrix를 절대 쓰지 않는다" | §2.3 — `P`는 언제나 row 단위로만 필요하며, §3.4의 tile-local masking이 tile을 self-contained하게 만든다 |
| "MATH backend는 O(N²)로 fallback한다" | §5.4 — custom logit-additive bias가 그 fallback을 강제하는 흔한 경로 중 하나다 |

> **⚠ Handoff 지점에서 다시 서술하는 scope note (§1 참조).** 위 표의 모든 줄, 그리고 §8.2와 §8.3의 모든 `s²` 항은 **표준 softmax attention**에 대한 회계다. boson / Lina TMR은 `CP=1`의 **GDN linear-attention**을 쓰고, linear attention은 `N×N` score matrix를 절대 만들지 않는다 — 따라서 그 layer들에 대해서는 `5·a·s²·b` 항도, `s = 34h/(5a)` crossover도, GPU당 26.8 GB attention anchor도 가리킬 대상 자체가 없다. 반면 §6.4에서 `s`에 linear한 것은 **전부 그대로 넘어온다**: residual stream, Q/K/V projection, MLP의 `4h` tensor 두 개, LayerNorm 저장 statistic, 그러므로 coefficient의 `34·sbh` 쪽과 `2·s·b·h·L` checkpointing floor 전부. [[ch-09]]를 위한 실무적 독법은 이렇다 — boson에게 긴 sequence는 kernel로 푸는 quadratic 문제가 아니라 checkpointing과 parallelism으로 푸는 **linear**한 activation-memory 문제다. 그리고 바로 그 점이 quadratic 서사를 알아 둘 가치가 있는 이유이며(그것이 GDN이 제거하도록 설계된 비용이다), 동시에 그것을 boson의 budget에 그대로 붙여 넣으면 안 되는 이유다.

> **▶ 인터랙티브 companion — [`figures/transformer-block-dataflow.html`](figures/transformer-block-dataflow.html) (패널 4와 6)**
> *16개 tensor에서 `34 + 5as/h`까지, 회계 convention 하나씩.* **패널 4**는 §8.2의 질문을 자기 제목으로 그대로 묻는다 — "위 16개의 합 = ch-03의 `s·b·h·(34 + 5as/h)` 인가?" — 그리고 residual stream / attention branch / MLP branch / LN 저장 statistic / `s²` 항 / block 합계 / `× L`을 column으로 갖는 4-row reconciliation 표로 답한다. 네 row는 정확히 이 chapter가 구분하는 네 가지 독법이다: **(a)** §6.4에서 열거한 16개 tensor, **(b)** Korthikanti의 `sbh(34 + 5as/h)`를 `4·sbh + 11·sbh + 19·sbh + 5as²b`로 분해한 것(linear 570.43 MB + quadratic 2.68 GB = block당 3.25 GB, `L = 80`에서 260.38 GB), **(c)** dropout-free 2026 재계산 `sbh(32 + 2as/h)`(block당 1.61 GB, `L = 80`에서 128.85 GB), 그리고 **(d)** 항목 1만 남기고 나머지를 버리는 gradient checkpointing. 표 아래 빨간 note가 차이를 column 단위로 짚는다 — 센 output에 대한 `+2`, 세지 않은 dropout mask에 대한 `−1`, 그리고 `s²` column의 tensor-하나-대-셋 격차 — 그래서 §8.2의 reconciliation을 문단으로 추론하는 대신 표에서 읽어 낸다. 오른쪽 column은 VJP save-rule(matmul → 입력, GELU → 입력, softmax → 출력, dropout → 1 B mask, LayerNorm → 입력 + (mean, rstd), residual add → 아무것도 안 함)과, `s = 34h/(5a) = 870.4`를 현재 dominance factor(`s = 4096`에서 4.7×)와 함께 계산하는 초록 callout을 나른다.
> 패널 4에는 **조건부** 빨간 경고 배너도 있는데, `5as = 2hL`이고 dtype이 bf16일 때만 뜬다 — reference config에서는 참이다 — 그리고 서로 다른 두 cell에 나타나는 `2,684,354,560 B`가 한 layer의 `5as²b`와 80개 layer 전부의 checkpoint floor 사이의 수치적 우연임을 말 그대로 서술한다. §8.2의 "거부해야 할 우연 하나"를 산문이 단언하는 대신 figure가 강제하는 것이다.
> **패널 6**은 [[ch-01]]로 고리를 닫는다: 이 block을 여섯 ledger 항목에 mapping하는 live 표 — weights(`12h² = 201,326,592` params × 2 B = block당 402.65 MB), gradients(402.65 MB), AdamW state(2.42 GB), **activations**(강조된 row, 16개 tensor 전부, block당 1.68 GB → 134.22 GB), logit spike(흐림, block에 대해서는 0), overhead(흐림, 0) — 여기에 "K, V가 decode 사이에 남는 것"이라는 보라색 머리의 추가 row 하나가 붙어 **ledger에 없음**으로 표시되고 inference formula `2·B·s·L·n_kv·d_head·b`를 나른다. 닫는 callout은 정적인 parameter당 비용(16 B/param, `B·s`에 불변)을 항목 4(`B·s`에 비례하고 일부는 `s²`에도 비례)와 대조하고, training에는 KV-cache 줄이 없다는 것을 다시 서술한다.

---

## 문헌에서 얻은 핵심 통찰

**1. Attention은 differentiable dictionary lookup이며, 모든 설계 선택은 그 lookup을 학습 가능하게 만드는 데서 따라 나온다** (Vaswani et al. 2017; [[qkv-scaled-dot-product]]). Hard `argmax`는 거의 모든 곳에서 gradient가 0이므로, 모든 value에 대한 softmax-weighted convex combination으로 대체된다. Projection이 셋인 이유는 `XXᵀ`로 뭉치면 세 가지가 동시에 독립적으로 깨지기 때문이다 — symmetry, norm이 균등화된 input에서의 Cauchy–Schwarz diagonal dominance, 그리고 학습 가능한 routing parameter의 부재. 이 mechanism은 "architecture 선택"이 아니다. Differentiable하면서 directional하고 learnable한 최소한의 구성물이다.

**2. `1/√d_k`는 화장품 같은 normalization이 아니라 필요조건이다** ([[sqrt-dk-scaling-variance]]). i.i.d. unit-variance 가정 아래 dot-product variance는 *정확히* `d_k`이므로 raw score는 `√d_k`로 자란다. 그것을 softmax에 넣으면 saturate되고, gradient가 `W_Q`와 `W_K`에 도달하는 **유일한** 경로인 softmax Jacobian `p_i(δ_ij − p_j)`가 붕괴한다. 측정된 비용은 완전히 평범한 score 두 개에 대해 `d_k = 64`에서 **약 586×** gradient 감소다. Vaswani는 downstream 효과도 측정했다: scaling 없는 dot-product attention은 큰 `d_k`에서 additive attention에 패배하지만, scaling된 attention은 그와 동등하면서 평범한 GEMM으로 남는다.

**3. Multi-head attention은 FLOP-neutral하고 memory-expensive하다** ([[multi-head-split-concat-wo]]). Parameter는 head 수 `a`와 무관하게 `4·d_model²`이고, FLOP은 `2·B·N²·d_model`이며 `a`가 *정확히* 상쇄된다. 그러나 score tensor는 `(B, a, N, N)`이므로 byte는 **FLOP 고정인 채 `a`에 linear하게** scaling하고, arithmetic intensity는 정확히 **저장 element당 `2·d_head` FLOPs(= bf16에서 저장 byte당 `d_head` FLOPs)**다. Head를 두 배로 하면 intensity가 절반이 되고 memory가 두 배가 된다. Vaswani의 Table 3 row (A)는 품질 곡선이 head 8개와 16개 사이에서 평평하고 양 끝에서 떨어짐을 보여준다 — 그래서 compute에서 공짜인 head 수가 ledger에서는 *결코* 공짜가 아니다.

**4. Position은 property가 아니라 input이며, 어디에 주입하느냐가 memory contract를 결정한다** ([[attention-permutation-equivariance]], [[sinusoidal-absolute-encoding]], [[rope-rotary-position-embedding]]). 맨 self-attention은 정확히 permutation-equivariant이며, 그 증명의 핵심은 conjugate된 score matrix와 `V` 사이에서 상쇄되는 `Pᵀ P = I`인 네 줄짜리다. 2017년의 답에 이미 rotation 구조가 들어 있었고, RoPE의 기여는 그것을 embedding에 additive하게가 아니라 **Q와 K에 multiplicative하게** 적용한 것이며, 이는 세 제약(relative 의존성, 원점에서의 identity, magnitude 보존)에 의해 유일하게 강제된다. 그 대가는 미학이 아니라 구조적이다: RoPE는 attention kernel *바깥*에 살아서 FlashAttention이 여전히 평범한 `(Q,K,V)` triple을 보게 하는 반면, logit-additive scheme은 `[B,H,T,T]` bias — `H=32, T=8192`에서 **layer당 4.000 GiB** — 를 필요로 하며 이는 kernel이 피하려고 존재하는 바로 그 tensor다.

**5. Block의 저장 tensor 목록은 operation당 규칙 하나로 유도 가능하다** ([[transformer-block-tensor-ledger]]). matmul은 INPUT을 저장하고(`dW = Xᵀ dY`), GELU는 INPUT을, softmax는 OUTPUT을(VJP `dS = P ⊙ (dP − rowsum(dP ⊙ P))`), dropout은 1-byte MASK를, LayerNorm은 input + (mean, rstd)를 저장하며, residual add는 아무것도 저장하지 않는다. 그 목록을 합한 것이 **곧** Korthikanti의 coefficient다: `34 = 11 + 19 + 4`. 두 자릿수 order의 memory 분석이 여섯 개의 derivation 규칙과 하나의 shape 질문으로 환원된다.

**6. Normalization 배치는 gradient-flow 결정이지 memory 결정이 아니다** ([[pre-ln-vs-post-ln]]). Post-LN은 residual highway 위에 norm을 올려서 backward가 `L`개의 LayerNorm Jacobian을 직렬로 통과하게 만든다. Xiong et al. 2020의 theorem은 initialization 시점에서 *마지막* layer의 gradient를 bound한다: Post-LN은 `O(d√(ln d))`로 depth와 무관하고, Pre-LN은 `O(d√(ln d / L))`로 `1/√L`만큼 감쇠한다. `1/√·`는 Pre-LN 쪽에 붙고 그 안은 layer index가 아니라 **총 depth `L`**이다. 실패 mode는 그 결과로 생기는 Post-LN gradient의 *depth 불균형*이며(정성적 서술 — 형식적 결과는 위 두 scaling이다), 그래서 2017년 논문은 warmup 4,000 step이 필요했고 Pre-LN은 필요 없다. Pre-LN은 highway에 깨끗한 `I`를 남긴다. 그러나 둘 다 같은 `4·sbh`(block당 67.11 MB)를 저장하고, checkpoint는 어느 쪽이든 `[B,T,h]`다. RMSNorm의 *유일한* 실제 byte 절약은 저장 statistic tensor를 절반으로 줄이는 것, model 전체에 걸쳐 5.24 MB → 2.62 MB이며 — 1.07 GB짜리 attention matrix 하나 앞에서는 무시할 만하다.

**7. Residual stream이 gradient checkpointing을 저렴하게 만드는 이유다** ([[residual-stream-memory-backbone]]). Decoder-only transformer는 `2L`개 sub-layer가 읽고 다시 더하는 하나의 `[B,T,h]` tensor다. 그 고정 폭의 additive backbone이 동시에 설명한다 — 왜 100-layer model이 학습되는지(identity gradient 경로), 왜 feature가 layer를 가로질러 조합되는지(이른 write가 절대 덮어쓰이지 않음), 그리고 왜 checkpoint가 **내부 tensor 1.64 GB에 대해 33.55 MB — 49×** 인지. [[ch-03]]의 `2·s·b·h·L`은 문자 그대로 "element당 2 byte로 residual stream을 L번 찍은 snapshot"이다.

**8. Training에는 KV cache가 없고, 동일한 대수식이 함정이다** ([[kv-cache-mechanism]], [[kv-cache-memory-formula]], [[gqa-mqa-mla-kv-heads]], [[train-vs-infer-kv-boundary]]). Cache는 causal attention의 immutability invariant에 의해 강제되며 `(N+1)/2`번의 token-forward-pass를 절약한다 — `d=8192`, N=1024에서 FLOP 기준 515.5×. 그 크기는 `2·B·s·L·n_kv_heads·d_head·bytes`이고 `n_heads`와 `d_model`에 무관하며, 그래서 Llama-3-405B가 parameter 50배에 cache는 3.9배다. **그러나 teacher forcing 때문에 training은 한 번의 병렬 forward pass를 돌리므로 amortize할 것이 없고 training KV cache는 0 byte다.** Training에 *실제로* 존재하는 K/V byte는 optimizer step 하나의 lifetime을 갖는 activation이며 checkpointing으로 삭제 가능하다 — 그리고 inference에서 8×를 주는 GQA 나눗수는 training에서 **2.40×**밖에 주지 않는다. FlashAttention의 backward가 Q도 저장하는데 GQA는 Q를 절대 줄이지 않기 때문이다.

---

## 핵심 정리

- Attention은 학습 불가능한 `argmax` lookup을 softmax convex combination으로 대체한다. Output은 언제나 value vector들의 convex hull 안에 있으며, 그래서 `W_O`가 존재한다.
- Projection 셋은 필수다: `XXᵀ`는 symmetric이고(relation은 directional하다), RMSNorm으로 norm이 균등화되면 diagonal이 strict max이며(attention이 identity로 붕괴한다), routing을 학습할 parameter가 없다.
- `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`는 4단계로 실행된다. Step 1–3이 `(B,N,N)` tensor를 만들고, backward가 보관하는 것은 step 3의 output이다.
- `Var(q·k) = d_k`가 정확히 성립하므로 `1/√d_k`가 모든 head dimension에서 unit score variance를 복원한다. 이를 건너뛰면 평범한 score에서 `d_k = 64` 기준 약 586× gradient를 잃고, 가정 자체가 training 중에 무너진다 — QK-norm이 다시 강제하는 것이 바로 그 가정이다.
- Causal mask는 additive이고 softmax 이전에 적용된다. `torch.finfo(dtype).min`을 쓰고(`-1e9`은 fp16에서 표현 불가, literal `-inf`는 전부 mask된 row에서 `0/0 = NaN`), diagonal을 포함하라(`j ≤ i`).
- Mask의 진짜 선물은 invariant다: `k_j, v_j`는 token `j`에만 의존하며 절대 수정되지 않는다. 이것이 teacher forcing(한 번의 병렬 pass로 training)*과* KV cache(inference memoization)를 동시에 허가한다 — course의 양쪽 축을 떠받치는 같은 사실이다.
- Head는 parameter에서 공짜이고(`4·d_model²`, head 수 `a`에 불변) FLOP에서도 공짜지만(`2·B·N²·d_model`, `a`가 정확히 상쇄), memory에서는 **공짜가 아니다**: `(B,a,N,N)`은 FLOP 고정인 채 `a`에 linear하며, 저장 element당 `2·d_head` FLOPs(bf16에서 저장 byte당 `d_head` FLOPs)다. `N=32,768`에서 score tensor는 Q+K+V를 합친 것의 **85배**이고, 그 비율은 `N`에 대해 linear하게 자란다.
- Self-attention은 정확히 permutation-equivariant하므로(`Attn(PX) = P·Attn(X)`) position을 주입해야 한다. Causal mask가 그 증명을 깨고, 그래서 NoPE decoder LM이 애초에 작동한다.
- RoPE는 세 제약에 의해 유일하게 강제되며 `⟨R_m q, R_n k⟩ = qᵀ R_{n−m} k`를 낳는다 — absolute position이 사라진다. Kernel 바깥에 살고, `L=8192`에서 cos/sin cache 4.00 MiB를 쓰며, fuse되면 backward-저장 activation을 0 추가한다. ALiBi의 `[B,H,T,T]` bias는 `H=32, T=8192`에서 **layer당 4.000 GiB**다.
- Pre-LN vs post-LN은 trainability 결정이며 activation byte는 **동일하다**(`4·sbh` = 어느 쪽이든 block당 67.11 MB).
- Block 하나는 reference config에서 약 16개 tensor, 합계 1.68 GB를 저장하고 그중 checkpoint는 33.55 MB다 — **49–50×** 비율이며, 80개 block에 걸쳐 `2·s·b·h·L = 2.68 GB` 대 checkpointing 없는 134.22 GB다. [[ch-03]]은 *같은* lever를 **97×**로 인용하는데, 그 분자가 이 chapter의 열거값 `block당 100 sbh-unit`이 아니라 Korthikanti의 published `(34 + 5as/h)·sbh = block당 194 sbh-unit`이기 때문이다. 분모는 같고 회계 convention만 다르다(§6.4, §8.2).
- Korthikanti의 `34·sbh + 5as²b`는 여섯 개 save-rule로 *유도 가능*하다: `34 = 11(attn) + 19(MLP) + 4(LN)`. Reference config에서 layer당 3.25 GB이고 그중 2.68 GB가 `s²` 항이다. Dropout 없는 2026년 run은 `32 + 2as/h` = layer당 1.61 GB로 모델링하는 편이 낫다.
- Crossover 둘, 질문 둘: attention block 내부는 **약 282 token**(`11h/5a`), attention 대 다른 모든 activation은 **약 870 token**(`34h/5a`). 둘 다 맞다. 바꿔 쓰지 마라.
- **Training의 KV-cache byte는 0이다.** KV-cache line item이 있는 training memory 분석은 double-counting이다. Training 쪽 GQA 나눗수는 inference의 8×가 아니라 `3H_q/(H_q + 2H_kv)` ≈ 2–3×다. Q는 절대 줄어들지 않기 때문이다.

---

## 질문

Discuss phase를 위해 준비되었다 — 서술이 아니라 causal하게 추론하라.

1. `XXᵀ`는 서로 독립적인 세 가지 이유로 실패한다(§2.2). Symmetry 문제만 고쳤다고 하자 — `W_Q ≠ W_K`를 쓰되 `W_V = I`로 묶어 value가 raw residual stream이 되게 한다. 세 실패 중 어떤 것들이 돌아오며, 학습된 model에서 무엇을 관측하리라 예상하는가?
2. `1/√d_k`는 init에서는 참이지만 training 중에 무너지는 가정 아래 유도된다(§2.5). Step 50,000에서 score가 가정된 variance의 5배로 drift한 head의 *관측 가능한* 신호를 예측하라 — loss curve에서, attention entropy에서, `W_Q`의 gradient norm에서. 그런 다음 QK-norm이 왜 이를 고치고 learning rate를 그냥 낮추는 것은 왜 못 고치는지 설명하라.
3. Head 수는 정확히 FLOP-invariant하지만 memory-linear하다(§4.3–§4.4). boson은 `h = 4096`에서 `a = 32`를 쓴다. `a`를 16으로 절반 줄이고 `d_head`를 256으로 두 배 늘리면 (a) parameter, (b) FLOP, (c) `5as²b` activation 항, (d) inference 시 KV cache, (e) Table 3 row (A) 기준 예상 품질이 각각 어떻게 변하는지 말하라. 이 다섯 중 A100-40GB에서 binding constraint는 무엇인가?
4. Permutation 증명(§5.1)은 `P M Pᵀ ≠ M`이기 때문에 causal attention에서 실패한다. 실험에 호소하지 말고, 그 실패가 왜 정확히 NoPE decoder LM이 absolute position을 encoding할 수 있게 해 주는지, 그리고 그 mechanism이 positional resolution에 어떤 상한을 함의하는지 설명하라.
5. §6.4의 save-rule 표는 softmax가 **output**을, GELU가 **input**을 저장한다고 말한다. 각각을 해당 VJP로부터 유도한 뒤, 같은 논리로 SwiGLU가 무엇을 저장해야 하는지, 그리고 동일한 `d_ff`에서 SwiGLU의 matrix 세 개가 GELU의 두 개보다 activation footprint를 좋게 만드는지 나쁘게 만드는지 판단하라.
6. §8.2는 3.25 GB와 1.68 GB를 "같은 block을 다른 granularity로 센 것"으로 화해시킨다. *역시* 옳으면서 네 번째 숫자에 도달하는 세 번째 회계를 구성하고, capacity-planning 문서에서 넷 중 무엇을 인용할지 정하는 규칙을 서술하라.
7. §7.6은 training의 KV-cache byte가 0이라고 단언하는 동시에 동일한 formula가 Llama-3-70B activation 2.50 GiB를 준다고 말한다. 동료가 recompute를 줄이려고 "gradient-accumulation micro-step 사이에 K와 V를 cache하자"고 제안한다. 이 제안을 진단하라: micro-step 사이에 실제로 공유되는 것은 무엇이고, 무엇이 아니며, 무엇이 깨지는가?
8. `CP=1`인 GDN linear-attention은 sequence-length와 무관한 inference state를 갖는다(§7.6). 그것이 boson training run에 대한 §8 ledger의 *어떤* 숫자라도 바꾸는가? Linear-attention 문헌이 아니라 §6.4의 save-rule로부터 논증하라.

---

## 참고자료

- Ashish Vaswani et al. "Attention Is All You Need." arXiv:1706.03762, 2017 (Table 3, p.9는 v7 PDF 사용). https://arxiv.org/abs/1706.03762 — [[qkv-scaled-dot-product]], [[sqrt-dk-scaling-variance]], [[causal-mask-neg-inf]], [[multi-head-split-concat-wo]], [[sinusoidal-absolute-encoding]]
- Ruibin Xiong et al. "On Layer Normalization in the Transformer Architecture." ICML 2020, arXiv:2002.04745. https://arxiv.org/abs/2002.04745 — [[pre-ln-vs-post-ln]]
- Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey Hinton. "Layer Normalization." arXiv:1607.06450, 2016.
- Biao Zhang, Rico Sennrich. "Root Mean Square Layer Normalization." NeurIPS 2019, arXiv:1910.07467. https://arxiv.org/abs/1910.07467
- Kaiming He et al. "Deep Residual Learning for Image Recognition." CVPR 2016, arXiv:1512.03385. https://arxiv.org/abs/1512.03385 — [[residual-stream-memory-backbone]]
- Jianlin Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864, 2021. https://arxiv.org/abs/2104.09864 — [[rope-rotary-position-embedding]]
- Bowen Peng et al. "YaRN: Efficient Context Window Extension of Large Language Models." arXiv:2309.00071, 2023. https://arxiv.org/abs/2309.00071
- Ofir Press, Noah A. Smith, Mike Lewis. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ALiBi). ICLR 2022, arXiv:2108.12409. https://arxiv.org/abs/2108.12409
- Adi Haviv et al. "Transformer Language Models without Positional Encodings Still Learn Positional Information." Findings of EMNLP 2022, arXiv:2203.16634. https://arxiv.org/abs/2203.16634 — [[attention-permutation-equivariance]]
- Amirhossein Kazemnejad et al. "The Impact of Positional Encoding on Length Generalization in Transformers." NeurIPS 2023, arXiv:2305.19466. https://arxiv.org/abs/2305.19466
- Elena Voita et al. "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned." ACL 2019, arXiv:1905.09418. https://arxiv.org/abs/1905.09418 — *수치는 `llm-arch` wiki에서 가져왔으며 여기서 재검증하지 않음*
- Noam Shazeer. "Fast Transformer Decoding: One Write-Head Is All You Need" (MQA). arXiv:1911.02150, 2019. https://arxiv.org/abs/1911.02150 — [[gqa-mqa-mla-kv-heads]]
- Joshua Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023, arXiv:2305.13245. https://arxiv.org/abs/2305.13245 — [[gqa-mqa-mla-kv-heads]]
- DeepSeek-AI. "DeepSeek-V2 / DeepSeek-V3 Technical Report" (Multi-head Latent Attention). arXiv:2405.04434, arXiv:2412.19437. — [[gqa-mqa-mla-kv-heads]]
- Woosuk Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023, arXiv:2309.06180. https://arxiv.org/abs/2309.06180
- Vijay Korthikanti et al. "Reducing Activation Recomputation in Large Transformer Models." arXiv:2205.05198, 2022 (`34·sbh + 5as²b` 분해는 §4.1). https://arxiv.org/abs/2205.05198 — [[selective-recompute-korthikanti]], [[transformer-block-tensor-ledger]]
- Sebastian Raschka. "Understanding and Coding the KV Cache in LLMs from Scratch," 2025 (124M model 실측 timing). — [[kv-cache-mechanism]]

**형제 chapter:** [[ch-01]](이 chapter가 채우는 여섯 항목 ledger), [[ch-02]](precision과 `B·T·V` logit spike), [[ch-03]](activation, checkpointing, selective recomputation, sequence parallelism), [[ch-04]](이 chapter가 prerequisite인 O(N²) memory 문제), [[ch-05]](FlashAttention), [[ch-06]](kernel zoo와 SDPA MATH fallback), [[ch-07]](parallelism taxonomy), [[ch-09]](boson hook이 착지하는 27B MoE capstone).
