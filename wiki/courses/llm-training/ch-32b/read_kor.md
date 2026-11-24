<!-- chapter: ch-32b
     track: midtraining
     kind: content
     title: 컨텍스트 길이 확장: 방법, 데이터 혼합, 짧은 컨텍스트 회귀
     deps: [ch-32a]
     sources: [[position-interpolation]], [[yarn]], [[eleuther-extending-the-rope]], [[hf-transformers-yarn-code]], [[llama-2-long]], [[rope-base-bounds-context-length]], [[longrope2]], [[dual-chunk-attention]], [[string-effective-context]], [[controlled-long-context-extension]], [[prolong]], [[prolong-recipe]], [[fu-2024-data-engineering-128k]], [[llama-3]], [[llama-3-recipe]], [[qwen-long-context-synth]], [[longred]], [[longpo]], [[skyladder]], [[qwen-3-long-context]], [[olmo-3-long-context]], [[gemma-3-long-context]], [[harm-de-vries-long-context-run]]
     figures: figures/rope-scaling-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32b — 컨텍스트 길이 확장: 방법, 데이터 혼합, 짧은 컨텍스트 회귀

> **핵심 통찰.** 짧은 길이에서 사전학습된 RoPE 모델은 사전학습에 비하면 작은 규모의 지속 학습 단계로 64K–512K 토큰까지 확장할 수 있다. [[fu-2024-data-engineering-128k]]에서는 50억 토큰, [[prolong]]에서는 400억 토큰, [[olmo-3-long-context]]에서는 500억–1,000억 토큰, [[llama-3]]에서는 약 8,000억 토큰을 사용했다. 위치 인코딩 변경(보간, base 증가, YaRN)은 모델이 새 위치를 아예 사용할 수 있는지를 결정한다. 데이터 혼합은 짧은 컨텍스트 능력을 얼마나 잃는지를 결정한다. 이 장의 통제 비교에서는 언제나 긴 데이터 비율이 높거나 긴 데이터의 출처가 더 좁을수록 짧은 컨텍스트 점수 또는 짧은 컨텍스트 validation loss가 나빠졌고, 고품질 짧은 데이터를 다시 섞으면 그 손실이 줄었다(ProLong Fig. 3; Olmo 3 §3.6.3; Fu et al. Table 5). 검토한 어떤 recipe도 모든 benchmark에서 짧은 컨텍스트 비용이 0이라고 보고하지 않는다.
>
> **지침.** 지속 학습으로 컨텍스트를 확장할 때는 RoPE를 변경하고(base 증가 또는 YaRN), 혼합물에 고품질 짧은 데이터를 37–66% 유지하라. 최종 ProLong recipe는 37%, Olmo 3는 66%를 사용한다. 근거는 ProLong의 50억 토큰 Llama-3-8B ablation에서 긴 데이터 비율이 오를수록 짧은 과제 평균이 단조롭게 하락했고, Olmo 3의 100억 토큰 시험에서는 긴 데이터 66%일 때 34%일 때보다 2.5점, 34%일 때는 0.8점 손실했기 때문이다. 현재 길이에서 짧은 컨텍스트 평가가 회복되고 retrieval이 풀렸을 때에만 다음 길이로 진행하라. Llama 3는 여섯 단계에서 그렇게 했다. 추론 시점 확장만 가능하면 정적 YaRN factor보다 길이 의존적 방법(dynamic scaling, DCA)을 우선하라. Qwen3 model card는 정적 YaRN이 짧은 입력 품질을 낮출 수 있다고 경고한다. 재학습 예산이 없고 짧은 컨텍스트 품질이 우선이면 원래 context window를 유지하라.

## 범용 모델에서 이 장이 중요한 이유

범용 모델은 긴 문서, repository, 도구 사용 transcript, 여러 turn의 대화 이력을 읽어야 하며, 지식·수학·코드에 관한 짧은 질문에도 답해야 한다. 컨텍스트 길이 확장은 전자의 능력을 추가하는 단계이고, 동시에 후자의 능력을 잃게 할 수 있는 단계다.

이 단계는 mid-training에 놓인다. attention 비용이 sequence length에 따라 커지고 긴 자연 문서가 드물기 때문에 사전학습은 4K–32K 토큰에서 실행한다(§1.4). 사전학습 뒤 더 긴 sequence length에서 continued-pretraining을 하면 window를 확장한다. Olmo 3 저자는 공개 recipe에서 세 배치 방식을 정리한다. Llama 3.1은 확장을 “midtraining 이전”에 적용하고, Qwen 2.5와 Qwen3는 “그 이후”에 적용하며, GLM 4.5는 “supervised finetuning 이후에만” 확장한다([[olmo-3-long-context]], §3.6, 2차 보고). Qwen3에서는 이 긴 컨텍스트 단계가 STEM·코드 비중이 높은 reasoning 단계 뒤에 온다([[qwen-3-long-context]]). 이후 post-training이 SFT, preference optimization, RL을 거친 뒤에도 긴 컨텍스트 능력이 살아남는지를 결정한다(§7).

이 장은 이 단계에서 세 질문에 답한다. (1) 어떤 방법과 mixture 선택이 짧은 컨텍스트 능력을 없애지 않고 사용 가능한 context를 추가하는가? (2) 어떤 선택이 MMLU·GSM8K·code 점수의 손실, 또는 긴 길이 perplexity는 낮지만 retrieval을 못 하는 모델이라는 narrowing을 일으키는가? (3) 이 trade-off를 어떻게 측정하는가? continual pretraining 장([[ch-32a]])은 forgetting 일반을 다루고, 다음 장([[ch-32c]])은 effective context length 측정을 다룬다. RoPE의 유도는 여기서 반복하지 않으며 llm-arch course의 ch-06과 ch-16에 있다.

## §1 학습된 길이 밖과 그 안에서 모델이 실패하는 이유

### 1.1 표기법과 범위 밖 위치

**정의.** Rotary position embedding(RoPE)은 query와 key 차원을 둘씩 짝지어 토큰 위치에 비례한 각도로 회전시킨다. head 크기가 d일 때 차원쌍 i의 주파수는 θ_i = b^(−2i/d)이고, 여기서 b는 RoPE base다(LLaMA와 Llama 2에서는 10,000). 파장 λ_i = 2π/θ_i는 한 번 완전히 회전하는 데 필요한 토큰 수다([[yarn]] §2.1, Eq. 8).

**문제.** 길이 L sequence로 학습된 모델은 L보다 큰 query-key 거리를 본 적이 없다. 변경 없이 L보다 긴 길이에서 평가하면 LLaMA 7B의 PG19 perplexity는 4,096–32,768 토큰에서 10³을 넘는다([[position-interpolation]] Table 1). 변경 없는 Llama 2 7B는 4,096에서 perplexity 7.87이고 8,192에서는 10²보다 크다([[dual-chunk-attention]] Table 1).

**메커니즘.** attention score는 거리에 대한 cosine 항들의 합이다. Chen 등은 이 score 함수를 [0, L] 안에서 맞추었고, 그 구간에서는 유계로 남지만 L 바깥에서는 8,000을 넘을 수 있음을 보였다. 한편 학습된 정수 위치들 사이를 보간할 때의 error bound는 extrapolation bound보다 최소 약 600배 작다([[position-interpolation]] §2.2–2.3, Eqs. 5–8). 긴 길이에서 지속 학습만 한다고 이것이 저절로 고쳐지지는 않는다. RoPE를 바꾸지 않은 Llama 2 7B는 긴 continual pretraining 뒤에도 retrieval probe에서 “4,000–6,000 토큰 너머에 attend할 수 없었다”([[llama-2-long]] §4.1).

### 1.2 충분히 학습되지 않은 저주파 차원

**문제.** 파장이 L보다 긴 차원은 사전학습 동안 한 바퀴도 완전히 돌지 않으므로 모델은 그 범위의 일부만 본다.

**계산 예.** Llama 2에서 d = 128, b = 10,000, L = 4,096이라 하자. 가장 긴 파장(i = 63)은 2π·10,000^(126/128) ≈ 54,410 토큰이므로, 4,096-token window는 한 주기의 약 7.5%만 덮는다. i ≥ 46인 차원은 λ_i > 4,096이고, 이는 64개 차원쌍 중 18개다(유도한 값. §2의 그림은 어떤 설정에도 이를 계산한다).

**근거.** LongRoPE2는 “theoretical critical dimension”을 d_tcd = 2⌈(d/2)·log_b(L_train/2π)⌉로 계산한다(Eq. 7). Phi3-mini(d = 96, b = 10,000, L_train = 2,048)에서는 d_tcd = 62, 즉 31번째 cosine 차원이다. 48번째 차원의 주기는 51,861 토큰이며 2,048 토큰은 그 4% 미만이다. 논문은 긴 거리 의존성이 드물기 때문에 d_tcd 아래의 고차원도 under-trained되었다고 가설을 세운다. 검색 결과 “real” critical dimension은 Phi3-mini에서 25, LLaMA3-8B에서 30으로, 이론값 31과 35보다 작다([[longrope2]] §2.2, §3.1–3.2). Table 5에서 검색한 d_rcd를 YaRN에 적용하면 LLaMA3-8B RULER-128k가 49.39에서 71.46으로 오르므로 이 해석을 뒷받침한다.

### 1.3 왼쪽으로 치우친 상대 위치 빈도

**정의.** 위치 빈도 f(i)는 학습 중 corpus가 거리 i에서 제공하는 query-key pair의 수다.

**수식.** f(i) = Σ_{s∈C} max(|s| − i, 0), 0 ≤ i < L ([[string-effective-context]] Eq. 2). C는 training sequence 집합, |s|는 sequence s의 길이, i는 상대 거리다.

**계산 예.** 길이 4 sequence 하나에서는 f(0) = 4, f(1) = 3, f(2) = 2, f(3) = 1이다. 모든 sequence가 full length여도 가장 큰 거리 L − 1은 sequence마다 한 번만 나타나지만 거리 0은 L번 나타난다.

**근거.** SlimPajama에서 L = 2,048일 때 i ≤ 1,024 거리가 position index의 80% 이상을 차지하고 i ≥ 1,536은 5% 미만이다. 2K와 4K에서 각각 1T 토큰 사전학습한 TinyLlama-1.3B 두 모델은 같은 training token 수가 아니라 f(1280)가 같은 count인 1,000억에 도달했을 때 같은 effective length(1,280)에 도달했다([[string-effective-context]] §2.2, §3 Finding 2; 단일 연구 결과). 공개 모델에서 관찰한 결과는 effective length가 training length보다 짧다는 것이다. 추론 때 잘 학습된 위치를 드물게 학습된 범위로 이동시키는 STRING은 학습 없이 Llama 3.1 70B의 RULER 128K 점수를 66.6에서 81.7로 올린다(Table 2). [[ch-32c]]가 이 측정 격차를 자세히 다룬다.

### 1.4 긴 학습 데이터가 드물고 긴 학습이 비싼 이유

**데이터.** C4와 RefinedWeb에서는 문서의 95% 이상이 2K 토큰보다 짧다. 단 RefinedWeb 토큰의 약 45%는 2K보다 긴 문서에서 나온다. Gutenberg 책의 75% 이상은 16K를 넘는다([[harm-de-vries-long-context-run]] §2; source당 1만 개 문서를 표본으로 한 practitioner evidence). 무작위 짧은 문서를 긴 window에 packing하면 의존 관계가 없는 token pair가 많이 생긴다.

**계산 수식.** 토큰당 training FLOPs는 FFN 48·N_l·d², QKVO 24·N_l·d², attention 6·N_l·d·(L+1)이다. N_l은 layer 수, d는 hidden size, L은 context length다([[harm-de-vries-long-context-run]] §1).

**계산 예.** FFN + QKVO 대비 attention overhead는 6d(L+1)/(72d²) = (L+1)/(12d)다. d = 4,096이면 L = 4,096에서 8.3%, L = 16,384에서 33.3%, L = 128,000에서 260%다. d = 8,192에서는 같은 overhead에 도달하는 길이가 두 배다. 그래서 recipe는 짧은 길이로 사전학습하고 나중에 확장한다. [[llama-3]] §3.4.2도 self-attention layer의 compute가 sequence length에 대해 quadratic하게 증가하므로 “긴 sequence를 더 일찍 학습하지 않는다”고 쓴다.

**함의.** 확장 run의 결과는 위치 인코딩뿐 아니라 training data에도 좌우된다. mixture는 짧은 고품질 데이터가 담고 있는 일반 지식을 대체하지 않으면서 긴 거리 의존성을 제공해야 한다(§5).

## §2 확장 방법과 각 방법이 바꾸는 것

대화형 그림 [figures/rope-scaling-explorer.html](figures/rope-scaling-explorer.html)은 선택한 head dimension, base, 원래 길이, scale factor에 대해 PI·NTK-aware scaling·YaRN의 차원별 frequency multiplier와 wavelength를 그린다. 아래 계산 예와 함께 사용하라.

### 2.1 Position Interpolation(PI)

**정의.** PI는 RoPE를 적용하기 전에 모든 position index를 scale factor s = L′/L로 나눈다. 따라서 새 길이 L′에서 본 가장 큰 거리는 기존 길이 L과 같다.

**수식.** f′(x, m) = f(x, m·L/L′) ([[position-interpolation]] Eq. 4). x는 token representation, m은 position, L은 pretrained window, L′은 목표 window다.

**계산 예.** L = 2,048, L′ = 8,192, s = 4이면 position 6,000은 1,500으로 매핑된다. 모든 차원의 frequency가 1/4배가 된다. 인접 token 사이 거리를 표현하는 high-frequency 차원도 예외가 없으므로, 원래 1 token 떨어진 이웃은 0.25 position 떨어진 것으로 처리된다.

**근거.** LLaMA 7B–65B는 Pile에서 fine-tuning 1,000 step 이내에 최대 32,768-token window에 도달한다. PI를 쓰면 7B와 33B는 200 step 뒤 target passkey window에 도달하지만, PI 없이 fine-tuning한 7B의 effective window는 10,000 step 뒤에도 2,048에서 2,560으로만 늘어난다([[position-interpolation]] Table 4).

**짧은 컨텍스트 비용.** 원래 2,048 window 안에서 LLaMA 7B BoolQ는 L′ = 8,192 / 16,384 / 32,768에서 76.1에서 각각 73.2 / 69.8 / 64.7로 떨어지고, WinoGrande는 69.6에서 69.0 / 67.8 / 66.9로 떨어진다([[position-interpolation]] Table 5; 단일 연구 결과). 손실은 s와 함께 커진다. 저자들은 position이 더 좁은 범위로 압축되기 때문이라고 해석한다(§3.2).

### 2.2 NTK-aware scaling(community origin)

**상태.** NTK-aware scaling은 처음 Reddit post로 공개됐다. 이 course는 해당 post를 읽지 않았고, 내용은 YaRN 논문과 YaRN 저자의 write-up을 통해 알려져 있다([[yarn]] §1; [[eleuther-extending-the-rope]]). 원 post에 관한 주장은 **일화적**이다. 아래 수식과 실험은 논문에서 온다.

**정의.** position을 나누는 대신 base를 올려 low frequency는 보간하고 high frequency는 거의 바꾸지 않는다.

**수식.** b′ = b·s^(d/(d−2)) ([[yarn]] App. A.2, Eqs. 19, 22), d는 head dimension이다. 차원 i의 frequency multiplier는 s^(−2i/(d−2))가 된다.

**계산 예.** d = 128, b = 10,000, s = 4이면 b′ = 10,000·4^(128/126) ≈ 40,890이다. i = 0의 multiplier는 1이고, i = 63은 4^(−126/126) = 1/4로 PI와 같다. 중간 차원은 두 끝 사이를 기하학적으로 보간한다.

**근거와 한계.** fine-tuning이 없을 때 NTK-aware scaling은 PI보다 긴 sequence perplexity가 낮지만, fine-tuning 뒤에는 PI보다 나쁘다([[eleuther-extending-the-rope]]; [[yarn]] App. A.2). LLaMA 7B, s = 16, 400 step에서 32K proof-pile perplexity는 PI 3.57, NTK-aware 8.49다([[yarn]] Table 5). YaRN 논문은 일부 차원이 범위 밖으로 extrapolate하므로 s를 목표 scale보다 크게 잡아야 한다고 설명한다(App. A.2, 해석). 통제 연구에서는 length-dependent(dynamic) NTK variant가 fine-tuned 방법 중 32K와 64K에서 가장 높은 RULER 점수를 냈다(§2.8).

### 2.3 NTK-by-parts와 YaRN

**정의.** NTK-by-parts는 차원이 원래 길이에서 몇 회전하는지에 따라 차원별로 보간 여부를 결정한다. YaRN은 NTK-by-parts에 attention temperature를 더한다([[yarn]] §3.2–3.3).

**수식.**
- r(i) = L/λ_i: 원래 L에서 차원 i가 도는 횟수(Eq. 10).
- γ(r) = 0 (r < α), 1 (r > β), 그 외에는 (r − α)/(β − α)(Eq. 11).
- θ′_i = (1 − γ(r(i)))·θ_i/s + γ(r(i))·θ_i (Eqs. 12–13).
- Attention은 softmax(q_mᵀk_n / (t·√d))이고, √(1/t) = 0.1·ln(s) + 1이다(Eqs. 14–15).

L은 원래 길이, λ_i는 wavelength, α·β는 rotation 단위 ramp limit이다. Llama family 권장값은 α = 1, β = 32이며 “실험적으로 찾았다”. s는 scale factor, t는 temperature, q_m·k_n은 m·n 위치의 query·key다.

**계산 예.** Llama 2에서 L = 4,096, d = 128, b = 10,000, s = 4라 하자. 차원 0은 λ = 6.3, r = 652 > 32이므로 γ = 1, frequency가 그대로다. 차원 21은 λ = 129.0, r = 31.75, γ = 0.992다. 차원 30은 λ = 471, r = 8.69, γ = (8.69 − 1)/31 = 0.248, multiplier = 0.752·0.25 + 0.248 = 0.436이다. 차원 46은 λ = 4,712, r = 0.87 < 1이므로 γ = 0, multiplier = 1/4다. 총 21 pair는 유지되고 18 pair는 4로 나뉘며 25 pair는 ramp에 놓인다. temperature factor는 0.1·ln 4 + 1 = 1.139다(수식에서 유도; 그림은 이 count를 재현한다).

**코드.** Hugging Face Transformers는 dimension index 위에 ramp를 구현하고 temperature를 cos·sin의 factor로 반환한다([[hf-transformers-yarn-code]], `modeling_rope_utils.py` v4.46.0 lines 188–237).
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
factor는 config load 시 고정되어 input length에 의존하지 않는다. 이것이 Qwen3 model card가 경고하는 “static YaRN”이다(§3.3).

**근거.** 64K-token PG19 chunk에서 s = 16, 400 step 뒤 s = 32, 200 step으로 확장한 Llama 2 7B·13B는 128K까지 passkey accuracy 99.4%에 도달한다. fine-tuning data는 pretraining data의 약 0.1% 미만이다([[yarn]] §1, §4.1, Table 9). s = 16 LLaMA 7B에서 YaRN의 32K perplexity는 네 방법 중 가장 낮은 2.77이며, NTK-by-parts 2.81, PI 3.57이다(Table 5).

**짧은 컨텍스트 비용.** s = 32에서 Llama 2 13B MMLU는 55.8에서 51.9, 7B는 43.8에서 41.7로 하락한다(Table 3). s = 16, 400 step LLaMA 7B의 MMLU는 base 35.7, PI 25.9, NTK-aware 27.7, NTK-by-parts 32.7, YaRN 30.0이다(Table 2). 이 논문에서 YaRN은 긴 컨텍스트 perplexity는 최고이나 MMLU는 최고가 아니다. fine-tuning data가 PG19 book뿐이었고, 저자들은 PG19와 pretraining data의 차이가 variance 일부를 설명한다고 본다(§4.4, 해석).

### 2.4 지속 사전학습의 base-frequency 증가(ABF)

**정의.** Adjusted base frequency(ABF)는 position을 나누지 않고, 긴 길이에서 continued pretraining 전에 b를 올린다.

**계산 예.** d = 128에서 b를 10,000에서 500,000으로 올리면 가장 긴 wavelength는 약 54,410에서 약 2,559,000 token으로 늘어난다(2π·b^(126/128), 유도). i > 0의 모든 차원 frequency는 낮아져 token당 회전각이 작아진다.

**근거.** Llama 2 Long은 Llama 2를 32,768(7B/13B) 또는 16,384(34B/70B) token에서 4,000억 token 지속 사전학습하고 b를 10,000에서 500,000으로 올린다([[llama-2-long]] §2.1). 32,768에서 800억 token으로 한 7B ablation에서는 ABF만이 FIRST-SENTENCE-RETRIEVAL을 32,768까지 유지하며, 다섯 short task 모두에서 PI보다 약간 높다(MMLU 46.24 대 45.84, HumanEval 17.07 대 15.24; Table 6, Fig. 5b). 4,000억 extra token 뒤 모든 size의 short-task score가 Llama 2와 같거나 높다(70B MMLU 68.9 → 71.7; Table 1). 저자들은 extra compute와 새 data 때문이라고 하므로(§3.1), ABF 자체의 비용이 0임을 보여주지는 않는다.

**현재 공식 사용.** Llama 3는 처음 사전학습부터 θ = 500,000을 쓰며 Xiong 등을 인용해 32,768까지 효과적이라고 하고, 여섯 extension stage 중 추가 base 변경은 보고하지 않는다([[llama-3]] §3.2, Table 3; §3.4.2). Qwen2.5-1M은 stage별로 10,000 → 1,000,000(32,768), 이어 1M / 5M / 10M(65,536 / 131,072 / 262,144)을 쓴다([[qwen-long-context-synth]] §3). ProLong은 64K에서 8×10⁶, 512K에서 1.28×10⁸을 쓰며, 64K long-context average는 base 5×10⁵에서 29.1, 4×10⁶에서 48.7, 8×10⁶에서 54.6이다([[prolong]] App. B.1, Table 18).

### 2.5 RoPE base의 하한

**정의.** Men 등은 B_{m,θ} = Σ_{i=0}^{d/2−1} cos(m·θ_i)를 정의한다. i.i.d. 가정 아래 이는 거리 m에서 비슷한 key가 random key보다 갖는 기대 attention advantage다([[rope-base-bounds-context-length]] Theorem 1, Eq. 9).

**수식.** Obtainable context는 L_θ = sup{L | 모든 m ≤ L에 대해 B_{m,θ} ≥ 0}(Eq. 10)다. lower-bound base_L은 L까지 B가 음수가 아니게 하는 가장 작은 base다(Eq. 11). 저자들은 stacked layer가 정보를 더 멀리 전달할 수 있으므로 bound가 “그다지 엄격하지 않다”고 한다(§4.3).

**계산 예.** d = 4이면 θ_0 = 1, θ_1 = b^(−1/2) 두 frequency가 있다. b = 100이면 θ_1 = 0.1이고 m = 2에서 B = cos 2 + cos 0.2 = 0.564, m = 3에서 −0.035이므로 L_θ = 2다. b = 10,000이면 θ_1 = 0.01이고 m = 3에서 B = 0.010, 최초 음수는 m = 22이므로 L_θ = 21이다(직접 계산). 큰 base는 B를 더 오래 양수로 유지한다.

**근거.** 수치 하한(Table 2)은 4K에 2.7×10⁴, 32K에 6.4×10⁵, 128K에 7.8×10⁶, 1M에 5.1×10⁸이다. table 옆에 head dimension은 적혀 있지 않다. base 500으로 32K fine-tuning한 Llama2-7B는 128K까지 낮은 perplexity를 유지했지만 “1k만큼 짧은 context에서도 관련 정보를 retrieve할 수 없었다”(§3, Fig. 3). base 100으로 4,096에서 1T token 사전학습한 2B model은 마지막 약 500 token에서만 retrieve했다(§5.3, Fig. 7).

**한계.** 이 bound는 single-layer 분석이다. 128K Llama 3 base 5×10⁵는 128K bound 7.8×10⁶보다 작다. report는 extension stage에서 다른 position-encoding 변경과 length별 retrieval 분석을 보고하지 않으므로 bound를 확인하거나 반박하지 않는다. ProLong base ablation(§2.4)은 큰 base일수록 long-context score가 높다는 방향과 일치하지만 이를 시험하도록 설계되지 않았다.

**함의.** 목표 길이 perplexity만으로 extension run을 확인할 수 없다. 여러 길이의 retrieval 또는 task 평가가 필요하다([[ch-32c]]).

### 2.6 LongRoPE2: 검색한 차원별 factor와 mixed-window 학습

**메커니즘.**
1. d_tcd를 계산하고, L_train에서 theoretical period 10개인 차원과 d_tcd 사이에서 더 낮은 critical dimension d_rcd를 검색한다.
2. i ≥ d_rcd 차원에 대해 [L/L_train, 2·L/L_train]의 scale factor를 population 64, 40 iteration evolutionary search로 찾고 non-decreasing 제약을 둔다. 그 아래 차원은 NTK scaling을 쓴다.
3. 긴 문서 처음에 needle을 두고 끝에서 묻는 answer token만의 perplexity, 즉 “needle-driven” perplexity로 candidate를 score한다(10개 PG19 validation book).
4. 10B token으로 두 종류의 128K segment를 mid-train한다. 짧은 문서는 original RoPE와 cross-document attention mask로 pack하고, 긴 문서는 rescaled RoPE를 쓴다. 추론 때 input length별로 RoPE를 전환한다([[longrope2]] §3.2–3.3, App. B).

**근거.** 같은 mid-training에서 128K로 확장한 LLaMA3-8B의 RULER 128K는 LongRoPE2 82.03, LongRoPE 73.40, NTK 73.19, YaRN 49.39다(Table 2). 4,096-context short-benchmark average는 LLaMA3-8B에서 original 56.5, YaRN 52.1, NTK 54.0, LongRoPE 54.6, LongRoPE2 55.7이고, Phi3-mini에서는 63.2, 53.6, 57.3, 58.5, 61.7이다(Table 4). retention은 각각 98.6%, 97.6%다(§4.2). mixed-window 학습을 빼면 Phi3-mini MMLU는 70.07에서 66.56으로 떨어진다(Table 7). needle perplexity를 일반 PG19 perplexity로 바꾸면 LLaMA3-8B RULER-128K는 82.03에서 78.68로 내려간다(Table 6).

**한계.** base model 두 개, target length 하나, post-training 없음, repeated run 없음(§4.1–4.2). 추론 시 RoPE 전환에는 KV cache를 한 번 재계산해야 한다(App. B).

### 2.7 추론 전용 확장: Dual Chunk Attention(DCA)

**정의.** DCA는 weight를 바꾸지 않는다. 추론에서 position index를 remap하여 어떤 query-key 거리도 pretraining length c를 넘지 않게 한다([[dual-chunk-attention]] §3).

**메커니즘.** (1) sequence를 크기 s chunk로 나눈다(Llama 2에서는 s = 3,072, 4,096 training length의 3/4). (2) chunk 안 query·key는 0 … s−1 position을 쓴다. (3) 두 chunk 이상 앞 key에는 모든 query에 c − 1 position을 주어 거리를 c − s에서 c − 1로 만든다. (4) 바로 앞 chunk에는 처음 w query position을 s, s+1, …, s+w−1로 두며 w = c − s다. 그래서 가장 가까운 w key는 작은 거리를 유지한다.

**계산 예.** 논문 Fig. 2에서 c = 10, s = 6, w = 4이면 successive-chunk query position은 [6, 7, 8, 9, 9, 9, 6, 7, 8, 9, 9, 9]다. chunk 1 마지막 key(position 5)와 chunk 2 첫 query(position 6)의 거리는 원래 sequence처럼 1이다.

**근거.** DCA Llama2 70B는 학습 없이 PG19 perplexity가 4,096에서 5.24, 65,536에서 5.59이며(Table 1), chat model은 16K에서 논문의 long-context research benchmark에 대해 gpt-3.5-turbo-16k의 94%에 도달한다(§4.2). Qwen2.5-1M은 262K-trained model을 1M token에서 실행하려 DCA와 YaRN attention scaling을 결합하며, 두 방법 모두 training length 안 동작은 바꾸지 않는다고 보고한다([[qwen-long-context-synth]] §5.1).

**한계.** DCA 논문은 DCA를 켠 short-benchmark 점수를 보고하지 않는다. 32K에 DCA + YaRN으로 학습한 Qwen2.5-72B-Instruct는 모든 길이에서 LV-Eval의 1M-trained 14B보다 높았다([[qwen-long-context-synth]] §6.1). 따라서 큰 모델의 inference-only 방법도 일부 benchmark에서는 학습된 작은 모델과 경쟁할 수 있다.

### 2.8 방법의 통제 비교

[[controlled-long-context-extension]]은 NTK-64K를 제외한 모든 fine-tuned 방법에 하나의 LLaMA2-7B base를 사용하고 length-upsampled SlimPajama 10억 token을 32K에서 학습한다. NTK-64K는 64K에서 학습한다. 32K / 64K RULER average는 Dynamic NTK(NTK-32K) 59.42 / 46.26, PI 57.66 / 0.00, YaRN 36.95 / 0.00, LongLoRA(approximate attention, fine-tuned) 3.53 / 0.00이다(Table 3). exact-attention 방법은 training length 안에서 작동하고 NTK와 CLEX만 그 밖으로 generalize한다(§5.2). short-context signal은 2K PG19 perplexity가 모든 fine-tuned exact method에서 상승하는 것이다(base 6.61; NTK-32K 6.63, YaRN 6.70, PI 6.88; Table 2). LongBench code completion LCC는 68.22에서 NTK-32K 56.78, PI 55.05, YaRN 54.06으로 하락한다(Table 11). 반면 4K RULER에서는 하락이 보이지 않는다(base 80.94; NTK-32K 86.58, PI 84.56; Table 3). 따라서 short-context signal은 metric에 따라 다르다. 이 연구는 base model 하나와 training length 하나를 썼다. YaRN 결과는 다른 data·scale factor를 쓴 YaRN 논문 비교와 다르며, 이는 recipe 세부가 method ranking을 바꾼다는 해석을 지지한다.

### 방법 요약

| 방법 | 바뀌는 것 | 학습 필요 | 측정된 짧은 컨텍스트 효과(설정) |
|---|---|---|---|
| PI | 모든 frequency × 1/s | 약 200–1,000 step | s = 16에서 7B BoolQ 76.1 → 64.7 ([[position-interpolation]] T5) |
| NTK-aware | base b′ = b·s^(d/(d−2)) | 없음, 또는 fine-tune(후자는 PI보다 나쁨) | s = 16에서 7B MMLU 35.7 → 27.7 ([[yarn]] T2) |
| YaRN | 차원별 ramp + temperature | 논문에서 400–600 step | s = 32에서 13B MMLU 55.8 → 51.9 ([[yarn]] T3) |
| ABF | base 증가, continued pretraining | Llama 2 Long에서 800억–4,000억 token | 4,000억 뒤 short task가 같거나 높음([[llama-2-long]] T1) |
| LongRoPE2 | 검색 factor + mixed window | 100억 token | LLaMA3-8B retention 98.6% ([[longrope2]] T4) |
| DCA | 추론에서 position remapping | 없음 | 미보고([[dual-chunk-attention]]) |
| STRING | 추론에서 position shift, L 안에서 | 없음 | 미보고([[string-effective-context]]) |

## §3 공식 보고서의 hybrid 설계

### 3.1 Olmo 3: full-attention layer에만 YaRN 적용

Olmo 3는 4개 layer 중 3개에 4,096-token window의 sliding-window attention을 쓰고, 나머지 layer에는 full attention을 쓴다([[olmo-3-long-context]] §3.2). 8,192 token에서 65,536 token으로 확장할 때 team은 base-frequency scaling, PI, YaRN을 각각 모든 RoPE layer에 적용하거나 full-attention layer에만 적용해 보았고, “YaRN을 full attention layer에만 적용하는 방식이 전체 성능이 가장 좋다”고 찾았다(§3.6.4, Fig. 13a; 정확한 값은 인쇄되어 있지 않다). OLMo-core commit 66f768b의 release script(`src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py`)는 다음과 같다.

```python
DEFAULT_SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 65536 * 64  # ~4M tokens
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
    )
        generate_doc_lengths=True,  # enables intra-document masking
```

`with_rope_scaling(..., full_attn_layers_only: bool = True)`는 sliding-window layer의 RoPE를 사전학습된 상태 그대로 둔다. sliding-window layer는 4,096보다 큰 거리를 결코 보지 않으므로, 고칠 out-of-range position이 없다(해석).

### 3.2 Gemma 3: local:global layer와 global layer의 interpolation

Gemma 3는 global layer 하나당 local layer 5개(1,024-token window, RoPE base 10K)를 사용하고, global layer의 base는 1M이다. 32K에서 pretrain한 뒤 4B, 12B, 27B model을 “RoPE를 rescale하면서 pre-training 끝에서” 128K로 확장하며, “scaling factor 8”을 사용한다([[gemma-3-long-context]] §2, §5.3). 보고서가 밝힌 설계 목표는 KV-cache memory다. 2B model에서 32K pre-fill을 할 때 global-only attention의 memory overhead는 60%인 반면, 1,024 window의 1:3 비율은 15% 미만이다(Fig. 5). ratio와 window가 perplexity에 미치는 영향은 작다(Figs. 3–4). Model은 “128K까지 generalize하지만, 계속 scale하면 빠르게 degrade한다”(Fig. 7). 27B pretrained model의 128K RULER는 72.9이고 instruction tuning 뒤에는 66.0이다(Table 15). 보고서는 128K extension이 short-context에 미친 효과를 분리하지 않는다.

### 3.3 Qwen3: 학습 중 ABF, 추론 중 YaRN과 DCA, 그리고 model-card 경고

Qwen3의 마지막 pretraining stage는 32,768 token에서 “수천억 token”을 학습한다. long corpus의 75%는 16,384–32,768 token이고 25%는 4,096–16,384 token이다. ABF로 base를 10,000에서 1,000,000으로 올리고, 추론에서 “sequence length capacity를 네 배 늘리기” 위해 YaRN과 DCA를 사용한다([[qwen-3-long-context]], report §3.2). Qwen3-8B model card에는 다음이 추가돼 있다.

> 주목할 만한 모든 open-source framework는 static YaRN을 구현한다. 이는 input length와 무관하게 scaling factor가 일정하게 유지됨을 뜻하며, **짧은 text의 성능에 영향을 줄 가능성이 있다.** long context를 처리해야 할 때에만 `rope_scaling` configuration을 추가할 것을 권한다.

또한 “평균 context length가 32,768 token을 넘지 않는다면 YaRN을 켜는 것을 권하지 않는다”고 한다. Card는 short-text loss의 수치를 주지 않는다. 메커니즘은 §2.3의 worked example에서 보인다. static factor는 original length보다 짧은 input을 포함한 모든 input에 ramp를 적용하는 반면, dynamic scaling은 s = max(1, l′/L)를 사용하므로 short input은 바꾸지 않는다([[yarn]] §3.4; [[eleuther-extending-the-rope]]는 fixed factor에 대해 “L보다 작은 length에서 성능이 평평하게 감소한다”고 적는다).

### 3.4 범용 모델에 주는 함의

이 보고서들은 같은 설계 원칙을 공유한다. 필요한 곳에만 position change를 적용하는 것이다. Olmo 3는 YaRN을 full-attention layer로 제한하고, Gemma 3는 global layer만 바꾸며, LongRoPE2는 short input에 original RoPE를 유지하고, Qwen은 long input에만 YaRN을 켜라고 권한다. 각각의 선택은 short-context computation을 pretrained model에 더 가깝게 둔다. 이는 보고서들에 공통인 설계 선택이지, 원칙 자체를 통제 비교한 결과는 아니다(해석).

## §4 확장이 short-context에 미친 측정 효과

| 설정 | short-context 측정 | 이전 → 이후 | 출처 |
|---|---|---|---|
| LLaMA-2 7B / 13B, 80K / 64K에서 50억 token, source별 upsampling | MMLU | 이후 43.3(7B), 52.4(13B); 이 표에는 base가 없음 | [[fu-2024-data-engineering-128k]] Table 3 |
| Xiong et al.가 보고한 Llama 2 7B / 13B base | MMLU | 45.3(7B), 54.8(13B) | [[llama-2-long]] Table 1 |
| Llama-3-8B + 학습 없는 base 변경(PE) | MMLU / GSM8K | 66.5 / 44.7 → 64.7 / 40.1 | [[prolong]] Table 2 |
| Llama-3-8B + Fu et al. SlimPajama long mixture | MMLU / GSM8K | 66.5 / 44.7 → 63.1 / 40.6 | [[prolong]] Table 2 |
| SFT 뒤 ProLong-8B vs Llama-3-8B-Instruct | 5-task 평균; MMLU; GSM8K | 69.8 → 69.4; 67.0 → 64.6; 68.5 → 58.9 | [[prolong]] App. B.7, Table 25 |
| Llama-3-8B, 32K ABF, 약 10억 token long-only CPT | 17-benchmark 평균 | 55.16 → 51.00 | [[longred]] Table 3 |
| Mistral-7B-v0.3, 128K ABF, long-only CPT | 17-benchmark 평균; MMLU | 51.09 → 40.68; 62.30 → 51.90 | [[longred]] Tables 3, 14 |
| Qwen2.5-14B-Instruct 128K → 1M | MMLU-Pro; GPQA; LiveCodeBench; IFEval | 63.7 → 63.3; 45.5 → 39.9; 42.6 → 38.6; 81.0 → 84.3 | [[qwen-long-context-synth]] Table 6 |
| Olmo 3 7B Stage 2 → Stage 3(500억 token) | Math / Code / GenQA aggregate | 59.8 → 54.4; 31.9 → 30.6; 71.3 → 72.5 | [[olmo-3-long-context]] Table 13 |
| Olmo 3 32B Stage 2 soup → Stage 3(1,000억) | Math / Code / GenQA | 69.7 → 61.4; 39.7 → 39.7; 79.4 → 79.7 | [[olmo-3-long-context]] Table 13 |
| Llama 2 70B → Llama 2 Long 70B(4,000억 token) | MMLU | 68.9 → 71.7 | [[llama-2-long]] Table 1 |

**Worked example(논문 간, 도출값).** Xiong et al.의 Llama 2 baseline과 Fu et al.의 확장 뒤 점수를 사용하면 7B는 45.3 → 43.3(−2.0), 13B는 54.8 → 52.4(−2.4)가 된다. 두 논문이 서로 다른 MMLU harness를 사용했을 수 있으므로 이 delta는 참고용일 뿐이다. Fu et al.은 이 결과를 “강한 MMLU로 입증된” short-context 성능 유지로 서술한다(§5.1).

**표를 읽는 법.**
1. long-only 또는 long 비중이 큰 data로 짧게 돌린 run이 가장 크게 잃는다. LongReD에서 8B의 평균은 4점, 7B Mistral은 10점 떨어졌고, math 및 multi-step task(GSM8K, GPQA, Olmo 3 Math)는 knowledge 또는 instruction-following 점수보다 더 많이 떨어진다. 이 효과의 방향은 [[prolong]], [[longred]], [[qwen-long-context-synth]], [[olmo-3-long-context]]에서 **재현**됐지만 크기는 설정마다 다르다.
2. broad data로 길게 돌린 run은 이득을 낼 수 있다. Llama 2 Long의 4,000억 token은 모든 size에서 MMLU를 올렸다. LongReD 저자들은 1,000억 token보다 많이 학습한 연구는 일부 short ability가 영향받지 않거나 향상됐다고 보고한다고 적는다([[longred]] Limitations). 이 결과에서는 extension stage의 budget과 data breadth가 extension 자체와 confound되어 있다(해석).
3. aggregate는 task-level 손실을 가린다. Qwen2.5-1M 보고서는 short score가 “similar”하다고 부르지만, 14B GPQA는 5.6점 떨어진다([[qwen-long-context-synth]] §6.2, Table 6).

### 4.1 Mechanism: drift와 forgetting

**정의.** LongReD는 short-text degradation을 (1) RoPE 변경 때문에 hidden state와 attention distribution이 달라지는 distribution drift, (2) continued pretraining 중의 catastrophic forgetting으로 나눈다([[longred]] abstract, §3).

**근거.** Drift는 8,192-token SlimPajama sample 1,000개에서 original hidden state와 extended hidden state의 cosine similarity로 측정한다. Llama-3-8B가 128K(base 1×10⁸)일 때 similarity는 RoPE 변경 직후 0.83, continued training 10억 token 뒤 0.94다(Table 1). 확장 model 15개 전반에서 similarity가 높을수록 MMLU retention도 높다. 32K에서 ABF 5×10⁶은 similarity 0.943에서 MMLU의 0.960을 유지하지만, PI×16은 0.862에서 0.866을 유지한다(App. D Table 12). Forgetting의 경우 base 2×10⁷, 32K에서는 short-text score가 첫 step들(대개 32 step 미만) 안에 회복한 뒤 training이 이어질수록 하락한다. long data 절반을 8K short text로 바꾸면 MMLU는 62.0에서 62.5로, HumanEval은 14.02에서 16.46으로 오른다(§3.2, Fig. 3, Table 2).

**함의.** 서로 다른 두 control이 필요하다. Drift는 short-range computation을 덜 바꾸는 position change를 고르거나(§3.4), original model을 향한 distillation으로 줄인다(§6.2). Forgetting은 mixture 안에 short data를 남겨 줄인다(§5.1).

## §5 범용 능력을 유지하는 long-context data mixture와 schedule

### 5.1 long:short 비율

**ProLong.** Llama-3-8B base를 64K에서 50억 token 학습한다. data는 책·repository long data와 ShortMix short data이고, 평가는 UltraChat SFT 뒤에 한다. “처음에는 long data를 더 넣을수록 long-context 성능이 좋아지지만, 어느 지점부터는 방해가 된다. long data를 더 넣을수록 short-context 성능도 일관되게 나빠진다”([[prolong]] Fig. 3). long-context 평균이 가장 높았던 비율은 long 60%였다. 최종 recipe는 code repository 30%, 책 30%, textbook 3%, ShortMix 37%다(Table 9). long data만 100% 썼을 때는 PG19 perplexity가 계속 좋아졌지만 downstream long-context 점수는 떨어졌다(§2.1, Fig. 1).

**Olmo 3.** 100억-token extension에서 long 66% / short 34%는 OlmoBaseEval 일부 집합을 2.5점 낮췄고, long 34% / short 66%는 0.8점 낮췄다. 최종 recipe는 long 34%를 쓴다([[olmo-3-long-context]] §3.6.3).

**Worked example.** 200억-token stage를 ProLong 비율로 구성하면 long data는 120억 token, short data는 80억 token이다. Olmo 3 비율이면 long은 68억 token, short는 132억 token이다. 같은 budget에서도 두 recipe의 long token 수는 약 1.8배 차이 난다. 어느 보고서도 다른 보고서의 비율을 동일 model에서 시험하지 않았다.

**Qwen의 length mix는 다른 양이다.** Qwen2.5-1M은 sequence의 75%를 현재 maximum length, 25%를 그보다 짧은 length로 사용한다([[qwen-long-context-synth]] §3). Qwen3 long corpus는 75%가 16K–32K, 25%가 4K–16K다([[qwen-3-long-context]]). 이는 long-context corpus 내부의 length distribution이며, 일반 short-context data의 비중이 아니다. 따라서 ProLong·Olmo 3 비율과 직접 비교할 수 없다.

**상태.** short data가 많을수록 short-context 손실이 줄어든다는 방향은 **재현됨(Replicated)** 이다([[prolong]], [[olmo-3-long-context]], [[longred]] Table 2). 최적 비율은 **열린 질문(Open question)** 이다. Olmo 3는 base-model short task를, ProLong은 post-SFT long task를 평가했고, short data 자체의 quality도 다르기 때문이다.

### 5.2 어떤 long data와 어떤 short data인가

- **Source balance.** Fu et al.은 SlimPajama의 domain 비율(CommonCrawl 67%, C4 15%, GitHub 4.5%, Wikipedia 4.5%, books 4.5%, ArXiv 2.5%, StackExchange 2.0%)을 유지하면서, 각 domain 안에서 4K보다 긴 document의 비율을 약 30%에서 약 70%로 높인다. 책만 upsample하면 0–4K에서 Book loss는 0.175 낮아지지만 StackExchange와 GitHub loss는 각각 0.021, 0.029 높아진다. source별 upsampling은 어느 domain에서도 유의한 loss 증가가 없다(|Δ| ≤ 0.01 기준; [[fu-2024-data-engineering-128k]] §5.3, Table 5).
- **Source type.** ShortMix를 40%로 고정했을 때 long-context 평균은 books + repositories를 1:1로 섞으면 54.6, books만 쓰면 53.8, code repositories만 쓰면 52.3, ArXiv면 51.4, CommonCrawl이면 50.9다([[prolong]] Table 4). Olmo 3는 gzip compressibility가 가장 높은 20%와 가장 낮은 20% text를 제거해 filtering한 olmOCR science PDF를 쓰고, document 안에 synthetic aggregation task를 삽입한다([[olmo-3-long-context]] §3.6.1–3.6.2).
- **Short data quality.** long data가 같을 때 ShortMix는 long 54.6 / short 65.5를 내며, SlimPajama는 52.9 / 64.2, FineWeb-Edu는 53.0 / 63.0, DCLM-Baseline은 52.0 / 64.8이다. 원래 Llama-3-8B의 short 평균은 66.0이다([[prolong]] Table 6). 동일 budget에서 Fu et al. mixture를 재현한 ProLong 결과는 자체 mixture의 long 54.6, SFT 뒤 short 67.5에 비해 long 51.8, SFT 뒤 short 65.4다(App. B.6, Table 24).
- **길이 비중보다 quality.** Llama 2 Long의 7B ablation에서 Llama 2 mixture에서 long text를 제거하고 continued pretraining하면 Llama 2 대비 NarrativeQA / Qasper / QuALITY가 19.48% / 39.14% / 67.1% 개선된다. Llama 2 Long mixture는 23.70% / 43.64% / 75.5%, Llama 2 mixture에서 long text만 upsample한 경우는 22.15% / 36.82% / 65.0%다. 저자들은 자기 mixture의 이득이 주로 length distribution이 아니라 data quality에서 온다고 결론낸다([[llama-2-long]] §4.2, Tables 7–8; 해석).

### 5.3 short-context 회복으로 gate를 거는 staged schedule

Llama 3 405B는 약 8,000억 token에 걸쳐 8K에서 128K까지 여섯 stage로 확장한다. 보고서는 성공적인 adaptation의 gate를 다음처럼 밝힌다. “(1) short-context evaluation에서 model 성능이 완전히 회복됐는지, (2) 그 length까지 ‘needle in a haystack’ task를 완벽하게 푸는지를 측정한다”([[llama-3]] §3.4.2). stage별 length와 token count는 공개하지 않았다([[llama-3-recipe]]). 확장 뒤 final annealing phase는 128K context에서 upsample한 high-quality data와 checkpoint averaging을 사용해 4,000만 token 동안 learning rate를 0까지 낮춘다(§3.4.3).

Qwen2.5-1M은 다섯 stage(4,096 → 32,768 → 65,536 → 131,072 → 262,144)를 쓴다. stage별 base는 32,768과 65,536에서는 1M, 그 뒤에는 5M과 10M이다. 14B model의 128K RULER는 stage 2–5에서 37.6 → 56.0 → 83.8 → 87.6으로 올라가므로, 262K stage도 128K accuracy를 계속 개선했다([[qwen-long-context-synth]] Table 2). ProLong의 512K stage는 64K에서 평가했을 때 equal-token 64K continuation보다 recall 98.5 대 95.0, re-ranking 32.9 대 28.0으로 높다([[prolong]] Table 7).

**함의.** 두 보고서 모두 target evaluation length보다 길게 학습하면 도움이 됐다고 보인다. short-context evaluation을 포함하는 stage gate는 regression을 사후 발견이 아니라 stopping criterion으로 만든다. Llama 3 보고서는 각 gate까지 필요한 token 수를 공개하지 않는다.

### 5.4 Document masking과 packing

- **Llama 3:** “같은 sequence 안의 서로 다른 document 사이 self-attention을 막는 attention mask를 사용한다. standard pre-training에서는 이 변경의 영향이 제한적이었지만, 매우 긴 sequence에서 continued pre-training할 때는 중요했다”([[llama-3]] §3.2; arXiv:2407.21783v3에서 인용).
- **ProLong:** document 사이 attention을 mask하면 long 54.6 / short 65.5이고, mask하지 않으면 53.6 / 64.9다([[prolong]] App. B.2, Table 20).
- **Olmo 3:** intra-document masking과 best-fit packing을 사용해 split document를 줄인다. document packing은 “더 긴 context length에서 성능을 높인다”(Fig. 13d; [[olmo-3-long-context]] §3.6.4).
- **Fu et al.:** document boundary와 무관하게 mask 없이 80K까지 pack하고도 7B에서 NIAH 88.0을 보고한다([[fu-2024-data-engineering-128k]] §4, Table 3). 따라서 retrieval을 위해 masking이 필수는 아니지만, ProLong ablation은 long·short 점수 모두에서 도움이 됨을 보인다.

### 5.5 Pretraining 중 context-window schedule(SkyLadder)

SkyLadder는 data가 아니라 attention window를 바꾼다. 식은 \(w(t) = \min(w_e, w_s + \lfloor\alpha t\rfloor)\)이며, \(t\)는 training step, \(w_s\)는 시작 window, \(w_e\)는 목표 window, \(\alpha\)는 step당 늘어나는 token 수다([[skyladder]] §4.1). \(w_s = 32\), \(\alpha = 1/8\)이면 step 8,000에서 \(w = 1,032\) token이고, 8,192에는 약 64K step 뒤 도달한다(§4.2; 도출된 worked example).

고정 1,000억-token budget에서 CommonCrawl과 random packing으로 8K까지 학습한 1B model은 nine-task standard 평균을 46.3에서 50.0으로 높인다(Table 1). 32K에서는 linear schedule이 long / standard 13.5 / 54.3이고, constant 32K window는 9.7 / 50.7, 4K pretraining 뒤 32K continued pretraining 30억 token은 10.0 / 52.9다(Table 7). 근거는 3B 이하 model, 32K 이하 window, setting당 한 run에만 걸친다(App. A.1). released checkpoint를 확장할 때가 아니라 from-scratch training에 적용된다.

### 5.6 Token budget

Fu et al.은 7B에서 NIAH accuracy가 5억 token에 81.1, 50억 token에 88.0, 100억 token에 84.0이라고 보고한다. 100억 token에서는 80K 너머 length generalization도 감소한다([[fu-2024-data-engineering-128k]] Fig. 3). Olmo 3의 RULER는 10억에서 1,000억까지 extension budget이 늘수록 증가하며, 더 긴 length에서 증가폭이 더 크다(Fig. 13e). ProLong의 400억 token은 약 8,000억 long-context token을 쓴 Llama-3.1-8B-Instruct보다 HELMET 128K 평균이 49.4 대 46.5다([[prolong]] Table 10). base model, target length, evaluation이 달라 report 사이 budget은 비교할 수 없다.

## §6 short-context regression 완화책

### 6.1 고품질 short-data replay

Replay는 §5.1에서 다룬 mixture control이다. 통제된 setting에서 효과의 크기는 다음과 같다. LongReD는 long data의 절반을 8K short text로 바꾸어 PIQA를 74.10에서 78.24로, TriviaQA를 70.67에서 72.82로 높였다(Table 2). ProLong의 ShortMix와 SlimPajama 차이도(short 평균 65.5 대 64.2) replay의 **비율**만큼이나 replay **source**가 중요함을 보인다([[prolong]] Table 6).

### 6.2 복원 distillation(LongReD)

**수식.** 최종 loss는 \(L_{\text{final}} = L_{\text{long}} + \alpha_1\cdot L_{\text{short}} + \alpha_2\cdot L_{\text{s2l}}\)이다([[longred]] Eq. 18). 각 항은 다음과 같다.

- \(L_{\text{long}}\)은 long text에서의 language-model loss다.
- \(L_{\text{short}} = -\Sigma\,\operatorname{Sim}(H_l(\text{extended}), H_l(\text{original}))\)은 normal position을 쓴 short text(1,024 token)에서 선택된 layer들에 걸쳐 합산한다(Eq. 8). 선택 layer는 두 model 사이 attention KL이 가장 큰 layer들과 마지막 layer다.
- \(L_{\text{s2l}}\)은 position index를 건너뛴 short text에서 extended model의 마지막-layer hidden state를, normal position을 쓴 같은 text에서 original model의 hidden state에 맞춘다(Eq. 17).
- \(\operatorname{Sim}\)은 cosine similarity이며, \(\alpha_1\), \(\alpha_2\)는 weight다. 32K에서는 각각 5와 10, 128K에서는 각각 2와 15를 쓴다(App. B.3).

**계산 예.** 32K ABF의 Llama-3-8B에서 long-only CPT는 원래 short-text 평균의 \(51.00 / 55.16 = 92.5\%\)를 유지한다. LongReD-C는 \(54.85 / 55.16 = 99.4\%\)를 유지하며 RULER도 82.80보다 높은 84.98이다(Table 3). 이 표의 mixed-length CPT short 평균은 45.86인데, 논문이 설명하지 않는 거의 0인 SQuADv2와 DROP score가 평균을 낮춘다(Table 16).

**한계.** 128K Llama-3-8B에서 LongReD의 RULER는 CREAM sampling이면 64.93, uniform이면 68.41이고 long-only CPT는 69.70이다. 즉 short-context 유지와 맞바꾸어 long-context score 일부를 잃는다. 128K Mistral-7B-v0.3에서는 두 score를 모두 높이지만 short 평균에는 여전히 3.4–4.8 point의 gap이 남는다(Table 3). 모든 run은 약 10억 token을 사용한다.

### 6.3 short-to-long preference optimization(LongPO)

LongPO는 continued pretraining이 아니라 이미 alignment된 instruct model에 적용한다. preference pair도 model 자신으로부터 만든다. chosen response는 관련 chunk만 주었을 때 model이 낸 답이고, rejected response는 전체 long document를 주었을 때 낸 답이다. KL reference는 chunk에 조건부인 model이며, chosen long sequence에 대한 NLL term을 추가한다([[longpo]] §3, Eqs. 12–14).

Mistral-7B-Instruct-v0.2에서 self-generated sample 45K개를 사용하면 InfiniteBench는 13.82에서 39.27로 오르고 MMLU는 59.15에서 59.99로 움직인다. 같은 data로 SFT와 DPO를 하면 InfiniteBench는 각각 30.03과 25.56이다(Tables 1, 3). 저자들은 SFT와 DPO가 대부분의 task에서 short-context task를 10–20 point 떨어뜨리지만 LongPO는 그렇지 않다고 보고한다(§1, Fig. 3). 근거는 7B instruct model에만 해당한다. negative term의 상세는 다음 절에서 다룬다.

## §7 어느 post-training 단계가 long context를 맡는가

보고서들은 서로 다른 결론을 내리며, library 안에는 그 불일치를 해결하는 controlled study가 없다.

- **Llama 3: long SFT data가 필요하다.** “기존 SFT recipe를 short-context data만으로 순진하게 적용하면 pre-training에서 얻은 long-context capability가 크게 퇴행했다.” 해결책은 synthetic long-context QA, hierarchical summarization, repository code reasoning이며, 이를 16K–128K bucket으로 나눈다. “synthetically generated long-context data 0.1%를 original short-context data와 섞는 것이 short-context와 long-context benchmark 모두에서 성능을 최적화한다.” short data만으로 한 DPO는 “SFT model이 long-context task에서 고품질인 한 long-context performance에 부정적 영향을 주지 않았다”([[llama-3]] §4.3.4; 표는 인쇄하지 않음).
- **ProLong: short SFT data만으로 충분하다.** Long continued training 400억 token 뒤 UltraChat 10억 token만으로 SFT하면 long-context 평균은 55.7이다. synthetic long instruction data를 token의 1 / 3 / 10 / 50%로 추가하면 8B generator에서는 각각 54.1 / 53.5 / 53.9 / 43.3이며, 70B generator에서도 어느 비율도 0%를 이기지 못한다([[prolong]] Table 8, App. B.5 Table 23).
- **Qwen2.5-1M: long SFT, short RL.** SFT stage 2는 최대 262,144 token의 short·long data를 섞는다. 이어지는 offline RL은 8,192 token 이하의 pair만 사용하지만 LongBench-Chat을 계속 높인다(7B 7.32 → 8.08; 14B 8.56 → 8.76; [[qwen-long-context-synth]] §4, Table 3).
- **Gemma 3와 Qwen3: post-training이 일부 long-context score를 낮춘다.** Gemma 3 27B의 RULER-128K는 pretrained일 때 72.9이고 instruction-tuned 뒤에는 66.0이다([[gemma-3-long-context]] Table 15). Qwen3-235B-A22B의 RULER 평균은 non-thinking mode 95.0, thinking mode 92.2다([[qwen-3-long-context]] Table 23).

**열린 질문(Open question).** ProLong 저자들은 Llama 3와의 충돌에 두 가설을 제시한다. 앞선 연구가 long-context continued training을 너무 적게 했거나, 훨씬 큰 short instruction set을 사용했을 수 있다는 것이다([[prolong]] §5). 두 가설 모두 검증되지 않았다. 한 가지 그럴듯한 조정 설명은 short data로 수행하는 SFT optimizer step 수가 많을수록 long SFT data의 필요가 커진다는 것이다. 이는 short-only DPO가 안전했던 이유에 대한 Llama 3 자체 설명과 일관된다(해석, 미검증). 검증되기 전까지 범용 모델 run은 extension 뒤에만 평가하지 말고, 모든 post-training stage 뒤에 long-context task를 평가해야 한다.

## §8 long-context stage 요약: 배치, gate, 평가 인계

1. **Extension 전.** extension 전 checkpoint에서 short-context panel(knowledge, math, code, instruction following)을 기록한다. 이것이 recovery gate를 판단할 기준점이다.
2. **Position 변경.** continued training에는 ABF 또는 YaRN을 고른다. base가 목표 length의 lower bound를 만족하는지 확인하고([[rope-base-bounds-context-length]] Table 2), perplexity만 보지 말고 retrieval을 검증한다.
3. **Mixture.** 고품질 short data를 37–66% 유지한다. long data는 여러 source에서 가져오고, document masking을 사용한다.
4. **Schedule.** 단계적으로 extension한다. short panel이 회복되고 현재 length의 retrieval을 풀었을 때만 다음 단계로 넘어간다. 목표 length보다 길게 학습하는 것도 고려한다.
5. **Extension 후.** 여러 length의 task suite로 effective length를 평가한 뒤([[ch-32c]]), SFT, preference optimization, RL 뒤에도 long-context score를 다시 확인한다(§7; [[ch-44a]]는 long-context RL을 다룬다).

## Negative samples와 negative feedback

**어떤 종류의 negative인가.** Continued-pretraining extension(§1–§5)은 negative를 사용하지 않는다. 이 장의 두 완화책 중 LongPO만 **gradient로서의 negative**(강의 정의의 type 4)를 쓴다. DPO 형태 항을 통해 rejected response의 likelihood를 낮춘다. LongReD는 negative를 쓰지 않으며, original model과의 similarity가 signal이다.

**Negative의 출처.** LongPO에서 rejected response \(y_L\)은 full long document를 준 model 자신의 greedy answer이고, chosen response \(y_S\)는 관련 chunk만 주었을 때의 answer이다. label은 construction에서만 나온다. 논문은 \(y_L\)이 “낮은 품질일 가능성이 높다”고 가정하지만, 이 가정의 measured error rate는 보고하지 않는다([[longpo]] §3.1, §4.1).

**메커니즘.** 목적함수는 \(L = -\mathbb{E}[\log \sigma(\beta\cdot\log \pi_\theta(y_S|x_L)/\pi_S(y_S|x_S) - \beta\cdot\log \pi_\theta(y_L|x_L)/\pi_S(y_L|x_S))]\)(Eq. 12)이다. 여기서 \(\pi_\theta\)는 policy, \(\pi_S\)는 reference로 쓰는 original short-context model, \(x_L\)은 long document와 instruction, \(x_S\)는 관련 chunk와 동일한 instruction, \(\beta=0.1\), \(\sigma\)는 sigmoid다. sequence-level term에서 token \(y\)의 log probability를 logit \(z_j\)로 미분하면 \(\partial \log p_y / \partial z_j = \mathbf{1}[j=y]-p_j\)다. rejected token의 \(\log p\)를 낮추면 그 logit은 내려가고, 다른 모든 logit은 확률 \(p_j\)에 비례해 올라간다. 따라서 제거된 mass는 이미 가장 likely한 token들로 주로 간다. rejected answer가 이미 \(\pi_\theta\) 아래 unlikely하면 이 push는 현재 top alternative에 mass를 더 집중시키며, 그것은 chosen answer가 아닐 수도 있다(강의 도출; 전체 처리는 negative-feedback 장에 있다).

**LongPO가 사용한 control.** (1) reference \(\pi_S(y|x_S)\)는 필요한 information이 들어 있는 chunk에 condition되어 있으므로, implicit KL은 long-input policy가 short-context model의 behavior에 가깝게 머물게 한다(§3.2, Eq. 10). (2) long input에서 chosen response에 대한 NLL term이 positive를 anchor한다(Eq. 14, preference term의 \(\lambda=0.01\)). (3) data는 학습할 model이 생성하므로 pair는 각 iteration 시작 시점에서 on-policy다(§4.1).

**근거.** LongPO는 RULER-NIAH에서 chosen response SFT와 rejected response SFT를 모두 넘는다(§5.3, Fig. 4; curve만 제시). short-to-long reference가 없는 같은 pair의 DPO는 InfiniteBench 25.56이고 LongPO는 39.27이며, short-context task를 악화시킨다(Table 1, Fig. 3). short-to-long constraint를 넣은 SFT도 short-context performance를 유지했다(§5.3). 따라서 이 논문에서 retention과 연결되는 component는 rejected term만이 아니라 constraint다(해석). rejected term만으로 얻은 gain의 비율은 측정하지 않았다.

**진단.** chosen과 rejected log-probability를 따로 기록한다(논문도 chosen·rejected reward를 기록한다, App. B.1, Fig. 6). 끝에서만 보지 말고 training 중 short-context panel을 추적한다. rejected log-probability가 내려가면서 chosen log-probability도 내려가면, mass가 두 response 어느 쪽으로도 가지 않는다는 신호다.

**범용성에 미치는 효과.** Mistral-7B에서는 모든 short metric이 base의 0.74 point 이내이거나 base보다 높다(Table 3). Qwen2.5-7B에서는 ARC-C가 1.45, MMLU가 0.64 떨어지는 반면 MT-Bench는 7.30에서 7.62로 오른다(Table 3). calibration, hallucination, refusal 측정은 보고하지 않는다.

## 레시피

| Model (정확한 release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| ProLong-64k-Base | 8B | long-context | tokens; max length; RoPE base | 20B; 64K (2¹⁶); 8×10⁶ | arXiv:2410.02660v4 Table 9 ([[prolong-recipe]]) | verified 2026-09-14 | App. B.1 Table 18: 54.6 vs 48.7 (4×10⁶) vs 29.1 (5×10⁵) |
| ProLong-64k-Base | 8B | long-context | data mixture (token share인지 sampling share인지는 미기재) | code repo 30%, books 30%, textbooks 3%, ShortMix 37% | v4 Table 9 | verified 2026-09-14 | Fig. 3(long 60%가 best long avg); Table 4; Table 6 |
| ProLong-64k-Base | 8B | long-context | LR; batch; masking | 1e-5, warmup 10%, cosine으로 1e-6; 4M tokens; cross-document masking | v4 Table 9, App. B.2 | verified 2026-09-14 | Table 20(masking); LR ablation은 미보고 |
| ProLong-512k-Base | 8B | long-context | tokens; length; RoPE base | 20B; 512K; 1.28×10⁸ | v4 Table 9 | verified 2026-09-14 | Table 7(equal tokens에서 512K vs 64K) |
| ProLong-512k-Base | 8B | long-context | peak LR | paper는 stage당 1e-5; `train_512K.sh` default는 5e-6 | v4 Table 9; ProLong repo @499fa29 | conflict | ablation 미보고 |
| ProLong-512k-Instruct | 8B | SFT | data; tokens; LR | UltraChat만; 1B tokens; 2e-5 | v4 Table 9, §5 | verified 2026-09-14 | Table 8: synthetic long 0%에서 55.7, 1%에서 54.1 |
| Llama 3.1 405B | 405B | long-context | stages; tokens; gate | 8K → 128K의 six stages; ≈800B; short eval 완전 회복 및 NIAH 해결 | arXiv:2407.21783v3 §3.4.2 ([[llama-3-recipe]]) | verified 2026-09-14 | ablation 미보고; stage별 값 미출력 |
| Llama 3 (all sizes) | 8B–405B | pretrain-stable | RoPE \(\theta\) | 500,000 | v3 §3.2, Table 3 | verified 2026-09-14 | 32,768 token에 대해 Xiong et al. (2023)을 인용 |
| Llama 3.1 405B | 405B | pretrain-decay/anneal | tokens; context; LR | 마지막 40M tokens; 128K; linear로 0까지 | v3 §3.4.3 | verified 2026-09-14 | ablation 미보고 |
| Llama 3.1 | not scoped | SFT | long-context share | synthetic long-context data 0.1%, 16K/32K/64K/128K bucket | v3 §4.3.4 | verified 2026-09-14 | “careful ablations”; table 없음 |
| Qwen2.5-7B/14B-1M | 7B, 14B | long-context | stage lengths; RoPE bases | 65,536 / 131,072 / 262,144; 1M / 5M / 10M | arXiv:2501.15383v1 §3 ([[qwen-long-context-synth]]) | verified 2026-09-14 | Table 2(14B RULER-128K 56.0 / 83.8 / 87.6) |
| Qwen2.5-7B/14B-1M | 7B, 14B | long-context | length mix | current max length 75%, 그보다 짧은 length 25% | §3 | verified 2026-09-14 | ablation 미보고 |
| Qwen2.5-7B/14B-Instruct-1M | 7B, 14B | preference | pair length | offline RL, pair ≤ 8,192 tokens | §4 | verified 2026-09-14 | Table 3(LongBench-Chat gain) |
| Qwen3 (all sizes) | 0.6B–235B | long-context | length; mix; base; tokens | 32,768; 16,384–32,768 75% / 4,096–16,384 25%; 10,000 → 1,000,000(ABF); “hundreds of billions” | arXiv:2505.09388v1 §3.2 ([[qwen-3-long-context]]) | verified 2026-09-15 | ablation 미보고 |
| Olmo 3 7B | 7B | long-context | mixture; tokens | long 34% / Dolmino short 66%; 50B | arXiv:2512.13961v2 §3.6 ([[olmo-3-long-context]]) | verified 2026-09-15 | §3.6.3: −0.8 vs −2.5 points(10B-token test) |
| Olmo 3 32B | 32B | long-context | tokens; batch | 100B; 8×1024×1024 tokens | §3.6; OLMo-core @66f768b `OLMo-3-1025-32B-long-context.py` | verified 2026-09-15 | Fig. 13e(더 긴 extension이 도움) |
| Olmo 3 7B, 32B | 7B, 32B | long-context | RoPE scaling; length | YaRN factor 8, beta_fast 32, beta_slow 1, old_context_len 8192, full-attention layer만; 65,536 | OLMo-core @66f768b scripts; `transformer/config.py` | verified 2026-09-15 | Fig. 13a(값 미출력) |
| Olmo 3 7B | 7B | long-context | batch; LR; schedule | 65,536×64 tokens; 2.0712×10⁻⁴; 200 steps linear warmup, 0까지 decay | OLMo-core @66f768b `OLMo-3-1025-7B-long-context.py` | verified 2026-09-15(decay horizon 미보고) | ablation 미보고 |
| Gemma 3 4B/12B/27B | 4B–27B | long-context | method | pretraining 끝에 32K → 128K; global layer에서 RoPE rescale factor 8(base 1M); local layer base 10K; tokens 미보고 | arXiv:2503.19786v1 §2, §5.3 ([[gemma-3-long-context]]) | verified 2026-09-15 | “factor of 8 to work well in practice”; table 없음 |
| Fu et al. LLaMA-2 7B | 7B | long-context | tokens; length; LR; batch | 5B; 80K; constant 2e-5; 4M tokens | arXiv:2402.10171v1 §4 ([[fu-2024-data-engineering-128k]]) | verified 2026-09-15 | Fig. 3(500M–10B tokens) |
| Fu et al. LLaMA-2 7B | 7B | long-context | mixture | SlimPajama domain ratio 고정; 각 domain 안에서 >4K share 약 30% → 약 70% | §5.3 | verified 2026-09-15 | Table 5(per-domain loss) |
| LongRoPE2 LLaMA3-8B-128k | 8B | long-context | tokens; masking | 10B; short window는 original RoPE + cross-document mask, long window는 rescaled RoPE | arXiv:2502.20082v1 §3.3, §4.1 ([[longrope2]]) | verified 2026-09-14 | Table 7(without mixed training) |
| LongReD Llama-3-8B 32K | 8B | long-context | \(\alpha_1\); \(\alpha_2\); \(T_s\); tokens | 5; 10; 1,024; 약 1B | arXiv:2502.07365v3 App. B.3, §5.1 ([[longred]]) | verified 2026-09-14 | Table 4(\(\alpha\)), Table 6(\(T_s\)) |
| Llama 2 Long | 7B–70B | long-context | tokens; RoPE base; peak LR | 400B; 10,000 → 500,000; 2e-5(7B/13B), 1e-5(34B/70B) | arXiv:2309.16039v3 §2.1 ([[llama-2-long]]) | verified 2026-09-14 | Tables 5–6, Fig. 5b(7B, 80B tokens) |
| Yarn-Llama-2 \(s=16\) | 7B, 13B | long-context | steps; batch; LR; data | 400; 64; 2×10⁻⁵; PG19 64k chunks | arXiv:2309.00071v3 §4.1 ([[yarn]]) | verified 2026-09-14 | App. B.2 Table 6(400 vs PI 1,000 steps) |
| LLaMA 7B/13B PI | 7B, 13B | long-context | steps; LR | 1,000; 2×10⁻⁵ | arXiv:2306.15595v2 §3.1 ([[position-interpolation]]) | verified 2026-09-14 | Tables 3–4(200 steps가 target 도달) |

**소규모 범용 run의 시작점.** Academic budget으로 8K에서 64K로 확장하는 8B dense RoPE model이라면, ProLong 64K stage가 가장 완전하게 문서화된 verified configuration이다. 64K에서 20B tokens, RoPE base 8×10⁶, code repository 30% + books 30% + textbooks 3% + high-quality short data 37%, cross-document masking, LR 1e-5에서 10% warmup 후 cosine decay로 1e-6, 4M-token batch를 쓴다. Llama-3-8B-Instruct에서 H100으로 실행했으며 2.2K H100 hours가 들었다([[prolong-recipe]]). model이 sliding-window layer를 쓴다면 verified alternative는 Olmo 3 7B configuration이다. full-attention layer에만 YaRN factor 8(beta_fast 32, beta_slow 1)을 적용하고, long 34% / short 66% data, 50B tokens, 약 4M-token batch, intra-document masking을 쓴다. Olmo 3 7B midtraining checkpoint에서 실행했다. 두 경우 모두 Llama 3의 stage gate, 즉 short-context panel이 회복되고 current length에서 retrieval을 풀었다는 조건이 verified stopping rule이며, long-context SFT data는 SFT 뒤 평가해 결정해야 한다(§7).

## 일반화 관점

**(a) Breadth를 늘리는 요인.**
- 한 domain이 아닌 여러 domain의 long data: source별 upsampling은 다른 domain loss 증가를 피하며([[fu-2024-data-engineering-128k]] Table 5), books + repositories는 둘 중 어느 하나만 쓰는 것보다 낫다([[prolong]] Table 4).
- mixture 안의 high-quality short data: ShortMix는 long·short average 모두에서 SlimPajama, FineWeb-Edu, DCLM을 이긴다([[prolong]] Table 6).
- broad data를 쓴 더 큰 continued-pretraining budget: Llama 2 Long의 400B tokens는 모든 size에서 MMLU를 올렸다([[llama-2-long]] Table 1).
- evaluation length를 넘어 학습하기: 262K와 512K stage가 각각 128K와 64K 결과를 높였다([[qwen-long-context-synth]] Table 2; [[prolong]] Table 7).
- 필요한 곳에만 제한한 position 변경: mixed-window training([[longrope2]] Table 7), full-attention layer에만 YaRN 적용([[olmo-3-long-context]] Fig. 13a).

**(b) Narrowing 또는 forgetting을 일으키는 요인.**
- long-only 또는 long-heavy mixture: short task의 monotonic decline([[prolong]] Fig. 3), −2.5 대 −0.8 points([[olmo-3-long-context]] §3.6.3), 92.5% retention([[longred]] Table 3).
- single-domain long data: book upsampling은 code·StackExchange loss를 올리며([[fu-2024-data-engineering-128k]] Table 5), YaRN의 PG19-only fine-tuning은 13B MMLU를 3.9 points 낮췄다([[yarn]] Table 3).
- 큰 interpolation factor: PI BoolQ loss는 \(s\)와 함께 커진다([[position-interpolation]] Table 5).
- short input에서 static inference-time scaling(수치 없는 caution, [[qwen-3-long-context]]).
- 검토한 report에서 math와 reasoning score가 가장 민감했다. Olmo 3 Math aggregate는 −5.4(7B), −8.3(32B), Qwen2.5-14B-1M GPQA는 −5.6, ProLong-8B GSM8K는 SFT 뒤 68.5 → 58.9다. 저자들은 후자를 Llama-3-8B-Instruct의 closed instruction data 탓도 있다고 본다([[olmo-3-long-context]] Table 13; [[qwen-long-context-synth]] Table 6; [[prolong]] Table 25).
- long context를 무시하는 post-training: short-only SFT는 Llama 3 long context를 퇴행시켰고([[llama-3]] §4.3.4), thinking mode는 Qwen3 RULER를 낮춘다([[qwen-3-long-context]] Table 23).

**(c) 이 stage에서의 측정 방법.**
- 각 stage 전후에 short-context panel을 task별로 보고한다. average만 보고하지 않는다(§4 point 3). 가장 많이 떨어진 math와 code를 포함한다.
- perplexity가 아니라 여러 length의 long-context task를 쓴다. 100% long data는 PG19 perplexity를 개선하면서 downstream score를 낮춘다([[prolong]] Fig. 1). 작은 base는 retrieval 없이도 낮은 perplexity를 낸다([[rope-base-bounds-context-length]] Fig. 3). 한 controlled study에서는 exact-attention method 안에서 perplexity가 downstream score를 추적했다([[controlled-long-context-extension]] Fig. 4). 따라서 필요하지만 충분하지 않은 check다.
- SFT 뒤 long-context ability를 평가한다. RAG와 re-ranking gain은 SFT 뒤에만 나타난다([[prolong]] §2.2, Fig. 2).
- held-out long-context suite를 유지한다. Olmo 3는 RULER에서 develop하고 HELMET을 unseen으로 보고한다([[olmo-3-long-context]] §3.6). ProLong은 held-out HELMET task, NoCha, RULER, ∞Bench를 확인한다([[prolong]] Limitations). [[ch-32c]]는 이 suite들의 알려진 measurement error를 다룬다.

## Common mistakes와 발견 방법

| Mistake | 관측 가능한 symptom | Check |
|---|---|---|
| perplexity만으로 extension run 검증 | target length에서 낮은 perplexity, target보다 짧은 length에서 retrieval 실패(32K에서 base 500이면 1K) | 4K, 16K, 64K, target에서 multi-needle 또는 RULER 형태 retrieval을 실행하고 [[rope-base-bounds-context-length]] Fig. 3 pattern과 비교 |
| target length lower bound보다 낮은 RoPE base | full length 학습에도 retrieval이 window 일부에 제한됨 | [[rope-base-bounds-context-length]] Table 2와 base를 비교하고 short run에서 더 큰 base 시험 |
| long-only continued-pretraining mix | 초기 recovery 뒤 training step에 따라 short-context average 하락 | 수백 step마다 short panel 기록([[longred]] Fig. 3), high-quality short data 37–66% 추가 |
| 한 long domain만 upsample(books만) | web, code, StackExchange validation loss 상승 | 0–4K에서 domain별 validation loss([[fu-2024-data-engineering-128k]] Table 5) |
| aggregate short score만 보고 | average는 “비슷”하지만 한 task가 5+ points 하락 | GSM8K, GPQA, MATH, code를 특히 포함한 task별 delta |
| 모든 traffic에 static YaRN 활성화 | `rope_scaling` 없는 경우보다 short-prompt quality 낮음 | config 유무 short panel을 A/B하고, dynamic scaling 또는 request별 enable 사용([[qwen-3-long-context]]) |
| sliding-window layer에 RoPE scaling 적용 | full-layer-only scaling보다 낮은 RULER | scaling config가 만지는 layer 확인(`full_attn_layers_only`, [[olmo-3-long-context]]) |
| long CPT에서 masking 없이 document를 cross-pack | long·short score가 조금 낮음 | 5B-token run에서 document masking ablate([[prolong]] Table 20) |
| SFT 전만 long context 평가 | short-only SFT 뒤 long-context ability 상실을 놓침 | SFT, DPO, RL 뒤 long suite 재실행([[llama-3]] §4.3.4; [[prolong]] Fig. 2) |
| length-mix ratio를 short-data ratio로 해석 | Qwen의 “75/25”를 general-data share로 그대로 복사 | ratio가 long data 안의 length를 뜻하는지, long 대 general data를 뜻하는지 읽기(§5.1) |

## 이해 점검

1. 2K에서 학습한 LLaMA 7B가 32K input을 읽게 하는 PI와 NTK-by-parts를 생각하자(\(s=16\)). 각 method가 high-frequency dimension에 무엇을 하는지 사용해, [[yarn]] Table 2에서 PI가 NTK-by-parts보다 MMLU를 더 낮추는 이유(25.9 대 32.7, base 35.7)를 설명하라.
2. 한 team이 base 5,000으로 4K model을 128K로 확장하고 128K perplexity가 4K보다 낮다고 보고한다. \(B_{m,\theta}\)를 사용해 이것이 model이 128K window를 사용할 수 있음을 보이지 않는 이유를 설명하고, 어떤 evaluation이 문제를 드러낼지 쓰라.
3. ProLong은 long data 60%가 SFT 뒤 best long-context average를 준다고 찾았지만 Olmo 3는 long 34%를 선택했다. ratio를 고르는 두 evaluation이 어떻게 다른 optimum으로 이어질 수 있는지, 새 model에서는 어떤 experiment가 둘 중 무엇이 맞는지 결정할지 설명하라.
4. STRING은 training 없이 effective length를 늘리고, Qwen2.5-1M의 262K stage는 128K RULER를 개선한다. position-frequency formula \(f(i)\)로 두 결과를 설명하라.
5. LongReD는 drift와 forgetting을 구분한다. short-data replay, restoration distillation, full-attention layer에 제한한 YaRN 각각이 어느 것을 다루고 왜 그런지 설명하라.
6. Llama 3에는 long SFT data 0.1%가 필요했지만 ProLong은 short-only SFT가 최선이었다. 두 report와 Llama 3 DPO observation 모두에 맞는 causal explanation을 제안하고, 이를 시험할 ablation을 설명하라.
7. LongPO에서 같은 pair의 DPO는 short-context task를 악화시켰지만 LongPO는 그렇지 않았다. chunk-conditioned reference가 implicit KL term에서 무엇을 바꾸는지, 왜 그것이 rejected term보다 retention에 더 중요할 수 있는지 설명하라.
8. original window가 32K일 때 static YaRN factor가 2K prompt에서도 model output을 바꿀 수 있는 이유를 ramp formula와 Hugging Face implementation으로 설명하라.

## 연결

- **Previous:** [[ch-32a]] — *Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining*. 그 장의 replay와 forgetting control은 §5.1과 §6.1의 일반형이다.
- **Next:** [[ch-32c]] — *Claimed versus Effective Context Length and Long-Context Evaluation*. §4, §5, §8에서 사용한 measurement suite를 다룬다.
- **Builds on this chapter:** [[ch-32d]] — *Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training*; [[ch-29a]] — *Long-Document Synthesis for Continued Pretraining and Long-Context SFT*; [[ch-44a]] — *Length in RL: Overlong Responses, Length Control, and Long-Context RL*.

## 출처

- [[position-interpolation]] — PI formula, bound, passkey step, scale factor별 BoolQ regression, recipe row.
- [[yarn]] — NTK-aware, NTK-by-parts, YaRN formula; Table 2, 3, 5 short·long-context result; recipe row.
- [[eleuther-extending-the-rope]] — NTK-aware와 dynamic scaling의 community origin(일화적 status), fixed-factor short-length effect.
- [[hf-transformers-yarn-code]] — Transformers v4.46.0의 static YaRN implementation.
- [[llama-2-long]] — ABF, unmodified-RoPE failure, ABF 대 PI short task, data-quality ablation, recipe row.
- [[rope-base-bounds-context-length]] — \(B_{m,\theta}\), lower-bound table, superficial long-context result.
- [[longrope2]] — critical dimension, needle-perplexity search, mixed-window training, retention table.
- [[dual-chunk-attention]] — DCA position remapping과 training-free perplexity result.
- [[string-effective-context]] — position-frequency formula와 statistic; STRING result.
- [[controlled-long-context-extension]] — LLaMA2-7B에서 extension method의 controlled comparison; short perplexity와 LCC drop.
- [[prolong]], [[prolong-recipe]] — long:short ratio, data source, short-data quality, masking, RoPE base, short-only SFT, recipe row.
- [[fu-2024-data-engineering-128k]] — source별 length upsampling, domain-loss table, token budget, extension 뒤 MMLU.
- [[llama-3]], [[llama-3-recipe]] — \(\theta=500{,}000\), six-stage gate, document mask, annealing, long SFT 0.1%, short-only DPO.
- [[qwen-long-context-synth]] — Qwen2.5-1M stage와 base, 75/25 length mix, short benchmark, short-only RL, DCA + YaRN.
- [[longred]] — drift와 forgetting 분석, restoration distillation objective와 result.
- [[longpo]] — short-to-long preference objective, KL reference, long·short result.
- [[skyladder]] — pretraining 중 context-window scheduling.
- [[qwen-3-long-context]] — Qwen3 long-context stage, mode별 RULER, static-YaRN model-card caution.
- [[olmo-3-long-context]] — Olmo 3 mixture ablation, full-attention layer YaRN, released script, stage score.
- [[gemma-3-long-context]] — local:global design, factor-8 interpolation, instruction tuning 전후 RULER.
- [[harm-de-vries-long-context-run]] — document-length statistic와 attention FLOPs formula.
