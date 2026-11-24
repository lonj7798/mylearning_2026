<!-- chapter: ch-32f
     track: midtraining
     kind: lab
     title: Lab: Annealing and Context Extension with a Short-Context Regression Gate
     deps: [ch-32e]
     sources: [[ruler]], [[helmet]], [[prolong]], [[prolong-recipe]], [[yarn]], [[paloma]], [[smollm2]], [[olmo-2-annealing]],
              [[controlled-long-context-extension]], [[position-interpolation]], [[llama-2-long]], [[llama-3]], [[llama-3-recipe]],
              [[signal-and-noise-eval]], [[cooldown-scaling-beyond-fixed-durations]], [[adding-error-bars-evals]],
              [[hf-rope-utils-v4460]], [[ruler-effective-length]]
     figures: figures/rope-extension-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32f — 실습: 짧은 문맥 회귀 게이트를 갖춘 annealing 및 문맥 확장

> **핵심 통찰.** Annealing과 문맥 확장은 각각 해당 단계가 겨냥하지 않은 점수도 바꾸며, 두 변화는 원인을 잘못 귀속하기 쉽다. 변경하지 않은 사전학습 혼합물에서 학습률만 감쇠해도 OLMo 2 7B 체크포인트의 OLMES는 69.6에서 74.0으로, MMLU는 59.8에서 61.8로 이동했다. 따라서 후보 데이터셋의 효과는 그 control과 비교해야만 측정할 수 있다([[olmo-2-annealing]], 표 11). 5B-token Llama-3-8B 확장 ablation에서는 long-data 비율이 높아질수록 short-task 평균이 단조롭게 내려갔고, 100% long data에서는 PG19 perplexity가 계속 좋아져도 downstream long-context 평균은 떨어졌다([[prolong]] §2.1, §3.2, 그림 1·3). 이 실습은 작은 체크포인트를 control 대비 두 후보 데이터셋으로 anneal하고, 고른 체크포인트를 두 RoPE 방법과 두 long:short 비율로 확장한다. paired short-context 차이가 사전 등록한 margin을 통과하고 RULER effective length와 HELMET RAG 점수로 long-context 이득을 보인 cell만 채택한다.
>
> **지침.** Annealing 데이터셋을 판단할 때에는 같은 token 수와 schedule로 pretraining mix에서 수행한 anneal과 비교하라. schedule만으로도 broad score가 변하기 때문이다([[olmo-2-annealing]] 표 11). long:short 비율 또는 확장 방법을 고를 때에는 target length의 perplexity가 아니라 downstream long-context task와 paired short-context 차이로 결정하라. perplexity는 [[prolong]] 그림 1에서 100% long data를 선호했기 때문이다. 이 실습의 1B·360M 체크포인트에서는, RULER effective length를 같은 harness에서 native length일 때의 해당 체크포인트 자체 점수에 상대적으로 정의한다. RULER의 임계값은 4K에서의 Llama 2 7B 점수(base model 79.4, aligned model 85.6; [[ruler-effective-length]] 표 3·12)이고, Lu 등 설정의 LLaMA2-7B는 4k에서 80.94를 기록했기 때문이다([[controlled-long-context-extension]] 표 3). 공개 보고서가 표 없이 확장이 “next to no degradation”을 일으켰다고만 쓰면([[smollm2]] §4.7), 그 주장을 이 실습 체크포인트에 대해 검증되지 않은 것으로 취급하고 직접 측정한다.

## 범용 모델에서 이 장이 중요한 이유

파이프라인은 pre-training → mid-training → SFT → preference optimization → RL → evaluation 순으로 진행된다. 이 실습은 [[ch-17]]이 pretraining을, [[ch-36]]이 SFT를 측정 가능한 산출물로 마무리하듯이, [[ch-32]]부터 [[ch-32e]]까지의 mid-training 단계를 측정 가능한 산출물로 마감한다. 여기서 나온 체크포인트는 SFT가 받는 시작점이므로, 이 단계가 short-context 능력에서 제거한 것은 이후의 모든 단계가 상속한다.

이 실습은 다음 세 가지를 측정 가능한 문제로 다룬다.

1. **Annealing 이득의 귀속.** Anneal에는 학습률 감쇠와 데이터 변경이 함께 들어간다. control arm이 없으면 감쇠의 효과를 데이터의 공으로 돌린다. OLMo 2 7B에서 감쇠만으로 OLMES 4.4점과 MMLU 2.0점이 더해졌다([[olmo-2-annealing]] 표 11).
2. **확장으로 인한 short-context 회귀.** Llama-3-8B(MMLU 66.5, GSM8K 44.7)에 비해 RoPE-base position extrapolation은 64.7 / 40.1, Fu 등의 long mix로 학습한 것은 63.1 / 40.6이었다([[prolong]] §2.3, 표 2). Position Interpolation으로 window를 32,768 token으로 늘리면, 원래 2,048-token window 안에서 측정한 LLaMA 7B zero-shot BoolQ가 76.1에서 64.7로 내려갔다([[position-interpolation]] 표 5).
3. **잘못된 대상을 재는 long-context 주장.** Perplexity와 단일 needle retrieval은 개선돼도 task score는 개선되지 않을 수 있다. 128K에서 NIAH는 [[helmet]] 표 5의 frontier model 여섯 개 중 다섯 개에서 100.0이며, NIAH는 ∞Bench QA와 0.63 상관인 반면 HotpotQA RAG task는 0.88 상관이다(그림 4).

## §1 실습 질문, 가설, 산출물 및 예산 경로

**질문.** 작은 체크포인트를 short-context 회귀 게이트를 통과시키면서 anneal하고 문맥 확장할 수 있는가?

**실행 전에 작성하는 가설.** 각 행은 출처가 측정한 변화 방향, 그 측정 설정, 그리고 이 실습에서 그 가설을 반박할 관측을 적는다.

| 축 | 예측 방향 | 출처 및 설정 | 반박 관측 |
|---|---|---|---|
| Pretraining mix에서 anneal(control) 대 시작 체크포인트 | control이 broad multiple-choice 점수를 올린다 | [[olmo-2-annealing]] 표 11(7B, 4T tokens 시점, 50B tokens): OLMES 69.6 → 74.0, MMLU 59.8 → 61.8, GSM* 28.5 → 27.0 | broad suite에서 control − start 구간이 0을 포함한다 |
| Targeted math 후보 대 control | math target은 0보다 큰 구간의 이득(출처의 7B에서는 35점), broad 변화는 ±1점 안 | [[olmo-2-annealing]] 표 12(7B, 576M tokens, math/web 35/65): GSM* 28.5 → 63.5, MMLU 59.8 → 60.1 | GSM8K는 올랐는데 held-out math partner가 오르지 않는다 |
| Broad educational-web 후보 대 control | knowledge·science multiple choice는 이득, commonsense task는 손실 | [[smollm2]] 표 1(anneal이 아닌 scratch에서의 350B-token run): FineWeb-Edu 대 DCLM의 MMLU 37.5 대 35.5, HellaSwag 60.1 대 62.3 | HellaSwag와 CommonsenseQA 구간이 0보다 높다 |
| Long share 70% 대 40% | 70%에서 short-context score가 더 낮다 | [[prolong]] 그림 3(Llama-3-8B, 64K에서 5B tokens; 단조 감소) | paired short-suite 차이 70% − 40%가 0보다 높다 |
| Base-frequency increase(ABF) + 학습 대 YaRN + 학습 | target length RULER에서 ABF ≥ YaRN, target length 바깥에서는 YaRN이 더 낮다 | [[controlled-long-context-extension]] 표 3(LLaMA2-7B, 32k에서 1B tokens): Dynamic NTK 32k 59.42 대 YaRN 36.95, 64k 46.26 대 0.00 | L′에서 YaRN − ABF RULER 구간이 0보다 높다 |
| Document masking on 대 off | long·short 평균 모두에서 작은 이득 | [[prolong]] 부록 B.2 표 20(8B, 5B tokens, UltraChat SFT 후 채점, seed 수 미보고): long 54.6 대 53.6, short 65.5 대 64.9 | 1B에서는 열린 질문이다. 구간의 부호를 예측하지 않는다 |

ABF 행은 고정된 증가 base를 Lu 등의 Dynamic NTK 변형과 비교한다. 따라서 이 방향은 이 실습을 위한 해당 결과의 해석이다.

**산출물.**

1. `runs.jsonl`: 각 cell마다 start checkpoint, data file hash, source별 token 수, 실제로 load된 RoPE 설정(base, `rope_scaling`, ramp bound), masking flag, LR schedule, seed, commit을 한 행으로 기록한다.
2. `eval/items/<cell>/<suite>.jsonl` 및 `eval/docs/<cell>/<paloma-domain>.jsonl`: start checkpoint와 모든 cell에 대해 item별 score와 document별 negative log-likelihood를 저장한다.
3. `decontamination.md`: 모든 후보 및 extension source와 모든 evaluation suite 사이의 n-gram overlap을 기록한다(§2.3).
4. `stage-memo.md`: 학습 전 확정한 margin, Part A 표, gate 표(§7), 선택한 recipe, 그리고 선택 cell이 잃은 모든 short-context score와 그 interval을 기록한다.

**예산 경로.**

| 항목 | Full path | Resource-constrained path |
|---|---|---|
| Start checkpoint | OLMo 2 1B stage 1 뒤(4T tokens, Dolmino anneal 전; [[olmo-2-annealing]] 부록 B). 그 revision이 공개되지 않았다면 final decay 전의 다른 RoPE checkpoint | SmolLM2-360M base(config: `max_position_embeddings` 8192, `rope_theta` 100000, head dimension 64; [[smollm2]]) |
| Native length L₀ → target L′ | 4,096 → 16,384(4×); 선택적으로 ABF 32,768(8×) cell | 8,192 → 16,384(2×) |
| Part A arm | control, targeted, broad; 각각 5B tokens | control, targeted; 각각 1B tokens |
| Part B cell | ABF와 YaRN × long 40%와 70%; ABF-40%에는 masking off; 학습 없는 YaRN 평가 | long 40%에서 ABF |
| Extension cell당 token | 1B | 0.5B |
| Short-context gate, RULER, HELMET RAG | 필수 | 필수 |

OLMo 2 부록 B는 1B와 7B의 차이(layer, width, head, batch, peak LR)를 나열하지만 sequence length나 RoPE θ는 나열하지 않는다. 따라서 실습은 7B 설정에서 L₀ = 4,096과 θ = 500,000을 읽어 derived로 둔다. 학습 전 load한 `config.json`이 둘 모두를 확인해야 한다. SmolLM2 §6은 360M의 학습 길이를 쓰지 않으므로, constrained path는 먼저 start checkpoint에서 2K, 4K, 8K RULER를 실행해 자체 effective length를 찾는다.

**Token 산술(derived).** OLMo 2 1B의 batch가 4,096 token짜리 512 sequence라면 한 step은 2,097,152 token이므로 5B-token anneal은 2,384 step이다. 16,384 token에서 step당 64 sequence인 extension cell은 step당 1,048,576 token을 사용하므로 1B tokens는 954 step이다. 두 수는 모든 sequence가 채워진다고 가정한다. log에 남긴 실제 token 수가 이를 대체한다.

## §2 Part A: 두 후보 데이터셋으로 체크포인트 annealing하기

### 2.1 정의, 문제, 메커니즘

**정의.** Anneal은 학습률을 0까지 감쇠시키면서 checkpoint의 학습을 계속하는 과정이며, 그 데이터 mixture는 pretraining 때와 다를 수 있다. **Microanneal**은 하나의 data source를 시험하기 위해 수행하는 짧은 anneal이다([[olmo-2-annealing]] §4.4.2).

**문제.** Anneal 뒤에 측정한 변화에는 세 효과가 섞여 있다. 감쇠, 후보 데이터, pretraining data 비율의 감소다. Llama 3는 데이터셋의 가치를 판단하기 위해 anneal을 사용했다. 50%까지 학습한 8B model에 새 데이터셋을 30% 가중치로 넣어 40B token 동안 learning rate를 선형으로 0까지 감쇠했다([[llama-3-recipe]] §3.1.3). SmolLM2는 3T-token checkpoint에서 후보 token 60B와 checkpoint 전 token 40B를 사용해 math data를 시험했다([[smollm2]] §3.1). 이 문단들에서 두 보고서는 control arm을 제시하지 않았지만, OLMo 2 표 11은 제시한다.

**메커니즘.**

1. 공개된 schedule에서 시작 checkpoint가 멈춘 step의 learning rate η₀를 기록한다.
2. 각 arm의 mixture를 만든다. OLMo 2의 “roughly the same quantity” 규칙을 따라 후보 token 50%와 pretraining-mix token 50%를 쓴다. Control arm은 pretraining mix 100%를 쓴다.
3. 모든 arm을 같은 token 수와 같은 schedule로 학습하고, η₀에서 0까지 선형 감쇠시킨다.
4. 같은 harness(§3)로 start checkpoint와 모든 arm을 평가한다.

**수식(선형 감쇠).**

```
η(k) = η₀ · (1 − k / K),   k = 0 … K
```

k는 anneal 내부의 optimizer step, K는 anneal step 수, η₀는 checkpoint 시점의 learning rate다. 이 실습은 OLMo 2, Llama 3, SmolLM2가 인용한 anneal에서 사용한 선형 형태를 쓴다. [[cooldown-scaling-beyond-fixed-durations]](§3.2, 그림 4)에서는 (1-sqrt) cooldown이 선형 cooldown보다 더 낮은 loss에 도달했다. 이 실습은 모든 arm에서 하나의 형태를 유지하고 그 형태를 기록한다.

### 2.2 Arm

| Arm | 후보(전체 token의 50%) | 역할 |
|---|---|---|
| N0 control | 없음(pretraining mix 100%) | 감쇠를 분리한다 |
| N1 targeted | FineMath 4+(10B tokens, 6.7M documents; [[smollm2]] §3.3.2). §2.3에 따라 decontamination하고 benchmark training split은 넣지 않는다 | 좁은 skill source |
| N2 broad | 점수가 높은 FineWeb-Edu subset | 예측한 혼합 패턴을 보이는 폭넓은 educational source. 이 예측은 DCLM과 비슷한 pretraining mix를 가정하므로, 실제 mix를 기록한다 |

**Arm 효과의 근거.**

- OLMo 2 7B microanneal에서 576M token의 math/web 35/65 mixture는 GSM*를 28.5에서 63.5로 올렸고 MMLU는 59.8 → 60.1이었다. 1.72B token의 10/90 mixture는 각각 61.0과 60.9를 보였다. 저자들은 math data가 지배적일 필요는 없지만 포함되어야 한다고 결론낸다([[olmo-2-annealing]] 표 12). 결과(단일 연구).
- FineMath 4+ anneal은 1.7B에서 InfiMM-WebMath와 비교해 “GSM8K에서 2배 개선, MATH에서 6배 개선”을 보였다([[smollm2]] §3.3.2, 그림 1). 결과(단일 연구).
- Llama 3에서 GSM8K와 MATH training set으로 anneal하면 8B validation score가 각각 24.0%와 6.4% 올랐고, 405B에서는 효과가 “negligible”했다. 공개한 annealing data는 benchmark training set을 제외했다([[llama-3]] §3.1.3). 결과(단일 연구). OLMo 2의 math mix에는 GSM8K-Train이 들어가고, SmolLM2의 decay stage에는 AugGSM8K가 0.02% 들어간다([[olmo-2-annealing]] §4.4.1; [[smollm2]] §4.5). 이 실습은 N1이 benchmark 형식이 아니라 source 자체를 측정하게 하려고 benchmark training split을 제외한다.

### 2.3 Decontamination

§3.2의 모든 suite를 기준으로 두 후보 모두에 SmolLM2 규칙을 적용한다. 13-gram matching을 하고 longest common subsequence 대비 최소 overlap ratio는 0.6으로 둔다([[smollm2]] §3.3.2). Suite별로 일치한 document를 기록하고 제거한다. Paloma 자체 decontamination 규칙은 code source를 다루지 않는다(G1; [[paloma]] §3). 따라서 실습 규칙을 code domain에도 적용하지 않았다면 perplexity panel의 code domain은 후보 데이터에 대해 decontamination되지 않은 것으로 표기한다.

### 2.4 Worked example: 200-item development set이 구별할 수 있는 것

OLMo 2는 GSM8K 1,319문제 가운데 200문제인 GSM*로 mixture를 결정했다([[olmo-2-annealing]] §4.4.2). Accuracy 28.5%에서 analytic 95% half-width는 1.96 · sqrt(0.285 × 0.715 / 200) = 6.3점이다(derived). PT Mix의 −1.5점 변화는 3문제이고 Web FW72의 +1.5점 변화도 3문제다. 저자들은 둘을 “within margin of error”라고 부른다(§4.3). 35점 microanneal 이득은 half-width 5.6개다. 따라서 200-item set은 math arm과 control을 구분할 수 있지만 1~2점의 broad effect는 구분할 수 없다. 이 실습은 그런 효과에 full test set과 paired interval을 사용한다(§5).

**조건과 한계.** 표 11과 표 12 근거는 4T token 시점의 7B 모델에 대한 것이다. OLMo 2 부록 B는 1B에서 training token을 scale하는 일이 “difficult”했다고 보고하므로, 1B에서 control 효과의 크기는 보고되어 있지 않다. 세 data order의 soup은 표 14의 24개 metric cell(여섯 mixture × 네 metric) 중 23개에서 최고의 single run과 같거나 더 좋았다. 하지만 mixture E의 GSM*는 best single 60.5에서 soup 43.0으로 떨어졌다. 따라서 anneal의 run-to-run variation은 무시할 수 없다(§5.3).

**범용 모델에 대한 함의.** Targeted anneal은 7B에서 target benchmark를 30점 이상 움직일 수 있다. Broad suite와 held-out partner가 그 이득이 일반 math ability인지, 한 benchmark의 형식에 맞춘 결과인지를 가른다.

## §3 Part A 측정: 표적 효과와 broad 효과

### 3.1 Domain별 held-out perplexity

**정의.** Paloma는 16개 source의 546개 domain에서 perplexity를 측정하고, domain별 perplexity의 비가중 평균인 macro average를 보고한다([[paloma]] §2, §4.1).

**문제.** 하나의 held-out loss는 domain 전반을 평균 내므로 domain별 변화를 가린다. Paloma의 C4-only baseline은 RedPajama arXiv에서 perplexity 391,171에 도달했다(§4.1). Targeted anneal은 자기 domain의 loss는 낮추고 다른 곳의 loss는 올릴 것으로 예상할 수 있다.

**수식.**

```
ppl(d)   = exp( − Σ_{t∈N_d} Σ_i ln p(t_i | t_<i) / T(N_d) )
macro(D) = (1/|D|) Σ_{d∈D} ppl(d)
BPB      = − ℓ / (B · ln 2)
```

N_d는 domain d의 document 집합, t는 한 document, t_i는 그 i번째 token, T(N_d)는 token 수, D는 domain 집합, ℓ은 합산 log-likelihood, B는 UTF-8 byte 수다. Tokenizer가 다를 때는 BPB(bits per byte)를 사용한다([[paloma]] App. B).

**프로토콜.**

1. 학습 전에 candidate별로 Paloma domain을 target group과 broad group으로 배정한다. N1에서는 math 관련 domain이 target group이고, 나머지 모두가 broad group이다.
2. BOS 뒤에서 각 document를 따로 score하고, context보다 긴 document는 서로 겹치지 않는 input으로 나눈다. Input을 이어 붙이면 variance 추세가 깨졌다. Pythia 1.4B를 2B token까지 학습했을 때 concatenated 입력은 92.23 ± 17.33, document별 입력은 42.57 ± 0.29였다(Table 17).
3. Arm 간 차이를 document 단위로 paired 비교할 수 있도록 document별 NLL을 저장한다.

**Worked example(설명용 값).** Broad domain 세 개의 perplexity가 N0에서는 12.0, 20.0, 6.0이고 N1에서는 12.4, 22.0, 6.1이라고 하자. Macro average는 12.67과 13.50이므로 6.6% 증가다. Domain별 log ratio는 0.033, 0.095, 0.017이고, 합산 log 변화 가운데 66%는 두 번째 domain에서 온다(0.095 / 0.145). Macro average는 높은 perplexity domain에 더 큰 가중을 사실상 주므로, memo에는 domain별 목록을 보고한다.

**한계.** Paloma는 자신이 시험한 downstream task 8개 모두와 상관하는 단일 perplexity source를 찾지 못했다. 예를 들어 c4-en ranking은 HellaSwag ranking과 −0.77의 상관을 보인다(App. A, Table 3). 여기서 perplexity는 진단 도구이고, 최종 판단은 task score가 한다.

### 3.2 Task suite의 역할

| Suite | N1(math)의 역할 | N2(educational web)의 역할 | 참고 |
|---|---|---|---|
| GSM8K (1,319) | target | broad | benchmark training split은 candidate에서 제외 |
| MATH | GSM8K의 held-out partner | broad | 보고하되 의사결정에는 쓰지 않음 |
| MMLU, ARC-Challenge, OpenBookQA | broad | target | multiple-choice 형식에서 정답률이 chance에 가까우면 cloze formulation 사용 |
| HellaSwag, CommonsenseQA, WinoGrande | broad | broad(예측된 loss) | [[smollm2]] Table 1의 방향 |
| MMLU-Pro, TriviaQA | held-out | held-out | SmolLM2는 두 benchmark를 “not monitored during training”으로 열거함(Table 4) |

**Targeted delta와 broad delta.** Arm N과 control N0에 대해, Δ_target은 arm의 target suite에서 `score_N − score_N0`이고, Δ_broad는 broad suite들에서의 `(score_N − score_N0)` 평균이다. 둘 다 paired interval을 붙인다(§5). Δ_target의 interval이 0을 제외하고 Δ_broad의 lower bound가 −δ보다 크면 arm을 “targeted”라고 한다. Δ_broad의 interval이 0을 제외하면 “broad”라고 한다.

### 3.3 Run-to-run variation

1B 모델의 마지막 30개 checkpoint는 ARC Challenge에서 accuracy 1.7% 범위를 보였고([[signal-and-noise-eval]] §3.1), bits-per-byte scoring은 benchmark의 90.0%에서 small-scale decision accuracy를 개선했다(§5.3). 따라서 실습은 (a) multiple-choice suite를 accuracy와 정답의 BPB 모두로 score하고, (b) control arm을 두 번째 data order로 반복한다. 두 control run의 차이를 모든 Part A delta 옆에 보고한다. Candidate 차이가 그 control-to-control 차이보다 작으면 undecided로 보고한다.

**Part B의 선택 규칙.** N*은 모든 broad suite에서 §5의 margin을 통과한 arm 중 Δ_broad lower bound가 가장 큰 arm이다. Candidate가 하나도 통과하지 못하면 N0을 선택한다. Part B는 N*에서 시작하며, short-context gate는 각 cell을 원 checkpoint가 아니라 N*와 비교한다.

## §4 Part B: 두 가지 long:short 비율에서 두 가지 확장 방법

### 4.1 Continued pretraining을 이용한 base frequency 증가(ABF)

**정의.** RoPE는 각 query·key 쌍 $i$를 주파수 $\theta_i=b^{-2i/d}$로 회전시킨다. ABF(adjusted base frequency)는 $b$를 더 큰 $b'$로 바꾸고 목표 길이에서 continued pretraining을 수행한다([[llama-2-long]] §4.1).

**문제.** 파장 $\lambda_i=2\pi/\theta_i$가 $L_0$보다 큰 쌍은 pretraining 중 한 바퀴의 회전도 완료하지 못했다([[yarn]] §3.2). OLMo 2 1B($b=500{,}000$, $d=128$, $L_0=4{,}096$)에서는 64개 쌍 중 32개의 $\lambda_i$가 4,096보다 크다(derived; 그림에서 계산). Continued training을 한 unmodified RoPE는 Llama 2 7B의 first-sentence retrieval에서 4,000–6,000 token 너머에 attend하지 못했다([[llama-2-long]] §4.1).

**식(NTK-aware base).**

```
b′ = b · s^{d/(d−2)},   s = L′/L₀
```

$b$는 pretrained base, $b'$는 새 base, $d$는 rotary head dimension, $s$는 scale factor다([[yarn]] App. A.2, Eq. 19).

**Worked example(derived).**

- Llama-3-8B, 8K → 64K: $s=8$, $d=128$, $b=5\times10^5$이면 $b'=5\times10^5\cdot8^{1.0159}=4.13\times10^6$이다. ProLong의 Table 18 caption은 Dynamic NTK가 “roughly suggests to use 4m”이라고 적는다. ProLong은 $8\times10^6$을 사용했고, 이는 long task 평균 54.6이었다. $4\times10^6$은 48.7, 원래 $5\times10^5$는 29.1이었으며 short 평균은 각각 65.5, 65.3, 65.0이었다([[prolong]] App. B.1). 결과는 단일 연구 결과다.
- SmolLM2-1.7B, 2,048 → 8,192: $s=4$, $d=64$, $b=10{,}000$이면 $b'=41{,}829$다. SmolLM2는 printed ablation 없이 그 값의 3.1배인 130,000을 사용했다([[smollm2]] §4.6).
- OLMo 2 1B, 4,096 → 16,384: $b'=2.04\times10^6$이다. 실습은 ProLong이 64K에서 더 좋았다고 보고한 비율인 $2\times b'=4.09\times10^6$을 사용한다. 선택 사항인 8× cell에는 $2\times4.13\times10^6=8.27\times10^6$을 쓴다. 이 2배 factor는 8B·64K에서 나온 ProLong 하나의 결과를 course choice로 가져온 것이며, 1B에서 시험되지 않았다.

[figures/rope-extension-explorer.html](figures/rope-extension-explorer.html) 그림은 독자가 $b$, $d$, $L_0$, $L'$를 설정하고, ABF 뒤와 YaRN 뒤 각 pair의 파장, NTK-suggested base, 논문과 transformers v4.46.0의 YaRN ramp, attention-logit multiplier를 보게 한다.

**Short context 근거.** 7B에서 32,768 길이로 80B token을 학습했을 때 ABF의 MMLU는 46.24였고, PI는 45.84, unmodified RoPE는 45.69였다. 또한 ABF만 first-sentence retrieval을 32,768까지 유지했다([[llama-2-long]] Table 6, Fig. 5b). Training 없이 inference에서 base를 바꾸면 Llama-3-8B-Inst의 짧은 길이 ODQA와 ICL이 하락했다([[helmet]] App. E.3). 각각 단일 연구 결과다.

### 4.2 Continued pretraining을 이용한 YaRN

**정의.** YaRN은 low-frequency pair를 factor $s$로 interpolate하고 high-frequency pair는 유지하며, 그 사이를 ramp로 연결하고 attention logit을 scale한다([[yarn]] §3.2–3.3).

**식.**

```
r(i) = L₀ / λ_i
γ(r) = 0 if r < α;  1 if r > β;  (r − α)/(β − α) otherwise
h(θ_i) = (1 − γ(r(i))) · θ_i / s + γ(r(i)) · θ_i
softmax( q_mᵀ k_n / (t √|D|) ),   √(1/t) = 0.1 ln s + 1
```

$r(i)$는 pair $i$가 $L_0$ 동안 만드는 회전 수, $\gamma$는 ramp, $\alpha=1$과 $\beta=32$는 Llama-family에 권장된 bound, $h$는 새 주파수, $t$는 temperature, $|D|$는 [[yarn]] §2.1(Eqs. 10–15)이 정의한 attention dimension이다.

**구현.** transformers v4.46.0에서 `_compute_yarn_parameters`는 ramp bound를 계산할 때 `config.max_position_embeddings`에서 pretrained length를 가져오며(L192, L230), Llama rotary embedding은 모든 sequence length에서 cos·sin에 `attention_factor = 0.1 ln s + 1`을 곱한다([[hf-rope-utils-v4460]]). 따라서 실습의 YaRN cell은 다음과 같이 설정한다.

```json
{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "beta_fast": 32.0, "beta_slow": 1.0},
 "max_position_embeddings": 4096, "rope_theta": 500000}
```

그리고 최대 16,384 token sequence로 학습·평가한다. 이 설정을 쓰기 전에는 설치된 transformers version과 model class에서 어느 config field가 ramp의 pretrained length를 공급하는지 확인한다. v4.46.0 code에는 `original_max_position_embeddings` field를 사용해야 한다는 TODO가 있다(L133).

**Worked example(derived).** OLMo 2 1B에서 `find_correction_dim`은 $\beta_{fast}$에 대해 $128\cdot\ln(4096/(32\cdot2\pi))/(2\ln500000)=14.70$, $\beta_{slow}$에 대해 31.60을 준다. 따라서 ramp bound는 floor 14, ceil 32다. pair 0–14는 $\theta_i$를 유지하고, pair 32–63은 $\theta_i/4$를 사용하며, pair 15–31은 섞인다. `max_position_embeddings`를 16,384로 설정하면 같은 code는 bound 21과 39를 반환한다. 그러면 pair 15–21(파장 136–466 token, 4,096 token 동안 30–8.8회전)은 원래 주파수를 유지하고, pair 32–38(파장 4,443–15,204 token)은 부분적으로만 interpolate되며, pair 39–63만 $\theta_i/4$를 쓴다. $s=4$의 attention factor는 1.1386이므로, 512-token short-context item을 포함한 모든 입력 길이에서 logit은 1.2965배가 된다(derived).

**근거.** LLaMA 7B에서 $s=16$, 400 step 뒤 YaRN의 32k proof-pile perplexity는 2.77로 PI의 3.57보다 낮았지만, MMLU는 30.0으로 base 35.7, NTK-by-parts 32.7, PI 25.9와 비교된다([[yarn]] Tables 2, 5). $s=32$에서는 Llama 2 13B MMLU가 55.8에서 51.9로 하락했다(Table 3). Lu et al.의 controlled setting(LLaMA2-7B, 32k에서 1B token, PI·YaRN의 scale factor 8.0)에서는 YaRN이 32k RULER 36.95, 64k RULER 0.00, 2k PG19 perplexity 6.70을 보였고 base는 6.61이었다([[controlled-long-context-extension]] Tables 2, 3, 7). 각각 단일 연구 결과다.

**Training-free reference cell Z.** Factor 4.0 YaRN을 학습 없이 평가한다. HELMET은 inference에서 YaRN을 쓴 Qwen2-Inst가 32,768 token 이후 하락했다고 보고한다([[helmet]] App. E.3). Cell Z는 configuration만의 효과에 비해 학습이 무엇을 더하는지 보여 준다.

### 4.3 Long:short 비율과 data construction

**정의.** Long share는 길이 $L'$ 이상인 단일 document에서 나오는 training token의 비율이고, short share는 더 짧은 document를 pack한 token의 비율이다. ProLong도 같은 구분을 사용한다. Long data는 single-document chunk이고 short data는 packed data다([[prolong]] §3).

**근거.** ProLong의 5B-token Llama-3-8B ablation에서는 “short-task performance monotonically decreases as the long data increases”였다. SFT 전에는 recall·RAG가 높은 long share를 선호했지만, SFT 후에는 long data가 많을수록 두 지표가 “drastically” 악화했고, 가장 좋은 long 평균은 60% long에서 나왔다([[prolong]] §3.2, Fig. 3). SmolLM2는 8K token 이상 document 40%를 사용했다(§4.6). ProLong은 Fu et al. mix가 long document를 “roughly 70% of tokens”로 구성한다고 쓴다(§2.2). 실습은 40%와 70%를 시험한다. 비율 효과는 단일 연구 결과다. ProLong은 별도 언급이 없으면 UltraChat SFT 뒤 ablation score를 보고하지만(§2.2; App. A.4), 이 실습은 base checkpoint를 gate하므로 비율 효과의 방향만 가져온다.

**구성.**

1. Long source는 ProLong Table 4에서 가장 좋은 두 단일 long source인 책과 code repository다(books/repos 1:1의 long 평균 54.6). Checkpoint tokenizer로 최소 16,384 token인 document를 유지하고, 각각을 길이 16,384의 single-document chunk로 자른다.
2. Short source는 Part A pretraining mix이며, §4.4의 masking을 적용해 16,384까지 pack한다.
3. Cell당 1B token의 token accounting(derived): 40% long은 400M token, 24,414 chunk이고, 70% long은 700M token, 42,725 chunk다. Long pool이 더 작으면 document당 repeat count를 기록하고 ABF와 YaRN cell에서 같아야 한다.
4. §2.3의 규칙으로 long source를 §6의 모든 long-context suite에 대해 decontaminate한다.

### 4.4 Document masking on과 off

**정의.** Document masking에서는 packed sequence 안의 token이 자기 document의 앞 token에만 attend한다([[prolong]] App. B.2).

**근거.** SFT 뒤 score한 ProLong Table 20에서 mask 없음은 long 53.6 / short 64.9, mask 있음은 long 54.6 / short 65.5였다. Arm당 run 수는 보고되지 않았다(단일 연구 결과). Llama 3는 document-boundary mask를 적용하며 standard pretraining에서는 “limited impact”, 매우 긴 sequence에서는 “important”하다고 설명하지만 수치를 제시하지 않는다([[llama-3-recipe]] §3.2).

**설계.** Cell E5는 E1(ABF, 40% long)을 masking off로 반복한다. 두 cell 모두에서 `position_ids`가 document마다 restart하는지 기록한다. Position id를 restart하면 packed short document는 자기 길이를 넘는 position을 훈련하지 않으므로, long-position signal은 long share에서만 나온다(해석이며, 이 효과만 분리한 source는 없다).

### 4.5 Run matrix(full path)

| Cell | Start | Method | Long share | Masking | Tokens | Purpose |
|---|---|---|---|---|---|---|
| E1 | N* | ABF, $b'=4.09\times10^6$ | 40% | on | 1B | method × ratio |
| E2 | N* | ABF | 70% | on | 1B | method × ratio |
| E3 | N* | YaRN, factor 4.0 | 40% | on | 1B | method × ratio |
| E4 | N* | YaRN, factor 4.0 | 70% | on | 1B | method × ratio |
| E5 | N* | ABF | 40% | off | 1B | masking |
| Z | N* | YaRN, factor 4.0 | — | — | 0 | training-free reference |
| E1-32K (optional) | N* | ABF, $b'=8.27\times10^6$, $L'=32{,}768$ | 40% | on | 1B | 8× extension |

모든 training cell은 같은 peak LR $2\times10^{-5}$(§Recipe), shared source에 대해서는 같은 data order, 같은 step count를 사용한다.

## §5 Short-context regression gate

### 5.1 Item과 document에 대한 paired interval

**정의.** 이 gate는 $L_0$보다 길지 않은 입력에서 각 cell $C$를 같은 item에 대한 시작 checkpoint $N^*$와 비교한다. Item $i$에 대해 $d_i = score_{C,i} - score_{N^*,i}$다.

**식**([[adding-error-bars-evals]] §4.2, Eq. 7).

```
d̄          = (1/n) Σ_i d_i
SE_paired   = sqrt( (1/(n−1)) Σ_i (d_i − d̄)² / n )
CI_95%      = d̄ ± z · SE_paired
```

$n$은 item 수, $d̄$는 평균 paired 차이이며, 비교 하나에서는 $z=1.96$이다. Document별 Paloma NLL에서는 $d_i$가 document $i$의 token당 평균 NLL 차이다.

**Worked example 1: margin은 zero와 같은 검정이 아니다.** $N^*$가 1,000 item 중 600개에 답한다. Cell $C$는 40개를 잃고 25개를 얻으므로 $d̄=-0.015$다. $\Sigma d_i^2=65$, $\Sigma(d_i-d̄)^2=65-1{,}000\times0.000225=64.775$, $SE_{paired}=\sqrt{64.775/999/1{,}000}=0.00805$이고, interval은 $[-0.0308,+0.0008]$이다. Margin $\delta=2$ point에서는 lower bound −3.08 point가 −2보다 낮으므로 interval에 zero가 포함되어도 cell은 실패한다. 15개를 잃고 10개를 얻는 cell은 $SE_{paired}=0.00500$, lower bound −1.48 point이므로 통과한다. 첫 cell의 unpaired standard error는 0.0220이며, 이는 $[-5.8,+2.8]$을 만들어 5.8 point 아래의 모든 margin에서 실패하게 한다.

**Worked example 2: document별 perplexity.** 다섯 document의 $\Delta$ mean NLL은 token당 nat 단위로 $[0.02,0.05,-0.01,0.03,0.01]$이다. $d̄=0.02$, $\Sigma(d_i-d̄)^2=0.002$, $SE=\sqrt{0.002/4/5}=0.01$이고 interval은 $[0.0004,0.0396]$이다. $\exp(0.02)=1.020$, 즉 interval이 zero를 제외하는 2.0% perplexity 증가다.

**Bootstrap.** Binary가 아닌 score 또는 작은 $n$에서는 paired bootstrap이 item index를 resample한다([[ch-36]] §6.2에 열거된 예가 있다).

```python
import numpy as np   # course code, not taken from a source

def paired_gate(ref, new, delta, alpha=0.05, n_boot=10_000, seed=0):
    d = np.asarray(new, float) - np.asarray(ref, float)          # same items, same order
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))              # n_boot × n int64: n = 14,000이면 1.1 GB; ch-36 §6.2처럼 chunk 처리
    lo, hi = np.quantile(d[idx].mean(1), [alpha / 2, 1 - alpha / 2])  # Bonferroni: alpha = 0.05 / comparisons
    return {"mean": d.mean(), "ci": (lo, hi), "pass": lo > -delta}
```

### 5.2 Suite, margin, multiple comparison

1. Short suite는 MMLU, ARC-Challenge, HellaSwag, WinoGrande, GSM8K(full test set)와 Paloma broad group이다. Held-out은 MMLU-Pro와 TriviaQA다.
2. Suite별 margin $\delta$는 Part B 전에 memo에 쓴다. Lab default는 1,000 item 이상 suite에 $\delta=2$ point, 더 작은 suite에 $\delta=3$ point다(course choice이며 source에서 온 값이 아니다). Llama 3는 short-context evaluation이 수치 없이 “fully recovered”했을 때 context stage를 진행시켰다([[llama-3-recipe]] §3.4.2).
3. 6 cell $\times$ 5 suite = 30 comparison이면 Bonferroni는 comparison별 양측 level을 $0.05/30=0.00167$로 두므로 $z=1.96$ 대신 $z=3.14$다(derived). Worked example 1의 lower bound는 $-0.015-3.14\times0.00805=-4.03$ point가 된다.
4. 모든 suite는 각 cell 자신의 RoPE configuration에서 score한다. YaRN cell에서는 short length에서도 적용되는 attention factor를 포함한다(§4.2).

**Source에서 regression이 보인 방식.** Lu et al.의 extended LLaMA2-7B model은 LongBench code completion LCC를 68.22에서 56.78(Dynamic NTK), 55.05(PI), 54.06(YaRN)으로 낮췄다. 논문은 LongBench 평균이 올랐음에도 이 drop을 논의하지 않는다([[controlled-long-context-extension]] Table 11). 평균만 보는 gate는 이 cell을 통과시키지만, suite별 gate는 통과시키지 않는다.

## §6 Long-context gate: RULER effective length와 HELMET category 하나

### 6.1 Small checkpoint를 위한 RULER effective length

**정의.** RULER는 선택한 길이에서 retrieval, multi-hop tracing, aggregation, QA에 걸친 13개 synthetic task를 생성하고, effective length를 “threshold를 넘는 최대 길이”로 정의한다. Aligned model을 채점할 때 threshold는 Llama2-7B chat model의 4K score인 85.6이고(Table 3), base model을 채점할 때는 Llama2-7B base model의 4K score인 79.4이다(App. A, Table 12)([[ruler-effective-length]]; [[ruler]]).

**문제.** 이 절대 threshold는 small model이나 다른 harness로 옮겨 쓸 수 없다. 128K를 주장한 Yarn-base(7B)는 4K에서 77.3점을 받아 79.4 규칙 아래에서는 “<4K”를 받았다(Table 12). Lu et al.은 자신들의 configuration에서 LLaMA2의 4K score를 80.94로 보고했는데, RULER 자체의 base-model score보다 1.5 point 높다([[controlled-long-context-extension]] Table 3). Native length에서 79.4보다 낮은 1B checkpoint는 모든 길이에서 “<4K”를 받으므로, 그 규칙으로는 cell 사이의 순위를 매길 수 없다.

**Course 규칙(source에서 온 규칙 아님).** $\tau$ = lab harness에서 $L_0$로 측정한 $N^*$의 RULER 평균 − 5 point로 둔다. Effective length는 시험한 모든 길이 $\le L$의 score가 적어도 $\tau$인 가장 큰 시험 길이 $L$이다. Full path의 길이는 2K, 4K, 8K, 16K, 32K이고 constrained path는 2K, 4K, 8K, 16K이다. RULER base-model 규칙(79.4)은 별도 column에 보고한다.

**Worked example(설명용 score).** $N^*$가 4K에서 62.0점이면 $\tau=57.0$이다. Cell E1의 4K, 8K, 16K, 32K score가 각각 61.2, 59.8, 57.4, 31.0이면 effective length는 16K이다. Cell E3의 score가 60.9, 58.1, 55.2, 12.4이면 effective length는 8K이다. 16K score가 E1보다 2.2 point 낮고 $\tau$ 아래로 떨어지기 때문이다.

**Sample size.** 논문은 task당·길이당 500 example을 쓴다(§4). Task당 100개를 쓰면 길이당 1,300 item이며 accuracy 80%에서 half-width는 2.17 point이고, task당 500개면 0.97 point이다(item을 independent로 간주해 도출). Lab은 cell screening에는 task당 100개를 쓰고, 선택한 cell과 runner-up에는 500개를 쓴다.

**Length curve가 short score와 다르다는 근거.** Llama3.1(8B)은 4K에서 95.5점, 128K에서 77.0점이며 effective length가 32K다. Mistral-v0.2(7B)는 4K 93.6점에서 128K 13.8점으로 떨어진다(Table 3). RULER는 보지 못한 길이에서 “abrupt performance drops”가 있고, 최대 training context size 안에서는 “almost linear degradation with input length on log scale”가 있다고 적는다(§6).

### 6.2 Application category로서의 HELMET RAG

**정의.** HELMET의 RAG category는 Wikipedia에서 가져온 retrieved hard-negative passage 사이에 gold passage를 배치하고 substring exact match로 채점한다([[helmet]] §2.1).

**RAG를 쓰는 이유.** 저자들은 RAG가 실행하기 쉽고 base model에도 작동하며, synthetic recall보다 다른 downstream category와 더 강하게 상관하므로 빠른 development에 권한다. HotpotQA는 $\infty$Bench QA와 0.88 상관을 보이며, NIAH의 0.63보다 높다(Abstract; §3.1; Fig. 4). Category는 서로 다르다. ICL은 다른 category와 0.34–0.61 상관을 보므로(Fig. 5), 한 category는 gate이지 완전한 평가는 아니다.

**Protocol.**

1. 길이는 8,192와 16,384를 쓰고(E1-32K에는 32,768도 쓴다). HELMET은 Llama-2 token으로 길이를 정의한다(§3). 각 input을 checkpoint tokenizer로 다시 측정하고 어떤 input도 $L′$를 넘지 않도록 truncate한다.
2. Base checkpoint에는 HELMET의 two-shot demonstration을 쓴다. 128K에서 two-shot prompting은 Llama-3.1-8B base의 JSON KV score를 77.3에서 98.0으로 올렸다(Table 8).
3. Greedy decoding을 사용하고, App. D와 같이 NQ, PopQA, TriviaQA에는 각 600 sample, HotpotQA에는 300 sample을 쓴다.
4. 8K input reference point는 SmolLM2-1.7B base 47.17, Llama3.2-1B 42.13, Qwen2.5-1.5B 47.54이다([[smollm2]] Table 11).

### 6.3 이 gate가 사용하지 않는 것

$L′$에서의 perplexity는 기록하지만 cell 선택에는 쓰지 않는다. 100% long data를 선호했지만 downstream long score는 떨어졌기 때문이다([[prolong]] Fig. 1). Lu et al.은 exact-attention fine-tuned method에서는 PPL@32K가 downstream score와 상관한다고 발견했지만, approximate-attention method인 LM-Infinite는 32k에서 perplexity가 좋았는데도 4k 너머 NIAH에서 실패했다([[controlled-long-context-extension]] §6). Passkey retrieval도 기록한다. [[llama-2-long]]은 unmodified RoPE를 제외한 모든 variant가 완벽한 passkey accuracy에 도달했고, 이를 “too simple”하다고 보았다(§4.1, fn. 4).

## §7 Gate 표와 stage memo

**Gate 표(설명용 수치이며, lab은 측정한 값으로 채운다).**

| Cell | Paloma broad macro Δ% [CI] | 가장 나쁜 short suite Δ (point) [Bonferroni CI] | δ에 실패한 suite | RULER effective length (course rule / 79.4 rule) | RULER avg @16K | HELMET RAG @8K / @16K | Gate |
|---|---|---|---|---|---|---|---|
| N* | 0 | 0 | — | 4K / <4K | 38.0 | 30.1 / 11.5 | reference |
| Z | +6.1 [+5.2, +7.0] | HellaSwag −4.1 [−5.9, −2.3] | 3 | 8K / <4K | 44.9 | 33.0 / 25.2 | fail |
| E1 | +1.2 [+0.6, +1.8] | GSM8K −0.6 [−1.8, +0.6] | 0 | 16K / <4K | 57.4 | 36.8 / 34.0 | pass |
| E2 | +2.3 [+1.6, +3.0] | MMLU −2.5 [−3.6, −1.4] | 1 | 16K / <4K | 58.8 | 37.0 / 35.1 | fail |
| E3 | +1.9 [+1.2, +2.6] | ARC-C −2.2 [−4.9, +0.5] | 1 | 8K / <4K | 55.2 | 35.5 / 30.2 | fail |

**결정 절차.**

1. Bonferroni interval에서 어느 suite든 $-\delta$ 아래인 cell, 또는 memo에 적은 perplexity margin을 넘는 lower bound를 가진 Paloma broad 증가가 있는 cell을 제거한다.
2. 남은 cell 중 course-rule effective length가 가장 큰 cell을 고른다. 동점이면 $L′$에서의 HELMET RAG, 그다음 가장 나쁜 suite loss가 더 작은 쪽으로 동점을 푼다.
3. 선택한 cell과 runner-up을 두 번째 data order로 다시 실행하고, 두 run을 모두 보고한다.
4. 선택을 확정한 뒤 held-out suite를 한 번 평가한다.

**Memo 섹션.** (1) Setup 및 source별 token count. (2) Part A 표: interval과 control-to-control 차이를 포함한 arm별 $\Delta_{target}$ 및 $\Delta_{broad}$. (3) Gate 표. (4) load된 모든 setting을 포함한 선택 recipe. (5) “Scores lost”: 선택 cell의 point estimate가 $N^*$보다 낮은 각 short suite 및 Paloma domain, 그 interval, $δ를 통과했는지 여부. (6) §1 hypothesis를 supported, contradicted, undecided로 표시한다.

**Acceptance criteria.** Margin과 hypothesis는 Part B 전에 확정한다. Load time에 기록한 RoPE setting은 YaRN cell의 ramp bound 14 및 32를 포함해 의도한 값과 같아야 한다. 모든 cell에 per-item file이 있어야 하며, 평가 뒤에는 어떤 cell도 제외하지 않는다.

## 레시피

| Model(정확한 release) | Size | Stage | Setting | Value | Source 위치 | Status | 이 값의 근거 | Lab 값 | 차이의 이유 |
|---|---|---|---|---|---|---|---|---|---|
| OLMo 2 1B | 1B | pretrain-stable | tokens; schedule; warmup; peak LR; batch(sequences) | 4T tokens; cosine을 5T로 설정하고 4T에서 중단; 2,000 warmup steps; $4.0\times10^{-4}$; 512 | arXiv:2501.00656v3 App. B([[olmo-2-annealing]]) | 2026-09-15 검증 | 1B용 ablation은 보고되지 않음 | start checkpoint | — |
| OLMo 2 1B | 1B | pretrain-decay/anneal | anneal data; tokens | Dolmino Mix 1124; 50B-token anneal 한 번 | App. B | 2026-09-15 검증 | 1B용 ablation은 보고되지 않음(Table 11은 7B에서 mix를 비교) | arm당 5B tokens | 예산 안에서 세 arm과 repeat 하나를 수행; microanneal은 10B token 아래에서도 결과를 보였다(§4.4.2) |
| OLMo 2 7B microanneals | 7B | eval-gate | mixture; tokens; LR | 후보/web 약 50/50; 576M–12.6B tokens; LR을 선형으로 0까지 | §4.4.2, Table 12 | 2026-09-15 검증 | Table 12 | 50/50; 선형으로 0까지 | 없음 |
| OLMo 2 7B mid-training comparison | 7B | mid-train | init; tokens; control | 4T checkpoint; 50B tokens; “PT Mix” row | Table 11 | 2026-09-15 검증 | Table 11 | control arm N0에서 5B tokens | 예산 |
| Llama 3 8B(experiment) | 8B | eval-gate | anneal-as-data-evaluation | 50% 학습된 model; 40B tokens에 걸쳐 LR을 선형으로 0까지; new data 30%, default mix 70% | arXiv:2407.21783v3 §3.1.3([[llama-3-recipe]]) | 2026-09-14 검증 | ablation 미보고 | 후보 50% | OLMo 2 비율을 따른다 |
| SmolLM2 1.7B(ablation) | 1.7B | eval-gate | math anneal | 3T checkpoint; candidate 60B + pre-checkpoint tokens 40B; LR을 선형으로 0까지 | arXiv:2502.02737v1 §3.1([[smollm2]]) | 2026-09-15 검증 | Fig. 1, Fig. 5 | 사용하지 않음 | 예산 |
| SmolLM2 1.7B | 1.7B | long-context | length; RoPE; mixture; start | 2K → 8K; $\theta$ 10,000 → 130,000; 8K 이상 document 40%(DCLM 10%, FineWeb-Edu 10%, Dolma books 20%) + stage-4 mix 60%; 최종 75B tokens 전 checkpoint; extension token 수는 미보고 | §4.6, Table 6 | 2026-09-15 검증 | ablation 미보고 | 40% long arm | 더 낮은 비율로 사용 |
| ProLong-64k-Base | 8B | long-context | RoPE base | $8\times10^6$(NTK suggestion은 약 $4\times10^6$) | arXiv:2410.02660v4 Table 9, App. B.1([[prolong-recipe]]) | 2026-09-14 검증 | Table 18: long average 54.6 vs 48.7 vs 29.1 | NTK 값의 2배(16K에서 $4.09\times10^6$) | NTK 값에 대한 같은 비율; 1B에서 시험되지 않음 |
| ProLong ablations | 8B | long-context | tokens; length; long share; masking | 64K에서 5B; long 60% / ShortMix 40%; document masks | App. A.4, §3.2, Table 20 | 2026-09-14 검증 | Fig. 3; Table 20 | 16K에서 1B tokens; 40%와 70%; E5 외에는 masking on | 예산; ratio가 시험한 축이다 |
| ProLong-64k-Base | 8B | long-context | peak LR; warmup; decay; batch | $1e-5$; 10%; cosine to $1e-6$; 4M tokens | Table 9 | 2026-09-14 검증 | ablation 미보고 | $2\times10^{-5}$ | 아래 7B-scale extension row는 base checkpoint에서 $2\times10^{-5}$를 쓴다; ProLong의 최종 run은 Instruct model에서 출발한다 |
| Yarn-Llama-2 $s=16$ | 7B, 13B | long-context | LR; warmup; weight decay; steps; batch; $\alpha,\beta$ | $2\times10^{-5}$; 20 steps; 0; 400; 64; $\alpha=1,\beta=32$ | arXiv:2309.00071v3 §4.1, §3.2([[yarn]]) | 2026-09-14 검증 | Table 6(YaRN 400 steps $\approx$ PI 1,000 steps) | factor 4.0; `beta_fast` 32, `beta_slow` 1 | factor는 $L′/L_0$로 정한다 |
| LLaMA2-7B extensions(Lu et al.) | 7B | long-context | length; tokens; LR; weight decay | 32k; 1B; $2e-5$; 0 | arXiv:2409.12181v2 §4, App. 9.2([[controlled-long-context-extension]]) | 2026-09-14 검증 | App. 9.4: 2B tokens의 NTK-64K가 1B보다 NIAH를 개선 | cell당 1B tokens | 없음 |
| Llama 2 Long | 7B, 13B | long-context | peak LR; schedule; warmup; RoPE base | $2e-5$; cosine; 2,000 steps; 10,000 → 500,000 | arXiv:2309.16039v3 §2.1([[llama-2-long]]) | 2026-09-14 검증 | Tables 5–6, Fig. 5b(7B) | ABF cells | — |
| LLaMA 7B, 13B(PI) | 7B, 13B | long-context | peak LR; optimizer; warmup; weight decay | $2\times10^{-5}$; AdamW(0.9, 0.95); 20 steps; 0 | arXiv:2306.15595v2 §3.1([[position-interpolation]]) | 2026-09-14 검증 | ablation 미보고 | 모든 cell이 LR 공유 | — |
| Llama 3.1 405B | 405B | long-context | stages; tokens; advance criterion | 여섯 stage 8K → 128K; 약 800B tokens; short-context evaluation이 “fully recovered”하고 NIAH를 풀면 advance | arXiv:2407.21783v3 §3.4.2([[llama-3-recipe]]) | 2026-09-14 검증 | 수치 없음 | margin $\delta$를 둔 paired CI | 이 report는 수치 기준을 주지 않는다 |
| RULER | — | eval-gate | tasks; examples; threshold | 13 tasks; length당 task별 500; aligned model에는 threshold 85.6(Llama2-7B chat의 4K), base model에는 79.4(Llama2-7B base의 4K) | arXiv:2404.06654v3 §4, Table 3, App. A, Table 12([[ruler-effective-length]]) | 2026-09-15 검증 | 해당 없음 | screening 100/task, final 500; $N^*$ 상대 $\tau$ | 계산; small-model scale |
| HELMET | — | eval-gate | lengths; demos; RAG metric; samples | 8K–128K Llama-2 tokens; 2-shot; SubEM; 600(NQ, PopQA, TQA), 300(HotpotQA) | arXiv:2410.02694v3 §2–3, App. D([[helmet]]) | 2026-09-14 검증 | Fig. 4; Table 8 | checkpoint tokens 기준 8K와 16K RAG | tokenizer가 Llama 2와 다름 |

**Small general-purpose run의 시작점.** Data test로 쓰는 anneal에서는 candidate와 pretraining mix를 50/50으로 섞고 LR을 선형으로 0까지 내리며, 같은 token 수·schedule로 pretraining mix만 사용한 control도 학습한다. OLMo 2는 4T tokens를 학습한 7B checkpoint에서 576M–12.6B-token microanneal을 사용했다. Base checkpoint에서 더 긴 length로 continued pretraining할 때 PI, YaRN, Lu et al., Llama 2 Long은 모두 7B–13B Llama model에 peak LR $2\times10^{-5}$를 썼고, Lu et al.의 comparison은 32k에서 1B tokens를 썼다. 이 source들 중 7B 미만에서 이를 시험한 것은 없다. RoPE base에서는 ProLong의 64K·5B-token Llama-3-8B ablation run이 pretrained base $5\times10^5$에서 NTK-aware 값의 약 2배인 $8\times10^6$를 썼고, long 60% / short 40% data와 document mask를 사용했다. 이 설정은 8B에서만 시험됐다.

## 일반화 관점

**(a) Breadth를 늘리는 것.**
- Anneal에서 domain data를 작은 비율로 넣는 것: 1.72B tokens에서 math 10%는 7B의 GSM*를 28.5에서 61.0으로 올렸고 MMLU도 59.8에서 60.9로 올렸다([[olmo-2-annealing]] Table 12).
- Extension training에서 short data를 유지하는 것: short data 40%가 가장 좋은 long average를 냈고 더 큰 long share보다 short score도 높았다([[prolong]] Fig. 3). 고품질 short mix인 ShortMix는 FineWeb-Edu만 쓴 경우의 short average 63.0보다 높은 65.5를 냈다(Table 6).
- Data order를 달리한 anneal을 averaging하는 것: [[olmo-2-annealing]] Table 14에서 soup는 여섯 mix 모두 OLMES에서 best single run과 같거나 더 좋았다. 단, mix E의 GSM*는 예외다.
- Packed extension data에서 document masking을 적용하는 것: ProLong 비교 하나에서 SFT 뒤 측정한 long average가 +1.0, short average가 +0.6이었다(Table 20).

**(b) Narrowing 또는 forgetting을 일으키는 것.**
- Anneal data에 benchmark training set을 넣는 것: 8B에서는 GSM8k가 +24.0%였고 405B에서는 효과가 negligible했다. Llama 3는 “true few-shot learning capabilities and out-of-domain generalization”을 평가하려고 이를 제외했다([[llama-3]] §3.1.3).
- 더 큰 extension factor로 position을 rescaling하는 것: 32,768까지 PI한 7B는 원래 window 안의 BoolQ가 76.1에서 64.7로 낮아졌다([[position-interpolation]] Table 5). $s=32$에서 13B YaRN의 MMLU도 55.8 → 51.9였다([[yarn]] Table 3).
- Long-only training: SFT 뒤 recall과 RAG는 SFT 전에는 올라간 경우에도 long data가 많아질수록 떨어졌다([[prolong]] §3.2).
- Average가 숨기는 task-level loss: YaRN에서 LongBench LCC는 68.22 → 54.06이었지만 average는 32.92 → 33.45로 올랐다([[controlled-long-context-extension]] Tables 4, 11).

**(c) 이 stage에서 이를 측정하는 방법.**
- 사전 등록한 margin으로 stage 시작 checkpoint에 대한 item별·document별 paired difference를 계산한다(§5; [[adding-error-bars-evals]] §4.2).
- 모든 anneal에 control arm을 두고 run-to-run variation을 위해 control을 반복한다(§2, §3.3; [[signal-and-noise-eval]] §3.1).
- Document를 따로 score한 per-domain perplexity를 쓴다([[paloma]] G5, Table 17).
- Perplexity나 passkey retrieval 대신 task family와 application category 하나로 effective length를 평가한다(§6; [[helmet]] Fig. 4; [[prolong]] Fig. 1).
- Final model에 영향을 주는 long-context 결정을 위해서는 SFT 뒤에도 평가한다. ProLong의 RAG 및 re-ranking trend는 SFT 뒤에만 나타났다([[prolong]] §2.2, Fig. 2). 이 lab은 base checkpoint를 gate하며, [[ch-30b]]는 SFT에서 long-context share를 다룬다.

## 흔한 실수와 탐지 방법

| Mistake | 관측 가능한 증상 | Check |
|---|---|---|
| Control anneal이 없음 | 모든 candidate가 비슷한 양만큼 broad score를 “개선”한다 | N0 arm; candidate − control을 보고(§2.2) |
| Candidate에 benchmark training split이 들어 있음 | held-out partner gain 없이 target gain만 있음 | decontamination report; MATH 및 MMLU-Pro partner(§2.3, §3.2) |
| 200-item subset으로 broad effect 결정 | ±1–2 point 변화를 effect로 보고 | 사용 전 half-width 계산: 200개에서 28.5%이면 6.3 point(§2.4) |
| Perplexity evaluation에서 document를 concat함 | evaluation subset 사이 variance가 큼 | document를 개별 score([[paloma]] Table 17) |
| transformers v4.46.0에서 `max_position_embeddings`를 $L′$로 설정한 YaRN | OLMo 2 1B에서 ramp bound가 14·32가 아니라 21·39 | load time에 bound를 log하고 figure 값과 대조(§4.2) |
| YaRN attention factor 없이 short-context gate 실행 | training-time과 serving config 사이 short score가 다름 | 각 cell을 정확히 load된 config로 평가(§5.2) |
| $L′$ perplexity로 ratio나 method 선택 | 70%-long cell을 고르고 short score와 RAG가 더 낮음 | §7 결정 절차는 RULER, RAG, paired short suite를 사용 |
| 1B model에 RULER absolute threshold(base 79.4, aligned 85.6)를 적용 | 모든 cell이 “<4K”를 받음 | absolute rule 옆에 relative threshold $\tau$를 보고(§6.1) |
| HELMET length를 checkpoint token으로 간주 | input이 $L′$를 넘거나 그보다 짧음 | input length를 다시 tokenize해 log(§6.2) |
| Long pool을 cell마다 다른 횟수로 반복 | repeat를 같게 만들면 method 차이가 달라짐 | `runs.jsonl`에 cell별 repeat count 기록(§4.3) |
| Average만 보는 short gate | 한 suite가 $\delta$를 넘게 떨어져도 cell이 통과 | suite별 Bonferroni interval(§5.2) |
| Masking comparison에서 position id도 바꿈 | masking 효과와 position reset이 confound됨 | E1과 E5의 `position_ids` policy를 log(§4.4) |
| 숫자 없이 “no degradation”을 보고 | memo에 scores-lost list가 없음 | memo section (5)를 필수로 둠(§7) |

## 이해 점검

1. Math arm이 control보다 GSM8K를 20 point 올렸는데 MATH는 변하지 않았다. 가능한 causal account 두 개를 제시하고, 무엇이 이를 구분하는지 lab file을 말하라.
2. 변하지 않은 pretraining mix에서 anneal했는데도 MMLU가 오르는 이유와, 왜 그 결과가 candidate − start comparison을 candidate 쪽으로 편향시키는지 설명하라.
3. NTK-aware formula를 사용해, 64K의 Llama-3-8B에 제안되는 base가 왜 $4\times10^6$에 가까운지 설명하라. ProLong Table 18에서 base가 그 두 배여도 short score를 낮추지 않고 long-task score를 올릴 수 있는 mechanism 하나를 제시하라.
4. YaRN cell이 16K RULER는 통과했지만 512-token input의 HellaSwag margin에는 실패했다. §4.2 transformers code를 따라 short length에서 attention을 바꾸는 지점을 추적하고, configuration change 하나와 이를 검증할 관측값을 제안하라.
5. Cell E2(70% long)는 E1(40% long)보다 16K RULER average가 높지만 short suite 하나에서 실패한다. ProLong의 SFT 전·후 결과를 사용해 E2를 선택하면 SFT 뒤 long-context score도 낮아질 수 있는 이유를 설명하라.
6. Worked example 1에서 interval은 zero를 포함하지만 cell은 실패한다. “interval이 zero를 포함한다”가 왜 regression이 없다는 증거가 아닌지, margin $\delta$가 제어하는 양이 무엇인지 설명하라.
7. Course rule은 $L_0$의 $N^*$에 상대적으로 effective length를 정의한다. 한 cell이 다른 cell보다 relative effective length는 길지만 더 나쁜 long-context model인 경우를 설명하고, 이를 감지하는 gate column을 말하라.
8. E5(masking off)가 E1보다 RULER는 높지만 short suite는 낮다. 관련 없는 document를 가로지르는 attention과 position id를 포함한 mechanism 하나, 그리고 둘을 분리하는 experiment를 제안하라.

## 연결

- 이전 장: [[ch-32e]] — Mid-Training, Annealing, and Context-Extension Recipes Side by Side.
- 다음 장: [[ch-18]] — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix.
- [[ch-32]] — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL(Part A에서 annealing data와 stage gate를 실행한다).
- [[ch-32a]] — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining(start checkpoint가 이미 decayed했을 때 re-warming).
- [[ch-32b]] — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression(Part B에서 방법을 실행한다).
- [[ch-32c]] — Claimed versus Effective Context Length and Long-Context Evaluation(RULER와 HELMET을 자세히 다룬다).
- [[ch-00]] — What General Capability Means and How It Is Measured(development 및 held-out suite).
- [[ch-04]] — Sequence Packing, Loss Masking, and Chat Templates(document masking 및 position id).
- [[ch-06]] — Checkpointing, In-Loop Evaluation, and Checkpoint Selection(run-to-run variation 및 checkpoint choice).
- [[ch-36]] — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split(paired bootstrap, seed).
- [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares(SFT를 거치며 long-context ability를 유지하는 법).
- [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions(multiple comparison과 decision rule).

## 출처

- [[olmo-2-annealing]] — OLMo 2 v3의 chapter excerpt: Table 3, §4.1, Table 11 control row, §4.4.2 microanneal recipe 및 Table 12, Table 14 soups, App. B 1B settings.
- [[smollm2]] — SmolLM2 v1의 chapter excerpt: annealing ablation setup, FineMath 및 Table 1 web result, decay stage, RoPE 130k와 long document 40%를 쓴 2K → 8K extension, HELMET Table 11, 360M/135M config.
- [[prolong]] 및 [[prolong-recipe]] — long:short ratio(Fig. 3), perplexity와 downstream의 불일치(Fig. 1), SFT 뒤 evaluation(Fig. 2), RoPE base(Tables 18–19), document mask(Table 20), 이전 방법의 short-context loss(Table 2), recipe row.
- [[yarn]] — NTK-aware base formula, NTK-by-parts ramp, temperature, short-benchmark cost(Tables 2–3), recipe row.
- [[hf-rope-utils-v4460]] — transformers v4.46.0 dynamic NTK 및 YaRN code, attention factor가 적용되는 위치의 chapter excerpt.
- [[controlled-long-context-extension]] — Lu et al.: PI, YaRN, NTK variant의 fixed-recipe comparison; RULER, 2k PG19, LongBench LCC drop.
- [[position-interpolation]] — PI와 original-window regression(Table 5), $2\times10^{-5}$ LR row.
- [[llama-2-long]] — ABF 정의와 7B PE ablation(Tables 5–6, Fig. 5b), recipe row.
- [[llama-3]] 및 [[llama-3-recipe]] — anneal-as-data-evaluation, benchmark-train-set annealing result, document mask, long-context stage criterion.
- [[ruler]] 및 [[ruler-effective-length]] — task generator; effective-length rule, aligned-model Table 3 및 base-model Table 12 row.
- [[helmet]] — RAG category, downstream task와의 correlation, two-shot base-model evaluation, inference-time position change(App. E.3).
- [[paloma]] — per-domain perplexity, macro average, separate-document evaluation(G5), perplexity-to-task correlation의 한계.
- [[signal-and-noise-eval]] — 1B의 checkpoint noise와 small-scale decision용 bits-per-byte scoring.
- [[cooldown-scaling-beyond-fixed-durations]] — cooldown shape comparison(linear vs 1-sqrt).
- [[adding-error-bars-evals]] — Miller paired standard error(Eq. 7)의 chapter excerpt.
