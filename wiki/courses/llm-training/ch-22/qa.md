<!-- chapter: ch-22 Q&A; deps: [[read]]; kernel answers only -->
# Ch-22 — Reading Q&A

## Q1. AlpaGasus — 가장 단순한 selection filter

### 기본 정보
2023, Alpaca의 follow-up. Question: Alpaca 52K SFT 데이터를 quality filter로 줄이면 더 좋아질까? Answer: YES — 9K로 *full 52K보다 우수*.

### Mechanism
GPT-3.5에 (instruction, input, response) tuple 보내고 rubric prompt로 0-5 rating:
- Relevance / Factual correctness / Completeness / Format appropriateness 각 0-5
- 평균 ≥ 4.5만 keep
→ 52K → 9K (top 17%) 생존

### 결과
| Dataset | Size | Performance | Training time |
|---|---|---|---|
| Full Alpaca | 52K | Baseline | 1× |
| **AlpaGasus** | **9K** | **모든 benchmark 우수** | **5.7× 빠름** |

→ 43K samples가 *negative marginal value* (drag model down). [[ch-21#Q9]] (zero noise) 직관의 strong evidence.

### Threshold 4.5의 *왜* (chapter line 41)
4.0은 *slightly worse*가 아니라 **rater의 uncertainty boundary**:
- 5.0/4.5 = rater confident GOOD
- 4.0 근처 = rater UNCERTAIN, error 집중
- 3.5/3.0 = rater confident BAD

→ 4.5 cutoff = *rater의 confident-good 영역*만 추출. 4.0 영역은 rater 본인이 헷갈리는 곳이라 noise/error 많음. [[ch-21#Q7]] verifier framework 적용: verifier의 *certain* 영역만 사용해야 효과적.

### Framework 매핑
- [[ch-20]] 9-axis: axis 5 verifier의 *LLM-judge 형태* (continuous, weak). Sky-T1과 같은 category.
- [[ch-21]] 4-axis: verifier 존재 ✓ but GPT-3.5 rater에 bound
- Cost spectrum: API only — 가장 저렴 (gradient도, target PPL도 없음)

### Limits
1. **Judge bias inheritance**: length / style / format / sycophancy → student 상속 ([[ch-20]] axis 5 quirks)
2. **Domain coverage 못 함**: math (계산 못 함), code (실행해 봐야), long-context (judge attention 짧음)
3. **Chatty IF에서만 작동**

Chapter rule (line 39): *"stop using AlpaGasus when capability the rater cannot reliably evaluate"*.

### 더 큰 thesis
AlpaGasus 결과가 implies: **bad samples ≠ neutral filler, bad samples = 학습을 *해치는* signal**. 이게 ch-22 전체의 organizing principle — "selection is the binding constraint, not quantity."

### 한 줄
**"GPT-3.5 0-5 rating + ≥4.5만 keep. 52K→9K로 우수. 4.5는 rater의 confident-good 영역, 4.0은 uncertainty boundary라 error 집중. Negative marginal value thesis의 origin."**

## Q2. Alpaca 52K score 분포는? — Right-skewed, mode at 4.0 (NOT mostly 3)

### 실제 distribution (AlpaGasus paper Fig 3)
```
5.0      ~15-20%
4.5      ~15-20%
4.0      ~25-30%  ← 모드
3.5      ~15-20%
3.0      ~10%
< 3.0    ~5%
```
4.0-4.5 구간이 전체의 ~50%. 1.0-2.0 거의 없음.

### Right-skewed 3 mechanisms
1. **Generator = Rater (같은 family)**: Alpaca generator GPT-3.5 + AlpaGasus rater GPT-3.5-turbo → 자기 output 더 너그럽게 평가
2. **LLM-judge bias 일반**: length / style / sycophancy bias → 평균 너그러움
3. **Generation pipeline self-selection**: Self-Instruct가 이미 fail sample reject → 도착한 52K가 upper half piled up

### Threshold 민감도 — 0.5 step → corpus 3× 차이
```
≥ 4.5: ~17%  → 9K
≥ 4.0: ~50%+ → 27K+
≥ 3.5: ~70%  → 36K+
```
4.0이 mode = rater uncertainty 영역 → AlpaGasus가 mode를 *피해* cutoff 놓은 진짜 이유.

### Generator-specific (universal NOT)
| Generator | Shape | Mode |
|---|---|---|
| Alpaca (GPT-3.5 self-instruct) | Right-skewed | ~4.0 |
| LIMA (hand-curated) | Very right-skewed | ~5.0 |
| Raw web scrape | Left-skewed | ~2.0-2.5 |
| GPT-4 generated | Extremely right-skewed | ~4.5+ |
| Multi-teacher | Bimodal | varies |

→ Chapter line 25: *"keep threshold a hyperparameter, not a fossil"*.

### Framework extension 후보
[[ch-21#Q11]] 4-axis predictor에 새 axis 후보: **Distribution shape of rater scores**. Filter design은 pool의 score distribution을 먼저 *측정*한 후 threshold 결정해야 함. AlpaGasus의 hidden lesson.

### 한 줄
**"Most ≈ 4.0 right-skewed. Generator=rater family, judge bias, self-selection이 원인. 4.0 = mode = uncertainty boundary. 4.5 cutoff = mode를 *피해서* 잘라낸 결정. 0.5 step → corpus 3×. Generator-specific."**

## Q3. IFD (Instruction-Following Difficulty) — Self-perplexity ratio

### Formula
```
PPL_cond(a|q) = student의 instruction-given response surprise
PPL_uncond(a) = student의 instruction-없는 response surprise
IFD = PPL_cond / PPL_uncond
```

### 3 regimes
| IFD | 의미 | 학습 가치 |
|---|---|---|
| **< 1** | Instruction이 prediction에 도움 (load-bearing) | **HIGH** |
| ≈ 1 | Instruction 무관 (boilerplate) | 0 |
| **> 1** | Instruction이 *방해* (mismatched, drop) | drop |

### 구체 example
- `< 1`: Python factorial — q가 code structure 강하게 predict, learning signal 있음
- `≈ 1`: "Tell me about something" → boilerplate response, q 무시
- `> 1`: q "integrate x²" + a "cake recipe" → Self-Instruct가 잘못 glue한 random pair

→ **IFD > 1은 Self-Instruct failure mode (random q + random a glue) signature**. 가장 잘 잡는 영역.

### Cherry-LLM keep zone
Top 5-15% just below 1.0 (hard but learnable, *sweet spot*). Trivial(IFD≪1)도 boilerplate(IFD≈1)도 아님.

### What IFD captures
- Boilerplate
- Instruction-response mismatch (synthetic pipeline failure)
- Trivial instructions

### What IFD misses (chapter line 64)
- **Factual correctness**: *"beautifully-conditioned lie has low IFD"*
- Plausible hallucination이 통과 → wrong-question-correctly의 *selection-side* manifestation ([[ch-21#Q10]] limit A의 selection 버전)
- **Defense**: IFD + verifier compose (math/code: unit test, open-ended: AlpaGasus rating)

### Warmup 필수 (chapter line 66)
Cold model → PPL_uncond badly calibrated on synthetic format → IFD가 *template-mismatch noise*에 dominated. Fix: 1-epoch warmup on ~1K samples → output distribution align with pool format.

→ [[ch-21#Q11]] framework로: warmup = *substrate alignment* 단계. Student가 pool format을 이미 이해해야 conditioning signal 측정 가능.

### Cost
2 forward passes per sample. ~1 epoch SFT cost on 7B. AlpaGasus(API)보다 비싸지만 DEITA scorer-training보다 싸고 LESS gradient datastore보다 훨씬 쌈.

### Framework — Student-internal verifier

지금까지의 verifier:
- External binary: math-verify ([[ch-20]])
- External continuous trained: RM (Nemotron)
- External LLM-judge: AlpaGasus
- **Internal self-perplexity: IFD** ✓

→ External teacher 없이도 selection 가능. [[ch-20]] verdict E6 (pass@k self-improvement)와 *isomorphic*:

| | pass@k ([[ch-20]]) | IFD ([[ch-22]]) |
|---|---|---|
| Target | New generation을 self-evaluate | Existing pool을 self-evaluate |
| External teacher | 없음 | 없음 |

→ 둘 다 *self-referential selection* family.

### Superfiltering — IFD ranking scale-invariant (chapter §4)
GPT-2-125M의 IFD와 Llama-2-7B의 IFD가 high Spearman ρ. 절대값 다르지만 ranking 일치 → tiny proxy로 large target filter, 20× cheaper.

→ IFD는 *data-intrinsic signal* (model-invariant). Verifier capability (model-specific)보다 transferable. [[ch-21#Q11]] 4-axis에 *signal portability* axis 후보.

Caveat: family-mismatched proxy는 깨짐.

### 한 줄
**"IFD = PPL(a|q)/PPL(a) ratio. Student 자신이 scorer (self-perplexity). <1=load-bearing, ≈1=boilerplate, >1=mismatched. Self-Instruct failure mode 잡는 데 특히 강함. Correctness는 못 잡음 → verifier와 compose. Scale-invariant ranking → 125M proxy로 7B filter 가능."**

## Q4. IFD의 "condition"은 few-shot인가? — NO, zero-shot conditioning

### 차이 = q가 prefix에 있는가 없는가, 그게 전부

**PPL_uncond**: `Input = <BOS> a` → model이 *답변만* 보고 next token predict
**PPL_cond**: `Input = <BOS> q a` → model이 *q + 답변* 보고 next token predict

→ 두 perplexity는 *같은 response a*를 *prefix 차이*로 두 번 평가.

### Few-shot과의 차이 — Prefix 크기

```
Zero-shot (IFD):   <BOS> q a_1 a_2 ...
                         └─ 단 현재 sample의 q 하나
Few-shot:          <BOS> (q1,a1) (q2,a2) (q3,a3) q a_1 a_2 ...
                         └─ 다른 demo 3개 ─┘ └─ 현재 ─┘
```

→ IFD는 *현재 sample의 q*만 prepend. Few-shot은 *여러 demo pair* prepend.

### 왜 zero-shot인가 — 학습 setup과 일치
Training loss: `Loss = -log p(a | q)` (autoregressive on response, given q).
IFD's PPL_cond = *학습 loss와 같은 conditional setup*. 그래서 IFD가 *학습 가치* 직접 측정.
Few-shot은 *inference-time prompting* 별도 paradigm → 학습 loss와 다른 quantity → IFD framework 외부.

### 가능한 confusion
"Conditional"이 *example로부터 학습*하는 느낌이라 few-shot 연상. 사실 *현재 sample의 q*만 context = *가장 minimal* conditioning form.

### 한 줄
**"IFD의 'condition' = q를 prefix prepend만 (zero-shot). Few-shot은 별도 inference paradigm. IFD cond는 학습 loss setup과 *exact mirror*라 학습 가치 측정 valid."**

## Q5. IFD를 Orca-2 dataset에 적용 — Cross-chapter synthesis + 2 caveats

### ✓ 학습자 idea 정확한 application
Orca-2 dataset = IFD가 가장 잘 작동하는 pool 종류:
- Synthetic generated (teacher emit)
- Reasoning trace boilerplate 가능성 큼 ("Let me think step by step..." template)
- Self-Instruct family pipeline

IFD가 Orca-2에서 catch:
1. **Boilerplate reasoning** (generic preamble, question 무관) → IFD ≈ 1 → drop
2. **Strategy-question mismatch** (teacher가 wrong strategy 적용) → IFD 부근 1
3. **Trivial questions에 long reasoning** → IFD ≈ 1

### ⚠ Caveat 1 — Wrong-but-smooth reasoning 못 잡음
Chapter line 64: *"beautifully-conditioned lie has low IFD."*

Reasoning context의 *catastrophic* failure:
- Step 2가 wrong (17×3=41) but 최종 답 우연히 correct (391)
- IFD: 매우 낮음 (reasoning이 question에 형식적으로 잘 conditioned)
- Verifier-correctness: 통과 (outcome only)
- 실제 quality: bad reasoning

→ [[ch-21#Q10]] limit A (*wrong-question-correctly*)의 *reasoning 버전*. IFD alone은 [[ch-20]] axis 5 quirks inheritance 그대로 통과. Defense: IFD + verifier compose (math: unit-test, open: AlpaGasus).

### ⚠ Caveat 2 — Strategy diversity bias (framework extension candidate)

Orca-2 *intentional value* = 5 reasoning strategy 다양성 + prompt erasing → strategy auto-select 학습.

IFD-filter가 만드는 unintended consequence:
- "Direct answer" strategy = question에 직접 답 → high q-a coupling → 매우 낮은 IFD → 우선 선택
- "Step-by-step reflection" = template-heavy preamble → 일부 boilerplate-like → 일부 drop
- "Self-critique" = "Let me reconsider..." 반복 → boilerplate signature → 일부 drop

→ **IFD-filter가 Orca-2의 strategy diversity를 *bias***. Orca-2 value 핵심이 filter 후 축소.

### Framework extension 후보 — Augmentation-Selection interaction

**같은 pool에 두 다른 axis 동시 적용**:
| Axis | Generation 시 | 사용 전 |
|---|---|---|
| Augmentation | 5 strategy 다양화 ([[ch-19]]/[[ch-20]]) | — |
| Filter | — | IFD selection ([[ch-22]]) |
| Direction | Variety 증가 | Density 증가 |

→ 데이터 pipeline = generation axes + selection axes의 sequential application. Naive하게 둘 다 적용하면 augmentation intent 손상 가능. **Filter design 시 augmentation의 source variance 고려 필수**. [[ch-21#Q11]] 4-axis에 *Augmentation-Selection interaction* axis 후보.

### Operational recipe
```
1. Warmup: M.SFT(random 1K, 1 epoch) for PPL_uncond calibration
2. Score: ifd[q,a] = M.PPL(a|q) / M.PPL(a) for all samples
3. Compose with verifier:
   - Math: keep if (ifd<1 AND boxed_value == gold)
   - Open: keep if (ifd<1 AND alpagasus_score >= 4.5)
4. Strategy balance: per-strategy retention 모니터링, threshold 조정
5. SFT M on filtered subset
```

### 한 줄
**"Orca-2 dataset = IFD가 가장 잘 작동하는 pool. Idea valid. 단 (1) wrong-but-smooth reasoning은 verifier compose 필수, (2) 5 strategy diversity가 IFD bias될 수 있어 monitoring 필요. 통찰: augmentation + selection axes orthogonal하지만 interact함."**

## Q6. Superfiltering (Weak-to-Strong) — IFD ranking이 model scale 너머 transfer

### 이름 분해
- Weak = 작은 모델 (GPT-2-125M)
- Strong = 큰 모델 (Llama-2-7B, 56× 큼)
- Weak-to-Strong = *weak가 strong을 위한 data를 골라줌*

전통: strong이 weak를 가르침 (distillation). Superfiltering은 *반대* — knowledge 흐름은 strong→weak이지만, *data scoring은 weak→strong* 가능. 비대칭 use.

### 핵심 finding
```
1M Alpaca samples:
  IFD_125M:  0.42, 0.91, 0.65, ...  (절대값 큼)
  IFD_7B:    0.18, 0.74, 0.31, ...  (절대값 작음)
  Ranking:   같음 (Spearman ρ 0.7-0.9)
```

Filter는 *ranking만* 필요 (top X% 선택) → 125M score로 충분.

### Why ranking transfers — Data-intrinsic signal

| Signal | 의미 | Transfer? |
|---|---|---|
| Data-intrinsic | "sample이 *원래* informative한가" | ✓ |
| Model-specific | "model이 이 sample 유용하다고 보는가" | ✗ |

IFD 본질:
- Bad sample (random q+a glue): 모든 model에서 IFD ≈ 1
- Good sample (q-a coupled): 모든 model에서 IFD < 1

→ Quality는 *data의 속성*. Model size 무관 같은 ranking. Chapter line 74: *"That invariance is rare among data-selection signals; it is the property that makes IFD industrially practical."*

### Cost saving — 20× cheaper filter
- Pool 1M samples, 2 forward pass each
- 7B score: baseline
- 125M score: 1/56 base cost ≈ 20× cheaper
- **Selection compute가 Training compute보다 큰 경우 흔함** (large pool, small SFT) → filter cost 절감이 전체 pipeline 결정적

### Caveat — Family mismatch면 깨짐 (chapter line 76)
- 다른 tokenizer (GPT-2 BPE vs Llama SP) → PPL 비교 불가
- 다른 training distribution (English-only proxy로 multilingual score 못 함)
- 다른 specialty (Code proxy로 chat pool 못 함)

올바른 proxy: 같은 family 작은 버전, 비슷한 pretraining, 비슷한 tokenizer.

### [[ch-21#Q11]] framework 확장
Q3에서 *signal portability* axis 후보 surface — Superfiltering이 *empirical evidence*. "Internal" verifier의 정의 확장: strict internal (target 자체) → loose internal (target *family*의 어느 model이든).

### 한 줄
**"125M IFD ranking ≈ 7B IFD ranking (절대값 다름, 순서 같음). Filter는 순서만 필요 → 작은 proxy로 큰 target SFT data 선택, 20× cheaper. 가능 이유: IFD는 data-intrinsic signal. Family mismatch면 깨짐."**

## Q7. Spearman correlation ρ — Rank-based correlation

### 정의
두 변수의 *rank(순위)* 사이의 monotonic 관계. -1 ~ +1.

| ρ | 의미 |
|---|---|
| +1.0 | Ranking 완벽 일치 |
| +0.7~0.9 | 강한 일치 (Superfiltering 영역) |
| 0 | Rank 관계 없음 |
| -1.0 | 완벽한 *역* 일치 |

### Pearson과 차이
- Pearson: *raw value* 기반 linear 관계
- Spearman: *rank* 기반 monotonic 관계 → 비선형이어도 OK, scale-invariant

```
y = x³ (단조 증가 비선형):
  Pearson(x,y):  낮음
  Spearman(x,y): 1.0
```

### Why Superfiltering uses Spearman (not Pearson)
125M IFD와 7B IFD의 *absolute value*는 systematically 다름 (7B가 모든 sample 낮은 IFD 도출). Pearson은 scale 차이에 misleading. Spearman은 *rank만* 보니 transfer claim에 정확한 metric.

```
Sample | IFD_125M | IFD_7B | 같은 rank?
A      | 0.42     | 0.18   | yes
B      | 0.91     | 0.74   | yes
...
Spearman ρ = 0.85 → top 15% set의 ~70-80% overlap
Pearson r ≈ 0.5  → scale 차이로 misleading
```

### Filter 결정은 *rank-based*
| Method | Value vs Rank | Transfer 가능? |
|---|---|---|
| AlpaGasus (≥4.5 threshold) | Value (absolute) | No (threshold가 value 기반) |
| Cherry-LLM IFD (top 15%) | Rank | Yes |
| DEITA (lexicographic) | Rank | Yes |
| LESS (cosine sim top 5%) | Rank | Yes |

→ Rank-based method만 weak-to-strong transfer 가능. 이게 chapter §4의 implicit design lesson.

### 한 줄
**"Spearman ρ = rank 기반 correlation, -1~+1. Pearson은 value, Spearman은 rank. Superfiltering이 Spearman만 보고: 125M과 7B IFD 절대값 systematically 다르니 Pearson misleading. Filter는 rank-based라 *Spearman이 정확한 metric*. ρ 0.7-0.9 = top X% set의 70-80% overlap."**

## Q8. Spearman ρ — 계산 step-by-step + Superfiltering 적용 validation

### Computation (5 sample toy)
1. **Score data 준비**: 같은 sample을 두 모델로 score
2. **Rank 변환**: 각 score list를 rank로 (낮음→1, 높음→N)
3. **Rank 차이 d_i = rank_A - rank_B**, 그 다음 d_i² 합산
4. **Formula**: `ρ = 1 - 6·Σd_i² / (n(n²-1))`

예시: 5 sample, Σd_i² = 0 → ρ = 1.0 (완벽). Σd_i² = 6 → ρ = 0.70 (강함).

### Interpretation
| ρ | Top X% overlap |
|---|---|
| 1.0 | 100% |
| 0.8 | ~75-85% |
| 0.7 | ~65-75% (Superfiltering 임계) |
| 0.5 | ~50-60% (transfer 약함) |
| 0 | random |

### Superfiltering 적용 validation procedure

**Pool 1M, target 7B, proxy 125M인 case 평가**:

```
Step 1: random 500-1000 sample 추출 (calibration subset)
Step 2: subset을 두 모델로 score (cheap)
Step 3: scipy.stats.spearmanr(ifd_weak, ifd_strong) → ρ
Step 4: Decision rule
   ρ > 0.7  → ✅ Use weak proxy for full filter
   0.5~0.7  → ⚠ Composite (proxy first pass, target final)
   < 0.5    → ❌ Score with target
Step 5: Top-15% direct overlap 추가 확인 (sanity check)
Step 6: Apply decision to full 1M pool
```

→ 이 6-step이 production-grade Superfiltering 적용 방식.

### IFD vs Spearman — 다른 layer

| | IFD | Spearman ρ |
|---|---|---|
| Measures | Sample quality | Two scorers ranking 일치 |
| Input | (q,a) pair | Two score lists |
| Use | Per-sample filter | Filter design validation |
| Layer | Filter time | Filter design time |

→ IFD = filter. Spearman = filter design의 *cross-validation meta-metric*. 둘 다 필요 다른 layer.

### 한 줄
**"Spearman ρ 계산 5-step: score→rank→d_i→d_i²합→formula. Superfiltering 적용 6-step: subset 추출→두 모델 score→ρ 계산→threshold 분기→overlap sanity→full pool 적용. IFD는 per-sample filter, Spearman은 design-time cross-validation. 다른 layer."**

## Q9. IFD 충분한데 왜 Superfiltering? — Purely cost optimization

### 핵심 — Quality 동일, *cost*만 차이
IFD with target = quality 보장. Superfiltering = *같은 결과, 50× cheaper*. *Pool size scale*이 binding constraint일 때만 의미.

### Pool size별 regime
| Pool size | IFD with 7B cost | Superfiltering 필요? |
|---|---|---|
| 1K (LIMA) | ~$5 | No, 그냥 7B |
| 50K (Alpaca) | ~$200 | No, 7B 직접 OK |
| 1M (Orca-2) | ~$5K | Yes, 큰 절감 |
| 100M (pretrain) | ~$500K | Critical |

### Filter cost > Training cost (1M pool)
```
1M samples × 2 forward × 7B = 28 PFLOPs (filter)
150K × 1 epoch × 7B × 3 (forward+backward) = 4.5 PFLOPs (SFT)
→ Filter 6× 더 비쌈
```

Chapter §3-§4 transition의 의미: §3 *quality*, §4 *operational scaling*. Filter cost가 *pipeline cost를 결정*하는 regime이 존재.

### Superfiltering 의미 있는 3 조건
1. Pool size > ~100K (filter cost 눈에 띄게 큼)
2. Target model big (7B+)
3. Recipe iteration 많음 (proxy로 한 번 score → threshold sweep cost ≈ 0)

위 조건 안 맞으면 IFD with target이 *simpler + 충분*.

### Hidden benefit — Recipe iteration
```
7B IFD: threshold sweep마다 1M × 7B forward → 매번 28 PFLOPs
125M proxy: 한 번 score → re-rank만 하면 됨 → ≈ 0 추가 cost
```

→ Cost 절감 + research iteration 속도. *큰 pool에서 critical*.

### ⚠ Caveat — 작은 pool에선 오히려 안 좋음
Pool < 10K + family-mismatched proxy: Spearman ρ noise level → ranking 부정확 → quality 손해. Superfiltering은 "scale로 인한 cost > scale로 인한 noise"인 regime에서만 유용.

### [[ch-21#Q11]] framework 확장
*Operational cost* axis 후보: pool size가 framework input 중 하나가 되어야. 4-axis (verifier/taxonomy/long-tail/substrate)에 *pool scale* 추가.

### 한 줄
**"IFD quality 충분, Superfiltering은 *cost only*. Pool < 100K → 불필요. Pool > 1M → filter cost가 training 능가 → critical. 학습자 직관 정확: 작은 case면 IFD with target이 simpler."**

## Q10. DEITA vs AlpaGasus — "차원 추가"가 부분 정확, 5 structural difference

### Difference 1 — Axes 수 (학습자 framing)
| | AlpaGasus | DEITA |
|---|---|---|
| Axes | 1 (quality) | 3 (complexity / quality / diversity) |

### Difference 2 — Sample-level vs Set-level (가장 깊음)
- AlpaGasus: 각 sample 독립 평가, parallel 가능
- DEITA complexity + quality: sample-level (parallel)
- **DEITA diversity: set-level — previously selected와 embedding-distance 비교 → sequential**

→ AlpaGasus는 *sample 속성*만. DEITA는 *sample 속성 + 집합 구성*.

### Difference 3 — Combination 방식
- AlpaGasus: 4 sub-criteria *averaged into scalar* (weighted sum)
- DEITA: *Lexicographic priority* — quality×complexity sort, diversity는 hard constraint
- DEITA insight: diversity는 *trade off 안 됨*. Weighted sum이면 near-dup collapse. Strict priority가 답.

### Difference 4 — Scorer (operational)
- AlpaGasus: External API per sample
- DEITA: ChatGPT ranking → 13B distilled scorer 학습 → 자체 inference. One-time distillation + amortized cheap scoring.

### Difference 5 — Failure mode
| AlpaGasus | DEITA가 추가로 다룸 |
|---|---|
| Bad response | + individually good but collectively redundant |
| | + correct but trivial (low complexity) |
| | + high quality but narrow coverage |

예: 1000개 동일 instruction의 perfect response → AlpaGasus 다 keep, DEITA 50-100개만.

### 결과
| | AlpaGasus | DEITA |
|---|---|---|
| Pool | 52K | 300K |
| Selected | 9K (17%) | 6K-10K (2-3%) |
| Aggressiveness | 보통 | 5-8× 더 aggressive |
| Outcome | All-bench > full 52K | Mistral-7B = Zephyr-7B-beta (~200K SFT 동급) |

### Framework — Selection의 4 sub-dimensions
| Sub-dim | AlpaGasus | DEITA |
|---|---|---|
| What | Quality only | Complexity + Quality |
| How | Threshold | Lexicographic priority |
| Scope | Sample-level | Sample + set-level |
| Scorer | External API | Distilled internal |

→ DEITA는 *4 sub-dim 모두* AlpaGasus를 확장. "차원 추가"는 첫 dim만 본 것.

### [[Q5]] (augmentation × selection) instance
DEITA의 complexity scorer는 Evol-Instruct mutation으로 학습 → **augmentation이 selection signal의 *학습 데이터***. Q5에서 surface한 axis가 DEITA에서 명시적 활용된 instance.

### 한 줄
**"AlpaGasus = 1-axis quality + threshold + external API. DEITA = 3-axis + sample+set-level + lexicographic priority + distilled internal scorer + 5-8× aggressive. 학습자 '차원 추가' 부분 정확이지만 *3번째 axis (diversity)가 set-level*인 게 가장 깊은 차이. Combination도 weighted sum 아닌 *lexicographic priority*."**

## Q11. DEITA diversity — Embedding distance의 *기준 vector*는?

### 답: 기준 vector *없음*. *Selected set 전체*와 *running 비교*

```python
selected = []
sorted_pool = sort_by_complexity_times_quality(pool)

for sample in sorted_pool:
    if not selected:
        selected.append(sample)  # 첫 sample 자동 admit
        continue
    similarities = [cosine_sim(sample, s) for s in selected]
    max_sim = max(similarities)  # 가장 가까운 selected와의 sim
    if max_sim < tau:  # ≈ 0.9
        selected.append(sample)
```

→ Cumulative comparison. Selected set이 커질수록 비교 대상도 늘어남.

### 5-sample 예시
A,B (math, A>B quality) + C,D (code, C>D quality) + E (creative):
- A admit (첫)
- B drop (A와 sim 0.99)
- C admit (A와 sim 0.13)
- D drop (C와 sim 0.99)
- E admit (모두와 sim 낮음)

→ 5 → 3, 3 cluster의 best representative.

### Why max_sim (NOT mean_sim)
mean_sim 쓰면 "대부분 다르지만 1개와 매우 비슷"한 near-dup이 slip in. max_sim = strict: 어느 selected라도 too close면 redundant.

### Why cosine (NOT Euclidean)
Cosine = 각도, magnitude 무관. Sentence length 영향 제거. Semantic similarity는 direction-based.

### Embedder는 *target student 아님*
Sentence-BERT / MPNet 같은 general-purpose semantic embedder. 이유: embedding은 *data의 의미* 측정 → student capability와 무관해야. [[Q6]] signal portability 관점에서 embedding은 *data-intrinsic*.

### Three reference frames (selection 종류 정리)
- AlpaGasus: *absolute 기준* (4.5 fixed value)
- IFD top X%: *relative 기준* (pool 내 percentile)
- DEITA diversity: *running 기준* (이미 선택된 sample들)

→ Sample 외부 → pool 내부 → selection 내부로 이동. Selection-internal frame이 *set-level* selection의 정의.

### 한 줄
**"기준 vector 없음. Candidate은 *previously selected set 전체*와 cosine similarity 비교, max_sim < τ면 admit. Greedy + cumulative. Selected 커질수록 비교 대상 ↑. max_sim (mean 아님) = strict. Cosine (Euclidean 아님) = magnitude 무관 semantic direction. Embedder는 SBERT 같은 별도 general semantic 모델."**

## Q12. DEITA diversity의 compute — Exponential? — *Quadratic*-ish

### Correction
Exponential = 기수가 변수 (e.g., O(2^N)). DEITA는 *두 인자의 곱* O(N × K) — quadratic-ish.

### 구체 numbers
```
N=300K, K=10K, d=768
Total comparisons: N × K = 3×10⁹
Each: d ops = 768
Total: ~2.3 × 10¹² ops
```

비교: IFD with 7B = 28 PFLOPs (~10× 더 큼). **DEITA diversity < IFD scoring**.

### Pool size에 따른 cost
| N | K | Total ops |
|---|---|---|
| 300K | 10K | 3×10⁹ (DEITA original) |
| 1M | 30K | 3×10¹⁰ |
| 10M | 100K | 10¹² |
| 100M | 1M | 10¹⁴ (pretrain scale, prohibitive) |

→ Pool > 10M에서 진짜 issue.

### Practical mitigations
1. **FAISS**: hierarchical index → O(K) → O(log K). 100× speedup.
2. **HNSW / ANN**: graph-based approximate → < 1% accuracy loss, 10-100× speedup
3. **Clustering**: k-means 먼저 → cluster-level diversity. 정확도 손해, 1000× 절감
4. **Subsample**: random subset에서 DEITA. Quick prototype.

DEITA paper 실제: pre-compute all embeddings + FAISS index → single A100에서 몇 시간. Production-grade.

### Selection method cost order (1M pool 기준)
| Method | Total cost |
|---|---|
| AlpaGasus | ~$1000 API + rate limit |
| Superfiltering (125M proxy) | 0.5 PFLOPs |
| **DEITA diversity (FAISS)** | **~10 GFLOPs** |
| IFD (target 7B) | 28 PFLOPs |
| LESS | 100s of PFLOPs |

→ DEITA diversity는 AlpaGasus보다 비싸지만 LESS/Prismatic보다 훨씬 쌈. *Practical sweet spot*.

### [[ch-21#Q11]] framework 확장 — *Cost scaling profile* axis
서로 다른 method가 다른 N-dependence:
- O(N): AlpaGasus, IFD, DEITA scoring
- O(N×K) ≈ quadratic: DEITA diversity, Prismatic eigen part
- O(N×d_θ): LESS (d_θ = θ-dim, huge)

→ Pool size 늘 때 *어떤 method가 dominant cost인가*는 method choice 결정. 새 axis 후보.

### 한 줄
**"Quadratic O(N×K) (NOT exponential). 학습자 직관 정확한 부분: 빠른 증가. 단 IFD with 7B보다 작음. FAISS + ANN으로 O(log K) reducible. DEITA paper는 single A100에서 몇 시간 — production-grade. Pool size별 method cost profile은 새 axis 후보."**

## Q13. IFD 방향 — *반대* (real-time correction)

### 학습자 misconception
- "IFD > 1 means instruction reinforces correct behavior" → **반대**
- "IFD < 0 means data is noise" → **불가능** (IFD는 항상 양수)

### 정확한 방향
```
IFD = PPL_cond(a|q) / PPL_uncond(a)

Instruction이 도움 → q 주면 a predict 쉬움 → PPL_cond 낮음 → ratio < 1 → KEEP
Instruction이 무관 → 두 PPL 비슷 → ratio ≈ 1 → boilerplate, drop
Instruction이 방해 → q 주면 더 어려움 → PPL_cond 높음 → ratio > 1 → DROP
```

→ **작을수록 좋음**. Cherry-LLM keep zone = 0.7-0.95 (just below 1, sweet spot).

### IFD < 0 불가능
Perplexity = exp(cross-entropy) → 항상 양수. 두 양수 ratio → 항상 양수. Boundary는 *0 아닌 1*. 학습자가 "less than 0"이라 한 부분은 "less than 1"과 confusion.

### Confusion source 추측
"Strong instruction = 학습 효과 큼" 직관에서 출발. 사실: strong instruction은 prediction을 *쉽게* 만들어야 함. IFD > 1은 instruction이 *방해함* = mismatched garbage = drop.

### 비유
- IFD < 1 (좋음): "지도 있어서 길 쉬워짐"
- IFD > 1 (나쁨): "지도 틀려서 더 헤맴" → 지도(instruction) drop

### Re-table
| IFD | 의미 | Action |
|---|---|---|
| 0.4 | strong instruction 도움 | ✅ KEEP |
| 0.7-0.95 | Cherry-LLM sweet spot | ✅ KEEP |
| ≈ 1.0 | instruction 무관 boilerplate | ❌ drop |
| 1.5+ | mismatched (Self-Instruct failure) | ❌ drop |

### 한 줄
**"IFD < 1 GOOD (instruction이 prediction 쉽게), IFD > 1 BAD (instruction이 방해, drop). 학습자 방향 정반대. IFD 항상 양수, boundary는 1. Cherry-LLM keep zone = 0.7-0.95."**

## Q14. LESS — Equation 의미 + Verification 방법

### Big picture question
"이 training sample을 조금 넣으면 *내가 신경 쓰는 validation 성능*이 얼마나 변할까?" → *targeted* selection, 다른 method와 달리 *target-aware*.

### Equation 분해
```
dL_val/dε ≈ -η · g_val^T · H^{-1} · g_i
```
- `L_val`: validation loss (예: MMLU 5-shot)
- `ε`: sample weight
- `dL_val/dε`: sample weight 늘리면 val loss 얼마나 변하나
- `g_val`: validation 위 gradient ("어디로 가야 val 좋아짐")
- `g_i`: sample i가 학습되며 만드는 gradient
- `H^{-1}`: loss landscape 곡률 보정

→ 핵심 직관: 두 gradient *방향이 같으면* (cosine > 0) sample이 *내가 가고 싶은 방향*으로 밈 → KEEP.

### Why intractable
H = θ×θ matrix. 7B 모델 → 200 exabytes 저장 불가.

### LESS의 4 contributions
1. **Adam 보정**: SGD 가정 wrong. g_i_adam[k] = g_i[k] / (√v̂[k]+ε). Top 5% subset이 SGD-rank와 30% disagree.
2. **LoRA warmup**: 4% SFT budget으로 gradient 안정화. Cold model gradient는 random.
3. **Random projection (Johnson-Lindenstrauss)**: 7B-dim → 8K-dim. ~900,000× compression, pairwise inner product 보존.
4. **Cosine similarity**: H^{-1} 무시하고 normalize한 cosine만 사용. Empirical shortcut.

### Verification
```
Pool: ~100K (FLAN, Tulu)
Targets: MMLU/BBH/TydiQA 각 5 exemplars
Procedure:
  1. LoRA warmup → gradient datastore (one-time)
  2. Target별 g_val_proj 계산
  3. cosine(g_i_proj, g_val_proj) rank
  4. Top 5% SFT
  5. Eval
Result: 5% LESS-selected > 100% random on all 3 benchmarks
```

→ Less data, better result *on target*.

### Transfer (가장 강한 evidence)
- Llama-2-7B datastore → Mistral-7B target: 작동
- Llama-2-7B datastore → Llama-2-13B target: 작동

→ Datastore one-time cost, 여러 model에 amortize.

### Limits
1. Correctness 못 잡음 (gradient-aligned이지만 wrong sample 통과)
2. Coverage gap (5 exemplars 밖 capability 무관)
3. Diversity 안 보장 (같은 corner 5%만 선택 가능)

### Framework — Selection의 *target-aware* 축
| Method | Target-aware? |
|---|---|
| AlpaGasus, IFD, DEITA | No (target-agnostic) |
| **LESS** | **Yes (capability-targeted)** |

→ [[ch-21#Q11]] 4-axis에 *Target specification axis* (agnostic vs specific) 추가 후보.

### 한 줄
**"LESS = *target-aware* gradient-based selection. Equation 의미: sample gradient가 validation gradient와 align하면 KEEP. H^{-1} intractable → 4 hack (Adam 보정, LoRA warmup, JL projection, cosine). Verification: 5% LESS > 100% random on MMLU/BBH/TydiQA. Datastore가 model family/size 너머 transfer. Limit: correctness/coverage/diversity 안 됨. 새 axis: target specification."**

## Q15. LESS는 dataset-mixture 선택인가? — Mechanism은 sample-level, effect는 induced mixture

### Mechanism — sample-level (학습자 framing 부분 wrong)
각 sample이 개별 gradient → 개별 score → 개별 admit/reject. Dataset weight 계산 *없음*. DoReMi 류 dataset-level mixture와 다른 카테고리.

### Effect — emergent mixture (학습자 직관 맞는 부분)
Top 5%에 source dataset 비율이 *결과적으로* 달라짐:
- Target MMLU → FLAN(reasoning) 과대표현, Wizard(creative) 과소표현
- "LESS가 FLAN을 중요하게 봤다"는 *induced byproduct*

→ Mechanism은 sample-by-sample이지만 결과는 dataset bias처럼 보임.

### 정확한 분류 — Selection 해상도 축
| 해상도 | Method | 단위 |
|---|---|---|
| Pool-level | source 선택 | 사용/안 사용 |
| Dataset-level mixture | DoReMi, DRO | 각 dataset ratio |
| **Sample-level filter** | **LESS**, IFD, DEITA, AlpaGasus | 각 sample score |

### 진짜 통찰 — *Quality* vs *Kind* intent (학습자 직관의 source)
| Method | Intent | Question |
|---|---|---|
| AlpaGasus | Quality | "얼마나 좋은가?" |
| IFD | Informativeness | "learning signal?" |
| DEITA | Q+C+D | "균형 잡혀 있나?" |
| **LESS** | **Kind / Target-match** | "내 capability에 맞나?" |

→ LESS는 *quality* 선택 아니라 *kind/target* 선택. 학습자 직관 "dataset 선택 느낌"의 진짜 source. "AlpaGasus = 좋은 책 / LESS = 수학 시험엔 수학책 / Dataset mixture = 7:3 ratio".

### [[ch-21#Q11]] framework — *Intent axis* 추가 후보
Quality / Diversity / Kind-target-match / Informativeness. Method choice는 intent에 따라 결정.

### 한 줄
**"Mechanism sample-level (dataset-mixture 아님). 단 effect는 induced mixture (top 5%에 source bias emerge). 학습자 직관 source: LESS는 *quality* 아니라 *kind/target-match* intent. 진짜 framework axis: selection intent (quality vs diversity vs kind)."**

## Q16. LESS 자세히 — 6 stages with 직관 (학습자 재요청)

### Stage 1: 무엇을 푸는가
"내가 MMLU를 잘 풀고 싶다. MMLU에 *직접* 도움 되는 sample만 골라줘." → AlpaGasus/IFD/DEITA와 달리 *target-specified*.

### Stage 2: 직관 — 등산 비유
- 정상 = MMLU 좋은 model parameter
- 가이드 = training sample
- 가이드가 끌어당기는 방향 = sample이 만드는 gradient
- 좋은 가이드 = 정상 방향으로 끌어당김
- LESS = "방향이 맞는 가이드만 데려가자"

g_val = "정상이 어디" (validation gradient on 5 MMLU exemplars)
g_i = "가이드 i의 방향" (sample i의 gradient)
Score = cosine_similarity(g_val, g_i)

### Stage 3: Equation 의미
```
dL_val/dε ≈ -η · g_val^T · H^{-1} · g_i
```
음수면 MMLU loss 감소 → KEEP. 양수면 증가 → DROP.

### Stage 4: H^{-1} — 왜 필요/왜 무시
필요: loss landscape의 dim별 민감도 보정 (좁은 골짜기 vs 넓은 평지). 무시 이유: 7B 모델 H는 200 exabytes (인류 storage ~10%) 저장 불가. 해결: Adam optimizer가 *자체적으로 Hessian 대각선 근사* 수행 (√v로 나누기).

### Stage 5: 4 hacks 자세히

**Hack 1 — Adam 보정**: g_i_adam = g_i / (√v + ε). Adam의 √v 나누기 ≈ H^{-1} 대각선 근사. SGD-rank vs Adam-rank의 top 5% 30% disagree, Adam이 empirically 우수.

**Hack 2 — LoRA warmup**: Cold model gradient는 random. LoRA (low-rank adapter)로 4% budget만 학습 → Adam m,v running averages 추정 → gradient 의미 있어짐.

**Hack 3 — Random projection (Johnson-Lindenstrauss)**:
- 1M samples × 7B float = 28 PB 저장 불가
- JL: random Gaussian matrix P (8K × 7B). g_i_proj = P @ g_i_adam
- 7B → 8K, 900,000× compression
- Pairwise inner product 근사 보존 → LESS score 의미 유지
- 1M × 8K = 32 GB (관리 가능) → **gradient datastore**

**Hack 4 — Cosine similarity**: 모든 보정 후 score = normalize한 dot product. Simple final operation.

### Stage 6: Verification
```
Pool: 100K (FLAN + Tulu + WizardLM)
Targets: MMLU/BBH/TydiQA, each 5 exemplars
Procedure: LoRA warmup → datastore → per-target g_val_proj → cosine rank → top 5% → SFT

Result:
  MMLU:   100% random 45.2 / 5% random 42.1 / 5% LESS 47.8
  BBH:    38.4 / 35.7 / 41.2
  TydiQA: 52.1 / 49.3 / 54.8
```
→ 5% LESS > 100% random on all 3 targets.

### Critical transfer evidence
- Llama-2-7B datastore → Mistral-7B target: 작동
- Llama-2-7B datastore → Llama-2-13B target: 작동
→ Datastore one-time cost, family/size transfer. Operational practicality.

### Why works (근본)
Gradient direction = data-intrinsic property (with small model-specific bias). [[Q6]] Superfiltering 통찰 (data property가 model보다 stable)과 같은 family.

### Limits
1. Correctness 안 잡음 (gradient-aligned wrong samples 통과)
2. Coverage gap (5 exemplars 밖 capability 무관)
3. Diversity 안 됨 (같은 corner cluster 가능)
→ DEITA diversity와 *orthogonal*, compose 가능.

### 한 줄
**"등산 비유: 정상(MMLU) 방향으로 끌어당기는 가이드(sample)만 선택. Equation = gradient cosine similarity, H^{-1}은 Adam이 대신, 4% LoRA warmup으로 gradient 안정화, JL projection으로 7B→8K (900,000×). 5% LESS > 100% random on MMLU/BBH/TydiQA. Datastore가 model 너머 transfer. Limit: correctness/coverage/diversity 안 됨, DEITA와 compose."**

## Q17. g_i와 g_val — 각각 정확히 무엇이고 어떻게 계산

### Gradient 기본
parameter θ = (θ_1, ..., θ_N). 7B → N = 7×10⁹.
∇_θ L = (∂L/∂θ_1, ..., ∂L/∂θ_N) — N-dim vector. Parameter와 같은 dim.

### g_i — Training sample i의 gradient
- 정의: `g_i = ∇_θ L_train(sample_i)` — sample 하나로 학습 시 parameter 변화 방향
- 계산: forward(sample) → loss → backward → 각 param별 ∂L/∂θ
- Tiny 2-param example: sample "Hi"→"Hello", forward로 prob 출력, loss = -log(0.6) = 0.51, backward → g_i = (+0.3, -0.5)
- 실제 SGD update: θ - η·g_i (gradient *반대* 방향)
- 7B 모델: g_i는 7B-dim vector = 28GB/sample (왜 JL projection 필요)

### g_val — Validation set의 *평균* gradient
- 정의: `g_val = ∇_θ L_val` where `L_val = (1/M)·Σ L(val_sample_j)`
- 핵심: 여러 sample의 *평균* loss 위 gradient
- Tiny example: 5 MMLU exemplars 각각 forward+backward → 5 gradients 평균
- Code:
```python
val_loss = sum(loss(model(v.input), v.target) for v in val_set) / len(val_set)
val_loss.backward()
g_val = {n: p.grad.clone() for n,p in model.named_parameters()}
```

### 차이 (side by side)
| | g_i | g_val |
|---|---|---|
| 무엇 위에서 | 1 training sample | 5 val exemplars 평균 |
| 의미 | "이 sample 학습하면" | "val 잘 하려면" |
| 언제 | Pool 모든 sample (1M번) | Target별 1번 |
| Dim | N (param 수) | N (param 수) |

→ 둘 다 *같은 type* (N-dim gradient). 다른 점은 *어떤 data 위에서*만.

### Score = cosine
```
g_i = (+0.3, -0.5), g_val = (+0.38, -0.40)
<g_i, g_val> = 0.314
‖g_i‖ = 0.583, ‖g_val‖ = 0.551
cosine = 0.314 / (0.583 × 0.551) = 0.978  ← 같은 방향, KEEP
```
반대 방향이면 cosine < 0 → DROP.

### 한 줄
**"g_i = 1 sample 위 backward로 얻는 N-dim gradient. g_val = M val exemplars 평균 loss 위 backward로 얻는 N-dim gradient. 둘 다 normal training backward와 같은 방법. 차이는 *어떤 data 위에서*만. LESS score = 두 gradient의 cosine."**

## Q18. LESS는 train-and-validate loop인가? — NO, gradient만 측정

### 학습자 misconception
"Group N samples → train → validate → check" — 이건 *Data Shapley* / *leave-one-out influence* 방식. 1M sample엔 infeasible (10K group × 학습 = 수만 시간).

### LESS의 정확한 mechanism — 학습은 *2번만*
```
Phase 1: 4% LoRA warmup (한 번 짧은 학습)
Phase 2: 모든 sample i의 g_i 측정 (학습 X, gradient만)
Phase 3: val_set의 g_val 측정 (학습 X)
Phase 4: cosine score → top 5% select (학습 X)
Phase 5: Selected 위 final SFT (마지막에 한 번)
```

→ 학습 phase 사이의 모든 step은 *gradient measurement only*.

### 핵심 — Gradient는 학습 *없이*도 계산 가능
- "학습한다" = forward + loss + backward + **parameter update**
- "gradient 계산" = forward + loss + backward + **update 생략**

LESS는 마지막 update step을 *안 함*. Backward로 gradient만 read하고 *버림*.

비유:
- 학습 = "이 사람 채용하고 결과 보기"
- LESS = "이 사람 채용한다면 회사가 *어느 방향으로* 갈지 추정 (실제 채용 X)"

### 왜 작동
g_i = "sample_i를 *한 발짝* 학습시키면 parameter 어디로 갈까" (1차 근사).
cosine(g_i, g_val) > 0 → sample이 원하는 방향으로 미는 것 → 실제 학습 안 해도 좋다고 예측.

### Cost 비교
- 학습자 mental model: 10K group × 학습 = 수만 시간 (infeasible)
- LESS: 4% warmup + 1M gradient (forward+backward, update X) = 수십 시간

→ 수만 배 cheaper. 1st-order approximation으로 학습 시뮬레이션 대체.

### Limit of 1st-order approximation
- 진짜 학습은 parameter 변화시킴
- 변한 parameter에서 다음 sample 효과 다름 (sequential)
- LESS는 이 *순간 측정*만, 누적 효과 무시
- 단 empirically 잘 작동

### 한 줄
**"LESS는 train-and-validate loop 아님. Gradient는 학습 *없이*도 backward로 계산 가능 (update만 생략). Mechanism: 4% warmup → 모든 sample의 g_i (학습 X) → g_val (학습 X) → cosine score → top 5% → final SFT. 학습은 2번뿐. '학습 시뮬레이션 없이 학습 결과 예측'이 핵심."**

## Q19. 왜 LoRA warmup 필요? — Cold gradient는 *format-driven*

### Reason 1: Cold gradient = format gradient (content 아님)
Cold base model은 chat format/template 모름. Sample 보여주면 loss의 *대부분*이 template 학습 필요에서 옴. 결과:
- 모든 sample의 gradient가 *"template 학습해라"* 방향
- Per-sample 차이 묻힘
- cosine(g_i, g_val) 모두 비슷 → discrimination 불가

After warmup:
- Template 이미 학습됨
- Gradient가 *content*에 의해 결정됨
- Per-sample이 *방향이 다름* → cosine ranking 의미 있음

### Reason 2: Adam의 running variance v 필요
LESS는 g_i_adam = g_i / (√v + ε) 사용. Cold state에서 v=0 → division undefined. Warmup이 Adam의 v running average를 populate.

### 왜 *LoRA*, full SFT 아니고?
- No warmup: cold gradient + Adam v=0 → 작동 안 함
- Full SFT warmup: circular (SFT 위한 selection을 SFT로?) + 비쌈
- **LoRA warmup**: cheap (~100M params vs 7B), 4% budget 충분, base + adapter로 가볍게 학습된 상태 만듦

### Warmup의 *진짜 역할* — Format gradient noise 제거
```
Cold gradient = format gradient + content gradient + noise
                ↑                ↑
                지배적            묻혀 있음

Warmup → format 학습 끝 → format gradient 제거
       → content + noise만 남음
       → content signal 드러남
```

### 비유
- Cold model = 한국어 모르는 학생: 수학/역사 문제에 모두 "한국어 배워야지"
- Warmed model = 한국어 안 학생: 수학은 수학적, 역사는 역사적 반응
- Selection은 후자에서만 의미

### 한 줄
**"Cold gradient는 content 아닌 format/template 학습 방향 → 모든 sample 비슷 → discrimination 불가. 또 Adam v=0이라 보정 불가. LoRA warmup 4%로 template gradient 제거 + Adam v 안정화. Full SFT 안 쓰는 이유: circular + 비쌈. LoRA는 cheap + 충분."**

## Q20. Adam과 JL projection — LESS에서 어떻게 적용

### Part A — Adam

**SGD vs Adam**:
- SGD: θ - η·g (모든 dim 같은 lr)
- Adam: θ - η·m̂/√v̂ (parameter별 adaptive)

**Adam 5 steps**:
1. g_t = gradient
2. m = β_1·m + (1-β_1)·g_t (momentum)
3. v = β_2·v + (1-β_2)·g_t² (variance, element-wise square)
4. Bias correct: m̂ = m/(1-β_1^t), v̂ = v/(1-β_2^t)
5. Update: θ - η·m̂/(√v̂ + ε)

각 parameter dim k가 *자기 history*로 scale: gradient 자주 큰 dim → v[k] 큼 → 1/√v[k]로 작아짐 (안정). 자주 작은 dim → 1/√v[k] 커짐 (충분히 학습).

**Hessian 보정과 관계**:
Fisher information: E[g_k²] ≈ H[k,k] (loss quadratic 근처). Adam의 v[k] ≈ E[g_k²] ≈ H[k,k] → 1/√v[k] ≈ 1/√H[k,k] = 대각선 Hessian inverse 근사.

**LESS 적용**:
```python
# Phase 1: LoRA warmup 동안 v 누적
# Phase 2: g_i_adam = g_i / (sqrt(v) + ε)
# Phase 3: g_val_adam도 같은 방식
# Score: cosine(g_i_adam, g_val_adam)
```
→ warmup의 부산물 v를 *Hessian 보정용 재사용*. 추가 계산 없이 보정. Ablation: Adam 없이 vs 있으면 top 5% 30% disagree.

### Part B — Johnson-Lindenstrauss projection

**Statement**: Random Gaussian matrix P (entry ~ N(0, 1/d))로 high-dim → low-dim projection 시 pairwise inner product 근사 보존.

```
g ∈ R^N (N = 7×10⁹)
P ∈ R^{d×N} (d = 8000), entries N(0, 1/d)
g_proj = P·g ∈ R^d

<P·a, P·b> ≈ <a, b>  (expectation 보존, concentration tight)
```

**d 결정**: d = O(log N / ε²). 1M sample, ε=0.1 → d ≈ 600. LESS는 안전하게 8000.

**압축비**: 7B dim float32 = 28GB/sample → 8K dim float32 = 32KB/sample = **~900,000× reduction**. 1M sample = 28PB → 32GB.

**LESS 적용**:
```python
P = random_gaussian(8000, 7B)  # 한 번 만들고 fix

for sample_i:
    g_i_proj = P @ g_i_adam     # JL projection
    g_i_proj = normalize(g_i_proj)
    store(g_i_proj)

g_val_proj = P @ g_val_adam     # *같은* P!
g_val_proj = normalize(g_val_proj)

score = dot(g_i_proj, g_val_proj)  # 8K cosine ≈ 7B cosine
```

**중요**: g_i와 g_val *같은 P 공유*. 다른 P면 inner product 보존 안 됨.

### 전체 LESS pipeline (Adam + JL 합쳐)
```
1. P = JL matrix (fixed)
2. warmed_model, v = LoRA_warmup
3. for sample_i: g_i → g_i_adam (Adam) → g_i_proj (JL) → datastore
4. per target: g_val → g_val_adam → g_val_proj
5. score = dot, top 5% select
6. final SFT
```

→ Adam = parameter dim 보정 (Hessian 근사). JL = vector size 보정 (저장 압축). *완전히 다른 차원*의 문제.

### 한 줄
**"Adam = parameter별 adaptive lr (1/√v) = 대각선 Hessian inverse 근사. LESS에서 warmup v를 재사용해 g_i/√v로 보정. JL = random Gaussian projection이 pairwise inner product 보존, 7B→8K 압축 (~900,000×). 같은 P로 g_i와 g_val 모두 project. 두 hack은 parameter dim과 vector size의 다른 차원에서 cost 절감."**

## Q21. Prismatic Synthesis — Gradient diversity for OOD generalization

### 정확한 question (학습자 framing refinement)
"General performance"보다 specific: **OOD (out-of-distribution) generalization**. "Pool이 gradient manifold를 커버하는가."

| Method | Question |
|---|---|
| LESS | Target에 align하는가? (narrow) |
| DEITA | Embedding 기준 다양한가? (general) |
| **Prismatic** | **Gradient manifold 커버하는가? (OOD)** |

### G-Vendi metric — 직관
"Effective number of distinct gradient directions". 1000 sample의 gradient를 화살표로:
- 모두 같은 방향 → G-Vendi = 1 (effective 1 sample)
- 모두 orthogonal → G-Vendi = N (effective N sample)

### G-Vendi 계산 6 steps
1. Per-sample normalized gradient: g_i / ‖g_i‖
2. JL projection to 8K (LESS의 hack 재사용)
3. Gram matrix: K[i,j] = <g_i, g_j> (cosine)
4. Density matrix: ρ = K / trace(K), trace(ρ) = 1
5. Eigenvalues: {λ_k}, 합 = 1, [0,1]
6. **Von-Neumann entropy + exponentiate**: G-Vendi = exp(-Σ λ_k log λ_k)

### Tiny example (3 samples)
```
g_1 = (1,0,0), g_2 = (0,1,0), g_3 = (1,0,0)  (g_3 = g_1)
K = [[1,0,1],[0,1,0],[1,0,1]], trace = 3
ρ = K/3
Eigenvalues: 2/3, 1/3, 0
H(ρ) = 0.636
G-Vendi = exp(0.636) = 1.89  (3 sample이지만 effective 1.89)
```

### *왜 gradient kernel, 아닌 embedding kernel*
**Case 1**: Embedding 가깝, gradient 멈 — "What is 2+2?" vs "Solve 17×23" (둘 다 math지만 다른 skill, 다른 gradient)
**Case 2**: Embedding 멈, gradient 가깝 — "Translate to French" vs "Translate to Spanish" (다른 언어지만 같은 skill, 같은 gradient)

Chapter line 154: *"Embedding geometry is about surface form; gradient geometry is about what the optimizer learns."*

**Empirical (300+ runs)**: G-Vendi vs OOD accuracy ρ ≈ 0.9. Embedding-Vendi (14× larger encoder), Skill-Set Entropy (GPT-4 + Qwen-72B labelers) 모두 훨씬 약함.

### Prismatic pipeline
1. Generate large candidate pool (10x target)
2. Greedy max-entropy selection: 각 candidate의 G-Vendi marginal contribution 측정
3. Low-density gradient region 우선 선택
4. Verify (answer-check for math, label-consistency for NLI)
5. Train

### 7B beats 671B claim (chapter line 144)
- Brute force: 671B teacher generation → 7B distill
- Prismatic: 7B teacher → 1M candidates → G-Vendi 기준 10K curate → 7B distill
- **Prismatic 우수 on OOD reasoning**

Reason (mechanical): 671B teacher의 자연 generation은 *high-probability mode*에 집중 → gradient manifold의 *vanishing fraction*만 커버. Prismatic은 *low-density region*으로 적극 sampling.

→ Chapter 강한 claim: *"Diversity, not generator scale, is the binding constraint."*

### Limits
1. Quality 안 잡음 (gradient-diverse but factually wrong sample 통과)
2. Coverage gap of *target* (pool 밖 capability는 추가 못 함, 단 generation 단계로 부분 mitigation)
3. **Cost**: gradient datastore + N×N eigen decomposition O(N³). N=10K 가능, N=100K prohibitive.

### [[Q15]] selection intent table 완성
| Method | Intent | Mechanism |
|---|---|---|
| AlpaGasus | Quality | LLM-judge threshold |
| IFD | Informativeness | Self-perplexity |
| DEITA | Q+C+D (surface) | Embedding distance |
| LESS | Target-match | Gradient alignment |
| **Prismatic** | **OOD generalization** | **Gradient entropy** |

→ Prismatic만 OOD를 *explicit goal*로. 다른 method는 *in-distribution* 가정.

[[Q12]] cost scaling profile axis: Prismatic O(N³) 가장 비쌈. 가장 강한 method지만 가장 expensive.

### 한 줄
**"Prismatic = 'pool이 gradient manifold 커버하는가' = OOD generalization. G-Vendi: gradient → Gram matrix → density matrix ρ → von-Neumann entropy → exp(H(ρ)) = effective distinct directions. Embedding과 차이: gradient diversity가 OOD 예측 (ρ=0.9), embedding diversity는 못 함. 7B Prismatic > 671B brute distill — *diversity가 scale보다 binding*. Limit: quality 안 잡음, O(N³)."**

## Q22. Prismatic도 LoRA warmup 필요? — NO, *frozen instruction-tuned* model 사용

### Chapter line 132
"For a candidate pool... and a **frozen instruction-tuned proxy LM**." → 핵심: frozen + 이미 instruction-tuned.

### LESS vs Prismatic — model state
| | LESS | Prismatic |
|---|---|---|
| Starting model | Base (pretrain) | Instruction-tuned (frozen) |
| Warmup | LoRA 4% 필요 | *없음* |

### Warmup 안 해도 되는 2 reasons

**Reason 1 — Template gradient 문제 없음**
- LESS cold model: format/template gradient 지배 → 모든 sample 비슷 → 1차 학습 (warmup) 필요
- Prismatic frozen instruction-tuned: template 이미 학습 → 처음부터 content-driven gradient → discrimination 됨

**Reason 2 — Adam 보정 안 씀**
- LESS: g_i_adam = g_i / (sqrt(v)+ε) → v 필요 → warmup으로 v 누적
- Prismatic: g_i_normalized = g_i / ‖g_i‖ → L2 norm만 → v 안 필요 → warmup 안 필요

Side effect: Prismatic은 Hessian 보정 안 함. 단 *방향 distribution*이 관심사라 dim별 sensitivity 덜 중요.

### Design diff 정리
| | LESS | Prismatic |
|---|---|---|
| Gradient processing | Adam-adjusted (g/√v) | L2 normalized (g/‖g‖) |
| Adam v 사용 | Yes | No |
| JL projection | Yes | Yes ← *공유* |
| 측정 | Pairwise cosine | Gram matrix → eigenvalues |

→ JL 공유, 나머지 완전 다른 design.

### Intent 차이가 design 결정
- LESS: "어떤 sample로 SFT돼야 target에 도움?" → SFT 과정 modeling → base부터, warmup으로 cold 문제 해결
- Prismatic: "Pool이 *학습된 model 위에서* 어떤 다양성 제공?" → frozen instruction-tuned proxy가 typical SFT start와 가깝, warmup skip

### Cost implication
Prismatic entry cost가 LESS보다 *낮음* (warmup 4% 절약, pre-trained model 재활용). 단 G-Vendi O(N³)이 *훨씬* 큼. 전체적으로 Prismatic이 더 비쌈.

### 한 줄
**"Prismatic warmup 안 함. 2 reasons: (1) frozen instruction-tuned model이라 format gradient 문제 없음, (2) Adam 보정 대신 L2 normalize만 사용해 v 불필요. LESS는 warmup으로 두 문제 동시 해결, Prismatic은 model 선택 + normalize 방식으로 회피. JL projection은 공유."**

## Q23. LESS vs Prismatic — Big O 비교

### Notation
N = pool size, K = selected size, P = parameters, d = JL projection dim

### LESS Big O
| Phase | Cost |
|---|---|
| LoRA warmup (one-time) | O(0.04 · N · P) |
| Gradient datastore (one-time) | **O(N · d · P)** ← dominant |
| Per-target query | O(N · d) |

→ Datastore one-time + queries amortizable. *여러 target에 재사용 가능*.

### Prismatic Big O
| Phase | Cost |
|---|---|
| Warmup | 0 (없음) |
| Gradient datastore | O(N · d · P) (LESS와 같음) |
| Gram matrix K | O(N² · d) |
| Eigen decomposition | O(N³) |
| Greedy selection (K steps) | O(N · K³) |

→ *Per-pool one-shot*. Amortize 불가 (target-specific 아니라 *pool entropy* 측정).

### Concrete (N=1M, K=10K, P=7B, d=8K)
```
N · d · P  = 5.6 × 10¹⁹   ← datastore (둘 다)
N² · d     = 8 × 10¹⁵     ← Gram (Prismatic)
N³         = 10¹⁸          ← eigen (Prismatic)
N · K³     = 10¹⁸          ← greedy (Prismatic)
```
→ N=1M에선 datastore dominant. Prismatic extra 비교적 작음.

### Concrete (N=100M, pretrain-scale)
```
N · d · P = 5.6 × 10²¹
N³        = 10²⁴   ← *datastore 능가*
```
→ Pretrain scale에선 Prismatic의 O(N³)가 dominant. LESS는 여전히 datastore.

### Amortization effect (M targets)
- LESS: O(N·d·P) + M·O(N·d) ≈ datastore only
- Prismatic: M·O(N·d·P) + M·O(N³)
- M=10 targets → LESS가 약 10× cheaper

→ **LESS의 진짜 win = datastore 재사용성**. Prismatic은 single-shot.

### Cost scaling profile axis ([[Q12]] 확장)
| Method | Big O | Amortizable? |
|---|---|---|
| AlpaGasus | O(N · API) | No |
| IFD | O(N · d_model) | No |
| Superfiltering | O(N · d_proxy) | Yes (proxy 재사용) |
| DEITA scoring | O(N · d_model) | Mid |
| DEITA diversity | O(N · K · d_embed) | No |
| **LESS** | **O(N · d · P) + O(N·d)/query** | **Yes (datastore)** |
| **Prismatic** | **O(N · d · P) + O(N³)** | **No** |

### 한 줄
**"LESS: O(N·d·P) datastore one-time + O(N·d) per query, *amortizable across targets*. Prismatic: O(N·d·P) + O(N²·d) Gram + O(N³) eigen + O(N·K³) greedy, *per-pool one-shot*. N=1M scale에선 둘 다 datastore dominant. N=100M에선 Prismatic eigen이 dominant. M targets면 LESS가 M× cheaper."**

## Q24. PPL과 probability의 *역* 관계 — IFD direction 재확인

### 학습자 confusion
"IFD < 1 ⟹ PPL_cond < PPL_uncond ⟹ '(a|q) 상황의 probability가 더 *낮다*'" → 마지막 step에서 PPL과 probability를 동일시. *역*임.

### 핵심 — PPL = 1/P
```
Cross-entropy: -log P(token)
Perplexity: exp(cross-entropy) = exp(-log P) = 1/P
```

→ Probability ↑ → PPL ↓ (반비례).
- PPL 낮음 = P 높음 = model 자신 있게 맞춤
- PPL 높음 = P 낮음 = model 헷갈림

### 비유 — "몇 개 option 중 헷갈리는가"
- P=0.5 (50/50 동전) → PPL=2 (2 option)
- P=0.1 (10 option uniform) → PPL=10
- P=0.001 → PPL=1000

### 3 scenario로 IFD 방향 확인
| Scenario | P(a) | P(a\|q) | PPL_uncond | PPL_cond | IFD | 해석 |
|---|---|---|---|---|---|---|
| q 도움 (Paris) | 0.001 | 0.5 | 1000 | 2 | 0.002 | < 1 ✓ KEEP |
| q 무관 (boilerplate) | 0.1 | 0.1 | 10 | 10 | 1.0 | ≈ 1 drop |
| q 방해 (cake/calculus) | 0.01 | 0.0001 | 100 | 10000 | 100 | > 1 ✗ DROP |

### 정확한 추론 chain
```
IFD < 1
⟹ PPL_cond < PPL_uncond
⟹ P_cond > P_uncond           ← 역수니까 부등호 *뒤집힘*
⟹ q가 a 예측 *쉽게* 만듦
⟹ 학습 signal 있음 (좋음)
```

### 한 줄
**"PPL = 1/P (반비례). 학습자 confusion은 마지막 step에서 PPL과 P를 동일시. PPL_cond 작음 → P_cond 큼 → q 도움 → IFD < 1 좋음. 비유: PPL = '몇 option 중 헷갈리는가', 낮을수록 model 자신감."**
