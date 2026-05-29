<!-- chapter: ch-28 — Modality: Long-Context Synthesis
     companion to: [[read]]
     append-only across cycles
-->

# Ch-28 Q&A — Long-Context Synthesis

Reading questions captured during the Read phase.

---

## Q1 — NIAH가 뭐야? Needle In A Haystack?

**Yes.** 2023년 11월 Kamradt blog post 출처.

### Setup
```
1. Long document에 한 문장 (needle) 숨김
2. 그 문장의 내용을 묻는 query
3. Model 답에 needle 내용이 exact-substring으로 나오는지 check
```

**Needle 예시**: "The best thing to do in San Francisco is eat a sandwich at Dolores Park on a sunny day."
**Haystack**: Paul Graham essays padded to target length (128K, 1M 등).
**Query**: "What is the best thing to do in San Francisco?"

### Signature visualization
2D heatmap — X축 context length, Y축 needle depth (%), color는 정답률. "Lost in the middle" 현상이 immediately visible.

### 왜 standard 됐는가
- Visually legible
- Cost zero (teacher 안 필요)
- Marketing easy

→ 128K/200K/1M release 모두 NIAH heatmap ship.

### NIAH의 한계 (가장 중요한 부분)

✗ One needle / one depth / one query / exact substring / no distractor / no reasoning
✓ 실제 long-context는 multi-needle + distractor + aggregation + reasoning 필요

→ NIAH @ 128K 99% 통과한 model이 multi-hop @ 32K에서 collapse 가능.

### Claim vs effective gap (구체적 숫자)

| Model | Claimed | NIAH | RULER effective |
|---|---|---|---|
| Llama-3.1-70B | 128K | ~99% | **~64K** (1/2) |
| Qwen-2.5-14B-1M | 1M | ~100% | ~85% |

**"Claimed context" 옆에 *어떤 metric*인지 명시 없으면 underspecified**.

### RULER가 NIAH를 어떻게 확장했나

13 task family. NIAH는 그중 1개 (S-NIAH = single-needle, 가장 쉬움).
- MK-NIAH: distractor 추가
- MV-NIAH: recall completeness
- MQ-NIAH: parallel retrieval
- VT: coreference reasoning
- CWE/FWE: aggregation
- QA: realistic reasoning

NIAH 통과 = *13개 중 1개* 통과. 나머지 12개가 진짜 capability.

### Ch-26/27 framework 연결

- **[[Q4]] distribution overfit** (ch-26): NIAH로 학습 → NIAH distribution에 over-fit, real use에서 깨짐. Eval distribution overfit의 가장 명확한 instance.
- **[[Q7]] eval as data spec** (ch-26): NIAH가 marketing metric되면서 multi-needle SFT 추가 필요해짐. Eval taxonomy → training data taxonomy.
- **[[Q11]] verifier hierarchy** (ch-27): NIAH (self, gameable) < RULER (teacher, synthetic) < BABILong (world, real PG19 prose + symbolic reasoning).

### 한 줄

**NIAH = needle in haystack 검색 test. 단순/싸/시각적이라 de facto metric됐지만 1 needle/1 depth/1 query/exact match라 *실제 long-context capability와 2배 gap*. RULER + BABILong이 진짜 measurement. "128K claim"은 어떤 metric 명시 안 하면 의미 없음.**

---

## Q2 — "Multiple needles / Various depths / Distractor / Aggregation / Reasoning" 각각 구체적으로?

5개 axis가 *서로 독립적*. NIAH는 모든 axis에서 minimum. 구체 예시:

### Multiple needles
```
NIAH: 1 needle ("The best movie is X.") → 1 query → 1 answer
Real: N needles (best drama A, best comedy B, best action C)
      → 다양한 query (각 장르별, list 모두, 역방향 lookup)
```
NIAH-trained model이 "한 query = 한 retrieval" 패턴에 over-fit.

### Various depths (lost in middle)
| Depth | Llama-3.1-70B @ 128K |
|---|---|
| 5% (front) | ~98% |
| 50% (middle) | **~60%** ← lost in middle |
| 95% (end) | ~95% |

Middle position에서 sharp drop. NIAH average만 report하면 패턴 숨겨짐.

### Distractor resistance
```
Document에:
  Page 10: "best thing in NY is Central Park."
  Page 40: "best thing in LA is the beach."
  Page 60: "best thing in SF is sandwich..."  ← real needle
  Page 95: "worst thing in SF is the fog."    ← partial distractor

Query: "best thing in SF?"
```
4개 "best thing in X" pattern + SF 두 곳 등장 + partial match 위험. NIAH는 *clean haystack*이라서 disambiguation 학습 안 됨.

### Aggregation across spans
```
Page 5:  "Alice is 25."
Page 30: "Alice's brother is 3 years older."
Page 80: "Bob is married to Alice's brother."
Page 90: "Bob's age = Alice's brother's age."

Query: "How old is Bob?"
```
필요: Alice=25 → brother=28 → Bob=28. **3 retrieval + arithmetic + identity matching**. 각 retrieval 90%면 3 retrieval = 0.9³ = 73% (multiplicative failure).

### Reasoning over retrieved items
```
Page 10: "rank > 5 → bonus"
Page 50: "Bob rank = 7"
Page 80: "bonus = $5000"

Query: "Does Bob get bonus, how much?"
```
필요: rule retrieve + fact retrieve + *conditional reasoning* (7 > 5) + value retrieve. NIAH-trained는 step 1만 잘함, conditional reasoning에서 literal copy로 fail.

### "NIAH @ 128K 99% but multi-hop @ 32K collapse" 구체 예시

```
Task A (NIAH @ 128K):  needle "magic word is pineapple" → "pineapple" → 99%
Task B (multi-hop @ 32K):
  Page 5:  X = 7
  Page 15: Y = X + 3
  Page 25: Z = Y × 2
  Query: "What is Z?"  → 답 20
  
같은 model: ~50% accuracy
```

→ **더 짧은 context (32K)에서 더 어려운 task로 깨짐**. Context "길이"가 capability 정의 X — *task complexity*가 더 중요.

### 한 줄

**5 axis (multiple needles / various depths / distractor / aggregation / reasoning) 모두 독립적. NIAH는 각 axis의 minimum. NIAH 99% pass는 짧은 context에서도 multi-hop으로 collapse 가능 — "긴 context 잘 함" ≠ "복잡한 task 잘 함".**

---

## Q3 — "RoPE base θ에 묶인 frequency spectrum, rescale 안 하면 aliasing" + 4-component stack이 모델 구분?

핵심 두 부분: **(A) Position encoding의 기술 문제**, **(B) 그걸 풀려고 4-components stack 필요**.

### (A) RoPE 기본

**RoPE = token을 position에 따라 *회전*시키는 position encoding.**

각 token vector를 64개 2D 평면으로 쪼개서 *각 평면이 다른 속도로 회전*:
- Dim pair 1 (i=0): 매우 빠른 회전 (짧은 거리 표현)
- Dim pair 64 (i=63): 매우 느린 회전 (긴 거리 표현)

시계의 초/분/시침 비유. 회전 속도 공식:
```
θ_i = θ_base^(-2i/d)
```

### `base θ`의 진짜 의미

**가장 느린 회전 하나의 *각속도 base***.
- Llama-2: θ_base = 10K → position 10K에서 가장 느린 dim이 1 radian 회전
- Llama-3: θ_base = 500K → position 500K에서 1 radian
- Fu 2024: θ_base = 200M → 더 극단

`θ_base`가 *암묵적으로* "model이 표현 가능한 position 범위" 정의.

### Frequency aliasing

신호처리의 aliasing 개념과 동일. RoPE에서:
```
가장 느린 wavelength = 2π * θ_base
Llama-2 (θ=10K): wavelength ~62K positions

Position 5K:  180°
Position 67K: 540° = 180° (한 바퀴 돌아 같은 각도!)
→ Model이 두 position 구별 못 함
```

**Training range 밖으로 가면 동일 encoding이 두 다른 position에 발생 → aliasing.**

### Rescale fix

θ_base를 *키움* → wavelength 늘림 → aliasing point 더 멀리.
```
θ=10K  → wavelength 62K
θ=500K → wavelength 3.1M
θ=200M → wavelength 1.25B
```

### (B) 4-components stack

| Component | 무엇 | 없으면 |
|---|---|---|
| 1. Position fix | RoPE θ rescale | 수학적 aliasing |
| 2. Long-doc CPT | 새 θ로 coherent long document에 추가 pretrain | 새 frequency에 익숙 X |
| 3. Synthetic long-SFT | Long context에서 instruction-answer 패턴 학습 | Instruction following X |
| 4. Synthetic long-eval | NIAH/RULER/BABILong 측정 | 작동 여부 모름 |

각 component *독립적이고 필수*.

### Frontier model들의 design choice 차이

| | Llama-3-128K | ProLong-512K | Qwen-2.5-1M |
|---|---|---|---|
| Position | θ=500K | θ=128M | DCA inference |
| CPT | 800B 6-stage | **20B + coherence filter** | 200B 3-stage |
| SFT | Self-distill from 405B | LongAlign + multi-needle | Qwen-Max synthesized |
| Eval | RULER | HELMET | NIAH+RULER (15-pt gap) |

같은 stack인데 각 axis별 trade-off가 다름 → **다른 frontier model**.

### Ch-25/27 framework 연결

- **[[Q15]] per-objective anchor density**: 4 component가 *5-9 orders of magnitude 다른 data density* (position fix 1줄 ~ CPT 800B token). Q15의 가장 극단적 instance.
- **[[Q11]] verifier hierarchy**: position fix (no verifier) → CPT (self) → SFT (teacher) → eval (external). Hierarchy 전체 spectrum.
- **[[Q9]] anchor type**: 각 component가 다른 anchor type (positional / coherence / instruction / capability).

### 한 줄

**RoPE는 token을 position에 따라 회전. base θ가 *표현 가능한 position 범위 base*. Training range 밖 = aliasing (두 position이 같은 회전 각도). Fix는 θ rescale. 그러나 rescale만으론 부족 — 익숙해지는 CPT + instruction SFT + 작동 확인 eval이 필요. 4-components stack의 각 axis에서 *다른 design choice*가 frontier model들을 구분 (Llama-3: massive compute / ProLong: clever curation / Qwen: inference trick).**

---

## Q4 — Staged context extension (8K → 16K → 32K)에서 base_θ도 함께 바뀌나, 아니면 처음부터 fixed?

**둘 다 가능. Recipe choice.**

### Approach 1 — Staged θ rescale (Llama-3)

매 context 확장 stage마다 θ도 함께 조정:

| Stage | Context | RoPE θ |
|---|---|---|
| A | 8K → 16K | ~80K (intermediate) |
| B | 16K → 32K | ~160K |
| C | 32K → 64K | ~320K |
| D | 64K → 128K | **500K** (final) |

각 stage에서 *세 axis 동시 조정*: context length, θ, data mix (short:long ratio).

### Approach 2 — Upfront target θ (Fu 2024, ProLong)

처음에 *한 번* target θ로 swap, 그 후 fixed:
```
Step 0: θ를 10K → 200M으로 한 번에 rescale
Step 1: 80K context로 5B tokens 학습 (θ 200M fixed)
```

ProLong: Llama-3의 500K를 start로 받아서 500K → 128M으로 다시 한 번 upfront swap, 512K context로 20B CPT.

### NTK formula (target θ 계산 base)

```
θ_new ≈ θ_orig × (target_context / orig_context)
```

예: Llama-2 → 128K target = θ_new ≈ 10K × 32 ≈ 320K. Llama-3는 500K로 round up.

실제 paper는 formula minimum보다 *더 큰 θ* 사용 (Fu 200M = formula의 1000×). Aggressive = long context 안전, conservative = short context 정확.

### Trade-offs

| | Staged | Upfront |
|---|---|---|
| θ 변화 | 매 stage | 한 번 |
| Hyperparameter 수 | 많음 | 적음 |
| Total compute | 많이 필요 | 적게 가능 |
| Context jump | 큰 jump OK (16×+) | 중간 jump (10-20×) |
| 안정성 | 높음 (gradual) | 낮음 (sharp) |
| 구현 복잡도 | 높음 | 낮음 |
| 사용 lab | 큰 lab (Meta) | 작은 lab (research) |

### Mental model

**Staged**: 매 stage가 이전 위에 build. Catastrophic shift 없음. 점진적 학습.
**Upfront**: θ shock-then-train. 처음 적응 후 한 stage로 끝.

### 한 줄

**둘 다 valid. Llama-3 = staged (큰 jump + 큰 compute), Fu/ProLong = upfront (smaller jump + smaller compute). NTK formula `θ_new ≈ θ_orig × ratio`가 minimum, 실제는 더 aggressive. Staged stable but expensive, upfront simple but sharp.**

---

## Q5 — RULER가 뭐야?

**NVIDIA 2024의 long-context benchmark. NIAH의 single-axis 한계 해소.**

### 13 task family

**NIAH 계열 (4)**:
- S-NIAH: original (baseline retrieval)
- MK-NIAH: N keys, query 1 → distractor resistance
- MV-NIAH: 1 key, k values, return all → recall completeness
- MQ-NIAH: multiple queries → parallel retrieval

**Reasoning 계열 (4)**:
- VT (Variable Tracing): `X2=X1, X3=X2, ..., X20=?` → coreference chain
- CWE (Common Word Extraction): 자주 단어 추출 → aggregation
- FWE (Frequent Word Extraction): Zeta 분포 top-K → aggregation tail
- QA: SQuAD/HotpotQA + distractor → realistic reasoning

### 핵심 method (3)

1. **Length × complexity 독립 variation** — model이 length 때문에 깨지는지 complexity 때문에 깨지는지 분리 측정
2. **500 ex/task/length** — statistical significance
3. **Explicit answer prefix** — 형식 변동 제거

### Effective context size (가장 중요한 contribution)

```
Baseline: Llama-2-7B @ 4K RULER 점수 = 85.6
Effective context = 측정 model의 점수가 85.6 이상 유지되는 *가장 긴* length
```

예: Llama-3.1-70B
- @ 4K: 97, @ 8K: 95, @ 16K: 91, @ 32K: 88, @ 64K: 86, @ 128K: **74 (below)**
- Effective = **64K** (claimed 128K의 1/2)

### Headline gap

| Model | Claimed | NIAH | RULER effective |
|---|---|---|---|
| Llama-3.1-70B | 128K | 99% | **64K** |
| Llama-3.1-405B | 128K | 99% | ~96K |
| Qwen-2.5-14B-1M | 1M | 100% | ~500K |

**모든 model의 effective < claimed**. Gap이 작을수록 좋은 model.

### 한계

- Still synthetic (real long-context use case와 align 부분적)
- Haystack = generic prose
- Static templates → eval-as-data-spec ([[Q7]]) risk
- Reasoning depth 얕음 (symbolic만) → BABILong 보완 필요

### RULER vs BABILong 보완

| | RULER | BABILong |
|---|---|---|
| Structure | Synthetic | bAbI in real PG19 prose |
| Length | 4K-128K | 0K-50M+ |
| Reasoning | Coreference, aggregation | Symbolic 20 bAbI tasks |

→ RULER = synthetic retrieval/aggregation, BABILong = symbolic reasoning in real prose. 둘 다 필요.

### Production usage

1. RULER로 effective context 측정
2. Effective를 *실제 deploy 한계*로 신뢰 (광고된 length 무시)
3. 본인 domain spot-check 추가
4. Model new version마다 재측정

### Ch-26/27 framework 연결

- **[[Q7]] eval as data spec**: 13 task가 training taxonomy 영향 (LongAlign이 RULER match)
- **[[Q11]] verifier hierarchy**: NIAH (self) < RULER (teacher synthetic) < BABILong (world real prose)
- **[[Q15]] per-objective density**: 13 task가 *서로 다른 anchor density* — S-NIAH (1 needle) → VT (20-hop chain)

### 한 줄

**RULER = NVIDIA의 13-task long-context benchmark. NIAH의 single-axis 한계 해소. *Effective context size*가 핵심 metric — Llama-2-7B@4K baseline (85.6) 이상 유지되는 가장 긴 length. "Claimed vs effective" gap 노출 (Llama-3.1-70B: 128K/64K). Anti-saturation design. BABILong으로 보완 (symbolic reasoning in real prose). Deployment 한계는 RULER effective 신뢰.**

---

## Q6 ★ — RULER는 QA-shaped. Super-long conversation 성능은 어떻게 측정? (학습자 chapter blind-spot critique)

**Critique 정확. Chapter에 *100+ turn pure conversation* eval standard *없음*. 진짜 open problem.**

### RULER의 본질적 한계

모든 13 task = `{static document + query → answer}`. Document QA pattern. *Conversation*과 형태 다름.

Conversation = `{user1, agent1, user2, agent2, ...}`:
- Document 같은 fixed substrate 없음 (*대화 자체가 evolving substrate*)
- Each turn affects later turns
- Score는 *cross-turn property*에 의해 결정
- "정답"이 덜 명확

### Conversation-specific 측정해야 할 8 axis (모두 RULER에 없음)

1. **State maintenance**: turn 5 fact가 turn 200에서 기억?
2. **Persona consistency**: turn 1 persona가 turn 100에서 유지?
3. **Preference accumulation**: user preference 누적?
4. **Cross-turn reference**: "전에 말한 그것" 해결?
5. **Commitment tracking**: agent가 약속 지킴?
6. **Topic drift recovery**: 원래 주제 복귀?
7. **Contradiction detection**: 자기 말 모순?
8. **Style/register stability**: tone 유지?

### 현재 chapter의 *부분적* 접근

| Paper | 무엇 | 한계 |
|---|---|---|
| LongChat | ShareGPT 18K real long conv 수집 | Training data only, eval standard 아님 |
| LongMIT | Synthesized 5-10 turn | *Short* by conversation 기준 |
| LongBench-Chat | Multi-turn LongBench | Task-grounded, not pure conv |
| τ-bench | 4-12 turn task | Task completion, not pure conv |

→ 100+ turn pure conversation standard benchmark *없음*.

### 왜 어려운가 (fundamental problem)

1. **"정답" 부재**: gold conversation 정의 불가능, multiple valid responses
2. **Subjectivity**: "consistent", "natural"이 주관적
3. **Compositional explosion**: 100-turn = 4,950 turn pair 검사 (O(N²))
4. **Real data scarcity**: 100+ turn real conversation 공개 데이터에 없음 (privacy + 대부분 5-20 turn에서 끝)

### Missing benchmark category

| | Static document | Conversation |
|---|---|---|
| Short | MMLU, SQuAD | MT-Bench |
| Long | RULER, LongBench | **? (empty)** |

**Long-conversation benchmark standard 없음** = 정확히 너가 짚은 빈 칸. 2026 frontier의 적시 research opportunity.

### Research-level 후보 approach

1. **Conversational NIAH**: state needle (turn 5에 "My dog is Buddy", turn 200에 "what's my dog's name?")
2. **Synthetic state machine + dialog**: 외부 state machine + 100-turn auto-gen, state tracking 측정
3. **Contradiction probe**: judge LLM이 모든 turn pair 비교 (O(N²) cost)
4. **User simulator + goal achievement**: persona/goal 가진 simulator와 agent의 100+ turn, goal 달성 + coherence
5. **Persona consistency probe**: turn 1 vs turn 100 style classifier similarity
6. **Multi-axis conversation benchmark**: 위 5개 + 8 axis 모두 결합 = "RULER for conversations"

### Ch-25/27 framework 연결

- **[[Q12]] customer-LLM realism (ch-25)**: super-long conversation 영역 확장. User simulator가 100 turn 일관성 있게 행동 필요 — 현재 어려움.
- **[[Q9]] categorization critique (ch-27)**: long-context capability vs long-conversation capability 구분 필요. 현재 섞여서 평가됨.
- **[[Q15]] per-objective density (ch-27)**: Long conversation의 anchor density가 document QA와 다름 (temporal axis). N-hop cross-turn state evolution pattern.

### Chapter 28의 implicit 인정

§6 takeaway / §1 RULER table / §2.3 LongChat 모두 *완전 해결 paper 없음*을 implicit하게 인정. 너 critique이 chapter의 blind spot 짚음.

### 한 줄

**RULER QA-shaped → 100+ turn pure conversation 평가 *no standard*. Conversation 8 axis (state/persona/preference/reference/commitment/drift/contradiction/style) 모두 RULER에 없음. Long-conversation benchmark category 비어있음 — 2026 frontier 적시 opportunity. 5+ research approach 후보 (conversational NIAH / state machine / contradiction probe / user simulator goal / persona drift). [[Q12]] customer realism + [[Q9]] categorization + [[Q15]] anchor density의 *super-long conversation 영역 확장*.**

→ Framework move: chapter의 missing benchmark category 식별 + 5 research approach 후보 제시. Ch-25/27 framework 3개 결합해서 conversation-specific evaluation gap 명료화.

---

## Q7 — Long-context의 *core insight*?

> **Long-context capability ≠ single capability. *세 independent axis의 co-design product*.**

세 axis (서로 직교, 어떤 axis로도 대체 불가):
1. **Position encoding** (mechanical) — model이 수학적으로 long position 표현 가능
2. **Data coherence** (statistical) — model이 long-range pattern을 본 적 있음
3. **Evaluation taxonomy** (measurement) — 측정 가능 + 진짜 capability 잡음

### "Implicit → explicit" 전환

| | Short-context | Long-context |
|---|---|---|
| Position encoding | 자동 작동 | θ rescale 명시 결정 |
| Data coherence | 자연스러움 | Coherence filter 디자인 (concat 안 됨) |
| Eval | Simple benchmarks | NIAH/RULER/BABILong 합성 필요 |

각 axis가 *implicit free parameter*에서 *explicit design choice*로 promotion됨.

### 4 sub-insights

1. **Single number는 underspecified**: "128K context"는 의미 없음. `(model × task × metric × length)` tuple 명시 필요.
2. **Synthesis is unavoidable**: 자연 데이터에 long-context 부족. CPT data / SFT / eval 모두 합성.
3. **Capability claim의 truthfulness가 metric에 dependent**: NIAH 99% + RULER 74 = "128K claim"의 truth value가 metric-relative.
4. **Marketing incentive와 truth는 antagonistic**: NIAH는 market에 좋음, RULER는 truth에 가까움. Researcher가 RULER+BABILong demand해야 함.

### Operational implications

- Model selection: NIAH만 보지 말 것
- Production: Effective context (~claim의 50-70%) 신뢰
- Training: 3 axis 모두 디자인
- Eval: NIAH+RULER+BABILong *모두*
- Synthetic data: coherent long docs (concat 안 됨)

### Ch-25/26/27 framework로의 수렴

- **[[Q9]] anchor (ch-25)**: 3 anchor (positional + coherence + instruction) 동시 활성
- **[[Q4]] distribution overfit (ch-26)**: NIAH-overfit = 가장 극단적 instance
- **[[Q7]] eval as data spec (ch-26)**: 최대 강도 — RULER가 training taxonomy 결정
- **[[Q11]] verifier hierarchy (ch-27)**: NIAH (self) < RULER (teacher) < BABILong (world) < real production
- **[[Q15]] per-objective density (ch-27)**: 가장 극단 — 4 component가 5-9 orders 다른 density

→ **Ch-28 = ch-25/26/27 framework의 *boundary case* 모음**. New framework 추가 X.

### Course의 *진짜* trajectory (chapter 26→27→28 pattern)

| Chapter | Modality | 무엇이 explicit이 됐나 |
|---|---|---|
| 26 | Tool calling | Verifier (implicit → 3-layer explicit) |
| 27 | World-conditional | Environment (no env → Docker/predicate) |
| 28 | Long-context | Position encoding (auto → RoPE θ) |

각 chapter = *implicit → explicit*의 한 layer 노출. Course의 진짜 axis.

### 한 줄

**Long-context = 세 axis (position / data / eval)의 co-design. Short-context의 implicit free parameter들이 explicit design choice로 promotion됨. "Context length N"은 `(model × task × metric × length)` tuple. NIAH = marketing, RULER+BABILong = truth. Synthesis가 unavoidable (3 axis 모두). Ch-25/26/27 framework의 boundary case 모음 — new framework 없음.**

---
