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

## Q-정정 (2026-09 revision)

**배경**: read.md가 2026-09 generality revision으로 새로 작성되었다. 아래 Qn의 kernel에는 이제 정정된 사실이 들어 있다. 괄호의 번호는 새 read.md "Corrections to the version you studied"의 항목 번호이다. Q1–Q7에 있는 section·line reference(예: "§6 takeaway", "§1 RULER table", "§2.3 LongChat")는 git commit 4a72e54 시점의 read.md를 가리킨다.

- **Q1**: 원래 Kamradt harness의 needle은 "eat a sandwich and sit in Dolores Park on a sunny day"이고, 채점은 exact-substring이 아니라 GPT-4 judge의 1–10 rubric이다 (8). Llama-3.1-70B의 claimed 128K / effective 64K는 RULER Table 3의 결과이고 Llama 3 report에는 RULER가 없다 (11). Qwen2.5-14B-1M의 "RULER ~85%"는 source에 없다. Report는 RULER를 128K까지만 쓰고 14B-Instruct-1M은 92.2이며, 1M에서는 Passkey Retrieval이 perfect이다 (13). "실제 capability와 2배 gap"은 model마다 다르다. 같은 Table 3에서 GradientAI Llama-3 70B는 claimed 1M에 effective 16K이다.
- **Q2**: "Llama-3.1-70B @ 128K depth별 98% / 60% / 95%" table은 source가 없는 숫자이다. Position 효과의 확인된 수치는 [[lost-in-the-middle]]의 GPT-3.5-Turbo 20-document 결과(1번째 75.8%, 10번째 53.8%, 20번째 63.2%)이다. "multi-hop @ 32K ~50%" 예시와 0.9³ 계산은 설명용 가정이며 측정값이 아니다.
- **Q3**: Fu 2024의 RoPE base 200M은 paper에 없다. Paper는 "Xiong et al. (2023)처럼 base를 조정했다"고만 쓴다 (2). Frontier 비교 table은 틀렸다. ProLong은 coherence filter가 없고 40B token(64K에서 20B, 512K에서 20B)을 쓰며 SFT는 UltraChat만 쓴다. Synthetic long SFT data는 1%만 넣어도 점수가 떨어졌다 (1, 18, 21). Qwen2.5-1M은 5-stage(4K → 262K)이고 token 수는 보고되지 않았으며, SFT query는 Qwen2.5가, answer는 Qwen-Agent가 만든다. Qwen-Max는 쓰이지 않았다 (31–32). Llama 3의 long SFT generator는 405B self-distillation이 아니라 "earlier version of Llama 3"이다 (25). "Aliasing" 설명은 RoPE 실패 원인을 정확히 기술하지 않는다. Training 범위를 넘으면 low-frequency 차원에서 학습 때 보지 못한 rotation angle이 나타난다는 설명이 맞고, 자세한 내용은 ch-32b에 있다.
- **Q4**: Llama 3의 "stage별 θ = 80K / 160K / 320K / 500K" table은 source에 없다. θ = 500,000은 pre-training 전체에 쓰인 architecture hyperparameter이고, 6-stage long-context stage 동안 base를 바꾼다는 기술은 없다 (3, 23). ProLong은 "500K → 128M upfront swap"이 아니라 stage마다 base가 하나씩이다: 64K stage 8×10^6, 512K stage 1.28×10^8 (20). NTK-aware scaling은 θ × (target/original) 선형식이 아니라 θ' = θ · s^(d/(d−2))이다. Staged vs upfront trade-off table(안정성, compute 등)은 측정 결과가 아니라 해석이다.
- **Q5**: Llama-3.1-70B의 길이별 RULER 점수는 97 / 95 / 91 / 88 / 86 / 74가 아니라 96.5 / 95.8 / 95.4 / 94.8 / 88.4 / 66.6이다 (11). Llama-3.1-405B의 "effective ~96K"는 Table 3에도 Llama 3 report에도 없다 (12). Qwen2.5-14B-1M의 "~500K"는 source가 없다. RULER leaderboard는 ">128K"로 표기한다 (13). 13 task, 500 examples per task per length, answer prefix, 85.6 threshold는 맞다.
- **Q6**: LongChat은 ShareGPT long tail(≥8K token, ≥4 turn, 18K conversation)을 골라낸 것이 아니다. Vicuna의 ShareGPT conversation을 16K로 자르고 7B는 80k, 13B는 18k conversation으로 학습했다 (16). LongMIT은 5–10 turn 대화 dataset이 아니라 multi-hop long-context instruction dataset(MIMG framework)이다 (17). "Long-conversation benchmark가 없다"는 결론도 수정한다. LongChat의 LongEval topic retrieval(2023), [[locomo]], [[longmemeval]] 같은 long-conversation 평가가 존재한다. 자세한 내용은 ch-29b에 있다.
- **Q7**: Core insight의 "data coherence" 축은 ProLong의 실제 결과로 바꿔 읽는다. Continued training에서는 long data에 high-quality short data(약 40%)를 섞어야 하고, 100% long data는 perplexity가 좋아져도 SFT 후 downstream 점수를 낮춘다 (1, 18). "Effective context ≈ claim의 50–70%"는 source가 있는 일반 수치가 아니다. RULER Table 3에서 비율은 model마다 다르다(Llama-3.1-70B 1/2, GradientAI Llama-3 70B 16K/1M). BABILong은 tested LLM이 context의 10–20%만 효과적으로 쓴다고 보고한다. "NIAH = self verifier" 같은 hierarchy 분류는 source 결과가 아니라 learner framework의 해석이다.

## Q-재독 1. NIAH, RULER, BABILong은 어떻게 측정하는가? (질문은 ch-29로 적었으나 해당 내용은 ch-28 §1) (2026-10-14)

**Kernel.** 공통: 프로그램이 needle/사실을 긴 haystack에 끼워 넣어 사람 표시 없이 정답을 알고, 생성기 변수(needle 종류·방해 문장 수·hop 수·길이)로 인스턴스를 무한 생성. NIAH(Kamradt): Paul Graham 에세이의 정한 깊이에 문장 하나를 넣고 정한 길이로 자른 뒤 needle이 답인 질문을 해 GPT-4 judge가 1–10 rubric으로 채점; GPT-4-128K는 (길이, 깊이) 225칸 중 207칸이 10점; 특정 위치의 사실 하나를 찾는지만 측정(방해 문장 무시·여러 개 반환·사실 결합은 미측정). RULER: 13개 과제 구성(검색 4: S/MK/MV/MQ-NIAH, 다단계 추적: 변수 추적 VT, 집계: CWE·FWE, QA: SQuAD·HotpotQA에 방해 문단) × 과제당 500개 × 4K–128K, 모델의 chat template 안에서 answer prefix를 주고 recall 기반 정확도; `L_eff = max{L : 13개 과제 평균 S(model, L) > 85.6}`(85.6 = Llama2-7B의 4K 평균): Llama 3.1 70B 96.5/95.8/95.4/94.8/88.4/66.6 → 64K(주장 128K), 8B 32K 87.4·64K 84.7 → 32K, GradientAI Llama-3 70B 16K 90.8·32K 85.4 → 16K(주장 1M); 32K에서 0.2점 차이로 L_eff가 두 배 바뀜(임계값 통계, 점수 행 전체가 더 정보적). RULER 실패 분석: Yi-34B-200K에서 방해 needle 추가 시 256K에서 약 40점 하락, 질의 1 → 8개 시 약 15점 하락, 128K에서 CWE 출력의 80% 이상이 one-shot 예시를 복사. BABILong: bAbI 추론 사실(에이전트의 장소 이동)을 PG19 책 문장 사이에 숨기고 길이만 늘림; LLM은 맥락의 10–20%만 효과적으로 사용, 배경 텍스트가 있으면 모든 모델이 QA2(두 근거 사실)에서 85% 미만, 일부는 QA1(사실 하나)을 16K–64K까지 통과; RAG는 QA1에서 모든 길이에 약 60%이나 QA2·QA3에서 random 미만. 측정 한계: HELMET(35개 모델, 128K, 단일 연구)에서 Spearman ρ — NIAH vs ∞Bench QA 0.63, RULER MK 0.81, HotpotQA RAG 0.88, 합성 과제 중 응용 범주와 평균 0.8을 넘는 것 없음; NoLiMa — 질문이 needle과 단어를 거의 공유하지 않으면 GPT-4o가 32K에서 99.3 → 69.7(Interpretation: 단어 매칭으로 locate 가능); BABILong — bAbI 문장이 짧고 PG19(1919년 이전 출판) 문체와 달라 fine-tuning한 모델이 사실과 배경을 가르는 토큰을 배울 수 있음; Lost in the Middle — 20개 문서에서 정답 문서가 첫 번째면 75.8%·10번째면 53.8%·20번째면 63.2%(GPT-3.5-Turbo, App. G Table 6), 깊이 평균이 곡선을 가림. 결론: 합성 과제는 한 능력의 통제된 진단 도구이고 응용 전반의 대리 지표로는 약함; 평가 gate 설계는 ch-32c. see [[read]] §1.

## Q-재독 2. "continued training으로 context를 extension한 뒤 모델은 긴 input을 받아들이지만 그 input 전반에서 instruction을 따르지 못할 수 있다"는 무슨 말인가? (2026-10-14)

**Kernel.** 두 능력의 구분(§2 Problem): (a) 긴 입력을 받아들임 = 오류·붕괴 없이 처리(context extension: RoPE 조정 + 긴 시퀀스 continued pretraining), (b) 긴 입력 전반에서 instruction을 따름 = 문서의 어느 부분을 묻든 읽고 지시대로 답함(긴 문서가 prompt에 있고 응답이 그 문서에 의존하는 SFT 데이터). continued pretraining은 next-token 예측이라 "문서를 읽고 질문에 답하라"는 형식을 가르치지 않는다는 설명은 이 course의 해석. LongAlign 직접 측정(ChatGLM3-6B-64k, Table 3, LongBench-Chat): ShareGPT만(긴 데이터 0k) 3.73, 긴 예시 5k 5.99, 10k 6.28; 짧은 과제는 거의 불변(MMLU 45.5/46.6/45.5, MT-Bench 5.34/5.50/5.51); 단일 연구·모델 하나. LongAlpaca-13B는 32K를 주장하나 RULER Table 3에서 effective length 4K 미만(데이터: 긴 문서 QA 9k + Alpaca 짧은 QA 3k, 5 epoch, 생성 모델·필터 미공개). LongAlign 절차: RoPE base 10,000 → 2,000,000 + 64K 미만 시퀀스 100억 토큰 continued training → 9개 출처의 64K 미만 문서 샘플링(긴 것 upsample) → 문서마다 4가지 과제 프롬프트(일반·요약·multi-hop 추론·정보 추출) 중 무작위 하나로 Claude 2.1에게 "텍스트 전체를 덮는" 질문 5개 → 질문 하나를 골라 Claude 2.1이 답 → 표본 100개 수동 확인(정답 94, 오답 2, 불완전 3, 무관 1) → 긴 예시 10k + ShareGPT 76k로 2 epoch. 전체를 덮는 질문은 이 문제에 대한 대응이라는 것은 이 course의 해석. 부가: 긴 SFT 데이터는 target이 짧아(LongAlign-10k 평균 200 토큰, target/시퀀스 비율 0.015) pack 단위 토큰 평균에서 긴 시퀀스가 의도의 2배·짧은 것이 1/3 가중치를 받음(worked example: K=2 pack·M=4 시퀀스); 보정으로 LongBench-Chat 5.76 → 6.21(ChatGLM3-6B-64k), Llama-2-7B-64k 5.89 → 6.10(ch-04 §4와 같은 구조). see [[read]] §2, ch-04 §4.

## Q-재독 3. long context에 loss를 전부 넣으면 안 되는가? (loss의 대부분을 long context가 먹으니까) (2026-10-14)

**Kernel.** 두 문제의 분리. (1) 긴 입력 토큰에 loss를 줄 것인가: LongAlign을 포함해 대부분의 SFT 레시피(Llama 3, Tülu, Nemotron, ProLong; ch-04 §4)는 응답(target) 토큰에만 loss를 주고 긴 문서는 loss에서 뺌 — LongAlign-10k는 target 평균 200 토큰·target/시퀀스 비율 0.015(시퀀스 길이 약 13,300 토큰으로 환산), ShareGPT는 330 토큰·0.193(장에는 19.3으로 인쇄, 단위 없음, 퍼센트로 읽은 값; 시퀀스 길이 약 1,700 토큰으로 환산) → 응답만 loss일 때 loss 토큰 수는 200 대 330으로 비슷한 규모, 긴 문서가 loss를 먹지 않음. 입력 전체에 loss를 주면 loss 토큰이 약 13,300 대 약 1,700으로 약 7.8배(이 course의 환산)가 되어 토큰 평균에서 긴 문서가 gradient를 지배. 단 입력에 loss를 주는 것이 항상 나쁜 것은 아님: Llama 2 Long Chat은 self-instruct 긴 데이터에서 긴 입력 prompt에 LM loss를 추가해 QuALITY 59.3 → 77.3, Qasper 35.7 → 38.9(ch-04 §4, 모델 크기 표에 없음); IM은 입력이 길고 응답이 짧은 데이터에서 효과; ProLong SFT는 instruction 토큰 마스킹 + 유효 토큰 평균. (2) LongAlign의 실제 문제는 응답만 loss인데도 pack 단위 토큰 평균에서 시퀀스 가중치가 불균등(worked example: K=2 pack·M=4 시퀀스에서 긴 시퀀스 하나가 의도의 2배, 10 토큰 시퀀스가 1/3) — 시퀀스마다 응답 토큰 수가 달라서 생기는 문제이며 보정 `K/(N_i·M)`로 시퀀스 단위 평균과 같아짐(LongBench-Chat 5.76 → 6.21, ChatGLM3-6B-64k). see [[read]] §2, ch-04 §4.

## Q-재독 4. pack이 무엇인가? (2026-10-14)

**Kernel.** pack = 여러 개의 학습 예시(sequence)를 이어 붙여 최대 길이(예: 64K 토큰)까지 채운 학습 단위 한 줄(ch-04 §1의 "row"에 해당). 계층: batch(한 번의 gradient 계산) ⊃ pack K개 ⊃ sequence(문서 + 질문 + 응답 한 예시) ⊃ 토큰. LongAlign worked example(§2): K=2 pack·M=4 sequence에서 pack A = 응답 토큰 200개인 긴 sequence 1개(한 줄을 혼자 채움), pack B = 짧은 sequence 3개(응답 토큰 10·20·30개, 합 60). pack 단위로 토큰 평균을 내면 pack마다 1/K 몫을 갖게 되어 pack B 안의 응답 토큰이 적은 sequence일수록 가중치가 작아짐 → 가중치 불균등의 원인. 이유: sequence 길이가 달라 가장 긴 것에 맞춰 padding하면 낭비 → 짧은 것 여러 개를 한 줄에 채움(ch-04 §1); LongAlign에서 packing과 길이순 batching으로 ChatGLM3-6B-64k 학습 시간 45.4h → 각각 20.5h·19.1h(Fig. 5). 주의: pack 안의 무관한 sequence끼리 attention이 섞이지 않게 mask 필요(ch-04 §2 intra-document masking); LongAlign이 쓴 mask 방식은 이번에 읽은 절에 없음. see [[read]] §2, ch-04 §1·§2.

## Q-재독 5. ProLong처럼 context length를 조금씩 올리는 것이 표준인가? (2026-10-14)

**Kernel.** 관찰: 이 장의 세 보고서는 모두 multi-stage다 — Llama 3.1 405B(8K→128K, 6 stages, 약 800B tokens, 단계별 길이·token·mix 미공개, 종료 조건 = short eval 회복 + NIAH 해결), Qwen2.5-1M(4,096→262,144, 5 stages, RoPE base 동반 상승, token 미공개), ProLong(8K→64K→512K, 각 20B tokens, 8배 점프라 점진적이지 않음). 단, 단계적 확장 vs 단일 단계를 비교한 ablation은 세 보고서에 없으므로 "표준/더 낫다"는 이 장에서 검증되지 않음. ProLong Table 7(+4B tokens at 512K > at 64K, 64K 평가: recall 98.5 vs 95.0)은 "평가 length보다 길게 학습" 효과이지 schedule 비교가 아님. continued-training mix(60/40 ablation 최고, 최종 63% long/37% ShortMix)와 SFT synthetic long share(ProLong 0% 최고 55.7, Llama 3.1은 0.1% 사용)는 length schedule과 별개 변수. see [[read]] §4, §5, Table 8.

## Q-재독 6. long continued training은 SFT 전에 오는데, post-training처럼 모든 token을 loss에 쓰는가? (2026-10-14)

**Kernel.** 장은 continued training의 loss 범위를 명시하지 않음. 일반 지식(장 밖)으로는 문서 데이터에 모든 token next-token loss(pretraining 방식). SFT 쪽은 장에 근거 있음: Artificial Needles는 answer tokens only(§6), LongAlign loss weighting은 target token 기준(§2). "masking" 두 종류 구분: ProLong document masking = 문서 간 attention 차단(loss는 전부 계산, App. B.2), SFT loss masking = prompt token loss 제외(ch-04). 미확인: ProLong ShortMix의 Tulu-v2 8%(Table 9)에도 전부 loss를 계산하는지. see [[read]] §2, §5, §6.

## Q-재독 7. post-training에서도 context length를 늘리는가? SFT 전에 SFT 데이터를 넣어 늘리는가? (2026-10-14)

**Kernel.** 이 장의 source에서 길이 확장(RoPE base 상승, 최대 길이 증가)은 SFT 전 continued training/pre-training 단계에서만 일어남(Llama 3.1 6 stages, Qwen2.5-1M 5 stages, ProLong 64K→512K, LongAlign은 이미 64K로 확장된 모델에서 SFT 시작). SFT의 long 데이터는 확장이 아니라 긴 입력 instruction following과 regression 방지 목적(Llama 3.1 0.1% examples, LongAlign 10k+76k; ProLong은 short-only가 최고)(§5 Interpretation). Qwen SFT의 short≤32K→short+long≤262K는 pretraining 최종 길이와 같아 새 확장이 아님. "SFT 전 SFT류 데이터"에 가까운 것: ProLong ShortMix의 Tulu-v2 8%(short 유지용), Qwen continued training의 synthetic long task(FIM/retrieval/reordering, 문서 변형 형식). Qwen offline RL(≤8,192 pair)은 short-to-long transfer 해석(Interpretation), 길이 확장 근거 아님. SFT/post-training만으로 길이를 늘린 사례는 source에 없음. see [[read]] §3, §4, §5.

## Q-재독 8. 길이를 한 번 늘리면 RL에서도 그 길이를 유지하는 데이터가 필요한가? (2026-10-14)

**Kernel.** 이 장의 증거는 필요성을 지지하지 않음: Llama 3.1은 short-only DPO가 long을 해치지 않았다고 보고(SFT 모델이 long에 강했을 때, ablation 표 없음, §4.3.4), Qwen2.5-1M은 ≤8,192 token pair의 offline RL로 LongBench-Chat 상승(7B 7.32→8.08, 14B 8.56→8.76, Table 3; short-to-long transfer는 Interpretation). 단 반증도 아님: 두 근거는 offline preference 학습이고 long rollout online RL은 source에 없으며, RL 전후 같은 long suite의 직접 대조 측정이 출판되지 않음(ch-28 Question 8). "RL에도 long 유지 데이터 필요"는 미검증 가설 — 검증법: RL 전후 동일 long suite(RULER+HELMET 범주) 측정, long 샘플 혼합 arm vs short-only arm 비교. see [[read]] §4, §5.

## Q-재독 9. SFT에서 long context 데이터는 보통 어떻게 확보하는가? (2026-10-14)

**Kernel.** 이 장의 source는 모두 합성이며 세 방식: (1) 긴 문서에서 QA 합성 — LongAlpaca(9k long QA+3k Alpaca, generator 미공개), LongAlign(9 source, 4 task prompt, Claude 2.1이 질문 5개→1개 선택→답, 수작업 94/100 정답), Llama 3.1(8K chunk에서 QA 생성, 학습은 전체 문서 context); (2) 구조 이용 — Llama 3.1 hierarchical summarization, Python import 기반 code reasoning, 16K/32K/64K/128K bucket; (3) 에이전트 multi-hop — LongMIT/MIMG(single-hop→BGE 유사도 선택→merging→verifier 점수>8.5→context refill; high-quality 94.8, multi-hop 94.8 vs Self-Instruct 61.3/33.1, token 수 불일치). Qwen2.5-1M은 segment 기반 query+Qwen-Agent 답, 수량·ablation 미공개. 반복 구조 "chunk에서 질문, 입력은 전체 문서"는 Interpretation. LLM verifier는 긴 입력에서 사람과 kappa 낮고 선택 정밀도 높음. 실제 사용자 long 로그는 장 밖. 섞는 비율은 별개(Q-재독 5, 7). see [[read]] §2, §3, §4.

## Q-재독 10. 16k 입력을 본 적이 없으면 모델이 실패할 수 있는가? (2026-10-14)

**Kernel.** 가능하나 이 장의 근거는 "정확한 길이"가 아니라 "그 길이 범위를 학습하지 않은 경우": SmolLM2 short-only SFT는 2,048 token 초과에서 long 능력 손실, Llama 3.1 short-only SFT는 regression(수치 없음, §4.3.4), LongAlign은 64K 확장 후에도 long instruction 데이터 없으면 LongBench-Chat 3.73 vs 6.28(Table 3; 긴 입력은 받지만 따르지 못함, §2), 표방 길이≠유효 길이(LongAlpaca-13B 32K→<4K, Llama-3.1-70B 128K→~64K; RULER Table 3). 학습 길이 사이(예: 8K·32K 학습, 16K 미학습) 실패 측정은 장에 없음(근거 부재≠안전). 학습은 bucket(16K/32K/64K/128K)·최대 길이 비중(Qwen 75%)으로 구성; ProLong Table 7은 평가보다 긴 길이 학습이 유리. 범위 밖 vs 범위 내 보간의 위험 차이는 장 밖 일반 지식(RoPE). 검증법(Interpretation): 서비스 입력 길이 분포 vs 학습 bucket 비교 후 해당 길이에서 RULER 유효 길이+응용 suite 측정. see [[read]] §1, §2, §4, §5.

## Q-재독 11. 단일 instruction이 16k인데 학습에서 10k 넘는 단일 instruction을 본 적이 없다면? (2026-10-14)

**Kernel.** 두 경우를 구분: A = 긴 문서 + 짧은 질문(LongAlign·Llama 3.1·LongMIT의 형태, ch-28 데이터 전부), B = instruction 자체가 긴 경우(다수 규칙·제약의 system prompt/지시문). ch-28 SFT 데이터에 B는 없고, ch-25는 자기 파이프라인이 long system prompt with many guardrails를 만들지 않는다고 명시하며 ch-29b로 넘김. 관련 관찰(ch-29b §5): Llama 2는 처음 instruction을 몇 turn 뒤 망각(속성 recall turn 2 100% → turn 4 10% → turn 6 0%, Table 30), GAtt는 4,048 token 미만 대화에서 20 turn까지 100%(단일 연구). 16K 단일 instruction의 실패 여부는 읽은 source에 직접 측정이 없어 미확인. 가설(Interpretation): 길이 처리는 continued training, 긴 지시문의 조건 준수는 해당 형태의 SFT 데이터가 필요(LongAlign 3.73 vs 6.28 관찰의 연장). 검증법: 길이 16K 고정, 규칙 수를 늘리며 준수율 측정(해당 suite는 이 장에 없음). see [[read]] §2, [[ch-29b]] §5, [[ch-25]].
