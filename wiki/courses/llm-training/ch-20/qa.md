<!-- chapter: ch-20 Q&A; deps: [[read]]; kernel answers only -->
# Ch-20 — Reading Q&A

## Q1. Rollout이란 무엇인가? (ch-19와 ch-20의 연결고리)

**Rollout = prompt 하나에 대해 모델이 한 번 sample해서 끝까지 생성한 token sequence 전체 (= one trajectory).** 100 token이든 30,000 token이든 *EOS까지 굴린 응답 한 덩어리* = 1 rollout.

**왜 "trajectory"라고도 부르나** (RL terminology): state(지금까지 생성된 sequence) → action(다음 token sample) → state → action → ... 가 EOS까지 이어진 경로. LLM의 한 응답이 곧 한 trajectory.

**ch-19 vs ch-20 — same word, different generator**:

| 챕터 | Rollout 생성 주체 | 용도 |
|---|---|---|
| ch-19 (rejection-sampling FT) | **student 자기 자신** (K번 self-sample) | verifier 통과한 rollout만 골라 self-distillation |
| ch-20 (teacher distillation) | **더 강한 teacher** (R1, GPT-4) | teacher의 reasoning trace 통째로 student가 imitation SFT |

→ 챕터 line 18의 *"the other branch of the same idea"*가 의미하는 바: 같은 framework(prompt → rollout sample → verifier filter → SFT data), 차이는 누가 sample했느냐뿐.

**R1-distill 맥락에서의 한 rollout 크기**: `<think>...수천~수만 token reasoning...</think> Answer: X.<EOS>` 전체가 *단 1개의 rollout*. R1-distill SFT data 한 sample = (problem, R1의 full reasoning trace + answer).

**pass@k와의 연결**: 같은 prompt로 K개의 rollout 생성 → 각각 verifier 통과 여부 체크 → pass@k = 통과 비율. K개의 독립적인 trajectory를 굴렸다는 뜻 (네가 ch-19 verdict E6에서 reinvent한 pattern). ch-20의 R1-distill curation도 같은 메커니즘으로 trace 품질 filtering.

---

**Q2-Q4** (Orca invariance / prompt erasing / Orca-2 strategy selection / DSBS multi-task) → [[qa-deep]]
**Q5-Q7** (weight sharing / control-axis framework / λ golden balance + ablation) → [[qa-deep-2]]

Splits follow CLAUDE.md "split if it grows" rule (120-line cap).

## Q8. `\boxed{}` — math reasoning의 final-answer convention

**LaTeX 명령어** — 원래 인자를 상자 안에 그려 강조. Math reasoning 맥락에서는 **parsing sentinel**로 사용 — 5K~30K token long CoT의 마지막에 final answer가 어디 있는지 기계적으로 추출하는 marker.

```
R1 output: "... Final answer: \boxed{8}"

Verifier pipeline:
1. regex \\boxed\{(.*?)\}로 마지막 boxed 값 추출 → "8"
2. SymPy로 gold answer와 symbolic equivalence 체크
3. 일치 → trace 통과
```

**R1이 왜 emit하나**: RL training reward가 `extract(\boxed{...}) == gold ? 1 : 0`. 모델이 `\boxed{}` 없이 답하면 reward 0 → RL이 boxed-emission을 강하게 enforce. Convention의 source는 Math Olympiad (AIME, IMO) 전통 → MATH dataset (Hendrycks 2021) → 모든 후속 reasoning model 상속.

**Ch-20에서 핵심 역할** (line 21, 102, 123, 126, 171): R1-distill / Bespoke-Stratos / Open-R1 / Sky-T1 filter pipeline의 기본 building block. Format filter (`\boxed{}` 있나?) → Math verify (SymPy로 boxed 값 == gold) → trace keep/drop. `\boxed{}` 없으면 filter 시작 못 함.

**Failure mode**: outcome filter라서 *답만 본다*. Trace 내부 reasoning은 검증 안 함. R1이 문제 misread → 다른 문제 풀이 → boxed 값이 우연히 gold와 일치 → filter 통과 → student가 wrong reasoning 학습 (chapter line 171 *"wrong-question-correctly"*). Process reward model (ch-24)가 유일한 방어선.

## Q9. "Median trace length"는 reasoning length인가?

**Yes** — token 단위의 reasoning 길이. R1-distill의 "trace" = teacher가 한 problem에 대해 emit한 *전체 output sequence* = `<think>...수천~수만 token...</think> \boxed{X}`. Answer 부분이 짧으니까 *거의 전부 reasoning length*. Q1의 rollout 정의와 동일 개념.

**Line 130 숫자**: R1 median ~5K (tail 30K, 10% >15K), Bespoke/Sky-T1 median ~3K (filter로 긴 trace 컷). R1 본가가 가장 길게 생각.

**왜 median이지 mean이 아닌가**: distribution이 heavy-tailed. 10% > 15K이라는 건 long tail이 무겁다는 뜻. Mean을 쓰면 long tail에 휘둘려 misleading. Median이 typical sample size를 정직하게 표현.

**Distillation에 implication** (line 143과 연결):
- 긴 trace = quality↑ (정교한 reasoning이 신호) + cost↑ (token budget 빨리 소모) + failure (trivial 문제에도 long trace 흉내)
- Open-Thoughts ablation: QwQ-32B(median 3K) > R1(median 5K) as teacher for Qwen2.5-7B student. 같은 token budget으로 student가 더 많은 (prompt, answer) pair를 봄.
- → **"긴 trace의 teacher" ≠ best teacher**. Student token budget 고려 필수.

**Control-axis framing 확장 (Q6)**: DSBS는 trace를 *training-only*로 통제, R1은 trace length를 *unrestricted*로 둠 (RL이 자유롭게 늘림), Open-Thoughts는 trace length도 *control axis*가 돼야 한다고 발견. → trace length는 R1이 통제 안 한 axis = R1 다음 method가 잡을 후보 (ch-22 quality selection에서 다시).

## Q10. Math-Verify의 geometry/proof 실패 — outcome-only 때문인가?

**아니. 두 *직교* limitation을 conflate한 framing.** 분리해야 함.

**Limitation A — Outcome-only** (line 171, *wrong-question-correctly*): `\boxed{}` 안의 final value만 검증, reasoning은 안 봄. *모든 domain에서 발생*. 예: "17×23=?" → R1이 reasoning은 garbage ("17+23=40, 그냥 391 추측")인데 boxed=391로 gold와 일치 → filter 통과 → student가 wrong reasoning 학습.

**Limitation B — Domain coverage** (line 141, 너가 놓친 것): Math-Verify는 SymPy 기반 → *symbolic/numeric equivalence*만 normalize 가능.

| Answer type | SymPy verify? | 이유 |
|---|---|---|
| `\boxed{42}`, `\boxed{(x+1)²}`, `\boxed{{-1,-2}}` | ✓ | algebra/numeric normalize 가능 |
| `\boxed{30°}` (geometry, numeric) | ✓ | 숫자 비교 |
| `\boxed{triangle is isosceles}` (geometry, 관계) | **✗** | SymPy가 의미 모름 |
| `\boxed{∃ infinite primes}` (proof, 존재성) | **✗** | proof 비교 불가 |
| Proof itself (boxed value 없음) | **✗** | Lean/Coq 필요 (formal proof만) |

→ Geometry/proof failure는 boxed-only 때문이 아니라 **SymPy가 그 domain을 표현 못 하기 때문**. 두 limitation 효과 차이:

| | Limit A | Limit B |
|---|---|---|
| 어디서 | 모든 domain | Geometry/proof만 |
| 실패 방식 | False positive (wrong reasoning 통과) | Filter 적용 불가 |
| 해결 | Process reward (ch-24) | 다른 verifier (LLM-judge, Lean) |

**Line 141의 핵심 명제 — *"Verifier determines the upper bound"***: Math-Verify가 못 잡는 axis(geometry, proofs) = student가 학습 못 하는 axis. 왜냐하면 해당 domain trace는 filter가 uniform하게 처리(다 통과 or 다 reject) → 좋고 나쁨 구분 없음 → student는 teacher의 평균 noise만 학습. Stratos가 SymPy + unit-test + LLM-judge 3-layer 쓰는 이유 = 각 verifier가 다른 domain cover. **Student 성능 순위 = verifier coverage 순위** (Stratos > Open-R1 > Sky-T1).

**Verifier control axes (Q6 확장)**: (1) Coverage = 어떤 domain까지, (2) Quality = 얼마나 정확히, (3) Independence = generator와 다른가 ([[ch-19]]). R1-distill 계열은 (2)(3)은 통제하지만 (1)은 uncontrolled → math 밖은 거의 not handled. ch-22/ch-44 후보 axis.

## Q11. §4.3 "Stronger ≠ Better teacher" — QwQ-32B > R1 (for Qwen2.5-7B)

**Result** (Open-Thoughts ablation, line 143): R1이 모든 benchmark에서 최강이지만 Qwen2.5-7B student SFT 시 QwQ teacher가 더 강한 student 만듦.

**3 mechanisms**: (i) **Distribution shift** — QwQ는 Qwen 계열 (같은 vocab/tokenizer) → SFT gradient 작음, 기존 capability 보존. R1은 DeepSeek-V3-Base 분포 → gradient 큼 → student base와 충돌. (ii) **Token budget** — R1 median 5K vs QwQ 3K → 같은 예산에 QwQ는 ~67% 더 많은 sample. Small student는 5K 깊이 다 internalize 못 함. (iii) **Format quirks** — R1의 중국어 leak, hollow `<think>`, reflection tic이 filter 통과 → small student가 verbatim copy.

**Sky-T1 contrast**: 작은 teacher를 *비용*만 위해 선택 → 약한 student. Capability axis 필수. **Best teacher = argmax(Capability × DistributionMatch × FormatCleanness)** — 3-dim AND. **Bottleneck이 student일 수 있다**: R1 이전 = "teacher = ceiling", R1 시대 = "student가 흡수 가능한 teacher = ceiling". → Control axis 7 (Q6 확장): teacher-student distribution match.

## Q14. "Question-side filtering matters more than answer-side" — 무슨 의미?

**Framing 정정**: *"matters more"* = "더 *중요하다*" (문제가 된다 X). 의미: **answer-side보다 question-side filter가 더 큰 leverage**.

**Pipeline 위치**: `Source → Q-filter (problem keep?) → Teacher inference → A-filter (trace keep?) → SFT`.

**Q-side가 더 중요한 3가지 이유**:
1. **Garbage in, garbage out** — bad problem (ambiguous/trivial/mislabeled)은 plausible trace 만들어 A-filter 통과. Root cause 차단은 Q-side만.
2. **Compute 효율** — teacher inference가 비쌈 (5K-30K token/trace). Q-side에서 70% drop = *3× 비용 절감*. Line 12 *"~10-20K correct traces"* 정신.
3. **Curriculum shaping** — A-filter는 pass/fail만 봄, 난이도 분포 못 shape. Q-side만 난이도 label로 직접 design 가능 (trivial=signal 0, hard=noise, medium-hard=sweet spot).

**왜 LLM-labeled + response-length이 embedding/fastText를 이기나** (line 182): embedding은 *의미 ≠ 난이도* ("3×5" vs "3+5" 같은 vector, 다른 난이도), fastText는 surface-only. LLM-labeled = content understanding (cost: 1 LLM call/problem). Response-length = teacher trace 길이 = teacher가 자동으로 만드는 *free signal*.

**A-side filter 한계** (line 181): format + correctness까지만. 그 이상 (length/difficulty/quality judge)은 *signal-removing*. Sky-T1의 LLM-judge가 약한 student와 correlated 이유.

**ch-19 framework 재정의**: 너의 *"verification is the moat"* → ch-20에서 **target이 answer가 아니라 question**. Stratos 17K가 R1-distill 800K 가깝게 가는 이유 = high-quality Q + high-quality trace.

**Control axis 9** (Q6/Q11/Q13 확장): question-side curation. Stratos (1× careful Q) / Open-R1 (2× broader Q) / OpenThoughts (16× LLM-labeled Q) → 3-way 차별화.
