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

## Q-정정 (2026-09 revision)

read.md가 2026-09 generality revision으로 다시 작성되었다. 아래 Qn의 kernel에 정정된 사실이 들어 있다. 앞선 entry의 line reference(예: "line 171")는 git commit 4a72e54 시점의 read.md를 가리킨다. Q2–Q7, Q12–Q13은 [[qa-deep]]와 [[qa-deep-2]]에 있다.

- **Q1**: ch-19는 generation methods chapter이고, 자기 sample에 대한 rejection sampling은 ch-31에서 다룬다. pass@k는 "통과 비율"이 아니라 k개 sample 중 적어도 1개가 정답일 확률이다. 통과 비율은 pass@1의 추정치이다.
- **Q2**: Orca Table 2의 16개 system message에는 plan-and-execute message가 없다. 실제 목록은 empty, detailed answer, explain-like-five(#4, #8), step-by-step and justify(#7, #11, #12), definition 분해(#15) 등이다. "style 다양화로 reasoning invariant을 학습한다"는 가설은 paper가 ablation으로 test하지 않았다(answer-only 비교 없음).
- **Q3**: Orca 2는 "5개 strategy lookup table"을 보고하지 않는다. step-by-step, recall-then-generate, recall-reason-generate, extract-generate, direct-answer "등"을 나열하고, task별 strategy는 "Orca 1의 성능을 참고해" 정한다. FLAN task는 126개 sub-category로 묶이고 sub-category마다 system instruction 1개를 쓴다.
- **Q4**: exemplar 수는 3–8개가 아니라 약 10-shot이다. paper는 λ 값을 보고하지 않으며, 공개 code README가 `α·label + (1−α)·rationale`에서 α = 0.5를 권장한다. teacher label이 틀린 rationale을 걸렀는지는 보고되지 않았다. "Open-R1이 wrong-question-correctly를 재발견했다"는 Open R1 자료에 없다. "rationale 50~200 vs R1 ~5K token"도 출처가 없으며, R1 SFT set의 평균 길이는 5,355.3 token이다(Table 5).
- **Q5, Q6**: "Orca v1은 항상 long trace를 낸다('2+2?'에도 400 token)"는 Orca와 Orca 2 어느 paper에도 보고되지 않았다. DSBS가 prefix로 task를 나누는 한 model이라는 설명은 맞다.
- **Q7**: label-only vs DSBS 수치표(ANLI ~50→58–60, SVAMP ~50→66), rationale density ablation, benchmark별 best λ 표는 paper에 없다. 실제 Table 2(T5 220M, 100% data, e-SNLI/ANLI/CQA/SVAMP)는 standard fine-tuning 88.38/43.58/62.19/62.63, multi-task 89.51/49.58/63.29/65.50이다.
- **Q8**: 세 reproduction의 math filter는 모두 SymPy가 아니다. Bespoke-Stratos는 gpt-4o-mini judge, Sky-T1은 exact match, Open R1은 Math-Verify와 Llama-3.3-70B-Instruct judge를 쓴다. process supervision은 ch-24가 아니라 ch-44이며, R1의 V3 judge prompt(Listing 4)는 reasoning process가 reference와 일치하는지도 판정한다. R1은 PRM을 unsuccessful attempts(G.2)에 넣고, model-based PRM은 "inevitably leads to reward hacking"이라고 서술한다.
- **Q9**: "R1 median ~5K, Bespoke/Sky-T1 ~3K, QwQ 3K vs R1 5K"는 어느 출처에서도 확인되지 않았다. 확인된 값은 R1 SFT set 평균 token 수이다: Math 6,094.2, Code 7,435.7, General 1,419.8, 전체 5,355.3(Table 5).
- **Q10**: Bespoke-Stratos에는 SymPy + unit test + LLM judge 3-layer stack이 없고, verifier coverage로 student 순위를 비교한 source도 없다. 따라서 "student 순위 = verifier coverage 순위(Stratos > Open-R1 > Sky-T1)"는 근거가 없다. Math-Verify의 내부 구현은 확인한 Open R1 page에 서술되어 있지 않다.
- **Q11**: QwQ-32B가 더 좋은 teacher라는 결과(44.2 vs 42.3 code, 44.2 vs 41.6 math, Table 8)는 맞다. 그러나 R1은 모든 benchmark에서 앞서지 않는다(CodeElo, GPQA-D, JEEBench에서만 앞선다, §4.6). distribution shift, token budget(~67%), format quirk 세 mechanism은 OpenThoughts가 test하지 않은 해석이다. Sky-T1 blog는 QwQ를 비용 때문에 골랐다고 쓰지 않는다.
- **Q12**: Bespoke-Stratos의 prompt당 sample 수는 보고되지 않았고 dataset은 16,710 rows이다. Open R1은 "220K × 2 = 440K"가 아니라 400k problem × 2(일부 4) = 800k trace를 생성했고, default 93,733 rows와 extended 131,396 rows를 공개했다. OpenThoughts의 question filter는 pass@16이 아니라 GPT-4o-mini difficulty rating(code)과 GPT-4.1-mini response length(math, science)이며, teacher sampling temperature는 보고되지 않았다.
- **Q13**: "Qwen2.5-7B-Instruct → R1-distill: AIME25 18→53, MT-Bench 7.2→5.8, IFEval 72→54"는 출처에서 확인되지 않았다. 확인된 비교는 다음과 같다: IFEval Llama-3.3-70B-Instruct 90.38 → R1-Distill-Llama-70B 78.74, Qwen2.5-32B-Instruct 80.96 → s1-32B 58.04(MathIF Table 13, controlled 비교 아님); Qwen3-14B-Base에 math trace SFT를 하면 IFEval 69.2 → 42.3, 같은 problem으로 RL을 하면 70.0이다(transferability Table 1).
- **Q14**: Sky-T1은 LLM judge를 쓰지 않았다. "~10–20K correct traces" guideline은 정정되었다(OpenThoughts3는 1M 규모까지 data 크기에 따라 성능이 오른다, Fig. 1, Fig. 8). Bespoke-Stratos-32B는 R1-Distill-Qwen-32B에 AIME 2024 63.3 vs 72.6(reported)으로 9.3 point 뒤진다. question-side filter가 answer-side filter보다 효과가 크다는 OpenThoughts 결론은 유지된다.

## Q-재독 1. Orca 2에서 "대화가 학습 데이터에 없었기 때문"은 무슨 말인가? (2026-10-02, 개정판 재독)

**Kernel.** Orca 2 학습 데이터(§4.1, 약 817K)는 FLAN 약 602K·few-shot 55K·수학 약 160K 모두 single-turn(지시 → 응답)이고, 후속 질문이 이어지는 chat 형식이 거의 없다. MT-Bench는 turn 1 뒤에 follow-up(turn 2)이 이어지는 2-turn 평가(장 밖 일반 지식)이며, Orca-2-13B 6.15 < LLaMA-2-Chat-13B 6.64(차이 0.49, 이 course 계산) < WizardLM-70B 7.76; 논문은 특히 turn 2 점수가 낮은 것을 "학습 데이터에 대화가 없었기 때문"이라 설명(Table 3, §6.4). 데이터의 합성 의사–환자 대화 2,000개는 카드상 specialized skills 학습 평가용 4항목 요약 과제로, 일반 chat 학습용이 아니라는 것은 이 course의 읽기(논문 설명 미확인). 한계: 저자들의 귀속일 뿐 대화 데이터를 추가해 비교한 ablation 없음; §1 Implication("강점은 질의 분포를 따른다")과 ch-08b §10 발견 1(task가 demonstration으로 있어야 배운다)과 같은 결 — 연결은 이 course의 해석. see [[read]] §1, 라이브러리 카드 [[orca-2]].

## Q-재독 2. MT-Bench가 뭔가? (2026-10-02)

**Kernel.** LLM 응답 품질을 강한 LLM(GPT-4)이 채점하는 대화형 벤치마크(Zheng et al. 2023, 라이브러리 카드 [[judge-llm-bias]]). 80개 multi-turn 질문, 8개 범주(writing, roleplay, extraction, reasoning, math, coding, knowledge I/STEM, knowledge II/인문·사회; 범주당 10문제 수작업, §2.2), 각 질문은 2 turn(첫 질문 + 후속 질문). judge 방식 세 가지: pairwise comparison, single-answer grading(1~10점; Orca 2의 6.15 등이 이 방식의 평균), reference-guided grading(§3.1). 신뢰도: GPT-4가 사람 전문가와 85% 일치(동점 제외, 사람끼리 81%, Table 5a); 편향도 측정됨 — position bias(순서 교환 일관성 GPT-4 65.0% / GPT-3.5 46.2% / Claude-v1 23.8%, Table 2), verbosity bias(길이만 늘린 답을 고르는 실패율 Claude-v1·GPT-3.5 91.3%, GPT-4 8.7%, Table 3), self-enhancement(GPT-4가 자기에게 사람보다 10%p 높은 승률; 저자들은 자기 편향 여부를 "단정할 수 없다"), 수학·추론 채점은 약함(독립 정답을 주면 GPT-4 실패율 14/20 → 3/20, Table 4). ch-08b의 "GPT-4 judge의 경향" 질문에 대한 근거; judge 편향 본격 논의는 outline상 ch-49·ch-42. 80문제라 한 문제의 영향이 큼(표본 노이즈 연결은 이 course의 해석). see [[read]] §1, 카드 [[judge-llm-bias]].

## Q-재독 3. single-answer만 트레이닝하면 multi-turn performance가 낮아지는가? (2026-10-02)

**Kernel.** 용어 정정: 학습 데이터의 형식은 single-**turn**(질문 1회·답 1회)이고, "single-answer grading"은 MT-Bench의 채점 방식(답 하나에 점수; 이 course가 앞선 답변에서 혼동을 만듦)이다. Orca 2 사례(MT-Bench 6.15 < LLaMA-2-Chat-13B 6.64)는 다른 모델과의 비교이고 학습 파이프라인 전체가 달라 대화 데이터 유무만 바꾼 비교가 아니며, 저자들의 "대화 데이터 부재" 설명에는 ablation이 없음; 장 Implication의 "fell below its instruct-model baseline"의 baseline도 같은 모델의 이전 값이 아니라 다른 모델(느슨한 표현). 더 직접적인 근거는 ch-25 "Controlled data evidence": 1.7B SFT에서 single-turn MagPie-Pro는 MT-Bench 4.31·IFEval 30.45·MMLU-Pro 12.19, multi-turn MagPie-Pro-MT는 5.40·31.66·11.97(MT-Bench +1.09, 이 course 계산; [[modern-mix-composition]], SmolLM2 Table 10) — 단일 연구, 1 epoch, 1.7B, MT-Bench는 2 turn(Result, single study). 같은 장의 MT-Eval: Llama-2-chat-13B가 single-turn 7.55 → multi-turn 5.47(−2.08, 상대 27.5% 하락; GPT-4 9.17 → 8.84, 3.6%), 오답 200건 분석에서 49.5%가 앞선 지시를 따르지 않았고 48%가 앞선 turn의 오류에 끌려감. 결론: 방향(single-turn만 쓰면 multi-turn이 상대적으로 낮음)은 통제 비교로 뒷받침되나 일반 법칙은 아님. see [[read]] §1, ch-25 "Controlled data evidence".

## Q-재독 4. Orca는 테스트 때 zero-shot CoT를 썼나? CoT가 있을 때/없을 때 성능은? (2026-10-02)

**Kernel.** Orca는 CoT 없는 zero-shot, 빈 system message, temperature 0.7(별도 표기 없으면, footnote 14)로 평가: BBH 49.7(ChatGPT 48.9, GPT-4 67.4, Vicuna-13B 23.3), AGIEval 41.7(ChatGPT 47.2)(Tables 8, 11). CoT prompting 평가는 논문 §8이 스스로 untested라 명시 — CoT 사용 시 성능은 보고되지 않음. AGIEval 채점은 응답의 첫 대문자를 정답 id와 비교(§4.2.2; 답이 앞에 나오는 형식 전제 — 이것이 CoT 미사용 이유라는 것은 이 course의 추측, 카드에 설명 없음). 유일한 변형: 과제별 세 system message 중 최선을 고르면 ChatGPT와의 격차 4.4점(Table 9; 기본 격차 47.2−41.7=5.5이므로 약 1.1점 개선, 이 course 계산) — 과제별로 test 점수를 보고 고른 결과이고 CoT prompting과는 다름. 전제 정정: "teacher의 CoT를 배워서 성능이 올랐다"는 논문이 입증하지 않음 — 같은 질의에서 explanation vs answer-only 타깃 비교 ablation 없음; 향상은 데이터 출처·크기가 다른 Vicuna-13B(7만 vs 500만, Table 1)와의 비교; 논문 결론(§9)은 데이터 크기·커버리지·base 모델 품질을 강조하고 supervision 종류를 분리한 실험 없음(카드 Verification의 정정 항목). see [[read]] §1, 카드 [[orca]].

## Q-재독 5. Orca는 무엇을 어떻게 학습시켜서 좋아졌다는 것인가? (2026-10-02)

**Kernel.** 레시피(카드 [[orca]], §3): LLaMA-13B를 FLAN-v2 하위 컬렉션 4개(CoT 150K, NiV2 440K, Flan 2021 2.5M, T0 2M; Dialog 0)의 zero-shot 질의 약 500만 개 + 16개 system message 중 하나로 교사에게 보내 얻은 응답으로 두 단계 fine-tuning: 1단계 ChatGPT가 500만 전부에 응답(FLAN-5M), 2단계 GPT-4가 그중 무작위 100만에 응답(FLAN-1M; GPT-4 응답은 평균 1.5배 길음). 단계마다 4 epoch, 교사 토큰에만 loss, max length 2,048, packing(시퀀스당 평균 2.7개), A100 80GB 20장에서 160h + 40h(수집: ChatGPT 2주, GPT-4 3주). 주 비교 대상 Vicuna-13B(데이터 약 7만, Table 1): BBH 49.7 vs 23.3(>100%↑), AGIEval 41.7 vs 29.3(+42%, 이 course 계산), ChatGPT는 BBH 48.9 / AGIEval 47.2; 모두 zero-shot·CoT 없음. 무엇이 효과를 냈는지는 분리되지 않음: 데이터 크기(약 5M vs 7만, 약 71배, 이 course 계산)·질의 출처·응답 스타일·교사가 함께 다르고, Table 10(41.7 vs 37.18)은 ChatGPT 단계와 데이터 5배를 동시에 바꾼 비교이며, explanation vs answer-only 비교 ablation 없음; 논문 결론(§9)은 데이터 크기·커버리지·base 모델 품질 강조. 평가 범위 한계(§8): 학습 데이터가 zero-shot 프롬프트를 흉내 낸 것이라 성능이 tuning 분포와 강하게 상관할 것; multi-turn·few-shot·CoT prompting 미시험. 따라서 "설명 trace를 배워서"는 가설이고, 보고된 것은 "이 레시피 전체가 같은 크기의 이전 imitation 모델보다 좋았다"까지. see [[read]] §1, 카드 [[orca]].

## Q-재독 6. 결국 teacher model의 응답을 잘 써서 성능이 올라간 것인가? (2026-10-02)

**Kernel.** "교사 응답을 썼다"만으로는 Orca와 Vicuna의 차이가 설명되지 않는다 — Vicuna-13B도 교사(ChatGPT)의 출력으로 학습한 imitation 모델이다(§1 Problem). 차이는 "어떻게 썼는가": 규모(약 5M vs 약 7만), 질의(FLAN-v2 여러 과제), 응답 형식(system message로 유도한 설명형), 교사 구성(ChatGPT→GPT-4); 요소별 기여는 분리되지 않음(explanation vs answer-only ablation 없음, Table 10은 두 변수 동시 변경). 교사 응답으로 배워도 교사 수준에는 못 미침: ChatGPT와의 격차 — BBH Sports Understanding 67.2 vs 79.6, Ruin Names 39.5 vs 56.0, Geometric Shapes 20.8 vs 25.6, AGIEval 41.7 vs 47.2(GPT-4 62), 수학 관련 AGIEval 과제에서 격차가 더 큼(§6.1)(카드 [[orca]]); "응답 형식은 따라 하지만 지식은 그만큼 옮겨지지 않는다"는 이 course의 해석. 소스와 맞는 문장: "교사 응답을 대규모·다양한 질의·설명 형식으로 학습시킨 레시피 전체가 소규모 단순 imitation보다 BBH/AGIEval zero-shot에서 좋았고, 요소별 기여는 분리되지 않았으며, 교사 수준에는 미치지 못했다." see [[read]] §1.

## Q-재독 7. rejection-sampling SFT가 뭔가? (2026-10-02)

**Kernel.** 한 prompt에 대해 모델에서 응답을 여러 개 샘플링하고, 검증을 통과한 것만 남겨(틀린 것·형식이 나쁜 것은 거절) 그 응답으로 SFT하는 방식. DeepSeek-R1(§3): cold-start SFT(수천 개) → reasoning RL → rejection-sampling SFT(DeepSeek-V3-Base에서 재시작) → 혼합 보상 RL; 세 번째 단계에서 약 80만 샘플(추론 약 60만 + 비추론 약 20만)이 만들어지고 R1-Distill 모델들이 이 데이터로 SFT만 함(Supp. B.3.3). 추론 부분: 규칙 보상 외에 generative reward model(정답과 모델 예측을 DeepSeek-V3에 주고 correct/incorrect 판정) 사용; 언어가 섞인 CoT·긴 문단·코드 블록 제거; prompt마다 여러 응답을 샘플링해 정답만 보존. 보고되지 않음: prompt당 샘플 수, 교사 샘플링 temperature, 통과율(yield). §8 UniReason(Qwen3-14B-Base, 수학 4.7만 문제)도 SFT 타깃을 Qwen3-32B trace 중 rejection sampling으로 남긴 것으로 사용. 정정 항목 4: 자기 샘플에 대한 rejection sampling은 ch-19가 아니라 ch-31에서 가르침. Orca(§1)는 거절 없이 교사 응답을 그대로 사용, R1은 정답 필터를 거침 — ch-18의 Verify 단계가 "거절" 단계에 해당(연결은 이 course의 설명). see [[read]] §3, §8, ch-31.
