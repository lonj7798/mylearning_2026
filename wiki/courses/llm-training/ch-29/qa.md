<!-- chapter: ch-29 — Lab: Synthetic Instruction Set with Filter + Dedup + Verify
     companion to: [[read]]
     mode: application probe — learner applied the ch-29 cascade to a real project
           (Lina TMR sales-call synthetic conversation pipeline), not a chapter re-read
     append-only across cycles
-->

# Ch-29 Q&A — Synthetic-data cascade applied to a live sales-call pipeline

Learner brought their production system (`boson-agent-dev` Lina TMR: human-customer ×
LLM-telemarketer) and used ch-29's `Generation → filter → dedup → verify → selection`
cascade to design a synthetic **conversation** pipeline. Target: SFT a salesperson agent,
cold-start (no real call transcripts). Kernels below; ★ = framework extension.

---

## Q1 ★★ — single-QA 합성은 쉬운데 conversation은 왜 어렵나

**conversation synthesis = data generation이 아니라 *environment simulation*.** single-QA = 분포에서 i.i.d. *점* 뽑기 (상태 없음, per-sample 검증 가능, 독립 sampling). conversation = 결합된 동역학계에서 *coherent trajectory* 뽑기 (두 에이전트 + 상태 + 동역학 + 결과). 생성기가 *함수*가 아니라 *세계*. → 이 대화가 자꾸 RL 문제로 변한 이유. category 차이지 degree 차이가 아님.

## Q2 ★★ — customer를 어떻게 통제하나 (actor vs controller)

원칙: **LLM은 *말만* 한다(actor). *무엇을 말할지*는 절대 안 정한다(controller).** "taxonomy 다 주고 알아서" → "확률 순으로 골라" → 매번 같은 이유로 실패: **control 결정은 LLM bias를 상속한다.** ch-26 [[Q3]] 정보 비대칭 / ch-27 [[Q11]] verifier hierarchy의 conversation 버전.

## Q3 ★ — layered-rules를 어떻게 떼나 (context distillation)

"생성 때 inject, 끝나고 90% 제거" = **context distillation** (Askell 2021; Agent-FLAN). scaffold inject(행동을 가르침 → strip) vs steering inject(추론 때도 쓸 런타임 노브 → keep). rules가 사라지는 게 아니라 *추론 제어→가중치*로 이동. 단 compliance stage(informed_consent / DNC / 계약 녹취)는 distill 금지 — **hard rail 유지** (regulatory). 검증 가능+필수인 곳엔 hard gate.

## Q4 ★★ — influence는 SFT/RL 중 어디서? → "단계"가 아니라 "환경"의 문제

"influence는 RL에서" 는 문제를 *이동*시킬 뿐. RL이 multi-turn influence를 배우려면 `P(s_{t+1}|s_t,a_t)`가 action에 의존해야 함. 시나리오가 bucket을 *사전 추첨*하면 RL 때도 transition이 action-독립 → policy gradient 0. **non-reactive 환경은 SFT뿐 아니라 RL도 망친다.** 정리: SFT = freewheel + coverage(reactive 불필요), RL = reactive 환경 필수(ch-31 bridge). frozen per-round arc는 *두 단계 다* 틀림.

## Q5 ★ — diversity를 어떻게 확보하나 (decompose)

"diversity"는 한 덩어리가 아님. **surface/persona/concern (조합적, 쉬움 — sampler로 해결) vs trajectory/difficulty/outcome (어려움 — LLM이 mode-collapse).** 후자는 "잘 프롬프트하면" 안 나옴, *engineer*해야. **deck/hand**: taxonomy=덱, 고객=작은 손패. diversity는 "모두가 모든 걸 가짐"이 아니라 "각자 *다른 작은 부분집합*"에서 나옴. "다 주기"는 coherence·diversity·tail 셋 다 깸.

## Q6 ★ — head vs tail (operational 정의)

**테스트: LLM이 label만 보고 잘 만드나? yes=head(예시 불필요), no=tail(real example grounding 필수 + sampler 강제).** head = 흔하고 generic (관심없어요/비싸요). tail = 드물고 상황에 박힘 (고지의무 위반 거절 같은 deal-killer), LLM prior에 없음. 핵심: **biased candidate set은 randomize로 못 빠져나온다** (top-10 확률 랭킹 안에서 random 뽑아도 tail은 애초에 top-10 밖).

## Q7 — coverage는 sampler 몫, LLM 선택 몫이 아니다

"확률 높은 순 고르기" = LLM의 head-heavy prior를 selection으로 enshrine → tail 매장. tail은 sampler가 **강제 등장**시켜야 (확률 순이 아니라 차라리 *anti-probability* rare-boost). sampler weight는 LLM 확률도 로그 빈도도 아니고 **target coverage 분포**에서 옴.

## Q8 ★ — 확장한 taxonomy를 어떻게 쓰나 (dual-use)

같은 taxonomy 파일이 두 일을 함: **(A) generation seed-bank** (leaf `examples`를 few-shot로 → label-expand 아니라 exemplar-ground), **(B) classifier/coverage 계량기** (`distinguishing_signals`+`definition`으로 turn labeling → 분포 측정 + residue 발견). = ch-26 [[Q7]] eval-as-data-spec. **residue가 보물** — 기존 카테고리로 classify하면 tail이 head로 뭉개짐(lossy projection); data가 taxonomy를 *재형성*하게 (bottom-up).

## Q9 — cold-start grounding: human-played 로그

real call transcript 없음, but **human이 customer 연기 + LLM이 TMR** 한 로그 존재 = loop-외 anchor. 단: (a) 연기라서 **content는 신뢰, frequency는 불신** (role-play distribution shift, ch-27 [[Q11]] world 바로 아래). (b) **TMR 턴은 옛 rule-based Lina라 학습 타깃 금지** — customer 쪽만 mine. 확장 taxonomy leaf는 gpt-5.4-mini가 상상한 것 → real 발화로 re-ground 필요(echo 제거).

## Q10 — trajectory를 P/N/U로 (call-level, outcome-anchored)

P/N/U는 **call당 1개**(per-turn 아님 — 옛 random-walk는 죽음), **종착 outcome**으로 정의(P=구매, N=거절/DNC, U=보류). 이게 v0의 장점: **intent를 outcome으로 검증 가능한 가장 낮은 해상도** → turn 1부터 미룬 verifier가 여기 삶(intent-class vs realized-terminal 일치). expand = class→shape→concern-event 시퀀스. terminal은 class가 *entail*하는 범위에서 고르고, 반전은 path가 *유발*해야(stapling 금지).

## Q11 — meta: 좋은 질문을 어떻게 하나

좋은 질문은 *발명*이 아니라 분석이 fork에 멈춘 자리에서 *수확*. recipe = [믿음+이유] + [깨질 지점] + [그 fork 판단 요청]. senior에게 최강: risk-prior("뭐가 제일 걱정?") / blind-spot("내가 안 묻는 게?"). 약한 질문 = 허락 구하기("이거 괜찮나요?"). → senior 대화용 질문 9개 + agenda 도출 (verifier 충분성 / sim-to-real 측정 / eval metric이 가장 약한 고리).

---

## Q-정정 (2026-09 revision)

**배경**: read.md가 2026-09 generality revision으로 새로 작성되었다. 정정 목록은 새 read.md의 "Corrections to the version you studied" 1–20번에 있다. 이 파일의 Q1–Q11은 read.md 본문을 다시 설명한 것이 아니라 Lina TMR conversation pipeline에 cascade를 적용한 application probe이다. 따라서 kernel 안에 이번에 정정된 사실(survival arithmetic, ln(V) loss gate, APIGen ablation 수치, IFD 정의와 pre-experience 방식, LIMA baseline, Self-Instruct의 output-first 방향, MinHash threshold 방향)을 담은 Qn은 없다.

- **Q1–Q11**: 정정할 사실 없음. 다만 Q2와 Q11에서 verifier를 설계 원칙으로 쓸 때, APIGen의 근거는 "layer를 하나씩 제거한 ablation(−18/−11/−6)"이 아니라 "실패한 sample을 training data에 다시 넣은 add-back ablation"이다. xLAM-7B에서 BFCL이 −4.06(semantic 실패 추가), −5.94(execution 실패 추가), xLAM-1B에서 −9.59, −12.17 변했고, format 단계 arm은 없다 (정정 3).
- **Q7–Q8의 coverage 논의와 연결**: 새 read.md §3.4는 filter 단계마다 category 비율과 coverage score를 기록하도록 한다. Cherry LLM에서 IFD 상위 5%는 "write story"류가 차지하고 "rewrite sentence"류는 하위 5%에 몰린다. 이 사실은 Q7의 "coverage는 selection이 아니라 sampler가 책임진다"는 kernel과 방향이 같다.
- **Line reference**: 이 파일의 이전 entry와 ch-29 관련 대화에서 가리키는 read.md line 번호는 git commit 4a72e54 시점의 read.md를 기준으로 한다.

## Q-재독 1. ROUGE-L은 어떻게 계산하는가? (2026-10-14)

**Kernel.** ROUGE-L = 두 문장의 LCS(같은 순서의 최장 공통 단어열, 인접 불필요)로 만든 F 값: P=LCS/|후보|, R=LCS/|기준|, F=2PR/(P+R)=2·LCS/(|a|+|b|). ch-29에서는 평가 지표가 아니라 Self-Instruct pool 유사도 필터(모든 기존 instruction과 비교해 전부 <0.7일 때만 추가, §2.2). Worked example: "Write a short poem about the sea"(7) vs "Write a poem about the ocean"(6): LCS=5, P=0.714, R=0.833, F=10/13=0.769 ≥ 0.7 → 거부(코드로 확인). 구현은 장에 없음(`admit_to_pool`은 `rouge_l`을 인자로 받음); `rouge_score` 패키지와 토큰화 옵션 영향은 장 밖 일반 지식. 한계: 표면 겹침만 봄, pool 크기의 제곱 비용(전체 샘플은 MinHash, §3.2), 0.7은 ablation 없음. 같은 질문: [[ch-18]] qa Q-재독 1. see [[read]] §2.2.

## Q-재독 2. Evol-Instruct에서 난이도를 어떻게 올리는가? 기존 response를 다시 쓰는가, instruction만 다시 쓰고 response를 재생성하는가? (2026-10-14)

**Kernel.** instruction만 rewrite하고 response는 새 instruction으로 재생성: D(0)={(I_k,R_k)}, 매 evolution에서 I(t)→I(t+1) rewrite 후 R(t+1) 생성, response는 "instruction alone"으로 prompting(WizardLM §3.1, §3.2, §4.2 fn.5). 기존 R(t)는 사용되지 않음. rewrite 입력이 instruction뿐인지는 card가 rewrite 대상을 I(t)로만 적어서 그렇게 읽힐 뿐, evolving prompt 원문은 미확인. 난이도 5연산: add constraints / deepening / concretizing / increased reasoning steps / complicate input(XML·SQL·Python·HTML·shell·JSON demonstration), 각 rewrite는 "a bit harder", 추가 10~20 단어(Example 3.1); step size ablation 없음(저자 해석: 극단적 복잡도는 generalization 해침). in-breadth 1연산은 같은 domain의 더 희귀한 새 instruction(Example 3.3). 6연산 중 1개를 균등 선택. 실패 4규칙 후 원본이 pool로 복귀. 응답 재생성 이유와 verifier 부착 확장은 Interpretation(장의 source 아님). see [[read]] §2.2, [[evol-instruct]].

## Q-재독 3. 판매 통화 합성에 쓰는 단일 system prompt를 어떻게 다양화하는가? (2026-10-14)

**Kernel.** 먼저 구분: A = 생성용 prompt(teacher가 대화를 만들게 하는 지시문), B = 데이터에 들어가는 system prompt(배포 모델이 받을 system message). 장의 근거: persona 삽입(Persona Hub, ch-19 §6), 제약 추가·구체화(Evol-Instruct in-depth, 10~20 단어, ch-29 §2.2), 같은 domain 새 prompt(in-breadth), 응답 스타일 system message 변형(Orca 16개, ch-20 §1), ROUGE-L<0.7 pool 필터(ch-29 §2.2). 제안(Interpretation): prompt를 독립 slot으로 분해해 조합 — 영업 담당자(말투·경험·정책), 고객(성격·숨은 상태·의도), 제품/사실, 시나리오, 결과 — 후 조건을 조금씩 추가(Evol-Instruct step size는 ablation 없음, 저자 해석: 극단적 복잡도는 generalization 해침). B의 긴 system prompt 규칙 준수는 별도 데이터/측정 필요(ch-29b §5). 주의: embedding 유사도 하락≠coverage(Prismatic: persona set Vendi 24.49 vs 22.03이나 OOD 더 낮음, ch-19 §6). 검증법(제안): 축 조합 held-out 평가, 같은 token 수의 단일 prompt baseline 비교, 현 prompt가 잘 작동한 조건을 ablation으로 분리. 선행 설계: [[ch-25]] qa Q-재독 9–11. see [[read]] §2.2, [[ch-19]] §6, [[ch-20]] §1.

## Q-재독 4. system message에 constraint를 계속 추가하는 evolving system prompt는 constraint 충돌/미준수 문제를 어떻게 다루는가? (2026-10-14)

**Kernel.** (질문 원문이 잘려 있어 "constraint 충돌 또는 teacher의 규칙 위반 위험"으로 해석.) Evol-Instruct elimination 4규칙은 constraint 충족을 검사하지 않으므로(§2.2) 별도 검증이 필요. 장의 장치: 작은 단계 10~20 단어(step size ablation 없음), constraint를 IFEval 30개 유형에서 sampling 후 충돌은 LLM 감지+사람 검증으로 제거(Multi-IF, ch-29b), 프로그램 검증(strict string의 false negative → loose 지표, GPT-4 76.89/79.30), guardrail = 객관적 pass/fail 기준의 규칙+규칙별 judge(RealGuardrails), 충돌 예시마다 aligned 예시 병행(Instruction Hierarchy benign compliance 83.1→60.4), IFBench(IFEval 25 유형 학습 모델이 unseen 58 constraint에서 하락). 구성안(Interpretation): 검사 방법이 붙은 constraint 목록 → 충돌 확인 후 1개 추가 → 응답 생성 → 모든 constraint pass/fail 검사 후 실패 행 제외 → constraint 수 n별 teacher 통과율로 상한 결정 → aligned 상황 병행. 통과율 곡선은 장에 측정 없음. 판매 통화 예: 녹취 동의 확인, 탐색 전 가격 미언급, 거절 시 재제안 1회. see [[read]] §2.2, [[ch-29b]] §4, §5.

## Q-재독 5. 대화를 생성한 뒤 전체 이력을 보고 system prompt를 다시 쓰는 것이 나은가? 대화에서 처리되지 않은 constraint를 추가해도 되는가? (2026-10-14)

**Kernel.** (질문을 "사후 rewrite + 발동하지 않은/어긴 constraint 추가"로 해석.) 사후 rewrite는 Humpback backtranslation과 같은 모양(502k segment마다 instruction 후보 생성 후 점수 필터; 자기평가 precision/recall M0 0.44/0.09, GPT-4 0.88/0.92 → 사후 쌍은 평가 단계 필요, ch-19 §8). 위험: 응답은 원래 prompt 아래 sampling, 학습은 rewrite prompt → conditioning 불일치(GAtt가 이전 turn loss 0으로 막는 이유, ch-29b §5); prompt가 규칙 목록이 아닌 대화 요약이 되는 문제는 추론. forward(prompt→대화) vs backward(대화→prompt) 비교는 장에 없음. 권고(Interpretation): 기본은 forward 유지, rewrite는 constraint 추가 목적에만 쓰고 응답 재생성 또는 충족 검사 재통과. 추가 constraint 3분류: exercised-pass(유지) / not-triggered(추가 가능; aligned 예시 역할, RealGuardrails·Instruction Hierarchy, 발동 여부는 규칙별 judge로 확인) / violated(제거 또는 응답 재생성). 비율 위험(항상 발동 → 모든 규칙이 항상 적용된다고 학습, 발동 안 하는 규칙 과다 → 규칙을 장식으로 무시)은 추론, 적정 비율 측정 source 없음→ablation 필요. 규칙 수별 준수율은 Q-재독 4 참조. see [[ch-19]] §8, [[ch-29b]] §5.

## Q-재독 6. IF(instruction following)를 유지하려면 system prompt가 몇 개 필요한가? (2026-10-14)

**Kernel.** 최소 개수를 정한 source 없음(개수별 ablation 없음). 데이터 규모 참고: RealGuardrails = 실제 system prompt 1,850개(guardrail 있는 것만), prompt당 conflicting+aligned 약 10개, 총 18,497; SFT+ mixture(7개 source 혼합)에서 Llama 3 8B S-IFEval 52.1→59.4, DPO 추가 시 77.9(mixture 전체 변경이라 prompt 수만의 효과 아님; ch-29b §5). Tülu 3 Persona IF(IFEval 25 유형 coverage) 제거 시 IFEval 72.8→53.6(ch-29, 유형 coverage 증거). Orca 16개 system message는 스타일 유도용(ch-20 §1). 해석(Interpretation): IF 유지는 prompt 개수보다 constraint 유형 다양성과 unseen 유형 평가가 중요(IFBench: IFEval 25 유형 강한 모델이 unseen 58개에서 하락). "개수"는 distinct prompt 수 / prompt당 대화 수 / constraint 유형 수로 분리되며 장은 분리 측정하지 않음. 결정법(제안): 같은 token 수로 distinct prompt 수 arm(예: 100/500/2,000) 비교, IFEval·S-IFEval·IFBench+short suite forgetting 측정, aligned:conflicting 동수 유지, IF 점수 plateau 지점이 필요 개수; 판매 통화 domain은 일반 IF(공개 mixture)와 domain prompt arm을 분리. see [[ch-29b]] §5, [[read]].
