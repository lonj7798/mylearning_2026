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
