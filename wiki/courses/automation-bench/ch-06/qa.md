<!-- qa for ch-06 — The Grading Engine: End-State Assertions — see [[read]]
     Kernel only; full chains in read.md / discuss transcript. Append-only. -->

# ch-06 Q&A

## Q1. must-not-occur는 "쓰면 안 되는 *툴*을 썼을 때"인가?

아니다 — **"최종 world에 있으면 안 되는 *결과(상태)*"** 다. 채점은 path-agnostic이라 *어떤 툴을 불렀는지
안 봄*; 최종 world만 읽음. 그래서 "금지된 툴"은 채점 개념이 아님. NDA에서 모델은 docusign 툴을 *반드시* 써야 함
(유효 2명) — 금지된 건 그 *결과*(DO-NOT-contact한테 봉투 존재). 메커니즘 = 잠복 guard: 시작 때 trivially 참이라
free로 제외 → 깨면(나쁜 결과 생성) 실패 카운트. 잘하면 0 기여, 사고치면 터짐. 유일하게 툴이 제한되는 곳은
limited_zapier allowlist(setup 가용성)지 채점이 아님.

## Q2. 최종 world의 *변화(diff)* 를 채점하나, *절대 최종상태* 를 채점하나?

**절대 최종상태.** 각 assertion은 최종 world에 대한 술어(`field == value` 등). "바뀌었냐"가 아니라 "지금 맞냐".
초기 상태(deepcopy 베이스라인)는 **free-assertion 필터로만** 쓰임: 초기참+최종참 → 제외; 초기참+최종거짓(가드
깸) → 실패; 초기거짓 → 정상 채점. "변화량=점수"가 아님.

## Q3. 평가항목별 측정 방법은?

모든 assertion = `type` dict → 핸들러가 world 필드를 꺼내 bool 반환(에러는 잡아 카운트, 미중단). 5종:
**① 정확 `==`** (`*_field_equals`: collection→record_id→field→`==value`); **② 존재** (`*_exists`: 컬렉션 훑어
매칭); **③ 부재/부정** (`*_not_*`: `not exists`/없으면 통과 — must-not-occur); **④ substring 포함**
(`*_contains`, `body_contains`=리스트 → 전부 포함해야 AND, 대소문자 무시); **⑤ 정규화 일치**
(`*_phone_equals`: 숫자만 남기고 `==` — 유일 예외). regex·유사도·embedding 0. partial_credit=passed/total,
tcc=(pc==1.0). positive(floor)+negative(ceiling) 둘 다 있어야 lazy·shotgun 차단. no LLM-judge →
결정성·비용·재현성.

## Q4. 환경에 "기록"은 어떻게? tool call 기반으로 harness가 기록하나?

아니다 — **tool 함수가 주입받은 단일 world 객체를 직접 mutate** 하는 게 기록. 새 레코드면 `.append()`
(`gmail_send_email` → `world.gmail.messages.append(message)`, label_ids에 SENT), 기존이면 필드 set
(`salesforce_contact_update` → contact 찾아 `nda_status="Sent"`). world는 참조 공유라 다음 tool·최종 grader가
즉시 같은 변화를 봄 → **world 자체가 record, 채점용 별도 로그 없음.** harness는 world 주입만; 기록 주체는 tool 코드.
(read tool은 world 받아도 안 바꿈; trajectory 메시지 로그는 export/cost용이지 채점용 아님; api 모드도 route_*가
같은 impl로 같은 곳에 씀.)

## Q5. 이걸 dual-LLM(customer+sales) 채점에 적용하면? ("consent 후에만 consultation")

방향 옳음(= τ²-dual-control + AutomationBench end-state 하이브리드, ch-10). 세 함정:
1. **customer LLM ≠ 심판.** 동의를 customer LLM의 *판단*으로 채점하면 LLM-judge 비결정성 재유입. → consent를
   **구조화 플래그**로: customer tool이 플래그를 *세팅*, grader는 플래그를 *Python으로 읽음*(산문 아님).
2. **쓰기 권한 분리.** sales가 `consent=true`를 위조 못 하게 — consent는 customer 쪽 tool만, stage는 sales 쪽만
   (dual-control은 각 측에 자기 tool만). 단일 에이전트인 AutomationBench엔 없던 문제.
3. **"~할 때만"은 순서(path) 속성** → 순수 end-state(path-agnostic)는 순서를 못 봄. 두 길: (a) event-log+timestamp
   비교(= 경로 채점, 네 "두 타이밍" 직관), (b) **enforce-at-write**(advance tool이 consent 검사, 없으면 거부/
   `policy_violation=true`) → 위반이 최종상태에 박혀 보통 assertion으로 잡힘(더 깔끔, end-state 유지).

caveat: customer sim → 시나리오 stochastic → **pass^k** 필요; consent가 *제대로 elicit했을 때만* 나오도록
**persona calibration**(sim2real). ([[benchmark-comparison]], [[taubench]])
