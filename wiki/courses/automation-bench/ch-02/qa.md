<!-- qa for ch-02 — Architecture: WorldState, Episode Loop, Toolset Modes — see [[read]]
     Index of clarifying questions raised while reading. Kernel answers only;
     full chains live in read.md / discuss transcript. Append-only. -->

# ch-02 Q&A

## Q1. WorldState의 "44개"가 뭐야? 그냥 앱 개수인가?

앱 개수 맞다. `WorldState`는 Pydantic 루트 객체 하나이고, 필드가 **앱마다 하나씩**(`gmail: GmailState`,
`salesforce: SalesforceState`…). 각 필드 = 그 앱의 데이터 전체(타입 박힌 레코드 목록). **코드를 세면 47개**
(`schema/world.py`) — read.md의 "44"는 undercount. 논문도 47 → see [[automationbench-results]]. 앱끼리
**스키마를 공유 안 함**(Salesforce `Contact` ≠ Gmail `Message`) → cross-app이 어려운 근본 이유.

## Q2. `world`가 담는 정보가 뭐야?

`meta`(schema_version, **current_time** = task의 "오늘") + 47개 앱 상태. 각 앱은 컬렉션 여러 개
(예: `SalesforceState`는 contacts/accounts/opportunities… 16개). **모양은 고정, 내용은 task마다 sparse** —
기본 전부 빈 `[]`이고 `initial_state`가 채운 컬렉션만 데이터. 객체 **하나**가 입력 + working memory + 채점대상
세 역할. 채점은 `world.salesforce.contacts[id].nda_status == "Sent"` 식으로 같은 객체를 읽음.

## Q3. world injection trick이 뭐야?

tool은 첫 인자로 `world`를 받지만 **모델 schema에선 숨김**(`args_to_skip=["world"]`, `tool_wrapper.py`).
모델은 `(contact_id, field, value)`만 봄. 실행 직전 harness가 `state["world"]`를 주입(`update_tool_args`).
**한 함수가 두 얼굴**: 실제 world 변경 구현(world 필요) + 모델용 깔끔한 tool(world 숨김). world는 참조로 넘어가
변경이 다음 tool에 즉시 보임 → multi-hop·determinism·싼 채점.

## Q4. meta-message compression이 뭐야?

zapier 모드에서 `search_tools` 결과(전체 설명+schema)는 토큰을 많이 먹음. `execute_tool`로 행동에 옮기고 나면
**이전 턴**의 검색 결과를 짧은 이름 목록(`[Previously found: ...]`)으로 다시 씀(`runner.py:240`). **현재 턴은
보존**(같은 턴 검색+실행 시 arg 이름 hallucination 방지). 최대 25턴 episode에서 context를 가볍게 유지.

## Q5. `extra="forbid"`가 load-bearing이라는 게 무슨 말?

load-bearing = 내력벽(빼면 무너짐). Pydantic config: 선언 안 된 키가 dict에 있으면 **즉시 ValidationError**
(`extra="ignore"`였다면 조용히 버림). `WorldState(**initial_state)` 구성 시점(`setup_state`)에 터져서, 손으로
쓴 task의 **오타를 시작 시점에 시끄럽게 적발**. 망가진 seeded 데이터로 조용히 채점하는 사고를 막음 → determinism/
정직한 채점의 맨 밑바닥 토대. 장식 아님.

## Q6. `forbid`로 모델이 스스로 틀린 부분을 고치나?

아니다. `forbid`는 **고치는 게 아니라 거부하고 멈춤**(ValidationError). 고치는 건 **사람(task 작성자)**. 그리고
이건 harness 안(`setup_state`, agent 루프 *전*)에서 일어나 — 평가받는 **LLM은 보지도 못함**. (Pydantic의 'model'
≠ LLM.) 자동 수정 자체가 또 하나의 silent 동작이라 일부러 strict를 택함.

## Q7. 그럼 실행 중 에러는? 모델이 tool-result로 받아 고쳐 재시도하나?

맞다 — 그건 **다른 층**(런타임). `execute_tool`의 `"Unknown tool: ... Use search_tools..."`, `api_fetch`의
404/400이 **tool-result로 돌아와** 모델이 인자 고쳐 재시도. 단 이 루프는 **에러가 *나는* 실수**만 구제. AutomationBench
실패의 72–91%(false confidence)는 **유효하지만 틀린 호출** — 정상 결과가 와서 재시도할 거리가 없음. 의미(semantic)
오류엔 눈멂. ([[automationbench-results]])

## Q8. limited_zapier에서 SOTA 성능은?

≈ **14.3%**(Gemini 3.1 Pro). 순서 api < zapier < limited (Gemini 9.6/12.8/14.3; Haiku 1.5/2.0/3.8).
discovery를 통째로 빼도 **+1.5~1.8pt뿐**이고 tool 다 줘도 ~14% → **벽은 discovery가 아니라 orchestration**
(policy·cross-app·함정). 출처/표 → [[automationbench-results]].
