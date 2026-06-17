<!-- qa for ch-01 — Architecture Is the Expensive-to-Reverse Decisions
     study index of clarifying questions raised during the Read phase.
     Kernel answers only; full reasoning lives in [[read]]. Append-only across cycles. -->

# ch-01 — Reading Q&A

Companion to [[read]]. One entry per clarifying question; kernel of the answer only.

---

## Q1. framing과 pattern의 차이가 뭐야?

**Kernel.** 층위가 다르다. **framing** = 판단 렌즈(척추 "뒤집기 비싼 결정" + First Law "모든 건 trade-off") — context-독립, 단 하나, *어떻게 평가할지*를 답함. **pattern** = 구체적 구조 선택(monolith/microservices/hexagonal/saga/CQRS…) — context-의존, 다수, *무엇을 지을지*를 답함.

관계: framing은 채점 기준표, pattern은 답안. §2.2의 *"X를 싸게 유지하는 대신 Y를 비싸게 만든다"* 문장이 곧 framing을 pattern에 적용한 결과. framing 없이 pattern만 내세우면 그게 챕터가 죽이려는 "best practice 신화" — L11: *"best practice는 누군가 가격표를 깜빡한 trade-off일 뿐"*, Y를 못 대면 "이해한 게 아니라 팔린 것"(L61).

비유: framing = 과학적 방법(정답이 아니라 검증법을 줌), pattern = 개별 가설. 그래서 framing은 하나, pattern은 계속 늘어난다. (see read.md §1–§2, L11/L27/L61/L84)

---

## Q2. architecture가 뭔지 감각적으론 알겠는데 아리송하다 — 이게 ch-01의 목표인가?

**Kernel.** 목표는 명사 정의가 ❌. ch-01은 **적용 가능한 테스트("이 결정, 뒤집기 얼마나 비싸지?")** 와 **그 선을 움직이는 법(Fowler 수정안)** 을 설치하는 게 목표. 아리송함의 원인 둘: (a) 옛 직관(architecture = 기술스택/다이어그램/마이크로서비스) 잔재 — 챕터가 명시적으로 거부함(L23); (b) architecture는 box가 아니라 **gradient**라서 경계가 본래 흐릿함(§1.1에 "borderline" 행). 게다가 §1.2 Fowler — *"좋은 architect는 변경을 쉽게 만들어 architecture를 줄인다"* — 라서 경계선 자체가 **움직이는 선**(인터페이스 뒤로 숨기면 "마이그레이션"→"리팩터"가 되어 더 이상 architecture 아님). 치료법: 정의를 더 파는 게 아니라 reversibility 테스트를 굴려보기. (see read.md §1.1/§1.2)

---

## Q3. 'architecture = applying best practices'가 무슨 말? 'best practice'가 "최고의 연습"이 맞아?

**Kernel.** 번역 교정: `practice`는 "연습"이 아니라 **"(일하는) 방식·관행"**. best practice = **"업계가 검증한 표준 방식 / 모범 사례"** (예: 비밀번호 해싱, 머지 전 코드 리뷰) — context 안 따져도 대체로 옳다고 여겨지는 레시피.

신화(§2.3, L82–84): 아키텍처를 *"마이크로서비스/DDD/REST 같은 정답 레시피를 외워다 적용하는 일"* 로 보는 관점. 챕터는 First Law로 이를 기각 — architecture엔 context-독립 best practice가 없다(모든 게 trade-off); best practice를 집으면 *이 시스템에 맞게 가격 매기는* 유일한 단계를 건너뛴 cargo-culting. Hard Parts: 어려운 결정엔 *"various compromises 중 고르도록 강요하는 best practice가 없다."* L11: *"best practice는 누군가 가격표 붙이는 걸 깜빡한 trade-off일 뿐."* → [[Q1]]의 "framing 없이 pattern만 쓰기"와 같은 실수. (see read.md §2.3, L11/L82–84)

---

## Q4. 그 trade-off에서 *무엇을* 저울질하는 거야?

**Kernel.** 핵심 교정: **"좋은 것 vs 나쁜 것"이 아니라 "좋은 것 A vs 좋은 것 B"** — 양쪽 접시 모두 네가 원하는 속성. 그래서 정답이 없음.

저울 위 두 가지: **(1)** 서로 당기는 architecture characteristics("-ilities") 한 쌍 (§3.1: deployability↔simplicity, 독립확장↔공유트랜잭션, auditability↔복잡도). **(2)** §2.2의 압축형 *"X를 싸게 유지하는 대신 Y를 비싸게 만든다"* — microservices(X=독립배포 / Y=분산트랜잭션·네트워크실패·운영세금), clean core(X=vendor교체 / Y=indirection 오버헤드).

저울 **눈금=공통화폐는 "변경/뒤집기 비용"**. 못 가지는 이유: cheap budget이 유한 → X를 싸게 만드는 구조가 Y를 비싸게 함(이불이 짧아 머리 덮으면 발 나옴). 어느 쪽이 무거운지는 [[Q1]]대로 *이 시스템의 critical few characteristic*(§3)이 결정. (see read.md §2.2/§3.1, §7 Lina 예)

---

## Q5. architecture characteristics("-ilities")도 다른 챕터에서 더 배워?

**Kernel.** 전용 챕터는 ❌ — ch-01 §3이 개념의 본거지. 대신 코스 전체의 **"화폐"로 계속 흐름**(따로 공부할 토픽이 아니라 매 장이 써대는 통화). 일생: **도출**(ch-01 §3, critical few 뽑기) → **선택을 암묵적으로 지배**(ch-02~07 모든 topology/pattern = 어떤 -ility 살리고 죽이냐) → **모니터링**(ch-08, observability를 critical characteristics에 묶음, 명시적 재등장) → **강제·보호**(ch-09, fitness function = "architectural characteristic의 객관적 무결성 평가", 빌드 깨는 자동검사).

ch-01 §6 loop / §4.3 "ADR로 기록→fitness function으로 보호" 줄기와 정확히 맞물림: ch-01에서 *고르고*(§3.3: 다른 문 닫는 베팅) ch-09에서 *지킴*. §3 읽을 때 각 -ility를 "나중에 fitness function으로 지켜야 할 것"으로 보면 ch-09에서 회수됨. (see outline ch-08/ch-09 concepts, read.md §3.3/§4.3/§6)

---

## Q6. characteristic("-ility") 예시를 많이 보여줘

**Kernel.** 이름이 다 -ility로 끝나진 않음(security/performance). [[richards-ford-fundamentals]] 3분류:
- **Operational**(런타임): performance, scalability, elasticity(스파이크 흡수), availability, reliability, fault tolerance/resilience(→ch-08), recoverability, robustness
- **Structural**(코드품질): modularity, modifiability/maintainability(=척추 그 자체), extensibility, testability(hexagonal 동기), deployability(microservices를 미는 것), configurability, reusability, supportability
- **Cross-cutting**: security, observability(→ch-08), auditability(event sourcing 매력 ch-07), interoperability, portability, usability/accessibility, compliance/privacy
- **숨은 셋**(암묵): simplicity, cost, feasibility — 늘 저울 반대편

**핵심은 충돌 쌍**(§3 본론): performance↔security, performance↔modifiability, scalability↔simplicity/consistency, deployability↔simplicity, availability↔consistency(CAP류), security↔usability. 외우는 게 아니라 어느 쌍이 당기는지가 스킬. Lina critical few 후보: modifiability(vendor 교체), resilience(SaaS 장애 격리), auditability(영업/규제), 지금은 simplicity>scalability. (see read.md §3)
