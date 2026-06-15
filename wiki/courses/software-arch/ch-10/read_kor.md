<!-- chapter: ch-10
     track: capstone
     kind: lab
     title: Lab: Architecting the Production Sales Agent (ADR + C4)
     deps: [[ch-09]], [[ch-07]]
     sources: [[ddd-bounded-context]], [[decompose-by-business-capability]], [[fowler-monolith-first]], [[richards-ford-hard-parts]], [[distributed-monolith]], [[helland-data-outside-inside]], [[transactional-outbox]], [[consumer-driven-contracts]], [[fielding-rest]], [[nygard-release-it]], [[c4-model]], [[richards-ford-fundamentals]], [[richardson-saga]], [[ddd-aggregates-tactical]], [[martin-clean-arch]], [[cockburn-hexagonal]], [[conway-team-topologies]], [[young-cqrs-es]], [[fowler-microservices]], [[newman-building-microservices]], [[martin-strangler-fig]]
-->

# 10장 — Lab: Architecting the Production Sales Agent (ADR + C4)

> **핵심 통찰.** 이 lab은 새로운 아이디어가 아니다 — 이것은 코스 전체를, 순서대로, 하나의 실제 시스템에 대해 한 번 실행하는 것이다. 아홉 개의 챕터가 toolkit(도구 모음)을 주었다; 남은 유일한 학습 과제는 architect가 그것을 꺼내드는 *sequence*(순서)와, 그 모든 꺼냄을 하나의 bet(베팅)으로서 가격을 매기는 규율이다. 그 sequence는 고정되어 있고 협상 불가능하다: 먼저 boundary(경계)를 찾아라(이것은 되돌리기 비싼 결정이므로, *deployment*(배포) 약속은 미루되 *modeling*(모델링) 약속은 일찍 한다), 각 boundary의 내부를 구조화하라, fashion(유행)이 아니라 trade-off(트레이드오프) 분석으로 topology(토폴로지)를 선택하라, contract(계약)를 설계하라, boundary를 가로지르는 비용이 무엇인지 결정하라, 힘(force)이 강제할 때에만 power tool(고급 도구)을 꺼내라, 그런 다음 edge(가장자리)를 안전하게 실패하도록 그리고 전체를 수정 가능하도록 엔지니어링하라. 산출물은 두 개의 artifact — 하나의 ADR set과 하나의 C4 스케치 — 이며, 당신이 이 코스를 배웠는지 가르는 시험은 C4 Container diagram의 모든 box가 그 뒤에 한 줄짜리 trade-off를 가지고 있는지다.

> 💡 **쉬운 설명:** 이 챕터는 capstone(졸업 작품) lab이다. 1~9장에서 배운 개별 도구들(bounded context, hexagonal, modular monolith, saga, outbox, circuit breaker 등)을 한 번에 다 꺼내서, Lina TMR이라는 실제 sales agent를 처음부터 설계해 보는 실습이다. 핵심 메시지는 단 하나다: "모든 설계 결정에는 비용이 따른다. 비용을 말할 수 없다면 그것은 결정한 게 아니라 베낀 것이다." 도구를 아는 것보다, 어떤 순서로 꺼내고 그 대가를 명확히 말하는 것이 이 코스의 진짜 목표다.

> **가이드라인.** 아래 여덟 step을 Lina TMR에 대해 순서대로 실행하라. 매 step에서, 세 가지를 하되 오직 세 가지만 하라: (1) pattern의 이름을 말하라, (2) bet을 *"이것은 X를 바꾸기 싸게 유지하는 대신 Y를 바꾸기 비싸게 만든다"* 형태로 진술하라, 그리고 (3) 그것이 architecturally significant(아키텍처적으로 유의미)하다면 ADR에 기록하라. 모든 topology 결정을 modular monolith ([[fowler-monolith-first]]) 쪽으로 default(기본값)로 두고, 모든 "best practice" 프레이밍을 거부하라 — best practice는 존재하지 않는다 ([[richards-ford-hard-parts]]). 낯선 사람이 당신의 ADR과 C4 Container diagram을 읽고 당신이 *무엇을* 결정했는지뿐 아니라 *무엇에 베팅하고 있었는지*까지 재구성할 수 있을 때 끝난 것이다.

---

## 0. The brief

**Lina TMR**은 production sales agent다: 여러 외부 SaaS tool API(CRM, email, calendar, e-signature, enrichment, messaging, billing)를 자율적으로 다루며 sales team을 대신해 deal을 pipeline을 따라 이동시키는 LLM agent다. 이것은 event(새 lead, inbound reply, stage 변경)와 scheduled sweep(예약된 일괄 스캔)에 의해 trigger되며, 행동한다 — mail을 보내고, meeting을 예약하고, CRM record를 업데이트하고, contract 초안을 작성한다 — bounded human oversight(제한된 인간 감독) 아래에서. 학습자는 이전에 *agent benchmark*(automation-bench 코스)를 만들었다; 이 lab은 그것의 역방향이자 보상이다: **agent를 평가하기를 멈추고 하나를 설계하라.**

> 💡 **쉬운 설명:** automation-bench 코스에서는 "agent가 일을 제대로 하는지 채점하는" 시스템을 만들었다. 이번 lab은 정반대다 — 채점당하는 쪽, 즉 실제로 일하는 agent 자체(Lina TMR)를 설계한다. Lina는 사람 영업사원을 대신해 CRM 업데이트, 이메일 발송, 미팅 예약 등을 자동으로 수행하는 LLM 기반 자동화 agent라고 생각하면 된다.

이것이 capstone에 알맞은 시스템인 이유는, 이 코스가 다루는 바로 그 결정들로 포화되어 있기 때문이다:

- 이것은 **boundary-rich**(경계가 풍부)하다: "lead", "conversation", "meeting", "contact"는 시스템의 서로 다른 부분에서 서로 다른 것을 의미한다 — [[ddd-bounded-context]]의 polysemy(다의성) 신호다.
- 이것은 **integration-heavy**(통합이 많음)하다: 모든 외부 SaaS 호출은, 정의상, [[helland-data-outside-inside|outside data]]다 — 버전이 매겨진, 오래되었을 수도 있는 snapshot이지, 결코 authoritative(권위 있는) live state가 아니다.
- 이것은 **failure-exposed**(실패에 노출됨)다: 수십 개의 third-party vendor 각각이 느려지거나 다운될 수 있고, 이들이 하나의 agent loop로 fan into(수렴)한다 ([[nygard-release-it]]).
- 이것은 **long-lived**(오래 사는)다: domain policy("우리는 lead를 어떻게 qualify하는가", "언제 escalate하는가")는 이번 분기에 현재 사용 중인 어떤 LLM API, vector DB, framework보다도 오래 살아남아야 한다 ([[martin-clean-arch]]).

산출물은, 모든 step이 이를 위해 봉사하도록 미리 정의하건대: **ADR-style design memo**(Title/Status/Context/Decision/Consequences record의 집합)와, distributed monolith(분산 모놀리스)가 가시화될 **C4 Context + Container 스케치**다. 템플릿은 §10–11에 있다; lab의 나머지가 이를 채운다.

> 💡 **쉬운 설명:** ADR은 "Architecture Decision Record"의 약자로, 하나의 중요한 아키텍처 결정을 Title/Status/Context/Decision/Consequences 다섯 칸으로 짧게 기록하는 메모다. C4는 시스템을 4단계(Context → Container → Component → Code) 줌으로 그리는 다이어그램 표기법인데, 여기서는 가장 바깥 2단계(Context, Container)만 그린다. 이 둘이 lab의 최종 제출물이다.

### 0.1 The myth this lab exists to kill

[[COLLECTION-PLAN]]의 doc-vs-reality 표는 열 개의 myth(통념)를, 주제당 하나씩 나열한다. capstone의 임무는 그 모든 것 *위에* 앉아 있는 myth를 죽이는 것이다 — spine(코스의 척추)이 첫 챕터부터 공격하는 meta-myth다:

> "Microservices / DDD / CQRS / REST are best practices you should adopt."

그것들은 best practice가 아니다. *Software Architecture: The Hard Parts*는 정반대 입장 위에 세워져 있다: 그것은 "various compromise 중에서 선택하도록 강제하는 best practice가 없는 어려운 문제들"을 다루며, "distributed architecture와 관련된 trade-off에 대해 비판적으로 사고하는 법"을 가르친다 ([[richards-ford-hard-parts]], book — thesis extracted). First Law(제1법칙)는 같은 것을 긍정형으로 진술한다: "Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet" (Richards & Ford, [[richards-ford-fundamentals]], book — thesis extracted, quoted as commonly published). 그래서 이 lab의 pass/fail 기준은 잔혹하고 단순하다: **선택의 비용을 말할 수 없다면, 당신은 선택을 한 것이 아니다 — 복사한 것이다.**

> 💡 **쉬운 설명:** 이 코스 전체를 관통하는 단 하나의 적은 "마이크로서비스/DDD/CQRS/REST는 무조건 도입해야 할 모범 사례"라는 믿음이다. 진짜 아키텍처는 모범 사례를 따르는 게 아니라 trade-off를 의식적으로 고르는 일이다. 그래서 합격 기준이 "비용을 말할 수 있는가"인 것이다 — 비용을 모르고 채택했다면 그건 사고가 아니라 모방이다.

---

## 1. Step 1 — Find the bounded contexts (the [[ch-02]] move)

boundary가 먼저다. 왜냐하면 그것이 시스템에서 가장 되돌리기 비싼 결정이고, 약속할 가치가 있는 유일한 종류의 경계는 가장 천천히 변하는 구조 위에 그어진 것이기 때문이다. Evans의 thesis(논지)를, Fowler가 인용한 대로:

> "Total unification of the domain model for a large system will not be feasible or cost-effective." — Eric Evans, via Fowler ([[ddd-bounded-context]])

어디를 잘라야 할지에 대한 operational signal(실무적 신호)은 linguistic(언어적)이다:

> "You need a different model when the language changes." — Fowler ([[ddd-bounded-context]])

그러니 Lina의 domain을 걸으며 한 단어가 같은 것을 의미하기를 멈추는 지점을 찾아라. polysemy(다의성)는 실재하고 즉각적이다:

| Word | In one context it means… | In another it means… | ⇒ boundary signal |
|------|---------------------------|----------------------|-------------------|
| **Lead / Contact** | stage, score, owner를 가진 pipeline entity | vendor ID와 sync status를 가진 CRM record | Pipeline vs CRM-Sync |
| **Conversation** | agent가 추론하는 inbound/outbound 메시지의 thread | provider-specific email/SMS payload의 sequence | Conversation vs (the messaging adapter) |
| **Meeting** | 참석자와 agenda를 가진 commitment | provider event-ID와 timezone을 가진 calendar event | Scheduling vs (the calendar adapter) |
| **Deal won** | handoff를 trigger하는 pipeline outcome | CRM `stage_name` field 업데이트 | Pipeline vs CRM-Sync |

Lina의 candidate bounded context들, 각각은 하나의 [[ddd-bounded-context|ubiquitous language]](보편 언어)를 가진 unified model이다:

1. **Lead / Pipeline** — lead, scoring, stage, qualification *policy*. 가장 천천히 변하고, 가치가 가장 높은 core.
2. **Conversation** — message thread, agent turn, reasoning loop, drafted-vs-sent state.
3. **Scheduling** — meeting intent, availability, booking commitment.
4. **CRM-Sync** — Lina의 inside model과 외부 CRM의 outside model 사이의 reconciliation(조정/대사).

**The bet (price it).** boundary를 *language/domain*(가장 천천히 변하는 구조, [[decompose-by-business-capability]]: "business capability가 상대적으로 안정적이므로 stable architecture") 위에 그리는 것은 model을 **각 context 내부에서 진화시키기 싸게** 유지하는 대신 **cross-context invariant(컨텍스트를 가로지르는 불변식)를 비싸게** 만든다 — 예컨대 Pipeline과 Scheduling을 가로질러 참이어야 하는 무엇이든 이제는 공유 객체가 아니라 명시적 translation과 eventual consistency(궁극적 일관성)(Step 5)를 필요로 한다. 여기서 *잘못된* boundary는 가용한 가장 비싼 실수다: "매일 대가를 치르는 model rot(모델 부패)." 이것이 *modeling*은 지금 약속하되 *deployment* 약속은 Step 3으로 미루는 이유다.

> 💡 **쉬운 설명:** bounded context는 "한 단어가 일관된 의미를 가지는 영역"이다. Lina에서 "Lead"는 Pipeline에서는 점수와 단계를 가진 영업 대상이지만, CRM-Sync에서는 외부 CRM의 레코드(벤더 ID, 동기화 상태)다. 같은 단어가 의미가 갈리는 그 지점이 곧 경계를 그어야 할 신호다. 경계를 비즈니스 언어(가장 안 변하는 것) 기준으로 그으면 내부 진화는 쉬워지지만, 경계를 넘나드는 일관성은 비싸진다 — 이게 바로 가격을 매긴 bet이다.

**Myth killed (ch-02's):** "DDD requires microservices." 거짓 — DDD는 modeling discipline(모델링 규율)이다; 이 네 context는 하나의 deployable 안의 네 *module*일 수 있다. 어떤 DDD source도 context를 topology에 묶지 않는다 ([[ddd-bounded-context]], [[COLLECTION-PLAN]]). 그것이 바로 Step 3이 통과하는 문이다.

**Context mapping.** Lina의 context들과 *바깥* 세계 사이에서, 하중을 지탱하는(load-bearing) 관계는 **Anticorruption Layer**(부패 방지 계층, ACL)다: "다른 context의 model을 번역하여 그것이 당신의 것으로 새어 들어오지 못하게 하라" ([[ddd-bounded-context]], Vernon의 *DDD Distilled*에서). 모든 SaaS vendor의 schema는 foreign model(외래 모델)이다; ACL은 Salesforce의 `Opportunity`나 HubSpot의 `Deal`이 Lina의 vocabulary가 되는 것을 당신이 막는 지점이다.

> 💡 **쉬운 설명:** anti-corruption layer(ACL)는 외부 시스템의 용어와 구조가 내 도메인 모델로 그대로 침투하는 것을 막는 번역 막(膜)이다. Salesforce가 "Opportunity"라 부르고 HubSpot이 "Deal"이라 불러도, ACL을 거치면 Lina 내부에서는 일관되게 자기 용어("Lead")로 변환된다. 이게 없으면 벤더가 바뀔 때마다 핵심 도메인 코드까지 흔들린다.

---

## 2. Step 2 — Structure the inside of each boundary (the [[ch-03]] move)

각 context 내부에서, 하나의 규칙이 모든 것을 지배한다. 한 번 진술하고, 네 번 적용하라:

> "Source code dependencies can only point inwards." — Robert C. Martin ([[martin-clean-arch]])

같은 아이디어를, testability(테스트 가능성) 목표로 표현하면:

> "Allow an application to equally be driven by users, programs, automated test or batch scripts, and to be developed and tested in isolation from its eventual run-time devices and databases." — Alistair Cockburn ([[cockburn-hexagonal]], Intent quote; the canonical alistair.cockburn.us URL had an expired TLS cert on 2026-06-15, so this is corroborated via the alistaircockburn.com mirror — cite the mirror, not a clean fetch).

Clean = Hexagonal = Onion: technology-free core, 안쪽을 가리키는 dependency, core가 소유한 interface(port)를 통해 edge에 있는 I/O ([[martin-clean-arch]], [[cockburn-hexagonal]]). Lina에게 이것은 선택적인 마감 손질이 아니다 — vendor churn(벤더 교체) 전반에 걸쳐 domain policy를 살아 있게 유지하는 bet이다.

> 💡 **쉬운 설명:** Clean Architecture, Hexagonal Architecture, Onion Architecture는 이름만 다를 뿐 같은 원리다 — "의존성은 항상 안쪽(핵심 도메인)을 향한다." 도메인 핵심은 어떤 외부 기술(DB, LLM API, 프레임워크)도 모르고, 외부와의 입출력은 core가 정의한 interface(port)를 통해서만 edge에서 일어난다. 덕분에 벤더나 기술이 바뀌어도 핵심 도메인 정책은 손대지 않는다.

**Ports for Lina** (core가 자기 자신의 언어로 정의하는 interface들):

- `ForQualifyingLeads` (primary/driving — event나 sweep이 core를 driving한다)
- `ForStoringPipeline` (secondary/driven — core가 persistence로 호출해 나간다)
- `ForSendingMessages`, `ForBookingMeetings`, `ForSyncingCRM`, `ForReasoning` (secondary/driven — 모든 SaaS vendor와 LLM 자체가 그 하나 뒤에 앉는다)

LLM provider는 *`ForReasoning` 뒤의 driven adapter*다. 이것은 Lina에서 dependency rule의 가장 중요한 단일 적용이다: 이번 분기에 당신이 호출하는 model(Claude, open weights model, fine-tune)은 "당신이 호출하는 tool이지, 앱 전체를 상속받는 base class가 아니다" ([[martin-clean-arch]]). 그것을 교체하는 일은 adapter swap이어야 하고, 결코 core surgery(핵심부 수술)여서는 안 된다.

> 💡 **쉬운 설명:** port는 core가 정의한 추상 인터페이스이고, adapter는 그 인터페이스의 구체 구현이다. driving(primary) port는 외부가 core를 깨워 일을 시키는 입구(예: 이벤트가 `ForQualifyingLeads`를 호출), driven(secondary) port는 core가 바깥 세계로 나가는 출구(예: core가 `ForSendingMessages`로 메일을 보냄)다. 특히 LLM은 `ForReasoning` port 뒤의 한 adapter일 뿐이라서, Claude를 다른 모델로 바꿔도 adapter만 갈아끼우면 되고 도메인 코드는 그대로다.

### 2.1 Tactical DDD — aggregates as in-process consistency boundaries

core 내부에서, **aggregate(애그리거트)는 즉각적 transactional consistency(트랜잭션 일관성)의 단위**다 ([[ddd-aggregates-tactical]]). Vernon의 네 규칙(book — thesis extracted)을, Lina에 적용하면:

1. **Model true invariants in consistency boundaries** — 예컨대 `Lead`와 그것의 `Score`는 즉시 함께 일관적이어야 한다; `Lead`와 booked `Meeting`은 그럴 필요가 없다.
2. **Design small aggregates** — `Lead`는 aggregate다; 전체 conversation history를 그 안에 접어 넣지 마라.
3. **Reference other aggregates by identity** — `Meeting`은 `Lead` 객체가 아니라 `LeadId`를 보유한다.
4. **Use eventual consistency outside the boundary** — won deal이 CRM-Sync를 trigger해야 할 때, domain event(`DealWon`)를 publish하라; 같은 transaction에서 손을 뻗어 CRM-Sync의 데이터를 쓰지 마라.

Rule 4는 lab의 나머지 전체에 하중을 지탱한다: 그것은 "distributed saga의 in-process 씨앗"이다 ([[ddd-aggregates-tactical]]). aggregate boundary를 *지금*, modular monolith 안에서 올바르게 잡으면, 미래의 service split이 올바른 consistency boundary를 공짜로 상속한다.

> 💡 **쉬운 설명:** aggregate는 "반드시 한 트랜잭션 안에서 함께 일관되어야 하는 객체 묶음"이다. Lead와 그 Score는 즉시 일관돼야 하니 한 aggregate지만, Lead와 Meeting은 약간 늦게 맞아도 되니 별개 aggregate다. 핵심은 Rule 4 — 경계 밖과는 같은 트랜잭션으로 묶지 말고 event를 발행해서 나중에(eventually) 맞춘다. 이걸 monolith 안에서 미리 잘 잡아두면, 훗날 서비스로 쪼갤 때 그 경계가 그대로 분산 saga의 경계가 된다.

**The bet (price it).** Port + DTO + mapper + small aggregate는 **indirection(간접 참조)과 boilerplate(상용구 코드)**다 — "small CRUD service에는 overkill" ([[martin-clean-arch]]). 그것들은 vendor와 framework churn 전반에 걸쳐 **비싼 부분(domain policy)을 보존하기 싸게** 유지하는 대신 **가장 단순한 CRUD path를 더 장황하게** 만든다. Lina의 long-lived core에게 이것은 옳은 bet이다; throwaway script에게는 낭비일 것이다. default가 아니라 bet으로서 이름 붙여라.

---

## 3. Step 3 — Choose the topology: modular-monolith-first + the quantum (the [[ch-04]] move)

이것이 lab 전체의 spine(척추)이므로, 가장 신중하게 가격을 매겨라.

**Default: a modular monolith.** Fowler의 경험적 주장은 무뚝뚝하다:

> "Almost all the successful microservice stories have started with a monolith that got too big and was broken up." — Fowler ([[fowler-monolith-first]])

> "Almost all the cases where I've heard of a system that was built as a microservice system from scratch, it has ended up in serious trouble." — Fowler ([[fowler-monolith-first]])

그 이유는 boundary uncertainty(경계 불확실성)이고, 그것은 Lina가 있는 바로 그곳 — 진화하는 domain의 초기 — 에서 가장 세게 문다:

> "Even experienced architects working in familiar domains have great difficulty getting boundaries right at the beginning." — Fowler ([[fowler-monolith-first]])

그래서 Step 1의 네 context는 하나의 deployable 안의 네 **module**이 된다: module당 하나의 schema/namespace, 그들 사이의 in-process interface, 어떤 module도 다른 것의 table로 손 뻗지 않음 ([[fowler-monolith-first]]). 권고는, 그대로 옮기면:

> "Start a new application as a monolith initially, even if you think it's likely that it will benefit from a microservices architecture later on." — Fowler ([[fowler-monolith-first]])

**The central bet of this chapter, stated explicitly.** modular monolith는 당신의 **boundary를 다시 긋기 싸게** 유지한다 — monolith 안에서 나쁜 boundary는 *refactor*(리팩터)지만, service를 가로지르면 그것은 *migration*(마이그레이션)이다 ([[fowler-monolith-first]]) — 그 대신 **module별 deployment, scaling, tech-choice 독립성을 비싸게** 만든다(Pipeline을 재배포하지 않고는 Conversation을 deploy하거나 scale할 수 없다; CRUD pipeline과 다른 hardware에서 LLM-heavy reasoning loop를 돌리려면 split해야 한다). 당신은, Lina에게 오늘, *boundary를 잘못 잡을 확률과 비용*이 *어느 한 module의 독립 배포가 주는 가치*를 초과한다는 데 베팅하는 것이다. 그것은 시작 시점에서 거의 확실히 참이며 나중에는 참이 아니게 될 수 있다 — 그것이 Step 8이 bet을 수정 가능하게 유지하는 이유다.

> 💡 **쉬운 설명:** modular monolith는 "하나로 배포되지만 내부는 모듈로 명확히 나뉜" 구조다. 마이크로서비스로 처음부터 쪼개면, 경계를 잘못 잡았을 때 그걸 고치는 일이 코드 리팩터가 아니라 서비스 간 데이터 마이그레이션이 되어 엄청 비싸진다. 초기에는 경계가 틀릴 확률이 높으니, "일단 한 덩어리로, 단 내부는 모듈로" 시작하는 게 기본값이다. 독립 배포의 편리함은 포기하지만, 경계를 싸게 고칠 수 있는 자유를 얻는 거래다.

### 3.1 The architecture quantum — the boundary test

*무엇이든* 추출하기 전에, quantum으로 그것을 테스트하라. **architecture quantum(아키텍처 퀀텀)**은 "독립적으로 배포 가능하고, 높은 functional cohesion(기능적 응집)을 가지며… 자신의 데이터를 포함하는 가장 작은 단위"다 ([[richards-ford-hard-parts]], book — thesis extracted). 탐지기: **service가 아니라 quantum을 세어라.** database를 공유하는 두 "service"는 *하나의* quantum이며 — [[distributed-monolith]]다.

> 💡 **쉬운 설명:** architecture quantum은 "혼자서 배포되고, 응집도 높고, 자기 데이터를 가진 가장 작은 단위"다. 서비스를 둘로 나눴어도 둘이 같은 DB를 쓴다면, 사실 quantum은 하나뿐이다 — 즉 겉보기만 분산이고 실제로는 distributed monolith다. "서비스 개수"가 아니라 "quantum 개수"를 세는 것이 진짜 분리 여부를 가리는 시험이다.

### 3.2 Disintegrators vs integrators — size is an output, not a rule

여기서 죽이는 myth (ch-04's): "rule of thumb(경험칙)으로 service size를 골라라." 아니다 — force(힘)를 열거하라 ([[richards-ford-hard-parts]]):

| For the Conversation/reasoning module, should we extract it? | |
|---|---|
| **Disintegrators (split smaller)** | divergent scalability(LLM 호출은 CRUD와 달리 느리고 + bursty하다); fault isolation(막힌 reasoning loop가 pipeline sweep을 멈추게 해선 안 된다); distinct code-volatility(prompt/model logic은 매주 바뀐다); 가능하다면 별도 team ownership |
| **Integrators (keep together)** | `Lead`와 `Pipeline` state에 대한 tight data dependency; reason→act→update 사이의 chatty workflow; 함께 바뀌는 shared domain type; event를 쓰면 seam을 *가로지르는* DB transaction이 불필요 |

**disintegrator가 integrator를 능가할 때에만** split한다 ([[richards-ford-hard-parts]]). Lina에게, *reasoning/Conversation* module이 가장 강한 extraction candidate다(scalability + fault-isolation + volatility가 모두 split 쪽으로 당긴다); CRM-Sync가 두 번째다(I/O-bound이고 독립적으로 실패한다). Pipeline은 *마지막으로* 추출할 것이다 — 그것이 응집적 core다. **Decision for the memo:** 네 개 모두 module로 시작하라; Conversation을 첫 extraction candidate로 flag하되 그것의 scaling/fault-isolation force가 실제로 물 *때* 추출하고, Strangler Fig를 통해(Step 8) 추출하라.

> 💡 **쉬운 설명:** "서비스를 얼마나 잘게 쪼갤까"는 규칙으로 정하는 게 아니라, 쪼개려는 힘(disintegrator)과 합치려는 힘(integrator)을 양쪽에 적어놓고 무게를 재서 정한다. Lina에서는 Conversation 모듈이 쪼갤 힘이 가장 세다(LLM 호출은 느리고 폭발적이며, 막혀도 다른 일을 멈추면 안 되고, 코드도 자주 바뀐다). 단 "지금" 쪼개는 게 아니라 그 힘들이 실제로 아플 때 쪼갠다 — 크기는 규칙이 아니라 힘 분석의 결과물이다.

### 3.3 The distributed-monolith trap and Conway

Myth killed: "split하면 loose coupling을 공짜로 얻는다." 추출한 module들이 함께 deploy되어야 하거나 DB를 공유한다면, 당신은 "독립성 없이 distributed system의 모든 고통"을 만든 것이다 ([[distributed-monolith]]). litmus test(리트머스 시험)는 independent deployability(독립 배포 가능성)다 ([[newman-building-microservices]]: "the single most important principle," book — thesis extracted).

그리고 socio-technical(사회-기술적) 제약: "Any organization that designs a system… will produce a design whose structure is a copy of the organization's communication structure." — Melvin Conway, 1968 ([[conway-team-topologies]]). Lina 전체를 소유하는 small team에게, 이것은 *monolith를 옹호한다*: 거울에 비출 team boundary가 없으므로, service split은 아무것도 사주지 않는 coordination cost(조정 비용)(deploy unit을 가로지르는 cognitive load)를 발명할 뿐이다. Inverse Conway Maneuver(역 콘웨이 책략) — "원하는 architecture를 유도하기 위해 team을 의도적으로 재구조화하라" ([[conway-team-topologies]]) — 는 예컨대 reasoning module 주위에 전담 team이 형성되는 경우에만/때에야 관련이 생긴다.

> 💡 **쉬운 설명:** Conway의 법칙은 "시스템 구조는 그것을 만든 조직의 소통 구조를 닮는다"는 것이다. Lina를 한 작은 팀이 다 만든다면, 거울에 비출 팀 경계가 없으니 굳이 서비스로 쪼갤 이유가 없다 — 오히려 쪼개면 없던 조정 비용만 생긴다. 반대로 나중에 reasoning 전담 팀이 생기면, 그때 일부러 조직을 재배치해 원하는 아키텍처를 유도하는 것이 Inverse Conway Maneuver다.

---

## 4. Step 4 — Design the tool-layer contracts (the [[ch-05]] move)

Lina의 tool/integration layer — `ForSendingMessages`, `ForSyncingCRM` 등 뒤의 adapter들 — 는 contract가 사는 곳이고, contract는 "service가 소유한 가장 되돌리기 비싼 artifact"다 ([[consumer-driven-contracts]]). 두 개의 contract surface(계약 표면)가 중요하다:

**(a) Lina가 consume하는 contract(SaaS vendor의 것).** **tolerant reader**(관대한 독자)가 되어라:

> "An implementation must be conservative in its sending behaviour and liberal in its receiving behaviour… message receivers should implement 'just enough' validation: that is, they should only process data that contributes to the business functions they implement." — Ian Robinson ([[consumer-driven-contracts]])

구체적으로: Lina의 CRM adapter는 CRM payload에서 필요한 field만 읽고 나머지는 무시해야 하며, 그래야 vendor가 field를 추가해도 Lina가 결코 깨지지 않는다. consumer 쪽의 strict schema validation은 정확히 "additive change를 breaking change로 바꾸는 것"이다 ([[consumer-driven-contracts]]).

> 💡 **쉬운 설명:** tolerant reader는 "보낼 때는 엄격하게, 받을 때는 관대하게(Postel의 법칙)"라는 태도다. Lina의 CRM 어댑터가 응답에서 자기가 쓰는 필드만 골라 읽고 나머지는 무시하면, 벤더가 새 필드를 추가해도 멀쩡하다. 반대로 받은 응답 전체를 엄격히 검증(strict schema validation)하면, 무해한 추가 변경조차 Lina를 깨뜨리는 breaking change로 둔갑한다.

**(b) Lina가 expose하는 contract**(만약/언제 Conversation이 추출되면, Lina의 나머지가 의존하는 API). 여기서 당신은 contract를 publish하고 consumer expectation을 obligation(의무)으로 취급한다:

> "When a provider accepts and adopts the reasonable expectations expressed by a consumer, it enters into a consumer contract." — Robinson ([[consumer-driven-contracts]])

CI의 consumer-driven contract test는 "independent deployability를 보존하는 유일한 메커니즘"이다 — provider는 deploy 전에 자신이 누군가를 깨뜨리는지 안다 ([[consumer-driven-contracts]], [[newman-building-microservices]]).

> 💡 **쉬운 설명:** consumer-driven contract(CDC) test는 "이 API를 쓰는 쪽(consumer)이 기대하는 바를 계약으로 명문화해 CI에서 자동 검증"하는 방식이다. 제공자(provider)는 배포 전에 자기 변경이 어떤 소비자를 깨뜨리는지 미리 안다. 이게 있어야 각 서비스가 서로를 깨뜨릴까 봐 동시에 배포할 필요 없이 독립적으로 배포할 수 있다.

### 4.1 API style and the honest-REST myth

default가 아니라, 당신이 필요로 하는 property로 style을 골라라 ([[fielding-rest]]):

- 캐시 가능하고 진화 가능한 resource를 위해서는 **REST/HTTP**(Lina 자신의 management API가 있다면).
- 저지연 internal service-to-service를 위해서는 **gRPC**(미래의 Pipeline↔Conversation 호출).
- 외부 vendor는 그들이 그러한 대로다; Lina가 적응한다.

Myth killed (ch-05's): "우리 API는 RESTful하다." Fowler가, Fielding을 인용하며: "Roy Fielding has made it clear that level 3 RMM is a pre-condition of REST" ([[fielding-rest]], from roy.gbiv.com — the ics.uci.edu mirror had a broken TLS chain, so quotes are from Fielding's own gbiv.com mirror). 업계가 REST라 부르는 거의 모든 것은 Level-2 HTTP-RPC다. **당신이 무엇을 ship하는지 ADR에서 정직하라** — 정확한 것은 비용이 들지 않으며 evolvability(진화 가능성)에 대한 올바른 기대를 설정한다.

> 💡 **쉬운 설명:** Richardson Maturity Model(RMM)은 API의 RESTful 수준을 0~3단계로 나눈다. Fielding(REST 창시자)은 Level 3(하이퍼미디어/HATEOAS)이라야 진짜 REST라고 못 박았다. 업계가 "REST API"라 부르는 대부분은 사실 Level 2(HTTP 동사+자원 URL을 쓰는 RPC)다. 거짓말할 필요 없이 ADR에 "우리는 Level 2다"라고 정직히 적으면, 나중에 진화 가능성에 대한 헛된 기대를 막을 수 있다.

### 4.2 Versioning and idempotency

versioned breakage보다 **additive, backward/forward-compatible(전/후방 호환) change**를 선호하라 ([[fielding-rest]]). 그리고 Lina가 수행하는 모든 *write*를 **idempotent(멱등)**하게 만들어라 — "make writes idempotent (idempotency keys) so retries are safe under at-least-once delivery" ([[fielding-rest]], [[transactional-outbox]]). 이것은 agent에게 선택사항이 아니다: Lina는 timeout 후에 "send email"이나 "create CRM record" 호출을 *재시도할 것이고*, idempotency key가 없으면 double-send(중복 전송)할 것이다. idempotency(contract 관심사)와 outbox(data 관심사, Step 5)는 "같은 대화"다 ([[transactional-outbox]]).

> 💡 **쉬운 설명:** idempotency(멱등성)는 "같은 요청을 여러 번 보내도 결과가 한 번 보낸 것과 같다"는 성질이다. Lina는 타임아웃 후 자동으로 재시도하는데, idempotency key(요청마다 붙이는 고유 식별자)가 없으면 같은 이메일을 두 번 보내거나 CRM 레코드를 두 번 만든다. at-least-once delivery(최소 1회 전달) 환경에서 안전하게 재시도하려면 모든 쓰기 작업에 멱등성이 필수다.

**The bet (price it).** publish되고 contract-test된 interface는 **boundary를 진화시키기 싸게** 유지한다(additive change는 tolerant reader를 깰 수 없고; breaking change는 CI에서 잡힌다) 그 대신 **discipline tax(규율 세금)**가 든다: 모든 변경이 contract suite를 돌리고, breaking change는 여전히 조율된 migration 비용이 든다. 그 비용은 실재한다; 그것은 independent deployability의 값이며, 그것만이 미래의 split을 할 가치가 있게 만드는 유일한 것이다.

---

## 5. Step 5 — Treat every SaaS response as outside data; saga + outbox (the [[ch-06]] move)

이것이 코스에서 가장 깊은 절개(cut)이고 *Lina가 하는 모든 외부 호출*에 적용된다. Helland의 근본 구별:

> Inside data is "the realm of SQL and SQL's DDL" — private, mutable, transactional, "now." Outside data "is immutable and each data item's schema is versioned"; it "is stable, such that a repeated request is unchanged, and a reading of it results in the same interpretation." — Pat Helland ([[helland-data-outside-inside]]; the ACM Queue reprint returned 403 at fetch, so claims are corroborated via the CIDR-2005 PDF, Semantic Scholar, and "the morning paper" summary — cite as a summary-corroborated source, not a clean fetch).

하중을 지탱하는 세 주장을, Lina에 적용하면:

1. **Services don't share transactions.** Lina는 자신의 DB *와* CRM을 둘러 transaction을 감쌀 수 없다. 그래서 Lina+vendor를 가로지르는 2PC(2-phase commit)는 막다른 길이다 ([[helland-data-outside-inside]]).
2. **Outside data must be immutable.** Lina가 수집하는 모든 CRM record, 모든 enrichment result, 모든 calendar event는 *snapshot*이며, 불변으로 저장되고 identity-stamp(식별 도장)가 찍힌다 — "재시도, 캐시, 재정렬, replay에 안전하다" ([[helland-data-outside-inside]]).
3. **Outside data may be stale, and that's fine.** Lina가 200ms 전에 읽은 CRM record는 이미 틀렸을 수 있다. 그것을 위해 설계하라. 그 비용에 대한 Fowler의 프레이밍: "business logic can end up making decisions on inconsistent information" ([[fowler-microservices]], [[helland-data-outside-inside]]).

**Lina에서 가장 leverage가 큰 단일 결정:** agent의 **inside model**(자신의 private Pipeline/Conversation state, ACID, "now")을 **그것이 수집하고 방출하는 것**(불변의, 버전 매겨진, 오래되었을 수도 있는 SaaS snapshot)과 분리하라. [[helland-data-outside-inside]]에 따르면, 이것은 여러 SaaS tool을 다루는 agent에게 "가용한 가장 leverage가 풍부한 boundary 결정"이다.

> 💡 **쉬운 설명:** Helland는 데이터를 두 종류로 나눈다. inside data는 내 DB 안의 데이터로 트랜잭션 가능하고 가변이며 "지금" 진실이다. outside data는 외부에서 받은 데이터로, 불변이고 버전이 찍혀 있으며 "내가 받은 그 순간의 스냅샷"일 뿐 지금도 맞는지는 보장 못 한다. Lina가 읽은 CRM 레코드는 0.2초 만에 낡을 수 있다. 그래서 외부 응답은 절대 "현재 공유 상태"로 다루지 말고, 도장 찍힌 불변 스냅샷으로 저장하고 낡을 수 있음을 전제로 설계해야 한다. 이게 SaaS를 많이 쓰는 agent에서 가장 큰 레버리지를 가진 결정이다.

### 5.1 Sagas for multi-step external operations

하나의 Lina operation이 여러 외부 시스템을 업데이트해야 할 때 — 예컨대 *win a deal* = CRM stage 업데이트 → win-notice email 전송 → handoff meeting 예약 → Slack 알림 — 당신은 그것을 하나의 transaction으로 할 수 없다. **saga**를 사용하라:

> "A saga is a sequence of local transactions. Each local transaction updates the database and publishes a message or event to trigger the next local transaction in the saga." — Chris Richardson ([[richardson-saga]])

(이 construct는 long-lived transaction을 위한 Garcia-Molina & Salem의 1987년 "Sagas" 논문에서 유래한다 — 하지만 그 PDF는 text layer가 없는 image-only라서, LLT/compensation thesis는 논문에 대한 지식에서 추출되었으며 **verbatim으로 인용된 것이 아니다**; Richardson의 microservices.io 페이지가 verbatim-quotable한 source다.)

**Choreography vs orchestration** — Lina의 "win a deal" flow에 대해서는, *orchestration*(오케스트레이션)이 더 나은 bet이다: 그것은 "복잡하고 상호의존적 step이 많은" flow이며, orchestrator는 "process를 보고/디버그할 한 곳"을 준다 ([[richardson-saga]]) — 이는 당신이 audit해야 하는 autonomous agent에게 엄청나게 중요하다. 그 비용은 "orchestrator가 bottleneck이 된다"는 것과 coupling point가 된다는 것이다.

> 💡 **쉬운 설명:** saga는 "여러 개의 작은 local transaction을 이벤트로 연쇄시켜 큰 작업을 완성하는" 패턴이다. 분산 환경에선 하나의 거대한 트랜잭션이 불가능하니까. 진행 방식은 두 가지 — choreography(각 단계가 다음 단계 이벤트를 발행하며 자율적으로 이어짐)와 orchestration(중앙 조율자가 단계를 지시함). Lina의 "딜 성사" 같은 복잡한 흐름은 orchestration이 낫다. 한곳에서 전 과정을 보고 감사·디버그할 수 있기 때문인데, 대신 그 조율자가 병목이자 결합점이 되는 대가가 따른다.

**The price of a saga — name it.** 당신은 **isolation(ACID의 "I")**을 거래로 내준다:

> "Lack of isolation (the 'I' in ACID)… means there's risk that the concurrent execution of multiple sagas and transactions can [cause] data anomalies." — Richardson ([[richardson-saga]])

그래서 Lina는 **countermeasure(대응책)**가 필요하다 — semantic lock(deal을 "win-in-progress"로 표시), commutative update, re-read, by-value tracking ([[richardson-saga]]). saga는 *공짜* transaction이 *아니다*. 그리고 모든 step은 **compensating transaction(보상 트랜잭션)**이 필요하다: CRM 업데이트가 성공한 후 win-notice email이 실패하면, 당신은 앞으로 retry하거나 compensate(CRM 변경을 되돌리거나 flag)한다. compensation을 설계하라, 바라지 말고.

> 💡 **쉬운 설명:** 일반 트랜잭션은 ACID 중 "I"(isolation, 격리)가 있어 작업 중간 상태가 남에게 안 보이지만, saga는 단계 사이에 커밋이 일어나므로 격리를 포기한다 — 그래서 진행 중인 다른 작업이 어중간한 상태를 볼 수 있다. 이를 막으려고 semantic lock("처리 중" 표시) 같은 대응책을 쓴다. 또 어느 단계가 실패하면 이미 커밋된 앞 단계를 되돌릴 compensating transaction(보상 트랜잭션, 예: CRM 단계 원복)을 미리 설계해 둬야 한다.

### 5.2 The transactional outbox

Lina는 자신의 DB를 쓰는 것 *과* 다음 saga step을 trigger하는 event를 publish하는 것을 atomic하게 할 수 없다:

> "How to atomically update the database and send messages to a message broker?" … "It is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker." — Richardson ([[transactional-outbox]])

해결책:

> "The solution is for the service that sends the message to first store the message in the database as part of the transaction that updates the business entities." — Richardson, with the guarantee "Messages are guaranteed to be sent if and only if the database transaction commits." ([[transactional-outbox]])

그것을 비동기로 relay하라 — **polling publisher**(단순, 더 많은 DB load/latency) 또는 **transaction-log tailing / CDC**(낮은 latency, 더 많은 infra) ([[transactional-outbox]]). delivery는 **at-least-once**이므로, 이 event를 consume하는 모든 Lina consumer는 **idempotent해야 한다** — 이것이 Step 4의 idempotency key와 고리를 닫는다.

> 💡 **쉬운 설명:** transactional outbox는 "DB 쓰기와 메시지 발행을 한 트랜잭션으로 묶는" 트릭이다. DB와 메시지 브로커를 동시에 원자적으로 갱신하는 건(2PC) 비현실적이므로, 보낼 메시지를 비즈니스 데이터와 같은 트랜잭션 안에서 DB의 outbox 테이블에 먼저 저장한다. 트랜잭션이 커밋되면 메시지도 확실히 남는다. 그 다음 별도 프로세스가 outbox를 읽어(polling 또는 CDC=변경 로그 추적) 브로커로 비동기 전송한다. 최소 1회 전달이므로 받는 쪽은 반드시 멱등해야 한다 — Step 4의 idempotency key가 여기서 다시 등장한다.

**The bet (price it).** inside/outside-data 규율 + saga + outbox는 **각 외부 integration을 독립적으로 실패 가능하고 replay 가능하게** 유지한다(재시도, 캐시, 추론하기 싸다) 그 대신 **cross-system isolation과 "live truth"(살아 있는 진실)를 포기한다** — Lina는 도처에서 staleness를 받아들이고 compensation을 설계해야 한다. 십수 개의 불안정한 SaaS vendor를 다루는 agent에게, 이것은 선택이 아니다; 그것은 애초에 작동하기 위한 비용이다. 유일한 선택은 그것을 의도적으로(설계된 model로) 치를지, 우발적으로(CRM 행을 live shared state로 다루는 [[distributed-monolith]]로) 치를지다.

---

## 6. Step 6 — Reach for CQRS/ES only if a force demands it (the [[ch-07]] move)

power tool은, default로 거부된다. CQRS:

> "CQRS stands for Command Query Responsibility Segregation. At its heart is the notion that you can use a different model to update information than the model you use to read information." — Fowler ([[young-cqrs-es]])

하중을 지탱하는 주의:

> "For some situations, this separation can be valuable, but beware that for most systems CQRS adds risky complexity." … "CQRS should only be used on specific portions of a system… and not the system as a whole." … "Many systems do fit a CRUD mental model, and so should be done in that style." — Fowler ([[young-cqrs-es]])

**For Lina:** Pipeline core는 대부분 CRUD다 — 거기서 CQRS를 거부하라. *하나의* portion이 그럴듯하게 그것을 벌어들인다: write model과 query shape가 심하게 갈리고 read load가 독립적으로 scale하는, sales activity에 대한 **reporting/analytics read model**(pipeline velocity, agent-action dashboard)이다. CQRS를 *그 portion에만* scope하고, outbox가 이미 방출하는 동일한 domain event로 먹여라.

> 💡 **쉬운 설명:** CQRS는 "정보를 쓸 때(Command)와 읽을 때(Query)에 서로 다른 모델을 쓴다"는 발상이다. Fowler의 경고가 핵심이다 — 대부분의 시스템엔 위험한 복잡도만 더하므로, 시스템 *전체*가 아니라 *특정 부분*에만 써야 한다. Lina의 Pipeline은 평범한 CRUD라 CQRS 불필요. 다만 영업 분석 대시보드처럼 읽기 형태가 쓰기와 완전히 다르고 읽기 부하가 따로 폭증하는 부분이라면, 거기 한정해서 outbox가 내보내는 같은 event로 읽기 모델을 채우는 게 정당화된다.

Event Sourcing:

> "Capture all changes to an application state as a sequence of events." — Fowler ([[young-cqrs-es]])

주의:

> "Clearly this stuff can get very messy, don't go down this path unless you really need to." — Fowler ([[young-cqrs-es]])

**For Lina:** *진짜* candidate force가 있다 — consequential action(중대한 행동)을 취하는 autonomous agent는 실제 **audit/replay need(감사/재생 필요)**를 가진다("왜 Lina가 이 lead에게 email을 보내고 저 meeting을 예약했나?"). Event Sourcing의 공짜 audit log와 replay-debugging은 agent accountability(책임성)에 직접 매핑된다. 하지만 sharp edge(날카로운 모서리)를 저울질하라: replay 시의 external side-effect(Lina는 replay할 때 실제 send를 gateway/suppress해야 한다)와 historical-event schema evolution(과거 event의 스키마 진화) ([[young-cqrs-es]]). **Decision for the memo:** *Conversation/agent-action* context를 event-source하고(audit이 가장 중요한 곳) Pipeline/Scheduling은 CRUD로 두라 — audit 요구가 실재할 *경우에 한해서만*, aspirational(희망사항)이 아니라.

> 💡 **쉬운 설명:** Event Sourcing(ES)은 "현재 상태가 아니라, 상태를 만든 모든 변경 사건(event)의 연속을 저장"하는 방식이다. 자율 agent는 "왜 이런 행동을 했나"를 추적해야 하므로(책임성), 모든 행동을 event로 남기는 ES의 공짜 감사 로그·재생 능력이 잘 들어맞는다. 단 함정이 있다 — 과거 event를 replay하면 실제 이메일이 또 나갈 수 있으니 막아야 하고(side-effect 억제), 오래된 event의 스키마가 바뀌면 골치 아프다. 그래서 감사가 진짜 필요한 Conversation 컨텍스트에만 적용하고, Pipeline 등은 평범한 CRUD로 둔다.

**Myth killed (ch-07's):** "CQRS와 Event Sourcing은 같다 / 함께 간다." 그것들은 합성되는 독립적 결정이다; 어느 쪽도 다른 쪽을 요구하지 않으며, 어느 쪽도 microservices를 요구하지 않는다 ([[young-cqrs-es]]). 둘 다의 in-process 씨앗은 domain event를 방출하는 aggregate다(Step 2.1, [[ddd-aggregates-tactical]]) — 당신이 이미 만든 것이다.

**The bet (price it).** 각 tool은 **하나의 특정 property를 싸게** 유지한다(CQRS: 독립적 read scaling / query-shape 자유; ES: audit + temporal replay) 그 대신 **그것을 채택하는 portion 전체에 걸쳐 risky complexity**가 든다. named force(asymmetric read scaling; 실제 audit/replay 필요)가 그 비용을 치를 가치가 있게 만들지 않는 한 둘 다 거부하라. default 답은 "no"다.

---

## 7. Step 7 — Place timeouts/breakers/bulkheads at every integration point (the [[ch-08]] move)

모든 외부 SaaS 호출은 integration point(통합 지점)이고, integration point는 failure가 들어오는 곳이다:

> "Integration Points without Timeouts is a surefire way to create Cascading Failures." — Michael Nygard ([[nygard-release-it]], *Release It!* — book, thesis extracted)

toolkit을, Lina에 적용하면 ([[nygard-release-it]]):

- **Timeout** — *모든* remote call(CRM, email, calendar, LLM)에 bound(한계)를 둔다. "An unbounded wait is a held resource"(무한 대기는 점유된 자원이다)이며, 점유된 자원은 하나의 느린 vendor가 전체 agent loop를 멈추게 하는 방식이다.
- **Circuit Breaker** — 각 불안정한 vendor를 감싼다; failure가 threshold를 넘으면, *open*하여 죽은 service에 요청을 쌓는 대신 빠르게 실패한다; 회복을 테스트하기 위해 주기적으로 *half-open*한다. enrichment API가 다운되면, Lina는 모든 deal을 멈추는 게 아니라 enrichment를 skip해야 한다.
- **Bulkhead** — vendor당 resource pool을 분할하고(그리고 LLM-call pool을 격리) 한 vendor의 outage가 "배 전체를 가라앉히지 않게" 한다. CRM timeout의 폭풍이 Lina가 mail을 보내는 데 필요한 thread를 소진해선 안 된다.
- **Fail Fast** / **Steady State** — 완료 불가능한 작업을 감지하고 즉시 return한다; 모든 축적(conversation log, cached snapshot)이 짝이 맞는 cleanup을 가지도록 보장한다 ([[nygard-release-it]]).

> 💡 **쉬운 설명:** 외부 호출이 들어오는 모든 지점에 네 가지 방어구를 둔다. Timeout(한없이 기다리지 말고 시간 제한), Circuit Breaker(반복 실패하는 벤더는 차단기를 "열어" 빠르게 실패시키고 죽은 서비스에 요청을 쌓지 않음), Bulkhead(벤더별로 자원 풀을 칸막이로 나눠 한 곳의 장애가 전체를 침몰시키지 않게 — 배의 격벽에서 따온 이름), Fail Fast(안 될 일은 즉시 포기). 이렇게 해야 한 벤더 장애가 agent 전체로 번지는 cascading failure를 막는다.

**Why this is design altitude, not ops:** "where you place circuit breakers and bulkheads defines your system's blast radius. A [[distributed-monolith]] is precisely a system that skipped them" ([[nygard-release-it]]). C4 Container diagram(Step 9)은 모든 integration point를 breaker가 사는 곳*으로서* 가시화해야 한다.

> 💡 **쉬운 설명:** 이 패턴들을 어디에 두느냐는 운영(ops) 문제가 아니라 설계(design) 문제다 — circuit breaker와 bulkhead의 위치가 곧 시스템의 blast radius(장애 파급 반경)를 정의하기 때문이다. distributed monolith가 위험한 이유가 바로 이것들을 빼먹은 시스템이라는 점이다.

**Async is itself a resilience decision.** synchronous call-chain 대신 event/outbox integration을 선택하는 것은 "이 pattern들이 그렇지 않으면 방어해야 하는 synchronous coupling을 제거한다" ([[nygard-release-it]], [[richardson-saga]]). Step 5의 Lina의 saga-over-outbox 설계는 이미 이것의 많은 부분을 사준다: 하류 vendor outage는 event를 지연시킬 뿐, agent를 synchronous하게 멈추지 않는다.

**The bet (price it).** resilience pattern은 **blast radius를 작게** 유지한다(한 vendor 다운 ≠ Lina 다운) 그 대신 **추가된 latency, complexity, 그리고 degraded-mode(성능 저하 모드) logic** 비용이 든다 — Lina는 이제 "enrichment가 open-circuit되었을 때"에 대한 정의된 behavior를 가져야 한다(그것 없이 진행할까? 나중을 위해 queue할까?). 그 degraded-mode 설계는 일이다; 그것을 건너뛰는 것이 distributed system이 cascade하는 방식이다.

---

## 8. Step 8 — Pick the first seam to strangle + fitness functions (the [[ch-09]] move)

modular monolith는 *시작* bet이지, *최종* bet이 아니다 — 그것을 수정 가능하게 유지하라. module의 force가 마침내 extraction을 정당화할 때(Step 3.2가 Conversation을 먼저 flag했다), rewrite하지 마라:

> "Replacing a serious IT system takes a long time, and the users can't wait for new features." — Fowler ([[martin-strangler-fig]])

**Strangler Fig**(교살자 무화과)를 사용하라: façade를 통해 intercept → 하나의 capability seam(과 그 데이터를, 함께, 그래야 migration 중간에 shared-DB dependency를 만들지 않는다, [[newman-building-microservices]]에 따라) 추출 → 검증 & 축소 → 반복 ([[martin-strangler-fig]]). "Investment and returns occur gradually and visibly"(투자와 수익이 점진적이고 가시적으로 발생한다)이며, "since these components are small, there isn't so much risk involved"(이 component들이 작으므로, 관련된 위험이 그리 크지 않다) ([[martin-strangler-fig]]). 그것은 modular-monolith-first의 운영적 대응물이다: 둘 다 "아직 신뢰할 수 없는 boundary에 대해 단일의 큰 비가역적 bet을 하기를 거부한다."

> 💡 **쉬운 설명:** Strangler Fig는 열대 무화과가 숙주 나무를 천천히 감싸 결국 대체하는 데서 따온 이름이다. 낡은 시스템을 한 번에 다시 쓰는(rewrite) 대신, façade(앞단 가로채기 막)를 두고 기능 한 조각씩(데이터까지 함께) 새 구현으로 옮긴 뒤 옛 부분을 줄여나가며 반복한다. 작은 조각씩이라 위험이 작고, 사용자는 그동안에도 계속 서비스를 쓴다. modular-monolith-first가 "처음에 한 번에 베팅하지 않기"라면, Strangler Fig는 "나중에 한 번에 갈아엎지 않기"다.

**Lina's first strangle target:** Conversation/reasoning module이다. 왜냐하면(Step 3.2에서) 그것이 가장 강한 disintegrator profile을 가지기 때문이다 — bursty LLM-bound scaling, fault-isolation 필요, 매주 volatility. 이미 존재하는 `ForReasoning` port 뒤에서 그것을 추출하라(Step 2): 그 port가 *바로* interception seam이다. 이것이 Step 2를 제대로 한 것의 보상이다 — strangler가 자를 깨끗한 자리를 가진다.

> 💡 **쉬운 설명:** Step 2에서 `ForReasoning` port를 미리 정의해 둔 덕분에, 나중에 Conversation 모듈을 떼어낼 때 그 port가 그대로 "자를 자리(seam)"가 된다. port 뒤의 adapter만 in-process 구현에서 원격 서비스 호출로 바꾸면 끝이다. 깔끔한 경계를 미리 만들어두는 일(Step 2)이 미래의 분리(Step 8)를 싸게 만든다는 게 핵심 보상이다.

### 8.1 Fitness functions — stop the bet from rotting

선택한 characteristic을 aspirational이 아니라 enforced(강제된) 상태로 유지하라. **fitness function**(적합도 함수)은 "어떤 architectural characteristic(들)에 대한 objective integrity assessment(객관적 무결성 평가)"다 — 보호된 characteristic이 erode(침식)될 때 build를 실패시키는 자동화된 check다 ([[richards-ford-fundamentals]], from *Building Evolutionary Architectures*). Lina를 위해, lab의 결정들을 fitness function으로 인코딩하라:

| Decision to protect | Fitness function |
|---|---|
| Dependency rule holds (Step 2) | ArchUnit-style test: 어떤 `domain` package도 vendor SDK나 framework type을 import하지 않음 |
| Modules don't reach into each other's tables (Step 3) | schema-access lint: module X의 코드는 오직 module X의 schema만 건드림 |
| Inside ≠ outside data (Step 5) | test: 어떤 SaaS DTO type도 core aggregate의 field에 나타나지 않음 |
| Every remote call has a timeout (Step 7) | static check / test: adapter에 un-timed HTTP client가 없음 |
| p99 agent-loop latency < target | monitor-as-fitness-function in CI/prod |

"This is how you keep the expensive-to-reverse decisions from rotting silently as the system evolves"(이것이 비싸게-되돌릴 결정들이 시스템이 진화하면서 조용히 부패하는 것을 막는 방법이다) ([[richards-ford-fundamentals]]).

> 💡 **쉬운 설명:** fitness function은 "아키텍처 특성이 무너지면 빌드를 깨뜨리는 자동 테스트"다. 예컨대 "domain 패키지가 벤더 SDK를 import하면 실패"(Step 2의 의존성 규칙), "모듈이 남의 테이블을 건드리면 실패"(Step 3) 같은 식이다. 사람이 잊어버려도 CI가 매번 검사하므로, 어렵게 내린 결정들이 시간이 지나며 슬그머니 어겨지는 것을 막는다.

**The bet (price it).** evolutionary discipline(진화적 규율)(strangler + fitness function)은 **architecture를 수정 가능하고 self-defending(자기 방어적)하게** 유지한다 그 대신 **지속적인 enforcement와 migration 노력** 비용이 든다 — fitness function은 당신이 작성하고 유지해야 하는 test이고; strangling은 step당 rewrite보다 느리다. 당신은 어떤 단일 결정도 조용히 비가역적이 되지 않도록 지속적으로 대가를 치른다. 그것이 코스 전체의 요점이다: architecture는 비싸게-되돌릴 집합이고, master move(필살기)는 그것을 *계속 줄이는 것*이다 — Fowler의 수정안, "a good architect makes change easier, thus reducing architecture"(좋은 architect는 변화를 더 쉽게 만들고, 그리하여 architecture를 줄인다) ([[richards-ford-fundamentals]]).

---

## 9. Worked partial example — the **CRM-Sync** bounded context

shape를 보여주기 위해, 하나의 context를 toolkit을 통해 end-to-end로 실행한다. (나머지 셋은 당신이 완성한다.)

**Context:** CRM-Sync — Lina의 inside Pipeline model을 외부 CRM의 outside model과 reconcile한다.

1. **Boundary (Step 1):** language가 이동한다 — "Lead"(pipeline entity) ⇄ "Contact/Opportunity"(vendor ID + sync status를 가진 CRM record). 별개 model ⇒ 자신의 context. **Bet:** sync vocabulary를 격리하는 것은 Pipeline의 model을 깨끗하게(진화시키기 싸게) 유지하는 대신 유지되어야 하는 명시적 translation layer(ACL)의 비용이 든다.
2. **Structure (Step 2):** core가 `ForSyncingCRM`(port)를 정의한다; `SalesforceAdapter` / `HubSpotAdapter`가 그것을 구현한다(driven adapter). core는 어느 vendor의 SDK에도 의존하지 않는다. **Aggregate:** `SyncRecord { leadId, vendorId, lastSyncedSnapshot, status }` — 작고, `Lead`를 identity로 참조한다.
3. **Topology (Step 3):** 처음에는 service가 아니라 *module*. Disintegrator(I/O-bound, 독립적으로 실패)는 그것을 *두 번째* extraction candidate로 만든다 — 하지만 integrator(tight `Lead` dependency)는 "아직 아니다"라고 말한다.
4. **Contract (Step 4):** CRM payload에 대한 tolerant reader — `{id, stage, owner, parentAccount}`만 읽고 나머지는 무시하여, vendor field 추가가 결코 Lina를 깨뜨리지 않게. 모든 CRM write에 idempotency key.
5. **Consistency (Step 5):** 모든 CRM read는 **outside data**다 — snapshot을 불변으로, fetch-time과 vendor version으로 stamp하여 저장; 결코 live truth로 다루지 않는다. CRM을 건드리는 "win a deal" flow는 orchestrated saga의 한 *step*이며, compensating "revert stage" transaction을 가지고; trigger event는 outbox를 통해 온다.
6. **Power tools (Step 6):** 여기서는 CQRS도, ES도 없다 — CRUD-shaped reconciliation이다. 이름을 붙여, 거부됨.
7. **Resilience (Step 7):** CRM adapter 주위에 timeout + circuit breaker + 전용 bulkhead pool. Degraded mode: breaker가 open이면, outbox를 통해 sync를 queue하고 진행한다; agent loop를 멈추지 마라.
8. **Evolution (Step 8):** fitness function — "어떤 Salesforce/HubSpot type도 core aggregate에 나타나지 않음"; 전담 integrations team이 형성되면(Conway) `ForSyncingCRM` 뒤의 미래 strangle target으로 flag됨.

shape는 Conversation, Scheduling, Pipeline에 대해 동일하다 — 오직 force만 다르다. **그 동일함이 교훈이다:** toolkit은 고정된 sequence다; 변하는 것은 trade-off weight(트레이드오프 가중치)다.

> 💡 **쉬운 설명:** 이 §9는 CRM-Sync 한 컨텍스트를 Step 1~8 전 과정으로 끝까지 돌려 보인 모범 답안이다. 나머지 세 컨텍스트(Conversation, Scheduling, Pipeline)도 똑같은 8단계 순서를 적용하면 된다. 단계(toolkit)는 고정이고, 각 단계에서 disintegrator/integrator 같은 힘의 무게만 컨텍스트마다 달라진다 — 바로 그 "순서는 같고 가중치만 다르다"가 이 코스의 핵심 교훈이다.

---

## 10. Deliverable template A — ADR skeleton (fill in)

Nygard의 구조를 사용하라: "Title, Status (proposed/accepted/deprecated/superseded), Context, Decision, Consequences"; "one or two pages"; "write each ADR as if it is a conversation with a future developer" — 그리고 오직 "architecturally significant"한 결정만 기록하라: "those that affect the structure, non-functional characteristics, dependencies, interfaces, or construction techniques" ([[nygard-release-it]]).

```
# ADR-00X: <short imperative title, e.g. "Default Lina to a modular monolith">

## Status
<proposed | accepted | deprecated | superseded by ADR-0YY>

## Context
<the forces in play: which architecture characteristics matter here, what
 the requirement is, what alternatives exist. State the uncertainty honestly.>

## Decision
<what we will do, in one or two sentences. Active voice.>

## Consequences
<THE BET, priced both ways:
   - This keeps CHEAP TO CHANGE: <...>
   - This makes EXPENSIVE: <...>
   - Risks / what would make us revisit this (the trigger to supersede): <...>>
```

> 💡 **쉬운 설명:** ADR은 한두 페이지짜리 "미래 개발자와의 대화"다. 모든 결정을 적는 게 아니라 구조·비기능 특성·의존성·인터페이스·구축 기법에 영향을 주는 "architecturally significant"한 결정만 기록한다. 가장 중요한 칸은 Consequences인데, 반드시 "무엇이 싸게 유지되는가 / 무엇이 비싸지는가 / 언제 이 결정을 다시 볼 것인가"의 세 방향을 모두 적어야 한다.

**Minimum ADR set this lab must produce** (각각 하나씩):

1. Lina의 bounded context (Step 1).
2. Modular-monolith-first topology + 첫 extraction candidate (Step 3). *(the spine ADR)*
3. 모든 SaaS response에 대한 inside/outside-data 규율 (Step 5).
4. multi-system flow를 위한 saga style (orchestration) + outbox (Step 5).
5. integration point에서의 resilience default (Step 7).
6. CQRS/ES scope 결정 — 무엇이 들어가고, 무엇이 거부되며, 왜인지 (Step 6).
7. 위의 것들을 지키는 fitness function (Step 8).

각 ADR의 Consequences section은 bet의 *양쪽* 면을 모두 이름 붙이지 않으면 무효다. 그것이 grading rubric(채점 루브릭)이다.

---

## 11. Deliverable template B — C4 Context + Container sketch (fill in)

C4는 "a set of hierarchical abstractions — software systems, containers, components, and code"다 ([[c4-model]]). 당신은 두 level을 만든다.

**Myth to keep front-of-mind (ch-01's, restated):** C4 **container**는 "applications and data stores… a separately runnable/deployable unit"이다 — "NOT a Docker container" ([[c4-model]], [[COLLECTION-PLAN]]). Lina의 diagram에서, "container"는 deployable agent app, 그것의 database, 그것의 message bus다 — 그것들이 어떻게 패키징되었든 상관없이.

> 💡 **쉬운 설명:** C4의 "container"는 도커 컨테이너가 절대 아니다. 여기서 container는 "따로 실행/배포되는 단위" — 즉 agent 앱 프로세스, 데이터베이스, 메시지 버스 같은 것들이다. 패키징 방식(도커든 아니든)과 무관하다. 이름이 헷갈려서 이 코스가 1장부터 죽이려 한 통념 중 하나다.

**Level 1 — System Context** (the system as one box; audience: everyone):

```
[Person: Sales rep] ──uses──> ( LINA TMR — autonomous sales agent )
                                        │
        ┌───────────────┬──────────────┼───────────────┬──────────────┐
        ▼               ▼              ▼               ▼              ▼
   [Ext: CRM]     [Ext: Email]   [Ext: Calendar]  [Ext: LLM API]  [Ext: Enrichment]
   <fill: which external systems Lina talks to, and the direction/purpose of each edge>
```

**Level 2 — Container** (zoom in: the deployable apps + data stores; audience: technical/ops). **이것이 distributed monolith가 가시화되는 diagram이다** — deploy-coupled container와 shared data store를 세어라:

```
┌────────────────────────── LINA TMR (system boundary) ──────────────────────────┐
│                                                                                  │
│   ( Agent App  — modular monolith )            [ Outbox / Message Bus ]          │
│     modules: Pipeline | Conversation |          <fill: relay style —             │
│              Scheduling | CRM-Sync               polling vs CDC>                  │
│        │                                                                         │
│        ▼                                                                         │
│   [ Lina DB — inside data, ACID ]   [ Snapshot store — outside data, immutable ] │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
        │ (every outbound edge below crosses a port + adapter with
        │  TIMEOUT + CIRCUIT BREAKER + BULKHEAD — mark each one)
        ▼
   <fill: edges to each external SaaS, labelled with the port name and the
    resilience pattern guarding it; mark which edges are synchronous vs
    event/outbox-driven>
```

> 💡 **쉬운 설명:** Level 1(System Context)은 "Lina를 하나의 상자로 보고, 누가 쓰고 어떤 외부 시스템과 대화하는가"를 보여주는 가장 넓은 그림이다. Level 2(Container)는 한 단계 줌인해 "배포 단위인 앱, DB, 메시지 버스"를 보여준다. 특히 Level 2가 중요하다 — 여기서 두 모듈이 같은 DB를 공유하거나 함께 배포돼야 하는 게 드러나면, 그게 바로 distributed monolith의 정체다.

**Completion checklist for your C4 sketch** (각각이 당신이 표면화해야 하는 course concept이다):

- [ ] **inside data**(`Lina DB`)가 **outside data**(snapshot store)와 별개로 그려졌는가? ([[helland-data-outside-inside]])
- [ ] 어떤 두 module이 schema를 공유하는가? 그렇다면 당신은 둘이 아니라 하나의 quantum을 그린 것이다 — 그리고 대기 중인 [[distributed-monolith]]다. ([[richards-ford-hard-parts]])
- [ ] 모든 외부 edge가 그것의 **port name**(dependency rule, [[martin-clean-arch]])과 그것의 **timeout/breaker/bulkhead**([[nygard-release-it]])로 표시되었는가?
- [ ] saga step이 synchronous call-chain이 아니라 **event/outbox-driven**(async)인가? ([[richardson-saga]], [[transactional-outbox]])
- [ ] 하나의 container를 가리키며 "이것이 내가 가장 먼저 strangle해 낼 것이고, 여기 그것을 trigger하는 force가 있다"고 말할 수 있는가? ([[martin-strangler-fig]])

모든 box와 edge가 그 뒤에 한 줄짜리 trade-off를 가진다면, 당신은 lab을 — 그리고 코스를 — 통과한 것이다.

---

## Where this goes

ch-11은 없다; 이것이 capstone이다. 하지만 forward pointer는 당신 자신의 repository로 향한다: 당신이 방금 초안 잡은 일곱 개의 ADR과 두 개의 C4 level을 가져다 Lina의 코드 옆에 version control 아래 두고, 그런 다음 §8.1의 fitness function을 CI에 wire하여 오늘 가격 매긴 bet들이 부패하기 시작하는 날 build를 실패시키게 하라. 코스의 최종 주장은 architecture가 그것의 enforcement만큼만 살아 있다는 것이다: First Law는 모든 결정이 trade-off라고 말하고, ADR은 어떤 trade를 했는지 기록하고, C4 diagram은 잘못된 trade가 어디서 가시화될지 보여주고, fitness function은 당신이 보고 있지 않을 때 trade가 조용히 스스로를 되돌리는 것을 막는 것이다. architecture는 비싸게-되돌릴 집합이다 — 그리고 architect의 일 전체는, 마지막으로 한 번 더 진술하건대, 그 집합을 작게, 이름 붙여진 채로, 그리고 수정 가능하게 유지하는 것이다.

> 💡 **쉬운 설명:** 이 lab의 산출물(ADR 7개 + C4 2단계)은 종이 위에서 끝나면 죽은 문서다. 코드 저장소에 함께 넣어 버전 관리하고, fitness function을 CI에 연결해야 비로소 살아 있는 아키텍처가 된다. 정리하면 네 가지가 짝을 이룬다 — First Law(모든 건 trade-off다) → ADR(어떤 trade를 했는지 기록) → C4(잘못된 trade가 어디서 드러날지 표시) → fitness function(그 trade가 몰래 뒤집히지 않게 감시). architect의 일은 "되돌리기 비싼 결정 집합을 작게, 이름 붙여, 수정 가능하게 유지"하는 것, 그 한 문장으로 코스가 끝난다.
