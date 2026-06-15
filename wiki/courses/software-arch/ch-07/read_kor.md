<!-- chapter: ch-07
     track: consistency
     kind: content
     title: CQRS and Event Sourcing: Optional Power Tools
     deps: [[ch-06]]
     sources: [[young-cqrs-es]], [[ddd-aggregates-tactical]], [[nygard-release-it]], [[COLLECTION-PLAN]], [[insights]]
-->

# 07장 — CQRS and Event Sourcing: Optional Power Tools

> **핵심 통찰.** CQRS(쓰기에는 하나의 model을, 읽기에는 다른 model을 사용하는 것)와 Event Sourcing(event의 log를 source of truth로 저장하고 그것을 replay하여 state를 재구성하는 것)은 우연히 잘 결합되는 두 개의 *독립적인* 결정이다 — 하나의 pattern이 아니고, package deal(묶음 거래)도 아니며, microservices의 전제 조건도 아니다. 각각은 특정하고 좁은 capability를 사는 power tool이며, 각각은 그 대가로 영구적인 통행료(toll)를 부과한다: CQRS는 read side를 영원히 eventually consistent(최종적으로 일관된)하게 만든다; Event Sourcing은 external side-effect(외부 부수 효과)와 historical-schema change(과거 schema 변경)를 영원히 비싸게 만든다. 이들은 [[ch-06]]의 consistency arc(일관성 호) *맨 위*에 위치하는데, 바로 이들이 가장 마지막에 손을 뻗는 pattern이기 때문이다 — 필수적인 inside/outside-data, saga, outbox 메커니즘을 거친 후에야, 그리고 오직 특정한 force(힘)가 그것을 요구할 때만. 거의 모든 시스템의 거의 모든 부분에 대한 기본 답은 Fowler가 직접 말하는 것이다: 잘 그려진 aggregate에 대한 평범한 CRUD([[young-cqrs-es]], [[ddd-aggregates-tactical]]).

> 💡 **쉬운 설명:** 이 챕터의 핵심은 "이 두 도구는 강력하지만, 대부분의 경우 *쓰지 않는 것*이 정답"이라는 점이다. power tool은 만능 망치가 아니라, 특정 작업에만 쓰는 전동 공구라고 생각하면 된다. 잘못 꺼내면 평생 갚아야 할 빚(영구 비용)이 생긴다.

> **가이드라인.** 둘 다 *기본적으로 거부하는* 베팅으로 취급하고, 명명된 force 아래에서만 대가를 지불하라. **CQRS**는 read model과 write model이 진정으로 갈라지는 시스템의 *부분*에만 손을 뻗어라 — query shape이 command shape과 일치하지 않는 복잡한 domain, 또는 두 경로가 독립적으로 scale되어야 하는 비대칭 read/write scaling. **Event Sourcing**은 평범한 audit table로는 충족할 수 없는 audit, temporal query(시점 질의), 또는 replay-debugging에 대한 실제 필요가 있을 때만 손을 뻗어라. 각각을 그것이 필요한 시스템의 가장 작은 부분으로 scope하고, 나머지는 전부 CRUD로 두고, *어떤 force*가 complexity toll(복잡성 통행료) 지불을 정당화했는지 기록하는 ADR([[ch-01]]에서)을 작성하라. force를 명명할 수 없다면, 너는 아직 비용을 매기지 못한 pattern을 발견한 것이다 — First Law(제1법칙)에 따르면 이는 네가 그것을 이해하지 못했다는 뜻이다.

> 💡 **쉬운 설명:** "force(힘)"란 요구사항에서 비롯되어 어떤 결정을 정당화하는 압력을 뜻한다. 예를 들어 "감사 기록이 법적으로 필요하다"는 것은 Event Sourcing 쪽으로 미는 force다. 네가 그 force의 이름을 댈 수 없다면, 단지 "멋져 보여서" 패턴을 쓰려는 것이고 — 그건 비용 계산을 안 했다는 신호다.

---

## 1. Where this chapter sits: the top of the consistency arc

[[ch-06]]은 boundary(경계)를 넘기 위한 필수적인 consistency 메커니즘을 확립했다: Helland의 inside-vs-outside data 구분, compensation(보상 동작)으로 event에 의해 연결된 local transaction(지역 트랜잭션)의 sequence로서의 saga, 그리고 at-least-once delivery(최소 한 번 전달) 하에서 신뢰할 수 있는 event 발행을 위한 transactional outbox. 이것들은 선택사항이 아니다. 네 sales agent의 state가 하나 이상의 transactional unit(트랜잭션 단위)에 존재하는 순간, 너는 그것들이 *필요*하거나 아니면 corruption(손상)을 배포하게 된다.

이 챕터는 다르다. CQRS와 Event Sourcing은 그 core 위에 층층이 쌓인 **선택적 power tool**이다. outline은 hexagonal/clean/aggregate를 한 챕터로 병합하고 saga/outbox/inside-outside를 다른 챕터로 병합하지만, CQRS/ES에는 의도적으로 *자신만의* 챕터를 부여한다 — [[young-cqrs-es]]에 담긴 한 가지 이유 때문이다: 이들의 핵심적(load-bearing) 가르침은 이들이 scope되고 거부 가능한(refusable) 추가물이라는 것이며, 이들을 필수 메커니즘과 분리해 두는 것이 reconciliation table(조정 표)이 경고하는 바로 그 conflation(혼동)을 방지한다. 만약 네가 이 챕터를 읽고 "agent를 event-source 해야겠다"고 결론짓는다면, 너는 그것을 잘못 읽은 것이다. 의도된 takeaway는 *거부의 규율(discipline of refusal)*이다: 각 tool이 정확히 무엇을 사고, 무엇을 비용으로 치르며, 어떤 좁은 조건에서 그 거래가 긍정적으로 뒤집히는지를 아는 것이다.

> 💡 **쉬운 설명:** 보통 강의는 "이 패턴을 써라"고 가르치는데, 이 챕터는 정반대로 "언제 *안 써야* 하는지"를 가르친다. 즉 패턴을 아는 것보다 "거부할 줄 아는 절제력"이 진짜 배워야 할 기술이다.

두 tool 모두를 위한 in-process seed(프로세스 내부의 씨앗)는 [[ch-03]]과 [[ddd-aggregates-tactical]]의 aggregate다. Vernon의 rule 4 — aggregate당 하나의 transaction, aggregate 간에는 **domain event**(도메인 이벤트)로 조율 — 는 이미 작은 event-emitting machine(이벤트 방출 기계)이다. CQRS와 Event Sourcing은 네가 그 seed를 event를 first-class(일급 시민)로 만들 만큼 진지하게 받아들일 때 얻게 되는 것이다. 그 계보가 중요하다: 이는 네가 framework를 덧붙여서 이 pattern들을 채택하는 것이 아니라, aggregate의 기존 domain event가 read side의 척추(CQRS)나 storage 그 자체(ES)가 되게 함으로써 채택한다는 뜻이다.

> 💡 **쉬운 설명:** "seed(씨앗)"라는 비유가 핵심이다. aggregate가 이미 domain event를 내보내고 있으므로, CQRS와 ES는 완전히 새로운 무언가가 아니라 그 씨앗을 더 진지하게 키운 결과물이다. 그래서 외부 framework를 끌어오는 게 아니라, 이미 있는 event를 활용하는 방식으로 자연스럽게 도달한다.

---

## 2. CQRS: two models, one bounded context

### 2.1 The definition, stated minimally

Fowler는 이 pattern을 Greg Young에게 귀속시키고 한 문장으로 진술한다:

> "CQRS stands for Command Query Responsibility Segregation. At its heart is the notion that you can use a different model to update information than the model you use to read information." — Fowler ([[young-cqrs-es]])

> "It's a pattern that I first heard described by Greg Young." — Fowler ([[young-cqrs-es]])

그것이 아이디어의 전부다. 그 조상은 CQS — Command Query Separation(명령-질의 분리), 메서드가 state를 변경하거나(command, 아무것도 반환하지 않음) 데이터를 반환하지만(query, 아무것도 변경하지 않음) 결코 둘 다 하지는 않는다는 Bertrand Meyer의 원칙이다. CQRS는 그 원칙을 메서드 수준에서 *model* 수준으로 끌어올린다: 네가 command를 통해 mutate(변경)하는 object/schema와 query를 위해 project(투영)하는 object/schema가 서로 다른 artifact일 수 있도록 허용되며, 서로 다른 shape을 가지고, 어쩌면 서로 다른 storage를 가지며, 비동기적으로 동기화 상태를 유지한다.

> 💡 **쉬운 설명:** CQS는 함수 하나 단위의 규칙("이 함수는 바꾸거나 읽거나 둘 중 하나만")이고, CQRS는 그것을 데이터 모델 통째로 끌어올린 것이다. 즉 "쓰기용 모델"과 "읽기용 모델"을 아예 별개의 테이블/구조로 분리하고, 둘 사이를 백그라운드에서 맞춰준다.

유용한 mental model: invariant(불변식)를 강제하는 normalized write side(정규화된 쓰기 측 — aggregate, transactionally consistent, 변경에 대한 source of truth)와, UI나 API가 필요로 하는 query에 정확히 맞춰진 — read 시점에 join 없이 — 하나 이상의 denormalized read side(비정규화된 읽기 측, 즉 projection). write side는 event를 emit하고(또는 application이 명시적으로 projection을 업데이트하고), read side는 그것을 consume하여 view를 materialize(구체화)한다.

### 2.2 The caution is the load-bearing part

정의는 사소하다. 가르침은 경고이며, excerpt는 그것을 핵심적(load-bearing) 주장으로 표시한다:

> "For some situations, this separation can be valuable, but beware that for most systems CQRS adds risky complexity." — Fowler ([[young-cqrs-es]])

> "CQRS should only be used on specific portions of a system… and not the system as a whole." — Fowler ([[young-cqrs-es]])

> "Many systems do fit a CRUD mental model, and so should be done in that style." — Fowler ([[young-cqrs-es]])

그 세 인용구를 이 챕터의 무게중심으로 읽어라. pattern이 틀린 게 아니다; *scope*가 팀이 잘못하는 지점이다. 전체 시스템에 적용된 CQRS는 architecture-level의 실수다; read와 write가 진정으로 갈라지는 그 하나의 bounded context에 적용된 CQRS는 targeted되고 올바른 결정이다. 이것은 [[ch-02]]의 strategic discipline(전략적 규율)에 대한 tactical analogue(전술적 유사물)다: 너는 하나의 model을 모든 곳에 적용하지 않고, 올바른 boundary에 올바른 model을 적용한다.

> 💡 **쉬운 설명:** 같은 패턴이라도 "어디에 쓰느냐"가 성패를 가른다. 시스템 전체에 CQRS를 깔면 재앙이고, read/write가 정말로 다른 한 부분에만 깔면 정확한 결정이다. 망치 자체는 죄가 없고, 모든 나사를 망치로 박으려는 게 문제다.

### 2.3 Pricing the bet

여기 그 거래가 명시적으로 가격 매겨져 있다 — CQRS가 무엇을 싸게 유지하고, 무엇을 영구적으로 비싸게 만드는지.

| | What it buys (kept cheap to change) | What it costs (made expensive / permanent) |
|---|---|---|
| **Read/write divergence** | read와 write model이 독립적으로 진화한다; 새로운 query shape은 write model의 schema migration이 아니라 새로운 projection이다 | 이제 두 개의 model과 그 사이의 mapping을 유지해야 한다; 모든 write-model 변경은 그것의 projection을 고려해야 한다 |
| **Scaling** | read 경로와 write 경로가 write model을 건드리지 않고 독립적으로 scale된다(read replica, cache, search index) | Operational surface(운영 표면): 일관되게 provision, monitor, back up해야 하는 두 개의 store |
| **Query performance** | denormalized read view가 runtime join 없이 복잡한 query에 답한다 | read view는 projection lag만큼 write view보다 **항상 뒤처져 있다** |
| **Consistency** | — | read side는 **구조적으로 eventually consistent**다 — 추가 작업 없이 방금 완료한 자신의 write를 strongly-consistent하게 read할 수 있는 CQRS 버전은 존재하지 않는다 |

마지막 행이 무는(bites) 것이다. model을 분리하는 순간, 너는 *단일 service 내부에서*, Helland가 데이터가 boundary를 넘는 순간 받아들여야 한다고 말하는 바로 그 staleness(낡음)를 다시 도출해낸 것이다([[ch-06]], [[helland-data-outside-inside]]). CQRS는 outside-data 의미론을 네 자신의 시스템 안으로 자발적으로 내재화한다. 전형적 증상: 사용자가 form을 제출하고, command가 성공하고, UI가 read model을 다시 query하는데, projection이 아직 따라잡지 못했기 때문에 새 record가 아직 거기 없다. 너는 이제 순수 CRUD에는 결코 없던 UX 문제(read-your-writes, 자신의 쓰기를 읽기)를 떠안게 된다. 그 대응책 — command response에서 낙관적으로 projected value를 반환하기, read에 version-stamp 찍기, 또는 projection을 polling하기 — 은 네가 산 divergence와 scaling의 대가로 지불하는 실제 engineering이다.

> 💡 **쉬운 설명:** read-your-writes 문제는 실무에서 가장 자주 사람을 당황시키는 함정이다. "방금 저장했는데 화면을 새로고침하니 안 보인다" — 이게 바로 read 모델이 아직 따라잡지 못한 상태다. 평범한 CRUD에서는 같은 행을 쓰고 같은 행을 읽으니 절대 안 생기는 문제인데, CQRS는 이 시차(projection lag)를 *일부러* 만든다.

그래서: CQRS는 *두 개의 model 비용 더하기 영구적인 read-side staleness가, 하나의 model이 갈라지는 read와 write를 모두 처리하도록 강제하는 비용보다 작다*는 베팅이다. 그 베팅은 복잡한 domain이나 비대칭 scaling 하에서 보상을 받는다. 다른 모든 곳에서는 — 그리고 그곳이 대부분이다 — 진다. 그래서 "most systems should stay CRUD(대부분의 시스템은 CRUD로 남아야 한다)"인 것이다.

### 2.4 Read-your-writes, walked through concretely

eventual-consistency toll은 하나의 request를 그것을 통해 추적해 보기 전까지는 추상적이므로, 여기 팀을 처음 놀라게 하는 정확한 sequence가 있다. agent가 `Deal` write model(normalized table, invariant 강제)과 `DealSummary` read projection(denormalized, dashboard가 필요로 하는 미리 계산된 field를 가진 deal당 한 행)을 가지고 있다고 가정하자.

1. command가 도착한다: `CloseDeal(deal_id=42)`. handler는 `Deal` aggregate를 load하고, invariant check를 적용하며, write store에 `stage = Closed Won`을 쓰고, commit한다. command는 시점 `t₀`에 success를 반환한다.
2. write commit이 `DealClosed` event를 emit한다([[ch-06]]의 outbox를 통해, 그래서 emit이 신뢰 가능하다).
3. projection updater가 `DealClosed`를 consume하고 `DealSummary` 행을 업데이트한다 — 하지만 *비동기적으로* 이것을 하며, `t₁ > t₀`에 완료된다.
4. `t₀`와 `t₁` 사이에, deal 42에 대한 `DealSummary`의 어떤 read든 **이전** stage를 반환한다. write는 성공했는데; read는 동의하지 않는다.

순수 CRUD에서는 이 window가 존재하지 않는다 — 너는 같은 행을 쓰고 읽었으므로, committed write 이후의 read는 항상 그것을 반영한다. CQRS는 별도의 read model의 대가로 그 window를 의도적으로 *만들어낸다*. 세 가지 표준 대응책, 각각 그 자체로 비용:

- **Optimistic return(낙관적 반환).** command handler가 그 response에 새로운 projected value를 직접 반환하여, 즉각적인 UI 업데이트가 뒤처진 read model을 다시 query하지 않게 한다. 비용: command 경로가 이제 read-model shape에 대해 알게 되어, 네가 산 separation을 누설한다.
- **Version-stamped reads(버전 도장이 찍힌 읽기).** command가 version token을 반환한다; client는 그것을 이후의 read에 전달하고, 그 read는 projection이 그 version까지 따라잡을 때까지 block하거나 retry한다. 비용: read latency와 client-side plumbing(배관 작업).
- **Poll/subscribe(폴링/구독).** client가 기대하는 값이 나타날 때까지 projection을 polling하거나(또는 change feed를 subscribe한다). 비용: 추가 round-trip과 UX의 "loading" 상태.

요점은 이것들이 어렵다는 게 아니라 — 일상적인 것들이다 — *CRUD 하에서는 하지 않을 작업*이라는 것이다. 그 delta(차이)가 CQRS의 정직한 가격이며, 그것은 divergence/scaling 이점과 함께 ADR의 Consequences 섹션에 나타나야 한다. ADR에서 이 window를 명명하지 않고 CQRS를 채택하는 팀은 베팅의 가격을 과소평가한 것이다.

> 💡 **쉬운 설명:** 세 가지 대응책의 공통점은 "원래 CRUD였으면 안 했을 추가 작업"이라는 점이다. CQRS의 진짜 가격은 모델 두 개를 만드는 것뿐 아니라, 이 시차를 매번 코드로 메꿔야 한다는 데 있다. ADR에 이 비용을 안 적으면, 나중에 "왜 이렇게 복잡해졌지?"라는 후회가 온다.

---

## 3. Event Sourcing: the log is the source of truth

### 3.1 The definition

Event Sourcing은 직교하는(orthogonal) 아이디어다. 현재 state를 저장하고 각 변경마다 덮어쓰는 대신, 너는 *변경의 sequence*를 저장하고 그 log를 system of record(기록 시스템)로 취급한다.

> "Capture all changes to an application state as a sequence of events." — Fowler ([[young-cqrs-es]])

현재 state는 권위 있게 저장되지 않는다; 그것은 *log에 대한 fold(접기)*다:

> "We can discard the application state completely and rebuild it by re-running the events from the event log on an empty application." — Fowler ([[young-cqrs-es]])

State는 left-fold(왼쪽 접기)가 된다: `state = events.reduce(apply, emptyState)`. event가 truth다; 어떤 "current state" table이든 그저 네가 버리고 다시 계산할 수 있는 cache(snapshot이나 projection)일 뿐이다. 이것은 기본 database model — current row가 truth이고 변경 이력은 존재한다 해도 부차적인 audit log인 — 의 심오한 역전이다.

> 💡 **쉬운 설명:** 보통의 DB는 "현재 값"이 진실이고 변경 이력은 곁다리다. Event Sourcing은 이걸 뒤집는다 — "무슨 일이 일어났는가(event들)"가 진실이고, 현재 값은 그 event들을 처음부터 차례로 적용(fold)해서 계산한 결과일 뿐이다. fold란 통장 거래내역(입출금 기록)을 처음부터 더해서 현재 잔액을 구하는 것과 똑같다.

fold를 구체적으로 만들어보자. CRUD database는 deal 42를 하나의 mutable row로 저장한다: `{id: 42, stage: "Closed Won", amount: 50000}`. 각 update는 제자리에서 덮어쓰고, 이전 값들은 사라진다. event-sourced store는 대신 *변경들*을 보관한다:

```
events for deal 42 (append-only, never updated):
  DealCreated      {id: 42, stage: "Prospecting", amount: 30000}
  DealAmountRaised {id: 42, amount: 50000}
  DealStageChanged {id: 42, stage: "Negotiation"}
  DealStageChanged {id: 42, stage: "Closed Won"}
```

현재 state는 그 sequence에 대한 `apply`의 fold다: empty에서 시작하여, `DealCreated`를 적용해 `{stage: Prospecting, amount: 30000}`을 얻고, `DealAmountRaised`를 적용해 `amount: 50000`을 얻고, 두 stage 변경을 적용하면, `{stage: Closed Won, amount: 50000}`에 도달한다 — CRUD 행이 가지고 있던 것과 같은 값이지만, *저장된* 것이 아니라 *도출된* 것이다. 차이는 그 뒤에 따라오는 모든 것이다: CRUD 행은 중간 사실들을 버렸고; log는 그것들을 보관했다. "deal이 아직 Negotiation에 있었을 때 amount는 얼마였나?"는 CRUD 행으로는 답할 수 없고 log에서는 세 번째 event까지의 fold다. 그 단일 속성 — 이력이 덧붙여진(bolted on) 것이 아니라 본질적(intrinsic)이라는 것 — 이 전체 value proposition(가치 제안)이고, §3.3이 그것의 전체 비용이다.

> 💡 **쉬운 설명:** 핵심 한 줄: CRUD는 "지금 50000원"만 기억하고 "어떻게 50000원이 됐는지"는 버린다. Event Sourcing은 그 과정 전체를 보관하므로 "지난 화요일엔 얼마였지?" 같은 과거 시점 질문에 답할 수 있다. 단, 이 본질적 이력이 곧 비용의 원천이기도 하다(§3.3).

### 3.2 What it buys

이점들은 "the log is the truth(log가 truth다)"로부터 직접 따라온다:

- **공짜 audit log.** Fowler: "event들을 serialize하여 Audit Log를 만드는 것이 쉽다(easy to serialize the events to make an Audit Log)"([[young-cqrs-es]]). 너는 auditing을 *추가*하지 않는다; auditing이 storage model이다. 모든 변경은, 완전한 before/after 맥락과 함께, 본질적으로 기록되는데 그 변경이 *곧* 기록이기 때문이다.
- **Temporal queries(시점 질의).** "어떤 시점에서든 application state를 결정한다(Determine the application state at any point in time)"([[young-cqrs-es]]). log를 timestamp T까지 replay하면 정확히 T 시점의 state를 갖는다. "지난 화요일에 이 deal의 pipeline stage는 어떻게 보였나?"는 네가 보관해두기를 바라는 log로부터의 forensic reconstruction(법의학적 재구성)이 아니라, T까지의 fold다.
- **Replay debugging(재생 디버깅).** 프로덕션 버그는, 고정된 버전의 코드에 대해 그 깨진 state를 만들어낸 정확한 event sequence를 replay함으로써 재현 가능하다. 버그는 더 이상 "우리는 이것이 일어났다고 생각한다"가 아니다; 그것은 "여기 그것을 만들어내는 결정론적 input이 있다"이다.

> 💡 **쉬운 설명:** 세 가지 이점 모두 "이력을 통째로 갖고 있다"는 한 가지 성질에서 나온다. 감사 로그가 공짜로 따라오고, 과거 어느 시점이든 재구성할 수 있고, 버그를 재현할 결정론적 입력이 손에 있다. 단지 마케팅 문구가 아니라 storage model 자체의 직접적 귀결이다.

### 3.3 What it costs — the sharp edges

Fowler의 여기서의 경고는 CQRS 경고보다도 더 무뚝뚝하고, excerpt는 그것을 핵심적(load-bearing)이라고 표시한다:

> "Clearly this stuff can get very messy, don't go down this path unless you really need to." — Fowler ([[young-cqrs-es]])

두 개의 sharp edge(날카로운 모서리)가 그것을 지저분하게 만들고, 둘 다 storage model에 구워 넣어진 *영구적인* 비용이다 — 코스의 척추([[insights]])가 commit하기 전에 식별하라고 말하는 바로 그런 종류의 expensive-to-reverse(되돌리기 비싼) 결정:

1. **Replay 시의 External side-effect.** log를 replay하는 것은 application의 `apply` 로직을 다시 실행한다. event handler가 email을 보냈거나, 카드를 결제했거나, 또는 — sales agent의 경우 — external SaaS API를 호출했다면, naive replay는 그것을 *다시* 한다. 너는 모든 external effect를 gateway(관문)로 감싸서 replay 중에는 억제되고 오직 첫 번째, live processing 시에만 발동되도록 해야 한다. 이것은 일회성 수정이 아니다; 모든 미래의 event handler가 존중해야 하는 제약이다. 한 번 잊으면 replay가 고객에게 이중으로 청구한다.

2. **Historical event의 Schema evolution.** 오늘의 `DealClosed` event는 field A, B, C를 가진다. 다음 분기에 비즈니스가 field D를 추가하고 B의 이름을 바꾼다. 하지만 log는 옛날 shape을 가진 *수년치* 옛 `DealClosed` event를 담고 있고, 그것들은 immutable truth다 — 너는 current-state 행을 `ALTER TABLE`하듯이 그것들을 migrate할 수 없다. 모든 replay는 모든 historical version을 해석할 수 있어야 한다. 너는 결국 event에 version을 매기고 up-caster(옛 event shape을 읽어 현재 것을 만들어내는 함수)를 작성하게 된다. Fowler는 더 넓은 마찰을 언급한다: "모든 변경을 event로 포장하는 것은… 모두가 편안하게 여기는 interface 스타일은 아니다(Packaging up every change… as an event is an interface style that not everyone is comfortable with)"([[young-cqrs-es]]).

> 💡 **쉬운 설명:** 두 sharp edge가 ES를 실무에서 위험하게 만드는 핵심이다. (1) replay는 코드를 *다시 실행*하므로, 옛날 event를 재생할 때 실제 이메일이 다시 나가거나 카드가 다시 결제될 수 있다 — gateway로 막지 않으면 재앙이다. (2) 옛 event는 절대 못 바꾸므로(immutable), schema가 바뀌면 옛 형식을 읽는 변환기(up-caster)를 영원히 유지해야 한다. 둘 다 "한 번 고치고 끝"이 아니라 영구 부담이다.

### 3.4 Pricing the bet

| | What it buys (kept cheap) | What it costs (made expensive / permanent) |
|---|---|---|
| **Audit** | 완전하고 본질적인 audit trail — auditing이 *곧* storage다 | 모든 event는 네가 결코 `UPDATE`로 없앨 수 없는 영구적이고 immutable한 이력이다 |
| **Time travel** | replay-up-to-T를 통한 과거 어느 순간의 state | replay는 모든 historical event version에 걸쳐 올바르게 유지되어야 한다 → 영원히 event versioning + up-caster |
| **Debugging** | 과거 어떤 state든 결정론적 재현 | external side-effect는 영원히 gateway 가능하고 replay-safe해야 한다 |
| **Storage** | append-only write(제자리 update 없음, contention에 우호적) | log가 무한히 증가한다 → snapshot 필요; current state의 read는 fold 비용이 든다(snapshot/projection으로 완화되지만, 그것이 기계장치를 추가한다) |

베팅: *replay-safety 더하기 historical-schema 규율의 비용이 진정한 audit/temporal/replay의 가치보다 작다*. 규제된 domain(금융, 의료) 또는 그 전체 가치가 이력인 domain에서는, 그 베팅이 보상한다. "current value 더하기 `last_modified` 컬럼 더하기 audit table"로 충분한 CRUD record에 대해서는, 보상하지 않는다 — 그리고 "an audit table(감사 테이블)"이 Fowler가 정말 필요하지 않으면 이 길로 가지 말라고 말할 때 암묵적으로 가리키는 더 싼 답이다.

> 💡 **쉬운 설명:** 마지막 문장이 실용적 결론이다. "감사 기록이 필요하다"는 이유만으로는 ES를 정당화하기 부족하다 — 대부분은 평범한 audit table 하나로 충분하기 때문이다. ES는 금융/의료처럼 이력 자체가 시스템의 핵심 가치인 domain에서만 값을 한다.

---

## 4. The myth this chapter exists to kill

### 4.1 The reconciliation row

[[COLLECTION-PLAN]]의 doc-vs-reality reconciliation table에서, 이 챕터에 배정된 myth(통념):

| Popular narrative | What the primary source actually says |
|---|---|
| "CQRS and Event Sourcing go together / are the same thing." | 이들은 *결합되는(compose)* **독립적인** 결정이다. Fowler: CQRS는 "위험한 complexity를 추가한다… 특정 부분에 사용하라… 전체 시스템이 아니라." 어느 것도 다른 것을 요구하지 않는다. |

이 conflation은 블로그 게시물과 컨퍼런스 발표 도처에 있다: "CQRS/ES"가 하나의 하이픈으로 연결된 것처럼 쓰여, 마치 하나를 채택하면 다른 것을 의무화하는 것처럼. 그렇지 않다. excerpt는 명시적이다:

> "CQRS and Event Sourcing are **separate** decisions that *compose*… But **CQRS does not require Event Sourcing, and Event Sourcing does not require CQRS.** Treating them as a package deal is a common, costly conflation. Neither requires microservices either; both work inside a modular monolith." — [[young-cqrs-es]]

### 4.2 The four quadrants — proof they're independent

독립성을 보는 가장 깔끔한 방법은 2×2다. 각 axis는 그 자체의 결정이다; 네 cell 모두 실제적이고 합리적인 architecture다.

|  | **State stored as current state (no ES)** | **State stored as event log (ES)** |
|---|---|---|
| **One model (no CQRS)** | Plain CRUD — 모든 시스템 대부분의 기본값 | Event-sourced single model: event를 append하고, current state로 fold하며, 별도의 read model 없음 |
| **Separate read/write models (CQRS)** | state store 위의 CQRS: normalized table에 write하고, denormalized read table로 project | "CQRS/ES" — 유명한 조합: event store가 write side이고, projection이 read model을 만든다 |

오른쪽 아래 cell이 *유명*하지만 *필수*는 아닌 이유: event store는 자연스럽게 좋은 write side이고(append-only, event가 이미 존재함), event를 read view로 fold하는 projection은 자연스럽게 좋은 read side다 — 그래서 둘은 거의 마찰 없이 결합된다. excerpt가 그 중력적 끌림(gravitational pull)을 담는다:

> "CQRS naturally aligns with event-based architectures." — [[young-cqrs-es]]

하지만 "자연스럽게 align한다"는 "요구한다"가 아니다. 너는 두 개의 relational schema와 zero event로 CQRS를 할 수 있다(CRUD의 오른쪽 위 cell). 너는 별도의 read side 없이 single model을 event-source할 수 있다(왼쪽 아래). 그 유명한 조합은 design space의 네 유효한 지점 중 하나이며, 너는 발표가 "CQRS/ES"라고 말했다고 해서 묶음 처리된 것을 물려받는 대신 네 cell을 의도적으로 선택해야 한다.

> 💡 **쉬운 설명:** 2×2 표가 이 챕터 통념 깨기의 핵심 증거다. "CQRS를 쓴다/안 쓴다"와 "ES를 쓴다/안 쓴다"는 완전히 별개의 축이고, 네 조합 모두 실존하는 정상적 설계다. "CQRS/ES"라는 묶음은 그중 우하단 한 칸일 뿐 — 자연스럽게 어울리지만 강제는 아니다.

### 4.3 And neither requires microservices

myth의 후반부 — 이것들이 service split을 함의하는 distributed-systems pattern이라는 것 — 도 거짓이며, 그것은 코스의 topology 척추([[ch-04]])로 곧장 연결된다. 두 pattern 모두 단일 process 안에서 행복하게 산다. CQRS는 하나의 modular monolith 안의 두 model이다; projection updater는 in-process event handler일 수 있다. Event Sourcing은 네 하나의 database 안의 append-only table이다. 어느 것을 채택하는 것도 service를 split할 이유가 *아니고*, service를 split하는 것도 어느 것을 채택할 이유가 *아니다*. 이 결정들을 네 deployment topology와 독립적인 axis에 두어라, 정확히 [[ch-02]]가 bounded context를 deployment와 독립적으로 유지했던 것처럼.

> 💡 **쉬운 설명:** 사람들은 흔히 "CQRS/ES = microservices"라고 묶어 생각하지만, 둘 다 단일 프로세스(modular monolith) 안에서 완벽히 작동한다. 배포 구조(서비스를 쪼갤지)와 데이터 패턴(CQRS/ES를 쓸지)은 서로 다른 축이므로 따로따로 결정해야 한다.

---

## 5. The decision discipline: refuse by default, name the force

### 5.1 The default is CRUD, stated by the source

이것은 consistency arc에서 중심 메시지가 *하지 마라(don't)*인 유일한 챕터다. 그 추론은 [[ch-01]]([[richards-ford-fundamentals]])에 설치되고 [[insights]]에서 재진술된 First Law로부터 깔끔하게 연쇄된다: 모든 pattern은 네가 어떤 변경을 싸게 유지할지에 대한 베팅이다; 비용을 명명할 수 없다면 너는 그것을 이해하지 못한 것이다. CQRS와 ES에 대해서는 비용이 크고, 영구적이며, 특정적이므로(§2.3, §3.4), 사전 확률(prior)은 채택에 강하게 반대해야 한다. Fowler는 기본값을 직접 제시한다:

> "Many systems do fit a CRUD mental model, and so should be done in that style." — Fowler ([[young-cqrs-es]])

잘 그려진 aggregate([[ddd-aggregates-tactical]]) 위의 CRUD는 세련되지 못한 자들을 위한 fallback이 아니다; 그것은 divergence/audit force가 부재할 때마다 *올바른* 답인데, 모든 것을 싸게 유지하기 때문이다: 변경할 하나의 model, 자신의 write에 대한 strongly-consistent read, replay hazard 없음, 유지할 event-schema museum(이벤트 스키마 박물관) 없음.

> 💡 **쉬운 설명:** CRUD가 "초보용 패턴"이라는 편견을 버려라. divergence나 audit force가 없으면 CRUD가 *가장 똑똑한* 선택이다 — 모델 하나만 관리하고, read-your-writes가 보장되고, replay 위험도 없고, 옛 event 형식을 박물관처럼 유지할 필요도 없으니까.

### 5.2 The forces that flip each bet

오직 *명명된* force가 존재할 때만 채택하라. force를 ADR에 기록하라.

| Tool | Adopt only when this force is present | Stay CRUD when |
|---|---|---|
| **CQRS** | read와 write model이 진정으로 갈라진다(query shape ≠ command shape인 복잡한 domain) **또는** 비대칭 read/write scaling이 독립적인 경로를 요구한다 | 동일한 model이 read와 write를 받아들일 만하게 서빙한다; query와 command shape이 가깝다 |
| **Event Sourcing** | 평범한 audit table이 충족할 수 없는 audit / temporal query / replay-debugging에 대한 실제 필요 | "current value + `last_modified` + audit table"이 요구사항을 커버한다 |

excerpt 자체의 정당한 trigger 목록: "별도의 read/write model로 더 잘 서빙되는 복잡한 domain, 또는 독립적으로 scale된 read와 write 경로를 요구하는 고성능 필요(a complex domain better served by separate read/write models, or high-performance needs that demand independently scaled read and write paths)"([[young-cqrs-es]]). 둘 다 선호가 아니라 *요구사항 안의 force*임에 유의하라 — 이것이 [[ch-01]]의 architecture-characteristics 규율을 여기 적용한 것이다: 요구사항으로부터 critical few(중요한 소수)를 도출하고, capability를 그 자체를 위해 극대화하지 말라.

### 5.3 Scope it to a portion, not the system

네가 채택할 때, §2.2의 scope 규칙은 협상 불가다: "시스템의 특정 부분에 사용하라… 전체 시스템이 아니라(use on specific portions of a system… not the system as a whole)." 구체적으로, 그것은 하나의 bounded context가 CQRS를 갖는 동안 그 이웃들은 CRUD로 남는다는 뜻이다; 하나의 aggregate type이 event-source되는 동안 model의 나머지는 current state를 저장한다. complexity의 폭발 반경(blast radius)은 force가 사는 곳으로 한정된다. 전체 시스템을 event-source하는 팀은 실제로 audit이 필요했던 소수의 entity의 이익을 위해 모든 사소한 CRUD entity를 replay-hazard, schema-museum 부채로 전환한 것이다 — 각 지역적 결정이 지역적으로 유혹적으로 보였던 곳에서조차 시스템 전반에 걸쳐 net-negative(순손실) 거래.

> 💡 **쉬운 설명:** blast radius(폭발 반경)란 "복잡성이 영향을 미치는 범위"다. 패턴을 시스템 전체에 깔면 폭발 반경이 전부가 되고, 정말 필요한 한 부분에만 깔면 그 부분으로 한정된다. 한 곳에서 좋아 보여도 전체로 확장하면 손해라는 게 핵심이다.

### 5.4 Recording the refusal as an ADR

*채택하지 않겠다*는 결정은 채택하겠다는 결정만큼이나 architecturally significant하며, 같은 내구성 있는 기록 — [[ch-01]]([[nygard-release-it]])의 ADR, Nygard가 "미래 개발자와의 대화(a conversation with a future developer)"로 틀 짓는 — 을 필요로 한다. 평범한 CRUD `Deal` table을 응시하는 미래의 엔지니어는 결국 "우리가 분명히 deal 이력에 신경 쓰는데, 왜 이것을 event-source하지 않았지?"라고 물을 것이다 — 그리고 ADR이 없으면 그는 ES를 추가하고 네가 의도적으로 피한 replay hazard를 물려받음으로써 그것을 "고칠" 가능성이 충분히 있다. refusal ADR(거부 ADR)은 그것을 미연에 차단한다. 그 형태는, Nygard의 Context / Decision / Consequences 골격을 사용하여:

```
ADR-014: Deal aggregate uses CRUD + audit table, not Event Sourcing
Status:    Accepted
Context:   We need an audit trail of deal stage/amount changes for sales
           reporting. Event Sourcing would give this intrinsically. However,
           the Deal aggregate's handlers fire external SaaS side-effects
           (Salesforce writes, customer emails); replay-safety for those is
           a permanent, error-prone burden (see §3.3). Audit is the only ES
           force present — no temporal-query or replay-debugging requirement.
Decision:  Store current state (CRUD) plus an append-only deal_audit table
           written in the same transaction as each Deal change.
Conseq.:   (+) No replay hazard against live customer systems; no event-schema
               museum; strongly-consistent reads of our own writes.
           (-) Audit is a parallel artifact we must remember to write on every
               Deal mutation (mitigated: enforced in the aggregate, fitness-
               function-checked later — see ch-09). No free temporal queries;
               if that force later appears, revisit this ADR.
```

이것은 trade-off 척추를 운영 가능하게 만든 것이다: 베팅이 명명되고, 그것을 *뒤집을* force가 기록되며, 미래의 재검토 조건이 명시적이다. ADR은 expensive-to-reverse 결정(storage model)이 어떻게 revisable(재검토 가능)하게 유지되는지다 — 이것이 정확히 ch-09가 fitness function으로 형식화할 evolution 규율이다.

> 💡 **쉬운 설명:** "안 한다"는 결정도 ADR로 기록해야 한다는 점이 직관에 반하지만 중요하다. 기록이 없으면 미래의 동료가 "왜 ES 안 썼지?" 하고 친절하게 "고쳐서" replay 지옥을 다시 불러올 수 있기 때문이다. refusal ADR은 그 미래의 실수를 미리 막는 방어막이다.

---

## 6. Applied to the sales agent (Lina TMR)

학습자의 프로덕션 sales agent — 많은 external SaaS tool API 위에서 작동하는 LLM — 는 정확히 pattern보다 *거부* 규율이 더 중요한 종류의 시스템이다. 두 결정을 그것을 통해 걸어보자.

### 6.1 Where CQRS might earn its place

agent는 ([[ch-02]] 분석으로부터) bounded context들을 가진다: lead/pipeline, conversation, scheduling, CRM-sync. 이들 대부분은 CRUD다: lead가 생성되고, 업데이트되고, 다시 read된다; meeting이 schedule된다. aggregate 위의 CRUD가 올바른데 — divergence가 부재하고 너는 read-your-writes를 원하기 때문이다. 그래야 agent가 방금 변경한 record의 stale view 위에서 결코 추론하지 않는다.

하나의 그럴듯한 CQRS 후보는 **conversation/analytics**다. write model은 "conversation에 turn 하나를 append"이다; agent와 dashboard가 원하는 read shape은 극도로 다르다 — "sentiment가 하락 추세인 모든 open deal," "rep 수준 pipeline velocity," cross-conversation aggregate. 그 query shape들은 append-a-turn write shape에서 날카롭게 갈라지고, read 볼륨(dashboard, agent 자신의 retrieval)이 write 볼륨을 압도한다 — 비대칭 scaling. *그것이* 명명된 force다. 만약 그것이 나타나면, CQRS를 그 하나의 context로 scope하라: normalized conversation write model, 분석적 read를 위한 denormalized projection, 그리고 agent가 대화보다 몇 초 뒤처진 view 위에서 행동하지 않도록 projection lag의 명시적 처리. 다른 모든 것은 CRUD로 남는다.

> 💡 **쉬운 설명:** 여기서 "force를 명명한다"가 실제로 어떻게 보이는지 알 수 있다. conversation/analytics에서는 (1) 읽기 모양과 쓰기 모양이 완전히 다르고 (2) 읽기가 쓰기보다 훨씬 많다 — 이 두 가지가 CQRS를 정당화하는 구체적 force다. 다른 context엔 이런 force가 없으니 그냥 CRUD로 둔다.

### 6.2 Why Event Sourcing is mostly the wrong bet here — and where its sharp edge is fatal

agent의 *자신의* state에 대한 Event Sourcing은 유혹적이고("agent가 내린 모든 결정을 audit하자!") 대부분 틀렸다: `last_modified` 더하기 decision-log table이 replay hazard나 event-schema museum 없이 audit을 준다. audit force는 실제이지만 더 싼 tool이 그것을 충족한다.

그리고 agent는 §3.3의 첫 번째 sharp edge를 *심각하게(acute)* 만든다. agent의 event handler는 무해한 내부 email을 보내지 않는다 — 그것은 **external SaaS API를 호출한다**: Salesforce opportunity 생성, 실제 email 전송, 실제 calendar slot 예약. Event Sourcing 하에서, replay는 `apply`를 다시 실행한다. 만약 그 external call들이 엄격하게 gateway 가능하고 replay-suppress되지 않으면, state를 재구성하거나 버그를 재현하기 위해 agent의 log를 replay하는 것이 *live 고객 시스템에 대해 실세계 side-effect를 다시 발동시킬* 것이다 — opportunity 재생성, email 재전송. external API 위의 agent에게, "replay 시의 external side-effect"는 각주가 아니다; 그것은 잠재적으로 시스템에서 가장 비싼 failure mode다. 그 단일 force는 agent의 core state를 CRUD-plus-audit-table 쪽으로 단호히 밀고 event-sourcing에서 멀어지게 해야 한다.

> 💡 **쉬운 설명:** 이게 sales agent에 ES를 쓰면 안 되는 결정적 이유다. agent의 handler는 진짜 이메일을 보내고 진짜 Salesforce 거래를 만든다. ES에서 replay는 그 handler를 *다시 실행*하므로, 버그 재현하려고 로그를 재생했다가 실제 고객에게 이메일이 다시 나가고 거래가 또 생성될 수 있다. 이건 사소한 부작용이 아니라 시스템 최악의 실패 모드다.

### 6.3 The inside/outside tie-back

이것은 [[ch-06]]에 연결된다: 모든 external SaaS response는 **outside data**다 — immutable, versioned, 어쩌면 stale([[helland-data-outside-inside]]). agent는 SaaS payload를 *결코* event-source할 권위 있는 live state로 취급해서는 안 된다. 만약 네가 agent에서 무언가를 event-source한다면, agent의 *자신의* domain 결정(네가 소유한 inside data)을 source하고, ingest된 SaaS snapshot은 그것이 본래 그러한 versioned, replay-inert(재생-무반응) outside data로 취급하라. outside data를 마치 네 자신의 truth인 것처럼 event log로 fold하는 것은 네 재구성 가능한 state를 vendor의 mutable 현실에 coupling시킬 것이다 — inside/outside 구분이 바로 그것을 방지하기 위해 존재하는 coupling. 네가 내리는 어떤 선택이든 force를 명명하는 ADR([[ch-01]])로 기록하여, 그것을 읽는 미래 엔지니어가 왜 agent가 event-sourced가 아니라 CRUD-with-audit인지 알게 하라.

> 💡 **쉬운 설명:** 핵심 원칙: 외부 SaaS에서 받은 데이터는 "남의 진실"이라 언제든 낡거나 바뀔 수 있으므로, 그것을 내 event log의 진실처럼 저장하면 안 된다. 굳이 event-source 한다면 *내가 소유한* 내부 결정만 저장하고, SaaS 응답은 버전 찍힌 스냅샷으로만 다뤄라. 안 그러면 내 시스템 상태가 외부 vendor의 변덕에 묶여버린다.

---

## 7. The aggregate seed, closed

챕터는 in-process seed를 언급하며 열렸다: Vernon의 rule 4 — aggregate당 하나의 transaction, 그것들 간에는 domain event([[ddd-aggregates-tactical]]). loop를 닫자. 그 단일 규칙은 전체 consistency arc가 자라난 kernel(핵)이다:

- 다른 aggregate와 조율하기 위해 **domain event**를 emit하는 aggregate(rule 4)는, 여러 service에 걸쳐 늘리면, **saga**다([[ch-06]], [[richardson-saga]]).
- 그 domain event를 state 변경과 함께 신뢰성 있게 영속화하는 것은 **transactional outbox**다([[ch-06]], [[transactional-outbox]]).
- 그 동일한 domain event가 *별도의 read model을 구동하게* 하는 것은 **CQRS**다(§2).
- 그 동일한 domain event가 *storage 그 자체가 되게* 하는 것은 **Event Sourcing**이다(§3).

네 개의 pattern, 하나의 seed. 이것이 [[ddd-aggregates-tactical]]이 aggregate를 "coding tip이 아니라 design-altitude(설계 고도) 아이디어"라고 부른 이유다: modular monolith에서 aggregate boundary를 올바르게 잡으면 위의 모든 pattern에 대한 올바른 consistency boundary를 공짜로 갖는다. 그것을 틀리게 잡으면(너무 크게) 너는 모든 downstream pattern이 물려받는 contention(경합)을 구워 넣는다. aggregate는 네가 consistency에 대해 비용을 치르거나 절약하는 곳이다, 단 한 번, 일찍.

> 💡 **쉬운 설명:** 이 챕터의 마무리는 "네 개의 패턴(saga, outbox, CQRS, ES)이 사실 하나의 씨앗(aggregate + domain event)에서 자랐다"는 통합이다. 그래서 aggregate 경계를 처음에 제대로 잡는 게 가장 중요하다 — 그 한 번의 결정이 아래 모든 패턴의 consistency 품질을 좌우하기 때문이다. 너무 크게 잡으면 모든 후속 패턴이 그 경합 문제를 물려받는다.

---

## Where this goes

consistency arc가 완성되었다: [[ch-06]]이 필수 메커니즘(inside/outside data, saga, outbox)을 주었고; 이 챕터가 선택적 power tool(CQRS, ES)과 그것들을 기본적으로 거부하는 규율로 그것을 마무리했다. 여기서부터 코스는 **evolution — 베팅을 재검토 가능하게 유지하기**로 전환한다. Ch-08은 resilience(회복력) 관점을 취한다: 네 architecture가 boundary를 넘는 모든 곳(모든 saga step, agent가 만드는 모든 external SaaS call)은 failure가 들어와 퍼지는 *integration point*(통합 지점)이며, 챕터는 stability toolkit(안정성 도구함) — timeout, circuit breaker, bulkhead, fail-fast — 을 ops 사후 처리가 아니라 폭발 반경을 어디에 그릴지에 대한 *design-altitude* 결정으로 설치한다. 연결 조직: 분산 transaction을 피하기 위해 [[ch-06]]에서 손을 뻗은 동일한 event-driven, async integration은 *그 자체로* resilience 결정인데, 그것이 그 stability pattern들이 그렇지 않으면 방어해야 했을 synchronous coupling을 제거하기 때문이다.

> 💡 **쉬운 설명:** 다음 챕터(ch-08)는 "경계를 넘는 모든 지점에서 장애가 들어오고 퍼진다"는 관점으로 회복력을 다룬다. 흥미로운 연결고리는, ch-06에서 분산 트랜잭션을 피하려고 택한 비동기 event 방식이 사실 회복력 측면에서도 이득이라는 점이다 — 동기적 결합(synchronous coupling)을 애초에 없애므로, circuit breaker 같은 안정성 패턴이 방어할 대상이 줄어든다.
