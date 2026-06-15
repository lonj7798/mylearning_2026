<!-- chapter: ch-06
     track: consistency
     kind: content
     title: Inside vs Outside Data — Sagas and the Transactional Outbox
     deps: [[ch-03]], [[ch-05]]
     sources: [[helland-data-outside-inside]], [[richardson-saga]], [[transactional-outbox]], [[fowler-microservices]]
-->

# 06장 — Inside vs Outside Data: Sagas and the Transactional Outbox

> **핵심 통찰.** 분산 소프트웨어의 모든 consistency pattern(일관성 패턴)은 하나의 물리적 사실에서 비롯된다: *데이터가 service boundary(서비스 경계)를 넘는 순간 lock과 공유 transaction을 잃는다.* Pat Helland는 이로 인해 생기는 두 가지 regime(영역, 체제)에 이름을 붙인다 — **inside data**(내부 데이터: private하고, mutable하고, ACID이며, 진짜 "now")와 **outside data**(외부 데이터: service 간에 오가는 메시지로 immutable하고, versioned되어 있고, identity가 찍혀 있으며, stale할 수 있음)다. 하나의 transaction을 두 service에 걸쳐 감쌀 수 없다는 사실을 일단 받아들이면, 이 챕터의 나머지는 강제된다: cross-service ACID transaction을 **saga**(여러 local transaction의 sequence이며, rollback이 아니라 *compensating* transaction으로 되돌린다)로 대체하고, dual-write 없이 각 step의 event를 durable하게 만들려고 **transactional outbox**(트랜잭셔널 아웃박스)를 사용한다. 이 중 무엇도 "best practice"가 아니다 — 각각은 outside data를 inside data인 척하기를 거부한 데 따른, 값이 매겨진 결과다.

> **가이드라인.** boundary 안에서는 SQL과 ACID를 자유롭게 써라; 어떤 datum(데이터 한 조각)이 떠나는 순간, 그것을 stale할 수 있는 immutable하고 versioned된 snapshot으로 취급하고, 그 staleness(낡음/오래됨)를 없는 척하기보다 그것을 위해 application을 설계하라. 하나의 비즈니스 operation이 여러 service에서 state를 바꿔야 할 때, 2PC(two-phase commit, 2단계 커밋)로 손을 뻗지 마라 — saga를 작성하고, 그것이 치르게 하는 **loss of isolation**(격리성 상실)에 값을 매기고, 그 상실이 만드는 anomaly(이상 현상)들에 대한 명시적 countermeasure(대응책: semantic lock, commutative update, re-read)를 추가하라. 각 step의 event를 신뢰성 있게 emit(방출)하려면, 비즈니스 변경과 *동일한 local transaction에서* 그것을 outbox table에 persist(영속화)한 다음 비동기로 relay(중계)하고, delivery가 at-least-once(최소 한 번)이므로 모든 consumer를 idempotent(멱등)하게 만들어라.

---

## 1. The Root Distinction: Inside Data vs Outside Data

[[ch-02]]장부터 [[ch-05]]장까지는 boundary를 *어디에* 둘지, 그리고 그 boundary 너머로 *어떤 contract(계약)를* 노출할지에 관한 것이었다. 이 챕터는 어떤 boundary든 진짜가 되는 순간 도착하는 청구서에 관한 것이다: 그것을 넘는 것이 consistency 측면에서 무엇을 치르게 하는가. Pat Helland의 "Data on the Outside versus Data on the Inside"(CIDR 2005)는 이 라이브러리 전체에서 가장 깊은 칼날인데, 단지 mechanics만이 아니라 모든 후속 pattern이 존재하는 *이론적 이유*를 주기 때문이다. [[helland-data-outside-inside]]

> 💡 **쉬운 설명:** "boundary를 넘는 것이 치르게 하는 청구서"라는 비유를 풀어 보자. 이전 챕터들은 가게의 벽을 어디에 세울지(boundary 배치)와 그 벽에 어떤 창구를 낼지(contract)를 다뤘다. 이번 챕터는 일단 벽이 진짜로 서고 나면, 한쪽 방의 정보를 다른 방에서 즉시·정확하게 알 수 없게 되는 대가를 다룬다. 그 대가가 곧 consistency 비용이다.

> **Sourcing note.** Helland의 ACM Queue 2020 reprint는 fetch 시 **403**을 반환했다; 아래의 inside/outside/immutability 주장들은 CIDR-2005 PDF, Semantic Scholar, 그리고 "the morning paper" summary를 통해 corroborate(교차 확인)했으며, 이는 [[COLLECTION-PLAN]]의 gap log에 따른 것이다. 인용부호 안의 짧은 구절들은 그 교차 확인 출처들에서 가져온 것이다; 더 긴 특징 서술들은 reprint에서 그대로 옮긴 깊은 verbatim이 아니라 충실한 paraphrase로 취급하라.

### 1.1 Two regimes, one boundary

Helland는 service가 다루는 모든 데이터를 두 종류로 나눈다:

| | **Inside data** | **Outside data** |
|---|---|---|
| Where it lives | 하나의 service에 private함 | service들 *사이의* 메시지 |
| Mutability | mutable함 | immutable함 |
| Transactional? | ACID; lock을 걸 수 있음 | lock 없음, 공유 transaction 없음 |
| Time | "now" — 현재의 truth | snapshot, *보낸 시점 기준으로* 참 |
| Schema | service 자신의 DDL | "각 데이터 항목의 schema는 **versioned**되어 있다" |
| Helland's phrase | "the realm of SQL and SQL's DDL" | "immutable… stable, such that a repeated request is unchanged" |

boundary 안에서는 [[ch-03]]가 지은 세계에 산다: aggregate(애그리거트)는 즉각적인 transactional consistency의 단위이며, Vernon의 rule 4("use eventual consistency outside the boundary")는 이미 local-ACID 세계가 aggregate 가장자리에서 멈춘다고 말해 주었다. Helland는 그 가장자리를 *service* boundary로 일반화하고, *왜* 거기서 멈춰야만 하는지를 설명한다. [[ddd-aggregates-tactical]]

> 💡 **쉬운 설명:** aggregate란 DDD에서 "한 덩어리로 함께 변경되고 함께 일관성을 지켜야 하는 객체들의 묶음"이다(예: 주문 + 주문항목들). [[ch-03]]에서 배운 것은 이 한 덩어리 안에서만 ACID 같은 즉시 일관성을 보장한다는 것이었다. Helland는 그 "한 덩어리 안에서만 lock 가능"이라는 경계선을 service 전체 수준으로 확대해서, service 바깥으로 나가면 lock이 불가능한 이유를 근본부터 설명한다.

### 1.2 The three load-bearing claims

이 챕터 전체는 [[helland-data-outside-inside]]의 세 가지 주장 위에 놓인다:

1. **Services do not share transactions.** "You cannot wrap a transaction around two services." 이것이 service 간 분산 2PC가 막다른 길인 *바로 그* 이유다 — 자율적인 service의 database에 걸쳐 붙잡고 있어도 되는 global lock manager는 존재하지 않는다.
2. **Outside data must be immutable.** "Messages themselves must also be immutable" — 그 내용은 "should never change across retries"여야 한다. immutable하고 identity가 찍힌 메시지는 retry, cache, reorder, replay해도 안전하다; mutable한 메시지는 그렇지 않다. 이것이 outbox와 event-driven flow를 애초에 작동하게 만드는 전제 조건이다.
3. **Outside data may be stale, and that's fine.** boundary 너머로 lock을 걸 수 없기 때문에, 받는 것은 보낸 시점 기준으로 참인 snapshot이다. **application 수준에서의 eventual consistency**(최종 일관성)를 위해 설계하며, staleness를 없는 척하기보다 그것을 reconcile(조정/화해)한다.

순서대로 읽으면, 이 세 주장은 이 챕터와 다음 챕터의 다른 모든 pattern을 *생성한다*. no-shared-transactions는 saga를 강제한다(§3). immutability는 outbox가 메시지를 안전하게 relay하게 해 주는 것이다(§5). possible-staleness는 eventual consistency를 "인프라 디테일"에서 application-design 문제로 바꾸는 설계 제약이다(§2).

이것이 단지 또 하나의 pattern이 아니라 이 라이브러리에서 *왜* 가장 깊은 칼날인지를 명시해 둘 가치가 있다. aggregate boundary([[ddd-aggregates-tactical]]), database-per-service, saga, outbox, CQRS read model([[young-cqrs-es]]), 그리고 immutable domain event는 *독립적으로* 메뉴판에서 골라 담는 발명품이 *아니다* — 그것들은 모두 하나의 사실의 결과다: 데이터가 service boundary를 넘는 순간 lock과 공유 transaction을 잃으므로, 그 데이터는 immutable하고, versioned되고, stale할 수 있는 것으로-받아들여져야 한다. 거꾸로 [[distributed-monolith]](분산 모놀리스)는 정확히, Helland를 *무시하고* cross-boundary 데이터를 계속 inside data처럼 다룰 때 — 공유 database, boundary를 넘는 동기적 "live" read, 분산 lock — 만들어지는 것이다. 그 anti-pattern([[ch-04]]의 주제)은 이 챕터의 관심사와 별개의 failure mode가 아니다; 그것은 이 챕터의 규율이 예방하기 위해 존재하는 바로 그것이다.

> 💡 **쉬운 설명:** 왜 이 한 가지 통찰이 그렇게 중요한가? 위에 나열된 모든 패턴(saga, outbox, CQRS 등)을 따로따로 "암기해야 할 best practice 목록"으로 보면 학습 부담이 크다. 하지만 사실은 전부 "boundary를 넘으면 lock을 잃는다"는 단 하나의 물리적 사실에서 연역되는 결과다. distributed monolith는 그 사실을 부정한 벌, 이 패턴들은 그 사실을 받아들인 대가 — 이렇게 보면 전부 하나의 줄기에서 가지친 것임을 알 수 있다.

### 1.3 Pricing the bet — the regime split itself

inside/outside 분할조차도 공짜 승리가 아니라 trade(거래/맞바꿈)다. 이 선을 긋는 것은 다음을 **싸게 유지한다**: 각 service의 storage를 독립적으로 진화시키는 능력, 메시지를 안전하게 retry하고 replay하는 능력, global lock 없이 service를 scale하는 능력. 그것은 다음을 **비싸게 만든다**: 더 이상 단순한 cross-service 질문을 던져서 transactionally-consistent한 답을 얻을 수 없다 — 모든 cross-boundary read는 잠재적으로 stale하고, 모든 cross-boundary write는 자체적인 consistency 기계장치가 필요하다. 시스템이 진정으로 하나의 consistency boundary 안에 산다면, 이 대가를 치러도 사는 것이 없다; 그것이 바로 이 코스가 modular monolith([[ch-04]], [[fowler-monolith-first]])를 기본값으로 삼고 가능한 한 많은 것을 *inside*에 두는 정확한 이유다.

> 💡 **쉬운 설명:** 핵심은 "boundary를 긋는 것 자체가 비용이 있다"는 것이다. 굳이 service를 쪼개지 않아도 되는 시스템에서 미리 쪼개 두면, 위에서 말한 비싼 비용(stale read, 별도 consistency 장치)만 전부 떠안고 얻는 이익은 없다. 그래서 기본값은 "쪼개지 말고 modular monolith로 안에 두라"이다.

---

## 2. Myth Check: Two False Beliefs This Chapter Kills

[[COLLECTION-PLAN]]의 reconciliation table은 이 챕터를 위해 두 가지 myth(신화/통념)를 표시한다. 둘 다 유혹적인데, 정확히 boundary를 위한 설계를 *회피하게* 해 주기 때문이다.

### 2.1 "Use distributed transactions (2PC) across services"

**서사:** 두 service에 걸친 atomicity가 필요하다면, two-phase commit을 돌려서 ACID를 되찾으면 된다.

**primary source들이 말하는 것:** 그럴 수 없다. Helland는 **services do not share transactions**라고 단정적이다 [[helland-data-outside-inside]]. Richardson은 DB *와* broker에 둘 다 write하는 인접 사례에 대해서도 똑같이 직설적이다:

> "It is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker." — Richardson [[transactional-outbox]]

자율적인 service에 걸친 2PC는 "고급"이 아니라 잘못된 도구다: 그것은 database-per-service 규칙에 의해 당신이 소유하지 않는 시스템들에 걸쳐 lock을 붙잡는 coordinator를 요구한다. 해법은 cross-service ACID transaction을 **saga**로 *대체*하고(§3) event emission을 **outbox**로 안전하게 만드는 것이다(§5) — 가질 수 없는 global lock을 쫓는 대신 lost isolation을 받아들이면서.

> 💡 **쉬운 설명:** 2PC(two-phase commit)는 "여러 데이터베이스에게 먼저 '준비됐나?'를 묻고(phase 1), 전부 OK면 '커밋해!'를 보내는(phase 2)" 분산 트랜잭션 방식이다. 문제는 이걸 하려면 누군가(coordinator)가 모든 시스템의 lock을 동시에 붙잡고 있어야 하는데, 남의 service의 database lock을 우리가 붙잡을 권한이 없다는 것이다. 그래서 service 간에는 원리적으로 불가능하다.

### 2.2 "Eventual consistency is an infra problem"

**서사:** eventual consistency는 message bus / database / platform이 알아서 정리하는 무언가다; application은 모든 것이 strongly consistent한 척할 수 있다.

**primary source들이 말하는 것:** 이 코스가 관심을 두는 design altitude(설계 고도)에서, eventual consistency는 **application-design** 문제다. Fowler는 "Microservice Trade-Offs"에서 명시적이다:

> "Business logic can end up making decisions on inconsistent information." — Lewis & Fowler [[fowler-microservices]]

Helland는 데이터 쪽에서 같은 말을 한다: outside data가 stale할 수 있으므로, 당신의 logic은 staleness를 *예상하도록* 작성되어야 한다 [[helland-data-outside-inside]]. 이것을 infrastructure에 위임할 수 없는데, infrastructure는 어떤 stale read가 무해하고(cache된 display name) 어떤 것이 파국적인지(곧 차감할 잔액) 전혀 모르기 때문이다. staleness가 *어디서* 허용되고 *어디서* reconcile되어야 하는지를 결정하는 것은 architectural decision이다 — 그것은 config 파일이 아니라 당신의 ADR([[ch-01]], [[nygard-release-it]])에 속한다.

> 💡 **쉬운 설명:** "eventual consistency는 시간이 지나면 결국 일치하게 되지만, 그 사이엔 잠깐 안 맞을 수 있다"는 개념이다. 흔한 오해는 "그건 인프라가 알아서 해 주는 거니까 코드는 신경 안 써도 된다"인데, 인프라는 "이 살짝 낡은 값으로 결정해도 되는가"를 판단할 수 없다. 화면에 표시할 이름이 1초 낡은 건 괜찮지만, 출금할 잔액이 1초 낡은 건 재앙이다. 이 구분은 사람(아키텍트)이 ADR에 적어 결정해야 한다. ADR은 Architecture Decision Record, 즉 "왜 이렇게 결정했는지"를 남기는 문서다.

---

## 3. The Saga: Distributed Transactions Without 2PC

§1과 §2가 자리 잡고 나면, saga는 영리한 선택지가 아니다 — 남아 있는 유일한 형태다.

### 3.1 Why sagas exist

전제 조건은 data ownership(데이터 소유권)이다. Database-per-service가 규칙이다:

> "Keep each microservice's persistent data private to that service and accessible only via its API." — Richardson [[transactional-outbox]]

그리고 그 규칙은 즉시 saga가 푸는 문제를 만든다:

> "The Database per Service pattern creates the need for this pattern." — Richardson [[richardson-saga]]

공유 database가 없다는 것은 분산 ACID transaction이 없다는 뜻이다. 그래서 cross-service 비즈니스 operation은 chain(사슬)으로 재구성된다:

> "A saga is a sequence of local transactions. Each local transaction updates the database and publishes a message or event to trigger the next local transaction in the saga." — Richardson [[richardson-saga]]

> 💡 **쉬운 설명:** 각 service가 자기 database를 독점하므로(database-per-service), 하나의 큰 트랜잭션으로 여러 service의 데이터를 한꺼번에 바꿀 방법이 없다. 그래서 "큰 트랜잭션 하나" 대신 "각 service에서 작은 local transaction을 차례로 실행하고, 끝날 때마다 다음 단계를 깨우는 event를 쏘는" 릴레이 방식으로 바꾼다. 이 릴레이 전체가 saga다.

### 3.2 The 1987 origin (thesis, not verbatim)

Richardson이 이 construct(구성물/개념)를 발명한 것은 아니다; 그는 그것을 용도 변경했다. 이 아이디어는 Hector Garcia-Molina와 Kenneth Salem의 "Sagas"(SIGMOD 1987)에서 온다.

> **Sourcing note.** 1987 PDF는 fetch 시 **image-only (no text layer)**였으므로, 다음은 논문에 대한 지식에서 추출한 *논문의 thesis이지 verbatim 인용이 아니다* — [[COLLECTION-PLAN]]과 [[richardson-saga]]에 따라 이 hedge(유보)를 존중하라.

원래 논문은 *하나의* database 안에서의 **long-lived transactions (LLTs)**(장기 실행 트랜잭션)를 위해 saga를 도입했다. LLT는 sub-transaction T1…Tn으로 분할되고, 각각은 그것을 *의미적으로 되돌리는* **compensating transaction**(보상 트랜잭션) C1…Cn과 짝지어진다. 보장은 시스템이 두 가지 깨끗한 형태 중 하나로 끝난다는 것이다: 전체 sequence T1…Tn이 실행되었거나, 아니면 prefix T1…Tj가 실행된 뒤 Cj…C1이 뒤따랐거나 — **결코 멈춰 버린 partial state가 아니다.** 결정적으로, 1987 construct는 이미 isolation을 *완화한다*: 다른 transaction들이 중간 sub-transaction 결과를 관찰할 수 있다. Richardson의 기여는 이 single-DB 도구가 cross-*service* consistency에 딱 맞는 도구임을 알아본 것이다.

이 혈통이 중요한 이유는 saga가 *무엇이 아닌지*를 알려 주기 때문이다. 그것은 rollback이 아니다(되돌릴 global transaction이 없다). compensation은 의미적으로-되돌리는 효과를 만들어 내는 새로운 forward transaction이다: 카드 청구를 row를 지워서 un-charge하는 것이 아니라, `refund()`하는 것이다.

> 💡 **쉬운 설명:** 핵심 오해 하나를 막는 부분이다. saga의 "되돌리기"는 데이터베이스 rollback처럼 흔적 없이 원상복구하는 게 아니다. 이미 commit돼서 외부에 보였던 변경은 지울 수 없다. 대신 "그 효과를 상쇄하는 새로운 작업"을 추가로 실행한다. 결제를 취소하려면 결제 기록을 삭제하는 게 아니라 환불(`refund()`)이라는 새 거래를 한 건 더 만드는 식이다. 그래서 compensation은 "지우기"가 아니라 "반대 방향의 새 forward 작업"이다.

### 3.3 Choreography vs orchestration

saga는 step에서 step으로 전진할 방법이 필요하다. Richardson은 두 가지를 준다:

> "Choreography - each local transaction publishes domain events that trigger local transactions in other services." — Richardson [[richardson-saga]]
> "Orchestration - an orchestrator (object) tells the participants what local transactions to execute." — Richardson [[richardson-saga]]

| | Choreography | Orchestration |
|---|---|---|
| Control | 탈중앙화; service들이 event에 반응함 | 중앙 orchestrator가 step들을 구동함 |
| Coupling | 더 낮음; 더 적은 의존성 | 더 높음, 하지만 flow logic이 한 곳에 산다 |
| Visibility | 전체 flow를 추적하기 어려움 | 프로세스를 보고/디버그할 한 곳 |
| Best when | 단순하고, 자연스럽게 event-driven한 flow | 복잡하고, 상호 의존하는 step이 많을 때 |
| Risk | 분산된 logic, monitoring 부담 | orchestrator가 bottleneck / single point of design gravity(설계 중력이 쏠리는 단일 지점)가 됨 |

이것 자체가 "어느 쪽이 더 나은가" 질문이 아니라 값이 매겨진 bet(내기/선택)이다. **Choreography는 다음을 싸게 유지한다:** 기존 event에 새 reaction을 추가하기(그냥 subscribe — 건드릴 중앙 코드 없음)와 coordinator bottleneck을 피하기. **그것은 다음을 비싸게 만든다:** flow를 *이해하기* — 비즈니스 프로세스가 누가-무엇을-듣는지의 emergent property(창발적 속성)로만 존재하므로, "이 주문이 왜 멈췄지?"를 디버그하려면 여러 service의 log에서 프로세스를 재구성해야 한다. **Orchestration은 다음을 싸게 유지한다:** *프로세스* 자체의 이해와 변경(읽기 쉬운 하나의 state machine이다)과 observability(한 곳이 진행 상황을 보고한다). **그것은 다음을 비싸게 만든다:** coupling — 이제 모든 participant가 orchestrator에 의존하며, orchestrator는 비즈니스 logic을 끌어모아 조용히 분산 시스템의 새로운 monolith가 될 수 있다. [[ch-04]]의 Hard Parts 규칙이 적용된다: size와 centralization은 trade-off 분석의 *output*이지 기본값이 아니다.

> 💡 **쉬운 설명:** 둘의 차이를 비유로 보면 — choreography는 "춤"이다. 지휘자 없이 각 무용수(service)가 음악(event)을 듣고 알아서 다음 동작을 한다. 유연하지만 전체 안무를 한눈에 보기 어렵다. orchestration은 "오케스트라"다. 지휘자(orchestrator)가 각 단원에게 "이제 너 차례"라고 지시한다. 전체 흐름이 한 곳에 적혀 있어 읽기/디버그가 쉽지만, 지휘자에게 모든 것이 의존하고 그가 점점 비대해질 위험이 있다.

> **Interactive — [`figures/saga-choreography-vs-orchestration.html`](figures/saga-choreography-vs-orchestration.html)를 열어라.** choreography와 orchestration 사이를 toggle하고, "Inject failure at Step 2 (Charge)" 또는 "Step 3 (Confirm)"을 설정한 다음 Run을 눌러라. committed된 step들이 초록색으로 켜지고, 실패한 step이 빨강으로 가고, **compensation들이 역순으로 실행되는 것**(보라색)을 지켜봐라. amber(호박색) 배너는 **lost-isolation window**(격리성 상실 구간)를 표시한다 — committed-but-not-yet-final(커밋됐지만 아직 최종이 아닌) step이 이미 다른 모든 transaction에게 보이는 구간이다. 그 window가 §4의 요점 전부이며; 이 figure는 그것이 열리고 닫히는 것을 *볼 수* 있게 해 준다.

### 3.4 The price: no isolation

이것이 learner가 내면화해야 하는 문장인데, 거의 모두가 값 매기기를 잊는 비용이기 때문이다:

> "Lack of isolation (the 'I' in ACID)… means there's risk that the concurrent execution of multiple sagas and transactions can [cause] data anomalies." — Richardson [[richardson-saga]]

고전적 ACID transaction은 A, C, I, D를 준다. saga는 일종의 **atomicity-of-outcome**(결과의 원자성: 모든 step, 또는 깨끗한 prefix-plus-compensation)와 local step별 durability를 주지만 — isolation을 *내던진다*. step 1의 commit과 saga의 최종 해소 사이에서, step 1의 intermediate state는 **다른 모든 이에게 보인다.** 그것이 figure의 amber window다. 구체적인 anomaly들:

- **Dirty reads(더티 리드):** 다른 saga가 당신이 곧 `refund()`할 `charged: $99`를 읽고, 살아남지 못할 청구에 기반해 행동한다.
- **Lost updates / non-repeatable reads(갱신 손실 / 반복 불가능 읽기):** 두 saga가 마지막 재고 단위를 예약하는데, 어느 쪽도 다른 쪽의 in-flight reservation을 보지 못하기 때문이다.

saga는 공짜 transaction이 아니기 때문에, Richardson은 "a saga developer must typically use **countermeasures**"라고 언급한다:

| Countermeasure | What it does |
|---|---|
| **Semantic lock** | in-flight row에 `PENDING` flag를 표시해 다른 saga들이 그것이 최종이 아님을 알게 한다 |
| **Commutative updates** | 순서가 중요하지 않은 operation(예: `debit`/`credit` 델타)을 설계해 동시 saga들이 서로를 덮어쓰지 않게 한다 |
| **Pessimistic view** | 관찰되면 가장 위험한 state가 마지막에 만들어지도록 saga step을 재정렬한다 |
| **Re-read value** | 행동하기 직전에 다시 읽고, 읽은 이후 바뀌었다면 중단한다(optimistic check) |
| **By-value tracking** | 요청의 *비즈니스 risk*에 따라 saga 처리를 route해서 고위험 flow가 더 강한 countermeasure를 받게 한다 |

ADR을 위한 정직한 framing: saga는 isolation을 대가로 cross-service progress를 사 주고, countermeasure들은 step당 추가된 복잡성을 대가로 *일부* isolation을 되사 준다. 각 countermeasure가 어떤 anomaly를 방어하는지 이름을 댈 수 없다면, bet의 값 매기기를 끝내지 못한 것이다(First Law, [[ch-01]], [[richards-ford-fundamentals]]).

> 💡 **쉬운 설명:** ACID의 I(isolation)는 "트랜잭션이 진행 중인 동안 그 중간 상태가 남에게 안 보인다"는 보장이다. saga는 각 step이 끝나는 즉시 commit해서 외부에 보이므로 이 보장을 포기한다. 그래서 "아직 saga가 끝나지도 않았는데 중간 결과를 본 다른 작업이 잘못된 판단을 내리는" 문제(dirty read 등)가 생긴다. countermeasure들은 이 문제를 부분적으로 막는 도구상자다 — 예를 들어 semantic lock은 중간 상태에 "아직 확정 아님(PENDING)" 딱지를 붙여 남이 그걸 믿지 않게 한다.

---

## 4. The Lost-Isolation Window, Concretely

이 window에 속도를 늦춰 머무를 가치가 있는데, naive한 saga 설계가 조용히 데이터를 손상시키는 곳이기 때문이다.

figure의 3-step order saga를 그려 보라: `reserve(item)` → `charge(card)` → `confirm(order)`. step 3이 실패한다고 하자. compensation `refund(card)`와 그다음 `release(item)`이 역순으로 실행되고, 시스템은 consistent한 state로 돌아간다. 여기까지는 안전하다. 하지만 intermediate state가 *언제* 관찰 가능했는지를 고려하라:

```
t0  Inventory: reserved=1   COMMITTED  (locally durable, globally non-final)
t1  Payment:   charged=$99  COMMITTED  (locally durable, globally non-final)
t2  Order:     confirm()    FAILS
t3  Payment:   refund()     compensation
t4  Inventory: release()    compensation
```

**t0과 t4 사이**에, inventory를 읽는 다른 transaction은 그 단위가 사라진 것으로 보고, payment를 읽는 fraud/ledger 프로세스는 $99 청구를 본다. 두 관찰 모두 *locally true(국소적으로 참)*이지만 *globally wrong(전역적으로 틀림)*이다 — saga가 아직 commit하기로 결정하지 않았다. 동시에 접속한 고객은 품절이라는 안내를 받고; downstream의 analytics job은 1초 뒤 환불될 매출을 장부에 올린다. 이것이 정확히, 단일 ACID transaction이라면 그 lock 뒤에 숨겼을 isolation이다.

설계상의 수는 window를 제거하는 것이 아니라(할 수 없다 — 그것은 Helland가 붙잡을 수 없다고 한 global lock을 요구한다) **그것을 bound(한정)하고 label(표시)하는** 것이다. semantic lock은 조용한 dirty read를 명시적인 "이건 PENDING이다, 아직 의존하지 마라" 신호로 바꾼다; 가장 민감한 state가 마지막에 쓰이도록 step 순서를 고르는 것(`pessimistic view`)은 최악의 anomaly에 대한 window를 줄인다. architecture decision은 당신의 domain이 *어떤* anomaly를 용납할 수 없는지, 그리고 어떤 countermeasure에 값을 치를 것인지다. 나머지 모든 것은 받아들인다.

> 💡 **쉬운 설명:** 타임라인을 다시 보면, t0~t4 사이에는 "재고가 빠졌고 $99가 청구됐다"는 상태가 실제로 DB에 commit돼서 누구나 볼 수 있다. 그런데 결국 saga가 실패해 전부 되돌려진다. 즉 "보이긴 했지만 곧 사라질 가짜 사실"을 다른 작업이 진짜로 믿고 행동하면 사고가 난다(엉뚱한 품절 안내, 곧 취소될 매출 집계). 이 위험 구간을 없앨 수는 없으니 — 없애려면 global lock이 필요한데 그건 불가능 — 대신 "지금 이 값은 PENDING이다"라고 딱지를 붙여 남이 속지 않게 하고, 가장 위험한 변경은 맨 마지막에 해서 구간을 짧게 만든다.

---

## 5. The Transactional Outbox: Emitting Events Without a Dual-Write

saga step은 "database를 update하고 **그리고** event를 publish한다"이다. 그 접속사(and)가 함정을 숨기고 있다.

### 5.1 The dual-write problem

> "How to atomically update the database and send messages to a message broker?" — Richardson [[transactional-outbox]]

naive한 sequence — DB row를 commit하고, 그다음 broker에 publish — 는 두 operation *사이에서* crash할 수 있다:

- commit 후, publish 전에 crash → state는 바뀌었지만 **event가 발사되지 않았다**; 다음 saga step이 결코 실행되지 않는다. saga가 non-final state에서 영원히 멈춘다.
- 먼저 publish하고, 그다음 DB commit이 rollback → **결코 일어나지 않은** write에 대해 event가 발사되었다; downstream service들이 유령(phantom)에 기반해 행동한다.

그리고 이것을 2PC로 고칠 수 없는데, (§2.1) "it is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker"이기 때문이다. [[transactional-outbox]]

> 💡 **쉬운 설명:** dual-write란 "두 개의 다른 시스템(DB와 메시지 broker)에 따로따로 쓰기"를 말한다. 둘을 묶을 트랜잭션이 없으니, 첫 번째는 성공하고 두 번째 직전에 프로그램이 죽으면 둘이 어긋난다. DB는 바뀌었는데 알림은 안 나갔거나(다음 단계가 영영 안 깨어남), 반대로 알림은 나갔는데 DB가 롤백돼(없던 일을 남들이 진짜로 믿음) 사고가 난다.

### 5.2 The outbox solution

해법은 event emission을 비즈니스 변경과 *동일한 단일 commit*의 일부로 만드는 것이다:

> "The solution is for the service that sends the message to first store the message in the database as part of the transaction that updates the business entities." — Richardson [[transactional-outbox]]

비즈니스 row를 update하는 동일한 local transaction 안에서, 나가는 event를 **outbox** table에 insert한다. 하나의 commit, 하나의 atomic unit(원자적 단위). 보장:

> "Messages are guaranteed to be sent if and only if the database transaction commits." — Richardson [[transactional-outbox]]

구체적으로, saga step은 두 table을 건드리는 하나의 local transaction이 된다:

```
BEGIN;
  UPDATE opportunities SET stage = 'Closed Won' WHERE id = :id;   -- business change
  INSERT INTO outbox (event_type, payload, created_at)            -- the outside-data event
    VALUES ('OpportunityWon', :payload, now());
COMMIT;   -- one atomic commit: either both rows land, or neither does
```

정확히 하나의 commit이 있으므로, 비즈니스 변경과 intent-to-publish(발행 의도)는 database가 이미 공짜로 주는 row-level atomicity를 공유한다 — 두 번째 시스템도, 2PC도 없다. 그런 다음 별도의 **relay**가 outbox를 읽고 publish한다:

> "Two patterns for implementing the Message relay: The Transaction log tailing pattern [and] The Polling publisher pattern." — Richardson [[transactional-outbox]]

| Relay | How | Keeps cheap | Makes expensive |
|---|---|---|---|
| **Polling publisher** | 한 프로세스가 outbox table에서 보내지지 않은 row를 polling하고 publish한다 | 운영 단순성; 새 인프라 없음 | 추가 DB load + publish latency(일정 간격으로 poll하므로) |
| **Transaction log tailing (CDC)** | DB commit log를 tail(꼬리 추적)하고(예: Debezium) committed된 outbox insert를 publish한다 | 낮은 latency; table에 polling load 없음 | 운영하고 추론할 인프라가 더 많아짐 |

> 💡 **쉬운 설명:** outbox의 핵심 아이디어는 "두 시스템에 따로 쓰는 문제"를 "한 데이터베이스 안에서 두 테이블에 한 번에 쓰는 문제"로 바꾸는 것이다. 비즈니스 변경과 "이 event를 보내라"는 메모(outbox row)를 같은 트랜잭션에서 함께 commit하므로, 둘은 항상 함께 살아남거나 함께 사라진다 — 어긋날 틈이 없다. 실제 메시지 발송은 나중에 relay라는 별도 프로세스가 outbox를 읽어서 처리한다. relay 방식은 두 가지인데, polling(주기적으로 테이블을 들여다봄, 단순하지만 약간 느림)과 CDC(DB의 변경 로그를 실시간으로 따라감, 빠르지만 인프라가 복잡)다.

### 5.3 At-least-once and idempotency

어느 relay든 **at-least-once**로 delivery한다 — relay는 row를 publish하고, sent로 표시하기 전에 crash하고, 재시작 시 republish할 수 있다. 따라서:

> "Consumers must be **idempotent**." — Richardson (Guideline) [[transactional-outbox]]

이것이 정확히, outbox(*data* 관심사)와 idempotency key([[ch-05]]의 *API-contract* 관심사)가 같은 대화인 이유다: ch-05의 contract("at-least-once delivery 하에서 retry가 안전하도록 write를 idempotent하게 만들어라," [[fielding-rest]], [[transactional-outbox]])는 여기서 outbox가 만들어 내는 delivery 의미론에 의해 *요구된다*. idempotency는 outbox의 producer-side 보장의 consumer-side 거울상이다. 그리고 Helland와 loop가 닫히는 것에 주목하라: outbox가 emit하는 event는 outside data *이다* — immutable하고, identity가 찍혀 있고, 재전달해도 안전한 — 이것이 at-least-once-plus-idempotency가 애초에 건전한 유일한 이유다. [[helland-data-outside-inside]]

> 💡 **쉬운 설명:** at-least-once는 "메시지가 최소 한 번은 도착하지만, 같은 게 두 번 이상 올 수도 있다"는 전달 보장이다. relay가 메시지를 보낸 직후 "보냈음" 표시를 하기 전에 죽으면, 재시작 후 같은 메시지를 또 보내기 때문이다. 그래서 받는 쪽(consumer)은 idempotent해야 한다 — "같은 메시지를 두 번 받아도 결과가 한 번 받은 것과 똑같아야 한다"는 뜻이다. 예: "이 주문에 환불 1건 처리"를 두 번 받아도 환불은 딱 한 번만 일어나야 한다. 이것이 [[ch-05]]에서 배운 idempotency key가 데이터 계층에서도 똑같이 필요한 이유다.

### 5.4 Pricing the outbox bet

**싸게 유지한다:** crash 하에서의 correctness — event를 결코 잃지 않고 유령을 발사하지 않으므로, saga가 조용히 멈추거나 갈라질 수 없다. 또한 service들을 진정으로 decoupled하게 유지한다(공유 DB 없음, 2PC 없음). **비싸게 만든다:** 추가 table, 운영하고 monitor할 relay 프로세스, publish latency(polling) 또는 인프라 복잡성(CDC), 그리고 모든 consumer가 idempotent해야 한다는 hard *requirement* — 그것 자체가 공짜가 아니라 설계 작업이다. write가 boundary를 넘어 자신을 알릴 필요가 전혀 없다면, 이 중 무엇도 빚지지 않는다; outbox는 건너는 데 대한 통행료지, 어디서나 내는 세금이 아니다.

> 💡 **쉬운 설명:** outbox는 만능 적용 대상이 아니다. boundary를 넘어 event를 보내야 하는 write에만 통행료처럼 붙는 비용이다. service 내부에서만 끝나는 write라면 이 추가 테이블·relay·idempotency 부담을 질 필요가 없다. "필요한 곳에만 내는 통행료지, 모든 도로에 매기는 세금이 아니다"라는 비유가 그 뜻이다.

---

## 6. Applied to the Sales Agent (Lina TMR)

learner의 프로덕션 sales agent — 많은 외부 SaaS tool API(CRM, email, calendar, ticketing) 위에서 동작하는 LLM — 는, 하나로 배포되든 아니든, 구조적으로 분산 시스템이다. Helland의 distinction은 그것에게 사용 가능한, 가장 큰 레버리지를 가진 단일 boundary 결정이다.

### 6.1 Every external SaaS response is outside data

agent가 Salesforce를 call해서 opportunity를 돌려받거나, Google Sheet row를 읽을 때, **그 응답은 outside data다**: versioned되고, stale할 수 있는 snapshot이며, Salesforce가 보낸 시점 기준으로만 참이다 — agent 내부의 authoritative live state가 *결코* 아니다. [[helland-data-outside-inside]] 설계상의 결과:

- agent는 자신의 **inside model**(자신이 구동하는 대화/딜에 대한 private하고 mutable한 작업 state)을, 자신이 ingest하는 **outside snapshot**과 엄격히 분리해 유지해야 한다. stale한 CRM read를 live truth로 취급하는 것은 agent 버전의 "business logic making decisions on inconsistent information" [[fowler-microservices]]이다 — 예: 다른 rep가 30초 전에 바꾼 deal stage에 기반해 행동하기.
- outside data가 stale할 수 있으므로, agent는 **consequential write(중대한 쓰기) 직전에 re-read**해야 한다(§3.4의 `re-read value` countermeasure): opportunity를 Closed Won으로 표시하기 전에 그것이 여전히 기대하는 stage에 있는지 확인한다.

> 💡 **쉬운 설명:** sales agent에게 "Helland의 inside/outside 구분"이 왜 가장 중요한 결정인가? agent가 Salesforce에서 읽어 온 딜 정보는 "읽어 온 그 순간의 사진"일 뿐, 지금 이 순간의 진실이 아니다(그 사이 다른 영업사원이 바꿨을 수 있다). 그래서 agent는 외부에서 가져온 사진(outside snapshot)과 자기가 진행 중인 작업 메모(inside model)를 절대 섞으면 안 되고, 진짜 중요한 변경(딜을 Closed Won으로 확정)을 하기 직전에는 반드시 다시 읽어 확인해야 한다.

### 6.2 A multi-app action is a saga

하나의 논리적 operation에서 다음을 해야 하는 agent action을 고려하라: Salesforce opportunity를 Closed Won으로 표시하고, 그다음 Gmail로 routing email을 보내고, 그다음 calendar에 follow-up task를 만든다. 이것들은 **공유 transaction이 없는 세 개의 독립 시스템이다** — 정확히 Helland의 no-shared-transactions 주장. 그래서 이 action은 saga *이고*, agent는 그것을 saga로 취급해야 한다:

- **Compensation을 미리 정의하라.** Salesforce가 이미 Closed Won으로 표시된 후 Gmail send가 실패하면, semantic undo는 무엇인가? 종종 외부 side-effect에 대해 *깨끗한 compensation이 없다*(email을 un-send할 수 없고, deal을 un-win하고 싶지 않을 수도 있다) — 그것 자체가 가장 중요한 발견이다. compensation이 불가능한 경우, saga step은 마지막에 순서를 정하거나(`pessimistic view`) state 변경이 아닌 notification으로 만들어야 한다.
- **Lost-isolation window에 값을 매겨라.** "deal marked won"과 "task created" 사이에서, 다른 agent run이나 CRM 안의 사람이 follow-up이 예약되지 않은 won deal을 본다. 그 anomaly는 용납 가능한가? 아니라면, semantic lock(`PENDING_FOLLOWUP` flag)이 되사기(buy-back)다.
- **Choreography vs orchestration은 agent 설계에 매핑된다.** step들을 구동하고 각 tool 결과에 반응하는 LLM planner는, 사실상 **orchestrator**다 — 읽기 쉽고 디버그 가능하지만, 전체 프로세스 logic(그리고 failure 처리)이 집중되는 단일 지점이다. 순수 event-reactive한 tool chain은 choreography일 것이다 — 더 decoupled되지만, multi-app action이 멈췄을 때 추적하기 훨씬 어렵다. failure가 auditable(감사 가능)해야 하는 agent에게는 orchestration bet이 보통 이긴다, 그리고 §3.3의 비용(orchestrator gravity)이 의식적으로 받아들여야 할 대가다.

> 💡 **쉬운 설명:** 여기서 가장 중요한 통찰은 "외부 side-effect는 깨끗하게 되돌릴 수 없다"는 점이다. 이미 보낸 이메일은 회수할 수 없고, 이미 Closed Won으로 만든 딜을 되돌리고 싶지 않을 수도 있다. 그래서 saga의 compensation 이론을 sales agent에 적용할 때는, "되돌릴 수 없는 단계는 맨 마지막에 두거나, 상태 변경 대신 단순 알림으로 바꾼다"는 실전 설계 원칙이 나온다. 그리고 LLM이 단계를 직접 지휘하면 그게 곧 orchestrator라는 점 — 추적·감사가 쉬워서 agent에는 대개 이 방식이 맞다.

### 6.3 The outbox for the agent's own writes

agent가 자신의 결정을 persist하고(예: "I marked deal X won and queued notification Y") 그것들을 다른 internal component에 emit한다면, 그것은 dual-write 문제에 직접 직면한다: 결과 action을 emit하지 않은 채 결정을 기록해서도 안 되고, record가 rollback된 action을 emit해서도 안 된다. outbox pattern이 변경 없이 적용된다 — 그리고 그것이 함의하는 at-least-once delivery는 relay가 구동하는 모든 외부 tool call이 **idempotent하거나 idempotency key로 guard되어야** 함을 의미한다(§5.3, [[ch-05]]), 그래야 relay retry가 routing email을 두 번 보내지 않는다. 이것이 learner의 이전 benchmark 작업이 측정했던 cross-application coordination skill이다; 여기서 그것은 평가되는 것이 아니라 *설계되는* 것이다.

> 💡 **쉬운 설명:** 마지막 줄의 대비가 핵심이다 — learner는 이전 코스(automation-bench)에서 "agent가 여러 앱에 걸쳐 작업을 조율하는 능력"을 *측정*하는 법을 배웠다. 이번 챕터에서는 같은 능력을 *직접 설계*한다. outbox와 idempotency가 바로 그 cross-app 조율을 안전하게 만드는 설계 도구다 — relay가 재시도로 같은 이메일을 두 번 보내는 사고를 막아 준다.

---

## Where this goes

이 챕터는 *필수* consistency mechanics를 설치했다: inside/outside data, saga와 그것의 lost isolation, 그리고 dual-write 없이 각 step의 event를 emit하는 outbox. 이것들은 boundary가 진짜가 되는 순간 당신이 빚지는 pattern들이다. [[ch-07]]은 그 위에 layer되는 **optional power tools**(선택적 강력 도구)로 향한다 — **CQRS**("use a different model to update information than the model you use to read information")와 **Event Sourcing**("capture all changes to application state as a sequence of events"). pivot은 다시 그 척추(spine)다: ch-06의 pattern들이 boundary를 넘음으로써 강제되는 반면, ch-07의 것들은 *기본적으로 거부된다* — Young의 load-bearing(하중을 견디는/핵심적인) 경고는, 둘 다 "add risky complexity"하며 구체적인 force(읽기/쓰기 scaling의 비대칭, 진짜 audit/replay 필요)가 요구하는 시스템의 특정 부분에만 속한다는 것이다. in-process seed는 이미 당신 손에 있다: [[ch-03]]에서 domain event를 emit하는 aggregate는 event sourcing으로부터 immutable-event log 하나 거리다 — 하지만 무언가 구체적인 것이 그것을 강제할 때까지 그 길을 거부해야 한다. [[young-cqrs-es]]

> 💡 **쉬운 설명:** 이 챕터(ch-06)와 다음 챕터(ch-07)의 결정적 차이는 "강제 vs 거부"다. ch-06의 saga·outbox는 boundary를 넘는 순간 *피할 수 없이 빚지는* 필수 도구다. 반대로 ch-07의 CQRS·event sourcing은 강력하지만 위험한 복잡성을 더하므로 *기본값은 "쓰지 않는다"*이고, 비대칭 scaling이나 진짜 감사/재생 요구 같은 구체적 압력이 있을 때만 시스템의 일부에만 도입한다. 그리고 [[ch-03]]에서 aggregate가 이미 domain event를 내보내고 있으니, event sourcing까지는 "그 event들을 immutable log로 쌓기" 한 걸음 거리다 — 다만 그 한 걸음을 강제하는 이유가 생기기 전에는 일부러 멈춰 있어야 한다.
