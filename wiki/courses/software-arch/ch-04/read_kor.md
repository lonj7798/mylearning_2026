<!-- chapter: ch-04
     track: topology
     kind: content
     title: Monolith, Modular Monolith, Microservices, and the Quantum
     deps: [[ch-03]]
     sources: [[fowler-microservices]], [[newman-building-microservices]], [[fowler-monolith-first]], [[decompose-by-business-capability]], [[conway-team-topologies]], [[richards-ford-hard-parts]], [[distributed-monolith]]
-->

# 04장 — Monolith, Modular Monolith, Microservices, and the Quantum

> **핵심 통찰.** Topology(토폴로지, 시스템을 어떤 배포 단위로 나누느냐의 구조)는 이 코스 전체에서 가장 되돌리기 비싼 결정이며, 바로 그것이 이 주제가 *첫 번째*가 아니라 *네 번째*에 오는 이유다: deployment boundary(배포 경계)를 그릴 권리는 language boundary([[ch-02]])와 boundary 내부 구조([[ch-03]])를 먼저 확보한 다음에야 얻어진다. Microservices(마이크로서비스)는 크기가 아니라 *independent deployability*(독립 배포 가능성)로 정의된다 — 그리고 그 속성은 그냥 주어지는 것이 아니라 *사서* 얻는 것이다. 그것을 얻기 위해 distribution(분산), eventual consistency(최종 일관성), operational overhead(운영 부담)를 비용으로 지불하며, 이 분야가 MicroservicePremium(마이크로서비스 프리미엄)이라고 부르는 복잡도 임계점을 넘어서야 비로소 이득을 본다. 올바른 기본값은 **modular monolith**(모듈러 모놀리스)다: 내부적으로는 엄격한 module boundary(모듈 경계)를 가지면서 단일 배포 단위로 묶인 형태인데, monolith 안에서 잘못된 boundary는 *refactor*(리팩터)지만 service들 사이에서 같은 잘못된 boundary는 *migration*(마이그레이션)이기 때문이다. Service 크기는 trade-off 분석(granularity disintegrators(세분화를 밀어내는 힘) 대 integrators(묶어두는 힘))의 **결과물**이지, 결코 어림짐작 규칙(rule of thumb)이 아니다 — 그리고 치명적인 실패 모드인 **distributed monolith**(분산 모놀리스)는 distribution tax(분산 세금)는 지불하면서 coupling(결합)은 그대로 유지할 때 얻게 되는 것이다.

> 💡 **쉬운 설명:** 이 챕터의 핵심을 한 문장으로 압축하면 "마이크로서비스는 '작아서' 마이크로서비스가 아니라 '혼자 따로 배포할 수 있어서' 마이크로서비스"라는 것이다. 그리고 그 독립 배포 능력은 공짜가 아니라 분산 시스템의 온갖 비용을 지불하고 사는 것이며, 그 비용을 충분히 상쇄할 만큼 시스템이 복잡해지기 전까지는 손해라는 점을 기억하면 된다.

> **가이드라인.** Topology를 *고르지* 말고, *도출하라*(derive). 모든 새로운 bounded context(바운디드 컨텍스트)는 깨끗한 내부 seam(접합선)을 가진 modular monolith 안에서 시작하라. 어떤 seam을 자체 service로 추출(extract)하기 전에, architecture-quantum 테스트(이것이 *자체 데이터를 가진* 독립 배포 가능하고 cohesive한 단위인가?)를 실행하고, 그것을 떼어내려는 disintegrator들과 붙들어두려는 integrator들을 명시적으로 열거하라 — disintegrator들이 명백히 이길 때만 추출하라. Conway's Law(콘웨이의 법칙)를 무시할 민담(folklore)이 아니라 지렛대(lever)로 사용하라(원하는 architecture를 유도하도록 팀을 형성하라). 실제로 microservices를 달성했는지를 가리는 리트머스 테스트는 잔인할 정도로 단순하다: service B를 배포하지 않고 service A를 배포할 수 있는가? 그렇지 않다면 당신이 가진 것은 network latency(네트워크 지연)가 붙은 monolith일 뿐이다.

---

## 1. The Decision That Comes Fourth on Purpose

이 코스의 세 챕터는 싸게 옮길 수 있는 boundary들에 관한 것이었다. [[ch-02]]는 *language*(언어) 위에서 잘랐고(bounded contexts), [[ch-03]]은 boundary *안에서* 잘랐다(hexagonal/clean core + aggregate). 둘 다 되돌릴 수 있다: 잘못된 것으로 판명된 context는 다시 모델링하는 작업이고; 새는(leaky) port는 refactor다. 이 챕터는 그 중심 결정이 진정으로 되돌리기 비싼 첫 번째 챕터이며 — 코스는 의도적으로 이 결정을 미뤄왔다.

> 💡 **쉬운 설명:** 여기서 "싸게 옮길 수 있다"는 말은, 잘못 그어도 코드 안에서 고치면 끝난다는 뜻이다. 반대로 이번 챕터의 결정(network 경계 긋기)은 한 번 그으면 여러 팀이 동시에 움직이고 데이터를 옮기고 계약을 깨야 하는 대공사가 되므로 "비싸다". 그래서 일부러 마지막에 다룬다.

그 미룸 자체가 교육의 전부다. [[insights]]에 명시된 이 코스의 척추(spine)는 *architecture란 되돌리기 비싼 결정들의 집합*이라는 것이다. 애플리케이션의 일생에서 가장 되돌리기 비싼 결정은 network boundary를 어디에 자르느냐인데, 잘못 자를 때의 비용이 코드 변경이 아니라 여러 팀이 조율하고, 데이터를 마이그레이션하고, contract를 깨는 재배포(re-deployment)이기 때문이다. 그래서 boundary 단계 전체([[ch-02]]..[[ch-03]])는 당신이 *싼* boundary들을 먼저 올바르게 잡고, 비싼 것은 미루고, 베팅할 만큼 충분히 좋은 모델을 가지고 여기에 도착하도록 존재한다.

### 1.1 What this chapter is, and is not

이것은 *topology* 챕터이지, distributed-systems-internals(분산 시스템 내부 구현) 챕터가 아니다. 우리는 세 가지 shape — monolith, modular monolith, microservices — 중에서 고르고 있으며, 주어진 seam이 어떤 shape을 원하는지 알려주는 하나의 테스트(architecture quantum)와 하나의 힘-분석(disintegrators 대 integrators)을 배우고 있다. 일단 *분리한 다음에* saga가 어떻게 consistency를 보존하는지, outbox가 어떻게 메시지가 전송됨을 보장하는지 — 그것들은 [[ch-06]]이다. 여기서는 *애초에* 분리할지 *여부*와 분리한다면 *어디서* 할지를 결정한다.

---

## 2. Microservices Defined by Deployability, Not Size

이 챕터가 만드는 가장 중대한(consequential) 교정은 정의 그 자체에 대한 것이다. 대중적인 멘탈 모델은 "microservices = 작은 service들"이다. Primary source(1차 출처)들은 다른 것에 동의한다: 정의적 속성은 **independent deployability**다.

### 2.1 The two definitions, side by side

Lewis & Fowler의 정전적(canonical) 진술, [[fowler-microservices]]에서 그대로 인용:

> "The microservice architectural style is an approach to developing a single application as a suite of small services, each running in its own process and communicating with lightweight mechanisms, often an HTTP resource API." — James Lewis & Martin Fowler, *Microservices* (martinfowler.com/articles/microservices.html, 2014)

그들이 monolith와 대비해 그리는 대조 — 역시 [[fowler-microservices]]에서 — 는 진짜 차별점을 짚는다: monolith는 "a single unit"이며 거기서는 "any changes to the system involve building and deploying a new version of the server-side application"(시스템에 대한 어떤 변경이든 서버 측 애플리케이션의 새 버전을 빌드하고 배포하는 것을 수반한다)이다. 위 정의에서 *size*(크기)라는 단어는 거의 미끼(distraction)에 가깝다. service를 microservice로 만드는 것은 그것이 *자기만의 시계에 맞춰* 출하된다(ships on its own clock)는 점이다.

> 💡 **쉬운 설명:** "ships on its own clock"는 "남 눈치 안 보고 자기 일정대로 배포한다"는 뜻이다. monolith는 한 줄만 고쳐도 전체를 다시 빌드해서 통째로 내보내야 하지만, microservice는 그 service만 따로 내보낼 수 있다. 크기가 작은 게 본질이 아니라 이 "따로 배포" 능력이 본질이라는 것이다.

Newman은 이것을 훨씬 더 날카롭게 진술한다. [[newman-building-microservices]](책; thesis 추출, O'Reilly excerpt와 그의 강연으로 corroborate됨 — fetch된 URL이 *아님*)에서, 그의 정의는 "independently releasable services modeled around a business domain"(business domain을 중심으로 모델링된 독립적으로 릴리스 가능한 service들)이며, 핵심을 지탱하는(load-bearing) paraphrase는:

> "Independent deployability is the single most important principle." — Sam Newman, paraphrased from *Building Microservices* 2e

여기서의 attribution(출처 표기) 규율을 주목하라: 그 문장은 깨끗한 인용 부호 안에 마치 verbatim fetch인 것처럼 넣어진 것이 아니라, 책 thesis의 attributed *paraphrase*(출처를 밝힌 의역)로 렌더링되어 있는데, excerpt가 그렇게 표시하고 있기 때문이다. Newman의 요점은 cosmetic(겉치레)이 아니라 causal(인과적)이다: independent deployability는 좋은 속성들의 부수 효과가 아니라 그것들의 *원인*이다. 그것을 추구하면 "forces loose coupling, well-defined contracts, and stable interfaces"(loose coupling, 잘 정의된 contract, 안정적인 interface를 강제한다)([[newman-building-microservices]]). 당신은 loose coupling을 먼저 얻은 다음 deployability를 얻는 것이 아니다; deployability에 헌신하면 그것이 coupling 규율을 *강제*한다.

> 💡 **쉬운 설명:** 인과의 방향이 핵심이다. "결합도를 낮추면 → 따로 배포할 수 있게 된다"가 아니라, "따로 배포할 수 있게 만들겠다고 결심하면 → 어쩔 수 없이 결합도를 낮추고 계약을 명확히 하게 된다"는 것이다. 독립 배포라는 목표가 나머지 좋은 습관들을 강제로 끌어낸다는 뜻이다.

### 2.2 The two disciplines that buy deployability

[[newman-building-microservices]]는 independent deployment을 실제로 가능하게 만드는 두 가지 타협 불가능한(non-negotiable) 것을 지목한다:

| Discipline | What it means | What breaks if you skip it |
|---|---|---|
| **Information hiding** | service는 구현 세부사항(database, 기술 선택, 내부 workflow)을 숨기면서 API를 통해 동작을 노출한다 — Parnas의 module-level 규칙을 service level로 끌어올린 것. | 내부 구현이 새어나가면(공유 테이블, 노출된 schema), consumer들이 그것에 coupling되고 independent deployment은 죽는다. |
| **Data ownership** | "Each microservice must own its data. Shared databases create hidden coupling and destroy independent deployability."(각 microservice는 자기 데이터를 소유해야 한다. 공유 database는 숨은 coupling을 만들고 independent deployability를 파괴한다.) (책 thesis, [[newman-building-microservices]]) 상호작용은 명시적 API나 event를 통해 일어나며, 결코 다른 service의 database를 통하지 않는다. | 공유 DB는 *암묵적이고 버전 관리되지 않는 contract*다; 당신의 테이블을 바꾸면 다른 service가 조용히 깨진다. |

> 💡 **쉬운 설명:** Information hiding은 "방 안을 보여주지 말고 문(API)만 통해 거래하라"는 규칙이다. Data ownership은 "각자 자기 데이터베이스를 따로 가지고, 남의 DB를 직접 들여다보지 말라"는 규칙이다. 두 service가 같은 DB 테이블을 공유하면, 한쪽이 테이블 구조를 바꾸는 순간 다른 쪽이 말없이 고장 나기 때문에 사실상 따로 배포가 불가능해진다.

이 둘은 Richardson이 transaction(트랜잭션) 각도에서 도달하는 것과 동일한 결론이다(Database per Service, [[ch-06]]에서 미리 보임) — Newman은 단지 *deployability* 각도에서 거기에 도착할 뿐이다.

### 2.3 The nine characteristics — read as a cost ledger

[[fowler-microservices]]는 Lewis & Fowler로부터 아홉 가지 특성을 나열한다. trade-off 고도(altitude)에서 읽으면, 그중 여럿은 기능(feature)이 아니다 — 그것들은 *당신이 대가를 지불해야 하는 전제조건(prerequisite)*이다:

1. **Componentization via services** — out-of-process(프로세스 외부) 컴포넌트; 그 보상은 independent deployment(§2.1의 요점)이다.
2. **Organized around business capabilities** — cross-functional(교차 기능) 팀이 "broad-stack implementation"(UI + storage + collaboration)을 소유한다. 이것은 Conway's-Law 귀결(§5)이다.
3. **Products not projects** — "you build it, you run it"(만든 사람이 운영한다); 제품을 그 전체 수명 동안 소유한다. (코드가 아니라 *조직* 차원의 헌신이다.)
4. **Smart endpoints and dumb pipes** — 로직은 service 안에, 배관(plumbing)은 단순하게 유지; "as decoupled and as cohesive as possible"(가능한 한 decoupled되고 cohesive하게).
5. **Decentralized governance** — service마다 올바른 기술을 고른다("Node.js for a reports page… C++ for a gnarly near-real-time component? Fine"(리포트 페이지에는 Node.js… 까다로운 near-real-time 컴포넌트에는 C++? 괜찮다)).
6. **Decentralized data management** — 각 service가 자기 database를 소유; polyglot persistence(다중 언어/저장소 영속성). 이것은 [[ch-06]]의 모든 consistency 비용의 씨앗이다.
7. **Infrastructure automation** — CI/CD, 자동화된 테스트와 배포는 *있으면 좋은 것이 아니라 전제조건*이다. 이것은 혜택이 아니라 세금 항목(tax line)이다.
8. **Design for failure** — "any service call could fail due to unavailability"(어떤 service call이든 가용성 부족으로 실패할 수 있다); 당신은 그것을 견뎌내야 한다(Netflix Simian Army). 또 다른 세금 항목 — 그리고 [[ch-08]]이 존재하는 이유 전부다.
9. **Evolutionary design** — service는 "replaceable rather than evolved"(진화되기보다 교체 가능한) 것이다; 이해가 자라남에 따라 boundary를 refactor하라.

특성 6, 7, 8은 microservices가 *premium*(프리미엄)을 가지는 이유다: per-service 데이터, 자동화, 또는 failure tolerance(장애 내성)를 건너뛸 수 없다. 그것들이 입장료(entry fee)다.

> 💡 **쉬운 설명:** "cost ledger(비용 장부)로 읽으라"는 말이 이 절의 열쇠다. 아홉 가지를 자랑거리 목록으로 보면 안 되고, "이걸 하려면 반드시 6번, 7번, 8번 비용을 내야 한다"는 청구서로 읽어야 한다는 것이다. 각 service마다 별도 DB, 완비된 CI/CD 파이프라인, 장애를 견디는 설계를 갖추지 않으면 microservices를 제대로 한 게 아니다.

---

## 3. Pricing the Bet: The MicroservicePremium

이것이 챕터의 중심 trade-off이므로, 명시적으로 가격을 매겨라. [[fowler-microservices]]의 "Microservice Trade-Offs" 글은 Fowler 자신의 말로 benefit/cost 장부(ledger)를 제공한다:

| Benefit | Fowler's words | The cost you pay for it | Fowler's words |
|---|---|---|---|
| Strong module boundaries | "reinforce modular structure… important for larger teams" | Distribution | "remote calls are slow and… always at risk of failure" |
| Independent deployment | "autonomous… less likely to cause system failures" | Eventual consistency | "everyone has to manage eventual consistency" |
| Technology diversity | "mix multiple languages… data-storage technologies" | Operational complexity | "need a mature operations team" |

요약하는 주장, [[fowler-microservices]]에서 그대로:

> "There is a Microservice Premium: microservices impose a cost on productivity that can only be made up for in more complex systems." — Martin Fowler

그리고 eventual-consistency 비용은 구체적으로 *application-level*(애플리케이션 수준)인데, 이것이 이 설계 코스가 신경 쓰는 부분이다 — 역시 [[fowler-microservices]]에서 그대로:

> "Business logic can end up making decisions on inconsistent information." — Martin Fowler

그 한 문장이 [[ch-06]]으로 가는 다리(bridge)다: eventual consistency는 "플랫폼 팀이 처리하는 인프라 문제"가 아니다. 그것은 당신의 domain logic에 떨어지는 *설계* 문제다. boundary를 건넌다는 것은 당신의 코드가 이제 stale data(오래된 데이터)에 대해 행동할 수 있다는 뜻이며, 당신은 그것을 위해 설계해야 한다.

> 💡 **쉬운 설명:** 여기가 많은 사람이 오해하는 지점이다. "최종 일관성? 그건 인프라/DB 담당자가 알아서 해주겠지"라고 생각하기 쉽지만, 실제로는 "고객의 잔액이 방금 바뀌었는데 내 service는 아직 옛 값을 보고 결정을 내린다" 같은 일이 *비즈니스 로직 코드 안에서* 벌어진다. 즉, 설계자가 직접 "오래된 데이터를 볼 수 있다"는 가정 위에서 코드를 짜야 한다는 뜻이다.

저자들 자신의 망설임(hedge) 자체가 곧 thesis다. [[fowler-microservices]]에서:

> "We write this with cautious optimism." — Lewis & Fowler

그들은 microservices가 우월하다고 부르기를 명시적으로 거부한다. trade는 upgrade가 아니다.

### 3.1 The bet, stated as keep-cheap / make-expensive

trade-off 척추(spine)를 존중하기 위해, 여기 코스의 고정된 형식으로 microservices 베팅을 적는다:

- **Keeps cheap to change(변경을 싸게 유지하는 것):** 하나의 capability를 *고립된 상태에서* 배포하고, 스케일링하고, 기술을 교체하는 것; 한 팀이 전역 릴리스를 조율하지 않고 출하할 수 있다. 변경의 폭발 반경(blast radius)이 하나의 service로 줄어든다.
- **Makes expensive(비싸게 만드는 것):** service boundary를 *건너는* 모든 것 — transaction(당신은 ACID를 잃는다, [[ch-06]]), synchronous call(당신은 latency + partial failure를 물려받는다, [[ch-08]]), published contract에 대한 schema 변경(조율된 multi-team migration, [[ch-05]]). 그리고 그것은 *모든* 변경이 — 그 변경이 distribution으로부터 이득을 보든 안 보든 — 상시(standing) 운영 세금(CI/CD, observability, failure tolerance)을 지불하게 만든다.

당신은 *예상되는* 변경이 첫 번째 목록에 의해 지배되고 두 번째를 거의 건드리지 않을 때에만 이 베팅을 한다.

> 💡 **쉬운 설명:** "keep-cheap / make-expensive"는 이 코스가 모든 패턴을 평가하는 고정 양식이다. "이 선택이 무엇을 싸게 만들고 무엇을 비싸게 만드는가?"를 항상 두 줄로 적는다. microservices는 '한 부분만 따로 손보기'를 싸게 만드는 대신 '경계를 넘나드는 모든 작업'을 비싸게 만든다. 그래서 앞으로 할 변경이 주로 한 부분 안에서 끝나는 종류라면 이득이고, 자꾸 경계를 넘나드는 종류라면 손해다.

---

## 4. MonolithFirst and the Modular-Monolith Default

여기가 챕터에서 가장 중요한 doc-vs-reality(문서 대 현실) 교정이다.

### 4.1 The myth, named

> **Myth (from the reconciliation table in [[COLLECTION-PLAN]]):** "Microservices are the modern best practice; monoliths are legacy."(microservices가 현대적 모범 사례이고; monolith는 레거시다.)

이것은 거짓이며, primary source는 단호하다. [[fowler-monolith-first]]에서, 그대로:

> "Almost all the successful microservice stories have started with a monolith that got too big and was broken up." — Martin Fowler

> "Almost all the cases where I've heard of a system that was built as a microservice system from scratch, it has ended up in serious trouble." — Martin Fowler

그 논리는 코스 thesis를 다시 진술한 것이다. boundary가 어려운 부분이며, 그것은 *초기에* 가장 어렵다 — [[fowler-monolith-first]]에서:

> "Even experienced architects working in familiar domains have great difficulty getting boundaries right at the beginning." — Martin Fowler

그리고 일단 service들을 가지면, 잘못된 boundary는 비싸다: service들 간에 기능을 refactor하는 것은 monolith에서보다 "much harder"(훨씬 더 어렵다)([[fowler-monolith-first]]). 이것이 챕터에서 가장 중요한 단 하나의 비대칭(asymmetry)이다:

> **Inside a monolith a bad boundary is a refactor. Across services the same bad boundary is a migration.**(monolith 안에서 나쁜 boundary는 refactor다. service들 사이에서 같은 나쁜 boundary는 migration이다.)

> 💡 **쉬운 설명:** 이 비대칭이 챕터 전체의 실용적 결론을 떠받친다. 같은 "선을 잘못 그었다"는 실수라도, monolith 안에서는 함수 몇 개 옮기는 리팩터로 끝나지만, service들로 쪼갠 뒤라면 API 계약을 다시 협상하고 데이터를 옮기고 여러 팀이 동시에 배포하는 대공사가 된다. 그러니 boundary를 신뢰하기 전에는 쪼개지 말라는 것이다.

명시적 권고, [[fowler-monolith-first]]에서 그대로:

> "Start a new application as a monolith initially, even if you think it's likely that it will benefit from a microservices architecture later on." — Martin Fowler

### 4.2 The modular monolith — the strictly-dominant default-when-unsure

**modular monolith**([[fowler-monolith-first]])는 단일 배포 단위는 유지하되 엄격한 내부 module boundary를 강제한다: module마다 하나의 schema/namespace, in-process interface(프로세스 내부 인터페이스)를 통한 통신, 어떤 module도 다른 module의 테이블에 손을 뻗지 않음. 그것은 microservices의 *조직적* 혜택(명확한 ownership, 강제된 cohesion) 대부분을 포착하면서 *distribution* 세금은 전혀 내지 않는다. 그리고 그것은 이상적인 준비 무대(staging ground)다 — 일단 module seam이 깨끗하면, 실제 service로의 추출은 고고학적(archaeological)이라기보다 기계적(mechanical)이 된다.

> 💡 **쉬운 설명:** "고고학적이 아니라 기계적"이라는 표현이 재밌다. seam이 더러우면(서로 얽혀 있으면) 추출할 때 "이 코드가 누구 거지? 왜 여기랑 연결돼 있지?"를 파헤치는 발굴 작업이 된다. seam이 깨끗하면 그냥 그 module를 들어내서 별도 service로 감싸기만 하면 되는 단순 작업이 된다. modular monolith는 미리 깨끗하게 정리해두어 나중 추출을 쉽게 만드는 전략이다.

베팅, 코스의 형식으로:

- **Keeps cheap to change(변경을 싸게 유지하는 것):** *boundary 그 자체*(그것은 in-process refactor다); 더해서 cross-service 트래픽에 대한 distribution 세금 0, eventual-consistency 세금 0, 상시 운영 세금 0을 낸다.
- **Makes expensive(비싸게 만드는 것):** 단일 module의 independent deployment과 independent scaling — conversation engine만 따로 스케일링할 수 없다; 전체 단위를 재배포한다. 또한 seam을 강제하기 위해 (network가 아니라) *규율(discipline)*에 의존하며, 규율은 강제 없이는 썩는다 — 이것이 정확히 [[ch-09]]의 fitness function(피트니스 함수)의 일이다.

distributed monolith(§7)와 비교했을 때, modular monolith는 당신이 확신하지 못할 때 *엄격히 지배한다(strictly dominates)*: 동일한 coupling 규율이 요구되지만, 당신은 아무것도 얻지 못하면서 distribution 세금을 내는 것을 피한다.

> 💡 **쉬운 설명:** "strictly dominates(엄격히 지배한다)"는 게임 이론 용어로, "어떤 상황에서도 다른 선택보다 나쁘지 않고 어떤 상황에서는 더 낫다"는 뜻이다. 확신이 없을 때 modular monolith는 distributed monolith가 요구하는 규율은 똑같이 요구하면서도, 쓸데없는 분산 비용은 안 내므로 항상 더 나은 선택이라는 것이다.

---

## 5. Where to Cut: Business Capability, Subdomain, and Conway

만약 당신이 *정말로* 분리한다면, seam이 개수보다 더 중요하다. 이 챕터의 두 개념이 "어디서"에 답한다.

### 5.1 Cut on the slowest-changing structure

[[decompose-by-business-capability]](Chris Richardson, microservices.io)에서, 그대로:

> "A business capability is a concept from business architecture modeling. It is something that a business does in order to generate value." — Chris Richardson

> "Define services corresponding to business capabilities." — Chris Richardson

그리고 이것이 *올바른* seam인 이유, 그대로:

> "Stable architecture since the business capabilities are relatively stable." — Chris Richardson

이것은 코스 thesis의 decomposition(분해) 재진술이다. 당신은 되돌리기 비싼 boundary를 시스템에서 *가장 적게* 변할 부분에 헌신한다. 조직은 요동치고(churn), 기술은 요동치고, 화면은 요동친다 — 비즈니스가 근본적으로 *하는 일*은 천천히 변한다. 거기서 잘라라.

> 💡 **쉬운 설명:** 핵심 직관은 "가장 안 변하는 곳에 가장 비싼 선을 그어라"다. 회사 조직도나 사용하는 기술 스택, UI 화면은 자주 바뀌지만, "주문을 받는다, 결제를 처리한다, 배송한다" 같은 비즈니스가 본질적으로 하는 일은 거의 안 바뀐다. 잘못 그으면 비싸게 고쳐야 하는 경계니까, 잘 안 바뀌는 안정적인 축 위에 그으라는 것이다.

capability 경로(outside-in, business architecture로부터)와 DDD-subdomain 경로(inside-out, domain model과 그것의 [[ch-02]] bounded context로부터)는 실무에서 수렴한다: 잘 찾아진 bounded context는 보통 *그 자체로* business capability다([[decompose-by-business-capability]]).

### 5.2 The anti-decomposition: never split by technical layer

고전적인 실수, [[decompose-by-business-capability]]에서 지목됨: technical layer(기술 계층)로 분리하기 — "a UI service, a logic service, a data service"(UI service, logic service, data service). 그러면 *모든* 비즈니스 변경이 모든 service를 건드리며, network 위에서 full coupling을 복원한다. 그것이 distributed monolith(§7)로 가는 직통 진입로(on-ramp)다.

> 💡 **쉬운 설명:** 흔한 함정이라 꼭 기억하자. "프론트엔드 service, 비즈니스 로직 service, 데이터 service"처럼 기술 층으로 나누면 그럴듯해 보이지만, "할인 기능 추가" 같은 기능 하나를 바꿀 때 세 service를 다 고쳐야 한다. 결국 네트워크 너머로 강하게 결합된 채로 셋이 항상 같이 움직이게 되어, 분산 모놀리스로 직행한다.

### 5.3 Conway's Law: the boundary is socio-technical

> **Myth (from [[COLLECTION-PLAN]]):** "Conway's Law is folklore."(Conway's Law는 민담이다.)

그것은 민담이 아니다; 그것은 원래의 1968년 논문의 thesis다. [[conway-team-topologies]]에서, 그대로:

> "Any organization that designs a system (defined broadly) will produce a design whose structure is a copy of the organization's communication structure." — Melvin Conway, 1968

Fowler의 메커니즘([[conway-team-topologies]]에서): "software coupling is enabled and encouraged by human communication"(소프트웨어 coupling은 사람 간 소통에 의해 가능해지고 조장된다)이며, 따름정리(corollary)는 "the modular decomposition of a system and the decomposition of the development organization must be done together"(시스템의 modular decomposition과 개발 조직의 decomposition은 함께 이루어져야 한다)이다.

세 가지 대응이 있다([[conway-team-topologies]]): 그것을 **ignore**(무시)하기(어차피 우연히 일어난다), 그것을 **accept**(수용)하기(가지고 있는 communication path에 architecture를 정렬), 또는 **Inverse Conway Maneuver**(역 콘웨이 책략) — 원하는 architecture를 *유도*하기 위해 의도적으로 팀을 재구성하는 것, "particularly effective with microservices organized around business capabilities"(business capability를 중심으로 조직된 microservices에 특히 효과적). 그 책략이 지렛대다: 조직 구조는 당신이 겪어내는 제약이 아니라, 당신이 돌릴 수 있는 손잡이(knob)다.

> 💡 **쉬운 설명:** Conway's Law는 "시스템 구조는 결국 그것을 만든 조직의 소통 구조를 닮는다"는 관찰이다. 예를 들어 백엔드 팀과 프론트엔드 팀이 따로 있으면 시스템도 백엔드/프론트엔드로 갈리기 쉽다. Inverse Conway Maneuver는 이걸 거꾸로 이용해서, "이런 architecture를 원하니까 팀부터 그 모양으로 짠다"는 전략이다. 즉, 팀 배치가 곧 설계 도구라는 것.

### 5.4 Team Topologies: cognitive load is the real constraint

Skelton & Pais(*Team Topologies*, 2019 — 책, framework 추출; teamtopologies.com이 corroborate)는 [[conway-team-topologies]]에서 운영적 정련(refinement)을 공급한다. 네 가지 팀 유형 — **stream-aligned**(기본값; 하나의 가치 있는 flow를 소유), **enabling**(일시적으로 다른 팀의 역량을 끌어올림), **complicated-subsystem**(깊은 전문가 복잡성을 소유), **platform**("reduce cognitive load"(인지 부하 감소)를 위한 내부 self-service) — 그리고 세 가지 상호작용 모드(**collaboration**, **X-as-a-Service**, **facilitating**).

핵심을 지탱하는 아이디어는 headcount(인원 수)가 아니라 **cognitive load**(인지 부하)다([[conway-team-topologies]]): 팀이 자신의 책임 전체를 머릿속에 담을 수 있도록 팀 boundary를 그어라. 한 팀의 인지적 단위를 쪼개거나, 두 팀의 단위를 융합하는 service boundary는 — 종이 위에서 boundary가 아무리 깨끗해 보이든 *상관없이* — "will be fought daily"(매일 싸움이 날 것이다). 이것이 decomposition이 순수하게 기술적인 것이 아니라 socio-technical(사회-기술적) trade-off인 이유다.

> 💡 **쉬운 설명:** cognitive load(인지 부하)는 "한 팀이 동시에 머릿속에 담고 책임질 수 있는 양"이다. 팀 규모(몇 명인가)가 아니라 이 부담이 진짜 제약이라는 게 핵심이다. 한 service가 두 팀에 걸쳐 있거나, 한 팀이 너무 많은 service를 떠안으면 매일 마찰이 생긴다. 그래서 경계는 기술만이 아니라 "사람이 감당 가능한가"까지 함께 따져야 한다(socio-technical).

---

## 6. The Architecture Quantum and the Granularity Forces

이제 테스트와 힘-분석 — 둘 다 [[richards-ford-hard-parts]](*Software Architecture: The Hard Parts*, 2021; 책, thesis 추출, O'Reilly/Amazon 설명이 corroborate)에서.

### 6.1 The stance: no best practices

그 책의 전체 전제는, [[richards-ford-hard-parts]]에서의 verbatim 의미로, distributed architecture의 어려운 부분들이 "difficult problems… with no best practices that force you to choose among various compromises"(여러 타협 중에서 선택을 강요하는, 모범 사례가 없는 어려운 문제들)라는 것이다. 그 책은 "how to think critically about the trade-offs involved with distributed architectures"(distributed architecture에 관련된 trade-off에 대해 비판적으로 사고하는 법)를 가르치고 "techniques to help you discover and weigh the trade-offs"(trade-off를 발견하고 저울질하는 데 도움이 되는 기법들)를 준다. 이것은 First Law([[ch-01]])의 운영화(operationalization)다: 비용을 이름 붙일 수 없다면, 당신은 그 패턴을 이해하지 못한 것이다.

### 6.2 The architecture quantum — the boundary test

**architecture quantum**([[richards-ford-hard-parts]])은 **independently deployable**하고, **high functional cohesion**(높은 기능적 응집)을 가지며, synchronous connascence(동기적 연관성)가 그 *안에는* 있되 가로질러서는 없고 — 결정적으로 — 자신의 **own data**(자체 데이터)를 포함하는, 가장 작은 단위다.

"own data" 조항이 이 개념의 힘 전부다. database를 공유하는 두 "service"는 independent하게 배포할 수 없으므로, 그들은 둘이 아니라 *하나의* quantum이다. 따라서 quantum은 형식적인(formal) distributed-monolith 탐지기다: **service가 아니라 quanta를 세어라.** service 박스를 세 개 그렸는데 그것들이 하나의 database를 공유한다면, 당신은 하나의 quantum과 세 개의 배포 골칫거리(deployment headache)를 가진 것이다.

> 💡 **쉬운 설명:** architecture quantum은 "진짜로 따로 떼어낼 수 있는 최소 단위"라고 보면 된다. 핵심 판별법은 "자기 데이터를 따로 가지고 있는가?"다. service 다이어그램에 박스를 세 개 그렸어도 셋이 같은 DB를 쓰면, 셋은 사실 하나의 덩어리(quantum 1개)다. 그래서 "박스 개수가 아니라 quantum 개수를 세라"는 말이 distributed monolith를 잡아내는 공식 도구가 된다. connascence는 "한쪽을 바꾸면 다른 쪽도 같이 바꿔야 하는 결합"을 가리키는 용어인데, 이 결합이 단위 안에는 있어도 단위 경계를 넘어서는 안 된다는 뜻이다.

### 6.3 Granularity: disintegrators vs integrators

> **Myth (from [[COLLECTION-PLAN]]):** "Pick service size by a rule of thumb (e.g. fits in two pizzas / N lines of code)."(어림짐작 규칙으로 service 크기를 골라라(예: 피자 두 판으로 먹일 수 있는 팀 / N줄의 코드).)

primary source의 교정, [[richards-ford-hard-parts]]에서: 크기를 두고 논쟁하지 말고, *힘을 열거하라*. 크기는 입력이 아니라 분석의 **결과물(output)**이다.

| Granularity **disintegrators** (forces to split *smaller*) | Granularity **integrators** (forces to keep *together*) |
|---|---|
| 서로 다른 **scalability / throughput**(확장성/처리량) — 한 부분이 다른 부분의 10배 용량이 필요함 | database **transaction**이 두 부분에 걸쳐야 함(가장 강한 integrator) |
| **Fault isolation**(장애 격리) — 한 부분이 실패해도 다른 부분을 끌어내려서는 안 됨 | 긴밀한 **data dependency**(데이터 의존성) — 서로의 데이터를 끊임없이 읽음 |
| 서로 다른 **security / access**(보안/접근) 요구사항 | 둘 사이의 무거운 **chatty workflow / orchestration**(수다스러운 workflow/오케스트레이션)(network 왕복이 지배하게 됨) |
| 별개의 **code volatility**(코드 변동성) — 한 부분이 훨씬 더 자주 변함 | 함께 변하는 **shared code**(공유 코드) |
| 별도의 **team ownership**(팀 소유권)(Conway 연결, §5.3) | |

당신은 disintegrator가 integrator를 명백히 능가할 때에만 quantum을 분리한다. 이 단일 framework가 *두* 가지 실패 모드 모두에 대한 치료법이다: premature microservices(disintegrator가 약했는데도 분리함)와 distributed monolith(integrator가 강했는데 — 특히 공유 transaction이 — 그런데도 분리함).

> 💡 **쉬운 설명:** 이 표가 챕터의 실전 도구다. 어떤 두 부분을 떼어낼지 말지 고민될 때, "떼라고 미는 힘(disintegrators)"과 "붙어 있으라고 당기는 힘(integrators)"을 양쪽에 적어놓고 저울질한다. 가장 강한 붙드는 힘은 "두 부분이 하나의 DB 트랜잭션으로 묶여야 하는 경우"다 — 이걸 무시하고 쪼개면 ACID 보장을 잃고 distributed monolith가 된다. 결론: 크기를 먼저 정하지 말고, 힘을 따진 결과로 크기가 따라 나오게 하라.

> **Use the figure now.** [`figures/granularity-balance.html`](figures/granularity-balance.html)을 열어 양팔 저울(two-pan scale) 위의 힘들을 토글해보라. disintegrator를 추가함에 따라 추천이 monolith → modular monolith → microservice로 기우는 것을 지켜보고, 분리하는 *동시에* shared-database 토글을 켜는 순간 발화하는 명시적인 *distributed-monolith 경고*를 주목하라 — 그것이 §7의 함정을 당신이 trigger할 수 있는 상태로 렌더링한 것이다.

---

## 7. The Distributed Monolith: Paying the Tax, Buying Nothing

> **Myth (from [[COLLECTION-PLAN]]):** "Split into microservices and you get loose coupling for free."(microservices로 분리하면 loose coupling을 공짜로 얻는다.)

이것은 챕터에서 가장 위험한 myth인데, *조용히 그리고 늦게* 실패하기 때문이다. [[distributed-monolith]]에서, 정의: "so tightly coupled and interdependent that they behave like a monolithic application, defeating the core benefit of adopting microservices"(너무 긴밀하게 coupling되고 상호 의존적이어서 monolithic 애플리케이션처럼 행동하며, microservices 채택의 핵심 혜택을 무너뜨리는) service들의 집합 — "a monolith that just happens to communicate over HTTP instead of function calls"(function call 대신 우연히 HTTP로 통신하게 된 monolith).

### 7.1 The four tells

[[distributed-monolith]]에서:

1. **Deployment dependencies**(배포 의존성) — service들이 함께 릴리스되어야 한다. 분리를 정당화했던 *그 하나의* 속성인 independent deployability를 잃었다.
2. **Synchronous coupling**(동기 결합) — 하나의 요청이 async event 대신 실시간 blocking call의 사슬을 통해 부채꼴로 퍼진다(fans out). 어떤 링크의 latency나 outage(중단)든 전체 사슬을 멈춘다.
3. **Shared database**(공유 database) — 두 service가 같은 schema를 읽고/쓰는데, 이는 *암묵적이고 버전 관리되지 않는 contract*다. 한 service의 테이블을 바꾸면 다른 것이 조용히 깨진다.
4. **Cascading failures**(연쇄 장애) — 긴밀한 runtime coupling은 하나의 느린 dependency가 전체 workflow를 끌어내린다는 것을 의미한다([[ch-08]]이 멈추기 위해 존재하는 그 failure mode).

> 💡 **쉬운 설명:** "조용히 그리고 늦게 실패한다(silently and late)"가 이 함정의 무서운 점이다. 쪼개고 나서 한동안은 잘 도는 것처럼 보이다가, 한참 뒤 "왜 prompt 한 줄 바꾸는데 5개 service를 같이 배포해야 하지?", "왜 한 service가 느려지니까 전부 멈추지?" 하고 뒤늦게 정체가 드러난다. 위 네 가지 징후(같이 배포해야 함, 동기 호출 사슬, DB 공유, 연쇄 장애)가 보이면 distributed monolith를 의심하라.

### 7.2 Why it is "worst of both worlds"

[[distributed-monolith]]에서의 verbatim 요약:

> "This network-based modularity gives you all the pain of distributed systems without the independence that makes microservices worthwhile."

학습자가 늘 가지고 다녀야 할 3자(three-way) 비교([[distributed-monolith]]에서):

| Topology | Distribution tax | Deploy independence | Verdict |
|---|---|---|---|
| Monolith | **None** | None | 너무 커지거나 팀이 너무 많아지기 전까지는 괜찮음 |
| Well-cut microservices | High | **Bought** | MicroservicePremium을 넘어서면 가치 있음 |
| **Distributed monolith** | **High** | **None** | 세금은 내고, 아무것도 못 삼 — 함정 |
| Modular monolith | None | None (but seams are clean) | 확신이 없을 때 strictly-dominant한 대안 |

[[distributed-monolith]]의 리트머스 테스트가 챕터의 한 줄 요약이다: **if you cannot deploy service A without also deploying B, you do not have microservices — you have a monolith with network latency.**(service B를 함께 배포하지 않고 service A를 배포할 수 없다면, 당신은 microservices를 가진 것이 아니다 — network latency가 붙은 monolith를 가진 것이다.) decomposition을 축하하기 전에 coupling부터 고쳐라.

> 💡 **쉬운 설명:** 이 표를 외워두면 좋다. monolith는 세금도 안 내고 독립성도 없지만 작을 때는 괜찮다. 잘 쪼갠 microservices는 세금은 비싸도 독립성을 '샀'으니 충분히 커지면 가치가 있다. distributed monolith는 세금만 내고 독립성은 못 사는 최악이다. modular monolith는 세금도 안 내고 독립성도 없지만 경계가 깨끗해서, 헷갈릴 때 가장 안전한 선택이다.

---

## 8. Applied to the Sales Agent (Lina TMR)

학습자의 프로덕션 시스템은 Lina TMR이다: 다수의 외부 SaaS tool API(CRM, email, calendar, ticketing) 위에서 작동하는 LLM agent. 이 챕터의 기계장치(machinery)를 그 위에 돌려보자.

### 8.1 Default topology: modular monolith, by the book

[[ch-02]]로부터 agent의 bounded context들은 그럴듯하게는 *lead/pipeline*, *conversation*, *scheduling*, *CRM-sync*이다. MonolithFirst 판결([[fowler-monolith-first]])이 직접 적용된다: domain이 아직 학습되는 중이므로(agent는 프로덕션에 있지만 진화 중이다), 그 context들 사이의 boundary는 아직 network를 가로질러 동결할 만큼 신뢰할 수 없다. 각 context를 하나의 배포 가능한 agent 안의 **module**로 만들고, 깨끗한 in-process interface와 module당 하나의 schema를 가져라. 그러면 잘못된 seam은 migration이 아니라 refactor다.

> 💡 **쉬운 설명:** Lina TMR은 학습자가 실제로 운영하는 영업 자동화 LLM agent다. 결론은 단순하다: 아직 도메인 경계가 확실하지 않으니, 네 개 영역(lead/pipeline, conversation, scheduling, CRM-sync)을 따로 service로 쪼개지 말고 한 덩어리 안의 module로 두라는 것이다. 경계가 틀리면 그냥 코드 안에서 고치면 되니까(refactor), 미리 쪼개서 마이그레이션 지옥에 빠지지 말라는 조언.

### 8.2 Run the granularity forces on each seam

§6.3을 정직하게, seam별로 적용하라:

| Seam | Disintegrators present? | Integrators present? | Verdict |
|---|---|---|---|
| **Conversation** engine | 다른 scalability(LLM-bound, 폭발적); fault isolation(멈춘 conversation이 sync job을 죽여서는 안 됨); 높은 code volatility(prompt가 매일 변함) | 거의 없음 — 대부분 event를 방출할 뿐 | 부하가 요구한다면 첫 추출의 가장 강한 *후보* |
| **CRM-sync** | 불안정한 vendor로부터의 fault isolation(Conway: 어쩌면 다른 팀) | lead/pipeline과의 긴밀한 data dependency; chatty | vendor의 불안정성이 fault isolation을 강제하지 않는 한 pipeline과 *함께* 유지 |
| **Scheduling** | 약함(Mild) | conversation 결과와의 transactional 연결 | 함께 유지 — integrator(공유 transaction)가 가장 강한 hold다 |

요점은 답이 아니다; 요점은 *크기가 힘-분석에서 떨어져 나왔다(fell out)*는 것이며, 정확히 [[richards-ford-hard-parts]]가 처방하는 대로다 — "모든 것을 microservices로 쪼개라"는 반사작용에서가 아니라.

> 💡 **쉬운 설명:** 이 표가 §6.3 도구를 실제 시스템에 적용한 본보기다. 예컨대 conversation engine은 LLM 때문에 부하가 들쭉날쭉하고(다른 scalability), prompt가 매일 바뀌고(높은 volatility), 다른 부분과 데이터를 별로 안 나누니(integrator 약함) 떼어낼 후보 1순위다. 반대로 scheduling은 conversation 결과와 한 트랜잭션으로 묶여야 해서(가장 강한 integrator) 붙여둔다. 핵심은 "몇 개로 쪼갤까"를 먼저 정한 게 아니라, 힘을 따진 결과로 크기가 자연히 결정됐다는 점이다.

### 8.3 The trap to avoid, named for this system

Lina TMR이 distributed monolith가 되는 가장 가능성 높은 방식([[distributed-monolith]]): 누군가 agent를 *conversation service*와 *CRM service*로 "쪼개"지만, 둘 다 같은 Postgres 테이블을 읽게 두고(tell #3, shared database) agent loop 안에서 서로를 synchronous하게 호출하게 둔다(tell #2). 이제 느린 CRM vendor가 모든 conversation을 멈추고(tell #4, [[ch-08]]이 다루는 cascading failure), CRM-sync를 재배포하지 않고는 prompt 변경을 출하할 수 없다(tell #1). 당신은 0의 독립성에 대해 full distribution 세금을 내고 있을 것이다. architecture-quantum 테스트가 이것을 *종이 위에서* 잡아낸다: shared DB ⇒ 하나의 quantum ⇒ 실제로 두 개의 service가 아님.

> 💡 **쉬운 설명:** 이 단락은 §7의 함정을 Lina TMR에 구체적으로 대입한 것이다. conversation과 CRM을 service로 나눴다고 착각하지만 같은 DB를 쓰고 서로 동기 호출하면, 네 가지 징후가 한꺼번에 나타나면서 분산 비용만 다 내고 독립성은 0이 된다. 다행히 quantum 테스트로 미리 알 수 있다: "둘이 같은 DB를 쓰네? 그럼 quantum 1개, 즉 사실은 service 1개"라고 설계 단계에서 바로 판정된다.

### 8.4 Conway, for a small team

만약 Lina TMR이 하나의 작은 팀에 의해 만들어진다면, [[conway-team-topologies]]는 technology가 아니라 *cognitive-load* 제약이 boundary를 정한다고 말한다. 한 팀은 자신의 cognitive load가 용량을 초과하지 않고는 네 개의 microservices를 independent하게 운영할 수 없다 — architecture가 "fought daily"(매일 싸움이 날 것)일 것이다. 하나의 stream-aligned 팀에게, modular monolith는 단지 안전한 기본값이 아니다; 그것은 Conway-correct(콘웨이적으로 올바른)한 것이다.

> 💡 **쉬운 설명:** 작은 팀 하나가 microservice 4개를 운영하려면 머릿속에 담아야 할 게 너무 많아져 인지 부하가 터진다. 그래서 §5.4의 cognitive-load 논리에 따르면, 작은 팀에게 modular monolith는 "안전해서" 고르는 게 아니라 "조직 구조상 그게 맞는 architecture라서" 고르는 것이다(Conway-correct).

---

## Where This Goes

이 챕터는 deployment boundary를 자를지 *여부*와 *어디서* 자를지를 결정했고, 그 절단을 하나의 베팅으로 가격 매겼다. 당신이 자르는 순간 — 또는 modular monolith가 외부 SaaS API와 대화하는 순간에도 — 당신은 **contract**(계약)를 만들며, published contract는 service가 소유하는 가장 되돌리기 비싼 artifact다: 그것을 깨는 것은 조율된 multi-team migration의 비용이 든다. [[ch-05]]는 integration contract를 다룬다: 일련의 제약(constraint)으로서의 REST(그리고 대부분의 "RESTful" API가 사실은 Level-2 HTTP-RPC에 불과하다는 불편한 진실), trade-off로서의 API-style 선택, additive versioning(추가적 버전 관리)과 idempotency(멱등성), 그리고 이 챕터가 당신에게 소중히 여기라고 가르친 independent deployability를 *보존하는* 단 하나의 메커니즘인 consumer-driven contract(소비자 주도 계약). 그다음 [[ch-06]]은 당신이 방금 그린 boundary를 건너는 것이 실제로 무슨 비용이 드는지 가격 매긴다: lock과 shared transaction의 상실, saga, 그리고 outbox.
