<!-- chapter: ch-05
     track: contracts
     kind: content
     title: Integration Contracts: REST, API Styles, and Safe Evolution
     deps: [[ch-04]]
     sources: [[fielding-rest]], [[consumer-driven-contracts]], [[transactional-outbox]], [[newman-building-microservices]], [[richards-ford-hard-parts]]
-->

# 05장 — Integration Contracts: REST, API Styles, and Safe Evolution

> **핵심 통찰.** 일단 boundary(경계)를 그었다면(ch-02부터 ch-04까지), 그 boundary는 *곧* 그것의 published contract(공개된 계약)이다 — 그리고 contract는 한 service가 소유한 것 중 가장 되돌리기 비싼(most expensive-to-reverse) artifact(산출물)이다. service 내부의 나머지 모든 것은 private하고 마음대로 refactor할 수 있다; contract는 다른 팀들이 이미 coupling(결합)해 둔 단 하나의 것이기 때문에, 그것을 바꾸는 것은 여러 팀이 조율된(coordinated) multi-team migration을 치르는 비용이 든다. 따라서 이 챕터의 모든 분야는 "좋은 API를 어떻게 만드는가"가 아니라 "다른 사람들이 의존하게 된 후에도 boundary를 진화시키는(evolve) 비용을 어떻게 싸게 유지하는가"이다. REST, the Richardson Maturity Model, gRPC/GraphQL 선택, additive versioning, idempotency, 그리고 consumer-driven contracts는 모두 그 하나의 게임에서의 수(move)다: edge(가장자리)를 재협상하지 않고도 내부(inside)를 바꿀 수 있는 능력을 스스로에게 사주는 것.

> **가이드라인.** 실제로 필요한 속성에 따라 API style을 골라라([[fielding-rest]]) — cacheable하고, evolvable하며, loosely-coupled한 resource에는 REST/HTTP; low-latency 내부 service-to-service에는 gRPC; client-driven aggregation에는 GraphQL — 그런 다음 세 가지 분야로 contract의 evolvability(진화 가능성)를 방어하라: **additively**(추가적으로) 진화시키고(절대 field를 제거하거나 용도 변경하지 마라), at-least-once delivery 하에서 retry가 안전하도록 모든 write를 **idempotent**(멱등)하게 만들고, consumer가 자신의 실제 기대를 provider의 CI에서 돌아가는 **executable contract test**로 표현하게 하라([[consumer-driven-contracts]]). "breaking(깨뜨리는)"인 변경은 실제 consumer의 contract를 깨는 것들뿐이다 — 그리고 당신은 그것을 deploy *전에* 알 수 있어야 하지, pager(호출기)가 울린 후에 알아서는 안 된다.

---

## 1. The Contract Is the Expensive-to-Reverse Artifact

Ch-04는 [[richards-ford-hard-parts]]에서 나온 냉정한 진실로 끝맺었다: monolith 내부의 나쁜 boundary는 refactor지만, service들에 걸친 나쁜 boundary는 migration이다. 이 챕터는 다음 단계를 밟는다. cross-service boundary를 비싸게 만드는 것은 network hop(네트워크 도약)이 아니다 — 그것은 다른 팀들이 그것에 기대어 코드를 구축한 **published contract**다. 두 번째 팀이 당신의 `stage_name` field를 읽거나 당신의 `POST /opportunities`가 `201`을 반환하는 것에 의존하는 코드를 작성하는 그 순간, 그 field와 그 status code는 당신이 일방적으로(unilaterally) 바꿀 수 있는 것이 더 이상 아니게 된다.

> 💡 **쉬운 설명:** 여기서 핵심은 "한 번 공개해서 남이 쓰기 시작하면, 그 약속은 더 이상 나 혼자만의 것이 아니다"라는 점이다. 내가 만든 함수의 내부 구현은 언제든 바꿔도 아무도 모르지만, 내가 외부에 노출한 응답 형식(field 이름, status code)은 다른 팀의 코드가 그것에 맞춰 짜여 있으므로, 그걸 바꾸면 남의 코드가 깨진다. 그래서 "되돌리기 비싸다(expensive-to-reverse)"고 부른다.

[[consumer-driven-contracts]]는 그 design-altitude(설계 고도) 차원의 귀결을 직접적으로 진술한다:

> "The published contract is the **most expensive-to-reverse** artifact a service owns: once consumers depend on it, breaking it costs coordinated multi-team migration — exactly the cost microservices exist to avoid." — from [[consumer-driven-contracts]]

이것은 챕터 전체를 재구성한다(reframe). ch-04에서, [[richards-ford-hard-parts]]의 architecture quantum(아키텍처 양자)은 independent deployability(독립적 배포 가능성)가 보호할 가치가 있는 속성임을 가르쳐 주었다. contract는 바로 independent deployability가 살거나 죽는 지점이다: 한 service는 모든 consumer에게 먼저 재배포를 요청하지 않고도 자신의 내부를 바꿀 수 있어야만 독립적으로 배포할 수 있다. 따라서 contract는 boundary의 부수적인 artifact가 아니다; 그것은 boundary의 *load-bearing surface*(하중을 지탱하는 면)다.

> 💡 **쉬운 설명:** "architecture quantum"은 독립적으로 배포 가능한 가장 작은 단위를 뜻한다(ch-04 용어). 건물에 비유하면 contract는 벽에 걸린 장식이 아니라 건물 무게를 떠받치는 기둥(load-bearing)이다. 기둥을 함부로 빼면 건물 전체가 무너지듯, contract를 함부로 바꾸면 그것에 의존하는 모든 service가 무너진다.

### 1.1 Why this is a course-spine moment, not an API-design tutorial

이 코스의 조직 원리([[insights]])는 architecture란 되돌리기 비싼 결정들의 집합이라는 것이다. 네 개의 pillar(기둥)는 그 결정들을 reversal cost(되돌리는 비용) 순서로 배치한다: boundaries(가장 비쌈, ch-02–04로 미뤄짐), 그다음 그것들을 가로지르는 contracts(이 챕터), 그다음 그것들에 걸친 consistency(ch-06–07), 그다음 그 모든 것을 수정 가능하게(revisable) 유지하는 evolution discipline(ch-08–09).

Contracts는 정확히 지금 위치에 있는데, 그것들이 *한 service가 소유한 가장 되돌리기 비싼 artifact*이기 때문이다 — database schema(private하고, API 뒤에서 migratable함)보다 비싸고, framework choice(ch-03의 [[martin-clean-arch]]에서 나온 Dependency Rule에 따라 port 뒤에서 교체 가능함)보다 비싸다. contract보다 되돌리기 더 비싼 단 한 가지는 contract가 올라앉아 있는 boundary뿐이다. 그것이 이 챕터가 topology(위상) 이후, consistency 이전에 오는 이유다: crossing(경계 가로지르기)을 정의하는 artifact의 값을 매기기(price) 전까지는 boundary를 가로지르는 비용(ch-06)에 대해 추론할 수 없다.

> 💡 **쉬운 설명:** "price/pricing"은 이 코스 전체에서 반복되는 비유로, 어떤 결정의 트레이드오프(무엇이 싸지고 무엇이 비싸지는지)를 명시적으로 따져본다는 뜻이다. 여기서는 "boundary를 가로지르는 비용을 계산하려면, 먼저 그 가로지름을 규정하는 contract의 비용부터 알아야 한다"는 논리 순서를 말한다.

| Artifact | Owner | Reversal cost | Why |
|---|---|---|---|
| Internal class / module structure | the service | cheap (refactor) | 외부의 누구도 그것을 보지 못한다 |
| Database schema | the service | medium (migration, but private) | [[newman-building-microservices]]에 따라 API 뒤에 숨겨져 있다 |
| Framework / vendor SDK | the service | medium (swap behind a port) | ch-03의 Dependency Rule로 격리되어 있다 |
| **Published contract** | the service **+ every consumer** | **expensive (multi-team migration)** | consumer들이 이미 그것에 coupling되어 있다 |
| The boundary itself | the org | most expensive (re-architecture) | ch-04: cross-service = migration |

---

## 2. REST Is a Set of Constraints, Not a URL Convention

contract를 어떻게 진화시킬지 값을 매기기 전에, 당신이 어떤 종류의 contract를 출하하고 있는지에 대해 솔직해져야 한다. API 설계에서 가장 많이 쓰이면서 가장 많이 오용되는 단어가 "REST"이므로, primary source(1차 출처)와 함께 거기서 시작한다.

REST는 "JSON over HTTP with nice URLs(예쁜 URL을 가진 HTTP 상의 JSON)"가 아니다. 그것은 Roy Fielding이 정의한, 명명된 architectural style(아키텍처 스타일)이며, 특정한 web-scale 속성들을 사주기(buy) 위해 선택된 특정한 constraint(제약) 집합으로부터 조립된 것이다. [[COLLECTION-PLAN]]의 gap log(공백 기록)에 따르면, dissertation(학위논문)의 정전(canonical)인 `ics.uci.edu` 사본은 crawl 시점에 TLS chain이 깨져 있었다; 아래 인용문들은 Fielding 본인의 `roy.gbiv.com` mirror(권위 있음)에서 가져왔으며, [[fielding-rest]]에 그렇게 인용되어 있다:

> "REST is a hybrid style derived from several of the network-based architectural styles… combined with additional constraints that define a uniform connector interface." — Fielding (via the roy.gbiv.com mirror, per [[fielding-rest]])

> 💡 **쉬운 설명:** "primary source(1차 출처)"란 누군가의 요약이나 해설이 아니라 원저자가 직접 쓴 원문을 뜻한다. 여기서는 REST를 만든 Roy Fielding의 박사 학위논문이 그것이다. 원래 대학(ics.uci.edu) 사본의 HTTPS 인증서 체인이 깨져 있어서, Fielding이 직접 운영하는 mirror(똑같은 원문을 복사해 둔 다른 주소) 사본을 인용했다는 출처 투명성 설명이다.

### 2.1 The six constraints and what each one buys

Fielding의 style은 이 constraint들의 *합(sum)*이다. 각각은 그 자체로 값이 매겨진 bet(내기)이다 — 어떤 속성과 맞바꾸는 대가로 무언가를 금지한다.

| Constraint | Fielding's wording (via [[fielding-rest]]) | What it buys | What it costs |
|---|---|---|---|
| Client–Server | "separating the user interface concerns from the data storage concerns… improve[s] the portability… and… scalability" | UI와 storage의 독립적 진화 | 조율해야 할 두 당사자 |
| Stateless | "Each request… must contain all of the information necessary to understand the request, and cannot take advantage of any stored context on the server" | horizontal scaling, 어느 server든 어느 request에든 응답 가능 | 모든 request가 context를 재전송함; 값싼 session이 없음 |
| Cache | "data within a response… [is] implicitly or explicitly labeled as cacheable or non-cacheable" | intermediary(중개자)가 반복 요청을 처리할 수 있음; latency + load 감소 | staleness(낡음) 관리 |
| Uniform Interface | "the central feature that distinguishes the REST architectural style… is its emphasis on a uniform interface between components" | component들이 독립적으로 진화함; 범용 tooling이 작동함 | interface가 generic하므로, bespoke한 것보다 덜 효율적임 |
| Layered System | components "cannot 'see' beyond the immediate layer" | proxy, gateway, cache를 투명하게 삽입 | layer마다 추가 latency |
| Code-on-Demand | "only an optional constraint" | runtime에 client를 확장 | 유일한 optional; 거의 쓰이지 않음 |

> 💡 **쉬운 설명:** 각 constraint를 "내기"로 보라는 것이 핵심이다. 예컨대 Stateless는 server가 session 상태를 들고 있지 않게 강제해서(금지) 어느 server든 아무 요청이나 처리할 수 있게(scaling 이득) 해준다. 대신 매 요청마다 필요한 정보를 다 실어 보내야 하는 비용을 치른다. 공짜 이득이 아니라 "무엇을 포기하고 무엇을 얻는다"는 거래임을 강조한다.

Uniform Interface constraint가 무거운 일을 하는 것이며, Fielding은 그것을 네 개의 sub-constraint로 분해한다([[fielding-rest]]를 통한 그대로의 목록): "identification of resources; manipulation of resources through representations; self-descriptive messages; and, hypermedia as the engine of application state." 그 네 번째 것 — **hypermedia as the engine of application state (HATEOAS)** — 을 붙들고 있어라. 거의 모든 실제 "REST API"가 떨어뜨리는(drop) 바로 그 constraint이며, Fielding이 협상 불가(non-negotiable)라고 주장하는 바로 그것이기 때문이다.

> 💡 **쉬운 설명:** HATEOAS는 "응답 안에 다음에 무엇을 할 수 있는지를 알려주는 링크(hypermedia control)를 server가 같이 담아 보내고, client는 그 링크를 따라가기만 한다"는 원칙이다. 비유하자면, 웹사이트를 쓸 때 우리가 URL을 외우지 않고 화면에 뜬 버튼/링크를 클릭해 이동하는 것과 같다. Fielding은 이게 빠지면 진짜 REST가 아니라고 못 박았는데, 현실의 거의 모든 "REST API"는 이걸 생략한다.

어휘를 정확하게 유지하기 위한 두 가지 정의, 둘 다 [[fielding-rest]]에서: *resource*는 "any information that can be named(이름 붙일 수 있는 모든 정보)"이다; *representation*은 "the current or intended state of that resource(그 resource의 현재 또는 의도된 state)"를 포착한다. 당신은 resource를 직접 조작하지 않는다 — 당신은 그것의 representation을 주고받는다.

> 💡 **쉬운 설명:** resource와 representation의 구분이 헷갈릴 수 있다. resource는 "고객 #501"이라는 추상적 개념이고, representation은 그 고객을 표현한 구체적 형태(예: 그 순간의 JSON 데이터)다. 우리는 추상적 고객 자체를 만질 수 없고, 그 고객의 JSON 사본을 주고받으며 다룬다. 즉 항상 사본(representation)을 교환한다.

---

## 3. The Richardson Maturity Model and the "Our API Is RESTful" Myth

이것은 이 챕터의 reconciliation myth(통념 바로잡기)로, [[COLLECTION-PLAN]]의 표와 outline의 authoring notes에 있는 per-chapter myth 목록에서 가져왔다:

> **Popular narrative:** "Our API is RESTful."
> **What the primary source says:** Almost all are **Level 2** (HTTP-RPC) on Richardson's scale. Fowler: "Roy Fielding has made it clear that level 3 [hypermedia] is a pre-condition of REST." Most "REST" isn't, by Fielding's definition. — reconciliation table, [[COLLECTION-PLAN]]

> 💡 **쉬운 설명:** "reconciliation myth"는 이 코스의 한 장치로, 업계에서 흔히 믿는 통념(popular narrative)과 1차 출처가 실제로 말하는 바를 나란히 놓고 격차를 드러내는 것이다. 여기서 통념은 "우리 API는 RESTful해"이고, 실상은 "거의 다 Level 2일 뿐, Fielding의 정의로는 REST가 아니다"라는 것이다.

Richardson Maturity Model (RMM)은 Leonard Richardson의 것을 따라 Martin Fowler가 기술하고 [[fielding-rest]]에 인용된 것으로, Fielding의 REST를 향한 네 단계의 사다리(ladder)다. 당신이 실제로 무엇을 만들었는지에 대해 솔직해지기 위한 가장 유용한 단일 도구다.

| Level | Name | What it adds (Fowler, via [[fielding-rest]]) |
|---|---|---|
| 0 | "Swamp of POX" | HTTP "as a tunneling mechanism… based on Remote Procedure Invocation" — 하나의 URI, 하나의 verb |
| 1 | Resources | "rather than making all our requests to a singular service endpoint, we… talk to individual resources" |
| 2 | HTTP Verbs | "using the HTTP verbs as closely as possible to how they are used in HTTP itself" (GET safe, status codes) |
| 3 | Hypermedia (HATEOAS) | "tell us what we can do next, and the URI… to do it"라고 알려주는 control들 |

> **사다리를 인터랙티브하게 탐색하기:** [`figures/richardson-maturity-model.html`](figures/richardson-maturity-model.html)을 열고 각 단(rung)을 클릭하여 구체적인 request/response 쌍과 그 level이 무엇을 사주는지 대(對) 무엇을 비용으로 치르는지를 보라. 그림은 Level 2를 기본값으로 하며 두 가지 사실을 눈에 띄게 표시한다: 대부분의 실제 "REST" API가 거기서 멈춘다는 것, 그리고 Fielding이 그것을 REST라고 부르기 전에 Level 3 hypermedia를 요구한다는 것. "mark reality" 상자를 토글하여 Fielding이 긋는 선을 보라.

### 3.1 The caveat that kills the myth

하중을 지탱하는 인용문(load-bearing quote), [[fielding-rest]]를 통한 그대로:

> "Roy Fielding has made it clear that level 3 RMM is a pre-condition of REST." — Fowler

그 귀결, 역시 [[fielding-rest]]에서: 업계가 "REST"라고 부르는 거의 모든 것은 Level 2 — 잘 수행된 HTTP-RPC다. resource가 주소 지정되고, verb가 올바르게 사용되고, status code가 의미를 가지며, GET이 safe하고 cacheable하다. 그것은 진정으로 좋은 API다. 다만 그것은 HATEOAS가 없기 때문에 *Fielding의 정의로는 REST가 아닐* 뿐이다: client가 여전히 URI template과 application의 state machine을 hard-code하며, server가 각 response에 실어 보내는 hypermedia control을 따르지 않는다.

> 💡 **쉬운 설명:** "Level 2 = 잘 만든 HTTP-RPC"라는 말이 핵심이다. 즉 흔히 보는 좋은 API들은 사실 RPC(원격 프로시저 호출)를 HTTP 위에 깔끔하게 얹은 것이고, 그것 자체로 훌륭하다. 단지 client가 URL 패턴을 코드에 박아 넣고(hard-code) 다음 동작을 스스로 계획한다는 점에서, server가 링크로 길을 안내하는 진짜 REST(Level 3)와는 다르다는 것이다.

### 3.2 Why the honesty matters (and why Level 2 is often the right bet)

이것은 현학(pedantry)을 위한 현학이 아니다. RMM이 design altitude에서 중요한 이유는 **Level 3가 값이 매겨진 bet이고, 대부분의 팀은 옳게도 그 값을 치르기를 거부한다**는 것이다:

- **Level 3가 싸게 유지하는 것:** server가 URI 구조와 workflow를 소유하므로, 단순히 링크를 따라가는 client를 깨뜨리지 않고 endpoint를 옮기고 state machine을 바꿀 수 있다. URI evolution이 공짜가 된다.
- **Level 3가 비싸게 만드는 것:** client가 runtime에 hypermedia를 navigate할 만큼 충분히 정교해야 하는데, 거의 어떤 SDK/codegen 생태계도 이를 하지 못한다 — 그들은 고정된 URI를 가정한다. 대부분의 팀이 실제로는 결코 거두지 않는 payoff(독립적 URI evolution)를 위해 큰 선행(upfront) 설계 비용 + client 복잡성 비용을 치른다.

따라서 이 통념의 해소는 "Level 3로 가라"가 아니다. 그것은: **당신이 어느 level을 출하했는지 알고, 그것을 솔직하게 이름 붙이고, 필요한 속성에 따라 level을 선택하라**이다. Level-2 API를 "RESTful"이라고 부르는 것은, 누군가가 client가 링크를 따른다는 가정 위에서 설계하기 전까지는 무해한 약칭(shorthand)이다 — 그 시점에 그 격차는 실제 bug가 된다. ch-01의 First Law([[richards-ford-fundamentals]], [[richards-ford-hard-parts]]를 통해)가 적용된다: Level 3가 당신에게 무엇을 비용으로 치르게 하는지 이름 붙일 수 없다면, 당신은 왜 (올바르게) Level 2에 있는지 이해하지 못한 것이다.

> 💡 **쉬운 설명:** First Law(소프트웨어 아키텍처의 제1법칙)는 "모든 것은 트레이드오프다"이다. 여기서의 적용은 이렇다: 당신이 Level 3를 안 쓰기로 했다면, 그게 무엇을 포기하는 선택인지(URI를 마음대로 못 옮기는 비용 등)를 말할 수 있어야 진짜로 이해한 것이다. 트레이드오프를 설명 못 하면, 그냥 관성으로 Level 2에 머문 것일 뿐 의식적 선택이 아니다.

---

## 4. API-Style Selection as a Trade-off

REST/HTTP는 여러 style 중 하나이며, "어느 style인가"는 default가 아니라 trade-off 결정이다. [[fielding-rest]]는 그 선택을 이렇게 구성한다: 필요한 속성에 따라 style을 골라라. [[COLLECTION-PLAN]]의 gap log에서 나온 솔직함의 단서(hedge)에 유의하라 — [[fielding-rest]]의 gRPC/GraphQL trade-off는 canonical-author quote가 아니라 **synthesis**(종합)다; 그것들은 독립적으로 출처가 있는 excerpt가 아니라 Fielding excerpt 안의 trade-off context로 존재하며, 여기서도 그렇게 제시된다.

> 💡 **쉬운 설명:** "synthesis(종합)"는 원저자가 직접 한 말이 아니라, 코스 저자가 여러 출처를 종합해 정리한 해설이라는 뜻이다. 따라서 아래 표의 gRPC/GraphQL 평가는 "Fielding이 한 말"이 아니라 "저자가 정리한 통설"로 받아들이라는 출처 정직성 표시다.

| Style | Strengths (synthesis, via [[fielding-rest]]) | Weaknesses | Pick it when |
|---|---|---|---|
| REST / HTTP | cacheable, evolvable, 어디에나 있음(ubiquitous); 어떤 proxy/CDN을 통해서도 작동 | 약한 typing, over/under-fetching, object graph에 대해 chatty(말이 많음) | cacheable하고, loosely-coupled하며, evolvable한 **resource**를 널리 노출하고 싶을 때 |
| gRPC | binary, contract-first (protobuf), streaming, low latency | 빈약한 browser/edge ergonomics; 양쪽 끝에 codegen 필요 | latency와 strict schema가 중요한 **internal** service-to-service |
| GraphQL | client가 정확히 원하는 field만 고름; graph에 대해 one round-trip | 복잡성이 server로 이동(N+1 queries, per-field caching, per-field auth) | **client**가 여러 resource에 걸쳐 aggregate해야 하고 당신이 schema를 통제할 때 |

> 💡 **쉬운 설명:** 표의 용어 몇 가지를 풀어 보면 — over/under-fetching은 필요한 것보다 너무 많거나 적게 받아오는 문제, chatty는 객체 그래프를 다 가져오려고 호출을 여러 번 해야 하는 수다스러움, N+1 queries는 목록 하나 가져오고 각 항목마다 추가 쿼리를 또 날려 결국 쿼리가 폭증하는 전형적 성능 함정이다. GraphQL은 client를 편하게 해주는 대신 이런 복잡성을 server가 떠안는다.

### 4.1 The pattern is priced as a bet, every time

그 형태에 주목하라: 각 style은 무언가를 싸게 유지하고 다른 무언가를 비싸게 만든다. REST는 *strict typing과 graph-fetching*을 비싸게 만듦으로써 *broad reach와 caching*을 싸게 유지한다. gRPC는 *edge/browser reach*를 비싸게 만듦으로써 *latency와 schema rigor*를 싸게 유지한다. GraphQL은 *운영 복잡성*을 server로 밀어냄으로써 *client 유연성*을 싸게 유지한다. "최고의" API style은 없다 — 그 싸게-유지하는-axis(축)가 당신이 가장 자주 할 것으로 예상하는 변경과 맞아떨어지는 style만 있을 뿐이다. 이것은 integration에 적용된 First Law다: "Everything in software architecture is a trade-off… if you think you've found something that isn't, you likely just haven't found the trade-off yet" ([[richards-ford-fundamentals]], [[richards-ford-hard-parts]]를 통해).

---

## 5. Safe Evolution: Additive Change, Tolerant Readers, and Idempotency

style을 고르는 것은 쉬운 절반이다. 비싼 절반은 *consumer가 의존하게 된 후에* 조율된 migration을 강제하지 않고 contract를 진화시키는 것이다. 세 가지 분야가 이 일을 한다.

### 5.1 Evolve additively — never remove or repurpose a field

[[fielding-rest]]는 규칙을 단순하게 진술한다: contract는 되돌리기 비싼 artifact이므로, versioned breakage(버전을 매긴 깨짐)보다 **backward/forward-compatible additive change**(하위/상위 호환되는 추가적 변경)를 선호하라. 구체적으로: 새 field를 추가하되, 기존 것을 절대 제거하거나 용도 변경하지 마라; 새 optional parameter를 추가하되, optional인 것을 required로 만들지 마라; 새 resource를 추가하되, 기존 것의 의미를 바꾸지 마라. additive change는 forward-compatible하다 — 오래된 client는 새 field를 무시하고 계속 작동한다.

> 💡 **쉬운 설명:** "additive(추가적)"의 핵심은 "더하기만 하고, 빼거나 바꾸지 않는다"이다. backward-compatible은 새 server가 옛 client를 깨지 않는 것, forward-compatible은 옛 client가 미래의(새로운 field가 추가된) 응답을 받아도 깨지지 않는 것을 뜻한다. field를 추가만 하면 옛 client는 모르는 field를 그냥 무시하므로 아무 일도 안 일어난다 — 그래서 일방적으로 배포해도 안전하다.

additive evolution이 애초에 가능한 이유는 **tolerant reader**(관용적 수신자) 분야이며, consumer가 자기 몫을 떠받쳐야 한다. [[consumer-driven-contracts]]에서, Robinson을 그대로 인용하면:

> "An implementation must be conservative in its sending behaviour and liberal in its receiving behaviour… message receivers should implement 'just enough' validation: that is, they should only process data that contributes to the business functions they implement." — Robinson, via [[consumer-driven-contracts]]

이것이 Postel's Law(포스텔의 법칙)이다: "conservative in what you send, liberal in what you accept(보낼 때는 보수적으로, 받을 때는 관대하게)." 그 귀결은, [[consumer-driven-contracts]]에 진술된 대로: 알 수 없는 field를 무시하는 consumer는 provider가 field를 자유롭게 *추가*하게 해준다(forward compatibility). 그 역(inverse)이 함정이다 — **consumer 측의 strict schema validation이 바로 무해한 additive change를 breaking change로 바꾸는 것이다.** 당신의 consumer가 예상치 못한 field를 가진 payload를 거부한다면, 당신은 provider의 동의 없이 일방적으로 provider의 contract를 깨지기 쉽게(brittle) 만든 것이다.

> 💡 **쉬운 설명:** tolerant reader는 "내가 쓰는 field만 처리하고, 모르는 field는 조용히 무시하는 수신자"다. 반대로 "응답에 모르는 field가 하나라도 있으면 에러 내고 거부하는" strict validation을 하면, provider가 단지 field 하나 추가한 것만으로도 내 쪽이 터진다. 즉 깨짐의 원인이 provider가 아니라 까다롭게 검증한 내(consumer)게 있다는 것이 반전 포인트다.

### 5.2 Make writes idempotent — retries must be safe

두 번째 분야는 data 측에서 오는데, 거기서는 contract 관심사와 delivery 관심사가 같은 대화다. [[transactional-outbox]](Richardson)는 비동기적이고 reliable한 messaging이 **at-least-once** delivery(최소 한 번 전달)임을 확립한다: relay(중계기)가 같은 event를 두 번 이상 publish할 수 있으므로, consumer는 dedupe(중복 제거)해야 한다. [[fielding-rest]]에서 나온 그 API-contract 짝은, idempotency key를 사용하여 **write를 idempotent**하게 만드는 것이다. 그래서 (write가 실패해서가 아니라 timeout이 나서) retry하는 client가 효과를 두 번 적용하지 않게 한다.

> 💡 **쉬운 설명:** at-least-once delivery란 "메시지가 최소 한 번은 전달되지만, 때로는 두 번 이상 전달될 수도 있다"는 보장이다. 왜 두 번이 가능한가? 예컨대 server가 처리를 끝냈는데 응답이 오기 전에 client가 timeout으로 판단하고 다시 보내면, 같은 요청이 두 번 처리될 수 있다. idempotent(멱등)는 "같은 요청을 몇 번을 보내도 결과가 한 번 보낸 것과 똑같다"는 성질이며, 이 중복 전달 문제를 안전하게 만드는 해법이다.

[[transactional-outbox]]는 그 연결을 명시적으로 만든다 — idempotency는 API 측과 data 측의 두 개의 별개 문제가 아니라, 하나다:

> "This is why idempotency, an API-contract concern… and the outbox, a data concern, are the same conversation." — from [[transactional-outbox]]

그 메커니즘: client는 각 write에 고유한 `Idempotency-Key`를 붙인다; server는 첫 실행의 결과와 함께 그 key를 기록하고, 같은 key를 운반하는 어떤 retry에서도 재실행 대신 기록된 결과를 반환한다. at-least-once delivery 하에서(ch-06의 예고에 따르면 어떤 분산 시스템에서도 이를 피할 수 없다), idempotency는 "불확실하면 retry한다"를 corruption(손상) 위험이 아니라 안전한 default로 만드는 것이다. 이것은 ch-06이 [[transactional-outbox]]와 [[richardson-saga]] 위에 구축하는 outbox + saga 메커니즘의 in-contract seed(계약 내부의 씨앗)다.

at-least-once가 불가피한 이유 — 그리고 따라서 idempotency가 non-optional인 이유 — 는 [[transactional-outbox]]가 명명하는 dual-write problem(이중 쓰기 문제)에서 온다. Richardson은 그 제약을 그대로 진술한다:

> "It is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker." — Richardson, via [[transactional-outbox]]

database write와 그 event publish를 atomically(원자적으로) commit할 수 없기 때문에, 유일하게 reliable한 설계는 같은 local transaction 안에 event를 persist하고 나중에 relay하는 것이다 — 그리고 delivery를 보장하는 어떤 relay든, crash-and-retry(충돌 후 재시도) 시 때때로 두 번 deliver할 것이다. Richardson이 제공하는 보장은 "messages are guaranteed to be sent if and only if the database transaction commits"(그대로, [[transactional-outbox]])이다 — *sent*이지, *sent exactly once*가 아니다. Exactly-once는 transport layer에서는 허구(fiction)다; idempotent consumer는 당신이 at-least-once *delivery* 위에 exactly-once *effect*를 제조하는(manufacture) 방법이다. 따라서 contract 분야(idempotency key)와 data 분야(outbox)는 하나의 전선(wire)의 양 끝이며, 그것이 이 챕터가 둘 다 심고(seed) ch-06이 그것들을 완성하는 이유다.

> 💡 **쉬운 설명:** dual-write problem이란 "DB에 쓰기 + 메시지 브로커에 이벤트 발행"이라는 두 개의 별개 시스템에 대한 쓰기를 한 번에 원자적으로 묶을 수 없다는 문제다. 2PC(two-phase commit, 2단계 커밋)라는 전통적 분산 트랜잭션 기법이 있지만 DB와 메시지 브로커에 걸쳐서는 현실적이지 않다. 그래서 해법은 "이벤트를 DB의 같은 트랜잭션 안에 같이 저장(outbox 테이블)하고, 그 후에 따로 발행"하는 것이다. 단, 발행 과정에서 충돌이 나면 같은 이벤트를 두 번 보낼 수 있으므로, 받는 쪽이 idempotent해야 "정확히 한 번 일어난 것 같은 효과"를 만들어낼 수 있다.

### 5.3 A taxonomy of changes, by reversal cost

additive/tolerant-reader 분야들은 가능한 모든 contract change를 세 개의 bucket(통)으로 분할하며, 그 bucket이 비용을 결정한다. 이것은 §1의 "contract는 되돌리기 비싼 artifact"의 운영적(operational) 형태다:

| Change | Compatible? | Cost | Examples |
|---|---|---|---|
| **Additive** (new optional field, new resource, new optional param) | tolerant reader에게 forward-compatible | cheap — 일방적으로 deploy | opportunity payload에 `parent_account_id` 추가 |
| **Tightening** (new required field, narrowed enum, removed field) | 옛 shape에 의존하는 consumer를 깨뜨림 | expensive — migration 필요 | `region`을 required로 만들기; `legacy_stage` 제거 |
| **Semantic** (same shape, changed meaning) | 최악의 종류 — compile은 되지만 조용히 깨짐 | most expensive — schema check로 감지 불가 | `amount`가 cents에서 dollars로 전환 |

semantic-change 행이 위험한 것이며 CDC(§6)가 존재하는 이유다: tolerant reader도 schema check도 semantic change를 둘 다 *통과*시키는데, shape가 변하지 않았기 때문이다. *값의 의미(value's meaning)*에 대해 assert하는 consumer-authored example만이 deploy 전에 그것을 잡아낸다. field의 의미를 그 이름만큼 immutable하게 유지하라.

> 💡 **쉬운 설명:** semantic change가 가장 무서운 이유를 예로 들면 — `amount` field의 형태(숫자)는 그대로지만 단위가 "센트"에서 "달러"로 바뀌면, 모든 검사 도구는 "숫자 맞네" 하고 통과시킨다. 그런데 옛 client는 100을 "1달러(=100센트)"로 읽던 걸 이제 "100달러"로 읽게 되어 100배 오류가 조용히 발생한다. 형태 검사로는 절대 못 잡고, 오직 "이 값은 센트여야 한다"는 의미를 검증하는 테스트만 잡아낼 수 있다.

### 5.4 Versioning is the fallback, not the plan

Versioning(`/v2/`, `Accept: application/vnd.app.v2+json` 등)은 어떤 변경이 진정으로 tightening 또는 semantic bucket에 떨어지고 additively하게 만들 수 없을 때 당신이 후퇴하는(fall back) 수단이다. 그것은 공짜가 아니다: 새 version은 두 개의 contract를 병렬로 운영하고, consumer를 migrate하고, 결국 옛것을 deprecate하는 것을 의미한다 — 조율된 multi-team 작업, 바로 additive 분야가 피하려고 존재하는 그 비용이다. 강제된 version bump를 boundary나 contract를 잘못 잡았다는 신호로 취급하고, 다른 어떤 expensive-to-reverse reversal처럼 migration을 위한 예산을 잡아라. 이 분야의 전체 목표는 additive bucket을 가능한 한 크게 유지하여 version-bump bucket을 비워 두는 것이다.

> 💡 **쉬운 설명:** versioning은 "최후의 수단이지 평소의 계획이 아니다"라는 게 요지다. v1, v2를 동시에 굴리는 순간 두 contract를 다 유지보수해야 하고, 모든 consumer를 새 버전으로 이주시켜야 하므로 비싸다. 그래서 가능하면 additive 변경으로 처리하고, 버전을 올려야만 하는 상황을 "설계를 잘못했다는 경고등"으로 받아들이라는 것이다.

---

## 6. Consumer-Driven Contracts: Knowing What "Breaking" Means Before You Deploy

Additive 분야는 안전하게 변경하는 *방법(how)*을 알려준다. Consumer-driven contracts (CDC)는 주어진 변경이 실제로 안전한지 *여부(whether)*를 — 자동으로, CI에서, 출하 전에 — 알려준다. 이것이 위의 모든 것을 운영화하는(operationalize) 메커니즘이다.

### 6.1 The inversion

핵심 수(move)는, [[consumer-driven-contracts]]에서, 의무의 통상적 방향을 뒤집는다(invert). consumer가 provider가 출하하는 무엇에든 적응하는 대신, provider가 consumer들이 표현하는 reasonable expectation(합리적 기대)을 채택한다:

> "When a provider accepts and adopts the reasonable expectations expressed by a consumer, it enters into a consumer contract." — Robinson, via [[consumer-driven-contracts]]

> 💡 **쉬운 설명:** "inversion(역전)"이 핵심 단어다. 보통은 "provider가 마음대로 만들고 consumer가 거기 맞춰라"인데, CDC는 이걸 뒤집어 "consumer가 '나는 이 field들이 이렇게 와야 한다'고 기대를 명시하면, provider가 그 기대를 자기 테스트로 받아들인다"로 바꾼다. 책임의 방향을 거꾸로 돌린 것이다.

이것이 강력한 이유, [[consumer-driven-contracts]]에서 그대로:

> "Consumer contracts allow us to reflect on the business value being exploited at any point in a provider's lifetime… [they] define which parts of that provider contract currently support the business value realized by the system." — Robinson

CDC를 이 챕터의 keystone(요석)으로 만드는 payoff, [[consumer-driven-contracts]]에서:

> "Consumer-driven provider contracts give us the fine-grained insight and rapid feedback we require to plan changes and assess their impact on applications currently in production." — Robinson

### 6.2 How it works in practice (synthesis, marked as such)

tooling 세부사항(Pact, Spring Cloud Contract)은 [[consumer-driven-contracts]]에 따라 **synthesis**이지, Robinson quote가 아니다:

1. 각 consumer는 **pact**(협약)를 publish한다 — 그것이 실제로 의존하는 구체적인 example request/response shape의 집합.
2. provider는 모든 consumer의 pact를 **자신의 CI pipeline에서** test로 돌리고, 어떤 변경이 실제 consumer를 깨뜨릴 것이라면 build를 실패시킨다.
3. 이것은 깨지기 쉽고 느린 end-to-end integration test를 빠르고 격리된 per-pair check로 대체한다.

이로부터 떨어져 나오는 "breaking change"의 정의가 바로 핵심이다, [[consumer-driven-contracts]]에서: **실제 consumer contract를 깨뜨리는 변경만이 breaking change다.** 아무도 읽지 않는 field를 제거하는 것은 breaking change가 *아니다*; field를 추가하는 것은 tolerant reader에게 *결코* breaking change가 아니다. CDC는 "이게 안전한가?"를 판단(judgment call)에서 자동화되고 증거에 기반한 답으로 바꾼다.

> 💡 **쉬운 설명:** pact는 "consumer가 적어 내는 기대 명세서"라고 보면 된다. 예컨대 "나는 GET /opportunities/501에 stage_name과 amount field가 이런 형태로 오기를 기대한다"를 구체적 예시로 적는다. provider는 모든 consumer의 이 명세서를 자기 CI에서 테스트로 돌리므로, 코드를 바꿨을 때 어떤 consumer가 깨지는지 배포 전에 빨간불로 알 수 있다. 느리고 깨지기 쉬운 통합 테스트(end-to-end) 대신, 짝(provider-consumer)마다 빠르게 검증한다.

### 6.3 Why CDC is the only mechanism that preserves independent deployability

이것은 곧장 ch-04로 다시 연결된다. [[newman-building-microservices]](Newman, 책 thesis — [[COLLECTION-PLAN]]에 따라 O'Reilly excerpts/talks로 추출되고 corroborate되었으므로, verbatim quote가 아니라 attributed paraphrase(귀속된 의역)로 제시됨)는 independent deployability를 microservice의 단 하나의 정의적 속성으로 만든다 — 다른 어떤 service의 release도 조율하지 않고 한 service를 출하하는 능력. [[consumer-driven-contracts]]는 그 속성이 현실과의 접촉에서 살아남게 하는 메커니즘으로 CDC를 명명한다:

> "It's the only mechanism that lets services keep their prized **independent deployability** (a provider knows, before deploy, whether it breaks anyone)." — from [[consumer-driven-contracts]]

CDC 없이는, "independent deployability"는 희망(hope)이다: 당신은 deploy하고 누군가를 깨뜨렸는지 pager로부터 알게 된다. CDC와 함께라면, 그것은 검증된 속성이다: 나쁜 변경이 출하되기 *전에* provider의 build가 빨간색이 된다. 이것은 ch-09의 fitness function의 contract-layer 유사물이다 — 보호된 속성(여기서는, "나는 어떤 consumer도 깨뜨리지 않는다")이 침식될 때 build를 실패시키는 자동화된 check.

> 💡 **쉬운 설명:** fitness function(ch-09 용어)은 "아키텍처가 지켜야 할 속성을 자동으로 검사해서, 어기면 빌드를 실패시키는 테스트"다. CDC는 바로 그 contract 층 버전이다. 즉 "나는 누구도 깨뜨리지 않는다"는 속성을 자동 검사로 못 박아 두고, 위반하면 배포 전에 빌드를 빨갛게 만든다. 그래야 independent deployability가 "그러길 바란다"가 아니라 "검증된 사실"이 된다.

### 6.4 Pricing the CDC bet

CDC는 공짜가 아니며, 챕터의 척추(spine)와 일관되게 우리는 그것에 값을 매긴다:

- **그것이 싸게 유지하는 것:** provider의 internals와 contract를 *additively*하게 자신감을 가지고 바꾸는 것; flaky한 end-to-end suite 대신 빠르고, local하며, deterministic한 feedback을 얻는 것; independent deployability가 실재로 유지되는 것.
- **그것이 비싸게 만드는 것:** 모든 consumer가 pact를 author하고 maintain해야 하며, provider의 CI가 그것들을 모두 돌려야 한다; 당신은 per-pair 유지보수 부담과 pact를 publish하는 조율 의례(coordination ritual)를 떠안는다. 같은 팀에 consumer 하나와 provider 하나가 있는 시스템이라면, 그 격식(ceremony)이 이익을 능가할 수 있다 — 이 bet는 독립적으로 소유된 consumer의 수가 늘어남에 따라 값을 한다.

> 💡 **쉬운 설명:** CDC도 트레이드오프라는 점을 잊지 말라는 것이다. consumer가 많고 서로 다른 팀이 소유할수록 CDC의 가치가 커진다. 반대로 provider와 consumer가 한 팀 안의 단 한 쌍뿐이라면, pact를 작성/유지하는 의례적 비용이 이득보다 클 수 있다. 즉 "consumer 수가 많을 때 본전을 뽑는 내기"다.

---

## 7. Applied to the Sales Agent: The Tool/Integration Layer Is a Contract Surface

학습자의 프로덕션 sales agent(Lina TMR)는 많은 외부 SaaS tool API — Salesforce, Gmail, calendar, spreadsheet, ticketing, CRM-sync — 에 걸쳐 작동하는 LLM agent다. 이 챕터의 분야는 두 전선(front)에서 그것에 mapping되며, 그 둘은 반대 방향으로 당기는데, 그것이 흥미로운 부분이다.

### 7.1 The agent as a *consumer* of contracts it does not own

agent의 integration surface 대부분은 다른 사람들의 contract에 대한 *inbound*(인바운드) 의존이다. agent는 Salesforce나 Gmail API를 소유하지 않는다; 그것은 그것들을 consume한다. 분야들은 그에 따라 뒤집힌다(invert):

- **공격적으로, tolerant reader가 되어라.** [[consumer-driven-contracts]]에 따라, agent는 각 SaaS response를 "just enough(딱 충분한 만큼)"만 처리해야 한다 — 비즈니스 function을 구동하는 field만, 그리고 나머지 모든 것은 무시한다. 자신의 response에 field를 추가하는 vendor가 agent를 깨뜨려서는 결코 안 된다. frozen vendor schema에 대한 strict validation은 정확히 §5.1의 함정이며, 경고 없이 변하는 수십 개의 third-party API에 걸쳐 작동하는 agent에게 그것은 보장된 outage(장애) 생성기다.
- **vendor가 허용하는 곳에서는 모든 tool call을 idempotent하게 만들어라.** [[transactional-outbox]]와 [[fielding-rest]]에 따라, agent는 자기 자신의 retry의 at-least-once 실행을 가정해야 한다. "send the win-notice email"이나 "mark opportunity Closed Won"이 두 번 발화할 수 있다면(timeout 후 retry), agent는 idempotency key나 dedupe guard가 필요하다. 그래야 flaky한 network가 고객에게 이메일을 두 번 보내거나 회의를 두 번 예약하지 않는다. 이것은 ch-06([[helland-data-outside-inside]])에서 모든 외부 response를 **outside data**(외부 데이터)로 다루는 것의 contract-layer seed다: agent의 retry는 그것이 통제하지 않는 boundary를 가로지르므로, 그 가로지름에 값을 매겨야 한다.

> 💡 **쉬운 설명:** 여기서 agent는 남이 만든 API를 "쓰는 쪽(consumer)"이라는 게 핵심이다. 그러니 분야가 뒤집힌다: 내가 통제할 수 없는 수십 개 vendor API가 언제든 응답에 field를 추가할 수 있으므로, 나는 모르는 field를 무시하는 tolerant reader가 되어야(안 그러면 vendor의 사소한 변경에 내 agent가 줄줄이 터진다) 하고, 또 내 retry가 두 번 실행돼도 고객에게 이메일이 두 번 안 가도록 idempotent 보호장치를 둬야 한다.

### 7.2 The agent as a *provider* of its own internal tool contract

다른 전선은 agent가 완전히 소유하는 것이다: agent의 reasoning core(ch-03의 [[martin-clean-arch]]를 통한 technology-free core)와 그것의 tool/integration adapter 사이의 boundary. 그 internal contract — "tool invocation과 result가 어떤 shape를 취하는가" — 는 정확히 §1에 따라, agent 자신의 가장 되돌리기 비싼 artifact다. 모든 prompt template, 모든 planner, 모든 adapter가 tool-result schema를 hard-code한다면, tool이 데이터를 반환하는 방식을 바꾸는 것은 agent 전체에 걸친 조율된 migration이 된다.

따라서 agent는 자신의 tool-call interface를 published contract로 다루고 내부적으로 CDC-style 분야를 적용해야 한다: reasoning core는 그것이 의존하는 result shape를 publish하고, adapter layer는 그것들을 만족시키며, boundary는 contract-test되어 adapter 변경이 planning을 조용히 깨뜨릴 수 없게 한다. 이것은 또한 API-style selection(§4)이 구체적으로 안착하는 곳이다 — agent의 *internal* core-to-adapter call은 gRPC-shaped 케이스(low-latency, in-process 또는 service-to-service, strict schema)인 반면, *external* SaaS call은 각 vendor가 출하하는 무엇이든의 style(대부분 Level-2 HTTP)이다. 아키텍트의 일은 vendor의 style이 adapter를 지나 core로 새어 들어가지 않게 유지하는 것이다 — ch-03의 Dependency Rule을 contract layer에서 강제하는 것.

> 💡 **쉬운 설명:** 같은 agent라도, 외부 API에 대해서는 "쓰는 쪽(consumer)"이지만 자기 내부(추론 코어 ↔ tool 어댑터) 경계에 대해서는 "제공하는 쪽(provider)"이다. 이 내부 경계도 contract이고, 그게 곳곳에 박혀 있으면(hard-code) tool 응답 형식 하나 바꾸는 데 agent 전체를 고쳐야 한다. 그래서 내부에도 CDC식 규율을 적용한다. 또 핵심은 "vendor의 너저분한 외부 style이 adapter에서 걸러져 core까지 침투하지 못하게" 막는 것 — 이게 ch-03의 Dependency Rule을 contract 층에서 적용한 모습이다.

### 7.3 The honest label, applied

팀이 "우리는 agent의 tool을 위한 REST API를 노출한다"고 말할 때, §3은 솔직한 정정을 요구한다: 그것은 거의 틀림없이 Level-2 HTTP-RPC이며, 그건 괜찮다. 명시적으로 이름 붙일 bet는 agent 설계의 어떤 부분이 hypermedia-style discoverability를 가정하는지 여부다(누군가가 의도적으로 Level 3에 값을 치르지 않았다면, 그래서는 안 된다). 학습자의 이전 코스에서 나온 AutomationBench 경험이 여기서 유용한 거울이다: 그 benchmark에서 agent는 ~400-tool catalog에 대한 `search_tools`/`execute_tool` indirection을 통해 tool을 discover한다 — 기저의 SaaS API들이 HATEOAS control을 출하하지 않기 때문에 바로 그 때문에 볼트로 덧붙여진(bolted-on) 의도적으로 *비*-hypermedia discovery 메커니즘이다. 그 덧붙임(bolt-on)이 Level 2에서 사는 비용이다; 그것을 이름 붙이는 것이 작동 중인 First Law다.

> 💡 **쉬운 설명:** AutomationBench(이전 코스)에서 agent가 `search_tools`로 tool을 찾고 `execute_tool`로 실행하던 그 구조가, 사실은 "외부 SaaS API들이 HATEOAS(다음 행동 링크)를 안 주기 때문에 어쩔 수 없이 따로 만들어 붙인 비-hypermedia 발견 장치"였다는 점을 짚는다. 즉 Level 2 세계에서는 tool 발견 기능을 별도로 덧대야 하고, 그 덧댐 자체가 "Level 2에 머무는 대가"임을 명확히 이름 붙이는 것 — 이것이 First Law(트레이드오프를 직시하라)의 실천이다.

---

## Where This Goes

이 챕터는 boundary의 surface — contract — 와 그것을 진화시키는 비용을 싸게 유지하는 분야들에 값을 매겼다: 필요한 속성에 따라 style을 고르고, additively하게 바꾸고, tolerant reader가 되고, write를 idempotent하게 만들고, "breaking"을 consumer-driven contract test로 검증하라. 두 개의 thread가 의도적으로 매달린 채로 남겨졌다: at-least-once delivery 하의 idempotency, 그리고 모든 외부 response가 boundary를 가로지르는 순간 **outside data**라는 발상.

Ch-06은 정확히 거기서 이어받는다. 그것은 Helland의 root distinction(근본적 구분)([[helland-data-outside-inside]])으로 시작한다 — inside data(private, mutable, ACID, "now") 대 outside data(immutable, versioned, 아마도 stale) — 그리고 데이터가 boundary를 가로지르는 순간 왜 lock과 shared transaction을 잃는지를 보여준다. 그 단 하나의 사실이 consistency 패턴들을 생성한다: **saga**([[richardson-saga]])는 더 이상 가질 수 없는 distributed transaction을 대체하고, **transactional outbox**([[transactional-outbox]]) — 이 챕터의 idempotency 분야의 파트너 — 는 dual-writing 없이 event를 reliable하게 emit하는 방법이다. 당신이 방금 안전하게 진화시키는 법을 배운 contract는 그 모든 outside data가 흐르는 전선(wire)이다.
