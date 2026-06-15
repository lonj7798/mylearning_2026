<!-- chapter: ch-08
     track: evolution
     kind: content
     title: Resilience as Architecture — Stability at the Integration Points
     deps: [[ch-06]]
     sources: [[nygard-release-it]], [[distributed-monolith]], [[richardson-saga]], [[richards-ford-fundamentals]]
-->

# 08장 — Resilience as Architecture: Stability at the Integration Points

> **핵심 통찰.** 시스템이 경계 너머로 손을 뻗는 모든 지점 — network call, broker, shared pool — 은 failure(실패)가 *들어오고* *퍼질* 수 있는 지점이다. Stability(안정성)는 바라는 것이 아니라, *전파를 어디서 멈출지*를 결정함으로써 engineering(공학적으로 설계)하는 것이다: 모든 remote wait(원격 대기)를 timeout으로 묶고, 불안정한 dependency(의존성)를 circuit breaker(회로 차단기)로 감싸고, 자원을 bulkhead(격벽)로 분할하고, fail fast(빠르게 실패)하라. 이것들은 배포 시점에 끼워 넣는 운영상의 사후 처리가 아니다 — *breaker나 bulkhead를 어디에 두느냐가 시스템의 blast radius(피해 반경)를 정의하며*, blast radius는 architectural(아키텍처적) 속성이다. [[distributed-monolith]](분산 모놀리스)는 바로 이런 결정을 건너뛴 시스템이다: breaker 없는 synchronous chain(동기 호출 체인)이라서, 하나의 vendor 장애가 모든 것을 타고 cascade(연쇄)된다.

> **가이드라인.** 모든 integration point(통합 지점)를 안전하다고 증명되기 전까지는 유죄로 취급하라. **timeout** 없이는 절대 remote call을 하지 말라; dependency가 실패하기 시작하면 **circuit breaker를 열어** 시체 위에 요청을 쌓는 대신 빠르게 실패하라; pool을 **bulkhead**로 분리해 익사하는 dependency 하나가 배 전체의 물을 빼지 못하게 하라; 그리고 각각을 *왜* 거기에 두었는지 **ADR**(Architecture Decision Record, 아키텍처 결정 기록)에 기록하라, 왜냐하면 "why(왜)"가 가장 먼저 사라지기 때문이다. 각각을 하나의 베팅으로 가격을 매겨라: 약간의 capability(능력)를 지불하고(일부 false rejection, 분할되어 못 쓰게 된 throughput) 부하 상황에서 실제로 중요한 속성을 되산다 — *total* failure(전면 실패) 대신 *bounded* failure(경계 지어진 실패).

> 💡 **쉬운 설명:** 이 챕터의 큰 그림은 이렇다. 시스템이 다른 시스템(외부 API, DB, 메시지 큐 등)과 연결되는 지점이 곧 위험이 흘러 들어오는 통로다. "resilience(복원력)"란 그 통로를 막거나, 막힐 경우 피해가 거기서 멈추도록 미리 설계해 두는 것이다. 핵심 키워드는 blast radius — 폭탄이 터졌을 때 피해가 미치는 반경처럼, 장애가 발생했을 때 영향이 어디까지 번지는가다. 이 반경을 좁히는 것이 곧 아키텍처 작업이라는 게 이 챕터의 주장이다.

---

## 1. The Spine, Re-Applied: Resilience Is a Decision About Reversibility

이 코스의 중심 주장은 ch-01에서 설치된 것으로, 아키텍처란 되돌리기 비싼 결정들의 집합이며, First Law(제1법칙)는 각 결정을 어떻게 판단할지 알려준다는 것이다:

> "Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet." — Richards & Ford, *Fundamentals of Software Architecture* (book; thesis extracted, quoted as commonly published) — [[richards-ford-fundamentals]]

Resilience pattern은 이 법칙이 가장 세게 물어뜯는 곳인데, 왜냐하면 순진한 해석은 "resilience는 많을수록 항상 좋다 — 어디에나 timeout을, 어디에나 breaker를 넣어라"이기 때문이다. 그것이 First-Law trap(제1법칙의 함정)이다. 너무 일찍 발동하는 timeout은 느리지만-올바른 call을 가짜 실패로 바꿔 버린다. 너무 민감하게 튜닝된 breaker는 잠깐 떨렸을 뿐인 dependency를 가둬 버린다. 너무 잘게 분할하는 bulkhead는 hot path(자주 쓰이는 경로)가 필요로 했던 capacity를 묶어 버린다. 각 pattern은 *priced bet*(가격이 매겨진 베팅)이며, 그 가격은 실재한다.

> 💡 **쉬운 설명:** "trade-off가 없어 보인다면 아직 못 찾은 것뿐"이라는 First Law를 resilience에 적용하면 이렇게 된다. timeout, breaker, bulkhead는 모두 "공짜 안전장치"처럼 보이지만 실제로는 각각 대가가 있다. timeout은 느린 정상 호출을 죽일 수 있고, breaker는 이미 회복된 서비스를 계속 거부할 수 있고, bulkhead는 남는 용량을 빌려 쓰지 못하게 묶는다. 그래서 "무조건 켜라"가 아니라 "어떤 대가를 치르고 무엇을 얻는지 계산하라"가 핵심이다.

이 챕터가 ch-06의 consistency mechanics 옆이 아니라 **evolution**(진화) 단계에 위치하는 이유는 [[richards-ford-fundamentals]]에 명시된 through-line(관통선) 때문이다: evolution은 *베팅을 수정 가능하게 유지하는 것*에 관한 것이다. Stability pattern은 outage(장애)가 one-way door(돌아올 수 없는 문)가 되지 않게 하는 방법이다. 그것들 없이는, 단 하나의 dependency 실패가 state를 오염시키고, pool을 고갈시키고, 관련 없는 capability까지 무너뜨릴 수 있다 — 그리고 *그것*으로부터 복구하는 것이 모든 reversal(되돌리기) 중 가장 비싸다. 그것들이 있으면, failure는 *bounded*된다: 네가 할당한 compartment(구획) 안에 머물고, 나머지 시스템은 망가진 부분을 고치는 동안 계속 돌아간다. 그것은 ch-09가 strangler-fig migration과 fitness function으로 더 느린 시간 척도에서 할 바로 그 동작이다 — 이 pattern들이 *failure*의 blast radius를 묶듯, *change*(변경)의 blast radius를 묶는다.

### 1.1 Architecture characteristics pick which failures you must survive

resilience는 공짜로 얻어지지 않으며, 균일하게 필요하지도 않다. [[richards-ford-fundamentals]]에 따르면 아키텍트의 첫 번째 임무는 모든 것을 최대화하는 것이 아니라 요구사항으로부터 *critical few*(핵심 소수) architecture characteristics(아키텍처 특성)를 *도출*하는 것이다. **Fault tolerance**(내결함성)와 **availability**(가용성)는 resilience pattern을 테이블에 올리는 "-ility"(특성)다; 만약 어떤 subsystem이 살아 있는 호출자가 없는 야간 batch job이라면, 공격적인 circuit breaker는 아무것도 사주지 않으면서 복잡성만 비용으로 청구한다. Resilience는 *요구사항이 요구하는 곳에* 지출하는 특성이다 — 그리고 신뢰할 수 없는 외부 세계를 마주하는 integration point가 바로 요구사항이 요구하는 곳이다.

> 💡 **쉬운 설명:** "모든 곳에 resilience를 넣자"는 잘못이다. 예를 들어 새벽에 한 번 돌고 끝나는 배치 작업에는 실시간 사용자가 없으니, breaker가 거부해 봐야 반응할 사람도 없고 복잡성만 늘어난다. 반대로 외부 SaaS API를 실시간으로 부르는 경로는 언제든 외부 장애가 새어 들어올 수 있으니 resilience가 꼭 필요하다. 즉 "어디에 fault tolerance / availability가 진짜 critical한가"를 먼저 정하고, 그곳에만 자원을 쓰라는 것이다.

### 1.2 The second Nygard contribution: record the why

[[nygard-release-it]]는 하나가 아니라 두 개의 아이디어다. 첫 번째는 stability pattern이다. 두 번째는, 화려하지 않아서 잊기 쉽지만, **Architecture Decision Record**다 — 그리고 이 둘은 정확히 *resilience 결정이야말로 그 근거가 가장 빨리 썩는 결정들이기 때문에* 함께 묶인다. 2초의 timeout budget, 30초의 breaker cooldown, 20개 connection으로 sizing된 bulkhead: 이것들은 6개월 뒤에 보면 자의적인 magic number(마법의 숫자)처럼 보이고, 왜 그렇게 선택되었는지 이해하지 못한 채 "정리"하거나 "튜닝"하고 싶은 유혹이 바로 신중하게 bounded된 blast radius가 조용히 스스로 un-bounded되는 방식이다. Nygard가 ADR이 푸는 문제를 framing한 방식이 이 둘을 짝지을 전체 이유다:

> "One of the hardest things to track during the life of a project is the motivation behind certain decisions." — Nygard, "Documenting Architecture Decisions" — [[nygard-release-it]]

> "An architecture decision record is a short text file in a format similar to an Alexandrian pattern." — Nygard — [[nygard-release-it]]

그 구조 — *Title, Status*(proposed/accepted/deprecated/superseded), *Context, Decision, Consequences* — 는 의도적으로 작고("one or two pages") "미래 개발자와의 대화"가 되도록 의도되었다. resilience 결정의 경우 *Consequences* 필드가 이 챕터가 계속 주장하는 trade-off가 실제로 기록되는 곳이다: "우리는 vendor X가 다운되었을 때 worker pool을 고갈시키지 않는 대가로 30초 cooldown 동안의 false rejection을 받아들인다." 문서화되지 않은 resilience parameter는 아무도 그 가격을 볼 수 없는 베팅이다 — 그것이 그 parameter가 제거되는 방식이다.

> 💡 **쉬운 설명:** ADR은 "이 결정을 왜 했는가"를 1~2쪽짜리 짧은 텍스트 파일로 남기는 관행이다. resilience 숫자들(예: timeout 2초, cooldown 30초)은 시간이 지나면 누군가 "왜 이렇게 이상한 값이지? 깔끔하게 바꾸자"며 손대기 쉽다. 그런데 그 값에는 보통 이유가 있었다(예: vendor의 p99 지연이 1.8초라서 2초로 잡음). Consequences 칸에 "이 값을 쓰면 cooldown 동안 멀쩡한 요청도 거부된다, 대신 pool 고갈을 막는다"라고 적어 두면, 미래의 개발자가 무심코 안전장치를 떼어내는 사고를 막을 수 있다.

---

## 2. Cascading Failure: The Anti-Pattern That Motivates Everything

애초에 stability를 engineering하는 이유는 단 하나의 failure mode이며, 어떤 치료법을 명명하기 전에 그것을 정확히 명명할 가치가 있다. [[nygard-release-it]](Michael Nygard, *Release It!* 2e — book, thesis extracted; csabapalfi/release-it notes와 Pragmatic Bookshelf 자료로 corroborate됨)로부터:

한 integration point에서의 failure는 "from subsystem to subsystem crashing each one"으로 전파된다. 그것이 **cascading failure**(연쇄 실패)다: 주변 시스템들이 서로를 끌어내릴 만큼 충분히 tightly coupled(강하게 결합)되어 있기 때문에 국소적으로 머물지 않는 localized fault(국소적 결함). Nygard의 가장 흔한 원인에 대한 직설적 요약:

> "Integration Points without Timeouts is a surefire way to create Cascading Failures." — Nygard, *Release It!* (book; thesis extracted) — [[nygard-release-it]]

그 mechanism(메커니즘)은 평범하며, 그것이 그것을 위험하게 만드는 점이다. Dependency D가 느려진다 — crash하는 게 아니라, 그냥 느려진다. 이제 D의 모든 caller는 30밀리초가 아니라 30초가 걸리는 응답을 기다리며 thread(또는 connection, 또는 event-loop slot)를 붙잡고 있다. 그 붙잡힌 자원들은 *finite*(유한)하다. Pool이 고갈된다. 새 요청들 — D와 전혀 관계없는 것들조차 — 이제 free thread를 기다리며 block된다. caller의 caller는 *자신의* dependency(즉 너)가 느려지는 것을 보고, 같은 고갈이 한 단계 위에서 일어난다. 단 하나의 느린 vendor가 이제 전체 service graph를 멈춰 세웠고, 누구의 database도 다운되지 않았다. **unbounded wait는 붙잡힌 자원이고, 붙잡힌 자원이 바로 하나의 느린 dependency가 전체 시스템을 멈추게 하는 방식이다.**

> 💡 **쉬운 설명:** cascade가 어떻게 일어나는지 구체적으로 보자. D가 죽는 게 아니라 그냥 느려지는 게 더 위험하다. crash라면 호출이 즉시 실패하고 끝나지만, 느려지면 모든 호출자가 응답을 기다리며 thread를 붙잡는다. thread(또는 DB connection) 개수는 정해져 있으니, 곧 풀이 바닥나고, D와 무관한 다른 요청들까지 "빈 thread가 없어서" 멈춘다. 그러면 너를 부르는 상위 서비스가 보기에 *네가* 느려진 것이고, 같은 고갈이 위로 번진다. DB도 멀쩡하고 코드도 멀쩡한데 전체가 멈추는 것 — 이게 cascading failure의 무서움이다.

이 함정에는 모든 아키텍트가 가격에 반영해야 할 second-order accelerant(2차 가속 요인)가 있다: *retry storm*(재시도 폭풍). call이 실패했을 때의 선의의 본능은 그것을 retry하는 것이다. 하지만 D가 *overloaded*(과부하)되어 느린 것이라면, 모든 retry는 이미 익사하고 있는 dependency에 부하를 더해, 그것을 더 깊이 밀어 넣고 회복 가능한 brownout(부분 장애)을 단단한 outage로 바꾼다. circuit breaker 없는, 그리고 backoff(지수 후퇴) 없는 retry는 resilience가 아니다 — failure에 직접 연결된 amplifier(증폭기)다. 이것이 정확히 [[nygard-release-it]]가 timeout을 breaker와 짝짓는 이유다: timeout은 하나의 wait를 묶지만, breaker만이 *이미 실패하고 있다고 알고 있는 dependency에 대한 부하 생성을 멈춘다*. 도움이 될 것처럼 보이는 pattern(더 세게 retry하기)이 cascade를 collapse(붕괴)로 바꾸는 바로 그것이며, 이것은 First Law가 resilience toolkit 자체 안에서 나타나는 것이다 — "retry"조차 비용이 있고, 그 비용은 storm이다.

> 💡 **쉬운 설명:** retry storm은 직관과 정반대로 작동한다. 보통 "실패하면 다시 해보자"가 옳아 보이지만, 상대가 과부하로 느린 거라면 retry는 불난 집에 기름을 붓는 격이다. 1000명이 동시에 재시도하면 이미 허덕이는 D는 더 빨리 완전히 죽는다. 그래서 retry만으로는 resilience가 아니라 오히려 증폭기이며, 반드시 (1) breaker로 "죽은 줄 아는 곳엔 그만 보내고", (2) backoff로 "재시도 간격을 점점 늘려" 부하를 줄여야 한다.

### 2.1 The distributed monolith is the cascading-failure machine

이것이 이 챕터를 topology(ch-04)로 다시 묶는 바로 그 failure다. [[distributed-monolith]]는 anti-pattern의 네 가지 징후를 열거하는데; 네 번째가 cascading failure 그 자체다:

> "Cascading failures — tight runtime coupling means one slow dependency drags down the entire workflow (the failure mode [[nygard-release-it]] exists to stop)." — [[distributed-monolith]] (community synthesis; not a single canonical author article)

그리고 두 번째 징후인 *synchronous coupling*(동기 결합)은 그 structural precondition(구조적 전제 조건)이다:

> "Synchronous coupling — a request fans out through a chain of real-time blocking calls instead of an async event or a message broker. Any link's latency or outage stalls the whole chain." — [[distributed-monolith]]

그래서 resilience toolkit과 distributed-monolith trap은 한 동전의 양면이다. distributed monolith는, 정의상, service로 분해했지만 stability pattern을 건너뛴 시스템이다: synchronous blocking chain을 유지한 채 network를 새로운 failure source로 추가한 것이다. [[distributed-monolith]] 자신의 trade-off framing이 그 비용을 명시적으로 만든다 — 그것은 "all the pain of distributed systems without the independence that makes microservices worthwhile"(microservices를 가치 있게 만드는 독립성 없이 분산 시스템의 모든 고통만 가진 것)이다. 아래의 pattern들은 failure를 *contain*(담아두는) service graph와 그것을 *conduct*(전도하는) service graph의 차이다.

> 💡 **쉬운 설명:** distributed monolith는 "겉은 microservices, 속은 모놀리스"인 안티패턴이다. 서비스를 여러 개로 쪼개 놓았지만 여전히 서로를 동기 호출로 줄줄이 부르고(synchronous coupling) breaker도 없으면, 모놀리스의 단점(강결합)은 그대로인데 네트워크 장애라는 새 위험까지 추가된다. 즉 분리의 장점(독립 배포·독립 장애)은 못 얻고 분산의 고통만 얻는다. 이 챕터의 pattern들이 바로 그 차이를 만든다: 장애를 가두는(contain) 구조냐, 전기처럼 흘려보내는(conduct) 구조냐.

---

## 3. The Stability Toolkit — Each Pattern Priced as a Bet

[[nygard-release-it]]는 toolkit을 준다. 이 코스가 요구하는 규율은 그중 어느 것도 "best practice"가 아니라는 것이다 — 각각은 특정 force(힘)가 존재할 때 네가 하는 거래다. 여기 toolkit이 있으며, 각 항목에 그것이 *변경을 싸게 유지하는 것*과 *비싸게 만드는 것*을 함께 적었다.

| Pattern | What it does (Nygard) | Keeps cheap | Makes expensive / costs |
|---|---|---|---|
| **Timeout** | "bound every remote wait" | caller의 liveness — thread를 영원히 붙잡지 않음 | tuning: 너무 짧으면 = 느리지만-올바른 call에서 false failure |
| **Circuit Breaker** | dead dependency 위에서 open하고 fail fast; recovery 테스트를 위한 half-open | outage 동안 caller latency & pool health | dep이 이미 회복했더라도 OPEN 동안 false rejection |
| **Bulkhead** | "partition resources… so a failure in one area can't drain the whole" | isolation — 익사하는 dependency 하나가 무관한 작업을 가라앉히지 못함 | utilization: idle partition에 묶인 stranded capacity |
| **Steady State** | 모든 accumulation에는 짝이 되는 cleanup이 있음 | unattended uptime — 사람의 babysitting 불필요 | 모든 log/session/cache에 대한 cleanup의 사전 설계 |
| **Fail Fast** | 성공할 수 없음을 감지하고 즉시 반환 | blast radius — "only the subsystem where the error occurred is affected" | 성공했을 수도 있는 optimistic retry를 포기 |

### 3.1 Timeout — the non-negotiable floor

timeout은 pattern 중 가장 싸고 가장 보편적이며, 누락에 대한 변명이 가장 적은 것이다. Nygard의 규칙은 무조건적이다: *bound every remote wait*. 그 베팅은 거의 공짜다 — "조금만 더 기다렸다면 성공했을 call"의 사라지는 tail(꼬리)을 포기하는 대가로, 어떤 단일 느린 dependency도 너의 자원을 무한정 고정할 수 없음을 보장받는다. 유일한 실제 비용은 tuning이다: dependency의 정당한 p99보다 짧은 timeout은 healthy-but-slow call을 가짜 실패로 변환하고, 이것은 (retry와 결합되어) *그 자체로* 부하 증폭 cascade가 될 수 있다. 아키텍처 결정은 timeout을 *할지 말지*가 아니라 *budget이 무엇인지*이며, 그 budget은 네가 commit한 latency characteristics로부터 도출되어야 한다.

> 💡 **쉬운 설명:** p99란 "전체 요청을 빠른 순으로 줄 세웠을 때 99번째 백분위수의 지연 시간", 즉 가장 느린 1%를 제외한 최악의 응답 시간이다. timeout을 정상 p99보다 짧게 잡으면 평소 멀쩡한 느린 호출까지 죽인다. 그러니 timeout 값은 감으로 찍는 게 아니라 "우리가 약속한 지연 특성(p99)"에서 거꾸로 계산해야 한다. timeout 자체는 거의 공짜지만, 값을 잘못 잡으면 오히려 cascade의 원인이 될 수 있다는 점이 핵심이다.

### 3.2 Circuit Breaker — fail fast over a corpse

timeout은 *단일* call을 묶는다. 하지만 dead dependency로 향하는 call의 stream(흐름)은, 각각이 실패하기 전에 충실하게 자신의 full timeout을 끝까지 기다리며, 여전히 slow-motion 재앙이다: 너는 이미 다운된 것을 아는 dependency에, 요청당, 너의 전체 timeout budget을 쓰고 있다. circuit breaker는 *기억하는* pattern이다. [[nygard-release-it]]로부터:

> "Circuit Breaker — track failures to a dependency; once over threshold, *open* the circuit and fail fast (skip the call) instead of piling up requests on a dead service. Periodically *half-open* to test recovery." — [[nygard-release-it]] (Nygard popularized this pattern)

state machine(상태 기계)이 pattern의 전부이며, 읽기보다 진정으로 *조작*해 볼 가치가 있다. 아래 companion을 열고, dependency를 fail시키고, 요청 burst(폭발)를 보내라: breaker가 OPEN으로 trip(작동)되어 ~0ms에 거부를 시작하는 것을(call이 dead vendor에 절대 닿지 않음) 보고, 그런 다음 dependency를 heal(회복)시키고 cooldown이 지나 HALF-OPEN이 되게 하면, 거기서 단일 probe(탐침)가 close할지 결정한다.

> **▶ Interactive:** [`figures/circuit-breaker.html`](figures/circuit-breaker.html) — *Send request* / *Send 5 requests*를 클릭하고, dependency를 *down*으로 토글하고 *heal*하며, trip threshold와 cooldown을 튜닝하라. "Calls fast-failed (OPEN)"과 "Latency last call" 카운터를 보라: OPEN 동안 latency는 ~0ms인데, breaker가 시체에 절대 닿지 않기 때문이다.

세 가지 state가 베팅의 가격을 정확히 매긴다:

- **CLOSED** — call이 통과한다; breaker가 failure를 센다. 비용: 아직 없음; threshold가 trip될 때까지 각 실패하는 call마다 full timeout을 지불한다.
- **OPEN** — fail fast, call을 skip. *여기가 breaker가 밥값을 하는 곳이다*: caller latency가 ~0으로 떨어지고 pool이 고갈을 멈춘다. 비용: **false rejection** — dependency가 30초 cooldown 중 1초 만에 회복하면, 그 window의 나머지 동안 완벽하게 처리 가능한 요청을 거부한다.
- **HALF-OPEN** — 정확히 하나의 probe만 통과시킨다. 성공하면 close; 실패하면 re-open하고 cooldown을 재시작한다. 비용: recovery가 단일 probe의 운에 달려 있어서, *간헐적으로* 회복하는 dependency는 flap(깜빡거림)할 수 있다.

아키텍처 결정은 breaker가 *어디에* 사는지(이상적으로는 per-dependency — global breaker 하나가 아니라 external vendor당 breaker 하나)와, threshold 및 cooldown이 false-rejection을 responsiveness(반응성)와 어떻게 trade하는지다. 그 parameter들은 숫자 형태의 architecture characteristics다.

> 💡 **쉬운 설명:** circuit breaker를 전기 차단기에 비유하면 이해가 쉽다. 평소엔 전류가 흐른다(CLOSED). 누전(연속 실패)이 임계치를 넘으면 차단기가 내려가(OPEN) 더 이상 전류를 안 보낸다 — 그래서 호출이 죽은 vendor에 닿지도 않으니 응답이 0ms로 즉시 거부된다. 일정 시간(cooldown) 뒤엔 "이제 고쳐졌나?" 확인하려고 딱 한 번 흘려본다(HALF-OPEN). 성공하면 복구(CLOSED), 실패하면 다시 내린다. 대가는 OPEN 상태에서 상대가 이미 회복했어도 cooldown이 끝날 때까지 멀쩡한 요청을 거부한다는 점이다.

quietly defeat the pattern(조용히 pattern을 무력화하는) 두 가지 placement(배치) 실수를 명명할 가치가 있다. 첫째, 많은 dependency 앞의 **single global breaker**(단일 전역 breaker): 그것이 trip되면, *모든 것*에 대해 fail fast하므로, 아픈 vendor 하나가 건강한 여덟 개로의 call을 죽인다 — 너는 막으려던 cascade를 breaker를 conductor(전도체)로 삼아 재창조한 것이다. breaker는 *unit of failure*(실패의 단위), 즉 개별 dependency에 scope(범위 한정)되어야 한다. 둘째, **breaker with no timeout underneath it**(아래에 timeout이 없는 breaker): breaker는 failure만 세는데, "call이 hanging 중"은 *무언가*가 포기해야만 — 즉 timeout이 있어야만 — failure가 된다. unbounded wait 위의 breaker는 절대 trip되지 않는데, call이 실패를 끝내지 않기 때문이다; 그냥 hang한다. timeout과 breaker는 대안이 아니라 *pair*(짝)다: timeout은 hang을 셀 수 있는 failure로 변환하고, breaker는 그 count를 사용해 시도를 멈춘다. 이것이 [[nygard-release-it]]가 timeout과 circuit breaker를 cascading failure에 대한 "the two most effective counters"(가장 효과적인 두 대항책)라고 부르는 정확한 의미다 — 그것들은 함께여야만 작동한다.

> 💡 **쉬운 설명:** 두 가지 흔한 실수를 풀어 보자. (1) 전역 breaker 하나로 여러 vendor를 막으면, vendor 하나만 아파도 차단기가 내려가 멀쩡한 나머지 vendor 호출까지 다 끊긴다 — 막으려던 cascade를 breaker가 직접 일으키는 꼴이다. 그래서 vendor마다 breaker를 따로 둬야 한다. (2) timeout 없는 breaker는 무용지물이다. breaker는 "실패 횟수"를 세서 작동하는데, timeout이 없으면 호출이 영원히 매달려 있을 뿐 "실패"로 집계되지 않으니 breaker가 영영 안 내려간다. 그래서 timeout과 breaker는 반드시 한 쌍으로 써야 한다.

### 3.3 Bulkhead — the ship's-hull metaphor, taken literally

breaker는 하나의 *named* dependency로부터 너를 보호한다. bulkhead는 한 dependency의 failure가 *다른 dependency가 필요로 하는 자원으로 흘러넘치는 것*으로부터 너를 보호한다. [[nygard-release-it]]의 metaphor는 정확하다:

> "Just as a ship's hull is divided into watertight compartments so that a breach in one section does not sink the vessel." — [[nygard-release-it]]

구체적으로: 각 downstream dependency에 *자신의* connection pool / thread pool / semaphore(세마포어)를, 독립적으로 sizing해서 주어라. vendor A가 느려져서 *자신의* pool을 saturate(포화)시키면, vendor B의 caller들은 영향받지 않는데, B가 자신의 compartment를 가지기 때문이다. bulkhead 없이는, A와 B가 하나의 pool을 공유하고, A의 둔화가 공유 pool을 고갈시켜, B가 완벽하게 건강한데도 B를 굶긴다 — 다시 cascade이며, 이제는 직접적인 call chain이 아니라 resource contention(자원 경합)을 통해서다. 그 베팅: *utilization*(조용한 partition에 묶인 idle capacity는 바쁜 partition이 빌려 쓸 수 없음)을 *isolation*(범람한 partition이 나머지를 가라앉힐 수 없음)과 trade한다. total failure의 비용이 불완전한 resource sharing의 비용을 압도할 때, 이 가격을 의도적으로 지불한다.

> 💡 **쉬운 설명:** bulkhead는 배의 방수 격벽에서 따온 이름이다. 배 밑을 여러 칸으로 나눠 두면 한 칸에 구멍이 나도 그 칸만 물이 차고 배는 안 가라앉는다. 소프트웨어에서는 vendor마다 connection/thread pool을 따로 떼어 주는 것이다. 한 pool을 공유하면 느린 vendor A가 그 pool을 다 차지해, A와 무관한 B 호출까지 빈 자리가 없어 굶는다. pool을 분리하면 A가 자기 칸만 채우고 B는 멀쩡하다. 대가는 utilization(자원 활용률) 손해다 — B 칸이 한가해도 그 여유를 A가 빌려 쓸 수 없으니까.

### 3.4 Fail Fast and Steady State — the unglamorous two

**Fail Fast**는 circuit breaker가 operationalize(운용화)하는 원칙을 일반적으로 진술한 것이다: 성공할 수 없음을 감지하고 *즉시* 반환해서, caller가 hang하는 대신 반응할 수 있게(degrade(성능 저하), queue, friendly error 반환) 한다. Nygard의 framing은 그 payoff를 직접 명명한다:

> "the idea is to fail as fast as you can so that only the subsystem where the error occurred is affected." — [[nygard-release-it]]

그 절 — *only the subsystem where the error occurred is affected*(오류가 발생한 subsystem만 영향받는다) — 은 한 문장으로 된 blast-radius 아이디어이며, 이것이 fail-fast가 cosmetic(겉치레)이 아니라 architectural인 이유다.

**Steady State**는 slow-burn(서서히 타는) 짝이다: 모든 accumulation(축적)(log, session, cache, temp file)에는 짝이 되는 cleanup이 있어야 하며, 그래서 시스템이 새벽 3시에 사람이 disk를 정리하거나 process를 재시작하지 않고도 unattended(무인)로 돌 수 있다. 그것은 *dependency*가 아니라 *time*(시간)에 맞서 방어하는 resilience pattern이며, failure mode가 초 단위가 아니라 주 단위로 멀리 있기 때문에 가장 자주 건너뛰는 것이다. 여기서의 베팅은 다른 것들과 반대로 뒤집혀 있다: runtime 비용은 거의 없고, 오직 *design-time* 비용(네가 축적하는 모든 자원의 lifecycle을 생각해 둬야 함)만 있으며, 그것을 *지불하지 않는* 것의 failure는 가장 창피한 종류다 — 한 달 동안 "stable"했다가 table이 무한정 자라서 쓰러진 서비스.

> 💡 **쉬운 설명:** Steady State는 "쌓이는 것마다 비우는 짝을 만들어 둬라"는 원칙이다. 로그, 세션, 캐시, 임시 파일은 가만 두면 계속 쌓인다. 이걸 정리하는 로직이 없으면 시스템은 처음엔 멀쩡하다가 몇 주 뒤 디스크나 테이블이 꽉 차서 갑자기 죽는다. 다른 pattern들과 달리 런타임 비용은 거의 없고, "설계할 때 모든 자원의 수명을 미리 생각해 두는" 수고만 든다. 그래서 가장 자주 빼먹고, 빼먹으면 가장 민망한 방식(새벽 3시에 호출되는 장애)으로 터진다.

### 3.5 The rest of the toolkit, and why size matters

[[nygard-release-it]]는 위 다섯 개보다 많은 pattern을 열거한다 — Handshaking, Shed Load, Back Pressure, Governor, Let It Crash — 그리고 이 family는 하나의 shape를 공유한다: 각각은 시스템이 uncontrolled failure(통제되지 않은 실패)가 대신 "아니오"라고 말하기 전에 *통제된 방식으로 아니오라고 말하는* mechanism이다. **Back Pressure**(배압)는 느린 consumer가 조용히 unbounded queue를 쌓는 대신 upstream에 느려지라고 신호하게 한다(다른 옷을 입은 Steady-State failure). **Shed Load**(부하 차단)는 시스템이 capacity를 넘었을 때 edge에서 요청을 drop하여, 전부를 무작위로 실패시키는 대신 *어떤* 요청을 실패시킬지 선택한다. **Governor**(조속기)는 자동화된 action을 의도적으로 늦춰서, 자동화된 실수(runaway script, 오작동하는 agent loop)가 사람이 알아채기 전에 unbounded damage를 줄 수 없게 한다. 이들 모두에 걸친 공통 trade-off: *약간의* 성공적 작업 — 거부된 요청, throttle된 throughput, 늦춰진 자동화 — 을 지불하여 예측 불가능한 collapse 대신 *bounded, predictable*한 degradation을 산다. 어느 것도 어디에나 켜는 default가 아니다; 각각은 요구사항이 "controlled partial failure가 uncontrolled total failure보다 낫다"고 말하는 곳에서만 올바른 베팅이며, 그것이 바로 availability/fault-tolerance characteristics가 제 일을 하는 것이다.

> 💡 **쉬운 설명:** 나머지 pattern들의 공통 주제는 "시스템이 스스로, 통제된 방식으로 '아니오'라고 말하기"다. Back Pressure는 "나 지금 벅차니 천천히 보내"라고 상류에 신호하는 것(안 그러면 큐가 무한정 쌓임), Shed Load는 용량 초과 시 입구에서 일부 요청을 일부러 버려 나머지를 살리는 것, Governor는 자동화의 속도를 일부러 늦춰 폭주(runaway agent loop 같은)가 사람이 눈치채기 전에 큰 피해를 못 주게 하는 것이다. 셋 다 "약간의 성공을 포기하고 예측 가능한 degradation을 산다"는 같은 거래다. 특히 Governor는 LLM agent 설계에 직접 와닿는다.

---

## 4. The Myth This Chapter Kills: "Resilience Is an Ops Problem"

대중적 narrative는 timeout, breaker, bulkhead를 operational knob(운영 손잡이) — SRE 영역, 아키텍처가 "done"된 후에 service mesh나 load balancer에서 튜닝하는 것 — 으로 취급한다. primary source(1차 출처)는 동의하지 않으며, 그 reconciliation(조정)이 이 챕터의 척추다.

| Popular narrative | What the primary source says | Resolved in |
|---|---|---|
| "Resilience / stability is an ops concern you add after the design." | Stability pattern은 *architectural*하다: circuit breaker와 bulkhead를 어디에 두느냐가 **시스템의 blast radius를 정의**하며, blast radius는 설계 시점에 결정되는 structural 속성이다. [[distributed-monolith]]는 정확히 그것들을 건너뛴 시스템이다. | [[nygard-release-it]], [[distributed-monolith]] |

[[nygard-release-it]]는 그것을 직접 진술한다: "These are *architectural* decisions: where you place circuit breakers and bulkheads defines your system's blast radius." 모든 service를 pool을 공유하는 synchronous chain으로 이미 배선한 후에는 blast-radius 제어를 retrofit(나중에 끼워 넣기)할 수 없다 — 그때쯤이면 cascade path가 topology에 구워져 있다. *어떤* integration point가 breaker를 얻고, *어떤* 자원이 자신의 bulkhead를 얻고, call이 애초에 *synchronous인지 아닌지*의 결정은 boundary를 그릴 때 내리는 것이지 deploy할 때가 아니다. 이것이 ch-01의 C4 *Container* diagram이 distributed monolith가 보이게 되는 곳인 이유다: container 사이의 synchronous arrow들이, 그 위에 breaker 없이, 그 아래에 shared data store와 함께, *cascade path를 그려낸 것 그 자체*다.

> 💡 **쉬운 설명:** 흔한 오해는 "resilience는 SRE/운영팀이 배포 후에 service mesh에서 튜닝하는 것"이라는 생각이다. 하지만 어디에 breaker/bulkhead를 둘지, 어떤 호출을 동기로 둘지는 *경계를 그리는 설계 시점*에 정해진다. 이미 모든 서비스를 공유 pool로 동기 체인 연결해 놓고 나면, cascade 경로가 구조에 박혀서 나중에 손볼 수 없다. C4 Container 다이어그램에서 container 사이를 잇는 동기 화살표에 breaker가 없고 아래에 공유 DB가 있다면, 그 그림 자체가 이미 그려진 cascade 경로다.

resolution의 다른 절반은 그 *cure*(치료)가 단지 pattern 결정이 아니라 부분적으로 boundary 결정이라는 것이며 — 그것이 다음 개념으로의 다리다.

---

## 5. Async/Event Integration Is Itself a Resilience Decision

가장 강력한 stability 동작은 synchronous call에 breaker를 추가하는 것이 아니다 — 그 call을 *애초에 synchronous가 아니게* 만드는 것이다. [[nygard-release-it]]는 이것을 명시적으로 만든다:

> "Choosing async/event integration (→ [[richardson-saga]]) is itself a resilience decision: it removes the synchronous coupling these patterns otherwise have to defend." — [[nygard-release-it]]

여기서 ch-08은 그것이 의존하는 consistency 챕터(ch-06)로 손을 뻗는다. synchronous request-chain은 *temporal*(시간적) coupling을 가진다: A는 B를 기다려야 하고 B는 C를 기다려야 하므로, C의 latency가 A의 latency이고 C의 outage가 A의 outage다. 그것이 timeout/breaker/bulkhead toolkit이 *방어*하기 위해 존재하는 정확한 coupling이다. event-driven integration은 coupling을 방어하는 대신 *제거*한다: A는 event를 emit하고 끝난다; B와 C는 할 수 있을 때 그것을 consume한다. C가 다운되어 있으면, event는 broker에서 기다리고; upstream에서는 아무것도 stall(멈춤)하지 않는다. 이것이 정확히 [[richardson-saga]]의 saga 구조다:

> "A saga is a sequence of local transactions. Each local transaction updates the database and publishes a message or event to trigger the next local transaction in the saga." — Chris Richardson, microservices.io/patterns/data/saga.html — [[richardson-saga]]

각 step이 blocking call이 아니라 event로 접착된 local transaction이기 때문에, downstream outage는 전체 operation을 *실패*시키는 대신 한 step을 *연기*한다. 하지만 — 그리고 이것이 trade-off이며, 결코 공짜 점심이 아니다 — 너는 비용을 제거한 것이 아니라, *옮긴* 것이다. synchronous chain의 비용은 cascading latency였고; saga의 비용은 **loss of isolation**(격리의 상실)이다:

> "Lack of isolation (the 'I' in ACID)… means there's risk that the concurrent execution of multiple sagas and transactions can [cause] data anomalies." — Richardson — [[richardson-saga]]

그래서 resilience 결정과 consistency 결정은 두 각도에서 본 *같은* 결정이다. synchronous로 가면 ACID-like 단순성을 유지하지만 cascade에 맞서 모든 link를 timeout + breaker + bulkhead로 방어해야 한다. async/event로 가면 cascade coupling을 녹이지만 saga의 countermeasure burden(대응책 부담)을 상속한다 — semantic lock(의미적 잠금), commutative update(교환 가능한 갱신), re-read(재읽기), by-value tracking(값 기반 추적)(saga 뒤에 있는 1987년의 원래 long-lived-transaction 아이디어; 출처 PDF가 image-only라서 이 계보는 verbatim 인용이 아니라 thesis-extracted됨 — [[richardson-saga]]). 둘 다 공짜로 사주는 옵션은 없다. 아키텍트의 일은 그들이 어떤 청구서를 지불하기로 선택하는지 아는 것이다.

> 💡 **쉬운 설명:** 가장 센 resilience 동작은 "동기 호출에 breaker를 다는 것"이 아니라 "아예 동기 호출을 안 쓰는 것"이다. 동기 체인 A→B→C는 C가 느리면 A도 느리고 C가 죽으면 A도 죽는 시간적 결합이 있다. 반면 event 방식은 A가 event만 던지고 끝나고, C가 죽어 있으면 event는 broker 큐에서 기다릴 뿐 상류는 안 멈춘다 — 이게 saga다. 단, 공짜는 아니다. 동기의 대가가 "연쇄 지연"이었다면, saga의 대가는 "isolation 상실"이다(ACID의 I). 즉 여러 saga가 동시에 돌면 중간 상태가 서로 보여서 데이터 이상이 생길 수 있고, 이를 막으려고 semantic lock 같은 대응책을 추가로 짜야 한다. resilience 선택과 consistency 선택이 사실 같은 결정의 양면이라는 게 이 절의 핵심이다.

### 5.1 Choreography vs orchestration is also a resilience choice

async/event 결정 *안에서조차*, saga의 두 coordination style(조율 방식)은 failure 하에서 다르게 trade한다. [[richardson-saga]]가 그것들을 framing한다:

> "Choreography - each local transaction publishes domain events that trigger local transactions in other services." — Richardson — [[richardson-saga]]
> "Orchestration - an orchestrator (object) tells the participants what local transactions to execute." — Richardson — [[richardson-saga]]

resilience-relevant 차이는 *failure-handling logic이 어디에 사는가*다. **Choreography**(코레오그래피)는 더 낮은 coupling과 단일 failure point 없음을 가진다 — 하지만 compensation logic(보상 로직)이 event에 반응하는 service들에 흩어져 있어서, 무언가 잘못되면 recovery path를 *보기*가 어렵고, 그것 자체가 availability 위험이다(추적할 수 없는 것은 빠르게 고칠 수 없다). **Orchestration**(오케스트레이션)은 saga의 control flow(제어 흐름) — 그리고 따라서 그것의 timeout, retry, compensation 처리 — 를 네가 추론하고 observe할 수 있는 한 곳에 집중시키며, 그 대가로 그 orchestrator를 자신의 failure가 workflow를 stall시키는 component로 만든다(그래서 그것 역시 자신의 resilience와 자신의 bulkhead가 필요하다). 길고 multi-step인 agent workflow의 경우 orchestration style이 보통 *operability*(운용성)에서 이기는데, 정확히 failure-handling이 centralized되고 observable하기 때문이며 — 이것이 §6으로 곧장 이어진다.

> 💡 **쉬운 설명:** saga를 조율하는 두 방식의 차이는 "장애 처리 로직이 어디 있느냐"다. Choreography는 지휘자 없이 각자 event를 보고 알아서 움직이는 방식 — 결합도는 낮지만 보상/복구 로직이 여러 서비스에 흩어져 있어 문제 추적이 어렵다. Orchestration은 지휘자(orchestrator) 하나가 "다음은 너, 그다음은 너"라고 지시하는 방식 — 모든 timeout/retry/보상이 한 곳에 모여 추적·관찰이 쉽지만, 그 지휘자가 죽으면 전체가 멈추니 지휘자에게도 별도 resilience가 필요하다. 길고 단계 많은 agent workflow에서는 관찰·운영이 쉬운 orchestration이 보통 유리하다.

---

## 6. Observability Is an Architectural Choice, Not a Dashboard

마지막 개념은 stability를 failure를 *예방*하는 것에서 그것을 *보는* 것으로 옮긴다. 무엇이 observable(관찰 가능)해야 하는지 결정하는 것은, [[richards-ford-fundamentals]]에 따르면, critical architecture characteristics에 묶인 아키텍처 결정이다 — 마지막에 뿌리는 것이 아니다.

그 link는 직접적이다: circuit breaker는 그것이 trip되었음을 알 수 있는 너의 능력만큼만 좋고, breaker의 trip은 dependency가 아프다는 first-class signal(일급 신호)이다 — raw latency graph보다 훨씬 더 actionable(조치 가능)하다. 만약 네가 주어진 path에 대해 **availability**가 critical characteristic이라고 결정했다면, *그 path의 breaker state, timeout-fire rate, 그리고 bulkhead saturation이 observable해야 한다*, 왜냐하면 그것들이 네가 보호하기로 commit한 characteristic의 leading indicator(선행 지표)이기 때문이다. 그것들을 선택하는 것이 네가 무엇에 반응할 수 있는지를 선택하는 것이다.

여기 그 포인트를 날카롭게 만드는 미묘한 inversion(역전)이 있다: resilience pattern은 *네가 가장 observe하고 싶은 바로 그 signal을 생성한다*. OPEN으로 전환되는 breaker는 timeout error의 홍수보다 더 깨끗하고, 더 이르고, 더 semantic(의미적)한 alarm인데, 그것이 시스템 자신이 "나는 dependency X를 포기했다"고 선언하는 것이기 때문이다. saturate되는 bulkhead는 물이 다른 곳에 닿기 전에 정확히 어떤 compartment가 물에 잠겼는지 알려준다. 그래서 pattern들은 failure를 *bounding*하는 mechanism일 뿐만 아니라 — 그것을 *보는* *instrument*(계측기)다. breaker와 bulkhead를 두기로 결정한 아키텍처는, 거의 부작용처럼, 자신의 가장 중요한 health signal이 무엇인지 결정한 것이다. 이것이 observability가 나중에 추가하는 dashboard가 아닌 이유다: 볼 가치가 있는 것들은 네가 integration point를 그리고 그것들을 어떻게 방어할지 선택했을 때 결정되었다.

> 💡 **쉬운 설명:** 보통 observability(관측 가능성)는 "나중에 대시보드 붙이는 일"로 여기지만, 실제로는 설계 결정이다. 핵심 통찰은 resilience pattern이 *관측하고 싶은 신호 자체를 만들어 낸다*는 것이다. breaker가 OPEN으로 바뀌는 순간은 "시스템이 vendor X를 포기했다"는 명확한 한 줄짜리 경보다 — timeout 에러가 수백 개 쏟아지는 것보다 훨씬 깔끔하고 빠르고 의미가 분명하다. bulkhead가 포화되면 "어느 칸이 잠겼는지"를 콕 집어 알려준다. 즉 breaker/bulkhead를 둔다는 결정이 곧 "무엇을 지켜봐야 하는가"를 정하는 결정이다.

이것이 ch-09가 완전히 발전시키는 **fitness function**(적합도 함수) 아이디어의 씨앗이다: [[richards-ford-fundamentals]]로부터, fitness function은 "an objective integrity assessment of some architectural characteristic(s)"(어떤 아키텍처 특성(들)에 대한 객관적 무결성 평가) — 보호되는 characteristic이 erode(침식)될 때 build를 fail시키는(또는 alert를 발생시키는) 자동화된 check다. "Breaker for vendor X must not stay OPEN longer than N minutes"(vendor X의 breaker는 N분보다 오래 OPEN 상태로 머무르면 안 된다)나 "p99 on the critical path < X"(critical path의 p99 < X)는 resilience characteristic을 aspiration(열망)에서 *enforcement*(강제)로 바꾼 것이다. Observability는 그런 fitness function을 애초에 measurable(측정 가능)하게 만드는 substrate(기반)다 — 볼 수 없는 characteristic은 enforce할 수 없고, instrument(계측)하기로 결정하지 않은 것은 볼 수 없다.

> 💡 **쉬운 설명:** fitness function은 "지키기로 한 특성이 무너지면 빌드를 깨거나 경보를 울리는 자동 검사"다. 예를 들어 "vendor X의 breaker가 N분 넘게 OPEN이면 안 된다", "critical path의 p99가 X 미만이어야 한다" 같은 규칙을 코드로 박아 두면, resilience가 단순한 바람이 아니라 강제 사항이 된다. 그런데 이런 검사는 값을 측정할 수 있어야 가능하다 — 그래서 observability(무엇을 계측할지 결정한 것)가 fitness function의 토대가 된다.

그리고 이 결정들 각각 — 어떤 integration point가 breaker를 얻는지, timeout budget이 무엇인지, 어떤 자원이 bulkhead되는지, 무엇이 observable해야 하는지 — 은 정확히 [[nygard-release-it]]가 ADR에 속한다고 말하는 종류의 "architecturally significant"(아키텍처적으로 중요한) 선택이다:

> "We will keep a collection of records for 'architecturally significant' decisions: those that affect the structure, non-functional characteristics, dependencies, interfaces, or construction techniques." — Nygard, "Documenting Architecture Decisions" (cognitect.com/blog/2011/11/15/documenting-architecture-decisions) — [[nygard-release-it]]

Resilience 선택은 정의상 non-functional characteristics와 dependencies에 영향을 주므로, 그것들은 정확히 ADR이 보존하기 위해 존재하는 것이다 — *왜* 이 dependency가 2초 timeout을 얻었고 저것은 30초 cooldown의 breaker를 얻었는지 설명하는 "a conversation with a future developer".

---

## 7. Applied to the Sales Agent (Lina TMR): Every Tool Call Is an Integration Point

Lina TMR은 수많은 external SaaS tool API — CRM, calendar, email, enrichment vendor, 그 외 십여 개 — 위에서 행동하는 LLM agent다. *그 tool call 하나하나가 integration point*이며, Nygard의 정확한 의미에서다: failure가 들어오고 퍼질 수 있는 지점. 이것이 이 챕터의 가장 중요한 application인데, 왜냐하면 이 pattern들 없는 agent의 failure mode가 catastrophic(파국적)이고 silent(조용)하기 때문이다.

순진한 agent loop를 그려 보라: planner가 CRM을 호출하기로 결정하고, 그다음 enrichment vendor를, 그다음 calendar를, synchronously하게, 하나의 reasoning turn(추론 턴) 안에서, timeout 없이. 이제 enrichment vendor가 안 좋은 오후를 맞아 400ms 대신 40초에 응답하기 시작한다. timeout 없이는, agent의 turn이 call당 40초 동안 *hang*한다. circuit breaker 없이는, enrichment가 필요한 다음 200개 conversation 각각이 독립적으로 그 40초를 기다린다 — agent의 worker pool이 고갈되고, 이제 enrichment를 *쓰지도 않는* conversation조차 worker를 얻을 수 없다. 느린 vendor 하나가 전체 agent fleet(함대)을 멈춰 세웠다. 이것이 §2의 cascading failure이며, agent-over-many-SaaS-tools topology는 정확히 *그토록 많은* integration point를 가지기 때문에 — 각각이 너가 uptime을 통제하지 않는 제3자가 소유한다 — 유난히 그것에 취약하다.

> 💡 **쉬운 설명:** Lina TMR 같은 LLM agent는 CRM, 캘린더, 이메일, enrichment 등 수십 개의 외부 SaaS를 부른다. 각 tool call이 곧 integration point이고, 외부 vendor의 가동률은 우리가 통제할 수 없다. timeout 없이 동기로 줄줄이 부르다가 enrichment vendor 하나가 40초로 느려지면, agent의 한 turn이 40초씩 매달리고, enrichment를 쓰는 200개 대화가 각자 40초를 기다리며 worker pool을 다 빨아먹는다 — 결국 enrichment와 무관한 대화까지 worker가 없어 멈춘다. agent는 integration point가 *워낙 많아서* 이 cascade에 특히 취약하다.

toolkit이 agent에 깔끔하게 mapping된다:

| Agent concern | Pattern | The bet, priced |
|---|---|---|
| A vendor hangs and pins the reasoning turn | **Timeout** on every tool call, budgeted from the agent's per-turn latency target | 드문 느리지만-올바른 응답을 포기; vendor 하나가 turn을 절대 얼리지 못하게 함 |
| A vendor is down for an hour | **Circuit breaker per vendor** — trip after N failures, fail fast, let the agent route around it (degrade, defer, tell the user) | cooldown 동안 false rejection을 받아들임; known-dead vendor에 매 call마다 전체 timeout budget을 쓰는 것을 멈춤 |
| One sick vendor starves all others | **Bulkhead per vendor** — separate connection/worker pools so the enrichment outage can't drain the CRM pool | partition당 일부 idle capacity를 묶음; 한 vendor의 outage 동안 healthy vendor를 완전히 사용 가능하게 유지 |
| A vendor's write succeeds but its response never returns | **Idempotent retries** + treat the response as **outside data** (immutable, possibly-stale snapshot, never authoritative live state — the ch-06 discipline from [[helland-data-outside-inside]] via [[richardson-saga]]) | eventual consistency를 받아들임; at-least-once delivery 하에서 retry를 안전하게 만듦 |
| A long multi-tool workflow shouldn't fail wholesale on one step | **Async/event steps with compensations** (saga shape) instead of one synchronous chain | saga의 lost-isolation countermeasure burden을 상속; step 간 cascade coupling을 녹임 |

아키텍처적 payoff — 최근 agent *benchmark*를 만들었고 이제 agent 자체를 설계하고 있는 이 학습자를 위한 research framing — 는 여기서 resilience가 import하는 library가 아니라 *boundary 결정*이라는 것이다. agent core와 각 vendor adapter 사이에 circuit breaker를 두는 것은 ch-03의 동일한 inward-dependency(안쪽으로 향하는 의존성) 동작이다: technology-free agent core는 변덕스러운 외부 세계에 절대 block되지 않고, 절대 그것으로부터 cascade되지 않는데, edge의 adapter가 timeout, breaker, bulkhead를 소유하기 때문이다. agent의 blast radius는 네가 그 edge를 그릴 때 결정된다 — 그리고 각 vendor가 받은 budget을 *왜* 받았는지 ADR에 기록해서, 다음 엔지니어가 화요일 오후의 enrichment outage가 sales agent 전체를 무너뜨리지 않은 유일한 이유인 breaker를 조용히 제거하지 않게 한다.

> 💡 **쉬운 설명:** 핵심 메시지는 "resilience는 import하는 라이브러리가 아니라 경계를 어디에 긋느냐의 결정"이라는 것이다. agent core(기술 중립적인 핵심 로직)와 각 vendor adapter 사이에 breaker를 두면, core는 외부 vendor에 직접 block되거나 cascade되지 않는다 — timeout/breaker/bulkhead를 모두 edge의 adapter가 떠안기 때문이다. 이건 ch-03의 "의존성은 안쪽(core)을 향한다" 원칙과 같은 동작이다. 그리고 각 vendor의 budget을 왜 그렇게 정했는지 ADR에 남겨야, 다음 사람이 "이 breaker 뭐지?"하고 떼어내 화요일 오후 장애를 전체 장애로 키우는 일을 막는다.

---

## Where This Goes

이 챕터는 runtime에서 *failure*의 blast radius를 묶었다: timeout, breaker, bulkhead, 그리고 async integration 자체가 resilience 결정이라는 인식. Ch-09는 시간에 걸친 *change*의 blast radius를 묶는다. 그것은 동일한 "keep the bet revisable" 아이디어를 한 단계 위로 가져간다: **strangler-fig** pattern은 old system 주위에 new system을 키우고 그것을 조각조각 은퇴시킴으로써 migration을 reversible하게 만들고("investment and returns occur gradually and visibly", per-step 위험이 낮음 — [[martin-strangler-fig]]), **fitness function**은 네가 여기서 보호하기로 결정한 architecture characteristics — dependency rule, latency budget, breaker behavior — 를 aspiration에서, 그것들이 썩을 때 build를 fail시키는 자동화된 check로 바꾼다. Resilience는 *outage*가 one-way door가 되지 않게 한다; evolution은 *architecture*가 그렇게 되지 않게 한다.
