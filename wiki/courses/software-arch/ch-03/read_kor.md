<!-- chapter: ch-03
     track: boundaries
     kind: content
     title: Structure Inside a Boundary: Hexagonal, Clean, and Aggregates
     deps: [[ch-02]]
     sources: [[martin-clean-arch]], [[cockburn-hexagonal]], [[ddd-aggregates-tactical]], [[richards-ford-fundamentals]], [[insights]], [[COLLECTION-PLAN]]
-->

# 03장 — Structure Inside a Boundary: Hexagonal, Clean, and Aggregates

> **핵심 통찰.** [[ch-02]]가 bounded context(경계 지어진 맥락)를 그리고 나면, 질문은 그 *내부*에 코드를 어떻게 배치할 것인가로 바뀐다 — 뒤집기에 비싼 부분, 즉 domain policy(도메인 정책)가 그 밑에서 vendor, framework, database가 바뀔 때마다 썩어버리지 않도록. 세 개의 이름 붙은 architecture(Hexagonal, Clean, Onion)는 동일한 한 문장짜리 답을 준다: 중심에는 technology-free core(기술 비의존 핵심)를 두고, 모든 I/O는 가장자리로 밀어내며, **source-code dependency(소스 코드 의존성)는 오직 안쪽만 가리키게 하라**. 그다음 tactical DDD(전술적 DDD)는 그 core를 어떻게 형태 지을지 알려준다: **aggregate(애그리거트)**는 즉각적인 transactional consistency(트랜잭션 일관성)의 단위이며, 그것 주위에 그리는 boundary 그 자체가 뒤집기에 비싼 결정이다 — 너무 크게 잡으면 architecture에 contention point(경합 지점)를 구워 넣게 되고; 제대로 잡으면 나중에 필요할 eventual-consistency(최종 일관성) seam(이음매), 즉 saga(사가)가 이미 그 안에 축소판으로 존재한다.

> 💡 **쉬운 설명:** 이 챕터의 큰 그림은 "비싼 것은 안쪽에, 싼 것은 바깥쪽에"다. 여기서 "비싸다"는 돈이 아니라 *나중에 바꾸기 어렵다*는 뜻이다. domain policy(예: 리드를 어떻게 자격 판정하는가)는 한번 정하면 바꾸기 어렵고, vendor(예: 어떤 LLM API를 쓰는가)는 자주 갈아치운다. 그래서 비싼 것을 중심에 두고 싼 것이 밖에서 바뀌어도 중심이 흔들리지 않게 만든다는 발상이다.

> **가이드라인.** interface(인터페이스), 즉 port(포트)를 core 안에서 core 자신의 언어로 정의하라; 그것을 바깥에서 adapter(어댑터)로 구현하라; 모든 boundary는 ORM row나 framework object가 아니라 plain data structure(순수 데이터 구조)로 건너라. core 안에서는 진짜 invariant(불변식)를 작은 aggregate로 모델링하고, aggregate당 한 transaction을 쓰고, 다른 aggregate는 identity(식별자)로 참조하며, domain event가 변경을 boundary 너머로 나르게 하라. indirection 세금(port, DTO, mapper)은 core가 그 vendor들보다 오래 살 만큼 충분히 장수할 때에만 지불하라 — 그것이 바로 학습자의 sales agent가 처한 상황이다.

---

## 1. The Dependency Rule Is the Whole Architecture in One Sentence

Robert Martin의 "Clean Architecture"는 한 벽 가득한 다이어그램을 단 하나의 강제 가능한 규칙으로 환원하며, excerpt는 그것을 그대로 기록한다 ([[martin-clean-arch]]):

> "Source code dependencies can only point inwards." — Robert C. Martin

그 문장이 load-bearing(하중을 떠받치는) 주장 전부다. 나머지 모든 것 — 동심원, port 이름, DTO 규율 — 은 그것을 위한 mechanism(메커니즘)이다. 이 규칙이 의미하는 바: 안쪽 원에 있는 어떤 것도 바깥쪽 원에 있는 어떤 것을 *지명(name)*해서는 안 된다. detail(세부 구현; framework, DB driver, LLM SDK)은 policy에 의존해도 되지만; policy는 **절대로** detail에 의존해서는 안 된다.

> 💡 **쉬운 설명:** "name(지명)한다"는 코드에서 `import` 한다거나 그 타입 이름을 직접 적는다는 뜻이다. 안쪽 코드 파일이 바깥쪽 라이브러리 이름을 한 번이라도 써버리면 그 순간 의존성 화살표가 바깥을 향하게 되어 규칙이 깨진다. 규칙의 핵심은 딱 이 "이름을 쓰느냐 마느냐"다.

동심원을, 안쪽에서 바깥쪽으로, excerpt가 배치하는 대로:

| Ring | Name | What lives here | Change rate |
|------|------|-----------------|-------------|
| 1 (center) | **Entities** | 전사적(enterprise-wide) 비즈니스 규칙 | "least likely to change"(가장 바뀔 가능성 낮음) |
| 2 | **Use Cases** | application-specific 규칙; entity로/로부터 데이터를 orchestrate | 느림 |
| 3 | **Interface Adapters** | controller, presenter, gateway — use-case 형식과 외부 형식 간 변환 | 중간 |
| 4 (rim) | **Frameworks & Drivers** | web framework, database, device — "tools rather than constraints"(제약이 아니라 도구) | 빠름 |

이 순서는 미적인 것이 아니다. 그것은 [[ch-01]]의 척추(spine)를 파일 배치에 적용한 것이다: **바꾸기 가장 어려운 것이 가장 깊은 곳에 앉아, 바꾸기 가장 쉬운 것으로부터 격리된다.** 이 규칙은 빠르게 움직이는 rim(테두리)의 churn(교체; Postgres를 DynamoDB로 바꾸기, 한 모델 vendor를 다른 것으로 바꾸기)이 느리게 움직이는 중심에 대한 수정을 강제할 수 없음을 기계적으로 보장한다.

> 💡 **쉬운 설명:** "spine"은 이 코스 전체를 관통하는 한 가지 원칙, 즉 "결정을 그것의 뒤집기 비용에 따라 다뤄라"를 가리킨다. change rate(변경 빈도)가 높은 것(rim)일수록 바깥에, 낮은 것(center)일수록 안에 둔다 — 자주 바뀌는 것이 자주 안 바뀌는 것을 건드리지 못하게 하기 위해서다.

### 1.1 The independence claims, and why they follow

Martin은 이 architecture가 사주는 세 가지를 말한다. excerpt가 그것들을 인용한다 ([[martin-clean-arch]]):

- **"Independent of Frameworks"** — "the framework is a tool you call, not a base class you inherit your whole app from."(framework는 당신이 호출하는 도구이지, 앱 전체가 상속받는 base class가 아니다.)
- **"Testable"** — "business rules can be tested without UI, database, or external elements."(비즈니스 규칙은 UI, database, 외부 요소 없이 테스트될 수 있다.)
- **"Independent of UI, Database, [and] any external agency"** — 각각이 core logic을 건드리지 않고 교체 가능하다.

이것들은 세 개의 별개 기능이 아니다; 그것들은 그 하나의 규칙의 세 가지 귀결(consequence)이다. 어떤 안쪽 코드도 바깥쪽 코드를 지명하지 않는다면, 구성상(by construction) 안쪽 코드는 바깥쪽 코드가 없거나 가짜(faked)인 상태로 compile되고, 실행되고, 테스트될 수 있다. Testability는 나중에 덧붙이는 add-on이 아니다 — 그것은 "dependency가 안쪽을 가리킨다"는 명제 그 자체를, test harness(테스트 하니스)의 관점에서 관찰한 것이다.

> 💡 **쉬운 설명:** 즉 테스트하기 쉽다는 건 별도로 노력해서 얻는 보너스가 아니라, 의존성을 안쪽으로만 향하게 만들면 *자동으로 따라오는* 결과다. 바깥(DB, UI)을 가짜로 갈아끼워도 안쪽 코드가 그대로 돌아가니, 그게 곧 테스트가 쉬운 상태다.

### 1.2 How the type system tries to break the rule (and the fix)

Dependency Rule(의존성 규칙)은 말하기 쉽고 실수로 어기기도 쉬운데, 위반이 return type을 타고 들어오기 때문이다. excerpt의 두 번째 인용이 그 방어막(guard)이다 ([[martin-clean-arch]]):

> "The important thing is that isolated, simple, data structures are passed across boundaries." — Martin

use case가 repository를 호출하고 repository가 ORM row를 돌려주면, 그 use case는 이제 ORM에 transitively(타동적으로) 의존하게 된다 — 안쪽을 향하던 화살표가 type을 통해 뒤집힌 것이다. inbound controller가 자신의 framework `Request` object를 use case에 그대로 넘겨도 같은 일이 벌어진다. 해결책은 기계적이다: **모든 boundary를 plain DTO로 건너라.** adapter에서 ORM row를 domain object로 매핑하라; controller에서 framework request를 command로 매핑하라. DTO는 바깥쪽 의존성이 안쪽으로 새어 들어오는 것을 막는 airlock(에어록)이다.

> 💡 **쉬운 설명:** DTO(Data Transfer Object)는 그냥 필드 몇 개를 담은 단순한 데이터 묶음이다. 핵심은 "원본 객체(ORM row, 프레임워크의 Request)를 안쪽으로 그대로 통과시키지 말고, 경계에서 우리만의 단순 데이터로 갈아 끼우라"는 것. 비유하자면 우주선 에어록처럼, 바깥 공기(외부 라이브러리 타입)가 안쪽으로 직접 들어오지 못하게 차단하는 칸막이다.

이것이 이 패턴의 첫 번째 구체적 대가이며, 지금 이름 붙여둘 가치가 있다: 당신은 mapper를 작성하게 된다. 세 필드짜리 CRUD endpoint에는 그 mapper가 순전한 의례(ceremony)다. 하지만 agent의 domain core에는, 그것이 아직 만나보지 못한 vendor를 core가 견뎌내게 해주는 seam이다.

### 1.3 The four rings, mapped onto Lina TMR

§1의 ring 표는 추상적이다; 규칙이 실효성을 갖도록 그것을 학습자의 시스템에 고정해보자. architecture가 올바르다는 진단(diagnostic)은, 표를 채울 수 있고 *동시에* 모든 dependency가 여전히 안쪽을 가리킨다는 것이다:

| Ring | Lina TMR component | May depend on | May NOT depend on |
|------|--------------------|---------------|-------------------|
| Entities | `Lead`, `Deal`, qualification 규칙, escalation policy | core 밖의 어떤 것도 의존 안 함 | LLM SDK, Salesforce 타입, web framework |
| Use cases | "qualify lead," "advance stage," "route notification" | entity + port | 구체 adapter나 vendor 타입 |
| Interface adapters | LLM adapter, Salesforce repo, webhook controller | use-case port + vendor SDK | 해당 없음 (여기가 SDK가 *허용되는* 곳) |
| Frameworks & drivers | Anthropic SDK, Postgres driver, HTTP server | 자기 자신 | core 안의 그 무엇도 |

올바름의 테스트는 grep이다: `entities/`나 `use_cases/` 아래의 어떤 파일이라도 `anthropic`, `salesforce`, 또는 web framework를 import한다면, 규칙은 깨진 것이고 "may NOT depend on" 열이 위반된 것이다. Ch-09의 fitness function(적합성 함수)은 바로 그 grep을 자동화된, 빌드를 실패시키는 검사로 바꾼다 — 그래서 규칙이 커밋들 사이에서 조용히 침식되지 않는다.

> 💡 **쉬운 설명:** grep으로 "안쪽 폴더가 바깥 라이브러리 이름을 import하는지" 한 줄 검색해보면 규칙 위반을 바로 잡아낼 수 있다는 뜻이다. fitness function은 이 검사를 CI에 넣어, 누군가 실수로 규칙을 어기면 빌드가 깨지게 만드는 자동 장치다. 규칙을 "사람의 기억"이 아니라 "기계의 강제"로 지키는 것.

---

## 2. Hexagonal: The Same Rule, Told from the Test Harness

Cockburn의 Hexagonal(Ports & Adapters) architecture는 Martin의 framing보다 앞서며, 동일한 규칙을 다른 각도에서 동기 부여한다: "policy는 detail에 의존하면 안 된다"가 아니라, "나는 화면과 database를 제거한 채로 내 application을 실행하고 싶다"는 각도다. excerpt는 모두가 인용하는 그 한 줄을 기록한다 — 표준 `.us` URL이 2026-06-15 fetch 시점에 만료된 TLS 인증서를 가지고 있었기 때문에 alistaircockburn.com 미러를 통해 교차 확인했다 ([[cockburn-hexagonal]], gap은 [[COLLECTION-PLAN]]에 기록됨):

> "Allow an application to equally be driven by users, programs, automated test or batch scripts, and to be developed and tested in isolation from its eventual run-time devices and databases." — Alistair Cockburn (via the .com mirror; the .us original is cert-blocked, see [[COLLECTION-PLAN]])

그것이 고치는 동기를, excerpt가 framing하는 대로: 통상적인 layering(계층화)은 business logic을 위로는 UI, 아래로는 database 양쪽과 조용히 얽어버려서, 화면과 DB 없이는 core를 테스트할 수 없고, 둘 중 어느 쪽도 수술 없이는 교체할 수 없다. Hexagonal은 *양쪽* 모두를 동등한 "바깥(outside)"으로 취급함으로써 그 비대칭(asymmetry)을 죽이며, 각각은 **port**를 통해 도달된다.

> 💡 **쉬운 설명:** 전통적 계층 구조에서는 UI는 "위", DB는 "아래"로 다르게 취급한다. Hexagonal의 통찰은 "둘 다 그냥 바깥세상일 뿐 위아래가 없다"는 것이다. 그래서 UI도 DB도 똑같이 port라는 동일한 종류의 통로로 core에 연결한다 — 이 대칭성이 핵심이다.

### 2.1 Ports and adapters, defined precisely

excerpt의 정의들 ([[cockburn-hexagonal]]):

- **port**는 application이 정의한, application 자신의 언어로 된 interface다 — 예: `ForPlacingOrders`, `ForStoringOrders`. excerpt의 그대로의 표현: "All input and output reaches/leaves the application through a port that isolates the application from external tools, technologies and delivery mechanisms."(모든 입력과 출력은 application을 외부 도구·기술·전달 메커니즘으로부터 격리하는 port를 통해 application에 도달하거나 떠난다.)
- **adapter**는 port와 구체 기술 사이를 번역한다: inbound 쪽에서는 REST controller, CLI, test driver; outbound 쪽에서는 Postgres repo, in-memory fake, message-bus client.

결정적 속성은 *누가 interface를 소유하느냐*다. application이 port를 소유한다. adapter는 그것을 따른다(obey). 그것이 한 수로 이루어지는 dependency inversion(의존성 역전)이다: outbound adapter(database)가 core가 선언한 interface에 의존하므로, runtime에는 데이터가 바깥으로 흐를지언정 화살표는 안쪽을 가리킨다.

그 마지막 절(clause)에서 대부분의 사람이 혼란스러워하므로, 속도를 늦출 가치가 있다. 어떤 architecture에든 두 개의 서로 다른 화살표가 있고, driven(피동) 쪽에서는 둘이 반대 방향을 가리킨다:

- **Runtime control flow** — *실행 시점에 누가 누구를 호출하는가.* driven port에서는 core가 바깥으로 호출한다: use case가 `repository.save(lead)`를 invoke한다. 흐름은 바깥을 가리킨다.
- **Source-code dependency** — *compile 시점에 어느 파일이 어느 타입을 지명하는가.* repository interface는 core 안에 선언되고; 바깥 ring의 Postgres class가 그것을 구현한다. 그래서 바깥 ring이 안쪽 ring의 타입을 지명한다. dependency는 안쪽을 가리킨다.

Dependency Rule은 *두 번째* 화살표를 다스리며, 첫 번째가 아니다. 당신은 종일 바깥으로 *호출(call)*해도 된다; 바깥을 *지명(name)*하는 것만 허용되지 않는다. dependency inversion은 정확히, runtime flow가 한 방향으로 가는 동안 compile-time dependency가 반대 방향으로 가게 하는 trick이다. 혹시라도 규칙이 불가능하게 느껴진다면 ("하지만 core가 database를 써야 하잖아!"), 당신은 두 화살표를 혼동한 것이다 — core는 *port*를 쓰고, database가 그것을 구현하며, SDK import는 오직 adapter 파일에만 산다.

> 💡 **쉬운 설명:** 가장 헷갈리는 지점이라 풀어 쓴다. "호출 흐름"과 "코드 의존성"은 별개의 화살표다. core가 `repository.save()`를 부를 때 *실행*은 바깥으로 나가지만(호출 흐름), 그 `repository`라는 인터페이스 이름은 core가 정의한 것이고 바깥의 Postgres 클래스가 그걸 가져다 구현한다(코드 의존성은 안쪽). 즉 "바깥에 일을 시키되, 바깥의 구체적 이름은 core가 모른다"가 가능해진다. 이게 dependency inversion이다.

### 2.2 Primary vs secondary (driving vs driven)

Cockburn은 의도적으로 "pretending that all ports are fundamentally similar"(모든 port가 근본적으로 비슷한 척하면서)라고 쓰지만, 실제로는 port가 두 가지 맛(flavor)으로 오며, 그 구분은 무엇을 어디에 둘지에 중요하다 ([[cockburn-hexagonal]]):

| | **Primary / driving** | **Secondary / driven** |
|---|---|---|
| Who initiates? | adapter가 core **안으로** 호출한다 | core가 adapter를 통해 **바깥으로** 호출한다 |
| Examples | UI, HTTP client, **test suite**, batch script | database, message broker, email gateway, **the LLM API** |
| Who owns the interface? | core (port는 core의 API다) | core (port는 core의 요구사항이다) |
| Direction of dependency | inward | inward (dependency inversion을 통해) |

이 대칭(symmetry)이 핵심 전부이며, excerpt가 그것을 말한다: "the application can be equally driven by an automated, system-level regression test suite, by a human user, by a remote http application, or by another local application"(application은 자동화된 시스템 수준 regression test suite에 의해서도, 사람 사용자에 의해서도, 원격 http application에 의해서도, 또는 다른 로컬 application에 의해서도 동등하게 구동될 수 있다), 그리고 데이터 쪽에서는 "configured to run decoupled from external databases using an in-memory… database replacement"(in-memory… database 대체물을 사용하여 외부 database로부터 분리되어 실행되도록 구성될 수 있다)일 수 있다.

> 💡 **쉬운 설명:** primary(driving)는 "core를 부르는 쪽" — 사용자, HTTP 요청, 그리고 *테스트 코드*도 여기에 속한다. secondary(driven)는 "core가 부려먹는 쪽" — DB, 이메일, 그리고 *LLM API*도 여기 속한다. 둘을 구분하면, 예컨대 테스트 suite도 그냥 또 하나의 primary adapter이므로 진짜 사용자 없이 core를 마음껏 구동해볼 수 있다는 점이 분명해진다.

**지금 figure를 사용하라.** [`figures/hexagon-dependency-rule.html`](figures/hexagon-dependency-rule.html)를 열고 각 adapter를 클릭해 그것이 core를 건드리지 않고 교체될 수 있음을 확인하라; 그다음 "show a forbidden outward dependency"를 토글하여 Dependency Rule이 금지하는 그 한 화살표와, 그것을 그리는 순간 무엇이 깨지는지를 보라. LLM-API adapter가 core를 그대로 둔 채 교체되는 것을 지켜보는 것이, 왜 이 패턴이 장수하는 agent에 대한 올바른 베팅인지를 느끼는 가장 빠른 길이다.

### 2.3 Why a hexagon? (a myth worth retiring early)

그 모양은 육(六)이라는 의미를 전혀 지니지 않는다. excerpt는 명시적이다 ([[cockburn-hexagonal]]): Cockburn은 통상적인 layered rectangle 대신 polygon(다각형)을 선택했는데, 순전히 core 주위에 여러 port와 그 adapter를 그릴 시각적 공간을 남기기 위해서였다. **변의 수는 부수적(incidental)이다.** 누군가 "the six concerns of hexagonal architecture"(hexagonal architecture의 여섯 가지 관심사)를 두고 논쟁하는 것을 듣는다면, 그들은 그림상의 편의에 의미를 역으로 끼워 넣은(reverse-engineered) 것이다. 내용은 "core + ports + adapters + inward dependencies"이지, "six"가 아니다.

> 💡 **쉬운 설명:** "왜 육각형이냐"에는 깊은 뜻이 없다. 그냥 사방에 통로(port)를 여러 개 그릴 자리가 필요해서 사각형 대신 다각형을 골랐을 뿐이다. "육각형이니 여섯 가지 의미가 있다"는 식의 해석은 그림을 보고 거꾸로 지어낸 것이니 무시하면 된다.

---

## 3. Same Idea, Three Names — and the One Myth This Chapter Kills

이것이 ch-03의 doc-vs-reality(문서 대 현실) 조정이다. 대중적 framing은 Hexagonal, Clean, Onion을 당신이 그 사이에서 골라야 하는 세 개의 경쟁하는 architecture로 취급한다 — blog 전쟁과 "which is better" thread를 낳는 선택이다. 1차 출처(primary source)들은 이를 단호히 부정한다.

> **Myth:** "Hexagonal vs Clean vs Onion is an architecture decision you have to make."(Hexagonal 대 Clean 대 Onion은 당신이 내려야 하는 architecture 결정이다.)
> **Reality:** They are the same idea with different diagrams: a technology-free core, dependencies pointing inward, I/O at the edges via interfaces the core owns.(그것들은 다른 다이어그램을 가진 동일한 발상이다: technology-free core, 안쪽을 가리키는 dependency, core가 소유한 interface를 통한 가장자리에서의 I/O.)

두 excerpt 모두 이것을 직접 주장하며, 내 paraphrase가 아니다. [[martin-clean-arch]]는 말한다: "Clean = [[cockburn-hexagonal]] (Ports & Adapters) = Onion. All three: technology-free core, inward-pointing dependencies, I/O at the edges through inverted interfaces." 그리고 [[cockburn-hexagonal]]는 말한다: "Hexagonal, Onion, and Clean are the same idea with different diagrams."

둘은 오직 *무엇을 강조하느냐*에서만 다르며, 그 차이는 진정으로 유용하다:

| | Emphasizes | Best for explaining |
|---|---|---|
| **Clean** ([[martin-clean-arch]]) | **규칙(rule)**, 깔끔하게 진술됨: "source code dependencies can only point inwards" | structure가 *왜* 올바른가; 무엇이 금지되는가 |
| **Hexagonal** ([[cockburn-hexagonal]]) | **testability 동기**, 생생하게: DB와 UI를 제거한 채 core를 구동하라 | 규칙을 따르면 *무엇을 얻는가*; 어떻게 테스트하는가 |

그래서 해결책은 "하나를 골라라"가 아니다. 그것은: 당신이 강제하고 있는 제약을 진술할 때는 Clean 어휘를 쓰고, "왜 인터페이스가 이렇게 많아?"라고 묻는 팀원에게 그것을 정당화할 때는 Hexagonal 어휘를 쓰라는 것이다. 그 답은 "그래야 test suite가 database 없이 이것을 구동할 수 있으니까"다. 세 번째 이름 뒤에 숨어 있는 세 번째 architecture는 없다.

이것이 [[ch-01]] spine에 중요한 이유는, 동일한 것들 사이에서 고르는 것은 유일하게 중요한 예산 — 뒤집기에 비싼 결정들 — 의 순전한 낭비이기 때문이다. 그 예산은 diagram brand가 아니라 boundary와 contract에 써라.

> 💡 **쉬운 설명:** "Clean이냐 Hexagonal이냐 Onion이냐"를 두고 고민하는 건 시간 낭비다 — 셋은 같은 것을 다른 그림으로 그렸을 뿐이기 때문이다. 진짜로 고민할 가치가 있는(=뒤집기 비싼) 결정은 어디에 경계를 긋고 어떤 계약(contract)을 둘지이지, 다이어그램의 이름표가 아니다.

---

## 4. Tactical DDD Inside the Core: The Aggregate Is the Consistency Unit

Hexagonal/Clean은 core의 *모양*(technology-free, 가장자리의 port)은 알려주지만, core *안에서* domain object를 어떻게 조직할지는 아무것도 말하지 않는다. 그것이 tactical DDD의 일이며, 이 챕터에 의도적으로 병합되었다: [[cockburn-hexagonal]]와 [[martin-clean-arch]]가 당신에게 벽을 주고; [[ddd-aggregates-tactical]]가 방들을 준다.

> 💡 **쉬운 설명:** 비유가 깔끔하다 — Hexagonal/Clean은 집의 *바깥 벽*(core 전체의 윤곽과 출입구)을 세워주고, tactical DDD는 그 안을 *방으로 나누는* 일을 한다. 즉 core 내부를 entity, value object, aggregate 같은 단위로 어떻게 칸막이할지를 다룬다.

인용 전에 주의 하나: tactical-DDD 출처는 **책의 논지를 추출한 것이지, 그대로 인용한 것이 아니다(book-thesis-extracted, not verbatim).** excerpt header가 명시적이다 — Evans의 *Domain-Driven Design*(2003)과 Vernon의 *Implementing DDD* / *DDD Distilled*는 "books, theses extracted, not quoted verbatim"(책이며, 논지가 추출되었고, 그대로 인용되지 않음)이다 ([[ddd-aggregates-tactical]]). 그래서 이 절의 모든 것은 저자들의 입장에 대한 출처가 명기된 paraphrase(의역)이며, 마치 fetch된 한 줄인 것처럼 따옴표 안에 제시되지 않는다.

### 4.1 The vocabulary

[[ddd-aggregates-tactical]]가 네 가지 building block을 기록하는 대로:

- **Entity** — attribute가 바뀌어도 identity(정체성)가 시간에 걸쳐 지속된다. `Order`는 그 위의 모든 필드가 바뀐 후에도 내일도 같은 order다.
- **Value object** — identity가 없다; 그 attribute들에 의해 온전히 정의되며, immutable(불변)이다. `Money(amount, currency)` — 같은 필드를 가진 두 instance는 서로 교환 가능하다.
- **Aggregate** — 데이터 변경을 위해 하나의 단위로 취급되는 entity와 value object의 cluster(군집)이며, 유일한 외부 진입점인 단일 **aggregate root(애그리거트 루트)**가 그 앞을 막는다.
- **Domain event** — domain에서 의미 있는 무언가가 일어났다는 기록(`OrderPlaced`)이며, aggregate 및 context boundary를 가로질러 변경을 전파하는 데 쓰인다.

> 💡 **쉬운 설명:** entity와 value object의 차이는 "이름표가 필요한가"로 기억하면 쉽다. entity는 사람처럼 *같은 한 명*이라는 정체성이 있어서 속성이 바뀌어도 동일 인물이다(이름표/ID 필요). value object는 지폐의 액수처럼 *값 자체*가 전부라서, 같은 값이면 어느 것이든 똑같이 취급한다(이름표 불필요). aggregate root는 그 군집의 "정문 단 하나"로, 외부는 반드시 이 문으로만 드나든다.

entity/value-object 구분은 현학(pedantry)이 아니다; 그것은 당신이 어떻게 저장하고 비교하는가를 바꾼다. entity는 identity column과 "시간에 걸쳐 같은 것"이라는 개념이 필요하다; value object는 둘 다 필요 없으며 inline으로 저장되고 mutate(변형)되기보다 통째로 교체되는 것이 가장 안전하다. Lina TMR의 domain에서, `Lead`는 entity다(같은 lead가 stage 변경에 걸쳐 지속된다), 반면 SaaS sheet에서 끌어오는 `FxRate`나 `AccountTier`는 value object다 — 그 필드들에 의해 온전히 정의되고, immutable하며, 편집되기보다 교체된다. 이것을 틀리면(value object를 entity로 취급하면) 새 복사본이었어야 할 것이 mutable shared state(가변 공유 상태)가 되어버린다.

aggregate가 load-bearing 개념이다. excerpt의 Core Insight는 그 논지를 진술한다(book-extraction 단서에 따라 paraphrase): aggregate는 *transactional consistency*의 단위이며; 그 boundary가 당신이 무엇을 원자적으로(atomically) 바꿀 수 있는지를 — 그리고 따라서 어디서 단위들 사이의 eventual consistency로 후퇴(fall back)해야 하는지를 — 결정한다.

> 💡 **쉬운 설명:** "한 transaction = 한 aggregate"라는 게 핵심이다. aggregate 안의 것들은 한 번에 다 같이 바뀌어야 하는(원자적) 묶음이고, aggregate 바깥의 것들은 "조금 늦게 맞춰져도 되는"(eventually consistent) 묶음이다. 즉 aggregate 경계 = "어디까지 한꺼번에 보장할 것인가"의 선이다.

### 4.2 Vernon's four aggregate rules

이 네 규칙이 실행 가능한 core이며, Vernon에게 귀속되고 [[ddd-aggregates-tactical]]에 "load-bearing claim"으로 기록되어 있다. Vernon의 입장으로 제시됨(paraphrase, verbatim 아님):

| # | Rule | What it buys you |
|---|------|------------------|
| 1 | **Model true invariants in consistency boundaries.**(진짜 invariant를 consistency boundary 안에 모델링하라.) aggregate = *함께, 즉시* 일관되어야 하는 object들의 집합. | "원자적"이라는 것의 명료한 정의: 안에 있는 모든 것은 한 transaction; 밖에 있는 모든 것은 eventually consistent해도 된다. |
| 2 | **Design small aggregates.**(작은 aggregate를 설계하라.) 적은 큰 것보다 많은 작은 것을 선호하라. | 큰 aggregate는 contention을 만들고 load를 느리게 한다; 작은 것은 lock scope와 concurrency 충돌을 줄인다. |
| 3 | **Reference other aggregates by identity.**(다른 aggregate는 identity로 참조하라.) `Order` object가 아니라 `OrderId`를 쥐어라. | boundary를 명료하게 유지한다; 큰 object graph가 우발적으로 함께 load되고 mutate되는 것을 방지한다. |
| 4 | **Use eventual consistency outside the boundary.**(boundary 밖에서는 eventual consistency를 써라.) 한 transaction = 한 aggregate; 다른 aggregate가 반응해야 할 때는, domain event를 publish하고 별도 transaction에서 update하게 하라. | 이것이 consistency phase의 모든 것으로 이어지는 seam이다. |

### 4.3 Rule 4 is the in-process seed of the distributed saga

이것이 이 챕터에서 가장 중요한 연결이며, 이 병합이 이루어진 이유다. excerpt가 그것을 직접 진술한다 ([[ddd-aggregates-tactical]]): 규칙 4는 "the in-process seed of the distributed [[richardson-saga]]: a saga is 'one transaction per aggregate, coordinated by events' stretched across services."(분산 [[richardson-saga]]의 in-process(프로세스 내) 씨앗이다: saga란 '한 aggregate당 한 transaction, event로 조율됨'을 service들에 걸쳐 늘려놓은 것이다.)

그것을 주의 깊게 읽어라. saga(ch-06)는 보통 *distributed-systems* 패턴으로 소개된다 — 여러 service를 갖게 되어 더 이상 그것들에 걸쳐 transaction을 쥘 수 없게 되었을 때 손을 뻗는 것. DDD의 주장은 동일한 모양이 단일 process *안에서도*, 지금 당장, 한 transaction에서 update될 수 없는 두 aggregate를 갖는 순간 존재한다는 것이다. "aggregate당 한 개의 local transaction, domain event로 접합되고, 무언가 downstream에서 실패할 때 compensating logic(보상 로직)이 있음"은, aggregate가 같은 module에 살든 network를 가로질러 살든 saga다.

> 💡 **쉬운 설명:** 이 챕터의 가장 중요한 한 방이다. saga는 보통 "마이크로서비스가 여러 개라 DB 트랜잭션을 하나로 못 묶을 때 쓰는 분산 패턴"으로 배운다. 그런데 사실 *같은 프로세스 안에서도* aggregate가 둘이면 이미 같은 구조가 생긴다 — 한 aggregate를 커밋하고, 이벤트를 쏘고, 다른 aggregate가 별도 트랜잭션으로 반응하고, 실패하면 되돌린다(compensate). 네트워크가 끼느냐 마느냐만 다를 뿐, 구조는 똑같다.

씨앗을 구체화하자. 두 aggregate, `Lead`와 `Booking`, 규칙 4가 transaction을 공유할 수 없다고 말하는 것들이다. in-process 버전은 이렇게 읽힌다:

```
# one local transaction: mutate exactly one aggregate
def qualify_lead(lead_id):
    lead = leads.load(lead_id)          # load one aggregate
    event = lead.qualify()              # mutate it; it emits LeadQualified
    leads.save(lead)                    # commit transaction #1
    publish(event)                      # hand off across the boundary

# a separate handler, a separate transaction, reacts to the event
def on_lead_qualified(event):
    booking = scheduling.create_for(event.lead_id)   # transaction #2
    bookings.save(booking)
    # if THIS fails, compensate: emit a LeadQualificationReverted event
```

여기 어디에도 network는 언급되지 않는다. 그런데도 그 구조 — local transaction, domain event, 별도로 commit되는 reaction, 실패 시 compensation — 가 *바로* saga다. service 분리가 바꾸는 유일한 것은 `publish(event)`가 선(wire)을 가로지르고 `on_lead_qualified`가 다른 process에서 실행된다는 점이다. consistency 추론(무엇이 atomic이고, 무엇이 eventual이며, 무엇이 compensate하는가)은 동일하며, 여기, 바꾸기 싼(cheap-to-change) in-process 코드에서 결정되었다.

그 보상(payoff)은 구체적이고 spine과 맞아떨어진다: modular monolith(모듈형 모놀리스) 안에서 올바른 aggregate boundary를 내재화하면, 나중의 service 분리가 올바른 consistency boundary를 **공짜로** 물려받는다. 뒤집기에 비싼 작업(무엇이 함께 atomic해야 하는지를 결정하는 것)이 한 번, design time에, 바꾸기 싼 in-process 코드에서 행해진다 — [[richards-ford-fundamentals]] microservice 프리미엄을 지불하기 훨씬 전에. aggregate를 틀리면 — 너무 크게 — architecture에 뒤집기 비싼 contention point를 구워 넣은 것이며, 바로 [[ch-01]]이 경고한 deep binding decision(깊이 묶인 결정)의 종류다. 이것이 또한 ch-04가 aggregate boundary를 *후보(candidate)* deployment seam으로 취급하는 이유다: 이미 올바른 in-process consistency 단위인 boundary만이 선(wire)을 가로질러 승격(promote)시키기 안전한 유일한 종류다.

> 💡 **쉬운 설명:** 실용적 이득을 한 줄로: "지금 한 프로세스 안에서 aggregate 경계를 제대로 그려두면, 나중에 마이크로서비스로 쪼갤 때 일관성 경계가 이미 맞아 있어 거저 얻는다." 반대로 aggregate를 너무 크게 잡으면 경합 지점이 박혀버려 나중에 떼어내기가 비싸진다. 그래서 "바꾸기 싼 지금" 제대로 정해두는 것이 남는 장사다.

---

## 5. Pricing the Bet: Indirection Is Never Free, and Sometimes It's the Only Right Move

trade-off spine은 모든 패턴이 베팅으로 가격 매겨질 것을 요구한다 — 무엇을 바꾸기 싸게 유지하고, 무엇을 비싸게 만드는가. Hexagonal/Clean + aggregate도 예외가 아니며, excerpt는 그 단점에 대해 유난히 솔직하다 ([[martin-clean-arch]]):

> The cost is indirection and boilerplate (DTOs, ports, mappers). For a small CRUD service it's overkill.(비용은 indirection과 boilerplate(DTO, port, mapper)다. 작은 CRUD service에는 과잉이다.)

베팅을 두 열 모두, 완전히 명시적으로 만들어보자.

### 5.1 What it keeps cheap to change

| Change you might face | Why it stays cheap |
|---|---|
| database 교체 (Postgres → DynamoDB) | 기존 `ForStoringState` port를 구현하는 새 secondary adapter; core는 손대지 않음. |
| LLM vendor 교체 / 로컬 모델 추가 | `ForGeneratingReplies` 뒤의 새 secondary adapter; domain policy는 결코 SDK를 import하지 않았음. |
| 시스템을 구동하는 새 방법 추가 (CLI, webhook, cron) | 기존 inbound port를 호출하는 새 primary adapter; core 변경 없음. |
| 인프라 없이 domain policy 테스트 | test suite *가* primary adapter다; in-memory fake가 driven adapter를 대신함. |

### 5.2 What it makes expensive

| Cost you take on | When it bites |
|---|---|
| 모든 boundary에서의 mapper와 DTO | 즉시, 모든 endpoint마다 — 사소한 CRUD에서는 순전한 의례. |
| Indirection: 호출이 이제 port와 adapter를 거쳐 hop함 | 코드를 읽으려면 interface를 그 구현까지 따라가야 함. |
| 정직하게 유지하는 규율 | 새어 들어온 ORM row나 framework object 하나가 조용히 안쪽 화살표를 뒤집는다; 강제(enforcement) 없이는(fitness function — ch-09), 규칙이 침식된다. |
| aggregate를 위한 선행(up-front) 모델링 비용 | 잘못된 consistency boundary를 그리는 것 자체가 뒤집기에 비싸다. |

### 5.3 The decision criterion

결정적 질문은 "이것이 clean한가?"가 아니라 "**이 core가 그 vendor들에 비해 얼마나 장수하는가(long-lived)?**"다. excerpt의 평결 ([[martin-clean-arch]]):

> For the learner's long-lived sales-agent core — where business policy must outlive whichever LLM API/vector DB/web framework is current — it's exactly the right bet: it keeps the *expensive-to-reverse* part (domain policy) insulated from the *cheap-to-swap* parts (vendors).(학습자의 장수하는 sales-agent core — business policy가 현재 어떤 LLM API/vector DB/web framework이든 그것보다 오래 살아야 하는 곳 — 에는, 그것이 정확히 올바른 베팅이다: 뒤집기에 *비싼* 부분(domain policy)을 교체하기 *싼* 부분(vendor)으로부터 격리해 둔다.)

버려질 스크립트: 건너뛰어라; indirection이 당신이 결코 하지 않을 교체보다 더 비싸다. 빠르게 움직이는 vendor 풍경 속의 장수하는 domain core: 지불하라; 당신은 vendor를 *반드시* 교체할 것이고, 각 교체가 core 재작성이 아니라 새 adapter이기를 원할 것이다. 이것이 [[insights]] 정식화다 — "what does this keep cheap to change, and what does it make expensive?"(이것은 무엇을 바꾸기 싸게 유지하고, 무엇을 비싸게 만드는가?) — 이 패턴에 대해 답한 것.

> 💡 **쉬운 설명:** 판단 기준은 딱 하나다 — "core가 vendor보다 오래 살 것인가?" 곧 버릴 스크립트라면 port/DTO/mapper 같은 우회 비용이 아깝다(쓰지 마라). 반대로 LLM·DB·프레임워크는 계속 갈아치우는데 비즈니스 정책은 오래 가야 한다면, 그 우회 비용은 "vendor 교체를 core 재작성이 아니라 adapter 추가로" 끝나게 해주므로 충분히 값어치를 한다.

---

## 6. Applied to the Sales Agent (Lina TMR)

학습자의 프로덕션 sales agent인 Lina TMR은 다수의 외부 SaaS tool API 위에서 동작하는 LLM agent다. 그것은 정확히 excerpt가 이 패턴에 대한 올바른 베팅으로 콕 집어낸 경우이므로, 그 적용은 억지가 아니다 — 출처가 염두에 둔 worked example(작동 예제) 그 자체다.

### 6.1 The core is the domain policy, not the model

Lina TMR의 뒤집기에 비싼 부분은 그것의 **domain policy**다: lead가 어떻게 qualify되는가, deal이 언제 stage를 advance하는가, 언제 사람에게 escalate하는가, "stale"(오래된)한 pipeline 항목이 무엇인가. 그 policy는 중심의 entity와 use case에 살아야 하고 model SDK를 `import`하지 말아야 한다. model은 core가 소유한 port 뒤의 **secondary / driven adapter**다 — 그것을 `ForGeneratingReplies`라고 부르자. model 시장이 바뀌면(더 싼 모델, 더 나은 것, 로컬 fallback), 당신은 새 adapter를 작성한다; policy는 손대지 않는다. figure의 LLM-API swap이 정확히 이 동작을 시연한다: 그것을 클릭하고, 교체하고, core가 가만히 있는 것을 보라.

대안 — vendor SDK를 inline으로 호출하는 domain logic — 과 대조하라. 그러면 "switch models"는 결정이 내려진 모든 파일을 건드리는 refactor가 되고, live API key 없이는 "Lina는 언제 escalate하는가?"를 unit-test할 수 없다. 그것이 [[cockburn-hexagonal]]가 죽이기 위해 존재하는 그 비대칭이다.

> 💡 **쉬운 설명:** 핵심 한 줄: "Lina의 진짜 자산은 LLM이 아니라 영업 규칙(언제 자격 판정/단계 진행/사람에게 넘김)이다." 그래서 규칙은 core 안에 두고, LLM은 `ForGeneratingReplies`라는 통로 뒤의 교체 가능한 부품으로 둔다. 만약 규칙 코드 안에서 직접 LLM SDK를 부르면, 모델 하나 바꾸려 해도 규칙 전체를 뜯어고쳐야 하고 API 키 없이는 테스트조차 못 한다.

### 6.2 Every SaaS API is a driven adapter behind a port

Salesforce, Gmail, Google Sheets, calendar — 각각은 core가 선언한 port(`ForCallingTools`, 또는 capability당 하나의 port)를 구현하는 secondary adapter다. 두 가지 귀결:

1. **그들의 응답은 live SDK object가 아니라 plain DTO로 들어온다.** §1.2에 따라, adapter에서 SDK 응답을 domain DTO로 매핑하는 것이 Salesforce의 object model이 agent의 core로 새어 들어오는 것을 막는다. 이것은 또한 [[ch-01]]이 표시했고 ch-06이 중심으로 만드는 inside-vs-outside-data(내부 대 외부 데이터) 정규화의 자연스러운 거처(home)이기도 하다: 외부 SaaS 응답은 *outside data*다 — 버전이 매겨진, 어쩌면 오래된 snapshot — 이고, adapter는 그것이 core가 보기 전에 도장 찍히고(stamped) 동결되는(frozen) 곳이다.
2. **resilience(회복탄력성)가 살 곳이 생긴다.** driven adapter는 timeout, circuit breaker(서킷 브레이커), bulkhead(벌크헤드)가 들어가는 곳(ch-08)이며, 그래서 느린 vendor 하나가 agent loop를 멈추게 할 수 없다. port는 "이 dependency는 실패할 수 있다"를 묻혀 있는 inline 호출이 아니라 명시적이고 감쌀 수 있는(wrappable) seam으로 만든다.

> 💡 **쉬운 설명:** 외부 SaaS의 응답을 그대로 core에 들이지 않고 adapter에서 우리 DTO로 "도장 찍어 동결"한다는 게 1번의 요지다 — 외부 데이터는 낡았을 수도 있는 스냅샷이라 경계에서 한 번 정제하고 들여보낸다. 2번은, timeout·circuit breaker·bulkhead 같은 장애 대비 장치를 둘 *자연스러운 자리*가 바로 이 adapter라는 것. circuit breaker는 자주 실패하는 호출을 잠시 차단하는 두꺼비집, bulkhead는 한 부분의 장애가 전체로 번지지 않게 칸을 나누는 격벽이라고 보면 된다.

### 6.3 Aggregates inside the agent

Vernon의 규칙을 Lina TMR의 domain에 적용하라. 합리적인 aggregate는 `Lead`(또는 `PipelineEntry`)다: agent가 stage를 advance하거나 qualification을 기록하는 그 순간에 일관되게 유지되어야 하는 entity와 value object의 cluster. 규칙 2는 작게 유지하라고 말한다 — 전체 conversation history와 모든 CRM-sync record를 하나의 거대한 `Lead` aggregate에 접어 넣지 마라, 그러지 않으면 모든 update가 contend(경합)한다. 규칙 3은 `Lead`가 `Conversation`을 통째 conversation object로 쥐는 게 아니라 identity로 참조하라고 말한다. 규칙 4는 lead를 advance하는 것이 calendar booking을 trigger해야 할 때, 그것이 nested write(중첩 쓰기)가 아니라 domain event(`LeadQualified`)로 도달되는 *별도* transaction이라고 말한다 — 이것이 in-process saga 씨앗이며, Lina TMR이 언젠가 scheduling을 자체 service로 분리한다면, consistency boundary가 이미 올바른 채로 진짜 distributed saga가 된다.

이것이 [[insights]]에 진술된 through-line(관통선)이다: clean한 bounded context를 가진 modular monolith를 기본값으로 하고, in-process consistency boundary를 먼저 제대로 잡고, granularity disintegrator(세분화를 밀어붙이는 힘)가 명확히 이길 때에만 service를 추출하라 — ch-04가 운영 가능하게(operational) 만드는 결정.

> 💡 **쉬운 설명:** Lina의 `Lead` aggregate를 작게 유지하라는 게 실무 지침이다. 대화 기록 전부와 CRM 동기화 기록까지 한 덩어리에 욱여넣으면 업데이트마다 서로 잠금을 두고 다툰다(경합). 또 lead가 자격 판정되어 캘린더 예약을 만들어야 할 때는, 같은 트랜잭션에 끼워 넣지 말고 `LeadQualified` 이벤트로 별도 트랜잭션을 돌려라 — 이게 §4.3의 in-process saga 씨앗이고, 나중에 예약 기능을 서비스로 떼어내도 일관성 경계가 이미 맞아 있게 된다.

---

## Where This Goes

이 챕터는 단일 boundary의 내부를 구조화했다: technology-free core, core가 소유한 port, 가장자리의 adapter, 안쪽으로만 향하는 dependency, 그리고 in-process consistency 단위로서의 작은 aggregate. 그 선택들 하나하나는 boundary 그 자체가 가만히 있다고 — 그 전체가 하나의 deployable(배포 단위) 안에 산다고 — 가정했다.

Ch-04는 그 가정에 정면으로 도전한다. 그것은 *runtime* boundary가 어디에 떨어져야 하는지를 묻는다: monolith, modular monolith, 또는 microservices — 그리고 그것을 코스에서 가장 뒤집기에 비싼 결정으로 가격 매긴다. 여기서 당신이 그린 aggregate boundary는 후보 seam이 된다; ch-04가 답하는 질문은 그것들 중 어느 것을(있다면) in-process module boundary에서 deployment boundary로 승격시킬 가치가 있는가이며, rule of thumb(경험칙)이 아니라 architecture quantum(아키텍처 퀀텀)과 granularity disintegrators-vs-integrators(세분화를 밀고 당기는 힘들)를 사용한다. §4.3의 in-process saga 씨앗이 그 다리(bridge)다: deployment boundary를 가로지르면 그 씨앗은 consistency phase의 진짜 distributed saga로 자란다.
