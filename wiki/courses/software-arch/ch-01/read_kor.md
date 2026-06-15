<!-- chapter: ch-01
     track: foundations
     kind: content
     title: Architecture Is the Expensive-to-Reverse Decisions
     deps:
     sources: [[richards-ford-fundamentals]], [[richards-ford-hard-parts]], [[nygard-release-it]], [[c4-model]], [[fowler-monolith-first]], [[insights]], [[COLLECTION-PLAN]]
-->

# 01장 — Architecture Is the Expensive-to-Reverse Decisions

> **핵심 통찰.** Architecture는 diagram도, framework도, 클라우드 청구서도 아니다. 그것은 *되돌리기 비싼(expensive to reverse)* 소수의 결정들 — boundary(경계), data ownership(데이터 소유권), contract(계약)에 대한 깊고 구속력 있는 약속들로, 코드와 팀이 그 주위에서 성장하고 나면 더 이상 싸게 되돌릴 수 없는 것들 — 이다. 따라서 이 코스의 모든 pattern은 *어떤 미래의 변경을 싸게 유지할 것인가에 대한 베팅(bet)*이며, First Law(제1법칙)는 그 베팅을 어떻게 채점하는지 알려준다: 모든 것은 trade-off(트레이드오프, 상충 관계)이며, 어떤 pattern이 공짜처럼 보인다면 단지 그 비용을 아직 찾지 못했을 뿐이다. best practice(모범 사례)란 누군가가 당신에게 그 가격을 견적해 주는 것을 잊은 trade-off일 뿐이다.

> 💡 **쉬운 설명:** "되돌리기 비싸다"라는 한 가지 기준이 이 코스 전체의 척추(spine)다. 로깅 라이브러리를 바꾸는 건 화요일에 5분이면 되니까 architecture가 아니다. 하지만 두 팀이 공유하는 service 경계를 바꾸는 건 몇 달짜리 마이그레이션이니까 architecture다. 모든 architecture 패턴을 "이걸 고르면 무엇을 싸게 유지하고, 그 대가로 무엇을 비싸게 만드는가?"라는 질문으로 읽으라는 뜻이다.

> **가이드라인.** architect(아키텍트)의 실제 일은 반복되는 네 가지 동작이다: (1) 모든 것을 극대화하는 대신, 이 시스템이 요구사항(requirement)으로부터 정말로 필요로 하는 *critical few*(핵심 소수) architecture characteristics("-ilities", -성/-력으로 끝나는 비기능적 속성들)를 도출한다; (2) 그 소수를 가장 잘 지원하는 구조를 선택하되, 그것이 무엇을 싸게 만들고 무엇을 비싸게 만드는지 명시적으로 이름 붙인다; (3) 그 결정과 그 결과를 ADR(Architecture Decision Record, 아키텍처 결정 기록)에 기록하여 *왜(why)*가 그것을 만든 사람들보다 오래 살아남게 한다; 그리고 (4) 구조를 한 번에 하나의 C4 zoom level(줌 레벨)로 소통한다. Fowler에 따른 meta-move(메타 동작)는 *되돌릴 수 없는 집합을 줄이는 것(shrink the irreversible set)*이다 — 좋은 architect는 변경을 더 쉽게 만들어서, 시스템이 가진 "architecture"의 양 자체를 줄인다.

---

## 1. The Spine: Architecture Is the Stuff That's Hard to Change

이 코스 전체는 단 하나의 정의에 매달려 있으며, 어떤 pattern보다 먼저 이것을 설치할 가치가 있다.

> **"Architecture is the deep, binding decisions you make about your software"**(Architecture란 당신의 소프트웨어에 대해 내리는 깊고 구속력 있는 결정들이다) — 되돌리기 비싼 것들 말이다.

이것은 [[richards-ford-fundamentals]]에 재현된 널리 쓰이는 실용적 정의다. 이 정의가 의도적으로 *제외한* 것에 주목하라. architecture가 당신의 tech stack(기술 스택)이라고 말하지 않는다. architecture가 microservices(마이크로서비스), Kubernetes, 또는 폴더 구조라고 말하지 않는다. 그런 것들은 세부사항(detail)이다. 그것들은 틀릴 수 있고 당신은 화요일에 그것들을 교체할 수 있다. architecture는 화요일에 틀리는 것이 당신에게 한 분기(quarter)를 — 혹은 재작성(rewrite)이나 팀 재편(team reorg)을 — 치르게 하는 결정들의 *부분집합(subset)*이다.

[[insights]]는 코스의 조직 원리를 한 줄로 진술한다:

> **Architecture is the set of decisions that are expensive to reverse; every pattern in this library is a bet about which changes you must keep cheap.**(Architecture는 되돌리기 비싼 결정들의 집합이다; 이 라이브러리의 모든 pattern은 어떤 변경을 싸게 유지해야 하는가에 대한 베팅이다.)

이 framing이 그 어떤 pattern 카탈로그보다 중요한 이유: 그것은 당신 앞에 놓인 어떤 결정에든 적용할 수 있는 *테스트(test)*를 준다. 어떤 선택이든 물어보라 — *이걸 되돌리는 데 얼마나 비싼가?* 답이 "사소하다"(로깅 라이브러리, 내부 헬퍼의 시그니처)라면, 그것은 architecture가 아니며 고민할 필요가 없다. 답이 "수개월에 걸친 다중 팀 마이그레이션"(service 경계, 공개된 API contract, 누가 어떤 데이터를 소유하는가)이라면, 그것은 architecture *이며* ADR과 trade-off 분석, 그리고 당신의 가장 느리고 가장 적대적인(adversarial) 사고를 받을 자격이 있다.

> 💡 **쉬운 설명:** 여기서 "test"는 코드 테스트가 아니라 머릿속에서 빠르게 돌리는 판별 질문이다. 어떤 결정을 마주할 때마다 "되돌리는 데 얼마나 드나?"라고 묻고, 싸면 무시하고 비싸면 진지하게 다뤄라. "적대적 사고(adversarial thinking)"란 자기 결정을 일부러 공격하며 약점을 찾아내는 사고방식을 말한다.

### 1.1 The reversibility gradient

결정들을 단일 축 — cost-to-reverse(되돌리는 비용) — 위에 그려보는 것이 도움이 되는데, 그 축이 이 코스 전체의 주제이기 때문이다.

| Decision | Cost to reverse | Architecture? |
|----------|-----------------|---------------|
| JSON library; a function name; log format | 분~시간 | 아니오 — 구현 세부사항(implementation detail) |
| Internal module API inside one deployable | 시간~일(리팩터링) | 경계선(borderline) |
| Splitting one service into two with separate DBs | 주~월(마이그레이션) | **예** |
| The published contract other teams consume | 조율된 다중 팀 마이그레이션 | **예** — service가 소유한 가장 비싼 artifact |
| Which data is *authoritative* and who owns it | 재아키텍처(re-architecture) | **예** |

오른쪽 행들이 바로 이후 챕터들이 boundary를 뒤로 미루고(ch-02..04), 공개된 contract를 service가 소유한 가장 되돌리기 비싼 artifact로 취급하며(ch-05), 경계를 넘는 비용을 잃어버린 transaction과 isolation(격리)으로 가격을 매기는(ch-06..07) 이유다. 그것들은 모두 동일한 척추를, 다른 결정에 적용한 것이다.

> 💡 **쉬운 설명:** "authoritative data"(권위 있는 데이터)란 어떤 값의 진짜 정답이 사는 곳을 말한다. 예를 들어 고객 이메일의 진실이 CRM에 있는지 우리 DB에 있는지를 정하는 것이다. 이걸 잘못 정하면 두 시스템이 서로 다른 진실을 주장하게 되어, 나중에 데이터 모델 전체를 다시 설계(re-architecture)해야 한다.

### 1.2 Fowler's amendment: shrink the irreversible set

"바꾸기 어렵다(hard to change)"는 정의에 대한 친절한 보정(corrigendum)이 있는데, [[richards-ford-fundamentals]]에 기록되어 있고 Fowler에게 귀속된다:

> **"A good architect makes change easier — thus reducing architecture."**(좋은 architect는 변경을 더 쉽게 만든다 — 그리하여 architecture를 줄인다.)

이것은 모순이 아니다; 그것은 정의가 함축하는 *지향(aspiration)*이다. architecture가 되돌리기 비싼 집합이라면, 숙련된 architect의 승리 조건은 이전에는 되돌릴 수 없던 것들을 되돌릴 수 있게 만드는 것 — "마이그레이션" 결정을 "리팩터링" 결정으로 변환하는 것이다. modular monolith(모듈러 모놀리스, 모듈식 단일 배포 단위) ([[fowler-monolith-first]])는 바로 이 동작을 boundary에 적용한 것이다: boundary를 단일 deployable(배포 단위) *안에* 유지하여 잘못된 절단(cut)이 cross-service migration(서비스 간 마이그레이션)이 아니라 리팩터링이 되게 한다. 이 생각을 붙들어라 — 그것은 척추의 가장 중요한 실용적 결과이며, ch-04는 그것을 기본값(default)으로 만들 것이다.

> 💡 **쉬운 설명:** 핵심 역설이 여기 있다. architecture를 잘하면 "architecture가 줄어든다"는 말은, 비싼 결정을 일부러 싼 결정으로 바꿔 놓았기 때문에 "되돌리기 비싼 집합"의 크기가 작아진다는 뜻이다. modular monolith가 그 대표 예다: 모듈 경계를 하나의 배포 단위 안에 두면, 경계를 잘못 그어도 코드 정리(리팩터링)로 끝나지 서비스를 쪼개 옮기는 대공사가 되지 않는다.

---

## 2. The First Law: Everything Is a Trade-off (and the "No Best Practices" Corollary)

척추가 architecture가 *무엇인지*를 알려준다면, First Law는 *그것에 관한 모든 결정을 어떻게 평가하는지*를 알려준다. [[richards-ford-fundamentals]]에서, Richards & Ford가 발표한 그대로 인용한다:

> **"Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet."**(소프트웨어 architecture의 모든 것은 trade-off다. trade-off가 아닌 무언가를 찾았다고 생각한다면, 당신은 아마도 그 trade-off를 아직 찾지 못했을 뿐이다.) — Richards & Ford

이미 trade-off로 사고하는 독자에게 이것은 진부한 말이 아니다 — 그것은 *운영 규율(operating discipline)*이다. 그것은 특정한 실패 모드를 금지한다: 견적된 비용 없이 pattern을 승리로 제시하는 것. 이 코스가 pattern을 소개할 때마다, 그것은 즉시 그것을 베팅으로 가격 매길 것이다 — *이것은 X를 싸게 바꿀 수 있게 유지하되, Y를 비싸게 만드는 대가로.* 만약 이 코스에서(혹은 어디서든) pattern을 읽었는데 그 Y를 이름 붙일 수 없다면, First Law는 당신이 아직 그것을 이해하지 못했다고 말한다; 당신은 단지 그것을 팔렸을(sold) 뿐이다.

> 💡 **쉬운 설명:** "팔렸다(sold)"는 표현은 광고 문구를 그대로 믿었다는 뜻이다. "마이크로서비스는 확장성을 줍니다" 같은 한쪽 면만 듣고 비용(Y)을 묻지 않았다면, 이해한 게 아니라 영업당한 것이다. First Law를 진짜로 쓰려면 모든 패턴마다 "X는 싸지고 Y는 비싸진다"의 Y를 직접 적어야 한다.

### 2.1 The corollary: there are no best practices

*Software Architecture: The Hard Parts* ([[richards-ford-hard-parts]])는 이것을 이 분야 전체에 정직함을 주는 corollary(따름정리)로 날카롭게 다듬는다. 이 책은, 저자들의 framing에 따르면, 다음에 관한 것이다

> 어려운 문제들 "with no best practices that force you to choose among various compromises"(다양한 타협 중에서 선택하도록 강요하는, best practice가 없는),

그리고 그것은 "how to think critically about the trade-offs involved with distributed architectures"(distributed architecture에 관련된 trade-off에 대해 비판적으로 사고하는 법)를 가르치고 "techniques to help you discover and weigh the trade-offs"(trade-off를 발견하고 저울질하도록 돕는 기법)를 주려 한다. (이것은 [[richards-ford-hard-parts]]에 따른 저자 자신의 framing에서 추출한 책의 논지로, O'Reilly 설명에 의해 뒷받침된다 — 한 문장짜리 그대로의 인용이 아니므로, 귀속된 의역(attributed paraphrase)으로 읽으라.)

실용적 해석: "best practice"는 architecture에서 category error(범주 오류)다. best practice는 맥락을 무시하는 레시피이고, architecture *는* 맥락을 저울질하는 규율이다. 이 코스 전체가 죽이는 가장 큰 신화 — trade-off 척추 노트에서 표시된 — 는 "microservices / DDD / CQRS / REST are best practices"(이것들이 best practice다)이다. 그것들 중 어느 것도 그렇지 않다. 각각은 이름 붙은 비용을 가진 trade이며, 앞으로의 챕터들은 당신에게 각각의 이름 붙은 비용을 가르치기 위해 존재한다.

> 💡 **쉬운 설명:** "best practice가 없다"는 말은 "아무것도 하지 말라"가 아니라 "맥락 없이 정답은 없다"는 뜻이다. 같은 microservices가 어떤 팀에는 약이고 어떤 팀에는 독이다. 그래서 정답을 외우는 대신, 우리 시스템에서 그 비용(Y)이 감당할 만한지를 매번 따져야 한다.

### 2.2 The discipline in practice: "find the Y"

First Law는 당신이 *하는 일*을 바꿀 때만 유용하므로, 그것을 고정된 의식(ritual)으로 만들어라: 모든 pattern마다, **"this keeps X cheap to change, at the cost of making Y expensive."**(이것은 X를 싸게 바꿀 수 있게 유지하되, Y를 비싸게 만드는 대가로) 문장을 써라. Y를 채울 수 없다면, 멈춰라 — 당신은 분석을 끝내지 못한 것이다. 코스가 발전시킬 pattern에서 가져온 두 가지 worked example(풀이 예제):

- **Microservices.** X = 각 capability(역량)의 independent deployability(독립적 배포 가능성)(나머지를 건드리지 않고 한 부분을 배포/확장/소유). Y = capability 간 transactional consistency(트랜잭션 일관성), 운영 단순성, 그리고 local function call(로컬 함수 호출)(이제는 실패할 수 있는 network hop(네트워크 홉)이 됨). First Law는 "microservices는 확장성을 준다"고 말하고 거기서 멈추는 것을 금지한다; 정직한 문장은 잃어버린 transaction과 distribution tax(분산 세금)를 이름 붙인다. Ch-04가 Y를 구체화한다; [[richards-ford-hard-parts]]의 granularity framework(세분성 프레임워크)는 문자 그대로 이 문장의 양쪽에 작용하는 힘들의 열거다.
- **A clean technology-free core (ch-03).** X = LLM vendor, vector DB, web framework를 교체할 자유 — 그것들은 core가 소유한 interface 뒤에 앉는다. Y = indirection(간접화) 그 자체: 작은 CRUD 앱에는 순수한 overhead인 port, DTO, mapper. 이 베팅이 보상받는지 여부는 core가 얼마나 오래 사느냐와 edge(가장자리)가 얼마나 변덕스러우냐에 달려 있다 — 이것은 context 판단이며, 즉 "no best practice"가 의미하는 바로 그것이다.

ritual은 확장된다: 손짓으로 얼버무린 Y나 빠진 Y를 가진 결정은 당신이 얻어내지 못한(earned) 결정이며, ADR(§4)은 당신이 *Consequences*(결과) 섹션에 Y를 적도록 강제되는 곳이다.

> 💡 **쉬운 설명:** "distribution tax"(분산 세금)란 시스템을 여러 service로 쪼갰을 때 항상 따라붙는 추가 비용 — 네트워크 지연, 실패 처리, 모니터링, 디버깅의 어려움 — 을 비유한 말이다. 공짜로 확장성을 얻는 게 아니라, 매달 내야 하는 세금처럼 운영 복잡성을 떠안는 것이다. "indirection"은 직접 호출 대신 인터페이스 한 겹을 끼워 넣는 것으로, 유연성은 늘지만 코드가 한 단계 더 돌아간다.

### 2.3 The myth, surfaced: "architecture = applying best practices"

이 챕터의 doc-vs-reality 조정(reconciliation)은([[COLLECTION-PLAN]]과 outline의 신화 표에서) 두 갈래다. 두 번째 신화는 **"architecture is a body of best practices you apply."**(architecture는 당신이 적용하는 best practice의 집합이다)이다. primary-source(1차 출처) 해소는 First Law 그 자체다: 맥락 없는 best practice는 없으며, 당신이 드러내고 저울질해야 하는 trade-off만 있을 뿐이다. "the best practice"에 손을 뻗는 architect는 architecture를 cargo-culting(화물 숭배, 이유 모르고 따라하기)과 구별하는 유일한 단계 — *이* 시스템에 대해 베팅을 가격 매기는 것 — 를 건너뛴 것이다. [[richards-ford-hard-parts]]의 corollary는 distributed architecture에서 가장 어려운 결정들이 "no best practices that force you to choose among various compromises"(다양한 타협 중에서 선택하도록 강요하는, best practice가 없는)임에 대해 직설적이다; 지속되는 기술은 그러한 타협들을 발견하고 저울질하는 것이지, 레시피를 외우는 것이 아니다. 첫 번째 신화(C4 "container")는 §5에서 해소할 것이다.

> 💡 **쉬운 설명:** "cargo-culting"은 2차 세계대전 후 남태평양 원주민들이 활주로 모형을 만들면 비행기가 다시 올 거라 믿었다는 일화에서 온 표현이다. 형식만 따라 하고 작동 원리를 모르는 모방을 뜻한다. 남이 microservices를 쓴다고 우리도 무작정 쓰는 것이 바로 architecture에서의 cargo-culting이다.

---

## 3. Architecture Characteristics: Derive the Critical Few, Don't Maximize All

First Law는 즉각적이고 가혹한 결과를 가진다: 당신은 모든 바람직한 속성을 극대화할 수 *없다*. [[richards-ford-fundamentals]]에서, architect의 첫 번째 진짜 과제는 비즈니스 요구사항으로부터 **critical few** architecture characteristics — "-ilities" — 를 도출하는 것인데, 왜냐하면 당신은 그것들을 모두 가질 수 없기 때문이다.

architecture characteristics는 실제로 구조를 추동하는 비기능적(non-functional) 속성들이다: scalability(확장성), performance(성능), availability(가용성), security(보안), **deployability**(배포 가능성), testability(테스트 가능성), modifiability(수정 가능성), fault tolerance(장애 내성) 등등. 목록은 길다; 그것이 함정이다. 기술은 목록을 아는 것이 아니다 — 그것은 *이 시스템에 중요한 서너 개를 골라내고 나머지는 그저 적당할(merely adequate) 것을 받아들이는 것*이다.

### 3.1 The "-ilities" pull in opposite directions

당신이 선택해야 하는 이유는 characteristics가 긴장(tension) 관계에 있기 때문이다. [[richards-ford-fundamentals]]는 가장 깔끔한 예를 주며, 그것은 topology 챕터(ch-04) 전체가 의존하는 예다:

> deployability 같은 characteristic은 정확히 당신을 microservices 쪽으로 미는 것이고; simplicity(단순성)는 정확히 당신을 monolith 쪽으로 되미는 것이다.

| If your critical characteristic is… | The structure it pulls toward | What it costs you |
|-------------------------------------|-------------------------------|-------------------|
| Deployability / independent release | services(별도 deployable) | 단순성, transactional consistency, ops/distribution tax |
| Simplicity / low operational cost | monolith / modular monolith | 부분의 독립적 확장 및 배포 |
| Elastic scalability of one hot path | 그 path를 자체 quantum으로 추출 | 나머지와의 shared-transaction 보장 |
| Auditability / temporal replay | event sourcing (ch-07) | "위험한 복잡성"; replay와 schema-evolution 위험 |

이 표를 다음과 같이 읽어라: *가운데 열의 모든 셀은 베팅이고, 오른쪽 열은 그 가격이다.* 그것이 구조의 일상적 선택으로 운영화된(operationalized) First Law다.

> 💡 **쉬운 설명:** "hot path"란 트래픽이 몰리는 특정 기능 경로를 말한다. 예컨대 결제 처리처럼 다른 부분보다 훨씬 자주 호출되어 따로 확장해야 하는 길이다. 이걸 "quantum"(아키텍처 양자, ch-04에서 정의될 독립 배포·확장 단위)으로 떼어내면 그 길만 따로 키울 수 있지만, 나머지와 한 트랜잭션으로 묶을 수 있는 보장을 잃는다.

### 3.2 Requirements derive the characteristics; the characteristics derive the style

위에서 아래로의 인과 사슬은: **business requirements → critical few characteristics → architectural style → patterns.**(비즈니스 요구사항 → 핵심 소수 characteristics → 아키텍처 스타일 → pattern) 대부분의 architecture 실패는 이 사슬의 역전이다 — 먼저 스타일을 고르고(유행이기 때문에) 그다음에 그것을 정당화하기 위해 요구사항을 끼워 맞추는 것. deployability가 이 시스템에 critical characteristic이기나 한지 묻기 *전에* "We'll do microservices"(우리는 microservices로 할 거야)라고 하는 것이 distributed monolith(분산 모놀리스)(ch-04)를 만들어내는 역전이다. 규율은 스타일이 *무엇을 위한 것인지* 그 소수의 characteristics를 이름 붙이기 전까지는 스타일에 이름 붙이기를 거부하는 것이다.

> 💡 **쉬운 설명:** "distributed monolith"(분산 모놀리스)는 최악의 조합이다 — 겉으로는 여러 service로 나뉘어 있는데(분산의 비용은 다 치름) 실제로는 서로 강하게 묶여 함께 배포해야 한다(monolith의 제약은 그대로). 즉 두 세계의 단점만 모은 것이다. 요구사항을 보지 않고 유행 따라 쪼개면 이게 나온다.

### 3.3 Why "the critical few" is itself a trade-off

여기 놓치기 쉬운 second-order(2차) First-Law 동작이 숨어 있다. *어떤* characteristics가 critical인지 고르는 것 자체가 되돌리기 어려운(irreversible-ish) 베팅인데, 왜냐하면 당신이 최적화하는 characteristics가 구조를 형성하고, 그 구조는 당신이 우선순위에서 내린 characteristics에 저항하기 때문이다. deployability가 critical이라고 결정하고 그것을 위해 빌드하면, 당신은 simplicity 예산(budget)을 써버린 것이다; 나중에 simplicity가 중요한 것이었다고 결정하는 것은 config 변경이 아니다 — 그것은 재아키텍처다. 이것이 ch-03의 clean core(깨끗한 코어)와 ch-04의 modular-monolith-first 기본값이 그토록 가치 있는 이유다: 그것들은 *싼* characteristics(vendor-modifiability, testability)를 진정으로 싸게 유지하면서, 비싼 구조적 약속(deployability-via-services)을 요구사항이 실제로 요구할 때까지 미룬다. 적은 characteristics를 고르는 것이 좋은 이유는 적은 것이 더 깔끔해서가 아니라, 당신이 약속하는 각각이 나머지에 대해 부분적으로 닫는 문이기 때문이다.

> 💡 **쉬운 설명:** "simplicity 예산을 써버렸다"는 비유가 핵심이다. characteristic은 무한정 쌓을 수 없는 한정된 자원과 같아서, deployability에 몰아주면 simplicity 쪽 잔고가 바닥난다. 그래서 어떤 characteristic을 critical로 고르는 것 자체가 "다른 문을 부분적으로 닫는" 되돌리기 어려운 결정이 된다.

---

## 4. ADRs: The Durable Record of the *Why*

당신이 내리고 나서 잊어버린 trade-off는 6개월 뒤 그것을 추론해낸 사람이 떠났을 때 다시 따져야 할(re-litigate) — 혹은 더 나쁘게는, 조용히 위반할 — trade-off다. [[nygard-release-it]]에서 Nygard의 두 가지 기여 중 두 번째가 해독제다: **Architecture Decision Record**.

> **"An architecture decision record is a short text file in a format similar to an Alexandrian pattern."**(architecture decision record는 Alexandrian pattern과 유사한 형식의 짧은 텍스트 파일이다.) — Nygard

ADR이 해결하는 문제는 분명하게 진술된다:

> **"One of the hardest things to track during the life of a project is the motivation behind certain decisions."**(프로젝트의 일생 동안 추적하기 가장 어려운 것 중 하나는 특정 결정 뒤의 동기다.) — Nygard

결정의 *무엇(what)*은 보통 코드에서 보인다; *왜(why)*는 거의 절대 보이지 않는다. 코드는 당신이 saga(사가)를 골랐다는 것을 보여준다; 그것은 당신이 그것을 고른 이유가 database와 message broker에 걸친 2PC(two-phase commit, 2단계 커밋)가 viable하지 않았고 팀이 잃어버린 isolation을 그 가격으로 명시적으로 받아들였기 때문이라는 것을 보여주지 않는다. 그 추론이 복구하기 비싼 부분이며, 그것이 정확히 ADR이 보존하는 것이다.

> 💡 **쉬운 설명:** "saga"는 여러 service에 걸친 작업을 하나의 큰 트랜잭션 대신 작은 단계들의 연쇄로 처리하고, 실패하면 보상 동작(compensation)으로 되돌리는 패턴이다(ch-06/07에서 다룸). "2PC"는 여러 시스템을 한 번에 commit하거나 한 번에 취소하는 고전적 방법이지만, DB와 message broker처럼 종류가 다른 시스템에 걸치면 잘 작동하지 않는다("not viable"). ADR은 "왜 2PC를 버리고 saga를 골랐는가"라는, 코드에 안 남는 이유를 기록한다.

### 4.1 What to record, and in what shape

Nygard는 *어떤* 결정이 ADR을 받을 자격이 있는지에 대해 구체적이다 — 그리고 그것은 정확히 척추의 "expensive to reverse" 집합이다:

> **"We will keep a collection of records for 'architecturally significant' decisions: those that affect the structure, non-functional characteristics, dependencies, interfaces, or construction techniques."**('아키텍처적으로 중요한' 결정들에 대한 기록 모음을 유지할 것이다: 구조, 비기능적 characteristics, 의존성, interface, 또는 구축 기법에 영향을 주는 것들.) — Nygard

구조는 의도적으로 최소화되어 있어서, 하나를 쓰는 비용이 추론을 잃는 비용보다 낮게 유지된다:

| Section | What it holds |
|---------|---------------|
| **Title** | 결정에 이름 붙이는 짧은 명사구 |
| **Status** | proposed / accepted / deprecated / **superseded**(결정은 죽지만; 기록은 남는다) |
| **Context** | 긴장 관계의 힘들 — 이것을 진짜 선택으로 만든 요구사항과 characteristics |
| **Decision** | 무엇이 선택되었는지, 능동태(active voice)로 |
| **Consequences** | 무엇이 더 쉬워지고 *무엇이 더 어려워지는가* — 즉 가격 매겨진 베팅 |

Nygard의 지침은 그것을 "one or two pages"(한두 페이지)로 유지하고 "write each ADR as if it is a conversation with a future developer"(각 ADR을 미래의 개발자와의 대화인 것처럼 써라)는 것이다. 그 Consequences 섹션이 당신의 repo에서 First Law가 사는 곳이다: "what becomes harder"(무엇이 더 어려워지는가)가 없는 ADR은 자신의 결정을 실제로 이해하지 못한 ADR이다.

### 4.2 Status is a feature, not bookkeeping

**superseded**(대체됨) status는 사람들이 건너뛰는 부분이며 척추에 가장 중요한 부분이다. ADR은 append-only(추가 전용) 역사다: 당신은 결정을 절대 삭제하지 않고, 거꾸로 링크하는 더 새로운 것으로 그것을 supersede(대체)한다. 이것은 비싼 결정의 *반전(reversal)* 자체를 기록되고 추론된 사건으로 만든다 — 이것이 조직이 경계 실수를 반복하는 대신 그것으로부터 배우는 유일한 방법이다. ADR 로그는, 문자 그대로, 당신의 되돌리기 비싼 집합의 변경 로그(change-log)다.

> 💡 **쉬운 설명:** "append-only"란 줄을 지우지 않고 뒤에만 덧붙이는 방식이다. 옛 결정을 지우는 대신 "이 결정은 ADR-017로 대체됨"이라고 표시하고 새 ADR이 옛 것을 링크한다. 그래야 "왜 예전엔 A였는데 지금은 B인가"라는 변천사가 통째로 남아, 같은 실수를 반복하지 않게 된다.

### 4.3 From recording the *why* to enforcing it: fitness functions

결정을 기록하는 것은 *왜*를 살려두지만, 시스템이 성장하면서 결정이 조용히 썩는 것을 막는 데에는 아무것도 하지 못한다 — 깨끗한 dependency rule(의존성 규칙)은 부주의한 import 하나씩 한 번에 침식된다. [[richards-ford-fundamentals]]의 evolutionary-architecture(진화적 아키텍처) 자료(Ford, Parsons & Kua)에서 온 보완물은 **fitness function**(피트니스 함수)이며, 다음과 같이 정의된다

> **"an objective integrity assessment of some architectural characteristic(s)."**(어떤 architectural characteristic(들)에 대한 객관적 무결성 평가.) — Ford, Parsons & Kua

구체적으로 그것은 자동화된 검사다 — 테스트, 메트릭, 모니터, ArchUnit rule — 로, 보호된 characteristic이 저하될 때 *빌드를 실패시킨다(fails the build)*. 그것은 "keep the dependency rule intact"(의존성 규칙을 온전히 유지하라)나 "p99 latency < X"(p99 지연 시간이 X 미만)를 wiki에 쓰인 지향에서 enforce된 불변식(invariant)으로 바꾼다. 책은 더 큰 목표를 이렇게 framing한다:

> **"An evolutionary architecture supports guided, incremental change across multiple dimensions."**(진화적 architecture는 여러 차원에 걸친 안내된 점진적 변경을 지원한다.) — Ford, Parsons & Kua

척추 연결은 직접적이다: ADR은 당신이 베팅을 했다는 것을 기록하고; fitness function은 그 베팅이 누구도 알아채지 못한 채 엔트로피에 잃어버려지는 것을 막는다. Ch-09는 fitness function을 진화적 architecture의 enforcement arm(집행 부서)으로 발전시킨다; 챕터 1은 단지 그것들을 여기 심어서, 당신이 §3에서 고르는 모든 characteristic을 단지 선언하는 것이 아니라 결국 *보호해야(protect)* 할 무언가로 읽게 한다.

> 💡 **쉬운 설명:** "fitness function"은 architecture의 자동 감시 장치다. 예를 들어 "도메인 코어는 절대 외부 라이브러리를 import하면 안 된다"는 규칙을 ArchUnit 같은 도구로 테스트화하면, 누군가 실수로 그 import를 넣는 순간 CI 빌드가 빨갛게 실패한다. "p99 latency < X"는 요청 100건 중 99번째로 느린 것조차 X보다 빨라야 한다는 성능 기준이다. 즉 위키에 적어둔 약속을 코드로 강제하는 것이다.

---

## 5. C4: One Notation, One Zoom Level at a Time

당신은 척추(비싼 결정), 법칙(trade-off), 고르는 방법(critical characteristics), 그리고 기록하는 방법(ADR)을 가졌다. 마지막 기초 도구는 그 결정들이 만들어내는 구조에 대해 이야기할 *공유 표기법(shared notation)*이다 — 왜냐하면 대부분의 architecture diagram은 추상화 수준을 뒤섞고 아무도 읽을 수 없는 임시(ad-hoc) 기호를 발명함으로써 실패하기 때문이다. Simon Brown의 **C4 model**([[c4-model]])은 고정된 추상화 계층으로 이것을 고친다.

> **"An easy to learn, developer friendly approach to software architecture diagramming."**(배우기 쉽고 개발자 친화적인 소프트웨어 architecture 다이어그래밍 접근법.) — Brown

> **"A set of hierarchical abstractions — software systems, containers, components, and code."**(계층적 추상화의 집합 — software systems, containers, components, code.) — Brown

요점은 *계층(hierarchy)*이다: C4 diagram은 정확히 하나의 zoom level에서의 지도다. 당신은 "the architecture"(아키텍처)를 그리지 않는다; 당신은 Context를, *또는* Containers를, *또는* 한 container의 Components를 그린다 — 결코 한꺼번에 전부가 아니다.

### 5.1 The four levels (the zoom)

구성 규칙(composition rule)은 [[c4-model]]을 통한 Brown의 그대로다:

> **"A software system is made up of one or more containers (applications and data stores), each of which contains one or more components, which in turn are implemented by one or more code elements (classes, interfaces, objects, functions, etc)."**(software system은 하나 이상의 container(애플리케이션과 데이터 저장소)로 구성되며, 각각은 하나 이상의 component를 포함하고, 그것은 다시 하나 이상의 code element(클래스, 인터페이스, 객체, 함수 등)로 구현된다.) — Brown

| Level | What it shows | Audience |
|-------|---------------|----------|
| **1. Context** | 시스템을 하나의 박스로, 그것의 사용자(Persons)와 그것이 대화하는 외부 시스템(external systems)에 둘러싸여 | 모두 |
| **2. Container** | 줌인: 시스템을 구성하는 별도로 배포 가능한 앱과 데이터 저장소, 그리고 그것들이 어떻게 통신하는지 | 기술 + ops |
| **3. Component** | 한 container로 줌인: 그 안의 interface 뒤에 묶인 책임들(grouped responsibilities) | 그 container의 개발자들 |
| **4. Code** | 한 component로 줌인: 클래스/UML 세부사항 — 선택적, 보통 IDE가 생성 | 손으로 그리는 경우 드묾 |

> **지금 interactive companion을 열고 네 레벨을 모두 클릭해 보라:** [figures/c4-zoom.html](figures/c4-zoom.html). 각 클릭은 이전 박스로 줌인한다 — Context → Container → Component → Code — 그래서 당신은 diagram이 한 레벨에 산다는 것을 *느낄(feel)* 수 있다. 상단 모서리의 토글은 모델 전체에서 가장 흔한 오독(misread)을 겹쳐 보여주며, 다음에서 다룬다.

> 💡 **쉬운 설명:** C4의 핵심은 "지도를 줌 레벨마다 따로 그린다"는 것이다. 세계 지도와 동네 골목 지도를 한 장에 섞어 그리면 아무도 못 읽는 것처럼, 시스템 전체와 클래스 세부사항을 한 다이어그램에 섞으면 안 된다. Context는 위성 사진, Code는 골목 단위라고 생각하면 된다. 위 HTML companion을 클릭해 가며 줌이 한 단계씩 들어가는 감각을 직접 느껴보라.

### 5.2 The myth, killed: a C4 "container" is **not** a Docker container

이것은 챕터의 첫 번째 조정 신화(reconciliation myth)다([[COLLECTION-PLAN]] 표와 outline에서). 널리 퍼진 서사 — 10년의 `docker` 근육 기억에 의해 강화된 — 는 **"a C4 container is a Docker container."**(C4 container는 Docker container다)이다. 그것은 거짓이며, 1차 출처는 명확하다. [[c4-model]]에서, container에 대한 Brown 자신의 정의:

> container는 **"applications and data stores"**(애플리케이션과 데이터 저장소)다 — 별도로 실행/배포 가능한 단위: server-side app, single-page app, mobile app, database, file system, message bus — **Docker container가 아니다.**

container의 C4 의미는 OCI/Docker container *보다 앞서며 더 넓다*; 그것은 "별도로 실행되며 코드나 데이터를 담는 것"을 의미하지, "Dockerfile로부터 빌드된 것"을 의미하지 않는다. PostgreSQL database는 C4 container다. browser SPA는 C4 container다. message bus는 C4 container다. 그것들 중 어느 것도 반드시 Docker container는 아니며, 둘을 혼동하는 것은 이 코스에서 가장 중요한 단 하나의 diagram 레벨 — Container diagram — 을 오염시킨다. 왜 그것이 가장 중요한가? 왜냐하면, [[c4-model]]이 언급하듯, **Container diagram은 정확히 distributed monolith가 가시화되는 곳**이기 때문이다: 당신은 deploy-coupled(배포 결합된) container들과 그것들이 공유하는 데이터 저장소를 센다. "container"를 잘못 이름 붙이면 가장 비싼 경계 실수를 *볼* 능력을 잃는다. (Ch-04가 architecture quantum(아키텍처 양자)을 통해 이 탐지기를 형식화한다.)

> 💡 **쉬운 설명:** 헷갈림의 핵심은 같은 단어 "container"가 두 세계에서 다른 뜻이라는 것이다. C4의 container는 "따로 돌아가는 실행 단위 또는 데이터 저장소"라는 넓은 개념(데이터베이스, 브라우저 앱, 메시지 버스 전부 포함)이고, Docker container는 Dockerfile로 만든 구체적 기술이다. 둘을 섞으면 Container diagram을 잘못 그리게 되는데, 바로 그 diagram이 "여러 박스로 나뉜 척하지만 실은 함께 배포되고 같은 DB를 공유하는" distributed monolith를 눈으로 발견하는 자리이기 때문에 치명적이다.

---

## 6. The Architect's Loop: How the Four Tools Compose

이 코스는 pattern 카탈로그가 아니라 하나의 연결된 논증(connected argument)이 되도록 의도되었으므로, 네 가지 기초 도구를 agent에 겨누기 전에 그것들이 어떻게 단일 loop(루프)를 형성하는지 진술할 가치가 있다. 그것들은 네 가지 독립된 주제가 아니다; 그것들은 하나의 반복된 행위의 네 단계다.

1. **Spine** — 당신 앞에 놓인 어떤 결정이 실제로 되돌리기 비싼지 식별한다. 대부분은 아니다; 당신의 노력을 그런 것들에만 쓴다(§1). 이것은 나머지 loop가 언제 적용되기나 하는지 알려주는 필터다.
2. **Characteristics** — 비싼 결정에 대해, 요구사항으로부터 critical few "-ilities"를 도출하되, 모두를 극대화할 수 없음을 받아들인다(§3). 이것이 선택을 유행적인(fashionable) 것이 아니라 *원칙적인(principled)* 것으로 만드는 것이다.
3. **First Law / price the bet** — 그 characteristics를 지원하는 구조에 이름 붙이고 "keeps X cheap, makes Y expensive" 문장을 쓴다(§2). Y가 없는 결정은 끝난 것이 아니다.
4. **ADR** — context, 결정, 그리고 가격 매겨진 결과를 기록하여 *왜*가 살아남게 한다; 나중에, 보호된 characteristic을 fitness function으로 방어한다(§4). 이것이 베팅을 신비로운 것이 아니라 *수정 가능한(revisable)* 것으로 만드는 것이다.
5. **C4** — 결과 구조를 하나의 zoom level로 소통하여 다른 사람들이 그것을 보고, 비평하고, 물려받을 수 있게 한다(§5). Container 레벨은 최악의 경계 베팅들이 가시화되는 곳이다.

loop가 닫혀 있는(closed) 것은 단계 4의 fitness function과 단계 5의 diagram이 단계 1로 되먹임(feed back)되기 때문이다: characteristic이 침식되거나 Container diagram이 deploy-coupled 박스들을 보이기 시작할 때, 당신은 새로운 비싼 결정을 드러낸 것이고 loop가 다시 돈다. [[insights]]는 같은 생각을 반대 방향에서 framing한다 — "read every excerpt as: what does this keep cheap to change, and what does it make expensive?"(모든 발췌를 다음과 같이 읽어라: 이것은 무엇을 싸게 바꿀 수 있게 유지하고, 무엇을 비싸게 만드는가?). 이 챕터 이후의 모든 챕터는 이 loop를 한 종류의 결정에 적용한 한 번의 통과다: boundary(ch-02..04), contract(ch-05), consistency(ch-06..07), resilience와 evolution(ch-08..09). capstone(ch-10)은 학습자의 실제 시스템에 대해 전체 loop를 끝에서 끝까지 돌린다.

> 💡 **쉬운 설명:** 이 다섯 단계가 한 번 쓰고 버리는 체크리스트가 아니라 계속 도는 순환(loop)이라는 점이 핵심이다. 단계 4의 fitness function이 빨갛게 터지거나 단계 5의 Container diagram에서 "함께 배포되는 박스들"이 보이면, 그것이 곧 "새로운 비싼 결정이 등장했다"는 신호이고, 다시 단계 1부터 돈다. 코스의 나머지 챕터들은 이 같은 루프를 boundary, contract, consistency 식으로 다른 종류의 결정에 한 바퀴씩 돌린 것이다.

---

## 7. Framing the Whole Course for the Sales Agent (Lina TMR)

이 코스의 연구-framing(research-framing) 보상은 agent를 평가하는 것이 아니다 — 학습자는 이미 그것을 위한 agent-benchmark 코스를 닫았다. 그것은 agent를 *설계하는(designing)* 것이다: 프로덕션 sales agent인 **Lina TMR**, 여러 외부 SaaS tool API(CRM, email, calendar, sheets, ticketing)에 걸쳐 작동하는 LLM agent. 챕터 1에서 척추를 설치하는 전체 요점은, 그 시스템에 대해 중요한 단 하나의 질문을 묻는 것이다: **그것의 결정 중 어느 것이 되돌리기 비싼 것들인가?**

### 7.1 Sorting the agent's decisions on the reversibility gradient

§1.1의 테스트를 Lina TMR의 실제 결정들에 대해 돌려보라:

| Decision about the agent | Cost to reverse | Architecture? |
|--------------------------|-----------------|---------------|
| 어떤 LLM model / vendor가 core reasoning을 뒷받침하는가 | 낮음–중간(interface 뒤에서 교체) | **아니오, 격리한다면** — 의도적으로 싸게 만들라 |
| 어떤 vector DB / framework 버전을 사용하는가 | 낮음–중간 | **아니오, 격리한다면** |
| agent의 **bounded contexts**(lead/pipeline vs conversation vs scheduling vs CRM-sync) | 높음 — 틀리면 마이그레이션 | **예** — 되돌릴 수 없는 결정(ch-02) |
| 각 capability가 module인가 별도 service인가 | 리팩터링(module) vs 마이그레이션(service) | **예** — 미뤄라(ch-04) |
| tool/integration 레이어가 나머지 agent에 노출하는 **contract** | 모든 caller에 걸친 조율된 변경 | **예** — 가장 비싼 artifact(ch-05) |
| 각 외부 SaaS 응답을 authoritative live state로 취급할 것인가 vs immutable, versioned, 어쩌면-오래된(possibly-stale) snapshot으로 | 데이터 모델의 재아키텍처 | **예** — 근본 구분(root distinction)(ch-06) |

이 표에서 두 가지가 즉시 떨어져 나오며, 그것들은 [[insights]]에 진술된 through-line(관통선)이다. 첫째, model/vendor/framework 선택 — 엔지니어들이 *고뇌하는(agonize)* 것들 — 은 *의도적으로 싸게 엔지니어링되어야(deliberately engineered to be cheap)* 한다(Fowler의 "shrink the irreversible set," §1.2): 그것들을 agent의 core가 소유한 interface 뒤에 숨겨서 현재 LLM API나 vector DB를 교체하는 것이 마이그레이션이 아니라 화요일의 변경이 되게 하라. 둘째, 진정으로 비싼 결정들은 boundary, contract, 그리고 inside-vs-outside 데이터 선이며 — 그것들이 정확히 코스가 첫날에 추측하는 대신 미루고, 가격 매기고, 기록하도록 가르치는 것들이다.

> 💡 **쉬운 설명:** "bounded context"(바운디드 컨텍스트, ch-02에서 본격적으로 다룸)란 같은 단어가 도메인마다 다른 뜻을 갖는 경계다. 예컨대 sales에서 "고객"은 파이프라인의 리드(lead)지만 support에서 "고객"은 티켓을 가진 계정이다. Lina TMR에서 가장 비싼 결정은 이런 경계, 외부에 노출하는 contract, 그리고 "외부 SaaS 응답을 살아있는 진실로 볼 것인가 오래된 스냅샷으로 볼 것인가"이며, 반대로 어떤 LLM을 쓸지는 인터페이스 뒤에 숨겨 일부러 싸게 만들어야 한다.

### 7.2 A first ADR for Lina TMR, in the §4 shape

도구들을 추상적이 아니라 구체적으로 만들기 위해, 여기 챕터의 어휘를 agent를 위해 당신이 쓸 *첫 번째* ADR로 적용한 것이 있다 — 척추가 예측하는 topology 베팅을 기록하는 것:

| Section | Content |
|---------|---------|
| **Title** | Start Lina TMR as a modular monolith with clean bounded contexts |
| **Status** | Accepted |
| **Context** | 도메인(sales pipeline + conversation + scheduling + CRM-sync)이 아직 잘 이해되지 않았다; 팀이 작다; boundary는 되돌리기 가장 비싼 결정이며 그것을 일찍 올바르게 잡는 것은, Fowler에 따르면, 전문가에게조차 매우 어렵다. |
| **Decision** | 하나의 deployable. bounded context별로 엄격한 내부 module boundary를 enforce한다; cross-module call은 호출하는 module이 소유한 in-process interface를 거친다; 어떤 module도 다른 module의 table에 손을 뻗지 않는다. |
| **Consequences** | *Cheap:* 잘못된 context boundary는 cross-service migration이 아니라 리팩터링이다; vendor 교체(LLM, vector DB)는 port 뒤에 숨는다. *Expensive (the Y):* 아직 단일 capability의 독립적 확장/배포는 없다; 측정된 힘(hot path, 별도 팀)이 quantum 추출을 정당화할 때까지 우리는 그것을 받아들인다(ch-04). |

이 단일 기록이 챕터 전체를 시연한다: *Context*는 critical characteristic과 reversibility 비용을 이름 붙인다; *Decision*은 구조를 고른다; *Consequences*는 X와 Y를 모두 이름 붙임으로써 베팅을 가격 매긴다. capstone lab(ch-10)은 Lina TMR을 위한 이것들의 전체 집합과, distributed monolith가 가시화될 C4 Context/Container 스케치를 빌드한다.

> 💡 **쉬운 설명:** 이 표 하나가 §4의 ADR 형식을 실전에 옮긴 견본이다. Context에서 "도메인이 아직 불명확하고 팀이 작다"고 긴장 관계를 적고, Decision에서 "하나로 배포하되 모듈 경계는 엄격히"라고 구조를 고르고, Consequences에서 "경계 실수가 리팩터링으로 끝나는 게 싸진 점(X), 단일 기능을 따로 확장 못 하는 게 비싸진 점(Y)"을 둘 다 적는다. Y를 빠뜨리지 않는 것이 핵심이다.

### 7.3 The default this implies (a forward bet)

척추는 이미 agent의 default topology를 예측하고, 코스의 나머지가 그것을 입증한다(earn): **clean bounded context를 가진 modular monolith**, granularity disintegrator(세분화를 분해하는 힘)가 명확히 이길 때만 service를 추출한다([[fowler-monolith-first]], [[richards-ford-hard-parts]]). 그 이유는 §1.2의 순수한 reversibility 산술이다 — modular monolith 안에서 잘못된 context boundary는 리팩터링이지만; service에 걸치면 그것은 마이그레이션이고, 당신이 여전히 도메인을 배우고 있는 어린 agent에게는, 모든 경계 실수가 싸게 유지되기를 원한다. 그리고 [[insights]]에 따르면, 코스 전체를 이 시스템에 엮는 through-line은 이미 정해져 있다: 모든 외부 SaaS API 응답은 *outside data*(외부 데이터)다 — immutable, versioned, 어쩌면-오래된 snapshot이지, 결코 authoritative live state가 아니다 — agent의 clean inside model은 그것을 팔 길이만큼 떨어뜨려 두어야 한다(ch-06). 챕터 1의 일은 단지 당신에게 어휘 — *expensive-to-reverse, trade-off, critical characteristics, ADR, C4* — 를 주어, 그 모든 베팅을 정직하게 만드는 것뿐이었다.

> 💡 **쉬운 설명:** "granularity disintegrator"란 한 덩어리를 더 잘게 쪼개도록 미는 힘(예: 한 기능만 따로 확장해야 하거나, 별도 팀이 독립 배포를 원하는 경우)이다(ch-04에서 정의). 이 힘이 명확히 이길 때만 service를 떼어내라는 뜻이다. "outside data를 팔 길이만큼 떨어뜨린다(at arm's length)"는, 외부 SaaS가 준 데이터를 절대 우리 시스템의 진실로 곧장 믿지 말고, 버전이 찍힌 오래될 수 있는 사본으로만 다루라는 규율이다(ch-06).

---

## Where This Goes

척추가 설치되었다: architecture는 되돌리기 비싼 집합이다; 모든 pattern은 가격 매겨진 베팅이다; 당신은 critical few characteristics로 고르고, *왜*를 ADR에 기록하며, 한 번에 하나의 C4 레벨로 그린다. 모든 것 중 가장 비싼 결정 — 따라서 코스가 가장 먼저 공격하는 것 — 은 **경계를 어디에 그릴 것인가(where to draw the boundaries)**이다.

Ch-02는 그 결정을 모델링 측면에서 다룬다: Eric Evans의 strategic DDD(전략적 DDD), "total unification of the domain model for a large system will not be feasible"(대규모 시스템의 도메인 모델을 완전히 통일하는 것은 실현 가능하지 않을 것이다)인 이유, 그리고 **bounded context**가 *ubiquitous language*(보편 언어, 유비쿼터스 언어)가 바뀌는 곳에 — "Customer"가 Sales에서는 파이프라인 lead를 의미하지만 Support에서는 티켓을 가진 계정을 의미하는 곳에 — 어떻게 그려지는지. 그것은 또한 다음 신화 — DDD가 microservices를 요구한다는 것 — 를 죽이는데, bounded context가 이 챕터가 방금 Lina TMR이 시작해야 한다고 주장한 바로 그 modular monolith 안의 module로서 완벽하게 잘 살 수 있는 *모델링(modeling)* 경계임을 보여줌으로써 그렇게 한다.
