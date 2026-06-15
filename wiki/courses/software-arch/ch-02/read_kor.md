<!-- chapter: ch-02
     track: boundaries
     kind: content
     title: Boundaries by Language: Strategic DDD and Bounded Contexts
     deps: [[ch-01]]
     sources: [[ddd-bounded-context]], [[decompose-by-business-capability]], [[conway-team-topologies]], [[fowler-monolith-first]], [[richards-ford-fundamentals]], [[COLLECTION-PLAN]], [[insights]]
-->

# 02장 — Boundaries by Language: Strategic DDD and Bounded Contexts

> **핵심 통찰.** boundary(경계)는 시스템에서 되돌리기에 가장 비싼 단 하나의 결정이다. 그래서 당신이 찾을 수 있는 가장 *느리게 변하는* 구조 위에 그것을 놓아야 한다 — 그리고 그 구조는 바로 language(언어)다. "Customer" 같은 단어가 같은 의미를 갖지 않게 되는 지점에서, 당신이 거기에 선을 그었든 아니든 model은 이미 갈라져 있다. Domain-Driven Design(도메인 주도 설계)의 strategic(전략적) 절반은 그 언어적 fault line(단층선)을 *먼저* 찾아내는 규율이다 — 어떤 deployment, framework, service 결정보다도 먼저 — 왜냐하면 언어 위에 그은 boundary는 당신이 commit(확정 투자)할 수 있는 boundary지만, 더 빨리 변하는 무언가(org chart(조직도), database schema, 현재의 tech stack) 위에 그은 boundary는 매일 대가를 치르는 model rot(모델 부패)이기 때문이다.

> 💡 **쉬운 설명:** "되돌리기 비싼 결정일수록 잘 안 변하는 땅 위에 지어라"가 핵심이다. 집을 모래 위에 지으면 매일 무너지고, 암반 위에 지으면 오래 간다. 여기서 암반에 해당하는 것이 "언어" — 즉 사업이 실제로 쓰는 단어들의 의미 — 이고, 모래에 해당하는 것이 조직도/DB스키마/현재 기술스택처럼 자주 바뀌는 것들이다.

> **가이드라인.** service를 찾기 전에 Bounded Context(경계 지어진 맥락)를 먼저 찾아라. 도메인을 실제로 말하는 사람들과 함께 도메인을 걸어 다니며, 한 noun(명사)이 의미를 바꾸는 seam(이음새)에 귀를 기울여라 — 그 polysemy(다의성)가 model을 분할하라는 신호다. 각 context 안에서는 모순 없이 하나의 Ubiquitous Language(보편 언어)를 강제하라; context들 사이에서는 그 관계를 명시적으로 만들고(Anticorruption Layer(반부패 계층), Shared Kernel(공유 커널), Conformist(순응자) 매핑), 그 선택을 배선상의 세부사항이 아니라 조직적 commitment로 취급하라. 이 모든 것을 modeling altitude(모델링 고도)에서 하라: context는 먼저 module이고, 나중에 — 특정한 압력이 distributed tax(분산 세금)를 정당화할 때에만 — service다.

---

## 1. Boundaries Are the Irreversible Decision (Why This Chapter Comes Second)

[[ch-01]]은 척추를 심었다: architecture는 되돌리기 비싼 결정들의 집합이며, First Law(제1법칙)는 모든 pattern을 하나의 bet(베팅)으로 가격 매기도록 강제한다([[richards-ford-fundamentals]], [[insights]]). 이 챕터는 그 척추를 *첫 번째* 구체적 architectural artifact — boundary — 위에서 현금화하며, 한 가지 이유로 의도적으로 두 번째 챕터다: 이 코스의 모든 bet 중에서, boundary는 reversal cost(되돌림 비용)가 가장 높은 것이다.

> 💡 **쉬운 설명:** "왜 하필 두 번째 챕터냐"에 대한 답이다. 가장 비싼 결정을 가장 먼저(1장의 원리 다음으로) 다루는 이유는, 이걸 잘못 그으면 나중에 고치는 비용이 다른 어떤 실수보다 크기 때문이다. 비싼 것부터 신중하게.

[[richards-ford-fundamentals]]에 인용된, 이 코스의 조직화 정의는 그 판돈을 명시한다:

> "Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet." — Richards & Ford (First Law, quoted as commonly published; book thesis, [[richards-ford-fundamentals]])

그렇다면 boundary가 만드는 trade-off는 무엇인가? boundary는 그 한쪽 면 안에 머무는 모든 변경을 *싸게 유지한다* — 한 context의 내부를 다시 쓰고, storage를 바꾸고, 객체를 재구조화해도 다른 쪽은 아무것도 알아채지 못한다. boundary는 그것을 가로지르는 모든 변경을 *비싸게 만든다* — 일단 두 context가 서로의 model에 의존하면, 그들 사이에서 행동을 옮기는 것은 더 이상 refactor가 아니다. Fowler는 [[fowler-monolith-first]]에서 이 비대칭을 정확하게 진술한다: 단일 deployable 안에서 나쁜 boundary는 refactor지만, 일단 그것을 분산하고 나면 "refactoring functionality between services is much harder than it is in a monolith." service들에 걸쳐서는 그것은 *migration*이다.

> 💡 **쉬운 설명:** boundary의 trade-off를 한 문장으로 요약하면 "안쪽 변경은 공짜에 가깝게, 가로지르는 변경은 매우 비싸게"다. 그리고 핵심은 비대칭이다 — 한 프로세스 안(monolith)에서는 경계를 잘못 그어도 그냥 코드 옮기기(refactor)지만, 서비스로 쪼개 놓은 뒤에는 같은 일이 데이터 이전을 동반한 대공사(migration)가 된다. 그래서 미리 service로 쪼개기 전에 경계부터 제대로 그어야 한다.

그 비대칭이 strategic DDD가 존재하는 이유 전부다. 만약 boundary를 옮기는 것이 쌌다면, 그것을 배치하기 위한 규율은 필요 없을 것이다 — 그냥 추측하고 조정하면 될 것이다. 그것들은 싸지 않으므로, *3년 뒤에도 여전히 올바른 선일* 선을 찾는 원칙적인 방법이 필요하다. DDD의 답: 그것을 language 위에 그어라, 왜냐하면 language는 건물 안에서 가장 느린 것이기 때문이다.

### 1.1 The slowest-changing-structure principle

이 원칙은 boundaries 단계 전체를 관통하며, decomposition 발췌([[decompose-by-business-capability]])에서 무뚝뚝하게 진술된다:

> "Stable architecture since the business capabilities are relatively stable." — Chris Richardson

[[decompose-by-business-capability]]는 그것을 일반화한다: "Org structures churn, tech churns, screens churn — what the business fundamentally does changes slowly. Cut there and your expensive-to-reverse decision lands on solid ground." DDD는 business-architecture 쪽이 아니라 modeling 쪽에서 같은 seam에 도달한다. 둘은 두 가지 어휘로 묘사된 같은 움직임이다 — 이 챕터와 ch-04가 둘 다 의지하는 지점이다.

> 💡 **쉬운 설명:** 같은 결론("천천히 변하는 곳을 잘라라")에 두 갈래 길로 도달한다는 뜻이다. 한 갈래는 "사업이 하는 일(business capability)"에서 출발하는 길이고, 다른 갈래는 DDD처럼 "사람들이 쓰는 말(model/language)"에서 출발하는 길이다. 출발점만 다를 뿐 도착하는 이음새는 같다.

| Candidate structure to cut on | How fast it changes | Boundary quality |
|---|---|---|
| Current UI / screens | 릴리스마다 | 최악 — 끊임없이 다시 잘라야 함 |
| Tech stack / framework | 몇 년마다 | 나쁨 — boundary를 vendor에 대한 bet에 결합시킴 |
| Org chart | 재조직은 매년 일어남 | 단독으로는 위험 — 하지만 Conway는 그것이 어차피 architecture를 *형성할* 것임을 의미함 (§5) |
| Database schema / technical layer | 모든 feature마다 | 최악의 종류 — [[distributed-monolith]]를 보장함 |
| **Business capability / ubiquitous language** | **느리게 — 사업의 정체성** | **최선 — commit할 가치가 있는 boundary** |

bet으로서 진술된 bet: **boundary를 language 위에 그으면 intra-context 변경의 비용이 거의 0으로 유지되는데, 그 대가는 language가 실제로 깨지는 곳을 찾기 위해 느리고 비싼 domain conversation(도메인 대화)을 미리 수행하도록 강제당하는 것이다.** 지금 분석 비용을 치러 영원히 migration 비용을 피한다.

> 💡 **쉬운 설명:** 공짜는 없다는 1장의 정신이 여기도 적용된다. 언어 위에 경계를 그으면 나중이 편하지만, 그 대신 "이 단어가 실제로 어디서 의미가 갈라지는가"를 도메인 전문가들과 오래 대화하며 찾아야 하는 선불 비용이 든다. 코드 분석 도구로는 못 찾는다 — 사람 머릿속에 있는 지식이라서다.

---

## 2. Why One Unified Model Fails

strategic DDD의 시작 움직임은 하나의 거부다. Evans는 전체 도메인을 포괄하는 하나의 model을 만드는 것을 거부한다. Fowler가 [[ddd-bounded-context]]에서 인용한 그 thesis:

> "Total unification of the domain model for a large system will not be feasible or cost-effective." — Eric Evans

(Evans의 *Domain-Driven Design*(2003)은 책이다; 이 문장은 Fowler가 그것을 재현하기 때문에 verbatim quote로 살아남았다. 이 챕터에서 Evans 논증의 나머지는 fetched verbatim이 아니라 attributed thesis(귀속된 주장)로 취급하라 — [[COLLECTION-PLAN]]의 gap-log hedge와 [[ddd-bounded-context]]의 excerpt header를 보라.)

실패의 메커니즘은 기술적이 아니라 언어적이다. Fowler가 [[ddd-bounded-context]]에서 표현하듯, "different groups of people will use subtly different vocabularies in different parts of a large organization," 그리고 그들에 걸쳐 강제된 단일 model은 모순을 축적한다. Sales와 Support와 Billing을 *동시에* 섬겨야 하는 `Customer` 클래스는 조건부 필드, mode flag, 그리고 "if this came from the support side" 분기들을 키워가다가 결국 아무도 그것에 대해 추론할 수 없게 된다. 그 model은 도메인을 unify(통합)하지 않았다; 그것은 도메인의 *모순들*을 하나의 객체로 unify했다.

> 💡 **쉬운 설명:** "하나의 거대한 Customer 클래스로 모든 부서를 다 처리하자"는 시도가 왜 망하는지에 대한 핵심이다. Sales/Support/Billing은 "Customer"라는 같은 단어를 쓰지만 머릿속 의미가 다르다. 한 클래스에 다 욱여넣으면 통합(unify)이 아니라 모순의 집합소가 된다 — 통합한 건 도메인이 아니라 도메인의 충돌들이다.

### 2.1 The failure is a coupling failure in disguise

이것은 [[insights]]의 coupling/cohesion 기둥과 직접 연결된다: god-model(신 객체 모델)은 도메인을 건드리는 모든 팀이 같은 클래스를 건드리기 때문에 최대로 coupled(결합)되어 있다. Support가 요청한 변경이 Sales를 깨뜨릴 수 있다 — deployment dependency를 통해서가 아니라 *semantic*(의미적) dependency를 통해서다. 통합된 model은 cohesive(응집적)해 보이지만(하나의 `Customer`, 얼마나 깔끔한가) 그 반대다 — 그것은 무관한 reason-to-change(변경 이유)들을 한 곳으로 끌어당기는 low-cohesion(저응집) 자석이다. DDD의 boundary는 각 context가 정확히 한 가지를 의미하는 `Customer`를 갖게 함으로써 cohesion을 복원한다.

> 💡 **쉬운 설명:** coupling(결합)은 보통 "A를 배포하면 B도 같이 배포해야 한다" 같은 배포 의존을 떠올리지만, 여기서는 더 미묘한 "의미적 결합"이다. 한 클래스를 여러 팀이 공유하면, 코드를 같이 배포하지 않아도 한 팀의 의미 변경이 다른 팀의 가정을 깨뜨린다. 겉보기엔 깔끔(하나의 Customer)하지만 실은 가장 결합도 높은 구조다.

### 2.2 What model rot actually looks like (worked example)

이 추상은 고개를 끄덕이기 쉽다; 하지만 메커니즘은 frame by frame으로 지켜볼 가치가 있다, 왜냐하면 그것은 점진적이고 각각의 개별 step이 합리적으로 보이기 때문이다. 어린 sales tool을 위한 정직한 단일 `Customer`로 시작하자:

```
class Customer:
    id
    name
    email
    pipeline_stage        # Sales cares about this
```

Support가 출시된다. Support의 `Customer`는 ticket과 SLA tier를 가진 account다 — 그러나 "이미 `Customer` 클래스가 있으므로," model을 그리는 대신 필드가 추가된다:

```
class Customer:
    id
    name
    email
    pipeline_stage        # meaningless for a Support account
    sla_tier              # meaningless for a Sales lead
    open_ticket_count     # meaningless for a Sales lead
```

이제 Billing이 도착하고, Billing의 `Customer`는 tax ID와 payment method를 가진 법적 실체다. 같은 움직임:

```
class Customer:
    ...
    pipeline_stage        # null for Billing
    sla_tier              # null for Sales
    tax_id                # null until they pay
    payment_method        # null for an unqualified lead
    # and: is_active means "in pipeline" to Sales,
    #      "has an open ticket" to Support,
    #      "has a valid card" to Billing
```

부패가 이제 눈에 보인다. 객체의 어떤 주어진 사용에서든 필드의 절반은 null이다. `is_active`가 결정타다: 세 팀이 하나의 boolean에 양립 불가능한 세 가지 의미를 부여했으므로, 모든 read는 out-of-band(별도의) "어떤 의미를 말한 거야?"를 필요로 한다. invariant(불변식)도 사라졌다 — Sales는 `pipeline_stage`가 required이길 원하고 Billing은 그것이 forbidden이길 원하므로, 클래스는 *둘 중 어느 것도* 강제할 수 없다. 이것은 Evans가 예측한 contradiction-accumulation(모순 축적)이 구체화된 것이다: 객체는 도메인을 unify하지 않았고, 모든 팀의 `Customer` 개념을 흡수했으며 그것들 중 어느 것에 대해서도 올바를 수 있는 능력을 잃었다. DDD의 해결책은 "필드를 더 영리하게 추가하라"가 아니다 — 세 개의 Ubiquitous Language가 한 클래스에서 충돌했음을 인식하고 그것들을 세 개의 context로 분할하는 것이다. 그러면 각각은 작고, 완전히 required이며, 올바른 `Customer`를 갖게 되고, 그들이 만나는 곳에는 명시적인 translation(번역)이 있다(§4).

> 💡 **쉬운 설명:** invariant(불변식)는 "이 객체가 항상 지켜야 하는 규칙"이다(예: "Sales의 Customer는 pipeline_stage가 반드시 있어야 한다"). 한 클래스가 세 팀을 다 섬기면 Sales는 "필수", Billing은 "있으면 안 됨"을 동시에 요구하니, 클래스는 어느 규칙도 강제할 수 없게 된다. 규칙을 강제하지 못하는 모델은 사실상 검증 기능이 죽은 것이다. 그래서 답은 영리한 필드 추가가 아니라 "셋으로 쪼개기"다.

---

## 3. The Bounded Context and the Ubiquitous Language

Bounded Context는 망할 운명의 통합 model 대신 DDD가 제공하는 단위다. Fowler의 정의, [[ddd-bounded-context]]에 verbatim:

> "A Bounded Context is a central pattern in Domain-Driven Design." — Fowler

> "DDD divides up a large system into Bounded Contexts, each of which can have a unified model." — Fowler

이 전환은 미묘하면서도 전면적이다: unification은 사라지지 않고, *축소된다*. 전체 시스템을 위한 하나의 model을 포기하고 *context당* 하나의 엄격하게 일관된 model을 얻는다. boundary 안에서, model은 strict하고, complete하며, contradiction-free(모순 없음)일 수 있다 — 왜냐하면 그것은 도메인의 한 조각에 대해서만 올바르면 되기 때문이다.

> 💡 **쉬운 설명:** "통합을 포기하는 게 아니라 통합의 범위를 좁힌다"가 핵심이다. 전체 시스템 하나의 완벽한 모델은 불가능하지만, "Sales context 안에서만 일관된 모델"은 충분히 가능하다. 책임 범위를 좁히면 그 안에서는 빈틈없이 엄격해질 수 있다.

한 context를 묶어주는 것은 그것의 **Ubiquitous Language**다: developer와 domain expert가 공유하는 단일 어휘로, 모든 용어가 정확히 하나의 의미를 갖는다. [[ddd-bounded-context]]는 그 관계를 날카롭게 표현한다 — "the model *is* a shared language between developers and domain experts." 그 표현은 load-bearing(하중을 지탱하는, 핵심적인)이다. language는 model에 *관한* 문서가 아니다; language와 model은 두 자리에서 본 같은 artifact다. domain expert가 "qualified lead"라고 말하고 developer의 클래스 이름이 `Prospect`일 때, 그 두 단어 사이의 간극은 미래의 버그다 — 누군가 결국 하나를 다른 것으로 translate할 것이고 잘못 translate할 것이다. Ubiquitous Language 규율은 코드의 noun과 verb가 expert의 noun과 verb *그 자체*이며, 그 사이에 glossary(용어집)가 없다고 주장함으로써 그것을 미연에 방지한다.

> 💡 **쉬운 설명:** "model = language" 라는 등식이 핵심이다. 도메인 전문가가 "qualified lead"라 부르는 것을 코드에서 `Prospect`라고 다르게 이름 붙이면, 언젠가 누군가 둘을 연결하다가 틀린다. 그래서 코드의 클래스/메서드 이름을 도메인 전문가가 실제로 쓰는 단어와 글자 그대로 일치시키라는 것이다 — 중간 번역표(glossary)가 끼어들 틈을 없애서 버그의 씨앗을 제거한다.

이것이 또한 분할 신호가 기술적이 아니라 언어적인 이유다, [[ddd-bounded-context]]에 인용됨:

> "You need a different model when the language changes." — Fowler

architect가 실제로 행동하는 두 가지 결과가 따라온다. 첫째, boundary는 diagram에서 도출되는 것이 아니라 *경청에 의해 발견된다* — domain expert들이 한 단어의 의미를 두고 논쟁하는 방에서 그것을 찾는다. 둘째, context *안에서* "one term, one meaning"을 강제하는 것이 model을 strict하게 유지하게 해준다: 방어해야 할 `Customer`의 두 번째 의미가 없으므로, 클래스는 nullable mode-flag 대신 required field와 진짜 invariant를 운반할 수 있다. 안쪽의 엄격함은 바깥쪽의 boundary에 의해 *구매된다*.

> 💡 **쉬운 설명:** 마지막 문장 "안의 엄격함은 밖의 경계가 사준 것"이 인과의 핵심이다. 한 context 안에서 Customer의 의미가 단 하나로 고정되어 있기 때문에, 그 안에서는 "이 필드는 무조건 필수" 같은 강한 규칙을 걸 수 있다. 경계를 그어 바깥의 다른 의미들을 차단했기 때문에 안에서 빡빡해질 자유가 생긴 것이다.

### 3.1 The polysemy signal (the canonical teaching case)

실용적인 detector는 **polysemy**다 — 하나의 단어가 두 의미를 운반하는 것. [[ddd-bounded-context]]는 교과서적 위반자들 `Customer`와 `Product`, 그리고 "meter"가 "subtly different things to different parts of the organization"을 의미했던 Fowler 자신의 utility 예시를 사용한다. 그 패턴:

| Word | In context A | In context B | Verdict |
|---|---|---|---|
| `Customer` | Sales: pipeline stage와 probability-to-close를 가진 *lead* | Support: open ticket과 SLA tier를 가진 *account* | 두 model, 두 `Customer` 타입, 그 사이의 명시적 translation |
| `Product` | Catalog: description과 image를 가진 marketing SKU | Fulfillment: weight와 warehouse bin을 가진 물리적 item | 동일 — 하나의 클래스로 강제하지 말 것 |

같은 noun이 두 개의 다른 속성 집합과 규칙으로 사용되는 것을 들으면, context boundary를 찾은 것이다. 규율은 *그것들을 하나의 클래스로 강제하기를 멈추고* 대신 그들 사이에 의도적인 translation을 둔 두 개의 model을 그리는 것이다. 그 translation이 §4의 주제다.

> 💡 **쉬운 설명:** polysemy(다의성)는 경계를 찾는 가장 시끄러운 신호다. "Customer라는 같은 단어를 두 팀이 다른 속성/규칙으로 쓴다"면 거기가 곧 경계선이다. 단어가 갈라지는 곳에 모델도 갈라야 한다 — 억지로 한 클래스에 합치지 말고 두 모델 + 번역으로.

### 3.2 Strategic vs tactical — where this chapter sits

[[ddd-bounded-context]]는 스스로를 신중하게 범위 짓는다: 이것은 *strategic* DDD다 — large-scale boundary와 integration 결정. context *안의* building block(entity, value object, aggregate, domain event)들은 *tactical* DDD이며 ch-03에 속한다([[ddd-aggregates-tactical]], 거기서 미리보기됨). 이 구분이 중요한 이유는, 그것들을 혼동하는 것이 팀들이 단 하나의 context boundary도 그리지 않으면서 aggregate를 여기저기 뿌리며 "DDD를 한다"고 착각하는 방식이기 때문이다 — 비싼 부분(boundary)을 틀리면서 싼 부분(object taxonomy)을 만지작거린다.

> 💡 **쉬운 설명:** DDD는 두 층으로 나뉜다. strategic(전략) = 큰 경계 긋기(이 챕터), tactical(전술) = 경계 안에서 entity/aggregate 같은 building block 만들기(다음 챕터). 흔한 실패는 전술만 열심히 하면서(aggregate 남발) 정작 비싼 전략(경계)을 안 그리는 것이다 — 비싼 걸 틀리고 싼 걸 다듬는 셈.

---

## 4. Context Mapping: The Relationship Between Boundaries Is Itself a Decision

boundary를 그리는 것은 작업의 절반이다; 나머지 절반은 양쪽의 context들이 어떻게 *관계 맺는지* 결정하는 것이다. 두 context는 언젠가 항상 정보를 교환해야 한다 — Sales는 closed deal을 Fulfillment에 넘기고, Support는 Sales가 소유한 account data를 읽는다. 당신이 선택하는 관계는 배선상의 세부사항이 아니라 조직적 결과를 가진 strategic decision이다. [[ddd-bounded-context]]는 Vernon의 *DDD Distilled*의 context-mapping pattern 카탈로그를 명명한다(책에서 온 attributed thesis, verbatim 아님):

> **Interactive companion:** [`figures/context-map-explorer.html`](figures/context-map-explorer.html) — 각 bounded context를 클릭해 같은 단어 `Customer`가 각각에서 어떻게 다른 model을 운반하는지 보고(§3.1의 polysemy), 그런 다음 두 context와 하나의 mapping pattern을 선택해 그 mapping이 만드는 bet을 보라: 무엇을 싸게 유지하고 무엇을 비싸게 만드는가. 그것은 아래 표의 공간적 버전이다.


| Mapping pattern | What it means | When you choose it |
|---|---|---|
| **Partnership** | 두 context가 함께 성공하거나 함께 실패한다; 팀들이 긴밀하게 조율한다 | 상호 의존, 정렬된 목표, coordination cost를 기꺼이 치를 의향 |
| **Shared Kernel** | 두 context가 모두 의존하는 작은 공유 model | 겹침이 작고, 안정적이며, 복제하기보다 공유하는 것이 더 쌈 — 하지만 이제 모든 변경에 두 팀이 다 필요 |
| **Customer-Supplier** | downstream context의 요구가 upstream의 우선순위에 영향을 줌 | upstream이 downstream을 고객으로 섬길 의향이 있음 |
| **Conformist** | downstream이 upstream model을 있는 그대로 그냥 채택함 | upstream에 대한 leverage(영향력)가 없고(예: vendor API) translation할 가치가 없음 |
| **Anticorruption Layer (ACL)** | 다른 context의 model을 당신의 것으로 변환해 그것이 새어 들어올 수 없게 하는 translation layer | integrate해야 하지만 외부의 혹은 지저분한 model이 당신의 깨끗한 것을 오염시키게 두기를 거부함 |
| **Open Host Service** | 한 context가 많은 consumer를 위해 잘 정의된 protocol을 publish함 | 당신이 많은 것의 upstream이고 N개의 맞춤 인터페이스 대신 하나의 안정적 인터페이스를 원함 |
| **Published Language** | 공유되고 잘 문서화된 교환 포맷 | 여러 context가 공통의 lingua franca(공통어)를 필요로 함(예: 정의된 event schema) |

### 4.1 The Anticorruption Layer is the load-bearing one for an agent

이것들 중, **Anticorruption Layer**는 이 학습자가 가장 많이 사용할 pattern이며, 그것의 bet을 정확히 진술할 가치가 있다. [[ddd-bounded-context]]는 그것을 "the other context's model so it can't leak into yours"를 translate하는 layer로 정의한다. 그 bet: **ACL은 당신의 core model을 진화시키기 싸게 유지한다 — 그것의 개념들은 결코 외부 schema에 의해 오염되지 않는다 — 그 대가는 boundary에서 translation code를 작성하고 유지하는 것이다.** vendor의 data shape이 결코 당신의 domain logic에 닿을 수 없도록 mapper/DTO 비용을 치른다. (이것은 ch-03이 [[martin-clean-arch]]를 통해 tactical하게 만드는 inward-dependency rule의 strategic-altitude 버전이다.)

> 💡 **쉬운 설명:** ACL을 비유하면 "통역사 겸 검역소"다. 외부 시스템(예: Salesforce)이 보내는 데이터를 내 도메인 언어로 번역하고, 이상한 형태가 내 핵심 모델로 새어 들어오지 못하게 막는다. 비용은 번역 코드(mapper/DTO)를 직접 짜고 유지하는 것 — 하지만 그 대가로 외부 스키마가 바뀌어도 내 핵심은 무사하다.

### 4.2 Mapping choices are org choices

[[ddd-bounded-context]]는 "choosing the relationship is a strategic decision with org consequences"임을 표시하며 [[conway-team-topologies]]로 cross-link한다 — 그것이 §5다. Shared Kernel은 이제 두 팀이 model level에서 coupled되어 그것에 대한 모든 변경을 조율해야 함을 의미한다; Conformist 관계는 한 팀이 다른 팀의 model에 대한 영구적 종속을 받아들였음을 의미한다. 이것들은 diagram을 입은 people decision(사람에 대한 결정)이다.

> 💡 **쉬운 설명:** mapping pattern을 고르는 건 기술 선택처럼 보이지만 실은 조직 결정이다. Shared Kernel을 고르면 두 팀이 영원히 같이 회의해야 하고, Conformist를 고르면 우리 팀이 상대 팀 모델에 영구히 끌려다니기로 서명한 것이다. 다이어그램 화살표 하나가 곧 사람들의 협업 관계를 규정한다.

---

## 5. The Boundary Is Socio-Technical: Conway's Law

화이트보드 위에서 완벽한 언어적 boundary를 찾고도 여전히 시스템이 그것을 존중하기를 거부할 수 있다 — 왜냐하면 당신이 초대하든 아니든 *조직*이 한 표를 갖기 때문이다. Conway의 원래 1968년 thesis, Fowler를 통해 [[conway-team-topologies]]에 verbatim:

> "Any organization that designs a system (defined broadly) will produce a design whose structure is a copy of the organization's communication structure." — Melvin Conway, 1968

메커니즘, [[conway-team-topologies]]의 Fowler에 따르면: "software coupling is enabled and encouraged by human communication," 그리고 그 따름정리로 "the modular decomposition of a system and the decomposition of the development organization must be done together." 한 팀의 cognitive unit(인지 단위)을 한가운데서 쪼개거나, 두 팀을 하나의 model로 융합하는 bounded context는 매일 저항받을 것이다 — communication structure가 조용히 코드를 자신의 shape으로 되돌려 구부릴 것이다.

> 💡 **쉬운 설명:** Conway's Law는 "시스템 구조는 그것을 만든 조직의 소통 구조를 닮는다"는 관찰이다. 즉 세 팀이 만들면 시스템도 어쩐지 세 덩어리가 된다. 그래서 아무리 화이트보드에서 완벽한 경계를 그어도, 조직의 실제 소통 구조와 어긋나면 코드가 슬그머니 조직 모양으로 되돌아간다.

### 5.1 Three responses and the lever

[[conway-team-topologies]]는 Fowler의 세 가지 대응을 펼친다:

1. **Ignore** 그것을 — 그래도 여전히 일어나고, 이제는 우연에 의해 일어난다.
2. **Accept** 그것을 — architecture를 당신이 실제로 가진 communication path에 정렬하라.
3. **Inverse Conway Maneuver(역 Conway 책략)** — 당신이 원하는 architecture를 *유도하기* 위해 의도적으로 팀을 재구조화하라; "particularly effective with microservices organized around business capabilities."

이것이 management 챕터가 아니라 *boundaries* 챕터에 속하는 이유: option 3은 architectural lever(지렛대)다. 올바른 context를 찾았지만 조직이 그것들과 맞지 않으면, 조직이 boundary를 corrupt하게 두는 대신 boundary에 맞추도록 조직을 옮길 수 있다. 한 팀이 실제로 context를 소유할 수 있는지를 결정하는 제약은 **cognitive load(인지 부하)**다 — [[conway-team-topologies]]는 "cognitive load, not headcount, is the real constraint on a team's boundary"임을 강조한다. 팀이 머릿속에 담을 수 없는 context는 종이 위에서 아무리 깨끗해 보여도 부패할 것이다.

> 💡 **쉬운 설명:** 핵심은 option 3이다 — Conway's Law를 거꾸로 이용해, 원하는 아키텍처를 얻으려고 조직을 먼저 그 모양으로 바꾸는 것(Inverse Conway Maneuver). 그리고 팀이 한 context를 감당할 수 있느냐의 기준은 "사람 수"가 아니라 "그 복잡도를 머릿속에 담을 수 있느냐(cognitive load)"다. 머리에 안 들어오는 경계는 결국 무너진다.

trade-off 프레이밍: **팀을 context에 정렬하는 것(Inverse Conway Maneuver)은 각 context를 고립 상태에서 진화시키기 싸게 유지하지만, 그 대가는 reorg(재조직)다 — 그리고 reorg는 그 자체로 비싸고 파괴적이다.** 그것은 공짜 승리가 아니라 진짜 bet이다.

---

## 6. Myth Killed: "DDD Requires Microservices"

이것은 [[COLLECTION-PLAN]]에서 ch-02에 배정된 doc-vs-reality 신화이며, primary source는 명백하다.

| Popular narrative | What the primary source actually says | Resolve in |
|---|---|---|
| "DDD requires microservices." | **거짓.** DDD는 modeling discipline이다; bounded context는 modular monolith 안의 module일 수 있다. 어떤 DDD source도 그것을 deployment topology에 묶지 않는다. | [[ddd-bounded-context]] |

[[ddd-bounded-context]]는 그것을 직접 진술한다: "DDD is a modeling discipline; it applies equally inside a modular monolith. Bounded contexts can be modules in one deployable." Evans, Fowler의 bliki entry, 혹은 Vernon의 *DDD Distilled* 중 어느 것도 bounded context를 별도로 배포 가능한 단위에 묶지 않는다. context는 *model 안의* boundary다; 그 boundary가 한 process 안의 module/namespace에 의해 강제되든 두 service 사이의 network hop에 의해 강제되든은 *별개의, 나중의, topology* 결정이다 — ch-04의 주제다.

> 💡 **쉬운 설명:** 가장 흔한 오해를 깨는 부분이다. "DDD를 하려면 microservice로 가야 한다"는 거짓이다. bounded context는 model 차원의 경계일 뿐, 그것을 한 프로세스 안의 module로 강제할지 별도 service로 강제할지는 완전히 다른 — 그리고 나중의 — 결정이다.

### 6.1 Why the conflation is dangerous

이 신화는 무해하지 않다. "DDD requires microservices"를 믿으면, 당신이 bounded context를 찾는 (올바르고, 싸고, 가치 높은) 작업을 하는 순간, 그 각각을 service로 바꾸는 distribution tax(분산 세금)도 *반드시* 치러야 한다고 결론짓는다. 그것이 바로 [[fowler-monolith-first]]가 경고하는 premature-decomposition(조급한 분해) trap이다:

> "Almost all the cases where I've heard of a system that was built as a microservice system from scratch, it has ended up in serious trouble." — Fowler

> "Even experienced architects working in familiar domains have great difficulty getting boundaries right at the beginning." — Fowler

올바른 순서는 두 결정을 분리한다. **지금 context를 찾아라(싸고, 되돌릴 수 있음 — 그것들은 그저 module boundary다); topology는 나중에 결정하라(비싸고, 어떤 seam이 진짜인지 사용이 보여준 후에).** [[fowler-monolith-first]]는 그 무대(staging ground)를 modular monolith라고 부른다: "the single deployable unit but enforced strict internal module boundaries… one schema/namespace per module, communication via in-process interfaces, no reaching into another module's tables." bounded context가 바로 그 module들이다. 신화는 reversal cost가 크게 다른 두 결정을 하나로 무너뜨리고, 그래서 팀들이 그것을 잘 치를 정보를 갖기 전에 되돌릴 수 없는 비용을 치르게 만든다.

> 💡 **쉬운 설명:** 신화가 위험한 이유는 "싼 결정(경계 찾기)"과 "비싼 결정(service로 쪼개기)"을 한 묶음으로 만들어 버리기 때문이다. 경계를 찾자마자 곧장 microservice로 가면, 어디가 진짜 이음새인지 검증되기도 전에 되돌릴 수 없는 분산 비용을 치른다. 올바른 순서는 "지금은 module로 경계만, 분산은 나중에"다.

### 6.2 The symmetric error: too many contexts

First Law([[richards-ford-fundamentals]])는 "draw contexts"를 공짜 재화로 취급하는 것을 금지하므로, *반대* 쪽의 비용도 명명하라. 분할은 무비용이 아니다: 당신이 그리는 모든 boundary는 이제 당신이 빚진 translation이다(§4). 양쪽에서 하나의 Ubiquitous Language를 공유하는 것으로 드러난 두 context — `Customer`가 양쪽에서 정말로 같은 것을 의미했던 곳 — 는 헛되이 분할된 것이고, 당신은 이제 아무 isolation도 사주지 않는 mapper code, 중복된 개념, cross-context call을 유지한다. 이것이 over-decomposition(과잉 분해) failure이며, god-model의 거울상이다: 하나의 객체가 모든 의미를 흡수하는 대신, 끊임없는 잡담으로 당신이 제거하려 했던 coupling을 재구성하는 열 개의 anemic context(빈혈성 맥락)를 갖게 된다.

> 💡 **쉬운 설명:** 반대 방향의 실수도 있다. 안 갈라도 될 곳을 갈라 놓으면, 의미는 같은데 둘로 쪼개진 두 context가 서로 끊임없이 통신하면서 오히려 결합을 되살린다. 너무 합치면 god-model, 너무 쪼개면 anemic context — 둘 다 결합이 돌아온다는 점에서 거울상이다.

detector는 같은 것을, 역방향으로 돌린 것이다: 후보 seam을 가로질러 language가 *변하지 않으면* — domain expert가 양쪽에서 그 단어를 동일하게 사용하고 어떤 invariant도 다르지 않으면 — 거기에는 boundary가 없고, 임의의 선만 있다. 규율은 language가 깨지는 곳에서 *정확히* 분할하고 다른 어디에서도 하지 않는 것이다. 이것이 §7의 checklist가 polysemy로 시작해 "draw it inside a modular monolith first"로 끝나는 이유다([[fowler-monolith-first]]): 단지 module인 context는 과분할했을 경우 다시 합치기 싸지만, service가 된 context는 되돌려야 할 migration이다. context를 module로 유지하면 boundary error의 *양쪽 방향* 모두를 교정하기 싸게 유지한다.

> 💡 **쉬운 설명:** 경계를 찾는 detector는 양방향으로 쓴다. 단어 의미가 갈라지면 → 쪼개라. 단어 의미가 같으면 → 거긴 경계가 아니다, 합쳐 둬라. 그리고 module로만 유지하면, 너무 쪼갰을 때 합치기도 너무 합쳤을 때 쪼개기도 둘 다 싸다 — service로 만들어 버리면 양쪽 다 대공사가 된다.

---

## 7. Pricing the Bet and Finding the Seams in Practice

챕터 전체는 outline의 용어로 진술된 하나의 가격 매겨진 bet으로 환원된다: **가장 느리게 변하는 구조(language/domain) 위에 그어진 bounded context는 commit할 가치가 있는 boundary다; 잘못된 것은 매일 대가를 치르는 model rot다.**

- **싸게 유지하는 것:** 한 context에 국한된 모든 변경 — 그 내부 model, storage, object는 자유롭게 churn할 수 있다; 안쪽의 Ubiquitous Language는 일관되게 유지된다; context에 정렬된 팀은 독립적으로 움직인다.
- **비싸게 만드는 것:** context boundary를 가로질러야 하는 모든 것 — 이제 명시적 mapping이 필요하고(§4), 일단 두 context가 진정으로 서로의 model에 의존하면 그들 사이에서 행동을 옮기는 것은 refactor이기를 멈춘다.
- **선불 비용:** language가 실제로 깨지는 곳을 찾기 위한 느리고, 인간적인 domain conversation. static-analysis 지름길은 없다; polysemy는 코드가 아니라 사람들의 머릿속에 산다.

### 7.1 The practitioner's seam-finding checklist

[[decompose-by-business-capability]](how-to layer를 제공)에서 종합됨:

1. **polysemy를 들어라.** 어디서 하나의 noun이 두 개의 속성 집합 / 두 개의 규칙 집합을 운반하는가? ([[ddd-bounded-context]]) 그것이 가장 시끄러운 신호다.
2. **함께 변하고** cluster 밖으로의 참조가 적은 data와 behavior의 cluster를 찾아라 — high cohesion, low coupling. ([[decompose-by-business-capability]])
3. **business capability에 대조 검증하라.** 후보 context가 "something that a business does in order to generate value"(Richardson, [[decompose-by-business-capability]])에 대응하는가? Capability와 subdomain은 보통 수렴한다.
4. **Conway에 대조 검증하라.** 한 팀이 cognitive load를 초과하지 않고 그것을 소유할 것인가? 한 팀의 communication path를 가로지르는 seam은 저항받을 것이다. ([[conway-team-topologies]])
5. **layer-based split을 거부하라.** "A UI service, a logic service, a data service"는 anti-decomposition이다 — 모든 business 변경이 모든 layer를 건드려 [[distributed-monolith]]를 보장한다. ([[decompose-by-business-capability]])
6. **먼저 modular monolith 안에 그것을 그려라.** 그것을 분산하는 비용을 치르기 전에 seam이 진짜인지 사용이 드러내게 하라. ([[fowler-monolith-first]])

> 💡 **쉬운 설명:** 이 6단계가 실전 사용설명서다. 흐름은 이렇다 — (1) 단어가 갈라지는 곳을 귀로 찾고 → (2~3) 함께 변하는 데이터/행동 묶음이자 사업 가치 단위인지 교차검증하고 → (4) 한 팀이 머릿속에 담을 수 있는지 확인하고 → (5) "UI/로직/데이터"식 수평 분할은 절대 금지(모든 변경이 전 계층을 건드려 distributed monolith가 됨), → (6) 일단 monolith 안 module로만 그어 두고 진짜인지 지켜본다.

---

## 8. Applied to the Sales Agent (Lina TMR)

Lina TMR은 많은 외부 SaaS tool API 위에서 동작하는 LLM agent다 — [[insights]]가 이 코스의 through-line(관통선)으로 미리 명명한 시스템이다. Strategic DDD는 그 architecture가 실제로 시작되는 곳이며, 그 움직임은 deployment에 관한 어떤 것이든 결정하기 *전에* agent의 bounded context를 찾는 것이다.

agent의 도메인 위에 polysemy detector를 돌리면 language가 명백히 깨지는 후보 context들이 떠오른다:

| Candidate context | Ubiquitous Language inside it | The polysemy signal |
|---|---|---|
| **Lead / Pipeline** | `Lead`, `Opportunity`, `Stage`, `probability-to-close` | 여기서 `Customer`는 *pipeline position을 가진 prospect*다 |
| **Conversation** | `Thread`, `Turn`, `Intent`, `Message` | 여기서 `Customer`는 *대화의 반대편에 있는 entity*다 — pipeline 의미가 전혀 없음 |
| **Scheduling** | `Meeting`, `Slot`, `Availability`, `Invite` | 여기서 `Customer`는 *calendar를 가진 attendee*일 뿐, 그 이상은 아님 |
| **CRM-Sync** | `Record`, `FieldMapping`, `SyncState`, `Conflict` | 여기서 `Customer`는 *Salesforce/HubSpot의 foreign row*다 — agent가 소유하지도 않음 |

그 마지막 context가 이 학습자에게 load-bearing한 것이다. CRM-Sync context는 모든 외부 SaaS API response가 시스템에 들어오는 곳이며, [[insights]]는 그 규율을 명명한다: 모든 외부 API response는 **outside data(외부 데이터)**다 — versioned되고, 어쩌면 stale(낡은)한 snapshot이지, 결코 authoritative live state(권위 있는 실시간 상태)가 아니다. 전략적으로, 그것은 **CRM-Sync가 Anticorruption Layer 뒤에 앉아야 함**을 의미한다(§4.1). Salesforce의 `Customer` 개념 — 그 field 이름, picklist 값, 특이성 — 은 boundary에서 *translate되어야* 하며, 결코 Lead/Pipeline이나 Conversation context로 새어 들어가도록 허용되어서는 안 된다. Salesforce schema 변경이 agent의 core domain logic에 닿을 수 있다면, boundary는 실패한 것이다.

> 💡 **쉬운 설명:** 학습자 자신의 sales agent에 이론을 그대로 적용한 부분이다. "Customer"라는 한 단어가 Lead/Conversation/Scheduling/CRM-Sync 네 맥락에서 각각 다른 의미를 갖는다는 것이 곧 네 개 context가 필요하다는 polysemy 증거다. 그중 CRM-Sync가 가장 중요한데, 외부 SaaS(Salesforce 등)의 데이터가 들어오는 관문이기 때문이다. 외부 응답은 "실시간 진실"이 아니라 "낡았을 수도 있는 스냅샷"으로 다루고, ACL로 번역해 내 핵심 모델을 오염에서 보호해야 한다.

agent를 위한 가격 매겨진 bet, 구체적으로: **이 네 context를 지금 module로 정의하는 것은 agent의 core(lead에 대해 어떻게 추론하는지, 대화를 어떻게 수행하는지)를, 현재가 어떤 CRM, calendar API, LLM provider이든 그것과 독립적으로 진화시키기 싸게 유지한다 — 그 비용은 context들 사이에 ACL mapper를 작성하고 네 개 모두에 걸쳐 하나의 뚱뚱한 `Customer` 객체를 공유하려는 유혹에 저항하는 것이다.** 이 챕터가 의도적으로 결정하지 *않는* 것에 주목하라: CRM-Sync가 별도의 service가 될지 여부. §6의 신화 깨기에 의해, 그것은 나중의 topology 결정이다. 지금 CRM-Sync는 context다 — 깨끗한 boundary와 ACL을 가진 module. agent는 modular monolith를 default로 삼고, 특정한 압력이 distributed tax를 정당화할 때에만 service를 추출해야 한다([[fowler-monolith-first]], [[insights]]).

CRM-Sync가 *이* agent에게 특별히 전략적으로 load-bearing한 context이고, 단지 넷 중 동등한 하나가 아닌 이유가 있다. LLM agent의 failure mode는 silent contamination(조용한 오염)이다: Salesforce field의 의미가 prompt나 domain model로 새어 들어가면, agent는 잘못된 개념에 대해 유창하게 추론하고 자신만만하고 틀린 action을 생산할 것이다 — 정확히 학습자가 agent-benchmark 코스에서 지배적이라고 본 "false confidence in incorrect tool calls" failure다. 따라서 CRM-Sync boundary의 ACL은 관료적 간접화가 아니다; 그것은 upstream의 churn(이름 바뀐 picklist, 새로운 required field, vendor API version bump)이 시스템에서 틀리면 비싼 부분에 닿지 못하게 막아주는 것이다. 이것은 동시에 두 후속 챕터의 전략적 씨앗이다: ACL을 구조적 규율로 만드는 inward-dependency rule(ch-03, [[martin-clean-arch]]), 그리고 그 외부 response 각각이 live truth가 아니라 immutable하고, versioned되고, 어쩌면 stale한 snapshot으로 취급되어야 한다고 말하는 inside-vs-outside-data 구분(consistency 단계). 당신이 여기서 그리는 boundary는 그 챕터들이 그 위에 짓는 것이다.

> 💡 **쉬운 설명:** 왜 CRM-Sync가 "그냥 넷 중 하나"가 아니라 가장 중요한지에 대한 인과다. LLM agent는 틀린 개념도 자신만만하게 그럴듯한 행동을 내놓는다(조용한 오염). 만약 Salesforce 필드 의미가 프롬프트나 도메인 모델로 새어 들면, agent는 잘못된 개념 위에서 유창하게 헛소리를 한다 — 이전 벤치마크 코스에서 본 "incorrect tool call에 대한 false confidence"가 바로 이것이다. 그래서 CRM-Sync 경계의 ACL은 형식적 번거로움이 아니라, 외부 변화가 핵심에 닿지 못하게 막는 진짜 방어선이다.

---

## Where This Goes

이 챕터는 *어디서* 잘라야 하는지 — context들 사이의 언어적 seam — 를 찾았고, context가 먼저 module이고 service는 어쩌면-나중임을 확립함으로써 모든 deployment 질문을 미뤘다. Ch-03은 한 zoom 레벨 더 깊이 들어간다: boundary를 그렸으니, 비싼 부분(domain policy)이 싼 부분(vendor, framework, 현재의 LLM API)으로부터 절연된 채로 유지되도록 그 *안의* 코드를 어떻게 구조화하는가? 답은 Dependency Rule(의존성 규칙) — "source code dependencies can only point inwards"([[martin-clean-arch]]) — 이며, primary source들이 노골적으로 진술하는 발견, 즉 Hexagonal, Clean, Onion이 세 이름 아래의 같은 아이디어라는 것이다. §8에서 CRM-Sync boundary에 배치한 Anticorruption Layer는 그 tactical inward-dependency 규율의 strategic preview다.
