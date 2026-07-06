<!-- qa for ch-02 — Boundaries by Language: Strategic DDD and Bounded Contexts
     study index of clarifying questions raised during the Read phase.
     Kernel answers only; full reasoning lives in [[read]]. Append-only across cycles. -->

# ch-02 — Reading Q&A

Companion to [[read]]. One entry per clarifying question; kernel of the answer only.

---

## Q1. 인트로의 'boundary'가 무슨 의미야?

**Kernel.** 물리적 선(네트워크/배포)이 아니라 **언어·모델이 갈라지는 개념적 선** — modeling 고도("contexts are modules first"). 정의적 성질 둘: **(1)** 안에선 한 단어가 한 뜻, 밖에선 뜻이 바뀜 — Customer가 Sales=파이프라인 리드 / Support=티켓 달린 계정이면 거기가 boundary (ubiquitous language의 단층선). **(2)** ch-01의 expensive-to-reverse 그 자체 — 선 안쪽 변경은 쌈(반대편 모름), 가로지르는 변경은 비쌈(refactor→migration, L25 비대칭). 그래서 "single most expensive-to-reverse decision".

**짝 구분:** boundary = *선*(where language breaks) / bounded context = 그 선이 *둘러싼 영역*(one model·one language). **ch-01 잇기:** signature/콘센트 = 한 context *안*의 작은 경계(core↔DB); ch-02 boundary = context *사이*의 큰 경계(Sales-모델↔Support-모델). 둘 다 "안은 싸고 가로지르면 비싸다", ch-02는 그걸 도메인 전체 규모로. 어디 긋나? = 언어가 갈라지는 곳(가장 느리게 변하는 구조). (see read.md §1/§1.1/§2, L11/L25/L43)

---

## Q2. ch-02에서 말하는 게 (ch-01 discuss의) 시그니처야?

**Kernel.** 아니다 — 같은 거 아니고 **같은 원리의 다른 고도**. **signature**(ch-01) = 호출의 *모양*(이름+인자+반환+에러), **구문적**, 커넥터 한 개의 형태, 한 context *안*에도 여럿. **boundary**(ch-02) = *언어·의미*가 갈라지는 선, **의미적**, context를 통째로 감싸는 한 겹. (ch-01 discuss에서 "boundary"를 느슨히 signature 뜻으로 썼을 뿐; ch-02는 DDD 전문어로 더 좁고 큼.)

**벽 vs 문틀:** boundary = *벽*(두 방이 다른 언어 쓴다는 사실을 선언) / signature = 그 벽에 난 *문틀*(건널 때 통과할 호출 모양). boundary를 가로지를 땐 항상 published contract(ACL=번역층)를 통과 → 그 번역층의 메서드가 곧 signature. 즉 signature = boundary라는 벽의 문 하드웨어. **Lina:** Sales `Customer`=리드 / Support `Customer`=티켓계정 → 이 의미 단층이 **boundary**; Support의 `getAccountStatus(id)->Status`가 **signature**; Sales가 ACL로 번역해 건넘. **공통:** 둘 다 ch-01 *expensive-to-reverse*("안은 싸고 가로지르면 비쌈") — 규모만 다름(signature=컴포넌트 고도 / boundary=도메인 고도). (see [[Q1]], read.md §2/§3 context mapping)

---

## Q3. DDD가 뭐야?

**Kernel.** **Domain-Driven Design** (Eric Evans 2003). 핵심: *복잡한 SW의 진짜 난제는 기술이 아니라 비즈니스 도메인 모델링 — 코드의 모델을 도메인 전문가가 말·생각하는 방식과 계속 일치시켜라.* **Domain** = 문제 영역(영업/지원/결제); **Driven** = DB·UI·프레임워크가 아니라 *도메인 모델이 설계를 끌고 간다*.

**핵심 통찰:** 큰 시스템 복잡도는 *하나의 통합 모델로 업무 전체*를 담으려다 polysemy로 터진다 → DDD의 답은 "통합 말고 **언어로 쪼개라**" = **bounded context**. 그래서 ch-02 = **strategic DDD**.

**두 반쪽:** *Strategic*(ch-02, architecture 고도) = bounded context로 쪼개고 context mapping으로 잇기, ubiquitous language. *Tactical*(뒤 챕터, 구현 고도) = Entity / Value Object / **Aggregate** / Repository / Domain Event. **Ubiquitous language** = 개발자·전문가·코드가 한 단어를 한 뜻으로(번역층 없이).

**ch-01 연결 + 신화:** DDD는 외우는 best practice가 ❌, ch-01의 "경계를 가장 느린 구조=언어 위에 그어라"를 실행하는 **framing**. 신화 ❌ *"DDD엔 microservices가 필요"* — **거짓**(monolith 안에서도 DDD; context=모듈 경계로 충분). best-practice 신화와 같은 모양(패턴을 framing 자리에 끼움). **Lina:** 도메인=B2B 영업 자동화, 언어=`lead`/`cadence`/`qualified`; `Customer`가 Sales/Support/Billing서 갈리면 bounded context 분리. (see [[Q1]], read.md §1)

---

## Q4. 'Why One Unified Model Fails'(§2)을 더 구체적으로.

**Kernel.** 실패는 *기술적*이 아니라 **언어적** — 같은 단어가 팀마다 다른 뜻인데 한 클래스에 욱여넣어 생김. **frame-by-frame 부패**(§2.2): `Customer`가 Sales(`pipeline_stage`)→Support(`sla_tier`, `open_ticket_count` 필드 추가)→Billing(`tax_id`, `payment_method` 추가)로 자람. 각 단계는 개별적으론 합리적("이미 클래스 있잖아")이라 아무도 못 느낌.

**구체적 증상 4:** ① **Null 폭발** — 한 용도당 필드 절반이 null. ② **의미 과부하** — `is_active`가 세 팀에 호환불가 세 뜻("파이프라인 있음"/"열린 티켓"/"유효 카드") → 읽을 때마다 코드 밖 "어느 뜻?" 필요. ③ **Invariant 죽음** — Sales는 `pipeline_stage` 필수, Billing은 금지 → 모순이라 둘 다 강제 못 함, nullable로 풀림. ④ **Semantic coupling**(§2.1) — Support 변경이 *같은 클래스를 만진다는 이유만으로* Sales를 깸(배포 아닌 의미 의존). god-model = 최대 결합; "도메인을 통합한 게 아니라 도메인의 *모순*을 한 객체에 통합".

**고침:** 필드를 영리하게가 ❌ → **세 Ubiquitous Language 충돌을 인정하고 세 context로 split**. 각 `Customer`가 작고·전부 required·correct, 만나는 곳엔 ACL 번역. *밖의 경계가 안의 엄격함을 사준다*(L121). **Lina:** `Customer`가 Lead/Conversation/Scheduling/CRM-Sync서 갈림; god-`Customer`면 §2.2 재현 + LLM이라 더 치명(틀린 개념 위 유창한 추론 → "자신만만하게 틀린 tool call", L255). (see read.md §2/§2.1/§2.2, L57/L61/L121)

---

## Q5. `class Customer`에 계속 추가하는 대신 더 나은 방법은?

**Kernel.** context마다 *자기* 모델을 따로 만들고 그 context가 부르는 이름을 줌: `sales.Lead`(stage 필수) / `support.Account`(sla_tier 필수) / `billing.BillingEntity`(tax_id 필수). 통찰: **모델 ≠ 실제 대상** — 현실 회사는 하나지만 세 클래스는 *세 개의 다른 투영*(중복 아님). 잇는 건 **공유 객체가 아니라 공유 식별자**(`customer_ref` 전역 id로 correlate, 모델은 각자 소유). 만나는 곳엔 **명시적 번역 한 곳(ACL/mapper)** — 예 `to_billing_entity(lead, payment)` 가 전환 시점에 변환, 다른 데로 안 샘.

**왜 나은가** = §2.2 4증상이 각각 사라짐: Null폭발→각 클래스 전부 채워짐 / 의미과부하→`is_active` 대신 context별 정확한 술어(`is_in_pipeline()`/`has_open_ticket()`/`has_valid_card()`) / Invariant→다른 클래스라 모순 없이 required 강제 / coupling→Support가 `Account` 바꿔도 `Lead` 못 닿음(공유면=통제된 번역 1곳). = "밖의 경계가 안의 엄격함을 사준다".

**⚠️ 항상 쪼개진 않음(§6.2):** 양쪽서 *진짜 같은 뜻*이면 isolation 0 + mapper만 떠안음. detector=polysemy, 갈라지는 자리에서만. **Lina:** Lead/Conversation/Scheduling/CRM-Sync 네 모델, CRM-Sync는 ACL 뒤(Salesforce 필드가 core로 못 새게). (see read.md §3, L101/L121, §4.1, §6.2)

---

## Q6. 규칙이 "객체에 너무 많이 담지 마 + null 조심"인가?

**Kernel.** ❌ 그 둘은 **증상이지 규칙이 아님.** 규칙으로 삼으면 오진: "너무 많이 담지 마"→큰 클래스를 *필드 그룹*으로만 쪼갬(SRP) = 한 언어 안 기술적 분해라 병 그대로; "null 조심"→기본값으로 null만 0 만들어도 여전히 틀린 모델. **진짜 규칙 = 언어적: 단어가 갈라지는 곳(polysemy)에서 쪼개라.** null·비대함은 *두 언어를 한 모델에 욱여넣은 결과*. 기준은 "크냐/null 있냐"가 ❌ → **"`Customer`가 모두에게 같은 뜻인가(속성·규칙·`is_active`)?"**.

**크기는 기준 아님(반례 둘):** 필드 50개라도 *한 팀 일관된 뜻*이면 안 쪼갬(풍부한 모델일 뿐); 필드 6개라도 `is_active`가 3뜻이면 쪼갬. → 크기는 필요조건도 충분조건도 아님.

**왜 안 쪼개는 게 유리(학습자 결론, 맞음):** 단지 이득0이 아니라 쪼개는 데 *적극적 비용* — 번역(ACL/mapper) 영구 부채 + cross-context 수다 + 개념 중복(§6.2, god-model의 거울상="빈약한 context 10개"). First Law: 양쪽 가격표(안 쪼개면 god-model 위험 / 쪼개면 번역세), polysemy가 승자 결정. **드리프트:** 오늘 일관돼도 내일 언어 갈라질 수 있음(Lead→인바운드/아웃바운드) → 그때 쪼갬; 그래서 일단 modular monolith 모듈로(쪼개기·되합치기 쌈, service면 migration). (see read.md §3/§6.2, [[Q5]])

---

## Q7. 이거(bounded context)는 데이터 스트럭처인가?

**Kernel.** ❌. `class Lead` *하나하나*는 데이터 스트럭처(+행동)지만 **bounded context는 데이터 스트럭처가 아님 — 고도가 다름.** 사다리: **Data structure**(필드 그릇, 순수 모양) ⊂ **Model**(+행동·불변식·규칙) ⊂ 표현하는 **Ubiquitous Language**(어휘 합의, 데이터 ❌) ⊂ **Bounded Context**(한 모델+언어를 둘러싼 *의미의 경계/scope*, 데이터 ❌). context = "이 선 안에서 단어가 한 뜻"인 경계선([[Q1]]).

**왜 중요(안 그럼 §2.2 재발):** context를 "그냥 데이터 스트럭처"로 보면 *모양(shape) 문제*로 사고 → "모두 위한 Customer 구조체 배치?" = god-model 본능. 올바른 질문은 모양이 ❌ **의미**: "Customer가 모두에게 같은 뜻인가?" = 데이터 설계가 아니라 **언어 설계**.

**증거(같은 모양, 다른 context):** `sales.Lead{id,name,email}` vs `marketing.Subscriber{id,name,email}` — 바이트 동일하지만 규칙 다름(consent/unsubscribe vs pipeline) → 다른 context. **데이터 스트럭처가 context를 정의하지 않음; 언어/의미가 정의하고 스트럭처는 결과.** ([[Q2]] signature(모양) vs boundary(의미)와 같은 구분.) (see read.md §3, L113/L115/L119)

---

## Q8. §3의 'model'과 'context'는 각각 무슨 뜻?

**Kernel.** **model = 도메인의 *선택적 추상*** — 한 슬라이스를 추론·해결하려 골라낸 개념+속성+행동+규칙(invariant)+관계. 셋: ① 선택적(필요한 것만, 나머진 버림) ② 데이터 스트럭처가 아니라 그 위 의미·행동·규칙([[Q7]]) ③ **결정적: model = 언어 그 자체** — L115 *"the model **is** a shared language between devs and domain experts"*(문서가 아니라 합의 어휘를 실행가능하게 한 것). model이 *아닌* 것: DB(저장)/다이어그램(그림)/도메인 전체(model은 그 축약).

**context = bounded context** = 한 모델이 적용되고 모든 용어가 *정확히 한 뜻*을 갖는, **명시적 선을 그은 상황의 범위.** 일상어 "context"(단어에 뜻 주는 상황: "bank"=둑 vs 은행)를 빌림 + **"bounded"**=그 상황에 경계선. 선 안=한 모델·한 언어·모순0 / 선 밖=다른 모델(같은 단어 다른 뜻). (Fowler L111)

**관계:** model=*내용물*(개념+언어+규칙) / context=그 내용물을 둘러싼 *경계*("여기서만 유효"). 비유: model=게임 규칙·한 언어 / context=경기장 라인·그 언어 공용어인 나라. **one context ↔ one model ↔ one Ubiquitous Language**(그릇·내용·접착제, 한 묶음). **Lina:** Sales *context*=파이프라인 일 전부의 경계 / Sales *model*=`Lead`·`Stage`·`probability`+규칙 / *Language*="lead/stage/qualified/close". (see read.md §3, L111/L113/L115)
