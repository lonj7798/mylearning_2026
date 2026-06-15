<!-- chapter: ch-09
     track: evolution
     kind: content
     title: Evolution — Strangler-Fig Migration and Fitness Functions
     deps: [[ch-04]], [[ch-08]]
     sources: [[martin-strangler-fig]], [[newman-building-microservices]], [[fowler-monolith-first]], [[richards-ford-fundamentals]], [[martin-clean-arch]], [[distributed-monolith]], [[decompose-by-business-capability]]
-->

# 09장 — Evolution: Strangler-Fig Migration and Fitness Functions

> **핵심 통찰.** 이 코스 전체에 걸쳐 당신이 가격을 매겨온 모든 architectural bet(아키텍처적 베팅, 즉 비용을 치르고 내린 설계 결정)은 불완전한 정보로 내려진 것이며, 그 결정을 내린 후에도 도메인은 계속 움직인다 — 그래서 아키텍처에 필요한 마지막 속성은 오늘의 correctness(정확성)가 아니라 *내일의 revisability(개정 가능성)*다. 이미 놓아버린 bet에는 두 가지 뚜렷한 위협이 있다. 첫 번째는 그것을 *변경*하는 것이 one-way door(한 번 지나면 돌아올 수 없는 문)라는 것이다: big-bang rewrite(전면 재작성)는 기존 시스템을 동결시키고, 모든 위험을 단 한 번의 cutover(전환)에 누적시키며, 끝날 때까지 아무것도 전달하지 못한다. strangler-fig(교살자 무화과, 점진적 마이그레이션 패턴)는 그것에 답한다 — 새 시스템을 기존 시스템 *주위에* 키우고, 한 번에 하나의 capability seam(능력 이음매)씩 라우팅하며, legacy 조각을 하나씩 은퇴시켜 각 단계가 되돌릴 수 있을 만큼 작도록 한다. 두 번째 위협은 더 조용하다: 신중하게 선택한 bet도 시스템이 진화하면서 *조용히 썩을(rot silently)* 수 있다 — dependency rule(의존성 규칙)이 import 하나씩 위반되고, p99 latency가 당신이 약속한 숫자를 슬금슬금 넘어가며, 결국 당신이 문서화한 아키텍처가 당신이 실제로 운영하는 아키텍처와 더 이상 일치하지 않게 된다. fitness function(적합도 함수, 아키텍처 속성을 자동 검증하는 장치)은 그것에 답한다 — 보호 대상 characteristic(특성)이 침식되는 순간 빌드를 실패시키는 자동화된 객관적 검사다. Migration은 bet을 *변경 가능하게(changeable)* 유지하고; fitness function은 변경되지 않은 채로 *부패(decaying)*하는 것을 막는다. 둘 다 서로 다른 timescale(시간 척도)에 적용된 동일한 규율이다: ch-08의 stability pattern들이 *failure(실패)*의 blast radius(폭발 반경)를 묶었던 방식 그대로, *change(변경)*의 blast radius를 묶는 것이다.

> **가이드라인.** 심각한 무언가를 교체해야 할 때, rewrite를 거부하라. legacy 시스템 앞에 façade(파사드)나 proxy를 두어 caller(호출자)가 눈치채지 못한 채 routing이 바뀔 수 있게 한 다음, 한 번에 하나의 business-capability seam(비즈니스 능력 이음매)씩 — *그 behavior와 그 data를 함께* — 추출하고, 병렬로 검증하며, 교살된 경로를 삭제하고, 반복하라. 당신은 점진적이고 가시적인 수익과 단계별 reversibility(되돌릴 수 있음)를 사는 대신, 두 시스템을 동시에 운영하고 그 기간 내내 coordination tax(조율 비용)를 짊어진다. 그리고 마침내 *이* 시스템에 중요한 소수의 architecture characteristic을 골랐을 때, 그것들을 wiki 속 열망으로 남겨두지 마라: 각각을 fitness function으로 인코딩하라 — ArchUnit dependency rule, contract test, 빌드를 실패시키는 임계값을 가진 p99 monitor — 그래서 그 characteristic이 단지 *의도된(intended)* 것이 아니라 *강제되도록(enforced)* 하라. evolution(진화)에 정직하게 가격을 매겨라: 그것은 지속적인 migration 규율과 끊임없는 enforcement(강제)를 비용으로 요구하지만, 그 반복되는 비용이야말로 되돌리기 어려운 결정들이 조용히 *돌이킬 수 없는 잘못된* 결정이 되지 않게 하는 정확한 방법이다.

> 💡 **쉬운 설명:** 이 챕터의 핵심은 "결정을 잘 내리는 것"과 "결정을 나중에 바꿀 수 있게 만드는 것"이 서로 다른 일이라는 점이다. strangler-fig는 큰 시스템을 한 번에 갈아엎지 않고 조금씩 새것으로 바꿔치기하는 방법이고, fitness function은 시간이 지나도 아키텍처가 처음 설계한 모습 그대로 유지되는지 자동으로 감시하는 장치다. 하나는 "변경을 안전하게", 다른 하나는 "유지를 정직하게" 만든다.

---

## 1. The Spine, Re-Applied: Evolution Is the Insurance on Every Other Bet

이 코스의 조직화 주장은, ch-01에서 설치된 것으로, 아키텍처란 되돌리기에 비용이 큰 결정들의 집합이며 단 하나의 법칙으로 판단된다는 것이다:

> "Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet." — Richards & Ford, *Fundamentals of Software Architecture* (book; thesis extracted, quoted as commonly published) — [[richards-ford-fundamentals]]

지금까지의 모든 챕터는 bet을 놓았다: ch-02는 가장 천천히 변하는 구조 위에 bounded context(경계 지어진 맥락)를 그렸고, ch-03은 의존성을 안쪽으로 향하게 했으며, ch-04는 modular monolith(모듈형 모놀리스)를 기본값으로 삼았고, ch-05는 contract(계약)를 공표했으며, ch-06은 saga(사가)를 위해 잃어버린 isolation(격리)을 받아들였고, ch-08은 실패를 묶기 위해 breaker(차단기)를 놓았다. 각각은 불확실성 아래에서 내려졌고, 각각은 *결국* 틀릴 것이다 — 분석이 나빴기 때문이 아니라, 당신이 분석한 도메인이 가만히 있지 않을 것이기 때문이다. 이 챕터는 그것들 모두에 대한 보험이다. 새로운 구조적 bet을 놓지 않는다; *bet을 변경하는 행위*를 저렴하게 만들고, *bet을 유지하는* 행위를 정직하게 자동화한다.

> 💡 **쉬운 설명:** 앞선 모든 챕터가 "이 도메인에는 이런 구조가 맞다"는 베팅을 했다면, 이 챕터는 그 베팅들이 언젠가 틀릴 것을 전제로 한 "보험"이다. 새 구조를 제안하는 게 아니라, 기존 구조를 바꾸는 비용을 낮추는 게 목적이다.

ch-09가 ch-01의 정의에 무엇을 하는지 정확히 말하는 방법이 있다. ch-01은 아키텍처를 되돌리기에 비용이 큰 결정들로 정의했고, *어떻게 짓느냐*에 따라 동일한 결정이 trivially-reversible(사소하게 되돌릴 수 있는)에서 one-way-door까지의 스펙트럼 어디에나 놓일 수 있다고 관찰했다. 이 챕터는 결정을 내린 *후에* 그 결정을 스펙트럼 *아래로* 옮기는 기법들의 집합이다 — service boundary(서비스 경계), vendor 선택, data-store 약속을 받아 그 reversal cost(되돌림 비용)를 낮춰서, 그것에 대해 틀린 것이 생존 가능하도록 만든다. 그것은 애초에 결정을 잘 내리는 것과는 근본적으로 다른 활동이며, 그래서 evolution이 각 bet을 놓은 챕터들 안으로 접혀 들어가는 대신 자기만의 phase를 가질 자격이 있는 이유다. 이 코스의 모든 bet을 올바르게 놓고도, 틀릴 비용을 낮추는 데 결코 투자하지 않는다면 여전히 깨지기 쉬운 아키텍처를 출시할 수 있다; 반대로, 단지 *충분한* 정도의 bet 집합이라도 개정을 저렴하게 유지하면, 콘크리트에 얼어붙은 훌륭한 bet 집합보다 오래 살아남을 것이다.

이것이 ch-08의 resilience(회복력) 옆 **evolution** phase에 자리하는 이유는 [[richards-ford-fundamentals]]에 명시된 관통선이다:

> "An evolutionary architecture supports guided, incremental change across multiple dimensions." — Ford, Parsons & Kua, *Building Evolutionary Architectures* (book; thesis extracted) — [[richards-ford-fundamentals]]

이 문장을 세 개의 하중을 견디는 단어로 읽어라. **Guided(인도된)**: 변경은 희망이 아니라 객관적인 무언가에 의해 조종된다 — 그것이 fitness function이 제공하는 것이다. **Incremental(점진적)**: 변경은 작고 되돌릴 수 있는 단계로 도착하며, 결코 하나의 큰 cutover가 아니다 — 그것이 strangler-fig가 제공하는 것이다. **Across multiple dimensions(여러 차원에 걸쳐)**: 코드 구조뿐 아니라 performance, security, data, deployability(배포 가능성) — ch-01이 도출하라고 한 모든 architecture characteristic이 썩을 수 있으므로, 모든 것이 지켜질 수 있다. ch-08은 *failure*의 blast radius를 시간상으로 묶었다(하나의 outage, 하나의 breaker). 이 챕터는 *change*의 blast radius를 시간상으로 묶는다(하나의 capability, 하나의 migration 단계; 하나의 위반된 rule, 하나의 실패한 빌드).

> 💡 **쉬운 설명:** "evolutionary architecture"라는 한 문장을 세 단어로 쪼개면 이 챕터 전체의 지도가 나온다 — guided는 fitness function(자동 감시), incremental은 strangler-fig(조금씩 이전), multiple dimensions는 코드뿐 아니라 성능·보안·데이터까지 모두 보호 대상이라는 뜻이다.

### 1.1 Fowler's amendment, finally cashed out

ch-01은 "architecture = hard-to-change decisions" 정의에 대한 Fowler의 우호적 수정안을 인용했다: *"a good architect makes change easier — thus reducing architecture."* 그것은 당신이 처음 들었을 때는 열망이었다. 이 챕터는 그것이 mechanism(메커니즘)이 되는 곳이다. strangler-fig는 하나의 거대한 되돌릴 수 없는 rewrite를 작고 되돌릴 수 있는 추출들의 연속으로 변환함으로써 *되돌릴 수 없는 집합을 줄인다(shrinks the irreversible set)*. fitness function은 *다르게 줄인다* — 우발적인 아키텍처 변경(위반된 dependency rule, 날아간 latency budget)의 비용을 즉각적이고 감지하기 저렴하게 만들어서, 그것이 결코 현장에서 되돌리기 비싼 사실로 복리화되지 않게 한다. 둘 다 코드로 된 Fowler의 수정안이다: 당신이 이미 내린 결정들의 reversal cost를 낮추는 것.

### 1.2 Why evolution is a phase, not an afterthought

흔한 해석은 migration과 "깨끗하게 유지하기"를 진짜 설계 작업이 끝난 후 아키텍처에 *일어나는* 운영상의 잡일로 취급한다. 이 코스는 그 프레이밍을 거부하며, 그 거부가 ch-08과 ch-09가 자기들만의 phase를 이루는 이유다. 아키텍처를 evolvable(진화 가능)하게 만드는 결정 그 자체가 자기만의 characteristic과 자기만의 비용을 가진 아키텍처 결정이며 — 그리고 그것은 오직 *선제적으로(up front)* 살 수 있는 결정이다. 모든 caller를 구체적인 구현에 hard-wire(고정 배선)한 시스템에 reversibility를 retrofit(나중에 끼워 넣기)할 수 없다; strangler가 저렴한 것은 오직 누군가가 필요해지기 전에 interception seam(가로채기 이음매)에 투자하기로 결정했기 때문이다. 2년 동안 dependency rule을 위반한 코드베이스에 "dependency rule이 결코 위반되지 않았다"를 retrofit할 수 없다; fitness function이 작동하는 것은 오직 그것이 처음부터 실행되었기 때문이다. Evolvability(진화 가능성)는 다른 모든 "-ility"처럼 당신이 도출하고 비용을 치르는 characteristic이다 — 차이점은 그 payoff(보상)가 전적으로 미래에 있다는 것이며, 정확히 그래서 가장 자주 건너뛰는 것이고 이 phase가 방어하기 위해 존재하는 것이다.

> 💡 **쉬운 설명:** "나중에 천천히 evolvable하게 만들면 되지"는 통하지 않는다. interception seam이나 fitness function 같은 장치는 *시작할 때 미리* 심어야 저렴하다. 이미 망가진 시스템에 사후에 끼워 넣으려면 엄청난 비용이 든다 — 그래서 evolution은 부차적 잡일이 아니라 독립된 설계 단계다.

---

## 2. Why Big-Bang Rewrites Fail

이 패턴은 오직 그것이 거부하는 대상과 대비될 때만 의미가 있다. [[martin-strangler-fig]] (Martin Fowler, "StranglerFigApplication," martinfowler.com/bliki, 2004/renamed 2019)로부터:

> "Replacing a serious IT system takes a long time, and the users can't wait for new features." — Fowler, "StranglerFigApplication" — [[martin-strangler-fig]]

그 한 문장이 rewrite에 반대하는 경제적 논증 전체를 담고 있다. 심각한 시스템은 수년간 축적된 business logic, edge case(예외 상황), 그리고 힘들게 얻은 bug fix를 대표한다. rewrite는 *아무것도* 전달하기 전에 그 모든 것을 재현하겠다고 제안한다 — 그리고 그 기간 동안 기존 시스템은 동결된다(거기에 추가하는 모든 새 기능은 당신이 포팅해야 하는 버려질 작업이다) 한편 비즈니스는 사용자가 기다릴 수 없는 기능을 계속 요구한다. 당신은 이제 절반만 지어진 대체물과 움직이는 목표물 사이의 경주에 들어가, 하나의 시스템을 출시하기 위해 두 개의 노력에 자금을 댄다.

risk profile(위험 프로파일)이 진짜 살인자다. rewrite는 그 *모든* 위험을 하나의 사건으로 집중시킨다: cutover. 모든 것이 새 시스템에서 작동하거나 작동하지 않으며, 당신은 가장 나쁜 순간에 그것을 알게 된다 — 전환의 순간, 프로덕션에서, 기존 시스템은 이미 은퇴한 채로. 우아한 부분 상태는 없다. strangler는 이 속성들 하나하나를 뒤집는다:

| Property | Big-bang rewrite | Strangler-fig migration |
|---|---|---|
| Value delivery | 끝날 때까지 아무것도 없음 | 증분마다 점진적이고 가시적 |
| Risk concentration | 모두 하나의 cutover에 | 많은 작은 단계에 분산 |
| Reversibility | cutover 이후 사실상 0 | 각 단계를 격리하여 되돌릴 수 있음 |
| Old-system work during migration | 동결됨 / 버려질 작업 | 계속 실행되고 출시됨 |
| Failure cost | rewrite 전체 | 하나의 capability의 재작업 |
| Learning | 선행된 추측 | 각 단계가 다음 단계를 가르침 |

Fowler는 이것이 *ease(쉬움)*가 아니라 *manageability(관리 가능성)*를 산다는 데 정직하다 — 이 챕터가 과장하기보다 재현해야 하는 hedge(신중한 단서)다:

> "Replacing a software system… is never going to be an easy task" — Fowler, "StranglerFigApplication" — [[martin-strangler-fig]]

strangler는 migration을 쉽게 만들지 않는다. 그것을 *생존 가능하게(survivable)* 만드는데, 이는 다르고 더 정직한 주장이다.

> 💡 **쉬운 설명:** rewrite의 진짜 위험은 "오래 걸린다"가 아니라 "모든 위험이 단 한 번의 전환 순간에 몰려 있다"는 점이다. 그 순간 모든 게 작동하거나 전부 망가진다 — 그것도 기존 시스템이 이미 사라진 프로덕션에서. strangler는 위험을 수십 개의 작은 단계로 흩어 놓아 "쉽게"는 아니어도 "살아남을 수 있게" 만든다.

### 2.1 The metaphor, and why it earns its name

Fowler는 Queensland에서 strangler fig를 관찰했다: 덩굴이 host tree(숙주 나무) 수관(canopy) 높은 곳에서 발아하고, 기존 줄기 *주위로* 뿌리를 아래로 키우며, 수년에 걸쳐 원래 나무가 그 안에서 죽어가는 동안 자립적으로 변한다. 새 구조는 기존 것 위에 그리고 주위에 지어지며, 결코 먼저 깨끗이 베어버리는 방식이 아니다.

> "Like the fig, it begins with small additions, often new features, that are built on top of, yet separate to the legacy code base." — Fowler, "StranglerFigApplication" — [[martin-strangler-fig]]

"on top of, yet separate to(위에 있되, 분리된)"는 생물학 속에 숨은 설계 제약이다: 새 코드는 legacy 코드와 공존하며, 점진적으로 가로채고 교체하되, 결코 둘 다 한 번에 뜯겨 나가는 순간을 요구하지 않는다.

### 2.2 The trap is *not finishing*

strangler에는 metaphor 자체가 경고하는 잘 알려진 failure mode가 있다: *성장을 멈추는* fig. 두 개의 seam을 추출한 다음 멈춰버리는 migration — 긴급한 기능 작업이 돌아왔거나, 그 노력을 소유한 팀이 해체되었기 때문에 — 은 시스템을 모든 것 중 가장 비싼 상태에 남긴다: 두 개의 코드베이스, 영구적인 façade, 분할된 data estate(데이터 자산), 그리고 결국의 payoff 없이 전체 distribution tax(분산 비용)만. 각 *단계*를 안전하게 만드는 reversibility는 또한 무한정 일시정지하는 것을 심리적으로 쉽게 만들며, 일시정지된 strangler는 좋은 의도를 가진 distributed monolith(분산 모놀리스)일 뿐이다. 따라서 이 패턴이 요구하는 규율은 단지 "점진적으로 가라"가 아니라 "legacy core가 사라지거나 의도적으로, 명시적으로 유지될 때까지 *계속 가라*"이다(step 4의 "or small enough to keep"은 *결정*이지, 표류해서 도달하는 장소가 아니다). 이것이 패턴이 쉽지는 않지만 관리 가능하다는 Fowler의 hedge에 대한 정직한 해석이다: migration이 멈출 때까지 관리는 결코 멈추지 않는다.

> 💡 **쉬운 설명:** strangler의 가장 흔한 함정은 실패가 아니라 *중간에 멈추는 것*이다. 두세 개만 옮기고 멈추면 두 시스템을 동시에 유지하는 최악의 상태에 갇힌다. 각 단계가 되돌리기 쉽다는 바로 그 안전함이, 역설적으로 "나중에 끝내자"며 무한정 미루기 쉽게 만든다. 멈춤은 우연이 아니라 명시적 결정이어야 한다.

이것이 또한 이 챕터의 두 절반이 처음 맞닿는 곳이다. 멈춘 strangler는 감지 가능하다: legacy core에 아직 남아 있는 capability의 수가 줄어들기를 멈추고, façade routing table이 바뀌기를 멈추며, parallel-run(병렬 실행) 비교가 추가되기를 멈춘다. 그 각각은 *측정 가능한* 신호이며, 이는 각각이 fitness function이 될 수 있음을 의미한다 — "진행 중"이라고 선언된 migration이 N주 동안 seam을 옮기지 않으면 실패하거나 경고하는 검사. dependency rule을 보호하는 동일한 규율이 *migration momentum(추진력) 그 자체*를 보호할 수 있어, "결국 끝낼 거야"를 희망에서 추적되고 빌드에 가시적인 약속으로 바꾼다. 변경되지 않은 bet을 정직하게 유지하는 도구(§6)는 또한 *변경 중인* bet이 도중에 조용히 얼어붙는 것을 막는 도구다.

---

## 3. The Mechanism at Design Altitude

이것은 *설계-결정* 코스이므로, strangler는 여기서 배포 레시피가 아니라 아키텍처 패턴으로서 중요하다 — interception boundary(가로채기 경계)를 어디에 두고 어느 seam을 자를 것인가. [[martin-strangler-fig]]는 네 단계 루프를 제공한다:

1. **Intercept(가로채기).** caller와 legacy 시스템 사이에 HTTP proxy, event interceptor, 또는 façade를 두어 routing이 *caller가 눈치채지 못한 채* 바뀔 수 있게 한다. 이것은 패턴 전체에서 가장 중요한 단일 아키텍처적 움직임이다: façade는 caller가 *어느* 구현이 응답하는지로부터 분리되기 때문에 이후의 모든 단계를 되돌릴 수 있게 만드는 indirection layer(간접 계층)다.
2. **Extract one seam(하나의 seam 추출).** 깨끗한 boundary를 가진 business capability를 고른다 — 정확히 ch-04의 [[decompose-by-business-capability|capability seam]] 규율 — 그것을 새 코드로 재구현하고, façade를 통해 *그* capability의 트래픽만 새 구현으로 라우팅한다.
3. **Verify and shrink(검증하고 축소).** 위험이 정당화하는 곳에서 기존과 새것을 병렬로 실행하고, 출력을 비교하며, 새 경로를 신뢰하게 되면 교살된 legacy 코드를 삭제하여 monolith가 진짜로 작아지게 한다.
4. **Repeat(반복).** legacy core가 사라지거나, 더 이상 교살할 가치가 없을 만큼 작아질 때까지.

> **See it move:** [`figures/strangler-fig-timeline.html`](figures/strangler-fig-timeline.html)을 열고 time slider를 "Day 0"에서 "Retired"까지 드래그하라. capability seam들이 legacy monolith에서 새 서비스로 한 번에 하나씩 마이그레이션되는 동안 façade가 고정된 채 유지되는 것을 보라 — 그리고 진행 중인 어떤 seam에서든 **Roll back** 버튼을 사용해 이 챕터가 계속 강조하는 속성을 확인하라: 각 단계는 격리하여 되돌릴 수 있으므로, 잘못된 움직임은 rewrite 전체가 아니라 하나의 capability의 재작업만큼만 비용이 든다.

### 3.1 The façade is the load-bearing decision

step 2–4가 저렴한 것은 *오직* step 1이 먼저 행해졌기 *때문*임에 주목하라. interception layer 없이는 "이 capability를 새 코드로 라우팅하라"가 모든 caller를 편집하는 것을 의미한다 — 이는 당신이 해방시키려는 바로 그 call site에 migration을 다시 결합시키고, rollback을 두 번째 편집 라운드로 만든다. façade는 migration을 *routing-table 변경*으로 변환하는 것이다. 그래서 strangler는 근본적으로 indirection에 관한 진술이며, ch-03의 dependency-inversion(의존성 역전) 습관과 그토록 자연스럽게 결합되는 이유다: 두 경우 모두 중간의 interface가 caller가 참여하지 않은 채 그 뒤의 구현을 교체할 수 있게 하는 것이다.

step 1 *안에는* 그냥 지나치기 쉬운 진짜 결정이 있다: **interception point를 어디에 둘 것인가**. façade는 HTTP edge(경로별로 라우팅하는 reverse proxy / API gateway)에, 프로세스 내부(monolith 자신이 호출하는 dispatch interface)에, 또는 event bus 위(한 부류의 event를 재라우팅하는 interceptor)에 놓일 수 있다. 각 배치는 다르게 가격이 매겨진다. HTTP-edge proxy는 가장 분리되어 있고 가장 되돌릴 수 있다 — route를 뒤집으면 트래픽이 이동한다 — 그러나 *HTTP endpoint인* capability만 교살할 수 있다. in-process dispatch seam은 외부 URL이 없는 capability에 도달하지만, 당신이 은퇴시키려는 artifact 안에 살기 때문에 façade 자체가 끝까지 운반되어야 한다. First Law(제1법칙)는 이미 step 1에서 물고 있다: 더 분리된 interception point는 그것을 갖추지 않고 지어진 legacy 시스템에 retrofit하는 데 더 많은 비용이 들고, 추가하기 더 저렴한 in-process seam은 더 적은 reversibility를 산다. 당신은 어떤 아키텍처 결정과도 같은 방식으로 interception point를 선택한다 — 어떤 속성을 가장 저렴하게 유지해야 하는지로.

> 💡 **쉬운 설명:** façade(가로채기 계층)는 이 패턴 전체를 떠받치는 결정이다. 그것이 없으면 "이 기능만 새 코드로"가 결국 "모든 호출 지점을 일일이 고치기"가 되어버린다. façade가 있으면 migration이 그저 "라우팅 표 한 줄 바꾸기"로 줄어든다. 그리고 그 façade를 HTTP 입구·프로세스 내부·이벤트 버스 중 어디에 둘지도 First Law 트레이드오프다 — 더 잘 분리된 곳일수록 나중에 끼워 넣기는 비싸다.

### 3.2 Why the advantages are not free

Fowler는 upside(이점)를 그대로 나열하고, 이 챕터는 각각에 가격을 매긴다:

> "Investment and returns occur gradually and visibly." — Fowler — [[martin-strangler-fig]]
> "Since these components are small, there isn't so much risk involved." — Fowler — [[martin-strangler-fig]]
> "The business can reap the value from these new components, allowing earlier return on the investment." — Fowler — [[martin-strangler-fig]]

각 증분은 또한 "modernization이 계속되면서 더 나은 결정을 내리도록 돕는다" — 학습이 복리화되며, 이는 rewrite의 선행된 추측의 반대다. 그러나 First Law는 공짜 점심을 금하므로, 비용을 명시적으로 명명하라. strangler는 **변경을 저렴하게 유지한다**: migration 경로(어떤 단일 seam이든 재라우팅되거나 rollback될 수 있다), 단계별 위험(작다), funding model(자금 조달 모델, 점진적). 그것은 **비싸게 만든다**: 전체 기간 동안 두 시스템을 병렬로 실행하기, migration이 지속되는 동안 façade를 영구적인 인프라 조각으로 유지하기, 그리고 일부 capability는 새 세계에 일부는 기존 세계에 사는 hybrid state를 견디기 — 이는 cross-capability transaction(능력 간 트랜잭션)이 이제 당신이 옮기기를 끝내지 못한 boundary를 가로지른다는 것을 의미하며, ch-06의 모든 inside-vs-outside-data(안쪽 대 바깥쪽 데이터) 우려를 끌어들인다. 수년간 질질 끄는 strangler migration은 되돌릴 수 있고 저렴한 것이 아니다; 그것은 당신이 우발적으로 가입한 영구적인 distributed system이다. reversibility는 진짜지만, 소유한 것이 아니라 빌린 것이다.

> 💡 **쉬운 설명:** strangler의 장점(점진적 수익, 작은 위험, 학습 복리)은 공짜가 아니다. 그 대가는 migration 내내 두 시스템을 동시에 굴리고, façade를 인프라로 유지하며, 한쪽은 신·한쪽은 구인 어정쩡한 상태를 견디는 것이다. 그래서 "되돌릴 수 있음"은 *소유*가 아니라 *임대*다 — 마이그레이션을 끝내지 않으면 임대료를 영원히 낸다.

### 3.3 Verification is the step that is most often skipped

step 3은 "verify and shrink"라고 말하며, *verify*는 단계별 안전이 실제로 사는 곳이다. 가장 강력한 형태는 **parallel run**(때로 dark launching 또는 shadow traffic이라 불림)이다: 라이브 트래픽의 사본을 legacy 경로와 새 구현 둘 다로 라우팅하고, 출력을 비교하며, 충분히 오래 일치한 후에만 권위 있는 답을 새 경로로 전환한다. 이것이 단계를 *원리상*이 아니라 *실제로* 되돌릴 수 있게 만드는 것이다 — 당신은 legacy가 여전히 응답하는 동안 새 구현이 틀렸음을 발견하므로, rollback은 "새벽 3시에 백업에서 복원"이 아니라 "비교를 멈추기"다. parallel run에는 자기만의 비용이 있다(동일한 작업의 두 번의 실행에 비용을 지불하며, side-effecting capability — 이메일 보내기, 카드 청구하기 — 는 순진하게 이중 실행될 수 없는데, 이는 ch-06의 안전하게-replay-가능한 outside data와 effectful한 inside operation 사이의 구분으로 곧장 이어진다). verification을 건너뛰는 것은 strangler가 실제로는 각 전환을 눈먼 채로 베팅하면서도 점진적으로 *보이게* 하는 방법이다 — 당신은 작은 단계는 유지하지만 그 단계를 취한 모든 이유였던 reversibility를 버린다.

> 💡 **쉬운 설명:** parallel run은 신·구 구현에 같은 트래픽을 동시에 흘려보내 결과를 비교하는 것이다. 새 코드가 틀렸을 때 기존 코드가 아직 진짜 답을 내고 있으므로, 롤백이 "비교만 끄기"로 끝난다. 이 검증을 건너뛰면 작은 단계로 쪼개 놓고도 매 전환을 도박처럼 하는 셈이라 reversibility가 사라진다. 단, 이메일 발송·결제처럼 부작용이 있는 작업은 두 번 실행하면 안 되므로 병렬 비교 대상에서 빼야 한다.

---

## 4. Extract the Behavior *and* Its Data — Newman's Sharpest Rule

strangler는 capability를 벗겨내라고 말하고; Newman은 rewrite를 [[distributed-monolith]]와 맞바꾸지 않는 유일한 방법을 말한다. [[newman-building-microservices]] (Sam Newman, *Building Microservices* 2e / *Monolith to Microservices* — books, theses extracted; corroborated by O'Reilly excerpts and Newman's talks)로부터:

Newman의 migration playbook은 명시적으로 점진적이다 — monolith에서, 종종 modular monolith에서 시작하여, seam을 찾고, strangler를 사용해 한 번에 하나의 capability를 벗겨낸다 — 그러나 타협 불가능한 정제와 함께: *behavior*와 그 *data*를 **함께** 추출하라. 그가 경고하는 failure는 새 서비스가 자기 코드를 가지면서도 여전히 monolith의 데이터베이스로 손을 뻗어 그 테이블에 접근하는 half-migration(절반의 마이그레이션)이다. 그것은 당신이 탈출하려는 것과 schema를 공유하기 때문에 *독립적으로 배포할 수 없는* 새 deployable을 준다 — 이는 정확히 anti-pattern의 정의다:

> "all the pain of distributed systems without the independence" — Newman, on the [[distributed-monolith]] (book; thesis extracted) — [[newman-building-microservices]]

data-ownership rule(데이터 소유권 규칙)은 ch-04와 ch-06이 다른 방향에서 도달한 바로 그것이며, migration을 위해 재진술된 것이다:

> "Each microservice must own its data. Shared databases create hidden coupling and destroy independent deployability." — Newman (book; thesis extracted) — [[newman-building-microservices]]

그래서 strangler 단계는 "코드를 옮기고, 그것을 기존 DB로 가리키고, data는 나중에 옮겨라"가 아니다. 그것은 "코드와 그 data를 하나의 원자적 capability extraction으로 옮겨라"이다, 비록 그것이 더 어려운 버전일지라도 — 왜냐하면 쉬운 버전은 *migration 도중에* distributed monolith를 만들어내는데, 이는 당신이 시작한 monolith보다 엄밀히 더 나쁘기 때문이다. 이것이 이 챕터의 두 절반이 만나는 곳이다: strangler는 *sequence(순서)*를 주고, Newman은 *unit(단위)*을 주며, 그 단위는 behavior-plus-data이거나 아무것도 아니다.

> 💡 **쉬운 설명:** strangler가 "어느 순서로 옮길까"를 알려준다면, Newman은 "한 번에 무엇을 옮길까"를 알려준다 — 답은 "코드와 데이터를 통째로 함께". 코드만 떼어내고 데이터는 여전히 옛 DB에서 읽으면, 새 서비스가 옛 시스템과 스키마를 공유해 따로 배포할 수 없게 된다. 그게 바로 distributed monolith, 즉 "분산의 고통은 다 겪으면서 독립성은 없는" 최악의 상태다.

data가 어려운 부분인 이유는 이름을 붙일 가치가 있는데, 그것이 "data를 나중에 옮겨라" 지름길을 그토록 유혹적이면서 그토록 틀리게 만드는 것이기 때문이다. Newman의 전체 framework는 **independent deployability(독립적 배포 가능성)**를 산성 시험(acid test)으로 삼아 조직된다:

> "Independent deployability is the single most important principle." — Newman (book; thesis extracted, paraphrased from *Building Microservices* 2e) — [[newman-building-microservices]]

그리고 independent deployability는 **information hiding(정보 은닉)**을 통해 달성된다 — 서비스는 "데이터베이스, 기술 선택, 내부 workflow 같은 구현 세부사항을 숨기면서 API를 통해 behavior를 노출한다." shared database는 information hiding의 최대 *실패*다: 새 서비스의 가장 사적인 구현 세부사항(그 테이블, 그 schema)이 또한 monolith의 것이기도 해서, 어느 쪽이든 schema 변경이 둘 다로 파급되고 그것들은 결코 따로 배포될 수 없다. 그래서 data는 behavior와 함께 옮겨져야 한다 — 정돈 선호가 아니라, data store가 *바로* 그 은닉됨이 독립성을 만드는 것이기 때문이다. data 이동을 미루는 것은 독립성을 미루는 것이며, 이는 "data를 나중에 옮기는" migration의 중간 상태가, Newman 자신의 정의에 의하면, 전혀 부분적 microservice가 아니라 서비스의 옷을 입은 distributed monolith임을 의미한다.

### 4.1 The operational counterpart to MonolithFirst

strangler는 ch-04가 [[fowler-monolith-first]]로 펼친 논증의 후반부다. 그 챕터의 주장:

> "Almost all the successful microservice stories have started with a monolith that got too big and was broken up." — Fowler, "MonolithFirst" — [[fowler-monolith-first]]

MonolithFirst는 *진입(entry)* 규율이다: 도메인을 이해하기 전에 되돌릴 수 없는 boundary bet을 놓지 마라. strangler는 *퇴장(exit)* 규율이다: 마침내 분할할 때, 하나의 cutover가 아니라 점진적이고 되돌릴 수 있게 하라. 그것들은 동일한 거부다 — 아직 신뢰할 수 없는 boundary에 단 하나의 크고 되돌릴 수 없는 bet을 놓는 것을 거부하기 — 가 시스템 수명의 양 끝에 적용된 것이다. boundary가 어려운 부분인 Fowler의 이유가 정확히 퇴장도 점진적이어야 하는 이유다:

> "Even experienced architects working in familiar domains have great difficulty getting boundaries right at the beginning." — Fowler, "MonolithFirst" — [[fowler-monolith-first]]

처음에 boundary를 올바르게 잡을 수 없다면, 단 하나의 big-bang extraction에서 *모든* boundary를 올바르게 잡을 수 없음은 확실하다. strangler는 각 boundary가 한 번에 하나의 seam씩 발견되고 수정되게 하며, 잘못된 seam의 비용은 그 seam의 재작업으로 묶인다.

이렇게 보면, MonolithFirst와 strangler는 두 개의 패턴이 아니라 boundary commitment(경계 약속)에 대한 하나의 연속된 정책이다: *틀려도 감당할 수 있는 것보다 더 큰 boundary bet을 결코 놓지 마라*. 시스템의 시작에서 그것은 "아직 service boundary를 전혀 그리지 마라"를 의미한다(modular monolith). 나중에 그것은 "하나의 service boundary를, 되돌릴 수 있게 그리고, 다음 것을 그리기 전에 그것으로부터 배워라"를 의미한다(strangler). 정책은 일정하다; 도메인에 대한 당신의 이해가 자라면서 오직 감당 가능한 bet 크기만 바뀐다. 이것이 evolution이 다른 모든 챕터에 대한 보험인 가장 깊은 의미다: 그것은 boundary commitment — 코스가 식별한 가장 되돌리기 비싼 단일 결정 — 를 당신이 실제로 아는 만큼에 비례하게 유지하는 규율이다.

> 💡 **쉬운 설명:** MonolithFirst("처음엔 쪼개지 마라")와 strangler("쪼갤 때는 한 번에 하나씩")는 사실 같은 원칙의 양 끝이다 — "틀려도 감당할 수 있는 크기보다 큰 경계 베팅은 하지 마라." 도메인을 모를 때는 감당 가능한 베팅 크기가 "0"이고(modular monolith), 알게 될수록 "하나씩"으로 커진다.

---

## 5. Doc-vs-Reality: "A Rewrite Is Cleaner, So It's Faster"

이 챕터가 죽여야 하는 대중적 서사는 엔지니어의 영원한 유혹이다: *legacy 시스템은 엉망이다; 깨끗한 rewrite가 더 빠를 것이고 결과는 더 나을 것이다.* [[COLLECTION-PLAN]]의 reconciliation table(조정 표)은 evolution 전체 호(arc)를 primary source(1차 출처)가 folklore(통념)를 이기는 것을 중심으로 프레이밍하며, strangler의 primary source는 이것에 대해 직설적이다.

| Popular narrative | What the primary source actually says |
|---|---|
| "A clean rewrite is faster than untangling the legacy system." | Fowler: 심각한 시스템을 교체하는 것은 "오래 걸리고, 사용자는 기다릴 수 없다." rewrite는 끝날 때까지 아무것도 전달하지 못하고 모든 위험을 하나의 cutover에 집중시킨다; strangler는 단계별로 묶인 위험과 함께 점진적으로 가치를 전달한다. — [[martin-strangler-fig]] |
| "Once we split it into services, the migration is basically done." | Newman: 여전히 monolith의 데이터베이스를 공유하는 서비스는 [[distributed-monolith]]다 — "독립성 없이 분산 시스템의 모든 고통." migration은 behavior *와 data*가 함께 추출될 때만 끝난다. — [[newman-building-microservices]] |
| "Microservices are the modern best practice; monoliths are legacy to be rewritten." | Fowler: MonolithFirst로 시작하라; "거의 모든 성공적인 microservice 이야기는 monolith에서 시작했다." 그것은 업그레이드가 아니라 MicroservicePremium(마이크로서비스 프리미엄)을 가진 trade다 — 그리고 migration은 rewrite가 아니라 점진적이다. — [[fowler-monolith-first]] |

모든 행에서 해결은 동일하다: rewrite가 *더 빠르게 느껴지는* 것은 정확히 가치 있기 때문에 어려운 부분들을 무시하기 때문이다 — 축적된 edge case와 data. strangler는 분기별로는 더 느리고 *expected risk-adjusted delivery(기대 위험 조정 전달)*에서는 더 빠르다, 왜냐하면 결코 전체 시스템을 단 하나의 사건에 베팅하지 않기 때문이다. folklore는 잘못된 변수(인식된 깨끗함)에 최적화하고 primary source는 올바른 것(묶이고 되돌릴 수 있는 전달)에 최적화한다.

두 번째 행은 성공으로 가장한 실패이기 때문에 자기만의 강조를 받을 자격이 있다. 새 서비스들이 여전히 monolith의 테이블에서 읽는 동안 "서비스로 분할했다"고 선언하며 승리를 외치는 팀은 migration의 *가시적인* 절반(새 deployable, 새 repo, 새 dashboard)을 했고 *하중을 견디는* 절반(data ownership)을 건너뛴 것이다. Newman의 정의에 의하면 그들은 전혀 microservice에 도달하지 못했다; 그들은 distributed monolith를 지었다 — 그리고 더 나쁘게는, 어려운 부분이 끝났다고 믿으면서 그것을 했기 때문에, 실제 어려운 부분(공유 schema 풀기)은 이제 그것을 뒷받침할 정치적 의지가 없다. primary source의 해결이 불편한 것은 정확히 org-chart(조직도) 변경이 아키텍처 변경으로 셈해지는 것을 거부하기 때문이다: migration은 deployment diagram 위의 서비스 개수가 아니라 data ownership으로 측정된다.

> 💡 **쉬운 설명:** "서비스로 쪼갰으니 끝났다"가 가장 위험한 오해다. 새 repo·새 대시보드 같은 *눈에 보이는* 절반만 하고 데이터 소유권이라는 *진짜 하중을 견디는* 절반을 건너뛰면 distributed monolith가 된다. 게다가 "끝났다"고 믿는 순간 진짜 어려운 작업(공유 스키마 풀기)에 아무도 힘을 실어주지 않는다.

rewrite 신화가 그토록 질긴 더 깊은 이유는 아키텍트가 저항하기 위해 명명해야 하는 cognitive bias(인지 편향)다: legacy 시스템의 복잡성은 *가시적*이고(당신은 매일 그 엉망을 쳐다본다) 그 *가치*는 비가시적이다(그것이 조용히 올바르게 처리하는 천 개의 edge case는 정확히 당신이 존재한다는 것을 잊은 것들이다). 따라서 rewrite 추정은 거의 전적으로 가시적인 부분에서 만들어진다 — "이것의 깨끗한 버전을 3개월에 쓸 수 있어" — 그리고 비가시적인 부분, edge case와 data migration이, 정확히 추정과 cutover를 날려버리는 것이다. strangler는 그 숨겨진 가치를 재현하기 더 저렴하게 만들지 않는다; 그것의 *발견(discovery)*을 점진적으로 만들어, 각 seam이 당신이 그것을 재구현하는 순간에 자기만의 숨겨진 복잡성을 표면화하게 하며, legacy 버전이 "correct"가 무엇을 의미하는지에 대한 executable specification(실행 가능한 명세)으로서 그 옆에서 여전히 실행되고 있다. 이것은 MonolithFirst, ch-04의 진입 측 논증과 동일한 논리다: 당신은 큰 bet을 할 만큼 충분히 알지 못하므로, *약속하기 전에 배우도록* 작업을 구조화한다. rewrite는 배우기 전에 약속하라고 요구한다; strangler는 순서를 뒤집는다.

> 💡 **쉬운 설명:** rewrite 신화가 안 죽는 이유는 인지 편향이다 — legacy의 *복잡함*은 매일 보여서 과대평가하고, 그것이 조용히 처리하는 수천 개 edge case의 *가치*는 안 보여서 깡그리 무시한다. strangler는 옛 코드를 옆에서 계속 돌리며 "정답이 뭔지"를 실행 가능한 명세로 삼으므로, 숨은 복잡성을 한 seam씩 발견하게 해준다.

---

## 6. Fitness Functions: Turning Characteristics from Aspiration into Enforcement

strangler는 bet을 *변경 가능하게* 유지한다. fitness function은 다른 failure mode를 다룬다 — 신중하게 선택한 bet이 당신이 보지 않는 동안 부패하는 것. [[richards-ford-fundamentals]]의 정의:

> "A fitness function is an objective integrity assessment of some architectural characteristic(s)." — Ford, Parsons & Kua, *Building Evolutionary Architectures* (book; thesis extracted) — [[richards-ford-fundamentals]]

각 단어를 풀어보라. **Objective(객관적)**: 누가 리뷰했는지에 의존하는 code-review 의견이 아니라; 매번 같은 방식으로 pass/fail을 반환하는 검사. **Integrity assessment(무결성 평가)**: 그것은 "이 코드가 좋은가?"가 아니라 "이 속성이 여전히 온전한가?"를 묻는다 — 그것은 미학이 아니라 *characteristic*을 지킨다. **Architectural characteristic(s)(아키텍처 특성)**: ch-01이 요구사항에서 도출하라고 한 "-ility"들 — *이* 시스템에 실제로 중요한 소수. fitness function은 ch-01의 "중요한 소수 characteristic을 도출하라"를 ch-09의 "그것들이 썩지 않게 하라"와 연결하는 메커니즘이다: 당신은 characteristic을 도출하고, 그것이 침식될 때 빌드를 실패시키는 function을 작성한다.

> 💡 **쉬운 설명:** fitness function은 세 단어로 정의된다 — *객관적*(누가 봐도 같은 pass/fail), *무결성 평가*(코드가 예쁜지가 아니라 속성이 멀쩡한지), *아키텍처 특성*(이 시스템에 진짜 중요한 -ility). 즉 "ch-01에서 고른 중요 특성이 시간이 지나도 살아 있는지"를 자동으로 검사하는 코드다.

### 6.1 The dependency rule as a fitness function

가장 깔끔한 예는 ch-03으로 곧장 연결된다. [[martin-clean-arch]]의 Dependency Rule:

> "Source code dependencies can only point inwards." — Robert C. Martin, "The Clean Architecture" — [[martin-clean-arch]]

ch-03에서 이것은 *design intent(설계 의도)*였다. intent의 문제는 그것이 한 번에 하나의 pull request씩 저하된다는 것이다: 서두르는 개발자가 framework type을 domain core로 import하고, 리뷰가 그것을 놓치며, 이제 policy가 detail에 의존한다 — rule이 금하는 바로 그것 — 그리고 framework를 교체해야 하고 core가 떨어지지 않을 때까지 아무도 눈치채지 못한다. fitness function은 intent를 *강제되게(enforced)* 만든다. "`domain.*`의 어떤 클래스도 `infrastructure.*`의 무엇도 import하지 않는다"를 단언하는 ArchUnit rule(또는 import-linter 검사, 또는 custom dependency test)이 CI에서 실행되고 위반하는 PR에서 *빌드를 실패시킨다*. 위반은 가능한 가장 저렴한 순간에 — merge 전에 — 잡힌다, 수년 후 되돌리기 비싼 사실로서가 아니라. 그것이 이 챕터가 파는 차이다: "dependency rule을 지켜라"가 한 줄의 test 코드가 되고, 당신이 문서화한 아키텍처가 당신이 운영하는 아키텍처다.

이 특정 fitness function이 표준 예인 이유는 그것이 지키는 characteristic — Dependency Rule을 통한 testability와 framework-independence — 가 그 침식이 *재앙이 되기 전까지 비가시적*인 것이기 때문이다. 단 하나의 금지된 import는 관찰 가능한 아무것도 하지 않는다: test는 여전히 통과하고, 기능은 여전히 작동하며, 위반은 불활성이다. 그것은 정확히 당신이 그 속성을 가장 필요로 하는 순간에만 비싸진다 — core가 독립적이어야 했던 framework나 데이터베이스를 교체하려 할 때, 그리고 아무도 표시하지 않은 수백 개의 작은 위반에 걸쳐 core가 조용히 그것에 융합되었음을 발견할 때. 이것이 이 챕터가 열었던 silent-rot(조용한 부패) 위협을, 구체화한 것이다: 당신의 가장 되돌리기 비싼 결정(domain policy)을 보호하는 속성 그 자체가 가장 조용히 저하되는 것이며, 정확히 그래서 그것은 *규율에 맡겨지기*보다 *기계에 의해 강제되어야* 한다.

> 💡 **쉬운 설명:** dependency rule이 fitness function의 대표 사례인 이유는, 그것이 깨져도 *재앙이 되기 직전까지 아무 증상이 없기* 때문이다. 금지된 import 하나는 테스트도 통과하고 기능도 멀쩡하다. 그러다 정작 LLM 모델이나 DB를 바꾸려는 순간, core가 수백 개의 작은 위반으로 이미 거기에 들러붙어 있음을 발견한다. 그래서 사람의 규율이 아니라 CI의 자동 검사로 막아야 한다.

### 6.2 A fitness function is not a unit test

fitness function을 테스트의 멋진 이름이 아니라 별개의 아이디어로 만드는 것이 무엇인지 정확히 하는 것은 가치가 있는데, 그 구분이 그것을 *architectural*하게 만드는 것이기 때문이다. unit test는 *function*이 올바르게 동작한다고 단언한다 — 이 입력이 주어지면, 저 출력을 기대하라; 그것은 *functionality(기능성)*를 지킨다. fitness function은 *architecture characteristic*이 전체 시스템에 걸쳐 성립한다고 단언한다 — dependency graph가 어떤 shape을 갖는다는 것, p99가 budget 아래에 머문다는 것, 어떤 module도 coupling ceiling(결합 상한)을 초과하지 않는다는 것; 그것은 단일 unit test가 볼 수 없는 *structural 또는 operational 속성*을 지키는데, 그 속성이 부분들이 어떻게 맞물리는지로부터 emergent(창발)하기 때문이다. 통과하는 test suite는 기능이 작동한다고 말해준다; 시스템이 조용히 유지보수 불가능해졌는지에 대해서는 아무것도 말해주지 않는다. fitness function은 *architecture* altitude(고도)에서의 검사다 — 정확히 그래서 그것이 설계 코스에 속하고 unit test는 그렇지 않은 이유다. 그것은 또한 fitness function이 코스 전체가 가격을 매겨온 trade-off들의 자연스러운 거처인 이유다: 당신이 베팅한 characteristic(dependency rule을 통한 testability, latency, deployability)은 정확히 fitness function이 방어하도록 지어진 것이다.

> 💡 **쉬운 설명:** unit test는 "함수 하나가 맞게 동작하는가"를, fitness function은 "시스템 전체가 약속한 형태/성능을 유지하는가"를 본다. 테스트가 다 통과해도 시스템이 조용히 유지보수 불가능한 진흙덩어리가 됐을 수 있는데, 그건 부분들의 *맞물림*에서 생기는 문제라 단위 테스트로는 안 보인다. 그래서 fitness function은 "아키텍처 고도"의 검사다.

### 6.3 The taxonomy of fitness functions

fitness function은 구조 테스트만이 아니다. [[richards-ford-fundamentals]]는 그것들을 여러 차원에 걸쳐 프레이밍하며, 실용적인 형태는:

| Characteristic guarded | Fitness function form | Fails the build when… |
|---|---|---|
| Modularity / clean dependencies | ArchUnit / import-linter rule | 금지된 cross-layer import가 나타날 때 |
| Performance | load test 내의 p99 latency assertion | p99가 약속한 임계값을 넘을 때 |
| Security | dependency-vuln scan / policy-as-code gate | 알려진-CVE 의존성이나 열린 포트가 도입될 때 |
| Reliability | CI 내의 chaos/contract test | circuit-breaker나 timeout이 제거될 때 (ch-08과 연결) |
| Coupling | cyclic-dependency / fan-out metric | 한 module의 afferent coupling이 budget을 초과할 때 |

일부는 CI에서 *원자적으로(atomically)* 실행되고(dependency rule); 일부는 monitor로 프로덕션에서 *연속적으로(continuously)* 실행된다(라이브 트래픽 위의 p99 latency 검사). 통합하는 속성은, 보호되는 characteristic이 *숫자나 rule*이 붙어 있고 위반될 때 *자동화된 결과(automated consequence)*가 있다는 것이다. 문서화되지 않고, 검사되지 않은 characteristic은 열망이다; fitness function은 약속이다.

triggered/continual(촉발형/연속형) 분할은 보이는 것보다 더 중요하다. *triggered* fitness function은 이산적인 순간에 — 빌드, deploy gate에서 — 실행되며 "우리가 막 출시하려는 artifact에서 속성이 온전한가?"에 답한다. *continual*한 것은 라이브 시스템에 대해 영원히 실행되며 "실제 부하 아래에서 지금 속성이 *여전히* 온전한가?"에 답한다. structural 속성(dependency shape, cyclomatic 한계)은 코드로부터 알 수 있으므로 CI에 속한다; operational 속성(p99 latency, error budget, availability)은 오직 프로덕션으로부터만 알 수 있으므로, load test에서는 *또한* 빌드를 실패시키는 alerting 임계값과 함께 monitoring에 속한다. 아키텍트의 일은 각 보호되는 characteristic을 그것이 실제로 측정될 수 있는 선의 쪽에 두는 것이다 — 빌드 시에만 검사된 continual 속성은 거짓 자신감을 준다(staging에서는 통과하고 프로덕션에서 썩었다), 그리고 프로덕션에서만 검사된 structural 속성은 고치기 저렴하기에는 위반을 너무 늦게 잡는다.

> 💡 **쉬운 설명:** fitness function은 두 종류다 — *triggered*는 빌드/배포 시점에 한 번 검사하고("출시하려는 이 코드에 속성이 멀쩡한가?"), *continual*은 라이브 시스템을 계속 감시한다("실제 부하에서 지금도 멀쩡한가?"). 핵심은 짝짓기다: 코드만 봐도 아는 구조적 속성은 CI에, 운영해 봐야 아는 성능 속성은 모니터링에 둬야 한다. 잘못 짝지으면 거짓 안심을 주거나 너무 늦게 잡는다.

### 6.4 Pricing the fitness-function bet

First Law는 여기에도 적용된다. fitness function은 **변경을 저렴하게 유지한다**: 자유롭게 refactoring하기, 왜냐하면 guardrail이 refactor가 보호되는 속성을 깨는 순간을 잡기 때문에; onboarding(새 인원 합류), 왜냐하면 rule이 아키텍처를 실행 가능하게 문서화하기 때문에. 그것은 **비싸게 만든다**: 검사를 작성하고 유지보수하기(fitness function은 그 자체로 썩거나 stale해질 수 있는 코드다), 임계값을 펄럭이지도 잠들지도 않게 튜닝하기, 그리고 위반을 ignore-list에 추가하기보다 *실제로 빌드를 실패시키는* 규율 — fitness-function 실패를 억제하기 시작하는 순간, 당신은 CI 배지를 단 열망을 다시 갖게 된다. evolutionary architecture의 정직한 가격은 *지속적인 enforcement*다: 그것은 일회성 setup이 아니라 상시 비용이며, 그 상시 비용이 정확히 시스템이 주위에서 변하는 동안 정직하게 유지되는 bet을 당신에게 사주는 것이다.

> 💡 **쉬운 설명:** fitness function의 가장 비싼 부분은 코드 작성이 아니라 *실제로 빌드를 실패시키는 규율*이다. 위반이 나올 때마다 ignore-list에 추가하기 시작하면, CI 배지만 달린 무늬뿐인 열망으로 되돌아간다. evolutionary architecture의 진짜 가격은 일회성 설치가 아니라 끊임없는 enforcement다.

### 6.5 The two halves are one discipline

strangler와 fitness function을 우연히 한 챕터를 공유하는 두 개의 무관한 도구로 읽고 싶을 수 있다. 그것들은 그렇지 않다. 그것들은 단일 속성 — bet을 revisable(개정 가능)하게 유지하기 — 의 두 얼굴이며, 서로를 직접 강화한다. fitness function은 strangler가 나중에 필요로 하는 seam을 *생성하고 보존하는* 것이다: dependency-isolation rule은 정확히 module의 boundary가 cross-module 호출로 채워지는 것을 막는 것이며, 이는 disintegrator(분해자) 압력이 언젠가 strangle을 정당화한다면 그 module을 *추출 가능하게* 유지하는 것이다. 함의를 거꾸로 돌리면 의존성은 대칭적이다: 진행 중인 strangler migration은 최대 구조적 churn(이탈)의 기간이다 — 새 서비스가 나타나고, data가 이동하고, façade routing이 바뀐다 — 이는 정확히 아키텍처가 가장 썩기 쉬운 때이며, 따라서 정확히 fitness function이 그 비용을 버는 때다. strangler는 잡을 깨끗한 seam이 필요하다; fitness function은 seam을 깨끗하게 유지한다; 그 seam을 사용하는 migration은 fitness function이 가장 중요해지는 순간이다. evolution은 하나의 capability이며, 두 timescale에서 측정된다 — 의도적 migration의 느린 timescale과 모든 commit의 빠른 timescale — 그리고 다른 하나 없이 하나를 하는 아키텍처는 절반만 보호된다.

> 💡 **쉬운 설명:** strangler와 fitness function은 한 동전의 양면이다. fitness function이 module 경계를 깨끗하게 지켜야 나중에 strangler가 그 경계를 깔끔하게 잘라낼 수 있다(잡을 seam을 만들어 둔다). 반대로 strangler migration이 한창일 때가 아키텍처가 가장 어지러워지는 때라, 바로 그때 fitness function이 부패를 막아 제값을 한다.

---

## 7. Applied to the Sales Agent: Which Seam to Strangle First, and What to Guard

학습자의 프로덕션 sales agent(Lina TMR)로 가져가자 — 많은 외부 SaaS tool API에 걸쳐 작동하는 LLM agent. 코스는 이 시스템을 깨끗한 bounded context(ch-04)와 깨끗한 internal dependency(ch-03)를 가진 **modular monolith**로 기본값을 정했다. evolution은 agent가 자라면서 그 기본값이 올바르게 유지되는 방법이다.

### 7.1 Choosing the first seam

strangler의 "깨끗한 boundary를 가진 하나의 capability를 골라라"는 ch-02가 식별한 bounded context — lead/pipeline, conversation, scheduling, CRM-sync — 위에 직접 매핑된다. evolution이 강제하는 질문은 *어느 seam을 먼저 추출할 것인가*이며, 답은 ch-04의 동일한 disintegrator analysis(분해자 분석)다: 단지 성가신 것이 아니라 특정 deploy/scale/team 압력이 distributed tax를 정당화하는 seam을 추출하라. Lina TMR의 경우 강력한 후보는 **CRM-sync** context다: 그것은 외부 SaaS API(Salesforce, HubSpot 등)에 가장 결합되어 있고, vendor가 느릴 때 독립적 scaling이 가장 필요할 가능성이 높으며, vendor가 API를 바꿀 때 독립적 deployment가 가장 필요할 가능성이 높은 부분이다. 결정적으로, Newman의 rule에 의해 당신은 CRM-sync의 *behavior와 그 data를 함께* 추출한다 — 외부 record의 synced snapshot이 코드와 함께 이동하므로, 새 서비스가 monolith의 데이터베이스로 손을 뻗는 대신 자기만의 store를 소유한다. 그리고 모든 외부 SaaS 응답이 **outside data**(ch-06)이기 때문에 — 불변(immutable)이고, versioned되며, 어쩌면 stale할 수 있는 snapshot이지, 결코 authoritative한 live state가 아니다 — seam은 자연스럽게 깨끗하다: agent의 *inside* model은 CRM-sync가 수집하는 것으로부터 이미 분리되어 있었으므로, 그것을 교살하는 것이 공유 가변 상태를 풀 필요가 없다. ch-06의 inside-vs-outside 규율이 여기서 strangler seam을 저렴하게 만드는 것이다; SaaS 응답을 live 공유 상태로 취급한 시스템이라면 자를 깨끗한 곳이 없었을 것이다.

> 💡 **쉬운 설명:** "어느 부분을 먼저 떼어낼까"의 답은 ch-04의 분해자 분석과 같다 — 단지 성가신 게 아니라 배포·확장·팀 압력이 분산 비용을 정당화하는 곳. Lina TMR에서는 CRM-sync가 후보다(외부 vendor에 가장 묶여 있고, 따로 확장·배포할 필요가 가장 크다). 게다가 외부 SaaS 응답은 이미 불변 snapshot(outside data)이라 공유 가변 상태가 없어 떼어내기 깔끔하다.

분석이 명시적으로 말하지 *않는* 것에 주목하라: 무언가를 추출하라고 말하지 않는다. 기본값은 ch-04의 modular monolith로 남으며, "어느 seam이 먼저인가"에 대한 정직한 답은 자주 "아직 아무것도 아니다"이다. 적당한 부하를 다루는 단일 팀 agent는 deploy-coupling 고통도, team-cognitive-load 고통도, service 분할이 완화할 scale asymmetry도 없다 — 그래서 CRM-sync를 추출하는 것은 유일한 정당화가 도착하지 않을 수도 있는 미래인 distributed system을 사는 것이다. strangler는 disintegrator 압력이 진짜가 될 때 손을 뻗는 *메커니즘*이지, 일정에 따라 추구할 목표가 아니다. "우리는 modular monolith로 머무르며, 여기 첫 strangle을 촉발할 특정 압력이 있다"를 기록하는 것 자체가 가격이 매겨진 ADR(Architecture Decision Record, 아키텍처 결정 기록)이다 — 그것은 seam을 명명하고, trigger를 명명하며, trigger가 발동될 때 팀을 되돌릴 수 있는 경로에 미리 약속시켜, 결정이 vendor outage가 마침내 문제를 강제할 때 공황 속에서가 아니라 미리 차분하게 내려지도록 한다.

> 💡 **쉬운 설명:** 중요한 반전: 이 분석은 "추출하라"고 말하지 않는다. 단일 팀이 적당한 부하를 다룬다면 정직한 답은 "아직 아무것도 떼지 마라"다. strangler는 분해자 압력이 *진짜*가 됐을 때 꺼내는 도구지, 달력 일정 목표가 아니다. 대신 "지금은 modular monolith를 유지하되, 이런 압력이 오면 CRM-sync부터 떼겠다"를 ADR로 미리 적어두면, 위기 때 공황이 아니라 평상시에 차분히 내린 결정이 된다.

trigger가 정말 발동되면, CRM-sync의 strangle은 네 단계 루프를 구체적으로 실행한다:

1. **Intercept.** agent core는 이미 in-process port(ch-03의 dependency inversion)를 통해 CRM-sync를 호출하므로 "façade"가 대부분 존재한다 — 움직임은 그 port를 routable하게 만들어 호출이 in-monolith 구현 *또는* 새 out-of-process 서비스로 갈 수 있게 하는 것이다. port가 항상 거기 있었기 때문에, step 1은 여기서 거의 공짜이며, 이것이 애초에 깨끗한 port를 가진 modular monolith를 지은 것의 payoff다.
2. **Extract one seam.** 자기만의 snapshot store를 소유하는 CRM-sync 서비스를 세우고, port를 통해 CRM-sync 트래픽만 그것으로 라우팅한다. outside-data 규율은 그것이 소유하는 snapshot이 불변 versioned 사본임을 의미하므로, 이동 중에 조율할 공유 가변 상태가 없다.
3. **Verify and shrink.** Parallel-run: 두 구현 모두 동일한 vendor record를 fetch하고 정규화하게 하고, 정규화된 snapshot을 비교하며, 일치한 후에만 권위를 새 서비스로 전환한다 — 그런 다음 in-monolith CRM-sync 코드를 삭제한다. Side-effecting 호출(CRM에 다시 쓰기)은 순진하게 이중 실행될 수 *없는* 것들이므로, 그것들은 병렬이 아니라 원자적으로 전환되며, 정확히 ch-06의 구분이 다시 표면화된다.
4. **Repeat or stop.** CRM-sync가 빠지면, 나머지 context에 disintegrator analysis를 다시 실행한다. 아무것도 bar를 통과하지 못하면, *의도적으로 멈추고* 그것을 기록하라 — 두 컴포넌트 시스템(agent monolith + CRM-sync 서비스)은 멈추는 것이 stall이 아니라 결정이었던 한, 불완전한 migration이 아니라 완벽하게 좋은 휴식 상태다.

전체 sequence는 CRM-sync를 첫 seam으로 한 [`figures/strangler-fig-timeline.html`](figures/strangler-fig-timeline.html) 동반 파일이 애니메이션하는 것이다 — 그것을 단계별로 밟고 step 2를 roll back하여 in-process port가 어떻게 agent의 첫 추출을 저렴하고 되돌릴 수 있게 만드는지 느껴보라.

> 💡 **쉬운 설명:** Lina TMR에서 step 1(Intercept)이 거의 공짜인 이유는, agent core가 이미 in-process port로 CRM-sync를 호출하고 있기 때문이다 — 즉 ch-04에서 깨끗한 port를 가진 modular monolith를 미리 지어둔 것이 여기서 보상으로 돌아온다. 또 step 4의 "멈춤"은 실패가 아니다. 두 조각짜리 시스템도, 그게 stall이 아니라 *결정*이었다면 완전히 정상적인 안식 상태다.

### 7.2 Fitness functions for the agent

이 시스템에서 지킬 가치가 있는 characteristic들을, fitness function으로 인코딩한 것:

- **The dependency rule on the agent core(agent core의 dependency rule).** domain policy(lead scoring, routing rule, conversation logic)가 특정 LLM-vendor SDK, vector-DB client, 또는 web framework를 결코 import하지 않는다는 ArchUnit-style 검사. 이것은 ch-03의 핵심 bet의 실행 가능한 형태다 — *되돌리기 비싼* 부분(어느 LLM API가 현재이든 그것을 outlive해야 하는 domain policy)을 *교체하기 저렴한* 부분으로부터 격리하여 유지하기. 한 model vendor에서 다른 것으로 마이그레이션하는 날, 이 rule이 migration이 core를 통한 수술이 아니라 edge에서의 swap임을 보장하는 것이다.
- **Resilience characteristics never regress(회복력 특성이 결코 퇴행하지 않음).** 모든 외부 SaaS integration point가 여전히 timeout과 circuit breaker를 갖는다는 CI 검사(ch-08과 연결) — 그래서 refactor가 한 느린 vendor가 전체 agent loop를 멈추게 하는 것을 막는 bulkhead(격벽)를 결코 조용히 제거할 수 없게.
- **Contract tolerance(계약 관용).** 각 SaaS integration에 대한 consumer-driven contract test(ch-05), agent가 vendor가 보장하지 않는 field에 의존하기 시작하면 빌드를 실패시킴 — tolerant-reader(관용적 독자) 자세를 강제 가능하게 만든 것.
- **Cost/latency budgets(비용/지연 예산).** 빌드-또는-경고-실패 임계값을 가진, agent loop 위의 p99 latency 또는 per-task cost monitor — agent의 economic characteristic을 가용성을 방어하듯 정확히 방어할 일급 속성으로 취급하기. LLM agent의 경우, per-task token cost는 있으면 좋은 metric이 아니다; 그것은 prompt가 자라고 tool call이 늘어나면서 가장 조용히 drift할 가능성이 높은 characteristic이며, 따라서 일회성 리뷰가 아니라 *continual* fitness function을 요구하는 바로 그런 종류의 속성이다.
- **Bounded-context isolation(경계 맥락 격리).** agent의 bounded context들(ch-02의 lead/pipeline, conversation, scheduling, CRM-sync)이 서로의 internal에 손을 뻗지 않는다는 ArchUnit-style 검사 — 오직 published in-process interface를 가로질러서만. 이것이 *modular monolith를 modular하게 유지하는* fitness function이다: 그것 없이는 strangler가 나중에 자를 seam이 cross-context 호출로 조용히 채워지고, ch-04의 "추출은 기계적이다" 약속이 증발한다. 여기서 fitness function은 오늘의 구조를 보호하는 것이 아니다; 그것은 *내일의 추출 선택권(option to extract)*을 보호한다 — evolvability가 evolvability를 지킨다.

> 💡 **쉬운 설명:** agent에서 지킬 다섯 가지 핵심 특성을 코드로 못 박는 것이다. 특히 마지막 두 개에 주목하라 — LLM agent에서 per-task 토큰 비용은 prompt가 길어지고 tool call이 늘면서 가장 조용히 새어나가는 특성이라 *연속* 감시가 필요하고, bounded-context 격리 검사는 "지금"이 아니라 "나중에 떼어낼 수 있는 선택권"을 지키는 것이다. 즉 evolvability가 evolvability를 보호한다.

agent 예시가 구체화하는 대칭에 주목하라: 위의 fitness function들은 *strangler seam을 나중에 사용할 만큼 깨끗하게 유지하는* 것이다. module boundary가 dependency-isolation fitness function에 의해 강제되는 modular monolith는 모든 bounded context가 추출 가능하게 유지되는 것이다; 그 강제가 없는 modular monolith는 깨끗한 seam이 살아남지 못하는 big ball of mud(거대한 진흙 공)로 저하되고, strangler는 잡을 것이 없다. 따라서 이 챕터의 두 절반은 독립적인 도구가 아니다 — fitness function은 strangler가 저렴하게 유지되는 조건을 *보존하는* 것이다. Migration 규율과 enforcement 규율은 bet을 revisable하게 유지하는 하나의 시스템이다.

payoff framing: 이전 코스는 학습자에게 agent를 *benchmark*하는 법을 가르쳤다; 이 코스는 그것을 *architect*하는 법을 가르친다 — 그리고 이 챕터는 아키텍처가 capstone(ch-10)이 만들 design memo(설계 메모)로부터 멀어지는 것을 막는 부분이다. fitness function 없는 design memo는 snapshot이다; 그것과 함께라면, 그것은 실행되는 시스템이 지켜야 하는 contract다.

---

## Where This Goes

이 챕터는 척추(spine)의 고리를 닫았다: 아키텍처는 되돌리기 비싼 결정들이며, evolution은 "되돌리기 비싼"이 "되돌릴 수 없는"으로 붕괴하지 않게 하는 방법이다. strangler-fig는 bet을 *변경하는 것*을 점진적이고 되돌릴 수 있게 만든다; fitness function은 *변경되지 않은* bet이 썩지 않게 한다. 둘 다 되돌릴 수 없는 집합을 줄이며, 마침내 ch-01의 Fowler의 수정안을 현금화한다.

이 챕터가 중심으로 삼은 단일 trade-off는 evolution phase 전체를 지배하는 것이다: **당신은 지속적이고 끊임없는 비용을 — migration 동안 두 시스템을 운영하기, 영구적인 façade를 유지하기, fitness function을 작성하고 튜닝하기, 빌드를 실패시키고 ignore-list를 거부하기 — 지불하여, 이미 놓은 bet을 얼어붙은 대신 revisable하게 유지한다.** 코스의 다른 모든 패턴은 어떤 유연성을 비용으로 당신에게 속성을 사주었다; evolution은 유연성 그 자체를 되사주며, 그 가격은 청구서가 결코 멈추지 않는다는 것이다. 그 가격을 지불할 가치가 있는지는, 다른 모든 것처럼, First-Law trade-off다 — 일회용 스크립트에는 그것은 순수한 overhead이고, 학습자의 오래 사는 프로덕션 agent에는 그것은 vendor들을 outlive할 수 있는 아키텍처와 조용히 되돌릴 수 없는 잘못된 답이 되는 아키텍처 사이의 차이다.

> 💡 **쉬운 설명:** 이 챕터의 단 하나의 trade-off는 이렇다 — "끝나지 않는 비용(두 시스템 운영·façade 유지·fitness function 관리·빌드 실패 감수)"을 계속 내는 대신, 이미 내린 결정들을 *얼리지 않고 바꿀 수 있는 상태*로 유지한다. 일회용 스크립트엔 낭비지만, 오래 살아남아야 할 Lina TMR 같은 시스템엔 vendor보다 오래 사는 아키텍처와 조용히 망가지는 아키텍처를 가르는 차이다.

Ch-10은 capstone lab이다: 그것은 전체 도구상자를 — bounded context(ch-02), 깨끗한 internal structure(ch-03), modular-monolith-first topology(ch-04), integration contract(ch-05), inside-vs-outside data와 saga(ch-06), 선택적 power tool로서의 CQRS/ES(ch-07), integration point에서의 resilience(ch-08), 그리고 이 챕터의 strangler-fig-and-fitness-function evolution 규율 — 취하여, 그것들 하나하나를 프로덕션 sales agent를 위한 *ADR로 기록된 가격 매겨진 bet* 더하기 *C4 Context/Container 스케치*가 되도록 강제한다. 산출물은 Lina TMR을 위한 진짜 design memo다: 모든 결정이 명명되고, 모든 비용이 진술되며, 모든 보호되는 characteristic이 그것을 정직하게 유지할 fitness function을 지니고 있다.
