<!-- chapter: ch-29c
     track: synthetic
     kind: content
     title: Agentic Environment and Task Synthesis at Scale
     deps: [ch-29b, ch-27, ch-32d]
     sources: [[swe-gym]], [[excerpts/swe-gym-v2-extract]], [[r2e-gym]], [[swe-smith]], [[swe-rebench]], [[glm-5]], [[kimi-k2]], [[agentscaler]], [[toucan-mcp]], [[deepseek-v3.1]], [[webshaper]], [[quest-deep-research]], [[tongyi-deepresearch]], [[agent-data-protocol]], [[agent-world]], [[reasoning-gym]], [[swe-bench-illusion]], [[swe-bench-pro]], [[agentic-benchmark-checklist]], [[browsecomp-plus]]
     figures: figures/task-funnels-and-filters.html
     revised: 2026-09 (generality revision)
-->

# Chapter 29c — 대규모 에이전트 환경 및 작업 합성

> **핵심 통찰.** 에이전트 학습에서 희소한 자원은 trajectory 자체가 아니라, 실제로 실행되는 환경 안에 있고 정답 여부를 검증할 수 있는 작업이다. 파이프라인이 사람이 환경을 설정하던 방식(SWE-Gym: 작업 2,438개에 주석 작업 약 200시간, [[excerpts/swe-gym-v2-extract]] §3.1)에서 자동 설치와 합성 작업 방식(SWE-smith: 작업 50,137개에 사람 노동 약 20시간 및 $1,360, [[swe-smith]] §2.1-2.2)으로 옮겨가면서, 검증 가능한 작업 하나를 만드는 비용은 내려갔다. 환경의 폭은 범용성을 조절하는, 실제로 측정된 변수다. Agent-World에서는 훈련 환경이 0개일 때 네 도메인 도구 사용 평균이 18.4%였고 1,978개일 때 38.5%였다([[agent-world]] Fig. 8). SWE-smith에서는 trajectory를 700개로 고정한 채 훈련 저장소를 4개에서 100개로 늘리자 resolve rate가 10.3%에서 15.1%로 올랐다([[swe-smith]] Fig. 5). 합성한 일반 에이전트 작업만으로 DeepSeek-V3.2-SFT에 RL을 수행하면 held-out Tau2Bench, MCP-Mark, MCP-Universe가 개선됐지만, 검색 및 코드 환경으로 제한한 RL은 그러지 못했다([[deepseek-v3.1]] §4.3, Fig. 5). 다만 coverage는 substrate에 제한된다. Python만 사용한 SWE-smith 모델의 SWE-bench Multilingual 점수는 8.4%로, base model의 6.5%보다 높았을 뿐이다([[swe-smith]] App. F.4).
>
> **지침.** agentic SFT나 RL용 작업을 만들 때는 verifier를 먼저 작성한다. verifier는 gold patch 전에는 실패하고 뒤에는 통과하는 테스트, 데이터베이스 최종 상태, 검증 함수, rubric tree가 될 수 있다. reference solution은 통과하고, 빈 답이나 지름길 해법은 실패할 때만 작업을 남긴다. 검증하지 않은 작업은 잘못된 보상을 만들기 때문이다. 빈 응답도 τ-bench Airline 작업의 38%를 통과하며([[agentic-benchmark-checklist]] §5.2), 결함 있는 테스트는 올바른 에이전트를 감점한다([[swe-rebench]] §2.3). trajectory 예산을 고정한 상태에서 폭을 늘리는 목적이라면, 작업 하나당 trajectory 수를 늘리기 전에 서로 다른 환경·저장소·작업 계열을 추가한다. SWE-smith의 저장소 비교는 trajectory 700개를 고정했고([[swe-smith]] Fig. 5), ADP 비교도 sample count를 같게 맞췄다([[agent-data-protocol]] Table 10). Agent-World는 환경 수별 RL sample 수를 고정했는지 밝히지 않았다. 목표가 실제 API 사용을 포함하면 실제로 실행되는 저장소나 실제 서버 substrate를 넣는다. 시뮬레이션 환경으로 학습한 Simulator-8B는 τ²-Bench에서 31.8이지만 MCP-Mark에서는 2.4였기 때문이다([[agent-world]] Table 1; 단일 연구). 반대로 배포 checkpoint가 아니라 알고리즘 반복 실험이 목적이라면 rollout당 비용이 더 낮은 시뮬레이션 또는 offline substrate를 쓴다. Tongyi DeepResearch가 offline Wikipedia 환경으로 한 방식이다([[tongyi-deepresearch]] §3.4.3). benchmark 점수로 일반 에이전트 능력을 주장할 때는 해당 benchmark의 저장소·서버·작업 template을 환경 pool에서 빼고, 학습 cutoff 뒤에 만든 작업의 두 번째 점수도 보고한다. SWE-Bench 밖의 동등하게 인기 있는 저장소에서는 모델의 file-path 정확도가 60–76%가 아니라 53% 미만이며([[swe-bench-illusion]] §4.1), SWE-rebench는 issue가 모델 출시일보다 앞선 결과에 오염 가능성 표지를 붙인다([[swe-rebench]] §3.2).

## 범용 모델에서 이 장이 중요한 이유

범용 모델의 파이프라인은 사전학습, 중간 학습, 지도 미세조정(SFT), 선호 최적화, 강화학습(RL), 평가의 여섯 단계로 구성된다. 에이전트 행동은 중간 학습에는 trajectory 데이터로([[ch-32d]]), SFT에는 필터링한 teacher trajectory로([[ch-27]]), RL에는 환경이 점수화한 rollout으로 들어간다([[ch-45b]]). 세 단계 모두 같은 입력이 필요하다. 즉, 행동을 실행하는 환경과 결과를 점수화하는 check가 각각 짝지어진 대규모 작업 집합이다. 이 장은 그 입력을 생산하는 방법을 다룬다.

측정 가능한 문제는 세 부분으로 나뉜다. (1) **수율(yield)**: 후보 작업 중 실행 가능하고 검증 가능한 작업으로 살아남는 비율, 그리고 살아남은 작업 하나당 사람 및 compute 비용. (2) **verifier 오류**: 작업 check가 틀린 해법을 통과시키는 비율(false positive), 또는 맞는 해법을 탈락시키는 비율(false negative). (3) **coverage**: 서로 다른 환경·언어·도구 생태계·작업 계열의 수, 그리고 이들로 학습한 결과가 학습에 쓰지 않은 환경의 점수까지 바꾸는지 여부다. 앞 장 [[ch-29b]]는 context가 누적되는 긴 대화를 합성했다. 이 장은 에이전트가 상태를 바꾸는 환경을 그 대화에 더한다. 다음 장 [[ch-29d]]는 작업이 생긴 뒤의 simulated user, trajectory 검증, 실패한 trajectory 처리 방법을 다룬다.

## §1 용어와 substrate 분류

**정의.**
- **환경(environment)**은 에이전트 행동(도구 호출, shell 명령, 파일 편집)을 받아 observation을 돌려주면서, 뒤 행동이 읽을 수 있는 상태를 보유하는 프로그램이다. [[agent-world]] §2는 환경을 database `D`와 toolset `F`로 이루어진 `e = (D, F)`로 형식화하며, 상태는 tool output을 통해서만 관측된다.
- **작업(task)**은 instruction과 초기 환경 상태의 쌍이다.
- **verifier**는 최종 환경 상태, 에이전트 답, 또는 양쪽을 score로 보내는 함수다. 예로 gold patch 전에는 실패하고 gold patch 후에는 통과하는 Fail-to-Pass(F2P) 테스트, 데이터베이스 최종 상태 비교, Python 검증 함수, rubric tree가 있다.
- **substrate**는 환경이 실행되는 시스템의 종류다.

**다루는 문제.** 환경마다 fidelity, 즉 observation이 실제 배포 시스템과 얼마나 가까운지가 다르고, 작업당 비용도 다르다. 여기에는 setup 시간, 저장 공간, latency, API fee가 포함된다. 하나의 substrate만 쓰는 학습 run은 그 substrate 고유의 실패 양식을 함께 물려받는다.

**분류.** Tongyi DeepResearch는 세 형식을 설명한다. 실제 응답 없이 작업·도구·상태 정의를 제공하는 prior-world 환경은 “완전한 안정성, 상호작용 비용 0”이지만 “현실 피드백 신호가 없다.” simulated 환경은 “현저한 sim-to-real gap”이 있고, real-world 환경은 “비싼 상호작용, 큰 비정상성, 탐색 위험”을 갖는다([[tongyi-deepresearch]] §2). 아래 표는 이 장의 출처가 사용한 substrate를 여섯 종류로 더 세분화한다.

| Substrate | 예 | observation 출처 | verifier | 작업당 비용(보고된 값) | fidelity의 주된 한계 |
|---|---|---|---|---|---|
| 실제 실행 가능 저장소 | SWE-Gym, R2E-Gym, SWE-smith, SWE-rebench, GLM-5 SWE 및 terminal | 코드와 테스트를 실행하는 Docker container | F2P 및 pass-to-pass 테스트 | SWE-Gym: 작업 2,438개에 image 6 TB; SWE-smith: 50k에 295 GB | 테스트 품질, flaky test, 언어 coverage |
| 실제 MCP / API 서버 | TOUCAN(서버 495개) | live remote server | LLM judge + 규칙 check | API key가 필요한 서버는 제외 | 비정상적 출력, 가용성 |
| 상태 보유 시뮬레이션 도구(LLM) | Kimi K2 tool simulator | 상태를 추적하는 LLM | 작업 rubric에 대한 LLM judge | 보고되지 않음 | hallucination 또는 일관성 없는 응답 |
| 프로그램형 database 환경 | AgentScaler, DeepSeek-V3.2 general agent, Agent-World | database를 읽고 쓰는 Python tool | 최종 상태 일치, tool sequence 일치, 검증 함수 | 보고되지 않음 | schema 현실성, 합성 tool의 도달 범위 |
| offline search corpus | Tongyi 2024 Wikipedia 환경; BrowseComp-Plus corpus | local retrieval tool | 답 일치 | 보고되지 않음 | corpus coverage 및 retriever 선택 |
| 절차적 generator | Reasoning Gym(100개 초과 generator) | 없음(단일 단계) 또는 game state | 알고리즘적 check | 필요할 때 생성 | 코딩할 수 있는 작업 계열로 제한 |

출처: [[excerpts/swe-gym-v2-extract]] §3.1; [[swe-smith]] Table 2; [[toucan-mcp]] §3.1; [[kimi-k2]] §3.1.1; [[agentscaler]] §2; [[deepseek-v3.1]] §3.2.3; [[agent-world]] §3.1; [[tongyi-deepresearch]] §3.4.3; [[browsecomp-plus]]; [[reasoning-gym]] §2.

**fidelity에 관한 근거.**
- Kimi K2는 도구 실행 뒤마다 상태를 “유지하고 갱신”하며 “제어된 stochasticity”를 도입하는 simulator로 SFT trajectory를 만든다. 동시에 진정성이 중요한 scenario를 위해 coding 및 software engineering에는 실제 sandbox도 추가한다([[kimi-k2]] §3.1.1). 논문은 이 두 부분을 분리하는 ablation을 제시하지 않는다.
- Agent-World Table 1에서 simulated environment로 학습한 baseline Simulator-8B는 τ²-Bench 31.8, BFCL V4 23.9, MCP-Mark 2.4를 얻는다. 같은 표의 학습하지 않은 Qwen3-8B는 각각 26.2, 40.4, 2.4다([[agent-world]] Table 1). 논문은 Simulator-8B가 어느 base model에서 학습됐는지 밝히지 않았다. 저자들은 시뮬레이션 환경이 실제 상태 전이를 포착하지 못한 결과로 해석한다. 다만 두 baseline은 데이터와 방법이 모두 다르므로 통제된 비교는 아니다(해석).
- Tongyi DeepResearch는 offline Wikipedia 환경의 RL reward curve가 실제 환경 curve와 “가깝게 일치한다”고 보고하고(§4.4, Fig. 10b 대 Fig. 8), 이 offline 환경을 빠른 알고리즘 반복용 platform이라고 설명한다(§4.4). 공개 모델의 최종 RL이 이 환경을 사용했는지는 밝히지 않았다.

**범용 모델에 대한 함의.** 단일 substrate는 배포 조건의 일부만 포괄한다. 실제 서버와 실행 가능한 저장소는 실제에 가까운 실패를 제공하고, 프로그램형 및 offline 환경은 규모와 재현성을 제공한다. held-out agent 결과가 가장 강하게 보고된 출처는 substrate를 결합한다(§4, §6).

## §2 SWE 환경 factory 계보

**정의.** SWE 환경 factory는 저장소를 runnable container와 함께 `(저장소 snapshot, issue text, hidden test)` 형태의 작업으로 바꾸는 파이프라인이다.

**문제.** 사람이 환경을 설정하면 규모를 키울 수 없다. SWE-Gym은 358개 저장소에서 raw 작업 instance 64,689개를 추출했지만, 실행 가능한 환경으로 바꾼 저장소는 11개뿐이었다. 검증된 instance 2,438개에는 “사람 주석 작업 약 200시간과 CPU core 시간 10,000시간”이 들었다([[excerpts/swe-gym-v2-extract]] §3.1).

**세대별 메커니즘.**
1. **SWE-Gym (arXiv 2024-12).** 실제 issue와 PR을 쓴다. task instance마다 dependency를 사람이 수동으로 설정한다. validation은 gold patch가 원래 코드보다 더 많은 test를 통과해야 한다는 조건을 요구한다. 저장소는 “오염을 피하기 위해” SWE-Bench와 겹치지 않게 골랐다(§3). trajectory 491개로 Qwen-2.5-Coder-32B를 fine-tuning하자 SWE-Bench Verified가 7.0%에서 20.6%로 올랐다(Table 3).
2. **R2E-Gym (arXiv 2025-04).** SWE-Gen은 issue가 아니라 commit에서 환경을 만든다. 기존 F2P test를 수집하거나 생성하고, code change를 problem statement로 back-translate한다([[r2e-gym]] §2). 이 방식으로 문제 8,135개와 SWE-Bench와 repository overlap이 없는 문제 4,578개 subset을 얻었다(Table 1). 각각 trajectory 400개를 사용했을 때 synthetic problem statement는 Pass@1 27.8%, 실제 issue는 28.0%였다(Fig. 3).
3. **SWE-smith (arXiv 2025-04).** 순서를 뒤집는다. 즉, “먼저 실행 환경을 정의하고, 그 다음 환경 안에서 task instance를 합성한다”([[swe-smith]] §2). repository당 Docker image 하나를 둔다. SWE-agent가 최대 100 step 안에 설치하고, 사람이 이를 확인한다. bug는 LM rewrite, AST transformation, PR reversal, bug combination으로 주입한다. 후보는 적어도 하나의 원래 통과 test를 깨뜨릴 때만 남긴다.
4. **SWE-rebench (arXiv 2025-05).** 설치를 자동화한다. task를 version별로 묶고 Qwen2.5-72B-Instruct가 task당 JSON installation recipe 후보를 최대 3개 작성한다. error log를 받아 recipe를 수정한다. task가 유효하려면 test-patch test가 solution patch 전에는 실패하고, 그처럼 실패한 test 모두가 solution patch 뒤에는 통과하며, 처음에 통과한 test는 계속 통과해야 한다. flaky task를 빼기 위해 test를 다시 실행한다([[swe-rebench]] §2.2-2.3).
5. **DeepSeek-V3.2와 GLM-5 (2025-12, 2026-02).** environment-setup agent가 방법을 더 많은 언어로 확장한다. DeepSeek-V3.2는 Python, Java, JavaScript, TypeScript, C, C++, Go, PHP 전반에서 gold patch의 F2P > 0 및 pass-to-fail = 0일 때만 환경이 구축됐다고 세고, code-agent RL 작업 24,667개를 사용한다([[deepseek-v3.1]] §3.2.3, Table 1). GLM-5는 LLM이 작성한 log parser가 있는 RepoLaunch 기반 파이프라인으로 “9개 프로그래밍 언어에 걸쳐 수천 개 repository에서 검증 가능한 환경 10k개 초과”를 구축한다([[glm-5]] §4.2.1).

**계산 예: 살아남은 작업 하나당 비용.** 아래 값은 논문에 인쇄된 총량으로 계산했다.

| Pipeline | 사람 시간 | 살아남은 작업 | 작업당 사람 시간 | 작업당 저장소 |
|---|---:|---:|---:|---:|
| SWE-Gym | 약 200 h | 2,438 | 200 × 60 / 2,438 = 4.9 min | 6 TB / 2,438 = 2.5 GB |
| SWE-smith | 약 20 h | 50,137 | 20 × 3,600 / 50,137 = 1.4 s | 295 GB / 50,137 = 5.9 MB |

SWE-smith의 저장소 절약은 task당 image 하나가 아니라 repository당 image 하나를 공유하는 데서 나온다. 저자들은 같은 수의 SWE-bench-style image라면 50–150 TB가 필요하다고 추정한다([[swe-smith]] §2.2). 이 절약에는 비용이 있다. 하나의 repository에서 나온 작업은 dependency snapshot 하나를 공유하므로, 그 snapshot 이전 code version은 포괄하지 못한다.

**수율(yield).** SWE-rebench는 약 450,000개 issue-linked PR 중 약 153,400개를 task filter 뒤에 남긴다(34.1%). 그 가운데 설치와 validation 뒤에는 21,336개가 남는다(13.9%). 그러므로 candidate 중 4.7%가 살아남는다(§2.1, §2.4; 비율은 계산값). SWE-smith에서 test를 깨뜨린 candidate의 전체 수율은 50.1%이고, PR Mirror의 33.8%부터 Combine의 96.9%까지 범위가 있다(Table 1). companion figure [task-funnels-and-filters.html](figures/task-funnels-and-filters.html)은 SWE-Gym, SWE-rebench, TOUCAN, QUEST funnel을 단계별로 보여 주며, 단계 사이 retention과 누적 retention을 확인하게 한다.

**합성 작업이 실제로 갖는 학습 가치를 보여 주는 근거.**
- Strategy별로 instance 1,000개, student당 trajectory 507개, Qwen 2.5 7B를 사용한 비교에서 SWE-bench Verified는 PR Mirror 9.2%, LM Rewrite 8.8%, Procedural 8.6%, LM Modify 5.7%였다([[swe-smith]] Table 4). Procedural bug의 candidate당 비용은 0.00¢였고 PR Mirror는 5.53¢였다(Table 1).
- Issue text별로 trajectory 259개씩 쓴 비교에서 original issue는 7.8%, LM-generated issue는 7.7%, F2P test를 issue로 쓴 것은 7.3%, fixed template는 6.4%였다(Table 5). test-based issue를 받으면 student는 500회 run 중 127회에서 bug 재현을 시도했지만, LM-generated issue에서는 379회였다. 저자들은 evaluation criterion이 prompt에 새어 들어간 탓이라고 해석한다(Interpretation).
- 난이도는 학습 가치를 예측하지 못했다. difficulty 2/4/6/8 SFT set의 점수는 각각 12.4 / 10.8 / 13.6 / 12.2%였다(§4.1).

**조건과 한계.** SWE-smith와 SWE-Gym의 결과는 모두 Claude 또는 GPT teacher를 쓴 rejection-sampling SFT다. 두 논문 모두 그 환경에서 RL을 학습하지 않았다. SWE-rebench는 agent를 학습하지 않았으므로 validation filter가 학습에 미치는 효과를 측정하지 않았다([[swe-rebench]] Guideline).

**범용 모델에 대한 함의.** 자동 설치와 bug 합성은 repository 및 language 수를 제어 가능한 변수로 바꾼다. §6은 trajectory 수만이 아니라 이 변수가 측정된 transfer가 나오는 지점임을 보인다.

## §3 코드 밖으로 도구와 작업을 다양화하기

**정의.** 도구 다양화는 호출할 수 있는 도구의 수와 종류, 그리고 그 도구들이 속하는 도메인의 범위를 늘리는 일이다. 작업 다양화는 그 도구들로 구성하는 목표의 종류를 늘리는 일이다.

**문제.** 도구 사용 모델은 처음 보는 도구에서 실패하고, 여러 도구를 순서대로 써야 하는 작업에서도 실패한다. 실제 API는 자격 증명, 가용성, 비용에 제약을 받는다. TOUCAN은 약 2,800개 MCP 서버를 크롤링했지만, 원격에서 API key 없이 쓸 수 있는 서버 871개(30.6%)만 남겼고, 모든 도구를 실제로 시험한 뒤에는 495개만 남겼다([[toucan-mcp]] §3.1).

**메커니즘.**
1. **실제 도구와 합성 도구를 계층적 도메인 진화로 결합하기(Kimi K2).** Kimi K2는 GitHub 저장소에서 “3,000개가 넘는 실제 MCP(Model Context Protocol) 도구를 직접 가져오고”, “계층적 도메인 생성 과정으로 합성 도구를 체계적으로 진화”시킨다. 먼저 금융 거래, 소프트웨어 애플리케이션, 로봇 제어 같은 핵심 범주에서 시작한 뒤, 각 범주 안에서 여러 구체적 응용 도메인을 진화시켜 20,000개가 넘는 합성 도구를 만든다([[kimi-k2]] §3.1.1). system prompt와 도구 조합으로 수천 개 agent를 만들며, 각 작업에는 성공 기준, 예상 도구 사용 패턴, checkpoint를 담은 rubric이 붙는다.
2. **데이터베이스 기반 환경을 확장하기(AgentScaler).** AgentScaler는 모든 함수를 도메인 데이터베이스의 읽기 또는 쓰기 연산으로 취급한다. 즉 `API(func, α) ≡ op(func)(α; D)`이다([[agentscaler]] §2). 30,000개가 넘는 API를 그래프로 연결하고 Louvain community detection으로 1,000개가 넘는 도메인으로 나눈 뒤, 각 도구를 도메인 데이터베이스 위에서 실행되는 Python 코드로 다시 쓴다(§2.1).
3. **실제 서버에서 작업 만들기(TOUCAN).** 공개 모델 다섯 개가 서버 하나, 여러 서버, 또는 선정한 25개 서버를 대상으로 작업을 쓴다. Kimi-K2는 작업의 난이도, 고유성, 품질, 현실성, 검증 가능성, 안정성을 평가하며, 교사 모델 세 개가 live server를 상대로 trajectory를 생성한다(§3.1). 관련 없는 도구를 견디는지를 위한 extension은 server metadata를 섞어, 주어진 도구만으로는 작업을 풀 수 없게 만든다. 그리고 도구 호출 수가 0인 trajectory만 남긴다(§3.2).
4. **환경 합성 agent(DeepSeek-V3.2, Agent-World).** DeepSeek-V3.2의 agent는 작업 category와 bash·search가 있는 sandbox를 받고, 검색한 데이터를 데이터베이스에 저장하며, 작업별 도구를 작성한다. 이어서 Python으로 구현한 “해결 함수와 검증 함수를 함께” 제공하는 작업을 제안한다. 이때 “해결 함수는 도구 함수를 호출하거나 논리 계산을 수행하는 일로 제한되고, 다른 함수를 호출하거나 데이터베이스에 직접 접근할 수 없다”([[deepseek-v3.1]] §3.2.3). Agent-World는 deep-research agent로 데이터베이스를 발굴하고, 컴파일되며 unit test의 50% 초과를 통과한 도구만 남긴다. 그 뒤 도구 의존성 그래프 위의 random walk로 작업을 조합한다([[agent-world]] §3.1).

**수식: AgentScaler 도구 그래프의 edge** ([[agentscaler]] §2.1, 식 1).

```
E = { (i, j) | sim( φ(P_func_i), φ(P_func_j) ) > τ ,  i ≠ j }
```

- `P_func_i`는 도구 i의 parameter list이고, `φ`는 이를 vector로 사상한다. `sim`은 cosine similarity다.
- `τ`는 similarity threshold다. 논문은 그 값을 인쇄하지 않았다.
- edge는 두 도구를 조합할 수 있음을 뜻한다. 그 뒤 LLM이 각 도메인 안의 의존성을 다시 확인한다.

**계산 예.** `φ`가 도구 A에는 vector `(1, 0, 1)`, 도구 B에는 `(1, 1, 0)`을 준다고 하자. dot product는 1이고 두 norm은 각각 `√2`이므로 `cos = 1 / 2 = 0.5`다. 예시 threshold `τ = 0.6`이면 edge를 넣지 않는다. `τ = 0.4`이면 A와 B는 같은 candidate domain에 들어간다. 따라서 threshold는 Louvain이 찾는 도메인 수와 조합 가능한 도구 chain의 길이를 결정한다.

**계산 예: Agent-World의 가중 walk.** Agent-World는 강한 의존성에는 edge weight 3, 약한 의존성에는 2, 독립적인 pair에는 1을 부여하고, “edge weight에 bias를 둔 probability distribution”으로 다음 도구를 표본 추출한다([[agent-world]] §3.1.1). 논문은 normalization을 인쇄하지 않았지만, distribution이 weight에 비례한다고 가정하면 strong successor 하나, weak successor 하나, independent successor 하나가 있을 때 각각으로 이동할 확률은 `3/6 = 0.50`, `2/6 = 0.33`, `1/6 = 0.17`이다. Agent-World는 weak edge와 independent edge의 sampling probability를 높여 난이도를 올린다. 그러면 출력이 덜 직접적으로 연결된 도구들을 잇는 chain이 길어진다.

**근거.**
- AgentScaler-30B-A3B(Qwen3-Thinking-30B-A3B backbone)는 τ²-Bench Telecom을 26.3에서 55.3으로, ACEBench-en Overall을 67.2에서 75.7로 올렸다([[agentscaler]] Table 1). 저자들이 out-of-distribution이라고 부르는 ACEBench-zh에서는 Overall이 74.2에서 81.5로 올랐다(Table 2).
- 119.3K instance로 한 TOUCAN SFT는 Qwen2.5-32B-Instruct의 BFCL V3 Overall을 61.73에서 70.45로, multi-turn을 26.38에서 46.50으로 올렸다([[toucan-mcp]] Table 2).
- DeepSeek-V3.2는 V3.2에서 pass@100이 0이 아닌 합성 general-agent 작업만 남겼다. 그 결과 환경 1,827개와 작업 4,417개가 남았다. 표본 작업 50개에서 pass@1은 DeepSeek-V3.2-Exp 12%, Sonnet-4.5 34%, Gemini-3.0 Pro 51%, GPT-5-Thinking 62%였다([[deepseek-v3.1]] §3.2.3, Table 5).

**조건과 한계.** AgentScaler는 learning rate, epoch, data size를 보고하지 않았고 30B까지 SFT만 학습한다([[agentscaler]] Limitation). Kimi K2는 합성 도구와 실제 도구를 분리한 ablation을 보고하지 않는다. AgentScaler, TOUCAN, Kimi K2 어느 쪽도 자기 도구 pool과 평가 benchmark 사이 overlap check를 보고하지 않는다(§7).

**범용 모델에 대한 함의.** 도구 다양성은 두 가지 서로 다른 작업에서 나온다. 하나는 실제 도구를 수집하는 일이며, 존재하고 접근 가능한 도구에 의해 범위가 제한된다. 다른 하나는 합성 도구를 진화시키는 일이며, generator가 아는 도메인에 의해 범위가 제한된다. 둘 중 어느 방식이 처음 보는 도구 ecosystem으로 transfer하는지에 대한 근거는 합성을 유도한 환경 benchmark가 아니라 MCP-Mark, MCP-Universe, ACEBench-zh 같은 held-out suite에서 나온다.

## §4 Verifier-first 작업 설계

**정의.** verifier-first 설계는 작업을 pool에 넣기 전에 검사를 먼저 작성하고 검증하는 방식이다. 검사를 검증할 수 없는 작업은 버린다.

**문제.** verifier의 오류는 곧 reward의 오류가 된다. benchmark 측에서는 빈 응답이 τ-bench Airline 작업의 38%를 통과하며, SWE-Lancer의 test file을 덮어쓰는 agent는 100% 점수를 받는다([[agentic-benchmark-checklist]] §5.2). 학습 측에서는 SWE-Bench Pro가 required-behavior 및 interface field를 제거했을 때 GPT-5 (high)의 점수가 25.9%에서 8.40%로 떨어졌다고 보고했으며, 저자들은 이를 verifier false negative 탓으로 해석한다([[swe-bench-pro]] §6.2, Table 3). 즉 실제로 맞는 해결책을 검사기가 틀렸다고 판정하면, 그 해결책을 낸 모델도 낮은 보상을 받는다.

**네 가지 verifier 계열과 작업 보존 전의 검증 규칙.**

| 계열 | 작업을 남기기 전에 수행하는 검증 | 출처 |
|---|---|---|
| Hidden test | gold patch에서 F2P > 0이고 pass-to-fail = 0인지 확인한다. flaky test를 제거하려 재실행하고, 주입한 bug가 기존에 passing이던 test를 1개 이상 깨야 한다. | [[deepseek-v3.1]] §3.2.3; [[swe-rebench]] §2.3; [[swe-smith]] §2.1 |
| Database state 또는 verification function | reference solution이 verification function을 통과해야 하며, 통과할 때까지 function을 고친다. solution path는 도구 호출만 써야 하고, 최종 database state 일치 또는 정확한 tool sequence 일치로 확인한다. | [[deepseek-v3.1]] §3.2.3; [[agentscaler]] §3.1; [[agent-world]] §3.1.1 |
| Search의 answer check | ground truth가 맞는지 검증하고, 다른 agent들이 낸 모든 candidate answer가 검증 가능하게 틀렸는지 확인한다. 유일한 답이 아닌 경우는 버린다. | [[deepseek-v3.1]] §3.2.3; [[glm-5]] §4.2.3 |
| Open-ended 작업의 rubric tree | tree를 다듬고 검증하며, 실행 가능한 Python evaluation script를 만든다. tree를 일관되게 만들 수 없는 작업은 버린다. | [[quest-deep-research]] §2.2 |

**메커니즘: 난이도 filter.** 대부분의 pipeline은 모델이 작업을 얼마나 자주 푸는지도 filter로 쓴다. 작업 하나에서 모델이 한 번 시도해 풀 확률을 `p`라고 하고, 시도들이 서로 독립이라고 가정하자. 이는 단순화이며 실제 시도들은 상관되어 있다.

```
keep_pass@k>0      = 1 − (1 − p)^k                    (DeepSeek-V3.2, k = 100)
keep_≥2_of_5       = 1 − (1 − p)^5 − 5p(1 − p)^4       (Agent-World)
keep_mixed_G       = 1 − p^G − (1 − p)^G               (Tongyi DeepResearch; G not printed)
keep_unsolved_8    = (1 − q)^8                         (GLM-5 stage 1, q = tool-free solve rate)
```

- `k`, `5`, `G`, `8`은 각 규칙에서 시도하는 횟수다. `q`는 도구 없이 푸는 모델의 정답 확률이다.

**계산 예.** `p = 0.01`인 작업은 DeepSeek-V3.2 규칙에서 `1 − 0.99^100 = 0.634`의 확률로 살아남는다. `p = 0.001`이면 생존 확률은 0.095다. Agent-World 규칙에서는 `p = 0.10` 작업이 `1 − 0.9^5 − 5(0.1)(0.9)^4 = 0.081`, `p = 0.30` 작업이 0.472의 확률로 살아남는다. Tongyi 규칙에서 `G = 8`이면 `p = 0.10`은 0.570, `p = 0.50`은 0.992다. GLM-5 첫 단계에서는 도구 없는 모델이 `q = 0.10` 확률로 맞히는 질문이 `0.9^8 = 0.430` 확률로 남는다. 따라서 같은 candidate pool이라도 이 규칙들은 서로 다른 난이도 대역을 고른다. pass@100은 드물게라도 성공하는 작업을 남기고, ≥2-of-5는 그런 작업을 제거하며, GLM-5 규칙은 parametric knowledge만으로 답할 수 있는 작업을 제거한다. 동반 그림의 Panel B는 네 곡선을 그리고 `p`, `k`, `G`를 바꿔 볼 수 있게 한다.

**수식: QUEST reward** ([[quest-deep-research]] §4.4, Eq. 1).

```
R = 0.75 · s_rubric + 0.25 · min(s_fact, s_rubric)
```

- `s_rubric ∈ [0, 1]`은 rubric-tree 점수다. open-ended 작업에서는 pairwise score `J(r_cand)/(J(r_cand)+J(r_ref))`를 0.5 초과면 1.0, `[0.475, 0.5)`이면 0.75, `[0.45, 0.475)`이면 0.5, `[0.425, 0.45)`이면 0.25, 그 외에는 0으로 바꾼다.
- `s_fact`는 determinate citation 가운데 supported로 판정된 비율이다.

**계산 예.** reference의 `J(r_ref) = 0.62`에 대해 report가 `J(r_cand) = 0.60`이면 pairwise score는 `0.60 / 1.22 = 0.492`이므로 `s_rubric = 0.75`다. `s_fact = 0.9`이면 `R = 0.5625 + 0.25 × 0.75 = 0.75`다. `s_fact = 0.2`이면 `R = 0.5625 + 0.05 = 0.6125`다. `s_rubric = 0`, `s_fact = 1`인 report는 `R = 0`을 받으므로, 인용은 잘 달았지만 내용이 틀린 report에는 보상이 없다.

**verifier 품질에 관한 근거.**
- QUEST는 표본 objective task 50개 중 evaluation script 2개가 실행 불가능했고 6개에 rubric 관련 오류가 있었다고 보고한다(App. A.2). objective task는 rubric refinement, tree verification, script check를 거쳐 17,000개 생성분에서 5,934개만 남았다(34.9%; App. A.1 Table 4).
- QUEST의 pointwise 0/0.5/1 rubric scoring은 사례 약 50%에서 약 1점에 도달했다. 저자들은 judge의 high-score bias 때문이라고 보며, 이를 고치기 위해 pairwise design을 도입했다(§7.4).
- R2E-Gym에서는 문제의 대다수에서 생성한 test 중 correct patch와 incorrect patch를 구분하는 비율이 20% 미만이며, 일부 문제에서는 test의 최대 10%가 toxic하다([[r2e-gym]] §4.2).
- GLM-5 terminal task에서는 refine agent가 “environment가 잠재적 exploit 또는 shortcut에 견고한지” 확인하며 Docker construction accuracy는 90% 초과다([[glm-5]] §4.2.2). 하지만 exploit 가능한 작업이 얼마나 남았는지는 보고하지 않았다.

**조건과 한계.** 이 장의 출처 중 어느 것도 학습용 verifier의 false-positive 또는 false-negative rate를 측정해 보고하지 않는다. 난이도 filter는 학습 중인 모델 또는 그 sibling model을 쓰므로 그 모델이 바뀌면 살아남는 pool도 바뀐다. Tongyi DeepResearch는 이 이유로 RL 중 pool을 refresh한다([[tongyi-deepresearch]] §3.4.3).

**범용 모델에 대한 함의.** shortcut을 받아들이는 verifier는 같은 template으로 만든 모든 환경에서 shortcut에 보상한다. reference solution은 통과하고 trivial agent는 실패하는지 확인하는 verifier validation이 있어야, held-out benchmark가 결코 보상하지 않을 행동을 training pool이 가르치는 일을 막을 수 있다.

## §5 합성 검색 및 연구 작업

**정의.** 검색 작업은 여러 웹 페이지 또는 corpus source에서 조각을 모아 답을 조립해야 하는 질문을 낸다. 연구 작업은 여러 기준으로 품질을 판정하는 보고서를 요구한다.

**문제.** 검색한 페이지에서 질문을 먼저 생성하면, 생성된 질문의 추론 구조가 검색해 온 정보와 맞지 않거나 답이 틀릴 수 있다([[webshaper]] Abstract). 단일 답 작업은 이진 보상만 주고 사실을 찾는 행동은 포괄하지만 보고서 합성은 포괄하지 못한다([[quest-deep-research]] §2).

**메커니즘.**
1. **지식 그래프 subgraph(GLM-5, Tongyi).** GLM-5는 200만 페이지가 넘는 Web Knowledge Graph를 만들고, 낮은 빈도부터 중간 빈도까지의 seed entity를 표본 추출하며, multi-hop subgraph로 확장한 뒤 각각을 질문으로 바꾼다([[glm-5]] §4.2.3). Tongyi DeepResearch는 예를 들어 유사한 속성을 가진 entity들을 합쳐 “질문 안의 불확실성을 전략적으로 증가”시켜 난이도를 높인다([[tongyi-deepresearch]] §3.4.1).
2. **형식화 기반 확장(WebShaper).** Knowledge Projection은 `R(V) = {u | ∃v ∈ V, (u, v) ∈ R or (v, u) ∈ R}`이다. 여기서 `E`는 entity 집합, `R ⊆ E × E`는 relation, `V ⊆ E`다([[webshaper]] Eq. 2). `T = R_1(T_1) ∩ ... ∩ R_k(T_k)`일 때 작업은 `?T`다(Eq. 6–7). layer-wise Expander는 leaf constant `C`를 답이 `C`인 sub-question으로 교체한다. 따라서 `q_{n+1}(T) = Expander(C, q_n(T))`는 같은 답을 유지한다(Eq. 11). validation call은 QwQ가 그 sub-question을 직접 답할 수 있으면 이를 거절한다(§3.2.3).
3. **다중 agent 검증 QA(DeepSeek-V3.2).** question-construction agent가 long-tail entity를 탐색한다. checkpoint와 prompt가 서로 다른 여러 answer agent가 답하고, verification agent는 ground truth가 맞고 모든 candidate answer가 검증 가능하게 틀린 경우에만 sample을 남긴다([[deepseek-v3.1]] §3.2.3). 이 과정으로 search-agent RL 작업 50,275개를 얻었다(Table 1).
4. **rubric tree(QUEST).** trending keyword가 웹 탐색의 seed가 된다. objective task에는 검증 가능한 rubric tree와 GPT-5가 작성한 evaluation script를 붙이고, open-ended task에는 task-specific child와 reference report를 갖는 고정 criteria 네 개를 붙인다(§2.2).

**계산 예: WebShaper 형식화.** “Which player of a team in the 2004-05 season, who was born in 90s? This team is founded in 1966 and is an East German football team.”라는 질문은 `[[V@T, playIn, V@X], [V@T, playAt, C@2004_05], [V@T, bornIn, C@90s], [V@X, foundIn, C@1966], [V@X, isA, C@East German football team]]`로 쓴다([[webshaper]] Eq. 10). leaf constant `C@1966`은 예를 들어 그 해에 일어난 사건을 묻는, 답이 1966인 sub-question으로 확장할 수 있다. target 자체는 바뀌지 않으므로 seed question에 쓴 answer check는 매 확장 뒤에도 그대로 유효하다.

**근거.**
- dataset마다 SFT sample 5,000개를 사용했을 때 QwQ-32B의 GAIA 평균은 WebShaper 53.3, WebWalkerQA 45.6, E2HQA 45.6, MHQA 41.7이었다([[webshaper]] Table 2).
- GLM-5는 도구 없이 추론하는 모델이 8회 중 1회 이상 맞히는 질문과 early agent가 몇 step 안에 푸는 질문을 제거한다([[glm-5]] §4.2.3). 하지만 이 filter들의 ablation은 인쇄하지 않았다.
- 30B scale에서 단일 답 합성 작업(Tongyi-DR)으로 학습한 recipe는 BrowseComp 43.4, HLE 32.9, GAIA 70.9에서 가장 좋았다. OpenResearcher는 “자신의 data synthesis recipe와 밀접하게 맞는 완전 offline benchmark”인 BrowseComp-Plus에서 54.8로 가장 좋았다. QUEST-30B는 8개 benchmark 중 4개에서 최고였다([[quest-deep-research]] Table 3, §6.2). 저자들은 “deep research agent가 보이는 capability는 그 data synthesis recipe에 의해 형성된다”고 결론낸다(Interpretation).

**조건과 한계.** WebShaper, Tongyi DeepResearch, QUEST는 합성한 질문과 GAIA, BrowseComp, HLE 사이의 overlap check를 보고하지 않는다. 예외는 QUEST가 context-summarization data에서 benchmark run을 제외한 경우다(App. A.3). BrowseComp-Plus에서는 retriever에 따라 accuracy가 달라진다. BM25에서는 gpt-5가 55.90%이고 Qwen3-Embedding-8B에서는 70.12%이므로, offline-corpus 작업은 비교 가능하게 하려면 retriever를 고정해야 한다([[browsecomp-plus]] Table 1).

**범용 모델에 대한 함의.** 작업 형식이 가르치는 capability를 선택한다. 단일 답 작업은 사실 찾기를, offline corpus는 그 corpus에 대한 retrieval을, rubric tree는 보고서 합성을 학습시킨다. 범용 research agent의 task pool에는 세 형식이 모두 필요하다.

## §6 일반성을 결정하는 요인으로서의 환경 다양성

**정의.** 환경 다양성(environment diversity)은 학습에 포함된 서로 다른 환경(repository, server, database, domain, generator)의 수다. 어떤 연구가 trajectory 또는 RL sample 수를 고정한 채 비교하면 data volume과 분리해 다양성만을 분리해 비교한다. 고정하지 않으면 둘은 함께 변한다.

**문제.** 적은 수의 환경에서 학습한 agent는 in-distribution 점수는 올릴 수 있지만, 새 환경ㆍ언어ㆍtool ecosystem에서는 실패할 수 있다.

**근거.**
1. **환경 수(Agent-World).** 0, 10, 100, 500, 1,000, 1,978개 환경으로 학습하자 네 subdomain 평균이 18.4%에서 38.5%로 올랐다(+20.1점). 증가폭이 가장 컸던 구간은 10 → 100과 100 → 500이었다([[agent-world]] §4.3.3, Fig. 8). 이 run의 model size와 각 환경 수에서 사용한 training sample 수는 명시되지 않았으므로, 본문은 data volume을 고정하지 않는다. **결과(단일 연구).**
2. **repository 수(SWE-smith).** Procedural Modification trajectory를 700개로 고정하면 repository 4, 25, 50, 100개에서 SWE-bench Verified 점수는 각각 10.3, 11.5, 12.9, 15.1%였다([[swe-smith]] §4.1, Fig. 5). §4.1 본문은 가장 작은 pool이 repository 4개라고 하지만 Fig. 5의 축에는 5라고 적혀 있다. student는 Qwen-2.5-Coder-Instruct 7B다.
3. **합성 general-agent RL(DeepSeek-V3.2).** DeepSeek-V3.2-SFT에 합성 general-agent task만으로 RL을 하면 Tau2Bench, MCP-Mark, MCP-Universe가 개선됐다. search와 code 환경에서만 RL을 한 V3.2-Exp는 이 benchmark들에서 개선되지 않았다([[deepseek-v3.1]] §4.3, Fig. 5; 수치는 figure에만 있다). 이 benchmark들의 환경은 “RL training에서 접하지 않았다”(§4.1).
4. **혼합 agent data(ADP).** OpenHands harness와 Qwen-3-8B를 같게 두고 non-web ADP mixture 약 30K sample로 학습하면 SWE-Bench Verified는 16.6%였고, 같은 크기까지 up-sample한 SWE-smith는 11.0%였다([[agent-data-protocol]] Table 10). GAIA에서는 AgentInstruct-only 학습이 0.6%, ADP가 9.1%였다(Table 6).
5. **절차적 cross-domain transfer(Reasoning Gym).** algorithmic generator로 GRPO를 수행하자 Qwen2.5-3B-Instruct의 held-out algebra는 23.83에서 52.89로, geometry는 0.83에서 23.17로 올랐다([[reasoning-gym]] Table 2).

항목 1–5는 서로 다른 setting의 별개 연구지만, 더 많은 서로 다른 환경 또는 task family가 더 높은 held-out score를 냈다는 방향에서는 일치한다. 항목 2와 4는 data budget을 고정했다(trajectory 700개, sample 약 30K). 항목 1, 3, 5는 fixed budget을 보고하지 않는다. 방향은 **replicated**됐지만 effect size는 연구 간 비교할 수 없다.

**계산 예: doubling당 gain.** SWE-smith의 양 끝점을 쓰면 repository 4 → 100은 `log2(25) = 4.64`번 doubling이고, `15.1 − 10.3 = 4.8`점 상승이므로 doubling당 약 1.0점이다(두 endpoint에서 도출; 논문은 curve를 approximately logarithmic이라고 설명한다). Agent-World의 10 → 1,978 환경에서는 저자들이 500개 환경 뒤 gain이 작아진다고 설명한다. step별 값은 Fig. 8에만 제시된다.

**반대 근거와 좁아짐.**
- 언어: Python-only SWE-smith model은 SWE-bench Multilingual에서 8.4%를 기록했다. Qwen 2.5 Coder Instruct는 6.5%, Claude 3.7 Sonnet은 43%였다. 저자들은 non-Python repository에서 edit가 “Python에 더 가까운 syntax를 반영했다”고 관찰했다([[swe-smith]] App. F.4).
- specialization: 각각 trajectory 700개로 학습한 두 7B model, 즉 repository 100개에서 학습한 모델과 SymPy에서만 학습한 모델은 2022-01-01 뒤 만들어진 SWE-bench Verified의 SymPy instance 22개에서 각각 13.6%, 21.2%를 기록했다. Verified의 나머지에서는 각각 15.3%, 14.0%였다([[swe-smith]] §4.1, Fig. 4).
- tool-call format: AgentScaler-4B의 ACEBench-en Special 점수는 학습 뒤 84.7에서 76.7로 떨어졌다(Table 1). QUEST-35B는 SFT 뒤 BrowseComp-Plus에서 “training-time tool-use pattern에 overfitting하여 허용되지 않은 tool을 호출할 가능성이 더 높아졌다”([[quest-deep-research]] §6.3).
- 절차적 transfer는 균일하지 않다. ARC는 algebra 학습 뒤 6.49에서 4.18로, games 학습 뒤 4.26으로 떨어졌다([[reasoning-gym]] Table 2).

**범용 모델에 대한 함의.** trajectory 수만 제시하는 training report는 다양성 변수를 측정하지 않은 채 남긴다. 따라서 서로 다른 환경의 수를 그 옆에 함께 써야 한다. 일반성을 주장하려면 전체 환경, 언어, tool ecosystem을 hold out해야 한다. 위의 narrowing 결과는 evaluation environment가 training environment와 다를 때에만 드러난다.

## §7 agent benchmark에 대한 환경 작업의 오염 제거

**정의.** 환경 오염 제거(environment decontamination)는 평가 benchmark에도 나타나는 repository, server, database schema, task template, question을 모두 학습 pool에서 빼고, 학습 데이터 cutoff 뒤에 만들어진 평가 작업을 우선 사용하는 절차다.

**문제.** agent benchmark는 public repository, public MCP server, public tool API를 재사용하며, 이는 환경 factory가 작업을 채굴하는 바로 그 source다. repository 이름과 issue text만 주었을 때 OpenAI와 Anthropic의 모델 10개는 SWE-Bench Verified instance에서 gold patch가 바꾼 file 하나를 60–76%의 경우에 맞혔다. 반면 SWE-Bench 밖 repository에서 뽑은 비교 가능한 작업 245개에서는 그 비율이 53% 미만이었다([[swe-bench-illusion]] §4.1).

**메커니즘.**
1. **repository 수준에서 benchmark repository를 제외한다.** SWE-smith는 SWE-bench test repository 12개를 제거한다([[swe-smith]] §2.1). SWE-Gym과 R2E-Gym-Subset은 SWE-Bench와 겹치지 않는 repository를 사용한다([[excerpts/swe-gym-v2-extract]] §3; [[r2e-gym]] §2).
2. **environment 수준에서 benchmark environment를 제외한다.** 각 agent benchmark의 server, database, tool schema를 목록화하고, 일치하는 항목을 tool pool에서 제거한다. AgentScaler는 생성한 τ-bench-domain database와 tool code가 “τ-bench가 제공한 official implementation과 높은 수준으로 일치한다”고 보고한 뒤([[agentscaler]] §2.1) τ-bench에서 평가한다. 하지만 제거 절차는 보고하지 않는다.
3. **새 작업에서 평가한다.** SWE-rebench는 benchmark에 model release 이전에 만들어진 issue가 포함되면 결과를 잠재적으로 오염됐다고 표시한다([[swe-rebench]] §3.2). GLM-5는 SWE-bench Verified가 “2년 넘게 공개된 static, public, human-validated test set”이므로 SWE-rebench에서 평가한다([[glm-5]] §6.2.4).
4. **RL에 쓰지 않은 environment에서 평가한다.** DeepSeek-V3.2는 MCP-Universe, MCP-Mark, Tool-Decathlon의 environment를 RL에서 보지 않았다고 명시한다([[deepseek-v3.1]] §4.1).

**계산 예: 두 benchmark를 나란히 읽기.** SWE-bench Verified에서 GLM-5는 77.8, Claude Opus 4.5는 80.9를 기록한다([[glm-5]] Table 7). 2026년 1월 SWE-rebench task에서는 각각 42.1%와 43.8%를 해결한다([[glm-5]] Table 9). 77.8에서 42.1로 떨어진 수치 자체는 오염의 측정값이 아니다. 두 benchmark는 scaffold, task filter, 난이도가 모두 다르기 때문이다. 정보가 되는 값은 하나의 protocol 아래에서 평가한 모델들 사이의 순위 또는 gap이 어떻게 달라지는지다. SWE-rebench는 DeepSeek-V3 두 version에 대해 이 비교를 수행했다. SWE-bench Verified에서는 39.7과 35.2였지만, 2025년 3–4월 slice에서는 21.3과 21.9였다. 저자들은 이를 오래된 benchmark가 오염됐을 가능성으로 해석한다([[swe-rebench]] Table 2, §3.3; Interpretation).

**조건과 한계.** TOUCAN, AgentScaler, Agent-World, WebShaper, Tongyi DeepResearch, ADP 중 어느 것도 자신의 evaluation benchmark와의 overlap check를 보고하지 않는다. Agent-World의 self-evolution round는 held-out arena task에서 약한 environment를 고르고 MCP-Mark(Postgres)에서 향상을 보고한다([[agent-world]] Table 2). 평가 benchmark가 어떤 작업을 합성할지 안내하면, 이후 점수는 target으로 삼지 않은 environment에서의 성능을 편향 없이 추정한 값이 아니게 된다(Interpretation). 일반적인 contamination 방법은 [[ch-48]]에 있다.

## §8 Agent 데이터 통합(Agent Data Protocol)

**정의.** Agent Data Protocol(ADP)은 어떤 agent trajectory도 action(`APIAction`, `CodeAction`, `MessageAction`)과 observation(`TextObservation`, `WebObservation`)이 번갈아 나타나는 형태로 표현하는 schema다([[agent-data-protocol]] §3.2).

**문제.** agent dataset마다 action syntax와 observation format이 다르다. 따라서 D개의 dataset을 A개의 agent harness에 섞어 쓰려면 D × A개의 converter가 필요하다.

**메커니즘.**
1. 각 raw dataset을 ADP로 한 번 변환한다.
2. harness마다 ADP → SFT converter 하나를 작성한다. 이 converter는 system prompt, action syntax, context 처리 방식을 설정한다.
3. 자동 검사를 실행한다. tool-call format이 맞는지, tool call 대부분이 thought와 짝지어져 있는지(기준 80%), conversation이 적절히 끝나는지를 검사한다.
4. 각 dataset을 multiplier `w_d`로 resample한다. epoch마다 `⌈w_d · n_d⌉`개 example을 뽑으며, `w_d < 1`이면 replacement 없이, `w_d > 1`이면 replacement를 허용해 뽑는다(App. C).

**계산 예: converter 비용.** 13개의 Raw → ADP converter에는 4,892줄의 code가 들었고, ADP → SFT converter는 평균 77줄이다(Tables 7–8). A = 100개의 harness라면 pair별 변환 방식은 약 `100 × 4,892 = 489,200`줄이 든다. ADP 방식은 `4,892 + 77 × 100 = 12,592`줄이면 된다(§6.3). **계산 예: resampling.** Orca AgentInstruct는 trajectory가 1,046.1K개이고 `w_d = 0.001`이므로 epoch마다 약 1,046개를 뽑는다. SWE-Gym은 0.5K개에 `w_d = 3`이다(Table 1, Table 9).

**근거.** SWE-Agent에서 Qwen-2.5-32B-Coder-Instruct는 SWE-Bench Verified에서 40.3%에 도달했다. 이는 SWE-smith data만 쓴 경우의 40.2%와 거의 같다. OpenHands에서는 36.8%에 도달했으며 SWE-Gym data의 20.6%보다 높았다(Table 5). AgentLab 7B는 WebArena에서 4.5%에서 21.0%로 올랐다(Table 3).

**조건과 한계.** coding harness는 non-web 부분 약 30K sample로 학습했고 AgentLab은 web 부분 약 20K sample로 학습했다(App. C.1). 그러므로 Table 6의 “cross-task transfer”는 그 부분들 내부에서의 transfer다. ADP는 source 간 deduplication, 4개 evaluation benchmark에 대한 decontamination, 일반적인(non-agent) benchmark를 보고하지 않는다.

**범용 모델에 대한 함의.** shared schema를 쓰면 environment mixture를 ablation할 수 있는 변수로 만들 수 있다. 하지만 harness 의존성은 없어지지 않는다. Qwen-2.5-7B-Instruct에서 SWE-smith만으로 학습하면 OpenHands 점수는 1.0%였지만(Table 6), 자체 SWE-agent harness에서 쓴 SWE-smith trajectory는 Qwen-2.5-7B-Coder-Instruct 기준 15.2%를 기록했다(Table 3). 두 수치의 base model도 다르므로, harness만이 유일한 차이는 아니다.

## Agentic training pipeline: 환경 작업이 들어가는 위치

1. **중간 학습(mid-training).** Tongyi DeepResearch는 AgentScaler식 database 환경에서 function-calling 데이터를 생성하고, 이를 32K와 128K 길이의 agentic continued pre-training에 넣으며 “일반 pre-training 데이터의 작은 비율”을 interleave한다([[tongyi-deepresearch]] §3.3). 보고서별 stage 배치는 [[ch-32d]]에서 비교한다.
2. **SFT.** teacher가 성공한 작업은 rejection-sampled trajectory가 된다. SWE-smith는 해결된 run 6,457개 중 5,016개를 쓰며 task당 최대 3개로 제한한다([[swe-smith]] §4). TOUCAN은 선택한 instance 119.3K개를 쓴다([[toucan-mcp]] §4.1). trajectory filtering의 자세한 내용은 [[ch-27]] 및 [[ch-29d]]에 있다.
3. **RL.** teacher가 한 번도 풀지 못한 작업은 RL용으로 남긴다. QUEST는 Tongyi DeepResearch가 모든 rollout에서 실패한 objective task 864개를 남긴다([[quest-deep-research]] App. A.5). DeepSeek-V3.2와 GLM-5는 code, search, terminal, 합성 환경에서 RL을 수행한다([[deepseek-v3.1]] Table 1; [[glm-5]] §3.3). multi-turn RL의 mechanics는 [[ch-45b]], rollout infrastructure는 [[ch-54]]에 있다.
4. **평가.** hold-out environment(MCP-Mark, MCP-Universe, SWE-rebench의 fresh task)는 transfer를 측정한다([[ch-51a]]).

## Negative samples and negative feedback

이 단계에서 “negative”는 과정 표준의 네 뜻 모두로 나타난다. (1) negative marginal value, (2) content로서의 negative, (3) conditioning으로서의 negative(여기서는 어떤 source도 사용하지 않음), (4) gradient로서의 negative다.

1. **negative의 출처.** (a) validation에 실패한 candidate task: SWE-smith candidate의 49.9%는 test를 깨지 못한다(Table 1). SWE-rebench candidate의 95.3%는 살아남지 못한다(§2.1, §2.4에서 도출). (b) 난이도 범위 밖의 task: DeepSeek-V3.2는 pass@100이 0인 task를 제거하고, GLM-5는 tool-free model이 답하는 질문을 제거하며, Tongyi는 항상 풀리거나 항상 실패하는 문제를 제거한다(§4). (c) verifier에 실패한 trajectory. (d) policy가 아니라 environment 때문에 생긴 rollout 실패: GLM-5는 “coding-agent sandbox는 본질적으로 불안정할 수 있고 model과 무관한 이유로 실패할 수 있다”고 적는다([[glm-5]] §4.1.2). (e) 결함 있는 test가 만든 verifier false negative([[swe-rebench]] §2.3; [[swe-bench-pro]] §6.2). 어떤 source도 training verifier의 false-negative rate를 보고하지 않는다.
2. **현재 관행이 negative를 다루는 방법.**
   - **버린다**(뜻 1): 실패 candidate, 범위 밖 task, 실패한 SFT trajectory([[swe-smith]] §3; [[toucan-mcp]] §3.1).
   - **다른 pool로 보낸다**: QUEST는 teacher-failed task를 버리지 않고 RL pool로 보낸다(App. A.5). Agent-World는 실패한 arena trace를 진단 agent에게 보내 새 target task를 작성하게 한다(§3.2.2).
   - **content로 보존한다**(뜻 2): AgentScaler는 최종 state가 맞으면 tool-call error가 있는 trajectory도 보존한다(§3.1). GLM-5 SFT는 “trajectory 내부의 erroneous segment”를 보존하지만 loss에서는 mask한다([[glm-5]] §3.1). TOUCAN의 irrelevance task는 풀 수 없는 요청에 tool call이 0개인 응답을 가르친다(§3.2).
   - **gradient로 쓴다**(뜻 4), 단 제외 규칙을 둔다: RL에서 실패 rollout은 negative advantage를 받지만, GLM-5는 environment-collapse 실패를 제외하고 Tongyi는 length-limit truncation 같은 일부 negative rollout을 제외한다([[tongyi-deepresearch]] §3.4.3).
3. **메커니즘.** logit `z`, softmax `p`인 token에 대해 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. 여기서 `y`는 sample된 token, `j`는 vocabulary 전체를 돈다. negative advantage는 이 gradient에 음수를 곱하므로 `z_y`를 낮추고, 빠진 probability mass는 `p_j`에 비례해 다른 token으로 간다. 그중 가장 큰 부분은 확률이 가장 높은 대안으로 간다. GLM-5의 agentic RL objective는 group reward 평균을 baseline으로 써서 `A_i = r_i − (1/K) Σ_k r_k`를 사용한다. `K`는 문제당 sample한 trace 수이고 `r_i`는 trace `i`의 reward다([[glm-5]] §4.1; reasoning-RL §3.2는 같은 양을 group standard deviation으로 나눈다). **계산 예.** §4.1 형식을 따라 K = 8 rollout, 정답 3개(`r = 1`), 오답 4개(`r = 0`), 그리고 sandbox가 crash해 0점 처리된 정답 rollout 1개가 있다고 하자. 평균은 `3/8 = 0.375`이므로, crash했지만 정답인 rollout은 `A = −0.375`를 받아 그 action이 아래로 밀린다. GLM-5의 규칙은 이 crashed sample을 제거한다. valid sample 7개는 8의 절반보다 많으므로 valid sample을 반복해 group을 padding한다. valid sample이 4개뿐이면(절반보다 많지 않음) group 전체를 drop한다([[glm-5]] §4.1.2).
4. **근거.** Tongyi DeepResearch는 “filter하지 않은 negative rollout set을 직접 최적화하면 training stability가 크게 나빠지고 긴 학습 뒤 policy collapse로 이어질 수 있다”고 보고한다(§3.4.3; 표 없음). GLM-5는 environment-failure sample 제거가 “spurious reward noise를 줄이고 training stability를 개선한다”고 쓴다(§4.1.2; 표 없음). DeepSeek-V3.2는 policy divergence가 threshold `δ`를 넘는 negative-advantage sequence를 mask하고, 원래 불안정했던 일부 run에서 stability가 개선됐다고 보고한다([[deepseek-v3.1]] §3.1). 이 source들 중 negative에 귀속되는 improvement 비율을 측정한 것은 없다. positive만 쓴 SWE-Gym self-improvement run은 on-policy success 868개와 off-policy success 491개로 학습했지만 SWE-Bench Lite가 15.3%에서 8.7%로 떨어졌다([[excerpts/swe-gym-v2-extract]] §4.2). 즉 on-policy filtering만으로 gain이 보장되지는 않는다.
5. **control.** rollout마다 failure reason을 기록하고 infrastructure failure를 loss에서 제외한다. 학습 전에 reference solution과 do-nothing agent로 verifier를 검증한다. length limit에 잘린 rollout에는 reward 0을 주는 대신 mask하거나 drop한다. off-policy sample의 negative update를 제한한다(DeepSeek-V3.2 sequence masking). teacher-failed task는 나쁜 task로 처리하지 말고 RL에 남긴다.
6. **diagnostic.** environment별 failure-reason histogram(policy 대 infrastructure 대 timeout), drop 또는 padding된 group 비율, positive- 및 negative-advantage sample의 log-probability를 분리 추적한 값, environment family별 pass@1 및 pass@k, 각 task template에서 do-nothing agent가 통과하는 비율을 기록한다.
7. **일반성에 미치는 효과.** 한 environment family에 집중된 false negative(예: 한 language의 flaky test)는 그 family에서 맞는 행동을 낮추고 policy를 verifier가 더 깨끗한 family 쪽으로 이동시킨다. irrelevance target(뜻 2)은 맞는 tool이 없을 때 abstention을 가르친다. TOUCAN 7B는 BFCL irrelevance가 67.93에서 75.18로 올랐지만 relevance는 72.22에서 66.67로 떨어졌다([[toucan-mcp]] Table 2). 이는 abstention trade-off를 보인다.

## 레시피

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | source 위치 | 상태 | 이 값의 근거 |
|---|---:|---|---|---|---|---|---|
| SWE-agent-LM-32B (Qwen2.5-Coder-32B-Instruct) | 32B | distill-SFT | trajectory; teacher; limit | 5,016(6,457 resolved run에서, task당 ≤ 3); SWE-agent의 claude-3-7-sonnet-20250219; ≤ 75 step, $2.00 | arXiv:2504.21798v2 §3, §4 | verified 2026-09-15 | §4: 쉬운 trajectory를 반복하면 성능 하락(표 없음); Fig. 1 trajectory scaling |
| SWE-agent-LM-32B | 32B | distill-SFT | LR; epoch; context; hardware | 5e-5; 최대 3; 32,768; H100 2–8개(torchtune, full fine-tuning) | App. F.1 | verified 2026-09-15 | ablation 미보고 |
| SWE-Gym-32B (Qwen2.5-Coder-32B-Instruct) | 7B, 14B, 32B | distill-SFT | trajectory; LR; epoch; batch; context | 491; 1e-4; 최대 5; global 8; 32,768 | arXiv:2412.21139v2 §4.1, App. B.2 | verified 2026-09-15 | Fig. 1: 100–491 trajectory에 따른 resolve rate |
| R2E-Gym-32B | 7B, 14B, 32B | distill-SFT | trajectory; teacher temperature; max context | 성공한 Sonnet-3.5-v2 trajectory 3,321개; 0.2; 20K | [[r2e-gym]] card (arXiv:2504.07164v1 §3, App. B) | verified 2026-09-14 | Fig. 2: 100–3,200 trajectory |
| SWE-rebench dataset | — | data | installation recipe candidate; validator model | task당 최대 3; Qwen2.5-72B-Instruct | [[swe-rebench]] card (arXiv:2505.20411v2 §2.2, App. C) | verified 2026-09-14 | App. C Table 3: 1/3/10 candidate가 18 instance 중 6/8/9개 configure |
| DeepSeek-V3.2 | arXiv:2512.02556에는 미보고(V3.1-Terminus start point: total 671B, activated 37B, V3.1 model card) | RL | agent task 수(code / search / general / code interpreter) | 24,667 / 50,275 / 4,417 / 5,908 | arXiv:2512.02556v1 §3.2.3 Table 1 | verified 2026-09-15 | §4.3 Fig. 5: synthetic general-agent RL 대 search+code RL |
| DeepSeek-V3.2 | 위 행 참조 | RL | general-agent task filter | DeepSeek-V3.2에서 non-zero pass@100; environment 1,827개 | §3.2.3 | verified 2026-09-15 | Table 5: 표본 50 task에서 pass@1 12%–62% |
| GLM-5 | total 744B, active 40B(§2.1, Table 10) | RL | environment 수; stale-sample rule; failed-group rule | SWE environment > 10k, language 9개; `w′ − w_0 > τ`이면 drop(`τ` 미보고); valid > K/2면 group padding, 아니면 drop | arXiv:2602.15763v2 §4.1, §4.1.2, §4.2.1 | verified 2026-09-15 | ablation 미보고 |
| GLM-5 | total 744B | RL | search-task difficulty filter | tool-free model이 8회 중 ≥ 1회 답하면 제거 | §4.2.3 | verified 2026-09-15 | ablation 미보고 |
| Kimi K2 | total 1.04T | SFT | tool pool | 실제 MCP tool 3000개 이상; 합성 tool 20,000개 이상 | arXiv:2507.20534v2 §3.1.1 | verified 2026-09-15 | Fig. 9: t-SNE coverage(ablation 없음) |
| AgentScaler-30B-A3B | 30B-A3B | SFT | API pool; domain; stage; LR, epoch, data size | API > 30,000; domain > 1,000; 2 stage; 미보고(§2–4, Limitation 확인) | arXiv:2509.13311v1 §2.1, §3.2 | verified 2026-09-15 / not reported | Fig. 3: ACEBench-en에서 stage 1 대 stage 2 |
| TOUCAN SFT (Qwen2.5-Instruct) | 7B, 14B, 32B | SFT | instance; LR; epoch; batch; max length | 119.3K(28.3K core, 40K irrelevance, 15.8K diversify, 35.2K multi-turn); 2e-5; 2; 64; 32,768 | arXiv:2510.01179v1 §4.1, App. C.2 Table 5 | verified 2026-09-15 | App. C.3 Table 6: 14B extension ablation |
| Agent-World-8B / 14B | 8B, 14B | RL | RL sample; step당 task × rollout; clip ε_low / ε_high; max trajectory | 5K; 32 × 8; 0.2 / 0.28; 80K token | [[agent-world]] ledger (arXiv:2604.18292v1 §4.1) | verified 2026-09-14 | ablation 미보고 |
| Agent-World | — | data | task keep rule; tool keep rule | ReAct run 5개 중 ≥ 2 성공; compile 및 unit test > 50% pass | §3.1, §3.1.1 | verified 2026-09-14 | ablation 미보고 |
| QUEST-35B (Qwen3.5-35B-A3B) | 35B-A3B | distill-SFT | task / trajectory / session; keep threshold | objective 5,070 / 19,435 / 39,861; open-ended 1,958 / 4,485 / 11,903; ε = 1 objective, 0.475 open-ended | arXiv:2605.24218v1 Table 1, §4.1 | verified 2026-09-15 | ablation 미보고 |
| QUEST-35B | 35B-A3B | distill-SFT | rollout; retry; max tool call; context threshold | task당 5; failed trajectory당 3; 100; 16K | App. A.4 | verified 2026-09-15 | ablation 미보고 |
| QUEST-35B | 35B-A3B | RL | task; reward; KL | objective 864 + open-ended 269; `R = 0.75 s_rubric + 0.25 min(s_fact, s_rubric)`; KL penalty 없음 | Table 1, §4.4 Eq. 1 | verified 2026-09-15 | §6.3 Fig. 5: MT, SFT, RL stage ablation |
| Tongyi DeepResearch 30B-A3B | total 30.5B, active 3.3B | mid-train | context stage; general-data share | 32K 다음 128K(64K–128K agentic data); “small proportion”(값 미보고) | arXiv:2510.24701v3 §3.3.1 | verified 2026-09-15 / not reported | ablation 미보고 |
| Tongyi DeepResearch 30B-A3B | total 30.5B | RL | reward; policy regime; baseline; clip 값, group size, LR | 정답 0/1; strict on-policy; leave-one-out; 미보고 | §3.4.3 | verified 2026-09-15 / not reported | Fig. 9: 32K/48K/64K context 비교 |
| WebShaper (Qwen-2.5-32B/72B, QwQ-32B) | 32B, 72B | distill-SFT + RL | seed; rollout; trajectory | seed 18k(5개 중 ≥ 1 정답이면 보존); question당 rollout 5; trajectory 5,000 | arXiv:2507.15061v1 §3.1, §3.3 | verified 2026-09-15 | Table 2: 5,000-sample dataset 비교 |
| ADP-trained Qwen-2.5-Coder-Instruct | 7B, 14B, 32B | SFT | sampling multiplier; sample | orca agentinstruct 0.001, synatra 0.01, code feedback 0.1, nebius SWE 0.2, agenttuning 2, SWE-Gym 3, others 1; 약 30K(coding) / 20K(web) | arXiv:2510.24702v2 App. C Table 9, C.1 | verified 2026-09-15 | Table 10: equal-scale ADP 대 SWE-smith |
| Reasoning Gym RG-Math (Qwen2.5-3B-Instruct) | 3B | RL | step; reward; curriculum rule | GRPO 800 step; accuracy 1.0 + format 0.2; 20 step 동안 performance > 70%이면 level 상승 | arXiv:2505.24760v2 §4.3, Fig. 5, §5 | verified 2026-09-15 | Table 5: curriculum 대 uniform level |

**작은 범용 run의 시작점.** 7B–32B Qwen2.5-Coder-Instruct base와 SWE environment에서는 teacher trajectory 수천 개에 rejection-sampling SFT를 하고 task당 최대 3 trajectory, LR 5e-5, 최대 3 epoch, 32,768-token context를 쓰면 SWE-smith 32B setting(trajectory 5,016개, H100 2–8개)을 재현한다. 7B–32B Qwen2.5-Instruct base에서 tool-use breadth를 얻기 위해 TOUCAN은 40K irrelevance example을 포함한 119.3K instance에 LR 2e-5, epoch 2, effective batch 64, 32,768 token을 썼다. 8B–14B에서 합성 tool environment로 RL을 하려면 Agent-World는 5K RL sample에서 step마다 task 32개 × rollout 8개, `ε_low` 0.2, `ε_high` 0.28, trajectory cap 80K token을 썼다. 이 행들은 서로 다른 논문, model, substrate에서 왔으며 어떤 것도 다른 setting에 맞추어 tuning한 값이 아니다.

## 일반화 관점

**(a) breadth를 늘리는 것.**
- 더 많은 서로 다른 environment: trajectory 700개를 고정한 repository 4 → 100개에서 10.3% → 15.1%([[swe-smith]] Fig. 5), fixed-budget statement 없이 environment 0 → 1,978개에서 18.4% → 38.5%([[agent-world]] Fig. 8). 방향은 replicated됐다.
- RL의 environment type 증가: search+code RL이 개선하지 못한 unseen Tau2Bench, MCP-Mark, MCP-Universe에서 synthetic general-agent RL이 개선됐다([[deepseek-v3.1]] §4.3).
- 같은 크기에서 한 source보다 mixed agent dataset: SWE-Bench Verified 16.6% 대 11.0%([[agent-data-protocol]] Table 10).
- 여러 task format: 30B에서 single-answer, offline-corpus, rubric-tree task는 각각 서로 다른 benchmark에서 선두다([[quest-deep-research]] Table 3).
- environment pool의 여러 language: DeepSeek-V3.2(8 language)와 GLM-5(9 language)가 이를 구축한다([[deepseek-v3.1]] §3.2.3; [[glm-5]] §4.2.1). language count만 분리한 ablation은 없다.

**(b) narrowing 또는 forgetting을 만드는 것.**
- single-language environment: Python-only training 뒤 SWE-bench Multilingual 8.4%([[swe-smith]] App. F.4).
- repository specialization: SymPy-only 7B model은 SymPy subset에서 100-repository 7B보다 7.6점 높고 Verified의 나머지에서는 1.3점 낮다([[swe-smith]] Fig. 4).
- training tool pattern에 과적합: SFT 뒤 BrowseComp-Plus에서 허용되지 않은 tool call([[quest-deep-research]] §6.3), ACEBench-en Special 84.7 → 76.7([[agentscaler]] Table 1), 7B에서 BFCL relevance 72.22 → 66.67 및 non-live AST 84.19 → 78.52([[toucan-mcp]] Table 2).
- specialized RL objective: QUEST RL은 “HLE와 GAIA 성능을 약간 희생한다”(§6.3).
- simulation-only substrate: Simulator-8B BFCL V4는 학습하지 않은 Qwen3-8B의 40.4보다 낮은 23.9다([[agent-world]] Table 1; 통제된 비교 아님).
- recipe-matched benchmark: OpenResearcher는 자신의 합성과 맞는 offline benchmark에서만 선두다([[quest-deep-research]] §6.2).

**(c) 이 단계에서 측정하는 방법.**
- environment, server, repository, language 전체를 hold out하고, 학습에 한 번도 쓰지 않은 environment(MCP-Mark, MCP-Universe, SWE-bench Multilingual)의 score를 in-family score 옆에 보고한다.
- date-split task를 쓴다. SWE-rebench는 issue creation date에 따라 결과가 잠재적으로 contaminated됐다고 표시한다([[swe-rebench]] §3.2).
- SWE-rebench처럼 fixed scaffold에서 SEM과 pass@5를 포함해 5회 run을 보고한다(§3.2). Llama-4-Maverick은 mean resolved 12.2%에서 pass@5 27.6이었으므로 single run은 오도할 수 있다(§3.3).
- 모든 grader에 null agent를 돌린다. do-nothing agent가 τ-bench Airline의 38%를 통과한다([[agentic-benchmark-checklist]] §5.2).
- 알려진 measurement error는 SWE-Bench Verified memorization signal([[swe-bench-illusion]] §4.1), offline search benchmark의 retriever 의존성([[browsecomp-plus]] Table 1), benchmark-guided task synthesis([[agent-world]] Table 2), agent에는 tool을 주고 base model에는 주지 않는 general-benchmark 비교([[tongyi-deepresearch]] Fig. 11)다. 마지막 것은 forgetting을 측정하지 못한다. agentic training 뒤 general-ability forgetting은 [[ch-30a]]와 [[ch-51a]]에서 다룬다.

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 징후 | 확인 방법 |
|---|---|---|
| reference solution에서 한 번도 실행하지 않은 verifier의 task를 보존 | teacher trajectory는 맞아 보이는데 한 task family의 RL reward가 0 근처에 머묾 | gold patch 또는 solution function을 실행하고 `F2P > 0`, pass-to-fail = 0 또는 verifier pass를 요구 |
| verifier가 trivial output을 허용 | reward는 오르는데 held-out success는 그대로이고 짧거나 빈 답도 reward를 받음 | 모든 task template에 do-nothing agent와 answer-enumerating agent 실행 |
| environment crash를 policy failure로 점수화 | 한 substrate의 reward variance가 급증하거나 failure가 container host/시간별로 몰림 | rollout마다 failure reason 기록, infrastructure failure 제외, dropped-group rate 보고 |
| environment가 아니라 trajectory를 셈 | trajectory를 늘려도 held-out score가 바뀌지 않음 | distinct repository, server, domain, language를 보고하고 fixed trajectory에서 environment 수 ablation |
| benchmark repository/server를 training pool에 채굴 | public benchmark와 date-split 또는 private-repository benchmark 사이의 큰 gap | repository 및 server exclusion list, SWE-rebench식 fresh task, SWE-Bench Pro commercial subset |
| benchmark domain code를 복제한 tool environment 합성 | in-domain benchmark가 다른 tool suite보다 더 크게 상승 | 학습 전 generated schema와 tool code를 benchmark implementation과 비교 |
| evaluation criterion을 task text에 누출 | agent가 reproduction script 작성이나 exploration을 멈춤 | run당 reproduction attempt 수를 센다(SWE-smith: 500회 중 379 대 127) |
| single-language SWE pool | non-Python edit에 Python syntax가 나타남 | SWE-bench Multilingual 또는 language별 held-out set에서 평가 |
| stale model에 묶인 difficulty filter | RL batch가 all-solved/all-failed가 되고 zero-advantage group이 증가 | all-equal reward group 비율을 추적하고 current checkpoint로 pool refresh |
| 다른 retriever로 offline search task 평가 | retriever만 바꿔도 score가 변함 | retriever와 corpus를 고정하고 보고 |
| harness별 conversion 없이 agent dataset 혼합 | 한 harness에서 잘 학습한 data가 다른 harness에서는 near-zero score(SWE-smith: Qwen-2.5-7B-Instruct/OpenHands 1.0% 대 7B Coder/SWE-agent 15.2%, base model도 다름) | shared schema와 harness당 converter 하나로 변환하고 각 target harness에서 평가 |

## 이해 점검

1. SWE-smith는 latest commit에 bug를 주입하고 SWE-Gym은 실제 historical issue를 쓴다. 이 선택이 task당 비용을 어떻게 낮추며, 어떤 종류의 task를 표현할 수 없게 하는지 설명하라.
2. 같은 model을 쓴 두 pool 중 하나는 non-zero pass@100, 다른 하나는 5회 중 최소 2회 성공으로 filter한다. 각각이 어떤 task를 남기며 RL learning signal을 어떻게 바꾸는지 설명하라.
3. GRPO에서 verifier false negative가 맞는 trajectory를 왜 아래로 밀어내는지, 제거된 probability mass가 어디로 가는지 추적해 설명하라.
4. AgentScaler의 τ-bench domain database는 official τ-bench implementation과 매우 비슷하다. 그 τ-bench gain이 ACEBench-zh gain보다 general tool use의 약한 증거인 이유를 설명하라.
5. 30B에서 single-answer recipe는 BrowseComp에서, rubric-tree recipe는 DeepResearch Bench에서 선두다. 이것이 task format이 capability를 결정하는 방식에 대해 뜻하는 바와, 범용 research agent의 task pool이 포함해야 할 것을 설명하라.
6. SWE-smith의 32B model은 SWE-bench Verified 40.2%, SWE-bench Multilingual 8.4%를 기록했다. environment pool에서 두 수치까지 이르는 인과 사슬을 설명하라.
7. GLM-5는 SWE-bench Verified 77.8, SWE-rebench 42.1%다. 이 gap만으로 contamination을 측정할 수 없는 이유와, 무엇을 비교해야 하는지를 설명하라.
8. Agent-World는 held-out arena environment의 failure를 바탕으로 합성할 task를 고르고 MCP-Mark에서 gain을 보고한다. MCP-Mark가 generality의 유효한 측정치로 남는 조건을 설명하라.

## 연결

- 이전: [[ch-29b]] — Long-Conversation and Accumulating-Context Synthesis.
- 다음: [[ch-29d]] — User Simulators, Trajectory Verification, and Failed Trajectories.
- 의존성: [[ch-27]] — Agentic Trajectory Data; [[ch-32d]] — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training.
- 여기서 사용하는 이전 장: [[ch-18]] — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix; [[ch-23]] — Model Collapse and Verification of Synthetic Data; [[ch-26]] — Tool and Function-Calling Data.
- 이 environment를 사용하는 뒤 장: [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares; [[ch-44]] — Process Supervision and Verifiable Rewards; [[ch-45b]] — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability; [[ch-45d]] — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL; [[ch-46a]] — Lab: Small Agentic SFT-then-RL Run with a Generality Gate; [[ch-54]] — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments.
- 평가 단계: [[ch-48]] — Contamination Detection and Its Effect on Reported Scores; [[ch-51a]] — Evaluating Agent Generality and Reliability.

## 출처

- [[swe-gym]] — SWE-Gym library card. 이 장은 card의 수치 일부(K = 10 per task, 약 10K H100-hour, Verified 7B 3.0% → 15.3%)가 arXiv:2412.21139v2에 없으므로 primary paper 수치를 [[excerpts/swe-gym-v2-extract]]를 통해 취한다.
- [[excerpts/swe-gym-v2-extract]] — raw instance 64,689개, annotation 200시간, task 2,438개, image 6 TB, Table 3 결과, App. B.2 recipe, self-improvement 하락.
- [[r2e-gym]] — SWE-Gen commit-based environment, 합성 대 실제 issue statement, generated-test distinguishability, recipe 행.
- [[swe-smith]] — environment-first design, bug strategy 및 yield, storage와 cost, repository scaling, specialization, Multilingual 결과, recipe.
- [[swe-rebench]] — automated installation, fail/pass validation, funnel, date-based contamination flag, DeepSeek-V3 비교, recipe 행.
- [[glm-5]] — SWE, terminal, search environment pipeline, search difficulty filter, environment-failure 및 stale-sample 처리, SWE-rebench 평가.
- [[kimi-k2]] — 실제 MCP와 합성 tool의 hierarchical domain evolution, simulator와 실제 sandbox, rubric-filtered trajectory.
- [[agentscaler]] — read-write database abstraction, tool-graph construction, state-alignment filter, τ-bench consistency statement, 결과와 regression.
- [[toucan-mcp]] — 실제 MCP server funnel, task 및 trajectory filter, irrelevance extension, BFCL 결과, SFT recipe.
- [[deepseek-v3.1]] — DeepSeek-V3.2 agent task 수, code-environment build rule, verification function을 갖는 general-agent synthesis, pass@100 filter, synthetic RL transfer, off-policy sequence masking.
- [[webshaper]] — Knowledge Projection formalization, layer-wise expansion, validation, dataset 비교.
- [[quest-deep-research]] — rubric tree, evaluation script와 error rate, retention funnel, reward formula, recipe-dependent capability, stage ablation.
- [[tongyi-deepresearch]] — 세 environment form, offline Wikipedia environment, RL data curation, negative-rollout exclusion, mid-training 배치.
- [[agent-data-protocol]] — schema, converter cost, sampling multiplier, mixed 대 single-source 결과, harness dependence.
- [[agent-world]] — environment-count scaling, task keep rule, weighted tool-graph walk, Simulator-8B 비교, self-evolution round, recipe 행.
- [[reasoning-gym]] — procedural generator, cross-domain transfer 및 regression, curriculum rule.
- [[swe-bench-illusion]] — §7에서 쓰는 SWE-Bench Verified memorization probe.
- [[swe-bench-pro]] — task specification이 interface detail을 빠뜨릴 때 생기는 verifier false negative.
- [[agentic-benchmark-checklist]] — 결함 있는 grader를 통과하는 do-nothing 및 shortcut agent.
- [[browsecomp-plus]] — offline search evaluation의 retriever 의존성.
