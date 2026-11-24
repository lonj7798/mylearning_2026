# Chapter 46a — 실습: 일반성 게이트를 둔 소규모 에이전트 SFT-후-RL 실행

> **핵심 통찰.** 에이전트 사후 학습은 자체 목표 지표가 포괄하지 못하는 능력을 양방향으로 바꾸며, 변화의 크기는 에이전트 데이터의 비중과 실패 rollout을 처리하는 방식에 좌우된다. 7B에서 에이전트 궤적만으로 SFT한 모델은 held-out 에이전트 과제에서 0.09, 일반 과제에서 0.22를 기록한 반면, 같은 궤적을 ShareGPT와 η = 0.2 비율로 샘플링하면 각각 0.67과 0.63을 기록했다([[agenttuning]], Table 5). 도구 데이터로 think-then-act GRPO를 적용한 뒤 Qwen2.5-7B-Instruct는 BFCL Multi-Turn에서 13.6에서 23.5로 상승하고 IFEval에서는 62.4에서 59.8로 하락했으며, 사용 가능한 도구가 없을 때도 답하는 비율은 34.8%에서 90.2%로 상승했다. 이는 해당 연구에서 보고한 어떤 tool-calling 또는 instruction-following 벤치마크도 포착하지 않은 변화다([[reasoning-trap-tool-hallucination]], Table 3, §4.4.2). 이 실습은 고정된 예산에서 에이전트 SFT 뒤 다중 턴 RL을 실행하고, 에이전트 비중과 실패 궤적 정책을 명시적인 축으로 두며, held-out 환경 패밀리, tool-calling slice, abstention 검사, 그리고 두 비에이전트 suite가 모두 paired interval과 함께 사전 명시한 마진을 통과할 때만 결과를 받아들인다.
>
> **지침.** SFT 혼합물에 에이전트 궤적을 추가할 때에는 학습에 쓴 패밀리의 점수가 아니라 held-out 에이전트 패밀리에서 비중을 선택한다. 그 이유는 7B와 13B에서 에이전트 전용 데이터가 혼합물에 비해 held-out 에이전트 점수와 일반 점수를 모두 낮췄기 때문이다([[agenttuning]] Table 5). 또한 119.3K Toucan 인스턴스로 미세 조정한 7B 모델은 BFCL V3 multi-turn에서 9.74점을 얻었지만 non-live AST에서는 5.67점, relevance에서는 5.55점을 잃었다([[toucan-mcp]] Table 2). 다중 턴 RL rollout이 context limit, step limit, 또는 timeout으로 끝날 수 있으면, 그 궤적을 mask할지 실패로 점수화할지를 실행 전에 결정하고 그룹 baseline이 둘 중 어느 방식을 사용하는지 기록한다. 동일한 rollout group에 대해서도 두 선택은 서로 다른 advantage를 만들고(§5), DeepSWE는 masking이 없을 때 reward collapse를 보고한다([[deepswe]] §2.3, Fig. 6, curve only). 계산 자원이 모델 하나만 허용한다면 데이터베이스 기반 도구 환경과 함께 1.5B–4B instruct checkpoint를 사용하되 모든 게이트는 유지한다. GiGPO는 ALFWorld와 WebShop에서 Qwen2.5-1.5B-Instruct를 2개의 H100 GPU로 150 iteration 학습했다([[gigpo-verl-agent]] App. E.1). 어떤 차이든 보고할 때에는 과제별 trial 수를 명시한 paired interval을 보고한다. §7의 worked example에서는 114개 과제와 과제당 4회 trial에서 최소 검출 가능 효과가 9.6점이기 때문이다.

## 이 장이 범용 모델에 중요한 이유

범용 모델의 파이프라인은 pre-training → mid-training → SFT → preference optimization → RL → evaluation 순서로 진행된다. 이 실습은 두 개의 RL 실습 중 두 번째이자 RL 단계의 마지막 장이다. held-out capability를 기준으로 단일 턴 preference 또는 RLVR 단계를 측정한 [[ch-46]] 뒤에 오며, [[ch-47]]로 시작하는 평가 단계에 앞선다.

에이전트 학습은 측정해야 하는 항목을 바꾸는 세 가지 점에서 단일 턴 경우와 다르다.

1. **정책은 자신이 내보내지 않은 토큰을 본다.** 도구 결과와 환경 관측값은 정책 자체 턴 사이에 context로 들어간다. 그 토큰 중 어느 것에 loss를 부여할지는 설정 선택이며, 잘못 선택해도 오류 메시지 없이 모델이 학습된다([[verl-rollout]], `agent_loop.rst` L33-72).
2. **rollout은 성공이나 실패 이외의 이유로 끝난다.** 궤적은 context limit, step limit, 또는 wall-clock timeout에 도달할 수 있다. 각 종료에는 reward를 배정하거나 제외해야 하며, 그 배정은 group 내 다른 모든 rollout이 받는 gradient를 바꾼다(§5).
3. **목표 지표는 행동의 좁은 단면만 포괄한다.** DeepSWE는 SWE-Bench-Verified와 validation curve 밖의 평가를 보고하지 않으며([[deepswe]]), GiGPO도 자신이 학습한 환경과 QA 데이터셋 밖의 평가를 보고하지 않는다([[gigpo-verl-agent]] §5). 두 보고서 모두 환경이 작동시키지 않은 능력에 어떤 일이 일어났는지 말할 수 없다. 이 장의 source set에서 목표 지표 밖 행동을 측정한 유일한 연구는 55.4점의 변화를 보고했다. tool-use RL은 필요한 도구가 없는 프롬프트에서 도구를 꾸며내거나 오용하는 비율을 34.8%에서 90.2%로 올렸다([[reasoning-trap-tool-hallucination]] Table 3).

따라서 이 실습의 측정 가능한 문제는 다음과 같다. **고정된 계산 예산에서 어떤 에이전트 데이터 비중과 어떤 실패 궤적 정책이 held-out 환경 패밀리에서도 유지되는 에이전트 과제 향상을 만들며, 비에이전트 suite를 명시한 마진보다 더 떨어뜨리지 않는가.**

설계 재료는 [[ch-29c]](환경과 과제 합성), [[ch-30b]](다중 기술 SFT 혼합물과 에이전트 비중), [[ch-45b]](다중 턴 RL mechanics), [[ch-45c]](context 관리), [[ch-43a]](음의 gradient)에서 가져온다. 이 장은 그것들을 실행하고 interval과 함께 수치를 보고한다.

## §1 실습 질문, 가설, 산출물, 예산 경로

**질문.** 주어진 에이전트 데이터 비중에서 에이전트 SFT 후 다중 턴 RL을 수행하면 일반 능력에 도움이 되는가, 해가 되는가?

**실행 전에 작성하는 가설.** 각 행은 source가 측정한 방향, 그 측정의 설정, 그리고 이 실습에서 그 방향을 반박하는 관측을 제시한다.

| 축 | 예측 방향 | 출처와 설정 | 반박하는 관측 |
|---|---|---|---|
| SFT에서 에이전트 비중 η = 0.2 대 0.5 | 비중이 높은 쪽은 학습된 패밀리에서 더 많이 얻고 held-out 패밀리와 비에이전트 suite에서 더 많이 잃는다 | [[agenttuning]] Table 5 (Llama-2-chat 7B/13B/70B, held-in 에이전트 과제 6개와 held-out 에이전트 과제 6개, 일반 과제 4개); η = 0과 η = 1 endpoint만 존재 | η = 0.5 arm의 held-out paired interval이 η = 0.2 arm보다 위에 놓인다 |
| 동등한 예제 수에서 단일 패밀리 대 혼합 패밀리 에이전트 slice | 혼합 slice가 자신의 패밀리 밖에서 더 높은 점수를 낸다 | [[agent-data-protocol]] Table 10 (Qwen-3-8B, OpenHands harness, 각각 약 30K sample: SWE-smith up-sampled 11.0 대 ADP 16.6 on SWE-Bench Verified); [[swe-smith]] Fig. 4 (700개 SymPy trajectory로 학습한 7B: SymPy subset 21.2 대 13.6, Verified-without-SymPy 14.0 대 15.3) | 같은 수에서 혼합 slice가 held-out 패밀리에서 더 낮은 점수를 낸다 |
| 에이전트 SFT 뒤 다중 턴 RL | 학습된 패밀리의 성공률이 상승한다 | [[gigpo-verl-agent]] Table 1 (Qwen2.5-1.5B-Instruct, GRPO: 3개 seed에서 ALFWorld 72.8 ± 3.6, WebShop success 56.8 ± 3.8) | 3개 seed에서 학습된 패밀리 성공률이 SFT checkpoint를 넘지 못한다 |
| 실패 궤적 정책: mask 대 penalize | masking은 같거나 더 높은 학습 패밀리 reward와 더 적은 degenerate trajectory를 만든다 | [[deepswe]] §2.3 (Qwen3-14B, compact filtering on/off; Fig. 6은 curve만 보여 주며 수치는 없음) | 동일 step에서 masking이 reward를 낮춘다 |
| 에이전트 RL 뒤 abstention | 사용 가능한 도구 없이 행동하는 비율이 상승한다 | [[reasoning-trap-tool-hallucination]] Table 3 (Qwen2.5-7B-Instruct, SynTool에서 GRPO: no-tool-available 34.8% → 90.2%, distractor-tool 54.7% → 100.0%) | abstention 검사가 paired interval 밖으로 움직이지 않는다 |

**이후에 사용하는 정의.** *에이전트 비중*(η)은 [[agenttuning]] Eq. 1에서 정의한 SFT mixture objective에서 일반 instruction data에 대한 agent trajectory의 sampling ratio다. 이는 예제 수가 아니라 sampling weight이며, 모든 source를 한 번씩만 샘플링할 때에만 둘이 일치한다. *턴*은 정책이 한 번 생성하고 환경이 한 번 응답하는 단위다. *void turn*은 action이 parse되지 않거나 환경에서 거부되어 환경 상태가 바뀌지 않는 턴이다. *loop turn*은 동일 trajectory 안에서 (state, action) 쌍이 이미 발생한 턴이다. *환경 패밀리*는 action space와 observation format을 공유하는 환경 집합이다(예: 데이터베이스 기반 도구 API 전체).

**산출물.**
1. `runs.jsonl`: base checkpoint, SFT mixture hash, η, RL algorithm 및 설정, context policy, failed-trajectory policy, seed, code commit, rollout-engine commit을 담은 run당 한 행.
2. `rollouts/<run>/*.jsonl`: turn별 token count, response mask, tool-call validity, termination reason, reward를 담아 §8 post-mortem용으로 보존한 모든 rollout.
3. `eval/items/<run>/<suite>.jsonl`: SFT checkpoint와 모든 RL checkpoint에 대해 §7의 모든 suite에서 item별·trial별 점수를 담은 파일.
4. `decontamination.md`: 모든 training slice와 모든 gate suite의 n-gram overlap count(§2.4).
5. `agentic-lab-memo.md`: 위 가설, 학습 전 기록한 non-inferiority margin(§7.4), development result, decision list, 그 뒤의 gate result.

**예산 경로.**

| 항목 | 전체 경로 | 자원 제약 경로 |
|---|---|---|
| Base model | Qwen2.5-7B-Instruct | Qwen2.5-1.5B-Instruct 또는 3B–4B instruct checkpoint |
| arm당 SFT 예제 | 30,000 (agent + general, η로 혼합) | 10,000 |
| RL 환경 패밀리 | 학습 3개, held out 1개 | 학습 2개, held out 1개; 모두 데이터베이스 기반 |
| 축 | η ∈ {0.2, 0.5}; context policy ∈ {none, keep-recent-2, summary}; failed-trajectory policy ∈ {mask, penalize} | η ∈ {0.2, 0.5}; failed-trajectory policy ∈ {mask, penalize} |
| configuration당 seed | 3 | 3 |
| 일반성 게이트 | 필수 | 필수 |

**계산 anchor(추정치가 아니라 검증된 값).** GiGPO는 Qwen2.5-1.5B-Instruct를 H100 GPU 2개로, Qwen2.5-7B-Instruct를 H100 GPU 4개로 각각 150 iteration 학습했다. 매 iteration에는 task당 trajectory 8개와 rollout당 task 16개, 즉 iteration당 환경 128개를 사용했다([[gigpo-verl-agent]] App. E.1). DeepSWE의 32B run은 iteration당 container 512개로 H100 GPU 64개를 6일간 사용했으며([[deepswe]] intro, §2.2), 이는 여기의 두 경로 모두 범위를 벗어난다. ALFWorld에서 1.5B의 iteration당 breakdown은 rollout, log-probability, update를 합쳐 362.83 s였다([[gigpo-verl-agent]] §5.6). 따라서 실습 환경의 rollout 비용이 동일하다는 검증되지 않은 가정 아래에서는 constrained path의 150 iteration이 약 15 GPU-hours다.

## §2 단계 A: 두 에이전트 비중에서의 에이전트 SFT

### 2.1 에이전트 슬라이스 선택

공개된 세 슬라이스는 서로 다른 기질(substrate)을 포괄하며, 보고된 차이 때문에 이 실습은 하나가 아닌 여러 패밀리를 사용한다.

| 슬라이스 | 규모 | 기질 | 비교할 보고 결과 |
|---|---|---|---|
| [[toucan-mcp]] | 495개의 실제 MCP 서버에서 수집한 trajectory 1,527,259개, 다중 턴 567,262개. 논문의 SFT subset은 119.3K개 | HTTP로 접근한 실제 tool server | Qwen2.5-7B-Instruct의 BFCL V3 overall 55.10 → 58.26, multi-turn 12.88 → 22.62, irrelevance 67.93 → 75.18, non-live AST 84.19 → 78.52, relevance 72.22 → 66.67 (Table 2) |
| [[agent-data-protocol]] | 13개 dataset을 통합한 trajectory 1.3M개, 평균 10.1 round | 혼합: API, code, message action (53% / 24% / 23%) | Qwen-2.5-7B-Coder-Instruct가 SWE-Agent와 함께 SWE-Bench Verified 0.4 → 20.2. 약 30K sample에서 Qwen-3-8B 기준 ADP 16.6, SWE-smith-only 11.0 (Tables 3, 10) |
| [[swe-smith]] | 128개 repository에 걸친 task instance 50,137개, rejection sampling trajectory 5,016개 | repository마다 하나의 Python execution environment | SWE-agent-LM-32B는 SWE-Bench Verified에서 pass@1 40.2. SymPy trajectory 700개로 학습한 7B는 SymPy subset에서 21.2점(100-repository model은 13.6), Verified 전체에서는 14.0점(100-repository model은 15.3) (§3, Fig. 4) |

SWE-smith specialization 결과는 environment family 수준에서 좁아짐을 가장 직접적으로 측정한다. 한 repository에서 나온 trajectory 700개는 7B에서 그 repository subset을 7.6점 올리고 benchmark의 나머지를 1.3점 내렸다. 32B에서 같은 대체는 SymPy를 33.3에서 42.4로, 나머지를 40.2에서 38.3으로 움직였다([[swe-smith]] §4.1). 같은 절의 repository-pool ablation은 다양성에서 반대 방향의 효과를 보인다. trajectory 수를 700개로 고정하고 4, 25, 50, 100개 repository에서 추출했을 때 SWE-Bench Verified의 해결 비율은 각각 10.3%, 11.5%, 12.9%, 15.1%였으며, 저자들은 이를 대략 로그형이라고 설명한다.

**실습 선택.** 적어도 두 패밀리에서 에이전트 슬라이스를 같은 비율로 추출하고, 세 번째 패밀리는 SFT에서 전부 held out으로 둔다. held-out 패밀리는 §7의 gate가 사용하는 패밀리이며 SFT와 RL 어느 쪽에도 나타나지 않는다.

### 2.2 일반 instruction data와의 혼합

[[agenttuning]]은 mixture objective를 다음과 같이 정의한다(Eq. 1).

```
J(θ) = η · E_{(x,y) ~ D_agent} [ log π_θ(y | x) ] + (1 − η) · E_{(x,y) ~ D_general} [ log π_θ(y | x) ]
```

여기서 π_θ(y | x)는 instruction과 history x가 주어졌을 때 fine-tuned model이 response y에 부여하는 확률이다. D_agent는 agent trajectory set, D_general은 general instruction set, η는 agent data의 sampling ratio다. 논문은 7B에서 η를 0부터 1까지 0.1 간격으로 탐색했고 held-out agent task에서 η = 0.2를 선택했다. η별 점수는 표로 제시되지 않았으며 Table 5에는 endpoint만 나온다(§2.2.2, App. A).

**계산 예시.** 예제 30,000개의 예산으로 mixture를 한 번 통과할 때 η = 0.2는 agent 예제 약 6,000개와 general 예제 24,000개를 뽑고, η = 0.5는 각각 약 15,000개를 뽑는다. agent 예제는 더 길다. ADP trajectory는 평균 10.1 round이고 SWE-smith trajectory는 평균 26.8 round이므로([[agent-data-protocol]] Table 2), 두 설정 모두에서 *token* 기준 agent 비중은 예제 기준 agent 비중보다 높다. source만으로는 그 배율을 사전에 계산할 수 없다. 따라서 실습은 각 arm에서 example share와 token share를 모두 기록하고, Recipe 표에는 인용한 각 값이 둘 중 어느 기준인지 명시한다.

**endpoint의 근거.** [[agenttuning]] Table 5의 held-in / held-out / general normalized score는 다음과 같다.

| 규모 | Mixed (η = 0.2) | General only | Agent only |
|---|---|---|---|
| 7B | 1.96 / 0.67 / 0.63 | 0.38 / 0.64 / 0.61 | 1.34 / 0.09 / 0.22 |
| 13B | 2.11 / 0.78 / 0.69 | 0.43 / 0.81 / 0.63 | 1.57 / 0.10 / 0.19 |
| 70B | 2.55 / 1.40 / 0.96 | 0.99 / 0.98 / 1.00 | 2.47 / 0.87 / 0.83 |

**조건과 한계.** 7B와 13B에서 mixture의 held-out score는 general-only training과 가깝다(0.67 대 0.64, 0.78 대 0.81). 혼합의 held-out 이점은 70B에서만 나타난다(1.40 대 0.98). 따라서 1.5B–7B 실습은 mixture가 held-out agent ability를 올리기보다 general ability를 보호할 것으로 예상해야 하며, 실행 전에 memo에 이를 적어야 한다. 점수는 task별로 평가한 model들의 평균을 1로 정규화한 값이므로 백분율이 아니며 다른 model set으로 옮겨갈 수 없다([[agenttuning]] §3.1).

### 2.3 trajectory filtering과 filtering하지 않았을 때의 결과

[[agenttuning]]은 final reward r = 1인 trajectory만 남긴다(Mind2Web은 r ≥ 2/3). 그 결과 instruction 35,341개 중 filtered trajectory 1,866개가 남으며 keep ratio는 5.29%다(Table 1). 7B에서 filtering하지 않은 set으로 학습했을 때 held-in은 1.34, held-out은 0.47이었고, filtered set은 각각 1.96과 0.65였다(Table 2). [[toucan-mcp]]은 failed tool response, tool call 없음, local file path가 있는 trajectory를 rule filter로 제거한다. 그 다음 question quality와 scenario realism이 5, completeness와 conciseness가 4 이상이며 desired tool-use percentage가 1.0인 instance만 남긴다(§3.1, §4.1). [[swe-smith]]는 8,686개 instance에서 17,906번 시도하여 36%를 해결한 rejection-sampling run에서 instance당 trajectory를 최대 3개까지만 남긴다(§3).

실습은 filter 결정을 가정으로 두지 않고 기록된 숫자로 재현한다. 즉 패밀리별 keep rate와 filtering 뒤 패밀리별 example count를 기록한다.

### 2.4 gate 실행 전 decontamination

[[agenttuning]]은 mismatch를 최대 4개 허용하는 10-gram token span을 매칭하고, contaminated token이 80%를 넘으면 example을 dirty, 20% 미만이면 clean으로 표시한다. 전체 contamination rate는 15.58%이며 held-in 1,021개 중 dirty example은 7개라고 보고한다(App. B, Table 7). [[toucan-mcp]], [[agent-data-protocol]], [[swe-smith]] 어느 것도 BFCL, τ-bench, τ²-bench, MCP-Universe에 대한 overlap check를 보고하지 않는다. 앞의 두 source는 이를 보고하지 않은 항목으로 열거한다.

**실습 규칙.** gate를 열기 전에 모든 training slice를 §7의 모든 gate suite와 동일한 n-gram rule로 매칭하고 count를 `decontamination.md`에 기록한다. gate suite에 대해 dirty item이 하나라도 있는 slice는 clean 처리하거나 해당 arm에서 제거하며, memo에는 어느 조치를 취했는지 기록한다.

### 2.5 시작점으로 사용할 SFT hyperparameter

세 슬라이스에서 검증된 값은 Recipe 표에 제시한다. [[toucan-mcp]]은 LR 2e-5, 2 epoch, effective batch 64, AdamW (0.9, 0.999), ε 1e-8, maximum sequence 32,768을 사용했다(App. C.2, Table 5). [[swe-smith]]는 LR 5e-5, 최대 3 epoch, maximum context 32,768로 torchtune full fine-tuning을 수행했고 H100 GPU 2–8개를 사용했다(App. F.1). [[agenttuning]]은 7B와 13B에서 LR 5e-5, 70B에서 1e-5, batch 64, sequence 4,096, warmup ratio 0.02의 cosine schedule, model output에만 loss를 적용하는 방식을 사용했다(§2.2.3, Table 6). 세 source는 7B learning rate에서 2.5배 차이가 난다. 실습은 값을 평균내지 않고 각 arm이 어떤 값을 사용했는지 기록한다.

## §3 단계 B: 작은 검증 가능 환경 집합에서의 다중 턴 RL

**환경 공급원.** 이 단계에는 판정 모델이 아니라 프로그램이 결과를 확인하는 환경 패밀리가 필요하다. 실행 가능한 소프트웨어 공학 환경 집합이 한 가지 공급원이다. [[r2e-gym]]과 [[swe-gym]]은 repository 과제와 이를 채점하는 test suite를 함께 제공하며, DeepSWE는 각 문제가 Docker image에 매핑된 R2E-Gym 문제 4.5K개로 학습했다([[deepswe]] §2.1-§2.2). [[deepswe]] §6은 동일한 recipe를 SWE-Smith 및 SWE-Gym 데이터에서 실행했을 때의 향상이 제한적이었다고 보고한다. 따라서 환경 집합은 결과를 바꾸는 선택이며, `runs.jsonl`에 arm별로 기록한다.

### 3.1 rollout이 만들어야 하는 것

정의: 다중 턴 rollout은 정책 생성과 환경 응답이 교대로 이어지는 시퀀스이며, 성공, 실패, step limit, context limit, 또는 timeout에서 끝난다. 학습 시퀀스는 이 모든 토큰을 이어 붙인 것과, 어느 위치가 loss를 받는지를 표시하는 **response mask**로 구성된다.

commit `753aed3`의 verl 구현([[verl-rollout]])에서의 mechanism은 다음과 같다.

1. `AgentLoopBase.run`은 `AgentLoopOutput(prompt_ids, response_ids, response_mask)`를 반환한다. mask가 1이면 LLM이 생성한 token이고 0이면 tool-response token이다(`agent_loop.rst` L33-72).
2. `ToolAgentLoop`는 모델 생성 token에 mask 1을, tool 및 template token에 mask 0을 준다(`continuous_token.py` L390-413).
3. policy loss는 이 mask를 입력으로 받는다(`verl/workers/utils/losses.py` L93-108).
4. `vLLMHttpServer.generate`는 prompt token id를 받아 token별 log-probability를 갖는 token id를 반환하므로, 턴 사이에서 시퀀스를 다시 tokenize하지 않는다(`vllm_async_server.py` L556-759).

verl agent-loop 문서는 token id를 이어 붙이는 대신 최종 message list에 chat template을 다시 적용하면 단일 턴 학습에서도 PPO가 수렴하지 않았다고 적는다(`docs/advance/agent_loop.rst` L149-182; 수치를 제시하지 않은 경험 보고). **실습 규칙:** 학습 시퀀스는 token id로 구성하고, 턴별로 mask가 1인 위치의 비율을 기록하며, 첫 optimizer step 전에 rendered history를 다시 tokenize했을 때 동일한 id가 재현된다는 test를 assert한다.

mask를 올바르게 적용해도 두 번째 불일치는 남는다. sampler와 learner는 같은 weight 아래 같은 token에 서로 다른 확률을 부여할 수 있다. DAPO 및 Qwen2.5-32B에서 token별 최대 차이는 1.0에 도달했고, 일부 token에서는 `π_sampler = 1`, `π_learner = 0`이었다([[rollout-training-mismatch-tis]], Fig. 1). 실습은 iteration마다 sampler-learner probability gap의 최대값과 평균값을 기록하고, gap이 커지는 것은 최적화할 metric이 아니라 학습을 멈춰야 할 이유로 취급한다.

### 3.2 알고리즘과 advantage

실습은 [[grpo]]와 같이 critic이 없는 group-based 알고리즘을 사용한다. 각 task에서 동일 initial state로부터 trajectory N개를 sample하고, 한 trajectory의 advantage는 그 trajectory return에서 group mean을 뺀 뒤 normalization factor로 나눈 값이다. [[gigpo-verl-agent]] Eq. 3은 이를 다음과 같이 쓴다.

```
A^E(τ_i) = ( R(τ_i) − mean{R(τ_j)}_{j=1..N} ) / F_norm({R(τ_j)}_{j=1..N})
```

여기서 `R(τ_i)`는 trajectory i의 total return이고, `F_norm`은 group standard deviation(GRPO의 default)이거나 1이다. 1이면 leave-one-out estimator가 된다. [[dr-grpo]]는 standard-deviation 항에 반대하는 근거를 제시한다. 이 항은 분산이 낮은 group의 가중치를 높인다. [[gigpo-verl-agent]] §5.2는 선택이 task-dependent라고 보고한다. `F_norm = 1`은 더 어려운 subtask(Look, Pick2, WebShop)에서 더 높은 점수를 냈고 다른 곳에서는 둘이 비슷했다. 따라서 실습은 모든 arm에서 `F_norm = 1`로 고정하고, 이것이 검증한 축이 아니라 고정 설정임을 기록한다.

[[deepswe]]는 standard deviation 없이, KL loss 없이, entropy coefficient 0인 leave-one-out 형태를 사용한다. learning rate는 `1e-6`, `clip_ratio_high`는 0.28, task당 trajectory는 8개다([[deepswe-recipe]], script L24, L34, L49). 실습은 7B에서 이 값들로 시작하고, 1.5B에서는 GiGPO의 ALFWorld 및 WebShop 값을 사용한다(LR `1e-6`, group size 8, rollout당 group 16개, rollout temperature 1.0, validation temperature 0.4, KL loss coefficient 0.01).

step-level credit assignment는 측정된 비용이 거의 없으며 이 실습의 검증 축은 **아니다**. [[gigpo-verl-agent]]는 anchor-state grouping에 0.01 s, step-advantage 계산에 0.53 s가 걸렸다고 보고했는데, iteration의 나머지 부분은 362.83 s였으므로 0.002% 미만이다(§5.6). `ω = 0`인 경우에도 실습은 step-group size distribution을 §6의 instrument로 기록한다. 이 분포는 반복된 state를 측정하며, 그것이 loop-turn signal이기 때문이다.

### 3.3 reward와 turn budget

[[gigpo-verl-agent]] App. E.1은 성공 reward 10, 실패 reward 0, invalid action penalty −0.1의 rule-based reward를 제시한다. episode limit은 ALFWorld에서 환경 step 50회, WebShop에서 15회이며 turn당 maximum response length는 token 512개다. [[deepswe]]는 최종 patch가 선택한 Pass2Pass 및 Fail2Pass test를 5분 제한 안에 통과하면 reward 1을 주고, test 하나라도 실패하거나 run이 timeout되면 0을 준다(§2.2).

여기에는 합치면 안 되는 두 종류의 negative signal이 있다. 턴 내부의 reward shaping인 step별 **invalid-action penalty**와, 끝나지 않은 trajectory를 어떻게 처리할지의 문제인 **미종료 trajectory 처리**(§5)다. 실습은 failed-trajectory 축만 바뀌도록 invalid-action penalty를 모든 arm에서 −0.1로 고정한다.

## §4 검증 축 1: context policy

정의: context policy는 턴 t에서 무엇을 prompt에 넣을지를 결정한다. 세 arm은 **none**(전체 history), **keep-recent-k**(task와 최근 k개의 observation-action pair), **summary**(최근 k개 pair보다 오래된 모든 내용을 model이 작성한 summary)다.

[[gigpo-verl-agent]] App. E.2는 ALFWorld와 WebShop에 keep-recent-2를 사용하고, search-augmented QA에는 전체 history를 사용한다. maximum prompt는 ALFWorld에서 token 2,048개, WebShop에서 token 4,096개다. 논문은 context policy 간 ablation을 보고하지 않는다. 따라서 그 설정은 측정된 결과가 아니라 설계 선택이다. [[deepswe]]는 maximum context 64k와 maximum environment step 100에서 평가했다고 보고하고, training script에는 `max_response_length` 32,768 및 `agent.max_steps` 50이 있다([[deepswe-recipe]], script L18-L19, L69). 블로그는 training context policy를 명시하지 않는다.

**실습이 측정할 것.** 각 context arm마다 trained-family success, truncation rate(context limit에서 끝난 rollout의 비율), 마지막 turn의 mean prompt token 수, held-out-family score를 측정한다. 기록할 예측은 keep-recent-k가 truncation rate를 낮추고, summary arm은 summary를 생성하는 데 드는 turn 비용을 치르는 대신 이를 더 낮춘다는 것이다. 이 장의 source set에서 어느 방향도 측정하지 않았으므로, 둘 다 이 설정에서는 **open question**이다. 실습은 자체 수치를 하나의 소규모 결과로 보고한다. 이론 및 학습된 summarization 대안은 [[ch-45c]]에 있다.

## §5 검증 축 2: failed-trajectory policy

정의: context limit, step limit, 또는 timeout에서 끝난 trajectory에는 verifier outcome이 없다. 두 policy를 비교한다.

- **Penalize**: 검증된 failure와 동일하게 reward 0을 할당하고, 그 trajectory로 학습한다.
- **Mask**: 그 trajectory에는 loss를 할당하지 않는다. 이것은 DAPO의 overlong filtering을 확장해 generation 중 timeout(20분)에 걸린 trajectory와 maximum step count에 도달한 trajectory도 mask하는 DeepSWE의 compact filtering이다([[deepswe]] §2.3).

저자들은 masking의 이유를 두 가지 든다. agent가 우연히 모든 test를 통과할 수 있다. 예를 들어 처음 10 step에서 정답을 낸 뒤 관련 없는 file을 계속 수정할 수 있으며, 그런 trajectory에 reward를 주면 collapse로 이어진다. 또한 step당 average response length는 감소하는 반면 environment step 수는 증가한다(Figure 6 및 7). Figure 6은 compact filtering 유무의 Qwen3-14B를 비교하며 curve만 보여 준다. 효과 크기에 대한 수치는 보고하지 않는다.

**source가 명시하지 않은 선택.** loss를 mask하는 일과 trajectory를 group baseline에서 제거하는 일은 서로 다른 연산이지만, 블로그는 첫 번째만 명시한다. 실습은 둘 중 무엇을 할지 선택하고 기록해야 한다. [[deepswe]]의 leave-one-out advantage를 사용하면 다음과 같다.

```
A_i = r_i − (1/(n − 1)) · Σ_{j ≠ i} r_j
```

여기서 `r_i ∈ {0, 1}`은 rollout i의 reward이고, `n`은 task당 rollout 수다. 두 variant는 서로 다른 수치를 낸다.

**계산 예시.** 하나의 task에서 `n = 8` rollout을 실행했다고 하자. 검증된 success 2개, 검증된 failure 4개, truncated 2개가 있다.

| Variant | Success advantage | Verified-failure advantage | Truncated trajectory |
|---|---|---|---|
| Penalize (truncated = reward 0, trained) | 1 − 1/7 = **0.857** | 0 − 2/7 = **−0.286** | −0.286, gradient 포함 |
| Mask loss, baseline에는 유지 | 1 − 1/7 = **0.857** | 0 − 2/7 = **−0.286** | −0.286, weight 0 |
| Mask loss, baseline에서 제거 (`n = 6`) | 1 − 1/5 = **0.800** | 0 − 2/5 = **−0.400** | advantage 없음 |

truncated rollout을 baseline에서 제거하면 verified failure에 가해지는 negative push는 40% 더 커지고(−0.286이 아니라 −0.400), success에 가해지는 positive push는 7% 더 작아진다. 모든 advantage가 0인 all-fail case를 포함한 이 표의 interactive version은 [figures/truncation-advantage.html](figures/truncation-advantage.html)에 있다. 여기서 reader는 `n`, success 수, truncated rollout 수를 설정하고 세 variant를 나란히 볼 수 있다.

**Zero-variance group.** 한 task의 n개 rollout이 모두 같은 reward를 가지면 모든 advantage가 0이고, 그 task는 update에 아무 기여도 하지 않는다. [[deepswe]] §6은 SWE-Smith와 SWE-Gym data에서 RL을 실행했을 때 `"a high solve-none rate"`와 함께 향상이 제한적이었다고 보고한다. 이것이 바로 이 조건이다. 실습은 모든 arm에서 iteration마다 zero-variance group의 비율을 기록한다. reward curve가 평평한데 이 비율이 증가하면, algorithm이 작동을 멈춘 것이 아니라 task set이 현재 policy에 비해 너무 어렵거나 너무 쉬워졌다는 뜻이다.

## §6 계측

아래의 각 수량은 iteration마다, 그리고 environment family마다 기록한다. 그래야 집계값의 변화가 어느 family에서 비롯됐는지 추적할 수 있다.

| 수량 | 정의 | 기록하는 이유 | 기준점 |
|---|---|---|---|
| 턴별 entropy | turn `t`의 token들에 대한 policy logit의 categorical entropy이며, response position만 남기도록 mask한다 | entropy collapse와 entropy explosion은 필요한 처방이 서로 반대다 | verl은 `loss_agg_mode`로 집계한 categorical entropy를 `actor/entropy`로 기록한다. `entropy_coeff`의 기본값은 0이다([[entropy-logging-patterns]]) |
| advantage 부호별 entropy | positive-advantage trajectory와 negative-advantage trajectory에서 위 수량을 따로 계산한다 | suppression이 작동하는 곳은 negative branch다(§ Negative samples) | negative-feedback standard가 요구하지만, 이 자료 집합의 어느 source도 agent에 대해 이를 보고하지 않는다 |
| tool-call 유효성 | action이 parse되고 environment가 받아들인 turn의 비율 | 이후 abstention check가 벌점을 주는 failure의 직접 측정값 | [[gigpo-verl-agent]]는 invalid action에 step당 −0.1의 penalty를 준다 |
| void turn | action이 실패했고 environment state가 바뀌지 않은 turn | “잘못된 action”과 “action을 하지 않음”을 분리한다 | — |
| loop turn | trajectory 안에서 `(state, action)` 쌍이 이미 나타났던 turn | 반복 state는 loop의 관측 가능한 형태다 | [[gigpo-verl-agent]] §5.5: size ≥ 10인 anchor-state group은 iteration 10에서 20%를 넘고, iteration 75에는 12.1%(10 ≤ size < 50) 및 3.1%(size ≥ 50)로 떨어지며, iteration 140에는 6–8에 집중한다 |
| trajectory 길이 | rollout당 turn 수와 rollout당 token 수 | §4의 context policy와 상호작용한다 | [[deepswe]] Fig. 7: step당 response length는 줄어드는 반면 environment step 수는 늘어난다 |
| truncation 비율 | context, step, 또는 time limit에서 끝난 rollout의 비율 | §5의 분모다 | [[deepswe]] §2.3 |
| zero-variance group 비율 | n개 reward가 모두 같은 task의 비율 | §5 참조 | [[deepswe]] §6 |
| environment family별 reward | pooling하지 않은 family별 mean reward | 합친 curve는 한 family가 붕괴하는 상황을 가린다 | family는 §1에서 정의한다 |
| sampler–learner 확률 차이 | token별 `\|log π_sampler − log π_learner\|`의 최댓값과 평균 | inference engine에서 생기는 off-policy drift | [[rollout-training-mismatch-tis]] Fig. 1 |

[[gigpo-verl-agent]] §5.5의 step-group size 분포는 사용할 수 있는 loop 계측 중 비용이 가장 낮다. group의 N개 trajectory 전체에서 environment state를 hash하여 계산하며, 1.5B에서는 iteration당 0.01초가 들고 추가 rollout이 필요 없다.

## §7 일반성 게이트

게이트는 결정을 적어 둔 뒤에 측정하고, 허용 margin을 사전에 고정한 측정값들의 집합이다. 게이트는 네 부분으로 이루어진다.

### 7.1 Held-out environment family

SFT slice에도 RL environment에도 나타나지 않는 family 하나를 사용한다. SFT checkpoint에 대한 paired interval과 함께 success rate로 보고한다. 이 항목만이 retention이 아니라 agentic transfer를 시험한다.

### 7.2 Tool-calling 및 dual-control slice

- **BFCL** slice는 하나의 숫자로 합치지 않고 category별로 보고한다. [[bfcl]]이 그 이유를 보여 준다. prompting mode의 gpt-4o-2024-11-20은 single-turn AST multiple call과 parallel call에서 각각 95.5와 94.0을 얻지만, multi-turn base에서는 59.0, memory에서는 6.0을 얻는다(Table 1, §5.6). 또한 model은 relevance와 irrelevance에서 서로 반대 방향으로 움직일 수 있다. prompting mode의 Qwen2.5-72B-Instruct는 relevance 100.0, irrelevance 72.8을 얻는다. 실습은 Simple, Multiple, Parallel, Irrelevance, Relevance, Multi-Turn을 각각 하나의 고정 mode(function-calling 또는 prompting)에서 보고한다. [[bfcl]]은 §5.1에서 mode effect를 분석한다. function-calling mode의 GPT-4-turbo-2024-04-09는 irrelevance 83.8, prompting mode에서는 35.6을 얻는다(Table 1).
- Dual-control behaviour에는 **τ²-bench** slice를 사용한다. [[tau2-bench]]의 telecom에는 write user tool 15개와 read user tool 15개를 사용하는 task 114개가 있다. 각 task는 temperature 0에서 네 번 실행하고, metric은 “서로 독립적인 k회 실행 중 성공한 비율”인 pass^k다. pass^k는 pass@k와 달리 k가 커질수록 떨어진다. 기준점은 다음과 같다. gpt-4.1의 pass^1은 retail 74%, airline 56%, telecom 34%다. telecom mode ablation에서 gpt-4.1은 default 0.34에서 no user 0.67, oracle plan 0.88로 움직이고, o4-mini는 0.42에서 0.73, 0.96으로 움직인다(Fig. 4 left). pass^1은 올랐는데 pass^4가 오르지 않은 model은 더 유능해진 것이 아니라 일관성이 낮아진 것이다.

### 7.3 Abstention 및 non-agentic regression

- **Abstention.** [[reasoning-trap-tool-hallucination]]의 구성에는 두 setting이 있다. No-Tool-Available은 system prompt에 필요한 tool이 없을 때이고, Distractor-Tool은 관련 없는 tool 하나를 제공할 때다. benchmark는 349개 Agent-SafetyBench tool로 만든 296 × 2개의 tool-query pair이며, human과의 agreement rate가 보고되지 않은 DeepSeek-R1 judge를 사용한다(App. A.1, A.4). 이 check가 필요한 이유는 reference result에 있다. think-then-act GRPO 뒤 Qwen2.5-7B-Instruct는 No-Tool-Available에서 34.8%에서 90.2%로, Distractor-Tool에서 54.7%에서 100.0%로 움직였고, BFCL Multi-Turn도 13.6에서 23.5로 올랐다(Table 3). tool이 training data에 전혀 없는 GSM8K만으로 GRPO를 실행해도 두 rate 모두 올랐다(Fig. 3).
- **MMLU-Pro**는 14개 discipline의 12,032개 question으로 이루어져 있고, 각 question에는 최대 10개의 option이 있다(83%의 question에는 10개가 있으며 평균은 9.47개다). 논문의 5-shot chain-of-thought protocol로 점수화한다. 이 실습에서 두 harness 속성이 중요하다. answer extraction이 실패하면 random option으로 fallback하므로, 기대한 format을 더 이상 내보내지 않는 model은 flag되지 않고 chance 수준의 점수를 받는다. 그리고 24개 prompt에 걸친 prompt-variation range는 약 2%, maximum은 3.74%다([[mmlu-pro]] §4, §6.3). 실습은 모든 checkpoint에 대해 prompt 하나와 extraction regex 하나를 고정하고, 그 고정 setting을 보고한다.
- **IFEval**은 검증 가능한 instruction type 25개에 걸친 prompt 541개이며, prompt-level strict accuracy로 보고한다([[ifeval]] §1, §2.2). tool RL 뒤 reference change는 −2.6 point다([[reasoning-trap-tool-hallucination]] Table 3). 이는 단일 541-item paired comparison이 분해할 수 있는 half-width보다 작다(§7.4).

### 7.4 Interval, resampling, 그리고 이 규모 실습이 감지할 수 있는 것

[[adding-error-bars-evals]]는 estimator를 제시한다. item별 score `s_i`와 n개 item에 대한 mean `s̄`인 score 하나의 경우, `SE = sqrt((1/(n−1)) Σ_i (s_i − s̄)² / n)`이다(Eq. 1). 같은 item에서 checkpoint 둘을 비교할 경우 paired standard error는 `SE_paired = sqrt(SE²_A + SE²_B − 2 SE_A SE_B Corr(s_A, s_B))`다(§4.2). 저자는 paired form을 “wherever practicable” 사용할 것을 권고하고, “complicated sampling scheme or estimator is being used”하는 경우가 아니면 bootstrapping은 불필요하다고 본다(§2.1). task당 네 trial의 pass^k는 그러한 estimator다. 따라서 이 실습은 pass^k에는 task별 paired bootstrap을, item-level suite에는 analytic paired form을 사용하며, 각 숫자에 어떤 방법을 썼는지 밝힌다(해석: source는 pass^k를 논의하지 않는다).

**계산 예시 1 — IFEval.** `n = 541`이고 score가 0.60 부근이면 `SE_Bernoulli = sqrt(0.6 × 0.4 / 541) = 0.0211`, 즉 2.11 point다. checkpoint 두 개의 unpaired difference는 `SE = sqrt(2) × 2.11 = 2.98` point이고, 95% half-width는 5.84 point다. 두 checkpoint의 item별 correlation이 0.7이면 paired SE는 `2.11 × sqrt(2 − 2 × 0.7) = 1.63` point이고 half-width는 3.20 point다. tool RL 뒤 보고된 −2.6-point IFEval change는 두 interval 모두 안에 있으므로, 541-item run 하나로는 이를 분해할 수 없다. 따라서 실습은 IFEval result를 측정된 drop이 아니라 선언한 margin에 대한 non-inferiority check로 명시한다.

**계산 예시 2 — held-out agent family.** [[adding-error-bars-evals]] Eq. 10은 minimum detectable effect를 `δ = (z_{α/2} + z_β) · sqrt((ω² + σ²_A/K_A + σ²_B/K_B)/n)`으로 준다. 여기서 `ω²`는 paired per-task conditional mean의 variance, `σ²`는 mean conditional variance, `K`는 task당 trial 수, `n`은 task 수다. 논문의 illustrative binary-score 값 `σ²_A = σ²_B = 1/6`과 각 checkpoint의 per-task difficulty variance `1/12`를 사용한다(§3.1, §4.2). 여기에 논문에서 쓰는 0.5가 아니라, 이 예시를 위해 선택한 두 checkpoint 간 correlation 0.7을 결합한다. 그러면 `ω² = 2 × (1/12) × (1 − 0.7) = 0.05`다(derived). `α = 0.05`, `β = 0.20`에서 `z_{0.025} + z_{0.20} = 1.96 + 0.8416 = 2.8016`이다. `n = 114`, `K = 1`이면 `δ = 2.8016 × sqrt((0.05 + 0.3333)/114) = 0.162`, 즉 16.2 point다. `K = 4`이면 `δ = 2.8016 × sqrt((0.05 + 0.0833)/114) = 0.096`, 즉 9.6 point다. `K = 4`에서 `δ = 5` point에 도달하려면 Eq. 9에 따라 `n = 2.8016² × 0.1333 / 0.05² = 419` task가 필요하다.

이 수치가 실습의 task budget을 정하며, 작은 gain을 주장하는 대신 memo가 non-inferiority margin을 명시하는 이유다. [figures/generality-gate-power.html](figures/generality-gate-power.html)는 reader가 자신의 `n`, `K`, `ω²`, `σ²`에 대해 `δ`와 `n`을 계산하게 한다. 따라서 실행 뒤에 설명하는 대신, 실행 전에 gate 규모를 정할 수 있다.

**여기서 따르는 두 규칙.** 각 task를 K회 resampling하면 conditional-variance term만 줄어든다. `Var(s_i) = σ²_i / K`다. 논문의 계산 예시에서는 `K = 4`가 total variance를 1/2로 줄이고, K를 무한히 늘려도 달성 가능한 감소량은 2/3이다(§3.1). sampling temperature를 낮추는 것은 대체 방법이 아니다. variance를 없애는 대신 conditional mean으로 옮길 수 있기 때문이다. 논문의 single-token 예시에서는 `T = 1`의 `1/12`에서 `T = 0`의 `1/4`로 `Var(x)`가 올랐다(§3.3). [[tau2-bench]]는 task당 네 trial, temperature 0에서 실행한다. 실습은 benchmark 숫자에는 benchmark 자체 setting을 유지하고, training diagnostic에만 training temperature를 쓴다.

### 7.5 Gate 결정 규칙

실행은 3개 seed에서 다음을 모두 만족하면 pass한다. (a) trained-family reward가 zero를 제외하는 paired interval과 함께 개선된다. (b) held-out family가 선언한 margin 아래로 떨어지지 않는다. (c) 어떤 BFCL category도, 어떤 τ²-bench pass^1 또는 pass^4 값도 선언한 margin 아래로 떨어지지 않는다. (d) 두 abstention rate가 선언한 margin을 넘어서 상승하지 않는다. (e) MMLU-Pro와 IFEval이 각각 non-inferiority margin을 통과한다. (a)를 통과하면서 (b), (c), (d), 또는 (e)에 실패하는 것이 이 실습이 감지하기 위해 존재하는 결과다. 그런 결과는 재-tuning으로 고치지 않고 그대로 보고한다.

## §8 한 agentic failure의 사후 분석

gate에서 점수가 가장 낮은 cell 하나를 고르고, 그 결과를 rollout까지 추적한다. 다음 네 가지 후보 원인과 이를 구별하는 증거를 사용한다.

| 후보 원인 | `rollouts/`에서 이를 구별하는 증거 | 참고 자료 |
|---|---|---|
| Reward hack | task와 무관한 이유로 verifier 자체의 검사를 통과하면서 reward가 높은 경우. 예를 들어 관계없는 file을 수정하면서 선택된 test만 통과시키는 patch | [[deepswe]] §2.3은 masking이 필요한 정확한 예로 이를 든다 |
| Loop | trajectory 길이가 길어질수록 loop-turn rate가 올라가고, 크기 10 이상인 anchor-state group이 증가함 | [[gigpo-verl-agent]] §5.5 |
| Context loss | 마지막 turn의 prompt가 context limit에 가까운 rollout에 failure가 집중되고, truncation rate가 상승함 | §4, §6 |
| Tool hallucination | tool-call validity가 떨어지거나, trained-family reward가 오르는 동안 §7.3의 abstention rate가 상승함 | [[reasoning-trap-tool-hallucination]] Table 3 |

memo에는 이를 뒷받침하는 rollout id와 함께 하나의 causal chain, 그 causal chain을 검증할 변경 하나, 그리고 그 변경을 실제로 실행했는지를 기록한다.

## 음성 샘플과 음성 피드백

**어떤 종류의 음성인가.** 네 가지 구분을 따른다. (1) *음의 한계가치(negative marginal value)*: filter가 버리는 trajectory, (2) *내용으로서의 음성(negative as content)*: 보통의 cross-entropy target 안에 failure를 써 넣는 것, (3) *조건부 입력으로서의 음성(negative as conditioning)*: control token 아래에서 failure를 학습하는 것, (4) *gradient로서의 음성(negative as gradient)*: sample의 likelihood를 명시적으로 낮추는 것이다. 이 실습은 Stage A에서 (1)을, Stage A의 한 slice에서 (2)를, Stage B에서 (4)를 사용한다. 이 장의 source 집합에는 type (3)이 등장하지 않는다. 이는 확정된 선택이 아니라 agentic training에서 아직 열린 질문이다.

**음성은 어디에서 오는가.** Stage A에서는 trajectory generator와 verifier에서 온다. [[agenttuning]]은 `r = 1`만 남겨 35,341개 instruction에서 trajectory 1,866개를 남긴다(Table 1). [[toucan-mcp]]은 failed tool response를 포함한 trajectory를 버린다(§3.1). [[swe-smith]]는 해결된 rejection-sampling attempt 36% 중 많아야 3개만 남긴다(§3). Stage B에서는 environment verifier에서 온다. test suite, assertion, 또는 answer check가 0을 반환한다. 이 source들 중 어느 것도 false-negative rate를 보고하지 않는다.

**실무에서 이들을 어떻게 다루는가.**

- *버리기(discard)* (type 1)가 Stage A의 기본값이다. 대안에 대한 한 측정은 [[agenttuning]] Table 2다. 7B에서 unfiltered trajectory는 held-in 1.34, held-out 0.47을 얻었고, filtered의 1.96, 0.65와 비교된다.
- *내용으로 사용하기(use as content)* (type 2)는 [[toucan-mcp]] Ext.1에 등장한다. server metadata를 섞어 주어진 tool로는 task를 풀 수 없게 만든 뒤, tool call이 0인 trajectory만 남긴다. 그 결과 119.3K SFT subset 안에 irrelevance instance 40K개가 들어간다. 7B Table 2에서 측정된 교환은 irrelevance `67.93 → 75.18`, relevance `72.22 → 66.67`이다. abstention training과 행동하려는 의지는 함께 움직이므로 둘 다 gate에 있어야 한다.
- *gradient로 사용하기(use as gradient)* (type 4)가 Stage B다. failed rollout은 §5에서 계산한 group baseline을 통해 negative advantage를 받는다.

**메커니즘.** target token `y`에 대한 logits `z`의 softmax에서 log-probability gradient는 다음과 같다.

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

여기서 `p_j`는 token `j`에 대한 model의 probability다. negative advantage는 부호를 뒤집으므로 update는 `p_y`를 낮추고, 다른 token의 probability는 현재 probability에 비례해 올린다. 따라서 rejected continuation에서 빠진 mass는 model이 이미 가장 가능성 높다고 생각한 다음 선택지로 간다. agentic setting에서 rejected tool call의 가장 가능성 높은 대안은 흔히 다른 tool call 또는 tool call 없는 일반 답변이다. 이것이 [[reasoning-trap-tool-hallucination]]이 관찰한 방향이다. tool 없이 답하는 것도 hallucination으로 세는 No-Tool-Available rate는 tool RL 뒤 `34.8%`에서 `90.2%`로 올랐다(Table 3). 이 메커니즘이 관찰을 설명한다는 것은 **해석(Interpretation)** 이다. 논문은 representational analysis를 보고한다(tool input에서 early 및 middle layer의 CKA가 `0.75` 미만, §5.1). 그러나 token-level account는 보고하지 않는다.

**이득과 failure에 대한 증거.**

- 이득: binary outcome reward를 쓰는 GRPO는 failed rollout 각각에 대해 negative-as-gradient가 된다. 1.5B에서 ALFWorld overall success를 4.1(prompting)에서 `72.8 ± 3.6`으로, WebShop success를 5.2에서 `56.8 ± 3.8`으로 올렸다([[gigpo-verl-agent]] Table 1). DeepSWE는 같은 종류의 signal로 약 200 RL step에서 SWE-Bench-Verified pass@1을 23%에서 42%로 올렸다([[deepswe]] Fig. 2).
- failure: 위 abstention 결과가 있고, [[reasoning-trap-tool-hallucination]] §6의 DPO mitigation도 있다. fabricated tool call을 rejected response로 둔 pair는 No-Tool-Available을 `90.2%`에서 `55.8%`로, Distractor-Tool을 `100.0%`에서 `71.4%`로 낮췄다. 반면 SynTool task reward는 `0.45`에서 `0.34`로 떨어졌다. negative gradient는 겨냥한 failure를 고쳤고 target performance를 대가로 지불했다.

**이 실습이 적용하는 control.**

1. verifier outcome이 없는 trajectory는 벌점 대신 *mask*한다([[deepswe]] §2.3). baseline treatment는 기록한다(§5).
2. step-level advantage가 있을 때 negative signal을 trajectory 전체가 아니라 해당 step에 *국소화(localize)* 한다([[gigpo-verl-agent]] Eq. 7-8). source의 weighting coefficient `ω`는 1로 고정되어 있고, WebShop sweep은 `ω = 0.8`에서 peak다(Table 5).
3. clipped objective로 update를 *제한(bound)* 한다. [[deepswe-recipe]]의 released script는 lower clip 없이 `clip_ratio_high` `0.28`을 기록한다.
4. *on-policy*를 유지한다. sampler-learner gap을 log하고 bound한다([[rollout-training-mismatch-tis]]).
5. abstention은 positive target으로 anchor한다. [[toucan-mcp]]의 irrelevance instance는 tool을 호출하지 않는 것에 대한 positive cross-entropy target이며 RL reward가 아니라 SFT mixture에 들어간다.

**진단.** chosen trajectory와 rejected trajectory의 return을 따로 log한다. advantage sign별 entropy, zero-variance group 비율, tool-call validity, 두 abstention rate를 log한다. trained-family success는 `k = 1`과 `k = 4` trial에서 보고한다. `k = 1`은 올랐지만 `k = 4`가 오르지 않는 경우는 [[tau2-bench]]가 설명한 더 낮은 consistency 패턴이기 때문이다(§4.2).

**효과 크기에 대해 정직하게 말하기.** 이 장의 source 집합 중 agentic gain을 positive-sample share와 negative-sample share로 분해하는 것은 없다. 비교 가능한 분해는 single-turn reasoning에만 있으며 [[ch-43a]]가 다룬다. 따라서 이 실습은 negative가 agentic gain을 일으킨다고 주장하지 않는다. 대신 처리 방법이 달라질 때 무엇이 바뀌는지 측정한다. 그것이 §5 axis다.

## 레시피

모든 행은 명시한 date에 명시한 locus에서 읽었다. “task당 sample”은 완전한 multi-turn trajectory를 뜻하고, “example”은 SFT instance를 뜻한다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | source 위치 | 상태 | 이 값의 근거 | 실습 값 | 차이의 이유 |
|---|---|---|---|---|---|---|---|---|---|
| AgentLM-7B / 13B | 7B; 13B | SFT | agent sampling ratio `η` | 0.2 | arXiv:2310.12823v2 §2.2.2, App. A ([[agenttuning]]) | 2026-09-14 검증됨 | 7B에서 `η`를 0.1 step으로 0–1 scan했고 held-out가 최고였으나 η별 score는 표로 제시하지 않음 | `η ∈ {0.2, 0.5}` | share가 검증할 axis임 |
| AgentLM-7B / 13B | 7B; 13B | SFT | peak LR; batch(unit 미기재); sequence; warmup ratio; schedule | 5e-5; 64; 4,096; 0.02; cosine | §2.2.3, App. A Table 6 | 2026-09-14 검증됨 | ablation 보고 없음 | 사용한 slice의 LR(다음 두 행 참고); batch 64; sequence 32,768 | agent trajectory가 4,096 token을 넘음 |
| AgentLM-7B / 13B / 70B | 전체 | SFT | loss masking | model output에만 loss | §2.2.3 | 2026-09-14 검증됨 | ablation 보고 없음 | 동일, 응답 mask unit test 추가 | §3.1 |
| Toucan SFT models | 7B / 14B / 32B | SFT | LR; epoch; effective batch; optimizer; max sequence | 2e-5; 2; 64; AdamW (0.9, 0.999) ε 1e-8; 32,768 | arXiv:2510.01179v1 App. C.2 Table 5 ([[toucan-mcp]]) | 2026-09-15 검증됨 | ablation 보고 없음 | 7B에서 LR 2e-5, 2 epoch | mixture가 가져오는 slice와 일치 |
| SWE-agent-LM-7B / 32B | 7B; 32B | SFT | LR; epoch; max context; hardware | 5e-5; 많아야 3; 32,768; H100 2–8개 | arXiv:2504.21798v2 App. F.1 ([[swe-smith]]) | 2026-09-15 검증됨 | Fig. 4, Fig. 5, Table 4-5는 이 값이 아니라 data ablation | 사용 안 함 | 다른 두 slice보다 2.5× 큰 LR를 조정하지 않음 |
| ADP-trained models | 7B / 8B / 14B / 32B | SFT | corpus; sampling multiplier | dataset 13개, trajectory 1.3M개; `w_d`는 0.001(orca agentinstruct)부터 3(swe-gym openhands) | arXiv:2510.24702v2 §2.1, App. C Table 9 ([[agent-data-protocol]]) | 2026-09-15 검증됨 | Table 6, Table 10: 약 30K sample에서 ADP 16.6, SWE-smith-only 11.0 | multiplier를 재사용하지 않고 family를 동일 비율로 섞음 | 실습의 family 수는 13이 아니라 3 |
| Qwen2.5-1.5B/7B-Instruct + GiGPO | 1.5B; 7B | RL | LR; group size `N`; rollout당 group; rollout / validation temperature; mini-batch; KL loss coefficient; `γ`; `ω` | 1e-6; 8; 16; 1.0 / 0.4; 256(ALFWorld), 64(WebShop); 0.01; 0.95; 1 | arXiv:2505.10978v3 App. E.1 ([[gigpo-verl-agent]]) | 2026-09-15 검증됨 | Table 5의 WebShop `ω` sweep은 `ω = 0.8` peak, §5.2의 `F_norm` comparison | 동일, `F_norm = 1` 고정 | `F_norm`은 task-dependent(§5.2)라서 고정해야 §5만 negative-signal axis가 됨 |
| Qwen2.5-1.5B/7B-Instruct + GiGPO | 1.5B; 7B | RL | reward; invalid-action penalty; episode step limit; max prompt / response token | 10 / 0; −0.1; 50(ALFWorld), 15(WebShop); 2,048 / 512 및 4,096 / 512 | App. E.1 | 2026-09-15 검증됨 | ablation 보고 없음 | 동일 reward와 penalty; family별 step limit | invalid-action penalty를 고정해 §5를 분리 |
| Qwen2.5-1.5B/7B-Instruct + GiGPO | 1.5B; 7B | RL | prompt 안의 history | 마지막 observation-action pair 2개(ALFWorld, WebShop); full history(search QA) | App. E.2 | 2026-09-15 검증됨 | ablation 보고 없음 | keep-recent-2는 세 arm 중 하나 | context policy가 검증할 axis(§4) |
| Qwen2.5-1.5B / 7B + GiGPO | 1.5B; 7B | RL | compute; iteration | H100 2개(1.5B), 4개(7B); 150 iteration | App. E.1 | 2026-09-15 검증됨 | 해당 없음 | 동일 | constrained path가 이 scale을 목표로 함 |
| DeepSWE-Preview | 32B | RL | LR; clip_ratio_high; task당 trajectory; rollout temperature; entropy coefficient; KL | 1e-6; 0.28; 8; 1.0; 0.0; `use_kl_loss=False` | rllm@709dec43b740 `scripts/agent/swe/deepswe_32b.sh` L24, L34, L49, L47, L53, L33 ([[deepswe-recipe]]) | 2026-09-14 검증됨(script만; blog는 이 값을 전혀 인쇄하지 않음) | ablation 보고 없음 | 7B에서 LR 1e-6, task당 trajectory 8개; GiGPO 행의 KL loss coefficient 0.01 | 실습은 가까이 두고 싶은 SFT checkpoint에서 시작하므로 KL term을 유지 |
| DeepSWE-Preview | 32B | RL | masked trajectory | blog: max context, max environment step, 20-minute generation timeout; script: `agent.overlong_filter=True`, `algorithm.mask_truncated_samples=False` | Blog §2.3; script L55, L70 | 2026-09-14 검증됨 | Blog Fig. 6(Qwen3-14B, curve만) | mask와 penalize를 둘 다 실행 | treatment가 검증할 axis(§5) |
| DeepSWE-Preview | 32B | RL | max response length; max environment step(training) | 32,768; 50 | script L18-L19, L69 | 2026-09-14 검증됨(script만) | ablation 보고 없음 | 위 step limit에 따라 family별 설정 | environment가 다름 |
| τ²-bench | 해당 없음 | eval-gate | task당 trial; temperature; telecom task 수 | 4; 0; 생성 2,285개 중 평가 114개 | arXiv:2506.07982v1 §4.1, Table 1 ([[tau2-bench]]) | 2026-09-15 검증됨 | 해당 없음 | 동일 | benchmark setting은 바꾸지 않음 |
| MMLU-Pro | 해당 없음 | eval-gate | prompting; item; answer extraction | 5-shot CoT; question 12,032개; regex 2개 후 random option | arXiv:2406.01574v6 §4 ([[mmlu-pro]]) | 2026-09-15 검증됨 | §6.3: 24-prompt range 약 2%, maximum 3.74% | 동일, prompt 하나와 extraction regex 하나 고정 | fallback은 format failure를 chance로 점수화 |
| IFEval | 해당 없음 | eval-gate | item; metric | prompt 541개, instruction type 25개; prompt-level strict accuracy | arXiv:2311.07911v1 §1, §2.2 ([[ifeval]]) | 2026-09-15 검증됨 | 해당 없음 | 동일 | — |

**작은 general-purpose run의 시작점.** 1.5B instruct checkpoint에서는 `η = 0.2` mixture로, model output에만 loss를 주면서 maximum sequence 32,768, effective batch 64, LR 2e-5로 2 epoch SFT한다. 이어 RL은 LR 1e-6, task당 trajectory 8개, rollout당 task 16개, rollout temperature 1.0, validation temperature 0.4, KL loss coefficient 0.01, discount 0.95, `F_norm = 1`, success reward 10과 failure reward 0, invalid action당 −0.1, family별 step limit으로 실행한다. 이 숫자들은 모두 위 `검증됨` 행에서 왔다. SFT 값은 119.3K instance에서 진행한 Toucan 7B–32B run, `η`는 Llama-2-chat에서 AgentTuning이 7B로 scan한 값, RL 값은 H100 2개로 150 iteration을 수행한 ALFWorld와 WebShop의 GiGPO Qwen2.5-1.5B run에서 왔다. 이 조합 자체를 위해 tuning된 값은 하나도 없다.

## 일반화 관점

**(a) 범위를 넓히는 것.**

- agent data만 학습하지 않고 general instruction data와 섞기: 7B에서 `η = 0.2`의 held-out 0.67, general 0.63은 agent-only의 0.09, 0.22와 비교된다([[agenttuning]] Table 5).
- trajectory 수를 700으로 고정한 채 한 environment가 아니라 여러 environment에서 agent slice를 가져오기: repository 4, 25, 50, 100개에서 뽑을 때 SWE-Bench Verified 해결 비율은 각각 10.3%, 11.5%, 12.9%, 15.1%다([[swe-smith]] §4.1, Fig. 5).
- 같은 scale에서 agent-data *source*를 섞기: OpenHands harness를 쓴 Qwen-3-8B는 약 30K ADP sample에서 SWE-Bench Verified 16.6을, up-sample한 SWE-smith 단독에서는 11.0을 얻는다([[agent-data-protocol]] Table 10).
- explicit abstention data를 positive target으로 쓰기: Toucan의 irrelevance extension은 7B BFCL irrelevance를 67.93에서 75.18로 올린다([[toucan-mcp]] Table 2).
- trained family 안에서 multi-turn RL 자체: 1.5B GRPO에서 ALFWorld overall은 `4.1 → 72.8 ± 3.6`이다([[gigpo-verl-agent]] Table 1).

**(b) 범위를 좁히거나 forgetting을 일으키는 것.**

- agent slice를 한 environment에 집중하기: 7B에서 SymPy trajectory 700개는 SymPy subset을 13.6에서 21.2로 올리고 Verified-without-SymPy는 15.3에서 14.0으로 내린다. 32B에서는 SymPy `33.3 → 42.4`, 나머지 `40.2 → 38.3`이다([[swe-smith]] §4.1).
- agent-only SFT: 7B와 13B의 general score는 0.22, 0.19이고 mixture의 0.63, 0.69과 비교된다([[agenttuning]] Table 5). mixture도 70B general benchmark를 양 방향으로 움직인다. MMLU `62.1 → 59.5`, HumanEval `30.8 → 28.7`, GSM8K `54.7 → 59.7`, MT-Bench `6.85 → 7.26`이다(Table 4).
- tool-use SFT가 행동 의지를 이동시키기: Toucan은 7B에서 multi-turn을 12.88에서 22.62로 올리고 non-live AST를 84.19에서 78.52로, relevance를 72.22에서 66.67로 내린다([[toucan-mcp]] Table 2).
- tool 관련 여부와 무관한 reasoning RL이 tool hallucination을 올리기: SynTool에서 think-then-act GRPO 뒤 `34.8% → 90.2%`, `54.7% → 100.0%`이며 GSM8K 단독 GRPO에서도 두 rate가 오른다([[reasoning-trap-tool-hallucination]] Table 3, Fig. 3). `<think>` block을 제거하면 effect와 task reward가 함께 낮아진다(reward 0.45에서 90.2 / 100.0인 것에 대해 reward 0.28에서 41.4 / 63.6, Table 2).
- 한 language 또는 한 runtime에서만 training하기: SWE-agent-LM-32B는 SWE-bench Multilingual에서 8.4%를 얻었고 Claude 3.7 Sonnet은 43%였다. edit가 “Python에 더 가까운 syntax를 반영했다”([[swe-smith]] App. F.4).

**(c) 이 단계에서 generality를 재는 방법.** paired interval(§7.1)이 있는 held-out environment family, 하나의 BFCL 숫자가 아니라 고정 mode별 per-category tool-calling score(§7.2), `pass^1`과 `pass^4`가 서로 다른 방향으로 움직일 수 있으므로 둘 이상의 `k`에서의 `pass^k`([[tau2-bench]] §4.2), target benchmark가 변화를 드러내지 못했다는 연구가 있었으므로 필요한 tool이 없는 prompt에서의 abstention check([[reasoning-trap-tool-hallucination]] §4.4.2), 그리고 §7.4 power calculation으로 크기를 정한 non-inferiority margin을 선언한 non-agentic suite를 사용한다. 알려진 measurement error는 다음과 같다. task 114개에 task당 trial 4개의 `pass^k`는 약 9.6 point를 분해한다(§7.4). MMLU-Pro는 reasonable prompt 24개에 걸쳐 약 2% 움직인다([[mmlu-pro]] §6.3). τ²-bench telecom user simulator 자체는 error rate 16%, critical error 6%다([[tau2-bench]] §1). abstention judge의 human agreement는 보고되지 않았다([[reasoning-trap-tool-hallucination]] App. A.4).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| Tool-response token이 loss를 가짐 | 같은 data size의 single-turn run보다 training loss가 더 빨리 떨어지고, model이 tool output처럼 보이는 text를 내기 시작함 | step 1 전에 모든 tool 및 template token에서 response mask가 0인지 assert한다([[verl-rollout]] `continuous_token.py` L390-413) |
| turn 사이에 history를 concatenate하지 않고 re-tokenize함 | 다른 anomaly 없이 reward가 평평하거나 하락함 | rollout의 token id를 chat template을 다시 적용해 만든 id와 비교한다. verl documentation은 single-turn training에서도 이 때문에 non-convergence가 난다고 보고함 |
| truncated rollout을 말하지 않고 failure로 점수화함 | verified failure의 negative advantage가 group composition으로부터 계산되는 값과 다름 | §5 table로 group 하나의 advantage를 손으로 다시 계산하고, 모든 rollout의 termination reason을 log함 |
| environment family 전체를 합쳐 reward curve를 그림 | aggregate reward는 오르지만 한 family는 0으로 떨어짐 | family별 reward를 그린다. §7.1 gate는 held-out family만 사용함 |
| task당 trial 없이 gain을 보고함 | `pass^1`은 오르지만 `pass^4`는 오르지 않음 | [[tau2-bench]] §4.2에 따라 trial count와 함께 둘 다 보고함 |
| score를 안정적으로 보이게 하려고 temperature를 낮춤 | score variance는 줄지만 mean도 이동함 | benchmark 자체 temperature를 유지한다. [[adding-error-bars-evals]] §3.3의 single-token example은 T가 1에서 0으로 갈 때 `Var(x)`가 `1/12`에서 `1/4`로 오른다고 보임 |
| paired run에 unpaired interval을 씀 | interval이 넓어 모든 gate가 통과함 | paired form을 쓴다. `n = 541`, correlation 0.7에서 half-width는 5.84에서 3.20 point로 내려감(§7.4) |
| gate suite가 training slice와 겹침 | held-out family가 trained family만큼 개선됨 | gate 전 §2.4의 n-gram decontamination을 실행한다. 세 data source 어느 것도 agent benchmark에 대해 그런 check를 보고하지 않음 |
| format breakage를 capability drop으로 읽음 | 다른 변화 없이 MMLU-Pro가 몇 point 하락함 | failed answer extraction 수를 센다. harness는 random option을 선택해 chance로 점수화함([[mmlu-pro]] §4) |
| abstention을 capability가 아니라 safety로 취급함 | abstention rate가 오르는 동안 relevance가 하락함 | irrelevance와 relevance를 함께 보고한다. Toucan은 둘을 반대 방향으로 움직였음([[toucan-mcp]] Table 2) |

## 이해 점검

1. 7B와 13B에서는 agent data와 general data를 섞은 held-out agent score가 general-only training과 가깝지만, 70B에서는 mixture가 훨씬 앞선다(0.67 대 0.64, 0.78 대 0.81, 1.40 대 0.98). 이 패턴은 1.5B 실습이 agent-data share에 관해 어떤 주장을 할 자격이 있고, 어떤 주장은 할 자격이 없다는 뜻인가?
2. truncated rollout의 loss를 mask하는 것과 group baseline에서 그것을 빼는 것은 서로 다른 operation이다. `n = 8`, success 2개, failure 4개, truncated rollout 2개인 leave-one-out advantage를 사용해 둘 다 계산하라. 어떤 verified rollout이 영향을 받고 어느 방향인지 설명하라.
3. 한 run에서 trained-family reward가 오르고, zero-variance group 비율도 오르며, iteration당 gradient를 운반하는 trajectory 수는 감소한다. 이 세 관찰과 양립하는 서로 다른 원인 두 가지와, 둘을 구별하는 measurement 하나를 제시하라.
4. Reasoning Trap 연구는 training data에 tool이 전혀 없는데도 GSM8K 단독 GRPO가 tool-hallucination rate를 올렸다고 밝혔다. softmax logit gradient를 사용해, output 분포 하나에서의 negative gradient가 어떻게 다른 input 분포에서 behaviour를 바꿀 수 있는지 mechanism 수준에서 설명하라. 그리고 논문이 account의 어느 부분을 측정했고 어느 부분을 측정하지 않았는지 명시하라.
5. Toucan SFT는 같은 checkpoint에서 BFCL irrelevance를 67.93에서 75.18로 올리고 relevance를 72.22에서 66.67로 내렸다. BFCL overall score만 보고하면 general-purpose model이 여기서 알아야 할 사실이 왜 가려지는가? 그 교환이 필연적인지 검증하려면 training mixture에 무엇을 더해야 하는가?
6. held-out environment family에는 task 114개가 있다. `α = 0.05`에서 80% power로 5-point change를 탐지하려 한다. §7.4 값을 이용해 task 수 또는 task당 trial 수가 얼마나 필요한지 계산하고, task당 trial을 늘리는 것이 task 수를 늘리는 것과 달리 한계에 부딪히는 이유를 설명하라.
7. GiGPO는 크기 10 이상 anchor-state group이 iteration 10에서 20%를 넘고 iteration 75에는 떨어진다고 보고한다. 각 regime에서 policy가 무엇을 하고 있는지, 그리고 이 distribution이 trajectory length만 쓰는 것보다 왜 더 좋은 loop instrument인지 설명하라.
8. DeepSWE는 SWE-Bench-Verified 밖의 evaluation을 보고하지 않는다. 이 장의 gate를 실행했고 held-out family가 하락했다고 가정하자. DeepSWE가 밝힌 design choice 중 첫 번째 candidate 두 개는 무엇이고, rerun에서 둘을 분리하려면 무엇을 바꾸겠는가?

## 연결

- **이전 장.** [[ch-46]] — *Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention*. 이 실습은 single-turn preference 또는 RLVR stage를 측정한다. 이 장은 turn, environment, verdict 없이 끝날 수 있는 rollout을 추가한다.
- **다음 장.** [[ch-47]] — *Evaluation Harness and Suite Design for General Capability*. §7에서 임시로 조립한 suite를 그 장에서 체계적으로 설계한다.
- **의존성.** [[ch-46]]은 negative-signal ablation design, paired-interval protocol, §7이 agent로 확장하는 held-out retention suite를 제공한다.
- [[ch-29c]] — *Agentic Environment and Task Synthesis at Scale*: Toucan, ADP, SWE-smith slice가 나오는 곳과 새 environment를 만드는 방법.
- [[ch-30b]] — *Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares*: §2의 `η` axis 뒤의 mixture theory.
- [[ch-45b]] — *Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability*: step-level credit assignment와 이 실습이 계측만 하는 stability failure를 포함해 §3과 §5의 mechanics를 전부 다룸.
- [[ch-45c]] — *Context Management for Long-Horizon Agents*: trained summarization을 포함해 §4에서 비교하는 context policy.
- [[ch-45d]] — *Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL*: 어떤 lab이 어느 stage에 agentic capability를 놓고, 그 뒤 general ability를 무엇으로 보고하는지.
- [[ch-43a]] — *Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages*: negative-feedback section의 derivation.
- [[ch-51a]] — *Evaluating Agent Generality and Reliability*: `pass^k`와 harness-level agent measurement을 포함한 §7의 evaluation counterpart.
- [[ch-30a]] — *Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control*: memo가 재사용하는 forgetting-report format.
- [[ch-36]] — *Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split*: §3.1 masking unit test의 multi-turn version.

## 출처

- [[agenttuning]] — agent-share axis: mixture objective, `η = 0.2` scan, 세 size에서 agent-only / general-only / mixed를 비교한 Table 5.
- [[toucan-mcp]] — MCP-server trajectory slice, SFT hyperparameter, irrelevance extension, 이것이 만드는 BFCL category trade.
- [[agent-data-protocol]] — 통합 multi-source agent corpus와 mixed agent SFT와 single-source agent SFT의 equal-scale comparison.
- [[swe-smith]] — environment-family specialization과 repository-pool diversity, cross-language transfer limit.
- [[gigpo-verl-agent]] — constrained path가 사용하는 multi-turn RL configuration, episode 및 step advantage formula, loop instrument인 anchor-state group distribution, iteration당 cost breakdown.
- [[deepswe]] 및 [[deepswe-recipe]] — verdict 없는 trajectory의 compact filtering, leave-one-out advantage, released script value, target benchmark 밖 evaluation이 없었다는 사실.
- [[tau2-bench]] — gate의 dual-control slice, `pass^k` 정의와 trial protocol, user-simulator error rate.
- [[bfcl]] — per-category tool-calling evaluation, function-calling과 prompting mode effect, relevance / irrelevance split.
- [[reasoning-trap-tool-hallucination]] — abstention check와, tool-calling 및 instruction-following benchmark가 그것이 재는 변화를 드러내지 못했다는 결과.
- [[adding-error-bars-evals]] — paired standard error, resampling relation, temperature warning, gate 크기를 정하는 power formula.
- [[mmlu-pro]] — item 수, prompting protocol, extraction failure 때 random-option fallback, prompt-variation range.
- [[ifeval]] — item 수, instruction type, regression check에서 보고하는 metric.
- [[verl-rollout]] — response-mask contract, token-in-token-out generation, re-tokenization warning.
- [[entropy-logging-patterns]] — verl에서 `actor/entropy`와 KL metric의 의미, 실습이 기록해야 하는 framework default.
- [[rollout-training-mismatch-tis]] — stop condition으로 log하는 sampler-learner probability gap.
- [[grpo]] — 실습 algorithm이 구축한 group-baseline advantage.
- [[dr-grpo]] — standard-deviation normalization에 반대하는 논거. 이것이 `F_norm = 1`을 고정하는 이유.
- [[r2e-gym]] 및 [[swe-gym]] — RL stage의 대체 task supply로 언급되는 executable SWE environment set.
