<!-- chapter: ch-32d
     track: midtraining
     kind: content
     title: Post-Training 이전의 저장소·실행 추적·궤적 데이터를 쓰는 Agentic Mid-Training
     deps: [ch-32b]
     sources: [[glm-4-5]], [[glm-4-5-recipe]], [[glm-5]], [[agentfounder]], [[cwm-code-world-model]], [[kimi-dev]], [[midtool]], [[state2state]], [[skill-pretraining]], [[kimi-k2]], [[kimi-k2-recipe]], [[kimi-k2-agentic-data]], [[kimi-k2-5]], [[agent-early-experience]]
     figures: figures/agentic-stage-map.html
     revised: 2026-09 (범용성 개정)
-->

# Chapter 32d — Post-Training 이전의 저장소·실행 추적·궤적 데이터를 쓰는 Agentic Mid-Training

> **핵심 통찰.** 이 장에서 읽은 보고서에서 모델이 생성한 agent 궤적은 mid-training 또는 continued pre-training 단계에 들어간다(GLM-4.5는 128K, GLM-5는 128K와 200K, CWM은 131K, AgentFounder는 32K와 128K, Kimi-Dev는 32K, MidTool은 8K). 어떤 보고서도 이것을 주된 일반 pre-training 단계에 넣지 않는다. Kimi K2는 agentic data synthesis를 pre-training이 아니라 post-training SFT에 둔다([[kimi-k2]] §2.2, §3.1.1). 통제된 비교는 동일한 post-training 뒤의 agentic 이득을 보여 준다. 8B mid-training mix에 ForagerAgent 궤적을 더하면 SWE-bench Verified가 +3.7점이었고([[cwm-code-world-model]] Table 4), 20.3B-token tool-use mix는 4B에서 BFCL overall을 +10.5점 높였다([[midtool]] Table 6). 동일 token budget에서 일반 능력을 측정한 연구는 하나뿐이다. skill-document mid-training은 6-benchmark general average를 −0.85에서 +0.51만큼 바꿨고, trajectory mid-training은 한 3B 설정에서 general-data control 대비 MMLU를 4.12점 낮췄다([[skill-pretraining]] Table 1).
>
> **지침.** agentic 데이터를 mid-training 단계에 더할 때는 일반 데이터를 mix에 남기고, 같은 token 수의 general-data control에 맞춰 그 비율을 고르라. CWM은 표준 평가 성능을 유지하는 데 이것이 필수였다고 하여 pre-training mix의 30%를 rehearsal로 유지했고([[cwm-code-world-model]] §4.2), SPT에서는 30% skill / 70% general mixture가 순수 skill data보다 agentic score가 높았다(1.6B에서 38.60 대 22.06, [[skill-pretraining]] Table 12). 보고서가 agentic mid-training이 일반 능력에 미친 효과를 주장하면, 같은 checkpoint를 그 단계 전후에 non-agentic benchmark로 평가했는지 확인하라. 그렇지 않으면 그 효과는 측정되지 않은 것으로 취급하라. agent 궤적이 training window보다 길면 잘라내지 말고 long-context stage에 넣어야 한다. AgentFounder의 128K 두 번째 stage는 같은 50B-token budget에서 일부 데이터를 truncate한 단일 32K stage보다 Table 4의 여섯 metric 모두에서 높은 점수를 냈다([[agentfounder]] Table 4).

## 범용 모델에서 이 장이 중요한 이유

Mid-training은 pre-trained checkpoint를 post-training 전에 curated 또는 domain-weighted data로 계속 학습하는 것이다([[skill-pretraining]], Related Work). Agentic mid-training은 환경에서 행동하는 내용의 데이터를 더한다. 여기에는 issue, pull request(PR), commit이 붙은 repository code, code와 program state를 짝지은 execution trace, reasoning·tool call·tool 또는 environment response의 연속인 agent trajectory, tool documentation, 그리고 한 논문에서는 environment-derived goal을 쓰는 reinforcement learning(RL)이 포함된다.

pre-training → mid-training → SFT → preference optimization → RL → evaluation pipeline에서 이 장은 context extension(ch-32b) 뒤에 놓인다. 긴 agent 궤적은 long-context data이기 때문이다. 여기서 측정해야 하는 문제는 세 가지다.

1. **배치 오류(placement errors).** agentic 데이터가 어느 training stage에 들어가는지에 대한 주장은 stage 하나 전체만큼 틀릴 수 있다. ch-27의 이전 버전과 검증되지 않은 card [[kimi-k2-agentic-data]]는 Kimi K2가 synthetic agent trajectory 약 1T token을 3–5% 비율로 pre-training에 섞었다고 쓴다. Kimi K2 보고서는 15.5T-token pre-training corpus를 web text, code, mathematics, knowledge로 설명하며, agentic data synthesis는 SFT에 둔다([[kimi-k2]] §2.2, §3.1.1).
2. **협소화(narrowing).** 좁은 agentic distribution으로 mid-training하면 그 분포 밖 점수가 낮아질 수 있다. 이 효과의 크기는 §8과 §9의 한 연구가 측정한다.
3. **귀속(attribution).** GLM-5와 GLM-4.5 비교에서는 model size, data, token count, attention type이 한꺼번에 바뀌며, CWM은 mid-training 이전 checkpoint의 row를 출력하지 않는다(§2, §4). 그러므로 agentic mid-training 뒤 base-model score만으로는 matched control 없이는 agentic data에 효과를 귀속할 수 없다.

[stage map figure](figures/agentic-stage-map.html)는 독자가 이 장의 보고서를 agent trajectory가 들어가는 위치로 filter하고, 각 보고서가 general ability에 대해 무엇을 측정했는지 확인하고, upsampled source와 long trajectory의 단위 변환을 확인하게 한다.

## §1 Stage map: 보고서마다 agentic 데이터가 들어가는 위치

**정의.** Stage map은 한 모델에 대해 pre-training, mid-training, SFT, RL에서 어떤 data type을 썼는지와 각 stage의 token budget 및 context length를 나열한다.

**문제.** 각 항목의 locus가 없으면 stage placement가 문서 사이에서 복사되고 오류가 퍼진다. 측정 가능한 확인 방법은 각 항목이 primary report의 명시된 section에서 발견되는지다.

| 보고서 | 주된 pre-training | Mid-training 또는 continued pre-training | Agent trajectory가 처음 등장하는 stage | 근거 위치 |
|---|---|---|---|---|
| Kimi K2 (총 1.04T / activated 32B) | 4K에서 15.5T: web, code, math, knowledge | 4K에서 anneal 400B, 이후 32K에서 60B; YaRN으로 128K; agent data는 나열하지 않음 | SFT (synthetic tool-use trajectories) | [[kimi-k2]] §2.2, §2.5, §3.1.1 |
| Kimi K2.5 | 4K에서 vision-text joint 15T; repository code, issue, code review, commit으로 code upweight; GUI screenshot과 “human-annotated demonstrations를 포함한” action trajectory | 32K → 262K, 500B → 200B: high-quality text·multimodal, long text, long video, reasoning, long-CoT | pre-training (GUI action trajectories; 비율과 model-generated fraction은 보고하지 않음) | [[kimi-k2-5]] Table 3, App. B.2-B.3 |
| GLM-4.5 (activated 355B / 32B) | 4K에서 15T + 7T | 32K에서 500B repository code; 32K에서 500B synthetic reasoning; 128K에서 100B long document + synthetic agent trajectory | mid-training, 128K stage | [[glm-4-5]] §2.3, Figure 3 |
| GLM-5 (activated 744B / 40B) | 27T corpus | 32K에서 1T; 128K에서 500B; 200K에서 50B; 이후 stage에서 agent trajectory up-sample | mid-training, 이후 stages | [[glm-5]] §1, §2.3 |
| CWM (32B dense) | 8K에서 8T | 131K에서 5T: 30% CWM-specific(trace, ForagerAgent), 40% code, 30% rehearsal | mid-training | [[cwm-code-world-model]] Figure 1, §4.2 |
| AgentFounder-30B (Qwen3-30B-A3B-Base에서 시작) | 논문 범위에 없음 | 32K에서 약 200B; 128K에서 100B | continued pre-training | [[agentfounder]] §2.1 |
| Kimi-Dev (Qwen2.5-72B-Base에서 시작) | 논문 범위에 없음 | 32K에서 약 150B를 봄 | mid-training | [[kimi-dev]] §3.2, App. A |
| MidTool (Qwen3-4B/8B-Base에서 시작) | 논문 범위에 없음 | 8,192에서 20.3B | mid-training | [[midtool]] Table 2, Table 10 |

**결과.** 위 보고서 중 post-training 전에 model-generated agent trajectory를 쓰는 모든 경우, 그것은 주된 pre-training phase 뒤에 들어간다. Kimi K2.5만 action trajectory를 pre-training corpus에 둔 보고서다. 이 보고서는 collection이 human-annotated demonstration을 포함한다고 말하지만 token share는 주지 않는다([[kimi-k2-5]] App. B.3). “어떤 frontier report도 synthetic agent trajectory를 pre-training에 두지 않는다”라는 문장은 여기서 읽은 보고서 집합으로 한정된다(Result, report별). 이 집합 밖 부재 주장은 Open question이다.

**ch-27에 대한 정정(학습자가 읽은 버전).** “Kimi-K2의 pretraining-mix injection”과 “top-scoring trajectory가 약 3–5%로 pretraining mix에 들어간다” → Kimi K2 pre-training은 rephrasing과 함께 web text, code, mathematics, knowledge를 쓴다. tool-use synthesis는 3,000개 이상의 실제 MCP tool, 20,000개가 넘는 synthetic tool, simulated user와 tool, LLM judge로 SFT data를 만든다([[kimi-k2]] §2.2, §3.1.1; card Verification은 “agentic trajectory의 pre-training data 주입”을 not reported로 열거한다). 근거가 확인된 agentic mid-training은 GLM-4.5, GLM-5, CWM, AgentFounder, Kimi-Dev에서 나온다.

**함의.** “agent trajectory를 pre-training에 넣는다”는 recipe를 복사하는 것은 이 보고서들에서 뒷받침되지 않는다. 근거가 있는 배치는 long-context mid-training stage(GLM, CWM, AgentFounder), 또는 released base model에서 시작하는 domain mid-training stage(Kimi-Dev, MidTool)다.

## §2 GLM-4.5와 GLM-5: 32K의 저장소 데이터, long-context stage의 agent 궤적

**정의.** Repository-level code training은 하나의 repository에 속한 file들과, 관련 issue·PR·commit을 하나의 training sequence로 이어 붙이는 학습이다([[glm-4-5]] §2.3).

**메커니즘(GLM-4.5, [[glm-4-5]] §2.3, Figure 3).**

1. 먼저 4K context에서 general token 15T로 pre-training하고, 이어 code·math·science를 up-sample한 7T token으로 학습한다.
2. 32K의 repository-level code stage에서는 한 repository의 file과 model-filtered issue, PR, commit을 사용한다. commit은 diff와 비슷한 형식으로 이어 붙이며, 이 stage의 예산은 500B token이다.
3. 32K의 synthetic reasoning stage에서는 수집한 질문에 대해 reasoning model이 생성한 reasoning process를 학습한다. 예산은 500B token이다.
4. 128K의 long-context 및 agent stage에서는 long document를 up-sample한다. 보고서는 이 단계에 “Large-scale synthetic agent trajectories are also incorporated at this stage”라고 명시하며, 예산은 100B token이다.
5. reasoning이나 repository code를 truncate하지 않기 위해 mid-training에서만 best-fit packing을 쓴다. RoPE base는 32K 단계에서 10,000에서 1,000,000으로 바꾸고, learning rate는 mid-training 마지막까지 cosine schedule로 \(2.5\times10^{-4}\)에서 \(2.5\times10^{-5}\)로 감소시킨다([[glm-4-5-recipe]]).

**GLM-5의 변경점([[glm-5]] §2.3, App. A).** Stage는 32K(1T token), 128K(500B token), 200K(50B token)다. 보고서는 “Long documents and synthetic agent trajectories are up-sampled at the later stages accordingly.”라고 쓴다. 완화한 repository filtering으로 issue–PR pair 약 1,000만 개와 고유한 issue–PR token 약 160B를 얻었다. Mid-training learning rate는 \(4\times10^{-5}\)에서 \(1\times10^{-5}\)까지 선형으로 낮춘다.

**계산 예(도출한 비중).**

- GLM-4.5의 stage 합계는 \(15T+7T+0.5T+0.5T+0.1T=23.1T\)로, 보고된 23T와 일관된다. Mid-training 비중은 \(1.1/23.1=4.8\%\)다. Agent가 들어 있는 128K stage는 전체 token의 \(0.1/23.1=0.43\%\)다.
- GLM-5의 합계는 \(27T+1T+0.5T+0.05T=28.55T\)로, 보고된 28.5T와 일관된다. Mid-training 비중은 \(1.55/28.55=5.4\%\)이고, 200K stage는 \(0.05/28.55=0.18\%\)다.
- 각 stage 안에서 agent trajectory가 차지하는 비중은 보고되지 않았다. 따라서 전체 token 중 agent trajectory의 비중은 계산할 수 없다.

**일반 능력에 관한 근거.** GLM-4.5는 base-model table에 MMLU 86.1, GSM8K 79.4, MATH 61.0, EvalPlus 78.1을 제시하지만 mid-training 전후 비교는 제시하지 않는다([[glm-4-5]] §4.1 Table 2). GLM-5-Base와 GLM-4.5-Base의 비교에서는 MMLU가 86.1 → 88.3, EvalPlus가 78.1 → 87.0으로 올랐고, GSM8K는 79.4 → 68.8, MATH는 61.0 → 56.4로 내려갔다([[glm-5]] App. B.1 Table 11). 두 모델은 model size, attention, data, token 수가 모두 다르다. 그러므로 GSM8K와 MATH 하락을 mid-training에 귀속할 수 없다(Result, uncontrolled). GLM-5는 200K stage가 “further bolstered the model's performance even within the 128K context window”라고 말하지만 수치는 제공하지 않는다(Result, qualitative).

**조건과 한계.** 두 보고서 모두 agent trajectory를 ablation하지 않았고, 그 비중을 제공하지 않았으며, mid-training trajectory를 어떤 방식으로 생성했는지도 밝히지 않는다.

**함의.** GLM 보고서가 제공하는 것은 효과 크기가 아니라 배치와 예산이다. 즉 agent trajectory는 long document와 함께 가장 긴 context stage에 넣는다는 배치, 그리고 그 stage의 token budget은 확인되지만, trajectory 자체가 만든 성능 변화량은 이 보고서들만으로 알 수 없다.

## §3 Agentic continued pre-training: AgentFounder

**정의.** Agentic continued pre-training(Agentic CPT)은 공개된 base model과 post-training 사이에 합성한 agent 데이터로 수행하는 next-token prediction이다([[agentfounder]] §2.1).

**저자들이 제시한 문제.** 저자들은 agentic foundation model이 없으면 “post-training 동안 모델이 다양한 agentic behavior를 배우는 일과 expert demonstration에 정렬되는 일을 동시에 해야 하므로, 근본적인 optimization tension이 생긴다”고 말한다([[agentfounder]] Abstract). 이는 저자들의 설명이다(Interpretation). 이를 측정 가능한 질문으로 바꾸면, 원래 base와 CPT base에 같은 SFT 데이터를 적용했을 때 CPT base가 더 높은 agent score를 내는지다.

**메커니즘([[agentfounder]] §2.1–§2.3).**

1. web 데이터, tool-call 결과, Wikipedia, 폐기된 post-training trajectory에서 entity-anchored knowledge memory를 만든다.
2. First-order Action Synthesis(FAS): 여러 style의 question을 만들고, tool을 실제로 호출하지 않은 채 first-step tool call을 포함한 problem analysis와 two-step reasoning answer를 생성한 뒤 LLM judge로 filter한다.
3. High-order Action Synthesis(HAS): 기존 trajectory의 각 step마다 대안 “thought and invocation” 후보 \(N\)개를 생성한다. 원래 step과 후보들을 shuffle하고, “I will choose option \(n_k\)”를 쓴 뒤 실제 response \(R_k\)를 붙이고, 마지막은 “My decision is {Correct/Incorrect}”로 끝낸다.
4. Stage 1은 32K에서 약 200B token으로 FAS, 짧은 HAS, knowledge reasoning을 학습한다. Stage 2는 128K에서 high-quality HAS 데이터 100B token으로 학습한다.

**수식.** step이 \(K\)개이고 step당 대안이 \(N\)개인 trajectory에는 \((N+1)\times K\)개의 candidate reasoning-action이 있으며, 이 중 실제 실행된 것은 \(K\)개다. \(K\): step 수, \(N\): step당 생성한 대안 수다. 계산 예로 \(K=4\), \(N=3\)이면 candidate는 16개이고 실제 environment response는 4개이므로, 16개 중 12개 candidate에는 관측된 결과가 붙지 않는다.

**근거.**

- 같은 SFT 데이터를 쓴 Qwen3-30B-A3B-Base와 AgentFounder-30B-Base의 Pass@1(Table 3)을 비교하면, SFT-B에서 BrowseComp-en은 28.6 → 39.9, BrowseComp-zh는 35.6 → 43.3, GAIA는 71.8 → 72.8, HLE는 27.0 → 31.5다. SFT-A/B/C의 평균 향상은 각각 5.75%, 6.13%, 6.45%다(Result, 단일 연구; seed는 보고하지 않음).
- 50B-token에서 two-stage와 one-stage를 비교한 Table 4에서는 BrowseComp-en Pass@1이 31.4 → 35.5, BrowseComp-zh가 34.3 → 37.2, GAIA가 69.9 → 72.8이다.
- 약 50B-token에서 data type을 비교한 Table 5에서는 FAS+HAS가 FAS만 쓴 경우보다 BrowseComp-zh Pass@1을 37.0 → 40.1로 높이지만 GAIA는 72.8 → 69.9로 낮춘다. 저자들은 GAIA 변화를 evaluation fluctuation으로 해석한다(Interpretation).
- Data scaling(§3.5.2)에서 평균 Pass@3은 315B token에 걸쳐 54.2%에서 62.2%가 되었으며, 저자들은 “초기 15B token 안에서” 3.8점을 얻었다고 쓴다.
- 같은 SFT data로 1,340 step 학습한 뒤 SFT loss는 baseline 0.8656, 315B CPT 0.7953이다(§3.6.1). 더 낮은 SFT loss는 optimization 측정치이지 capability 측정치는 아니다.

**계산 예(수확 체감).** 처음 15B token에서는 \(3.8/15=0.25\)점/billion token이다. 남은 300B token에서는 \((8.0-3.8)/300=0.014\)점/billion token이다. 따라서 처음 15B token 뒤 token당 한계 이득은 18배 낮다(§3.5.2에서 도출).

**조건과 한계.** deep-research 외 평가로 제시된 것은 ACEBench뿐이다. released Qwen3-30B-A3B는 67.2, AgentFounder-30B는 70.0이다(Table 6). 이 두 model은 같은 post-training을 받지 않았다. knowledge, math, chat benchmark는 보고하지 않았고, CPT learning rate, batch size, replay도 보고하지 않았다.

**함의.** AgentFounder는 합성 agent 데이터로 CPT한 뒤 fixed SFT를 적용하면 deep-research score가 높아진다는 근거다. 일반 능력의 retention에 관한 근거는 아니다.

## §4 실행 trace와 Docker trajectory: CWM

**정의.** Execution trace는 실행된 각 line마다, 실행 전과 실행 후의 local variable을 기록한다([[cwm-code-world-model]] §2.2). ForagerAgent trajectory는 실제 repository의 Docker image 안에서 LLM agent가 실행한 shell command와 file edit, 그리고 environment가 반환한 response를 기록한다(§2.3).

**문제.** Code는 보통 정적인 text로 학습한다. 저자들은 “LLM에게 이러한 code world modeling capability를 가르치는 일은 대개 post-training 전에 고려되지 않는다”고 쓴다(§1). 측정 가능한 질문은 같은 SFT를 적용한 뒤 mid-training의 trace와 trajectory data가 SWE-bench Verified를 올리는가이다.

**메커니즘([[cwm-code-world-model]] §2.2–§2.3, §4.2).**

1. 실행할 수 있는 repository image를 만들고, Python function 1억 2,000만 개 이상, CodeContests solution(incorrect submission과 correct submission을 균형 있게 포함), repository commit 약 7만 개를 trace한다.
2. 각 trace는 먼저 source context를 두고, 이어 custom token (`<|frame_sep|>`, `<|line_sep|>`, `<|action_sep|>`)으로 분리한 frame들로 format한다. 각 frame은 local variable의 JSON dictionary와 실행한 line으로 이루어진다.
3. ForagerAgent를 mutate-fix task(테스트된 function에 synthetic bug를 넣은 task)와 issue-fix task(실제 issue)에 실행한다. SWE-bench repository와 그 fork는 제거하고, Jaccard < 0.5에서 MinHash deduplication한 뒤 trajectory 300만 개를 남긴다.
4. global batch 3,300만 token, context length 131K에서 5T token을 mid-train한다. mix는 CWM-specific data 30%, general code 40%, pre-training mix rehearsal 30%다.
5. 각 dataset의 share는 mid-training 종료 때 목표 epoch 1~4회에 도달하도록 정한다. 이 target은 epoching에 관한 scaling-law experiment로 선택한다.

**수식(target epoch에서 정한 share, §4.2의 서술을 식으로 쓴 것).**

\[
p_d = \frac{e_d \times U_d}{T}
\]

- \(p_d\): dataset \(d\)의 sampling share
- \(e_d\): 목표 epoch 수
- \(U_d\): \(d\)의 unique token 수
- \(T\): mid-training token budget

가상의 dataset으로 계산하면 \(U_d=120B\), \(e_d=4\), \(T=5{,}000B\)일 때 \(p_d=480/5{,}000=9.6\%\)다. \(e_d=1\)이면 share는 2.4%다.

**근거(8B, 6T pre-training + 1T mid-training ablation, RL 없는 SFT; Table 4).**

| PRs | Tracing | Forager | CruxEval-O | CruxEval-I | Agentic SBV NLL | SBV pass@1 |
|---|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 45.4 | 44.1 | 0.39 | 14.6 |
| ✓ | ✗ | ✗ | 44.6 | 45.8 | 0.37 | 18.6 |
| ✓ | ✓ | ✗ | 73.9 | 51.5 | 0.38 | 18.4 |
| ✓ | ✓ | ✓ | 74.5 | 54.8 | 0.29 | 22.1 |

Tracing data는 CruxEval-O를 29.3점 올리고 SWE-bench Verified(SBV)는 변하지 않는다. ForagerAgent data는 agentic SBV negative log-likelihood를 0.38에서 0.29로 낮추고 SBV pass@1을 3.7점 올린다(Result, 단일 연구; seed는 보고하지 않음).

**일반 능력.** Table 13은 mid-training 뒤 CWM(CWM-Mid)의 MMLU 73.6, MMLU-Pro 52.3, GPQA 31.7, GSM8k 84.7을 제시한다. 비교 대상으로 Qwen3-32B는 각각 83.6, 65.5, 49.5, 93.4다. Mid-training 전 8T checkpoint 행은 없으므로 5T mid-training stage가 이 benchmark들에 미친 효과는 측정되지 않았다. 저자들은 standard evaluation을 위해 30% rehearsal share가 필수적이라고 설명하지만 printed ablation은 제공하지 않는다(§4.2). 저자들은 CWM이 general-purpose chatbot을 목표로 한 model이 아니며 RLHF stage를 생략했다고 쓴다(§5).

**함의.** CWM은 이 장에서 mid-training mix 내부의 agent trajectory를 controlled ablation으로 검증한 유일한 보고서다. 이 ablation이 측정하는 것은 code와 SWE metric뿐이다.

## §5 Agentless 훈련을 skill prior로 사용: Kimi-Dev

**정의.** Agentless 형식은 자유로운 multi-turn agent 상호작용 대신 issue 해결을 고정된 single-turn 단계(file localization, code edit, test writing)로 나눈다. 각 단계는 “검증 가능한 reward를 갖는 single-turn 문제로 따로 최적화할 수 있다”([[kimi-dev]] §2.1). Skill prior는 나중에 agent fine-tuning을 시작할 때 쓰는 checkpoint다.

**메커니즘([[kimi-dev]] §3.2, App. A).**

1. Qwen2.5-72B-Base에서 시작한다.
2. Agentless prompt template 안에 natural diff patch 약 50B token을 만들고, prompt 부분의 loss는 mask한다. 이어 commit message 뒤에 code change를 붙인 PR commit pack 약 20B token을 만든다.
3. localization rollout 약 10B token을 만들되, “정확히 올바른 file localization을 달성한 rollout만” 남긴다.
4. 실행하지 않는 file-view 및 search tool을 써서 simulated agent interaction 약 10B token을 만든다. loss mask는 system prompt에만 적용하므로 model은 action과 observation을 학습한다. Agent가 잘못 localize한 file을 열고, 데이터에는 `I realize that I do not need to modify this file`를 삽입한다.
5. 이 synthetic 부분을 4배 upsample한다. 32K, global batch 256, learning rate (2\times10^{-5})에서 (2\times10^{-6})으로 가는 cosine schedule, warm-up 약 3B token으로 약 150B token까지 학습한다.

**계산 예(unique token과 seen token).** Unique token은 (50+20+20=90B)다. Seen token은 (50+20+20\times4=150B)로, 보고된 약 150B와 일치한다(도출). Synthetic share는 unique token 기준 (20/90=22.2\%\), seen token 기준 (80/150=53.3\%\)다. Unique-token share 22.2%를 인용해 seen-token share 53.3% 대신 쓰면 synthetic share를 (53.3/22.2=2.4)배 작게 말하게 된다.

**근거.**

- Mid-training budget 50B, 100B, 약 150B token subset을 각각 같은 2,000 pair의 light SFT 뒤에 비교하면 BugFixer pass@1이 증가한다(Figure 2; 값은 figure에만 표시됨).
- Prior 비교(§4.2, Figure 5)에서 RL prior는 Base prior가 최대로 낸 pass@1에 도달하는 데 (1.5\times2^{28}) 대신 (2^{23}) SWE-Agent SFT token이 필요해 48배 줄어든다(도출: (1.5\times2^5=48)). Mid-trained(MT) prior는 zero-shot과 one-step 설정에서 뒤처지지만, 200 trajectory((2^{24}) token) 뒤에는 SFT 및 RL prior와 “빠르게 동등해진다.”
- SFT gradient step 1회 뒤 end-to-end RL(§4.2)에서 MT prior는 6,202 problem 중 260개가 pass@8 > 0이고, SFT 및 RL prior는 2,062개다. MT pass@1은 RL step 10회 뒤 2% 아래로 떨어졌다. Figure 7은 SFT-prior와 RL-prior run만 그린다.

**조건과 한계.** 평가는 SWE-bench Verified, SWE-bench-Live, SWE-bench Multilingual만 포괄한다. MT prior 단독은 end-to-end RL의 시작점으로 가장 약했다. 따라서 이 연구에서 mid-training은 mid-training → SFT → RL prior의 한 부분이었다.

**함의.** 이 연구에서 agentic mid-training은 reasoning SFT 또는 RL을 거쳐야만 이후 stage 결과를 올렸다. End-to-end agent RL 직전에 agentic mid-training만 둔 경우에는 그렇지 않았다.

## §6 Tool-use mid-training 데이터 합성: MidTool

**정의.** MidTool-Mix는 필터링한 web, PDF, code 문서, 그 문서에 grounding한 QA와 trajectory, 실제 API와 MCP skill에서 만든 native tool-use trajectory로 구성한 20.3B-token mid-training corpus다([[midtool]] §2, Table 2).

**메커니즘([[midtool]] §2.3-§2.4, §3.1).**

1. FineWeb dump(2020-2025), 영어 FinePDFs, GitHub code, tool artifact를 수집하고, fastText classifier로 filter한 뒤 deduplicate한다.
2. Context-grounded augmentation: 문서를 tool boundary, parameter, workflow에 관한 QA와 trajectory로 바꾼다.
3. Native trajectory: tool source에 점수를 매기고 persona와 plan을 합성하며, GPT-5-family model로 생성한다. 이어 turn order, schema grounding, required argument, response consistency를 검증하고 AWM rollout과 filter한 Nemotron Agentic trace를 더한다.
4. 모든 trajectory를 “special control token 없이” 쓰는 plain chat template으로 normalize한다.
5. Qwen3-4B/8B-Base를 8,192 token에서 1 epoch, LR \(3 \times 10^{-5}\)(WSD), 4M-token batch로 mid-train한다. 그 뒤 100K TOUCAN sample으로 SFT하고, 선택적으로 64 step의 GRPO를 수행한다.

**계산 예(composition).** Source token은 \(4.4 + 2.6 + 3.8 = 10.8B\)이고, augmentation token은 \(4.1 + 2.1 + 1.5 = 7.7B\)이며, native trajectory는 \(1.8B\)다. 따라서 합계는 \(20.3B\) token이다. Native trajectory의 token 비율은 \(1.8 / 20.3 = 8.9\%\)이며, 원문에서는 9%로 표기한다.

**근거(Qwen3-4B-Base + SFT, mid-training corpus만 바꾼 비교; Table 6).**

| Mid-training data | BFCL overall | \(\tau^2\)-Bench Pass@1 | MCP-Universe score |
|---|---:|---:|---:|
| 없음 | 39.73% | 8.54% | 13.20 |
| Dolmino-20BT(matched budget, general) | 43.10% | 7.37% | 5.41 |
| processed source, trajectory 없음 | 42.30% | 7.30% | 12.20 |
| processed source + native agentic trajectory | 47.59% | 4.23% | 6.80 |
| processed source + context-grounded trajectory | 44.66% | 8.99% | 8.46 |
| MidTool-Mix(전체) | 50.25% | 12.23% | 18.66 |

Native-trajectory 행과 context-grounded 행은 각각 processed source에 synthesis branch 하나를 더한 것이다. 둘은 순서대로 누적한 단계가 아니라 서로 대안인 branch다(Table 6 caption). Processed source에 native trajectory를 더하면 mid-training 없음 대비 BFCL은 7.9점 오르지만 \(\tau^2\)-Bench Pass@1은 4.3점 내린다. Table 6의 여덟 metric 모두에서 no-mid-training 행보다 높은 것은 full mixture뿐이다(Table 6에서 도출; Result, 단일 연구; seed는 보고하지 않음). Variant 행들은 token budget을 맞추지 않았다. MidTool-Mix와 token budget을 맞춘 것은 Dolmino-20BT뿐이며, 저자들은 budget-matched variant를 future work로 둔다(App. D). 8B에서는 SFT 뒤 BFCL overall이 47.62%에서 51.12%로 변한다(Table 3).

**조건과 한계.** Table 3의 SFT 행에서 BFCL hallucination column은 높을수록 좋은 지표인데, MidTool-Mix 사용 시 4B에서 60.46%에서 56.95%로, 8B에서 65.03%에서 59.82%로 하락한다. RL 뒤에는 MidTool-Mix가 더 높다(4B: 55.19% → 60.10%; 8B: 52.79% → 64.22%). SFT 뒤 \(\tau^2\)-Bench telecom Pass@1은 4B에서 3.73%에서 1.54%로, 8B에서 4.39%에서 1.54%로 하락한다(Table 4). MCP-Universe web search는 모든 행에서 0.00으로 유지된다(Table 5). Tool-use가 아닌 benchmark는 보고하지 않았다.

**함의.** 이 ablation에서 context-grounded branch 없이 native trajectory를 processed source에 추가하면 \(\tau^2\)-Bench와 MCP-Universe score가 no-mid-training 행보다 낮아졌다. 세 benchmark 모두에서 그 행을 넘은 variant는 full mixture 하나뿐이다.

## §7 Environment-derived mid-training: State2State

**정의.** State2State는 무작위 explorer가 앞서 도달한 environment observation에 도달하는 것을 목표로 하는 RL stage다. 사람에게 지정된 task의 RL보다 먼저 실행한다([[state2state]] §3). 이 장의 다른 방법과 다른 점은 corpus에서 next-token prediction을 하는 대신 policy-gradient objective를 쓴다는 것이다.

**메커니즘([[state2state]] §3.2-§3.4).**

1. Valid action 위에서 random policy로 explore한다.
2. Invalid observation을 제거하고 다양한 target을 sample하며, 각 exploration trajectory를 3회 replay하여 reproducible target만 남긴다.
3. \(\operatorname{match}(o(s_t),o^\star)\)이면 reward \(r_t=1\), 아니면 \(0\)이다. \(o(s_t)\)는 state \(s_t\)에서의 observation, \(o^\star\)는 target observation, match는 normalized exact match다.
4. GRPO와 dynamic sampling(모든 reward가 같은 group을 버림)을 80 step 동안 최적화하고, state-reaching validation score가 가장 높은 checkpoint에서 downstream training을 계속한다(§4.3). Learning rate는 \(10^{-6}\), KL loss coefficient는 \(0.01\), batch는 16, group size는 8이다(App. E.1).

**계산 예(group advantage).** Rollout 8개로 된 group에서 2개가 성공하면 mean reward는 \(0.25\), standard deviation은 \(\sqrt{0.25\times0.75}=0.433\)이다. 성공 rollout의 advantage는 \((1-0.25)/0.433=+1.73\), 실패 rollout의 advantage는 \((0-0.25)/0.433=-0.58\)이다. 성공이 0개 또는 8개인 group은 standard deviation이 0이므로 dynamic sampling이 버린다.

**근거([[state2state]] Table 1, ID와 OOD success 평균).** Qwen3-4B에서 task RL은 ALFWorld 86.13, ScienceWorld 49.63이고, State2State + RL은 각각 92.00과 55.50이다. Qwen3-8B에서는 92.36과 52.13에 비해 97.45와 56.00이다. 단독 State2State는 Qwen3-4B의 ScienceWorld score를 23.63에서 21.00으로 낮춘다. 순서도 중요하다(Table 2, Qwen3-4B, ScienceWorld average). SFT + State2State + RL은 65.88이고 State2State + SFT + RL은 60.50이다. Cross-environment(Table 5)에서 ScienceWorld State2State 뒤 ALFWorld RL은 89.44이며, ScienceWorld task RL 뒤에는 86.87, mid-stage가 없으면 86.13이다.

**조건과 한계.** Text environment 두 개와 GUI subset만 사용했으며 general benchmark는 없다. Seed도 보고하지 않았다. 저자들은 SFT가 environment prior를 overwrite하기 때문에 순서 효과가 난다고 해석한다(Interpretation).

## §8 Skill을 pre-training 데이터로 쓰기: SPT

**정의.** Skill Pre-Training(SPT)은 공개 skill package에 causal language modeling을 적용한다. 각 package는 `SKILL.md` 지침 파일과 그것이 참조하는 reference, script, config로 이루어진다. 이 단계에는 trajectory도 environment도 없다([[skill-pretraining]] Skill Pre-Training).

**수식(Eq. 1).**

\[
p_\alpha(x)=\alpha\cdot p_{\mathrm{skill}}(x)+(1-\alpha)\cdot p_{\mathrm{gen}}(x)
\]

여기서 총 token 수 \(B\)는 고정한다. \(\alpha\)는 skill 비율이고, \(p_{\mathrm{skill}}\), \(p_{\mathrm{gen}}\)은 각각 skill block과 general block 위의 empirical distribution이며, \(B\)는 mid-training token budget이다. 가령 가상의 \(B=100M\)에서 \(\alpha=0.3\)이면 skill token은 30M, general token은 70M이다.

**근거([[skill-pretraining]] Table 1, 모두 347,770,880-token budget, 5 seeds).** 7B 모델에 xLAM-FC SFT를 붙였을 때 agentic average는 mid-training 없음 28.50, Dolmino 33.33, AgentBank trajectory 44.05, SkillCorpus 53.46이다. 같은 순서의 general average는 75.31 / 75.16 / 74.38 / 75.07이다. 3개 backbone과 2개 SFT recipe 전체에서 SkillCorpus는 direct SFT 대비 general average를 \(-0.85\)에서 \(+0.51\) 사이로 바꾼다. Instella-3B + xLAM-FC block에서 AgentBank trajectory mid-training은 MMLU 50.10, general average 70.34를 내며, Dolmino의 54.22와 72.05에 비해 각각 \(-4.12\) MMLU, \(-1.71\) general average다. 같은 block의 SkillCorpus는 MMLU 51.62, general average 71.43으로, Dolmino 대비 \(-2.60\) MMLU, \(-0.62\) general average다. 나머지 5개 backbone·SFT block에서는 AgentBank MMLU가 Dolmino와 3.0점 이내다. 가장 큰 차이는 7B + xLAM-FC의 60.57 대 57.60이다. 1.6B mixture sweep(Table 12)에서 skill 30%는 general source로 Dolmino를 쓰며 agentic 38.60을 낸다. 이는 pure skill data의 22.06보다 높고, general average는 general-only의 59.29에 비해 58.64다.

**조건과 한계.** Backbone은 1.6B-7B이고 skill repository는 하나뿐이다. Interactive long-horizon 평가는 하지 않았다(Limitations).

## §9 일반 능력에 미치는 효과: 각 보고서가 실제로 측정한 것

**정의.** 이 장에서 “측정했다”는 말은, 이후 단계들을 같게 유지한 채 agentic mid-training 단계를 넣은 모델과 넣지 않은 같은 모델을 non-agentic benchmark에서 평가했다는 뜻이다.

| 보고서 | Non-agentic 평가 | Agentic stage에 대해 통제했는가 | 보고된 효과 |
|---|---|---|---|
| GLM-4.5 | base-model 표 | 아니오 | 귀속할 수 없음 |
| GLM-5 | base 표 대 GLM-4.5-Base | 아니오(size, data, attention이 다름) | GSM8K 79.4 → 68.8, MMLU 86.1 → 88.3; 귀속할 수 없음 |
| CWM | MMLU, GPQA, GSM8k에서의 CWM-Mid | pre-mid 행이 없고 rehearsal을 필수라고 부름 | 측정하지 않음 |
| AgentFounder | 없음(ACEBench는 tool use임) | — | 측정하지 않음 |
| Kimi-Dev | 없음 | — | 측정하지 않음 |
| MidTool | 없음 | — | 측정하지 않음 |
| State2State | 없음 | — | 측정하지 않음 |
| SPT | OLMES task 6개, seed 5개 | 예, 동일 token 수 | skill은 general average가 −0.85에서 +0.51, trajectory는 한 block에서 general control 대비 MMLU −4.12 |
| Kimi K2.5 | visual RL 뒤의 MMLU-Pro, GPQA(agentic mid-training이 아님) | — | agent data에는 적용 불가 |

**상태.** Frontier scale에서 agentic mid-training이 general ability에 미치는 효과는 미해결 질문(Open question)이다. 통제된 수치는 1.6B-7B 모델에서 나온 것뿐이다([[skill-pretraining]]). Post-training 단계의 관련 reward-free 근거([[agent-early-experience]])는 agent benchmark를 보고하지만 general-capability 평가는 하지 않는다(card Verification).

## §10 긴 agent trajectory를 long-context 학습 데이터로 쓰기

**문제.** Agent trajectory에는 tool output이 계속 누적되므로, 하나의 trajectory가 pre-training context window보다 길어질 수 있다. Trajectory를 잘라 버리면 뒤쪽 step, 흔히 최종 답을 포함하는 step이 사라진다.

**보고서별 메커니즘.**

- GLM-4.5는 128K에서 agent trajectory를 추가하고, GLM-5는 128K 및 200K stage에서 이를 up-sample한다([[glm-4-5]] §2.3; [[glm-5]] §2.3).
- CWM은 mid-training token 5T 전체를 131K에서 실행한다. Mid-training document의 약 30%가 65K token을 넘기 때문이다. 긴 document 때문에 data-parallel rank가 기다리지 않도록 document를 length bucket으로 묶는다. 또한 작은 batch size로 long-context data를 학습했을 때 “lackluster performance”를 보고한다([[cwm-code-world-model]] §4.2, footnote 4). 최종 post-trained CWM의 RULER는 mid-training checkpoint의 결과가 아니라, 32K에서 84.3, 128K에서 69.7이다(App. J, Table 14).
- AgentFounder는 두 번째 stage에서 긴 HAS 데이터를 128K로 학습한다. 단일 stage 32K run에서는 일부 HAS data가 truncate된다(Table 4, §3.4.1).
- Kimi K2.5는 long text, long video, reasoning, long-CoT data를 포함하는 mid-training stage에서 context를 32,768 token에서 262,144 token으로 늘린다([[kimi-k2-5]] Table 3). Kimi K2는 YaRN으로 128K까지 늘리기 전에 4K에서 anneal(400B), 그 다음 32K에서 anneal(60B)한다([[kimi-k2]] §2.5).

**수식.** 길이 \(L\)인 trajectory와 학습 window \(W\)에 대해 window 밖으로 나가는 token 수는 다음과 같다.

\[
\max(0, L-W)
\]

Token \(D\)인 stage에 들어갈 수 있는 full-length sequence 수는 많아야 다음과 같다.

\[
\left\lfloor \frac{D}{W} \right\rfloor
\]

**계산 예.** 200,000-token trajectory를 128,000-token window에 넣으면 window 밖 token은 \(200{,}000-128{,}000=72{,}000\)개다. 이는 전체 trajectory의 \(72{,}000/200{,}000=36\%\)다. AgentFounder Stage 2는 \(D=100B\), \(W=128{,}000\)이므로 full-length sequence는 많아야 \(\lfloor100{,}000{,}000{,}000/128{,}000\rfloor=781{,}250\)개다. 이는 상한이며, 보고서는 실제 sequence count를 주지 않는다.

**근거.** 50B token에서 Stage 1 & 2와 Stage 1 only를 비교하면 BrowseComp-en Pass@1은 35.5 대 31.4이고 Pass@3는 52.0 대 49.9다([[agentfounder]] Table 4). 저자들은 비용 때문에 128K 단일-stage 학습을 평가하지 않았다고 명시한다. 따라서 이 비교는 context length 효과와 stage 순서 효과를 함께 섞는다.

**조건과 한계.** Agentic long-context stage가 일으키는 short-context regression을 측정한 보고서는 없다. 일반적인 context extension으로 생기는 short-context regression은 ch-32b에서 다룬다.

## §11 Agentic mid-training 데이터에서 모델이 예측하도록 학습되는 대상

**정의.** Loss mask (m_t)는 어떤 token이 next-token loss에 기여하는지를 선택한다.

**수식.**

\[
L=-\frac{\sum_t m_t\log p_\theta(x_t\mid x_{<t})}{\sum_t m_t}
\]

(x_t)는 (t)번째 token, (m_t\in\{0,1\}), (p_\theta)는 모델 분포다.

| 보고서 | Action(agent token) | Observation(tool/environment output) | Mask 처리 |
|---|---|---|---|
| CWM ForagerAgent | 학습함 | 학습하되 observation의 50%는 loss를 mask함 | observation 절반을 확률적으로 mask |
| Kimi-Dev simulated interaction | 학습함 | 학습함 | system prompt만 mask |
| Kimi-Dev diff patch | 학습함 | — | prompt 부분 |
| AgentFounder, MidTool, GLM | next-token prediction이라는 설명 외에는 명시하지 않음 | 명시하지 않음 | 보고하지 않음 |

**계산 예(가상 trajectory).** 한 trajectory에 agent token 2,000개와, 각각 1,500 token인 observation 4개(총 observation token 6,000개)가 있다고 하자. Mask가 없으면 observation은 학습되는 token (8{,}000)개 중 (6{,}000/8{,}000=75\%\)다. CWM처럼 observation의 50%를 mask하면 학습되는 token의 기대값은 (2{,}000+3{,}000=5{,}000)개이고, observation은 그중 (3{,}000/5{,}000=60\%\)다.

**해석.** Observation에도 학습 loss를 걸면 environment response를 예측하도록 가르친다. Kimi-Dev는 이를 policy와 world modeling을 통합하는 것이라고 부른다([[kimi-dev]] App. A). 두 보고서 모두 observation loss와 action-only loss를 비교하는 ablation은 하지 않았다.

## Negative samples와 negative feedback

용어는 ch-31a(지도 학습의 Negative Samples)와 ch-43a(Negative Samples와 Negative Gradients)가 구분한 네 의미, 즉 음의 한계가치(negative marginal value), 내용으로서의 negative, conditioning으로서의 negative, gradient로서의 negative를 따른다.

**1. 이 stage에서 negative가 나오는 곳.**

- AgentFounder FAS의 LLM-judge rejection: App. B.1은 filtering 전 correct 50% / incorrect 50%, 43.5% 제거, filtering 후 accuracy 82%를 보고한다([[agentfounder]]).
- Kimi-Dev rollout의 exact-localization filtering, MidTool native trajectory의 validation failure: MidTool은 discard 전 feedback을 주어 retry한다([[kimi-dev]] App. A; [[midtool]] §2.3).
- Test가 label한 실패 ForagerAgent trajectory 및 incorrect CodeContests submission([[cwm-code-world-model]] §2.2-§2.3).
- 정확한 observation match로 label한 state-reaching 실패 rollout([[state2state]] Eq. 1).

**계산 예(AgentFounder App. B.1에서 도출한 filter error rate).** Sample 100개를 잡으면 correct 50개, incorrect 50개다. 43.5개를 제거하면 56.5개가 남고, 그중 82%=46.3개는 correct, 10.2개는 incorrect다. Filter는 correct sample 3.7개(correct의 7.3%)와 incorrect sample 39.8개(incorrect의 79.7%)를 제거했다. 이는 82%와 50% 수치가 같은 correctness label을 사용한다고 가정한 계산이다.

**2. 현재 관행이 이를 다루는 방식.**

- **Discard(negative marginal value라고 가정):** AgentFounder FAS, Kimi-Dev localization rollout, MidTool invalid generation.
- **보통 target으로 유지:** CWM은 world-model coverage를 위해 “trajectory가 성공했는지에 따라 filter하지 않는다.” 실패 agent action도 성공 action과 동일하게 cross-entropy로 학습한다. 네 category 어느 것에도 정확히 들어맞지는 않는다. 실패를 correction하거나 label하지 않기 때문이다.
- **내용으로서의 negative:** AgentFounder HAS는 실패 trajectory 뒤에 `My decision is Incorrect`를 붙인다. Kimi-Dev는 잘못 연 file 뒤에 `I realize that I do not need to modify this file`를 넣는다. 둘 다 cross-entropy로 학습하며 failure에서 probability mass를 직접 제거하지 않는다.
- **Gradient로서의 negative:** State2State failure는 음의 GRPO advantage를 받는다.
- 대조적으로 GLM-5 SFT는 erroneous trajectory segment를 유지하되 그 loss를 mask한다([[glm-5]] §3.1). 이는 mid-training이 아니라 SFT에서의 선택이다.

**3. Gradient로서 negative의 메커니즘.** Logit z에 대한 softmax에서 \(\partial \log p_y / \partial z_j = 1[j=y]-p_j\)다. 세 action의 \(p=(0.7,0.2,0.1)\), 실패 sample \(y=3\), advantage \(A=-0.58\)이라고 하자. \(A\cdot\log p_y\)의 gradient ascent는 logit을 다음 방향으로 움직인다.

\[
A\cdot(1[j=y]-p_j)=-0.58\times(-0.7,-0.2,0.9)=(+0.41,+0.12,-0.52)
\]

1차 근사에서 probability 변화는 \(\Delta p_j\propto p_j(\Delta z_j-\sum_kp_k\Delta z_k)\)이고, \(\sum_kp_k\Delta z_k=0.26\)이다. 따라서 \(\Delta p\propto(+0.106,-0.028,-0.078)\)이다. Action 1(\(p=0.7\))이 제거된 mass 전부를 받고 action 2도 mass를 잃는다. 즉 가능성이 낮은 failure를 낮추면 확률을 현재 mode에 집중시키게 된다.

**4. 수치가 있는 근거.** 이 장의 mid-training 논문 중 failure 유지와 discard를 ablate한 것은 없다. Failure를 포함하는 CWM ForagerAgent data 전체는 SBV를 3.7점 높인다(Table 4). AgentFounder의 FAS+HAS 대 FAS(Table 5)는 Correct/Incorrect label과 HAS의 나머지 요소를 함께 섞는다. State2State는 downstream RL을 개선하지만(Table 1), positive와 negative advantage를 분리하지 않는다. 이 source 중 어느 것도 개선분의 어느 비율이 negative 때문인지 측정하지 않는다.

**5. Control.** Failure를 국소화한다(Kimi-Dev는 한 번의 잘못된 file opening 뒤 correction을 주입). Imitate하는 대신 label한다(HAS verdict text). Uncertain segment를 mask한다(GLM-5 SFT). Contrast 없이 있는 group을 discard한다(State2State dynamic sampling). General data를 mixture에 유지한다(CWM rehearsal 30%).

**6. Diagnostic.** Failure를 target으로 유지했다면 inference에서 반복 실패 action rate를 평가한다. Held-out 성공 trajectory의 agentic negative log-likelihood(CWM Table 4 column)를 기록한다. RL 기반 mid-training에서는 target 난이도별 success rate와 step별 entropy를 기록한다.

**7. 일반성에 미치는 효과.** Filter하지 않은 failure를 target으로 유지하면 실패 action의 확률을 높일 수 있지만, 이를 측정한 source는 없다(Open question). MidTool의 hallucination 관련 score는 negative ablation 없이 양 방향으로 움직인다(§6).

## 레시피

| 모델(정확한 release) | 크기 | Stage | Setting | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| GLM-4.5 | total 355B / act. 32B | mid-train | repo-level code token; sequence length | 500B; 32K | arXiv:2508.06471v1 Figure 3, §2.3 ([[glm-4-5-recipe]]) | verified 2026-09-14 | ablation 미보고 |
| GLM-4.5 | 355B / 32B act. | long-context | long document + synthetic agent trajectory | 128K에서 100B | Figure 3, §2.3 | verified 2026-09-14 | ablation 미보고 |
| GLM-4.5 | 355B / 32B act. | mid-train | RoPE base; mid-training 말미 LR | 32K에서 10,000 → 1,000,000; cosine으로 2.5e-5 | §2.4 | verified 2026-09-14 | ablation 미보고 |
| GLM-5 | 744B / 40B act. | mid-train / long-context | stage token | 32K: 1T; 128K: 500B; 200K: 50B | arXiv:2602.15763v2 §2.3 ([[glm-5]]) | verified 2026-09-15 | §2.3: 200K stage가 128K context 내 성능을 개선(수치 없음) |
| GLM-5 | 744B / 40B act. | mid-train | LR schedule | linear 4e-5 → 1e-5 | App. A | verified 2026-09-15 | ablation 미보고 |
| GLM-5 | 744B / 40B act. | mid-train | issue–PR data | ≈10M pair; ≈160B unique token; stage/epoch 미기재 | §2.3 | verified 2026-09-15 | ablation 미보고 |
| GLM-4.5 / GLM-5 | 위와 같음 | mid-train | stage별 agent-trajectory share; generator model | 미보고 | GLM-4.5 §2.3 및 Figure 3; GLM-5 §2.3 및 App. A | 미보고 | — |
| CWM | 32B | pretrain | token; context; global batch | 8T; 8,192; 8.4M token | arXiv:2510.02387v1 §4.2 ([[cwm-code-world-model]]) | verified 2026-09-15 | hyper-parameter는 scaling law에서 (§4.3) |
| CWM | 32B | mid-train | token; context; global batch | 5T; 131k; 33M token | §4.2 | verified 2026-09-15 | footnote 4: 작은 batch의 long-context 성능이 “lackluster”(수치 없음) |
| CWM | 32B | mid-train | mixture(token share) | CWM-specific 30%; general code 40%; pre-training rehearsal 30% | §4.2 | verified 2026-09-15 | rehearsal이 standard evaluation에 “proved essential”(수치 없음) |
| CWM | 32B | mid-train | dataset별 epoch | target epoch 1–4; mid-training 끝에 target에 이르게 share 설정 | §4.2 | verified 2026-09-15 | scaling-law epoching experiment(table 미출력) |
| CWM | 32B | pretrain → mid-train | LR schedule; optimizer | warmup 2,000 step; cosine peak 8×10⁻⁴, 13T horizon에서 100× decay, 마지막 5T가 mid-training; AdamW (0.9,0.95), weight decay 0.1, clip 1.0 | §4.1 | verified 2026-09-15 | scaling laws (§4.3) |
| CWM | 32B | mid-train | ForagerAgent loss | agent/environment turn 학습; observation 50% loss mask; success filter 없음 | §2.3 | verified 2026-09-15 | ablation 미보고 |
| CWM 8B ablation | 8B | mid-train | budget | pre-training 6T + mid-training 1T | §7.1 Table 4 | verified 2026-09-15 | ForagerAgent 추가 후 SFT SBV 18.4 → 22.1 |
| CWM | 32B | SFT | token; batch; context; LR; rehearsal | 100B; 2M token; 32k; warmup 1k 후 constant 1×10⁻⁵; mid-training mix에서 ≈30% | §5.1 | verified 2026-09-15 | constant LR이 cosine과 “similar evaluation metrics”(수치 없음) |
| AgentFounder-30B | total 30B(A3B) | mid-train | CPT stage 1 | 32K에서 ≈200B token | arXiv:2509.13310v1 §2.1 ([[agentfounder]]) | verified 2026-09-15 | Table 4(50B): Stage 1 & 2 > Stage 1 only |
| AgentFounder-30B | total 30B(A3B) | long-context | CPT stage 2 | 128K에서 100B token | §2.1 | verified 2026-09-15 | §3.5.2: average Pass@3 +1.8(65B vs 50B), +1.0(315B vs 210B) |
| AgentFounder-30B | 30B total(A3B) | mid-train | LR, batch, general replay | 미보고 | §2–§3, App. B | 미보고 | — |
| Kimi-Dev | 72B | mid-train | data(unique token) | ≈50B diff patch; ≈20B PR commit pack; ≈20B synthetic, 4× upsample | arXiv:2509.23045v3 §3.2, App. A ([[kimi-dev]]) | verified 2026-09-15 | Figure 2: 50B < 100B < 150B BugFixer pass@1(값은 figure) |
| Kimi-Dev | 72B | mid-train | seen token; context; batch; LR; warmup | ≈150B; 32K; 256 sequence; 2e-5 cosine→2e-6; ≈3B token | App. A | verified 2026-09-15 | ablation 미보고 |
| MidTool | Qwen3-4B-Base, Qwen3-8B-Base | mid-train | corpus; epoch; sequence length | 20.3B token(native trajectory 9%); 1; 8,192 | arXiv:2608.20314v1 Table 2, Table 10 ([[midtool]]) | verified 2026-09-15 | Table 6 corpus ablation(4B + SFT) |
| MidTool | 4B, 8B | mid-train | LR; schedule; warmup; batch; optimizer | 3×10⁻⁵; WSD; 50 step; packed 4M token; AdamW(0.9,0.999), wd 0.01 | Table 10 | verified 2026-09-15 | ablation 미보고 |
| MidTool | 4B, 8B | SFT | data; length; LR | TOUCAN 100K sample; 32,768; cosine 2×10⁻⁵ | §3.1, Table 11 | verified 2026-09-15 | ablation 미보고 |
| SPT(MeCo-1.6B-DCLM-160B) | 1.6B | mid-train | skill share α; LR; length; epoch | sweep 최선 30%; cosine 2×10⁻⁵; 4,096; 1 | arXiv:2608.26563v1 Tables 9,12 ([[skill-pretraining]]) | verified 2026-09-14 | Table 12: agentic 38.60(30%) 대 22.06(100%), 5 seed |
| State2State | Qwen3-4B, Qwen3-8B | mid-train(RL) | LR; KL loss; batch; group; step | 1e-6; 0.01; 16; 8; 80(best validation checkpoint 유지) | arXiv:2608.04934v1 App. E.1, §4.3 ([[state2state]]) | verified 2026-09-15 | Table 1: 양 size에서 State2State + RL > RL |
| Kimi K2(Kimi-K2-Base) | 1.04T / 32B act. | pretrain-decay/anneal, long-context | annealing과 activation | LR 2e-5 → 7e-6; 4k에서 400B, 이후 32k에서 60B; YaRN으로 128k | arXiv:2507.20534v2 §2.5 ([[kimi-k2-recipe]]) | verified 2026-09-14 | ablation 미보고 |
| Kimi K2.5 | 1.04T / 32B act. | long-context | joint long-context mid-training | 32,768 → 262,144 token; 500B → 200B token | arXiv:2602.02276v2 Table 3 ([[kimi-k2-5]]) | verified 2026-09-15 | ablation 미보고 |

**작은 general-purpose run의 시작점.** 4B–8B base model은 검증된 MidTool row를 따라 20.3B-token tool-use corpus를 8,192 token에서 1 epoch mid-train하고, LR 3×10⁻⁵ WSD, warmup 50 step, 4M-token batch를 사용한 뒤 32,768 token 및 LR 2×10⁻⁵로 SFT할 수 있다. 이 run은 H200 GPU 32개에서 수행됐다([[midtool]] §3.1). MidTool-Mix에는 filtered web, PDF, code source 외 별도 general replay가 없고 general benchmark도 측정하지 않았다. 1.6B backbone에는, 고정 token budget에서 skill document 30% / general data 70% split을 더한 검증 SPT row가 있으며, 1.6B에서 Tulu 3 SFT를 사용해 5 seed로 선택했다([[skill-pretraining]] Table 12). CWM의 rehearsal 30% share는 32B에서 검증됐지만 출력된 ablation은 없다.

## 일반화 관점

**(a) Breadth를 키우는 것.**

- Agentic data와 general data를 섞는 것: SPT의 skill share 20–50% mixture는 agentic score에서 pure skill data와 general-only data를 모두 이겼고, 30%에서 general average는 general-only보다 0.65점 안에 있었다(Table 12: 58.64 대 59.29)([[skill-pretraining]]).
- Mid-training과 SFT에서 pre-training mix를 rehearsal하는 것: CWM은 mid-training 30%, SFT ≈30%를 사용했다([[cwm-code-world-model]] §4.2, §5.1). 필수적이었다고 보고했지만 ablate하지 않았다(Result, qualitative).
- Heterogeneous한 agentic source: MidTool에서는 document-grounded data와 trajectory를 결합했을 때만 Table 6의 여덟 metric을 모두 개선했다([[midtool]]).
- Environment-derived objective는 한 test에서 environment 간 transfer를 보였다: ALFWorld 89.44 대 86.87([[state2state]] Table 5). Agentless-trained prior는 적은 trajectory로 SWE-bench-Live와 Multilingual에 generalize한다([[kimi-dev]] App. G.4).

**(b) Narrowing 또는 forgetting을 일으키는 것.**

- Trajectory-only mid-training은 3B에서 general-data control 대비 MMLU를 4.12점 낮췄다([[skill-pretraining]] Table 1).
- Context-grounded branch 없이 processed source와 native trajectory를 쓴 경우, no mid-training 대비 τ²-Bench Pass@1은 4.3점, MCP-Universe score는 6.4점 낮아졌다([[midtool]] Table 6).
- Mid-training 및 SFT 뒤 MidTool model은 두 size 모두 BFCL hallucination column(높을수록 좋음), 두 size 모두 τ²-Bench telecom Pass@1에서 낮은 score를 보인다([[midtool]] Tables 3–4).
- Mid-training-only prior는 나쁜 RL 시작점일 수 있다: SFT/RL prior의 2,062개에 비해 6,202 문제 중 usable한 것은 260개뿐이다([[kimi-dev]] Figure 7).
- 이후 imitation이 environment-derived prior를 지울 수 있다: State2State를 SFT 전에 놓을 때의 gain은 SFT 뒤에 놓을 때보다 작다(Table 2; 저자 Interpretation).

**(c) 이 stage에서 측정하는 법.**

1. Agentic stage 전후 checkpoint를 고정 non-agentic suite(knowledge, math, code, chat)에서 평가하고, matched-token general-data control을 추가한다(SPT 및 MidTool Dolmino 설계).
2. CWM, Kimi-Dev, MidTool, AgentFounder가 mid-training 효과를 SFT 뒤에 보고하므로, base checkpoint뿐 아니라 고정된 SFT recipe 뒤에도 평가한다.
3. Held-out tool ecosystem 및 out-of-distribution split을 포함한다(MCP-Universe; ALFWorld 및 ScienceWorld OOD).
4. Evaluation repository와 task를 decontaminate한다. CWM과 Kimi-Dev는 SWE-bench repository를 제거하고, SPT는 13-gram diagnostic을 쓰며 package를 제거하고, MidTool은 DeCon을 실행한다.
5. Seed를 보고한다. SPT는 5개를 보고했고 다른 논문은 seed count를 보고하지 않았다.
6. In-environment RL reward를 일반성 signal로 쓰지 않는다. MidTool의 RL reward curve는 converge하지만 post-RL benchmark gap은 남는다([[midtool]] App. C.2).

## Common mistakes와 탐지 방법

| 실수 | 관측 가능한 증상 | 검사 |
|---|---|---|
| 보고서의 agentic data를 잘못된 stage에 배치 | Recipe가 Kimi K2에 “agentic pre-training”을 인용 | Section을 찾는다: [[kimi-k2]] §3.1.1은 Post-Training 아래에 있음 |
| Source가 upsample했는데 unique token 기준 share를 인용 | Kimi-Dev synthetic share를 22%라고 인용 | 각 source에 upsampling factor를 곱한다. Seen share는 53.3% |
| 서로 다른 model version 사이 base-model score 변화를 agentic mid-training에 귀속 | “GLM-5 agentic mid-training이 GSM8K를 낮췄다” | Stage 전후 동일 model을 요구 |
| SFT loss를 capability measure로 사용 | 낮은 SFT loss를 더 좋은 agent ability라고 보고 | Benchmark score를 비교. MidTool App. C.1은 loss가 capability가 아니라고 명시 |
| Long trajectory를 pre-training window에서 truncate | Training sequence에 최종 답 또는 마지막 tool call이 없음 | Trajectory length histogram을 W와 비교; W 밖 token 수를 센다 |
| Document/general data 없이 trajectory만 mid-train | General/transfer benchmark 하락(SPT 3B에서 general control 대비 MMLU −4.12; MidTool native-trajectory variant에서 no mid-training 대비 τ² −4.3) | Matched-token general-data control 실행 |
| Measurement 없이 실패 trajectory를 보통 target으로 유지 | 모델이 실패 action을 반복 | Inference에서 repeated-failure rate를 측정; success filtering ablate |
| Training environment에서만 평가 | RL reward는 converge하지만 benchmark gap은 불명 | RL 뒤 held-out environment에서 평가 |

## 이해 점검

1. Kimi K2 보고서와 예전 course page가 agentic data가 training 어디에 들어가는지에 대해 다르게 말한다. 불일치를 해결할 procedure를 설명하고, 그 procedure가 다른 보고서에서도 같은 오류를 막는 이유를 설명하라.
2. CWM Table 4에서 tracing data는 CruxEval-O를 29.3점 높이지만 SWE-bench Verified는 바꾸지 않고, ForagerAgent data는 agentic NLL과 SWE-bench Verified를 모두 바꾼다. 이 pattern이 어느 data가 어느 skill을 가르친다는 것을 시사하는지, 그 설명을 시험할 추가 experiment는 무엇인지 설명하라.
3. Kimi-Dev의 mid-trained prior는 SFT trajectory 200개 뒤 SFT/RL prior에 가깝지만, SFT step 한 번 뒤 usable RL problem은 260개만 지원한다. Prior가 imitation에는 충분하고 RL에는 충분하지 않을 수 있는 이유를 설명하라.
4. Context-grounded branch가 없는 MidTool processed-source + native-trajectory variant는 BFCL을 7.9점 개선하면서 τ²-Bench를 4.3점 낮춘다. Native trajectory와 document-grounded data가 담는 내용을 바탕으로 causal explanation을 제시하고, 이를 어떻게 시험할지 말하라.
5. Equal token의 SPT Instella-3B + xLAM-FC block에서 trajectory mid-training은 general-data control 대비 MMLU를 4.12점, skill document는 2.60점 낮춘다. Trajectory data가 MMLU를 더 낮출 수 있는 두 mechanism과 이를 분리할 measurement 하나를 설명하라.
6. AgentFounder의 billion token당 gain은 15B token 뒤 18배 감소한다. 이것이 더 큰 agentic CPT budget과 더 큰 general-data share 사이의 결정을 어떻게 바꾸는지, general-purpose model에서 결정하려면 어떤 evidence가 빠졌는지 설명하라.
7. CWM은 실패 ForagerAgent trajectory를 보통 target으로 학습하는 반면 Kimi-Dev는 정확히 맞는 localization만 유지한다. 각 선택이 policy와 world model에 미칠 예상 효과와, 어느 보고서도 어느 쪽이 더 낫다고 결론내리지 못하는 이유를 설명하라.
8. State2State는 SFT 뒤에 둘 때 SFT 전보다 더 큰 gain을 보인다. Imitation learning이 RL로 배운 prior를 제거할 수 있는 이유와, 그 제거를 직접 측정하는 방법을 설명하라.

## 연결

- Dependency: ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression(long agent trajectory에는 확장된 window가 필요하고 short-context regression을 검사한다).
- Previous chapter: ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation.
- Next chapter: ch-32e — Mid-Training, Annealing, and Context-Extension Recipes Side by Side.
- Related: ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL; ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining(rehearsal share); ch-27 — Agentic Trajectory Data(trajectory format 및 여기서 바로잡은 Kimi K2 placement); ch-29c — Agentic Environment and Task Synthesis at Scale; ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares(agentic data placement의 SFT 부분); ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages; ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability; ch-45d — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL(cross-lab stage map); ch-46a — Lab: Small Agentic SFT-then-RL Run with a Generality Gate; ch-51a — Evaluating Agent Generality and Reliability.

## 출처

- [[glm-4-5]] — GLM-4.5 mid-training stage(repo code, synthetic reasoning, 128K agent trajectory) 및 base-model table.
- [[glm-4-5-recipe]] — 검증된 GLM-4.5 mid-training token, context, RoPE, LR row.
- [[glm-5]] — GLM-5의 32K/128K/200K stage, issue–PR data, mid-training LR, base-model 비교(chapter excerpt from arXiv:2602.15763v2).
- [[agentfounder]] — Agentic CPT rationale, FAS/HAS synthesis, 2-stage 32K/128K training, Tables 3–6, scaling, FAS filter statistic(chapter excerpt from arXiv:2509.13310v1).
- [[cwm-code-world-model]] — execution trace, ForagerAgent, 30/40/30 mixture, epoch target, Table 4 ablation, Table 13 general score, long-context setting(chapter excerpt from arXiv:2510.02387v1).
- [[kimi-dev]] — Agentless mid-training recipe, upsampling, loss mask, prior comparison(chapter excerpt from arXiv:2509.23045v3).
- [[midtool]] — Tool-use mid-training corpus, hyperparameter, Tables 3–6 result 및 narrowing signal(chapter excerpt from arXiv:2608.20314v1).
- [[state2state]] — Environment-derived state-reaching RL을 mid-stage로 사용; Tables 1–5(chapter excerpt from arXiv:2608.04934v1).
- [[skill-pretraining]] — 유일한 controlled general-average measurement, trajectory 대 skill 대 general mid-training, mixture equation 및 sweep.
- [[kimi-k2]] — Pre-training data, annealing, 그리고 agentic data synthesis의 post-training 위치.
- [[kimi-k2-recipe]] — 검증된 Kimi K2 annealing 및 long-context activation row.
- [[kimi-k2-agentic-data]] — 잘못된 “agentic pre-training” 주장의 검증되지 않은 source로만 인용하며 evidence로는 사용하지 않음.
- [[kimi-k2-5]] — K2.5 training-stage table, pre-training의 code 및 GUI action-trajectory data(chapter excerpt from arXiv:2602.02276v2).
- [[agent-early-experience]] — General-capability evaluation 없는 reward-free agent training 관련 자료.
