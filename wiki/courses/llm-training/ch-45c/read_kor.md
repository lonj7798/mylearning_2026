# 챕터 45c — 장기 지평 에이전트를 위한 컨텍스트 관리

> **핵심 통찰.** 장기 지평 에이전트는 context window에 담을 수 있는 양보다 많은 history token을 생성하며, 고정 checkpoint의 보고 점수는 무엇을 남길지 정하는 규칙에 따라 달라진다. BrowseComp에서 동일 checkpoint는 51.4 → 67.6(DeepSeek-V3.2, window 80%에서 Discard-all; [[deepseek-v3.1]] §4.1, §4.4), 60.6 → 74.9(Kimi K2.5, Discard-all; [[kimi-k2-5]] §5), 62.0 → 75.9(GLM-5; report는 75.9를 낸 policy를 서로 다르게 두 번 설명한다. §4.3 참조; [[glm-5]] §4.2.4, Table 7, App. B.2)로 움직인다. 이 차이는 같은 표에 있는 대부분 model release 간 차이보다 크다. 따라서 context policy를 밝히지 않은 agentic score는 다른 score와 비교할 수 없다. 둘째 결과는 policy를 hand-written으로만 둘 필요 없이 학습할 수 있다는 것이다. identical base model과 identical budget에서 RL objective 안에서 학습한 summarization은 같은 64K working context의 multi-turn GRPO 39.0%보다 held-out BrowseComp-Plus split에서 53.0%에 도달했다([[supo-summarization-rl]] Table 1). 또 32K working context와 최대 10개 branch를 쓰는 folding agent는 같은 방식으로 학습한 327K-context ReAct agent의 0.540보다 0.620 pass@1에 도달했다([[context-folding]] Table 1).
>
> **지침.** agentic benchmark를 보고할 때 context management 유무의 점수를 함께 보고하고 trigger, policy, step budget을 밝혀야 한다. 세 report가 자기 checkpoint에서 두 수치를 모두 공개했을 때 차이는 13.9~16.2점이었다([[deepseek-v3.1]] §4.4, [[kimi-k2-5]] §5, [[glm-5]] Table 7). harness가 tool result를 tool message로 보내고 model에 interleaved reasoning mode가 있다면 이전 reasoning을 history에 남겨야 한다. MiniMax는 prior-round thinking을 남긴 경우와 버린 경우에 대해 SWE-Bench Verified 69.4 대 67.2, τ² 87 대 64, BrowseComp 44.0 대 31.4, GAIA 75.7 대 67.9, xBench 72.0 대 66.0을 보고했다([[minimax-m2-interleaved-thinking]]). 반대로 harness가 tool result를 user message로 감싸면 DeepSeek는 non-thinking mode를 권한다([[deepseek-v3.1]] §3.2.1). 학습할 수 없고 하나만 바꿀 수 있다면 reasoning과 action은 보존하고 마지막 k round보다 오래된 tool observation을 fold하라. GLM-5에서 이것만으로 k = 5일 때 55.3% → 62.0%였다([[glm-5]] §4.2.4). 학습할 수 있고 task에 한 window보다 긴 history가 필요하면 inference에 compression을 덧붙이지 말고 RL objective 안에서 compression step을 학습하라. 이를 시험한 두 report에서 summarizing rollout에 plain GRPO를 적용한 것은 gain을 transfer하지 못했다([[supo-summarization-rl]] Table 1; [[resum]] Table 2). priced inference API에서 agent를 실행한다면 context를 append-only로 유지하고 prefix를 안정적으로 유지하라. token 하나의 변경도 그 위치 뒤 KV-cache를 무효화하며, Manus는 작성 시점에 cached input token과 uncached input token의 가격 차이가 10배라고 보고했다([[manus-context-engineering]]).

## 이 장이 범용 모델에 중요한 이유

이전 장은 여러 turn에 걸쳐 agent를 학습하면서 trajectory를 하나의 객체로 취급했다. observation을 mask하고, model 자신의 token에 credit을 배정하며, update를 안정적으로 유지했다(ch-45b). 이 처리는 trajectory가 context window에 들어간다는 가정을 둔다. 현재 agentic report가 겨냥하는 task family에서는 그렇지 않다. DeepSeek는 BrowseComp에서 test case의 “20%+가” 128K 한계를 넘는다고 측정했다([[deepseek-v3.1]] §4.1). folding paper는 main trajectory가 약 8K token인 반면 처리된 token은 100K를 넘고, hard subset에서 training 중 response length가 약 100K에서 160K token 이상으로 늘어난다고 보고한다([[context-folding]] §4.2–4.3).

trajectory가 window를 넘으면 무엇인가를 제거해야 하며, 제거 규칙이 policy의 일부가 된다. 이것은 범용 능력 모델에 세 가지 결과를 낳는다.

1. **모델이 학습하는 대상을 바꾼다.** reward를 낸 rollout이 summarization policy 아래에서 생성됐다면 policy gradient는 raw state가 아닌 summarized state 위에서 정의된다. 이 mismatch를 무시하는 것은 SFT의 chat-template mismatch(ch-04)와 같은 오류다. exception은 발생하지 않고 loss curve는 평범해 보이지만, 배포된 agent는 학습된 agent와 다르게 행동한다.
2. **점수가 뜻하는 바를 바꾼다.** Discard-all과 800 step으로 생성한 benchmark 수치는, context management 없이 window limit에서 멈춘 같은 benchmark 수치와 다른 측정이다. 둘은 published table에서 같은 benchmark 이름 아래에 나타난다.
3. **단순 plumbing이 아니라 capability다.** 무엇을 앞으로 운반할지 정하는 것은 transfer하는 skill이다. summarize하도록 학습한 agent는 학습했던 horizon을 넘어서도 작동한다([[supo-summarization-rl]] App. C.2 Table 3; [[memagent]] Table 1; [[mem1]] Table 1). 이 transfer가 학습 task family 밖에서도 유지되는지는 아직 열린 부분이다.

이 장은 multi-turn agentic RL(ch-45b) 뒤, claimed context length와 effective context length를 분리한 long-context evaluation 장(ch-32c) 뒤의 RL phase에 있다. 두 장을 잇는다. ch-32c가 stated size까지 window를 사용할 수 없음을 측정했다면, 이 장은 agent builder가 그 문제를 다루는 방법이다.

## §1 측정 가능한 문제

**정의.** *working context*는 한 decoding step에서 모델이 보는 token sequence다. run의 *effective context*는 run이 처리하는 total history이며, policy가 이를 reset하거나 compress하면 working context를 넘을 수 있다. [[supo-summarization-rl]] §5.1은 이를 `L_effect := L_RL × (S + 1)`로 쓴다. 여기서 `L_RL`은 RL 중 사용하는 working context length이고, `S`는 rollout에서 허용되는 summarization event의 최대 수다.

**측정 가능한 형태로 쓴 문제.** 두 효과가 겹친다.

*첫 번째 효과: window가 찬다.* system prompt가 `s` token이고 turn당 비용이 `r + a + o`(reasoning, action, observation)이면, 모든 것을 남기는 agent는 `n = ⌊(W − s) / (r + a + o)⌋` turn 뒤 window `W`에 닿는다. 수는 작지 않다. [[manus-context-engineering]]은 전형적인 Manus task에서 “around 50 tool calls”와 평균 input-to-output token ratio 약 100:1을 보고한다. [[mirothinker]]는 256K window에서 최대 600 tool call을 실행한다(§6.1).

*두 번째 효과: window가 차기 전에도 길이가 늘면 accuracy가 떨어진다.* [[chroma-context-rot]]는 task를 고정하고 input length만 18 model에 걸쳐 바꾼다. LongMemEval에서 모든 model family는 같은 question으로 만든 약 113k-token full prompt보다 약 300-token focused prompt에서 더 높은 점수를 낸다. [[context-length-alone-hurts]]도 retrieval success를 통제한 뒤 같은 결론에 도달한다. [[lost-in-the-middle]]은 이 효과의 position-dependent 형태를 제시한다. practitioner report도 같은 현상을 이름 붙인다. GLM-5는 “model accuracy degrades substantially under extremely long contexts (e.g., beyond 100k tokens)”라고 하고([[glm-5]] §4.2.4), Manus는 “window가 기술적으로 지원하더라도 특정 context length를 넘으면 model performance가 저하되는 경향”이라고 한다([[manus-context-engineering]]).

mechanism은 확정되지 않았다. [[anthropic-context-engineering]]은 해석 하나를 제시한다. 모든 token은 다른 모든 token에 attend할 수 있으므로 n개 token에는 n²개의 pairwise relationship이 생기며, training distribution에는 긴 sequence보다 짧은 sequence가 더 많아서 model이 context-wide dependency에 특화된 parameter를 덜 갖는다는 것이다. 이 결과를 “attention budget”이라 부른다. 이는 **해석**으로 취급해야 한다. [[chroma-context-rot]]는 mechanism 설명을 명시적으로 거절한다.

**함의.** window를 키워도 문제는 사라지지 않고 위치만 옮겨 간다. [[resum]] Table 3이 직접 시험했다. Tongyi-DeepResearch-30B-A3B를 context limit 128k, call budget 120으로 두어도 summarization은 BrowseComp에서 keep-everything보다 낫다(44.5 대 42.2 Pass@1). 32k에서의 차이(34.5 대 27.7)보다 작지만 0은 아니다.

원하는 구성의 token bookkeeping은 [figures/context-budget.html](figures/context-budget.html)을 열어 확인할 수 있다. 이 그림은 다섯 policy의 turn number 대비 context length를 그리고, 각각이 window가 소진되기 전 완료하는 turn 수를 보고한다.

## §2 context policy가 수행할 수 있는 연산

context는 block의 sequence다. system prompt와 task `s`, 그리고 round `i`마다 reasoning block `rᵢ`, action `aᵢ`, observation `oᵢ`가 있다. [[glm-5]] §4.2.4는 trajectory를 정확히 `(q, r₁, a₁, o₁, r₂, a₂, o₂, …, r_n, a_n, o_n)`로 쓴다. 문헌의 모든 policy는 이 block 가운데 어느 것을 살아남게 할지 선택한다.

| 연산 | 제거하는 것 | 보존하는 것 | 사용된 곳 |
|---|---|---|---|
| Keep everything (ReAct) | 없음 | 모두 | [[supo-summarization-rl]], [[resum]], [[context-folding]]의 baseline |
| Hide-Tool-Result | 가장 최근 round보다 오래된 observation | 모든 reasoning, 모든 action | [[kimi-k2-5]]의 HLE-with-tools |
| keep-recent-k | 마지막 k round보다 오래된 observation을 placeholder string으로 교체 | reasoning, action, 마지막 k개 observation | [[glm-5]](k = 5); [[mirothinker]](K = 5) |
| Recent-history truncation | 마지막 고정 token budget을 제외한 모든 것 | tail | [[resum]]의 “Recent History” baseline(가장 최근 22k token) |
| Summary / compaction | overflow한 trajectory | `s`와 생성된 summary | [[deepseek-v3.1]] §4.4; [[anthropic-context-engineering]]; [[openai-codex-max-compaction]]; [[supo-summarization-rl]]; [[resum]] |
| Discard-all | 이전의 모든 tool-call history | `s`만 | [[deepseek-v3.1]] §4.4; [[kimi-k2-5]]; [[glm-5]] |
| Branch and fold | 완료한 sub-trajectory의 내부 | main thread의 templated return message | [[context-folding]] §2.2 |
| External memory | window 밖으로 옮긴 content | window 안의 file path와 identifier | [[anthropic-context-management]]; [[manus-context-engineering]]; [[memagent]]; [[mem1]] |
| Sub-agent isolation | sub-agent의 trace | routing된 result | [[kimi-k2-5]] Agent Swarm; [[anthropic-context-engineering]] |

여기서는 서로 독립적인 두 축을 결정하며, report는 종종 둘을 혼동한다.

- **어떤 block type을 버리는가.** observation은 가장 크고 중복도 가장 높으므로 모든 heuristic policy는 여기서 시작한다. reasoning은 더 작지만 §3이 보이듯 버리는 비용이 더 크다.
- **제거가 복원 가능한가.** [[manus-context-engineering]]의 design rule은 compression을 복원 가능하게 유지하는 것이다. page의 content는 버리되 URL은 남기고, document는 버리되 path는 남긴다. 명시한 이유는 “어느 observation이 열 turn 뒤 critical해질지 신뢰성 있게 예측할 수 없기” 때문에 “어떤 irreversible compression도 위험을 수반한다”는 것이다. Discard-all은 반대 끝에 있다. run 안에서는 되돌릴 수 없다.

## §3 tool call 사이의 reasoning 보존은 template 결정이다

**정의.** *Interleaved thinking*은 모델이 reasoning block을 출력하고, tool을 호출하고, result를 받은 뒤 다음 reasoning block을 출력하되, 앞선 reasoning block이 history에 계속 남아 있는 pattern이다([[minimax-m2-interleaved-thinking]]).

**문제.** single-turn reasoning model에서 물려받은 chat-completion convention은 다음 turn 시작에서 reasoning content를 버린다. [[deepseek-v3.1]] §3.2.1은 tool loop에서의 비용을 이렇게 설명한다. “두 번째 round message가 도착할 때 reasoning content를 버리는 DeepSeek-R1 전략을 재현하면 상당한 token inefficiency가 생긴다. 이 접근은 이후의 각 tool call마다 모델이 전체 문제를 중복하여 다시 추론하게 만든다.”

**DeepSeek가 채택한 규칙**(§3.2.1, 원문에서 두 문장을 모두 인용):

- “과거 reasoning content는 새 user message가 conversation에 도입될 때만 버린다. tool 관련 message(예: tool output)만 append된다면 reasoning content는 interaction 전체에 걸쳐 보존된다.”
- “reasoning trace를 제거할 때에도 tool call history와 그 result는 context 안에 계속 보존된다.”

report가 이끌어 낸 결과가 있다. tool interaction을 user message로 흉내 내는 framework(Roo Code와 Terminus를 지목한다)는 모든 tool result에서 discard를 발동시킨다. 그래서 DeepSeek는 그 harness에서는 non-thinking mode를 권한다. 이는 error message가 없는 harness-level bug이며, ch-04의 template mismatch와 같은 계열이다.

**효과 크기의 근거.** [[minimax-m2-interleaved-thinking]]은 같은 model을 두 조건에서 보고한다.

| benchmark | 이전 thinking 보존 | 이전 thinking 폐기 | 차이 |
|---|---:|---:|---:|
| SWE-Bench Verified | 69.4 | 67.2 | +2.2 |
| τ² | 87 | 64 | +23 |
| BrowseComp | 44.0 | 31.4 | +12.6 |
| GAIA | 75.7 | 67.9 | +7.8 |
| xBench | 72.0 | 66.0 | +6.0 |

이는 **결과(단일 연구)**이며 보고 품질도 약하다. post에는 harness, run count, tool budget, table caption이 없다. MiniMax-M2에 대한 **official** 결과라는 것 이상은 아니다. 그래도 benchmark 사이의 순서는 유익하다. 가장 작은 gap은 file system이 turn 사이에 state를 운반하는 SWE-Bench Verified이고, 가장 큰 gap은 state가 conversation 안에만 사는 τ²와 BrowseComp다.

**조건과 한계.** 보존은 window와 상호작용한다. 모든 reasoning block을 남기는 일이 trajectory 자체를 길게 만든다. [[kimi-k2-5]]는 어느 block type을 희생할지 선택하여 긴장을 푼다. Hide-Tool-Result에서는 “가장 최근 round의 tool message(observation과 return value)만 보존하고, 이전 모든 step의 reasoning chain과 thinking process는 완전하게 보존한다.” [[mirothinker]] §3.3도 k-round window에서 같은 선택을 한다. 마지막 K round 밖 tool response는 아무것도 없는 것으로 교체되고 thought와 action은 절대 mask하지 않는다. evaluation에서 K = 5이며, 저자들은 ablation 없이 이것이 “performance degradation으로 이어지지 않는다”고 쓴다.

**범용 모델에 대한 함의.** reasoning block은 agent가 학습한 것을 자체적으로 요약한 결과이고 observation은 raw evidence다. budget이 선택을 강제할 때 위 네 independent report는 모두 derived text는 남기고 raw text는 버린다. 이는 design choice로서는 **반복 확인됨(Replicated)**이지만 controlled measurement로는 **반복 확인되지 않았다**. 어느 report도 반대 선택을 ablate하지 않았다.

## §4 측정된 효과가 있는 heuristic policy

### 4.1 DeepSeek-V3.2: 하나의 trigger에서 세 전략

[[deepseek-v3.1]] §4.4는 trigger와 전략을 정의한다. context management는 “token usage가 context window length의 80%를 초과할 때” 발동한다. 세 전략은 (1) overflow한 trajectory를 summarize하고 rollout을 다시 시작하는 **Summary**, (2) tool-call history의 앞 75%를 버리는 **Discard-75%**, (3) 모든 이전 tool-call history를 버려 context를 reset하는 **Discard-all**이다. report는 마지막 것이 [[anthropic-context-management]]가 설명한 기능인 “new context tool(Anthropic, 2025a)과 유사하다”고 적는다. 병렬 baseline인 Parallel-fewest-step은 독립 trajectory N개를 sample한 뒤 step 수가 가장 적은 것을 남긴다.

BrowseComp의 보고 결과는 context management 없이 51.4, Summary는 “평균 step을 364로 늘리며 최대 60.2의 performance improvement를 달성”, Discard-all은 67.6이다. 후자는 “상당히 더 적은 step을 사용하면서 parallel scaling과 비슷하다”(§4.1, §4.4, Figure 6).

report는 방법론적 결론도 직접 제시한다. “test-time compute는 context management를 통해 serial하게도, parallel하게도 scale할 수 있다 … 따라서 model performance를 benchmark할 때 실제 compute cost를 고려하는 것이 중요하다.”

### 4.2 GLM-5: 오래된 observation을 fold한 뒤 reset

[[glm-5]] §4.2.4는 reasoning과 action을 보존하고 stale observation만 fold한다.

```
o_i  ←  "Tool result is omitted to save tokens."      for i = 1, …, n − k
```

여기서 k = 5다. BrowseComp에서 keep-recent-k 없이 55.3%, 적용하면 62.0%였다. report는 다른 k 값과 round count 대신 token threshold로 발동하는 경우도 “same results로 이어진다”고 덧붙인다. 그러나 sweep은 인쇄하지 않았으므로 측정이 아니라 주장이다.

hybrid인 *Hierarchical Context Management*는 keep-recent-k를 계속 실행하면서 total context가 parameter search로 “선택한” `T = 32k`를 넘을 때 추가로 Discard-all을 수행한다. Discard-all 단독과 비교해 “all budget에서 일관된 gain, final score 75.9”를 낸다고 보고했다.

### 4.3 모델 간 표

[[glm-5]] Table 7은 여섯 모델에서 같은 benchmark를 두 번 보고한다. 문헌에서 “with”와 “without”의 controlled comparison에 가장 가까운 것이다.

| 모델 | BrowseComp | context management 적용 BrowseComp | 차이 |
|---|---:|---:|---:|
| GLM-5 | 62.0 | 75.9 | +13.9 |
| GLM-4.7 | 52.0 | 67.5 | +15.5 |
| DeepSeek-V3.2 | 51.4 | 67.6 | +16.2 |
| Kimi K2.5 | 60.6 | 74.9 | +14.3 |
| Claude Opus 4.5 | 37.0 | 57.8 | +20.8 |
| Gemini 3 Pro | 37.8 | 59.2 | +21.4 |

DeepSeek와 Kimi pair는 두 report가 스스로 제시한 수치와 일치한다([[deepseek-v3.1]] §4.4; [[kimi-k2-5]] §5). 따라서 효과는 독립 report 세 개에서 **반복 확인됨(Replicated)**이다. Claude와 Gemini 행은 model owner가 아닌 GLM-5 team이 만들었으므로 통상적인 third-party evaluation caveat가 따른다. 추가 한계도 세 가지 있다.

첫째, “without” 조건은 동일하지 않다. GLM-5는 context management 없이 가장 최근 5 turn의 detail을 보존한다고 밝힌다([[glm-5]] App. B.2). 반면 [[kimi-k2-5]]는 context overflow를 outright failure로 세고, [[deepseek-v3.1]] §4.1은 “test case의 20%+가 이 limit을 넘는” 128K limit에서의 score 51.4를 보고한다.

둘째, report는 자체 75.9를 무엇이 만들었는지 내부적으로 일관되지 않다. §4.2.4는 그 점수를 keep-recent-k와 `T = 32k` Discard-all을 합친 Hierarchical Context Management에 돌린다. 하지만 App. B.2는 with-context-management column을 “DeepSeek-V3.2 및 Kimi K2.5와 같은 discard-all strategy”라고 설명한다. 두 설명이 모두 완전할 수 없으며 report는 GLM-5의 Discard-all-only number를 인쇄하지 않아 둘을 분리할 수 없다. GLM-5의 13.9점 gap 자체는 측정된 것으로, 그것을 keep-recent-k와 Discard-all에 나누는 것은 **보고되지 않음**으로 취급한다.

셋째, step budget은 구조상 두 조건에서 다르다. context management는 더 많은 step을 허용하려고 존재한다. 그러므로 표의 모든 gain 일부는 같은 compute를 더 잘 쓴 효과가 아니라 추가 serial test-time compute다.

### 4.4 context policy로서의 sub-agent

[[kimi-k2-5]]는 multi-agent orchestration을 별도 topic이 아니라 context-management architecture로 다룬다. sub-agent는 “central orchestrator의 global context를 직접 mutate하거나 contaminate하지 않고 independent working memory를 유지하며 local reasoning을 수행한다. full interaction trace가 아니라 task-relevant output만 선택적으로 다시 route된다.” 보고 수치는 Agent Swarm BrowseComp 78.4, single agent 60.6이며, WideSearch에서 target Item-F1에 도달하는 execution time은 3×–4.5× 줄었다. report는 parallel compute 효과와 context isolation 효과를 분리하지 않으므로 17.8점 gain을 context management 단독에 돌릴 수 없다. [[anthropic-context-engineering]]는 return channel의 크기까지 포함해 같은 pattern을 설명한다. sub-agent는 수만 token을 쓸 수 있지만 “often 1,000-2,000 tokens”만 반환할 수 있다.

## §5 학습된 context management

위 heuristic은 그 환경에서 한 번도 학습하지 않은 모델에 inference 때 적용된다. 여기의 네 방법은 compression step 자체를 학습한다. 네 방법 모두 segment로 잘린 trajectory 끝의 reward 하나라는 같은 구조 문제를 가지며, 절단을 가로질러 credit을 배정하는 방식이 다르다.

### 5.1 SUPO: MDP 내부의 summarization

**메커니즘**([[supo-summarization-rl]] §3–§4, Algorithm 2)은 단계별로 다음과 같다.

1. `L_t = |(s_t, a_t, o_t)| < L`인 동안 보통의 tool loop를 실행한다. `L`은 summarization threshold다.
2. 길이가 `L`을 넘고 summary를 `S`개보다 적게 발행했으면 `s_{t+1} := (s_t, v_sum)`으로 둔다. 즉 summarization instruction을 append하고 **마지막 action-observation pair를 버린다**.
3. 모델은 summary `a_t`를 출력하고 다음 state는 원 task prompt와 summary인 `s_{t+1} := (s₁, a_t)`가 된다. summary count를 하나 증가한다.
4. `H` step 또는 `S` summary에서 멈춘다.

2단계의 discard는 우연이 아니라 길이 제어 장치다. 없으면 도달 가능한 context는 `L + 2L_A + L_O + |v_sum|`이다(`L_A`는 action 길이, `L_O`는 observation 길이). 있으면 상한은 `L + |v_sum| + L_A`이며, 이는 “summarization 전 긴 observation 때문에 trajectory 끝 summary가 RL training context length에 의해 clip되지 않도록 보장한다”(§4.2).

**Gradient.** Theorem 3.2는 long rollout의 policy gradient를 `I^j + 1`개 summarized sub-trajectory에 대한 합으로 분해한다. 따라서 각 segment는 수정하지 않은 trainer에 ordinary trajectory로 전달할 수 있다. segment는 update에 영향을 주지 않는 zero-mask dummy trajectory로 `⌈N / B_mini⌉ × B_mini`까지 padding한다.

**Advantage.** rollout `j`의 모든 segment는 **rollout** group 안에서 계산한 advantage 하나를 공유한다(Eq. 3). ablation 대안(Eq. 4)은 **trajectory** group 안에서 normalize하여 rollout reward를 segment마다 한 번씩 반복한다. 더 자주 summarize하는 rollout은 segment를 더 많이 내므로, trajectory-group version은 group statistic에서 긴 rollout에 더 큰 weight를 준다.

**결과**(held-out 128개와 100개 instance set; [[supo-summarization-rl]] Table 1):

| 환경 | 알고리즘 | working context | effective context | 이후 accuracy | tool call |
|---|---|---|---|---:|---:|
| CodeGym(Qwen2.5-32B-Instruct) | GRPO | 32K | 32K | 44.5% | 52.1 |
| CodeGym | SUPO | 4K | 32K(4K×8) | 47.7% | 54.7 |
| BrowseComp-Plus(Seed-OSS-36B-Instruct) | GRPO | 64K | 64K | 39.0% | 6.7 |
| BrowseComp-Plus | SUPO | 64K | 192K(64K×3) | 53.0% | 19.2 |

CodeGym 행이 더 날카로운 비교다. working context가 **4K**인 agent가 같은 effective budget에서 32K GRPO agent와 같거나 더 높다. **결과(단일 연구)**이며 seed 하나, environment 둘이다.

**외삽.** 64K/192K로 학습한 checkpoint는 학습 때보다 큰 summary budget에서 평가해도 계속 개선된다. 64K×1에서 35.0%, 64K×3(학습 설정)에서 53.0%, 64K×6에서 56.0%, 64K×12에서 59.0%, 64K×24에서 60.0%였다(App. C.2 Table 3). 같은 sweep에서 base model은 28.0 → 37.0, GRPO model은 39.0 → 50.0이므로 학습된 summarizer는 출발점도 높고 slope도 유지한다.

### 5.2 ReSum: segment 전체로 advantage broadcast

[[resum]]은 agent를 바꾸지 않고 history가 limit에 가까워지면 external summary model을 부른다. 따라서 retraining 없이 released checkpoint에도 적용할 수 있다. RL variant ReSum-GRPO는 credit assignment를 다음처럼 broadcast한다.

```
Â_g   = (R_g − mean({R_1, …, R_G})) / std({R_1, …, R_G}),   R_g ∈ {0, 1}
Â_g^(i) = Â_g        for every segment i = 1, …, n_g of rollout g
```

`R_g`는 마지막 segment에서 뽑은 final answer의 binary correctness, `G`는 group size, `n_g`는 rollout `g`가 잘린 segment 수다. 명시한 목적은 “final success에 기여한 early exploration step이 적절한 bonus를 받도록 보장”하는 것이며, summary하지 않는 짧은 trajectory는 standard GRPO와 동일하게 처리한다.

1K training sample, 4 epoch 결과(Pass@1, [[resum]] Tables 1–2)에서 WebSailor-3B는 BrowseComp-zh에서 ReSum-GRPO로 8.2% → 20.5%였고 ReAct paradigm의 GRPO는 11.8%에 그쳤다. WebSailor-30B는 BrowseComp-zh에서 33.3%에 도달했다. paper는 두 설정을 “training-free setting에서 ReAct보다 4.5% improvement, ReSum-GRPO가 추가 8.2% gain”이라고 요약한다. ReAct에 GRPO를 적용한 뒤 ReSum 아래에서 평가하면 transfer하지 않는다. “GRPO fails to enable agents to master summary-conditioned reasoning.” 학습한 30B summary model ReSumTool-30B는 같은 backbone에서 summarizer로 쓴 Qwen3-235B와 DeepSeek-R1-671B에 맞먹거나 이긴다(WebSailor-3B BrowseComp-zh에서 13.7% 대 11.1%, 13.0%). 이는 summarization 역할이 general capability test가 아니라 좁은 skill임을 뜻한다.

### 5.3 Context folding: 경계가 있는 main thread

[[context-folding]]은 flat summary 대신 tool 두 개를 쓴다. `branch(description, prompt)`는 sub-task용 sub-context를 열고, `return(message)`는 이를 fold한다. 내부 action-observation pair를 없애고 templated message만 main thread에 남긴다. inference에서 KV-cache는 prefix가 pre-branch context와 같은 branch position으로 rollback하므로 fold에는 re-prefill 비용이 없다.

FoldGRPO는 outcome reward만 썼을 때 관찰한 두 failure mode(main thread에 token-heavy work를 접지 않고 남김, branch가 돌아오지 않음)를 겨냥하여 advantage 안 outcome reward에 token-level process reward를 더한다. main thread가 working-context limit의 50%를 넘으면 branch-creating turn token을 제외한 main-thread token에 `Q = −1`, GPT-5-nano가 자기 sub-task 밖에서 행동했다고 판단한 branch의 모든 token에 `Q = −0.2`, 실패한 tool-call turn token에 `Q = −1`, 나머지에는 `Q = 0`이다(§2.3.2).

32K working context, 최대 10 branch에서 BrowseComp-Plus / SWE-Bench Verified의 pass@1([[context-folding]] Table 1)은 untrained folding agent 0.420 / 0.492, `+ GRPO` 0.567 / 0.564, `+ FoldGRPO` 0.620 / 0.580이다. 이 장에서 중요한 context-size 비교는 같은 data로 GRPO 학습한 327K-context ReAct agent 0.540 / 0.574, GRPO 학습한 32K × 10 summary agent 0.527 / 0.550이다. behaviour statistic(Table 2)은 이유를 보인다. FoldGRPO는 plain GRPO의 finish rate 0.738 / 0.612를 0.935 / 0.962로 올리고 main-thread length는 22,285 / 48,908 token에서 7,752 / 8,885로 줄인다.

### 5.4 학습된 skill로서의 memory

두 paper는 trajectory를 완전히 제거하고 fixed-size state를 유지한다.

[[memagent]]는 document를 chunk로 model에 stream하고 매 chunk 뒤 fixed-length memory를 **overwrite**한다. 따라서 “chunk당 total compute는 O(1)을 유지하며 end-to-end complexity는 chunk 수에 엄격히 linear하다.” training은 60K-token document에서 8K window를 1024 query / 5000 chunk / 1024 memory / 1024 output으로 나누고 DAPO의 multi-conversation extension을 쓴다. RULER-HQA에서 7B model은 7K에서 81.25, 3.5M token에서 71.88인 반면 Qwen2.5-Instruct-7B-1M은 61.72 → 0.00(896K)이다(Table 1).

[[mem1]]은 같은 생각을 tool use에 적용한다. 매 turn에서 이전 internal state, query, observation을 새 internal state로 consolidate하고 “previous turn의 모든 tag를 prune”한다. context에는 최대 두 `<IS>`, 두 `<query>`, 하나 `<info>`만 남는다. 2-objective task만으로 학습했지만 16-objective task에서 Qwen2.5-14B-Instruct의 EM 0.567 대비 EM 1.97에 도달했다. peak token은 `10.4 × 10²` 대 `38.4 × 10²`로, “peak token의 27.1%, total inference time의 29.3%”다(§4.2, Table 1). paper는 이 행동 획득에서 “SFT significantly underperforms RL”도 보고한다.

비용 측면은 third party가 보고한다. [[resum]] App. F.3은 매 turn consolidate하는 것이 threshold에서 summarize하는 것보다 generation이 많기 때문에 BrowseComp에서 MEM1-GRPO가 “mere 1.2% improvement를 위해 ReSum보다 거의 3× 더 많은 token을 소비한다”고 찾았다.

### 5.5 네 방법의 공통점

| 방법 | 압축하는 것 | trigger | 절단을 가로지르는 credit assignment | failure 제어 |
|---|---|---|---|---|
| SUPO | 전체 history → summary | context ≥ working length의 95% | 모든 segment가 하나의 rollout-group advantage를 공유 | answer를 내지 않은 rollout의 overlong mask |
| ReSum | 전체 history → external tool의 summary | history가 limit에 접근 | trajectory advantage를 모든 segment에 broadcast | 보고되지 않음; 짧은 rollout은 변경 없음 |
| FoldGRPO | sub-trajectory → return message | agent가 발행한 `branch` / `return` | advantage 안에 token-level process reward 추가 | `clip(R_i + Q, 0, 1)`은 실패 rollout에 penalty를 주지 않음 |
| MEM1 / MemAgent | 모든 것 → fixed-size state | 매 turn / 매 chunk | conversation당 ordinary group advantage | 보고되지 않음 |

공통 pattern은 binary reward 하나, 여러 segment, 그리고 segment token이 전체 rollout의 fate를 물려받을지 결정하는 문제다. 다음 절이 적용되는 곳이 이 결정이다.

## §6 계산 예시: 들어가는 turn 수와 reset 하나의 비용

`W = 128,000`, `s = 1,500`, `r = 400`, `a = 80`, `o = 1,500`으로 둔다. full turn 하나의 비용은 `r + a + o = 1,980` token이다. 아래 수치는 모두 손으로 검산할 수 있다.

**모두 보존(Keep everything).** `n = ⌊(128,000 − 1,500) / 1,980⌋ = ⌊63.88⌋ = 63` turn 뒤 run이 멈춘다. 처리한 total history는 `63 × 1,980 = 124,740` token이다.

**k = 5인 keep-recent-k.** `n` turn 뒤 context는 `1,500 + n(400 + 80) + 5 × 1,500 + (n − 5) × 12`이다. placeholder string에 12 token을 쓴다. 이를 128,000과 같게 두면 `1,500 + 480n + 7,500 + 12n − 60 = 128,000`, 따라서 `492n = 119,060`, `n = 241`이다. agent는 63 대신 241 turn을 완료한다. 같은 window에서 horizon은 3.8× 길다. agent가 읽은 observation은 `241 × 1,980 = 477,180` token이고, 그중 `236 × 1,500 = 354,000`은 더 이상 보이지 않는다. 이 산술이 k-round 규칙이 가능한 intervention 중 가장 싼 이유다. harness code 한 줄만 들고 model 변경이 필요 없다.

**80%에서 Discard-all.** trigger는 `0.8 × 128,000 = 102,400` token이다. `s = 1,500`에서 시작하면 context는 `⌈(102,400 − 1,500) / 1,980⌉ = ⌈50.96⌉ = 51` turn 뒤 trigger를 넘고, 1,500으로 reset한다. 이후 cycle도 각각 51 turn이다. DeepSeek Summary 전략의 평균인 364 step([[deepseek-v3.1]] §4.4)에 도달하려면 약 `364 / 51 ≈ 7`번 reset한다. cycle 1의 어떤 것도 cycle 7에서는 보이지 않는다. 모델이 answer나 external file에 다시 쓴 것만 예외다.

**800-token summary를 붙인 80% Summary.** cycle length는 같지만 각 cycle은 `1,500 + 800 = 2,300` token에서 다시 시작한다. 따라서 첫 cycle 뒤에는 `⌈(102,400 − 2,300) / 1,980⌉ = 51` turn도 들어간다. reset 하나는 800 token 추가 generation과 summarization prompt의 prefill을 요구한다. 7번 reset이면 추가 생성 token은 5,600이다. history에 비해 작다. 이것이 [[deepseek-v3.1]]가 Summary를 효과적이지만 효율은 “relatively low”라고 부른 이유다. 비용은 summary token 자체가 아니라 summarization call이 step 하나를 쓰고 새 prefix를 re-prefill한다는 데 있다.

**산술의 읽는 법.** §2 표의 모든 policy는 information을 내주고 turn을 산다. keep-recent-k는 오래된 observation을 포기하고 그것을 참조한 reasoning은 남긴다. Discard-all은 reasoning까지 포함해 모두 포기한다. Summary는 detail을 포기하고 model이 고른 trace를 남긴다. [figures/context-budget.html](figures/context-budget.html)에서 `W`, `o`, `k`, trigger를 바꾸며 세 curve가 교차하는 지점을 볼 수 있다.

## §7 비용, KV-cache, 그리고 학습된 행동으로서의 compaction

agentic inference는 prefill-dominated다. [[manus-context-engineering]]은 production에서 평균 input-to-output token ratio가 약 100:1이라고 보고한다. 따라서 step당 비용은 prefix 중 얼마나 많은 부분을 cache에서 제공할 수 있는지가 정한다. 각각 명시한 mechanism을 가진 세 operational rule이 뒤따른다.

1. **안정적인 prefix.** token 하나만 바뀌어도 그 위치 뒤 cache가 무효화된다. system prompt 맨 위의 timestamp가 빈번한 원인으로 지목된다.
2. **deterministic serialization을 갖춘 append-only context.** 안정적이지 않은 JSON key ordering은 error 없이 cache를 깨뜨린다.
3. **tool은 제거하지 말고 mask한다.** tool definition을 지우면 그 뒤 모든 것의 cache가 무효화된다. 또한 history의 이전 action이 더 이상 존재하지 않는 tool을 참조하게 되어 “종종 schema violation 또는 hallucinated action으로 이어진다.” Manus는 대신 shared tool-name prefix(`browser_`, `shell_`)까지 response를 prefill하여 decoding을 제한한다.

규칙 2는 context를 rewrite하는 §2의 모든 policy와 직접 충돌한다. 이 긴장은 rewrite가 *어디서* 일어나느냐로 해소된다. [[context-folding]] §2.2는 KV-cache를 branch position으로 rollback한다. 그 위치에서 prefix는 pre-branch context와 byte-identical하므로 fold는 re-prefill 비용이 없다. 반대로 Discard-all은 history와 함께 cache도 버린다. 이것은 continuous rewriting보다 소수의 잘 정의된 위치에서 reset을 선호할 이유다.

compaction도 이제 harness feature가 아니라 학습된 행동이다. [[openai-codex-max-compaction]]은 GPT-5.1-Codex-Max를 “*compaction*이라는 process를 통해 multiple context window에 걸쳐 작동하도록 native하게 학습한 첫 model”이라고 설명한다. model은 “context window limit에 접근하면 session을 자동 compact하여 fresh context window를 부여받고, task가 완료될 때까지 이를 반복한다.” 내부 관찰에서는 task가 “24시간 이상” 실행됐다. post는 SWE-bench Verified에서 medium reasoning effort의 GPT-5.1-Codex보다 thinking token이 30% 적었다고 보고한다. appendix table은 GPT-5.1-Codex(high)와 GPT-5.1-Codex-Max(xhigh)를 비교하며 SWE-bench Verified 73.7% → 77.9%, SWE-Lancer IC SWE 66.3% → 79.9%, Terminal-Bench 2.0 52.8% → 58.1%를 보인다. 두 column은 model과 reasoning effort가 모두 다르므로 compaction만 분리하지 못한다. compaction 주장은 **official이지만 ablation되지 않음**으로 취급한다.

platform 측 수치도 같은 형태다. [[anthropic-context-management]]은 internal agentic-search evaluation에서 context editing과 memory tool을 결합하면 “39%” improvement, context editing만 쓰면 “29%” improvement를 보고한다. 100-turn web-search evaluation에서는 그렇지 않으면 실패할 workflow를 완료하면서 token consumption이 84% 줄었다고 한다. evaluation set, baseline, metric, run count는 공개하지 않았고, post는 39%와 29%가 relative인지 absolute인지도 말하지 않는다. 방향이 양수라는 근거로는 쓸 수 있지만 크기의 근거로는 쓸 수 없다.

## 음성 샘플과 음성 피드백

context management는 authoring standard §6.1의 서로 다른 세 가지 의미로 음성을 만든다. 이들을 혼동하는 일이 이 paper들이 보고한 instability 대부분의 원인이다.

**음성의 출처.** long-horizon rollout은 세 방식으로 실패한다. judge 또는 unit test가 검증한 잘못된 final answer, answer 없이 `H` step 또는 `S` summary를 모두 소진한 run, malformed tool call을 낸 turn이다. 첫째는 environment가, 둘째는 harness가, 셋째는 syntax가 label한다. 이 장의 어느 source도 false-negative rate를 보고하지 않았다.

**현재 관행이 각각에 하는 일.**

1. **gradient로서의 음성(sense 4).** 잘못된 final answer는 `R_g = 0`을 내며 GRPO에서 group mean 아래 advantage가 된다([[grpo]]). [[resum]]과 [[supo-summarization-rl]]에서는 summarization token을 포함한 *rollout의 모든 segment*에 적용된다. 이것이 segmented training의 고유한 위험이다. rollout은 summary와 무관한 이유로 실패할 수 있는데 broadcast는 그 summary token의 likelihood도 낮춘다. softmax logit gradient `∂ log p_y / ∂ z_j = 1[j = y] − p_j`는 빠진 mass의 행선지를 말한다. 현재 probability에 비례해 재분배되므로 유능한 summary를 아래로 누르면 long compressed history에서 흔히 더 짧고 덜 informative한 summary인, 모델이 원래 쓸 문장에 mass가 몰린다.
2. **국소화된 gradient 음성.** [[context-folding]] §2.3.2는 trajectory-level penalty 대신 token-level penalty를 붙인다. branch-creating turn을 제외하고 50% threshold를 넘은 unfolded main-thread token에는 `Q = −1`, scope 밖 branch token에는 `−0.2`, 실패 tool-call turn에는 `−1`이다. advantage는 `Â_{i,t} = (clip(R_i + Q_{i,t}, 0, 1) − mean({R_i})) / std({R_i})`다. clip을 읽으면 `R_i = 0`일 때 음의 `Q`는 합을 0 아래로 내리지 못한다. 따라서 penalty는 **성공 trajectory 안에서만** advantage를 바꾼다. design은 실패 rollout을 의도적으로 두 번 벌주지 않는다.
3. **버린 sample로서의 음성(sense 1).** [[supo-summarization-rl]] §4.2는 answer 없이 `H` 또는 `S`에 도달한 rollout을 mask한다. “mask가 없으면 objective는 step 또는 trajectory limit 안에 answer를 내지 못했어도 좋은 summarization strategy를 보인 long rollout을 억제하는 쪽으로 bias될 수 있다. 이는 본질적으로 long-horizon task에서 summarization pattern의 collapse로 더 이어질 수 있다.” ablation이 근거다. mask를 빼면 BrowseComp-Plus는 53.0% → 44.0%, CodeGym은 47.7% → 45.3%가 된다(Table 1). 저자들은 unmasked BrowseComp-Plus run이 “degenerated performance” 때문에 5 epoch가 아닌 3 epoch에 멈췄다고 적는다. [[mirothinker]]도 data level에서 trivial format error나 action loop로 실패한 incorrect trajectory를 벌주기보다 버린다.
4. **content로서의 음성(sense 2).** [[manus-context-engineering]]은 inference에서 반대 방식을 주장한다. “wrong turn을 context에 남겨라. model은 failed action과 그 observation 또는 stack trace를 보면 내부 belief를 암묵적으로 갱신한다.” 측정이 붙지 않았으므로 이는 **일화적 근거**이고 quantitative claim을 뒷받침할 수 없다. 그러나 stack trace도 observation이므로 aggressive observation dropping과 직접 충돌한다. 오래된 observation을 모두 fold하는 policy는 이 권고가 의존하는 증거를 정확히 제거한다.

**여기서 음성을 안전하게 만드는 제어.** non-terminating rollout은 zero score를 주지 말고 mask한다([[supo-summarization-rl]]). process penalty는 outcome failure와 합산되지 않게 clip한다([[context-folding]]). advantage normalization은 segment group 대신 rollout group 안에서 수행하여 segment가 더 많은 rollout이 group statistic을 지배하지 않게 한다([[supo-summarization-rl]] Eq. 3 대 Eq. 4: CodeGym 47.7% 대 42.1%, BrowseComp-Plus 53.0% 대 49.0%). 그리고 tool-observation token은 loss에서 mask한다. 이는 ch-45b 규칙이며 [[context-folding]] §2.3.1에도 다시 적혀 있다.

**진단.** summarization rate `p_summary = #rollouts with a summary / #rollouts`와 conditional success rate `p_success|summary = #successful rollouts with a summary / #rollouts with a summary`를 따로 log한다. [[supo-summarization-rl]] §5.2.2는 둘 다 추적하고, 후자가 training 내내 상승한다고 보고한다. 이는 agent가 summarization을 피하는 것이 아니라 summarization skill 자체가 좋아진 신호다. [[context-folding]] Table 2의 finish rate와 mean main-thread length, ch-45b의 advantage-sign별 token count도 추가한다.

**일반성에 미치는 효과.** 측정된 위험은 [[supo-summarization-rl]] §4.2가 설명한 collapse다. sign이 잘못된 penalty는 행동을 개선하는 대신 행동 자체를 없애며, observable은 working context가 saturate하면서 summarization rate가 0으로 가는 것이다. 이 장 source들은 large k의 pass@k나 context-managed agent의 general-capability panel을 보고하지 않는다. 따라서 ch-43의 coverage question(entropy, output diversity, RL 아래 pass@k)은 이 단계에서 **열림(Open)**이다.

## 레시피

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3.2 | 671B total / 37B activated (V3.1 model card) | eval-gate | context-management trigger | token usage > 80% of context window length | arXiv:2512.02556 §4.4 | verified 2026-09-15 | Fig. 6 compares Summary, Discard-75%, Discard-all, Parallel-fewest-step on BrowseComp |
| DeepSeek-V3.2 | same | eval-gate | BrowseComp without / with Discard-all | 51.4 / 67.6 | §4.1, §4.4 | verified 2026-09-15 | Fig. 6 (accuracy vs real steps) |
| DeepSeek-V3.2 | same | eval-gate | Summary strategy: average steps, best score | 364 steps; up to 60.2 | §4.4 | verified 2026-09-15 | Fig. 6 |
| DeepSeek-V3.2 | same | SFT/RL template | reasoning retention in tool use | reasoning discarded only on a new user message; tool-call history always kept | §3.2.1, Fig. 4 | verified 2026-09-15 | no ablation reported |
| GLM-5 | not reported in the cited sections | eval-gate | keep-recent-k | k = 5; observations older than the last k rounds replaced by a placeholder string | arXiv:2602.15763v2 §4.2.4 | verified 2026-09-15 | BrowseComp 55.3% → 62.0% |
| GLM-5 | same | eval-gate | Hierarchical Context Management threshold T | T = 32k, selected by parameter search | §4.2.4 | verified 2026-09-15 | final BrowseComp 75.9 vs Discard-all alone (Fig. 8) |
| GLM-5 | same | eval-gate | BrowseComp judge | official OpenAI evaluation prompt, o3-mini as judge | §4.2.4 | verified 2026-09-15 | case studies showing sensitivity to judge prompt and model |
| Kimi K2.5 | not reported in the cited sections | eval-gate | HLE-with-tools policy | Hide-Tool-Result: keep only the most recent round of tool messages; keep all reasoning | App. "Context Management Strategies" | verified 2026-09-15 | no ablation reported |
| Kimi K2.5 | same | eval-gate | BrowseComp without / with Discard-all | 60.6 / 74.9 | §5 prose (the 60.6 also appears in Table 6; 74.9 is not in a table) | verified 2026-09-15 | Table 6 against single-agent and proprietary baselines |
| MiroThinker-v1.0 | 8B / 30B / 72B | eval-gate | recency retention K | K = 5 most recent tool responses; thoughts and actions always kept | arXiv:2511.11793 v3 §3.3, §6.1 | verified 2026-09-14 (card) | authors state no degradation; no ablation table |
| SUPO on Qwen2.5-32B-Instruct (CodeGym) | 32B | RL | working context / max summaries / effective | 4K / S = 7 / 32K | arXiv:2510.06727v1 §5.1 | verified 2026-09-15 | Table 1: 47.7% vs GRPO 44.5% at equal effective context |
| SUPO on Seed-OSS-36B-Instruct (BrowseComp-Plus) | 36B | RL | working context / max summaries / effective | 64K / S = 2 / 192K | §5.1 | verified 2026-09-15 | Table 1: 53.0% vs GRPO 39.0% |
| SUPO (both) | 32B / 36B | RL | summarization threshold L; max steps H | L = 95% of working context length; H = 100 | §5.1 | verified 2026-09-15 | no ablation of L or H reported |
| SUPO (both) | 32B / 36B | RL | group size; clip; LR; KL and entropy | G = 8; ε_low 0.20, ε_high 0.28; 1 × 10⁻⁶ constant; no KL term, no entropy term | §5.1 | verified 2026-09-15 | no ablation reported |
| ReSum-GRPO on WebSailor-3B/-7B/-30B | 3B / 7B / 30B | RL | training samples; epochs; context; tool budget | 1K samples; 4 epochs; 32k window; 60 tool calls | arXiv:2509.13313v3 §4.1, §4.3 | verified 2026-09-15 | Table 2 against GRPO and MEM1-GRPO |
| ReSum (inference) | any WebSailor backbone | eval-gate | summary tool and trigger | ReSumTool-30B, invoked as history approaches the context limit (4k prompt + 28k responses) | §4.1, App. D | verified 2026-09-15 | Table 1: matches or beats Qwen3-235B and DeepSeek-R1-671B as summarizers |
| Folding agent on Seed-OSS-36B-Instruct + FoldGRPO | 36B | RL | working context; branches | 32,768 tokens; up to 10 branches (327,680 theoretical) | arXiv:2510.11967v1 §3.2 | verified 2026-09-14 (card) | Fig. 5: plateau beyond 320K |
| same | 36B | RL | process penalties | −1 unfolded main-thread tokens above 50% of the limit; −0.2 out-of-scope branch; −1 failed tool call | §2.3.2 | verified 2026-09-14 (card) | Tables 1–2: 0.620/0.580 vs GRPO 0.567/0.564 with finish rate 0.935/0.962 vs 0.738/0.612 |
| RL-MemAgent-7B / -14B | 7B / 14B | RL | training window allocation | 8K total: 1024 query, 5000 chunk, 1024 memory, 1024 output; 60K-token documents | arXiv:2507.02259v2 §4 | verified 2026-09-15 | Table 1: 71.88% at 3.5M tokens for the 7B model |
| MEM1-7B | 7B | RL | retained context per turn | at most two `<IS>`, two `<query>`, one `<info>` | arXiv:2506.15841v2 §3.1 | verified 2026-09-15 | Table 1: EM 1.97 at 16 objectives with 10.4 × 10² peak tokens |
| Claude Sonnet 4.5 context editing + memory tool | not reported | eval-gate | trigger threshold; cleared block types; eval set | not reported (checked the announcement post only) | anthropic.com/news/context-management | not reported | "39%" and "29%" improvements and 84% token reduction, with no baseline or metric definition given |


**소규모 범용 run의 시작점.** released instruction model에 training budget이 없다면, 위 verified 행이 지지하는 구성은 다음과 같다. interleaved reasoning을 history에 보존하고 tool result는 user message가 아니라 tool message로 보낸다([[deepseek-v3.1]] §3.2.1, 671B MoE, agentic benchmark에서 평가). 마지막 5 round보다 오래된 tool observation을 fold한다([[glm-5]] §4.2.4, 128K급 model에서 BrowseComp로 측정). total context가 32k를 넘으면 full reset을 추가한다. 이는 GLM-5가 자기 model과 harness에서 parameter search로 고른 threshold다. 이 수치는 모두 web-search tool과 judge-scored benchmark를 쓰는 frontier-scale model에서 tuning됐다. `k = 5`와 `T = 32k`의 다른 model size·task family transfer는 보고되지 않았으므로 default처럼 쓰기 전 자신의 evaluation에서 다시 선택해야 한다.

## 일반화 관점

**(a) 범위를 넓히는 것.** compression step을 inference에 적용하는 대신 학습하면, 시험한 모든 report에서 trained budget을 넘어 generalize한다. SUPO 64K/192K checkpoint는 64K×24까지 계속 좋아진다(학습 설정 53.0% 대비 60.0%; [[supo-summarization-rl]] App. C.2). 8K에서 학습한 MemAgent는 3.5M token에서 71.88%를 유지한다([[memagent]] Table 1). 2-objective task에서 학습한 MEM1은 objective가 16으로 늘 때 baseline이 collapse하는 반면 개선한다([[mem1]] Table 1). 최대 10 branch로 학습한 folding agent는 50-question compound task에서 평균 32.6 branch를 쓴다([[context-folding]] §4.4.2). 둘째 breadth effect는 architecture다. [[resum]] summary tool은 external이므로 retraining 없이 세 size의 released agent를 upgrade한다(Table 1).

**(b) 좁아짐 또는 망각을 일으키는 것.** 측정된 mechanism은 셋이다. 첫째, irreversible compression은 나중에 가치가 나타나는 evidence를 버린다. [[manus-context-engineering]]은 이 논거로 restorable compression을 만들며, [[anthropic-context-engineering]]도 recall 전에 precision에 맞춘 compaction prompt의 같은 위험을 지목한다. 둘째, non-terminating rollout을 벌주면 summarization 행동 자체가 사라진다. mask ablation은 BrowseComp-Plus에서 9점을 잃고 unmasked run은 5 epoch 대신 3에서 중단할 만큼 악화됐다([[supo-summarization-rl]] §4.2, §5.1, Table 1). 셋째, uniform context는 uniform behaviour를 유도한다. [[manus-context-engineering]]은 context가 비슷한 action-observation pair로 차면 agent가 반복 action pattern에 빠진다고 보고하고 structured variation을 대응책으로 주입한다. gradient 없이 순전히 context construction이 output distribution을 좁히는 사례다. 넷째는 보고되었지만 측정되지 않았다. [[kimi-k2-5]]는 reactive truncation이 “often sacrifice[s] structural information or intermediate reasoning”이라고 하나 loss를 수량화하지 않았다.

**(c) 이 단계를 측정하는 법.** source 하나가 만족하거나 위반한 다음 네 요구가 있다.

- **두 조건을 함께 보고한다.** 동일 checkpoint와 동일 judge에서 context management 유무를 모두 낸다([[glm-5]] Table 7은 여섯 model에 그렇게 한다).
- **step budget을 고정하거나 그것에 대해 plot한다.** [[deepseek-v3.1]] Figure 6과 [[glm-5]] Figure 8은 모두 real step에 대한 accuracy를 그린다. 이것만이 “더 좋은 context policy”와 “더 많은 serial compute”를 분리한다.
- **held-out split과 held-out 대상이 무엇인지 말한다.** [[supo-summarization-rl]]은 BrowseComp-Plus 830 question 중 100개와 다른 seed problem의 CodeGym 128개를 hold out한다. [[resum]]은 benchmark별로 hold out한다. 둘 다 non-agentic capability panel은 보고하지 않아 context-management RL 뒤 general ability retention은 측정되지 않았다.
- **abstention과 wrong answer를 따로 score한다.** [[chroma-context-rot]]는 length에 따라 두 failure mode가 다르게 움직이고 model family마다 다름을 보인다. 합쳐진 agentic score는 “policy가 evidence를 버렸다”와 “evidence를 버리고도 model이 answer를 주장했다”를 구별하지 못한다.

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| context policy를 확인하지 않고 report 사이 agentic score를 비교 | 다른 차이가 없는데 BrowseComp에서 두 모델이 15점 차이 | evaluation appendix에서 trigger·strategy를 읽고 like with like만 비교한다([[glm-5]] Table 7은 두 column을 갖는다) |
| model이 thinking mode인데 harness가 tool result를 user message로 보냄 | turn당 token 수가 계속 늘고 모델이 매 reasoning block에서 plan을 재도출 | serialized request를 보고 tool result role이 `tool`인지 `user`인지 확인한다([[deepseek-v3.1]] §3.2.1) |
| OpenAI-compatible client가 prior reasoning을 조용히 drop | long-horizon benchmark는 떨어지지만 single-turn benchmark는 그대로 | client가 reasoning field를 return·resend하는지 확인하고 retained 대 discarded run 하나를 비교한다([[minimax-m2-interleaved-thinking]]) |
| training 때와 inference 때 context policy가 다름 | training reward는 오르나 evaluation은 움직이지 않음 | training-time와 deployment policy를 따로 평가한다. [[resum]] Table 2는 GRPO-on-ReAct가 ReSum inference로 transfer하지 못함을 보인다 |
| non-terminating rollout을 failure로 score | training에서 summarization rate가 0으로 내려가고 working context가 saturate | `p_summary`를 log하고 step/summary-limit rollout을 mask한다([[supo-summarization-rl]] §4.2) |
| rollout이 아니라 segment에 advantage normalize | summary가 많은 rollout이 group statistic을 지배하고 짧은 rollout advantage가 작아짐 | held-out split에서 Eq. 3·Eq. 4를 비교한다(CodeGym 47.7% 대 42.1%) |
| run 중 dynamic tool set을 추가·제거 | cache hit rate 하락, schema violation, undefined tool call 출현 | tool list를 고정하고 대신 logit을 mask한다([[manus-context-engineering]]) |
| prompt 위쪽에 timestamp 또는 nondeterministic serialization | 기능 변화 없이 cache hit rate가 거의 0 | 연속 request prefix를 byte 단위로 diff한다([[manus-context-engineering]]) |
| 두 가지가 바뀐 benchmark gain을 compaction에 귀속 | 두 column의 model과 reasoning effort가 모두 다름 | 한 checkpoint의 with/without 비교를 요구한다([[openai-codex-max-compaction]]은 제공하지 않음) |
| self-correction error trace에 의존하면서 observation을 fold | fold point 이후 같은 실패 action을 반복 | 실패 tool-call observation을 fold window 밖에 남기거나 restorable pointer를 남긴다([[manus-context-engineering]] 대 [[glm-5]]) |

## 이해도 점검

1. GLM-5 keep-recent-k는 observation을 fold하고 reasoning block은 남긴다. MiniMax는 reasoning block 유지가 BrowseComp에서 12.6점 가치라고 보고한다. state를 앞으로 운반하는 block type 관점에서 두 finding이 모순이 아니라 일관된 이유를 설명하라.
2. §6 예시에서 keep-recent-k는 horizon을 63에서 241 turn으로 늘렸다. observation이 1,500이 아니라 6,000 token이면 무엇이 변하는지 도출하고 두 policy 비율이 어느 방향으로 움직이는지 설명하라.
3. [[supo-summarization-rl]]은 trajectory group 대신 rollout group에서 advantage를 계산하며, trajectory-group variant는 CodeGym에서 5.6점 낮다. 어떤 rollout은 segment 8개, 다른 것은 1개일 때 trajectory-group statistic의 어느 property가 바뀌는지 causal account를 제시하라.
4. [[context-folding]] advantage는 `clip(R_i + Q_{i,t}, 0, 1)`을 쓴다. 실패 rollout 안 process penalty에 무슨 일이 생기는지 도출하고, failure에 penalty가 stack되게 두지 않는 이유를 설명하라.
5. 동료가 agent score 74, 경쟁자는 61이라며 자기 model이 더 좋다고 한다. 비교를 받아들이기 전 필요한 세 가지 구체 configuration fact를 나열하고, 각각의 precedent로 어느 published table을 가리킬지 말하라.
6. [[manus-context-engineering]]은 failed action을 context에 남기라고 하고 [[glm-5]]는 다섯 round보다 오래된 모든 observation을 fold한다. 두 권고가 다른 행동을 내는 task를 구성하고 어느 observable로 그 task에 어느 쪽이 맞는지 판별할지 말하라.
7. [[resum]]은 ReAct rollout의 plain GRPO가 ReSum inference로 transfer하지 않는다고 찾는다. policy gradient가 계산된 state distribution 관점에서 mismatch를 설명하라.
8. MemAgent는 8K window에서 학습해 3.5M token에서 71.88% accuracy를 유지하고, long context용으로 학습한 1M-context model은 896K에서 0.00%다. 이 비교가 long-context architecture work에 대해 입증하는 것과 입증하지 않는 것을 말하라.

## 연결

- **이전:** ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability. 여기의 모든 method가 segmented trajectory용으로 수정하는 loss-masking rule과 turn-level credit assignment를 제공한다.
- **의존:** ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation. 이 장의 동기인 “stated window는 stated size까지 사용 가능하지 않다”는 측정을 제공한다.
- **다음:** ch-45d — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL. 이 장의 context policy를 그것을 보고한 model들의 full recipe 안에 놓는다.

## 출처

- [[chroma-context-rot]] — task를 고정한 18-model input-length accuracy 측정. §1·일반화 관점에서 쓴 LongMemEval focused 대 full 비교와 abstention 대 hallucination 분리.
- [[context-length-alone-hurts]] — retrieval success를 통제한 length-driven degradation; §1을 뒷받침한다.
- [[lost-in-the-middle]] — long context 안 retrieval의 position dependence. §1과 §7 recitation rule의 배경.
- [[anthropic-context-engineering]] — context engineering·attention budget 정의, compaction·structured note-taking·sub-agent architecture라는 세 long-horizon technique, 1,000–2,000-token sub-agent return size.
- [[anthropic-context-management]] — context editing과 memory tool platform feature, 39% / 29% / 84% 수치와 빠진 baseline, DeepSeek Discard-all과 닮은 “new context tool”.
- [[minimax-m2-interleaved-thinking]] — 다섯 agentic benchmark의 prior-round thinking 보존 대 폐기(§3).
- [[deepseek-v3.1]] — V3.2 report. tool call thinking-retention rule(§3.2.1), window 80% trigger, Summary / Discard-75% / Discard-all 비교, BrowseComp 51.4 → 67.6 pair(§4.1, §4.4).
- [[glm-5]] — k = 5 keep-recent-k와 55.3 → 62.0 효과, `T = 32k` Hierarchical Context Management, judge standardization, 여섯 model with/without 표(§4.2.4, §6.1.3, Table 7).
- [[kimi-k2-5]] — HLE의 Hide-Tool-Result, BrowseComp Discard-all 60.6 → 74.9, overflow-counts-as-failure default, proactive context management로서의 Agent Swarm(§5, appendix).
- [[supo-summarization-rl]] — summarization-augmented MDP, gradient decomposition, rollout-group advantage, overlong mask와 ablation, test-time summary-budget sweep(§3–§5, Tables 1·3).
- [[resum]] — external summary tool, advantage broadcast, ReSumTool-30B, 1K-sample RL result, MEM1 token-cost 비교, 32k–128k context sweep(§3–§4, Tables 1–3).
- [[context-folding]] — branch·return tool, fold에서 KV-cache rollback, clip이 있는 FoldGRPO token-level process reward, 32K × 10 대 327K ReAct 비교(§2–§4, Tables 1–2).
- [[memagent]] — multi-conversation DAPO로 학습한 fixed-size overwritten memory, 8K training window allocation, 3.5M token까지의 RULER-HQA 표.
- [[mem1]] — `<IS>` tag를 쓰는 constant-memory consolidation, 2-objective → 16-objective generalization, peak-token 비교.
- [[manus-context-engineering]] — operational metric으로서 KV-cache hit rate, 100:1 input-to-output ratio, tool의 mask-not-remove, restorable compression, recitation, failure를 context에 남기기(practitioner evidence이며 controlled experiment 없음).
- [[openai-codex-max-compaction]] — native training된 multi-window capability로서 compaction, thinking token 30% 감소 주장, ablation되지 않은 appendix 표.
- [[mirothinker]] — thought/action을 보존하면서 K = 5로 tool output을 최근성 기반 보존, 256K window에서 600 tool call.
- [[browsecomp-plus]] — [[supo-summarization-rl]]과 [[context-folding]]이 training·evaluation environment로 쓴 retrieval-grounded BrowseComp corpus.
- [[grpo]] — §5의 모든 segmented method가 수정하는 group-relative advantage.
