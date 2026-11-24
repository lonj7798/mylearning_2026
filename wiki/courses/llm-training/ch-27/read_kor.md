<!-- chapter: ch-27
     track: synthetic
     kind: content
     title: Agentic Trajectory Data
     deps: [ch-26]
     sources: [[agenttuning]], [[fireact]], [[excerpts/lumos]], [[autoact]], [[agent-flan]], [[agentinstruct]], [[excerpts/webarena-data]], [[excerpts/swe-gym]], [[excerpts/swe-smith]], [[r2e-gym]], [[deepswe]], [[excerpts/swe-rl]], [[kimi-k2]], [[kimi-k2-recipe]], [[eto-trial-and-error]], [[excerpts/learning-from-failure-nat]], [[swe-rebench]], [[swe-bench-illusion]], [[agentic-finetuning-misalignment]]
     figures: figures/action-space.html
     revised: 2026-09 (generality revision)
-->

# Chapter 27 — Agentic Trajectory Data

> **핵심 통찰.** Agent trajectory는 model action과 environment observation의 sequence이며, external check가 score할 outcome으로 끝난다. Trajectory만 훈련하면 나온 environment에서는 개선하지만 transfer하지 못한다. Agent trajectory 1,866개만 훈련한 AgentLM-7B는 held-in agent task 1.34, held-out agent task 0.09이지만, 같은 data를 `η = 0.2`로 ShareGPT와 섞으면 1.96, 0.67이다([[agenttuning]] Table 5). Software-engineering issue에서는 Llama-3.3-70B-Instruct에 patch-similarity reward로 RL을 한 결과 SWE-bench Verified 41.0%이며 out-of-domain category 5개 모두에서 base model 이상이었다. 같은 PR seed로 만든 SFT baseline은 5개 중 4개에서 base보다 낮았다(strict MATH scoring)([[excerpts/swe-rl]] Tables 1, 3). Repository diversity는 controlled comparison 하나에서 held-out result를 바꿨다. Qwen2.5-Coder-Instruct trajectory 700개를 고정했을 때 training repository 수를 4→100으로 늘리면 SWE-bench Verified resolve rate는 10.3%→15.1%다([[excerpts/swe-smith]] §4.1, 그림 5; 이 그림의 model size는 미명시).
>
> **지침.** Agent trajectory를 SFT에 넣을 때 general instruction data와 섞고 held-out agent environment 및 general benchmark에서 share를 고르라. Agent-only training은 7B/13B held-out score를 collapse시켰다([[agenttuning]] Table 5). AgentTuning scan은 7B에서 `η = 0.2`를 선택했고 Agent-FLAN은 ShareGPT:agent 1:1과 capability weight `1 : 0.25 : 0.75`를 썼다([[agent-flan]] App. A). Executable environment와 checkable outcome이 있으면 held-out environment family 및 training data 이후에 만들어진 benchmark에서 pass@1/pass@k를 보고하라. 같은 model도 SWE-bench Verified보다 fresh task에서 낮다(DeepSeek-V3-0324 39.7% vs 21.3%, [[swe-rebench]] Table 2). Failed trajectory를 쓸 때는 discarded/content/conditioning label/gradient 중 무엇인지 밝히고 unseen variation에서 success-only baseline과 비교하라. Use에 따라 measured effect sign이 다르기 때문이다(Negative samples and negative feedback 참고). 그렇지 않으면 failure를 버리고 filter를 엄격히 유지하라. Unfiltered self-generated trajectory로 Llama-2-70B-chat을 훈련하면 no fine-tuning보다 낮았다(HotpotQA F1 32.51 vs 32.84, [[autoact]] Table 2).

## 학습한 버전에 대한 정정

1. “All six were published between Oct-2023 and Apr-2024” → AgentInstruct는 arXiv v1 2024-07이며 agent trajectory가 아닌 instruction pair를 만든다([[agentinstruct]], Year; Findings “Agentic training”).
2. AgentTuning “Mixing ratio: 1:10 agent:ShareGPT preserves chat quality”, takeaway “[[agenttuning]]'s 1:10 rule generalizes” → mixture weight는 sampling 기준 `η = 0.2`(agent 20%, ShareGPT 80%)이고 7B held-out task에서 0–1 scan으로 선택했다. Reported ablation은 agent-only/general-only/mixed이며 1:1/1:10/1:50이 아니다([[agenttuning]], §2.2.2, App. A, Table 5).
3. FireAct “~2,000 trajectory”, “CoT-only 38.9, ReAct-only 37.3, Reflexion-only 35.2, three-method mix 40.0”, “method-specific system prompt”, “Wikipedia search” → Generation에는 HotpotQA question 2,000개를 썼다. GPT-3.5에서는 ReAct 39.4, +CoT 41.0, +Reflexion 38.8, all three 40.0이므로 three-way mix가 최고가 아니다. 모든 data는 ReAct format으로 바꾸며 method는 암묵적으로 고르고 tool은 SerpAPI의 Google search다([[fireact]], §3, §4, Table 4).
4. Lumos “40K task → 200K triple”, “Lumos-O(onetime)”, “monolithic ReAct 대비 ~20이 아니라 ~8점 drop” → planning annotation 55,382개, grounding annotation 55,499개다. Lumos-O는 OnePass다. Unseen-task result는 Lumos-I-All-13B의 WebShop average reward 50.3, InterCodeSQL success 7.3%이며 8-vs-20 비교는 없다([[excerpts/lumos]], §4.1, Table 1e).
5. AutoAct “self-consistency filter”, “iterate… iteration 4에서 saturate”, “13B ~36 EM, GPT-4-distilled baseline 4점 이내” → reward = 1 trajectory만 유지하고 iterate하지 않으며 trajectory 200개로 훈련한다. Llama-2-13B HotpotQA F1은 40.49, FireAct 36.94다([[autoact]], §2.3, §3, Table 1).
6. Agent-FLAN “hallucination class 네 개(format/action/parameter/relevance)”, “~85K”, “hallucinated call 5× 감소”, “GPT-4 prompt negative”, “MT-Bench 0.5 이내” → hallucination type은 format/action 둘, agent sample 24,703개다. Negative는 no/irrelevant tool로 gpt-3.5-turbo가 답한 ToolBench query 761개에서 만든다. H_ReAct 15.6→9.9, H_General 13.5→11.9이고 MT-Bench는 보고하지 않는다([[agent-flan]], §4.3, App. B–C, Table 3).
7. Agent-FLAN “custom `<tool>` token이 markdown-code-fenced JSON이 됨” → Thought-Action-Action Input turn을 multi-turn chat으로 바꾸고 inserted statement로 JSON argument를 split한다([[agent-flan]], §4.1).
8. AgentInstruct “stage 4개”, “generator agent 43개”, “25M”, “MATH 3×”, “refinement loop 최대 3 iteration” → flow는 content transformation/seed instruction generation/refinement 셋, reading-comprehension question type 43개, flow pair 약 22M + Orca-2.5 pair 약 3.8M이다. Report gain은 AGIEval +40%, MMLU +19%, GSM8K +54%, BBH +38%, AlpacaEval +45%이고 MATH 3×/iteration count는 없다([[agentinstruct]], §2, §3.1, Abstract).
9. WebArena “app 다섯 개—GitLab, Reddit-clone(Postmill), Shopping(Magento), OpenStreetMap, Calendar”, “GPT-4 ~35%” → Site는 e-commerce store(OneStopShop), online-store CMS, Reddit-style forum, GitLab, map과 English Wikipedia/calculator/scratchpad이며 calendar는 없다. GPT-4는 human 78.24% 대비 14.41%다([[excerpts/webarena-data]], §2.2, Table 2).
10. SWE-Gym “Qwen-2.5-Coder-7B 3.0%→15.3% SWE-Bench Verified… verifier 20.3%”, “teacher Qwen-2.5-Coder-32B/Claude-3.5, K=10”, “Lite compatible task 491개”, `browse` action → gpt-4o-2024-08-06/claude-3-5-sonnet-20241022에서 sample한 successful trajectory 491개 fine-tuning은 Verified 1.8→10.6(7B), 4.0→16.4(14B), 7.0→20.6(32B), Lite 3.0→15.3(32B)다. Verifier 포함 32B는 Verified 32.0%(Best@16), Lite 26.0%; SWE-Gym Lite는 instance 230개이며 browser tool은 disabled다([[excerpts/swe-gym]], §3.2, §4.2, Table 3, §5.1).
11. SWE-RL base/hyperparameter/filter 주장의 정정: base는 Llama-3.3-70B-Instruct, raw PR instance 약 11M 중 RL seed 273k개, step 1,600, 16k context, step당 문제 32개 × rollout 16개, H100 512개로 약 32시간이다. LR/β/ε는 보고하지 않았고 열거한 filter는 paper filter가 아니다([[excerpts/swe-rl]], §2, §3.1, App. A).
12. SWE-RL reward “`ratio()` [0,1] 및 binary threshold experiment” → wrong format에는 −1, 그 외 `SequenceMatcher` similarity다. Ablation은 exact-match 0/1 reward이며 repair 29.0 vs 34.8이다([[excerpts/swe-rl]], 식 1, §3.6).
13. SWE-RL comparison/out-of-domain 수치 정정: Table 1은 SWE-Gym-32B 32.0, SWE-RL 41.0을 목록으로 들며 DeepSeek-Coder-V2는 없다. Base 대비 HumanEval+ 76.2→79.9, MATH strict 63.2→73.7, MMLU 86.49→86.82이며 BBH는 평가하지 않는다([[excerpts/swe-rl]], Tables 1, 3).
14. “Rule-based dense reward scale, execution-based sparse reward does not” → DeepSWE는 0/1 test reward만으로 Qwen3-32B RL을 훈련해 SWE-bench Verified Pass@1 42.2%이고 Kimi K2 RL은 concurrent instance 10,000개 초과를 지원하는 sandbox에서 unit test를 실행한다([[deepswe]], §2.2, §4; [[kimi-k2]], §3.2.1).
15. Kimi K2 agentic pretraining/sub-agent generator 주장 정정: “Pretrain injection beats post-training retrofit”이라는 [[kimi-k2]]의 명시적 주장도 source에는 없으며, agentic data synthesis는 post-training SFT에 있고 pre-training은 rephrasing을 포함한 web text/code/math/knowledge 15.5T token이다. Agentic pre-training corpus/sub-agent generator는 보고하지 않는다([[kimi-k2]], §2.2, §3.1.1, Verification).
16. Kimi K2 self-rubric/single scalar 주장 정정: K2 critic은 core/prescriptive/human-annotated rubric을 대상으로 actor response를 pairwise compare해 rank하며 verifiable-reward rollout으로 refine한다. Weighted scalar 하나는 설명하지 않는다([[kimi-k2]], §3.2.2, Verification).
17. Kimi K2 “tool library 20,000+”, “SWE-Bench Verified ~65% Claude-3.5-Sonnet 경쟁” → real MCP tool 3000+와 synthetic tool 20,000+이며 SWE-bench Verified는 agentic single attempt 65.8, multiple attempt 71.6이고 Claude Sonnet 4/Opus 4/GPT-4.1 등을 비교한다([[kimi-k2]], §3.1.1, Table 3).
18. 100K tail에 FlashAttention-3/gradient checkpointing이 필요하다는 주장 → 이 chapter source에는 없다(long-sequence memory choice는 [[ch-05]]). Source는 SWE-Gym SFT max context 32,768, trajectory 약 19 turn/평균 19,000 token, R2E-Gym SFT 20K context 및 future work로 context parallelism을 보고한다([[excerpts/swe-gym]], §4.2, App. B.2; [[r2e-gym]], App. B).
19. Cost table의 AgentTuning ~$20K/FireAct ~$3K/AgentInstruct >$500K/SWE-Gym ~10K H100-hour/SWE-RL ~1M H100-hour/K2 ~1T agentic token → source에 없다. Reported figure는 SWE-RL H100 512개 약 32시간, SWE-smith dataset construction $1,360이다([[excerpts/swe-rl]], §3.1; [[excerpts/swe-smith]], §2.2).
20. “Negative 없는 Agent SFT는 open-loop controller”, negative가 모든 agent corpus에 필요하다는 takeaway 정정: Agent-FLAN negative는 correct plain-text target과 pair한 prompt이며 ordinary cross-entropy로 훈련한다. 7B에서 Agent-H HScore 84.5→89.1, T-Eval 66.3→66.0이다([[agent-flan]], Table 3, Findings “Negative samples”).
21. “ch-26의 [[front-loading-reasoning]]과 연결”, “[[ch-26]]으로 reasoning/tool-call data synthesize 방법을 안다” → Reasoning trace는 [[ch-24]], ch-26은 tool/function-calling data를 다룬다.

## 이 장이 범용 모델에 중요한 이유

범용 assistant는 code 실행, repository edit, browse, 여러 turn의 tool call처럼 action도 한다. 이 skill data는 measurable한 한 가지 점에서 chat data와 다르다. 각 observation은 environment가 만들므로 trajectory success를 judge reading 대신 external test로 check할 수 있다. 이 property는 filtering/RL을 가능하게 하고 model을 trained-on environment(held-in)와 never-seen environment(held-out)에서 score할 수 있어 narrowing도 쉽게 측정하게 한다.

Pipeline에서 agentic trajectory는 세 지점에 들어간다. Mid-training은 post-training 전 repository/execution-trace data를 더할 수 있다([[ch-32d]]). SFT는 filtered teacher trajectory를 쓴다(이 장 및 mixture share의 [[ch-30b]]). RL은 같은 environment를 outcome reward로 쓴다([[ch-45b]]). 이 장은 trajectory 생성/filter, SFT mixture가 차지할 수 있는 share, source가 transfer에 대해 측정한 것을 다룬다.

## §1 Agent data의 용어와 네 축

**Trajectory.** Trajectory는 `(task, a₁, o₁, a₂, o₂, …, a_n)` sequence다. 각 `aᵢ`는 model turn(thought/action), 각 `oᵢ`는 environment reply다. AgentTuning은 이를 final reward `r ∈ [0, 1]`을 가진 conversation `(u₁, a₁, …, u_n, a_n)`으로 쓴다([[agenttuning]] §2).

**Scaffold.** Scaffold는 model 주위의 prompt, tool set, control loop다. 예를 들어 OpenHands CodeAct(bash terminal/file editor) 또는 Agentless Mini(fixed localize–repair–test pipeline)다([[excerpts/swe-gym]] §4.2; [[excerpts/swe-rl]] App. B).

이 장의 모든 agent dataset은 축 네 개로 기술할 수 있다.

| 축 | 답하는 질문 | Source example |
|---|---|---|
| Environment | Action을 실행하고 state를 유지하는 것은 무엇인가? | WebArena self-hosted website; repository Docker image; state를 “maintains and updates”하는 K2 LLM tool simulator |
| Action space | Legal output은 무엇인가? | WebArena browser action 12개; search/replace edit; OpenHands bash + editor; MCP tool call |
| Observation | Action 뒤 model이 보는 것은 무엇인가? | accessibility tree/screenshot/HTML; shell output; simulated tool JSON |
| Success signal | Trajectory를 keep/reward할지 결정하는 것은 무엇인가? | Page/database state functional check; hidden unit test; oracle patch similarity; task rubric LLM judge |

Source: [[excerpts/webarena-data]] §2.3–2.4; [[excerpts/swe-gym]] §4.2; [[excerpts/swe-rl]] 식 1; [[kimi-k2]] §3.1.1.

[figures/action-space.html](figures/action-space.html)은 reader가 environment 네 개 사이를 switch하여 축을 비교하고 model size별 AgentTuning held-in/held-out bar를 보며 editable patch의 SWE-RL reward/group advantage를 계산하게 한다.

**Loss masking.** Agent SFT에서는 loss를 model turn에만 계산한다. AgentTuning은 loss를 “on model output only” 적용하고, ETO는 instruction/observation token을 mask하며, DeepSWE multi-turn RL은 environment observation을 mask한다([[agenttuning]] §2.2.3; [[eto-trial-and-error]] 식 5; [[deepswe]] §2.3).

    L(θ) = −(1 / Σₜ mₜ) · Σₜ mₜ · log π_θ(xₜ | x₍<t₎)

- `xₜ`: serialized trajectory의 token t, `x₍<t₎`: 이전 모든 token.
- `mₜ ∈ {0, 1}`: token t가 model turn이면 1, system prompt/task/observation token이면 0.
- `π_θ`: training 중인 model.

**계산 예시(illustrative token count).** Trajectory가 task token 300개, model turn 3개(120, 80, 150 token), observation 3개(900, 1,500, 40 token)를 가진다고 하자. Sequence는 3,090 token, `Σₜ mₜ = 350`이므로 token 11.3%가 loss를 carry한다. Token을 전혀 mask하지 않으면 loss-bearing token 3,090개 중 2,740개(88.7%)가 inference 때 model이 생성하지 않는 task text/shell output/page text다. 따라서 loss 대부분이 environment output prediction을 측정한다. Masking은 self-correction trace에서도 중요하다. History로 context에 남은 failed action은 mask하고 recovery turn은 훈련할 수 있다.

## §2 Agent SFT 설계, 2023–2024

### §2.1 AgentTuning: filter한 GPT-4 trajectory와 mixture weight

**정의.** AgentTuning은 ShareGPT conversation과 섞은, AgentBench task 여섯 개에서 나온 GPT-4 trajectory 1,866개로 Llama-2-chat(7B, 13B, 70B)을 fine-tune한다([[agenttuning]] §2, Table 1).

**문제.** Open model의 AgentBench 평균 점수는 0.42였고 API model은 2.24였으며, 목표는 general score를 낮추지 않고 agent score를 올리는 것이었다([[agenttuning]] Fig. 1b, Abstract).

**방법.**
1. Instruction을 수집한다. ALFWorld, WebShop, Mind2Web, Knowledge Graph의 train split, Operating System용 Self-Instruct, Database용 BIRD에서 task를 유도한 것을 사용한다(§2.1.1).
2. GPT-4가 1-shot example과 함께 action을 하게 한다. 모든 action에는 ReAct thought가 있다(§2.1.2). ReAct는 각 action 앞에 free-text reasoning step(thought)이 있고 그 뒤에 environment observation이 이어지는 trajectory format이다.
3. reward `r = 1`인 trajectory를 유지한다(Mind2Web은 `r ≥ 2/3`). 35,341개의 instruction에서 1,866개의 trajectory가 남으므로 keep ratio는 5.29%다(§2.1.3, Table 1).
4. 아래 mixture로 훈련한다.

    J(θ) = η · E₍(x,y)∼D_agent₎[log π_θ(y|x)] + (1 − η) · E₍(x,y)∼D_general₎[log π_θ(y|x)]

- `D_agent`: 1,866개 trajectory, `D_general`: ShareGPT, `η`: agent sampling ratio.
- `x`: instruction과 history, `y`: model response, `π_θ`: fine-tune 중인 model.
- `η`는 7B에서 0부터 1까지 0.1 간격으로 scan했으며, held-out task에서는 `η = 0.2`가 가장 좋았다(§2.2.2).

**계산 예시(derived).** Batch size 64(paper에는 unit을 명시하지 않음, [[agenttuning]] Table 6)와 `η = 0.2`라면, batch 하나에는 기대값으로 agent example 12.8개와 general example 51.2개가 들어간다. Trajectory 1,866개를 한 번 모두 보려면 `1,866 / 12.8 ≈ 146` step이 필요하고, 그 동안 general example 약 7,464개를 draw한다. 실제로 sample한 ShareGPT conversation 수와 epoch 수는 보고하지 않았다.

**증거.** Filtering은 중요하다. 7B에서 filtered data의 점수는 held-in 1.96 / held-out 0.65였고, unfiltered는 1.34 / 0.47이었다(Table 2). Main result와 mixture ablation은 §3에 있다.

### §2.2 FireAct: 한 corpus에 넣은 여러 prompting method

FireAct는 HotpotQA에서 CoT, ReAct, Reflexion prompt로 수집한 successful GPT-4 trajectory를 모두 ReAct format으로 변환하여 fine-tune한다([[fireact]] §3). ReAct trajectory 500개로 Llama-2-7B를 fine-tune하면 HotpotQA EM은 14.8에서 26.2로, GPT-3.5는 31.4에서 39.2로 올랐다(Table 2). Method를 더하는 효과는 일률적이지 않다. GPT-3.5의 경우 ReAct 39.4, +CoT 41.0, +Reflexion 38.8, 세 가지 모두 40.0이었고(Table 4), 가장 좋은 mix는 base model에 따라 달랐다(§6, Fig. 4). 조건은 task type 하나(QA)와 tool 하나(Google search)다([[fireact]] §8).

### §2.3 Lumos: 분리한 planning module과 grounding module

Lumos는 Llama-2-7B/13B에서 planning module(task → subgoal)과 grounding module(subgoal → executable action)을 훈련하고, execution module이 tool을 실행한다([[excerpts/lumos]] §2). Training annotation은 기존 benchmark의 ground-truth reasoning step을 GPT-4가 이 format으로 변환하여 만들며, filtering 뒤 planning annotation 55,382개와 grounding annotation 55,499개가 된다(§4.1). 같은 data에서 modular formulation은 integrated single-module formulation보다 높은 점수를 냈다. 예를 들어 HotpotQA LLM accuracy는 Lumos-I 45.9, integrated 39.6이었다(Table 3c). 결과는 single study의 결과다.

### §2.4 AutoAct: 훈련할 model 자체가 합성한 trajectory

AutoAct는 소수의 QA example과 15-tool library에서 시작한다. Backbone model이 self-instruct를 수행하고, tool을 선택하고, ReAct trajectory를 합성하며, `reward = 1`인 것만 남긴 뒤, 이를 LoRA-tune하여 plan, tool, reflect agent로 만든다([[autoact]] §2.2–2.3). Llama-2-70B-chat과 trajectory 200개로 HotpotQA F1 48.47에 도달했으며, 이는 GPT-4 trajectory로 훈련한 FireAct의 42.70보다 높다(Table 1). Filter를 제거하면 F1은 32.51로 낮아져, fine-tuning을 하지 않은 경우의 32.84보다도 낮다(Table 2).

### §2.5 Agent-FLAN: chat-format 변환, capability weight, tool-hallucination sample

Agent-FLAN은 Llama-2의 agent tuning에서 format token이 content보다 더 빨리 학습되고, capability마다 converge하는 속도가 다르며, tuning된 agent는 tool call을 hallucinate한다는 세 관찰을 보고한다([[agent-flan]] §3). 이에 대한 대응은 다음과 같다. ReAct turn을 multi-turn chat으로 변환한다(T-Eval 61.8 → 64.9). Reasoning : retrieval : understanding에 `1 : 0.25 : 0.75`의 weight를 준다(37.3M token 대신 18.1M token, T-Eval 66.3). 그리고 negative sample을 추가한다(§4.1–4.3, Table 2). T-Eval은 tool-use evaluation suite다. Agent-H는 저자가 만든 tool-hallucination benchmark이며, 그 HScore는 `(1 − H_ReAct)`와 `(1 − H_General)`의 평균이다. 여기서 `H_ReAct`와 `H_General`은 correct answer가 plain reply인 item에서 response가 ReAct 또는 general tool-call keyword를 포함하는 share다([[agent-flan]] §4.3). ShareGPT와 agent data를 1:1로 섞고 agent sample 24,703개를 사용한다(App. A, Table 6). Llama2-7B overall score는 같은 data로 재구현한 AgentTuning의 38.2에 비해 41.7이었다(Table 1). Negative sample은 §7에서 다룬다.

### §2.6 AgentInstruct는 trajectory data가 아니다

Microsoft의 AgentInstruct는 Orca-3용 instruction pair 약 22M개를 만들기 위해 multi-agent flow를 사용한다. 여기서 “agentic”은 generator를 가리킨다. Tool use와 web agent는 17개 skill 중 2개일 뿐이며 agent benchmark는 보고하지 않았다([[agentinstruct]] §2, §3.1, Findings). 이름은 AgentTuning의 trajectory dataset과 비슷하지만, taxonomy-driven synthesis로 분류한다([[ch-21]]).

## §3 Held-in environment와 held-out environment

**정의.** Held-in environment는 training trajectory를 제공한 environment이고, held-out environment는 제공하지 않은 environment다. AgentTuning의 held-out set은 SciWorld, MiniWoB++, HotpotQA, WebArena, ReWOO, Digital Card Game이며, general set은 MMLU, HumanEval, GSM8K, MT-Bench다([[agenttuning]] Table 3).

**문제.** Held-in environment의 점수는 format과 task familiarity를 측정한다. Model이 새 environment에서 action할 수 있는지는 보여 주지 않는다.

**증거([[agenttuning]] Table 5; 점수는 task마다 normalize되어 average model의 점수는 1이다).**

| 크기 | Data | Held-in | Held-out | General |
|---|---|---|---|---|
| 7B | agent-only | 1.34 | 0.09 | 0.22 |
| 7B | general-only | 0.38 | 0.64 | 0.61 |
| 7B | mixed (`η = 0.2`) | 1.96 | 0.67 | 0.63 |
| 13B | agent-only | 1.57 | 0.10 | 0.19 |
| 13B | general-only | 0.43 | 0.81 | 0.63 |
| 13B | mixed | 2.11 | 0.78 | 0.69 |
| 70B | agent-only | 2.47 | 0.87 | 0.83 |
| 70B | general-only | 0.99 | 0.98 | 1.00 |
| 70B | mixed | 2.55 | 1.40 | 0.96 |

**읽는 예.** 7B에서 agent-only data는 general-only data와 비교해 held-in을 0.38에서 1.34로 올리고, held-out은 0.64에서 0.09로 낮춘다. 이는 86% 하락이다. Mixture는 held-out을 general-only의 0.64에 가까운 0.67로 유지하면서 held-in을 1.96으로 올린다. 7B와 13B에서 mixture는 held-out에서 general-only를 이기지 못한다(0.67 vs 0.64, 0.78 vs 0.81). 70B에서만 이를 넘는다(1.40 vs 0.98). 결과는 single study이며 seed 수는 명시하지 않았다. 저자들은 agent generalization에 general data가 필요하다고 결론낸다(Interpretation, §3.4).

같은 시기의 다른 held-out 측정은 다음과 같다.

- 새 action이 있는 unseen task에서 Lumos-I-All-13B는 새 action interface의 2–3-shot example과 함께 WebShop average reward 50.3, InterCodeSQL success 7.3%를 기록했다([[excerpts/lumos]] Table 1e, §4.4).
- HotpotQA로 fine-tune한 FireAct GPT-3.5는 Bamboogle에서 prompting의 40.8 대비 44.0을 기록했다. 그러나 HotpotQA로 fine-tune한 Llama-2-7B는 StrategyQA에서 fine-tuning 전 59.0 대비 52.0을 기록했다([[fireact]] Table 3, App. A.9 Table 14).
- ScienceWorld unseen variation에서 ETO는 SFT average reward 53.0을 65.0으로 올렸다([[eto-trial-and-error]] Table 2).
- 7B, 13B, 70B에서 Agent-FLAN은 general-data training과 비교해 MMLU, GSM8K, HumanEval을 −0.3에서 +1.1 point만큼 바꾸었다([[agent-flan]] Table 4).

**한계.** 이 논문의 held-out environment도 같은 research community의 text game, QA, web task에 속한다. 다른 scaffold나 unseen tool protocol을 시험한 것은 없다. [[ch-51a]]는 scaffold transfer와 reliability metric을 다룬다.

**의미.** Mixture share는 held-out score로 선택해야 하는 generality control이다. Held-in score는 오르는데 held-out score가 내리면, 그 gain은 training environment와 그 format에 특이적이며 transferable agent ability를 측정하지 않는다(Interpretation).

## §4 Environment-grounded trajectory

### §4.1 WebArena: self-hosted website의 functional check

WebArena는 self-hosted site category 네 개(e-commerce store, Reddit-style forum, GitLab, online-store CMS)에 map, English Wikipedia, calculator, scratchpad를 더한 evaluation environment다. Reset script가 있는 Docker container로 제공한다([[excerpts/webarena-data]] §2.2). Observation은 HTML, screenshot, accessibility tree일 수 있다. Action space에는 `noop`, `click`, `hover`, `type`, `press`, `scroll`, `tab_focus`, `new_tab`, `tab_close`, `go_back`, `go_forward`, `goto`의 12개 action이 있다(§2.3–2.4, Fig. 4). Task 812개는 functional correctness로 score한다. Answer에는 `exact_match`, `must_include`, `fuzzy_match`를 쓰고, URL 또는 page/database state에는 programmatic check를 쓴다(§3). 달성할 수 없는 task의 expected answer는 `"N/A"`다(§3).

CoT를 쓴 GPT-4의 점수는 prompt가 unachievable task에서 stopping을 허용했을 때 11.70%, 그 hint가 없을 때 14.41%였다. Human은 78.24%였고, hint가 있을 때 GPT-4는 feasible task의 54.9%를 impossible이라고 판단했다(Table 2, §5.1). 위의 agent-training paper에서 WebArena는 held-out environment다(AgentTuning Table 3; Agent-FLAN §4). 그러므로 training source가 아니라 transfer test로 쓰는 것이 가장 적절하다.

### §4.2 SWE-Gym: 실행 가능한 repository와 rejection-sampling SFT

**정의.** SWE-Gym은 repository 11개에서 나온 Python task instance 2,438개이며, 각 instance에는 pre-installed environment와 unit test가 있다. SWE-Gym Lite는 230-instance subset이다([[excerpts/swe-gym]] §3, Table 2).

**방법.**
1. 여러 temperature에서 gpt-4o-2024-08-06과 claude-3-5-sonnet-20241022로 OpenHands CodeAct(bash terminal과 file editor, browser disabled)를 실행한다(§4.2).
2. Test를 실행하여 각 trajectory에 label을 붙이고, success만 남긴다(rejection sampling). 이 run은 compute budget으로 제한되어, 평균 약 19 turn과 19,000 token인 successful trajectory 491개를 만들었다(§4.2).
3. Qwen-2.5-Coder-Instruct 7B/14B/32B를 fine-tune한다. Full fine-tuning, LR `1e-4`, 최대 5 epoch, global batch 8, max context 32,768을 사용한다(App. B.2).

**계산 예시(yield).** SWE-Gym Lite에서 max turn 30으로, temperature 0의 gpt-4o-2024-08-06은 instance 230개 중 19개를 resolve했다. Yield는 8.26%다(App. B.4). 이 yield라면 successful trajectory 491개를 얻으려면 약 `491 / 0.0826 ≈ 5,944`번 시도해야 한다(derived; 실제 run은 model, temperature, task set을 섞었다).

**증거([[excerpts/swe-gym]] Table 3, SWE-bench Verified resolve rate).** 7B는 1.8 → 10.6, 14B는 4.0 → 16.4, 32B는 7.0 → 20.6이었다. Fine-tuning은 action 하나를 반복하며 멈춘 trajectory의 share를 낮췄다(32B Verified 29.4% → 23.8%). Pass@k는 sample한 trajectory `k`개 중 적어도 하나가 task를 resolve한 task의 share다. Best@k는 verifier가 sample `k`개 중 하나를 선택했을 때 resolve되는 share이므로 `Best@k ≤ pass@k`다. Trained verifier를 쓰면 32B agent는 `k = 8`에서 Best@k 29.8%, `k = 16`에서 32.0%였고, pass@k는 `k = 8`에서 37.8%, `k = 16`에서 42.8%였다(§5.1.1).

**한계.** 32B model이 만든 successful trajectory 868개와 teacher trajectory 491개를 함께 넣어 32B model을 fine-tune하면 Lite 점수는 15.3%에서 8.7%로 낮아졌다. 저자들은 self-improvement가 “is not yet working”이라고 말한다(§4.2).

### §4.3 Task와 repository diversity를 늘리기

SWE-smith는 repository마다 environment를 만들고 existing test를 깨뜨려 task를 만든다(LM rewrite, AST modification, reverted PR, combined bug). Repository 128개에서 instance 50k개를 만들었다([[excerpts/swe-smith]] §2). Claude 3.7 Sonnet trajectory 5,016개로 Qwen2.5-Coder-Instruct-32B를 fine-tune하면 SWE-bench Verified에서 40.2%를 기록했다(Table 3). Trajectory 700개를 고정했을 때 resolve rate는 repository 4개에서 10.3%, 25개에서 11.5%, 50개에서 12.9%, 100개에서 15.1%로 올랐다. 저자들은 이를 대략 logarithmic하다고 설명한다(§4.1, Fig. 5; model size는 명시하지 않음). SWE-agent-LM-32B를 SymPy trajectory 700개로 더 fine-tune하면 Verified의 SymPy subset은 33.3%에서 42.4%로 올랐다. 이 subset은 2022-01-01 뒤에 만들어진 instance 22개이므로 9.1 point는 instance 2개다. 반면 SymPy를 제외한 Verified는 40.2%에서 38.3%로 내려갔다(Fig. 4). R2E-Gym은 back-translated issue text가 있는 commit에서 environment 8,135개를 만든다. Trajectory 400개에서 synthetic problem statement는 27.8%, real GitHub issue는 28.0%를 기록했다([[r2e-gym]] §2, §3.1).

### §4.4 Agent benchmark의 contamination

SWE-bench Verified는 public repository 12개에서 가져오며, 그 model score는 fresh task의 score보다 높다. March–April 2025에 만든 SWE-rebench task에서 DeepSeek-V3-0324는 21.3%를 resolve했지만 Verified에서는 39.7%였다([[swe-rebench]] Table 2). Repository name과 issue text만 주었을 때 API model 10개는 Verified instance의 60–76%에서 gold-patch file을 이름으로 맞혔지만, SWE-bench 밖의 repository에서는 53% 미만이었다([[swe-bench-illusion]] §4.1). Training pipeline은 SWE-bench repository를 제외한다(SWE-RL App. A; DeepSWE §2.1). 이는 training-data overlap은 제어하지만 base model이 pre-training에서 본 내용까지 제어하지는 못한다([[ch-48]]).

## §5 SWE-RL: software-evolution data에서의 RL과 out-of-domain 결과

**정의.** SWE-RL은 merged GitHub PR에서 얻은 `(issue, code context, oracle patch)` seed에 GRPO를 적용한다. GRPO(group-relative policy optimization)는 prompt 하나의 rollout group 안에서 normalize한 각 rollout의 reward를 advantage로 쓰는 policy-gradient RL이다([[ch-40]]). Task는 single-turn이며, 관련 file 전체 내용을 받은 뒤 search/replace edit를 작성한다([[excerpts/swe-rl]] §2, Fig. 2).

**Data.** Jan 2015부터 Aug 2024까지의 GitHub event와 clone한 repository 4.6M개에서 aggregated PR 24M개를 얻는다. Bot PR, 비어 있거나 지나치게 큰 change, CodeLlama filter가 flag한 hunk를 제거한 뒤 PR instance 약 11M개가 남는다. 이 중 273k개(linked issue, bug-fix request, programming-file change)를 RL seed로 고른다. SWE-bench가 사용하는 repository는 제외한다(App. A, §2).

**Reward.**

    R(o) = −1 if o has wrong format; otherwise R(o) = compare(patch_pred, patch_gt)

- `o`: model rollout, `patch_pred`: `o`에서 추출한 patch, `patch_gt`: oracle patch.
- `compare`: Python `difflib.SequenceMatcher(...).ratio()` = `2M / T`. `M`은 matching block에서 match된 character 수, `T`는 두 string 길이의 합이다.

GRPO advantage는 같은 prompt의 rollout `G`개에 대해 `Aᵢ = (rᵢ − mean(r₁…r_G)) / std(r₁…r_G)`다(Eq. 2).

**계산 예시.** Oracle line `return a-b\n`(character 11개)과 prediction `return a+b\n`(character 11개)은 match된 character 10개를 공유하므로 ratio는 `2·10 / 22 = 0.909`다. Reward가 `(−1, 0.2, 0.6, 0.8)`인 rollout 네 개의 group을 보자. Mean은 0.15, population standard deviation은 0.698이므로 advantage는 `(−1.65, 0.07, 0.64, 0.93)`다. Malformed rollout은 가장 큰 negative advantage를 받고, similarity 0.2인 patch는 거의 0의 advantage를 받는다. Paper는 problem 하나당 rollout 16개를 sample했으므로 실제 group은 더 크다([[excerpts/swe-rl]] §3.1).

**설정.** Llama-3.3-70B-Instruct, step 1,600, context 16k, step당 problem 32개 × rollout 16개, global step당 Adam step 하나, H100 GPU 512개, 약 32시간을 쓴다(§3.1). Evaluation은 Agentless Mini를 사용한다. File localization, temperature 1.0에서 repair sample 500개, reranking용으로 생성한 reproduction test와 regression test를 사용한다(§3.1, App. B).

**증거.** SWE-bench Verified는 Llama3-SWE-SFT-70B의 36.2%와 비교해 41.0%였다(Table 1). Oracle file과 greedy decoding을 주면 repair 점수는 base 5.4, SFT 29.6, RL 34.8이었다(Table 2). Exact-match 0/1 reward는 29.0에 도달했고 average reward는 0 근처에 머물렀다(§3.6).

**Out-of-domain 결과([[excerpts/swe-rl]] Table 3; zero-shot greedy).** SFT baseline은 같은 seed에서 나온 synthetic editing data, Llama 3 coding, general SFT data로 훈련했고 2B token을 사용했다(App. C).

| Benchmark | Llama-3.3-70B-Instruct | SWE-SFT-70B | SWE-RL-70B |
|---|---|---|---|
| HumanEval+ | 76.2 | 73.2 | 79.9 |
| BigCodeBench-Hard (I / C) | 28.4 / 29.1 | 25.7 / 24.3 | 28.4 / 29.1 |
| CRUXEval (I / O) | 60.5 / 61.9 | 68.4 / 75.1 | 71.6 / 75.5 |
| MATH (strict / lenient) | 63.2 / 70.9 | 54.0 / 71.7 | 73.7 / 73.7 |
| MMLU | 86.49 | 85.26 | 86.82 |

다섯 category 각각 안에서 평균을 내고(strict MATH), 다시 그 다섯 평균을 내면 base 63.17, SFT 61.84, RL 68.54다(derived). SFT model은 function coding, library use, strict MATH, MMLU에서 point를 잃고 CRUXEval에서는 얻었다. 저자들은 MMLU에서 0.8 point 초과, CRUXEval에서 3점, MATH에서 3점 초과의 개선은 각각 significant하다고 보고한다(§3.5). 결과는 single study이며 arm마다 run 하나다. 저자들은 RL이 general reasoning을 획득하고 SFT는 task distribution 쪽으로 steer한다고 해석한다(Interpretation, §3.5). [[ch-38a]]는 다른 SFT-versus-RL 비교를 모은다.

**한계.** Reward는 human reference patch에 의해 제한된다. Oracle과 text가 다른 correct patch는 낮게 score된다. Training은 single-turn이고 navigation과 test execution은 evaluation scaffold에만 있다. Execution reward는 나중에 multi-turn RL에 사용되었다. DeepSWE는 Qwen3-32B를 SFT stage 없이 RL만으로, 0/1 test reward와 R2E-Gym problem 4.5K개를 써 훈련했다. Pass@1은 약 200 RL step에 걸쳐 23%에서 42%로 올랐으며(Fig. 2), final model은 Pass@1 42.2%, Pass@16 71.0%에 도달했다. SFT warm start 네 개에서 RL을 한 것은 iteration 100회 뒤에도 개선되지 않았다([[deepswe]] intro, §4, §6).

## §6 Kimi K2: post-training에서의 agentic data synthesis

**Report가 설명하는 내용([[kimi-k2]] §3.1.1; [[kimi-k2-recipe]]).**
1. Tool spec: GitHub에서 가져온 real MCP(Model Context Protocol, model에 tool을 노출하는 표준 interface) tool 3000개 이상과, hierarchical domain evolution(category → application domain → tool)으로 만든 synthetic tool 20,000개 이상.
2. Agent: 합성한 system prompt와 tool combination에서 나온 agent 수천 개.
3. Task: 각 task에는 success criterion, expected tool-use pattern, checkpoint가 있는 rubric이 있다.
4. Trajectory: LLM-generated user persona가 multi-turn dialogue를 진행한다. Tool simulator는 “maintains and updates state after each tool execution”하고, “introduces controlled stochasticity to produce varied outcomes including successes, partial failures, and edge cases”한다.
5. Filter: LLM judge는 rubric을 만족하는 trajectory만 남긴다. 저자들은 이를 large-scale rejection sampling이라고 부른다.
6. Coding과 software engineering에는 real execution sandbox를 더하고 test pass rate를 feedback으로 쓴다.

Report는 pipeline이 “tens of thousands of diverse and high-quality training examples”를 만든다고 말한다. SFT dataset size, rejection rate, hyperparameter는 보고하지 않았다([[kimi-k2-recipe]] SFT rows).

**RL stage.** Verifiable-reward task의 Gym-like set(math, STEM, logic, hack-check layer를 둔 instruction following, faithfulness, coding, Kubernetes sandbox에서 unit test를 쓰는 software engineering, safety)을 subjective task용 self-critique rubric reward, task별 token budget, PTX loss, temperature decay와 결합한다. PTX loss는 선택한 high-quality sample에 대한 auxiliary supervised language-modeling loss로서 RL objective에 더한다([[kimi-k2]] §3.2). 저자들은 forgetting과 RL task set에 대한 overfitting을 줄이기 위해 PTX loss를 더하지만 ablation은 제공하지 않는다(§3.2.3).

**결과.** SWE-bench Verified는 agentless 51.8, agentic single attempt 65.8, multiple attempt 71.6이다. Tau2-Bench는 66.1, MMLU는 89.5, IFEval은 89.8이다(Table 3). Report는 tool use를 불필요하게 켜면 일부 task에서 performance가 낮아질 수 있다고 말한다(§5).

**Report에 없는 내용.** Agentic pre-training corpus, pre-training에서 agentic token share, planner/executor/critic generator는 없다([[kimi-k2]] Verification). Post-training 이전 agentic data의 증거는 [[ch-32d]]에 있다.

## Negative sample과 negative feedback

이 section은 course standard에서 말한 “negative”의 네 가지 의미를 agent trajectory에 적용한다. Derivation은 [[ch-31a]] (supervised)와 [[ch-43a]] (gradient)에 있다.

**Negative의 출처.** Failed trajectory는 environment의 final reward(AgentTuning `r < 1`, ETO reward `[0, 1]`), hidden/generated test(SWE-Gym, DeepSWE), oracle patch와의 similarity(SWE-RL), rubric에 대조한 LLM judge(K2)로 label한다. Issue text만으로 task를 풀 수 없거나 test가 틀리면 false negative가 생긴다. SWE-rebench는 validation에 실패한 task를 제거하고 flaky test를 버리기 위해 test를 다시 실행하지만 false-negative rate는 보고하지 않는다([[swe-rebench]] §2.3, Findings).

**현재 관행이 type별로 하는 일.**

| Type | Source와 사용 | 측정한 효과 |
|---|---|---|
| 1. Negative marginal value(discard) | AgentTuning, FireAct, AutoAct, SWE-Gym, K2는 failure를 버린다 | AgentTuning 7B unfiltered data: held-out 0.47 vs filtered 0.65(Table 2); AutoAct unfiltered 32.51 vs no fine-tuning 32.84(Table 2) |
| 2. Content로서의 negative | Agent-FLAN: tool을 유도하는 prompt에 correct plain-text target을 붙임; AgentInstruct: 없는 API에 대한 FAILED target; SWE-Gym/R2E-Gym verifier: target이 `"NO"`인 failed trajectory | Agent-FLAN 7B: HScore 84.5 → 89.1, T-Eval 66.3 → 66.0([[agent-flan]] Table 3) |
| 3. Conditioning으로서의 negative | NAT: suffix “Please generate a solution that **incorrectly** answers the question” 뒤에 failed trajectory를 훈련하고, inference에서는 “correctly” prompt를 씀 | Llama-2-7B, positive 2k: math average 55.90(positive only) → 63.39(label 없는 negative) → 64.64(NAT); 13B, positive 5k: 70.76 → 69.10 → 71.28; low-quality negative: 2k에서 −3.16; random-string label 64.04 vs Correct/Incorrect 63.55([[excerpts/learning-from-failure-nat]] Tables 2, 5, 7) |
| 4. Gradient로서의 negative | ETO: 더 낮은 reward의 agent trajectory를 rejected로 한 DPO([[ch-39]]); SWE-RL과 DeepSWE: negative group advantage | ETO vs SFT, Llama-2-7B: WebShop 63.1 → 67.4, ScienceWorld unseen 53.0 → 65.0([[eto-trial-and-error]] Table 2) |

**Type 4의 mechanism.** Action에 대한 softmax에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이며, 여기서 `z_j`는 action `j`의 logit, `p_j`는 그 probability다. 따라서 `log p_y`를 낮추면 현재 probability에 비례하여 다른 모든 logit이 올라간다. 계산 예를 보자. `p = (0.6, 0.3, 0.1)`에서 action 3을 내린다. `log p₃`의 gradient는 `(−0.6, −0.3, 0.9)`이고, 반대 방향으로 size 1의 step 하나를 적용하면 `p = (0.710, 0.263, 0.026)`이 된다. Ratio `p₁/p₂`는 2.00에서 2.70으로 오르므로, 제거한 mass는 이미 가장 likely했던 action에 주로 간다. 내린 trajectory가 이미 unlikely했을 때 이는 probability를 concentrate하고 diversity를 낮춘다([[ch-43a]]).

ETO의 trajectory-level DPO loss는 `−log σ(β log[π_θ(e_w|u)/π_ref(e_w|u)] − β log[π_θ(e_l|u)/π_ref(e_l|u)])`다(Eq. 11). `u`는 task, `e_w`와 `e_l`은 reward가 더 높고 더 낮은 trajectory, `π_ref`는 iteration 시작 시점의 policy, `β`는 constraint weight, `σ`는 sigmoid다. 계산 예: `β = 0.1`, chosen log-ratio `+2.0`, rejected log-ratio `−3.0`이면 margin은 `0.5`, loss는 `0.474`다.

**효과 크기.** 어떤 agent source도 rejected term 때문에 생긴 improvement의 share를 분리하지 않았다. ETO의 RFT와의 비교는 loss와 chosen sample도 함께 바꾼다([[eto-trial-and-error]] Findings). NAT는 positive-only baseline과 비교하여 negative를 측정했고, positive가 늘수록 gain이 줄며 low-quality negative에서는 효과가 반대가 됨을 보였다(Table 5, Fig. 3).

**Failure mode와 control.**

- **Localize:** Final reward로 만든 ETO의 step-level pair는 LR `1e-6`, `β 0.1`에서 WebShop을 8.3으로 떨어뜨렸다(Table 4). Limitation은 첫 wrong action을 식별하지 못한다고 적는다.
- **Iteration을 제한:** ETO는 WebShop과 ScienceWorld에서 첫 두 iteration에 개선됐고 세 번째 뒤에는 하락했다. ALFWorld에서는 첫 iteration만 개선됐다(§4.3).
- **Behavior cloning으로 anchor:** SFT가 없는 ETO는 WebShop 12.5로, untuned 17.9보다 낮았다(Table 5).
- **불확실한 failure를 mask:** DeepSWE는 context limit, step limit, 20-minute generation timeout에 도달한 trajectory를 0점으로 score하지 않고 mask한다(§2.3).
- **Recovery turn을 훈련할 때 context에 남겨 둔 failed action을 mask:** (§1).

**Diagnostic.** Chosen/rejected log-probability를 따로 log한다(ETO는 이를 보고하지 않는다). Advantage sign별로 training statistic을 나누고 entropy를 track하며, 큰 `k`에서 pass@k와 pass@1을 함께 보고한다(SWE-Gym 32B: pass@16 42.8% vs Best@16 32.0%).

**Generality에 미치는 효과.** Prompt나 training data가 stopping에 대해 무엇을 signal하는지에 따라 abstention/refusal rate가 달라진다. WebArena에서는 unachievable task에서 stopping을 허용하는 prompt hint가 GPT-4로 하여금 feasible task의 54.9%를 impossible이라고 부르게 했다([[excerpts/webarena-data]] §5.1). 반대 방향에서는 benign web-navigation demonstration으로 SFT한 Llama-3.1-8B-Instruct의 harmful-task attack success가 32.88%에서 64.38%로 올랐고 refusal은 26.03%에서 6.85%로 낮아졌다([[agentic-finetuning-misalignment]] Table 1).

## 레시피

| Model(정확한 release) | 크기 | Stage | 설정 | 값 | Source 위치 | 상태 | 이 값의 증거 |
|---|---|---|---|---|---|---|---|
| AgentLM-7B / 13B / 70B | 7B–70B | SFT | Agent sampling ratio `η`(ShareGPT `1 − η`) | 0.2 | arXiv:2310.12823v2 §2.2.2, App. A | verified 2026-09-14 ([[agenttuning]]) | 7B held-out에서 `η`를 0–1, 0.1 step으로 scan; endpoint는 Table 5에 있음 |
| AgentLM-7B / 13B / 70B | 7B–70B | SFT | Trajectory; keep rule | 1,866; `r = 1`(Mind2Web `r ≥ 2/3`) | Table 1; §2.1.3 | verified 2026-09-14 | Table 2(7B): filtered 1.96 / 0.65 vs unfiltered 1.34 / 0.47 |
| AgentLM-7B / 13B | 7B, 13B | SFT | Peak LR | 5e-5 | §2.2.3; App. A Table 6 | verified 2026-09-14 | ablation 보고 없음 |
| AgentLM-70B | 70B | SFT | Peak LR | 1e-5 | §2.2.3; App. A Table 6 | verified 2026-09-14 | ablation 보고 없음 |
| AgentLM-7B / 13B / 70B | 7B–70B | SFT | Loss masking; batch(unit 미명시); sequence length | model output만; 64; 4,096 | §2.2.3; Table 6 | verified 2026-09-14 | ablation 보고 없음 |
| Agent-FLAN (Llama2) | 7B–70B | SFT | ShareGPT : agent mix(unit 미명시) | 1:1 | arXiv:2403.12881v1 App. A | verified 2026-09-14 ([[agent-flan]]) | ablation 보고 없음 |
| Agent-FLAN (Llama2) | 7B–70B | SFT | Capability weight reasoning : retrieval : understanding | 1 : 0.25 : 0.75 | App. A | verified 2026-09-14 | Table 2 half-data ablation(7B) |
| Agent-FLAN (Llama2) | 7B–70B | SFT | Agent sample; negative-sample source; epoch; peak LR | 24,703; ToolBench query 761개; 1; 2e-5 | App. B Table 6; App. C; App. A Table 5 | verified 2026-09-14 | Table 3: negative 포함 HScore 89.1 vs 미포함 84.5 |
| FireAct Llama-2 / CodeLlama | 7B–34B | distill-SFT | Data; method; epoch; LR | GPT-4 ReAct trajectory 500개; LoRA int8; 30; 3e-4 | arXiv:2310.05915v1 §5, App. B.2 | verified 2026-09-14 ([[fireact]]) | Fig. 3 data size 100–1,000; LoRA vs full 26.2 vs 30.2(Table 11) |
| Lumos (Llama-2) | 7B, 13B | SFT | Planning; grounding annotation | 55,382; 55,499 | arXiv:2311.05657v3 §4.1 | verified 2026-09-15 (primary text) | Table 3: modular vs integrated formulation |
| SWE-Gym agent (Qwen-2.5-Coder-Instruct) | 7B, 14B, 32B | distill-SFT | Trajectory; teacher | successful 491개; gpt-4o-2024-08-06, claude-3-5-sonnet-20241022 | arXiv:2412.21139 §4.2 | verified 2026-09-15 (primary text) | Fig. 5: trajectory 491개까지의 gain |
| SWE-Gym agent | 7B, 14B, 32B | distill-SFT | LR; max epoch; global batch; max context | 1e-4; 5; 8; 32,768 | App. B.2 | verified 2026-09-15 (primary text) | ablation 보고 없음 |
| SWE-agent-LM-32B (Qwen2.5-Coder-Instruct) | 32B | distill-SFT | Trajectory; teacher; LR; max epoch; max context | 5,016; Claude 3.7 Sonnet; 5e-5; 3; 32,768 | arXiv:2504.21798v2 §4, App. F.1 | verified 2026-09-15 (primary text) | §4.1 Fig. 5: trajectory 700개에서 repository 4 → 100 |
| Llama3-SWE-RL-70B | 70B | RL | Base; seed; step; context; step당 prompt × rollout | Llama-3.3-70B-Instruct; 273k; 1,600; 16k; 32 × 16 | arXiv:2502.18449v2 §2, §3.1 | verified 2026-09-15 (primary text) | §3.6: continuous vs exact-match reward(34.8 vs 29.0 repair) |
| Llama3-SWE-RL-70B | 70B | RL | LR; KL `β`; clip `ε` | 보고하지 않음 | checked §2–3, App. A–D | 보고하지 않음 | — |
| Llama3-SWE-RL-70B | 70B | RL | Compute | H100 GPU 512개, wall-clock 약 32시간 | §3.1 | verified 2026-09-15 (primary text) | 해당 없음 |
| Llama3-SWE-SFT-70B | 70B | SFT | Data; token; context | synthetic editing + Llama 3 coding 및 general SFT; 2B; 16k | App. C | verified 2026-09-15 (primary text) | Table 3 out-of-domain 비교 |
| DeepSWE-Preview (Qwen3-32B) | 32B | RL | Stage 위치; reward; problem; compute | RL만, SFT 없음; 선택한 test 통과 시 1, 그 외 0; R2E-Gym 4.5K; H100 64개 × 6일 | Together AI blog 2025-07 intro, §2.1–2.2 | verified 2026-09-14 ([[deepswe]]) | §6: SFT warm start 뒤 RL은 개선되지 않음 |
| ETO (Llama-2-7B-Chat) | 7B | preference | Loss; `β`; LR; iteration | trajectory-level DPO; 0.1(WebShop, SciWorld), 0.5(ALFWorld); 1e-6; 2 / 1 | arXiv:2403.02502v2 §4.1 | verified 2026-09-14 ([[eto-trial-and-error]]) | §4.3 Fig. 4: iteration 3 뒤 하락 |
| NAT (Llama-2-Chat) | 7B, 13B | SFT | Epoch; batch; peak LR; schedule | 2; 64; 5e-5; cosine, warmup 3% | arXiv:2402.11651v2 §4.1 | verified 2026-09-15 (primary text) | ablation 보고 없음 |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT | Tool repository; example | MCP 3000개 이상 + synthetic tool 20,000개 초과; “tens of thousands” | arXiv:2507.20534 §3.1.1 | verified 2026-09-14 ([[kimi-k2-recipe]]) | Fig. 9 coverage만; ablation 없음 |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT | Dataset size, epoch, LR, masking | 보고하지 않음 | §3.1; body와 appendix | 보고하지 않음 | — |

**작은 범용 run의 시작점.** Filter한 agent trajectory 수천 개를 7B–70B chat model에 더할 때는, AgentTuning이 Llama-2-chat과 ShareGPT에 사용한 대로 `η = 0.2`, model output에만 loss를 계산하는 방식, `r = 1` filter에서 시작하라. 그 뒤 held-out environment에서 `η`를 다시 scan하라. 7B와 13B mixture는 held-out task에서 general-only data를 넘지 못했기 때문이다. SWE trajectory 수백 개에서 수천 개로 7B–32B coder를 fine-tune할 때는, LR `1e-4`, 최대 5 epoch, global batch 8, context 32,768 token의 full fine-tuning이 SWE-Gym 설정과 일치한다. SWE-smith는 trajectory 5,016개에 LR `5e-5`, 최대 3 epoch를 썼다.

## 일반화 관점

**(a) 폭을 넓히는 것.**

- General data와 agent data를 섞기: 7B에서 agent-only의 held-out 0.09 대비 0.67([[agenttuning]] Table 5).
- Trajectory 수를 고정한 채 training repository를 늘리기: 10.3% → 15.1%([[excerpts/swe-smith]] Fig. 5).
- 같은 seed에서 SFT 대신 RL 하기: out-of-domain benchmark의 category mean 68.54 vs 61.84([[excerpts/swe-rl]] Table 3; derived).
- Synthesis에서 tool과 agent diversity: K2의 real tool 3000개 이상과 synthetic tool 20,000개 이상, rubric task. Diversity만 분리한 ablation은 없다([[kimi-k2]] §3.1.1; Interpretation).
- Task type 전반에 걸친 unified training: Lumos-I-All-13B는 unseen WebShop에서 single-type variant보다 높은 점수를 냈다(50.3 vs 43.8–47.3)([[excerpts/lumos]] Table 1e).

**(b) Narrowing 또는 forgetting을 일으키는 것.**

- Agent-only SFT: 7B–13B에서 held-out 0.09–0.10([[agenttuning]] Table 5).
- SWE data SFT: HumanEval+ 76.2 → 73.2, strict MATH 63.2 → 54.0([[excerpts/swe-rl]] Table 3).
- Repository specialization: 32B에서 instance 22개인 SymPy subset에 +9.1 point, Verified 나머지에는 −1.9([[excerpts/swe-smith]] Fig. 4).
- Single-task QA fine-tuning: Llama-2-7B의 StrategyQA 59.0 → 52.0([[fireact]] Table 14).
- Benign agentic SFT에서의 safety forgetting: refusal 26.03% → 6.85%([[agentic-finetuning-misalignment]] Table 1).
- 더 강한 teacher 없이 self-generated trajectory 사용: SWE-Gym Lite 15.3% → 8.7%([[excerpts/swe-gym]] §4.2).

**(c) 이 단계에서 이를 측정하는 방법.**

- 모든 data mixture에 대해 held-in, held-out, general score를 분리해 보고한다([[agenttuning]] Table 3).
- 큰 `k`에서 pass@1과 pass@k를 보고하고, verifier가 있다면 Best@k도 분리해 보고한다([[excerpts/swe-gym]] §5.1).
- SWE-bench Verified 옆에 date-filtered benchmark 또는 outside-repository benchmark를 둔다([[swe-rebench]]; [[swe-bench-illusion]]).
- Behavior metric으로 stuck-in-loop rate와 empty-patch rate를 쓴다([[excerpts/swe-gym]] Table 3).
- Agent SFT 전후에 harmful-task attack success, refusal, benign task에서의 over-refusal을 측정한다([[agentic-finetuning-misalignment]] App. F).
- 알려진 measurement error: AgentTuning normalized score는 평가한 model 집합에 의존한다(§3.1). WebArena success는 prompt hint에 의존한다(Table 2). 같은 model도 scaffold 선택이 resolve rate를 바꾸며, 예를 들어 Lite에서 7B zero-shot은 MoatlessTools와 OpenHands로 각각 7% vs 1%다([[excerpts/swe-gym]] §4.3).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| General data 없이 agent trajectory로 훈련 | Held-in agent score는 오르고 held-out 및 MMLU/GSM8K는 하락 | Held-out environment에서 AgentTuning의 세 arm 비교(agent-only, general-only, mixed)를 실행 |
| Observation token에 loss 계산 | 긴 tool output이 loss를 지배하고 model이 shell/page text를 모방 | Trajectory별 `mₜ = 1`인 token 수를 센다. Model turn만 남아야 함 |
| Filter하지 않았거나 품질 낮은 self-generated trajectory 유지 | Fine-tuned score가 untrained model보다 낮음 | AutoAct Table 2처럼 no fine-tuning과 비교 |
| SWE-bench Verified만으로 checkpoint 선택 | Fresh set 또는 outside-repository set과 큰 gap | SWE-rebench-style date-filtered task와 outside repository에서 평가 |
| Best@k 또는 pass@k를 pass@1로 보고 | Sample 하나로 leaderboard number를 재현할 수 없음 | 모든 number에 `k`, verifier, scaffold를 보고 |
| RL에서 truncated/timed-out trajectory를 failure로 score | Reward collapse, long task에 penalty | Context, step, time limit으로 끝난 trajectory를 mask(DeepSWE compact filtering) |
| Agent-FLAN-style negative를 negative gradient로 취급 | Wrong call의 likelihood가 직접 떨어지리라 기대 | Loss를 검사한다. Cross-entropy 아래 correct text target은 type 2 |
| Agent SFT 뒤 chat에서만 safety 평가 | Agent context의 refusal 하락을 알아차리지 못함 | Agent domain(WebDojo-style)에서 attack success와 refusal을 측정 |
| K2를 agentic pre-training의 증거로 인용 | 출처 위치 없는 주장 | [[kimi-k2]] §2.2와 §3.1.1을 확인하고 [[ch-32d]] source를 사용 |

## 이해를 확인해 보세요

1. 70B에서 AgentTuning mixture는 held-out task에서 general-only data를 넘지만, 7B와 13B에서는 넘지 못한다. 이 pattern을 낳을 수 있는 설명 두 개를 들고, 둘을 구별하는 experiment를 설명하라.
2. §1 example의 token count를 이용하여, mask하지 않은 observation token이 agent SFT run이 최적화하는 대상을 왜 바꾸는지 설명하라.
3. SWE-RL worked group에서 malformed rollout은 standard deviation을 지배한다. 이 사실은 네 rollout이 모두 well formed인 group과 비교했을 때 best patch의 advantage를 어떻게 바꾸는가?
4. SWE-SFT baseline과 SWE-RL은 같은 PR seed를 쓰는데도 SFT는 strict MATH를 9.2 point 낮추고 RL은 10.5 point 올린다. “SFT versus RL” 외에 이를 낳을 수 있는 두 training setup의 차이를 나열하라.
5. SWE-Gym self-improvement run은 Lite를 15.3%에서 8.7%로 낮췄고, DeepSWE RL-only run은 23%에서 42%로 올렸다. 두 run에서 failed trajectory와 on-policy sample을 사용하는 방식은 무엇이 다른가?
6. Unlikely action을 내리면 왜 probability가 가장 likely한 action에 concentrate되는가? Agent에서 큰 `k`의 pass@k에는 이것이 어떻게 나타나는가?
7. K2 tool simulator는 controlled stochasticity를 도입한다. Deterministic simulator라면 어떤 narrowing을 예상하며, 이 chapter의 어느 measurement가 이를 탐지할 수 있는가?
8. SymPy repository specialization은 SymPy에서 9.1 point를 얻고 다른 곳에서 1.9를 잃었다. 어느 deployment condition에서 이 trade가 허용 가능한가? 받아들이기 전에 어떤 evaluation을 실행하겠는가?

## 연결

- 이전: [[ch-26]] Tool and Function-Calling Data — single-call/multi-turn tool data, abstention target, unseen-tool split.
- 다음: [[ch-28]] Long-Context Data Synthesis and Synthetic Evaluation Task Families.
- 관련:
  - [[ch-24]] Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data
  - [[ch-29c]] Agentic Environment and Task Synthesis at Scale
  - [[ch-29d]] User Simulators, Trajectory Verification, and Failed Trajectories
  - [[ch-30b]] Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares
  - [[ch-31a]] Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood
  - [[ch-32d]] Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training
  - [[ch-38a]] SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity
  - [[ch-39]] Offline Preference Optimization: DPO and Its Variants
  - [[ch-40]] Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO
  - [[ch-43a]] Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages
  - [[ch-45b]] Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability
  - [[ch-48]] Contamination Detection and Its Effect on Reported Scores
  - [[ch-51a]] Evaluating Agent Generality and Reliability

## 출처

- [[agenttuning]] — trajectory filtering count, mixture objective와 `η = 0.2`, held-in/held-out/general Table 5, hyperparameter.
- [[fireact]] — method-mix result, Bamboogle/StrategyQA transfer, LoRA setting.
- [[excerpts/lumos]] — planning/grounding formulation, annotation count, unseen-task result(chapter excerpt는 arXiv v3와 대조해 verified했으며 library card는 verification보다 앞섬).
- [[autoact]] — self-synthesized trajectory, `reward = 1` filter, unfiltered-data ablation.
- [[agent-flan]] — chat-format conversion, capability weight, 1:1 mix, negative sample, Agent-H.
- [[agentinstruct]] — trajectory data가 아닌 instruction data인 이유.
- [[excerpts/webarena-data]] — site, observation/action space, functional check, GPT-4/human success rate(chapter excerpt는 arXiv와 대조해 verified했으며 library card는 verification보다 앞섬).
- [[excerpts/swe-gym]] — environment count, teacher trajectory, Table 3, verifier Best@k/pass@k, self-improvement result(chapter excerpt는 arXiv와 대조해 verified).
- [[excerpts/swe-smith]] — test를 깨뜨리는 task synthesis, repository-count scaling, specialization trade-off(chapter excerpt는 arXiv v2에서 가져옴).
- [[r2e-gym]] — commit-based environment, synthetic vs real issue text, 20K SFT context.
- [[deepswe]] — RL-only execution-reward training, compact filtering, SFT warm-start result.
- [[excerpts/swe-rl]] — data curation, reward equation, setting, Tables 1–3, reward ablation(chapter excerpt는 arXiv v2와 대조해 verified).
- [[kimi-k2]] — tool-use data synthesis pipeline, RL gym/rubric critic, result, report에 없는 내용.
- [[kimi-k2-recipe]] — K2 SFT/RL ledger row.
- [[eto-trial-and-error]] — failure에 대한 trajectory-level DPO, iteration/granularity ablation.
- [[excerpts/learning-from-failure-nat]] — failure-conditioned SFT와 positive count/negative quality에 대한 의존성(chapter excerpt는 arXiv v2에서 가져옴).
- [[swe-rebench]] — date-filtered SWE task와 Verified-vs-fresh gap.
- [[swe-bench-illusion]] — SWE-bench Verified의 file-path memorization probe.
- [[agentic-finetuning-misalignment]] — benign agentic SFT로 인한 safety forgetting.
