<!-- chapter: ch-27
     track: synthetic
     title: Modality — Agentic Trajectories
     sources: [[agentinstruct]], [[agenttuning]], [[lumos]], [[fireact]], [[autoact]], [[agent-flan]], [[webarena-data]], [[swe-gym]], [[swe-rl]], [[openhands-data]], [[kimi-k2-agentic-data]], [[kimi-k2]], [[terminal-bench-trajectories]], [[explorer]]
     figures: figures/action-space.html
-->

# 27장 — 모달리티: Agentic Trajectories

> **핵심 통찰.** "Agent trajectory"는 긴 chat이 아니다. 각 action이 다음 observation에 반영되어야 하는 *environmental side-effects*를 가진 `(observation, thought, action)` tuple의 sequence다. 이 때문에 data의 의미가 달라진다. Dataset은 `(prompt, response)` 쌍이 아니라 *executable world에 조건화된* 것이다. Agentic post-training의 모든 설계 선택은 이 구조적 사실 하나에서 나온다. world를 어떻게 observe하는가, 어떤 action이 legal한가, world가 final state를 어떻게 grade하는가.
>
> **가이드라인.** Environment를 먼저 고르고, teacher는 그다음, model은 세 번째로 고르라. Action space가 trajectory format을 정의하고, trajectory format이 data pipeline을 정의한다. 그 뒤에야 teacher-model과 student-model 선택이 중요해진다. Data stage에서 modality를 mixing하는 것(AgentInstruct의 multi-skill pipelines, Kimi-K2의 pretraining-mix injection)은 작동한다. Model stage에서 modality를 mixing하는 것(모든 것에 하나의 monolithic agent LoRA)은 작동하지 않는다.

---

## 이 장이 필요한 이유

[[ch-26]]까지는 single autoregressive context 안에서 reasoning과 tool-call data를 합성하는 법을 알게 되었다. Agent는 다음 개념 단계다. 모델은 더 이상 self-contained response를 내지 않는다. 외부 world, 즉 shell, browser, Python kernel, Git repo가 action을 읽고 state를 mutate하고 observation으로 답하는 loop 안에서 *한 턴*을 생성한다. Loop는 수십 턴까지 갈 수 있다. Trajectory는 흔히 15K–100K tokens에 닿는다([[openhands-data]], [[swe-gym]]). 중요한 reward는 어떤 single token이 fluent했는지가 아니라 *final world-state가 predicate를 만족했는지*다.

이 장은 그 loop의 data side에 대한 design-space map이다. 여섯 lineage가 SFT corner(AgentInstruct, AgentTuning, Lumos, FireAct, AutoAct, Agent-FLAN)를 덮고, 두 benchmark가 environment corner(WebArena, SWE-Gym)를 덮는다. 하나의 frontier RL recipe는 rule-based reward가 어떻게 scale되는지 보여 준다(SWE-RL). 하나의 frontier model report는 1T-class lab이 실제로 end-to-end에서 무엇을 하는지 보여 준다(Kimi-K2). 이들을 묶는 thread는 environment × action space × observation format × success signal이라는 four-axis taxonomy다. 앞으로 agent paper를 읽을 때 이 taxonomy를 쓰게 될 것이다.

---

## 1. 여섯 SFT recipe — design-space tour

여섯 논문은 모두 2023년 10월부터 2024년 4월 사이에 나왔다. 각각 다른 axis를 vary했다.

| Paper | Year | Trajectories | Teacher | Core design move |
|---|---|---|---|---|
| [[agenttuning]] | 2023.10 | 1,866 | GPT-4 | **Mixing ratio**: 1:10 agent:ShareGPT preserves chat quality |
| [[fireact]] | 2023.10 | ~2,000 | GPT-4 | **Method diversity**: CoT + ReAct + Reflexion in one corpus |
| [[lumos]] | 2023.11 | 40K tasks → 200K triples | GPT-4 | **Module decomposition**: Plan / Ground / Execute separable heads |
| [[autoact]] | 2024.01 | ~10K | *none* (self) | **Self-differentiation**: one base model plays Plan/Tool/Reflect |
| [[agent-flan]] | 2024.03 | ~85K | GPT-4 | **Negative examples**: four hallucination modes explicitly corrected |
| [[agentinstruct]] | 2024.07 | 25M | GPT-4 | **Agentic pipeline**: 43-generator flows per skill |

이들을 대화로 읽어라. AgentTuning은 small curated corpus가 올바르게 mix되면 작동한다는 점을 세운다. FireAct는 trajectory를 *어떻게* 수집하느냐가 중요하다고 덧붙인다. 하나의 prompting method보다 세 가지가 낫다. Lumos는 trajectories가 *internal structure*(plan / ground / execute)를 가지며, 그 구조가 monolithic ReAct blob 안에 숨지 말고 training data에 드러나야 한다고 덧붙인다. AutoAct는 GPT-4 teacher가 꼭 필요한지 시험한다. 답은 "좁은 QA domain에서는 아니다"다. Agent-FLAN은 앞의 네 논문이 다루지 않은 불만에 답한다. *trained model이 tool이 필요 없는 prompt에서 tool call을 hallucinate하면?* 네 가지 explicit negative-example classes(format / action / parameter / relevance hallucination)를 도입한다. 마지막으로 AgentInstruct는 *generation pipeline 자체*를 multi-agent system으로 만들어 이 아이디어를 25M pairs로 scale한다.

Corpus-size column은 중요한 사실을 숨긴다. AgentTuning과 FireAct는 각각 약 2K trajectories다. Agent-FLAN은 85K, AgentInstruct는 25M이다. 네 논문 사이에 세 자릿수 규모 차이가 있다. Small-corpus paper의 quality claim은 **structure의 diversity**(method, environment, decomposition)가 volume을 보상한다는 것이다. AgentInstruct의 scaling claim은 pipeline이 갖춰지면 고갈될 때까지 돌리는 편이 낫다는 것이다. 두 claim은 각자의 benchmark suite에서 경험적으로 참이다. Open question은 small-diverse corpora가 held-out tasks에서 large-pipeline corpora보다 낮은 ceiling에 부딪히는가다. Agent-FLAN의 ablation table은 답을 암시한다. 단일 capability type(instruction-follow, agent-reason, generalization)을 제거하면 AgentBench가 0.3–0.5 points 손실되고, negatives 제거는 hallucination rate를 세 배로 만든다. Structure는 모든 scale에서 중요하지만, 85K+ scale에서는 *ablation delta*가 줄어든다. 그래서 AgentInstruct가 decomposition 논쟁을 건너뛰고 scale만 하는 것이다.

### 1.1 AgentInstruct의 six-flow taxonomy — pipeline-of-specialists template

[[agentinstruct]](Mitra et al. 2024, Microsoft)는 이 목록에서 가장 야심찬 단일 논문이다. 중심 추상화는 **agentic flow**다. 각 stage가 자신의 prompt, tool access, output schema를 가진 specialized LLM agents의 pipeline이다. 모든 skill에 대한 네 가지 generic stage:

1. **Content Transformation** — 한 agent가 raw input(web document, codebase, API spec)을 canonical intermediate structure(passage + candidate-questions list, function + test stub, schema + example call)로 rewrite한다.
2. **Seed Instruction Generation** — 10–40개의 *parallel* "generator" agents가 각각 distinct sub-skill(literal question / inferential question / multi-hop / numerical reasoning / …)을 만들도록 prompt된다. Reading-comprehension skill 하나만 해도 question category별 하나씩 **43 generator agents**를 쓴다.
3. **Instruction Refinement** — "suggester" agent가 improvement를 제안하고, "editor" agent가 적용한다. Instruction당 최대 3 iterations. 이것이 structural coverage를 잃지 않고 *phrasing* diversity를 얻는 방식이다.
4. **Answer Generation + Validation** — GPT-4가 gold answer를 만들고, LLM-judge filter가 low-quality pairs를 drop한다.

17 skills(reading-comprehension, math, code, tool-use, RAG, creative-content, web-agent, long-context, …) 전체를 합쳐 proprietary **AgentInstruct-25M** corpus를 만들었다. Orca-3(Mistral-7B base + AgentInstruct SFT)는 Mistral-7B-Instruct보다 AGIEval에서 40%, GSM8K에서 54%, MATH에서 3× 앞섰다.

자체 pipeline에 대한 takeaway 두 가지. 첫째, **generator count는 diversity knob**이지 budget waste가 아니다. 43개의 narrow-prompt agents가 하나의 broad-prompt agent를 43번 sample하는 것보다 더 넓은 sub-skill distribution을 덮는다. 둘째, **refinement loop는 선택 사항이 아니다.** Single-shot GPT-4 generation은 금방 plateau에 도달한다. Iterative suggester/editor는 약 1.5× 비용으로 hard sub-skills에서 약 3 points를 더한다.

눈여겨볼 skill-specific variant도 있다. **Tool-use flow**는 real API docs에서 seed를 가져오고(synthetic schema가 아님), generator agents가 다양한 tool-count complexity(1 tool → 2 tools → composed chains)로 query를 synthesize하게 하며, refinement를 schema-correctness check로 route한다. **RAG flow**는 content agents로 passage cluster를 만들고, query agents가 passage 간 evidence fusion이 필요한 question을 만든다. **Long-context flow**는 generator prompt가 실행되기 전에 documents를 8K+ tokens까지 stitch한다. 이 세 경우 모두 pattern은 같다. 하나의 *upstream* agent가 substrate를 준비하고, 그 뒤 *many* downstream agents가 varied difficulty로 sample한다. 이것은 self-instruct lineage의 "one prompt, many samples" 접근과 반대다. AgentInstruct의 bet은 **substrate diversity + prompt specialization**이 **one diverse prompt**를 이긴다는 것이다.

### 1.2 Lumos의 Plan/Ground/Execute format spec

[[lumos]]는 새 agent-trajectory format을 설계할 때 복사할 논문이다. 모든 trajectory는 세 aligned supervised targets로 분해된다.

```
Plan:    (task, gold_answer) → list[subtask]
Ground:  (subtask, env_state) → action in unified grammar
Execute: action → observation (from real env or tool)
```

Unified action grammar는 명시적이다. `Search[query]`, `Retrieve[doc_id]`, `Calculate[expr]`, `Click[element]`, `Type[element, text]`, `Back`, `Finish[answer]`. 이것이 Lumos가 학습하는 grammar다. 기존 dataset(HotpotQA, ALFWorld, WebShop, Mind2Web, Musique, GSM8K, MATH, StrategyQA, ScienceQA)에서의 conversion은 GPT-4를 *annotator*로 사용한다. Raw trajectory를 받아 three-layer decomposition을 내는 prompt다.

두 training mode가 나온다. **Lumos-I(iterative)**는 observation마다 replan한다. **Lumos-O(onetime)**는 전체 task를 upfront로 plan하고 순차적으로 execute한다. 각 module은 자체 LoRA나 자체 head가 될 수 있다. Held-out environment generalization에서 modular decomposition은 monolithic ReAct fine-tune이 약 20 points 손실하는 데 비해 약 8 points만 손실한다. 이유는 *action grammar*가 concrete tool이 달라도 environments 사이에서 공유되기 때문이다.

Downstream 계획이 "retriever / browser / code executor를 재훈련 없이 swap"하는 것이라면 Lumos format을 사용하라. Downstream 계획이 "one giant SFT blob"이라면 AgentInstruct format을 사용하라. 둘은 경쟁자가 아니라 서로 다른 API-stability 선택이다.

Three-module decomposition은 monolithic ReAct trace보다 module별로 더 깨끗한 supervision signal도 만든다. Planning module은 clear structural target을 가진 `(task → list[subtask])` pair를 본다. Grounding은 unified grammar로 제약된 action을 내는 `(subtask + env_state → action)`을 본다. Execute는 순수 environment interaction이다. 7B model은 각 module이 unified ReAct agent보다 더 좁은 output distribution을 갖기 때문에 효과적으로 specialize할 수 있다. 이것은 AutoAct가 재사용하는 insight다([[autoact]]는 Plan/Tool/Reflect로 split). Kimi-K2의 sub-agent orchestration(planner / executor / critic)에서도 반복된다. **Role specialization은 2023→2025 문헌 전반의 지속적 design pattern**이다. Final-deployment model이 monolithic이더라도 toolbox에 넣어 두라. *Data*는 synthesis 동안 role-partitioned될 수 있기 때문이다.

### 1.3 Agent-FLAN의 네 hallucination class — negative-example ontology

[[agent-flan]]은 SFT-trained agent가 chat에서 tool을 과도하게 call한다면 다시 읽어야 할 논문이다. Hallucinated tool call을 네 가지 distinct failure mode로 분류한다.

| Mode | Trigger | Gold response |
|---|---|---|
| **Format** hallucination | Model emits malformed tool-call JSON | Corrected call *or* a text refusal |
| **Action** hallucination | User query doesn't need a tool | Text-only answer, no call |
| **Parameter** hallucination | Right tool, wrong args | Tool call with correct args |
| **Relevance** hallucination | Tool list doesn't contain a needed tool | "I cannot help with this tool set" refusal |

각 class는 base model error에서 뽑은 common failure pattern으로 GPT-4를 prompt해 만든 자체 synthetic-negative-example pool을 가진다. Agent-FLAN-7B는 AgentBench held-out에서 AgentTuning baseline 대비 hallucinated tool call을 5× 줄인다. 교훈: **negatives 없는 agent SFT는 open-loop controller**다. 네 negative class를 추가하는 것이 closed-loop correction이다.

Agent-FLAN의 두 번째 기여는 놓치기 쉽다. **Format alignment**다. 논문은 agent trajectory를 rewrite해 Llama-2 pretraining에 나오지 않는 special tokens와 delimiter를 피한다. 예를 들어 custom `<tool>`/`</tool>` pair를 plain markdown-code-fenced JSON block으로 바꾼다. Training distribution을 pretraining distribution에 가깝게 유지하면 catastrophic forgetting이 줄고 preserved-chat-quality metrics가 좋아진다(MT-Bench는 base Llama-2-Chat에서 0.5 points 이내). 2024 논문들이 점점 수렴한 detail이다. Agent formatting을 위해 새 token을 만들지 말라. 단, 그 token으로 pretrain할 계획이라면 예외다([[kimi-k2-agentic-data]]는 그렇게 한다. 대부분 SFT-only paper는 그래서는 안 된다).

### 1.4 FireAct — method diversity beats method depth

[[fireact]](Chen et al. 2023, Princeton + Cambridge)는 scale 측면에서는 AgentInstruct의 반대편에 있지만 orthogonal claim을 한다. 같은 약 2K HotpotQA + Bamboogle question pool에 대해 **세 가지** prompting method로 trajectory를 병렬 수집한다. Chain-of-Thought(GPT-4 reasoning only), ReAct(GPT-4 with Wikipedia search in `Thought/Action/Observation` loop), Reflexion(GPT-4 attempts, reflects on failure, retries up to N=3). 각 question은 method당 trajectory 하나를 갖고, 각 trajectory는 method name으로 labeled되어 student가 method-specific formatting을 학습할 수 있다.

Ablation이 논문의 핵심 결과다. CoT-only SFT는 HotpotQA EM 38.9, ReAct-only 37.3, Reflexion-only 35.2지만, 세 method mix는 **40.0**에 도달한다. 같은 data volume에서 diversity만으로 strict improvement. Inference에서는 method-specific system prompt로 같은 model이 style을 바꿀 수 있다. 함의: **수집에 사용하는 prompting method 자체가 hyperparameter**이며, training-time answer는 "all of them"이다.

### 1.5 AutoAct — zero-teacher lower bound

[[autoact]]는 API budget이 0이고 narrow-domain QA로 충분할 때 참고할 recipe다. 하나의 base model(Llama-2-7B/13B)이 separate LoRAs를 통해 Plan, Tool, Reflect 세 역할을 수행하고, loop가 자신의 training data를 만든다.

1. Meta-agent가 각 turn의 role을 분류하도록 base model을 prompt한다.
2. Base model이 각 role 아래 raw HotpotQA questions에서 trajectories를 rollout한다.
3. Self-consistency filter: final answer가 gold와 match하거나 self-consistency majority와 match하는 trajectory를 유지한다.
4. 세 LoRA를 role-specific subset으로 fine-tune한다.
5. Iterate — retrained sub-agents가 다음 round를 위한 새 rollout을 생성한다.

13B에서 AutoAct model은 HotpotQA 약 36 EM에 도달한다. GPT-4-teacher-distilled baseline과 약 4 points 차이이며 API spend는 0이다. Iteration 4에서 saturate한다. Open weakness: **self-consistency는 base model의 bias에 anchor된다.** 7B base가 어떤 question type을 체계적으로 잘못 읽으면 어떤 iteration도 이를 고치지 못한다. 그래서 AutoAct의 successor들(그리고 frontier-scale self-improvement 논문들)은 self-play를 external verifier와 pair한다. 순수 self-referential loop는 drift한다.

---

## 2. Environment-grounded corpora — world가 채점할 때

2024→2025의 전환은 teacher ceiling을 물려받는 teacher-distilled trajectories에서, world 자체가 success를 label하는 **environment-grounded** trajectories로 옮겨가는 것이다.

### 2.1 WebArena — deterministic state를 가진 self-hosted browser

[[webarena-data]]는 다섯 real open-source apps(GitLab, Reddit-clone(Postmill), Shopping(Magento), OpenStreetMap, Calendar)를 deterministic initial DB state와 per-task reset script가 있는 Docker-compose bundle로 packaging한다. 812 tasks는 retrieval, browsing, form-filling, multi-step transactions를 포괄한다.

Action vocab: `click [element_id]`, `type [element_id] [text]`, `hover`, `press [key]`, `scroll`, `tab`, `new_tab`, `goto [url]`, `go_back`, `stop [answer]`. Observation은 accessibility tree(DOM의 text representation. text-only agent에 선호) 또는 screenshot+tree(multimodal, VisualWebArena에서 사용)다.

Success는 reference trajectory와의 similarity가 아니라 final URL / page content / DOM state에 대한 **predicate**다. Predicate category는 세 가지다. info-lookup(gold-string match), content-producing(created content에 대한 predicate), state-modifying(DB state에 대한 predicate). Trajectory-collection practice: SeeAct-style scaffold로 GPT-4를 실행하고, success predicate를 돌리고, 통과한 trajectory만 유지한다. Community dataset scale: successful trajectories 수만 개, per-trajectory 비용 \$5–\$20(GPT-4V).

**Environment-drift** hazard는 실제이며 과소평가된다. Docker image는 pin해야 한다. App version(GitLab, Magento)이 조용히 upgrade되면 old DOM structure에 의존하던 task success predicate가 깨진다. WebArena v1.0 위에서 만든 dataset은 success-predicate pass를 다시 돌리지 않으면 v1.2에서 re-executable하지 않을 수 있다. Long-lived agent corpora의 경우 bundle image를 영원히 pin하거나(storage cost는 있지만 reproducible), periodic re-validation을 계획하라. 두 번째 hazard는 **shortcut learning**이다. 일부 task는 UI를 navigation하지 않고 알려진 shortcut URL을 URL-hacking하여 풀 수 있다. Strict success predicate가 완화책이지만 leakage는 항상 어느 정도 존재한다. 그래서 frontier numbers(GPT-4 약 35%, VisualWebArena 약 20%)는 scaffold sensitivity variance가 5–10 points다.

### 2.2 SWE-Gym — 2,438 executable GitHub issues

[[swe-gym]](Pan et al. 2024, Berkeley + CMU + Apple)는 SWE-side analogue다. 11개 Python repo(astropy, sympy, django, matplotlib, …)에서 가져온 2,438 real GitHub issues로, 각각 pre-PR commit의 repo + PR test files가 적용된 Docker image로 packaging된다(테스트는 존재하지만 code는 아직 통과하지 못함). Hidden test command가 포함된다.

Action space는 **OpenHands scaffold**([[openhands-data]])다. `str_replace_editor`(view/create/str_replace/insert/undo_edit), `execute_bash`(shell + pytest), `browse`(filesystem), `finish`(submit patch). Trajectories는 흔히 median 약 15K tokens, 긴 debugging session에서는 tail이 100K를 넘는다.

Recipe: (1) 각 SWE-Gym task에서 teacher(Qwen-2.5-Coder-32B 또는 Claude-3.5)로 OpenHands를 실행, 최대 K=10 rollouts. (2) hidden tests 실행, 각 trajectory pass/fail labeling. (3) all-pass만 filter. (4) rejection-sampling SFT로 student 학습. 숫자: Qwen-2.5-Coder-7B는 SWE-Bench Verified에서 3.0% → RS-SFT 후 **15.3%** → inference에서 trained verifier best-of-N을 붙이면 **20.3%**. 32B는 **32.0%**로 release 당시 open SOTA(2024년 12월). Verifier는 SFT alone 대비 +5 points.

Verifier는 SWE-Gym의 (trajectory, success) pairs로 학습된 별도 model이다. Execution-labeled data에서 trajectory success를 예측해 rank하는 법을 배운다. Inference에서는 SFT policy로 K trajectories를 sample하고, verifier로 각 trajectory를 score한 뒤 가장 높은 것을 고른다. Scaling behavior: trajectory count와 verifier-N 모두 SWE-Bench Verified에서 log-linear returns를 보이며, 32B + K=10에서 plateau가 보이지 않는다. 이것은 **execution-labeled rejection-sampling SFT**가 agent-training recipe로서 가장 깔끔하게 경험적으로 입증된 사례이며, OpenHands scaffold가 2025년 de facto agent data pipeline이 된 이유다.

기억해야 할 SWE-Gym practicalities 두 가지. 첫째, Docker-image maintenance cost는 작지 않다. Full Python environment + hidden test suite + issue metadata가 들어간 2,438 images는 layer dedup에 따라 수백 GB에서 몇 TB까지 storage를 소모한다. Pipeline development 중 fast iteration에는 "SWE-Bench Lite와 즉시 compatible한 491 tasks" subset을 쓰고, full 2,438은 production training run에 쓴다. 둘째, **language-narrow는 실제 ceiling**이다. SWE-Gym은 Python-only이며, published numbers에서 Go/Rust/TypeScript transfer는 전혀 test되지 않았다. Multi-language SWE trajectories가 2026 frontier다(Rust cargo-integrated tasks, TypeScript jest-integrated tasks를 추가하는 논문을 예상하라). 하지만 현재 recipe는 Python-bounded다.

### 2.3 Terminal-Bench trajectories와 Explorer — terminal 및 web trajectories

[[terminal-bench-trajectories]](2026)는 Terminal-Bench 2.0 CLI tasks에 대한 수만 trial의 full agent traces(messages, tool calls, observations)를 release한다. Benchmark를 reusable trajectory corpus로 바꾸는 것이다. [[explorer]](Pahuja et al. 2025, MS)는 web side에서 반대로 간다. 고정된 812-task benchmark를 쓰는 대신 먼저 web을 *explore*(broad intent generation)하고, successful trajectories를 training data로 refine한다. 49K unique URLs에 걸쳐 94K successful multimodal web trajectories가 release되었다. Design pattern, 즉 **intent discovery와 trajectory refinement를 decouple**하는 방식은 이제 scale 있는 web-agent data의 기본값이다.

---

## 3. SWE-RL — open-source scale의 rule-based reward

[[swe-rl]](Wei et al. 2025, Meta FAIR)은 scale 있는 SWE task RL에 executable environment가 필요하지 않다는 것을 증명한 논문이다. Trick은 dense하고 cheap하며 의외로 game하기 어려운 rule-based reward다.

$$
r = \texttt{difflib.SequenceMatcher(None, predicted\_patch, ground\_truth\_patch).ratio()}
$$

그게 전부다. `difflib.SequenceMatcher.ratio()`는 matching-block coverage 기반으로 `[0, 1]` float를 반환한다. GitHub에서 scraped한 (issue, code_context, ground_truth_patch) triple에 대해, agent가 unified-diff patch를 emit하고 human PR diff와 비교해 score한다. Training 중 unit test는 실행하지 않는다. Eval time(SWE-Bench Verified)에만 실행한다.

**Data:** GitHub Archive BigQuery에서 mined한 11M (issue, context, patch) triples. Filters: PR merged, linked issue, ≤10 files, ≤500 lines, Python-primary, MinHash dedup. **Algorithm:** GRPO with group size G=8, KL coefficient β=0.02, LR 1e-6. **Base:** Llama-3.1-70B-Instruct. **Cost:** 약 1M H100-hours.

Headline result: **Llama3-SWE-RL-70B는 SWE-Bench Verified에서 41.0%**. Release 당시 open SOTA이며 DeepSeek-Coder-V2-Instruct(18.0%)를 이기고 SWE-Gym-32B와 match한다. 논문의 가장 도발적인 발견은 out-of-domain transfer다. SWE만으로 training해도 baseline 대비 HumanEval+ +6, MATH +4, BBH +3이 된다. Hypothesis: software-engineering task에서의 RL이 domains를 넘나드는 "long-horizon grounded planning"을 가르친다. (이것은 ch-26의 [[front-loading-reasoning]]과 연결된다.)

**Similarity가 training signal로 execution보다 나은 이유.** Execution reward는 sparse하다. 많은 test가 unrelated reasons(dependency version, unrelated test flakiness, setup error) 때문에 fail한다. Similarity reward는 dense하다. *모든* sample이 gradient signal을 얻는다. 비용은 gameability다. Context를 그대로 copy한 patch가 아무것도 고치지 않고 partial credit을 받을 수 있다. Mitigation: format filters(diff여야 함, code를 수정해야 함, comment만 수정하면 안 됨). Authors는 binary thresholding과 continuous reward도 비교하며, continuous가 이긴다.

한 가지 caveat: SWE-RL은 **single-turn**이다. issue → patch이며 file navigation이나 test running mid-trajectory가 없다. Executable env에서 multi-turn RL은 여전히 SWE-Gym의 영역이다. 두 recipe는 경쟁이 아니라 보완이다. SWE-RL은 cheap-dense-signal stage를 맡고, SWE-Gym은 environment-grounded multi-turn stage를 맡는다.

Decontamination은 별도 문단을 받을 만하다. SWE-Bench Verified도 SWE-RL이 scrape하는 GitHub universe에서 나오므로, date-based filtering(training data가 benchmark issue보다 앞섬)과 commit-hash blocklist가 필수다. 논문은 둘 다 보고한다. 더 subtle한 risk도 있다. **Similarity reward는 human diff처럼 보이는 patch 쪽으로 bias**되어, 모델이 사람이 쓴 것보다 *더 나은* patch를 만들 수 있는 능력을 가릴 수 있다. Training time에는 이것이 ceiling behavior처럼 보인다. Eval time에 SWE-Bench Verified(unit test로 functional correctness를 test하고 string similarity를 보지 않음)에서는 training reward와 eval reward의 gap이 training-curve-plateau로 나타나는 동안 eval curve는 계속 오른다. 이 mismatch를 제대로 읽으려면 두 metric을 머릿속에서 분리해야 한다. Agent pipeline에 처음 들어온 ML practitioner가 자주 무너뜨리는 부분이다.

---

## 4. Kimi-K2 — frontier lab이 실제로 하는 일

[[kimi-k2]]와 [[kimi-k2-agentic-data]](Moonshot AI, 2025)는 가장 완전한 public frontier-lab agentic recipe를 함께 설명한다. K2는 32B active인 1T-parameter MoE이며, **MuonClip** optimizer(Muon + QK-Clip: update 후 Q/K projection matrix를 rescale해 attention logit magnitude를 cap함. plain Muon은 이를 1000 이상으로 drift하게 둔다)로 loss spike 없이 15.5T tokens에서 pretrained되었다.

Agentic recipe는 네 stage다.

1. **Agentic PRETRAINING data(약 1T tokens).** Synthetic environment simulator가 수만 tool schemas(web search, file ops, code exec, DB, calendar, enterprise APIs)와 그럴듯한 tool-response shapes를 만든다. Multi-agent simulation이 trajectories를 만든다. "user" agent가 query를 내고, "planner"가 decomposes하고, "executor"가 tool call을 emit하고, "critic"이 review한다. Trajectory당 최대 5 sub-agents. Critic-LLM이 success/coherence/tool-call validity를 rate하며, top-scoring trajectories는 total token budget의 **약 3–5%** weight로 pretraining mix에 들어간다.
2. **SFT.** K2 technical report의 20,000+ tool library 위 real-world agentic tasks(SWE-Bench-style issues, τ-bench tasks, tool-calling datasets).
3. **Joint RL stage.** 두 reward stream을 하나의 scalar로 결합한다. **RLVR**(verifiable rewards — math, code, tool-call correctness) + **self-critique rubric reward**(모델이 task에 맞는 rubric을 만들고 자신의 completion을 그 rubric으로 score). Self-critique stream은 open-ended-tasks에 대한 alignment component이고, RLVR stream은 verifiable-tasks에 대한 capability component다. 하나의 RL stage가 두 skill slice를 함께 학습한다.
4. **Evaluation.** K2-Instruct는 **SWE-Bench Verified 약 65%**에 도달하고, τ-bench에서 open models를 lead하며, tool-use에서 Claude-3.5-Sonnet과 경쟁한다.

논쟁적인 claim은 stage 1이다. Moonshot은 agentic behavior가 post-training 후처리로 붙는 것이 아니라 **pretraining 중 설치되는** 것이 최선이라고 주장한다. Agent-format tokens를 pretrain distribution에 섞으면 base model이 native tool-calling "vocabulary"를 가지므로 post-training이 base distribution과 싸우지 않는다. 이것이 §1의 SFT-only lineage와 가장 날카롭게 갈라지는 지점이다. 일반화되는지는 아직 open empirical question이다. Moonshot 밖에서는 아무도 1T-token agentic-pretrain mix를 scale로 재현하지 못했다.

K2 report에서 기억할 detail 세 가지. 첫째, self-critique rubric reward는 automated verifier가 없는 **open-ended** tasks로 alignment를 확장한다. 모델이 (a) task-appropriate rubric을 만들고, (b) 자신의 completion을 만든 뒤, rubric에 따라 completion을 score한다. Joint RL은 이것을 RLVR과 하나의 scalar로 결합하므로 한 stage가 verifiable 및 open-ended skills를 모두 학습한다. 이것은 1T-MoE scale에서 operationalized된 [[constitutional-ai]]의 self-rating idea의 후손이다. 둘째, **MuonClip**은 trillion-scale stability에 optional이 아니다. QK-Clip 없는 plain Muon은 attention-logit max가 1000을 넘어서며 diverge하는 것이 관찰되었다. QK-Clip은 각 update 후 Q/K projection matrix를 rescale해 max가 threshold 아래에 머물게 하며, 전체 15.5T-token run에서 zero loss spikes를 낸다. 셋째, **20K+ tool library**는 pretraining과 post-training trajectories가 모두 sample되는 surface area다. 하나의 tool-schema registry가 모든 stage를 feed하기 때문에 agent behavior가 flow 전반에서 coherent하다. 이 unification은 scientific contribution만큼이나 engineering contribution이다.

K2 report가 조심스럽게 다루지만 뽑아 둘 가치가 있는 네 번째 detail: **synthetic-env mismatch**. Simulator-generated tool responses는 plausible하지만 real-world API failure mode와 동일하지 않다. Real API는 `429 rate limited`, `500 upstream timeout`, partial results, stale caches를 반환한다. Simulator는 더 깨끗한 response를 내는 경향이 있다. 이것은 ALFWorld / WebShop이 real web에 비해 "stylized"인 것과 같은 analogue다. K2는 SFT 중 real-env trajectories를 섞어 완화한다(τ-bench는 real이고, SWE-Bench-style tasks는 real code를 execute한다). Methodology-vs-dataset opacity가 또 다른 caveat다. Moonshot은 *how*(four-stage flow, critic-filter, 3–5% pretrain weight)를 공개하지만 *what*(1T-token agentic corpus 자체)은 공개하지 않는다. Recipe는 재현할 수 있다. 비슷한 비용을 쓰지 않으면 corpus는 재현할 수 없다.

---

## 5. Action-space design — 필요한 단 하나의 table

Action space는 다른 모든 것을 제약하는 단일 설계 선택이다. Observation format, trajectory length, success signal, teacher cost, eval harness가 모두 따라온다. Interactive side-by-side는 [[figures/action-space.html]]을 보라.

| Environment | Observation type | Action vocab | Reward signal | Typical traj length |
|---|---|---|---|---|
| **Web browser** ([[webarena-data]], [[explorer]]) | Accessibility tree (text) *or* screenshot+tree (multimodal) | `click[id] / type[id,text] / hover / press / scroll / tab / new_tab / goto[url] / go_back / stop[answer]` | Predicate over URL + DOM + DB state (three categories: info-lookup, content-producing, state-modifying) | 5–30 steps, 10K–50K tokens |
| **Terminal** ([[terminal-bench-trajectories]], [[openhands-data]]) | Shell stdout/stderr + exit code | `execute_bash[cmd]`, possibly `execute_ipython_cell` | Test-suite pass, file-state check, or CLI-predicate | 5–50 steps, 5K–40K tokens |
| **Repo / SWE** ([[swe-gym]], [[swe-rl]], [[openhands-data]]) | File contents + dir listing + test output | `str_replace_editor.view/create/str_replace/insert/undo_edit`, `execute_bash`, `browse`, `finish` | Hidden pytest pass (SWE-Gym) *or* difflib-ratio vs gold PR (SWE-RL) | 10–100+ steps, 15K–100K tokens |
| **Sandbox** ([[kimi-k2-agentic-data]], [[agentinstruct]] tool-use subset) | Simulated tool-response JSON | OpenAI-style `tool_calls` JSON over 20K+ schemas | Critic-LLM score + JSON-schema validity + no-repeat-loop | 1–20+ tool calls, variable |

이 table에서 나오는 structural lesson은 세 가지다.

- **Observation format이 teacher cost를 결정한다.** Accessibility tree는 싸다. Text-only라 GPT-4가 normal rate로 처리한다. Screenshot은 비싸다. Web trajectory당 GPT-4V \$5–\$20. Terminal stdout은 무료다. 그래서 같은 budget이면 WebArena text-mode dataset이 VisualWebArena dataset보다 10× 크다.
- **Reward signal이 RL tractability를 결정한다.** Dense rule-based signals(SWE-RL의 difflib, predicate-with-partial-credit)는 11M-sample scale의 RL을 지원한다. Binary test-suite rewards(SWE-Gym)는 rejection-sampling SFT에는 싸게 쓸 수 있지만 RL에서는 sparse하다. 대부분 sample이 0 gradient를 받는다. Critic-LLM rewards(Kimi-K2, Agent-FLAN filter)는 dense하지만 drift-prone이다. self-critique rubric anchoring이나 fixed-scale calibration이 필요하다.
- **Action vocab size는 generalization과 safety를 trade off한다.** 20K+ schemas 위 OpenAI-style `tool_calls` JSON은 최대한 general하지만 hallucinate하기 쉽다(Agent-FLAN의 네 negative class 대부분이 이 failure mode다). Tight 10-action WebArena vocab은 더 안전하다. Grammar가 허용하지 않으므로 모델은 `hack_database` action을 invent할 수 없다.
- **Trajectory length가 training infrastructure를 결정한다.** SWE의 median 15K tokens는 normal SFT example과 겨우 다르다. 100K+ tail은 full trajectory에 대한 FlashAttention-3-class long-context support와 gradient checkpointing이 필요하다. Web agents의 50K tokens는 그 중간이다. Kimi-K2의 simulated-sandbox trajectories는 "median several thousand tokens"라 training이 싸다. SWE-Gym trajectories의 median 15K는 그렇지 않다. Training stack보다 data pipeline을 먼저 만들고 있다면 length distribution을 먼저 확인하라. Training team이 고마워할 것이다.

Table이 보여 주지 못하는 마지막 점: **teacher choice는 action distribution에 새어 들어간다.** GPT-4는 언제 navigate하고, 언제 search하고, 언제 stop할지 자체 prior가 있다. GPT-4를 teacher로 trajectory를 수집하면 이 prior가 student에 imprint된다. Dataset 중간에 teacher를 바꾸는 것(community OpenHands release처럼 Claude-3.5와 GPT-4 rollout을 섞는 것)은 action distribution을 diversify하지만 corpus를 약간 inconsistent하게 만든다. Claude는 더 긴 `think` block을 쓰는 경향이 있고, GPT-4는 더 빨리 움직이는 경향이 있다. 둘 다 괜찮다. Mixing에는 label이 필요해 downstream filtering이 stratify할 수 있어야 한다.

---

### 5.2 Rough cost reckoning — 각 recipe의 실제 비용

아래 숫자는 논문에서 온 order-of-magnitude estimate이지 정확한 disclosure가 아니다. 느슨하게 보되, claim의 sanity check에 사용하라.

| Recipe | Data scale | Compute (rough) | Notes |
|---|---|---|---|
| AgentTuning SFT | 1,866 trajectories + GPT-4 distill | ~$20K API + 100s of GPU-hours for SFT | Cheapest real agent recipe |
| FireAct SFT | ~2,000 traj × 3 methods + GPT-4 | ~$3K API + SFT compute | Method-mix dominates spend |
| AgentInstruct 25M | 25M pairs + multi-agent flows | >$500K GPT-4 API est. | Pipeline cost is the bottleneck |
| SWE-Gym RS-SFT (32B) | 2,438 tasks × K=10 rollouts | ~10K H100-hours | Docker + teacher cost dominates |
| SWE-RL 70B | 11M GH triples + GRPO | ~1M H100-hours | RL dominates; teacher-free |
| Kimi-K2 agentic pretrain | ~1T agentic tokens | Part of 15.5T-token pretrain | Integrated into base training |

## 6. ch-28로 가져갈 것

1. **Dataset은 environment다.** 읽는 모든 agent-SFT paper를 머릿속에서 `(environment, action vocab, observation format, success predicate)`로 축약하라. 나머지는 teacher-model plumbing이다.
2. **Negatives는 load-bearing이다.** [[agent-flan]]의 네 hallucination class는 optional add-on이 아니다. 이것이 빠진 agent corpus는 chat에서 tool을 과도하게 call하는 model을 만든다.
3. **Mixing ratio는 raw agent-data volume보다 중요하다.** [[agenttuning]]의 1:10 rule은 일반화된다. Agent data가 SFT mix를 지배하면 general chat quality가 떨어진다. General data가 지배하면 agent skill이 설치되지 않는다.
4. **Frontier scale에서는 pretrain injection이 post-training retrofit보다 낫다.** [[kimi-k2]]는 이를 명시적으로 주장한다. [[swe-rl]]의 out-of-domain transfer는 이유를 암시한다. Base distribution 안의 agentic token은 "next token"의 의미 자체를 바꾼다.
5. **Rule-based dense reward는 scale되고, execution-based sparse reward는 그렇지 않다(RL training time 기준).** 1M H100-hours에서 SWE-RL 41.0%가 existence proof다. Execution은 eval에 아껴라.
6. **Role specialization은 반복된다. 사용하라.** Lumos(Plan/Ground/Execute), AutoAct(Plan/Tool/Reflect), AgentInstruct(suggester/editor/generator), Kimi-K2(user/planner/executor/critic). Deployed model이 monolithic이어도 role-partitioning the *data*는 더 깨끗한 supervision signal을 준다.
7. **Environment-drift는 long-lived-corpus problem이다.** WebArena DOM upgrades, SWE-Gym dependency drift, terminal-env library changes. Image를 pin하거나 periodic re-validation을 계획하라. 2024년에 나온 benchmark는 2026년에 그냥 re-executable하지 않다.

다음 장은 "executable world에 조건화"에서 "very long context에 조건화"로 구조적 전환을 확장한다. Long-context synthesis에서는 world가 128K–1M-token document로 대체되고 action은 "read carefully"가 된다. Action space는 `read(span)` + `attend(position)`으로 collapse하지만, trajectory length는 steps가 아니라 tokens에서 100K에서 1M으로 폭발한다. 다른 modality지만 같은 structural lesson이다. Environment가 data를 shaping한다.
