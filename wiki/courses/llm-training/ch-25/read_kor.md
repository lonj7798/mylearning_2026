<!-- chapter: ch-25
     track: synthetic
     kind: content
     title: Multi-Turn Conversation Synthesis
     deps: [ch-24]
     sources: [[baize-construction]], [[baize-construction-recipe]], [[ultrachat-pipeline]], [[ultrachat-two-model-protocol]], [[camel]], [[prosocial-dialog]], [[soda]], [[soda-commonsense-grounding]], [[openassistant]], [[persona-hub]], [[capybara]], [[smol-talk]], [[modern-mix-composition]], [[tulu-3]], [[wildchat]], [[wildchat-statistics]], [[lmsys-chat-1m]], [[lost-in-multi-turn]], [[mt-eval]], [[false-promise-imitating-proprietary-llms]], [[xstest]], [[agentinstruct]]
     figures: figures/turn-distribution.html, figures/aptitude-unreliability.html
     revised: 2026-09 (generality revision)
-->

# 제25장 — Multi-Turn Conversation Synthesis

> **핵심 통찰.** Published multi-turn synthesis 방법은 second speaker를 세 방식으로 만든다. API call 하나가 양쪽 말을 모두 쓰는 Baize, topic taxonomy 아래 user/assistant를 별도 call로 번갈아 쓰는 UltraChat, role을 고정한 agent 둘이 task를 instruct/solve하는 CAMEL이다. SODA의 controlled comparison에서 external content로 scene을 ground하면(commonsense triple, ProsocialDialog의 기존 corpus situation) ungrounded GPT-3.5 dialogue 100개보다 human judge가 선호했다([[soda-commonsense-grounding]] §3.3). LMSYS-Chat-1M/WildChat conversation 평균은 2.02/2.54 turn이고([[wildchat-statistics]] Table 1; [[lmsys-chat-1m]] Table 1), [[lost-in-multi-turn]]이 시험한 model은 task가 여러 turn에 걸쳐 드러날 때 single-turn score의 평균 39%를 잃는다. Realistic해 보이는 dialogue corpus도 이 robustness를 보장하지 않는다. [[false-promise-imitating-proprietary-llms]] imitation data는 crowd rating은 올렸지만 benchmark accuracy는 올리지 못했다.
>
> **지침.** 범용 model용 multi-turn SFT data를 합성할 때 user/assistant turn은 separate call로 생성하고, dialogue마다 external content source를 condition한다. Role exchange([[ultrachat-pipeline]] §4.4)와 dull ungrounded dialogue([[soda-commonsense-grounding]] §3.3)에 대한 문서화된 대응이기 때문이다. Underspecified request robustness가 목표면 requirement가 여러 turn에 걸쳐 들어오는 dialogue를 넣는다. Fully specified single-turn instruction은 이 behavior를 재지 못한다([[lost-in-multi-turn]] §6). Corpus를 real-user reference와 turn count/token length/embedding coverage로 비교하고, held-out multi-turn suite와 single-turn control에서 trained model을 평가한다([[mt-eval]] Table 3). Short assistant-style chat만 필요하면 Llama-3.1-405B-Instruct가 만든 three-turn synthetic mix는 1.7B에서 측정된 baseline이다([[modern-mix-composition]], SmolLM2 Table 10).

## 학습한 버전에 대한 정정

1. “Ch-24가 generic synthetic-pipeline loop를 넘겨 주었다” → generic generate-filter-deduplicate-verify-select-mix loop는 ch-18(The Synthetic-Data Design Pattern)이다. Ch-24는 reasoning trace를 다룬다.
2. “Seed pool: Quora 약 54K, StackOverflow 약 57K, Alpaca 약 52K, MedQuAD 약 47K → dialogue 111.5K개” → self-chat dialogue 111.5k개는 Quora question 약 55k개와 Stack Overflow question 약 55k개에서 나온다. Alpaca example 51,942개는 training에 추가한 single-turn data이고, MedQuAD는 dialogue 46,867개로 된 별도 healthcare set이다([[baize-construction]], §3, Table 2).
3. Baize template의 `[|Human|]` / `[|AI|]`와 “약 8-turn cap” → Self-chat template은 `[Human]` / `[AI]`를 쓰고 natural stopping point까지 실행한다. `[|Human|]`는 inference prompt에 속한다([[baize-construction]], App. A, App. B).
4. “Median 4 turn, IQR 3–6, turn당 약 100 token; total cost 약 $1,000” → Average는 3.6–3.9 turn, average response length는 약 36(unit 미명시), cost는 약 $100이다([[baize-construction]], §3, Table 2).
5. “Self-Distill with Feedback … critique를 DPO pair로 사용” → SDF는 Quora question마다 response 네 개를 sample하고, ChatGPT가 각각 1–100으로 score하며, 최고 response가 새 LoRA module을 훈련한다. DPO는 없다. Baize(arXiv 2023-04)는 DPO(arXiv 2023-05)보다 앞선다([[baize-construction]], §4, App. C).
6. “Baize-13B가 human-eval comparison의 58%에서 Alpaca-13B를 이겼다” → Paper에는 없다. Reported evaluation은 ChatGPT 대비 GPT-4 score이며 Alpaca-13B 76%, Baize-v2-13B 92%다([[baize-construction]], Fig. 3).
7. UltraChat “meta-topic 30개 → subtopic 1,100개 이상; C4 passage 약 100K개; Q-world 약 600K / Writing 약 400K / Assistance 약 500K” → Paper는 topic 30개와 각각 subtopic 30–50개, Sector I의 opening question 약 500,000개, C4 piece 10,000개와 piece당 instruction 5개, total dialogue 1,468,352개를 말한다. Sector별 dialogue count는 보고하지 않는다. 1,100+와 약 100k 수치는 repository README에서 왔다([[ultrachat-two-model-protocol]]; [[ultrachat-pipeline]] arXiv v1 §4.1–4.3, Table 5).
8. “Filter stage는 opaque하다” → Paper는 filter 하나를 이름으로 든다. User turn을 더 realistic하게 하려고 excessively polite user statement(“Thank you”, “Thanks”, “You're welcome”)를 제외한다([[ultrachat-pipeline]] §4.4).
9. CAMEL “50 × 50 × 20 grid”, “AI Society dialogue 약 1M개”, “cost 약 $5–10K” → Assistant role 50개 × user role 50개 × task 10개 = conversation 25,000개이며 cost는 보고하지 않는다([[camel]], §4.1).
10. CAMEL “median 6–8 turn, tail은 20, turn당 약 50 token” 및 stop rule로서의 “semantic termination” → Turn/token distribution은 보고하지 않는다. Conversation은 condition 다섯 개 중 하나에서 stop한다(no instruction 3 round, role reversal, end-of-task token, token limit, message 40개). AI Society conversation은 주로 assistant role reversal로 끝난다([[camel]], §4.1, App. J Fig. 8).
11. Role example “Architect, Astronaut, Biologist, Graduate Student”, “Astronaut × Bartender” pair, “As an accountant” leak → Paper의 role list/finding에 없다. Report한 failure mode는 role flipping, instruction repetition, flake reply, infinite loop다([[camel]], App. E, §4.1).
12. “Inception prompting이 AgentInstruct, APIGen-MT, ToolACE로 전파된다” → AgentInstruct와 APIGen-MT는 CAMEL을 cite하지만 Inception Prompt를 reuse한다고 설명하지 않는다. AgentInstruct suggester agent는 task를 더 complex하거나 unsolvable하게 만들며 implausible role pair를 pre-filter하지 않는다([[camel]] Connections; [[agentinstruct]] §2).
13. “Yejin Choi lineage self-instruct → SODA → Prosocial-Dialog가 하나의 arc다” → ProsocialDialog(arXiv 2022-05)는 SODA와 Self-Instruct(둘 다 arXiv 2022-12)보다 앞선다. Self-Instruct는 이 group의 연구가 아니다. SODA는 ProsocialDialog의 Canary를 safety filter로, ProsocialDialog를 extra COSMO training data로 쓴다([[prosocial-dialog]] Verification; [[soda-commonsense-grounding]], §3.1, §4).
14. “Atomic 10X, commonsense-triple database 약 10M record”, “1.5M triple이 `{xWant, xNeed, xEffect, xReact}`에 균형” → SODA paper는 Atomic10x size를 말하지 않는다. x-relation 여섯 개에서 triple 1.5 million을 쓰며 share는 xAttr 18%, xEffect 17%, xIntent 23%, xNeed 7%, xReact 25%, xWant 11%다([[soda-commonsense-grounding]], §3.2 Table 3, App. A.1).
15. SODA “total turn 4–10”, “median 7–8, max 약 12, turn당 average 20 token”, “cost 약 $10K” → Filtering은 turn 4개 미만 또는 20개 초과 conversation을 제거했다. SODA average는 turn 7.6, utterance length 16.1(unit 미명시), triple-to-dialogue 하나의 cost는 약 $0.02다([[soda-commonsense-grounding]], §3.1, Table 2).
16. “COSMO-3B … BlenderBot-3B를 … parameter 1/3로 이긴다” → 둘 다 3B다. DailyDialog에서 human judge는 BlenderBot-3B와의 comparison 중 72%에서 COSMO-3B overall을 선호했다([[soda-commonsense-grounding]], §5.1 Table 5).
17. ProsocialDialog “crowd worker가 harm category 10개에 걸쳐 problematic prompt 약 10K개를 author”, “unique RoT 300개 이상”, “median 3 turn” → GPT-3가 Social Chemistry, ETHICS, SBIC situation에서 problematic turn을 작성한다. Unique RoT는 160,295개, average turn은 5.7, 최대는 6이다([[prosocial-dialog]], §3.2–3.4, App. A.1).
18. “CANARY-400M은 problematic prompt 89%에 constructively engage, BlenderBot-3B는 32%” 및 “Constitutional AI의 precursor” → 뒷받침되지 않는다. Canary는 T5-large safety-label/RoT generator이며 dialogue agent는 Prost(2.7B)다([[prosocial-dialog]], §4, Verification).
19. OASST1 “fully-labeled tree 10K, CC-BY 4.0”, “너무 작음(약 250K message)”, “OASST2가 약 90K를 추가” → Complete tree 10,968개/66,497개, message 161,443개다. Paper는 fully permissive licence를 말하고 Hugging Face card는 Apache-2.0을 적는다. OASST2는 source가 다루지 않는다([[openassistant]], §4, Verification).
20. OASST “typo rate 약 3%”, “20+ turn의 power-law tail”, “synthetic corpus의 sibling variance 거의 0” → 이를 측정한 source는 없다. OASST guideline은 prompter에게 typo를 피하라 하고, current collection setting은 tree depth를 5로 cap한다([[openassistant]], App. A, App. G).
21. “Persona-conditioning은 unique vocabulary 최대 2×, persona-consistent follow-up 3×, IFEval +5–10; Tülu-3/Qwen 2.5에 통합” → Primary source는 이 수치를 주지 않는다. Tülu 3는 persona dataset을 제거하면 IFEval이 72.8에서 53.6으로 낮아지고 unseen IFEval-OOD는 17.6에서 18.0으로 바뀐다고 보고한다([[modern-mix-composition]]; [[tulu-3]] arXiv v5 Tables 10, 32).
22. Persona Hub “web bio / about-me / LinkedIn에서 mined한 persona 1B” → Persona는 RedPajama v2 web text에서 infer(Text-to-Persona)하고 relationship으로 expand(Persona-to-Persona)하며, deduplication 뒤 1,015,863,523개다([[persona-hub]], §2).
23. Capybara “dialogue 약 20K, multi-turn 60% 이상; round 2–6회 iterate; consumer GPU 하나에서 competitive MT-Bench” → Released file은 row 16,006개이고 exchange 두 개 이상은 66.7%다. 약 20,000은 7B V1.9 training set이다. Round rule/GPU/MT-Bench claim은 card에 없다([[capybara]]).
24. SmolTalk “explicit multi-turn dialogue는 1M 중 약 30K — 3%” → 가장 큰 component인 MagPie-Ultra(약 1.1M 중 431k)는 three-turn conversation으로 생성됐고 Everyday-Conversations(2.38k)도 multi-turn이므로 sample의 최소 39.4%는 multi-turn이다(derived). Component list에는 APIGen-Function-Calling 87.5k, LongAlign 3.73k, Explore-Instruct-Rewriting 32k도 있다([[modern-mix-composition]]; SmolLM2 §5.1.1, App. F Table 9).
25. “OASST/WildChat anchor slice 5–15%”와 “RoT-grounded engagement가 helpfulness/safety에서 refusal보다 낫다” → 둘 다 주는 source는 없다. Tülu 3 8B SFT에는 prompt 939,344개 중 WildChat prompt 100,000개가 있어 10.6%다(derived). ProsocialDialog는 그것만으로 train하면 “negativity-prone chatbot”이 나올 수 있다고 경고한다([[tulu-3]] Table 7; [[prosocial-dialog]] §9).

## 이 장이 범용 모델에서 중요한 이유

범용 assistant는 conversation에서 쓰인다. LMSYS-Chat-1M average conversation은 2.0 turn([[lmsys-chat-1m]] Table 1)이고 WildChat은 약 41%가 multi-turn, 3.7%가 10 turn 초과다([[wildchat]] §3). 중요한 multi-turn skill은 측정 가능하다. Turn 1 instruction을 recall하고, earlier constraint를 잃지 않고 revision을 적용하며, 처음에 request가 underspecified일 때 recover하는 능력이다([[mt-eval]] §1; [[lost-in-multi-turn]] §1). 이 장은 synthetic-data stage에서 SFT(ch-29e, ch-30) 전에 위치한다. Multi-turn SFT data 생성, real user와 distribution 비교, narrowing을 찾는 evaluation을 다룬다. Reasoning-trace data(ch-24)는 single-response depth를 공급하고, 이 장은 tool-calling(ch-26)과 long conversation(ch-29b)이 확장하는 conversation structure를 공급한다.

## §1 단위: turn, exchange, 통계의 의미

**정의.** *Message*(또는 utterance)는 speaker 한 명의 text다. *Exchange*(round라고도 함)는 user message 하나와 assistant message 하나다. Paper는 “turn”을 두 단위 중 어느 것에도 쓰며, 일부는 어느 단위인지 밝히지 않는다.

**문제.** Unit을 모르면 paper 간 turn count는 비교할 수 없다. 따라서 synthetic corpus가 real traffic보다 2배 길거나 짧아 보일 수 있다.

**계산 예시.** [[soda-commonsense-grounding]] Table 1의 SODA sample은 Madeleine/Coach가 각 4개씩 말한 utterance 8개, 즉 exchange 4개다. SODA Table 2의 average 7.6 turn은 utterance count다. UltraChat Table 5는 SODA를 3.6 turn으로 재측정했는데 `7.6 / 2 = 3.8`에 가깝고 exchange를 센다([[ultrachat-pipeline]] Table 5). 두 수치는 같은 data를 설명한다.

**Reference 통계.** 표는 print된 수치를 모았다. Source가 밝힌 tokenizer/unit도 적는다.

| Corpus | User side 생성 방식 | Conversation | Avg. turn(unit) | User / assistant length | Source |
|---|---|---|---|---|---|
| Baize v1 Quora / StackOverflow | assistant와 같은 call | 54,456 / 57,046 | 3.9 / 3.6 | response 35.9 / 36.0(unit 미기재) | [[baize-construction]] Table 2 |
| UltraChat | 별도 user-model call | 1,468,352 | 3.8(exchange) | utterance 309.3 token | [[ultrachat-pipeline]] Table 5 |
| CAMEL AI Society | role-locked agent | 25,000 | 미보고(cap 40 message) | 미보고 | [[camel]] §4.1 |
| SODA | narrative-conditioned LLM | 1,486,896 | 7.6(utterance) | utterance 16.1(unit 미기재) | [[soda-commonsense-grounding]] Table 2 |
| ProsocialDialog | GPT-3 problematic speaker, human feedback | 58,137 | 5.7(utterance) | utterance 20.0(unit 미기재) | [[prosocial-dialog]] Table 1 |
| Capybara | 미문서화(Amplify-Instruct) | 16,006 | mean 2.58, median 3, max 17(exchange) | 미보고 | [[capybara]] |
| OpenAssistant | human volunteer | 46,283(WildChat 측정) | 2.34 | 33.41 / 211.76 Llama-2 token | [[wildchat-statistics]] Table 1 |
| ShareGPT | real user | 94,145 | 3.51 | 94.46 / 348.45 Llama-2 token | [[wildchat-statistics]] Table 1 |
| LMSYS-Chat-1M | real user, model 25개 | 1,000,000 | 2.02 | 69.83 / 215.71 Llama-2 token | [[wildchat-statistics]] Table 1 |
| WildChat | real user, ChatGPT/GPT-4 | 1,039,785 | 2.54 | 295.58 ± 1609.18 / 441.34 Llama-2 token | [[wildchat-statistics]] Table 1 |

[figures/turn-distribution.html](figures/turn-distribution.html)은 이 average를 나란히 plot하고 turn-count view와 same-tokenizer user/assistant length view를 전환하게 한다.

**함의.** Real-user reference는 average가 짧지만 long prompt tail이 있다(WildChat user token mean 295.58, standard deviation 1609.18). Synthetic row는 mixed unit에서 average 2.58–7.6을 보고하고 user/assistant length split이 없다. Unit/tokenizer를 맞추기 전에는 real traffic과 mean/spread가 맞는지 판단할 수 없다.

## §2 Family 1: single-call self-chat(Baize)

**정의.** Self-chat은 seed topic을 중심으로 model call 하나가 speaker role 양쪽의 전체 transcript를 작성하는 method다([[baize-construction]], arXiv 2023-04).

**다루는 문제.** 2023년 초 open model에는 multi-turn training data가 부족했다. 저자들은 ShareGPT log가 가진 privacy/copyright concern을 피하고자 했다(§3).

**방법.**

1. Quora 또는 Stack Overflow에서 seed question을 sample한다(각각 약 55k).
2. Self-chat template에 이를 넣고 `gpt-3.5-turbo`를 한 번 call한다.
3. Model은 human이 “has no more question”할 때까지 `[Human]` / `[AI]` turn을 작성한다.
4. Parsed dialogue와 single-turn Alpaca example 51,942개로 LLaMA를 LoRA training한다.

Template의 원문(App. A):

```
Forget the instruction you have previously received. The following is a conversation between a human and an AI assistant. The human and the AI assistant take turns chatting about the topic: '${SEED}'. Human statements start with [Human] and AI assistant statements start with [AI]. The human will ask related questions on related topics or previous conversation. The human will stop the conversation when they have no more question. The AI assistant tries not to ask questions. Complete the transcript in exactly that format.
[Human] Hello!
[AI] Hi! How can I help you?
```

**증거.** Dialogue 111.5k개는 약 $100이 들었다(§3). Single-call generation은 AI turn을 separate call로 각각 재생성했을 때보다 짧은 AI turn을 만들었다(Quora average response length v1은 35.9, Quora v2는 149.6). 따라서 v1.5 data에서는 AI turn을 한 번에 하나씩 교체했다(§3, Table 2). SDF는 question마다 sample한 response 네 개를 ChatGPT로 score하고 최고 response로 training했다. ChatGPT 대비 GPT-4 score는 7B에서 87%에서 90%, 13B에서 89%에서 92%로 올랐다(§4, Fig. 3). 결과는 judge 하나와 Vicuna prompt 80개를 쓴 single study다. Open LLM Leaderboard에서 Baize-v2-13B는 LLaMA-13B 대비 MMLU를 37.7에서 39.4로, TruthfulQA를 39.9에서 48.3으로 올렸지만 HellaSwag은 78.9에서 77.1로 낮아졌다(Table 7).

**조건과 한계.** User side는 answer를 plan하는 same call이 작성한다. 저자들은 future work로 “diversify simulated user queries”를 든다(§7). Filtering, deduplication, cross-turn dependency check는 보고하지 않는다.

**범용 모델에 대한 의미.** Self-chat은 낮은 cost로 multi-turn format을 주지만 user turn은 assistant turn과 같은 distribution에서 나온다. 따라서 SFT mix에서 user behaviour의 유일한 source가 되어서는 안 된다(Interpretation).

## §3 Family 2: topic taxonomy 아래의 two-call dialogue(UltraChat)

**정의.** Two-call pipeline은 full history에 condition하여 model call 하나로 다음 user message를 쓰고, 별도 call로 assistant reply를 쓴다([[ultrachat-pipeline]], arXiv 2023-05).

**다루는 문제.** 저자들은 dialogue corpus diversity가 opening line의 diversity와 user response style의 diversity에 달려 있다고 말한다(§4). History만 준 user model은 “tends to assume the role of an AI assistant”라고도 말한다(§4.4).

**방법.** Exchange `t`개 뒤 history `h_t`, user prompt `p_U`, assistant prompt `p_A`가 있을 때:

```
u_{t+1} = UserLLM(p_U, h_t)
a_{t+1} = AssistantLLM(p_A, h_t ∪ {u_{t+1}})
h_{t+1} = h_t ∪ {u_{t+1}, a_{t+1}}
```

`u`는 user message, `a`는 assistant message이며 두 call 모두 ChatGPT Turbo였다. Opening line은 sector 세 개에서 나온다(arXiv v1 §4.1–4.3).

1. **Questions about the World:** ChatGPT가 topic 30개와 topic당 subtopic 30–50개를 나열한다. Subtopic마다 question 10개를 만들고, question마다 question 10개를 더 만든다. 두 번째 branch는 가장 frequent한 Wikidata entity 10,000개를 사용하며 entity마다 meta-question 5개, specific question 10개, extended question 20개를 만든다. Opener로 question 약 500,000개를 sample한다. Figure 1은 round 3–7개를 보인다.
2. **Creation and Writing:** Material type 20개(Table 3)의 instruction이며, 그중 약 80%는 더 detailed한 instruction으로 expand한다.
3. **Assistance on Existing Materials:** C4 piece 10,000개에 각각 instruction 다섯 개를 만들고, `{text} Based on the passage above, {instruction}` 같은 template 일곱 개와 join한다(Table 4). Paper는 concatenate한 set이 opener 500,000개라고 말한다. `10,000 × 5`는 text-instruction pair 50,000개가 되며, template step이 어떻게 500,000을 만드는지는 paper 안에서 설명하지 않아 unresolved다.

User prompt는 model에 “to adopt various user personalities”하라고 instruct한다. Post-generation filter는 excessively polite user statement를 제거한다(§4.4). Repository README는 subtopic 1,100개 이상, C4 material 약 100k개, Sector 2–3은 round 2–4개라고 제시한다([[ultrachat-two-model-protocol]]).

**증거.** UltraChat은 dialogue 1,468,352개, average exchange 3.8개, dialogue당 token 1,467.4개, MTLD lexical diversity 74.3, ChatGPT-rated coherence 9.06을 가진다(Table 5). UltraLLaMA(LLaMA-13B)는 response에만 loss를 계산하고 dialogue를 최대 2,048 token으로 split하며 total batch 512로 훈련했다(§4.6). 뒤의 controlled comparison에서 1.7B로 각각 epoch 하나를 훈련했을 때 UltraChat SFT는 IFEval 27.26, MT-Bench 4.66이었고, three-turn MagPie-Ultra set은 35.49, 5.22였다([[modern-mix-composition]], SmolLM2 Table 10). 결과는 single study다.

**조건과 한계.** Two-call design과 taxonomy의 효과를 분리한 ablation은 없다. User model도 persona instruction이 붙은 assistant-tuned LLM이다.

## §4 Family 3: Inception Prompting을 쓰는 role-playing agent(CAMEL)

**정의.** Role-playing은 role, output format, termination을 고정한 system prompt 아래에서 *AI user*가 한 번에 instruction 하나를 주고 *AI assistant*가 solution을 반환하는 two-agent protocol이다([[camel]], arXiv 2023-03; Baize와 UltraChat보다 앞선다).

**다루는 문제.** Chat agent 둘을 대화하도록 내버려 두면 role flipping, repeated instruction, “flake” reply, infinite loop가 나타났다(§4.1).

**방법.** Task-specifier prompt가 human-given task를 specific하게 만든다. 그 뒤 message history `M_t = {(I_0,S_0), …, (I_t,S_t)}`에서 user는 `I_{t+1} = U(M_t)`를 만들고 assistant는 `S_{t+1} = A(M_t, I_{t+1})`를 만들며 pair를 append한다(§3.1 Eq. 1–4). `I`는 instruction, `S`는 solution이다. User system prompt 원문(Fig. 2):

```
Never forget you are a <USER_ROLE> and I am a <ASSISTANT_ROLE>.
Never flip roles! You will always instruct me.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: <TASK>. Never forget our task!
You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:
1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>
2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None
...
You should instruct me not ask me questions.
...
When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task.
```

Assistant prompt는 answer가 `Solution:`으로 시작하고 `Next request.`로 끝나도록 요구한다. Conversation은 condition 다섯 개 중 하나에서 stop한다. 3 round 동안 instruction 없음, assistant가 instruction을 줌, end-of-task token, token limit, message 40개다(§4.1).

**증거.** AI Society는 assistant role 50개 × user role 50개 × task 10개 = conversation 25,000개다(§4.1). Conversation은 주로 “Assistant Instruct”(role reversal)로 끝나며 message cap 40개에 도달하는 경우는 드물다(App. J, Fig. 8). AI Society task 100개에서 human은 GPT-4가 summarize한 role-play solution을 single-shot gpt-3.5-turbo보다 76.3% vote에서 선호했고, 반대는 10.4%였다(Table 1). AI Society, Code, Math, Science를 순차적으로 fine-tune한 LLaMA-7B는 LLaMA-7B와 비교한 GPT-4 judge question 120개를 모두 이겼다(Table 2). CAMEL-7B의 HumanEval pass@1은 LLaMA-7B의 10.5 대비 14.0이었다(Table 3). 13B에서 fine-tuning은 MMLU/TruthfulQA는 올렸지만 ARC-C(`56.2 → 55.6`)와 HellaSwag(`80.9 → 79.3`)은 낮췄다(App. R, Table 7).

**조건과 한계.** End-of-task token은 user agent의 judgment만 기록한다. Paper는 task completion, cross-turn dependency, turn-length distribution을 check하지 않는다. “instruct, not ask” rule은 design상 user question/clarification을 제외하므로 user side는 speech act 하나를 다룬다(Interpretation).

**의미.** CAMEL data는 domain 전반의 instruction-solution exchange를 훈련한다. Underspecified user나 question을 하는 user를 model하지 않는다. §9는 여러 turn에 걸쳐 드러나는 request가 average score를 39% 낮춘다고 보고한다.

## §5 Grounded dialogue: ProsocialDialog(2022-05)와 SODA(2022-12)

**정의.** Grounded dialogue synthesis는 dialogue가 그 content에 관한 것이 되도록 external source content(social situation, commonsense triple, document)를 generation prompt에 넣는다. 이 course는 style만 바꾸는 persona/format instruction과 달리 이런 content를 *information anchor*라고 부른다(Interpretation; qa.md Q9에서 전개).

**ProsocialDialog.** [[prosocial-dialog]]은 unsafe user utterance를 무시하거나 동의하는 dialogue agent를 다룬다(Abstract). 방법(§3.2, App. A): (1) Social Chemistry(36k), ETHICS(9.7k), SBIC(12k)에서 situation을 가져온다. (2) Few-shot GPT-3가 situation을 problematic first-person utterance로 바꾼다. (3) Crowdworker가 rule-of-thumb(RoT)을 선택하거나 쓰고, 그에 grounded된 feedback을 작성한다. (4) GPT-3가 problematic side를 계속하고 worker가 다시 response하며 최대 6 turn까지 간다. (5) Worker 세 명이 GPT-3 utterance마다 safety label을 붙인다. 결과는 dialogue 58,137개, utterance 331,362개, unique RoT 160,295개, safety annotation 497,043개다(§3.4). Canary(T5-large)는 safety label/RoT를 생성하고 Prost(2.7B)는 RoT/response를 생성한다. Test example 400개에서 GPT-3와 비교하면 human은 prosociality에서 Prost(RoT & Response)를 63.4% 대 9.3%로 선호했다(Table 4). 저자들은 ProsocialDialog만으로 training하면 “can result in a negativity-prone chatbot”이 될 수 있다고 경고하고 Prost에는 casual-dialogue dataset 여섯 개와 `9:3:3:3:3:3:1`로 mix한다(§9, App. B.2).

**SODA(CO3 framework).** [[soda-commonsense-grounding]]은 context 없이 dialogue를 sample하면 “without contexts results in dull conversations”한다는 발견을 다룬다(§1). 방법은 다음과 같다.

1. Atomic10x에서 social x-relation triple을 retrieve한다. 예: `Head: PersonX moves a step closer to the goal; Relation: xNeed; Tail: to take the first step`(§2.2).
2. Relation template로 sentence로 바꾼다. 예: xNeed → `PersonX [Tail in past tense]. [Head].`이며 PersonX는 common US name으로 교체한다(§2.3, Table 8).
3. text-davinci-002에 prompt한다. `[sentence-form commonsense] Rewrite this story with more specific details in two or three sentences:`(App. A.1).
4. Triple이 person 하나면 second speaker를 infer한 뒤 first speaker를 prefix로 대화를 생성한다(§2.4).
5. Filter: erroneous pattern(6.3%), 4 turn 미만 또는 20 turn 초과(5.7%), speaker 2명 초과(11.3%), non-human speaker(5.6%), Canary “needs intervention”(4.3%), Rewire API toxicity 0.5 초과(약 1%), head event가 implied되었는지를 보는 GPT-3.5 check(§3.1).

**계산 예(filter yield).** SODA는 generated conversation 약 2.2 million 중 68.9%가 남아 1,486,896개가 된다고 말한다. 정확히 2,200,000개라면 yield는 `1,486,896 / 2,200,000 = 67.6%`다. 적은 68.9%는 initial conversation이 약 `1,486,896 / 0.689 ≈ 2,158,000`개였음을 뜻한다. “2.2 million”은 반올림한 값이다(derived).

**증거.** Head-to-head human evaluation에서 SODA dialogue는 “The following is a long in-depth conversation between two people.” prompt로 GPT-3.5에서 sample한 dialogue 100개보다 specificity, interestingness, 다른 criterion에서 선호됐다(natural flow를 제외하면 모두 significant; §3.3, Fig. 3). Narrative를 30%, role instruction을 50% 확률로 drop한 SODA + ProsocialDialog training COSMO-3B는 unseen DailyDialog set에서 BlenderBot-3B보다 overall comparison의 72%에서, human-written DailyDialog response보다 55%에서 선호됐다(§4, §5.1 Table 5). 결과는 test example 100개와 judge 3명을 쓴 single study다.

**조건과 한계.** 두 dataset 모두 social chit-chat 또는 safety feedback을 model하며 information-seeking assistant use는 model하지 않는다. SODA의 grounding source도 social situation만 다루며 physical/event-centered relation은 future work로 남긴다(§2.2 fn. 3).

## §6 Human-written reference: OpenAssistant conversation tree

**정의.** Conversation tree는 root prompt, prompt마다 assistant reply 여러 개, reply마다 user follow-up 여러 개, sibling 사이의 ranking을 저장한다. Thread는 root에서 node까지의 모든 path다([[openassistant]] §2, arXiv 2023-04).

**증거.** OASST1에는 message 161,443개(prompter 91,829개, assistant 69,614개), tree 66,497개(이 중 complete 10,968개), language 35개에서 온 volunteer 13,500명 초과, quality rating 461,292개가 있다(§4). Message 8,576개는 synthetic이다(§4). Current collection setting은 tree depth를 5, child를 2로 cap하며 목표 tree size는 message 9개다(App. G, Table 5). WildChat의 re-measurement는 conversation당 turn 2.34개, user token 33.41개, assistant token 211.76개(Llama-2 tokenizer)를 제시한다([[wildchat-statistics]] Table 1). OASST1 SFT는 Falcon-40B의 lm-eval-harness average를 72.29에서 74.04로 올렸지만 Pythia-12B에서는 60.33에서 60.28로 올리지 못했다(Table 1).

**조건과 한계.** OASST는 prompter에게 typo와 공지되지 않은 topic change를 피하라고 하는 guideline을 따르는 volunteer가 작성한다(App. A). Annotator의 89.1%는 자신을 male로 identify한다(§7). 이는 human-written assistant dialogue의 reference이지 unconstrained user behaviour의 reference는 아니다.

**의미.** Sibling structure는 같은 context에서 ranked alternative를 제공한다. 이는 SFT data보다 preference data다(Interpretation; paper는 이 ranking으로 reward model을 train한다, App. H). 여기서 얻는 preference pair 수는 보고하지 않는다.

## §7 Conditioning 축: persona, scenario, system prompt

**정의.** *Conditioning 축*은 dialogue마다 sample하여 generation prompt에 넣는 variable이다. Persona, scenario/intent, system prompt, topic, document가 이에 해당한다.

**방법과 증거.**

- **Persona.** [[persona-hub]]는 LLM에 RedPajama v2 web text를 누가 읽거나 쓸 가능성이 높은지 묻고, relationship으로 expand한 뒤 cosine similarity 0.9에서 deduplicate하여 persona 1,015,863,523개를 만든다(§2). Generated problem은 persona끼리의 similarity보다 서로 덜 similar하며, topic constraint를 추가하면 problem similarity가 오른다(§4.1.2, Fig. 10). Tülu 3의 persona dataset에는 prompt 29,980개인 Persona IF가 있다(Table 7). Tülu 3 8B SFT에서 모든 persona dataset을 빼면 IFEval은 72.8에서 53.6으로 낮아지지만 unseen IFEval-OOD는 17.6에서 18.0으로 간다([[modern-mix-composition]]; [[tulu-3]] Tables 10, 32). 결과는 single study다. Gain은 development benchmark에서 생겼고 held-out constraint benchmark는 움직이지 않았다.
- **System prompt.** MagPie-Ultra는 Llama-3.1-405B-Instruct-FP8을 사용한 Magpie-style generation에 system prompt를 더하고 three-turn conversation을 만든다. Llama-3.1-8B-Instruct, Llama-Guard-3-8B, ArmoRM score, gte-large embedding deduplication으로 filter한다([[modern-mix-composition]], SmolLM2 §5.1.1). SmolTalk은 model이 “to make the model support a variety of system prompt formats”하도록 SystemChats2.0 sample을 더한다(SmolTalk dataset card에는 30k, SmolLM2 Table 9에는 35.9k). System-prompt variable만 분리한 ablation은 없다.
- **Scenario와 intent.** 이 library의 source는 user simulator용 scenario-first generation을 측정하지 않는다. Qa.md Q13–Q14의 two-stage design(scenario를 sample/filter한 뒤 generate)은 course proposal이다(Interpretation). Evaluation은 ch-29d에 있다.

**측정한 formulaic user turn.** Exchange가 두 개 이상인 Capybara row에서 second user turn은 row 10,681개 중 1,787개(16.7%)가 “Considering”으로, 1,090개(10.2%)가 “Given”으로 시작한다([[capybara]], derived). User turn의 first word를 세는 일은 model call이 필요 없으며 template 하나를 반복하는 user generator의 check다.

**의미.** Persona/system prompt는 생성되는 내용을 바꾸지만, Tülu 3 결과는 training data constraint와 닮은 benchmark에서 19.2 point gain이 unseen constraint에서는 0.4 point 변화와 공존할 수 있음을 보인다. 둘 다 보고하라.

## §8 일반화 목표: real-user distribution의 coverage

**정의.** Conversation data에서 *coverage*는 real user의 prompt, intent, language, turn structure가 training에 represent된다는 뜻이다. [[wildchat]] 또는 [[lmsys-chat-1m]] 같은 real-traffic reference에 대조해 측정한다.

**Reference corpus.** LMSYS-Chat-1M에는 April–August 2023에 collect한 user 210,479명의 conversation 1,000,000개가 있고 language는 154개, model은 25개다. 저자들은 sample한 English conversation 100K개에서 question 대다수가 coding/software와 관련됨을 보고한다(cluster 20개 중 1, 2, 6, 16, 18)(§2, §3.1–3.2, Table 1). WildChat은 IP address 204,736개에서 나온 conversation 1,039,785개, language 68개가 있다. Turn의 53%는 English이며 user turn의 10.46%는 Detoxify 또는 OpenAI Moderation API가 toxic으로 flag한다(§2–4) ([[wildchat-statistics]]).

**측정 1: cross-dataset NLL.** WildChat은 dataset A의 first-turn prompt로 Llama-2 7B를 fine-tune하고 dataset B에서 average negative log-likelihood(NLL)를 보고한다. `NLL(B | θ_A) = −(1/N) Σ log p_θA(x_i)`이며, `θ_A`는 A로 train한 weight, `x_i`는 B prompt의 token, `N`은 token count다(§3, Fig. 3; “average NLLs”로부터 per-token averaging을 가정). 계산 예: Alpaca-trained model은 WildChat prompt에서 10.87이고 WildChat-trained model은 같은 prompt에서 5.18, OpenAssistant prompt에서 3.28이다. OpenAssistant-trained model은 OpenAssistant prompt에서 3.09다(Fig. 3). WildChat-trained model은 각 다른 dataset에서 해당 dataset의 own model과 NLL 1.04 이내다(Alpaca 3.28 vs 2.24, Dolly 3.29 vs 2.74, OpenAssistant 3.28 vs 3.09, ShareGPT 5.91 vs 5.23). 반면 Alpaca-trained model은 OpenAssistant/ShareGPT/WildChat prompt에서 8.42, 11.11, 10.87이다.

**측정 2: embedding overlap.** WildChat은 dataset마다 first-turn prompt 10,000개를 text-embedding-ada-002로 embed하고 t-SNE에서 WildChat이 다른 dataset과 overlap하면서 추가 area도 cover함을 보인다(§3, Fig. 4). Synthetic corpus에서는 dataset A 대신 synthetic user turn을 넣어 같은 procedure를 적용할 수 있다(course proposal).

**측정 3: real traffic에서 나온 held-out suite.** Arena-Hard-200은 benchmarking potential에서 GPT-3.5-Turbo/Claude-2/GPT-4가 모두 9 이상을 준 Chatbot Arena prompt 200개다. GPT-3.5-Turbo가 8 초과로 score한 prompt 50개에서는 GPT-4가 user vote의 52%에서 GPT-3.5-Turbo를 이겼고, 2 미만으로 score한 prompt 50개에서는 22%였다(§4.4, Fig. 5. Prose는 이를 Table 5라고 부름) ([[lmsys-chat-1m]]). WildChat은 WildBench용 conversation 14,743개를 reserve한다(§2). WildBench construction은 이 library의 card가 다루지 않는다.

**Realistic style이 capability가 아니라는 증거.** Crowdworker는 imitation-model output 약 70%를 ChatGPT와 같거나 더 좋다고 평가했다. 그러나 broad imitation data(ShareGPT-Mix)는 13B에서 Natural Questions accuracy를 20에서 15로 낮췄고 targeted NQ-synthetic data는 27로 올렸다([[false-promise-imitating-proprietary-llms]] Fig. 1, Table 1). Imitation data가 20M에서 150M token으로 늘자 ChatGPT가 list를 쓸 때 list를 쓰는 output의 share는 50%에서 81%로 올랐다(Table 2). 결과는 LLaMA 7B/13B의 single study다.

**Real-user prompt가 broad chat에 도움이 된다는 증거.** Tülu 3 8B SFT에서 WildChat prompt 100,000개를 제거하면 average는 60.1에서 58.9, AlpacaEval 2는 12.4에서 7.5로 낮아진다. Safety는 93.1에서 95.2로, IFEval-OOD는 17.6에서 20.8로 오른다([[tulu-3]] Tables 10, 32). 결과는 single study다. Real-user prompt를 포함하면 chat metric은 4.9 point, average는 1.2 올랐으며 safety는 2.1, IFEval-OOD는 3.2 낮아졌다.

## §9 Multi-turn degradation: current model이 실패하는 것

**Lost in Conversation.** [[lost-in-multi-turn]] (arXiv 2025-05)은 task 여섯 개(code, database, action, math, data-to-text, summary)의 fully specified instruction을 가져와 *shard*로 나눈다. 첫 shard는 intent를 말하고 이후 shard마다 requirement 하나를 더한다. GPT-4o-mini user simulator가 turn마다 shard를 최대 하나만 공개한다(§3.1–3.2). Setting은 FULL(original single turn), CONCAT(all shard를 한 turn에), SHARDED(turn마다 shard 하나)다. Score `S = {S_i}`가 0–100이고 simulation `N = 10`이면:

```
P = (1/N) Σ S_i          A = percentile_90(S)          U = percentile_90(S) − percentile_10(S)
```

`P`는 average performance, `A`는 aptitude(best-case score), `U`는 unreliability(good/bad run 사이 spread)다(§4.2).

**계산 예시.** Sorted score 10개에서 nearest-rank percentile을 쓰자(90th는 rank `⌈0.9·10⌉ = 9`, 10th는 `⌈0.1·10⌉ = 1`; paper는 interpolation을 밝히지 않음). FULL run `[70,80,80,80,90,90,90,90,100,100]`은 `P = 87`, `A = 100`, `U = 100 − 70 = 30`이다. SHARDED run `[0,10,30,40,60,80,90,90,100,100]`은 `P = 60`, `A = 100`, `U = 100 − 0 = 100`이다. Average performance는 31% 낮아지지만 aptitude는 바뀌지 않는다. Loss는 reliability에 있다. [figures/aptitude-unreliability.html](figures/aptitude-unreliability.html)은 reader가 score 10개를 edit하고 다시 계산된 P/A/U를 보게 한다.

**증거.** LLM 15개와 simulated conversation 200,000개 초과에서 SHARDED는 FULL보다 average 39% 낮았다. CONCAT은 FULL의 average 95.1%였으므로 drop은 rephrasing 때문이 아니다(§5–6). FULL에서 SHARDED로 갈 때 aptitude는 average 16% 떨어지고 unreliability는 112% 올랐다(§6.2). Final recap turn을 더하거나 매 turn에 모든 shard를 반복하는 SNOWBALL은 loss 일부를 회복했다. SNOWBALL은 FULL-to-SHARDED deterioration을 15–20% 완화했다(§7.1). GPT-4o-mini/GPT-4o에서 user/assistant temperature를 0으로 설정해도 SHARDED unreliability는 약 30 point로 남았다(§7.2, Table 3). 저자들은 failure를 early assumption과 premature answer attempt 탓으로 돌린다(Abstract). 결과는 simulated user를 쓴 single study다.

**MT-Eval.** [[mt-eval]] (arXiv 2024-01)은 LMSYS-Chat-1M에서 recollection, expansion, refinement, follow-up pattern 네 개를 정의한다. Dialogue 168개, turn 1,170개, dialogue당 6.96 turn이 있고 follow-up을 제외한 각 query의 single-turn version도 있다(§1, §3, Table 1). Table 3의 계산 예에서 Llama-2-chat-13B는 single-turn 7.55, multi-turn 5.47(`−2.08`, relative drop 27.5%)이고 GPT-4는 9.17, 8.84(`−0.33`, 3.6%)다. Llama-2-chat model은 single-turn에서는 Vicuna를 이기지만 multi-turn에서는 뒤진다(§4.4). Response 200개의 manual analysis에서 99개(49.5%)는 earlier instruction을 따르지 않았고 96개(48%)는 earlier turn의 error에 misdirect됐다. Gold history에 condition하면 recollection/refinement가 개선됐다(§4.5–4.6, Table 7).

**Controlled data 증거.** 1.7B에서 multi-turn MagPie-Pro-MT SFT는 MT-Bench 5.40, IFEval 31.66이고 single-turn MagPie-Pro는 4.31, 30.45다. MMLU-Pro는 11.97 vs 12.19다([[modern-mix-composition]], SmolLM2 Table 10). 결과는 epoch 하나의 single study이며 MT-Bench는 turn 두 개다.

**Data에 대한 의미.** §2–§5 pipeline은 seed question/task/narrative에서 dialogue를 열고 related follow-up을 더한다(위에 인용한 prompt의 Interpretation). 이 pipeline 중 어느 것도 underspecified/sharded request를 만들지 않으며 SHARDED setting에서 평가하지도 않는다. Sharded-instruction SFT data가 unreliability를 낮추는지는 이 chapter의 Open question이다. Ch-29b는 sharded/multi-turn instruction-following source로 만든 training data를 다룬다.

## Negative sample과 negative feedback

“Negative”의 의미 네 가지가 적용된다.

- **Negative marginal value(discarded).** SODA는 pattern, length, speaker, safety, grounding filter로 generated conversation의 31.1%를 버린다(§3.1). UltraChat은 polite user statement를 제거한다(§4.4). OASST는 spam-flagged/moderator-removed message와 그 child를 함께 삭제한다(§3.4). Baize SDF는 candidate 네 개를 score하고 best 하나만 남기며 나머지 셋은 쓰지 않는다(§4).
- **Content로서의 negative.** ProsocialDialog는 GPT-3의 problematic utterance를 context에 넣고 prosocial response에만 cross-entropy로 Prost를 train한다. 저자들은 problematic side로 train하는 것을 misuse라고 부른다(§4.2, §9). Canary safety label(예: Needs Intervention)도 ordinary target text로 train한다.
- **Conditioning으로서의 negative.** 이 chapter source는 사용하지 않는다.
- **Gradient로서의 negative.** 여기의 SFT pipeline은 사용하지 않는다. OASST sibling ranking은 pairwise reward-model loss `−E[log σ(r(x,y_w) − r(x,y_l))]`에 들어간다. `r`은 reward model, `x`는 context, `y_w`/`y_l`은 더 높고 낮은 rank의 reply다(App. H). Reward modeling은 ch-41(Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization)에서 다룬다.

**Diagnostic과 generality에 미치는 효과.** Safety-grounded dialogue data에서는 harm과 함께 over-refusal을 측정한다. [[xstest]]에서 guardrail system prompt는 Mistral-7B가 unsafe prompt를 full refusal하는 비율을 23.5%에서 87.5%로, safe prompt를 full refusal하는 비율을 0.8%에서 9.6%로 올렸다(Table 2). Tülu 3는 CoCoNot 같은 contrastive prompt가 safe prompt의 over-refusal 방지에 도움이 됐다고 보고한다([[tulu-3]] §4.2). Discard filter에는 SODA처럼 filter별 rejection share를 log한다. Filter가 user style 전체(예: short/rude message)를 제거하는 일이 보이게 한다.

## 레시피

| Model(정확한 release) | 크기 | Stage | 설정 | 값 | Source 위치 | 상태 | 이 값의 증거 |
|---|---|---|---|---|---|---|---|
| Baize v1 | 7B, 13B, 30B | distill-SFT | data; max input length | self-chat 111.5k(Quora 54,456; StackOverflow 57,046) + Alpaca 51,942; token 512 | arXiv:2304.01196v4 §3, §5, Table 2 ([[baize-construction-recipe]]) | verified 2026-09-14 | ablation 보고 없음 |
| Baize v1.5 | 7B, 13B | distill-SFT | AI-turn generation; loss masking | AI turn을 한 번에 하나의 call로 regenerate; AI response만 loss | arXiv:2304.01196v4 §3, §4 | verified 2026-09-14 | v1 data와의 ablation 보고 없음 |
| Baize v2 | 7B, 13B | SFT(SDF) | candidate; scorer; target | Quora question당 4개; ChatGPT score 1–100; best-ranked | arXiv:2304.01196v4 §4, App. C | verified 2026-09-14 | Fig. 3: 7B 87% → 90%, 13B 89% → 92% |
| UltraLLaMA | 13B | distill-SFT | data; sequence length; loss; batch | UltraChat dialogue 1,468,352개; `≤2,048` token으로 split; response만; A100 128개에서 total batch 512 | arXiv:2305.14233v1 §4.6, Table 5 ([[ultrachat-pipeline]]) | verified 2026-09-15 | ablation 보고 없음 |
| CAMEL-7B | 7B | distill-SFT | epoch; LR; schedule | 3; `2e-5`; cosine, warmup ratio 0.04 | arXiv:2303.17760v2 App. K.2 Table 5 ([[camel]]) | verified 2026-09-14 | ablation 보고 없음 |
| COSMO-3B / 11B | 3B, 11B | SFT | data; input dropout | SODA + ProsocialDialog; narrative 30% drop, role instruction 50% drop | arXiv:2212.10465v3 §4 ([[soda-commonsense-grounding]]) | verified 2026-09-15 | ablation 보고 없음; LR/step 미보고(body/appendix 확인) |
| Prost | 2.7B | SFT | mixture weight | ProsocialDialog : DailyDialog : TopicalChat : PersonaChat : WoW : EmpatheticDialogues : BST = `9:3:3:3:3:3:1` | arXiv:2205.12688v2 App. B.2 ([[prosocial-dialog]]) | verified 2026-09-14 | §9: ProsocialDialog만 train하면 “can result in a negativity-prone chatbot”; mixture ablation 없음 |
| WildLlama | 7B | SFT | data; batch; LR; length; epoch | 2023-07-16까지 WildChat; conversation 128개; `2e-5`; token 2,048; 3 | arXiv:2405.01470v1 §5 ([[wildchat-statistics]]) | verified 2026-09-15 | Table 9: MT-Bench average 6.35 vs Vicuna 6.13, Llama-2 Chat 6.26 |
| Tülu 3 8B SFT | 8B | SFT | WildChat share; Persona IF | prompt 939,344개 중 100,000(10.6%, derived); 29,980 | arXiv:2411.15124v5 Table 7 ([[tulu-3]]) | verified 2026-09-15 | Table 10: WildChat 없음 AE2 12.4 → 7.5; Persona 없음 IFEval 72.8 → 53.6 |
| Tülu 3 8B SFT | 8B | SFT | LR; batch; length; epoch | linear `5e-6`, warmup 0.03; 128; 4,096; 2 | arXiv:2411.15124v5 §4.3, Table 11 | verified 2026-09-15 | “found after a hyperparameter search”(§4.3); table는 여기 없음 |
| SmolLM2-1.7B SFT | 1.7B | SFT | data; epoch | SmolTalk sample 약 1.1M(MagPie-Ultra 431k, three-turn); 2 | arXiv:2502.02737 §5.1.1, App. F Tables 9–10 ([[modern-mix-composition]]) | verified 2026-09-15 | Table 10: SmolLM2-SFT IFEval 57.09, MT-Bench 6.11 |
| SmolTalk component | — | SFT | MagPie-Ultra size | 400K(dataset card) vs 431k(paper Table 9) | HF HuggingFaceTB/smoltalk card; arXiv:2502.02737 Table 9 ([[smol-talk]]) | conflict | Paper table은 trained mix를 설명하므로 431k를 우선 |
| Nous-Capybara-34B V1.9 | 34B | SFT | base; epoch | Yi-34B-200K; Capybara에서 3 | HF NousResearch/Nous-Capybara-34B@6beb706 README ([[capybara]]) | verified 2026-09-14 | ablation 보고 없음 |

**작은 범용 run의 시작점.** 1.7B model에서는 SmolLM2 recipe가 약 1.1M sample을 epoch 2로 train하며, 가장 큰 component는 405B generator가 만든 three-turn system-prompted conversation이다(SmolLM2 App. F). 이는 SmolLM2 자체 base model에서 수행했다. 8B model에서는 Tülu 3가 WildChat real prompt 100,000개를 포함한 prompt 939,344개로 LR `5e-6`, effective batch 128, token 4,096에서 epoch 2를 train한다(Tülu 3 §4.3, Table 7). 두 source 어느 것도 multi-turn data share를 ablate하지 않는다.

## Long-context 범위: short dialogue에서 long conversation까지

이 chapter의 corpus는 모두 짧다. Synthetic average는 2.58–7.6 turn(단위는 §1)에 걸쳐 있고, SODA는 20 turn 초과 conversation을 filter하며 CAMEL은 message 40개로 cap하고 OASST는 tree depth 5로 cap하며 UltraChat은 training에서 dialogue를 2,048 token으로 split한다. Real traffic에는 tail이 있다. WildChat conversation의 3.7%가 turn 10개를 넘는다(§3). MT-Eval은 multi-turn degradation을 낳는 factor로 relevant content까지의 distance와 error propagation을 찾는다(§9). Recollection task에서 GPT-4를 제외한 모든 model은 conversation이 길어질수록 initial global instruction을 따르기 더 어려워하고, refinement task에서는 model 11개 중 9개가 last six turn보다 first six turn에서 높게 score한다(§4.4, Table 4).

여기의 pipeline은 다음을 만들지 않는다. Summary/memory를 사이에 둔 시간적으로 분리된 session, later turn이 retrieve/update/abstain해야 하는 injected fact, same-distribution distractor, guardrail이 많은 long system prompt, 누적되는 tool output이다. 이는 Conversation Chronicles/LongMemEval 위에 세워진 ch-29b(Long-Conversation and Accumulating-Context Synthesis)의 주제다. Qa.md Q9/Q13/Q14의 anchor/scenario-first framework는 거기에서도 design input으로 이어진다. Qa.md Q18에서 length coverage를 lower priority로 둔 ranking은 short chat에만 적용된다.

## 일반화 관점

**(a) 폭을 넓히는 것.**

- Real-user prompt: Tülu 3 SFT에서 WildChat을 제거하면 average/AlpacaEval 2가 낮아진다([[tulu-3]] Table 10). WildChat-trained model은 다른 corpus 네 개에서 각 in-domain model의 NLL 1.04 이내에 머물지만 Alpaca-trained model은 8.42–11.11에 도달한다([[wildchat-statistics]] Fig. 3).
- External grounding: SODA contextualized dialogue는 human judgment에서 context-free sampling보다 specificity/interestingness가 좋고([[soda-commonsense-grounding]] §3.3), COSMO는 unseen DailyDialog로 generalize한다(Table 5).
- Multi-turn format 자체: MagPie-Pro-MT는 1.7B에서 MagPie-Pro보다 MT-Bench를 개선한다([[modern-mix-composition]], SmolLM2 Table 10).
- Skill에 맞춘 targeted data: Broad imitation이 못 한 곳에서 NQ-synthetic imitation은 NQ를 개선한다([[false-promise-imitating-proprietary-llms]] Table 1).

**(b) Narrowing 또는 forgetting을 일으키는 것.**

- Capability 없는 style: Broad ChatGPT imitation은 list use/authoritative tone을 올리고 NQ를 낮춘다([[false-promise-imitating-proprietary-llms]] Tables 1–2).
- Single-style safety data: ProsocialDialog만으로 training하면 negativity-prone chatbot이 나온다고 설명한다([[prosocial-dialog]] §9).
- Broad benchmark에서 dialogue SFT: CAMEL 13B는 ARC-C/HellaSwag point를 잃고([[camel]] App. R), Baize-v2-13B도 HellaSwag point를 잃는다([[baize-construction]] Table 7).
- Development-benchmark fit: Persona IF data는 IFEval을 19.2 point, IFEval-OOD를 0.4 움직인다([[tulu-3]] Table 32).

**(c) 이 단계에서 측정하는 방법.**

- Distribution: WildChat/LMSYS-Chat-1M에 대조한 unit-matched turn count와 same-tokenizer length(§1), cross-dataset NLL/embedding overlap(§8), user turn first-word count(§7).
- Held-out multi-turn behaviour: MT-Eval single-turn vs multi-turn gap, P/A/U가 있는 FULL vs CONCAT vs SHARDED(§9), real traffic에서 선택한 Arena-Hard-style prompt(§8).
- Retention: Dialogue SFT 전후의 broad benchmark set(MMLU, ARC, HellaSwag, TruthfulQA), IFEval 옆 unseen-constraint suite.
- 알려진 measurement error: Judge는 긴 response/first position을 선호([[baize-construction]] Limitations)하고 crowd rater는 confident style에 현혹된다([[false-promise-imitating-proprietary-llms]] §4.4). MT-Bench는 turn 두 개다. ShareGPT/WildChat/LMSYS Chat 같은 public chat log는 benchmark test set과 overlap할 가능성이 높다([[tulu-3]] App. B.2).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| Unit이 다른 turn count 비교 | Synthetic corpus가 real traffic보다 2× 길어 보임 | Utterance/exchange로 sample 하나를 다시 센다(SODA 7.6 vs 3.6, §1) |
| Assistant call이 user turn 작성 | User turn이 다음 answer가 cover할 정확한 것을 묻고 user length가 assistant length와 가까움 | WildChat(295.58 / 441.34)에 대조한 same-tokenizer user/assistant length ratio |
| User model의 role exchange | User turn이 answer 또는 “How can I help?”를 포함 | User turn classifier/regex; UltraChat §4.4가 이 failure를 보고 |
| Template을 반복하는 follow-up | Opening word 하나가 second user turn을 지배 | First-word histogram(Capybara: “Considering” 16.7%) |
| End-of-task token을 success로 취급 | Unsolved task인데 dialogue가 done으로 mark됨 | Completion check를 분리; CAMEL은 verify하지 않음(§4.1) |
| Crowd/judge preference만으로 data 판단 | Preference는 높지만 NQ/MMLU는 flat 또는 하락 | SFT 전후 held-out factual/broad benchmark |
| Fully specified first message만 사용 | Underspecified request에 model이 즉시 answer | U가 있는 FULL vs SHARDED evaluation(§9) |
| Over-refusal eval 없는 safety dialogue data | Unsafe와 vocabulary가 겹치는 safe prompt를 refusal | XSTest safe-prompt refusal rate |
| Data가 target한 benchmark만 보고 | IFEval은 오르지만 unseen constraint suite는 flat | IFEval을 IFEval-OOD 또는 IFBench와 pair |
| Dataset card에서 composition 인용 | Number가 trained mix와 다름 | Card와 paper table 비교(SmolTalk 400K vs 431k) |

## 이해를 확인해 보세요

1. SODA는 dialogue당 turn 7.6, UltraChat은 SODA에 3.6을 보고한다. 둘 다 맞을 수 있는 이유와 이를 해결하지 않고 synthetic corpus를 WildChat과 비교할 때 생기는 error를 설명하라.
2. Baize v1.5는 AI turn을 one call at a time으로 regenerate했다. Table 2의 어떤 변화가 이를 유발했고, 양쪽 side를 쓰는 single call은 왜 그 pattern을 만들기 쉬운가?
3. CAMEL AI Society conversation은 `<CAMEL_TASK_DONE>`보다 role reversal로 더 자주 끝난다. SFT data selection에서 end token을 success label로 쓰는 것이 무엇을 뜻하는가?
4. Tülu 3 ablation에서 persona data를 빼면 IFEval은 19.2 point 낮아지고 IFEval-OOD는 낮아지지 않는다. Causal explanation 두 개와 이를 분리할 experiment를 제시하라.
5. Lost in Conversation은 aptitude 16% 하락, unreliability 112% 상승을 찾는다. Fully specified first message가 있는 SFT data를 더 넣어도 이것을 고치기 어려운 이유는 무엇이며, 어떤 data property가 도움이 될 수 있는가?
6. WildChat-trained model의 WildChat prompt NLL은 5.18, Alpaca-trained model은 10.87이다. 이것이 측정하는 것과 WildChat이 capability에 유용함을 과장할 수 있는 이유 하나를 설명하라.
7. False Promise paper는 imitation data가 accuracy보다 style을 더 빨리 바꾼다고 찾는다. 이 chapter lens의 어느 measurement가 synthetic multi-turn corpus에서도 같은 failure를 발견할 수 있으며, GPT-4 judge 하나가 이를 발견하지 못하는 이유는 무엇인가?
8. ProsocialDialog는 prosocial speaker turn에만 train한다. Problematic speaker text를 target이 아니라 context로 쓰는 이유와 반대 선택이 일으킬 failure를 설명하라.

## 연결

- 이전: ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data.
- 다음: ch-26 — Tool and Function-Calling Data(tool call은 이 장의 multi-turn structure에 typed argument/executable result를 더한다).
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix(§2–§5가 instantiate하는 generic loop).
- ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing(§7의 persona conditioning).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection; ch-23 — Model Collapse and Verification of Synthetic Data(§5/§8의 filtering과 real-data anchoring).
- ch-29b — Long-Conversation and Accumulating-Context Synthesis(이 장 범위 밖의 long/multi-session conversation).
- ch-29d — User Simulators, Trajectory Verification, and Failed Trajectories(qa.md Q12의 customer-simulator question을 포함한 user-simulator realism).
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks; ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate(이 dialogue를 어떻게 train하는가).
- ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming(§5 data의 over-refusal 측정).

## 출처

- [[baize-construction]] — self-chat template, corpus statistic, SDF, GPT-4 score, leaderboard change. [[baize-construction-recipe]] — Baize training setting.
- [[ultrachat-pipeline]] — two-call design, sector construction, role-exchange observation, Table 5 statistic, UltraLLaMA training. [[ultrachat-two-model-protocol]] — paper/README number를 조화시키는 excerpt.
- [[camel]] — Inception Prompt, termination condition, AI Society size, agent/transfer evaluation, CAMEL-7B recipe.
- [[prosocial-dialog]] — collection pipeline, statistic, Canary/Prost, human evaluation, mixture warning.
- [[soda]] — SODA card(unverified card; 이 chapter의 number는 excerpt에서 옴). [[soda-commonsense-grounding]] — CO3 step, filter/rate, contextualization ablation, COSMO evaluation.
- [[openassistant]] — tree format, composition, collection setting, guideline, SFT result.
- [[persona-hub]] — persona construction/similarity finding.
- [[capybara]] — released-file turn statistic과 follow-up phrasing count.
- [[smol-talk]] — SmolTalk card(composition conflict를 기록). [[modern-mix-composition]] — SmolLM2 Tables 9–10/Tülu 3 Tables 10, 32 excerpt.
- [[tulu-3]] — WildChat/persona ablation, SFT hyperparameter, CoCoNot note.
- [[wildchat]] — WildChat card. [[wildchat-statistics]] — Table 1 excerpt, turn distribution, coverage NLL, embedding analysis, WildLlama.
- [[lmsys-chat-1m]] — statistic/topic cluster/Arena-Hard-200 excerpt.
- [[lost-in-multi-turn]] — sharded simulation, P/A/U metric, result, recap/temperature experiment excerpt.
- [[mt-eval]] — interaction pattern 네 개, single-vs-multi-turn score, error analysis excerpt.
- [[false-promise-imitating-proprietary-llms]] — crowd rating, NQ result, style convergence excerpt.
- [[xstest]] — safety dialogue data의 over-refusal 측정.
- [[agentinstruct]] — suggester-editor refinement(correction 12).
