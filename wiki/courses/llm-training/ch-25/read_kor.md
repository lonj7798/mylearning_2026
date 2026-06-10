<!-- chapter: ch-25
     track: synthetic
     title: Modality — Multi-Turn Conversation Synthesis
     sources: [[baize]], [[baize-construction]], [[ultrachat-construction]], [[ultrachat-pipeline]], [[camel]], [[soda]], [[prosocial-dialog]], [[openassistant]], [[system-prompt-diversity]], [[wildchat]], [[capybara]], [[smol-talk]]
     figures: figures/turn-distribution.html
-->

# 25장 — 모달리티: 멀티턴 대화 합성

> **핵심 통찰.** 대화는 추론 시점에 원하는 정답 분포, 즉 실제 사용자가 실제 어시스턴트와 여러 턴에 걸쳐 말하는 분포를 어떤 단일 LLM도 기본적으로 만들어 내지 못하는 첫 데이터 모달리티다. 모든 멀티턴 합성 파이프라인은 이 간극을 메우기 위해 설계된 우회로다. 우회로는 두 번째 화자를 어떻게 만들어 내는지에 따라 세 계열로 나뉜다. **self-chat**(한 모델이 양쪽을 모두 연기; Baize), **two-model role-play**(섹터 또는 역할 프롬프트 아래 별도의 user-LLM과 assistant-LLM; UltraChat, CAMEL), **grounded narration**(상식 triple, rule-of-thumb, 실제 passage에서 장면을 시드; SODA, Prosocial-Dialog, UltraChat-Sector-3). **OpenAssistant**는 각 합성 계열을 재는 실제 인간 기준선이다. 턴 수의 긴 꼬리, 사용자 register 분산, branching factor는 합성 파이프라인이 여전히 놓치는 지표다.
>
> **가이드라인.** 오늘날 멀티턴 SFT 데이터에는 하나의 레시피를 고르지 말고 명시적 slice로 mix를 구성하라. 저렴한 breadth에는 self-chat, 주제 coverage에는 taxonomy 아래 two-model role-play, task-oriented dialogue에는 role-pair inception prompting, 감정 register에는 commonsense-grounded dialogues, 현실성에는 OASST/WildChat anchor slice를 둔다. generation을 실행하기 전에 taxonomy 크기를 정하라. taxonomy가 다양성 레버이고 teacher는 renderer일 뿐이다.

---

## 이 장이 필요한 이유

Ch-24는 single-turn instruction data에 대해 일반적인 synthetic-pipeline loop(generate → filter → dedup → verify → select → mix)를 넘겨주었다. 대화는 그 loop의 한 가정을 깨뜨린다. 데이터 단위가 `(prompt, response)` 쌍이 아니라 **교대 턴으로 이뤄진 트리**이며, 그 통계량들, 즉 턴 수, 턴별 토큰 길이, topic drift, persona consistency는 각각 독립적인 설계 주의가 필요하다. Self-Instruct를 단순히 K-turn으로 확장하면, 지나치게 유창한 사용자, 빠진 감정 register, 공식화된 turn-taking 때문에 fine-tune된 어시스턴트가 눈에 띄게 합성적으로 보이는 dialogue가 나온다.

대화 설계를 좌우하는 속성은 세 가지다.

1. **사용자 턴이 어시스턴트 턴보다 어렵다.** 어시스턴트 스타일은 teacher LLM이 alignment를 통해 생성하도록 학습된 것이다. 사용자 스타일은 그렇지 않다. 여기의 모든 파이프라인은 부분적으로 "그럴듯한 사용자 턴을 어떻게 얻을 것인가?"에 대한 우회로다.
2. **멀티턴은 조합 폭발이 있지만 coherence의 자유도는 하나뿐이다.** 대부분의 턴 순서는 말이 되지 않는다. 합성은 topology를 제약해야 한다.
3. **Topic taxonomy, role pair, grounding source, system prompt는 각각 독립적인 다양성 축이다.** 현대적 mix는 이 네 축을 모두 쌓는다.

---

## 1. Baize — self-chat 원시형

2023년 봄. [[baize-construction]]에는 GPT-4 예산도, 인간 대화 로그도 없었기 때문에, 하나의 LLM이 한 번의 API call 안에서 양쪽을 모두 role-play하도록 프롬프트했다. 정확한 template는 다음과 같다.

```
The following is a conversation between a human and an AI assistant.
[|Human|] <seed question>
[|AI|] <first ChatGPT response>
[|Human|] <ChatGPT playing user>
[|AI|] <ChatGPT playing AI>
...
```

ChatGPT는 종료 marker나 약 8-turn cap까지 양쪽을 계속 이어간다. 사후 parsing은 role marker 기준으로 나눈다. Seed pool: Quora 약 54K, StackOverflow 약 57K, Alpaca 약 52K, MedQuAD 약 47K → 총 111.5K dialogues. Median 4 turns, IQR 3–6, tail to 10. 턴당 평균 약 100 tokens. Dialogue당 API call 하나, 총비용 약 \$1,000.

하중을 받는 약점은 **user-realism gap**이다. Single-model self-chat은 "호기심 많고 말 잘하는 인간" 같은 사용자를 만든다. 실제 사용자는 갑작스럽고, 오타가 많고, 주제를 바꾼다. 순수 Baize로 SFT된 학생 모델은 유창한 사용자를 기대하도록 학습하고 실제 사용자를 받으면 성능이 떨어진다. Self-Distill with Feedback 확장(ChatGPT에게 학생 모델을 비평하도록 다시 prompt하고, 비평을 DPO pair로 사용)은 부분적으로 보상하지만 사용자 분포를 고치지는 못한다. Baize가 *증명한* 것은 합성 self-chat만으로도 멀티턴 capability가 전이된다는 점이다. 이전의 믿음은 실제 인간 dialogue가 필요하다는 것이었다. Baize-13B는 human-eval 비교의 58%에서 Alpaca-13B를 이겼고, self-chat을 이후 모든 파이프라인이 넘어야 하는 minimum-viable baseline으로 만들었다.

---

## 2. UltraChat — topic taxonomy 아래의 two-model role-play

[[ultrachat-construction]]은 Baize의 single call을 **턴당 두 번의 별도 ChatGPT-Turbo call**(user-prompt 하나, assistant-prompt 하나)로 바꾸고, seed pool을 **손으로 설계한 three-sector taxonomy**로 바꾼다. Taxonomy가 다양성 레버이고, two-model split은 대화적 마찰을 준다.

**Sector 1 — Questions about the World.** 30 meta-topics → 1,100+ subtopics → 각각 최대 10 seed questions, Turbo로 확장. 두 번째 branch: top 10K Wikidata entities × 5 meta-questions × (10 specifics + 20 related). 각 seed는 3–7-round dialogue로 굴러간다.

**Sector 2 — Writing and Creation.** 20 writing types(essay, poem, script, email, code, recipe, …) × type당 200 instructions. 80%는 details로 확장된다. 각각 2–4-round draft-and-revise dialogue가 된다.

**Sector 3 — Assistance on Existing Materials.** 약 100K C4 passages. 각각 최대 5 user questions. passage + question + template가 첫 user turn이다. 2–4 rounds의 grounded follow-up.

Two-model loop를 도식화하면 다음과 같다.

```
for round in range(R_s):                       # R_1 ∈ [3,7], R_{2,3} ∈ [2,4]
    assistant_turn = assistant_LLM(system=assistant_prompt_s, history)
    history.append(assistant_turn)
    user_turn = user_LLM(system=user_prompt_s, history)
    history.append(user_turn)
```

각 턴은 전체 history에 조건화된다. 별도의 call 두 개는 각 side가 실제로 방금 나온 토큰에 *반응*하도록 강제한다. Single-call self-chat은 모델이 양쪽을 너무 일관되게 계획하게 만든다. 출력: 약 1.5M dialogues(Q-world 약 600K, Writing 약 400K, Assistance 약 500K), HuggingFace preview list length 4–14로 sector round bound와 일치한다. UltraChat이 표준으로 만든 것: taxonomy-first diversity(generation 전에 열거, generic prompt에서 아무것도 기대하지 않음), 분리된 user/assistant generator(이후 거의 모든 파이프라인이 복사), family별 release로 downstream mix가 sector를 독립적으로 가중할 수 있게 함([[smol-talk]]은 일부 subset만 사용). Filter stage는 불투명하다. 실무자의 교훈은 filter heuristic보다 generation protocol이 더 잘 전이된다는 점이다.

---

## 3. CAMEL — Inception Prompting을 쓰는 role-play multi-agent synthesis

[[camel]]은 다른 질문을 던진다. 주제가 아니라 *roles*가 다양성 레버라면? `(assistant_role, user_role, domain)`의 50 × 50 × 20 grid에서 sample한다. unlock은 **Inception Prompting**이다. AI User의 system prompt는 논문과 reference implementation에서 그대로 다음과 같다.

```
Never forget you are a <user-role> and I am a <assistant-role>.
Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete the task: <task>.
You must instruct me based on my expertise and your needs to solve the task.

Give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
You should instruct me, not ask me questions.

Here is the format you must strictly follow:
Instruction: <YOUR INSTRUCTION>
Input:       <YOUR INPUT, or "None">

Do not add anything else other than your instruction and the optional input.

When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task.
```

AI Assistant prompt는 이것을 mirror한다. "나는 `Solution: <YOUR SOLUTION>` 형식으로 solution을 제공해야 하며, 당신에게 지시하는 쪽으로 role을 뒤집으면 안 된다." 하중을 받는 속성은 두 가지다. **role lock**("Never flip roles! Never instruct me!")은 첫 줄 constraint다. GPT-3.5는 그렇지 않으면 어시스턴트의 자연스러운 충동, 즉 clarification question을 묻는 쪽으로 돌아간다. 그리고 **termination protocol**이다. `<CAMEL_TASK_DONE>`은 user가 task가 실제로 해결되었을 때만 내는 single-token marker다. 이것은 **turn cap이 아니라 semantic termination**이다. Dialogue는 trivial하면 4턴, 어려우면 20턴까지 간다.

50 assistant roles(Accountant, Architect, Astronaut, Biologist, …)와 50 user roles(Entrepreneur, Graduate Student, Journalist, …) 자체도 GPT-3.5에게 common roles를 열거하라고 prompt하여 한 번 생성했다. 20 task domains per variant(Society, Code, Math, Science)와 교차된다. 통계: median 6–8 turns, tail to 20, 턴당 평균 약 50 tokens(엄격한 `Instruction:` / `Solution:` format 때문에 짧다). AI Society 약 1M dialogues, 2023 Turbo rate 기준 비용 약 \$5–10K. Inception-prompting pattern은 [[agentinstruct]], [[apigen-mt]], ToolACE로 이어진다. 명시적 role prompts와 termination marker가 있는 두 LLM을 보면 lineage는 여기로 돌아온다. Gotchas: 50×50은 진정으로 다른 dialogue 2,500개가 아니다(많은 role pair 품질이 낮다. Astronaut × Bartender가 전형적인 예다). 또 "As an accountant, I'd suggest…" 같은 formulaic leak는 downstream model이 unlearn해야 한다.

---

## 4. SODA와 Prosocial-Dialog — grounded synthesis (Yejin Choi group)

Topic(UltraChat)과 role(CAMEL)은 *무엇*과 *누구*에 조건화한다. **Grounding**은 *왜*에 조건화한다. 즉 dialogue가 감정적으로 말이 되게 만드는 scene, motive, norm이다.

**SODA**([[soda]])는 Atomic 10X에서 시작한다. 이는 약 10M records의 commonsense-triple database이며 `(PersonX event, relation, mental-state-or-reason)` 형태다. 예: `(PersonX fails the exam, xWant, to study harder)`, `(PersonX moves abroad, xEffect, feels lonely)`, `(PersonX surprises PersonY, xReact, feels proud)`. 세 단계:

```
Step 1 — triple sampling
  Sample 1.5M triples, balanced across {xWant, xNeed, xEffect, xReact}.

Step 2 — narrative generation
  Prompt GPT-3.5:
    "Given the triple (PersonX <event>, <relation>, <state>),
     write a 1-2 sentence narrative describing a social scene where
     this arises. Include a second person PersonY as interlocutor."

Step 3 — dialogue generation
  Prompt GPT-3.5:
    "Write a short conversation between PersonX and PersonY set in
     this scene. Each speaker alternates. 4 to 10 turns total."
```

Filter: scene-coherence, length ≥ 3 turns, toxicity classifier, persona-consistency(PersonX가 PersonX로 유지). 출력: 1.5M dialogues, median 7–8 turns, max 약 12, 턴당 평균 20 tokens(매우 짧다. 사실 Q&A가 아니라 수다스러운 사회적 exchange). 비용 약 \$10K. Triple이 없으면 GPT-3.5는 그럴듯하지만 *emotionally flat*한 대화를 만든다. Anchoring은 암묵적 감정 상태(`feels lonely`, `wants X because Y`)를 주입해 trivial하지 않은 turn content를 움직인다. SODA로 학습한 COSMO-3B는 parameter가 3분의 1인데도 natural / engaging / specific human-eval 차원에서 BlenderBot-3B를 이긴다.

**Prosocial-Dialog**([[prosocial-dialog]])는 grounding을 *safety*에 적용한다. 주장: 어려운 prompt를 거부하는 것은 안전하지도 유용하지도 않다. 사회적으로 grounded된 rule-of-thumb으로 건설적으로 engage하는 것이 안전하고 유용하다. Semi-synthetic, human-in-the-loop 방식이다. Crowd worker가 10개 harm categories(stereotypes, insults, self-harm, violence planning, misinformation, …)에 걸쳐 약 10K problematic prompts를 작성한다. Worker는 적용 가능한 rules-of-thumb, 즉 *"It's rude to mock someone's appearance."* 같은 짧은 윤리 지침을 식별한다. Dataset 전체에서 300+ unique RoTs가 나온다. GPT-3가 RoT에 grounded된 후보 assistant response를 draft하고, worker가 edit 및 rank하며, high-quality를 유지한다. Multi-turn extension은 back-and-forth editing으로 만든다. 58K dialogues, median 3 turns, 각 turn은 RoT로 tagged. CANARY-400M은 problematic prompts의 89%에 건설적으로 engage하며, BlenderBot-3B는 32%다. 이것은 refusal이 아니라 engage-and-redirect pattern이며, [[constitutional-ai]]의 직접적인 선행자다.

Yejin Choi lineage `self-instruct → SODA → Prosocial-Dialog`는 하나의 arc다. Model output에서 bootstrap하되, 항상 structured external signal(commonsense triple, norm)에 bootstrap을 *ground*한다. 순수 model-output bootstrap은 teacher의 style을 distill한다. Grounded bootstrap은 teacher가 스스로 만들지 않았을 정보를 주입한다.

---

## 5. OpenAssistant — 실제 인간 기준선

[[openassistant]]는 합성 파이프라인이 아니다. 이 장의 모든 파이프라인이 비교되는 dataset이며, 이 공간에서 근본적으로 다른 data structure를 가진 하나의 artifact다. 그것은 **conversation tree**다.

```
root (user prompt)
 ├── assistant_reply_A      [quality 4.2]
 │    ├── user_followup_A1
 │    │    ├── assistant_reply_A1a  [quality 3.8]
 │    │    └── assistant_reply_A1b  [quality 4.1]
 │    └── user_followup_A2
 │         └── assistant_reply_A2a  [quality 3.6]
 └── assistant_reply_B      [quality 3.4]
      └── user_followup_B1
           └── assistant_reply_B1a  [quality 3.0]
```

각 node는 parent와, 가능하면 many children을 가진다(턴마다 여러 candidate replies). 각 message는 여러 labeler가 단 quality/helpfulness/harmlessness label과 sibling set 내 rank를 carries한다. Conversation은 root-to-leaf path다. OASST1: 약 13,500 contributors, 161K messages, 10K fully-labeled trees, 35 languages, CC-BY 4.0. OASST2는 약 90K를 더 추가한다.

Tree를 flatten하면 branching-factor signal을 버린다. Tree를 유지하면 다음을 측정할 수 있다. **user-turn diversity**(parent assistant turn의 siblings 사이에서 follow-up이 얼마나 다양한가? Synthetic corpora는 teacher의 "가장 그럴듯한 다음 user turn"이 뾰족해서 sibling variance가 거의 0이다. OASST는 높다), **register variance**(real users에는 casual/formal/rude/confused registers가 포함된다. Baize/UltraChat users는 균일하게 polite하고 articulate하다), **path-length distribution**(OASST path는 power-law tail로 20+ turns까지 간다. Synthetic histogram은 대체로 hard cap을 가진 Gaussian이다), **typos + abrupt topic changes**(OASST 약 3% typo rate, Baize는 거의 0). 새로운 pipeline의 diagnostic: 같은 수의 dialogue를 생성한 뒤 이 네 통계를 OASST와 비교하라. 보통 user-diversity와 register-variance에서 gap이 발견된다.

OASST는 단독 SFT corpus로 쓰기에는 너무 작고(약 250K messages), contributor가 Western/English/technical 쪽으로 치우쳐 있다. 현대 mix에서의 역할은 synthetic majority의 stylistic narrowness를 상쇄하는 **realism anchor slice**(mix의 5–15%)다.

---

## 6. 장 전체의 turn-count와 token-length

Dataset card와 public preview에서 재구성했다. Interactive histogram은 `figures/turn-distribution.html`을 보라.

| Dataset | Dialogues | Median turns | Max turns | Avg tokens/turn | Speaker protocol | Grounding |
|---|---|---|---|---|---|---|
| [[baize]] | 111.5K | 4 | ~10 | ~100 | 1 LLM, 2 roles | seed question |
| UltraChat Q-world | ~600K | 6 | 14 | ~120 | 2 LLMs | topic taxonomy |
| UltraChat Writing | ~400K | 4 | ~8 | ~150 | 2 LLMs | writing type |
| UltraChat Assistance | ~500K | 4 | ~8 | ~160 | 2 LLMs | C4 passage |
| [[camel]] AI Society | ~1M | 7 | 20 | ~50 | 2 LLMs, role-pair | task spec |
| [[soda]] | 1.5M | 7 | 12 | ~20 | 2 speakers (same LLM) | Atomic triple |
| [[prosocial-dialog]] | 58K | 3 | ~6 | ~40 | user + assistant | rule-of-thumb |
| [[openassistant]] OASST1 | 10K trees / 161K msg | 4–6 path | 20+ | ~80 | human + human | none (free) |
| [[wildchat]] | ~1M | 3 | 50+ | ~60 (user short) | real user + ChatGPT | real task |

내재화할 만한 pattern은 세 가지다. **Synthetic median은 3–7에 몰려 있는 반면 real-human tail(OASST, WildChat)은 훨씬 길다.** 이 tail에서 follow-up quality가 test되고 synthetic pipeline은 여기서 진다. **SODA는 턴이 가장 짧다(약 20 tokens).** Atomic triple은 factual Q&A가 아니라 social chit-chat register를 끌어내기 때문이다. Baize/UltraChat turn은 Q&A / writing task라서 3–8배 길다. **CAMEL은 termination이 semantic이므로 dialogue가 가장 길다.** 어려운 task는 길게, 쉬운 task는 짧게 간다. 이것이 올바른 행동이며 turn-capped pipeline에는 없다.

---

## 7. System-prompt diversity — 2024년의 확장

2024년에 frontier는 topic 및 role diversity를 넘어 **system-prompt diversity**로 이동했다. [[system-prompt-diversity]]: 큰 persona pool에서 *서로 다른* system prompt 아래 각 dialogue를 sample한다. 표준 source: Persona-Hub(Ge 2024, web bio / about-me / LinkedIn-style profile에서 mined한 1B personas)와 SystemChat-2.0(Hartford 2024, roles, constraints — *"always answer in bullet points"* — styles, safety modes를 포괄하는 distinct system prompts가 붙은 7K dialogues). 경험적 발견: persona-conditioning은 topic-only conditioning보다 unique vocabulary를 최대 2×, persona-consistent follow-up을 3× 늘린다. Tülu-3와 Qwen 2.5 post-training에 통합되었고, topic-only-diverse SFT 대비 IFEval +5–10 points.

이것은 UltraChat을 대체하는 것이 아니라 *확장*한다. UltraChat의 taxonomy는 topic grid이고, persona-conditioning은 behavioral grid다. 두 축은 합성된다. Gotchas: persona-extraction bias(web-mined personas는 online professionals와 English speakers를 과대표집한다. 실제 사용자보다 좁은 population), 그리고 conditioning axis를 너무 많이 쌓기(role + style + mode + safety)는 teacher와 student 모두를 혼란스럽게 한다.

---

## 8. 파이프라인 구성 — 현대적 mix가 하는 일

현대 production SFT mix는 하나의 recipe를 고르지 않고 blend한다. 대비되는 두 예:

**[[capybara]] — compact composition.** 약 20K dialogues, 60%+ multi-turn. **Amplify-Instruct**: composed pool(Airoboros + Evol-Instruct + Orca + Vicuna + LessWrong posts + CamelAI)에서 seed를 sample하고, initial response를 생성하고, 그럴듯한 follow-up user turn을 합성한 뒤 2–6 rounds 반복한다. Seed composition이 diversity mechanism이고, amplification이 multi-turn mechanism이다. 의도적으로 작게 만든다. scale보다 quality다. Nous-Capybara-7B를 single consumer GPU에서 경쟁력 있는 MT-Bench 수준으로 훈련한다.

**[[smol-talk]] — 1M-scale composition.** 표준적인 2025 open recipe. Smol-Magpie-Ultra(400K, Llama-3.1-405B를 쓴 Magpie) + Smol-summarize(100K) + OpenHermes 2.5 subset(100K) + Smol-rewrite(50K) + MetaMathQA subset(50K) + Smol-constraints(36K) + **SystemChats 2.0(30K, multi-turn)** + NuminaMath-CoT + Self-Oss-Starcoder2. 명시적 multi-turn dialogue는 1M 중 약 30K, 즉 *3%*다. 현대 scale에서 multi-turn capability는 single-turn data 전반에 스며든 in-context-follow-through pattern에서 주로 학습된다. 명시적 dialogue slice는 이제 primary supervision이 아니라 *realism insurance*다. 이것이 2023년(UltraChat의 1.5M이 SFT mix를 지배)에서 2025년(multi-turn이 minority slice)으로의 구조적 변화다.

---

## 9. 실무자 checklist

1. **다양성 축을 먼저 정의하라.** Topic(UltraChat), role(CAMEL), commonsense(SODA), system prompt(SystemChat), seed composition(Capybara). 1–2개의 primary axis를 고른다.
2. **예산이 허락하면 사용자와 어시스턴트에 별도의 model call을 사용하라.** Single-call self-chat은 싸지만 너무 일관적이다.
3. **Turn cap보다 semantic termination을 선호하라.** CAMEL의 `<CAMEL_TASK_DONE>`은 dialogue length를 task hardness에 맞춘다. Hard cap은 tail을 눌러 버린다.
4. **실제 인간 anchor를 포함하라.** Synthetic이 재현하지 못하는 user-register variance를 주입하려면 5–15% OASST 또는 WildChat slice를 둔다.
5. **Turn-length distribution을 OASST와 비교해 측정하라.** 10턴 이후 tail이 평평하면 turn-capping이 너무 공격적이다. User turn이 모두 50 tokens보다 길면 synthetic user 학생은 추론 시점의 사용자를 보지 못한다.
6. **Post 단계에서 formulaic phrases를 제거하라.** `[|Human|]` markers, "As an accountant…", "PersonX said…"는 모두 leak된다. Regex strip이 표준이다.
7. **Safety에는 refusal보다 RoT-grounded engagement를 선호하라.** [[prosocial-dialog]]와 [[constitutional-ai]]는 모두 helpfulness *및* safety에서 pure-refusal을 이긴다.
8. **Persona stacking을 제한하라.** Dialogue당 conditioning axis는 최대 두세 개.

---

## 10. 다음 장이 기반으로 삼는 것

Ch-26은 모달리티를 **tool and function-calling data**로 좁힌다. 이는 한 턴이 API call이고 다음 턴이 그 result인 멀티턴 variant다. CAMEL Inception-prompting template는 ToolACE와 APIGen-MT에서 다시 나타난다. 이번에는 `<CAMEL_TASK_DONE>`이 아니라 executable verifier가 termination signal이다. 전환은 "topic으로 grounded된 dialogue"에서 "typed function schema + executable verifier로 grounded된 dialogue"로의 이동이다. 이 장의 모든 레버, 즉 taxonomy, role pair, grounding, termination은 tool-call-specific instantiation으로 다시 나타난다.

---

## Connections

- [[baize]] / [[baize-construction]] — Xu 2023; self-chat primitive; 111.5K dialogues; `[|Human|]` / `[|AI|]` prompt; median-4-turn distribution.
- [[ultrachat-construction]] / [[ultrachat-pipeline]] — Ding 2023; three-sector taxonomy; two-model role-play; 1.5M dialogues; preview lengths 4–14.
- [[camel]] — Li 2023 (KAUST); Inception Prompting; 50×50×20 role × domain grid; `<CAMEL_TASK_DONE>` termination; ~1M AI Society dialogues.
- [[soda]] — Kim 2023 (Yejin Choi); Atomic 10X triple → narrative → dialogue; 1.5M commonsense-grounded dialogues; COSMO-3B outcome.
- [[prosocial-dialog]] — Kim 2022 (Yejin Choi); 58K dialogues anchored to 300+ rules-of-thumb; precursor to Constitutional AI.
- [[openassistant]] — Köpf 2023 (LAION); tree-structured crowdsourced dialogue; OASST1 161K messages / 10K trees / 35 languages; reference real-human baseline.
- [[wildchat]] — Zhao 2024; ~1M real opt-in ChatGPT logs; realism anchor distinct from OASST.
- [[system-prompt-diversity]] — 2024+ class; Persona-Hub 1B personas; SystemChat-2.0; extends taxonomy with behavioral axis.
- [[capybara]] — LDJ / Nous 2023; Amplify-Instruct seed composition; 20K compact multi-turn.
- [[smol-talk]] — HuggingFace 2024/2025; 1M SmolLM2 SFT mix; 30K SystemChats multi-turn slice.
- [[ch-24]] — 이 장이 conversation에 대해 instantiate하는 synthetic-pipeline design-pattern framing.
- [[ch-26]] — tool and function-calling data; typed schemas와 executable verifiers로 conversation synthesis를 확장한다.
- `figures/turn-distribution.html` — Baize / UltraChat / CAMEL / SODA / OpenAssistant / WildChat 전반의 interactive turn-count + token-length histograms.
