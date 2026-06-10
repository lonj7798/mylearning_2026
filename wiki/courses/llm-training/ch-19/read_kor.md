<!-- chapter: ch-19
     track: synthetic
     title: Generation Methods — Bootstrap, Evol, Extraction, Persona, Rephrase
     sources: [[self-instruct]], [[alpaca]], [[evol-instruct]], [[wizardmath]], [[wizardcoder]], [[code-evol-instruct]], [[magpie]], [[persona-hub]], [[rephrasing-the-web]], [[humpback]]
     figures: figures/generation-methods.html
-->

# 19장 — 생성 방법: 부트스트랩, Evol, 추출, 페르소나, 재표현

> **핵심 통찰.** 합성 instruction을 만드는 독립적인 방법은 다섯 가지뿐이다. 2023년 이후 공개 post-training의 모든 파이프라인은 이 다섯 가지를 조합한 것이다. **부트스트랩(Bootstrap)**(Self-Instruct)은 teacher가 seed pool에서 외삽하도록 한다. **진화(Evolve)**(Evol-Instruct)는 기존 instruction을 복잡도나 다양성 축을 따라 다시 쓴다. **추출(Extract)**(Magpie)은 정렬된 모델의 chat template을 이용해 *아무 prompt 없이* instruction을 요청한다. **페르소나 조건화(Persona-condition)**(Persona-Hub)는 teacher가 자신의 전형적 응답에서 벗어나도록 조종하는 "누구인가"를 붙인다. **재표현(Rephrase)**(WRAP)은 실제 문서에 생성을 grounding한 뒤 다시 쓴다. 다섯 방법은 비용, 다양성, seed 의존성, 실패 모드라는 네 축에서 서로 다른 trade-off를 가지며, 이미 가진 데이터의 형태에 맞춰 optimizer를 고르듯 방법을 고르면 된다.
>
> **가이드라인.** 좋은 seed task 175개와 API 예산이 있으면 bootstrap을 써라. 이미 52K SFT 세트가 있고 더 어려운 질문을 원하면 evolve를 써라. GPU는 있지만 API가 없고 목표 domain이 Llama-3-Instruct로 잘 포괄된다면 extract를 써라. 합성 데이터가 한 가지 목소리로 수렴하고 있다면 persona로 condition하라. 원재료가 noisy web text이고 목표가 pretraining 효율이라면 rephrase하라. 이들을 조합하라. 진지한 2025년 파이프라인은 모두 그렇게 한다.

---

## 이 장이 필요한 이유

Ch-18은 generator–verifier stack을 설정했다. 누가 teacher를 호출하는지, rollout을 어떻게 framing하는지, 왜 verifier-gated synthesis가 순진한 sampling보다 나은지 다뤘다. 이 장은 첫 번째 slot, 즉 *generator*를 공개 post-training을 지배하는 다섯 가지 method family로 채운다. Ch-20은 distillation-as-data(Orca / R1-distill), ch-21은 top-down taxonomy synthesis(GLAN / Phi), ch-22는 이 generator들의 output 전반에 걸친 quality-driven selection을 이어서 다룬다.

아래 순서는 시간순이다. 각 방법이 이전 방법의 한계에 대한 반응으로 발명되었기 때문이다.

1. **Self-Instruct (2022)** — 첫 번째 bootstrap. "human annotator 없이 instruction data를 만들 수 있는가?"라는 질문에 답했다.
2. **Alpaca (2023)** — 더 강한 teacher로 같은 bootstrap을 수행했다. "작은 academic group도 이 비용을 감당할 수 있는가?"에 답했다.
3. **Evol-Instruct (2023)** — "seed가 포괄하는 범위를 넘어선 complexity distribution이 있는가?"에 답했다.
4. **WizardMath / WizardCoder (2023)** — "evolution을 domain별로 특화할 수 있는가?"에 답했다.
5. **Humpback (2023)** — "방향을 뒤집어 text → instruction으로 갈 수 있는가?"에 답했다.
6. **WRAP (2024)** — "같은 pretraining loss에서 rephrasing이 crawling을 대체할 수 있는가?"에 답했다.
7. **Persona-Hub (2024)** — "output diversity를 가장 크게 제어하는 단일 knob는 무엇인가?"에 답했다.
8. **Magpie (2025)** — "prompt조차 필요한가?"에 답했다.

각 답은 선행자의 실패 모드 하나를 닫았다. 이 장은 다섯 method 이름을 암기하기보다, 닫힌 실패들의 사슬로 읽는 편이 더 유용하다.

---

## 1. Bootstrap — Self-Instruct와 seed-plus-sample-plus-filter loop

[[self-instruct]]부터 보자. LM만으로 usable instruction data를 생산한 첫 번째 파이프라인이다. 전체 recipe는 네 단계다.

**Stage 1 — Seed pool.** 사람이 작성한 task 175개, 각 task는 instruction 하나와 instance 하나로 구성되며 classification, generation, open-ended, extraction을 포괄한다. 그게 전부다. 논문에서 가장 오해받는 숫자는 이 seed pool이 의도적으로 작다는 점이다. seed가 클수록 ROUGE filter의 rejection rate가 빨리 올라가고 output manifold가 더 빠르게 collapse한다.

**Stage 2 — Instruction generation.** 8개의 in-context example(seed 6개, 이전에 생성된 example 2개)로 LM에게 새 task를 요청한다. 논문의 verbatim template은 다음과 같다.

```
Come up with a series of tasks:
Task 1: <seed 1>
Task 2: <seed 2>
...
Task 8: <seed 8>
Task 9:
```

**Stage 3 — Classification branching.** 생성된 instruction이 classification task인지 LM에게 묻는다. 그렇다면 instance는 *input-first* template을 쓴다(LM이 항상 같은 label을 만들지 않도록). 아니라면 *output-first*를 쓴다. 이 branch는 load-bearing이다. 없으면 classification task는 하나의 modal label로 퇴화한다.

**Stage 4 — Filtering.** 세 가지 rule:

- **기존 instruction 중 어느 하나와 ROUGE-L > 0.7인 instruction은 모두 drop한다.** 이것이 논문의 verbatim threshold다. 더 느슨하면 set은 seed의 paraphrase로 collapse한다.
- input == output인 instance, output이 너무 짧거나 길거나, instruction이 "image/graph/file"을 언급하는 경우를 drop한다(LM은 실제로 image를 만들 수 없다).
- 형식이 잘못된 generation을 drop한다.

**Yield.** 약 252K개의 raw generation에서 filter는 약 52K개의 instruction과 약 82K개의 paired instance를 accept한다. text-davinci-001 시대의 GPT-3에 적용하면 Super-NaturalInstructions에서 절대 점수 +33을 만든다. 이는 private human data를 사용한 InstructGPT-001과 맞먹는다.

왜 이것이 작동하는가: LM은 많은 task에 대한 latent knowledge를 갖고 있지만, 이를 표현하도록 스스로 *plan*하지는 못한다. 다만 in-context nudge가 주어지면 *elicitation*할 수 있다. 8-shot prompt가 그 nudge다. ROUGE filter는 새 draw가 instruction-space에서 이전에 보지 못한 region에 landing하도록 강제한다. 이 파이프라인은 task-space 위의 Markov chain이며, ROUGE filter가 rejection step이다.

전체 operator breakdown과 classification-branch failure mode는 [[excerpts/self-instruct]]를 보라.

---

## 2. Alpaca — 저렴한 replication

[[alpaca]]는 새로운 방법이 아니다. Self-Instruct를 세 가지 변경으로 다시 실행한 것이다. teacher로 `text-davinci-003`(001보다 강함), student로 LLaMA-7B(open base), accepted instruction 52K개(새 filter 없음). 논문의 contribution은 **operational**하다. data generation은 500달러 미만, fine-tuning은 100달러 미만임을 증명했다. 이 단일 cost number가 lab credit card를 가진 모든 academic group에게 synthetic SFT를 열었다.

한계: Alpaca의 student는 특정 teacher의 modal style에 묶여 있다. teacher를 GPT-3.5로 ablate해도 downstream skill은 거의 변하지 않는다. GPT-4로 ablate하면 GPT-4의 refusal pattern이 통째로 들어온다. teacher는 style filter이며, 모든 descendant(Vicuna, WizardLM, 모든 X-Instruct set)가 이를 상속한다. Alpaca 시대의 밈인 "GPT-3.5를 LLaMA로 distill했다"는 기술적으로는 맞지만 전략적으로는 오해를 낳는다. 실제로 distill된 것은 teacher가 *선호하는 instruction-space의 sub-manifold*였다.

---

## 3. Evolution — Evol-Instruct의 다섯 In-Depth + 하나의 In-Breadth operator

[[evol-instruct]]는 **complexity axis**를 도입했다. Self-Instruct의 ROUGE filter는 *word space*의 diversity를 강제한다. 하지만 *difficulty* distribution은 건드리지 않는다. Alpaca의 52K는 difficulty가 대체로 flat하다. Evol-Instruct의 핵심 주장은 complexity histogram이 first-class training knob이며, hard instruction의 long tail이 median-difficulty instruction 52K개를 더 추가하는 것보다 downstream skill을 더 개선한다는 것이다.

논문은 강한 teacher LLM을 prompting해 적용하는 여섯 operator를 열거한다. 논문의 verbatim prompt header:

**In-Depth Evolving (instruction을 더 어렵게 만들기):**
1. **Add constraints** — "please add one more constraints/requirements into #The Given Prompt#".
2. **Deepening** — "increase the depth and breadth of the #The Given Prompt#".
3. **Concretizing** — "replace general concepts with more specific concepts".
4. **Increased reasoning steps** — "if #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning".
5. **Complicate input** — input 자체에 code block, table, XML, nested structure를 추가한다.

**In-Breadth Evolving (instruction을 더 다양하게 만들기):**
6. **Mutation** — 전체 topic은 유지하되 더 드문 domain의 완전히 새로운 instruction을 생성한다.

**Pipeline.** Seed = 52K Alpaca instruction. 각 seed마다 operator 하나를 random으로 고르고, teacher에게 적용하도록 prompt한 뒤 response를 생성한다. **elimination step**을 실행한다.
- input과 "same-or-similar"인지 LLM 자신이 check — drop.
- response에 "sorry" / refusal marker가 포함됨 — drop.
- punctuation-only 또는 empty response — drop.
- response가 input을 verbatim copy — drop.

4 round 반복한다. Yield: filtering 후 약 250K evolved instruction. 원본 + evolved mix로 LLaMA-7B/13B/70B에 SFT한다.

왜 이것이 작동하는가: operator는 원래 instruction과 *orthogonal*하게 밀도록 선택된다. "Add constraints"는 example을 difficulty axis를 따라 이동시키고, "mutation"은 topic axis를 따라 이동시킨다. In-Breadth와 In-Depth 두 family가 decorrelate되므로, 둘을 섞으면 논문 Figure 3의 특징적인 long-tail complexity histogram이 생긴다.

한계: teacher는 자기 competence 너머로 evolve하기를 거부한다. 실제로 math problem에 "increased reasoning steps"를 두 번 적용하면 teacher의 ceiling에서 포화된다. GPT-4는 자신이 풀 수 없는 문제를 만들지 않는다. 이것이 WizardMath가 operator를 특화해야 했던 이유다.

verbatim prompt, rejection statistics, operator collision analysis는 [[excerpts/evol-instruct]]를 보라.

---

## 4. Domain specialization — WizardMath와 WizardCoder

**WizardMath**([[wizardmath]]). Evol-Instruct의 math-specialization은 중요한 twist를 추가한다. *bidirectional* evolution이다. Operator는 두 방향으로 실행된다.

- **Downward evolution** — constraint를 줄이고, concept를 더 단순한 것으로 바꾸고, chain을 짧게 만들고, arithmetic을 쉽게 만든다. competition-level seed의 grade-school variant를 만든다.
- **Upward evolution** — constraint를 추가하고, 다른 concept와 compose하고, reasoning depth를 늘리고, multiple solution step을 요구한다. GSM8K seed의 competition-level variant를 만든다.

놀라운 점은 downward direction이었다. 70B model을 *더 쉬운* 문제로 training하는 것이 도움이 된다는 것은 자명하지 않다. 논문의 주장: 더 넓은 difficulty spectrum은 reasoning manifold를 smooth하게 만든다. 모델은 항상 가장 어려운 경로를 시도하기보다, 문제를 풀기 전에 문제의 수준을 인식하는 법을 배운다. 이는 GSM8K/MATH ablation에서 downward-only, upward-only, bidirectional evolution으로 SFT했을 때 모두 bidirectional blend보다 낮은 성능을 보이는 데 나타난다.

SFT 뒤의 RLEIF(Reinforcement Learning from Evol-Instruct Feedback) stage는 ch-26에서 자세히 풀어볼 IRM × PRM reward product가 처음 등장하는 지점이다.

**WizardCoder**([[wizardcoder]]). code specialization은 여섯 generic operator를 다섯 code-native operator로 대체한다.

1. 새로운 constraint나 requirement 추가.
2. 흔한 requirement를 덜 흔한 것(`deque`)으로 교체(`list` 대신).
3. depth / reasoning step 증가.
4. problem complexity 심화(time/space bound, edge case, misleading wording).
5. 특정 language나 library 요구.

Seed = 20K Code-Alpaca. 세 round와 standard elimination filter 후 yield = 약 78K evolved pair. StarCoder-base 위의 WizardCoder-15B는 release 당시 HumanEval 57.3 / HumanEval+ 50.6을 달성해 Claude와 Bard를 이겼다. operator별 HumanEval ablation은 [[excerpts/wizardcoder]]를 보라.

두 Wizards의 구조적 교훈: **operator는 domain-specific이다**. generic Evol-Instruct를 code에 실행하면 code-specific failure mode(edge case, complexity bound, library idiom)를 건드리지 않는 operator에 mass를 낭비한다. math에 실행하면 solvable problem은 만들지만 reasoning manifold를 smooth하게 하는 difficulty spread는 만들지 못한다. operator를 target skill의 failure surface에 맞춰라.

---

## 5. Extraction — Magpie의 prefix-only trick

[[magpie]]는 명백하지만 이전에는 탐구되지 않았던 질문을 던진다. 정렬된 모델이 instruction을 내기 위해 *prompt*가 필요한가? 답은 아니다.

Instruction-tuned model의 chat template은 다음과 같다.

```
<|start_header_id|>user<|end_header_id|>\n\n{user_instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_response}<|eot_id|>
```

Magpie의 trick: 모델에 **pre-query prefix**만 넣는다. 즉 `<|start_header_id|>user<|end_header_id|>\n\n`만 feed하고 EOS까지 sample한다. instruction tuning 동안 수백만 user turn을 본 모델은 그 자체로 plausible user instruction을 채워 넣는다. seed도, API도, prompt engineering도 없다.

그 instruction을 full template으로 다시 feed해 response를 얻는다. 두 번의 call, 하나의 aligned model, 임의의 scale.

**논문의 숫자.** MAGPIE-Air: Llama-3-8B-Instruct에서 3M raw pair, 206 GPU hours. MAGPIE-Pro: Llama-3-70B-Instruct에서 1M raw pair, 614 GPU hours. 여덟 filtering metric(input length, output length, task category, input quality, input difficulty, `all-mpnet-base-v2` + FAISS를 통한 minimum neighbor distance, `FsfairX-LLaMA3-RM-v0.1`의 reward, reward-difference). Threshold는 reward에 대해 `tau1 = -12`, reward-difference에 대해 `tau2 = 0`. `Llama-Guard-2` 기준 safety flag rate <1%.

왜 이것이 작동하는가: instruction-tuned model은 pre-query prefix에서 날카롭고 low-entropy한 distribution을 갖는다. user-turn continuation은 instruction tuning에서 모델이 *predict하도록 훈련된* 부분이다. temperature 1로 그 distribution에서 sampling하면 다양하지만 well-formed한 instruction이 나온다. "plausible user query"에 대한 model posterior가 정확히 training-distribution prior이기 때문이다.

왜 heavy filtering이 필요한가: 추출된 instruction은 현실 세계가 아니라 *teacher의* instruction distribution을 상속한다. Llama-3-Instruct는 "tell me about your day" prompt보다 "explain X"를 더 많이 본다. Magpie의 raw output도 이를 반영한다. 3M → 300K curation에서 방법의 성패가 갈린다. filter-threshold ablation은 [[excerpts/magpie]]를 보라.

**Magpie가 pipeline에서 제거하는 것.** seed pool 없음. teacher API 없음(aligned model은 open-weight). prompt engineering 없음. 남는 것은 filter design과 compute다.

---

## 6. Persona conditioning — Persona-Hub의 1B diversity primitive

[[persona-hub]는 diversity가 다른 모든 방법의 bottleneck이라고 주장한다. Self-Instruct의 252K → 52K filter rate는 diversity loss다. Evol-Instruct의 operator-collision rate도 diversity loss다. Magpie의 3M → 300K curation도 diversity loss다. 모든 pipeline은 자기 teacher의 mean-response mode와 싸우고 있다.

Persona-Hub의 lever: **generation prompt에 persona를 붙인다**. "write a word problem about fractions"라는 같은 instruction도 1,000개의 서로 다른 persona(emergency-room nurse, jazz drummer, rural Kenya의 eighth-grade teacher, retired bridge engineer)로 condition하면 LM의 modal voice가 아니라 persona의 domain 주위에 cluster되는 1,000개의 problem을 만든다.

**Scale.** 1,015,863,523개의 persona를 두 방법으로 모았다.

- **Text-to-Persona.** web corpus(RedPajama v2)의 각 document에 대해 "Who is likely to read / write / like / dislike this text?"로 LM을 prompt한다. persona를 수집한다.
- **Persona-to-Persona.** relationship prompt로 확장한다. "Given persona X, list 10 people who might interact with them professionally / personally / antagonistically." 6 round 반복한다.

0.9 similarity에서 MinHash로 deduplicate한 뒤, 0.9 embedding cosine으로 다시 deduplicate한다(count보다 diversity가 더 중요하면 더 tight하게). 세 prompting mode:

- **Zero-shot:** persona + task specification만.
- **Few-shot:** demonstration 추가.
- **Persona-enhanced few-shot:** 각 demonstration에 대한 persona도 derive해 함께 condition.

**핵심 empirical result.** Output similarity는 persona similarity보다 *낮다*. 80% 비슷한 두 persona가 40–50%만 비슷한 problem을 만든다. persona는 *distributional amplifier*처럼 작동한다. teacher는 persona space 자체에는 없던 topic-space variance를 주입한다.

**Math scaling experiment.** 1.07M persona-synthesized problem, Qwen2-7B SFT → MATH 64.9%. release 당시 7B에서 gpt-4-turbo-preview와 맞먹었다. scaling curve는 1M에서 포화되지 않는다. persona-conditioning은 raw data scaling에 비견되는 headroom을 가진 것으로 보인다.

Persona-Hub를 꺼낼 때: synthetic data가 한 가지 voice로 collapse하고, 다른 방법(Evol operator, Magpie filter)이 이를 깨지 못할 때다. style diversity가 binding constraint라면 persona가 diversity knob다. dedup-threshold ablation과 math-validity audit(96.5%)은 [[excerpts/persona-hub]]를 보라.

---

## 7. Web rephrasing — WRAP의 chunk-and-rewrite

위 모든 방법은 SFT용 *instruction* data를 만든다. [[rephrasing-the-web]]은 다르다. raw web text를 더 깨끗한 style로 rewrite해 *pretraining* data를 만든다.

Pipeline. C4 document를 약 300 token으로 chunk한다(더 긴 chunk는 rephrase 중 information loss를 일으킨다). 각 chunk를 frozen Mistral-7B-Instruct에 네 style prompt 중 하나로 feed한다.

- **Easy** — grade-school vocabulary.
- **Medium / Wikipedia-like** — encyclopedic, neutral tone.
- **Hard / Terse** — dense, technical phrasing.
- **Q/A** — question-answer pair로 reformulate.

real과 synthetic을 1:1로 mix한다. 각 document는 raw text와 rephrase로 모두 등장한다. 그 mix로 decoder-only transformer를 pretrain한다.

**Results.** 128M / 350M / 1.3B parameter에서 WRAP은 약 3× pretraining speedup, matched loss에 필요한 data 약 5× 감소, Pile subset 전반의 perplexity >50% reduction을 보고한다. Headline: **WRAP으로 C4의 15%만 학습한 350M이 C4 전체를 학습한 1.3B를 이긴다**. 같은 실험에서 40× data-efficiency multiplier와 4× parameter-efficiency multiplier가 stack된 것이다.

왜 이것이 작동하는가: raw C4는 style-flat하고 일부는 information-dense하지만 다른 부분은 boilerplate-heavy하다. rephrase는 quality filter(boilerplate가 paraphrase되어 사라짐)이자 style amplifier(Q/A style은 raw prose보다 token당 추출 가능한 fact density가 높음)로 작동한다. 1:1 mix는 real-world distribution을 보존해 모델이 Mistral-7B-Instruct의 voice에 overfit하지 않게 한다.

논문의 cautions: 300 token보다 긴 rephrase는 information을 drop한다. lightweight post-process로 boilerplate intro("Here's a paraphrase...")를 제거해야 하며, 그렇지 않으면 pretraining을 오염시킨다. 그리고 "held-out Mistral rephrases에 대해 reproducible"인지 보는 sanity check가 leakage를 잡는다. style-ablation과 data-efficiency plot은 [[excerpts/rephrasing-the-web]]를 보라.

이것이 ch-21(Phi-textbooks, Nemotron synthetic)이 frontier scale에서 쌓아 올리는 foundation이다.

---

## 8. Humpback — inverse direction으로서의 backtranslation

[[humpback]]은 loop를 닫는다. Self-Instruct: instruction → instance. Humpback: instance → instruction. raw document가 주어지면 aligned seed model에게 그 document를 response로 만들 법한 instruction을 생성하도록 prompt한다. `<inferred_instruction, document>` pair를 quality로 curate하고 fine-tune한다.

이 방법은 2025년 open pipeline에서의 중요성보다, 그것이 *증명한 것* 때문에 더 중요하다. instruction-data space와 response-data space는 competent teacher 아래에서 isomorphic하다. 둘 중 어느 하나를 생성하고 다른 하나를 recover할 수 있다. 이후 모든 pipeline은 이를 가정으로 취급한다. Magpie는 aligned-model prior에서 instruction을 생성하고, WRAP은 response(rewrite)를 생성하며, taxonomy method(ch-21)는 tree에서 둘 다 생성한다.

Humpback의 operational lesson: **curator가 generator와 같은 model family일 때 self-curation은 external curation을 이긴다**. seed model의 judgment는 fresh reward model보다 자신의 output에 대해 더 잘 calibrated되어 있다. 이것은 Magpie의 filter stack이 활용하는 같은 관찰이다.

---

## 9. Comparison table — cost, diversity, seed, failure

다섯 method family를 네 축으로 비교한다. Cost는 2025년 현재 API pricing 또는 open-weight compute를 가정한 accepted example 1K개당 비용이다. Diversity는 embedded output 전체의 1 – mean cosine similarity로 측정한다(높을수록 좋다). "Required seed"는 해당 method가 시작하기 위해 반드시 필요한 minimum artifact다.

| Method | Cost / 1K examples | Diversity (1 - avg cos) | Required seed | Characteristic failure mode |
|---|---|---|---|---|
| **Self-Instruct bootstrap** | ~$1–3 (API teacher) | 0.45 | 175 human-written tasks | Mode collapse — filter가 포화되고 새 draw가 seed의 paraphrase가 됨 |
| **Alpaca cheap replication** | ~$1 (text-davinci-003 era) | 0.42 | Self-Instruct seeds (175) | Teacher-style lock-in — student가 GPT-3.5의 refusal & verbosity pattern을 상속 |
| **Evol-Instruct evolution** | ~$5–10 (multi-round teacher calls) | 0.58 (after 4 rounds) | Existing 52K SFT set | Operator saturation — teacher가 자기 skill ceiling 너머로 evolve하기를 거부 |
| **WizardMath / WizardCoder** | ~$10–20 (bidirectional + filter) | 0.55 (domain-restricted) | Domain seeds (GSM8K / Code-Alpaca) | Reward-hacking downstream (PRM noise); operator를 통한 benchmark contamination |
| **Magpie extraction** | ~$0.05 (open-weight compute only) | 0.50 pre-filter, 0.68 post-filter | None — aligned model only | Distribution narrowing — 추출된 instruction이 real user가 아니라 aligned model의 training prior와 일치 |
| **Persona-Hub conditioning** | ~$2–8 (teacher call + persona) | **0.72** (post-dedup) | A persona bank (released 200K, full 1B gated) | Persona != demographics — model-inferred identity가 bias를 embedding |
| **WRAP rephrase (pretraining)** | ~$0.10 per 1K tokens (open-weight teacher) | 0.40 (style-only) | Raw web corpus | Boilerplate leakage — filtering하지 않으면 "Here's a paraphrase..."가 training을 오염 |
| **Humpback backtranslation** | ~$1–3 (aligned seed model) | 0.48 | Raw documents + seed aligned model | Inferred-instruction mismatch — recover된 instruction이 실제 user가 물을 법한 것이 아닐 수 있음 |

처음 세 row(Self-Instruct, Alpaca, Evol)는 같은 diversity band(0.42–0.58)에 있다. Persona-Hub(0.72)로의 jump가 table에서 가장 큰 single-method diversity gain이며, 그래서 persona-conditioning은 standalone method라기보다 이후 pipeline 내부의 *component*로 등장한다. Magpie row는 aligned model weights를 이미 가지고 있을 때 가장 큰 cost drop(API-teacher method 대비 두 orders of magnitude)을 제공한다. 공개 team에게는 점점 더 그 조건이 사실이 되고 있다.

---

## 10. 실제로는 어떻게 조합되는가

2025년 실제 pipeline은 pure하지 않다. 대표적 blend:

- 작은 hand-curated task set으로 Self-Instruct를 **Seed**한다(175-seed trick은 여전히 load-bearing).
- 최신 aligned open model에서 Magpie로 대규모 **Extract**한다.
- 세 Evol-Instruct operator로 combined pool을 **Evolve**한다(target이 chat이면 "complicate input"은 skip).
- diversity를 회복하기 위해 sampled subset을 10K-persona bank로 **Condition**한다.
- long-form 또는 pretraining-adjacent slice를 WRAP-style chunk-and-rewrite로 **Rephrase**한다.
- 전체를 ch-22 quality stack으로 **Filter**한다.

이 composition은 다섯 method family의 replacement가 아니라 superset이다. 각 family는 final distribution에 distinct *inductive bias*를 기여한다. Self-Instruct seed는 task topology를 설정하고, Evol-Instruct는 difficulty axis를 늘리고, Magpie는 volume을 싸게 넓히고, Persona-Hub는 style collapse를 깨고, WRAP은 이를 pretraining scale로 이식한다.

Interactive companion — [`figures/generation-methods.html`](figures/generation-methods.html) — 은 하나의 source example을 다섯 method 모두에 side-by-side로 통과시킨다. 독자는 "input"이 같아도 generated output이 왜 갈라지는지 볼 수 있다.

---

## Connections

- [[ch-18]] — generator/verifier framing. 이 장은 generator slot을 채운다.
- [[ch-20]] — distillation-as-data(Orca / R1-distill)는 이 generator들과 강한 teacher를 함께 쓴다.
- [[ch-21]] — taxonomy synthesis(GLAN / Phi)는 bootstrap에 대한 top-down alternative다.
- [[ch-22]] — 이 generator들 전체에 대한 quality selection.
- [[excerpts/self-instruct]], [[excerpts/evol-instruct]], [[excerpts/magpie]], [[excerpts/persona-hub]], [[excerpts/wizardcoder]], [[excerpts/rephrasing-the-web]], [[excerpts/wizardmath]] — deep walkthrough.
