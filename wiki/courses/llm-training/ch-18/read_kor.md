<!-- chapter: ch-18
     track: synthetic
     title: The Synthetic-Data Design Pattern
     sources: [[self-instruct]], [[nemotron-4-synthetic]], [[apigen]], [[openmathinstruct-2]], [[nathan-lambert-synthetic-data]], [[sebastian-raschka-synthetic]], [[synthetic-data-scaling-laws]]
     figures: figures/synth-loop.html
-->

# 18장 — 합성 데이터 설계 패턴

> **핵심 통찰.** Self-Instruct의 52K bootstrapped instructions부터 Nemotron-4의 98%-synthetic alignment stack, APIGen의 three-layer function-call verifier, OpenMathInstruct-2의 14M CoT traces에 이르기까지 모든 production synthetic-data pipeline은 같은 six-stage operational loop의 instance다: **generate -> filter -> dedup -> verify -> select -> mix**. 논문들은 *어떤* stage에 투자하는지가 다르지, *어떤* stage가 존재하는지는 다르지 않다.
>
> **가이드라인.** synthetic-data paper를 읽기 전에 여섯 stage를 찾아라. 하나를 설계하기 전에 loop를 *먼저* 쓰고, 각 stage에 concrete mechanism을 배치하라. 비어 있는 stage가 있다면, 예컨대 verifier가 없거나 dedup이 없거나 mix policy가 없다면, 그것은 단순화가 아니다. 알려진 failure mode다.

---

## 이 장이 존재하는 이유

이 course의 synthetic track은 열두 장과 수십 개 pipeline을 다룬다. 논문 목록으로 가르치면 자료는 흐릿한 약어 더미가 된다. Self-Instruct가 Alpaca를 낳고, Evol-Instruct를 낳고, WizardLM을 낳고, Magpie를 낳고, Persona-Hub을 낳고, UltraFeedback을 낳고, Nemotron-4를 낳고, Tulu-3을 낳는다. 열 번째 acronym쯤 되면 이해가 아니라 암기가 된다.

목록이 그렇게 가르쳐지는 이유는 publication order라는 우연이다. 그러나 그렇게 가르쳐져서는 안 되는 이유는 이 모든 paper가 서로 다른 plug-in을 가진 같은 six-stage loop이기 때문이다. Nathan Lambert는 "Frontiers in Synthetic Data"에서 이 점을 직설적으로 말한다. **강한 base model과 robust verification**이 있으면 "synthetic data can do almost all of the work", 즉 합성 데이터가 거의 모든 일을 할 수 있다. verification layer가 bottleneck이며, modality와 무관하게 항상 같은 shape다.

이 장은 loop를 영구적인 mental model로 설치한다. 이후 모든 장, 즉 generation methods(ch-19), code / math / tools / long-context를 위한 domain-specific pipeline, filter and verifier design, judge calibration, mixing policy는 loop의 한 stage를 조사한다. 아직 읽어 본 적 없는 pipeline도 여섯 질문을 이미 알고 있기 때문에 알아볼 수 있게 된다.

가져가야 할 것 세 가지:

1. 순서대로 외운 six-stage loop와 각 stage가 답하는 **one-line question**.
2. Self-Instruct, Nemotron-4, APIGen, OpenMathInstruct-2가 모든 stage를 어떻게 instantiate하는지 보여주는 **4 x 6 table**. stage를 비워 두고 어떤 대가를 치르는지도 포함한다.
3. **Stage별 failure mode taxonomy**. "dataset은 2M samples인데 downstream accuracy가 flat" 같은 증상을 보면 어느 stage가 고장 났는지 찾을 수 있다.

---

## 1. 운영 루프

데이터가 흐르는 순서의 여섯 stage는 다음과 같다.

```
 [seeds + task def]
        |
        v
 (1) generate  --->  (2) filter  --->  (3) dedup  --->  (4) verify  --->  (5) select  --->  (6) mix
        ^                                                                                      |
        |___________________________ optional iteration (new seeds, teacher rotation) __________|
```

Stage마다 한 줄이다. 외워라.

- **(1) Generate.** *Teacher는 무엇을 생산하는가?* Prompt, response, trajectory, rewrite.
- **(2) Filter.** *Structural 또는 surface reason으로 무엇을 drop하는가?* Schema violation, wrong language, ill-formatted generation, too-long/too-short output, banned topic.
- **(3) Dedup.** *Redundant해서 무엇을 drop하는가?* n-gram overlap, ROUGE, MinHash, embedding distance에 의한 near-duplicate.
- **(4) Verify.** *틀렸기 때문에 무엇을 drop하는가?* Gold-answer match, execution check, semantic judge, reward-model score.
- **(5) Select.** *Surviving sample 중 무엇을 얼마나 유지하는가?* Difficulty, coverage, informativeness(IFD, LESS), preference pair를 위한 reward-ranking.
- **(6) Mix.** *이 dataset이 training 중 다른 dataset과 어떻게 compose되는가?* Real data 대비 ratio, task family 간 ratio, curriculum position.

Filter와 verify는 사람들이 가장 많이 뭉개고, pipeline이 가장 자주 틀리는 구분이다. **Filter는 cheap surface-level rejection**이다(regex, JSON parse, length threshold). **Verify는 expensive ground-truth rejection**이다(code 실행, symbolic expression match, reward model scoring, LLM judge adjudication). 모든 high-quality pipeline은 verify cost를 지불한다. 모든 low-quality pipeline은 이를 건너뛰고 filtering을 "verification"이라고 부른다.

Loop는 iterative하다. cleaned corpus가 생기면 이를 새 seed로 feed back하거나(Nemotron의 self-instruct-style seeding), teacher를 자기 cleaned output으로 retrain하거나(Self-Rewarding LM, Nemotron iterated checkpoints), 더 강한 judge로 verification을 다시 실행하고 싶을 때가 많다(West-of-N). iteration axis는 ch-28에서 돌아온다.

Interactive version은 [figures/synth-loop.html](figures/synth-loop.html)을 보라. 어느 stage든 click하면 failure mode와 아래 네 flagship pipeline에서 가져온 concrete example이 나온다.

---

## 2. 왜 loop가 올바른 분석 단위인가

Pipeline은 loop의 *configuration*이다. 나머지 track이 다루는 modality에서 네 가지 예를 보자.

- **Bootstrap instruction data (Self-Instruct, Alpaca, WizardLM).** (1) generate와 (2) filter는 강하고, (4) verify는 최소다. format validity 정도다.
- **Alignment at scale (Nemotron-4, Tulu-3).** (1) generate와 (4) reward-model을 통한 verify가 강하다. (5) select는 DPO chosen/rejected selection에 RM score를 쓴다. (6) mix는 staged다(code SFT then general SFT).
- **Tool / function calling (APIGen, ToolACE, xLAM).** (1) generate는 약하다(유능한 LLM이면 충분). (4) verify는 무지막지하게 강하다. format, execution, semantic 세 layer가 직렬로 쌓인다.
- **Reasoning-trace synthesis (OpenMathInstruct-2, Numina-Math).** (1) generate는 expensive teacher sampling으로 강하다(problem당 K=32). (4) verify는 SymPy gold-answer match로 강하다. (3) dedup은 약하다. diversity가 cross-problem novelty가 아니라 per-problem solution variety에서 오기 때문이다.

같은 shape라는 관찰은 실제 leverage를 준다. APIGen의 execution layer가 raw generation의 약 40%를 reject하는 이유를 이해하면, OpenMathInstruct-2의 SymPy check도 왜 큰 비율을 reject하는지, Self-Instruct의 ROUGE-L filter가 왜 그 자체로 insufficient했는지 이미 안다. 같은 loop의 같은 slot이 modality-specific verifier로 채워진 것이다. **Code에서 이해한 verifier는 "Python sandbox"를 "SymPy check"로 바꾸면 math로 일반화된다.** 이 porting이 loop를 올바른 분석 단위로 만든다.

### 2.1 Loop의 비용 프로필

Loop는 *돈이 어디로 가는지*도 생각하게 만든다. Nathan Lambert의 2025 post-training write-up은 명시적이다. "post-training now consumes a substantial fraction of total FLOPs"이며, 이는 (a) multi-round rejection sampling, (b) multi-model generation fleet, (c) large RL rollout이 구동한다. Stage별로 나누면:

- **(1) Generate**가 지배적인 *inference* cost다. OpenMathInstruct-2의 teacher sampling만 약 650K H100-hours를 소비했다. student training run보다 훨씬 크다. APIGen의 약 $8K teacher-API spend가 작아 보이는 것은 dataset이 60K examples이기 때문이다. math-style K=32 per problem으로 scale하면 숫자는 맞먹는다.
- **(2) Filter**는 거의 공짜다. regex와 JSON parsing이다.
- **(3) Dedup**은 중간 정도다. 수천만 pair에 대한 MinHash는 manageable하다. growing set에 대한 ROUGE-L은 O(n^2)이고 Self-Instruct-scale run에서도 물린다.
- **(4) Verify**에는 두 regime이 있다. Cheap verifier(SymPy, Python sandbox, schema match)는 CPU minutes로 bound된다. Expensive verifier(GPT-4-as-judge, reward-model scoring)는 종종 stage (1)과 비용이 비슷하다. APIGen은 sandbox execution과 GPT-4 judge를 둘 다 실행하므로 stage 4만으로도 stage 1을 넘을 수 있다.
- **(5) Select**는 모든 sample에 RM을 실행하지 않는 한 대개 싸다(Nemotron은 candidate당 한 번 실행).
- **(6) Mix**는 data-time cost가 아니라 training-time decision이다.

운영적 결론은 이렇다. **pipeline은 대부분 stage 1과 4에 돈을 쓴다.** 나머지는 plumbing이다. Stage 4를 무시하는 budget discussion은 verification이 공짜인 척하는 것이다. 공짜가 아니며, optional로 취급하면 downstream gain이 1M 근처에서 plateau하는 14M-example corpus가 생긴다.

---

## 3. 네 pipeline, 여섯 stage — 매트릭스

아래 표는 출력해 오래 봐야 할 것이다. Row는 stage, column은 flagship pipeline이다. 각 cell은 그 pipeline이 그 stage에서 무엇을 하는지와, 괄호 안에 source의 concrete number를 말한다.

| Stage | Self-Instruct (2022) | Nemotron-4 340B (2024) | APIGen (2024) | OpenMathInstruct-2 (2024) |
|---|---|---|---|---|
| **(1) Generate** | GPT-3 (text-davinci-001)을 8 in-context examples(6 seed + 2 generated)로 prompt해 새 instruction을 만들고, 이어 instance generation. 175 seed tasks. Output: 약 252K raw candidates. | code, QA, topic-following, function-calling, refusal에 걸친 task-family-seeded prompts. iterated Nemotron checkpoints가 teacher. code용 Genetic Instruct(self-instruct + WizardCoder mutation + LLM fitness). | 3,673-API pool에서 function을 sample(k=1..3)하고, DeepSeek-Coder-V2 또는 GPT-4에 (query, gold call) pair를 prompt. Diversity sampler가 rare API category를 upweight. | Llama-3.1-405B-Instruct가 problem당 K=32 CoT solution을 T=1.0, top-p=0.95로 sample. MATH + GSM8K(각 7.5K)에서 seed augmentation 뒤 600K problems. |
| **(2) Filter** | ill-formatted generation drop; input == output인 instance drop; "image/graph/file" 언급 drop(GPT-3가 다룰 수 없음); output length bounds. | staged: code filter는 general filter와 별도. topic-following track은 distractor turn을 유지(의도적 noise). code compilation의 format validity. | **Layer 1 of 3:** JSON parse 가능, schema field 존재, type match(int/str/bool/enum/list). | boxed final answer를 regex-extract. boxed answer가 없으면 reject. |
| **(3) Dedup** | 모든 prior instruction에 대해 **ROUGE-L > 0.7**이면 drop. raw instruction candidate의 약 50% drop. | Light cross-prompt dedup. 논문은 aggressive near-dup removal보다 task-family partitioning에 더 의존한다. | (query, call) pair 전체에 대해 **MinHash**. | 한 problem의 accepted solution *내부* near-duplicate suppression. 서로 다른 problem이 overlapping solution을 만드는 것은 의도적으로 허용. |
| **(4) Verify** | **Minimal** — gold-answer checker 없음, executor 없음. format + ROUGE뿐. 논문이 인정한 약점. | **Reward model**(Nemotron-4-340B-Reward)이 response score. preference pair에서 "chosen" vs "rejected" selection에도 judge로 사용. 작은 human anchor(약 20K)가 RM calibrate. | **Layers 2 and 3 of 3:** (2) reference impl에 대해 5-sec Python sandbox에서 call 실행. exception이면 reject. (3) GPT-4 judge가 (query, call, execution-result)를 보고 "Yes"만 accept. combined rejection 약 40% raw. | **SymPy** symbolic equivalence(MATH) + exact numeric match(GSM8K). human-audited sample에서 residual false-positive rate(right answer, wrong reasoning) 약 7%. |
| **(5) Select** | implicit: (2)+(3)을 통과한 모든 것. difficulty scoring 없음, coverage targeting 없음. | DPO/RPO를 위한 **RM-score-driven selection**: highest-scoring response = chosen, lowest-scoring = rejected. Selection이 전체 preference-pair construction step. | four call-shape bucket(simple / multiple / parallel / parallel-multiple)에 걸쳐 모든 survivor 유지. 끝에서 bucket balance. | 모든 verified solution 유지. per-solution difficulty weighting 없음. Selection은 downstream solution이 아니라 upstream question augmentation을 통해 *problem* 위에서 이뤄짐. |
| **(6) Mix** | GPT-3 위 single-stage SFT. mix 없음. 이것이 전체 dataset. | **Staged**: code SFT first(800K), then general SFT(200K); then DPO(160K) followed by RPO(300K). continued pretraining에 alignment-style QA도 추가. Total pipeline >98% synthetic. | Mistral-7B / Mixtral(xLAM models) 위 single-stage SFT. | Llama-3.1-{1.5B, 8B, 70B} 위 single-stage SFT. 이 stage에서는 real-data mixing 없음. |
| **Output scale** | 52K instructions / 82K instances (from 252K raw) | 약 800K code + 200K general SFT, 160K DPO, 300K RPO | 3,673 APIs에 걸쳐 약 100K raw에서 60K verified samples(40% rejection) | 14M (problem, solution) pairs, augmented problem당 avg 23 solutions |

이 matrix가 명백히 만들지만 flat paper list가 숨기는 점들이 몇 가지 있다.

- **Self-Instruct는 stage 4를 사실상 비워 둔다.** 이것이 founding paper의 진짜 약점이다. 모든 successor(Alpaca -> Evol-Instruct -> Wizard* -> Nemotron)는 특정 later stage를 upgrade한 것이다. 계보는 "new ideas"가 아니라 "empty cell을 채우기"다.
- **APIGen의 명성은 전부 stage 4다.** 세 layer를 제거하면 generic function-calling instruction dataset이 된다. 그들의 ablation: semantic layer를 빼면 BFCL-V1 6점 하락, execution을 빼면 11점 하락, format을 빼면 18점 하락. **세 layer 모두 load-bearing**이다. 이것이 논문의 전체 기여다.
- **Nemotron의 innovation은 stage 5다.** Scale synthetic generation은 쉽다. human annotator 없이 chosen/rejected pair를 고르는 것이 어렵다. reward-model-as-selector가 20K human example로 약 1.4M-example preference corpus를 지배하게 한다.
- **OpenMathInstruct-2의 innovation은 problem level의 stage 1**(question augmentation into 600K problems)과 solution level의 K=32가 결합된 것이다. Stage 4는 aggressive하지만 math에서는 표준이다(SymPy is off-the-shelf).

이 장에서 하나만 기억한다면 이것이다. 네 paper는 지면 위에서는 달라 보이지만 loop 위에서는 같은 일을 한다.

---

## 4. Stage별 실패 모드

모든 stage에는 characteristic failure가 있다. 증상을 알아보고, stage를 찾고, 고쳐라.

**(1) Generate — template로 collapse.** Teacher는, 특히 few-shot seed로 prompt할 때, 몇 가지 phrasing으로 빠르게 좁아진다. Self-Instruct의 diversity stats(paper Table 2)는 ROUGE filter가 없을 때 iteration에 걸쳐 root-verb entropy가 떨어짐을 보인다. 증상은 training distribution이 eval과 overlap하는 benchmark는 풀지만 out-of-template prompt에서는 collapse하는 downstream model이다. Fix: upstream에 structural diversity를 주입하라(Persona-Hub personas, topic-tag conditioning, seed rotation, Nemotron식 category-seeded prompt).

**(2) Filter — over-prune 또는 under-prune.** Over-prune(너무 엄격)은 유용한 long-tail signal을 죽인다(Self-Instruct 2022 run에서 "graph" 금지는 math-diagram-related한 모든 것을 지웠다). Under-prune(너무 관대)은 schema violation을 SFT에 들여보내고, model은 syntactically broken output을 만드는 법을 배운다. Fix: task별로 pre- and post-filter distribution을 비교하고, acceptance rate가 task family 전반에서 안정적인지 확인하라.

**(3) Dedup — diversity 손실.** Failure mode는 미묘하다. dedup metric이 surface-level(ROUGE, MinHash)이면 entity name만 다르고 underlying task structure는 같은 prompt 10,000개를 여전히 받아들일 수 있다. 모델은 그 structure에 overfit한다. Fix: surface dedup을 embedding-space 또는 topic-space dedup(InsTag)과 짝짓고, task-family entropy를 monitor하라.

**(4) Verify — verifier가 없으면 signal도 없다.** 이것이 synthetic data의 cardinal sin이고 Nathan Lambert가 계속 강조하는 지점이다. "verification is the bottleneck", 즉 검증이 bottleneck이다. verifier 없는 pipeline은 teacher가 우연히 내놓은 품질의 데이터를 만든다. floor도 ceiling도 없다. APIGen의 40% rejection rate는 낭비가 아니라 dataset이다. Fix: generating을 시작하기 전, 후가 아니라, modality에 맞는 verifier(gold answer, executor, RM, judge)를 고르라. cheap verifier가 *없는* modality(open-ended writing, subjective tasks)에서는 calibrated judge를 사용하고 bias를 audit해야 한다(ch-26).

**(5) Select — biased mix.** verification을 통과한 모든 것을 keep하면, teacher가 우연히 만들어 낸 distribution을 상속한다. 보통 쉽고 짧고 흔한 style의 answer 쪽으로 bias된다. 증상은 simple benchmark에서는 좋아 보이지만 어려운 benchmark에서 plateau하는 모델이다. Fix: difficulty(IFD, LESS, Superfiltering — ch-25), reward-model score(Nemotron), 또는 coverage(task-family quota)에 대해 explicit selection하라. "kept = passed-verification"이라는 질문 없는 default는 overriding할 가치가 있다.

**(6) Mix — ratio와 order가 중요하다.** 두 failure mode가 있다. 첫째, synthetic/real ratio. 2025년 scaling-laws literature([[synthetic-data-scaling-laws]])는 pretraining에서 약 30% rephrased synthetic이 optimal이라고 찾는다. high fraction의 pure-generated synthetic은 model-collapse signature를 재현한다. 둘째, order. Nemotron이 code SFT와 general SFT를 분리한 것은 장식이 아니다. homogeneous하게 섞으면 code performance가 degrade된다. Fix: mix ratio와 ordering을 hyperparameter로 다루고 ablate하라. "more synthetic = better"라고 가정하지 말라.

### 4.1 Anchor set 원칙

네 pipeline 모두에서 pattern이 반복된다. *작은* high-quality human data가 훨씬 큰 synthetic corpus를 calibrate한다. Nemotron-4는 이를 노골적으로 말한다. "작은 human anchor set(약 20K)이 훨씬 더 큰 synthetic alignment corpus" 약 1.4M examples를 지탱할 수 있다. Self-Instruct의 anchor는 175 seed tasks다. APIGen의 anchor는 executable API reference 3,673개(ground-truth implementation)다. OpenMathInstruct-2의 anchor는 pre-augmentation 이전 MATH + GSM8K training problems(15K)다.

Anchor set은 stage 1의 seed일 뿐 아니라 stages 2–4를 *validate*하는 것이기도 하다. filter와 verifier가 anchor에서는 모든 것을 pass하고 raw synthetic에서는 대부분을 reject한다면 pipeline은 healthy하다. anchor 자체를 높은 비율로 reject한다면 filter가 miscalibrated된 것이다. 이 invariant를 regression test로 유지하라. 이것은 "100% synthetic, no human data", 즉 100% 합성이고 인간 데이터가 없는 pipeline이 production에 존재하지 않는 이유도 설명한다. 몇천 example뿐이어도 항상 anchor가 있다. public slogan "98% synthetic"은 training mixture의 *bulk*를 말할 뿐, human anchor의 부재를 말하지 않는다.

### 4.2 Iteration 아래의 누적 오류

Loop가 반복될 때, 즉 cleaned synthetic을 seed로 다시 넣거나 teacher를 자기 filtered output으로 retrain할 때, stages 2와 4의 error는 compound한다. Nemotron paper는 이를 flag한다. "같은 scorer를 iteration 전반에서 재사용하면 reward-model error가 누적된다." Lambert의 방어는 **accumulation over replacement**다. real data를 완전히 대체하지 말고, 항상 old anchor 위에 new synthetic을 쌓아라. 이것은 ch-14에서 다룬 model-collapse theory의 실무적 counterpart다. 이 장에서의 요점은 iteration이 공짜가 아니라는 것이다. 같은 loop가 위험이 발생하는 site이고, 방어는 stages 5와 6에 있다(real data를 mix에 유지하고 anchor를 살려 둔다).

---

## 5. 왜 loop가 modality를 넘어 port되는가

이것이 design-pattern lens의 payoff다. 여섯 stage를 interface로 다루면, 새로운 paper는 한두 slot만 swap한 것이 된다.

- **Code generation (WizardCoder, OSS-Instruct, OPC):** stage 1 = "snippet -> instruction" 또는 evol-instruct mutation. stage 4 = unit-test execution. executor만 다른 APIGen과 같은 shape다.
- **Long context (LongAlign, ProLong):** stage 1 = document chunking + multi-doc fusion. stage 4 = needle-in-a-haystack check 또는 position-conditioned retrieval. SymPy를 retrieval-style verifier로 바꾼 OpenMathInstruct-2와 같은 shape다.
- **Multi-turn agent trajectories (AgentInstruct, APIGen-MT):** stage 1 = sandboxed environment에서 rollout. stage 4 = task-success check. turn을 넘어 확장된 APIGen과 같은 shape다.
- **Pretraining rephrase (WRAP, Cosmopedia):** stage 1 = real document rephrase. stage 4는 ground truth가 answer가 아니라 "rephrase is coherent and faithful"이므로 유난히 약하다. 그래서 pretraining synthesis에서는 stage 6(mix ratio)이 지배적 lever다(ch-22).

한 번 보면 porting은 기계적이다. Loop를 인식하는 것이 다음 20개의 synthetic-data paper를 읽는 일을 피곤한 암기가 아니라 빠른 분석으로 만든다.

### 5.1 Raschka의 stage-1 taxonomy 다시 읽기

Sebastian Raschka의 practitioner overview는 stage 1을 네 sub-type으로 catalog한다. **rewrite, backtranslate, bootstrap, full-generate**. 이 classification이 유용한 이유는 각 sub-type이 stage 4에서 다른 failure profile을 갖기 때문이다.

- **Rewrite** (WRAP, Cosmopedia, paraphrase augmentation)는 label을 보존한다. stage 4는 correctness check가 아니라 faithfulness check다. cheap verifier이고, catastrophic할 일이 드물다.
- **Backtranslate** (다른 언어 또는 modality를 통한 round-trip)는 construction상 self-consistent하다. stage 4는 대부분 round trip이 information을 잃지 않았는지 validate한다.
- **Bootstrap** (Self-Instruct family): stage 1이 seed에서 fresh (input, output) pair를 생성한다. task가 verifiable하지 않으면 stage 4에는 ground truth가 없다. 이것이 대부분 Self-Instruct-era pipeline의 가장 약한 부분이다.
- **Full-generate** (Phi-style textbook synthesis, Nemotron category-seeded prompts): stage 1이 전체 document 또는 dialogue를 scratch에서 상상한다. stage 4는 judge 또는 RM에 기대야 한다. hallucination risk도 가장 높다.

Raschka의 요점은 stage-1 choice가 stage 4가 해야 할 일을 *미리 결정한다*는 것이다. Rewrite + backtranslate는 verification에서 일을 덜어낸다. bootstrap + full-generate는 verification에 일을 몰아넣는다. cost side에서 본 같은 loop다.

---

## 6. "Verification is the bottleneck" — 2025 합성의 운영 원칙

네 pipeline을 분해해 보면 메시지는 일관된다. Self-Instruct의 알려진 약점은 비어 있는 verify stage다. Nemotron-4의 차별점 전체는 verify/select slot에 서 있는 reward model이다. APIGen의 명성은 three-layer verifier 위에 있다. OpenMathInstruct-2의 OMI-1 대비 gain은 더 큰 teacher와 더 엄격한 SymPy verification이 대략 반씩 담당한다. 2025 landscape를 요약하는 Nathan Lambert는 단호하다. "부족한 자원은 verifiable prompts다."

이 원칙에는 나머지 track에 들어가기 전에 지금 내재화해야 할 세 가지 운영 corollary가 있다.

1. **Verifiable task는 compound하고, unverifiable task는 그렇지 않다.** Math, code, function-calling, schema-constrained task는 모두 검증이 싸고, 이 domain의 pipeline은 깨끗하게 scale한다(OpenMathInstruct-2의 14M solutions, APIGen의 3/3 layer가 모두 load-bearing임을 보이는 per-layer ablation). Open-ended writing, summarisation, creative task에는 cheap verifier가 없다. 그쪽의 progress는 judge quality에 gated되고, 그것이 ch-26 주제다.

2. **Generation은 commodity이고, verification은 moat다.** Frontier API가 있으면 누구나 stage 1을 실행할 수 있다. production-grade pipeline을 가르는 것은 stage 4의 engineering이다. reference-API sandbox(APIGen), reward model plus small human anchor(Nemotron), symbolic equivalence checker(OMI-2), executable unit tests(WizardCoder/OSS-Instruct). post-training team에 합류한 사람에게 주는 lesson은 이렇다. load-bearing이 되고 싶다면 stage 4를 소유하라.

3. **"Synthetic data can do almost all of the work"**, 즉 합성 데이터가 거의 모든 일을 할 수 있다 — Lambert의 2024년 6월 주장 — **given** a strong base model and robust verification. 두 전제 중 하나라도 빠지면 주장은 실패한다. Base-model 쪽은 ch-02..ch-08의 focus다. verification 쪽은 이 track의 focus다. 이 장의 loop는 그 두 절반이 어떻게 맞물리는지를 보여준다.

나머지 track은 본질적으로 여섯 cell을 점점 더 정교하게 채우는 11장이다. Loop를 기억하라.

---

## 7. 이 track이 doctrine으로 가르치지 않을 것

나머지 synthetic track이 어떻게 구성되는지에 대한 note다. 개별 paper는 case이지 doctrine이 아니다.

- ch-19는 stage 1의 interchangeable implementation으로서 **generation methods**(bootstrap, evol, extraction, persona, rephrase)를 catalog한다.
- ch-20부터 ch-23은 **modality chapters**다. instruction, reasoning-trace, tool-calling, long-context. 각 장은 stages 1과 4가 어떻게 specialize되는지에 대한 case study다.
- ch-24부터 ch-26은 **cross-cutting machinery**다. filtering, dedup, judges, 즉 stages 2, 3, 4다.
- ch-27은 **mixing and scaling laws**다. stage 6이며, 2025년 empirical result를 다룬다.
- ch-28은 **iteration and self-improvement**다. loop-around-the-loop다.
- ch-29는 synthetic-track lab이다.

그 어떤 장도 loop를 다시 가르치지 않을 것이다. 각 장은 당신이 loop를 머릿속에 들고 있다고 가정하고, paper가 어떤 cell에서 innovation하는지를 논의한다.

### 7.1 모든 synthetic-data paper를 위한 읽기 체크리스트

synthetic track의 다음 paper를 열 때, result를 읽기 전에 loop를 한 번 걸어가라.

- **Stage 1:** Teacher는 누구인가? Seed는 무엇이고 몇 개인가? Prompt shape는 무엇인가?
- **Stage 2:** 어떤 surface filter가 있는가? Acceptance rate는 얼마인가?
- **Stage 3:** 어떤 dedup metric을 어떤 threshold로 쓰는가? Cross-corpus에 적용하는가, within-item에 적용하는가?
- **Stage 4:** Ground truth는 무엇인가? 없다면 judge는 무엇이고 어떻게 calibrate되는가?
- **Stage 5:** survivor 중 무엇을 training에 넣을지 결정하는 selection criterion은 무엇인가?
- **Stage 6:** 이 dataset은 training time에 real data와 어떻게 compose되는가? mix ratio와 curriculum order는 무엇인가?

질문에 답할 수 없다면 그 cell은 비어 있거나(때로 legitimate하지만 보통 weakness), paper에서 모호하게 설명된 것이다(flag할 가치가 있다). 이 checklist만 있으면 synthetic-data paper에 효과적인 note를 할 수 있다.

---

## 연결과 다음 단계

- **[[ch-17]]** — pretraining을 위한 dataset curation. 이 장은 post-training mirror다. 둘 다 같은 filter/dedup/verify triad에 기대며, 서로 다른 data source에 적용한다.
- **[[ch-19]]** — generation methods. 즉시 다음 step은 loop의 stage 1을 깊이 보는 것이다.
- **[[ch-20]]..[[ch-23]]** — modality chapters. 각각 이 loop를 spine으로 재사용한다.
- **[[ch-25]]** — filtering-for-quality(IFD, LESS, Superfiltering) — stages 2 and 5 in depth.
- **[[ch-26]]** — judges and judge bias — modality에 cheap ground truth가 없을 때의 stage 4.
- **[[ch-27]]** — synthetic-data scaling laws and mix ratios — stage 6.
- **[[ch-44]]** — RLVR(verifiable rewards)은 같은 verify-stage technology를 data time뿐 아니라 training time에 재사용한다.

## 더 읽을거리

- [[self-instruct]] — founding paper. canonical four-step pipeline과 ROUGE-L filter를 읽어라.
- [[nemotron-4-synthetic]] — industrial-scale alignment. reward-model-as-selector와 staged SFT를 읽어라.
- [[apigen]] — three-layer verification과 per-layer ablation의 가장 깨끗한 articulation을 읽어라.
- [[openmathinstruct-2]] — teacher-strength scaling과 SymPy-based gold-answer verification을 읽어라.
- [[nathan-lambert-synthetic-data]] — "verification is the bottleneck", 즉 검증이 bottleneck이라는 framing. easy-vs-hard-verifiable split도 있다.
- [[sebastian-raschka-synthetic]] — stage 1의 rewrite / backtranslate / bootstrap / full-generate taxonomy.
- [[synthetic-data-scaling-laws]] — stage-6 empirics. rephrased vs pure-generated at scale.

## 동반 시각화

**[figures/synth-loop.html](figures/synth-loop.html)** — interactive six-stage loop. 어떤 node든 click하면 그 characteristic failure mode와 위 네 flagship pipeline 중 하나에서 가져온 concrete example을 볼 수 있다. Loop가 reflex가 될 때까지 사용하라.
