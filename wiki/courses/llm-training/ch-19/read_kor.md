# Chapter 19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, Rephrasing

> **핵심 통찰.** 이 chapter generation method는 주로 prompt가 어디서 오는지 다르다. task175개 seed pool(Self-Instruct), existing set의 LLM rewrite(Evol-Instruct), aligned model 자체 user-turn distribution(Magpie), web text로 만든 persona collection(Persona Hub), real document(WRAP,Humpback)이다. 여러 method는 LLM-judged win rate(Evol-Instruct,Magpie,Humpback) 또는 target skill과 가까운 benchmark(WizardCoder,WizardMath)에서 gain을 보고한다. broad imitation data는 crowdworker rating을 올렸지만 LLaMA7B Natural Questions accuracy를17에서10으로 낮췄고([[false-promise-imitating-proprietary-llms]] Table1), verbosity instruction만 바꿔도 model 하나의 AlpacaEval win rate는22.9%에서64.3% 범위였으며([[length-controlled-alpacaeval]] §4.1), Alpaca prompt로 train한 model의 WildChat user prompt average NLL은10.87이고 WildChat train model은5.18이었다([[wildchat]] Fig.3). 따라서 method가 sample하는 prompt distribution은 judge뿐 아니라 real usage 및 held-out capability against measure해야 한다.
>
> **지침.** target이 general instruction following이고 aligned open-weight model이 있으면 filtering을 포함한 Magpie-style extraction을 사용한다. Magpie-Pro-300K-Filtered는 Llama-3-8B에서 Llama-3-8B-Instruct의22.92 대비 AlpacaEval2 length-controlled(LC) win rate25.08에 도달했기 때문이다([[magpie]] Table1). 같은 SFT model의 GSM8K는71.72 대비47.92이므로 targeted reasoning data를 add한다(Table3). domain 하나에 더 hard instruction이 필요하면 evolve하고 held-out dev set에서 round 수를 choose한다. WizardCoder pass@1은3 round 뒤 peak했기 때문이다([[code-evol-instruct]] §3.2, Fig.3). noisy web text의 pretraining efficiency가 goal이면 rephrase하고 real text를1:1 mix로 유지한다. synthetic-only data가 many sub-domain Pile perplexity를 높였기 때문이다([[rephrasing-the-web]] §7 RQ2). every case에서 length-controlled 또는 benchmark-based score, target skill 밖 before/after suite, held-out real user prompt against coverage check를 report한다. judge score는 length와 style을 reward하며 imitation data는 underlying capability 없이 이를 transfer할 수 있기 때문이다.

## 이전에 공부한 version의 correction

1. “If yes, instances use an *input-first* template ... If no, *output-first*.” → 반대다. classification task는 input-first generation이 label 하나에 bias된 input을 만들었으므로 output-first(class label first, label별 input next)를 쓴다. non-classification task는 input-first다([[self-instruct]] arXiv:2212.10560v2 §2.2,Table7-8; [[excerpts/self-instruct]]).
2. “~252K raw generation에서 filter가~52K instruction과~82K instance를 accept” → paper는 filtering 뒤 instruction52,445개(classification11,584개), instance82,439개를 report하며 raw count는 없다.252는 user-oriented human-evaluation set size다([[self-instruct]] Table1,§4.4).
3. “text-davinci-001-era GPT-3에 적용” → vanilla GPT-3(“davinci”,175B)에 적용했다. InstructGPT001은 comparison이다. unseen SuperNI task119개에서 ROUGE-L은6.8에서39.9(+33.1), InstructGPT001은40.8이었다([[self-instruct]] §3,Table3).
4. “larger seed가 ROUGE filter rejection rate를 accelerate” 및 “~50K accepted example 뒤 rejection rate80% above spike” → seed-size/rejection-rate analysis는 없다. size analysis는 human-evaluation gain이 instruction~16K 뒤 plateau함을 보인다([[self-instruct]] §4.5,Fig.7).
5. “input == output, output too short/long instance drop” → identical instance와 same-input/different-output instance를 remove한다. heuristic은 too long/short instruction 및 input을 repeat하는 output을 remove한다([[self-instruct]] §2.2).
6. Alpaca “52K accepted instruction(no new filter)” → released code는 length,blacklisted word,format,ROUGE-L F-measure >0.7로 instruction을 filter하고 output은 check하지 않으며 instruction당 instance 하나를 generate하고 classification split이 없다([[alpaca]], `generate_instruction.py`@761dc5b L52-101,L196).
7. “teacher를 GPT-3.5로 ablate해도 downstream skill 거의 안 변함; GPT-4 ablation은 refusal pattern” → Alpaca는 teacher ablation을 report하지 않고 text-davinci-003 자체가 GPT-3.5-series다. blog는 Alpaca가 dataset style을 reflect하고 ChatGPT보다 shorter answer를 낸다고 한다([[alpaca]], blog).
8. Evol-Instruct elimination “same/similar,sorry,punctuation-only,input verbatim copy” → (1) ChatGPT가 judge한 no information gain; (2) “sorry” 포함 및80 word below; (3) punctuation/stop word만; (4) “given prompt” 같은 evolving prompt word copy다. failed instruction은 pool로 return한다([[evol-instruct]] §3.2).
9. “Mutation—rarer domain,same overall topic” → in-breadth evolving은 same domain의 “even more rare” instruction을 similar length,difficulty로 쓴다([[evol-instruct]] Example3.3).
10. “filtering 뒤~250K evolved instruction; LLaMA7B/13B/70B SFT”, “Figure3 long-tail complexity histogram” →4 round total instruction250k, SFT에는70k sample(v3 LLaMA13B,v1 7B);65B,70B는 ablation이다. paper는 histogram 대신 round별 mean difficulty를 report한다(Fig.5a,App.I Table3)([[evol-instruct]] §4.2,Table2).
11. “GPT-4는 자신이 solve 못할 problem을 produce하지 않음” → WizardLM/WizardCoder evolver는 gpt-3.5-turbo고 teacher-ceiling effect를 cited paper가 measure하지 않는다. WizardCoder15B에서 GPT-4 evolver는62.2, GPT-3.5는59.8 HumanEval pass@1이다([[evol-instruct]] §4.2; [[code-evol-instruct]] Table4).
12. WizardMath operator의 simplified claim → downward은 difficulty를 lower하거나 different topic의 easier question을 쓰며 upward은 constraint add,concretize,reasoning increase다. GPT-4는 round5개(2 downward,3 upward)를 run한다([[wizardmath]] arXiv:2308.09583v3 §3.1; [[excerpts/wizardmath]]).
13. “downward-only/upward-only/bidirectional이 bidirectional blend보다 underperform, broader spectrum이 problem level recognize” → Mistral-7B SFT Table6에서 original59.7 GSM8K/15.1 MATH, downward2 round74.5/34.7, upward3 round78.6/42.5, all5 round81.2/46.2다. difficulty-recognition explanation은 없다([[wizardmath]] Table6).
14. “IRM × PRM reward product를 ch-26에서 unpack” → product reward는 WizardMath §3.3, process reward model은 ch-44, ch-26은 tool/function-calling data다.
15. WizardCoder heuristic list → 다섯 heuristic은 constraint add(~10 word), common requirement을 less common/more specific으로 replace, reasoning step add, erroneous code misdirection, higher time/space complexity requirement다([[code-evol-instruct]] §3.1).
16. “3 round 뒤78K evolved pair/filter,HumanEval+50.6,operator-by-operator ablation” →78k는 seed+merged3 round cumulative고 elimination filter/per-heuristic ablation은 없다.15B EvalPlus greedy는59.8,Claude-Plus53.0,Bard44.5이며 sentence가 HumanEval/HumanEval+를 밝히지 않고50.6은 없다([[code-evol-instruct]] v1 §4.5,v2 §4.3).
17. “Magpie(2025), method가 prior method limit에 reaction” → Magpie(arXiv2024-06,ICLR2025)와 Persona Hub(arXiv2024-06)는 same month, Humpback(2023-08)은 WRAP(2024-01)보다 earlier다([[magpie]]; [[persona-hub]]; [[humpback]]).
18. “3M→300K curation이 method의 성패, sharp low-entropy pre-query distribution” → Magpie-Air3M-Raw22.96 AE2LC,300K-Filtered22.66; Magpie-Pro300K-Filtered25.08,1M-Raw24.16([[magpie]] Table12). paper는 user-turn implicit memorization을 hypothesize(§2.1 Remark)하고 entropy는 measure하지 않는다.
19. Persona-to-Persona “professional/personal/antagonistic 10 people list” → prompt는 “Who is in close relationship with the given persona?”이며 iteration6번이다([[persona-hub]] §2.2).
20. persona similarity80%→problem40-50%,1M saturation 없음/raw scaling headroom,dedup ablation,full1B gated → similarity0.9에서 problem similarity 대부분0.6-0.75(Fig.10c); trend는 raw-data comparison 없이 scaling law와 generally align; dedup ablation 없음( diversity가 count보다 중요하면 threshold0.5 suggested); initial release persona200,000,gated stated 안 됨([[persona-hub]] §1,§2.3,§4.1.2).
21. “40× data efficiency,4× parameter efficiency” → C4 15%는 real token1/0.15≈6.7× fewer,1.3B/350M≈3.7×(derived from [[rephrasing-the-web]] §1). >50% perplexity는 ACL2024 abstract/body와 맞지만 arXivv1 abstract는 “more than10%”라고 해 version conflict([[excerpts/rephrasing-the-web]]).
22. WRAP “300 token보다 길면 information drop,Q/A denser fact,reproducible held-out Mistral rephrase” → >300 input은 often info loss; author는 downstream evaluation style reflect 및 higher quality로 gain을 attribute. leakage analysis는 real-synthetic pair SimCSE cosine을 baseline과 compare([[rephrasing-the-web]] §3.1,§7 RQ3).
23. Humpback “self-curation external보다 better, own output calibration fresh RM보다 better”, “aligned seed model prompt” →250-example dev에서 M0 precision/recall0.44/0.09,M1 0.52/0.44,GPT-4 0.88/0.92; reward model comparison 없음. instruction은3,200 seed pair로 fine-tune한 backward model에서 온다([[humpback]] §2.2,App.B Table9).
24. §9 common cost/diversity table 및 “Persona-Hub0.72 biggest diversity gain” → source에 common diversity measurement/value 없다. reported cost: Self-Instruct generation~$600(App.A.2),Alpaca<$500([[alpaca]] blog),Magpie Air/Pro instance1,000당$0.12/$1.1([[magpie]] §3.4),WRAP85B token~25K A100 hour([[rephrasing-the-web]] §9.1). embedding diversity는 OOD accuracy와 opposite rank 가능([[prismatic-synthesis]] Table2).
25. “ch-18 generator-verifier stack” → ch-18은 generate,filter,deduplicate,verify,select,mix의 six-stage loop를 가르친다.

## 이 chapter가 general-purpose model에 중요한 이유

여기서 비교하는 SFT dataset(Alpaca,WizardLM,WizardCoder,WizardMath,Magpie,Persona Hub math,Humpback)은 아래 method로 만들었고 WRAP은 same idea를 pretraining text에 apply한다. generation method는 prompt distribution(task,topic,format,language,difficulty)과 response distribution(student가 imitate할 style/knowledge) 두 개를 fix한다. general-purpose model에는 user ask를 cover하는 prompt와 style만이 아니라 capability를 transfer하는 response가 필요하다. 이 chapter는 ch-18 loop의 generate stage에 있고 quality/diversity selection(ch-22), preference optimization/RL보다 앞선다. method마다 mechanism, author evaluation, breadth/narrowing evidence를 제시한다. §2-§8은 studied version의 method order를 유지하며 unsourced comparison table은 §9(imitation versus capability),§10(judge length bias),§11(real user prompt coverage),§12(method compose evidence table)로 replace한다.

## §1 용어

- **SuperNI**: human-written instruction이 있는 NLP task collection인 Super-NaturalInstructions; Self-Instruct는 training에 쓰지 않은 task119개에서 evaluate한다.
- **NLL**: negative log-likelihood, −log p(x)를 token 또는 example에 average한 것; lower면 model이 data에 higher probability를 assign한다.
- **AE2 LC**: AlpacaEval2 length-controlled win rate(§10). **RtA**: refuse-to-answer rate, model이 decline하는 prompt share(jailbreak prompt에는 higher가 safer).
- **Seed set**: generation을 condition하는 human-written example(Self-Instruct task175개,Evol-Instruct Alpaca instruction52k).
- **Generator/evolver**: new prompt를 쓰는 model; **response generator**: training target을 쓰는 model. 다를 수 있다(Magpie App.F.5는 response generator만 swap).
- **Prompt distribution**: dataset에 prompt 종류별로 나타날 probability. **Coverage**는 trained model이 reference distribution prompt에 assign하는 probability 양이며 WildChat은 held-out user prompt NLL로 measure한다(§11).
- **Imitation data**: stronger model이 쓴 response를 weaker model의 SFT target으로 사용하는 data([[false-promise-imitating-proprietary-llms]] Abstract).
- **Judge win rate**: fixed prompt set에서 baseline response보다 model response를 LLM judge가 prefer할 probability의 average; **LC win rate**는 length difference가 설명하는 부분을 remove한다(§10).

| Method(arXivv1) | Prompt source | Response source | Source의 main evaluation |
|---|---|---|---|
| Self-Instruct(2022-12) | seed175개 +GPT-3 davinci | GPT-3 davinci | SuperNI unseen task; human-rated novel instruction252개 |
| Alpaca(blog2023-03) | seed175개 +text-davinci-003 | text-davinci-003 | author5명, text-davinci-003과 blind pairwise |
| Evol-Instruct / WizardLM(2023-04) | gpt-3.5-turbo가 rewrite한 Alpaca52k | gpt-3.5-turbo | AlpacaEval,MT-Bench,WizardEval(GPT-4 judge) 포함 benchmark9개 |
| WizardCoder(2023-06) | gpt-3.5-turbo가 rewrite한 Code Alpaca20k | gpt-3.5-turbo | HumanEval,HumanEval+,MBPP,DS-1000,MultiPL-E |
| WizardMath(2023-08) | GPT-4가 rewrite한 GSM8k+MATH train | GPT-4-0613 | GSM8k,MATH |
| Humpback(2023-08) | ClueWeb text 위 backward model | web text 자체 | AlpacaEval,human preference,MMLU |
| WRAP(2024-01) | C4 document | Mistral-7B-Instruct rephrase | Pile perplexity; zero-shot task13개 |
| Magpie(2024-06) | user header만 받은 Llama-3-Instruct | same model family | AlpacaEval2LC,Arena-Hard,Open LLM Leaderboard task |
| Persona Hub(2024-06) | persona로 prompt한 GPT-4 | gpt-4o solution(math) | MATH(OOD),synthetic test |

[figures/generation-methods.html](figures/generation-methods.html)의 PanelD는 이 table method별 pipeline step과 main reported number를 list하므로 §2-§8을 읽으며 method를 step-by-step compare할 수 있다.

## §2 Bootstrap: Self-Instruct

**정의.** Self-Instruct는 small task pool로 condition한 language model에서 instruction data를 generate하고 accepted output을 그 pool에 다시 add한다([[self-instruct]]).

**문제.** human-written instruction set은 “limited in quantity, diversity, and creativity”(Abstract)이며 measurable target은 private annotation 없이 unseen task의 zero-shot accuracy였다.

**메커니즘**(§2.2):

1. pool을 task175개(instruction1개,instance1개씩)로 시작한다.
2. in-context example로 instruction8개(human-written6개,model-generated2개)를 sample하고 model이 “Task9:”를 continue하게 한다.
3. new instruction이 classification task인지(small,limited label space를 가진 task) model에게 묻는다.
4. instance를 generate한다. classification task에는 class label을 first generate한 뒤 label별 input을 generate한다(output-first). input-first는 “can generate inputs biased toward one label”이고 grammar error detection에서는 “usually generates grammatical input”이기 때문이다. other task에는 input first를 generate한다.
5. filter: every existing instruction과 ROUGE-L similarity가0.7 below일 때만 instruction을 add하고 image,picture,graph 같은 keyword instruction을 drop하며 identical instance와 same-input/different-output instance를 drop하고 too long/short instruction 및 input을 repeat하는 output을 drop한다.

**Formula.** ROUGE-L은 token sequence a(candidate),b(reference)의 longest common subsequence(LCS)를 쓴다.

```
P = LCS(a, b) / |a|     R = LCS(a, b) / |b|     F = 2 P R / (P + R)
```

`|a|`,`|b|`는 token count, P는 precision,R은 recall,F는 F-measure다. Alpaca code는 F를0.7과 compare한다(`generate_instruction.py` L196). Self-Instruct paper는 variant를 name하지 않고 “ROUGE-L similarity”라고 한다.

**Worked example.** candidate “write a short poem about the ocean”(token7개)과 pool instruction “write a poem about the sea”(token6개)의 LCS는“write a poem about the”=5다. P=5/7=0.714,R=5/6=0.833,F=0.769>0.7이므로 candidate는 reject된다. candidate “explain why the sea is salty”(token6개)는 LCS “the sea”=2,F=0.333이므로 accept된다. filter는 lexical near-copy를 reject하지만 same skill을 다른 word로 ask하는 new instruction은 reject하지 않는다. [figures/generation-methods.html](figures/generation-methods.html)의 PanelA는 any two instruction의 F-measure와 accept/reject decision을 compute한다.

**근거.** GPT-3 davinci(175B)를 OpenAI API로 instruction52,445개,instance82,439개에서2 epoch fine-tune했다. SuperNI ROUGE-L은6.8→39.9이며 InstructGPT001은40.8이었다(Table3). author가 rate한 newly written user-oriented instruction252개에서 “acceptable with minor imperfections”도 valid로 count하면 model은 InstructGPT001보다5% behind다(§4.4). sample200개 audit은 valid instruction92%지만 all field valid는54%만 찾았다(Table2). human-evaluation gain은 instruction16K 뒤 “almost plateau”하고 SuperNI gain은 “at around hundreds”에서 plateau한다(§4.5). cost는 generation 약$600, fine-tuning$338(App.A.2-A.3)이다. **Result(single study).**

**조건과 한계.** instance quality는 verify하지 않았다. audited example46%에는 invalid field가 적어도 하나 있다. seed pool은 closest SuperNI test instruction과 average ROUGE-L0.21, user-oriented set과0.34이며 identical instruction 하나가 있다(App.A.1). same released data로 LLaMA13B를 train한 [[tulu-1-how-far-can-camels-go]]에서 five capability benchmark 모두 base보다 낮았다. MMLU42.3→30.4,GSM14.5→11.0,BBH39.3→30.7,TyDiQA43.2→41.3,Codex-Eval P@10 28.6→12.5(Table3).

**함의.** novel task instruction252개 human evaluation은 judge win rate보다 generality measurement에 더 가깝다. Tülu1 result는175B base에 도움 된 data가 different13B base에서는 accuracy를 낮출 수 있음을 보인다.

## §3 Low-cost replication으로서의 Alpaca

[[alpaca]](blog2023-03)는 Self-Instruct의 네 부분을 바꿨다. generator로 text-davinci-003, call당 instruction20개를 request하는 new prompt, instruction당 instance1개, classification split 없음이다(README “Data Generation Process”). released code는 word3개 이하 또는150개 초과, blacklisted word, non-ASCII start, ROUGE-L F >0.7 instruction을 reject하며 output은 check하지 않는다(`generate_instruction.py`@761dc5b). LLaMA7B는 prompt token을 mask해 example52K에서3 epoch,LR2e-5로 fine-tune했다. data cost는$500 below, training은$100 below(80GB A1008개에서3시간)였다(blog;README). only evaluation은 student author5명이 Self-Instruct evaluation set에서 text-davinci-003과 output을 blind compare한 것이며90 win 대89 win이었다(blog). author는 evaluation이 “may be limited in scale and diversity”이고 hallucination이 “even compared to text-davinci-003” common하다고 한다.

**Breadth 근거.** Tülu1의13B에서 Alpaca data는 MMLU42.3→45.0,GSM14.5→9.5,BBH39.3→36.6,TyDiQA43.2→31.1,Codex-Eval28.6→29.9로 바꿨다([[tulu-1-how-far-can-camels-go]] Table3). MMLU는2.7,Codex-Eval은1.3 올랐고 other three는 fall했으며 multilingual QA는12.1 point를 잃었다. Alpaca prompt52,002개는 language1개,user token average19.67인 반면 WildChat은 language68개,295.58 token이다([[wildchat]] Table1). **함의.** low-cost replication은 teacher response style과 seed pool prompt distribution을 inherit한다. teacher와 pairwise judgment는 둘 중 어느 것도 test하지 않는다.

## §4 Evolution: Evol-Instruct, WizardCoder, WizardMath

### §4.1 Evol-Instruct

**정의.** Evol-Instruct는 LLM으로 each instruction을 round마다 more complex 또는 rarer instruction으로 rewrite하고 all round에서 train한다([[evol-instruct]], WizardLM paper).

**문제.** human-written set에는 high-difficulty instruction이 적다. ChatGPT judge의 mean difficulty(1-10)는 Alpaca3.00,ShareGPT4.63이다(App.I Table3).

**메커니즘**(§3.2,§4.2):

1. each round의 instruction마다 prompt6개 중 하나를 equal probability로 sample한다.
2. in-depth(prompt5개): constraint add,deepening,concretizing,reasoning step increase,input complicating(last는 XML,SQL,Python,HTML,shell,JSON in-context example 사용). rewrite마다10-20 word만 add할 수 있다. add-constraint prompt는 “Please add one more constraints/requirements into #Given Prompt#”(Example3.1)다.
3. in-breadth(prompt1개): “This new prompt should belong to the same domain as the #Given Prompt# but be even more rare”(Example3.3).
4. instruction alone으로 gpt-3.5-turbo가 response를 generate한다.
5. failure(no information gain ChatGPT judge; “sorry”와80 word below; punctuation/stop word만; prompt word copy)를 eliminate하고 failed instruction은 pool로 return한다.
6. Alpaca seed52k에서 M=4 round 뒤(instruction250k,API request624k), all round에서70k를 SFT sample한다.

**근거.** 70k sample의 LLaMA13B는 benchmark9개 average58.96으로 Vicuna-13b(ShareGPT70k)54.60,retrained Alpaca-13b43.44보다 높다(Table1). mean difficulty는 round1의5.48에서 round4의7.08로 오른다(Table3). evolver를 Llama-2-70B-Chat으로 replace하면 average56.27이다(Table2). **Result(single study).**

**조건과 한계.** seed는 improve할 skill을 바꾼다. ShareGPT seed는 nine-benchmark average61.87을 냈지만 GSM8k를37.15에서31.46으로 낮췄고 author는 ShareGPT math instruction4.3% 대 Alpaca11.8%로 attribute한다(§4.5,Table2). Vicuna-13b는 TruthfulQA에서 더 높았다(52.68 대50.55)(Table1). contamination check는 report되지 않았고(card Verification), metric9개 중3개는 GPT-4 judged다.

### §4.2 WizardCoder: code heuristic과 round selection

[[code-evol-instruct]](WizardCoder paper,arXiv2023-06)는 prompt6개를 “Please increase the difficulty of the given programming test question a bit” template 하나와 §3.1의 heuristic5개로 replace한다.

```
(1) Add new constraints and requirements to the original problem, adding approximately 10 additional words.
(2) Replace a commonly used requirement in the programming task with a less common and more specific one.
(3) If the original problem can be solved with only a few logical steps, please add more reasoning steps.
(4) Provide a piece of erroneous code as a reference to increase misdirection.
(5) Propose higher time or space complexity requirements, but please refrain from doing so frequently.
```

deepening,input complicating,in-breadth evolving은 remove했다(v1 §3.1). each round 뒤 evolved data 모두를 Code Alpaca seed(~20k)와 merge한다. round1-4 cumulative size는38k,58k,78k,98k이고78k set이 final이다(v1 §4.5). **Complexity 대 quantity**(Table5,HumanEval pass@1,each round separately trained): matched sample count에서 seed20.0k→45.7,round1 18.8k→56.1,round2→53.0,round3→54.3,round4→51.2다. matched2.3M token에서는44.5,51.8,52.4,50.0,49.4다. **Worked reading.** round1은 seed sample count94%(18.8/20.0)에서10.4 point 더 높으므로 gain은 more example 때문이 아니다. matched token에서 round2 뒤 decline은 more round가 monotonic better가 아님을 보인다.

**Held-out breadth.** WizardCoder34B는 MultiPL-E language8개 모두에서 CodeLlama-Instruct-34B above다(Table2). WizardCoder15B는 HumanEval57.3,MBPP51.8(Table1)다. **Measurement error.** v1은 reported test benchmark HumanEval pass@1으로 round/model을 choose했고 v2는 external dev set(MBPP-400)을 describe한다(v1 §3.2;v2 §3.2,§5). decontamination은 test item당 gte-large embedding top5 training sample을 retrieve하고 GPT-4가 match를 decide한다(App.C). non-code task effect는 report하지 않는다.

### §4.3 WizardMath: downward/upward evolution

[[wizardmath]](arXiv2023-08;v3=ICLR2025)는 constraint add,concretizing,reasoning increase의 upward alongside “revising high difficulty questions to lower difficulty, or ii) producing a new and easier question with another different topic” downward direction을 add한다. GPT-4는 GSM8k,MATH train question each를 round5개(2 downward,3 upward) evolve하여 unique instruction448k를 만들고 possible contamination30k를 remove해 GPT-4-0613 step-by-step answer 포함418k를 남긴다(v3 §3.1,§4.1; [[excerpts/wizardmath]]).

**Worked example**(Table6,Mistral-7B SFT,pass@1). original7.5k data는 GSM8K59.7,MATH15.1이다. downward round2 add는+14.8,+19.6; upward round3 add는+18.9,+27.4; all5 add는+21.5,+31.1(81.2,46.2;§4.3 text는46.5 print)이다. combined gain은 each direction alone보다 크지만 sum보다 작다(14.8+18.9=33.7>21.5). 이는 two direction이 일부 same thing을 teach함과 consistent하다(Interpretation). process reward model과 instruction reward model RL은 Mistral-7B를82.8/48.1에서90.7/55.4로 올린다(Table3). reward model 둘 다 GPT-4-0613 label로 train했고(§4.1) process supervision은 ch-44가 다룬다.

**Narrowing.**2023 release에서 [[mammoth]]는 WizardMath-70B가 AQuA20.0(Llama-2-70B40.9),SAT-Math13.2(51.3)이고 GSM8K는81.6이라 report한다(Table3-4). **함의.** domain evolution은 in-domain accuracy를 올리지만 same broad area의 out-of-domain task도 fall할 수 있으므로 둘 다 evaluation에 넣어야 한다.

## §5 Extraction: Magpie

**정의.** Magpie는 aligned model에 user message 전 chat template 부분만 보내 user instruction을 generate한 뒤 full template로 response를 generate한다([[magpie]] §2).

**문제.** open data-creation method에는 human seed 또는 hand-written prompt가 필요해 scale과 scope를 limit한다(Abstract).

**메커니즘**(§2.1,App.E.1):

```
x = T_pre-query ⊕ q ⊕ T_post-query
T_pre-query  = <|start_header_id|>user<|end_header_id|>
T_post-query = <|eot_id|><|start_header_id|>assistant<|end_header_id|>
Step 1: input T_pre-query only; sample until end-of-sequence  → instruction q
Step 2: input T_pre-query ⊕ q ⊕ T_post-query; greedy decoding → response
```

`q`는 generated user query이고 ⊕는 concatenation이다. instruction sampling은 temperature{1.0,1.1,1.2,1.25}와 top-p{1.00,0.995,0.990}를 mix한다. higher value는 quality를 slightly lower하고 difficulty/diversity를 raise했다(App.D.3). author는 alignment 중 instruction loss가 masked여도 model이 user turn을 implicitly memorized했다고 hypothesize한다(§2.1 Remark; Interpretation). filter metric8개는 input length,output length,task category,input quality,input difficulty,minimum neighbor distance(all-mpnet-base-v2+FAISS),FsfairX-LLaMA3-RM-v0.1의 reward r*,base-model response reward r_base와의 difference r*−r_base다(App.C,Table5). output-length filter는 longest response를 keep한다.

**근거**(Llama-3-8B-Base SFT,2 epoch; Magpie set은 conversation300K,baseline size는 listed). AlpacaEval2LC는 Magpie-Pro-300K-Filtered25.08,Magpie-Air-300K-Filtered22.66,WildChat652K14.62,Evol Instruct143K8.52,Self-Instruct(Llama-3)100K7.21,Llama-3-8B-Instruct22.92다(Table1). Arena-Hard는18.9,14.9,8.7,5.1,4.0,20.6이다. compute는4×A100에서 Air206 GPU-hour,Pro614 GPU-hour이고 instance1,000당$0.12,$1.1이다(§1,§3.4). **Result(single study).**

**조건과 한계.**

- Reasoning: Open LLM Leaderboard average61.58 대 Llama-3-8B-Instruct66.13,GSM8K47.92 대71.72(Table3). math,code,reasoning data150K를 add하면 GSM8K63.08.
- Filtering 대 scale: Air3M-Raw22.96LC 대300K-Filtered22.66;Pro300K-Filtered25.08 대1M-Raw24.16(Table12). filter configuration 하나가 all benchmark에서 best는 아니다(Table13).
- Response generator: same instruction의 Qwen2-7B-Instruct response는 LC22.66을15.01로 낮추므로 gain 일부는 response model에 속한다(Table14).
- Inherited prompt prior: Magpie-Pro instruction “over half”가 information seeking이고(§3.2),Llama-Guard-2는 Air99.128%,Pro99.347%를 safe로 label한다(Table6). coverage analysis는 all synthetic인 Magpie-Pro,Alpaca,Evol Instruct,UltraChat을 t-SNE로 compare하며(§3.1) real user prompt comparison은 report하지 않는다(§11).
- Safety behavior: SFT model은 Llama-3-8B-Instruct보다 jailbreak을 덜 refuse한다(RtA0.80 대0.93,Table15).

**함의.** extraction은 aligned model의 user-turn distribution을 reproduce한다. resulting SFT model은 instruction-following judge에서 잘하고 math reasoning에서 trail하며 author는 small reasoning instruction share로 attribute한다(§4.2,Interpretation). Llama-Guard-2 기준 unsafe prompt는 Magpie data1% below인 반면 OpenAI moderation API는 LMSYS-Chat-1M conversation5%를 flag한다(§11). classifier가 다르므로 measured gap이 아니라 check할 lead다.

## §6 Persona conditioning: Persona Hub

**정의.** persona-driven synthesis는 LLM이 해당 persona 관점의 data를 쓰도록 persona description을 data-synthesis prompt에 insert한다([[persona-hub]] Abstract).

**메커니즘**(§2.1-§2.3):

1. Text-to-Persona: RedPajama v2 document에 대해 LLM에게 “Who is likely to [read|write|like|dislike|...] the text?”를 묻는다.
2. Persona-to-Persona: “Who is in close relationship with the given persona?”를 iteration6번 묻는다.
3. 1-gram MinHash(signature128,threshold0.9)로 deduplicate하고, embedding cosine similarity0.9 above persona를 remove한다(count보다 diversity가 중요할 때0.5 suggested). 이어 heuristic quality filter를 apply한다. result는 persona1,015,863,523개다.
4. persona+task(zero-shot,few-shot,persona-enhanced few-shot)로 model을 prompt한다.

**근거.** GPT-4는 MATH instance 없는 persona1.09M개에서 math problem1.09M개를 썼고 gpt-4o가 solution을 썼다.1.07M에서 fine-tune한 Qwen2-7B는 greedy MATH64.9%로 gpt-4-turbo-0125-preview64.5%보다 높다(Table2). math expert2명은 challenging problem200개 중7개를 invalid로 judge했다(96.5% valid). similarity study(level별 persona pair100개,temperature0,text-embedding-3-small)에서 problem similarity는 persona similarity와 “tends to be correlated with but lower than”하며 prompt가 topic constraint를 add하면 rise하고 persona similarity0.9에서 대부분0.6-0.75이다(§4.1.2,Fig.10). **Result(single study).** SFT hyperparameter는 report되지 않았다.

**조건과 한계.** lower embedding similarity는 different skill coverage와 같지 않다. [[prismatic-synthesis]](Qwen2.5-72B-Instruct math sample10k)에서 seed0.1k/persona10k set은 seed10k/persona0.1k set보다 embedding Vendi score가 높았지만(24.49 대22.03; Vendi는 similarity matrix eigenvalue entropy의 exponential이며 all item dissimilar면 item 수와 같다) OOD accuracy는 낮았다(38.15 대54.77)(Table2). Persona Hub author도 in-distribution synthetic-test result는 “should be taken as a reference only”라 한다(§4.1.2). **함의.** persona conditioning은 surface context를 vary한다. 위 evidence는 math transfer에서 underlying problem(seed)을 vary하는 것이 더 중요했음을 보인다. 이 comparison의 gradient-based diversity measure는 ch-22가 가르친다.

## §7 Rephrasing: WRAP

**정의.** Web Rephrase Augmented Pretraining(WRAP)은 instruction-tuned model에 specified style로 web document paraphrase를 prompt하고 real/rephrased text를 함께 pretrain한다([[rephrasing-the-web]];[[excerpts/rephrasing-the-web]]).

**메커니즘**(§3.1-§3.2):

1. NLTK sentence splitter로 C4 document를 약300 token chunk로 split한다. longer input은 “often led to a loss of information”.
2. frozen Mistral-7B-Instruct로 each chunk를 Easy,Medium(Wikipedia-like),Hard(terse),Q/A(conversation question-answer format) four style 중 하나로 rephrase한다.
3. real/synthetic data를1:1 sample하므로 document each가 raw once,rephrased once seen다.
4. sequence length1024에서 decoder-only model(128M,350M,1.3B)을 pretrain한다.

**근거.** ACL abstract는 about3× speed-up,average Pile perplexity50% 초과 improve,task13개 zero-shot accuracy2% 초과 higher를 report한다. arXivv1 abstract는 perplexity “more than10%”, body는50%라 하므로 **version conflict**다. C4 15%의 real+synthetic rephrase로 train한350M model은 C4 all에서 train한1.3B를 outperform한다(§1).

**Worked example: task13개 average 읽기**(Table1,1.3B,WRAP row는 run3개 average; §7 RQ2는 Table1 synthetic data가 QA prompt로 generate됐다고 함). real token85B의 Synthetic+C4는 general task8개 average52.3,real token170B Full C4는50.3이다. task별 difference ARC-E+2.8,BoolQ+8.0,Winogrande−0.1,PIQA−0.1,HellaSwag−0.4,TruthfulQA+7.1,OBQA−1.5,LogiQA+0.2다. task2개가(8.0+7.1)/8=1.9로2.0-point gap을 contribute하고 task8개 중4개는 lower다. specialized-knowledge task5개 average는44.0 대42.4이며1T token TinyLlama는44.6이다. author는 synthetic data가 “can not impart ‘new knowledge’. It can only help pre-train faster”라고 conclude한다(§5.2).

**조건과 한계.**

- evaluation-matched style: author는 gain을 “that closely reflects downstream evaluation style”에 attribute한다(Abstract). Fig.1(b,c)는Q/A style,§7 RQ2는 Table1 synthetic data를 QA-prompt data라 하고 zero-shot task는 QA다.1.3B/150B-token run에서 Medium only average47.9,Q/A only51.0(Table2). 이는 generality의 benchmark-format confound다(Interpretation).
- Real data: synthetic-only training은 many sub-domain Pile perplexity를 degrade했고 author는 clean synthetic text에 real web text special character/structure가 없기 때문이라 attribute한다(§7 RQ2,Fig.4).
- Rephraser: Qwen-1.8B,Mistral-7B rephrase는 Vicuna-13B보다 lower perplexity였고 every rephraser가 real C4 alone보다 better였다(RQ1,Fig.3).
- Cost: token85B에 token/hour3M으로 A100 hour~25K(§9.1).

**함의.** rephrasing은 information content가 아니라 style을 바꾼다. rephrase style과 unlike한 domain perplexity 및 per-task result로 breadth를 measure한다. larger-scale textbook-style synthesis는 ch-21이 다룬다.

## §8 Backtranslation: Humpback

[[humpback]](arXiv2023-08)은 Open Assistant seed pair3,200개에서 LLaMA를 reverse direction으로 fine-tune해 backward model p(x|y)를 얻고, ClueWeb segment502k 각각에 candidate instruction 하나를 generate하며, seed-trained model이5-point prompt로 each pair를 rate하게 한다. iteration2번 뒤 top-score set A5에는 pair41,821개가 있다(§2.2-§2.3,Table1). seed+curated data에서 train한 LLaMA65B는 text-davinci-003 대비 AlpacaEval win rate83.71%에 도달한다(Table3). uncurated augmented data는 size가 grow해도 win rate를 올리지 못했다(§3.3,Fig.2). positive20%의 author-labeled250-example dev set에서 curation precision/recall은 M0 0.44/0.09,M1 0.52/0.44,GPT-4 0.88/0.92였다(App.B,Table9). **함의.** response가 teacher가 아니라 human web text에서 오므로 teacher style transfer를 avoid한다. self-curator는 iteration0에서 good pair 대부분을 discard한다(recall0.09). evaluation은 judge-based다.

## §9 Imitation 대 capability

[[false-promise-imitating-proprietary-llms]](arXiv2023-05)는 output-only loss로 ChatGPT output0.3M-150M token(ShareGPT-Mix:ShareGPT~50K,HC3 27K,Discord example10k)을 GPT-2 1.5B,LLaMA7B,13B에 fine-tune했다([[excerpts/false-promise-imitating-proprietary-llms]]). crowdworker는 imitation output 약70%를 ChatGPT와 equal 또는 better로 rate했다(Fig.1). 5-shot MMLU,3-shot Natural Questions,0-shot HumanEval에서 broad imitation model은 “do not improve (or even decline) in accuracy as compared to the base model, even when adding additional imitation data”(§4.3,Fig.4 top).

**Worked example**(Table1,NQ accuracy). LLaMA7B는 base17,ShareGPT-Mix10(−7),NQ-synthetic22(+5)다. LLaMA13B는20,15(−5),27(+7)이며 ChatGPT는31이다. targeted set은 seed QA pair10개에서 generate한 example6,000개이고 broad model은 token150M까지 train했다. token150M에서 imitation model은 ChatGPT가 list를 쓸 때81%에서 list를 사용한다(base13%). ChatGPT와의 length correlation은−0.11에서0.62로 rise한다(Table2). imitation data increase보다 base size increase가 accuracy를 더 raise했다(Fig.4 bottom).

**해석과 한계.** author는 imitation data가 knowledge가 아니라 style을 transfer하며 regression은 “distribution shift and tension between the conversational-style fine-tuning data and the downstream benchmarks”에서 올 수 있다고 conclude한다(§4.3). evaluator로 GPT-4를 써도 crowdworker와 same trend였으므로 GPT-4 judge는 gap을 detect하지 않는다(§4.4). [[tulu-1-how-far-can-camels-go]]는13B에서 complementary result를 report한다. GPT4-Alpaca data는 AlpacaEval63.1%지만 TyDiQA23.5(base43.2)이고 win rate는 unique response token average와r=0.96 correlation이다(§5.4,Table3). Tülu1 author는 imitation data가 diverse skill을 cover할 때 도움이 된다고 argue한다(§6;Interpretation). ch-29e는 instruction tuning general result로 이를 다룬다.

## §10 Confound로서의 judge-score length bias

**정의.** AlpacaEval은 GPT-4-turbo judge로 fixed instruction805개에서 model response와 baseline response를 compare한다. win rate는 mean preference probability다([[length-controlled-alpacaeval]] §2).

**문제.** gpt4_1106_preview를 verbose 또는 concise하게 prompt하면 win rate가22.9%에서64.3%로 움직였다(§4.1). 이 chapter method 둘은 long response를 select/produce한다(Magpie는 longest response keep,imitation data는 ChatGPT length correlation raise). 따라서 raw win-rate gain은 length에서 올 수 있다.

**Formula**(§3,Eq.1-2):

```
q(y=1) = logistic( θ_m − θ_b + φ_{m,b} · tanh( Δlen / std(Δlen) ) + (ψ_m − ψ_b) γ_x )
winrate_LC = 100 · E_x[ logistic( θ_m − θ_b + (ψ_m − ψ_b) γ_x ) ]
```

θ_m,θ_b는 model term; Δlen =len(z_m)−len(z_b)는 model/baseline response length difference;φ_{m,b}는 fitted length coefficient;γ_x는 instructionx difficulty,ψ는 model별 weight다. LC win rate는 length term을 zero로 둔다.

**Worked example**(illustrative parameter; any model fit 아님). θ_m−θ_b=0,instruction term=0,φ=1.0,response가 baseline보다 one standard deviation longer라고 하자. raw predicted preference=logistic(tanh(1))=logistic(0.762)=0.682이므로 raw win rate68.2%; LC=logistic(0)=50%다. θ_m−θ_b=0.5면 raw=logistic(1.262)=77.9%,LC=62.2%다. [figures/generation-methods.html](figures/generation-methods.html)의 PanelC는 chosen θ,φ,length difference에서 rate 둘을 recompute한다.

**근거.** length control은 verbosity prompt across normalized standard deviation을25%에서10%로 lower하고 Chatbot Arena Spearman correlation을0.94에서0.98로 raise했다(model38개; bootstrap p=0.07 대AlpacaEval)(§4.1-§4.2,Table1). length control 아래 largest rank loss는 RLHF-trained open model이었다(§4.2). truncation attack은 regularization 없이 GPT-4 LC score를3.7에서25.9,regularization으로12.2로 raise했다(§4.3).

**한계.** LC는 mediator 하나(length)만 remove한다. list use,tone,factual error는 남고 Chatbot Arena 자체도 surface feature를 reward할 수 있다(§2). **함의.** Magpie는 LC win rate를 report하지만 Evol-Instruct는 length-controlled인지 밝히지 않은 AlpacaEval score를 report한다([[evol-instruct]] Table1). paper 사이 compare에는 same metric이 필요하고 judge result에는 benchmark-based check가 beside 있어야 한다(ch-49).

## §11 Real user prompt에 대한 coverage

**정의.** generated prompt set은 이 data로 trained model이 real distribution held-out prompt에 high likelihood를 assign할 때 real distribution을 cover한다. WildChat은 dataset별 fine-tuned Llama-2 7B 아래 first-turn user prompt average NLL로 이를 measure한다(train70%,validation30%)([[wildchat]] §3,Fig.3;[[excerpts/wildchat]]).

**Real distribution.** WildChat은 opt-in ChatGPT conversation1,039,785개,IP 기준 user204,736명,language68개,turn2.54다(Table1). English first-turn category는 assisting/creative writing61.9%,analysis/decision explanation13.6%,coding6.7%,factual information6.3%,math reasoning6.1%다(Table4). [[lmsys-chat-1m]]은2023-04부터08까지 IP210K에서 model25개와 conversation1,000,000개다(Abstract,§3.1). English prompt100K의 k-means cluster20개에서 coding cluster sum29.64%,unsafe cluster3개12.28%(Fig.3에서 derived),conversation5%가 OpenAI moderation API로 flag됐고(§3.3),cluster2개는 script-generated template prompt라 author는 figure가 “might not reflect the real-world topic distributions”라 한다([[excerpts/lmsys-chat-1m]]).

**Worked example**(Fig.3 NLL; printed unit). WildChat prompt에서 WildChat train5.18,Open Assistant6.57,ShareGPT7.61,Dolly8.77,Alpaca10.87이다. Alpaca-trained model은 in-distribution model보다10.87−5.18=5.69 worse다. Alpaca prompt에서 Alpaca train2.24,WildChat3.28(1.04 worse)다. asymmetry5.69 대1.04가 coverage gap이다. 이는 WildChat distribution이 Alpaca kind prompt를 contain하지만 Alpaca distribution에는 WildChat kind가 lack함과 consistent하다(Interpretation). [figures/generation-methods.html](figures/generation-methods.html) PanelB는 full matrix와 any pair gap을 보여준다.

**Method별 측정.**

- Self-Instruct: test set seed overlap(App.A.1),novel-instruction human eval; real-user comparison 없음.
- Evol-Instruct: BERT embedding t-SNE, k-means cluster20개 qualitative(App.J); WizardEval author-built real-world instruction218개(§4.4).
- Magpie: synthetic dataset3개와 t-SNE(§3.1); WildChat SFT는 coverage reference가 아닌 baseline.
- Persona Hub: generated item similarity; instruction use case는 inferred persona가 있는 WildChat demonstration 사용(§4.3).
- WRAP: Pile domain21개 perplexity이며 prompt가 아닌 text coverage measurement.

**한계.** Magpie(information seeking),WildChat,LMSYS category share는 taxonomy/classifier가 달라 controlled comparison이 아니다(Open question). WildChat은 opt-in free ChatGPT service user,LMSYS는 public demo/arena user를 reflect하며 둘 다 toxic prompt를 include하고 LMSYS는 script-generated prompt도 include한다. log 자체가 SFT data가 될 수도 있다. WildLlama(WildChat의 Llama-2 7B)는 Vicuna7B6.13 대비 MT-bench6.35(Table9)다. LMSYS HighQuality-7B(OpenAI/Anthropic response conversation45K,token33M)는 Vicuna-7B-v1.5(token370M)의 MMLU49.8,MT-Bench6.17 대비47.7,6.03을 score한다([[lmsys-chat-1m]] Table6). human-written data prompt-distribution coverage는 ch-15도 다룬다.

## §12 Method compose: evidence table

| Method | Vary하는 것 | Breadth 근거(source) | Narrowing 근거(source) |
|---|---|---|---|
| Self-Instruct | seed pool의 new task | GPT-3에서 SuperNI ROUGE-L+33.1; novel-task human eval([[self-instruct]] Table3,§4.4) | LLaMA13B에서 benchmark5개 모두 fall([[tulu-1-how-far-can-camels-go]] Table3) |
| Alpaca | same,stronger generator | author pairwise eval beyond none([[alpaca]]) |13B에서 TyDiQA−12.1,GSM−5.0(Tülu1 Table3) |
| Evol-Instruct | existing prompt difficulty/rarity | nine-benchmark avg58.96 대54.60([[evol-instruct]] Table1) | seed choice가 GSM8k37.15 ↔31.46 move(Table2) |
| WizardCoder | code difficulty | MultiPL-E language8/8개 CodeLlama-Instruct-34B above([[code-evol-instruct]] Table2) | non-code report 없음;v1 test set에서 round select |
| WizardMath | both direction difficulty | original 대비 GSM8K+21.5,MATH+31.1([[wizardmath]] Table6) |70B AQuA20.0 대base40.9([[mammoth]] Table3) |
| Magpie | aligned model user-turn prior | official22.92 대비 AE2LC25.08([[magpie]] Table1) |71.72 대비 GSM8K47.92(Table3) |
| Persona Hub | persona context | MATH data 없이 MATH64.9%([[persona-hub]] Table2) | persona-diverse set OOD가 seed-diverse보다 lower([[prismatic-synthesis]] Table2) |
| WRAP | real document style | half real token에서 general task8개 avg52.3 대50.3([[rephrasing-the-web]] Table1) | synthetic-only Pile perplexity worse(Fig.4);new knowledge 없음(§5.2) |
| Humpback | real text용 instruction |65B AlpacaEval83.71%([[humpback]] Table3) |M0 self-curation recall0.09(Table9) |

이 chapter source 중 어느 것도 controlled composition의 several method에서 train하고 breadth를 measure하지 않으므로 table은 method를 rank하지 않는다(Open question). method는 data의 다른 part를 vary한다. which task exists(seed,real prompt), how hard(evolution), whose phrasing(persona,rephrasing), whose response is imitated(response generator)다. composition이 이 part를 더 cover하는 것은 Generalization lens measurement로 component each contribution을 check할 때뿐이다. ch-29 lab이 그런 set 하나를 build/filter한다.

## 음성 샘플과 음성 피드백

**네 meaning**(course standard): (1) positive target으로 쓰면 harm하는 sample인 negative marginal value, (2) input 또는 corrected target에 둔 failure인 content로서 negative, (3) control token 아래 train한 failure인 conditioning으로서 negative, (4) sample likelihood explicit decrease인 gradient로서 negative. sample에서 probability mass를 remove하는 것은(4)뿐이다.

**1. Negative가 오는 곳.** heuristic filter(Self-Instruct ROUGE-L,keyword,format;Alpaca length,blacklist),LLM judgment(ChatGPT Evol-Instruct no-information-gain check),learned score(Magpie reward/reward difference;Humpback self-rating),reward-ranked sample(Magpie-DPO rejected response)이다. label error도 있다. Humpback M0 curator precision0.44,recall0.09(App.B,Table9)이므로 dev set good pair91%를 discard했다(false negative). Self-Instruct filter는 audited example46%가 invalid field인 data를 통과시켰고(Table2) 이는 filter false positive다.

**2. Practice가 하는 일.** 위 every generation pipeline에서 discard(meaning1)하지만 Self-Instruct,Alpaca,Evol-Instruct는 discarded sample count를 report하지 않는다. gradient(meaning4)는 Magpie DPO extension에 나타난다. selected instruction each에 temperature0.8에서 k=5 response를 generate하고 ArmoRM-Llama3-8B-v0.1가 score하며 lowest-reward response가 rejected sample이다(§2.2,§4.1). meaning2,3은 이 source에 없으며 ch-31a 참고.

**3. Gradient case mechanism.** logit z의 next-token softmax에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이고 y는 rejected token,p_j는 tokenj probability다. `log p_y`에 sizeη descent step은 z_y를`−η(1 − p_y)`, every other z_j를`+η p_j`만큼 바꾼다. **Worked example.** token(a,b,y)의 p=(0.60,0.30,0.10),η=1이면 logit change는(+0.60,+0.30,−0.90),new probability는(0.710,0.263,0.026)다. penalize하지 않은 tokenb는0.037을 잃고 most likely tokena는0.110을 얻는다(computed). Magpie rejected response는 train 중인 Llama-3-8B SFT model이 아니라 generating aligned model에서 sample된다(§4.1). token이 policy 아래 low probability일 수 있으므로 위 gradient로 push down하면 mass는 mainly policy most-likely alternative로 간다(Interpretation;Magpie는 rejected response policy log-probability를 measure하지 않음). DPO derivation은 ch-39,displacement analysis는 ch-43a.

**4. 근거.** Benefit: Magpie-Pro SFT 뒤 Magpie-Pro-DPO pair100K의 Llama-3-8B는 SFT 뒤25.08/18.9 대비 AE2LC50.10,Arena-Hard35.7에 도달한다(Table1). chosen/rejected term 사이 gain은 split되지 않았고 DPOβ도 report되지 않았다. Discard: uncurated Humpback data는 size로 win rate improve하지 못했고(§3.3,Fig.2),Magpie-Air filtering3M→300K도 LC를 raise하지 않았다(22.96 대22.66,Table12). **Effect size.** 여기 어떤 source도 negative contribution improvement share를 measure하지 않는다.

**5. 통제.** curator를 trust하기 전에 labeled dev set에서 filter precision/recall을 measure한다(Humpback Table9). 가능하면 train 중 policy에서 rejected response를 draw하거나 positive NLL term을 keep한다. Evol-Instruct처럼 “sorry” rule을 short length(80 word below)에 conditional로 두어 apology를 포함한 full answer가 discard되지 않게 한다.

**6. 진단.** filter별 discard count를 report하고 DPO 중 chosen/rejected log-probability를 separately log하며 filtering 전후 refusal rate,response length를 track한다. length/refusal filter가 둘 다 change하기 때문이다.

**7. Generality에 미치는 효과.** refusal discard(Evol-Instruct “sorry” rule)나 longest response keep(Magpie output-length filter)은 refusal이 SFT target에 appear하는 frequency를 바꾼다. 어떤 source도 refusal behavior effect를 measure하지 않는다. Magpie SFT model jailbreak RtA는 Llama-3-8B-Instruct0.93 대비0.80(Table15)이나 paper는 이를 specific filter에 attribute하지 않는다(Open question). DPO rejected term이 calibration/pass@k에 미치는 effect도 report되지 않았다.

## 레시피

아래 row는 stated locus에서 읽은 exact release setting이다.

| Model(정확한 release) | Size | Stage | Setting | Value | Source 위치 | Status | 이 value의 근거 |
|---|---|---|---|---|---|---|---|
| GPT3-SELF-INST(GPT-3 davinci) |175B|SFT|data|instruction52,445;instance82,439|arXiv:2212.10560v2 Table1|verified 2026-09-15|§4.5 Fig.7: human-eval gain은 instruction16K 뒤 plateau|
| GPT3-SELF-INST |175B|SFT|epoch;prompt loss weight;other|2;0;undisclosed OpenAI API default|v2 App.A.3|verified 2026-09-15|loss weight “we find this works better in our case”(number 없음);training task overfit avoid 2 epoch|
| Self-Instruct data generation |175B generator|SFT(data gen)|in-context example;ROUGE-L rule|8(6 human,2 generated);pool all instruction에 <0.7이면 add|v2 §2.2|verified 2026-09-15|ablation 없음|
| Self-Instruct data generation |175B generator|SFT(data gen)|instruction decoding;instance decoding|temp0.7,top_p0.5,presence penalty2,max1024;temp0,presence penalty1.5,max300|v2 App.A.2 Table4|verified 2026-09-15|ablation 없음|
| Alpaca7B(LLaMA7B)|7B|distill-SFT|example;epoch;LR;batch;max length;weight decay|52K;3;2e-5;128;512;0|README@761dc5b “Fine-tuning”|verified 2026-09-14|ablation 없음|
| Alpaca data generation |—|distill-SFT|teacher;seed;request당 instruction;code default|text-davinci-003;175;20;temp1.0,top_p1.0,ROUGE-L F reject>0.7|blog;README;generate_instruction.py L117-118,L196|verified 2026-09-14(code default;run argument 보고 없음)|ablation 없음|
| WizardLM-13b(LLaMA13B)|13B|SFT(data gen)|evolver;seed;round;total|gpt-3.5-turbo;Alpaca52k;4;250k|arXiv:2304.12244v3 §4.2|verified 2026-09-14|Table2:Llama-2-70B-Chat evolver avg56.27 대58.96|
| WizardLM-13b|13B|SFT|example;optimizer;LR;batch;epoch;max token|250k에서 sample70k;Adam;2e-5;GPU당4,V100 8개;3;2048|v3 §4.2|verified 2026-09-14|Table2 250k variant60.30;App.L Table4 epoch2.5/2.75/3=57.92/58.24/58.96|
| WizardCoder15B(StarCoder15B)|15B|SFT|data;evolver|~78k(seed+3 round);gpt3.5-turbo|arXiv:2306.08568v2 §4.2;v1 §4.5|verified 2026-09-14|Fig.3 pass@1은3 round 뒤 peak|
| WizardCoder15B|15B|SFT|batch(unit unstated);seq length;step;warmup;peak LR;schedule|512;2048;200;30 step;2e-5;cosine,fp16|v2 §4.2|verified 2026-09-14|ablation 없음|
| WizardMath-Mistral-7B|7B|SFT|data;evolver;round|418k(448k unique,contamination30k remove);GPT-4,5 round(2 down,3 up),round당 evolution6,temp0.7;GPT-4-0613 answer|arXiv:2308.09583v3 §3.1,§4.1|verified 2026-09-15|Table6 all round81.2/46.2 대original59.7/15.1|
| WizardMath(Llama2 7B/13B;70B;Mistral-7B)|7B–70B|SFT|epoch;LR;batch;seq length|3;2e-5/1e-5/5e-6;512;2048|v3 §4.1|verified 2026-09-15|ablation 없음|
| Llama-3-8B+Magpie-Pro-300K-Filtered|8B|SFT|example;LR;schedule;warmup;epoch|300K;2e-5;cosine;100 step;2|arXiv:2406.08464v2 §4.1,App.E.2 Table8|verified 2026-09-14|Table12 300K-Filtered25.08 대1M-Raw24.16LC(data only)|
| same|8B|SFT|effective batch;optimizer;max length|32(4×1×8);AdamW β(0.9,0.999),ε1e-8;8192|App.E.2 Table8;§4.1|verified 2026-09-14|ablation 없음|
| Magpie-Air generation(Llama-3-8B-Instruct)|8B generator|SFT(data gen)|instruction decoding;response decoding|temp{1.0,1.1,1.2}×top-p{1.00,0.995,0.990} each300K +1.25×same each100K;greedy|App.E.1 Table7|verified 2026-09-14|higher value difficulty/diversity raise(App.D.3 Fig.11)|
| Llama-3-8B+Magpie-Pro-DPO|8B|preference|pair;sampling;LR;epoch;batch;β|100K;k=5,T=0.8,ArmoRM max/min;5e-7 cosine,10% warmup;1;128;β 보고 없음|§4.1;App.E.2 Table9|verified 2026-09-14(β 보고 없음)|Table1 LC50.10 대SFT25.08|
| Qwen2-7B on Persona Hub math|7B|distill-SFT|example;problem generator;solution generator;LR,epoch,batch|1.07M;GPT-4 zero-shot persona prompt;gpt-4o;보고 없음|arXiv:2406.20094v3 §4.1.2|verified 2026-09-14/not reported(hyperparameter)|Fig.9 instance count 대 accuracy(text number 없음)|
| WRAP1.3B(C4+Q/A rephrase)|1.3B|pretrain|step×batch;seq length;peak/min LR;schedule|300k×1M token(300B token,derived);1024;2e-4/1e-5;cosine,1% warmup|ACL2024 §3.2;Fig.2 caption|verified 2026-09-15|ablation 없음|
| WRAP1.3B|1.3B|pretrain|optimizer;weight decay;clip;real:synthetic;chunk|Adam(0.9,0.999);0.01;1.0;1:1;~300 token|§3.1-§3.2|verified 2026-09-15|Table2 Med+C449.3 대Med only47.9;chunk size empirical observation,no number|
| WRAP rephrase generation|7B rephraser|pretrain(data gen)|model;throughput;85B token compute|Mistral-7B-Instruct;A100당3M token/hour;GPU hour~25K|§3.1,§9.1|verified 2026-09-15|RQ1 Fig.3:Qwen-1.8B comparable,throughput3×|
| Humpback65B(LLaMA65B)|65B|SFT|data;curation;LR;batch|seed3,200+curated~42k(total45k);k=5(score≥4.5인 A5),iteration2;1e-5→9e-6 linear(“most models”);32|arXiv:2308.06259v3 §2.3,§3.1,Table3|verified 2026-09-14|Fig.2:A5,A4 대no curation(7B)|
| WildLlama(Llama-2 7B)|7B|SFT|data;batch;LR;seq length;epoch|2023-07-16까지 WildChat;conversation128;2e-5;2048;3|arXiv:2405.01470v1 §5|verified 2026-09-15|ablation 없음(Vicuna setting reuse)|
| Imitation model(LLaMA7B/13B,ShareGPT-Mix)|7B,13B|distill-SFT|block;epoch;optimizer;LR;warmup;batch|token2048;1;weight magnitude로 gradient rescale AdamW;2e-3;step1000;32|arXiv:2305.15717v1 §4.1|verified 2026-09-15|Fig.4 top: more imitation data가 MMLU/NQ/HumanEval raise 안 함|

2026-09-15 verified row는 [[self-instruct]], [[wizardmath]], [[rephrasing-the-web]], [[wildchat]] library card에 Verification section이 없고 [[false-promise-imitating-proprietary-llms]], [[length-controlled-alpacaeval]], [[lmsys-chat-1m]] card가 아직 없어 해당 날짜 primary PDF에서 읽었다. checked extract는 chapter excerpt에 있다.

**작은 general-purpose run의 시작점.** extraction-based8B instruction SFT의 verified reference는 Magpie run이다. Llama-3-70B-Instruct가 generate한 filtered Magpie-Pro conversation300K,2 epoch,peak LR2e-5 cosine decay/warmup100 step,AdamW(0.9,0.999),effective batch sequence32,max length8192,device4다. ablate한 것은 data quantity/filtering뿐(Table12)이고 optimization setting은 아니다. same run은 official instruct model보다 GSM8K47.92 대71.72,jailbreak refusal0.80 대0.93으로 trail하므로 Generalization lens listed reasoning data/safety check가 필요하다.

## 일반화 관점

**(a) breadth를 늘리는 것.**

- real user prompt: WildChat train model은 Alpaca prompt best model보다 NLL1.04 above인 반면 Alpaca train model은 WildChat best보다5.69 above([[wildchat]] Fig.3).
- surface diversity보다 seed diversity: sample10k에서 seed10k/persona0.1k OOD accuracy54.77 대seed0.1k/persona10k38.15([[prismatic-synthesis]] Table2).
- evolved round와 seed merge: matched size single-round code data51.2-56.1 대seed45.7([[code-evol-instruct]] Table5); math two difficulty direction(+21.5 GSM8K,+31.1 MATH;[[wizardmath]] Table6).
- rephrase 옆 real text keep: general task Med+C449.3 대Med-only47.9([[rephrasing-the-web]] Table2).
- stronger base: more imitation data가 못한 imitation accuracy raise를 larger base가 했다([[false-promise-imitating-proprietary-llms]] Fig.4 bottom);same Tülu mix LLaMA-2 7B avg45.7 대LLaMA7B38.3([[tulu-1-how-far-can-camels-go]] Table4).
- extracted chat data에 targeted reasoning data add: GSM8K47.92→63.08([[magpie]] Table3).

**(b) narrowing/forgetting 원인.**

- broad imitation: NQ7B17→10,13B20→15([[false-promise-imitating-proprietary-llms]] Table1).
- LLaMA13B의 base-GPT-3 Self-Instruct: MMLU−11.9,Codex-Eval−16.1([[tulu-1-how-far-can-camels-go]] Table3).
- single-turn teacher data: Alpaca/GPT4-Alpaca TyDiQA43.2에서31.1/23.5;author는 little multilingual data로 attribute(Tülu1 §5.1,Table3;Interpretation).
- domain evolution: WizardMath-70B AQuA20.0,SAT-Math13.2 대Llama-2-70B40.9,51.3([[mammoth]] Table3-4).
- generator prompt prior: Magpie SFT GSM8K,jailbreak refusal trail([[magpie]] Table3,15).
- seed composition: ShareGPT seed GSM8k37.15→31.46([[evol-instruct]] Table2).
- synthetic-only pretraining text: many sub-domain higher Pile perplexity([[rephrasing-the-web]] Fig.4).

**(c) 이 stage 측정.**

- seed-to-test overlap check가 있는 seed-independent held-out novel-task human evaluation([[self-instruct]] §4.4,App.A.1).
- capability benchmark alongside length-controlled judge score,response length report([[length-controlled-alpacaeval]] §4;[[tulu-1-how-far-can-camels-go]] §5.4).
- base 대비 knowledge,reasoning,multilingual,coding,safety before/after suite([[tulu-1-how-far-can-camels-go]] Table3).
- template cluster removal/per-category share와 held-out WildChat/LMSYS-Chat-1M coverage NLL([[wildchat]] Fig.3;[[lmsys-chat-1m]] Fig.3).
- average만 아닌 per-task result(§7 WRAP example).
- reported benchmark과 separate dev set에서 round/checkpoint selection,generated prompt decontamination([[code-evol-instruct]] §3.2,App.C;[[wizardmath]] §4.1).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 검사 |
|---|---|---|
| classification task에 input-first generation | label 하나가 generated instance dominate | classification task별 label histogram;output-first로 switch([[self-instruct]] §2.2) |
| ROUGE-L filter를 diversity guarantee로 취급 | accepted prompt가 new word로 same skill ask | filtering 후 embedding/task category cluster,WildChat category share compare |
| long response select 뒤 raw AlpacaEval gain report | mean response length와 함께 win rate rise | LC win rate/length report;concise-answer instruction rerun([[length-controlled-alpacaeval]] §4.1) |
| judge win을 capability gain으로 읽음 | judge rise,benchmark flat/lower | base 대비 before/after MMLU,NQ,HumanEval,TyDiQA([[false-promise-imitating-proprietary-llms]] Fig.4) |
| test benchmark에서 evolution round choose | separate dev에서 best round differ | MBPP-400 같은 dev holdout([[code-evol-instruct]] §5) |
| extraction data가 real user cover한다고 assume | real prompt NLL high;math/code/unsafe prompt few | held-out WildChat coverage NLL/category share([[wildchat]] Fig.3,Table4) |
| embedding similarity로만 diversity measure | embedding diversity high,OOD accuracy low | gradient-based measure/held-out accuracy compare([[prismatic-synthesis]] Table2) |
| response-model effect를 prompt method에 credit | other response generator에서 gain disappear | response generator만 swap([[magpie]] Table14) |
| rephrase 시 real text drop | QA up,code/forum/book perplexity worse | real data with/without domain별 Pile perplexity([[rephrasing-the-web]] Fig.4) |
| self-curator를 measure 없이 trust | curated set small,good pair discard | labeled dev precision/recall([[humpback]] Table9) |
| rephrase style-matched task average | task1-2개가 average gain 대부분 | per-task diff/sign count(§7 example) |

## 이해 점검

1. classification grammar-error detection에서 input-first가 무엇을 잘못 만들고 label-first가 왜 fix하는지 설명하라.
2. sea poem/sea salty ROUGE-L filter example이 few skill만 cover할 수 있는 이유와 이를 detect하는 measurement를 설명하라.
3. imitation output70%가 ChatGPT만큼 좋게 rate됐지만 NQ가 fall했다. §9-§10으로 crowdworker/GPT-4 judge가 reward하고 knowledge 없이 SFT를 통해 transfer하는 property를 설명하라.
4. WildChat matrix5.69 대1.04 NLL asymmetry를 prompt distribution 둘로 설명하라.
5. Magpie SFT가 AE2LC는 official instruct보다 높지만 GSM8K는23.8 point trail하는 extraction mechanism과 tested data change를 설명하라.
6. WizardMath combined gain21.5가 separate sum33.7보다 작은 의미와 두 direction 포함이 중요한 이유를 설명하라.
7. WRAP QA8-task average+2.0 중 task2개 contribution1.9가 general-improvement claim에 중요한 이유와 style match/capability gain을 separate할 extra evaluation을 설명하라.
8. persona-conditioned data가 lower pairwise embedding similarity인데 transfer worse한 경우 dataset each variation type을 설명하라.

## 연결

- Previous: ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix(generate stage/filter).
- Next: ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage(stronger model response imitation).
- ch-09 — Pretraining Data Composition and Capability Coverage(WRAP §7); ch-15 — Human Preference and Instruction Data(real prompt distribution §11); ch-21 — Taxonomy-Driven and Textbook-Style Synthesis.
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection(G-Vendi,§6); ch-25 — Multi-Turn Conversation Synthesis(WildChat/LMSYS reference); ch-29 — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification.
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks; ch-31a — Negative Samples in Supervised Training; ch-39 — Offline Preference Optimization: DPO; ch-43a — Negative Samples and Negative Gradients.
- ch-44 — Process Supervision and Verifiable Rewards(WizardMath reward model); ch-47 — Evaluation Harness; ch-48 — Contamination Detection; ch-49 — Judge Models(§10).
- ch-09 — Pretraining Data Composition and Capability Coverage(WRAP을 pretraining data로 다룸,§7).
- ch-15 — Human Preference and Instruction Data: Annotation Protocols, Agreement, and Prompt Coverage(real prompt distribution,§11).
- ch-21 — Taxonomy-Driven and Textbook-Style Synthesis(top-down prompt design과 textbook-style text).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection(G-Vendi 및 generation 뒤 selection,§6).
- ch-25 — Multi-Turn Conversation Synthesis(WildChat,LMSYS-Chat-1M multi-turn reference).
- ch-29 — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification.

## 출처

- [[self-instruct]] — pipeline/output-first/filter/data/audit/SuperNI/novel task/plateau/cost(primary PDF;[[excerpts/self-instruct]] extract).
- [[alpaca]] — pipeline change/released filter/SFT/evaluation/cost.
- [[evol-instruct]] — in-depth/in-breadth/elimination/scale/seed/evolver/difficulty.
- [[code-evol-instruct]] — WizardCoder heuristic/round merge/size/Table5/stop/decontamination/MultiPL-E; [[wizardcoder]]는 same paper model card이며 unverified operator list는 여기 쓰지 않음.
- [[wizardmath]] — down/up evolution/data/Table6/reward ablation(primary;[[excerpts/wizardmath]] extract); [[mammoth]] — WizardMath-70B OOD regression.
- [[magpie]] — template/decoding/filter/SFT-DPO/reasoning/filter-response ablation/safety/cost.
- [[persona-hub]] — persona construction/dedup/math/similarity/limit; [[prismatic-synthesis]] — seed/persona diversity and embedding/gradient diversity.
- [[rephrasing-the-web]] — WRAP style/mix/setting/Table1-2/version conflict/cost(primary;[[excerpts/rephrasing-the-web]] extract).
- [[humpback]] — backtranslation/self-curation/precision-recall; [[tulu-1-how-far-can-camels-go]] —13B per-data regression/unique-token correlation.
- [[false-promise-imitating-proprietary-llms]] — imitation setup,NQ/style/base size(extract); [[length-controlled-alpacaeval]] — LC regression/gameability/Arena/truncation(extract).
- [[wildchat]] — statistics/category/coverage NLL/WildLlama(extract); [[lmsys-chat-1m]] — statistics/topic/caveat/unsafe/SFT subset(extract).
- [[persona-hub]] — persona construction, deduplication, math result, similarity study, stated limits.
- [[prismatic-synthesis]] — seed-diverse 대 persona-diverse set 및 embedding 대 gradient diversity.
- [[rephrasing-the-web]] — WRAP style, mixing, training setting, Table1-2, version conflict, cost(primary; [[excerpts/rephrasing-the-web]] extract).
- [[humpback]] — backtranslation, self-curation, curation precision/recall.
- [[tulu-1-how-far-can-camels-go]] —13B per-dataset regression과 AlpacaEval win rate의 unique response token correlation.
- [[length-controlled-alpacaeval]] — LC regression, gameability, Arena correlation, truncation attack(extract).
