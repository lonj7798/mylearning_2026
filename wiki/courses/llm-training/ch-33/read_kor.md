<!-- chapter: ch-33
     track: sft
     kind: content
     title: Case Studies A: How Tülu 3 and Llama 3 Measured and Protected Broad Capability
     deps: [ch-31a]
     sources: [[tulu-3]], [[tulu-3-sft-mix]], [[tulu-3.1]], [[tulu-3-1]], [[rlvr-tulu3]], [[ifbench]], [[open-instruct-allenai-recipes]], [[open-instruct-allenai-recipes-recipe]], [[llama-3]], [[llama-3-recipe]], [[long-context-llama3]], [[llama-3-synthetic-pipeline]], [[dpo]], [[persona-hub]]
     figures: figures/tulu-llama-recipe.html
     revised: 2026-09 (generality revision)
-->

# 제33장 — 사례 연구 A: Tülu 3와 Llama 3가 broad capability를 측정하고 보호한 방법

> **핵심 통찰.** Tülu 3는 development evaluation suite에서 design decision을 내렸고 development가 끝날 때까지 두 번째 unseen suite score를 보지 않았다(§2.2). SFT, DPO, RLVR 전체에서 8B의 paired development benchmark 5개 평균은 64.9에서 68.8, paired unseen benchmark 5개 평균은 29.9에서 32.4로 올랐다. 하지만 development gain 일부는 transfer하지 않았다. Persona data를 빼면 IFEval은 72.8에서 53.6으로 내려가지만 IFEval-OOD는 17.6에서 18.0으로 움직였다([[tulu-3]], arXiv:2411.15124v5 Table 31--32). Llama 3는 generalist model 하나를 broad하게 유지하는 mechanism을 설명한다. Code/multilingual expert model, capability-specific data pipeline, quality/difficulty/semantic-deduplication filter, RM/SFT/DPO stage의 model averaging, short-only SFT가 long-context ability를 regress시킨 뒤 long-context SFT data 0.1%처럼 target한 share가 있다. 이 mechanism의 ablation number는 보고하지 않는다([[llama-3]], arXiv:2407.21783v3 §4.1.5, §4.2.3, §4.3). 어느 report도 모든 capability를 개선하는 post-training stage를 보이지 않는다. Tülu 3 8B safety average는 SFT 뒤 93.1에서 DPO와 RLVR 뒤 85.5로 내려갔다(Table 9, 23).
>
> **지침.** Recipe가 고정 benchmark set을 상대로 tuning된다면 skill별 disjoint unseen evaluation을 유지하고 모든 data ablation을 그 위에서 재실행한다. Tülu 3 persona data 제거는 IFEval을 19.2점 내렸지만 IFEval-OOD는 내리지 않았기 때문이다(17.6 → 18.0, Table 32). 그렇지 않다면 development gain은 development-suite result로만 보고한다. Base model이 이미 long-context ability를 가졌다면 synthetic long-context example을 SFT에 섞고 long-context evaluation으로 SFT를 gate한다. Short-only SFT가 Llama 3에서 “significant regressions”를 만들었고 example 0.1% share가 short/long benchmark 균형을 맞춘다고 보고됐기 때문이다(§4.3.4; 표 없음). Base에 보호할 long-context ability가 없으면 이 Llama 3 결과는 적용되지 않는다. Verifiable-reward stage가 specific benchmark(Tülu 3: GSM8K, MATH, IFEval)를 target하면 target 옆에 모든 development evaluation average를 추적하고 모든 regression을 보고한다. GSM8K 89.4와 IFEval 84.8에 이른 Tülu 3 8B run은 “other metric에서 더 나빠 overall average를 끌어내리는 경향”이 있었기 때문이다(§6.4). Stage에 target benchmark가 없다면 Tülu 3가 preference mix에서 했듯 development average로 선택한다(§5.2.2).

## 범용 모델에서 이 장이 중요한 이유

두 report는 pre-trained base 위 post-training을 설명한다. SFT, preference optimization, 그리고 Tülu 3에서는 세 번째 RL stage다. Llama 3는 base를 만든 pre-training/long-context stage도 설명한다([[llama-3]] §3). ch-30부터 ch-31a는 SFT design choice를 하나씩 다룬다. 이 장은 complete pipeline 두 개를 따라가며 stage마다 측정 가능한 질문 세 개를 묻는다.

1. 어느 benchmark가 올랐고 어느 benchmark가 내려갔으며, 크기는 얼마인가?
2. Development에 쓴 benchmark gain이 decision에 한 번도 쓰지 않은 benchmark에도 나타났는가?
3. Recipe는 어떤 capability를 실제로 훈련하거나 평가했는가?

Tülu 3는 data, training code, intermediate SFT/DPO checkpoint, hyperparameter table(Table 1, 11, 20, 21)을 공개하므로 첫 두 질문은 표에서 답할 수 있다. Llama 3는 human preference data, code/multilinguality/math/long context/tool use/factuality capability-specific pipeline을 갖는 six round를 실행한다(§4.3). Breadth mechanism은 보이지만 controlled number는 적게 보고한다. Source가 보고한 값은 locus와 함께 인용하고, 어느 source도 보고하지 않은 값은 “미보고”로 표시한다.

Interactive figure [figures/tulu-llama-recipe.html](figures/tulu-llama-recipe.html)는 skill/model size를 바꿔 stage별 development/unseen score를 비교하고, SFT ablation 각각의 두 suite 효과, Tülu 3.1을 포함한 8B development benchmark의 stage별 값, locus가 달린 Llama 3 round structure를 보여 준다.

## §1 Tülu 3: pipeline과 evaluation design

### 1.1 Stage

Tülu 3는 Llama 3.1 8B, 70B, 405B base model 위에 만든 post-trained model family다(arXiv:2411.15124, v1 2024-11; 여기서는 v5 2025-04를 인용). Recipe는 stage 네 개다(§2.3).

1. **Prompt curation.** Public dataset과 persona-conditioned synthetic prompt이며 evaluation suite에 대해 decontaminate했다(§3).
2. **SFT**: prompt-completion example 939,344개로 수행한다(§4, Table 7).
3. **Preference tuning**: length-normalized DPO를 pair 271,409개(8B) 또는 334,302개(70B)에 수행한다(§5, Table 15).
4. **Reinforcement Learning with Verifiable Rewards(RLVR)**: learned reward model 대신 deterministic correctness check가 reward인 PPO를 prompt 29,946개에 수행한다(§6, Table 22).

### 1.2 Development suite와 unseen suite

**정의.** *Development suite*는 data, hyperparameter, checkpoint를 고르는 evaluation 집합이다. *Unseen suite*는 다른 benchmark/prompt format으로 같은 skill을 포괄하며, development 동안 score를 보지 않는다.

**다루는 문제.** Development score를 비교해 내린 모든 choice는 skill이 아니라 specific benchmark에 fit할 수 있다. 두 번째 measurement 없이는 그 효과 크기가 보이지 않는다.

**Mechanism**(§2.2, §7, Table 24):
1. Core skill마다 development benchmark 하나 이상을 둔다. Knowledge는 MMLU/PopQA/TruthfulQA, reasoning은 BBH/DROP, math는 MATH/GSM8K, coding은 HumanEval/HumanEval+, instruction following은 IFEval/AlpacaEval 2, safety는 task 6개 average다.
2. Skill마다 unseen benchmark도 둔다. GPQA/MMLU-Pro, AGIEval English, DeepMind Mathematics, BigCodeBench(Table 33에서 BigCodeBench-Hard라 label한 hard subset task 148개), IFEval-OOD/HREF다. Unseen safety evaluation은 없다(§2.2, §7.3).
3. Unseen task는 few-shot dialog 대신 instruction이 있는 zero-shot prompt처럼 real use에 더 가까운 format을 쓴다(§7.3).
4. 새 unseen benchmark 두 개를 만들었다. IFEval-OOD에는 IFEval constraint 25개와 disjoint인 category 6개의 verifiable constraint 52개가 있다(§7.3.1, §7.4.2). HREF에는 Llama 3.1 405B Instruct 대비 win rate로 score하는 instruction-following task 11개가 있다. Task 9개는 Llama 3.1 70B Instruct가 judge이며, Open QA/Fact Checking은 human-written reference에 대한 embedding similarity를 쓴다. Composite procedure는 human judgment와 69.4% 일치하며 inter-human agreement는 67%다(§7.3.2).

**수치 예시: skill별 gain transfer**(8B, Table 31, SFT checkpoint → final checkpoint. Table 31의 “Avg.” 행은 full suite가 아니라 이 benchmark 5개의 mean이다):

| Skill(development → unseen) | Development 변화 | Unseen 변화 |
|---|---:|---:|
| Math(MATH → DeepMind Math) | 31.5 → 43.7(+12.2) | 32.3 → 35.4(+3.1) |
| Instruction following(IFEval → IFEval-OOD) | 72.8 → 82.4(+9.6) | 17.6 → 24.3(+6.7) |
| Knowledge(MMLU → GPQA) | 65.9 → 68.2(+2.3) | 31.9 → 35.7(+3.8) |
| Reasoning(BBH → AGIEval) | 67.9 → 66.0(−1.9) | 56.2 → 59.3(+3.1) |
| Coding(HumanEval → BigCodeBench) | 86.2 → 83.9(−2.3) | 11.5 → 7.4(−4.1) |

DeepMind Mathematics의 math gain은 3.1점으로 MATH gain 12.2점의 약 4분의 1이다. 저자들은 development process가 “MATH에 어느 정도 overfit했다”고 말하고 format cause를 가설로 든다. Trained model이 DeepMind Mathematics가 요구하지 않는 LaTeX로 reasoning/answer를 써 reasoning을 방해하고 answer extraction을 실패하게 했다는 것이다(§7.4.1; 저자 해석). 저자들은 SFT 이후 stage가 더 어려운 unseen coding evaluation을 개선한다고 말한다. 이는 70B(BigCodeBench 12.2 → 21.6)에서는 맞지만 8B(BigCodeBench 11.5 → 7.4 하락)에서는 맞지 않는다(Table 31; 이 장의 표 읽기).

**Model 전반 benchmark overfitting 근거.** Tülu 3 8B는 IFEval에서 Llama 3.1 8B Instruct보다 높다(82.4 대 80.6, Table 2). IFEval-OOD에서는 낮다(24.3 대 26.1, Table 33). 저자들은 model이 “generally overfit to IFEval”한다고 결론낸다(§7.4.2). Follow-up IFBench는 new constraint 58개에서 같은 pattern을 찾는다. GPT-4.1과 Claude 3.7 Sonnet은 50% 아래 score다([[ifbench]] §1, Fig. 1). 둘은 overlapping author가 있는 Ai2 결과이므로 아직 independent replication은 아니다.

**조건과 한계.** Benchmark pair는 content와 prompt format이 모두 다르므로 gap은 skill transfer와 format sensitivity를 섞는다. Table 31은 checkpoint당 CI 없는 run 하나를 보고한다. 8B SFT MMLU는 Table 6/31에서는 65.9지만 Table 9/10/32에서는 62.1이다. 이 장은 각 table이 출력한 대로 인용한다.

**함의.** Table 32에서 final 8B SFT mix는 SFT model 5개 중 development average 최고(64.1)와 unseen average 최고(29.9)를 모두 가진다. 두 average 모두 data choice를 지지한다. 어떤 choice가 transfer하지 않았는지는 skill별 unseen column만 보인다. Persona data 없이 IFEval-OOD는 18.0, WildChat 없이 20.8이며 full mix의 17.6보다 높다. 8B unseen split은 SFT 이후 stage가 BigCodeBench를 낮춘 것도 보이지만, Table 31의 저자 summary는 이를 말하지 않는다.

## §2 Tülu 3 SFT: multi-skill mixture 만들기와 시험하기

### 2.1 Mixture 구성

Final mix는 source 19개에서 example 939,344개이며 count는 [excerpts/tulu-3-sft-mix.md](excerpts/tulu-3-sft-mix.md)에 열거된다(Table 7). Ai2 blog는 prompt 57%가 public resource, 43%가 in-house synthesis라고 한다([[tulu-3-sft-mix]]; allenai.org/blog/tulu-3-technical). Library card [[tulu-3-sft-mix]]는 component 18개를 열거한다. Table 7은 OpenMathInstruct 2(example 50,000개)도 포함하며, 이것 없이는 합계가 939,344가 되지 않는다.

구성 procedure(§4.1.2, Fig. 3)는 다음과 같다.
1. Llama 3.1 8B를 Tülu 2 mix에서 훈련하고 state-of-the-art model보다 뒤지는 skill을 열거한다.
2. 뒤지는 skill마다 skill-specific mixture/model을 만들고, 그 skill 하나에서 최고 score인 mixture를 유지한다. Skill의 upper bound를 근사한다.
3. Skill mixture를 preview mix로 합친 다음 뒤지는 skill을 올리도록 dataset을 추가/제거하고, decontaminate하며, 큰 source를 downsample한다. Intermediate mix 1--3은 dataset을 더했으며 mix 4--5는 small drop을 낸 decontamination round였다(Fig. 3 caption).

Synthetic prompt는 Persona Hub persona 약 250K개에 condition한다(§3.1.2; method는 [[persona-hub]]). Math/coding problem과 math solution은 GPT-4o가, Python solution은 claude-3-5-sonnet이 생성했다. Verifiable-instruction example 29,980개는 IFEval constraint type 25개를 포괄한다(§3.1.2). 마지막 사실은 §1.2에 중요하다. Persona IF data가 IFEval 자체 taxonomy에서 만들어졌기 때문이다.

### 2.2 Decontamination

**정의.** Decontamination은 training example과 evaluation example의 overlap을 제거해 benchmark score가 memorized item을 재지 않게 한다.

**Mechanism**(§3.2):
1. Completion은 model이 자주 regenerate하므로 prompt(user turn)만 비교한다.
2. Test token을 포함하는 8-gram을 test/training instance가 모두 가지면, 그 token은 training instance와 *match*한다.
3. Test instance token의 50% 초과가 같은 training instance와 match하면 significant overlap이다.
4. Development 또는 unseen evaluation 하나의 instance 2% 초과와 overlap하면 training dataset은 contaminated다.
5. Unseen evaluation에 contaminated된 dataset은 전부 제거한다. Development evaluation에는 model을 유의하게 해치지 않으면 dataset 전부를, 아니면 matching instance만 제거한다.

Full-string/embedding-based matching도 시험했다. Embedding matching은 “mere distributional similarity와 actual paraphrasing을 구분하기 어렵다”는 이유로 버렸다. 반면 8-gram matching은 number만 바꾼 math problem 같은 near-duplicate를 flag했다(§3.2).

**수치 예시.** Token 20개의 test prompt가 training prompt 하나와 token 12개에 걸쳐 8-gram을 공유하면 `12/20 = 60% > 50%`이므로 pair가 overlap한다. Instance 500개의 evaluation에서는 10개 초과(`500의 2%`)가 overlap하면 dataset은 contaminated다.

**근거.** Removal rate는 Evol CodeAlpaca(HumanEval) 3.5%, WildChat GPT-4(safety) 5.4%, NuminaMath-TIR(MATH) 11.3%다(Table 8). Cleaning 전 Evol CodeAlpaca는 HumanEval 70.7%, LMSys Chat 1M은 Do-Anything-Now 90.3%와 overlap했다. ShareGPT/LMSys Chat/DaringAnteater/OpenAssistant 2는 사용하지 않았다(Table 37). 저자들은 real API usage dataset이 existing test set과 overlap할 가능성이 높다고 언급한다(App. B.2).

### 2.3 두 suite의 ablation

Table 10은 8B SFT mix에서 data group 하나씩을 빼며, Table 32는 같은 model을 unseen suite에서 다시 평가한다. Figure의 두 번째 panel은 각 change를 plot한다.

| 제거한 group | 핵심 development 변화 | Paired benchmark의 unseen 변화 |
|---|---|---|
| Persona data(IF, math, code) | IFEval 72.8 → 53.6 | IFEval-OOD 17.6 → 18.0 |
| Math data | MATH 31.5 → 23.5; GSM8K 76.2 → 64.1 | DeepMind Math 32.3 → 23.3 |
| WildChat(real-user chat) | AlpacaEval 2 12.4 → 7.5; IFEval 72.8 → 70.1 | BigCodeBench 11.5 → 7.4; IFEval-OOD 17.6 → 20.8 |
| Safety data | Safety average 93.1 → 74.7 | unseen safety evaluation 없음 |

Outcome 세 종류가 있다. Math data는 transfer했다. 없으면 MATH/DeepMind Mathematics가 둘 다 8--9점 하락한다. Persona IF data는 transfer하지 않았다. IFEval 19.2점 효과에는 IFEval-OOD counterpart가 없다. 저자들은 SFT choice가 precise instruction following development evaluation에, knowledge recall/reasoning에는 “어느 정도” overfit했다고 말한다(§7.4.1). Safety data 제거는 다른 모든 development score를 1.9점 안에 남겼고, 저자들은 safety가 “orthogonal”하다고 설명한다(§4.2). CoCoNot 같은 contrastive prompt가 safe prompt의 over-refusal을 막는다고 보고했지만 수치는 없다(§4.2).

**Data 양.** Mix의 stratified subsample 5%--100%는 average/GSM8K를 높이지만 data가 늘수록 TruthfulQA는 내려간다(§4.2, Fig. 4; text 수치 없음). 8B SFT checkpoint TruthfulQA는 46.8로 Tülu 2 8B SFT 49.4보다 낮다(Table 9). Preference tuning에 다른 prompt를 남겨 두기 위해 SFT stage는 이 크기로 cap했다(§4.2).

**조건.** Ablation마다 8B run 하나다. §2.4는 effect를 seed-to-seed range와 비교한다.

### 2.4 Result를 바꾸는 training setting

**Loss aggregation.** Gradient accumulation/data parallelism에서 token별 per-batch mean은 batch 분할에 따라 example weight를 다르게 한다(§4.3.2).

$$L_{\text{batch}} = \frac{l_{n_1} + l_{n_2}}{n_1 + n_2} \qquad L_{\text{accum}} = \frac{1}{2}\left(\frac{l_{n_1}}{n_1} + \frac{l_{n_2}}{n_2}\right)$$

`n_i`는 sample `i`의 non-padding token 수, `l_{n_i}`는 sample `i`의 summed token loss다. 수치 예시는 `n_1 = 10`, `l_{n_1} = 20`; `n_2 = 90`, `l_{n_2} = 90`이다. 그러면 `L_batch = 110/100 = 1.10`, `L_accum = (2.0 + 1.0)/2 = 1.50`이다. `L_batch` 아래 모든 token weight는 1/100이다. `L_accum` 아래 sample 1의 각 token은 `1/(2 × 10) = 1/20`, sample 2의 각 token은 `1/(2 × 90) = 1/180`이다. Short sample token 하나가 long sample token 하나보다 9배 count된다. Tülu 3는 모든 token을 동등하게 weight하는 **sum loss**(denominator 없음)를 쓰며 learning rate를 다시 tune해야 한다. Llama 3.0/Tülu 2 mix sweep에서는 LR `5e-6`의 sum loss가 최고였다(§4.3.2, Fig. 5). 2025-06 recipe open-instruct command는 `--reduce_loss sum`을 넘겼지만 pinned commit `098424c` 기본값은 mean loss다([[open-instruct-allenai-recipes]], tulu3.md@8781471 L49; PR #1024).

**Seed.** 8B SFT seed 5개 average는 59.8--60.1(range 0.3), 70B seed 3개는 70.0--72.6(range 2.6)다. Best two-seed soup는 8B 60.2, 70B 72.5여서 best single run을 release했다(Table 14). 8B seed range 0.3은 Table 10 average change 모두(1.2--2.1점)보다 작다. 그 size에서는 average effect가 seed noise보다 크다. 70B seed range 2.6은 그 크기 effect를 넘을 수 있으며 70B ablation은 보고하지 않는다.

**Base model.** 같은 mix에서 GSM8K/MATH는 Llama 3.1 8B 76.2/31.5, Llama 3.1 70B 91.1/53.7, Qwen 2.5 7B 79.2/49.4, Qwen 2.5 Math 7B 86.3/56.4다(Table 12). Base를 Llama 3.1 8B에서 Qwen 2.5 7B로 바꾸면 MATH가 17.9점 바뀐다. Math SFT data 전체 제거의 8.0점(Table 10)보다 크다.

**Chat template.** Llama 3.0와 intermediate mix에서 훈련한 template 5개는 average 51.6--53.0이다. Llama 3 template가 최저였고, Tülu 3는 later stage와 generation consistency를 유지하려고 최고 eos variant 53.0 대신 “no `\n`” template 52.8을 쓴다(Table 13, §4.3.1). SFT setting은 Recipe 절에 있다.

## §3 Tülu 3 preference tuning

**Data pipeline**(§5.2.1, Fig. 7):
1. SFT에 쓴 prompt, 같은 source에서 SFT에 쓰지 않고 남긴 prompt, 새 prompt(TruthfulQA를 뺀 UltraFeedback, IF-augmented prompt)를 선택한다.
2. Prompt당 response 4개를 model 22개 pool에서 sample한다. Prompt 일부에서는 response 하나가 Tülu 3 SFT model에서 온다(on-policy).
3. GPT-4o-2024-08-06이 response마다 helpfulness, instruction following, honesty, truthfulness에서 1--5점을 매긴다.
4. Mean rating 최고 response가 *chosen*, 더 낮은 mean response 하나를 무작위 sample해 *rejected*로 쓴다.

**Finding**(§5.3; 명시하지 않은 것은 8B의 결과, 단일 연구). Unique prompt가 많을수록 DPO가 개선됐다(Fig. 8). UltraFeedback prompt를 pair 383K로 duplicate하면 pair 64K와 비슷했고 DROP/GSM8K/AlpacaEval은 약간 낮췄다(Fig. 9). Unused prompt가 reused SFT prompt보다 좋고, 둘을 섞는 것이 최고였다(Fig. 10). On-policy response 추가는 average를 개선했다(Fig. 11). GPT-4o, Llama 3.1 405B, GPT-4 Turbo judge average는 57.3, 57.2, 57.0이다(Table 17). Persona math/code preference data는 target을 개선하지 않아 persona IF preference만 유지했다(§5.3).

**Objective.** Length-normalized DPO(DPO-norm, Eq. 6)다.

$$\max_{\pi_\theta} \mathbb{E}_{(y_c, y_r)\sim D}\left[\log \sigma\left(\frac{\beta}{|y_c|}\log\frac{\pi_\theta(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \frac{\beta}{|y_r|}\log\frac{\pi_\theta(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\right)\right]$$

`x`는 prompt, `y_c`/`y_r`는 chosen/rejected response, `|y|`는 token 단위 response length, `π_θ`는 policy, `π_ref`는 SFT reference, `σ`는 logistic function, `β`는 scale이다. `β = 5`, `|y_c| = 100`, summed chosen log-ratio `+2.0`; `|y_r| = 50`, summed rejected log-ratio `−1.0`인 예시를 보자. Margin은 `5/100 × 2.0 − 5/50 × (−1.0) = 0.10 + 0.10 = 0.20`, loss는 `−log σ(0.20) = 0.598`이다. 같은 수에서 `β = 0.1` standard DPO([[dpo]])는 margin `0.1 × (2.0 − (−1.0)) = 0.30`, loss `0.554`다. `|y|`로 나누면 summed log-ratio가 token당 average가 되므로, response가 token만 더 많다고 margin을 얻지 못한다. 저자들은 normalization이 preference의 length bias를 “직관적으로 완화하는 데 도움”이 된다고 말한다(§5.1.2; 해석).

**Algorithm 비교.** Early SFT checkpoint(average 55.7)의 UltraFeedback에서 DPO-norm은 57.3, standard DPO는 55.2, SimPO는 최고 52.9, PPO는 최고 55.5에 도달했다(Table 18). Later controlled comparison에서 PPO는 “comparable … (albeit slightly lower)”했고, node 2개에서 약 28시간이 걸렸다. DPO는 node 1개에서 약 4시간이다(§5.4.1).

**Stage 효과**(8B, SFT → DPO, Tülu 3.1 model card table. SFT 값은 Table 9/31, DPO 값은 Table 23과 일치): AlpacaEval 2 12.4 → 33.5, IFEval 72.8 → 81.1, MATH 31.5 → 42.0, TruthfulQA 46.8 → 56.1; HumanEval 86.2 → 83.9, BBH 67.9 → 65.8, safety 93.1 → 87.2.

## §4 Tülu 3 RLVR와 Tülu 3.1 update

### 4.1 Objective와 data

RLVR은 다음을 최적화한다(§6, Eq. 7--8).

$$\max_{\pi_\theta}\ \mathbb{E}_{y\sim\pi_\theta(x)}\big[v(x,y) - \beta\,\mathrm{KL}[\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)]\big], \qquad v(x,y) = \begin{cases}\alpha & \text{if correct}\\ 0 & \text{otherwise}\end{cases}$$

`v`는 verification function, `α = 10`이며 “pilot experiment에 기반했고 더 tune하지 않았다.” `β`는 KL coefficient, `π_ref`는 DPO starting checkpoint다. End-of-sequence token 없는 response는 −10.0을 받는다(Table 21).

**수치 예시.** `β = 0.05`이고 KL term을 reference 대비 response summed log-ratio로 estimate한다고 하자. Summed log-ratio 12 nat인 correct response는 `10 − 0.05 × 12 = 9.4`를 받는다. 같은 log-ratio인 incorrect response는 `0 − 0.6 = −0.6`이다. EOS token 없는 response는 reward value −10.0(Table 21)이므로 같은 log-ratio에서 total `−10 − 0.6 = −10.6`이다. PPO는 GAE/value model로 reward를 advantage로 바꾸고 advantage를 whiten한다(§6.2, Table 21). Negative advantage response는 update에서 likelihood가 낮아진다.

**Data**(Table 22): GSM8K train 7,473개(extracted answer exact match), MATH train 7,500개(“flex” answer extraction), constraint-specific verifier가 있는 IF prompt 14,973개, total 29,946개다. Code prompt는 없다. Code execution feedback은 related work로만 언급한다(§6, footnote 17). Value model이 general reward model에서 initialize됨, dropout off, advantage whitening, epoch마다 prompt reshuffle 같은 implementation detail은 [excerpts/tulu-3-rlvr.md](excerpts/tulu-3-rlvr.md)에 인용한다. Library card [[rlvr-tulu3]]는 binary `{0,1}` reward, unit test code verifier, 다른 PPO setting을 설명하지만 paper에는 어느 것도 없다.

### 4.2 Regression을 포함해 RLVR이 바꾼 것

8B DPO → RLVR(Table 23): GSM8K 84.3 → 87.6(+3.3), MATH 42.0 → 43.7(+1.7), IFEval 81.1 → 82.4(+1.3), AlpacaEval 2 33.5 → 34.5, BBH 65.8 → 66.0이다. TruthfulQA는 56.1 → 55.0(−1.1), MMLU는 68.7 → 68.2, safety는 87.2 → 85.5(−1.7)다. 같은 harness의 Llama 3.1 8B Instruct는 GSM8K 83.4, MATH 42.5, IFEval 80.6이다. 70B에서는 GSM8K가 93.5에 머물고 IFEval은 82.6 → 83.2다(Table 23).

Stage 전반 가장 큰 safety decline은 jailbreak task에 있다. Do-Anything-Now 88.3(SFT) → 69.7(DPO) → 62.0(final), JailbreakTrigger 95.8 → 87.0 → 85.5, WildJailbreakTest 86.7 → 81.1 → 78.8이다. HarmBench는 98.4 → 94.4 → 94.7, WildGuardTest는 99.2 → 98.9 → 98.5다. Benign prompt compliance도 score하는 XSTest는 90.4 → 92.4 → 93.3으로 오른다(Table 25).

**Checkpoint selection.** Checkpoint는 100 step마다(70B는 40마다) 평가했고, 8B는 MATH/IFEval 최고 model을 release했다. GSM8K 89.4%와 IFEval 84.8% run은 “other metric에서 더 나빠지는 경향”이 있어 선택하지 않았다(§6.4). 모든 RL model은 final보다 이른 checkpoint를 택했다(§6.3).

### 4.3 Over-optimization과 IFBench follow-up

더 낮은 `β`는 policy가 reference에서 더 멀리 움직이게 한다. 더 많은 KL은 “typically lower average score”를 낸다(§6.2.1, Fig. 21). Constraint prompt에서 `β = 0.01` model은 “letter e should appear 14 times”로 pen을 재라는 요청에 measurement 없이 comma-separated `e` 14개를 답했다(App. B.4, Fig. 28). `β = 0.1` model은 question에 답했다(Fig. 29).

[[ifbench]]는 Tülu-3-8B-DPO에서 GRPO로 IF-RLVR을 훈련했다. Prompt는 IFTrain의 new training constraint type 29개와 IFEval type 25개에서 뽑은 constraint 1개부터 `n`개다. IFEval은 81.1 → 92.2, IFTrain과 disjoint constraint 58개를 쓰는 IFBench는 25.2 → 44.6이 됐다. 하지만 AlpacaEval 2는 33.5 → 21.3, MMLU는 68.7 → 66.4로 내렸다(Table 3, 6). Constraint 없이 completion이 prompt에 얼마나 잘 답하는지를 GPT-4.1 judge에 물으면, IF-RLVR 전 policy 평균은 10점 중 7, IF-RLVR policy는 6.4다(§1, §5). Constraint reward 훈련은 unseen-constraint score를 올리고 general response quality를 내렸다(**결과, 단일 연구**). Paper §1은 같은 run의 다른 IFBench 값 28.9 → 45.9를 출력하며, 이 장은 Table 6을 쓴다.

### 4.4 Tülu 3.1

Llama-3.1-Tulu-3.1-8B model card는 update가 “training final RL stage만 개선한 것”이라고 한다. PPO를 GRPO로 바꿨다(“no reward model”; GRPO는 value model 대신 group baseline도 쓴다). “Further hyperparameter tuning”도 했다([[tulu-3.1]]). Tülu 3 8B 대비 card는 average 64.8 → 66.3, MATH 43.7 → 47.8, GSM8K 87.6 → 90.0, IFEval 82.4 → 83.9, TruthfulQA 55.0 → 59.9, safety 85.5 → 81.2를 보고한다.

이는 controlled PPO-versus-GRPO comparison이 아니다. Run 둘 사이 algorithm, value model, `β(0.05 → 0.01)`, learning rate(`3e-7` linear → `5e-7` constant), prompt당 sample(GRPO 16개, iteration당 prompt 48개), PPO update iteration `K(4 → 1)`, mini-batch(1 → 2), EOS penalty(`−10.0 → 0.0`), checkpoint episode가 모두 바뀌었다. RL dataset name `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`와 response length 2,048 token은 같다([[open-instruct-allenai-recipes-recipe]]; [excerpts/tulu-3.1-delta.md](excerpts/tulu-3.1-delta.md)). Library card [[tulu-3-1]]은 “Tülu 3.1”을 Llama 3.1/OLMo 2 base refresh로 설명하지만 model card/Tülu 3 technical blog는 이를 지지하지 않으므로 이 장은 그 설명을 사용하지 않는다.

## §5 Llama 3: six-round loop

### 5.1 Round 하나

Llama 3 report(arXiv:2407.21783, v1 2024-07, Llama 3.1 결과)는 같은 step을 round 6개에 걸쳐 반복하는 post-training을 설명한다(§4.1, §4.1.6, Figure 7; detail은 [[llama-3-recipe]]).

1. **Preference data 수집.** Annotation prompt마다 previous round 뒤 deploy한 서로 다른 model 두 개에서 response 두 개를 sample한다. Annotator는 네 level(“significantly better”, “better”, “slightly better”, “marginally better”)에서 preference를 고르고 chosen response를 edit할 수 있다(§4.2.1). Model이 좋아질수록 prompt complexity가 높아진다.
2. **Reward model 훈련.** 지금까지 모은 모든 preference data에서, margin term을 뺀 Llama 2 objective로 훈련한다. “significantly better” 또는 “better” label의 pair만 남긴다(§4.1.2, §4.2.1).
3. **Rejection sampling.** Human-annotation prompt마다 최신 policy에서 output `K`개(보통 10--30)를 sample한다. 보통 previous round의 최고 checkpoint나 capability별 최고 checkpoint에서 sample하고, reward model이 가장 높게 score한 output을 유지한다(§4.2.2).
4. **SFT.** Pre-trained model을 rejection-sampled/synthetic/human-curated data에서 target token에만 loss를 두고 SFT한다. 가장 큰 model은 LR `10⁻⁵`에서 8.5K--9K step을 쓴다(§4.1.3).
5. **DPO.** Pair가 current policy distribution에 가깝게 하려고 most recent preference batch에서 DPO한다(“we primarily use the most recent batches of preference data”)(§4.1.4).
6. **Model averaging.** RM/SFT/DPO stage에서 서로 다른 data/hyperparameter run을 average한다(§4.1.5).

Final SFT mix는 example share로 general English 52.66%, code 14.89%, multilingual 3.01%, exam-like 8.14%, reasoning/tools 21.19%, long context 0.11%다. Round마다 “wide range benchmark 전반 performance를 tune”하기 위해 조정한다. Final mix는 high-quality source 일부에는 multiple epoch를 적용하고 다른 source는 downsample한다(Table 7, §4.2.2). Round별 data share와 round별 evaluation result는 보고하지 않는다. 2024-04 Meta blog의 Llama 3 8B/70B post-training 설명은 SFT, rejection sampling, PPO, DPO였다([[llama-3-synthetic-pipeline]], “Instruction fine-tuning”). 2024-07 report는 PPO를 explore했고 DPO가 “less compute … and performed better, especially on instruction following benchmark like IFEval”라고 하지만 수치는 없다(§4.1.4).

### 5.2 DPO modification

Llama 3는 LR `10⁻⁵`, `β = 0.1`의 standard DPO에 change 두 개를 더한다(§4.1.4; 인용은 [excerpts/llama-3-dpo-nll.md](excerpts/llama-3-dpo-nll.md)).
1. **Formatting token masking.** 두 response에서 header/termination token을 loss에서 제거한다. Masking 없이는 tail repetition 또는 abrupt termination token이 관측됐으며, 저자들은 두 response가 공유하는 token이 opposing update를 받는다고 가설을 둔다.
2. **Chosen response의 NLL term.** Coefficient 0.2를 더한다. `$L = L_{\text{DPO}} + 0.2 \cdot \mathrm{NLL}(y_c \mid x)$`이며 “Pang et al. (2024)와 유사”하다. Formatting을 유지하고 chosen log-probability가 내려가는 것을 막는다. NLL을 token별 normalize하는지는 보고하지 않는다.

## §6 Llama 3: generalist 하나를 broad하게 유지하는 mechanism

### 6.1 Capability expert와 pipeline

Llama 3는 별도 *expert* model을 훈련하고, 이를 model 하나의 released version에 넣을 data를 만드는 데 쓴다.
- **Code expert**는 code 85% 초과 token 1T에서 continued pre-training하고 16K token까지 long-context fine-tuning한다. Human code annotation 수집에 쓰고 coding prompt의 rejection-sampling policy다(§4.3.1).
- **Multilingual expert**는 multilingual token 90%에서 continued pre-training하며, main model pre-training이 끝날 때까지 non-English annotation 수집에 쓴다(§4.3.2).

Capability pipeline은 correctness check를 가진 data를 더한다. Execution-feedback code dialog 약 1M개(약 20% solution이 처음에는 틀렸고 parser/linter/unit-test feedback 뒤 self-correct), backtranslation dialog 약 1.2M개, code의 less common language 번역이다(§4.3.1). Pipeline-generated data에는 보고한 failure 두 개가 있다. Code pipeline에서 405B를 “자체 생성 data”로 훈련하는 것은 도움이 안 되며 “성능을 저하시킬 수도 있다.” 8B/70B는 더 큰 model data에서 개선됐고, 이 때문에 execution feedback을 도입했다(§4.3.1). Strict model-as-judge code filter는 “challenging prompt example을 불균형하게 제거”해 처음에는 benchmark score를 낮췄다. 가장 challenging response를 judge를 통과할 때까지 revise하면 downstream performance가 최고였지만 수치는 없다(§4.3.1). Tool-use data는 rejection sampling이 gain이 없어 이를 건너뛰었고, model이 tool을 불필요하게 호출하지 않도록 tool 없이 답하는 easy query를 추가했다(§4.3.5).

### 6.2 Quality, difficulty, semantic deduplication filter

대부분 model-generated SFT data를 model-based technique 4개로 prune한다(§4.2.3).
1. Fine-tuned Llama 3 8B classifier의 **topic classification**(coarse/fine bucket).
2. **Quality**: reward-model score top quartile에 들거나 maximum Llama 3 rating을 받으면 example을 유지한다. 두 signal은 자주 disagree하며, union이 internal test set에서 recall 최고였다.
3. **Difficulty**: Llama 3 70B의 Instag intention tag(intention이 많을수록 complex)와 Llama 3 three-point difficulty rating.
4. **Semantic deduplication**: RoBERTa로 dialog를 cluster하고, cluster를 quality × difficulty로 sort하며, 지금까지 본 example과의 maximum cosine similarity가 threshold 미만일 때만 greedily example을 유지한다. Threshold 값은 미보고다.

**수치 예시**(illustration으로 threshold 0.9). Quality × difficulty가 A(2.7), B(2.0), C(1.8), D(1.2)인 cluster를 생각한다. Similarity는 A--B 0.95, A--C 0.60, B--C 0.70, A--D 0.50, B--D 0.40, C--D 0.92다. A를 유지한다. B의 maximum similarity 0.95 ≥ 0.9이므로 drop한다. C의 maximum 0.70은 유지한다. D의 maximum(C와 0.92)은 drop한다. Cluster는 A/C를 유지한다. Near-duplicate pair A--B, C--D는 high-score member를 유지하므로 sort key가 어느 duplicate가 살아남는지를 결정한다.

Rule-based cleaning은 excessive emoji 같은 pattern도 제거하고 overused apologetic phrase의 share를 balance한다(§4.2.3). 어떤 filter도 ablation number는 보고하지 않는다.

### 6.3 Model size별 safety data balance

Adversarial prompt와 닮았지만 helpful answer가 마땅한 borderline prompt를 더해 false refusal rate를 낮춘다. Figure 18은 comparable safety를 위해 8B가 70B보다 helpfulness data에 상대적으로 더 높은 safety data 비율을 필요로 함을 보이며, 저자들은 size별 mix를 tailor했다(§5.4.3). Safety DPO에서는 embedding space에서 “nearly orthogonal”한 response pair가 특히 효과적이라고 보고한다(§5.4.3). Llama Guard 3는 system-level filtering을 위한 별도 classifier며 model training 일부가 아니다(§5.4.7).

### 6.4 Llama 3가 breadth/contamination을 잰 방법

- **Decontamination.** Post-training data는 exact prompt match로 decontaminate했다(§5.2). Pre-training에는 8-gram analysis로 contamination/effect를 추정한다. HellaSwag는 85% flagged, 8B에서 추정 +14.8; NaturalQuestions는 52% flagged, 8B/70B/405B에서 추정 +1.6/+0.9/+0.8이다. HumanEval/MBPP/MMLU/MMLU-Pro는 추정할 수 없다(Table 15).
- **Held-out human evaluation.** “Modeling team은 human-evaluation prompt에 access하지 못했다”(§5.3). Report conclusion은 pre-training data를 contamination 예방 incentivized separate team이 처리했다고 더한다(§10). 2024-04 blog는 modeling team에서 withheld한 use case 12개, prompt 1,800개 set을 설명한다([[llama-3-synthetic-pipeline]]).
- **Robustness/adversarial pair.** Pre-trained model은 MMLU label variant, few-shot label distribution, answer-order permutation, prompt format에서 test됐고 저자들은 특히 405B가 이 change에 “very robust”하다고 한다(§5.1.2, Fig. 13--14; post-trained robustness test 미보고). Adversarial question answering/GSM-Plus에서는 pre-trained/post-trained 모두 paired standard benchmark보다 “substantially lower” score며, PAWS에는 그런 gap이 없다(§5.1.3, Fig. 15).

## §7 두 recipe의 long-context post-training

**Llama 3 base.** Context는 약 token 800B에 걸친 stage 6개에서 8K에서 128K로 자란다. Short-context evaluation이 “recovered completely”하고 해당 length에서 needle-in-a-haystack을 풀 때만 stage를 진행한다(§3.4.2). RoPE `θ`는 모든 pre-training의 architecture setting으로 500,000까지 올린다(§3.2, Table 3). Long-context stage마다 rescale하지 않는다. Stage별 length/token count는 미보고다. Library card [[long-context-llama3]]는 stage별 token budget과 stage에 따라 “10K에서 500K로 rescaled”한 RoPE base를 열거하지만 report에는 둘 다 없다.

**Llama 3 SFT**(§4.3.4):
1. Short-context-only SFT는 pre-training의 long-context capability에 “significant regressions”를 만들었다.
2. Earlier Llama 3 model의 synthetic data는 use 세 가지를 포괄한다. Long pre-training document random 8K-token chunk에서 생성한 QA pair(전체 document가 context), 8K chunk hierarchical summarization 뒤 global understanding이 필요한 QA, Python file을 import하는 다른 file이 최소 5개일 때 그 file을 제거하고 dependent를 이름 붙이며 file을 regenerate하게 하는 repository code reasoning이다.
3. Example은 token 16K/32K/64K/128K에서 bucket한다.
4. Short data에 이 data 0.1%를 섞으면 저자 ablation에서 “short-context와 long-context benchmark 양쪽의 performance를 optimize”한다. Ablation은 출력하지 않는다.

**수치 예시: example share 대 token share.** Long-context example은 SFT example의 0.11%다. Long-context example 평균은 token 38,135.6이며 final response token은 740.5다. 모든 example 평균은 token 846.1, final response 310.4다(Table 7). Processed token 전체에서 share는 `0.0011 × 38,135.6 / 846.1 ≈ 4.96%`(계산값)이며, example share의 약 45배다. Loss가 final response만 덮는다면 loss token share는 `0.0011 × 740.5 / 310.4 ≈ 0.26%`(계산값)다. Report는 prompt token을 mask한다(§4.1.3). 그러나 context 안 earlier assistant turn도 훈련하는지 말하지 않으므로 두 번째 수치는 이 가정 아래에서만 성립한다. 둘 다 Table 7의 rounded 0.11% share를 쓴다. 첫 수치는 processed context 중 long한 양, 둘째는 loss가 long-context target에서 오는 양이다.

**Llama 3 DPO.** Short-context-only DPO data는 “SFT model이 long context task에서 high quality라면 long-context performance에 negative impact를 주지 않았다.” 저자들은 DPO가 SFT보다 optimizer step을 적게 쓰기 때문이라고 의심한다(§4.3.4; 해석). Post-trained result에는 8B/70B/405B의 multi-needle retrieval 98.8/97.5/98.1, InfiniteBench En.QA 27.1/36.7/30.5가 있다(Table 21). Method 및 claimed/effective context length gap은 ch-32b/ch-32c가 다룬다.

**Tülu 3.** SFT example은 최대 token 4,096, DPO example은 2,048이다(Table 11, 20). Mix 평균은 turn 2.4이며 대부분 sample이 token 2,048 아래다(§8.3). Evaluation suite에는 long-context/multi-turn benchmark가 없고(Table 24), 저자들은 둘 다 future work로 든다(§8.3). Tülu 3가 fine-tune하는 Llama 3.1 8B/70B base는 Llama 3 Table 1에서 long-context model로 표시된다. 그러므로 Tülu 3는 Llama 3가 보고한 regression을 발견할 gate 없이 long-context base에서 short-context SFT를 실행했다. Tülu 3가 long-context ability를 잃었는지는 어느 source도 보고하지 않는다.

## §8 각 recipe가 gate한 것과 측정하지 않은 것

| Capability | Tülu 3 trained / development eval / unseen eval | Llama 3 trained / reported eval |
|---|---|---|
| Knowledge, reasoning, math, code | 예 / 예 / 예 | 예 / 예 |
| Precise instruction following | 예 / IFEval / IFEval-OOD | 예 / IFEval |
| Safety와 over-refusal | 예 / XSTest 포함 six-task average / 아니오 | 예 / size/language별 violation/false refusal rate |
| Multilingual | Aya data만 / 아니오 / 아니오 | expert + data / MGSM, multilingual MMLU |
| Long context | 아니오 / 아니오 / 아니오 | SFT 0.11% / NIH, ZeroSCROLLS, InfiniteBench |
| Tool use | 아니오 / 아니오 / 아니오 | 예 / Nexus, API-Bank, API-Bench, BFCL |
| Multi-turn | 평균 2.4 turn / 아니오 / 아니오 | SFT 평균 4.7 turn / multi-turn human evaluation |

Source: [[tulu-3]] Table 24, §8.3; [[llama-3]] Table 7, §5.2--5.4.

**Gate.** Tülu 3는 development average + target skill로 SFT mix/preference mix/RL checkpoint를 골랐고(§4.1.2, §5.2.2, §6.4), unseen suite는 나중에만 썼다. Llama 3는 long-context pre-training stage를 short-context recovery와 NIH로 gate했고(§3.4.2), round마다 SFT mix를 “wide range of benchmark”에 맞춰 tune했으며(§4.2.2), safety mix를 violation/false refusal rate에서 tune했다(§5.4.3).

**미측정 또는 미보고.** Tülu 3에는 unseen safety evaluation, long-context/tool-use/multilingual evaluation이 없고 ablation은 seed 하나다. Llama 3는 benchmark decision의 development/unseen split, round별 result, long-context share ablation/PPO comparison 수치, filter ablation을 보고하지 않는다. Tülu 3는 rejection sampling을 시험해 compute 대비 “minimal” gain을 찾았고(§8.2), Llama 3는 매 round에 쓴다. 어느 report도 matched compute에서 둘을 비교하지 않는다.

## 부정 sample과 부정 feedback

이 절은 ch-31a의 “negative” 네 의미를 두 pipeline에 적용한다.

**Negative의 출처와 처리.**

| Negative signal | Label source | 사용한 의미 | Locus |
|---|---|---|---|
| Contamination/lower quality로 제거한 dataset; 버린 rejection sample `K − 1`개 | 8-gram matcher; reward model | (1) negative marginal value, discard | Tülu 3 §3.2, §4.1.2; Llama 3 §4.2.2 |
| Chosen response가 constraint를 실패한 IF-augmented pair(66k 초과 생성, 약 26k 통과) | constraint verifier | (1) filter로 test; verified subset이 average를 조금 낮춰 Persona IF와 unverified set 사용 | Tülu 3 §5.3 |
| Sample answer가 일관되게 informative하면서 incorrect인 question의 noncompliance(CoCoNot)/refusal target | noncompliance taxonomy; source snippet에 대해 judge인 Llama 3 | (2) content로서 negative | Tülu 3 §3.1.2, §4.2; Llama 3 §4.3.6 |
| Execution/error feedback 뒤 rewrite한 failing code/wrong math trace | parser, linter, unit test; answer check | (2) corrected content; failed attempt가 training dialog에 남는지는 미보고 | Llama 3 §4.3.1, §4.3.3 |
| DPO rejected response | GPT-4o rating(Tülu 3); human, “significantly better”/“better”만(Llama 3) | (4) gradient | Tülu 3 §5.2.1; Llama 3 §4.2.1 |
| 0 또는 −10.0 score의 RLVR response | exact match, constraint verifier | (4) lower advantage를 통한 gradient | Tülu 3 Eq. 8, Table 21 |

어느 report도 verifier/judge false-negative rate를 재지 않는다. Open-instruct는 IF verifier가 나중에 바뀌었으며 Tülu 3 재현에는 old verifier가 필요하다고 문서화한다([[open-instruct-allenai-recipes]], tulu3.md L272--276).

**Mechanism.** Logit `z`의 softmax에서 `∂ log p_y / ∂ z_j = 1[j = y] - p_j`다. `p_j`는 token `j` 확률, `y`는 push-down하는 token이다. `log p_y`를 낮추면 `z_y`가 내려가고 다른 logit은 `p_j`에 비례해 올라간다. 수치 예시 `p = (p_y, p_2, p_3) = (0.02, 0.90, 0.08)`은 gradient `(0.98, -0.90, -0.08)`을 준다. `log p_y`에 step size `η` descent를 하면 `z_y`는 `0.98η` 내려가고 `z_2`, `z_3`는 `0.90η`, `0.08η` 오른다. 작은 `η`에서 probability change는 `dp_j = p_j\,(dz_j - \sum_k p_k\,dz_k)`다. `dz_j`는 logit `j` 변화이며 `\sum_k p_k\,dz_k = 0.7968η`다. `dp_y = -0.0355η`, `dp_2 = +0.0929η`, `dp_3 = -0.0573η`(계산값)다. 가장 높은 확률 token은 `y`에서 제거한 mass 전체를 받고 third token에서도 mass를 가져간다. DPO rejected term은 `y_r`의 모든 token에 이를 적용한다. Chosen/rejected response가 공유하는 token이 conflicting update를 받는 이유라는 Llama 3 저자의 가설이다(§4.1.4). 같은 mechanism은 margin이 커지는 동안 chosen log-probability가 내려가게도 한다. 두 report 모두 chosen log-probability를 log하지 않으며, Llama 3는 이 감소를 막으려 NLL term을 더한다(§4.1.4).

**쓴 control.** Llama 3는 formatting token mask, chosen response NLL coefficient 0.2, low-margin pair discard, most recent batch의 DPO data를 쓴다. Tülu 3는 length normalization, preference pool의 on-policy response, RLVR KL penalty(8B `β = 0.05`), truncated response −10.0 penalty를 쓴다.

**근거와 효과 크기.** 어느 report도 같은 pair의 chosen response SFT와 DPO를 비교하는 식으로 rejected response contribution을 isolate하지 않는다. 따라서 gain 중 negative가 차지한 share는 귀속할 수 없다. Stage-level result에는 regression이 있다. Tülu 3 8B DPO는 AlpacaEval 2/IFEval을 올리면서 safety 93.1 → 87.2, HumanEval 86.2 → 83.9를 낮췄다(§3). Constraint prompt `β = 0.01` RLVR은 content-free output을 냈다(App. B.4).

**진단.** Chosen/rejected log-probability를 따로 log한다. KL/response length/verifiable reward를 함께 추적한다(Tülu 3 Fig. 19, 23). Sweep의 각 `β`에서 sampled output을 읽는다. Safety average를 jailbreak/over-refusal task로 쪼갠다(Table 25). Unseen constraint benchmark pass@1을 보고한다.

**Generality 효과.** Rejected/failed response를 push-down한 DPO/RLVR stage는 8B에서 lower jailbreak robustness(Do-Anything-Now 88.3 → 62.0), higher XSTest(90.4 → 93.3)와 동시에 나타났다(Table 25). 이는 stage 전체를 가로지르는 correlation이다. Rejected term의 causal share는 이 두 recipe에서는 Open question이다(ch-39/ch-43a 유도 참조).

## 설정 recipe

행은 이 장과 관련된 setting을 인용한다. “v5”는 arXiv:2411.15124v5이고, “v3”는 arXiv:2407.21783v3이며, “t3”는 commit `098424c`의 open-instruct `docs/tulu3.md`다. Canonical ledger는 [[open-instruct-allenai-recipes-recipe]], [[llama-3-recipe]]다. [[tulu-3]] card 자체에는 아직 verified ledger가 없고 paper와 충돌하는 값(예: RLVR episode 10,000,000)을 적고 있다.

| Model (정확한 release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | data | 19 source의 example 939,344개(`allenai/tulu-3-sft-mixture`) | v5 Table 7 | verified 2026-09-15 | Fig. 3 intermediate mix; Table 10 removal; Fig. 4 subsampling |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR; schedule; warmup ratio | 5 × 10⁻⁶; linear; 0.03 | v5 Table 11 | verified 2026-09-15 | Fig. 5: Llama 3.0 + Tülu 2 mix에서 5e-6 sum loss가 최고, final mix는 아님 |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | epoch; effective batch; max length | 2; 128 sequence; 4,096 token | v5 Table 11 | verified 2026-09-15 | Fig. 6: 2–7 epoch(Tülu 2 mix) 중 2가 최고; batch ablation은 미보고 |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | loss aggregation | sum loss(paper; 2025년 6월 doc command의 `--reduce_loss sum`); t3@098424c에는 flag가 없고 mean loss가 default | v5 §4.3.2; t3@8781471 L49; PR #1024 | conflict | Fig. 5(sum vs mean, Tülu 2 mix); paper와 2025 command는 sum loss에 일치하지만 pinned command는 그렇지 않음 |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | compute; selection rule | GPU 32개, 6 h; 5 seed 중 최고 single seed | v5 §4.3, Table 14 | verified 2026-09-15 | Table 14: seed 59.8–60.1, soup 60.2 |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | peak LR; epoch; batch; length; compute | 2 × 10⁻⁶; 2; 128; 4,096; GPU 64개, 50 h | v5 Table 11, §4.3 | verified 2026-09-15 | “hyperparameter search”, 값은 출력하지 않음 |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | loss; β; LR; schedule; warmup; epoch | DPO-norm(Eq. 6); 5; 5 × 10⁻⁷; linear; 0.1; 1 | v5 Table 20 | verified 2026-09-15 | Table 18: DPO-norm 57.3 vs SFT base 55.7, PPO 55.5, DPO 55.2(UltraFeedback, early SFT) |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | pair; batch; max length; judge | 271,409; 128; 2,048; GPT-4o-2024-08-06 | v5 Tables 15, 20; §5.2.1 | verified 2026-09-15 | Table 16 mixing run; Table 17 judge |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | compute | H100 GPU 8개, 10시간 | v5 §5.4.1 | verified 2026-09-15 | n/a |
| Llama-3.1-Tulu-3-70B-DPO | 70B | preference | LR; pair | 2 × 10⁻⁷; 334,302 | v5 Tables 19, 15 | verified 2026-09-15 | Table 19: Mix 2에서 2.0e-7일 때 74.35 vs 5.0e-7일 때 71.14 |
| Llama-3.1-Tulu-3-8B-RM | 8B | reward-model | LR; batch; epoch; length; compute | 3 × 10⁻⁶; 256; 1; 2,048; H100 GPU 8개, 9시간 | v5 Table 36, §6.3 | verified 2026-09-15 | 이 값의 ablation은 미보고; Fig. 21에서 PPO value model을 RM으로 초기화하는 것이 최고; final RM은 Tülu 3 SFT의 8B preference mix로 학습(§6.4) |
| Llama-3.1-Tulu-3-8B | 8B | RL | algorithm; reward; no-EOS reward | PPO; verified면 α = 10, 아니면 0; −10.0 | v5 Eq. 8, Table 21 | verified 2026-09-15 | α는 pilot experiment에서 정했고 tuning하지 않음 |
| Llama-3.1-Tulu-3-8B | 8B | RL | prompt | 29,946(GSM8K 7,473; MATH 7,500; IF 14,973) | v5 Table 22 | verified 2026-09-15 | Fig. 19 single-task run |
| Llama-3.1-Tulu-3-8B | 8B | RL | compute | GPU 8개, 65시간 | v5 §6.3 | verified 2026-09-15 | n/a |
| Llama-3.1-Tulu-3-8B | 8B | RL | LR; β; warmup; batch; K; response length | 3 × 10⁻⁷; 0.05; 0.0; 224; 4; 2,048 | v5 Table 21 and caption | verified 2026-09-15 | β sweep [0.1, 0.05, 0.03, 0.01](Fig. 19, 21); §6.4는 β를 0.15까지 시험 |
| Llama-3.1-Tulu-3-8B | 8B | RL | total episode | 100,000(paper); command에서 planned 10,000,000 | v5 Table 21; t3 L329 | conflict | released checkpoint는 “earlier than final”(§6.3); 그 episode는 미보고 |
| Llama-3.1-Tulu-3-8B | 8B | eval-gate | checkpoint rule | 매 100 step; 최고 MATH와 IFEval | v5 §6.4 | verified 2026-09-15 | GSM8K 89.4 / IFEval 84.8 run은 다른 metric 때문에 reject |
| Llama-3.1-Tulu-3-70B | 70B | RL | LR; batch; episode; β; warmup | 1 × 10⁻⁷; 640; 400,000; β 0.7(text) vs 0.07(caption/t3 command); warmup 0.1(text/t3 command) vs 0.07(caption) | v5 §6.4, Table 21 caption; t3 L375–383 | conflict | none reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | algorithm; LR; β; prompt × sample; released checkpoint | GRPO; 5 × 10⁻⁷ constant; 0.01; 48 × 16 = 768; episode 1,474,560(step 1920) | model card “Hyperparamters”(원문 철자), “Learning curves” | verified 2026-09-15 | “further hyperparameter tuning”; ablation table 없음 |
| Llama 3.1 405B | 405B | preference | round | 6 | v3 §4.1.6 | verified 2026-09-15 | none reported |
| Llama 3.1 405B | 405B | SFT | rejection sampling | human-annotation prompt마다 보통 K 10–30; RM이 best 선택 | v3 §4.2.2 | verified 2026-09-15 | none reported |
| Llama 3.1 “largest models” | not scoped | SFT | LR; step; loss | 10⁻⁵; 8.5K–9K step; target token만 | v3 §4.1.3 | verified 2026-09-15 | “work well across different rounds and data mixes” |
| Llama 3.1 | not scoped | SFT | batch; epoch; schedule | not reported | v3 §4.1.3, §4.2.2, Table 7 checked | not reported | n/a |
| Llama 3.1 | not scoped | SFT | long-context share | synthetic 0.1%; bucket 16K/32K/64K/128K; Table 7 example의 0.11% | v3 §4.3.4, Table 7 | verified 2026-09-15 | “careful ablations”, table 없음 |
| Llama 3.1 405B | 405B | preference | DPO LR; β; NLL coefficient; masking | 10⁻⁵; 0.1; chosen에 0.2; header와 termination token | v3 §4.1.4 | verified 2026-09-15 | PPO comparison은 prose에만 있음 |
| Llama 3.1 405B | 405B | merge | model averaging | RM, SFT, DPO에서 서로 다른 data/hyperparameter의 run; weight 미보고 | v3 §4.1.5 | verified 2026-09-15 | ablation 미보고 |

**작은 general-purpose run의 시작점.** Llama 3.1 8B base와 939,344-example Tülu 3 SFT mixture라면, verified 8B 행은 SFT를 LR 5 × 10⁻⁶(linear, warmup 0.03), 최대 4,096 token sequence 128개로 2 epoch, sum loss로 실행한다. 그 뒤 DPO-norm을 β = 5, LR 5 × 10⁻⁷(linear, warmup 0.1), 최대 2,048 token pair 128개로 1 epoch 실행한다. 이어 PPO-based RLVR을 α = 10, β = 0.05, LR 3 × 10⁻⁷, batch 224, K = 4, 2,048-token response로 29,946개 verifiable prompt에 실행한다. Source는 8B에서 각각 GPU 32개 6시간(SFT), H100 GPU 8개 10시간(DPO), GPU 8개 65시간(RL)을 썼고, value model을 초기화하는 reward model에는 H100 GPU 8개 9시간을 추가로 썼다(§4.3, §5.4.1, §6.3). Sum loss는 `conflict` 행에서 온다. Paper는 이를 사용했고 pinned open-instruct command는 사용하지 않는다. Base model에 long context가 있으면 long-context SFT data를 추가한다. Llama 3는 model size를 밝히지 않은 채 example 기준 0.1%를 보고한다(§4.3.4). 이 값 중 어느 것도 다른 base model로 transfer됨을 보이지 않았다. OLMo 2 report는 OLMo 2 base가 이 recipe보다 높은 learning rate를 필요로 했다고 말한다([[open-instruct-allenai-recipes-recipe]]; ch-34).

## 일반화 관점

**(a) 폭을 넓힌 요인.**

- Diverse real-user chat: WildChat을 제거하면 8B five-benchmark unseen average가 29.9 → 28.8, BigCodeBench가 11.5 → 7.4, AlpacaEval 2가 12.4 → 7.5로 낮아졌다([[tulu-3]] Tables 10, 32).
- 한 benchmark의 형식이 아니라 해당 skill에 맞는 skill data: Math data는 MATH와 DeepMind Mathematics를 함께 움직였다(Table 32).
- Duplicated prompt보다 unique prompt, 그리고 preference data의 on-policy response(Tülu 3 Fig. 8–11).
- IFTrain(29 type)과 IFEval(25 type)의 constraint에서, prompt마다 1개에서 n개의 constraint를 둔 IF-RLVR은 disjoint-constraint benchmark IFBench를 25.2 → 44.6으로 올렸다([[ifbench]] Table 6). Prompt당 constraint 수와 variable range의 ablation은 [[ifbench]] §4.1–4.3에 있다.
- Hard prompt를 남긴 capability expert, execution feedback, difficulty-aware filtering([[llama-3]] §4.3.1, §4.2.3; mechanism은 ablation number 없이 기술됨), 그리고 pre-trained long-context ability를 보존한 long-context SFT share(§4.3.4; table 없이 보고됨).

**(b) Narrowing 또는 forgetting을 일으킨 요인.**

- Benchmark taxonomy로 만든 data: IFEval의 25 constraint type을 따르는 IF 부분을 가진 persona data를 제거하면 IFEval은 −19.2, IFEval-OOD는 +0.4 변했다(Table 32).
- SFT data를 더 넣으면 TruthfulQA가 낮아졌다(Tülu 3 Fig. 4).
- Preference/RL stage는 8B에서 jailbreak robustness와 safety average를 낮췄다(Table 25). Tülu 3.1 RL run(GRPO와 다른 setting 변경)은 safety 81.2로 끝났고 Tülu 3 8B는 85.5였다([[tulu-3.1]]).
- Constraint RLVR의 낮은 KL은 content-free answer를 냈다(App. B.4). Constraint-only reward는 AlpacaEval 2를 33.5 → 21.3으로 낮췄다([[ifbench]] Table 3).
- Llama 3의 short-only SFT는 long-context ability를 regression시켰다(§4.3.4). Strict quality filter는 hard prompt를 제거해 score를 낮췄고(§4.3.1), 405B가 자기 생성 code data로 학습했을 때는 개선되지 않았다(§4.3.1).

**(c) 이 stage에서 측정하는 방법.**

- Format이 다른 per-skill development/unseen split을 두고, 모든 ablation에서 평가한다([[tulu-3]] §7.4.1).
- Constraint-disjoint instruction-following benchmark(IFEval-OOD, IFBench)와 constraint를 제거한 LLM-judge quality를 쓴다([[ifbench]] §5).
- Safety를 jailbreak와 over-refusal task로 나누고 model size별 violation/false-refusal rate를 본다(Tülu 3 Table 25; Llama 3 Fig. 18).
- 두 suite 모두에 대해 8-gram token-overlap decontamination을 하고 contamination으로 인한 estimated performance gain도 구한다([[tulu-3]] §3.2; [[llama-3]] Table 15).
- Base가 long context를 지원하면 SFT gate 안에 long-context evaluation을 넣는다(Llama 3 §4.3.4, Table 21).
- Ablation noise floor로 seed variation을 사용한다(Tülu 3 Table 14). 알려진 error는 answer extraction의 format sensitivity(§7.4.1), real-user data의 benchmark overlap(App. B.2), internal table inconsistency(MMLU 65.9 vs 62.1)다.

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| Development score만으로 data/checkpoint 선택 | development average는 오르지만 unseen score는 평평하거나 하락 | Disjoint unseen suite에서 각 ablation을 재실행(Tülu 3 Table 32 pattern) |
| 평가 constraint taxonomy로 IF data 구성 | IFEval이 IFEval-OOD/IFBench보다 훨씬 크게 상승 | 둘 다 보고하고 training에서 constraint type을 hold out |
| Tülu 3.1을 PPO-vs-GRPO ablation으로 취급 | +1.5 average를 GRPO 하나에 귀속 | 바뀐 모든 setting(β, LR, schedule, prompt당 sample, K, mini-batch, value model, EOS penalty, released episode)을 열거 |
| `total_episodes 10,000,000`을 Tülu 3 run length로 복사 | run length/schedule이 paper와 다름 | Table 21은 100,000, Tülu 3.1은 episode 1,474,560에서 release |
| Tülu 3 SFT를 mean loss로 재현 | 같은 LR에서 average가 paper보다 낮음 | Loss-reduction setting 검사; Tülu 3는 sum loss로 LR tuning(§4.3.2) |
| Long-context base에 short-only SFT | Short benchmark는 개선하지만 needle/InfiniteBench score 하락 | SFT gate에 long-context evaluation을 넣고 base checkpoint와 비교 |
| RLVR/DPO의 target gain만 보고 | TruthfulQA/safety 하락이 full table에서만 보임 | Stage마다 모든 development evaluation을 공개하고 safety를 task별로 분리 |
| Constraint RLVR에서 아주 낮은 KL coefficient | Verifiable reward는 오르지만 output이 task content를 잃음 | Sweep의 각 β에서 output sample을 읽고 KL과 average score 추적(Tülu 3 Fig. 21) |
| Exact match만으로 decontaminate | Number만 바꾼 math problem처럼 몇 token만 다른 near-duplicate가 training set에 남음 | 두 suite에 대해 8-gram token-overlap 확인; Tülu 3의 8-gram check는 HumanEval의 70.7%가 Evol CodeAlpaca와 overlap함을 발견(Tülu 3 §3.2, Table 37) |
| Ablation당 seed 하나 | Seed range보다 작은 차이를 effect로 읽음 | Seed를 실행; Tülu 3 70B seed range는 2.6 point(Table 14) |
| DPO loss에 formatting token을 남김 | Tail repetition 또는 abrupt termination token | Header와 termination token mask(Llama 3 §4.1.4) |

## 이해 확인

1. Math data removal은 MATH와 DeepMind Mathematics를 둘 다 낮췄지만, persona data removal은 IFEval을 낮추고 IFEval-OOD는 낮추지 않았다. 각 dataset의 어떤 속성이 이 차이를 설명하며, 새 benchmark에 대해 무엇을 예측하는가?
2. GSM8K와 IFEval score가 가장 높은 checkpoint가 왜 generalist model에 release하기에는 틀릴 수 있는가? §6.4와 Table 23을 사용해 설명하라.
3. Softmax gradient를 사용하여 DPO의 rejected term이 왜 chosen response의 log-probability를 낮출 수 있는지, Llama 3의 NLL term과 token masking이 각각 이를 어떻게 바꾸는지 설명하라.
4. Llama 3 long-context example은 SFT example의 0.11%, processed token의 약 4.96%, final-response token의 약 0.26%다. Long-context ability 유지 여부를 예측하는 데 어느 숫자가 유용할 것으로 보는가? Short-only DPO가 short-only SFT와 같은 regression을 내지 않을 수 있는 이유는 무엇인가?
5. Tülu 3는 rejection sampling이 compute 가치가 없다고 찾았지만 Llama 3는 매 round에 쓴다. 어느 결과도 틀리지 않은 채 이를 설명할 수 있는 두 setup의 차이 세 가지를 열거하라.
6. Tülu 3.1은 average를 1.5 point 개선하고 safety를 4.3 point 낮췄다. 어느 experiment가 있어야 두 변화 중 어느 하나를 GRPO에 귀속할 수 있는가?
7. Llama 3는 reward model **또는** Llama rating에서 high quality인 example을 남기고, quality × difficulty로 deduplicate한다. §4.3.1의 code-filter result를 고려할 때 difficulty를 sort key에서 빼면 kept set은 어떻게 달라지는가?
8. Tülu 3 safety는 SFT에서 다른 skill과 orthogonal하다고 보고했지만 DPO/RLVR 중 safety는 하락했다. Table 24의 어느 measurement gap이 더 강한 결론을 막는가?

## 연결

- **이전:** ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood. “negative”의 네 의미를 여기서 두 pipeline에 적용한다.
- **다음:** ch-34 — Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports. 비교를 이어가며 Tülu 3/Llama 3를 포함한 side-by-side SFT ledger를 담는다.
- ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing(Tülu 3에서 persona synthesis 사용).
- ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares; ch-30c — Weight Averaging and Model Merging for Generalist Models.
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation(Llama 3 loop).
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation.
- ch-39 — Offline Preference Optimization: DPO and Its Variants; ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO; ch-44 — Process Supervision and Verifiable Rewards.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation; ch-48 — Contamination Detection and Its Effect on Reported Scores; ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming.

## 출처

- [[tulu-3]] — Tülu 3 report(arXiv:2411.15124v5): 여기서 인용한 모든 Tülu 3 table, equation, ablation, hyperparameter는 paper에서 읽었다. Card는 검증하지 않았으며 RLVR episode, verifier, stage gain에서 paper와 충돌한다.
- [[tulu-3-sft-mix]] — SFT mixture와 57/43 split의 dataset card view; paper Table 7은 OpenMathInstruct 2를 추가한다.
- [[tulu-3.1]] — Llama-3.1-Tulu-3.1-8B model card: GRPO hyperparameter, benchmark table, released episode.
- [[tulu-3-1]] — Completeness를 위해 열거한다. Multi-base-refresh 설명은 model card/blog가 뒷받침하지 않으며 사용하지 않는다.
- [[rlvr-tulu3]] — RLVR summary card; reward, verifier, PPO value가 paper와 충돌하므로 사용하지 않는다.
- [[ifbench]] — IFBench와 IF-RLVR: unseen-constraint generalization과 narrowing 수치.
- [[open-instruct-allenai-recipes]], [[open-instruct-allenai-recipes-recipe]] — Released launch command, loss-reduction drift, IF verifier change, Tülu 3.1 run setting.
- [[llama-3]], [[llama-3-recipe]] — Llama 3 report(arXiv:2407.21783v3): round, RM, rejection sampling, SFT, DPO, averaging, capability pipeline, filter, safety, contamination, long context.
- [[long-context-llama3]] — Long-context card; §3.4.2/§4.3.4에서 확인한 statement만 쓰며 stage별 budget은 report에 없다.
- [[llama-3-synthetic-pipeline]] — 2024년 4월 Meta blog: Llama 3 8B/70B의 SFT + rejection sampling + PPO + DPO 설명과 withheld 1,800-prompt human evaluation set.
- [[dpo]] — DPO-norm과 비교하는 standard DPO objective.
- [[persona-hub]] — Tülu 3 prompt에 쓴 persona-conditioned synthesis method.
