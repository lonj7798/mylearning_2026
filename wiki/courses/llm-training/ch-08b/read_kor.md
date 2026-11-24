<!-- chapter: ch-08b
     track: pretraining
     kind: content
     title: Next-Token Pretraining이 일반 능력을 만드는 이유: In-Context Learning, Emergence, Predictability
     deps: [ch-08a]
     sources: [[gpt-2-unsupervised-multitask]], [[gpt-3-few-shot]], [[icl-data-distributional-properties]], [[induction-heads]], [[emergent-abilities]], [[emergent-abilities-mirage]], [[emergence-loss-perspective]], [[emergence-loss-perspective-recipe]], [[same-loss-better-downstream]], [[embers-of-autoregression]], [[predicting-downstream-elusive]], [[scaling-laws-unreliable-downstream]], [[phi-textbooks]], [[physics-of-lm-3]]
     figures: figures/metric-artifact-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 08b — Next-Token Pretraining이 일반 능력을 만드는 이유: In-Context Learning, Emergence, Predictability

> **핵심 통찰.** broad text의 next-token prediction은 text가 task demonstration을 포함하므로 많은 task를 한꺼번에 학습한다. GPT-2(1.5B)는 CoQA training 없이 CoQA F1 55에 도달하고([[gpt-2-unsupervised-multitask]] §3.5), GPT-3 175B는 K개의 in-context example과 weight update 없이 2-digit addition 100.0%, TriviaQA 71.2%에 도달한다([[gpt-3-few-shot]] Tables 3.9, 3.3). 통제 실험에서 ICL은 training data가 bursty하고 rare class가 많을 때만 나타나며([[icl-data-distributional-properties]] §3), [[induction-heads]]가 연구한 transformer LM에서는 induction head와 함께 약 2.5-5B token 창에서 일찍 형성된다(Argument 1). ability가 예측 가능해 보이는지는 측정에 의존한다. exact match/multiple-choice grade에서 jump하는 ability는 시험 사례에서 token edit distance/Brier score에서는 부드럽게 바뀌지만([[emergent-abilities-mirage]] §3-4), corpus/tokenizer 하나에서는 MMLU/GSM8K가 pretraining loss 약 2.2까지 chance에 머물며 MMLU/C-Eval threshold는 continuous metric에서도 남는다([[emergence-loss-perspective]] §3). loss는 transfer를 정하지 않는다. 단순 setting에서 같은 pretraining loss model도 transfer가 다르고([[same-loss-better-downstream]] §3), GPT-3.5/GPT-4는 deterministic task rare variant와 low-probability target output에서 덜 정확하다([[embers-of-autoregression]] §5-6).
>
> **지침.** 비교 model이 chance 근처인 task에서 pretraining ablation을 판단하면 thresholded metric 옆에 continuous per-item score(correct answer log-probability, token edit distance, BPB)를 읽고 expected accuracy model이 0점을 낼 가능성이 낮도록 충분히 큰 evaluation set을 써라. 시험한 model family에서 thresholded metric이 continuous score가 보인 개선을 숨겼다([[emergent-abilities]] App. A.1; [[emergent-abilities-mirage]] Fig. 3-4; §6 계산 예). 모든 비교 model이 chance 위이면 thresholded metric을 item-sampling error와 함께 보고하라(ch-51). pretraining loss로 checkpoint 둘을 비교하면 corpus/tokenizer가 같은 checkpoint만 비교하라. loss value는 이들 사이 비교 불가다([[emergence-loss-perspective]] §7). downstream probe도 더하라. 같은 loss가 [[same-loss-better-downstream]]에서 같은 transfer를 주지 않았다(§3). 일반 능력 claim이 common task format에 기대면 deterministic task의 rare/low-probability variant도 더하라. GPT-4 rot-1 정확도는 0.82, rot-2는 0.02다([[embers-of-autoregression]] §5.1).

## 범용 모델에서 이 장이 중요한 이유

ch-08a는 pretraining loss가 parameter/token과 함께 예측 가능하게 떨어지고 downstream accuracy는 예측하기 더 어렵다는 것을 보였다. 이 장은 둘 사이 질문을 다룬다. broad text 다음 token 예측이라는 하나의 목적함수가 training target이 이름 붙이지 않은 능력을 왜 만들며, 작은 run에서 언제 그 능력을 forecast할 수 있는가이다.

질문은 세 방식으로 측정할 수 있다. 첫째 ch-08a model-ladder처럼 작은 model에서 pretraining decision(data mixture, filter, token budget)을 고를 때, small scale chance 능력은 thresholded metric에서 selection signal을 주지 않는다. 둘째 base model general claim이 few-shot benchmark 점수에 기대면 task learning/task recognition/contamination을 섞는다(§2). 셋째 equal pretraining loss를 equal capability proxy로 쓰면 §8의 setting에서 proxy가 실패한다.

Pipeline 위치: pretraining(ch-08a~ch-17 outline block, 이 장 포함; ch-15/ch-16은 later stage) → mid-training(ch-32~ch-32f) → SFT → preference optimization → RL → evaluation. 뒤 stage는 여기서 정한 두 속성을 상속한다. base model이 context에서 인식할 수 있는 task format과 잘 처리하지 못하는 low-probability variant다. 자기 chapter 전에 쓴 용어: *few-shot / K-shot*(prompt에 둔 K solved example), *cloze*(fill-in-the-blank completion format), *BPB*(byte length로 나눈 bit negative log-likelihood로 tokenizer effect를 줄임; ch-08a §6), *nat*(natural logarithm loss).

## §1 Unsupervised multitask learning으로서의 language modeling(GPT-2)

**정의.** language model은 symbol sequence 확률을 next-symbol conditional로 factorize하고 observed text log-probability를 maximize하도록 학습한다([[gpt-2-unsupervised-multitask]] §2, Eq. 1).

```
p(x) = Π_{i=1..n} p(s_i | s_1, ..., s_{i-1})
```

x는 text, s_i는 i번째 symbol(GPT-2에서는 byte-level BPE token), n은 길이다.

**문제.** supervised system은 task 하나의 p(output | input)을 배운다. GPT-2 저자들은 이를 “competent generalist rather than narrow expert”가 아닌 “narrow expert”로 설명하며, 당시 최대 multitask NLP effort 둘이 (dataset, objective) pair 10개/17개로 학습했다고 지적한다. meta-learning 관점에서는 training example 10개/17개와 비교한다(§1).

**메커니즘.**
1. general system은 p(output | input, task)를 model해야 한다. language는 task/input/output을 한 sequence로 쓸 수 있다. 예를 들면 (translate to french, english text, french text)다(§2).
2. 이런 sequence의 supervised objective는 output token에서만 평가한 language-modeling objective다. 저자들은 unsupervised objective global minimum이 supervised objective global minimum이기도 하며, open question은 optimization이 실제 도달하는지라고 주장한다(§2).
3. broad web text는 자연 발생 task demonstration을 포함한다. Table 1은 non-English page를 의도적으로 뺐는데도 WebText에서 찾은 English-French translation pair를 열거한다. language detector는 WebText에서 French text 10 MB를 찾았다(§2.1, §3.7).
4. test time task는 text로 지정한다. summarization은 article 뒤 `TL;DR:`, translation은 `english sentence = french sentence` pair, reading comprehension은 document + `A:`다(§3.5-3.7).

**계산 예(task hint).** CNN/Daily Mail에서 `TL;DR:`이 있는 GPT-2 평균 ROUGE(R-AVG)는 21.40, hint 없이는 15.03, article random sentence 셋은 20.98이다(Table 4). `21.40 − 15.03 = 6.37`은 `TL;DR:` text hint 효과(본문은 6.4로 round, §3.6)다. `21.40 − 20.98 = 0.42`는 random-sentence baseline 대비 margin이다. model은 task가 text에서 이름 붙으면 행동을 바꾸지만 summary 점수는 random sentence 셋을 고르는 extractive baseline의 반 ROUGE point 안에 있다.

**증거.** WebText(8 million document, 40 GB; §2.1)로 학습한 네 model(117M, 345M, 762M, 1542M parameter; Table 2)을 task-specific training 없이 평가했다. 논문은 이를 zero-shot이라 부른다. translation/Natural Questions prompt는 example pair를 포함한다(§3.7-3.8).

- CoQA(conversational question answering): greedy decoding development set F1 55로, 127,000개 이상 QA pair로 학습한 baseline 4개 중 3개와 같거나 높다(§3.5).
- Language modeling: dataset 8개 중 7개 state of the art. LAMBADA(passage 마지막 word 예측)는 prior state of the art perplexity 99.8 → 8.6, accuracy 19% → 52.66%, stop-word filter면 63.24%다(§3.1, §3.3, Table 3).
- Natural Questions: exact match 4.1%, model이 가장 confident한 question 1%에서는 63.1%다(§3.8).
- zero-shot performance는 task 전반 capacity와 log-linear하게 개선한다(Abstract, Fig. 1).

상태: 수치마다 **단일 연구 결과**다. scaling trend는 §2 [[gpt-3-few-shot]]이 175B까지 연장하지만 overlap author가 있는 같은 organization report이므로 independent replication은 아니다.

**조건과 한계.** WMT-14 English→French는 BLEU 5로 word-by-word lexicon substitution보다 낮고 French→English는 11.5 BLEU다(§3.7). 저자들은 practical application에서 zero-shot performance가 “still far from use-able”이고 많은 practical task에서 GPT-2가 random보다 나을 가능성이 낮다고 한다(§6). train/held-out WebText perplexity는 size와 함께 같이 개선하며 저자들은 이를 underfitting으로 읽는다(§4, Fig. 4).

**범용 모델 함의.** GPT-2 data design은 target benchmark 하나와 similarity로 고른 Common Crawl subsample을 거부하며 목표를 “to avoid making assumptions about the tasks to be performed ahead of time”이라고 했다(§2.1). step 1-3 mechanism은 corpus에 많은 task demonstration이 있어야 한다. §10은 이것이 data selection에 뜻하는 바를 다시 다룬다.

## §2 Scale에서의 few-shot in-context learning(GPT-3)

**정의.** *In-context learning*(ICL)은 forward pass 안의 task adaptation이다. model은 natural-language instruction 및/또는 K demonstration을 condition으로 받고 weight update 없이 새 instance를 complete한다([[gpt-3-few-shot]] §1, Fig. 1.1). *zero-shot*은 instruction만(K = 0), *one-shot*은 K = 1, *few-shot*은 2,048-token context에 맞는 많은 demonstration이며 보통 K = 10~100이다(§2).

**문제.** fine-tuning은 보통 task-specific example 수천~수십만 개를 요구한다(§1). 저자들은 “the potential to exploit spurious correlations in training data fundamentally grows with the expressiveness of the model and the narrowness of the training distribution”이라고 주장한다(§1). ICL은 좁은 task별 training distribution 없이 task를 수행하는 방법으로 제안된다.

**메커니즘과 protocol.**
1. pretraining은 outer loop(논문 용어)다. 175B autoregressive model과 더 작은 model 7개(125M-13B)는 각각 300B token에서 학습한다(Table 2.1).
2. inner loop는 context에서 실행한다. K example은 task training set에서 random draw하고 newline 1개 또는 2개로 분리한다(§2.4).
3. multiple choice는 token별 normalization한 completion likelihood를 비교해 점수화한다. ARC/OpenBookQA/RACE는 `P(completion | context) / P(completion | answer_context)`이며 answer_context는 `"Answer: "` 또는 `"A: "`다(§2.4).
4. K가 클수록 “usually but not always better”이므로 development set이 있으면 그곳에서 K를 고른다(§2.4).

**계산 예(arithmetic, GPT-3 175B, task당 random problem 2,000개, exact match; Table 3.9).**

| Task | Zero-shot | One-shot | Few-shot |
|---|---|---|---|
| 2-digit addition | 76.9 | 99.6 | 100.0 |
| 3-digit addition | 34.2 | 65.5 | 80.4 |
| 4-digit addition | 4.0 | 14.0 | 25.5 |
| 2-digit multiplication | 19.8 | 27.4 | 29.2 |

표 읽기: 3-digit addition의 zero-shot에서 few-shot 이동은 weight를 바꾸지 않고 `80.4 − 34.2 = 46.2`점 accuracy를 올린다. §3.9.1 본문은 3-digit addition 80.2%이고 표는 80.4다. 저자들은 training data에서 3-digit test problem을 두 string form으로 search하여 addition 2,000개 중 17개(0.8%), subtraction 2,000개 중 2개(0.1%)를 찾았다(§3.9.1). 13B model은 2-digit addition/subtraction을 약 절반 풀고 다른 arithmetic operation은 모두 10% 미만이다(§3.9.1).

**증거.**
- TriviaQA(closed book): zero/one/few-shot 64.3 / 68.0 / 71.2(§1, Table 3.3).
- LAMBADA accuracy: 76.2 / 72.5 / 86.4. few-shot cloze format은 model에 정확히 word 하나를 기대한다고 알려 준다(Table 3.2, §3.1.2).
- K = 100 word manipulation: random insertion은 few-shot 67.2, zero-shot 8.26; reversed word는 few-shot 0.44다(Table 3.10). 저자들은 artificial task zero-shot failure를 model이 test time에 task를 학습한다는 증거로 읽는다(§3.9.2).
- accuracy 단위 benchmark 42개에서 zero/few-shot gap은 model size와 함께 커진다(Fig. 1.3, §1).

상태: 수치마다 **단일 연구 결과**다.

**조건과 한계.**
- 일부 comparison task few-shot 결과는 chance 근처다. WiC(두 sentence word meaning 동일 여부)는 few-shot 49.4%이며 논문은 random chance라 부른다(§3.7). ANLI(adversarial NLI)는 175B보다 작은 model이 few-shot이어도 약 33%, three-way chance level이며 저자들은 175B가 Round 3에서 “signs of life”를 보인다고 한다(§3.8, §5).
- few-shot learning이 task를 “from scratch” 배우는지 training에서 본 task를 인식하는지는 모호하며 task별로 다를 수 있다고 한다(§5). 논문의 **Open question**이다.
- Contamination: filtering bug가 training data에 benchmark overlap 일부를 남겼다. clean subset은 13-gram overlap 없이 평가했다. PIQA(29% flagged, 3점 하락)와 Winograd(45% flagged, 2.6% 하락)는 asterisk며 Wikipedia language-modeling benchmark 4개와 CBT는 보고하지 않는다(§4).

**범용 모델 함의.** few-shot evaluation은 base-model generality를 직접 probe하며 protocol choice(K, normalization, format)가 점수를 바꾼다. ch-00 §5가 이 course의 base-model formulation(cloze/multiple choice)을 정하고, ch-47은 harness가 이 선택을 기록하는 법을 다룬다.

## §3 어떤 data property가 in-context learning을 만드는가

**정의**([[icl-data-distributional-properties]] §1, §3).
- *Burstiness*: item이 시간상 uniform하게가 아니라 context 안 cluster로 나타남.
- *Long tail*: 드물게 나타나는 class가 매우 많음.
- *Dynamic meaning*: item 하나가 여러 label로 map됨(label multiplicity), 또는 class 하나가 높은 within-class variation을 가짐.
- *In-weights learning*(IWL): ICL과 달리 standard supervised learning처럼 weight에 저장된 정보를 사용.
- *Zipfian marginal*: class rank x의 probability `p(X = x) ∝ x^(−α)`이며 α ≥ 0은 skew를 정한다(α = 0은 uniform)(§3, Eq. 1).

**문제.** language에서 학습한 transformer는 meta-training 없이 ICL을 보인다. 연구는 architecture/data distribution/both 중 training regime 어느 부분이 이를 만드는지 묻는다(§1).

**메커니즘(experimental design).**
1. data는 Omniglot handwritten character: class 1,623개, 각각 image 20개다(§2.1).
2. training sequence는 image-label pair 8개와 query image 하나다. model은 query label cross-entropy로 학습하며 label은 supervised learning처럼 training 전반 fixed다(§2.1-2.2).
3. *bursty* sequence에서는 query class가 context에 3번 나타나며 second class도 3번 나타나 가장 frequent label shortcut을 막는다. non-bursty sequence는 uniform sample이다(§2.1).
4. ICL은 holdout class에서 4-shot 2-way classification으로 평가한다. label은 sequence마다 0/1로 random reassign되고 chance는 1/2다. IWL은 context에 나타나지 않는 trained class에서 평가하며 chance는 보통 1/1,600이다(§2.3).
5. model은 embedding size 64, head 8개 12-layer causal transformer이며 500k step 학습한다(App. A).

**결과**(논문은 수치 표가 아니라 figure trend로 보고).
- training bursty sequence가 많을수록 ICL은 많고 IWL은 적다(Fig. 2).
- p(bursty) = 0.9에서 class 수를 100 → 1,600 → 12,800으로 올리면 ICL은 높고 IWL은 낮다. “we need both burstiness and a large number of classes for in-context learning to emerge”(§3.1, Fig. 3).
- label multiplicity/within-class variation은 각각 ICL을 높인다(Figs. 4-5).
- uniform class frequency에서는 tradeoff라 ICL/IWL 둘 다 유지하는 model이 없다. Zipfian class(12,800 class, p(bursty) = 0.9)에서는 α = 1이 common class 10개에서 ICL/IWL 모두 유지한다. rare class는 어느 α에서도 memorized되지 않는다(§3.2, Fig. 6).
- ICL은 transient할 수 있다. model은 “can in some cases lose an initial bias towards in-context learning, moving towards in-weights learning over the course of training”(§3.1; Fig. 2 caption도 참조).
- matched depth/hidden size/parameter count vanilla RNN/LSTM은 architecture당 15-run hyperparameter sweep 전체에서 ICL chance를 넘지 못한다(§3.3, Fig. 7, App. C.1).

**계산 예(Zipf share, Eq. 1에서 도출).** M class(§2 K는 shot 수이므로 여기서는 K 대신 M)를 쓰면 top class share는 `1 / H_M(α)`이고 `H_M(α) = Σ_{k=1..M} k^(−α)`, k는 class rank다.
- α = 0, M = 12,800: 모든 class share `1 / 12,800 = 0.0078%`.
- α = 1, M = 12,800: H ≈ 10.03, top class 10.0%, top 10 class는 `H_10(1) / H = 2.93 / 10.03 = 29.2%`.
- α = 3: top 3 class는 `(1 + 1/8 + 1/27) / 1.202 = 1.162 / 1.202 = 96.7%`이며 α = 3에서 common class 셋이 data 97%라는 논문 진술과 맞는다(§3.2, footnote 3).
α = 1에서 example 약 71%는 top 10 밖 class에서 오고 각 class share는 1% 아래다(class 11: `1 / (11 × 10.03) = 0.91%`). 이것이 저자들이 ICL과 연결하는 long tail이다.

**조건과 한계.** task는 text next-token prediction이 아니라 image-label sequence classification이며 저자들은 next-token training/symbolic input을 future work로 열거한다(§4, “Future directions”). ICL은 novel class이지만 이전에 본 label에서 평가한다(App. B). α = 1 결과는 “for this particular training regime”에서 성립한다(Fig. 6 caption). 상태: **단일 연구 결과**다.

**범용 모델 함의.** 결과는 size만 아니라 distributional property를 ICL 조건으로 이름 붙인다. burstiness/class skew를 바꾸는 corpus-level operation(domain을 uniform share 쪽으로 rebalance, within-document repetition 제거)이 LLM pretraining ICL을 바꾸는지는 **Open question**이다. 논문은 text corpus를 시험하지 않는다. 저자들은 supervised dataset이 uniform distribution으로 rebalance되는 경우가 많고, 이런 data로 학습한 non-language model은 ICL 획득 기회를 놓칠 수 있다고 제안한다(§4, “Broader implications”; 저자들의 **해석**).

## §4 In-context copying의 mechanism으로서의 induction head

**정의.** *induction head*는 random token 반복 sequence에서 두 속성을 보이는 attention head다([[induction-heads]] Key Concepts).
1. *Prefix matching*: current 및/또는 recent token 뒤에 온 earlier token에 attention한다.
2. *Copying*: output이 attention한 token의 logit을 높인다.
둘은 `[A][B] … [A] → [B]`를 완성한다. small model circuit은 다른 layer head 둘로 이루어진다. previous-token head가 각 token 정보를 다음 position으로 복사하고 induction head가 그것으로 current token이 이전에 나타난 위치를 찾는다.

**문제.** ICL은 행동이다. 논문은 이를 구현하는 circuit을 찾고 specific task 선택 없이 ICL을 측정한다.

**수식(ICL score).**

```
ICL_score = mean over sequences of [ loss(token 50) − loss(token 500) ]     (nats)
```

`loss(token i)`는 context position i model prediction cross-entropy다. 논문은 definition을 Key Concepts에서 “500th minus 50th”, Argument 1에서 “50th token loss minus the 500th token loss”로 한 번씩 쓴다. 보고 값은 positive이므로 두 번째 ordering이 사용된 것이다. 저자들은 다른 index choice도 결론을 바꾸지 않는다고 보고한다.

**계산 예.** score 0.4 nat은 position 500 correct token의 token당 probability가 평균적으로 position 50의 `e^0.4 = 1.49`배(geometric-mean ratio)라는 뜻이다. phase change 전 score는 0.15 nat 아래(`e^0.15 = 1.16`), 후에는 약 0.4 nat이다(Argument 1).

**증거.** 각 training run 하나의 decoder-only model 34개: small attention-only/small MLP(1-6 layer, d_model 768, head 12, context 8,192, 약 10B token), 4 layer/13M부터 40 layer/13B non-embedding parameter full-scale model, “smeared key” model(Model Analysis Table)이다. *phase change*는 다음 abrupt training window의 논문 명칭이다.

- **Co-occurrence(Argument 1):** 약 2.5-5B token window에서 ICL score가 오르고 induction head가 형성되며 training loss가 bump(loss curve 유일 non-convex 부분)를 보이고 token별 loss trajectory가 방향을 바꾼다. final ICL 75% 초과가 이 window에서 형성된다. one-layer model은 induction head가 없고 “not much in-context learning ever forms”다. Model Details는 small model phase change를 약 1-3B token에 둔다. 상태: correlational.
- **Co-perturbation(Argument 2):** smeared-key architecture는 각 head가 current/previous key를 섞게 한다. `k_j^h ← σ(α_h) k_j^h + (1 − σ(α_h)) k_{j−1}^h`이며 k_j^h는 position j head h key vector, α_h는 head h trainable scalar, σ는 logistic function이다. 이것으로 ICL은 one-layer model에도 형성되고 two-layer model에서는 더 이르게 형성된다. Argument 2는 “two-layer and larger models”라 하지만 Model Details는 one/two layer smeared-key model만 제시한다. 상태: interventional.
- **Ablation(Argument 3):** test time induction head 제거는 small attention-only model ICL 거의 전부를 제거한다. full-scale model ablation은 없다. 상태: small model에 causal.
- **Generality(Argument 4):** 13B에서 strict definition을 만족하는 head는 literal copying 외에도 translation(layer 40 중 7, copying 0.20, prefix-matching 0.85), abstract pattern completion(layer 8, 0.69 / 0.94)을 수행한다. literal copying은 layer 21, 0.89 / 0.75다.

**조건과 한계.** large model에서 저자들은 evidence를 medium/correlational로 평가한다. model당 snapshot 15개는 time resolution이 낮고, shared cause(예: attention layer composition 학습)가 induction head와 다른 ICL mechanism을 모두 만들 수 있다(Argument 1, “Assessing the Evidence”). phase change 뒤 ICL score는 2-layer/13B가 거의 같으며 large model의 small model 대비 advantage 대부분은 context 첫 10 token 안에서 생긴다(Unexplained Curiosities, “Seemingly Constant In-Context Learning Score”). 상태: **단일 연구 결과**, causal 부분은 small model에 한정된다.

**범용 모델 함의.** mechanism은 context 내 repetition에서 작동한다. §3과 연결된다. bursty data는 context window 안에 repeated item을 둔다(**해석**, 이 course; 어느 논문도 link를 직접 시험하지 않음). ICL score는 다른 token index loss로 정의되며 논문은 final index 8,192에도 계산하므로 같은 measurement는 long context에 적용한다. ch-32c는 long-context quality proxy로 loss/perplexity를 다룬다.

## §5 Emergent ability: 주장

**정의.** “An ability is emergent if it is not present in smaller models but is present in larger models”([[emergent-abilities]] §2). training FLOPs 대비 performance plot에서 pattern은 일정 scale까지 near-random, 이후 random보다 훨씬 높은 값으로 상승한다. 저자들은 ability가 emergence하는 scale이 immutable property가 아니며 higher-quality data/다른 training change로 낮아질 수 있다고 말한다(§2).

**문제.** loss는 compute와 부드럽게 scale한다(ch-08a). 일부 ability가 그렇지 않으면 small-model experiment는 이를 forecast할 수 없다.

**증거**(Table 1, above-random performance가 나타나는 scale):

| Ability | Model family | Training FLOPs | Parameters |
|---|---|---|---|
| 3-digit addition/subtraction, few-shot | GPT-3 | 2.3E+22 | 13B |
| MMLU, 57-topic average, few-shot | GPT-3 | 3.1E+23 | 175B |
| Word in Context(WiC), few-shot | PaLM | 2.5E+24 | 540B |
| math word problem의 chain-of-thought | LaMDA | 1.3E+23 | 68B |
| instruction fine-tuning 뒤 instruction following | FLAN | 1.3E+23 | 68B |
| 8-digit addition scratchpad(fine-tuned) | LaMDA | 8.9E+19 | 40M |

- 저자들은 200개 초과 task collaborative benchmark BIG-Bench 210 task 전체를 manual classify했다. Appendix E는 GPT-3/LaMDA에서 emergent인 task 25개, PaLM에서만 emergent인 추가 42개(목록에서 도출한 count), smoothly scale하는 task와 random을 이기지 못하는 task를 열거한다(App. A.3, App. E).
- scale만 factor는 아니다. BIG-Bench task 14개에서 PaLM 62B는 above random인데 LaMDA 137B/GPT-3 175B는 near random이다. 저자들은 ablation 없이 higher-quality data(더 multilingual/code data)와 architecture difference를 가능한 이유로 부른다(§5.2, App. F). 상태: 저자들의 **해석**.
- 저자 자체 cross-entropy analysis: LaMDA로 본 emergent BIG-Bench task 여섯 개 모두 target cross-entropy는 exact match/BLEU/accuracy가 random인 scale에서 개선한다(App. A.1, “Outcome 2”). 그러나 이것이 “does not provide any straightforward indicators of how to predict such emergent behaviors”라고 결론낸다.

**조건과 한계.** emergent task classification은 near-random 기준에 의존한다. co-author 둘은 emergent label task마다 동의했다(App. A.3). Table 1 model family는 data/architecture가 다르므로 scale은 하나의 controlled series에서 나온 것이 아니다. 상태: table row마다 cited original study의 **결과**이며 BIG-Bench task classification은 이 논문 자체다.

**범용 모델 함의.** small-scale ladder 모든 model에서 chance인 ability는 thresholded metric 아래 ladder에서 한 data/architecture decision에 signal을 주지 않는다. §6/§7은 그 scale 아래에도 측정할 수 있는 것을 설명하는 두 account를 준다.

## §6 Metric artifact로서의 emergence

**정의.** [[emergent-abilities-mirage]](Schaeffer, Miranda, Koyejo; arXiv 2023-04)는 fixed task/model family에서 apparent emergence가 researcher의 nonlinear/discontinuous metric 선택에서 올 수 있고 token별 error rate는 부드럽게 바뀐다고 제안한다(Abstract).

**메커니즘(§2).**
1. token별 cross-entropy가 parameter N의 power law로 하락한다고 가정한다: `L_CE(N) = (N/c)^α`, c > 0, α < 0. 저자들은 illustration에만 이 형식을 쓴다.
2. token 하나가 맞을 확률은 `p(N) = exp(−L_CE(N))`다.
3. independent token 가정의 L-token target exact match는 `Accuracy(N) ≈ p(N)^L`이다. p에 nonlinear하다.
4. token edit distance는 `TED(N) ≈ L · (1 − p(N))`이고 p에 거의 linear하다.
5. apparent jump의 추가 factor 둘은 small scale low accuracy를 분해하기에 작은 evaluation set(resolution = 1 / test-set size), large scale sample model size 부족이다(§2).

**계산 예(산술 편의를 위해 이 course가 고른 상수: c = 10^8, α = −0.5, L = 5).**

| Parameter N | `L_CE = (N/c)^α` | `p = e^(−L_CE)` | `Accuracy = p^5` | `TED = 5(1 − p)` |
|---|---|---|---|---|
| 10^8 | 1.000 | 0.368 | 0.007 | 3.16 |
| 10^9 | 0.316 | 0.729 | 0.206 | 1.36 |
| 10^10 | 0.100 | 0.905 | 0.607 | 0.48 |
| 10^11 | 0.032 | 0.969 | 0.854 | 0.16 |

token별 loss는 매 step 같은 factor(√10 = 3.16)로 떨어진다. TED는 3.16 → 1.36 → 0.48 → 0.16, step별 하락은 1.80/0.88/0.32로 계속 작아진다. Accuracy는 0.007 → 0.206 → 0.607 → 0.854, step별 증가는 0.199/0.401/0.247이다. model size 네 개만 있으면 accuracy series는 §5 emergence 정의 pattern, 즉 최소 size near zero 뒤 zero보다 훨씬 높은 값이 된다.

Resolution: 100-item test set N = 10^8에서 expected correct item은 `100 × 0.0067 = 0.67`, zero correct 관측 확률은 `(1 − 0.0067)^100 = 0.51`이다. 약 절반 evaluation에서 small model은 정확히 0점이다. 1,000 item이면 zero 확률은 0.0012로 떨어진다.

[Figure: metric-artifact-explorer.html](figures/metric-artifact-explorer.html)는 target length L/test-set size를 바꾸고 same per-token probability를 exact match/token edit distance로 score하며 model size별 zero score 확률을 보게 한다.

**근거에서 이름 붙인 metric.** *Multiple Choice Grade*는 highest probability가 correct option이면 1, 아니면 0이다. *Exact String Match*는 output string이 target과 같으면 1, 아니면 0이다. *Brier score*는 option predicted probability와 one-hot correct answer 사이 squared error다(§7 계산 예). *emergence score*는 scale 전체 metric total change가 neighbor scale 전형 change보다 크면 큰 BIG-Bench statistic이다(§4, Eq. 1).

**증거.**
- InstructGPT/GPT-3 API model(350M, 1.3B, 6.7B, 175B)의 2-shot 2-digit multiplication/4-digit addition은 Accuracy에서 4/5-digit target emergence를 보지만 같은 output token edit distance에서는 부드럽게 개선한다(§3, Fig. 3). additional generated test data에서는 family 모든 model이 above-chance accuracy다(§3, Fig. 4).
- BIG-Bench meta-analysis: preferred metric 39개 중 최대 5개가 emergence score로 emergence를 보인다. Jason Wei 2022 list “137 emergent abilities of large language models”(paper reference [32], [[emergent-abilities]]와 다른 문서)의 hand-annotated task-metric-model triplet에서 emergence는 metric 4개 아래 나타나며 Multiple Choice Grade/Exact String Match가 claimed emergent ability 92% 초과를 차지한다(§4, Fig. 5).
- Multiple Choice Grade에서 emergent인 LaMDA task는 Brier score에서는 emergent가 아니다(§4, Fig. 6).
- 저자들은 Omniglot character sequence를 L image 모두 correct일 때만 correct로 세는 등 metric 재정의로 vision model apparent emergence를 유도한다(§5).
상태: **단일 연구 결과**다. likelihood-based score가 thresholded metric이 못 보인 곳에서 개선한다는 관측은 [[emergent-abilities]] App. A.1(여섯 BIG-Bench task cross-entropy)에서 **Replicated**지만 그 저자들은 여전히 task를 emergent로 classify한다.

**조건과 한계.** step 3 independence 가정은 false이며 저자들은 qualitative agreement만 준다고 한다(§2 footnote 1). “nothing in this paper should be interpreted as claiming that large language models cannot display emergent abilities”라고 쓴다(§7). [[emergent-abilities]]는 long target partial-credit argument가 적용되지 않는 classification task에서도 emergence가 나타난다고 지적한다(§5.1).

**범용 모델 함의.** small-scale pretraining ablation에서 partial-credit metric(token edit distance, target log-probability)과 큰 evaluation set은 exact match near zero task에서 signal을 얻는 별개 방법 둘이다. 다음 절은 continuous metric도 threshold를 없애지 못하는 task를 보인다.

## §7 Loss-threshold 관점과 downstream accuracy 예측이 계속 어려운 이유

**정의.** [[emergence-loss-perspective]](Du et al.; arXiv 2024-03)는 corpus/tokenizer/architecture를 고정하고, 낮은 pretraining loss model에는 있고 높은 loss model에는 없는 ability를 emergent ability로 다시 정의한다(§4).

**수식(§4, Eqs. 3-5).**

```
normalized_performance(L) = f(L) if L < η, else 0
L(N) = L_∞ + (N_0 / N)^{α_N}
performance > 0  only if  N ≥ N_0 · (η − L_∞)^{−1/α_N}
```

L은 pretraining loss, η는 threshold, f는 monotonically decreasing function(random guessing은 0으로 map), N은 fixed token count model size, L_∞는 irreducible loss, N_0/α_N은 fitted constant다. 논문은 fitted constant 값을 보고하지 않는다.

**증거.** English:Chinese 4:1 corpus 하나와 65k SentencePiece vocabulary([[emergence-loss-perspective-recipe]])의 300M-32B model 30개 초과:
- task 12개 중 8개(TriviaQA, HellaSwag, RACE, WinoGrande, Chinese task 4개)는 시작부터 낮은 loss와 함께 performance가 개선한다. performance/loss Spearman rank correlation은 −0.947~−0.996이고 서로 다른 size checkpoint가 performance-versus-loss curve 하나에 놓인다(Table 2, Fig. 1).
- MMLU, C-Eval, GSM8K, GSM8K-Chinese는 loss 약 2.2까지 random level에 머문다(§3.1).
- MMLU/C-Eval은 CorrectChoiceProb(correct answer predicted probability), Brier score continuous metric 둘에서도 threshold가 남는다. 이것이 metric-artifact account에 대한 논문 test다(§3.2, Fig. 4). GSM8K/GSM8K-Chinese는 continuous metric으로 다시 score하지 않았다.

**계산 예(Brier score, §3.2/App. C).** option 4개 question 하나에서 `Brier = Σ_j (y_j − ŷ_j)²`이며 y는 one-hot correct answer, ŷ는 predicted probability다. uniform prediction(각 0.25)은 `(1 − 0.25)² + 3 × 0.25² = 0.5625 + 0.1875 = 0.75`다. 항상 (1, 0, 0, 0)을 예측하면 option 1 정답이면 0, 아니면 1 + 1 = 2다. 정답이 option 넷에 균등하면 `0.75 × 2 = 1.5`다. 두 predictor 모두 question을 무시한다. 따라서 model은 task를 배우지 않고 Brier score를 1.5에서 0.75 쪽으로 낮출 수 있다. Du et al.은 question을 무시하는 predictor best score 0.75를 random-guess level로 보고 loss가 threshold에 닿을 때까지 Brier가 이를 넘지 않는다고 한다(§3.2, App. C).

**loss보다 accuracy 예측이 어려운 이유.** [[predicting-downstream-elusive]](Schaeffer et al.; arXiv 2024-06)는 multiple-choice score를 correct choice negative log-likelihood → vocabulary probability → available choice로 renormalized probability → Accuracy/Brier score의 네 step으로 추적한다(§4). ARC-Challenge에서 log-likelihood는 sample 약 90%가 per-sample score-compute Spearman correlation 0.75 초과, choice renormalization 뒤 40%, Accuracy는 더 적다(§4, Fig. 3). mechanism은 incorrect choice mass다. correct choice probability 0.4, 나머지 0.6이 0.2/0.2/0.2이면 Accuracy 1이다. 0.6이 incorrect choice 하나에 있으면 Accuracy 0이다(§5). correct-choice probability forecast 하나는 두 경우를 분리하지 못한다. 이 연구에서 Brier score도 predictability를 회복하지 못했다(§2, Fig. 3C).

**세 관점 비교(이 course의 해석).**

| 관점 | Horizontal axis | 예측하는 것 | 증거 한계 |
|---|---|---|---|
| Scale emergence([[emergent-abilities]]) | FLOPs 또는 parameter | scale 아래 absent, 위 present ability | emergence scale은 data/training과 함께 바뀜; forecast method 아님 |
| Metric artifact([[emergent-abilities-mirage]]) | Parameter | continuous metric의 smooth change | GPT-3 arithmetic, LaMDA BIG-Bench, vision toy에서 시험 |
| Loss threshold([[emergence-loss-perspective]]) | 같은 corpus/tokenizer의 pretraining loss | loss < η 전 chance task; MMLU/C-Eval은 continuous metric도 같음 | corpus/tokenizer 하나; AdamW만; routed/non-Transformer 없음(§7) |

한 해석에서는 관점들이 compatible하다(**해석**, 이 course). loss는 smooth하게 개선되고 [[emergent-abilities]] App. A.1/[[emergent-abilities-mirage]] §3이 본 task에서는 continuous per-item score가 thresholded score보다 먼저 움직였으며, 일부 task는 어떤 score도 chance를 떠나기 전 loss level을 요구한다. [[scaling-laws-unreliable-downstream]]는 연구 하나의 base rate를 더한다. Gadre et al.(2025) task 46개 중 18개(39%)가 validation loss와 predictably scale한다(Fig. 1).

**조건과 한계.** Du et al.은 loss value가 tokenizer/corpus 간 비교 불가, 다른 architecture/optimizer 미시험, 더 큰 scale 새 tipping point 비보장을 말한다(§7). 약 2.2 fitted threshold는 English:Chinese 4:1 corpus/65k tokenizer에 속한다. Schaeffer et al.(2024)은 multiple-choice benchmark만의 model family 다섯 개를 연구하고 per-choice forecast가 되는지는 시험하지 않는다([[predicting-downstream-elusive]] §6, App. B). 상태: 논문 각각 **단일 연구 결과**다.

**범용 모델 함의.** small-model ablation으로 pretraining data를 고를 때 task가 ablation model 어느 것도 넘지 못하는 loss threshold를 가지면 Du et al. setting에서 그 task는 accuracy/CorrectChoiceProb/Brier로 option을 rank할 수 없다. training 시작부터 개선하는 task(Du et al. 12개 중 8개)는 accuracy 또는 continuous score로 option을 rank할 수 있다.

## §8 같은 pretraining loss, 다른 downstream 결과

**정의.** *saturation regime*은 model predicted conditional distribution이 true distribution과 같아 loss가 data entropy와 같은 상태다([[same-loss-better-downstream]] §3). *Implicit bias*는 같은 loss parameter setting 중 일부 setting을 다른 것보다 선호하는 optimizer 성질이다.

**문제.** pretraining loss는 checkpoint/model size rank에 쓴다. model 둘이 minimal loss에 도달해도 transfer가 다르면 loss는 rank하지 못한다.

**메커니즘.**
1. data는 known model(PCFG, HMM, OPT-125M sample)에서 생성해 minimal loss를 계산할 수 있다: 3.196, 3.758, 1.865(§3). pretraining objective는 masked-language modeling이고 transfer는 fine-tuning 또는 frozen representation 위 linear classifier인 *linear probe*로 잰다(§3).
2. convergence 뒤 training 지속, model size 증가, adversarial pretraining이라는 intervention 셋은 loss를 minimum에 두고 transfer를 바꾼다(§3).
3. theory: global minimizer에서 stochastic-gradient covariance는 Hessian ∇²L(loss second derivative matrix)과 같다. global minimizer set Γ 근처에서 learning rate η(§7 threshold 아님), batch B의 SGD는 small η, 약 1/η² step limit에서 coefficient 1/(4B)로 Tr[∇²L]를 낮추는 방향으로 Γ를 따라 drift한다(§4, Theorem 4.3).
4. flatness는 Hessian trace로 재며, `∇_θ log f_θ(x_−t)_{x_t}` expected squared norm으로 estimate한다. x는 sentence, t는 mask position, x_−t는 t를 mask한 sentence, f_θ는 그 position predicted probability vector, x_t는 prediction에서 sample한 값이다(§5, App. B.1).

**계산 예(logit space estimator; 이 course의 도출).** softmax class probability p에서 logit z에 대한 log p_y gradient는 sampled class y one-hot vector e_y에서 p를 뺀 `e_y − p`다. p = (0.5, 0.3, 0.2)일 때:
- y = 1(probability 0.5): ‖(0.5, −0.3, −0.2)‖² = 0.25 + 0.09 + 0.04 = 0.38
- y = 2(0.3): ‖(−0.5, 0.7, −0.2)‖² = 0.25 + 0.49 + 0.04 = 0.78
- y = 3(0.2): ‖(−0.5, −0.3, 0.8)‖² = 0.25 + 0.09 + 0.64 = 0.98
- expected value: 0.5 × 0.38 + 0.3 × 0.78 + 0.2 × 0.98 = 0.62 = 1 − Σ p_j².
논문 estimator는 이 logit-space vector에 network Jacobian을 곱한 all-parameter gradient에 같은 sampling을 적용한다.

**증거.**
- Algorithm(235M, PCFG; Table 1): AdamW loss 3.204, task A 89.9%/task B 49.2%; adversarial pretraining 3.206, 83.1%/42.3%; true conditional lookup table optimal 3.196, 71.2% / 39.7%.
- Size: 같은 loss에서 scale up은 linear-probe accuracy를 PCFG 6.9%, HMM 4.5%, OPT-generated data 2.0% 개선한다(§3, Fig. 2). PCFG에서 larger model Hessian trace는 19.8 → 12.6, task-B probe accuracy는 40.4% → 50.5%다(§5, Fig. 5).
- convergence 뒤 training(SGD, 12% warmup/fixed LR 1e-3): loss 불변인데 trace는 떨어지고 accuracy는 235M PCFG task C 1.6%, 67M HMM task-10 4.0% 오른다(§5, Fig. 3, App. B.2).
상태: **단일 연구 결과**다.

**조건과 한계.** model은 autoregressive가 아닌 masked-language model(BERT-style transformer/LSTM)이다. PCFG/HMM data는 pretraining/downstream distribution이 같다. real large-scale data saturation 도달은 computationally challenging하다고 설명한다(§3, §7). Du et al.은 같은 pretraining loss model이 fine-tuning 뒤 다를 수 있다는 관련 관측을 인용한다([[emergence-loss-perspective]] §5).

**범용 모델 함의.** validation loss를 그대로 두는 token-budget/optimizer change가 transfer에 neutral이라고 입증된 것은 아니다. ch-08a §8(한 model family에서 pretraining token이 늘며 post-trained quality 하락)은 같은 주의의 별도 근거다.

## §9 Autoregression의 잔불: next-token training이 poor generalization을 예측하는 곳

**정의.** [[embers-of-autoregression]](McCoy et al.; arXiv 2023-09)은 Internet text next-word prediction training problem에서 세 sensitivity를 예측한다(§2.2).
1. *Task probability*: 같은 complexity rare task는 frequent task보다 accuracy가 낮다.
2. *Output probability*: deterministic task도 correct output이 low-probability string이면 accuracy가 낮다.
3. *Input probability*: input도 같으며 output보다 effect가 작다.

**메커니즘(§3.3).** model task는 `P(output | input) ∝ P(input | output) · P(output)` maximum output 선택으로 쓸 수 있다. deterministic task는 output 하나만 nonzero P(input | output)를 갖는다. P(input | output)를 불완전하게 estimate하는 model은 candidate 여러 개에 nonzero likelihood를 주며 prior P(output)가 choice에 영향을 준다. 저자들은 이는 computational-level description일 뿐 model이 likelihood/prior를 계산한다는 주장이 아니라고 명시한다(§3.3 footnote 3).

**계산 예(이 course가 고른 수).** correct decoding이 low-probability sentence "Well, if they don't code, so be it."인 rot-13 input을 보자. model likelihood estimate가 correct에 0.6, regularized "Well, if they don't come, so be it."에 0.3이고 prior relative probability가 0.01/0.05면 product는 0.6 × 0.01 = 0.006, 0.3 × 0.05 = 0.015다. model은 regularized sentence를 낸다. 논문은 바로 이 one-word change를 시험하고 GPT-3.5/GPT-4가 correct보다 regularized version을 더 자주 냈다고 보고한다(§6.1, Fig. 6.2).

**증거**(gpt-3.5-turbo-0613/gpt-4-0613, temperature 0; high-probability sentence 100개와 derived medium/low-probability set; §4). *shift cipher* rot-k는 각 letter를 alphabet에서 k position 뒤 letter로 바꾼다(§3.1).
- Task probability: shift 1-25 decode에서 GPT-4는 rot-1 0.82, rot-3 0.76, rot-2 0.02이고 GPT-3.5는 rot-13 0.21, 다른 모든 shift 0.00이다(§5.1). C4에서 1/3/13 shift가 가장 흔하다(§5.1, Fig. 5.2).
- Output probability: GPT-4 rot-13 decode accuracy는 high-probability output 0.51, low-probability 0.13이며 두 model logistic regression은 p < 10⁻⁴다(§6.1). GPT-4 word-order reversal은 97% 대 53%다(Table 1).
- Input probability: GPT-4 rot-13 encoding은 high-probability input 21%, low-probability 11%다(Table 1). shift-cipher/reversal/Pig Latin/acronym encoding test에서 logistic regression은 GPT-4 shift cipher(p < 0.05), GPT-3.5 acronym(p < 10⁻⁵)에서만 significant input-probability effect를 찾는다(§7.1).
- common/rare task variant 7쌍 전체에서 model은 common variant가 더 좋다(§5.6).
- chain-of-thought/step-by-step prompt는 decoding accuracy를 높이고 encoding accuracy는 낮추는 듯하지만 shift-level pattern/output-probability effect는 남는다(§9.4, Fig. 9.1).
상태: **단일 연구 결과**다. task-probability effect는 저자가 인용하는 contemporaneous counterfactual-task work와 맞는다(§5.6).

**조건과 한계.** tested model은 architecture/training data가 공개되지 않은 proprietary chat model이다(Limitations). 이런 model은 instruction tuning도 거치지만 논문은 이를 분석하지 않는다(§2.1). base(pretrained-only) model은 시험하지 않는다. task는 단순하고 저자 말로 substantial practical utility가 없으며 practical task로 effect 전이는 future work다(Limitations).

**범용 모델 함의.** common format/high-probability answer만으로 측정한 deterministic skill은 generality를 과장할 수 있다. low-probability task variant는 model이 procedure를 적용하는지 frequent string을 재생하는지 probe한다. ch-47a는 이를 바탕으로 counterfactual/perturbed evaluation을 만든다.

## §10 Data decision의 함의: breadth 대 benchmark-shaped data

pretraining distribution이 transfer에 미치는 finding 넷이다. item 1/2는 §1/§3/§9에서, item 3/4는 source 둘을 더한다. 각각 evidence/scope와 함께 제시한다.

1. **Demonstration을 통한 task coverage.** GPT-2 저자는 text의 demonstration에서 task가 학습된다고 제안한다(§1). language detector가 WebText에서 French text 10 MB만 찾았는데도 GPT-2는 French→English BLEU 11.5에 도달하며(§1), rot-2처럼 text에서 rare한 task는 GPT-4에서도 near zero에 머문다(§9). 상태: 단일 연구 둘을 합친 **해석**이며, 어느 연구도 base model 하나에서 task frequency/accuracy를 측정하지 않는다.
2. **Distributional shape.** Omniglot setting에서 ICL에는 burstiness와 rare class long tail이 필요하다(§3). 상태: text corpus에서 시험하지 않은 **단일 연구 결과**.
3. **추출 가능한 지식의 phrasing 다양성.** synthetic-biography experiment에서 person당 서로 다른 template biography 1,000개는 fixed biography 하나를 1,000회 pass한 것보다 knowledge capacity가 약간 높고, low-diversity data에서 memorized knowledge는 거의 0% extractable이다([[physics-of-lm-3]] Result 2, App. A.3, Part 3.1 인용). 상태: GPT-2-scale model의 **단일 연구 결과**.
4. **Evaluation과 similarity.** task format 하나를 겨냥한 filtered/synthetic code로 학습한 phi-1은 exercise set close match가 있는 HumanEval 71개 중 81.7%, 없는 93개 중 26.9%를 푼다. StarCoder-Prompted는 57.7%, 29.0%다([[phi-textbooks]] Table 3, τ = 0.95이며 τ는 close match AST match-rate threshold). similar-minus-non-similar gap은 phi-1 54.8점, StarCoder-Prompted 28.7점(도출)이다. exercise set으로 학습하지 않은 StarCoder-Prompted도 similar subset에서 더 높으므로 gap 일부는 더 쉬운 problem을 반영한다. close match를 제거해 재학습한 phi-1은 74.6%, 32.3%(gap 42.3, 도출), 같은 total 50.6%다. 저자들은 pruning result를 phi-1 score가 contamination으로 설명되지 않는 증거로 읽는다(§5.2). 상태: **단일 연구 결과**이며, 더 큰 gap을 benchmark format 쪽 narrowing으로 읽는 것은 이 course의 **해석**이다.

**권고.** data filter/synthetic source를 benchmark effect로 고르면 similarity-split evaluation(새 data close match 유무 item)을 쓰고, 그 data로 학습하지 않은 model과 gap을 비교하며, low-probability/counterfactual task variant를 더하라. phi-1의 matched/unmatched HumanEval gap은 54.8점이고 StarCoder-Prompted는 28.7점(item 4)이며 total score 하나는 그 차이를 보이지 않기 때문이다. benchmark target 없이 data를 선택하는 경우(GPT-2 접근, §1)는 ch-00 development/unseen suite를 사용하라. ch-09/ch-10a는 이를 real pretraining mixture와 classifier-based selection에 적용한다.

## 레시피

아래 행은 이 장 인용 run의 context length, data diversity, optimization setting이다. GPT-2는 training setting 대부분을 생략해 not reported로 적었다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| GPT-2(WebText LM) | 117M, 345M, 762M, 1542M | pretrain-stable | layer; d_model | 12, 24, 36, 48; 768, 1024, 1280, 1600 | GPT-2 report(OpenAI PDF, 2019) Table 2 | verified 2026-09-15 | size “approximately log-uniformly spaced”(§3); ablation 없음 |
| GPT-2 | 넷 모두 | pretrain-stable | context length; batch size; vocabulary | 1024 token; 512(unit 미기재); byte-level BPE token 50,257개 | §2.3, §2.2 | verified 2026-09-15 | GPT 512에서 context를 올림(§2.3); ablation 없음 |
| GPT-2 | 넷 모두 | pretrain-stable | learning rate | 5% held-out WebText sample best perplexity용 model별 tune; 값 미인쇄 | §3 | not reported(§1-§7, Tables 1-17, sample만 있는 Appendix A 확인) | model별 held-out perplexity(§3) |
| GPT-2 | 넷 모두 | pretrain-stable | optimizer; schedule; token/step seen; compute | not reported | §1-§7, Tables 1-17, Appendix A 확인 | not reported | — |
| GPT-2 | 넷 모두 | pretrain-stable | data source/filtering | WebText: ≥3 karma Reddit outbound link(45M), 2017년 12월 전 link, deduplicate/heuristic clean해 8M 초과 document / 40 GB; Wikipedia/non-English page 제거 | §2.1, §3.7 | verified 2026-09-15 | quality로 Common Crawl 거부; ablation 없음 |
| GPT-3 | 125M-175B(8 size) | pretrain-stable | token seen; context length; packing | 모든 size 300B token; 2,048 token; document pack/end-of-text token 분리, cross-document masking 없음 | arXiv:2005.14165v4 Table 2.1, §2.1, App. B | verified 2026-09-15 | ablation 없음 |
| GPT-3 175B | 175B | pretrain-stable | batch(token); peak LR | 3.2M; 0.6 × 10⁻⁴ | Table 2.1 | verified 2026-09-15 | measured gradient noise scale로 batch guide(§2.3) |
| GPT-3 Small | 125M | pretrain-stable | batch(token); peak LR | 0.5M; 6.0 × 10⁻⁴ | Table 2.1 | verified 2026-09-15 | 위와 같음 |
| GPT-3 | all size | pretrain-stable / decay | optimizer; clip; weight decay; warmup; decay; batch ramp | Adam β1=0.9, β2=0.95, ε=10⁻⁸; global-norm clip 1.0; weight decay 0.1; 375M token linear warmup; 260B token 동안 peak 10%까지 cosine, 이후 10% constant; size별 first 4-12B token에서 32k token부터 linear batch ramp | App. B | verified 2026-09-15 | ablation 없음 |
| GPT-3 | all size | pretrain-stable | mixture: token; sampling weight; 300B epoch | filtered Common Crawl 410B, 60%, 0.44; WebText2 19B, 22%, 2.9; Books1 12B, 8%, 1.9; Books2 55B, 8%, 0.43; Wikipedia 3B, 3%, 3.4 | Table 2.2 | verified 2026-09-15 | “datasets we view as higher-quality are sampled more frequently”(§2.2); ablation 없음 |
| GPT-3 | all size | pretrain-stable | weight × 300B / token으로 재계산한 epoch | Common Crawl 0.44; WebText2 3.47; Books1 2.00; Books2 0.44; Wikipedia 3.00 | Table 2.2 도출 | derived(formula: weight × 300B / token; Table 2.2 input). WebText2 printed epoch 2.9/Wikipedia 3.4와 충돌하며 report는 설명 안 함. Table 2.2 weight는 token 아닌 training example fraction이므로 token-share formula는 이 course 가정 | ablation 없음 |
| GPT-3 | all size | eval-gate | shot; decoding | K=0부터 2,048 context까지, 보통 10-100, 가능하면 development set에서 선택; free-form task beam width 4, length penalty 0.6 | §2, §2.4 | verified 2026-09-15 | larger K “usually but not always better”(§2.4) |
| Chan et al. Omniglot transformer(image-label sequence task, release 없음) | 12 layer; 831,479 parameter(App. C.1 count) | pretrain-stable | architecture; step; optimizer; schedule; seed | embedding 64, head 8; 500k step; Adam, 4,000 step에 3 × 10⁻⁴ linear warmup 뒤 inverse square root decay; run 5개(Figs. 5-6은 3개) | arXiv:2205.05055v6 App. A, App. C.1 | verified 2026-09-15 | ablation 없음 |
| Chan et al. Omniglot transformer | 위와 같음 | pretrain-stable | data distribution | p(bursty)=0.9; class 1,600 또는 12,800; Zipf α=1 | §3.1-3.2 | verified 2026-09-15 | tested exponent 0/0.5/1/1.5/2/3 중 top 10 common class ICL/IWL 모두 높은 α=1 식별(Fig. 6) |
| Olsson et al. small model(release 없음) | 1-6 layer | pretrain-stable | context; width; step; warmup; weight decay | 8,192 token; d_model 768, head 12; 10,000 step(약 10B token); first 1.5 × 10⁹ token warmup; step 4,750(약 5B token)에 weight decay 감소 | arXiv:2209.11895v1 Model Details | verified 2026-09-15 | ablation 없음; Model Details는 weight-decay change가 phase change 뒤이고 range 안 scheduled change는 warmup뿐이라고 함 |

Du et al. run의 sequence length 2048/AdamW (0.9, 0.95)를 포함한 recipe 값은 [[emergence-loss-perspective-recipe]]에 있다.

**작은 범용 실행의 시작점.** verified row만으로: 2,048-token context에 document를 pack하고 end-of-text token으로 분리하라. Adam β1=0.9, β2=0.95, ε=10⁻⁸, gradient clip 1.0, weight decay 0.1, 375M token linear warmup, peak 10%까지 cosine decay를 쓴다(GPT-3: 125M~175B all 8 size, 300B token, V100 cluster). peak LR/batch는 model size와 가장 가까운 GPT-3 Table 2.1 row를 읽어라. 125M은 0.5M-token batch에 6.0 × 10⁻⁴다. source가 quality가 다르다고 판단하면 GPT-3는 WebText2/Books1/Wikipedia를 300B token에서 1.9~3.4회(printed Table 2.2 epoch) sample하고 filtered Common Crawl/Books2는 1회 미만(0.44/0.43) sample했다. base-model evaluation은 GPT-3처럼 development split에서 K를 고른 zero/few-shot을 보고하라. 이 값들은 인용 report에서 ablation으로 선택되지 않았다.

## 일반화 관점

**(a) 이 단계에서 무엇이 폭을 넓히는가.**
- broad task-agnostic corpus: WebText에서 reading comprehension/translation/summarization/QA로 zero-shot transfer하며 size와 log-linear 개선([[gpt-2-unsupervised-multitask]] Abstract, Fig. 1).
- demonstration in-context use의 scale: 42 benchmark에서 zero-to-few-shot gap이 model size와 함께 증가([[gpt-3-few-shot]] Fig. 1.3).
- Omniglot setting의 many rare class bursty data와 α = 1 근처 Zipfian marginal([[icl-data-distributional-properties]] §3).
- question으로 extract해야 하는 knowledge의 phrasing diversity([[physics-of-lm-3]] App. A.3).
- simplified MLM setting에서 equal-loss model 중 larger model과 flatter solution 쪽 longer training([[same-loss-better-downstream]] §3, §5).

**(b) 무엇이 좁아짐 또는 false breadth claim을 만드는가.**
- target benchmark 같은 shape data: phi-1 HumanEval similar/non-similar 81.7%/26.9%, phi-1 exercise로 학습 안 한 StarCoder-Prompted 57.7%/29.0%([[phi-textbooks]] Table 3; gap 차이를 narrowing으로 읽는 것은 §10 item 4 **해석**).
- uniform class balance: uniform class frequency에서 ICL/IWL 둘을 유지한 model이 없고 continued training으로 ICL은 fade할 수 있음([[icl-data-distributional-properties]] §3).
- rare task variant/low-probability output: rot-2 0.02 대 rot-1 0.82, output probability로 0.51 대 0.13([[embers-of-autoregression]] §5.1, §6.1).
- few-shot result를 부풀리는 contamination: GPT-3 PIQA/Winograd flagged([[gpt-3-few-shot]] §4).
- same loss의 다른 training algorithm: transfer 줄이도록 만든 adversarial pretraining은 task B 42.3%, loss 3.206이고 AdamW는 loss 3.204에서 49.2%(235M, PCFG; [[same-loss-better-downstream]] Table 1).

**(c) 이 단계에서 어떻게 측정하는가.**
- 학습 전반 token index ICL score(early position loss − late position loss), K에 따른 ICL curve([[induction-heads]] Key Concepts; [[gpt-3-few-shot]] Fig. 1.2).
- 모든 thresholded metric 옆 continuous per-item score와 `(1 − expected accuracy)^n`이 작을 만큼 큰 evaluation set(§6).
- corpus/tokenizer가 같은 checkpoint의 performance 대 pretraining loss plot, task가 여전히 random baseline인지 점검([[emergence-loss-perspective]] §3).
- benchmark Accuracy extrapolation 대신 incorrect choice probability mass sample별 추적. 논문은 이를 제안하지만 working forecast는 보이지 않는다([[predicting-downstream-elusive]] §6, App. B).
- data 선택에 쓰기 전 exact evaluation setup에서 downstream scaling이 predictable한 task별 점검([[scaling-laws-unreliable-downstream]] §7).
- rare variant, low-output-probability, similarity-split evaluation(§9, §10).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| small model chance task extrapolate | random baseline flat curve 뒤 largest run jump | 같은 output의 correct answer log-probability/token edit distance plot; 이 corpus task threshold와 loss 비교 |
| small evaluation set으로 ability absent 선언 | small scale correct item 0개 | `(1 − expected accuracy)^n` 계산; zero score가 unlikely할 때까지 set 확대 |
| tokenizer/corpus 간 pretraining loss 비교 | family 둘에서 “lower loss but worse downstream” | corpus/tokenizer가 같은 checkpoint 사이에서만 loss 비교([[emergence-loss-perspective]] §7) |
| equal validation loss를 equal capability로 취급 | 같은 loss run 둘이 probe/fine-tuning 뒤 다름 | 둘 다 downstream probe; research 비교면 flatness estimate 추가 |
| benchmark Accuracy 자체 trend로 forecast | task에 따라 forecast error 큼/작음 | sample별 forecast; incorrect choice probability mass 추적 |
| predictability 회복을 위해 Brier score 전환 | Brier trend도 compute와 weak correlation | Brier는 incorrect-choice mass 의존([[predicting-downstream-elusive]] §4); working forecast 없이 논문이 제안한 per-choice probability 추적(§6) |
| few-shot gain을 new task learning으로 읽음 | web text에 나타나는 task gain | text에 없을 법한 artificial task(GPT-3 word manipulation)/contamination check 추가 |
| frequent task variant만 평가 | rot-13 같은 format high score | rare variant(rot-2)/low-probability target output 추가 |
| benchmark gain으로 data 선택 | new data와 similar benchmark item에 gain 집중 | added data에 대한 benchmark similarity split |
| larger K가 항상 도움이라 가정 | 일부 task에서 few-shot이 one/zero-shot 아래 | development split에서 K 선택, 쓴 K 보고 |

## 이해 점검

1. GPT-2는 language-modeling objective가 text sequence로 쓴 supervised objective와 같은 global minimum을 가진다고 주장한다. 실제로 그 minimum에 도달하는 것이 corpus에 어떤 task가 나타나는지에 왜 여전히 의존하는지 French translation 결과로 설명하라.
2. GPT-3 Table 3.9에서 3-digit addition은 zero-shot 34.2에서 few-shot 80.4로 오른다. 이 gain을 만드는 서로 다른 mechanism 둘과, 논문의 어떤 evidence가 이를 구별하는지 설명하라.
3. Chan et al.은 uniform class frequency가 ICL/IWL 중 하나만 주고 α = 1은 둘을 준다고 찾는다. 각 class가 context 안과 training 전반에 얼마나 자주 나타나는지로 skew가 둘을 허용하는 이유를 설명하라.
4. induction-head mechanism이 standard transformer에서 최소 attention layer 둘을 필요로 하는 이유와 smeared-key change가 왜 그 requirement를 없애는지 설명하라. 그 change 결과는 phase change 원인에 무엇을 함의하는가?
5. §6 계산 예로 같은 model output에서 token edit distance/exact match가 다른 그림을 주는 이유와 100-item test set이 apparent jump를 강하게 만드는 이유를 설명하라.
6. Du et al.은 MMLU/C-Eval에서 Brier score 아래도 약 2.2 loss threshold가 남는다고 찾는다. 이것이 GPT-3 arithmetic task metric-artifact account와 모순되지 않는 이유를 설명하라.
7. validation loss가 같은 checkpoint 둘 중 하나가 더 긴 training 뒤 것이다. 더 긴 training checkpoint가 더 잘 transfer할 수 있는 SGD-noise 논증과 이 evidence가 autoregressive LLM pretraining에는 아직 확립하지 못하는 이유를 설명하라.
8. GPT-4는 rot-1 0.82, rot-2 0.02를 decode한다. `P(input | output) · P(output)` decomposition으로 설명하고 base model이 procedure를 배웠는지 평가하는 함의를 말하라.

## 연결

- **이전(array 순서):** ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability(dependency; parameter/token 함수로서 pretraining loss와 downstream prediction 문제).
- **다음(array 순서):** ch-09 — Pretraining Data Composition and Capability Coverage(이 장에 의존; §10을 real mixture에 적용).
- **관련 later chapter:** ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection(§10 similarity check); ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining(§10 item 3); ch-29e — Instruction Tuning and Generalization to Unseen Tasks(instruction following을 emergent augmentation으로, §5); ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation(loss/perplexity long-context proxy, §4); ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation(rare/counterfactual variant, §9); ch-48 — Contamination Detection and Its Effect on Reported Scores(§2 contamination); ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions(§6 resolution).
- **측정 계약:** ch-00 — What General Capability Means and How It Is Measured.

## 출처

- [[gpt-2-unsupervised-multitask]] — Radford et al., OpenAI report, 2019: p(output | input, task) framing, WebText 구성, zero-shot 결과(CoQA, LAMBADA, summarization hint, translation, Natural Questions), model 표, 미보고 training setting(chapter excerpt; library card 없음).
- [[gpt-3-few-shot]] — Brown et al., arXiv 2020-05: zero/one/few-shot ICL 정의, evaluation protocol, arithmetic/word-manipulation 표, limitation, contamination analysis, Table 2.1-2.2/App. B recipe(chapter excerpt; library card 없음).
- [[icl-data-distributional-properties]] — Chan et al., arXiv 2022-05: burstiness, class count, dynamic meaning, Zipfian skew, transformer 대 RNN 결과, Omniglot training setup(chapter excerpt; library card 없음).
- [[induction-heads]] — Olsson et al., Transformer Circuits Thread 2022-03(arXiv 2022-09): induction-head 정의, ICL score, phase-change window, argument 여섯 개와 evidence strength, model detail(chapter excerpt; library card 없음).
- [[emergent-abilities]] — Wei et al., arXiv 2022-06(TMLR 2022-08): emergence 정의, Table 1 scale, BIG-Bench classification, cross-entropy analysis, PaLM 62B task(chapter excerpt; library card 없음).
- [[emergent-abilities-mirage]] — Schaeffer, Miranda, Koyejo, arXiv 2023-04: metric-artifact model, GPT-3 arithmetic re-scoring, BIG-Bench metric meta-analysis, induced emergence(chapter excerpt; library card 없음).
- [[emergence-loss-perspective]] — Du et al., arXiv 2024-03: same-loss-same-performance 결과, 약 2.2 loss threshold, Brier score baseline, Eqs. 3-5.
- [[emergence-loss-perspective-recipe]] — Du et al. run의 corpus/tokenizer/optimizer setting.
- [[same-loss-better-downstream]] — Liu, Xie, Li, Ma, arXiv 2022-10: saturation regime, equal-loss intervention, trace-of-Hessian theory/measurement.
- [[embers-of-autoregression]] — McCoy et al., arXiv 2023-09: task/output/input probability effect, shift-cipher/reversal 수치, prompting/scaling check(chapter excerpt; library card 없음).
- [[predicting-downstream-elusive]] — Schaeffer et al., arXiv 2024-06: log-likelihood에서 Accuracy까지 transformation sequence, ARC-Challenge correlation, incorrect-choice mechanism.
- [[scaling-laws-unreliable-downstream]] — Lourie, Hu, Cho, arXiv 2025-07: predictable loss-to-task scaling인 task 46개 중 18개.
- [[phi-textbooks]] — Gunasekar et al., arXiv 2023-06: benchmark-shaped training data similarity-split HumanEval 결과.
- [[physics-of-lm-3]] — Allen-Zhu and Li, Part 3.3, arXiv 2024-04: phrasing diversity와 저장 지식 extractability.
