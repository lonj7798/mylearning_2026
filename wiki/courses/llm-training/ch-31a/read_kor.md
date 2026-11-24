<!-- chapter: ch-31a
     track: sft
     kind: content
     title: 지도 학습에서의 음성 샘플: 수정, 실패 조건화, 비평, 비우도
     deps: [ch-31]
     sources: [[unlikelihood-training]], [[cringe-loss]], [[chain-of-hindsight]], [[lema-learning-from-mistakes]], [[physics-of-lm-2-2-learning-from-mistakes]], [[critique-fine-tuning]], [[dust-into-gold-negative-distillation]], [[learning-from-failure-nat]], [[orpo]], [[nft-negative-aware-finetuning]], [[redi-reinforcement-distillation]], [[negative-examples-likra]], [[pretraining-with-human-preferences]], [[likelihood-displacement]]
     figures: figures/negative-gradient-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 31a — 지도 학습에서의 음성 샘플: 수정, 실패 조건화, 비평, 비우도

> **핵심 통찰.** 실패 샘플은 지도 손실에 네 가지 방식으로 들어갈 수 있으며, 그중 샘플의 우도를 명시적으로 낮추는 한 가지 방식만 그 실패 샘플에서 확률 질량을 제거한다. 실패를 보통의 target으로 학습시키면 결과가 엇갈렸다. MATH에서 정답·오답 GPT-3.5 trace를 섞어 증류한 LLaMA-7B 학생은 3.03점을 얻었고, 정답 trace만 쓴 경우는 5.29점이었다([[dust-into-gold-negative-distillation]] Table 2). 반면 라벨을 붙이지 않은 GPT-3.5 실패는 LLaMA-2-Chat-7B agent의 수학 평균을 55.90에서 63.39로 올렸지만, tool-call 오류율도 3.58%에서 10.47%로 올렸다([[learning-from-failure-nat]] Tables 2 and 6). 수정된 target, critique, 라벨 조건화, 또는 제한된 음성 항으로 실패를 쓴 방식은 LEMA, CFT, NAT, NFT, REDI의 주 비교에서 positive-only arm을 이겼으며, 예외는 각 절에 적는다. 생성 모델 연구 중 그 기여를 분리해 측정한 NFT Qwen2.5-32B 실험에서는 향상의 80%가 positive에서, 20%가 negative에서 왔다([[nft-negative-aware-finetuning]] §5.3).
>
> **지침.** 실패에 더 강한 모델이 작성한 수정 또는 critique가 함께 있다면, 실패한 시도를 입력에 넣고 수정문을 target으로 학습시킨다. 이렇게 하면 같은 학습 token 수에서 음성 gradient 없이도 정확도가 올랐고([[lema-learning-from-mistakes]] Table 2), sequence 길이를 맞춘 비교에서도 그랬다([[critique-fine-tuning]] Table 10). 실패에 pass/fail 라벨만 있다면 conditioning으로 쓴다. 즉 프롬프트에 "incorrectly" 같은 라벨을 넣고 추론 때는 positive 라벨로 바꾸거나, positive log-likelihood 항 옆에 제한된 음성 항을 둔다(REDI α = 0.8, NFT ε = 1.0). 대칭적인 push-down이나 clipping 없는 push-down은 해당 연구들에서 collapse하거나 성능을 떨어뜨렸기 때문이다([[redi-reinforcement-distillation]] §4.3, [[nft-negative-aware-finetuning]] §5.4). positive보다 약한 생성기가 만든 실패라면 버린다. GPT-3.5 대신 fine-tuned LLaMA-2-7B가 만든 negative를 썼을 때 NAT 수학 평균은 positive-only 모델보다 각각 3.16점과 6.20점 낮았다([[learning-from-failure-nat]] Table 5).

## 범용 모델에서 이 장이 중요한 이유

[[ch-31]]은 positive-only self-training을 다룬다. rejection-sampling fine-tuning(RFT), STaR, ReST-EM은 verifier가 통과시킨 샘플만 남기고 나머지를 버린다.
버리는 비율은 생성 샘플의 절반을 넘을 수 있다. NAT agent 데이터에서는 약 7,000개 GSM8K 질문을 각각 세 번 샘플링해 약 9,000개 positive trajectory와 12,000개 negative trajectory를 얻었다. 따라서 생성 trajectory 중 12/21 = 57%가 실패였다([[learning-from-failure-nat]] App. B). 저자들은 계획이나 tool use가 필요한 과제에서는 버려지는 비율이 "60%를 넘을 수 있다"고 말한다(§1).

이 장은 SFT(supervised fine-tuning) 단계에서 그 실패가 무엇을 더할 수 있는지, 실패를 쓰는 각 방식이 출력 분포를 어떻게 바꾸는지를 묻는다.
범용 모델에서는 세 가지 이유에서 답이 중요하다. 첫째, 일부 방식은 entropy를 올리고 positive-only 학습이 entropy를 낮추는 상황에서도 pass@k(k개 샘플 중 적어도 하나가 정답일 확률)를 유지한다([[nft-negative-aware-finetuning]] Fig. 8; [[redi-reinforcement-distillation]] App. D.3). 둘째, 일부 방식은 학습하지 않은 일반 instruction following 또는 capability benchmark도 바꾼다([[critique-fine-tuning]] Table 8; [[chain-of-hindsight]] Table 7). 셋째, 잘못된 사용법은 모델 성능을 positive-only baseline보다 낮춘다(§5.6).

파이프라인 위치는 SFT와 distillation SFT다. skill이 어느 단계에서 이미 있어야 하는지를 보이는 두 pretraining 연구도 포함한다([[physics-of-lm-2-2-learning-from-mistakes]], [[pretraining-with-human-preferences]]). DPO의 rejected 항은 [[ch-39]]에서, policy-gradient RL의 negative advantage는 [[ch-43a]]에서 다룬다.

## §1 “negative”의 네 가지 뜻

**negative sample**은 verifier, judge, classifier, 또는 사람이 틀렸거나 바람직하지 않다고 라벨한 샘플이다. 이 과정은 네 가지 뜻을 엄밀히 구분하며, 이후 각 절은 자신이 어느 뜻을 논하는지 밝힌다.

| 뜻 | 손실에 들어가는 것 | 실패에서 확률 질량을 제거하는가? | 이 장의 예시 |
|---|---|---|---|
| (1) 음의 한계 가치(negative marginal value) | 아무것도 넣지 않는다. positive target으로 쓰면 성능을 낮출 샘플이어서 filter로 제거한다 | 아니오 | 오류가 50%를 넘는 WebInstruct 응답([[critique-fine-tuning]] §2.1) |
| (2) 내용으로서의 negative | 실패를 입력에 두거나, 수정 target 안에 넣거나, 실패로 학습하는 별도 module에 넣는다. 모든 항은 보통의 cross-entropy다 | 아니오 | LEMA, retry data, CFT, Dust-into-Gold의 negative LoRA, Likra의 negative head |
| (3) 조건화로서의 negative | "incorrectly" 같은 단어 또는 나쁜 품질 control token처럼 라벨 아래에서 failure를 target으로 둔다. 이 라벨은 추론 때 쓰지 않는다 | 아니오. 자기 라벨 아래에서 그 우도는 올라간다 | Chain of Hindsight, NAT, conditional pretraining |
| (4) gradient로서의 negative | 실패의 우도를 낮추는 항 | 예 | Unlikelihood, CRINGE, ORPO의 odds-ratio 항, NFT, REDI |

LoRA(low-rank adaptation)는 동결한 가중치에 더하는 저랭크 행렬만 학습한다. QLoRA는 quantized base model에 LoRA를 적용한다. §5.4에서 비교점으로 쓰는 GRPO와 DAPO는 RL 단계에서 가르치는 policy-gradient RL 알고리즘이다.

오직 뜻 (4)만 §2의 likelihood displacement와 concentration effect를 만들 수 있다. 뜻 (2), (3)도 실패 token 자체를 target으로 학습하면 해로울 수 있다. 작은 rank의 LoRA fine-tune에서 inline retry는 정확도를 낮췄고(§3.2), NAT의 라벨된 실패는 tool-call 오류율을 3.58%에서 7.90%로 올렸다(§4.2). 라벨이나 수정 없이 실패에 단순 maximum likelihood를 적용하면 Dust into Gold의 MIX(§3.4)와 5,000개 positive를 쓴 NAT 13B run에서는 정확도가 낮아졌지만, 다른 세 NAT 수학 설정에서는 높아졌다(§4.2). 그러므로 어떤 실패가 음의 한계 가치(뜻 1)를 갖는지는 설정에 따라 달라지며, positive-only arm과 비교해 측정해야 한다.


## §2 보통의 SFT가 target이 아닌 token에 하는 일, 그리고 push-down이 추가하는 것

**정의.** 모델은 vocabulary 전체에 대한 logit vector $z$를 출력한다. softmax $p_j = \exp(z_j) / \Sigma_i \exp(z_i)$는 logit을 다음 token 확률로 바꾼다. SFT는 target token $y$의 cross-entropy $L = -\log p_y$를 최소화한다.

**문제.** SFT는 특정한 오답 token을 명시하지 않는다. 정답만 학습할 때 그럴듯한 오답이 얼마나 많은 확률을 잃는지가 측정 가능한 질문이다.

**메커니즘과 수식.**

```
∂(−log p_y) / ∂z_j = p_j − 1[j = y]
∂(log p_k)  / ∂z_j = 1[j = k] − p_j
```

- $z_j$: token $j$의 logit, $p_j$: 그 확률, $1[\cdot]$: 조건이 참이면 1, 거짓이면 0, $y$: target token, $k$: negative token이다.
- $-\log p_y$에 gradient descent를 적용하면 모든 non-target logit은 $\eta\cdot p_j$만큼 낮아진다($\eta$는 step size). 따라서 확률이 가장 높은 오답 token이 가장 많이 낮아진다.
- $\log p_k$에 gradient descent를 적용하는 것($k$에 대한 push-down)은 $z_k$를 $\eta(1-p_k)$만큼 낮추고, 다른 모든 logit을 $\eta\cdot p_j$만큼 올린다.

**계산 예.** logit이 $(2, 1, 0, -1)$인 네 token A, B, C, D의 확률은 $p = (0.644, 0.237, 0.087, 0.032)$이다.

1. $\eta = 1$로 target B에 SFT step을 적용한다. gradient는 $(0.644, -0.763, 0.087, 0.032)$이고, 새 logit은 $(1.356, 1.763, -0.087, -1.032)$, 새 확률은 $p = (0.353, 0.531, 0.083, 0.032)$이다. A는 0.291을 잃고 C는 0.004를 잃었다. D의 logit은 0.032만 내려갔는데, 이는 normalizer가 내려간 양보다 작다. 따라서 D의 확률은 오히려 0.0004 올랐다. 정답 token 한 개에 대한 SFT step은 이미 낮은 확률의 오답을 억제하지 않는다.
2. $\eta = 1$로 D에 push-down을 적용한다. 새 logit은 $(2.644, 1.237, 0.087, -1.968)$이고, 새 확률은 $p = (0.751, 0.184, 0.058, 0.0075)$이다. D는 0.025를 잃고 A는 0.107을 얻으며, B에서 0.053, C에서 0.029의 확률 질량도 가져간다. 다른 모든 logit은 자기 확률에 비례해 올라가므로, 이미 낮은 확률의 token에 push-down을 적용하면 가장 가능성 높은 token과 나머지 token들의 격차가 커진다.
3. 대신 A에 push-down을 적용하면 새 확률은 $p = (0.513, 0.341, 0.108, 0.038)$이다. 확률 질량은 B, C, D로 퍼진다.

대화형 그림 [figures/negative-gradient-explorer.html](figures/negative-gradient-explorer.html)에서는 logit을 바꾸고 SFT, push-down, unlikelihood, CRINGE 중 하나를 고른 뒤 한 번 또는 여러 번 step을 적용할 수 있다. 그러면 제거된 확률 질량을 어느 token이 받는지 볼 수 있으며, 기본값은 위 수치를 재현한다.

**positive-only SFT가 그럴듯한 오답을 여전히 높은 확률로 둔다는 근거.**

- chosen HH-RLHF 응답으로 fine-tune한 OPT-350M에서는 rejected 응답의 log-probability도 chosen 응답의 log-probability와 함께 오른다([[orpo]] §3, Fig. 3). 결과(단일 연구).
- ARC-Challenge로 SFT한 Mistral-7B-v0.1에서는 학습 중 “incorrect answers do not seem to be sharply distinguished”, 즉 오답이 정답과 뚜렷하게 구분되지 않는 것으로 보인다([[negative-examples-likra]] §3.1, Fig. 2). 결과(단일 연구).

**조건과 한계.** 계산 예는 logit을 직접 갱신한다. 실제 network에서는 token과 응답이 parameter를 공유하므로, 한 응답에 대한 push-down이 비슷하지만 선호되는 응답도 낮출 수 있다. DPO refusal training에서는 이 때문에 Llama-3-8B-Instruct의 refusal rate가 74.4%에서 33.4%로 낮아졌다([[likelihood-displacement]] §6.2). 이 multi-token 분석은 [[ch-43a]]에서 다룬다.

**함의.** SFT는 모델이 이미 가능성이 높다고 평가한 오답 token에서 확률 질량을 제거한다. 명시적 negative 항은 특정 오답을 겨냥해 추가로 낮추며, 그 부작용은 그 negative가 이미 얼마나 높은 확률이었는지에 달려 있다.

## §3 내용으로서의 negative(뜻 2)

### 3.1 실수-수정 쌍: LEMA

**정의.** LEMA(Learning from Mistakes)는 chain-of-thought(CoT) 데이터에 더해 `(질문과 오답 풀이 → 수정)` 쌍으로 fine-tune한다([[lema-learning-from-mistakes]] §2).

**문제.** 정답 CoT 경로를 더 넣는 일은 어느 지점부터 도움이 멈춘다. LLaMA-2-70B는 CoT token 5.8M개에서 GSM8K 82.1을, 6.8M개에서 82.2를 기록했다(§4.1).

**메커니즘.**

1. 여러 모델(LLaMA-2, WizardMath, GPT-3.5-Turbo, GPT-4 등)에서 풀이를 sample하고, 최종 답이 틀린 것만 남긴다(Eq. 1).
2. GPT-4가 틀린 단계, 그 이유에 대한 설명, 수정된 풀이를 작성한다. 최종 답이 맞는 수정만 남긴다(Eq. 2).
3. 오답 풀이는 input에 넣고 학습한다. 이때 “output 부분의 loss만 back-propagation에 참여한다”(App. B.2).

**근거.** GSM8K 쌍 12,523개, MATH 쌍 6,306개, CSQA 쌍 7,241개를 사용했다(§3.2). LLaMA-2-70B는 GSM8K에서 81.4 → 83.5, MATH에서 23.6 → 25.0으로 올랐다. 자체 training data를 쓰지 않은 두 test set에서도 SVAMP는 80.3 → 81.6, ASDiv는 80.7 → 82.2였다(GSM8K training data는 사용함, §3.1; Table 1). 학습 token 수를 5.8M으로 같게 맞추면 LLaMA-2-70B의 CoT는 82.1, LEMA는 83.5였다(Table 2). example 수를 같게 맞추면 다섯 backbone 중 네 개에서 LEMA가 도움이 되었고, LLaMA-2-7B만 예외였다(Fig. 4). 결과(단일 연구).

**조건과 한계.** 수정은 GPT-4만 만들었다. 사람 검토에서 GPT-4 수정 50개 중 35개는 excellent, 4개는 poor로 평가된 반면, GPT-3.5-Turbo 수정 20개 중 거의 절반은 poor였다. 그래서 GPT-3.5-Turbo를 쓰지 않았고, 더 약한 corrector의 정확도 효과도 측정하지 않았다(§2.1, App. D.6). GPT-4는 자신이 MATH에서 낸 오류 2,696개 중 8.0%를 수정했다(App. D.6). 보고된 정확도는 test set에서 평가한 저장 checkpoint 20개(2,000 step 동안 100 step마다 하나) 중 최고 checkpoint의 정확도다(§3.3). 이는 두 arm 모두에 유리한 선택이다. 그래도 최고 checkpoint 세 개의 평균은 LLaMA-2-70B GSM8K에서 81.3 대 83.4였다(App. D.1, Table 5).

**함의.** 수정 데이터는 추가 positive에는 없는 신호를 담고 있지만, 각 gradient는 여전히 수정된 정답 출력의 likelihood를 올린다.

### 3.2 pretraining의 retry 데이터: Physics of Language Models 2.2

**정의.** retry 데이터는 틀린 풀이 단계 뒤에 특수 token `[BACK]`과 올바른 단계를 삽입한다([[physics-of-lm-2-2-learning-from-mistakes]] §4).

**메커니즘과 계산 예.** 각 문장에 틀린 parameter를 확률 $q$(retry rate)로 삽입하고, 두 번째 틀린 parameter는 $q^2$의 확률로, 계속 같은 방식으로 삽입한다. 따라서 step당 삽입 오류의 기대 개수는 $q/(1-q)$다. $q=0.2$이면 0.25개, $q=0.5$이면 1.0개가 된다(§4에서 도출). 모든 run은 training token 수를 같게 쓰므로, $q$가 높을수록 학습하는 문제 수는 줄어든다(§4).

**근거(synthetic iGSM 수학의 GPT2-12-12).**

- iGSM-med$^{op=23}_{qp}$에서 “retry rate = 0.5를 쓰면 accuracy가 78%에서 94%로 뛴다”고 보고했고, “실수를 masking할 필요는 없다”고 했다(Result 2-3).
- $q=0.2$에서는 모델이 문제 하나당 0.3회보다 적게 retry하고, $q=0.5$에서는 2~4회 retry한다. label masking은 이 횟수를 줄인다(Result 4).
- iGSM-med$^{op=23}_{pq}$에서 probe가 오류를 감지할 때 단계를 다시 생성하면 78% ⇒ 80%지만, retry pretraining은 78% ⇒ 95%였다(§4, Result 6).
- 오류 없는 pretraining model을 retry 데이터로 LoRA fine-tune하면 유의미한 향상이 없었다. “작은 LoRA rank에서는 fine-tuning이 오히려 해롭고 label masking이 중요해진다”(Result 7, Fig. 7).
- 생성하기 쉬운 가짜 실수, 즉 뒤의 풀이 단계를 삽입한 뒤 `[BACK]`으로 철회하는 `retry_weak`도 “모델의 accuracy를 유의미하게 높인다”(Result 8).

**조건과 한계.** synthetic 데이터와 작은 architecture 하나만 다뤘다. 저자들은 “여기서 쓴 synthetic data가 미래 LLM 구축에 직접 도움이 될 수 있다”고 주장하지 않는다(§1). 충분한 retry 데이터로 full fine-tuning하면 정확도는 올랐지만 retry pretraining의 두 배 training token을 썼고, 저자들은 이를 “본질적으로 continued pretraining”이라고 부른다(§5, Fig. 7 caption). 결과(단일 연구).

**함의.** 이 연구에서 오류 수정은 LoRA fine-tune이 아니라 pretraining 또는 full continued pretraining 중 습득됐다. mask 규칙은 §7을 보라.

### 3.3 critique target: Critique Fine-Tuning

**정의.** CFT는 $x$를 질문, $y$를 noisy response, $c$를 GPT-4o가 쓴 critique라고 할 때 $log P(c \mid [x; y])$를 최대화한다([[critique-fine-tuning]] §2.3).

**근거.** Qwen2.5-Math-7B와 WebInstruct example 50K개를 썼다. 응답의 약 56%가 정답으로 판정됐다. 여섯 수학 benchmark 평균은 raw response SFT 35.1, GPT-4o 검증 response SFT 40.4, GPT-4o answer SFT 50.4, CFT 57.1이었다(Table 2). 오류를 50% 넘게 담은 raw response를 SFT하면 Qwen2.5-7B는 37.4에서 20.2로 떨어졌는데, 이는 뜻 (1)의 사례다. IFEval instruction-level loose는 0.362(SFT 0.330), MT-Bench는 6.49(SFT 5.23)으로 올랐다(Table 8). CFT example을 SFT 데이터와 token 길이가 맞게 골랐을 때 CFT-Short는 55.2였고, GPT-4o answer SFT는 50.4였다(Table 10). 결과(단일 연구).

**조건과 한계.** 검사한 critique 50개 중 약 20%에는 부정확한 내용이 있었다(§4). CFT 50K개와 AceMath SFT example 50K개를 섞으면 54.8로, CFT 단독보다 낮았다(Table 9). AIME24는 문항이 30개이며, 저자들은 그 정확도가 “randomness의 영향을 크게 받는다”고 명시한다(§3.4). training seed는 보고하지 않았다.

**함의.** 오류를 input에만 둔 채 응답을 판정하는 법을 학습시키면, 이 7B model에서는 수학 정확도와 instruction-following 점수가 함께 올랐다.

### 3.4 distillation의 틀린 teacher trace: Dust into Gold

**정의.** distillation SFT는 teacher output으로 student를 학습한다. 표준 관행은 최종 답이 맞는 trace만 남기는 것이다([[dust-into-gold-negative-distillation]] Eq. 1).

**계산 예(complementarity).** 5,000문항 MATH test set에서 positive trace로 학습한 LLaMA-7B student는 253문제를 풀었고, negative trace로 학습한 student는 166문제를 풀었다. 둘 다 푼 문제는 29개였다. intersection over union IoU는 $29/(253+166-29)=29/390=0.074$다(Table 1).

**근거(7개 subject에 걸친 MATH accuracy 평균, greedy decoding, GPT-3.5-Turbo trace, Table 2).** original data fine-tune은 3.88, positive-only CoT distillation은 5.29, MIX(positive와 negative 모두에 대한 maximum likelihood)는 3.03, negative training NT(positive log-likelihood에서 negative log-likelihood의 $0.05\times$를 뺌, App. Eq. 15)는 4.48, negative의 sequence-level unlikelihood(weight 0.05, App. Eq. 16)는 4.96, Negative Assistant Training은 6.81이었다. 그 논문에서 NAT로 줄인 Negative Assistant Training은 §4.2의 NAT와 관계없다. 이는 negative에 ordinary likelihood(뜻 2)를 적용하는 별도 LoRA를 학습하고 동결한 뒤, $[-0.5,0.5]$로 제한한 attention weight를 통해 positive LoRA에 그 출력을 합친다. 따라서 negative module의 출력은 더하거나 뺄 수 있다(Eq. 4-5).

**조건과 한계.** 절대 정확도는 3~7%이며 seed와 variance는 보고하지 않았다. 결과(단일 연구).

## §4 conditioning으로서의 negative(뜻 3)

### 4.1 Chain of Hindsight

**정의.** Chain of Hindsight(CoH)는 사람의 preference rating으로 만든, 예를 들어 `How to explain neural networks to a child? Bad: {a subpar answer} Good: {an excellent answer}` 같은 sequence로 학습한다([[chain-of-hindsight]] §2).

**메커니즘.**

1. feedback token은 mask하고 model이 생성한 token을 예측한다. “inference 때 model generation을 방해하므로 다른 token에는 loss를 적용하지 않는다.” 학습 설명은 또 loss를 “마지막 model output sequence의 각 timestep에 걸쳐” 평균낸다고 쓴다. 따라서 앞의 더 낮은 rating 답 자체가 target인지 여부는 완전히 명시되어 있지 않다(§2).
2. 모델이 짝을 이룬 답을 복사하지 못하게 past token의 0~5%를 무작위로 mask한다.
3. Pile에 대한 pretraining log-likelihood term을 $\lambda=1.5$의 weight로 더한다(App. C).
4. inference 때 prompt에는 positive feedback token만 넣는다.

**근거(GPT-J 6B와 OPT, human labeler 75명).** summarization 평균 win rate는 CoH 45.3, RLHF 30.8이었고, CoH는 unlikelihood를 붙인 SFT 21.4에 대해서는 61.7이었다(Table 1). HH의 자동 dialogue 평가에서는 response 쌍 중 선호 응답을 고르는 정확도가 metric이다. SFT에 unlikelihood를 더하자 정확도가 내려갔고, 저자들은 이를 “unlikelihood hurts model generation ability”라고 해석했다(§4, Fig. 4). lm-evaluation-harness 22개 task의 zero-shot 평균은 GPT-J 40.60(GPT-J 논문에서 복사), SFT 40.54, CoH 40.95였다(Table 7, seed 5개). CoH와 모든 baseline에 Pile log-likelihood term을 적용했으므로(§2), 이 보존 효과를 conditioning만의 효과로 돌릴 수는 없다. 더 작은 model size에서는 CoH가 SFT보다 근소하게 낮았다(Fig. 5). 결과(단일 연구).

### 4.2 failure-conditioned agent training: NAT

**정의.** Negative-Aware Training(NAT)은 성공 trajectory 뒤에는 `Please generate a solution that **correctly** answers the question.`을 붙이고, 실패 trajectory 뒤에는 같은 문장의 `**incorrectly**`를 붙인다. loss는 model이 생성한 text에만 적용하며, inference에는 positive 문장을 쓴다([[learning-from-failure-nat]] §3.3).

**근거(LLaMA-2-Chat-7B, positive 2k개, negative 10k개, GSM8K·ASDiv·SVAMP·MultiArith 평균, Table 2).** positive만 쓰면 55.90, label 없이 negative를 추가한 NUT는 63.39, NAT는 64.64였다. 계산하면 8.74점 향상 중 7.49점은 label 없이도 나타났고, label이 추가한 부분은 1.25점이다. 13B에서 positive 5k개를 쓸 때 NUT는 69.10으로 positive-only 70.76보다 낮았지만 NAT는 71.28이었다.

- random string 두 쌍을 포함해 시험한 다섯 label 쌍의 점수는 63.15~64.04(random string은 64.04)였다(Table 7). 저자들은 효과가 단어 자체가 아니라 두 집합을 분리하는 데서 온다고 결론낸다.
- GSM8K에서 NUT는 tool-call error rate를 3.58%에서 10.47%로 올렸고, NAT의 error rate는 7.90%였다(Table 6).
- GPT-3.5 negative 대신 fine-tune한 LLaMA-2-7B의 negative를 쓰면 positive-only보다 평균이 3.16점(positive 2k개)과 6.20점(5k개) 낮았다. 반면 GPT-3.5 negative는 각각 8.74점과 3.25점을 더했다(Table 5, §5.2).
- HotpotQA에서 positive 500개를 쓸 때 NAT는 negative 500개에서 최고 점수였고, 1,000개에서는 더 낮았다(§6.1, Fig. 5).

**조건과 한계.** LLaMA-2-Chat 7B와 13B, GSM8K 기반 math·HotpotQA·StrategyQA만 다뤘으며 ground-truth label이 필요하다(§8). 결과(단일 연구).

### 4.3 control token을 쓰는 conditional pretraining

**근거.** GPT-2-small(124M)을 3.32B token으로 pretrain하면서 toxicity classifier가 설정한 `<|good|>` 또는 `<|bad|>`를 각 문장 앞에 넣고, `<|good|>` 뒤에서 sample했다. 평균 toxicity score는 maximum likelihood의 0.0141에 비해 0.0011이었다. 동시에 LAMBADA에서는 maximum likelihood를 약간 넘었고, feedback objective들 가운데 GLUE에서는 maximum likelihood에 가장 가까웠다([[pretraining-with-human-preferences]] §4.1, §4.3). flagged segment에 unlikelihood를 적용하면 toxicity task의 LAMBADA accuracy는 “very low”였고 PEP8 code task에서는 HumanEval 점수가 가장 낮았다. 하지만 PII task에서는 LAMBADA에서 maximum likelihood와 같았다(§4.3). feedback objective를 전체 pretraining에 적용한 run은 maximum-likelihood pretraining 뒤 1.6B-token fine-tune에만 최선 feedback objective를 적용한 run보다 좋았다(PII score 0.0013 대 0.0018, §5). 결과(단일 연구).

**§4의 함의.** CoH와 conditional pretraining은 실패를 사용하면서도 각각의 SFT 또는 maximum-likelihood baseline 수준의 general benchmark 점수를 유지했다(GPT-J 6B fine-tuning, GPT-2-small pretraining). 이 좁은 주장에 대해서는 두 연구에서 재현됐다.

## §5 지도 손실 내부의 제한된 gradient로서의 negative(뜻 4)

### 5.1 Unlikelihood training

**수식** ([[unlikelihood-training]] Eq. 4):

```
L_UL-token = −α · Σ_{c ∈ C_t} log(1 − p(c | x_<t)) − log p(x_t | x_<t)
```

- $C_t$: 시점 $t$의 negative candidate token(이전 context token 또는 model 자신의 continuation에서 반복된 n-gram의 token), $α$: weight, $x_t$: target token이다.
- candidate 하나 $k$에 대해 $∂[−log(1 − p_k)]/∂z_k = p_k$이고, $j ≠ k$이면 $∂/∂z_j = −p_k·p_j/(1 − p_k)$다(도출; 논문의 표기는 App. A Eq. 12).

**계산 예.** §2의 logit과 $α=1$을 쓰면, D에 대한 unlikelihood gradient는 0.032이고 $p_D$를 0.032에서 0.031로 옮긴다. A에 대한 gradient는 0.644이고 $p_A$를 0.644에서 0.404로 옮긴다. 이 항의 하한은 0이며, $z_k$에 대한 gradient는 $α·p_k$다. 따라서 negative가 이미 unlikely해질수록 negative에 가하는 update는 0으로 작아진다. §2의 push-down은 하한이 없고 $z_k$에 대한 gradient가 $1-p_k$이므로, $p_k$가 작아질수록 gradient는 1에 가까워진다.

**근거(16-layer Transformer, Wikitext-103, token-level과 sequence-level unlikelihood, Table 2).** beam search의 4-gram repetition은 .523 → .013, unique token 수는 9.5k → 19.1k였다(human text는 .006과 19.8k). perplexity는 25.64 → 26.72로 늘었다. crowdworker는 비교의 82%에서 baseline보다 이를 선호했다(Table 3). 결과(단일 연구).

**조건과 한계.** negative는 오답이 아니라 반복이다. 다른 종류의 negative를 쓰면 결과가 더 나빴다. 낮은 rating response를 negative로 둔 CoH와 HH dialogue에서는 SFT에 unlikelihood를 더하자 preference-classification accuracy가 내려갔다([[chain-of-hindsight]] §4, Fig. 4). classifier가 flag한 pretraining segment를 negative로 쓰면, unlikelihood는 toxicity task에서 LAMBADA accuracy를 매우 낮게 만들었고 code task에서는 HumanEval 최저 점수를 냈으며, PII task에서는 maximum likelihood와 같은 성능이었다([[pretraining-with-human-preferences]] §4.3). 이 task 의존성은 두 source에서 재현됐다.

### 5.2 CRINGE

**수식** ([[cringe-loss]] Eq. 3-5): $L = L_CE + α·log(1 + exp(s_neg − s₊))$. 여기서 $L_CE$는 positive sequence의 cross-entropy, $s_neg$는 label된 negative token의 logit, $s₊$는 negative를 제외한 model top-$k$ token의 softmax에서 sample한 token의 logit, $α$는 negative 항의 weight다. 공개 code는 `torch.topk(x, k=self.k + 1, axis=-1)`을 호출하고 negative를 mask한 뒤 `Categorical(logits=logits)`로 sample하여 `ce_loss + self.alpha * cr_loss`를 반환한다(App. A.1, Listing 1).

**계산 예.** negative A(logit 2)를 sample된 B(logit 1)와 비교하면 $σ(1)=0.731$이다. 따라서 $η=1$일 때 A의 logit은 1.269로 떨어지고 B의 logit은 1.731로 오른다. 확률은 $(0.337, 0.534, 0.095, 0.035)$가 된다. 빠진 mass의 대부분은 sample한 대안으로 간다. A에 맞선 negative D에서는 $σ(−3)=0.047$이므로 update가 작다.

**근거(Table 2-3).** BlenderBot 400M의 WikiToxic safety에서 safety classifier가 accept한 generation 비율은 baseline 59.4, unlikelihood 86.7, iterative CRINGE 99.9였고, ConvAI2 F1은 각각 15.9, 16.5, 16.6이었다. FITS dialogue(2.7B)의 unseen topic에서는 single-iteration CRINGE가 18.4 F1, iterative CRINGE가 17.8이었으며, 저자들은 overfitting을 가능한 원인 하나로 든다(§4.4). 결과(단일 연구).

### 5.3 ORPO의 odds-ratio 항

**수식** ([[orpo]] Eq. 6-7): $L = L_SFT + λ·(−log σ(log[odds(y_w)/odds(y_l)]))$, 여기서 $odds(y)=P(y)/(1-P(y))$다. $P$는 length-normalized sequence probability, $y_w$와 $y_l$은 각각 chosen 및 rejected response, $σ$는 logistic sigmoid, $L_SFT$는 $y_w$의 negative log-likelihood, $λ$는 odds-ratio 항의 weight다.

**계산 예.** $P(y_w)=0.5$, $P(y_l)=0.2$이면 odds는 1과 0.25이고, log odds ratio는 1.386, $L_OR=−log σ(1.386)=0.223$이다.

**근거.** OPT-350M에서 $λ=1.0$을 쓰면 rejected log-probability가 감소했다(Fig. 7). Mistral-7B에서 $λ=1.0$은 rejected뿐 아니라 chosen log-probability도 낮췄고, $λ=0.1$은 rejected를 낮추지 않았다(App. E.1). positive 항 $L_SFT$는 항상 포함된다. 자세한 내용과 DPO 비교는 [[ch-39]]에 있다.

### 5.4 NFT: 암묵적인 negative policy

**정의.** NFT(Negative-aware Fine-Tuning)는 online supervised method다. 매 step마다 question 하나당 답 16개를 sample하고 verifier로 label한 뒤 두 집합 모두로 학습한다([[nft-negative-aware-finetuning]] §3, App. C).

**메커니즘.**

1. sampling policy는 $r_q·π⁺ + (1-r_q)·π⁻ = π_old$로 나뉜다. $r_q$는 해당 question의 correctness rate다(Eq. 7).
2. negative policy는 학습 중인 model을 통해 $π_θ⁻=(π_old-r_q·π_θ⁺)/(1-r_q)$로 쓴다(Eq. 8). 따라서 오답에 대해 $π_θ⁻$의 likelihood를 최대화하면 그 오답에서 $π_θ⁺$는 낮아진다. data와 model capacity가 무한하면 optimum은 $π_θ⁺=π⁺$다(Theorem 3.1).
3. token-level loss(Eq. 10)는 $−ω(q)·[r·log R+(1-r)·log\ max_v((1-r̂_q·R)/(1-r̂_q), ε)]$다. 여기서 $R=π_θ⁺(a_t)/π_old(a_t)$, $r∈\{0,1\}$은 answer $a$의 verifier reward, $r̂_q$는 sample한 $K$개 answer 중 정답 비율, $π_old$는 sample을 만든 policy, $ω(q)$는 prompt weight다. $max_v$는 gradient를 유지한 채 값을 $ε$에서 clip한다(Algorithm 1).

**계산 예.** $x=(1-r̂_q·R)/(1-r̂_q)$를 $max_v$의 argument라고 하자. $R$에 대한 negative 항 gradient는 $1/max(x,ε)$로 scale되며, 여기서는 이를 penalty weight라고 부른다. $r̂_q=0.25$, $ε=1.0$일 때 $R=1.2$이면 $x=(1-0.3)/0.75=0.933$이므로 penalty weight는 1.07이 아니라 1.0이다. $R=3.6$이면 $x=0.133$, weight는 7.5가 아니라 1.0이다. 이 clip은 확률이 커지는 오답을 얼마나 강하게 벌할지 상한을 둔다.

**근거(Table 1, DAPO-Math-17k training; AIME24·AIME25·AMC23의 avg@32와 MATH500·OlympiadBench·Minerva의 avg@1을 합친 여섯 benchmark 평균).** Qwen2.5-Math-7B에서 RFT는 48.3, NFT 51.7, DAPO 51.2, GRPO 49.5였다. Qwen2.5-32B에서는 RFT 52.8, NFT 59.2, DAPO 59.9였다. RFT의 entropy(model이 자기 sample에 부여한 token distribution의 entropy)는 training 중 떨어진 반면 NFT와 DAPO의 entropy는 올랐다(Fig. 8). $ω(q)=(1-r̂_q)/r̂_q$, $ε≤1$, on-policy data(current model에서 sample하므로 $R=1$)에서는 NFT와 GRPO의 gradient가 같다(Propositions 4.1-4.2). default $ω(q)=1-r̂_q$에서는 NFT가 Dr. GRPO와 align한다(§4, App. A). 저자들은 “$ε→0$의 지나치게 공격적인 penalization은 overall performance를 떨어뜨린다”고 썼다(§5.4). 결과(단일 연구).

### 5.5 REDI: offline distillation의 비대칭 weight

**수식** ([[redi-reinforcement-distillation]] Eq. 5): $L=−log π(y_w|x)/|y_w|+α·log π(y_l|x)/|y_l|$이며 pair에 대해 평균낸다. $x$는 문제, $y_w$는 같은 문제의 correct teacher trace, $y_l$은 incorrect trace, $|y|$는 token 길이, $α∈[0,1]$은 negative weight다($α=0$은 positive SFT).

**계산 예.** 두 번째 항에는 하한이 없다. token당 $log π(y_l|x)/|y_l|$가 −1에서 −10으로 떨어지면 $α=0.8$에서 $y_w$에 무슨 일이 일어나든 loss는 $0.8×9=7.2$만큼 내려간다. weight를 낮추면 이 감소는 느려지고, epoch 하나만 쓰면 update를 받는 횟수도 제한된다(해석).

**근거(Qwen2.5-Math-1.5B, OpenR1-Math에서 positive 78k개 뒤 pair 53k개; MATH-500·AIME24·AMC23·Minerva·OlympiadBench에서 16 sample의 pass@1 평균).** 3-epoch SFT model(평균 41.9)에서 DPO는 47.2, SimPO 47.2, $α=1.0$, LR $2e−7$은 47.7, $α=0.8$, LR $1e−6$은 48.3이었다(Table 2). $α=1.0$, LR $1e−6$은 collapse했다. 즉 “positive($y_w$)와 negative($y_l$) response 모두의 likelihood가 빠르게 내려가고 task accuracy도 하락”했다(§2.2, §4.3). 5-epoch SFT model에서는 45.8 → 49.5였고, DeepSeek-R1-Distill-Qwen-1.5B의 48.6보다 높았다(Table 1). pass@16은 유지되거나 올랐다(App. D.3). 결과(단일 연구).

### 5.6 제한이 없거나 anchor가 없는 likelihood 감소의 failure mode

| 설정 | 관찰 | 출처 |
|---|---|---|
| NT: wrong teacher trace에 unbounded $−0.05·log P$ 항을 positive log-likelihood에 더함, LLaMA-7B MATH | 4.48, positive-only CoT distillation 5.29보다 낮음 | [[dust-into-gold-negative-distillation]] Table 2 |
| REDI $α=1.0$, LR $1e−6$(positive 항 포함) | chosen과 rejected의 log-probability가 collapse하고 accuracy 하락 | [[redi-reinforcement-distillation]] §4.3 |
| DPO $β=0.001$, LR $1e−6$, 같은 data | MATH-500 약 80.9%에서 peak를 찍은 뒤 gradient step size 급증과 함께 collapse | [[redi-reinforcement-distillation]] §4.2 |
| NFT $ε→0$(positive 항 포함) | overall performance 하락 | [[nft-negative-aware-finetuning]] §5.4 |
| ORPO에서 odds ratio 대신 probability-ratio penalty(SFT 항 포함) | rejected log-probability가 빠르게 −4 아래로 하락 | [[orpo]] App. B |
| 유사한 refusal pair에 DPO | training-set refusal rate 74.4% → 33.4% | [[likelihood-displacement]] §6.2 |

첫 다섯 행에서는 negative likelihood를 낮추는 힘에 제한이 약했다(unbounded term, $α=1.0$, $β=0.001$, $ε→0$, 또는 odds ratio 대신 probability ratio). 여섯째 행에서는 preferred와 dispreferred response가 유사했다. 여섯 행 중 넷에는 positive log-likelihood 항이 있었으므로, positive 항만으로는 failure를 막지 못했다. 서로 다른 setting의 다섯 source에서 재현됐다.

## §6 negative가 가져오는 이득의 크기

| 출처와 설정 | negative 없음 | negative 있음 | negative에 귀속할 수 있는 몫 |
|---|---|---|---|
| NFT, Qwen2.5-32B(base 29.6) | RFT 52.8 | NFT 59.2 | 논문이 제시한 값은 20%(§5.3), Table 1로부터 계산하면 $6.4/29.6 = 21.6\%$(도출) |
| NFT, Qwen2.5-Math-7B(base 31.6) | RFT 48.3 | NFT 51.7 | $3.4/20.1 = 16.9\%$(도출) |
| NAT, LLaMA-2-Chat-7B, positive 2k개 | 55.90 | NAT 64.64 | 8.74점; label이 없을 때는 7.49점 |
| LEMA, LLaMA-2-70B GSM8K, 5.8M token | 82.1 | 83.5 | 같은 token 수에서 1.4점 |
| REDI, Qwen2.5-Math-1.5B | SFT-3ep 41.9 | 48.3 | 분리되지 않음: stage 2도 chosen trace 53k개로 학습함 |
| Likra, Mistral-7B-v0.1 ARC | SFT .6630 | .8123 | scoring 수준의 결과이며 multiple choice에만 해당 |

두 가지 주의점이 있다. NFT의 RFT baseline도 constant prompt weight $ω(q)=1$을 쓰지만 NFT는 $1-r_q$를 쓴다(App. C). 그러므로 20%에는 weight 차이도 들어 있다. Likra가 “negative 하나가 positive example을 하나 더 추가하는 것보다 각각 `10×` 더 도움이 될 수 있다”고 한 주장은 64–256개 example 단계에서, 생성할 수 없는 별도의 negative head를 사용하는 multiple-choice scoring에서만 성립한다([[negative-examples-likra]] §1, §3.2). 이 head는 wrong answer에 보통 likelihood를 적용해 학습한다(뜻 2). 생성 능력에 대해서는 결과를 시험하지 않았다.

## §7 context에 남겨 둔 실패 시도의 loss mask

| 형식 | 실패가 놓이는 위치 | 실패 token에 주는 loss | 근거 |
|---|---|---|---|
| 틀린 풀이 뒤의 correction | input | 없음 | [[lema-learning-from-mistakes]] App. B.2 |
| noisy response에 대한 critique | input | 없음 | [[critique-fine-tuning]] §2.3 |
| hindsight feedback phrase | input과 target sequence | feedback token에는 없음 | [[chain-of-hindsight]] §2 |
| inline retry, scratch부터의 pretraining | target | $q ≤ 0.5$에서는 masking이 불필요 | [[physics-of-lm-2-2-learning-from-mistakes]] Result 2-3 |
| inline retry, error-free model의 LoRA fine-tune | target | 작은 rank에서는 masking이 중요 | 같은 논문, Result 7 |
| failure-conditioned trajectory | negative label 아래의 target | model-generated token에는 있음, tool observation은 mask | [[learning-from-failure-nat]] §3.3 |

**규칙.** pretrained model을 self-correction trace로 fine-tune할 때는 failed attempt를 input에 넣거나 그 token을 mask한다. 유일한 controlled test에서 unmasked error가 small-rank LoRA fine-tuning을 해쳤기 때문이다. 반대로 retry data가 pretraining의 일부라면 retry rate가 0.5 이하일 때 masking은 필요하지 않다.

## 음성 샘플과 음성 피드백

이 절은 SFT 단계에 과정의 표준을 적용한다. 뜻 (1)–(4)는 §1 표를 가리킨다.
1. **음성 샘플의 출처.** 최종 답 검사(LEMA, NAT, Dust into Gold), 잘린 답도 negative로 세는 self-sample verifier(NFT App. C), 둘 모두 correct라고 말해야 하는 두 judge(REDI §3.1: Llama judge 또는 Math-Verify 중 하나라도 correct로 label하지 않으면 그 trace는 negative), GPT-4o judge(CFT: 점검한 50개 중 critique error 약 20%), classifier(conditional pretraining, CRINGE의 반복 labeling), human label(CoH rating, CRINGE의 FITS binary feedback)이 있다. 이 source들 중 어느 것도 false-negative rate를 보고하지 않는다.
2. **현재 관행이 하는 일.** distillation과 RFT pipeline은 failure를 버린다(뜻 1). 대안은 content(§3), conditioning(§4), bounded gradient(§5)다.
3. **메커니즘.** $∂ \log p_y/∂z_j = 1[j = y] − p_j$이다. SFT는 이미 wrong token의 probability에 비례해 그 token을 낮춘다. push-down은 모든 다른 logit을 $η·p_j$만큼 올리므로, unlikely negative에 대해서는 mass가 가장 likely한 token에 집중된다(§2).
4. **근거.** 이득은 §3–§5, failure mode는 §5.6과 MIX 3.03 대 positive-only distillation 5.29([[dust-into-gold-negative-distillation]] Table 2)에 있다.
5. **control.** positive log-likelihood term을 유지한다(REDI, ORPO, CRINGE, unlikelihood). negative를 제한한다(unlikelihood의 $p_k$-scaled gradient, NFT $ε=1.0$, REDI $α=0.8$). negative를 on-policy로 만들거나 strong generator에서 얻는다(NFT; NAT Table 5). near-miss negative를 선호한다([[negative-examples-likra]] §4.3). correction이 있으면 failure를 input으로 옮긴다(§7).
6. **진단.** chosen/rejected log-probability를 분리해 기록한다(REDI §4.2). training 중 entropy(NFT Fig. 8), $k=16$ 이상에서 pass@1과 pass@$k$(REDI App. D.3), action 또는 format error rate(NAT Table 6), problem당 retry 수(Physics 2.2 Result 4)를 기록한다.
7. **범용성에 미치는 효과.** coverage에서는 pass@16이 유지되고(REDI), entropy가 오른다(NFT). calibration과 hallucination은 이 source들이 측정하지 않았다. refusal behavior에서는 DPO refusal training에서 displacement가 refusal rate를 낮췄다([[likelihood-displacement]]). 이 source들의 SFT negative에서는 refusal과 over-refusal을 측정하지 않았다. forgetting에서는 CoH와 conditional pretraining이 general benchmark score를 유지했고(§4), unlikelihood는 feedback이 있는 pretraining task 3개 중 2개에서 이를 낮췄다(§5.1).

## 레시피

| 모델(exact release) | 크기 | 단계 | 설정 | 값 | source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| LLaMA-2-Chat 7B (NAT, math) | 7B | SFT | epoch; batch; peak LR; schedule; hardware | 2; 64; $5×10^{-5}$; cosine, 3% warm-up; 4×A100, DeepSpeed ZeRO-3 | arXiv:2402.11651v2 §4.1 | verified 2026-09-15 | ablation 미보고 |
| LLaMA-2-Chat 7B (NAT, math) | 7B | SFT | data; label; loss mask | GPT-3.5 positive 2k 또는 5k + negative 10k; `correctly`/`incorrectly` suffix; model-generated text에 loss | §3.3; App. B | verified 2026-09-15 | Fig. 2: negative 0–12k, 약 11k에서 plateau; Table 7: label wording |
| LLaMA-2-70B (LEMA) | 70B | SFT (QLoRA) | rank; dropout; LR; batch; steps | 64; 0.05; $1e^{-4}$; 96; 2,000 | arXiv:2310.20689v4 §3.3 | verified 2026-09-15 | ablation 미보고 |
| LLaMA-2 7B–70B (LEMA) | 7B–70B | SFT | correction pair; loss mask | GSM8K 12,523, MATH 6,306, CSQA 7,241; output token만 | §3.2; App. B.2 | verified 2026-09-15 | Fig. 4 및 Table 2: equal example과 equal token |
| Qwen2.5-Math-7B (CFT) | 7B | SFT | data; epoch; LR; schedule; global batch; selection | WebInstruct-CFT 50K; 1; $5e^{-6}$; cosine, warm-up ratio 0.1; 512; MATH-500에서 best | arXiv:2501.17703v4 §2.1, §3.1 | verified 2026-09-15 | Table 5–7, 9–10: data source, teacher, mixing, length |
| GPT-J 6B (CoH) | 6B | SFT | optimizer; batch; pretraining-loss weight; past-token mask | Adam (0.9, 0.95, $1e^{-8}$); feedback data batch 512, pretraining data 2048(unit 미명시); Pile에서 $λ=1.5$; 0–5% | arXiv:2302.02676v8 §2; App. C | verified 2026-09-15 | ablation 미보고 |
| 16-layer Transformer (unlikelihood) | $d_{model}$ 1024 | pretrain-stable; SFT | sequence length; token-level update; sequence-level update; sequence update 비율 | 1,536; 8 GPU에서 ≤150k; 1,500; 0.5 | arXiv:1908.04319v2 §6 | verified 2026-09-15 | Table 2: token 대 sequence variant |
| 16-layer Transformer (unlikelihood) | $d_{model}$ 1024 | pretrain-stable | $α$ | 미보고 | §5–6, App. A–D 점검 | not reported | — |
| BlenderBot 2 (CRINGE, FITS) | 2.7B | SFT | $α$; $k$; iteration; batch; LR; warm-up; clip; max step | 0.5; 5; 1 또는 2; 16; $5e^{-6}$–$5e^{-5}$; 100; 0.1; 8,000 | arXiv:2211.05826v1 App. Table 9 | verified 2026-09-15 | Table 3: single 대 iterative; Fig. 4: top-3 configuration |
| GPT-2-small (conditional pretraining, toxicity) | 124M | pretrain-stable | token; LR; batch; threshold $t$ | 3.32B; $5·10^{-4}$; 64; $5.6·10^{-4}$ | arXiv:2302.08582v2 §3.3; App. A Table 1 | verified 2026-09-15 | App. A Fig. 8: threshold ablation |
| GPT2-12-12 (retry data, iGSM-med) | GPT2-small | pretrain-stable | retry rate; LR; weight decay; batch; context; step | 0.01–0.5; 0.002; 0.05; 512; 768; 100,000 | arXiv:2408.16293v1 §4; App. D.1 | verified 2026-09-15 | Fig. 4b: equal token에서 retry rate |
| Qwen2.5-Math-7B (NFT) | 7B | RL (online, supervised loss) | data; LR; prompt × sample; update; $ε$; $ω(q)$; context; temperature; GPU | DAPO-Math-17k; $1e^{-6}$, linear warm-up; rollout당 $512×16$, 각각 16 gradient step, rollout 320; 1.0; $1-r_q$; 4K; 1.0; H100 64개 | arXiv:2505.18116v3 §5.1; App. C | verified 2026-09-15 | Fig. 10: $ε$; Fig. 9: $ω(q)$ |
| Qwen2.5-32B (NFT) | 32B | RL (online, supervised loss) | context; GPU | 16K; H100 128–256개 | App. C | verified 2026-09-15 | App. C: 32K와 비교해 16K는 “does not significantly affect performance” |
| Qwen2.5-Math-1.5B (REDI stage 1) | 1.5B | distill-SFT | data; LR; schedule; batch; epoch; max length | positive 78k; $5e^{-5}$; 10% linear warm-up, linear decay; 128; ablation 3, final 5; 32,768 | arXiv:2505.24850v2 §3.1–3.2; App. C.3 | verified 2026-09-15 | Fig. 2: 5 epoch 동안 accuracy |
| Qwen2.5-Math-1.5B (REDI stage 2) | 1.5B | distill-SFT | data; $α$; LR; batch; epoch; final model compute(stage 1 5 epoch + stage 2) | pair 53k; 0.8; $1e^{-6}$; 32; 1; A100-80GB GPU hour 약 136(8-GPU node 하나에서 17시간) | §3.1; App. C.4 | verified 2026-09-15 | Table 2 및 §4.4: $α∈\{0.2,0.5,0.8,1.0\}$ |
| Mistral-ORPO-$β$ | 7B | preference | $λ$; peak LR | 0.1; $8e^{-6}$ | arXiv:2403.07691v2 §6.1; App. C ([[orpo]]) | verified 2026-09-14 | App. E: $λ∈\{0.1,0.5,1.0\}$ |

단위: `prompts × samples`는 rollout step당 질문 수와 질문당 답 수를 센다. NFT는 rollout 320회와 각 회당 gradient step 16회이므로 gradient step이 5,000회를 넘는다(App. C).

**작은 범용 run의 시작점.** verifier가 label한 math trace로 1.5B model을 distill하고 같은 strong teacher의 wrong trace도 이용할 수 있다면, REDI 설정이 문서화된 시작점이다. positive에서 SFT를 5 epoch(LR $5e^{-5}$, batch 128) 수행한 뒤 pair에서 $α=0.8$, LR $1e^{-6}$, batch 32로 1 epoch 수행한다(Qwen2.5-Math-1.5B, OpenR1-Math positive trace 78k와 pair 53k; 두 stage 합계 약 A100-80GB GPU hour 136). negative가 model 자신의 online sample이면, NFT는 Qwen2.5-Math-7B와 DAPO-Math-17k에서 $ε=1.0$, $ω(q)=1-r_q$, LR $1e^{-6}$, 질문당 sample 16개를 썼다(H100 64개, 4K context). failed agent trajectory가 success label만 갖는다면 NAT는 LLaMA-2-Chat-7B에서 positive 2k–5k와 negative 10k에 suffix 형식, epoch 2, LR $5×10^{-5}$, batch 64를 썼다(A100 4개). 이 값들 중 어느 것도 multi-domain SFT mixture 안에서 시험되지 않았다.

## 일반화 관점

**(a) breadth를 늘리는 것.**
- STEM question의 critique target은 general instruction-following score를 올렸다. Qwen2.5-Math-7B의 IFEval loose는 SFT 0.330에서 0.362로, MT-Bench는 5.23에서 6.49로 올랐다([[critique-fine-tuning]] Table 8). 단일 연구 결과다.
- negative term은 sampling diversity를 유지했다. NFT에서는 token entropy가 올랐고 RFT에서는 내렸으며([[nft-negative-aware-finetuning]] Fig. 8), REDI는 pass@1을 올리면서 pass@16을 유지하거나 올렸다([[redi-reinforcement-distillation]] App. D.3). 각각 단일 연구 결과이고 metric도 entropy와 pass@16으로 다르다.
- correction data는 직접 학습하지 않은 task로 전이됐다. GSM8K 기반 LEMA data에서 SVAMP와 ASDiv가 올랐다([[lema-learning-from-mistakes]] Table 1).
- conditioning은 general score를 유지했다. CoH zero-shot average는 SFT 40.54 대 40.95였고([[chain-of-hindsight]] Table 7), conditional pretraining은 GLUE에서 maximum likelihood에 가장 가까운 feedback objective였다([[pretraining-with-human-preferences]] §4.3). 이 주장에는 재현 근거가 있다. CoH는 Pile log-likelihood term도 함께 학습했다(§4.1).

**(b) narrowing 또는 forgetting을 일으키는 것.**
- failure를 plain target으로 쓰는 경우: MIX 3.03 대 positive-only distillation 5.29([[dust-into-gold-negative-distillation]] Table 2), 50%를 넘게 wrong인 response로 SFT해 Qwen2.5-7B가 37.4에서 20.2로 하락([[critique-fine-tuning]] Table 2), weaker generator의 negative는 NAT average를 positives only보다 3.16점 낮게 남겼다([[learning-from-failure-nat]] Table 5). 이 source들에서 재현되지만 보편적이지는 않다. unlabeled GPT-3.5 failure는 NAT math average를 4 setting 중 3개에서 올렸다(NUT, Table 2).
- rated 또는 classifier-flagged negative에 대한 unlikelihood는 HH preference-classification accuracy를 낮췄고([[chain-of-hindsight]] §4, Fig. 4), feedback이 있는 pretraining task 3개 중 2개에서 LAMBADA 또는 HumanEval score를 낮췄다([[pretraining-with-human-preferences]] §4.3). 재현됐지만 task 의존적이다.
- iterative negative training은 seen topic에 fit될 수 있다. 두 번째 iteration 뒤 CRINGE의 unseen topic F1은 18.4에서 17.8로 떨어졌다([[cringe-loss]] Table 3).
- 제한 없는 push-down: §5.6.
- stage mismatch: LoRA fine-tune의 retry data는 significant gain이 없었고 small rank에서 해로웠다([[physics-of-lm-2-2-learning-from-mistakes]] Result 7).

**(c) 이 단계에서 측정하는 법.**
- equal step과 equal token에서 positive-only arm과 비교한다(LEMA Table 2).
- chosen/rejected log-probability, entropy, 큰 $k$의 pass@$k$를 추적한다(negative-feedback 절 item 6).
- held-out domain과 unseen split을 보고한다. GPQA와 HumanEval(REDI Table 5), test-unseen topic(CRINGE Table 3), IFEval과 MT-Bench(CFT Table 8)가 예다.
- 이 source들에서 알려진 measurement error는 test set에서 best-of-20 checkpoint를 고른 것(LEMA §3.3), question 30개인 AIME24(CFT §3.4), REDI checkpoint가 불명확한 out-of-domain table이다. REDI Table 5는 App. C.4에 따라 5-epoch SFT model에서 초기화한 final Qwen-REDI-1.5B를 명시하지만 OlympiadBench 43.4는 Table 2의 3-epoch-initialized REDI와 같고, Table 1은 Qwen-REDI-1.5B에 45.2를 준다. 이 밖에 REDI ablation table에서 configuration마다 best checkpoint를 고른 것(REDI Table 2), seed나 variance 보고 없이 absolute accuracy 3–7%인 것(Dust into Gold)이 있다. 인용한 14개 source 중 contamination check를 보고한 것은 없다(`contamination` full-text search).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 검사 |
|---|---|---|
| failed sample을 ordinary target으로 추가 | positive-only arm보다 낮은 accuracy(Dust into Gold MIX; positive 5k인 NAT 13B), 또는 높은 accuracy와 더 많은 tool/format error(NAT 7B에서 3.58% → 10.47%) | equal step에서 positive-only와 mixed arm을 학습하고 format/tool-call error를 센다 |
| symmetric 또는 unclipped negative term | chosen과 rejected log-probability가 함께 하락, accuracy가 peak 후 하락, gradient norm 상승 | 매 step 두 log-probability와 gradient norm을 기록하고 early checkpoint를 보존한다 |
| LoRA self-correction fine-tune에서 failed attempt를 unmasked로 사용 | small rank에서 original model보다 낮은 accuracy | masked 대 unmasked arm을 돌리고 problem당 retry를 센다 |
| inference에서 failure label을 남기거나 positive label을 빠뜨림 | unlabeled mixing보다 이득이 없음 | positive label 유무로 평가한다. conditional pretraining이 toxicity와 PII에 한 것처럼 decoding 시 control token probability를 0으로 둔다(App. A) |
| positive보다 약한 model이 만든 negative | positives only보다 낮은 average | generator별 negative를 ablate하고 near-miss negative를 선호한다 |
| audit하지 않은 judge 또는 critique label | relabel한 data에서 gain이 사라짐 | CFT처럼 최소 50 label을 inspect하고 error rate를 보고한다 |
| 전체 gain을 negative 탓으로 돌림 | negative term을 빼도 gain 대부분이 남음 | 같은 prompt weighting으로 $α=0$ 또는 RFT arm을 둔다 |
| positive-only SFT가 plausible wrong answer를 억제할 것이라 기대 | chosen과 함께 rejected log-probability가 상승 | SFT 중 held-out rejected response의 log-probability를 추적한다 |

## 이해 점검

1. §2에서 D에 push-down을 가하자 $p_A$는 0.644에서 0.751로 올랐지만, A에 push-down을 가하자 mass는 B, C, D로 퍼졌다. $∂ \log p_k/∂z_j = 1[j = k] − p_j$로 두 결과를 설명하고, D의 unlikelihood term이 $p_D$를 0.0015만 바꾼 이유도 설명하라.
2. LEMA와 CFT는 모두 모든 gradient를 positive로 유지한다. 이것이 likelihood displacement를 배제하는 이유는 무엇인가? 또한 LEMA의 equal-token 비교(82.1 대 83.5)는 equal-example 비교로는 보이지 않는 무엇을 보이는가?
3. retry data는 pretraining에서는 mask가 필요 없었지만 small-rank LoRA fine-tuning에서는 masking의 이득이 있었다. Result 4의 retry count와 Result 7의 stage finding을 사용해 causal account를 제시하라.
4. NAT는 label을 random string으로 했을 때 64.04를 기록했다. model은 label에서 무엇을 배우며, inference에 label이 없으면 왜 gain이 사라지는가?
5. GRPO-matched prompt weight에서 NFT gradient는 on-policy GRPO의 gradient와 같다. 그럼에도 $ε$가 필요한 이유는 무엇인가? $\hat r_q=0.25$일 때 $R$이 1.2에서 3.6으로 오르면 clip이 있을 때와 없을 때 negative weight에는 무슨 일이 일어나는가?
6. REDI negative term에는 lower bound가 없다. $α=0.8$, one epoch, positive term이 함께 $α=1.0$, LR $1e^{-6}$에서 관찰한 collapse를 어떻게 막는지 설명하라.
7. NFT는 32B gain의 20%를 negative에 귀속하지만 Likra는 negative example당 최대 10배 더 큰 gain을 보고한다. 두 결과가 양립하게 만드는 setting 차이를 열거하라.
8. 범용 assistant용 SFT mixture에 failed tool-use trajectory 20%를 추가한다. 이 장의 measurement 중 어떤 세 가지가 narrowing을 가장 먼저 탐지하며, 그 이유는 무엇인가?

## 연결

- 이전: [[ch-31]] — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation. positive-only filtering이 이 장이 확장하는 baseline이다.
- 다음: [[ch-33]] — Case Studies A: How Tülu 3 and Llama 3 Measured and Protected Broad Capability.
- 이후: [[ch-39]] — Offline Preference Optimization: DPO and Its Variants(reference model이 있는 rejected term); [[ch-43a]] — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages(이 장에 의존).

## 출처

- [[unlikelihood-training]] — unlikelihood objective, gradient, Wikitext-103 결과.
- [[cringe-loss]] — negative token용 top-$k$ contrastive loss, code, safety와 FITS 결과, overfitting 주석.
- [[chain-of-hindsight]] — feedback-conditioned sequence, loss mask, human evaluation, alignment-tax 표.
- [[lema-learning-from-mistakes]] — input에 wrong solution을 둔 correction pair, equal-token 비교.
- [[physics-of-lm-2-2-learning-from-mistakes]] — retry data, masking, pretraining 대 LoRA fine-tuning.
- [[critique-fine-tuning]] — critique target, instruction-following 전이, critique error rate.
- [[dust-into-gold-negative-distillation]] — wrong teacher trace의 complementarity, MIX·NT·unlikelihood baseline.
- [[learning-from-failure-nat]] — failure-conditioned agent trajectory, label ablation, negative quality.
- [[orpo]] — positive-only SFT가 rejected log-probability를 올림, odds-ratio term.
- [[nft-negative-aware-finetuning]] — implicit negative policy, $ε$ clip, entropy, 80/20 attribution.
- [[redi-reinforcement-distillation]] — asymmetric $α$, collapse, pass@16, recipe.
- [[negative-examples-likra]] — multiple-choice scoring에서 negative의 example당 효과, near-miss negative.
- [[pretraining-with-human-preferences]] — conditional pretraining 대 unlikelihood 대 filtering.
- [[likelihood-displacement]] — push-down의 shared-parameter side effect, [[ch-43a]]로 이어짐.
