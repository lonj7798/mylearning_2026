<!-- chapter: ch-08b
     companion to: [[read]]
     kind: qa
     track: pretraining
-->

# ch-08b — Reading Q&A

## Q1. In-context learning(ICL)이 뭔가?

**Kernel.** weight를 바꾸지 않고, prompt 안의 instruction과 K개 예시만 보고 새 문제를 푸는 것(§2). Zero-shot K=0, one-shot K=1, few-shot K=10~100. GPT-3 175B 2-digit addition은 zero-shot 76.9 → one-shot 99.6 → few-shot 100.0, 3-digit addition은 34.2 → 80.4(Table 3.9). ICL은 gradient/optimizer step이 없는 forward pass이며, "learning"은 예시가 늘수록 성능이 오르는 행동을 가리키는 이름이다. GPT-3의 표현으로 pretraining이 outer loop(weight가 바뀜), in-context 적응이 inner loop(weight 불변)이다. ICL이 task를 처음부터 배우는지 pretraining에서 본 task를 알아보는지는 논문 스스로 Open question(§2). see [[read]] §2.

## Q2. "Unsupervised multitask learning"과 factorizing이 뭔가?

**Kernel.** Factorizing은 문장 전체의 확률을 토큰별 조건부확률의 곱으로 쪼개는 것: `p(x)=Π p(s_i | s_1..s_{i-1})`(§1). Unsupervised는 사람이 만든 label이 필요 없다는 뜻(정답 = 텍스트의 다음 토큰). Multitask인 이유: task·input·output을 한 sequence로 쓸 수 있고, 웹 텍스트에 task demonstration이 자연 발생해 있음(WebText에서 비영어 페이지를 제거했는데도 10MB의 프랑스어 텍스트 발견, "TL;DR:" 힌트로 CNN/DM R-AVG 15.03 → 21.40). 저자들은 unsupervised objective의 global minimum이 supervised objective의 global minimum이기도 하다고 논증하고, 실제 optimization이 거기 도달하는지는 열린 질문으로 남긴다. see [[read]] §1.

## Q3. GPT-2 구조는? user/assistant 구분이 있나?

**Kernel.** 없다. GPT-2는 chat template(ch-04)이 없는 순수 autoregressive LM이다. "TL;DR:", "Q:/A:"는 special token이 아니라 웹 텍스트에 사람들이 쓴 관습적 표현이 corpus에 섞여 있어 모델이 next-token prediction으로 익힌 것. 구조 세부(decoder-only, causal attention, byte-level BPE)는 ch-08b 소스가 다루지 않는 일반 지식. 크기는 117M/345M/762M/1542M(§1). see [[read]] §1.

## Q4. Byte-level BPE tokenizer가 뭔가? 1글자 = 1토큰인가?

**Kernel.** 아니다. BPE는 가장 작은 단위에서 시작해 코퍼스에서 가장 자주 붙어 나오는 인접 쌍을 반복적으로 합쳐 vocabulary를 만든다(low/lower/lowest/newest/widest 예: `es` → `est` → `lo`). 흔한 단어는 1토큰, 드문 단어는 여러 조각. Byte-level은 시작 단위를 유니코드 글자 대신 UTF-8 byte(256가지)로 잡아 `<UNK>` 없이 모든 텍스트를 표현한다(GPT-2 vocabulary 약 50,257개는 일반 지식). see [[read]] §1.

## Q5. "unsupervised objective의 global minimum이 supervised objective의 global minimum"이 무슨 뜻인가? well-trained = well-trained SFT/RL인가?

**Kernel.** 이론적 상한에 대한 확률론적 사실이다: 텍스트 분포 전체를 정확히 모델링한 모델은 그 안의 모든 조건부 분포(각 task의 output | task, input)도 정확히 안다. 실제로 유한한 데이터·compute로 학습된 모델은 그 minimum에 도달하지 못한다. 따라서 (1) "loss가 낮은 모델 = 각 task supervised objective 최적"은 성립하지 않고(ch-08a §6-§7: 17-task 평균 오차 0.05% vs HellaSwag 25.73%, Schaeffer et al.), (2) "잘 pretrain된 모델 = 잘 SFT/RL된 모델"도 성립하지 않는다(ch-08a §8: OLMo-1B는 2.5T 토큰을 넘으면 SFT 후 AlpacaEval 최대 3% 하락; SFT/RL은 각자의 failure mode를 가짐, ch-07 §4-§5). see [[read]] §1.

## Q6. ICL과 unsupervised learning의 차이? ICL은 learning/training인가?

**Kernel.** 서로 다른 축이다. Unsupervised는 학습 방식(label 필요 여부), ICL은 학습이 끝난 모델의 추론 시점 행동(weight 변경 여부). 인과 관계: 특정 데이터 속성(§3: burstiness, long tail)을 가진 텍스트로 한 unsupervised pretraining이 ICL 능력을 만든다. ICL 순간에는 gradient/weight update가 없어 supervised/unsupervised 이분법이 적용되지 않는다. §4의 induction head는 pretraining 때 형성된 고정 회로가 새 context에 적용되는 forward pass 계산이다. see [[read]] §2, §3, §4.

## Q7. downstream = forward pass = inner loop인가?

**Kernel.** 아니다, 세 축이 다르다. Downstream은 무엇을 재는가(task; upstream loss와 대비), forward pass는 어떻게 계산하는가(backward pass와 대비), inner loop는 weight update 없이 context로 적응하는 방식(outer loop = weight update가 있는 학습과 대비). Forward pass는 outer loop(pretraining의 매 step) 안에서도 쓰이므로 inner loop와 outer loop를 가르는 기준은 backward pass·weight update의 유무. Downstream은 SFT/RL 이후 모델에도 측정하고, generation 채점은 forward pass를 여러 번 돈다.

## Q8. §3의 다섯 용어(burstiness, long tail, dynamic meaning, IWL, Zipfian marginal)

**Kernel.** 실험(Omniglot 1,623 class × 20 image; 8개 image-label 쌍 + query, 12-layer causal transformer)에서 각 용어가 조작된 방식:
- **Class** = 손글씨 문자 한 종류(같은 class에 20장의 서로 다른 손글씨가 있음).
- **Burstiness** = 한 context(하나의 sequence) 안에서 같은 class가 우연 이상으로 다시 나타나는 성질. Bursty sequence는 query class 3번 + 다른 class 3번(가장 빈번한 label을 찍는 지름길을 막는 decoy). Burstiness가 **있다**는 것이 뭉침이 있다는 뜻이고, batch(여러 sequence를 묶은 gradient 단위)와는 다른 층위(Q11).
- **Long tail** = 드문 class가 매우 많음. 실험에서는 class 수를 100 → 1,600 → 12,800으로 늘렸고, 이때 각 class의 확률은 균등(uniform)했다. 확률이 균등해도 class 종류가 많으면 개별 class는 절대적으로 드물다.
- **Zipfian marginal** = 별도 실험에서 확률 자체를 불균등하게(`p ∝ x^-α`) 만든 것. α=1, M=12,800에서 상위 10개 class가 29.2%, 나머지 약 71%가 long tail. α=3이면 상위 3개가 97%.
- **Dynamic meaning** = 같은 item이 여러 label에 매핑되거나(label multiplicity), 한 class 안 변동이 큼(within-class variation). 둘 다 ICL을 높임.
- **결과**: burstiness와 많은 class 수가 **둘 다** 있어야 ICL이 생김(§3.1). 균등 class 빈도에서는 ICL과 IWL이 상쇄 관계, α=1 Zipfian에서는 상위 10개 class에 한해 둘 다 유지되고 드문 class는 어떤 α에서도 외워지지 않음(§3.2).
- 이미지 분류 실험이라 text corpus로 검증되지 않았음(Open question). see [[read]] §3.

## Q9. ICL과 IWL은 무엇이고 어떻게 측정하나?

**Kernel.** ICL 평가: 학습 때 본 적 없는 holdout class로 4-shot 2-way 분류, label을 매번 0/1로 무작위 재배정(chance 1/2). 외워둘 수 없으므로 context의 예시만으로 풀어야 한다. IWL 평가: 학습된 class인데 context에 그 class의 예시가 전혀 없음(chance 약 1/1,600). weight에 저장된 정보만으로 풀어야 한다. 즉 "ICL↑ IWL↓"는 모델 전략의 무게중심이 "외워서 쓰기"에서 "context를 그때그때 비교하기"로 옮겨갔다는 뜻. see [[read]] §3.

## Q10. Long tail이 길다는 건 class가 equally distributed라는 뜻인가?

**Kernel.** §3의 long-tail 실험(class 수 100→1,600→12,800)은 각 class 확률이 균등했고, 꼬리가 긴 이유는 skew가 아니라 class 종류의 개수다. 확률을 의도적으로 불균등하게 만든 것은 별도의 Zipfian 실험이다. 일상적 의미의 long tail(소수가 지배하고 다수가 드묾)은 Zipfian에 가깝다.

## Q11. Burstiness는 batching의 문제인가?

**Kernel.** 아니다. Burstiness는 하나의 context 안의 반복 구조이고, batch는 서로 다른 sequence를 묶는 단위라 각 sequence는 forward pass에서 독립적이다. 다만 sequence 구성(packing, intra-document masking, 문서 내부 반복 제거)은 context 내용을 바꾸므로 burstiness에 영향을 줄 수 있다는 것이 이 course의 해석(Interpretation). §3 Implication은 이것을 Open question으로 남긴다(corpus 수준 작업이 LLM pretraining의 ICL을 바꾸는지 논문이 시험하지 않았음). ch-04 §2(packing과 document mask)와 연결. see [[read]] §3, [[ch-04]].

## Q12. Multi-round 데이터에서 같은 instruction에 대한 반응이 얼마나 나왔는지 확인할 때 Zipfian을 쓸 수 있나? (2026-09-28)

**Kernel.** 재려는 대상에 따라 §3의 개념이 다르다. (1) instruction의 전체 빈도 분포 → Zipfian marginal(빈도 × 순위가 거의 일정하면 α≈1; 예: 빈도 438/219/146/109/88은 각 곱이 약 438). (2) 한 대화 안에서 같은 instruction의 재등장 → burstiness(multi-turn 한 대화 = 한 context). (3) 같은 instruction에 서로 다른 응답 → dynamic meaning의 label multiplicity(item=instruction, label=응답) — 질문의 의도에 가장 가까운 것. 한계: (a) "class"를 정하는 기준(문자열 일치/embedding clustering/intent 분류)이 결과를 좌우하며 소스에 없음, (b) §3의 개념은 Omniglot에서 조작한 변수이지 text 데이터 진단 도구로 검증되지 않음, (c) 이 통계가 SFT 결과와 어떤 관계인지 미시험(Open question) — 기술 통계(audit)로는 유용하나 일반화 예측 지표로는 근거 없음. ch-04의 (user, assistant) 쌍 dedup 논의와 연결: instruction 빈도(marginal) + 묶음 내 응답 다양성(label multiplicity)의 결합은 이 course의 제안이며 미검증(Interpretation). see [[read]] §3, [[ch-04]].

## Q13. §5 Emergent ability의 정의와 "emergence scale은 immutable하지 않다"는 무슨 뜻인가? (2026-09-28)

**Kernel.** Emergent = 작은 모델에서는 그 task 성능이 random 근처이다가, 어느 크기 이상에서 random보다 훨씬 높아지는 능력(§5 정의, Wei et al.). Training FLOPs 대비 성능 plot이 "평평하다가 급상승" 모양. Table 1 예: GPT-3의 3-digit 덧셈은 13B(2.3E+22), MMLU 평균은 175B(3.1E+23), PaLM의 WiC는 540B(2.5E+24)에서 random을 넘기 시작. "immutable하지 않다": BIG-Bench 14개 task에서 PaLM 62B는 random 위인데 LaMDA 137B·GPT-3 175B는 random 근처 — 저자들이 더 좋은 데이터(다국어·코드 비중)와 architecture 차이를 가능한 이유로 들지만 ablation 없음(Interpretation). 문제: loss는 compute에 부드럽게 줄어드는데(ch-08a) 능력은 급상승하면 작은 모델 실험으로 예측 불가. 저자들도 cross-entropy는 exact match/accuracy가 random인 scale에서 이미 개선됨을 인정(App. A.1) — §6(지표 착시 반박), §7(loss threshold 관점)으로 이어짐. §5는 "저자들의 주장"을 소개하는 절이지 emergence의 실재를 결론짓는 절이 아님. see [[read]] §5.

## Q14. Burstiness가 있으면 좋은가? 다른 성질들은? (2026-09-28)

**Kernel.** "좋다"는 목표에 따라 다르다. §3 결과: burstiness↑ → ICL↑ IWL↓(Fig. 2, 단독 불충분: burstiness와 많은 class 수가 둘 다 필요); long tail(class 100→1,600→12,800, p(bursty)=0.9) → ICL↑ IWL↓(Fig. 3); dynamic meaning(label multiplicity, within-class variation) → 각각 ICL↑(Figs. 4-5, IWL 효과는 장에 기술 없음); 균등 class 빈도 → ICL/IWL 상쇄(둘 다 유지하는 모델 없음); Zipfian α=1 → 상위 10개 class에 한해 ICL·IWL 동시 유지(드문 class는 어떤 α에서도 외워지지 않음, Fig. 6). IWL은 데이터 성질이 아니라 반대편 능력(자주 쓰는 지식을 weight에 저장). ICL이 목표면 앞의 성질들이 유리하나 IWL도 필요한 범용 모델에서 극단화가 좋다는 근거는 없음. 한계: Omniglot 단일 연구, text 코퍼스 미검증(Open question); 저자들은 supervised 데이터의 uniform 재균형이 ICL 기회를 놓칠 수 있다고 해석(Interpretation). see [[read]] §3.

## Q15. ICL vs IWL 중 무엇이 더 좋은가? (2026-09-28)

**Kernel.** 이 장에 둘의 순위를 매긴 결과는 없다. 평가가 서로 다른 시험이라 점수를 직접 비교할 수 없음(ICL: holdout class, context 예시 있음, chance 1/2 / IWL: 학습 class, context 예시 없음, chance 약 1/1,600). ICL은 새롭거나 드문 대상·weight 변경 없는 즉시 반영에 맞고, IWL은 자주 나오는 대상·context 없이 사용하는 경우에 맞음. §3: 균등 빈도에서는 상쇄(둘 다 유지하는 모델 없음), Zipfian α=1에서는 상위 10개 class는 IWL·드문 class는 ICL로 분업, 드문 class는 어떤 α에서도 외워지지 않음. §2의 ICL 근거: task별 학습 데이터 불필요, spurious correlation 주장(GPT-3 저자들); 한계: WiC 49.4%(random), ANLI 175B 미만 약 33%(chance), "그 자리에서 배우는가 vs 알아보는가"는 Open question. 범용 모델은 둘 다 필요하다는 것과 LLM 행동의 ICL/IWL 분류는 이 course의 해석(Interpretation, 미검증). see [[read]] §2, §3.
