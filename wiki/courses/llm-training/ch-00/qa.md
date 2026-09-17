<!-- ch-00 reading Q&A; back to [[read]] -->

## Q1. Pre-training, mid-training, SFT, preference optimization, RL은 무엇이 다른가?

차이는 주로 데이터 분포와 목적에 있다. Pre-training은 넓은 문서에서 모든 다음 토큰을 예측해 언어·지식·형식의 기반을 만들고, mid-training은 그 체크포인트에 특정 도메인·추론·긴 문맥용 데이터를 계속 적용해 필요한 능력을 보강한다. SFT는 `user` 입력과 좋은 `assistant` 응답의 예시에서 assistant 응답 토큰만 맞히게 해 대화 행동을 가르친다. Preference optimization은 같은 입력의 선호·비선호 응답 쌍으로 선호 응답의 상대 확률을 높이고, RL은 검증기·보상 모델의 점수를 최대화하도록 생성 결과를 바꾼다. 평가는 가중치를 갱신하지 않고 각 단계의 개발·미공개 스위트 점수를 측정한다.

## Q2. ChatML 같은 역할 토큰은 언제 가르치는가?

대화 형식은 보통 SFT에서 명시적으로 가르친다. 예를 들어 `<|im_start|>user`와 `<|im_start|>assistant`를 넣은 전체 대화를 토큰화하되, 손실은 assistant 응답 토큰에만 적용한다. 사전학습 웹 데이터에 유사한 문자열이 섞일 수는 있지만 그 사실에 의존하면 안 된다. 훈련과 추론에서 같은 chat template과 종료 토큰을 써야 하며, 학습 전에는 토큰화한 배치를 decode해 역할·종료 토큰과 손실 마스크 위치를 확인한다.

## Q3. Tool calling과 long context는 어느 단계에서 가르치는가?

둘 다 특정 단계에 자동으로 속하지 않는다. 긴 문맥은 pre-training에서 길이와 장거리 의존성을 일부 얻지만, 긴 입력·검색·다문서 추론 데이터와 긴 컨텍스트 윈도우를 목표로 한 mid-training을 별도로 실행하는 경우가 많다. Tool calling은 보통 SFT에서 도구 스키마, 호출 형식, 인자, 관찰 결과 뒤의 다음 행동을 포함한 궤적을 가르치고, 성공 여부를 실행으로 검사할 수 있으면 RLVR/RL로 보강한다. 전용 장: [[ch-32c]], [[ch-26]], [[ch-27]].

## Q4. 추론은 언제 가르치는가?

Pre-training은 수학·코드·설명·증명 텍스트의 다음 토큰을 예측하면서 문제 구조와 풀이 패턴의 기반을 만든다. Mid-training은 목표 영역의 고품질 문제·풀이·긴 문맥 자료로 그 기반을 보강할 수 있다. SFT는 `문제 → 단계별 풀이 → 최종 답` 시연을 통해 풀이를 출력하는 형식과 전략을 가르친다. 정답을 프로그램·수식·단위 테스트 등으로 자동 판정할 수 있으면 RLVR/RL이 맞는 답으로 이어지는 더 긴 탐색을 강화한다. 단계별 풀이를 출력한다고 해서 내부 추론이 자동으로 정확해지는 것은 아니므로, 최종 답 정확도와 미공개 문제의 전이를 함께 평가한다.

## Q5. Mid-training 데이터는 보통 어떤 형식인가?

단일 표준은 없지만, 보통 각 행이 하나의 텍스트 문서인 JSONL/Parquet 레코드이며 학습 때는 `<eos>`로 구분해 긴 토큰 시퀀스로 packing하고 모든 다음 토큰에 causal-LM loss를 적용한다. 레코드는 긴 논문·책·코드 저장소·수학 풀이·도메인 문서처럼 목표 능력에 맞는 원문일 수 있다. `messages` 대화 형식과 assistant-only loss는 보통 SFT의 신호다. Mid-training에 대화나 도구 궤적을 넣을 수는 있지만 그때도 모든 토큰을 예측하면 계속 사전학습이고, assistant 응답만 예측하면 SFT에 가까운 학습이다.

## Q6. Mid-training 데이터를 처음 pre-training에 넣어도 되는가?

가능하다. 같은 문서를 초기 혼합에 넣는 것은 금지되지 않으며, 실제 pre-training 혼합에도 코드·수학·논문 같은 목표 도메인이 들어간다. Mid-training은 보통 그 데이터의 비중, 문맥 길이, 품질 기준, 학습률을 뒤 단계에서 바꾸는 설계다. 희소한 고품질 데이터는 처음부터 높은 비중으로 반복하면 일찍 과적합하거나 일반 텍스트·다른 능력을 희생할 수 있고, 너무 낮은 비중이면 목표 능력이 충분히 생기지 않는다. 따라서 같은 데이터의 `처음부터 혼합`과 `base checkpoint 뒤에 집중`을 같은 총 토큰·평가 설정으로 비교해 개발·미공개 점수, 망각, 중복률을 보고 결정한다.

## Q7. 데이터를 몇 번 반복해도 되는가?

고정된 안전 횟수는 없다. 데이터 원천별 평균 반복은 `그 원천에서 학습에 샘플된 총 토큰 ÷ 그 원천의 고유 토큰`으로 계산하고, 문서별 최대 반복과 near-duplicate까지 따로 기록한다. 넓은 pre-training에서는 거대한 원천을 한 번도 완전히 보지 못할 수 있고, 희소한 고품질 원천은 여러 번 샘플될 수 있다. 반복 비중을 올릴 때마다 같은 시점의 원천별 held-out loss, 목표 능력의 개발·미공개 평가, 일반 능력과 망각 지표를 비교한다. 훈련 loss만 계속 내려가고 원천 held-out loss나 미공개 점수가 나빠지면 반복을 줄이거나 데이터를 새로 보강한다.

## Q8. Reasoning을 직접 다루는 장은 무엇인가?

핵심 장은 [[ch-24]]로, chain-of-thought·long chain-of-thought·step-level reasoning trace를 생성·필터·검증하고 SFT에 쓰는 방법과 수학 밖 전이를 다룬다. 보완 장은 reasoning trace 증류인 [[ch-20]], reasoning 목적 mid-training인 [[ch-32]]와 [[ch-32a]], reasoning SFT의 마스킹·템플릿 설정인 [[ch-30]], 정답 검증과 보상 설계인 [[ch-44]] 및 [[ch-45]]다. Chat template의 thinking token 설계는 [[ch-04]]에서 다룬다.

## Q9. 일반 능력은 여러 벤치마크의 평균 점수인가?

아니다. 여러 벤치마크는 서로 다른 능력 축을 관측하기 위한 도구일 뿐이다. 이 과정에서 일반 능력은 학습에서 겨냥하지 않았고 개발 결정에도 쓰지 않은 과제·영역에서 성능이 유지되거나 향상되는 성질이다. 따라서 축별 미공개 점수와 불확실성을 프로필로 보고, 평균은 요약값으로만 쓴다. 목표 축의 점수 상승이 다른 비목표 축의 유의미한 하락을 상쇄하면 모델은 평균이 올라가도 좁아진 것이다. 일반화를 주장하려면 개발 세트와 독립된 미공개 평가, 새 형식 또는 새 과제군, 오염 점검이 필요하다.

## Q10. 새 벤치마크에서 모델이 낮은 점수를 받으면?

그 가능성이 일반화 평가의 목적이다. 개발에 쓰지 않은 새 벤치마크의 낮은 점수는 그 형식·과제군·도메인으로의 전이가 아직 입증되지 않았다는 증거다. 단, 바로 능력 부족으로 결론 내리지는 않는다. 먼저 프롬프트와 chat template, 답 추출·채점 규칙, 난이도와 오류율, 오염 여부를 확인하고 같은 설정에서 강한 기준 모델과 비교한다. 평가가 유효하고 여러 독립적인 새 벤치마크에서도 같은 하락이 나오면 그 능력 축의 일반화 공백으로 보고, 다음 학습 실험을 설계한다. 그 벤치마크 점수를 보고 데이터를 고르면 이후에는 미공개 평가가 아니라 개발 평가가 된다.

## Q11. Capability coverage map은 프런티어 회사의 전체 관심사인가?

아니다. 이 과정의 map은 지식, 추론, 코드, 지시 따르기, 다중 턴, 긴 문맥, 다국어, 도구·에이전트, 사실성·캘리브레이션, 안전을 포함하지만 완전한 목록도 단일 회사의 우선순위표도 아니다. 회사는 출시 목적에 따라 latency·비용·throughput, privacy·저작권·데이터 동의, 악용 위험·보안, 제품 UX, 신뢰성·가용성, 모델의 운영 가능성도 별도 목표로 둔다. map의 목적은 “이 모델이 어디서 잘하는가”와 “무엇을 아직 측정하지 않았는가”를 드러내는 것이며, coverage가 없는 축에는 일반화 주장을 하지 않는 것이다.

## Q12. Instruction Following은 사실·추론 질문에 답하는 것과 무엇이 다른가?

넓은 뜻의 사용자 요청은 모두 지시처럼 보이지만, 평가 축으로서 instruction following은 내용의 정답성과 독립된 형식·제약·행동 규칙 준수다. `1+1은?`에 `2`라고 답하는 것은 주로 산술·지식 평가다. `1+1을 계산하되 JSON 한 줄로, key는 answer만 쓰고, 숫자는 문자열로 쓰며, 다른 글자는 쓰지 말라`에 `{"answer":"2"}`라고 답하는 것은 산술과 instruction following을 동시에 요구한다. `2`는 내용은 맞지만 지시 따르기에는 실패다. IFEval류 평가는 내용 지식의 어려움을 줄이고 단어 수, 목록 개수, 금지어, 특정 접두사·형식 같은 검증 가능한 제약 준수를 측정한다.

## Q13. Development suite와 unseen suite는 무엇인가?

Development suite는 데이터 혼합, 학습률, 알고리즘, 체크포인트 같은 선택을 할 때 반복해서 읽는 평가 묶음이다. Unseen suite는 같은 능력 축을 재지만 최종 선택이 끝날 때까지 점수를 보지 않는 독립 평가 묶음이다. 둘 다 학습 데이터에서 제거하고 오염을 점검해야 한다. 개발 점수가 오르면 그 benchmark에 맞춘 선택의 효과와 능력 향상이 섞일 수 있으므로, unseen 점수까지 같은 방향으로 변해야 그 능력 안의 전이 증거가 된다. unseen 점수를 한 번 보고 다음 실험 선택에 쓰면 그때부터 development suite가 된다.

## Q14. Public benchmark도 unseen suite가 될 수 있는가?

될 수 있다. unseen은 benchmark의 공개 상태가 아니라 특정 팀이 그 점수를 모델·데이터·하이퍼파라미터 선택에 쓰지 않았다는 운영 상태다. 예를 들어 Tülu 3 연구는 개발용과 구분해 MMLU-Pro, AGIEval English, DeepMind Mathematics, BigCodeBench-Hard, IFEval-OOD 같은 공개 benchmark를 미공개 평가로 배정했다. 그러나 다른 팀이 이미 그것을 개발에 썼거나 모델 학습 데이터에 문항·해설이 들어갔으면 같은 benchmark는 그 팀의 unseen 평가가 아니다. 공개 정적 benchmark는 시간이 갈수록 오염·적응 위험이 커지므로, 문항 출처·오염 검사·점수 열람 로그를 기록하고 중요한 출시 결정에는 새로 만든 private 또는 live 평가를 함께 쓴다.

## Q15. Tülu 3 8B SFT data ablations에서 transfer, same sign, opposite sign은 무엇인가?

각 수는 `최종 SFT 점수 − 해당 데이터 원천을 뺀 SFT 점수`다. 따라서 양수는 그 원천을 넣었을 때 점수가 오른 것이고, 음수는 넣었을 때 점수가 떨어진 것이다. 개발 변화 `Δ_dev`와 미공개 변화 `Δ_uns`가 모두 양수 또는 모두 음수이면 same sign이다. `Δ_uns`도 잡음 구간을 넘을 때만 그 원천의 효과가 unseen 평가로 transfer했다고 말한다. 부호가 다르면 opposite sign이며, 개발에서의 개선이 미공개 평가에서는 악화로 바뀐 것이다. Persona의 instruction-following 결과는 `+19.2` 대 `−0.4`로 숫자 부호는 반대지만 미공개 변화가 매우 작아 “opposite-sign harm”보다 “does not transfer(전이 증거 없음)”로 읽는다. WildChat의 `+2.7` 대 `−3.2`는 두 변화가 모두 의미 있다면 진짜 opposite sign이다. 원시 점수 수준이 아니라 각 비교의 변화량을 비교하는 이유는 dev와 unseen benchmark의 난이도와 점수 척도가 다르기 때문이다.

## Q16. Persona data와 WildChat이 너무 overfit한 것인가?

그렇게 단정할 수 없다. Persona 결과는 IFEval의 25개 개발 제약에서는 큰 이득이지만 IFEval-OOD의 새 제약에서는 이득이 확인되지 않았다는 것으로, 그 instruction-following 형식 안에서의 개발-benchmark 과적합 또는 전이 부재를 시사한다. WildChat의 부호 반전은 새 제약에서 해로울 가능성을 시사하지만, Table 32의 각 ablation은 한 번의 학습 실행이라 시드 변동이 보고되지 않았다. 또한 결과는 instruction following 한 축에 한정된다. “데이터가 너무 많이 반복되어 memorization했다”는 반복 과적합, “모델 전체가 나빠졌다”는 일반 망각, “IFEval 형식에만 특화됐다”는 benchmark-format 과적합은 서로 다른 주장이다. 여기서 뒷받침되는 것은 마지막 주장에 대한 제한된 증거다.

## Q17. Unseen benchmark가 약하면 benchmark 문항을 학습에 넣지 않고 어떻게 개선하는가?

문항·정답·근접 패러프레이즈는 학습에 넣지 않는다. 먼저 낮은 점수를 능력 구성요소로 분해한다. 예를 들어 instruction following이면 제약의 조합, 길이, 위치, 다중 턴 유지, JSON·도구 형식 같은 원인을 가설로 세운다. 그 구성요소를 덮되 evaluation의 템플릿·문항·정답과 분리된 새 훈련 예시를 만들고, source/template/task-cluster 수준에서 train·development·unseen을 나눈다. 개발 세트로 데이터 혼합을 고른 뒤, 보관한 unseen 과제군에서 한 번 평가한다. 그 점수를 본 뒤 해당 suite는 다음 라운드의 development 자료가 되므로, 새 unseen suite를 만든다. 학습 데이터와 각 평가 split 사이의 exact·near-duplicate·paraphrase 오염 검사를 기록한다.

## Q18. Data selection 때마다 대형 학습과 benchmark 평가를 반복해야 하는가? Prismatic Synthesis는 무엇인가?

모든 후보를 대형 모델로 끝까지 학습하지 않는다. 일반적인 절차는 값싼 문서·응답 scoring과 dedup으로 후보를 줄이고, 작은 proxy 모델 또는 짧은 anneal로 소수의 혼합을 비교하며, development suite에서 상위 후보를 고른 뒤, 큰 학습의 최종 후보만 unseen suite에서 확인하는 것이다. 작은 규모의 순위가 큰 규모로 전이하는지는 별도 가설이므로 proxy와 target 규모의 비교가 필요하다. [[ch-22]]의 Prismatic Synthesis는 아마 질문의 “Prism”으로, embedding 공간의 다양성만 보지 않고 example gradient의 방향·다양성을 G-Vendi로 측정해 synthetic SFT 후보를 고르는 방법이다. 이것은 평가를 대체하지 않는다. 선택 heuristic이므로 random-selection baseline과 held-out 평가에서 실제 이득을 검증해야 하며, 이 과정은 ch-10a(quality filtering), ch-13(mixture weights), ch-17(작은 filter/mixture ablation), ch-22에서 다룬다.

## Q19. Benchmark item sampling error는 무엇인가?

benchmark의 n개 문항은 가능한 모든 문제의 표본이므로, 같은 능력의 다른 n개 문항을 뽑으면 점수가 달라진다. 0/1 정확도 p의 표준오차는 `sqrt(p(1-p)/n)`이고 95% 구간은 대략 `p ± 1.96·SE`다. 예를 들어 65% 정확도에서는 100문항의 SE가 4.77점(95% 약 ±9.3점)인 반면, 14,042문항 MMLU의 SE는 0.40점(95% 약 ±0.78점)이다. 두 모델은 같은 문항에서 비교해 문항별 결과가 상관되므로 paired SE를 써야 하며, 같은 문항에서 둘 다 맞거나 틀린 경우가 많을수록 차이 추정의 오차가 작아진다. HumanEval처럼 164개 과제만 있는 benchmark에서는 2~4점 차이가 표본 오차 안일 수 있다. 여러 질문이 한 지문·언어·문서에 묶이면 문항 독립 가정이 깨져 cluster SE가 필요하다. 이것은 같은 문항으로 재학습했을 때의 seed variance, 같은 모델의 prompt-template 변경 효과와는 다른 잡음 원인이다.

## Q20. Benchmark 샘플 수가 많으면 모델을 더 신뢰할 수 있는가?

문항 수가 많으면 특정 benchmark에서 관측한 점수와 두 모델의 점수 차이를 더 정확히 추정할 수 있다. 이는 모델 전체의 신뢰성이나 다른 능력 축으로의 일반화를 자동으로 보장하지는 않는다. 큰 n은 sampling error만 줄인다. 따라서 특정 데이터 원천 또는 학습 설정의 효과를 주장할 때는 문항 수·문항의 독립성·paired SE와 함께 다음을 확인한다.

1. **Seed 재현성:** 같은 개입을 다른 학습 seed로 반복했을 때 평균 변화의 부호와 크기가 유지되는가? 유지되지 않으면 recipe 효과가 아니라 학습 run 변동일 수 있다.
2. **Prompt-format 견고성:** 의미가 같은 두 개 이상 prompt/chat template에서 변화 부호가 유지되는가? 유지되지 않으면 모델 능력보다 평가 형식에 맞춘 효과일 수 있다.
3. **오염 검사:** training·mid-training·SFT 데이터에 평가 문항, 해설, near-duplicate, paraphrase가 없는가? 있으면 점수는 기억 효과를 포함한다.
4. **미공개 전이:** development suite에서 고른 개입이 개발 중 보지 않은 같은 능력 축의 unseen suite에서도 개선되는가? 그렇지 않으면 development-benchmark gain으로 기록한다.
5. **비목표 축의 보존:** 목표 점수 상승과 함께 지식·코드·안전·긴 문맥 등 비목표 축이 잡음보다 크게 떨어지지 않는가? 떨어지면 넓어진 것이 아니라 일부 능력과 교환한 것이다.

모든 초기 후보에 이 다섯 검사를 같은 비용으로 적용할 필요는 없다. 값싼 proxy/짧은 anneal과 development suite로 후보를 먼저 줄이고, 큰 학습의 최종 후보에 대해 seed·unseen·축별 검사를 강화한다.

## Q21. Narrowing signals와 starting checkpoint 대비 forgetting은 무엇인가?

Narrowing signals는 학습 뒤 목표 점수만 좋아지고 모델이 제공할 수 있는 능력의 범위·다양성·안전성이 줄어드는지를 찾는 공통 진단이다. §7은 (1) 시작 체크포인트 대비 비목표 축의 forgetting, (2) pass@1은 올라도 pass@k가 떨어지는 탐색 가능성 감소, (3) 출력 다양성 감소, (4) 안전한 요청의 over-refusal, (5) train-eval 오염으로 부풀린 점수, (6) 특정 답 형식·제약에만 맞춘 과적합을 보고한다. Forgetting은 개입 전 checkpoint θ₀와 개입 뒤 θ₁을 같은 문항·동일 prompt·동일 채점 설정으로 비교해, 개입이 겨냥하지 않은 축에서 `score(θ₁) − score(θ₀)`가 잡음보다 유의미하게 음수인 경우다. 이는 문자 그대로 지식이 지워졌다는 뜻만은 아니다. 출력 형식 변화로 채점기가 답을 못 읽은 경우도 있으므로, 큰 하락에서는 raw output과 채점 과정을 검사해야 한다. 예: math-only SFT가 수학 평균 27.7→49.8을 올리면서 IFEval 69.2→42.3을 낮춘 경우는 목표 수학 향상과 instruction-following forgetting을 함께 보고해야 한다.

## Q22. SFT 중 시작 모델의 다른 능력은 어떻게 보존하는가? Replay가 답인가?

Replay는 직접적인 방법이다. target SFT batch와 함께 시작 모델이 이미 잘하던 일반 instruction·코드·안전·지식 예시 또는 pretraining text를 섞어, 그 예시의 response loss 또는 next-token loss도 최적화한다. 목적은 `L = L_target + λ·L_replay`처럼 target 향상과 일반 분포 보존을 함께 최적화하는 것이다. 고정 비율은 없으므로 replay share, learning rate, epoch 수를 바꿔 target gain 대 non-target drop 곡선에서 선택한다. 보완책은 LoRA처럼 update 자유도를 제한하기, 작은 learning rate·early stopping, 시작 모델이 낸 답을 target에 가까운 rewrite로 쓰는 self-distillation/on-policy data, fixed general prompts에서 시작 모델 logits와의 KL 또는 cross-entropy를 제어하기, fine-tuning 뒤 시작 가중치와 weight interpolation을 하는 것이다. Replay도 만능은 아니다. 너무 작은 일반 샘플은 능력 축을 덮지 못하고, replay 비중이 너무 크면 target을 못 배운다. 따라서 input checkpoint와 output checkpoint를 held-out 비목표 suite에서 paired 비교해 검증한다. 자세한 근거와 recipe는 [[ch-30a]]다.

## Q23. pass@1과 pass@k로 forgetting을 검사할 수 있는가?

부분적으로 가능하다. code·math처럼 정답을 자동 검증할 수 있는 생성 과제에서 input checkpoint와 output checkpoint의 pass@1, pass@k, 그리고 문제별 solved-set을 비교한다. post-training 뒤 pass@1은 올라도 큰 k에서 시작 모델만 풀 수 있는 문제가 남거나 pass@k가 내려가면, 모델이 예전에 샘플할 수 있던 해결 경로·출력 다양성을 잃은 narrowing 신호다. 하지만 pass@k는 sampling 가능한 정답이 있는 능력만 측정한다. knowledge recall, safety refusal, instruction 제약 준수, 긴 문맥 같은 축의 forgetting은 해당 held-out suite의 paired per-item score를 직접 비교해야 한다. §7의 AIME24 예시는 k=1024에서 base만 푼 문제가 13.3%였으므로 RLVR가 pass@1을 높여도 base의 전체 해결 범위를 완전히 포함하지 않는다는 뜻이다.

## Q24. RL은 base model이 이미 아주 낮은 확률로 풀 수 있는 문제의 pass@k를 높이는가?

핵심은 맞지만 `pass@1`과 `pass@k`를 구분해야 한다. base가 정답 경로를 매우 낮은 확률 q로 이미 생성할 수 있으면 pass@1은 q이고 pass@k는 `1−(1−q)^k`다. verifier RL은 정답 경로의 확률 질량을 높여 q를 올리므로 pass@1을 크게 올리고 같은 유한 k의 pass@k도 올릴 수 있다. 이 현상은 새 reasoning strategy를 발명했다기보다 이미 있던 경로를 더 자주 샘플하게 만든 sampling-efficiency 개선일 수 있다. [[rlvr-beyond-base-model]]은 더 큰 k에서 base가 RLVR 모델을 다시 앞서는 경우를 보여, RLVR가 base의 희귀한 해법 일부를 0 확률에 가깝게 만들어 coverage를 줄일 수 있다고 주장한다. 반대 증거인 [[prorl]]은 충분히 오래 RL하면 새 영역으로 확장될 수 있다고 주장하므로, pass@1만으로 어느 해석도 확정하지 않는다.
