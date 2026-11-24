<!-- chapter: ch-34
     track: sft
     kind: content
     title: Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports
     deps: [ch-33]
     sources: [[qwen-2.5]], [[qwen-2.5-recipe]], [[qwen-long-context-synth]], [[online-merging-optimizer]], [[qwen-3]], [[qwen-3-hybrid-thinking]], [[qwen3-2507-instruct-thinking-split]], [[qwen3-coder]], [[qwen-3-5]], [[qwen-3-5-model-card]], [[olmo-2]], [[olmo-2-tulu-recipe]], [[open-instruct-allenai-recipes-recipe]], [[olmo-3]], [[olmo-3-model-flow]], [[allenai-olmo3-open-instruct-scripts]], [[allenai-olmo3-open-instruct-scripts-recipe]], [[phi-3]], [[phi-4]], [[phi-4-reasoning-sft-rl]], [[gsm1k]], [[emergence-loss-perspective-recipe]], [[tulu-3]], [[llama-3-recipe]], [[smollm-3]], [[smollm3-training-configs]], [[deepseek-v3-recipe]]
     figures: figures/lab-compare.html
     revised: 2026-09 (generality revision)
-->

# Chapter 34 — 사례 연구 B: Qwen, OLMo, Phi 보고서에서의 일반성 대 전문화

> **핵심 통찰.** 세 report family 각각은 broad training이 specialized score를 낮추거나 specialized data가 broad score를 낮춘 stage를 최소 하나씩 제시한다. Qwen3-32B에서 thinking-mode fusion과 general-RL stage는 IFEval을 73.0에서 85.0으로, in-house ToolUse를 63.3에서 85.5로 올렸다. 반면 thinking-mode AIME'24는 83.8에서 81.4로, LiveCodeBench v5는 68.4에서 65.7로 떨어졌다(각 stage당 점수 하나). 저자들은 versatility를 위해 이를 받아들였고, 2025년 7월 Qwen은 hybrid Qwen3-235B-A22B 대신 분리된 Instruct 및 Thinking model을 배포했다([[qwen-3-hybrid-thinking]] Table 22; [[qwen3-2507-instruct-thinking-split]]). Phi-4의 data ablation에서 web data 없이 훈련한 13B model은 phi-3-medium 대비 MATH 4.9점, HumanEval 12.1점을 얻었지만 TriviaQA 14.8점을 잃었다. Olmo 3 7B Instruct RL은 AIME 2024를 23.5에서 44.3으로 올리면서 PopQA는 20.7에서 14.1로 낮췄다([[phi-4-reasoning-sft-rl]] Table 3; [[olmo-3-model-flow]] Table 26). 인쇄된 SFT peak learning rate는 1e-6(Phi-4)부터 1e-4(병합한 Olmo 3 32B Think run 두 개 중 하나)까지다. Companion figure의 SFT ledger grid에 있는 model row 11개 중 batch는 7개, packing은 8개, checkpoint-selection rule은 8개에서 보고되지 않았다.
>
> **지침.** Stage가 mode 또는 capability family(non-thinking answer, tool use, open-ended chat)를 추가하면, Qwen3의 thinking-mode AIME'24와 LiveCodeBench v5가 general score 상승과 함께 single-run stage table에서 2.4점 및 2.7점 하락한 것처럼, stage 전후의 모든 mode에서 이전 stage의 specialist benchmark를 다시 실행하라. 작은 model이 더 강한 teacher와 tokenizer를 공유하고 target이 math와 code라면, small model에서 RL을 하는 대신 teacher logits에 대한 on-policy distillation을 사용하라. Qwen3-8B에서 이는 1,800 GPU hour로 RL의 17,920 GPU hour 대비 AIME'24 74.4, RL 67.6에 도달했고, RL이 90.0에 머문 AIME'24 pass@64를 90.0에서 93.3으로 올렸다([[qwen-3-hybrid-thinking]] Table 21). 그 외에는 RL을 사용하고 큰 k의 pass@k를 추적하라. 다른 base에 SFT recipe를 재사용할 때는 learning rate를 sweep하라. OLMo 2는 자기 base가 Llama 3.1 recipe보다 높은 learning rate를 필요로 했다고 보고한다([[olmo-2-tulu-recipe]] §5). Synthetic 또는 distilled data가 public benchmark를 올리면, Phi family 전반에서 GSM8k-over-GSM1k gap을 찾은 GSM1k처럼, training data 이후에 나온 evaluation set 또는 match된 fresh set을 추가하라([[gsm1k]] §5.1).

## 이 장이 범용 모델에 중요한 이유

범용 모델은 pre-training, mid-training, SFT, preference optimization, RL을 거치며, 각 stage는 target capability를 올리면서 다른 capability를 낮출 수 있다. [[ch-33]]은 generalist 하나를 최적화한 Tülu 3와 Llama 3를 따라갔다. 이 장은 각 stage에서 다른 선택을 한 세 family를 읽는다.

1. **Qwen**은 thinking 및 non-thinking mode, tool use, multilingual task용 checkpoint 하나를 훈련한 뒤 선택을 일부 되돌린다. Qwen3 report는 thinking과 non-thinking behaviour를 checkpoint 하나에 fuse하고, Qwen3-2507 release는 다시 분리하며, Qwen3.5는 checkpoint 하나에서 두 mode를 제공한다.
2. **OLMo**는 모든 stage(data, script, intermediate checkpoint)를 release하므로 stage effect를 table에서 읽고 config로 재현할 수 있다.
3. **Phi**는 구성한 training data에 의존한다. Phi-4에는 synthetic pre-training 및 mid-training data를, Phi-4-reasoning에는 o3-mini의 teacher trace를 사용한다. Overfitting check와 knowledge loss를 모두 보고한다.

각 family에 대해 이 장은 어느 stage가 어느 benchmark를 올렸는지, 어느 benchmark가 떨어졌는지, lab이 gain이 memorized된 것이 아니라 general한 것인지 어떻게 시험했는지를 묻는다. 마지막에는 case-study model과 Tülu 3, Llama 3, SmolLM3, DeepSeek-V3의 SFT-stage ledger를 두며, 보고되지 않은 모든 cell을 표시한다. Companion figure [figures/lab-compare.html](figures/lab-compare.html)은 reader가 report table과 stage 두 개를 고르게 하고, sign으로 정렬한 benchmark별 변화를 그리며, ledger를 reported/not-reported/conflicting cell의 grid로 보여준다.

## §1 일반성을 위해 report 읽기: stage delta와 그 noise

**정의.** *Stage delta*는 `Δ_b = s_b(after) − s_b(before)`, 즉 같은 pipeline의 checkpoint 두 개 사이 benchmark b score의 변화다. *Specialization cost*는 stage가 target으로 삼지 않은 benchmark에서, targeted benchmark가 오르는 동안 관찰되는 negative Δ다.

**문제.** Final score만으로는 어느 stage가 capability를 낮췄는지 알 수 없으므로, final score만 가진 reader는 어느 stage를 바꿔야 할지 결정할 수 없다.

**절차.**

1. 여러 stage에서 같은 benchmark를 보고하는 table을 찾는다. Qwen3 Table 22, Qwen3 Table 21, OLMo 2 Table 16, Olmo 3 Table 26, Phi-4 Table 9, Phi-4-reasoning Table 2, Qwen2.5-1M Table 6이다.
2. Stage의 data와 reward에 따라 각 benchmark를 targeted 또는 not targeted으로 표시한다.
3. 모든 benchmark의 Δ를 계산하고 sign으로 정렬한다.
4. `|Δ|`를 그 benchmark size 및 sampling protocol의 run-to-run noise와 비교한다.
5. Benchmark가 training data에 나타날 수 있는지, fresh 또는 perturbed set이 존재하는지 확인한다.

**Noise formula.** Question `n`개로 된 benchmark에서 각 question을 `k`개 sample의 mean accuracy로 score하고, question i에서 model의 success probability를 `p_i`라고 하자. Score의 sampling standard error는 다음과 같다.

    SE = sqrt( Σ_i p_i(1 − p_i) / (n² k) )

여기서 `n`은 question 수, `k`는 question당 sample 수, `p_i`는 question i에서의 success probability다. 이 식은 sampling noise만 세며, training run 사이의 차이(seed, data order)는 여기에 더해지고 포착되지 않는다.

**계산 예시.** Qwen3는 AIME을 question 30개, question당 sample 64개로 score한다([[qwen-3-hybrid-thinking]] §4.6). 30개 question 모두 `p = 0.82`(Stage 2 score 83.8에 가까움)라면, `SE = sqrt(30 × 0.82 × 0.18 / (900 × 64)) = 0.0088`, 즉 0.88점이다. 고정된 mean score에서 동일한 `p_i`는 `Σ_i p_i(1 − p_i)`를 최대로 만들므로 0.88은 이 sampling error의 upper bound다. Independent evaluation 두 개의 차이는 `SE = 0.88 × √2 = 1.24`점이다. Reported thinking-mode Stage 2에서 Stage 4까지 변화는 `81.4 − 83.8 = −2.4`점으로, 이 standard error의 약 1.9배다. Question 하나는 `100/30 = 3.33`점이므로, −2.4점은 sample에 걸쳐 평균한 question 0.72개와 같다.

**Noise에 관한 근거.** Phi-4-reasoning report는 자기 model 및 OpenAI와 DeepSeek model을 포함한 비교 대상 모두에서 average-of-5 run 두 개가 “can differ significantly (by up to 5-10 percentage points on AIME)”라고 말한다. 또한 Phi-4-reasoning의 예로 63.1 (6.3)처럼 standard deviation을 붙여 AIME 2025를 50 run의 pass@1로 보고한다([[phi-4-reasoning-sft-rl]], arXiv:2504.21318 §1, §5.1, Table 1). **결과(단일 연구).**

**조건과 한계.** Qwen3 Table 22는 stage당 score 하나를 출력하고 variance를 제공하지 않으며, 위 formula는 run-to-run training noise를 제외한다. 그러므로 −2.4점 변화는 실제 하락과 양립하지만 table만으로 확립되지는 않는다. 저자들 자신의 해석은 §3에 있다.

**함의.** 범용 모델의 stage gate에는 small benchmark의 반복 evaluation(Qwen3처럼 question당 sample 64개, 또는 Phi-4-reasoning처럼 independent run 50개)과, 측정된 noise보다 더 크게 떨어지면 안 되는 not-targeted benchmark 목록이 필요하다.

## §2 Qwen2.5: 검증된 post-training 설정과 Online Merging Optimizer

**Pipeline.** Qwen2.5 open-weight model(0.5B-72B)은 18T token으로 pre-training하며, 마지막 stage에서 context length를 4,096 → 32,768 token으로, RoPE base를 10,000 → 1,000,000으로 바꾼다([[qwen-2.5-recipe]], arXiv:2412.15115v2 §3.1, §3.3). Post-training은 세 stage로 구성된다.

1. **SFT**: “over 1 million SFT examples”, epoch 2회, sequence length 32,768, learning rate는 “gradually decreased from 7 × 10⁻⁶ to 7 × 10⁻⁷”, weight decay 0.1, gradient clipping 1.0이다(§4.1). 그러나 §1에서는 SFT, DPO, GRPO 전체 post-training data가 “amounts to 1 million examples”라고 하므로, example count는 report 내부에서 충돌한다. Batch, packing, loss masking은 보고하지 않았다.
2. **Offline RL**: SFT model은 새 math, coding, instruction-following, logical-reasoning query에 대한 response를 resample한다. Execution feedback 또는 answer matching을 통과한 response를 선택하고 failure는 reject한다. Pair는 약 150,000개이며, Online Merging Optimizer를 사용해 learning rate 7 × 10⁻⁷에서 epoch 1회 DPO를 한다(§4.2). DPO `β`는 보고하지 않았다.
3. **Online RL**: Query당 response 8개, global batch 2048, episode당 sample 2048(episode당 query 256개에서 유도)을 쓰는 GRPO다. Reward-model score variance가 더 큰 query를 먼저 처리한다(§4.3).

Long SFT를 두 stage로 나누는 방식(instruction은 최대 32,768 token, 그 뒤 short와 long을 섞되 최대 262,144 token)은 Qwen2.5-72B-Instruct가 아니라 proprietary MoE model Qwen2.5-Turbo에 속한다(§4.4). Open Qwen2.5-7B/14B-Instruct-1M model은 같은 two-stage SFT와 최대 8,192 token pair에 대한 offline RL을 사용한다([[qwen-long-context-synth]], arXiv:2501.15383v1 §4). Qwen2.5-72B-Instruct의 score는 MMLU-redux 86.8, HumanEval 86.6, MATH 83.1, IFEval(strict prompt) 84.1이다. MMLU 86.1 수치는 base model의 것이다([[qwen-2.5]] Tables 2, 6).

Report의 measurement statement 두 개는 generality와 관련된다. Contamination filter는 test sequence와 training sequence의 longest common subsequence 길이가 ≥ 13이고 shorter sequence의 ≥ 0.6일 때 해당 training sequence를 제거한다(§5). Reward-model section은 RL number 없이 “a higher score on RM benchmarks does not necessarily correlate with superior performance of the resulting RL model”이라고 말한다(§5.2.3).

**Online Merging Optimizer: 정의.** Preference training에서 매 step마다 gradient-based update와 SFT model 및 pre-trained model 간 parameter difference를 결합하는 optimizer다([[online-merging-optimizer]], arXiv:2405.17931v1 §4).

**문제.** *Alignment tax*는 RLHF 중 pre-training 및 SFT ability가 사라지는 현상이다. RLHF model을 훈련 후 한 번 SFT reference와 merge하면 benchmark score는 회복되지만 MT-Bench와 AlpacaEval 2.0은 낮아진다(§3).

**메커니즘(OnDARE variant).**

1. Preference loss에 대한 통상의 Adam update Δθ를 계산한다.
2. Δθ의 각 entry를 확률 p로 남기고 나머지는 0으로 한다(random sparsification, rescaling 없음; §4.2).
3. SFT delta `τ_r = θ_r − θ_b`에도 같은 일을 한다.
4. Sparse vector 두 개의 weighted sum을 적용한다.

    θ^(t) = θ^(t−1) + (1 − α)·F_R(Δθ) + α·F_R(τ_r)

여기서 `θ^(t)`는 step t의 policy parameter, Δθ는 Adam update, `θ_r`는 SFT reference, `θ_b`는 pre-trained model, `F_R`은 reserve rate p의 Bernoulli keep-mask, `α`는 merging weight다(“Larger α introduces stronger regularization”, §4.2).

**계산 예시.** Readability를 위해 paper grid인 1e−7~1e−4보다 과장한 `α = 0.1`을 쓰고, parameter 4개에서 `Δθ = (0.40, −0.20, 0.10, 0.30)`, `τ_r = (1.0, 0.5, −0.5, 0.0)`이라고 하자. Mask가 Δθ에서는 entry 1, 2, 4를, `τ_r`에서는 entry 1, 3, 4를 남긴다고 하자. Update는 `0.9 × (0.40, −0.20, 0, 0.30) + 0.1 × (1.0, 0, −0.5, 0) = (0.46, −0.18, −0.05, 0.27)`이다. Entry 3의 plain Adam update는 SFT direction(−0.5)에 반대인 +0.10이었지만, merged update는 그 방향을 따르는 −0.05다.

**근거.** UltraFeedback DPO에서 OnDARE는 AdamW 대비 benchmark average / MT-Bench / AlpacaEval 2.0 LC를 Qwen1.5-1.8B-Chat에서 +0.5 / +0.24 / +0.05, Qwen1.5-7B-Chat에서 +1.1 / +0.12 / +0.28, LLaMa-3-8B-Instruct에서 +1.3 / +0.19 / +1.57만큼 바꿨다. Offline linear merge는 각각 +0.2 / −0.03 / −0.96, −0.2 / −0.28 / −1.65, −1.2 / +0.11 / +0.02였다(Table 1). **결과(단일 연구)**이며, cell당 run 하나다.

**조건과 한계.** 1.8B-8B parameter에서 시험했다. 논문의 prose는 benchmark average가 `α = 5e−7`에서 peak라고 말하지만, Table 2는 peak(41.8)를 `5e−6`에서 보인다. Qwen2.5 report는 optimizer 이름만 언급하며 `α`, p, variant, ablation을 제공하지 않는다([[qwen-2.5]] §4.2). 따라서 Qwen2.5에서의 effect는 측정되지 않았다. 일반적인 merging method는 [[ch-30c]]에서 다룬다.

**함의.** Preference training이 SFT 대비 benchmark average를 낮춘다면, SFT delta 쪽으로 매 step 끌어당기는 방식은 시험된 control 하나다. Table 1의 benchmark-average gain은 0.5에서 1.3점이므로 retention benchmark를 측정해야 한다.

## §3 Qwen3: hybrid thinking, stage trade-off, 2507 split

**정의.** *Hybrid thinking* model은 하나의 weight 집합에서 prompt flag로 선택하여 explicit reasoning block을 포함하거나 직접 답한다.

**메커니즘.** Flagship model(Qwen3-235B-A22B, Qwen3-32B)은 post-training을 네 stage로 수행한다([[qwen-3-hybrid-thinking]], arXiv:2505.09388v1 §4.1-4.4).

1. **Long-CoT cold start.** Qwen2.5-72B-Instruct는 verify하기 어려운 query 또는 chain of thought 없이도 올바르게 답하는 query를 제거한다. QwQ-32B는 N개 candidate를 생성한다. Wrong answer, repetition, guesswork, thinking/summary inconsistency, language mixing, validation item과의 similarity가 있으면 response를 제거한다. Report는 “both the number of training samples and the training steps”를 최소화하라고 권한다. Size와 hyperparameter는 보고하지 않았다.
2. **Reasoning RL.** Query-verifier pair 3,995개로 GRPO를 수행했다. Qwen3-235B-A22B의 AIME'24는 170 step에 걸쳐 70.1에서 85.1로 올랐다.
3. **Thinking-mode fusion.** Stage 2 model에 대한 continual SFT다. Thinking data는 Stage 1 query에서 Stage 2 model이 rejection-sample한 것이고, non-thinking data는 coding, math, instruction following, multilingual task, writing, QA, role-play를 포괄한다.
4. **General RL.** 20개 초과 task의 reward를 쓴다. Instruction following, mode flag를 포함한 format following, preference alignment, 실제 environment를 상대로 multi-turn tool use를 하는 agent ability, RAG가 대상이다.

Fusion template(Table 9)는 non-thinking answer에서도 빈 reasoning block을 유지한다.

```
<|im_start|>user                     <|im_start|>user
{query} /think<|im_end|>             {query} /no_think<|im_end|>
<|im_start|>assistant                <|im_start|>assistant
<think>                              <think>
{thinking content}
</think>                             </think>

{response}<|im_end|>                 {response}<|im_end|>
```

Thinking이 user가 설정한 budget에 도달하면 thinking을 중단하고 stop-thinking instruction인 `Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n`을 삽입한다. Report는 이 budget behaviour가 fusion에서 “is not explicitly trained but emerges naturally”라고 말한다(§4.3).

**Trade-off의 근거(Qwen3-32B, Table 22, stage당 score 하나).**

| Benchmark | Stage 2(thinking) | Stage 4(thinking) | Δ | Stage 4(non-thinking) |
|---|---:|---:|---:|---:|
| IFEval strict prompt | 73.0 | 85.0 | +12.0 | 83.2 |
| ToolUse(in-house) | 63.3 | 85.5 | +22.2 | 86.5 |
| CounterFactQA(in-house) | 50.4 | 68.1 | +17.7 | 66.4 |
| BFCL v3 | 69.0 | 70.3 | +1.3 | 63.0 |
| MMLU-Redux | 91.4 | 90.9 | −0.5 | 85.7 |
| GPQA-Diamond | 68.8 | 68.4 | −0.4 | 54.6 |
| AIME'24 | 83.8 | 81.4 | −2.4 | 31.0 |
| LiveCodeBench v5 | 68.4 | 65.7 | −2.7 | 31.3 |

저자들은 “for challenging tasks like AIME'24 and LiveCodeBench, the performance in thinking mode actually decreases after these two training stages. We conjecture this degradation is due to the model being trained on a broader range of general tasks, which may compromise its specialized capabilities in handling complex problems. During the development of Qwen3, we choose to accept this performance trade-off to enhance the model's overall versatility.”라고 쓴다. Score는 **결과(단일 연구)**이고, 원인에 대한 부분은 **해석**이다.

**분리.** 2025-07-21에 Qwen은 “we decided to stop using hybrid thinking mode. Instead, we'll train Instruct and Thinking models separately so we can get the best quality possible”라고 발표했다([[qwen3-2507-instruct-thinking-split]]). Qwen3-235B-A22B-Instruct-2507 card는 model이 “supports only non-thinking mode”라고 말하고, Thinking-2507 card는 “supports only thinking mode”라고 말한다. Original hybrid의 non-thinking mode와 비교하면 Instruct-2507은 AIME25에서 24.7 대비 70.3, SimpleQA에서 12.2 대비 54.3이다. 그러나 card는 더 많은 “long-tail knowledge coverage”도 목록에 두므로, 이는 hybrid와 separate training의 controlled test가 아니다. Qwen3.5-397B-A17B card는 다시 checkpoint 하나에서 양 mode를 제공하며, thinking은 default이고 `enable_thinking: False`로 비활성화한다([[qwen-3-5-model-card]]). **열린 질문**: 어떤 Qwen source도 data를 고정한 hybrid training의 cost를 측정하지 않는다.

**함의.** Mode switch도 평가할 capability다. 각 mode의 specialist benchmark, mode-following test(Qwen의 in-house ThinkFollow는 Stage 3 뒤 88.7에서 Stage 4 뒤 98.9로 올랐다), thinking-mode loss의 크기를 보고하라.

## §4 Qwen3 strong-to-weak distillation

**정의.** *Off-policy distillation*은 teacher가 생성한 response를 cross-entropy로 student에 훈련한다. *On-policy distillation*은 student에서 response를 sample하고, 그 response에서 student의 token distribution이 teacher의 것을 맞추도록 훈련한다.

**문제.** Smaller model 여섯 개(0.6B-14B dense 및 30B-A3B) 각각에 네 stage를 실행하면 development compute가 든다. Report는 logit distillation이 “requir[es] only 1/10 of the GPU hours compared to the four-stage training method”라고 말한다(§4).

**메커니즘**(§4.5)은 다음과 같다.

1. Off-policy phase: `/think` 및 `/no_think` mode 모두에서 teacher output을 SFT target으로 삼으며, 이것이 reasoning과 mode switching을 가르친다.
2. On-policy phase: “the student model produces responses in either /think or /no_think mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence.” 이 절은 이 phase에 reward model이 있다고 설명하지 않는다.

**수식.** Student sample `y ~ π_S(·|x)`의 각 position t에서 token-level KL objective는 다음과 같다.

    L = Σ_t D_KL( π_S(·|x, y_<t) ‖ π_T(·|x, y_<t) ) or Σ_t D_KL( π_T(·|x, y_<t) ‖ π_S(·|x, y_<t) )

여기서 `π_S`는 student, `π_T`는 teacher, x는 prompt, `y_<t`는 student 자신의 prefix다. Report는 어느 direction을 쓰는지, full vocabulary를 쓰는지 logits subset을 쓰는지 밝히지 않는다.

**계산 예시.** Token 세 개로 된 한 position에서 teacher는 (0.7, 0.2, 0.1), student는 (0.4, 0.4, 0.2)를 준다고 하자. Reverse KL은 `0.4 ln(0.4/0.7) + 0.4 ln(0.4/0.2) + 0.2 ln(0.2/0.1) = −0.224 + 0.277 + 0.139 = 0.192 nats`다. Forward KL은 `0.7 ln(0.7/0.4) + 0.2 ln(0.2/0.4) + 0.1 ln(0.1/0.2) = 0.392 − 0.139 − 0.069 = 0.184 nats`다. 어느 쪽이든 student는 이 position의 세 token 모두에 target을 받는다. 반면 verifier를 쓰는 RL은 sample한 response 전체에 scalar 하나를 준다.

**근거(Qwen3-8B, math 및 code query만, 두 run은 같은 off-policy checkpoint에서 시작; Table 21).** AIME'24는 RL로 55.0 → 67.6, on-policy distillation으로 55.0 → 74.4다. AIME'25는 42.8 → 55.5 및 42.8 → 65.5, LiveCodeBench v5는 42.0 → 52.9 및 42.0 → 60.3, MMLU-Redux는 86.4 → 86.9 및 86.4 → 88.3, GPQA-Diamond은 55.6 → 61.3 및 55.6 → 63.3이다. AIME'24 pass@64는 RL 뒤 90.0에 머물고 distillation 뒤 93.3으로 올랐다. GPU hour는 RL 17,920, distillation 1,800(ratio 9.96)이다. **결과(단일 연구).**

**조건과 한계.** 비교는 math와 code query를 쓴다. Knowledge 및 chat retention은 MMLU-Redux와 GPQA-Diamond으로만 대표된다. Teacher logit에 대한 token-level KL은 student와 teacher가 vocabulary를 공유해야 하며, Qwen3 model은 모두 entry 151,669개의 같은 tokenizer를 쓴다(§2). Report는 student가 teacher를 넘는 경우를 시험하지 않는다. RL run의 setting은 보고되지 않았으므로, 이 table은 RL baseline이 tuning되었는지 보여줄 수 없다. Lab이 teacher data를 어느 지점에 넣고 어떻게 선택하는지는 [[ch-35]] 및 [[ch-35a]]에서 다룬다.

**함의.** Smaller general model family에 대해 flagship에서 하는 on-policy distillation은 report가 제시한 lower-cost route다. 저자들은 pass@64 상승을 student의 exploration space가 확장된 것으로 읽는다(§4.7, **해석**). 이는 AIME'24와 AIME'25에서만 측정했다.

## §5 Qwen lineage의 agentic training과 공개 reference

**Qwen3 general RL.** Stage 4에는 “Agent Ability”가 있으며, 여기서 “the model is allowed to perform complete multi-turn interaction cycles with real environment execution feedback”이라고 한다(§4.4). Table 22에서 in-house ToolUse는 Stage 4 동안 thinking에서 70.4 → 85.5, non-thinking에서 73.2 → 86.5로 올랐고, BFCL v3는 68.4 → 70.3 및 61.5 → 63.0으로 올랐다. BFCL multi-turn evaluation은 YaRN으로 64k context에서 model을 deploy한다(§4.6).

**Qwen3-Coder(2025년 7월 blog).** Pre-training은 code 비율 70%인 7.5T token을 사용하며 “preserving general and math abilities”라고 설명한다. Automatically test-cased task를 쓰는 code RL은 “brought gains to other tasks”를, long-horizon agent RL은 “20,000 independent environments in parallel”을 사용했다고 한다([[qwen3-coder]]). Blog는 general 또는 cross-task claim의 number를 인쇄하지 않으므로, 이 source에서는 이 claim에 controlled evidence가 없다(official source, no measurement).

**Qwen3.5(2026년 2월 model card).** Card는 method detail 없이 “Reinforcement learning scaled across million-agent environments with progressively complex task distributions”라고 주장한다([[qwen-3-5-model-card]]). Benchmark note는 evaluation setting을 정의한다. Search agent는 “a simple context-folding strategy(256k): once the cumulative Tool Response length reaches a preset threshold, earlier Tool Responses are pruned from the history”를 쓴다. BrowseComp에서 같은 model은 context folding으로 69.0, DeepSeek-V3.2와 Kimi K2.5가 쓴 “discard-all” strategy로 78.6을 얻는다. 같은 checkpoint가 context policy 두 개 사이에서 9.6점 차이가 나므로, agent score는 model과 context policy를 함께 기술한다(**해석**). Context management는 [[ch-45c]]에서 다룬다.

**공개 reference: Olmo 3 Instruct.** Dolci Instruct SFT에는 tool-use prompt 227,579개가 있다(Table 30). Function-calling trajectory는 실제 MCP tool을 쓰는 Science QA trajectory 22.6K개(function 8개, multi-step 42.3%), Web Search QA 6.6K개(function 3개, multi-step 76.1%), simulated SimFC trajectory 200K개(function 42.6K개, multi-turn 42.3%, multi-step 23.8%)를 포함한다(Table 27). 저자들은 unified format(OpenAPI tool definition, XML tag 안의 pythonic call, dedicated special token, environment role)이 “crucial for stable and high-quality tool-use behavior”였다고 보고한다(§5.2.1). 7B model은 SFT → DPO → RL 전체에서 BFCL이 48.9 → 49.6 → 49.8, tool을 쓴 LitQA2가 38.0 → 43.3 → 38.2로 움직였다(Table 26)([[olmo-3-model-flow]]).

**함의.** 이 report들의 tool-use gain은 고정된 scaffold와 context policy를 동반한다. 범용 agent에는 적어도 context policy 두 개 아래의 같은 benchmark와, agent RL 동안 non-agent benchmark가 하락하지 않았다는 check가 필요하다.

## §6 OLMo 2와 Olmo 3: 공개 stage, release된 config, 측정된 stage effect

**OLMo 2**는 4,096-token sequence length와 long-context stage 없이 7B, 13B, 32B model을 훈련한 뒤(arXiv:2501.00656v3 Table 3), Tülu 3 recipe를 적용한다([[olmo-2-tulu-recipe]] §5).

- **SFT.** 7B와 13B에는 `tulu-3-sft-olmo-2-mixture`(prompt 939,104개)를, 1B와 32B에는 `-0225`(prompt 866,138개)를 쓴다. 후자는 date cutoff를 언급하는 synthetic prompt를 제거하고, completion 5개가 majority에 도달하는 math item만 유지한다. Multilingual subset을 제거하면 average가 약 0.5점 낮아져 유지했다. Learning rate를 sweep했다(7B: 1e-5, 선택한 2e-5, 3e-5; 13B: 선택한 5e-6; 32B: 선택한 4e-6). Configuration당 seed 1개, final configuration에는 최대 4개를 썼다. Table 17에서 epoch 3회와 learning rate 4e-6일 때 sum loss는 development average 49.76, mean loss는 48.25다.
- **DPO.** Development SFT model과 20-model pool에서 나온 on-policy response를 GPT-4o가 rate한다. 가장 높은 rating의 completion을 chosen으로, 임의의 더 낮은 rating completion을 rejected로 둔다.
- **RLVR.** 7B 및 13B는 GSM8K, MATH, constraint prompt에서 PPO를 쓴다. 13B는 math score가 development model보다 낮아진 뒤 RLVR pass 두 번(GSM8K, 그다음 MATH)을 추가로 해야 했다. 1B와 32B는 GRPO를 쓴다.

Report는 “OLMo 2 required significantly higher learning rates compared to the Llama 3.1 training recipe”라고 말한다(§5). **결과(단일 연구).**

OLMo 2 32B의 stage effect(Table 16)는 다음과 같다.

| Stage | AVG | AlpacaEval 2 | GSM8K | IFEval | MATH | Safety | PopQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT | 61.7 | 16.9 | 78.4 | 72.4 | 35.9 | 93.8 | 35.4 |
| DPO | 68.8 | 44.1 | 85.7 | 83.8 | 46.8 | 91.9 | 36.4 |
| Instruct(RLVR) | 68.8 | 42.8 | 87.6 | 85.6 | 49.7 | 85.9 | 37.5 |

Average는 RLVR로 변하지 않지만 safety는 6.0점 낮아진다. 13B에서는 safety가 SFT 94.3에서 Instruct 89.7로 낮아진다.

OLMo 2는 mid-training run도 평균낸다. Text는 서로 다른 data order의 anneal 3개가 “consistently” best single run과 같거나 더 낫다고 말한다(Table 14). 그러나 mix E에서 soup은 GSM* 43.0이고 best single run은 60.5다. Prose summary와 table은 24개 cell 중 하나에서 맞지 않는다.

**Olmo 3**는 모든 stage의 이름을 밝힌다([[olmo-3-model-flow]], arXiv:2512.13961v2). Dolma 3 Mix(5.9T token)로 pre-training하고, sequence length 8,192에서 네 layer 중 세 layer에 sliding-window attention을 쓴다. 이어 Dolmino mid-training(100B token), Longmino로 65,536 token까지 extension(7B는 50B token, 32B는 100B token; full-attention layer에만 YaRN 적용), 그다음 Dolci SFT, DPO, RL을 수행한다. 7B Think의 release된 SFT launch script는 다음과 같다(open-instruct@d8a7f1c `scripts/train/olmo3/7b_think_sft.sh` L7-19, [[allenai-olmo3-open-instruct-scripts-recipe]]).

```bash
CHECKPOINT=/weka/.../long-context/olmo25_7b_lc_64k_6T_M100B_..._yarn-fullonly_50B-fb13a737/step11921/
LR=5e-5
python src/scripts/train/sft/OLMo2-7B-sft.py train \
    olmo2.5-6T-LC-sigma-reasoning-mix-decontam-v2-special-tokens-v3-think-FIX \
        $CHECKPOINT \
        ai2/jupiter-cirrascale-2 \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --global_batch_size=1048576 \
```

Dolci Think SFT는 7B에서 prompt 2,268,468개를 쓴다(Table 17). Training은 “document packing instead of padding”, 7B에서 1M-token batch, epoch 2회, token 32,768을 사용한다(부록 A.6.1). Checkpoint는 evaluation suite로 score한 learning-rate sweep, 이어 “vibe-test” question으로 선택한다. Final thinking SFT checkpoint는 “a linearly weighted merge of two checkpoints trained with different learning rates”다(§4.2; Table 47은 32B에서 1.0e-4와 5.0e-5를 soup한 것을 목록으로 든다).

Olmo 3 finding 두 개는 imitation과 memorization의 한계를 다룬다. 첫째, “further supervised finetuning on thinking traces generated by Qwen3 32B ... outright hurts the performance of Olmo 3 Think SFT”다. Lab은 대신 그 completion을 더 나쁜 response와 pair로 만들고, length-normalized DPO로 훈련한다(released script와 Table 48의 `β = 5`)(§4.3, 부록 A.6.2; open-instruct@00d42c4 `7b_think_dpo.sh` L32-37). 둘째, RL-Zero는 pre-training, mid-training, evaluation set에 대해 decontaminate한 data에서 base로부터 직접 RLVR을 훈련하며, negative control을 실행한다. 같은 prompt에서 random binary reward로 훈련하면 “does not improve performance on any of our benchmark” suite다(§6.2, 그림 27). Motivation은 data가 공개되지 않은 base에서는 contamination이 “makes spurious rewards as effective as true rewards”이기 때문이다(§6). Mixed-domain RL-Zero run은 모든 domain을 개선했지만 “each domain ... under-optimized compared to the single-domain setup”으로 남았다(§6.2).

Olmo 3 7B Instruct(Table 26, run 세 개의 average)에서 SFT → DPO → RL은 AIME 2024를 6.7 → 23.5 → 44.3, LiveCodeBench v3를 20.0 → 18.8 → 29.5, IFEval을 81.7 → 82.0 → 85.6, MMLU를 67.1 → 69.1 → 69.1, PopQA를 16.5 → 20.7 → 14.1, AlpacaEval 2 LC를 21.8 → 43.3 → 40.9, safety를 89.5 → 89.9 → 87.6으로 움직였다.

**함의.** Release된 script와 stage table은 reader가 stage를 재현하고 score가 바뀐 위치를 찾게 한다. Random-reward control은 RL-Zero gain이 memorized evaluation data를 eliciting한 것인지 시험한다(control은 Instruct RL stage가 아니라 RL-Zero에서 실행했다). Instruct stage table은 reasoning이 오르는 RL stage에서 knowledge recall(PopQA)과 safety가 떨어질 수 있음을 보인다.

## §7 Phi-3와 Phi-4: synthetic data 비중, overfitting check, factual recall

**데이터.** Phi-3 report는 순차적인 pre-training phase 두 개를 설명한다. Phase 1은 주로 web data, phase 2는 synthetic data와 결합한 “more heavily filtered web data (a subset of phase 1)”다. Token split이나 generator model은 제공하지 않는다([[phi-3]], arXiv:2404.14219v4 §2). 이후 Phi-4 report는 phi-3의 phase 2를 “primarily synthetic tokens and a much smaller allocation for ultra-filtered and reasoning-heavy web data”라고 특징짓는다(arXiv:2412.08905v1 §3.1). Phi-3-mini(3.8B, 3.3T token)는 TriviaQA에서 64.0이며 Mixtral 8x7B의 82.2보다 낮다. 저자들은 이를 factual knowledge에 대한 제한된 capacity 탓으로 돌린다([[phi-3]] §3, §6).

Phi-4(14B, 약 10T token)는 pre-training과 mid-training에서 synthetic data “about 400B unweighted tokens”을 사용한다([[phi-4-reasoning-sft-rl]], §2.2-§3). §3.1의 finding 두 개는 trade-off를 설명한다. Fixed token budget에서 synthetic subset 12 epoch는 fresh web token을 더 쓴 4 epoch보다 MMLU가 높았다(그림 2). 또한 “Models trained only with synthetic data underperformed on the knowledge-heavy benchmarks and demonstrated increased hallucinations”였다. Web data 없이 훈련한 13B ablation은 phi-3-medium 대비 MMLU +0.8, MATH +4.9, HumanEval +12.1, TriviaQA −14.8로 변했다(Table 3). Mixture ablation에서 “the only benchmark that shows a clear benefit from web data is TQA”였고, lab은 targeted 및 knowledge-heavy web data를 “to improve knowledge benchmarks” 추가했다. Chosen mixture보다 synthetic 비중이 큰 mix 두 개는 “marginally better”였다(§3.2, Table 4). **결과(단일 연구).**

**GSM1k: 정의.** LLM assistance 없이 사람이 작성한 grade-school math problem 1,205개로 된 set이다. Solve rate, solution step, answer magnitude를 GSM8k와 match했고 public release에서 withheld했다([[gsm1k]], arXiv:2405.00332v4 §1, §3).

**문제.** GSM8k score에는 memorized test item이 포함될 수 있다. Match된 fresh set은 그 gap을 직접 측정한다.

**수식.** Paper는 각 model을 two-proportion Z-test로 시험한다.

    z = (p̂₁ − p̂₂) / sqrt( p̂(1 − p̂)(1/n₁ + 1/n₂) ), p̂ = (n₁p̂₁ + n₂p̂₂)/(n₁ + n₂)

여기서 `p̂₁`, `p̂₂`는 GSM8k와 GSM1k accuracy, `n₁`, `n₂`는 problem 수, `p̂`는 pooled accuracy다.

**계산 예시.** Phi-3-mini-4k-instruct는 GSM8k에서 0.788, GSM1k에서 0.748을 얻는다(부록 F). `n₁ = 1,319` GSM8k test problem([[emergence-loss-perspective-recipe]] 부록 B Table 6), `n₂ = 1,205`이면 `p̂ = (1,319 × 0.788 + 1,205 × 0.748)/2,524 = 0.769`, `SE = sqrt(0.769 × 0.231 × (1/1,319 + 1/1,205)) = 0.0168`, `z = 0.040/0.0168 = 2.38`이다. Paper는 `z = 2.385`, `p = 0.009`를 인쇄한다. `P(Z > 2.38) = 0.0086`이므로, appendix가 test를 two-tailed이라고 부르지만 인쇄된 p는 upper-tail value다.

**근거.** “Several families of models, including the Phi and Mistral families of models, show systematic tendencies to perform stronger on GSM8k compared to GSM1k for almost every release and scale”(§5.1)이며, gap은 8%에 이른다(Abstract). Phi-2는 6% 하락하지만 여전히 GSM1k의 절반 초과를 푼다(§5.3). GSM8k log-likelihood는 gap과 상관된다(Spearman 0.36, p = 0.03). 저자들은 contamination이 “is likely not the full story”라고 결론내린다(§5.4). **결과(단일 연구).**

**조건과 한계.** Checkpoint의 판정은 양방향 모두에서 prompt format에 따라 달라진다. Worked example의 Phi-3-mini-4k-instruct는 standard prompt에서 gap 0.040(p = 0.009), alternative prompt에서 0.007(p = 0.318)을 보인다. Phi-3-mini-128k-instruct는 standard prompt에서 0.011(p = 0.260), alternative prompt에서 0.035(p = 0.014)을 보인다(부록 F). Score는 report 자체 setting이 아니라 LM Evaluation Harness 5-shot format을 쓴다.

**Phi-4의 fresh test.** Phi-4는 training data를 모두 모은 뒤인 2024년 11월 6일 또는 그 이후 release된 question 78개로 된 2024년 11월 AMC-10/12 contest에서 temperature 0.5로 평가했으며, 150개 중 평균 91.8을 얻었다(§1.1, 부록 C; 91.8은 그림 1 bar label이다). 부록 C는 question당 generation 10개라고 하지만 그림 1 caption은 run 100개라고 하므로 sample count는 report 내부 conflict다. Footnote 8은 final candidate 세 개가 모두 89를 넘었고, final model은 “before measuring its score but after seeing the scores for the other two candidates” 선택했다고 밝힌다. **결과(단일 연구).** Final choice 전에 확인한 candidate score는 contest set을 model selection의 일부로 만들기 때문에 이 disclosure가 중요하다.

**Factual recall과 abstention.** Post-training data는 model이 답을 모를 때 refusal을 가르친다. SFT는 model이 보통 틀렸던 question에 refusal을 pair로 두고, DPO는 Phi-4가 때때로 정답을 낸 question에서는 refusal보다 correct answer를, 때때로 틀린 답을 낸 question에서는 wrong answer보다 refusal을 선호한다(§4.4, 부록 A.1). SimpleQA에서 base model은 question의 90.0%에 incorrectly answer한다. SFT model은 38.7%를 incorrectly answer하고 57.5%에는 attempt하지 않는다. Final model은 3.0%를 correctly answer, 15.8%를 incorrectly answer, 81.1%에는 attempt하지 않는다(그림 6). Simple-evals F1은 “gives our base model a higher score than our final model”이다. Report는 invented biography 같은 factual hallucination을 weakness로 든다(§8).

**함의.** Phi-4 ablation에서 web data를 synthetic data로 대체하면 math와 code score는 오르고 TriviaQA는 낮아졌다. Large synthetic share를 쓰는 범용 recipe에는 knowledge benchmark, match된 또는 post-dated reasoning set, accuracy와 별도로 보고한 abstention count가 필요하다.

## §8 Phi-4 post-training과 Phi-4-reasoning: pivotal token, teachable prompt, transfer

**Pivotal Token Search(PTS): 정의.** Sampled solution에서 correct final answer의 추정 확률이 threshold `p_gap` 이상 변하는 token을 찾아, chosen/rejected response가 같은 prefix 뒤의 single token인 DPO pair를 만드는 방법이다([[phi-4-reasoning-sft-rl]], arXiv:2412.08905v1 §4.3).

**문제.** Full-length chosen response에서는 correctness와 무관한 low-probability token도 gradient를 받고, chosen response 안의 harmful token도 positive signal을 받는다(§4.3).

**메커니즘.**

1. Query Q와 sampled solution `t₁…t_n`에 대해 completion을 sample하고 oracle로 검사하여 `p(success | t₁…t_i)`를 추정한다.
2. 각 segment의 success probability 변화가 threshold `p_gap` 미만이거나 segment가 token 하나가 될 때까지 sequence를 재귀적으로 나눈다. Phi-4 data에서 쓴 `p_gap`은 보고하지 않았다. 그림 3 illustration은 `p(success)`를 ≥ 0.2만큼 바꾸는 token을 표시한다.
3. 변화가 `p_gap` 이상인 single token은 pivotal이다. Query와 prefix가 DPO prompt가 되고, `p(success)`를 올리거나 낮추는 token `t_acc`, `t_rej`가 pair가 된다.
4. `0.2 ≤ p(success) ≤ 0.8`인 question만 유지한다.

**계산 예시.** `p(success) = 0.50`인 prefix 뒤에서 sampling이 next token “divide”에는 0.85, “multiply”에는 0.25를 준다고 하자. Pair는 (prefix, “divide”)를 (prefix, “multiply”)보다 선택하는 것이다. 아래 negative-feedback section은 DPO가 이러한 pair의 probability를 어떻게 움직이는지 계산한다.

**근거(Table 9).** SFT에서 DPO stage 1(PTS)로 GPQA는 47.3 → 53.6, MATH는 77.1 → 80.5가 되었지만 IFEval은 66.2 → 63.0으로 낮아졌다. Judge-guided DPO stage 2(약 850k GPT-4o-labeled pair) 뒤 final model은 GPQA 56.1, ArenaHard 75.4(66.5에서)를 얻었지만 DROP은 86.1에서 75.5로 낮아졌다. Stage 2만 실행하면 GPQA 52.4, MATH 77.6이었다. 저자들은 PTS는 reasoning-heavy task에, judge-guided DPO는 GPT-4-judged ArenaHard에 도움이 된다고 결론낸다(§4.5). **결과(단일 연구).**

**Phi-4-reasoning(2025년 4월).** Phi-4-reasoning은 SFT만 쓰고, Phi-4-reasoning-plus는 RL을 추가한다(arXiv:2504.21318v1 §1).

- **Teachable prompt.** Lab은 “at the edge of Phi-4's current abilities”인 seed를 target으로 한다. Verifiable answer가 없으면, strong reference model의 plurality answer와 weaker model(Phi-4 또는 GPT-4o)이 동의하는 비율로 difficulty를 추정한다. Rubric-based LLM evaluator도 reasoning step의 수와 complexity를 score한다(§2.1).
- **Teacher.** High effort의 o3-mini는 medium effort보다 강한 teacher다. Medium effort는 “similar effect to DeepSeek-R1”이면서 token efficiency는 더 높다(§3.2).
- **SFT.** Prompt-response pair 1.4 million 초과, unique token 8.3B, global batch 32 및 32K context로 약 16K step, AdamW, [1e-6, 2e-5] grid에서 최고인 learning rate 1e-5, warmup step 450, weight decay 1e-4, “2+ passes over reasoning data sources”, trained token 16B를 쓴다(§3, §3.1-3.2). Data-mixture weight는 data cluster당 epoch이며, domain별로 tune한 후 결합한다(§3.1).
- **Decontamination.** AIME-2024, GPQA, LiveCodeBench, SimpleQA, DROP, ArenaHard를 포함해 목록으로 든 서로 다른 benchmark 23개에 대해 decontaminate한다. AIME-2025는 data 이후에 나온다(§2.2).

**Transfer 근거.** 두 model은 training에서 직접 target으로 삼지 않았다고 설명한 TSP, 3SAT, BA-Calendar에서 Phi-4보다 “30 to 60 percentage points” 향상한다(§1). General benchmark(Table 2)에서 Phi-4 → Phi-4-reasoning은 IFEval Strict 62.3 → 83.4, FlenQA 82.0 → 97.7, ArenaHard 68.1 → 73.3, MMLUPro 71.5 → 74.3이다. 같은 table에서 Kitab no-context recall은 8.2 → 4.9로 낮아졌지만 no-context precision은 19.3 → 23.2로 올랐다. 저자들은 model이 parametric knowledge에서 답할 때 precision은 개선되고 recall은 저하된 것이라고 설명한다(§5.2). Table 2는 reasoning model은 temperature 0.8, Phi-4는 0.0에서 평가하므로 각 차이의 일부는 decoding에서 올 수 있다. **결과(단일 연구).**

**Phi-4-reasoning-plus RL reward**(§4.1). Correct answer의 accuracy reward는 control length를 넘으면 length가 커질수록 1.0에서 0.5로 낮아지고, incorrect answer의 reward는 length가 커질수록 −1.0에서 −0.5로 높아진다.

    R⁺ = 0.5 + 0.25(1 + cos πρ⁺), R⁻ = −0.5 − 0.25(1 + cos πρ⁻), R_final = (8/13)·R_acc + (1/13)·R_rep

여기서 `ρ⁺ = min(1, max(L − 25,600, 0)/(31,744 − 25,600))`, `ρ⁻ = min(1, L/3,702)`는 L token response의 normalized length다. `R_acc`는 correct answer에서는 R⁺, incorrect answer에서는 R⁻이며, end token이 없으면 −0.5, invalid thinking block이면 −1.0으로 override한다. `R_rep ≤ 0`은 5-gram repetition penalty다. `R_rep = 0`이면 25,600 token 이하 correct answer는 `8/13 = 0.615`, 31,744 token correct answer는 0.308, 3,702 token 이상 incorrect answer는 −0.308, L이 0에 가까운 incorrect answer는 −0.615다. 설정은 math seed 72,401개에 iteration당 64개, batch 64, learning rate 5e-8, `G = 8`, KL `β = 0.001`, entropy coefficient 0.001이다. Release된 checkpoint는 “the model with the best observed AIME 2024 score, which is the model trained for 90 steps, over only ∼6k examples”다(§4.2). AIME 2025는 63.1 (6.3)에서 78.0 (4.6)으로 올랐다(Table 1). AIME 2024로 step을 선택하고 AIME 2024를 보고하면 그 score는 model selection의 일부가 된다. AIME 2025는 selection에 쓰지 않았고 training data 이후에 나왔다(§2.2).

**함의.** 이 report에서 STEM과 code의 reasoning SFT는 planning과 instruction following으로 transfer했지만 parametric recall은 낮아졌다. 범용 distillation recipe에는 reasoning suite 옆에 recall 및 abstention benchmark가 필요하다.

## §9 모델별 context length

Context length는 세 가지 서로 다른 방식으로 보고된다. Training에 사용한 length, inference-time extrapolation으로 도달한 length, model card에 광고한 length다. Extension method는 [[ch-32b]], claimed length와 effective length는 [[ch-32c]]에서 다룬다.

| 모델 | 훈련 length | inference-time extension | 보고된 short-context check | 출처 |
|---|---|---|---|---|
| Qwen2.5 open-weight 7B-72B | 32,768(final pre-training stage, RoPE base 1M) | YaRN + DCA ×4로 131,072; 72B-Instruct RULER-128K 67.0 → 88.4 | DCA + YaRN에서 32K 내 score 불변 | [[qwen-2.5]] §3.3, Table 16 |
| Qwen2.5-Turbo(API MoE) | 262,144(4 stage, RoPE base 10M; long SFT) | YaRN + DCA로 1M | 보고되지 않음 | [[qwen-2.5-recipe]] §3.3, §4.4 |
| Qwen2.5-7B/14B-Instruct-1M | 262,144(5 stage) | DCA + YaRN으로 1M | 14B: GPQA 45.5 → 39.9, LiveCodeBench 42.6 → 38.6, IFEval 81.0 → 84.3 | [[qwen-long-context-synth]] §3-§6, Table 6 |
| Qwen3 4B-235B | 32,768(long-context stage, 수천억 token) | YaRN + DCA ×4로 128K; card: static YaRN은 short-text quality를 낮출 수 있음 | 보고되지 않음; RULER는 thinking mode가 non-thinking보다 낮음(8B: 84.4 vs 89.1) | [[qwen-3-hybrid-thinking]] §3.2, Table 23 |
| Qwen3-235B-A22B-2507 | 262,144 native | Instruct: DCA와 MInference로 1,010,000까지 | 보고되지 않음 | [[qwen3-2507-instruct-thinking-split]] |
| Qwen3-Coder-480B-A35B | 256K native | YaRN으로 1M | 보고되지 않음 | [[qwen3-coder]] |
| Qwen3.5-397B-A17B | 262,144 native | 1,010,000까지 | card는 thinking quality를 유지하려면 ≥128K context를 권장 | [[qwen-3-5-model-card]] |
| OLMo 2 7B/13B/32B | 4,096 | 없음 | — | [[olmo-2-tulu-recipe]] Table 3 |
| Olmo 3 7B/32B | 65,536(Longmino 50B/100B token, long-context data 34%, full-attention layer에 YaRN) | 보고되지 않음 | 10B-token extension ablation: long 66% / short 34% mix는 OlmoBaseEval subset을 2.5점, long 34% / short 66% mix는 0.8점 낮춤 | [[olmo-3-model-flow]] §3.6, §3.6.3 |
| phi-3-mini | 4K default; LongRope로 128K variant | — | phi-3.5는 table 없이 4K loss가 없다고 주장 | [[phi-3]] §2, §4 |
| Phi-4 / Phi-4-reasoning | 16K(250B-token mid-training, RoPE base 250K) / 32K(SFT에서 RoPE base 두 배) | — | Phi-4 Table 2, 4K → 16K checkpoint의 phi-3-medium 대비 변화: MMLU +3.0 → +2.7, GSM8k +2.2 → +1.2, TQA −0.7 → −1.5, HumanEval +7.8 → +9.0; Phi-4-reasoning: 보고되지 않음 | [[phi-4-reasoning-sft-rl]] (arXiv:2412.08905v1 §3, §3.3, Table 2) |
| SmolLM3-3B | 64K(50B stage 두 번; config RoPE θ 2,000,000, blog 1.5M, 그 뒤 5M) | YaRN으로 128K | Reasoning mid-training 및 APO 뒤 RULER 하락, 0.9/0.1 merge로 회복 | [[smollm3-training-configs]] |
| Llama 3.1 405B | 128K(6 stage, 약 800B token; long SFT data 0.1%) | — | Short-only SFT가 long context를 regression시킴 | [[llama-3-recipe]] §3.4.2, §4.3.4 |
| DeepSeek-V3 | 128K(YaRN, 4K → 32K → 128K, 각 1,000 step) | — | 보고되지 않음 | [[deepseek-v3-recipe]] §4.3 |

SmolLM3에서 release된 4k→32k config는 step당 `12 × 6 × 1 × 32,768 = 2,359,296` token이고, step 20,000회에 걸쳐 47.2B token이다. 이는 blog의 50B와 비교된다(유도됨). Qwen2.5-1M의 offline RL은 최대 8,192 token pair만 사용했지만 LongBench-Chat을 올렸다(7B: 7.32 → 8.08). 따라서 이 report에서는 long RL data 없이도 long-context alignment가 향상됐다([[qwen-long-context-synth]] Table 3).

## Negative sample과 negative feedback

Case들은 [[ch-31a]]와 [[ch-43a]]가 정의한 “negative”의 네 의미 중 세 가지를 사용한다.

1. **Negative marginal value(버려짐).** Qwen3 cold-start filter는 bad response 여섯 종류와 Qwen2.5-72B-Instruct가 reasoning 없이 답하는 query를 제거한다(이 장 §3). OLMo 2의 `-0225` mixture는 5-sample majority가 없는 math item을 뺀다. Olmo 3 RL은 reward variance가 0인 group을 버린다([[allenai-olmo3-open-instruct-scripts]]). 어느 report도 filter의 false-negative rate를 제공하지 않는다.
2. **Content로서의 negative.** Phi-4 SFT data는 model이 보통 틀렸던 question과 answer할 수 없는 question variant를 refusal target과 pair로 둔다(Phi-4 §4.4, 부록 A.1). 이것들은 content로서의 negative다. 그 DPO pair(refusal을 wrong answer보다, correct answer를 refusal보다 선호)는 gradient로서의 negative다. SimpleQA “not attempted”는 57.5%(SFT)에서 81.1%(final)로 움직인다.
3. **Conditioning으로서의 negative.** Case-study report 중 어느 것도 control token 아래 failure를 훈련한다고 설명하지 않는다.
4. **Gradient로서의 negative.** Qwen2.5 DPO는 execution 또는 answer check에 실패한 response를 reject한다(Qwen2.5 §4.2). OLMo 2는 임의의 더 낮은 rating completion을 reject한다. Olmo 3는 quality delta를 키우기 위해 Qwen3-32B trace를 “even worse responses”와 pair로 둔다(Olmo 3 §4.3). Phi-4 PTS는 single token을 reject한다. Phi-4-reasoning-plus는 wrong answer에 [−1.0, −0.5]의 accuracy reward를 준다. Correct answer가 있는 group처럼 reward가 group mean보다 낮으면 wrong answer는 negative group advantage를 받는다. Wrong answer만 있는 group에서는 더 긴 wrong answer가 positive advantage를 받을 수 있다(§4.1-4.2 formula에서 유도).

**메커니즘.** 한 position에서 logit z와 softmax probability p에 대해 `∂log p_y/∂z_j = 1[j = y] − p_j`다. `log p_y`를 낮추는 gradient step은 각 logit을 `η(p_j − 1[j = y])`만큼 바꾼다. 다른 모든 logit은 token의 현재 probability에 비례한 양만큼 오르고, softmax는 이미 가장 확률이 높던 token에 제거된 mass의 가장 큰 share를 준다. `p = (0.5 "divide", 0.3 "multiply", 0.2 other)`, `η = 0.5`에서 “multiply”를 내리면 logit 변화는 `(+0.25, −0.35, +0.10)`이고 `p = (0.598, 0.197, 0.206)`이 된다. “multiply”에서 제거한 0.103 중 0.098(94%)은 가장 likely alternative로 간다. 같은 prefix의 PTS pair에서는 DPO loss의 gradient-descent step이 logit j를 `βσ(−m)(1[j = acc] − 1[j = rej])`에 비례해 바꾼다. 여기서 m은 DPO margin이며 chosen/rejected token의 `p_j` term은 상쇄된다. Logit에 같은 step size 0.5를 주면 `p = (0.683, 0.151, 0.166)`이다. Pair의 logit만 움직이고 “other”는 normalization을 통해서만 바뀐다. **해석**: PTS는 negative signal을 decision 하나에 국소화하며, 이는 [[ch-43a]]가 열거한 control 중 하나다.

**Negative reward와 length.** Phi-4-reasoning-plus에서 `G = 8`, `R_rep = 0`이고 위 네 case의 reward가 `(0.615, 0.615, 0.308, −0.308, −0.308, −0.308, −0.615, −0.615)`이면 group mean은 −0.077이고, group standard deviation(모집단 형태) 0.480으로 나누기 전 mean-centered advantage는 short correct +0.692, long correct +0.385, long incorrect −0.231, short incorrect −0.538이다. Short wrong answer는 long wrong answer보다 더 크게 아래로 밀린다. Report는 RL 중 response length가 AIME accuracy와 상관한다고 관찰한다(그림 7c). Olmo 3는 truncated response에서 관련 효과를 찾았다. 이를 mask하면 “overlong negative sequences”가 훈련되지 않고 average completion length가 더 높아져, Olmo 3.1 RL-Zero는 이를 더 이상 mask하지 않았다(부록 A.6.4).

**이 report의 control.** 국소화(PTS single token), KL term 또는 reference로 update 제한(DPO `β`, Phi-4-reasoning-plus KL `β = 0.001`, §4.2 objective에 `π_θold`에 대한 KL로 인쇄됨, Online Merging Optimizer의 SFT delta 방향 pull), 정보 없는 group 제거(zero-variance filtering), rule-based false negative를 피하기 위한 reference-answer judge로 label 검증(Qwen3 §4.4), random reward로 spurious gain 시험(Olmo 3 §6.2)이다.

**Diagnostics.** DPO에서는 chosen/rejected log-probability를 따로 log한다. RL statistic을 advantage sign으로 나누고, correct/incorrect group의 response length를 추적하며, pass@1 및 큰 k의 pass@k(Qwen3 Table 21)를 보고하고, abstention은 error와 별도로 센다(Phi-4 그림 6).

**Generality에 미치는 효과.** Refusal data를 쓴 Phi-4 post-training은 SimpleQA incorrect rate를 base 90.0%에서 final 15.8%로 낮췄지만, simple-evals F1은 base model을 final보다 높게 score했다(그림 6). Qwen3-32B trace imitation은 SFT를 해쳤지만, 같은 trace가 더 나쁜 response에 대한 chosen response로는 도움이 된 Olmo 3 finding은 positive imitation이 도움이 되지 않게 된 뒤 contrast가 training signal을 전달한다는 것과 양립한다([[olmo-3-model-flow]] §4.3; **해석**). 어느 case study도 improvement 중 rejected term의 비중을 측정하지 않는다.

## 레시피

값은 인쇄된 그대로다. “not reported”는 확인한 대상을 열거한다. SFT 행은 순서대로 example 또는 token, epoch, peak learning rate 및 schedule, batch, packing, loss masking, maximum length, checkpoint-selection rule을 준다.

| 모델(exact release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Qwen2.5 Instruct, open-weight | 0.5B-72B | SFT | 8개 SFT 설정 | example 1 million 초과; 2; 7e-6 → 7e-7(shape 미명시), weight decay 0.1, clip 1.0; 보고 안 됨; 보고 안 됨; 보고 안 됨; 32,768; 보고 안 됨 | arXiv:2412.15115v2 §4.1([[qwen-2.5-recipe]]) | conflict(§1: SFT/DPO/GRPO 전체 1 million example); 나머지는 2026-09-14 검증됨 | ablation 보고 없음 |
| Qwen2.5-Turbo(API MoE) | 보고 안 됨 | SFT | long-context stage | stage 1 ≤ 32,768, Qwen2.5와 같은 data/step; stage 2 short + long ≤ 262,144, ratio 미보고; short instruction에서 RL | §4.4([[qwen-2.5-recipe]]) | 2026-09-14 검증됨 | ablation 보고 없음 |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | preference | pair; loss; epoch; LR; optimizer; β | 약 150,000; DPO; 1; 7e-7; Online Merging Optimizer(α, p, variant 미보고); 미보고 | §4.2([[qwen-2.5-recipe]]) | 2026-09-14 검증됨 | ablation 보고 없음 |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | RL | algorithm; query당 response; batch; episode당 query | GRPO; 8; 2048; 256 | §4.3([[qwen-2.5-recipe]]) | 2026-09-14 검증됨; 256은 유도됨(2048 ÷ 8) | ablation 보고 없음 |
| Qwen3-235B-A22B, Qwen3-32B | flagship | SFT | 8개 SFT 설정(cold start 및 thinking-mode fusion) | 보고 안 됨; cold start “minimize both the number of training samples and the training steps” | arXiv:2505.09388v1 §4.1, §4.3([[qwen-3-hybrid-thinking]]) | not reported(§4.1-4.7, 부록 A 확인) | Table 22 stage effect만 있음 |
| Qwen3-235B-A22B | 235B-A22B | RL | query-verifier pair; algorithm; step | 3,995; GRPO; 170(AIME'24 70.1 → 85.1) | §4.2 | 2026-09-15 검증됨 | single run |
| Qwen3-8B | 8B | distill-SFT | on-policy logit distillation: start; teacher; objective; GPU hour | off-policy distilled checkpoint; Qwen3-32B 또는 235B-A22B; student sample에서 teacher logit에 대한 KL; 1,800(RL: 17,920) | §4.5; §4.7 Table 21 | 2026-09-15 검증됨(KL direction, prompt, step 미보고) | Table 21: AIME'24 74.4 vs 67.6 |
| OLMo-2-1124-7B-SFT | 7B | SFT | 8개 SFT 설정 | prompt 939,104; 2; 2e-5 linear, warmup ratio 0.03(script) 또는 0.3(Table 17 caption); sequence 128개; 보고 안 됨; 보고 안 됨; 4,096(script); development-suite average(Table 17)로 score한 LR sweep, seed 1개 뒤 최대 4 seed | arXiv:2501.00656v3 §5, Table 17; open-instruct@098424c `scripts/train/olmo2/finetune_7b.sh` L24-32([[open-instruct-allenai-recipes-recipe]]) | epoch/LR 2e-5/batch/length는 2026-09-14 검증됨(paper §5와 script 일치); warmup ratio, mixture(script `-0225`, paper/YAML `tulu-3-sft-olmo-2-mixture`), `olmo2_1124_7b_sft.yaml` L11의 LR(1.0e-5)은 conflict | §5 sweep 1e-5 / 2e-5 / 3e-5; Table 17: epoch 3, 4e-6에서 sum 49.76 vs mean 48.25 |
| OLMo-2-0325-32B-SFT | 32B | SFT | prompt; LR; batch | 866,138; 4e-6; sequence 256개(8 node × 8 GPU × 1 × 4) | §5; `finetune_32b.sh` L8, L24-32 | 2026-09-14 검증됨; batch는 script에서 유도 | §5: 1e-6~5e-6 sweep, 4e-6에 extra seed 1개 |
| OLMo 2 32B Instruct | 32B | RL | algorithm; LR; KL β; prompt당 sample | GRPO; 5e-7; 0.1; 16 | arXiv:2501.00656v3 §5 | 2026-09-15 검증됨 | 32B의 RL sweep 미보고 |
| Olmo 3 7B Think SFT | 7B | distill-SFT | 8개 SFT 설정 | prompt 2,268,468, “Total Tokens” 45.4B(unit 미정의); 2; 5e-5; 1,048,576 token; document packing; rule 없이 masking 명시; 32,768; evaluation suite LR sweep 후 vibe test | arXiv:2512.13961v2 §4.2, Tables 17, 47, App. A.6.1; open-instruct@d8a7f1c `7b_think_sft.sh` L7-19([[allenai-olmo3-open-instruct-scripts-recipe]]) | 2026-09-15 검증됨(masking rule 미보고) | §4.2 LR sweep; Table 18은 100K base mix에서 source별 ablation |
| Olmo 3 32B Think SFT | 32B | distill-SFT | prompt; token; LR; batch; checkpoint rule | 2,253,916; 45.2B; 1.0e-4를 5.0e-5와 soup; 4M token; LR 두 개의 linear merge | Tables 17, 47; A.6.1; §4.2 | 2026-09-15 검증됨 | §4.2 |
| Olmo 3 7B Instruct SFT | 7B | SFT | prompt; token; LR; start | 2,152,112; 3.4B; 8.0e-5; Think SFT checkpoint | Tables 30, 47; `7b_instruct_sft.sh` L6-18 | 2026-09-15 검증됨 | §5.5 및 Table 29: Think SFT에서 시작하면 response length 변화가 작으면서 benchmark score 상승 |
| Olmo 3 DPO run(7B/32B) | 7B, 32B | preference | loss; β; epoch; warmup | `dpo_norm`; 5; 1; 0.1 | open-instruct@00d42c4 `7b_instruct_dpo.sh` L49-54, `7b_think_dpo.sh` L32-37, `32b_think_dpo.sh` L45-50; Table 48 | 2026-09-14 검증됨 | A.6.2: LR 및 dataset size sweep |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | 8개 SFT 설정 | prompt 939,344; 2; 5e-6 linear, warmup 0.03; 128; 미명시; 미명시; 4,096; 미명시 | arXiv:2411.15124v5 §4.3, Table 11([[tulu-3]]); docs/tulu3.md L35-53([[open-instruct-allenai-recipes-recipe]]) | 나열 값은 2026-09-14 검증됨(Table 11/docs 일치); loss reduction conflict(§4.3.2는 sum loss 선택, 098424c pinned command는 mean loss) | §4.3.2: 5e-6 sum loss 최상, 2 epoch 초과는 도움 안 됨 |
| Llama 3.1 405B | 405B | SFT | 8개 SFT 설정 | 미인쇄(Table 7은 share만); 미인쇄; 8.5K-9K step 동안 1e-5; 미인쇄; 미명시; prompt token mask; 미인쇄; data 또는 hyperparameter가 다른 run의 average | arXiv:2407.21783v3 §4.1.3, §4.1.5, Table 7([[llama-3-recipe]]) | 2026-09-14 검증됨(미인쇄 값은 미보고) | “work well across different rounds and data mixes”; number 없음 |
| SmolLM3-3B | 3B | SFT | 8개 SFT 설정 | token 1.8B(1B non-reasoning, 0.8B reasoning); 4(“~8B tokens”; 1.8B × 4 = 7.2B 유도); 미보고; 미보고; BFD packing; user turn/tool result에 loss mask; 미보고; APO checkpoint와 mid-training checkpoint를 0.9/0.1로 linearly merge한 soup인 released model | huggingface.co/blog/smollm3 “Supervised Finetuning”, “Model Merging”([[smollm3-training-configs]]) | 2026-09-15 검증됨(LR/batch/length/checkpoint rule 미보고) | blog: reasoning-ratio ablation의 mixture; number 없음 |
| DeepSeek-V3 | 671B(37B active) | distill-SFT | 8개 SFT 설정 | instance 1.5M; 2; cosine 5e-6 → 1e-6; 미인쇄; sample masking을 써 packing; 미인쇄; 미인쇄; 미보고 | arXiv:2412.19437v2 §5.1([[deepseek-v3-recipe]]) | 2026-09-14 검증됨 | Table 9(V2.5): R1-style data vs short CoT |
| Phi-4 | 14B | SFT | 8개 SFT 설정 | 약 8B token; 미보고; 1e-6; 미보고; 미보고; 미보고; 미보고; 미보고 | arXiv:2412.08905v1 §4.1([[phi-4-reasoning-sft-rl]]) | 2026-09-15 검증됨 | ablation 보고 없음 |
| Phi-4 | 14B | preference | round 1; round 2 | `0.2 ≤ p(success) ≤ 0.8`인 question의 PTS pair; 약 850k GPT-4o-judged pair | §4.2-4.3 | 2026-09-15 검증됨 | Table 9 stage 및 ablation column |
| Phi-4-reasoning | 14B | distill-SFT | 8개 SFT 설정 | pair 1.4M 초과, unique token 8.3B, trained 16B; “2+ pass”, cluster당 epoch weight; 1e-5, warmup step 450, weight decay 1e-4, AdamW, 약 16K step; global batch 32(unit 미정의); 미명시; 미명시; 32K; 미보고 | arXiv:2504.21318v1 §3, §3.1-3.2 | 2026-09-15 검증됨 | §3.1 LR grid [1e-6, 2e-5]; o3-mini high > medium |
| Phi-4-reasoning-plus | 14B | RL | seed; batch; LR; G; KL β; entropy; max length; checkpoint rule | 72,401(iteration당 64); 64; 5e-8, cosine warmup 10 step; 8; 0.001; 0.001; 32K(output은 31K에서 clip); 최고 AIME 2024, step 90 | §4, §4.2 | 2026-09-15 검증됨 | 그림 7a: 90 step 뒤 gain 없음 |
| phi-3-mini/small/medium | 3.8B-14B | SFT, preference | 모든 설정 | 보고되지 않음 | arXiv:2404.14219v4 §2, §5([[phi-3]]) | not reported(v1/v4 body 및 appendix 확인) | — |

**Unit check.** Phi-4-reasoning은 global batch 32가 32,768-token sequence를 센다면 `16,000 steps × 32 × 32,768 = 16.8B` token position으로, trained token 16B와 비교된다. Report는 batch unit과 packing을 모두 밝히지 않는다(유도됨). Tülu 3 8B는 `939,344 ÷ 128 = 7,339` step/epoch, 2 epoch에서 약 14,680 step이다(유도됨). Olmo 3 “Total Tokens”는 한 pass를 세는지 두 epoch 모두를 세는지 말하지 않으므로, 45.4B는 step당 1,048,576 token에서 약 43,300 step(한 pass) 또는 약 86,600 step(두 epoch)에 해당한다(유도됨).

**작은 범용 run의 시작점.** 0.9~2.3 million example에 대한 7B-8B base의 full-parameter SFT에서 검증된 값은 epoch 2회다(OLMo 2 7B, Olmo 3 7B, Tülu 3 8B; Qwen2.5 0.5B-72B도 해당). Peak learning rate는 base와 batch에 따라 다르다. 4,096 token 이하 sequence 128개에서 5e-6(Tülu 3 8B의 Llama 3.1 8B), 같은 batch/length에서 2e-5(OLMo 2 7B), 1,048,576 token packing batch와 32,768-token sequence에서 5e-5(Think) 또는 8e-5(Instruct)(Olmo 3 7B)다. 1.4 million 초과 long reasoning trace로 훈련한 14B base에서 Phi-4-reasoning은 global batch 32(unit 미정의), 32K context에서 1e-5를 썼고, SmolLM3-3B는 1.8B-token set에서 epoch 4회를 썼다. Base 또는 batch가 이 행들과 다르면, 가장 가까운 base/batch의 행 주위로 learning rate를 적어도 세 개 sweep하라(OLMo 2 7B sweep 크기). OLMo 2는 자기 base가 Llama 3.1 recipe보다 높은 learning rate를 필요로 했기 때문이다(§5). Knowledge benchmark를 포함한 development suite에서 선택하고, 별도 held-out suite에 보고하며, 선택한 checkpoint를 모든 response mode에서 평가하라. 그 외에는 match되는 행을 재사용하되 held-out 및 per-mode check는 여전히 실행하라.

## 일반화 관점

**(a) 폭을 넓힌 것.**

- Non-thinking data와 general RL 추가는 thinking mode에서 Qwen3-32B IFEval을 12.0점, ToolUse를 22.2점, CounterFactQA를 17.7점 올렸다([[qwen-3-hybrid-thinking]] Table 22). **결과(단일 연구).**
- On-policy distillation은 Qwen3-8B AIME'24 pass@64를 90.0에서 93.3으로, MMLU-Redux를 86.4에서 88.3으로 올렸다. RL은 pass@64를 90.0에 남겼다(Table 21). **결과(단일 연구).**
- STEM 및 code reasoning SFT는 직접 target으로 삼지 않았다고 한 TSP, 3SAT, BA-Calendar에서 Phi-4-reasoning을 30-60점, IFEval Strict를 62.3에서 83.4로 올렸다([[phi-4-reasoning-sft-rl]] §1, Table 2). **결과(단일 연구)**이며 decoding-temperature confound가 있다.
- Short-only offline RL은 Qwen2.5-1M model의 long-context LongBench-Chat을 올렸다([[qwen-long-context-synth]] Table 3). **결과(단일 연구).**
- Earlier checkpoint와 merge하면 후속 stage에서 잃은 ability를 회복했다. SmolLM3의 APO soup과 mid-training checkpoint의 0.9/0.1 merge는 base RULER를 128K까지 회복했다([[smollm3-training-configs]]). **결과(단일 연구)**이며 blog는 비교 table을 인쇄하지 않는다.

**(b) narrowing 또는 forgetting을 일으킨 것.**

- Fusion 및 general RL 뒤 Qwen3-32B thinking-mode AIME'24는 −2.4, LiveCodeBench v5는 −2.7이다. 저자들은 broader training 탓이라고 본다(Table 22). **결과(단일 연구)**와 **해석**이다.
- Web data 없이 훈련한 13B Phi ablation은 phi-3-medium보다 TriviaQA가 14.8점 낮다([[phi-4-reasoning-sft-rl]] Table 3). Phi-3-mini는 TriviaQA에서 Mixtral 8x7B보다 18.2점 낮다([[phi-3]] §3). Phi-4-reasoning의 no-context Kitab recall은 8.2에서 4.9로 낮아졌지만 no-context precision은 19.3에서 23.2로 올랐다(Table 2). 같은 방향은 Phi report 세 개에서 보이지만 모두 같은 lab에서 나왔으므로 agreement는 independent replication이 아니다.
- RL은 Olmo 3 7B Instruct PopQA를 20.7에서 14.1로, LitQA2를 43.3에서 38.2로 낮췄다([[olmo-3-model-flow]] Table 26). DPO 및 RLVR은 OLMo 2 32B safety를 93.8에서 85.9로 낮췄다([[olmo-2-tulu-recipe]] Table 16). 각각 **결과(단일 연구)**다.
- Judge-guided DPO는 Phi-4 DROP을 86.1에서 75.5로, PTS DPO는 IFEval을 66.2에서 63.0으로 낮췄다(Table 9). **결과(단일 연구).**
- Long-context pre-training stage와 long SFT를 추가한 Qwen2.5-14B-Instruct-1M은 Qwen2.5-14B-Instruct와 short benchmark에서 양방향으로 다르다(GPQA −5.6, IFEval +3.3). 저자들은 short-task performance가 비슷하다고 말한다([[qwen-long-context-synth]] §6.2, Table 6). **결과(단일 연구).**
- Imitation saturation: Qwen3-32B trace에 SFT를 더 수행하면 Olmo 3 Think SFT를 해쳤다([[olmo-3-model-flow]] §4.3). **결과(단일 연구).**

**(c) 이 단계에서의 측정 방법.**

- Mode와 capability family 전반의 stage table 및 새 behaviour를 위한 in-house test(Qwen3의 ThinkFollow, CounterFactQA, ToolUse).
- Model selection에 영향을 주었는지 disclosure를 포함한 post-dated test(Phi-4 2024년 11월 AMC, Phi-4-reasoning의 AIME 2025).
- Prompt format 두 개에서 반복하는 matched fresh set 및 significance test(GSM1k).
- Standard deviation을 보고하는 repeated sampling(Phi-4-reasoning: AIME 2025에서 50 run).
- RLVR gain을 해석하기 전 random-reward negative control(Olmo 3 RL-Zero).
- Threshold가 명시된 decontamination rule: 8-gram overlap ≥ 0.5(Olmo 3 §4.2.1), LCS ≥ 13 token 및 shorter sequence의 ≥ 0.6(Qwen2.5 §5).
- Accuracy와 별도의 abstention(Phi-4 SimpleQA), 큰 k의 pass@k(Qwen3 Table 21).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 확인 |
|---|---|---|
| Variant의 setting을 family 전체의 것으로 귀속(Qwen2.5-Turbo의 262,144-token SFT stage를 Qwen2.5-72B-Instruct의 것으로 읽음) | 해당 section이 전혀 인쇄하지 않은 checkpoint에 value가 나타남 | 해당 locus 문장에서 checkpoint name을 찾고 variant마다 ledger row 하나를 둠 |
| 서로 다른 unit(example, prompt, pair, unique token, trained token)에서 “SFT size” 비교 | Recipe 두 개가 epoch count 근처의 factor만큼 다름 | step × batch × length를 재계산해 epoch로 나누고 Olmo 3 “Total Tokens” 같은 undefined unit을 표시 |
| Question 30개 benchmark에서 single-run 1-3점 stage delta를 regression 또는 gain으로 읽음 | Delta가 small math set의 run-to-run spread(Phi-4-reasoning의 average-of-5 AIME evaluation 두 번 사이 최대 5-10점)보다 작음 | Phi-4-reasoning의 AIME 2025처럼 evaluation을 반복(예: 50 run)하고 standard deviation 보고 |
| Table보다 prose summary 신뢰(OLMo 2 soup “consistently” better; mix E GSM* 43.0 vs 60.5) | Table cell이 text와 모순 | 인용한 table의 모든 cell을 읽음 |
| Fresh test가 selection에 영향을 준 뒤에도 untouched라고 취급 | Final choice 전에 candidate score를 봄(Phi-4 부록 C footnote) | Fresh set이 model selection 대비 처음 score된 시점을 기록 |
| Hybrid model을 mode 하나에서만 평가 | 같은 Qwen3-32B checkpoint에서 non-thinking GPQA-Diamond 54.6, thinking 68.4 | 각 mode의 specialist benchmark와 mode-following test 보고 |
| 모든 request에 static YaRN 활성화 | `rope_scaling` 추가 뒤 short-context score 변화(Qwen3-8B card는 static YaRN이 shorter text quality를 낮출 수 있다고 경고) | Scaling setting 유무로 short-context suite 실행하고 card 권고대로 long input에만 `rope_scaling` 추가 |
| Release된 config 대신 blog value 복사 | Reproduced run의 RoPE base가 다름(SmolLM3: blog 1.5M, config 2,000,000) | Commit의 config file을 읽고 두 값을 conflict로 기록 |
| Data를 공개하지 않은 base의 RLVR gain을 learned reasoning으로 해석 | Random reward에서도 gain이 나타남 | 같은 prompt에서 random-reward control 실행(Olmo 3 §6.2) |
| F1만으로 hallucination reduction score | Incorrect answer가 낮아지는데 F1도 하락(Phi-4 SimpleQA) | Correct, incorrect, not-attempted rate를 따로 보고 |

## 이해 점검

1. Qwen3-32B thinking-mode AIME'24는 Stage 3-4 뒤 2.4점 낮아졌다. §1 standard-error formula와 Phi-4-reasoning의 run-to-run evidence를 사용해 이것을 regression이라 부르려면 어떤 추가 measurement가 필요한지, broader training이라는 저자 conjecture가 result가 아니라 interpretation인 이유를 설명하라.
2. Qwen3 report는 versatility를 위해 thinking-mode loss를 받아들였고, Qwen3-2507은 mode를 분리했으며, Qwen3.5는 다시 checkpoint 하나에서 둘을 제공한다. Data를 고정한 어떤 experiment가 hybrid training이 specialist quality에 cost를 주는지 결정하며, 각 mode에서 어떤 benchmark가 필요한가?
3. Same checkpoint에서 한 RL이 못 한 pass@64 상승을 teacher logit에 대한 on-policy distillation이 낼 수 있는 이유를, token별 target distribution과 response별 scalar reward의 차이를 이용해 설명하라. 이 이점은 어느 조건에서 사라지는가?
4. Phi-4 report의 13B synthetic-only ablation은 MATH와 HumanEval에서 얻고 TriviaQA 14.8점을 잃었으며, Phi-4 post-training은 SimpleQA not-attempted rate를 base 6.8%에서 final 81.1%로 올렸다. Incorrect answer가 줄었는데 SimpleQA F1이 왜 낮아졌는지, 범용 recipe는 대신 어떤 metric을 보고해야 하는지 설명하라.
5. PTS gradient에서는 `p_j` term이 상쇄되지만, negative-feedback section example에서 rejected token 하나만 내리면 제거된 mass의 94%가 가장 likely alternative로 간다. 두 경우가 다른 이유와 이것이 long reasoning trace의 full-sequence DPO에 함의하는 것을 설명하라.
6. Olmo 3는 Qwen3-32B trace에 SFT를 더 하면 해롭지만, 같은 trace가 더 나쁜 response에 대한 chosen response로는 도움이 된다고 찾았다. Imitation과 contrast의 차이로 설명하고, 자신의 model에서 이를 어떻게 시험할지 말하라.
7. Phi-4-reasoning-plus는 AIME 2024로 RL checkpoint를 선택하고 AIME 2024를 보고한다. AIME 2025가 더 informative한 number인 이유와, 같은 문제가 Phi-4의 AMC evaluation에 어떻게 적용되는지 설명하라.
8. OLMo 2 32B average는 RLVR로 변하지 않았지만 safety는 6.0점 낮아졌다. Average가 이를 숨기는 이유를 설명하고, 이를 flag했을 stage gate를 작성하라.

## 연결

- **이전 및 의존성:** [[ch-33]] Case Studies A: How Tülu 3 and Llama 3 Measured and Protected Broad Capability — OLMo 2와 Olmo 3가 재사용하는 Tülu 3 recipe, Llama 3의 averaging 및 long-context SFT share.
- **다음:** [[ch-35]] Distillation in Practice A: Where Labs Insert Teacher Data — §4와 §8의 teacher data placement를 확장한다.
- [[ch-35a]] Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters — 더 넓은 recipe 집합에서 Qwen3 no-CoT query filter와 Phi-4-reasoning teachable-prompt selection.
- [[ch-30a]] Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control — §1 stage delta의 measurement method.
- [[ch-30c]] Weight Averaging and Model Merging for Generalist Models — Online Merging Optimizer, OLMo 2 soup, SmolLM3 merge.
- [[ch-31a]] Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood — refusal target과 rejected content.
- [[ch-43a]] Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages — negative-feedback section의 derivation.
- [[ch-32b]] Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression 및 [[ch-32c]] Claimed versus Effective Context Length and Long-Context Evaluation — §9의 detail.
- [[ch-45c]] Context Management for Long-Horizon Agents — §5의 context folding과 discard-all strategy.
- [[ch-47a]] Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation 및 [[ch-48]] Contamination Detection and Its Effect on Reported Scores — GSM1k, post-dated contest, random-reward control.
- [[ch-51]] Metric Noise, Confidence Intervals, and Go/No-Go Decisions — §1 및 §7의 statistic.

## 출처

- [[qwen-2.5]] — Qwen2.5 report(arXiv:2412.15115v2): post-training data/setting, contamination filter, 72B score, reward-model statement, DCA + YaRN result.
- [[qwen-2.5-recipe]] — Qwen2.5 SFT, DPO, GRPO 및 Qwen2.5-Turbo long-context SFT의 ledger row.
- [[qwen-long-context-synth]] — Qwen2.5-1M report(arXiv:2501.15383v1): two-stage long SFT, short-only offline RL, Table 6 short-benchmark 변화.
- [[online-merging-optimizer]] — 누락된 card 대신 쓰는 chapter excerpt: OnDARE equation, Table 1 및 Table 2 결과, α discrepancy.
- [[qwen-3]] — Qwen3 library card. 검증하지 않았으므로 모든 Qwen3 value는 report에서 읽어 아래 excerpt에 기록했다.
- [[qwen-3-hybrid-thinking]] — arXiv:2505.09388v1 chapter excerpt: 4 stage, Table 9 template, Tables 21-23, Qwen3-8B model card long-context note.
- [[qwen3-2507-instruct-thinking-split]] — chapter excerpt: 2025년 7월 announcement 및 Instruct-2507/Thinking-2507 model card.
- [[qwen3-coder]] — Qwen3-Coder blog의 chapter excerpt: token 7.5T, code RL, parallel environment 20,000개.
- [[qwen-3-5]] — Qwen3.5 library card. 검증하지 않았으며 inherited-pipeline, GRPO, 19× speedup statement는 사용하지 않았다.
- [[qwen-3-5-model-card]] — Qwen3.5-397B-A17B model card chapter excerpt: mode control, RL claim, context-folding evaluation setting.
- [[olmo-2]] — OLMo 2 library card. Report에 없는 32K extension과 GPU-hour figure가 있어 검증하지 않았으며 value는 아래 excerpt에서 가져온다.
- [[olmo-2-tulu-recipe]] — arXiv:2501.00656v3 chapter excerpt: Table 3 context, SFT mixture/sweep, DPO/RLVR, Tables 14 및 16-17.
- [[open-instruct-allenai-recipes-recipe]] — release된 OLMo 2 및 Tülu 3 launch value와 논문과의 conflict.
- [[olmo-3]] — Olmo 3 library card. 검증하지 않았으므로 value는 아래 excerpt에서 가져온다.
- [[olmo-3-model-flow]] — arXiv:2512.13961v2 chapter excerpt: stage, Dolci size, SFT setting, Delta Learning, function-calling data, Table 26, RL-Zero control.
- [[allenai-olmo3-open-instruct-scripts]] 및 [[allenai-olmo3-open-instruct-scripts-recipe]] — line locus가 있는 Olmo 3 SFT/DPO/RL launch script.
- [[phi-3]] — Phi-3 report(arXiv:2404.14219v4): pre-training phase, TriviaQA gap, LongRope context, unreported post-training setting.
- [[phi-4]] — Phi-4 library card. 검증하지 않았고 report 두 개를 merge하므로 value는 아래 excerpt에서 가져온다.
- [[phi-4-reasoning-sft-rl]] — arXiv:2412.08905v1 및 arXiv:2504.21318v1 chapter excerpt: synthetic-data ablation, Table 2(4K/16K checkpoint), PTS, Table 9, refusal data/SimpleQA, AMC, Phi-4-reasoning SFT/RL setting, transfer/variance.
- [[gsm1k]] — arXiv:2405.00332v4 chapter excerpt: GSM1k design, Phi-family finding, worked example에 쓴 App. F row.
- [[emergence-loss-perspective-recipe]] — Z-test example에 쓴 GSM8K test-set size(1,319).
- [[tulu-3]] — SFT ledger의 Tülu 3 report value(Table 11, §4.3); card는 검증하지 않아 number에 쓰지 않았다.
- [[llama-3-recipe]] — Llama 3.1 405B SFT, averaging, long-context row.
- [[smollm-3]] — SmolLM3 library card. 검증하지 않았으며 SFT detail에 사용하지 않았다.
- [[smollm3-training-configs]] — chapter excerpt: release된 context-extension config 및 SmolLM3 blog SFT, APO, merge statement.
- [[deepseek-v3-recipe]] — DeepSeek-V3 SFT 및 context-extension row.
