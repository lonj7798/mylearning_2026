<!-- chapter: ch-32e
     track: midtraining
     kind: content
     title: Mid-Training·Annealing·Context Extension Recipe의 병렬 비교
     deps: [ch-32c, ch-32d, ch-14a]
     sources: [[llama-3]], [[llama-3-recipe]], [[olmo-2]], [[olmo-2-midtraining]], [[olmo-3]], [[olmo-3-base-stages]], [[olmo-core-olmo3-configs]], [[smollm-3]], [[smollm3-midtraining]], [[smollm3-training-configs]], [[qwen-2.5]], [[qwen-2.5-recipe]], [[qwen-long-context-synth]], [[prolong]], [[prolong-recipe]], [[deepseek-v3]], [[deepseek-v3-recipe]], [[deepseek-v3.1-recipe]], [[nemotron-nano-2]], [[glm-4-5]], [[glm-4-5-recipe]], [[glm-5]], [[minicpm]], [[cooldown-scaling-beyond-fixed-durations]], [[llama-2-long]], [[yarn]]
     figures: figures/stage-budget-ledger.html
     revised: 2026-09 (범용성 개정)
-->

# Chapter 32e — Mid-Training·Annealing·Context Extension Recipe의 병렬 비교

> **핵심 통찰.** 여기서 비교한 공개 보고서에서 명시적 mid-training은 pretraining token의 1.3%~12%를 쓰고, context extension은 0.09%~5.1%를 쓴다(비율 계산은 §3, §4). 보고서들은 세 가지 관행에서 일치한다. learning rate가 낮은 값 또는 0으로 내려가는 동안 data mix를 바꾸고, long-context stage에도 short data를 남기며, 여러 보고서가 평가 길이보다 더 긴 길이에서 학습한다. 다만 인쇄된 ablation 또는 before/after 비교가 있는 값은 드물다. OLMo 2의 mid-training은 7B 평균을 53.0에서 62.9로 높였고(OLMo 2 Table 9), 34% long / 66% short extension mix는 66% / 34%의 2.5점과 비교해 base-model score를 0.8점 낮췄으며(Olmo 3 §3.6.3), Nemotron-H 8B에서는 128K 대신 512K에서 학습하면 RULER-128K가 73.68에서 81.04로 올랐다(Nemotron Nano 2 Table 4). Stage 수와 learning rate는 인쇄된 ablation 없이 각 보고서가 선택한 단일 설정이다. RoPE base는 ProLong만 ablation했고(Table 18), merge weight는 SmolLM3와 Nemotron Nano 2만 비교했으나 표는 인쇄하지 않았다. 논문 또는 blog와 released configuration 사이에서 한 RoPE base와 두 learning rate, 총 세 값이 다르다.
>
> **지침.** 보고서의 stage budget을 복사할 때는 K를 명시적으로 1,000 또는 1,024로 정하고, batch가 무엇을 세는 단위인지 밝힌 뒤 seen token으로 변환하라. 그 뒤 같은 보고서의 pretraining token 수로 나누어야 한다. 그렇지 않으면 DeepSeek-V3의 extension은 선택에 따라 125.8B 또는 122.9B token이 되고, OLMo 2의 total에는 souping ingredient 전체가 들어간다는 차이를 놓친다. 작은 RoPE model을 4K–8K에서 32K–65K로 늘릴 때는 extension mix에 short 또는 mid-training data를 남기고(Olmo 3은 token의 66%, ProLong은 37%), short-context suite와 held-out long-context suite로 각 stage를 gate하라. 두 보고서 모두 long 비중이 커질수록 short-task score가 내려갔고, Olmo 3 7B는 short data를 66% 넣은 extension stage에서도 Math 5.4점을 잃었다(Olmo 3 §3.6.3, Table 13; ProLong Fig. 3). reasoning mid-training이나 preference optimization 같은 뒤 stage가 long-context score를 낮추면, extension을 다시 돌리기 전에 long-context score가 강한 이전 checkpoint를 작은 가중치로 섞는 linear merge를 시험하라. SmolLM3는 post-trained 0.9 / mid-training checkpoint 0.1 merge로 base RULER score를 128k까지 회복했다고 보고한다. 반대로 뒤 stage가 long-context score를 낮추지 않았다면, Olmo 3 7B base model처럼 long-context merge 없이 extension을 한 번 수행하는 구성이 사용됐다(§6).

## 범용 모델에서 이 장이 중요한 이유

Base model의 pretraining run은 보통 learning rate가 더 낮고 data mix가 바뀐 stage에서 끝난다. 보고서는 이 stage를 annealing, decay phase, cooldown, 또는 mid-training이라 부르고, 많은 보고서는 pretraining 끝부분에 context-extension stage도 둔다. 이들은 pretraining(ch-14a)과 supervised fine-tuning(ch-30) 사이에 놓인다. Token 비중은 작지만, SFT와 RL이 어느 capability에서 출발하는지를 정한다. OLMo 2 7B의 GSM8K score는 50B token의 mid-training에서 24.1에서 67.5로 바뀌었고(OLMo 2 Table 9), Olmo 3의 mid-training thinking·instruction data는 post-training 이전에 base-model average를 높였다(Olmo 3 Table 10).

범용 모델에서는 같은 stage가 capability를 좁힐 수도 있다. Math·code 비중이 큰 mid-training mix는 Olmo 3의 multiple-choice 및 generative QA score를 낮췄고(Table 7), context-extension stage는 Olmo 3 32B의 Math cluster를 69.7에서 61.4로 낮췄다(Table 13). SmolLM3의 reasoning mid-training은 merge가 RULER score를 복원하기 전까지 long-context RULER score를 낮췄다([[smollm3-midtraining]]). ch-32는 mid-training의 mechanism을, ch-32b는 context-extension method를, ch-32c는 long-context evaluation을, ch-32d는 agentic mid-training data를 다룬다. 이 장은 공개된 setting을 나란히 놓고 unit을 정규화하며, 2026-09-14 또는 2026-09-15에 locus에서 검증한 값을 표시하고, 검증된 row만으로 시작 configuration을 만든다.

함께 제공되는 그림 [figures/stage-budget-ledger.html](figures/stage-budget-ledger.html)은 stage budget을 pretraining token 대비 비율로 log scale에서 비교하고, 각 비율 뒤의 input과 locus를 열어 보며, K를 1,000 또는 1,024로 읽어 step·batch·sequence length에서 token 수를 다시 계산하게 한다.

## §1 Unit: stage budget이 세는 대상

**정의.**

- **Seen token**은 반복을 포함해 optimizer가 처리한 token이다. **Unique token**은 반복 전 data에서 서로 다른 token이다.
- 한 subset의 **epoch**는 그 subset을 한 번 통과하는 것이다. 어떤 subset을 4 epoch 학습했다면 unique token의 4배가 seen token에 기여한다.
- **Sequence length**는 training window 길이다. **Document length**는 하나의 source document 길이다. 여러 document를 한 window에 pack하거나, 한 document를 window 여러 개로 자르면 두 길이는 달라진다.
- **Souping ingredient**는 released checkpoint를 만들 때 weight를 평균내는 여러 run 중 하나다(§6).

**문제.** 보고서는 변환 없이는 서로 비교할 수 없는 unit으로 budget을 인쇄한다. 이 장의 source에는 네 경우가 있다.

1. **Unique와 seen의 차이.** GLM-5의 issue–PR data는 1.55T seen token인 mid-training phase 안에서 “약 160B unique tokens”다([[glm-5]] §2.3). SmolLM3 reasoning mid-training은 4 epoch의 35B token, 즉 약 140B seen token이다([[smollm3-midtraining]]). OLMo 2의 300B Dolmino mix는 StackExchange, Wikipedia, math data를 각각 네 번 반복한다(Table 13의 `Source %`가 400; [[olmo-2-midtraining]]).
2. **Binary prefix.** ProLong은 K = \(2^{10}\), M = \(2^{20}\), B = \(2^{30}\)으로 정의한다(footnote 2). 따라서 이 보고서의 “20B tokens”는 decimal token으로 \(21.47 \times 10^9\)개다([[prolong-recipe]]).
3. **Unit이 없는 batch.** DeepSeek-V3는 extension phase의 batch size를 1920과 480으로 인쇄하지만, 단위를 쓰지 않는다([[deepseek-v3-recipe]] §4.3).
4. **병렬 run을 포함한 total.** OLMo 2는 7B에 대해 4.05T token을 보고한다. 이는 3.90T pretraining에 평균낸 세 개의 50B mid-training run을 모두 더한 수다. 그러나 각 ingredient가 실제로 본 token은 3.95T다([[olmo-2-midtraining]] §2.3).

**식.**

```
D_stage = Σ_p  steps_p × B_p × L_p          f = D_stage / D_pretrain
```

- \(p\)는 stage 안의 phase를 가리킨다. `steps_p`는 phase \(p\)의 optimizer step 수다.
- \(B_p\)는 sequence 개수로 센 global batch이고, \(L_p\)는 token으로 센 sequence length다.
- \(D_{pretrain}\)은 같은 보고서가 인쇄한 pretraining-stage token 수이며, \(f\)는 stage의 token share다.

**계산 예: DeepSeek-V3 extension budget.** DeepSeek-V3는 32K에서 batch 1920인 1,000-step phase와, 128K에서 batch 480인 1,000-step phase를 인쇄한다(§4.3). Batch가 sequence를 세고 K = 1,024라고 두면 다음과 같다.

1. Phase 1: \(1000 \times 1920 \times 32{,}768 = 62.91\text{B}\) token.
2. Phase 2: \(1000 \times 480 \times 131{,}072 = 62.91\text{B}\) token.
3. 합계는 125.8B이고, \(f = 125.8\text{B} / 14.8\text{T} = 0.85\%\)다.

K = 1,000이면 각 phase는 61.44B이고 합계는 122.9B다. Olmo 3 보고서는 이 stage에 “DeepSeek V3: 123B”라고 적는데([[olmo-3-base-stages]] §3.6), 이는 decimal 해석과 맞는다(해석; Olmo 3은 계산을 보이지 않는다). DeepSeek-V3.1 model card는 630B-token 32K phase를 V3 phase의 “10-fold”라고 설명한다([[deepseek-v3.1-recipe]]). 이는 V3 phase 1이 약 63B임을 뜻하며, 어느 계산과도 3% 이내로 맞는다. **도출한 값.** 원 보고서는 token total을 인쇄하지 않는다.

**Sequence length와 document length.** 두 보고서는 training window가 long document가 온전한 상태로 남을 확률을 바꾼다고 명시한다. Nemotron Nano 2는 concatenate-and-chunk loading이 long coherent document를 자를 확률을 낮출 수 있으므로, 128k target을 위해 Phase LC를 524,288 token에서 학습했다([[nemotron-nano-2]] §2.6). Olmo 3은 extension 중 concatenate-then-split을 best-fit packing으로 바꿨다. 전자는 “평균적으로 underlying document length distribution보다 짧은 training instance”를 만들기 때문이다([[olmo-3-base-stages]] §3.6.4). 그러므로 long:short ratio는 unit과 함께만 의미가 있다. Qwen2.5-1M의 “현재 maximum length에서 75% sequences”는 sequence 비율이고([[qwen-long-context-synth]] §3), Olmo 3의 34%는 token 비율이다(Table 11: 50.0B 중 16.95B).

**함의.** Unit 없이 복사한 budget은 K 해석 때문에 source와 2.4% 다를 수 있고, epoch 때문에 4배 다를 수 있으며, souping ingredient 수만큼 달라질 수도 있다. 아래 Recipe 표는 각 값을 source가 인쇄한 unit과 함께 기록한다.

## §2 Decay와 annealing ledger

**정의.** **Decay phase**(annealing 또는 cooldown이라고도 부른다)는 learning rate가 낮은 값 또는 0까지 내려가는 run의 마지막 부분이다. MiniCPM의 Warmup-Stable-Decay(WSD) schedule은 이를 명시적으로 분리한다([[minicpm]] §4.2, Eq. 1).

```
WSD(T; s) = (s / W) · η       for s < W
          = η                 for W < s < T
          = f(s − T) · η      for T < s < S,   0 < f(s − T) ≤ 1 decreasing
```

- `s`는 step이고, `W`는 warmup이 끝나는 지점, `T`는 stable stage가 끝나는 지점, `S`는 training 종료 지점이다.
- `η`는 최대 learning rate이고, `f`는 decay function(linear, cosine 또는 exponential)이다.

**문제.** 높은 constant learning rate로 학습한 run은 decay를 거친 같은 run보다 loss가 더 높다. MiniCPM은 0.036B model에서 decay가 시작되면 loss가 “significant rapid decline”을 겪고 cosine schedule의 loss에 도달한다고 보고한다([[minicpm]] §4.2, Fig. 5). Loss 하락이 decay phase에 집중되므로, 여러 보고서는 바로 이때 data mix도 바꾼다.

**실제로 쓰는 mechanism.**

1. Main schedule을 멈추거나 자른다. OLMo 2 7B는 5T-token cosine을 4T token에서 멈추고 “mid-training으로 switch”한다([[olmo-2-midtraining]] §4.1). Olmo 3 32B는 5.93T cosine을 5.5T에서 자른다([[olmo-3-base-stages]] Table 35).
2. Mix를 더 높은 quality 또는 task-bearing data 쪽으로 바꾼다. Llama 3은 “very high quality” data source를 upsample한다([[llama-3-recipe]] §3.4.3). MiniCPM은 pretraining data에 “high-quality SFT data”를 섞고([[minicpm]] §6.2), SmolLM3는 code 비중을 15%에서 24%로, math 비중을 10%에서 13%로 높인다([[smollm3-midtraining]]).
3. Learning rate를 내린다. Llama 3, OLMo 2, Olmo 3, SmolLM3는 linear하게 0까지 내리고, DeepSeek-V3는 단계별 constant 값을 쓰며, Nemotron Nano 2와 GLM-5는 floor까지 내리고, MiniCPM은 exponential decay를 쓴다.
4. 선택적으로 이 phase checkpoint를 average한다(§6).

**계산 예: MiniCPM의 exponential decay.** MiniCPM은 `f(s − T) = 0.5^((s−S)/T)`를 인쇄하면서 “T is set to be 5000 steps (20B tokens)”라고 적는다([[minicpm]] §6.2). 인쇄된 그대로라면 \(s < S\)에서 지수는 음수이고 \(f > 1\)이 되므로, Eq. 1의 \(0 < f \le 1\)과 모순된다. Decay 시작부터 5,000 step의 half-life라고 읽으면(해석), \(η = 0.01\)일 때 5,000 step 뒤 learning rate는 \(0.01 \times 0.5^{5000/5000} = 0.005\)이고, 20,000 step 뒤에는 \(0.01 \times 0.5^{20000/5000} = 6.25 \times 10^{-4}\)이다. 저자들은 learning rate가 최대 learning rate의 10% 아래로 내려간 뒤에도 loss가 “still drops”한다고 쓴다(§6.4). 이 해석에서는 \(\log_2(10) \times 5000 = 16{,}610\) step에서 그 값에 도달한다.

**근거: decay phase의 길이.**

| Model | Decay 또는 anneal | Pretraining token의 비중 | Locus | Status |
|---|---|---:|---|---|
| DeepSeek-V3 671B | 10T까지 constant LR 뒤 4.3T에 걸쳐 cosine \(2.2\mathrm{e}{-4} \rightarrow 2.2\mathrm{e}{-5}\); 그 뒤 \(2.2\mathrm{e}{-5}\)에서 333B, \(7.3\mathrm{e}{-6}\)에서 167B | 29.1% (cosine) + 3.4% (마지막 500B) | [[deepseek-v3-recipe]] §4.2 | verified 값에서 도출 |
| Nemotron-Nano-12B-v2-Base | 20T 중 마지막 3.6T에 걸쳐 WSD \(4.5\mathrm{e}{-4} \rightarrow 4.5\mathrm{e}{-6}\) | 18.0% | [[nemotron-nano-2]] §2.5 | verified 값에서 도출 |
| SmolLM3-3B | blog: decay stage 10T → 11.1T, “final 10% training steps에서 linear decay to 0” | 11.1T의 9.9%(blog stage boundary; blog의 stated total은 11.2T); step의 11.1%(config) | [[smollm3-midtraining]]; [[smollm3-training-configs]] stage1_8T.yaml L206-212 | 도출 |
| MiniCPM 0.036B ablation | token의 10% decay는 “sufficient”, 2.5%는 “falls short” | 10% (명시됨) | [[minicpm]] §4.2 Fig. 5 | verified |
| OLMo 2 7B | 각 50B mid-training run에 걸쳐 0까지 linear decay | run당 1.3% | [[olmo-2-midtraining]] §2.3 | verified 값에서 도출 |
| Llama 3.1 405B | 마지막 40M token, LR를 0까지 linear, 128K context | 0.00026% | [[llama-3-recipe]] §3.4.3 | verified 값에서 도출 |

비중은 다섯 자릿수 범위에 걸쳐 있으므로, “anneal”이라는 이름은 서로 다른 operation을 가리킨다. Llama 3의 40M-token anneal은 floor가 \(8 \times 10^{-7}\)인 1,200,000-step cosine의 끝에서 오며, 보고서는 anneal 시작 learning rate를 인쇄하지 않는다([[llama-3-recipe]] §3.4.1, §3.4.3). Nemotron Nano 2의 3.6T decay는 stable learning rate에서 시작한다. Decay length의 controlled comparison은 작은 scale에 있다. Hägele et al.은 33M~360M parameter에서, 그리고 100B token으로 학습한 1B model에서 약 20% step의 cooldown이 length-matched cosine schedule과 맞먹는다고 보고한다([[cooldown-scaling-beyond-fixed-durations]] §3.2, §5). ≤1B parameter에서는 방향이 **replicated**됐다(MiniCPM 10%, Hägele et al. 20%). 이 장의 source 중 1B보다 큰 model에서 decay length를 시험한 것은 없다.

**근거: decay phase의 data가 하는 일.**

- MiniCPM Table 1([[minicpm]])에서, 2.4B stable checkpoint를 high-quality와 SFT data로 decay한 뒤 4B SFT token으로 fine-tune한 A-2는 C-Eval 52.6, MMLU 50.9, GSM8K 42.3을 기록했다. Pretraining data만으로 decay한 A-1은 각각 40.0, 44.6, 27.7이었다. 1.2B에서는 plain decay 뒤 SFT token을 두 배로 늘린 B-2의 MMLU가 47.9로 남았다(B-1도 47.9). 반면 decay 안에 SFT data를 넣은 B-3은 49.6이었다. B-3의 GSM8K는 B-1보다 낮았다(31.8 대 34.2). Setting당 한 run이고 seed는 보고하지 않았다. **결과(단일 연구).**
- Llama 3([[llama-3-recipe]] §3.1.3)는 GSM8k와 MATH training set으로 anneal하면 8B model의 validation score가 각각 24.0%, 6.4% 올랐고 405B에서의 효과는 “negligible”했다고 보고한다. 저자들은 “true few-shot learning capabilities and out-of-domain generalization”을 평가하기 위해 released annealing data에서는 benchmark training set을 제외했다. **결과(단일 연구).**
- OLMo 2([[olmo-2-midtraining]] Table 8)는 2T token 뒤 high-quality token 100B에서 각각 0까지 decay한 peak learning rate \(3\mathrm{e}{-4}\), \(6\mathrm{e}{-4}\)가 OLMES 73.8, 73.9를 냈다고 보고한다. Pretraining learning rate는 이 suite의 post-decay score를 바꾸지 않았고, GSM8K는 \(6\mathrm{e}{-4}\)에서 2.8점 높았다. **결과(단일 연구).**
- GLM-4.5는 stable phase 없는 cosine schedule로 mid-training 끝에서 \(2.5\mathrm{e}{-5}\)까지 decay하며, early experiment에서 “WSD schedule로 학습한 model은 general benchmark(SimpleQA, MMLU)에서 더 나쁘게 수행”했고 이것이 stable stage underfitting을 보인다고 쓴다. 수치는 인쇄하지 않았다([[glm-4-5-recipe]] §2.4). 여기서 WSD가 더 나쁘다고 보고한 source는 이것뿐이다. **결과(단일 연구), 수치 없음.**

**조건과 한계.** A-2 대 A-1은 decay data와 model이 본 SFT-format data 양을 둘 다 바꾼다. B-2는 1.2B에서만 SFT token 수를 통제한다. Llama 3 benchmark-anneal 결과는 training-set exposure와 capability gain을 섞기 때문에 저자들이 이를 제외했다. 이 연구들 중 tuned benchmark 밖의 held-out task family를 측정한 것은 없다.

**범용 모델에 대한 함의.** Decay phase는 benchmark-adjacent data를 학습하는 경우를 포함해 benchmark score를 가장 싸게 움직일 수 있는 곳이다. 따라서 이 phase의 evaluation gate에는 decay mix가 포함하지 않는 task가 필요하다(§3).
## §3 Mid-training budget과 evaluation gate

**정의.** **Mid-training**은 주 pretraining mix 뒤, SFT 전에 있는 stage로, 보통 learning-rate decay 중에 data distribution을 목표 capability 쪽으로 바꾼다. OLMo 2는 Abdin et al. (2024)과 OpenAI (2024)에서 이 용어를 가져왔고 이 stage를 “training FLOPs의 5–10%”라고 설명한다([[olmo-2-midtraining]] §2.3, footnote 4). SmolLM3는 long-context adaptation과 reasoning adaptation 모두에 이 용어를 쓴다([[smollm3-midtraining]]). GLM-4.5와 GLM-5는 context를 늘리고 repository·agent data를 추가하는 stage에 이를 쓴다([[glm-4-5-recipe]]; [[glm-5]]).

**문제.** Mid-training은 후보 mix를 비교하여 고른다. Release scale에서 후보를 완전히 비교하려면 후보마다 전체 mid-training run 비용이 들며, mix를 고르는 데 쓰는 benchmark가 그 mix와 겹칠 수 있다.

**Budget ledger.** 비율은 verified 값에 §1 식을 적용해 도출했으며, 그림도 이 비율을 표시한다.

| Model | Mid-training token (seen) | Pretraining token | f | Data focus | Locus |
|---|---:|---:|---:|---|---|
| OLMo 2 7B | 평균낸 \(3 \times 50\mathrm{B}\) | 3.90T | run당 1.3%; 모든 run 3.8% | high-quality web, FLAN, peS2o, Wikipedia, math | [[olmo-2-midtraining]] §2.3, Table 13 |
| OLMo 2 13B | 평균낸 \(3 \times 100\mathrm{B} + 1 \times 300\mathrm{B}\) | 5T | 100B run당 2.0%; 전체 12.0% | 반복을 포함한 동일 mix | §2.3, Table 13 |
| OLMo 2 32B | 평균낸 \(3 \times 100\mathrm{B} + 1 \times 300\mathrm{B}\) | 6.06T (§2.3); 7T (Table 9 caption) | 전체 9.9% (6.06T 기준) | 동일 mix | §2.3, Table 9 |
| Olmo 3 7B | 100B | 5.93T | 1.7% | math, code, QA, thinking trace, instruction data | [[olmo-3-base-stages]] Tables 5, 35 |
| Olmo 3 32B | 평균낸 \(2 \times 100\mathrm{B}\) | 5.5T | run당 1.8%; 두 run 3.6% | 동일 mix | Table 35 |
| SmolLM3-3B | 35B unique \(\times 4\) epoch \(\approx 140\mathrm{B}\) | 11.1T (blog decay stage 끝; stated total 11.2T) | seen 1.3%; unique 0.3% | reasoning trace (OpenThoughts3, Llama-Nemotron) | [[smollm3-midtraining]] |
| GLM-4.5 355B | \(500\mathrm{B}+500\mathrm{B}+100\mathrm{B}=1.1\mathrm{T}\) | 22T (15T + 7T) | 5.0% | repo code, synthetic reasoning, long document, agent trajectory | [[glm-4-5-recipe]] Fig. 3 |
| GLM-5 744B | \(1\mathrm{T}+500\mathrm{B}+50\mathrm{B}=1.55\mathrm{T}\) | 27T | 5.7% | SWE data, long document, agent trajectory | [[glm-5]] §1, §2.3 |

OLMo 2의 “training FLOPs의 5–10%” 진술은 모든 ingredient를 세면 13B와 32B total에 가깝다. 보고된 5.6T와 6.6T total 중 600B이므로 각각 token의 10.7%, 9.1%다. 그러나 단일 7B run은 pretraining token의 1.3%이고, 세 7B run을 모두 세어도 4.05T total의 3.7%다. 일반 진술은 FLOPs 단위이고 size별 비율은 token 단위이므로, 두 값은 서로 다른 것을 측정한다(**도출**).

**보고된 evaluation gate.**

1. **Microanneal.** OLMo 2는 candidate source와 general web data를 50/50으로 섞어 7B checkpoint를 anneal하고 learning rate를 linear하게 0까지 내린다. Microanneal 19개가 쓴 token은 총 130B로, “3개의 50B-token souping ingredient보다 적은 compute”다([[olmo-2-midtraining]] §4.4.2). Table 12에서 baseline GSM* 28.5에 비해, math/web 35/65 mix는 576M token으로 63.5에, 10/90 mix는 1.72B token으로 61.0에 도달했다. Olmo 3은 이를 5B target + 5B web token과 10B web-only anneal의 비교로 표준화한다([[olmo-3-base-stages]] §3.5.1).
2. **SFT를 포함한 integration test.** Olmo 3은 candidate mix에서 전체 100B anneal을 실행하고 각 결과를 fine-tune하여, “midtrain에서 관찰한 gain이 base model capability를 넘어선 improvement를 yield하는지 verify”한다(§3.5.1). Table 6의 Round 1, 3, 5에서 base average는 49.7 → 50.7 → 53.1, SFT average는 35.2 → 35.3 → 37.3이다.
3. **Development와 held-out split.** OLMo 2는 1,319개 GSM8K test example 중 200개(GSM*)로 mix 결정을 하고, 나머지 1,119개를 held out으로 보고한다([[olmo-2-midtraining]] §2 footnote 6). Dolmino math pool에는 GSM8K train split도 들어 있다(Table 5).
4. **Domain trade-off run.** Olmo 3은 의도적으로 편향된 100B mix 두 개를 학습했다. Math-code-thinking mix는 final mix의 Math 57.3, Code 31.2와 비교해 Math 60.8, Code 35.6을 얻었지만, GenQA는 73.1 대신 65.9였다(Table 7).
5. **Selection 전 decontamination.** Olmo 3은 Round 5 전에 모든 split에서 benchmark contamination을 제거했다. GSM8K contamination은 score를 올리지 않았으며, 저자들은 Marin을 따라 그 이유를 format mismatch로 본다(§3.5.4).
6. **Context stage-advance rule.** Llama 3은 “short-context evaluation에서 model performance가 완전히 회복”되고, 그 길이까지 ‘needle in a haystack’ task를 “perfectly solves”할 때만 각 context stage를 진전시킨다([[llama-3-recipe]] §3.4.2).

**Effect size 근거.** OLMo 2 Table 9(pretraining → pretraining & mid-training)에서 7B average는 53.0 → 62.9, MMLU는 59.8 → 63.7, GSM8K는 24.1 → 67.5다. 13B average는 58.9 → 68.3, 32B average는 66.3 → 73.3이다([[olmo-2-midtraining]]). Table 9는 development benchmark(MMLU, ARC-C, HellaSwag, WinoGrande, NQ, DROP)와 held-out evaluation(AGIEval, GSM8K, MMLU-Pro, TriviaQA)을 분리한다. 단, GSM8K 항목은 footnote 6의 뜻에서만 held out이다. Test example 200개가 mix 결정을 이끌었고, Dolmino math pool에는 GSM8K train split이 들어 있다. Olmo 3 Table 10에서 token 수를 고정한 intermediate mix에서 thinking trace와 instruction data를 빼면 base average가 50.7에서 48.8로, Math가 48.7에서 43.1로 내려갔다([[olmo-3-base-stages]]). 각각 **결과(단일 연구)**다.

**Gate가 찾아낸 formatting failure.** Olmo 3은 mid-training data에 chat special token을 넣으면 base model이 inference에서 그 token을 출력한다는 점을 찾았다. Microanneal에서 GSM8K는 49.43에서 0으로, CruxEval은 32.89에서 18.91로 떨어졌다. Chat template을 ordinary text로 쓰면 score는 각각 46.02와 29.65였다([[olmo-3-base-stages]] §3.5.4). Released mix는 대신 newline-based formatting을 쓴다. SFT에서 template이 미치는 영향은 ch-04를 보라.

**함의.** Mid-training ablation을 인쇄한 보고서(OLMo 2, Olmo 3)는 microanneal, broad base suite, SFT probe를 함께 써서 mix를 고르고, score를 잃는 domain도 보고한다. 범용 모델 run에는 세 가지가 모두 필요하다. 단일-domain microanneal은 target domain gain과 다른 곳의 loss를 함께 보이기 때문이다. Olmo 3 Tables 8–9에서 Reddit-to-Flashcards는 MMLU를 55.6 → 58.8로 높이는 대신 GSM8K를 22.4 → 21.2로 낮췄고, reasoning data는 GSM8K를 18.4 → 26.8로 높이는 대신 MMLU를 55.2 → 53.7로 낮췄다.

## §4 Context-extension ledger

**정의.** **Context extension**은 pretraining 때보다 긴 sequence length에서 계속 학습하는 것이며, 보통 position encoding을 바꾼다. **Adjusted base frequency(ABF)**는 RoPE base \(b\)를 키운다([[llama-2-long]]). **YaRN**은 차원별로 RoPE frequency를 재조정하고 attention logit에 temperature를 곱한다([[yarn]]). 방법 간 비교는 ch-32b에서 다룬다.

**문제.** Attention 비용은 sequence length와 함께 커지므로, 보고서들은 4K–8K에서 pretrain한 뒤 후반에 context를 늘린다([[llama-3-recipe]] §3.4.2가 그 이유를 제시한다). Extension stage는 long-context score를 올리되, 그 전에 측정한 short-context score를 내리지 않아야 한다.

**수식(RoPE wavelength).**

```
θ_i = b^(−2i/d),   λ_i = 2π / θ_i = 2π · b^(2i/d),   i = 0, …, d/2 − 1
```

- \(b\)는 RoPE base, \(d\)는 attention head dimension, \(θ_i\)는 차원 쌍 \(i\)가 position 하나마다 회전하는 양이다.
- \(λ_i\)는 차원 쌍 \(i\)가 한 바퀴 회전하는 데 필요한 position 수다.

**계산 예.** \(d=128\)일 때 가장 느린 쌍(\(i=63\))은 \(λ=2π·b^{126/128}\)이다. \(b=10{,}000\)이면 \(λ≈54{,}410\) positions, \(b=500{,}000\)(Llama 3)이면 \(λ≈2.56\mathrm{M}\), \(b=5{,}000{,}000\)(64K의 SmolLM3)이면 \(λ≈24.7\mathrm{M}\)이다. \(b\)를 키우면 pair 0을 제외한 모든 wavelength가 길어져, training window 안에서 한 바퀴 회전하는 pair 수가 줄어든다(**해석**; 보고서가 이 계산을 직접 싣지는 않는다). Qwen2.5-1M과 DeepSeek-V3의 YaRN temperature는 \(\sqrt{1/t}=0.1\ln(s)+1\)이며, \(s\)는 length ratio다. \(s=40\)이면 1.369, \(s=4\)이면 1.139다([[qwen-long-context-synth]] Eq. 1; [[deepseek-v3-recipe]] §4.3).

**Ledger.** `n.r.`은 확인한 source에서 보고하지 않았다는 뜻이다.

| 모델 | Pretrain 길이 → 목표 | Stage와 token | Position 방법과 설정 시점 | Long:short (단위) | Gate 또는 근거 | 위치 |
|---|---|---|---|---|---|---|
| Llama 3.1 405B | 8K → 128K | 6 stages, 약 800B tokens; stage별 길이 n.r. | pretraining에서 RoPE \(θ=500{,}000\) 설정; extension 때 변경 보고 없음 | n.r. | short-context eval 완전 회복 + 각 길이에서 NIAH | [[llama-3-recipe]] §3.2, §3.4.2 |
| Qwen2.5 0.5B–72B | 4,096 → 32,768 | final pretraining stage; tokens n.r. | 이 stage에서 ABF 10,000 → 1,000,000; inference에서는 YaRN + DCA로 131,072까지 | n.r. | 72B-Instruct RULER-128K가 DCA + YaRN으로 67.0 → 88.4(Table 16) | [[qwen-2.5-recipe]] §3.3 |
| Qwen2.5-Turbo (MoE, API) | 32,768 → 262,144 | 3 stages; tokens n.r. | base 10,000,000 | 최대 길이에서 40% / 더 짧은 길이에서 60%(sequences) | n.r. | [[qwen-2.5-recipe]] §3.3 |
| Qwen2.5-7B/14B-1M | 32,768 → 262,144 | 65,536 / 131,072 / 262,144; tokens n.r. | stage별 base 1M / 5M / 10M | 75% / 25%(sequences) | 14B RULER-128K 37.6 → 56.0 → 83.8 → 87.6(Table 2) | [[qwen-long-context-synth]] §3 |
| ProLong 8B (Llama-3-8B-Instruct에서 시작) | 8K → 512K | 64K에서 20B + 512K에서 20B(\(B=2^{30}\)) | base \(5×10^5→8×10^6→1.28×10^8\) | 64K에서 long 63%(textbook 3% 포함) / ShortMix 37%; share type 미명시 | Table 18: base 8e6 54.6 vs 4e6 48.7 vs 5e5 29.1 | [[prolong-recipe]] Table 9 |
| DeepSeek-V3 671B | 4K → 128K | 32K에서 1000 steps + 128K에서 1000; 도출 125.8B | decoupled key에 YaRN, \(s=40, α=1, β=32\) | n.r. | SFT 뒤 128K까지 NIAH(Fig. 8) | [[deepseek-v3-recipe]] §4.3 |
| Olmo 3 7B / 32B | 8,192 → 65,536 | 50B / 100B | \(θ=5×10^5\) 유지; full-attention layer에만 YaRN factor 8 | long 34% / 66%(tokens) | RULER dev, HELMET held out; Fig. 13 | [[olmo-3-base-stages]] §3.6; [[olmo-core-olmo3-configs]] |
| SmolLM3-3B | 4,096 → 64K(inference에서는 YaRN으로 128K) | 32K에서 50B + 64K에서 50B | \(θ\) 50,000 → 1.5M(blog) 또는 2,000,000(config) → 5M; pretraining부터 매 4번째 layer에 NoPE | n.r.(math, code, reasoning upsample) | ablation에서 RULER, HELMET | [[smollm3-midtraining]]; [[smollm3-training-configs]] |
| Nemotron-Nano-12B-v2-Base | 8,192 → 128k 목표 | Phase LC: 524,288에서 18.9B | position embedding 없음(hybrid Mamba-Transformer) | Phase 3 blend 80% / synthetic long-document QA 20%(weights) | Table 4(Nemotron-H 8B): 128k에서 RULER-128k 73.68 vs 512k에서 81.04 | [[nemotron-nano-2]] §2.1, §2.6 |
| GLM-4.5 355B | 4K → 128K | 32K stages 1T, 그 뒤 128K에서 100B | 32K extension에서 base 10,000 → 1,000,000; 128K n.r. | n.r. | n.r. | [[glm-4-5-recipe]] §2.3–2.4 |
| GLM-5 744B | 4K → 200K | 32K에서 1T, 128K에서 500B, 200K에서 50B | n.r. | n.r. | 200K stage가 128K window 안에서도 결과를 개선(§2.3에 수치 없음) | [[glm-5]] §2.3 |

Pretraining 대비 budget(**도출**): Llama 3 5.1%, GLM-5의 128K + 200K stages 2.0%, Olmo 3 32B 1.8%, SmolLM3 0.90%, DeepSeek-V3 0.85%, Olmo 3 7B 0.84%, GLM-4.5의 128K stage 0.45%, Nemotron Nano 2 0.095%다.

**근거 1: 목표 길이보다 길게 학습한다.** 네 source는 목표 길이보다 더 길게 학습한 뒤 목표 길이에서 score가 더 높았다고 보고한다. Nemotron-H 8B는 모두 synthetic data로 학습했을 때 RULER-128k가 512k에서 81.04, 256k에서 79.04, 128k에서 73.68이었다([[nemotron-nano-2]] Table 4). Qwen2.5-14B-1M은 262,144-token stage가 128K RULER를 83.8에서 87.6으로 올렸다([[qwen-long-context-synth]] Table 2). ProLong은 64K 대신 512K에서 학습한 4B tokens가 64K 평가에서 recall 98.5 대 95.0, re-rank 32.9 대 28.0을 냈다([[prolong]] Table 7). GLM-5는 128K 뒤의 200K stage가 “128K context window 안에서도 model performance를 further bolstered”했다고 쓴다([[glm-5]] §2.3, 수치 없음). 네 model family에서 **재현**됐지만, 각 연구는 setting마다 한 run만 썼다.

**근거 2: extension mix 안의 short data.** Olmo 3의 10B-token extension 실험에서 long 66% / short 34%는 OlmoBaseEval 일부를 2.5 points 낮췄고, long 34% / short 66%는 0.8 points 낮췄다([[olmo-3-base-stages]] §3.6.3). ProLong의 5B-token Llama-3-8B ablation에서는 long share가 커질수록 short-task average가 단조롭게 떨어졌고, SFT 뒤 long-context average는 long 60%에서 가장 좋았다([[prolong]] §3.2, Fig. 3). 두 연구 모두 long share가 커지면 short-task loss가 커진다는 방향은 **재현**했지만, 선호한 long share는 data, length, gate가 달라 34%와 60%로 다르다.

**근거 3: short-context regression은 여전히 생긴다.** Olmo 3 Table 13은 midtrained base와 extended base를 비교한다([[olmo-3-base-stages]]): 7B Math 59.8 → 54.4, GenQA 71.3 → 72.5; 32B Math 69.7 → 61.4, MMLU 76.9 → 76.2이다. Qwen2.5-14B-Instruct-1M은 128K Qwen2.5-14B-Instruct와 비교해 GPQA 45.5 → 39.9, LiveCodeBench 42.6 → 38.6인 반면 IFEval은 81.0 → 84.3이다([[qwen-long-context-synth]] Table 6). Qwen 보고서는 이를 비슷하다고 요약한다. Olmo 3 §3.7 본문은 Olmo 3와 OLMo 2를 비교할 뿐 Stage 2 → Stage 3 변화를 논평하지 않는다. 각각 **결과(단일 연구)**다.

**근거 4: 배치 시점.** Llama 3는 final anneal 전에 extension한다([[llama-3-recipe]] §3.4.2–3.4.3). Qwen2.5는 final pretraining stage에서 extension하고, Qwen2.5-1M은 intermediate base checkpoint에서 계속한다([[qwen-long-context-synth]] §3). Olmo 3는 mid-training 뒤, SmolLM3는 decay stage 뒤 reasoning mid-training 전, Nemotron Nano 2는 Phase 3 뒤에 extension한다. GLM-4.5와 GLM-5는 mid-training 안에서 extension한다. Olmo 3 보고서는 GLM 4.5가 “supervised finetuning 이후에만 extension을 적용한다”고 쓴다(§3.6). 그러나 이는 GLM-4.5의 Figure 3 및 §2.3과 모순되어, 이 장은 GLM-4.5 자체 보고서를 따른다. 어떤 source도 placement를 controlled experiment로 비교하지 않았다. **미해결 질문**이다.

**조건과 한계.** Olmo 3와 Qwen2.5-1M이 development metric으로 쓴 RULER는 synthetic이며 HELMET과 일부 겹친다(Olmo 3 footnote 25). ch-32c는 claimed context와 effective context가 왜 다른지 설명한다. Nemotron length ablation은 release된 12B model이 아니라 Nemotron-H 8B에서 했다. ProLong ablation은 8B parameters, 5B-token run이다.

## §5 Agentic mid-training ledger

**정의.** **Agentic mid-training**은 repository 수준 code, software-engineering history, 또는 synthetic agent trajectory를 mid-training에 넣어 SFT와 RL이 multi-step tool·code context를 이미 본 model에서 시작하게 만드는 stage다. Data construction은 ch-32d에서 다룬다.

| Setting | GLM-4.5 355B | GLM-5 744B |
|---|---|---|
| Stage 길이와 token | 32K에서 500B repo code; 32K에서 500B synthetic reasoning; 128K에서 100B long-context 및 agent data (Fig. 3) | 32K에서 1T; 128K에서 500B; 200K에서 50B (§2.3) |
| Repository data format | repository 하나의 file을 concatenate; model-filter한 issue, PR, commit을 diff-like format으로 표현 (§2.3) | repo file, commit diff, issue, PR, 관련 file을 하나의 sequence에 넣음; filtering 후 약 10M issue–PR pair, 약 160B unique token (§2.3) |
| Agent trajectory | 128K에서 “large-scale synthetic agent trajectories” (§2.3) | “later stages에서 up-sampled” (§2.3) |
| Agent-trajectory token share | 보고하지 않음 | 보고하지 않음 |
| Stage별 SWE-data share | 보고하지 않음 | 보고하지 않음 |
| Trajectory generator와 filter | mid-training에 대해서는 보고하지 않음 | 보고하지 않음 |
| Learning rate | cosine, mid-training 끝에서 \(2.5\times10^{-5}\)까지 decay (§2.4) | linear \(4\times10^{-5}\rightarrow1\times10^{-5}\) (App. A) |
| RoPE 또는 position setting | 32K extension에서 base 10,000 → 1,000,000 (§2.4) | 보고하지 않음 |
| Packing | mid-training에서만 best-fit packing (§2.3) | 비슷한 long text를 interleave하여 packing (§2.3) |
| Agentic share ablation | 보고하지 않음 | 보고하지 않음 |
| Source | [[glm-4-5-recipe]] | [[glm-5]] |

**근거.** 두 보고서 모두 agentic mid-training data가 agent benchmark 또는 general benchmark에 미치는 효과를 분리하지 않는다. 이들이 보고하는 gain에는 mid-training, SFT, RL이 함께 들어 있다. GLM-5 introduction은 mid-training을 “복잡한 workflow에서 안정성을 보장하기 위해 long-context agentic data에 특별히 초점을 맞춘다”고 설명한다([[glm-5]] §1). 이는 design statement이며 측정 결과가 아니다. Token share와 breadth에 미치는 효과는 **미해결 질문**이다.

**함의.** 범용 run은 이 표를 복사할 때 stage 길이와 data format은 복사할 수 있지만 agentic token share는 복사할 수 없다. 어느 보고서도 그 값을 인쇄하지 않기 때문이다. Share를 고르기 전에 ch-32d와 ch-32f의 ablation design이 필요하다.

## §6 명시된 단계로서의 checkpoint averaging과 merging

**정의.** **Checkpoint average(soup)**는 여러 checkpoint weight의 원소별 평균이다. **Linear merge**는 둘 이상의 model을 가중 평균하는 것이다.

```
θ_soup = (1/n) Σ_{k=1..n} θ_k          θ_merge = (1 − α) · θ_A + α · θ_B
```

- `θ_k`는 같은 architecture의 (n)개 run 또는 checkpoint의 weight이고, `α` ∈ [0, 1]은 model B에 주는 merge weight다.

**계산 예.** APO soup의 scalar weight 하나가 0.30((θ_A))이고 mid-training checkpoint에서는 0.50((θ_B))이라고 하자. SmolLM3의 0.9 / 0.1 merge는 (0.9 × 0.30 + 0.1 × 0.50 = 0.32)를 만든다. 따라서 병합된 각 weight는 post-training 값에서 mid-training 값 쪽으로 거리의 10%만 이동한다([[smollm3-midtraining]]).

**Ledger.**

| Model | 위치 | 평균내는 대상 | Weight | 근거 | Locus |
|---|---|---|---|---|---|
| Llama 3.1 405B | annealing 끝 | annealing 중 checkpoint(Polyak averaging) | n.r. | ablation 보고 없음 | [[llama-3-recipe]] §3.4.3 |
| OLMo 2 7B | mid-training | data order가 다른 3 runs × 50B | uniform | Table 14: 여섯 mix 모두에서 soup ≥ best single의 OLMES; GSM* mix A 71.0 → 74.0, mix E 60.5 → 43.0 | [[olmo-2-midtraining]] §4.5 |
| OLMo 2 13B, 32B | mid-training | 3 × 100B + 1 × 300B | uniform | 세 100B run만 평균내는 것보다 empirically better라고 서술했으나 수치는 없음 | §4.5 |
| Olmo 3 32B | midtraining | data-order seed가 다른 2 runs × 100B | uniform | Math cluster가 두 run보다 각각 +2.9, +1.6, MMLU는 약 +1 | [[olmo-3-base-stages]] §3.5.4 |
| Olmo 3 7B | midtraining | 없음 | — | “Initial experimentation for the 7B model did not show similar gains” | footnote 24 |
| Olmo 3 32B | long-context 끝 | step 10,000, 11,000, 11,921의 checkpoint | uniform | ablation 보고 없음 | §3.6.4 |
| SmolLM3-3B | APO 뒤 | APO checkpoint soup 뒤, (0.9 ×) soup + (0.1 ×) mid-training checkpoint | 0.9 / 0.1 | best-performing weights라고 서술; RULER는 128k까지 base score를 회복했지만 수치는 인쇄하지 않음 | [[smollm3-midtraining]] |
| Nemotron-Nano-v2 12B | RL 뒤 | reasoning-strong RL checkpoint와 chat-strong RL checkpoint | α를 0.1–0.9에서 0.1 간격으로 sweep | “values around 0.5 offered a good trade-off”(표 없음) | [[nemotron-nano-2]] §3.2 |

Olmo 3 32B의 step 11,921은 (100	ext{B tokens} / 8{,}388{,}608	ext{ tokens per batch})와 같으므로, 마지막으로 평균낸 checkpoint는 100B-token extension run의 끝이다(Table 35에서 **도출**).

**조건과 한계.** OLMo 2 caption은 souping이 best single checkpoint와 “consistently” 같거나 낫다고 하지만, 그 표의 mix E 행 자체는 best single run의 GSM* 60.5와 soup의 43.0을 보인다. Olmo 3은 32B에서 merge가 유용했지만 7B에서는 그렇지 않았다. SmolLM3의 recovery는 score 없이 보고됐다. 모든 경우가 **결과(단일 연구)**이며, averaging 이득은 model size와 metric에 따라 달라진다.

**함의.** Earlier-stage checkpoint와 병합하는 것은 이 source들에서 새 training 없이 이미 잃은 capability를 되돌린다고 보고된 유일한 방법이다(SmolLM3). Merge weight는 release마다 선택하며, 평가는 later stage가 낮춘 capability를 반드시 포함해야 한다. Merge method는 ch-30c에서 다룬다.

## §7 검증한 것, 서로 충돌하는 것, 그리고 보고되지 않은 것

2026-09-14와 2026-09-15에 수행한 검증에서는 논문 또는 blog와 공개된 configuration 사이, 또는 한 report의 서로 다른 부분 사이에서 네 가지 충돌을 발견했다.

1. SmolLM3의 4k→32k RoPE \(\theta\): blog는 1.5M이라고 적지만, `long_context_4k_to_32k.yaml`은 2,000,000이다([[smollm3-training-configs]]).
2. Olmo 3 7B의 midtraining 및 extension peak LR: Table 35는 \(2.074\mathrm{e}{-4}\)라고 적지만, 두 7B script는 모두 \(2.0712\mathrm{e}{-4}\)이다([[olmo-core-olmo3-configs]]). 7B extension script의 `max_duration`(5T tokens)과 `hard_stop`(step 597046)은 50B stage와 대응하지 않는다. 32B long-context script의 `load_path`는 pretraining checkpoint를 가리키지만, 논문은 merged midtrained model에서 extension을 시작한다.
3. ProLong 512K stage peak LR: Table 9는 \(1\mathrm{e}{-5}\)라고 적지만, `train_512K.sh`는 \(5\mathrm{e}{-6}\)이다([[prolong-recipe]]).
4. OLMo 2 32B pretraining token: §2.3은 6.06T, Table 9 caption은 7T라고 적는다([[olmo-2-midtraining]]).

독자가 기대하지만 source가 인쇄하지 않은 값은 다음과 같다. Llama 3의 stage별 extension length와 token; DeepSeek-V3의 extension token; Qwen2.5와 Qwen2.5-1M의 extension token과 learning rate; Llama 3, DeepSeek-V3, SmolLM3, GLM-4.5, GLM-5의 long:short ratio; GLM-4.5와 GLM-5의 agentic token share; Llama 3의 merge weight.

Library card [[olmo-2]], [[olmo-3]], [[smollm-3]]는 이 검증보다 앞서 작성되었으며 report와 다른 값을 담고 있다. 예를 들어 OLMo 2에 대해 “cooldown에서 32K로 extension했다”고 적지만 실제로는 전체 training을 4096에서 수행한다. 이 장에서 이 세 model에 쓰는 값은 검증한 excerpt [[olmo-2-midtraining]], [[olmo-3-base-stages]], [[smollm3-midtraining]]에서 가져온다.


## 레시피

각 행은 행에서 링크한 검증 완료 card와 excerpt에서 가져왔다. “v3”, “v4” 등은 arXiv version을 뜻한다. 값은 인쇄된 그대로이고 단위는 §1을 따른다.

| Model(정확한 release) | Size | Stage | Setting | 값 | Source location | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | pretrain-decay/anneal | final anneal | 마지막 40M tokens; LR을 0까지 선형 감소; 128K context; high-quality source upsampled | arXiv:2407.21783v3 §3.4.3 ([[llama-3-recipe]]) | 2026-09-14 검증 | ablation 보고 없음 |
| Llama 3.1 405B | 405B | merge | anneal averaging | annealing 중 checkpoint의 Polyak averaging; weight 미인쇄 | v3 §3.4.3 | 2026-09-14 검증 | ablation 보고 없음 |
| Llama 3 8B (experiment) | 8B | eval-gate | anneal-as-data-evaluation | 50% 학습 8B; 40B tokens 동안 LR을 0까지 선형 감소; new data 30%, default mix 70% | v3 §3.1.3 | 2026-09-14 검증 | scaling-law run보다 효율적이라고 서술, 수치 없음 |
| Llama 3.1 405B | 405B | long-context | stages; tokens; position | 8K → 128K 여섯 stage; 약 800B tokens; pretraining부터 RoPE θ 500,000; stage별 length 미인쇄 | v3 §3.2, §3.4.2 | 2026-09-14 검증 | short-context eval 회복 및 NIAH 해결 시 stage 진행 |
| DeepSeek-V3 | 671B / 37B act. | pretrain-decay/anneal | decay | constant LR로 10T 뒤 4.3T cosine 2.2e-4 → 2.2e-5; 이어333B at 2.2e-5, 167B at7.3e-6 | arXiv:2412.19437v2 §4.2 ([[deepseek-v3-recipe]]) | 2026-09-14 검증 | ablation 없음 |
| DeepSeek-V3 | 671B / 37B act. | long-context | phases | decoupled key k^R의 YaRN(s40, α1, β32); 32K batch1920×1000 steps;128K batch480×1000; LR7.3e-6 | v2 §4.3 | 2026-09-14 검증 | Fig.8 NIAH가 SFT 뒤128K까지 |
| DeepSeek-V3 | 671B / 37B act. | long-context | tokens | 125.8B(K=1,024,batch=sequence) 또는122.9B(K=1,000) | v2 §4.3 input | 도출 | V3.1:630B phase가 V3 phase의 “10-fold” ([[deepseek-v3.1-recipe]]) |
| Nemotron-Nano-12B-v2-Base | 12B | pretrain-decay/anneal | schedule; phases | 20T horizon; WSD stable4.5e-4,min4.5e-6, 마지막3.6T decay; 60%,90% phase switch; batch768×8192 | v4 §2.2,§2.5 ([[nemotron-nano-2]]) | 2026-09-15 검증 | schedule ablation 없음 |
| Nemotron-Nano-12B-v2-Base | 12B | long-context | Phase LC | 524,288에서18.9B; constant LR4.5e-6; batch12; Phase3 weight×0.8+synthetic long-doc QA20% | v4 §2.6 | 2026-09-15 검증 | Table4 RULER-128k: 128k/256k no synth/256k/512k=73.68/70.19/79.04/81.04 |
| MiniCPM-2.4B, 1.2B | 2.4B,1.2B | pretrain-decay/anneal | decay data; form | pretraining+high-quality SFT; exponential anneal, “T is set to be 5000 steps (20B tokens)”; stable LR0.01 | v3 §6.2 ([[minicpm]]) | 2026-09-15 검증 | A-2/A-1: C-Eval52.6/40.0,GSM8K42.3/27.7(각1run) |
| MiniCPM ablation models | 0.036B | pretrain-decay/anneal | decay length | total token10%는 충분,2.5%는 부족 | v3 §4.2 Fig.5 | 2026-09-15 검증 | 40N,60N,80N loss curves |
| SmolLM3-3B | 3B | pretrain-decay/anneal | decay stage(blog) | 10T→11.1T; web63%,code24%,math13%; final10% steps 선형 0 decay | [[smollm3-midtraining]] | 2026-09-15 검증 | 3B 50B–100B mixture ablation, 값 미인쇄 |
| SmolLM3-3B | 3B | pretrain-decay/anneal | decay(config) | start step4,198,001;522,000/4,720,000 decay;2,359,296 tok/step→약9.90T start,약1.23T tokens | [[smollm3-training-configs]] | 도출 | n/a |
| OLMo2 7B | 7B | pretrain-stable | truncation | 5T 동안3.0e-4의10%까지 cosine,4T 후 truncate; warmup2000 | v3 Table3 §4.1 ([[olmo-2-midtraining]]) | 2026-09-15 검증 | Table8 peak3e-4/6e-4 뒤 OLMES73.8/73.9 |
| OLMo2 7B | 7B | mid-train | budget; schedule | 3 runs×50B Dolmino, LR 선형0, 평균 | v3 §2.3,§4.5 | 2026-09-15 검증 | Table14:6 mix OLMES soup≥single; mixE GSM*60.5→43.0 |
| OLMo2 13B,32B | 13B,32B | mid-train | budget | 3×100B+1×300B, 평균 | v3 §2.3,§4.5 | 2026-09-15 검증 | three100B 평균보다 empirically better,수치 없음 |
| OLMo2 7B | 7B | mid-train | 50B mix(%) | DCLM47.2,FLAN16.6,Dolmino Math20.8,Wikipedia7.11,peS2o5.85,StackExchange2.45 | v3 Table13 | 2026-09-15 검증 | Table12 math35/65 GSM*63.5,10/90=61.0 |
| OLMo2(all) | 7B | eval-gate | microanneal | target+web50/50,LR 선형0;19runs,130B; GSM*=1,319 GSM8K 중200 | v3 §4.4.2,§2fn6 | 2026-09-15 검증 | Table12 |
| Olmo3 7B | 7B | mid-train | schedule;batch;tokens | warmup0;peak2.074e-4;선형0;256×8192=2,097,152;100B | v2 Table35 ([[olmo-3-base-stages]]) | 충돌 | Table6 Round1/3/5; script가 다음 행처럼 다름 |
| Olmo3 7B script | 7B | mid-train | LR;batch;tokens | `LR=0.00020712352850360292`;`GLOBAL_BATCH_SIZE=2**21`;`MAX_TOKENS=100_000_000_000`;`LinearWithWarmup(warmup=0,alpha_f=0.0)` | [[olmo-core-olmo3-configs]] | 충돌 | ablation 없음 |
| Olmo3 32B | 32B | mid-train | budget;merge | data-order seed 다른2×100B uniform average;peak2.071e-4;512×8192 | v2 Table35 §3.5.4 | 2026-09-15 검증 | single보다 Math+2.9/+1.6;7B gain 없음 |
| Olmo3 7B,32B | 7B,32B | mid-train | mix(99.95B share) | CC HQ22.5,Dolmino Math10.7,StackEdu FIM10,CraneCode10,Reddit Flashcards5.9,Flan5,Nemotron Synth QA5; thinking/instruction 별도 | v2 Table5 | 2026-09-15 검증 | Table10 thinking+instruction 평균50.7/48.8 |
| Olmo3 | 7B | eval-gate | microanneal;integration |5B target+5B web 대10B web;100B anneal 뒤SFT|v2§3.5.1|2026-09-15 검증|Tables6–9|
| Olmo3 7B |7B|long-context|schedule;length;tokens|8192→65536;warmup200;peak2.074e-4;선형0;64×65536=4,194,304;50B|v2 Table35|충돌|Fig13e 1B–100B budget이 RULER 상승|
| Olmo3 7B script |7B|long-context|position;LR|`YaRNRoPEScalingConfig(factor=8,beta_fast=32,beta_slow=1,old_context_len=8192)`;LR2.0712e-4;`warmup=200`|[[olmo-core-olmo3-configs]]|충돌|Fig13a full-attention YaRN 최상|
| Olmo3 7B,32B |7B,32B|long-context|mix;packing|long34%(olmOCR PDF+synthetic CWE/REX),Dolmino short66%;gzip20/20 filter;best-fit packing;intra-doc mask|v2§3.6 Table11|2026-09-15 검증|10B에서34/66 −0.8,66/34 −2.5|
| Olmo3 32B |32B|long-context|tokens;merge|100B;128×65536;steps10000,11000,11921 평균|v2 Table35§3.6.4|2026-09-15 검증|averaging ablation 없음|
| SmolLM3-3B |3B|mid-train|reasoning mid-training|35B×4epochs(약140B);ChatML;wrapped packing|[[smollm3-midtraining]]|2026-09-15 검증|ablation 없음,RULER drop 연결|
| SmolLM3-3B blog |3B|long-context|stages|50B 4k→32k θ1.5M;50B32k→64k θ5M;math/code/reasoning upsample;inference YaRN128k|blog|충돌|long-doc upsample가 RULER/HELMET 못 올림(수치 없음)|
| SmolLM3-3B config |3B|long-context|stages|θ2M;32768×72×20000;LR2e-5,warmup1000,19000 step cosine0; stage2 θ5M,65536×36×22000|[[smollm3-training-configs]]|충돌|도출47.2B,51.9B|
| SmolLM3-3B |3B|merge|recovery|APO soup;MergeKit 0.9×soup+0.1×mid-training|blog|2026-09-15 검증|tried merge 중 best,RULER가128k까지 base 회복,수치 없음|
| Qwen2.5 open |0.5B–72B|long-context|length;base|final pretraining 4096→32768;ABF10000→1000000;token 미제시|[[qwen-2.5-recipe]]|2026-09-14 검증|ablation 없음|
| Qwen2.5-Turbo |미보고|long-context|stages;mix|32768→65536→131072→262144;base10M;max length40%,short60%|v2§3.3|2026-09-14 검증|ablation 없음|
| Qwen2.5-7B/14B-1M|7B,14B|long-context|stages;bases;mix|65536/131072/262144;base1M/5M/10M;max75%,short25%;tokens/LR/batch 미보고|[[qwen-long-context-synth]]|2026-09-14 검증|Table2 RULER37.6/56.0/83.8/87.6|
| ProLong-64k-Base|8B|long-context|stage1|Llama3-8B-Instruct 시작;20B(B=2^30)64K;base8e6;repo30%,book30%,textbook3%,ShortMix37%;LR1e-5,warmup10%,cosine1e-6;batch4M|[[prolong-recipe]] Table9|2026-09-14 검증|Table18 8e6=54.6 대4e6=48.7;Fig3 long60%|
| ProLong-512k-Base|8B|long-context|stage2|512K20B;base1.28e8;peak LR1e-5(paper) 대5e-6(script)|v4 Table9;train_512K.sh|충돌|Table7 64K eval에서512K 우세|
| GLM-4.5|355B/32B act.|mid-train|stages|32K repo500B;32K synthetic reasoning500B;128K long+agent100B;RoPE10000→1M|[[glm-4-5-recipe]]|2026-09-14 검증|ablation 없음|
| GLM-4.5|355B/32B act.|mid-train|LR|cosine,끝2.5e-5|v1§2.4|2026-09-14 검증|early WSD SimpleQA/MMLU 더 나쁨,수치 없음|
| GLM-5|744B/40B act.|mid-train|stages;LR|32K1T;128K500B;200K50B;LR4e-5→1e-5|[[glm-5]] §2.3 App.A|2026-09-15 검증|200K가128K 내 결과도 개선,수치 없음|
| GLM-5|744B/40B act.|mid-train|SWE/agent data|약10M issue–PR,약160B unique;agent share 미보고|v2§2.3|2026-09-15 검증|ablation 없음|

**소규모 범용 run의 출발점.** 위 검증 행 또는 두 값을 모두 밝힌 충돌 행에서만 가져온 값이며, whole configuration을 ablation한 source는 없다. 4K–8K pretrained 3B–8B RoPE model을 32K–65K로 늘린다면: (1) Olmo3 7B처럼 mix를 바꿔 100B(5.93T의1.7%) midtraining 후 decay하고 5B+5B microanneal, 10B web-only control, SFT 뒤 100B integration test를 쓴다. (2) Olmo3의 long34%/short66%(65536에서50B,64-sequence)와 ProLong의 long63%(5B ablation 최선은60%)는 서로 다르다. 두 10B run이 가능하면 step6 gate에서 둘을 비교하고, 아니면 두 source에서 long share 증가가 short-task score를 낮췄으므로34%부터 시작한다. (3) Olmo3 full-attention YaRN factor8,θ5e5; SmolLM3 θ1.5M(blog)/2M(config)→5M; ProLong base8e6(54.6 대4e6의48.7) 중 하나를 기록해 선택한다. (4) Olmo3 warmup200→2.074e-4 선형0, SmolLM3 warmup1000→2e-5 cosine0, ProLong10%→1e-5 cosine1e-6 중 하나로 re-warm/decay한다. Peak LR은약20배 차이이고 ablation 없음. (5) Olmo3처럼 best-fit packing+intra-document mask(ProLong mask54.6 대53.6). (6) 길이별 RULER dev,HELMET held-out,Math 포함 full short suite로 stage 전후 gate한다(Olmo3 7B Math59.8→54.4). (7) post-training 뒤 RULER가 떨어지면 strong long-context midtraining checkpoint와0.9/0.1 merge 후 short suite도 다시 실행한다.

## 일반화 관점

**(a) Breadth를 넓히는 것.** Thinking+instruction의 broad mix는 고정 token에서 모든 OlmoBaseEval cluster를 올렸다(48.8→50.7,단일 연구). Balanced mix는 GenQA73.1을 유지했지만 math-code-thinking mix는65.9였다. Short66%는 short34%의−2.5보다−0.8만 cost했고, ProLong도 short share와 short-task average가 함께 올랐다(방향 재현). Target보다 긴 학습은 Nemotron,Qwen,ProLong,GLM5 네 family에서 target score를 올렸다(재현). OLMo2 6 mix soup과 Olmo3 32B merge도 breadth 신호다.

**(b) Narrowing/forgetting.** Math-code-thinking은 MC Non-STEM77.4→69.6, reasoning microanneal은 MMLU55.2→53.7. Extension은 Olmo3 Math(7B59.8→54.4;32B69.7→61.4), Qwen GPQA45.5→39.9를 낮췄다. SmolLM3 reasoning mid-training 뒤 RULER가 merge 전 하락했고 APO data는24k로 제한됐다. Benchmark set anneal은 Llama3 8B GSM8k+24.0%이나 out-of-domain 근거가 없고, special chat token은 GSM8K49.43→0을 만들었다. Soup도 mixE GSM*60.5→43.0처럼 한 metric을 낮출 수 있다.

**(c) 측정.** 각 stage 전후 per-cluster base suite, 각 candidate mix의 SFT probe, decision benchmark의 held-out split(OLMo2 GSM*:dev200/held-out1119)과 full decontamination, RULER development/HELMET held-out을 쓴다. Olmo3 mix rounds는 earlier checkpoint에서 했고 final midtraining number가 아니며 Round1/3는 non-decontaminated다. RULER/LongBench-Chat은128K/100K에서 끝나고 대부분 ablation은 seed 없는 single run이다.

## 흔한 실수와 탐지 방법

| 실수 | 증상 | 확인 |
|---|---|---|
| 단위 없는 budget 복사 | 2.4% token 차이 또는 seen token4× 부족 | K를 명시해 steps×batch(sequence)×length,epoch 재계산 |
| 모든 soup ingredient 합산 | share가 checkpoint 한 개 compute보다3× 큼 | per-run과 total 분리(OLMo2 ingredient3.95T,total4.05T) |
| blog RoPE를 config와 대조 안 함 | release와 첫 stage score가 다름 | SmolLM3 θ1.5M 대2M diff |
| script duration을 budget으로 읽음 |50B stage가 trillions token 또는 wrong stop|`max_duration`,`hard_stop`을 paper와 대조|
| 대부분 long으로 extension|RULER 상승,short suite 하락|34/66 대66/34 10B control|
| mix 포함 benchmark로 선택|decontam 뒤 gain shrink|holdout+decontaminated winning rerun|
| special chat token 잔존|`<|im_start|>` 출력,0점|sample grep와 포함/제외 microanneal|
| soup 항상 도움이라고 믿음|GSM*−17.5|ingredient와 soup을 metric별 평가|
| Llama3 stage length 지어냄|per-stage token을 Llama3에 귀속|여섯 stage와약800B만 인쇄|
| secondary summary 신뢰|lab own report와 모순|primary report 확인|
| post-training 뒤 long check 생략|DPO/APO/SFT 뒤 RULER하락|stage마다 RULER와0.9/0.1 merge test|

## 이해 점검

1. DeepSeek-V3의125.8B/122.9B가 “32K” 해석에 따라 갈리는 이유, Olmo3 123B와 V3.1 “10-fold”가 이를 결정하지 못하는 이유, 이를 결정할 printed field를 설명하라.
2. OLMo2의5–10% FLOPs와 single7B run의1.3% token이 souping과 denominator에서 어떻게 함께 나오는지, release checkpoint 한 개 compute를 예측하는 수가 무엇인지 설명하라.
3. MiniCPM A-2/A-1 C-Eval+12.6과 B-3/B-1 GSM8K 하락을 사용해 B-2 control과 decay SFT의 범용성 함의를 설명하라.
4. Olmo3 short66%에도 Math가5.4 하락한 두 mechanism과 이를 가르는 ablation 하나를 제안하라.
5. Nemotron concatenate-and-chunk와 Olmo3 packing 변경으로, 짧은 document에도 training sequence length가 중요한 이유를 설명하라.
6. 0.9/0.1 merge가 post-training을 전부 되돌리지 않고 RULER를 회복하는 이유와 post-training gain 취소 여부를 보일 측정을 설명하라.
7. Fixed budget에서 GLM agentic share의 agent/general short-context 효과를 재는 mid-training ablation을 설계하라.

## 연결

- **Previous:** ch-32d — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training(§5 data construction).
- **Next:** ch-32f — Lab: Annealing and Context Extension with a Short-Context Regression Gate(Recipe/§4 gate 적용).
- **Dependency:** ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation(RULER/HELMET 한계).
- **Dependency:** ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures(§1–§2 denominator/schedule).
- **Mechanisms:** ch-32, ch-32b; **Related:** ch-32a, ch-30c.

## 출처

- [[llama-3]], [[llama-3-recipe]] — 마지막 4,000만 토큰 anneal과 Polyak averaging(§3.4.3), anneal을 데이터 평가로 사용하는 방법과 GSM8k/MATH anneal 결과(§3.1.3), 8,000억 토큰을 연장한 6단계 일정과 단계 통과 기준(§3.4.2), RoPE θ(§3.2)를 다룬다.
- [[olmo-2]] — report card. 이 문서의 mid-training 값은 검증 이전 값이므로, 여기서는 [[olmo-2-midtraining]]으로 대체한다.
- [[olmo-2-midtraining]] — 검증한 발췌 자료. schedule truncation(Table 3, §4.1), Dolmino 토큰 예산과 soups(§2.3, §4.5), 데이터 혼합(Table 13), microanneal(Table 12), Table 9의 효과, Table 14의 soups를 제공한다.
- [[olmo-3]] — report card. 단계별 값은 [[olmo-3-base-stages]]로 대체한다.
- [[olmo-3-base-stages]] — 검증한 발췌 자료. Table 35의 단계 설정, Dolmino Mix(Table 5), 혼합 데이터 평가(Tables 6–10), special-token 결과, Longmino mix(Table 11), long/short ablation(§3.6.3), Table 13의 단계 비교, 모델 병합을 다룬다.
- [[olmo-core-olmo3-configs]] — 공개된 mid-training 및 long-context 스크립트. 학습률, YaRN, batch, duration 필드와 Table 35와의 차이를 제공한다.
- [[smollm-3]] — report card. mid-training 값은 [[smollm3-midtraining]]으로 대체한다.
- [[smollm3-midtraining]] — 블로그 발췌 자료. pretraining 단계와 decay, 각각 500억 토큰인 두 연장 단계, reasoning mid-training, 0.9 / 0.1 병합을 다룬다.
- [[smollm3-training-configs]] — nanotron 설정. decay step, 연장 단계의 θ, 학습률, batch, step 수와 블로그의 θ 값 충돌을 제공한다.
- [[qwen-2.5]], [[qwen-2.5-recipe]] — ABF를 사용한 4,096 → 32,768 → 1,000,000 확장, Qwen2.5-Turbo 단계와 40/60 혼합, DCA + YaRN 결과(§3.3, Tables 16–17)를 다룬다.
- [[qwen-long-context-synth]] — Qwen2.5-1M의 단계 길이, base 값, 75/25 혼합, 단계별 RULER 점수(Table 2), short-context 비교(Table 6), YaRN temperature(Eq. 1)를 제공한다.
- [[prolong]], [[prolong-recipe]] — Table 9의 recipe, long/short ablation(Fig. 3), RoPE base ablation(Table 18), 512K 대 64K 결과(Table 7), document masking(Table 20), 학습률 충돌을 다룬다.
- [[deepseek-v3]], [[deepseek-v3-recipe]] — pretraining decay와 마지막 저학습률 토큰(§4.2), YaRN 연장 단계(§4.3)를 다룬다.
- [[deepseek-v3.1-recipe]] — V3를 기준으로 설명한 6,300억 및 2,090억 토큰 연장 단계(model card). §1에서 교차 검증 자료로 사용한다.
- [[nemotron-nano-2]] — 검증한 발췌 자료. 3.6T 토큰에 걸친 WSD decay, 524,288 토큰의 Phase LC, 합성 long-document QA 혼합, 길이 ablation(Table 4), checkpoint interpolation을 다룬다.
- [[glm-4-5]], [[glm-4-5-recipe]] — mid-training 단계와 토큰 예산(Fig. 3, §2.3), RoPE base 변경과 학습률 일정(§2.4)을 다룬다.
- [[glm-5]] — 검증한 발췌 자료. 32K/128K/200K mid-training 단계, SWE 데이터 크기, long-context 데이터, Appendix A의 학습률을 제공한다.
- [[minicpm]] — 검증한 발췌 자료. WSD 정의(Eq. 1), 10% decay 결과(Fig. 5), decay 단계의 SFT 데이터(Table 1), MiniCPM 단계 설정과 명시된 decay 공식을 다룬다.
- [[cooldown-scaling-beyond-fixed-durations]] — 3,300만–10억 파라미터에서 cosine과 일치한 20% cooldown(§3.2, §5). decay 길이 재현에 사용한다.
- [[llama-2-long]] — adjusted base frequency(ABF)의 정의 출처.
- [[yarn]] — YaRN과 attention temperature의 정의 출처.
