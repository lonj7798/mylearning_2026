<!-- chapter: ch-32
     track: sft
     title: Multi-Stage Pipelines — Mid-Training, Cold-Start, Long-Context
     deps: [ch-31]
     sources: [[interplay-pretraining-midtraining-rl]], [[olmo-3]], [[olmo-2]],
              [[deepseek-r1]], [[long-context-llama3]], [[prolong]], [[longalign]],
              [[llama-3]], [[qwen-3-5]], [[front-loading-reasoning]]
     figures: figures/pipeline-stages.html
-->

# 32장 — Multi-Stage Pipelines: Mid-Training, Cold-Start, Long-Context

> **핵심 통찰.** 2025년에 이르러 "SFT"는 단일 stage가 아니라 **five-stage pipeline** 안의 하나의 checkpoint가 되었다. 그 pipeline은 *pretrain → mid-training → long-context extension → SFT → RL*이며, 각 stage는 고유한 data mix, token budget, eval gate, failure mode를 가진다. 개념적 변화가 가장 큰 곳은 mid-training이다. 예전에는 비공식적인 "cooldown" 또는 "annealing" step으로 취급되던 것이 이제는 RL이 대체할 수 없는 reusable prior를 설치하는 **distinct distribution-shift stage**가 되었다. [[interplay-pretraining-midtraining-rl]]은 controlled evidence를 제공한다. Fixed compute 아래에서 mid-training은 RL-only post-training보다 낫고, RL은 이전 stage들이 모델의 *edge of competence*에 headroom을 남겨두었을 때만 capability를 확장한다.
>
> **가이드라인.** 각 stage가 특정 deliverable을 소유한다고 보라. Pretrain은 **broad priors**를 소유한다([[front-loading-reasoning]] 기준 quality보다 diversity). Mid-training은 small high-quality mix를 통해 **reusable structure**(math, code, long-doc comprehension)를 소유한다(OLMo 3의 Dolmino = 2.2T pool에서 나온 **100B tokens**). Long-context extension은 **position encoding + coherent long documents**를 소유한다([[long-context-llama3]]: 10K → 500K RoPE base, 여섯 sub-stage에 걸친 ~800B tokens). SFT는 target behaviour의 **format and cold-start**를 소유하며, R1-Zero의 language-mixing을 고치는 R1의 **~800K trace** cold-start를 포함한다. RL은 **edge-of-competence exploration**을 소유한다. 각 stage는 targeted eval suite로 gate하라. 단일 aggregate benchmark로 gate하지 마라.

---

## 1. 왜 "mid-training"은 이제 실제 stage인가

2023년까지 지배적인 그림은 두 stage였다. Web-scale mix로 pretrain하고, SFT + RLHF로 post-train하는 것이다. Mid-training은 pretrain cooldown 또는 SFT warmup 안에 보이지 않게 접혀 있었다. 세 가지가 분리를 강제했다.

1. **Cooldown이 더 이상 pretraining처럼 보이지 않게 되었다.** OLMo 2의 Stage 2는 이미 3.9T token의 OLMo-Mix-1124 pretraining 이후 curated "Dolmino" mix로 50B token을 돌렸다([[olmo-2]]). Data distribution은 눈에 띄게 달랐다(math, code, high-quality web이 up-weighted되었고 pretrain mix에서 그대로 sampling되지 않았다). OLMo 3는 이를 고유 dataset(Dolma 3 Dolmino, **~2.2T high-quality pool에서 나온 ~100B tokens**)과 고유 compute allocation([[olmo-3]] 기준 pretrain 1024 H100s, post-train 256 H100s와 달리 **128 H100s**)을 가진 **named stage**로 공식화한다.
2. **Reasoning data가 SFT-sized가 아니게 되었다.** Open-Thoughts-style rejection-sampled long CoT trace는 각각 8K-32K token인 sequence 수십만 개에 달한다. 이는 *SFT에는 너무 크고 너무 specific*하며 *pretraining에는 너무 narrow*한 multi-billion-token data mass이다. OLMo 3-Think, Qwen 3, Phi-4-reasoning은 모두 이를 dedicated mid-training pass로 보낸다.
3. **Controlled ablation이 이 stage가 RL로 다시 할 수 없는 일을 한다는 것을 보였다.** [[interplay-pretraining-midtraining-rl]]은 extrapolative(harder composition)와 contextual(new surface form) generalization을 독립적으로 측정할 수 있는 synthetic-reasoning testbed를 계측한다. Matched compute budget 아래에서 **reusable prior에 대한 mid-training은 같은 budget을 RL-only post-training에 쓰는 것보다 우수하다**. RL도 여전히 도움은 된다. 하지만 pretraining이 headroom을 남겨두었고 RL prompt가 *edge of competence*에 놓일 때뿐이다. Mid-training은 그 headroom을 설계하는 stage이다.

덜 알려진 네 번째 이유는 **process supervision**이다. [[interplay-pretraining-midtraining-rl]]은 RL 중 process-level reward가 reward-hacking을 줄이고 reasoning *fidelity*(final-answer accuracy만이 아니라 structural correctness)를 개선한다는 것을 보여준다. Process reward는 모델이 만드는 intermediate step이 이미 parsable하고 semantically grounded할 때만 작동한다. 이는 structured data(math problem with step-annotated solutions, test trace가 있는 code)에 대한 mid-training이 raw-web pretrain이나 instruction-formatted SFT보다 더 안정적으로 설치하는 속성이다.

공개된 곳에서 OLMo 3, Llama 3, DeepSeek-R1의 attested token scale을 붙인 2025-era pipeline은 다음과 같다.

| Stage | What it owns | Representative data mix | Budget (OLMo 3 7B) | Eval gate |
|---|---|---|---|---|
| Pretrain | Broad priors, vocabulary coverage | Dolma 3 Mix: web + code + math + science, **5.9T tokens** | 1024 H100s | Broad LM perplexity; MMLU; code HumanEval baseline |
| Mid-training | Reusable structure on harder distributions | Dolmino: math/code/science/reading/IF, **100B from 2.2T pool** | 128 H100s | Targeted: GSM8K, MATH, HumanEval, MMLU-Pro, IFEval |
| Long-context | Position encoding + long-doc comprehension | Longmino: **50B** from 639B long-doc pool | 128 H100s (shared) | NIAH heatmap, RULER, BABILong |
| SFT (cold-start) | Format, target behaviour, readability | Dolci-SFT: rejection-sampled CoT + instructions | 256 H100s | Format compliance, MT-Bench, domain-specific probes |
| RL (DPO + RLVR) | Edge-of-competence exploration, verifiable rewards | Dolci-DPO + Dolci-RLVR | 256 H100s (shared) | AIME / Codeforces / RULER / IFEval / reward-hacking audit |

이를 **네 번의 distribution shift**로 생각하라. web → curated-hard → long-coherent → instructive-format → edge-of-competence prompts. 각 shift는 충분히 좁아서 한 stage가 다른 stage의 일을 흡수할 수 없고, 각각 독립적으로 fail할 수 있는 eval을 가진다.

---

## 2. Mid-training의 operational definition

**Mid-training**은 continued-pretraining-style training pass로, (i) pretraining checkpoint에서 시작하고, (ii) high-quality narrow mix(pretraining mix도 아니고 instruction-formatted SFT data도 아님)에서 실행하며, (iii) budget으로 **pretraining token의 1-3%**를 사용한다. 이것은 SFT가 아니다. Loss는 response token만이 아니라 모든 token에 적용되고, data는 instruction-response format이 아니다. 이것은 pretraining도 아니다. Mix는 downstream stage가 가르칠 수 없는 것을 향해 의도적으로 skew되어 있다.

경험적으로 OLMo 3의 Dolmino는 **math problems + solutions, science papers, code, instruction-following seed data, reading-comprehension passages**를 다루며, 5.9T pretrain의 **100B tokens ≈ 1.7%**이다. OLMo 2의 Dolmino cooldown은 3.9T의 **50B tokens ≈ 1.3%**였다. Llama 3의 herd paper에서 "annealing"은 문서화가 덜 되어 있지만 pipeline의 같은 지점과 유사한 proportional budget에 있다. Lab 전반에서 패턴은 안정적이다. Post-training이 시작되기 전에 *curated hard* mix에 대해 *small* run을 수행한다.

**왜 obvious alternative인 "hard data를 그냥 pretrain에 넣으면 되지 않나"보다 나은가.** [[front-loading-reasoning]]은 실제로 front-loading을 주장하지만, 더 날카로운 구분을 붙인다. Pretrain 중에는 **diversity**를 우선시하고, mid-training + SFT 중에는 **quality**를 우선시해야 한다. Dolmino mix를 5.9T pretraining run에 던져 넣으면 signal을 잃는다. (a) 모델이 hard example을 generalize하게 하는 broad prior를 아직 만들지 못했고, (b) 나머지 pretrain mix가 100B high-quality token의 signal을 average-out하기 때문이다. Base가 안정화된 *뒤에* 따로 돌리면 signal을 집중시킬 수 있다. [[interplay-pretraining-midtraining-rl]]은 이것이 이후 RL이 exploit할 수 있는 "reusable structure"를 설치한다고 보고한다.

**왜 "SFT를 그냥 더 크게 만들자"보다 나은가.** SFT data는 instruction-formatted이다. 실제 scientific paper 100M-token corpus를 QA format으로 다시 쓰지 않고서는 SFT로 변환할 수 없으며, 그 rewrite는 long-form data가 가르치려는 cross-paragraph coherence를 정확히 파괴한다. Mid-training은 paper를 있는 그대로 next-token loss로 학습하고, instruction-response frame이 담을 수 없는 long-range structure를 모델이 배우게 한다.

**잘못된 budget을 골랐다는 concrete signal 세 가지.**

- *Budget too small* (<0.5% of pretrain): base의 broad prior가 SFT 몇 epoch 안에 mid-training distribution을 덮어쓴다. Mid-training checkpoint에서는 Dolmino-like gain을 보지만, SFT/DPO를 지나며 decay하는 것을 보게 된다.
- *Budget too large* (>5% of pretrain): narrow mix에서 pretraining을 다시 수행하고 있는 것이다. 이는 Phi-4가 비판받는 narrow-domain overfit과 정확히 같다. MMLU는 유지되지만 out-of-mix task의 breadth가 미끄러지기 시작한다.
- *Mix too instruction-heavy*: 모델이 SFT가 자체 format을 설정하기 전에 instruction artifact(assistant-style opening, bulleted formatting)를 습득한다. Downstream DPO는 implicit format을 *unlearn*해야 하며, 이는 preference를 낭비한다.

실용적인 diagnostic은 pretrain endpoint와 mid-training endpoint **둘 다**에서 targeted eval suite를 실행하는 것이다. 이득은 hard targeted task(GSM8K, MATH, HumanEval, MMLU-Pro)에 집중되어야 하며, broad breadth(TriviaQA, HellaSwag)에서는 *near-zero change*여야 한다. Breadth가 움직인다면 mix가 drift 중이다.

---

## 3. Cold-start SFT: R1-Zero vs R1

DeepSeek-R1에서 가장 많이 인용되는 결과는 R1-Zero이다. DeepSeek-V3-Base에서 rule-based reward만으로, SFT trace 없이 pure RL을 수행한 것이다([[deepseek-r1]]). 덜 인용되는 결과는 **왜 R1이 애초에 존재하는가**이다. R1-Zero의 rollout은 두 가지 방식으로 체계적으로 degrade되었다. Response가 English, Chinese, math symbol이 섞인 interleaved mix가 되었고, `<think>` block은 trace가 distillation에 쓸모없을 만큼 읽기 어려운 dense notation으로 drift했다. 이 failure mode는 model report에 입증되어 있다.

R1은 **RL 이전의 cold-start SFT pass**로 둘 다 고친다.

- **~800K cold-start examples**, 대략 **~600K reasoning + ~200K non-reasoning**으로 분할.
- **Reasoning trace의 origin**: R1-Zero 자체(및 이전 internal RL checkpoint)에서 rejection-sampled하고, correctness에 대해 DeepSeek-V3 judge로 filtering한 다음, language mixing을 제거하고 `<think>...</think><answer>...</answer>` template을 깔끔하게 강제하기 위해 **human-readability reformat**을 거친다.
- **Filter criteria**: final answer가 correct여야 하고, trace가 monolingual이어야 하며(R1-Zero의 language-mixing에 대한 operative fix), trace length ≤ 32K tokens, format reward를 gaming하는 hollow `<think>` block이 없어야 한다.
- **Training config**: V3 base에서 response token에 cross-entropy를 적용하는 standard SFT. 정확한 SFT hyperparameter는 report에서 이어지는 RL run과 별도로 분리되어 있지 않다.

그다음 R1은 전체 four-stage post-training pipeline을 실행한다. **cold-start SFT → reasoning RL (GRPO, LR 3e-6, KL 0.001, eps 10, T 1.0, G 16, 32K generation, 512 samples/step) → rejection-sampling SFT → alignment RL**. Cold-start SFT는 *readability* intervention이다. Reasoning RL은 *capability* intervention이다. Rejection-sampling SFT는 RL discovery를 supervised data로 다시 접어 넣는 *distillation* intervention이다. Alignment RL은 *general-helpfulness* intervention이다.

학습자가 가져가야 할 비교는 이것이다. **R1-Zero는 강한 base 위에서 RL만으로 long-CoT capability를 설치할 수 있음을 보여준다. R1은 여전히 ship하려면 cold-start가 필요하다는 것을 보여준다.** Cold-start는 capability bootloader가 아니다. 결과 모델을 deployable하게 만드는 **format installer**이다. Qwen 3의 four-stage pipeline(long-CoT cold-start → reasoning RL → thinking-mode fusion → general RL)도 같은 논리를 쓴다([[qwen-3-5]]). Qwen 3.5는 algorithmic change 없이 이를 계승한다.

**Cold-start sizing rule of thumb.** 입증된 지점은 DeepSeek의 800K이다. 더 작은 cold-start(Sky-T1, s1, LIMO)는 모두 **1K-30K traces**에 있다. 더 큰 pipeline(OLMo 3-Think의 Dolci-SFT, Llama-3-style iterated SFT)은 수백만 단위로 간다. Sizing lever는 **base가 target behaviour에서 얼마나 떨어져 있는가**이다. R1은 R1-Zero의 rollout이 readable CoT template에서 멀리 drift했기 때문에 ~800K를 사용했다. RL이 clean distribution 위에서 계속 진행되기 전에 cold-start가 format을 다시 가르쳐야 했다. Base가 이미 target format을 말한다면(예: instruct-tuned checkpoint를 thinking mode로 확장) cold-start는 LIMO-style 817-example regime으로 줄어들 수 있다. Format을 처음부터 만든다면 RL trigger를 당기기 전에 10²-10⁴ hand-vetted traces와 최대 10⁶ rejection-sampled + judge-filtered traces를 budget에 넣어라.

---

## 4. Long-context mid-training: position encoding + data curation

Long-context는 OLMo-3 terminology보다 앞선 paper에서 가장 자주 "mid-training"으로 묘사되는 stage이다. 이 stage에는 분리 가능한 두 가지 일이 있으며, 이를 뭉뚱그리면 대부분의 실망스러운 128K-window release가 생긴다.

**Job 1: position-encoding extension.** RoPE는 pretraining base frequency에서 training range 밖의 position을 aliasing 없이 표현할 수 없다. [[long-context-llama3]]는 여섯 stage schedule(8K → 16K → 32K → 64K → 128K, stage당 ~100-200B tokens, 총 ~800B)로 RoPE base를 **10,000 → 500,000**으로 rescale한다. [[prolong]]은 Llama-3-8B에서 512K를 위해 **10K → 128M NTK-aware**로 더 밀어붙인다. Scheduling rule은 empirical하다. **각 stage는 context를 두 배로 늘리고 base를 비례적으로 조정**하며, 다음 expansion 전에 새 position이 stabilize될 만큼 충분히 실행한다.

**Job 2: long-coherent-document training.** Position encoding만으로는 long range에 attend할 *capacity*를 준다. 하지만 그 capacity를 *사용하는 법*을 가르치지는 않는다. [[prolong]]의 thesis는 **concatenated short documents**(naive shortcut, 즉 random short piece를 context window까지 pack하는 것)로 training하는 것이 **genuinely coherent long documents**로 training하는 것보다 측정 가능하게 나쁘다는 것이다. Ablation: matched token count에서 curated coherent docs를 concatenated short docs로 바꾸면 **HELMET에서 10+ points**를 잃는다. ProLong의 30B-token mix는 **~40% code repositories(full repo, README → source → tests concatenated), 25% books, 15% academic with references, 10% long forum threads, 10% misc long web**이다. Web은 upweighted가 아니라 *downweighted*(×0.5)된다. Long web document 대부분은 long-range structure가 약한 scraped listing이기 때문이다.

**Job 3 (SFT-side): long instruction alignment.** Base가 long context를 지원하게 되면 SFT에는 long-context sub-mix가 필요하지만 *작게* 유지해야 한다. [[longalign]]은 아홉 document source에서 10k long-instruction example을 합성하며 **pick-one-of-5 cross-span trick**을 사용한다. Teacher가 전체 document를 포괄하는 candidate question 5개를 만들고, 그중 하나를 random으로 골라 answer synthesis를 한다. 이렇게 하면 local retrieval 대신 cross-span coverage가 강제된다. [[long-context-llama3]]는 long-context SFT를 **total SFT sample의 ~0.1%**로 유지한다. 이를 1% 이상으로 올리면 short-context capability에서 ~1 MMLU point를 잃는다. Binding constraint는 long-context gain이 아니라 short-context regression이다.

Three-job decomposition은 stage table에 깔끔하게 들어간다. Job 1 + Job 2는 long-context mid-training stage에 속한다(position rescale은 stage boundary에서 수행). Job 3은 SFT에 속한다. 이를 섞으면 window는 있지만 그 위에서 reasoning하지 못하는 모델(Job 2 skipped)이나 short input에서 degrade되는 chat model(Job 3 fraction too high)이 나온다.

**Llama 3의 six-stage schedule, explicit numbers.** [[long-context-llama3]]는 다음 per-stage profile을 문서화한다(§3.4를 paraphrase).

| Stage | Context | Tokens | RoPE base | Short:long mix |
|---|---|---|---|---|
| A | 8K → 16K | ~100B | partial rescale | 80:20 |
| B | 16K → 32K | ~100B | partial rescale | 70:30 |
| C | 32K → 64K | ~150B | partial rescale | 60:40 |
| D | 64K → 128K | ~200B | 500K (final) | 40:60 |

추가 intermediate stabilization stage를 포함해 총 ~800B tokens가 된다. Schedule design rule은 empirical하다. 각 sub-stage는 다음 doubling 전에 RoPE rescale이 stabilize될 만큼 충분히 실행되어야 하고, short:long ratio는 earlier-stage short-context behaviour가 anchored된 상태를 유지하도록 점진적으로 이동한다. Llama 3의 effective RULER context는 claimed window가 128K임에도 **405B에서 ~96K, 70B에서 ~64K**이다. 이 gap은 *expected*이며, staged schedule은 이를 제거하기보다 bounded되게 유지한다.

**ProLong의 20B budget은 universal하지 않다.** [[prolong]]은 **20B CPT tokens + 5B SFT tokens**만으로 Llama-3-8B를 8K에서 512K로 확장한다. 이 budget에서 작동하는 이유는 두 가지다. Base가 이미 Llama-3(500K RoPE base)이므로 extension이 비교적 modest하고, 30B-token data mix가 coherence(code repo, book, academic with references)에 강하게 filtered되어 있기 때문이다. Llama-3를 더 약한 base로 바꾸거나, coherence filter를 concatenated short docs로 바꾸면 이 budget은 깨진다. 교훈은 **data quality가 token budget을 산다**는 것이다. 하지만 base quality는 quality filtering에 필요한 data budget을 산다.

---

## 5. OLMo 3의 model-flow worldview — 전체 pipeline을 말로 풀기

[[olmo-3]]은 2025년 reference release이다. Final weights만이 아니라 **entire trajectory**를 scientific artifact로 취급하기 때문이다. 모든 intermediate checkpoint, 모든 per-stage dataset, 모든 eval suite가 public이다. 7B Base → Think flow를 따라가 보자.

**Base flow.** **Dolma 3 Mix (5.9T tokens)**로 pretrain한다. 구성은 web, olmOCR을 통한 science PDFs, code, math problems/solutions, encyclopedic text이며 1024 H100s를 쓴다. → 2.2T high-quality pool에서 sampling된, math, science, code, instruction following, reading comprehension을 강조하는 **Dolmino (100B tokens)**로 mid-train한다. 128 H100s. → 639B long-doc pool에서 나온 **Longmino (50B tokens)**로 long-context extend한다. Shared 128 H100s. 각 stage의 eval gate는 general LM eval suite + staged targeted probes(mid-train에서는 GSM8K + MATH, long-context에서는 RULER + NIAH)이다. Output은 128K context를 가진 OLMo 3-Base 7B이다.

**Think flow.** Base에서 **Dolci-SFT**(rejection-sampled long-CoT traces + instruction-following seeds, "thinking-specific")를 수행하고, 이어 **Dolci-DPO**(thinking-specific preferences), 그리고 verifiable math/code/IF reward에 대한 **RLVR**을 수행한다. 각 stage는 **256 H100s**를 사용한다(branch 간 shared). Post-training은 세 branch가 갈라지는 곳이다. Think는 thinking-specific SFT/DPO를 사용한다. Instruct는 chat-focused SFT/DPO를 사용한다. **RL-Zero는 SFT 없이** Base에서 직접 RL을 수행한다(R1-Zero의 OLMo-3 analog). 이는 open base에서 pure-RL을 연구하기 위한 explicit research artifact이다.

**Data curriculum summary, as a single diagram in words.** Dolma 3 (9.3T raw source tokens) → Dolma 3 Mix (5.9T filtered pretrain) → Dolmino (100B mid-train, sampled from 2.2T curated) → Longmino (50B long-context, sampled from 639B long-doc pool) → Dolci (three sub-mixes for SFT / DPO / RLVR, undisclosed totals). 각 data artifact는 explicit filter rule에 따라 preceding pool의 *subset*이다. Dolma 3 Mix는 Dolma 3를 decontaminate하고 re-mix한다. Dolmino는 2.2T high-quality pool 안에서 math/code/science를 up-sample한다. Longmino는 extension을 training하기에 충분히 긴 document만 유지한다. Dolci는 Dolmino-adjacent seed 위에 instruction-formatted된 것이다.

**Engineering point**는 OLMo 3의 openness가 dataset뿐 아니라 transition rule까지 포함한다는 것이다. Longmino가 Dolmino의 seed pool과 additional long-doc source에서 나온다는 점을 볼 수 있다. 그래서 mid-training과 long-context extension이 128-H100 budget을 공유할 수 있다. Later stage가 earlier stage의 부분 specialization이지 fresh start가 아니기 때문이다. Closed model도 거의 확실히 같은 일을 한다. OLMo 3은 단지 이를 verify할 수 있는 곳이다.

**Per-stage eval gates.** OLMo 3의 또 다른 disclosure는 *stage가 끝났다고 어떻게 결정하는가*이다. 각 stage는 고유 gate를 가진다.

- **Pretrain → Mid-train.** Broad LM suite로 gate한다: OLMES broad, MMLU, HellaSwag, code HumanEval. Run이 smooth하게 converging하고(loss spike 없음) MMLU가 사용된 token budget에 대한 recipe-anchored expectation 근처일 때 멈춘다.
- **Mid-train → Long-context.** Targeted hard task로 gate한다: GSM8K, MATH, MMLU-Pro, HumanEval+, IFEval. 이 gain은 baseline pretrain endpoint에 비해 sharp해야 하며, broad-eval은 flat해야 한다.
- **Long-context → SFT.** Long-context probe로 gate한다: NIAH heatmap @ 128K, RULER effective context, BABILong reasoning-in-a-haystack. 중요하게도, MMLU와 HumanEval에서 regression > 1 point가 없는지 확인하기 위해 *short-context eval을 다시 실행*한다. Short-context가 regressed했다면 long-context data mix(Job 2)가 너무 dominant한 것이다.
- **SFT → DPO.** Format + chat quality로 gate한다: MT-Bench, AlpacaEval-2, IFEval strict, format-compliance probe.
- **DPO → RLVR.** Preference alignment + verifiable task head-room으로 gate한다: AIME pre-RL baseline, MATH, Codeforces Elo baseline, reward-model-scored held-out.

각 gate에는 benchmark number뿐 아니라 training log의 **failure signature**가 대응된다. Stable MMLU와 함께 mid-training loss spike가 있으면 보통 mix가 pathological subdomain(예: Unicode artifact가 많은 math source)을 upweight했다는 뜻이다. Stable NIAH와 함께 long-context에서 RULER collapse가 있으면 Job 1은 성공했지만 Job 2가 실패했다는 뜻이다. Window는 있지만 reasoning이 없다. DPO를 지나며 MT-Bench가 떨어지면 대개 on-policy preference가 다른 SFT checkpoint에서 stale하게 재사용되었다는 뜻이다([[llama-3]]가 이를 명시적으로 지적한다).

**Compute allocation as a check on your pipeline.** OLMo 3가 공개한 split인 **pretrain 1024 H100s, mid-train + long-context 128, SFT/DPO/RLVR 256**은 유용한 sanity ratio이다. Pretrain이 dominant cost이다. Mid-training은 rough accounting에서는 놓칠 만큼 작다. Post-training은 middle-sized specialty stage이며, 여기에서는 efficiency gain(OLMo 3는 Open Instruct에서 Olmo Core로 옮기며 **8× SFT throughput**, in-flight weight update와 continuous batching으로 **4× RL efficiency**를 보고)이 실제로 중요하다. Post-training compute가 pretraining compute에 접근한다면, Llama 3 style의 multi-round iterative SFT를 하고 있거나 stage allocation이 잘못된 것이다.

---

## 6. Open question — 더 나은 data가 mid-training을 건너뛸 수 있는가?

2024-2025 debate에서 반복되는 주장: *pretraining data가 충분히 좋으면 mid-training과 cold-start는 불필요해진다.* 이 입장에 대한 가장 강한 evidence는 indirect하다. Phi-4의 synthetic-heavy pretrain은 pretrain/mid-train boundary를 흐리고, R1-Zero는 강한 base 위 pure RL이 SFT 없이도 합리적인 reasoning number에 도달할 수 있음을 보여준다. 반대편의 가장 강한 evidence는 direct하고 quantitative하다.

- **[[interplay-pretraining-midtraining-rl]]** — *fixed compute* 아래에서 mid-training은 extrapolative와 contextual generalization 모두에서 RL-only post-training을 이긴다. Effect는 edge-of-competence task에서 가장 강하다. Better pretraining alone이 mid-training의 일을 흡수할 수 있다면 fixed-compute comparison은 RL-only를 선호해야 한다. 그렇지 않다.
- **[[front-loading-reasoning]]** — reasoning data를 pretraining에 front-load하면 보고된 **19% average gain**을 주고 이후 SFT가 도달할 수 있는 *ceiling을 높인다*. 하지만 논문은 late SFT가 early injection의 durable advantage를 **완전히 재구성할 수 없다**고 명시한다. 즉, better pretraining은 mid-training을 덜 가치 있게 만드는 것이 아니라 *더* 가치 있게 만든다. Mid-training + RL이 exploit할 수 있는 headroom을 높이기 때문이다.
- **R1 vs R1-Zero deployment evidence** — pure-RL은 강한 AIME / MATH number에 도달하지만, model은 language-mixed trace를 ship한다. Cold-start는 capability shortcut이 아니다. Pretrain-only configuration이 명백히 배우지 못하는 format fix이다.

정직한 답은 다음과 같다. *Better data는 각 stage contribution이 peak하는 지점을 이동시키지만 어떤 stage도 제거하지 않는다.* 더 작고 더 high-quality인 pretrain + 더 작은 mid-training + 더 작은 RL run은 가능하다. 그리고 이는 Phi-4가 실험하는 것이라고 볼 수 있다. 하지만 그것은 rebalancing이지 merger가 아니다. Stage는 **job이 다르기 때문에** 개념적으로 구분된 채로 남는다. Broad priors, reusable structure, position encoding, format cold-start, edge exploration은 서로 다른 일이다. 이를 합치면 synthetic pretrain에 overfit하는 모델(Phi-4의 narrow-domain criticism)이 나오거나, reasoning은 할 수 있지만 읽을 수 없는 모델(R1-Zero)이 나온다.

**볼 만한 specific counter-evidence pattern.** Lab이 "mid-training을 건너뛰었는데도 작동한다"고 주장하면 세 가지를 확인하라. (i) 실제로 건너뛰었는가, 아니면 다른 이름의 pretrain cooldown에 접어 넣었는가? OLMo 2의 Stage 2는 OLMo 3가 "mid-training"을 공식화하기 전까지 "cooldown"이라고 불렸다. 내용은 같다. (ii) Benchmark mix가 무엇인가? MMLU + HellaSwag에 근거한 주장은 mid-training의 effect를 완전히 숨길 수 있다. 이 benchmark들은 pretraining이 이미 가진 broad prior를 반영하기 때문이다. Discriminating eval은 MATH, AIME, MMLU-Pro, HumanEval+, IFEval, RULER이다. Task가 harder하거나 targeted할수록 mid-training contribution이 더 크게 보인다. (iii) Pretraining mix가 어떤가? [[front-loading-reasoning]]처럼 reasoning data를 pretraining에 front-load한 lab은 mid-training의 apparent delta를 줄일 수 있다. Mid-training의 일부 일이 earlier로 이동했기 때문이다. 이는 rebalancing이지 elimination이 아니다.

**Stage-dependency summary.** Interaction을 기억하는 가장 명확한 방법:

- Mid-training은 pretraining quality와 *multiplicative*이다. [[front-loading-reasoning]]이 effect를 보여준다. Better pretrain priors → mid-training gains는 dampen되지 않고 amplify된다.
- RL은 mid-training quality와 *multiplicative*이다. [[interplay-pretraining-midtraining-rl]]은 이를 직접 보여준다. Edge of competence의 RL은 pretrain + mid-train이 edge를 흥미로운 곳에 놓아두어야 한다.
- Cold-start SFT는 model의 existing format behaviour와 *additive*이다. 다시 실행하기 가장 싼 stage이며 target deployment surface에 가장 민감하다.
- Long-context extension은 reasoning pipeline과 *orthogonal*이다. Mid-training 전후 어느 쪽에 넣어도 reasoning stage는 크게 신경 쓰지 않는다. 하지만 SFT 전에는 반드시 일어나야 한다. 그렇지 않으면 long-SFT data가 실제로 길 수 없다.

**[figures/pipeline-stages.html](figures/pipeline-stages.html)** 에서 stage budget을 조작해 보라. 다섯 stage(pretrain → mid-train → long-context → SFT → RL)를 가로 Gantt로 보여주며, Llama-3, OLMo-3, DeepSeek-R1, Phi-4-reasoning configuration 사이를 toggle할 수 있고, stage별 token-count와 data-mix composition을 볼 수 있다.

---

## 7. Compute budget별 recipe

모든 lab에 1024 H100이 있는 것은 아니다. Budget이 줄어들면 stage structure는 예측 가능하게 collapse된다.

- **Frontier budget (10²-10³ H100 nodes).** Full five-stage pipeline. Mid-training과 long-context가 각자의 data mix와 eval gate를 갖는다. OLMo 3, Llama 3, DeepSeek-R1, Qwen 3.5가 여기에 해당한다.
- **Mid-tier budget (10¹-10² H100s, starting from an open base).** Pretrain은 완전히 건너뛴다. Llama-3 / Qwen-3 / OLMo-3에서 시작하라. Target이 specialized reasoning / coding / long-doc model이라면 **mid-training을 유지**하라. Target이 chat assistant이고 base가 이미 instruction-tuned라면 **mid-training을 건너뛰라**. 새로운 format(thinking mode, new tool schema)으로 밀고 들어간다면 cold-start SFT를 유지하라. Base가 target window를 이미 지원하지 않는다면 long-context extension을 유지하라. Verifiable reward가 없다면 이 budget에서 RL은 optional이다.
- **Small-budget (≤8 H100s, single-node).** 두 stage: SFT + optional DPO. 필요한 mid-training 및 long-context 속성을 이미 가진 base를 사용하라. Lever는 stage count가 아니라 data quality이다. [[front-loading-reasoning]] 기준 여기서는 *quality*를 우선시하라. Diversity는 선택한 base 안에 들어 있다.

Stage count decision은 **data quality가 base가 이미 가진 것에 비해 어디에 위치하는가**에 관한 것이다. Data가 base pretrain mix보다 quality가 높지만 mid-training target을 움직이지 않는다면 SFT에 속한다. Data가 needle을 움직이고 instruction-formatted가 아니라면 mid-training pass에 속한다. 그 pass가 여덟 GPU에서 몇 억 token에 불과하더라도 그렇다.

---

## 8. 세 lab, 세 stage allocation

가장 잘 문서화된 2025 pipeline 세 개를 나란히 놓으면 rebalancing이 보인다.

| Stage | OLMo 3 7B ([[olmo-3]]) | Llama 3 405B ([[llama-3]], [[long-context-llama3]]) | DeepSeek-R1 ([[deepseek-r1]]) |
|---|---|---|---|
| Pretrain | Dolma 3 Mix **5.9T** | **15.6T**, 8K native | V3 inherited: **14.8T**, 2.788M H800-hrs |
| Mid-training | Dolmino **100B** (1.7%) | "Annealing" stage, scale undisclosed | Inherited from V3; no separately named stage |
| Long-context | Longmino **50B** (0.8%) | Staged 8K→128K, **~800B** tokens (5.1%), RoPE 500K | Inherited from V3 |
| Cold-start SFT | Dolci-SFT (rejection-sampled CoT + IF) | 6-round SFT + Rejection Sampling | **~800K traces** (reasoning + non-reasoning) |
| DPO / RL | Dolci-DPO → RLVR | 6-round DPO with NLL stabilization | Reasoning-RL (GRPO, 32K gen) → rejection SFT → alignment-RL |

세 가지 관찰.

1. **OLMo 3는 pretrain에 상대적으로 덜 쓰고(5.9T), post-pretrain stage를 명시적으로 이름 붙이는 데 더 투자한다.** Mid-training + long-context stage를 합쳐도 pretrain token의 ~2.5%에 불과하지만, 각각은 고유 eval gate를 가진 distinct run이다. 이것이 "stage-as-artifact"의 가장 명확한 예이다.
2. **Llama 3는 long-context에 비율상 훨씬 더 많이 쓴다(~pretrain의 5%).** 이는 Meta의 design target을 반영한다. Dense 405B 위에 128K-window frontier model을 ship하는 것이 목표였고, 그래서 Llama 3의 effective RULER context(405B에서 ~96K)가 dense open model 중 최고권이다. Long-context는 Llama 3에서 specialized mid-training stage가 아니라 major pretrain-adjacent effort였다.
3. **DeepSeek-R1은 pretrain과 mid-training을 V3에서 상속하고, distinct한 거의 모든 투자를 RL loop에 넣는다.** R1의 innovation은 post-V3이다. Cold-start SFT, GRPO hyperparameter, multi-stage SFT-RL-SFT-RL pipeline이 그것이다. V3 base가 mid-training과 long-context stage를 들고 있으므로 R1이 ~800K SFT sample에서 *작동할 수 있다*. 더 약한 base에서 R1을 replicate할 수 없다. R1이 건너뛰는 stage는 이미 V3에서 비용을 지불했기 때문이다.

이 table의 올바른 읽기는 "누가 더 많이 썼는가"가 아니라 "각 lab은 어떤 stage를 locally optimize했는가"이다. OLMo 3는 stage-structure 자체(flow의 openness)를 optimize했다. Llama 3는 long-context + iterative post-training을 optimize했다. DeepSeek-R1은 strong-base에서의 RL을 optimize했다. 작은 lab은 전체 pipeline을 복사하는 것이 아니라 자기 edge와 맞는 stage를 복사한다. 전체 pipeline은 frontier budget이 필요하다.

---

---

## 9. 아직 field가 합의하지 못한 것

세 가지 open question이 남아 있으며, 각각 2026년까지 추적할 가치가 있다.

- **Mid-training은 더 공격적인 synthetic pretrain과 compose되는가?** Phi-4-reasoning은 "mid-train inside pretrain" 노선을 밀고, OLMo 3는 분리된 상태로 유지한다. Matched compute에서 두 철학을 controlled comparison한 결과는 아직 public하지 않다. [[interplay-pretraining-midtraining-rl]]이 가장 가까운 evidence이며 controlled-but-small-scale이다.
- **Cold-start가 RL의 ceiling에 얼마나 영향을 주는가?** R1은 RL 전에 800K cold-start trace를 사용했다. s1과 LIMO는 10³-sample cold-start로도 RL 이후 강한 reasoning을 만들 수 있음을 보여준다. Open question은 RL gain의 *upper bound*가 cold-start size에 의존하는지, 즉 800K trace가 RL이 탐색하는 policy distribution을 영구적으로 shape하는지이다. Evidence는 mixed이며 양쪽 모두 강한 anecdote가 있다.
- **Long-context extension은 정말 mid-training과 separable한가, 아니면 OLMo-3 convention인가?** Llama 3와 Qwen 3는 long-context를 staged pretrain extension으로 취급한다. OLMo 3는 이를 Dolmino와 *overlap*하는 Longmino pool에서 공급되는 named post-pretrain stage로 취급한다. 이 overlap은 두 stage가 완전히 separable하지 않음을 시사한다. 이는 §4의 three-job decomposition과 일관되지만 깔끔한 stage-table 그림을 복잡하게 만든다.

여기가 CLAUDE.md의 "core insight per landmark paper" 원칙이 open research와 만나는 지점이다. Interplay paper는 그 claim을 위해 읽고, OLMo 3는 artifact를 위해 읽고, 둘 다 답하지 못하는 question은 별도 note로 남겨라.

## Connections and what's next

- **[[interplay-pretraining-midtraining-rl]]** — stage-by-stage allocation argument를 뒷받침하는 controlled-experiment evidence. §Main causal claims를 읽어라.
- **[[olmo-3]]** — 가장 명확한 public model-flow example. Stage budget과 eval gate의 primary source. Model-flow diagram과 expanded training-data section을 읽어라.
- **[[deepseek-r1]] + model-report** — cold-start SFT pattern. Readability fix를 보려면 R1 vs R1-Zero comparison을 읽어라.
- **[[long-context-llama3]] + [[prolong]] + [[longalign]]** — long-context three-job decomposition. 각 source가 하나의 job을 깔끔하게 다룬다.
- **[[front-loading-reasoning]]** — stage 간 diversity-vs-quality asymmetric-allocation rule.
- **ch-31 (previous)** — SFT와 RL 사이의 bridge로서 DPO / preference optimization. 이 장은 전체 pipeline으로 zoom out한다.
- **ch-33 (next)** — Tulu 3와 Llama 3를 이 stage stack 위에서 multi-round post-training을 *실행*하는 case study로 다룬다.
- **ch-34 (next)** — Qwen 2.5/3와 Phi 3/4를 통해 mid-training이 어디서 끝나고 SFT가 어디서 시작되는지에 lab들이 어떻게 의견을 달리하는지 보여준다.

## Further reading

- [[interplay-pretraining-midtraining-rl]] — controlled framework; edge-of-competence result가 가장 actionable하다.
- [[olmo-3]] — model-flow report; per-stage compute disclosure는 드물고 유용하다.
- [[deepseek-r1]] paper + model-report — 둘 다 읽어라. Paper는 R1-Zero의 claim을 frame하고, model-report는 cold-start를 상세히 다룬다.
- [[long-context-llama3]] — explicit stage budget과 RoPE rescale value가 있는 production 128K recipe.
- [[prolong]] — long-context data rule을 anchoring하는 document-coherence ablation.
- [[front-loading-reasoning]] — stage 간 diversity-vs-quality allocation rule.

## Companion visualization

**[figures/pipeline-stages.html](figures/pipeline-stages.html)** — 다섯 stage(pretrain → mid-train → long-context → SFT → RL)의 horizontal Gantt. 네 가지 disclosed configuration(Llama 3 / OLMo 3 / DeepSeek-R1 / Phi-4-reasoning) 중에서 선택할 수 있다. 각 stage는 token budget을 bar width로, data-mix composition을 stacked fill(web / math / code / long-doc / instruction / preference)로, eval gate를 tooltip으로 보여준다. Configuration을 toggle하면 lab 간 *rebalancing*이 드러난다. 예를 들어 Phi-4는 OLMo 3에 비해 pretrain을 압축하고 mid-training을 키우며, DeepSeek-R1의 cold-start SFT는 mid-training(V3에서 상속)과 비교해 매우 작고(800K samples ≈ 수천만 tokens), Llama 3는 outsized long-context stage(~800B tokens)를 가진다. 자신의 pipeline을 설계할 때 compute allocation의 mental model로 사용하라.
