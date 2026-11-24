# Chapter 32 — Mid-Training: Annealing Data, Stage Gate, 이후 SFT/RL에 미치는 효과

> **핵심 통찰.** Mid-training은 main pre-training 뒤의 짧은 training phase다. learning rate가 decay하는 동안 data mixture가 curated/domain-specific/synthetic/instruction-formatted data 쪽으로 이동하며, 결과 checkpoint는 post-training 전 evaluation gate로 검사한다. target skill 효과는 크다. OLMo 2 7B GSM8K는 mid-training 뒤 24.1 → 67.5로 올랐으며 이는 50B-token run 세 개 평균이다. 일부 general skill은 작거나 negative effect이며 targeted gain은 scale이 커질수록 작아진다(Llama 3: 8B GSM8k validation +24.0%, 405B에서는 “negligible”). controlled study는 mixed mid-training이 domain-only continued pretraining보다 forgetting을 낮춤을([[midtraining-bridges-distributions]], 70M–1B) 보고하고, mid-training data가 later RL scaling distance를 바꿈을([[octothinker]], math Llama 3.2) 보이나 frontier scale broad capability 효과는 아직 측정하지 않았다.
>
> **지침.** general-purpose model 목표에서는 mid-training mix에 high-quality general data를 큰 비중으로 유지하라(OLMo 2 Dolmino web 약50%, Olmo3 Dolmino web/PDF 32.5%). Olmo3 domain-skewed 100B mix에서 target cluster는 final FIM/math-code-thinking mix보다 최대 6.0 point 올랐지만 untargeted cluster는 최대 31.6 point 떨어졌다(FIM/Gen-QA mix, Olmo3 Table7). candidate dataset은 matched web-only baseline과 short anneal하여 target/untarget benchmark 모두의 변화를 읽어라. small-scale single-benchmark gain은 large scale에 유지되지 않을 수 있다(Llama3 §3.1.3). later RL이 있으면 base benchmark뿐 아니라 RL 뒤 mid-training choice를 평가하라. base score는 70B/100B-token OctoThinker checkpoint를 구분하지 못했으나 RL result는 구분했다(§3.5). RL이 없으면 base/post-SFT evaluation이 가능한 gate다(Olmo3 integration test).

## 이 챕터가 general-purpose model에 중요한 이유

modern open model pipeline은 “pre-training then fine-tuning”보다 많은 stage다: pre-training → mid-training(annealing/domain upsampling/many report의 long-context extension) → SFT → preference optimization → RL → evaluation. Mid-training은 LR이 떨어지고 distribution이 later stage가 쓸 data로 움직이는 지점이다.

1. **decay phase data는 base model을 측정 가능하게 바꾼다.** MiniCPM .036B WSD run에서 loss는 stable보다 decay에서 더 빨리 줄었고([[minicpm]] §4.2 Figure5), MiniCPM-2.4B decay에 high-quality/SFT data를 더하면 동일 이후 SFT에서 MMLU 44.6→50.9였다(§5 Table1). 여기 narrow mix는 model을 narrow하게 한다(§2.4).
2. **benchmark-adjacent data가 이 stage에서 base model에 들어간다.** OLMo2 mid-training math pool은 GSM8K train split을 포함하며([[olmo-midtraining-evidence]] OLMo2 Table5), Olmo3 decontamination은 candidate source에서 complete GSM8K leakage와 DROP training example 60,000개 이상을 찾았다(§3.5.4). decontamination 없으면 contamination은 base model에 이 stage에서 들어간다.
3. **SFT/RL이 할 수 있는 일을 바꾼다.** mid-training data는 SFT forgetting([[midtraining-bridges-distributions]]), base RL scaling([[octothinker]]), mid-training/RL compute split의 in-distribution vs hard OOD task effect([[interplay-pretraining-midtraining-rl]])를 바꾼다.

이 장은 LR schedule([[ch-03]])/domain mixing([[ch-13]])에 의존한다. new-domain continued pretraining은 [[ch-32a]], long-context mechanic은 [[ch-32b]]/[[ch-32c]], agentic mid-training data는 [[ch-32d]]다.

## §1 Mid-training이란: data phase, learning-rate phase, gate

### 1.1 Source의 정의

- **OLMo2**: “mid-training(5–10% training FLOPs)”은 run remaining length 동안 LR을 linear decay to zero하고 high-quality web/curated non-web/synthetic math를 up-sample하는 second base-model stage다([[olmo-2]]; arXiv:2501.00656v3 §2.3).
- **Liu et al.**: general pretraining보다 “more specialized” data이면서 “maintaining a mixture with general pretraining data”인 intermediate phase다. continued pretraining은 general-data weight 0의 limit다([[midtraining-bridges-distributions]] §2.1–2.2).
- **OctoThinker**: pre/post-training “intermediate” compute/token budget stage로 domain expansion, long-context extension, data quality, synthetic data, “preparing for post-training”에 쓴다([[octothinker]] §2).
- **mid-training survey**: data distribution, LR scheduling, long-context extension 세 part로 practice를 묶는다([[mid-training-survey]] Abstract). gradient noise scale/information bottleneck/curriculum explanation은 new experiment 없는 **Interpretation**이다.

이 장에서 **mid-training**은 pre-training checkpoint에서 continued next-token training을 (1) changed data mixture로, (2) 보통 LR decay 중에, (3) post-training으로 보낼 checkpoint를 결정하는 evaluation gate 뒤에 하는 것이다. **Annealing**은 LR이 low/zero로 내려가는 중 training이며 **stage gate**는 stage checkpoint accept evaluation/threshold set이다.

### 1.2 Budget은 공통 percentage를 따르지 않는다

team이 mid-training token 수를 정해야 하지만 source share는 .00026%–20%다. printed token count에서 derived한 share=stage token÷stage end cumulative token이다.

| Run | Stage | Stage token | Cumulative token | Share |
|---|---|---|---|---|
| Llama3.1 405B | final anneal |40M|15.6T|40M ÷ 15.6T = .00026%|
| Llama3.1 405B | long-context stage |~800B|15.6T|5.1%|
| OLMo2 7B | one mid-training run |50B|3.90T+50B=3.95T|1.27%|
| Olmo3 7B | Stage2 mid-training |100B|6.0T(Table13)|1.67%|
| SmolLM3 3B | Stage3 decay |1.1T|11.1T|9.9%|
| MPT-style 7B | domain upsampling |200B|1T|20%|

Source: [[llama-3]] §3.4.2–3.4.3; [[olmo-midtraining-evidence]](OLMo2 §2.3, Olmo3 Table13, SmolLM3 blog); [[domain-upsampling-end-of-training]] §3.2. almost five order(20%÷.00026%≈77,000)이며 stage name도 다르다. Llama3 40M anneal은 SmolLM3 1.1T decay와 비교되지 않는다. “mid-training=pre-training의1–3%” 일반 rule은 source가 지지하지 않는다. local evidence만 있다: MiniCPM .036B는 total token 10% decay면 sufficient, 2.5%는 “falls short”([[minicpm]] §4.2); Blakeney 7B/1T는 10–20%가 best trade-off([[domain-upsampling-end-of-training]] §3.3). **[figures/pipeline-stages.html](figures/pipeline-stages.html)** panel A는 documented run 6개 stage token/LR phase/data change/locus를 비교한다.

### 1.3 Learning-rate phase

**Warmup-Stable-Decay(WSD)**는 warmup, long constant LR, separate decay schedule다([[minicpm]] §4.2 Eq.1).

```
WSD(T; s) = (s / W)·η          if s < W
          = η                  if W < s < T
          = f(s − T)·η         if T < s < S,   with 0 < f(s − T) ≤ 1 decreasing
```

- s=current step, W=last warmup, T=last stable, S=final step.
- η=maximum LR, f=decay function(MiniCPM exponential; OLMo2/Olmo3 linear to zero).

WSD stable checkpoint는 pre-training restart 없이 different data mix의 multiple decay run으로 fork할 수 있다. MiniCPM은 stable checkpoint를 decay하거나 high LR로 더 학습한 뒤 later decay할 수 있음을 보고한다([[minicpm]] §4.2). OctoThinker는 200B constant-LR stage 뒤 20B decay branch에 이 form을 쓴다([[octothinker]] §4).

Olmo3 7B released mid-training script(OLMo-core@66f768b, `src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py`; full quote [[olmo-midtraining-evidence]])의 config다.

```python
40  DEFAULT_SEQUENCE_LENGTH = 8192
41  GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens
42  MAX_TOKENS = 100_000_000_000  # 100B
43  LR = 0.00020712352850360292
80          scheduler=LinearWithWarmup(warmup=0, alpha_f=0.0),
103             load_optim_state=True,
```

same commit scheduler(`src/olmo_core/optim/scheduler.py` L384,L399)는 `η(s) = η₀ − (η₀ − η₀·α_f) · s / S`다. η₀=configured LR(2.0712×10⁻⁴), α_f=η₀ 대비 final LR fraction(여기 .0이므로 LR zero), s=warmup 뒤 current step, S=warmup 뒤 total step이다.

```
η(s) = η₀ − (η₀ − η₀·α_f) · s / S
```

- `load_optim_state=True`는 optimizer state를 새로 시작하는 setting이 아니라 pre-training에서 이어 받는 setting이다.
- α_f=0.0은 final learning rate가 configured η₀의 0배, 즉 zero에 도달함을 뜻한다.

## §2 Learning-rate decay 중 high-quality data annealing

### 2.1 Decay stage의 SFT-style data(MiniCPM)

정의는 decay stage pre-training data에 instruction/SFT-formatted data를 섞고 별도 SFT를 실행하는 것이다. 문제는 high-quality labeled data가 SFT에서만 쓸 때보다 decay에 넣을 때 더 도움되는가다.

근거([[minicpm]] §5 Table1; 2.4B/1.2B, seed 미보고, **결과 단일 연구**):

| Run | Decay data | SFT token | C-Eval | MMLU | GSM8K | MATH | HumanEval |
|---|---|---|---|---|---|---|---|
| A-1 2.4B | pre-training only |4B|40.0|44.6|27.7|5.1|27.7|
| A-2 2.4B | + high-quality/SFT data |4B|52.6|50.9|42.3|5.4|30.4|
| B-2 1.2B | pre-training only |12B|41.2|47.9|34.4|7.3|43.9|
| B-3 1.2B | + high-quality/SFT data |6B|49.1|49.6|31.8|10.5|44.5|

A-2−A-1 MMLU=50.9−44.6=+6.3이다. B-3은 B-2 SFT token 절반인데 C-Eval +7.9/MATH +3.2지만 GSM8K −2.6이다. 저자는 specialization이 “should start from decay phase”라고 결론낸다(§5). 표 benchmark 7개는 knowledge/math/code뿐이며 general chat quality/forgetting은 측정하지 않았다. Released MiniCPM recipe는 pre-training+UltraChat/SlimOrca/OssInstruct/EvolInstruct/proprietary high-quality SFT data로 decay한 뒤 약6B token separate SFT를 하며, 이는 “still necessary”였다(§6.2–6.3). 이후 Olmo3 Tulu3 SFT chat special token microanneal은 base가 token을 emit하게 해 GSM8K 49.43→0/CruxEval32.89→18.91이었고 ordinary-text template은 46.02/29.65였다([[olmo-midtraining-evidence]] Olmo3 §3.5.4). 그래서 Olmo3는 mid-training instruction data의 special token/template을 제거했다.

### 2.2 Training 끝 domain upsampling

**domain upsampling(DU)**은 ending pre-training에서 large less-filtered web을 줄이거나 빼고 smaller domain-specific dataset/code의 batch share를 높인다([[domain-upsampling-end-of-training]] §3.2). 7B MPT architecture/1T token/setting당 reported run 하나에서 (1) baseline 0.8T: large CC34.35%, small CC36.70%, domain7.17%, code21.78%(Table2), (2) last .2T: large CC0%, small CC30%, domain35%, code35%(Table4), (3) baseline-mix 유지 run과 비교했다. small CC는 “large distribution shift” 방지용으로 남겼다(§3.2). 20% DU result(Table5)는 MMLU35.69→42.59(+6.90), GSM8K14.71→22.97(+8.26), HumanEval17.23→23.40(+6.17), CoreAvg35.37→39.32, Language Understanding61.52→60.08(−1.44)다.

| DU training share |0%|5%|10%|20%|30%|
|---|---|---|---|---|---|
| MMLU |35.69|40.20|**43.19**|42.59|41.78|
| GSM8K |14.71|16.98|20.47|22.97|**24.56**|
| Core Average |35.37|37.63|38.46|**39.32**|38.89|
| Language Understanding |**61.52**|61.05|60.41|60.08|60.35|

20%→30% DU는 GSM8K +1.59, MMLU −.81, CoreAvg −.43이다. math/code gain은 이어지고 broad average는 내려간다. 저자는 이 setup에 10–20% best trade-off이고 DU mix는 training 전체에 쓰면 안 된다고 결론낸다(Figure3 caption, **결과 단일 연구**). **[figures/pipeline-stages.html](figures/pipeline-stages.html)** panel B는 five duration/benchmark 0% 대비 변화를 보여준다. 한 model size/setting당 run 하나/seed·variance 없음/DU-window behavior 미기재 inverse-square-root schedule/Gauntletv.3 broad measure 하나라는 한계가 있다. 다른 schedule/scale 10–20% 일반화는 **Open question**이다.

### 2.3 Llama3 annealing, checkpoint averaging, souping

Llama3.1 405B는 final40M token에서 128K context LR을 linearly zero로 decay, “very high quality” source upsample, anneal checkpoint Polyak average로 final pre-trained model을 만든다([[llama-3]] arXiv:2407.21783v3 §3.4.3; [[llama-3-recipe]] row). report는 mix를 print하지 않고 true few-shot/OOD generalization 평가를 위해 common benchmark training set을 anneal data에서 제외한다(§3.1.3). order는 long-context 뒤 annealing이다(§3.4.2–3.4.3).

**Souping**은 same mix/different data order run의 weight averaging으로 OLMo mid-training recipe 일부다. OLMo2 7B는 50B run3개, 13B/32B는 100B run3개+300B run1개 average한다(arXiv:2501.00656v3 §2.3). candidate 50B mix6개에서 soup는 OLMES/OLMES-Gen/MMLU best single checkpoint 이상이나 mix E GSM*은 60.5→43.0으로 떨어진다(Table14). Olmo3 32B는 100B run2개 merge해 두 run 대비 Math +2.9/+1.6이며 7B는 initial experiment에 similar gain이 없어 single run이다([[olmo-midtraining-evidence]] Olmo3 §3.5.4). general merging은 [[ch-30c]].

### 2.4 Mid-training mix 내용과 measured trade-off

OLMo2 Dolmino50B(Table13) mix: filtered DCLM web47.2%, decontaminated FLAN16.6%, math20.8%, Wikipedia/Wikibooks7.11%, peS2o paper5.85%, StackExchange2.45%. 7B pre-training→mid-training(Table9): MMLU59.8→63.7, DROP40.7→60.8, NQ29.0→36.9, GSM8K24.1→67.5, MMLU-Pro27.4→31.0, TriviaQA74.6→78.0. 1B WinoGrande는67.8→66.5인데 average31.9→43.7이다. Olmo3 Dolmino100B(Table5 category sum): web/science PDF32.5%, code20.0%, math19.2%, synthetic QA13.9%, thinking trace8.34%, instruction6.1%.

Olmo3 domain-skewed100B mix trade-off([[olmo-midtraining-evidence]] Olmo3 Table7, **결과 단일 연구**):

| Mix | MC STEM | MC Non-STEM | GenQA | Math | Code | FIM |
|---|---|---|---|---|---|---|
| Gen-QA(no math/code/thinking) |66.3|78.1|72.5|27.5|11.9|.1|
| Math-code-thinking(no QA/instruction) |62.5|69.6|65.9|**60.8**|**35.6**|**37.7**|
| Final Round5 |66.4|77.4|73.1|57.3|31.2|31.7|

Final 대비 math-code-thinking은 Math+3.5/Code+4.4, MC Non-STEM−7.8/GenQA−7.2다. Gen-QA는 MC Non-STEM+.7/Math−29.8이다. 한 group skew cost가 other gain보다 커 balanced mix를 골랐다(§3.5.4). fixed100B instruction/thinking trace 추가는 base-eval 모든 cluster를 올렸다(average48.8→50.7, Math43.1→48.7; Table10). general model에는 web/document 32.5–50%를 유지한 mixed corpus evidence가 있으며 added specialized source는 target하지 않는 cluster도 점검해야 한다.

## §3 Data probe로서 annealing

### 3.1 방법

**data probe anneal**(Llama3 “annealing to assess data quality”, OLMo “microanneal”)은 partly trained checkpoint LR을 candidate 포함 mix에서 decay하고 baseline anneal과 benchmark를 비교해 candidate dataset을 측정한다. Blakeney et al.은 small-scale model이 MMLU 같은 benchmark에 “random accuracy”를 register하므로 small-scale mix ablation이 noise를 측정할 수 있고 full-scale ablation은 pre-training만큼 cost한다고 지적한다([[domain-upsampling-end-of-training]] §1).

Source recipe: (1) **Llama3 8B**는 50%-trained 8B에서 new dataset weight30%/default70%로 40B token linear-zero anneal한다(arXiv:2407.21783v3 §3.1.3, result table 없음). (2) **OLMo2 microanneal**은 math source와 거의 같은 general web token을 섞고 LR linear-down “as if annealing run”한다. microanneal19개는 total130B token, “less than 3 full 50B annealing run”이었다(§4.4.2). (3) **Olmo3 microanneal**은 target5B+web5B, web-only10B baseline이며 math만 candidate source25개에 run80개다. promising source는 OlmoBaseEval 및 SFT 뒤 재평가하는 100B “integration test”로 간다(Olmo3 §3.5.1).

probe에서 읽는 quantity는 matched baseline difference다.

```
Δ_b(D) = score_b(anneal on D + web) − score_b(anneal on web only)
```

- b: benchmark/cluster 하나, D: candidate dataset. 양 anneal은 same checkpoint/token count/LR schedule에서 시작한다.

계산 예제(Olmo3 Table9, meta-reasoning/program-verifiable vs web-only 5B): Δ_GSM8K=26.8−18.4=+8.4, Δ_HumanEval=19.5−7.9=+11.6, Δ_MMLU=53.7−55.2=−1.5, Δ_GenQA=52.9−53.7=−.8. target Δ만 보면 negative 두 개를 놓친다. OLMo2 microanneal experiment3은 data format을 검출했다. code-form answer TinyGSM은 GSM*28.5→25.0, natural-language rewrite는65.5였다([[olmo-midtraining-evidence]] OLMo2 Table12). removal attribution은 reverse probe다. 10% DU window에서 math dataset을 빼면 MMLU29.71/GSM8K11.37로 math DU(43.19,20.47)와 no-DU baseline(35.69,14.71) 모두보다 낮았다([[domain-upsampling-end-of-training]] Table7).

### 3.2 Probe gain은 scale에서 줄어든다

Llama3은 experiment로 GSM8k/MATH train set에 anneal했다. pre-trained8B validation gain은24.0%/6.4%이나 “405B model improvement는 negligible”이었다([[llama-3]] §3.1.3; **결과 단일 연구**). 저자는 405B가 in-domain sample을 필요로 하지 않는다고 해석한다(**Interpretation**). 따라서 small/partly-trained probe는 final large model in-domain data 가치를 과대평가할 수 있다. released annealing에는 이 benchmark train set을 쓰지 않았다.

### 3.3 Contamination은 mid-training data에 집중된다

- OLMo2는 GSM8K 대부분을 hold out했다. decision에는 test1319개 중200 GSM*을 쓰고 final number는 나머지1119개에만 보고했다(arXiv:2501.00656v3 footnote6).
- Olmo3는 cited work의 “memorization occurs most strongly near end of training” 때문에 모든 OLMES benchmark all split에 대해 mid-training/long-context data를 decontaminate한다(§3.5.3). Flan에는 DROP validation 등 templated benchmark data가 있고 DROP training 60,000개 이상을 제거했다. DROP/Minerva/SQuAD는 contaminated data에서 높고, complete detected leakage에도 GSM8K는 decontamination 뒤 높았으며 저자는 format mismatch로 본다(§3.5.4).

Detection은 [[ch-48]]에 있다. mid-training source와 overlap한 benchmark gate는 capability뿐 아니라 memorization을 재므로 decontaminated data 또는 held-out split을 써야 한다.

## §4 Controlled evidence: distribution 사이 bridge로서 mid-training

[[midtraining-bridges-distributions]](arXiv v1 2025-10)는 mid-training이 target domain SFT를 높이고 pre-training distribution forgetting을 낮추는지 controlled study다.

Setup: C4에서128B token pre-training한 Pythia architecture 70M–1B; C4와 섞은 Starcoder/math(MAmmoTH+OpenMathInstruct)/FLAN/KnowledgeQA/DCLM mid-training mix를 same token C4 continuation과 비교; GSM8K/SciQ/CodeSearchNet-Python(PyCode)/LIMA SFT; SFT 뒤 C4 validation loss로 forgetting; seed5개(§3–4).

**저자가 진술한 mechanism**(§2.3). post-training은 initialization θ₀에서 target loss J_T로 K gradient step을 한다. J_P가 L_P-smooth, J_T가 L_T-smooth, η≤1/L_T이면 다음이다.

```
ΔP(K) ≤ −η Σ_{t=0}^{K−1} ⟨∇J_P(θ_t), ∇J_T(θ_t)⟩  +  L_P·η·(J_T(θ₀) − J_T*)
```

- ΔP(K)=J_P(θ_K)−J_P(θ₀): K SFT step 뒤 pre-training loss increase(forgetting).
- J_P/J_T: pre-training/target distribution population loss. J_T*: J_T minimum.
- η: SFT LR, L_P: J_P smoothness constant, ⟨·,·⟩: gradient inner product.

gradient가 align하면 first term은 negative다. second term은 θ₀가 target optimum에서 먼 만큼 커진다. mid-training은 θ₀만 바꾸므로 J_T(θ₀)를 낮추는 mix는 forgetting bound를 낮춘다(**Interpretation**, measurement가 아닌 bound).

**Proximity advantage**(Eq.6)는 `PA(M → T) = prox(M, T) − prox(C4, T)`이며 prox는 corpus token-unigram similarity다. Figure2 pure-source similarity로 `PA(StarCoder → PyCode)=.70−.54=.16`, `PA(Math → GSM8K)=.65−.61=.04`다. code mix는 data distribution을 target 쪽으로 더 이동시킨다.

**결과**(1B Table2, **결과 단일 연구**):

| SFT target | Mid-training mix | SFT loss Δ vs C4 | C4 loss Δ vs C4(forgetting) |
|---|---|---|---|
| PyCode | Starcoder20% |−.286|−.005|
| GSM8K | Math12% |−.091|−.116|
| GSM8K | Starcoder20% |−.015|+.024|
| SciQ | Starcoder20% |+.015|+.044|
| LIMA | FLAN5% |−.001|+.000|

1. benefit은 domain-specific다. matched mix는 target loss를 낮추며 mismatched mix는 benefit이 작고 forgetting을 올릴 수 있다(Starcoder→SciQ C4 loss+.044).
2. gain은 모든 size proximity advantage와 correlate한다(Figure3 panel r=.466–.622).
3. **mixed mid-training은 domain-only continued pretraining보다** target loss/forgetting 모두에서 높다. 160M PyCode: Starcoder20% mix SFT loss2.134/C4 loss5.079, 100% Starcoder continuation2.219/5.369(Table3).
4. **timing과 weight는 interact한다.** code에서 80% weight는 early introduction이면 best지만105B token introduction에서는10%보다 나쁘다. later weight 상승(42B10%→63B20%→84B30%)은 late introduction을 보상하지 못했다(§6 Figure4).

limit: parameter 최대1B, benchmark accuracy가 아닌 loss metric, SFT만(no RL), C4 loss 하나만 forgetting measure다. 저자는 scale/RL이 untested라 적었다(§9). multi-trillion pre-training 7B+ transfer는 **Open question**이다. general model에는 specialized phase에 general data를 섞는 것이 이 controlled study/OLMo mix 모두에서 지지된다(다른 metric/scale로 방향상 **Replicated**). timing result는 depth 있게 acquire할 specialized domain은 late heavy mix보다 early introduction benefit이라는 **Interpretation**이다.

## §5 Later RL을 위해 mid-training이 바꾸는 것

### 5.1 OctoThinker: mid-training data가 base RL scaling을 결정

문제: verifiable reward에서 base model 직접 RL인 zero RL(R1-Zero처럼; [[ch-40]])은 Qwen2.5-3B-Base에서는 작동했지만 Llama-3.2-3B-Base는 `\boxed:{}`로 시작해4,096-token limit까지 반복했다([[octothinker]] §2.2).

Llama-3.2-3B-Base controlled mid-training(20B token, cosine LR peak3e-5; 뒤 MATH8K GRPO prompt/rollout16/temperature1.0/KL0; §2.1,§3.1)은 다음을 보였다.

1. **Math web quality**: MegaMath-Web-Pro는 RL 뒤 “significant gain”, FineMath-4plus는 marginal gain이며 response가 length limit까지 자랐다(§3.2).
2. **QA format**: web:QA=9:1에서 short-CoT QA는 web-only 대비 RL gain이 없고 long-CoT QA는 RL gain은 있으나 sudden score drop/length spike의 unstable training이다(§3.3).
3. **Instruction data**: instruction1%(instruction:web:QA=1:89:10)는 short-CoT QA RL을 개선하고 length를 stabilize한다. long-CoT에서는 late decline을 막지 못한다(§3.4).
4. **Budget**: mid-training20B/70B/100B checkpoint에서 base score 70B/100B는 comparable이나 RL result는20→70→100B로 좋아졌다(§3.5). “base model evaluation에서 보이지 않아도 mid-training budget 증가가 RL performance를 높일 수 있다.”

final **Stable-then-Decay**는 constant LR 2e-5(3B)로200B token 학습 후, long/short/both CoT QA30%의 20B-token decay branch3개(cosine2e-5→2e-6)를 만든다(Tables2–5). pilot은 QA30% 초과 gain plateau를 찾았다(§4.2.1). MATH500 Llama3.2-3B base7.40→OctoThinker-Long-3B-Base25.80(Table7), zero RL 뒤 Llama3.2-3B-Zero10.0 vs OctoThinker-Long-3B-Zero65.2이며 Qwen2.5-3B-Zero66.4에 가깝다(Figure1 bar label).

조건/한계: evaluation은 모두 math benchmark, mid-training/RL 뒤 general capability 미보고, RL figure는 MATH8K max400 step, seed 미보고다. result는 math RL-ready base를 만들 수 있음을 보일 뿐 breadth effect는 보이지 않는다. long branch long-CoT는 longer output/RL instability도 유도하여 prompt template과 staged response length limit(2,048→4,096→8,192)이 필요했다(§3.4). RL length control은 [[ch-44a]].

### 5.2 Interplay study: mid-training/RL compute split

[[interplay-pretraining-midtraining-rl]](arXiv v1 2025-12)는 2–10 operation synthetic arithmetic-reasoning10B token으로 pre-training한 100M model에서, 20% 2–10-operation data와 섞은11–14-operation next-token mid-training을 same distribution GRPO와 비교한다(§2, App.A.3.3,A.10). **extrapolative generalization**은 training보다 operation이 많은 problem을 푸는 것이며 OOD-edge=11–14, OOD-hard=15–20이다.

compute equivalence(App.A.10.1 Eq.4–8): T token training cost≈6PT FLOP, RL sample은 rollout token sequence당≈(8+2γ)·P·L_total FLOP이므로:

```
T_RL ≈ (4/3 + γ/3) · N · r · L_total ,   and   T_mid = (1 − β)·T ,  T_RL = β·T
```

- P: non-embedding parameter, N: RL sample, r=6: sample당 rollout, L_total=2,048: prompt+response token.
- γ=1은 reference-model forward를 쓸 때(main text5/3 factor), β=total budget T 중 RL share.

T=2.1B token-equivalent(β=1이 Full RL arm; paper §5 Full mid-training은1B supervised token), β=.2 Light-RL이면 T_RL=.42B; `N=(3/5)×.42×10⁹÷(6×2,048)=.252×10⁹÷12,288≈20,500` sample, batch1024 약20 step; T_mid=1.68B다. β=1이면 N≈102,500/batch1024 약100 step으로 §5 Full RL100 step과 맞는다.

**결과(단일 연구, synthetic task)**: fixed compute에서 mid-training+RL은 RL alone보다 OOD-hard +10.8%(Figure1 caption). OOD-edge는 full mid-training+light RL pass@1 best, OOD-hard는 heavier RL best(Observation3). budget order도 바뀐다:4.2B equivalent Heavy-RL OOD-hard pass@128 best, 8.4B 이상 Full RL best(Observation8, App.A.10). “mid-training beats RL”은 일부 metric/budget에만 성립한다.

조건/한계:100M,2,200-token vocabulary synthetic arithmetic grammar, natural-language breadth 없음. “majority compute mid-training/light RL, complex task exploration heavier RL” guidance는 이 setting에서 나온 것이며 real model에는 **Interpretation**이다.

- OctoThinker controlled evidence는 math benchmark만을 대상으로 하므로 general capability retention의 evidence가 아니다.
- base benchmark score가 comparable해도 RL outcome은 mid-training token budget에 따라 달라질 수 있다.
- synthetic interplay의 compute-equivalence 결론은 natural-language model의 universal allocation rule이 아니다.

## §6 Cold-start data, mid-training reasoning trace, distillation

**Cold start**는 RL 전 small SFT stage로 reasoning output format/readability를 정한다. DeepSeek-R1 SFT data set 두 개는 size가 100배 이상 차이나 혼동하기 쉽다.

- DeepSeek-R1 cold start는 R1-Zero output long-CoT example “thousands”다. temperature1.0 sample, correct answer/readability filter, DeepSeek-V3 refine, human annotation verification으로 만든다([[deepseek-r1]]; [[deepseek-r1-recipe]] v2 §3, Supp.B.3.2). exact count는 보고하지 않았다.
- 약800K(804,745: reasoning 약600k+non-reasoning 약200k)는 later rejection-sampling SFT stage로 DeepSeek-V3-Base에서 restart한다. 이는 R1-Distill student6개 distill에 같은 set이다(Supp.B.3.3 Table5, B.4.2,B.4.3).
- stage effect(Table3): cold start Dev1은 R1-Zero IF-Eval46.6→71.7, AlpacaEval2.0 24.7→50.1이지만 AIME2024 77.9→59.0이며 저자는 small cold-start set 탓으로 본다. non-reasoning SFT Dev3은 AlpacaEval2.0 55.8→62.1, Aider-Polyglot25.6→44.8(§4)이다. reasoning-focused stage 뒤 generality repair다.

reasoning trace는 이제 pipeline earlier distillation인 mid-training에도 들어간다(teacher output을 next-token data로 사용; SFT view [[ch-20]],[[ch-35]]).

| Run | Mid-training teacher trace | Amount | Source |
|---|---|---|---|
| Olmo3 7B | QwQ/Llama Nemotron/OpenThoughts2/Gemini reasoning trace; synthetic meta-reasoning |100B의8.34%(derived)|Olmo3 Table5|
| SmolLM3 3B | OpenThoughts3-1.2M, R1 trace 포함 Llama-Nemotron post-training subset |35B token×4 epoch≈140B|SmolLM3 blog|
| OctoThinker-Long | OpenR1-Math-220K, AM-DeepSeek-Distilled-40M |20B decay token30%|OctoThinker Table5|

Source: [[olmo-midtraining-evidence]], [[smollm-3]], [[octothinker]]. base가 mid-training에서 already long trace/instruction data를 보면 SFT는 처음 format을 introduce하지 않는다. Olmo3 thinking/instruction data는 post-training 전 base eval을 높였고(Table10), OctoThinker RL은 already long-CoT output base에서 시작했다. reasoning mid-training 뒤 필요한 cold-start size나 cold start skip 가능성은 source가 측정하지 않아 **Open question**이다. measured cost 두 가지는 OctoThinker long-CoT branch의 RL stabilizer 필요(§3.4), SmolLM3 RULER regression이 reasoning mid-training stage로 trace된 것이다(§7).

## §7 Long-context extension과 mid-training 대비 placement

여러 report는 long-context extension을 mid-training 곁에서 실행하며 model family별 placement가 다르다. Olmo3 저자 요약: “Llama3.1은 midtraining 전에, Qwen2.5/3은 뒤에, GLM4.5는 supervised finetuning 뒤에만 extension을 적용한다”(arXiv:2512.13961v2 §3.6). Llama3 자체 report는 six-stage8K→128K extension(~800B token)을 40M-token anneal보다 앞에 둔다. 각 stage는 short-context evaluation이 “recovered completely”이고 해당 length needle-in-a-haystack을 풀 때만 진행한다(§3.4.2).

측정한 short-context cost/control:

- **Olmo3**: Stage3는 full-attention YaRN(8,192→65,536 token)으로 long document34%/Dolmino short66%를 50B(7B) 또는100B(32B) token 섞는다. 10B extension test에서 long66%/short34%는 OlmoBaseEval subset −2.5 point, long34%/short66%는 −.8였다(§3.6.3). short data share에도 Stage2→3에서7B Math59.8→54.4/GenXL49.1→43.6,32B Math69.7→61.4였다(Table13).
- **SmolLM3**: long-context extension(4k→32k→64k,100B token) 뒤 reasoning mid-training(140B token)이 왔다. post-training RULER이 degrade했고 team은 “reasoning mid-training stage로 trace”했다. long-context strong mid-training checkpoint .1과 preference-trained soup .9 linear merge는128k까지 base RULER score를 회복했다(SmolLM3 blog “Model Merging”).
- **Llama3**: short-context data-only SFT는 long-context regression을 만들었다. table 미출력 ablation에서 synthetic long-context SFT .1%가 short/long benchmark를 함께 optimize한 share였다(§4.3.4).

Olmo3 mix test에서 long data가 약2배(66% vs34%)면 OlmoBaseEval subset drop은 약3배(2.5÷.8≈3.1)였다(10B run). budget 하나의 test다. general model에는 long-context extension 뒤 모든 stage(reasoning mid-training/SFT/preference training)에 long-context retention check, long-context stage 자체에는 short-context retention check가 필요하다. placement evidence는 [[ch-32b]], effective context length 측정은 [[ch-32c]]에 있다.

## §8 Alignment와 specification mid-training

official non-paper report 두 개가 alignment content document training을 mid-training stage로 시험했고, 서로 다른 condition에서 다른 결과에 도달한다.

**Model Spec midtraining(MSM)**([[anthropic-model-spec-midtraining]], Anthropic Alignment Science blog 2026-05, paper arXiv:2605.02087)은 Model Spec을 논의하는 synthetic document LoRA training 뒤 alignment SFT(AFT)다. Qwen2.5-32B-Instruct/Qwen3-32B에서 MSM+AFT agentic-misalignment average는68%→5%,54%→7%였으며 deliberative-alignment-style AFT baseline48%/14%보다 낮다. Qwen MSM은41M token이고 MSM+AFT는 AFT sample1,250–80k 모든 size에서 AFT-only보다 높았다([[anthropic-model-spec-midtraining-recipe]]). MSM 뒤 stage는 SFT뿐이며 MSM 뒤 RL은 시험하지 않았다(paper§7).

**o4-mini scale alignment midtraining**([[openai-alignment-midtraining-generalization]], OpenAI Alignment blog 2026-03)은 o4-mini-sized model에 aligned action을 고르는 AI fictional scenario document230k(약340M token) 또는 matched misaligned split을 other data mix 없이 주고 safety training 없는 SFT/RLVR를 뒤에 둔다. training distribution 가까운 QA eval에 effect가 나며 chat eval에서는 aligned/misaligned run이 two eval 외 모두 similar했다. agentic eval에서는 run3개 average가 “not significantly different”였다. training distribution 근처 misalignment effect는 “reasoning post training을 거치면 disappear하는 경향”이 있다.

condition 비교: MSM은 already post-trained32B checkpoint에 one specific spec/reason document, SFT만 뒤에 둔다. OpenAI는 standard pretraining 완료 model에 scenario fiction, SFT+RLVR를 둔다. MSM 저자는 high-data regime Qwen3-32B에서 CoT AFT가 MSM+AFT에 converge했으므로 MSM이 “high-compute reasoning post-training에 scale하지 않을 수 있다”고 한다([[anthropic-model-spec-midtraining]]). principle document mid-training이 large-scale RL을 지나 지속되는지는 **Open question**이다. general model에는 downstream full pipeline 뒤 평가해야 한다. RL이 SFT 뒤 measured difference를 지울 수 있기 때문이다(OpenAI result가 뒷받침하는 **Interpretation**).

## Negative sample과 negative feedback

Mid-training은 [[ch-31a]], [[ch-43a]] 네 의미 중 두 가지로 negative를 만들고 쓰며 negative gradient는 쓰지 않는다.

1. **negative marginal value(type1)**: data probe는 positive training data로 썼을 때 performance를 낮추는 source를 찾는다. OLMo2 code-answer TinyGSM은 GSM*28.5→25.0이다. removal probe는 반대 label이다. DU window math source를 빼면 MMLU29.71로 no-DU35.69보다 낮아 source는 positive marginal value다. practice는 type1을 discard/rewrite하며 natural-language TinyGSM rewrite는65.5다. label signal은 probe Δ이며 source 두 개 모두 noise를 보고하지 않는다.
2. **content로서 negative(type2)**: OpenAI misalignment-midtraining은 misaligned action/reason 문서 약340M token에서 ordinary next-token loss를 학습했다. chat eval은 alignment midtraining과 similar하고 no-midtraining baseline 이상이며, 저자는 document가 alignment failure를 salient하게 한다고 가설한다(**Interpretation**). MSM은 MSM 뒤 “anti-spec” response AFT가 anti-spec AFT-only보다 lower misalignment이라고 보고한다(card,paper§5.3).
3. **diagnostic**: type1에는 target뿐 아니라 untargeted cluster Δ를 보고한다(§3.1 Δ_MMLU=−1.5). type2에는 full post-training pipeline 뒤 near/far distribution alignment를 모두 평가한다.
4. **generality effect**: mid-training negative-content document의 calibration/over-refusal change를 어떤 source도 측정하지 않는다. negative gradient는 preference optimization/RL에 속한다([[ch-39]],[[ch-43a]]).

## 레시피

모든 row는 stated locus primary source에서 복사했다. “verified 2026-09-15”는 해당 revision에서 읽었고 2026-09-14는 verified library ledger다.

| Model | Size | Stage | Setting | Value | Source | Status | Evidence |
|---|---|---|---|---|---|---|---|
| Llama3.1 |405B|anneal|token/LR/data/average|final40M;128K linear-zero;very-high-quality upsample;Polyak|[[llama-3]]§3.4.3|verified 2026-09-14|benchmark training set excluded|
| Llama3.1 |405B|long-context|stage/token/gate|six8K→128K;~800B;short eval recover+NIAH solved|§3.4.2|verified|stage length/token 미출력|
| Llama3 8B experiment|8B|eval-gate|probe|50%-trained;40B linear-zero;30%new/70%default|§3.1.3|verified|number 미출력|
| Llama3 experiment|8B/405B|anneal|GSM8k/MATH train|8B+24.0%/+6.4%;405B negligible|§3.1.3|verified|released data 아님|
| OLMo2|7B|midtrain|50B run/LR/merge|3×50B different order;linear-zero;average|§2.3§4.5|verified|six mix broad soup≥best|
| Olmo3|7B|midtrain|token/seq/batch|100B;8192;2²¹ token|OLMo-core script|verified|integration avg49.7→53.1|
| Olmo3|7B|mix|web+PDF/code/math/QA/thinking/instruction|32.5/20/19.2/13.9/8.34/6.1|Table5|derived|Table7/10|
| Olmo3 development|7B|probe|5B target+5B web vs10B web|then100B integration+SFT eval|§3.5.1|verified|80 math probe/25 source|
| MiniCPM .036B|.036B|decay|length|10% sufficient;2.5% falls short|§4.2|verified|Figure5|
| MPT-style|7B|DU|window/mix|final20%, largeCC0/small30/domain35/code35|Table4|verified|10–20% trade-off|
| OctoThinker|3B|stable|200B/50k/2e-5/8192/batch512/wd.1/no warmup|Table2–3|verified|20→70→100B improves RL|
| OctoThinker|3B|decay|20B/5k/cosine2e-5→2e-6/QA30%|Table4–5|verified|QA>30% plateau|
| SmolLM3|3B|reasoning|35B×4≈140B;ChatML|blog|verified|RULER regression/merge recovery|
| DeepSeek-R1 Dev1|671B MoE|SFT|cold-start|thousands; exact 없음|[[deepseek-r1]]|verified|AIME59 vs R1-Zero77.9|
| Liu Pythia|70M–1B|midtrain|Starcoder20/Math12/FLAN5/KnowledgeQA20/DCLM20+C4;128B C4 cosine3e-4|Table2|verified|mix beats100% continuation|
| MSM Qwen|32B|midtrain|41M;LoRA r64 α128 LR1e-4 epoch1|paper/AppB.4|conflict(AppE40M)|Figure4|
| OpenAI o4-mini scale|not reported|midtrain|230k document≈340M;no other data|blog|verified|chat/agentic not significant|
| OLMo2 7B |7B|stable|peak LR/warmup/cosine|3×10⁻⁴;2,000;5T cosine 후4T truncate|[[olmo-2]] Table3 §4.1|verified|LR3–12×10⁻⁴ decay 뒤 OLMES2점 이내|
| OLMo2 13B/32B |13B,32B|midtrain|run/merge|3×100B+1×300B average|§2.3|verified|7B와 same step/larger batch|
| OLMo2 7B |7B|mix|Dolmino50B|DCLM47.2/FLAN16.6/math20.8/Wiki7.11/peS2o5.85/StackExchange2.45|Table13|verified|web+math+instruction best overall|
| OLMo2 7B |7B|probe|~50/50 source/web, linear-down,19 run130B|§4.4.2|verified|Table12|
| Olmo3 7B |7B|optimizer|AdamW LR2.0712e-4,betas(.9,.95),wd.1; warmup0/alpha_f0; optimizer state loaded|OLMo-core script|verified|ablation 없음|
| Olmo3 32B |32B|merge|different seed100B run2개 merge|§3.5.4|verified|Math +2.9/+1.6|
| Olmo3 7B |7B|long-context|50B;long34/short66;YaRN×8;65,536;warmup200/alpha_f0;CP8|§3.6|verified|10B 66/34−2.5 vs34/66−.8|
| MiniCPM2.4B |2.4B|stable|~1T/batch3.93M/maxLR.01 WSD|§6.2|verified|wind-tunnel|
| MiniCPM2.4B |2.4B|decay|pretraining+high-quality SFT; exponential T=5000(20B)|§6.2|verified|A-2 vs A-1|
| SmolLM3 3B |3B|decay|10T→11.1T;web63/code24/math13;final10% linear-zero|blog|verified|ablation 없음|
| synthetic 100M interplay |100M|midtrain|LR1e-4/wd.1/cosine min3e-5/warmup15%/batch512K/seq2048;20%op2–10+80%op11–14|App.A.3.3,A.10|verified|Figure6 Obs3|

**작은 7B general run 시작점.** verified Olmo3는100B,seq8192,batch2²¹,AdamW(.9,.95),wd.1,LR2.0712×10⁻⁴ linear-zero/no warmup/pre-training optimizer state다. OLMo2 DCLM47.2%/Olmo3 web+PDF32.5%처럼 general data를 크게 유지하며 math/code/QA/instruction을 더하고 chat special token은 제거한다. candidate는 Olmo3 5B+5B vs10B probe 또는 Llama3 50%-8B/40B/30% candidate probe를 쓴다. long-context extension 뒤에는 short data majority(Olmo3 7B66%,50B)를 유지한다. 이 값은 cited team 7B run이며 other size ablation은 없다.

## 일반화 관점

- **Breadth 증가:** general data 포함 mix는 target loss/forgetting에서 domain-only continuation보다 높고([[midtraining-bridges-distributions]] Table3), Olmo3 balanced mix는 target/non-target cluster를 같이 유지한다. fixed budget instruction/thinking 추가는 every base cluster 상승(Table10), OLMo2 soup는 six mix broad average에서 best single 이상, DU10–20%는 targeted score 계속 상승 전 broad average peak다.
- **Narrowing:** math-code-thinking mix는 final 대비 MC Non-STEM−7.8/GenQA−7.2, DU20→30%는 GSM8K↑ MMLU/Core↓, mismatched Starcoder→SciQ C4 forgetting+.044, long-context/reasoning stage는 Olmo3 Math59.8→54.4/SmolLM3 RULER regression, special token/long-CoT format artifact, 8B gain405B negligible이다.
- **측정:** matched web-only baseline 대비 모든 cluster Δ(untargeted first), short SFT/후속 RL 뒤 candidate 평가, all benchmark split decontaminate/held-out final split, stage boundary retention check, alignment midtraining full pipeline/RL 뒤 평가. error는 DU/OctoThinker/MiniCPM one-run/no variance, GSM*200 question, contamination score raise/lower다.

## 흔한 실수와 탐지 방법

| 실수 | 증상 | 확인 |
|---|---|---|
| fixed budget % rule | ablation trace 없는 budget | .00026% Llama3~20% DU 비교/duration sweep |
| target benchmark만 probe | MMLU/GenQA 하락 source accept | web-only 대비 all Δ(Olmo3 Δ_MMLU−1.5) |
| domain-only | target gain/SFT forgetting↑ | general share+C4 held-out loss |
| chat special token | base가 `<|im_start|>` emit/GSM parse fail | generation grep/with-without microanneal |
| benchmark split 포함 | gate만 rise | all split n-gram decon/final holdout |
| small probe 신뢰 | 8B gain final 무효 | larger checkpoint repeat |
| base score로 RL-ready 판단 | pre-RL same/post-RL diff | candidate fixed short RL |
| long context short data 부족 | short cluster drop | stage gate short eval/mix compare |
| reasoning 뒤 retention 없음 | post RULER drop | later each stage RULER/merge |
| R1 cold-start=800K 인용 | size plan error | thousands vs804,745 rejection SFT 확인 |

## 이해 점검

1. MiniCPM B-3 half SFT token/C-Eval·MATH↑ GSM8K↓가 data placement evidence인 이유와 general chat에 못 말하는 것을 설명하라.
2. DU20→30% GSM↑/MMLU·Core↓ mechanism과 추가 measurement를 제안하라.
3. Liu forgetting bound의 alignment/effort term을 mix가 각각 낮추는 방법과 bound 한계를 설명하라.
4. 8B large gain/405B negligible이 405B mix probe에 주는 함의와 risk-reducing design을 말하라.
5. OctoThinker 70B/100B base equal/RL diff의 causal explanation 두 가지를 말하라.
6. OOD-edge/hard와 budget에서 split optimum이 다른 이유를 pass@1/pass@128으로 설명하라.
7. MSM/OpenAI difference의 data/model/downstream stage와 likely cause를 나열하라.
8. short data66%에도 Math59.8→54.4인 reason 둘과 SFT 전 gate를 설계하라.

## 연결

- 이전 ch-17 mixture-ablation lab; 다음 ch-32a continual pretraining without forgetting.
- dependency ch-03 LR schedule, ch-13 domain mixing.
- same phase ch-32b long context, ch-32c effective context, ch-32d agentic data, ch-32e recipe, ch-32f lab.
- later ch-20 distillation, ch-30c merge, ch-31 cold start, ch-40 RL, ch-48 contamination.

## 출처

- [[olmo-2]], [[olmo-3]], [[olmo-midtraining-evidence]] — definition/mix/soup/probe/decon/long context config.
- [[minicpm]], [[llama-3]], [[llama-3-recipe]], [[domain-upsampling-end-of-training]] — WSD/anneal/probe/DU.
- [[octothinker]], [[interplay-pretraining-midtraining-rl]], [[midtraining-bridges-distributions]], [[mid-training-survey]] — RL readiness/compute split/controlled bridge/taxonomy.
- [[deepseek-r1]], [[deepseek-r1-recipe]], [[smollm-3]] — cold start/reasoning trace/RULER.
- [[anthropic-model-spec-midtraining]], [[anthropic-model-spec-midtraining-recipe]], [[openai-alignment-midtraining-generalization]] — alignment document midtraining.
- Llama3.1 final anneal은 40M token이며 long-context stage 약800B token보다 훨씬 작다.
- Llama3 released anneal data는 common benchmark training set을 제외했다.
- OLMo2 soup는 mix E GSM*에서60.5→43.0으로 떨어진 caveat가 있다.
- Olmo3 7B는 early experiment similar merge gain이 없어 single mid-training run을 썼다.
- Microanneal은 candidate source만이 아니라 format도 probe할 수 있다.
- Candidate Δ는 targeted benchmark뿐 아니라 untargeted cluster에서 negative일 수 있다.
- Small-scale probe가 measurement noise일 수 있는 이유는 random accuracy다.
- Full-scale mixture ablation cost는 pre-training만큼일 수 있다.
- Llama3 8B probe는50%-trained checkpoint에서 시작한다.
- Olmo3 integration test는 SFT 뒤에도 candidate를 평가한다.
- OLMo2 GSM* decision set은200 question이고 final report set은1,119 question이다.
- Olmo3는 all OLMES benchmark all split으로 decontaminate한다.
- DROP/Minerva/SQuAD는 contaminated data에서 더 높았다.
- GSM8K는 complete leakage에도 decontamination 뒤 더 높았고 format mismatch 해석이다.
- Early specialized-domain introduction은 late heavy weight보다 유리할 수 있다.
- Late weight schedule10%@42B→20%@63B→30%@84B는 late introduction을 보상하지 못했다.
- OctoThinker FineMath-4plus는 marginal RL gain과 length-limit response를 보였다.
- Short-CoT QA만은 web-only보다 RL gain이 없었다.
- Long-CoT QA는 gain과 sudden drop/length spike를 함께 만들었다.
- Instruction1%는 short-CoT RL length를 stabilize했다.
- Long-CoT 경우 instruction1%는 late decline을 막지 못했다.
- Mid-training+RL OOD-hard +10.8%는 fixed compute synthetic study result다.
- Heavy/Full RL ordering은4.2B,8.4B 이상 budget에서 바뀐다.

계산 예제: S=100,000,000,000 token÷batch당2,097,152 token=47,683.7, 즉 47,684 step. released long-context script가 `Olmo-3-1025-7B/stage2/step47684`를 load해 일치한다(long-context script L100). halfway s=23,842에서 η=2.0712×10⁻⁴×(1−.5)=1.0356×10⁻⁴다. `load_optim_state=True`는 AdamW moment estimate를 pre-training에서 carry하므로 stage는 warmup 없이 시작한다.

OLMo2는 higher pre-training LR가 decay 효과를 높이는지 test했다. 7B peak LR 3×10⁻⁴–12×10⁻⁴로 300B token 학습하고 50B/100B token zero-decay했다. OLMES average 차이는 2 point 미만이며 “higher LR makes mid-training more effective, but exactly by amount pretraining is worse”였다(arXiv:2501.00656v3 §4.1 Table8; **결과, 단일 연구**). 저자가 든 exception은 2T run GSM8K high-LR +2.8 point이며 추가 연구가 필요하다. general model에서 decay stage는 pre-training 대비 cheap하며 data mixture decision은 LR schedule과 별개다. schedule shape는 [[ch-03]]에 있다.
