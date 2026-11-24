<!-- chapter: ch-06
     track: foundations
     kind: content
     title: Checkpointing, In-Loop Evaluation, and Checkpoint Selection
     deps: [ch-05]
     sources: [[early-stopping-and-checkpointing]], [[paloma]], [[olmes]], [[signal-and-noise-eval]], [[datadecide]], [[merging-in-pretraining]], [[model-soups]], [[wise-ft]], [[llama-3]], [[tulu-3]], [[olmo-2]], [[olmo-3]], [[olmo-core-olmo3-configs]], [[smollm3-training-configs]]
     figures: figures/checkpoint-state.html, figures/checkpoint-selection.html
     revised: 2026-09 (generality revision)
-->

# Chapter 6 — Checkpointing, In-Loop Evaluation, Checkpoint Selection

> **핵심 통찰.** Training run 하나는 model 하나를 만들지 않는다. Checkpoint sequence를 만들며, 어느 checkpoint(또는 checkpoint 평균)를 release할지의 선택은 이 stage의 대다수 hyperparameter 선택보다 더 크게 측정 capability를 바꾼다. 두 measurement가 이 선택을 잘 결정한다. Domain별 held-out loss는 단일 validation loss가 숨기는 fit gap을 드러낸다. [[paloma]]에서 C4-only 1B baseline은 RedPajama arXiv domain에서 perplexity 391,171에 도달한다(§4.1). 또한 signal-to-noise ratio를 아는 fixed format으로 downstream evaluation을 수행해야 한다. [[signal-and-noise-eval]]에서는 benchmark의 small-scale SNR이 small-scale ranking이 larger model로 transfer하는지를 예측한다(`R² = 0.626`, §4.1).
>
> **지침.** Run이 resume될 만큼 길다면 model과 optimizer state와 함께 data-loader position, RNG state, step counter, scheduler state를 저장하고, load할 때 dataset fingerprint로 loader position을 확인하라([[olmo-core-olmo3-configs]]). Checkpoint를 선택해야 한다면 single best checkpoint보다 마지막 checkpoint 여러 개의 average를 선호하라. [[signal-and-noise-eval]]에서 averaging은 task 30개 decision accuracy를 68.9%에서 71.3%로 올리고 13B scaling-law prediction error를 absolute point 1.03에서 0.86으로 낮췄으며(§5.2), [[olmo-2]]에서는 checkpoint 3개 soup이 시험한 mix 6개 모두에서 best single mid-training run과 같거나 더 나았다(§4.5, Table 14). Stage가 pre-training이 아니라 fine-tuning이면, development decision에 한 번도 쓰지 않은 held-out suite에서 멈춰라. [[tulu-3]]에서는 시험한 모든 model이 structurally matched IFEval-OOD보다 IFEval에서 더 높다(Tülu 3 8B: 82.4 vs 24.3, §7.4.1, Table 31).

---

## 이 장이 범용 모델에 중요한 이유

[[ch-05]]은 step이 실제로 보는 token 수를 확정했다. 이 장은 step 사이에서 무슨 일이 일어나는지, 즉 무엇을 disk에 기록하고 무엇을 측정하며 그 measurement가 어느 weight가 release model이 되는지를 어떻게 결정하는지 다룬다. §1-§5 및 §7은 pre-training과 mid-training이고 §6은 SFT지만, mechanism은 뒤에서도 반복된다. [[tulu-3]]은 SFT에 쓰는 같은 procedure로 RLVR checkpoint를 선택한다.

범용 capability란 target으로 삼지 않은 task와 domain에서도 수행하는 것이다. Measurement loop의 모든 부분은 이 property의 손실을 숨길 수 있다.

1. Held-out loss 하나는 domain에 걸쳐 average를 내므로 한 domain의 fit gap은 average에서 보이지 않는다.
2. Adjacent checkpoint 사이에서 두 candidate recipe 사이보다 더 많이 움직이는 downstream benchmark score는, instance가 아무리 많아도 recipe의 순위를 매길 수 없다.
3. Decision을 내기 위해 사용하는 development suite는 그에 맞춘 decision이 충분히 쌓이면 generalization을 더 이상 측정하지 않는다.
4. Development suite에서 best score인 checkpoint 하나를 선택하면 capability와 그 suite noise를 함께 선택한다.

§2~§6은 이 문제들을 순서대로 다룬다. §1과 §7은 mechanical prerequisite를 다룬다. Data order를 조용히 바꾸는 resume은 그 위의 모든 measurement를 무효화한다.

---

## §1 Resume 가능한 checkpoint에 들어 있는 것

**정의.** Resume 가능한 checkpoint는 멈춘 run이 같은 trajectory를 계속하게 하는 state다. Model parameter, optimizer state, 그리고 다음 step이 무엇을 보게 되는지를 결정하는 bookkeeping state를 포함한다.

**측정 가능한 문제.** Parameter와 optimizer state는 복원하지만 bookkeeping state는 복원하지 않는 resume은 error도 loss curve의 눈에 띄는 jump도 만들지 않는다. 그러나 model이 다음에 보는 data를 바꾸므로, [[ch-05]]의 tokens-seen ledger와 §2의 모든 per-domain measurement가 더 이상 run을 설명하지 못한다.

**메커니즘.** OLMo-core는 두 half를 분리한다. Train module을 checkpointer에 넘겨 model/optimizer state를 쓰고, 나머지는 보통 Python dictionary다(`src/olmo_core/train/trainer.py` L87–96, [[olmo-core-olmo3-configs]]).

```python
class TrainerStateDict(TypedDict):
    global_step: int
    global_train_tokens_seen: int
    global_train_petaflops: float
    max_steps: Optional[int]
    data_loader: Dict[str, Any]
    epoch: int
    world_size: int
    rng: Dict[str, Any]
    callbacks: Dict[str, Dict[str, Any]]
```

Data-loader entry도 dictionary다(`src/olmo_core/data/data_loader.py` L441–449).

```python
def state_dict(self) -> Dict[str, Any]:
    return {
        "dataset_fingerprint_version": self.dataset.fingerprint_version,
        "dataset_fingerprint": self.dataset.fingerprint,
        "batches_processed": self.batches_processed,
        "tokens_processed": self.tokens_processed,
        "seed": self.seed,
        "epoch": self._epoch,
    }
```

이 code의 property 세 개는 carefully 읽을 가치가 있다. 각각은 resume이 멈추지 않았던 경우와 동등하지 않은 조건이다.

1. **Dataset fingerprint는 검사하고 checkpoint의 seed가 config seed를 override한다.** Load 시 `dataset_fingerprint`가 다르면 `ignore_fingerprint_mismatch=True`가 아닌 한 `RuntimeError`(“Dataset fingerprint does not match the fingerprint in the checkpoint … This will probably result in a different data order!”)를 낸다. 이 option이면 같은 text의 warning으로 낮아진다(L451–467). `seed` 차이는 반대로 처리한다. Loader는 warning(“Restoring data loading state with a different data seed, will use data seed from state dict for data order consistency.”)을 낸 뒤 `self.seed = state_dict["seed"]`를 할당한다(L469–473). 따라서 resume 때 config의 data seed를 바꿔도 data order는 바뀌지 않으며, 둘 중 run을 멈출 수 있는 것은 fingerprint뿐이다.
2. **RNG 복원은 world size에 조건부다.** `trainer.py` L838–848은 `state_dict["world_size"] == get_world_size()`일 때만 rank별 RNG를 복원한다. 그렇지 않으면 “Trainer will not restore rank RNG states since the RNG states in the checkpoint were saved with a different world size.”라고 log한다. 그러므로 다른 GPU 수에서 resume하면 설계상 RNG stream의 exact reproduction을 포기한다.
3. **Position은 token으로 저장하고 batch로 재계산한다.** Fixed-sequence-length loader는 load 때 `self.batches_processed = self.tokens_processed // self.global_batch_size`를 다시 계산한다(L757–764). 따라서 다른 global batch size로 resume하면 failure 대신 가장 가까운 batch boundary에 도착한다.

**숫자 예시.** Run의 `global_batch_size = 4,194,304` token(Olmo 3 7B setting인 8,192 × 512)이고 `tokens_processed = 419,430,400` token 뒤에 멈춘다고 하자. 이는 batch 100개다. Batch size를 두 배인 8,388,608 token으로 하여 resume하면 `batches_processed = 419,430,400 // 8,388,608 = 50`이다. Loader는 올바른 token position을 보고하지만, `global_step`에서 복원한 step counter는 이에 맞지 않는다. Run은 token 419,430,400에서 이어 가되 step 50이 아니라 step 100에서 계속된다. Step으로 표현한 schedule은 이제 token ledger에 대해 offset된다. 이것이 [[ch-05]]의 규칙, 즉 run 전에 unit을 고정하라는 규칙이 launch뿐 아니라 resume에도 적용되는 이유다.

**의도적으로 schedule을 바꾸는 resume.** Olmo 3 7B pre-training은 script 두 개로 실행됐다. 첫 script는 `CosWithWarmup(warmup_steps=2000)`과 `max_duration=Duration.tokens(int(5e12))`를 썼다. 둘째 script는 `step596047`에서 `load_trainer_state=True, load_optim_state=True`, `max_duration=Duration.tokens(int(7e12))  # Changed from 5T -> 7T`로 resume하고 scheduler를 `HalfCosWithWarmup(warmup_steps=original_max_steps // 2 + original_warmup_steps // 2)`로 바꿨다. File의 comment는 이를 “Scheduler updated to extend lr from where we left off”라고 한다([[olmo-core-olmo3-configs]]). Paper는 결과를 “The first half of the learning rate schedule is a cosine schedule over 5T tokens. We stretch the second half of the schedule to reach a target length of one epoch (5.93T tokens)”라고 설명한다([[olmo-3]], arXiv:2512.13961 Fig. 3 caption). Script의 7T는 schedule을 그리는 horizon이지 stop point가 아니다. Run은 corpus 한 epoch인 5.93T token에서 멈췄다. Scheduler state와 scheduler definition은 다른 fact다. Checkpoint에는 step count가, script에는 shape가 들어 있다.

**Configurable이며 automatic하지 않다.** SmolLM3의 nanotron config는 `checkpoint_interval: 2000` 옆에서 `load_optimizer: true`, `load_lr_scheduler: true`를 independent switch로 둔다([[smollm3-training-configs]]). 둘 다 true여야 resume은 continuation이고, 하나라도 false면 old weight로 초기화한 new run이다.

[`figures/checkpoint-state.html`](figures/checkpoint-state.html)의 interactive inventory는 각 state component를 끄고 그 경우 위 code가 무엇을 하는지, 각 entry의 file과 line을 함께 읽게 한다.

**범용 모델에 대한 함의.** Checkpoint의 bookkeeping half가 선택한 mixture proportion을 model이 실제로 본 mixture proportion으로 만든다. Crash 뒤 loader reset은 shuffled order의 시작을 다시 제공하므로, 앞쪽에 등장하는 source를 과대 대표한다. §7은 이를 탐지하는 방법을 제공한다.

## §2 In-loop measurement I: domain별 held-out loss

**정의.** Domain별 held-out loss는 data source 또는 domain마다 하나씩인 여러 disjoint validation set에서 model을 평가하고, 수치 하나로 합치지 않고 각각 보고한다.

**측정 가능한 문제.** [[paloma]] (Magnusson et al., arXiv:2312.10523, v1 2023-12)는 이를 fit 문제로 제시한다. Held-out set 하나의 perplexity는 가장 많은 sampled token을 내는 domain이 지배하는 micro-average이므로, model은 aggregate에서는 개선되면서 특정 domain에서는 악화할 수 있다.

**수식.** Domain set D에 대한 Paloma의 macro average는 다음과 같다.

    macro-ppl(D) = |D|⁻¹ · Σ_{d ∈ D} perplexity(d)

여기서 `perplexity(d) = exp(−ℓ_d / T_d)`이고, `ℓ_d`는 domain d document에 대한 summed log-likelihood, `T_d`는 domain d의 token count다(§3, §4.1). 비교하는 model의 vocabulary가 다르면 Paloma는 대신 byte per bit를 쓴다. `BPB = −ℓ / (B · ln 2)`이며 B는 UTF-8 byte 수다(부록 B).

**숫자 예시.** Domain 두 개가 각각 100 token이라고 하자. Model A는 domain 1에서 summed NLL 250 nat, domain 2에서도 250 nat이므로 각각 perplexity 12.18, macro average 12.18이다. Model B는 domain 1에서 150 nat, domain 2에서 400 nat이므로 perplexity 4.48과 54.60이다. Pooled token 200개에 대한 micro average는 `exp(550/200) = 15.64`, macro average는 `(4.48 + 54.60)/2 = 29.54`다. Token을 pool하면 B는 A보다 28% 나빠 보이지만 macro average에서는 142% 나쁘다. Domain 2가 유지하려는 capability라면 후자가 중요한 진술이다.

**근거.** Paloma는 pre-training corpus만 달리하고 architecture, token budget, tokenizer, decontamination, data order는 고정한 1B baseline 6개를 훈련한다(§4.1, 부록 G: 1B, sequence length 2048, LionW, peak LR 2.0e-4, 70k step까지 cosine, 35k step ≈ 150B token에서 정지). 보고 결과는 다음과 같다.

- C4-only baseline은 RedPajama arXiv domain에서 perplexity 391,171, Dolma peS2o에서 14에 도달한다(§2, §4.1).
- 약 20B-token checkpoint와 약 150B-token checkpoint 사이에 C4 및 mC4-en baseline은 각각 domain 65개와 43개에서 **악화**한다. 이 두 baseline 밖에서는 model-domain pair 6개만 악화한다(부록 D.1.1). 같은 interval에서 이 run의 aggregate loss curve 하나는 monotone decreasing이다.
- Perplexity는 downstream task와 같은 순서로 model 순위를 매기지 않는다. Source perplexity 순위와 task accuracy 순위의 Spearman ρ는 6개 baseline에 걸쳐 −0.77(c4-en vs HellaSwag)와 0.94(RedPajama vs HellaSwag)를 포함한다(부록 A, Table 3). **결과(단일 연구).**

**Source가 밝힌 조건과 한계.** Paloma는 English와 code만 포괄하며, code source는 decontaminate하지 않았고, fringe-source perplexity는 document length를 추적한다. 또한 evaluation은 concatenation이 아니라 document별로 score해야 한다. Pythia 1.4B를 2B token에서 evaluation token 4M개로 평가하면 concatenated는 `92.23 ± 17.33`, separate는 `42.57 ± 0.29`를 낸다(부록 H, Table 17). 이 variance 차이는 측정하려는 effect를 압도할 만큼 크다.

**실제 training loop에서의 모습.** Olmo 3 7B script는 자체 multi-domain dataset을 가진 held-out-loss callback, 즉 `LMEvaluatorCallbackConfig(eval_dataset=... DataMix.v3_small_ppl_validation ..., eval_interval=10_000)`을 붙인다([[olmo-core-olmo3-configs]]). 따라서 held-out loss는 file 하나가 아니라 이름 있는 validation mixture에 대해 step 10,000마다 측정한다.

**범용 모델에 대한 함의.** Domain별 loss는 “model이 개선된다”와 “model이 mixture를 지배하는 domain에서 개선된다”를 구분하는 가장 저렴한 in-loop signal이다. Task format이나 emergent ability가 필요 없으므로 어떤 benchmark보다 먼저 narrowing을 포착한다.

---

## §3 In-loop measurement II: 고정 format의 downstream evaluation

**정의.** In-loop downstream evaluation은 미리 고정한 formatting 및 scoring procedure 아래 training 중 intermediate checkpoint를 task benchmark로 score한다.

**측정 가능한 문제.** [[olmes]] (Gu et al., arXiv:2406.08446, v1 2024-06)는 같은 dataset의 같은 model도 setup에 따라 서로 다른 published number를 낸다고 문서화한다. Table 1은 reference 6개에서 Llama2-7B의 ARC-Challenge score가 43.2~53.7이고, Llama3-8B는 Hugging Face Open LLM Leaderboard에서 60.2, Llama 3 model card에서 78.6이라고 목록으로 든다. 차이는 model quality가 아니라 formulation, shot count, normalization이다.

**메커니즘, 단계별.**

1. **Formulation**을 고른다. MCF(multiple-choice formulation)에서는 option이 letter label과 함께 prompt에 나타나며 model은 label token으로 score한다. CF(cloze formulation)에서는 각 answer string을 자체 probability로 따로 score한다.
2. CF의 **normalization**을 고른다. `none` = `ln P(a|q)`, `character` = `ln P(a|q) / characters(a)`, `pmi` = `ln[P(a|q) / P(a|u)]`이며 uninformative prompt `u = "Answer:"`를 쓴다.
3. Prompt, split, sample(split이 1,500 초과일 때 `Random(1234).sample(...)`로 instance 1,000개), 고정 curated 5-shot example을 고정한다.
4. MCF와 CF 중 더 좋은 것을 보고한다(§3.4).

**Formulation이 release 때만이 아니라 training 중에도 중요한 이유.** [[olmes]] 그림 1은 training 전반에서 OLMo-7B-0424를 MMLU validation set의 두 format으로 평가한다. MCF는 training token 약 400B까지 random 근처에 머물다가 더 강한 signal이 되고, CF는 초반부터 signal을 주다가 후반에 plateau한다. MCF만 instrument한 run은 첫 400B token 동안 flat line을 본다. ARC-Challenge에서 base model 15개 전반을 보면 가장 약한 8개는 MCF에서는 random 근처지만 CF에서는 random 이상이다(§3.4, 그림 2). **결과(단일 연구)**이며 effect는 benchmark가 아니라 model의 property다.

**Production에서 선택을 의도적으로 한다는 근거.** Olmo 3 7B downstream callback은 step 10,000마다 task 26개를 실행한다([[olmo-core-olmo3-configs]]). ARC-Challenge, ARC-Easy, MMLU split 4개의 OLMES task 12개를 `*_bpb_5shot` 및 `*_mc_5shot_fast` pair variant로, MC counterpart 없이 `hellaswag_bpb_5shot`을 실행한다. `basic_skills_*_rc_5shot` task 6개, bits per byte로 score하는 generative task 6개(`codex_humaneval_gold_bpb_3shot`, `codex_mbpp_gold_bpb_3shot`, `minerva_math_500_gold_bpb_0shot`, `mt_mbpp_{cpp,java,rust}_gold_bpb_3shot`), 그리고 file에 “Sanity check for MCQA ability”라고 comment한 `copycolors_10way_fast`도 실행한다. Pair BPB 및 MC variant는 OLMES의 “evaluate with both and use the better one”을 side by side로 log한 metric 두 개로 구현한다.

**Cadence는 cost decision이고 framework마다 cost가 다르다.** OLMo-core는 process 안에서 step 10,000마다 evaluation하고 1,000마다 checkpoint를 쓴다([[olmo-core-olmo3-configs]]). SmolLM3는 2,000 step마다 checkpoint를 쓰며 written checkpoint를 읽는 separate Slurm job으로 evaluation을 launch한다. Stage 1-2의 `eval_interval: 4000`, stage 3의 6000, long-context stage의 `train_steps: 20000`에 대한 400000은 evaluation을 disable한다([[smollm3-training-configs]]). Asynchronous design은 live model 대신 checkpoint를 측정하는 대가로 evaluation cost를 training throughput에서 제거한다.

**범용 모델에 대한 함의.** Task list가 해당 run에서 “generally capable”의 operational definition이다. Olmo 3의 list는 multiple-choice STEM 및 non-STEM, generative QA, math, code, code infilling을 포괄하므로 capability 하나를 target으로 한 data change가 다른 capability에 주는 cost를 드러낸다([[olmo-3]] §3.3.1).

---

## §4 Signal, noise, ablation 규모에서 무엇을 결정할 수 있는 benchmark인가

**정의.** Benchmark와 model scale에 대해 **noise**는 run 하나의 adjacent checkpoint 사이 score variability이고, **signal**은 비슷한 compute로 훈련한 model population의 score spread다. 둘의 비가 signal-to-noise ratio, SNR이다.

**측정 가능한 문제.** 이 stage의 decision은 small run으로 내린다. Candidate recipe 두 개의 score 차이가 같은 run의 consecutive checkpoint 두 개의 score 차이보다 작으면, 비교는 정보를 담지 않는다.

**수식**([[signal-and-noise-eval]], Heineman et al., arXiv:2508.13144, v1 2025-08, §3.1–3.2, 식 2)은 다음과 같다.

    Rel. Std.(m) = sqrt( (1/(n−1)) · Σ_i (m_i − m̄)² ) / m̄        (noise)
    Rel. Dispersion(M) = max_{j,k} |m_j − m_k| / m̄               (signal)
    SNR = Rel. Dispersion(final checkpoints of a model population) / Rel. Std.(final n checkpoints of one run)

여기서 `m_i`는 run 하나의 final n checkpoint 중 i번째 score, `m̄`는 그 mean, `m_j`, `m_k`는 population M의 model 두 개가 가진 final-checkpoint score다.

**숫자 예시.** Run 하나의 final checkpoint 5개 score가 40, 42, 41, 43, 39라고 하자. Mean `m̄ = 41`, sample standard deviation은 `sqrt((1+1+0+4+4)/4) = sqrt(2.5) = 1.58`이므로 `Rel. Std. = 1.58/41 = 0.0386`이다. 비슷한 compute로 훈련한 model 4개 population의 score가 38, 41, 44, 45이고 mean 42라면 `Rel. Dispersion = (45 − 38)/42 = 0.167`이다. `SNR = 0.167/0.0386 = 4.3`이다. 이제 두 번째 benchmark가 model 사이에 같은 spread를 갖지만 final-checkpoint score가 36, 44, 40, 46, 39라고 하자. Standard deviation은 4.0, `Rel. Std. = 0.0976`, `SNR = 1.7`이다. 두 benchmark 모두 fixed checkpoint에서는 같은 model들을 똑같이 잘 분리하지만, checkpoint를 임의로 고르면 첫 번째만 신뢰할 수 있다.

**근거.**

- Benchmark 30개 전반에서 small-scale SNR은 decision accuracy, 즉 small/large scale에서 같은 순위인 model pair 비율을 `R = 0.791`, `R² = 0.626`으로 예측한다. Signal만 또는 noise만으로는 예측하지 못한다(§4.1, 그림 2). Setup은 1B ranking을 예측하는 60M–750M [[datadecide]] model이고, noise는 final checkpoint 5개에서 얻는다.
- Target scale의 noise는 scaling-law prediction error와 상관한다. Task 30개에서 `R = 0.653`, `R² = 0.426`이며, 13B target의 noise는 1,000 step 간격 final checkpoint 30개에서 얻는다(§4.2).
- Metric을 accuracy에서 bits per byte로 바꾸면 task 30개 average SNR은 10.0에서 31.5, decision accuracy는 77.0%에서 83.7%로 오른다. Task별로 Minerva MATH는 51.0 → 90.0, HellaSwag는 74.3 → 95.3이다(§5.3, 그림 6). 이것이 §3의 Olmo 3 in-loop task list가 BPB variant로 쓰인 이유다.
- SNR은 scale 간 transfer하지 않는다. ARC-Easy는 1.5B-4T에서 7.89였다가 32B-6T에서 5.10으로, SocialIQA는 8.73에서 1.95로 낮아지는 반면, Minerva MATH 500은 1.5B-4T의 0.91에서 7B-4T의 4.45로 오른다(부록 B.3, Table 4). **결과(단일 연구).**

**Small run이 정말 decision을 내리는지에 대한 보완 결과.** [[datadecide]] (Magnusson et al., arXiv:2504.11393, v1 2025-04)는 corpus 25개 × size 14개 × seed 3개(1,050 model, 최대 100B token)를 훈련하고, small size의 ranking이 1B winner를 얼마나 자주 예측하는지 측정한다. 150M의 ranking은 pairwise comparison 약 80%를 맞히며, scaling-law baseline 8개 중 어느 것도 single-scale ranking의 compute-to-decision-accuracy frontier를 이기지 못한다(§3.1–3.2). Task별 그림은 다르다. ARC-Easy는 5 order of magnitude 적은 compute로 예측 가능하지만, BoolQ는 target run의 intermediate checkpoint가 있어야 trivial decision accuracy를 넘고, SocialIQA는 시험한 모든 scale에서 unreliable하다(§3.1, 그림 2). Accuracy 대신 continuous likelihood metric을 쓰면 MBPP와 HumanEval은 trivial에서 decision accuracy 80%로 이동한다(§3.4, 그림 6).

**조건과 한계.** DataDecide는 token-to-parameter ratio를 100으로 고정하고 multiple-choice cloze task만 평가한다(§2.4, §5). Signal and Noise는 training-time noise만 연구하며 evaluation configuration의 noise는 future work로 든다(§6).

**Frontier run이 이를 적용하는 방식.** [[olmo-3]]은 OLMo 2 13B training의 final checkpoint 50개와 약 `4·10²³` FLOPs의 external base model 10개에 같은 SNR procedure를 실행해 OlmoBaseEval을 만들었다. 그런 뒤 너무 noisy한 benchmark, 즉 model이 majority/minority class prediction 사이에서 대체로 oscillate한다고 한 BoolQ 같은 binary benchmark를 제거하고, CruxEval은 macro-average 밖의 separate number로 옮겼다(§3.3.3). Suite는 task 43개로 “over 4 times more benchmarks than OLMo 2”다. 1B parameter 미만 decision용 BPB Base Easy suite, final run용 Base Main suite, development suite overfitting을 막기 위한 benchmark 4개(MMLU Pro, DeepMind Math, LBPP, BBH)의 Held-out set으로 나뉜다(§3.3.4).

---

## §5 Checkpoint selection: run 내 averaging과 run 간 merging

**정의.** Checkpoint averaging은 release할 parameter θ를 여러 checkpoint의 arithmetic mean으로 대체한다. `θ_S = (1/|S|) · Σ_{i ∈ S} θ_i`다. θ_i가 trajectory 하나에서 오면 보통 averaging 또는 merging, shared initialization을 가진 separate run에서 오면 souping이라고 한다.

**측정 가능한 문제.** Development score가 가장 좋은 checkpoint 하나를 고르는 것은 §4에서 수량화한 capability와 noise를 동시에 선택하는 것이다. ARC Challenge에서 1B model의 final-n noise는 1.7 percentage point에 걸친다([[signal-and-noise-eval]] §3.1). 이는 많은 recipe effect의 크기다.

**숫자 예시.** Parameter 두 개와 checkpoint 두 개 `θ_1 = (1.0, −0.5)`, `θ_2 = (1.4, −0.1)`의 average는 `(1.2, −0.3)`다. Averaging은 output이 아니라 weight에 작용하므로, average model은 forward pass 하나, checkpoint 두 개의 output ensemble은 forward pass 두 개가 든다([[model-soups]] §2). 또한 average는 두 function의 average와 다른 function이며, accuracy가 둘 사이에 놓인다는 보장은 없다. 아래 empirical claim은 average가 종종 둘 모두보다 높다는 것이다.

**근거. 가장 직접적인 language-model setting부터 순서대로 제시한다.**

1. **Run의 마지막 checkpoint를 average하면 decision이 좋아진다.** [[signal-and-noise-eval]] §5.2, Table 1에서 60M-5xC → 1B-5xC의 task 30개 decision accuracy는 68.9%(final checkpoint)에서 71.3%(small 및 target model 모두 final checkpoint average)로 오른다. Task 두 개를 제외한 모두에서 향상한다. 13B scaling-law prediction error는 absolute percentage point 1.03에서 0.86으로 낮아지고 task 30개 중 20개에서 향상한다. Early stopping에서는 exponential moving average가 거의 모든 training step에서 single checkpoint보다 더 높은 decision accuracy를 낸다(그림 5).
2. **Data order만 다른 run을 soup한다.** [[olmo-2]] §4.5, Table 14, 7B model, 50B-token mid-training stage, candidate mix 6개, 각각 permutation 3개는 다음과 같다.

   | Mix | best single OLMES(MCF) | 3× soup | best single GSM* | 3× soup GSM* |
   |---|---:|---:|---:|---:|
   | A | 75.6 | 77.0 | 71.0 | 74.0 |
   | B | 75.3 | 77.3 | 73.0 | 77.0 |
   | C | 76.3 | 76.8 | 66.0 | 66.0 |
   | D | 77.5 | 77.8 | 59.5 | 60.0 |
   | E | 73.4 | 75.3 | 60.5 | 43.0 |
   | F | 77.1 | 77.9 | 73.5 | 74.5 |

   Soup은 mix 6개 모두에서 OLMES의 best single run과 같거나 더 낫다. 그러나 OLMES average에는 포함되지 않고 math development set으로 쓴 GSM8K question 200개 sample인 GSM*에서는(§4.5, §A.1) mix E가 17.5점 떨어진다. Aggregate claim과 per-task claim은 다르고 source는 첫 claim만 한다. **결과(단일 연구).** OLMo 2 7B는 50B Dolmino sample의 checkpoint 3개 average로 ship한다. 13B와 32B는 checkpoint 4개 average(100B sample의 3개 및 300B sample의 1개)로 ship하며, report는 이것이 100B run 3개만 average한 것보다 empirical하게 낫다고 말한다.
3. **Pre-training 내부의 대규모 merging.** [[merging-in-pretraining]] (ByteDance Seed, arXiv:2505.12082, 2025-05)는 dense 411M–70B model 및 최대 20B/200B MoE model에서 WSD schedule의 trajectory 하나 checkpoint를 merge한다. Constant-LR phase 중 merging은 Seed-MoE-1.3B/13B HumanEval을 31.1에서 36.6으로, Seed-MoE-10B/100B를 54.3에서 61.6으로 올렸다(§4.1). Cosine-decay phase 초기에 merge하면 annealing 종료와 comparable한 결과가 나왔고, token 1.4T에서 run을 fork한 결과 merged constant-LR checkpoint는 annealed model과 match했다(§4.1, 그림 2-3). Paper가 준 setting 두 개는 다음과 같다. Merged checkpoint 사이 interval V는 model size와 함께 증가한다(0.7B/7B에서 약 4B token, 1.3B/13B에서 8B, 10B/100B에서 80B). Training 완료 후에는 더 많은 checkpoint를 merge할수록 도움이 되며 `N = 3`은 `N = 15`보다 약 1점 낮다(§4.3).
4. **Production practice.** [[llama-3]] §3.4.3은 “During pre-training on the final 40M tokens, we linearly annealed the learning rate to 0 … Finally, we compute the average of model checkpoints (Polyak (1991) averaging) during annealing to produce the final pre-trained model.”이라고 한다. §4.1.5는 post-training으로 이를 확장한다. “we average models obtained from experiments using various versions of data or hyperparameters at each RM, SFT, or DPO stage.” Olmo 3 32B README도 stage 두 개에서 같은 일을 말한다. “we soup (with simple averaging of parameters) the outputs of two separate midtraining runs and we soup the final three checkpoints produced by the long-context stage”([[olmo-core-olmo3-configs]]). **Practice 수준에서는 organization 간 replicated됐고, controlled number는 OLMo 2 Table 14와 아래 두 논문이다.**

**방법의 출처와 명시된 조건.** [[model-soups]] (Wortsman et al., arXiv:2203.05482, ICML 2022)는 image/text classification의 controlled study다. Random hyperparameter search의 model 72개를 사용한 CLIP ViT-B/32(Table 3)에서 best individual model은 ImageNet 80.38 / distribution shift 5개 average 47.83, uniform soup은 79.97 / 51.45, greedy soup은 81.03 / 50.75다. Uniform soup은 target distribution에서 best model보다 0.41점 낮지만 shift에서는 3.62점 높다. Greedy procedure는 held-out validation accuracy로 sort하고 average가 나빠지지 않을 때만 ingredient를 더한다. 일부 ingredient가 나쁘면 uniform soup이 실패하기 때문에 존재하며, paper는 이를 high learning rate로 fine-tune한 model 사이 error barrier 탓으로 본다(§3.3.1). 명시된 조건은 두 가지다. 모든 ingredient는 “fine-tuned independently from a shared initialization θ0 with different hyperparameter configurations”한 model이어야 한다(§3.2, 그림 3 caption). 따라서 shared initialization을 가진 model에서만 method가 정의된다. 또한 §5는 soup이 calibration을 크게 개선하지 못하는 failure를 limitation으로 든다.

**Pre-fine-tuning model과 interpolate하는 특수 case.** [[wise-ft]] (Wortsman et al., arXiv:2109.01903, CVPR 2022)는 fine-tuning 전 model(θ0)과 후 model(θ1) 사이를 interpolate한다. `θ_α = (1 − α)·θ0 + α·θ1`다. CLIP ViT-L/14@336px(Table 1)에서 fine-tuning은 ImageNet을 76.6에서 86.2로 올리고 shift 5개 average는 73.4에서 68.6으로 낮춘다. `α = 0.5`는 86.8과 76.9로, fine-tuned model보다 둘 다 높다. 저자들은 target shift를 모르면 `α = 0.5`를 권한다(§4). 두 논문은 image classification이고 WiSE-FT §7은 language-model experiment가 없음을 limitation으로 든다. 따라서 language model에서는 transfer된 result가 아니라 hypothesis가 붙은 mechanism이다. Course가 두 결과에 읽어 넣는 mechanism, 즉 shared initialization model 간 linear mode connectivity는 [[olmo-2]]와 [[llama-3]]가 의존하는 것이다(**해석**).

[`figures/checkpoint-selection.html`](figures/checkpoint-selection.html)은 위 OLMo 2 Table 14 행을 그려 aggregate gain과 mix-E GSM* drop을 같은 chart에서 읽게 한다.

**범용 모델에 대한 함의.** Averaging은 selected metric보다 breadth를 개선한다는 evidence가 있는 이 장의 유일한 selection procedure다. [[model-soups]]의 soup gain은 target distribution보다 distribution shift에서 더 크고, [[wise-ft]] result는 전적으로 breadth result다. 이에 맞는 risk는 average가 ingredient 하나가 가진 narrow capability를 잃을 수 있다는 것이며, mix-E GSM* 행이 이를 보인다.

---

## §6 Fine-tuning에서의 early stopping과 checkpoint selection

**정의.** Early stopping은 monitor하는 metric이 더 이상 개선되지 않으면 training을 멈추고 best value를 달성한 checkpoint를 반환한다. 고전적 procedure([[early-stopping-and-checkpointing]])는 `eval_every` step마다 held-out set을 평가하고, validation loss가 `delta`보다 크게 개선될 때마다 checkpoint를 저장하며, 그렇지 않으면 counter를 증가시키고, counter가 `patience`에 도달하면 중단한다(보통 evaluation 3~5회). 이를 정의하는 parameter 세 개는 metric, `delta`, `patience`다. 이 절의 나머지는 첫 번째에 관한 것이다.

**범용 모델의 측정 가능한 문제.** Fine-tuning mixture에 포함되지 않은 capability가 저하되는 동안 target validation loss와 target-benchmark accuracy는 계속 개선될 수 있다. Target metric만 읽는 stopping rule은 이를 볼 수 없다.

**Stopping point가 중요하다는 근거.** [[tulu-3]]은 LR 5e-6에서 sum loss로 Tülu 2 SFT mixture에 Llama 3.0을 fine-tuning하며 epoch를 sweep했다. “Surprisingly, we additionally found that training for longer did not yield further improvements, and so used 2 epochs for training”이라고 보고한다(§4.3, 그림 6은 epoch 2~7의 average performance를 그림). Final Tülu 3 SFT run은 8B에서 LR 5e-6, 70B에서 2e-6으로 epoch 2회를 쓴다. **결과(단일 연구)**, mixture 하나와 base model 하나의 결과다.

**Metric이 stopping point보다 중요하다는 근거.** Tülu 3는 evaluation suite를 decision에 쓴 development set과 “we did not examine … when developing our models”인 unseen set으로 나눴다(§2.1, §7.2–7.4). Unseen set은 각 development benchmark를 같은 skill을 시험하는 benchmark와 pair로 둔다(§7.4.1, Table 31).

| Skill(dev → unseen) | 8B SFT dev / unseen | 8B DPO dev / unseen | 8B Final dev / unseen |
|---|---:|---:|---:|
| Average | 64.9 / 29.9 | 68.3 / 31.9 | 68.8 / 32.4 |
| Reasoning(BBH → AGIEval) | 67.9 / 56.2 | 65.8 / 61.8 | 66.0 / 59.3 |
| Coding(HumanEval → BigCodeBench) | 86.2 / 11.5 | 83.9 / 9.5 | 83.9 / 7.4 |
| Inst. following(IFEval → IFEval-OOD) | 72.8 / 17.6 | 81.1 / 23.9 | 82.4 / 24.3 |

두 가지 해석이 있다. 첫째, pipeline은 average에서 generalize한다. Final checkpoint는 두 suite 모두에서 best average다. 둘째, coding column은 반대 방향으로 움직인다. Development coding은 같은 stage 전반에서 약 84~86에 머무르지만 unseen coding은 11.5에서 7.4로 낮아진다. IFEval-OOD는 “test precise instruction following abilities of LLMs and whether they are able to follow constraints that go beyond the 25 constraints included in IFEval”을 위해 만들었고, category 6개에 걸쳐 constraint 52개로 구성된다(§7.3.1). Tülu 3의 해석은 “those models that do well on IFEval are likely overfitting to the specific set of constraints included in the dataset”이다(§7.3.1, §7.4.2). Table 32의 SFT data ablation도 같은 split을 보인다. Data choice는 average에서는 generalize하지만, 저자들은 precise instruction following과 어느 정도 knowledge recall/reasoning에서 development evaluation에 overfit한다고 말한다.

**Tülu 3가 RL stage에서 실제로 실행하는 selection rule.** “For our final runs, we examine model checkpoints every 40-100 steps and choose the best checkpoint on our development evaluation set”(§6, RLVR implementation detail, item 3)이다. Asymmetry에 주의해야 한다. Selection은 development suite를 쓰고 unseen suite는 이후에 selection을 audit하도록 남긴다. 이것이 unseen suite가 의미를 유지하는 유일한 arrangement다.

**조건과 한계.** Tülu 3 epoch result는 scale 하나의 mixture 하나에 관한 것이다. Unseen-suite comparison은 비교한 다른 model이 그 benchmark로 훈련했을 가능성을 배제할 수 없다(§7.4.2, Table 33 caption).

**범용 모델에 대한 함의.** 하나는 development 중 전혀 consult하지 않는 disjoint suite 두 개가 fine-tuning 시점 generalization measurement의 operational definition이다. Stopping과 reporting에 suite 하나를 함께 쓰면, 그 suite에 stopping rule이 얼마나 잘 최적화했는지를 측정할 뿐이다.

---

## §7 Resume 뒤 domain별 및 shard별 loss tracking

**정의.** Shard별 loss tracking은 loader가 제공하는 shard identity와 함께, data source별로 나눈 training loss를 log한다.

**측정 가능한 문제.** §1은 error를 내지 않으면서 다음 step이 보는 것을 바꾸는 resume path 세 개를 제시한다. Loader state를 전혀 복원하지 않는 경우(OLMo-core의 `load_trainer_state=False` 또는 weight/optimizer state만 저장하는 training loop), `ignore_fingerprint_mismatch=True`로 suppress한 fingerprint mismatch, rank별 RNG를 복원하지 않는 world-size change다. 첫 두 경우 aggregate loss curve는 continuous하게 유지된다. 같은 mixture data에 대한 model loss는 order와 무관하게 같기 때문이다.

**탐지 방법.** 이미 본 data는 같은 source의 unseen data보다 loss가 낮다. 따라서 order를 다시 시작하는 loader는 shuffled order 초기에 나타나는 source의 source별 loss에 downward step을 만들고, 나머지 source에는 변화를 만들지 않는다. 다음 quantity 두 개로 이를 볼 수 있다.

1. 영향을 받은 source에만 step을 보이는 source별 training loss.
2. 학습하지 않은 data에서 평가하는 §2의 multi-domain held-out loss. 이는 **움직이지 않는다**. 둘의 divergence가 signature다. Training loss는 내려가고 held-out loss는 평평하다.

**숫자 예시.** Mixture가 web 50%, code 30%, math 20%이고, loader reset이 이미 제공한 web shard를 다시 제공한다고 하자. 해당 document를 두 번째 pass할 때 생기는 loss reduction을 δ라고 하자. δ는 측정된 constant가 아니므로 example의 assumption으로 취급해 `δ = 0.05` nat로 둔다. 그러면 web curve는 0.05만큼 내려가고 code와 math는 움직이지 않으며, mixture-weighted total은 `0.5 × 0.05 = 0.025` nat만 움직인다. 이는 pre-training loss curve의 보통 step-to-step variation보다 작다. True δ 값이 무엇이든 source별 split이 이 크기의 change를 읽을 수 있게 한다.

**Code에서 가능한 직접 check.** [[olmo-core-olmo3-configs]]의 fingerprint comparison은 `ignore_fingerprint_mismatch`를 default로 두는 한, 이 failure의 탐지되지 않는 version을 exception으로 바꾼다. 이는 세 check 중 가장 저렴하며, 나중 diagnosis가 아니라 load time에 둬야 한다.

**범용 모델에 대한 함의.** Data subset을 반복하면 effective mixture가 바뀌고 mixture가 breadth를 결정한다. Source별 loss panel은 §2의 domain별 held-out loss와 같은 instrument를 held-out data가 아니라 training data에 적용한 것이며, change를 specific source에 귀속할 수 있는 유일한 방법이다.

---

## 레시피

| 모델(exact release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Olmo 3 Base 7B | 7B | pretrain-stable | checkpoint save interval | 1000 step(`save_interval=1000`, `save_async=False`) | OLMo-core@66f768b `src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py` | 2026-09-17 검증됨 | ablation 보고 없음 |
| Olmo 3 Base 7B | 7B | pretrain-stable | in-loop held-out-loss interval; dataset | 10,000 step; `DataMix.v3_small_ppl_validation` | same file, `LMEvaluatorCallbackConfig` | 2026-09-17 검증됨 | ablation 보고 없음 |
| Olmo 3 Base 7B | 7B | pretrain-stable | in-loop downstream interval; task set | 10,000 step; task 26개(BPB/MC 5-shot OLMES subset, BasicSkills 6개, generative BPB 6개, `copycolors_10way_fast`) | same file, `DownstreamEvaluatorCallbackConfig` | 2026-09-17 검증됨 | [[olmo-3]] §3.3.3의 SNR selection procedure |
| Olmo 3 Base 7B | 7B | pretrain-stable | metrics collection interval | 10 step(`metrics_collect_interval=10`) | same file, `TrainerConfig` | 2026-09-17 검증됨 | ablation 보고 없음 |
| SmolLM3-3B | 3B | pretrain-stable | checkpoint interval; resume switch | 2000 step; `load_optimizer: true`, `load_lr_scheduler: true` | huggingface/smollm SmolLM3 `stage1_8T.yaml` L1–9 | 2026-09-17 검증됨 | ablation 보고 없음 |
| SmolLM3-3B | 3B | pretrain-stable / decay | downstream eval interval | 4000 step(stage 1–2); 6000(stage 3); long-context stage에서 400000 vs 20000 train_steps, 즉 disabled | `stage1_8T.yaml`, `stage2_8T_9T.yaml`, `stage3_9T_11T.yaml`, `long_context_4k_to_32k.yaml`, `lighteval.eval_interval` | 2026-09-17 검증됨 | ablation 보고 없음 |
| Llama 3 405B | 405B | pretrain-decay/anneal | final-checkpoint rule | final 40M token의 annealing 중 checkpoint average(Polyak averaging), 128K context에서 LR linear to 0 | arXiv:2407.21783 §3.4.3 | 2026-09-17 검증됨 | ablation 보고 없음 |
| Llama 3(all size) | 8B–405B | reward-model / SFT / preference | model averaging | 각 RM/SFT/DPO stage에서 data version 또는 hyperparameter가 다른 experiment model의 average | arXiv:2407.21783 §4.1.5 | 2026-09-17 검증됨 | ablation 보고 없음; averaging weight 미보고 |
| OLMo 2 7B | 7B | mid-train | final-checkpoint rule | 50B Dolmino Mix 1124 sample의 permutation 3개에서 훈련한 checkpoint 3개 average | OLMo 2 report §4.5 | 2026-09-17 검증됨 | §4.5 Table 14: 6 mix 모두에서 soup ≥ best single(OLMES MCF) |
| OLMo 2 13B, 32B | 13B, 32B | mid-train | final-checkpoint rule | checkpoint 4개 average: 100B sample 3개, 300B sample 1개 | OLMo 2 report §4.5 | 2026-09-17 검증됨 | 100B run 3개만 average한 것보다 empirical하게 낫다고 §4.5가 말함 |
| Olmo 3 Base 32B | 32B | mid-train; long-context | merge rule | midtraining run 두 개 simple parameter average; final long-context checkpoint 3개 average | OLMo-core@66f768b `src/scripts/official/OLMo3/README.md` | 2026-09-17 검증됨 | ablation 보고 없음 |
| Seed-MoE-1.3B/13B(PMA) | 1.3B active / 13B total | pretrain-stable | merge interval V; checkpoint 수 N | V ≈ 8B token; `N = 10` | arXiv:2505.12082 §4.3 | 2026-09-17 검증됨 | §4.3 그림 5: 204B token에서 V = 16B/32B는 baseline 미만; N = 3은 N = 15보다 약 1점 낮음 |
| Tülu 3 SFT(Tülu 2 mixture의 Llama 3.0) | 8B | SFT | epoch; loss; LR | epoch 2; sum loss; 5.00e-6 | Tülu 3 report §4.3, 그림 5–6 | 2026-09-17 검증됨 | 그림 6: epoch 2–7 average performance에서 2가 최상; “training for longer did not yield further improvements” |
| Tülu 3 8B, 70B | 8B, 70B | SFT | epoch; LR | epoch 2; 5e-6(8B), 2e-6(70B) | Tülu 3 report §4.3 | 2026-09-17 검증됨 | 같은 sweep |
| Tülu 3(RLVR final run) | 8B, 70B | RL | checkpoint selection rule | 40–100 step마다 checkpoint inspect; development evaluation set에서 best 선택 | Tülu 3 report §6, RLVR implementation detail, item 3 | 2026-09-17 검증됨 | 해당 없음 |
| Signal and Noise(DataDecide small model) | 60M–750M | eval-gate | noise window | final checkpoint 5개 | arXiv:2508.13144 §4.1; App. A.3.2 | 2026-09-17 검증됨 | Table 2: OLMo 2 7B의 거의 모든 benchmark에서 n = 5는 ±1σ 내 |
| Signal and Noise(OLMo 2 13B target) | 13B | eval-gate | noise window | 1,000 step 간격 final checkpoint 30개 | arXiv:2508.13144 §4.2, footnote 4 | 2026-09-17 검증됨 | sample-size/compute trade-off로 명시 |

**작은 범용 run의 시작점.** Single node 또는 small cluster에서 parameter 약 10B 미만 run이라면, 위 verified row는 optimizer step 1,000–2,000마다 checkpoint를 쓰는 방식을 뒷받침한다(Olmo 3 7B와 SmolLM3-3B 모두 이 range). Multi-domain held-out loss와 downstream suite는 checkpoint보다 더 성긴 interval에서 평가한다(Olmo 3 7B는 process 안에서 10,000 step, SmolLM3는 asynchronous 4,000 step). Small-scale downstream task는 accuracy 대신 bits per byte로 score하고([[signal-and-noise-eval]] §5.3, task 30개 SNR 10.0 → 31.5), single best 대신 final checkpoint average를 release한다([[signal-and-noise-eval]] §5.2). 이 숫자에 붙은 조건은 다음과 같다. Olmo 3 7B는 약 4M-token global batch(8,192 token × sequence 512개)에서 5.93T token이고, 7B run GPU 수는 보고하지 않았다. [[olmo-3]] §2.4의 runtime breakdown은 32B를 다룬다. SmolLM3는 sequence length 4,096에서 11T token이며, averaging result는 60M–1B의 DataDecide model에서 측정했다.

---

## 일반화 관점

**(a) 폭을 넓히는 것.**

- Single checkpoint를 고르는 대신 run의 마지막 checkpoint를 average: task 30개 decision accuracy +2.4점, 13B absolute scaling-law error 1.03 → 0.86([[signal-and-noise-eval]] §5.2, Table 1).
- Data order만 다른 run soup: 7B에서 시험한 mid-training mix 6개 모두 OLMES aggregate가 best single과 같거나 더 높음([[olmo-2]] §4.5, Table 14).
- Constant learning rate 아래 pre-training 내 merge: Seed-MoE-1.3B/13B HumanEval 31.1 → 36.6, 10B/100B 54.3 → 61.6([[merging-in-pretraining]] §4.1).
- 측정된 setting에서 fine-tuning 전 model과 interpolate: `α = 0.5`에서 fine-tuned model보다 shift accuracy +8.3점이며 target accuracy도 상승([[wise-ft]] Table 1, image classification).
- Aggregate가 아니라 domain별로 측정하여 run 종료 전 domain cost가 보이게 함([[paloma]] §4.1).

**(b) narrowing/forgetting의 원인과 숨는 방식.**

- Target metric이 계속 개선되는 point를 지나 training: Tülu 3 epoch sweep은 average performance에서 2~7 중 2 epoch가 최고([[tulu-3]] §4.3, 그림 6).
- Development suite에 최적화해 measurement가 멈춤: Tülu 3 8B는 IFEval 82.4, task structure를 유지하고 constraint set을 바꾸는 IFEval-OOD 24.3이며 report는 비교한 모든 model에서 같은 gap을 찾음([[tulu-3]] §7.3.1, §7.4, Tables 31, 33).
- Component가 저하하는데 aggregate에서 선택: Tülu 3 unseen coding은 SFT/DPO/final에서 11.5 → 9.5 → 7.4로 낮지만 development coding은 약 84에 머묾([[tulu-3]] Table 31).
- Ingredient 하나가 narrow capability를 가진 경우 averaging 자체: OLMo 2 mix E는 OLMES +1.9의 같은 3× soup에서 GSM* −17.5([[olmo-2]] Table 14).
- 이미 본 data를 다시 제공하는 resume은 loss curve를 바꾸지 않으며 effective mixture를 바꿈(§7).

**(c) 이 단계의 측정 방법.**

- Decontaminated data의 multi-domain held-out loss를 macro-average하고 document는 따로 score([[paloma]] §3 G1–G5; 부록 H concatenated-input variance는 피해야 할 failure).
- Fixed-format downstream suite. Training 초반 약 400B token 전 MCF가 random 근처이므로 CF와 MCF를 모두 보고([[olmes]] §3.4, 그림 1).
- Small-scale comparison을 신뢰하기 전 final n checkpoint에서 benchmark-level SNR 측정. Small model은 n = 5, metric은 BPB([[signal-and-noise-eval]] §3.1, §5.3).
- Development decision에 전혀 consult하지 않은 held-out suite를 development suite 옆에 보고([[tulu-3]] §7.4; [[olmo-2]] §2.5의 “a standard practice of declaring development vs held-out evaluation tasks”; [[olmo-3]] §3.3.4).
- Per-domain held-out loss 옆에 source별 training loss를 두어 divergence가 capability change가 아닌 data-pipeline change를 식별하게 함(§7).

---

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 확인 |
|---|---|---|
| Data-loader position 없이 resume | Aggregate training loss는 매끄럽게 이어지고 먼저 제공된 source의 source별 loss가 내려감 | Source별 training loss를 log하고 움직이지 않는 multi-domain held-out loss와 비교 |
| Run 재시작을 위해 dataset-fingerprint mismatch suppress | Load 때 error가 없고 pre-crash run과 data order가 다름 | `ignore_fingerprint_mismatch`를 default로 두고 `RuntimeError`를 blocker로 취급([[olmo-core-olmo3-configs]]) |
| 다른 GPU 수에서 resume하고 reproducibility 가정 | “Trainer will not restore rank RNG states …” log, 같은 step original run과 loss 차이 | Resume log를 읽고 run metadata에 world size 기록/비교 |
| Step schedule을 추적하며 resume에서 global batch size 변경 | Token position은 맞지만 step counter가 불일치 | Load 때 `global_step == tokens_processed // global_batch_size` assert 또는 token으로 schedule 추적 |
| Low-SNR benchmark에서 small-scale ablation 판단 | 같은 run adjacent checkpoint 사이에 ranking flip | Final checkpoint 5개에서 Rel. Std.를 계산해 주장 차이와 비교([[signal-and-noise-eval]] §3.1) |
| Pre-training 초반 MCF만 평가 | 수백 billion token 동안 downstream curve가 chance 근처 flat | 같은 task의 CF(or BPB), MCF variant log([[olmes]] 그림 1) |
| Ablation 규모에서 accuracy로 in-loop benchmark score | Task가 trivial decision accuracy여서 ranking signal 없음 | BPB 또는 다른 continuous likelihood metric으로 변경([[signal-and-noise-eval]] §5.3; [[datadecide]] §3.3) |
| Held-out perplexity를 위해 document concatenate | Held-out loss가 test effect를 숨길 만큼 noisy | Document별 score; Pythia 1.4B concatenate `92.23 ± 17.33`, separate `42.57 ± 0.29`([[paloma]] 부록 H) |
| Development suite를 generalization evidence로 보고 | Stage 전반 dev와 unseen score diverge | Disjoint unseen suite를 유지하고 둘 다 보고([[tulu-3]] §7.4) |
| Single best-scoring checkpoint release | Release score가 같은 run의 다음 checkpoint에서 재현되지 않음 | Final checkpoint average와 비교([[signal-and-noise-eval]] §5.2) |
| 서로 다른 initialization 또는 매우 다른 LR의 checkpoint soup | Merged model이 모든 ingredient보다 훨씬 낮음 | Shared initialization 확인, held-out set의 greedy procedure 사용([[model-soups]] §3.3.1, Recipe 1) |

---

## 이해 점검

1. Resume run의 aggregate training loss는 continuous이고 multi-domain held-out loss는 변하지 않지만, resume step에서 source 6개 중 2개의 per-source training loss가 떨어졌다. 무슨 일이 일어났고 aggregate curve 두 개가 왜 이를 보이지 못했는지 설명하라.
2. Paloma C4 baseline은 20B~150B checkpoint 사이 domain 65개에서 악화한다. Micro/macro average 차이로, 같은 data의 pooled held-out loss가 interval 전체에서 개선될 수 있는 이유를 설명하라.
3. Benchmark 하나는 signal과 noise가 높고, 다른 하나는 signal과 noise가 낮다. 150M parameter에서 data mixture 두 개를 고를 때 어느 것이 더 좋은 instrument이며, 어느 quantity가 이를 정하는가?
4. Underlying model과 data가 그대로인데 accuracy에서 BPB로 scoring metric을 바꾸면 benchmark SNR이 왜 오르는가? Signal/noise 정의에서 논증하라.
5. Mid-training run 3개 soup이 OLMES aggregate에서는 best single을 이기지만 GSM*에서 17.5점을 잃는다. 이 pattern을 만드는 mechanism 하나와, 그것이 noise인지 구별하기 위해 무엇을 측정할지를 말하라.
6. Tülu 3는 development suite에서 RLVR checkpoint를 선택하지만 unseen suite를 보고한다. Unseen suite에서 선택하면 그것을 보고할 가치가 있게 하는 property가 왜 사라지는지 설명하라.
7. WiSE-FT result는 image classification에서만 측정했다. SFT language model로 transfer될 것이라 기대하는 argument와, 그 argument가 evidence가 아닌 이유를 말하라.
8. Resume이 parameter, optimizer state, step counter를 복원하지만 run을 GPU 512개에서 256개로 옮긴다. 더 이상 reproducible하지 않은 quantity를 열거하고, 각각 trajectory에 영향을 주는지 또는 reproduce할 능력에만 영향을 주는지 말하라.

---

## 연결

- **이전 — [[ch-05]] Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen.** ch-05는 unit(token/step, sequence/step, world size)을 고정한다. 이 장 §1은 이 unit이 checkpoint 안에서 다시 나타나, 바뀐 batch size/world size가 resume의 의미를 조용히 바꾸는 모습을 보인다.
- **다음 — [[ch-07]] Training Failure Modes: Numerical, Masking, and Capability-Level Failures.** ch-07은 여기서 만든 instrumentation을 쓴다. §2와 §7의 domain/source별 curve가 그 장 capability-level failure detection의 input이다.

---

## 출처

- [[paloma]] — domain별 held-out perplexity, 통제된 1B corpus baseline 6개, macro-average 정의, evaluation guideline 5개, document-separate evaluation requirement.
- [[olmes]] — fixed evaluation format, CF vs MCF와 400B-token MCF acquisition curve, 한 model의 published ARC-Challenge number spread.
- [[signal-and-noise-eval]] — signal/noise/SNR 정의, decision accuracy 예측 SNR, checkpoint averaging/BPB intervention, checkpoint 5개/30개 noise window.
- [[datadecide]] — small-scale ranking이 large-scale winner를 예측하는지, task별 decision accuracy, code/math continuous likelihood metric.
- [[olmo-2]] — mix 6개 souping comparison(Table 14), release된 7B/13B/32B averaging rule, development-versus-held-out evaluation recommendation.
- [[olmo-3]] — OlmoBaseEval: task clustering, Base Easy BPB proxy suite, SNR filtering procedure, benchmark 4개 Held-out set, 7B LR schedule의 늘린 second half.
- [[olmo-core-olmo3-configs]] — trainer/data-loader state dictionary, fingerprint/world-size guard, checkpoint/evaluation interval, README의 32B souping statement 발췌.
- [[smollm3-training-configs]] — checkpoint interval, `load_optimizer` / `load_lr_scheduler` switch, stage별 interval의 asynchronous checkpoint-driven evaluation 발췌.
- [[llama-3]] — annealing 중 checkpoint average(§3.4.3), RM/SFT/DPO stage model average(§4.1.5).
- [[tulu-3]] — epoch sweep/2-epoch SFT setting, development/unseen suite split, Table 31, IFEval vs IFEval-OOD, RLVR checkpoint-selection rule.
- [[merging-in-pretraining]] — stable/decay phase PMA, model size에 따른 merge interval/checkpoint count, broken trajectory 복구 PMA-init 발췌.
- [[model-soups]] — uniform/greedy soup, CLIP ViT-B/32 table, averaging failure 조건 발췌.
- [[wise-ft]] — pre-fine-tuning/fine-tuned model interpolation, `α = 0.5`에서 OOD accuracy 회복 발췌.
- [[early-stopping-and-checkpointing]] — patience 고전 early-stopping loop. 범용 fine-tuning에서 held-out-suite rule이 대체하는 §6 baseline procedure.
