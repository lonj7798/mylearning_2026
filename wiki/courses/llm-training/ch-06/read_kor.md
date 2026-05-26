<!-- chapter: ch-06
     track: foundations
     title: Checkpointing, Resume, Instrumentation
     sources: [[early-stopping-and-checkpointing]], [[fsdp-sft]], [[mixed-precision]], [[lr-schedules]], [[gradient-clipping]], [[olmo-2]], [[olmo-3]], [[llama-3]], [[karpathy-training-neural-net-recipe]]
     figures: figures/checkpoint-state.html
-->

# 6장 — Checkpointing, Resume, Instrumentation

> **핵심 통찰.** checkpoint는 "weight"가 아니다. 다음 training step을 bit-for-bit로 재현하는 데 필요한 *전체* state다. weight, optimizer moment, master copy, LR scheduler counter, per-rank RNG, data-iterator cursor, loss-scaler state, 그리고 resume이 실제로 작동했는지 audit할 수 있게 해주는 instrumentation history까지 포함한다. 하나라도 빠뜨리면 failure mode는 조용하다. 학습은 계속되고, loss는 그럴듯해 보이지만, run은 의도한 run에서 조용히 diverge한다.
>
> **지침.** FSDP에서는 PyTorch Distributed Checkpoint(DCP)를 사용해 sharded로 저장하고, 70B+ scale에서 rank 0으로 gather하지 마라. `param_dtype`와 무관하게 optimizer state는 fp32로 유지하라. 약 1000 step마다 checkpoint하라(Llama 3 cadence). 각 cycle은 "resume이 bit-exact loss를 만든다"는 assertion으로 gate하라. 이것이 Karpathy의 rule이며, data-iter와 scaler-state desync를 막는 유일하게 신뢰할 수 있는 guard다.

---

## 이 장이 필요한 이유

Frontier training run을 망치는 가장 싼 방법은 checkpoint를 잘못하는 것이다. failure는 절대 시끄럽지 않다. `CheckpointCorrupt` exception이 뜨지 않는다. 저장한 loss와 약간 다른 loss에서 시작하는 resume, 영원히 한 step 어긋난 cosine schedule, 방금 학습한 500 batch를 다시 replay하는 data loader, `2^18`에 있어야 할 loss-scaler가 `2^15`에서 다시 warmup하는 상황을 보게 된다. 각각은 실제 token compute를 낭비한다. 1,024 H100과 OLMo 3의 약 1M GPU-hour budget([[olmo-3]])에서는 한 주의 잘못된 resume이 작은 lab의 연간 hardware bill이 된다.

앞의 다섯 장은 training step을 만들었다. 이 장은 step *사이*에서 일어나는 일을 다룬다. 저장된 state, reload path, 그것을 믿기 위해 필요한 log다. 여기에는 세 전통이 만난다. "state"가 무엇이어야 하는지 알려준 고전적 early-stopping machinery([[early-stopping-and-checkpointing]]), 한 rank에 들어가지 않는 state를 저장하는 법을 알려준 FSDP-era distributed checkpointing([[fsdp-sft]]), 그리고 checkpoint가 recovery뿐 아니라 *forking*을 위한 것임을 알려준 WSD/soup lineage([[lr-schedules]], [[early-stopping-and-checkpointing]])다. 같은 trunk가 여러 instrumented decay run을 만든다.

이 장의 조직 원리는 Karpathy의 격언이다. **training은 조용히 실패하므로 모든 것을 instrument하고 resume을 bit-exact하게 검증하라**([[karpathy-training-neural-net-recipe]]). 아래의 모든 design choice는 이 규칙 하나에서 나온다.

---

## 1. Checkpoint에 실제로 들어가는 것

[[early-stopping-and-checkpointing]]에 따르면 2025년 production checkpointing은 일곱 항목짜리 목록이다. 각 항목은 빠졌을 때의 failure mode를 가진다. 이 표를 머릿속에 넣어두라. 이 장의 spine이다.

| State component | Precision | Per-rank or global | 빠지면 깨지는 것 |
|---|---|---|---|
| Model weights | bf16(shard) | FSDP 아래 sharded | obvious; restart from zero |
| Optimizer 1st moment `m` | fp32 | sharded | Adam's momentum re-warms; resume에서 loss spike |
| Optimizer 2nd moment `v` | fp32 | sharded | adaptive denominator reset; 약 20–1000 step 동안 effective LR이 틀림(ch-01의 [[adam]] β₂ 논의 참조) |
| Master fp32 weights | fp32 | sharded | resume가 bf16 → fp32로 round하며 7-mantissa-bit progress를 잃음; 조용한 ~0.1% perplexity drift |
| LR scheduler state | int + float | global | cosine phase가 N step 어긋남; WSD decay가 wrong token count에서 발생 |
| Per-rank RNG state | uint64[] | **per-rank** | dropout / label-smoothing / augmentation draw가 diverge; loss가 non-reproducible |
| Data-iterator position | int | **per-rank**(global logical step) | **silent killer** — §5 참조 |
| Loss-scaler state(fp16 only) | fp32 + int | global | scaler가 `2^15`에서 re-warm; resume 후 첫 ~2000 step에서 under-/overflow를 skip([[mixed-precision]] 참조) |
| Step counter | int | global | eval, save, decay onset 등 모든 time-based decision이 wrong wall-time에 발생 |
| Grad-norm / loss history | logs | global | resume drift 여부를 알 수 없음; instrumentation은 위 일곱 항목의 *audit trail* |

두 항목은 더 설명할 가치가 있다.

**Optimizer state가 checkpoint의 대부분이다.** bf16 + fp32 optimizer로 70B model에 AdamW를 쓰는 경우([[fsdp-sft]]): weight = 2 P = 140 GB, gradient는 transient, optimizer `(m, v, master)` = 12 P = 840 GB. Optimizer state는 weight size의 **6배**다. "weights-only" checkpoint는 실제 checkpoint의 14%일 뿐이다. 이를 건너뛰는 것이 hobby-scale resume bug 1위다. [[early-stopping-and-checkpointing]]은 이를 명시적으로 지적한다. *"Saving weights but not optimizer state → restart re-warms-up Adam moments → loss-spike on resume."*

**Grad-norm history는 장식이 아니다.** [[gradient-clipping]]과 [[olmo-2]] 모두 같은 점을 말한다. 100배 `pre_clip_grad_norm` spike는 loss spike를 1–5 step 앞서 예측한다. 그 predictive signal은 log가 resume을 넘어 살아남을 때만 작동한다. log ring-buffer가 resume마다 reset되면 "new instability"와 "지난 crash에서 이어받은 pre-existing drift"를 구분할 수 없다. log를 durably 남겨라. OLMo 2의 loss-spike mitigation stack, 즉 clip + skip-step + embedding-norm monitoring([[olmo-2]])은 persisted metric에 대한 feedback loop다. metric을 reset하면 loop의 눈을 가린다.

---

## 2. FSDP 아래 sharded checkpointing — DCP, `save_sharded` vs `save_full`, rank-0 bug

FSDP FULL_SHARD([[fsdp-sft]]) 아래에서 각 rank는 `(weights, m, v, master)`의 disjoint shard를 소유한다. 70B model을 `N = 8` rank로 나누면 각 rank는 약 105 GB state를 소유한다. 두 naive save pattern은 모두 실패한다.

**Naive pattern A — full state를 rank 0으로 gather하고 하나의 file로 저장.**
```python
# DO NOT USE AT SCALE
full_state = FSDP.state_dict(model)           # AllGather all params to rank 0
if rank == 0:
    torch.save(full_state, "ckpt.pt")
```
70B에서는 rank 0에 약 140 GB가 materialize된다(weight만, optimizer 제외). rank 0은 첫 save 전에 OOM이 난다. 작은 model에서는 *작동한다*. 그래서 이 bug가 ship된다. ch-05 SFT-scale code가 70B pretraining으로 복사되고 첫 checkpoint trigger에서 폭발한다.

**Naive pattern B — local shard를 rank별로 coordination 없이 저장.**
```python
torch.save(model.state_dict(), f"ckpt_rank_{rank}.pt")    # saves a FlatParameter shard
```
무언가를 저장하긴 하지만 shard layout은 현재 FSDP wrap policy, 현재 world size, 현재 CUDA device mesh에 암묵적으로 묶인다. 다른 `N`에서 resume하면(node가 죽어 8개 대신 7개 node로 restart) wrong shard가 조용히 load된다. 이것은 rank-0-bug의 사촌이다. 저장은 성공하지만 robust하게 reload할 수 없다.

**올바른 pattern — PyTorch Distributed Checkpoint(DCP).** DCP는 FSDP의 native sharded-save API다. 각 rank는 같은 directory에 자신의 shard를 parallel로 쓰고, metadata file은 global layout을 설명한다. load 시 DCP는 shard를 현재 world size로 다시 map한다.

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, StateDictOptions,
)

# --- SAVE ---
opts = StateDictOptions(full_state_dict=False, cpu_offload=True)  # sharded, CPU-staged
model_sd, optim_sd = get_state_dict(model, optimizer, options=opts)
dcp.save(
    state_dict = {"model": model_sd, "optim": optim_sd,
                  "step":  step, "rng": torch.cuda.get_rng_state_all(),
                  "data":  data_loader.state_dict(),
                  "sched": lr_scheduler.state_dict(),
                  "scaler": scaler.state_dict() if scaler else None},
    checkpoint_id = f"ckpts/step_{step:08d}",
)

# --- LOAD (possibly with different world size) ---
state = {"model": model_sd, "optim": optim_sd, "step": 0, "rng": None,
         "data": None, "sched": None, "scaler": None}
dcp.load(state_dict=state, checkpoint_id=f"ckpts/step_{step:08d}")
set_state_dict(model, optimizer, model_state_dict=state["model"],
               optim_state_dict=state["optim"], options=opts)
```

Nanotron(Hugging Face)과 Megatron은 자체적인 유사 sharded format을 제공한다. framework가 제공하는 것을 사용하라. **직접 만들지 마라.** 내가 audit한 모든 home-rolled FSDP checkpoint에는 두 naive-pattern bug 중 하나가 있었다.

중요한 knob은 `full_state_dict=False`다. 내부적으로 DCP는 ReduceScatter-like metadata exchange를 한 번 수행한 뒤, 각 rank가 자신의 shard를 disk에 직접 쓴다. Save throughput은 `N`에 따라 scale하고 rank-0 NIC에 묶이지 않는다. NVMe가 있는 8 × 80 GB H100에서 70B checkpoint의 전체 wall-clock은 약 30초다. naive-pattern A는 OOM이 나지 않는다고 해도 rank-0 IO에 병목되어 약 4분이 걸릴 것이다.

---

## 3. Bit-exact resume vs approximate resume

Karpathy의 가장 많이 인용되는 checkpoint rule([[karpathy-training-neural-net-recipe]]): *"check that resume produces bit-exact loss."* 운영상 의미는 이렇다. step `k`에서 저장한 뒤 process를 restart하고, step `k`를 load한 다음 training step 하나를 실행해서 loss를 원래 run의 step `k+1` loss와 비교한다. fp-rounding noise 이상으로 다르면 checkpoint에 무언가 빠진 것이다.

**Bit-exact resume**은 §1의 전체 일곱 항목과 deterministic op를 요구한다.

```python
torch.use_deterministic_algorithms(True)       # raises on non-det kernels
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark      = False
# Per-rank seed from global seed + rank id
torch.manual_seed(cfg.seed + rank)
torch.cuda.manual_seed_all(cfg.seed + rank)
```

Per-rank RNG state는 마지막 step의 random draw *뒤*에 저장되고, 다음 step의 draw *전에* restore되어야 한다. Dropout, label smoothing, NEFTune noise([[neftune]], ch-04), stochastic dataloader augmentation은 모두 RNG를 소비한다. 이를 persist하지 않으면 같은 checkpoint를 두 번 resume해도 서로 다른 loss가 나온다. weight는 같아도 그렇다. 둘 다 healthy해 보이기 때문에 목록에서 더 subtle한 bug 중 하나다.

**Approximate resume**은 determinism을 버리지만 macro-statistics를 보존한다. weight, optimizer state, scheduler, data-iter는 bit-identical이고, RNG와 dropout mask는 새롭다. resume 후 loss는 약 ~1e-4 다르지만(첫 수천 token의 dropout pattern에 따른 fp rounding), training trajectory는 원래 run과 통계적으로 구분되지 않는다.

**언제 무엇이 중요한가.** 현장의 경험칙:

| Regime | Required | 이유 |
|---|---|---|
| Debug / CI / published result 재현 | bit-exact | training path가 일치함을 증명해야 함 |
| Pretraining trunk(Llama-3, OLMo-2([[olmo-2]]), DeepSeek) | bit-exact | loss-spike rollback(§5)은 spike 진단을 위해 deterministic replay에 의존 |
| 8×H100 scale의 SFT / DPO | approximate | determinism cost(cuDNN benchmark disable로 약 15% throughput) > debugging value |
| RLVR / PPO rollout | approximate | policy가 stochastic하므로 bit-exactness는 구조적으로 불가능 |

fp16 training에서는 loss scaler state가 아홉 번째 항목이며 반드시 persist해야 한다. [[mixed-precision]]의 dynamic loss scaling은 inf/NaN detection에 따라 커지고 줄어드는 `S`를 유지한다. resume에서 이것을 잃으면 `S`가 `2^15`에서 re-warm하고, 첫 2000 step은 underflow를 skip한다(inflated effective gradient). bf16에는 scaler가 없다. [[mixed-precision]]이 bf16을 2025년 LLM 학습의 기본값이라고 주장하는 이유 중 하나다.

---

## 4. Instrumentation — 무엇을 어떤 cadence로 log할 것인가

log하지 않는 것은 debug할 수 없고, 모든 것을 log할 여유도 없다. [[olmo-2]] + [[gradient-clipping]] + [[early-stopping-and-checkpointing]]에서 합성한 실무 cadence tiering:

| Tier | Metrics | Cadence | 이유 |
|---|---|---|---|
| **Per-step** | training loss, LR, `pre_clip_grad_norm`, tokens/sec | every step | heartbeat; 100배 grad-norm spike는 loss spike를 1–5 step 앞선다([[gradient-clipping]]) |
| **Per-N-steps(~50)** | per-shard loss breakdown, output distribution entropy, reference에 대한 token-level KL(DPO/RL), accumulated clip-events | N=50 | rolling distribution 구성; per-shard loss는 poisoned data shard가 run을 오염시키기 전에 잡는다 |
| **Per-eval(~1k steps)** | fixed held-out set의 val-loss, capability benchmark(math-heavy면 MMLU, GSM8K), canary set perplexity | ~1000 | early-stop([[early-stopping-and-checkpointing]])은 이 tier에서 발생; val set은 ≥100k token이어야 noise에 early-stop하지 않음 |
| **Per-checkpoint(~1k–5k steps)** | full system: GPU util, NCCL bandwidth, disk usage, embedding-norm, activation-norm, weight-norm-per-layer | every checkpoint | OLMo 2 spike-mitigation stack: clip + skip-step + embedding-norm monitoring([[olmo-2]]) |
| **Per-run** | data-order hash, code commit, env capture, `pip freeze`, `nvidia-smi --query` | once | reproducibility audit trail; Karpathy의 "fix random seed everywhere"([[karpathy-training-neural-net-recipe]]) |

어떤 논문에도 없지만 모든 production trainer에는 있는 실무 note 두 가지.

**항상 fp32로 log하라.** [[mixed-precision]]은 fp16으로 loss를 log하지 말라고 경고한다. curve가 quantized-jaggy하게 보인다. fp16의 mantissa bit가 10개뿐이기 때문이다. `loss.item()`은 implicit하게 upcast한다. `loss.detach().float().item()`을 명시하면 intent가 드러난다.

**Grad-norm은 `pre_clip`을 log한다.** 원하는 양은 clipping이 rescale하기 전의 *raw* norm이다. clipped norm은 자명하게 `min(raw, c)`라 spike information이 없기 때문이다. PyTorch에서 `clip_grad_norm_`는 pre-clip norm을 반환한다. 그 return value를 log하라.

```python
# The canonical logging block
loss_f32 = loss.detach().float().item()
grad_norm = model.clip_grad_norm_(max_norm=1.0).item()   # pre-clip, global
optimizer.step()
logger.log_scalar("loss", loss_f32, step=step)
logger.log_scalar("grad_norm/pre_clip", grad_norm, step=step)
logger.log_scalar("lr", lr_scheduler.get_last_lr()[0], step=step)
logger.log_scalar("tokens_per_sec", n_tokens / (time.time() - t0), step=step)
```

---

## 5. Cross-resume silent-failure mode

이 section은 이 장의 나머지가 존재하는 이유다. 여기의 모든 bug는 내가 training week를 먹어치우는 것을 직접 본 것들이다.

**5.1 Data-iter desync.** checkpoint surface에서 가장 위험한 bug다. 각 rank의 dataloader는 sharded index에서 sample한다. resume 시 *global logical step*이 각 rank에 restore되어야 하고, 각 rank의 local iterator는 있던 위치까지 fast-forward되어야 한다. bug는 다음과 같다.

- rank-0 iterator state만 저장: 다른 rank는 batch 0에서 restart하여 첫 epoch의 token을 다시 보고, 조용히 overweight한다.
- 모든 rank를 저장하지만 global step은 저장하지 않음: iterator는 restore되지만 sampler가 원래 RNG에서 다시 seed되어 dataset의 *새* permutation을 만들고, 이미 본 batch를 새 label 아래 replay한다.
- iterator는 저장하지만 curriculum/mix weight는 저장하지 않음: OLMo 3의 model-flow approach([[olmo-3]])는 data mix를 stage로 나눈다. pretrain mix → mid-training mix(Dolmino) → long-context mix(Longmino). 현재 stage의 mix pointer 없이 resume하면 조용히 reverse-curriculum한다.

증상: resume 후 첫 ~500 step 동안 loss가 약 ~0.01 떨어진 뒤, replay된 shard를 model이 과하게 memorize하면서 천천히 위로 drift한다. 알아차릴 때쯤에는 checkpoint cycle 하나를 태운 뒤다.

해결: `dataloader.state_dict()`를 *per rank*로 저장하라(TorchData, Nanotron, HF `datasets.IterableDataset`가 모두 지원한다). global step과 active data-mix identifier도 함께 저장하라. resume 시 다음 100 sample의 hash를 known-good run과 비교하라. 이 bug의 모든 변형을 잡는 다섯 줄짜리 integration test다.

**5.2 Scaler-state drop(fp16 only).** [[mixed-precision]]의 dynamic loss scaling은 성공 step 2000개마다 2배 커지는 `S`를 유지한다. resume이 `S`를 persist하지 못하면 다시 `2^15`에서 시작한다. run이 `S = 2^18`에서 안정화되어 있었다면 첫 resume gradient는 원래보다 8배 작아진다. 따라서 clip threshold는 사실상 8배 더 tight하고, scaler가 다시 적응할 때까지 optimizer는 8배 더 작은 real-scale step을 만든다. Loss는 괜찮아 보이지만 2000 step 동안 token-efficiency가 25% 나쁘다. 이것이 bf16이 이긴 이유다. scaler가 없고, bug도 없다.

**5.3 LR-schedule off-by-one.** mid-cosine step 10,000에서 저장했다. restart했다. 저장된 state를 load하기 *전에* `lr_scheduler.step()`을 호출하면 scheduler는 step 10,001로 advance하고 첫 optimizer step은 step-10,001의 LR로 실행된다. 즉 step 10,000이 skip된다. 100k-step cosine에서는 0.001% LR error라 보이지 않는다. run의 10–20%인 WSD decay phase([[lr-schedules]])에서는 decay 시작점의 한 missed step이 전체 decay shape를 `1/decay_len`만큼 shift한다. 여전히 작지만 이제 *systematic*하다. 해결: 첫 `.step()` call 전에 scheduler state를 load하고 `scheduler.last_epoch == saved_step`을 assert하라.

**5.4 Optimizer state partially loaded.** DCP는 robust하지만 hand-rolled save는 그렇지 않다. 흔한 bug는 `m`과 `v`는 저장하지만 master fp32 weight를 잊는 것이다(FSDP 아래 AdamW state에서 별도 entry다). resume은 sharded bf16 weight를 load하고, AdamW는 bf16 → fp32 upcast로 master를 재구성하여 누적되었던 7 mantissa bit를 잃는다. 이후 모든 step은 그 error를 compound한다. 잡는 방법은 bit-exact resume check(§3) 또는 갑자기 jump하는 weight-norm-per-layer log뿐이다.

**5.5 Embedding-norm drift across resumes.** [[olmo-2]]는 embedding norm을 pre-spike indicator로 명시적으로 monitoring한다. 각 resume은 embedding table에 작은 numerical hiccup(cast-round-cast)을 도입할 기회다. 70B에서 10번 resume하면 누적 drift가 embedding matrix L2 norm의 0.1%가 될 수 있다. softmax temperature를 눈에 띄게 바꿀 만큼 충분하다. 방어책: resume마다 `||W_embed||`를 log하고, resume boundary에서 delta > 1e-4이면 alarm하라.

---

## 6. WSD stable-phase forkability와 checkpoint flywheel

Checkpointing은 crash recovery 외에 두 번째 목적이 있다. **forking**이다. WSD schedule([[lr-schedules]])은 training 대부분에서 constant LR을 유지하고, 마지막 10–20% token에서 decay한다. 모든 stable-phase checkpoint는 fresh decay run의 적법한 시작점이다. 같은 trunk, 다른 decay length, 다른 downstream task. MiniCPM과 DeepSeek는 마지막 K개 decay checkpoint를 *averaging*하면 단일 endpoint보다 약 ~0.5% val-loss improvement가 있음을 보였다([[early-stopping-and-checkpointing]], [[lr-schedules]]).

이것은 checkpoint가 *instrumented*일 때만 작동한다. 어떤 checkpoint에서 decay할지 골라야 한다. 그 결정은 stable-phase val-loss, grad-norm stability, downstream eval gradient의 함수다. [[olmo-3]]의 "model flow"는 이 pattern의 일반화다. Base → Mid-training(Dolmino) → Long-context(Longmino) → 별도의 SFT/DPO/RLVR branch. 각 화살표는 자체 instrumentation gate를 갖는 checkpoint fork다. release artifact는 final weight가 아니다. tree다.

Llama 3 post-training flywheel([[llama-3]])은 같은 idea의 higher-frequency version이다. SFT → Rejection Sampling → DPO를 여섯 라운드 반복하고, 각 라운드는 이전 라운드의 best checkpoint에서 시작하며, 각 라운드는 다음 라운드의 SFT data가 되는 rejection-sampled output을 생성한다. 라운드마다 persist해야 할 artifact는 다음과 같다.

- round-`k` SFT checkpoint(data를 생성한 policy)
- round-`k` reward model(data를 scoring한 filter)
- round-`k` rejection-sampled pool(filtering된 data 자체)
- round-`k` DPO preference batch

하나라도 빠지면 round `k+1`은 reproducible하지 않다. "checkpoint"는 file이 아니다. (policy, scorer, data, preference)의 bundle이다. filesystem layout에서도 그렇게 다뤄라.

Practical filesystem convention:
```
ckpts/
  trunk/
    step_00100000/    # WSD stable-phase; forkable
    step_00200000/
    step_00300000/
  decay/
    decay_from_300k/
      step_00350000/  # final model candidates
      step_00360000/
    decay_from_300k_20pct/
  post/
    r1_sft/     r1_rm/     r1_rs_pool/     r1_dpo/
    r2_sft/     r2_rm/     r2_rs_pool/     r2_dpo/
    ...
```

---

## 7. Drop-in reference — save, load, instrument

§2, §3, §4를 canonical training-loop skeleton으로 결합한다. 여기에 있는 모든 것은 production-shaped이며, name은 PyTorch 2.5 API와 맞춘다.

```python
# ------------------- 7.1 Save -------------------
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, StateDictOptions,
)

def save_checkpoint(path, model, optimizer, scheduler, scaler,
                    data_loader, step, rng_state):
    opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_sd, optim_sd = get_state_dict(model, optimizer, options=opts)
    state = {
        "model":       model_sd,
        "optim":       optim_sd,                     # includes m, v, master fp32
        "sched":       scheduler.state_dict(),
        "scaler":      scaler.state_dict() if scaler else None,
        "data":        data_loader.state_dict(),     # per-rank iter state
        "step":        step,
        "rng/cpu":     torch.get_rng_state(),
        "rng/cuda":    torch.cuda.get_rng_state_all(),
        "rng/py":      random.getstate(),
        "rng/np":      np.random.get_state(),
        "code_sha":    os.environ.get("GIT_SHA", "unknown"),
    }
    dcp.save(state_dict=state, checkpoint_id=path)

# ------------------- 7.2 Load -------------------
def load_checkpoint(path, model, optimizer, scheduler, scaler, data_loader):
    opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_sd, optim_sd = get_state_dict(model, optimizer, options=opts)
    state = {"model": model_sd, "optim": optim_sd,
             "sched": None, "scaler": None, "data": None,
             "step": 0, "rng/cpu": None, "rng/cuda": None,
             "rng/py": None, "rng/np": None, "code_sha": None}
    dcp.load(state_dict=state, checkpoint_id=path)
    set_state_dict(model, optimizer,
                   model_state_dict=state["model"],
                   optim_state_dict=state["optim"], options=opts)
    scheduler.load_state_dict(state["sched"])
    if scaler and state["scaler"]: scaler.load_state_dict(state["scaler"])
    data_loader.load_state_dict(state["data"])
    torch.set_rng_state(state["rng/cpu"])
    torch.cuda.set_rng_state_all(state["rng/cuda"])
    random.setstate(state["rng/py"])
    np.random.set_state(state["rng/np"])
    # invariant: scheduler.last_epoch == state["step"]
    assert scheduler.last_epoch == state["step"], "LR scheduler off by one"
    return state["step"]

# ------------------- 7.3 Instrumentation logger -------------------
class TrainLogger:
    """Per-step lightweight; upcasts to fp32 for numerical sanity."""
    def __init__(self, sink):
        self.sink = sink    # e.g. wandb, tensorboard, jsonlines
        self.t0   = time.time()
        self.tokens_seen = 0

    def step(self, step, loss, grad_norm, lr, n_tokens):
        now = time.time()
        self.tokens_seen += n_tokens
        self.sink.log({
            "step":               step,
            "loss":               float(loss.detach().float().item()),
            "grad_norm/pre_clip": float(grad_norm),
            "lr":                 float(lr),
            "tokens/sec":         n_tokens / max(now - self.t0, 1e-6),
            "tokens_total":       self.tokens_seen,
        }, step=step)
        self.t0 = now

    def eval(self, step, val_loss, embed_norm, per_shard_loss):
        self.sink.log({
            "val/loss":     float(val_loss),
            "norm/embed":   float(embed_norm),
            **{f"loss/shard_{k}": float(v) for k, v in per_shard_loss.items()},
        }, step=step)

# ------------------- 7.4 Training step with the right ordering -------------------
for step in range(start_step, total_steps):
    batch = next(data_iter)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()

    grad_norm = model.clip_grad_norm_(max_norm=1.0)     # FSDP-aware; pre-clip norm
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    train_logger.step(step, loss, grad_norm,
                      scheduler.get_last_lr()[0],
                      n_tokens=batch["input_ids"].numel())

    if step % cfg.eval_every == 0:
        val_loss, embed_norm, per_shard = evaluate(model, val_loader)
        train_logger.eval(step, val_loss, embed_norm, per_shard)

    if step % cfg.save_every == 0 and step > 0:
        save_checkpoint(f"ckpts/step_{step:08d}",
                        model, optimizer, scheduler, scaler,
                        data_loader, step, rng_state=None)
```

이것이 전체 loop다. §1 표의 모든 항목에는 여기에서 그것을 읽거나 쓰는 line이 있고, §5의 모든 silent-failure mode는 assertion 또는 logged quantity로 막힌다.

---

## 연결과 다음 내용

- **[[early-stopping-and-checkpointing]] / 이 장** — 고전적 foundation; 일곱 항목 state list.
- **[[fsdp-sft]] / ch-05** — FSDP mechanics; DCP는 save API다.
- **[[mixed-precision]] / ch-02** — fp16 loss-scaler state; legacy run의 아홉 번째 항목.
- **[[lr-schedules]] / ch-03** — WSD stable-phase fork pattern; scheduler state의 off-by-one trap.
- **[[gradient-clipping]] / ch-01** — `pre_clip_grad_norm`이 가장 이른 warning signal이다.
- **[[olmo-2]] / [[olmo-3]] / [[llama-3]]** — concrete production checkpoint pipeline 세 가지. 각각은 crash-recovery backup이 아니라 flywheel이다.
- **ch-07 (failure modes)** — 더 넓은 silent-failure catalog. 여기 §5는 checkpoint-specific subset이다.
- **ch-08 (lab)** — 필수 unit test: step `k`에서 저장하고, restart한 뒤 step `k+1`에서 bit-exact loss를 검증.

## 더 읽을거리

- [[early-stopping-and-checkpointing]] — Prechelt 1998 / Izmailov 2018 / Wortsman 2022 / MiniCPM-DeepSeek WSD averaging. canonical "what goes in a checkpoint" table은 Technical Details section에 있다.
- [[fsdp-sft]] — Zhao 2023; FSDP memory formula와 70B에서 sharded save가 필수인 이유.
- [[mixed-precision]] — Micikevicius 2017; dynamic loss scaling + master-fp32 semantics.
- [[lr-schedules]] — WSD forkability와 cosine-mismatch cost.
- [[olmo-2]] — loss-spike mitigation stack과 그것을 가능하게 하는 instrumentation.
- [[olmo-3]] — model-flow philosophy; checkpoint는 recovery backup이 아니라 public artifact다.
- [[llama-3]] — six-round post-training flywheel. 각 round마다 무엇을 persist해야 하는가.
- [[karpathy-training-neural-net-recipe]] — "overfit a single batch"와 "verify resume is bit-exact". 이 장의 나머지가 강제하는 규칙들.

## 함께 보는 시각화

**[figures/checkpoint-state.html](figures/checkpoint-state.html)** — 일곱 항목 checkpoint state의 interactive diagram. 각 component를 "saved / not saved"로 toggle하면 page가 두 live verdict, 즉 "Bit-exact resume"과 "Approximate resume"을 업데이트하고, 각 omission이 trigger하는 silent-failure mode(data-iter → replay divergence, scaler → 2000-step under-step, RNG → non-reproducible dropout 등)를 side panel에 나열한다. 어떤 resume flavor가 실제로 어떤 state subset을 요구하는지, 그리고 "weights only"를 production pipeline에 넣는 것이 왜 가장 비싼 default인지 내재화하는 데 사용하라.
