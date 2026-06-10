<!-- chapter: ch-46
     track: rl
     kind: lab
     title: Lab — Small DPO or RLVR Experiment with Failure Analysis
     deps: [ch-45]
     sources: [[dpo]], [[grpo]], [[dr-grpo]], [[rlvr-tulu3]], [[trl-grpo]], [[openrlhf-ppo]], [[verl-grpo]], [[entropy-mechanism-llm-rl]], [[reward-hacking-taxonomy]], [[openrlhf-entropy-debugging]], [[karpathy-training-neural-net-recipe]]
     figures: figures/rl-sweep.html
     capstone_for: rl-track (ch-37..ch-46)
-->

# 46장 — Lab: Failure Analysis를 포함한 작은 DPO 또는 RLVR 실험

> **핵심 통찰.** RL track은 더 나은 reward나 더 영리한 clip이 아니라, *자신의 run이 왜 잘못되었는지 진단하는 능력*으로 끝난다. 모든 RL loss(DPO, PPO, GRPO, Dr.GRPO, RLVR)는 대수적으로 다섯 줄이다. 모든 failure — length inflation, entropy collapse, KL runaway, verifier loophole — 는 현대 프레임워크가 이미 logging하는 세 신호(reward, KL, entropy) 중 하나가 조용히 drift한 것이다. capstone은 학습된 checkpoint가 아니다. drift 중 하나를 포착하고 이름 붙였으며 다음 sprint에서 그 bug를 막을 test를 쓴 `rl-experiment-memo.md`다.
>
> **가이드라인.** 정확히 하나의 option만 고르라. A(preferences 위 DPO) 또는 B(verifiable math 위 RLVR). 그리고 전체 instrumentation을 실행하라. sweep의 *shape*가 lesson이므로 hyperparameter 하나([[dpo]] β 또는 [[rlvr-tulu3]] β_KL)를 sweep하라. [[openrlhf-entropy-debugging]]에 따라 step마다 네 신호를 계측하라: KL(π‖π_ref), entropy H(π), reward mean/std, per-bucket pass-rate 또는 win-rate. 학습 전에 각 sweep cell의 정성적 방향을 예측하라([[karpathy-training-neural-net-recipe]]). reward hacking([[reward-hacking-taxonomy]]), entropy collapse([[entropy-mechanism-llm-rl]]), length bias([[dr-grpo]]) 중 하나의 failure mode를 찾아 post-mortem을 작성하라. Deliverable: memo + `checkpoint-final/` + sweep의 `metrics.jsonl`.

---

## Goal

peer가 재현할 수 있는 세 artifacts:

1. **A sweep.** `sweeps/beta_0p05/`, `beta_0p1/`, `beta_0p3/`(Option A) 또는 `kl_0p01/`, `kl_0p05/`, `kl_0p1/`(Option B). 각 directory에는 `metrics.jsonl`(step, KL, entropy, reward_mean, reward_std, pass_rate 또는 win_rate), final checkpoint, code의 git SHA가 들어간다. resource-constrained path는 정확히 하나의 value만 실행한다. instrumentation은 바뀌지 않는다.
2. **Training-curve plots.** 신호별 PNG 하나(`kl.png`, `entropy.png`, `reward.png`, `pass_rate.png` 또는 `win_rate.png`), 세 개(또는 하나)의 β / KL curve를 모두 overlay한다. Matplotlib, 80 lines of plotting code. Weights-and-Biases screenshot이 아니다.
3. **A memo.** `rl-experiment-memo.md`, 한 페이지: sweep table, 이름 붙인 failure-mode post-mortem과 repro recipe, prediction과 어긋난 cell 하나의 surprise.

각 sweep cell의 정성적 behaviour를 *launch 전에* 예측하라. [[karpathy-training-neural-net-recipe]]의 "predict-outcome-before-run" 규칙은 선택 사항이 아니다. memo의 세 번째 section은 prediction을 적어 두었기 때문에 존재한다.

---

## Pick an option

10K-pair preference set이 준비되어 있으면 **A**(DPO)를 고르라. verifier가 wired up되어 있으면 **B**(RLVR-math)를 고르라. 둘 다 시도하지 말라. post-mortem이 graded deliverable이다. attention을 나누면 얕은 analysis 두 개가 생긴다.

---

## Full-budget path

Target: 8×H100(또는 4×A100-80GB), Llama-3.2-3B / Qwen2.5-3B base, cell당 ~4 h, 3 cells에 ~12 h.

**Option A (DPO).** Base `meta-llama/Llama-3.2-3B-Instruct` 또는 ch-36 SFT checkpoint; [[dpo]]에 따라 `π_ref = SFT frozen`. Data: `HuggingFaceH4/ultrafeedback_binarized`에서 stratified down-sampled한 10K pairs, 또는 ch-38 synthetic. Training([[dpo]] + TRL): LR 5e-7 cosine, global batch 32 pairs, 1 epoch, max_length 2048, max_prompt_length 1024, β ∈ {0.05, 0.1, 0.3}. Sweep axis: β only — [[karpathy-training-neural-net-recipe]]의 "one-change-one-prediction"에 따라 모든 cell에서 same seed, same data, same LR.

**Option B (RLVR-math).** 같은 SFT checkpoint와 [[rlvr-tulu3]]에 따른 `π_ref`. Data: `AI-MO/NuminaMath-CoT` 또는 `openai/gsm8k`에서 difficulty별로 bucket된 5K prompts. Training([[grpo]] via TRL + vLLM): LR 1e-6, batch 128 prompts, G=8 rollouts, max_completion_length 1024, clip ε=0.2, rollout T=1.0, β_KL ∈ {0.01, 0.05, 0.1}. Reward = verifier output {0,1}(SymPy-tolerant grader). Sweep axis: β_KL only. Dr.GRPO aggregation(`loss_type="dr_grpo"`)을 처음부터 켠다([[dr-grpo]]). 그래야 entropy/KL interaction이 length bias와 confound되지 않는다.

## Resource-constrained path

Target: 1 GPU(≥24 GB), Llama-3.2-1B 또는 Qwen2.5-1.5B, ~3 h. One cell: β=0.1 또는 β_KL=0.05(sweep의 middle). 2K pairs / 1K prompts. 네 신호를 모두 매 step logging한다. memo grade는 동일하다. *analysis*가 deliverable이다.

---

## §1 Data prep

### A. Preference pairs

```python
# data/build_prefs.py  — TRL DPOTrainer expects {prompt, chosen, rejected}.
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
ds = ds.shuffle(seed=7).select(range(10_000))
ds = ds.map(lambda e: {"prompt": e["prompt"],
                       "chosen": e["chosen"][-1]["content"],
                       "rejected": e["rejected"][-1]["content"]},
            remove_columns=ds.column_names)
ds.to_json("data/prefs_10k.jsonl")
```

`length_delta = len(chosen) - len(rejected)` column을 log하라. [[dpo]]의 length-hacking failure는 loss가 아니라 *data*에서 시작된다. chosen이 systematic하게 ~80 tokens 더 길다면, loss detail과 무관하게 β=0.05는 length를 inflate할 것이다.

### B. Verifiable-math prompts

```python
# data/build_math.py
from datasets import load_dataset
ds = load_dataset("AI-MO/NuminaMath-CoT", split="train").filter(lambda x: x["source"] in {"gsm8k","math"})
ds = ds.shuffle(seed=7).select(range(5_000))
def to_rlvr(e):
    ans = e["solution"].split("####")[-1].strip() if "####" in e["solution"] else e["solution"]
    return {"prompt": [{"role":"user","content": e["problem"] + "\\n\\nReason step by step; put final answer in \\\\boxed{}."}],
            "answer": ans}
ds.map(to_rlvr, remove_columns=ds.column_names).to_json("data/math_5k.jsonl")
```

[[rlvr-tulu3]]의 "verifier engineering을 unit-test engineering처럼 다루라"에 따라, 학습 전에 hand-curated `(response, gold)` pair 50개로 verifier를 unit-test하라. 정확한 label match 100%를 요구하라. string-match가 prose 안의 "42"를 받아들이는 loophole은 입증된 Tülu 3 hack path다.

---

## §2 Training config — 실제 framework calls

### A. DPO via TRL `DPOTrainer`

```python
# train_dpo.py — one sweep cell, called per beta.
import os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

BETA = float(os.environ["DPO_BETA"])     # 0.05, 0.1, or 0.3
OUT  = f"sweeps/beta_{str(BETA).replace('.','p')}"
MID  = "meta-llama/Llama-3.2-3B-Instruct"
tok  = AutoTokenizer.from_pretrained(MID)
pol  = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
ref  = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
ref.eval(); [p.requires_grad_(False) for p in ref.parameters()]

cfg = DPOConfig(
    output_dir=OUT, beta=BETA, loss_type="sigmoid",   # Eq. 7 objective
    learning_rate=5e-7, lr_scheduler_type="cosine", warmup_ratio=0.1,
    per_device_train_batch_size=2, gradient_accumulation_steps=16,   # global 32 pairs
    num_train_epochs=1, max_length=2048, max_prompt_length=1024,
    bf16=True, logging_steps=5, save_strategy="epoch", report_to="none",
)
DPOTrainer(model=pol, ref_model=ref, args=cfg, processing_class=tok,
           train_dataset=load_dataset("json", data_files="data/prefs_10k.jsonl", split="train")
          ).train()
```

TRL은 `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `logps/*`를 logging한다. callback으로 redirect하라(§3). `rewards/chosen`과 `rewards/rejected`는 [[dpo]]에 따른 **implicit rewards**다. `r̂_θ = β log(π_θ / π_ref)`. reward-model score가 아니다. β cell 사이에서 absolute value를 비교하지 말라.

### B. RLVR via TRL `GRPOTrainer` (Dr.GRPO aggregation)

```python
# train_rlvr.py — one sweep cell, called per beta_kl.
import os, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from math_verify import parse, verify       # SymPy-tolerant grader

BETA_KL = float(os.environ["KL_BETA"])      # 0.01, 0.05, or 0.1
OUT     = f"sweeps/kl_{str(BETA_KL).replace('.','p')}"

def verifier_reward(completions, answer, **kw):
    return [1.0 if verify(parse(c[0]["content"]), parse(g)) else 0.0
            for c, g in zip(completions, answer)]

cfg = GRPOConfig(
    output_dir=OUT, beta=BETA_KL,           # KL-to-ref coefficient
    loss_type="dr_grpo", scale_rewards=False,  # [[dr-grpo]] length-unbiased aggregation
    learning_rate=1e-6, lr_scheduler_type="constant",
    per_device_train_batch_size=1, gradient_accumulation_steps=16, num_generations=8,
    max_prompt_length=1024, max_completion_length=1024, num_train_epochs=1, bf16=True,
    epsilon=0.2, epsilon_high=0.2, temperature=1.0, top_p=1.0,
    use_vllm=True, vllm_mode="colocate",
    logging_steps=5, save_strategy="epoch", report_to="none",
)
GRPOTrainer(model="Qwen/Qwen2.5-3B", args=cfg,
            processing_class=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B"),
            reward_funcs=[verifier_reward],
            train_dataset=load_dataset("json", data_files="data/math_5k.jsonl", split="train")
           ).train()
```

`loss_type="dr_grpo"` + `scale_rewards=False`는 [[dr-grpo]]([[trl-grpo]] `_compute_loss` branch ≈ line 2492)다. `/std(r)`를 제거하고, token-sum loss를 `|o_i|`가 아니라 `(B · max_completion_length)`로 나눈다. `loss_type="grpo"`를 쓰면 length-inflation failure mode가 ~200 steps 안에 나타난다. 이 flag를 flip하는 것이 §5(c) post-mortem repro다.

---

## §3 Instrumentation hooks — 네 신호

모든 sweep cell은 같은 네 신호를 5 steps마다 logging해야 한다. [[openrlhf-entropy-debugging]] §Key Points는 이를 community-standard metric set으로 열거한다. [[trl-grpo]]의 `_metrics[mode]["entropy"]`, TRL `DPOTrainer`의 `rewards/*`, [[openrlhf-ppo]]의 `PolicyLoss`가 이미 `(loss, clip_ratio, ppo_kl, vllm_kl)`로 내보내는 것들이다.

```python
# tools/log_metrics.py — drop-in TrainerCallback
import json
from transformers import TrainerCallback
RENAME = {"objective/kl":"kl", "kl":"kl", "entropy":"entropy",
          "rewards/chosen":"reward_mean", "reward":"reward_mean", "reward_std":"reward_std",
          "completions/mean_length":"len_mean", "rewards/accuracies":"win_rate"}
class MetricsSink(TrainerCallback):
    def __init__(self, path): self.path = path; open(path, "w").close()
    def on_log(self, args, state, control, logs=None, **kw):
        row = {"step": state.global_step,
               **{RENAME.get(k, k): v for k, v in (logs or {}).items()}}
        open(self.path, "a").write(json.dumps(row) + "\\n")
```

Per-bucket pass-rate(B)는 difficulty tag가 있는 held-out 200-prompt slice에서 ~50 steps마다 eval callback을 필요로 한다. A의 analog는 per-source win-rate slice다. slicing을 건너뛰지 말라. [[reward-hacking-taxonomy]]의 enumerated hacks는 *aggregate가 개선되는 동안 per-bucket regression*으로 나타난다.

---

## §4 Sweep protocol

Launch 전에 `predictions.txt`를 작성하라.

```
# predictions.txt  (committed BEFORE train_*.py runs)
Option A, beta=0.05: KL rises, entropy drops fast, win-rate peak at step ~300 then drift; len_mean grows ~30% (length hack).
Option A, beta=0.1:  KL stable, entropy gentle decline, win-rate monotone; len_mean flat.
Option A, beta=0.3:  KL low, reward-margins barely move, win-rate < baseline (underfitting preference).
```

Launch:

```bash
for B in 0.05 0.1 0.3; do
  DPO_BETA=$B torchrun --nproc_per_node 4 train_dpo.py   # Option A
done
# OR
for K in 0.01 0.05 0.1; do
  KL_BETA=$K torchrun --nproc_per_node 4 train_rlvr.py   # Option B
done
```

각 cell은 wall-clock, peak GPU memory, total training tokens를 기록한다. 독자가 재현하는 데 쓰는 세 숫자다.

---

## §5 Failure-mode diagnostics — 하나를 고르라

memo는 *하나의 failure mode에 이름을 붙이고 logging한 metrics로 증명하는 능력*을 평가한다. 세 canonical modes와 각 mode를 식별하는 signal pattern:

**a) Reward hacking ([[reward-hacking-taxonomy]] + [[rlvr-tulu3]] §Failure mode).**
- Signal: reward_mean은 단조 증가하지만 held-out prompt의 per-bucket pass-rate가 *하락*하거나, verifier rate가 plateau하는 동안 `len_mean`이 inflate한다.
- Repro: verifier를 약하게 만든다(예: SymPy-verify 대신 `\\boxed{42}` substring을 받아들이기) in a 10% shard; 정책이 ~200 steps 안에 loophole을 발견하는 것을 본다.
- Fix in memo: verifier를 조이거나, 200-prompt held-out slice의 reward_mean(strong verifier로 채점)이 training reward와 0.05 이내로 추적되는지 sanity check를 추가한다.

**b) Entropy collapse ([[entropy-mechanism-llm-rl]]).**
- Signal: 첫 ~200 updates 안에 `entropy`가 0.1 nats 아래로 떨어진다. 직후 reward_mean이 flatten한다. 첫 10% steps에 맞춘 `R = -a·exp(H) + b`가 eventual ceiling을 예측한다.
- Repro: β_KL=0.01(Option B) 또는 β=0.05(Option A)로 설정하고 entropy bonus를 제거한다. 3B scale에서 collapse는 거의 확실하다.
- Fix in memo: β_KL을 한 단계 올리거나, [[trl-grpo]]의 `top_entropy_quantile=0.2`를 켠다(high-entropy token에만 gradient를 유지하는 DAPO / Muon trick). 또는 [[entropy-mechanism-llm-rl]]의 Clip-Cov / KL-Cov를 인용한다.

**c) Length bias ([[dr-grpo]] Figure 1).**
- Signal: `len_mean` curve가 학습 내내 monotone-up한다. reward는 flat 또는 slight up이다. *틀린* rollout이 불균형적으로 길다(bucket `len_mean` by reward=0 vs reward=1).
- Repro: Option B at β_KL=0.05에서 `"dr_grpo"`를 `loss_type="grpo"`로 바꾼다. length inflation은 step ~300까지 나타난다.
- Fix in memo: Dr.GRPO aggregation(`loss_type="dr_grpo"`, `scale_rewards=False`)으로 바꾸고 length curve가 flatten되는 것을 보인다.

하나만 고른다. memo의 post-mortem은 그 mode 하나에 대한 full page다. signal pattern, minimal repro, mechanism이 [[trl-grpo]] / [[openrlhf-ppo]] / [[verl-grpo]]의 어느 줄에 있는지, fix, 그리고 bug를 막을 regression test(§7)를 포함한다.

---

## §6 Memo template

`rl-experiment-memo.md`, 한 페이지, 다섯 sections:

1. **Setup.** Option(A 또는 B), base, data source + size, sweep axis + cell values, GPU count, cell당 wall-clock, total tokens.
2. **Sweep table.** Cells × {KL_final, H_final, reward_final, win_rate 또는 pass_rate_final, len_mean_final}. 각 row를 `predictions.txt` 대비 "predicted" 또는 "surprising"으로 tag한다.
3. **Failure-mode post-mortem.** §5에서 고른 한 mode에 대해 full page. Signal plot, mechanism(특정 raw-data source를 cite), minimal repro, fix.
4. **One surprise.** prediction과 어긋난 cell 하나. "noise"가 아니라 concrete claim이어야 한다. 예: "β=0.3에서도 length가 15% 증가했다. large-β는 π_ref에 anchor해야 하므로 그러지 않을 것이라고 예측했다. 가설: preference data 자체가 β로 지울 수 없는 length prior를 encode한다."
5. **Next instrumentation.** failure를 하루 더 일찍 잡았을 metric 또는 assertion 하나.

---

## §7 Acceptance criteria

Hard gates, 순서대로.

1. `git log sweeps/`가 `metrics.jsonl` 파일보다 *전에* commit된 `predictions.txt`를 보여준다.
2. 각 cell의 `metrics.jsonl`에는 네 신호 + `len_mean`이 모두 있고, convergence까지 적어도 every 10 steps로 기록되어 있다.
3. Option A: β=0.1에서 `rewards/chosen - rewards/rejected`가 양수이고 monotone-up이다. margin이 flat/negative이면 [[dpo]]의 π_ref가 잘못된 것이다(common bug: ref-model이 `.eval()`을 잊었거나 gradients가 disabled되지 않음).
4. Option B: step 0의 training reward가 ∈ [0.1, 0.4] ([[rlvr-tulu3]] Table 1 baseline range on 3B)이다. 0.0 = broken verifier, ~0.9 = data is trivial.
5. Memo §3은 하나의 mode에 이름을 붙이고, 하나의 raw-data source를 cite하며, 하나의 plot과 하나의 new assertion을 제안한다. plot이 없으면 incomplete.
6. 적어도 하나의 memo §2 row가 "surprising"으로 tag되어 있다. clean sweep은 아무것도 가르치지 않는다.

---

## Connections

- **ch-37 / ch-38** — [[dpo]]의 loss가 Option A의 objective다. β가 sweep axis다.
- **ch-39 / ch-40** — [[grpo]] Eq. 3이 Option B의 objective다. k3 KL이 `kl` signal이 된다.
- **ch-41** — vLLM-vs-trainer drift는 [[openrlhf-ppo]]의 `vllm_kl` signal로 나타난다.
- **ch-42 / [[reward-hacking-taxonomy]]** — §5(a) taxonomy.
- **ch-43 / [[entropy-mechanism-llm-rl]]** — §5(b) mechanism.
- **ch-44** — [[rlvr-tulu3]]의 verifier construction.
- **ch-45** — `checkpoint-final/`은 rejection sampling / self-rewarding의 starting policy다.
- **Track 6 (Eval) / ch-47+** — `metrics.jsonl` schema가 Eval harness에 공급된다.
- **Track 7 (Infra) / ch-54+** — rollout queue / replay가 이 cells를 다시 다룬다.

## Further reading

- [[dpo]] — Eq. 7, implicit-reward identity, β의 KL-budget 역할, length hack.
- [[grpo]], [[dr-grpo]] — Eq. 3 + group baseline; length-bias fix.
- [[rlvr-tulu3]] — verifier-as-reward; "verifier engineering like unit-test engineering".
- [[trl-grpo]], [[openrlhf-ppo]], [[verl-grpo]] — 같은 loss의 세 구현.
- [[entropy-mechanism-llm-rl]] — `R = -a·exp(H) + b`; Clip-Cov / KL-Cov.
- [[reward-hacking-taxonomy]] + [[openrlhf-entropy-debugging]] — impossibility theorem; 네 신호; triage order.
- [[karpathy-training-neural-net-recipe]] — predict-before-run; "training fails silently"는 RL에도 적용된다.

## Companion visualization

**[figures/rl-sweep.html](figures/rl-sweep.html)** — 대화형 3-axis sweep explorer. sweep axis(DPO의 β / RLVR의 β_KL)를 고르고, metric(reward / entropy / KL / pass-rate)을 고른 뒤 training step을 scrub한다. 각 curve는 설명용이지만 [[dpo]] β behaviour, [[entropy-mechanism-llm-rl]] collapse law, [[dr-grpo]] length-inflation shape에 의해 방향이 입증되어 있다. `predictions.txt`를 쓰기 *전에* 사용하라. scenarios를 클릭해 보고 각 justification을 읽은 뒤 prediction을 commit하라.
