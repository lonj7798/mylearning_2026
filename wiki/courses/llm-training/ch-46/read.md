<!-- chapter: ch-46
     track: rl
     kind: lab
     title: Lab — Small DPO or RLVR Experiment with Failure Analysis
     deps: [ch-45]
     sources: [[dpo]], [[grpo]], [[dr-grpo]], [[rlvr-tulu3]], [[trl-grpo]], [[openrlhf-ppo]], [[verl-grpo]], [[entropy-mechanism-llm-rl]], [[reward-hacking-taxonomy]], [[openrlhf-entropy-debugging]], [[karpathy-training-neural-net-recipe]]
     figures: figures/rl-sweep.html
     capstone_for: rl-track (ch-37..ch-46)
-->

# Chapter 46 — Lab: Small DPO or RLVR Experiment with Failure Analysis

> **Core insight.** The RL track closes not with a better reward or a cleverer clip, but with the ability to *diagnose why your own run went wrong*. Every RL loss (DPO, PPO, GRPO, Dr.GRPO, RLVR) is algebraically five lines; every failure — length inflation, entropy collapse, KL runaway, verifier loophole — is a silent drift of one of three signals (reward, KL, entropy) that every modern framework already logs. The capstone is not a trained checkpoint; it is a `rl-experiment-memo.md` where you caught one of those drifts, named it, and wrote a test that would block the bug next sprint.
>
> **Guideline.** Pick exactly one option — A (DPO over preferences) or B (RLVR over verifiable math) — and run the full instrumentation. Sweep one hyperparameter ([[dpo]] β or [[rlvr-tulu3]] β_KL) because the *shape* of the sweep is the lesson. Instrument four signals per step per [[openrlhf-entropy-debugging]]: KL(π‖π_ref), entropy H(π), reward mean/std, per-bucket pass-rate or win-rate. Predict every sweep cell's qualitative direction before training ([[karpathy-training-neural-net-recipe]]). Find one failure mode — reward hacking ([[reward-hacking-taxonomy]]), entropy collapse ([[entropy-mechanism-llm-rl]]), or length bias ([[dr-grpo]]) — and write its post-mortem. Deliverable: memo + `checkpoint-final/` + the sweep's `metrics.jsonl`.

---

## Goal

Three artifacts, each reproducible by a peer:

1. **A sweep.** `sweeps/beta_0p05/`, `beta_0p1/`, `beta_0p3/` (Option A) or `kl_0p01/`, `kl_0p05/`, `kl_0p1/` (Option B). Each directory contains `metrics.jsonl` (step, KL, entropy, reward_mean, reward_std, pass_rate or win_rate), a final checkpoint, and the git SHA of the code. Resource-constrained path runs exactly one value; the instrumentation is unchanged.
2. **Training-curve plots.** One PNG per signal (`kl.png`, `entropy.png`, `reward.png`, `pass_rate.png` or `win_rate.png`), all three (or one) β / KL curves overlaid. Matplotlib, 80 lines of plotting code, not Weights-and-Biases screenshots.
3. **A memo.** `rl-experiment-memo.md`, one page: sweep table, one named failure-mode post-mortem with a repro recipe, one surprise where a cell broke your prediction.

Predict each sweep cell's qualitative behaviour *before launch*. [[karpathy-training-neural-net-recipe]]'s "predict-outcome-before-run" rule is not optional — the memo's third section exists because you wrote predictions down.

---

## Pick an option

Choose **A** (DPO) if you have a 10K-pair preference set ready; **B** (RLVR-math) if a verifier is wired up. Do not attempt both — post-mortem is the graded deliverable; split attention produces two shallow analyses.

---

## Full-budget path

Target: 8×H100 (or 4×A100-80GB), Llama-3.2-3B / Qwen2.5-3B base, ~4 h per cell, ~12 h for 3 cells.

**Option A (DPO).** Base `meta-llama/Llama-3.2-3B-Instruct` or your ch-36 SFT checkpoint; `π_ref = SFT frozen` per [[dpo]]. Data: 10K pairs from `HuggingFaceH4/ultrafeedback_binarized` down-sampled stratified, or ch-38 synthetic. Training ([[dpo]] + TRL): LR 5e-7 cosine, global batch 32 pairs, 1 epoch, max_length 2048, max_prompt_length 1024, β ∈ {0.05, 0.1, 0.3}. Sweep axis: β only — same seed, same data, same LR across cells per [[karpathy-training-neural-net-recipe]] "one-change-one-prediction".

**Option B (RLVR-math).** Same SFT checkpoint and `π_ref` per [[rlvr-tulu3]]. Data: 5K prompts from `AI-MO/NuminaMath-CoT` or `openai/gsm8k`, bucketed by difficulty. Training ([[grpo]] via TRL + vLLM): LR 1e-6, batch 128 prompts, G=8 rollouts, max_completion_length 1024, clip ε=0.2, rollout T=1.0, β_KL ∈ {0.01, 0.05, 0.1}. Reward = verifier output {0,1} (SymPy-tolerant grader). Sweep axis: β_KL only; Dr.GRPO aggregation (`loss_type="dr_grpo"`) up front per [[dr-grpo]] so the entropy/KL interaction is not confounded by length bias.

## Resource-constrained path

Target: 1 GPU (≥24 GB), Llama-3.2-1B or Qwen2.5-1.5B, ~3 h. One cell: β=0.1 or β_KL=0.05 (middle of the sweep). 2K pairs / 1K prompts. All four signals logged every step. Memo grade is identical — the *analysis* is the deliverable.

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

Log a `length_delta = len(chosen) - len(rejected)` column; [[dpo]]'s length-hacking failure starts in the *data*, not the loss. If chosen is systematically ~80 tokens longer, β=0.05 will inflate length regardless of loss details.

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

Per [[rlvr-tulu3]] "treat verifier engineering like unit-test engineering": unit-test the verifier on 50 hand-curated `(response, gold)` pairs *before* any training — require 100% exact label match. The string-match-accepts-"42"-inside-prose loophole is the attested Tülu 3 hack path.

---

## §2 Training config — real framework calls

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

TRL logs `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `logps/*` — redirect via callback (§3). `rewards/chosen` and `rewards/rejected` are **implicit rewards** per [[dpo]]: `r̂_θ = β log(π_θ / π_ref)`. They are NOT a reward-model score; do not compare absolute values across β cells.

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

`loss_type="dr_grpo"` + `scale_rewards=False` is [[dr-grpo]] ([[trl-grpo]] `_compute_loss` branch ≈ line 2492): drop `/std(r)`, divide token-sum loss by `(B · max_completion_length)` instead of `|o_i|`. With `loss_type="grpo"` the length-inflation failure mode appears within ~200 steps — flipping this flag is the §5(c) post-mortem repro.

---

## §3 Instrumentation hooks — the four signals

Every sweep cell must log the same four signals every 5 steps. [[openrlhf-entropy-debugging]] §Key Points enumerates them as the community-standard metric set; they are what [[trl-grpo]]'s `_metrics[mode]["entropy"]`, TRL `DPOTrainer`'s `rewards/*`, and [[openrlhf-ppo]]'s `PolicyLoss` return `(loss, clip_ratio, ppo_kl, vllm_kl)` already emit.

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

Per-bucket pass-rate (B) requires an eval callback every ~50 steps on a held-out 200-prompt slice with difficulty tag; analogous slice for A is per-source win-rate. Do not skip slicing — [[reward-hacking-taxonomy]]'s enumerated hacks show up as *per-bucket regressions while the aggregate improves*.

---

## §4 Sweep protocol

Write `predictions.txt` before launching:

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

Each cell records wall-clock, peak GPU memory, and total training tokens — the three numbers a reader uses to reproduce you.

---

## §5 Failure-mode diagnostics — pick one

The memo grades on your ability to *name one failure mode and prove it with the metrics you logged*. The three canonical modes, with the signal pattern that identifies each:

**a) Reward hacking ([[reward-hacking-taxonomy]] + [[rlvr-tulu3]] §Failure mode).**
- Signal: reward_mean rises monotonically but per-bucket pass-rate on held-out prompts *declines*, or `len_mean` inflates while verifier rate plateaus.
- Repro: weaken the verifier (e.g. accept `\\boxed{42}` substring instead of SymPy-verify) in a 10% shard; watch the policy discover the loophole within ~200 steps.
- Fix in memo: tighten verifier, or add a sanity check that reward_mean on the 200-prompt held-out slice (scored by the *strong* verifier) tracks training reward within 0.05.

**b) Entropy collapse ([[entropy-mechanism-llm-rl]]).**
- Signal: `entropy` drops below 0.1 nats within the first ~200 updates; reward_mean flattens shortly after; `R = -a·exp(H) + b` fit on the first 10% of steps predicts the eventual ceiling.
- Repro: set β_KL=0.01 (Option B) or β=0.05 (Option A) and remove any entropy bonus; the collapse is near-certain on 3B scale.
- Fix in memo: raise β_KL one notch, or enable `top_entropy_quantile=0.2` in [[trl-grpo]] (keeps gradient on high-entropy tokens only, the DAPO / Muon trick), or cite Clip-Cov / KL-Cov from [[entropy-mechanism-llm-rl]].

**c) Length bias ([[dr-grpo]] Figure 1).**
- Signal: `len_mean` curve is monotone-up across training; reward is flat or slightly up; *wrong* rollouts are disproportionately long (bucket `len_mean` by reward=0 vs reward=1).
- Repro: swap `loss_type="grpo"` for `"dr_grpo"` in Option B at β_KL=0.05; length inflation appears by step ~300.
- Fix in memo: switch to Dr.GRPO aggregation (`loss_type="dr_grpo"`, `scale_rewards=False`) and show the length curve flattening.

You pick ONE. The memo's post-mortem is a full page on that one mode: signal pattern, minimal repro, which line of [[trl-grpo]] / [[openrlhf-ppo]] / [[verl-grpo]] the mechanism lives on, the fix, and a regression test (§7) that would block the bug.

---

## §6 Memo template

`rl-experiment-memo.md`, one page, five sections:

1. **Setup.** Option (A or B), base, data source + size, sweep axis + cell values, GPU count, wall-clock per cell, total tokens.
2. **Sweep table.** Cells × {KL_final, H_final, reward_final, win_rate or pass_rate_final, len_mean_final}. Tag each row "predicted" or "surprising" against `predictions.txt`.
3. **Failure-mode post-mortem.** One full page on the one mode you picked in §5. Signal plot, mechanism (cite the specific raw-data source), minimal repro, fix.
4. **One surprise.** A cell that disagreed with your prediction. Not "noise"; a concrete claim. Example: "β=0.3 still grew length by 15% — I predicted it wouldn't because large-β is supposed to anchor to π_ref. Hypothesis: the preference data itself encodes a length prior that β can't undo."
5. **Next instrumentation.** One metric or assertion you'd add that would have caught the failure one day earlier.

---

## §7 Acceptance criteria

Hard gates, in order.

1. `git log sweeps/` shows `predictions.txt` committed *before* any `metrics.jsonl` file.
2. Each cell's `metrics.jsonl` has all four signals + `len_mean`, ≥ every 10 steps through convergence.
3. Option A: `rewards/chosen - rewards/rejected` is positive and monotone-up for β=0.1. A flat/negative margin means [[dpo]]'s π_ref is wrong (common bug: ref-model forgot `.eval()` or gradients not disabled).
4. Option B: training reward at step 0 ∈ [0.1, 0.4] ([[rlvr-tulu3]] Table 1 baseline range on 3B); 0.0 = broken verifier, ~0.9 = data is trivial.
5. Memo §3 names one mode, cites one raw-data source, includes one plot, proposes one new assertion. No plot = incomplete.
6. At least one memo §2 row tagged "surprising" — a clean sweep teaches nothing.

---

## Connections

- **ch-37 / ch-38** — [[dpo]]'s loss is Option A's objective; β is the sweep axis.
- **ch-39 / ch-40** — [[grpo]] Eq. 3 is Option B's objective; its k3 KL becomes the `kl` signal.
- **ch-41** — vLLM-vs-trainer drift shows up as the `vllm_kl` signal from [[openrlhf-ppo]].
- **ch-42 / [[reward-hacking-taxonomy]]** — §5(a) taxonomy.
- **ch-43 / [[entropy-mechanism-llm-rl]]** — §5(b) mechanism.
- **ch-44** — verifier construction from [[rlvr-tulu3]].
- **ch-45** — `checkpoint-final/` is the starting policy for rejection sampling / self-rewarding.
- **Track 6 (Eval) / ch-47+** — `metrics.jsonl` schema feeds the Eval harness.
- **Track 7 (Infra) / ch-54+** — rollout queue / replay revisits these cells.

## Further reading

- [[dpo]] — Eq. 7, implicit-reward identity, β's KL-budget role, length hack.
- [[grpo]], [[dr-grpo]] — Eq. 3 + group baseline; the length-bias fix.
- [[rlvr-tulu3]] — verifier-as-reward; "verifier engineering like unit-test engineering".
- [[trl-grpo]], [[openrlhf-ppo]], [[verl-grpo]] — three implementations of the same loss.
- [[entropy-mechanism-llm-rl]] — `R = -a·exp(H) + b`; Clip-Cov / KL-Cov.
- [[reward-hacking-taxonomy]] + [[openrlhf-entropy-debugging]] — impossibility theorem; the four signals; triage order.
- [[karpathy-training-neural-net-recipe]] — predict-before-run; "training fails silently" applies to RL.

## Companion visualization

**[figures/rl-sweep.html](figures/rl-sweep.html)** — interactive 3-axis sweep explorer. Pick sweep axis (β for DPO / β_KL for RLVR), pick metric (reward / entropy / KL / pass-rate), and scrub training step. Each curve is illustrative but direction-attested by [[dpo]] β behaviour, [[entropy-mechanism-llm-rl]] collapse law, and [[dr-grpo]] length-inflation shape. Use it *before* writing `predictions.txt`: click through the scenarios, read each justification, then commit your predictions.
