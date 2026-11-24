# 챕터 56 — OpenRLHF 내부: PPO, DPO, Ray Orchestration, Forgetting Control

> **핵심 통찰.** OpenRLHF는 다른 stack이 함께 섞는 세 가지를 분리한다. clipped surrogate만 계산하는 작은 `nn.Module` loss, experience maker에서 per-token reward에 KL-to-reference term을 더하는 reward-shaping step, policy·critic·reference·reward model·vLLM engine을 소유한 Ray actor set이다. commit 64c1cc4에서 KL term은 default로 fixed coefficient 0.01로 reward에 더해지고 `--algo.kl.use_loss`는 이를 actor loss로 옮긴다([[openrlhf-ppo]], `train_ppo_ray.py` L419, L484–486; `ppo_actor.py` L296–314). repository에는 RL에 pretraining/SFT loss를 섞는 mechanism이 없으므로, 노출하는 capability-retention control은 KL term, reference model, in-loop evaluation뿐이다.
>
> **지침.** RL run이 reward domain 밖 capability를 잃지 않아야 한다면 non-zero `--algo.kl.init_coef`로 KL term을 켜고, broad held-out set을 `--eval.dataset`으로 설정해 training 중 `eval_*_pass1`을 log하며 reference KL을 함께 기록한다. training task에서 base policy 대비 KL은 prior-task performance 손실 양을 예측하기 때문이다([[rls-razor]] §4). reward domain만 중요하고 뒤 stage가 breadth를 고칠 예정이면 β를 낮출 수 있으나, 그때 남는 detector는 in-loop broad evaluation뿐이다. KL term이 data mixing을 대신할 것이라 기대하면 안 된다. InstructGPT 1.3B sweep에서 β를 default의 100배인 2.0으로 올려도 public-NLP regression은 복구하지 못했지만 γ ≥ 20으로 pretraining gradient를 섞으면 복구했다([[rlhf-instructgpt]] App. E.6, Figs. 33–34).

---

## 이 장이 범용 모델에 중요한 이유

이 장은 pipeline의 RL stage(pre-training → mid-training → SFT → preference optimization → RL → evaluation)에 있으며 그 stage의 specific implementation 하나를 읽는다. paper objective는 의도를 말하지만 trainer code는 KL term이 advantage 안인지 loss 안인지, negative-advantage token을 bound하는지, 어떤 prompt가 optimizer까지 가는지, run 중 무엇을 측정하는지를 정한다. 그중 세 결정은 task score가 아니라 capability breadth를 바꾸며 OpenRLHF source에서 모두 보인다.

1. **reference KL을 어디에 어떤 coefficient로 넣는가.** loop 안 pre-RL policy에 대한 유일한 explicit anchor다.
2. **어떤 prompt가 optimizer에 도달하는가.** `--algo.dynamic_filtering_enable`은 mean reward가 band 밖인 prompt group을 버린다. 따라서 effective training distribution은 dataset의 moving subset이다.
3. **training 중 무엇을 평가하며 best checkpoint를 어떤 metric으로 고르는가.** OpenRLHF는 loop에서 datasource별 pass@1·pass@n을 계산하고 best-checkpoint tracking용 `eval_*_pass1` key 하나를 자동 선택한다.

ch-55는 같은 question으로 verl을 읽었다. 두 stack은 mechanism보다 default에서 더 다르므로 flag를 하나도 넘기지 않은 run은 framework마다 다른 configuration을 물려받는다.

---

## §1 Repository layout과 composition unit

아래 file tree는 GitHub tree API로 2026-09-14에 읽은 `main` commit b117b2b의 `openrlhf/` package다. 이 장 나머지의 line locus는 source card가 pin한 commit 64c1cc4(2026-04-19)를 가리킨다([[openrlhf-ppo]] Verification).

```
openrlhf/
  cli/        lora_combiner.py  serve_rm.py  train_dpo.py  train_ppo_ray.py  train_rm.py  train_sft.py
  datasets/   prompts_dataset.py  reward_dataset.py  sft_dataset.py  utils.py
  models/     actor.py  loss.py  model.py  ring_attn_utils.py  utils.py
  trainer/    dpo_trainer.py  ppo_trainer.py  ppo_trainer_async.py  rm_trainer.py  sft_trainer.py
    ppo_utils/  experience.py  experience_maker.py  kl_controller.py  length_penalty.py
                replay_buffer.py  samples_generator.py
    ray/        launcher.py  ppo_actor.py  ppo_critic.py  utils.py  vllm_engine.py  vllm_worker_wrap.py
  utils/      agent.py  config.py  deepspeed/  distributed_sampler.py  distributed_util.py
              logging_utils.py  loss_utils.py  math_utils.py  seqlen_balancing.py  utils.py
              vlm_utils.py
```

(`__init__.py`는 생략했고 package 안의 나머지 `.py` file은 모두 열거했다.)

이 layout의 property 네 가지가 이 장 나머지에 중요하다. **training entry point는 넷**이고 그중 RL entry point는 `train_ppo_ray.py`뿐이다. non-Ray PPO script도 package 안 knowledge-distillation trainer도 없다. **`models/loss.py`는 loss module만 둔다.** `PolicyLoss`, `ValueLoss`, `DPOLoss`는 log-probability와 advantage를 받아 scalar를 반환하는 `nn.Module`이며 reference model이나 configuration을 읽지 않는다. **KL controller는 `trainer/ppo_utils/kl_controller.py`에 있다.** experience maker 옆에 scalar `value`를 보유하며 experience maker와 actor가 모두 읽는다. **`trainer/ray/`는 actor class를 둔다.** `ppo_actor.py`는 policy, `ppo_critic.py`는 value, `launcher.py`는 placement group과 reference/reward-model worker, `vllm_engine.py`·`vllm_worker_wrap.py`는 generation이다. PPO job은 Ray actor handle set과 driver loop다.

[figures/openrlhf-ray.html](figures/openrlhf-ray.html) interactive diagram에서 Ray actor를 각각 click해 ownership, 보내는 tensor, 행동을 바꾸는 configuration flag를 볼 수 있다. §3의 KL placement 두 가지도 switch로 바꿀 수 있다.

## §2 `PolicyLoss`: clipped objective와 반환값

**정의.** `PolicyLoss`는 response token 위 PPO clipped surrogate를 계산한다. separate lower·upper clip bound, negative-advantage token용 optional dual clip, vLLM sampler log-probability와 trainer log-probability 차이를 correction하는 optional importance-sampling weight를 지원한다.

**해결하는 문제.** 오래된 policy sample에서 계산한 policy-gradient step은 작은 change에만 유효하다. bound가 없으면 큰 probability ratio와 큰 negative advantage를 가진 token 하나가 ratio에 linear하게 커지는 loss term을 만들어 batch를 지배할 수 있다.

**코드**([[openrlhf-ppo]] Technical Details, 64c1cc4의 `openrlhf/models/loss.py` L136–148, L175–182, 원문):

```python
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
if self.dual_clip is None:
    loss = -torch.min(surr1, surr2)
else:
    clip1 = torch.min(surr1, surr2)
    clip2 = torch.max(clip1, self.dual_clip * advantages)
    loss = -torch.where(advantages < 0, clip2, clip1)
...
loss = (masked_mean(loss, action_mask, dim=None) if self.token_level_loss
        else masked_mean(loss, action_mask, dim=-1).mean())
clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
return loss, clip_ratio, ppo_kl, vllm_kl
```

기호는 다음과 같다. `policy_loss_type="ppo"`에서 `ratio`는 response token별 `r_t = exp(log π_θ(y_t) − log π_old(y_t))`다(L122–124). `"gspo"`에서는 response 전체 masked mean log-ratio의 exponential을 모든 token에 broadcast한다(L125–132). `advantages`는 `A_t`다. `clip_eps_low`, `clip_eps_high`는 `ε_low`, `ε_high`이며 둘 다 default 0.2다(`train_ppo_ray.py` L387–388). `dual_clip`은 c이며 default `None`, 설정하면 > 1.0을 assert한다(L106–107). `action_mask`는 response token을 표시한다.

**계산 예시(advantage sign별 bound).** `ε_low = ε_high = 0.2`, `A = −1`로 둔다([[openrlhf-ppo]] “Findings relevant to negative feedback”).

| r | surr1 = r·A | surr2 = clip(r)·A | loss = −min | r의 gradient |
|---:|---:|---:|---:|---|
| 0.7 | −0.70 | −0.80 | 0.80 | 0(clipped branch) |
| 1.0 | −1.00 | −1.00 | 1.00 | non-zero |
| 1.5 | −1.50 | −1.20 | 1.50 | non-zero, linear 증가 |
| 5.0, c = 3 | −5.00 | −1.20 | 3.00 | 0(dual-clip branch) |

표는 separate bound 둘을 말한다. A < 0에서 r이 `1 − ε_low` 아래로 가면 loss는 constant이고, `dual_clip`이 없으면 위로는 unbounded다. A > 0에서는 역할이 바뀌어 `ε_high`가 update를 bound한다. OpenRLHF는 `dual_clip`을 default off로 둔다. verl은 매 call에 c = 3.0으로 같은 bound를 적용한다([[verl-ppo-loss]] `core_algos.py` L1323–1334, `actor.yaml` L83). 두 repository 모두 c ablation을 보고하지 않는다.

**loss가 반환하는 것.** scalar 넷: loss, `clip_ratio`, `ppo_kl`, `vllm_kl`이다. `clip_ratio`는 양 advantage sign을 합쳐 `surr2 < surr1`인 token 비율이고 dual-clip activation은 세지 않는다(L180). 그래서 `clip_ratio` 상승만으로 positive-advantage token과 negative-advantage token 중 어느 쪽이 clipping됐는지 알 수 없다. `ppo_kl`은 −(log π_θ − log π_old)의 masked mean이다. IS correction이 켜지면 L154가 correction branch 안에서 `log_ratio`를 reassign하므로 이때 `ppo_kl`은 policy-vs-old ratio가 아니라 `vllm_kl`과 같은 quantity를 보고한다([[openrlhf-ppo]] “`ppo_kl` at this commit”). 이는 general fact가 아닌 이 commit의 property다.

---

## §3 Reference KL을 적용하는 위치

**정의.** KL-to-reference term은 policy 자신이 생성한 token에서 frozen reference model `π_ref`(RLHF에서는 보통 SFT checkpoint)로부터 policy가 얼마나 움직였는지 측정한다. OpenRLHF는 flag 하나로 두 위치 중 하나에 적용한다.

**메커니즘, 단계별(reward placement, default).**

1. reference actor가 rollout의 `base_action_log_probs`를 반환한다.
2. `compute_approx_kl`이 log π_θ − log π_ref를 token별 KL estimate로 바꾸고 [−10, 10]으로 clamp한다(`models/utils.py` L48–79).
3. `compute_reward`가 token별 reward를 만든다. 모든 response token에 `−β·kl_t`, 마지막 response token에는 scalar reward(`--reward.clip_range`, default (−10, 10)로 clip)를 쓴다(`models/utils.py` L82–110).
4. configured estimator가 shaped reward로 advantage를 계산한다. `gae`, `reinforce`, `reinforce_baseline`에서는 `--algo.advantage.no_std_norm`이 없으면 batch에서 mean-center하고 batch standard deviation으로 나눈다(`experience_maker.py` L315–328).
5. `PolicyLoss`는 이미 KL cost를 포함한 advantage를 받는다.

```python
# openrlhf/models/utils.py @64c1cc4, compute_approx_kl (verbatim, L62-79)
log_ratio = log_probs.float() - log_probs_base.float()
if kl_estimator == "k1":
    pass  # log_ratio is already p - q
elif kl_estimator == "k2":
    log_ratio = log_ratio**2 / 2.0
elif kl_estimator == "k3":
    log_ratio = (-log_ratio).exp() - 1 + log_ratio
return log_ratio.clamp(min=-10, max=10)
```

`compute_reward`는 이어서 `reward = last_reward + kl_reward`를 쓴다. `kl_reward = -kl_coef * kl`은 모든 response token을 덮고, `last_reward`는 `action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)`가 찾은 final response token으로 scatter한 scalar reward다(`models/utils.py` L92–108).

**메커니즘(loss placement).** `--algo.kl.use_loss`를 쓰면 experience maker는 KL 계산을 완전히 건너뛰고 `kl`을 0으로 만든다(`experience_maker.py` L205–214). actor가 update 중 다시 계산한다.

```python
# openrlhf/trainer/ray/ppo_actor.py @64c1cc4, L296-314 (excerpt)
if self.args.algo.kl.use_loss:
    if self.args.algo.kl.init_coef > 0:
        kl = compute_approx_kl(action_log_probs, base_action_log_probs,
                               kl_estimator=self.args.algo.kl.estimator)
    kl_loss = masked_mean(kl, experience.action_mask)
else:
    kl_loss = 0
loss = actor_loss + kl_loss * kl_ctl
```

**계산 예시.** token 네 개인 response에서 token별 log-ratio `log π_θ − log π_ref = (0.5, −0.2, 0.3, 0.4)`, terminal scalar reward 1.0, `β = 0.01`을 둔다.

- k1 estimate는 (0.5, −0.2, 0.3, 0.4), mean 0.250이다.
- k3 estimate는 `e^{−0.5}−1+0.5 = 0.1065`, `e^{0.2}−1−0.2 = 0.0214`, `e^{−0.3}−1+0.3 = 0.0408`, `e^{−0.4}−1+0.4 = 0.0703`, mean 0.0598이다.
- k1 reward placement의 token별 reward는 (−0.005, +0.002, −0.003, `1.0 − 0.004`)다. `γ = λ = 1`, zero value function이면 첫 token return은 0.99다. 그러므로 모든 token advantage가 KL cost를 운반하며 whole response는 return unit 0.01 penalty를 받는다.
- k1 loss placement는 policy loss에 `0.01 × 0.250 = 0.0025`를 더하고 gradient는 advantage estimator가 아니라 log π_θ를 직접 통과한다.

같은 token에서 estimator 둘은 약 네 배(0.250 대 0.0598) 다르므로 β는 서로 transfer되지 않는다. CLI도 reward placement에는 k1, loss placement에는 k2/k3를 권한다고 출력한다(`train_ppo_ray.py` L675–680).

**Controller.** `AdaptiveKLController`는 target KL을 향해 β를 multiplicative update하고 `FixedKLController.update`는 no-op이다(`kl_controller.py` L4–29). `proportional_error = clip(KL_obs / KL_target − 1, −0.2, 0.2)`, 이어서 `β ← β · (1 + proportional_error · n_steps / horizon)`이다. KL_obs는 batch measured mean KL, KL_target은 `--algo.kl.target`, n_steps는 rollout batch size × samples/prompt, horizon은 `--algo.kl.horizon`(default 10000)이다. controller에 관해 잘못 말하기 쉬운 사실이 둘 있다. 첫째 **adaptive controller는 default가 아니다.** `--algo.kl.target` default `None`은 `FixedKLController(init_coef=0.01)`을 고른다(`ppo_trainer.py` L171–176; `train_ppo_ray.py` L417–419). call site에는 “TODO: KL controller must be FixedKLController; AdaptiveKLController is incompatible here” comment가 있다(`ppo_trainer.py` L252–253). 둘째 **adaptive rule은 InstructGPT 방식이 아니다.** class docstring은 arXiv:1909.08593(Ziegler et al. 2019)를 인용하고, InstructGPT는 모든 RL model에 fixed `β = 0.02`를 썼다([[rlhf-instructgpt]] App. C.4).

**조건과 한계.** OpenRLHF는 이 setting 어느 것에도 experiment를 보고하지 않으며 repository에 training result가 없다([[openrlhf-ppo]] Summary). 위 모든 수치는 measured recommendation이 아닌 default 또는 code behaviour다.

---

## §4 존재하는 forgetting control과 존재하지 않는 control 하나

**측정 가능한 문제.** RL run은 prompt distribution 하나에서 reward 하나를 최적화한다. 위험한 quantity는 그 밖의 모든 것, 즉 RL이 시작하기 전 model이 pass한 benchmark performance다.

**base policy KL이 loss를 예측한다는 근거.** [[rls-razor]]는 Qwen 2.5 3B-Instruct를 세 task(Open-Reasoner-Zero math, SciKnowEval Chemistry L-3 subset, ToolAlpaca tool use)에서 SFT와 binary success reward·explicit KL regularization 없는 GRPO로 fine-tune했다. 이후 HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval을 측정했다(§3.1). new-task score 대 prior-task score Pareto frontier에서 RL run은 prior-task score를 거의 그대로 두는 반면 SFT run은 그것을 교환했다. paper forgetting law는 *new task* distribution τ에서 평가한 base/fine-tuned policy KL `E_{x∼τ}[KL(π₀ ‖ π)]`가 model을 만든 algorithm과 무관하게 forgetting을 예측한다는 것이다. controlled ParityMNIST에서 quadratic fit `R² = 0.96`, LLM experiment에서 `R² = 0.71`이다(§4, Fig. 3, Fig. 11). **결과(단일 연구)**다. 조건은 LLM result의 3B model family 하나, benchmark 여섯, KL term 없는 GRPO다. paper는 explicit KL penalty를 RL에 더하면 같은 curve가 움직이는지 시험하지 않았고, 측정 KL direction `π₀ ‖ π`는 OpenRLHF가 log하는 direction(log π_θ − log π_ref, KL(π_θ ‖ π_ref) estimate)과 반대다. OpenRLHF `kl` metric은 paper quantity가 아니라 correlated proxy로 취급한다.

**OpenRLHF에 없는 control.** InstructGPT objective에는 KL penalty 외 second term이 있다.

```
objective(φ) = E_{(x,y)∼D_{π_φ^RL}}[ r_θ(x,y) − β log( π_φ^RL(y|x) / π^SFT(y|x) ) ]
               + γ · E_{x∼D_pretrain}[ log π_φ^RL(x) ]
```

([[rlhf-instructgpt]] §3, Eq. 2. `π_φ^RL`은 RL policy, `π^SFT` frozen SFT model, `D_pretrain` pre-training distribution, β KL reward coefficient, γ pre-training loss coefficient다.)

implementation은 RL episode보다 8배 많은 pre-training example을 뽑고 PPO/pre-training gradient를 같은 buffer에 consecutive step으로 계산했으며 pre-training gradient를 `γ = 27.8`로 scale했다(App. C.4). sweep에서 `γ ≥ 20`은 validation reward 일부 손실을 대가로 1.3B model SQuADv2·DROP regression을 복구했고 `γ = 27.8`은 1.3B~175B에서 작동했다(App. E.6). 반대로 β를 올려도 안 됐다. default 0.02의 100배인 `β = 2.0`에서도 regression은 지속되고 validation reward는 크게 하락했다(App. E.6, Fig. 34). PPO-ptx에서는 보고된 regression이 시험한 모든 dataset에서 완화됐고 HellaSwag는 GPT-3를 넘었으나 DROP·SQuADv2·translation은 여전히 GPT-3보다 낮았다(§4.2, Fig. 29). **결과(단일 연구)**, 6B reward model로 1.3B–175B에서 얻은 결과다.

64c1cc4의 `cli/train_ppo_ray.py`, `trainer/ray/ppo_actor.py`, `trainer/ppo_trainer.py`에서 `ptx`와 pre-training dataset argument를 grep하면 아무것도 나오지 않는다. OpenRLHF PPO path에는 pre-training/SFT loss-mixing term이 없다. broad해야 하는 run의 결과는 다음과 같다.

1. β와 reference model만 in-loop anchor다. `--algo.kl.init_coef`를 0보다 크게 두고 reference actor를 resident로 유지한다.
2. mixing은 RL loop 밖에서 해야 한다. general data의 별도 SFT pass 또는 pre-RL model과 checkpoint averaging이며 OpenRLHF는 둘 다 automate하지 않는다.
3. in-loop measurement만 남은 detector다. §5가 loss가 발생하는 중 관측할 수 있는 loop 부분을 설명한다.

---

## §5 In-loop evaluation, checkpoint selection, dynamic filtering

**Evaluation option**(`train_ppo_ray.py` L286, L535–540)은 `--eval.steps`(default −1, off), `--eval.dataset`, `--eval.split`(default `train`), `--eval.temperature`(default 0.6), `--eval.n_samples_per_prompt`(default 4)이다. `--eval.dataset`은 `--reward.remote_url`과 함께만 허용된다(L672–673). 따라서 in-loop evaluation에는 remote reward endpoint 또는 held-out prompt를 score할 수 있는 agent function이 필요하다.

**계산하는 것.** `compute_eval_metrics`는 eval sample을 `datasource` field로 group화한다. datasource별로 `eval_<ds>_pass1`, `eval_<ds>_pass<n>`(n sample group의 max reward), `eval_<ds>_response_length_mean`, `eval_<ds>_truncated_rate`를 log하고 overall length·truncation rate·sample count도 log한다(`ppo_trainer.py` L82–144). 이 mechanism이 broad-domain monitoring을 가능하게 한다. datasource value가 다른 held-out domain 여러 개를 eval file 하나에 넣으면 같은 reward axis 위에 각각 pass@1·pass@n curve가 생긴다.

**Checkpoint selection은 narrowing risk다.** `_detect_eval_metric_key`는 explicit `--ckpt.best_metric_key`를 받고, 없으면 metric name을 sorted order로 scan해 `_pass1`로 끝나는 첫 key를 고른다(`ppo_trainer.py` L321–335). datasource가 여러 개면 alphabetically first다. `code`, `math`, `safety`를 평가하는 run은 key를 명시하지 않으면 `code`로 checkpoint를 고른다.

**Dynamic filtering.** default range `(0, 1)`의 `--algo.dynamic_filtering_enable` + `--algo.dynamic_filtering_range`는 group mean score가 open interval 밖이면 prompt의 entire group을 drop하고 dataloader에서 replacement prompt를 dispatch한다(`ppo_utils/samples_generator.py` L169–195). `--reward.remote_url` 또는 agent function, `--rollout.n_samples_per_prompt > 1`이 필요하다(L693–701). 살아남은 prompt fraction은 `dynamic_filtering_pass_rate`로 log한다(`ppo_trainer.py` L542–543).

**계산 예시.** prompt당 n = 8 sample, verifier reward {0, 1}이면 group mean은 correct sample k개에 대해 k/8이다. default range `(0, 1)`은 strict inequality므로 k = 0과 k = 8은 둘 다 drop된다. training 어느 시점에 prompt 30%를 sample 8개 모두로 풀고 20%는 하나도 못 풀면 pass rate는 50%이며 generator는 step당 사용하는 prompt의 약 두 배를 소비한다. training이 진행되면 k = 8 share가 커져 surviving distribution은 current policy가 어렵게 보는 것으로 이동한다. 이는 intended effect다. 모든 group이 non-zero advantage variance를 갖는다. 동시에 mastered prompt를 training mixture에서 제거하는 curriculum이다. mastered slice retention이 중요하면 그 prompt는 더는 gradient를 만들지 않으므로 evaluation으로 확인해야 한다.

---

## §6 DPO: loss, concatenated forward, reference model

**Loss.** `h = (log π_θ(y_c|x) − log π_θ(y_r|x)) − (log π_ref(y_c|x) − log π_ref(y_r|x))`일 때 OpenRLHF `DPOLoss`는 `ℓ = −(1 − ε)·log σ(β·h) − ε·log σ(−β·h)`를 계산한다. `ipo`가 set되면 `ℓ = (h − 1/(2β))²`다([[openrlhf-dpo]], 64c1cc4 `loss.py` L264–281).

```python
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

`y_c`, `y_r`는 chosen/rejected response다. `log π(y|x)`는 prompt token을 mask한 response-token log-probability의 **합**이다(`dpo_trainer.py` L380–386). β는 `--model.beta`(default 0.1), ε는 `--model.label_smoothing`(default 0.0), σ는 logistic function이다. `chosen_rewards`, `rejected_rewards`는 따로 log하는 implicit reward다(L191–192).

**Concatenated forward.** `concatenated_forward`는 chosen/rejected를 common length까지 right-pad하고 batch dimension을 따라 concatenate하므로 call 하나가 둘을 덮는다(`dpo_trainer.py` L305–309, L350–360). docstring의 이유는 FSDP에서 두 forward보다 “faster”이기 때문이다. memory claim은 없으며 함의할 수도 없다. concatenate는 2× batch를 만들므로 backward용 activation은 어느 경우든 forward 둘의 activation이다. 이득은 activation memory 반감이 아니라 collective-heavy forward 하나를 두 번 대신 실행하는 것이다([[openrlhf-dpo]] Verification).

**Total loss와 NLL anchor.** trainer는 optional term 둘을 `preference_loss + aux_loss * aux_loss_coef + nll_loss * nll_loss_coef`로 더한다(`dpo_trainer.py` L157–184). NLL term은 `-all_logps_mean[: chosen_ids.shape[0]].mean()`(L328)이다. 각 chosen response token별 mean log-probability의 negative를 batch 평균한 것으로, sum도 rejected-side term도 아니다. CLI help는 Llama 3.1 technical report를 인용한다(`train_dpo.py` L246). auxiliary term은 MoE router balancing loss이며 default coefficient 0이다(`train_dpo.py` L244). evaluation은 preference loss와 accuracy만 계산하고 optional term 둘을 넣지 않는다(L273–285).

**Reference model.** `train_dpo.py`는 reference path가 없으면 policy path를 reference로 쓴다(L342–343). reference는 `eval()`로 설정되고 forward는 `torch.no_grad()` 아래에서 실행된다(L147, L160–163). reference CPU parameter offload `--ref.offload`는 **default off**다(train_dpo.py L213; `deepspeed.py` L483–495). training이 stage 3이면 reference eval config도 ZeRO stage 3, 아니면 stage 0을 쓴다.

**이 선택이 general benchmark result를 바꾼 근거.** Tülu 3 report는 early SFT checkpoint와 UltraFeedback에서 preference objective·β를 ablate하고 development evaluation average를 보고한다(§5.4.1, Table 18).

| 알고리즘 | LR | γ−β ratio | β | epoch | batch | 평균 score |
|---|---:|---:|---:|---:|---:|---:|
| SFT base(preference training 없음) | — | — | — | — | — | 55.7 |
| SimPO | 5e-7 | 0.5 | 2 | 1 | 128 | 51.8 |
| SimPO | 5e-7 | 0.3 | 10 | 1 | 128 | 52.9 |
| DPO | 5e-7 | — | 0.1 | 3 | 32 | 55.2 |
| PPO | 1e-6 | — | 0.0325 | 1 | 64 | 54.5 |
| PPO | 1e-6 | — | 0.05 | 1 | 64 | 55.5 |
| length-normalized DPO | 1e-7 | — | 5 | 3 | 32 | 56.1 |
| length-normalized DPO | 5e-7 | — | 10 | 3 | 32 | 55.2 |
| length-normalized DPO | 5e-7 | — | 15 | 3 | 32 | 55.7 |
| length-normalized DPO | 5e-7 | — | 2 | 3 | 32 | 46.8 |
| length-normalized DPO | 5e-7 | — | 5 | 3 | 32 | 53.4 |
| length-normalized DPO | 5e-7 | — | 5 | 1 | 32 | 57.3 |

**결과(단일 연구).** 읽어야 할 점 둘이 있다. 첫째, 11 configuration 중 9개가 시작 SFT checkpoint와 같거나 낮고 이를 넘는 것은 둘뿐이다. preference training은 broad-average score에서 monotone하지 않다. report 결론도 “length-normalized DPO만 overall base checkpoint를 이겼다”이다(§5.4.1). 둘째, objective 하나 안 spread(length-normalized DPO 46.8~57.3)가 objective 간 spread보다 크며 β·epoch가 원인이다. 같은 LR, β = 5에서 3 epoch는 53.4, 한 epoch는 57.3이다. length-normalized objective β는 log-ratio를 response length로 나누므로 plain objective β = 0.1과 비교할 수 없다(report Eq. 6). 조건은 base checkpoint 하나, dataset 하나, cell당 seed 하나이며 table caption은 exact evaluation subset을 말하지 않는다.

---
## §7 장문 시퀀스: OpenRLHF의 링 어텐션

**정의.** 링 어텐션은 링으로 배치된 N개의 장치에 하나의 시퀀스를 분할한다. 각 장치는 쿼리 블록 하나를 보유하고, 키-값 블록을 링을 따라 회전시킨다. 이때 다음 블록을 전송하는 작업을 현재 블록의 어텐션 계산과 겹친다([[ring-attention]] §3).

**해결하는 측정 가능한 문제.** 어텐션의 활성화 메모리는 시퀀스 길이에 따라 증가한다. 논문의 표 1은 bfloat16에서 레이어당 최대 활성화 바이트를 다음과 같이 제시한다. 일반 transformer는 2bhs², memory-efficient attention은 8bsh, 링 어텐션은 **6bch**이다. 여기서 b는 배치 크기, h는 은닉 차원, s는 시퀀스 길이, c는 블록 크기다. 링 어텐션 수식에는 s가 없다. 활성화 메모리는 전체 시퀀스 길이가 아니라 장치별 블록 크기에 따라 스케일한다.

**도움이 되는지를 결정하는 제약.** 크기 c인 블록의 블록별 어텐션에는 4dc² FLOPs가 들고, 키와 값 블록을 전송하려면 4cd 바이트가 필요하다. 전송과 계산을 겹치려면 4dc²/F ≥ 4cd/B, 즉 **c ≥ F/B**여야 한다. 여기서 F는 장치당 FLOPS이고 B는 단방향 인터커넥트 대역폭이다(§3.2). 장치당 최소 시퀀스 길이는 s = 6c이다. 논문의 표 2는 다음과 같다.

| 호스트 | FLOPS (TF) | 인터커넥트 (GB/s) | 최소 블록 크기 | 최소 시퀀스 길이 |
|---|---|---|---|---|
| A100 NVLink | 312 | 300 | 1.0K | 6.2K |
| A100 InfiniBand | 312 | 12.5 | 24.5K | 149.5K |
| TPU v4 | 275 | 268 | 1.0K | 6.2K |

**계산 예시.** 32,768 토큰으로 패킹한 훈련 배치와 `--ds.ring_attn_size 8`을 가정하자. 그러면 각 rank는 4,096 토큰을 보유한다. NVLink로 연결된 A100에서는 rank당 블록이 1.0K 최소값을 넘으므로, 키-값 전송은 계산 뒤에 숨는다. 12.5 GB/s의 InfiniBand로만 연결된 A100에서는 최소 블록이 24.5K 토큰이다. 따라서 4,096 토큰 슬라이스는 각 step의 대부분에서 링이 통신을 기다리게 한다. 같은 플래그가 한 클러스터에서는 처리량 이득이고 다른 클러스터에서는 처리량 손실이다. [figures/ring-attention-sharding.html](figures/ring-attention-sharding.html)에서는 인터커넥트, 링 크기, 패킹 길이를 설정하고, 구성이 c ≥ F/B 임계값의 어느 쪽에 있는지와 6bch 활성화 수치를 함께 볼 수 있다.

**OpenRLHF가 구현하는 것** (`openrlhf/models/ring_attn_utils.py`의 64c1cc4, 및 `cli/train_ppo_ray.py` L506–513):

- `--ds.ring_attn_size`(기본값 1, 즉 비활성화)는 링 그룹 크기다. `--ds.ring_attn_head_stride`(기본값 1)는 “매번 링 어텐션을 수행할 헤드 수다. 헤드 수의 약수여야 한다. 값이 클수록 훈련이 더 빠를 수 있으나 메모리를 더 소비한다.”
- 링 어텐션에는 패킹이 필요하다. `--ds.ring_attn_size > 1`인데 `--ds.packing_samples`가 설정되지 않으면, CLI는 "[Warning] --ring_attn_size > 1 requires --packing_samples."를 출력하고 패킹을 활성화한다(L638–641).
- `unpad_and_slice_tensor`는 패딩을 제거하고, 배치를 하나의 `(1, total_seqs)` 시퀀스로 이어 붙이며, 링 크기의 배수가 되도록 패딩한 뒤 각 rank에 연속된 슬라이스를 부여한다. 패딩 길이는 `ring_attn_pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size`다. position id는 패킹된 시퀀스마다 다시 구성하므로, 패킹된 각 예시는 0에서 다시 시작한다.
- `gather_and_pad_tensor`는 rank별 logits 또는 log-probabilities를 all-gather하고, 패딩을 제거하며, 원래의 `(batch, seqlen)` 형태를 복원한다. 따라서 loss 모듈은 샤딩을 보지 못한다.
- 링 크기는 유효 프로세스 수에 곱해진다. CLI는 `n_samples_per_prompt × rollout.batch_size ÷ rollout.micro_batch_size`가 적어도 `actor.num_nodes × actor.num_gpus_per_node ÷ ring_attn_size ÷ tensor_parallel_size`인지 확인한다(L703–709). 배치 배치에는 `duplicate_actors = ring_attn_size × tensor_parallel_size`를 사용한다(L90–139).

**조건과 한계.** 이것은 훈련 측 context-parallel 옵션이다. rollout 길이 한계는 바꾸지 않는다. rollout 길이 한계는 vLLM의 생성 인수와 `--data.max_len`으로 설정되며, 기본값은 프롬프트와 응답을 합쳐 2048 토큰이다(`train_ppo_ray.py` L379). 장문 컨텍스트 RL에는 둘 다 필요하다. 즉, 긴 응답을 허용하는 rollout 예산과 그 응답을 보유할 수 있는 훈련 측 레이아웃이 필요하다.

---
## §8 비동기 rollout, 부분 rollout, 샘플러 드리프트

`--train.async_enable`은 `PPOTrainer`를 `PPOTrainerAsync`로 교체한다. 후자는 `GenerateSamplesActor`와 `TrainingActor`를 동시에 실행한다([[async-rollout]], `ppo_trainer_async.py` L37–350). `rollout_queue`의 크기는 `--train.async_queue_size`(기본값 1)다. 같은 수의 토큰으로 미리 채워진 짝 `rollout_slots` 큐는 생성기가 생성 전에 가져가야 하는 계수 세마포어로 동작한다(L287–297). `asyncio.Lock`를 감싼 Ray actor인 `VLLMLock`은 가중치 broadcast와 생성 사이를 직렬화한다. 따라서 한 배치는 하나의 가중치 집합으로 생성된다(L19–34). `--train.partial_rollout_enable`은 (`train_ppo_ray.py` L277–283에 정의되어 있으며, 해당 위치에서 async 모드가 필요하다고 assert한다, L669–670) lock을 제거한다. 대신 broadcast 전에는 vLLM `pause_generation(mode="keep")`를 호출하고, broadcast 후에는 `resume_generation`을 호출한다(`vllm_engine.py` L132–136). 따라서 진행 중인 샘플 하나는 두 가중치 집합에서 나온 토큰을 포함할 수 있다. verl은 진행 중인 요청을 중단하고 최대 60 s를 기다린 다음 캐시를 비운다([[verl-rollout]] `vllm_async_server.py` L932–1003). 중단과 일시 정지는 서로 다른 트레이드오프다.

importance-sampling 보정(`--algo.advantage.is_correction_enable`, 기본 비활성화, 범위 [0.5, 5.0], 유형 `tis`)이 존재하는 이유는 trainer가 다시 계산한 `old_log_probs`가 vLLM 샘플러가 사용했던 log-probabilities가 아니기 때문이다. `tis`는 w = exp(old_log_probs − rollout_log_probs)를 해당 범위로 clamp하고 loss에 곱한다. `icepop`은 범위 밖 토큰을 0으로 만든다. `seq-mask-tis`는 기하평균 w가 범위 밖인 시퀀스를 버린다(`loss.py` L150–173). `vllm_kl`은 진단값이다([[openrlhf-ppo]] Technical Details).

**조건과 한계.** 저장소는 async 모드의 처리량 또는 수렴 측정치를 보고하지 않는다. OpenRLHF 논문의 arXiv 버전 6개에는 async 대 sync 결과가 없다([[async-rollout]] Verification). 동기 모드가 “더 좋은 안정성”을 가지며 부분 rollout이 “가장 공격적인 off-policy”라는 README의 서술(L671–679)에는 이를 뒷받침하는 실험이 없다.

---
## 부정 샘플과 부정 피드백

이 장의 두 목적함수는 모두 부정을 **그래디언트로** 사용한다(과정 표준의 유형 4). 즉, 특정 샘플의 우도를 명시적으로 낮춘다. 어느 것도 부정을 콘텐츠나 조건부 입력으로 사용하지 않는다.

**부정 샘플의 출처.** PPO와 critic 없는 변형에서 토큰은 반환값이 baseline보다 낮을 때 A_t < 0이다. `gae`에서는 critic의 value, `rloo`에서는 leave-one-out 그룹 평균, `reinforce_baseline`과 `dr_grpo`에서는 그룹 평균, `group_norm`에서는 그룹 표준편차로 나눈 그룹 평균이 baseline이다(`experience_maker.py` L264–270). k1 추정치가 양수인 모든 토큰에서 KL 보상 −β·kl_t 자체가 음수다. 배치 수준 평균 중심화(L315–328)는 원시 보상이 모두 양수여도 `gae`, `reinforce`, `reinforce_baseline`에 음의 advantage를 만든다. DPO에서 부정 샘플은 preference pair를 만든 것이 무엇이든 라벨을 부여한 거부 응답이다. trainer에는 false-negative 추정치가 없고 저장소도 이를 보고하지 않는다.

**메커니즘.** logits z에 대한 target token y의 softmax에서,
∂ log p_y / ∂z_j = 1[j = y] − p_j.
샘플링한 토큰의 우도를 낮추면 해당 토큰에서 확률 질량을 제거하고, 대안들의 현재 확률에 비례하여 재분배한다. 낮추는 토큰이 이미 낮은 확률을 가졌다면, 풀린 질량의 대부분은 이미 가장 가능성 높은 대안으로 간다. 이는 분포를 다양화하는 대신 집중시킨다. 따라서 음의 가지에 대한 상한이 중요하다. A < 0에서 `PolicyLoss`는 r = 1 − ε_low보다 아래에서 그래디언트가 0이고 그 위에서는 loss가 무한정 증가한다. 이를 상한으로 제한하는 것은 `dual_clip`뿐이다(§2).

DPO에서 ℓ를 미분하면 ∂ℓ/∂log π_θ(y_r|x) = β(1 − ε)σ(−βh) − βε σ(βh)이다([[openrlhf-dpo]], L272–275에서 유도). ε = 0이면 이것은 모든 h에서 양수이므로, 모든 step이 거부 응답의 log-probability를 낮춘다. ε = 0.1, β = 0.1이면 부호는 h가 10·ln 9 ≈ 22.0을 초과할 때 한 번만 뒤집힌다. IPO에서는 ∂ℓ/∂log π_θ(y_r|x) = 2(1/(2β) − h)이므로, h > 1/(2β)가 되면 낮추는 힘이 멈추고 반전된다. β = 0.1에서는 5.0이다. 따라서 label smoothing과 IPO는 단지 대안 목적함수가 아니라 음의 그래디언트에 대한 상한이다.

**이 저장소에서 사용할 수 있는 제어값.**

| 제어값 | 플래그 / 코드 | 음의 가지에 미치는 효과 |
|---|---|---|
| Dual clip | `--actor.dual_clip` (기본 비활성화) | A < 0에서 토큰별 loss를 c·\|A\|로 상한 제한한다. r = c보다 위에서 그래디언트는 0이다 |
| 하한 clip 경계 | `--actor.eps_clip_low_high` | ε_low가 A < 0 그래디언트가 꺼지는 위치를 결정한다 |
| IS 범위 | `--algo.advantage.is_correction_*` | 샘플러 불일치가 큰 토큰의 가중치를 낮추거나 버린다 |
| KL 보상 | `--algo.kl.init_coef` | 어느 방향이든 π_ref에서 멀어지는 것에 토큰별 비용을 추가한다 |
| 양의 앵커(DPO) | `--model.nll_loss_coef` | chosen 응답에만 cross-entropy 압력을 추가한다 |
| Label smoothing / IPO | `--model.label_smoothing`, `--model.ipo_enable` | 위에서 설명한 대로 거부 측 그래디언트를 상한 제한한다 |

**진단값과 그 한계.** DPO는 `chosen_reward`와 `reject_reward`를 별도로 기록한다. 따라서 두 암묵적 보상이 함께 하락하는 실행, 즉 margin은 개선되지만 두 log-probabilities 모두 낮아지는 실행은 보인다. 그러나 원시 chosen 및 rejected log-probabilities는 기록하지 않는다([[openrlhf-dpo]]). PPO는 `ppo_clip_ratio`, `ppo_kl`, `kl`, `logprobs_diff`를 기록하고, IS 보정이 켜져 있으면 `vllm_kl`도 기록한다(`ppo_actor.py` L291–310). `clip_ratio`는 advantage 부호별로 나누지 않고 dual-clip 활성화도 제외한다. 따라서 어느 가지가 포화되는지 알고 싶으면 부호로 나눈 clip 비율을 직접 추가해야 한다.

**효과 크기에 관한 정직성.** OpenRLHF 저장소에는 실험이 없으므로, 어떤 개선 중 얼마가 음의 advantage 토큰 또는 거부 항에서 오는지에 관한 증거도 제공하지 않는다. 부정 샘플이 개선의 대부분을 이끈다는 주장은 이 장의 출처 어디에서도 뒷받침되지 않는다.

---

## 레시피

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| OpenRLHF 기본값(모델 없음) | — | RL | KL 계수 β (`--algo.kl.init_coef`) | 0.01 | `cli/train_ppo_ray.py` L419 @64c1cc4 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | KL 배치 위치(`--algo.kl.use_loss`) | `False` → KL을 토큰별 보상에 추가 | `train_ppo_ray.py` L484–486; `experience_maker.py` L205–214 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | KL 추정기(`--algo.kl.estimator`); clamp | `k1`; [−10, 10] | `train_ppo_ray.py` L421–429; `models/utils.py` L79 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | KL controller(`--algo.kl.target`; horizon) | `None` → `FixedKLController`; 10000 | `train_ppo_ray.py` L417–418; `ppo_trainer.py` L171–176 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | clip ε; dual clip | 0.2; `None` (비활성화, > 1.0이어야 함) | `train_ppo_ray.py` L387–389; `loss.py` L106–107 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | entropy 계수 | `None` (entropy 항 없음) | `train_ppo_ray.py` L431–436 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | advantage 추정기; 배치 정규화 | `gae`; `--algo.advantage.no_std_norm`이 없으면 평균 중심화 후 std로 나눔(gae, reinforce, reinforce_baseline) | `train_ppo_ray.py` L477–483; `experience_maker.py` L315–328 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | eval steps; eval temperature; 프롬프트당 eval 샘플 | −1 (비활성화); 0.6; 4 | `train_ppo_ray.py` L286, L535–540 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | dynamic filtering; 범위 | 비활성화; (0, 1) | `train_ppo_ray.py` L562–567 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | ring attention 크기; head stride | 1 (비활성화); 1 | `train_ppo_ray.py` L506–513 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | 최대 전체 시퀀스 길이(`--data.max_len`) | 2048 토큰(프롬프트 + 응답) | `train_ppo_ray.py` L379 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF 기본값(모델 없음) | — | RL | async 활성화; 큐 크기; 부분 rollout | `False`; 1; `False` | `train_ppo_ray.py` L274–283 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF `train_dpo.py` CLI 기본값 | 모든 값 | preference | preference loss; β; label smoothing; NLL 계수 | sigmoid DPO; 0.1; 0.0; 0 | `train_dpo.py` L237–247 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| OpenRLHF `train_dpo.py` CLI 기본값 | 모든 값 | preference | reference 경로; reference offload | 설정하지 않으면 policy 경로; 비활성화 | `train_dpo.py` L342–343, L213 | 2026-09-14 검증됨 | 보고된 ablation 없음 |
| InstructGPT (PPO-ptx, 모든 크기) | 1.3B–175B | RL | KL 보상 계수 β; 배치 위치 | 0.02; 보상의 토큰별 항 | arXiv:2203.02155 App. C.4 | 2026-09-17 검증됨 | App. E.7 Fig. 36: 사람 Likert 최적점은 약 0.01–0.02 |
| InstructGPT (PPO-ptx, 모든 크기) | 1.3B–175B | RL | pre-training loss 계수 γ; pre-training 예시 | 27.8; RL episode 수의 8× | arXiv:2203.02155 App. C.4 | 2026-09-17 검증됨 | App. E.6 Fig. 33: γ ≥ 20은 1.3B에서 SQuADv2와 DROP을 복구; Fig. 34: β를 2.0으로 올려도 복구하지 못함 |
| InstructGPT (PPO-ptx, 모든 크기) | 1.3B–175B | RL | clip ε; rollout temperature; episodes; batch / minibatch; 내부 epochs | 0.2; 1.0; 약 31k개의 고유 프롬프트에 걸친 256k episodes; 512 / 64; 1 | arXiv:2203.02155 App. C.4 | 2026-09-17 검증됨 | 보고된 ablation 없음 |
| Tülu 3 ablation checkpoint (초기 Tülu 3 SFT 모델, UltraFeedback) | 해당 위치에 명시되지 않음 | preference | 길이 정규화 DPO: β; epochs; LR; batch | 5; 1; 5e-7; 32 | Tülu 3 report §5.4.1 Table 18 | 2026-09-17 검증됨 | Table 18: SFT base의 55.7 대비 평균 57.3; β = 2는 46.8 |

**소규모 범용 RL 실행의 시작점.** 여기의 모든 수치는 위 `verified` 행에서 왔다. clip ε(0.2)와 KL 추정기(k1, 보상 배치 위치)는 OpenRLHF 기본값을 유지하고, `--algo.kl.init_coef`는 0이 아닌 기본값 0.01로 설정한다. verl이 기본값으로 사용하는 것처럼 음의 advantage 가지에 상한을 두려면 `--actor.dual_clip 3.0`을 설정한다. checkpoint 선택이 알파벳순으로 결정되지 않도록, 여러 `datasource` 값을 가진 `--eval.dataset`과 명시적 `--ckpt.best_metric_key`도 설정한다. 이는 실험을 보고하지 않는 저장소의 프레임워크 기본값이다. 이 값들을 택한 조건은 문서화되지 않았으므로, 결과가 뒷받침하는 권고가 아니라 측정할 시작 구성이다. InstructGPT의 β = 0.02와 γ = 27.8은 31k-프롬프트 instruction 분포에서 6B reward model을 사용한 1.3B–175B GPT-3 계열 모델에서 나온 값이며, γ에 해당하는 OpenRLHF 항은 없다.

---

## 일반화 관점

**(a) 폭을 넓히는 것.** KL 항을 활성화한 채 유지하면 루프가 pre-RL policy에 고정점을 갖게 된다. 그리고 그 항이 제어하는 양은 [[rls-razor]]에서 forgetting을 예측하는 양이다(§4, LLM 실험에서 R² = 0.71, 통제된 설정에서 R² = 0.96). 측정된 곳에서는 RL update에 pre-training 또는 SFT 그래디언트를 섞는 것이 더 큰 β보다 잘 작동했다. γ ≥ 20은 1.3B에서 SQuADv2와 DROP을 복구한 반면 β = 2.0은 그러지 못했다([[rlhf-instructgpt]] App. E.6). OpenRLHF는 이 혼합을 구현하지 않으므로 이 제어는 프레임워크 밖에서 적용해야 한다. 여러 `datasource` 값을 사용하는 in-loop 평가는 폭을 실행 종료 후에 발견하는 일이 아니라 모니터링하는 양으로 만든다(`ppo_trainer.py` L82–144).

**(b) 좁아짐을 일으키는 것.** `--algo.kl.init_coef 0`으로 설정하면 루프 안의 유일한 앵커가 제거된다. Dynamic filtering은 숙달된 프롬프트와 풀지 못한 프롬프트를 훈련 혼합물에서 제거하므로, 유효 분포는 policy의 현재 frontier 쪽으로 좁아진다. best-checkpoint 자동 탐지는 override하지 않으면 하나의 datasource의 pass@1을 기준으로 선택한다. DPO 쪽에서는 β와 목적함수 변형이 Tülu 3 ablation에서 넓은 평균을 10점 이상 움직였고, 11개 구성 중 9개는 시작 SFT checkpoint와 같거나 그보다 낮았다(Table 18). 앵커 없는 음의 그래디언트는 이미 가장 가능성 높은 대안에 확률 질량을 집중시킨다(부정 샘플 섹션 참조). 이 때문에 pass@1이 올라가도 출력 다양성은 줄어든다.

**(c) 이 단계에서 측정하는 방법.** 보상 도메인 밖의 도메인을 적어도 하나 포함하여 `eval_<ds>_pass1`과 `eval_<ds>_pass<n>`를 기록한다. pass@n이 의미 있으려면 `--eval.n_samples_per_prompt`를 1보다 크게 설정한다. 모든 step에서 `kl`과 `logprobs_diff`를 기록하고, step 수가 아닌 KL에 대해 held-out 점수를 그린다. KL은 공개된 근거가 있는 예측변수이기 때문이다. DPO에서는 `chosen_reward`와 `reject_reward`를 따로 기록하고 둘 다 함께 하락하는지를 본다. filtering이 켜져 있다면 `dynamic_filtering_pass_rate`를 추가한다. pass rate가 떨어지면 훈련 분포가 이동하고 있다는 뜻이다.

**알려진 측정 오류.** 이 commit에서 IS 보정이 켜져 있으면 `ppo_kl`은 policy-vs-old 비율이 아니라 sampler-drift 양을 보고한다. `clip_ratio`는 두 advantage 부호를 섞는다. `eval.split` 기본값은 `train`이므로, train split도 포함한 eval 파일은 플래그를 설정하지 않으면 잘못된 행에서 평가된다. temperature 0.6에서의 in-loop pass@1은 benchmark harness의 greedy 수치와 비교할 수 없다.

---

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| adaptive KL controller가 활성화되어 있다고 가정 | KL이 target에서 크게 벗어나도 β가 움직이지 않음 | `--algo.kl.target`은 기본값으로 `None`이며 `FixedKLController`를 선택한다. step마다 `kl_ctl.value`를 출력한다 |
| k1용으로 조정한 β를 k3와 함께 `--algo.kl.use_loss`로 옮김 | 같은 rollout에서 KL penalty가 약 4× 약해짐(§3 계산 예시) | 전환 전에 한 배치에서 두 추정기를 모두 기록한다. CLI의 k1 대 k2/k3 권고를 따른다 |
| `concatenated_forward`가 DPO 활성화 메모리를 절반으로 만든다고 기대 | 절반이라면 허용했을 배치 크기에서 out-of-memory | concatenation은 2× batch를 만든다. `micro_batch_size`를 줄이거나 `--ref.offload`를 활성화한다 |
| 대형 DPO 실행에서 `--ref.offload`를 끈 채 둠 | no-grad forward 중 reference model out-of-memory | `--ref.offload`는 기본으로 꺼져 있다(`train_dpo.py` L213) |
| 어느 advantage 부호가 clipping되는지 보려고 `clip_ratio`에 의존 | `clip_ratio`는 오르지만 원인을 알 수 없고 dual-clip 활성화는 보이지 않음 | A ≥ 0과 A < 0에 대해 clip fraction을 별도로 계산한다 |
| 저대역폭 인터커넥트에서 ring attention 활성화 | `ring_attn_size`에 따라 step 시간이 줄지 않고 늘어남 | rank당 슬라이스 길이를 [[ring-attention]] Table 2의 c = F/B와 비교한다 |
| 패킹 없이 ring attention 활성화 | 조용한 구성 변경 | CLI가 "[Warning] --ring_attn_size > 1 requires --packing_samples."를 출력하고 패킹을 켠다 |
| 여러 eval datasource에서 best-checkpoint 선택을 자동 탐지에 맡김 | 저장된 “best” checkpoint가 하나의 도메인을 추적 | `--ckpt.best_metric_key`를 명시적으로 설정한다 |
| dynamic filtering을 순수 효율 기능으로 해석 | held-out 폭이 악화되는 동안 `dynamic_filtering_pass_rate`가 훈련에 따라 하락 | 살아남은 혼합물뿐 아니라 필터가 제거한 프롬프트에서도 평가한다 |
| async 안정성에 관한 README 문장을 측정 결과로 취급 | 측정되지 않은 주장으로 의사결정 | [[async-rollout]] Verification: 저장소나 논문의 6개 버전 어디에도 async 대 sync 결과가 없다 |

---

## 이해도 점검

1. 기본 구성은 KL 항을 loss에 더하는 대신 모든 응답 토큰의 보상에 −β·kl_t를 더한다. 이것이 log π_θ에 도달하는 그래디언트를 어떻게 바꾸는지, 그리고 CLI가 각 배치 위치에 서로 다른 KL 추정기를 권하는 이유를 설명하라.
2. A < 0에서 PPO loss는 r = 1 − ε_low보다 아래에서 상수이고 r = 1보다 위에서 선형으로 증가한다. `surr1`/`surr2` 코드에서 두 진술을 유도하고, `dual_clip`만이 상단을 제한하는 이유를 설명하라.
3. [[rls-razor]]는 새 작업 분포에서 KL(π₀ ‖ π)을 측정하는 반면, OpenRLHF는 rollout에서 KL(π_θ ‖ π_ref)의 추정치를 기록한다. 기록한 metric이 논문의 양이 아니라 proxy인 이유와, 논문의 양을 기록하려면 무엇을 계산해야 하는지 설명하라.
4. InstructGPT의 sweep은 γ ≥ 20이 public-NLP 퇴행을 복구했지만 β = 2.0은 그러지 못했다고 밝혔다. 각 항이 어느 분포를 제약하는지 관점에서, 이를 설명하는 두 항의 메커니즘상 차이를 설명하라.
5. 기본 범위와 이진 보상을 사용하는 dynamic filtering은 모든 샘플이 정답인 그룹과 어느 샘플도 정답이 아닌 그룹을 버린다. 이것이 실행 과정에서 훈련 분포를 어떻게 바꾸는지 설명하고, 버려진 프롬프트가 잊히는지 드러낼 측정을 설계하라.
6. Tülu 3 ablation에서 β = 2의 길이 정규화 DPO는 46.8점이었고, 같은 목적함수에서 β = 5 및 한 epoch는 57.3점, SFT base는 55.7점이었다. 낮은 β가 더 나쁜 넓은 평균을 만들 수 있는 이유와, 최종 평가 전에 그 현상을 확인하려면 실행 중 무엇을 기록할지 설명하라.
7. InfiniBand로 연결된 A100에서 한 실행이 32,768 패킹 토큰과 `--ds.ring_attn_size 8`을 설정한 뒤 step 시간이 증가했다. c ≥ F/B를 사용하여 원인을 설명하고, 각각 이 문제를 해결하는 구성 변경 두 가지를 제시하라.
8. `concatenated_forward`는 2× batch에 대해 하나의 forward pass를 수행한다. 이것이 메모리 논거가 아니라 속도 논거인 이유를 설명하고, docstring이 공으로 돌리는 distributed-training 메커니즘의 이름을 쓰라.

---

## 연결

- **이전 — ch-55: verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation.** verl의 `use_kl_loss` 대 `use_kl_in_reward`는 `--algo.kl.use_loss`와 같은 전환이다. 3.0인 verl의 dual-clip 기본값은 OpenRLHF의 `None`과 대비된다.
- **다음 — ch-57: TRL Internals: SFT, DPO, GRPO, and Distillation Trainers.** ch-55 및 이 장과 대조하여 읽을 동일 목적함수의 세 번째 구현이다.
- **ch-58: Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability.** §4와 §5는 이 장 feature matrix의 두 행을 제공한다. loss mixing(여기에는 없음)과 in-loop broad evaluation(여기에는 있으나 checkpoint-selection 주의사항 포함)이다.
- **ch-38: KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax.** §3과 §4가 코드로 추적하는 두 항의 목적함수다.
- **ch-38a: SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.** [[rls-razor]]를 포함한 retention 근거를 그 자체로 다룬다.
- **ch-39: Offline Preference Optimization: DPO and Its Variants.** 우도 변위와 거부 측 그래디언트를 제한하는 앵커를 포함하여 §6 loss 뒤의 유도식을 다룬다.
- **ch-43: Entropy, Output Diversity, and KL Control in RL.** 알고리즘 질문으로서 k1/k2/k3의 편향과 분산, KL 배치 위치를 다룬다.
- **ch-50: Slice Analysis, Forgetting Slices, and Failure Bucketing.** §5가 제공하는 datasource별 metric을 어떻게 사용할지 다룬다.
- **ch-05: Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen.** 다른 context-parallel 옵션 중 링 어텐션과 `ring_attn_size`가 들어가는 world-size 산술을 다룬다.

---

## 출처

- [[openrlhf-ppo]] — commit 64c1cc4에서 `PolicyLoss` 코드, KL 추정기와 보상 shaping, controller 기본값, advantage 추정기와 배치 정규화, IS-correction 변형을 다룬다.
- [[openrlhf-ppo-recipe]] — Recipe 표에서 인용한 framework-default 행이다.
- [[openrlhf-dpo]] — `DPOLoss`, train step, `concatenated_forward`, NLL 및 MoE auxiliary 항, reference-model 처리, 유도한 rejected-side 그래디언트를 다룬다.
- [[openrlhf-dpo-recipe]] — DPO CLI 기본값, `DPOTrainer` constructor 기본값, 예시 Llama-3-8B script 값을 다룬다.
- [[async-rollout]] — async trainer의 두 actor, 큐와 slot 쌍, `VLLMLock`, vLLM pause/resume을 통한 partial rollout, 처리량 측정 부재를 다룬다.
- [[verl-ppo-loss]] — verl의 dual-clip 기본값 3.0과 §2의 대비를 다룬다.
- [[verl-rollout]] — verl의 abort-and-clear weight-sync 경로와 §8의 대비를 다룬다.
- [[rlhf-instructgpt]] — Equation 2, β = 0.02의 토큰별 KL penalty, γ = 27.8의 PPO-ptx, App. E.6–E.7의 γ와 β sweep을 다룬다.
- [[rls-razor]] — §4와 Generalization lens에서 사용한 경험적 forgetting 법칙과 SFT-versus-RL retention 비교를 다룬다.
- [[ring-attention]] — 표 1의 activation 수치, c ≥ F/B overlap 조건, host별 표 2의 최소 블록 및 시퀀스 길이를 다룬다.
- [[dpo]] — §6의 코드가 구현하는 DPO 목적함수, 암묵적 보상, 그래디언트를 다룬다.
- [[tulu-3]] — preference-tuning ablation(§5.4.1, Table 18), β와 목적함수 변형이 넓은 평균을 움직인다는 근거를 다룬다.
- [[openrlhf-entropy-debugging]] — OpenRLHF, verl, TRL 전반의 entropy와 KL 문제에 대한 실무자 triage 노트다. 출처 신뢰도: 일화적(framework README, issue tracker, community digest)이며, 이 장에서는 그 정량적 주장을 증거로 사용하지 않는다.
