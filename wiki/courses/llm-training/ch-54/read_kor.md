<!-- chapter: ch-54
     track: infra
     kind: content
     title: Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments
     deps: [ch-53, ch-43a, ch-45b]
     sources: [[best-of-n]], [[on-off-policy-rlhf]], [[async-rlhf-noukhovitch]], [[areal-async-rl]], [[async-rollout]], [[verl-rollout]], [[openrlhf-ppo]], [[verl-ppo-loss]], [[rollout-training-mismatch-tis]], [[glm-5]], [[deepseek-v3.1]], [[minimax-forge-agent-rl]], [[replay-buffer-rlhf]], [[kimi-k1-5]], [[deepswe]], [[r2e-gym]], [[qwen3-coder]], [[skyrl-agent]], [[asearcher]], [[agent-lightning]], [[prime-intellect-environments-hub]], [[topr-tapered-off-policy-reinforce]], [[asymmetric-reinforce]], [[rls-razor]], [[retaining-by-doing]], [[rlvr-beyond-base-model]]
     figures: figures/async-pipeline.html
     revised: 2026-09 (generality revision)
-->

# Chapter 54 — Rollout 인프라: Off-policy 데이터, 비동기 RL, Agentic 환경

> **핵심 통찰.** RL run에서 policy-gradient 수식은 engineering의 작은 부분이다. generation이 wall-clock time을 지배한다. verl의 colocated 128-GPU DAPO baseline은 356.30 s step 중 177.85 s를 generation에 쓰며, 같은 run을 disaggregated 64:64 rollout/train layout으로 옮기면 400 step이 1 d 16 h 48 m에서 17 h 22 m으로 줄었다([[verl-rollout]], `fully_async_policy/README.md` L333–361). generation을 비동기로 만들면 이전 weight version이 만든 trajectory로 학습하며, 그 staleness는 측정·제어할 수 있다. [[areal-async-rl]]의 1.5B math ablation에서 naive PPO는 staleness η = 0의 AIME24 42.0에서 η = 4의 23.3으로 떨어지지만, decoupled PPO objective를 쓴 같은 run은 42.2를 유지한다(§7.4, Table 2). 두 번째 독립적인 off-policyness는 같은 weight에서도 sampler와 trainer가 같은 token에 다른 probability를 부여하는 데서 온다([[rollout-training-mismatch-tis]]).
>
> **지침.** generation과 training이 GPU를 공유하고 step이 가장 긴 response를 기다리느라 지배된다면, explicit maximum-staleness parameter가 있는 decoupled layout으로 옮겨라. [[areal-async-rl]]은 matched 또는 더 높은 final accuracy에서 end-to-end time 2.77× 감소를 보고한다(§7.2, Table 1). staleness가 training step 하나를 넘으면 cap을 올리기 전에 objective를 바꿔라. decoupled PPO objective는 η = 8까지 synchronous oracle의 AIME24 score에서 1점 이내를 유지했으나, 없는 같은 run은 최대 18.7점을 잃었다(§7.4, Table 2). rollout과 gradient가 다른 backend에서 오면 sampler log-probability를 log하고 truncated ratio를 적용하라. [[rollout-training-mismatch-tis]]는 Qwen2.5-32B의 DAPO에서 token-probability difference 최대 1.0을 측정했다. trajectory가 sandbox 안에서 실행되면 모든 sample의 failure reason을 기록하고 environment crash를 reward에서 제외하라. 그렇지 않으면 model failure로 학습에 들어간다고 [[glm-5]]가 보고한다(§4.1.2). 반대로 한 node의 짧은 synchronous run이고 generation이 step time의 30% 미만이면 synchronous loop를 유지하고 evaluation에 노력을 쓰라.

---

## 범용 모델에 이 장이 중요한 이유

지금까지 pipeline은 pre-training → mid-training → SFT → preference optimization → RL → evaluation이다. 이 장은 RL stage 안에 있지만 objective에 관한 장은 아니다. track 앞부분의 policy-gradient, preference, negative-gradient 장이 이를 다룬다. 이 장은 objective에 들어갈 data가 어디서 오고 얼마나 오래됐으며 어떤 sample을 유지하고 어떤 물리적 system이 만드는지에 관한 장이다.

이것이 breadth에 중요한 이유는 세 가지다.

1. **Data source는 prior ability가 얼마나 살아남는지 바꾼다.** [[rls-razor]]는 empirical forgetting law를 보고한다. fine-tuning 뒤 catastrophic forgetting 양은 **새** task에서 측정한 fine-tuned policy와 base policy 사이 KL `E_{x∼τ}[KL(π₀‖π)]`로 예측되며, controlled MNIST에서 quadratic fit R² = 0.96, LLM experiment에서 R² = 0.71이다(§4, Fig. 3, Fig. 11). [[retaining-by-doing]]은 8B까지 Llama·Qwen model의 instruction following, general knowledge, arithmetic에서 같은 ordering을 보고한다. comparable target-task performance에서 RL은 SFT보다 덜 잊고, 저자는 그 차이를 on-policy data에 귀속한다(Abstract, Fig. 2). Rollout infrastructure가 data가 실제로 얼마나 on-policy인지 결정한다.
2. **이 장의 모든 throughput mechanism은 filter이기도 하다.** Staleness cap, prompt replay, dynamic filtering, scheduler window는 어떤 prompt와 trajectory가 gradient에 도달할지를 정한다. throughput을 위해 고른 filter도 curriculum이며, curriculum은 training distribution을 좁힐 수 있다.
3. **Agentic RL은 execution environment를 loop 안에 넣는다.** Sandbox가 crash하면 reward signal에 model과 무관한 term이 들어간다. environment failure와 model failure를 분리하는 것은 이 stage에만 있는 data-quality 문제다.

[figures/async-pipeline.html](figures/async-pipeline.html)을 §4와 함께 읽어라. staleness cap과 rollout/trainer speed ratio를 설정하고, 생기는 queue depth·generator lead 및 그 cap에서 측정한 [[areal-async-rl]] accuracy를 확인할 수 있다.

---

## §1 Rollout loop와 시간이 쓰이는 곳

**정의.** *Rollout*은 generation pass 하나다. prompt가 inference engine으로 가고 engine은 token id와 token별 log-probability를 돌려주며, reward function 또는 verifier가 result를 score하고 trainer가 advantage를 계산해 gradient step을 취한다. *Synchronous* arrangement에서는 process 하나가 같은 GPU에서 두 phase를 번갈아 수행한다. *Decoupled* 또는 *asynchronous* arrangement에서는 generator와 trainer가 별도 resource에서 동시에 돌며 queue로 연결된다.

**측정 가능한 문제.** synchronous arrangement에서 trainer는 generation 중 idle이고 generation은 batch에서 가장 긴 response가 끝날 때까지 끝날 수 없다. 독립 source 두 곳이 효과 크기를 보고한다.

- [[verl-rollout]]: colocated 128-H20 baseline(Qwen2.5-Math-7B, DAPO, max response length 28K)에서 step은 356.30 s이고 그중 `gen`은 177.85 s, 즉 49.9%다(`fully_async_policy/README.md`@753aed3 L360). rollout GPU 64개와 training GPU 64개, `partial_rollout=True`, `staleness_threshold=0.5`로 재구성하면 400 step은 1 d 16 h 48 m 대신 17 h 22 m, 2.35배가 걸린다. 마지막 checkpoint AIME-2024 acc/mean@1은 0.2958 대비 0.3094다(L333–361, L369). **결과(단일 연구).**
- [[areal-async-rl]]: 16 node에서 1.5B R1-Distill-Qwen model의 PPO 250 step은 AReaL 14.8 training hour, 자체 synchronous variant 41.0 hour였고 AIME24는 42.2 대 42.0이다. 7B에서는 25.4 대 57.7 hour, 63.1 대 63.0으로 end-to-end 최대 2.77×다(§7.2, Table 1). **결과(단일 연구).** Interruptible generation만의 ablation은 4 node에서 generation throughput 12%(1.5B), 17%(7B) 증가를 보인다(§7.5, Fig. 6b).

**범용 모델에 대한 함의.** Throughput은 cost만의 질문이 아니다. [[asearcher]]는 초기 online-RL search agent의 10 이하 turn limit이 algorithm이 아니라 긴 trajectory가 synchronous batch를 막았기 때문이라고 보고한다. execution과 update를 decouple해 trajectory당 128 turn limit이 가능해졌고, training 중 agent는 trajectory 하나에서 tool call 100회 초과와 generated token 400k 초과에 도달했다(§1, Fig. 1). long horizon에서만 나타나는 behavior는 long horizon을 감당할 수 없는 system으로는 학습할 수 없다.

---

## §2 Best-of-N: 학습 없는 baseline과 그 ceiling

**정의.** Best-of-N(BoN)은 고정 policy에서 response N개를 sample하고 reward model로 각각 score한 뒤 최고 score 하나를 반환한다. 아무것도 학습하지 않는다.

**Infrastructure 장에 속하는 이유.** 현재 policy를 resample하는 것만으로 target reward 중 얼마가 이미 도달 가능한지 측정한다. gradient step, trainer, rollout-training loop가 필요 없으므로 어떤 RL infrastructure가 존재하기 전에도 실행할 수 있다.

**수식.** [[best-of-n]]은 sampling policy로부터 BoN policy의 analytic divergence를 다음처럼 제시한다(Table 10 caption; App. G.3 fn. 15).

```
KL(BoN ‖ base) = log N − (N − 1)/N
```

`N`은 prompt당 sample 수다. 이 식은 tie가 없다고 가정하며 reward model quality를 포함해 다른 어떤 것에도 의존하지 않는다.

**계산 예시.** N = 8이면 `ln 8 = 2.0794`, `7/8 = 0.875`이므로 KL = 1.204 nat이다. N = 64이면 `ln 64 = 4.1589`, `63/64 = 0.9844`, KL = 3.174다. N = 512이면 `6.2383 − 0.998 = 5.240`이다. 논문 Table 10은 같은 N에 1.2, 3.2, 5.2를 인쇄하므로 산술이 source와 맞는다.

**근거 및 한계.** 같은 appendix에서 저자들은 “at a given average reward, the best-of-N and PPO policies have similar quality as judged by human labelers (not shown)”라고 쓴다. PPO policy는 KL에서 supervised policy와 더 멀다. `KL(ppo, sup)`은 1.3B에서 18.0, 6.7B에서 14.0이고 BoN-64는 3.2다(App. G.3; Tables 9–10). **결과(단일 연구)**이며 N별 human 수치는 인쇄하지 않는다. setting은 2020년 Reddit TL;DR summarization과 1.3B·6.7B model이므로 reasoning 또는 agentic task로는 재측정 없이 전이하지 않는다.

**Ceiling.** [[best-of-n]] §4.3(Fig. 5)는 reward model에 대한 optimization이 증가하면 human preference가 먼저 오르고, 이후 reward model이 labeler와 anti-correlated될 때까지 떨어진다고 보고한다. N sample의 argmax 자체가 reward model에 대한 optimization이므로 BoN도 그 ceiling을 물려받는다. N을 높인 gain은 RL도 over-optimize할 동일 proxy에 대한 gain이다. reward-hacking 처리는 ch-42를 보라.

---

## §3 On-policy 대 off-policy data: 실제로 측정한 것

**이 장이 하지 않는 주장.** online-versus-offline gap의 고정 분해(예: distribution shift에 약 80%를 귀속)는 때때로 Tang et al. (2024)에 돌려진다. 그 논문에는 그런 분해가 없다. Abstract는 반대로 offline data coverage와 data quality가 각각 그 gap을 설명하지 못한다고 한다([[on-off-policy-rlhf]], Abstract; §5.1–§5.2).

**[[on-off-policy-rlhf]]가 보고한 것.** Tang et al. (2024)은 loss(IPO), hyper-parameter, source preference data를 고정하고 response가 오는 곳만 바꾼다. T5X policy 770M, 3B, 11B와 네 dataset(OpenAI summarization, Anthropic helpfulness, Chat arena side-by-side, Anthropic harmlessness)에서 다음을 보고한다.

- On-policy sampling은 네 dataset 모두에서 고정 golden policy 대비 더 높은 peak win rate에 도달한다(§2.1, Fig. 1).
- Data coverage는 설명하지 못한다. online run 자체 sample의 shuffled stream으로 offline 학습해도 Chat arena sxs 외에는 plain offline보다 개선이 작다(§5.1, Fig. 4).
- Data quality도 설명하지 못한다. 4k-step online policy에서 sample해 golden model로 다시 label한 pair의 offline training은 SFT보다 조금만 개선한다(§5.2, Fig. 5).
- Offline training은 policy를 **더 좋은 pairwise classifier이면서 더 나쁜 generator**로 만든다. offline policy는 preference set에서 classification accuracy 약 70%로 peak하지만 online policy는 50% 아래에 머물고, offline run 안에서 classification accuracy는 win rate와 상관이 작다(§5.3, Figs. 7–8).
- Preferred response의 log-probability는 training 중 SFT 값 아래로 내려가며 offline에서 가장 강하다(§5.3.3, Fig. 8 bottom).

**상태.** 각 항목은 **결과(단일 연구)**다. 논문은 PPO나 DPO experiment를 하지 않았으므로 PPO 대 DPO 근거는 아니다.

**[[async-rlhf-noukhovitch]]가 더한 것.** Noukhovitch et al. (2024)은 staleness를 직접 측정한다. minibatch `N`개를 생성한 뒤 minibatch update `N`개를 하므로 update 1은 on-policy, update `N`은 `N−1` step stale이다. Pythia 410M policy와 gold reward model의 TL;DR에서 PPO win rate는 N = 1에서 최고이고 N이 커지며 떨어진다. dropoff는 logarithmic이어서 N = 1과 N = 2는 거의 같다(§3.2, Fig. 3). 모든 N은 같은 win-rate-versus-KL Pareto curve 위에 있으므로 staleness는 frontier를 옮기기보다 frontier를 따라가는 progress를 늦춘다. N ∈ {1, 2, 4, 8, 16}에서 loss를 비교하면 PPO는 on-policy에서 최고지만 빨리 저하되고 RLOO도 그렇다. Online DPO가 가장 robust하며 N = 64에서도 계속 학습하는 유일한 method다(§3.3, Fig. 4).

**조건.** 410M policy, task 하나, reward model 하나다. 같은 논문의 8B chatbot run은 staleness sweep이 아니라 throughput 결과다. synchronous와 asynchronous Online DPO는 No Robots에서 둘 다 GPT-4o win rate 57.20%에 도달하고 compute time은 230 대 142다(§4, Table 1).

**범용 모델에 대한 함의.** staleness로 저하되는 양은 generation quality이고 개선되는 양은 discrimination-like score다. reward만 log하면 stale run은 generation이 나빠지는 동안 멀쩡해 보일 수 있다.

---

## §4 Asynchrony: 설정하는 양으로서의 staleness

### 4.1 Staleness cap

[[areal-async-rl]]은 staleness를 trajectory generating policy가 current policy보다 뒤처질 수 있는 최대 training step 수인 explicit hyper-parameter η로 만든다. current policy version `i`, 누적 generated trajectory `N_r`, prompt `B`개인 training batch size에서 새 generation request는 다음 동안에만 받아들인다(§5.1, Eq. 3).

```
⌊(N_r − 1) / B⌋ ≤ i + η
```

`η = 0`이면 system은 synchronous RL로 줄어든다. batch를 assemble할 때 buffer의 오래된 trajectory를 먼저 쓴다.

**계산 예시.** `B = 512`, `η = 8`이고 trainer가 `i = 100` update를 마쳤다고 하자. constraint는 `⌊(N_r − 1)/512⌋ ≤ 108`이므로 `N_r ≤ 55,808 = 109 × 512`이다. generator는 trainer보다 최대 `η + 1 = 9` batch 앞설 수 있으며 그 밖의 request는 trainer가 전진할 때까지 거부된다.

### 4.2 Cap과 함께 objective도 바뀌어야 한다

η만 높이면 run이 저하된다. 논문이 든 수정 이유는 작은 η가 아주 긴 trajectory 몇 개가 in flight일 때마다 generation을 stall시켜 throughput이 큰 η 쪽으로 밀기 때문이다(§5.1). 답은 token을 sample한 *behavior* policy `π_behav`와 trust-region center인 *proximal* policy `π_prox`를 분리하는 decoupled PPO objective다(§5.2 Eqs. 4–5).

```
J(θ) = E_{q∼D, a_t∼π_behav} Σ_t (π_prox/π_behav) · min( u_t(θ)·Â_t , clip(u_t(θ), 1−ε, 1+ε)·Â_t )
u_t(θ) = π_θ(a_t | s_t) / π_prox(a_t | s_t)
```

`Â_t`는 token t의 advantage, `ε`는 clip range이고 `π_prox`는 current update 직전 parameter로 설정되며 global batch가 올 때 token probability를 다시 계산한다. `π_behav`를 trust-region center로 쓰면 최신 policy가 오래되고 더 낮은 quality policy 쪽으로 끌린다는 것이 논문의 주장이다(§5.2). Proposition 1은 여러 version에서 token이 온 sequence가 single behavior policy sampling과 동등함을 보여 interrupted generation을 유효하게 한다(§5.2, §D proof).

**측정.** 1.5B model, math, 250 step에서 decoupled objective 없음/있음(§7.4, Table 2):

| 최대 staleness η | AIME24 없음 | AIME24 있음 | MATH-500 없음 | MATH-500 있음 |
|---|---|---|---|---|
| 0 (oracle) | 42.0 | 42.0 | 89.2 | 89.2 |
| 1 | 41.8 | 42.1 | 89.9 | 89.8 |
| 2 | 40.0 | 41.8 | 89.6 | 89.6 |
| 4 | 23.3 | 42.2 | 66.9 | 89.5 |
| 8 | 35.7 | 41.0 | 87.8 | 89.2 |
| 16 | 35.8 | 38.7 | 87.4 | 89.1 |
| ∞ | 34.0 | 36.9 | 87.1 | 88.1 |

같은 sweep에서 effective training throughput은 η = 0의 128.7k tokens/s에서 η = ∞의 396.8k로 오른다(§7.4, Fig. 5c). 같은 논문의 production default는 coding η = 4, math η = 8이다(§7.1). decoupled objective 없는 η = 4 행은 η = 8에 대해 non-monotone이며 논문은 설명하지 않는다. 개별 행은 모두 seed 1 고정의 한 seed로 다뤄라(§B).

### 4.3 Framework가 cap을 구현하는 법

- **[[async-rollout]] (OpenRLHF).** `GenerateSamplesActor`와 `TrainingActor`는 동시에 돌고 `rollout_queue = Queue(maxsize=async_queue_size)` 및 같은 수 token으로 미리 채운 counting semaphore 역할의 `rollout_slots` queue로 연결된다(ppo_trainer_async.py@64c1cc4 L287–297). 기본 queue size는 1이다(`train_ppo_ray.py` L274–283). Enqueue payload에는 **policy-version field가 없고 어느 loop도 sample age를 filter하지 않는다**(L149–151, L227–232). README는 더 큰 queue가 “more off-policy”이고 async training이 “may affect convergence”한다고 말하며, repository는 이 mode의 throughput·convergence 측정을 보고하지 않는다(README L665–679; arXiv:2405.11143 v1–v6에서 확인).
- **[[verl-rollout]] (`fully_async_policy`).** 별도 GPU의 rollouter와 trainer, `staleness_threshold`(release experiment 0.5, README L380–396에서 0 / 0.1 / 0.3 / 0.5 ablation), `partial_rollout=True`를 사용한다. 이는 parameter sync 전 generation을 interrupt하고 뒤에 interrupted sample을 resume한다(L55–58, L222–226, L351–352).
- **[[glm-5]].** 각 response는 segment를 만든 model version sequence `(w₀, …, w_k)`를 기록한다. current version `w′`에서 `w′ − w₀ > τ`이면 sample을 discard한다(§4.1.2). 이는 admission-control rate limit이 아니라 trajectory별 filter이며, 여러 weight update에 걸친 multi-turn agent rollout에 필요하다. inference engine에 weight를 push할 때마다 optimizer를 reset한다(§4.1.1).
- **[[minimax-forge-agent-rl]].** Forge는 scheduler 자체를 staleness control로 본다. Strict FIFO는 한 slow trajectory의 head-of-line blocking을 받고, 먼저 끝난 것을 greedy fetch하면 training stream이 “initially dominated by short, easy tasks and later by clustered hard ones”가 된다(§1.2). *Windowed FIFO*는 trainer가 sliding window `[T_i, T_{i+W−1}]` 안의 completed trajectory를 가져오게 한다(예: `W = 0.3N`). 그 밖의 completed task fetch를 막고 head task가 소비될 때만 window가 전진한다(§3.1). 이는 latency control만이 아니라 distribution control이다.

[figures/async-pipeline.html](figures/async-pipeline.html)은 네 mechanism을 나란히 보여 주고 staleness setting별 측정 AReaL accuracy를 보고한다.

---

## §5 Sampler–trainer mismatch: 두 번째 off-policyness 원천

Staleness는 configured cap이 도입하는 off-policyness다. η = 0에도 존재하고 설정하지 않는 두 번째 원천은 generation backend와 training backend가 같은 token에 같은 probability를 부여하지 않는 것이다.

**측정.** [[rollout-training-mismatch-tis]]는 동일 parameter에서도 sampler vLLM과 learner FSDP가 같은 token에 다른 probability를 준다고 보고한다. Qwen2.5-32B DAPO에서 maximum per-token difference는 1.0에 도달한다. 즉 `π_vllm = 1`, `π_fsdp = 0`인 token이 있고, 작은 GSM8K example에서는 약 0.4다(Fig. 1; “Normal RL training”). vLLM patch 두 개(sample에 실제 쓴 probability 반환, `lm_head`를 fp32로 cast) 뒤에도 gap은 남는다.

**수정.** Truncated importance sampling은 policy-gradient term에 cap을 씌운 learner/sampler ratio를 곱한다(formula paragraph).

```
E_{a∼π_sampler(θ)}[ min(π_learner(a, θ)/π_sampler(a, θ), C) · R(a) · ∇θ log π_learner(a, θ) ]
```

`C`는 truncation cap이다. **Source의 variance argument에서의 계산 예시:** ratio 16 token에서 un-truncated importance sampling은 gradient noise를 16² = 256×, `C = 2` TIS는 4×, `C = 8` TIS는 64× 증폭한다. 공개 script는 32B DAPO run에 `C = 8`, 0.5B GSM8K run에 `C = 2`를 사용한다(recipe ledger, `dapo_qwen32b_bf16.sh` L5, L106; `gsm8k_qwen0_5b_int8.sh` L2, L32).

**보고한 효과.** DAPO-32B의 INT8 rollout에서 entropy는 0.2 아래로 떨어져 계속 감소하고 response는 비정상적으로 길어지며 k1 KL estimator는 negative value를 자주 log한다. TIS를 넣으면 세 가지 모두 되돌아간다(Figs. 5–6). Logged reward는 `E_{π_sampler}[R]`이므로 TIS 없는 BF16 run은 TIS run보다 *더 높은* reward를 log하면서 AIME는 더 낮다(Fig. 6). 이는 training-reward metric과 held-out metric이 불일치하는 직접 예다. gap이 작을 때(양쪽 BF16 rollout의 DeepSeek-R1-Distill-Qwen-1.5B)는 TIS gain이 없다(Fig. 3). Factor analysis에서 sampler·learner parallelism 차이는 maximum mismatch를 높인다(learner Ulysses SP8 대 sampler TP2는 0.5 초과 response의 두 자릿수 count, Fig. 7). 긴 uninterrupted response도 높이며(20K 대 4K cap, Fig. 9), 어떤 sampler backend도 일관되게 낫지 않다(Fig. 11).

**구현.**

- verl: `imp_ratio = exp(old_log_prob − rollout_log_probs)`를 lower bound 없이 `max=imp_ratio_cap`에서 clamp하고 token별 loss에 곱한다(blog pinned commit의 `core_algos.py` L586–590). 현재 verl feature는 token- 또는 sequence-level `rollout_is_weights`이며 YAML default `rollout_is: null`은 off다([[verl-ppo-loss]], `config/algorithm/rollout_correction.yaml`; token·sequence option은 `config/algorithm.py` L85–89).
- OpenRLHF: type `tis`(`[low, high]`로 clamp), `icepop`(out-of-band token을 zero), `seq-mask-tis`가 있는 `--algo.advantage.is_correction_enable`; default threshold `[0.5, 5.0]`, default off; diagnostic은 `vllm_kl = masked_mean(rollout_log_probs − old_log_probs)`다([[async-rollout]], `loss.py` L150–173).
- [[glm-5]]: asynchronous setting에서는 trajectory 하나 안에서 rollout engine이 여러 번 update할 수 있어 정확한 `π_θ_old` track에는 checkpoint history가 필요하다. GLM-5는 rollout log-probability를 behavior proxy로 직접 써서 `r_t(θ) = exp(log π_θ(a_t|s_t) − log π_rollout(a_t|s_t))`로 만들고, ratio가 `[1 − ε_ℓ, 1 + ε_h]` 밖인 token을 gradient에서 완전히 mask한다(§4.1.2, Eqs. 3–5).

**MoE는 이를 악화시키며, fix는 routing replay다.** [[deepseek-v3.1]](DeepSeek-V3.2 report)은 inference에서 고른 expert route를 training pass에 재사용하는 *Keep Routing*, sampling top-p/top-k truncation mask를 `π_θ`에 적용하는 *Keep Sampling Mask*를 GRPO stabilizer 네 개 중 둘로 열거한다. Keep Routing은 MoE model의 RL training stability에 “crucial”하고 DeepSeek-V3-0324부터 쓰였다고 한다(§3.1). verl은 `enable_rollout_routing_replay`에서 vLLM server가 token id와 함께 routed expert를 반환하게 하는 plumbing을 제공한다([[verl-rollout]], `vllm_async_server.py` L716–718).

**Token-in-token-out.** 마지막 mismatch는 text다. [[verl-rollout]] agent-loop 문서는 final message history에 chat template을 다시 적용하면 tool parser가 assistant content를 rewrite하고 decode-encode가 항상 invertible하지 않아 per-turn prompt·response id의 concatenation과 다른 token id가 생긴다고 보고한다. 그러면 “make[s] PPO training not even converged in single-turn”이다(`docs/advance/agent_loop.rst` L155–182). [[glm-5]]는 같은 design을 TITO라 부르며 generation request를 intercept해 token id·metadata를 기록하는 gateway를 둬 re-tokenization을 막는다(§4.1.2). rollout이 multi-turn이거나 tool call을 쓰면 rendered text가 아니라 engine이 반환한 token id에서 training sequence를 만들라. verl은 text path가 single-turn에서도 non-convergence를 낸다고 보고한다.

---

## §6 Reuse 및 selection: curriculum으로서의 replay, prioritization, filtering

**Replay라고 부르는 서로 다른 두 가지.**

1. *Trajectory replay*는 저장된 completion을 나중 gradient step에서 재사용한다. current policy와 해당 token을 만든 policy의 importance ratio를 보정해야 하며, 이는 §4·§5와 같은 machinery지만 gap이 더 크다.
2. *Prompt replay*는 prompt를 다시 선택하고 current policy에서 fresh completion을 생성한다. old token이 gradient에 들어가지 않아 importance correction은 필요 없지만 prompt distribution은 더 이상 dataset의 distribution이 아니다.

**Library card에 대한 주의.** [[replay-buffer-rlhf]]는 primary artifact 하나의 extract가 아니라 framework synthesis다. 2026-09 audit는 명시한 locus에서 구체 claim 두 개를 확인하지 못했다. TRL experimental replay trainer를 prompt-only라고 기술한 것과 DeepSeek-R1 “§3.2”의 trajectory replay negative result다(R1 §3.2는 distilled-model evaluation이며 실패 시도 §4.2는 PRM·MCTS). 이 card는 두 claim의 근거가 아니라 design space의 pointer로 써라.

**증명된 reuse mechanism은 partial rollout이다.** [[kimi-k1-5]]는 rollout마다 fixed output-token budget을 주며 unfinished trajectory를 replay buffer에 쓰고 다음 iteration에서 이어 간다. 따라서 current segment만 on-policy computation이 필요하고 segment를 loss에서 제외할 수 있다(§2.6.2). 발견된 repetition은 일찍 끝내고 penalty를 줄 수 있다. [[verl-rollout]]의 `partial_rollout=True`는 system level에서 같은 생각이다. parameter sync 전 interrupt하고 뒤에 resume한다(`fully_async_policy/README.md` L55–58). 이 mechanism은 long-tail latency 문제를 controlled multi-version trajectory로 바꾸며, [[areal-async-rl]] Proposition 1은 이를 single behavior policy로 취급할 근거를 준다.

**Dynamic filtering.** [[rollout-training-mismatch-tis]] script의 release DAPO recipe는 accuracy로 group을 filter하고 training batch를 채우기 위해 generation batch를 최대 10회 regenerate한다(step당 prompt 512, generation batch prompt 1,536, prompt당 sample 16; `dapo_qwen32b_bf16.sh` L28–35). 모든 sample이 성공하거나 모두 실패한 group은 zero group-relative advantage를 주므로 버린다. [[deepswe]]의 *compact filtering*은 max context, max environment step, generation timeout 20분으로 끝난 trajectory의 loss를 mask한다. agent가 우연히 모든 test를 통과할 수 있고 그런 trajectory에 reward를 주면 Qwen3-14B ablation에서 collapse했기 때문이다(§2.3, Fig. 6; 효과는 수치 없이 curve로 보임).

**이 모든 것이 curriculum인 이유.** Accuracy filter는 intermediate difficulty prompt를 유지한다. [[replay-buffer-rlhf]]가 TRL experimental GRPO replay trainer에 대해 기술한 group-reward variance 가중 prioritized prompt replay도 같은 이유로 같은 prompt를 유지한다. 2026-09 audit는 명시 locus에서 이 trainer 설명을 확인하지 못했으므로 attested implementation이 아니라 design pattern으로 취급하라. Windowed FIFO([[minimax-forge-agent-rl]] §3.1)는 window가 허용한 task만 유지한다. 모두 assembled dataset보다 effective training distribution을 좁히며, 어느 source도 drop한 domain에 무슨 일이 생기는지 보고하지 않는다. Counter-practice 두 report는 직접 말한다. [[deepseek-v3.1]]은 domain별 sequential stage 대신 reasoning·agent·human-alignment training을 **하나의** RL stage에 합치고(§3), [[minimax-forge-agent-rl]]은 reasoning·general QA·agent task를 동시에 섞으며 sequential training이 domain 간 negative transfer를 만든다고 한다(§4.1). Filter 또는 scheduler window가 gradient에 닿는 prompt를 고르면, filter가 닿지 못하는 domain별 regression set을 hold out하라. 여기 cite한 source 중 자기 filter가 제거한 domain에 대한 효과를 측정한 것은 없다.

---

## §7 Loop 안의 environment: agentic rollout infrastructure

rollout이 multi-turn agent이면 generator는 inference server와 execution environment pool을 합친 것이다. 아래 report의 pool size는 concurrent Docker container 512개부터 parallel environment 20,000개까지여서 environment capacity는 GPU capacity와 나란한 system requirement가 된다.

**보고된 규모.**

| 시스템 | Environment 규모 | 위치 |
|---|---|---|
| [[deepswe]] (Qwen3-32B) | RL iteration당 Docker container 512개(batch 64 × pass 8); H100 64개로 6일 | blog §2.2, intro |
| [[r2e-gym]] | commit에서 만든 executable problem 8,135개; SWE-Bench repository overlap 없는 4,578-problem subset | §2, Table 1 |
| [[qwen3-coder]] (480B-A35B) | 독립 environment 20,000개 병렬 | blog, “Scaling Long-Horizon RL” |
| [[glm-5]] | real-world software-engineering environment 10K 초과, terminal·multi-hop search domain 추가; concurrent rollout 1k 초과 | §3.3, §4.1.1 |
| [[asearcher]] (QwQ-32B) | trajectory당 128 turn; training 중 tool call 100회 초과, generated token 400k 관측 | §1, Fig. 1 |

**Orchestration.** [[deepswe]]는 container 수천 개를 실행하면 Docker daemon이 crash했다고 보고해 scheduling을 Kubernetes로 옮겼다. worker node마다 CPU core 약 200개와 preloaded SWE-bench image가 있는 local NVMe 6 TB 초과가 있고, cluster는 CPU core 1000개를 넘게 scale하며 autoscaler는 약 20분 underutilized node를 제거한다(§2.2). Image caching과 node-local storage가 이 rate에서 environment reset을 충분히 싸게 만든다.

**Long-tail latency.** [[minimax-forge-agent-rl]]은 agent rollout completion time이 “from seconds to hours”라고 말한다(§1.2). 공개된 답은 세 가지다.

1. Interrupt·resume — partial rollout([[kimi-k1-5]] §2.6.2, [[verl-rollout]] `fully_async_policy`) 및 AReaL interruptible rollout worker(§5.2, §7.5).
2. Decode path를 uninterrupted로 유지 — [[glm-5]]는 prefill과 decode를 dedicated resource에서 실행해 prefill burst가 long-horizon decode를 stall하지 못하게 하며, 이를 multi-turn agentic RL tail behavior의 fix라고 한다(§3.6.2).
3. CPU-bound stage overlap — [[skyrl-agent]]는 rollout을 runtime initialization, agent run, reward calculation으로 나누고 stage마다 bounded queue를 둔다. *Async Pipeline*은 이를 GPU generation과 overlap해 naive asynchronous batching보다 1.55× speedup을 낸다(§3.2, Fig. 1b, Fig. 3).

**Model 것이 아닌 failure.** [[glm-5]]는 every sample의 failure reason을 기록하고 environment collapse sample을 제외한다. 이는 “reflect environment instability rather than the model's capability”이기 때문이다. Group-based method에서는 incomplete group이 남고 valid sample이 절반 초과면 valid sample 반복으로 pad, 아니면 drop한다(§4.1.2). [[deepswe]] compact filtering은 exhausted budget에 대해 같은 일을 한다. “sandbox died”, “tool timed out”, “context exhausted”, “model produced a wrong patch”는 네 distinct label로 다뤄라. 마지막만 training signal이다.

**Interface.** 세 design이 반복된다.

- **Gym-style wrapping.** [[skyrl-agent]] agent loop는 OpenAI-style function call로만 행동한다. `step()`을 tool로 wrap해 Gym environment를 더하고 dataset마다 자기 tool·verifier를 bind하므로 job 하나에 task type을 섞을 수 있다(§3.1, Listing 1).
- **Masked sequence 하나 대신 transition record.** [[skyrl-agent]]는 input token, output token, log-probability가 있는 every LLM call을 기록하고 backend-agnostic format으로 바꾼다. Prefix를 공유하는 transition은 mask로 pack하며 context가 바뀌지 않으면 concatenate-and-mask로 줄어든다(§3.3). [[agent-lightning]]도 action 하나는 LLM call 하나이고 episode 각 call은 episode final return을 받으며 grouped transition에 single-turn RL method를 적용한다(§3.3.2). concatenate-and-mask를 반대하는 stated reason은 mask가 training을 agent logic에 couple하고 RoPE의 token-continuity assumption을 깨며 긴 sequence를 만든다는 것이다(§3.3.2, §5.1). 이는 측정이 아닌 이유다. 논문은 training curve만, numeric table·masked multi-turn training 비교는 없다.
- **Agent-side decoupling.** [[agent-lightning]]의 Training-Agent Disaggregation은 RL framework 옆 OpenAI-like API server와 agent 안 client를 둔다(§3.4.1). [[minimax-forge-agent-rl]]은 이를 더 밀어 agent가 gateway로 request를 route하고 framework는 internal에 agnostic하므로 context compression, history rewriting, multi-agent loop에 training-loop 변경이 필요 없게 한다(§2.3).

**Environment가 오는 곳.** [[r2e-gym]] SWE-Gen은 test를 collect·generate하고 code change를 problem statement로 back-translate해 commit에서 executable environment를 만든다. 따라서 human-written issue가 필요 없다(§2). [[prime-intellect-environments-hub]]는 environment를 trainer가 native로 소비하는 자체 evaluation report가 있는 shareable artifact로 보며 sandbox는 같은 verifier interface에 plug한다. 이 post는 measurement 없는 announcement이므로 interface claim만, quantity는 뒷받침하지 않는다.

**가져갈 가치가 있는 efficiency 결과.** [[skyrl-agent]]는 DeepSWE가 사용한 동일 R2E-Gym instance 4.5K에서 Qwen3-32B의 SA-SWE-32B를 RL만으로 학습해 SWE-Bench Verified Pass@1 39.4%를 H100 hour 4,601에 달성했다. 같은 simple ReAct evaluation에서 DeepSWE는 36.4%, H100 hour 9,180이었다(§4.3, Table 2). 논문은 차이를 objective가 아니라 tool design과 dispatcher에 귀속하며, breadth 측면에서 SWE-only 학습 model이 Terminal-Bench, BrowseComp-Plus, WebArena도 개선했다고 보고한다(Abstract, Table 3).

---

## 음성 샘플과 음성 피드백

**“Negative”의 어느 의미인가.** 이 절은 ch-43a taxonomy의 sense 4인 **gradient로서의 음성**, 즉 `Â < 0`이며 update가 likelihood를 낮추는 token 또는 sequence에 관한 것이다. Environment crash는 음성이 아니라 corrupted label이며 §7에서 처리한다.

**여기서 negative advantage가 나오는 곳.** Group-relative baseline은 mixed group의 약 절반을 construction상 음수로 만든다. [[deepswe]]의 leave-one-out advantage에서 rollout n = 8, success 2개이면 failure는 `0 − 2/7 ≈ −0.29`, success는 `1 − 1/7 ≈ 0.86`이다. 여덟 개가 모두 fail하면 모든 advantage는 0이다(stated component에서 card가 도출; post는 formula를 인쇄하지 않음).

**Stale negative가 불안정한 이유.** PPO clipped surrogate에서는 advantage sign 두 개가 asymmetrically clip된다. [[openrlhf-ppo]](`loss.py` L136–141)에서 도출하면 `Â < 0`의 objective는 `Â · max(r, 1 − ε_low)`이므로 `r < 1 − ε_low`이면 gradient가 0이지만 **r이 증가할수록 loss는 bound 없이 선형 증가한다**. `Â > 0`은 `Â · min(r, 1 + ε_high)`이고 위로 bounded다. stale 또는 mismatch negative sample은 r이 클 수 있는 경우이므로 staleness가 치는 쪽은 unbounded side다.

**Control 1 — dual clip.** `Â < 0`의 token별 loss를 `c·|Â|`로 cap하면 `r > c`부터 gradient가 0이다([[verl-ppo-loss]] `core_algos.py` L1323–1361, arXiv:1912.09729 인용). `ε_low = ε_high = 0.2`, `Â = −1` 예시에서 r = 0.7이면 loss는 `max(0.7, 0.8) = 0.8`이고 r에 flat하다. r = 1.5이면 loss 1.5에 gradient가 흐르고, r = 5.0, c = 3이면 loss는 3으로 cap되고 gradient 0이다. verl default `clip_ratio_c = 3.0`이고 OpenRLHF `dual_clip` default는 `None`이라 설정 전에는 bound가 off다. 두 project 모두 c ablation은 보고하지 않는다.

**Control 2 — asymmetric importance sampling.** [[topr-tapered-off-policy-reinforce]]는 negative ratio를 `[0, 1]`로 truncate하고 positive weight는 1로 둔다.

```
α = clip(π(y|x)/µ(y|x), 0, 1)  if R(x, y) < 0  else  1
ℓ = stop-grad(α) · r(x, y) · log π(y|x)
```

`µ`는 behavior policy다. 저자 표현으로 두 term은 “SFT update for positive examples” + “TIS update for negative examples”다(§3, Eq. 8; Fig. 1 right). negative weight에 positive floor가 있으면 naive REINFORCE처럼 결국 degeneracy가 생기므로 lower limit을 정확히 0으로 둔다(§2.1, §3). GSM8K Llama 3 8B에서 off-policy가 커질수록 naive REINFORCE는 저하되고 PPO는 개선을 멈추지만 TOPR는 계속 개선한다(Fig. 1 left; figure는 table 아닌 curve를 인쇄).

**Control 3 — ratio 대신 baseline 이동.** [[asymmetric-reinforce]]는 importance correction 없이 tunable baseline의 `A = r − V`를 쓴다. Theorem 4.2는 phase transition을 준다. `V < V^µ`(behavior policy expected reward)이면 limit policy는 µ보다 개선하고 wide support를 유지하며, `V ≥ V^µ`이면 support가 “suddenly shrinks”해 일반적으로 한 element가 된다. Llama-3.1-8B-Instruct, MATH, ±1 reward, prompt당 G = 8 sample, behavior policy 250 step마다 refresh에서 `δV = 0`의 7 run 전부가 train·test accuracy에서 collapse했고, 저자는 `δV = −0.1`의 7 run이 더 stable하다고 보고한다(§5.2, Fig. 5 left). stale data에서는 per-prompt mean reward보다 baseline을 약간 **낮게** 잡아 positive를 negative보다 크게 weight하라는 권고다.

**Control 4 — sample drop.** [[deepseek-v3.1]]은 negative이고 far off-policy일 때 sequence 전체를 mask한다. `Â_{i,t} < 0` **및** `(1/|o_i|) Σ_t log(π_old/π_θ) > δ`일 때 `M_{i,t} = 0`이며 `π_old`는 inference engine이 반환한 probability다(§3.1, Eq. 9). [[glm-5]]는 clipping 대신 rollout ratio가 `[1 − ε_ℓ, 1 + ε_h]` 밖인 token을 양쪽 tail 모두 mask한다(§4.1.2, Eq. 5). [[rollout-training-mismatch-tis]]는 AReaL decoupled PPO가 truncation 대신 ratio threshold 초과 sample 전체를 drop한다고 말한다.

**한 번만 서술하는 메커니즘.** Vocabulary softmax에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. token `y`를 누르면 probability mass는 현재 가장 likely한 alternative로 가며 그 alternative가 더 나을 필요는 없다. sample이 stale이면 push-down은 current policy가 더 이상 갖지 않는 probability에 대해 계산된다. [[rollout-training-mismatch-tis]]는 sampler gap의 구체 예를 든다. negative-advantage rollout에서 `π_learner/π_sampler < 1`이면 `π_learner` 감소가 `π_sampler`에는 나타나지 않아 update가 learner를 계속 누른다. 저자는 exact mechanism은 열린 질문이라고 하면서 이것을 entropy collapse 설명으로 제시한다(**해석**; mechanism paragraph).

**진단.** Advantage sign별로 mean ratio, clip fraction, dual-clip activation count를 따로 log하라. [[openrlhf-ppo]] `clip_ratio` metric은 두 sign을 함께 세고 dual-clip activation은 세지 않아(L180) 어느 side가 saturate하는지 알 수 없다. `vllm_kl`, token entropy, 큰 k의 pass@1·pass@k, 각 filter가 drop한 sample share를 log하고 “environment failure”, “budget exhausted”, “model failure”로 분리하라.

**효과 크기에 대한 정직성.** 이 장 source 중 asynchronous run improvement에서 negative sample의 몫을 측정한 것은 없다. 위 control은 stability result이지 attribution result가 아니다. positive/negative split claim은 측정하는 ch-43a에 속한다.

---

## 레시피

값은 각 locus에서 인용했다. Framework default, released config, published model run은 별도 사실이며 label을 붙였다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| AReaL, R1-Distill-Qwen | 1.5B–32B | RL | max staleness η | 4(coding); 8(math) | arXiv:2505.24298v5 §7.1 | verified 2026-09-17 | §7.4 Table 2 sweep, seed 1 |
| AReaL, R1-Distill-Qwen 1.5B | 1.5B | RL | batch prompt; prompt당 answer; temperature / top-p | 512; 16; 1.0 / 1.0 | §B.1 Table 3 | verified 2026-09-17 | ablation 없음 |
| AReaL, R1-Distill-Qwen 1.5B | 1.5B | RL | clip ε; PPO minibatch; LR constant; max generation length | 0.2; 4; 2.0×10⁻⁵; 27,648 token | §B.1 Table 3 | verified 2026-09-17 | ablation 없음 |
| AReaL device split | all | RL | inference : training device | device 3/4를 inference에 할당 | §7.1 | verified 2026-09-17 | 모든 run fixed ratio(§7.1) |
| verl `fully_async_policy`, Qwen2.5-Math-7B | 7B | RL | `staleness_threshold`; `partial_rollout` | 0.5; True | github.com/verl-project/verl@753aed3 README L351–352 | verified 2026-09-14 ([[verl-rollout]]) | 0 / 0.1 / 0.3 / 0.5 ablation L380–396 |
| verl `fully_async_policy`, Qwen2.5-Math-7B | 7B | RL | `rollout.n`; `ppo_mini_batch_size`; max response | 16; 32; 28K token | same file L335, L339–340 | verified 2026-09-14 | ablation 없음 |
| verl `fully_async_policy`, Qwen2.5-Math-7B | 7B | RL | hardware; step; train batch | H20 128(rollout 64 : train 64); 400; 512 | L333, L344–345, L361 | verified 2026-09-14 | 32 / 64 / 128 GPU 비교 L353–361 |
| verl framework default | — | RL | `clip_ratio_c`; `rollout_is` | 3.0; `null`(off) | `actor.yaml` L83; correction yaml | verified 2026-09-14 ([[verl-ppo-loss]]) | ablation 없음 |
| OpenRLHF framework default | — | RL | `async_enable`; queue; partial rollout | False; 1; False | OpenRLHF@64c1cc4 L274–283 | verified 2026-09-14 ([[async-rollout]]) | ablation 없음 |
| OpenRLHF framework default | — | RL | IS correction enable; type; threshold | False; `tis`; [0.5, 5.0] | same file L257–270 | verified 2026-09-14 | ablation 없음 |
| Qwen2.5-32B DAPO(flash-rl script) | 32B | RL | TIS cap C | 8 | `dapo_qwen32b_bf16.sh` L5, L106 | verified 2026-09-14 ([[rollout-training-mismatch-tis]]) | Fig. 1 TIS 대 no TIS; C sweep 없음 |
| Qwen2.5-32B DAPO | 32B | RL | step prompt; generation batch; sample; dynamic filter | 512; prompt 1,536; 16; accuracy filter, generation batch 최대 10 | same script L28–35 | verified 2026-09-14 | ablation 없음 |
| Qwen2.5-0.5B-Instruct GSM8K PPO | 0.5B | RL | TIS cap C | 2 | `gsm8k_qwen0_5b_int8.sh` L2, L32 | verified 2026-09-14 | Fig. 2, Fig. 4 |
| DeepSWE-Preview from Qwen3-32B | 32B | RL | container/iteration; hardware | 512(batch 64 × pass 8); H100 64 × 6일 | together blog §2.2 | verified 2026-09-14 ([[deepswe]]) | ablation 없음 |
| DeepSWE-Preview | 32B | RL | compact-filter trigger | max context; max environment step; 20-minute timeout | §2.3 | verified 2026-09-14 | Qwen3-14B curve(Fig. 6), 수치 없음 |
| Qwen3-Coder-480B-A35B-Instruct | 480B(35B active) | RL | parallel environment | 20,000 | Qwen blog 2025-07-22 | verified 2026-09-17 | ablation 없음 |
| Llama-3.1-8B-Instruct (AsymRE) | 8B | RL | δV; behavior refresh; sample/prompt | ≈ −0.1; gradient 250 step마다; 8 | arXiv:2506.20520v2 §5.2–5.3 | verified 2026-09-14 ([[asymmetric-reinforce]]) | δV = 0에서 7/7 collapse(Fig. 5 left) |
| LLaMA 3.1 8B async Online DPO | 8B | RL | GPU split | GPU 8개 중 vLLM generation 1, training 7 | arXiv:2410.18252 §4 | verified 2026-09-17 | win rate 57.20% 둘 다, compute 230 대 142 |
| GLM-5 | 744B | RL | τ; ε_ℓ, ε_h; sync K | 보고되지 않음 | §4.1.1–§4.1.2 body 확인 | not reported | — |
| Kimi k1.5 | not reported | RL | partial-rollout token budget | 보고되지 않음 | arXiv:2501.12599 §2.6.2 | not reported | — |

**소규모 범용 run의 시작점.** 아래 수는 verified 행과 source 조건에서만 온다. Synchronous로 시작해 generation이 차지하는 step time을 측정하라. verl colocated 7B DAPO baseline 128 H20은 49.9%였고 그 setting에서 decoupling이 보상됐다. Decouple하면 `η = 1`에서 시작하라. [[async-rlhf-noukhovitch]]는 TL;DR 410M에서 N = 1·2가 거의 같았고, [[areal-async-rl]]은 naive PPO에도 math 1.5B에서 η ≤ 2가 synchronous oracle의 약 2점 이내였다. η = 4에서 naive PPO 23.3인데 42.2를 유지한 decoupled 또는 ratio-corrected objective로 바꾼 뒤에만 η를 올려라. rollout과 gradient가 backend 또는 precision이 다르면 sampler IS correction을 켜고 released DAPO 32B `C = 8`, 0.5B `C = 2`에서 시작하라. verl dual clip은 `c = 3.0`으로 둔다. 각 값은 자기 setting 밖 published ablation이 없으므로 default가 아니라 재측정할 시작점이다.

---

## 일반화 관점

**(a) Breadth를 늘리는 것.**

- *On-policy data는 prior ability를 보존한다.* [[rls-razor]]: 새 task에서 base model KL로 forgetting을 예측(R² = 0.96 ParityMNIST, R² = 0.71 LLM)하며 on-policy update는 low-KL solution 쪽 bias가 있다. 그 KL을 최소화한 oracle SFT distribution은 RL보다 더 적게 잊어 mechanism이 algorithm이 아니라 KL bias임을 보인다(§4). [[retaining-by-doing]]: task family 세 개, 8B까지 Llama·Qwen에서 comparable target performance의 RL이 SFT보다 덜 잊는다(Abstract, Fig. 2). 방향은 독립 논문 두 개에서 **replicated**됐다.
- *Mixed-domain single-stage RL.* [[deepseek-v3.1]]은 reasoning·agent·alignment RL을 한 stage에 합치고(§3), [[minimax-forge-agent-rl]]은 reasoning·general QA·agent task를 동시에 섞으며 sequential training negative transfer를 말한다(§4.1). 각 **결과(단일 연구)**, ablation 없는 official report다.
- *Longer horizon, 더 많은 transfer.* [[skyrl-agent]]는 SWE만 학습한 SA-SWE-32B가 Terminal-Bench, BrowseComp-Plus, WebArena를 개선했다고 보고한다(Abstract, Table 3). [[asearcher]]는 synthesized search QA RL로 GAIA +15.0, xBench +22.4, Frames +15.6을 보고한다(Fig. 1).

**(b) Narrowing 또는 forgetting 원인.**

- *Staleness.* decoupled objective 없으면 η = 4가 AIME24 18.7점, MATH-500 22.3점을 잃는 [[areal-async-rl]] Table 2가 이 장에서 가장 clean하다. [[async-rlhf-noukhovitch]] Figs. 3–4도 TL;DR에서 stale step 수의 logarithmic dropoff라는 같은 방향을 보인다.
- *Reward-metric drift.* [[rollout-training-mismatch-tis]] Fig. 6에서 TIS 없는 BF16은 TIS보다 training reward가 높지만 AIME는 낮다. Training reward는 sampler 아래에서 계산되어 held-out metric 반대 방향으로 갈 수 있다.
- *Entropy collapse.* uncorrected INT8 rollout gap에서는 entropy가 0.2 아래로 계속 떨어진다([[rollout-training-mismatch-tis]] Fig. 5). [[asymmetric-reinforce]] Fig. 8은 큰 baseline일수록 entropy가 빨리 떨어지고 `V ≥ V^µ`에서 support가 하나로 collapse한다고 예측한다(Thm. 4.2).
- *Coverage loss.* [[rlvr-beyond-base-model]]은 RLVR model이 base보다 pass@1은 높지만 large k에서 base가 math·code·visual reasoning benchmark pass@k를 match 또는 exceed한다고 보고한다. 저자는 이를 RLVR이 base가 이미 sample할 path의 probability를 높인 것으로 해석한다(**해석**). pass@1만 log하면 보이지 않는다.
- *Curriculum으로서 filter.* accuracy dynamic filtering, variance-weighted prompt replay(design pattern, unconfirmed implementation; §6), Windowed FIFO는 effective prompt distribution을 좁히고 dropped domain 효과를 source가 보고하지 않는다. **열린 질문.**

**(c) 이 stage에서 측정하는 법.**

1. **Fixed non-task prompt set에서 base model KL**을 fixed decoding setting으로 N step마다 log한다. [[rls-razor]]는 new-task KL을 forgetting predictor로 validate하며 unrelated fixed prompt를 더하면 의도하지 않은 ability drift의 직접 해석 가능한 series가 된다.
2. Held-out split에서 pass@1·training reward만이 아니라 large k의 **pass@k와 token entropy**를 측정한다.
3. Filter·replay buffer·scheduler window가 닿지 못하는 **domain별 held-out regression set**을 fixed temperature/top-p에서 둔다.
4. Throughput·staleness metric과 같은 cadence로 MMLU-class knowledge, IFEval-style instruction following, chat quality, out-of-domain reasoning을 **broad periodic evaluation**한다. regression을 training window에 귀속할 수 있다.
5. **Staleness·system metric을 first-class series**로: realized staleness distribution(cap만 아님), `vllm_kl`, advantage sign별 clip fraction, reason별 dropped sample share.

---

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 검사 |
|---|---|---|
| objective 변경 없이 staleness cap 증가 | training reward 상승, held-out accuracy 하락; negative-advantage token clip fraction 큼 | η = 0 또는 queue 1 short job을 다시 실행해 equal step held-out accuracy 비교([[areal-async-rl]] Table 2) |
| queue depth가 realized staleness와 같다고 가정 | measured policy-version gap이 configured cap 초과 | [[glm-5]]처럼 segment마다 `(w₀, …, w_k)` version log; OpenRLHF async payload에는 version field 없음([[async-rollout]] L149–151) |
| sampler–trainer probability gap 무시 | negative k1 KL, entropy < 0.2, 비정상적으로 긴 response | `vllm_kl`과 token probability max difference log; DAPO-32B 1.0, GSM8K example 0.4([[rollout-training-mismatch-tis]]) |
| chat history re-tokenize로 training sequence 구성 | single-turn에도 PPO non-convergence; engine output·training input token count 차이 | turn별 concatenated id = training sequence assert([[verl-rollout]] L155–182); TITO gateway 채택([[glm-5]] §4.1.2) |
| routing replay 없이 MoE policy 학습 | 유사 dense model엔 없는 MoE 특이 instability | routing replay enable([[deepseek-v3.1]] Keep Routing §3.1; verl `enable_rollout_routing_replay`) |
| environment crash를 model failure로 score | reward variance가 cluster health 추적; all-zero reward group이 시간상 cluster | sample별 failure reason 기록, environment collapse 제외 및 group pad/drop([[glm-5]] §4.1.2) |
| 먼저 끝난 trajectory를 greedy consume | batch 평균 task difficulty drift, gradient oscillation | fetch window bound([[minimax-forge-agent-rl]] §3.1), batch별 difficulty distribution log |
| filter를 공짜로 취급 | retained domain held-out accuracy 상승, 다른 곳 하락 | filter 밖 domain별 regression set과 fixed cadence 평가 |
| BoN을 “PPO와 matching”이라 보고 | per-N human evaluation 없는 claim | [[best-of-n]] App. G.3은 equal average reward에서 similar quality라고 “not shown”; KL 3.2(BoN-64) 대 18.0 / 14.0(PPO) |

---

## 이해 점검

1. [[areal-async-rl]]의 naive PPO는 η = 0 AIME24 42.0에서 η = 4 23.3으로 떨어지지만 decoupled PPO는 같은 staleness에서 42.2다. decoupled objective가 gradient에서 무엇을 바꿔 같은 data를 usable하게 하는지와 Eqs. 4–5의 어느 term이 작동하는지 설명하라.
2. AReaL은 **작은** η가 conservative choice만이 아니라 system problem이라고 한다. admission rule `⌊(N_r − 1)/B⌋ ≤ i + η` 및 response length long tail에서 그 주장을 재구성하라.
3. [[on-off-policy-rlhf]]는 offline training이 policy를 better pairwise classifier, worse generator로 만든다고 한다. “training stream reward”가 unreliable stopping signal인 이유와 문제를 잡을 extra series를 설명하라.
4. Staleness와 sampler–trainer gap은 모두 off-policyness며 truncated ratio로 보정한다. 하나를 구분하는 property와 [[glm-5]]가 rollout log-probability를 behavior proxy로 쓰는 반면 [[areal-async-rl]]은 proximal policy 아래 probability를 recompute하는 이유를 설명하라.
5. `Â < 0` PPO objective는 r에서 unbounded, `Â > 0`은 bounded다. clipped surrogate에서 유도하고 stale data가 positive보다 negative에 더 위험한 이유, dual clip·TOPR·lowered baseline이 같은 failure를 서로 어디에서 자르는지 설명하라.
6. [[deepswe]]는 context·step·time을 다 쓴 trajectory를 mask하고 [[glm-5]]는 sandbox crash trajectory를 drop한다. 둘 다 gradient에서 sample을 제거한다. 하나가 reward-correctness fix, 다른 하나가 data-quality fix인 이유와 생략 시 각각 무엇이 corrupt되는지 설명하라.
7. Dynamic accuracy filtering, variance-prioritized prompt replay, Windowed FIFO는 서로 다른 이유로 도입됐다. 셋 모두 training distribution의 같은 무엇을 바꾼다고 논증하고 cost를 보일 최소 measurement를 설계하라.
8. [[rls-razor]] forgetting law는 controlled η stale rollout이 아닌 SFT·RL에서 측정했다. law가 staleness로 확장되는지 시험하려면 무엇을 run하고 어떤 결과가 extension을 falsify하는지 말하라.

---

## 연결

- **Previous:** ch-53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness. 거기 harness가 이 장 filter·staleness cap이 낼 narrowing을 검출한다.
- **Depends on:** ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. dual clip·TOPR·asymmetric baseline 유도가 있고 이 장은 stale data에 적용한다.
- **Depends on:** ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability. 여기 §7은 같은 loop의 system side, environment 실행 장소와 failure label이다.
- **Related:** ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity, [[rls-razor]]와 [[retaining-by-doing]] 전문.
- **Related:** ch-42 — Reward Hacking and Judge Design, §2 Best-of-N을 제한하는 over-optimization ceiling.
- **Next:** ch-55 — verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation, 이 장이 locus로 인용한 code path를 연다.
- **Later in infra track:** ch-56 — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls; ch-57 — TRL Internals: SFT, DPO, GRPO, and Distillation Trainers; ch-58 — Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability. Infra track은 ch-58에서 끝난다.
- **Capstone:** ch-59 — Capstone: Reproduce One Stage of an Open General-Model Recipe with a Generality Gate, generality gate 아래 stage를 reproduce하고 여기 instrumentation을 쓴다.

---

## 출처

- [[areal-async-rl]] — η·admission rule, decoupled PPO objective, §4 Table 2 staleness ablation.
- [[async-rlhf-noukhovitch]] — stale minibatch update 수에 따른 RLHF 저하 controlled measurement, loss robustness 비교.
- [[on-off-policy-rlhf]] — controlled on-policy 대 offline 비교, §3 generation 대 discrimination result.
- [[best-of-n]] — analytic BoN KL formula, printed value, §2 BoN을 제한하는 over-optimization result.
- [[async-rollout]] — OpenRLHF async trainer queue·slot·lock·partial rollout·IS correction mode, policy-version field 없음.
- [[verl-rollout]] — token-in/token-out vLLM server, weight-sync abort·resume, agent-loop masking, §1 `fully_async_policy` timing table.
- [[verl-ppo-loss]] — negative advantage dual clip, `rollout_is_weights`, negatives 절 worked clipping example.
- [[openrlhf-ppo]] — clipped surrogate sign asymmetry, dual-clip default, `clip_ratio` diagnostic 한계.
- [[rollout-training-mismatch-tis]] — sampler-learner probability gap, TIS·variance argument, entropy·reward-drift side effect.
- [[glm-5]] — TITO, direct double-sided token masking, model-version gap staleness filter, environment-failure exclusion, tail latency prefill-decode disaggregation.
- [[deepseek-v3.1]] — MoE rollout Keep Routing·Keep Sampling Mask, negative-advantage sequence off-policy masking.
- [[minimax-forge-agent-rl]] — distribution control Windowed FIFO, gateway black-box agent training, prefix-tree merging, mixed-domain RL.
- [[replay-buffer-rlhf]] — replay design framework synthesis; §6에서 unconfirmed claim을 표시한 pointer.
- [[kimi-k1-5]] — fixed output budget partial rollout, replay buffer continuation, segment loss exclusion.
- [[deepswe]] — iteration당 512 container Kubernetes sandbox, compact filtering, leave-one-out advantage 산술.
- [[r2e-gym]] — executable SWE environment 절차적 구성과 count.
- [[qwen3-coder]] — environment scaling의 명시한 답인 parallel environment 20,000개.
- [[skyrl-agent]] — async pipeline dispatcher, transition-based backend bridge, DeepSWE 대비 cost·out-of-domain 비교.
- [[asearcher]] — fully async agentic RL의 turn limit 제거, enabled long-horizon behavior.
- [[agent-lightning]] — concatenate-and-mask 대안의 training-agent disaggregation·call별 transition.
- [[prime-intellect-environments-hub]] — 자체 evaluation report가 있는 shared/versioned artifact로서 environment.
- [[topr-tapered-off-policy-reinforce]] — negatives 절 asymmetric tapered IS rule.
- [[asymmetric-reinforce]] — stale data δV = −0.1 권고와 baseline phase transition.
- [[rls-razor]] — forgetting law, base KL predictor, oracle-SFT control.
- [[retaining-by-doing]] — comparable target performance에서 RL이 SFT보다 덜 잊는 독립 replication.
- [[rlvr-beyond-base-model]] — RL 뒤 pass@k coverage loss, large-k pass@k가 measurement list에 있는 이유.
