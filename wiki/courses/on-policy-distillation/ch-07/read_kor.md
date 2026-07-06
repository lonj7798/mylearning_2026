<!-- chapter: ch-07
     track: capstone
     kind: lab
     title: 캡스톤 랩 — boson-agent 대화 데이터 SFT 파이프라인을 위한 온-폴리시 증류 전략
     deps: [[ch-05]], [[ch-06]]
     sources: [[tm-on-policy-distillation]], [[nrehiew-sft-rl-opd]], [[ross-dagger-exposure-bias]], [[agarwal-gkd]], [[hf-trl-gkd-recipe]], [[qwen3-strong-to-weak-distillation]]
-->

# 07장 — 캡스톤 랩: boson-agent 대화 데이터 SFT 파이프라인을 위한 온-폴리시 증류 전략

> **핵심 통찰.** `boson-agent-synthetic-data-dev`는 이미 온-폴리시적인 일을 하고 있다. 다만 테이블의 잘못된 쪽에서 하고 있다. **customer simulator**는 실시간으로 온-폴리시 샘플링을 한다(11개 모델 로테이션). 그러나 실제로 배포하는 모델인 **seller**(`Qwen3.6-27B-Lina-chk-*`)는 생성된 transcript에 대한 SFT로 **오프-폴리시** 학습된다. 20–50턴짜리 판매 통화는 오프-폴리시 모방의 오류가 누적되는 장기 지평(long-horizon) 영역 그 자체다([[ch-03]]). 이 캡스톤의 전략은 다음과 같다. scenario skeleton과 customer simulator를 *environment*로 유지하되, **seller가 자기 자신의 turn을 샘플링**하게 만들고 강한 교사가 reverse KL로 각 seller token을 채점하게 한다. 이렇게 배포 모델을 [[ch-01]] 지도에서 오프-폴리시 코너에서 온-폴리시 증류 코너로 옮긴다.

> **가이드라인(한 줄 메모).** relay를 온-폴리시 롤아웃 환경으로 재사용하라. 고정된 scenario skeleton 안에서 학생이 seller turn을 샘플링하게 하라. `tool_use`를 포함해 seller가 생성한 토큰만 채점하고, customer와 `tool_result`는 마스킹하라. 서빙 가능한 교사(우선 같은 계열의 large Qwen, 품질 상한으로 Claude-via-GOLD)를 사용하고, `lmbda→1`, reverse KL 쪽의 `beta`, 토큰별 clipping, 엔트로피 모니터링을 적용하라. 성공은 단순히 낮은 training KL이 아니라 eval gateway에서 *scripted path로부터의 drift 감소*로 측정하라.

---

## 1. 시스템을 지도 위에 놓기

파이프라인 자체(`agents/lina-tmr-customer-gateway/`)를 보면 다음과 같다.

- **생성물:** 20–50턴의 다중 턴 한국어 보험 TMR(tele-sales) 통화. dual-view clean JSON과, `tool_use`/`tool_result` 블록 및 compaction 전후 스냅샷을 보존하는 raw JSONL로 export된다(`export/synthetic_writer.py`, `export/raw_capture.py`).
- **방식:** scenario YAML이 **customer**를 구동한다. customer는 `customer_rotation.py`의 11개 모델 로테이션(6개 Qwen/boson 변형 + Claude Haiku + GPT-5-mini + Grok + 2개 Gemini)에서 실시간 샘플링되며, **stage-puppeted seller**(`test-lina`, boson `Qwen3.6-27B`)를 상대로 동작한다. 전체 과정은 `relay/orchestrator.py`가 turn-by-turn으로 오케스트레이션하고, barge-in(`relay/interrupt_cut.py`)과 백그라운드 compaction(임계값 약 100 messages)을 포함한다.
- **현재 학습:** 이것은 **데이터 생성 전용**(v0.14)이다. reward도, RL도, distillation loop도 없다. `Qwen3.6-27B-Lina-chk-*` 체크포인트는 **이 transcript들에 대해 SFT**된 것이다.

이제 [[ch-01]]의 세 축 위에 컴포넌트별로 배치해 보자.

| 컴포넌트 | 데이터 소스 | 신호 | 판정 |
|---|---|---|---|
| Customer simulator | **on-policy**(scenario마다 live sampling) | — (environment다) | 이미 on-policy다 — 하지만 우리가 배포하는 모델은 아니다 |
| **Seller student**(우리가 배포하는 것) | **off-policy**(generated transcripts에 대한 SFT) | dense, forward-KL | exposure bias에 취약한 코너([[ch-02]] §6) |

이 불일치가 기회 전체다. *environment*는 온-폴리시지만, *우리가 신경 쓰는 policy*는 그렇지 않다.

> **인터랙티브 동반 자료:** [`figures/boson-opd-flow.html`](figures/boson-opd-flow.html) — **Current (off-policy SFT)**와 **Proposed (on-policy distillation)**를 토글하면서 정확히 무엇이 바뀌는지 확인한다. seller가 sampler가 되고, 교사가 reverse KL로 그 turn을 채점하며, relay가 transcript factory가 아니라 rollout environment가 된다.

---

## 2. 이 시스템이 OPD의 승리 영역에 있는 이유

[[ch-05]]의 진입 테스트를 적용해 보자.

1. **장기 지평인가(Long horizon)?** 20–50턴이다. exposure-bias 항은 T²로 스케일한다([[ch-03]]). seller가 초반에 특이한 반론이나 barge-in 복구를 놓치면, transcript가 다루지 않은 상태로 drift하고 통화는 나선형으로 무너진다. 이것은 LLM agent가 바로 여기에서 보여 주는 "분포 이동 아래에서 자신 있게 틀리는" 실패다. ✅ 온-폴리시를 강하게 지지한다.
2. **교사를 사용할 수 있는가(Teacher available)?** 그렇다. large Qwen(같은 계열) 또는 Claude(크로스 계열, GOLD 경유, [[ch-06]])가 있다. ✅
3. **대신 깔끔하게 검증 가능한 보상이 있는가(Clean verifiable reward)?** 없다. "통화가 성사됐는가 / 컴플라이언스를 지켰는가"는 sparse하고 noisy한 에피소드 종료 신호다. 이는 RL의 약점이다. 좋은 sales turn에 대한 unit-test oracle도 없다. 조밀한 토큰별 교사 신호가 더 신뢰할 수 있는 레버다. ✅ RL보다 OPD를 지지한다.

셋 모두 같은 방향을 가리킨다. seller는 온-폴리시 증류의 승리 영역에 있다. 남은 것은 엔지니어링이다.

---

## 3. 설계

### 3.1 롤아웃 환경 — relay 재사용

온-폴리시 요구사항은 "학생이 실제로 마주할 상태에서 학생으로부터 샘플링하라"는 것이다. relay는 이미 그런 상태를 만들어 낸다. 따라서 다음과 같이 한다.

- `relay/orchestrator.py`를 **rollout environment**로 유지한다. scenario skeleton(stages, personas, per-round director notes)과 live customer는 *environment*로 고정된다.
- seller를 "stage-puppeted transcript producer"에서 **학습 중인 student policy**로 교체한다. 각 seller turn에서 학생이 자기 응답을 *샘플링*한다(이것이 온-폴리시 샘플이다). customer는 이전처럼 반응한다.
- 전체 통화 하나가 하나의 rollout이며, 각 **seller turn**이 채점되는 segment다. 이것은 DAgger의 "현재 policy를 실행한 뒤, 그것이 방문한 상태에 label을 붙인다"([[ross-dagger-exposure-bias]])와 같다. 다만 label은 단일 action이 아니라 교사 분포다.

### 3.2 교사 선택

[[ch-05]]("data source > teacher")와 [[ch-06]](serving이 병목)에 따르면 다음과 같다.

- **첫 번째 선택 — 같은 계열의 large Qwen**(예: 로테이션에 이미 있는 72B / 35B-A3B Qwen): shared tokenizer ⇒ 일반 `GKDTrainer`, 깔끔한 토큰별 정렬, log-prob 서빙 비용이 가장 낮다.
- **품질 상한 — Claude**(크로스 계열): **GOLD/ULD** 크로스 토크나이저 정렬이 필요하다([[ch-06]]). 한국어 sales behavior 품질은 더 높을 수 있지만 serving/alignment 비용이 더 크다. 같은 계열 교사가 성능이 부족하고 예산이 허용될 때만 사용하라.

### 3.3 무엇을 채점할 것인가(token mask)

대화는 균일하지 않다. seller가 *생성한 것*만 채점하라.

- **채점(Grade):** `tool_use` 토큰을 포함한 **assistant seller-turn tokens**. tool call은 seller의 *action*이며, agent correctness가 바로 여기에 있다("자신 있게 틀리는 tool call"이 실패 모드다).
- **마스킹(Mask, do not grade):** **customer turns**(environment다)와 **`tool_result` tokens**(tool이 반환한 것이지 seller가 생성한 것이 아니다).
- 채점되는(seller가 생성한) 위치에 대해서만 토큰별 reverse KL을 계산한다. log-prob가 비교 가능하려면 교사는 학생이 본 것과 *동일한 context*를 보아야 한다.

### 3.4 파이프라인 구조 다루기

- **Compaction boundary**(`raw_capture.py` pre/post snapshots): 학생이 해당 turn에서 실제로 가진 *actual* context, 즉 compacted summary를 포함한 context에 대해 turn을 채점하라. 교사에게도 동일한(어쩌면 compacted된) context를 제공하여 `π_teacher(·|context)`가 `π_θ(·|context)`와 같은 prefix에서 측정되게 하라. context가 연속적이었던 것처럼 compaction boundary를 가로질러 채점하지 마라.
- **Barge-in**(`interrupt_cut.py` dual-cut): barge-in은 seller turn을 truncation한다. cut 이전에 seller가 실제로 실현한 토큰만 채점하라. barge-in *이후*의 recovery turn은 가치가 높은 온-폴리시 신호(새로운 상태)다. 유지하라.
- **Context limit (32K):** 전체 통화가 아니라 **turn 단위로** 채점하라. turn별 학생 context는 이미 limit을 지키며, turn별 채점은 교사 pass 비용도 감당 가능하게 유지한다.

### 3.5 노브([[ch-06]])

- `lmbda → 1`: 완전한 온-폴리시 seller turn. 장기 지평 agent에서는 이것이 핵심이다.
- `beta` toward **reverse KL**: 27B seller는 프런티어 교사를 완전히 모방할 수 없으므로, 자신이 *재현할 수 있는 것*에 mode-seek한다.
- `temperature`: 적당히 설정한다. 지나치게 좁아지지 않으면서 현실적인 sales phrasing을 얻기 위해서다.
- **토큰별 clipping은 필수다.** 한국어 sales dialogue에는 담화/스타일 토큰(높임말, filler)이 많고, 이들은 높은 KL을 가지지만 task content는 적다([[ch-05]]). clipping을 적용해 이 토큰들이 update를 지배하거나 entropy collapse를 가속하지 못하게 하라.

---

## 4. 이 시스템에서 베팅 가격 매기기

- **얻는 것:** 20–50턴 통화에서 exposure bias를 치료한다. 즉 seller가 실제로 도달하는 상태(반론, barge-in recovery, post-compaction continuation)에서 채점받으며, 오프-폴리시 SFT는 이런 상태를 결코 건드리지 못한다. 여기에 [[ch-05]]의 효율성도 있다. 조밀한 토큰별 신호이며, RL 대안보다 약 10× 저렴하다.
- **치르는 비용:** **한국어 tool-calling seller turn마다 매 스텝 교사 forward pass**가 필요하다([[ch-06]]의 병목). 교사가 크로스 계열이면 GOLD 복잡성이 추가된다. 대화가 많은 도메인이라 entropy-collapse와 style-token 위험이 있다(clipping + monitoring으로 완화). 그리고 relay-as-rollout-env harness를 구축해야 한다.
- **대안과 비교하면:**
  - *더 저렴한 SFT refresh* — 여전히 오프-폴리시다. drift를 고치지 못한다. 우리가 이기려는 대상이 바로 이것이다.
  - *Closure/compliance reward model을 쓰는 full RL* — sparse하고 noisy하며 hackable하다. 좋은 turn에 대한 oracle이 없다. 여기서는 dense teacher보다 약하다.
  - *하이브리드(frontier pattern)* — drift를 고치고 강한 교사의 behavior를 이전하기 위한 on-policy distillation에, 얇은 outcome reward(closure/compliance)를 위에 얹는다. sales에 맞게 조정한 "Pretrain→SFT→RL/Expert→OPD-merge" 형태([[nrehiew-sft-rl-opd]])다. 장기적으로 가장 강한 베팅이다.

---

## 5. 무엇을 측정할 것인가

Training-side KL이 내려가는 것은 필요하지만 충분하지 않다. reverse KL은 모델이 collapse하는 동안에도 낮아질 수 있다. OPD가 고치려는 바로 그 대상을 측정하라.

- 학습 중 **엔트로피**(collapse 감시, [[ch-05]]).
- **Turn별 reverse KL** 추세, turn type별 분할(early vs post-objection vs post-barge-in). 온-폴리시 승리는 *어려운* turn에서 가장 잘 드러나야 한다.
- **Eval gateway의 drift reduction**(`lina-tmr-customer-gateway-eval/`, seller가 puppeted가 아니라 *real autonomy*로 실행되는 곳): OPD seller가 새로운 objection/barge-in에서 SFT baseline보다 sensible path에서 **덜** 벗어나는가? 그 deviation 감소가 exposure-bias reduction의 직접적인 downstream measurement이며, 캡스톤의 진짜 성공 지표다.

---

## 6. 깨진 미신: "boson pipeline은 이미 on-policy다"

그렇다. 하지만 *customer*에 대해서만 그렇다. customer simulator는 온-폴리시로 샘플링하므로 *environment*는 현실적이다. 그러나 배포하는 모델인 seller는 그 environment가 만든 transcript에 대해 오프-폴리시로 학습된다. "데이터가 live sampling으로 생성되었다"를 "배포 policy가 온-폴리시로 학습되었다"와 혼동하는 것이 [[ch-01]]의 지도가 막기 위해 존재하는 정확한 함정이다. 온-폴리시는 *학습되는 모델이 누구의 샘플로부터 배우는가*의 속성이다. 현재 그것은 seller 자신의 샘플이 아니다.

---

## 7. 전략 메모(deliverable)

> **`boson-agent-synthetic-data-dev`를 위한 온-폴리시 증류 — 권고안.**
> 1. relay(`relay/orchestrator.py`)를 온-폴리시 rollout environment로 **재정의(Reframe)**하라. scenario skeleton + customer rotation을 environment로 고정한다.
> 2. 그 environment 안에서 학생(배포 가능한 `Qwen3.6-27B`)으로부터 seller turn을 **샘플링(Sample)**하라. 통화 하나 = rollout 하나, seller turn 하나 = graded segment 하나다.
> 3. 교사를 기준으로 토큰별 reverse KL을 사용해 seller-generated tokens(`tool_use` 포함)을 **채점(Grade)**하라. customer + `tool_result`는 **마스킹(mask)**한다. 교사는 학생의 정확한(어쩌면 compacted된) context를 본다.
> 4. **교사:** 같은 계열의 large Qwen(일반 `GKDTrainer`)으로 시작하라. 필요할 때만 GOLD를 통해 Claude로 escalation하라.
> 5. **노브:** `lmbda→1`, reverse KL 쪽의 `beta`, 적당한 temperature, 토큰별 clipping, entropy monitoring.
> 6. eval gateway에서 novel objections/barge-ins에 대한 drift reduction을 **측정(Measure)**하라. training KL만 보지 마라.
> 7. **로드맵:** 지금은 OPD로 drift를 고친다. 나중에 frontier version을 위해 얇은 closure/compliance outcome reward를 추가한다(hybrid).

---

## 8. 전략 논의를 위한 열린 질문

이것들이 실제 의사결정 지점이다. 각 항목에 대한 관점을 가져오라.

1. **교사:** 같은 계열의 large Qwen(저렴하고 깔끔함) vs Claude-via-GOLD(한국어 sales behavior가 더 좋을 수 있으나 더 비쌈). *당신의* 예산에서 품질/비용 경계는 어디인가?
2. **Tool token:** `tool_use` 토큰을 prose와 같은 reverse-KL weight로 채점할 것인가, 아니면 down-weight할 것인가(tool call은 "스타일"보다 "정답/오답"에 가깝다)? 잘못된 tool call에는 reverse KL이 제공하는 것보다 *더 강한* 신호가 필요한가?
3. **결과 신호(Outcome signal):** 순수 OPD인가, 아니면 처음부터 OPD + 얇은 closure/compliance reward의 hybrid인가?
4. **드리프트 지표(Drift metric):** eval gateway에서 "sensible path에서 벗어났다"를 추적 가능한 숫자로 어떻게 operationalize할 것인가?
5. **콜드 스타트(Cold start):** 현재 SFT checkpoint를 initial student(warm start)로 유지하고 거기서 OPD할 것인가, 아니면 base에서 OPD할 것인가?

## 추가 읽을거리

- Thinking Machines, "On-Policy Distillation" — https://thinkingmachines.ai/blog/on-policy-distillation/ ([[tm-on-policy-distillation]])
- Agarwal et al., "On-Policy Distillation of Language Models" (GKD) — https://arxiv.org/abs/2306.13649 ([[agarwal-gkd]])
- TRL `GKDTrainer` / GOLD 문서 — https://huggingface.co/docs/trl/gkd_trainer ([[hf-trl-gkd-recipe]])
- nrehiew, "SFT, RL, and On-Policy Distillation Through a Distributional Lens" — https://nrehiew.github.io/blog/sft_rl_opd/ ([[nrehiew-sft-rl-opd]])
