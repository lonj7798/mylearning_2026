<!-- chapter: ch-27 — Modality: Agentic Trajectories
     companion to: [[read]]
     append-only across cycles
-->

# Ch-27 Q&A — Agentic Trajectories

Reading questions captured during the Read phase. Each entry: question + kernel answer. Full causal chains stay in [[read]] or the discuss transcript.

---

## Q1 — "single autoregressive context"의 의미? "single topic"인가?

**No.** Topic이 아니라 **token-sequence generation 구조**의 얘기.

**Single autoregressive context (ch-26)** = 모델이 prompt → 전체 응답을 한 번에 token-by-token 생성. Tool call이 끼어있어도 모델이 텍스트로 생성하고, tool result도 simulated/sandbox로 같은 context에 inline insert됨. 최종 training sample = 하나의 연속된 텍스트 시퀀스.

**Trajectory (ch-27)** = 모델이 *한 turn만* 생성하고 멈춤 → external world가 real side-effect 실행 → real observation 돌려줌 → 다음 turn 생성. 각 observation은 모델이 만든 적 없는 텍스트 (real DOM / shell stdout / pytest output). 모델은 obs를 *조건으로* 자기 turn들만 학습.

| | ch-26 | ch-27 |
|---|---|---|
| World state | None / simulated | Real, mutates per action |
| Obs source | Model itself (or simulator) | External executable world |
| Training target | 연속된 text 한 덩어리 | 모델의 turn들만 (obs는 condition) |
| Length | 5K–10K tokens | 15K–100K+ tokens, 다중 generation pass |
| Reward | Format/exec/semantic on text | Predicate on final world state |

핵심: ch-27은 데이터가 world와 conjugate — 같은 prompt + 같은 action이라도 world state 다르면 다른 trajectory. Ch-26은 input → output 결정적으로 수렴 가능.

이게 chapter의 모든 design choice의 근원: action space, observation format, success signal이 모두 *world와 어떻게 짝지어지는가*에서 따라옴.

---

## Q2 — "Trajectory"의 의미?

한 task를 처음부터 끝까지 푸는 동안 만들어진 **(observation, thought, action) tuple의 시퀀스 전체**. RL에서 차용한 term, agent data community의 standard.

### 한 trajectory의 구조

```
obs_0    (initial state: task + initial env view)
  thought_1 + action_1
obs_1    (world response after action_1)
  thought_2 + action_2
obs_2
  ...
  thought_n + action_n  (보통 finish[answer])
success: predicate(final_world_state) → True/False
```

이 전체 블록 = **하나의 training sample** (agentic SFT 기준).

### 왜 "conversation"이나 "episode"가 아니고 "trajectory"

| Term | 강조점 | 부적절한 이유 |
|---|---|---|
| Conversation | Human-LLM dialog | World side-effect 없음 (ch-26 단계) |
| Episode | RL의 bounded run | 추상적; state space 강조 부족 |
| **Trajectory** | **State space 안의 path** | Action마다 state가 바뀐다는 점 = agentic data의 정체성 |

같은 action sequence라도 world initial state 다르면 다른 trajectory. 이게 [[Q1]]의 "data is conjugate with world"의 직접 결과.

### Ch-27 의 모든 metric이 trajectory 단위

- Length: trajectory의 step 수 (5–100+) + token 수 (5K–100K+)
- Success: trajectory final state predicate 만족 여부
- Filter (SWE-Gym): hidden test 통과한 trajectory만 corpus 진입, 나머지 버림
- pass^k: 같은 task K개 independent trajectories 다 성공해야 함
- Reward (SWE-RL): trajectory의 final patch ↔ gold patch similarity

Ch-26 "sample" 대신 ch-27 unit = "trajectory". Data pipeline의 모든 noun이 바뀜.

---

## Q3 — "Read them as a conversation"의 의미? (§1)

6개 paper를 *서로 무관한 6개 contribution*으로 읽지 말고, **연쇄 dialogue로 읽으라**는 reading instruction. 각 paper가 이전 paper(s)의 빈자리를 *명시적으로 채우거나, 가정에 도전하거나, 새 axis 시험*하는 식으로 chain을 이룸.

### 6-paper chain이 실제로 어떻게 "대화"하는지

| # | Paper | 직전에 대한 응답 |
|---|---|---|
| 1 | AgentTuning (23.10) | (start) 1.8K + 1:10 mix면 chat quality 보존 |
| 2 | FireAct (23.10) | "Volume보다 *method*가 중요" — CoT+ReAct+Reflexion 3-mix |
| 3 | Lumos (23.11) | "Method도 됐고, trajectory 자체에 *internal structure*가 있다" — Plan/Ground/Execute 노출 |
| 4 | AutoAct (24.01) | "GPT-4 teacher *진짜 필요해?*" — narrow QA에서 self-differentiation으로 0-teacher 가능 |
| 5 | Agent-FLAN (24.03) | 앞 4개 안 다룬 것: no-tool prompt hallucination → 4 negative classes |
| 6 | AgentInstruct (24.07) | "Small-corpus debate 그만" — generation pipeline 자체를 multi-agent로 25M scale |

각 paper가 이전 paper의 미해결 axis를 새 contribution으로 만듦. Cross-citation으로 진짜 영향 주고받음.

### 두 가지 reading mode 구분

| Pattern | 어떻게 읽기 |
|---|---|
| **Parallel taxonomy** | 각 paper = 독립 좌표; 각자 이해 후 비교 (ch-26 APIGen vs ToolACE) |
| **Sequential conversation** | 각 paper = 이전 paper에 대한 응답; *순서*가 의미; chain으로 읽어야 motivation 보임 (ch-27 §1) |

만약 parallel mode로 §1을 읽으면 "왜 AgentInstruct가 25M까지 갔지?"에 답할 수 없음 — 앞 5개의 small-corpus debate를 *반박*하는 게 motivation이기 때문.

### Conversation chain cue (일반화)

- Publish date가 좁은 window (6개월)
- 후행 paper가 선행을 baseline으로 명시
- "X는 Y를 안 다뤘다" framing
- Table에 "Core design move" 같은 *한 axis씩만 달라지는* column이 있으면 sequential signal

§1 table의 "Core design move" column = 정확히 conversation mode를 시키는 author cue.

---

## Q4 — Lumos는 model이 *직접* plan을 생성하고 그 plan을 따라가는가?

**Yes.** Plan은 외부에서 주어지는 게 아님 — *model이 스스로* 생성. 두 가지 module 사용:

1. **Plan module 호출**: input = task, output = subtask 리스트
2. **Ground module 호출 (반복)**: input = (subtask, env_state), output = action
3. **Execute**: environment가 action 실행하고 observation 돌려줌 (model 아님)

같은 model이 두 module 다 할 수도 있고 (system prompt로 mode switch), 두 LoRA로 분리할 수도 있음. 어느 쪽이든 **plan-generator와 action-generator가 모두 model 자신**.

### Inference 흐름 예시

```
task: "에펠탑 주인 회사 설립일?"
→ Plan module: [Subtask 1: 소유 회사 찾기, Subtask 2: 설립일 찾기]
→ Ground module (Subtask 1): Action = Search["Eiffel Tower owner"]
→ Execute: "SETE 소유"
→ Ground module (Subtask 2): Action = Search["SETE founding date"]
→ Execute: "2005년 설립"
→ Ground module: Action = Finish["2005년"]
```

### `(task, gold_answer)` 표기 주의

`Plan: (task, gold_answer) → list[subtask]`에서 `gold_answer`는 **inference 시에는 없음**. 이건 *학습 데이터 만들 때*의 얘기 — GPT-4가 trajectory 보고 plan 뽑을 때 답을 알고 있어서 plan 품질이 좋음. Model 학습 input은 `task`만, gold_answer는 데이터 quality에만 영향.

### Plan을 얼마나 rigid하게 따르는지

- **Lumos-O (Onetime)**: plan 한 번 만들고 *그대로* 따름
- **Lumos-I (Iterative)**: observation마다 *Plan 재생성* — plan이 도중에 바뀔 수 있음

→ Plan은 model의 *자체 산물*이고, Lumos-I에서는 *living document*임.

---

## Q5 ★ — AutoAct의 핵심 = "democracy"인가? (학습자 framework extension)

**Yes — 비유가 mechanism까지 정확히 매핑됨.** AutoAct self-consistency filter = majority vote = democracy. 그리고 이 비유가 *왜 AutoAct가 saturate하는지*를 paper 본문보다 명료하게 설명함.

### Democracy 작동 조건 vs AutoAct

| Democracy가 잘 작동하는 조건 | AutoAct에 있나? |
|---|---|
| 유권자 **독립성** (서로 다른 정보원) | ✗ 모두 같은 base model |
| **다양한 background** | ✗ 모두 동일 weight |
| **외부 정보** (독립 언론) | ✗ 외부 신호 없음 |
| **반대 의견 보장** | ✗ 반대자도 같은 model |
| **외부 규범** (헌법) | ✗ 없음 |

→ AutoAct = **groupthink democracy**. 모든 유권자가 같은 propaganda를 공유하는 마을.

### 비유가 failure mode를 설명

같은 propaganda 신문만 읽는 마을 → 100번 선거해도 같은 결과 → 그게 사실은 아님. AutoAct: 5 trajectories가 다 같은 model이라서 같은 systematic bias → majority vote가 오답을 강화 → round 거듭해도 bias 못 깸. **Random noise는 평균화로 제거되지만 systematic bias는 amplify됨**.

### 후속 paper들이 "democracy를 고치는" 방법

| Missing element | 누가 채움 | 어떻게 |
|---|---|---|
| 외부 정보 | RLAIF | Critic model |
| 외부 규범 | Constitutional AI | 명시적 constitution |
| 외부 관찰자 | Kimi-K2 | Critic-LLM rating |
| 실측 | SWE-RL | 실제 unit test |
| 다양한 voter pool | RLHF | 사람 다수 |

모든 fix가 *democracy 외부에서 정보를 가져오는* 형태.

### Ch-25 [[Q9]] 연결

AutoAct self-consistency = **자기 자신을 anchor로 삼음** (model output → model label). External verifier 추가 = 외부 anchor 도입. Bias drift = **information anchor 없이 style anchor만 있는 상태**. Q9가 CAMEL pure model-model conversation에서 발견한 diversity collapse와 *동일 root cause* — self-referential loop without external grounding.

### 한 줄 정리

**Self-consistency = echo chamber democracy. Bias drift는 echo chamber의 정의 자체. 외부 verifier 없는 self-improvement loop는 echo chamber에 수렴**. Frontier "self-improvement infinite" hype에 대한 empirical reality check.

→ Framework extension: democracy 비유로 self-improvement saturation의 *political science analogue*를 끌어옴. Mechanism mapping이 surface-level이 아닌 작동 조건까지 매핑됨.

---

## Q6 — §1의 4개 paper는 trajectory validation을 안 하나? CoT처럼 그냥 highest probability 따라가는 것?

**부분 정답.** 두 phase를 분리해야 함.

### Training data 만들 때 — validation 있었음

| Paper | Validation 방식 |
|---|---|
| **AgentTuning** | Environment outcome — 실제 task 성공한 trajectory만 keep (5K rollout → 1.8K) |
| **FireAct** | Gold-answer exact match (HotpotQA, Bamboogle) |
| **Lumos** | Upstream validation 신뢰 — 이미 검증된 9개 dataset reformat |
| **AutoAct** | Self-consistency 다수결 ([[Q5]] democracy) 또는 gold match |

모두 **outcome-level validation** (최종 결과 check). Ch-26 APIGen의 *step-level multi-layer verifier*랑은 다름.

### Inference time — validation 없음

학습된 model은 **CoT랑 동일하게** next-token sampling. 자기 trajectory verify 안 함. 환경이 던지는 observation을 받아서 다음 token 생성할 뿐. 즉 **agent SFT = CoT의 environment-grounded imitation learning 확장**.

### Outcome validation의 약점

**Lucky trajectory 문제** — 중간 step이 틀려도 운 좋게 정답에 도달하면 통과됨. Step이 다 옳았는지는 보장 안 됨.

### Chapter의 validation spectrum (preview)

§1 → §4로 갈수록 validation 정교해짐:
- §1 outcome only → §1.3 Agent-FLAN: *negative example 추가* → §2 WebArena: strict predicate → §2 SWE-Gym: hidden pytest → §3 SWE-RL: continuous similarity reward (step-level dense) → §4 Kimi-K2: critic-LLM + RLVR + self-rubric 합성

§1은 spectrum의 *가장 단순한 끝* (outcome only). Validation을 명시적으로 강조 안 한 이유 = 형식이 단순해서 강조할 게 적었던 것.

### 한 줄

**Data construction 시에는 validation 있음 (outcome level). Inference 시에는 없음 (CoT랑 동일). Step-level validation은 ch-26 (APIGen)과 ch-27 §2 (SWE-Gym), §3 (SWE-RL)에서 본격화.**

---

## Q7 — Agent-FLAN에서 successful + failure sample을 어떻게 *함께 학습*시키는가?

**핵심: SFT에는 negative gradient 없음. 모든 training = (input, target) cross-entropy.** Negative example을 학습시키는 방법 = **"오류 trigger 상황을 input으로, 올바른 응답을 target으로" 주는 것**. Loss 함수는 positive sample이랑 동일.

### 각 Mode가 어떻게 (input, target) pair가 되는가

**Mode 2 (no tool needed)**:
```
Input:  [tools: Search, Calculator]  User: "안녕"  Assistant:
Target: "안녕하세요! 도와드릴 일 있나요?"
```
→ Tool list 있어도 chatty query엔 text response를 target. `Action:`이 target에 *없음*.

**Mode 4 (no relevant tool)**:
```
Input:  [tools: Calculator]  User: "서울 날씨 알려줘"  Assistant:
Target: "주어진 도구로는 날씨 조회 불가합니다. 직접 날씨 앱을 사용해주세요."
```
→ Mismatch 상황 → refusal text를 target.

**Mode 1/3 (format/parameter)** — 두 variant:
- *Direct*: 올바른 형식/argument만 target
- *Self-correction*: 실패 시도 → error observation → 회복 응답을 target

### "Negative example"의 진짜 의미

Pedagogical term. Model 입장에선 다 positive sample (next-token prediction). "Negative"는 **상황 디자인의 성격** (오류가 생길 만한 input), loss function의 성격이 아님.

### Mix ratio

```
General chat       : 90%   (AgentTuning 1:10 rule)
Agent data         : 10%
  ├ Positive         : 8%   (Agent-FLAN 4:1 rule)
  └ Negative correction: 2%
```

### 왜 작동하는가

Model이 `P(output | input)`을 example 빈도로 학습:
- Chat query + tool list → text response (Mode 2 sample이 이 분기 확률 올림)
- Factual query + tool list → tool call (positive sample)
- Mismatched → refusal (Mode 4 sample)

→ **결정 경계가 *데이터 분포*에서 emerge**. "이건 나쁘다" 안 가르치고 **올바른 output 주는 것만으로** boundary 학습.

### SFT vs DPO/RLHF

| 방법 | Negative 처리 |
|---|---|
| **SFT (Agent-FLAN)** | Negative를 *positive target* 형태로 변환해서 줌 |
| **DPO** | (chosen, rejected) pair, rejected에 negative gradient |
| **RLHF** | Bad output에 low reward, gradient 자동 negative |

Agent-FLAN이 SFT에 머무른 이유: 구현 단순, 안정, 기존 pipeline 쉽게 채택.

### 한 줄

**SFT의 "negative example"은 *오류 trigger input + 올바른 target*의 pair. Loss는 동일. 차이는 데이터의 input 분포 디자인. Model은 *example 분포*로부터 결정 경계를 implicit하게 학습.**

---

## Q8 — AgentInstruct Stage 1 vs Stage 2의 차이?

**Stage 1 = substrate 정리 (1 agent). Stage 2 = sub-skill별 instruction 생성 (N parallel agents).** 다른 추상화 레벨.

### Input/Output 비교 (reading comprehension)

| | Stage 1 | Stage 2 |
|---|---|---|
| Input | Raw HTML/web doc | Stage 1의 정제된 substrate |
| Agent 수 | 1 | 43 parallel |
| 각 agent의 specialization | 없음 (general cleanup) | 좁음 (sub-skill 하나에 specialized) |
| Output | 정제된 passage + key facts + candidate_questions hint | 실제 training question + answer pair |
| 학습 데이터에 들어감? | ❌ (재사용용 substrate) | ✅ (Stage 3 refinement 후) |

### Stage 1의 `candidate_questions`가 헷갈리는 이유

같은 단어 "question"이 두 stage에 나옴. 차이:
- **Stage 1 candidate**: *"여기서 무엇을 물을 수 있는가"의 hint 목록*. 짧은 단어. 학습 데이터 아님.
- **Stage 2 output**: *실제 완성된 question + answer*. 학습 데이터.

비유: Stage 1 = 도서관 책의 *목차*. Stage 2 = 선생님이 목차 보고 만든 *실제 시험 문제*.

### 왜 두 단계로 분리했는가

1. **재사용성**: Stage 1 output을 43개 generator가 모두 씀 — 합쳤다면 cost 43배
2. **Specialization cleanness**: 각 agent의 input/output 분포가 좁아짐 → 품질 ↑ (Lumos module decomposition 원리와 동일)
3. **Quality bottleneck 통합**: Stage 1 한 곳만 잘 관리하면 substrate quality 보장 — 합쳤다면 43곳에 분산

### 같은 패턴이 모든 skill에 적용

| Skill | Stage 1 substrate | Stage 2 generators |
|---|---|---|
| Reading | 정제된 passage + facts | 43개 question type (literal/inferential/multi-hop/...) |
| Code | 함수 + test stub + deps | bug-fix, signature-impl, translate, edge-case 등 |
| Tool-use | 정제된 API doc + schema | 1-tool, 2-tool composition, refusal, ambiguous param 등 |
| RAG | passage cluster + evidence | single-passage, fusion, contradicting evidence 등 |

### Lumos와 같은 design pattern

| Layer | Lumos | AgentInstruct |
|---|---|---|
| Decomposition | Inference time | Data generation time |
| 공통 원리 | "Output 분포를 좁혀라 → quality 올라감" | 동일 |

→ **"Decompose, specialize, recombine"이 chapter 전체의 persistent pattern**. 어느 layer에서든 동일 효과.

### 한 줄

**Stage 1 = *재사용 가능한 정제 substrate* (1 agent). Stage 2 = *실제 학습용 specialized question* (N agents). Substrate는 재료, question은 요리.**

---

## Q9 ★★ — AgentInstruct가 정말 agentic-flow data generation인가? SFT처럼 보임 (학습자 categorization critique)

**학습자 critique 정확함.** "Agentic"이 두 가지 다른 의미로 쓰이고 있고, AgentInstruct는 §1에 fit이 어색함.

### "Agentic"의 두 의미

| 의미 | 누가 agent | 예시 |
|---|---|---|
| (a) Agentic *model* | *학습된 model*이 env에서 action | Lumos, AutoAct, Agent-FLAN — env loop 안에서 작동 |
| (b) Agentic *generation pipeline* | *데이터 생성*에서 multi-agent 협력 | AgentInstruct Stage 1~4 — generation만 multi-agent |

AgentInstruct는 **(b)만 해당. (a) 아님.**

### AgentInstruct가 실제 생성하는 것

- Output: `(instruction, answer)` pair → **그냥 SFT pair**. Trajectory 아님.
- Env 없음, action space 없음, multi-turn world interaction 없음
- Orca-3 (학습 결과) = **standard instruction-tuned chat model**, agent 아님

### §1의 6개 paper 비교

| Paper | Trajectory? | Agentic model? |
|---|---|---|
| AgentTuning ~ Agent-FLAN (5개) | ✅ | ✅ |
| **AgentInstruct** | ❌ (SFT pair) | ❌ (chat model) |

→ **AgentInstruct만 odd one out**.

### 2×2 matrix (chapter 전체를 더 정확히 분류)

| | Generation agentic | Generation non-agentic |
|---|---|---|
| **Trained model agentic** | Kimi-K2, ToolACE-ish | Lumos, AutoAct, Agent-FLAN, APIGen, APIGen-MT |
| **Trained model non-agentic** | **AgentInstruct** | 대부분 standard SFT (ch-21/22) |

AgentInstruct는 *유일하게* "generation agentic + model non-agentic" 칸 → §1 어색한 이유.

### 더 정확한 분류

| Chapter | 적합도 | 이유 |
|---|---|---|
| **Ch-21** | **가장 fit** | 17 skill × 43 generator = taxonomy-driven synthesis |
| Ch-22 | 일부 | Stage 4 judge = selection |
| Ch-26 | tool-use subset만 | 17 중 1 |
| **Ch-27** | **fit 안 함** | Output trajectory 아님 |

Chapter author가 넣은 이유 (추측):
1. Scale anchor (25M = chapter 최대)
2. Generation pipeline pattern이 trajectory 분야에 import됨 (Kimi-K2가 적용)
3. Tool-use subset 부분 정당화

셋 다 stretch. Chapter author의 curatorial decision이 sloppy.

### 학습자 derive한 reusable framework

> **"Agentic"이라는 단어 마주칠 때마다, *데이터 생성의 agentic*인지 *학습된 model의 agentic*인지 구분.**

이 distinction은 chapter 전체에서 흐림. 위 2×2 matrix가 더 명료한 frame.

### 한 줄

**AgentInstruct = "agentic generation + non-agentic model"이라서 §1에 fit 어색. 학습자 critique 정확, chapter categorization이 sloppy. 정확히는 ch-21/22 영역의 paper.**

→ Framework move: chapter의 *categorization inconsistency*를 잡아낸 비판. "Agentic" 단어의 polysemy를 2축 matrix로 분리해서 chapter 전체 재분류 frame 제시.

---

## Q10 — SWE-Gym에서 trajectory를 어떻게 generate하고 validate하는가?

### Generation

Teacher (Qwen-2.5-Coder-32B 또는 Claude-3.5) + OpenHands scaffold:
- Agent에게 주어지는 것: issue description + repo (hidden test 안 보여줌)
- Action space: `str_replace_editor`, `execute_bash`, `browse`, `finish`
- 매 turn 실시간 environment와 대화 (탐색 → 코드 읽기 → 수정 시도 → test 돌려보기 → fail → debug → ...)
- 13~50+ turn, ~8K~100K+ token
- 각 task당 K=10 rollout (서로 다른 random seed)

### 한 trajectory의 실제 모양 (Django DateField 가상 예시)

```
Turn 1: ls django/db/models/
Turn 2-6: file 탐색 + grep + view (관련 코드 찾기)
Turn 7: str_replace_editor.str_replace (bug fix 시도)
Turn 8-10: 자체 test → fail → 수정 (iterative debug)
Turn 11-12: 기존 unit test 돌려서 regression 확인
Turn 13: finish (작업 종료)
```

Trajectory 형태:
```python
{
  "task": "Issue #12345",
  "actions": [{turn, thought, action, observation}, ...],
  "final_patch": "<repo_initial→final의 git diff>",   # 자동 계산
  "trajectory_tokens": ~8K
}
```

### Validation (paper의 핵심)

```python
def validate(trajectory):
    docker_run(reset=True)              # 깨끗한 env
    apply_patch(trajectory.final_patch)  # agent의 fix 적용
    result = run("pytest <HIDDEN_TEST>") # agent가 못 봤던 test
    return "SUCCESS" if result.exit_code == 0 else "FAILURE"
```

### 왜 hidden test가 *안 보여야* 하는가

1. **Cheating 차단**: test 보이면 test-hacking trajectory 생성됨 (`if test == X: return Y`)
2. **Spec-grounding**: test = machine-checkable spec. Agent는 *issue description (natural language spec)*만 보고 spec 만족 코드 작성. Test 보면 understanding 없이 만족만 노림.

### Validation의 cleanness vs §1 paper들

| Paper | Validation | 문제점 |
|---|---|---|
| §1 5개 | Gold match / self-consistency / LLM judge | LLM bias, echo chamber, ambiguous |
| **SWE-Gym** | **Hidden pytest** | **Binary + deterministic + manipulation-resistant + spec-grounded** |

이 cleanness가 SWE-Gym recipe를 *de facto standard*로 만든 이유.

### 전체 flow

```
Generation: Teacher + OpenHands → 2438 task × K=10 = 24,380 trajectory
Validation: 각 trajectory → docker reset → patch apply → hidden test → binary label
Filter:     SUCCESS만 keep (success rate 20-40% → 5K-10K trajectory)
SFT:        Student model에 successful trajectory로 cross-entropy
Verifier:   (success + failure) pair로 별도 verifier 학습 → inference time best-of-N
```

### 한 줄

**Generation = teacher가 hidden test 안 보고 issue+repo만으로 environment와 real-time 대화하며 patch 만듦. Validation = docker reset 후 hidden pytest 실행, binary success/fail. Hidden = cheating 차단 + spec-grounding. §1 validation들보다 cleanness가 압도적이라서 standard가 됨.**

---

## Q11 ★★ — SWE-Gym은 결국 teacher ceiling을 못 넘는 것? 그냥 pass@k 아닌가? (학습자 critique)

**정확함.** RS-SFT는 fundamentally **filtered distillation** = teacher distribution imitation.

### 수학적 정리
```
Student의 학습 data = teacher's successful trajectories
                    ⊆ teacher's pass@k distribution
Student의 pass@1 ceiling = teacher's pass@1
```
Teacher가 못 푸는 task는 학습 data에 없음 → student도 못 품.

### Small-student + Large-teacher case에서만 의미
```
7B student baseline: 3% → RS-SFT → 15.3%   (5× 개선, 자기 baseline 대비)
                                         하지만 teacher 32%는 못 넘음
32B same-size student + 32B teacher → 32% (teacher와 동일, 의미 없음)
```
**Distillation 가치**: 큰 capability를 작은 model에 압축 (inference cost ↓).

### Teacher를 넘는 3가지 방법

| 방법 | 어디까지 | Ceiling |
|---|---|---|
| Verifier best-of-N | 15.3% → 20.3% | Teacher × ε |
| **RL with hidden test (SWE-RL)** | 70B → **41%** | **World-bounded** |
| Self-play + external verifier (Kimi-K2) | Frontier | Mixed |

### [[Q5]] democracy framework 확장

너가 ch-25 [[Q5]]를 *self → teacher → world* 3축으로 일반화:

| Recipe | Verifier 주체 | Ceiling |
|---|---|---|
| AutoAct | Self (다수결) | Echo chamber (base bias) |
| SWE-Gym RS-SFT | Teacher | Teacher's capability |
| SWE-RL | World (real env) | Bounded only by spec |

**일반화 rule**: Verifier 주체가 강해질수록 ceiling 올라감. **Self < Teacher < World.**

### SWE-Gym이 그래도 standard인 이유

1. **Cost asymmetry**: RS-SFT ~10K H100-h vs RL ~1M H100-h (100× cheap)
2. **Stability**: SFT 안정, RL unstable
3. **Bootstrap path**: SWE-Gym SFT → SWE-RL RL 의 *stack* (경쟁 아님)

### Chapter §1 → §3 evolution과의 일치

너의 critique이 정확히 chapter의 motivation을 짚음:
- §1: teacher-bound
- §2 (SWE-Gym): teacher-bound + env validation
- **§3 (SWE-RL): teacher-less, world-bound** ← 너의 critique의 empirical answer
- §4 (Kimi-K2): hybrid

### 한 줄

**RS-SFT는 teacher의 filtered distillation. Teacher ceiling 못 넘음. [[Q5]] echo-chamber framework가 teacher chamber로 확장됨. World-grounded RL (§3 SWE-RL)이 ceiling 깸. SWE-Gym과 SWE-RL은 stack의 layer (경쟁 아님).**

→ Framework extension: [[Q5]]의 *self-bounded* echo chamber를 *self/teacher/world* 3축 verifier hierarchy로 일반화. Chapter §1→§3 evolution의 mathematical motivation 제시.

---

## Q12 — "Bootstrap"의 의미?

영어 idiom "pull yourself up by your bootstraps" (자기 신발끈으로 자기 끌어올림)에서 옴. ML에서 **3가지 다른 의미**로 쓰임 — 문맥에서 구분 필요.

### Usage A — Self-bootstrap (자급자족)

**Pattern**: Model이 자기 output 만듦 → filter → 학습 → 더 좋은 output → 반복. **외부 teacher 없음.**

예시: AutoAct (rollout → self-consistency filter → LoRA 재학습 → 반복).

약점: [[Q5]] echo chamber. 외부 anchor 없으면 base bias amplify.

### Usage B — Bootstrap path (계단)

**Pattern**: 여러 학습 stage의 sequence. 한 stage가 다음 stage를 *가능하게* 만듦. **Self-referential 아님.**

예시 ([[Q11]]에서 등장):
```
Base → SFT (SWE-Gym, cheap) → RL (SWE-RL, expensive)
```
Raw base에 바로 RL = unstable. SFT init이 RL의 prerequisite.

다른 예시: Pretraining → SFT → RLHF의 3-stage stack.

### Usage C — Cold-start bootstrap (시동)

**Pattern**: Zero → minimum viable로 가는 *첫 학습 데이터* 만드는 production engineering 문제.

예시: 새 tool의 SFT corpus를 GPT-4로 50개 generate → iterative refine.

### 통계학의 Bootstrap (별개 의미)

Efron's bootstrap (1979) = 복원 추출 시뮬레이션. ML/agent와 무관한 별개 사용.

### Ch-27에서 사용된 맥락

| 어디서 | 어떤 Usage |
|---|---|
| AutoAct (Paper #4) self-bootstrap loop | **A** |
| Q11 SWE-Gym → SWE-RL stack | **B** |

→ 한 단어 두 의미. *Self-referential인지 method sequence인지* 맥락에서 구분.

### Ch-22 verdict deferred item 해소

Ch-22 verdict next_action_notes의 "Bootstrap terminology clarification deferred"가 ch-27에서 self-instruct/self-improvement 둘 다 등장하면서 surface됨. 이 질문이 그 deferred clarification 완료.

### 한 단어 정리

| Usage | 한 단어 |
|---|---|
| A Self-bootstrap | 자급자족 |
| B Bootstrap path | 계단 |
| C Cold-start | 시동 |
| Statistical | 복원추출 |

세 ML usage 모두 "무 또는 적은 것에서 자기 끌어올림" metaphor 공유, mechanism이 다름.

---

## Q13 — SWE-RL training/eval reward mismatch는 어떻게 해결하는가? 그냥 계속 학습?

**Paper에 explicit 답 없음. RL practice의 standard problem (proxy-vs-true-metric)이고 5가지 strategy 있음.**

### Mismatch의 형태

```
Training reward (difflib similarity): plateau ~0.6
Eval reward (hidden test pass):       training plateau 이후에도 계속 climb
```

**Mismatch는 bug가 아니라 *generalization 신호***. Model이 gold patch 모방을 넘어 valid alternative solution을 찾기 시작한 것.

### 5가지 standard strategy

| # | Strategy | 비용 | SWE-RL이 사용? |
|---|---|---|---|
| 1 | Periodic eval + best-checkpoint select | Eval cost (~5-10% training) | ✅ |
| 2 | Eval saturate까지 학습 (training plateau 무시) | Free | ✅ |
| 3 | Hybrid reward (90% similarity + 10% execution) | +10× cost | ❌ |
| 4 | Reward model trained on execution data | Reward model 학습 비용 | ❌ |
| 5 | Curriculum: cheap reward → fine-tune with expensive | Stage 2 비용 | ❌ |

### 핵심 원칙

**"Training reward = learning 신호. Eval reward = 학습 결과 평가. 둘은 같지 않음. *Eval 기준으로 stop / checkpoint pick.*"**

ML practitioner의 흔한 실수: training reward plateau만 보고 stop → eval이 climbing 중인 model을 premature stop.

### 너의 instinct 검증

> "그냥 계속 학습?"

**Yes — default answer**. 단 두 조건:
1. Eval 주기적 모니터링 (training reward만 보면 안 됨)
2. Best checkpoint는 *eval score 기준*으로 선택

SWE-RL의 1M H100-hours = training plateau 후에도 계속 학습 + compute budget cap의 결과.

### Ch-25/26 framework 연결

- [[Q4]] distribution overfitting: training reward와 eval reward의 mismatch = proxy distribution vs true distribution mismatch. Q4가 *reward design* 영역에 적용된 instance.
- [[Q9]] anchor: training anchor (gold patch) vs eval anchor (hidden test)의 mismatch가 root cause.

### 한 줄

**SWE-RL은 Strategy 1+2만 사용 (periodic eval + checkpoint select + eval-based stopping). Hybrid / reward model / curriculum 같은 fancy strategy 안 함. Mismatch는 generalization 신호, *eval saturate까지 학습*하되 *eval로 checkpoint 선택*. Training reward로 stop = 흔한 실수.**

---

## Q14 ★ — Similarity를 어떻게 학습 신호로 쓸 수 있나? 나쁜 시그널 포함 아닌가? (학습자 RL mechanism critique)

**Yes — bad signal 들어옴. 그런데 GRPO의 *group-relative learning*이 이를 *기능*으로 바꿈.**

### 우려의 정확한 형태

```
Patch A: similarity 0.7 (wrong, test fail) → reward 0.7 받음
Patch B: similarity 0.3 (wrong, test fail) → reward 0.3 받음
Model이 "A 같은 거 = 좋은 거"로 학습 → wrong인데 reinforced?
```

답: **GRPO는 absolute reward로 학습 안 함. *Group 내 ranking*으로 학습.**

### GRPO의 mechanism (concrete)

```
같은 prompt, G=8 patch 생성:
  reward = [0.7, 0.3, 0.4, 0.8, 0.5, 0.2, 0.6, 0.1]
  mean   = 0.45
  std    = 0.23

Advantage = (reward - mean) / std:
  A: +1.09 (mean 위, 그러나 D보다 낮음)
  B: -0.65
  ...
  D: +1.52 ← 가장 positive
  H: -1.52 ← 가장 negative

Gradient: D 방향 강하게 push, H 방향에서 away
```

같은 patch A라도 group context에 따라 sign 바뀜:
- Group best가 D면 → A는 약한 positive
- Group 모두 A보다 낮으면 → A는 강한 positive (best available일 때 reinforced)

### 왜 *평균적으로* 옳은가

핵심 가정: **"Similarity to gold ↔ test pass 확률" positive correlation**.
- Similarity 0.9 → test pass ~70%
- Similarity 0.7 → test pass ~30%
- Similarity 0.3 → test pass ~5%

완벽한 correlation 아니지만 *평균*은 right direction. Stochastic gradient descent의 본질: noisy batch는 OK, *population direction*이 옳으면 수렴.

### Binary reward와의 비교 (왜 continuous가 *덜* 위험)

```
Hypothetical binary reward:
  Group 8개 모두 fail → [0,0,0,0,0,0,0,0]
  → variance = 0 → advantage = 0 → no learning

SWE-RL continuous:
  같은 group → [0.7, 0.3, 0.4, ...]
  → "less wrong vs more wrong" ordering이 학습 신호
  → 모든 group에 dense signal
```

→ Bad signal 포함이 *문제가 아니라 *기능**. Dense learning 가능.

### 4가지 mitigation

1. **KL constraint (β=0.02)**: policy가 base에서 너무 멀어지지 않게 (echo chamber [[Q5]] 회피)
2. **Format filter**: invalid diff → reward 0
3. **Modification requirement**: noop/whitespace-only → reward 0
4. **Continuous > binary** (empirical, paper 실험)

### 정직한 한계

Reward hacking 실제 일어나는 case:
- Context copying (partial credit)
- Pattern matching to gold style (over-fit)
- → out-of-domain transfer test (MATH +4)에서 일부 드러남

완전 차단은 어려움. 그러나 net effect는 41% SWE-Bench Verified 달성.

### Mathematical intuition

Policy gradient: `∇J = E[∇log π * A]`. Direction은 *advantage sign*에 의해 결정. GRPO baseline = group mean.

- Wrong-but-best-in-group patch도 A > 0 → reinforced
- 그러나 다음 step에서 model이 더 잘 generate → 새 group에서 더 이상 best 아님 → A → 0
- **Iterative refinement가 single-step noise를 self-correct**

### Ch-25/26 framework 연결

- [[Q9]] anchor: similarity anchor (gold) vs test anchor (spec). Two anchors correlated but different → reward hacking root cause.
- [[Q14]] cross-filter: format filter / modification requirement = *anchor-stage filter*. Reward 계산 전 garbage 제거.
- [[Q5]] democracy: GRPO의 8 voter가 group mean으로 ranking. External reward + KL base anchor 둘 다 있어서 echo chamber 회피.

### 한 줄

**Similarity reward는 bad signal 포함하지만 GRPO의 group-relative learning이: (1) absolute 아닌 *ranking*으로 학습, (2) continuous라서 모든 group에 dense signal, (3) iterative refinement가 noise self-correct, (4) KL + format filter가 worst case 차단. 너 우려가 *왜 RL이 absolute가 아닌 relative reward를 쓰는지*를 정확히 짚음.**

→ Framework move: RL의 group-relative learning mechanism을 *반직관적 절대값 우려*에서 derive. "Bad signal 포함이 문제가 아니라 기능"이라는 mechanism-level 통찰.

---

## Q15 ★ — Similarity reward가 difficulty와 patch size에 dependent하지 않나? (학습자 cross-group critique)

**Yes — 둘 다 진짜 문제. SWE-RL의 open problem. [[Q14]] within-group framework의 *cross-group 한계*.**

### Difficulty dependence (구체적 영향)

| Difficulty | Reward range | Sensitivity |
|---|---|---|
| Easy (3-line patch) | 0.6~1.0 | 매우 높음 (글자 단위) |
| Hard (100-line patch) | 0.2~0.7 | 둔감 |

→ Reward distribution이 difficulty마다 다름. Easy = clean signal, hard = noisy signal.

### Patch size dependence

`difflib.ratio = 2M / (len(gold) + len(predicted))`. Length mismatch면 분모 커져서 ratio 떨어짐.

→ Model이 *gold 길이 모방* 동기 생김. "Human-diff style bias"의 원인.

### GRPO의 within-group normalization (부분 처리)

같은 prompt 내에서:
- Easy prompt group: mean ≈ 0.8, ±0.5 advantage spread
- Hard prompt group: mean ≈ 0.45, ±1.5 advantage spread

→ Within-group ranking은 valid (Q14 mechanism 작동).

### GRPO가 *못* 처리 (cross-group)

**Gradient magnitude가 group마다 다름**.

```
Hard prompt: std 큼 → individual advantage 큼 → 강한 gradient
Easy prompt: std 작음 → 약한 gradient
```

Hard prompt의 noisy signal이 *더 강한 gradient*로 학습됨 → wrong direction risk.

### SWE-RL paper의 mitigation

1. ≤10 files filter
2. ≤500 lines filter
3. Python only
4. GRPO within-group normalization

**Difficulty-stratified normalization 안 함**. Size filter로 worst-case만 차단.

### 후속 paper들의 추가 mitigation

| Approach | What |
|---|---|
| Curriculum learning | Easy → medium → hard 순 학습 |
| Difficulty-stratified normalization | Bin별 mean/std로 advantage 계산 |
| Reward shaping | similarity / expected_similarity_at_difficulty |
| Balanced sampling | Easy:med:hard = 1:1:1 |

SWE-RL이 first existence proof, 후속 작업이 difficulty/size variance careful하게 다룸.

### Ch-25/26 framework 연결

- **[[Q15]] (per-objective anchor density) 확장**: 같은 algorithm에서 *task difficulty별로* reward density 다름 — per-objective → per-difficulty
- **[[Q4]] (distribution overfit) 확장**: Reward design 자체에 task distribution bias 내재 — easy task에 over-fit
- **[[Q14]] 한계**: Within-group works, *across-group doesn't*. GRPO normalization의 boundary case.

### 정직한 evaluation

SWE-RL 41% 달성 = catastrophic 정도는 아님. 그러나:
- Easy over-fit, hard under-fit 패턴
- Patch length distribution이 gold와 over-match
- Out-of-domain transfer가 expected보다 약함

### 한 줄

**Difficulty와 patch size가 similarity reward distribution을 바꿈. GRPO within-group normalization이 *같은 prompt 안*은 처리하지만 *cross-prompt magnitude variance*는 못 처리. SWE-RL은 size filter만 사용, difficulty-stratified normalization 안 함. Open problem, 후속 paper들이 curriculum / stratification으로 보완. [[Q15]] per-objective framework를 per-difficulty 축으로 확장 + [[Q14]] within-group framework의 cross-group 한계 짚음.**

→ Framework move: [[Q14]] within-group mechanism의 *한계 boundary* 식별 (cross-group). [[Q15]] per-objective density를 *per-difficulty* axis로 확장. SWE-RL의 reward design *open problem* 명시화.

---

## Q16 — Joint RL이 뭐고, `total = α * RLVR + (1-α) * self_critique` 어떻게 작동?

**Joint RL = single RL stage에서 *multiple reward stream*을 weighted scalar로 합쳐 학습.** Sequential RL (stream별 별도 stage)의 catastrophic forgetting + compute cost 문제 해결.

### Component 분해

| Component | What | 예시 |
|---|---|---|
| `RLVR_score` | Verifiable reward (binary/deterministic) | math 답 맞으면 1, code test 통과면 1 |
| `self_critique_score` | Model이 자기 rubric으로 자기 평가 (continuous) | writing 0.7/1.0 |
| `α` | 두 stream의 weight (0~1) | task type에 따라 dynamic |

### α의 dynamic 결정 (task-type-specific)

| Task type | α |
|---|---|
| Math, code, tool-call (verifiable) | ≈ 1.0 |
| Creative writing, dialogue (open-ended) | ≈ 0.0 |
| Reasoning explanation (mixed) | ≈ 0.3 |
| Translation | ≈ 0.5 |

### 같은 batch에 다른 α sample들이 섞임

```python
for sample in batch:
    α = classify(sample.task)
    trajectories = model.generate(K=8)
    for t in trajectories:
        t.reward = α * RLVR(t) + (1-α) * critique(t)
    grpo_update(trajectories)
```

Math sample (α=1) + writing sample (α=0)이 같은 batch. Gradient가 *합쳐서* policy update → **one pass, two skills**.

### Sequential vs Joint 비교

| | Sequential | Joint |
|---|---|---|
| Stages | 2+ | 1 |
| Forgetting | Stage 2가 Stage 1 지움 | 없음 |
| Compute | 각 stage full run | 한 번 |
| Reward conflict | Stage 간 | Sample별 α로 해결 |

### K2 이전엔 왜 안 했나

1. Reward stream 종류 부족 (Self-critique는 2022 Constitutional AI, RLVR는 2024)
2. Compute scale 부족 (joint은 1T-scale batch 필요)
3. α tuning + task classifier 어려움

### 한계 4가지

1. α 설정 자체가 hyperparameter (틀리면 둘 다 약함)
2. RLVR (binary) vs critique (continuous) scale mismatch → normalization 필요
3. Self-critique echo chamber 위험 ([[Q5]])
4. Task classifier 약하면 α 잘못 → hidden weakness

### Ch-25/26/27 framework 연결

- [[Q15]] per-objective anchor density의 *inter-task within one stage* 확장. α가 density 균형의 explicit knob.
- [[Q9]] anchor: RLVR anchor (객관적 verifier) + critique anchor (dynamic rubric) — 두 anchor type 동시 install.
- [[Q14]] group-relative learning의 *stream-relative* 확장.
- [[Q11]] verifier hierarchy: RLVR (world) + self-critique (self) = world+self hybrid. Hierarchy 결합 instance.

### 한 줄

**Joint RL = `α * RLVR + (1-α) * critique`로 verifiable + open-ended를 single RL stage에서 동시 학습. α는 task type에 따라 dynamic. Sequential의 catastrophic forgetting 해결. K2가 first frontier instance. Hidden weakness는 task classifier + α tuning.**

---
