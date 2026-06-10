<!-- chapter: ch-20 Q&A part 2; deps: [[read]], [[qa]]; kernel answers only -->
# Ch-20 — Reading Q&A (Part 2: Orca / Orca-2 / DSBS deep dives)

Continued from [[qa]]. Part 1 covers Q1 (rollout vocabulary + rollout-as-framework-connector).

## Q2. Orca의 16 system message는 왜 필요했나? + Prompt erasing은 augmentation인가?

**Orca v1의 hypothesis** (proof 아닌 operational claim): single style SFT → student가 (reasoning + style)을 entangle해서 학습, style이 cue로 작동. Multi-style (16개) SFT → 같은 reasoning이 다양한 surface form으로 등장 → SGD가 *공통분모 = invariant*을 추출하도록 압박 → 그 invariant이 reasoning structure.

**일반 원리 — Invariance learning by augmentation**: feature F를 학습시키려면 F를 *제외한* 모든 것을 다양화. SGD가 "여러 sample 공통 = F"로 수렴.
- Image: 회전/뒤집기 → shape invariant 학습
- WRAP ([[ch-19]] qa-deep Q11): 같은 doc 4가지 paraphrase → style-invariant content 학습
- Orca: 같은 reasoning 16가지 explanation style → style-invariant reasoning 학습

**Hypothesis가 깨지는 조건**: 16개 style이 reasoning *path 자체*를 바꿔버리면 (ELI5는 analogy로, plan-execute는 decomposition으로 — 답에 이르는 길이 다름), invariant이 reasoning이 아니라 "task type → reasoning type" 같은 얕은 layer로 collapse. Orca는 손으로 짜서 style만 바꾸고 reasoning은 보존하도록 설계.

**Prompt erasing은 augmentation이 아니다** — *shortcut removal*임.

| | Orca v1 (augmentation) | Orca-2 prompt erasing |
|---|---|---|
| Sample 수 | 늘림 (16배) | 유지 (1×) |
| 조작 대상 | Input + output 둘 다 다양화 | Input의 system msg *삭제*, output은 그대로 |
| 메커니즘 | 정보 *추가* | 정보 *제거* (channel close) |

Training/inference distribution mismatch fix: system message는 student에게 strategy를 알려주는 *shortcut*. Inference엔 그 hint 없음 → training 때 일부러 가려서 student가 *task만 보고 strategy를 internalize*하도록 압박. Response에 strategy 흔적이 남아있으니 student는 역방향으로 "이런 task → 이런 strategy" mapping 학습.

**Information bottleneck 계열**: output에 supervision 살리고 input에서 hint 제거 → model이 hint를 *내부에서 재계산*하도록 강제. 같은 trick의 다른 예시: BERT MLM (token mask), denoising autoencoder (noise mask), R1-Zero (reasoning hint 없음 → RL pressure로 자체 형성).

## Q3. Orca-2의 5 strategy는 언제 어느 것을 쓸지 어떻게 결정했나?

**Hand-crafted lookup table (task type → strategy). 알고리즘 X, GPT-4 X, classifier X.**

```
For each training sample x:
  1. x의 task type 확인 (dataset label로 이미 알려진 값)
  2. 손으로 짠 table에서 lookup
  3. 해당 strategy의 system message 가져옴
  4. Teacher(GPT-4)에게 [system_msg + x] 던지고 trace 받음
  5. (input: x만, output: trace)로 SFT data 저장  ← prompt erasing
```

예시 mapping:
- Math word problem (GSM8K, SVAMP) → step-by-step
- Multi-choice QA (BoolQ, ARC) → explain-then-answer
- Reading comprehension (RACE, DROP) → extract-then-answer
- Closed-book QA → recall-then-answer
- Simple lookup ("Capital of France?") → direct answer

**왜 5개**: 16개로 하지 않은 이유 — Orca v1의 16개는 같은 task의 *style* 차이였지만, Orca-2의 5개는 *behavior pattern* 차이. 너무 잘게 쪼개면 strategy별 sample 수가 적어져서 selection 학습이 약해짐. 5개는 empirical balance (design choice).

**진짜 bottleneck — 두 layer 다 인간**:

1. Strategy 5개를 누가 골랐나 → Mitra et al. 저자들
2. 각 task → 어느 strategy인지 누가 정했나 → 저자들의 hand-crafted table

→ Student가 학습하는 strategy-selection 능력은 **저자들 expertise의 distillation**. Scale 안 함 (새 task type 나오면 table에 row 추가하는 사람 필요).

**R1이 이 bottleneck을 어떻게 제거했나**:

| | Strategy 정의 | Strategy 선택 |
|---|---|---|
| Orca-2 | 인간이 5개 hand-craft | 인간이 task→strategy table 작성 |
| R1 | RL이 `<think>` 안에서 emergent하게 형성 | Model이 매 prompt마다 자체 결정 |

R1-Zero의 *aha moment* (line 153): 인간이 "이 task엔 reflection이 적절"이라고 정해주지 않았는데도, RL pressure 하에서 model이 스스로 *"Wait, let me reconsider"*를 emit하기 시작. Prompt erasing은 *인간의 selector*를 student에 internalize, R1은 *RL로 selector를 처음부터 인간 없이* 만듦. 두 generation 사이의 가장 깊은 차이.

## Q4. Distilling Step-by-Step — SVAMP 한 문제로 끝까지 따라가기

**용어 정리 (DSBS 이해의 prerequisite)**:

| 용어 | 의미 |
|---|---|
| T5 text-to-text | T5는 모든 task를 텍스트→텍스트로 처리. Classification head 없음. 그냥 텍스트 생성기. |
| Prefix / Tag (`[label]`, `[rationale]`) | 입력 맨 앞에 붙이는 *그냥 일반 텍스트* 문자열. 모델에게 "어떤 task를 할지" 알려줌. 특별한 token 아님. |
| Prefix-routed | Prefix가 모델의 출력 행동을 결정(routing). 같은 모델 weight, prefix만 다르면 다른 모드. T5의 표준 multi-task 방식. |
| Task L (label task) | 입력 받고 *답*만 뱉기 (e.g., `[label] x` → "8") |
| Task R (rationale task) | 입력 받고 *풀이*를 뱉기 (e.g., `[rationale] x` → "Start with 3, add 5, get 8") |
| Label-only training | Task L만 학습. Rationale은 사용 안 함. (DSBS 없는 baseline) |
| Multi-task training | Task L과 R을 *한 모델* weight으로 같이 학습 (DSBS). 같은 encoder 공유. |

비유: 친구한테 "답만 말해줘: 3+5는?" → "8" / "풀이 보여줘: 3+5는?" → "3에서 시작, 5 더해서 8". 같은 친구, 같은 뇌. 다른 framing → 다른 출력. `[label]`/`[rationale]`이 그 framing 역할.

**Stage 1 — Teacher (PaLM-540B) rationale extraction (few-shot CoT)**:

원본 sample: `x = "Mary had 3 apples. She bought 5 more. How many?"`, `y_gold = 8`. PaLM에게 3-8개 CoT exemplar (`Q → Rationale → Answer` format)를 보여준 뒤 target question 던짐.

PaLM output: `"Rationale: Start with 3 apples. She bought 5 more, so add. 3+5=8. Answer: 8"`. Parse + filter (answer==gold check) → keep. 저장: `(x, rationale="Start with 3...", y_gold=8)`.

**Stage 2 — Student (T5) training: 한 triple → 두 training sample** (task-prefix 방식, 별도 decoder 아님):

```
Sample A (label task):
  input:  "[label] Mary had 3 apples. She bought 5 more. How many?"
  target: "8"

Sample B (rationale task):
  input:  "[rationale] Mary had 3 apples. She bought 5 more. How many?"
  target: "Start with 3 apples. She bought 5 more, so add. 3+5=8."
```

같은 T5 (같은 encoder + 같은 decoder weights). Prefix가 출력 모드 routing. Loss: `L_total = L_label + λ·L_rationale` (둘 다 cross-entropy on next-token prediction — loss function 자체는 동일, target sequence만 다름). λ≈0.5, benchmark별 sweep.

**Stage 3 — Inference: `[rationale]` prefix 절대 안 씀**:

```
Input:  "[label] Tom has 4 oranges. He gave 1 away. How many?"
Output: "3"   ← 1 token decode, label-only baseline과 동일 비용
```

Rationale은 *gradient로만* student에 흘러들어옴. Inference cost zero overhead.

**왜 multi-task > label-only** (concrete): label-only는 encoder가 `[label] x`만 봄 → surface token mapping만 학습 → "bought" vs "gave away" 구분 약함. DSBS는 encoder가 `[rationale] x`도 처리해야 함 → "Start with X, add Y, total X+Y" reasoning-chain feature를 hidden state에 강제 encoding. Encoder weight 공유라 `[label]` 호출 시에도 그 feature 살아있음 → label decode 시 풍부한 representation 활용. 고전 ML 용어: **auxiliary task regularization** (segmentation aux로 classification 돕는 것과 같은 원리).

**Vs Orca / R1**: Orca·R1은 trace를 output에 always-emit, DSBS만 `[rationale]` prefix로 training-only로 씀 (inference 시 버려짐). DSBS filter는 answer match뿐 (rationale 검증 X). Rationale 길이 DSBS 50~200 vs R1 ~5K tokens (*long-CoT*가 새 axis). **"770M이 540B 이긴다"의 caveat**: ✓ DSBS-770M (fine-tuned) > PaLM-540B *few-shot* / ✗ vs *fine-tuned* PaLM. 의미: 540B는 fine-tune 비실용 → (b) small fine-tuned가 (a) huge few-shot을 parameter당 압도. **80% data**: rationale supervision이 sample당 정보 풍부 → 2025 [[s1]]/[[limo]] seed. Benchmark labels: ANLI={entail/neutral/contra}, CQA=A~E, SVAMP=숫자.

**Failure mode — wrong-rationale-correct-answer가 filter를 통과** (excerpt line 83):

PaLM이 *"3×5=15, 어 답은 8이래, 그냥 8"*처럼 reasoning은 틀렸는데 답만 우연히 맞춤 → filter (answer match only) 통과 → student가 multiplication reasoning을 학습 → test sample "had 2, bought 4"에 "2×4=8" 출력 (틀림). Open-R1이 2025년 재발견한 *"wrong-question-correctly"* (excerpt line 83). Answer-level filter는 reasoning-level 오류 못 잡음 → [[ch-19]] *"verification is the moat"*의 직계 ancestor.

**R1이 해결한 방식**: DSBS는 `PaLM(unverified) → answer filter → student`. R1은 `R1(RL+verifier로 훈련된 teacher) → answer filter → student` — *teacher의 rationale 분포 자체가 이미 verifier-grounded*라 wrong-reasoning이 거의 없음. **"DSBS thesis + verified teacher = R1-distill"** 가 ch-20 §2→§3 진화의 정확한 의미. DSBS는 *"rationale = supervision, label = byproduct"* 명제를 처음 architecture로 증명, R1은 그걸 decoder-only + always-emit + verified로 재구현.
