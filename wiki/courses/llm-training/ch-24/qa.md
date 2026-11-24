<!-- chapter: ch-24 — Modality: Reasoning Traces (CoT → Long-CoT → Step-Level)
     deps: [[read]]
     scope: clarifying questions captured during Read phase. Kernel-only.
-->

# Ch-24 — Q&A

Back to [[read]]. Cross-chapter links: [[../ch-22/qa]], [[../ch-23/qa]].

---

## Q1 — "Modality-specific drill-down" 의미?

**Question**: chapter §1 마지막 "modality-specific drill-down" 정확한 뜻?

**Kernel**:

- **Modality** = SFT/synthetic data의 *종류* (reasoning, dialogue, code, instruction-following, translation, tool-use, long-context 등). Modality 별로 verifier·teacher·augmentation 다 다름.
- **Modality-specific** = 하나의 modality에 한정.
- **Drill-down** = general 원리 → specific case로 *깊게 zoom-in*.

**왜 ch-24가 유일한 drill-down**:

| 챕터 | scope |
|---|---|
| ch-21/22/23 | *general case* — modality 무관 (taxonomy, selection, verification) |
| ch-24 | reasoning modality에 한정해서 깊게 (6-knob, 14 papers) |
| ch-25+ | 다시 general 또는 다른 modality는 얕게 |

→ course가 reasoning *외* modality는 이만큼 깊게 다루지 않음. ch-24는 Track 4 RL chapters의 *connector* (RL이 reasoning dataset을 consume).

**Transfer 활성화**:
ch-24의 6-knob framework (teacher / pool / solutions / trace style / verifier depth / budget)을 *reasoning 전용*이 아니라 *다른 modality로 옮길 수 있는 template*로 받아야 함.
- Dialogue: verifier = LLM-judge 다층 / pool = persona × scenario
- Code: verifier = unit test → execution trace / solutions = K test passes
- Translation: verifier = BLEU → round-trip → human-eval

---

## Q2 — TIR이 뭐?

**Question**: OpenMathInstruct-1/2에서 나온 TIR 의미?

**Kernel**:

**TIR = Tool-Integrated Reasoning**. CoT 텍스트 + 실행 가능한 Python 블록 + 코드 output을 *한 trace 안에 interleave*하는 형식.

**형식 spec**:
```
[CoT prose] → <llm-code>Python</llm-code> → <llm-code-output>...</llm-code-output> → [continued prose] → \boxed{answer}
```

**왜 도입**:
- LLM 자연어 reasoning은 강하지만 *arithmetic 약함* (multi-digit, fraction, factorial)
- Teacher가 자연어로 "12×7=84"라고 쓸 때 틀릴 위험 → student가 누적 학습
- TIR 해결: 어려운 산술은 Python에 위임 → final-answer가 코드 실행 결과면 거의 정확

**OpenMathInstruct-1**: Mixtral-8x7B teacher (약함) + TIR format + SymPy canonical equivalence filter. **TIR이 teacher의 산술 약점 보완**.

**OpenMathInstruct-2**: Llama-3.1-405B teacher (강함) + **TIR 버림**, pure text-CoT 채택. *Counter-intuitive*. 이유: 405B의 산술 정확도가 충분히 높아 코드 실행 *noise*(parsing, output format)가 *이득보다 큼*. → TIR 가치 = *teacher 약점이 클 때만 큼*.

**Framework 연결**:
- ch-23 Gate: TIR = 코드 실행을 *in-trace verifier*로 embed. Final-answer match보다 깊은 verifier
- ch-22 σ_synth²: TIR이 산술 distortion 감소 mechanism
- ch-24 knob 5 (verifier depth): TIR은 *중간* level (final-match < TIR < step-level/MCTS)

---

## Q3 — OpenMathInstruct 1/2 deep-dive?

**Question**: OpenMathInstruct 더 자세히?

**Kernel**:

### OMI-1 (NVIDIA, 2024-02) — *open teacher 가능성 증명*

| Element | Spec |
|---|---|
| Seed | GSM8K 7.5K + MATH 7.5K = 15K problems |
| Teacher | Mixtral-8x7B-Instruct (Apache-2.0, *weak*) |
| K | 32-64 solutions/problem |
| Format | **TIR** (CoT + `<llm-code>` + `<llm-code-output>`) |
| Filter | SymPy canonical equivalence (MATH) / numeric string (GSM8K), *final-only* |
| Output | **1.8M solutions** (~120 per GSM8K, ~100 per MATH) |
| Compute | 500K GPU-hours |
| Result | OpenMath-Llama2-70B = 84.6 GSM8K / 50.7 MATH |

**Acceptance rate ~2%**. MATH-level-5에서 Mixtral 거의 fail.

**TIR이 student에게 가르치는 inductive bias 3가지**:
1. Python syntactic correctness (Track 4 RL executor reward에 필수)
2. *"Execution result is authoritative"* — splice-back이 산술 위임 prior 형성
3. Self-segmenting boundary — Step-DPO/OmegaPRM이 post-hoc step-level signal 추출할 hooks

**Silent leak (§8 "false positives compound" 근원)**:
Final-only filter가 *compensating-error shortcut* 허용. 예: `3×4+2=14 (wrong) → 14-2=12 (wrong but cancels) → \boxed{12} ✓`. 5-10% accepted traces에 wrong intermediate. ch-24 §5 step-level verifiers의 raison d'être.

### OMI-2 (NVIDIA, 2024-10) — *teacher strength dominance 증명*

OMI-1 pipeline 거의 그대로, **3 swaps**:
1. **Teacher**: Mixtral → **Llama-3.1-405B-Instruct**
2. **Problem pool**: 15K → 600K (teacher-prompted paraphrase + novel-question by topic tag)
3. **TIR → pure text-CoT** (405B 산술이 충분히 정확 → 코드 위임 net negative)

| | OMI-1 | OMI-2 |
|---|---|---|
| Teacher | Mixtral-8x7B | Llama-3.1-405B |
| Pool | 15K | 600K |
| Total samples | 1.8M | 14M |
| Trace | TIR | pure text-CoT |
| Cost | 500K GPU-hr | 650K H100-hr |
| Best student | 70B → 84.6/50.7 | **8B → 91.7/67.8** |

**Headline ablation**: *Llama-3.1-405B at 1M samples beats Mixtral at 10M samples* → priority order: **Teacher strength > solutions/problem > problem count**.

**Scaling curve**: linear in log(dataset) up to ~5M, *flat beyond* on this teacher. Teacher upgrade = only way to un-knee.

**Question-aug ablation**: paraphrase +1, novel-question +3, total +4 MATH. MetaMath's rule-based +9 still beats teacher-prompted +4 (trade-off with augmentation diversity).

**Load-bearing caveats**:
- **Short-CoT ceiling**: OMI-2 students don't acquire backtracking. 얼마나 scale해도 o1-style 안 나옴. → ch-24 §4 long-CoT lineage가 *separate stack*.
- **Contamination leak**: 405B teacher가 MATH/GSM8K test 본 적 있을 가능성. "Novel" 문제가 thin paraphrase 가능. MinHash 방어 정도.

### 4가지 lesson (chapter §2 contention)
1. Open teacher만으로 GPT-4 distill 따라잡기 가능 — OMI-1
2. Teacher strength > data scale dominant — OMI-2 ablation
3. TIR value = teacher 약점 dependent — OMI-1 → OMI-2 swap
4. Short-CoT pipeline은 reflective 못 가르침 (ceiling) — OMI-2 caveat

### Framework 연결
- **ch-22 σ_synth²**: 405B teacher = σ_synth² 직접 감소 lever (산술 정확도 ↑)
- **ch-23 gate**: gold-answer match = gate, 단 *final-only*라 step-level σ_synth² leak → §5 step-level verifier로 해결
- **ch-22 axis 5 (aug × selection)**: K-sampling (generation-side selection) + gold-match (filter-side selection) 둘 다 적용한 textbook instance

---

## Q4 — OMI-1에서 TIR의 purpose? (CoT만으로 부족?)

**Question**: CoT block = thinking 드러내기. 그럼 TIR의 *별도* purpose?

**Kernel**:

**TIR의 본질** = *Division of labor*. *"LLM이 잘하는 것은 LLM에게, 못하는 것은 deterministic interpreter에게."*

| Component | 책임 |
|---|---|
| CoT prose | planning, decomposition, interpretation (LLM 강점) |
| `<llm-code>` Python | precise computation, symbolic algebra (LLM 약점) |
| `<llm-code-output>` | deterministic ground truth |
| Continued prose | output 해석 + 다음 step plan |

**왜 CoT만으론 부족 (LLM의 고질적 약점)**:
- Multi-digit arithmetic (1247 × 89): 약함
- Symbolic algebra (3x² + 7x - 6 = 0): 약함
- Long calculation chains: error accumulation
- → Mixtral 7B 산술 정확도 ~30%. TIR이 이 약점을 *architectural*하게 우회

**OMI-1 economics에서 TIR이 결정적이었던 이유**:
- Pure CoT acceptance rate ~1%, TIR ~2% → **dataset 크기 2배**
- 1.8M dataset이 가능했던 이유 = TIR가 *enabler*. TIR 없으면 동일 compute로 ~1M

**Ablation (OMI-1)**:
- CoT-only: MATH **-8** loss
- PoT-only (no prose): GSM8K **-5** loss
- TIR (full): 둘 다 intact
- → 두 component 모두 *load-bearing*. TIR = 둘의 합집합

**Student에게 install되는 3가지 inductive bias**:
1. **Python syntactic correctness** (Track 4 RL의 code executor reward에 필수)
2. **"Execution result is authoritative"** — *learned epistemic strategy*. 산술 어려운 상황에서 자동으로 코드 defer
3. **Self-segmenting boundary** — `<llm-code>` block이 ch-24 §5+ Step-DPO/OmegaPRM의 step-level signal 추출 anchor 역할

**핵심**: TIR ≠ CoT의 확장. TIR = *CoT가 못 푸는 부분을 다른 시스템에 위임하는 구조*. Architectural workaround.

---

## Q5 — TIR generation mechanism: LLM이 output을 직접 만드는가?

**Question**: `<llm-code-output>`은 LLM이 적는 건가? 아니면 코드 실제 돌려서 결과를 trace에 넣는 건가? Filter는 어떻게 작동?

**Kernel**:

**답: LLM이 output을 직접 만들지 않음. 외부 sandbox가 *실제로 코드 실행*하고, *실제 결과*를 trace에 splice. LLM은 그 결과를 보고 다음 token 생성.**

**Trace 안의 token ownership 2종**:
- **LLM-generated**: prose + `<llm-code>...code body...</llm-code>` 까지
- **Sandbox-injected**: `<llm-code-output>...result...</llm-code-output>` (LLM이 *읽기만* 함)

**Generation loop (per solution sample)**:
```
1. LLM generate w/ stop_tokens=["</llm-code>"]
2. Stop hit → extract code → run in sandbox → capture stdout
3. Splice "<llm-code-output>{stdout}</llm-code-output>" INTO trace
4. LLM resume from extended prompt (output을 trust하고 conditioning)
5. Repeat until final answer reached
```

→ *Interactive agentic loop*, static generation 아님. vLLM stop_tokens + sandbox call.

**Filter는 *별도 단계, 최종 answer만***:
- Trace 전체 generate 끝난 후 `\boxed{...}` extract
- SymPy canonical equivalence (MATH) / numeric string match (GSM8K)
- 통과 → accept, 실패 → reject
- 중간 `<llm-code-output>` 값이 맞는지는 *체크 안 함*

**Acceptance rate ~2%** (1.8M / (15K × K=64)).

**Silent leak (ch-24 §8 origin)**:
- Mixtral이 `print(17+5)` (실수로 + 사용) → sandbox output "22"
- LLM "22" trust하고 진행
- 최종 answer가 우연히 gold와 일치 → ACCEPT (compensating-error shortcut)
- Filter는 *step correctness 못 봄*. ch-24 §5 step-level verifier (rStar-Math, OmegaPRM)의 raison d'être

**왜 runtime splice가 필수**:
LLM이 trace를 *post-hoc*으로 코드 재실행하면 → LLM이 hallucinate한 output 기반 다음 trace 진행 → *코드-output 의미적 일관성 깨짐* → student가 그걸 학습. *Atomic transaction* 필수.

**Cost implication**:
500K GPU-hours 중 상당 부분이 *teacher inference 외* sandbox loop overhead. Per-block ms 단위 sandbox call × K=64 × 15K problem.

---

## Q6 — `<llm-code-output>` 이후에 CoT가 또 붙는가?

**Question**: Sandbox가 output을 splice한 다음 LLM이 CoT를 더 추가하는 구조?

**Kernel**:

**✅ Yes**. Trace는 *prose ↔ code ↔ output 순환* 구조 — multi-block interleave.

**Turn structure**:
| Turn | Generator | Token |
|---|---|---|
| 1 | LLM | CoT prose (planning) |
| 2 | LLM | `<llm-code>code</llm-code>` |
| 3 | Sandbox | `<llm-code-output>result</llm-code-output>` (inject) |
| 4 | LLM | Interpretation prose + next plan |
| 5a | LLM | Another `<llm-code>` (if more compute needed) |
| 5b | LLM | `\boxed{answer}` (if done) |

**Code blocks per trace (variable)**:
- ~1.5-2 (GSM8K), ~2-3 (MATH), 5-6 (MATH-level-5)
- LLM이 *언제 끝낼지* emergent decision

**Multi-block interleave가 중요한 3가지 이유**:
1. **Multi-step decomposition** — 각 sub-computation을 separate code block에 위임
2. **Intermediate value 명시화** — "So she has 12 left" 같은 articulation이 prose에 등장
3. **Conditional continuation** — next code block이 *previous output에 조건부*

**"Self-segmenting" payoff** (chapter §2):
각 prose↔code 경계 = *step* 자연 boundary. Ch-24 §5 Step-DPO가 post-hoc *step-level preference signal* 부착할 때 이 boundary를 *anchor*로 사용. Single monolithic CoT였다면 step extraction이 ad-hoc heuristic 필요. → TIR = *현재 산술 + 미래 step-level RL substrate* 둘 다 의도된 design.

---

## Q7 — TIR-trained model은 inference 시에도 stop_token + sandbox 필수?

**Question**: Training이 TIR format이면 inference 시에도 `</llm-code>` stop token 필수?

**Kernel**:

**✅ Yes — TIR-aware inference stack 필수**.

**Stop token + sandbox 없으면** student의 행동:
- `<llm-code>` 적고 → `<llm-code-output>` 안에 **자기 산술 결과를 hallucinate**
- 그 hallucinated output을 trust하고 다음 reasoning 진행
- Simple arithmetic은 우연히 맞을 수 있지만, *complex arithmetic*에서 거의 fail
- → **TIR이 도와주려 했던 정확히 그 case에서 실패** (irony)
- Net 결과: student = "CoT 모델 + format overhead", TIR training의 purpose 자체 무효화

**Inference deployment 요구사항**:
| Component | Required |
|---|---|
| Inference engine | `stop_tokens=["</llm-code>"]` 지원 (vLLM, TGI 가능) |
| Sandbox executor | Python subprocess / Docker / gVisor |
| Splice middleware | output을 engine에 re-feed |
| Security | untrusted code 실행 → isolation |

**Cost overhead**:
- Latency 2-3× (pause × N code blocks + sandbox call × N)
- Production stack: math 서비스 deploy = 별도 인프라
- 일반 chat-serving stack 재사용 불가

**OMI-2 text-CoT swap의 *implicit operational reason*** (chapter §2 명시 부분 + 학습자 도출):
> 405B teacher arithmetic 강함 (training signal에서 TIR 이득 작아짐) + TIR-trained student deployment cost 큼 → **net trade-off reverse**. Text-CoT가 *engineering simplification*도 됨.

**★ Framework extension** — ch-22 axis 3 (signal portability) 3-level decomposition:
1. **Data-intrinsic** (ch-22 original): chunked text, format-agnostic
2. **Model-specific** (ch-22 original): IFD requires same tokenizer family
3. **Inference-stack-specific** (★ ch-24 implicit): TIR requires sandbox-aware inference

→ Ch-24가 ch-22 axis 3에 *third level*을 implicit 추가. *Training signal이 deployment infra와 결합되는 경우* (TIR이 대표 instance).

---

## Q8 — TIR training이 (A) math 능력 vs (B) tool-delegation 능력?

**Question**: TIR로 학습하면 수학 자체 성능 향상인가, 아니면 코딩으로 풀어제끼는 능력 향상인가?

**Kernel**:

**주로 (B) tool-delegation. (A)는 decomposition skill 정도 미미. 순수 mental arithmetic은 *atrophy 가능*.**

**Student가 학습하는 5가지 capability**:
| Capability | TIR로 향상? |
|---|---|
| Tool-use orchestration (언제 code) | ✅✅✅ (main outcome) |
| Code generation precision | ✅✅✅ |
| Result interpretation | ✅✅ |
| Problem decomposition | ✅ 약간 |
| Pure mental arithmetic (tool 없이) | ❌ 거의 안 늚, 가끔 **atrophy** |

**Evidence**:
- OMI-1 모델을 *tool 없이* (pure CoT) inference → GSM8K 80% → **~55-60%** 추락. *80% gain이 tool에서 옴*. 
- Mental arithmetic atrophy: "What is 17×83?" → "Let me compute. `<llm-code>print(17*83)</llm-code>`" (sandbox 없는데 emit). *Defer behavior가 prior로 깊이 박힘*.
- Decomposition skill은 *delegation 동기 부여 byproduct*. Standalone하게 늘어난 건 아님.

**Teacher transfer 시각 (ch-20 axis 5 연결)**:
TIR training = reasoning capability transfer가 아니라 *epistemic strategy transfer*. Student = "math 잘하는 모델"이 아니라 "math를 *어떻게 outsource할지* 잘 아는 모델". Teacher Mixtral의 "산술 어려우면 코드 위임" prior이 그대로 inherit.

**OMI-2 text-CoT swap의 *진짜 의미***:
1. 405B teacher 산술 강함 → delegation 안 해도 정답
2. Student가 teacher의 *direct arithmetic ability* inherit
3. → Net: **internal arithmetic + tool-independence + portable skill**
- OMI-2 8B (text-CoT) > OMI-1 70B (TIR) on both GSM8K + MATH + tool-independence + mental arithmetic

**Framework view**:
- ch-22 axis 3 (signal portability): TIR signal = *tool 능력*으로 portable, *reasoning 능력*으로는 less portable
- ch-23 σ_synth²: TIR이 줄이는 σ_synth² = *teacher 산술 error component만*. *Conceptual reasoning error*는 unaffected → compensating-error shortcut 잔존
- Benchmark misleading: GSM8K/MATH 점수가 (A)와 (B) 분리 측정 안 함 → "math 능력 향상"으로 보이는 gain이 사실 "tool-delegation 능력 향상"

---

## Q9 — OMI-1 vs OMI-2 comparative analysis

**Question**: OMI-1 장점/한계, OMI-2가 어떻게 극복?

**Kernel**:

### OMI-1 strengths
- Open teacher demonstration (Mixtral Apache-2.0 only, no GPT-4)
- K-sampling + filter recipe 정형화 (이후 모든 math SFT의 template)
- TIR format pioneering (self-segmenting trace = step-level downstream substrate)

### OMI-1's 4 limitations
| # | Limitation | Symptom |
|---|---|---|
| 1 | Teacher ceiling (Mixtral) | MATH-level-5 거의 fail, OpenMath-70B MATH 50.7 |
| 2 | Problem pool 작음 (15K seed) | Variety 부족, family saturation |
| 3 | Final-only verifier | 5-10% compensating-error leak |
| 4 | TIR inference burden + signal portability | sandbox 인프라 필수, tool 없으면 ~25pt drop |

### OMI-2's 3 swaps — 각 limitation 대응

**Swap 1: Teacher Mixtral → Llama-3.1-405B** (해결: 1, 4)
- MATH-level-5 ~60-70% (Mixtral의 3배)
- 405B direct arithmetic 정확 → tool delegation 불필요 → tool-independence
- **Cost**: 500K → 650K H100-hours, academic 접근성 ↓ (NVIDIA만)

**Swap 2: Problem pool 15K → 600K** (해결: 2)
- *Teacher-prompted* augmentation: paraphrase + novel-question (topic tag 기반)
- Ablation: paraphrase +1, novel +3, total **+4 MATH**
- **새 limitation 노출**: MetaMath rule-based +9가 더 좋음 → teacher-driven augmentation은 *teacher distribution 안에 갇힘*. Rule-based (FOBAR inverse) 같은 *symbolic operator*가 더 distribution-pushing

**Swap 3: TIR → pure text-CoT** (해결: 4)
- 405B 산술 정확 + code-exec noise + inference 단순화 → text-CoT win
- 잃은 것: self-segmenting boundary (단 OMI-2 자체가 step-level 안 함이므로 직접 cost 아님)

### OMI-2가 해결 못한 / 새로 도입한 limitation
- **잔존**: Limitation 3 (final-only verifier) — 405B teacher로 leak rate 5-10% → 1-3%로 줄어듦뿐, mechanism 미해결. §5 step-level verifier 필요
- **새 A**: Test set contamination — 405B가 MATH/GSM8K test 본 적 있음. "Novel" question이 test paraphrase 가능. MinHash decontam = necessary but not sufficient
- **새 B**: Short-CoT ceiling — non-reflective. 얼마나 scale해도 o1-style backtracking 안 나옴. *Architectural ceiling, not data ceiling*. §4 long-CoT (s1, LIMO, R1 distill)이 separate stack인 이유
- **새 C**: Academic inaccessibility — 650K H100-hours = NVIDIA-only. Released dataset이 유일한 사용법

### 4 progression lessons
1. Teacher strength > data scale (OMI-2 explicit ablation: 405B 1M > Mixtral 10M)
2. TIR value = teacher 약점 dependent (OMI-1 → OMI-2 swap)
3. Question augmentation의 teacher-bias ceiling (OMI-2 +4 vs MetaMath rule-based +9)
4. Short-CoT는 reflective 못 가르침 (architectural ceiling)

### Framework axis view
| Axis | OMI-1 | OMI-2 |
|---|---|---|
| σ_synth² (ch-23) | medium | low |
| Signal portability (ch-22 axis 3) | low (TIR-bound) | high (any stack) |
| Verifier depth (ch-24 knob 5) | final-only | **final-only (unchanged)** |

**★ Insight**: chapter §2 guidance "*pick verifier first*"는 ch-24 *전체* 권고, OMI 라인 자체는 verifier 고정한 채 *teacher + augmentation*만 swap. §5의 rStar-Math/OmegaPRM이 *verifier-first* path를 별도로 다룸. 두 path가 complementary axis.

---

## Q10 — OMI에서 CoT는 어떻게 validate하는가?

**Question**: Final answer는 check하는 거 알겠음. CoT 부분은 어떻게 validate?

**Kernel**:

**답: Validate 안 함**. *Final-answer만 check* (`\boxed{}` extraction + SymPy/numeric match). 이게 §8 "false positives compound" 의 근원, §5 papers의 raison d'être.

**Filter가 *안 보는* 것**:
- CoT prose의 논리적 일관성 ❌
- 각 code block의 correctness ❌  
- Step 간 causal coherence ❌
- Wrong intermediate → cancel out → 정답 도달 case (compensating-error shortcut) ❌

**왜 OMI가 CoT validate 안 한 이유**:
1. **Cost**: K=64 × 15K × 추가 validation = 2× teacher inference (500K → 1M GPU-hr)
2. **Ground truth 부재**: CoT 정답 unique 아님 (5가지 풀이 방법 가능). LLM-judge만 가능, exact match 없음
3. **Tradeoff philosophy**: *scale로 noise 묻기* — 5-10% noise 받고 1.8M dataset 확보

**Natural mitigation (간접)**:
- **K-sampling 자연선택**: correct CoT가 정답 도달할 확률 ↑, K가 클수록 accepted set의 CoT quality 자연 boost
- **TIR splice-back anchor**: sandbox output이 *deterministic* → downstream prose는 *반드시 그 output에 conditional* → 산술 부분만 partial CoT validation

**CoT를 *직접* validate하는 4가지 방법 (다른 papers)**:
| Method | Paper | Granularity | Cost |
|---|---|---|---|
| Self-verification | MetaMath SV (§3) | trace-level | 2× teacher (self-bias 위험) |
| Step execution | rStar-Math (§5) | step-level | huge (sandbox per step) |
| MC step rollout | OmegaPRM (§5) | step-value | massive (per-step rollouts) |
| Pairwise preference | Step-DPO (§5) | step-pair | OmegaPRM scorer 필요 |

**σ_synth² 4-component decomposition** (OMI 관점):
| Component | 누가 잡음? |
|---|---|
| 1. Final-answer error | OMI filter ✅ |
| 2. Arithmetic intermediate error | TIR (inference-time sandbox 필요) ✅ |
| 3. Compensating-error shortcut | ❌ (5-10% leak) — §5가 해결 |
| 4. Conceptual reasoning error | ❌ 여전히 open |

**Selection level**:
- Trace-level (OMI: final match) ← coarse
- Step-level (§5: rStar/OmegaPRM)
- Token-level (§5: Step-DPO fine granularity)

→ OMI는 가장 coarse. §5가 progressively finer. 학습자가 §5를 *역도출* — "final-only 부족, CoT 직접 validate 필요"는 §5 papers의 정확한 motivating question.

---

## Q11 — OMI-1 benchmark도 `</llm-code>` stop + sandbox 사용?

**Question**: Benchmark 시에도 stop_token에서 멈춰서 tool이 math 풀게 하나?

**Kernel**:

**✅ Yes — 정확히 training과 동일한 TIR loop**. Reported numbers (80.2/44.5 for OpenMath-Mistral-7B, 84.6/50.7 for OpenMath-Llama2-70B)는 모두 *TIR-with-sandbox 기준*.

**Benchmark pipeline**:
```
Per-problem:
  TIR generation (stop_token + sandbox + splice 같음, K=1)
  → extract \boxed{} → match against gold
  → accuracy 누적
```

→ 사실상 *training acceptance loop와 같은 pipeline*. 차이: training은 K=64 중 정답 sample 저장, benchmark는 K=1로 정답 여부 채점.

**Sandbox vs verifier 구분**:
| Stage | Sandbox (TIR loop) | Gold-match verifier |
|---|---|---|
| Training data gen | ✅ 매 code block | ✅ Final answer filter |
| Inference (deployment) | ✅ 매 code block | ❌ (gold 모름) |
| Inference (benchmark) | ✅ 매 code block | ✅ Final answer 채점 |

**Reported numbers의 inference-stack coupling**:
- Sandbox 끄면 GSM8K 80.2 → ~55-60 (Q7/Q8)
- "Model의 80%"가 아니라 **"Model + sandbox stack의 80%"**
- OMI-2 (text-CoT)와 비교 시 fair하려면 *둘 다 native mode*에서 측정

**Hidden assumption in benchmarks**:
- TIR-trained model의 reported numbers는 *that specific inference stack에서만 valid*
- Same weight, different stack → different score
- Paper benchmark에서 *explicit하지 않은* operational fact
- → Q7에서 도출한 *signal portability 3rd level (inference-stack-specific)* 정확한 확장

**Deployment implication**: production에서 OMI-1 deploy 시 *inference 인프라 = benchmark 인프라*. "Benchmark는 sandbox, production은 pure CoT" 하면 *공식 점수 ≠ 실제 service 성능*.

---

## Q12 — OMI-2 augmented problems의 gold answer는 어떻게 verify?

**Question**: OMI-2가 15K → 600K로 problem 확장했는데, 새 problem의 gold answer는?

**Kernel**:

**답: Teacher self-consensus**. Human-verified gold 없음. ~3-5% systematically wrong.

**두 augmentation 종류**:
| Operator | Gold answer 출처 |
|---|---|
| Paraphrase | Seed gold inherit (비교적 안전) |
| Novel question | ❌ **Teacher consensus pseudo-gold** |

**Mechanism (novel question)**:
1. Teacher generates Q_new (topic tag conditioning)
2. Teacher samples K=32 solutions for Q_new
3. Majority vote on `\boxed{}` answers → pseudo-gold
4. Low consensus → reject problem
5. Solutions matching pseudo-gold → accepted

→ **Teacher = 작성자 + 풀이자 + 채점자** 다 됨. Self-referential loop.

**3 failure modes**:
1. **Systematic teacher error**: combinatorics 같은 특정 영역 bias → K=32 모두 *같은 wrong answer* → consensus로 통과
2. **Compensating-error consensus**: 잘못된 path 일관되게 → false consensus
3. **Problem ambiguity**: under-specified problem + teacher single interpretation lock-in → false consensus

**ch-23 model collapse 연결**:
```
Teacher G_0 → generates Q + solutions + consensus → pseudo-gold (all G_0)
→ Filter → SFT data → Student G_1 inherits G_0 biases (correlated, compounded)
```
σ_synth² high (모든 step이 same teacher). Real anchor 부재 → *partial model collapse risk*.

**MetaMath와 우월한 비교 (§3)**:
- FOBAR: gold = *symbolic inversion*에서 derive ("x more, has 8" → x = 5 mathematically)
- σ_synth² ≈ 0 (teacher 의견 무관)
- 이게 MetaMath +9 vs OMI-2 +4 차이의 *근본 이유* — symbolic operator > teacher consensus

**OMI-2 mitigation 시도 (partial)**:
- Strong teacher (405B) → error rate 5% → 3%, mechanism은 동일
- Low-consensus drop → false-negative ↓, but false-positive에 영향 없음
- MinHash decontam → test leak 해결, augmentation quality는 별도

**안 한 mitigation**: cross-teacher verification, symbolic operator, sandbox-executable check, human spot-check. *Cost*가 이유 (600K × additional verifier).

**Cascading verification chain**:
| Stage | Verifier |
|---|---|
| A: Problem gen | ❌ none |
| B: Answer determination | ⚠ self-consensus only (5% error) |
| C: Solution filter | ✅ pseudo-gold match |

→ Earliest stage가 bottleneck. Chapter guideline "*pick verifier first*"의 진정한 의미: Stage A/B verifier가 dominant. OMI-2는 Stage C에만 집중.

**Reported number의 hidden discount**:
- MATH 67.8 = base accuracy − (Stage A/B 5% error) − (test contamination) − (TIR-stack coupling)
- Single number 안에 3가지 caveat 누적

**★ Framework extension** — OMI-2의 진정한 weakness 위치 확정. *Solution-level verification은 강하지만 problem-level verification은 거의 없음*. 600K augmented problems 중 ~5%가 *wrong-gold-with-consensus*.

---

## Q13 — OMI-1 vs OMI-2 direct head-to-head

**Question**: 두 paper 직접 비교?

**Kernel**:

### 17-axis comparison
| Axis | OMI-1 | OMI-2 |
|---|---|---|
| Teacher | Mixtral-8x7B (약) | Llama-3.1-405B (강) |
| Augmentation | ❌ none (15K seed only) | ✅ paraphrase + novel-question (15K → 600K) |
| Total samples | 1.8M | 14M (7.8×) |
| Trace format | **TIR** | **pure text-CoT** |
| Tool-independence | ❌ sandbox 필수 | ✅ stand-alone |
| Compute | 500K GPU-hr | 650K H100-hr |
| Reproducibility | Open recipe | NVIDIA-only (405B serving) |
| Best student | Llama2-**70B**: 84.6/50.7 | Llama3.1-**8B**: 91.7/67.8 |
| Reflective capability | ❌ short-CoT | ❌ short-CoT (architectural ceiling) |
| Verifier coverage | Solution-level only (Stage C) | Same + weak Stage B (self-consensus) |

### 4 핵심 delta
1. **Teacher = single lever**: teacher upgrade가 *5 downstream change* trigger (TIR drop, augmentation enabled, harder problems pass, tool-independence, model-size reduction)
2. **Dataset 7.8× ↑, quality bottleneck shift**: compensating-error 5-10%→1-3%, 새 augmentation pseudo-gold bottleneck 3-5% 추가
3. **TIR enabler → liability**: teacher strength threshold에서 design choice가 *binary flip*. OMI-1 시점 essential, OMI-2 시점 net negative
4. **Model size economics inversion**: OMI-2 **8B** > OMI-1 **70B** on both benchmarks. Teacher upgrade ≈ *9× model scaling*

### OMI-1을 여전히 쓰는 case
- Tool-integrated agent training (TIR format이 target)
- Cross-teacher diversity (Mixtral + Llama traces mix)
- Academic reproducibility study (Mixtral re-runnable, 405B NVIDIA-only)
- Low-compute baseline

### Framework axis 패턴
**★ Verifier axis 양쪽 모두 unchanged** (final-only). OMI-1 → OMI-2는 *verifier 외 모든 knob*을 swap한 controlled study. §5 (rStar-Math/OmegaPRM)이 *별도 path*로 verifier axis를 다룸.

### One-line summary
- OMI-1 = "open teacher만으로 math SFT 가능 증명"
- OMI-2 = "teacher strength가 모든 다른 knob을 dominate 증명"
- 두 paper + §5 step-level verifier line = ch-24의 *두 complementary axis*

---

## Q14 — OMI-2 generation strategy 정리 + validation?

**Question**: 405B가 QA with CoT 만들고, 그걸 training data로 사용? 그럼 어떻게 validate?

**Kernel**:

**Generation strategy** (한 줄 정확함):
- 405B teacher가 augmented problem에 대해 K=32 text-CoT solutions 생성
- *Single forward pass*, NO TIR loop (OMI-1과 차이)
- 단 OMI-2는 augmentation도 teacher가 함 → *3-source mix*

**Generation 흐름**:
```
Seed 15K → augment to 600K (paraphrase + novel) → K=32 CoT solutions/problem → filter → 14M
```

**Validation의 4-layer 구조**:
| Layer | What | OMI-2 validation |
|---|---|---|
| 1: Problem | Q가 well-formed? unique answer? | ❌ none |
| 2: Gold answer | Q의 정답은? | Source-dependent (3 branches) |
| 3: Solution-Gold | CoT의 최종이 gold와 매칭? | ✅ SymPy/numeric (강) |
| 4: CoT reasoning | 추론 step 자체 정확? | ❌ none (Q10) |

**Layer 2 — 3 branches by problem source**:
| Source | Volume share | Gold 출처 | Quality |
|---|---|---|---|
| Seed (15K) | ~2.5% | Human-verified GSM8K/MATH | ✅ 강 |
| Paraphrase (~30%) | inherit seed gold | ⚠ paraphrase fidelity 가정 (formal check 없음) |
| Novel (~67%) | ❌ Teacher self-consensus (K=32 majority) | ~3-5% wrong |

→ **Gold quality가 source에 따라 dramatic하게 다름**. 대다수 sample이 *self-consensus pseudo-gold*에 의존.

**Compounding gap (net estimate)**:
- Novel question Layer 2 weak (5%) × Layer 4 absent (5-10%)
- ~3-8% of OMI-2 dataset에 validation gap
- Reported 67.8 MATH의 hidden discount

**"더 robust한 방법" — ch-24 후속 papers**:
| Layer | OMI-2 | 해결 paper |
|---|---|---|
| 1 (problem) | none | MetaMath FOBAR (§3, symbolic well-formed) |
| 2 (gold) | self-consensus | MetaMath symbolic derive, cross-teacher (§3) |
| 3 (solution-gold) | final-match | rStar-Math step exec (§5) |
| 4 (CoT) | none | Step-DPO, OmegaPRM (§5) |

**3-line ch-24 progression**:
- OMI 라인 (teacher scale)
- MetaMath 라인 (symbolic operator) — Layer 1-2 강화
- Step-level 라인 (verifier depth) — Layer 3-4 강화
- → 셋 다 끌어올려야 진정한 quality. OMI-2는 *neither symbolic nor step-level* → axis별로 ceiling

---

## Q15 — OMI-2는 본질적으로 distillation?

**Question**: OMI-2가 distillation처럼 느껴짐. 맞나?

**Kernel**:

**✅ Yes — *Rejection-sampling sequence-level distillation with self-consensus filter***.

**Distillation taxonomy 위치**:
| Type | OMI-2 매칭 |
|---|---|
| Logit (Hinton) | ❌ no logit access |
| Sequence-level | ✅ base mechanism |
| Behavior cloning | ✅ same |
| Rejection-sampling | ✅ K=32, final-match filter |
| Self-consensus filter | ✅ augmented portion only |
| Recursive | ❌ single generation |

**Distillation framing이 드러내는 3가지**:
1. **Student ceiling = Teacher ceiling** — 405B 못 푸는 문제는 8B student도 못 풂. *Reflective long-CoT capability는 405B에 없음 → §4 separate stack 정당화*
2. **Bias inheritance** (ch-20 axis 5 + ch-22 verdict E2 통합 view): teacher의 stylistic/mathematical/systematic-error/contamination biases 모두 student에 transfer
3. **No emergent capabilities** — pure distillation은 *transfer only, creation 아님*. Capability 확장은 teacher 변경 또는 RL 필요

**OMI-2의 *partial* distillation (seed 2.5% = anchor)**:
| Portion | Distillation purity |
|---|---|
| Seed 2.5% (human-verified gold) | ❌ external anchor (collapse 방지) |
| Paraphrase ~30% | Mid (seed gold inherit) |
| Novel ~67% | ✅ Pure self-distillation |

→ Seed 2.5%가 *작아 보이지만 architecturally critical anchor*. 100% novel이었다면 ch-23 model collapse risk 훨씬 높음.

**ch-22/23 framework lens**:
- ch-22 axis 6 (selection intent): "quality" = *teacher-defined quality* = behavior cloning. Self-referential
- ch-23 σ_synth² ≈ 0.67 × teacher_KL + 0.33 × small → *405B teacher의 distribution shift이 dominant*
- ch-23 gate: effective external coverage = seed 2.5% only. Novel에 대한 gate는 internal

**Distillation purity spectrum (다른 papers와 비교)**:
| Paper | Purity | External anchor |
|---|---|---|
| MathScale | Highest | None |
| s1/LIMO (§4) | Highest | R1/o1 trace |
| OMI-2 | High-mid | Small seed |
| OMI-1 | Lower | Full 100% seed |
| MetaMath | Lower | Symbolic derivation |
| rStar-Math (§5) | Lower | Sandbox |

**★ Paper framing puncture**:
- Paper: "open synthetic data alternative to GPT-4 distillation" (license framing)
- 실제: "**405B Llama distillation으로 GPT-4 distillation 대체**" — teacher만 바꿈
- "Synthetic data"가 *capability creation*이 아니라 *capability transfer*

→ Field-level implication: 모든 SFT papers를 *어떤 teacher의 distillation인지*로 re-read하면 true contribution이 *teacher 선택* vs *methodology*로 명확해짐.

---

## Q16 — s1의 difficulty filter mechanism?

**Question**: s1 (§4)에서 difficulty filter 어떻게 적용?

**Kernel**:

**답: Dual-baseline check** — *Qwen-2.5-7B + Qwen-2.5-32B 모두 fail*해야 retain.

**s1 3-filter pipeline (순서 중요)**:
```
59K pool → difficulty (80% drop, mechanical) → diversity (topic spread) 
        → quality (hand-check, small set) → 1K s1K
```

**Difficulty mechanism**:
```
For each Q:
  Baseline 1: Qwen-2.5-7B-Instruct → K samples → final-answer match
  Baseline 2: Qwen-2.5-32B-Instruct → 동일
  Rule: BOTH must fail → retain. Else drop.
```

→ AND-of-failures gating. *Cross-validated difficulty signal*.

**왜 두 baseline?**
- 7B-only: "7B엔 어렵지만 32B엔 쉬운" 문제 포함 → long-CoT 불필요
- 32B-only: "32B에 어렵지만 domain mismatch" false positive
- BOTH fail = model-size invariant difficulty signal

**Ablation 증거**:
| Recipe | AIME24 |
|---|---|
| Qwen-2.5-32B base | ~17 |
| Random 1K from 59K | ~24 |
| **s1K curated** | **56.7** |

**+32 AIME24 point lift = 순전히 curation에서**. Same base, same compute, same trace count, same Gemini teacher. *Difficulty filter가 dominant component*.

**Thesis "Activation not Installation"**:
- Strong base model이 reasoning capability *이미 가짐*
- SFT는 *install이 아니라 activate*
- Easy problems 포함 → 학습 신호 ≈ 0 + premature-stop bias 주입 + dilute reflective signal
- Hard problems = *maximum information gain* (long-CoT가 cap을 열어줌)
- → Difficulty filter가 *sweet spot* (hard but solvable with reasoning) isolate

**Filter 순서 rationale (cheap-first, ch-22 verdict E7 reuse)**:
| Filter | Cost | Volume drop | Position 이유 |
|---|---|---|---|
| Difficulty | High but mechanical | 59K → ~10K | Volume-cost ratio 최대화 |
| Diversity | Low | 10K → ~3K | Difficulty-passed에서 spread |
| Quality | Very high (manual) | 3K → 1K | 작은 set만 hand-check feasible |

**s1 vs LIMO 비교**:
| | s1 | LIMO |
|---|---|---|
| Filter | Mechanical dual-baseline | Hand-curated by experts |
| Reproducibility | High | Low (subjectivity) |
| AIME24 | 56.7 | **63.3** |
| MATH500 | 93.0 | **95.6** |

→ LIMO 약간 better (~7pt AIME24) but s1 mechanically scalable. LIMO 우위 원인 추정: trace style quality (reflective markers, branching, self-verification segments) 명시적 검수.

**ch-22/23/24 framework**:
- **ch-22 axis 6 selection intent**: ★ *7th axis 추가* — **capability-gap-targeted selection** (s1의 활성화 framing). ch-22 verdict E1 (performance-gap allocation)과 *isomorphic structure*
- **ch-23 σ_synth²**: s1은 Gemini-generated (high σ_synth² source) but *small 1K scale* → c(p)·σ_synth² 식에서 *p 작게 유지* 전략. *σ_synth² 줄이는 대신 contamination 비율 줄임*
- **ch-24 knob 5 verifier depth**: s1는 verifier 깊이 안 늘림 (final-only), 대신 *selection density* 극대화. OMI vs s1 = *verifier 고정한 채 scale 방향 반대*

**Punchline**: Random 1K (24) → s1K (56.7) = **+32 point는 difficulty filter alone**. "1K can beat 14M, 단 *the 1K must be carefully difficulty-filtered*". ch-22 axis 5 augmentation × selection의 극한 case: scale ≈ 0, selection 극도로 aggressive, *selection 단독 dominate*.

---

## Q-정정 (2026-09 revision)

**배경**: read.md가 2026-09 generality revision으로 새로 작성되었다. 아래 Qn의 kernel에는 이제 정정된 사실이 들어 있다. 괄호의 번호는 새 read.md "Corrections to the version you studied"의 항목 번호이다. Q1–Q16에 있는 section·line reference(예: "§8", "chapter §2")는 git commit 4a72e54 시점의 read.md를 가리킨다.

- **Q1**: "ch-21/22/23이 general case이고 ch-24가 유일한 modality drill-down"은 틀리다. General synthetic-data chapter는 ch-18–ch-23이고, ch-25–ch-28도 modality chapter이다. "Track 4 RL chapters"도 틀리며, RL chapter는 ch-40–ch-46a이다 (2–3).
- **Q2**: OpenMathInstruct-1의 generator는 Mixtral-base(5-shot)이고, paper는 SymPy filter를 명시하지 않는다. OpenMathInstruct-2에는 TIR과 text-CoT 비교 실험이 없다. Format ablation은 두 text format, OpenMath CoT 44.5 vs Llama CoT 40.6의 비교이다 (5–6, 8).
- **Q3**: Sampling 수는 GSM8K prompt당 128(총 256), MATH prompt당 224(총 896)이다. 84.6 / 50.7은 OpenMath-CodeLlama-70B의 점수이고, OpenMath-Llama2-70B는 84.7 / 46.3이다. "acceptance ~2%", "500K GPU-hours", "650K H100-hours", "5-10% wrong intermediate", "paraphrase +1 / novel +3", "MetaMath +9", "~5M 이후 flat"은 source에 없다(OpenMathInstruct-2는 14M에서 saturation이 없다고 쓴다). Teacher ablation은 Llama3.1-405B-Instruct vs Llama3.1-8B-Base(matched coverage) 37.9 vs 30.1이다. 고정 256K pairs에서 unique question 1K → 6.5K가 +10.5이므로 "teacher > solutions/problem > problem count" 순서는 source와 맞지 않는다 (5–10).
- **Q4**: "Mixtral 산술 정확도 ~30%", "pure CoT acceptance ~1% vs TIR ~2%", "CoT-only MATH −8 / PoT-only GSM8K −5" ablation은 OpenMathInstruct-1에 없다 (6).
- **Q5**: Acceptance rate 계산의 입력(K=64, 15K problems)이 틀리다. 실제 sampling 수는 Q3의 값이고, 생성은 code block 최대 3개이며 첫 execution error에서 멈춘다. SymPy filter와 GPU-hours는 보고되지 않았다 (5–6).
- **Q6**: Code block 수는 0개 16.4%, 1개 81.7%, 2개 이상 2%이다(App. A.1). Step-DPO는 TIR boundary가 아니라 prompt의 "Step i:" prefix로 step을 나누고, OmegaPRM은 solution을 길이 기준 16 pieces로 나눈다.
- **Q7, Q11**: "OpenMathInstruct-2가 TIR를 버렸다"는 전제가 source에 없으므로, 이를 근거로 한 "text-CoT swap의 operational reason"은 근거가 없다. "Sandbox를 끄면 GSM8K 80.2 → ~55-60"도 보고된 수치가 아니다. OpenMathInstruct-1 평가가 code execution을 포함한다는 점은 맞다 (8).
- **Q8**: "Tool 없이 inference하면 GSM8K 80% → ~55-60%"는 source에 없는 수치이다.
- **Q9**: "MATH-level-5 ~60-70%", "leak rate 5-10% → 1-3%", "405B 1M > Mixtral 10M", augmentation +1 / +3 / +9는 source에 없다. OpenMathInstruct-1의 512K model은 MATH validation Level 5에서 16.3%이다 (6, 10).
- **Q10**: "5-10% noise"와 cost 계산은 source에 없고, OpenMathInstruct-1은 flawed-reasoning 정답을 anecdotal하게 "rare"라고 쓴다. MetaMath SV는 "정답 여부를 판정하고 수정하는 prompt"가 아니라, question과 answer를 서술문으로 바꾸고 숫자 하나를 x로 가린 backward question이다. Step-DPO는 OmegaPRM scorer를 쓰지 않으며, 첫 오류 step은 사람 또는 GPT-4가 찾고 chosen step은 reference model이 직접 sampling한다 (11, 27–28).
- **Q12**: 새 question의 답은 32개 solution(temperature 0.7)의 majority vote이고 minimum vote threshold는 0이다. 즉 low consensus로 problem을 버리지 않는다(App. C.1, Table 9). "~3-5% systematically wrong", "MinHash decontam", "MetaMath +9 vs OMI-2 +4"는 source에 없다. Decontamination은 embedding top-5 검색 후 Llama3.1-405B-Instruct paraphrase check이다 (7).
- **Q13**: Augmentation은 "paraphrase + novel question"이 아니라 few-shot "similar question" 생성이다. Compute 수치, "compensating-error 5-10% → 1-3%", TIR → text-CoT 비교는 source에 없다. OpenMathInstruct-1 70B와 OpenMathInstruct-2 8B는 base model(CodeLlama/Llama 2 vs Llama3.1)이 달라서 "teacher upgrade ≈ 9× model scaling"은 통제된 비교가 아니다 (7–10).
- **Q14, Q15**: "seed 2.5% / paraphrase ~30% / novel ~67%" 구성은 틀리다. Table 5 기준 13.97M pairs 중 original question에 대한 solution augmentation이 2.92M(약 20.9%), 새 question에 대한 question-solution augmentation이 11.05M(약 79.1%)이다(derived). 별도 paraphrase category는 없다. s1의 trace teacher는 Gemini 2.0 Flash Thinking이고, LIMO v3의 teacher는 DeepSeek R1, R1-Distill-Qwen-32B, QwQ-32B이다 (7, 19).
- **Q16**: s1 filter 순서는 quality(자동) → difficulty → diversity이고, difficulty filter는 51,581개를 24,496개로 줄인다(약 52.5% 제거, "80% drop"이 아니다). 정답 판정은 Claude 3.5 Sonnet이 한다. "base ~17", "Random 1K ~24", "+32 point"는 틀리다: budget forcing(약 30,000 token cap)에서 1K-random 36.7 vs s1K 50.0 AIME24(+13.3, Table 2)이고, Qwen2.5-32B-Instruct는 26.7이다(Table 1). Paper는 difficulty 단독을 dominant component로 보지 않는다. Difficulty 지표만 쓴 1K-longest는 33.3이고, 세 criteria의 조합이 필요하다고 쓴다. LIMO는 v3에서 "expert hand-curation"이 아니라 rule-based chain score로 고르고 800 samples로 63.3 / 95.6을 보고한다. 817 samples는 v1이며 57.1 / 94.8이다 (16–19).

## Q-재독 1. §3 "같은 문제에 solution을 더 만드는 대신 새 problem 또는 다시 쓴 problem을 만든다"는 무슨 말인가? (2026-10-13)

**Kernel.** 수학 추론 데이터는 (문제, 풀이) 쌍이고 늘리는 방법이 두 가지: solution-side(같은 문제에 풀이를 더 만듦)와 problem-side(새 문제를 만들거나 기존 문제를 다시 써서 만듦, §3 정의). 이유: 고정된 문제 집합에 풀이를 더 만들면 곧 포화 — LLaMA-2-7B에서 answer augmentation이 약 80K 샘플에서 GSM8K 59.6%로 포화([[metamath]] §4.5). MetaMath 추가 20K의 GSM8K 증가: AnsAug(원래 문제의 새 풀이) +0.1, Rephrasing +0.4, FOBAR +2.3, SV +2.6; 다양성 증가와 정확도 증가의 Pearson 상관 0.972. 같은 방향 증거: OpenMathInstruct-2(ch-18) pair 256K 고정, 고유 문제 6.5K가 1K보다 MATH 10.5점 높음(8B 학생 1개, 수학만); MathScale — 시드 문제 약 2만에서 개념 2,018개·지식 포인트 8,892개를 뽑아 랜덤 워크로 조합, 지식 포인트 절반 시 25K 예시 macro 평균 상대 8.6% 하락. 주의: (1) 새 문제의 풀이 정확성은 보장되지 않음(MathScale: GPT-4가 5K 풀이 중 26%를 틀렸다고 판정, 교체해도 도움 없음); (2) 일반화 여부는 별개 — GSM-Symbolic에서 Gemma2-9b-it 원본 87.0 → 변형 79.1±3.0(7.9점, 2.6 SD); 위 augmentation 논문 중 GSM-Symbolic 결과를 보고한 것은 없어 Open question. see [[read]] §3.

## Q-재독 2. problem side를 다시 쓸 때 어떻게 검증하는가? (How do I rewrite the problem side to verify it?) (2026-10-13)

**Kernel.** 검증 가능성의 사다리(강 → 약). (1) 답을 보존하는 재작성: MetaMath Rephrasing — 8-example prompt로 질문을 다시 쓰고 GPT-3.5-Turbo로 풀게 한 뒤 답이 ground truth와 같은 경로만 보존(§3); 재작성이 의미를 바꾸는 빈도는 장에 측정 없음. (2) 답을 복원할 수 있는 재작성: FOBAR(숫자 하나를 x로 바꾸고 "정답이 {a*}라면 x는?"을 덧붙임)와 SV(질문과 답을 평서문으로 바꾸고 숫자 하나를 x로 가림) — 모델의 x가 원래 숫자와 같은지로 검증(정정 항목 11: SV는 "A'가 맞는지 확인하고 고쳐라"가 아님). 예시(Jane의 사과 5+3=8 → x 복원)는 이 course의 가상 예. (3) 프로그램 템플릿: GSM-Symbolic — 100개 질문을 템플릿화해 100개 변형 × 50세트(§3.2); 정답을 코드로 계산한다는 것은 장이 서술하지 않는 일반 지식. (4) gold answer가 없는 새 질문: 다수결(OpenMathInstruct-2: 질문당 32개 풀이, T 0.7, 최소 득표 threshold 0; Table 9 threshold 0/8/16/24 → 50.1/49.2/44.4/42.0, 크기 불일치), LLM judge(MathScale: GPT-4가 5K 풀이 중 26% 오답 판정, 교체해도 10.6/11.5 vs 10.2/11.1로 도움 없음), 일정 잡음 허용(OpenMathInstruct-2: ≥256K pair에서 오답 20%까지 SFT 견고, judge·reward-model 필터링 이득 없음; 정정 항목 1; 단일 연구, 수학만, 8B 학생). 다수결은 같은 교사가 반복하는 오류를 거르지 못함. 답의 검증은 문제 자체의 검증(재작성된 문제가 well-posed이고 같은 의미인지)과 다르며 장에 측정 없음. see [[read]] §3, Common mistakes.

## Q-재독 3. rStar의 다섯 가지 action과 trace 보존 방식, MCTS를 generator에 어떻게 쓰는가? (2026-10-13)

**Kernel.** 트리: 루트=질문, 노드=추론 한 단계, 루트→최종 답 경로=trace 하나; MCTS 네 단계 select/expand/simulate/back-propagate(§5). rStar([[rstar]], 추론 시점 탐색, fine-tuning 없음): generator(같은 작은 LM)가 각 노드에서 다섯 action 중 하나로 다음 단계를 생성 — A1 한 단계 생각, A2 남은 단계 전부, A3 다음 하위 질문과 그 답, A4 그 하위 질문을 few-shot CoT로 다시 답함, A5 조건을 나열해 질문을 다시 씀(정정 항목 21의 정의). 보상: 종단 노드의 보상 = self-consistency 다수결의 confidence(이 course의 풀이: 그 답이 다른 rollout 답과 얼마나 일치하는가), 경로의 모든 노드에 더함; 중간 노드 self-rewarding·외부 도구·학습된 value model 없음(§3.2). trace 보존: discriminator Phi3-mini-4k(3.8B)가 후보 trajectory의 앞 20–80%를 받아 나머지를 완성하고, 답이 일치하는 trajectory만 유지(§3.3, §4.1). 결과: 32 rollout에서 LLaMA2-7B GSM8K 12.51(few-shot CoT) → 63.91, StrategyQA 58.82 → 67.25(Table 2); LLaMA3-8B 200 GSM8K 문제에서 A3만 70.5, 다섯 action 모두 75.0(Table 1). rStar-Math([[rstar-math]])는 탐색을 7B policy와 7B PPM의 학습 데이터로 바꿈: 단계 = 자연어 comment + Python(코드가 실행되는 후보만 노드), 1–2라운드 보상 = 종단 정답 일치 ±1을 `q(s_i)^k = q(s_i)^{k−1} + q(s_d)^k`로 누적, 선택 UCT = Q(s) + c·sqrt(ln N_parent/N(s)) with c=2, SFT = 문제당 평균 Q 상위 2개의 정답 trajectory, PPM 쌍 = 단계마다 정답으로 이어진 Q 상위 2개 vs 오답으로 이어진 Q 하위 2개; 1라운드 DeepSeek-Coder-V2-Instruct(236B) 8 rollout, 2–4라운드 7B 모델 16 rollout, 3라운드부터 PPM이 초기 q 제공; UCT 예: q=2, N=4, N_parent=10 → 0.5 + 2·0.759 = 2.017. 주의: 90.0 MATH / 53.3 AIME24는 PPM이 점수 매기는 64 trajectory 필요, greedy policy는 78.4 / 26.7(Qwen2.5-Math-7B-Instruct MATH 82.6보다 낮음); 문장제만; 라운드 4 후 풀리지 않은 20문제 샘플 중 19개가 참조 답이 틀림. see [[read]] §5, figures/rstar-mcts.html.

## Q-재독 4. 비슷한 reasoning trace로 학습하면 overfitting이 될 수 있는가? (2026-10-13)

**Kernel.** "유사한 trace가 overfitting의 원인"을 직접 측정한 실험은 장에 없고, 인접한 증거가 있다. (1) 같은 문제의 풀이만 늘리면 포화: MetaMath answer augmentation은 약 80K 샘플에서 GSM8K 59.6%로 포화(LLaMA-2-7B), 추가 20K에서 AnsAug +0.1 vs FOBAR +2.3·SV +2.6; OpenMathInstruct-2는 256K pair 고정에서 고유 문제 6.5K가 1K보다 MATH 10.5점 높음(8B 학생, 수학만) → Guideline: 풀이를 늘리기 전에 서로 다른 문제를 늘려라. (2) 수학 한 영역 teacher trace로 SFT하면 좁아짐: Qwen3-14B-Base 47K 수학 문제, base → SFT(thinking trace) → RL(GRPO)의 Math avg 27.7 → 49.8 → 53.8, Non-reasoning avg 45.7 → 21.1 → 53.2, IFEval 69.2 → 42.3 → 70.0([[transferability-of-llm-reasoning]] Table 1); 한계: SFT LR 5×10⁻⁵ vs RL 1×10⁻⁶로 update 크기가 맞지 않고 저자들이 ablate하지 않음; 원인은 "수학만의 teacher trace"이지 "유사성" 자체가 아님. 같은 방향: long-CoT SFT·reasoning RL이 16개 변형 중 15개에서 constraint 준수를 낮추고 15개에서 수학 정확도를 높임(MathIF Table 4); OpenThinker3-7B(1.2M trace, 안전 데이터 없음) HarmBench 55.5 vs Qwen2.5-7B-Instruct 14.5(Table 30). (3) rStar-Math는 문제당 평균 Q 상위 2개 정답 trajectory만 보존하고 라운드마다 base에서 재학습 — 보존된 trace끼리 얼마나 비슷한지와 그에 따른 편협화는 장에 측정 없음(장의 증거는 747K 문제 중 coverage 60.17% → 90.25%). (4) 반대 방향: OpenThoughts에서 self-reflection 문구를 제거하면 평균 trace 길이 11,593 → 328 토큰, 평균 점수 51.4 → 26.3(Table 22) — 스타일 변이를 없애는 것도 위험. 가설 형식: "top-Q 선택이 거의 같은 추론 경로만 남기면 policy가 좁아질 수 있다. 근거: 장에 측정 없음. 검증: 보존된 trace 간 쌍별 유사도와 perturbed-problem 성능의 비교." 탐지 도구: GSM-Symbolic 변형(Gemma2-9b-it 87.0 → 79.1±3.0, 7.9점·2.6 SD), 수학 밖 before/after 평가; 다양성 측정 자체는 ch-22. see [[read]] §2, §3, §5, §7, Generalization lens (b).

## Q-재독 5. 영업 통화 모델에 "대화 단계 분석 → 상황 판단·다음 행동 결정" reasoning trace를 학습시키면 첫 턴에서 항상 같은 reasoning("자기소개를 해야 한다")을 하는가? (2026-10-13; 학습자의 도메인 확장 아이디어, 메시지가 영어로 번역된 형태여서 해석을 명시하고 답함)

**Kernel.** 대체로 같은 trace가 나오나 "반드시"는 아니다. (1) 학습되는 것은 p(trace | context) — 첫 턴 trace가 거의 전부 같으면 확률이 매우 높아지지만 temperature > 0이면 변형 가능(반복 prefix의 loss gradient가 (1−p)에 비례해 0으로 수렴한다는 ch-04 논의와 같은 구조; Interpretation). (2) 모델이 "상황"이 아니라 "첫 턴이라는 위치"에 반응할 수 있음 — 고객 선발화·재통화·끊김에서도 "자기소개"를 출력할 위험(Q-재독 4의 narrowing과 같은 구조; ch-24 §3, §7). (3) trace가 응답의 원인인지(trace 제거·교체 시 응답 변화)는 ch-24에서 측정하지 않았고 증거는 수학 중심 — 대화 도메인은 가설. 설계 제안(이 course의 제안, 소스로 미검증): 첫 턴 상황 변형 데이터(고객 선발화·재통화·끼어들기)를 학습과 held-out 평가에 포함; trace 제거·교체 intervention test; ch-24 §3의 perturbed-problem 평가를 대화로 옮겨 같은 통화의 다른 첫 상황별 점수 분포 확인. 영업 대화의 "좋은 다음 행동"은 정답이 없어 judge 기반 검증이 필요(ch-18 discuss의 unverifiable zone과 동일 문제). see [[read]] §3, §7, Generalization lens (c).

## Q-재독 6. MCTS에서 step을 트리에 어떻게 배치하는가? (When doing Monte Carlo searching, how do you place the steps?) (2026-10-13; 질문이 모호해 "step을 트리의 어디에 어떻게 놓는가"로 해석하고 명시)

**Kernel.** node 하나 = 부모까지의 풀이 + 새 step 하나; 같은 부모 아래 후보들은 형제(앞부분 동일, 다음 step만 다름); root→leaf = trace 하나(§5). step의 크기: rStar는 action이 정함(A1 생각 한 단계, A2 남은 단계 전부, A3 하위 질문+답 한 쌍, A4 재답변, A5 조건 나열로 다시 쓴 질문); rStar-Math는 step = 자연어 설명 + Python 코드, 한 node에서 후보 8개(4라운드 16개) 생성, 앞 step들과 함께 코드가 실행되는 후보만 node, 최대 깊이 16, 문제당 rollout 16(어려운 문제 +16, 4라운드 최대 128)(Recipe). 예시(이 course가 만든 가상 예): root → s1a "5+3=8"(자식 s2a "8−2=6" 답 6 +1, s2b "8+2=10" 답 10 −1), s1b "5−2=3"(자식 s2c "3+3=6" +1); 1~2라운드 방식 `q(s_i)^k = q(s_i)^{k−1} + q(s_d)^k`로 s1a는 q=0·N=2·Q=0, s2a Q=1, s2b Q=−1. 학습 데이터 연결: SFT = 문제당 평균 Q 상위 2개 정답 경로, PPM = 같은 부모 아래 형제 중 정답으로 이어진 Q 상위 2개(positive) vs 오답으로 이어진 Q 하위 2개(negative) 쌍 — 형제는 prefix가 같아 차이가 그 step 하나뿐. 장에 없음: step 경계의 구체적 형식(구분자·prompt), rStar에서 node마다 action을 고르는 규칙(장의 UCT 서술은 rStar-Math). see [[read]] §5, Recipe.

## Q-재독 7. thinking path(reasoning trace) 자체는 어떻게 검증하는가? (2026-11-23)

**Kernel.** 대부분의 pipeline은 trace 중간을 검증하지 않고 최종 답만 확인한다(Q10). 장에 나온 verifier를 강도 순으로 정리(§1 Verifier 정의): (0) 검증 없음 — s1은 correctness filter 없음, LIMO는 rule score(길이 30%·verification words 20%·tentative words 25%·connectives 25%)로 형식을 점수화할 뿐 정확성은 보지 않음(§4). (1) final-answer match — OMI·MetaMath·MAmmoTH; 중간이 틀린 positive는 "rare"라고만 서술, 수치 없음. (2) execution — rStar-Math는 step마다 Python이 앞 step과 함께 실행되는 후보만 node로 둠; 실행된다는 것이 맞다는 뜻은 아님(§5). (3) prefix rollout — `c_t` = prefix에서 k번 rollout한 결과 중 정답 비율. 예: 8번 중 5번 정답 → c_t = 0.625, hard label 1; 0/8 → 첫 오류 후보. 측정하는 것은 "이 prefix에서 정답에 도달할 수 있는가"이지 step의 논리적 정확성이 아님; Math-Shepherd hard label은 사람이 label한 step 160개 기준 정확도 86%(N=4), N이 크면 false positive 때문에 낮아짐(§6). (4) second model — rStar discriminator의 완성 일치, Step-DPO의 GPT-4/사람 첫 오류 위치 지정. (5) 학습된 PRM/PPM — (3)의 label로 학습하므로 (3)의 noise를 물려받음. 검증되지 않는 것: trace가 답의 원인인가(faithfulness; 장에 측정 없음, Q-재독 5), gold answer 없는 domain(rStar-Math Limits: test case·human label·mutual verification 필요). 맥락: OMI-2는 SFT가 wrong-answer 20%까지 robust하고 judge/RM filter로 이득이 없다고 보고(정정 항목 1) — SFT data에서는 step 검증보다 문제 다양성이 더 큰 요인. see [[read]] §1, §4–§6, Common mistakes L302/L305.

## Q-재독 8. 요즘 트렌드는 thinking path를 더 정밀하게 측정하는가? (2026-11-23)

**Kernel.** 두 갈래. (1) verifier 정밀화(평가·data filtering): ProcessBench(약 3,400개 풀이에 사람이 첫 오류 step 표시, PRM 평가를 best-of-N → 첫 오류 찾기로 전환), Qwen PRM lessons(MC label < LLM-judge·사람 label; MC와 judge가 동의하는 label만 쓰는 consensus filtering; best-of-N은 답은 맞고 과정은 틀린 풀이를 허용하는 편향), generative verifier([[generative-reward-models]] STaR-DPO judge RewardBench 81.9% vs GenRM 78.9%; ThinkPRM은 PRM800K label 약 1%로 학습), 증명 domain의 verifier + meta-verifier(DeepSeekMath-V2). (2) RL 학습 중에는 path를 채점하지 않음: [[deepseek-r1]]은 rule-based reward만 사용(PRM 미사용 이유: step 정의, 중간 step 판정, reward hacking); CoT monitor를 reward에 넣으면 obfuscation(Baker et al. 2025; [[metr-frontier-reward-hacking]]). faithfulness 측정: hint를 사용했을 때 CoT에 적는 비율 Claude 3.7 Sonnet 약 25%, R1 약 39%(Anthropic 2025-05). outcome-only의 한계: [[spurious-rewards-rlvr]] Qwen2.5-Math-7B 무작위 reward로 MATH-500 +21.4(ground truth +29.1), 다른 계열에서는 효과 미미. **주의**: ProcessBench, Qwen PRM lessons, ThinkPRM, DeepSeekMath-V2, Anthropic faithfulness, Baker et al.은 라이브러리에 없는 소스로, teacher의 기억에 근거한 것이며 PDF로 검증하지 않음 — ch-44에서 소스 추가 후 검증. see [[ch-44]], [[ch-42]].

## Q-재독 9. PRM이 뭔가? (2026-11-23)

**Kernel.** PRM(Process Reward Model) = 풀이의 **각 step**에 "이 step이 맞을 확률"을 매기는 reward model. ORM(Outcome RM)은 풀이 전체에 점수 하나. [[lets-verify]] 방식: step 끝 token에서 positive/negative/neutral을 예측하고, forward pass 한 번으로 모든 step 점수를 얻음; 풀이 점수 = step별 positive 확률의 곱(neutral은 positive로 계산). 예(이 course의 가상 예): step 점수 0.95, 0.90, 0.20 → 0.171, step 3이 오류 후보. Label 출처: 사람(PRM800K: 문제 12K개, 풀이 75K개, step label 약 800K개; 첫 오류 step까지만 label) 또는 자동(Monte-Carlo `c_t`: [[math-shepherd]], [[omegaprm]], ch-24 §6). 용도: best-of-N reranking(MATH 500문제, best-of-1860: PRM 78.2% vs ORM 72.4% vs majority vote 69.6%; OOD AP/AMC best-of-100: 72.9 vs 63.8 vs 61.3), search guidance(rStar-Math PPM, §5), RL reward(R1은 reward hacking 때문에 사용하지 않음, Q-재독 8). 주의: PRM과 ORM의 학습 data 양이 같지 않음(ORM은 문제당 100개 샘플로 한 자릿수 더 많음; 저자들은 matched comparison이 아니라고 밝힘). see [[read]] §5–§6, [[ch-44]].

## Q-재독 10. "최종 답이 틀린 풀이(wrong-answer solution)"는 풀이는 맞고 답만 틀린 것인가? (2026-11-23)

**Kernel.** 아니다. 최종 답 ≠ gold answer인 풀이 전체를 가리키며, 과정의 정오는 보지 않은 이름. 과정 × 답 2×2: ① 과정·답 맞음, ② 과정 틀림·답 맞음(우연; final-answer match로 못 거름 → PRM 필요), ③ 과정 맞음·답 틀림(마지막 실수, 또는 gold answer 오류 — rStar-Math 4라운드 후 미해결 20개 중 19개가 reference answer 오류, §5), ④ 과정·답 틀림(대부분). OpenMathInstruct-2 Fig. 5는 wrong-answer(④·③) 또는 mispaired(다른 문제의 풀이) 풀이를 최대 20%(256K 중 51,200개) 섞어도 정확도 변화가 거의 없음을 보였고(8B 학생 1개, 수학만), ②의 효과는 측정하지 않음. see [[read]] L94, L97; Check your understanding 2 (L315).

## Q-재독 11. random noise와 systematic noise는 무엇이고, ②(과정 틀림·답 맞음) trace는 self-correction을 가르치는가? (2026-11-23)

**Kernel.** (Q-재독 12에서 수정: 아래 구분은 정의이며 실제 ② data에는 말없는 상쇄와 명시적 reflection이 섞여 있을 수 있음.) 말없는 상쇄형 ②는 self-correction이 아님: self-correction은 trace가 오류를 명시하고 고침(LEMA의 correction, rStar-Math backtracking); 말없는 상쇄형은 두 번째 오류가 첫 오류를 말없이 상쇄한 것이라, SFT는 두 오류 token 모두의 likelihood를 올리고 새 문제에서는 상쇄가 일어나지 않음. **Random noise** = 오류가 sample마다 서로 다른 곳·방식(같은 문맥에서 정답 step 90표 vs 각 오답 1표 → 다수인 정답이 학습되고 오류는 상쇄됨; variance). **Systematic noise** = 같은 문맥에서 같은 방향으로 반복되는 오류(예: teacher가 이중 할인 문제마다 "20%+10%=30%"를 씀, 정답 1−0.8×0.9=28% → 그 문맥에서는 오답이 다수일 수 있고 같은 teacher의 data를 늘려도 줄지 않음; bias). 예시는 이 course의 가상 예. 대입: OpenMathInstruct-2 Fig. 5의 wrong-answer·mispaired 혼합은 random에 가까움; 같은 teacher의 ② 오류는 systematic일 수 있으나 ch-24 소스에서 측정한 적 없음(가설). see [[read]] L94, §5; Q-재독 10.

## Q-재독 12. trace 안에 self-reflection이 있는지 없는지 어떻게 아는가? (2026-11-23)

**Kernel.** final-answer match로는 알 수 없음 — "오류 step + 정답" trace에는 ②-a 말없는 상쇄와 ②-b 명시적 reflection("잠깐, 다시 계산")이 섞여 있고 둘 다 filter를 통과(Q-재독 11의 구분은 정의였지 관찰이 아님; 비율을 측정한 소스는 ch-24에 없음; reasoning-model teacher의 long-CoT, 예: s1·LIMO에는 ②-b가 흔할 수 있음 — 가설). 확인 방법: keyword 탐지(LIMO rule score의 verification·tentative words, §4; 형식적 "wait"일 수 있음), step label 후 뒤에 그 오류를 언급·수정한 step이 있는지 대조(PRM noise), LLM judge 분류(judge 정확도 별도 검증), 사람 sample audit(Common mistakes L302), 사례 관찰(rStar-Math Fig. 4 backtracking; 비율 아님). reflection 문장이 있어도 실제로 답을 고친 원인인지는 별개(faithfulness, Q-재독 8). loss 위치: trace 전체에 loss(s1)면 "오류를 만들고 고치기"를 학습, LEMA는 틀린 풀이를 input에만 두고 correction에만 loss(App. B.2) → 고치기만 학습. see [[read]] §4–§5, Negative samples §2.

## Q-재독 13. LIMO가 "wait"류 단어에 점수를 주면, 학생 모델은 맞든 틀리든 wait을 계속 쓰게 되는가? (2026-11-23)

**Kernel.** LIMO의 rule score는 RL reward가 아니라 SFT data를 한 번 고르는 selection 점수(문제당 최고점 chain, 이후 상위 800개; [[limo]] §3.1.2). 그러나 점수로 고르는 것은 학습 분포를 그 점수 방향으로 옮기므로 방향은 학습자 직관과 같음: verification words(20%)·tentative words(25%)가 길이로 정규화된 빈도로 점수화되고, 그 단어가 실제 오류 뒤에 나왔는지는 보지 않음 → 오류와 무관한 wait도 선택됨. 관련 증거: Sky-T1-Flash는 "But wait"·"Let me reconsider" 반복을 overthinking으로 정의하고, 자기 생성 길이 선호 pair로 SimPO 1 epoch 후 AIME24 출력 길이 6,881 → 4,365(−37%), 정확도 43.3 유지([[sky-t1-flash-overthinking]]) → 상당 부분의 reflection은 정확도에 기여하지 않았음; [[inverse-scaling-test-time-compute]]는 일부 과제에서 긴 reasoning이 정확도를 낮춤(R1 distractor 5개 70% → 30%); 반대로 s1의 "Wait" 강제는 AIME24 50.0 → 53.3(§4 표). LIMO 자체는 점수 L5 → L1로 갈수록 AIME24·MATH500 하락(§6.3.1)이라 benchmark 정확도에는 도움이었음. 미측정: LIMO 학생 모델의 wait 빈도가 직전 step의 정오와 무관한지 — 측정하려면 step label을 붙이고 "오류 뒤 wait 비율 vs 정답 뒤 wait 비율"을 비교해야 함(이 course의 제안). see [[read]] §4.

## Q-재독 14. OMI-2 20% 결과와 rStar-Math Table 7은 어떻게 동시에 성립하며, "최종 답이 틀린 풀이"도 systematic noise일 수 있지 않은가? (2026-11-23)

**Kernel.** 학습자 답: noise 종류가 다르다 — random noise는 data 안에서 상쇄되고 systematic noise(teacher bias)는 상쇄되지 않음; 두 실험은 서로 다른 것을 비교. 추가: rStar-Math 78.4 data는 MCTS라는 다른 생성 과정에서 나와 filter 효과와 생성 효과가 섞임. 반문은 타당: OMI-2 Fig. 5의 두 noise 중 Incorrect Pairing(정답 풀이를 무관한 문제에 붙임)은 40%에서도 강했고, wrong-answer 풀이는 20%까지만 거의 영향 없음(≥256K pair) — 문제와 무관한 noise가 더 잘 견뎌짐. wrong-answer 풀이가 teacher 자신의 sample이라면 teacher bias를 담을 수 있음(카드에 출처 미기재 — PDF 확인 필요). ②의 해로움을 직접 시사하는 근거: 같은 coverage에서 405B teacher 37.9 vs 8B-Base teacher 30.1(Table 2), 저자들의 예비 분석은 약한 teacher의 "답은 맞고 reasoning은 틀린" 풀이가 많은 것을 원인으로 연결(§2.2.2). see [[openmathinstruct-2]], [[read]] L94; Q-재독 10–11.

## Q-재독 15. MetaMath에서 FOBAR·SV(+2.3, +2.6)가 Rephrasing(+0.4)보다 효과가 컸던 이유는? (2026-11-23)

**Kernel.** 학습자 답: 문제 관계의 diversity — 풀이 방식 자체가 달라짐. 정리: Rephrasing은 표면(문장) diversity로 같은 계산 순서가 그대로 통함; FOBAR·SV는 같은 양들 사이의 관계에서 다른 미지수를 풀게 해(예: 5+3=? → x+3=8, 이 course의 가상 예) 계산 diversity를 만듦 → 외운 풀이 순서가 통하지 않고 양들 사이의 관계를 써야 함. MetaMath의 diversity gain(새 sample마다 기존 data의 가장 가까운 sample까지의 ada-002 embedding 거리의 평균, §4.1)은 문제 텍스트 기준이라 계산 diversity를 간접적으로만 반영; 정확도 증가와의 상관 0.972는 augmentation 종류 몇 개에 대한 상관이지 인과가 아님. 논문에 backward 문제 정확도 분석(§4.6, Fig. 6)이 있으나 수치는 카드에 없음. 같은 구분이 평가에도 적용: GSM-Symbolic은 표면(숫자·이름) 변형, GSM-NoOp은 관계와 무관한 절 추가(o1-preview 17.5%, Phi-3-mini 65% 이상 하락) — 관련 있는 관계를 골라내는지 test. see [[read]] §3; Q-재독 1–2.

## Q-재독 16. 말하는 방식·대화 방식을 가르칠 때 "이 상황에서 어떤 방식이 대화를 이어가는 데 효과적인가"를 묻는 data를 늘리면 더 attractive한 모델이 되는가? (2026-11-23; 학습자의 도메인 확장)

**Kernel.** §3 framework의 대화 domain 대응(이 course의 제안, 소스로 미검증): 관계 = 상황(고객 상태·대화 단계·목표) ↔ 효과적인 응답; 학습자 제안은 forward(상황 → 응답) 확장; FOBAR/SV에 해당하는 backward = (응답, 고객 반응)에서 그 응답이 겨냥한 상황을 추론; MathScale에 해당 = persona × 감정 × 단계 × 목표 조합(ch-23c, [[ch-25]]). 빈 곳 두 가지: (1) verifier — 수학은 gold answer로 걸렀지만(MetaMath) "효과적인가"에는 gold가 없음; rStar-Math Limits(§5): code·일반 과제는 test case·human label·mutual verification 필요; Q-재독 2 사다리의 가장 약한 단계. (2) "attractive"의 정의 — 대화를 계속하고 싶은 정도로 정의하면 sycophancy로 기울 수 있음(측정하기 쉬운 겉모습에 점수를 주면 겉모습을 배움 — Q-재독 13의 LIMO "wait"와 같은 구조); 영업 모델은 목표 달성률(예약·다음 단계 진행)처럼 결과로 측정할 수 있는 정의가 필요. see [[read]] §3, §5.

## Q-재독 17. 방향을 뒤집은(backward) data를 쓸 때 A → A'이면 항상 A' → A로 되돌아와야 하는가? (2026-11-23)

**Kernel.** 수학에서는 그렇고, 그 일치 자체가 verifier: FOBAR·SV는 원래 문제의 숫자 하나를 x로 가리므로 정답 = 가린 원래 숫자이고, MetaMath는 답이 ground truth와 같은 경로만 남김([[metamath]] §3.3, §4.1). 성립 조건 = 관계가 결정적이고 나머지 숫자가 고정되면 x가 하나로 정해짐(선형 관계). 수학에서도 깨지는 예(이 course의 일반 지식): x² = 9 → ±3, 반올림 같은 many-to-one 관계; MetaMath가 이런 경우를 어떻게 처리했는지는 카드에 없음. 대화·instruction에서는 상황 ↔ 응답이 many-to-many라 엄격한 왕복이 성립하지 않음(예: "가격이 비싸다"는 상황 → "할부 안내" 응답이지만, 이 응답에서 거꾸로 추론하면 "결제 방법 문의" 상황도 타당; 이 course의 가상 예). 소스의 대안: [[humpback]](ch-19)은 backward model p(x|y)로 web text의 instruction을 생성하고, 원래 instruction을 복원하는지가 아니라 model의 5점 rating으로 self-curation; curation 없는 data는 크기를 늘려도 AlpacaEval win rate가 오르지 않음(§3.3, Fig. 2). 약한 일치 조건(이 course의 제안): 원래 상황 A가 backward 추론의 상위 후보에 드는가; 추론된 상황에서 A'가 여전히 적절한 응답인가. see [[read]] §3; Q-재독 15–16.

## Q-재독 8 검증 메모 (2026-11-24)

Q-재독 8에서 teacher 기억에 근거한다고 표시한 ProcessBench, Qwen PRM lessons는 원문 확인 후 library card로 추가됨: [[processbench]], [[qwen-prm-lessons]]. ProcessBench 3,400개, PRM보다 prompted critic이 첫 오류 찾기에 강함(QwQ-32B-Preview 71.5, o1-mini 87.9 vs 자동 label PRM 최대 42.1; Table 3), consensus filtering(약 40% 유지, ProcessBench F1 40.1 → 46.3; §3.1.3)은 확인됨. 추가로 측정된 ②(답 맞음·과정 틀림) 비율: Qwen2.5-Math-7B-Instruct 정답 응답 중 GSM8K 5.1%, Omni-MATH 43.4%([[qwen-prm-lessons]] Fig. 6); ProcessBench 정답 풀이 중 GSM8K 3.5% → Omni-MATH 51.8%(§3.3, Table 2). 같은 860k 응답에서 MC label은 Best-of-8 최고(65.9)·ProcessBench 최저(40.1) — 평가 방식에 따라 label 출처 순위가 뒤집힘(Tables 3–4). DeepSeekMath-V2도 [[deepseekmath-v2]]로 추가(verifier + meta-verifier, §2.1). ThinkPRM, Anthropic faithfulness, Baker et al.은 아직 미검증. see [[read]] "2026-11 additions".
