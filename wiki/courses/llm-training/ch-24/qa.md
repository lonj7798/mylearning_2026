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















