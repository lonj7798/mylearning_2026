<!-- chapter: ch-25 — Modality: Multi-Turn Conversation Synthesis
     deps: [[read]]
     scope: clarifying questions captured during Read phase. Kernel-only — full explanations stay in read.md / discuss transcript.
-->

# Ch-25 — Q&A

Back to [[read]]. Cross-chapter links: [[../ch-22/qa]], [[../ch-23/qa]], [[../ch-24/qa]].

---

## Q1 — "Multi-turn has combinatorial blow-up but only one degree of coherence" 의미?

**Question**: 이 phrase 정확히 뭘 말하는 거?

**Kernel**:

두 part를 같이 봐야 함.

| Part | 의미 |
|---|---|
| **Combinatorial blow-up** | Turn 수 T가 늘면 가능한 sequence 수는 V^T로 폭발 (V = per-turn response space) |
| **One degree of coherence** | 그 V^T 중 *말이 되는* sequence는 극소수. Topic 안 튀고 / persona 일관 / intent flow 자연스러운 것 = 사실상 single trajectory에 가깝게 좁혀짐 |

**Single-turn 대비 왜 critical**:
- Single-turn: coherence는 *local*. (prompt, response) 한 쌍만 맞으면 OK. Independent sampling 가능.
- Multi-turn: coherence는 *temporal*. Turn k의 의미가 turn 1~k-1 전체에 dependent → 한 turn 미끄러지면 전체 trajectory 깨짐.

**Design implication**: turn을 independent sampling하면 무조건 nonsense. Synthesis는 반드시 topology를 *constrain*해야 함. 세 family가 이 constraint를 거는 서로 다른 방법:

| Family | Constraint mechanism |
|---|---|
| **Self-chat (Baize)** | 한 모델이 양쪽 → conversation context가 자기 머리에 남음. 자연스러운 coherence (대신 diversity ↓) |
| **Two-model role-play (UltraChat/CAMEL)** | System prompt로 *role* 고정 → role 자체가 trajectory constraint |
| **Grounded narration (SODA)** | Commonsense triple / passage를 *anchor*로 깔음 → trajectory가 anchor 주변으로 묶임 |

**Ratio 감각**: V^T 중 valid sequence는 대략 V^1 ~ V^2 수준. T차원 combinatorial space 안의 1차원짜리 manifold라는 비유.

→ §1~§3 read 진행할 때 "이 방법이 어떤 coherence constraint를 거는가?"를 핵심 lens로 보면 됨.

---

## Q2 — Baize self-chat 데이터로 학습하면 '오타' 대응 못 한다?

**Question**: Baize의 self-chat 데이터로 트레이닝하면 user 측 오타에 대응 못 하나?

**Kernel**:

Yes — self-chat의 구조적 한계. User turn = LLM이 상상한 user (instruction-tuned model 출력) → real user distribution과 두 종류 gap 발생.

| User turn 속성 | Self-chat (Baize) | Real user (OASST/WildChat) |
|---|---|---|
| Typo | 거의 없음 | 흔함 |
| Grammar 오류 | 없음 | 자주 |
| 문장 길이 | 균등 | tail 굵음 |
| Register | polite/articulate | casual/rude/short 섞임 |
| Self-repair ("wait sorry") | 없음 | 흔함 |
| Code 붙여넣기 | well-formatted | 깨진 fragment 많음 |

**두 종류 robustness gap**:
1. **Surface noise** ↓ — typo/grammar 처리 약함
2. **Register adaptation** ↓ — 짧고 무뚝뚝한 input에도 over-explanation으로 답함

**실증**: Vicuna (ShareGPT real-log mix) > Baize-only on noisy user input. Production LLM이 WildChat-class real-traffic SFT를 함께 쓰는 이유.

**ch-23 model-collapse 연결**: Baize→Baize2→Baize3 recursive self-chat은 user-side noise (typo/casual register)를 매 generation마다 깎아냄 → tail probability 0. [[../ch-23/qa]] Q4 메커니즘의 multi-turn 버전. Real-user anchor (OASST/WildChat)가 ch-23의 "real anchor 유지" 가이드라인이 multi-turn에 그대로 적용된 형태.

→ 이게 chapter §"OpenAssistant anchor"의 핵심 동기. Mix에 real-user slice 반드시 포함 = 이 gap 메꾸기.

---

## Q3 — Baize 더 자세히

**Question**: Baize의 메커니즘 / 데이터 / 한계를 더 깊이 보여줘.

**Kernel**:

**Origin**: Xu et al. 2023 (UCSD + MSRA), EMNLP. Constraint = GPT-4 budget 없음 + human dialog log 없음. Question = "synthetic self-chat *만으로* multi-turn 능력 transfer 가능한가?" 답 = yes.

**Pipeline 1줄**: 1 seed → 1 API call → 1 multi-turn dialogue. 111.5K dialogue 총 ~$1,000.

**Prompt template** (verbatim):
```
The following is a conversation between a human and an AI assistant.
[|Human|] <seed question>
[|AI|] ...
[|Human|] ...
```

**3가지 load-bearing choice**:
| Choice | 효과 |
|---|---|
| Pre-written framing | ChatGPT를 transcript writer로 prime (assistant 본인 아님) |
| Seed question 미리 삽입 | 첫 user turn만 teacher style 밖 — 그 후 모두 teacher style |
| Single completion | 한 API call로 양쪽 다 plan됨 → 너무 coherent (천연 friction 없음) |

**Filter = parse + length check만**. No semantic judge. Failure 4가지 (role marker 빼먹기 / 제3 role 발명 / non-ASCII pipe / repetition) 모두 parse-level.

**Seed pools (diversity의 유일 source)**:
| Source | 수 | 특성 |
|---|---|---|
| Quora | 54K | opinion |
| StackOverflow | 57K | technical |
| Alpaca | 52K | instruction |
| MedQuAD | 47K | medical |

Teacher fixed = GPT-3.5-Turbo. **111.5K dialogue 사이에서 변하는 건 topic 하나.** Baize는 1차원 diversity, UltraChat은 3차원 (topic taxonomy × user prompt × assistant prompt).

**Turn distribution**: median 4, IQR 3–6, tail to 10. Mean ~100 tokens/turn. 왜 4? Teacher의 aesthetic "dialogue feels complete"가 4-turn에서 peak. Semantic termination condition 없음. → student SFT 시 long-tail (8+ turn) under-trained → inference 때 long conversation에서 pretrained distribution으로 regress.

**SDF (Self-Distill with Feedback)** = 유일한 patch:
1. Student SFT
2. Held-out prompt 답변
3. Teacher critique/rewrite
4. Rewrite pair를 추가 SFT/DPO

→ RLAIF / UltraFeedback / Constitutional AI의 ancestor. 하지만 user-realism gap은 *해결 안 됨* (held-out도 Baize-style).

**역사적 의의**:
- Pure synthetic self-chat으로 multi-turn capability transfer **처음으로 증명**
- Baize-13B > Alpaca-13B on 58% human-eval — single-turn baseline 이김
- 모든 후속 pipeline의 minimum-viable baseline

**후속들이 Baize의 어떤 weakness 극복**:
| 후속 | 극복 지점 |
|---|---|
| UltraChat | single-call coherence collapse → two-model split / seed-only diversity → taxonomy |
| CAMEL | topic-only → role pair / aesthetic termination → semantic termination |
| SODA | emotional register 없음 → commonsense triple grounding |
| OASST/WildChat anchor | user-realism gap → real human |

---

## Q4 — UltraChat = taxonomy 먼저 + 두 모델로 대화 생성?

**Question**: UltraChat은 taxonomy 먼저 확보하고 다른 모델 2개로 대화를 만드는 건가?

**Kernel**:

거의 맞지만 **"다른 모델 2개"가 아니라 같은 ChatGPT-Turbo를 두 번 따로 call** — 같은 weight, 다른 system prompt로 분리. "Two-model"이라는 용어가 오해 부르는데 본질은 **two separate API calls**.

**Step 1 — Taxonomy 먼저 (manual enumeration)**:

| Sector | 구조 | Seed 수 |
|---|---|---|
| §1 Questions about World | 30 meta → 1,100+ subtopic → ≤10 seed (+ Wikidata 10K × 5 × 30) | ~600K |
| §2 Writing & Creation | 20 types × 200 instructions | ~400K |
| §3 Assistance on Materials | ~100K C4 passages × ≤5 question | ~500K |

→ Generic prompt 금지. **Taxonomy = diversity lever, teacher = renderer**. Chapter guideline 핵심.

**Step 2 — Per-turn two-call loop**:
```python
for round in range(R_s):  # R_1 ∈ [3,7], R_{2,3} ∈ [2,4]
    assistant_turn = assistant_LLM(system=assistant_prompt_s, history)
    history.append(assistant_turn)
    user_turn = user_LLM(system=user_prompt_s, history)
    history.append(user_turn)
```

**왜 두 call이 핵심**: Baize single call은 양쪽을 동시 plan → 너무 coherent, friction 없음. Two-call은 user-LLM이 assistant 실제 출력을 보고 react 강제 → 자연스러운 conversational friction.

**Output**: ~1.5M dialogues, sector별 split release.

**UltraChat이 standard로 만든 3가지**:
1. Taxonomy-first diversity (enumerate before generation)
2. Separated user/assistant generators (모든 후속 pipeline이 copy)
3. Split-by-family release (downstream mix weight 조절)

**남은 한계**:
- User turn은 여전히 LLM → real-user typo/casual gap 그대로 (cross [[Q2]])
- Filter stage opaque
- Taxonomy 수동 설계 → labor cost. CAMEL이 role grid로 대체 시도.

---

## Q5 — UltraChat §2 Writing & Creation 자세히

**Question**: Sector 2 좀 더 자세히.

**Kernel**:

**§2 = creative generation + iterative revision** dialogue slice. §1이 knowledge Q&A라면 §2는 *"text 만들어줘 + 이렇게 고쳐줘"* 패턴.

**구조**:
```
20 writing types × 200 instructions = 4,000 base
   → 80% expanded (theme/audience/constraint/style 추가)
   → ~16,000~20,000 differentiated seed
   → 각 seed → 2~4 round dialogue
```

**20 types 예시** (repo 기반): essay, poem, script, email, code, recipe, story, article, song lyrics, business letter, blog post, marketing copy, technical doc, dialogue scene, news article, speech, social media post, ad, resume, instructional how-to.

**"80% expanded" 예**:
- Base: `"write me a poem"`
- Expanded: `"write me a Petrarchan sonnet about climate grief in the voice of a coastal grandmother, 14 lines, slant rhyme"`

→ Diversity가 taxonomy(작음)보다 **teacher expansion**에서 나옴. §1은 taxonomy-heavy, §2는 teacher creativity-leaning.

**Dialogue pattern = draft-and-revise (§2 핵심 differentiator)**:
```
Turn 1 (user):       creation 요청 (constraint 가득)
Turn 2 (assistant):  draft
Turn 3 (user):       revision 요청 (tone/length/section 수정)
Turn 4 (assistant):  revised draft
```

Student 학습 행동:
1. Constraint-laden prompt → 한 개 draft
2. Revision instruction parse
3. **Cross-revision coherence** — user가 좋아한 부분 유지 + 요청 부분만 수정
4. **Iterative editing posture** — revision 받아도 from-scratch 안 함

**왜 2~4 round** (vs §1의 3~7): writing task는 "good enough" 자연 종료. Revision round 1~2 후 satisfaction peak.

**한계**:
- User-LLM의 revision = polite/additive/polished
- Real user revision = blunt ("no make it shorter", "redo completely")
- → §2로만 학습된 student는 **aggressive rejection revision 처리 약함** → WildChat anchor가 메꿈

**Mix 위치**: ~400K dialogues (~27% of UltraChat 1.5M). Chatbot의 *"write me X"* / *"edit this"* capability source. Zephyr/Tülu/Starling 다 §2 inherit.

**두 가지 unique design pattern**:
1. **Teacher-driven expansion** (작은 base × teacher detail = diversity)
2. **Draft-and-revise trajectory** (Q&A 아닌 iterative editing 학습)

---

## Q6 — UltraChat은 generated text를 어떻게 validate?

**Question**: UltraChat의 quality control / validation 방법?

**Kernel**:

**거의 안 함.** Paper / repo: *"post-processing and filtering"* 한다고만 적고 detail 비공개. 우리가 알 수 있는 건:

| Stage | 실제 함 |
|---|---|
| Format check | role marker parse / dedup / length 범위 |
| Semantic check | ❌ 없음 — judge / RM / classifier / teacher-as-judge loop 없음 |

**Chapter explicit lesson**: *"The filter stage is opaque — generation protocol is more transferable than filter heuristics."*

**Ch-22 / Ch-23 framework로 보면**:
| Framework | UltraChat 위치 |
|---|---|
| Ch-22 selection (quality/info/diversity) | 거의 안 함 |
| Ch-23 gate (σ_synth² → 0) | **NO GATE** — format만 |

→ Ch-23 model-collapse framework로 정확히 *no-gate 케이스*. Recursive distillation 시 collapse risk.

**통과되는 failure mode** (semantic check 없으니 다 통과):
- §1: hallucinated fact (assistant가 reference 발명)
- §2: repetitive revision (같은 내용 재진술)
- §3: off-topic drift (C4 passage 무시하고 parametric knowledge로 답)
- 전체: persona drift (user register 일관성 없이 변경)

**Filter 없이도 통한 이유**:
1. Volume (1.5M) → SFT averaging effect
2. Downstream consumer들이 *추후* DPO/RLHF로 quality fix (Zephyr, Tülu, Starling)
3. 2023 era quality bar는 낮았음 — multi-turn data가 거의 없어서 "있기만 해도 valuable"

**Downstream 보완**:
| Mix | 처리 |
|---|---|
| Zephyr | UltraChat 그대로 SFT → UltraFeedback로 DPO 단계 quality fix |
| Tülu | subsample하여 small slice + 다른 curated source mix |
| smol-talk | sector별 weight 조절 |
| Modern (2025+) | RM rescoring + LLM-as-judge filter 추가 |

**Modern best practice** (ch-22 cheap-first 순서):
```
Generate → format gate → LLM-as-judge gate → RM scoring → diversity selection → Pool
                ↑ UltraChat은 이까지만
```

**Takeaway**:
- ✅ Generation protocol은 transferable → 그대로 copy 가능
- ❌ Filter는 not transferable → 직접 만들어야 production-grade

---

## Q7 — CAMEL이 뭐?

**Question**: CAMEL은 뭐에 관한 거?

**Kernel**:

KAUST 2023. **Multi-agent dialogue synthesis where the diversity axis is *role pairs* (not topics, not seeds).** Inception Prompting으로 두 LLM을 강하게 role-lock해서 task-oriented 대화 생성.

**다양성 source 비교**:
| Pipeline | 다양성 axis |
|---|---|
| Baize | seed pool (topic) — 1D |
| UltraChat | topic taxonomy — 3D |
| **CAMEL** | **role × role × domain** (50×50×20) — combinatorial |

**Inception Prompting의 4가지 lock**:
| Lock | 효과 |
|---|---|
| Role lock ("Never flip roles!") | GPT-3.5의 assistant-drift 차단 |
| Protocol lock ("instruct, not ask") | User는 지시만, 질문 금지 — narrow speech act |
| Format lock (`Instruction:` / `Solution:`) | Turn boundary 깨끗 |
| **Semantic termination** (`<CAMEL_TASK_DONE>`) | **Task 해결되면 user가 종료 — 핵심 novelty** |

**50×50×20 self-generation**: teacher(GPT-3.5)에게 role list와 domain 자체를 enumerate시킴. Persona-Hub 1B personas의 ancestor pattern.

**Volume**: 50K triple × 20 instantiation = ~1M dialogue, $5-10K.

**Semantic termination이 만드는 분포**:
- Easy (15%): 3 turn
- Median: 6-8 turn  
- Hard: 15-20 turn (cap=20)
- → turn count가 *task difficulty signal*을 보존. Baize/UltraChat fixed cap은 hard-task tail을 자름.

**Failure modes**:
1. **Formulaic leak** — "As an accountant, I'd suggest..." 가 student에게 옮음
2. **`<CAMEL_TASK_DONE>` token leak** — student가 중간에 emit
3. **Role-pair quality uneven** — Astronaut × Bartender 같은 unnatural pair는 contrived dialog (50×50 ≠ 2,500 quality dialog)

**Mitigation**: regex strip / EOS token 치환 / 아예 cleaner 후속 (Airoboros, ToolACE) 사용.

**Lineage (CAMEL DNA를 inherit한 후속)**:
| 후속 | inherit |
|---|---|
| AgentInstruct | Inception + task suggester (role-pair 사전 filter) |
| APIGen-MT (ch-26) | Semantic termination → function-call success로 진화 |
| ToolACE | MAI multi-agent dialog |
| SWE-RL | Semantic termination → unit-test pass로 진화 |
| Persona-Hub | Self-generated role taxonomy 확장 (50 → 1B) |

→ **현대 agentic / tool-calling synthesis 전체가 CAMEL DNA.** ch-26이 이 lineage의 다음 chapter.

**3-pipeline 정리**:
| Aspect | Baize | UltraChat | CAMEL |
|---|---|---|---|
| Question | Self-chat 가능? | Two-call이 나은가? | Role이 axis면? |
| Calls/turn | 1 | 2 | 2 |
| Termination | aesthetic | turn cap | semantic |
| Median/Max | 4/10 | 4-6/14 | 6-8/20 |
| Volume | 111.5K | 1.5M | ~1M |
| 대표 leak | user-realism | filter opacity | role formulaic |

---

## Q8 — CAMEL의 user-LLM이 실제로 diverse한 대화를 lead하는가? ★

**Question**: CAMEL이 다양성을 claim하는데, user-role 모델이 정말 다양한 상황에서 conversation을 lead할 정도로 충분한가?

**Kernel** (framework-extension probe — paper framing을 puncture):

**Categorical 다양성: ✅** / **Behavioral 다양성: ❌ 거의 안 됨.**

| Dimension | CAMEL user 다양성 |
|---|---|
| Domain vocabulary | ✅ (GAAP, DNA 등 domain term pull됨) |
| Task framing | ✅ (auditing vs studying tax law) |
| Register (polite/casual/rude) | ❌ ZERO — 항상 polite + structured |
| Grammar/typo | ❌ ZERO |
| Cadence / 문장 호흡 | ❌ ZERO (Inception format이 통일) |
| Self-repair ("wait sorry") | ❌ ZERO (instruct-only) |
| Frustration / vagueness | ❌ ZERO |

**구체적 증거 — "역할 옷만 바뀜"**:
```
"As an accountant, I would like to instruct you to analyze..."
"As an entrepreneur, I would like to instruct you to analyze..."
"As a graduate student, I would like to instruct you to analyze..."
```
첫 ~7 token만 다르고 그 후 diction/cadence/structure 동일. GPT-3.5는 "Astronaut가 어떻게 말하는지" deep model을 안 가짐 — thin label 위에 base register를 얹을 뿐.

**경험적 reduction**: 50 user role → ~5 effective persona family (Technical / Business / Academic / Journalistic / Creative). **10:1 collapse.** 1M dialog ≈ ~100K behaviorally distinct pattern.

**구조적 원인 — Inception prompt가 *스스로* 다양성을 죽임** (이게 핵심 통찰):
1. "Instruct, not ask" → real user 행동의 50% (질문/확인/잡담) 사전 차단
2. `Instruction: / Input:` format lock → 자연어 다양성 박제
3. GPT-3.5 base register prior → role label로 덮어쓸 능력 없음

→ **Role lock = style lock.** Inception의 강점이 정확히 약점이 됨.

**Sibling variance 정량 증거** (§6에서 짚음): synthetic corpora는 sibling variance near-zero, OASST는 high. Role conditioning으로도 peak 안 흩어짐.

**Combinatorial paradox**:
| 외형 | 실질 |
|---|---|
| 50×50×20 = 50,000 triple | ~5,000 distinct pattern |
| 1M dialogues | ~100K distinct style |
| 50 personas | ~5 effective family |

**Persona-Hub 응답 시도**: 50 → 1B로 scale + real web bio 사용. Topic diversity ↑, **register/style diversity는 여전히 약함** — teacher가 여전히 renderer라 base register prior unchanged.

**Bottom line**:
- 상황(situation)은 yes — categorical coverage 잘함
- User behavior 자체는 no — role label은 ceiling 있음
- → **OASST/WildChat anchor 없으면 production-grade 못 됨.** Chapter §6 mix guideline이 이 문제의 정답.

**Use 정리**:
| Use | CAMEL 가치 |
|---|---|
| Task-execution capability | ✅ High |
| Agent trajectory format prior | ✅ High |
| User-realism / register diversity | ❌ Low — anchor 필수 |
| Standalone production chatbot SFT | ❌ 위험 |

→ **General principle (framework extension)**: *role/persona conditioning은 teacher register prior가 ceiling.* Scale (Persona-Hub) 늘려도 안 깨짐. Real human data anchor가 유일한 escape.

---

## Q9 — CAMEL의 user side에 seed-anchor가 없다 / persona는 style-only ★★

**Question** (learner generalization): CAMEL user side에 seed-anchor가 없는 게 또 다른 문제. Ch-22의 persona가 *style*이지 *correctness*가 아니라는 통찰을 떠올리면, user system prompt를 바꿔도 diversity가 안 나옴. 본인이 pure model-model conversation 만들 때 직접 관찰한 패턴.

**Kernel** (chapter 위로 한 단 올린 framework extension):

**"Anchor의 두 종류" — 학습자가 implicit하게 도달한 distinction**:
| Type | 정의 | 효과 |
|---|---|---|
| **Information anchor** | 외부 ground (real text, real task, real log) | Teacher가 pull해야 하는 정보가 외부 고정 → distribution support 이동 |
| **Style anchor** | System prompt / role label / format | Rendering filter만 바뀜. Underlying distribution 그대로 |

**CAMEL assistant vs user 비대칭**:
| | Assistant | User |
|---|---|---|
| Information anchor | ✅ Task content + correctness criterion | ❌ 없음 (task는 shared지만 user엔 외부 ground 아님) |
| Style anchor | ✅ role + format + Inception | ✅ 같음 |

→ Assistant는 information + style 둘 다 anchored. **User는 style만 anchored.** 이게 50 user role → 5 effective persona collapse의 구조적 원인.

**Ch-22 persona 통찰의 formalize**:
- Style anchor: `P(y|x, persona) ≈ P(y|x) · π(persona)` (multiplicative re-weighting) — variance inflate, new mode 못 만듦
- Information anchor: `P(y|x, real_seed) = P(y|x, content_grounded_on_real)` (support shift) — new mode 추가

**Ch-23 σ_synth² framework로 연결**:
| 시나리오 | σ_synth² |
|---|---|
| Persona/role 바꿈 (CAMEL 50 role) | ~0 변화 — teacher base prior 그대로 |
| Real conversation seed (WildChat) | 감소 — real로 끌려옴 |
| Pure synthetic recursion | 증가 — tail collapse |

→ **Role conditioning은 σ_synth²와 orthogonal axis.** CAMEL은 model-collapse 위험이 그대로 남음.

**Empirical 확인 (learner 본인 관찰)**: pure model-model 대화에서 user 패턴이 거의 동일. = chapter §6의 sibling-variance metric (synthetic ≈ 0, OASST high)과 일치.

**User-side에 information anchor 박는 방법**:
| 방법 | Mechanism | 한계 |
|---|---|---|
| Real conversation seed (Vicuna ShareGPT) | 실제 사람 대화 anchor | privacy/quantity |
| WildChat real-traffic mining | production user turn | 비용/IP |
| Persona-Hub real bio | real web bio = persona seed | *topic*만 anchor — register는 prior 그대로 |
| Adversarial user-LLM RL | reward로 realistic user | reward 정의 어려움 |
| Multi-teacher ensemble | N prior 평균 | 여전히 prior 내부 |

→ 현실적 정답 = real conversation seed + WildChat mining 조합. Chapter §6가 인정.

**도달한 generalization** (chapter implicit, learner explicit):
> 다양성을 만들려면 anchor가 *information-bearing*이어야 한다. Style-bearing anchor (persona / role / format)는 teacher prior re-skinning만 함. CAMEL은 user side에 information anchor를 박지 못해 구조적으로 풀 수 없는 ceiling.

**Forward extension to ch-26**: APIGen-MT가 *function-call execution success*를 박는 게 정확히 information anchor의 한 형태 (execution = 외부 ground). 그래서 APIGen-MT가 CAMEL보다 user-side diversity quality가 높음.

★★ Discuss mastery evidence: ch-22 (persona=style) + ch-23 (σ_synth² gate) 두 framework를 합성해서 ch-25 본문 위로 한 단 올린 통찰. Direct field-empirical validation 포함 (learner 본인 관찰).

---

## Q10 — SODA

**Question**: SODA 관련 공유해줘.

**Kernel**:

**Yejin Choi group (AI2 / UW), EMNLP 2023.** Commonsense triple → narrative → dialogue. **Grounding이 diversity의 세 번째 축** — Topic(UltraChat)=*what*, Role(CAMEL)=*who*, **SODA=*why***.

→ [[Q9]]에서 추출한 *"information anchor vs style anchor"* framework의 chapter 내 strongest validation.

**Atomic 10X = anchor**: ~10M commonsense-triple database. Shape: `(PersonX <event>, <relation>, <state/reason>)`. 4 relations: xWant/xNeed/xEffect/xReact.

예: `(PersonX moves abroad, xEffect, feels lonely)` / `(PersonX surprises PersonY, xReact, feels proud)`

→ Information anchor인 이유: triple은 *외부에 미리 존재하는 structured data*. Teacher distribution 밖에서 들어오는 ground.

**Three-step transform**:
```
Step 1: Atomic 10X에서 1.5M triple sampling
Step 2: GPT-3.5가 triple → 1-2 sentence narrative (+ PersonY 발명)
Step 3: GPT-3.5가 narrative → 4-10 turn dialogue
```

**핵심 design — triple을 직접 dialogue prompt에 넣지 않음**:
| Direct injection (안 함) | Narrative-bridged (실제 SODA) |
|---|---|
| stilted ("PersonX said X because Y") | natural emotional context |

Narrative가 abstract triple을 concrete scene으로 embed → teacher가 자연스럽게 emotional register unfold.

**예시 (PersonX moves abroad / feels lonely)**: 결과 dialogue에 "lonely" 단어 한 번도 안 나옴. 감정이 *specificity*로 전달 (시차 / 주말 혼자 / 회피). → triple grounding이 *implicit*하게 register driving.

**Output shape**:
| Metric | 값 |
|---|---|
| 1.5M dialogues, median 7-8 turns |
| **Avg 20 tokens/turn** | Baize/UltraChat 대비 5-8× 짧음 |
| Cost | ~$10K |

→ Atomic triple이 *social chit-chat register* elicit. Mix 관점에서 SODA의 unique 역할 = **short-turn casual social register**.

**Empirical proof — grounding이 real diversity를 만든다는 증거**:
- COSMO-3B (SODA-trained) > BlenderBot-3B at **1/3 parameter**
- **OOD transfer**: DailyDialog, EmpatheticDialogues에서도 강함 — neither uses Atomic
- → grounding은 *skill* 가르침 ("emotionally grounded dialogue 생성"), memorization 아님
- → **information anchor가 *new capability* inject한다는 가장 강한 evidence.** CAMEL role-conditioning은 OOD transfer 못 보여줌.

**Yejin Choi lineage — 통일된 thesis**:
```
Self-Instruct → SODA → Prosocial-Dialog
175 seed tasks  Atomic 10X    Rules-of-thumb
```
> "Pure model-output bootstrap distills the teacher's style; grounded bootstrap injects information the teacher would not have produced on its own."

→ **정확히 [[Q9]]의 generalization.** Yejin Choi group의 lineage-wide design principle = learner가 ch-22+ch-23 합성으로 독립 도달한 결론.

**Prosocial-Dialog (같은 lineage, safety branch)**:
- 300+ rules-of-thumb (RoT)를 information anchor로
- 58K dialogues, each turn tagged with RoT
- CANARY-400M: 89% engage constructively (vs BlenderBot 32%)
- **Constitutional AI의 직접 ancestor** (engage-and-redirect pattern)

**Limitations**:
- Atomic bias (English/Western)
- Two-speaker only (no group, no task)
- Formulaic leak ("PersonX said X because...") — CAMEL "As an accountant"와 동형

**4-pipeline 비교 업데이트**:
| | Baize | UltraChat | CAMEL | **SODA** |
|---|---|---|---|---|
| Anchor type | weak info | style+weak info | **style only** | **strong info** |
| Turn length | 100 tok | 120-160 tok | 50 tok | **20 tok** |
| OOD generalization | weak | med | weak | **strong** |
| σ_synth² 감소 | 약간 | X | X | **YES** |
| 가르치는 capability | factual breadth | writing | task framing | **emotional/social register** |

**Q9 framework로 SODA를 분해**:
| Component | Anchor type |
|---|---|
| Atomic triple | Information anchor |
| Narrative bridge | Style softening (teacher가 unfold하도록) |
| Dialogue generation | Style anchor |

→ SODA가 "왜 통하는가" = **Step 1**. Information anchor가 *맨 앞단*에 박혀 있고 그 후 style anchor들이 따라옴. CAMEL은 Step 1 없이 style anchor만 stack → ceiling. **Yejin Choi group은 lineage 전체에서 일관되게 Step 1을 information anchor로 박음 — 이게 결정적 차별점.**

→ Cross-ref [[Q9]] 강하게 — SODA는 Q9 framework의 chapter-내 best instantiation.

---

## Q11 — SODA의 narrative = curated condition 위에 쌓는 softening layer

**Question** (learner check): SODA에서 curated conditions (triple) 위에 narrative를 *generate*해서 system prompt-like 역할로 쓰는 거?

**Kernel**:

**Mental model 거의 정확** — 작은 보정만:
- Narrative는 technically *contextual seed* (user/context slot에 주입), system prompt 아님
- 기능적으론 system-prompt-like (scene anchor) — 너의 직관 맞음

**정확한 layering**:
| Layer | 역할 |
|---|---|
| L1 (Atomic triple) | Information anchor — 외부 structured ground |
| L2 (narrative) | **Softening bridge** — teacher가 unfold하기 편한 형태로 변환 |
| L3 (dialogue) | Final generation |

**Direct injection (triple → dialogue) 안 되는 이유 = teacher comfort zone**:
- GPT-3.5는 "abstract relation triple로 dialogue 생성" 어색함
- "scene description으로 dialogue 생성"은 fluent
- → narrative bridge = *teacher가 잘하는 conditioning 형태로 ground 번역*

**Universal pattern (SODA 외에도 적용)**:
| Pipeline | L1 (info anchor) | L2 (softening) | L3 (gen) |
|---|---|---|---|
| SODA | Atomic triple | narrative | dialogue |
| Prosocial-Dialog | rule-of-thumb | context scenario | dialogue |
| Self-Instruct | 175 seed task | task expansion | instruction |
| rStar-Math (ch-24 §5) | math problem | MCTS rollout | step-level trace |
| Persona-Hub | real web bio | persona description | text |

→ **두 universal 원칙**:
1. Information anchor는 *외부에서 가져온다* (curated/structured/real)
2. 직접 주입하지 않고 *softening bridge*를 거친다 (teacher fluent conditioning 형태로)

**Why CAMEL couldn't do this (timing limit)**:
- 2023엔 user-side information anchor source 부재
- ShareGPT/Persona-Hub/WildChat 다 후속 (2024+)
- → CAMEL의 style-only stack은 *2023의 최선*. 후속이 점진적으로 메꿈.

→ 이 layering pattern 통찰 = SODA-specific trick이 아니라 **modern grounded synthesis의 일반 architecture.**

---

## Q12 — SODA 추가 detail + 너의 telemarketer-customer 관찰 ★★★

**Question**: SODA 더 깊이. 그리고 직접 해본 telemarketer-customer 시나리오에서 customer-LLM이 거의 항상 거절함.

**Note**: 이 질문은 **ch-19에서 parking한 customer-LLM realism callback이 organically hit한 순간** (memory: `project_llm_training_customer_llm_realism_callback`).

**Kernel**:

### Part A — SODA 추가 detail

**Atomic 10X provenance** (recursive grounded synthesis):
```
ConceptNet → ATOMIC (2019, 24K) → ATOMIC-2020 (1.33M) → Atomic 10X (10M, GPT-3 + critic)
```
Atomic 10X 자체가 grounded synthesis 결과물. SODA는 그것을 *다시* anchor로 써서 dialogue 생성. **한 단계 위 synthesis가 다음 단계의 ground 역할.** Yejin Choi group 전체 lineage의 design pattern.

**Filter (UltraChat 대비 한 단계 위)**:
| Filter | SODA | UltraChat |
|---|---|---|
| Scene coherence | ✅ | ❌ |
| Persona consistency | ✅ | ❌ |
| Toxicity classifier | ✅ | ❌ |

**COSMO model**: T5-3B + SODA-finetune. 3B로 9B BlenderBot-3B를 인간 평가 모든 axis에서 능가. DailyDialog/EmpatheticDialogues OOD transfer. → grounding이 capability를 transfer한다는 직접 증거.

**Mix 위치**: Smol-Talk ~20% / Tülu small slice. Register diversity (short chitchat) 보강 — 다른 source는 다 long-form Q&A/task.

**Limitations**:
- Atomic bias (English/Western/middle-class)
- Two-speaker only (multi-party 없음)
- Formulaic leak ("PersonX said X because...")
- Atomic 10X 자체가 ~5-10% noise (synthesis 결과물)

---

### Part B — Customer-LLM이 항상 거절하는 이유 (4-layer)

**Layer 1 — RLHF helpful-assistant prior**: "unsolicited contact"/"sales pressure"/"persuasion" → safety + resistance + critical-thinking prior activate. Customer role play해도 이 priors가 conditional distribution dominate.

**Layer 2 — "Good user" default**: LLM의 user role default = rational/decisive/firm. Telemarketing 맥락에서 = polite refuse. *통계적*으론 옳음 (real 응답률 ~1-3%) but *생성 다양성*으로는 무용.

**Layer 3 — Distribution collapse**: real customer는 multi-modal:
- Polite refuse 60% / Hostile 20% / Confused 10% / Curious 5% / Interested 3% / Lonely-chatty 2%

→ LLM은 single mode (polite refuse)로 collapse.

**Layer 4 — Information anchor 부재 ([[Q9]] direct)**: customer에게 외부 ground 0. Role label "customer"는 *style anchor only*. Teacher가 default emit.

---

### Part C — SODA framework로 분석

| SODA component | Telemarketer 시나리오 |
|---|---|
| Atomic triple (info anchor) | ❌ 없음 |
| Narrative bridge | ❌ 없음 |
| Dialogue gen | ✅ (그래서 default mode collapse) |

**진단**: SODA가 emotional/social domain에서 통한 이유 = Atomic 10X가 그 domain anchor를 미리 가짐. **Telemarketer = consumer behavior domain → anchor source mismatch.** Atomic은 "외롭다"는 cover하지만 "보험에 관심 있다"는 cover 안 함. → **너가 본 현상은 SODA 한계가 아니라 anchor source의 domain mismatch.**

---

### Part D — Fix: multi-axis information anchor

Customer-side에 외부 anchor를 박아야 함. 후보:

**Anchor 1 — Pre-sampled intent** (가장 cheap, 가장 효과적):
```
intents = [actively shopping, open to info, annoyed,
           lonely-chatty, confused, hostile, naive,
           sophisticated, busy, interested-but-suspicious]
```

**Anchor 2 — Demographic + product receptivity** (Persona-Hub bio sample)

**Anchor 3 — Emotional state at call time** (SODA Atomic-style)

**Anchor 4 — Real call log** (gold standard, privacy issue)

**합성한 prompt template**:
```
CUSTOMER PROFILE:
- Demographic: <demo>
- Past behavior: <past>
CUSTOMER STATE:
- Mood: <mood>
- Intent: <intent>  ← 이게 single-mode collapse 깨는 핵심
- Product knowledge: <knowledge>
```

→ Intent 한 필드 추가만으로 distribution이 multi-modal로 회복.

---

### Part E — Q9 framework의 production validation

너의 관찰은 [[Q9]] generalization의 *direct empirical confirmation*:

- Style anchor만 (role="customer") → single mode collapse
- Information anchor 추가 (intent="lonely") → multi-modal recovery

→ **Q9 framework가 sales-AI/chatbot eval/customer service training 같은 transactional/consumer domain에서 production-grade 검증됨.**

**추가 extraction**: anchor의 *domain 적합성*이 변수임. Atomic은 social, customer behavior는 다른 ontology 필요. **Anchor source 자체가 use case-specific.**

★★★ Triple combo:
1. SODA 추가 detail (Atomic provenance, COSMO, filter, mix)
2. Forward-tracked callback hit (ch-19 customer-LLM realism)
3. [[Q9]] framework의 production-grade validation + domain-specificity extraction

---

## Q13 — Scenario-first design (agent-drives vs scenario-drives)

**Question** (learner extraction): SODA filter는 multi-turn 생성 *전에* 적용됨. 그래서 TMR-customer는 scenario에 based해야지, customer-agent가 conversation을 drive하면 안 됨 — 같은 collapse 막기 위해.

**Kernel**:

**SODA filter timing 보정**: SODA pipeline 내부 explicit filter는 post-gen이지만, *Atomic 10X 자체가 upstream에서 critic model로 pre-filtered* (West et al. 2022). → Functionally는 "garbage anchor 사전 차단"이 맞음. Anchor source의 quality가 SODA pipeline의 ceiling.

**Generalization 확정 — Scenario-first design**:

| Design | Driver | 결과 |
|---|---|---|
| Agent-drives | customer-LLM (teacher prior) | Single-mode collapse |
| **Scenario-first** | **scenario (외부 anchor)** | **Multi-modal** |

→ Customer-agent를 *driver*에서 *renderer*로 demote. Agent가 scenario를 "연기"할 뿐 conversation을 "drive"하지 않음.

**Two-stage design 일반 원리**:
```
Stage A — Generate & filter anchors (quality/diversity 강제)
Stage B — Generate from anchors (teacher fluently render)
```
Single-stage = collapse 보장. Two-stage = collapse 방지.

**Production pipeline 이미 이걸 함**:
| Pipeline | Stage A | Stage B |
|---|---|---|
| SODA | Atomic 10X balanced sampling | narrative + dialogue |
| **AgentInstruct** | **task suggester** (implausible 사전 reject) | dialogue gen |
| **APIGen-MT** | function spec + tool list 사전 validate | tool-call trajectory |
| ToolACE | task generator + validator | multi-agent dialog |

**TMR-customer 적용**:
```
Stage A: scenario axis 정의 → Cartesian × filter implausible → curated library
Stage B: setup narrative + telemarketer-customer loop (scenario-conditioned)
Stage C: post-gen scenario-conversation consistency check
```

→ SODA architecture를 domain만 swap. Design unchanged.

★★ Generalization arc 진행: Q9 → Q11 → Q12 → Q13 — 각 step이 한 차원 깊어진 framework extension. Q13에서 modern synthetic pipeline의 *operational level*에 도달.

---

## Q14 — Cross-compatibility filter (persona × scenario)

**Question** (learner): SODA filter를 직접 적용 가능. Scenario와 user-system prompt 둘 다 generate하고, 각 user-system에 대해 *emotionally possible*한 scenario만 pick.

**Kernel**:

**정확히 AgentInstruct의 "task suggester" 패턴의 persona×scenario 변형.** 너가 independently 재도출.

**Filter timing 비교**:
| | SODA original | Learner design |
|---|---|---|
| Timing | Post-dialogue (narrative ↔ dialogue) | **Pre-dialogue (persona ↔ scenario)** |
| Target | dialogue가 narrative와 일관? | **persona가 scenario에 있을 법한가?** |
| Cost | 비싼 gen 후 reject | **싼 judge로 사전 reject** |

**왜 중요한가 — 3가지**:

1. **Cartesian product의 implicit incompatibility 차단**: "suspicious retiree" × "buy-now scenario" = absurd. Filter가 compositional consistency 강제.

2. **Filter 자체가 information-bearing**: compatibility judgment = world knowledge 적용. Filter도 [[Q9]]의 information-anchor component.

3. **Cost saving**: dialogue gen은 expensive (multi-turn × tokens). Judge는 cheap (single call). Implausible combo를 비싼 단계 전에 reject.

**구현 옵션**:
| Option | Mechanism | Tradeoff |
|---|---|---|
| A: LLM-as-judge | "plausible? YES/NO + reason" | Cheap, scalable, but judge LLM bias |
| B: Rule-based table | 사전 enumerate incompatibility | Cheap, deterministic, but coverage limit |
| C: Hybrid (production) | Rule → LLM-judge → embedding cascade | 최강, 가장 비쌈 |

**Judge bias mitigation**: judge에게도 *information anchor* 줘야 — 통계/rule을 함께 inject ("85% retirees hang up in 30s"). Judge가 LLM intuition 아닌 evidence-anchored로.

**Bidirectional consideration**:
- 일부 persona는 narrow compatible scenario set (정상 — real distribution도 그러함)
- **Coverage check 추가 필요**: 각 scenario가 ≥K persona와 compatible해야 함. K=2 미만이면 scenario reject (generalize 안 됨)

**Industrial precedent**:
| Pipeline | Filter axis |
|---|---|
| AgentInstruct | (role_A, role_B, task) triple |
| APIGen-MT | (function, tool, scenario) triple |
| ToolACE | (task, agent, environment) triple |
| **Learner design** | **(persona, scenario) pair** |

→ 모두 같은 architecture: **anchor curation stage에 explicit cross-filter**.

**Generalization arc 완성** (Q9 → Q14):
```
Q9:  information anchor vs style anchor
Q11: anchor → softening → generation
Q12: anchor가 domain-specific
Q13: anchor curation 별도 stage (scenario-first)
Q14: anchor stage에 cross-compatibility filter
```

→ Q14에서 *operational engineering complete*. 코드로 옮기면 production-grade pipeline. 남은 stage = judge LLM의 bias 제거 (ch-26/ch-44에서 더 깊이).

★★ Operational engineering level framework extension.

---

## Q15 — OASST: SFT 한계 vs DPO 강점 (per-objective anchor density) ★★

**Question** (learner): OASST는 short conv anchor엔 OK인데 10+ turn은 disaster. Tree-based는 too limited + bad signal 학습 위험. 근데 그 bad signal은 DPO엔 쓸 만함.

**Note**: ch-19에서 parking한 두 번째 forward-tracked callback (long-conversation anchor) hit. 두 callback (customer-realism + long-conv) 모두 organically surface됨.

**Kernel**:

**Teacher reframing 인정**: 처음에 OASST를 SFT anchor로 frame한 게 좁았음. OASST는 *원래 preference dataset*으로 설계됨:

| Component | Original purpose |
|---|---|
| Tree branching (sibling replies) | **Preference comparison pair** |
| Per-node quality rating | **RM training signal** |
| Per-node helpful/harmless tags | **DPO/Constitutional label** |
| Root-to-leaf path | (secondary) SFT 가능 |

→ Anthropic HH-RLHF처럼 *preference-labeling-first*. SFT는 side product.

**SFT 관점 (sparse, limited)**:
- 10+ turn path <100개 → statistical noise dominate
- Tree depth 얕음, branching factor 작음
- Contributor pool narrow (~13.5K)
- → SFT anchor로는 short-conv (1-7 turn)에서만 reliable

**DPO 관점 (rich, valuable)**:
```
parent_context
├── reply_A   [rank 1]  ← chosen
├── reply_B   [rank 2]
└── reply_C   [rank 3]  ← rejected
```
- 모든 분기에서 즉시 `(prompt, chosen, rejected)` triplet 추출
- ~80K assistant nodes → ~50K+ preference pair
- 길이 무관 — 모든 depth에서 sibling pair 발생
- → "Bad" signal이 *feature*, not bug

**일반 원리 — Anchor support density는 *training-objective-specific***:
| Objective | 필요한 density |
|---|---|
| SFT | complete trajectory (path density) |
| DPO | paired branching at same context (sibling density) |
| RM | ranking judgment (rating density) |
| Safety/Constitutional | rule-violation examples (negative density) |

→ **같은 source가 objective별로 완전히 다른 utility.** "Anchor density"는 source-level 아닌 *(source × objective)*-level 평가.

**Tree 구조 — same structure, opposite valuations**:
- SFT lens: branching factor 작음 + path narrow → 한계
- DPO lens: sibling pair 풍부 + real human disagreement signal → 강점

**ch-32 connection (preference track)**:
- OASST tree-structured preference data = original-source quality 최고 (real human ranking)
- UltraFeedback 등 LLM-as-judge preference는 [[Q9]] register prior 문제 다시 발생
- → OASST의 *bad signal*은 noise 아닌 *human-judged hierarchy substrate*

**Generalization arc 업데이트** (Q9 → Q15):
```
Q9:  information vs style anchor
Q11: anchor → softening → generation
Q12: anchor domain-specific
Q13: scenario-first (two-stage)
Q14: cross-compatibility filter at anchor stage
Q15: anchor density is (source × objective)-specific
     — same OASST: SFT 한계, DPO 강점
```

★★ Framework extension: anchor evaluation 자체에 *objective axis*를 추가. Single source can simultaneously fail for SFT and excel for DPO. Forward-tracked callback (long-conv anchor) hit + teacher reframing acceptance 포함.

---

## Q16 — System-prompt diversity 확보 방법

**Question**: System-prompt diversity는 어떻게 secure?

**Kernel**:

**Chapter §7 framing**: 2024+ frontier가 topic/role → system prompt로 이동. Persona-Hub (1B personas) + SystemChat-2.0 (7K curated)이 canonical source. Empirical: 2× unique vocab, 3× persona-consistent follow-up, +5-10 IFEval vs topic-only.

**6 source method**:
| Method | Mechanism | Scale | Cost |
|---|---|---|---|
| 1: Web-mining (Persona-Hub) | Bio/about-me/LinkedIn 크롤 | ~1B | infra-heavy |
| 2: Hand-curated + LLM expand (SystemChat) | 5-10 axis hand-enumerate | 7K-10K | $50, 1주 |
| 3: Procedural composition | Cartesian sample 1-3 axes | 10^4-10^6 | zero |
| 4: Meta-synthesis | LLM generate prompts | unbounded | cheap |
| 5: Real production mining | Custom GPT/Claude Project | bounded | privacy issue |
| 6: Adversarial expansion | Seed → LLM variation → filter | unbounded | iterative |

**Practical compositional architecture** (production):
```
Axis 1: ROLE (~500) — profession/functional/personality
Axis 2: STYLE (~50) — tone/format/length  
Axis 3: CONSTRAINT (~100) — output format/behavior/forbidden
Axis 4: MODE (~20) — safety/specialty
Axis 5: META (~5) — self-identification/mission

Composition rule: 1-3 axes per prompt (cap stacking)
→ 10^4 ~ 10^6 distinct prompts
```

**4-stage quality filter**:
1. **Distinctness**: embedding cos < 0.85
2. **Coherence**: no internal contradiction ("terse" + "verbose" = reject)
3. **Plausibility**: LLM-as-judge — real product team이 쓸 법한가?
4. **Stack depth cap**: max 2-3 axes (chapter §7 risk: "compositional overloading confuses teacher+student")

**Risks (chapter §7)**:
- Web-mining bias (English/online-professional skew)
- Compositional overloading (4+ axes stack)
- Over-specialization (system prompt 너무 literal 학습 → 적절한 deviation 못함)

**Q9-Q15 framework로 진단**:
| | Anchor 분류 |
|---|---|
| System prompt | **Style anchor** (Q9) — register/tone만 |
| Information content | ❌ teacher prior 그대로 |
| Customer intent | ❌ single-mode collapse 안 풀림 |

→ **System-prompt diversity 단독으론 [[Q14]] customer-refuse 문제 못 풀음.** Persona-Hub 1B로 customer-LLM 만들어도 여전히 거절. **Persona는 style만 바꾸고, intent가 information anchor로 박혀야 함.**

**Modern complete pipeline (Q14 + system-prompt)**:
```
Stage A1: System prompts (compositional, style axis)
Stage A2: Scenarios/intents (information axis)
Stage A3: CROSS-FILTER (sys_prompt, scenario) compatibility
Stage A4: Validated pair library
Stage B:  Dialogue gen
Stage C:  Post-gen consistency
```

→ **System prompt = style axis. Scenario = information axis. Cross-filter가 둘을 결합.** 한 axis만으론 ceiling.

**Cost-effective 추천**:
- Small: Method 2 (curated) + Method 6 (adversarial expand). $50, 1주
- Medium: Method 3 (procedural) + Method 4 (meta-synth). $500, 2주
- Production: Method 1 (web-mining) + Method 3 (composition). infra-dependent

**모든 scale 공통**: stack cap 2-3 / distinctness filter / customer modeling 시 Q14 cross-filter로 information anchor 결합.

---

## Q17 — Pre-diversify vs post-hoc swap (Strategy A vs B)

**Question**: System prompt를 미리 다양화하고 generate? 또는 잘 작동하는 single prompt 기반으로 만들고 system prompt만 바꾸는가?

**Kernel**:

**Answer: Strategy A (pre-diversify → per-prompt generation).** Strategy B는 student가 system prompt를 *무시*하도록 학습시키는 anti-pattern.

**Strategy 정의**:
```
A: for sys_prompt in pool:
       conv = generate(sys_prompt)
   → distinct conversation per prompt, native conditioning

B: base_conv = generate(well_tuned_prompt)
   for variant_prompt in pool:
       save(variant_prompt, base_conv)  ← 같은 content, 다른 label
```

**Strategy B가 망하는 이유 — content-label mismatch**:
- "Respond in haiku" / "You are accountant" / "Be terse" 모두 같은 content
- Student 학습 신호: *"system prompt가 뭐든 같은 response 해도 OK"*
- Failure mode: "Respond in haiku" 받아도 일반 산문, "You are X" 받아도 X 행동 안 함
- **System prompt를 cosmetic label로 학습 — IFEval 점수 정반대 효과**

**Strategy A 메커니즘**: teacher가 generation 시점에 prompt 따라 response 만들어야만 valid signal. Student가 `P(behavior | sys_prompt)` 조건부 분포 학습 가능.

**Production pipelines = 전부 Strategy A**:
| Pipeline | |
|---|---|
| SystemChat-2.0 | 7K prompt × per-prompt gen |
| Persona-Hub-Instruct | per-persona gen |
| Tülu-3, Qwen-2.5 | diverse persona-conditioned gen |
| Anthropic internal | per-persona pipeline |

→ Strategy B 쓰는 release 없음.

**너의 우려 (quality)는 valid — 해결책 = Q16의 4-stage filter cascade**:
1. Distinctness (embedding < 0.85)
2. Coherence (no contradiction)
3. **Plausibility** (LLM-as-judge: real product team이 쓸 prompt?) ← 이게 quality 보장
4. Stack cap 2-3 axes

→ Filter 통과 prompt는 individually well-tuned. 그 아래 generation도 quality 유지.

**Practical compromise — Stratified mix** (production standard):
```
Core SFT slice (~20K): 1-3 검증된 prompt × 대량 gen → capability backbone
Diversity slice (~30K): Q16 filter 통과 diverse prompts × per-prompt 5-20 gen → IFEval +5-10
Realism anchor (~10K): OASST/WildChat 5-15% → user-realism
```

→ Strategy A 기반 + quality/diversity tradeoff를 *분리해서 mix*. Tülu-3/Smol-Talk가 이 구조.

**Inference-time 효과**:
- Strategy A 학습 → 새 system prompt에 generalize (controllable)
- Single prompt 학습 → 학습 default에 lock-in (uncontrollable)

→ Chapter §7의 *"reliably controllable via system prompt, reduces inference-time prompt engineering"* = Strategy A 효과.

**Decision table**:
| Q | A |
|---|---|
| Pre-diversify? | **Yes (Strategy A)** |
| Quality 우려? | Q16 filter로 individual quality 보장 |
| Volume? | Stratified mix |
| Strategy B 쓸 곳? | **없음** (anti-pattern) |
| Cost-effective recipe | hand-write 100 → LLM expand 1K → filter → per-prompt gen |

**Real tradeoff 명시**: single quality vs many controllability는 single-strategy로 못 풂. **Stratified mix가 답.**

---

## Q18 — Multi-turn mix에서 가장 중요한 것? (length vs system-prompt)

**Question**: Multi-turn conversation mixture에서 length diversity vs system-prompt diversity 중 뭐가 더 중요?

**Kernel**:

**Direct answer**: System-prompt > length (Tier 2 vs Tier 3). 그러나 둘 다 위에 *Tier 1 master variables*가 있음.

**Priority hierarchy** (impact 순):

| Tier | 항목 | Failure if missing |
|---|---|---|
| 1 | **Real-human anchor (5-15% OASST/WildChat)** | User-realism 완전 붕괴 (synthetic 한계의 유일 escape) |
| 1 | Single-turn capability volume | Multi-turn substrate 없음 |
| 2 | Multi-axis conditioning (3+) | Single-axis ceiling |
| 2 | **System-prompt diversity** | IFEval -5~10, controllability 약함 |
| 3 | Length distribution coverage | >15 turn 약함 (modern은 드물어 soft failure) |

**System-prompt vs Length 직접 비교**:
| | System-prompt | Length |
|---|---|---|
| IFEval | +5-10 | marginal |
| Controllability | HIGH | low |
| Failure mode | HARD (customer-facing 깨짐) | soft |
| Cost | medium | low (mix만 잘하면 free) |

→ System-prompt 우선. Length는 여러 source mix (SODA short + UltraChat med + CAMEL long)면 자동 cover.

**진짜 가장 중요한 것 — Chapter §8 modern era shift**:
> "Explicit multi-turn = ~30K of 1M (3%). Multi-turn capability는 single-turn data의 in-context follow-through에서 학습. Explicit dialogue slice = realism insurance, not primary supervision."

→ **Multi-turn 자체보다 substrate (single-turn massive) + realism anchor가 결정적.**

**Q9-Q15 framework로 ranking** (master variable 확인):
| 항목 | Anchor type |
|---|---|
| Real anchor (OASST/WildChat) | **Information** ★ |
| Grounded (SODA) | **Information** ★ |
| Scenario (Q14) | **Information** ★ |
| Topic (UltraChat) | weak info |
| Role (CAMEL) | Style |
| System prompt (Q16) | **Style** |
| Length | Structural axis |

→ **Information anchor 있는 component가 다 Tier 1-2.** Style anchor만 stack하면 ceiling. System-prompt도 style — IFEval 효과는 있지만 ceiling 안에서.

**Decision tree**:
1. Real anchor 5-15% 확보 → 없으면 다 무의미
2. Multi-axis (3+) 구성 → 그 안에 system-prompt 포함
3. Length는 source mix로 자연스럽게 cover

**핵심 takeaway**:
- Length vs system-prompt 비교: **system-prompt 우선**
- 그러나 **real-human anchor + information anchor density가 master variable** — 둘 다 그 아래
- Length diversity는 단독 design 항목 아니어도 됨 (mix만 잘 하면 free)

---
