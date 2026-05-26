<!-- chapter: ch-26 — Modality: Tool and Function-Calling Data
     deps: [[read]]
     scope: clarifying questions captured during Read phase. Kernel-only — full explanations stay in read.md / discuss transcript.
-->

# Ch-26 — Q&A

Back to [[read]]. Cross-chapter links: [[../ch-22/qa]], [[../ch-24/qa]], [[../ch-25/qa]] (especially Q9, Q14).

---

## Q1 — APIGen 자세히

**Question**: APIGen 더 자세히.

**Kernel**:

**Salesforce AI Research, NeurIPS 2024**. Single-turn FC의 **verifier-first thesis** 확립. → ch-25 [[../ch-25/qa]] Q14 cross-compatibility filter의 industrial-scale instantiation.

**Pipeline (5 step)**:
```
1. API curation: ToolBench 16K → 3,673 executable
2. Seed sampling: k=1-3 functions/gen, diversity sampler
3. Query + solution gen: DeepSeek-Coder-V2-Instruct or GPT-4
4. 3-LAYER VERIFIER cascade
5. Dedup: MinHash on (query, call)
```

**3,673 API floor — binding constraint**: APIGen은 executable reference impl 없이는 verifier 못 박음 철학. ToolBench 16K → 3,673만 keep (Salesforce가 mock or endpoint 통제 가능한 것). 후속 [[toolace]]가 *반대 선택* (26K API + LLM-simulated execution).

**3-layer verifier**:
| Layer | Mechanism | Cost | Catches |
|---|---|---|---|
| 1: Format | JSON parse + schema validate | free | malformed / wrong arg / wrong type |
| 2: Execution | Python sandbox, 5s timeout | ~$0.001/call | non-existent API / impossible args / runtime error |
| 3: Semantic | GPT-4 judge on (query, call, result) | ~$0.01/call | wrong unit / wrong target / right func wrong intent |

**Ablation (각 layer가 자기 keep을 earn)**:
| Config | BFCL-V1 | Δ |
|---|---|---|
| Full 3-layer | **88.24** | — |
| Remove semantic | 82.2 | −6.0 |
| Remove execution | 77.3 | **−10.9** |
| Remove format | 70.1 | −18.1 |

**3가지 통찰**:
1. **Format-only ≈ 70 = "Glaive 2023 ceiling"** (정확히 측정). Format만으론 70 이상 불가능.
2. **Execution이 가장 큰 single-layer lift (-11)** — 이게 3,673 API ceiling 이유. Execution 유지 vs API 확장 trade-off.
3. **Semantic catches residual 6%** (wrong unit / wrong intent). LLM-judge or human only.

**Cascade ordering — cheap-first (ch-22 §4 cost-aware principle)**:
```
100K raw → Format (75%) → 75K → Execution (90%) → 68K → Semantic (88%) → 60K
Total ~40% rejection — feature, not bug
Cost saving 32% vs naive ordering
```

**Output**: 60K samples mapping *exactly* to BFCL-V1 categories:
| APIGen type | % |
|---|---|
| Simple | 40 |
| Multiple | 25 |
| Parallel | 20 |
| Parallel-Multiple | 15 |

→ **"Whoever sets eval taxonomy sets data-generation taxonomy."**

**Headline — Hallucination 5× reduction**:
- ToolLLaMA: ~15%
- **xLAM-7B (APIGen-trained): <3%**
- 2024 second half practitioner migration의 trigger

**Downstream**: xLAM-7B (Mistral base) → BFCL-V1 88.24%, Sept 2024 release 시점 **#1 among <13B**.

**xLAM-2 staged recipe (2025 cleanest open multi-turn FC recipe)**:
1. APIGen 60K SFT
2. APIGen-MT 5K SFT
3. Optional DPO (β=0.1) on (correct, hallucinated-name) pair

**Limits**:
- Executable-API ceiling → ToolACE 응답
- **LLM-judge blind spot**: open problem (chapter 어떤 paper도 못 풂)
- No multi-turn → APIGen-MT 응답
- CC-BY-NC-4.0 license

**Q9 / Q14 framework로 분해**:

| Q14 (learner) | APIGen 구현 |
|---|---|
| Stage A1: scenarios | API curation (3,673 executable) |
| Stage A2: style anchors | Query gen by teacher |
| **Stage A3: cross-filter** | **3-layer verifier (format/exec/semantic)** |
| Stage A4: validated pairs | 60K accepted |
| Stage B: dialog gen | single-turn call gen + execution |
| Stage C: post-gen filter | MinHash dedup |

→ **너가 ch-25 Q14에서 conceptually 도출한 architecture를 APIGen이 exact하게 구현**. Production-grade validation.

**Q9 information anchor ranking**:
| Anchor type | Q9 strength | APIGen 사용 |
|---|---|---|
| Real environment execution | ★★★ | ✅ Layer 2 |
| Schema-defined structure | ★★ | ✅ Layer 1 |
| LLM-as-judge | ★ | ✅ Layer 3 |
| Persona/role label | none (style) | ✗ |

→ APIGen은 **strongest information anchors만 cascade로 stack**. Style anchor에 의존 안 함.

**CAMEL vs APIGen — Q9 lens로 1년 evolution**:
| | CAMEL (2023) | APIGen (2024) |
|---|---|---|
| Termination | `<CAMEL_TASK_DONE>` (user) | **Execution success** (environment) |
| Anchor | Style only | **Information cascade** |
| Verifier | None | **3-layer** |
| Hallucination | High (role drift) | **<3%** |

→ Information anchor strength axis에서 한 단계 진화. ch-25 CAMEL이 *style anchor maximum*이었다면, APIGen은 *information anchor stacking*.

---

## Q2 — APIGen-MT Blueprint generation

**Question**: Multi-turn에서 blueprint는 어떻게 generate?

**Kernel**:

**Blueprint = dialog 아닌, structured task config (POMDP ground-truth plan).** Multi-turn correctness를 dialog 생성 전 lock. → [[../ch-25/qa]] Q13/Q14 framework의 multi-turn FC instantiation.

**Phase 1 — 5 step**:

### Step 1: POMDP + API graph 모델링
```
domain := (states, actions, observations, transitions, policy)
API graph := directed graph (nodes=API, edges=dependency)
```
Domain 자체가 structured information anchor (Q9 ★★★).

### Step 2: 5 samplers (taxonomy-first diversity)
| Sampler | 대상 |
|---|---|
| API | graph subset |
| Policy | domain constraint |
| Domain data | concrete entities (DB rows, user state) |
| Persona | user persona |
| Example | few-shot validated blueprints |

→ Diversity는 sampler에서 (curation), not LLM creativity.

### Step 3: LLM generates structured config (no dialog)
```jsonc
{
  "domain": "airline",
  "user_persona": "budget traveller, prefers refundable fares",
  "instruction": "Book JFK→LAX under $250 and a hotel at LAX under $150 for Friday.",
  "ground_truth_actions": [
    {"api": "search_flights", "args": {"from":"JFK","to":"LAX","max_price":250}},
    {"api": "book_flight",    "args": {"flight_id": "$F.id"}},
    ...
  ],
  "expected_final_state": {"flight_booked": true, "hotel_booked": true}
}
```
Committee: GPT-4o + DeepSeek V3.

### Step 4: 3-stage validation cascade
| Validator | Mechanism |
|---|---|
| **Action validation** | APIGen 3-layer 적용 (format + execution + policy) per call |
| **Alignment validation** | LLM committee: "이 action sequence가 user intent satisfy?" — multi-turn analogue of semantic check |
| **Semantic review** | Committee aggregation + refinement (edge case 해결) |

→ Cascade 다 통과한 blueprint만 *validated primitive*로 pool 진입.

### Step 5: Reverse task recombination (scaling)
```
Validated primitive pool: search_flight, book_flight, search_hotel, book_hotel
   ↓ compose
Complex blueprint: "flight + hotel for same trip"
   → 재검증 거의 불필요 (primitives 이미 validated)
```
→ Linear committee cost 안 폭발시키고 combinatorial coverage. **5K trajectory로 충분한 이유**.

**Phase 2 (rollout, 간단)**:
- Two LLM actors: user-sim (persona+instruction conditioned), agent (실제 API call)
- Rejection sampling: trajectory가 expected_final_state 도달 + ground_truth_actions match
- ~70% success — correctness 이미 lock이므로 reject는 *realism* issue (not correctness)

**3 architectural insight**:
1. **Verification cost를 mechanical layer로** (subjective→mechanical 변환)
2. **Compound-error 해결**: 12-turn × 95% = 54%였던 게, plan validated이라 ~70% (realism axis만 reject)
3. **Consistency by design** (BFCL pass^k 대응) — structurally consistent trajectories

xLAM-2-70B τ-bench: pass^1 = 56.2%, pass^4 = 39.4% (gap 작음).

**Q13/Q14 framework 매핑 (learner design ↔ APIGen-MT)**:
| Q13/Q14 | APIGen-MT |
|---|---|
| Stage A: scenario curation | **Phase 1 blueprint gen** |
| Q14 cross-filter | **3-stage validation cascade** |
| Validated pair library | Validated primitive pool |
| Stage B: gen conditioned | **Phase 2 rollout (realization)** |
| Stage C: post-gen filter | Rejection sampling |

→ 너의 Q13 *"agent를 driver에서 renderer로 demote"* = APIGen-MT Phase 2의 정확한 design. **Production paper가 너의 design principle을 그대로 implement.**

**Q9 information anchor cascade**:
| Anchor | 위치 | Strength |
|---|---|---|
| API graph + dependency | Phase 1 substrate | ★★★ |
| Executable reference impl | Validator 1 | ★★★ |
| Policy constraints | Validator 1 | ★★ |
| LLM committee | Validator 2,3 | ★ |
| Persona prompt | Phase 2 user-sim | none (style, rendering only) |

→ 3가지 info anchor stack + persona는 rendering layer only.

**Limits (paper explicit)**:
- Schema dependence (API graph quality에 종속)
- Simulation gap: LLM-sim user는 real human quirk 못 만듦 — [[../ch-25/qa]] Q12 customer-LLM realism 문제의 multi-turn FC 재현
- Validation overhead (verifiability의 price)

---

## Q3 — User-LLM이 ground_truth_actions를 보고 agent를 lead? ★★

**Question**: Phase 2에서 user model이 ground_truth_actions를 보고 그 tool을 쓰도록 lead하는 거?

**Kernel**:

**Yes, partial visibility (Setup C). 그리고 이게 APIGen-MT의 *open weakness***. Paper가 "simulation gap"으로 우회적으로만 인정.

**3 possible setups**:
| Setup | User-LLM 가시성 | Success rate | 문제 |
|---|---|---|---|
| A: Full plan | exact actions + args | ~100% | Student가 reasoning 학습 안 함 |
| B: Goal only | instruction + persona | <50% | Rejection cost 너무 큼 |
| **C: Partial hints** | goal + action shape | **~70%** | **Easiness bias** |

APIGen-MT의 70% success rate가 Setup C의 evidence.

**Training vs Inference distribution shift**:
- Training: user가 모든 constraint를 first turn에 articulate (lucid, helpful)
- Inference (real user): iteratively reveal, ambiguous, with clarification need

→ Agent가 *clarification question / ambiguity handling / intent inference* 학습 못함.

**Q12 framework로 분해 — 동일 문제의 다른 modality**:
| | Q12 (telemarketer) | APIGen-MT |
|---|---|---|
| Symptom | customer always refuses | user always lucidly articulates |
| Cause | no information anchor for intent | user has *too much* info (plan visibility) |
| Mitigation | inject intent diversity | **inject information asymmetry** |

→ Q12 = under-helpful user (single mode), APIGen-MT = over-helpful user (script-following). **Opposite failure modes of same user-LLM realism gap.**

**Q9 extension — "Information asymmetry principle"** (★ new generalization):

| User-LLM input | Q9 분류 | 효과 |
|---|---|---|
| Persona | style anchor | good |
| Goal/instruction | info anchor | good (intent ground) |
| **Plan visibility** | **info anchor** | **BAD — scripting** |

→ Information anchor가 *항상 좋은 게 아님*. **Anchor를 *어디에* 박는가**가 중요.

> **Information asymmetry by design**: User-LLM은 *어떻게 사람이 행동할지*만 anchor. *Agent가 학습해야 할 reasoning 단서*는 user에 박으면 student가 그 reasoning 학습 못함.

= Q9 framework의 *spatial axis* 추가 (anchor를 *얼마나* + *무엇을* + **어디에**).

**Ideal fix — information asymmetry**:
- User-LLM: ✅ persona + goal, ❌ action sequence + tool list + arg format
- Agent: ✅ tool list + schema, ❌ user's full plan
- → User는 goal만, agent가 plan 발견. Rejection rate 증가하지만 student가 real user에 robust.

**Benchmark blind spot**: τ-bench/BFCL multi-turn 모두 LLM-simulated user로 평가. Training과 eval이 *동일하게 easy*. Real-world gap이 benchmark에 안 잡힘. → §8 가이드라인 *"deploy in real production for ground truth"*의 이유.

★★ Framework extension: Q9에 *spatial axis (anchor를 어디에 박는가)* 추가. APIGen-MT의 simulation gap을 paper보다 깊은 layer로 진단.

---

## Q4 — 이게 overfitting 아닌가? ★

**Question**: APIGen-MT의 easiness bias는 overfitting 아닌가?

**Kernel**:

**Yes, 정확히 — *distribution overfitting* (sim2real overfitting). Classical overfitting과 메커니즘이 살짝 달라서 standard fix가 안 통함.**

**Classical vs Distribution overfitting**:
| | Classical | Distribution |
|---|---|---|
| Cause | example 암기 | distribution shape에 prior 박힘 |
| Symptom | same-dist test fail | OOD fail, in-dist OK |
| Fix | regularization / more data | **distribution diversification** (more *same-dist data 무의미*) |
| Detect | train>>test accuracy | benchmark high + production low |

→ APIGen-MT의 5K 모두 *동일 easy distribution* (lucid scripted user). Student가 그 *distribution shape*에 specifically optimize됨. 단순 데이터 양 증가로 안 풀림.

**Why standard regularization fails**:
| 방법 | 효과 |
|---|---|
| Dropout / weight decay | ❌ parametric overfit 아님 |
| Early stopping | ❌ converged model이 어차피 wrong dist |
| More synthetic data | ❌ 같은 pipeline = same dist |
| **Real anchor 추가** | ✅ data-side만 통함 |

→ Bias가 *parameter level* 아닌 *data distribution level*. Architecture fix로 안 됨.

**APIGen-MT의 double risk**: 5K small + uniform distribution → classical overfit + distribution overfit 동시 위험. xLAM-2 staged recipe가 classical 부분 완화 (60K SFT backbone) 하지만 **어느 stage에도 real user anchor 없음**.

**Real fix — chapter §9 #4 directive**:
> "Include a real-human anchor. 5-15% OASST or WildChat slice."

→ **APIGen-MT는 chapter 자체 가이드라인 위반.** 5K 전부 synthetic.

**가능한 real-anchor source**:
| Source | 가용성 |
|---|---|
| Production log mining | Privacy (Anthropic/OpenAI internal만) |
| ChatGPT Custom GPT log | License/IP issue |
| Human red-team trajectory | 비쌈, scale 제한 |

→ **Open-source pipeline의 구조적 한계.**

**Q15 framework로 진단 — anchor support density mismatch**:
```
Training (APIGen-MT 5K):
  ▮▮▮▮▮▮▮▮▮ ← lucid scripted (dense)
  ░         ← real ambiguous (zero)

Inference (real deployment):
  ░         ← lucid scripted (rare)
  ▮▮▮▮▮▮    ← real ambiguous (dense)
```
→ Training support와 inference support가 *disjoint*. Q15 (per-objective anchor density)의 *deployment-vs-training axis* extension.

**Detection signals** (benchmark가 못 잡음):
| Signal | 어떻게 보는가 |
|---|---|
| τ-bench vs real production gap | benchmark high, deployment low |
| BFCL pass^1 vs pass^k | single OK, repeated fail |
| User articulation perturbation | training-like OK, ambiguous fail |
| Cross-style transfer | Korean/short/typo fail |

→ 자기 distribution으로 평가하는 benchmark는 이 bias detect 못함. Real deployment만 진실.

**일반화 — Synthetic pipeline의 universal failure mode**:
| Pipeline | 자기 form의 distribution overfit |
|---|---|
| Baize | self-chat lucid → real typo fail |
| UltraChat | coherent → real friction fail |
| CAMEL | role-locked → real free-form fail |
| SODA | Atomic emotional → non-emotional fail |
| **APIGen-MT** | **scripted user → real ambiguous fail** |

→ **모든 synthetic pipeline이 distribution overfitting 위험.** §9 #4의 5-15% real anchor가 universal mitigation.

**도달한 generalization (★)**:
> Synthetic pipeline의 quality metric (BFCL, τ-bench)이 *자기 distribution으로 평가*하면 distribution overfitting detect 못함. Real anchor 없는 한 production deployment gap이 systemic.

= Q15 framework의 *deployment-vs-training axis* extension. APIGen-MT는 chapter §9 #4 위반.

---

## Q5 — ToolACE의 5 levels?

**Question**: ToolACE 5 levels가 뭐?

**Kernel**:

**Complexity Evaluator가 target하는 difficulty taxonomy. BFCL category distribution과 hand-tuned matching.**

**5 Levels**:
| # | Level | Share | 학습 capability |
|---|---|---|---|
| 1 | Simple single-call | 30% | baseline 한 query → 한 call |
| 2 | Multiple (choose from list) | 25% | function-selection (≥2 candidate 중 선택) |
| 3 | Parallel (multiple calls same turn) | 20% | independence detect (dep 없는 ≥2 call) |
| 4 | Nested / multi-turn | 15% | state + sequential (call A 결과 → call B) |
| 5 | **Info-incomplete** | 10% | **clarification asking** — underspecified query → 질문 |

**핵심 architectural contribution = Complexity Evaluator + forced resampling**:

Naive generation 시 → 70%가 simple로 collapse (teacher easy-mode attractor). 이게 Q9 framework가 예측하는 정확한 failure.

```
1. Sample target level from {30/25/20/15/10}
2. Generate dialog (3 role-players)
3. Evaluator classifies into level
4. If classified ≠ target: reject + resample
5. Repeat until distribution match
```

→ **Forced sampling으로 distribution shape explicit 제어.** Teacher prior가 distribution 망치는 문제의 mechanical 해결책.

**Q9/Q14 framework로 분해**:
| Component | Q9 분류 |
|---|---|
| Target distribution {30/25/20/15/10} | **Information anchor** (외부 spec) |
| Complexity evaluator | Q14 cross-filter (level match) |
| Forced resampling | distribution shape enforcement |

**Distribution이 BFCL과 정확히 매칭 — "benchmark-shaped data"**:
> "Whoever sets the eval taxonomy sets the data-generation taxonomy."

| APIGen | ToolACE | BFCL V1 | Granite |
|---|---|---|---|
| 4 types | 5 levels | 7 categories | 7 capabilities |

→ 같은 taxonomy 재발견. 모두 BFCL eval axes 정렬.

**★ Level 5가 [[Q3]] distribution overfitting을 부분 해결**:
- [[Q3]] APIGen-MT 문제: user가 over-helpful → clarification 학습 못함
- ToolACE Level 5: explicitly underspecified query 생성 → agent가 clarification 학습
- → APIGen-MT distribution overfit의 partial mitigation (10% slice)
- 한계: artificially info-incomplete (evaluator-judged) ≠ real ambiguous user. 부분만 cover.

**Ablation contribution**:
| Module | BFCL-V1 drop |
|---|---|
| Remove TSS | −4.3 |
| **Remove complexity controller** | **−3.1** |
| Remove model-judge | −5.2 |
| Remove rule-checks | −2.8 |

→ Evaluator alone 3.1점. 측정 가능한 contribution.

ToolACE-8B BFCL-V1 **91.41%** (xLAM-7B 88.24% 능가, matched scale).

**ToolACE vs APIGen trade-off** (chapter §8이 both 사용 권하는 이유):
| | APIGen | ToolACE |
|---|---|---|
| API count | 3,673 executable | 26,507 LLM-sim |
| Verifier | 3-layer | dual-layer |
| Distribution control | 4-type generation | **5-level evaluator + resample** |
| Hallucination | <3% | higher |

→ 다른 axis 최적화. Production recipe = APIGen + ToolACE + Hammer 다 mix.

**핵심 takeaway**:
- 5 levels는 *category list* 아닌 **distribution shape enforcement mechanism**
- Complexity evaluator가 teacher easy-mode collapse를 *information anchor*로 깸
- Level 5 (info-incomplete)는 [[Q3]] critique의 partial fix — real-anchor 대체는 아니지만 progress

---

## Q6 — ToolACE 5-level generation + validation operational mechanics

**Question**: 5-level을 *어떻게* generate하고 *어떻게* validate?

**Kernel**:

**Two-side architecture**: Generation (MAI) + Validation (Dual-Layer).

### GENERATION — Multi-Agent Interactive Dialog (MAI)

**3 LLM roles**:
| Role | System prompt |
|---|---|
| user-LLM | "Generate queries matching target complexity" |
| assistant-LLM | "Tools: [...]. Decide which call(s)" |
| tool-simulator-LLM | "Simulate API X response. Given args, generate schema-matching response" |

**Per-level generation (concrete)**:
| Level | 핵심 mechanism |
|---|---|
| L1 Simple | 1 API → user query → 1 call |
| L2 Multiple | ≥2 similar APIs → query maps to ONE → assistant picks correctly |
| L3 Parallel | 1 API → query requires multi-invocation ("Paris and Tokyo") → parallel calls |
| L4 Nested | Chained-dep APIs → multi-turn, call B uses call A result |
| L5 Info-incomplete | Underspecified query → assistant MUST ask clarification → user provides |

→ Level별로 prompt가 다름. "Generate level-X"라고 generic하지 않고, level-specific 행동을 explicit 강제.

**Forced Resampling Loop (핵심 mechanism)**:
```
target_level ← sample from {30/25/20/15/10}
while True:
  dialog ← MAI(target_level, apis)
  classified ← LLM_judge.classify(dialog)
  if classified == target_level: break
```

→ Hybrid: **a priori** (prompt conditioning) + **a posteriori** (classification) + **resampling**. Teacher easy-mode attractor를 active control로 깸.

### VALIDATION — Dual-Layer Verifier

**Layer 1 — Rule-based**:
| Check | |
|---|---|
| JSON schema parse | ✓ |
| Required params present | ✓ |
| Param types match | ✓ |
| Enum values in range | ✓ |
| Partial execution | only ~10% (Python mock 있는 subset) |

→ APIGen은 *all 3,673* execute, ToolACE는 *subset only*. Trade-off.

**Layer 2 — Model-based (GPT-4 3-way verdict, ALL must pass)**:
- (a) Query clarity (ambiguous reject — L5 별도 처리)
- (b) Call correctness (wrong unit/target/intent catch)
- (c) **Response consistency** (tool-simulator hallucination catch) ← ToolACE unique

→ APIGen에 없는 dimension (c) — execution 약화 보완. ~40% rejection.

### APIGen vs ToolACE verifier 비교

| Layer | APIGen | ToolACE |
|---|---|---|
| Format | ✓ | ✓ + enum |
| **Execution** | **All 3,673** | **~10% subset** |
| Semantic | yes/no judge | **3-way judge** |

### Full pipeline sequence

```
1. Target sampling: level + apis
2. Generation loop: MAI + evaluator + resample until level match
3. Validation cascade: rule_check + 3-way judge
4. ~40% overall rejection → 11,300 final dialogs
```

### ★ Recursion problem (paper-acknowledged limitation)

> "GPT-4 for generation AND judging — circular quality ceiling."

```
Generation: GPT-4 (3 roles)
Evaluator:  GPT-4 (level classifier)
Validation: GPT-4 (3-way judge)
```

→ **하나의 LLM이 generate + classify + judge.** 

3 consequences:
1. GPT-4가 "level X처럼 보이는" output → 같은 GPT-4가 "yes, level X" 판정 → third-party verification 없음
2. GPT-4 bias가 generation/eval 양쪽에 동일하게 박힘 → bias가 invisible
3. Ceiling이 GPT-4 능력에 직접 묶임

→ Q9 framework: evaluator가 *self-generated information* anchor라 진짜 anchor (real execution / real human judge)보다 약함. APIGen은 real execution으로 이 recursion 피함, ToolACE는 trade-off로 위험 떠안음.

### Q9/Q14 framework로 분해

| Component | Q9 분류 |
|---|---|
| 5-level distribution | Information anchor (BFCL-spec) |
| Per-role prompts | Style anchor |
| Complexity evaluator | Q14 cross-filter (category level) |
| Rule-based check | Info anchor ★★ |
| Partial execution | Info anchor ★★★ (small subset) |
| 3-way model judge | Weak info anchor ★ |

→ ToolACE = **info + style anchor cascade + category-level cross-filter**. Q14 framework의 distribution-level instantiation.

### [[Q4]] distribution overfitting lens

- Q4 문제: synthetic이 자기 distribution에 overfit
- ToolACE: distribution을 BFCL-shaped로 *control* (direction은 잡음)
- **하지만 여전히 self-generated. Real-user anchor 없음.**
- → distribution overfitting의 *direction control*은 하지만 *avoidance*는 못함

---

## Q7 — BFCL 방식 + "benchmark이 데이터 스펙 되었다" 의미

**Question**: BFCL은 어떻게 평가? "Benchmark becomes data spec"이 무슨 뜻?

**Kernel**:

### Part A — BFCL methodology

**Berkeley Function Calling Leaderboard** (Berkeley Sky Computing / Gorilla team). FC 평가의 de facto standard.

**4 versions (induced data evolution)**:
| V | When | 추가 | Induced data |
|---|---|---|---|
| V1 | Feb 2024 | 7 categories, ~2K test | APIGen/NexusRaven |
| V2 Live | Aug 2024 | +1,500 real user queries | ToolACE complexity, Hammer irrelevance |
| V3 Multi-Turn | Sep 2024 | Stateful dialog | APIGen-MT blueprint |
| V4 Agentic | 2025 | Long-horizon + web/memory | SWE-Gym ([[ch-27]]) |

**7 Categories**: Simple / Multiple / Parallel / Parallel-Multiple / Relevance-Detection / Multi-Turn / Multi-Step.

**AST matcher (scoring)**:
```
1. Parse predicted + gold into (name, kwargs)
2. name: exact match
3. kwargs: sorted by key, whitespace stripped, literals canonicalized
4. required args present
```

**Lenient on representation** (모두 equivalent):
- `get_weather(city="Paris")` ≡ `get_weather(city='Paris',)` ≡ with optional args
- Argument order / quote style / trailing comma / whitespace 무관
- `1.0 ≡ 1`, `"red" ≡ 'red'`

**Strict on semantics** (모두 reject):
- `get_weather(loc="Paris")` (wrong arg name)
- Missing required args / wrong function name

→ Philosophy: "Lenient on representation, strict on semantics."

**pass^k metric (V3+)**: k independent trials *모두* success해야 score.
- xLAM-2-70B τ-bench: pass^1=56.2%, pass^4=39.4% (30% relative drop)
- Consistency 부족하면 pass^k 급락 → APIGen-MT blueprint design의 motivation

### Part B — "Benchmark becomes data spec" 의미

**일반 pattern**: benchmark는 *독립적 measure*, data pipeline은 *arbitrary design*.

**BFCL pattern**: benchmark category structure가 *data pipeline 설계의 blueprint*. Benchmark = single measure 아닌 **field-shaping force**.

**Chapter §6 quote**: 
> "Whoever sets the eval taxonomy sets the data-generation taxonomy."

**증거 — 모든 pipeline이 BFCL 구조 mirror**:
| Pipeline | Data 구조 | BFCL 대응 |
|---|---|---|
| APIGen | 4 data types | BFCL 첫 4 categories 정확히 |
| ToolACE | 5 levels {30/25/20/15/10} | BFCL 분포에 hand-tuned |
| Granite | 7 capabilities | BFCL 7 categories 동일 |
| Hammer | 30% irrelevance aug | BFCL category 5 target |
| APIGen-MT | Blueprint-rollout | BFCL V3 Multi-Turn |

→ 각 paper의 data 구조가 BFCL의 derivative. Paper들이 명시적으로 인정: ToolACE *"training against the eval, not coincidentally matching"*.

**V2 Live = community's anti-overfitting check**:
```
V1 (Feb): 7 categories, synthetic
    ↓ everyone optimizes V1
V1 saturation
    ↓ V2 Live adds real user queries (Aug)
V1-overfit model이 V2 Live에서 drop
    ↓
ToolACE/Hammer 같은 real-distribution-aware pipeline 등장
```

→ V2 Live가 self-correcting mechanism. 2025 reporting rule: "lead with V2 Live + V3 multi-turn. V1 saturated."

**Concentration of power**:
- BFCL maintainer가 V3 추가 → 전체 field가 multi-turn pipeline에 투자
- V4 추가 → agentic data 투자
- **BFCL이 field의 *research direction*을 implicit하게 set**

### Part C — [[Q4]] framework로 보면 (field-level instantiation)

[[Q4]] distribution overfitting의 **field-level 버전**:

| | Q4 (single pipeline) | BFCL data spec (field) |
|---|---|---|
| Overfit target | 자기 training dist | **BFCL category dist** |
| Detector | external real anchor | **V2 Live real queries** |
| Risk | production deployment gap | **모든 pipeline collectively overfit** |
| Mitigation | real-human anchor 5-15% | **V2 Live같은 unseen test** |

→ 한 pipeline이 아니라 *field 전체*가 한 benchmark에 overfit. Q4 critique의 BFCL-level extension.

### 정리

**BFCL 한 줄**: FC의 de facto benchmark. AST matcher × 7 categories × 4 versions × pass^k.

**"Benchmark becomes data spec" 한 줄**: BFCL 7 categories가 *모든 pipeline data design의 blueprint*가 됨. APIGen/ToolACE/Granite/Hammer 모두 BFCL derivative. Benchmark가 *field-shaping normative force*.

**Implication**:
- 좋음: 표준화 → cross-paper comparison
- 나쁨: collective overfitting → V2 Live가 self-correction
- 더 나쁨: maintainer가 implicit하게 field research direction set

---
