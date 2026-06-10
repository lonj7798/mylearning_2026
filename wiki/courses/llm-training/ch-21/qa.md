<!-- chapter: ch-21 Q&A; deps: [[read]]; kernel answers only -->
# Ch-21 — Reading Q&A

## Q1. "Top-down expansion"의 의미 — taxonomy-driven generation의 방향

**Literal**: Root(abstract category) → Leaf(specific concept) 방향으로 *specializing* 하면서 트리를 내려감. Teacher는 leaf에 도달하면 거기서 실제 instruction을 작성.

**ch-19 (bottom-up)과의 차이**:

| | Bottom-up (ch-19 Bootstrap/Evol) | Top-down (ch-21 GLAN/Phi) |
|---|---|---|
| Start point | Concrete seed (real example) | Abstract category (root) |
| Move | Mutate seed → variants | Specialize category → subcategories |
| Coverage 결정 | Seed pool span | Curator tree structure |
| Gap visibility | 보이지 않음 | 명시적으로 보임 (빈 가지) |

**예시 (GLAN)**:
```
Science → Physics → Mechanics → Newtonian → 
  Conservation laws → Momentum → "1D elastic collision" 
  → 여기서 instruction 작성
```
6 levels of LLM-driven recursion. 각 level은 *카테고리 list*이지 example이 아님.

**왜 ch-19 E5 (diversity ceiling)의 직접 답**:
- Bottom-up: seed가 *우연히* span한 영역만 cover. 빠진 영역 invisible.
- Top-down: curator가 *의도적*으로 가지 설계. "Number Theory 가지 없다" = 명시적으로 보임 → 추가 가능.
- → ceiling이 *seed-bound (invisible)* → *curator-bound (visible + fixable)*로 이동.

**핵심 통찰**: *Structure first, content second*. Leaf에서 content 만들기 *전에* 트리 구조로 coverage 결정. Generate는 last step.

## Q2. GLAN — dedup과 verifier 없어도 되나?

### Dedup — 부분 yes, 결정적 nuance

**구조적으로 불필요한 것**: *Inter-leaf* dedup. 트리가 서로 다른 영역 cover하니까 leaf간 near-dup 안 생김.

**여전히 필요한 것**: *Intra-leaf* dedup. 같은 leaf 안에서 prompt cosmetic variation만 하면 반복 발생. Cosmopedia 사례(line 209, 250, 254):
- 1차: 30M prompts → too many duplicate-class outputs (cosmetic variation: child vs adult 같은 leaf 안 variation만)
- Fix: post-hoc filter 강화 아니라 *prompt taxonomy 자체를 restructure* (audience × format × context 각각을 real taxonomy로)
- Failure signal: 40% 이상 dedup된다면 = 트리가 너무 얕음

→ 너의 "question level dedup만이면 되나" 직관 정확. 단, 정답은 *filter 강화가 아니라 tree branching factor 추가*.

### Verifier — GLAN은 *없음*. Intentional design

| Method | Axis 5 (verifier) | Axis 9 (Q-side curation) |
|---|---|---|
| GLAN | **0** (teacher only) | **최대** (taxonomy) |
| Nemotron-4 | 강함 (RM-as-judge) | 강함 (6 task families) |
| Phi | 0 (style filter only) | 강함 (curated topics) |
| R1-distill ([[ch-20]]) | 강함 (math-verify) | 보통 |

**GLAN의 verifier 부재 이유**:
1. Broad knowledge target — humanities/social science는 verifier 자체가 존재하지 않음
2. Teacher (GPT-4) = implicit filter라는 베팅
3. Scale priority — per-sample verify 비용 > generation 비용

**Format-level mitigation** (line 103): `# Varied difficulty; verifier-friendly formats for math/code`. Verifier 내장은 안 하지만 *출력 format*을 verifier-friendly하게 → downstream에 위임.

**Cost** (line 114): *"GLAN does not produce skills the teacher lacks. The ceiling is the teacher."* → Axis 5=0이라 [[ch-20]] teacher bias inheritance가 최대치로 발현.

**Nemotron-4 대조** (line 122-127): 같은 lab이 Nemotron-4-340B-Reward를 동시 학습 → 20K HelpSteer2 anchor로 RM 학습 → generator sample을 filter + preference pair 선택. = Top-down generation + 강한 verifier 결합.

### Framework 위치 (line 227-229)

3-layer architecture:
1. Top-down taxonomy = *breadth* (GLAN/Phi/Nemotron 모두 여기)
2. Bottom-up bootstrap = *cross-leaf imagination* ([[ch-19]])
3. Dedup + verify = *quality* ([[ch-18]] loop, [[ch-22]] selection)

→ GLAN은 layer 1 specialization. Layer 3는 *deliberately empty*. [[ch-22]]/[[ch-23]]에서 채움.

## Q3. "Intra-leaf dedup"이 정확히 무엇? LLM 전인가 후인가?

### 정의
**Leaf** = taxonomy 트리의 말단 concept node. 핵심: leaf 1개 → instruction *여러 개* (~10/concept, line 247).
**Intra-leaf dedup** = 같은 leaf에서 만들어진 N개 sample 사이의 중복 체크.

### Bad example — Cosmetic variation
```
Leaf: Fermat's Little Theorem
  Sample 1: "Prove 2^10 ≡ 1 (mod 11) using FLT"
  Sample 2: "Prove 3^16 ≡ 1 (mod 17) using FLT"   ← 숫자만 다름
  Sample 3: "Prove 5^28 ≡ 1 (mod 29) using FLT"
```
서로 다른 leaf와는 구분되지만 (inter-leaf OK), 같은 leaf 안에선 near-duplicate.

### Timing — detection vs fix

| 단계 | 무엇 | LLM 위치 |
|---|---|---|
| Detection | N개 output 서로 비교 (dedup score) | LLM *후* |
| Fix | Prompt 자체에 structural variation 강제 | LLM *전* |

**Cosmopedia 교훈** (line 209): post-hoc filter 강화는 *misleading*. 진짜 정답은 LLM 전 prompt 재설계.

### Cosmopedia의 fix — prompt-family expansion

1차: `Explain {topic} to {child|adult}` → 30M 중 절반 near-dup (style만 다름)
Fix: 트리에 *structural axis* 추가
```
audience(3) × format(4) × context(4) = 48 distinct prompt families / topic
```
→ "Prompt-family structure" 의미: cosmetic axis (style, tone) 아니라 *structural axis* (audience capability, output format, context domain).

### 핵심
**Top-down dedup의 실제 해결책 = "tree branching factor 추가"**. Post-hoc filter는 비용만 늘고 효과 약함. ch-18 loop의 "dedup" step이 top-down에선 *upstream*으로 이동.

## Q4. Top-down은 teacher ceiling 못 깬다? — Two-ceiling decomposition (framework extension)

**Learner thesis**: tree-structure axis는 teacher data extraction일 뿐, verifier 생략하니 teacher에 highly reliant → ceiling 못 깸.

**확장 — ceiling 단일 아니라 *두 개*. Top-down은 *하나* 깨고 *다른 하나* 락**:

| Ceiling | 의미 | Bottom-up | Top-down |
|---|---|---|---|
| **Coverage** | 공간의 어느 영역을 sample 가능한가 | Seed-bound (낮음) | Curator-bound (높음) |
| **Quality** | 각 sample이 얼마나 좋을 수 있나 | Teacher-bound | Teacher-bound (동일) |

→ Top-down은 *coverage* 깸 + *quality* 락. 이게 trade-off 정확한 modal.

### Quality ceiling을 깨는 mechanism들 (forward-link)

핵심: 모두 *axis 5 (verifier coverage)*의 specialization. Axis 5=0 (GLAN/Phi)이면 teacher 외 평가자 없음. Axis 5>0이면 teacher와 *독립된 기준* 존재 → teacher error catch → student가 teacher 능가 가능.

| Mechanism | Chapter | Ceiling-breaking 이유 |
|---|---|---|
| Verifier-grounded RL (R1) | [[ch-44]] | Math/code verifier가 teacher 무시. Student가 reward 따라 teacher가 못 푸는 문제도 학습 |
| Process reward | [[ch-24]] | Outcome verifier의 limit A 극복. Reasoning path 평가 → wrong-question-correctly trap 차단 |
| Pass@k self-improvement | [[ch-19]] E6 | Student → verifier → student 루프. Teacher 불필요 (self-distillation) |
| Critic-as-judge + anchor (Nemotron RM) | [[ch-21]] §3 | Anchor data로 RM 학습 → independent quality signal |

### Adoption boundary — GLAN/Phi가 작동/실패하는 조건

**작동**:
1. Student capacity ≪ Teacher capacity → teacher ceiling이 *binding constraint 아님* (7B vs GPT-4 격차 크니까 student가 먼저 saturate)
2. Target = broad SFT/knowledge breadth → quality보다 *coverage*가 binding

**실패**:
1. Frontier student (70B+) → teacher 거의 따라잡으면 quality axis가 binding → verifier 필요
2. Reasoning/math/code target → outcome verifier 가능 → axis 5 활용 시 ceiling 돌파 가능

### Framework-tag 재진술
**"Top-down은 axis 9 최대화 + axis 5 비워둠 → coverage ceiling 깸 + quality ceiling 락 → student << teacher일 때만 binding constraint 깬다."**

→ 2025년 현재 7-13B는 GLAN/Phi, 70B+ MoE는 R1-style verifier RL 분기되는 이유.

## Q5. "RM is both filter and judge" — Nemotron design의 효율성

### 두 역할 한 모델

| Role | Output mode | Use | Decision |
|---|---|---|---|
| Filter | Binary (threshold τ) | SFT data 포함/제외 | "이 sample 충분히 좋은가?" |
| Judge | Ranking (pairwise) | DPO preference pair (chosen, rejected) | "둘 중 어느 게 더 나은가?" |

### 구체 예 — 같은 N samples → 두 dataset
```
Prompt: "Solve x²+3x-4=0"
Samples + RM scores: A(0.92), B(0.71), C(0.58), D(0.18)

Filter (τ=0.7):    A, B → SFT data
Judge (pairs):     (A,D), (A,C), (B,D), (B,C) → DPO preference data
```

### 전통적 pipeline과 대비

**분리**: SFT용 LLM-judge + preference용 human labeler → 다른 calibration, anchor 중복.
**Nemotron 통합**: Generator → RM(continuous score) → threshold(filter) + pair(judge) → 한 anchor set(20K HelpSteer2)이 SFT + preference 둘 다 covering.

### Verifier 종류 분리 ([[ch-20]] axis 5 확장)

| Verifier | Output | Filter? | Judge? |
|---|---|---|---|
| Math-verify (R1, Stratos) | Binary (boxed == gold) | ✓ | ✗ (binary는 두 통과 sample ranking 불가) |
| Unit-test (code) | Binary (pass/fail) | ✓ | ✗ |
| **RM (Nemotron)** | **Continuous** | **✓** | **✓** |
| LLM-judge (Sky-T1) | Continuous (but weak) | ✓ | △ |

→ **Continuous + anchor-trained verifier가 binary verifier보다 strictly 더 powerful**. Open-ended domain (humanities, creative writing)은 binary 불가능 → RM 필수.

### [[Q4]] (two-ceiling) 연결
Nemotron-4 thesis: top-down generation + 강한 continuous RM이 *broad domain*에서 ceiling-breaking 최적 조합. GLAN(coverage only) + Math-verify(binary, narrow) 합집합을 RM이 cover. Cost: anchor data + 340B RM 자체.

## Q6. "Staged SFT" — 순서가 만드는 차이 (line 152)

### 구조
**Code SFT first → General SFT second**. Mixed simultaneous 아닌 2-stage 순차.

### 왜 순서가 결정적인가 — 3 mechanism

**1. Cleaner-signal-first**: Code = binary verifier 가능 (compile / unit test) → label noise 거의 0. General = noisy. 깨끗한 base 깔고 noisy로 확장하면 noisy gradient가 base를 덜 흔듦.

**2. Format prior installation**: Code SFT가 student에 install — structured output, step-by-step decomposition, determinism, strict format following. General SFT는 soft loss라 위 습관을 덮지 않음. *Code = scaffolding, General = breadth layer*.

**3. Catastrophic forgetting 비대칭**:
| 순서 | 효과 |
|---|---|
| **Code → General** | General loss soft → code 습관 보존 |
| General → Code | Code loss hard (rigid syntax) → 큰 gradient → dialogue 덮어씀 |

→ *덜 destructive한 stage가 뒤*. Code가 더 destructive(rigid)니까 *먼저*.

### 대안 비교
| Order | 결과 |
|---|---|
| Mixed | Clean signal이 noise에 dilution |
| **Code → General** | 최선 |
| General → Code | Code stage가 dialogue 손상 |

### Adopters
Phi-4 (explicit), Qwen-2.5, Tülu-3, Nemotron-4 (line 138-147 pseudocode 내부 순서).

### [[ch-20]] R1 4-stage isomorphism
R1: cold-start SFT(clean) → reasoning RL(verifier) → rejection sampling(partial verifier) → final RL(noisy RM).
**원리 동일: strong-signal-first, broad-noisy-last**. Staged SFT는 SFT-only 버전, R1은 RL extension.

### Framework position
ch-21은 *data generation* 중심이지만 staged SFT는 *consumption schedule*. **Data quality/verifiability gradient를 학습 순서로 변환하는 design**. Synthetic data pipeline의 hidden axis — 같은 batch 섞으면 dilution, 순서대로 먹이면 scaffold + layer.

### 한 줄
**"Cleaner signal로 scaffolding 깔고 그 위에 breadth. 역순은 noisy가 clean을 덮어쓰니까 불가능."**

## Q7. 왜 code data는 less noisy, general은 noisy? — grammar는 *enabler*, verifier는 *actor*

### "Noisy"의 3 axes 분리

| Axis | Code | General |
|---|---|---|
| A. Output space 크기 | 작음 | 큼 (valid response 무한) |
| **B. Correctness checkability** | **Binary + automatable** | Subjective, learned judge 필요 |
| C. Surface strictness | 엄격 (syntax error=fail) | Loose |

학습자 hypothesis ("grammar/structure")는 axis A + C. 진짜 답 = **axis B**.

### Causal chain — grammar는 enabler
```
Grammar → Compiler/interpreter → Unit test → Automated binary verifier
                                                    ↓
                                       Teacher error post-hoc 잡음
                                                    ↓
                                       SFT label noise ≈ 0
```
→ Grammar 자체가 noise를 줄이는 게 *아님*. Grammar는 *verifier를 가능*하게 함. **Noise reduction의 actor는 verifier**.

### General이 noisy인 진짜 이유
- "Write a poem about loneliness" → compiler 없음, unit test 불가 → **automated verifier 자체가 존재하지 않음**
- 강제로 learned judge (LLM-judge / RM) 사용 → judge가 학습 데이터 noise 상속 (inter-annotator agreement ~70-80%, length bias)
- 결과: SFT data label noise 10-30%

### 학습자 hypothesis의 limit
Code SFT도 *세부 quality*에선 noise 존재 (style, perf, idiomaticity). 단 *correctness binary noise* ≈ 0이라 scaffold로 *충분히* 깨끗. Math도 동치: grammar(LaTeX) + deterministic semantics → verifier 가능. Humanities: grammar 있지만 *deterministic semantics 없음* → verifier 불가 → noisy.

### 깊은 함의
**"Grammar는 verifier의 precondition. 진짜 noise reduction actor는 verifier. General에 verifier가 없는 건 grammar 없어서이기도 하지만 더 근본적으로 *deterministic semantics가 없어서*."** → 코드/수학에 verifier가 가능한 진짜 이유.

## Q8. Code SFT에 주석(comment) 넣는 게 좋은가? — *Generation method에 의존*하는 axis

### Two competing forces

**Cost (학습자 worry)**: comment = free-form natural language → verifier 못 검사 → wrong comment가 학습되면 wrong association.

**Benefit**: comment는 *why*를 encode (verifier는 *what*만). Plan-then-code scaffold — R1의 "trace 먼저, answer 나중"과 isomorphic, code 버전.

### 핵심 mechanism — Comments inherit *partial* verifier coverage by co-generation

```
Teacher generates: (comment + code) as one unit
Code passes unit test → SFT data로 keep
                       ↓
Comment는 teacher가 code와 *일관되게* 만든 것
→ Comment noise ≈ Teacher coherence rate (작음)
→ Comment가 verifier에 *간접 보호* 받음
```

| Comment source | Coverage | Noise |
|---|---|---|
| Raw scrape (The Stack) | 0 (drift) | 높음 |
| **Teacher synth + code passes** | *Indirect via co-gen* | **낮음** |
| Teacher synth separately | 0 | 중간 |

→ "주석 도움 되나"의 정답 = *어떻게 생성됐는가*에 의존. Phi-1이 raw Stack 안 쓰고 *textbook-quality synthesis* 한 이유.

### Phi-1 example pattern
Docstring containing intent + reasoning + math (recurrence relation). Verifier 통과 = teacher coherence 보증 = comment quality 보증.

### [[Q4]] (two-ceiling) 연결
Pure code: verifier가 correctness만 cover. Code+synth comment: verifier가 correctness cover + comment가 co-gen으로 reasoning *implicitly* cover. → **Comment는 verifier coverage를 자연어 reasoning까지 확장하는 trick**.

### Cost & mitigation
**Worst case**: model이 "verbose comment + wrong code" 학습 — R1 thinking-leak isomorphism.
**Mitigation**: plan-style comment only (intent + reasoning, not surface description); teacher *same-call*로 comment + code 동시 생성; separate generation 금지.

### [[Q6]] (staged SFT)와 연결
Code-with-synth-comments-first가 *최선 scaffolding*: code structure + reasoning scaffold 둘 다 clean. General SFT가 위 올라올 때 student는 이미 "plan in language, execute in structure" pattern 보유.

### 한 줄
**"Comment는 verifier-absent 영역이지만 *teacher-co-generated + code-verifier-passed*면 *partially verified by association* → reasoning scaffold benefit. Scrape면 학습자 worry대로 noisy."**

## Q9. All-textbook으로 pretrain하면 best model? — NO. Two-axis trade-off

### 너 직관의 문제 — Quality와 Coverage axis conflate

| Axis | Textbook synth | Raw filtered web | Best |
|---|---|---|---|
| Per-token quality | 높음 | 낮음 | 둘 다 |
| Distribution coverage | 좁음 (curator's worldview) | 넓음 | 둘 다 |
→ "All-textbook"은 axis 1 max + axis 2 좁음. Best는 *둘 다*.

### Phi 자체가 *all-textbook 안 함* (line 166-169)

| Model | Synthetic % | Filtered web % |
|---|---|---|
| Phi-1 | ~17% | ~84% |
| **Phi-4** | **~10%** | **~rest** |

"Textbooks Are All You Need" = marketing, not technical claim (line 177). 가장 textbook-friendly한 Phi-4도 90% filtered web.

### 5 reasons all-textbook이 fail

1. **Contamination** (line 179): Phi-1 synthetic exercises ↔ HumanEval overlap. 50.6% under suspicion. All-textbook = contamination 전면화.
2. **Narrow capability** (line 180): Phi-1 Python only. Curator anticipated 영역만 cover.
3. **Curator bias = *systematic noise*** (random noise보다 위험): web noise는 통계적으로 상쇄, textbook bias는 모든 출처가 같은 view라 상쇄 안 됨.
4. **Real-user distribution mismatch** (line 219-221): user query는 cross-domain, informal, messy. Textbook style 학습 → real query fail.
5. **Model collapse risk** ([[ch-23]] forward, line 266): recursive synthetic → distribution mode collapse. Phi 방어책 = single-shot teacher.

### "Zero noise" 재정의 — [[Q7]] framework 적용

Textbook-likeness classifier ≠ verifier. **Style filter**일 뿐 **correctness filter** 아님. Style이 textbook이라고 content가 옳다 보장 안 됨 — Phi contamination이 정확히 이 failure mode.

→ Q7의 진짜 통찰 ("verifier = noise reduction actor") 적용 시: textbook synthesis는 *verifier 없음*. 그래서 noise 0이 아니라 *systematic bias로 압축된 noise*.

### Best 구성 (chapter line 227-229 + Phi-4 pattern)

3-layer + substrate:
```
~80-90%: Filtered web (real human data) → coverage 확보
~10-20%: Synthetic textbook (curated taxonomy) → quality 부스트
Curriculum: clean base → noisy 위로 ([[Q6]] staged 원리)
```

### [[Q4]] + [[Q7]] 통합 — Three ceilings

All-textbook이 잠그는 ceiling:
- Coverage ceiling: curator's tree
- Quality ceiling: teacher's capability
- **Distribution ceiling** (NEW): real-world usage ≠ textbook style

→ Real deployment에서 distribution ceiling이 binding constraint. Web 없으면 fail.

### 한 줄
**"Zero noise는 *style noise* 0이지 *systematic bias* 0 아님. Filtered web의 random noise > textbook의 systematic bias for generalization. Phi-4도 90% filtered web. Textbook은 *quality booster*, substrate 아님."**

→ [[ch-23]] forward: model collapse 본격 다룸.

## Q10. Math-only + all-textbook은 작동? "ALL human data"가 synthetic 가능?

### Part 1 — Math-only edge case

학습자 직관 *부분 정확*. Math는 [[Q9]] 5 reasons 중 3개가 binding constraint 아닌 *드문 domain*:

| 조건 | 일반 | **Math** |
|---|---|---|
| Verifier ([[Q7]]) | ✗ | ✓ |
| Curator's tree completeness | 큰 영역 누락 | **거의 완전** (자연 taxonomy) |
| Distribution coverage | 모든 인간 활동 | 수학 problem space만 |
| Long-tail | 무한 | 제한적 |

**Evidence**: Llemma (Proof-Pile-2 200B), MetaMath (synthetic rephrase), MathScale ([[mathscale]] top-down taxonomy 2M), DeepSeek-Math (120B + RL).

**그러나 *순수* all-synthetic 안 됨 — 3 caveat**:
1. Language substrate 필요 — "if a train travels 60 mph..." 읽으려면 영어 학습 데이터 필요
2. Verifier-RL이 SFT alone보다 강함 (DeepSeek-Math pattern) — [[Q4]] quality ceiling
3. Frontier math (research conjectures, novel proofs)는 textbook 외부

**Realistic math-only build**: ~80% filtered math web (Proof-Pile, arXiv, StackExchange) + ~15% synthetic textbook + ~5% verifier-RL.

→ Textbook **비율** 일반 case (10%)보다 높음. 단 *0 web* 여전히 fail.

### Part 2 — "ALL human data → synthetic" — *circular impossibility*

**Bootstrapping paradox**: ALL synthetic을 만들 generator는 ALL을 *already 알아야* 함. 그러면 *generator = 목표 모델*. Recursive.

**3 fundamental limits**:
1. **Generator는 자기 distribution 밖 sample 불가** — paradigm shift 못 emit. Recombine만 가능.
2. **Compression is lossy** — synthesis = teacher worldview 압축. Lossless 가능하려면 generator = 완벽 모델 = 목표 자체.
3. **"Noise"가 signal인 경우**: register switching, typo robustness, cultural/temporal specificity, code-switching, disagreement, real human intents. Curator의 noise = information theory의 signal일 수 있음.

**Borges paradox**: ALL human data 완벽 합성 = 인류 모든 상호작용 재현 = synthesis 아닌 collection. 비용 무한.

**Frontier 시도** (real data dependence 줄이려는):
- Persona-driven (personas 자체가 curator 상상)
- Web-rephrase (Cosmopedia, Phi-4) — 여전히 real seed 필요
- Self-improving — model collapse risk
- Multi-teacher diversity — bounded by union of teachers

→ 모두 0으로 못 감. **Real data = grounding anchor, not replaceable substrate**.

### 학습자 직관이 가리키는 진짜 framework axis

`synthetic_optimal_ratio = f(domain_narrowness, verifier_existence, curator_completeness, long_tail_relevance⁻¹, language_substrate_need⁻¹)`

| Use case | Synthetic 비율 |
|---|---|
| Math from scratch | ~15% textbook + 80% web + 5% RL |
| Code from scratch | ~15% (Phi-1) + 84% Stack |
| General SFT | 50-98% (Nemotron 98%, GLAN 100%) on base |
| General pretrain (Phi-4) | ~10% |
| Frontier general | real web dominant |

→ **Narrow + verifier-rich + curator-complete domain일수록 synthetic ↑**.

### 한 줄
**"Math-only edge case는 textbook 비율 ↑ 가능 (~15%), 단 0 web 여전히 fail. ALL synthetic은 bootstrapping paradox로 *논리적* 불가능. Real data = grounding anchor."**

## Q11. "각 domain마다 다른 type의 data 필요?" — Toolkit 같고 Mix 다름

### Precision 분리
- **Type (toolkit)**: 모든 domain 동일 — real text + synthetic textbook + verifier signal
- **Mix (ratio + 구체화)**: domain별로 다름 — 비율, 어떤 verifier, 어떤 taxonomy, staging

→ 같은 toolkit, domain-specific configuration. (빵/파스타/피자 비유)

### Domain properties가 mix를 *predict*
```
optimal_mix = f(
    verifier_existence,        # ↑ synthetic
    taxonomy_completeness,     # ↑ top-down
    long_tail_relevance,       # ↑ real web
    language_substrate_need    # ↑ real web
)
```

| Target | Verifier | Tax | Long-tail | Substrate | Mix |
|---|---|---|---|---|---|
| Math | ✓ | 완전 | 중요 | 필요 | 80%web/15%synth/5%RL |
| Code | ✓ | 부분 | 약간 | 필요 | 84%web/16%synth |
| Dialogue | △RM | 부분 | 큼 | 필요 | 90%+web/RM-SFT |
| Knowledge QA | ✗ | 부분 | 매우큼 | 필요 | web dominant |

### Multi-domain implication
Generalist >> specialist 어려움. Generalist는 *모든 domain mix의 union*. 단일 batch 불가 → [[Q6]] staged SFT가 *시간축 분리* 해법.

### Ch-21 thesis 재진술
| Paper | 활용 property | Limit |
|---|---|---|
| GLAN | Taxonomy completeness | Verifier ✗ → teacher ceiling |
| Nemotron | Verifier (RM) + Taxonomy | RM anchor 필요 |
| Phi | Taxonomy + style filter | Verifier ✗ → contamination, narrow |

→ Chapter는 *one method 추천 안 함*. *Domain → property → method* 매핑이 진짜 목적.

### 한 줄
**"같은 toolkit, 다른 mix. Mix는 4 property로 deterministic하게 predict. 학습자 '각 domain마다 다르다' 직관 정확, 단 *property로부터 도출*되는 것."**

## Q12. Taxonomy / Long-tail / Substrate — 한국어 정의

### Taxonomy completeness (분류 체계 완전성)
**의미**: domain 지식을 얼마나 깔끔하게 트리로 분할 가능한가.
**완전하다** = 전문가 합의된 자연 분류 + 빈 가지 적음.
**불완전하다** = 영역 fuzzy, 합의된 분류 없음.

| Domain | 완전성 | 이유 |
|---|---|---|
| Math | 거의 완전 | algebra/geometry/analysis 수백 년 합의 |
| Code | 부분 | language별 paradigm 진화 |
| Dialogue | 불완전 | 경계 fuzzy |
| Creative writing | 거의 없음 | 분류가 작가 정체성 침해 |

**의의**: 완전하면 GLAN-style top-down 가능; 불완전하면 bottom-up seed 필요.

### Long-tail relevance (롱테일 희귀 case 중요도)
**의미**: 흔하지 않은 *희귀 case*가 실제 성능에 얼마나 결정적인가.
**중요** = 꼬리가 user value의 일부.
**덜 중요** = head case만 잘하면 만족.

| Domain | 예시 | 중요도 |
|---|---|---|
| Math | Frontier conjectures, IMO creative tricks | 중요 (textbook 외) |
| Knowledge QA | 희귀 역사 사실 | 매우 큼 (Wikipedia 전체 long-tail) |
| Creative writing | 비표준 dialect, 마이너 장르 | 무한 |

**의의**: 중요하면 real web 필수; textbook은 head만 cover.

### Language substrate need (언어 기반 필요도)
**의미**: 기본적인 *언어 이해* 자체가 필요한가.
**필수** = 거의 모든 domain (problem이 자연어).
**덜 필요** = pure symbolic (Lean/Coq formal proof).

| Domain | 필요? | 어디서 |
|---|---|---|
| Math word problems | 필수 | Web + textbook |
| Pure formal proof | 거의 불필요 | Lean library |
| Dialogue | 필수 + register 다양성 | Web |

**의의**: substrate 없으면 *문제 자체를 못 읽음*. 거의 모든 실용 domain의 prerequisite.

### 통합 — [[Q11]] mix formula 재진술
- **Real web을 push하는 axes**: Substrate + Long-tail
- **Synthetic을 push하는 axes**: Verifier + Taxonomy
- 두 force balance가 mix 결정.

