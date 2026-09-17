<!-- chapter: ch-19 Q&A; deps: [[read]]; kernel answers only -->
# Ch-19 — Reading Q&A

## Q1. For classification, is bootstrap the most efficient method?

**No — rephrase is.** Two reasons:

1. **Bootstrap has a known modal-label collapse** for classification (line 52): without input-first branching, classification tasks degenerate to a single modal label. Self-Instruct had to engineer around this.
2. **Rephrase preserves labels by construction** ([[ch-18]] line 156): if you have a real labeled dataset, rephrasing the input keeps the label correct without verifier work. Bootstrap invents both input AND label, requires a verifier to catch wrong labels, and bootstrap pipelines have empty stage 4.

Deduction chain for classification: have labels → preserve labels → rewrite/backtranslate → cheap faithfulness verifier → safe to scale. Real pipelines compose: rephrase + persona + backtranslate + label-consistency NLI check + 2:1 mix with original.

## Q2. What's the difference between rephrase and bootstrap? Is bootstrap "starting from scratch"?

**Bootstrap uses seeds — but the output is NEW content, not the seeds themselves. Rephrase uses source documents — and the output preserves the source's content, just reworded. "Starting from scratch" is full-generate, not bootstrap.**

| Method | Starts from | Output is |
|---|---|---|
| Bootstrap | A few seed tasks (175 in Self-Instruct) | NEW tasks not in the seed pool |
| Rephrase | An existing real document | A paraphrase of that document |
| Full-generate | Nothing concrete (just topic / taxonomy) | New content from scratch (Phi textbooks) |

**Seed vs source — the key axis**: seed (bootstrap) = *template* for what output should look like; source (rephrase) = actual *content* output must preserve. Analogy: bootstrap = "write articles like these examples"; rephrase = "paraphrase this Wikipedia entry."

**Why this matters for classification**: bootstrap invents both content AND label (verifier needed); rephrase keeps label untouched while rewording (no verifier needed).

## Q3. Sales-call pipeline sketch — deferred gaps

Sketch composed all 5 methods + multi-agent dialogue. Three under-specified gaps: (1) customer LLM realism (deferred to [[ch-25]]); (2) stage-4 split — compliance regex (verifiable) + quality judge (unverifiable); (3) stages 5-6 missing — objection-cluster selection + mix ratio 1:5–1:20. Revisit at [[ch-25]] / [[ch-26]].

## Q4. Evol-Instruct의 5 In-Depth + 1 In-Breadth operators

**왜 필요**: Self-Instruct/Alpaca는 단어 다양성은 있지만 *난이도 분포*가 평탄. Evol의 주장: complexity histogram이 first-class training knob. **6개 operator** (read.md L82-90):
- In-Depth (난이도 ↑): Add constraints / Deepening / Concretizing / Increased reasoning steps / Complicate input
- In-Breadth (다양성 ↑): Mutation (같은 task family 안에서 rare domain으로 변이)

**Pipeline**: Alpaca 52K → 무작위 operator → elimination filter (same/refusal/empty/copy drop) → 4 rounds → ~250K. **왜 작동**: In-Depth(difficulty)와 In-Breadth(topic)가 직교 → long-tail histogram.

**한계 — Teacher Saturation**: Teacher가 자기가 못 푸는 문제를 안 만듦. WizardMath는 *bidirectional* (upward + downward, downward는 reasoning manifold smoothing). **교훈** (line 129): operators are domain-specific — WizardCoder는 5개 code-native operator 사용.

## Q5. Why can't Evol-Instruct generate problems harder than the teacher's level?

**One-liner**: same LLM is both generator AND verifier → ceiling tied to teacher's solving competence. To break it, decouple verification from generation.

Three-layer mechanism: (1) LLMs autoregressively sample from training distribution — no "construct + verify"; (2) pipeline requires teacher to produce response → can't solve = refuses or step-downs (saturation); (3) RLHF-trained self-aware refusal prevents confabulation beyond ceiling.

**Fix** (line 117): WizardMath's RLEIF adds IRM + PRM (independent). See Q6 for pattern map. **Terminology**: "no independent verifier" (structural) > "verification is limited" (quality). Ceiling is *teacher's*, not base model's (student).

## Q6. How to use LLM-as-verifier in Evol-Instruct

**One-liner**: helps only when (a) verifier *independent* of generator, or (b) verifier task is *comparison against ground truth*, not original solving.

| Pattern | Ceiling-breaking? |
|---|---|
| Same LLM as judge | ❌ Correlated failures |
| Cross-model judge (GPT-4 + Claude) | ⚠ Partial decorrelation |
| Self-verification (CoT ×2) | ❌ Catches noise, not bias |
| Multi-sample agreement | ❌ Confident-wrong gets unanimous |
| **Trained judge (PRM / IRM)** | ✅ Independent training signal |
| **LLM judge + gold reference** | ✅ Comparison ≠ solving |

**Why trained judges break ceiling**: PRM/IRM on gold labels know things teacher doesn't (WizardMath's RLEIF, [[ch-44]]/[[ch-26]]). **Why gold-ref works**: comparison easier than solving. **2024+ recipe**: cross-model judges + reference-based matching; custom judge only if budget + contamination concern.

---

**Q7-Q14 (Magpie, Persona, WRAP, Humpback deep dives) moved to [[qa-deep]] per CLAUDE.md "split if it grows" rule.**

---

## Q-정정 (2026-09 revision)

**배경**: read.md가 2026-09 generality revision으로 새로 작성되었다. 아래 Qn의 kernel에는 이제 정정된 사실이 들어 있다. 괄호의 번호는 새 read.md "Corrections to the version you studied"의 항목 번호이다. Q1–Q6(이 파일)과 Q7–Q14([[qa-deep]])에 있는 line reference(예: "line 52", "L82-90", "line 150")는 git commit 4a72e54 시점의 read.md를 가리킨다.

- **Q1**: Self-Instruct의 classification branch 방향이 반대로 적혀 있다. Classification task는 label을 먼저 만들고 각 label에 맞는 input을 만드는 output-first 방식을 쓰고, input-first는 non-classification task에 쓴다. Input-first가 한 label 쪽으로 치우친 input을 만들었기 때문이다 (1). Self-Instruct에는 correctness verifier가 없지만 filter(ROUGE-L, keyword, 중복 instance, 길이)는 있으며, 200개 audit에서 모든 field가 valid한 비율은 54%이다.
- **Q4**: In-breadth operator는 "같은 domain 안에서 더 rare한 instruction"을 만든다. Elimination 규칙은 (1) ChatGPT가 판정한 information gain 없음, (2) "sorry" 포함이면서 80 words 미만, (3) punctuation과 stop word만 있음, (4) evolving prompt의 단어("given prompt" 등) 복사이다. 250k는 4 round 전체 instruction 수이고, SFT에는 그중 70k를 sampling했다. Paper는 histogram이 아니라 round별 mean difficulty(Alpaca 3.00 → round 4 7.08)를 보고한다. WizardMath의 "reasoning manifold smoothing" 설명은 paper에 없다. WizardCoder의 5개 heuristic에는 "language/library 지정"이 없고 "erroneous code를 misdirection으로 제시"가 있다 (8–10, 12–13, 15).
- **Q5**: WizardLM과 WizardCoder의 evolver는 GPT-4가 아니라 gpt-3.5-turbo이고, "teacher ceiling" 효과를 측정한 결과는 인용된 paper에 없다. 따라서 Q5의 mechanism은 source가 없는 Interpretation이다. WizardCoder에서 GPT-4 evolver는 GPT-3.5 대비 HumanEval pass@1 62.2 vs 59.8이다. IRM × PRM reward의 xref는 ch-26이 아니라 ch-44이다 (11, 14).
- **Q6**: WizardMath의 IRM과 PRM은 gold label이 아니라 GPT-4-0613이 만든 ranking과 step 판정으로 학습된다. 즉 data를 만든 teacher와 같은 GPT-4 계열 model의 label이며, "gold label로 학습된 독립 judge"의 예가 아니다. ch-26 xref는 ch-44로 바뀐다 (12, 14).
- **Q7**: Magpie 비용은 "~$2/M"가 아니라 1,000 instance당 $0.12(Air), $1.1(Pro)이다. "sharp low-entropy distribution" 설명은 paper에 없고, paper는 implicit memorization을 가설로 제시한다. Magpie는 reward model과 reward difference filter를 쓰므로 "empty stage 4"는 틀리다. Magpie-Air에서 3M-Raw(22.96 LC)와 300K-Filtered(22.66 LC)는 거의 같다 (18, 24).
- **Q9**: "80% 유사한 persona → 40-50% 유사한 output"은 text에 없는 수치이다. Text가 주는 값은 persona similarity 0.9에서 대부분의 problem similarity가 0.6–0.75라는 것이다. Embedding similarity가 낮다고 transfer가 좋은 것은 아니다: Prismatic Synthesis에서 persona-diverse set의 OOD accuracy는 38.15, seed-diverse set은 54.77이다 (20).
- **Q10**: 1:1 mix는 맞다. "5× less data"는 paper §1의 표현이고, Table 1은 real token 기준(85B vs 170B)으로 비교한다. "15% of C4"는 1/0.15 ≈ 6.7× 적은 real token이다. Pile perplexity 개선은 ACL version abstract에서 ">50%", arXiv v1 abstract에서 ">10%"로 version 사이에 차이가 있다 (21).
- **Q11**: Fig. 1(b, c)의 결과는 4가지 style이 아니라 Q/A style rephrase로 얻은 것이고, §7 RQ2는 Table 1의 synthetic data도 QA prompt로 만든 data라고 설명한다. Perplexity 감소는 domain별로 균일하지 않고, synthetic-only는 여러 Pile sub-domain에서 perplexity가 나빠진다. 8개 general task 평균 +2.0 중 1.9는 BoolQ와 TruthfulQA 두 task에서 나오고 8개 중 4개 task는 낮아진다 (21–22).
- **Q12, Q13**: Instruction은 "aligned seed model"이 아니라 3,200개 seed pair로 reverse 방향 fine-tune한 backward model이 만든다. Self-curation이 external curation보다 낫다는 근거는 없다: 250개 dev set에서 precision/recall은 M0 0.44/0.09, M1 0.52/0.44, GPT-4 0.88/0.92이다. Q13의 "cross-model curator가 더 낫다"는 방향은 이 결과와 일치한다 (23).
- **Q14**: Humpback은 segment마다 candidate instruction을 1개 생성한다(§2.2). "1-3 per doc"는 paper의 값이 아니다 (23).
