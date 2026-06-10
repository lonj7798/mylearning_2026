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
