<!-- chapter: ch-19 Q&A part 2; deps: [[read]], [[qa]]; kernel answers only -->
# Ch-19 — Reading Q&A (Part 2: Magpie / Persona / WRAP / Humpback)

Continued from [[qa]]. Part 1 covers Q1-Q6 (method comparison + Evol + verifier patterns).

## Q7. What is Magpie's prefix-only trick? Is it instruction tuning?

**No — opposite.** You don't train; you *exploit* an already-aligned model's chat template. Two forward passes per data pair:

1. Feed ONLY `<|user|>` prefix → model fills in plausible user instruction (it learned the user-turn marginal as a side effect of SFT).
2. Feed that back through full template → get response.

**Why it works** (line 150): aligned models have sharp low-entropy distribution on user-prefix continuation. They implicitly learned user-turn distribution. Temperature-1 sampling gives diverse-but-well-formed instructions.

**Removes**: seeds, teacher API, prompt engineering. **Remains**: filter design + compute (~$2/M vs ~$500/M for Alpaca; 3M → 300K via 8 metrics).

**Limitation**: inherits teacher's distribution ("explain X" not real-user vibes); empty stage 4 — needs downstream RM/DPO.

## Q8. Can I use Magpie data as a replay dataset?

**Yes for general-capability preservation, NO for domain-specific work.** Magpie imports *teacher's* distribution + quirks + verifier gap → augmentation OK, primary anchor dangerous. Decision rule: Magpie ≤ 10% of replay mix; for domain work, rephrase-from-real beats it. Filter with your own verifier before mixing (Magpie has empty stage 4).

## Q9. Does Persona-conditioning really work? What does it fix vs not fix?

**Yes — strong empirical evidence — but only for *framing variance*, not correctness/depth/format.**

Evidence (line 175-179): amplifier effect (80%-similar personas → 40-50% similar outputs); Qwen2-7B on MATH 64.9% (matched gpt-4-turbo) with 1.07M persona-synthesized; math-validity audit 96.5%.

| Bottleneck | Persona helps? |
|---|---|
| Framing variance (same task, different voice/context) | ✅ Yes |
| Factual correctness | ❌ No — verifier's job |
| Reasoning depth | ❌ No — Evol operators' job |
| Format diversity | ❌ No — explicit format variation |

**One-liner**: persona fixes how things are *said*, not *right* or *hard*. For sales-call: right tool for customer-voice diversity, paired WITH stage-4 verifier + format variation.

## Q10. WRAP: 원본 C4를 training에서 제외시켰나?

**No — 명시적으로 keep + 1:1 mix.** Line 194: 각 문서가 원본 + rephrase 두 형태로 training set에 등장. 이유: (1) 100% synthetic이면 Mistral voice overfit; (2) raw C4의 noisy real distribution이 generalization에 필요; (3) [[ch-18]] line 137 "accumulation over replacement" 원칙의 pretraining 적용.

**Token math**: "5× less data" 주장은 raw만 카운트한 것. 실제론 WRAP = 15% raw + 15% rephrase = 30% total tokens vs baseline 100% = **3.3× token 효율**. 5×는 measurement convention 차이.

## Q11. WRAP의 verification + "augmentation vs data refinement" framing

**Verification은 aggregate level**: per-sample verifier 거의 없음. Lightweight 조치만 (chunk cap, boilerplate strip, held-out leakage check). 진짜 verification = Pile-subset perplexity uniform 감소. Pretraining의 noise-robustness 덕분에 가능.

**Framing — augmentation이 아니라 *data refinement***: 외부 정보 추가 X. 15% raw를 4가지 style로 가공해서 student가 사실을 style-invariant하게 학습. Information은 15%에 bounded. 같은 textbook을 4가지로 공부하는 것. 이 원칙이 [[ch-21]]의 Phi/Nemotron/R1-distill로 이어짐.

## Q12. Humpback도 base model을 못 넘는가? (Evol-Instruct와의 차이)

**Yes — 하지만 *다른 종류*의 천장.** Evol은 *generation ceiling* (푸는 능력), Humpback은 *comprehension ceiling* (이해 능력). Document는 real이라 seed model이 생성 못 했어도 OK — 단지 inferred instruction을 추론할 만큼 이해는 해야 함.

**핵심 비대칭**: 보통 **이해 > 생성**. Humpback이 seed model의 generation 범위를 넘는 content에서도 (I, D) pair 추출 가능. **여전히 bounded**: comprehension 밖이면 inferred instruction이 generic/wrong. Self-curation circularity는 그대로.

## Q13. Humpback의 verification + cleaned raw-data 필수성

**Verification target**: "inferred instruction이 real document와 matching하나?" 더 나은 옵션: cross-model curator, bidirectional NLI, round-trip generation comparison, human spot-check on 1%.

**Cleaned raw data가 필수 — unique vulnerability**: 5개 method 중 **유일하게 document가 직접 response가 됨**. Garbage docs → garbage student. Bootstrap/Magpie/Persona는 generate하니까 noise-tolerant. WRAP은 Mistral rephrase로 smoothing. Humpback만 raw data quality에 *직접* 노출.

**Required preprocessing** = [[ch-09]]..[[ch-17]] 데이터 track이 사실상 prerequisite: dedup, quality filter, boilerplate removal, length normalization, PII/safety filter, format check.

## Q14. Humpback의 inverse problem — P(I|D)는 distribution

**문제**: forward `P(D|I)`는 well-defined이지만, 역방향 `P(I|D)`는 distribution. 같은 D에 대해 여러 plausible I 존재 → many-to-one inverse.

**Feature mode**: paraphrastic variants → student가 robustness 학습 (multi-prompt augmentation 효과).
**Bug mode**: 다른 intent variants가 같은 D에 landing → incoherent (예: "List organelles" → prose D = format/scope 혼란).

**원논문 접근**: 보수적 sampling (1-3 per doc) + aggressive self-curation. K-sampling 확대하려면 cross-model curator + diversity check + format/length match 필요.

**Bayesian framing**: `P(I|D) = P(D|I)·P(I)/P(D)`. 핵심은 prior `P(I)` — seed model이 internalize한 "사람들이 쓰는 instruction 분포". 이 prior가 실제 user distribution과 다르면 prior mismatch.
