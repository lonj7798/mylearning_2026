<!-- chapter: ch-18 — companion Q&A
     deps: [[read]]
     scope: questions raised while reading; kernel answers only — see [[read]] for full reasoning
-->

# Ch-18 — Reading Q&A

Questions the learner raised while reading [[read]]. Each entry is the **kernel** of the answer; full causal chains live in the chapter or the discuss transcript.

---

## Q1. What does "permanent mental model" mean? (line 22)

Just terminology. It means **a reusable analytical framework** — internalize the six-stage loop once, reuse it forever as the default lens for reading any synthetic-data paper. No deeper technical meaning.

## Q2. Filter vs Verify — what's the real axis?

**Surface form vs ground truth**, not "non-LLM vs LLM."

- **Filter** answers *"does this look okay?"* — cheap, syntactic. (regex, JSON parse, length cap, lang-ID)
- **Verify** answers *"is this actually correct?"* — expensive, semantic. (run code, SymPy, RM score, judge)

For code: filter = AST parses; verify = unit tests pass. Verify can be deterministic (no LLM needed) — the axis is *connection to ground truth*, not "is an LLM involved."

## Q3. Why does Pipeline A (no verify) destroy student models?

Cross-entropy puts probability mass on whatever appears in training data. Wrong-but-fluent samples → student learns to **imitate the teacher's failure modes with confidence**. Errors are *correlated* (not random noise), so the distribution shifts directionally toward those bugs. Forward-link: this is the engine of model collapse — see [[ch-23]].

## Q4. Does few-shot generation work for single QA pairs?

Yes — and most pipelines (Self-Instruct, Alpaca, Evol-Instruct, Magpie) produce single QA pairs. Two independent axes:

- **Prompt structure to teacher**: few-shot vs one-shot vs zero-shot
- **Output unit per call**: usually one QA pair regardless

Magpie is the zero-shot exception (prefix-only sampling, no seeds).

## Q5. What is ROUGE-L filter?

A **stage-3 dedup mechanism** based on Longest Common Subsequence similarity. Self-Instruct drops any new instruction with `ROUGE-L > 0.7` against the accepted pool. Surface metric — misses semantic duplicates with no shared words. Cost is O(n²); MinHash + LSH is the scalable upgrade.

The dedup metric menu: n-gram overlap, ROUGE-L, MinHash, embedding distance — all plugged into the same "score similarity, drop above threshold" filter pattern.

## Q6. What is an anchor? (section 4.1)

**A small, trusted, human-curated set that calibrates the entire synthetic pipeline.** Two jobs:

- **Job A** — seed for stage 1 Generate (optional)
- **Job B** — calibrate filter/verifier as a regression test (always present)

Per-pipeline anchors: Self-Instruct 175 seeds, Nemotron-4 20K human prefs, APIGen 3,673 API references, OMI-2 ~15K MATH/GSM8K problems. Amplification ratio (anchor : synthetic) = leverage from the human investment. The "100% synthetic" claim is always a marketing slogan — anchors always exist.

## Q7. Is anchor the same as seed?

No — **anchor ⊇ seed**. Seed is one *use* of anchor (Job A). Anchor's load-bearing job is calibration (Job B). Some pipelines use anchor only for Job B (Nemotron, APIGen) and never as seeds.

## Q8. How is anchor used for long-conversation synthesis?

Three amplification patterns (covered formally in [[ch-25]] and [[ch-28]]):

1. **Self-chat from seed** (Baize): anchor = seed Q → LLM plays both roles → N-turn dialogue
2. **Scaffold expansion** (UltraChat / CAMEL / Persona-Hub): anchor = taxonomy or persona → LLM instantiates the structure
3. **Document-grounded** (LongAlign / ProLong): anchor = long document → multi-turn QA grounded in chunks

Each pattern has a matching verifier: coherence judge / RM / groundedness probe.

## Q9. What does "Rewrite preserves labels; faithfulness check, not correctness check" mean? (line 156)

Rewrite-style generation (WRAP, Cosmopedia, paraphrase) reuses an existing source document, so:

- The original is already validated → no need to check correctness
- The verifier only checks **faithfulness** — did the rewrite preserve the source's meaning?

Cheap (NLI / embedding sim / LLM compare against source). Rarely catastrophic because verifier failures cause minor surface drift, not fact errors. **But not "never catastrophic"** — iterated rewrites still drift cumulatively, which is why [[ch-23]] applies even here.

## Q10. Why can't we just use an LLM as the verifier?

**Generator-verifier independence is the load-bearing property.** If GPT-4 generates `23 × 47 = 1051` and you ask GPT-4 "is this right?", it confidently says yes — same biases, same blind spots.

Production verifiers connect to **ground-truth signals the LLM doesn't have**: code → sandbox; math → SymPy; preference → RM trained on human anchor; rewrite → original document (free reference!).

Rewrite is unusual because the source document *is* the ground-truth reference, so an LLM-as-comparator works fine there. For code / math / preference, ground truth lives outside the LLM. Per line 173: *"Generation is a commodity; verification is the moat."*

## Q11. Backtranslate — verify and filter mechanisms (line 157)

**Backtranslate** = round-trip through another language or modality (English → French → English; text → speech → text). Original is the implicit ground-truth reference, same shape as rewrite.

**Stage 4 (verify)** — checks the round trip preserved info, not truth:

| Verifier | What it measures | Example threshold |
|---|---|---|
| Bidirectional NLI entailment | Original ⇔ round-trip both entail each other | Both directions must say "entailment" |
| Embedding cosine similarity | Semantic distance in vector space | `cos(emb(orig), emb(rt)) > 0.85` |
| BERTScore / METEOR | Token-level semantic overlap with embeddings | `BERTScore F1 > 0.9` |
| Label-preservation check | Apply original's classifier to round-trip | Sentiment / intent / topic unchanged |
| Round-trip BLEU (NMT) | n-gram overlap between forward translation and gold target | Task-specific |
| Entity preservation | Named entities (people, places, dates, numbers) appear in both | Exact-match drop if any entity vanishes |

**Stage 2 (filter)** — cheap surface drops: length ratio outside `[0.5, 2.0]`, language-ID mismatch, repetition / token loops, encoding artifacts (mojibake), format break (JSON / markdown structure lost), empty output.

**Why grouped with rewrite**: both shift work *off* stage 4 — the original is a free ground-truth reference. Bootstrap and full-generate don't have this and must reach for executor / SymPy / RM / judge.

## Q12. What does "compound" mean in "verifiable tasks compound"? (line 171)

**Compound = 복리처럼 self-reinforcing.** 각 iteration의 이득이 다음 라운드 입력 품질을 높이는 self-reinforcing loop.

- **Verifiable**: 정확한 verifier → 오답 제거 → 데이터 신호↑ → 모델↑ → 더 나은 데이터 → ... (OMI-2 14M 솔루션 스케일 메커니즘).
- **Unverifiable**: cheap verifier 없음 → judge 편향 누적 → 모델이 judge에 over-fit → 품질 정체 or 하락 (ch-26 reward hacking).

**Career**: verifiable domain은 시간이 갈수록 자동으로 좋아짐. Unverifiable은 judge engineering이 병목.

---

## Pattern Map

```
Stage 1 type            Ground truth available    Verifier shape
──────────────────────────────────────────────────────────────────
new code                execution result          sandbox
new math                symbolic equivalence      SymPy
preference pair         human preference          RM + anchor
rewrite                 original document         NLI / LLM compare
multi-turn dialogue     coherence + groundedness  judge / topic check
```

The verifier's *shape* is determined by what ground-truth signal the modality provides — not by which LLM you happen to be using.
