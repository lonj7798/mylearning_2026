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

---

## Q-정정 (2026-09 revision)

2026-09 revision에서 [[read]]를 primary source 기준으로 다시 작성했다. 아래 항목은 기존 entry의 kernel 중 수정된 사실이다. 기존 entry의 line reference(예: "line 22", "line 156", "line 171", "line 173")는 git commit 4a72e54 시점의 read.md를 가리킨다.

- **Q3** — "verify 없는 pipeline은 student를 망가뜨린다"는 일반 규칙이 아니다. APIGen에서는 check에 실패한 sample을 다시 넣으면 BFCL이 xLAM-7B에서 4.06~5.94점, xLAM-1B에서 9.59~12.17점 떨어졌다(Fig. 5). OpenMathInstruct-2에서는 256K pair 이상일 때 오답 solution 20%까지 accuracy 손실이 거의 없었다(Fig. 5). 영향은 output 형태, student 크기, data 규모에 따라 다르다.
- **Q6** — Nemotron-4의 human data는 "20K human prefs"가 아니다. 약 20K 중 10K는 SFT용, 10K는 reward model과 preference fine-tuning용 HelpSteer2이다(§3.2). "anchor가 pipeline을 calibrate한다"는 문장은 report에 없고 이 course의 해석이다.
- **Q7** — "Nemotron과 APIGen은 anchor를 seed로 쓰지 않는다"는 틀렸다. APIGen은 seed QA example을 sampling하고 verified sample을 seed set에 다시 추가한다(§3.1). Nemotron-4는 human-written example을 few-shot으로 써서 수행할 수 없는 task의 question을 생성하고, LLM에게 그 question에 대한 refusal response를 쓰게 한다(§3.2.5).
- **Q8** — ProLong은 document-grounded synthetic long instruction data의 성공 사례가 아니다. ProLong은 synthetic long data를 SFT에 섞었을 때 short UltraChat만 쓴 경우보다 점수가 낮았다(0%에서 55.7, 50%에서 43.3, Table 8). 최종 SFT는 UltraChat만 사용했다.
- **Q10** — OpenMathInstruct-2는 SymPy를 쓰지 않는다. 새 question에는 gold answer가 없어서 같은 teacher의 solution 32개 중 majority answer를 기준으로 쓰고, evaluation은 GPT-4o judge로 한다. 또한 LLM 기반 verifier도 실제로 쓰인다. APIGen의 semantic check는 LLM이고, Nemotron-4는 초기에 LLM-as-judge를 쓰다가 Chat-Hard accuracy가 더 높은 reward model(0.87 vs 0.54)로 바꿨다(§3.2.3). 인용된 "line 173" 문장은 revision에서 삭제했다.
- **Q12** — OpenMathInstruct-2의 14M 규모는 "정확한 verifier로 오답을 제거"한 결과로 설명되지 않는다. vote threshold는 0이었고, judge와 reward model filtering은 gain이 없었으며(Table 3), 논문은 teacher 강도(37.9 vs 30.1, Table 2)와 unique question 수(Fig. 6)의 효과를 보고한다. reward hacking과 judge 설계는 ch-26이 아니라 [[ch-42]]와 [[ch-49]]에서 다룬다.
- **Pattern Map** — "new math → symbolic equivalence → SymPy" 행은 OpenMathInstruct-2에 해당하지 않는다. gold answer가 없는 math question의 verifier는 majority vote이고, 이 방식은 teacher가 반복하는 같은 오류를 걸러내지 못한다.
