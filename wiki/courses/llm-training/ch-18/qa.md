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

## Q-재독 1. ROUGE-L < 0.7이 뭔가? (2026-09-28, 개정판 재독)

**Kernel.** Self-Instruct가 새 instruction을 추가할지 정하는 유사도 필터: 이미 채택된 모든 instruction과 비교해 ROUGE-L이 전부 0.7 미만일 때만 추가(§2.2). ROUGE-L = 두 문장의 LCS(같은 순서로 나타나는 가장 긴 공통 단어열, 인접하지 않아도 됨)로 계산한 F 값(`P=LCS/|a|, R=LCS/|b|, F=2PR/(P+R)`). Worked example(§4): "Write a short poem about the sea"(7) vs "Write a poem about the ocean"(6) → LCS 5, P 0.714, R 0.833, ROUGE-L 0.769 ≥ 0.7이라 제외; "Compose verses describing waves at night"는 겹침이 없어 추가되지만 같은 종류의 요청 — 표면 겹침 threshold는 lexical repetition의 상한일 뿐 task repetition을 막지 못함. 다양성은 일반화를 재는 단위(고유 문제·task 종류)에서 재야 함(OpenMathInstruct-2: pair 256K 고정, 1K vs 6.5K 고유 문제에서 MATH 10점 이상 차이). 정정: "약 50% 탈락"은 논문에 없음(정정 항목 2); 0.7의 ablation은 없음(Recipe). see [[read]] §4.

## Q-재독 2. ROUGE-L을 비교하는 "두 문장"은 response인가, system/user message인가? (2026-09-28)

**Kernel.** instruction 텍스트끼리 비교한다. Self-Instruct 샘플은 instruction + (선택) input + output으로 구성되며 ROUGE-L < 0.7 필터는 새 instruction을 task pool의 모든 기존 instruction과 비교(excerpt Filtering 단계). input·output은 대상이 아님. 단일 turn 형식이라 system/user/assistant role 구분이 없음(chat template과 다른 형식; instruction이 chat의 user 요청 문장에 대응하지만 role 구조는 없음). 같은 filtering 단계의 다른 규칙: instruction 수준 — "image, picture, graph" 키워드 제외, 너무 길거나 짧은 instruction 제외; instance 수준 — 중복 instance, input이 같고 output이 다른 instance 제거(정확 일치), 너무 길거나 짧은 output 제거. 따라서 ROUGE-L은 instruction 다양성 관리용이고, 한 instruction에 달린 output의 다양성은 이 필터가 보지 않음(ch-08b Q12의 label multiplicity와 같은 결의 문제). see [[read]] §4, excerpts/self-instruct.md.

## Q-재독 3. instruction 하나에 ROUGE-L을 재는 것이 task 유사도 측정으로 맞는가? (2026-09-28)

**Kernel.** 문장 반복 방지 용도로는 맞고, task 유사도 측정으로는 맞지 않는다: ROUGE-L은 instruction의 표면 단어 겹침(LCS 기반)을 잴 뿐이라 두 방향으로 어긋난다. (1) 같은 task인데 놓침(장의 예): "Write a short poem about the sea" vs "Compose verses describing waves at night" — 겹침 없어 추가됨. (2) 다른 task인데 버림(이 course의 예, 장 밖): "Translate this sentence into French" vs "...into German" — LCS 4, P=R=0.8, ROUGE-L 0.8 ≥ 0.7이라 제외되나 목표 언어가 다른 별개 task. Self-Instruct의 ROUGE-L 세 가지 쓰임: 필터(새 vs 기존 instruction, <0.7), 다양성 진단(Fig. 4, 생성 vs 시드 instruction 겹침 분포), 평가(Table 3, unseen SuperNI 119 tasks에서 모델 출력 vs 정답). 장의 결론: 다양성은 일반화를 재는 단위(고유 문제·task 종류)에서 재라(OpenMathInstruct-2: pair 수 고정, 고유 문제 수만 바꿔도 결과 큰 차이). 의미 수준 dedup은 outline상 ch-12(미확인). see [[read]] §4, excerpts/self-instruct.md.

## Q-재독 4. Nemotron에서는 ROUGE-L > 0.7인 prompt가 학습에 쓰인 적이 없는가? (2026-09-28)

**Kernel.** 그렇게 추론할 수 없다. ROUGE-L < 0.7은 Self-Instruct(§2.2)의 규칙이고, Nemotron-4 340B 보고서는 prompt 간 ROUGE-L 필터를 서술하지 않는다. §2 표 Step 3: Self-Instruct = ROUGE-L < 0.7 필터; Nemotron-4 = Genetic Instruct 코드 데이터의 dedup만 언급되고 방법 미서술(§3.3.1; 정정 항목 8 — 그 적합도 검사도 컴파일러가 아니라 LLM); excerpt: task-family partitioning으로 다양성을 관리하고 공격적 near-dup 제거에는 덜 의존. Nemotron의 보고된 필터: dialogue의 reward-model 점수 threshold(Filter), ground truth/Python verifier 또는 LLM judge→reward-model judge(Verify), reward model로 chosen response 선택(Select) — 모두 점수 기준이며 instruction 간 유사도를 보지 않음. 따라서 "0.7 초과 prompt가 쓰였는지"는 알 수 없음(보고 없음, 측정하려면 데이터 공개와 별도 측정 필요). "Not reported"(소스를 확인했으나 서술 없음)와 "안 했다고 보고됨"은 다르며, Nemotron이 dedup을 안 했다는 뜻도 아님. see [[read]] §2, 정정 항목 8.
