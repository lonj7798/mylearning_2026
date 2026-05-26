<!-- chapter: ch-48
     track: eval
     kind: content
     title: Contamination Workflow
     deps: [ch-47]
     sources: [[deduplicating-training-data]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[scaling-laws-data-quality]], [[faithful-synth-eval]], [[dolma]], [[fineweb]], [[anthropic-sleeper-agents-data]], [[bespoke-stratos]]
     figures: figures/contamination-detect.html
-->

# 48장 — 오염(Contamination) Workflow

> **핵심 통찰.** Contamination은 ruled in/out할 수 있는 단일 사건이 아니다. 그것은 "no overlap"과 "verbatim leakage" 사이의 연속선 위에 존재하는 *signal-to-noise problem*이다. 모든 detector(n-gram match, MinHash, canary string, embedding neighbour)는 특정 false-positive와 특정 false-negative를 교환하며, bucket size와 cutoff의 선택은 config detail이 아니다. 그것이 곧 contamination claim이다. 방어 가능한 memo는 detector가 무엇을, 어떤 operating point에서, 어떤 recall gap을 갖고 찾았는지 말한다. **"the eval is clean"**이라고 말하지 않는다.
>
> **가이드라인.** 어떤 training stage도 실행하기 전에 eval-set hash를 pin하라. 두 n-gram granularity로 hash하라(recall을 위한 fine ~8 tokens, precision을 위한 coarse ~13–50 tokens). O(1) per-token lookup을 위해 Bloom filter에 저장하고, overlap은 corpus 단위가 아니라 *eval instance별 fraction*으로 계산하라. eval마다 세 숫자를 보고하라: fine-n-gram leakage %, coarse-n-gram leakage %, semantic-neighbour leakage %(embedding cosine 또는 MinHash Jaccard). paraphrased leakage는 n-gram으로 잡히지 않는다. memo에서 explicit recall gap으로 언급하라. training이 RM preference data 또는 RS→SFT synthesis로 흘러 들어갈 때는 raw pretraining corpus에서만이 아니라 *모든 stage boundary*에서 decontamination을 실행하라.

---

## 1. Contamination이 보기보다 어려운 이유

[[deduplicating-training-data]]는 기본적인 난처함을 정량화했다. 2021년에 **LM1B validation set의 4.6%가 이미 training set에 있었고**, **C4의 3.2%도** 그랬다. 아무도 의도적으로 하려 하지 않았던 시대에도 그랬다. dedup 이전의 모든 reported perplexity number는 memorized validation sentence의 long-tail 때문에 부풀려져 있었다.

이 문제의 현대적 버전에는 세 가지 새로운 특징이 있다.

1. **Eval set은 release 후 며칠 안에 leak된다.** Common Crawl snapshot은 GitHub와 HuggingFace page를 1–2주 안에 다시 crawl한다. 월요일에 release된 benchmark가 금요일이면 CC dump에 verbatim으로 나타날 수 있다. leakage는 당신 lab의 잘못이 아니다. open web의 base-rate다.
2. **Synthetic data는 teacher memorization을 운반한다.** [[bespoke-stratos]]는 명시적으로 flag한다. *"AIME와 MATH prompt는 공개되어 있으며, teacher가 solution을 memorized했을 수 있다."* R1이 pretraining 중 AIME 2024를 봤다면, 모든 distilled trace가 그 memorization을 물려받는다. seed prompt만 decontaminate하는 것은 충분하지 않다. generated trace도 filter해야 한다.
3. **Adversarial contamination이 존재한다.** [[anthropic-sleeper-agents-data]]는 trigger-conditioned example이, 표면 형태가 반복되지 않아 n-gram matching을 피하는 paraphrased variant로 주입될 수 있음을 보인다. clean n-gram report는 adversarial threat model 아래 clean corpus의 증거가 *아니다*.

따라서 이 장의 workflow는 "script를 실행하고 verdict를 얻는다"가 아니다. operating point를 고르고, known-positive set에 대해 측정하고, FP/FN을 정직하게 보고하며, recall gap에 이름을 붙이는 것이다.

---

## 2. Detection table — methods × n-gram size × cutoff × FP/FN

| Method | Typical n-gram / unit | Cutoff | FP regime | FN regime | Source |
|---|---|---|---|---|---|
| Exact-substring (suffix array) | 50 tokens | ≥1 match | 거의 0(50-token collision이 우연히 생길 확률은 천문학적으로 낮음) | paraphrase, translation, 어떤 edit도 놓침 | [[deduplicating-training-data]] ExactSubstr |
| Token n-gram overlap | 8 tokens | ≥1 match in instance | 낮음에서 중간(common phrases: "the United States of America") | light paraphrase를 놓침 | Llama / GPT decontam convention |
| Token n-gram overlap | 13 tokens | ≥80% of eval-instance n-grams hit | 매우 낮음 | moderate paraphrase, section reorderings를 놓침 | [[llama-3]] convention, GPT-3 convention |
| MinHash + LSH (doc-level) | 5-gram shingles, ~9000 sigs, 20×450 bands | Jaccard ≥ 0.8 | 매우 낮음 | 긴 doc 안의 짧은 exact overlap을 놓침 | [[deduplicating-training-data]] NearDup |
| Canary string | fixed 64–128-char unique token | literal substring | 0(by construction) | exact canary에 대해서만 0; *generalisation claim 없음* | [[anthropic-sleeper-agents-data]] analogue |
| Embedding neighbour | sentence-embedding cosine | ≥0.9 | 중간(topic matches) | paraphrase를 잡지만 noisy; manual triage cost 높음 | [[faithful-synth-eval]] cluster occupancy |
| Perplexity gap | — | loss(eval) << loss(control) | noisy; topic / length와 confound | 강한 memorization만 잡음 | membership-inference style |

**이 표를 읽는 법.** 각 행은 frontier를 따라 움직인다. 더 빡빡한 cutoff와 긴 n-gram은 FP를 줄이지만 FN을 늘린다. 단일 행이 지배하지 않는다. 방어 가능한 workflow는 **최소 두 행**을 사용한다. high-precision exact 또는 coarse-n-gram detector로 verbatim leakage를 bound하고, lower-precision paraphrase-sensitive detector(embedding 또는 MinHash Jaccard)로 paraphrase gap을 bound한다.

Bucket size choice — 실무 rule of thumb:
- n=8 → recall-favoured; Wikipedia-style corpora에서 common phrase로 인한 1–3% FP rate를 예상하라.
- n=13 → 2020년대 GPT-3 / Llama default; web에서 FP rate는 경험적으로 <0.1%.
- n=50 → [[deduplicating-training-data]]의 ExactSubstr threshold; 사실상 0 FP.

n-gram size choice를 한 문장으로 설명할 수 없다면 memo는 defensible하지 않다.

---

## 3. Concrete decontamination pipeline (pseudocode)

이것이 reference pipeline이다. 실제 구현(OLMES, Llama internal, OLMo 3의 decontam utility)은 구조적 변형이다.

```python
# decontaminate.py — eval-set leakage against a pretraining / SFT / RM corpus.
# Attested primitives: MinHash+LSH from [[deduplicating-training-data]],
# per-snapshot hashing from [[fineweb]], filter-order convention from [[dolma]].

from typing import Iterable
from bloom import BloomFilter              # pybloom / rbloom; O(1) check
from datasketch import MinHash, MinHashLSH # 5-gram shingles

EVAL_SETS    = load_eval_sets()           # list of (name, [instance_text])
N_FINE       = 8                          # recall-favoured n-gram
N_COARSE     = 13                         # precision-favoured n-gram (Llama 3 convention)
OVERLAP_CUT  = 0.80                       # fraction of eval-instance n-grams that must hit
MINHASH_SIGS = 9000                       # [[deduplicating-training-data]] NearDup
LSH_BANDS    = 20; LSH_ROWS = 450         # threshold ~0.8 Jaccard

# Step 1: pin eval set hashes ONCE, before any training stage.
fine_bf   = BloomFilter(capacity=1e9, error_rate=1e-6)
coarse_bf = BloomFilter(capacity=1e9, error_rate=1e-6)
eval_ngrams_by_instance = {}              # {(eval_name, idx): set_of_coarse_ngrams}

for name, instances in EVAL_SETS:
    for idx, text in enumerate(instances):
        toks = tokenize(text)
        fine   = {tuple(toks[i:i+N_FINE])   for i in range(len(toks)-N_FINE+1)}
        coarse = {tuple(toks[i:i+N_COARSE]) for i in range(len(toks)-N_COARSE+1)}
        for g in fine:   fine_bf.add(hash(g))
        for g in coarse: coarse_bf.add(hash(g))
        eval_ngrams_by_instance[(name, idx)] = coarse

# Step 2: stream the training corpus; flag docs that collide with eval n-grams.
lsh = MinHashLSH(threshold=0.8, num_perm=MINHASH_SIGS, params=(LSH_BANDS, LSH_ROWS))
for name, instances in EVAL_SETS:
    for idx, text in enumerate(instances):
        m = MinHash(num_perm=MINHASH_SIGS)
        for sh in shingles(text, k=5): m.update(sh.encode())
        lsh.insert(f"{name}:{idx}", m)

flagged = []                              # (doc_id, eval_key, overlap_fraction, reason)
for doc_id, doc_text in corpus_stream():  # e.g. jsonl shards
    toks = tokenize(doc_text)
    # (a) cheap pre-filter: any fine n-gram hit at all?
    hits = [g for g in ngrams(toks, N_FINE) if hash(g) in fine_bf]
    if not hits: continue
    # (b) coarse overlap per eval-instance
    doc_coarse = set(ngrams(toks, N_COARSE))
    for key, eval_coarse in eval_ngrams_by_instance.items():
        if not doc_coarse & eval_coarse: continue
        frac = len(doc_coarse & eval_coarse) / max(1, len(eval_coarse))
        if frac >= OVERLAP_CUT:
            flagged.append((doc_id, key, frac, "coarse-ngram"))
    # (c) MinHash-LSH for paraphrase-ish
    m = MinHash(num_perm=MINHASH_SIGS)
    for sh in shingles(doc_text, k=5): m.update(sh.encode())
    for neighbour in lsh.query(m):
        flagged.append((doc_id, neighbour, m.jaccard(lsh.get(neighbour)), "minhash"))

# Step 3: remove flagged docs from the corpus; ALSO recurse into RM / SFT pools.
drop_from_corpus(flagged)
```

세 가지 성질이 중요하다.
- **Eval-set hash는 training 전에 pin된다.** 그렇지 않으면 "decontaminated" corpus가 잘못된 eval에 대해 decontaminate된 것이다. [[olmo-3]]가 decontamination을 reusable utility로 제공하는 이유도 이 ordering bug가 흔하기 때문이다.
- **Per-instance overlap fraction이지 per-corpus가 아니다.** corpus-level 0.01% overlap은, 그것이 이제 모두 leak된 5개의 eval instance에 집중되어 있다면 무의미하다.
- **Recurse.** pretraining corpus를 flag하고, SFT mix에 대해 다시 실행하고, RM preference pool에 대해 다시 실행하고, rejection-sampling outputs에 대해 다시 실행하라([[llama-3]] §Post-training). 각 synthesis stage가 contamination을 다시 도입한다.

---

## 4. Canary strings와 watermarks — 그리고 그것들이 할 수 없는 것

**Canary string**은 eval set에 의도적으로 삽입된 unique token(64–128 char hex blob, nonsense word, structured tag)이다. 나중에 모델이 이를 emit하면 memorization의 증거가 된다. 이는 [[anthropic-sleeper-agents-data]]의 trigger construction을 방어 목적으로 재활용한 analog다.

Canary가 주는 것:
- 그 exact string의 verbatim leakage에 대한 **zero-FP detector**.
- provenance trail: canary가 downstream model output에 나타나면 leakage를 특정 eval artifact에 귀속할 수 있다.

Canary가 주지 않는 것:
- eval의 **non-canary** 부분에 대한 어떤 claim도 없다. clean canary report는 주변 question의 90% paraphrased leakage와 양립 가능하다.
- preprocessing에 대한 robustness: 대부분의 data pipeline은 whitespace를 normalize하거나 lowercase하거나 non-ASCII를 strip한다. 이를 견디는 canary form을 고르라.
- **paraphrased leakage**에 대한 robustness: training corpus가 eval의 rephrased version을 포함하면 canary는 결코 나타나지 않지만 answer는 memorized일 수 있다.

**Watermarks**(token distribution의 statistical bias)는 dual problem이다. output이 특정 model에서 왔는지를 detect하지, input이 특정 eval에서 왔는지를 detect하지 않는다. contamination과 직교한다. distillation-lineage tracking([[bespoke-stratos]]의 teacher-provenance concern)에는 유용하지만 train-eval overlap에는 유용하지 않다.

**Paraphrase limit (정직한 숫자).** AIME-style math에서 paraphrased leakage — problem statement를 semantics를 보존하며 rewording하는 것 — 은 실제로 모든 n-gram threshold ≥6을 살아남는다. embedding neighbour at cosine ≥0.9는 대부분 잡지만 manual triage가 필요하다. realistic budget에서 경험적 ceiling은 paraphrased item에 대해 대략 70–80% recall, 5–10% FP rate다. memo에 이 gap을 명시하라. 덮지 말라.

---

## 5. Downstream contamination: train → RM pref → eval

이 failure mode는 [[llama-3]] recipe의 data flow를 읽으면 명백해진다.

1. Pretraining corpus가 eval instance를 포함한다(accidental 또는 public benchmark의 CC re-crawl).
2. SFT는 prior checkpoint에서 rejection-sampled output을 사용한다. 그 output은 eval prompt와 비슷하거나 동일한 prompt에 답한다. RM은 "그럴듯해 보이는지"를 기준으로 ranking하고, 이는 memorized answer를 select한다.
3. ranked top-K가 next-round SFT data가 되고, 이제 memorized eval content가 *농축*된다.
4. DPO / RLVR preference data가 추가로 샘플링된 prompt에서 수집된다. preference label은 이제 "model이 memorized answer와 맞았는가"를 encode한다.
5. Eval harness는 강한 performance를 보고한다. 그 숫자의 일부는 reasoning이 아니라 recall이다.

[[olmo-3]]가 보여주는 방어적 posture: decontamination은 pretraining mix마다 한 번, *그리고* SFT mix마다 한 번, *그리고* preference mix마다 한 번 실행된다. [[olmo-2]]의 Dolmino cooldown(end-of-training 근처 50B tokens)은 가장 위험한 stage다. late-stage high-quality pool은 contamination exposure를 집중시킨다.

[[bespoke-stratos]]는 compact example이다. AIME은 public → R1이 memorizes → distilled traces가 inherit → seed prompt만 decontaminate하면 generated trace에 *embedded된* memorization을 놓친다. fix는 problem statement만이 아니라 *answer-and-solution-pattern*을 eval과 match해 filter하는 것이다.

**모든 stage boundary에서 실행할 세 check:**
- Pretraining → SFT: SFT mix를 eval n-grams에 대해 다시 hash한다.
- SFT → RM preference: preference prompt *and* chosen/rejected response를 다시 hash한다. gold answer를 verbatim으로 quote하는 response는 red flag다.
- RM preference → RL rollouts: rollout time에 model이 canary string 또는 verbatim eval passage를 emit하는지 check한다. [[faithful-synth-eval]] external-verifier machinery가 이 infrastructure를 제공한다.

---

## 6. Live-in-wild contamination

현대 eval team이 계획에 넣는 두 empirical pattern:

1. **Public benchmark는 7–14일 안에 Common Crawl로 leak된다.** MMLU question은 question-bank website, tutoring site, GitHub gist clone에 각 update 후 2주 안에 나타났다. benchmark release보다 최근 CC snapshot을 쓰는 pretraining run은 구조적으로 노출되어 있다.
2. **Distillation teacher는 간접적으로 leak된다.** frontier model이 public API traffic을 serve했고, 유료 사용자가 모든 AIME problem을 물어봤다면, 그 teacher는 사실상 eval에 노출된 것이다. [[bespoke-stratos]]는 이 risk를 명시한다.

방어 가능한 대응은 *date hygiene*이다. eval release보다 앞서는 "corpus freeze date"를 pin하고 freeze gap을 보고하라. gap이 음수(corpus가 eval보다 최신)라면 decontamination은 mandatory이며, memo는 n-gram match만으로 일부 contamination을 복구할 수 없다는 caveat를 달아야 한다.

---

## 7. Reporting conventions — 하나가 아니라 세 숫자

단일 percentage로 collapse되는 contamination report는 report가 아니다. 외부 scrutiny를 견디는 관례는 eval별 leakage number *세 개*를 나란히 보고한다.

1. **Verbatim / coarse-n-gram leakage %.** Per-instance: 어떤 training document든 coarse-n-gram overlap cutoff(예: 13-gram의 ≥80%)를 넘는 eval instance의 fraction. precision-floor number다. 잘 decontaminated된 corpus에서 기대값은 0.0%다. 양수가 나오면 받아들일 것이 아니라 고칠 bug다.
2. **Near-duplicate / MinHash leakage %.** Per-instance: 어떤 training document든 5-gram shingles에서 MinHash Jaccard ≥0.8인 eval instance의 fraction. moderate rewrites, section reorderings, translation-like edits를 잡는다. 기대값: decontamination pipeline이 제대로 configure되었다면 coarse-n-gram number에 가깝다. 훨씬 높으면 dedup parameter가 잘못되었다는 signal이다.
3. **Semantic-neighbour leakage %.** Per-instance: 어떤 training document든 sentence-embedding cosine ≥0.9(또는 domain-specific threshold)인 fraction. 이는 *known FP가 있는 estimate*다. manual-triage rate를 적용해 보고하라. 기대값은 topic-adjacent non-contamination을 포함하므로 다른 둘보다 높다.

세 숫자를 모두 보고하면 reader는 단일 knob을 믿는 대신 precision/recall frontier를 추론할 수 있다. coarse number만 보고하는 것은 흔한 "we looked clean" maneuver이며, 방어 가능한 convention은 이를 insufficient로 취급한다.

자주 추가되는 column: eval vs held-out matched-topic control의 **residual perplexity gap**. 위 세 숫자가 모두 0이어도 model이 equal difficulty control보다 eval에서 materially lower loss를 보고하면 memorization을 의심하라. 이것은 약한 signal이다(high noise, style confound). 그러나 memorization-through-paraphrase pathway를 probe하는 유일한 signal이다.

---

## 8. Defensible memo — template checklist

Contamination memo는 README가 아니라 formal artifact다. 각각 hard claim을 가진 여덟 sections:

- [ ] **Eval sets covered.** Name, version hash, release date. eval별 한 row.
- [ ] **Corpus stages scanned.** Pretraining / mid-training / cooldown / SFT / preference / rollouts — 각각 token count와 freeze date.
- [ ] **Detectors and operating points.** n-gram size(s), cutoff(s), MinHash params, embedding threshold, canary strings(hashed, not printed).
- [ ] **Per-eval leakage table.** 각 eval: {% instances with coarse-n-gram hit, % with MinHash-Jaccard hit, % with embedding-neighbour hit}. 세 숫자를 모두 보고하라. union도 intersection도 아니다.
- [ ] **Known recall gap.** 명시 문장: "이 workflow는 verbatim 및 near-verbatim leakage를 잡지만, cosine 0.9를 넘지 않는 paraphrased leakage는 잡지 못한다. held-out paraphrase probe 기준 estimated FN floor는 X%다."
- [ ] **What was removed.** stage별, eval별 document counts. token count delta. removal 후 re-hashing이 zero hits를 산출한다는 confirmation.
- [ ] **What the memo does NOT claim.** 열거하라: "adversarial contamination audit 없음," "semantic-distribution audit 없음," "distilled data에 대한 teacher-model memorization audit 없음."
- [ ] **Reproducibility.** Code commit SHA, hash-pinning artifact(sealed eval-n-gram Bloom filter), run-time and compute.

sections 4, 5, 7 중 하나라도 빠진 contamination memo는 defensible하지 않다. press release다.

**Review에서 명시적으로 지적할 anti-patterns.** 다른 사람의 memo를 읽고 있다면 다음을 flag하라.
- detector specification 없이 "We ran decontamination"이라고 말함. 어떤 detector? 어떤 cutoff? 어떤 stage에서?
- 단일 percentage "our leakage is X%." Per-eval, per-instance, or per-corpus? Verbatim or semantic?
- freeze-date disclosure 누락. corpus가 audit 없이 어떤 benchmark보다 최신이면 contamination이 null hypothesis다.
- decontamination을 pretraining mix에만 실행. [[llama-3]]와 [[olmo-3]] 모두 SFT / preference / rollout stage가 contamination을 다시 도입한다는 것을 분명히 한다. one-stage audit은 incomplete하다.
- classifier-driven quality filter를 decontamination으로 취급. Quality와 contamination은 직교한다. quality filter는 memorized correct answer를 *선택*한다.

---

## Companion visualization

**[figures/contamination-detect.html](figures/contamination-detect.html)** — 대화형 FP/FN explorer. Panel 1: n-gram size와 cutoff를 설정하고 synthesized train/eval corpus에서 FP/FN curve를 본다. n=8 vs n=13 vs n=50의 tradeoff frontier가 보인다. Panel 2: paraphrase-leakage simulation — slider가 rewrite ratio(eval-instance token 중 synonym으로 대체된 fraction)를 조절한다. paraphrase fraction이 올라가면 n-gram recall이 collapse하고, embedding-neighbour recall은 더 완만하게 degrade되는 것을 보라. memo를 draft하기 전에 사용하라. visualization은 "pick two detectors" 규칙을 손에 잡히게 만든다.

---

## Connections

- **ch-47 (Eval Harness Design)** — eval harness는 contamination damage가 나타나는 곳이다. 이 장의 memo는 harness가 숫자를 신뢰하기 전에 요구해야 할 것이다.
- **ch-49 (Judge Models)** — contaminated RM은 contaminated judge다. preference pool decontamination은 전제 조건이다.
- **[[deduplicating-training-data]]** — foundational MinHash / exact-substring primitives.
- **[[dolma]] / [[fineweb]]** — filter-cascade conventions; dedup 옆의 pinned-hash stage로서 decontamination.
- **[[llama-3]]** — downstream contamination을 구조적으로 만드는 RS → SFT → DPO loop.
- **[[olmo-2]] / [[olmo-3]]** — fully-documented open flow에서의 per-stage decontamination.
- **[[bespoke-stratos]]** — concrete teacher-memorization contamination pathway.
- **[[anthropic-sleeper-agents-data]]** — adversarial contamination threat model; paraphrase-evasion motivation.
- **[[faithful-synth-eval]]** — downstream-stage check에 재사용되는 external-verifier infrastructure.
- **[[scaling-laws-data-quality]]** — contaminated corpus가 실제보다 더 좋은 scaling curve 위에 있는 것처럼 보이는 이유.
