<!-- chapter: ch-12
     track: data
     title: Deduplication — Exact, Approximate, Semantic
     sources: [[deduplicating-training-data]], [[minhash-lsh]], [[d4]], [[ccnet]], [[dolma]], [[fineweb]], [[c4]]
     figures: figures/minhash-lsh.html
-->

# 12장 — Deduplication: Exact, Approximate, Semantic

> **핵심 통찰.** Pretraining corpus는 set이 아니다. *Near-copy의 길고 두꺼운 tail을 가진 multiset*이다. Lee 2021은 dedup을 하지 않으면 세 가지 비용을 동시에 낸다는 것을 보였다. Training time(같은 gradient를 다시 내려감), privacy(verbatim memorization이 duplication count에 거의 선형으로 scale), evaluation honesty("held-out" validation text의 >4%가 이미 train에 있음)다. 직관에 반하는 부분은 "모든 duplicate를 제거하라"가 적어도 세 가지 서로 다른 operation의 family라는 점이다. Suffix-array level의 exact-substring, document level의 MinHash+LSH near-duplicate, embedding-cluster semantic dedup이다. 이들은 서로 다른 것을 제거하고, 비용도 다르며, failure mode도 다르다. 잘못된 granularity를 고르는 것이 C4 mistake의 현대판이다.
>
> **가이드라인.** Dedup은 cascade order **URL → exact-substring → MinHash+LSH → (optional) semantic**으로 실행하라. 각 stage는 다음 stage가 효율적으로 다룰 수 없는 population을 제거한다. LSH의 `(r, b)`는 table이 아니라 target Jaccard에서 tune하라. S-curve $1 - (1 - J^{r})^{b}$는 magic number가 아니라 design knob이다. Semantic dedup은 near-duplicate filter가 아니라 **diversity lever**로 다루라. 그것은 distinct-but-topically-redundant document를 기꺼이 삭제하며, efficiency를 개선하는 동시에 domain coverage를 좁힐 수 있다.

---

## 이 장이 존재하는 이유

모든 web-data story의 첫 장은 dedup이다. Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini는 2021년에 foundational number를 제시했다([[deduplicating-training-data]]). **C4 token의 3.04%가 near-duplicate cluster 안에 있고, 하나의 61-word English sentence가 >60,000번 반복되며, un-deduplicated corpus로 학습한 model은 256-token unprompted completion의 ~1%에서 training-set text를 verbatim으로 emit한다.** Deduplication은 memorization을 10배 낮추고, 더 적은 step으로 같은 perplexity target에 도달하며, 보고된 benchmark win을 조용히 부풀리던 validation-set overlap >4%를 제거한다.

그 발견은 이후 모든 open corpus가 만들어지는 방식을 바꿨다. CCNet([[ccnet]]), Dolma([[dolma]]), FineWeb([[fineweb]])은 모두 dedup을 앞단에 배치한다. FineWeb은 첫 surprise를 추가했다. **Per-snapshot MinHash가 downstream task에서 global MinHash를 능가한다.** Global dedup은 같은 page의 "re-indexed"되었지만 실제로는 구별되는 version을 삭제할 만큼 공격적이기 때문이다.

Ch-11은 tokenizer-and-lineage layer를 만들었다. Shard, doc-id, content-addressed hashing이다. 이 장은 *그 어떤 것으로든 학습하기 전에 무엇을 제거할 것인가*에 대한 것이다.

---

## 1. Web corpus에서 실제로 무엇이 duplicated되는가

[[deduplicating-training-data]]에서:

| Duplicate type | Example | Who catches it | Who misses it |
|---|---|---|---|
| CC snapshot 간 동일 URL | 6개월 뒤 다시 crawl된 같은 page | URL Bloom-filter ([[dolma]]) | MinHash(이것도 찾지만 URL-hash가 100배 싸다) |
| Exact paragraph repeat | boilerplate footer, cookie banner | paragraph-hash dedup ([[dolma]]) | 높은 threshold의 doc-level MinHash |
| Document 간 긴 verbatim substring | 61-word sentence, MIT-license header, copy-pasted Wikipedia intro | **suffix-array ExactSubstr** | Duplicate가 doc의 작은 비율이면 doc-level MinHash |
| Near-duplicate document | 5% 수정된 article reprint, SEO doorway page | **MinHash + LSH** | exact method |
| Semantic near-copy | 다른 outlet이 다시 쓴 같은 news story | **embedding-space dedup (SemDeDup / D4)** | 위의 모든 것 |
| Topically redundant | 같은 event에 대한 news article 만 개 | embedding clustering + cluster-size cap | 위의 모든 것(low threshold의 SemDeDup 포함) |

각 row에는 자신의 algorithmic tool이 있다. 이 장이 도출할 세 horizontal cut, 즉 exact, approximate, semantic은 table의 세 tool family에 대응한다. 어느 하나도 지배적이지 않다. Production cascade는 셋 모두를 사용한다.

---

## 2. Lee 2021 — dedup이 optional이 아니라는 증거

논문은 **두 가지 complementary tool**을 제공한다([[deduplicating-training-data]]).

1. **ExactSubstr** — concatenated corpus 위의 suffix array. 길이 `>= 50 tokens`인 모든 duplicate substring을 찾고, 각각의 copy 하나를 제거한다.
2. **NearDup** — 5-gram shingle에 대한 MinHash + LSH. 다른 document와 Jaccard similarity `>= ~0.8`인 document를 drop한다.

Eval-contamination table은 논문의 가장 치명적인 결과다. 전부 인용할 가치가 있다.

| Corpus | % of validation that overlaps training (≥50 tokens) |
|---|---|
| LM1B | **4.6%** |
| C4 | **3.2%** |
| RealNews | **1.6%** |
| Wiki-40B | **0.6%** |

([[deduplicating-training-data]] §"Train-test contamination"에서.) 이것들은 연구자들이 수년 동안 perplexity improvement를 보고하던 validation set이었다. 손기술은 악의적이지 않았다. 아무도 LM1B를 만들 때 suffix-array contamination analysis를 예상하지 않았다. 하지만 보고된 숫자는 generalization이 아니라 memorization의 측정을 부분적으로 포함했다. 이 table 이전의 모든 paper는 조용히 rescale된다.

Memorization number도 마찬가지로 load-bearing이다.

> Dedup이 없으면 unprompted 256-token completion의 ~1%가 verbatim training copy다. Dedup하면 이것이 약 10배 감소한다.

Duplication count와 memorization rate는 대략 선형이다([[deduplicating-training-data]] Figure 4의 취지). `k`번 duplicated된 document는 `k`에 대략 비례하는 확률로 memorized된다. 이것이 membership inference와 training-data extraction의 privacy surface이며, dedup aggressiveness로 직접 제어할 수 있다.

Training-efficiency가 삼각형을 닫는다. Dedup은 C4 token의 ~5%를 제거하지만 **더 적은** step으로 target perplexity에 도달한다. 제거된 token이 쉬운 re-descended token이었기 때문이다. Pretraining loss는 raw token이 아니라 *unique* content에 대한 `1/N` process다.

---

## 3. MinHash — P(collision) = J(A, B) 도출

핵심 primitive다. 각 document `D`를 shingle set `S(D)`로 표현하자. Lee 2021에서는 whitespace-tokenized text의 5-gram이다. 두 document의 similarity는 **Jaccard index**다.

$$
J(A, B) \;=\; \frac{|S(A) \cap S(B)|}{|S(A) \cup S(B)|}.
$$

모든 pair에 대해 `J`를 naive하게 계산하면 $O(N^2)$ set-intersection operation이 든다. MinHash의 통찰(Broder 1997, [[minhash-lsh]])은 **document당 하나의 숫자**가 expectation상 충분하다는 것이다.

**Construction.** 가능한 shingle universe의 random permutation `π`를 고정한다. `π` 아래에서 `D`의 MinHash signature를 다음처럼 정의한다.

$$
h_{\pi}(D) \;=\; \min_{s \in S(A)} \pi(s).
$$

즉 `π`를 `D`의 모든 shingle에 적용하고 minimum을 취한다. 이것은 하나의 integer다.

**Theorem (Broder).** Uniform하게 random하게 뽑은 단일 permutation `π`에 대해,

$$
\Pr[\,h_{\pi}(A) = h_{\pi}(B)\,] \;=\; J(A, B).
$$

**Proof.** Set $U = S(A) \cup S(B)$를 보자. `π` 아래에서 `U`의 각 element는 global minimum value를 받을 확률이 동일하다. `s*`를 argmin이라 하자. $h_{\pi}(A) = h_{\pi}(B)$인 것은 오직 `s*`가 `S(A)`와 `S(B)` 둘 다에 있을 때, 즉 `S(A) ∩ S(B)`에 있을 때뿐이다. `U`의 uniform random element가 `S(A) ∩ S(B)`에 있을 확률은 정확히 $|S(A) \cap S(B)| / |S(A) \cup S(B)| = J(A, B)$다. ∎

따라서 하나의 permutation은 success probability가 `J`인 단일 Bernoulli trial을 준다. Additive error `ε`, confidence `1-δ` 안에서 `J`를 추정하려면 Hoeffding에 따라 $m \geq \tfrac{1}{2\varepsilon^2}\ln(2/\delta)$ independent permutation이 필요하다. `ε = 0.05, δ = 0.01`이면 `m ≈ 1060`이다. Lee 2021은 **9000** signature를 사용한다. Hoeffding이 요구하는 것보다 공격적으로 많다. 이들은 `J` distribution의 right tail(`J`를 uniformly estimate하는 것이 아니라 `J ≥ 0.8` pair를 찾는 것)에 관심이 있었기 때문이다.

실제로 astronomical shingle universe 위의 real permutation은 independent hash function $h_i(s) = (a_i s + b_i) \bmod p$로 대체된다. "`h_i` 아래에서 shingle의 min"은 computationally 동일하고 document당 $O(|S(D)|)$ hash를 사용한다. Modern implementation(datasketch, Spark)은 이를 그대로 제공한다.

---

## 4. LSH — r × b banding construction과 S-curve

Signature comparison도 naive하게 하면 여전히 $O(N^2)$다. Locality-sensitive hashing은 **similar signature를 같은 bucket에 넣어** 비용을 near-linear로 줄인다.

**Banding.** 길이 `m`의 MinHash signature를 각각 `r` row를 가진 `b` band로 나눈다. 따라서 $m = r \cdot b$다. 각 band에 대해 band의 `r` integer를 bucket으로 hash한다. 두 document가 *어떤* band에서든 bucket을 공유하면 **candidate**다.

Collision probability를 도출하자. Jaccard가 `J`인 두 document:

- Single row에서 agree할 확률: `J`(section 3).
- 한 band의 모든 `r` row에서 agree할 확률: $J^{r}$.
- 그 band에서 disagree할 확률(적어도 한 row가 다름): $1 - J^{r}$.
- 모든 band에서 disagree할 확률(independent hash가 주어졌을 때 band 간 독립): $(1 - J^{r})^{b}$.
- 적어도 한 band를 공유할 확률(LSH-candidate event):

$$
\boxed{\;P_{\mathrm{LSH}}(J; r, b) \;=\; 1 \;-\; (1 - J^{r})^{b}.\;}
$$

이것이 **S-curve**다. 세 regime이 있다.

| J | P_LSH (qualitative) |
|---|---|
| `J ≪ t` | near 0(random pair는 candidate가 아님) |
| `J ≈ t` | 0.5 근처를 가파르게 통과 |
| `J ≫ t` | near 1(true duplicate는 거의 모든 band에서 collide) |

**Threshold** `t`, 즉 `P_LSH(J) = 0.5`가 되는 `J`는 다음으로 잘 근사된다.

$$
t \;\approx\; \left( \frac{1}{b} \right)^{1/r}.
$$

Derivation: $P_{\mathrm{LSH}}(t) = 0.5$로 두면 $(1 - t^{r})^{b} = 0.5$이고, 따라서 $1 - t^r = 0.5^{1/b} \approx 1 - \frac{\ln 2}{b}$다. 그러므로 $t^r \approx \frac{\ln 2}{b}$이고 $t \approx (\ln 2 / b)^{1/r} \approx (1/b)^{1/r}$이다. `b ≥ 8`이면 작은 constant는 사라진다. Lee 2021의 `r = 450, b = 20`이면 $t \approx (1/20)^{1/450} \approx 0.9934^{\text{...no}}$다. 다시 계산하면 $\ln(1/20)/450 = -3.00/450 = -0.00666$이므로 $t \approx e^{-0.00666} \approx 0.9934$다. **이는 논문과 모순된다.**

[[deduplicating-training-data]]를 더 주의 깊게 읽어라. 논문은 **threshold ≈ 0.8**도 보고하고, **9000 signatures**, **b = 20 bands**, **r = 450 rows**도 보고한다. `(1/b)^{1/r}`을 final accept threshold로 다루면 이것들은 내부적으로 일관되지 않는다. 그 parameter에서 0.5-threshold는 `t ≈ 0.993` 근처에 있으므로 LSH만으로는 거의 모든 pair를 "accept"하지 않을 것이다. Resolution은 Lee et al.이 threshold가 아니라 *recall*을 위해 parameterize했다는 것이다. `r`과 `b`를 사용해 모든 `J ≥ 0.8`에서 `P_LSH ≈ 1`이 되게 한 뒤, 모든 candidate pair에 대해 true Jaccard를 명시적으로 계산하고 0.8 이상만 keep하는 post-filter를 실행한다. 대부분의 production implementation도 같다. LSH는 candidate generator이고, Jaccard check는 verifier다. S-curve의 0.5-threshold는 final accept/reject가 아니라 recall/cost tradeoff를 제어한다.

**[figures/minhash-lsh.html](figures/minhash-lsh.html)**에서 curve를 만져 보라. `r`과 `b`를 slide하면 S-curve가 움직인다. `J_target = 0.8`에 대해 typical production setting은 `r = 9, b = 20`이다(180 signatures, `t ≈ 0.687`, `P(J=0.8) ≈ 0.944`, false-positive `P(J=0.5) ≈ 0.038`). Lee 2021의 9000-signature choice는 paper의 conclusion을 논박 불가능하게 만들고자 한 recall-maxed outlier다.

---

## 5. ExactSubstr와 suffix array — span-level이 중요할 때

MinHash는 document-granularity tool이다. 다른 두 5000-token article 안에 embedded된 100-token verbatim passage는 document-level Jaccard ≈ 0.02를 갖는다. MinHash에는 보이지 않지만, memorization-driving duplicate다.

Suffix-array ExactSubstr([[deduplicating-training-data]])가 여기서 등장한다.

- 전체 corpus를 하나의 string `T`로 concatenate한다.
- $O(N \log N)$로 `T`의 suffix array `SA`를 만든다(SA-IS algorithm).
- `SA`를 walk한다. `SA`의 adjacent suffix들은 longest-common-prefix `lcp[i]`를 공유한다. `lcp[i] ≥ 50` token이면 duplicate substring을 mark한다.
- 각 duplicate span의 copy 하나를 제거한다.

**50-token threshold**는 empirical하다. 더 짧은 span에는 합법적으로 반복되는 common phrase("the quick brown fox," standard license boilerplate prefix)가 포함된다. 더 긴 span은 거의 항상 literal copy-paste다.

Span-level dedup은 특히 다음 경우 중요하다.

- Document가 **large boilerplate**(MIT license, Wikipedia의 Creative Commons footer, Stack Overflow의 "Thanks for contributing..." frame)를 포함한다.
- Document가 **quoted training example**(C4의 61-word sentence, 2024+ corpus의 quoted ChatGPT output)을 포함한다.
- Duplicate passage가 host document의 작은 비율이라 document-level near-dup가 실패했다.

Memory가 binding constraint다. 1T token에 대한 suffix array는 entry당 8 byte로 8 TB다. Multi-TB-RAM node에서는 가능하지만 commodity hardware에서는 불가능하다. Production pipeline(GoogleLM, OLMo)은 prefix별로 shard하고 shard별 suffix array를 실행한 뒤 second pass로 shard across를 reconcile한다. Dolma의 `paragraph-dedup` stage는 더 싼 approximation이다. Paragraph granularity에서 Bloom-filter exact-match를 수행해 memory의 1%로 ExactSubstr win의 80%를 잡는다.

---

## 6. Semantic dedup — SemDeDup과 D4

Exact와 approximate dedup은 surface-form tool이다. "Biden won the 2020 election"과 "The 2020 US presidential election was won by Joe Biden"가 disjoint 5-gram을 가진 같은 content라는 것을 볼 수 없다.

SemDeDup([[d4]], Abbas et al. 2023)은 embedding-space dedup으로 답한다.

1. Sentence-or-doc encoder(originally images에는 CLIP, text에는 OpenAI/BGE/E5)로 모든 document를 embed한다.
2. Embedding을 cluster한다(`k`-means, `k ≈ sqrt(N)`).
3. 각 cluster 안에서 cosine-similarity `τ`(typical `τ = 0.95`)를 넘는 모든 pair에 대해 하나만 keep하고 다른 하나를 drop한다.

논문의 headline은 **web-scale data의 20-50%를 downstream accuracy loss 없이 drop할 수 있고**, 때로는 OOD generalization이 개선된다는 것이다. 후속인 D4는 explicit diversity lever로 cluster-size capping을 추가한다. Oversized cluster는 within-cluster dedup pass 전에 down-sample되며, 이것은 "dedup"을 "coverage equalization"으로 바꾼다.

**실무자 trap 세 가지.**

**Embedding choice matters.** E5를 OpenAI-embedding-v3로 바꾸면 SemDeDup decision이 document의 ~10%에서 바뀐다. Embedding은 model의 similarity 개념을 encode한다. 그 개념은 자체적으로 같은 bias를 가진 pretraining data로 학습된다. FineWeb-Edu의 educational-value classifier([[fineweb]])는 인접 기술이다. 역시 embedding-driven이고 dataset-opinionated이며, 같은 주의가 적용된다.

**`τ`는 duplicate threshold가 아니라 diversity knob이다.** `τ = 0.95`에서는 near-copy를 제거한다. `τ = 0.80`에서는 topical-redundant(같은 game에 대한 두 news article)를 제거한다. `τ = 0.60`에서는 domain coverage를 제거한다. **Downstream eval sweep** 외에는 `τ`를 고르는 principled way가 없다. 올바른 답은 어떤 downstream capability가 중요한지에 달려 있기 때문이다.

**Aggressive semantic dedup은 domain coverage를 좁힌다.** 이것이 이 장의 sting이다. D4는 매우 aggressive한 clustering(high `k`, low `τ`)이 niche domain의 long tail, 즉 legal case law, minority-language sub-corpora, niche code repository를 제거하며, 그 domain들의 downstream metric이 급락한다는 것을 발견한다. FineWeb의 per-dump-rather-than-global MinHash([[fineweb]]) finding도 같은 모양이다. Global dedup은 너무 공격적이다. *Crawl artifact상 near-duplicate이지만 실제로는 구별되는 content*를 치기 때문이다.

Semantic dedup은 **efficiency**(더 작은 set으로 더 빠르게 학습)와 **coverage equalization**(English web news의 over-sampling 방지)을 위한 tool이다. MinHash와 같은 tool이 아니다. 생각 없이 쌓으면 corpus는 tail을 잃는다.

**[figures/minhash-lsh.html](figures/minhash-lsh.html)의 right panel**은 이 recall-vs-diversity tradeoff를 sliding knob으로 보여 준다. SemDeDup aggression을 올리면 genuine duplicate recall은 부드럽게 올라가지만, retained document의 cluster-ID distribution entropy로 측정한 domain-coverage는 `τ`가 ~0.85 아래로 떨어지면 *collapse*한다.

---

## 7. Production cascade와 FineWeb이 default를 깬 지점

**Canonical dedup order**, [[dolma]] §"Filter cascade"와 [[fineweb]] §"Pipeline"에서 합성:

1. CC snapshot 간 **URL-hash Bloom filter**. 가장 싼 stage. Exact re-crawl을 잡는다.
2. **Per-snapshot MinHash near-dup**(FineWeb의 핵심 발견: global이 아니라 per-snapshot).
3. **ExactSubstr** 또는 paragraph-hash Bloom(boilerplate, license text, quoted passage).
4. Language / quality / PII / content filter(10장, 13장, 14장에서 확장).
5. **Paragraph-level dedup last**([[dolma]]): 앞선 filter가 어떤 paragraph가 살아남는지를 바꾸므로 dedup은 surviving distribution 위에서만 의미가 있다.
6. **(Optional, post-quality)** diversity control을 위한 semantic dedup 또는 cluster-size capping.

**FineWeb의 surprise.** 96개 Common Crawl dump 전체에 대한 global MinHash는 downstream accuracy에서 per-dump MinHash보다 *성능이 낮았다*. 이유: 많은 진짜 high-quality page는 snapshot마다 한 번씩 crawl되어 snapshot across로 다시 나타난다. Global dedup은 이를 near-duplicate로 보고 하나만 keep해 effective signal을 반으로 줄인다. Per-dump dedup은 re-crawl을 보존한다. 그것들은 약간 다른 extraction point에서의 같은 high-quality page라는 free ensemble처럼 작동한다. 교훈: **aggression은 quality가 아니다.** Dedup은 coverage-shaping operation이다.

**Dolma의 paragraph-last rule**도 같은 flavor다. `dolma-ngram`은 paragraph를 n-gram으로 split하고 duplicated n-gram의 fraction이 `T = 1.0`(default, 즉 100% duplicated)을 넘으면 paragraph를 drop한다. 이를 quality filtering 전에 실행하면 subsequent language/quality filter가 어차피 제거했을 paragraph를 dedup하느라 compute를 쓴다. 더 나쁘게는 유용한 content의 유일한 remaining copy가 나중에 quality filtering에 실패할 document 안에 있는데, 그 paragraph의 *한* copy를 삭제해 corpus에 copy가 0개 남을 수 있다.

---

## 8. Dedup이 해로울 때

검증된 failure mode 세 가지.

**8.1 Tail의 over-dedup.** Code: `J = 0.7`에서 code corpus를 MinHash하면 수천 repository에 존재하는 near-duplicate function body를 삭제한다. 하지만 그 반복이야말로 *model이 standard idiom을 배우는 방식*이다. The Stack의([[dolma]]) code pipeline은 MinHash를 실행하되 canonical pattern을 보존하도록 threshold를 tune한다.

**8.2 Cross-domain collision.** Web, paper, code를 mix한 corpus 전체에 대해 global dedup을 하면 paper의 method section이 같은 paper를 요약한 blog-post summary와 near-duplicate로 mark될 수 있다. Blog post를 삭제하는 것은 옳다. Paper를 삭제하는 것은 아니다. Fix: **source across가 아니라 source within에서 dedup**하라. Dolma는 source별 dedup pass를 실행해 이를 채택한다.

**8.3 Semantic-dedup narrowing.** 이미 다뤘다. Signature shape: MMLU는 개선된다(benchmark-heavy center에 diversity-equalization이 도움이 되기 때문). Long-tail eval(BIG-Bench Hard subset, niche-language translation)은 떨어진다. Fix는 더 낮은 `τ` discipline이거나 삭제가 아니라 saturate하는 cluster-size cap이다.

통합 규칙: **dedup은 filter가 아니라 resampling operation이다.** 그것은 학습하는 empirical distribution을 바꾼다. 그 distribution에 대한 모든 변경은 downstream eval signal(ch-14, ch-47)이 정당화할 때만 받아들여야 한다.

---

## 연결과 다음 내용

- **[[deduplicating-training-data]] (Lee 2021)** — foundational eval-contamination and memorization evidence, §2의 spine.
- **[[minhash-lsh]] (Broder 1997)** — §3-§4의 `P = J` theorem과 `r × b` banding construction.
- **[[d4]] / SemDeDup** — §6의 embedding-space dedup과 diversity-narrowing tradeoff.
- **[[ccnet]] / [[dolma]] / [[fineweb]]** — dedup을 더 넓은 filter cascade에 배선하는 세 production pipeline. §7은 이들의 config에서 구성되었다.
- **[[c4]]** — Lee 2021이 처음 폭로한 uncleaned baseline.
- **ch-11 (tokenizer and lineage)** — dedup이 작동하는 doc-id와 content-addressed hashing을 제공한다.
- **ch-13 (domain mixing, DoReMi)** — deduplicated shard를 받아 그 위에서 mix weight를 학습한다.
- **ch-14 (scaling, contamination, retention)** — dedup aggression을 정당화하는 downstream eval signal.

## 더 읽을거리

- [[deduplicating-training-data]] — Lee et al. 2021 / 2022. §3(ExactSubstr), §4(NearDup), §5(memorization), §6(contamination table)를 읽어라.
- [[minhash-lsh]] — Broder 1997 and Indyk-Motwani 1998. *Mining of Massive Datasets*(Leskovec-Rajaraman-Ullman)의 3장은 canonical pedagogical treatment다.
- [[d4]] — Abbas et al. 2023. SemDeDup algorithmic detail과 web-scale ablation.
- [[fineweb]] — Penedo et al. 2024. Per-dump-vs-global MinHash ablation, default를 깬 surprise.
- [[dolma]] — Soldaini et al. 2024. Six-stage filter cascade, paragraph-dedup ordering argument.
- [[ccnet]] — Wenzek et al. 2019. Pre-Lee template.

## 동반 시각화

**[figures/minhash-lsh.html](figures/minhash-lsh.html)** — 두 개의 interactive panel. **Left panel:** Jaccard threshold slider를 drag하고 `r`(rows per band)과 `b`(bands)를 조정하라. S-curve `P(collision) = 1 - (1 - J^r)^b`가 live로 다시 그려지고, 작은 table은 0.5-threshold `t ≈ (1/b)^{1/r}`, total signatures `m = r·b`, 선택한 target Jaccard에서의 false-positive / false-negative rate를 보고한다. **Right panel:** SemDeDup knob. Cosine-threshold `τ`를 0.5에서 1.0까지 sweep하면 두 curve가 반대 방향으로 움직인다. True near-duplicate recall은 aggression과 함께 상승하고, domain-coverage entropy는 하락한다. §4 banding math와 §6 diversity tradeoff를 한 호흡으로 직관화하는 데 사용하라.
