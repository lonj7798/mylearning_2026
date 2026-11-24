<!-- chapter: ch-11
     track: pretraining
     kind: content
     title: 토크나이저, 데이터 출처, PII 제거
     deps: [ch-10a]
     sources: [[vocabulary-scaling-laws]], [[tokenizer-language-unfairness]], [[tokenization-counts-arithmetic]], [[under-trained-tokens]], [[tokenizer-domain-adaptation]], [[new-embedding-initialization]], [[quantifying-memorization]], [[extracting-training-data]], [[wimbd]], [[infini-gram]], [[olmotrace]], [[olmo-3]], [[olmo-core-olmo3-configs]], [[dolma]], [[fineweb-2]], [[llama-3]], [[gemma-2]], [[deepseek-v3]], [[mistral-nemo]], [[deduplicating-training-data]], [[atlas-multilingual-scaling-laws]]
     revised: 2026-09 (generality revision)
-->

# 11장 — 토크나이저, 데이터 출처, PII 제거

> **핵심 통찰.** 토크나이저는 학습이 시작되기 전에 각 언어, 숫자, code file에 드는 token 수를 고정하며, 이 비용은 균등하지 않다. cl100k_base 토크나이저는 같은 Italian text에 English의 1.64배, Standard Arabic에는 3.04배, Shan에는 15.05배 token을 필요로 한다([[tokenizer-language-unfairness]] Table 1). Vocabulary size는 compute-allocation 선택이다. Non-vocabulary parameter 2.87B와 2.3e21 FLOPs에서 43K vocabulary는 32K에 비해 ARC-Challenge를 29.1에서 32.0으로 높였다([[vocabulary-scaling-laws]] Table 3). 반면 vocabulary에 존재하지만 학습 중 드물게 본 token은 학습되지 않은 채 남는다([[under-trained-tokens]] §5가 시험한 model에서 보통 vocabulary의 0.1–1%, Qwen1.5 32B에서는 1.6%, Table 1). Training text의 verbatim memorization은 duplication, model size, prompt length와 함께 증가하고([[quantifying-memorization]] §4), GPT-2에서 개별 전화번호와 주소가 추출됐으므로([[extracting-training-data]] Fig. 1), PII 제거와 training-data search tool은 학습 전에 실행되는 pre-training pipeline의 stage다.
>
> **지침.** Model이 언어, code, arithmetic 전반에서 작동해야 한다면 model과 같은 mixture로 토크나이저를 학습하고, pre-training 전에 각 target language의 token premium과 digit segmentation을 측정하며, 계획한 compute에 맞춰 vocabulary 크기를 정하라. English-centric tokenizer에는 10을 넘는 premium이 남고, 고정 FLOPs에서도 vocabulary size가 downstream accuracy를 바꿨기 때문이다([[tokenizer-language-unfairness]]; [[vocabulary-scaling-laws]] Table 2–3). Pre-training 뒤 token을 추가해야 하면 기존 embedding의 평균으로 초기화하고, 새 row가 학습될 만큼 충분한 data로 학습하거나([[tokenizer-domain-adaptation]] Fig. 6에서 약 50B token), base model로 평가하는 stage에서는 새 special token을 제외하라. OLMo 3 mid-training data의 학습되지 않은 special token은 base model이 그것을 출력하게 해 GSM8K를 49.43에서 0으로 낮췄다([[olmo-3]] §3.5.4). Corpus에 personal data가 있을 수 있다면 학습과 search index 구축 전에 이를 detect해 mask 또는 제거하고, manual sample에서 detector precision을 추정하며, 최종 training data의 search index를 유지하라. Training document 하나에만 나타나는 sequence도 extraction이 성공했으며([[extracting-training-data]] Abstract), memorization audit에는 training data를 prompt로 제공해야 하기 때문이다([[quantifying-memorization]] §4.3). 그렇지 않으면, 새 data에서 premium과 under-trained token을 측정한 뒤에만 토크나이저를 재사용하라.

## 이 장이 일반 목적 model에 중요한 이유

일반 목적 model의 pipeline은 pre-training → mid-training → SFT → preference optimization → RL → evaluation이다. ch-10과 ch-10a는 어떤 document가 pre-training pool에 들어갈지를 결정했다. 이 장은 학습 전에 그 pool에 대해 내리는 세 가지 결정과, pool이 소비되는 방식을 다루는 reference section 하나를 다룬다.

1. **Tokenization.** 토크나이저는 text를 integer id로 mapping한다. Vocabulary 및 segmentation rule은 content 단위당 token 수를 정하며, 이는 document당 training compute, context window에 들어가는 content 양, 숫자와 rare word가 model에 제시되는 방식을 정한다(§1–§5).
2. **Provenance.** 여기서 provenance는 임의의 string에 대해 어떤 training document가 이를 포함하는지 답할 수 있는 능력이다. 이것이 없으면 올바른 답을 memorized answer와 구분할 수 없다(§6).
3. **PII 제거.** Personally identifiable information(PII)은 contact detail을 동반한 이름, 전화번호, email address처럼 사람을 식별할 수 있는 정보다. Memorization은 측정 가능하고 extraction도 작동하므로, PII는 pipeline stage로 제거하거나 mask한다(§7).
4. **Mixture 소비.** 설정한 mixture weight는 restart 후에도 state가 유지되어야 하는 sampler를 통해 token share가 된다(§8, reference only).

다음 장(ch-12)은 duplicate를 제거한다. 이는 이 장과 두 방식으로 상호작용한다. Training sequence의 memorization은 duplicate 수와 함께 증가한다([[quantifying-memorization]] §4.2). 이 연구는 generic training sequence를 측정했으며, 반복 PII string에도 같은 의존성이 있을 것으로 예상한다(Interpretation). 일부 deduplication 및 search tool은 해당 tool에 맞춰 선택한 토크나이저를 사용한다. OLMo 3는 training tokenizer 대신 p50k 토크나이저로 MinHash 5-gram set을 만들고([[olmo-3]] App. A.2.2), OLMoTrace는 Llama-2 토크나이저로 training data를 index한다([[olmotrace]] §3).

## §1 토크나이저 학습: BPE와 Unigram이 만드는 것

**정의.** Byte-pair encoding(BPE)은 single character 또는 byte에서 시작해 가장 빈도가 높은 인접 pair를 새 token으로 반복 추가하여 vocabulary를 만든다. 새 text의 segmentation은 학습된 merge를 학습된 순서로 적용한다([[tokenization-counts-arithmetic]] §1). Unigram tokenization은 큰 candidate vocabulary에서 시작해 unigram model 아래 corpus likelihood를 가장 적게 낮추는 piece를 제거하고 text를 가장 확률이 높은 piece로 segment한다. [[under-trained-tokens]] §1은 BPE의 주된 대안인 이를, “BPE보다 뛰어난다고 제시하는 연구가 있음에도” 흔히 사용되지 않는 방식으로 설명한다. Pre-tokenization은 BPE 실행 전에 text를 나누는 regular expression이므로, token은 split point를 넘을 수 없다([[tokenizer-domain-adaptation]] §2.4).

**문제.** 같은 text도 서로 다른 토크나이저에서 서로 다른 token 수를 받는다. Dagan 등은 InCoder 토크나이저가 Llama 토크나이저보다 code를 26% 적은 token으로 encode(normalized sequence length 0.74)하는 반면, GPT-2 토크나이저는 19% 더 필요하다(1.19)고 보고한다([[tokenizer-domain-adaptation]] Table 1).

**메커니즘(BPE 학습).**
1. Pre-tokenization rule로 training text를 나눈다.
2. 모든 character(또는 256개 byte value)로 vocabulary를 초기화한다.
3. Word frequency로 가중한 모든 인접 token pair를 count한다.
4. 가장 빈도가 높은 pair를 새 token으로 추가하고 그 occurrence를 치환한다.
5. Vocabulary가 target size V에 이를 때까지 3–4단계를 반복한다.

**계산 예(toy corpus, 이 course의 illustration).** Word count는 hug 10, pug 5, pun 12, bun 4, hugs 5다. 최초 pair count는 u+g = 10 + 5 + 5 = 20, p+u = 17, u+n = 16, h+u = 15, g+s = 5, b+u = 4다. Merge 1은 `ug`를 추가한다. 다시 count하면 u+n = 16, h+ug = 15, p+u = 12다. Merge 2는 `un`을 추가한다. 다시 count하면 h+ug = 15, p+un = 12다. Merge 3은 `hug`를 추가한다. 이제 vocabulary는 {b, g, h, n, p, s, u, ug, un, hug}다. 보지 못한 word “bugs”를 segment하면 merge 1을 적용해 [b, ug, s], 즉 3 token이 된다. “thug”를 segment하려면 vocabulary에 없는 `t` token이 필요하다. Character-level tokenizer는 unknown token을 출력해야 하는 반면 byte-level tokenizer는 `t`의 byte로 fallback한다. Byte-level representation은 “학습 중 보지 못했어도 모든 Unicode codepoint를 encode한다”([[tokenizer-language-unfairness]] §4.4).

**근거: release된 토크나이저의 구성.**

| Model | 토크나이저 구성 | Vocabulary | Locus |
|---|---|---|---|
| Llama 3 (8B/70B/405B) | tiktoken의 100K token에 non-English token 28K 추가 | 128,000 | [[llama-3]] §3.2, Table 3 |
| OLMo 3 (7B/32B) | OLMo 2와 같고 cl100k에서 파생 | 100,278, 100,352까지 padding | [[olmo-3]] §3.2; [[olmo-core-olmo3-configs]] tokenizer.py L85–94 |
| Gemma 2 | split digit, preserved whitespace, byte-level encoding을 둔 SentencePiece | 256,128 | [[gemma-2]] §3.1, Table 1 |
| DeepSeek-V3 | Byte-level BPE; punctuation과 line break를 결합한 token을 학습 중 random split | 128K | [[deepseek-v3]] §4.1 |
| Mistral NeMo | 100개 초과 언어로 학습한 tiktoken-based Tekken | 131,072 | [[mistral-nemo]] “Tekken” section, config.json |
| GPT-2 XL, Llama 2 7B | under-trained token 연구에서 보고 | 50,257; 32,000 | [[under-trained-tokens]] Table 1 |

이 report에는 두 pattern이 나타난다. Byte-level encoding을 갖는 SentencePiece vocabulary(Gemma 2), 그리고 기존 BPE vocabulary에서 만든 vocabulary(Llama 3는 tiktoken의 100K에 token 28K를 추가하고 OLMo 3는 cl100k-derived vocabulary를 사용한다)다. Llama 3는 Llama 2 대비 자신의 토크나이저가 English compression을 token당 3.17 character에서 3.94로 높였고, non-English token 28K가 “English tokenization에 영향 없이 compression ratio와 downstream performance 모두를 개선했다”고 보고한다. Downstream 수치는 보고하지 않는다([[llama-3]] §3.2) (**Result, 단일 연구**).

**Padding.** OLMo-core는 throughput을 위해 embedding matrix를 128의 배수로 round한다. `pad_multiple * ((self.vocab_size + pad_multiple - 1) // pad_multiple)`([[olmo-core-olmo3-configs]] tokenizer.py L77–82). vocab_size 100,278에서는 (100,278 + 127) // 128 = 784이고, 784 × 128 = 100,352다. 추가 74개 row는 토크나이저가 만들어 내는 token id에 절대 대응하지 않으므로 input이나 target으로 나타나지 않는다. Output row로서는 §4에서 설명하는 push-down gradient를 여전히 받는다. [[under-trained-tokens]] §2.2는 “tokenizer vocabulary size 위” row를 untrained embedding의 reference set으로 사용한다.

**조건과 한계.** Pre-tokenization은 compression과 accuracy 모두에 영향을 준다. “Identity” pre-tokenizer는 Llama보다 code를 30% 더 잘 compress했으나 1.5B에서 code-generation result는 “significantly worse”였다([[tokenizer-domain-adaptation]] §3) (**Result, 단일 연구**). Tokenizer training data도 중요하다. More code로 학습하면 code compression이 개선됐고 multilingual data로 학습하면 multilingual compression이 개선됐다([[tokenizer-domain-adaptation]] §2.3, Fig. 2).

**함의.** 일반 목적 토크나이저는 model과 같은 language, code, format mixture에서 학습한다. 토크나이저의 merge statistic이 이들 중 무엇에 짧은 encoding을 줄지 정하기 때문이다.

## §2 Vocabulary size와 compute: vocabulary scaling law

**정의.** Vocabulary parameter는 V와 함께 count가 증가하는 parameter다. Output layer만 세면 `N_v = V·d`다([[vocabulary-scaling-laws]] §2.2, footnote 1). Non-vocabulary parameter `N_nv`는 나머지 모든 parameter다.

**문제.** 더 큰 vocabulary는 같은 character를 더 적은 token으로 encode하므로 FLOP당 더 많은 text를 처리하지만, 큰 vocabulary의 rare token은 update를 적게 받는다. 측정 가능한 질문은 고정 compute budget C에서 어느 V가 loss를 최소화하는가다([[vocabulary-scaling-laws]] §1).

**메커니즘.**
1. Token count가 V에 의존하므로 data를 character H로 측정한다.
2. Fitting한 compression function `D = H·f(V)`로 변환한다.
3. V에 의존하지 않는 loss(unigram-normalized loss `L_u`)로 model을 비교한다.
4. 각 FLOP budget에 대해 가장 낮은 `L_u`를 내는 V를 찾고, 이어 budget 전반에 power law를 fit한다.

**수식**([[vocabulary-scaling-laws]] Eq. 3–5):

```
f(V) = a·(ln V)² + b·ln V + c,        a = 0.0064, b = −0.1581, c = 1.2047
C ≈ 6·(N_nv + V·d)·H·f(V)
L_u = −(1/T)·Σ_i log[ p(w_i | w_<i, V) / p(w_i | V) ]
N_v,opt ∝ N_nv^γ,                     γ ≈ 0.83 (Approach 2); 0.42/0.50 = 0.84 (Approach 1)
```

`f(V)`는 character당 token 수이고, `d`는 embedding dimension, `T`는 sequence length, `p(w_i | V)`는 tokenized corpus에서 token `w_i`의 unigram frequency다. Natural logarithm을 쓰면 논문의 table을 재현한다(아래에서 확인).

**계산 예(Table 3, overtraining budget).** `N_nv = 2.87B`, `d = 3200`, budget은 `2.3e21 FLOPs`다.
- `V = 32K`: `f = 0.0064·(10.373)² − 0.1581·10.373 + 1.2047 = 0.2534` token per character다. Table 3은 `D = 128.5B`, `H = 509.1B`를 출력하므로 `D/H = 0.2524`다. `N_v = 32,000 × 3,200 = 0.1024B`다. `C = 6 × (2.87B + 0.1024B) × 128.5B = 2.29e21`이다.
- `V = 43K`: `f = 0.2464`다. Table 3은 `D = 127.0B`, `H = 517.5B`, `D/H = 0.2454`를 출력한다. `N_v = 0.1376B`다. `C = 6 × 3.0076B × 127.0B = 2.29e21`이다.
- 두 run의 FLOP 비용은 같다. 43K model은 token을 1.5B 적게 보지만 character를 8.4B 더 본다.

[figures/pipeline-ops.html](figures/pipeline-ops.html)의 panel 1은 reader가 V, d, `N_nv`, H를 바꾸고 이 수식으로 다시 계산한 `f(V)`, D, `N_v`, C를 보게 한다. 이 figure는 출력된 D가 아니라 `f(V)`에서 D를 계산하므로 두 run 모두 `2.30e21`을 보인다.

**근거.** Non-vocabulary parameter 2.87B에서([[vocabulary-scaling-laws]] Table 2–3):

| Budget (FLOPs) | V | ARC-C | HellaSwag | BoolQ | 7개 평균 |
|---|---|---|---|---|---|
| 2.8e20 (undertrained) | 32K → 24K | 23.6 → 24.2 | 34.4 → 36.0 | 59.8 → 61.5 | 43.2 → 43.9 |
| 1.2e21 (compute-optimal) | 32K → 35K | 28.5 → 29.1 | 47.5 → 48.1 | 56.4 → 57.1 | 47.9 → 48.5 |
| 2.3e21 (overtrained) | 32K → 43K | 29.1 → 32.0 | 53.0 → 54.1 | 59.5 → 61.9 | 50.3 → 51.6 |

예측한 optimum은 scale과 함께 증가한다. `N_nv = 3B`에서 43K(Approach 2), 7B에서 67K, 70B에서 231K다(Table 1). Abstract는 Llama2-70B의 optimum이 자신의 32K에 비해 적어도 216K라고 말한다. `N_nv = 302M`에서 data가 부족하면 optimum은 16K에서 10K로 옮겨갔고, excess data가 있으면 24K로 옮겨갔다(§5, Fig. 7). 논문은 task별 standard deviation 0.5–2.1점을 출력하지만 training seed 수는 말하지 않는다(**Result, 단일 연구**).

**조건과 한계.**
- Model은 3B까지의 dense Transformer, data는 English SlimPajama다. 저자들은 언어가 capacity를 두고 경쟁하기 때문에 multilingual data를 future work로 열거한다(App. B.2, B.4).
- Quadratic `f(V)`는 `ln V = −b/(2a)`에서 최소가 되고 `V ≈ 231K`(도출)이며 그 size 위에서는 다시 증가한다. Real BPE vocabulary는 자신의 training distribution에서 그렇게 compression하지 않을 수 없다. V가 약 200K보다 크면 이 fit은 compression을 기술하지 않으므로, 그 size에서 Table 1 예측(예: 70B에서 231K)은 `f(V)`의 extrapolation에 근거한다(도출).
- 다른 setting에서, general data로 pre-train한 1.5B model(NL 1.5B)의 토크나이저를 code fine-tuning 전에 32k, 64k, 128k, 256k GPT-4-style vocabulary로 바꾸면 HumanEval Pass@1은 20.5, 20.6, 20.8, 20.5%였다(Pearson `r = −0.13`, `p = 0.87`)([[tokenizer-domain-adaptation]] Table 3). 위 고정-FLOPs vocabulary 효과는 task별 −0.4에서 +2.9점, 평균 +0.6에서 +1.3점이며 task별 standard deviation 0.5–2.1점에 비해 그렇고, 3B를 넘어 시험하지 않았다.
- Multilingual vocabulary에서는 ATLAS가 multilingual vocabulary가 compute-optimal frontier를 monolingual vocabulary보다 위로 옮긴다고 보고하며, English에서 그 정도가 가장 크다([[atlas-multilingual-scaling-laws]] §3, Fig. 1). ch-13a가 이를 다룬다.

**함의.** Vocabulary size는 작은 model에서 복사하는 것이 아니라 model size와 token budget으로 정하며, 평가는 언어별 및 code compression(§3)을 포함한다.

## §3 Breadth에 대한 토크나이저 효과: 언어 premium과 digit segmentation

### 언어 premium

**정의.** 언어 B에 대한 언어 A의 tokenization premium은 토크나이저 t 아래 sentence `s_A`와 그 번역 `s_B`의 `|t(s_A)| / |t(s_B)|`다([[tokenizer-language-unfairness]] §3). Fertility는 관련된 측정치로, word당 token 수다.

**문제.** Premium이 1보다 크면 같은 content가 학습에 더 많은 token, inference에 더 많은 처리 token을 요구하고 context window를 더 빨리 채운다.

**근거.** FLORES-200(200개 언어의 sentence 2,000개)에서([[tokenizer-language-unfairness]] Table 1, Table 4–5):

| 언어 | GPT-2 / RoBERTa | cl100k_base (ChatGPT, GPT-4) |
|---|---|---|
| Portuguese | 1.94 | 1.48 |
| Italian | 2.01 | 1.64 |
| Bulgarian | 5.51 | 2.64 |
| Standard Arabic | 4.40 | 3.04 |
| Burmese | 16.89 | 11.70 |
| Shan | 18.76 | 15.05 |

Multilingual use를 위해 만든 tokenizer(XLM-R, NLLB, mT5, M2M100, BLOOM)는 parity에 더 가깝지만, “다섯 model 모두 premium이 2.5를 넘는 언어를 갖는다”(§4.3). Byte-level encoding도 premium을 없애지 못한다. ByT5의 UTF-8 premium은 0.87에서 3.94이며, ASCII character는 UTF-8에서 1 byte, Cyrillic·Greek·Arabic character는 2 byte, CJK character와 Shan consonant·diacritic은 3 byte를 차지하기 때문이다(§4.4). RoBERTa의 processing time은 tokenized length에 선형으로 증가하고 Shan은 English의 거의 두 배 시간이 걸린다(§5.2, Fig. 2). FineWeb2는 multilingual ablation에 subword fertility로 Gemma tokenizer를 골랐다. 평균은 word당 2.10 token으로 Bloom의 2.16에 비해 낮았고, mT5의 2.03 및 XGLM의 1.76은 더 낮았지만 일부 character를 encode할 수 없다. 또한 Mistral-v3, Phi3, Llama3, Command-R tokenizer에서 Telugu fertility가 word당 9.74–10.11 token이라고 보고한다([[fineweb-2]] §3.1, Table 2) (**큰 disparity 존재에 대해서는 [[tokenizer-language-unfairness]]와 [[fineweb-2]] 전반에서 Replicated**).

**계산 예.** cl100k_base에서 English로 8,192-token context를 채우는 document를 생각하자. 같은 content는 Standard Arabic에서 `8,192 × 3.04 = 24,904` token이 필요하므로, 8,192-token window에는 그 `1/3.04 = 32.9%`만 들어간다. Shan에서는 `8,192 × 15.05 = 123,290` token이 필요하고 window에는 6.6%가 들어간다. Model size가 고정되면 token당 training compute는 일정(`C ≈ 6ND`)하므로, 같은 content는 Standard Arabic에서 training compute가 3.04배 든다(도출). [figures/pipeline-ops.html](figures/pipeline-ops.html)의 panel 2는 다른 premium과 window size에 대해 이 계산을 반복한다.

**조건과 한계.** Premium은 English-centric named entity가 있는 parallel corpus 하나에서 측정됐고 논문은 downstream accuracy를 보고하지 않는다([[tokenizer-language-unfairness]] §6). English에서 vocabulary를 제거해도 compression 비용은 작다. cl100k_base vocabulary의 3분의 1만으로 English FLORES-200은 약 10% 길어진다(§6, Fig. 3). Korean과 Arabic에서 Tekken이 “2x and 3x more efficient”하다는 vendor claim은 측정 corpus가 명시되지 않은 official blog에서 왔으며([[mistral-nemo]] “Tekken” section), controlled comparison이 아니다.

### Digit segmentation

**정의.** Digit segmentation은 숫자를 token으로 나누는 rule이다. Pure BPE(학습된 merge가 무엇이든), single digit, fixed chunk가 있다.

**문제.** Arithmetic accuracy는 token boundary가 place value와 정렬되는지에 의존한다.

**메커니즘.** GPT-3의 p50k_base는 일부 3-digit string에는 single token을 두고 다른 것에는 두지 않는다(논문의 예: 710에는 자체 token이 있을 수 있지만 711에는 없을 수 있다). 따라서 같은 길이 숫자도 서로 다른 segmentation을 받는다([[tokenization-counts-arithmetic]] §1, Fig. 2). cl100k_base는 모든 1-, 2-, 3-digit string에 token이 있으며 긴 숫자를 3개씩 left-to-right(L2R) chunk로 나눈다. PaLM, Llama 1 및 2, Mistral은 single digit를 사용한다. OLMo(2024)는 pure BPE를 사용했다(Table 1).

**계산 예(논문 Fig. 1의 문제).** `8302080 + 3529456 = 11831536`이다. L2R chunk는 `[830][208][0] + [352][945][6] = [118][315][36]`이다. Comma가 강제하는 right-to-left(R2L) chunk는 `[8][302][080] + [3][529][456] = [11][831][536]`이다. R2L 형식에서는 answer의 각 chunk가 addend의 대응 chunk와 같은 place value를 포괄한다. L2R 형식에서는 answer는 8 digit이고 addend는 7 digit라 answer의 모든 chunk boundary가 addend에 대해 shift된다.

**근거.** 7–9 digit addend를 가진 addition problem 90개, 8-shot, greedy에서 GPT-3.5는 L2R 75.6%, R2L 97.8%였고 GPT-4는 84.4% 대 98.9%였다([[tokenization-counts-arithmetic]] Fig. 1). Answer가 addend보다 길면 L2R accuracy는 8.25%까지 떨어지고 error는 네 번째 digit에 나타난다(§4.1, §4.3). Comma를 다른 single-token separator로 바꿔도 비슷한 이득이 있었으며, direction을 바꾸지 않고 token을 추가하면 그렇지 않았다(§3.2–3.3) (**Result, 단일 연구**; tokenization은 pre-training이 아니라 inference에서 바꿨다).

**조건과 한계.** 이 연구는 closed model과 inference-time formatting을 사용한다. OLMo-core는 “The R2L dolma2 tokenizer”라고 문서화한 `dolma2_sigdig` tokenizer를 포함한다([[olmo-core-olmo3-configs]] tokenizer.py L30–32). 이 장의 source는 그것의 evaluation을 보고하지 않는다. Gemma 2는 digit를 split한다([[gemma-2]] §3.1). Report된 ablation은 없다.

**함의.** Number handling은 tokenizer design 시점에 고정된다. 일반 model의 tokenizer 평가는 언어 premium과 함께 digit-segmentation check를 포함한다.

## §4 Under-trained token

**정의.** Under-trained token은 model training data에 거의 없었거나 전혀 없었던 vocabulary entry다([[under-trained-tokens]] §1). Unreachable token은 decoded string을 다시 tokenize했을 때 절대 만들어지지 않는 entry다(§2.1).

**문제.** 이 token은 vocabulary slot을 차지하고 input에 나타날 때 garbled output 또는 hallucinated output을 낼 수 있다. 저자는 external data를 처리하는 agent의 위험을 지적한다(§1).

**메커니즘.** Logit `z_j`에 대한 softmax log-likelihood gradient는 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. Target y가 된 적이 없는 token은 매 step에서 negative term `−p_j`만 받고 output row `e_j`는 각 hidden state h에 대해 `−p_j·h` 방향으로 움직인다. Never-seen row는 모두 같은 종류의 update를 받으므로 row가 방향을 공유한다(저자의 설명). [[under-trained-tokens]] §2.2는 unseen output row가 mean output vector에서 “moving away”하여 model이 “never the correct prediction”인 token에 “highly negative logits”을 할당할 수 있게 된다고 설명한다. Input embedding row는 자기 token이 input에 나타날 때만 gradient를 받는다. 그렇지 않으면 weight decay(0 방향)로만 바뀌거나 initialization에 머문다(§2.2).

**계산 예.** Target token 0인 logit `z = [2, 1, 0, −1]`은 `p = [0.644, 0.237, 0.087, 0.032]`를 준다. `log p_0`의 gradient는 `[+0.356, −0.237, −0.087, −0.032]`다. Token 3은 이미 가장 낮은 likelihood였지만 그 logit은 계속 push-down된다. Constant learning rate `3e-4`에서 decoupled weight decay `λ = 0.1`이면, unseen input row는 step마다 `(1 − 3e-5)`를 곱하고 100,000 step 뒤 constant-LR 가정 아래 초기 norm의 `(1 − 3e-5)^100,000 = 5.0%`를 유지한다. OLMo 3 7B script는 `embeddings.weight`에 `weight_decay=0.0`을 설정하므로([[olmo-core-olmo3-configs]] pretrain-1.py), 이 configuration에서 unseen input row는 초기값을 유지한다.

**Indicator와 verification**([[under-trained-tokens]] §2.2–2.4):

```
u_ref = (1/|t_ref|) · Σ_{i ∈ t_ref} E_out,i
C(A, x)_i = 1 − (A_i · x) / (‖A_i‖ ‖x‖)      (tied embeddings: rank tokens by C(E_out, u_ref))
L2(E_in)_i = ‖E_in,i‖                         (untied embeddings: rank tokens by input-row norm)
```

`t_ref`는 known unused token set, `E_out`과 `E_in`은 output 및 input embedding matrix, `A_i`는 i번째 row다. Candidate 상위 2%를 해당 token을 반복하도록 prompt한다. Maximal output probability가 1%보다 낮으면 token을 confirmed한다.

**근거**([[under-trained-tokens]] Table 1, confirmed/tested): GPT-NeoX 20B 10/993; OLMo v1.7 7B 178/993; Llama3 8B 556/2540; Qwen1.5 32B 2450/2966; Gemma 2B 3161/5117; Command R 306/5012다. 저자는 confirmed token을 “typically around 0.1–1% of the vocabulary”로 요약하며 prevalence는 “varies significantly”한다고 말한다(§5). OLMo v1.7 7B에서 indicator는 10 orders of magnitude 전반의 first-epoch token count와 상관한다(Fig. 2). Tokenizer를 자신의 training corpus로 학습한 GPT-NeoX 20B는 10개 confirmed token을 가지지만, 다른 data에서 GPT-2 tokenizer를 재사용한 GPT-J 6B와 Phi-2는 200개와 103개다(GPT-2 XL 자체는 67개). Rakuten 7B의 extended Japanese vocabulary에는 extended vocabulary에 “proportional”한 under-trained fragment가 있다(§3.2). Llama3의 추가 token 28K에는 “krvldkf” 같은 additional under-trained token이 포함된다(§3.2). StarCoder2에는 single document로 정의된 긴 token이 있다(§3.2) (**Result, 단일 연구**, 20개 초과 model 전반).

**조건과 한계.** Conservative threshold로 candidate 상위 2%만 verify한다. OLMo v1.7 7B에서는 모든 token을 verify하면 top 2%의 175개에 비해 191개가 confirmed됐다(§2.4). 논문은 이런 token이 초래하는 downstream accuracy 손실을 측정하지 않는다.

**함의.** Model training data와 다른 tokenizer training data, 그리고 vocabulary extension은 모두 model이 전혀 학습하지 않는 row를 만든다. Indicator에는 embedding matrix만 필요하고 verification은 candidate 상위 2%만 prompt하므로, full run 전에 short training run의 checkpoint에서 실행할 수 있다(Interpretation).

## §5 Pre-training 뒤 토크나이저 확장

**정의.** Tokenizer extension은 pre-trained model에 새 id(언어 piece, domain token, 또는 turn·thinking·tool call용 special token)를 추가하고 embedding 및 output matrix에 row를 추가하며, continued pre-training 또는 fine-tuning 동안 새 row를 학습하는 일이다.

**문제.** 추가 row는 학습되지 않은 상태로 시작한다. 초기화는 학습 전 model output distribution을 바꿀 수 있고, 새 row가 학습되려면 data가 필요하다. 측정 가능한 효과는 extension 전후 model distribution의 KL divergence, 새 token의 emission rate, continued training 뒤 old task score다.

**메커니즘.**
1. 토크나이저에 token을 추가하고 embedding matrix를 resize한다.
2. 새 row를 초기화한다.
3. 새 token을 포함하는 data로 학습한다.
4. 새 token을 쓰지 않는 task에서 retention을 측정한다.

**수식**([[new-embedding-initialization]]): hidden state h, output row `e_1..e_n`, `Z = Σ_j exp(h·e_j)`일 때 token n+1을 추가하면 다음과 같다.

```
KL(p ‖ p') = log(1 + exp(h·e_{n+1}) / Z)
zero init:  e_{n+1} = 0  →  KL = log(1 + 1/Z)
mean init:  e_{n+1} = (1/n)·Σ_j e_j  →  exp(h·e_{n+1}) ≤ Z/n (Jensen)  →  KL ≤ log(1 + 1/n)
```

p와 p'은 extension 전후 original n개 token에 대한 distribution이다.

**계산 예.** GPT-2 vocabulary에서 `n = 50,257`이라고 하자. Prefix에 대한 모든 기존 logit이 −10이라고 가정한다(illustrative assumption). 그러면 `Z = 50,257 × e^−10 = 2.28`이다. Zero-initialized 새 token의 logit은 0이고 probability는 `1/(1 + 2.28) = 0.305`, KL은 `log(1 + 1/2.28) = 0.363 nats`다. Logit이 −15이면 `Z = 0.0154`이고 새 token이 probability의 0.985를 차지한다. Mean initialization은 모든 prefix에서 KL을 `log(1 + 1/50,257) = 1.99 × 10⁻⁵ nats`로 bound한다. 당시 default였던 small-norm random initialization 아래 token 세 개를 추가한 Hewitt의 GPT-2 demonstration은 무관한 prompt 뒤 “Aragorn Aragorn Aragorn Aragorn Frodo Aragorn”을 생성했다([[new-embedding-initialization]]) (**practitioner-evidence**, model 하나). transformers v4.46.0은 `resize_token_embeddings`에서 `mean_resizing: bool = True`를 설정하고, old embedding의 mean과 covariance를 갖는 normal distribution에서 새 row를 sample한다(modeling_utils.py L2080–2116).

**근거.**
- Mid-training 중 special token: Chat special token을 포함한 Tulu3-SFT data에서 OLMo 3 microanneal은 base model이 inference에서 그 token을 출력하게 했다. GSM8K는 49.43에서 0, CruxEval은 32.89에서 18.91로 하락했고, ordinary text로 쓴 같은 template은 46.02와 29.65를 냈다. 저자는 하락을 “the introduction of special tokens to the embedding vocabulary when they have not been seen in pretraining”에 돌리고, loss가 주로 answer parsing이라고 지적하며 mid-training data에서 special token을 제거했다([[olmo-3]] §3.5.4) (**Result, 단일 연구**). OLMo 3 Instruct는 SFT에서 tool-call tag에 special token을 추가했지만, plain-text tag보다 좋다는 preliminary evidence만 보고했다(§5.2.1).
- General data로 pre-train한 1.5B model의 tokenizer를 바꾸고 code로 fine-tune한 경우: 새 tokenizer model은 fine-tuning 5B token 뒤 unchanged-tokenizer model보다 낮았지만, gap은 50B token 뒤 “almost disappears and even inverts (on Pass@1)”했다([[tokenizer-domain-adaptation]] §3.1, Fig. 6). Fast Vocabulary Transfer initialization(기존 embedding space를 새 vocabulary에 mapping, §3.3.1) 대 no transfer는 HumanEval Pass@1 20.5% 대 18.4%, Pass@100 65.3% 대 58.6%였다(Table 4). Llama tokenizer를 교체하지 않고 80k로 확장한 것은 “only small gains”(Pass@1 20.8%)을 냈다(Table 4). 7B와 code token 500B에서 바꾼 tokenizer는 HumanEval Pass@1에서 Code Llama 7B의 1.7점 이내였다(Table 5) (**Result, 단일 연구**).
- Continued pre-training을 한 vocabulary extension(Rakuten 7B)은 추가 token 중 under-trained fragment를 남겼다([[under-trained-tokens]] §3.2).

**조건과 한계.** Dagan 등은 switch 뒤 code-generation metric만 측정했다. English 및 multilingual ability retention은 보고하지 않았다(일반 model의 **Open question**). KL bound는 noise를 추가하지 않을 때만, 그리고 새 token이 input에 나타나기 전까지만 성립한다(**practitioner-evidence**). OLMo 3 result는 chat template 없이 평가한 base-model stage의 special token에 관한 것이다.

**Retention 측정.** 새 언어 또는 domain을 위해 토크나이저를 확장할 때 continued training 전후로 다음을 보고한다. (a) 새 token이 없는 held-out general suite의 score, (b) 새 언어와 기존 언어에서 §3의 언어별 premium, (c) 새 row의 §4 under-trained indicator, (d) 이를 요구하지 않는 prompt에서 base model이 새 special token을 출력하는 emission rate. (a)와 (c)는 [[tokenizer-domain-adaptation]] 및 [[under-trained-tokens]]의 측정을 따르고, (d)는 [[olmo-3]] §3.5.4의 failure를 따른다. ch-32a는 continued pre-training 중 forgetting control을, ch-04는 chat-template과 special-token co-design을 다룬다.

**함의.** Extension은 가능하지만 source는 target domain에서의 recovery만 보인다. Extended model이 original을 대체하기 전에 일반 model에는 위 retention report가 필요하다.

## §6 Provenance 도구: training data의 count, search, trace

**정의.** Provenance tool은 training corpus에 관한 exact-match 질문에 답한다. String이 얼마나 자주 나타나는지(count), 어떤 document가 이를 포함하는지(search), model output의 어느 부분이 training data에 verbatim으로 나타나는지(trace)다.

**문제.** Training data와 verbatim으로 일치하는 output은 general ability로 생성된 것이 아니라 memorized됐을 수 있다. [[quantifying-memorization]] §4.3은 memorization을 올바르게 audit하려면 “likely requires prompting the model with training data”라고 말하며, 이는 training data에 대한 access와 search를 요구한다.

**메커니즘(infini-gram).**
1. Corpus를 tokenize하고 token id를 byte array로 저장한다. `|V| < 65,536`을 가정하면 token당 2 byte이며, document는 `\xff\xff`로 구분한다.
2. Suffix array를 만든다. 이는 모든 suffix의 sorted order이며, 2B–500B-token shard에서는 각각 5 byte pointer로 저장한다.
3. Query length L의 count는 그 query로 시작하는 first suffix와 last suffix를 binary search하여 `O(L + log N)` 시간에 구한다.
4. ∞-gram estimate로 확장한다. Corpus에 나타나는 prompt의 longest suffix를 사용한다([[infini-gram]] §2–§3).

```
P_∞(w_i | w_1:i−1) = cnt(w_{i−(n−1):i−1} w_i) / cnt(w_{i−(n−1):i−1}),
n = max{ n′ ∈ [1, i] : cnt(w_{i−(n′−1):i−1}) > 0 }
```

`cnt(·)`는 corpus에서의 occurrence 수이고, n은 effective context length다.

**계산 예.** Index는 token당 7 byte가 든다. RedPajama의 token 1.4T에서는 `1.4 × 10¹² × 7 bytes = 9.8 TB`이며, 논문은 disk 10 TB와 128-CPU node 하나에서 약 48시간을 보고한다([[infini-gram]] §1, §3). 3T-token corpus는 21 TB가 필요하다(도출). 2-byte token array는 vocabulary id가 65,536 미만이라고 가정하지만 dolma2 tokenizer는 id 100,278개다([[olmo-core-olmo3-configs]]). OLMoTrace는 Llama-2 tokenizer로 OLMo 2 training data를 index한다([[olmotrace]] §3). Index는 tokenizer-specific이다. Infini-gram은 같은 corpus에 대해 tokenizer별 index 하나를 만들었으므로([[infini-gram]] §5.1), tokenizer를 바꾸면 re-indexing이 필요하다.

**근거.**
- **WIMBD**는 10개 corpus에 대한 map-reduce count와 Elasticsearch index를 결합한다. RedPajama 및 LAION-2B-en document의 약 50%가 duplicate임을 발견했고, 점검한 corpus 네 개에는 PromptSource dataset 82개 중 67개가 없었지만 COPA는 RedPajama에 완전히 포함됐음을 발견했다([[wimbd]] Abstract, §4.4.1).
- **OLMoTrace**는 OLMo-2-32B-Instruct의 전체 training data(document 3.16B개, Llama-2 token 4.61T)을 index하고, response의 maximal verbatim span을 찾으며, span unigram probability가 가장 낮은 `⌈0.05 × L⌉` span을 유지하고 span당 document를 최대 10개 retrieve한다. Conversation 98개(mean response token 458)에서 latency 평균은 4.46 s, 유지한 span 평균은 10.4 token, retrieve한 document의 96.7%는 pre-training data에서 왔다([[olmotrace]] Table 1, §3–§4). AIME 2024 answer의 계산 단계 `"\binom{10}{4} = ... = 210"`은 post-training data에 verbatim으로 나타난다(§5).
- **Dolma** contamination audit: WSC, COPA, 그 밖의 evaluation set 네 개가 Dolma에 100% 포함되었으므로 evaluation에서 제외했다([[dolma]] App. L).

**조건과 한계.** Exact-match tool은 paraphrase를 놓친다. OLMoTrace 저자는 retrieve한 document가 “should not be interpreted as having a causal effect on the LM output”이라고 말한다([[olmotrace]] Limitations). WIMBD의 contamination rule은 input field exact match의 upper bound다([[wimbd]] §4.4.1). Causal effect를 추정하는 influence-based attribution은 ch-12a에서 다룬다.

**함의.** 일반 model에서 final tokenized training data의 search index는 “model이 새 problem을 풀 수 있다”와 “답이 data 안에 있다”를 구분하는 instrument다. Training tokenizer로 구축하고 checkpoint와 함께 보관한다.

## §7 Pipeline stage로서의 PII 및 secret 제거

**정의.** PII 제거는 개인을 식별하는 span(email address, phone number, IP address, sensitive attribute와 연결된 name)을 detect하고, span을 mask하거나 document를 제거하거나 source domain을 제거한다. Secret 제거는 code의 credential에 같은 일을 한다.

**문제.** 세 가지 측정은 PII를 publishing 문제가 아니라 training 문제로 만든다.
- Prevalence: Regex match를 extrapolate하면 C4에는 email address 7.6M개와 phone number 19.7M개, mC4-en에는 email 201M개와 phone number 4B개가 있다([[wimbd]] Table 5).
- Memorization: Extractable fraction은 duplicate, model size(parameter 10×당 19 percentage point), context(GPT-Neo 6B에서 token 50개일 때 33%, 450개일 때 65%)에 따라 log-linearly 증가한다([[quantifying-memorization]] §4.1–4.3).
- Extraction: GPT-2 XL에 대한 attack은 candidate 1,800개 중 memorized training example 604개를 confirmed했으며, 이 중 named individual을 가진 것이 46개, contact information을 가진 것이 32개였다. Abstract는 자신이 열거한 PII, IRC conversation, code, UUID 각각이 training document 하나에 나타난다고 말한다([[extracting-training-data]] Abstract, Table 1). Training string memorization은 [[extracting-training-data]]와 [[quantifying-memorization]] 전반에서 **Replicated**지만, 한 번 나타나는 string에서는 둘이 다르다. [[quantifying-memorization]] §6은 자신이 연구한 model에서 그런 string은 드물게 memorized된다고 설명한다.

**메커니즘.**
1. Candidate를 detect한다. Regular expression([[dolma]] §5.3; [[wimbd]] App. B.3.2), domain list([[llama-3]] §3.1.1), model-based document classification([[olmo-3]] §3.4.2), code용 secret scanner([[dolma]] §6.3)를 쓴다.
2. Document별로 mask, remove, keep을 결정한다. Dolma는 document에 PII span이 5개 이하이면(전체 document의 0.02%) special token으로 span을 교체하고, 그보다 많으면(0.001%) document를 제거한다([[dolma]] §5.3). OLMo 3는 “intended for public dissemination” document type(예: 논문의 author list)에서는 PII를 유지하고, PII가 있는 다른 document type은 제거한다([[olmo-3]] §3.4.2).
3. Source별 removal rate와 estimated precision을 기록한다. Deduplication에 대한 배치는 pipeline마다 다르다. Dolma는 URL 및 document deduplication, quality·content filter(그중 PII), paragraph deduplication 순서로 실행한다([[dolma]] §5.5). OLMo 3는 PDF pool에서 PII filtering 전에 MinHash deduplication을 실행한다([[olmo-3]] §3.4.2). 어느 순서든 duplicate는 memorization을 높인다([[quantifying-memorization]] §4.2; ch-12).
4. Output을 audit한다. OLMoTrace는 display 시점에 regex PII filter로 retrieve한 document를 block한다([[olmotrace]] Limitations).

**수식(precision-adjusted count).** Manual sample에서 precision q를 추정한 regex match `N_match`가 있을 때, match 중 expected true PII string 수는 `N_match·q`이고, expected false removal 수는 `N_match·(1 − q)`다. 이 source들은 recall을 추정하지 않는다([[wimbd]] App. B.3.2).

**계산 예.** C4 phone number는 `19.7M matches × 0.92 precision = 18.1M` expected true match와 1.6M false match다. The Stack phone number는 `45.4M matches × 0.09 precision = 4.1M` true 및 41.3M false match다([[wimbd]] Table 5, 도출). Web text에 맞춘 phone-number rule은 code에서 대체로 non-PII string을 제거한다. Dolma masking rule에서 email address 3개가 든 web document는 3 span을 교체한 text를 유지하고, span 6개인 document는 제거한다.

**Model에 미치는 효과의 근거.**
- Masking 대 removal은 Dolma 1.2B ablation model에서 측정된 효과가 없었다([[dolma]] §5.3) (**Result, 단일 연구**).
- OLMo 3의 model-based PDF filter는 남은 science-PDF pool의 4.9%를 제거하고 document 148M개를 남겼다. Filter의 ablation은 보고하지 않는다([[olmo-3]] §3.4.2). Web pool 설명에는 PII step이 없다(§3.4.1).
- Llama 3는 PII volume이 큰 domain을 제거한다. Detector, threshold, removal rate는 보고하지 않는다([[llama-3]] §3.1.1).
- Deduplication은 memorization을 줄이지만 제거하지는 않는다. C4의 exact-substring deduplication으로 memorized 50-token substring에 속하는 generated token fraction은 1.926%에서 0.138%로 낮아졌다([[deduplicating-training-data]] §6.2, Table 4). 35회 미만 반복한 sequence에서는 exact-deduplicated model이 1.2%, 원래 model이 3.6%를 memorized했고, 약 100회 반복 이상에서는 감소가 없었다([[quantifying-memorization]] §5.2) (**Result, 단일 연구**: 1.5B C4 model set 하나를 두 논문에서 분석).
- Gemma 2는 50-token training prompt의 50-token continuation에서 verbatim memorization이 0.1% 미만이라고 보고한다([[gemma-2]] §7, Figure 1).

**조건과 한계.** Regex detector는 contact detail 없는 name을 놓치며 recall은 알려져 있지 않다. Document-level removal은 domain distribution을 바꾼다. 이 장의 어떤 source도 Dolma masking 비교 외에 PII 제거의 capability effect를 측정하지 않는다(**Open question**). Detection은 tokenizer-independent지만 special token masking은 row가 학습되어야 하는 token을 도입한다(§4).

**함의.** PII 처리는 학습과 search index 구축 전에 끝내며, precision은 source type(web, code, paper)별로 측정하고 removal rate를 기록하여 나중에 coverage effect를 점검할 수 있게 한다.

## §8 참고: mixture weight가 token share가 되는 방식과 resume state가 보존하는 것

**정의.** OLMo-core의 source mixture에서 `target_ratio`는 source에서 가져올 요청 training token의 share이고, `max_repetition_ratio`는 source 반복을 허용하며, `max_source_fraction`은 사용할 수 있는 source token share의 상한을 둔다([[olmo-core-olmo3-configs]] source_mixture.py L29–60).

**메커니즘**(source_mixture.py L309–348):

```python
needed_for_source = int(self.requested_tokens * source_config.target_ratio)
max_for_source = int((num_for_source * source_config.max_source_fraction)
                     * source_config.max_repetition_ratio)
if max_for_source < needed_for_source:
    raise OLMoConfigurationError("Insufficient tokens for source ...")
training_steps = math.ceil(self.requested_tokens / self.global_batch_size)
requested_instances = training_steps * num_instances_per_batch
```

그 뒤 file별 instance count는 Hamilton의 largest-remainder method로 round되어 total이 `requested_instances`와 같아진다(L371–400). Official OLMo 3 script는 pre-tokenized `.npy` file의 fixed manifest를 소비하며 file 하나당 `label,path` line 하나다([[olmo-core-olmo3-configs]] mixes).

**계산 예(mixture).** Requested token `R = 100B`, global batch는 4,194,304 token(8,192 × 512, OLMo 3 7B stage-1 값), sequence length는 8,192라고 하자.
- Web: available 200B, target 0.60 → 필요 60B; repetition 1.0이면 허용되며 source를 0.30 pass 돈다.
- Code: available 20B, target 0.30 → 필요 30B > 20B → configuration error다. `max_repetition_ratio = 2.0`이면 cap은 40B이고 run은 code를 1.5 pass 사용한다.
- Math: available 5B, target 0.10 → 필요 10B로, repetition 2.0(2.0 pass)이 필요하다.
- `training_steps = ceil(100B / 4,194,304) = 23,842`; `requested instances = 23,842 × 512 = 12,207,104`(100,000,595,968 token)이다.

Configured ratio는 token share이므로 unique code token 20B와 code share 0.30은 repeated code를 뜻한다. ch-14가 repeated token의 가치를 다룬다.

**Resume state.** Data loader는 `dataset_fingerprint_version`, `dataset_fingerprint`, `batches_processed`, `tokens_processed`, `seed`, `epoch`를 저장한다(data_loader.py L441–449). Load 때 fingerprint가 다르면 `ignore_fingerprint_mismatch=True`가 아니면 error를 낸다(“This will probably result in a different data order!”). Seed가 다르면 saved seed로 대체한다(L451–477). Global order는 `get_rng(self.seed + self.epoch)`로 만든 permutation이고(L667–673), resumed training은 그 order의 `batches_processed` batch를 skip한다(L734–735). 이는 `self.tokens_processed // self.global_batch_size`로 다시 계산한다(L757–763).

**계산 예(resume).** OLMo 3 7B checkpoint manifest는 step 1,000에서 token 4,194,304,000을 기록한다([[olmo-core-olmo3-configs]] OLMo-3-1025-7B.csv).
- 같은 batch로 resume: `4,194,304,000 // 4,194,304 = 1,000` batch, sequence 512개씩이므로 sequence 512,000개를 skip한다.
- 2,097,152-token batch로 resume: batch 2,000개, sequence 256개씩 = sequence 512,000개로 position이 정확하다.
- 3,145,728-token batch(sequence 384개)로 resume: batch 1,333개 = sequence 511,872개이므로 sequence 128개를 두 번 학습한다.

OLMo 3 7B의 두 번째 pre-training script는 schedule을 token 5T에서 7T로 연장한 뒤 `load_trainer_state=True`로 resume한 반면, mid-training script는 new mix에서 `load_trainer_state=False`로 새 data order를 시작했다([[olmo-core-olmo3-configs]] pretrain-2.py L104–111, midtrain.py). [figures/pipeline-ops.html](figures/pipeline-ops.html)의 panel 3은 편집 가능한 source 및 batch size에서 token need, repetition, resume offset을 계산한다.

**함의.** Coverage를 형성하는 mixture는 실제로 소비한 mixture다. Manifest 변경 또는 fingerprint 무시는 restart 뒤 어떤 document를 볼지 바꾸며, 그때 configured token share는 다른 run을 기술한다(**Interpretation**).

## Negative sample과 negative feedback

이 section은 style standard의 용어를 사용한다. 이 stage에서는 “negative”의 네 의미 중 두 가지가 나타난다.

**1. Negative가 오는 곳.**
- **Negative marginal value**(제거 document): PII rule, secret scanner, decontamination으로 제거한 document([[dolma]] §5.3, §6.3, App. L; [[olmo-3]] §3.4.2, §3.5.3)다. Content가 capability를 낮춰서가 아니라 privacy 또는 evaluation integrity를 위해 제거한다. Label은 regex(매치가 있는 corpus와 PII type 전반에서 precision 7%–100%, recall은 미상; [[wimbd]] Table 5) 또는 error rate를 보고하지 않은 LLM classifier([[olmo-3]] §3.4.2)에서 온다.
- **Gradient로서의 negative**(모든 softmax step): 각 non-target token logit은 매 position에서 `−p_j`를 받는다(§4). Target이 된 적 없는 token은 이 term만 받는다.

**2. 현재 관행이 이들을 다루는 방식.** 제거 document는 버리거나(Dolma: PII span 5개 초과 document, OLMo 3: public이 아닌 PDF type), special token으로 mask하여 content로 유지한다(Dolma: span 5개 이하). 어떤 source도 PII document를 negative로 학습하지 않는다. Standard pre-training은 untrained row의 push-down을 control하지 않는다. OLMo 3는 embedding을 weight decay에서 제외한다([[olmo-core-olmo3-configs]] pretrain-1.py).

**3. 메커니즘.** `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. 모든 non-target에서 제거한 mass `(Σ_{j≠y} p_j)`는 target y로 간다. 이미 unlikely한 token은 `p_j`가 작아 매 push-down도 작지만, 모든 batch의 모든 position에서 반복되고 positive term이 이를 상쇄하지 않는다. 이것이 [[under-trained-tokens]] §2.2가 unseen token의 매우 negative logit에 대해 제시한 메커니즘이다. Zero 또는 small-norm row로 추가한 새 token은 logit이 0 부근이다. Hewitt가 pre-training 뒤 일어날 수 있다고 보고한 것처럼([[new-embedding-initialization]]) prefix의 기존 logit이 크게 negative이면, 그 logit은 모든 기존 token에서 probability를 가져간다(§5 계산 예: 모든 logit이 −10일 때 mass의 0.305).

**4. 수치를 둔 근거.** Under-trained token indicator는 OLMo v1.7 7B에서 10 orders of magnitude 전반 training count와 상관하며([[under-trained-tokens]] Fig. 2), vocabulary의 0.1–1%가 confirmed under-trained다(§3). Small-norm random initialization으로 추가한 token은 GPT-2 sample을 지배했다([[new-embedding-initialization]]). Mid-training data의 untrained special token은 해당 token emission을 통해 GSM8K를 49.43에서 0으로 낮췄다([[olmo-3]] §3.5.4). PII document masking 대 removal은 측정된 효과가 없었다([[dolma]] §5.3). 이 장의 source는 capability 변화 중 어느 share가 이 negative 때문에 생겼는지 측정하지 않는다(style standard §6.3).

**5. Control.** 추가 row를 기존 row 평균으로 초기화한다(`KL ≤ log(1 + 1/n)`). Special token은 그것 없이 평가하는 stage에서 제외한다([[olmo-3]] §3.5.4). Tokenizer를 model data에서 학습하여 unseen row를 줄인다([[under-trained-tokens]] §5). Document의 다른 content가 필요하고 masking token이 학습될 경우 removal보다 masking을 선호한다. Precision을 측정한 뒤 uncertain PII detection을 source type별로 mask한다.

**6. Diagnostic.** Input-row norm과 unused-token mean에 대한 output-row cosine distance, repetition prompt의 maximal output probability, base model의 special-token emission rate, held-out text에서 extension 전후 next-token distribution의 KL, 각 PII rule의 source별 removal rate 및 precision을 쓴다.

**7. Generality에 미치는 효과.** Under-trained token은 unusual input에 대한 robustness를 낮추고, untrained special token은 answer format을 깨며, document removal은 이 filter에서 어느 source도 보고하지 않은 정도로 coverage를 바꾼다.

## 레시피

2026-09-15 표시 행은 이 장의 primary document에서 명시한 locus를 읽었고, 2026-09-14 행은 verified library card에서 왔다.

| Model(정확한 release) | 크기 | Stage | 설정 | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama 3 (3.1 herd) | 8B, 70B, 405B | pretrain-stable | tokenizer; vocabulary | 100K tiktoken token + non-English token 28K; 128,000 | arXiv:2407.21783v3 §3.2, Table 3 ([[llama-3]]) | verified 2026-09-15 | §3.2: Llama 2 대비 English compression 3.17 → 3.94 chars/token; 28K token의 downstream gain은 수치 없이 진술 |
| Olmo 3 7B | 7B | pretrain-stable | tokenizer; vocab_size; embedding row | dolma2(cl100k-derived, OLMo 2와 같음); 100,278; 100,352(128의 배수) | arXiv:2512.13961 §3.2; OLMo-core@66f768b tokenizer.py L77–94, OLMo-3-1025-7B-pretrain-1.py ([[olmo-core-olmo3-configs]]) | verified 2026-09-15 | ablation 보고 없음 |
| Olmo 3 7B | 7B | pretrain-stable | sequence; global batch; data seed; embedding weight decay | 8,192; 8,192 × 512 = 4,194,304 token; 34521; 0.0(다른 parameter 0.1) | OLMo-core@66f768b OLMo-3-1025-7B-pretrain-1.py | verified 2026-09-15 | ablation 보고 없음 |
| Olmo 3 7B | 7B | pretrain-stable | schedule horizon; stop; resume | planned 5T; step 597,046 hard stop; part 2는 trainer state resume, horizon 7T, epoch 1회 stop | OLMo-core@66f768b pretrain-1.py L99–101, pretrain-2.py L104–111 | verified 2026-09-15 | “decided to extend schedule to 7T”(code comment); ablation 보고 없음 |
| Olmo 3 7B | 7B | mid-train | global batch; token; data seed; trainer state | token 2,097,152; 100B; 1337; load하지 않음(optimizer state는 load) | OLMo-core@66f768b OLMo-3-1025-7B-midtrain.py | verified 2026-09-15 | ablation 보고 없음 |
| Olmo 3 mid-training data | 7B, 32B | mid-train | instruct data의 chat special token | 제거; newline-based formatting | arXiv:2512.13961 §3.5.4 ([[olmo-3]]) | verified 2026-09-15 | microanneal: special token에서 GSM8K 49.43 → 0, CruxEval 32.89 → 18.91; plain-text template은 46.02 / 29.65 |
| Olmo 3 Instruct | 7B, 32B | SFT | tool-call tag | dedicated special token으로 추가 | arXiv:2512.13961 §5.2.1 | verified 2026-09-15 | preliminary result가 special token을 선호; 수치 미보고 |
| Gemma 2 | 2B, 9B, 27B | pretrain-stable | tokenizer; vocabulary; embedding | SentencePiece, split digit, preserved whitespace, byte-level encoding; 256,128; tied | Gemma 2 report §3.1, Table 1 ([[gemma-2]]) | verified 2026-09-14 | ablation 보고 없음 |
| DeepSeek-V3 | 총 671B | pretrain-stable | tokenizer; boundary handling | byte-level BPE, 128K; punctuation-plus-line-break token random splitting | DeepSeek-V3 report §4.1 ([[deepseek-v3]]) | verified 2026-09-14 | 진술한 목적: token-boundary bias 감소; 수치 없음 |
| Mistral-Nemo-Base-2407 | 12B | pretrain-stable | tokenizer; vocab_size | Tekken(tiktoken-based, 100개 초과 언어); 131,072 | Mistral blog “Tekken”; config.json ([[mistral-nemo]]) | verified 2026-09-14 | blog Figure 2 compression(image; corpus 미기술) |
| FineWeb2 ablation model | 1.46B | pretrain-stable | tokenizer; embedding size | Gemma tokenizer; 256,008 | arXiv:2506.20920v1 §3.1, App. A.4 Table 3 ([[fineweb-2]]) | verified 2026-09-14 | Table 2: unknown token 없이 비교한 tokenizer 중 최저 평균 fertility 2.10(mT5 2.03, XGLM 1.76은 낮지만 [UNK] 출력) |
| Vocabulary-scaling run | non-vocab 2.87B | pretrain-stable | 2.3e21 FLOPs에서 V; sequence; batch; LR | 32K 대 43K(predicted); 2,048; sequence 512개; AdamW 4e-4 → 4e-5 | arXiv:2407.13623v3 Table 3, App. A.7 ([[vocabulary-scaling-laws]]) | verified 2026-09-15 | Table 3: 평균 50.3 → 51.6; seed 미보고 |
| Tokenizer-switch run | 1.5B | mid-train(continued code pre-training, token 5B–500B) | new-row initialization; recovery token | Fast Vocabulary Transfer; ≥ 50B token | arXiv:2402.01035v2 §3.1, Fig. 6, Table 4 ([[tokenizer-domain-adaptation]]) | verified 2026-09-15 | Table 4: FVT 20.5% 대 no FVT 18.4% HumanEval Pass@1 |
| transformers `resize_token_embeddings` | n/a(library default) | SFT(token을 추가하는 모든 stage) | new-row initialization | `mean_resizing=True`(old mean 및 covariance) | huggingface/transformers v4.46.0 modeling_utils.py L2080–2116 ([[new-embedding-initialization]]) | verified 2026-09-15 | framework default; Hewitt(2021)의 KL bound; ablation 보고 없음 |
| Dolma v1.6 web | n/a(data) | data filter(§5.2 stage 없음) | PII rule | regex email/IP/phone; ≤ 5 span mask; > 5 span document 제거 | arXiv:2402.00159v2 §5.3, App. I ([[dolma]]) | verified 2026-09-14 | §5.3: 1.2B ablation에서 masking 대 removal 효과 없음 |
| Olmo 3 science PDF | n/a(data) | data filter(§5.2 stage 없음) | PII classifier; removal | 첫 page의 Gemma 3 12B + 첫 5,000 character의 Gemma 3 4B + rule; 4.9% 제거 → document 148M개 | arXiv:2512.13961 §3.4.2 ([[olmo-3]]) | verified 2026-09-15 | annotator-built taxonomy; ablation 보고 없음 |
| Llama 3 web | n/a(data) | data filter(§5.2 stage 없음) | PII rule | high-PII site의 domain-level removal | arXiv:2407.21783v3 §3.1.1 | not reported(detector, threshold, removal rate; multilingual paragraph를 포함해 §3.1 및 §3.1.1 점검) | ablation 보고 없음 |

**작은 general-purpose run의 시작점.** Multilingual 및 code mixture로 학습하는 parameter 2B–8B의 dense model(위 행이 다루는 크기: non-vocabulary parameter 2.87B, OLMo 3 7B, Llama 3 8B, Gemma 2 2B–9B)에서는 verified 행이 다음을 뒷받침한다. Model 자체 mixture로 학습한 byte-level BPE tokenizer를 사용하고, OLMo 3 7B처럼 vocabulary를 128의 배수로 padding한다(100,278 id → 100,352 row). Model size에 대한 vocabulary-scaling prediction과 vocabulary size를 대조한다(English data의 non-vocabulary parameter 2.87B, 2.3e21 FLOPs에서 43K가 32K보다 좋았고, Llama 3는 8B–405B에서 128,000, Gemma 2는 2B–27B에서 256,128을 쓴다). Input-row norm은 initialization에 머무르므로 output-row indicator로 under-trained token check를 실행하는 경우에만 embedding을 weight decay에서 제외한다. OLMo 3가 microanneal 뒤 했듯 chat special token은 pre-training 및 mid-training data에서 제외한다. Web text에는 Dolma-style regex PII rule(≤ 5 span mask, 그 외 제거)을 적용하고 manual sample에서 source type별 precision을 측정하며, web에 맞춘 phone rule을 code에 적용하지 않는다. 나중에 token을 추가할 경우 mean initialization(transformers default)을 쓰고, 1.5B tokenizer-switch run이 code score를 회복한 뒤의 continued training token 최소 50B를 계획한다.

## 일반화 관점

**(a) Breadth를 높이는 것.** Model mixture에서 학습한 tokenizer: tokenizer training data에서 code 및 multilingual text share가 해당 domain의 compression을 정하고([[tokenizer-domain-adaptation]] Fig. 2), matched tokenizer의 GPT-NeoX는 GPT-2를 재사용한 GPT-J의 200개에 비해 confirmed under-trained token이 10개다([[under-trained-tokens]] Table 1). 추가 non-English vocabulary: Llama 3의 token 28K는 English tokenization에 영향을 주지 않고 compression과 downstream performance를 개선했다([[llama-3]] §3.2; downstream 수치 미보고). 하지만 추가 token에는 under-trained entry도 있다([[under-trained-tokens]] §3.2). Compute에 맞춘 vocabulary: 2.3e21 FLOPs에서 평균 +1.3점([[vocabulary-scaling-laws]] Table 3). Place value에 맞춘 digit segmentation: GPT-3.5 addition accuracy 75.6% → 97.8%([[tokenization-counts-arithmetic]] Fig. 1).

**(b) Narrowing 또는 forgetting을 일으키는 것.** cl100k_base의 Shan 15.05 및 Burmese 11.70 premium은 context window에 English content의 10% 미만만 남기며 content unit당 training compute를 곱한다([[tokenizer-language-unfairness]] Table 1, §5.3). 다른 data에서 tokenizer를 재사용하면 under-trained token이 생긴다([[under-trained-tokens]] §3.2). Mid-training data의 untrained special token은 base-model answer format을 망친다([[olmo-3]] §3.5.4). 충분히 학습하지 않은 tokenizer switch는 50B token 전 target-domain accuracy 비용을 내고([[tokenizer-domain-adaptation]] Fig. 6), non-target ability 효과는 미측정이다. Code의 low-precision PII rule(The Stack phone number 9%)은 주로 non-PII string을 제거한다([[wimbd]] Table 5). Duplicate sequence는 더 많이 memorized된다([[quantifying-memorization]] §4.2). 이는 generic training sequence에서 측정됐고 PII에 특화된 측정은 아니다.

**(c) 이 stage의 측정 방법.** FLORES-200 같은 parallel corpus의 언어별 premium과 domain별 normalized sequence length([[tokenizer-language-unfairness]] §3; [[tokenizer-domain-adaptation]] §2.1), length-match 및 length-mismatch arithmetic의 digit-segmentation test([[tokenization-counts-arithmetic]] §4.1), short run 뒤 under-trained token indicator([[under-trained-tokens]] §2), vocabulary 비교 시 unigram-normalized loss([[vocabulary-scaling-laws]] Eq. 4), duplicate bucket 및 prompt length별 extractable-memorization rate([[quantifying-memorization]] §3.2), evaluation answer의 verbatim span tracing([[olmotrace]])을 사용한다. 알려진 measurement error는 다음과 같다. Raw loss는 vocabulary 간 비교할 수 없다([[vocabulary-scaling-laws]] §2.2). Exact-match trace는 paraphrase를 놓치고 cause를 확립하지 못한다([[olmotrace]] Limitations). Regex PII count는 recall이 측정되지 않았다([[wimbd]] App. B.3.2). Corpus 어디에서나 match한 것을 true continuation 대신 세면 measured memorization은 2.1배 커진다(반복 100회에서 32.6% 대 15.8%, [[quantifying-memorization]] §4.4).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| 서로 다른 tokenizer model의 training loss 비교 | 더 큰 vocabulary model이 같은 step에서 더 나빠 보임 | Unigram-normalized loss 또는 character당 bit 비교([[vocabulary-scaling-laws]] §2.2) |
| 작은 model config로 vocabulary size 선택 | Compute와 함께 optimum이 이동하여 큰 budget의 gain을 놓침 | 계획한 `N_nv`와 FLOPs에서 vocabulary scaling fit을 다시 계산([[vocabulary-scaling-laws]] Table 1) |
| 다른 data에서 학습한 tokenizer 재사용 | Confirmed under-trained token 수백 개, rare string에서 garbled output | Embedding-norm 및 repetition-prompt check 실행([[under-trained-tokens]] §2) |
| 언어별 premium 무시 | Target-language document가 truncate되거나 content unit당 token에서 under-represent됨 | 모든 target language에서 FLORES-200 premium 측정([[tokenizer-language-unfairness]] §4) |
| Mid-training data에 special token 추가 | Base model이 template token을 출력; OLMo 3 microanneal에서 GSM8K가 0으로 하락 | Plain prompt에서 special token emission rate([[olmo-3]] §3.5.4) |
| 추가 row의 zero 또는 small-norm initialization | 학습 전 sample이 새 token에 지배됨 | Held-out text에서 resize 전후 distribution KL; mean initialization([[new-embedding-initialization]]) |
| Web PII regex를 code에 적용 | Precision 보정 뒤 기대 PII count보다 code match count가 훨씬 큼(The Stack phone number: precision 9%에서 match 45.4M) | Source type당 match 100개에서 precision([[wimbd]] Table 5) |
| Deduplication을 PII protection으로 간주 | 많이 반복된 sequence memorization이 남음 | Deduplication 뒤 duplicate bucket별 extractable fraction([[quantifying-memorization]] §5.2) |
| Verbatim trace를 cause의 증거로 읽음 | Document가 behavior를 “taught”했다는 주장 | Trace는 co-occurrence만 보임([[olmotrace]] Limitations); ablation 또는 influence method(ch-12a) 사용 |
| 바뀐 batch size 또는 mix로 resume | Fingerprint error, 또는 `ignore_fingerprint_mismatch=True`의 silent reordering; replay된 sequence | `tokens_processed`가 새 global batch로 나누어떨어지는지와 restart별 dataset fingerprint 점검([[olmo-core-olmo3-configs]] data_loader.py L451–477, L757–763) |
| available token보다 큰 target ratio 설정 | Configuration error, 또는 repetition cap을 올린 뒤 unplanned repetition | Launch 전 source별 needed/available token 계산([[olmo-core-olmo3-configs]] source_mixture.py L309–321) |

## 이해도 점검

1. `C ≈ 6(N_nv + V·d)·H·f(V)`를 사용해 [[vocabulary-scaling-laws]] Table 3의 43K 및 32K run이 FLOPs가 같은 이유와, data가 부족할 때 optimum vocabulary가 왜 내려가는지 설명하라.
2. Model이 cl100k_base로 English와 Standard Arabic의 같은 content를 학습한다. 3.04 premium이 Arabic token share, context usage, training compute를 어떻게 바꾸는지와, 더 큰 multilingual vocabulary가 이 중 무엇을 바꿀 수 있는지 설명하라.
3. Place-value alignment로 left-to-right 3-digit chunking이 sum이 addend보다 digit가 많을 때 주로 실패하는 이유와, direction을 바꾸지 않고 extra token을 추가해도 도움이 안 된 이유를 설명하라.
4. Softmax gradient를 써서 target이 한 번도 아닌 token이 왜 매우 negative logit으로 끝나는지, 이것이 zero-initialized new token을 왜 위험하게 하는지 설명하라. Mean initialization은 damage를 어떻게 bound하는가?
5. OLMo 3는 plain text chat template이 GSM8K 3.4점을 잃는 반면 같은 template의 special token은 49.4점을 잃는다고 발견했다. Mechanism을 설명하고 special token을 pre-training stage에 넣기 전에 어떤 measurement를 추가할지 말하라.
6. Dagan 등은 새 tokenizer에서 token 50B 뒤 code accuracy 회복을 보였다. 이것이 general model의 non-code ability 유지가 증명되지 않는 이유와, 이 경우의 retention report를 설계하라.
7. Quantifying Memorization은 35회 미만 반복 sequence에서는 deduplication이 memorization을 낮추지만 약 100회 이상에서는 낮추지 못한다고 발견했다. 이유와 이것이 PII masking 및 deduplication 순서에 뜻하는 바를 설명하라.
8. Infini-gram index와 OLMoTrace result가 tokenizer에 의존하는 이유와, model tokenizer를 확장할 때 무엇을 rebuild해야 하는지 설명하라.

## 연결

- **이전(dependency):** ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection. 이 장이 tokenize하고 PII를 clean하며 index할 document pool을 만든다.
- **다음:** ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic. Deduplication은 §7에서 측정한 memorization을 유발하는 repetition을 줄인다.
- ch-00 — What General Capability Means and How It Is Measured(retention report에 쓰는 held-out suite).
- ch-04 — Sequence Packing, Loss Masking, and Chat Templates(turn, thinking, tool call용 special token).
- ch-09 — Pretraining Data Composition and Capability Coverage(tokenizer가 encode해야 하는 language 및 code share).
- ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining(memorization scaling 및 influence function의 전체 처리).
- ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale(target ratio 선택 방법).
- ch-13a — Multilingual Coverage and Vocabulary as Capability Axes(언어별 budget 및 multilingual vocabulary 비용).
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination(§8의 repeated token 가치 및 decontamination record).
- ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL(OLMo 3 special-token result가 속하는 stage).
- ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining(tokenizer extension 동안 retention).
- ch-48 — Contamination Detection and Its Effect on Reported Scores(§6 tool을 바탕으로 한 search-based detection).

## 출처

- [[vocabulary-scaling-laws]] — compression fit `f(V)`, compute formula, unigram-normalized loss, optimal-vocabulary prediction, 2.87B comparison, 한계.
- [[tokenizer-language-unfairness]] — premium 정의, English-centric·multilingual·byte-level tokenizer의 FLORES-200 premium, latency 및 context 효과.
- [[tokenization-counts-arithmetic]] — number tokenization strategy, L2R 대 R2L addition accuracy, length-mismatch error.
- [[under-trained-tokens]] — 정의, embedding indicator, verification protocol, model 전반 confirmed count, recommendation.
- [[tokenizer-domain-adaptation]] — normalized sequence length, tokenizer training-data effect, tokenizer switching, FVT 및 extension result, code용 vocabulary-size null result.
- [[new-embedding-initialization]] — added token의 KL derivation, mean-initialization bound, GPT-2 demonstration, transformers `mean_resizing` default.
- [[quantifying-memorization]] — extractability 정의, size·duplication·context scaling, deduplicated-model result, auditing statement.
- [[extracting-training-data]] — GPT-2 XL extraction attack, memorized example 604개, PII category, single-document memorization.
- [[wimbd]] — count 및 search tooling, duplicate 및 contamination 발견, regex PII count와 precision.
- [[infini-gram]] — suffix-array index layout, ∞-gram estimate, index size 및 latency, tokenizer-specific index.
- [[olmotrace]] — OLMo 2 training data의 verbatim span tracing, pipeline, latency 및 stage statistic, limitation, PII display filter.
- [[olmo-3]] — tokenizer lineage, special-token microanneal, tool-call token, PDF PII filter, decontamination.
- [[olmo-core-olmo3-configs]] — padded vocabulary, OLMo 3 7B script, source mixture code, data loader resume state.
- [[dolma]] — regex PII mask 및 removal rule, code secret removal, contamination audit.
- [[fineweb-2]] — multilingual ablation용 fertility에 따른 tokenizer 선택.
- [[llama-3]] — 128K tokenizer construction 및 compression, domain-level PII filtering.
- [[gemma-2]] — split digit 및 byte-level encoding을 둔 256K SentencePiece tokenizer; memorization rate.
- [[deepseek-v3]] — 128K byte-level BPE와 결합 punctuation token의 random splitting.
- [[mistral-nemo]] — Tekken tokenizer vocabulary 및 vendor compression claim.
- [[deduplicating-training-data]] — deduplication 전후 memorized-token rate.
- [[atlas-multilingual-scaling-laws]] — multilingual vocabulary의 compute cost(ch-13a로의 forward link).
