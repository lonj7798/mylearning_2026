<!-- chapter: ch-11
     track: pretraining
     kind: content
     title: Tokenizers, Data Provenance, and PII Removal
     deps: [ch-10a]
     sources: [[vocabulary-scaling-laws]], [[tokenizer-language-unfairness]], [[tokenization-counts-arithmetic]], [[under-trained-tokens]], [[tokenizer-domain-adaptation]], [[new-embedding-initialization]], [[quantifying-memorization]], [[extracting-training-data]], [[wimbd]], [[infini-gram]], [[olmotrace]], [[olmo-3]], [[olmo-core-olmo3-configs]], [[dolma]], [[fineweb-2]], [[llama-3]], [[gemma-2]], [[deepseek-v3]], [[mistral-nemo]], [[deduplicating-training-data]], [[atlas-multilingual-scaling-laws]]
     figures: figures/pipeline-ops.html
     revised: 2026-09 (generality revision)
-->

# Chapter 11 — Tokenizers, Data Provenance, and PII Removal

> **Core insight.** The tokenizer fixes how many tokens each language, number, and code file costs before training starts, and that cost is unequal: the cl100k_base tokenizer needs 1.64 times as many tokens as English for the same Italian text, 3.04 times for Standard Arabic, and 15.05 times for Shan ([[tokenizer-language-unfairness]] Table 1). Vocabulary size is a compute-allocation choice: at 2.87B non-vocabulary parameters and 2.3e21 FLOPs, a 43K vocabulary raised ARC-Challenge from 29.1 to 32.0 over 32K ([[vocabulary-scaling-laws]] Table 3), while tokens that exist in the vocabulary but are rarely seen in training remain untrained (typically 0.1-1% of the vocabulary in the models tested by [[under-trained-tokens]] §5; 1.6% in Qwen1.5 32B, Table 1). Verbatim memorization of training text grows with duplication, model size, and prompt length ([[quantifying-memorization]] §4), and individual phone numbers and addresses were extracted from GPT-2 ([[extracting-training-data]] Fig. 1), so PII removal and training-data search tools are stages of the pre-training pipeline that run before training.
>
> **Guideline.** When a model must work across languages, code, and arithmetic, train the tokenizer on the same mixture as the model, measure the token premium of each target language and the digit segmentation before pre-training, and size the vocabulary for the planned compute, because premiums above 10 remain in English-centric tokenizers and vocabulary size changed downstream accuracy at fixed FLOPs ([[tokenizer-language-unfairness]]; [[vocabulary-scaling-laws]] Tables 2-3). When tokens must be added after pre-training, initialize them from the mean of the existing embeddings and either train on enough data for the new rows to be learned (about 50B tokens in [[tokenizer-domain-adaptation]] Fig. 6) or keep new special tokens out of stages that are evaluated as a base model, because untrained special tokens in OLMo 3 mid-training data made the base model emit them and reduced GSM8K from 49.43 to 0 ([[olmo-3]] §3.5.4). When a corpus may contain personal data, detect and mask or remove it before training and before building search indexes, estimate detector precision on a manual sample, and keep a search index over the final training data, because extraction has succeeded for sequences that occur in a single training document ([[extracting-training-data]] Abstract) and auditing memorization requires prompting with training data ([[quantifying-memorization]] §4.3). Otherwise, reuse a tokenizer only after measuring its premium and under-trained tokens on the new data.

## Why this chapter matters for a general-purpose model

The pipeline for a general-purpose model is pre-training → mid-training → SFT → preference optimization → RL → evaluation. ch-10 and ch-10a decided which documents enter the pre-training pool. This chapter covers three decisions made on that pool before training, plus one reference section on how the pool is consumed:

1. **Tokenization.** A tokenizer maps text to integer ids. Its vocabulary and segmentation rules set the number of tokens per unit of content, which sets training compute per document, the amount of content that fits in a context window, and how numbers and rare words are presented to the model (§1-§5).
2. **Provenance.** Provenance here means the ability to answer, for any string, which training documents contain it. Without this, a correct answer cannot be separated from a memorized one (§6).
3. **PII removal.** Personally identifiable information (PII) is information that can identify a person, such as names with contact details, phone numbers, and email addresses. Memorization is measurable and extraction works, so PII is removed or masked as a pipeline stage (§7).
4. **Mixture consumption.** Configured mixture weights become token shares through a sampler whose state must survive restarts (§8, reference only).

The next chapter (ch-12) removes duplicates. It interacts with this chapter in two ways. Memorization of training sequences rises with the number of duplicates ([[quantifying-memorization]] §4.2); the study measures generic training sequences, and the same dependence is expected for repeated PII strings (Interpretation). Some deduplication and search tools use a tokenizer chosen for the tool: OLMo 3 builds MinHash 5-gram sets with the p50k tokenizer rather than its training tokenizer ([[olmo-3]] App. A.2.2), and OLMoTrace indexes training data with the Llama-2 tokenizer ([[olmotrace]] §3).

## §1 Tokenizer training: what BPE and Unigram produce

**Definition.** Byte-pair encoding (BPE) builds a vocabulary by starting from single characters or bytes and repeatedly adding the most frequent adjacent pair as a new token; segmentation of new text applies the learned merges in the order they were learned ([[tokenization-counts-arithmetic]] §1). Unigram tokenization starts from a large candidate vocabulary, removes pieces whose removal least reduces corpus likelihood under a unigram model, and segments text into the most probable pieces; [[under-trained-tokens]] §1 describes it as the main alternative to BPE, which "despite work suggesting it outperforms BPE" is not in common use. Pre-tokenization is a regular expression that splits text before BPE runs, so no token can cross a split point ([[tokenizer-domain-adaptation]] §2.4).

**Problem.** The same text receives different token counts under different tokenizers. Dagan et al. report that the InCoder tokenizer encodes code with 26% fewer tokens than the Llama tokenizer (normalized sequence length 0.74), while GPT-2's tokenizer needs 19% more (1.19) ([[tokenizer-domain-adaptation]] Table 1).

**Mechanism (BPE training).**
1. Split the training text by the pre-tokenization rule.
2. Initialize the vocabulary with all characters (or all 256 byte values).
3. Count all adjacent token pairs, weighted by word frequency.
4. Add the most frequent pair as a new token and replace its occurrences.
5. Repeat steps 3-4 until the vocabulary reaches the target size V.

**Worked example (toy corpus, this course's illustration).** Word counts: hug 10, pug 5, pun 12, bun 4, hugs 5. Initial pair counts: u+g = 10 + 5 + 5 = 20, p+u = 17, u+n = 16, h+u = 15, g+s = 5, b+u = 4. Merge 1 adds `ug`. Recount: u+n = 16, h+ug = 15, p+u = 12. Merge 2 adds `un`. Recount: h+ug = 15, p+un = 12. Merge 3 adds `hug`. The vocabulary is now {b, g, h, n, p, s, u, ug, un, hug}. Segmenting the unseen word "bugs" applies merge 1 and gives [b, ug, s]: 3 tokens. Segmenting "thug" requires a token for `t`, which is not in the vocabulary; a character-level tokenizer must emit an unknown token, while a byte-level tokenizer falls back to the byte for `t`. Byte-level representation "encodes any Unicode codepoint, even if unseen during training" ([[tokenizer-language-unfairness]] §4.4).

**Evidence: what released tokenizers are.**

| Model | Tokenizer construction | Vocabulary | Locus |
|---|---|---|---|
| Llama 3 (8B/70B/405B) | 100K tokens from tiktoken plus 28K non-English tokens | 128,000 | [[llama-3]] §3.2, Table 3 |
| OLMo 3 (7B/32B) | Same as OLMo 2, derived from cl100k | 100,278, padded to 100,352 | [[olmo-3]] §3.2; [[olmo-core-olmo3-configs]] tokenizer.py L85-94 |
| Gemma 2 | SentencePiece with split digits, preserved whitespace, byte-level encodings | 256,128 | [[gemma-2]] §3.1, Table 1 |
| DeepSeek-V3 | Byte-level BPE; tokens combining punctuation and line breaks are randomly split during training | 128K | [[deepseek-v3]] §4.1 |
| Mistral NeMo | Tekken, tiktoken-based, trained on more than 100 languages | 131,072 | [[mistral-nemo]] "Tekken" section, config.json |
| GPT-2 XL, Llama 2 7B | reported in the under-trained token study | 50,257; 32,000 | [[under-trained-tokens]] Table 1 |

Two patterns appear in these reports: a SentencePiece vocabulary with byte-level encodings (Gemma 2), and a vocabulary built from an existing BPE vocabulary (Llama 3 adds 28K tokens to tiktoken's 100K; OLMo 3 uses a cl100k-derived vocabulary). Llama 3 reports that its tokenizer raised English compression from 3.17 to 3.94 characters per token versus Llama 2, and that the 28K non-English tokens "improved both compression ratios and downstream performance, with no impact on English tokenization"; the downstream numbers are not reported ([[llama-3]] §3.2) (**Result, single study**).

**Padding.** OLMo-core rounds the embedding matrix to a multiple of 128 for throughput: `pad_multiple * ((self.vocab_size + pad_multiple - 1) // pad_multiple)` ([[olmo-core-olmo3-configs]] tokenizer.py L77-82). For vocab_size 100,278: (100,278 + 127) // 128 = 784, and 784 × 128 = 100,352. The 74 extra rows never correspond to a token id produced by the tokenizer, so they never appear as inputs or targets; as output rows they still receive the push-down gradient described in §4. [[under-trained-tokens]] §2.2 uses rows "above the tokenizer vocabulary size" as the reference set of untrained embeddings.

**Conditions and limits.** Pre-tokenization affects both compression and accuracy: the "Identity" pre-tokenizer compressed code 30% better than Llama's but gave "significantly worse" code-generation results at 1.5B ([[tokenizer-domain-adaptation]] §3) (**Result, single study**). Tokenizer training data matters: training on more code improved code compression and training on multilingual data improved multilingual compression ([[tokenizer-domain-adaptation]] §2.3, Fig. 2).

**Implication.** A general-purpose tokenizer is trained on the same mixture of languages, code, and formats as the model, because the tokenizer's merge statistics decide which of these receive short encodings.

## §2 Vocabulary size and compute: the vocabulary scaling law

**Definition.** Vocabulary parameters are the parameters whose count grows with V: N_v = V·d, counting the output layer only ([[vocabulary-scaling-laws]] §2.2, footnote 1). Non-vocabulary parameters N_nv are all others.

**Problem.** A larger vocabulary encodes the same characters in fewer tokens, so more text is processed per FLOP, but rare tokens in a large vocabulary receive few updates. The measurable question is which V minimizes loss at a fixed compute budget C ([[vocabulary-scaling-laws]] §1).

**Mechanism.**
1. Measure data in characters H, because the token count depends on V.
2. Convert with a fitted compression function D = H·f(V).
3. Compare models with a loss that does not depend on V (unigram-normalized loss L_u).
4. For each FLOP budget, find the V with the lowest L_u, then fit power laws across budgets.

**Formulas** ([[vocabulary-scaling-laws]] Eq. 3-5):

```
f(V) = a·(ln V)² + b·ln V + c,        a = 0.0064, b = −0.1581, c = 1.2047
C ≈ 6·(N_nv + V·d)·H·f(V)
L_u = −(1/T)·Σ_i log[ p(w_i | w_<i, V) / p(w_i | V) ]
N_v,opt ∝ N_nv^γ,                     γ ≈ 0.83 (Approach 2); 0.42/0.50 = 0.84 (Approach 1)
```

f(V) is tokens per character; d is the embedding dimension; T is the sequence length; p(w_i | V) is the unigram frequency of token w_i in the tokenized corpus. The natural logarithm reproduces the paper's tables (checked below).

**Worked example (Table 3, overtraining budget).** N_nv = 2.87B, d = 3200, budget 2.3e21 FLOPs.
- V = 32K: f = 0.0064·(10.373)² − 0.1581·10.373 + 1.2047 = 0.2534 tokens per character. Table 3 prints D = 128.5B and H = 509.1B, so D/H = 0.2524. N_v = 32,000 × 3,200 = 0.1024B. C = 6 × (2.87B + 0.1024B) × 128.5B = 2.29e21.
- V = 43K: f = 0.2464; Table 3 prints D = 127.0B, H = 517.5B, D/H = 0.2454. N_v = 0.1376B. C = 6 × 3.0076B × 127.0B = 2.29e21.
- Both runs cost the same FLOPs. The 43K model sees 1.5B fewer tokens but 8.4B more characters.

Panel 1 of [figures/pipeline-ops.html](figures/pipeline-ops.html) lets the reader change V, d, N_nv, and H and see f(V), D, N_v, and C recomputed with these formulas. The figure computes D from f(V) rather than from the printed D, so it shows 2.30e21 for both runs.

**Evidence.** At 2.87B non-vocabulary parameters ([[vocabulary-scaling-laws]] Tables 2-3):

| Budget (FLOPs) | V | ARC-C | HellaSwag | BoolQ | Average of 7 |
|---|---|---|---|---|---|
| 2.8e20 (undertrained) | 32K → 24K | 23.6 → 24.2 | 34.4 → 36.0 | 59.8 → 61.5 | 43.2 → 43.9 |
| 1.2e21 (compute-optimal) | 32K → 35K | 28.5 → 29.1 | 47.5 → 48.1 | 56.4 → 57.1 | 47.9 → 48.5 |
| 2.3e21 (overtrained) | 32K → 43K | 29.1 → 32.0 | 53.0 → 54.1 | 59.5 → 61.9 | 50.3 → 51.6 |

The predicted optimum grows with scale: 43K (Approach 2) at N_nv = 3B, 67K at 7B, 231K at 70B (Table 1); the abstract states that Llama2-70B's optimum is at least 216K against its 32K. The optimum also moved from 16K to 10K when data was scarce and to 24K with excess data at N_nv = 302M (§5, Fig. 7). The paper prints standard deviations of 0.5-2.1 points per task but does not state the number of training seeds (**Result, single study**).

**Conditions and limits.**
- Models up to 3B, dense Transformers, English SlimPajama data; the authors list multilingual data as future work because languages compete for capacity (App. B.2, B.4).
- The quadratic f(V) reaches its minimum at ln V = −b/(2a), V ≈ 231K (derived), and rises above that size, which a real BPE vocabulary cannot do on its training distribution. When V is above about 200K, the fit does not describe compression, so the Table 1 predictions at that size (for example 231K at 70B) rest on extrapolation of f(V) (derived).
- In a different setting, changing the tokenizer of a 1.5B model pre-trained on general data (NL 1.5B) to GPT-4-style vocabularies of 32k, 64k, 128k, and 256k before code fine-tuning gave HumanEval Pass@1 of 20.5, 20.6, 20.8, and 20.5% (Pearson r = −0.13, p = 0.87) ([[tokenizer-domain-adaptation]] Table 3). The vocabulary effects at fixed FLOPs above range from −0.4 to +2.9 points per task and from +0.6 to +1.3 points on the average, against per-task standard deviations of 0.5-2.1 points, and were not tested beyond 3B.
- For multilingual vocabularies, ATLAS reports that a multilingual vocabulary shifts the compute-optimal frontier upward relative to a monolingual vocabulary, most for English ([[atlas-multilingual-scaling-laws]] §3, Fig. 1); ch-13a covers this.

**Implication.** Vocabulary size is chosen with the model size and token budget, not copied from a smaller model, and its evaluation includes per-language and code compression (§3).

## §3 Tokenizer effects on breadth: language premiums and digit segmentation

### Language premiums

**Definition.** The tokenization premium of language A relative to language B is |t(s_A)| / |t(s_B)| for a sentence s_A and its translation s_B under tokenizer t ([[tokenizer-language-unfairness]] §3). Fertility is the related measure of tokens per word.

**Problem.** A premium above 1 means the same content costs more tokens to train on, more tokens to process at inference, and fills the context window sooner.

**Evidence.** On FLORES-200 (2,000 sentences in 200 languages) ([[tokenizer-language-unfairness]] Table 1, Tables 4-5):

| Language | GPT-2 / RoBERTa | cl100k_base (ChatGPT, GPT-4) |
|---|---|---|
| Portuguese | 1.94 | 1.48 |
| Italian | 2.01 | 1.64 |
| Bulgarian | 5.51 | 2.64 |
| Standard Arabic | 4.40 | 3.04 |
| Burmese | 16.89 | 11.70 |
| Shan | 18.76 | 15.05 |

Tokenizers built for multilingual use (XLM-R, NLLB, mT5, M2M100, BLOOM) are closer to parity but "all five models have languages with premiums of more than 2.5" (§4.3). Byte-level encoding does not remove the premium: ByT5's UTF-8 premiums range from 0.87 to 3.94, because ASCII characters take 1 byte in UTF-8, Cyrillic, Greek, and Arabic characters take 2, and CJK characters and Shan consonants and diacritics take 3 (§4.4). Processing time in RoBERTa grows linearly with tokenized length, and Shan takes almost twice English's time (§5.2, Fig. 2). FineWeb2 chose the Gemma tokenizer for its multilingual ablations by subword fertility (average 2.10 tokens per word, against 2.16 for Bloom; mT5 at 2.03 and XGLM at 1.76 were lower but cannot encode some characters) and reports Telugu fertility of 9.74-10.11 tokens per word for the Mistral-v3, Phi3, Llama3, and Command-R tokenizers ([[fineweb-2]] §3.1, Table 2) (**Replicated** across [[tokenizer-language-unfairness]] and [[fineweb-2]] for the existence of large disparities).

**Worked example.** A document fills an 8,192-token context in English under cl100k_base. The same content in Standard Arabic needs 8,192 × 3.04 = 24,904 tokens, so the 8,192-token window holds 1/3.04 = 32.9% of it. In Shan, 8,192 × 15.05 = 123,290 tokens are needed and the window holds 6.6%. At a fixed model size, training compute per token is constant (C ≈ 6ND), so the same content costs 3.04 times the training compute in Standard Arabic (derived). Panel 2 of [figures/pipeline-ops.html](figures/pipeline-ops.html) repeats this calculation for other premiums and window sizes.

**Conditions and limits.** Premiums are measured on one parallel corpus with English-centric named entities, and the paper reports no downstream accuracy ([[tokenizer-language-unfairness]] §6). Removing vocabulary from English costs little compression: one third of cl100k_base's vocabulary makes English FLORES-200 about 10% longer (§6, Fig. 3). Vendor claims such as Tekken being "2x and 3x more efficient" for Korean and Arabic come from an official blog with an unspecified measurement corpus ([[mistral-nemo]] "Tekken" section) and are not controlled comparisons.

### Digit segmentation

**Definition.** Digit segmentation is the rule that splits numbers into tokens: pure BPE (whatever merges were learned), single digits, or fixed chunks.

**Problem.** Arithmetic accuracy depends on whether token boundaries align with place values.

**Mechanism.** GPT-3's p50k_base has single tokens for some 3-digit strings and not others (the paper's example: 710 may have its own token while 711 does not), so equal-length numbers receive different segmentations ([[tokenization-counts-arithmetic]] §1, Fig. 2). cl100k_base has tokens for all 1-, 2-, and 3-digit strings and splits long numbers left-to-right (L2R) into chunks of 3. PaLM, Llama 1 and 2, and Mistral use single digits; OLMo (2024) used pure BPE (Table 1).

**Worked example (the paper's Fig. 1 problem).** 8302080 + 3529456 = 11831536. L2R chunks: [830][208][0] + [352][945][6] = [118][315][36]. Right-to-left (R2L) chunks forced by commas: [8][302][080] + [3][529][456] = [11][831][536]. In the R2L form each chunk of the answer covers the same place values as the corresponding chunks of the addends. In the L2R form the answer has 8 digits and the addends 7, so every chunk boundary of the answer is shifted relative to the addends.

**Evidence.** On 90 addition problems with 7-9 digit addends, 8-shot, greedy: GPT-3.5 75.6% (L2R) vs 97.8% (R2L); GPT-4 84.4% vs 98.9% ([[tokenization-counts-arithmetic]] Fig. 1). When the answer is longer than the addends, L2R accuracy falls to 8.25%, and errors fall on the fourth digit (§4.1, §4.3). Replacing commas with other single-token separators gave similar gains; adding tokens without changing the direction did not (§3.2-3.3) (**Result, single study**; tokenization was changed at inference, not in pre-training).

**Conditions and limits.** The study uses closed models and inference-time formatting. OLMo-core includes a `dolma2_sigdig` tokenizer documented as "The R2L dolma2 tokenizer" ([[olmo-core-olmo3-configs]] tokenizer.py L30-32); no evaluation of it is reported in the sources for this chapter. Gemma 2 splits digits ([[gemma-2]] §3.1) without a reported ablation.

**Implication.** Number handling is fixed at tokenizer design time. A general model's tokenizer evaluation includes a digit-segmentation check alongside language premiums.

## §4 Under-trained tokens

**Definition.** Under-trained tokens are vocabulary entries that were nearly or entirely absent from the model's training data ([[under-trained-tokens]] §1). Unreachable tokens are entries never produced when their decoded string is re-tokenized (§2.1).

**Problem.** Such tokens occupy vocabulary slots and, when they appear in inputs, can produce garbled or hallucinated outputs; the authors note the risk for agents that process external data (§1).

**Mechanism.** The softmax log-likelihood gradient with respect to logit z_j is `∂ log p_y / ∂ z_j = 1[j = y] − p_j`. A token that is never the target y receives only the negative term −p_j at every step, and its output row e_j moves along −p_j·h for each hidden state h. Because every never-seen row receives the same kind of update, the rows share a direction (the authors' explanation); [[under-trained-tokens]] §2.2 describes unseen output rows "moving away" from the mean output vector so the model can assign "highly negative logits for tokens that are never the correct prediction". An input embedding row receives a gradient only when its token appears in the input; otherwise it changes only through weight decay (toward zero) or stays at its initialization (§2.2).

**Worked example.** Logits z = [2, 1, 0, −1] with target token 0 give p = [0.644, 0.237, 0.087, 0.032]. The gradient of log p_0 is [+0.356, −0.237, −0.087, −0.032]: token 3's logit is pushed down even though it was already the least likely. With decoupled weight decay λ = 0.1 at constant learning rate 3e-4, an unseen input row is multiplied by (1 − 3e-5) each step and keeps (1 − 3e-5)^100,000 = 5.0% of its initial norm after 100,000 steps (arithmetic under a constant-LR assumption). The OLMo 3 7B script sets `weight_decay=0.0` for `embeddings.weight` ([[olmo-core-olmo3-configs]] pretrain-1.py), so in that configuration unseen input rows keep their initial values.

**Indicators and verification** ([[under-trained-tokens]] §2.2-2.4):

```
u_ref = (1/|t_ref|) · Σ_{i ∈ t_ref} E_out,i
C(A, x)_i = 1 − (A_i · x) / (‖A_i‖ ‖x‖)      (tied embeddings: rank tokens by C(E_out, u_ref))
L2(E_in)_i = ‖E_in,i‖                         (untied embeddings: rank tokens by input-row norm)
```

t_ref is a set of known unused tokens, E_out and E_in are the output and input embedding matrices, and A_i is row i. The top 2% of candidates are prompted to repeat the token; a token is confirmed when its maximal output probability is below 1%.

**Evidence** ([[under-trained-tokens]] Table 1, confirmed/tested): GPT-NeoX 20B 10/993; OLMo v1.7 7B 178/993; Llama3 8B 556/2540; Qwen1.5 32B 2450/2966; Gemma 2B 3161/5117; Command R 306/5012. The authors summarize confirmed tokens as "typically around 0.1–1% of the vocabulary", with prevalence that "varies significantly" (§5). On OLMo v1.7 7B the indicators correlate with first-epoch token counts across ten orders of magnitude (Fig. 2). GPT-NeoX 20B, whose tokenizer was trained on its training corpus, has 10 confirmed; GPT-J 6B and Phi-2, which reuse GPT-2's tokenizer on different data, have 200 and 103 (GPT-2 XL itself has 67); Rakuten 7B's extended Japanese vocabulary contains under-trained fragments "proportional to the extended vocabulary" (§3.2). Llama3's 28K added tokens include additional under-trained tokens such as "krvldkf" (§3.2). StarCoder2 has long tokens defined by single documents (§3.2) (**Result, single study**, across 20+ models).

**Conditions and limits.** Only the top 2% of candidates are verified, with a conservative threshold; on OLMo v1.7 7B, verifying every token confirmed 191 against 175 from the top 2% (§2.4). The paper does not measure downstream accuracy loss caused by these tokens.

**Implication.** Tokenizer training data that differs from model training data, and vocabulary extension, both create rows the model never learns. The indicators need only the embedding matrices, and the verification step prompts only the top 2% of candidates, so they can be run on a checkpoint from a short training run before the full run (Interpretation).

## §5 Extending a tokenizer after pre-training

**Definition.** Tokenizer extension adds new ids (language pieces, domain tokens, or special tokens for turns, thinking, or tool calls) to a pre-trained model, adds rows to the embedding and output matrices, and trains the new rows during continued pre-training or fine-tuning.

**Problem.** The added rows start untrained. Their initialization can change the model's output distribution before any training, and the new rows need data to be learned. The measurable effects are the KL divergence between the model's distributions before and after extension, the emission rate of the new tokens, and scores on old tasks after continued training.

**Mechanism.**
1. Add tokens to the tokenizer and resize the embedding matrices.
2. Initialize the new rows.
3. Train on data that contains the new tokens.
4. Measure retention on tasks that do not use the new tokens.

**Formula** ([[new-embedding-initialization]]): with hidden state h, output rows e_1..e_n, and Z = Σ_j exp(h·e_j), adding token n+1 gives

```
KL(p ‖ p') = log(1 + exp(h·e_{n+1}) / Z)
zero init:  e_{n+1} = 0  →  KL = log(1 + 1/Z)
mean init:  e_{n+1} = (1/n)·Σ_j e_j  →  exp(h·e_{n+1}) ≤ Z/n (Jensen)  →  KL ≤ log(1 + 1/n)
```

p and p' are the distributions over the original n tokens before and after extension.

**Worked example.** GPT-2 vocabulary, n = 50,257. Suppose every existing logit for a prefix equals −10 (an illustrative assumption). Then Z = 50,257 × e^−10 = 2.28. A zero-initialized new token has logit 0 and probability 1/(1 + 2.28) = 0.305, and KL = log(1 + 1/2.28) = 0.363 nats. With logits of −15, Z = 0.0154 and the new token takes 0.985 of the probability. Mean initialization bounds KL by log(1 + 1/50,257) = 1.99 × 10⁻⁵ nats for every prefix. Hewitt's GPT-2 demonstration with three tokens added under the then-default small-norm random initialization generated "Aragorn Aragorn Aragorn Aragorn Frodo Aragorn" after an unrelated prompt ([[new-embedding-initialization]]) (**practitioner-evidence**, one model). transformers v4.46.0 sets `mean_resizing: bool = True` in `resize_token_embeddings`, sampling new rows from a normal distribution with the old embeddings' mean and covariance (modeling_utils.py L2080-2116).

**Evidence.**
- Special tokens during mid-training: OLMo 3 microanneals on Tulu3-SFT data that contained chat special tokens made the base model emit those tokens at inference; GSM8K fell from 49.43 to 0 and CruxEval from 32.89 to 18.91, while the same template written as ordinary text gave 46.02 and 29.65. The authors attribute the drop to "the introduction of special tokens to the embedding vocabulary when they have not been seen in pretraining", note that the loss is mainly answer parsing, and removed special tokens from mid-training data ([[olmo-3]] §3.5.4) (**Result, single study**). OLMo 3 Instruct did add special tokens for tool-call tags in SFT, reporting only preliminary evidence that this beat plain-text tags (§5.2.1).
- Switching the tokenizer of a 1.5B model pre-trained on general data and then fine-tuned on code: models with a new tokenizer trailed the unchanged-tokenizer model after 5B tokens of fine-tuning, and the gap "almost disappears and even inverts (on Pass@1) after 50B tokens" ([[tokenizer-domain-adaptation]] §3.1, Fig. 6). Fast Vocabulary Transfer initialization (a mapping of the old embedding space onto the new vocabulary, §3.3.1) versus no transfer: HumanEval Pass@1 20.5% vs 18.4%, Pass@100 65.3% vs 58.6% (Table 4). Extending the Llama tokenizer to 80k instead of replacing it gave "only small gains" (20.8% Pass@1) (Table 4). At 7B with 500B code tokens, changed tokenizers scored within 1.7 points of Code Llama 7B on HumanEval Pass@1 (Table 5) (**Result, single study**).
- Vocabulary extension with continued pre-training (Rakuten 7B) left under-trained fragments among the added tokens ([[under-trained-tokens]] §3.2).

**Conditions and limits.** Dagan et al. measure only code-generation metrics after the switch; retention of English and multilingual ability is not reported (**Open question** for general models). The KL bound holds without added noise and only until a new token appears in the input (**practitioner-evidence**). The OLMo 3 result concerns special tokens in a base-model stage evaluated without a chat template.

**Retention measurement.** When a tokenizer is extended for a new language or domain, report, before and after continued training, (a) scores on a held-out general suite that contains no new tokens, (b) per-language premium from §3 on the new and old languages, (c) the under-trained indicators of §4 for the new rows, and (d) the base model's emission rate of new special tokens on prompts that do not call for them. Items (a) and (c) follow the measurements in [[tokenizer-domain-adaptation]] and [[under-trained-tokens]]; (d) follows the failure in [[olmo-3]] §3.5.4. ch-32a covers forgetting control during continued pre-training, and ch-04 covers chat-template and special-token co-design.

**Implication.** Extension is feasible, but the sources show recovery on the target domain only; a general model needs the retention report above before the extended model replaces the original.

## §6 Provenance tools: counting, searching, and tracing training data

**Definition.** A provenance tool answers exact-match questions about a training corpus: how often a string occurs (count), which documents contain it (search), and which parts of a model output occur verbatim in training data (trace).

**Problem.** An output that matches training data verbatim may be memorized rather than generated from general ability. [[quantifying-memorization]] §4.3 states that correctly auditing memorization "likely requires prompting the model with training data", which requires access to and search over the training data.

**Mechanism (infini-gram).**
1. Tokenize the corpus and store token ids as a byte array (2 bytes per token, assuming |V| < 65,536), with documents separated by `\xff\xff`.
2. Build a suffix array: the sorted order of all suffixes, stored as pointers of 5 bytes each for shards of 2B-500B tokens.
3. Count a query of length L by binary search for the first and last suffix that start with it, in O(L + log N) time.
4. Extend to the ∞-gram estimate: use the longest suffix of the prompt that occurs in the corpus ([[infini-gram]] §2-§3).

```
P_∞(w_i | w_1:i−1) = cnt(w_{i−(n−1):i−1} w_i) / cnt(w_{i−(n−1):i−1}),
n = max{ n′ ∈ [1, i] : cnt(w_{i−(n′−1):i−1}) > 0 }
```

cnt(·) is the number of occurrences in the corpus and n is the effective context length.

**Worked example.** The index costs 7 bytes per token. For RedPajama's 1.4T tokens: 1.4 × 10¹² × 7 bytes = 9.8 TB; the paper reports 10 TB of disk and about 48 hours on one 128-CPU node ([[infini-gram]] §1, §3). A 3T-token corpus needs 21 TB (derived). The 2-byte token array assumes a vocabulary below 65,536 ids; the dolma2 tokenizer has 100,278 ids ([[olmo-core-olmo3-configs]]), and OLMoTrace indexes OLMo 2 training data with the Llama-2 tokenizer ([[olmotrace]] §3). Indexes are tokenizer-specific: infini-gram built one index per tokenizer for the same corpora ([[infini-gram]] §5.1), so a tokenizer change requires re-indexing.

**Evidence.**
- **WIMBD** combines map-reduce counting with an Elasticsearch index over ten corpora. It found that about 50% of RedPajama and LAION-2B-en documents are duplicates, and that 67 of 82 PromptSource datasets were absent from the four corpora checked while COPA was fully contained in RedPajama ([[wimbd]] Abstract, §4.4.1).
- **OLMoTrace** indexes the full training data of OLMo-2-32B-Instruct (3.16B documents, 4.61T Llama-2 tokens), finds maximal verbatim spans of a response, keeps the ⌈0.05 × L⌉ spans with the lowest span unigram probability, and retrieves up to 10 documents per span. On 98 conversations (mean response 458 tokens) latency averaged 4.46 s, kept spans averaged 10.4 tokens, and 96.7% of retrieved documents came from pre-training data ([[olmotrace]] Table 1, §3-§4). A calculation step of an AIME 2024 answer, "\binom{10}{4} = ... = 210", appears verbatim in post-training data (§5).
- **Dolma** contamination audit: WSC, COPA, and four other evaluation sets are 100% contained in Dolma, so they were excluded from its evaluations ([[dolma]] App. L).

**Conditions and limits.** Exact-match tools miss paraphrases. OLMoTrace's authors state that retrieved documents "should not be interpreted as having a causal effect on the LM output" ([[olmotrace]] Limitations). WIMBD's contamination rule is an upper bound for exact matches of input fields ([[wimbd]] §4.4.1). Influence-based attribution, which estimates causal effects, is covered in ch-12a.

**Implication.** For a general model, a search index over the final tokenized training data is the instrument that separates "the model can solve new problems" from "the answer is in the data". It is built with the training tokenizer and kept with the checkpoint.

## §7 PII and secret removal as a pipeline stage

**Definition.** PII removal detects spans that identify individuals (email addresses, phone numbers, IP addresses, names linked to sensitive attributes) and either masks the span, removes the document, or removes the source domain. Secret removal does the same for credentials in code.

**Problem.** Three measurements make PII a training problem rather than a publishing problem.
- Prevalence: regex matches extrapolate to 7.6M email addresses and 19.7M phone numbers in C4, and 201M emails and 4B phone numbers in mC4-en ([[wimbd]] Table 5).
- Memorization: extractable fractions grow log-linearly with duplicates, model size (19 percentage points per 10× parameters), and context (33% at 50 tokens vs 65% at 450 tokens for GPT-Neo 6B) ([[quantifying-memorization]] §4.1-4.3).
- Extraction: an attack on GPT-2 XL confirmed 604 memorized training examples out of 1,800 candidates, including 46 with named individuals and 32 with contact information; the abstract states that the PII, IRC conversations, code, and UUIDs it lists each occur in one training document ([[extracting-training-data]] Abstract, Table 1). Memorization of training strings is **Replicated** across [[extracting-training-data]] and [[quantifying-memorization]]; the two differ for strings that occur once, which [[quantifying-memorization]] §6 describes as rarely memorized in the models it studied.

**Mechanism.**
1. Detect candidates: regular expressions ([[dolma]] §5.3; [[wimbd]] App. B.3.2), domain lists ([[llama-3]] §3.1.1), model-based document classification ([[olmo-3]] §3.4.2), or secret scanners for code ([[dolma]] §6.3).
2. Decide per document: mask, remove, or keep. Dolma replaces spans with special tokens when a document has 5 or fewer PII spans (0.02% of documents) and removes documents with more (0.001%) ([[dolma]] §5.3). OLMo 3 keeps PII in document types "intended for public dissemination" (for example author lists in papers) and removes other document types with PII ([[olmo-3]] §3.4.2).
3. Record removal rates and estimated precision per source. Placement relative to deduplication differs by pipeline: Dolma runs URL and document deduplication, then quality and content filters (PII among them), then paragraph deduplication ([[dolma]] §5.5); OLMo 3 runs MinHash deduplication on its PDF pool before PII filtering ([[olmo-3]] §3.4.2). Duplicates raise memorization in either order ([[quantifying-memorization]] §4.2; ch-12).
4. Audit outputs: OLMoTrace blocks retrieved documents with a regex PII filter at display time ([[olmotrace]] Limitations).

**Formula (precision-adjusted count).** With N_match regex matches and precision q estimated from a manual sample, the expected number of true PII strings among the matches is N_match·q, and the expected number of false removals is N_match·(1 − q). Recall is not estimated by these sources ([[wimbd]] App. B.3.2).

**Worked example.** C4 phone numbers: 19.7M matches × 0.92 precision = 18.1M expected true matches and 1.6M false matches. The Stack phone numbers: 45.4M matches × 0.09 precision = 4.1M true and 41.3M false matches ([[wimbd]] Table 5, derived). A phone-number rule tuned on web text would remove mostly non-PII strings from code. Dolma's masking rule: a web document with 3 email addresses keeps its text with the 3 spans replaced; a document with 6 spans is removed.

**Evidence of effect on the model.**
- Masking versus removal had no measured effect on Dolma's 1.2B ablation models ([[dolma]] §5.3) (**Result, single study**).
- OLMo 3's model-based PDF filter removed 4.9% of the remaining science-PDF pool, leaving 148M documents; no ablation of the filter is reported ([[olmo-3]] §3.4.2). The web pool description has no PII step (§3.4.1).
- Llama 3 removes domains with high volumes of PII; no detector, threshold, or removal rate is reported ([[llama-3]] §3.1.1).
- Deduplication reduces memorization but does not remove it: the fraction of generated tokens in memorized 50-token substrings fell from 1.926% to 0.138% with exact-substring deduplication of C4 ([[deduplicating-training-data]] §6.2, Table 4); for sequences repeated fewer than 35 times the exact-deduplicated model memorized 1.2% versus 3.6%, with no reduction above about 100 repeats ([[quantifying-memorization]] §5.2) (**Result, single study**: one set of 1.5B C4 models, analyzed in both papers).
- Gemma 2 reports verbatim memorization below 0.1% on 50-token continuations of 50-token training prompts ([[gemma-2]] §7, Figure 1).

**Conditions and limits.** Regex detectors miss names without contact details and have unknown recall. Document-level removal changes the domain distribution; no source in this chapter measures the capability effect of PII removal beyond the Dolma masking comparison (**Open question**). Detection is tokenizer-independent, but masking with special tokens introduces tokens whose rows must be trained (§4).

**Implication.** PII handling is completed before training and before search indexes are built, its precision is measured per source type (web, code, papers), and its removal rates are recorded so that coverage effects can be checked later.

## §8 Reference: how mixture weights become token shares, and what resume state preserves

**Definition.** In OLMo-core's source mixture, `target_ratio` is the share of requested training tokens drawn from a source; `max_repetition_ratio` allows a source to be repeated; `max_source_fraction` caps the share of a source's tokens that may be used ([[olmo-core-olmo3-configs]] source_mixture.py L29-60).

**Mechanism** (source_mixture.py L309-348):

```python
needed_for_source = int(self.requested_tokens * source_config.target_ratio)
max_for_source = int((num_for_source * source_config.max_source_fraction)
                     * source_config.max_repetition_ratio)
if max_for_source < needed_for_source:
    raise OLMoConfigurationError("Insufficient tokens for source ...")
training_steps = math.ceil(self.requested_tokens / self.global_batch_size)
requested_instances = training_steps * num_instances_per_batch
```

Per-file instance counts are then rounded with Hamilton's largest-remainder method so the total equals `requested_instances` (L371-400). The official OLMo 3 scripts consume fixed manifests of pre-tokenized `.npy` files, one `label,path` line per file ([[olmo-core-olmo3-configs]] mixes).

**Worked example (mixture).** Requested tokens R = 100B; global batch 4,194,304 tokens (8,192 × 512, the OLMo 3 7B stage-1 value); sequence length 8,192.
- Web: 200B available, target 0.60 → needs 60B; allowed with repetition 1.0; 0.30 passes over the source.
- Code: 20B available, target 0.30 → needs 30B > 20B → configuration error. With `max_repetition_ratio = 2.0` the cap is 40B and the run uses 1.5 passes over code.
- Math: 5B available, target 0.10 → needs 10B, which requires repetition 2.0 (2.0 passes).
- training_steps = ceil(100B / 4,194,304) = 23,842; requested instances = 23,842 × 512 = 12,207,104 (100,000,595,968 tokens).
The configured ratios are token shares, so a 0.30 code share with 20B unique code tokens means repeated code. ch-14 covers the value of repeated tokens.

**Resume state.** The data loader saves `dataset_fingerprint_version`, `dataset_fingerprint`, `batches_processed`, `tokens_processed`, `seed`, and `epoch` (data_loader.py L441-449). On load, a different fingerprint raises an error unless `ignore_fingerprint_mismatch=True` ("This will probably result in a different data order!"), and a different seed is replaced by the saved one (L451-477). The global order is a permutation generated with `get_rng(self.seed + self.epoch)` (L667-673), and resumed training skips `batches_processed` batches of that order (L734-735), recomputed as `self.tokens_processed // self.global_batch_size` (L757-763).

**Worked example (resume).** The OLMo 3 7B checkpoint manifest records 4,194,304,000 tokens at step 1,000 ([[olmo-core-olmo3-configs]] OLMo-3-1025-7B.csv).
- Resume with the same batch: 4,194,304,000 // 4,194,304 = 1,000 batches of 512 sequences = 512,000 sequences skipped.
- Resume with a 2,097,152-token batch: 2,000 batches of 256 sequences = 512,000 sequences; the position is exact.
- Resume with a 3,145,728-token batch (384 sequences): 1,333 batches = 511,872 sequences; 128 sequences are trained twice.
OLMo 3 7B's second pre-training script resumed with `load_trainer_state=True` after the schedule was extended from 5T to 7T tokens, while the mid-training script started a new data order with `load_trainer_state=False` on a new mix ([[olmo-core-olmo3-configs]] pretrain-2.py L104-111, midtrain.py). Panel 3 of [figures/pipeline-ops.html](figures/pipeline-ops.html) computes token needs, repetition, and resume offsets for editable sources and batch sizes.

**Implication.** The mixture that shapes coverage is the one actually consumed. A changed manifest or an ignored fingerprint changes which documents are seen after a restart, and the configured token shares then describe a different run (**Interpretation**).

## Negative samples and negative feedback

This section uses the terms of the style standard. Two of the four meanings of "negative" occur at this stage.

**1. Where negatives come from.**
- **Negative marginal value** (removed documents): documents removed by PII rules, secret scanners, and decontamination ([[dolma]] §5.3, §6.3, App. L; [[olmo-3]] §3.4.2, §3.5.3). They are removed for privacy or evaluation integrity, not because their content lowers capability. The labels come from regexes (precision from 7% to 100% across corpora and PII types with matches, recall unknown; [[wimbd]] Table 5) or from LLM classifiers without a reported error rate ([[olmo-3]] §3.4.2).
- **Negative as gradient** (every softmax step): each non-target token's logit receives −p_j at every position (§4). Tokens that are never targets receive only this term.

**2. What current practice does with them.** Removed documents are discarded (Dolma for documents with more than 5 PII spans, OLMo 3 for non-public PDF types) or masked with special tokens and kept as content (Dolma for 5 or fewer spans). No source trains on PII documents as negatives. The push-down on untrained rows is not controlled in standard pre-training; OLMo 3 excludes embeddings from weight decay ([[olmo-core-olmo3-configs]] pretrain-1.py).

**3. Mechanism.** `∂ log p_y / ∂ z_j = 1[j = y] − p_j`. The mass removed from all non-targets (Σ_{j≠y} p_j) goes to the target y. For a token that is already unlikely, p_j is small, so each push-down is small, but it repeats at every position of every batch and is never offset by a positive term, which is the mechanism [[under-trained-tokens]] §2.2 gives for the very negative logits of unseen tokens. A new token added with a zero or small-norm row has logit near 0. When the existing logits for a prefix are large and negative, which Hewitt reports can happen after pre-training ([[new-embedding-initialization]]), that logit takes probability from every existing token (§5 worked example: 0.305 of the mass when all logits are −10).

**4. Evidence with numbers.** Under-trained token indicators correlate with training counts across ten orders of magnitude on OLMo v1.7 7B ([[under-trained-tokens]] Fig. 2); 0.1-1% of vocabularies are confirmed under-trained (§3). Tokens added with small-norm random initialization dominated GPT-2 samples ([[new-embedding-initialization]]). Untrained special tokens in mid-training data reduced GSM8K from 49.43 to 0 through emission of those tokens ([[olmo-3]] §3.5.4). Masking versus removing PII documents had no measured effect ([[dolma]] §5.3). No source in this chapter measures what share of any capability change is caused by these negatives (§6.3 of the style standard).

**5. Controls.** Initialize added rows from the mean of existing rows (KL ≤ log(1 + 1/n)); keep special tokens out of stages evaluated without them ([[olmo-3]] §3.5.4); train the tokenizer on the model's data so fewer rows stay unseen ([[under-trained-tokens]] §5); prefer masking over removal when a document's other content is needed and the masking token will be trained; mask uncertain PII detections by source type after measuring precision.

**6. Diagnostics.** Input-row norms and output-row cosine distance to the unused-token mean; maximal output probability on repetition prompts; the base model's emission rate of special tokens; KL between pre- and post-extension next-token distributions on held-out text; removal rate and precision per source for each PII rule.

**7. Effect on generality.** Under-trained tokens reduce robustness to unusual inputs; untrained special tokens break answer formats; document removal changes coverage by an amount no source reports for these filters.

## Recipe

Rows marked 2026-09-15 were read at the stated locus in the primary document for this chapter; rows marked 2026-09-14 are from verified library cards.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3 (3.1 herd) | 8B, 70B, 405B | pretrain-stable | tokenizer; vocabulary | 100K tiktoken tokens + 28K non-English tokens; 128,000 | arXiv:2407.21783v3 §3.2, Table 3 ([[llama-3]]) | verified 2026-09-15 | §3.2: English compression 3.17 → 3.94 chars/token vs Llama 2; downstream gain from 28K tokens stated without numbers |
| Olmo 3 7B | 7B | pretrain-stable | tokenizer; vocab_size; embedding rows | dolma2 (cl100k-derived, same as OLMo 2); 100,278; 100,352 (multiple of 128) | arXiv:2512.13961 §3.2; OLMo-core@66f768b tokenizer.py L77-94, OLMo-3-1025-7B-pretrain-1.py ([[olmo-core-olmo3-configs]]) | verified 2026-09-15 | no ablation reported |
| Olmo 3 7B | 7B | pretrain-stable | sequence; global batch; data seed; embedding weight decay | 8,192; 8,192 × 512 = 4,194,304 tokens; 34521; 0.0 (other params 0.1) | OLMo-core@66f768b OLMo-3-1025-7B-pretrain-1.py | verified 2026-09-15 | no ablation reported |
| Olmo 3 7B | 7B | pretrain-stable | schedule horizon; stop; resume | planned 5T; hard stop at step 597,046; part 2 resumed with trainer state, horizon 7T, stop at 1 epoch | OLMo-core@66f768b pretrain-1.py L99-101, pretrain-2.py L104-111 | verified 2026-09-15 | "decided to extend schedule to 7T" (code comment); no ablation reported |
| Olmo 3 7B | 7B | mid-train | global batch; tokens; data seed; trainer state | 2,097,152 tokens; 100B; 1337; not loaded (optimizer state loaded) | OLMo-core@66f768b OLMo-3-1025-7B-midtrain.py | verified 2026-09-15 | no ablation reported |
| Olmo 3 mid-training data | 7B, 32B | mid-train | chat special tokens in instruct data | removed; newline-based formatting | arXiv:2512.13961 §3.5.4 ([[olmo-3]]) | verified 2026-09-15 | microanneal: GSM8K 49.43 → 0, CruxEval 32.89 → 18.91 with special tokens; 46.02 / 29.65 with plain-text template |
| Olmo 3 Instruct | 7B, 32B | SFT | tool-call tags | added as dedicated special tokens | arXiv:2512.13961 §5.2.1 | verified 2026-09-15 | preliminary results favor special tokens; numbers not reported |
| Gemma 2 | 2B, 9B, 27B | pretrain-stable | tokenizer; vocabulary; embeddings | SentencePiece, split digits, preserved whitespace, byte-level encodings; 256,128; tied | Gemma 2 report §3.1, Table 1 ([[gemma-2]]) | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B total | pretrain-stable | tokenizer; boundary handling | byte-level BPE, 128K; random splitting of punctuation-plus-line-break tokens | DeepSeek-V3 report §4.1 ([[deepseek-v3]]) | verified 2026-09-14 | stated purpose: reduce token-boundary bias; no numbers |
| Mistral-Nemo-Base-2407 | 12B | pretrain-stable | tokenizer; vocab_size | Tekken (tiktoken-based, >100 languages); 131,072 | Mistral blog "Tekken"; config.json ([[mistral-nemo]]) | verified 2026-09-14 | blog Figure 2 compression (image; corpus not described) |
| FineWeb2 ablation model | 1.46B | pretrain-stable | tokenizer; embedding size | Gemma tokenizer; 256,008 | arXiv:2506.20920v1 §3.1, App. A.4 Table 3 ([[fineweb-2]]) | verified 2026-09-14 | Table 2: lowest average fertility (2.10) among compared tokenizers without unknown tokens (mT5 2.03 and XGLM 1.76 lower but emit [UNK]) |
| Vocabulary-scaling runs | 2.87B non-vocab | pretrain-stable | V at 2.3e21 FLOPs; sequence; batch; LR | 32K vs 43K (predicted); 2,048; 512 sequences; AdamW 4e-4 → 4e-5 | arXiv:2407.13623v3 Table 3, App. A.7 ([[vocabulary-scaling-laws]]) | verified 2026-09-15 | Table 3: average 50.3 → 51.6; seeds not reported |
| Tokenizer-switch runs | 1.5B | mid-train (continued code pre-training, 5B-500B tokens) | new-row initialization; tokens to recover | Fast Vocabulary Transfer; ≥ 50B tokens | arXiv:2402.01035v2 §3.1, Fig. 6, Table 4 ([[tokenizer-domain-adaptation]]) | verified 2026-09-15 | Table 4: FVT 20.5% vs no FVT 18.4% HumanEval Pass@1 |
| transformers `resize_token_embeddings` | n/a (library default) | SFT (any stage that adds tokens) | new-row initialization | `mean_resizing=True` (old mean and covariance) | huggingface/transformers v4.46.0 modeling_utils.py L2080-2116 ([[new-embedding-initialization]]) | verified 2026-09-15 | framework default; KL bound in Hewitt (2021); no ablation reported |
| Dolma v1.6 web | n/a (data) | data filter (no §5.2 stage) | PII rule | regex email/IP/phone; ≤ 5 spans masked; > 5 spans document removed | arXiv:2402.00159v2 §5.3, App. I ([[dolma]]) | verified 2026-09-14 | §5.3: masking vs removal no effect in 1.2B ablations |
| Olmo 3 science PDFs | n/a (data) | data filter (no §5.2 stage) | PII classifier; removal | Gemma 3 12B on first page + Gemma 3 4B on first 5,000 chars + rules; 4.9% removed → 148M docs | arXiv:2512.13961 §3.4.2 ([[olmo-3]]) | verified 2026-09-15 | annotator-built taxonomy; no ablation reported |
| Llama 3 web | n/a (data) | data filter (no §5.2 stage) | PII rule | domain-level removal of high-PII sites | arXiv:2407.21783v3 §3.1.1 | not reported (detector, threshold, removal rate; checked §3.1 and §3.1.1, including the multilingual paragraph) | no ablation reported |

**Starting point for a small general-purpose run.** For a dense model between 2B and 8B parameters (the sizes covered by the rows above: 2.87B non-vocabulary parameters, OLMo 3 7B, Llama 3 8B, Gemma 2 2B-9B) trained on a multilingual and code mixture, the verified rows support: a byte-level BPE tokenizer trained on the model's own mixture, with the vocabulary padded to a multiple of 128 as in OLMo 3 7B (100,278 ids → 100,352 rows), and a vocabulary size checked against the vocabulary-scaling prediction for the model size (43K was better than 32K at 2.87B non-vocabulary parameters and 2.3e21 FLOPs on English data; Llama 3 uses 128,000 at 8B-405B and Gemma 2 uses 256,128 at 2B-27B). Keep embeddings out of weight decay only if under-trained token checks are run with output-row indicators, since input-row norms then stay at initialization. Keep chat special tokens out of pre-training and mid-training data, as OLMo 3 did after its microanneal. Apply a Dolma-style regex PII rule (mask ≤ 5 spans, remove otherwise) to web text, measure its precision per source type on a manual sample, and do not apply web-tuned phone rules to code. When adding tokens later, use mean initialization (transformers default) and plan at least the 50B tokens of continued training after which the 1.5B tokenizer-switch runs recovered their code scores.

## Generalization lens

**(a) What increases breadth.** A tokenizer trained on the model's mixture: the share of code and multilingual text in tokenizer training data set the compression of those domains ([[tokenizer-domain-adaptation]] Fig. 2), and GPT-NeoX's matched tokenizer had 10 confirmed under-trained tokens against 200 for GPT-J reusing GPT-2's ([[under-trained-tokens]] Table 1). Added non-English vocabulary: Llama 3's 28K tokens improved compression and downstream performance without affecting English tokenization ([[llama-3]] §3.2; downstream numbers not reported), although the added tokens also include under-trained entries ([[under-trained-tokens]] §3.2). A vocabulary sized for compute: +1.3 average points at 2.3e21 FLOPs ([[vocabulary-scaling-laws]] Table 3). Digit segmentation aligned with place values: 75.6% → 97.8% addition accuracy for GPT-3.5 ([[tokenization-counts-arithmetic]] Fig. 1).

**(b) What causes narrowing or forgetting.** High premiums: 15.05 for Shan and 11.70 for Burmese under cl100k_base leave under 10% of a context window's English content and multiply training compute per unit of content ([[tokenizer-language-unfairness]] Table 1, §5.3). Reusing a tokenizer on different data creates under-trained tokens ([[under-trained-tokens]] §3.2). Untrained special tokens in mid-training data break base-model answer formats ([[olmo-3]] §3.5.4). Switching a tokenizer without enough training costs target-domain accuracy before 50B tokens ([[tokenizer-domain-adaptation]] Fig. 6), and the effect on non-target abilities is unmeasured. PII rules with low precision in code (9% for phone numbers in The Stack) remove mostly non-PII strings ([[wimbd]] Table 5). Duplicated sequences are memorized more ([[quantifying-memorization]] §4.2); this was measured on generic training sequences, not on PII specifically.

**(c) How to measure it for this stage.** Per-language premium on a parallel corpus such as FLORES-200 and per-domain normalized sequence length ([[tokenizer-language-unfairness]] §3; [[tokenizer-domain-adaptation]] §2.1). Digit-segmentation tests with length-match and length-mismatch arithmetic ([[tokenization-counts-arithmetic]] §4.1). Under-trained token indicators after a short run ([[under-trained-tokens]] §2). Unigram-normalized loss when comparing vocabularies ([[vocabulary-scaling-laws]] Eq. 4). Extractable-memorization rate by duplicate bucket and prompt length ([[quantifying-memorization]] §3.2). Verbatim span tracing of evaluation answers ([[olmotrace]]). Known measurement errors: raw loss is not comparable across vocabularies ([[vocabulary-scaling-laws]] §2.2); exact-match tracing misses paraphrases and does not establish cause ([[olmotrace]] Limitations); regex PII counts have unmeasured recall ([[wimbd]] App. B.3.2); measured memorization is 2.1 times higher when a match anywhere in the corpus counts instead of the true continuation (32.6% vs 15.8% at 100 repeats, [[quantifying-memorization]] §4.4).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Comparing training loss across models with different tokenizers | A larger-vocabulary model appears worse at equal steps | Compare unigram-normalized loss or bits per character ([[vocabulary-scaling-laws]] §2.2) |
| Choosing vocabulary size from a smaller model's config | Optimum shifts with compute; gains missed at larger budgets | Recompute with the vocabulary scaling fit at the planned N_nv and FLOPs ([[vocabulary-scaling-laws]] Table 1) |
| Reusing a tokenizer trained on different data | Hundreds of confirmed under-trained tokens; garbled output on rare strings | Run embedding-norm and repetition-prompt checks ([[under-trained-tokens]] §2) |
| Ignoring per-language premium | Target-language documents truncated or under-represented in tokens per unit of content | Measure premium on FLORES-200 for every target language ([[tokenizer-language-unfairness]] §4) |
| Adding special tokens to mid-training data | Base model emits template tokens; GSM8K fell to 0 in the OLMo 3 microanneal | Emission rate of special tokens on plain prompts ([[olmo-3]] §3.5.4) |
| Zero or small-norm initialization of added rows | Samples dominated by new tokens before training | KL between pre- and post-resize distributions on held-out text; mean initialization ([[new-embedding-initialization]]) |
| Applying web PII regexes to code | Match count on code far above the PII count expected after precision adjustment (The Stack phone numbers: 45.4M matches at 9% precision) | Precision on 100 matches per source type ([[wimbd]] Table 5) |
| Treating deduplication as PII protection | Memorization of highly repeated sequences remains | Extractable fraction by duplicate bucket after deduplication ([[quantifying-memorization]] §5.2) |
| Reading a verbatim trace as proof of cause | Claims that a document "taught" a behavior | Traces show co-occurrence only ([[olmotrace]] Limitations); use ablation or influence methods (ch-12a) |
| Resuming with a changed batch size or mix | Fingerprint error, or silent reordering with `ignore_fingerprint_mismatch=True`; replayed sequences | Check that tokens_processed divides the new global batch; log dataset fingerprint per restart ([[olmo-core-olmo3-configs]] data_loader.py L451-477, L757-763) |
| Setting target ratios above available tokens | Configuration error, or unplanned repetition when the repetition cap is raised | Compute needed/available tokens per source before launch ([[olmo-core-olmo3-configs]] source_mixture.py L309-321) |

## Check your understanding

1. Using C ≈ 6(N_nv + V·d)·H·f(V), explain why the 43K and 32K runs in Table 3 of [[vocabulary-scaling-laws]] have equal FLOPs, and why the optimum vocabulary moves down when data is scarce.
2. A model is trained on equal content in English and Standard Arabic with cl100k_base. Explain how the 3.04 premium changes the Arabic token share, context usage, and training compute, and which of these a larger multilingual vocabulary can change.
3. Explain, with place-value alignment, why left-to-right 3-digit chunking fails mainly when the sum has more digits than the addends, and why adding extra tokens without changing the direction did not help.
4. Using the softmax gradient, explain why a token that is never a target ends with a very negative logit, and why this makes a zero-initialized new token dangerous. How does mean initialization bound the damage?
5. OLMo 3 found that a chat template in plain text cost 3.4 GSM8K points while the same template with special tokens cost 49.4. Explain the mechanism and state what measurement you would add before placing special tokens in any pre-training stage.
6. Dagan et al. show recovery of code accuracy after 50B tokens with a new tokenizer. Explain why this does not establish that a general model keeps its non-code abilities, and design the retention report for that case.
7. Quantifying Memorization finds deduplication reduces memorization for sequences repeated fewer than 35 times but not above about 100. Explain why, and what this implies for the order of PII masking and deduplication.
8. Explain why an infini-gram index and an OLMoTrace result depend on the tokenizer, and what must be rebuilt when a model's tokenizer is extended.

## Connections

- **Previous (dependency):** ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection. It produces the document pool that this chapter tokenizes, cleans of PII, and indexes.
- **Next:** ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic. Deduplication reduces the repetition that drives the memorization measured in §7.
- ch-00 — What General Capability Means and How It Is Measured (held-out suites used for retention reports).
- ch-04 — Sequence Packing, Loss Masking, and Chat Templates (special tokens for turns, thinking, and tool calls).
- ch-09 — Pretraining Data Composition and Capability Coverage (language and code shares that the tokenizer must encode).
- ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining (full treatment of memorization scaling and influence functions).
- ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale (how target ratios are chosen).
- ch-13a — Multilingual Coverage and Vocabulary as Capability Axes (per-language budgets and multilingual vocabulary costs).
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination (value of repeated tokens from §8; decontamination records).
- ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL (the OLMo 3 special-token result belongs to this stage).
- ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining (retention during tokenizer extension).
- ch-48 — Contamination Detection and Its Effect on Reported Scores (search-based detection built on §6 tools).

## Sources

- [[vocabulary-scaling-laws]] — compression fit f(V), compute formula, unigram-normalized loss, optimal-vocabulary predictions, 2.87B comparisons, limits.
- [[tokenizer-language-unfairness]] — premium definition, FLORES-200 premiums for English-centric, multilingual, and byte-level tokenizers, latency and context effects.
- [[tokenization-counts-arithmetic]] — number tokenization strategies, L2R versus R2L addition accuracy, length-mismatch errors.
- [[under-trained-tokens]] — definitions, embedding indicators, verification protocol, confirmed counts across models, recommendations.
- [[tokenizer-domain-adaptation]] — normalized sequence length, tokenizer training-data effects, tokenizer switching, FVT and extension results, vocabulary-size null result for code.
- [[new-embedding-initialization]] — KL derivation for added tokens, mean-initialization bound, GPT-2 demonstration, transformers `mean_resizing` default.
- [[quantifying-memorization]] — extractability definition, scaling with size, duplication, and context, deduplicated-model results, auditing statement.
- [[extracting-training-data]] — extraction attack on GPT-2 XL, 604 memorized examples, PII categories, single-document memorization.
- [[wimbd]] — count and search tooling, duplicate and contamination findings, regex PII counts and precision.
- [[infini-gram]] — suffix-array index layout, ∞-gram estimate, index size and latency, tokenizer-specific indexes.
- [[olmotrace]] — verbatim span tracing over OLMo 2 training data, pipeline, latency and stage statistics, limitations, PII display filter.
- [[olmo-3]] — tokenizer lineage, special-token microanneal, tool-call tokens, PDF PII filter, decontamination.
- [[olmo-core-olmo3-configs]] — padded vocabulary, OLMo 3 7B scripts, source mixture code, data loader resume state.
- [[dolma]] — regex PII masking and removal rule, secret removal in code, contamination audit.
- [[fineweb-2]] — tokenizer choice by fertility for multilingual ablations.
- [[llama-3]] — 128K tokenizer construction and compression, domain-level PII filtering.
- [[gemma-2]] — 256K SentencePiece tokenizer with split digits and byte-level encodings; memorization rate.
- [[deepseek-v3]] — 128K byte-level BPE and random splitting of combined punctuation tokens.
- [[mistral-nemo]] — Tekken tokenizer vocabulary and vendor compression claims.
- [[deduplicating-training-data]] — memorized-token rates before and after deduplication.
- [[atlas-multilingual-scaling-laws]] — compute cost of multilingual vocabularies (forward link to ch-13a).
