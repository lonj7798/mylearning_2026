<!-- chapter: ch-12a
     track: pretraining
     kind: content
     title: Memorization, Knowledge Acquisition, and Generalization During Pretraining
     deps: [ch-12]
     sources: [[quantifying-memorization]], [[how-much-do-lms-memorize]], [[memorization-without-overfitting]], [[physics-of-lm-3]], [[physics-of-lm-3-recipe]], [[physics-of-lm-3-1-knowledge-storage]], [[physics-of-lm-3-2-knowledge-manipulation]], [[reversal-curse]], [[factual-knowledge-acquisition-pretraining]], [[long-tail-knowledge]], [[influence-functions-generalization]], [[repeated-data-scaling]], [[data-constrained-scaling]], [[deduplicating-training-data]], [[paloma]]
     figures: figures/knowledge-acquisition-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining

> **Core insight.** A model can reproduce a training sentence word for word and still be unable to answer a question about the fact in it. In controlled experiments on synthetic biographies, a 124M GPT2 trained on one fixed biography per person answered held-out questions at 9.7% after QA fine-tuning, and at 96.6% when each person had five differently worded, shuffled biographies ([[physics-of-lm-3-1-knowledge-storage]] Fig. 3). Facts trained in one direction are not retrieved in the other ([[reversal-curse]], [[physics-of-lm-3-2-knowledge-manipulation]]), and BLOOM-176B answers TriviaQA questions with about 10 relevant pretraining documents at 25% against above 55% with 10^4 ([[long-tail-knowledge]] §3.1). Repeating 0.1% of the data 100 times degraded an 800M model to the performance of a 400M model ([[repeated-data-scaling]] Abstract).
>
> **Guideline.** When a fact matters and appears in few forms, add rewritten versions (different wording, sentence order, and direction) to pretraining data rather than repeating the same string, because rewriting raised held-out extraction from 9.7% to 96.6% in [[physics-of-lm-3-1-knowledge-storage]] and paraphrased injections were forgotten more slowly than duplicated ones at OLMo-7B ([[factual-knowledge-acquisition-pretraining]] Table 2). When upweighting a high-quality subset, keep its repeat count outside the degradation band E = k·N^α that [[repeated-data-scaling]] fitted at 100B training tokens and monitor a copying evaluation, because at the worst repeat count 3% repeated data reduced effective model size 3× on copying and at most 1.15× on test loss. Otherwise, when repetition cannot be avoided, repeat the whole dataset uniformly for up to about 4 epochs ([[data-constrained-scaling]]). Evaluate knowledge with held-out entities, paraphrased and reversed questions, and accuracy stratified by how often the fact occurs in the corpus.

## Why this chapter matters for a general-purpose model

A general-purpose model must answer questions about facts in phrasings, directions, and combinations that its training text did not contain. The measurable problem is the gap between two quantities: how much of the training data a model can reproduce, and how much of the knowledge in that data it can use under a new prompt. Pretraining is where this gap is decided. [[physics-of-lm-3-1-knowledge-storage]] reports that when pretraining stores a fact in a form that cannot be extracted, no QA fine-tuning setting it tried recovered it (Result 2), so later stages (SFT, preference optimization, RL) inherit what pretraining data design produced.

This chapter sits between deduplication (ch-12) and domain mixing (ch-13). Deduplication decides how often a string repeats; mixing decides how often a domain appears. Both are exposure decisions, and this chapter gives the evidence on what exposure count and exposure form do to memorization, extraction, and forgetting. For the course goal it answers three questions:

1. Breadth: which data forms make stored facts usable for unseen questions (§5, §6, §9).
2. Narrowing: which data patterns shift a model toward memorized strings at the cost of general behavior such as copying and in-context learning (§2, §9).
3. Measurement: how to tell memorization from generalization, and which measurements undercount or overcount each (§1, §2, §3, §8).

## §1 Three different quantities: verbatim memorization, stored information, and extractable knowledge

**Definitions.** Four measurements appear in this chapter. They measure different things and should not be compared as if they were one number.

1. **Extractable memorization** ([[quantifying-memorization]] Definition 3.1). A string s is extractable with k tokens of context if some length-k prefix p with [p || s] in the training data makes the model produce s under greedy decoding. Example: the training text contains "My phone number is 555-6789", and the 4-word prefix "My phone number is" yields "555-6789".
2. **Exact memorization** ([[memorization-without-overfitting]] Definition 1). A training context (s, y) is memorized if argmax f(s) = y. The fraction over training contexts is M(f).
3. **Unintended memorization** ([[how-much-do-lms-memorize]] §2). The information a model holds about specific training samples beyond what a reference model of the data distribution already predicts.
4. **Knowledge extraction** ([[physics-of-lm-3-1-knowledge-storage]] §1). Accuracy on questions about entities whose questions never appeared in training, after the model was trained on text about those entities.

**The problem.** A high value on (1) or (2) does not imply a high value on (4). A model can also emit a training string because the string is predictable, not because it was stored: GPT-2, which was not trained on the Pile, correctly completes about 6% of the Pile evaluation set, against 40% for GPT-Neo 1.3B, which was ([[quantifying-memorization]] §4.1).

**Formula (exact memorization).**

M(f) = (1/|C|) · Σ_{(s,y)∈C} 1{argmax f(s) = y}

- C: the set of training contexts; s: the input prefix; y: the ground-truth next token; f(s): the model's logits; 1{·}: 1 if the condition holds, else 0.

Worked example: five training contexts, where the argmax token matches the ground truth in three. M(f) = 3/5 = 0.6.

**Formula (unintended memorization, estimated with likelihoods).** [[how-much-do-lms-memorize]] defines mem_U(x, θ, θ̂) = H^K(x | θ) − H^K(x | θ, θ̂) and estimates it as

mem_U(x) ≈ −log2 p(x | θ) − ( −log2 max{ p(x | θ̂), p(x | θ) } )

- x: one training sample; θ: a reference model that approximates the data distribution; θ̂: the trained model; H^K: Kolmogorov complexity (shortest description length), approximated by code length under arithmetic coding, which equals negative log-likelihood (§2.3).

Worked example: the reference model assigns p(x | θ) = 2^−40, so x costs 40 bits. If the trained model assigns 2^−10, mem_U = 40 − 10 = 30 bits: the trained model holds 30 bits about x that the distribution does not explain. If the trained model assigns 2^−50, the max selects the reference, and mem_U = 40 − 40 = 0 bits.

**Implication.** A memorization audit, a capacity estimate, and a knowledge evaluation answer different questions. The rest of the chapter states which quantity each result measures.

## §2 How verbatim memorization scales with model size, duplication, and context length

**Problem.** Deduplication decisions (ch-12) need to know how much duplication raises memorization and whether larger models make it worse.

**Mechanism of the measurement** ([[quantifying-memorization]] §3.2):
1. Sample training sequences with lengths ℓ ∈ {50, 100, ..., 500} and, for duplication analysis, 1,000 sequences per duplicate-count bucket between 2^(n/4) and 2^((n+1)/4).
2. Prompt the model with the first ℓ − 50 tokens.
3. Decode greedily and mark the sequence extractable if the next 50 tokens match exactly.
4. Average the extractable fraction over lengths, and repeat for each model size.

**Formula.** The paper reports log-linear fits. For model size on the duplicate-normalized sample:

frac(N) ≈ c + β · log10 N, with β = 0.19 (19 percentage points per tenfold increase in parameters, R² = 99.8%; §4.1)

- frac: fraction of sampled sequences extractable; N: parameter count; c: intercept (not reported in the text).

Worked example (derived): from GPT-Neo 125M to 6B, log10(6×10^9 / 1.25×10^8) = log10(48) = 1.68, and 1.68 × 19 ≈ 32 percentage points more extractable sequences on this sample.

**Evidence.**
- Model size: log-linear increase for GPT-Neo 125M-6B on the Pile; T5 and OPT show the same direction ([[quantifying-memorization]] §4.1, §5.1, §5.3). Larger models also reach a memorization threshold in fewer passes ([[memorization-without-overfitting]] §4). **Replicated.**
- Duplication: log-linear increase over 2 to 900 duplicates, and memorization still occurs for sequences with few duplicates ([[quantifying-memorization]] §4.2).
- Context: 33% of sampled sequences are extractable from the 6B model with 50 tokens of context, against 65% with 450 tokens (§4.3). The authors call this discoverability.
- Uniform sample: at least 1% of the Pile is extractable by GPT-J 6B but not by GPT-2 XL (§4.4).
- Deduplicated training: for 1.5B models trained on C4, sequences repeated below 35 times are memorized at 1.2% after exact-substring deduplication against 3.6% without; sequences repeated at least 408 times in the original C4 show no benefit (§5.2). [[deduplicating-training-data]] reports the unprompted side: 1.926% of generated tokens belong to copied 50-token sequences for the original-C4 model, against 0.138% after exact-substring deduplication (Table 4).

**Conditions and limits.** Greedy decoding and exact match give lower bounds: beam search with 100 beams adds less than 2 percentage points on average (maximum 5.6), and at 100 repetitions 32.6% of outputs occur somewhere in the training set while 15.8% match the true continuation ([[quantifying-memorization]] §4.4). Masked language models memorize about an order of magnitude less (T5-XL 3.5% against GPT-Neo 2.7B 53.6% at 100 repeats, §5.1). OPT-66B memorizes a smaller fraction of the Pile than GPT-Neo 125M; the authors cannot separate data curation from distribution shift as the cause (§5.3).

**Implication for a general-purpose model.** The authors conclude that when training data is skewed by duplicates, larger models are more likely to learn "unintended dataset peculiarities" (§6). This course reads that as a reason for deduplication to become stricter as model size grows (**Interpretation**). Because extraction depends on prefix length, the authors state that auditing requires prompting with training data (§4.3).

## §3 Capacity: bits per parameter and what happens when data exceeds it

**Definition.** Capacity is the maximum unintended memorization a trained model reaches across dataset sizes ([[how-much-do-lms-memorize]] Definition 5).

**Problem.** Whether a model memorizes or generalizes depends on whether the information in the data fits in its parameters.

**Mechanism** ([[how-much-do-lms-memorize]] §3.2):
1. Build datasets of uniformly random token sequences, where no generalization is possible and the information content is known exactly.
2. Train GPT-2-architecture models (100K-20M parameters, 10^6 steps, batch 2,048) to saturation on datasets of increasing size.
3. Measure memorized bits; the plateau across dataset sizes is the capacity.

**Formula.** H(dataset) = N_seq · S · log2 V, and α = capacity / parameters.
- N_seq: number of sequences; S: tokens per sequence (64); V: vocabulary size (2,048); α: bits per parameter.

Worked example (derived from Table 1): each sequence carries 64 × log2 2048 = 64 × 11 = 704 bits. The 8-layer, d_model 256 model has 6.86 × 10^6 parameters and a measured capacity of 2.51 × 10^7 bits (α = 3.65). It can store about 2.51 × 10^7 / 704 ≈ 35,650 random sequences.

**Evidence.**
- GPT-style models store about 3.6 bits per parameter (Abstract); Table 1 means are 3.51 in bfloat16 and 3.83 in float32, so doubling precision adds about 9% capacity ([[how-much-do-lms-memorize]] §3.2).
- On deduplicated FineWeb text, double descent (test loss rising and then falling as data grows) begins when dataset information exceeds capacity, and with enough data the extraction rate on training sequences converges to the extraction rate on test sequences: "all successful training data extraction is attributable to generalization" (§4, Fig. 3, Fig. 8). **Result (single study).**
- For facts, [[physics-of-lm-3]] measures a lower bound of 2 bits of knowledge per parameter at 1,000 exposures per fact and about 1 bit at 100 exposures, for GPT2 models up to 0.5B on synthetic biographies (Results 1, 4). The two estimates measure different quantities (raw random bits against extractable knowledge bits), so they do not contradict each other (**Interpretation**).
- Membership inference (deciding whether a sample was in training) follows a sigmoid in capacity and dataset size; the fitted law implies F1 of 0.5, equal to chance, for models trained with 10^2 or more tokens per parameter ([[how-much-do-lms-memorize]] §5.2.2, Table 2).

**Conditions and limits.** Models up to 1.5B in the validation runs; 64-token sequences; perfectly deduplicated data. Natural corpora with duplicates are not covered.

**Implication.** When the information in a deduplicated corpus exceeds capacity, the model cannot store most samples individually, and the authors propose that sharing information across samples is what produces generalization (§4, **Interpretation**). Duplicated sequences, which §2 shows are extracted more often, fall outside this regime because their information is counted once but seen many times (**Interpretation**, this course).

## §4 Training dynamics: memorization before overfitting, and forgetting after exposure

**Problem.** A fact is seen at discrete points during training. The retained amount depends on how much each exposure adds and how fast the gain decays.

**Memorization without overfitting.** [[memorization-without-overfitting]] defines T(N, τ), the number of passes a model with N parameters needs for M(f) ≥ τ.
- Larger models memorize faster: T(N, 0.9) decreases monotonically with N for causal LM on WikiText-103, and the effect remains at a fixed learning rate (§4, §4.2, Fig. 1, Fig. 5).
- Larger models memorize a larger fraction of training data before validation perplexity starts to rise (§4.2, Fig. 4). The authors conclude that overfitting alone does not explain these memorization dynamics.
- A held-out batch seen once is forgotten quickly at first, and memorization then approaches a floor, the forgetting baseline, which rises with model size (§5, Fig. 8). Repeated injection raises the baseline (differences of order 10^-2); spaced repetition changes it by order 10^-3 (Fig. 10, 125M).

**Micro-acquisition and forgetting.** [[factual-knowledge-acquisition-pretraining]] resumes OLMo pretraining and injects fictional facts every 100 steps.

1. Each exposure raises the log-probability of probe answers, and the gain then decays (§4.1, Fig. 2). The gain is read at its maximum within a 50-step window, because AdamW momentum spreads one batch's update over several steps (§3).
2. Probes test memorization (the same sentence), semantic generalization (a paraphrase), and compositional generalization (a fact that combines sentences).
3. Effectivity, the immediate gain per exposure, did not improve for checkpoints trained on 177B, 500B, or 1.5T tokens, and improved from 1B to 7B (§4.2, Fig. 3).
4. Retainability R, the fraction of the gain left after t further steps, falls linearly in log t (§4.3).

**Formula** (Definition 3, Eq. 4):

R(q, t) = [ℓ(q; θ_{t_LAM+t}) − ℓ(q; θ_{t_pre})] / [ℓ(q; θ_{t_LAM}) − ℓ(q; θ_{t_pre})],  ΔR ≈ −a · log(t2 / t1)

- ℓ(q; θ): log-probability of probe q's target span under parameters θ; t_pre: last step before the first exposure; t_LAM: step of the local maximum after the last exposure; a: decay constant (larger means faster forgetting).

Worked example. The fitted line in token units is R(T) = a · (x0 − log10 T), with a from Table 2 and the x-intercept x0 from Table 6 (reported in log(tokens); the base-10 reading matches the token axis of Figure 5, derived). At the 500B-token checkpoint with batch 2,048 sequences, memorization probes:
- Duplication: a = 0.25, x0 = 11.02. After 10^10 tokens, R = 0.25 × (11.02 − 10) = 0.255.
- Paraphrase: a = 0.21, x0 = 11.37. After 10^10 tokens, R = 0.21 × (11.37 − 10) = 0.288.
- Batch 128 sequences, duplication: a = 0.31, x0 = 9.45 (Tables 10-11). After 10^10 tokens, 0.31 × (9.45 − 10) < 0, so R = 0.

**Evidence.** Duplicated injections give larger immediate gains but faster forgetting, ending at a similar level to paraphrased injections after 2,000 steps (§4.1). Decay constants are lower for paraphrase than duplication in 7 of the 9 stage-and-depth cells of Table 2. With batch 128 instead of 2,048, x-intercepts fall "by dozens of times" (Tables 6 and 11: 11.02 against 9.45 for duplicated memorization at 500B, a factor of about 37, derived). **Result (single study)**, one model family and fictional facts.

**Conditions and limits.** Probes are scored by log-probability, not by generated answers (App. A). Batch size and learning rate were each compared at two values only. The authors state that the learnability threshold may differ from the x-intercepts (footnote 7).

The figure [figures/knowledge-acquisition-explorer.html](figures/knowledge-acquisition-explorer.html) (panel B) plots these fitted lines for every checkpoint, probe type, and batch size, and lets the reader read R at a chosen number of tokens since the last exposure.

**Implication.** In this model of acquisition, a fact is learned only if the gain from each exposure is not lost before the next exposure (**Interpretation**, [[factual-knowledge-acquisition-pretraining]] §4.4). Larger batches and paraphrased exposures lengthened the retention horizon in this setting.

## §5 Storage versus extraction: why rephrasings matter

**Definitions.** An exposure is one occurrence of a fact in training. Knowledge augmentation is rewriting the same fact in varied forms: multiM (M differently worded biographies per person), permute (shuffled sentence order), fullname (pronouns replaced by the full name) ([[physics-of-lm-3-1-knowledge-storage]] §4.2).

**Problem.** Web text states popular facts in many forms and rare facts in few. The question is whether form, not only count, decides extractability.

**Mechanism of the experiment** ([[physics-of-lm-3-1-knowledge-storage]] §2, App. C-D):
1. Generate biographies for N = 100,000 synthetic people with six attributes (birth date, birth city, university, major, company, company city).
2. Pretrain GPT2 (124M, rotary embedding) from scratch on the biographies: 80,000 steps, batch 96, 512-token windows.
3. Fine-tune on QA pairs for half of the people (P_train).
4. Measure exact-match QA accuracy on the other half (P_test), whose questions were never trained.
5. Probe the hidden states with linear classifiers to locate where each attribute is stored.

**Evidence** (mean accuracy over six attributes on P_test, bioS data, Fig. 3; the majority-guess baseline is 2.7%). bioS is the template-generated biography set; bioR is the set of biographies written by Llama (§2).

| Pretraining data | BIO pretrain, then QA fine-tune | Mixed training (BIO + QA) |
|---|---|---|
| single (one biography, fixed order) | 9.7 | 86.6 |
| single + fullname | 48.9 | 85.9 |
| single + permute1 | 4.4 | 82.5 |
| single + permute5 | 70.0 | 93.7 |
| multi5 | 41.0 | 91.8 |
| multi5 + permute | 96.6 | 95.5 |

The figure [figures/knowledge-acquisition-explorer.html](figures/knowledge-acquisition-explorer.html) (panel A) plots nine of the 17 augmentation rows of Figure 3 (the six above plus multi2, multi5 + fullname, and multi5 + permute + fullname) for both training procedures, with hover values and a table view.

Worked example (derived): P_test has 50,000 people × 6 questions = 300,000 questions. At 9.7%, about 29,100 are answered; at 96.6%, about 289,800.

- Result 2: with one biography per person, next-token accuracy on attribute tokens exceeds 99%, yet held-out QA accuracy is near zero for every fine-tuning setting tried (the 9.7% mean in Fig. 3 comes mostly from birth date, the first attribute, at 33.5%), including a 682M model with 1,350 exposures per person (§4.1, Fig. 2).
- Result 4: in bioS single, a probe predicts the company name at about 2% until the token right before it, where accuracy reaches 100%; with multi5 + permute, all six attributes are predictable at nearly 100% from the position right after the name (§5.1, Fig. 5). The authors' explanation is that without augmentation the model stores an attribute as a continuation of the preceding sentence rather than as a property of the name (**Interpretation**).
- Result 6: adding 100,000 augmented "celebrity" people raised accuracy on unaugmented "minority" people from 4.4% to 86.8% (bioS) and from 10.0% to 76.3% (Llama-written bioR), without changing the minority data or using minority QAs (§6, Fig. 8). Replacing celebrity data with WikiBook text did not help (Remark 6.1).
- Result 1: mixed training (QA data inside pretraining at a 2:8 BIO-to-QA token ratio) reached 86.6% even on bioS single (§3).
- Corroboration: [[physics-of-lm-3]] reports that 1,000 exposures of one fixed biography store slightly less than 1,000 differently templated biographies, and that low-diversity data can be memorized but is nearly 0% extractable (App. A.3). [[factual-knowledge-acquisition-pretraining]] finds slower forgetting for paraphrased exposures (§4). Status: **Result (single study, controlled synthetic data)** with partial corroboration.

**Conditions and limits.** Synthetic biographies with six attributes; models up to 682M in this paper (a follow-up with 1B models and N = 20M is reported as confirming, footnote). English-to-French translation raised accuracy to about 40% (footnote). Encoder-only masked-LM models did not store knowledge for extraction except for single-word attributes (Result 7).

**Implication.** For a general model, the variety of forms in which a fact appears during pretraining decides whether later stages can use it. Rare entities benefit when other entities of the same type appear in varied forms: 100,000 augmented people raised accuracy on 100,000 unaugmented people from 4.4% to 86.8% (Result 6).

## §6 Manipulation and direction: classification, inverse search, and the reversal curse

**Definitions.** Knowledge manipulation is answering a function of a stored fact (for example, whether a birth month is even). Inverse search is recovering the entity from its attribute. The reversal curse is the failure of a model trained on "A is B" to assign higher likelihood to A given B than to a random name ([[reversal-curse]] §1).

**Problem.** A general model is asked about facts in directions and transformations the training text did not state.

**Evidence on manipulation** ([[physics-of-lm-3-2-knowledge-manipulation]], models pretrained on bioS multi5 + permute, which extract birth dates at nearly 100%):
- Even-or-odd birth month without chain of thought needs 10,000 training samples to reach 75% held-out accuracy; ranking months needs 50,000 samples for 85% (Result 3).
- Chain-of-thought hints (writing the attribute before the answer) help only when generated at test time. For "major mod 5" with 10,000 training people: 23.6% without hints (chance 20%), 86.4% when the model writes the hint, 24.1% when trained with hints but answering without (Fig. 4; Result 4).

Worked example (Footnote 14): with hint accuracy 91.0% and a random guess of 50% when the hint is wrong, predicted accuracy is 0.910 + (1 − 0.910) × 0.50 = 0.955; the measured accuracy is 94.2%. Accuracy with hints is close to what the hint-generation accuracy predicts, so the remaining errors come mainly from retrieving the attribute (**Interpretation**).

**Evidence on direction.**
- Inverse search accuracy is near zero for all 16 non-reversed bioS datasets under QA fine-tuning and mixed training, with full-name accuracy 0.0 in every non-reversed fine-tuning row; accuracy rises only when the name appears after the attributes in pretraining text ([[physics-of-lm-3-2-knowledge-manipulation]] Result 7, Fig. 6).
- GPT-3-175B fine-tuned on fictitious facts: DescriptionToName 96.7% in the trained direction and 0.1% reversed; NameToDescription 50.0% and 0.0% ([[reversal-curse]] Table 1). Each fact had 30 paraphrases in the trained order. Across 20 learning-rate and batch-size settings on GPT-3-350M, reversed accuracy stayed between 0.0% and 0.3% (App. B.2, Fig. 7). Log-probability of the correct name did not differ from a random name at 350M-175B (Fig. 4).
- GPT-4 named a celebrity's parent 79% of the time and the celebrity from the parent 33% (§2.2; training data unknown, so tentative).
- Influence functions on 810M-52B pretrained models show that training sequences influence a completion only when phrases related to the prompt appear before phrases related to the completion; flipping the order decays influence to near zero ([[influence-functions-generalization]] Abstract, §5.3.4).
- Status: **Replicated** by three groups with fine-tuning, training from scratch, and influence analysis.

**Conditions and limits.** Given "A is B" in context, models deduce "B is A" ([[reversal-curse]] App. B.6); the failure is in learning from training text. Allen-Zhu and Li attribute it to left-to-right autoregressive training (**Interpretation**, [[physics-of-lm-3-2-knowledge-manipulation]] §5).

**Implication.** Bidirectional access to knowledge must be present in the data. [[physics-of-lm-3-2-knowledge-manipulation]] Result 9 lists reversed restatements (for example through a rewrite prompt), retrieval, and line numbers as mitigations.

## §7 Long-tail knowledge: dependence on how many documents contain a fact

**Definition.** A relevant document for a question-answer pair is a pretraining document containing both the salient question entity and the answer entity, found by entity linking ([[long-tail-knowledge]] §2).

**Problem.** Average QA accuracy hides whether the model knows rare facts.

**Mechanism** ([[long-tail-knowledge]] §2-§3):
1. Entity-link the pretraining corpus (The Pile, ROOTS, C4, OpenWebText, Wikipedia).
2. Entity-link TriviaQA and Natural Questions; count documents where question and answer entities co-occur.
3. Bin questions by log10 count and measure 4-shot exact-match accuracy per bin.
4. Retrain a 4.8B model on C4 without the relevant documents of sampled questions to test causality.

**Evidence.**
- BLOOM-176B TriviaQA accuracy rises from 25% to above 55% as relevant documents increase from 10^1 to 10^4 (§3.1).
- In the counterfactual retraining (about 30% of C4 removed), the accuracy loss grows with the question's original relevant-document count (§3.2, Fig. 5).
- For Natural Questions items with fewer than 100 relevant documents, accuracy is log-linear in parameters (R² = 0.98), and extrapolation needs over 10^18 parameters to match a supervised baseline or humans (§4.2, Fig. 6).
- Relevant-document counts correlate across corpora at Spearman 0.87-0.97 (Table 1); the authors use this to argue that diversifying pretraining sources would give limited benefit for rare facts (§4.1).
- Retrieval augmentation raises accuracy most on rare questions (§4.3, Fig. 9).

**Connection to dynamics (Interpretation, derived).** [[factual-knowledge-acquisition-pretraining]] §4.4 hypothesizes a learnability threshold: a fact whose encounter interval exceeds it is forgotten before it accumulates. Worked example under two assumptions (the fact's documents are evenly spaced; the threshold is near the Table 6 x-intercept, which footnote 7 says need not hold): a fact in 100 documents of a 2T-token run is seen every 2×10^12 / 100 = 2×10^10 tokens (10^10.3). That interval is shorter than 10^11.02 (duplicated memorization, batch 4M tokens) but longer than 10^9.45 (batch 128 sequences). With 10 documents the interval is 2×10^11, longer than both.

**Conditions and limits.** Entity linking has about 60% precision for TriviaQA (§2.3). Human accuracy with background text is highest on rare questions, which the authors use to argue that rare questions are not harder in themselves (§3.1, Fig. 7).

**Implication.** A general model's factual accuracy depends on document frequency in the corpus, correlationally at 125M-176B (GPT-Neo and BLOOM) and causally at 4.8B (§3.1-§3.2). Evaluation should report accuracy by frequency bin, and data design for rare facts should add exposures in varied forms (§5) or rely on retrieval.

## §8 Tracing a behavior to training documents: influence functions

**Definition.** An influence function estimates how a measurement of the model would change if one training sequence were added to the training set ([[influence-functions-generalization]] §2.1).

**Problem.** Deciding whether a capability comes from memorized passages or from many related documents requires attribution to training data.

**Formula** (Eq. 25):

I_f(z_m) ≈ −∇_θ f(θ_s)ᵀ (G + λI)⁻¹ ∇_θ L(z_m, θ_s)

- z_m: a candidate training sequence; θ_s: final pretrained weights; f(θ) = log p(z_c | z_p; θ): log-probability of a completion z_c given prompt z_p; L: training loss; G: Gauss-Newton Hessian, approximated with EK-FAC (eigenvalue-corrected Kronecker-factored curvature); λ: damping. Positive I_f means adding z_m raises the completion's log-probability.

**Mechanism.**
1. Fit the EK-FAC approximation of G once per model (MLP parameters only).
2. For each query, compute v = (G + λI)⁻¹ ∇_θ f once.
3. For each candidate sequence, compute the gradient ∇_θ L(z_m) and the score −vᵀ∇_θ L(z_m).
4. Rank candidates, using TF-IDF filtering (keeping candidates with high term overlap with the query) or query batching to reduce cost (§3.2).

Worked example with two parameters: ∇f = (1, 1) and G + λI = diag(2, 0.5).
- Sequence 1 with ∇L = (−1, −0.5): (G + λI)⁻¹∇L = (−0.5, −1); dot with ∇f = −1.5; I_f = +1.5. Training on it raises the completion's log-probability.
- Sequence 2 with ∇L = (−1, +0.5): (G + λI)⁻¹∇L = (−0.5, +1); dot = +0.5; I_f = −0.5. Training on it lowers the completion's log-probability.
The inverse curvature multiplies the second coordinate, which has low curvature, by 2 and the first by 0.5, so the second coordinate decides the sign.

**Evidence** ([[influence-functions-generalization]], 810M-52B pretrained models):
- Influence is spread over many sequences, with a power-law tail; the authors found no single sequence dominating typical assistant outputs, while famous-passage queries returned the exact passages (§5.2.1, §5.3.3).
- Top influential sequences for the 810M model share tokens with the query; for the 52B model they share themes with little token overlap (§5.3.1, Fig. 1).
- Influence of English sequences on Korean and Turkish translations of the same query is negligible at 810M and increases with model size (§5.3.1, Fig. 16).
- Status: **Result (single study)** on private models.

**Conditions and limits.** Influence functions approximate a local response (the proximal Bregman response function) rather than full retraining; only MLP parameters and a fraction of the corpus were covered; fine-tuning was not studied (§1).

**Implication.** Cross-lingual and abstract transfer, which a general model depends on, appear at larger scale in this analysis. Influence scans are a tool for checking whether a benchmark behavior is supported by many related documents or by a few near-copies.

## §9 Consequences for pretraining data design

**Non-uniform repetition.** [[repeated-data-scaling]] trains models for 100B tokens where a fixed share of tokens repeats a small subset.
- An 800M model with 10% of tokens from 100 repeats of 0.1% of the data degraded to the performance of a 400M model (Abstract; §2 states "nearly" 340M).
- With 3% repeated data at the worst repeat count, copying (loss on a paragraph repeated 11 times) degraded to a 3× smaller effective model while test loss degraded at most 1.15× (§2, Fig. 5). The paper connects copying to induction heads (attention circuits that complete repeated patterns; ch-08b), and their prefix-matching scores were also preferentially degraded at low repeated fractions (Fig. 6).
- The peak damage coincides with training loss on the repeated subset approaching zero (§2, Fig. 2), which is a diagnostic.

Worked example (§5.1): an 800M model has loss about 2.0 nats/token and a 400M model about 2.2. If memorizing the repeated 10% drives its loss to 0 while the other 90% degrades to 2.2, the average is 0.9 × 2.2 + 0.1 × 0 = 1.98, below 2.0, so the training objective prefers memorization.

**Formula (degradation band, §2).** E = k · N^α, with left boundary k = 4.2×10^6, α = −0.56 and right boundary k = 5.1×10^7, α = −0.50.
- E: repeated epochs over the subset; N: parameters; the band marks at least 50% of the maximum degradation at 100B training tokens.

Worked example (derived): at N = 10^9, left E = 4.2×10^6 × 10^(−5.04) ≈ 38 and right E = 5.1×10^7 × 10^(−4.5) ≈ 1,613. With 10% of 100B tokens repeated, that band corresponds to repeated subsets of 10^10 / 1,613 ≈ 6.2M to 10^10 / 38 ≈ 261M unique tokens. The figure [figures/knowledge-acquisition-explorer.html](figures/knowledge-acquisition-explorer.html) (panel C) computes the band for any model size and repeated share.

**Uniform repetition.** [[data-constrained-scaling]] repeats the whole dataset: up to 4 epochs changed loss negligibly (an 8.7B model at 4 epochs ended 0.5% above 1 epoch in validation loss). Both results hold together: the harm in [[repeated-data-scaling]] comes from a subset (for example 0.1% of the data) repeated 100 or more times while the rest is unique, and [[data-constrained-scaling]] repeats every token the same number of times (**Interpretation**).

**Negative marginal value.** Duplicated strings and heavily repeated subsets are samples with negative marginal value (the first of the four meanings of "negative" used in this course, taught in ch-43a: they lower performance when used as positive targets). This stage removes or rewrites them; it does not use negative gradients.

**Decision rules.**
- When a fact or document type is rare and important, generate paraphrased, reordered, and reversed restatements and add them to pretraining data, because augmentation raised held-out extraction from 9.7% to 96.6% ([[physics-of-lm-3-1-knowledge-storage]]) and inverse search rose above near-zero only when attributes preceded the name in pretraining text ([[physics-of-lm-3-2-knowledge-manipulation]] Fig. 6). Otherwise, rely on retrieval at inference for the long tail ([[long-tail-knowledge]] §4.3).
- When QA-shaped or instruction data is available, include some of it in pretraining rather than only in SFT, because mixed training reached 86.6% where pretrain-then-fine-tune reached 9.7% on the same data ([[physics-of-lm-3-1-knowledge-storage]] Result 1).
- When upweighting a subset, compute its repeat count and compare with the E = k·N^α band; monitor training loss on the subset and a copying evaluation, because damage peaked when subset loss neared zero ([[repeated-data-scaling]]).
- When choosing batch size for knowledge-heavy continued pretraining, prefer the larger batch at equal tokens, because batch 128 shortened the x-intercept of retainability by a factor of about 37 relative to 2,048 at OLMo-7B, at the 500B-token checkpoint, with only these two batch sizes tested and the optimizer state re-initialized for the small-batch run ([[factual-knowledge-acquisition-pretraining]] §4.3, Tables 6, 11). Otherwise, when memory forces a small batch, shorten the interval between exposures of the target facts, which follows from the learnability-threshold hypothesis rather than from a tested intervention (**Interpretation**, [[factual-knowledge-acquisition-pretraining]] §4.4).

## Recipe

Rows marked 2026-09-15 were read at the stated locus in the primary PDF for this chapter; rows marked 2026-09-14 come from verified library cards. These are controlled research runs, not released general-model recipes; ch-14a collects production pretraining recipes.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT2-12-12 with rotary embedding, bioS | 124M | pretrain-stable | optimizer; LR; warmup; decay; batch; steps; context | AdamW, weight decay 0.1, ε 1e-6; 0.001; 1,000 steps linear; cosine to 0.0001; 96; 80,000; 512-token windows of concatenated biographies | arXiv:2309.14316v3 App. C ([[physics-of-lm-3-1-knowledge-storage]]) | verified 2026-09-15 | no ablation reported; Remark C.1: attribute-token accuracy above 99% |
| GPT2-12-12, bioS pretraining data | 124M | pretrain-stable | data form; entities | multi5 + permute: 5 differently templated biographies per person, sentences shuffled; N = 100,000 people, six attributes | arXiv:2309.14316v3 §2, §4.2 | verified 2026-09-15 | Fig. 3: 96.6% held-out QA accuracy against 9.7% for one fixed biography |
| GPT2-12-20 with rotary embedding, bioR | 302M | pretrain-stable | steps (other settings as above) | 150,000 | arXiv:2309.14316v3 App. C | verified 2026-09-15 | no ablation reported |
| GPT2-12-12, mixed training | 124M | pretrain-stable | QA share of training sequences | QA_r = 0.8 (2:8 BIO-to-QA tokens) | arXiv:2309.14316v3 App. C, Remark 3.1 | verified 2026-09-15 | App. C Fig. 10: higher QA ratio gave higher OOD QA accuracy |
| GPT2 QA fine-tuning (124M-682M checkpoints) | 124M-682M | SFT | full fine-tune grid; LoRA | LR {0.001, 0.0003, 0.0001} × weight decay {0.01, 0.001}, no warmup, cosine to 10%, batch 48, 50,000 steps, best test accuracy shown; LoRA weight decay 0.01, LR 0.0003 | arXiv:2309.14316v3 App. D | verified 2026-09-15 | Fig. 2: no setting recovers extraction on bioS single |
| GPT2, bioS(100K), 1,000 exposures | Figure 1(a) models | pretrain-stable | weight decay; LR; batch; steps | 0.02; 0.001; 192; about 175K | arXiv:2404.05405v1 App. A.1 Parameter 1 ([[physics-of-lm-3-recipe]]) | verified 2026-09-14 | final result "not very sensitive" to LR at 1,000 exposures (App. A.1) |
| OLMo-7B resumed at the 500B-token checkpoint, fact injection | 7B | pretrain-stable (continued) | batch; injection schedule; steps; initial LR | 2,048 × 2,048 tokens (4M); every 100 steps, 10 times (duplicate or paraphrase) or once; 1,000 steps with injection + 1,500 without; 0.000237 | arXiv:2406.11813v3 App. D, Table 5 ([[factual-knowledge-acquisition-pretraining]]) | verified 2026-09-15 | §4.3 Tables 2, 6: forgetting fits |
| OLMo-7B, small-batch comparison | 7B | pretrain-stable (continued) | batch | 128 sequences (16× smaller) | arXiv:2406.11813v3 §4.3 | verified 2026-09-15 | Tables 10-11: faster decay, x-intercepts "decreased by dozens of times" |
| Repeated-data scan models | not reported as a list; Fig. 6 legend 1.57M-805M | pretrain-stable | tokens; context; repeated share; source data | 100B; 8,192; 3%, 10%, 20%, 50%, 90% of tokens from repeats; 400B-token source (55% filtered Common Crawl, 32% books) | arXiv:2205.10487v1 §3, Fig. 4 ([[repeated-data-scaling]]) | verified 2026-09-15 | Fig. 2-4: degradation band; no regularization studied (§5.5) |
| Memorization-dynamics models | 125M / 1.3B / 13B | pretrain-stable | peak LR; global batch (tokens); schedule; sequence | 6.0e-4, 0.5M / 2.0e-4, 1M / 1.0e-4, 2M; polynomial, 375M-token warmup; 512; Adam β2 0.98, weight decay 0 | arXiv:2205.10770v2 App. A.4 Table 1 ([[memorization-without-overfitting]]) | verified 2026-09-15 | §4.2 Fig. 5: LR sweep, larger models faster at fixed LR |
| Capacity models (GPT-2 architecture) | 100K-20M | pretrain-stable | steps; batch; precision; data | 10^6; 2,048; bfloat16 (float32 comparison); uniform tokens, V = 2,048, S = 64, 5 seeds | arXiv:2505.24832v3 §3.2 ([[how-much-do-lms-memorize]]) | verified 2026-09-15 | Table 1: float32 raises α from 3.51 to 3.83 |
| GPT-3 fine-tuned on fictitious facts | 350M-175B | SFT | LR multiplier; batch; epochs; documents | 0.2; 16; 10 in the sweep, not reported for the scaling runs (App. B.3); 900 per subset (30 facts × 30 paraphrases) | arXiv:2309.12288v4 §2.1, App. B.2-B.3 ([[reversal-curse]]) | verified 2026-09-15 | App. B.2 Fig. 7: 20-setting sweep, reversed accuracy 0.0-0.3% |
| Extraction audit (GPT-Neo, T5, OPT) | 125M-66B | eval-gate | prefix; suffix; decoding; sample | ℓ − 50 tokens, ℓ ∈ {50, ..., 500}; 50 tokens exact match; greedy; 1,000 sequences per length and duplicate bucket | arXiv:2202.07646v3 §3.2 ([[quantifying-memorization]]) | verified 2026-09-15 | §4.4: beam search 100 adds under 2 points on average |
| Long-tail counterfactual LM | 4.8B | pretrain-stable | data; epochs; removal | C4; 1; relevant documents of 100 TriviaQA questions per log bin (about 30% of C4) | arXiv:2211.08411v2 §3.2 ([[long-tail-knowledge]]) | verified 2026-09-15 | Fig. 5 |
| ExactSubstr deduplication | not applicable | data filter (no §5.2 stage) | minimum duplicated span | 50 tokens | arXiv:2107.06499v2 §4.1 ([[deduplicating-training-data]]) | verified 2026-09-14 | Table 4: 1.926% → 0.138% copied tokens at 1.5B |

**Starting point for a small general-purpose run.** For a controlled check of data form at small scale, the verified rows support: a synthetic entity set with five rewritten, shuffled descriptions per entity, a 124M GPT2-style model trained for 80,000 steps at batch 96 with 512-token windows, AdamW (weight decay 0.1) and LR 0.001 with 1,000 warmup steps and cosine decay to 0.0001, then QA fine-tuning on half of the entities and testing on the other half, as [[physics-of-lm-3-1-knowledge-storage]] ran on 100,000 synthetic people. For knowledge-retention checks during continued pretraining of a 1B-7B model, inject probe facts every 100 steps in duplicate and paraphrase variants and track retainability, as [[factual-knowledge-acquisition-pretraining]] did at OLMo-7B with 4M-token batches. For memorization audits, use the extraction protocol with prefixes up to 450 tokens and 50-token exact-match suffixes on duplicate-stratified samples ([[quantifying-memorization]], GPT-Neo 125M-6B on the Pile).

## Generalization lens

**(a) What increases breadth.** Varied restatements of the same fact: multi5 + permute gave 96.6% held-out extraction against 9.7% for one fixed biography ([[physics-of-lm-3-1-knowledge-storage]] Fig. 3). Augmented data about other entities of the same type: minority accuracy 4.4% → 86.8% ([[physics-of-lm-3-1-knowledge-storage]] Result 6). QA-shaped data inside pretraining (Result 1). Reversed restatements for inverse queries ([[physics-of-lm-3-2-knowledge-manipulation]] Fig. 6). Model scale: more abstract and cross-lingual influence patterns at 52B than at 810M ([[influence-functions-generalization]] §5.3.1), higher long-tail accuracy for BLOOM-176B than 560M ([[long-tail-knowledge]] §3.1), larger effectivity at 7B than 1B ([[factual-knowledge-acquisition-pretraining]] Fig. 3). Paraphrased rather than duplicated exposures, and larger batches, which slowed forgetting ([[factual-knowledge-acquisition-pretraining]] Tables 2, 6, 11).

**(b) What causes narrowing or forgetting.** A small subset repeated inside the degradation band: 3× effective-size loss on copying against 1.15× on test loss ([[repeated-data-scaling]] Fig. 5). Duplicated text: faster forgetting than paraphrased text in 7 of 9 cells ([[factual-knowledge-acquisition-pretraining]] Table 2) and higher extractable memorization ([[quantifying-memorization]] §4.2). One-form, one-direction facts: near-zero extraction and near-zero reverse retrieval ([[physics-of-lm-3-1-knowledge-storage]] Result 2; [[reversal-curse]] Table 1). Training on CoT without generating CoT at test time: 24.1% against 86.4% ([[physics-of-lm-3-2-knowledge-manipulation]] Fig. 4). Long encounter intervals and small batches, which shorten retention ([[factual-knowledge-acquisition-pretraining]] Tables 6, 11). Pretraining on heavily repeated data followed by fine-tuning: a 1.6× effective-size reduction against training from scratch ([[repeated-data-scaling]] §5.4 reports this for 90% repeated tokens; §2 describes the same comparison with 50%).

**(c) How to measure it for this stage.** Held-out-entity QA splits, where no question about the test entities is trained ([[physics-of-lm-3-1-knowledge-storage]] §2.1). Paraphrased and reversed probes ([[factual-knowledge-acquisition-pretraining]] Table 1; [[reversal-curse]] §2.1). Accuracy binned by relevant-document count ([[long-tail-knowledge]] §3.1). Extraction rates on training data compared with a model not trained on that data or with test-set extraction ([[quantifying-memorization]] §4.1; [[how-much-do-lms-memorize]] §4). Copying evaluations and training loss on repeated subsets ([[repeated-data-scaling]] Fig. 2, 5). Per-domain held-out perplexity rather than one average ([[paloma]]). Influence scans for attributing a behavior to documents ([[influence-functions-generalization]]). Known measurement errors: extraction depends on prefix length (33% vs 65%) and decoding and match definition ([[quantifying-memorization]] §4.3-§4.4); entity-linked counts have about 60% precision ([[long-tail-knowledge]] §2.3); real-model reversal and manipulation tests cannot rule out training-data coverage ([[reversal-curse]] §2.2; [[physics-of-lm-3-2-knowledge-manipulation]] §4); membership inference is predicted to be at chance above 10^2 tokens per parameter, so it cannot confirm absence of a sample ([[how-much-do-lms-memorize]] §5.2.2); retainability is measured by log-probability, not generation ([[factual-knowledge-acquisition-pretraining]] App. A).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Treating low extraction with short prefixes as low memorization | Extraction rate rises when the prefix is lengthened | Audit with prefixes up to 450 tokens on duplicate-stratified samples ([[quantifying-memorization]] §4.3) |
| Reading verbatim recall as usable knowledge | Model completes training sentences but fails paraphrased or held-out-entity questions | Hold out entities from QA training and probe with paraphrases ([[physics-of-lm-3-1-knowledge-storage]] §2.1) |
| Evaluating facts only in the stated direction | Forward questions correct, entity-from-attribute questions near zero | Add reversed probes for every fact family ([[reversal-curse]]; [[physics-of-lm-3-2-knowledge-manipulation]] Result 7) |
| Upweighting a small high-quality subset by many repeats | Training loss on the subset approaches zero; copying or in-context evaluations fall more than test loss | Compare the repeat count with E = k·N^α and track a repeated-paragraph copying loss ([[repeated-data-scaling]] Fig. 2, 5) |
| Expecting SFT to make pretraining knowledge extractable | In-distribution QA near 100%, held-out-entity QA near zero after fine-tuning | Probe extraction before SFT; change pretraining data form instead ([[physics-of-lm-3-1-knowledge-storage]] Result 2) |
| Reporting one average factual accuracy | Gains concentrated in high-frequency bins | Bin by relevant-document count ([[long-tail-knowledge]] Fig. 1) |
| Comparing memorization across model families without a control | Differences caused by data predictability rather than storage | Include a model not trained on the data (GPT-2 on Pile prompts: 6% against 40%) ([[quantifying-memorization]] §4.1) |
| Training CoT data and evaluating direct answers | No gain on direct-answer manipulation tasks | Evaluate with and without generated hints ([[physics-of-lm-3-2-knowledge-manipulation]] Fig. 4) |
| Using a small batch for knowledge-injection continued pretraining | Probe log-probability gains vanish within about 10^9.5 tokens | Compare retainability at two batch sizes ([[factual-knowledge-acquisition-pretraining]] Tables 6, 11) |

## Check your understanding

1. A model reaches over 99% next-token accuracy on attribute tokens in its biographies but answers held-out-entity questions at 9.7%. Using the probing results, explain where the attribute is stored in this model and why QA fine-tuning cannot move it onto the name.
2. Adding augmented biographies of other people raised accuracy on unaugmented people from 4.4% to 86.8%. Explain a mechanism by which data about different entities changes how the model stores a rare entity's facts, and why WikiBook text did not have this effect.
3. Duplicated injections produce larger immediate log-probability gains than paraphrased injections, yet the chapter recommends paraphrases. Use the decay constants and x-intercepts to explain the trade-off, and state what measurement would be needed to confirm the recommendation on generated answers.
4. In the repeated-data experiments, why does damage peak at an intermediate repeat count rather than at the highest repeat count? Use the 0.9 × 2.2 + 0.1 × 0 arithmetic and the capacity argument.
5. Why do repeated subsets damage copying more than test loss, and what does this imply for a general model's in-context learning?
6. A lab adds chain-of-thought examples for attribute comparisons to pretraining data and sees no change on a direct-answer benchmark. Explain this result from Result 4 and the hint-accuracy calculation, and propose an evaluation that measures the change the data did produce.
7. Membership inference fails at chance on a model trained with 200 tokens per parameter. Explain why this does not show that a specific benchmark item was absent from training, and which method from this chapter could give evidence either way.

## Connections

- **Previous (dependency):** ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic. It sets duplicate counts, whose memorization effects §2 and §9 quantify.
- **Next:** ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale.
- ch-08b — Why Next-Token Pretraining Produces General Ability: In-Context Learning, Emergence, and Predictability (induction heads and in-context learning, damaged by repetition in §9).
- ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination (uniform repetition and contamination).
- ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures (production recipes, as opposed to the controlled runs in this Recipe).
- ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing (rewriting methods for §9).
- ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data (CoT data, related to §6).
- ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control (forgetting after pretraining).
- ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining (making new knowledge extractable from small corpora).
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation.
- ch-48 — Contamination Detection and Its Effect on Reported Scores (limits of membership inference and extraction as detectors).

## Sources

- [[quantifying-memorization]] — extraction definition, log-linear scaling with size, duplicates, and context, GPT-2 baseline control, deduplicated-model and OPT replications, audit protocol.
- [[how-much-do-lms-memorize]] — unintended memorization vs generalization, 3.51-3.83 bits per parameter capacity, double descent at capacity, membership inference scaling.
- [[memorization-without-overfitting]] — exact memorization, larger models memorize faster and before overfitting, forgetting baseline, training settings.
- [[physics-of-lm-3]] — 2 and 1 bits of knowledge per parameter at 1,000 and 100 exposures, diversity of phrasing and capacity.
- [[physics-of-lm-3-recipe]] — bioS(N) optimizer settings used in the Recipe.
- [[physics-of-lm-3-1-knowledge-storage]] — storage vs extraction experiments, augmentation table, probing, celebrity-minority result, mixed training, training settings.
- [[physics-of-lm-3-2-knowledge-manipulation]] — classification and comparison sample requirements, CoT results and hint arithmetic, inverse search, mitigations.
- [[reversal-curse]] — fine-tuning evidence in both directions, hyperparameter sweep, real-celebrity test, instruction experiment.
- [[factual-knowledge-acquisition-pretraining]] — micro-acquisition, effectivity, retainability fits, decay constants and x-intercepts, batch-size effect, learnability-threshold hypothesis.
- [[long-tail-knowledge]] — accuracy vs relevant-document count, counterfactual retraining, scaling extrapolation, corpus correlations, retrieval.
- [[influence-functions-generalization]] — influence formula and EK-FAC pipeline, sparsity, abstraction and cross-lingual influence with scale, word-order sensitivity, memorization checks.
- [[repeated-data-scaling]] — non-uniform repetition damage, degradation band fit, copying and induction-head effects, loss arithmetic, fine-tuning after repetition.
- [[data-constrained-scaling]] — uniform repetition up to about 4 epochs.
- [[deduplicating-training-data]] — copied-token rates before and after deduplication, ExactSubstr span length.
- [[paloma]] — per-domain perplexity as a measurement of coverage.
