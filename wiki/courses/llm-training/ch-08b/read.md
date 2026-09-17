<!-- chapter: ch-08b
     track: pretraining
     kind: content
     title: Why Next-Token Pretraining Produces General Ability: In-Context Learning, Emergence, and Predictability
     deps: [ch-08a]
     sources: [[gpt-2-unsupervised-multitask]], [[gpt-3-few-shot]], [[icl-data-distributional-properties]], [[induction-heads]], [[emergent-abilities]], [[emergent-abilities-mirage]], [[emergence-loss-perspective]], [[emergence-loss-perspective-recipe]], [[same-loss-better-downstream]], [[embers-of-autoregression]], [[predicting-downstream-elusive]], [[scaling-laws-unreliable-downstream]], [[phi-textbooks]], [[physics-of-lm-3]]
     figures: figures/metric-artifact-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 08b — Why Next-Token Pretraining Produces General Ability: In-Context Learning, Emergence, and Predictability

> **Core insight.** Next-token prediction on broad text trains many tasks at once, because text contains task demonstrations: GPT-2 (1.5B) reaches 55 F1 on CoQA with no CoQA training ([[gpt-2-unsupervised-multitask]] §3.5), and GPT-3 175B, given K in-context examples and no weight updates, reaches 100.0% on 2-digit addition and 71.2% on TriviaQA ([[gpt-3-few-shot]] Tables 3.9, 3.3). In controlled experiments, in-context learning appears only when training data is bursty and has many rare classes ([[icl-data-distributional-properties]] §3), and in the transformer language models studied by [[induction-heads]] it forms early in training, in a window of roughly 2.5-5B tokens, together with induction heads (Argument 1). Whether an ability looks predictable depends on measurement: abilities that jump under exact match or multiple-choice grade change smoothly under token edit distance or Brier score in the tested cases ([[emergent-abilities-mirage]] §3-4), yet with one corpus and tokenizer MMLU and GSM8K stay at chance until pretraining loss falls to about 2.2, and for MMLU and C-Eval the threshold remains under continuous metrics ([[emergence-loss-perspective]] §3). Loss does not determine transfer: models with equal pretraining loss transfer differently in simplified settings ([[same-loss-better-downstream]] §3), and GPT-3.5 and GPT-4 are less accurate on rare variants of deterministic tasks and on low-probability target outputs ([[embers-of-autoregression]] §5-6).
>
> **Guideline.** When a pretraining ablation is judged on a task where the compared models are still near chance, read a continuous per-item score (log-probability of the correct answer, token edit distance, or bits-per-byte) next to the thresholded metric, and use an evaluation set large enough that a model with the expected accuracy is unlikely to score zero, because in the tested model families the thresholded metric hid improvement that the continuous score showed ([[emergent-abilities]] App. A.1; [[emergent-abilities-mirage]] Fig. 3-4; §6 worked example). Otherwise, when every compared model is above chance, report the thresholded metric with its item-sampling error (ch-51). When two checkpoints are compared by pretraining loss, compare only checkpoints with the same corpus and tokenizer, because loss values are not comparable across them ([[emergence-loss-perspective]] §7), and add downstream probes, because equal loss did not give equal transfer in [[same-loss-better-downstream]] (§3). When a claim of general ability rests on common task formats, add rare or low-probability variants of the same task, because GPT-4 decodes rot-1 at 0.82 accuracy and rot-2 at 0.02 ([[embers-of-autoregression]] §5.1).

## Why this chapter matters for a general-purpose model

ch-08a showed that pretraining loss falls predictably with parameters and tokens, and that downstream accuracy is harder to predict. This chapter addresses the question that sits between those two facts: why a single objective, predicting the next token of broad text, produces abilities that no training target named, and when those abilities can be forecast from smaller runs.

The question is measurable in three ways. First, when a pretraining decision (data mixture, filter, token budget) is selected on small models, as in the model-ladder method of ch-08a, an ability that is at chance at small scale gives the selection no signal on a thresholded metric. Second, when a claim that a base model is general rests on few-shot benchmark scores, those scores mix task learning, task recognition, and contamination (§2). Third, when equal pretraining loss is used as a proxy for equal capability, the proxy fails in the settings documented in §8.

Pipeline position: pretraining (the outline block from ch-08a to ch-17, including this chapter; ch-15 and ch-16 belong to later stages) → mid-training (ch-32 to ch-32f) → SFT → preference optimization → RL → evaluation. Later stages inherit two properties set here: which task formats the base model can already recognize from context, and which low-probability variants it handles poorly. Terms used before their own chapter: *few-shot / K-shot* (K solved examples placed in the prompt), *cloze* (a fill-in-the-blank completion format), *BPB* (bits per byte: negative log-likelihood in bits divided by the string length in bytes, which reduces tokenizer effects; defined in ch-08a §6), *nats* (loss measured with the natural logarithm).

## §1 Language modeling as unsupervised multitask learning (GPT-2)

**Definition.** A language model factorizes the probability of a symbol sequence into next-symbol conditionals and is trained to maximize the log-probability of observed text ([[gpt-2-unsupervised-multitask]] §2, Eq. 1):

```
p(x) = Π_{i=1..n} p(s_i | s_1, ..., s_{i-1})
```

where x is a text, s_i is its i-th symbol (a byte-level BPE token in GPT-2), and n is its length.

**The problem.** A supervised system learns p(output | input) for one task. GPT-2's authors describe such systems as "narrow experts rather than competent generalists" and note that the two largest multitask NLP efforts of the time trained on 10 and 17 (dataset, objective) pairs, which they compare to 10 and 17 training examples in a meta-learning sense (§1).

**Mechanism.**
1. A general system must model p(output | input, task). Language lets the task, the input, and the output all be written as one sequence, for example (translate to french, english text, french text) (§2).
2. The supervised objective for such a sequence is the language-modeling objective evaluated only on the output tokens. The authors argue that the global minimum of the unsupervised objective is therefore also the global minimum of the supervised objective, and the open question becomes whether optimization reaches it in practice (§2).
3. Broad web text contains naturally occurring task demonstrations. Table 1 lists English-French translation pairs found in WebText, although non-English pages were deliberately removed; a language detector found 10 MB of French text in WebText (§2.1, §3.7).
4. At test time the task is specified by text: "TL;DR:" after an article for summarization, "english sentence = french sentence" pairs for translation, a document plus "A:" for reading comprehension (§3.5-3.7).

**Worked example (task hint).** On CNN/Daily Mail, GPT-2 with "TL;DR:" scores an average ROUGE (R-AVG) of 21.40; without the hint it scores 15.03; three random sentences from the article score 20.98 (Table 4). The difference 21.40 − 15.03 = 6.37 is the effect of the text hint "TL;DR:" (the text rounds it to 6.4, §3.6). The difference 21.40 − 20.98 = 0.42 is the margin over the random-sentence baseline. The model changes behavior when the task is named in text, but its summaries score within half a ROUGE point of an extractive baseline that picks three random sentences.

**Evidence.** Four models (117M, 345M, 762M, 1542M parameters; Table 2), trained on WebText (8 million documents, 40 GB; §2.1), evaluated with no task-specific training (the paper calls this zero-shot; the translation and Natural Questions prompts contain example pairs, §3.7-3.8):
- CoQA (conversational question answering): 55 F1 on the development set with greedy decoding, matching or exceeding 3 of 4 baselines trained on 127,000+ question-answer pairs (§3.5).
- Language modeling: state of the art on 7 of 8 datasets; on LAMBADA (predict the last word of a passage) perplexity goes from 99.8 (previous state of the art) to 8.6 and accuracy from 19% to 52.66%, or 63.24% with a stop-word filter (§3.1, §3.3, Table 3).
- Natural Questions: 4.1% exact match; 63.1% on the 1% of questions the model is most confident about (§3.8).
- Zero-shot performance improves log-linearly with capacity across tasks (Abstract, Fig. 1).
Status: **Result (single study)** for each number; the scaling trend is extended to 175B by [[gpt-3-few-shot]] (§2 below), a report from the same organization with overlapping authors, so it is not an independent replication.

**Conditions and limits.** WMT-14 English→French gives 5 BLEU, below a word-by-word lexicon substitution; French→English gives 11.5 BLEU (§3.7). The authors state that zero-shot performance "is still far from use-able" for practical applications and that on many practical tasks GPT-2 is likely no better than random (§6). Train and held-out WebText perplexity improve together with size, which the authors read as underfitting (§4, Fig. 4).

**Implication for a general-purpose model.** GPT-2's data design rejected a Common Crawl subsample selected for similarity to one target benchmark, stating the goal "to avoid making assumptions about the tasks to be performed ahead of time" (§2.1). The mechanism in steps 1-3 requires that demonstrations of many tasks exist in the corpus; §10 returns to what this implies for data selection.

## §2 Few-shot in-context learning at scale (GPT-3)

**Definition.** *In-context learning* (ICL) is task adaptation inside the forward pass: the model is conditioned on a natural-language instruction and/or K demonstrations and completes a new instance, with no gradient update ([[gpt-3-few-shot]] §1, Fig. 1.1). *Zero-shot* uses only an instruction (K = 0), *one-shot* uses K = 1, and *few-shot* uses as many demonstrations as fit in the 2,048-token context, typically K = 10 to 100 (§2).

**The problem.** Fine-tuning typically needs thousands to hundreds of thousands of task-specific examples (§1), and the authors argue that "the potential to exploit spurious correlations in training data fundamentally grows with the expressiveness of the model and the narrowness of the training distribution" (§1). ICL is proposed as a way to perform tasks without a narrow per-task training distribution.

**Mechanism and protocol.**
1. Pretraining is the outer loop (the paper's term): a 175B autoregressive model and 7 smaller models (125M-13B) are each trained on 300B tokens (Table 2.1).
2. The inner loop runs in context: K examples are drawn at random from the task's training set, separated by 1 or 2 newlines (§2.4).
3. Multiple choice is scored by comparing completion likelihoods, normalized per token; for ARC, OpenBookQA, and RACE the score is P(completion | context) / P(completion | answer_context), with answer_context the string "Answer: " or "A: " (§2.4).
4. K is chosen on a development set when one exists, because larger K is "usually but not always better" (§2.4).

**Worked example (arithmetic, GPT-3 175B, 2,000 random problems per task, exact match; Table 3.9).**

| Task | Zero-shot | One-shot | Few-shot |
|---|---|---|---|
| 2-digit addition | 76.9 | 99.6 | 100.0 |
| 3-digit addition | 34.2 | 65.5 | 80.4 |
| 4-digit addition | 4.0 | 14.0 | 25.5 |
| 2-digit multiplication | 19.8 | 27.4 | 29.2 |

Reading the table: moving from zero-shot to few-shot on 3-digit addition raises accuracy by 80.4 − 34.2 = 46.2 points without changing any weight. The prose in §3.9.1 gives 80.2% for 3-digit addition; the table gives 80.4. The authors searched the training data for the 3-digit test problems in two string forms and found 17 of 2,000 addition problems (0.8%) and 2 of 2,000 subtraction problems (0.1%) (§3.9.1). The 13B model solves 2-digit addition and subtraction about half the time and all other arithmetic operations less than 10% of the time (§3.9.1).

**Evidence.**
- TriviaQA (closed book): 64.3 / 68.0 / 71.2 zero / one / few-shot (§1, Table 3.3).
- LAMBADA accuracy: 76.2 / 72.5 / 86.4; the few-shot cloze format tells the model that exactly one word is expected (Table 3.2, §3.1.2).
- Word manipulation with K = 100: random insertion 67.2 few-shot vs 8.26 zero-shot; reversed words 0.44 few-shot (Table 3.10). The authors read the zero-shot failure on these artificial tasks as evidence that the model learns them at test time (§3.9.2).
- Across 42 accuracy-denominated benchmarks, the gap between zero-shot and few-shot grows with model size (Fig. 1.3, §1).
Status: **Result (single study)** per number.

**Conditions and limits.**
- Few-shot results stay near chance on some comparison tasks: WiC (whether a word has the same meaning in two sentences) is at 49.4% few-shot, which the paper calls random chance (§3.7); on ANLI (adversarial natural language inference) models smaller than 175B are at about 33%, the three-way chance level, even few-shot, and the authors describe 175B as showing "signs of life" on Round 3 (§3.8, §5).
- The authors state that it is ambiguous whether few-shot learning learns a task "from scratch" or recognizes a task seen in training, and that the position may differ per task (§5). This is an **Open question** in the paper.
- Contamination: a filtering bug left some benchmark overlaps in the training data. The authors evaluated "clean" subsets without 13-gram overlap; PIQA (29% flagged, 3-point drop) and Winograd (45% flagged, 2.6% drop) are marked with an asterisk, and four Wikipedia language-modeling benchmarks plus CBT are not reported (§4).

**Implication for a general-purpose model.** Few-shot evaluation is a direct probe of base-model generality, and its protocol choices (K, normalization, format) change the score. ch-00 §5 sets the base-model formulation (cloze or multiple choice) used in this course, and ch-47 covers how the harness records these choices.

## §3 Which data properties produce in-context learning

**Definitions** ([[icl-data-distributional-properties]] §1, §3).
- *Burstiness*: items appear in clusters within a context instead of uniformly over time.
- *Long tail*: a large number of rarely occurring classes.
- *Dynamic meaning*: one item maps to several labels (label multiplicity), or one class has high within-class variation.
- *In-weights learning* (IWL): using information stored in weights, as in standard supervised learning, in contrast to ICL.
- *Zipfian marginal*: class rank x has probability p(X = x) ∝ x^(−α), where α ≥ 0 sets the skew (α = 0 is uniform) (§3, Eq. 1).

**The problem.** Transformers trained on language show ICL without meta-training. The study asks which part of the training regime causes this: the architecture, the data distribution, or both (§1).

**Mechanism (experimental design).**
1. Data are Omniglot handwritten characters: 1,623 classes with 20 images each (§2.1).
2. Each training sequence has 8 image-label pairs followed by a query image; the model is trained with cross-entropy on the query label. Labels are fixed across training, as in supervised learning (§2.1-2.2).
3. In a *bursty* sequence the query class appears 3 times in the context, and a second class also appears 3 times so that the most frequent label is not a shortcut. Non-bursty sequences are sampled uniformly (§2.1).
4. ICL is evaluated as 4-shot 2-way classification on holdout classes, with labels randomly reassigned to 0 and 1 in each sequence; chance is 1/2. IWL is evaluated on trained classes that do not appear in the context; chance is usually 1/1,600 (§2.3).
5. The model is a 12-layer causal transformer with embedding size 64 and 8 heads, trained for 500k steps (App. A).

**Results** (the paper reports these as figure trends, not as tables of numbers).
- More bursty sequences in training give more ICL and less IWL (Fig. 2).
- At p(bursty) = 0.9, raising the number of classes from 100 to 1,600 to 12,800 raises ICL and lowers IWL; "we need both burstiness and a large number of classes for in-context learning to emerge" (§3.1, Fig. 3).
- Label multiplicity and within-class variation each raise ICL (Figs. 4-5).
- With uniform class frequencies the models show a tradeoff: no model keeps both ICL and IWL. With Zipfian classes (12,800 classes, p(bursty) = 0.9) the exponent α = 1 keeps both ICL and IWL on the 10 most common classes; rare classes are not memorized at any α (§3.2, Fig. 6).
- ICL can be transient: models "can in some cases lose an initial bias towards in-context learning, moving towards in-weights learning over the course of training" (§3.1; see also Fig. 2 caption).
- Vanilla RNNs and LSTMs matched for depth, hidden size, and parameter count never exceed chance on ICL, across a 15-run hyperparameter sweep per architecture (§3.3, Fig. 7, App. C.1).

**Worked example (Zipf shares, derived from Eq. 1).** For M classes (M is used here instead of K, which §2 uses for the number of shots), the top class has share 1 / H_M(α), where H_M(α) = Σ_{k=1..M} k^(−α) and k is the class rank.
- α = 0, M = 12,800: every class has share 1 / 12,800 = 0.0078%.
- α = 1, M = 12,800: H ≈ 10.03, so the top class has 10.0% and the top 10 classes have H_10(1) / H = 2.93 / 10.03 = 29.2%.
- α = 3: the top 3 classes have (1 + 1/8 + 1/27) / 1.202 = 1.162 / 1.202 = 96.7% of the data, which matches the paper's statement that the three most common classes form 97% of the data at α = 3 (§3.2, footnote 3).
At α = 1 about 71% of examples come from classes outside the top 10, and each of those classes has a share below 1% (class 11: 1 / (11 × 10.03) = 0.91%); this is the long tail that the authors associate with ICL.

**Conditions and limits.** The task is image-label sequence classification, not next-token prediction on text; the authors list next-token training and symbolic inputs as future work (§4, "Future directions"). ICL is evaluated on novel classes but previously seen labels (App. B). The α = 1 result holds "for this particular training regime" (Fig. 6 caption). Status: **Result (single study)**.

**Implication for a general-purpose model.** The result names distributional properties, not only size, as conditions for ICL. Whether corpus-level operations that change burstiness or class skew (rebalancing domains toward uniform shares, removing within-document repetition) change ICL in LLM pretraining is an **Open question**: the paper does not test text corpora. The authors note that supervised datasets are frequently rebalanced toward uniform distributions and suggest that non-language models trained on such data may miss the opportunity to acquire in-context learning (§4, "Broader implications"; **Interpretation** by the authors).

## §4 Induction heads as a mechanism for in-context copying

**Definition.** An *induction head* is an attention head that, on a repeated sequence of random tokens, shows two properties ([[induction-heads]] Key Concepts):
1. *Prefix matching*: it attends to earlier tokens that were followed by the current and/or recent tokens.
2. *Copying*: its output raises the logit of the attended-to token.
Together the two properties complete [A][B] … [A] → [B]. In small models the circuit has two heads in different layers: a previous-token head copies information from each token into the next position, and the induction head uses that information to find where the current token occurred before.

**The problem.** ICL is a behavior; the paper looks for the circuit that implements it and measures ICL without choosing specific tasks.

**Formula (ICL score).**

```
ICL_score = mean over sequences of [ loss(token 50) − loss(token 500) ]     (nats)
```

where loss(token i) is the cross-entropy of the model's prediction at position i of a context. The paper writes the definition once as "500th minus 50th" (Key Concepts) and once as "50th token loss minus the 500th token loss" (Argument 1); the reported values are positive, so the second ordering is the one used. The authors report that other index choices do not change the conclusions.

**Worked example.** A score of 0.4 nats means the per-token probability of the correct token at position 500 is on average e^0.4 = 1.49 times that at position 50 (a geometric-mean ratio). Before the phase change the score is below 0.15 nats (e^0.15 = 1.16); after it, about 0.4 nats (Argument 1).

**Evidence.** 34 decoder-only models with one training run each: small attention-only and small MLP models (1-6 layers, d_model 768, 12 heads, context 8,192, about 10B tokens), full-scale models from 4 layers / 13M to 40 layers / 13B non-embedding parameters, and "smeared key" models (Model Analysis Table). The *phase change* is the paper's name for the abrupt training window described next.
- **Co-occurrence (Argument 1):** in a window of roughly 2.5-5B tokens, ICL score rises, induction heads form, the training loss shows a bump (the only non-convex part of the loss curve), and the per-token-loss trajectory changes direction. Over 75% of final ICL forms in this window. One-layer models form no induction heads, and "not much in-context learning ever forms" in them. Model Details places the phase change of the small models at about 1-3B tokens. Status: correlational.
- **Co-perturbation (Argument 2):** a smeared-key architecture lets each head mix the current and previous key, k_j^h ← σ(α_h) k_j^h + (1 − σ(α_h)) k_{j−1}^h, where k_j^h is the key vector of head h at position j, α_h is a trainable scalar for head h, and σ is the logistic function. With it, ICL forms in one-layer models and forms earlier in two-layer models. Argument 2 says "two-layer and larger models", but Model Details presents smeared-key models only at one and two layers. Status: interventional.
- **Ablation (Argument 3):** removing induction heads at test time removes almost all ICL in small attention-only models. No ablations exist for the full-scale models. Status: causal for small models.
- **Generality (Argument 4):** in the 13B model, heads that meet the strict definition also perform translation (layer 7 of 40, copying score 0.20, prefix-matching score 0.85) and abstract pattern completion (layer 8, 0.69 / 0.94), besides literal copying (layer 21, 0.89 / 0.75).

**Conditions and limits.** For large models the authors rate the evidence as medium and correlational: 15 snapshots per model give low time resolution, and a shared cause (for example, learning to compose attention layers) could produce both induction heads and other ICL mechanisms (Argument 1, "Assessing the Evidence"). After the phase change the ICL score is about the same for a 2-layer model and the 13B model; large models gain most of their advantage over small models within the first ten tokens of context (Unexplained Curiosities, "Seemingly Constant In-Context Learning Score"). Status: **Result (single study)**, with the causal part limited to small models.

**Implication for a general-purpose model.** The mechanism operates on repetitions inside the context. This connects to §3: bursty data places repeated items inside one context window (**Interpretation**, this course; neither paper tests the link directly). The ICL score is defined by loss at different token indices, and the paper also computes it against the final index 8,192, so the same measurement applies to long contexts; ch-32c discusses loss and perplexity as proxies for long-context quality.

## §5 Emergent abilities: the claim

**Definition.** "An ability is emergent if it is not present in smaller models but is present in larger models" ([[emergent-abilities]] §2). On a plot of performance against training FLOPs, the pattern is near-random performance up to a scale, then a rise to well above random. The authors state that the scale at which an ability emerges is not an immutable property of the ability; it can be lower with higher-quality data or other training changes (§2).

**The problem.** Loss scales smoothly with compute (ch-08a), but if some abilities do not, then small-model experiments cannot forecast them.

**Evidence** (Table 1, scale at which above-random performance appears):

| Ability | Model family | Training FLOPs | Parameters |
|---|---|---|---|
| 3-digit addition/subtraction, few-shot | GPT-3 | 2.3E+22 | 13B |
| MMLU, 57-topic average, few-shot | GPT-3 | 3.1E+23 | 175B |
| Word in Context (WiC), few-shot | PaLM | 2.5E+24 | 540B |
| Chain-of-thought on math word problems | LaMDA | 1.3E+23 | 68B |
| Instruction following after instruction fine-tuning | FLAN | 1.3E+23 | 68B |
| Scratchpad for 8-digit addition (fine-tuned) | LaMDA | 8.9E+19 | 40M |

- The authors manually classified all 210 tasks of BIG-Bench (a collaborative benchmark of over 200 tasks); Appendix E lists 25 tasks as emergent with GPT-3 or LaMDA and 42 more as emergent only with PaLM (counts derived from the lists), plus tasks that scale smoothly and tasks where no model beats random (App. A.3, App. E).
- Scale is not the only factor: on 14 BIG-Bench tasks PaLM 62B is above random while LaMDA 137B and GPT-3 175B are near random; the authors name higher-quality data (more multilingual and code data) and architecture differences as possible reasons without an ablation (§5.2, App. F). Status: **Interpretation** by the authors.
- The authors' own cross-entropy analysis: for all six emergent BIG-Bench tasks examined with LaMDA, cross-entropy on the target improves at scales where exact match, BLEU, or accuracy is still at random (App. A.1, "Outcome 2"). They conclude that this "does not provide any straightforward indicators of how to predict such emergent behaviors".

**Conditions and limits.** Classification of a task as emergent depends on what counts as near-random; two co-authors agreed on every task labeled emergent (App. A.3). The model families in Table 1 differ in data and architecture, so the scales are not from one controlled series. Status: **Result** per table row, each taken from the cited original study; the classification of BIG-Bench tasks is this paper's own.

**Implication for a general-purpose model.** An ability that is at chance in every model of a small-scale ladder gives no signal to data or architecture decisions made on that ladder under the thresholded metric. §6 and §7 give two accounts of what can still be measured below that scale.

## §6 Emergence as a metric artifact

**Definition.** [[emergent-abilities-mirage]] (Schaeffer, Miranda, Koyejo; arXiv 2023-04) proposes that, for a fixed task and model family, apparent emergence can come from the researcher's choice of a nonlinear or discontinuous metric, while the per-token error rate changes smoothly (Abstract).

**Mechanism (§2).**
1. Assume per-token cross-entropy falls as a power law in parameters N: L_CE(N) = (N/c)^α, with constants c > 0 and α < 0. The authors use this form for illustration only.
2. The probability that one token is correct is p(N) = exp(−L_CE(N)).
3. Exact match on an L-token target, assuming independent tokens, scores Accuracy(N) ≈ p(N)^L. This is nonlinear in p.
4. Token edit distance scores TED(N) ≈ L · (1 − p(N)). This is approximately linear in p.
5. Two further factors create apparent jumps: evaluation sets too small to resolve low accuracy at small scale (resolution = 1 / test-set size), and too few model sizes sampled at large scale (§2).

**Worked example (constants chosen by this course for arithmetic convenience: c = 10^8, α = −0.5, L = 5).**

| Parameters N | L_CE = (N/c)^α | p = e^(−L_CE) | Accuracy = p^5 | TED = 5(1 − p) |
|---|---|---|---|---|
| 10^8 | 1.000 | 0.368 | 0.007 | 3.16 |
| 10^9 | 0.316 | 0.729 | 0.206 | 1.36 |
| 10^10 | 0.100 | 0.905 | 0.607 | 0.48 |
| 10^11 | 0.032 | 0.969 | 0.854 | 0.16 |

Per-token loss falls by the same factor (√10 = 3.16) at every step. TED falls 3.16 → 1.36 → 0.48 → 0.16; its decreases per step are 1.80, 0.88, 0.32, each smaller than the one before. Accuracy goes 0.007 → 0.206 → 0.607 → 0.854; it is below 0.01 at 10^8, and its increases per step are 0.199, 0.401, 0.247. With only these four model sizes, the accuracy series has the pattern of the emergence definition in §5: near zero at the smallest size, then well above zero.

Resolution: with a 100-item test set at N = 10^8, the expected number of correct items is 100 × 0.0067 = 0.67, and the probability of observing zero correct is (1 − 0.0067)^100 = 0.51. In about half of such evaluations the small model scores exactly 0. With 1,000 items the probability of zero falls to 0.0012.

[Figure: metric-artifact-explorer.html](figures/metric-artifact-explorer.html) lets the reader change the target length L and the test-set size and see the same per-token probabilities scored by exact match and by token edit distance, with the probability of a zero score at each model size.

**Metrics named in the evidence.** *Multiple Choice Grade* scores 1 if the highest probability is on the correct option and 0 otherwise. *Exact String Match* scores 1 if the output string equals the target and 0 otherwise. *Brier score* is the squared error between the predicted probabilities over the options and the one-hot correct answer (worked example in §7). The *emergence score* is a BIG-Bench statistic that is large when a metric's total change across model scales is large relative to the typical change between neighboring scales (§4, Eq. 1).

**Evidence.**
- InstructGPT/GPT-3 API models (350M, 1.3B, 6.7B, 175B), 2-shot 2-digit multiplication and 4-digit addition: under Accuracy the family shows emergence for 4- and 5-digit targets; under token edit distance, with the same outputs, performance improves smoothly (§3, Fig. 3). With additional generated test data, every model in the family has above-chance accuracy (§3, Fig. 4).
- BIG-Bench meta-analysis: at most 5 of 39 preferred metrics show emergence by the emergence score; in the hand-annotated task-metric-model triplets of Jason Wei's 2022 list "137 emergent abilities of large language models" (reference [32] of the paper, a different document from [[emergent-abilities]]), emergence appears under 4 metrics, and Multiple Choice Grade plus Exact String Match account for more than 92% of claimed emergent abilities (§4, Fig. 5).
- LaMDA tasks that are emergent under Multiple Choice Grade are not emergent under Brier score (§4, Fig. 6).
- The authors induce apparent emergence in vision models by redefining the metric, for example counting a sequence of Omniglot characters as correct only if all L images are classified correctly (§5).
Status: **Result (single study)**; the observation that a likelihood-based score improves where the thresholded metric does not is **Replicated** by [[emergent-abilities]] App. A.1 (cross-entropy on six BIG-Bench tasks), although those authors still classify the tasks as emergent.

**Conditions and limits.** The independence assumption in step 3 is false, and the authors state it gives only qualitative agreement (§2 footnote 1). They write that "nothing in this paper should be interpreted as claiming that large language models cannot display emergent abilities" (§7). [[emergent-abilities]] notes that emergence also appears on classification tasks where the partial-credit argument for long targets does not apply (§5.1).

**Implication for a general-purpose model.** For a small-scale pretraining ablation, a metric with partial credit (token edit distance, log-probability of the target) and a larger evaluation set are two separate ways to obtain signal on a task where exact match is near zero. The next section shows tasks where continuous metrics do not remove the threshold.

## §7 The loss-threshold view and why downstream accuracy stays hard to forecast

**Definition.** [[emergence-loss-perspective]] (Du et al.; arXiv 2024-03) redefines an emergent ability as one present in models with lower pretraining loss and absent in models with higher loss, with corpus, tokenizer, and architecture fixed (§4).

**Formula (§4, Eq. 3-5).**

```
normalized_performance(L) = f(L) if L < η, else 0
L(N) = L_∞ + (N_0 / N)^{α_N}
performance > 0  only if  N ≥ N_0 · (η − L_∞)^{−1/α_N}
```

L is pretraining loss; η is the threshold; f is monotonically decreasing, with random guessing mapped to 0; N is model size at a fixed token count; L_∞ is irreducible loss; N_0 and α_N are fitted constants. The paper does not report fitted values for these constants.

**Evidence.** More than 30 models (300M-32B) on one English:Chinese 4:1 corpus with a 65k SentencePiece vocabulary ([[emergence-loss-perspective-recipe]]):
- For 8 of 12 tasks (TriviaQA, HellaSwag, RACE, WinoGrande and four Chinese tasks), performance improves with lower loss from the start, with Spearman rank correlation between performance and loss from −0.947 to −0.996, and checkpoints of different sizes fall on one performance-versus-loss curve (Table 2, Fig. 1).
- MMLU, C-Eval, GSM8K, and GSM8K-Chinese stay at random level until loss falls to about 2.2 (§3.1).
- For MMLU and C-Eval, the threshold remains under two continuous metrics, CorrectChoiceProb (the predicted probability of the correct answer) and Brier score; this is the paper's test of the metric-artifact account (§3.2, Fig. 4). GSM8K and GSM8K-Chinese were not re-scored with continuous metrics.

**Worked example (Brier score, §3.2 and App. C).** For one 4-option question, Brier = Σ_j (y_j − ŷ_j)², with y the one-hot correct answer and ŷ the predicted probabilities. A uniform prediction (0.25 each) scores (1 − 0.25)² + 3 × 0.25² = 0.5625 + 0.1875 = 0.75. Always predicting (1, 0, 0, 0) scores 0 when option 1 is correct and 1 + 1 = 2 otherwise, so 0.75 × 2 = 1.5 when the correct answers are spread evenly over the four options. Both predictors ignore the question. A model can therefore lower its Brier score from 1.5 toward 0.75 without learning the task. Du et al. treat 0.75, the best score of a predictor that ignores the question, as the random-guess level, and they report that Brier score stays no better than that level until loss reaches the threshold (§3.2, App. C).

**Why accuracy is harder to forecast than loss.** [[predicting-downstream-elusive]] (Schaeffer et al.; arXiv 2024-06) traces a multiple-choice score through four steps: negative log-likelihood of the correct choice → its probability over the vocabulary → its probability renormalized over the available choices → Accuracy or Brier score (§4). On ARC-Challenge, about 90% of samples have a per-sample score-compute Spearman correlation above 0.75 for the log-likelihood, 40% after renormalization over choices, and fewer for Accuracy (§4, Fig. 3). The mechanism is mass on incorrect choices: if the correct choice has probability 0.4 and the other 0.6 is spread as 0.2 / 0.2 / 0.2, Accuracy is 1; if 0.6 sits on one incorrect choice, Accuracy is 0 (§5). A forecast of the correct-choice probability alone cannot separate these cases. Brier score did not restore predictability in this study (§2, Fig. 3C).

**Three views compared (Interpretation, this course).**

| View | Horizontal axis | What it predicts | Evidence limit |
|---|---|---|---|
| Emergence with scale ([[emergent-abilities]]) | FLOPs or parameters | Abilities absent below a scale, present above | Emergence scale changes with data and training; not a forecast method |
| Metric artifact ([[emergent-abilities-mirage]]) | Parameters | Smooth change under continuous metrics | Tested on GPT-3 arithmetic, LaMDA BIG-Bench, vision toys |
| Loss threshold ([[emergence-loss-perspective]]) | Pretraining loss, same corpus and tokenizer | Tasks at chance until loss < η; for MMLU and C-Eval also under continuous metrics | One corpus and tokenizer; AdamW only; no routed or non-Transformer models (§7) |

The views are compatible in one reading (**Interpretation**, this course): loss improves smoothly, continuous per-item scores moved before thresholded scores in the tasks examined by [[emergent-abilities]] (App. A.1) and [[emergent-abilities-mirage]] (§3), and some tasks still require a loss level before any score leaves chance. [[scaling-laws-unreliable-downstream]] adds a base rate for one study: of 46 tasks in Gadre et al. (2025), 18 (39%) scale predictably with validation loss (Fig. 1).

**Conditions and limits.** Du et al. state that loss values are not comparable across tokenizers or corpora, that other architectures and optimizers were not tested, and that new tipping points at larger scale are not guaranteed (§7). The fitted threshold near 2.2 belongs to their English:Chinese 4:1 corpus and 65k tokenizer. Schaeffer et al. (2024) study five model families on multiple-choice benchmarks only and do not test whether per-choice forecasts work ([[predicting-downstream-elusive]] §6, App. B). Status: **Result (single study)** for each paper.

**Implication for a general-purpose model.** When small-model ablations are used to choose pretraining data and a task has a loss threshold that none of the ablation models reaches, that task cannot rank the options under accuracy, CorrectChoiceProb, or Brier score in the Du et al. setting. Tasks that improve from the start of training (8 of 12 in Du et al.) can rank options by accuracy or by a continuous score.

## §8 Same pretraining loss, different downstream results

**Definition.** The *saturation regime* is the state in which a model's predicted conditional distribution equals the true one, so its loss equals the entropy of the data ([[same-loss-better-downstream]] §3). *Implicit bias* is the preference of an optimizer, among parameter settings with the same loss, for some settings over others.

**The problem.** Pretraining loss is used to rank checkpoints and model sizes. If two models reach the minimal loss but transfer differently, loss cannot rank them.

**Mechanism.**
1. Data are generated from known models (a probabilistic context-free grammar, PCFG; a hidden Markov model, HMM; and samples from OPT-125M), so the minimal loss is computable: 3.196, 3.758, and 1.865 (§3). The pretraining objective is masked-language modeling, and transfer is measured by fine-tuning or by a *linear probe* (a linear classifier trained on frozen representations) (§3).
2. Three interventions keep loss at the minimum and change transfer: continuing training after convergence, increasing model size, and adversarial pretraining (§3).
3. Theory: at a global minimizer the stochastic-gradient covariance equals the Hessian ∇²L (the matrix of second derivatives of the loss). Near the set Γ of global minimizers, SGD with learning rate η (not the §7 threshold) and batch size B, in the limit of small η and over about 1/η² steps, drifts along Γ in the direction that lowers Tr[∇²L], with coefficient 1/(4B) (§4, Theorem 4.3).
4. Flatness is measured by the trace of the Hessian, estimated as the expected squared norm of ∇_θ log f_θ(x_−t)_{x_t}, where x is a sentence, t a masked position, x_−t the sentence with position t masked, f_θ the model's predicted probability vector for that position, and x_t is sampled from that prediction (§5, App. B.1).

**Worked example (the estimator in logit space; derived, this course).** For a softmax output with probabilities p over classes, the gradient of log p_y with respect to the logits z is e_y − p, where e_y is the one-hot vector for sampled class y. Take p = (0.5, 0.3, 0.2):
- y = 1 (probability 0.5): ‖(0.5, −0.3, −0.2)‖² = 0.25 + 0.09 + 0.04 = 0.38
- y = 2 (0.3): ‖(−0.5, 0.7, −0.2)‖² = 0.25 + 0.49 + 0.04 = 0.78
- y = 3 (0.2): ‖(−0.5, −0.3, 0.8)‖² = 0.25 + 0.09 + 0.64 = 0.98
- Expected value: 0.5 × 0.38 + 0.3 × 0.78 + 0.2 × 0.98 = 0.62 = 1 − Σ p_j².
The paper's estimator applies the same sampling to gradients with respect to all parameters, which multiplies these logit-space vectors by the network Jacobian.

**Evidence.**
- Algorithm (235M, PCFG; Table 1): AdamW reaches loss 3.204 with task A 89.9% and task B 49.2%; adversarial pretraining reaches 3.206 with 83.1% and 42.3%; a lookup table of the true conditionals has the optimal 3.196 and 71.2% / 39.7%.
- Size: at the same loss, scaling up improves linear-probe accuracy by 6.9% (PCFG), 4.5% (HMM), and 2.0% (OPT-generated data) (§3, Fig. 2); on PCFG, larger models have trace of the Hessian falling from 19.8 to 12.6 while task-B probe accuracy rises from 40.4% to 50.5% (§5, Fig. 5).
- Training after convergence (SGD with 12% warmup and a fixed learning rate of 1e-3): trace falls and accuracy rises by 1.6% (235M, PCFG task C) and 4.0% (67M, HMM task-10) while loss is unchanged (§5, Fig. 3, App. B.2).
Status: **Result (single study)**.

**Conditions and limits.** Models are masked-language models (BERT-style transformers and LSTMs), not autoregressive; pretraining and downstream distributions are the same for PCFG and HMM data; reaching saturation on real large-scale data is described as computationally challenging (§3, §7). Du et al. cite a related observation, that models with equal pretraining loss can differ after fine-tuning ([[emergence-loss-perspective]] §5).

**Implication for a general-purpose model.** A token-budget or optimizer change that leaves validation loss unchanged is not shown to be neutral for transfer. ch-08a §8 (post-trained quality decreasing with more pretraining tokens in one model family) is a separate reason for the same caution.

## §9 Embers of autoregression: where next-token training predicts poor generalization

**Definition.** [[embers-of-autoregression]] (McCoy et al.; arXiv 2023-09) predicts three sensitivities from the training problem, next-word prediction over Internet text (§2.2):
1. *Task probability*: lower accuracy on rare tasks than on frequent tasks of equal complexity.
2. *Output probability*: lower accuracy when the correct output is a low-probability string, even for deterministic tasks.
3. *Input probability*: the same for inputs, with a smaller effect than for outputs.

**Mechanism (§3.3).** The model's task can be written as choosing the output that maximizes P(output | input) ∝ P(input | output) · P(output). For a deterministic task only one output has nonzero P(input | output). A model that estimates P(input | output) imperfectly assigns nonzero likelihood to several candidates, and the prior P(output) then affects the choice. The authors state this as a computational-level description, not a claim that the model computes a likelihood and a prior (§3.3 footnote 3).

**Worked example (numbers chosen by this course).** A rot-13 input whose correct decoding is the low-probability sentence "Well, if they don't code, so be it." Suppose the model's likelihood estimate gives 0.6 to the correct decoding and 0.3 to the regularized sentence "Well, if they don't come, so be it.", and its prior gives them relative probabilities 0.01 and 0.05. The products are 0.6 × 0.01 = 0.006 and 0.3 × 0.05 = 0.015, so the model outputs the regularized sentence. The paper tests exactly this kind of one-word change and reports that both GPT-3.5 and GPT-4 produced the regularized version more often than the correct one (§6.1, Fig. 6.2).

**Evidence** (gpt-3.5-turbo-0613 and gpt-4-0613, temperature 0; 100 high-probability sentences plus derived medium- and low-probability sets; §4). A *shift cipher* rot-k replaces each letter with the letter k positions later in the alphabet (§3.1).
- Task probability: decoding shift ciphers with shifts 1-25, GPT-4 scores 0.82 on rot-1, 0.76 on rot-3, and 0.02 on rot-2; GPT-3.5 scores 0.21 on rot-13 and 0.00 on every other shift (§5.1). In C4, shifts 1, 3, and 13 are the most common (§5.1, Fig. 5.2).
- Output probability: GPT-4 decodes rot-13 with 0.51 accuracy when the output is a high-probability sentence and 0.13 when it is low-probability; logistic regression gives p < 10⁻⁴ for both models (§6.1). GPT-4 on word-order reversal: 97% vs 53% (Table 1).
- Input probability: GPT-4 encodes into rot-13 with 21% accuracy for high-probability inputs and 11% for low-probability inputs (Table 1). Across the shift-cipher, reversal, Pig Latin, and acronym encoding tests, logistic regression finds a significant input-probability effect only for GPT-4 on the shift cipher (p < 0.05) and GPT-3.5 on acronyms (p < 10⁻⁵) (§7.1).
- Across 7 pairs of common and rare task variants, models are better on the common variant (§5.6).
- Chain-of-thought and step-by-step prompts raise decoding accuracy and appear to lower encoding accuracy, but the shift-level pattern and the output-probability effect remain (§9.4, Fig. 9.1).
Status: **Result (single study)**; the task-probability effect agrees with contemporaneous counterfactual-task work that the authors cite (§5.6).

**Conditions and limits.** The tested models are proprietary chat models with undisclosed architecture and training data (Limitations); such models also go through instruction tuning, which the paper does not analyze (§2.1). No base (pretrained-only) model is tested. The tasks are simple and, in the authors' words, do not have substantial practical utility; transfer of the effects to practical tasks is left to future work (Limitations).

**Implication for a general-purpose model.** A deterministic skill measured only on common formats and high-probability answers can overstate generality. The low-probability variant of a task is a probe for whether the model applies a procedure or reproduces frequent strings; ch-47a builds counterfactual and perturbed evaluations on this idea.

## §10 Implication for data decisions: breadth versus benchmark-shaped data

Four findings bear on how the pretraining distribution affects transfer. Items 1 and 2 draw on §1, §3, and §9; items 3 and 4 add two further sources. Each is stated with its evidence and scope.

1. **Task coverage through demonstrations.** GPT-2's authors propose that tasks are learned from demonstrations present in text (§1). GPT-2 reaches 11.5 BLEU on French→English although a language detector found only 10 MB of French text in WebText (§1); tasks rare in text, such as rot-2, stay near zero for GPT-4 (§9). Status: **Interpretation** combining two single studies; neither measures task frequency and accuracy on one base model.
2. **Distributional shape.** Burstiness and a long tail of rare classes are required for ICL in the Omniglot setting (§3). Status: **Result (single study)**, not tested on text corpora.
3. **Phrasing diversity for extractable knowledge.** In synthetic-biography experiments, 1,000 differently templated biographies per person give slightly higher knowledge capacity than 1,000 passes over one fixed biography, and knowledge memorized from low-diversity data is nearly 0% extractable ([[physics-of-lm-3]] Result 2, App. A.3, citing Part 3.1). Status: **Result (single study)**, GPT-2-scale models.
4. **Similarity to the evaluation.** phi-1, trained on filtered and synthetic code aimed at one task format, solves 81.7% of the 71 HumanEval problems with close matches in its exercise set and 26.9% of the 93 without; StarCoder-Prompted solves 57.7% and 29.0% ([[phi-textbooks]] Table 3, τ = 0.95, where τ is the AST match-rate threshold for "close match"). The similar-minus-non-similar gap is 54.8 points for phi-1 and 28.7 for StarCoder-Prompted (derived). StarCoder-Prompted, which was not trained on the exercise set, also scores higher on the similar subset, so part of the gap reflects easier problems. phi-1 retrained after removing the close matches scores 74.6% and 32.3% (gap 42.3, derived), with the same 50.6% total; the authors read the pruning result as evidence that phi-1's score is not explained by contamination (§5.2). Status: **Result (single study)**; reading the larger gap as narrowing toward the benchmark format is an **Interpretation** of this course.

**Recommendation.** When a data filter or synthetic source is chosen by its effect on a benchmark, use a similarity-split evaluation (items with and without close matches in the new data), compare the gap with a model not trained on that data, and add a low-probability or counterfactual variant of the task, because for phi-1 the gap between matched and unmatched HumanEval items was 54.8 points, against 28.7 points for StarCoder-Prompted (item 4), and a single total score does not show that difference. Otherwise, when data is selected without a benchmark target (GPT-2's approach, §1), use the development and unseen suites of ch-00. ch-09 and ch-10a apply this to real pretraining mixtures and classifier-based selection.

## Recipe

The rows below are the context-length, data-diversity, and optimization settings of the runs this chapter cites. GPT-2 omits most training settings; those are listed as not reported.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT-2 (WebText LMs) | 117M, 345M, 762M, 1542M | pretrain-stable | layers; d_model | 12, 24, 36, 48; 768, 1024, 1280, 1600 | GPT-2 report (OpenAI PDF, 2019) Table 2 | verified 2026-09-15 | sizes "approximately log-uniformly spaced" (§3); no ablation reported |
| GPT-2 | all four | pretrain-stable | context length; batch size; vocabulary | 1024 tokens; 512 (unit not stated); 50,257 byte-level BPE tokens | §2.3, §2.2 | verified 2026-09-15 | context raised from GPT's 512 (§2.3); no ablation reported |
| GPT-2 | all four | pretrain-stable | learning rate | tuned per model for best perplexity on a 5% held-out WebText sample; values not printed | §3 | not reported (checked §1-§7, Tables 1-17, and Appendix A, which contains only samples) | per-model held-out perplexity (§3) |
| GPT-2 | all four | pretrain-stable | optimizer; schedule; tokens or steps seen; compute | not reported | checked §1-§7, Tables 1-17, Appendix A | not reported | — |
| GPT-2 | all four | pretrain-stable | data source and filtering | WebText: outbound Reddit links with ≥3 karma (45M links), links before Dec 2017, de-duplicated and heuristically cleaned to >8M documents / 40 GB; Wikipedia removed; non-English pages removed | §2.1, §3.7 | verified 2026-09-15 | Common Crawl rejected for quality; no ablation reported |
| GPT-3 | 125M-175B (8 sizes) | pretrain-stable | tokens seen; context length; packing | 300B tokens for every size; 2,048 tokens; documents packed and separated by an end-of-text token, no cross-document masking | arXiv:2005.14165v4 Table 2.1, §2.1, App. B | verified 2026-09-15 | no ablation reported |
| GPT-3 175B | 175B | pretrain-stable | batch (tokens); peak LR | 3.2M; 0.6 × 10⁻⁴ | Table 2.1 | verified 2026-09-15 | batch size guided by measured gradient noise scale (§2.3) |
| GPT-3 Small | 125M | pretrain-stable | batch (tokens); peak LR | 0.5M; 6.0 × 10⁻⁴ | Table 2.1 | verified 2026-09-15 | same as above |
| GPT-3 | all sizes | pretrain-stable / decay | optimizer; clip; weight decay; warmup; decay; batch ramp | Adam β1 = 0.9, β2 = 0.95, ε = 10⁻⁸; global-norm clip 1.0; weight decay 0.1; linear warmup over 375M tokens; cosine to 10% of peak over 260B tokens, then constant at 10%; batch ramped linearly from 32k tokens over the first 4-12B tokens depending on size | App. B | verified 2026-09-15 | no ablation reported |
| GPT-3 | all sizes | pretrain-stable | mixture: tokens; sampling weight; epochs at 300B | Common Crawl (filtered) 410B, 60%, 0.44; WebText2 19B, 22%, 2.9; Books1 12B, 8%, 1.9; Books2 55B, 8%, 0.43; Wikipedia 3B, 3%, 3.4 | Table 2.2 | verified 2026-09-15 | "datasets we view as higher-quality are sampled more frequently" (§2.2); no ablation reported |
| GPT-3 | all sizes | pretrain-stable | epochs recomputed as weight × 300B / tokens | Common Crawl 0.44; WebText2 3.47; Books1 2.00; Books2 0.44; Wikipedia 3.00 | derived from Table 2.2 | derived (formula: weight × 300B / tokens, inputs from Table 2.2); conflicts with the printed epochs for WebText2 (2.9) and Wikipedia (3.4), which the report does not explain; Table 2.2 defines the weight as the fraction of training examples, not of tokens, so the token-share formula is an assumption of this course | no ablation reported |
| GPT-3 | all sizes | eval-gate | shots; decoding | K from 0 up to the 2,048-token context, typically 10-100, chosen on a development set when available; beam width 4, length penalty 0.6 for free-form tasks | §2, §2.4 | verified 2026-09-15 | larger K "usually but not always better" (§2.4) |
| Chan et al. Omniglot transformer (image-label sequence task, not released) | 12 layers; 831,479 parameters (App. C.1 count) | pretrain-stable | architecture; steps; optimizer; schedule; seeds | embedding 64, 8 heads; 500k steps; Adam, linear warmup to 3 × 10⁻⁴ at 4,000 steps, then inverse square root decay; 5 runs (3 for Figs. 5-6) | arXiv:2205.05055v6 App. A, App. C.1 | verified 2026-09-15 | no ablation reported |
| Chan et al. Omniglot transformer | same | pretrain-stable | data distribution | p(bursty) = 0.9; 1,600 or 12,800 classes; Zipf exponent α = 1 | §3.1-3.2 | verified 2026-09-15 | Fig. 6: of the tested exponents 0, 0.5, 1, 1.5, 2, 3, the authors identify α = 1 as the point where ICL and IWL on the 10 most common classes both stay high |
| Olsson et al. small models (not released) | 1-6 layers | pretrain-stable | context; width; steps; warmup; weight decay | 8,192 tokens; d_model 768, 12 heads; 10,000 steps (about 10B tokens); warmup over the first 1.5 × 10⁹ tokens; weight decay reduced at step 4,750 (about 5B tokens) | arXiv:2209.11895v1 Model Details | verified 2026-09-15 | no ablation reported; Model Details states that the weight-decay change falls after the phase change and that warmup is the only scheduled change inside its range |

Recipe values for the Du et al. runs, including sequence length 2048 and AdamW (0.9, 0.95), are in [[emergence-loss-perspective-recipe]].

**Starting point for a small general-purpose run.** From verified rows only: train at a 2,048-token context with documents packed and separated by an end-of-text token; use Adam with β1 = 0.9, β2 = 0.95, ε = 10⁻⁸, gradient clipping at 1.0, weight decay 0.1, linear warmup over 375M tokens, and cosine decay to 10% of the peak (GPT-3, all 8 sizes from 125M to 175B, 300B tokens, V100 cluster). For peak learning rate and batch size, read the GPT-3 Table 2.1 row closest to the model size; the 125M row uses 6.0 × 10⁻⁴ with a 0.5M-token batch. When sources differ in judged quality, GPT-3 sampled WebText2, Books1, and Wikipedia 1.9 to 3.4 times (printed epochs in Table 2.2) and filtered Common Crawl and Books2 less than once (0.44 and 0.43) in 300B tokens. For base-model evaluation, report zero-shot and few-shot results with K chosen on a development split, as GPT-3 did. None of these values was selected by an ablation in the cited reports.

## Generalization lens

**(a) What increases breadth at this stage.**
- A broad, task-agnostic corpus: zero-shot transfer across reading comprehension, translation, summarization, and question answering from WebText, improving log-linearly with size ([[gpt-2-unsupervised-multitask]] Abstract, Fig. 1).
- Scale for in-context use of demonstrations: the zero-to-few-shot gap grows with model size across 42 benchmarks ([[gpt-3-few-shot]] Fig. 1.3).
- Bursty data with many rare classes and a Zipfian marginal near α = 1, in the Omniglot setting ([[icl-data-distributional-properties]] §3).
- Phrasing diversity for knowledge that must be extracted by questions ([[physics-of-lm-3]] App. A.3).
- Among equal-loss models, larger models and longer training toward flatter solutions, in simplified MLM settings ([[same-loss-better-downstream]] §3, §5).

**(b) What causes narrowing or false breadth claims.**
- Data shaped like the target benchmark: 81.7% vs 26.9% on similar vs non-similar HumanEval problems for phi-1, against 57.7% vs 29.0% for StarCoder-Prompted, which was not trained on phi-1's exercises ([[phi-textbooks]] Table 3; reading the gap difference as narrowing is an Interpretation, §10 item 4).
- Uniform class balance: no model kept both ICL and IWL under uniform class frequencies; ICL can fade with continued training ([[icl-data-distributional-properties]] §3).
- Rare task variants and low-probability outputs: rot-2 at 0.02 vs rot-1 at 0.82; 0.51 vs 0.13 by output probability ([[embers-of-autoregression]] §5.1, §6.1).
- Contamination that inflates few-shot results: PIQA and Winograd flagged in GPT-3 ([[gpt-3-few-shot]] §4).
- A different training algorithm at the same loss: adversarial pretraining, designed to reduce transfer, gives task B 42.3% at loss 3.206 against 49.2% at loss 3.204 for AdamW (235M, PCFG; [[same-loss-better-downstream]] Table 1).

**(c) How to measure it for this stage.**
- ICL score by token index (loss at an early position minus loss at a late position) over training, and in-context learning curves over K ([[induction-heads]] Key Concepts; [[gpt-3-few-shot]] Fig. 1.2).
- A continuous per-item score next to every thresholded metric, and an evaluation set large enough that (1 − expected accuracy)^n is small (§6).
- Performance plotted against pretraining loss for checkpoints sharing corpus and tokenizer, with a check whether the task is still at its random baseline ([[emergence-loss-perspective]] §3).
- Per-sample tracking of probability mass on incorrect choices, instead of extrapolating benchmark Accuracy; the paper proposes this and does not show a working forecast ([[predicting-downstream-elusive]] §6, App. B).
- A per-task check that downstream scaling is predictable in the exact evaluation setup before using it to choose data ([[scaling-laws-unreliable-downstream]] §7).
- Rare-variant, low-output-probability, and similarity-split evaluations (§9, §10).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Extrapolating a task that is at chance in small models | Flat curve at the random baseline, then a jump at the largest run | Plot log-probability of the correct answer or token edit distance for the same outputs; check loss against the task's threshold for this corpus |
| Declaring an ability absent from a small evaluation set | Zero correct items at small scale | Compute (1 − expected accuracy)^n; enlarge the set until a zero score is unlikely |
| Comparing pretraining loss across tokenizers or corpora | "Lower loss but worse downstream" between two families | Compare loss only between checkpoints that share corpus and tokenizer ([[emergence-loss-perspective]] §7) |
| Treating equal validation loss as equal capability | Two runs with the same loss differ on probes or after fine-tuning | Run downstream probes on both; for research comparisons, add a flatness estimate |
| Forecasting benchmark Accuracy from its own trend | Forecast error large on some tasks, small on others | Forecast per sample; track probability mass on incorrect choices |
| Switching to Brier score to recover predictability | Brier trend still weakly correlated with compute | Brier score depends on incorrect-choice mass ([[predicting-downstream-elusive]] §4); track per-choice probabilities, which the paper proposes without showing a working forecast (§6) |
| Reading few-shot gains as learning a new task | Gains on tasks that occur in web text | Add artificial tasks unlikely in text (GPT-3 word manipulation) and contamination checks |
| Evaluating only frequent task variants | High score on rot-13-like formats | Add rare variants (rot-2) and low-probability target outputs |
| Selecting data by benchmark gain | Gain concentrated on benchmark items similar to the new data | Similarity split of the benchmark against the added data |
| Assuming a larger K always helps | Few-shot score below one-shot or zero-shot on some tasks | Choose K on a development split; report the K used |

## Check your understanding

1. GPT-2 argues that the language-modeling objective has the same global minimum as a supervised objective written as a text sequence. Explain why reaching that minimum in practice still depends on which tasks appear in the corpus, using the French translation result.
2. In Table 3.9 of GPT-3, 3-digit addition rises from 34.2 zero-shot to 80.4 few-shot. Explain two different mechanisms that could produce this gain, and which evidence in the paper separates them.
3. Chan et al. find that uniform class frequencies give either ICL or IWL but not both, while α = 1 gives both. Explain, in terms of how often each class appears in context versus across training, why skew allows both.
4. Explain why the induction-head mechanism needs at least two attention layers in a standard transformer and why the smeared-key change removes that requirement. What does the result of that change imply about the cause of the phase change?
5. Using the worked example in §6, explain why token edit distance and exact match give different pictures of the same model outputs, and why a 100-item test set strengthens the apparent jump.
6. Du et al. find a loss threshold near 2.2 that persists under Brier score for MMLU and C-Eval. Explain why this does not contradict the metric-artifact account for the GPT-3 arithmetic tasks.
7. Two checkpoints have the same validation loss, one after longer training. Explain the SGD-noise argument for why the longer-trained one could transfer better, and why this evidence does not yet establish that for autoregressive LLM pretraining.
8. GPT-4 decodes rot-1 at 0.82 and rot-2 at 0.02. Explain this with the P(input | output) · P(output) decomposition and state what it implies for evaluating whether a base model has learned a procedure.

## Connections

- **Previous (array order):** ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability (dependency; pretraining loss as a function of parameters and tokens, and the downstream prediction problem).
- **Next (array order):** ch-09 — Pretraining Data Composition and Capability Coverage (depends on this chapter; applies §10 to real mixtures).
- **Related later chapters:** ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection (§10 similarity checks); ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining (§10 item 3); ch-29e — Instruction Tuning and Generalization to Unseen Tasks (instruction following as an emergent augmentation, §5); ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation (loss and perplexity as long-context proxies, §4); ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation (rare and counterfactual variants, §9); ch-48 — Contamination Detection and Its Effect on Reported Scores (§2 contamination); ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions (§6 resolution).
- **Measurement contract:** ch-00 — What General Capability Means and How It Is Measured.

## Sources

- [[gpt-2-unsupervised-multitask]] — Radford et al., OpenAI report, 2019: p(output | input, task) framing, WebText construction, zero-shot results (CoQA, LAMBADA, summarization hint, translation, Natural Questions), model table, unreported training settings (chapter excerpt; no library card).
- [[gpt-3-few-shot]] — Brown et al., arXiv 2020-05: definitions of zero/one/few-shot ICL, evaluation protocol, arithmetic and word-manipulation tables, limitations, contamination analysis, Table 2.1-2.2 and App. B recipe (chapter excerpt; no library card).
- [[icl-data-distributional-properties]] — Chan et al., arXiv 2022-05: burstiness, class count, dynamic meaning, Zipfian skew, transformer-versus-RNN results, Omniglot training setup (chapter excerpt; no library card).
- [[induction-heads]] — Olsson et al., Transformer Circuits Thread 2022-03 (arXiv 2022-09): induction-head definition, ICL score, phase change window, six arguments and their evidence strength, model details (chapter excerpt; no library card).
- [[emergent-abilities]] — Wei et al., arXiv 2022-06 (TMLR 2022-08): definition of emergence, Table 1 scales, BIG-Bench classification, cross-entropy analysis, PaLM 62B tasks (chapter excerpt; no library card).
- [[emergent-abilities-mirage]] — Schaeffer, Miranda, Koyejo, arXiv 2023-04: metric-artifact model, GPT-3 arithmetic re-scoring, BIG-Bench metric meta-analysis, induced emergence (chapter excerpt; no library card).
- [[emergence-loss-perspective]] — Du et al., arXiv 2024-03: same-loss-same-performance result, loss threshold near 2.2, Brier score baselines, Eq. 3-5.
- [[emergence-loss-perspective-recipe]] — corpus, tokenizer, and optimizer settings of the Du et al. runs.
- [[same-loss-better-downstream]] — Liu, Xie, Li, Ma, arXiv 2022-10: saturation regime, equal-loss interventions, trace-of-Hessian theory and measurements.
- [[embers-of-autoregression]] — McCoy et al., arXiv 2023-09: task, output, and input probability effects; shift-cipher and reversal numbers; prompting and scaling checks (chapter excerpt; no library card).
- [[predicting-downstream-elusive]] — Schaeffer et al., arXiv 2024-06: transformation sequence from log-likelihood to Accuracy, ARC-Challenge correlations, incorrect-choice mechanism.
- [[scaling-laws-unreliable-downstream]] — Lourie, Hu, Cho, arXiv 2025-07: 18 of 46 tasks with predictable loss-to-task scaling.
- [[phi-textbooks]] — Gunasekar et al., arXiv 2023-06: similarity-split HumanEval results for benchmark-shaped training data.
- [[physics-of-lm-3]] — Allen-Zhu and Li, Part 3.3, arXiv 2024-04: phrasing diversity and extractability of stored knowledge.
