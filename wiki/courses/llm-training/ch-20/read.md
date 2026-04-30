<!-- chapter: ch-20
     track: synthetic
     title: Distillation-as-Data and R1-Distill Lineage
     sources: [[orca]], [[orca-2]], [[distilling-step-by-step]], [[deepseek-r1]], [[deepseek-r1-distill-synth]], [[bespoke-stratos]], [[openr1]], [[sky-t1]], [[open-thoughts]], [[dolphin]]
     figures: figures/distill-lineage.html
-->

# Chapter 20 — Distillation-as-Data and R1-Distill Lineage

> **Core insight.** Distillation in 2023–2025 is not logit-matching. It is *sampling from the teacher, filtering the samples, and SFT-ing the filtered set*. The teacher's value is not its weights but its **output distribution** — the trace of tokens between the prompt and the answer. Everything that distinguishes Orca from vanilla Alpaca, and R1-Distill from Orca, is a decision about (a) how to **elicit** the trace (system-prompt scaffolding, format tags, multi-strategy prompting), (b) how to **filter** it (gold-answer match, unit-test pass, LLM-judge, symbolic equivalence), and (c) how much of the teacher's **quirks** you are willing to inherit.
>
> **Guideline.** For reasoning distillation, preserve the teacher's trace wrapper verbatim, rejection-sample on a domain-appropriate verifier (SymPy / unit tests / LLM judge), and keep ~10–20K correct traces for a strong student — not ~1M — unless you are also doing RL downstream. Choose the teacher with the best *output distribution for your student base*, not the teacher with the best benchmark score; QwQ-32B beats R1 on OpenThoughts ablations even though R1 wins every headline eval.

---

## Why this chapter exists

Chapter 19 covered rejection-sampling fine-tuning on *one's own* rollouts — the Llama-3 and WebInstruct-style self-distillation loops that convert verifier signal into supervised data. This chapter is the other branch of the same idea: **what happens when the rollouts come from a stronger teacher model**. Once you commit to that, three things stop being research questions and become engineering decisions:

1. What *kind* of trace does the teacher emit, and how do you steer it? — Orca's 16 hand-crafted system messages, R1's `<think>` wrapper, QwQ's reflection tokens.
2. What filter keeps a trace? — `\boxed{}` + SymPy equivalence, `assert` over public tests, GPT-4o-mini LLM-judge, or nothing at all ([[open-thoughts]]).
3. What does the student *actually learn* — the reasoning, or just the teacher's stylistic tics? — the "Wait, let me reconsider..." tic, the 4K-token preamble, the refusal pattern, the benchmark contamination.

The Orca→R1-Distill arc is the cleanest two-generation case study in open post-training. Orca (2023) argues richer *teacher supervision* beats larger *student parameters*. Orca-2 (2023) argues richer *strategy variety* beats richer *single-trace verbosity*. DeepSeek-R1 (2025) strips almost all of that human design away — you just run RL on the teacher, sample 800K reasoning trajectories, filter on correctness, and SFT. The three open reproductions (Bespoke-Stratos, Open-R1, Sky-T1) then collapse the 800K corpus to 17K–440K with tighter filters and produce students within 2–3 points of the official distill. The chapter exists to lay out that arc, extract the recipe primitives, and make the licensing and teacher-bias tradeoffs explicit.

This is also the chapter where "distillation as data" becomes a *deployment* question. You cannot paper over: did the teacher's terms of service allow you to generate this corpus? Can you redistribute the 800K traces? Does the student inherit refusal patterns you did not intend? — §7 answers each.

---

## 1. Orca — explanation traces via system-prompt scaffolding

[[orca]] is the 2023 paper that set the template. The question it answers: *what does it take for a 13B student to learn GPT-4's reasoning rather than just its answers?* The answer, operationally, is the **system message**. Mukherjee et al. prepended one of **16 hand-crafted system messages** to every FLAN-v2 / Big-Bench / Chain-of-Thought / GSM-style task before querying GPT-4 (and, for cheaper coverage, ChatGPT). Each system message is a **teaching instruction** — the teacher is told not only to answer but to expose its reasoning in a specific style. The working list (reconstructed from the paper's Appendix A):

| # | System message (paraphrased) | What trace style it induces |
|---|---|---|
| 1 | "" (empty) | Direct short answer; serves as a control |
| 2 | "You are an AI assistant. User will give you a task. Your goal is to complete the task faithfully. While performing the task think step-by-step and justify your steps." | Generic CoT: numbered reasoning then final answer |
| 3 | "You are an AI assistant that helps people find information. Provide a detailed answer so the user doesn't need to search outside to understand the answer." | Long-form explanatory: definitions + worked example |
| 4 | "You are an AI assistant that follows instruction extremely well. Help as much as you can." | Instruction-literal: high fidelity to phrasing, no elaboration unless asked |
| 5 | "You are a helpful assistant, who always provides explanation. Think like you are answering to a five-year-old." | ELI5: analogy-heavy, short sentences |
| 6 | "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps. You should describe your reasoning as if you were explaining to a five-year-old." | ELI5 + CoT hybrid |
| 7 | "You are a teacher. Given a task, explain in simple steps what the task is asking, any guidelines it provides, and how to use those guidelines to find the answer." | Guideline-extraction style: restate task rules, then apply |
| 8 | "User will give you a task with some instructions. Your job is to follow the instructions as faithfully as possible. While answering think step-by-step and justify your answer." | Literal-then-CoT: combines #4 and #2 |
| 9 | "Explain how you used the definition to come up with the answer." | Post-hoc rationale: answer first, then justify |
| 10 | "You are a helpful assistant that answers questions by first writing a precise step-by-step plan, then executing the plan." | Plan-and-execute: two-phase decomposition |
| 11 | "You are an AI assistant that helps the user with complex tasks. Your responses should always include an answer and a brief explanation of the reasoning behind it." | Short CoT with mandatory rationale |
| 12 | "You are an AI assistant, who knows every language and how to translate one language to another. Given a task, you explain in simple steps what the task is asking, any guidelines it provides, and how to use those guidelines to find the answer." | Translation-calibrated reasoning |
| 13 | "Given a definition of a task and a sample input, break the definition into small parts. Each of those parts will have some instruction. Explain their meaning by showing an example that meets the criteria in the instruction. Use the following format: Part #: a key part of the definition. Usage: Sample response that meets the criteria from the key part. Explain why you think it meets the criteria." | Part-wise definition unpacking; teaches decomposition |
| 14 | "Given a task and the definition, explain how the definition leads to the answer by providing at least one example for each step." | Example-grounded reasoning |
| 15 | "You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five-year-old." | MCQ-specific: answer + eliminate-distractors |
| 16 | "You are an AI assistant that helps people find information. Provide a detailed answer so the user doesn't need to search outside to understand the answer. Think like you are a world-class expert in this field." | Expert-grounded explanatory |

Why 16 and not 1? Because a single CoT prompt produces a *single style*, and the student then pattern-matches on that surface form. With 16 styles the student has to learn **the reasoning that is invariant across styles** — the generalization target. This is the paper's operational definition of "explanation trace" as a training signal.

**The two-stage progressive learning mix.** Orca doesn't just hammer GPT-4. It does 5M ChatGPT traces first (cheap teacher, broad coverage), then 1M GPT-4 traces (expensive teacher, depth) on the same prompt pool — an explicit curriculum. The student sees easier explanations before harder ones; in SFT terms this is data-order curriculum, not loss re-weighting. Training Llama-13B on this mix brought BigBench-Hard zero-shot from Vicuna-13B's 44.0 to Orca-13B's 49.7 — a 13B matching 10×-larger baselines on reasoning evals, two years before "reasoning distillation" was a phrase.

**What Orca got wrong that Orca-2 fixes.** A student trained on trace-rich supervision tends to **always emit a long trace**, even when the task doesn't warrant it ("2+2 = ?" → 400 tokens of scratch work). [[orca-2]] attacks this with **strategy variety**: the teacher now emits one of five target-behavior modes — (i) direct answer, (ii) step-by-step, (iii) explain-then-answer, (iv) recall-then-answer, (v) extract-then-answer — chosen by a *task-specific* system message. The key trick is **Prompt Erasing**: the student is trained without the strategy-selecting system message, so at inference it must *itself* choose which strategy to apply. This is the bridge from "learn to execute a strategy the teacher picked" to "learn when each strategy applies" — the same move that later shows up in R1's `<think>` auto-regulation (the model learns when to think longer without being told).

---

## 2. Distilling Step-by-Step — rationale as joint-training signal

[[distilling-step-by-step]] (Hsieh et al., ACL 2023) is the rationale-distillation paper that should be read before R1. Its claim is sharper than Orca's: **a 770M T5 student can beat a 540B PaLM teacher's few-shot performance on 4 benchmarks using only 80% of the labeled data** — if you extract the teacher's CoT and train the student on a **joint (label, rationale)** multi-task objective.

Two-stage recipe, verbatim from the paper:

```
1. Extract (x, rationale_teacher, label_teacher) triples by few-shot CoT prompting PaLM-540B
   on each training example (3–8 in-context CoT exemplars).
2. Train T5-770M with multi-task loss
        L = L_label(y | x) + λ · L_rationale(r | x)
   where both heads share the T5 encoder. Decode only y at inference.
```

Why it works — and why it matters for R1 later:

- **Labels alone are a compressed supervision signal** — the student memorizes (x, y) mappings.
- **Rationales force the shared encoder to encode reasoning-relevant features** — the student's encoder learns what steps are *taken* on the way to `y`, not just what `y` is for each `x`.
- **The rationale head is discarded at inference** but its gradients have already shaped the encoder. The multi-task regularization is cheap at deployment.

Benchmarks beaten: ANLI (NLI), e-SNLI (explained NLI), CQA (commonsense), SVAMP (math word problems). The 770M student uses 80% of the available labeled data. This is the **first paper that explicitly argues rationale supervision carries a better learning signal per token than label supervision** — the thesis every later work (Orca, Orca-2, R1-Distill, Sky-T1) inherits. The failure mode Hsieh flags is the one that still bites two years later: **teacher rationale quality bounds the student's ceiling**; a hallucinated rationale teaches wrong reasoning, and the student has no way to detect it because it has no verifier.

The 2025 descendants relax the multi-task objective — R1-Distill just does single-head SFT on `<think>…</think><answer>…</answer>` — but the underlying claim is the same: *rationales are the supervision, labels are the byproduct*.

---

## 3. DeepSeek-R1 distill — what the teacher actually generated

The move from Orca/DSBS to R1 is a move in who the teacher *is*. Orca's teacher was GPT-4, a chat model with no reasoning-specific post-training. R1 is a reasoning model: it was *trained by RL to emit long traces* ([[deepseek-r1]], [[deepseek-r1-distill-synth]]), so when you sample from it, the trace distribution you get is already adapted to verifier-grounded correctness. The distillation recipe thus looks almost boringly simple from the student's side — it is pure SFT — but the **teacher-side pipeline that produced the 800K trace pool** is anything but.

The [[deepseek-r1]] report says the R1 pipeline is **four stages** after the DeepSeek-V3-Base checkpoint:

1. **Cold-start SFT** on ~thousands of hand-cleaned long-CoT examples with human-readable format — fixes the R1-Zero readability problem (English/Chinese mixing, hollow `<think>` tags).
2. **Reasoning RL (GRPO + rule-based reward).** Hyperparameters publicly disclosed: LR 3e-6, KL coefficient 0.001, GRPO clip ratio ε = 10 (intentionally loose), rollout temperature 1.0, group size G = 16, max generation 32,768 tokens, 32 unique prompts/step → 512 training samples/step.
3. **Rejection-sampling SFT (RS-SFT)** — this is the stage that **produces the 800K distill corpus**: the stage-1 RL model samples multiple traces per prompt, a V3 judge filters them for readability + correctness, and the kept set is ~600K reasoning + 200K non-reasoning. This is the corpus that becomes the 6 distilled checkpoints' training set.
4. **Stage-2 Alignment RL** with helpfulness + harmlessness preference rewards (separate from the distill corpus, used for the final R1 model only).

**What's actually in the 800K, as far as anyone public knows:**

| Slice | Approx. size | What it is | Verifier used |
|---|---|---|---|
| Math (problem → R1 long CoT) | ~200–300K | NuminaMath / olympiads / AIME-style | Exact-match on final boxed answer (sympy) |
| Code (problem → R1 long CoT → solution) | ~200–300K | LeetCode / APPS / CodeContests | Public unit-test execution |
| Logic / science reasoning | ~50–100K | GPQA-style, logical puzzles | LLM-judge (V3 or V3-reasoning) |
| Non-reasoning SFT | ~200K | Writing, roleplay, translation, Q&A | No verifier; V3-judge for quality filtering |

The report is explicit that **this pool, not any additional trick, is what transfers**: the 6 distilled students (Qwen2.5-Math 1.5B/7B/14B/32B, Llama-3.1-8B, Llama-3.3-70B) are trained with **pure SFT** on this corpus. No RL on the student. The training lesson DeepSeek makes directly: **dense students benefit more from copied reasoning structure than from rediscovering that structure via their own RL**. A dense 32B student doing RL from scratch needs enormous compute for a weaker result than SFT-on-R1-traces for one epoch.

**The opacity line.** The public DeepSeek docs give the 800K figure and the broad composition, but not a per-source breakdown, not the rejection-sample yield ratio, not the judge prompt, not the data-order. The open reproductions (next section) exist in large part to fill that gap.

---

## 4. Open R1-distill reproductions — the filter is the recipe

Three 2025 efforts — [[bespoke-stratos]], [[openr1]], [[sky-t1]] — all reproduce the R1-distill idea with fully-open data, but with very different answers to the question *how much curation, and which filter?*. Comparing them is the cleanest way to see what the recipe primitives actually are.

| Attribute | Bespoke-Stratos-17k | OpenR1-Math-220k | Sky-T1-data-17K |
|---|---|---|---|
| Team | Bespoke Labs (Sathiamoorthy et al.) | HuggingFace (Tunstall, Beeching, Lambert, Ben Allal, Penedo, …) | NovaSky / Sky Computing Lab (UC Berkeley) |
| Teacher | DeepSeek-R1 (official API, 671B MoE) | DeepSeek-R1 (HF-hosted + API mix) | QwQ-32B-preview (open weights, local vLLM) |
| Prompt pool | 7K math (NuminaMath / MATH / AIME) + 5K code (APPS / CodeContests / TACO / LeetCode) + 5K science/STILL-2 | 220K math problems only (NuminaMath cn_k12 / olympiads / aops_forum / amc_aime / orca_math + AIME archive) | 10K math + 5K code + 2K science/STILL-2 |
| Sampling | T = 0.6, 1 trace per prompt + up to 3 retries on format fail | T = 0.6, 2 traces per prompt (some up to 8×) | T = 0.7, 1 trace per prompt, max 8K tokens |
| Filter — math | SymPy symbolic-equivalence on extracted `\boxed{}` answer | Math-Verify (open-source SymPy) on `\boxed{}` answer | SymPy on `\boxed{}` |
| Filter — code | Run candidate against public unit tests; reject any fail | n/a (math-only corpus) | Run against public unit tests |
| Filter — science/open | GPT-4o LLM-judge (correct/incorrect verdict) | n/a | GPT-4o-mini LLM-judge |
| Format filter | Reject missing `</think>` or missing `\boxed{}` | Reject missing `</think>` / `\boxed{}` | Rewrite via GPT-4o to canonical `<\|im_start\|>` chat template; drop filler preambles |
| Dedup | MinHash cross-prompt | Built into NuminaMath; no post-dedup | None published |
| Reject rate | ~30–50% (code dominates rejections) | ~20% (math-only; SymPy strict) | ~5% format failures; re-generate |
| **Final #traces** | **17,000** | **~440K (220K × 2)** | **17,000** |
| Median trace length | ~3K tokens (tail 10K+) | ~5K tokens (tail 30K; 10% >15K) | ~3K tokens (tail 10K) |
| Teacher cost | ~$800 API | ~$10K (HF H100 + API mix) | Negligible (local QwQ) |
| Student training cost | ~$4K (8×H100, hours) on Qwen2.5-32B-Instruct | Multi-day 8×H100 on Qwen2.5-7B-Instruct | $450 (8×H100, 19 hrs) on Qwen2.5-32B-Instruct |
| **Flagship student eval** | Stratos-32B: AIME24 ~63%, MATH500 ~93%, LCB ~57% | OpenR1-Qwen-7B: MATH ~80%, AIME24 ~40% (+3–5 AIME from follow-up GRPO) | Sky-T1-32B: AIME24 ~43.3%, MATH500 ~82.4%, LCB-Easy ~86.3%, GPQA-Dia ~56.8% |
| Delta vs R1-Distill-Qwen-32B (800K) | within 2–3 pts | (7B; not directly comparable) | ~20 pts behind on AIME (teacher ceiling: QwQ < R1) |
| License | Apache-2.0 dataset | Apache-2.0 dataset | Apache-2.0 dataset; teacher is open weights |

Three non-obvious lessons fall out of this table.

**4.1 Curation beats scale — but only up to a ceiling.** Bespoke-Stratos's 17K reaches within 2–3 points of R1-Distill-Qwen-32B's 800K. The marginal 783K traces buy you ~2 points on AIME. This is the same "less is more" lesson [[open-thoughts]] later verifies across 1000+ ablations; the student's latent capability plus the teacher's format template is what transfers, not the breadth of prompts. The ceiling is the teacher — 17K of R1 traces is not equal to 17K of QwQ traces on AIME.

**4.2 The verifier determines the upper bound.** Open-R1's Math-Verify catches algebraic/numeric equivalence but fails on geometry and proofs. Stratos's SymPy + tests + LLM-judge stack has the widest coverage and correspondingly the strongest student. Sky-T1's GPT-4o-mini judge for open-ended is the weakest filter and correlates with the weakest GPQA score. **Ablation from Stratos**: removing code-verification halves LiveCodeBench gain; removing math symbolic equivalence halves MATH gain. **Ablation from OpenThoughts**: LLM-labeled *difficulty* filters beat embedding-based or fastText filters on the question side.

**4.3 A stronger teacher is not always a better teacher.** [[open-thoughts]] ablation: **QwQ-32B beats DeepSeek-R1 as a teacher** for training a Qwen2.5-7B student, even though R1 wins every benchmark R1 is evaluated on. Explanations: (i) QwQ's output distribution is closer to Qwen2.5's base distribution → less distribution shift for the student; (ii) QwQ's traces are shorter (median 3K vs R1's 5K) → the student sees more (prompt, answer) pairs per token budget; (iii) R1's trace format has idiosyncrasies (occasional Chinese, hollow `<think>`) that survive filtering and confuse small students. Sky-T1 makes the same choice for a different reason — cost — and gets a weaker model, showing that "closer-to-student distribution" alone is not sufficient; the teacher must still be capable.

See [figures/distill-lineage.html](figures/distill-lineage.html) for the full genealogy tree (GPT-4 → Orca → Open-Orca / Dolphin; R1 → R1-Distill → Bespoke-Stratos → Sky-T1 → OpenThoughts) with clickable teacher metadata and trace counts.

---

## 5. What the student inherits from the teacher

Distillation-as-data is supervised learning on tokens from a specific distribution. The student learns **everything about that distribution**, not only the parts you meant to transfer.

**5.1 Stylistic tics.** R1's traces have characteristic reflection markers: *"Wait, let me reconsider..."*, *"Hmm, that doesn't seem right..."*, *"Let me verify this step..."*. These are not accidents — they are the behaviors that emerged under R1-Zero's RL pressure ([[deepseek-r1]] §"aha moment"). The student copies them verbatim. This is useful for reasoning tasks (the tic cues the student to do a self-check). It is an anti-feature for short-form tasks where the student now says "Wait, let me reconsider" before answering "2". [[orca-2]]'s Prompt Erasing is the most principled mitigation: train the student to *select* when to emit the tic, not to always emit it.

**5.2 Language mixing.** R1's RL stage was run on a bilingual (English/Chinese) base. Raw R1 outputs occasionally interleave Chinese characters in reasoning steps even for English prompts. The cold-start SFT + language-consistency reward in R1's later stages partially fix this, but residual leakage shows up in distilled students. Stratos and Open-R1 reject format-violation traces; the rejection is imperfect because the bilingual leak is mid-trace, not in the wrapper.

**5.3 Benchmark contamination.** AIME and MATH problems are public; the teacher may have memorized answers during its own training. The student memorizes the memorization. **Open-R1 flagged this**: gains on AIME25 are smaller than gains on AIME24 because the teacher is less saturated on the newer problems. The clean-evaluation protocol is to report on held-out contests (AIME25 when your teacher's cutoff is 2024, USAMO when AIME is contaminated, etc.) — not the standard eval the teacher trained on.

**5.4 Refusal patterns.** A teacher that emits `"I cannot help with that"` on safety-triggering prompts will teach the student to emit the same pattern, often with *the same phrasing*. This is desirable when the refusal is calibrated; it is undesirable when the refusal is over-broad (GPT-4-era "As an AI language model, I cannot ...") and now shows up in your student for prompts that are not actually harmful. [[dolphin]]'s "uncensoring" filter is exactly the operation of detecting and removing these patterns before training — a post-hoc correction that acknowledges the teacher-bias transfer is systematic, not occasional. Hartford's recipe:

```
1. Regex + classifier detect refusal patterns ("As an AI language model…", "I cannot …").
2. Classifier detect alignment-steering preambles (moral disclaimers).
3. Classifier detect formulaic bias disclaimers.
4. Drop those samples.
5. Train on the remainder.
```

This moves alignment to a separate training stage rather than baking it into SFT. It is philosophically controversial but operationally necessary if you want the student's alignment to be *your* alignment, not the teacher's.

**5.5 The "wrong-question-correctly" failure.** [[openr1]] calls this out explicitly: Math-Verify only checks the final `\boxed{}` answer equals the gold answer. If R1 misreads the problem, solves a *different* problem, and happens to produce an answer that also matches gold (common in multiple-choice or numeric answers with coincidental collisions), the trace passes the filter and poisons the student with a reasoning path that is valid-looking but solves the wrong question. No outcome filter catches this; process reward (ch-24) is the only defense.

---

## 6. OpenThoughts — the recipe meta-experiment

[[open-thoughts]] is the 2025 project that ran 1,000+ ablations over the data-recipe space. It is the single largest source of *empirical* answers about what actually matters in reasoning distillation. The headline findings every practitioner should internalize:

- **Sampling multiple answers per question is the easiest diversity trick** — ≥16× expansion per source with non-trivial gain. (Open-R1 does 2× per problem; Stratos does 1×.)
- **Strong source concentration beats source diversity.** Using a small number of top-quality problem pools outperforms a broader but noisier mix.
- **No answer-side filter beats keeping everything.** Once your filter is aggressive enough to catch format violations, further answer filtering (difficulty pruning, length pruning) loses more signal than it removes noise.
- **Question-side filtering matters more than answer-side filtering.** LLM-labeled difficulty and response-length filters on the *questions* outperform embedding-based or fastText heuristics.
- **Deduplication is domain-sensitive.** Exact dedup for math and science; **no dedup for code** (different problems with syntactically similar solutions should not collapse).
- **Teacher choice is not monotone with teacher benchmark score.** QwQ-32B > R1 as a teacher for Qwen2.5-7B even though R1 scores higher on every target benchmark.

The OpenThinker3-7B model — Qwen2.5-7B-Instruct SFT'd on the OpenThoughts3-1.2M corpus (850K math + 250K code + 100K science) with QwQ-32B as teacher — reaches AIME25 53%, LiveCodeBench 06/24-01/25 51%, GPQA-Diamond 54% — the strongest open-data 7B reasoning model as of the paper. The recipe primitives above are load-bearing; ablating any one costs measurable points.

---

## 7. Licensing cliffs — which teacher outputs are redistributable

This section is unglamorous and load-bearing. The reason three open reproductions exist with different teachers is not recipe preference; it is licensing.

| Teacher | Weight license | Output-license question | Redistributable corpus? |
|---|---|---|---|
| GPT-4 / GPT-4o (OpenAI API) | Proprietary | OpenAI ToS historically prohibits using outputs to train a competing model. Dataset hosts must comply with API ToS even if they release the dataset under Apache-2.0. | **Contested.** Dolphin-v1 claims Apache-2.0 on the *dataset* but the GPT-4 outputs' downstream use is governed by the API ToS the generator agreed to. Lawyer-gated in commercial settings. |
| Claude (Anthropic API) | Proprietary | Anthropic ToS similarly prohibits training competing models on outputs. | Not redistributable for training-competing-model purposes. |
| DeepSeek-R1 (671B MoE) | **MIT on weights**, per DeepSeek's release | DeepSeek explicitly permits model outputs for training, including distillation. | **Yes**, with attribution. Bespoke-Stratos, Open-R1, OpenR1-Math-220k all rely on this. |
| QwQ-32B-preview (Alibaba) | **Apache-2.0** (open weights) | Output license is inherited from the model license; Apache-2.0 permits derivative works including training. | **Yes.** Sky-T1-data-17K and OpenThoughts3 both leverage this. |
| Llama-3.x (Meta) | **Llama-3 Community License** (custom; not OSI) | Outputs used to train *another* Llama-3-derivative are permitted; outputs used to train a *non-Llama* model are permitted only up to 700M MAU threshold; explicit "attribute Llama" clause. | Yes, with the MAU and attribution constraints. |
| Qwen-2.5 (Alibaba) | Apache-2.0 (most sizes) | Same as QwQ. | Yes. |

The teacher-output license is the single biggest factor in whether your distill corpus is *publicly releasable as a dataset* vs *usable only internally*. The 2025 consolidation on DeepSeek-R1 and QwQ as teachers is not because they produce the best traces in every domain — it is because their outputs are the only ones that can be redistributed under permissive licenses at all, which in turn is why the open-reproduction community can iterate on recipes at all.

Two second-order consequences:

- **Student license is downstream of teacher license.** A student trained on GPT-4 outputs can be released under any license the student author chooses, but the student's *weights still carry the teacher's ToS risk* — OpenAI has not enforced against distilled students, but the legal position remains that distilling is a ToS violation.
- **Dataset commercial use is narrower than research use.** Stratos is Apache-2.0 because R1 outputs are freely licensed; Dolphin-v1's Apache-2.0 on the dataset is more fragile because the underlying GPT-4/GPT-3.5 outputs are API-governed, not model-license-governed.

For a commercial product, the current safe path is: **R1 or QwQ as teacher; Apache-2.0 / MIT on the dataset; attribute the teacher; keep the filtering code open for auditability.**

---

## Connections and what's next

- **ch-19 (rejection sampling / self-distillation)** — the other branch: rejection-sample *your own* rollouts. This chapter is rejection-sampling *someone else's* rollouts, with licensing consequences.
- **[[orca]] / [[orca-2]]** — the system-prompt-scaffolded distillation template; the 16-message list in §1 is still the reference point for "how do you elicit varied reasoning from a chat-tuned teacher."
- **[[distilling-step-by-step]]** — joint-label-and-rationale multi-task training; the conceptual ancestor of R1-Distill.
- **[[deepseek-r1]] / [[deepseek-r1-distill-synth]]** — the teacher pipeline and the 800K corpus composition; ch-24 (RLVR at scale) covers the R1-Zero → R1 RL path.
- **[[bespoke-stratos]] / [[openr1]] / [[sky-t1]]** — three open-reproduction recipes; §4's table is the comparison you will reach for when you need to choose a filter stack.
- **[[open-thoughts]]** — the 1000-ablation recipe study; §6's findings are the empirical baseline for any new reasoning-SFT corpus.
- **ch-21 (taxonomy-driven synthesis)** — GLAN / Nemotron / Phi-textbooks; the *complement* to this chapter — what happens when you have no strong teacher to distill and must synthesize from a taxonomy instead.
- **ch-22 (quality / diversity / gradient selection)** — LESS, DEITA, Prismatic-Synthesis; the *sample-selection* layer that sits on top of whatever corpus this chapter produced.
- **ch-23 (model collapse)** — what happens when you distill from a model that was itself distilled from a model; the iterative-distillation failure mode this chapter does not yet trigger but ch-23 measures.

## Further reading

- [[orca]] — Mukherjee 2023; 16 system messages, 5M ChatGPT + 1M GPT-4 progressive learning.
- [[orca-2]] — Mitra 2023; 5-strategy instruction, Prompt Erasing.
- [[distilling-step-by-step]] — Hsieh 2023 ACL; joint (label, rationale) multi-task; T5-770M beats PaLM-540B few-shot.
- [[deepseek-r1]] — Guo et al. 2025 Nature; 4-stage pipeline, GRPO hparams, 800K corpus composition.
- [[deepseek-r1-distill-synth]] — blog/README extract; corpus-opacity caveats.
- [[bespoke-stratos]] — Sathiamoorthy 2025; 17K / $800 / SymPy+tests+LLM-judge.
- [[openr1]] — HF Open-R1 2025; 220K × 2-trace math corpus; Math-Verify.
- [[sky-t1]] — NovaSky 2025; $450 QwQ recipe; GPT-4o rewriter.
- [[open-thoughts]] — Guha 2025; 1000+ ablations; QwQ > R1 as teacher.
- [[dolphin]] — Hartford 2023–2025; Orca reproduction + refusal filter as the canonical teacher-bias-removal recipe.

## Companion visualization

**[figures/distill-lineage.html](figures/distill-lineage.html)** — interactive genealogy tree of the two distillation lineages. Left branch: GPT-4 → Orca → Open-Orca / Dolphin (system-prompt-scaffolded, chat-teacher era). Right branch: DeepSeek-R1 → R1-Distill / Bespoke-Stratos / Open-R1; QwQ-32B → Sky-T1 / OpenThoughts (reasoning-teacher era). Click any node for teacher model, trace count, filter stack, license terms, and flagship student eval numbers. Use it to internalize which decisions carry forward across generations (system-prompt scaffolding → `<think>` wrapper; MinHash dedup → cross-prompt filtering) and which are local fashion choices (number of traces, T=0.6 vs 0.7).
