<!-- chapter: ch-19
     track: synthetic
     title: Generation Methods — Bootstrap, Evol, Extraction, Persona, Rephrase
     sources: [[self-instruct]], [[alpaca]], [[evol-instruct]], [[wizardmath]], [[wizardcoder]], [[code-evol-instruct]], [[magpie]], [[persona-hub]], [[rephrasing-the-web]], [[humpback]]
     figures: figures/generation-methods.html
-->

# Chapter 19 — Generation Methods: Bootstrap, Evol, Extraction, Persona, Rephrase

> **Core insight.** There are only five independent ways to make a synthetic instruction. Every pipeline in open post-training from 2023 onward is a composition of these five. **Bootstrap** (Self-Instruct) asks the teacher to extrapolate from a seed pool. **Evolve** (Evol-Instruct) rewrites an existing instruction along a complexity or diversity axis. **Extract** (Magpie) asks an aligned model for an instruction *without any prompt* by exploiting its chat template. **Persona-condition** (Persona-Hub) attaches a "who" that steers the teacher away from its modal response. **Rephrase** (WRAP) grounds generation in a real document and rewrites it. The five methods trade off differently on four axes — cost, diversity, seed dependence, and failure mode — and you pick the method the way you pick an optimizer: by the shape of the data you already have.
>
> **Guideline.** If you have 175 good seed tasks and an API budget, bootstrap. If you already have a 52K SFT set and want harder questions, evolve. If you have GPU but no API and the target domain is well-covered by Llama-3-Instruct, extract. If your synthetic data is collapsing onto one voice, condition on personas. If your raw material is noisy web text and the goal is pretraining efficiency, rephrase. Compose them — every serious 2025 pipeline does.

---

## Why this chapter exists

Ch-18 set up the generator–verifier stack: who calls the teacher, how rollouts are framed, why verifier-gated synthesis beats naive sampling. This chapter fills in the first slot — the *generator* — with the five method families that dominate open post-training. Ch-20 will pick up with distillation-as-data (Orca / R1-distill), ch-21 with top-down taxonomy synthesis (GLAN / Phi), ch-22 with quality-driven selection across the output of any of these generators.

The order below is chronological because each method was invented in reaction to the previous one's limit:

1. **Self-Instruct (2022)** — the first bootstrap. Answers the question "can we make instruction data without human annotators?"
2. **Alpaca (2023)** — the same bootstrap on a stronger teacher; answers "can a small academic group afford this?"
3. **Evol-Instruct (2023)** — answers "is there a complexity distribution beyond what seeds cover?"
4. **WizardMath / WizardCoder (2023)** — answers "can evolution be specialized per domain?"
5. **Humpback (2023)** — answers "can we invert the direction: text → instruction?"
6. **WRAP (2024)** — answers "can rephrasing replace crawling at equal pretraining loss?"
7. **Persona-Hub (2024)** — answers "what single knob most controls output diversity?"
8. **Magpie (2025)** — answers "do we even need a prompt?"

Each answer closed one failure mode of its predecessor. Reading the chapter as a chain of closed failures is more useful than memorising five method names.

---

## 1. Bootstrap — Self-Instruct and the seed-plus-sample-plus-filter loop

From [[self-instruct]]: the first pipeline that produced usable instruction data from an LM alone. The full recipe is four stages.

**Stage 1 — Seed pool.** 175 human-written tasks, one instruction plus one instance each, covering classification, generation, open-ended, extraction. That's it. The single most misunderstood number in the paper: the seed pool is deliberately small. Larger seeds accelerate the ROUGE filter's rejection rate and the output manifold collapses faster.

**Stage 2 — Instruction generation.** Eight in-context examples (six seed, two previously generated) prompt the LM for a new task. The verbatim template from the paper:

```
Come up with a series of tasks:
Task 1: <seed 1>
Task 2: <seed 2>
...
Task 8: <seed 8>
Task 9:
```

**Stage 3 — Classification branching.** The LM is asked whether the generated instruction is a classification task. If yes, instances use an *input-first* template (to avoid the LM always producing the same label). If no, *output-first*. This branch is load-bearing — without it, classification tasks degenerate to a single modal label.

**Stage 4 — Filtering.** Three rules:

- **Drop any instruction with ROUGE-L > 0.7 against any existing instruction.** This is the verbatim threshold from the paper; anything laxer and the set collapses onto paraphrases of the seeds.
- Drop instances where input == output, outputs are too short/long, or the instruction mentions "image/graph/file" (the LM cannot actually produce images).
- Drop ill-formatted generations.

**Yield.** From ~252K raw generations the filter accepts ~52K instructions paired with ~82K instances. Applied to text-davinci-001-era GPT-3, this produces +33 absolute points on Super-NaturalInstructions — matching InstructGPT-001, which used private human data.

Why this works: the LM has latent knowledge of many tasks it cannot *plan* to express, but can *elicit* given an in-context nudge. The 8-shot prompt is the nudge. The ROUGE filter enforces that each new draw lands in a previously-unseen region of instruction-space. The pipeline is a Markov chain over task-space where the ROUGE filter is the rejection step.

See [[excerpts/self-instruct]] for the full operator breakdown and the classification-branch failure modes.

---

## 2. Alpaca — the cheap replication

[[alpaca]] is not a new method. It's Self-Instruct rerun with three changes: `text-davinci-003` as teacher (stronger than 001), LLaMA-7B as student (open base), 52K accepted instructions (no new filter). The paper's contribution is **operational** — it proved the pipeline cost under $500 for data generation and under $100 for fine-tuning. That single cost number opened synthetic SFT to every academic group with a lab credit card.

The limit: Alpaca's student is frozen against a particular teacher's modal style. Ablating the teacher to GPT-3.5 barely changes downstream skills; ablating to GPT-4 introduces GPT-4's refusal patterns wholesale. The teacher is a style filter, and every descendant (Vicuna, WizardLM, every X-Instruct set) inherits it. The Alpaca-era meme "we distilled GPT-3.5 into LLaMA" is technically accurate and strategically misleading — what was really distilled was the teacher's *preferred sub-manifold* of instruction-space.

---

## 3. Evolution — Evol-Instruct's five In-Depth + one In-Breadth operators

[[evol-instruct]] introduced the **complexity axis**. Self-Instruct's ROUGE filter enforces diversity in *word space*; it does not touch the *difficulty* distribution. Alpaca's 52K is roughly flat in difficulty. Evol-Instruct's key claim: the complexity histogram is a first-class training knob, and a long tail of hard instructions improves downstream skill more than another 52K of median-difficulty ones.

The paper enumerates six operators applied by prompting a strong teacher LLM. The verbatim prompt headers from the paper:

**In-Depth Evolving (make the instruction harder):**
1. **Add constraints** — "please add one more constraints/requirements into #The Given Prompt#".
2. **Deepening** — "increase the depth and breadth of the #The Given Prompt#".
3. **Concretizing** — "replace general concepts with more specific concepts".
4. **Increased reasoning steps** — "if #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning".
5. **Complicate input** — add code blocks, tables, XML, or nested structure to the input itself.

**In-Breadth Evolving (make the instruction more diverse):**
6. **Mutation** — generate a brand new instruction in a rarer domain, same overall topic.

**Pipeline.** Seed = 52K Alpaca instructions. Per seed, pick one operator at random, prompt the teacher to apply it, generate a response. Run an **elimination step**:
- LLM's own "same-or-similar" check against the input — drop.
- Response contains "sorry" / refusal markers — drop.
- Punctuation-only or empty response — drop.
- Response copies the input verbatim — drop.

Iterate 4 rounds. Yield: ~250K evolved instructions after filtering. SFT on LLaMA-7B/13B/70B with the original + evolved mix.

Why this works: the operators are chosen to push *orthogonally* to the original instruction. "Add constraints" moves the example along the difficulty axis; "mutation" moves along the topic axis. The two In-Breadth vs In-Depth families decorrelate, which is why mixing them produces the characteristic long-tail complexity histogram in the paper's Figure 3.

The limit: the teacher refuses to evolve beyond its own competence. In practice "increased reasoning steps" applied twice to a math problem saturates at the teacher's ceiling — GPT-4 will not produce a problem it cannot solve. This is why WizardMath had to specialize the operators.

See [[excerpts/evol-instruct]] for the verbatim prompts, rejection statistics, and operator collision analysis.

---

## 4. Domain specialization — WizardMath and WizardCoder

**WizardMath** ([[wizardmath]]). The math-specialization of Evol-Instruct adds a critical twist: *bidirectional* evolution. Operators run in two directions:

- **Downward evolution** — reduce constraints, replace concepts with simpler ones, shorten the chain, make arithmetic easier. This generates grade-school variants of competition-level seeds.
- **Upward evolution** — add constraints, compose with another concept, increase reasoning depth, require multiple solution steps. This generates competition-level variants of GSM8K seeds.

The downward direction was the surprise. It's not obvious that training a 70B model on *easier* problems helps. The paper's claim: a broader difficulty spectrum smooths the reasoning manifold — the model learns to recognize a problem's level before solving it, rather than always attempting the hardest path. This shows up in the GSM8K/MATH ablation where SFT with downward-only, upward-only, and bidirectional evolution all underperform the bidirectional blend.

The RLEIF (Reinforcement Learning from Evol-Instruct Feedback) stage that follows SFT is the first appearance of the IRM × PRM reward product that ch-26 will unpack in detail.

**WizardCoder** ([[wizardcoder]]). The code specialization replaces the six generic operators with five code-native ones:

1. Add new constraints or requirements.
2. Replace a common requirement with a less common one (`deque` instead of `list`).
3. Increase depth / reasoning steps.
4. Deepen problem complexity (time/space bounds, edge cases, misleading wordings).
5. Require a specific language or library.

Seed = 20K Code-Alpaca. Yield = ~78K evolved pairs after three rounds and the standard elimination filter. WizardCoder-15B on StarCoder-base hits HumanEval 57.3 / HumanEval+ 50.6 at release — beating Claude and Bard. See [[excerpts/wizardcoder]] for the operator-by-operator HumanEval ablation.

The structural lesson from both Wizards: **operators are domain-specific**. Running generic Evol-Instruct on code wastes mass on operators that don't touch the code-specific failure modes (edge cases, complexity bounds, library idioms). Running it on math produces solvable problems but not the difficulty spread that smooths the reasoning manifold. Match operators to the target skill's failure surface.

---

## 5. Extraction — Magpie's prefix-only trick

[[magpie]] asks the obvious but previously unexplored question: does an aligned model *need* a prompt to emit an instruction? The answer is no.

An instruction-tuned model's chat template looks like:

```
<|start_header_id|>user<|end_header_id|>\n\n{user_instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_response}<|eot_id|>
```

Magpie's trick: feed the model only the **pre-query prefix** — `<|start_header_id|>user<|end_header_id|>\n\n` — and sample until EOS. The model, having seen millions of user turns during instruction tuning, fills in a plausible user instruction on its own. No seed, no API, no prompt engineering.

Then feed that instruction back through the full template to get the response. Two calls, one aligned model, arbitrary scale.

**Numbers from the paper.** MAGPIE-Air: 3M raw pairs from Llama-3-8B-Instruct, 206 GPU hours. MAGPIE-Pro: 1M raw pairs from Llama-3-70B-Instruct, 614 GPU hours. Eight filtering metrics (input length, output length, task category, input quality, input difficulty, minimum neighbor distance via `all-mpnet-base-v2` + FAISS, reward from `FsfairX-LLaMA3-RM-v0.1`, reward-difference). Thresholds `tau1 = -12` for reward, `tau2 = 0` for reward-difference. Safety flag rate <1% per `Llama-Guard-2`.

Why this works: instruction-tuned models have a sharp, low-entropy distribution on the pre-query prefix — the user-turn continuation is the part they were *trained* to predict. Sampling from that distribution at temperature 1 produces diverse but well-formed instructions, because the model's posterior over "plausible user queries" is exactly the training-distribution prior.

Why you filter heavily: the extracted instructions inherit the *teacher's* instruction distribution, not the real world's. Llama-3-Instruct sees more "explain X" than "tell me about your day" prompts, and Magpie's raw outputs reflect that. The 3M → 300K curation is where the method lives or dies. See [[excerpts/magpie]] for the filter-threshold ablation.

**What Magpie removes from the pipeline.** No seed pool. No teacher API (the aligned model is open-weight). No prompt engineering. What remains is filter design and compute.

---

## 6. Persona conditioning — Persona-Hub's 1B diversity primitive

[[persona-hub] argues diversity is the bottleneck of every other method. Self-Instruct's 252K → 52K filter rate is diversity loss; Evol-Instruct's operator-collision rate is diversity loss; Magpie's 3M → 300K curation is diversity loss. Every pipeline is fighting its own teacher's mean-response mode.

Persona-Hub's lever: **attach a persona to the generation prompt**. The same "write a word problem about fractions" instruction, conditioned on 1,000 different personas (emergency-room nurse, jazz drummer, eighth-grade teacher in rural Kenya, retired bridge engineer) produces 1,000 problems that cluster around the persona's domain rather than the LM's modal voice.

**Scale.** 1,015,863,523 personas, assembled by two methods:

- **Text-to-Persona.** For each document in a web corpus (RedPajama v2), prompt the LM with "Who is likely to read / write / like / dislike this text?" Collect personas.
- **Persona-to-Persona.** Expand via relationship prompts: "Given persona X, list 10 people who might interact with them professionally / personally / antagonistically." Iterate 6 rounds.

Deduplicate with MinHash at 0.9 similarity, then embedding cosine at 0.9 (tighter if diversity matters more than count). The three prompting modes:

- **Zero-shot:** persona + task specification only.
- **Few-shot:** add demonstrations.
- **Persona-enhanced few-shot:** derive personas for each demonstration, condition on those too.

**The key empirical result.** Output similarity is *less* than persona similarity. Two personas that are 80% similar produce problems that are only 40–50% similar. The personas act as a *distributional amplifier* — the teacher injects topic-space variance that wasn't in the persona space itself.

**Math scaling experiment.** 1.07M persona-synthesized problems, Qwen2-7B SFT → 64.9% on MATH. At release, that matched gpt-4-turbo-preview at 7B. The scaling curve does not saturate at 1M; persona-conditioning appears to have headroom comparable to raw data scaling.

When to reach for Persona-Hub: synthetic data is collapsing onto one voice and other methods (Evol operators, Magpie filter) can't break it. The persona is the diversity knob when style diversity is the binding constraint. See [[excerpts/persona-hub]] for the dedup-threshold ablation and the math-validity audit (96.5%).

---

## 7. Web rephrasing — WRAP's chunk-and-rewrite

Every method above produces *instruction* data for SFT. [[rephrasing-the-web]] is different: it produces *pretraining* data by rewriting raw web text into cleaner styles.

The pipeline. Chunk a C4 document to ~300 tokens (longer chunks cause information loss during rephrase). Feed each chunk to frozen Mistral-7B-Instruct with one of four style prompts:

- **Easy** — grade-school vocabulary.
- **Medium / Wikipedia-like** — encyclopedic, neutral tone.
- **Hard / Terse** — dense, technical phrasing.
- **Q/A** — reformulate as a question-answer pair.

Mix real and synthetic 1:1 — each document appears as raw text and as a rephrase. Pretrain a decoder-only transformer on the mix.

**Results.** At 128M / 350M / 1.3B parameters, WRAP reports ~3× pretraining speedup, ~5× less data for matched loss, >50% perplexity reduction across Pile subsets. The headline: **350M trained on 15% of C4 with WRAP beats 1.3B trained on all of C4**. That is a 40× data-efficiency multiplier and a 4× parameter-efficiency multiplier stacking in the same experiment.

Why this works: raw C4 is style-flat and information-dense-in-parts, boilerplate-heavy in others. The rephrase acts as both a quality filter (boilerplate is paraphrased away) and a style amplifier (Q/A style is denser in extractable facts per token than raw prose). The 1:1 mix preserves the real-world distribution so the model doesn't overfit to Mistral-7B-Instruct's voice.

The cautions from the paper: rephrases longer than 300 tokens drop information; a lightweight post-process must strip boilerplate intros ("Here's a paraphrase...") or they contaminate pretraining; and the "reproducible against held-out Mistral rephrases" sanity check catches leakage. See [[excerpts/rephrasing-the-web]] for the style-ablation and the data-efficiency plot.

This is the foundation ch-21 (Phi-textbooks, Nemotron synthetic) builds on at frontier scale.

---

## 8. Humpback — backtranslation as the inverse direction

[[humpback]] closes the loop. Self-Instruct: instruction → instance. Humpback: instance → instruction. Given a raw document, prompt an aligned seed model to generate the instruction that would plausibly produce that document as a response. Curate the `<inferred_instruction, document>` pairs by quality, fine-tune.

The method matters less for open pipelines in 2025 than for what it *proved*: instruction-data space and response-data space are isomorphic under a competent teacher. You can generate either one and recover the other. Every later pipeline treats this as assumed — Magpie generates instructions from aligned-model priors; WRAP generates responses (rewrites); taxonomy methods (ch-21) generate both from a tree.

Humpback's operational lesson: **self-curation beats external curation when the curator is the same model family as the generator**. The seed model's judgment is better-calibrated on its own outputs than a fresh reward model would be. This is the same observation Magpie's filter stack exploits.

---

## 9. The comparison table — cost, diversity, seed, failure

The five method families on four axes. Cost is per 1K accepted examples assuming current (2025) API pricing or open-weight compute. Diversity is measured as 1 – mean cosine similarity across embedded outputs (higher is better). "Required seed" is the minimum artifact the method cannot start without.

| Method | Cost / 1K examples | Diversity (1 - avg cos) | Required seed | Characteristic failure mode |
|---|---|---|---|---|
| **Self-Instruct bootstrap** | ~$1–3 (API teacher) | 0.45 | 175 human-written tasks | Mode collapse — filter saturates; new draws become paraphrases of seeds |
| **Alpaca cheap replication** | ~$1 (text-davinci-003 era) | 0.42 | Self-Instruct seeds (175) | Teacher-style lock-in — student inherits GPT-3.5's refusal & verbosity patterns |
| **Evol-Instruct evolution** | ~$5–10 (multi-round teacher calls) | 0.58 (after 4 rounds) | Existing 52K SFT set | Operator saturation — teacher refuses to evolve beyond its own skill ceiling |
| **WizardMath / WizardCoder** | ~$10–20 (bidirectional + filter) | 0.55 (domain-restricted) | Domain seeds (GSM8K / Code-Alpaca) | Reward-hacking downstream (PRM noise); benchmark contamination via operator |
| **Magpie extraction** | ~$0.05 (open-weight compute only) | 0.50 pre-filter, 0.68 post-filter | None — aligned model only | Distribution narrowing — extracted instructions match the aligned model's training prior, not real users |
| **Persona-Hub conditioning** | ~$2–8 (teacher call + persona) | **0.72** (post-dedup) | A persona bank (released 200K, full 1B gated) | Persona != demographics — model-inferred identities embed biases |
| **WRAP rephrase (pretraining)** | ~$0.10 per 1K tokens (open-weight teacher) | 0.40 (style-only) | Raw web corpus | Boilerplate leakage — "Here's a paraphrase..." contaminates training if unfiltered |
| **Humpback backtranslation** | ~$1–3 (aligned seed model) | 0.48 | Raw documents + seed aligned model | Inferred-instruction mismatch — the recovered instruction may not be what a user would actually ask |

The first three rows (Self-Instruct, Alpaca, Evol) sit in the same diversity band (0.42–0.58). The jump to Persona-Hub (0.72) is the biggest single-method diversity gain in the table, and it's why persona-conditioning shows up as a *component* inside later pipelines rather than as a standalone method. The Magpie row is the biggest cost drop (two orders of magnitude vs API-teacher methods) once you already have the aligned model weights — which is increasingly true for open teams.

---

## 10. How these compose in practice

Real 2025 pipelines are not pure. A representative blend:

- **Seed** with Self-Instruct on a small hand-curated task set (the 175-seed trick is still load-bearing).
- **Extract** at scale via Magpie on the most recent aligned open model.
- **Evolve** the combined pool with three Evol-Instruct operators (skip "complicate input" if the target is chat).
- **Condition** a sampled subset on a 10K-persona bank to rescue diversity.
- **Rephrase** any long-form or pretraining-adjacent slice with WRAP-style chunk-and-rewrite.
- **Filter** the whole thing with the ch-22 quality stack.

That composition is a superset of the five method families, not a replacement. Each family contributes a distinct *inductive bias* to the final distribution: Self-Instruct seeds set the task topology; Evol-Instruct stretches the difficulty axis; Magpie widens the volume cheaply; Persona-Hub breaks style collapse; WRAP ports it all to pretraining scale.

The interactive companion — [`figures/generation-methods.html`](figures/generation-methods.html) — runs a single source example through all five methods side-by-side, so the reader can see why the generated outputs diverge even though the "input" is shared.

---

## Connections

- [[ch-18]] — generator/verifier framing. This chapter populates the generator slot.
- [[ch-20]] — distillation-as-data (Orca / R1-distill) uses these generators + strong teachers.
- [[ch-21]] — taxonomy synthesis (GLAN / Phi) is the top-down alternative to bootstrap.
- [[ch-22]] — quality selection over any of these generators.
- [[excerpts/self-instruct]], [[excerpts/evol-instruct]], [[excerpts/magpie]], [[excerpts/persona-hub]], [[excerpts/wizardcoder]], [[excerpts/rephrasing-the-web]], [[excerpts/wizardmath]] — deep walkthroughs.
