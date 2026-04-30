<!-- chapter: ch-18
     track: synthetic
     title: The Synthetic-Data Design Pattern
     sources: [[self-instruct]], [[nemotron-4-synthetic]], [[apigen]], [[openmathinstruct-2]], [[nathan-lambert-synthetic-data]], [[sebastian-raschka-synthetic]], [[synthetic-data-scaling-laws]]
     figures: figures/synth-loop.html
-->

# Chapter 18 — The Synthetic-Data Design Pattern

> **Core insight.** Every production synthetic-data pipeline — from Self-Instruct's 52K bootstrapped instructions to Nemotron-4's 98%-synthetic alignment stack to APIGen's three-layer function-call verifier to OpenMathInstruct-2's 14M CoT traces — is an instance of the same six-stage operational loop: **generate -> filter -> dedup -> verify -> select -> mix**. Papers differ in *which* stages they invest in, not in *which* stages exist.
>
> **Guideline.** Before reading any synthetic-data paper, locate its six stages. Before designing one, write the loop down *first* and staff each stage with a concrete mechanism. If a stage is empty — no verifier, no dedup, no mix policy — that is not a simplification; it is a known failure mode.

---

## Why this chapter exists

The synthetic track of this course covers twelve chapters and dozens of pipelines. Taught as a list of papers, the material is a blur: Self-Instruct begets Alpaca begets Evol-Instruct begets WizardLM begets Magpie begets Persona-Hub begets UltraFeedback begets Nemotron-4 begets Tulu-3 — and by the tenth acronym you are memorising rather than understanding.

The reason the list is taught that way is an accident of publication order. The reason it *shouldn't* be taught that way is that every one of those papers is the same six-stage loop with different plug-ins. Nathan Lambert, in "Frontiers in Synthetic Data," makes the point bluntly: "synthetic data can do almost all of the work" given **a strong base model plus robust verification** — the verification layer is the bottleneck, and it is always the same shape regardless of modality.

This chapter installs the loop as your permanent mental model. After this, every later chapter — generation methods (ch-19), domain-specific pipelines for code / math / tools / long-context, filter and verifier design, judge calibration, mixing policies — is an investigation of one stage of the loop. You will recognise pipelines you've never read before because you already know the six questions to ask.

Three things to walk away with:

1. The six-stage loop, memorised in order, with the **one-line question** each stage answers.
2. A **4 x 6 table** showing how Self-Instruct, Nemotron-4, APIGen, and OpenMathInstruct-2 instantiate every stage — including where they leave a stage empty and pay for it.
3. A taxonomy of **per-stage failure modes**, so that when you see "the dataset is 2M samples but downstream accuracy is flat" you can locate which stage broke.

---

## 1. The operational loop

The six stages, in the order data flows through them:

```
 [seeds + task def]
        |
        v
 (1) generate  --->  (2) filter  --->  (3) dedup  --->  (4) verify  --->  (5) select  --->  (6) mix
        ^                                                                                      |
        |___________________________ optional iteration (new seeds, teacher rotation) __________|
```

One line per stage — memorise these:

- **(1) Generate.** *What does the teacher produce?* A prompt, a response, a trajectory, a rewrite.
- **(2) Filter.** *What do we drop for structural or surface reasons?* Schema violations, wrong language, ill-formatted generations, too-long/too-short outputs, banned topics.
- **(3) Dedup.** *What do we drop for being redundant?* Near-duplicates by n-gram overlap, ROUGE, MinHash, or embedding distance.
- **(4) Verify.** *What do we drop for being wrong?* Gold-answer match, execution check, semantic judge, reward-model score.
- **(5) Select.** *Which of the surviving samples do we keep and in what quantity?* Difficulty, coverage, informativeness (IFD, LESS), reward-ranking for preference pairs.
- **(6) Mix.** *How does this dataset compose with others during training?* Ratio vs real data, ratio across task families, curriculum position.

Filter vs verify is the distinction most people flatten and most pipelines get wrong. **Filter is cheap surface-level rejection** (regex, JSON parse, length threshold). **Verify is expensive ground-truth rejection** (running code, matching a symbolic expression, scoring with a reward model, adjudicating with an LLM judge). Every high-quality pipeline pays the verify cost. Every low-quality pipeline skips it and calls filtering "verification."

The loop is iterative. Once you have a cleaned corpus, you often want to feed it back as new seeds (Nemotron's self-instruct-style seeding), retrain the teacher on its own cleaned output (Self-Rewarding LM, Nemotron iterated checkpoints), or rerun verification with a stronger judge (West-of-N). We return to the iteration axis in ch-28.

See [figures/synth-loop.html](figures/synth-loop.html) for an interactive version: click any stage to see the failure mode and a concrete example pulled from the four flagship pipelines below.

---

## 2. Why the loop is the right unit of analysis

A pipeline is a *configuration* of the loop. Four examples, in the modalities the rest of this track covers:

- **Bootstrap instruction data (Self-Instruct, Alpaca, WizardLM).** Strong on (1) generate and (2) filter; minimal (4) verify — just format validity.
- **Alignment at scale (Nemotron-4, Tulu-3).** Strong on (1) generate and (4) verify-via-reward-model; (5) select uses RM scores for DPO chosen/rejected selection; (6) mix is staged (code SFT then general SFT).
- **Tool / function calling (APIGen, ToolACE, xLAM).** Weak on (1) generate (any capable LLM suffices); brutally strong on (4) verify — three layers (format, execution, semantic) stacked in series.
- **Reasoning-trace synthesis (OpenMathInstruct-2, Numina-Math).** Strong on (1) generate via expensive teacher sampling (K=32 per problem); strong on (4) verify via SymPy gold-answer match; weak on (3) dedup because diversity comes from per-problem solution variety, not cross-problem novelty.

The identical-shape observation has real leverage. If you understand why APIGen's execution layer rejects ~40% of raw generations, you already know why OpenMathInstruct-2's SymPy check rejects its own large fraction and why Self-Instruct's ROUGE-L filter was insufficient on its own. They are the same slot of the same loop, populated with modality-specific verifiers. **A verifier you understand from code generalises to math by replacing "Python sandbox" with "SymPy check"** — this porting is what makes the loop the right unit of analysis.

### 2.1 The cost profile of the loop

The loop also forces you to think about *where the money goes*. Nathan Lambert's 2025 post-training write-up is explicit: "post-training now consumes a substantial fraction of total FLOPs," driven by (a) multi-round rejection sampling, (b) multi-model generation fleets, (c) large RL rollouts. Broken down by stage:

- **(1) Generate** is the dominant *inference* cost. OpenMathInstruct-2's teacher sampling alone consumed ~650K H100-hours — more than the student training run by a large margin. APIGen's ~$8K teacher-API spend is small only because the dataset is 60K examples; scale to math-style K=32 per problem and the number matches.
- **(2) Filter** is near-free — regex and JSON parsing.
- **(3) Dedup** is moderate — MinHash on tens of millions of pairs is manageable; ROUGE-L against a growing set is O(n^2) and does bite Self-Instruct-scale runs.
- **(4) Verify** has two regimes. Cheap verifiers (SymPy, Python sandbox, schema match) are bounded by CPU minutes. Expensive verifiers (GPT-4-as-judge, reward-model scoring) are often comparable to stage (1) in cost. APIGen runs both — sandbox execution and a GPT-4 judge — so stage 4 alone can exceed stage 1.
- **(5) Select** is usually cheap unless it involves running an RM over every sample (Nemotron does this once per candidate).
- **(6) Mix** is a training-time decision, not a data-time cost.

The operational consequence: **your pipeline is mostly spending money on stages 1 and 4**, and everything else is plumbing. Budget discussions that ignore stage 4 are pretending verification is free. It isn't, and treating it as optional is how you end up with a 14M-example corpus whose downstream gains plateau around 1M.

---

## 3. Four pipelines, six stages — the matrix

The table below is the thing to print and stare at. Rows are stages; columns are flagship pipelines. Each cell says what that pipeline does at that stage and, in parentheses, a concrete number from the source.

| Stage | Self-Instruct (2022) | Nemotron-4 340B (2024) | APIGen (2024) | OpenMathInstruct-2 (2024) |
|---|---|---|---|---|
| **(1) Generate** | GPT-3 (text-davinci-001) prompted with 8 in-context examples (6 seed + 2 generated) for new instructions, then instance generation. 175 seed tasks. Output: ~252K raw candidates. | Task-family-seeded prompts across code, QA, topic-following, function-calling, refusal. Iterated Nemotron checkpoints as teacher. Genetic Instruct for code (self-instruct + WizardCoder mutations + LLM fitness). | Sample (k=1..3) functions from 3,673-API pool, prompt DeepSeek-Coder-V2 or GPT-4 for (query, gold call) pair. Diversity sampler upweights rare API categories. | Llama-3.1-405B-Instruct samples K=32 CoT solutions per problem at T=1.0, top-p=0.95. 600K problems after seed augmentation from MATH + GSM8K (7.5K each). |
| **(2) Filter** | Drop ill-formatted generations; drop instances where input == output; drop "image/graph/file" mentions (GPT-3 cannot handle them); length bounds on outputs. | Staged: code filter separate from general filter. Topic-following track keeps distractor turns (intentionally noisy). Format validity on code compilation. | **Layer 1 of 3:** JSON must parse, schema fields present, types match (int/str/bool/enum/list). | Regex-extract boxed final answer; reject when no boxed answer exists. |
| **(3) Dedup** | **ROUGE-L > 0.7** against all prior instructions. Drops ~50% of raw instruction candidates. | Light cross-prompt dedup; the paper relies more on task-family partitioning than on aggressive near-dup removal. | **MinHash** over (query, call) pairs across the corpus. | Near-duplicate suppression **within a problem's accepted solutions**; different problems intentionally produce overlapping solutions and that is fine. |
| **(4) Verify** | **Minimal** — no gold-answer checker, no executor. Just format + ROUGE. This is the paper's acknowledged weakness. | **Reward model** (Nemotron-4-340B-Reward) scores responses; also used as judge to select "chosen" vs "rejected" in preference pairs. Small human anchor (~20K) calibrates RM. | **Layers 2 and 3 of 3:** (2) execute the call in a 5-sec Python sandbox against the reference impl; reject on exception. (3) GPT-4 judge shown (query, call, execution-result); only "Yes" accepted. Combined rejection ~40% of raw. | **SymPy** symbolic equivalence (MATH) + exact numeric match (GSM8K). Residual false-positive rate (right answer, wrong reasoning) ~7% on human-audited sample. |
| **(5) Select** | Implicit: everything that passes (2)+(3) goes in. No difficulty scoring, no coverage targeting. | **RM-score-driven selection** for DPO/RPO: highest-scoring response = chosen, lowest-scoring = rejected. Selection is the entire preference-pair construction step. | Keep all survivors across four call-shape buckets (simple / multiple / parallel / parallel-multiple). Balance by bucket at the end. | Keep all verified solutions; no per-solution difficulty weighting. Selection is on *problems* via question augmentation upstream, not on *solutions* downstream. |
| **(6) Mix** | Single-stage SFT on GPT-3. No mix — it is the whole dataset. | **Staged**: code SFT first (800K), then general SFT (200K); then DPO (160K) followed by RPO (300K). Also adds alignment-style QA in continued pretraining. Total pipeline >98% synthetic. | Single-stage SFT on Mistral-7B / Mixtral (xLAM models). | Single-stage SFT on Llama-3.1-{1.5B, 8B, 70B}. No real-data mixing at this stage. |
| **Output scale** | 52K instructions / 82K instances (from 252K raw) | ~800K code + 200K general SFT, 160K DPO, 300K RPO | 60K verified samples (from ~100K raw at 40% rejection) across 3,673 APIs | 14M (problem, solution) pairs, avg 23 solutions per augmented problem |

A few things this matrix makes obvious and a flat list of papers hides:

- **Self-Instruct leaves stage 4 essentially empty.** That is the one true weakness of the founding paper, and every successor (Alpaca -> Evol-Instruct -> Wizard* -> Nemotron) is an upgrade to a specific later stage. The lineage is not "new ideas"; it is "fill in the empty cells."
- **APIGen's entire reputation is stage 4.** Drop the three layers and you have a generic function-calling instruction dataset. Their ablation: removing the semantic layer costs 6 points of BFCL-V1, removing execution costs 11 points, removing format costs 18 points. **All three layers are load-bearing** — that is the paper's whole contribution.
- **Nemotron's innovation is stage 5.** Synthetic generation at scale is easy; picking chosen/rejected pairs *without* a human annotator is hard, and the reward-model-as-selector is what lets 20K human examples govern a ~1.4M-example preference corpus.
- **OpenMathInstruct-2's innovation is stage 1 at the problem level** (question augmentation into 600K problems) coupled with K=32 at the solution level. Stage 4 is aggressive but standard for math (SymPy is off-the-shelf).

If you remember one thing from this chapter: these four papers look different on the page and do the same thing on the loop.

---

## 4. Failure modes per stage

For every stage, there is a characteristic way it breaks. Recognise the symptom; locate the stage; fix it.

**(1) Generate — collapse to templates.** The teacher, especially when prompted with few-shot seeds, rapidly narrows to a handful of phrasings. Self-Instruct's diversity stats (Table 2 in the paper) show root-verb entropy dropping across iterations without the ROUGE filter. The symptom is downstream models that solve benchmarks where the training distribution overlaps the eval and collapse on out-of-template prompts. Fix: inject structural diversity upstream (Persona-Hub personas, topic-tag conditioning, seed rotation, category-seeded prompts a la Nemotron).

**(2) Filter — over-prune or under-prune.** Over-prune (too strict) kills useful long-tail signal (banning "graph" wiped anything math-diagram-related in Self-Instruct's 2022 run). Under-prune (too permissive) lets schema violations into SFT, and the model learns to produce syntactically broken outputs. Fix: compare pre- and post-filter distributions on a per-task basis; check that acceptance rate is stable across task families.

**(3) Dedup — lost diversity.** The failure mode is subtle: if your dedup metric is surface-level (ROUGE, MinHash), you can still admit 10,000 prompts that differ only in entity names but share the same underlying task structure. The model overfits to that structure. Fix: pair surface dedup with embedding-space or topic-space dedup (InsTag), and monitor task-family entropy.

**(4) Verify — no verifier means no signal.** This is the cardinal sin of synthetic data and the one Nathan Lambert keeps highlighting: "verification is the bottleneck." A pipeline without a verifier produces data of whatever quality the teacher happens to emit — no floor, no ceiling. APIGen's 40% rejection rate is not waste; it is the dataset. Fix: pick a verifier appropriate to the modality (gold answer, executor, RM, judge) before you start generating, not after. For modalities where there is *no* cheap verifier (open-ended writing, subjective tasks), you must use a calibrated judge and audit its bias (ch-26).

**(5) Select — biased mix.** If you keep everything that passes verification, you inherit whatever distribution the teacher happened to produce — usually biased toward easy, short, common-style answers. The symptom is a model that looks great on simple benchmarks and plateaus on hard ones. Fix: explicit selection on difficulty (IFD, LESS, Superfiltering — ch-25), on reward-model score (Nemotron), or on coverage (task-family quotas). The question "kept = passed-verification" is a silent default worth overriding.

**(6) Mix — ratio and order matter.** Two failure modes. First, the synthetic/real ratio: the 2025 scaling-laws literature ([[synthetic-data-scaling-laws]]) finds ~30% rephrased synthetic is optimal for pretraining; pure-generated synthetic at high fractions reproduces model-collapse signatures. Second, order: Nemotron's separation of code SFT and general SFT is not decorative — mixing them homogeneously degrades code performance. Fix: treat the mix ratio and ordering as hyperparameters, ablate them, and never assume "more synthetic = better."

### 4.1 The anchor-set principle

Across all four pipelines, a pattern repeats: a *small* set of high-quality human data calibrates a much larger synthetic corpus. Nemotron-4 states this baldly — "a small human anchor set (~20K) can support a much larger synthetic alignment corpus" of ~1.4M examples. Self-Instruct's anchor is 175 seed tasks. APIGen's anchor is the 3,673 executable API references (ground-truth implementations). OpenMathInstruct-2's anchor is MATH + GSM8K training problems (15K), pre-augmentation.

The anchor set is not just the seed for stage 1; it is also what *validates* stages 2–4. If your filter and verifier pass everything on the anchor and reject most raw synthetic, the pipeline is healthy. If they reject the anchor itself at high rates, your filters are miscalibrated. Maintain this invariant as a regression test. This also explains why "100% synthetic, no human data" pipelines do not exist in production — there is always an anchor, even if it is just a few thousand examples; the public slogan "98% synthetic" refers to the *bulk* of the training mixture, not the absence of a human anchor.

### 4.2 Compounding errors under iteration

When the loop is iterated — feeding cleaned synthetic back as seeds, or retraining the teacher on its own filtered output — errors in stages 2 and 4 compound. Nemotron's paper flags this: "reward-model errors compound when the same scorer is reused across iterations." Lambert's defence is **accumulation over replacement**: never fully replace the real data, always stack the new synthetic on top of the old anchor. This is the practical counterpart to the model-collapse theory covered in ch-14. For this chapter the point is that iteration is not free; the same loop is the site of the danger, and the defence lives at stages 5 and 6 (keep the real data in the mix, keep the anchor alive).

---

## 5. Why the loop ports across modalities

This is the payoff of the design-pattern lens. Treat the six stages as an interface, and every new paper becomes a swap of one or two slots:

- **Code generation (WizardCoder, OSS-Instruct, OPC):** stage 1 = "snippet -> instruction" or evol-instruct mutation; stage 4 = unit-test execution. Same shape as APIGen with a different executor.
- **Long context (LongAlign, ProLong):** stage 1 = document chunking + multi-doc fusion; stage 4 = needle-in-a-haystack check or position-conditioned retrieval. Same shape as OpenMathInstruct-2 with a retrieval-style verifier replacing SymPy.
- **Multi-turn agent trajectories (AgentInstruct, APIGen-MT):** stage 1 = rollout in a sandboxed environment; stage 4 = task-success check. Same shape as APIGen extended across turns.
- **Pretraining rephrase (WRAP, Cosmopedia):** stage 1 = rephrase a real document; stage 4 is unusually weak because ground truth is "the rephrase is coherent and faithful," not an answer. This is why stage 6 (mix ratio) is the dominant lever for pretraining synthesis (ch-22).

Porting is mechanical once you've seen it. Recognising the loop is what makes reading the next 20 synthetic-data papers fast instead of exhausting.

### 5.1 Raschka's stage-1 taxonomy, reread

Sebastian Raschka's practitioner overview catalogues stage 1 as four sub-types: **rewrite, backtranslate, bootstrap, full-generate**. The classification is useful because each sub-type has a different failure profile at stage 4:

- **Rewrite** (WRAP, Cosmopedia, paraphrase augmentation) preserves labels; stage 4 is a faithfulness check, not a correctness check. Cheap verifier; rarely catastrophic.
- **Backtranslate** (round-trip through another language or modality) is self-consistent by construction; stage 4 mostly validates that the round trip didn't lose information.
- **Bootstrap** (Self-Instruct family): stage 1 generates a fresh (input, output) pair from seeds. Stage 4 has no ground truth unless the task is verifiable; this is where most Self-Instruct-era pipelines are weakest.
- **Full-generate** (Phi-style textbook synthesis, Nemotron category-seeded prompts): stage 1 imagines a whole document or dialogue from scratch. Stage 4 has to lean on a judge or RM; also the highest hallucination risk.

Raschka's point: the stage-1 choice *predetermines* how much stage 4 has to do. Rewrite + backtranslate shift work off verification; bootstrap + full-generate shift work onto it. This is the same loop, viewed from the cost side.

---

## 6. "Verification is the bottleneck" — the operating principle of 2025 synthesis

Pull the four pipelines apart and one message is consistent. Self-Instruct's known weakness is its empty verify stage. Nemotron-4's entire differentiator is the reward model standing in the verify/select slots. APIGen's reputation rests on the three-layer verifier. OpenMathInstruct-2's gains over OMI-1 are roughly equal parts bigger teacher and tighter SymPy verification. Nathan Lambert, summarising the 2025 landscape, is categorical: the "scarce resource is verifiable prompts."

This principle has three operational corollaries worth internalising now, before the rest of the track:

1. **Verifiable tasks compound; unverifiable tasks do not.** Math, code, function-calling, schema-constrained tasks — all cheap to verify, and pipelines in these domains scale cleanly (OpenMathInstruct-2's 14M solutions, APIGen's per-layer ablation showing 3 of 3 layers load-bearing). Open-ended writing, summarisation, creative tasks have no cheap verifier; progress there is gated on judge quality, which is the ch-26 topic.

2. **Generation is a commodity; verification is the moat.** Anyone with a frontier API can run stage 1. What separates production-grade pipelines is the engineering at stage 4: the reference-API sandbox (APIGen), the reward model plus small human anchor (Nemotron), the symbolic equivalence checker (OMI-2), the executable unit tests (WizardCoder/OSS-Instruct). The lesson for someone joining a post-training team: if you want to be load-bearing, own stage 4.

3. **"Synthetic data can do almost all of the work"** — Lambert's Jun 2024 claim — **given** a strong base model and robust verification. Drop either precondition and the claim fails. The base-model side is the focus of ch-02..ch-08; the verification side is the focus of this track. This chapter's loop is how those two halves fit together.

The rest of the track is, in essence, 11 chapters of filling in the six cells with increasing sophistication. Keep the loop in mind.

---

## 7. What this track will not teach as doctrine

A note about how the rest of the synthetic track is structured. Individual papers are cases, not doctrines:

- ch-19 catalogues **generation methods** (bootstrap, evol, extraction, persona, rephrase) as interchangeable implementations of stage 1.
- ch-20 through ch-23 are **modality chapters** — instruction, reasoning-trace, tool-calling, long-context — each a case study in how stages 1 and 4 specialise.
- ch-24 through ch-26 are **cross-cutting machinery** — filtering, dedup, judges — i.e. stages 2, 3, and 4.
- ch-27 is **mixing and scaling laws** — stage 6, with the 2025 empirical results.
- ch-28 covers **iteration and self-improvement** — the loop-around-the-loop.
- ch-29 is the synthetic-track lab.

None of those chapters will re-teach the loop. Each will assume you carry it in your head and will discuss which cell(s) the paper innovates on.

### 7.1 Reading checklist for every synthetic-data paper

When you open the next paper in the synthetic track, walk through the loop once before reading the results:

- **Stage 1:** Who is the teacher? What are the seeds, and how many? What is the prompt shape?
- **Stage 2:** What surface filters exist? What is the acceptance rate?
- **Stage 3:** What dedup metric, at what threshold? Is it applied cross-corpus or within-item?
- **Stage 4:** What is the ground truth? If none, what is the judge and how is it calibrated?
- **Stage 5:** What selection criterion decides which survivors get trained on?
- **Stage 6:** How does this dataset compose with real data at training time? What is the mix ratio and curriculum order?

If you cannot answer a question, that cell is either empty (sometimes legitimate, more often a weakness) or described ambiguously in the paper (worth flagging). This checklist is all you need to take effective notes on a synthetic-data paper.

---

## Connections and what's next

- **[[ch-17]]** — dataset curation for pretraining; this chapter is the post-training mirror. Both lean on the same filter/dedup/verify triad, applied to different data sources.
- **[[ch-19]]** — generation methods. The immediate next step: stage 1 of the loop, in depth.
- **[[ch-20]]..[[ch-23]]** — modality chapters. Each reuses this loop as the spine.
- **[[ch-25]]** — filtering-for-quality (IFD, LESS, Superfiltering) — stages 2 and 5 in depth.
- **[[ch-26]]** — judges and judge bias — stage 4 when the modality has no cheap ground truth.
- **[[ch-27]]** — synthetic-data scaling laws and mix ratios — stage 6.
- **[[ch-44]]** — RLVR (verifiable rewards) reuses the same verify-stage technology at training time, not just at data time.

## Further reading

- [[self-instruct]] — the founding paper. Read for the canonical four-step pipeline and the ROUGE-L filter.
- [[nemotron-4-synthetic]] — industrial-scale alignment; read for reward-model-as-selector and staged SFT.
- [[apigen]] — read for the cleanest articulation of three-layer verification and its per-layer ablation.
- [[openmathinstruct-2]] — read for teacher-strength scaling and SymPy-based gold-answer verification.
- [[nathan-lambert-synthetic-data]] — the "verification is the bottleneck" framing; also the easy-vs-hard-verifiable split.
- [[sebastian-raschka-synthetic]] — the rewrite / backtranslate / bootstrap / full-generate taxonomy for stage 1.
- [[synthetic-data-scaling-laws]] — stage-6 empirics; rephrased vs pure-generated at scale.

## Companion visualization

**[figures/synth-loop.html](figures/synth-loop.html)** — interactive six-stage loop. Click any node to see its characteristic failure mode and a concrete example pulled from one of the four flagship pipelines above. Use it until the loop is reflex.
