<!-- chapter: ch-21
     track: synthetic
     title: Taxonomy-Driven and Textbook Synthesis
     sources: [[glan]], [[nemotron-4-synthetic]], [[phi-textbooks]], [[phi-1-5]], [[phi-3]], [[phi-4]], [[hf-cosmopedia]], [[mathscale]]
     figures: figures/taxonomy-expansion.html
-->

# Chapter 21 — Taxonomy-Driven and Textbook Synthesis

> **Core insight.** When the seed pool is the bottleneck — as it was for Self-Instruct's 175 tasks and every pipeline descended from it — you can replace the seed entirely with a **pre-curated taxonomy of human knowledge** and let the teacher expand it top-down. The taxonomy *is* the diversity prior. Whatever the curator carved into disciplines, subjects, and concepts is what the downstream model will learn; whatever the curator omitted is invisible to the student.
>
> **Guideline.** For a new capability, first decide: does a bottom-up (seed + bootstrap + filter) pipeline give you enough coverage, or are there branches of the space no seed ever samples? If the latter, build a taxonomy with ≥ 3 levels, expand each leaf into concrete units (syllabus sessions, topic pages, exercise types), and generate at the leaf. Budget the curator-audit cost explicitly: the top two levels of the tree dominate the bias of the final corpus.

---

## Why this chapter exists

Ch-18 gave you the generate → filter → dedup → verify → select → mix loop. Ch-19 surveyed *how* to generate (bootstrap, evol, rephrase, persona). Ch-20 covered distillation-as-data. All of those assume a seed — real prompts, real problems, real user logs — as the entry point.

This chapter is about what happens when there is no seed, or when the seed is too narrow. Three lines of work converged on the same answer between 2023 and 2025:

1. **Phi (Microsoft, 2023→)** — swap noisy web for a curated topic list, synthesize "textbook-quality" prose around each topic, use that as *pretraining* data. A 1.3B model rivals 10× larger ones on code and reasoning.
2. **GLAN (Microsoft, 2024)** — the same idea applied to instruction data: decompose human knowledge into a Field → Subfield → Discipline → Subject → Session → Concept tree, generate an instruction per leaf concept, no seed examples needed.
3. **Nemotron-4 340B (NVIDIA, 2024)** — synthesize >98% of the alignment corpus by running a generator/critic loop over task-family categories, using the model's own reward head as the judge.

The unifying abstraction is **top-down synthesis**: the structure comes from a human- (or GPT-4-) curated category system, and the teacher fills each cell. This chapter is that abstraction — what it buys you, what it costs, and where it silently misses.

The companion **[figures/taxonomy-expansion.html](figures/taxonomy-expansion.html)** is a live tree-expander: you start with 5 root disciplines, click "expand," and watch the fan-out multiply by the depth and branching-factor hyperparameters. It exists because the tree-size arithmetic is the single most important operational fact in this chapter and it's hard to internalize from prose alone.

---

## 1. Bottom-up vs top-down

Self-Instruct (ch-18) starts with 175 seed tasks, prompts the teacher for "more tasks like these," and filters with ROUGE-L. The resulting 52K instructions lie in the neighborhood of the seed pool — new instances, but small perturbations of a pattern a human already wrote down. Taxonomy-driven synthesis inverts this: no seed pool, you write down the *coordinate system* of the space and the teacher authors instances inside each cell.

| | Bottom-up (Self-Instruct, Evol-Instruct) | Top-down (GLAN, Phi-1.5, Cosmopedia) |
|---|---|---|
| Entry point | 175–1000 seed instances | 5–50 root categories |
| Diversity source | random walks through teacher's output distribution | structural walks across a curated tree |
| Coverage guarantee | statistical | constructive (every leaf generated once) |
| Blindspot | branches no seed touches | branches the curator never drew |
| Curator work | labeling seeds | building + auditing the tree |
| What scales | generation-side (more samples per seed) | tree-side (more branches, deeper) |

Neither is imagination-free — both inherit the curator's view of the world — but the locus of that inheritance moves. In a seed pool, bias is diffuse; in a taxonomy, it concentrates in the first two levels of the tree where a handful of labels determine everything underneath.

---

## 2. GLAN — taxonomy as the whole design document

[[glan]] ("Generalized Instruction Tuning via Taxonomy-Driven Synthesis", Microsoft 2024) is the clearest statement of the pattern. Their opening move:

> Prior instruction-tuning data synthesis (Self-Instruct, Evol-Instruct) is seeded by a small real instruction pool and inherits its biases. GLAN replaces the seed with a **taxonomy**: human knowledge is decomposed semi-automatically into fields → subfields → disciplines. Each discipline gets an auto-generated subject list; each subject gets an auto-generated syllabus of class sessions; each class session is enumerated as a concept list. Instructions are generated at the concept level, guaranteeing coverage across all branches.

The tree has six levels in practice:

```
Field      (e.g. Mathematics, Computer Science, Humanities, Natural Sciences)
  Subfield  (e.g. Algebra, Number Theory, Topology under Mathematics)
    Discipline  (e.g. Linear Algebra under Algebra)
      Subject   (e.g. Eigenvalues and Eigenvectors)
        Session (one class in the subject's syllabus, with learning objectives)
          Concept (a single pedagogical unit inside the session)
            Instruction  ← generation happens here
```

The top two levels are hand-curated by the authors plus GPT-4; every level below is auto-generated by prompting GPT-4 with the parent's name and "list the children." That means the *shape* of human knowledge (Math vs CS vs Humanities at level 0; Algebra vs Topology at level 1) is a human design choice, and the *density* inside each shape is GPT-4's.

Pseudocode for the full expansion:

```python
def glan_expand(root_fields, teacher, depth_budget):
    """Taxonomy-driven top-down synthesis (GLAN §3)."""
    # Level 0: Fields (hand-curated, ~36 entries)
    fields = root_fields

    # Levels 1-2: Subfields -> Disciplines (GPT-4 decomposes; authors audit)
    disciplines = []
    for f in fields:
        subfields = teacher.ask(f"List the major subfields of {f}.")
        for sf in subfields:
            disciplines += teacher.ask(f"List the disciplines within {sf}.")

    # Level 3: Subjects (GPT-4)
    subjects = []
    for d in disciplines:
        subjects += teacher.ask(f"List comprehensive subjects taught in {d}.")

    # Level 4: Syllabus sessions per subject
    sessions = []
    for s in subjects:
        sessions += teacher.ask(
            f"Write a multi-class syllabus for {s} with explicit learning objectives.")

    # Level 5: Concepts per session
    concepts = []
    for sess in sessions:
        concepts += teacher.ask(f"List the key concepts covered in {sess}.")

    # Level 6: Instructions (generation at the leaf)
    corpus = []
    for c in concepts:
        # Varied difficulty; verifier-friendly formats for math/code
        for difficulty in ["easy", "medium", "hard"]:
            instr, response = teacher.generate_instance(concept=c, difficulty=difficulty)
            corpus.append((instr, response))
    return corpus
```

The branching factors — how many disciplines per field, subjects per discipline, concepts per session — are hyperparameters. At GLAN's published settings the final corpus is "multi-million instruction-response pairs spanning all leaf concepts." The interactive companion lets you play with these numbers directly; watching the leaf count go from ~5K (shallow, narrow) to ~5M (deep, wide) is the point.

**What the taxonomy bought them.** Mistral-7B fine-tuned on GLAN data beats Alpaca, WizardLM, and CodeAlpaca training on the same base across MATH, GSM8K, HumanEval, MBPP, BBH, ARC, and MMLU — *without any task-specific real data*. Adding a new capability is as cheap as adding a subtree.

**What it did not buy them.** GLAN does not produce skills the teacher lacks. It extracts GPT-4's implicit curriculum into an explicit tree and hands it to Mistral-7B. The ceiling is the teacher, same as all distillation.

---

## 3. Nemotron-4 340B — generator/critic as a shallow taxonomy

[[nemotron-4-synthetic]] takes a different slice. Instead of a deep tree of *content*, NVIDIA curated a shallow tree of *task families* (coding, general QA, topic-following, document-based reasoning, function calling, refusal) and let the generator/critic loop do the heavy lifting. Two things are load-bearing.

**The RM is both filter and judge.** When ground truth is missing (open-ended QA, dialogue), Nemotron-4-340B-Reward picks which generator sample enters the SFT set and which pair enters the preference set. The *same lab* trains the generator, trains the RM on 20K human HelpSteer2 anchors, and uses that RM to govern the next iteration. Anchor-set math:

| Data slice | Scale | Human-labeled |
|---|---|---|
| SFT (code) | ~800K | No — RM-judged |
| SFT (general) | ~200K | No — RM-judged |
| DPO pairs | ~160K | No — RM-judged |
| RPO pairs | ~300K | No — RM-judged |
| HelpSteer2 (RM training) | ~10K | Yes |
| SFT human anchors | ~10K | Yes |
| **Total** | **~1.48M** | **~1.3% human** |

The paper states it directly: "over 98% of the training data for alignment is synthetic." The human slice trains the reward model; everything else is the RM's verdict on synthetic candidates.

**Generator/critic loop in pseudocode.**

```python
def nemotron_iteration(policy_ckpt, rm, task_families, K=4):
    sft_corpus, dpo_corpus = [], []
    for family in task_families:
        for p in synth_prompts(policy_ckpt, family):
            candidates = [policy_ckpt.generate(p) for _ in range(K)]
            scored = sorted(((c, rm.score(p, c)) for c in candidates), key=lambda x: -x[1])
            sft_corpus.append((p, scored[0][0]))                       # top as SFT target
            dpo_corpus.append((p, scored[0][0], scored[-1][0]))        # (best, worst) pair
    return sft_corpus, dpo_corpus
```

For code, NVIDIA replaces `synth_prompts` with **Genetic Instruct** — self-instruct + WizardCoder mutations + LLM fitness function seeded by a small code pool. It is the same tree shape but with genetic fan-out inside the family.

**Staged SFT** — code SFT first, general SFT second — shows up in Phi-4, Qwen, and Tülu too. Code data has cleaner correctness signal; training on it first installs structured-format following, which general-SFT on top doesn't damage.

---

## 4. The Phi line — textbook synthesis at pretraining scale

The Phi papers pushed the same idea one level earlier in the stack: not instruction data, but **pretraining data**. The thesis, stated in the title of the first paper: [[phi-textbooks]] — "Textbooks Are All You Need."

> Pretraining loss is *quality*-bounded long before it is *quantity*-bounded; swapping a noisy web crawl for a small "textbook-quality" corpus (filtered real + GPT-3.5-synthesized textbooks + exercises) yields a 1.3B model that rivals models ~10× bigger trained on ~100× more tokens.

Four generations, the same core move, progressively broader:

| Model | Year | Params | Pretrain tokens | Textbook/synthetic tokens | Web tokens | Code tokens | Notes |
|---|---|---|---|---|---|---|---|
| Phi-1 | 2023 | 1.3B | ~7B | ~1B synthetic (Python textbooks) + ~180M synthetic exercises (SFT) | ~6B filtered Stack / StackOverflow | (subset of the above) | First use of a "textbook-likeness" classifier on The Stack |
| Phi-1.5 | 2023 | 1.3B | ~27B | ~20B synthetic (20K-topic taxonomy, GPT-3.5) | optional 'phi-1.5-web' ablation | inherited from phi-1 | Extends recipe from code → common-sense |
| Phi-3-mini | 2024 | 3.8B | 3.3T | Phase-2 majority (exact split undisclosed, ~trillion-scale synthetic) | Phase-1 majority (filtered web) | mixed into both phases | Two-phase curriculum, GPT-4-class teachers |
| Phi-4 | 2024 | 14B | ~unspecified, synthetic weighted ~10% (≈400B unweighted) across 50 categories | ~400B unweighted | filtered web (rest of budget) | part of the 50 categories | 50 synthetic types; pivotal-token DPO |

Three facts anchor the table.

- **The classifier comes first.** Phi-1 trained a random-forest on a hand-labeled "educational vs non-educational" seed (features from a small LM's embeddings), scored The Stack, and kept the top slice as the 6B filtered-web core. Everything else is synthesis around whatever the classifier couldn't find.
- **Phi-1.5 formalized the taxonomy.** From [[phi-1-5]]: "a carefully curated list of 20,000 topics" covering common-sense, grade-school science, logic, everyday reasoning — each expanded by GPT-3.5. Flat, not hierarchical, but the first public "taxonomy-as-prior" for pretraining.
- **Phi-4 scaled categories, not depth.** Phi-4's ~400B unweighted synthetic tokens span "50 types" — a category index, not a hierarchy. Depth inside each category is delegated to GPT-4-class teachers plus rejection sampling. Phi-4-reasoning then distilled 1.4M o3-mini traces (~16B SFT tokens) on top and needed only **90 GRPO steps** to reach competitive AIME. That number is the sharpest summary of Phi philosophy: *all the gain is in the SFT data; RL is polish*.

**The TBAYN quote and its limitations.** "Textbooks Are All You Need" is marketing more than a technical claim. What Phi-1 actually showed: ~7B tokens of curated + synthesized code data trains a 1.3B model to 50.6% HumanEval / 55.5% MBPP; filtered + synthetic beats filtered-only beats raw web in ablation. Caveats:

- **Contamination:** later analyses of the Phi line flag non-trivial overlap between synthetic exercises and HumanEval prompts. The 50.6% is under suspicion.
- **Narrow domain:** Phi-1 is Python. Phi-1.5 extends to reasoning but is weakest on the harder benchmarks (MMLU-Pro, MATH) that became standard later.
- **Closed data:** no Phi corpus has been released. [[hf-cosmopedia]] is the open reproduction; its 1B validation model lags Phi-1.5 on several tasks.

Honest reading: curated + synthetic is enough to make a small model competitive *on benchmarks that resemble the training data*. The generalization claim is weaker; contamination is the biggest unresolved critique.

---

## 5. Where taxonomies come from — the hidden design choice

Every pipeline in §2–4 has a taxonomy; where does it come from?

| Pipeline | Taxonomy source | Levels | Curator |
|---|---|---|---|
| GLAN | Hand-curated root fields + GPT-4 auto-expansion below | 6 | Authors + GPT-4 |
| Phi-1.5 | Hand-curated list of "20,000 topics" (flat) | 1 | Microsoft team |
| Phi-3 | Not disclosed — "reviewed by researchers" | ? | Microsoft team |
| Phi-4 | "50 synthetic categories" list | 1 | Microsoft team |
| Nemotron-4 | ~6 task families (hand-picked) | 1 | NVIDIA team |
| Cosmopedia | Mixed: Stanford outlines + Khan + OpenStax + WikiHow + 145 web clusters + UltraChat/OpenHermes seeds | 2–3 | Hugging Face team |
| MathScale | Graph *extracted* from seed problems via GPT-3.5 concept mining | 2 (topic × knowledge-point) | Seed-derived, not pre-curated |

Five of seven are hand-curated at the top. That is the **curator-bias-injection point**. Whatever a Microsoft researcher considers a "field of knowledge" or a "synthetic category" becomes the top-level coordinate system of a multi-million-sample corpus. If the curator's personal frame omits, say, engineering ethics, or rhetoric, or specific applied CS subfields, the trained model will be fluent within the curator's frame and blind outside it.

This is unavoidable in taxonomy-driven methods *by construction*: you asked for a taxonomy precisely because you wanted the coverage to be constructive, and constructive coverage requires a construction. The only alternatives are:

- **Extract the taxonomy from data, not from a curator.** MathScale does this: prompt GPT-3.5 to name the concepts in each seed problem, aggregate into a graph, sample edges. The taxonomy inherits the seed's biases but avoids a curator's personal priors.
- **Crowd-curate.** Not used in any of the papers above; would address the single-curator bias but has its own demographic biases.
- **Audit post-hoc.** Cosmopedia's 145-cluster web-sourced branch is an audit move: it supplements the curated branch with "whatever the web talks about" to catch the curator's omissions. The HF team argues this catches real gaps. It also doubles the deduplication workload.

**Practical rule from [[hf-cosmopedia]]:** *"prompts must be rewritten carefully to avoid near-duplicate generations."* Cosmetic variation of the prompt — "explain X to a child" vs "explain X to an adult" — is not enough. Structural variation of the prompt family (audience × format × context, each from a real taxonomy) is. The Cosmopedia team found this the hard way: their first pass produced 30M prompts with too many duplicate-class outputs; the fix was restructuring the prompt taxonomy, not filtering harder post-hoc.

---

## 6. Coverage vs imagination — the bottom line

Here is the tension stated bluntly.

**Top-down gives you coverage.** Every leaf of the tree gets one (or k) samples. Every discipline the curator named appears in the corpus, proportionally to the tree's local branching factor. This is the guarantee a seed-based method cannot make.

**Bottom-up gives you imagination.** A seed pool contains combinations the curator never thought of — a user who asked "explain how eigendecomposition relates to principal component analysis in a fraud-detection context for a non-technical manager" is producing a leaf that crosses four branches of GLAN's tree. No tree expansion emits that combination naturally; the teacher would have to walk through the cross-product of levels to find it, and the cross-product is combinatorially too large to enumerate.

The companion HTML makes this visible: as you expand a GLAN-style tree, you see the leaf count multiply along the axes the curator drew, and you see the *uncovered* combinations as white space between branches. The white space is where seed-based methods accidentally find gold.

**The practical answer for 2025 pipelines.** Mix. Tülu 3 (ch-33), the Phi-4 mixture, Nemotron's RM-governed pipeline, and Cosmopedia all layer both:

1. A taxonomy-driven core (coverage guarantee, curator-audited).
2. A seed-based or user-log-derived supplement (imagination, captures combinations).
3. A dedup+verify layer (ch-18) to collapse the overlap and keep the best of each.

The chapter's three running examples — GLAN, Nemotron, Phi — are all the first layer. They are *necessary* for capability coverage. They are not sufficient for generalization to combinations the curator did not anticipate. Ch-22 (quality, diversity, gradient-based selection) and ch-23 (model collapse, verification) cover the layers that pull the second benefit into the stack.

---

## 7. Operational cheat-sheet

**When to build a taxonomy.**
- The target capability is broad (general instruction-following, general pretraining).
- Existing seed data is known-biased (e.g. all from ShareGPT → all English, all casual).
- You need auditable coverage claims ("all disciplines of mathematics are represented").
- You plan to add new capabilities incrementally (adding a subtree is cheap).

**When not to.**
- The target capability is narrow and a good seed pool exists (code completion, SQL generation).
- The teacher is too weak to produce leaves reliably at depth.
- The audit budget for the top two levels is < 1 week of an expert's time — bias will be severe.

**Scaling rules of thumb.**
- GLAN shape: 30–50 root fields × ~10 subfields × ~10 disciplines × ~20 subjects × ~10 sessions × ~10 concepts × ~10 instructions ≈ 3M–30M samples. Wider shallower trees undercover depth; deeper narrower trees undercover breadth. The interactive companion lets you play with these.
- Phi-1.5 shape: 20K flat topics × 5–10 passes each ≈ 100K–200K documents × ~100K tokens/document ≈ 10B–20B tokens. The flat topic list is simpler than a tree and is adequate when the teacher (GPT-3.5 in 2023) is already a strong curriculum-by-default writer.
- Nemotron shape: 6 families × ~million-scale generations each × RM-filter ≈ 1M-scale SFT + 500K-scale preference. The RM is the cost driver.
- Cosmopedia shape: 30M prompts × ~1K tokens output ≈ 25B tokens. Deduplication is the real bottleneck; plan for it upfront.

**Red flags.**
- Your trained model performs great on the first few benchmarks you tried, then mysteriously plateaus on one orthogonal benchmark. → The orthogonal benchmark probes a tree branch you didn't draw. Audit the taxonomy, don't tune hyperparameters.
- Your corpus deduplicates by > 40%. → Cosmetic variation, not structural. Rewrite prompt-family structure.
- Your teacher is the same model family as your student. → You are running recursive distillation. Risks in ch-23.
- Your taxonomy was written by one person. → The corpus is that one person's mental model at 10M-sample scale. Have two more curators audit levels 0 and 1 at minimum.

---

## Connections and what's next

- **[[ch-18]]** — taxonomy-driven is one operating mode of the generate → filter → dedup → verify loop; this chapter specialized the generator step. RM-as-judge (Nemotron) is the filter+verify specialization.
- **[[ch-19]]** — bootstrap / evol / rephrase / persona are the bottom-up alternatives. Persona-hub is the closest cousin: a taxonomy *of askers* rather than of topics.
- **[[ch-20]]** — R1-Distill lineage is distillation-of-traces; Phi-4-reasoning is the textbook-synthesis-plus-distillation hybrid.
- **[[ch-22]]** — quality/diversity/gradient-based selection is how you combine taxonomy-core + seed-supplement without blowing up corpus size.
- **[[ch-23]]** — model collapse and recursive-training risk; the Phi line's defense (single-shot teacher, no recursive self-distillation) is one answer.
- **[[ch-32]] / [[ch-34]]** — Phi-3/4 case study in the SFT recipes section; Nemotron-Ultra as another case.

## Further reading

- [[glan]] — full extract of the six-level taxonomy and branching-factor table.
- [[phi-textbooks]] — Phi-1 paper, the textbook-quality classifier, TBAYN origin.
- [[phi-1-5]] — the 20K-topic list as a taxonomy precursor.
- [[nemotron-4-synthetic]] — generator/critic details + Genetic Instruct.
- [[hf-cosmopedia]] — the open replication; read it for the dedup-is-the-bottleneck lesson.
- [[mathscale]] — domain-specific seeded taxonomy as a contrasting approach.

## Companion visualization

**[figures/taxonomy-expansion.html](figures/taxonomy-expansion.html)** — interactive tree expander. Start with 5 root disciplines, click "expand" to fan out three levels (subjects → lessons → questions), and move the depth and branching-factor sliders to watch the leaf count grow. The "white space" visualization overlays combinations the tree did not cover — the bottom-up imagination-vs-coverage tradeoff made concrete.
