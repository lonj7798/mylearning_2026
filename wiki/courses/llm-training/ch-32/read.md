<!-- chapter: ch-32
     track: sft
     title: Multi-Stage Pipelines — Mid-Training, Cold-Start, Long-Context
     deps: [ch-31]
     sources: [[interplay-pretraining-midtraining-rl]], [[olmo-3]], [[olmo-2]],
              [[deepseek-r1]], [[long-context-llama3]], [[prolong]], [[longalign]],
              [[llama-3]], [[qwen-3-5]], [[front-loading-reasoning]]
     figures: figures/pipeline-stages.html
-->

# Chapter 32 — Multi-Stage Pipelines: Mid-Training, Cold-Start, Long-Context

> **Core insight.** By 2025 "SFT" stopped being a single stage and became one checkpoint inside a **five-stage pipeline**: *pretrain → mid-training → long-context extension → SFT → RL*, each with its own data mix, its own token budget, its own eval gate, and its own failure mode. The conceptual shift is largest at mid-training: what used to be an informal "cooldown" or "annealing" step is now a **distinct distribution-shift stage** that installs reusable priors RL cannot replace. [[interplay-pretraining-midtraining-rl]] gives the controlled evidence: under fixed compute, mid-training beats RL-only post-training, and RL only expands capability when prior stages have left headroom at the model's *edge of competence*.
>
> **Guideline.** Treat each stage as owning a specific deliverable. Pretrain owns **broad priors** (diversity over quality per [[front-loading-reasoning]]). Mid-training owns **reusable structure** (math, code, long-doc comprehension) via small high-quality mixes (OLMo 3's Dolmino = **100B tokens** from a 2.2T pool). Long-context extension owns **position encoding + coherent long documents** ([[long-context-llama3]]: 10K → 500K RoPE base, ~800B tokens across six sub-stages). SFT owns **format and cold-start** for the target behaviour, including R1's **~800K trace** cold-start that fixes R1-Zero's language-mixing. RL owns **edge-of-competence exploration**. Gate each stage on a targeted eval suite — never a single aggregate benchmark.

---

## 1. Why "mid-training" is now a real stage

Through 2023 the prevailing picture had two stages: pretrain on the web-scale mix, then post-train with SFT + RLHF. Mid-training was folded invisibly into either the pretrain cooldown or the SFT warmup. Three things forced a split.

1. **Cooldowns stopped looking like pretraining.** OLMo 2's Stage 2 already ran 50B tokens on a curated "Dolmino" mix after 3.9T tokens of OLMo-Mix-1124 pretraining ([[olmo-2]]); the data distribution was visibly different (math, code, and high-quality web up-weighted, not sampled from the pretrain mix). OLMo 3 formalizes this as a **named stage** with its own dataset (Dolma 3 Dolmino, **~100B tokens from a ~2.2T high-quality pool**) and its own compute allocation (**128 H100s** versus 1024 for pretrain and 256 for post-train; [[olmo-3]]).
2. **Reasoning data stopped being SFT-sized.** Open-Thoughts-style rejection-sampled long CoT traces run to hundreds of thousands of sequences of 8K-32K tokens each — a multi-billion-token data mass that is *too large and too specific* for SFT and *too narrow* for pretraining. OLMo 3-Think, Qwen 3, and Phi-4-reasoning all funnel this through a dedicated mid-training pass.
3. **Controlled ablations showed the stage does something RL cannot redo.** [[interplay-pretraining-midtraining-rl]] instruments a synthetic-reasoning testbed where extrapolative (harder compositions) and contextual (new surface forms) generalization can be measured independently. Under matched compute budgets, **mid-training on reusable priors outperforms using the same budget for RL-only post-training**. RL still helps — but only where pretraining has left headroom and RL prompts sit at the *edge of competence*. Mid-training is the stage that engineers that headroom.

The fourth, less-publicized reason is **process supervision**. [[interplay-pretraining-midtraining-rl]] shows that process-level rewards during RL cut reward-hacking and improve reasoning *fidelity* (structural correctness, not just final-answer accuracy). Process rewards only work when the intermediate steps the model produces are already parsable and semantically grounded — a property that mid-training on structured data (math problems with step-annotated solutions, code with test traces) installs more reliably than either raw-web pretrain or instruction-formatted SFT.

The 2025-era pipeline, with attested token scales from OLMo 3, Llama 3, and DeepSeek-R1 where disclosed:

| Stage | What it owns | Representative data mix | Budget (OLMo 3 7B) | Eval gate |
|---|---|---|---|---|
| Pretrain | Broad priors, vocabulary coverage | Dolma 3 Mix: web + code + math + science, **5.9T tokens** | 1024 H100s | Broad LM perplexity; MMLU; code HumanEval baseline |
| Mid-training | Reusable structure on harder distributions | Dolmino: math/code/science/reading/IF, **100B from 2.2T pool** | 128 H100s | Targeted: GSM8K, MATH, HumanEval, MMLU-Pro, IFEval |
| Long-context | Position encoding + long-doc comprehension | Longmino: **50B** from 639B long-doc pool | 128 H100s (shared) | NIAH heatmap, RULER, BABILong |
| SFT (cold-start) | Format, target behaviour, readability | Dolci-SFT: rejection-sampled CoT + instructions | 256 H100s | Format compliance, MT-Bench, domain-specific probes |
| RL (DPO + RLVR) | Edge-of-competence exploration, verifiable rewards | Dolci-DPO + Dolci-RLVR | 256 H100s (shared) | AIME / Codeforces / RULER / IFEval / reward-hacking audit |

Think of it as **four distribution shifts**: web → curated-hard → long-coherent → instructive-format → edge-of-competence prompts. Each shift is narrow enough that one stage cannot absorb another's job, and each has its own eval that can fail independently.

---

## 2. Mid-training, defined operationally

**Mid-training** is a continued-pretraining-style training pass that (i) starts from the pretraining checkpoint, (ii) runs on a high-quality narrow mix (not the pretraining mix and not instruction-formatted SFT data), and (iii) uses **1–3% of pretraining tokens** as its budget. It is not SFT — loss is applied on every token, not just response tokens, and data is not instruction-response formatted. It is not pretraining — the mix is deliberately skewed toward what the downstream stages cannot teach.

Empirically OLMo 3's Dolmino covers **math problems + solutions, science papers, code, instruction-following seed data, and reading-comprehension passages** at **100B tokens ≈ 1.7%** of the 5.9T pretrain. OLMo 2's Dolmino cooldown was **50B tokens ≈ 1.3%** of 3.9T. Llama 3's "annealing" in the herd paper, though less documented, lives at the same point in the pipeline and similar proportional budget. The pattern is stable across labs: a *small* run on a *curated hard* mix before any post-training happens.

**Why this beats the obvious alternative — "just put the hard data in pretrain."** [[front-loading-reasoning]] actually argues *for* front-loading, but with a sharpening: during pretrain you should prioritize **diversity**, during mid-training + SFT you should prioritize **quality**. If you dump the Dolmino mix into a 5.9T pretraining run you lose signal because (a) the model hasn't yet built the broad priors that let hard examples generalize, and (b) the rest of the pretrain mix will average-out the signal from 100B high-quality tokens. Running it separately, *after* the base has stabilized, keeps the signal concentrated — [[interplay-pretraining-midtraining-rl]] reports this installs "reusable structure" that later RL can exploit.

**Why this beats "just make SFT bigger."** SFT data is instruction-formatted. You cannot convert a 100M-token corpus of real scientific papers into SFT without rewriting them into QA format — and that rewrite destroys exactly the cross-paragraph coherence the long-form data is supposed to teach. Mid-training trains on the papers as-is, with next-token loss, and lets the model learn long-range structure the instruction-response frame cannot carry.

**Three concrete signals you picked the wrong budget.**

- *Budget too small* (<0.5% of pretrain): the base's broad priors overwrite the mid-training distribution within a few epochs of SFT. You will see Dolmino-like gains at the mid-training checkpoint, then watch them decay through SFT/DPO.
- *Budget too large* (>5% of pretrain): you are re-running pretraining on a narrow mix, which is exactly the narrow-domain overfit Phi-4 gets criticized for. MMLU holds but breadth on out-of-mix tasks starts to slip.
- *Mix too instruction-heavy*: the model picks up instruction artifacts (assistant-style openings, bulleted formatting) before SFT has a chance to set its own format. Downstream DPO then has to *unlearn* the implicit format, which wastes preferences.

The practical diagnostic is a targeted eval suite run at **both** the pretrain endpoint and the mid-training endpoint. The gain should be concentrated on the hard targeted tasks (GSM8K, MATH, HumanEval, MMLU-Pro) with *near-zero change* on broad breadth (TriviaQA, HellaSwag). If breadth moves, your mix is drifting.

---

## 3. Cold-start SFT: R1-Zero vs R1

DeepSeek-R1's most-quoted result is R1-Zero — pure RL from DeepSeek-V3-Base, rule-based reward only, no SFT traces ([[deepseek-r1]]). The less-quoted result is **why R1 exists at all**. R1-Zero's rollouts degraded in two systematic ways: responses became an interleaved mix of English, Chinese, and math symbols; and the `<think>` block drifted into unreadable dense notation that made the traces useless for distillation. These failure modes are attested in the model report.

R1 fixes both with a **cold-start SFT pass before RL**:

- **~800K cold-start examples**, split roughly as **~600K reasoning + ~200K non-reasoning**.
- **Origin of the reasoning traces**: rejection-sampled from R1-Zero itself (and earlier internal RL checkpoints), filtered through a DeepSeek-V3 judge on correctness, and then **passed through a human-readability reformat** to eliminate language mixing and enforce the `<think>...</think><answer>...</answer>` template cleanly.
- **Filter criteria**: final answer must be correct, trace must be monolingual (the operative fix for R1-Zero's language-mixing), trace length ≤ 32K tokens, no hollow `<think>` blocks gaming the format reward.
- **Training config**: standard SFT on the V3 base with cross-entropy on response tokens; the exact SFT hyperparameters are not separately broken out from the following RL run in the report.

Then R1 runs the full four-stage post-training pipeline: **cold-start SFT → reasoning RL (GRPO, LR 3e-6, KL 0.001, eps 10, T 1.0, G 16, 32K generation, 512 samples/step) → rejection-sampling SFT → alignment RL**. Cold-start SFT is the *readability* intervention; reasoning RL is the *capability* intervention; rejection-sampling SFT is the *distillation* intervention that folds RL discoveries back into supervised data; alignment RL is the *general-helpfulness* intervention.

The comparison learners should carry is this: **R1-Zero demonstrates that RL alone can install long-CoT capability on a strong base; R1 demonstrates that you still want a cold-start to ship.** Cold-start is not a capability bootloader — it is a **format installer** that makes the resulting model deployable. Qwen 3's four-stage pipeline (long-CoT cold-start → reasoning RL → thinking-mode fusion → general RL) uses the same logic ([[qwen-3-5]]); Qwen 3.5 inherits it without algorithmic change.

**Cold-start sizing rule of thumb.** The attested point is DeepSeek's 800K. Smaller cold-start (Sky-T1, s1, LIMO) all sit at **1K-30K traces**; larger pipelines (OLMo 3-Think's Dolci-SFT, Llama-3-style iterated SFT) run into the millions. The sizing lever is **how far the base is from the target behaviour**. R1 used ~800K because R1-Zero's rollouts had drifted far from a readable CoT template — the cold-start had to reteach format before RL could continue on a clean distribution. If your base already speaks your target format (e.g., extending an instruct-tuned checkpoint into thinking mode), cold-start can shrink to the LIMO-style 817-example regime. If you are building the format from scratch, budget for 10²-10⁴ hand-vetted traces and up to 10⁶ rejection-sampled + judge-filtered traces before you pull the RL trigger.

---

## 4. Long-context mid-training: position encoding + data curation

Long-context is the stage most frequently described as "mid-training" in papers that predate the OLMo-3 terminology. It has two separable jobs — and lumping them causes most of the disappointing 128K-window releases.

**Job 1: position-encoding extension.** RoPE at its pretraining base frequency cannot represent positions beyond its training range without aliasing. [[long-context-llama3]] rescales RoPE base from **10,000 → 500,000** in a six-stage schedule (8K → 16K → 32K → 64K → 128K, ~100-200B tokens per stage, ~800B total). [[prolong]] pushes further with **10K → 128M NTK-aware** for 512K on Llama-3-8B. The scheduling rule is empirical: **each stage doubles context and adjusts the base proportionally**, running long enough for the new positions to stabilize before the next expansion.

**Job 2: long-coherent-document training.** Position encoding alone gives you the *capacity* to attend over long ranges; it does not teach the model to *use* that capacity. [[prolong]]'s thesis is that training on **concatenated short documents** (the naive shortcut — pack random short pieces up to the context window) is measurably worse than training on **genuinely coherent long documents**. Ablation: swapping curated coherent docs for concatenated short docs at matched token count costs **10+ points on HELMET**. ProLong's 30B-token mix is **~40% code repositories (full repo, README → source → tests concatenated), 25% books, 15% academic with references, 10% long forum threads, 10% misc long web**. Web is *downweighted* (×0.5), not upweighted, because long web documents are mostly scraped listings with weak long-range structure.

**Job 3 (SFT-side): long instruction alignment.** Once the base supports long context, SFT needs a long-context sub-mix — but kept *small*. [[longalign]] synthesizes 10k long-instruction examples from nine document sources with a **pick-one-of-5 cross-span trick**: the teacher generates 5 candidate questions covering the whole document, then one is randomly picked for answer synthesis. This forces cross-span coverage instead of local retrieval. [[long-context-llama3]] keeps long-context SFT at **~0.1% of total SFT samples** — raising it above 1% costs ~1 MMLU point of short-context capability. The binding constraint is short-context regression, not long-context gain.

Three-job decomposition fits into the stage table cleanly: Job 1 + Job 2 live in the long-context mid-training stage (with position rescale at stage boundaries); Job 3 lives in SFT. Mix them up and you get either a model that has the window but can't reason across it (Job 2 skipped) or a chat model that degrades on short inputs (Job 3 fraction too high).

**Llama 3's six-stage schedule, explicit numbers.** [[long-context-llama3]] documents the following per-stage profile (paraphrased from §3.4):

| Stage | Context | Tokens | RoPE base | Short:long mix |
|---|---|---|---|---|
| A | 8K → 16K | ~100B | partial rescale | 80:20 |
| B | 16K → 32K | ~100B | partial rescale | 70:30 |
| C | 32K → 64K | ~150B | partial rescale | 60:40 |
| D | 64K → 128K | ~200B | 500K (final) | 40:60 |

Additional intermediate stabilization stages bring the total to ~800B tokens. The schedule design rule is empirical: each sub-stage must run long enough for the RoPE rescale to stabilize before the next doubling, and the short:long ratio shifts gradually so the earlier-stage short-context behaviour stays anchored. Llama 3's effective RULER context is **~96K for 405B, ~64K for 70B** even though the claimed window is 128K — a gap that is *expected* and that the staged schedule keeps bounded rather than eliminates.

**ProLong's 20B budget is not universal.** [[prolong]] extends Llama-3-8B from 8K to 512K on only **20B CPT tokens + 5B SFT tokens**. Two reasons it works at that budget: the base is already Llama-3 (500K RoPE base), so the extension is relatively modest; and the 30B-token data mix is heavily filtered for coherence (code repos, books, academic with references). Swapping Llama-3 for a weaker base, or swapping the coherence filter for concatenated short docs, breaks the budget. The lesson is that **data quality buys token budget** — but base quality buys the data budget that quality-filtering requires.

---

## 5. OLMo 3's model-flow worldview — full pipeline in words

[[olmo-3]] is the 2025 reference release because it treats the **entire trajectory** (not just the final weights) as the scientific artifact. Every intermediate checkpoint, every per-stage dataset, every eval suite is public. Trace the 7B Base → Think flow:

**Base flow.** Pretrain on **Dolma 3 Mix (5.9T tokens)** — web, science PDFs via olmOCR, code, math problems/solutions, encyclopedic text; 1024 H100s. → Mid-train on **Dolmino (100B tokens)** sampled from a 2.2T high-quality pool emphasizing math, science, code, instruction following, reading comprehension; 128 H100s. → Long-context extend on **Longmino (50B tokens)** from a 639B long-doc pool; shared 128 H100s. Eval gate at each stage: general LM eval suite + staged targeted probes (GSM8K + MATH at mid-train, RULER + NIAH at long-context). Output: OLMo 3-Base 7B with 128K context.

**Think flow.** From Base, run **Dolci-SFT** (rejection-sampled long-CoT traces + instruction-following seeds, "thinking-specific"), then **Dolci-DPO** (thinking-specific preferences), then **RLVR** on verifiable math/code/IF rewards. Each stage uses **256 H100s** (shared across branches). Post-training is where the three branches diverge: Think uses thinking-specific SFT/DPO; Instruct uses chat-focused SFT/DPO; **RL-Zero uses no SFT** and runs RL directly from Base (the OLMo-3 analog of R1-Zero), as an explicit research artifact for studying pure-RL from an open base.

**Data curriculum summary, as a single diagram in words.** Dolma 3 (9.3T raw source tokens) → Dolma 3 Mix (5.9T filtered pretrain) → Dolmino (100B mid-train, sampled from 2.2T curated) → Longmino (50B long-context, sampled from 639B long-doc pool) → Dolci (three sub-mixes for SFT / DPO / RLVR, undisclosed totals). Each data artifact is a *subset* of the preceding pool by explicit filter rules: Dolma 3 Mix decontaminates and re-mixes Dolma 3; Dolmino up-samples math/code/science within a 2.2T high-quality pool; Longmino keeps only the documents long enough to train the extension; Dolci is instruction-formatted on top of Dolmino-adjacent seeds.

The **engineering point** is that OLMo 3's openness includes the transition rules, not just the datasets. You can see that Longmino draws from Dolmino's seed pool plus additional long-doc sources, which is *why* mid-training and long-context extension can share a 128-H100 budget: the later stage is partly a specialization of the earlier one, not a fresh start. Closed models almost certainly do the same; OLMo 3 is simply the place you can verify it.

**Per-stage eval gates.** OLMo 3's other disclosure is *how they decide a stage is done*. Each stage has its own gate:

- **Pretrain → Mid-train.** Gate on broad LM suite: OLMES broad, MMLU, HellaSwag, code HumanEval. Stop when the run is smoothly converging (no loss spikes) and MMLU is near the recipe-anchored expectation for the token budget spent.
- **Mid-train → Long-context.** Gate on targeted hard tasks: GSM8K, MATH, MMLU-Pro, HumanEval+, IFEval. The gain on these should be sharp relative to baseline pretrain endpoint; broad-eval should be flat.
- **Long-context → SFT.** Gate on long-context probes: NIAH heatmap @ 128K, RULER effective context, BABILong reasoning-in-a-haystack. Critically, *re-run short-context evals* to confirm no regression > 1 point on MMLU and HumanEval. If short-context regressed, the long-context data mix (Job 2) is too dominant.
- **SFT → DPO.** Gate on format + chat quality: MT-Bench, AlpacaEval-2, IFEval strict, format-compliance probe.
- **DPO → RLVR.** Gate on preference alignment + verifiable task head-room: AIME pre-RL baseline, MATH, Codeforces Elo baseline, reward-model-scored held-out.

Each gate has a corresponding **failure signature** in the training logs, not just a benchmark number. Loss spikes in mid-training with stable MMLU usually mean the mix upweighted a pathological subdomain (e.g., a Unicode-artifact-heavy math source). A RULER collapse in long-context with stable NIAH means Job 2 failed while Job 1 succeeded — the window is there, the reasoning isn't. An MT-Bench drop across DPO usually means on-policy preferences were re-used stale from a different SFT checkpoint ([[llama-3]] calls this out explicitly).

**Compute allocation as a check on your pipeline.** OLMo 3's disclosed split — **1024 H100s for pretrain, 128 for mid-train + long-context, 256 for SFT/DPO/RLVR** — is a useful sanity ratio. Pretrain is the dominant cost; mid-training is small enough that it would be missed in a rough accounting; post-training is a middle-sized specialty stage where efficiency gains (OLMo 3 reports **8× SFT throughput** from moving Open Instruct to Olmo Core, and **4× RL efficiency** from in-flight weight updates and continuous batching) actually matter. If your post-training compute approaches pretraining compute, you are either doing multi-round iterative SFT (Llama 3 style) or your stages are mis-allocated.

---

## 6. Open question — can better data skip mid-training?

A recurring claim in 2024-2025 debate: *if pretraining data is good enough, mid-training and cold-start become unnecessary.* The strongest evidence for this position is indirect — Phi-4's synthetic-heavy pretrain blurs the pretrain/mid-train boundary, and R1-Zero shows pure RL on a strong base can reach reasonable reasoning numbers without SFT. The strongest evidence against it is direct and quantitative:

- **[[interplay-pretraining-midtraining-rl]]** — under *fixed compute*, mid-training beats RL-only post-training on both extrapolative and contextual generalization. The effect is strongest at edge-of-competence tasks. If better pretraining alone could absorb mid-training's job, the fixed-compute comparison would favor RL-only; it does not.
- **[[front-loading-reasoning]]** — front-loading reasoning data into pretraining gives a reported **19% average gain** and *raises the ceiling* reachable by later SFT. But the paper explicitly notes that late SFT **cannot fully reconstruct** the durable advantage of early injection. That is, better pretraining makes mid-training *more* valuable, not less — it raises the headroom that mid-training + RL can exploit.
- **R1 vs R1-Zero deployment evidence** — pure-RL reaches strong AIME / MATH numbers, but the model ships language-mixed traces. The cold-start isn't a capability shortcut; it's a format fix that the pretrain-only configuration demonstrably does not learn.

The honest answer: *better data shifts where each stage's contribution peaks but does not eliminate any stage*. A smaller, higher-quality pretrain + a smaller mid-training + a smaller RL run is plausible — and is arguably what Phi-4 is experimenting with — but it is a rebalancing, not a merger. The stages remain conceptually distinct because their **jobs are distinct**: broad priors, reusable structure, position encoding, format cold-start, edge exploration. Conflating them gives you either a model that overfits on synthetic pretrain (Phi-4's narrow-domain criticism) or a model that can reason but cannot be read (R1-Zero).

**A specific counter-evidence pattern to watch.** When a lab claims "we skipped mid-training and it still works," check three things. (i) Did they actually skip it, or did they fold it into the pretrain cooldown under a different name? OLMo 2's Stage 2 was called a "cooldown" before OLMo 3 formalized "mid-training"; the content is the same. (ii) What is their benchmark mix? A claim grounded in MMLU + HellaSwag can hide mid-training's effect entirely because those benchmarks reflect broad priors the pretraining already has. The discriminating evals are MATH, AIME, MMLU-Pro, HumanEval+, IFEval, RULER — the harder or more targeted the task, the larger mid-training's visible contribution. (iii) How does their pretraining mix compare? Labs that front-load reasoning data into pretraining (per [[front-loading-reasoning]]) *can* reduce mid-training's apparent delta because some of mid-training's job has moved earlier. That is rebalancing, not elimination.

**Stage-dependency summary.** The clearest way to remember the interaction:

- Mid-training is *multiplicative* with pretraining quality. [[front-loading-reasoning]] shows the effect: better pretrain priors → mid-training gains are amplified, not dampened.
- RL is *multiplicative* with mid-training quality. [[interplay-pretraining-midtraining-rl]] shows this directly: RL at the edge of competence needs pretrain + mid-train to have put the edge somewhere interesting.
- Cold-start SFT is *additive* with the model's existing format behaviour. It is the cheapest stage to re-run and the one most sensitive to the target deployment surface.
- Long-context extension is *orthogonal* to the reasoning pipeline. You can insert it before or after mid-training without the reasoning stages caring much — but it has to happen before SFT, otherwise long-SFT data can't actually be long.

Play with the stage budgets in **[figures/pipeline-stages.html](figures/pipeline-stages.html)** — a five-stage horizontal Gantt where you can toggle between Llama-3, OLMo-3, DeepSeek-R1, and Phi-4-reasoning configurations and see token-counts plus data-mix composition per stage.

---

## 7. Recipe by compute budget

Not every lab has 1024 H100s. The stage structure collapses predictably when budget shrinks:

- **Frontier budget (10²-10³ H100 nodes).** Full five-stage pipeline. Mid-training and long-context each get their own data mix and eval gate. This is OLMo 3, Llama 3, DeepSeek-R1, Qwen 3.5.
- **Mid-tier budget (10¹-10² H100s, starting from an open base).** Skip pretrain entirely — start from Llama-3 / Qwen-3 / OLMo-3. **Keep mid-training** if your target is a specialized reasoning / coding / long-doc model; **skip mid-training** if your target is a chat assistant and your base is already instruction-tuned. Keep cold-start SFT if you are pushing into a new format (thinking mode, new tool schema). Keep long-context extension if the base doesn't already cover your target window. RL is optional at this budget unless you have verifiable rewards.
- **Small-budget (≤8 H100s, single-node).** Two stages: SFT + (optional DPO). Use a base that already has the mid-training and long-context properties you need. Your lever is data quality, not stage count — per [[front-loading-reasoning]], prioritize *quality* here; diversity lives in the base you picked.

The stage count decision is about **where your data quality sits relative to what the base already has**. If your data is higher quality than the base's pretrain mix but doesn't move the needle on mid-training targets, it belongs in SFT. If it does move the needle and is not instruction-formatted, it belongs in a mid-training pass — even if that pass is only a few hundred million tokens on eight GPUs.

---

## 8. Three labs, three stage allocations

Put the three best-documented 2025 pipelines next to each other and the rebalancing becomes visible.

| Stage | OLMo 3 7B ([[olmo-3]]) | Llama 3 405B ([[llama-3]], [[long-context-llama3]]) | DeepSeek-R1 ([[deepseek-r1]]) |
|---|---|---|---|
| Pretrain | Dolma 3 Mix **5.9T** | **15.6T**, 8K native | V3 inherited: **14.8T**, 2.788M H800-hrs |
| Mid-training | Dolmino **100B** (1.7%) | "Annealing" stage, scale undisclosed | Inherited from V3; no separately named stage |
| Long-context | Longmino **50B** (0.8%) | Staged 8K→128K, **~800B** tokens (5.1%), RoPE 500K | Inherited from V3 |
| Cold-start SFT | Dolci-SFT (rejection-sampled CoT + IF) | 6-round SFT + Rejection Sampling | **~800K traces** (reasoning + non-reasoning) |
| DPO / RL | Dolci-DPO → RLVR | 6-round DPO with NLL stabilization | Reasoning-RL (GRPO, 32K gen) → rejection SFT → alignment-RL |

Three observations.

1. **OLMo 3 spends proportionally less on pretrain (5.9T) and more on naming its post-pretrain stages explicitly.** The mid-training + long-context stages together are only ~2.5% of pretrain tokens, but each is a distinct run with its own eval gate. This is the clearest example of "stage-as-artifact."
2. **Llama 3 spends proportionally much more on long-context (~5% of pretrain).** That reflects Meta's design target — a 128K-window frontier model shipped on a dense 405B — and it is why Llama 3's effective RULER context (~96K at 405B) is the best among dense open models. Long-context was not a specialized mid-training stage for Llama 3; it was a major pretrain-adjacent effort.
3. **DeepSeek-R1 inherits pretrain and mid-training from V3 and invests almost everything distinct into the RL loop.** R1's innovations are post-V3: cold-start SFT, GRPO hyperparameters, the multi-stage SFT-RL-SFT-RL pipeline. The V3 base carries the mid-training and long-context stages — which is why R1 works *at all* from ~800K SFT samples. You cannot replicate R1 from a weaker base, because the stages R1 skips are already paid for in V3.

The right reading of the table is not "who spent more" but "which stage did each lab optimize locally?". OLMo 3 optimized the stage-structure itself (openness of the flow). Llama 3 optimized long-context + iterative post-training. DeepSeek-R1 optimized RL-from-a-strong-base. A smaller lab copies whichever stage matches its own edge — not the whole pipeline, which would require frontier budget.

---

---

## 9. What the field has not yet settled

Three open questions remain — each worth tracking through 2026.

- **Does mid-training compose with more aggressive synthetic pretrain?** Phi-4-reasoning pushes a "mid-train inside pretrain" line; OLMo 3 keeps them separate. Controlled comparisons across the two philosophies at matched compute are not yet public. [[interplay-pretraining-midtraining-rl]] is the closest evidence and is controlled-but-small-scale.
- **How much cold-start is a ceiling on RL?** R1 used 800K cold-start traces before RL; s1 and LIMO show 10³-sample cold-starts can still produce strong reasoning after RL. The open question is whether the *upper bound* on RL gains depends on cold-start size — whether 800K traces permanently shape the policy distribution RL explores from. Evidence is mixed; both sides have strong anecdotes.
- **Is long-context extension really separable from mid-training, or is this an OLMo-3 convention?** Llama 3 and Qwen 3 treat long-context as a staged pretrain extension; OLMo 3 treats it as a named post-pretrain stage fed from a Longmino pool that *overlaps* with Dolmino. The overlap suggests the two stages are not fully separable — which is consistent with the three-job decomposition in §4 but complicates the clean stage-table picture.

These are the places where CLAUDE.md's "core insight per landmark paper" principle meets open research: read the interplay paper for its claim, read OLMo 3 for the artifact, and keep a separate note for the questions neither answers yet.

## Connections and what's next

- **[[interplay-pretraining-midtraining-rl]]** — the controlled-experiment evidence that grounds the stage-by-stage allocation argument. Read §Main causal claims.
- **[[olmo-3]]** — the clearest public model-flow example; the primary source for stage budgets and eval gates. Read the model-flow diagram and the expanded training-data section.
- **[[deepseek-r1]] + model-report** — the cold-start SFT pattern; read the R1 vs R1-Zero comparison for the readability fix.
- **[[long-context-llama3]] + [[prolong]] + [[longalign]]** — the long-context three-job decomposition; each source covers one job cleanly.
- **[[front-loading-reasoning]]** — the diversity-vs-quality asymmetric-allocation rule across stages.
- **ch-31 (previous)** — DPO / preference optimization as the bridge between SFT and RL; this chapter zooms out to the whole pipeline.
- **ch-33 (next)** — Tulu 3 and Llama 3 as case studies in *executing* multi-round post-training on top of this stage stack.
- **ch-34 (next)** — Qwen 2.5/3 and Phi 3/4 as case studies showing where labs disagree on where mid-training ends and SFT begins.

## Further reading

- [[interplay-pretraining-midtraining-rl]] — the controlled framework; the edge-of-competence result is the most actionable.
- [[olmo-3]] — the model-flow report; the per-stage compute disclosure is rare and useful.
- [[deepseek-r1]] paper + model-report — read both; the paper frames R1-Zero's claim, the model-report details the cold-start.
- [[long-context-llama3]] — the production 128K recipe with explicit stage budgets and RoPE rescale values.
- [[prolong]] — the document-coherence ablation that anchors the long-context data rule.
- [[front-loading-reasoning]] — the diversity-vs-quality allocation rule across stages.

## Companion visualization

**[figures/pipeline-stages.html](figures/pipeline-stages.html)** — a horizontal Gantt of the five stages (pretrain → mid-train → long-context → SFT → RL). Select among four disclosed configurations (Llama 3 / OLMo 3 / DeepSeek-R1 / Phi-4-reasoning). Each stage shows its token budget as bar width, its data-mix composition as a stacked fill (web / math / code / long-doc / instruction / preference), and a tooltip with the eval gate. Toggling configurations reveals the *rebalancing* between labs — e.g., Phi-4 compresses pretrain and enlarges mid-training relative to OLMo 3; DeepSeek-R1's cold-start SFT is tiny (800K samples ≈ tens of millions of tokens) compared to its mid-training (inherited from V3); Llama 3 has an outsized long-context stage (~800B tokens). Use it as a mental model for how to allocate compute when you design your own pipeline.
