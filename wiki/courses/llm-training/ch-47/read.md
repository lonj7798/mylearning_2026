<!-- chapter: ch-47
     track: eval
     kind: content
     title: Eval Harness Design
     deps: [ch-36]
     sources: [[olmo-3]], [[llama-3]], [[judge-llm-bias]], [[ruler]], [[longbench]], [[needle-in-haystack-data]], [[bfcl]], [[babilong]], [[harmbench-data]], [[wildguard-data]], [[salad-bench]], [[webarena-data]], [[faithful-synth-eval]]
     figures: figures/eval-config-matrix.html
     opens_track: eval (ch-47..ch-53)
-->

# Chapter 47 — Eval Harness Design

> **Core insight.** A benchmark number is not a property of the model. It is the output of a 6-tuple: `(task_shape, prompt_template, matcher, inference_config, subset_slice, harness_version)`. Change any coordinate of that tuple and the number can move ±5–15 points with the weights frozen. The eval track opens here because every downstream decision — data mix (ch-51), RM calibration (ch-49), long-context claims (ch-50), safety release gates (ch-52) — consumes numbers *through* a harness. "OLMES reports 74% on ARC-E" is a statement about the harness as much as about the model. The engineering move is to stop treating the harness as a black box and start treating it as a piece of infrastructure you *version, slice, and replay* — which is exactly what `lm-eval-harness`, `OLMES`, and `AgentBench` were built to let you do.
>
> **Guideline.** Before quoting any eval number, name all six coordinates. Choose `task_shape` (MCQ / generation / open-ended / agentic) to match what the capability actually requires — do not report HumanEval in the MCQ frame just because it's cheaper. Pick the `matcher` (exact-match / regex / AST / unit-test executor / LLM-judge / classifier) with the bias profile you can afford: AST for [[bfcl]], substring for NIAH, unit-test for code, LLM-judge with swap-and-average per [[judge-llm-bias]] for open-ended. Freeze the `inference_config` (temperature, top-p, max_tokens, stop, system prompt, few-shot count) and version it: `temperature=0.0, max_tokens=512, 5-shot` is an eval hyperparameter, not a detail. Slice by subset and difficulty by default — aggregate means hide the hacks ([[faithful-synth-eval]]). Pin the harness commit SHA in the release note; when the number changes without weight changes, attribute the delta to the harness diff, not the model.

---

## Why this chapter exists

Ch-36 closed the data/SFT half of the course with a training run. The first question a reader asks about that run is "did it get better?" — and that is the question the eval track exists to answer *rigorously*. Rigour here is not sophistication; it is **accounting**. The same MMLU number can be 62% or 71% depending on whether you used the letter-perplexity matcher (OLMES style), the generate-then-regex matcher (lm-eval default), or a CoT-with-LLM-judge pipeline. [[olmo-3]] makes the point that the model *flow* — data, checkpoints, eval scaffolding — is the artifact; this chapter extracts the harness piece of that flow and names every dial on it. Subsequent chapters (ch-48: MCQ; ch-49: generation/LLM-judge; ch-50: long-context; ch-51: capability-specific; ch-52: safety; ch-53: contamination) each zoom into one coordinate of the 6-tuple.

---

## §1 Task shape determines everything downstream

There are four canonical shapes, each with a native metric family:

| task_shape | examples | canonical metric | inference_config default | matcher type |
|---|---|---|---|---|
| **Multiple-choice (MCQ)** | MMLU, ARC, HellaSwag, WinoGrande | accuracy via letter-perplexity OR generate-and-match | `T=0` + log-probs over `{A,B,C,D}` tokens | argmax-log-prob |
| **Free generation, verifiable** | GSM8K, MATH, HumanEval, [[bfcl]] | pass@1 / pass@k / pass^k | `T=0` for pass@1; `T=0.8 top_p=0.95 n=k` for pass@k | exact / regex / AST / unit-test |
| **Free generation, open-ended** | MT-Bench, AlpacaEval, summarization | win-rate / Elo / ROUGE | `T=0.7` greedy-beam variant | LLM-judge / human |
| **Agentic / multi-step** | [[webarena-data]], [[bfcl]] V3/V4, SWE-bench | task-success predicate, pass^k | `T=0` or `T=0.6`, tool-call-budget | predicate over final env state |

The first rule of harness design is "pick the shape the capability lives in." A model's *math ability* lives in free-generation-verifiable; scoring it as MCQ over `(A) 12 (B) 14 (C) 16 (D) 18` measures ranking, not solving. [[llama-3]]'s Table 6 footnote that "HumanEval is reported generate@k, not pass@1-MCQ" is not a cosmetic choice; it is the only honest shape for the capability.

The second rule is "match the metric to the shape." pass@k averages over k rollouts and is consistent with generation. Pass^k ([[bfcl]] V3+) requires *every* of k independent rollouts to succeed — it measures agent *reliability*, which is the trait the agentic shape exists to measure. Reporting pass@1 when you mean pass^k conflates capability with consistency.

---

## §2 The harness as a 6-tuple, and three reference implementations

Three harnesses dominate 2024–2026 practice. Their axes overlap but differ enough that moving between them changes numbers.

| axis | lm-eval-harness (EleutherAI) | OLMES ([[olmo-3]] co-release) | AgentBench / WebArena-style |
|---|---|---|---|
| **task types** | MCQ (dominant), generation, perplexity | MCQ, generation, generative-classification | stateful agentic / multi-turn / tool |
| **prompt-format control** | YAML task files, per-task chat-template hook | explicit prompt format version string in task id | role/turn stack + env observation format |
| **slicing** | per-subtask, not per-difficulty by default | per-subset + per-difficulty first-class; `format_version` field | per-task-category; `pass^k` slice by task family |
| **versioning** | commit SHA + task-file hash | explicit `olmes:mmlu:v1.1` ids; numbers attributed to id | Docker image digest + task JSON hash |
| **extension point** | write a `TaskConfig` subclass | contribute a new `olmes-task` with format-version | write a new env Dockerfile + predicate script |
| **LLM-judge support** | pluggable but not first-class | second-class | not applicable |
| **executable eval** | not native | not native | native (env state predicates) |
| **inference backend** | hf, vllm, openai, tgi | hf, vllm, local-chat-api | any (talks to a served policy) |

What is common: all three separate *task definition* from *model client*, which is what makes them harnesses rather than scripts. What differs: OLMES is designed for *release-scale reproducibility* (the task id embeds the format version so "olmes:arc-c:v1.0" is a contract, not a suggestion); lm-eval is designed for *breadth-first academic reproducibility* (one harness, many tasks, one score table); AgentBench-style frameworks are designed for *statefulness* ([[webarena-data]] ships a Docker-compose bundle because an agent task is the environment, not the prompt).

A practical consequence: never cross-cite numbers between harnesses without naming the harness. "ARC-C 84%" is a half-sentence; "ARC-C olmes:v1.1 84%" is a fact.

---

## §3 Inference-time config is the silent third column of the score

The harness runs the task through a generator. The generator has knobs. Those knobs are *eval hyperparameters*:

| knob | MCQ letter-perplexity | MCQ generate-and-match | code pass@1 | code pass@k | open-ended + judge | long-context NIAH |
|---|---|---|---|---|---|---|
| `temperature` | N/A (scoring logits) | 0.0 | 0.0 | 0.8 (attested [[llama-3]] eval) | 0.7 | 0.0 |
| `top_p` | N/A | 1.0 | 1.0 | 0.95 | 0.95 | 1.0 |
| `max_tokens` | N/A | ~16 | 512 | 512 | 1024–2048 | 128–256 |
| `stop` | N/A | `\n\n`, `Q:` | lang-specific delimiters | same | `</s>`, judge-end | exact-answer marker |
| `n` (samples) | 1 | 1 | 1 | 10–100 | 1 (× swap) | 1 |
| `few-shot` count | 0–5 (OLMES: 5 default) | 0–5 | 0–3 | 0–3 | 0 | 0 |
| `system prompt` | harness-default | harness-default | "You are a Python programmer" | same | task-specific | task-specific |

Each row moves ±1–10 absolute points. Moving `temperature` from 0.0 → 0.2 on GSM8K drops pass@1 by 1–3 points on 7–13B models; moving `max_tokens` from 256 → 1024 on HumanEval raises pass@1 by 3–6 points because solutions get to finish. [[judge-llm-bias]] shows a 22–40% flip rate from `(A,B)` vs `(B,A)` ordering on GPT-4-judge evals — position is an inference-config knob too, even though you don't type it into a sampler.

**Engineering rule.** If two runs disagree by >1 point on the same task and same weights, suspect inference_config before suspecting the model. Pin the config as JSON alongside the weights; OLMES embeds this in the task id, lm-eval dumps it to `results.json`, AgentBench puts it in the env manifest.

---

## §4 Matchers: the last-mile surface where accuracy is decided

A matcher converts model output into {correct, incorrect}. Five families:

1. **Exact / substring.** NIAH's original matcher ([[needle-in-haystack-data]]): does the needle string appear in the output? Brittle — paraphrase fails it. Cheap — zero extra compute.
2. **Regex / structured extract.** GSM8K's `\\boxed{.*}` pull, MMLU's letter-extract. Sensitive to format drift: a model that answers "The answer is **B**." in bold markdown breaks the regex.
3. **AST / semantic equivalence.** [[bfcl]] parses predicted tool calls and gold calls into `(name, kwargs)`, normalizes arg order and literal form (`1.0 ≡ 1`), and matches. Eliminates format spuriousness at the cost of language-specific parsers.
4. **Executable / unit-test.** HumanEval, [[webarena-data]] predicates, [[bfcl]]'s live-API subset. The matcher *runs* the output. Strongest fidelity, highest infra cost (sandboxing, determinism, timeouts).
5. **Learned classifier or LLM-judge.** [[harmbench-data]] ships a fine-tuned Llama-2-13B-Chat as the success classifier for non-copyright behaviours; [[judge-llm-bias]] quantifies biases of GPT-4-as-judge (position ~20–30% flip, verbosity, self-enhancement). Use when no verifier exists for the capability; never as the only matcher if one does.

**Mitigations for LLM-judge matchers** (from [[judge-llm-bias]]):
- Swap sides, take a win only if consistent; else tie. Drops position bias by ~15 pp.
- Attach a reference answer on objective tasks. Raises human-agreement by ~10 pp on MT-Bench.
- Never let the candidate model judge itself. Self-enhancement bias is attested on both GPT-4 and Claude.
- For refusal tasks, separate `response_harm_label` from `refusal_label` ([[wildguard-data]]) — a single score conflates two capabilities.
- For classifier matchers ([[harmbench-data]]'s fine-tuned Llama-2-13B-Chat, [[wildguard-data]]'s WildGuard-7B, [[salad-bench]]'s MD-Judge), stress-test the matcher on: benign paragraphs, unrelated harmful completions, refusal-then-comply outputs. An ungrounded classifier quietly corrupts the score column.

**Matcher hierarchy — use the strongest one that fits**:

```
exact / substring           (cheapest, most brittle)
    ↓  fails on paraphrase / format drift
regex / structured extract  (cheap, format-sensitive)
    ↓  fails on equivalent representations (1.0 vs 1)
AST / semantic equivalence  (format-agnostic, lang-specific)
    ↓  cannot check side effects
executable / unit-test      (ground-truth on executable capabilities)
    ↓  no executor available
learned classifier / LLM-judge (catch-all, bias profile must be owned)
```

Move down this list only when the one above does not fit the capability. A model that reports HumanEval using LLM-judge is wasting signal; it has an executor.

---

## §5 Slicing — aggregate numbers hide the hacks

[[faithful-synth-eval]] makes the point for synthetic-data audit, but it transfers directly to eval: an average is a compression, and the compression hides the tail. [[ruler]] operationalises this for long-context — the 13-task suite with length × complexity knobs exists so the same model can be 99% on single-needle-4K and 40% on multi-hop-32K, and neither number alone is informative.

Default slicing dimensions:

- **Per-subset.** MMLU has 57 subjects; a headline 70% can be 90% on STEM and 50% on humanities. OLMES surfaces this by default.
- **Per-difficulty.** BBH uses a human-solve-rate bucketing. GSM8K has been re-bucketed by chain length. Long-context benches slice by token budget (4K/8K/32K/128K per [[ruler]] Table 3).
- **Per-domain / per-language.** [[longbench]] is bilingual English/Chinese; reporting one aggregate hides per-language collapse.
- **Per-category (safety).** [[salad-bench]]'s 66 leaf categories and [[harmbench-data]]'s 7 semantic × 4 functional grid — the unit of reporting is the cell, not the mean.
- **Per-failure-mode (agent).** [[webarena-data]]'s three task types (info-lookup / content-producing / state-modifying); pass^k per type, because a 50% aggregate can be 80/50/20.

A harness that cannot slice is an artefact, not a measurement device. lm-eval returns per-subtask JSON; OLMES promotes difficulty buckets to first-class fields; AgentBench returns per-task trajectories so slicing is post-hoc.

**A minimal slice schema.** Every harness run should emit, per task:

```jsonl
{"task": "mmlu", "subset": "anatomy",        "difficulty": null,  "n": 135, "score": 0.71, "harness": "olmes:v1.1"}
{"task": "mmlu", "subset": "machine_learning","difficulty": null, "n": 112, "score": 0.54, "harness": "olmes:v1.1"}
{"task": "ruler","subset": "mk_niah",         "difficulty": "64K","n": 500, "score": 0.41, "harness": "ruler:v1.0"}
{"task": "bfcl","subset": "parallel_multiple","difficulty": null, "n": 200, "score": 0.62, "harness": "bfcl:v2.0-live"}
```

The aggregate "the model is at 0.7" is derived, never primary. [[ruler]] Table 3's "claimed context length vs effective context length" cross-reads this way: the aggregate is `85.6`, the slice reveals where the 85.6 is earned and where it is lost.

---

## §6 Versioning — when the number changes without the weights

The hardest number to attribute is one that moved because the harness moved. Three canonical drift sources:

1. **Prompt-template change.** OLMES explicitly pins the format string in the task id (`olmes:mmlu:v1.1` vs `v1.2`); [[llama-3]] and [[olmo-3]] both report numbers against named harness versions. When lm-eval-harness renamed `mmlu` → `mmlu_flan_n_shot` in 2023, recorded scores shifted 2–4 points overall.
2. **Matcher update.** [[bfcl]] V1 → V2 added Live data and a stricter AST normalizer; V2 scores on an unchanged model are reliably 3–8 points lower than V1. V1 saturated; V2 is the honest number.
3. **Data refresh / contamination fix.** WildGuardTest's v1 → v1.1 dropped items that failed inter-annotator agreement; per-item numbers moved but reliability (Fleiss κ) rose. The release note must say "v1.1 is harder and cleaner" — otherwise readers attribute the drop to regression.

**Release discipline.** Every eval row in a model card must carry `(harness_name, harness_version, inference_config_hash, slice_id)`. Treat the tuple as the unit of comparison. [[olmo-3]]'s "fully open" philosophy maps naturally to eval: the model *flow* includes eval provenance, not just weights.

---

## §7 Composing a harness — what a release-grade config file looks like

The 6-tuple is an abstraction; a concrete release config has all six coordinates named in one place. A minimal `eval.yaml` for an SFT checkpoint from ch-36:

```yaml
# eval.yaml — one file per release candidate
model:
  checkpoint: s3://.../sft-3B/step-12000
  tokenizer:  meta-llama/Llama-3.2-3B-Instruct
  harness_client: vllm  # or hf, openai, tgi

tasks:
  - id: olmes:mmlu:v1.1
    shape: mcq
    inference: { temperature: 0.0, max_tokens: 16, few_shot: 5 }
    slice_by: [subset]

  - id: olmes:gsm8k:v1.0
    shape: generation-verifiable
    inference: { temperature: 0.0, max_tokens: 512, few_shot: 8, stop: ["\n\nQuestion:"] }
    matcher: { type: regex, pattern: "\\\\boxed\\{(.+?)\\}" }
    slice_by: [chain_length_bucket]

  - id: bfcl:v2.0-live
    shape: generation-verifiable
    inference: { temperature: 0.0, max_tokens: 512 }
    matcher: { type: ast, normalize_literals: true }
    slice_by: [category]     # simple / parallel / multiple / relevance

  - id: ruler:v1.0
    shape: generation-verifiable
    inference: { temperature: 0.0, max_tokens: 256 }
    matcher: { type: recall_substring }
    slice_by: [task, length]   # 13 tasks x {4K,8K,16K,32K,64K,128K}

  - id: mt-bench:gpt4-0613:ref-guided:swap
    shape: open-ended
    inference: { temperature: 0.7, max_tokens: 1024 }
    matcher:
      type: llm_judge
      judge_model: gpt-4-0613
      protocol: swap-and-average   # [[judge-llm-bias]] §4
      reference_guided: true

  - id: harmbench:v1.0:test
    shape: generation-verifiable
    inference: { temperature: 0.0, max_tokens: 512 }
    matcher: { type: classifier, model: allenai/harmbench-llama-2-13b }
    slice_by: [semantic_category, functional_category]

provenance:
  harness_commit_shas:
    olmes:      "a1b2c3d"
    bfcl:       "e4f5a6b"
    ruler:      "c7d8e9f"
    lm-eval:    "b0c1d2e"
  docker_digests:
    webarena:   "sha256:..."    # if agent tasks included
  decontamination_audit: "artifacts/decon-report-v3.json"
```

Every row is a self-contained fact: given this file and the checkpoint, a third party can reconstruct the number. This is what "version the harness" means operationally. [[llama-3]]'s Table 6 is a readable projection of this structure; OLMo 3 ([[olmo-3]]) ships its own version.

**The first failure mode this file prevents** is the silent cross-harness comparison: a reviewer who sees "MMLU 71%" from release A and "MMLU 74%" from release B, both without a harness id, concludes B is better — when the true delta was (a) 2 points from a prompt-format bump between `olmes:v1.0` and `olmes:v1.1`, and (b) 1 point from moving `few_shot` from 0 to 5. Neither is a capability difference.

---

## Connections

- **ch-36** — the SFT checkpoint is the first artefact consumed by this harness. Its `metrics.jsonl` schema (training-time) parallels the eval-time schema introduced here.
- **ch-48** — MCQ harness deep-dive: MMLU / ARC / HellaSwag; letter-perplexity vs generation-match.
- **ch-49** — LLM-judge / pairwise generation; builds on [[judge-llm-bias]] §4 mitigations introduced here.
- **ch-50** — long-context: RULER / NIAH / BABILong / LongBench as a family; [[ruler]] generator protocol is the axis.
- **ch-51** — capability-specific: code (HumanEval / MBPP / SWE-bench), math (GSM8K / MATH), tool-calling ([[bfcl]] V2/V3/V4).
- **ch-52** — safety harness: [[harmbench-data]], [[salad-bench]], [[wildguard-data]]; classifier matchers and attack versioning.
- **ch-53** — contamination and decontamination; a versioning problem that crosses data-track and eval-track.
- **Track 8 (Infra) / ch-54+** — the harness is an inference-serving workload; pin the server version alongside the harness version.

## Further reading

- [[olmo-3]] — OLMES design philosophy; model-flow transparency extended to eval.
- [[llama-3]] — release-scale capability suite; per-capability reporting convention.
- [[judge-llm-bias]] — biases of LLM-judge matchers; swap-and-average; reference-guided grading.
- [[ruler]] — parameterised long-context generator; length × complexity separation.
- [[bfcl]] — tool-calling harness with AST matcher + executable subset; V1→V4 version generations.
- [[needle-in-haystack-data]] — minimum-viable long-context harness; exact-match matcher limits.
- [[longbench]] — natural-task long-context eval; task-specific metric zoo.
- [[babilong]] — hybrid synthetic-natural harness; 10M-token scaling protocol.
- [[harmbench-data]] — behavior library + attack wrapper; held-out classifier as matcher.
- [[wildguard-data]] — moderation eval; three-label separation; κ as reliability signal.
- [[salad-bench]] — hierarchical safety taxonomy; per-category slicing as the reporting unit.
- [[webarena-data]] — agentic harness; predicate matchers; Docker-pinning as versioning.
- [[faithful-synth-eval]] — why aggregate numbers hide distribution corruption; tail slicing.

## Companion visualization

**[figures/eval-config-matrix.html](figures/eval-config-matrix.html)** — interactive eval-config explorer. Pick `task_shape`, `harness`, and `inference_config` preset; the panel shows the expected metric surface (which matcher fires, what slicing is available, what versioning primitive applies) and the three most common pitfalls for that cell. Use it when mapping a new capability onto a harness: click through the four shapes across the three harnesses and read the pitfalls, then commit the 6-tuple (`task_shape / prompt_template / matcher / inference_config / subset_slice / harness_version`) as JSON alongside the first reported number.
