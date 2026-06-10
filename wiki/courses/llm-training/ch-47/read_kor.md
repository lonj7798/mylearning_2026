<!-- chapter: ch-47
     track: eval
     kind: content
     title: Eval Harness Design
     deps: [ch-36]
     sources: [[olmo-3]], [[llama-3]], [[judge-llm-bias]], [[ruler]], [[longbench]], [[needle-in-haystack-data]], [[bfcl]], [[babilong]], [[harmbench-data]], [[wildguard-data]], [[salad-bench]], [[webarena-data]], [[faithful-synth-eval]]
     figures: figures/eval-config-matrix.html
     opens_track: eval (ch-47..ch-53)
-->

# 47장 — Eval Harness 설계

> **핵심 통찰.** benchmark number는 모델의 속성이 아니다. 그것은 6-tuple `(task_shape, prompt_template, matcher, inference_config, subset_slice, harness_version)`의 출력이다. 이 tuple의 어떤 좌표든 바꾸면 weights가 고정되어 있어도 숫자는 ±5–15 points 움직일 수 있다. eval track이 여기서 시작하는 이유는 모든 downstream decision — data mix(ch-51), RM calibration(ch-49), long-context claims(ch-50), safety release gates(ch-52) — 이 harness를 *통해* 숫자를 소비하기 때문이다. "OLMES reports 74% on ARC-E"는 모델에 대한 진술인 만큼 harness에 대한 진술이다. engineering move는 harness를 black box로 취급하는 것을 멈추고, *version, slice, replay*해야 하는 infrastructure 조각으로 다루는 것이다. `lm-eval-harness`, `OLMES`, `AgentBench`는 바로 이를 가능하게 하도록 만들어졌다.
>
> **가이드라인.** eval number를 인용하기 전에 여섯 좌표를 모두 이름 붙이라. capability가 실제로 요구하는 것에 맞춰 `task_shape`(MCQ / generation / open-ended / agentic)를 고르라. 더 싸다는 이유로 HumanEval을 MCQ frame으로 보고하지 말라. 감당할 수 있는 bias profile에 맞춰 `matcher`(exact-match / regex / AST / unit-test executor / LLM-judge / classifier)를 고르라. [[bfcl]]에는 AST, NIAH에는 substring, code에는 unit-test, open-ended에는 [[judge-llm-bias]]에 따른 swap-and-average LLM-judge. `inference_config`(temperature, top-p, max_tokens, stop, system prompt, few-shot count)를 freeze하고 version하라. `temperature=0.0, max_tokens=512, 5-shot`은 detail이 아니라 eval hyperparameter다. 기본적으로 subset과 difficulty로 slice하라. aggregate mean은 hack을 숨긴다([[faithful-synth-eval]]). release note에는 harness commit SHA를 pin하라. weights change 없이 number가 바뀌면 delta를 model이 아니라 harness diff에 귀속하라.

---

## 이 장이 존재하는 이유

Ch-36은 data/SFT 절반을 training run으로 마쳤다. 독자가 그 run에 대해 처음 묻는 질문은 "더 좋아졌는가?"이며, eval track은 그 질문에 *엄밀하게* 답하기 위해 존재한다. 여기서 rigor는 sophistication이 아니라 **accounting**이다. 같은 MMLU number도 letter-perplexity matcher(OLMES style)를 썼는지, generate-then-regex matcher(lm-eval default)를 썼는지, CoT-with-LLM-judge pipeline을 썼는지에 따라 62% 또는 71%가 될 수 있다. [[olmo-3]]는 model *flow* — data, checkpoints, eval scaffolding — 가 artifact라는 점을 강조한다. 이 장은 그 flow에서 harness 조각을 추출하고 모든 dial에 이름을 붙인다. 이후 장들(ch-48: MCQ; ch-49: generation/LLM-judge; ch-50: long-context; ch-51: capability-specific; ch-52: safety; ch-53: contamination)은 6-tuple의 한 좌표씩 확대한다.

---

## §1 Task shape가 downstream 전체를 결정한다

네 가지 canonical shape가 있으며, 각각 native metric family를 가진다.

| task_shape | examples | canonical metric | inference_config default | matcher type |
|---|---|---|---|---|
| **Multiple-choice (MCQ)** | MMLU, ARC, HellaSwag, WinoGrande | accuracy via letter-perplexity OR generate-and-match | `T=0` + log-probs over `{A,B,C,D}` tokens | argmax-log-prob |
| **Free generation, verifiable** | GSM8K, MATH, HumanEval, [[bfcl]] | pass@1 / pass@k / pass^k | `T=0` for pass@1; `T=0.8 top_p=0.95 n=k` for pass@k | exact / regex / AST / unit-test |
| **Free generation, open-ended** | MT-Bench, AlpacaEval, summarization | win-rate / Elo / ROUGE | `T=0.7` greedy-beam variant | LLM-judge / human |
| **Agentic / multi-step** | [[webarena-data]], [[bfcl]] V3/V4, SWE-bench | task-success predicate, pass^k | `T=0` or `T=0.6`, tool-call-budget | predicate over final env state |

harness design의 첫 번째 규칙은 "capability가 사는 shape를 고르라"다. 모델의 *math ability*는 free-generation-verifiable에 산다. `(A) 12 (B) 14 (C) 16 (D) 18` 위 MCQ로 채점하면 solving이 아니라 ranking을 측정한다. [[llama-3]] Table 6 footnote의 "HumanEval is reported generate@k, not pass@1-MCQ"는 cosmetic choice가 아니다. 그 capability에 대해 유일하게 정직한 shape다.

두 번째 규칙은 "metric을 shape에 맞추라"다. pass@k는 k rollout을 평균하며 generation과 일관된다. Pass^k([[bfcl]] V3+)는 k개의 independent rollout이 *모두* 성공해야 한다. 이는 agentic shape가 측정하려는 trait인 agent *reliability*를 측정한다. pass^k를 의미하면서 pass@1을 보고하면 capability와 consistency를 혼동한다.

---

## §2 Harness를 6-tuple로 보기, 그리고 세 reference implementation

2024–2026년 practice를 지배하는 harness는 세 가지다. 축은 겹치지만 차이가 충분히 커서, 이들 사이를 이동하면 숫자가 바뀐다.

| axis | lm-eval-harness (EleutherAI) | OLMES ([[olmo-3]] co-release) | AgentBench / WebArena-style |
|---|---|---|---|
| **task types** | MCQ (dominant), generation, perplexity | MCQ, generation, generative-classification | stateful agentic / multi-turn / tool |
| **prompt-format control** | YAML task file, task별 chat-template hook | task id 안의 명시적 prompt format version string | role/turn stack + env observation format |
| **slicing** | per-subtask, not per-difficulty by default | per-subset + per-difficulty first-class; `format_version` field | per-task-category; `pass^k` slice by task family |
| **versioning** | commit SHA + task-file hash | explicit `olmes:mmlu:v1.1` ids; numbers attributed to id | Docker image digest + task JSON hash |
| **extension point** | write a `TaskConfig` subclass | contribute a new `olmes-task` with format-version | write a new env Dockerfile + predicate script |
| **LLM-judge support** | pluggable but not first-class | second-class | not applicable |
| **executable eval** | not native | not native | native (env state predicates) |
| **inference backend** | hf, vllm, openai, tgi | hf, vllm, local-chat-api | any (talks to a served policy) |

공통점: 세 harness 모두 *task definition*을 *model client*에서 분리한다. 이것이 script가 아니라 harness가 되게 하는 조건이다. 차이점: OLMES는 *release-scale reproducibility*를 위해 설계되었다. task id가 format version을 내장하므로 "olmes:arc-c:v1.0"은 제안이 아니라 계약이다. lm-eval은 *breadth-first academic reproducibility*를 위해 설계되었다. 하나의 harness, 많은 task, 하나의 score table이다. AgentBench 스타일 framework는 *statefulness*를 위해 설계되었다. [[webarena-data]]가 Docker-compose bundle을 제공하는 이유는 agent task가 prompt가 아니라 environment이기 때문이다.

실용적 결과: harness 이름 없이 숫자를 cross-cite하지 말라. "ARC-C 84%"는 반쪽 문장이다. "ARC-C olmes:v1.1 84%"는 fact다.

---

## §3 Inference-time config는 score의 조용한 세 번째 column이다

harness는 task를 generator에 통과시킨다. generator에는 knob가 있다. 이 knob들은 *eval hyperparameters*다.

| knob | MCQ letter-perplexity | MCQ generate-and-match | code pass@1 | code pass@k | open-ended + judge | long-context NIAH |
|---|---|---|---|---|---|---|
| `temperature` | N/A (scoring logits) | 0.0 | 0.0 | 0.8 (attested [[llama-3]] eval) | 0.7 | 0.0 |
| `top_p` | N/A | 1.0 | 1.0 | 0.95 | 0.95 | 1.0 |
| `max_tokens` | N/A | ~16 | 512 | 512 | 1024–2048 | 128–256 |
| `stop` | N/A | `\n\n`, `Q:` | lang-specific delimiters | same | `</s>`, judge-end | exact-answer marker |
| `n` (samples) | 1 | 1 | 1 | 10–100 | 1 (× swap) | 1 |
| `few-shot` count | 0–5 (OLMES: 5 default) | 0–5 | 0–3 | 0–3 | 0 | 0 |
| `system prompt` | harness-default | harness-default | "You are a Python programmer" | same | task-specific | task-specific |

각 행은 ±1–10 absolute points를 움직인다. GSM8K에서 `temperature`를 0.0 → 0.2로 움직이면 7–13B 모델의 pass@1이 1–3 points 떨어진다. HumanEval에서 `max_tokens`를 256 → 1024로 움직이면 solution이 끝까지 나올 수 있어 pass@1이 3–6 points 오른다. [[judge-llm-bias]]는 GPT-4-judge eval에서 `(A,B)` vs `(B,A)` ordering만으로 22–40% flip rate가 생김을 보인다. position도 sampler에 입력하지 않을 뿐 inference-config knob다.

**Engineering rule.** 같은 task, 같은 weights에서 두 run이 >1 point 차이나면 model보다 inference_config를 먼저 의심하라. config를 weights 옆에 JSON으로 pin하라. OLMES는 이를 task id에 embed하고, lm-eval은 `results.json`에 dump하며, AgentBench는 env manifest에 넣는다.

---

## §4 Matchers: accuracy가 결정되는 last-mile surface

matcher는 model output을 {correct, incorrect}로 변환한다. 다섯 family:

1. **Exact / substring.** NIAH의 original matcher([[needle-in-haystack-data]]): output에 needle string이 나타나는가? 취약하다. paraphrase는 실패한다. 싸다. extra compute가 없다.
2. **Regex / structured extract.** GSM8K의 `\\boxed{.*}` pull, MMLU의 letter-extract. format drift에 민감하다. model이 "The answer is **B**."처럼 bold markdown으로 답하면 regex가 깨진다.
3. **AST / semantic equivalence.** [[bfcl]]은 predicted tool call과 gold call을 `(name, kwargs)`로 parse하고, arg order와 literal form(`1.0 ≡ 1`)을 normalize해 match한다. language-specific parser 비용을 치르는 대신 format spuriousness를 제거한다.
4. **Executable / unit-test.** HumanEval, [[webarena-data]] predicates, [[bfcl]]의 live-API subset. matcher가 output을 *실행*한다. fidelity가 가장 높고 infra cost도 가장 크다(sandboxing, determinism, timeouts).
5. **Learned classifier or LLM-judge.** [[harmbench-data]]는 non-copyright behaviours에 대해 fine-tuned Llama-2-13B-Chat을 success classifier로 제공한다. [[judge-llm-bias]]는 GPT-4-as-judge의 bias(position ~20–30% flip, verbosity, self-enhancement)를 정량화한다. capability에 대한 verifier가 없을 때 사용하라. verifier가 있다면 유일한 matcher로 쓰지 말라.

**LLM-judge matcher를 위한 mitigation**([[judge-llm-bias]]에서):
- side를 swap하고, consistent할 때만 win으로 잡고 아니면 tie로 둔다. position bias를 ~15 pp 낮춘다.
- objective task에는 reference answer를 첨부한다. MT-Bench에서 human-agreement를 ~10 pp 높인다.
- candidate model이 자신을 judge하게 하지 말라. self-enhancement bias는 GPT-4와 Claude 모두에서 입증되었다.
- refusal task에서는 `response_harm_label`과 `refusal_label`을 분리하라([[wildguard-data]]). 단일 score는 두 capability를 혼동한다.
- classifier matcher([[harmbench-data]]의 fine-tuned Llama-2-13B-Chat, [[wildguard-data]]의 WildGuard-7B, [[salad-bench]]의 MD-Judge)는 다음에 대해 stress-test하라: benign paragraphs, unrelated harmful completions, refusal-then-comply outputs. ungrounded classifier는 score column을 조용히 부패시킨다.

**Matcher hierarchy — 맞는 것 중 가장 강한 것을 쓰라**:

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

위의 것이 capability에 맞지 않을 때만 아래로 내려가라. HumanEval을 LLM-judge로 보고하는 model은 signal을 낭비하는 것이다. executor가 있다.

---

## §5 Slicing — aggregate number는 hack을 숨긴다

[[faithful-synth-eval]]는 synthetic-data audit에서 이 점을 설명하지만, eval에도 그대로 적용된다. average는 compression이고, compression은 tail을 숨긴다. [[ruler]]는 이를 long-context에 대해 operationalize한다. length × complexity knob를 가진 13-task suite가 존재하는 이유는 같은 모델이 single-needle-4K에서는 99%, multi-hop-32K에서는 40%일 수 있고, 어느 숫자 하나만으로는 informative하지 않기 때문이다.

기본 slicing dimensions:

- **Per-subset.** MMLU는 57 subjects가 있다. headline 70%는 STEM 90%, humanities 50%일 수 있다. OLMES는 이를 기본적으로 surface한다.
- **Per-difficulty.** BBH는 human-solve-rate bucketing을 사용한다. GSM8K는 chain length로 re-bucket되었다. Long-context bench는 token budget(4K/8K/32K/128K per [[ruler]] Table 3)으로 slice한다.
- **Per-domain / per-language.** [[longbench]]는 bilingual English/Chinese다. 하나의 aggregate는 language별 collapse를 숨긴다.
- **Per-category (safety).** [[salad-bench]]의 66 leaf categories와 [[harmbench-data]]의 7 semantic × 4 functional grid. reporting unit은 mean이 아니라 cell이다.
- **Per-failure-mode (agent).** [[webarena-data]]의 세 task type(info-lookup / content-producing / state-modifying); type별 pass^k. 50% aggregate가 80/50/20일 수 있기 때문이다.

slice할 수 없는 harness는 measurement device가 아니라 artifact다. lm-eval은 per-subtask JSON을 반환한다. OLMES는 difficulty bucket을 first-class field로 올린다. AgentBench는 per-task trajectory를 반환하므로 slicing은 post-hoc이다.

**Minimal slice schema.** 모든 harness run은 task별로 다음을 emit해야 한다.

```jsonl
{"task": "mmlu", "subset": "anatomy",        "difficulty": null,  "n": 135, "score": 0.71, "harness": "olmes:v1.1"}
{"task": "mmlu", "subset": "machine_learning","difficulty": null, "n": 112, "score": 0.54, "harness": "olmes:v1.1"}
{"task": "ruler","subset": "mk_niah",         "difficulty": "64K","n": 500, "score": 0.41, "harness": "ruler:v1.0"}
{"task": "bfcl","subset": "parallel_multiple","difficulty": null, "n": 200, "score": 0.62, "harness": "bfcl:v2.0-live"}
```

aggregate "모델이 0.7에 있다"는 파생값이며 primary가 아니다. [[ruler]] Table 3의 "주장된 context length vs effective context length"도 이렇게 cross-read된다. aggregate는 `85.6`이고, slice는 그 85.6을 어디서 벌고 어디서 잃는지 드러낸다.

---

## §6 Versioning — weights 없이 숫자가 바뀔 때

귀속하기 가장 어려운 숫자는 harness가 움직였기 때문에 움직인 숫자다. 세 가지 canonical drift source:

1. **Prompt-template change.** OLMES는 task id에 format string을 명시적으로 pin한다(`olmes:mmlu:v1.1` vs `v1.2`). [[llama-3]]와 [[olmo-3]] 모두 named harness version에 대해 숫자를 보고한다. 2023년에 lm-eval-harness가 `mmlu` → `mmlu_flan_n_shot`으로 이름을 바꿨을 때 recorded score는 전체적으로 2–4 points 움직였다.
2. **Matcher update.** [[bfcl]] V1 → V2는 Live data와 더 엄격한 AST normalizer를 추가했다. unchanged model의 V2 score는 V1보다 안정적으로 3–8 points 낮다. V1은 saturated였고, V2가 honest number다.
3. **Data refresh / contamination fix.** WildGuardTest v1 → v1.1은 inter-annotator agreement를 통과하지 못한 item을 제거했다. per-item number는 움직였지만 reliability(Fleiss κ)는 올랐다. release note는 "v1.1 is harder and cleaner"라고 말해야 한다. 그렇지 않으면 reader가 drop을 regression으로 귀속한다.

**Release discipline.** model card의 모든 eval row는 `(harness_name, harness_version, inference_config_hash, slice_id)`를 가져야 한다. tuple을 comparison의 단위로 취급하라. [[olmo-3]]의 "fully open" 철학은 eval에 자연스럽게 매핑된다. model *flow*는 weights뿐 아니라 eval provenance도 포함한다.

---

## §7 Harness 구성 — release-grade config file의 형태

6-tuple은 추상화다. 구체적인 release config는 한 곳에 여섯 좌표를 모두 이름 붙인다. ch-36의 SFT checkpoint를 위한 minimal `eval.yaml`:

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

각 row는 self-contained fact다. 이 file과 checkpoint가 있으면 third party가 숫자를 재구성할 수 있다. 이것이 operational하게 "version the harness"가 뜻하는 바다. [[llama-3]] Table 6은 이 구조의 readable projection이다. OLMo 3([[olmo-3]])도 자체 version을 제공한다.

**이 file이 막는 첫 failure mode**는 silent cross-harness comparison이다. reviewer가 release A의 "MMLU 71%"와 release B의 "MMLU 74%"를 보는데 둘 다 harness id가 없으면, B가 더 낫다고 결론 내린다. 실제 delta는 (a) `olmes:v1.0`에서 `olmes:v1.1`로 prompt-format이 올라가며 생긴 2 points, (b) `few_shot`을 0에서 5로 움직이며 생긴 1 point였을 수 있다. 둘 다 capability difference가 아니다.

---

## Connections

- **ch-36** — SFT checkpoint는 이 harness가 소비하는 첫 artifact다. training-time의 `metrics.jsonl` schema는 여기서 도입한 eval-time schema와 병렬적이다.
- **ch-48** — MCQ harness deep-dive: MMLU / ARC / HellaSwag; letter-perplexity vs generation-match.
- **ch-49** — LLM-judge / pairwise generation; 여기서 도입한 [[judge-llm-bias]] §4 mitigation 위에 구축한다.
- **ch-50** — long-context: RULER / NIAH / BABILong / LongBench as a family; [[ruler]] generator protocol이 축이다.
- **ch-51** — capability-specific: code(HumanEval / MBPP / SWE-bench), math(GSM8K / MATH), tool-calling([[bfcl]] V2/V3/V4).
- **ch-52** — safety harness: [[harmbench-data]], [[salad-bench]], [[wildguard-data]]; classifier matchers and attack versioning.
- **ch-53** — contamination and decontamination; data-track과 eval-track을 가로지르는 versioning problem.
- **Track 8 (Infra) / ch-54+** — harness는 inference-serving workload다. server version을 harness version과 함께 pin하라.

## Further reading

- [[olmo-3]] — OLMES design philosophy; model-flow transparency를 eval로 확장.
- [[llama-3]] — release-scale capability suite; per-capability reporting convention.
- [[judge-llm-bias]] — LLM-judge matcher의 bias; swap-and-average; reference-guided grading.
- [[ruler]] — parameterised long-context generator; length × complexity separation.
- [[bfcl]] — AST matcher + executable subset을 가진 tool-calling harness; V1→V4 version generations.
- [[needle-in-haystack-data]] — minimum-viable long-context harness; exact-match matcher limits.
- [[longbench]] — natural-task long-context eval; task-specific metric zoo.
- [[babilong]] — hybrid synthetic-natural harness; 10M-token scaling protocol.
- [[harmbench-data]] — behavior library + attack wrapper; held-out classifier as matcher.
- [[wildguard-data]] — moderation eval; three-label separation; reliability signal로서 κ.
- [[salad-bench]] — hierarchical safety taxonomy; reporting unit으로서 per-category slicing.
- [[webarena-data]] — agentic harness; predicate matchers; versioning으로서 Docker-pinning.
- [[faithful-synth-eval]] — aggregate number가 distribution corruption을 숨기는 이유; tail slicing.

## Companion visualization

**[figures/eval-config-matrix.html](figures/eval-config-matrix.html)** — 대화형 eval-config explorer. `task_shape`, `harness`, `inference_config` preset을 고르면, panel이 expected metric surface(어떤 matcher가 작동하는지, 어떤 slicing이 가능한지, 어떤 versioning primitive가 적용되는지)와 해당 cell의 가장 흔한 pitfalls 세 가지를 보여준다. 새 capability를 harness에 mapping할 때 사용하라. 세 harness에 걸쳐 네 shape를 클릭하고 pitfalls를 읽은 뒤, 처음 보고하는 number 옆에 6-tuple(`task_shape / prompt_template / matcher / inference_config / subset_slice / harness_version`)을 JSON으로 commit하라.
