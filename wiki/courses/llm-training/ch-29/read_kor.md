<!-- chapter: ch-29
     track: synthetic
     kind: lab
     title: Lab — Synthetic Instruction Set with Filter + Dedup + Verify
     deps: [ch-28]
     sources: [[self-instruct]], [[evol-instruct]], [[cherry-llm]], [[ifd]], [[minhash-lsh]], [[deduplicating-training-data]], [[apigen]], [[lima]], [[alpagasus]], [[deita]], [[karpathy-training-neural-net-recipe]]
     figures: figures/filter-cascade.html
     capstone_for: synthetic-track (ch-18..ch-29)
-->

# 29장 — Lab: Filter + Dedup + Verify가 있는 Synthetic Instruction Set

> **핵심 통찰.** Synthetic SFT pool은 모든 sample이 각 filter stage를 거치는 survival rate를 이름 붙여 말할 수 있기 전까지 가치가 없다. 이 track의 capstone은 생성하는 5K instructions가 아니다. 모든 slot에 integer가 들어간 flow diagram `5000 raw → N_format → N_IFD → N_dedup → N_verified`와, matched-size [[lima]] subset 대비 delta-win-rate를 통해 pipeline이 제 몫을 했음을 증명하는 것이다.
>
> **가이드라인.** [[self-instruct]] bootstrap + [[evol-instruct]] depth operators로 generate하라. Format-validity → IFD difficulty([[cherry-llm]] / [[ifd]]) → MinHash dedup([[minhash-lsh]] / [[deduplicating-training-data]])를 cascade하라. Reasoning/tool-call subset은 [[apigen]]의 3-layer verifier(format → exec → judge)로 gate하라. Pool과 matched-size [[lima]]-filtered subset으로 약 1B SFT를 훈련하고, 각 filter를 isolation ablation하라. `synthetic-set-memo.md`를 ship하라. Ablation discipline은 [[karpathy-training-neural-net-recipe]]를 따른다. 하나의 change, 하나의 prediction, 하나의 outcome.

---

## Goal

Peer가 repo에서 재현할 수 있는 세 artifact:

1. **Pool.** `instructions.jsonl` 약 5K rows(full-budget) 또는 약 1K(resource-constrained). 각 row: origin operator(seed / evol-*), IFD score, MinHash band signature, 그리고 reasoning/tool-call subset에서는 verifier verdict.
2. **Cascade report.** `survival.json`에는 stage별 integer count, `ablations.json`에는 removed stage별 win-rate delta.
3. **Memo.** `synthetic-set-memo.md`, 한 페이지: survival table, 20-sample spot-check, 그리고 하나의 *specific* failure mode(generic "the model hallucinates"가 아님). [[karpathy-training-neural-net-recipe]]에 따라 실행 전에 각 ablation outcome을 예측하라. Prediction과 일치하는 ablation은 아무것도 가르치지 않는다. Surprise가 memo가 존재하는 이유다.

---

## Full-budget path

Target: 8 × H100(또는 8 × A100-40GB), 약 1B base, teacher API 약 \$20, wall clock 약 1일.

- **Teacher.** Generation에는 `gpt-4o-mini` 또는 `claude-3-5-haiku`, LLM-judge에는 `gpt-4o`. [[apigen]]은 DeepSeek-Coder-V2 + GPT-4를 사용했다. Judge가 generator와 엄격히 분리되고 더 강해야 자신의 output에 sign off하지 않는다.
- **Seed.** [[self-instruct]] style의 hand-written tasks 175개(public file 재사용). classification / open / extraction / reasoning / tool-call을 mix.
- **Generation budget.** Cascade가 약 5K target으로 prune할 headroom을 갖도록 raw 약 10K.
- **SFT model.** `meta-llama/Llama-3.2-1B` 또는 `Qwen2.5-1.5B`. 8 GPUs에서 약 30분 run이 가능할 만큼 작고, benchmark delta가 noise를 넘을 만큼 크다.
- **Eval.** MT-Bench single-turn + AlpacaEval-2 length-controlled + held-out reasoning probe 하나 + tool-call probe 하나. 같은 raw pool에서 matched-size [[lima]]-recipe-filtered subset 대비 win-rate를 report.

## Resource-constrained path

Target: 1 × GPU(≥16 GB), teacher API 약 \$3, wall clock 약 4시간.

- Raw 약 2K 생성, survivor 약 1K target.
- `SmolLM-360M` 또는 `Qwen2.5-0.5B`(약 500M)로 SFT. Delta는 작아지지만 cascade shape는 동일하다.
- 당신에게 가장 새로운 filter 두 개만 ablate하라. IFD가 익숙하면 MinHash + verifier를 유지하고, dedup이 익숙하면 IFD + verifier를 유지한다. 세 가지를 모두 cover하려다 budget을 날리지 말라.

---

## Stage 1: Generation

두 operator를 사용한다. 둘 중 어느 하나도 final pool을 단독으로 만들지 않는다.

### Self-Instruct bootstrap — breadth engine

Attested [[self-instruct]]: 175 seeds, 8 in-context examples(6 seed + 2 prior accepted), classification-vs-non-classification branching for the instance format.

```python
# self_instruct.py — breadth generator
import random
from dataclasses import dataclass
from typing import Literal, List

@dataclass
class Task:
    instruction: str; input: str; output: str
    kind: Literal["classification", "open"]
    origin: str  # "seed" | "self_instruct" | f"evol:{op}"

def build_instruction_prompt(seeds: List[Task], recent: List[Task]) -> str:
    picks = random.sample(seeds, 6) + random.sample(recent, min(2, len(recent)))
    body = "\n".join(f"Task {i+1}: {t.instruction}" for i, t in enumerate(picks))
    return "Come up with a series of tasks:\n" + body + f"\nTask {len(picks)+1}:"

def generate_instruction(client, seeds, recent) -> str:
    return client.complete(build_instruction_prompt(seeds, recent),
                           stop=["\nTask"], temperature=1.0, max_tokens=200).text.strip()

def generate_instance(client, instr: str, kind: str) -> tuple[str, str]:
    tpl = INPUT_FIRST if kind == "classification" else OUTPUT_FIRST
    return parse_instance(client.complete(tpl.format(instruction=instr)))
```

### Evol-Instruct depth — complexity engine

Attested [[evol-instruct]]: five In-Depth operators + one In-Breadth, one randomly chosen per call.

```python
# evol_instruct.py — depth operators
EVOL_PROMPTS = {
    "add_constraints":  "Rewrite the instruction to impose one extra constraint. Instruction: {x}",
    "deepening":        "Rewrite to increase depth and breadth of the question. Instruction: {x}",
    "concretizing":     "Rewrite, replacing general concepts with specific ones. Instruction: {x}",
    "reasoning_steps":  "Rewrite so that more reasoning steps are required. Instruction: {x}",
    "complicate_input": "Add complexity to the input (code, table, nesting). Instruction: {x}",
    "breadth":          "Mutate to a new instruction in a rarer domain. Instruction: {x}",
}

def evolve(client, instr: str) -> tuple[str, str]:
    op = random.choice(list(EVOL_PROMPTS))
    return client.complete(EVOL_PROMPTS[op].format(x=instr),
                           temperature=0.9, max_tokens=400).text.strip(), op

def evol_elimination(original: str, evolved: str) -> bool:
    # [[evol-instruct]] §Elimination: same-or-similar / refusal / punct-only / verbatim
    if looks_same(original, evolved): return False
    if any(m in evolved.lower() for m in ("sorry", "i cannot", "as an ai")): return False
    if len(evolved.strip(".!? ")) < 5 or evolved.strip() == original.strip(): return False
    return True
```

Self-Instruct/Evol-Instruct를 70/30으로 mix한다. Evol 쪽으로 기울면 breadth가 collapse하고, self-instruct 쪽으로 기울면 complexity tail이 flatten된다. [[evol-instruct]] Figure 1은 이것이 win condition이라고 문서화한다.

---

## Stage 2: Filter cascade

세 gate를 순서대로 둔다. 순서가 중요하다. Format이 가장 싸고 garbage를 가장 많이 죽인다. IFD는 format-clean pool에 대해 sample당 forward pass 하나다. MinHash는 post-IFD reduction의 이점을 본다.

### Filter 1 — Format validity

[[self-instruct]] §Filtering에 따라 empty outputs, `input == output`, length < 3 또는 > 2K tokens, "image"/"graph"/"file" mentions(implied multimodal), tool-call subset의 ill-formed JSON을 drop한다.

```python
def format_valid(s: Task) -> bool:
    if not s.instruction.strip() or not s.output.strip(): return False
    if s.input.strip() and s.input.strip() == s.output.strip(): return False
    if not 3 <= len(s.output.split()) <= 2000: return False
    if any(w in s.instruction.lower() for w in ("image","graph","file","picture")): return False
    return True
```

Expected survival: 70–85%(Self-Instruct는 약 67% 보고. Evol-Instruct 자체 elimination은 별도이며 survival이 더 높다).

### Filter 2 — IFD difficulty

[[ifd]]: `IFD(q, a) = PPL(a|q) / PPL(a)`. [[cherry-llm]]에 따라 scoring model을 약 1K random samples에서 1 epoch warm up한다. `IFD < 1`을 유지하고, band 내에서 desc로 rank하며 top 10–15%를 keep한다.

```python
# ifd_filter.py
import math, torch, torch.nn.functional as F

@torch.no_grad()
def avg_nll(model, tok, context: str, target: str) -> float:
    # length-normalized sum_t -log p(target_t | context, target_<t)
    ctx = tok(context, return_tensors="pt").input_ids.to(model.device)
    tgt = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    ids = torch.cat([ctx, tgt], dim=1)
    lp = F.log_softmax(model(ids).logits[:, ctx.shape[1]-1:-1], dim=-1)
    return -lp.gather(-1, tgt.unsqueeze(-1)).mean().item()

def ifd_score(model, tok, q, a) -> float:
    return math.exp(avg_nll(model, tok, q, a) - avg_nll(model, tok, "", a))

def ifd_filter(samples, model, tok, keep_frac=0.15):
    scored = [(ifd_score(model, tok, s.instruction, s.output), s) for s in samples]
    kept = sorted([(sc, s) for sc, s in scored if sc < 1.0], key=lambda t: -t[0])
    return [s for _, s in kept[: int(keep_frac * len(samples))]]
```

Expected survival: [[cherry-llm]]의 "top 10% beats full set"에 따라 post-format pool의 약 10–15%.

### Filter 3 — MinHash dedup

[[deduplicating-training-data]] NearDup에서 온 5-gram shingles, MinHash signatures, LSH bands. Attested pretraining setup은 약 0.8 Jaccard에 대해 `b=20 × r=450`이지만, lab scale에서는 `datasketch` default `threshold=0.8`이면 충분하다. Evol-Instruct paraphrase가 빠져나가면 0.7로 tighten한다.

```python
# minhash_dedup.py — uses datasketch
from datasketch import MinHash, MinHashLSH

def shingles(text: str, k: int = 5) -> set[str]:
    toks = text.split()
    return {" ".join(toks[i:i+k]) for i in range(len(toks) - k + 1)}

def fingerprint(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text): m.update(sh.encode("utf-8"))
    return m

def minhash_dedup(samples, threshold: float = 0.8, num_perm: int = 128):
    lsh, kept = MinHashLSH(threshold=threshold, num_perm=num_perm), []
    for i, s in enumerate(samples):
        mh = fingerprint(s.instruction + " " + s.output, num_perm)
        if not lsh.query(mh):
            lsh.insert(str(i), mh); kept.append(s)
    return kept
```

Expected survival after dedup: post-IFD pool의 80–95%. >30% kill이면 generator가 collapse한 것이다. Self-Instruct/Evol-Instruct mix를 rebalance하거나 temperature를 올려라. `survival.json`을 `{"raw": 10000, "format": 7800, "ifd": 1170, "dedup": 1050, "verified": 990}`처럼 report한다.

---

## Stage 3: Verifier gate

Reasoning/tool-call subset에만 적용한다. Check할 수 없는 open-ended prompt를 verify하지 말라. [[apigen]]에 따라 cheapest→strictest 세 종류: exact-match(reasoning), sandboxed execution(tool/code, 5s timeout), LLM-judge Yes/No-with-reason.

```python
# verifier.py
import re
from typing import Callable

def verify_exact_match(gold: str, pred: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    m = re.search(r"(?:answer|=)\s*[:=]?\s*([^\n.]+)", pred, re.I)
    return bool(m) and norm(m.group(1)) == norm(gold)

def verify_execution(call: dict, ref_impl: Callable, timeout: float = 5.0) -> bool:
    try:
        with time_limit(timeout): ref_impl(**call["arguments"]); return True
    except (Exception, TimeoutError): return False

def verify_judge(client, q: str, a: str) -> bool:  # [[apigen]] Yes/No-with-reason
    v = client.complete(f"Query: {q}\nResponse: {a}\n\nDoes the response correctly "
                        "fulfill the query? Answer 'Yes' or 'No' with one sentence of "
                        "reasoning.", temperature=0).text.strip()
    return v.lower().startswith("yes")
```

[[apigen]] ablation: format 제거 –18% BFCL-V1, exec 제거 –11%, judge 제거 –6%. 모든 layer가 load-bearing이다. Judge rejection이 <3%라면 pool은 이미 clean한 것이다. Runtime에는 skip하더라도 ablation은 한 번 실행하라.

---

## Stage 4: Ablation training

Matched size `N`의 checkpoint 다섯 개. `N`을 고정하라. 그렇지 않으면 filter quality가 아니라 scale을 재는 것이다.

- `full` — 모든 filter on. Headline claim.
- `no-ifd` — format → dedup → verify. IFD를 isolate.
- `no-dedup` — format → IFD → verify. Dedup을 isolate.
- `no-verify` — format → IFD → dedup. Verifier를 isolate.
- `lima-matched` — raw에서 N random, LIMA-recipe proxy filter. [[lima]]를 이기거나 진다.

Resource-constrained: `full`, `lima-matched`, 그리고 당신에게 가장 새로운 ablation 두 개. SFT config: `lr=2e-5`, cosine + 10% warmup, `epochs=2`([[lima]]의 15-epoch는 1K hand-curated용이다. 약 1K–5K teacher-generated에서는 2 epochs가 [[deita]]와 맞는다). `pre_clip_grad_norm`과 step-1 loss를 log하라. `ln(V) ± 20%` 밖이면 misconfigured다. 실제 run 전에 고쳐라.

## Memo template

`synthetic-set-memo.md`, 한 페이지, 네 section:

1. **Survival table.** Stage별, split별(general / reasoning / tool) integer counts. 놀라웠던 stage에 대한 한 줄 comment.
2. **Quality spot-check.** `full`에서 20개 + `lima-matched`에서 20개. "would not ship" fraction을 count하고 각 dominant failure mode를 name.
3. **Headline ablation.** MT-Bench + AlpacaEval-2에서 `full` vs `lima-matched` win-rate, 그리고 세 filter delta. 각 항목에 "predicted" 또는 "surprising" tag.
4. **One failure mode.** 구체적이어야 한다. 예: "Evol-Instruct `complicate_input` produced nested JSON the tokenizer split into 600-token inputs, which the IFD scorer's 512-context model silently truncated, inflating IFD scores." Repro row ID 하나와 다음에 instrument할 것을 포함하라.

## Acceptance criteria

Hard gates, 순서대로.

1. `survival.json` integer count가 모든 stage에서 `instructions.jsonl` row count와 ±1 이내로 match.
2. `ifd_filter_check.py` histogram이 1 아래 중심과 작은 right tail을 보임. 1 위에 centered면 warm-up을 건너뛰었거나 cond/uncond tokenizer가 다르다([[ifd]] §Practical guidance).
3. `dedup_check.py`가 `no-dedup`에는 있고 `full`이 collapsed한 duplicate cluster를 보여 줌. 그렇지 않으면 threshold가 너무 loose하다.
4. `full`의 모든 reasoning/tool-call row가 `verifier_verdict` field를 carries.
5. 다섯 checkpoint 모두 step-1 loss가 `ln(V)`의 20% 이내이고, 100 steps 동안 `pre_clip_grad_norm` < 10. Karpathy의 SFT sanity.
6. `synthetic-set-memo.md`가 네 section을 가진 한 페이지이며, §4가 re-trigger할 만큼 specific한 failure mode를 name.
7. 적어도 하나의 ablation actual delta가 predicted delta와 달랐음. 그렇지 않으면 lab이 아무것도 가르치지 못한 것이다.

---

## Connections

- **ch-18** — 이 lab은 synthetic-data design pattern을 end-to-end로 instantiate한다. teacher → generator → filter cascade → verifier → consumer.
- **ch-19 / ch-20** — [[self-instruct]](breadth)와 [[evol-instruct]](depth)는 braided generators다.
- **ch-22** — [[lima]]의 "diversity > raw count"는 `lima-matched` ablation이 reject하려는 null hypothesis다.
- **ch-23 / ch-25 / ch-27** — [[cherry-llm]] / [[ifd]], [[minhash-lsh]] / [[deduplicating-training-data]], [[apigen]]은 filter + verifier primitives다.
- **ch-30** — `full` checkpoint를 single-turn synthetic SFT reference로 이어받는다.
- **Track 4 (RL)** — 여기의 exact-match / execution / judge verifiers는 RLVR + GRPO reward signal을 채우는 것과 같은 primitives다.

## Further reading

- [[self-instruct]] + [[evol-instruct]] — Stage 1에서 braided되는 breadth 및 depth generators.
- [[cherry-llm]] / [[ifd]] — `IFD = PPL(a|q)/PPL(a)`; warm-up recipe; top 10%.
- [[minhash-lsh]] / [[deduplicating-training-data]] — 5-gram shingles, LSH bands, 0.8 Jaccard.
- [[apigen]] — 모든 layer가 load-bearing임을 보여 주는 3-layer verifier ablation.
- [[lima]] — 1K hand-curated baseline; Superficial Alignment Hypothesis; ch-29가 reject하려는 null.
- [[alpagasus]] / [[deita]] — single-axis rubric vs three-axis decomposition; ch-29가 채택하지 않은 filter alternatives.
- [[karpathy-training-neural-net-recipe]] — predict-before-you-run ablation discipline.

## Companion visualization

**[figures/filter-cascade.html](figures/filter-cascade.html)** — self-contained horizontal Sankey: `5000 → N_format → N_IFD → N_dedup → N_verified`. Slider가 각 stage threshold를 조정한다. Box height는 survivor에 비례하고, hover하면 dominant rejection reason을 보여 준다. \$20 API bill을 commit하기 전에 cascade shape를 internalize하는 데 사용하라. IFD keep-fraction 0.15에서는 post-format pool의 85%가 그 stage 하나에서 죽는다.
