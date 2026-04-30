<!-- chapter: ch-29
     track: synthetic
     kind: lab
     title: Lab — Synthetic Instruction Set with Filter + Dedup + Verify
     deps: [ch-28]
     sources: [[self-instruct]], [[evol-instruct]], [[cherry-llm]], [[ifd]], [[minhash-lsh]], [[deduplicating-training-data]], [[apigen]], [[lima]], [[alpagasus]], [[deita]], [[karpathy-training-neural-net-recipe]]
     figures: figures/filter-cascade.html
     capstone_for: synthetic-track (ch-18..ch-29)
-->

# Chapter 29 — Lab: Synthetic Instruction Set with Filter + Dedup + Verify

> **Core insight.** A synthetic SFT pool is worthless until you can name the survival rate of every sample through every filter stage. The capstone of this track is not the 5K instructions you generate — it is the flow diagram `5000 raw → N_format → N_IFD → N_dedup → N_verified` with integers in every slot and a delta-win-rate against a matched-size [[lima]] subset to prove the pipeline earned its keep.
>
> **Guideline.** Generate with [[self-instruct]] bootstrap + [[evol-instruct]] depth operators. Cascade format-validity → IFD difficulty ([[cherry-llm]] / [[ifd]]) → MinHash dedup ([[minhash-lsh]] / [[deduplicating-training-data]]). Gate reasoning/tool-call subsets with a 3-layer verifier per [[apigen]] (format → exec → judge). Train ~1B SFT on your pool vs a matched-size [[lima]]-filtered subset; ablate each filter in isolation. Ship `synthetic-set-memo.md`. Ablation discipline per [[karpathy-training-neural-net-recipe]]: one change, one prediction, one outcome.

---

## Goal

Three artifacts, each reproducible by a peer from your repo:

1. **A pool.** `instructions.jsonl` with ~5K rows (full-budget) or ~1K (resource-constrained). Each row: origin operator (seed / evol-*), IFD score, MinHash band signature, and — on reasoning/tool-call subsets — a verifier verdict.
2. **A cascade report.** `survival.json` with integer counts per stage + `ablations.json` with win-rate deltas per removed stage.
3. **A memo.** `synthetic-set-memo.md`, one page: survival table, 20-sample spot-check, and one *specific* failure mode (not generic "the model hallucinates"). Predict each ablation's outcome before you run it per [[karpathy-training-neural-net-recipe]]; ablations matching predictions teach nothing, surprises are why the memo exists.

---

## Full-budget path

Target: 8 × H100 (or 8 × A100-40GB), ~1B base, teacher API ~$20, ~1 day wall clock.

- **Teacher.** `gpt-4o-mini` or `claude-3-5-haiku` for generation; `gpt-4o` for the LLM-judge. [[apigen]] used DeepSeek-Coder-V2 + GPT-4 — keep the judge strictly stronger than and separate from the generator so it cannot sign off on its own output.
- **Seed.** 175 hand-written tasks in the [[self-instruct]] style (reuse the public file); mix classification / open / extraction / reasoning / tool-call.
- **Generation budget.** ~10K raw so the cascade has headroom to prune to a ~5K target.
- **SFT model.** `meta-llama/Llama-3.2-1B` or `Qwen2.5-1.5B` — small enough for a ~30-minute run on 8 GPUs, large enough for benchmark deltas to clear noise.
- **Eval.** MT-Bench single-turn + AlpacaEval-2 length-controlled + one held-out reasoning probe + one tool-call probe. Report win-rate vs a matched-size [[lima]]-recipe-filtered subset of the same raw pool.

## Resource-constrained path

Target: 1 × GPU (≥16 GB), teacher API ~$3, ~4 hours wall clock.

- Generate ~2K raw; target ~1K survivors.
- SFT on `SmolLM-360M` or `Qwen2.5-0.5B` (~500M). Deltas shrink but cascade shapes are identical.
- Ablate only the two filters newest to you — if IFD is familiar, keep MinHash + verifier; if dedup is familiar, keep IFD + verifier. Don't budget-blow trying to cover all three.

---

## Stage 1: Generation

Two operators; neither generates the final pool alone.

### Self-Instruct bootstrap — the breadth engine

Attested [[self-instruct]]: 175 seeds, 8 in-context examples (6 seed + 2 prior accepted), classification-vs-non-classification branching for the instance format.

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

### Evol-Instruct depth — the complexity engine

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

Mix 70/30 Self-Instruct/Evol-Instruct. Skewing to evol collapses breadth; skewing to self-instruct flattens the complexity tail [[evol-instruct]] Figure 1 documents as the win condition.

---

## Stage 2: Filter cascade

Three gates, in order. Order matters: format is cheapest and kills most garbage; IFD is one forward pass per sample on a format-clean pool; MinHash benefits from the post-IFD reduction.

### Filter 1 — Format validity

Per [[self-instruct]] §Filtering: drop empty outputs, `input == output`, length < 3 or > 2K tokens, "image"/"graph"/"file" mentions (implied multimodal), ill-formed JSON in tool-call subset.

```python
def format_valid(s: Task) -> bool:
    if not s.instruction.strip() or not s.output.strip(): return False
    if s.input.strip() and s.input.strip() == s.output.strip(): return False
    if not 3 <= len(s.output.split()) <= 2000: return False
    if any(w in s.instruction.lower() for w in ("image","graph","file","picture")): return False
    return True
```

Expected survival: 70–85% (Self-Instruct reports ~67%; Evol-Instruct's own elimination is separate, higher-survival).

### Filter 2 — IFD difficulty

[[ifd]]: `IFD(q, a) = PPL(a|q) / PPL(a)`. Warm the scoring model 1 epoch on ~1K random samples per [[cherry-llm]]; keep `IFD < 1`, rank desc within band, top 10–15%.

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

Expected survival: ~10–15% of post-format pool per [[cherry-llm]] ("top 10% beats full set").

### Filter 3 — MinHash dedup

From [[deduplicating-training-data]] NearDup: 5-gram shingles, MinHash signatures, LSH bands. Attested pretraining setup is `b=20 × r=450` for ≈0.8 Jaccard; at lab scale `datasketch` default at `threshold=0.8` suffices. Tighten to 0.7 if Evol-Instruct paraphrases escape.

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

Expected survival after dedup: 80–95% of post-IFD pool. >30% kill = collapsed generator — rebalance the Self-Instruct/Evol-Instruct mix or raise temperature. Report `survival.json` as `{"raw": 10000, "format": 7800, "ifd": 1170, "dedup": 1050, "verified": 990}`.

---

## Stage 3: Verifier gate

Applies only to reasoning/tool-call subsets — don't verify open-ended prompts you cannot check. Three kinds cheapest→strictest per [[apigen]]: exact-match (reasoning), sandboxed execution (tool/code, 5s timeout), LLM-judge Yes/No-with-reason.

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

[[apigen]] ablation: removing format –18% BFCL-V1, exec –11%, judge –6% — every layer load-bearing. If your judge rejects <3%, your pool is already clean; skip at runtime but still run the ablation once.

---

## Stage 4: Ablation training

Five checkpoints at matched size `N` — hold `N` fixed or you are measuring scale, not filter quality.

- `full` — all filters on; headline claim.
- `no-ifd` — format → dedup → verify; isolates IFD.
- `no-dedup` — format → IFD → verify; isolates dedup.
- `no-verify` — format → IFD → dedup; isolates verifier.
- `lima-matched` — N random from raw, LIMA-recipe proxy filter; beats [[lima]] or loses to it.

Resource-constrained: `full`, `lima-matched`, and the two ablations newest to you. SFT config: `lr=2e-5`, cosine + 10% warmup, `epochs=2` ([[lima]]'s 15-epoch is for 1K hand-curated; at ~1K–5K teacher-generated, 2 epochs matches [[deita]]). Log `pre_clip_grad_norm` and step-1 loss; outside `ln(V) ± 20%` = misconfigured, fix before the real run.

## Memo template

`synthetic-set-memo.md`, one page, four sections:

1. **Survival table.** Integer counts per stage, per-split (general / reasoning / tool); one-line comment on the stage that surprised you.
2. **Quality spot-check.** 20 from `full` + 20 from `lima-matched`; count "would not ship" fractions; name the dominant failure mode in each.
3. **Headline ablation.** Win-rate `full` vs `lima-matched` on MT-Bench + AlpacaEval-2, plus the three filter deltas; tag each "predicted" or "surprising."
4. **One failure mode.** Specific, not generic: e.g. "Evol-Instruct `complicate_input` produced nested JSON the tokenizer split into 600-token inputs, which the IFD scorer's 512-context model silently truncated, inflating IFD scores." Include one repro row ID and what you would instrument next.

## Acceptance criteria

Hard gates, in order.

1. `survival.json` integer counts match `instructions.jsonl` row counts within ±1 at every stage.
2. `ifd_filter_check.py` histogram centers below 1 with a small right tail. Centered above 1 = warm-up skipped or cond/uncond tokenizers differ ([[ifd]] §Practical guidance).
3. `dedup_check.py` shows `no-dedup` contains a duplicate cluster `full` collapsed. If not, threshold is too loose.
4. Every reasoning/tool-call row in `full` carries a `verifier_verdict` field.
5. All five checkpoints reach step-1 loss within 20% of `ln(V)` and `pre_clip_grad_norm` < 10 through 100 steps — Karpathy's SFT sanity.
6. `synthetic-set-memo.md` is one page with four sections; §4 names a failure mode specific enough to re-trigger.
7. At least one ablation's actual delta disagreed with your predicted delta — otherwise the lab taught you nothing.

---

## Connections

- **ch-18** — this lab instantiates the synthetic-data design pattern end-to-end: teacher → generator → filter cascade → verifier → consumer.
- **ch-19 / ch-20** — [[self-instruct]] (breadth) and [[evol-instruct]] (depth) are the braided generators.
- **ch-22** — [[lima]]'s "diversity > raw count" is the null hypothesis the `lima-matched` ablation tries to reject.
- **ch-23 / ch-25 / ch-27** — [[cherry-llm]] / [[ifd]], [[minhash-lsh]] / [[deduplicating-training-data]], and [[apigen]] are the filter + verifier primitives.
- **ch-30** — takes the `full` checkpoint forward as the single-turn synthetic SFT reference.
- **Track 4 (RL)** — exact-match / execution / judge verifiers here are the same primitives that populate RLVR + GRPO reward signals.

## Further reading

- [[self-instruct]] + [[evol-instruct]] — the breadth and depth generators braided in Stage 1.
- [[cherry-llm]] / [[ifd]] — `IFD = PPL(a|q)/PPL(a)`; warm-up recipe; top 10%.
- [[minhash-lsh]] / [[deduplicating-training-data]] — 5-gram shingles, LSH bands, 0.8 Jaccard.
- [[apigen]] — 3-layer verifier ablation showing every layer is load-bearing.
- [[lima]] — 1K hand-curated baseline; Superficial Alignment Hypothesis; the null ch-29 tries to reject.
- [[alpagasus]] / [[deita]] — single-axis rubric vs three-axis decomposition; the filter alternatives ch-29 declined.
- [[karpathy-training-neural-net-recipe]] — predict-before-you-run ablation discipline.

## Companion visualization

**[figures/filter-cascade.html](figures/filter-cascade.html)** — self-contained horizontal Sankey: `5000 → N_format → N_IFD → N_dedup → N_verified`. Sliders tune each stage's threshold; box heights are proportional to survivors; hover shows the dominant rejection reason. Use it to internalize the cascade shape *before* committing a $20 API bill — at IFD keep-fraction 0.15, 85% of the post-format pool dies at that one stage.
