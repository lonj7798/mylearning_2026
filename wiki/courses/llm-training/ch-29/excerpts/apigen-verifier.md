---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/apigen.md
source_url: https://arxiv.org/abs/2406.18518
created_at: "2026-04-23"
---

# Excerpt: APIGen — the 3-layer verifier ch-29's Stage 3 implements

**Source library:** `wiki/raw-data/llm-training/papers/apigen.md`
**Artifact:** Format → Execution → Semantic-judge 3-layer gate

---

## Why this source anchors ch-29

Ch-29's IFD and MinHash filters are *quality proxies* — they measure whether a sample is informative and non-duplicate. Neither touches correctness. On the reasoning and tool-call subsets this is disqualifying: IFD will happily rank a confidently-wrong math answer highly. APIGen's contribution to ch-29 is the recipe for checking correctness cheaply enough to run on every candidate.

The APIGen paper's headline claim (source line 48) is that all three verifier layers are load-bearing: removing semantic judge costs –6% on BFCL-V1, removing execution costs –11%, removing format costs –18%. Ch-29's verifier ablation is the direct test of this claim on *your* pool.

---

## The 3 layers, stated exactly

From the source (lines 29–33):

1. **Format check** — JSON must parse, fields must match function schema, types enforced (`int / str / bool / enum / list`).
2. **Execution check** — run the call against reference implementations in a **Python sandbox with 5-second timeout**; must not raise.
3. **Semantic check** — **LLM-as-judge (GPT-4)** shown `(query, call, execution result)` must answer "Yes" to "does the call correctly fulfill the query?"

Ch-29's `verify_exact_match`, `verify_execution`, `verify_judge` in Stage 3 map 1:1 to these layers. The exact-match variant is added for non-tool reasoning where no execution is possible — same contract (reject on mismatch), different mechanism.

---

## The rejection rate APIGen reports

From the source (line 48): *"the 3-layer filter rejects ~40% of raw generations."*

This is the number ch-29's `no-verify` ablation is designed to isolate. If your verifier rejects far less than 40%, one of:

- Your teacher is already very good (GPT-4o-mini rejected 12% in internal APIGen testing vs 40% for a weaker model).
- Your reasoning/tool-call subset is too small or too easy.
- Your semantic judge is too permissive — swap to a stricter prompt.

If it rejects far more than 40%, the generator is mis-aligned with the execution environment or the judge prompt is pathological. Either way, the memo's §4 failure mode should name which of the three layers dominated the rejections.

---

## The judge contract — why "Yes" with reasoning

From the source (line 47):

> Semantic: GPT-4 judge prompt requires "Yes" / "No" verdict with reasoning; only "Yes" accepted.

The "with reasoning" is not decoration — it is there to anchor the judge against length-bias and sycophancy. Ch-29's `verify_judge` preserves the same prompt shape. A judge asked for `Yes/No` alone reliably drifts toward `Yes` over a long session; requiring one-sentence reasoning mitigates this.

Temperature 0 is attested (`temperature=0` in the ch-29 code) because the judge is a classifier, not a generator.

---

## The executable-API requirement — why ch-29 scopes carefully

From the source (line 57):

> Executable-API requirement limits scale — the pipeline is bottlenecked on having reference implementations.

APIGen curated 3,673 APIs with real or mock reference implementations. Ch-29 cannot — that is a multi-week effort. The lab instead scopes the tool-call subset to **≤ 20 functions with hand-written Python mock implementations** (e.g. `get_weather(city, date)`, `send_email(to, subject, body)`, `schedule_meeting(time, attendees)`). This is enough to exercise the 3-layer verifier and produce a non-trivial ablation delta.

The reasoning subset is larger (no execution environment needed) — exact-match on GSM8K-style prompts with a final-number convention.

---

## The hallucination-rate claim — what success looks like

From the source (line 48): post-filter hallucination rate on BFCL-V1 is <3% for xLAM-7B vs ~15% for ToolLLaMA (which uses no execution verification).

A lab-scale equivalent: hand-review 20 tool-call samples from your `full` run and 20 from `no-verify`. Count the "would actually misbehave if called" fraction. If `full` is below 5% and `no-verify` is above 15%, the verifier is doing the work the paper claims.

---

## What ch-29 keeps, changes, drops

| APIGen default | Ch-29 choice | Reason |
|----------------|--------------|--------|
| 3,673 executable APIs | ~20 hand-written mocks | lab-scale resource constraint |
| DeepSeek-Coder-V2 / GPT-4 generator | `gpt-4o-mini` / `claude-3-5-haiku` | cost |
| GPT-4 judge | `gpt-4o` | stronger than generator; attested separation |
| 5-second execution timeout | Same | attested; same sandbox contract |
| Single-turn only | Same | multi-turn is [[apigen-mt]], out of scope |
| MinHash on `(query, call)` | MinHash on `(instruction, output)` | ch-29's existing dedup stage already covers this |
| All three layers | Same | attested load-bearing by ablation |

---

## Connections

- **ch-27** — the full-read chapter on [[apigen]] + the function-calling-data line.
- **ch-41 / ch-42 (tool-use)** — the downstream chapters that consume ch-29's tool-call subset as SFT seed for tool-calling RL.
- **ch-29 §4** — the `no-verify` ablation is the clearest test of whether the verifier earns its API bill; without it, the Stage 3 section is unfalsifiable.
- **Track 4 (RL)** — the same three verifier primitives populate the reward function for RLVR (rule-based RL) and the verifiable reward in GRPO-style training.
