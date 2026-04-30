---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Kuratov et al. — "BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack"
source_url: https://arxiv.org/abs/2406.10149
created_at: "2026-04-23"
---

# Excerpt: BABILong — reasoning-in-a-haystack as a scalable generator

**Source:** `wiki/raw-data/llm-training/papers/babilong.md`
**Paper:** Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, Mikhail Burtsev, 2024
**arXiv:** https://arxiv.org/abs/2406.10149

---

## Bibliographic header

> *"BABILong turns short synthetic reasoning problems into arbitrarily long-context tasks by embedding bAbI facts inside real book text, showing that long-context retrieval is much easier than long-context reasoning."*

BABILong's contribution is not another NIAH variant — it is a *hybrid synthetic-natural* construction that lets the reasoning task stay constant while the context grows to 1M / 10M / 50M tokens. That construction is the durable piece.

---

## The hybrid synthesis recipe

From the raw-data:

> *"The reasoning problem is synthetic and templated. BABILong inherits the bAbI tasks, which are generated from simulated micro-world interactions like movements, object transfers, counting events, temporal ordering, and simple deduction. The background context is natural text. The paper uses PG19 books."*

Each final example is:

```
[PG19 prose sentences]
[bAbI task fact 1 from the selected bAbI sample]
[more PG19 sentences]
[bAbI task fact 2]
...
[more PG19 sentences]
[bAbI question]
```

with the question optionally placed at the start or the end. Both the task facts and the PG19 background are **unprocessed** — the task sentences remain distinguishable from book prose as "style contrast," which is itself an interesting property (see §"limitations" below).

---

## The length-scaling protocol

> *"BABILong is designed to be extendable to arbitrary lengths because the background text can continue to grow without changing the underlying task. Public benchmark configurations include lengths such as 0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1M, and 10M. The paper positions 10M tokens as the largest predefined split and reports some model evaluations up to 50M tokens."*

The key word is **extendable**. The background text can grow indefinitely; only the bAbI core stays fixed. A lab evaluating a 2030-era 100M-context model can extend BABILong to 100M without re-designing the benchmark. Contrast with LongBench, whose corpora are fixed at their 6K–31K lengths.

The 50M evaluations in the paper are for recurrent-memory systems; transformer attention hasn't needed that regime yet.

---

## The 20-task taxonomy — why this is harder than NIAH

bAbI inherits 20 reasoning tasks:

- single supporting fact
- two supporting facts
- three supporting facts
- yes/no questions
- counting
- lists/sets
- simple negation
- indefinite knowledge
- basic coreference
- conjunction
- compound coreference
- time reasoning
- basic deduction
- basic induction
- positional reasoning
- size reasoning
- path finding
- agent motivations
- simple relations
- reasoning by exclusion

The hardness gradient from "single supporting fact" (essentially NIAH) to "three supporting facts" (multi-hop over spans) to "counting" (aggregation) to "path finding" (graph reasoning) is what lets BABILong separately expose retrieval and reasoning failures at each context length.

The paper's headline finding from Table 2: **many long-context models retrieve correctly but still fail the downstream reasoning step.** That is the core "reasoning-in-a-haystack" result.

---

## Generation-protocol advantages

> *"The benchmark is attractive for eval design because it is mostly free of standard human-labeling bottlenecks: answer labels come directly from the bAbI task definition; no LLM judge is needed for grading; no human annotation pass is required for each new length."*

Three consequences:

1. **Deterministic grading** — bAbI answers are short, exact-match-friendly strings ("yes" / "no" / "kitchen" / a number). No LLM judge, no API costs, no judge-bias.
2. **Contamination resistance** — the template is synthesised fresh; the facts don't appear verbatim in training data (PG19 is public but the inserted bAbI sentences are not).
3. **Cheap length extension** — you can sample more PG19 text for free; no new annotations required.

This is what makes BABILong a sustainable *training signal*, not just an eval harness. Generators that can grow with no additional annotation cost are the right shape for long-context training data in future frontier regimes.

---

## Contrast with RULER

> *"Complements RULER: RULER broadens synthetic long-context evaluation across retrieval, tracing, aggregation, and QA, while BABILong goes deeper on symbolic reasoning over inserted facts."*

The split:

| Dimension | RULER | BABILong |
|---|---|---|
| Breadth | 13 tasks across retrieval / tracing / aggregation / QA | 20 reasoning tasks |
| Depth | shallow reasoning per task | deep symbolic reasoning |
| Background | essay / noise | real book prose |
| Max length | 128K / 256K main | 10M public, 50M paper |
| Grading | recall-match | exact-match |

Lab practice: run both. RULER gives the task-breadth view; BABILong gives the reasoning-depth view at extreme length.

---

## A limitation worth naming

> *"The short bAbI sentences remain distinguishable from book prose, which helps scalability, but it also means the setup is cleaner than real document reasoning."*

The style contrast between bAbI ("John went to the kitchen.") and PG19 book prose makes the bAbI sentences easier to *detect* as "the important ones" than genuine domain-specific facts buried inside a coherent document. A perfectly-adapted attention pattern on BABILong may still fail on real long-document reasoning where the relevant sentences aren't stylistically marked.

This is the reason LongBench-Chat exists alongside BABILong: realistic queries balance the synthetic contamination-resistance.

---

## Connections

- Chapter synthesis: [[ch-28]]
- Broader task-family counterpart: [[excerpts/ruler-task-family]]
- Retrieval ancestor: [[excerpts/niah-kamradt-original]]
- Used in ProLong's eval mix: [[excerpts/prolong-coherence]]
