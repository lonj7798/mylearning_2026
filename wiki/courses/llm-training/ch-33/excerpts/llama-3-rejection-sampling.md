---
chapter: ch-33
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/llama-3-synthetic-pipeline.md
source_url: https://ai.meta.com/blog/meta-llama-3/
created_at: "2026-04-23"
---

# Excerpt: Llama 3's public synthetic pipeline — what Meta actually disclosed

**Source library:** `wiki/raw-data/llm-training/blogs/llama-3-synthetic-pipeline.md`
**Artifact:** Meta's blog + `synthetic-data-kit` CLI disclosure

---

## Why this source anchors ch-33

The llama-3 model report ([[llama-3]]) gives the loop *structure*; this blog gives the *philosophy* behind the loop — "prompt quality and preference-ranking quality have outsized impact", "multiple rounds of QA on annotations before training", and (critically) "the exact synthetic-data mix is NOT published". Ch-33's §3.3 ("Why multi-round beats single-pass") leans on this excerpt for the reasoning thread, and ch-33's §3.2 and the HTML figure's caveat both depend on this source to flag what Meta did not disclose.

---

## The attested philosophy

From the source (lines 7, 15):

> The public Llama 3 recipe is a loop, not a single dataset. Meta repeatedly reuses the latest best checkpoint to regenerate SFT and preference data, and the quality of prompts plus preference rankings is treated as a first-order training variable.

> Meta's public Llama 3 materials disclose a post-training stack built around supervised fine-tuning, rejection sampling, PPO, and DPO. The useful public lesson is not the hidden prompt library, which Meta does not publish, but the staging: generate candidates, reject weak ones, rank preferences with QA, then feed the newest aligned checkpoint back into the next round.

So the "loop" is the contribution. The specific prompt corpus is private; the *architecture of the curation process* is public. Ch-33's §3 takes this philosophy seriously and treats each round's role as the design artifact to understand.

---

## The specific disclosed moves

From the source (lines 26–32):

- **Pretraining context:** new mix of publicly available online data, more than 15T tokens, more code than Llama 2.
- **SFT data:** tuned models use SFT; prompt quality is critical.
- **Rejection sampling:** post-training loop includes rejection sampling; the practical role is to keep the best sampled outputs and feed them back as training data.
- **Preference data:** PPO and DPO rely on preference rankings; Meta applied multiple rounds of QA to human annotations; *no separate synthetic-preference dataset is publicly described*.
- **Synthetic task data:** used heavily across coding, math, multilinguality, reasoning, long context, tool use, and factuality, but the *exact per-domain mixture is not disclosed*.
- **Factuality data:** pipeline asks Llama 3 to generate factual questions from pretraining snippets, scores answers for correctness and informativeness, turns *consistently-wrong-but-informative* responses into refusals.
- **Code-related data:** blog and model card show strong code gains, but the exact synthetic code corpus is not publicly specified.

The factuality pipeline is the most interesting disclosed mini-pipeline: it explicitly engineers refusals out of the model's own consistent wrongness rather than out of rule-based policy. Ch-33's HTML figure places this in round 3 based on the blog's narrative thread.

---

## What is NOT disclosed — the caveat ch-33 depends on

From the source (line 39):

> Disclosure gap: the exact synthetic-data mix and prompt templates are not public, so faithful reproduction is limited.

From the source (line 42):

> Preference noise: if ranking quality drops, the whole loop degrades quickly.

These two sentences are why ch-33 §3.2 labels every per-round capability attribution as *inferred*, and why the HTML figure has a standing caveat at the bottom of the Llama 3 column. The point is not to invent round-by-round deltas — Meta did not publish them — but to reason carefully from the prose about what each round was *likely* doing, given the order in which capabilities appear in the blog's narrative.

---

## What ch-33 keeps from this source

- The "loop is the contribution" framing (§3 intro and §3.3).
- The factuality-pipeline specifics (HTML figure, round 3).
- The disclosure gap (explicit caveat in §3.2 table and in HTML figure's bottom caveat).
- The "prompt quality and preference ranking have outsized impact" claim (§3.3 reason 3).

---

## Connections

- **ch-33 §3** — where this excerpt is cited.
- **[[llama-3]]** — the tech report; this blog is the accessible companion.
- **[[llama-2]]** — the predecessor loop (PPO-based).
- **[[tulu-3]]** — contrast: full disclosure vs partial disclosure.
- **[[open-thoughts]]** / **[[hf-cosmopedia]]** — useful contrast: these *do* publish the corpus recipe, Meta does not.
- **[[magpie]]** — related self-bootstrapping synthetic generator.
