---
chapter: ch-52
course: llm-training
phase: read
excerpt_of: Mazeika et al. 2024 — "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal"
source_url: https://arxiv.org/abs/2402.04249
created_at: "2026-04-23"
---

# Excerpt: HarmBench — Behavior Library + Attack-Wrapper Separation

**Source:** `wiki/raw-data/llm-training/papers/harmbench-data.md`
**Primary paper:** Mantas Mazeika et al., "HarmBench", 2024
**arXiv:** https://arxiv.org/abs/2402.04249
**Repo:** https://github.com/centerforaisafety/HarmBench

---

## Why HarmBench is the default adversarial-robustness benchmark

Pre-HarmBench, a "safety benchmark" was usually a flat list of harmful prompts plus a substring scorer. HarmBench replaced that with a three-layer design that every subsequent safety benchmark has adopted in some form: behavior library, attack wrappers, and held-out judge. The raw-data source is explicit:

> *"HarmBench separates: the **behavior label** from the **attack wrapper**. That separation is extremely useful if you want to synthesize refusal data, adversarial SFT data, or red-team replay buffers without entangling target behavior design with one particular jailbreak style."*

For ch-52's taxonomy-vs-attack-vs-judge framing, HarmBench is the cleanest public example of all three.

---

## The behavior library

7 semantic categories + 4 functional categories. The semantic categories:

- `Cybercrime & Unauthorized Intrusion`
- `Chemical & Biological Weapons / Drugs`
- `Copyright Violations`
- `Misinformation & Disinformation`
- `Harassment & Bullying`
- `Illegal Activities`
- `General Harm`

The functional categories (operationally more important):

- `standard` — self-contained harmful request.
- `copyright` — asks for copyrighted text; needs a different scorer.
- `contextual` — context string + harmful request tied to the context.
- `multimodal` — request paired with an image.

The behavior inventory is **human-authored but policy-informed**: authors distilled acceptable-use policies from OpenAI, Anthropic, Meta, and Inflection AI via GPT-4, then manually designed behaviors that clearly violated the distilled summary. This is the key move — the behaviors are not GPT-4-generated unsafe prompts; they are human targets informed by a synthetic policy distillation.

Two curation rules that matter for ch-52's "is this benchmark meaningful" question:

- **Differential harm / low searchability.** Prefer behaviors where an LLM adds capability beyond what a user could trivially Google. This is what keeps the benchmark from collapsing into "refuse anything sensitive" over time.
- **Dual-intent filtering.** Remove or rewrite behaviors where many benign users would plausibly want the same capability. This is the single most important lesson for any safety-data work: naive harmful-prompt lists routinely mix clearly malicious requests with normal expert use, and refusal training on such mixes produces over-refusal.

---

## The attack wrappers

HarmBench uses ~18 attack families, organized by construction style:

- **Direct Request.** Behavior string with no wrapper. Measures base refusal.
- **Human Jailbreaks.** In-the-wild human-authored templates (DAN-style).
- **Token-optimization.** `GCG`, `GCG-Multi`, `GCG-Transfer`, `PEZ`, `GBDA`, `UAT`, `AutoPrompt` — learn adversarial suffixes or token sequences.
- **Attacker-LLM generation.** `Zero-Shot`, `Stochastic Few-Shot`, `PAIR`, `TAP`, `TAP-Transfer` — use an attacker model to iteratively propose jailbreaks.
- **Evolution / persuasion.** `AutoDAN` evolves handcrafted jailbreaks; `PAP` rewrites requests using persuasive strategies.

The key operational insight: these attack families are orthogonal to the behavior library. You can generate new training data for any defense by pairing any behavior with any attack family. That is why HarmBench is also a data-synthesis pipeline, not only an eval.

---

## The judge

For non-copyright behaviors, HarmBench fine-tunes **Llama-2-13B-Chat** on a manually labeled validation set of completions. For copyright behaviors, it uses **MinHash-style matching** over overlapping chunks. The asymmetry is deliberate — for copyright, "attempted" reproduction is not sufficient evidence of violation; the protected content must actually appear.

The classifier is stress-tested on tricky cases:
- Benign paragraphs labeled harmful (false positive check).
- Unrelated harmful completions labeled as the target behavior (substring-match defense).
- Completions that start with a refusal and then comply (refusal-prefix defense).

This last case is important for ch-52: if the judge can be gamed by a visible refusal prefix that precedes compliance, every number downstream is corrupted.

---

## Validation / test split

HarmBench enforces an official val/test partition of behaviors so attacks and defenses do not optimize directly on the final eval target. This is the standard ML discipline applied to safety benchmarks — rare in the pre-2024 safety benchmark literature.

The operational rule for ch-52: if you train refusal SFT on HarmBench, train only on `val`; report on `test`. Mixing the splits is the safety-data equivalent of leakage.

---

## Connections

- [[wildguard-data]] — complement focused on moderation labels over prompt-response pairs.
- [[salad-bench]] — hierarchical counterpart with finer leaf taxonomy.
- [[circuit-breakers-data]] — consumes HarmBench as seed data for the harmful completion set.
- Chapter synthesis: [[ch-52]] §1, §2, §9.
