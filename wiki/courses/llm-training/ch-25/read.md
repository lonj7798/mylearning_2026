<!-- chapter: ch-25
     track: synthetic
     title: Modality — Multi-Turn Conversation Synthesis
     sources: [[baize]], [[baize-construction]], [[ultrachat-construction]], [[ultrachat-pipeline]], [[camel]], [[soda]], [[prosocial-dialog]], [[openassistant]], [[system-prompt-diversity]], [[wildchat]], [[capybara]], [[smol-talk]]
     figures: figures/turn-distribution.html
-->

# Chapter 25 — Modality: Multi-Turn Conversation Synthesis

> **Core insight.** Conversation is the first data modality where the ground-truth distribution you want at inference — a real user talking to a real assistant across several turns — is *not* what any single LLM produces by default. Every multi-turn synthesis pipeline is an engineered workaround for that gap. The workarounds sort into three families by how they manufacture the second speaker: **self-chat** (one model plays both sides; Baize), **two-model role-play** (separate user-LLM + assistant-LLM under a sector or role prompt; UltraChat, CAMEL), and **grounded narration** (seed the scene from a commonsense triple, rule-of-thumb, or real passage; SODA, Prosocial-Dialog, UltraChat-Sector-3). **OpenAssistant** is the real-human baseline against which each synthetic family is measured — its turn-count tail, user-register variance, and branching factor are the metrics synthetic pipelines still miss.
>
> **Guideline.** For multi-turn SFT data today, do not pick one recipe; compose a mix with explicit slices — self-chat for cheap breadth, two-model role-play under a taxonomy for topical coverage, role-pair inception prompting for task-oriented dialogue, commonsense-grounded dialogues for emotional register, and an OASST/WildChat anchor slice for realism. Size the taxonomy before running any generation; the taxonomy *is* the diversity lever, the teacher is only the renderer.

---

## Why this chapter exists

Ch-24 handed you the generic synthetic-pipeline loop (generate → filter → dedup → verify → select → mix) on single-turn instruction data. Conversation breaks one assumption of that loop: the unit of data is not a `(prompt, response)` pair but a **tree of alternating turns** whose statistics — turn count, per-turn token length, topic drift, persona consistency — each need independent design attention. A naive K-turn extension of Self-Instruct produces dialogues that fine-tune visibly synthetic assistants: unrealistically articulate users, absent emotional register, formulaic turn-taking.

Three properties of conversation drive the design:

1. **The user turn is harder than the assistant turn.** Assistant style is what the teacher LLM was aligned to produce; user style is not. Every pipeline here is partly a workaround for "how do I get a believable user turn?"
2. **Multi-turn has combinatorial blow-up but only one degree of coherence.** Most turn orderings are nonsense; synthesis must constrain topology.
3. **Topic taxonomy, role pair, grounding source, and system prompt are each independent diversity axes.** Modern mixes stack all four.

---

## 1. Baize — the self-chat primitive

Spring 2023. [[baize-construction]] had no GPT-4 budget and no human dialogue logs, so they prompted one LLM to role-play both sides in a single API call. The exact template:

```
The following is a conversation between a human and an AI assistant.
[|Human|] <seed question>
[|AI|] <first ChatGPT response>
[|Human|] <ChatGPT playing user>
[|AI|] <ChatGPT playing AI>
...
```

ChatGPT continues both sides until a termination marker or ~8-turn cap. Post-hoc parsing splits on role markers. Seed pools: Quora ~54K, StackOverflow ~57K, Alpaca ~52K, MedQuAD ~47K → 111.5K dialogues total. Median 4 turns, IQR 3–6, tail to 10. Mean ~100 tokens per turn. One API call per dialogue; total cost ~\$1,000.

The load-bearing weakness is **the user-realism gap.** Single-model self-chat produces users that are "a curious articulate human" — real users are abrupt, typo-ridden, and switch topics. A student SFT'd on pure Baize learns to expect articulate users and degrades when served real ones. The Self-Distill with Feedback extension (re-prompt ChatGPT to critique the student, use critiques as DPO pairs) partially compensates but does not fix the user distribution. What Baize *did* prove: multi-turn capability transfers from synthetic self-chat at all — prior belief was that you needed real human dialogues. Baize-13B beat Alpaca-13B on 58% of human-eval comparisons, establishing self-chat as the minimum-viable baseline every later pipeline had to beat.

---

## 2. UltraChat — two-model role-play under a topic taxonomy

[[ultrachat-construction]] replaces Baize's single call with **two separate ChatGPT-Turbo calls per turn** (one user-prompt, one assistant-prompt) and replaces seed pools with a **hand-designed three-sector taxonomy**. The taxonomy is the diversity lever; the two-model split gives conversational friction.

**Sector 1 — Questions about the World.** 30 meta-topics → 1,100+ subtopics → up to 10 seed questions each, Turbo-expanded. A second branch: top 10K Wikidata entities × 5 meta-questions × (10 specifics + 20 related). Each seed rolls into a 3–7-round dialogue.

**Sector 2 — Writing and Creation.** 20 writing types (essay, poem, script, email, code, recipe, …) × 200 instructions per type; 80% expanded with details. Each becomes a 2–4-round draft-and-revise dialogue.

**Sector 3 — Assistance on Existing Materials.** ~100K C4 passages; up to 5 user questions each; passage + question + template is the first user turn. 2–4 rounds of grounded follow-up.

The two-model loop, schematically:

```
for round in range(R_s):                       # R_1 ∈ [3,7], R_{2,3} ∈ [2,4]
    assistant_turn = assistant_LLM(system=assistant_prompt_s, history)
    history.append(assistant_turn)
    user_turn = user_LLM(system=user_prompt_s, history)
    history.append(user_turn)
```

Each turn conditions on full history. Two separate calls force each side to *react* to actually-emitted tokens — single-call self-chat lets the model plan both sides too coherently. Output: ~1.5M dialogues (Q-world ~600K, Writing ~400K, Assistance ~500K), HuggingFace preview list lengths 4–14, consistent with the sector round bounds. What UltraChat established as standard: taxonomy-first diversity (enumerate before generation; nothing from a generic prompt), separated user/assistant generators (every later pipeline copies this), and split-by-family release so downstream mixes can weight sectors independently ([[smol-talk]] uses only a subset). The filter stage is opaque — the lesson for practitioners is that generation protocol is more transferable than filter heuristics.

---

## 3. CAMEL — role-play multi-agent synthesis with Inception Prompting

[[camel]] asks a different question: what if *roles* are the diversity lever, not topics? Sample from a 50 × 50 × 20 grid of `(assistant_role, user_role, domain)`. The unlock is **Inception Prompting** — the AI User's system prompt, verbatim from the paper and the reference implementation:

```
Never forget you are a <user-role> and I am a <assistant-role>.
Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete the task: <task>.
You must instruct me based on my expertise and your needs to solve the task.

Give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
You should instruct me, not ask me questions.

Here is the format you must strictly follow:
Instruction: <YOUR INSTRUCTION>
Input:       <YOUR INPUT, or "None">

Do not add anything else other than your instruction and the optional input.

When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task.
```

The AI Assistant prompt mirrors this: "I must provide a solution in the format `Solution: <YOUR SOLUTION>` and must not flip to instructing you." Two load-bearing properties: the **role lock** ("Never flip roles! Never instruct me!") is the first-line constraint because GPT-3.5 otherwise defaults to the assistant's natural impulse to ask clarifying questions; and the **termination protocol** — `<CAMEL_TASK_DONE>` is a single-token marker, emitted by the user only when the task is actually solved. This is **semantic termination, not a turn cap**. Dialogues run 4 turns if trivial, 20 if hard.

The 50 assistant roles (Accountant, Architect, Astronaut, Biologist, …) and 50 user roles (Entrepreneur, Graduate Student, Journalist, …) were themselves generated once by prompting GPT-3.5 to enumerate common roles. Crossed with 20 task domains per variant (Society, Code, Math, Science). Statistics: median 6–8 turns, tail to 20, avg ~50 tokens per turn (short because of the rigid `Instruction:` / `Solution:` format). AI Society ~1M dialogues; cost ~\$5–10K at 2023 Turbo rates. The Inception-prompting pattern propagates forward into [[agentinstruct]], [[apigen-mt]], ToolACE — wherever you see two LLMs with explicit role prompts and a termination marker, the lineage runs back here. Gotchas: 50×50 is not 2,500 genuinely different dialogues (many role pairs are low-quality — Astronaut × Bartender is the canonical example); and "As an accountant, I'd suggest…" is a formulaic leak that downstream models must unlearn.

---

## 4. SODA and Prosocial-Dialog — grounded synthesis (Yejin Choi group)

Topic (UltraChat) and role (CAMEL) condition on *what* and *who*; **grounding** conditions on *why* — the scene, motive, or norm that makes dialogue make emotional sense.

**SODA** ([[soda]]) starts from Atomic 10X, a commonsense-triple database of ~10M records shaped `(PersonX event, relation, mental-state-or-reason)` — e.g. `(PersonX fails the exam, xWant, to study harder)`, `(PersonX moves abroad, xEffect, feels lonely)`, `(PersonX surprises PersonY, xReact, feels proud)`. Three steps:

```
Step 1 — triple sampling
  Sample 1.5M triples, balanced across {xWant, xNeed, xEffect, xReact}.

Step 2 — narrative generation
  Prompt GPT-3.5:
    "Given the triple (PersonX <event>, <relation>, <state>),
     write a 1-2 sentence narrative describing a social scene where
     this arises. Include a second person PersonY as interlocutor."

Step 3 — dialogue generation
  Prompt GPT-3.5:
    "Write a short conversation between PersonX and PersonY set in
     this scene. Each speaker alternates. 4 to 10 turns total."
```

Filter: scene-coherence, length ≥ 3 turns, toxicity classifier, persona-consistency (PersonX stays PersonX). Output: 1.5M dialogues, median 7–8 turns, max ~12, avg 20 tokens per turn (very short — chatty social exchanges, not factual Q&A). Cost ~\$10K. Without the triple, GPT-3.5 produces plausible but *emotionally flat* conversations; anchoring injects an implicit emotional state (`feels lonely`, `wants X because Y`) that drives non-trivial turn content. COSMO-3B trained on SODA beats BlenderBot-3B on natural / engaging / specific human-eval dimensions at a third the parameters.

**Prosocial-Dialog** ([[prosocial-dialog]]) applies grounding to *safety*. The claim: refusing difficult prompts is neither safe nor helpful; engaging constructively with a socially-grounded rule-of-thumb is. Semi-synthetic, human-in-the-loop: crowd workers author ~10K problematic prompts across 10 harm categories (stereotypes, insults, self-harm, violence planning, misinformation, …); workers identify applicable rules-of-thumb — short ethical guidelines like *"It's rude to mock someone's appearance."* 300+ unique RoTs emerge across the dataset. GPT-3 drafts candidate assistant responses grounded in the RoT; workers edit and rank; high-quality retained; multi-turn extension via back-and-forth editing. 58K dialogues, median 3 turns, each turn tagged with its RoT. CANARY-400M engages constructively with 89% of problematic prompts vs. BlenderBot-3B's 32%. This is an engage-and-redirect pattern, not refusal — a direct precursor to [[constitutional-ai]].

The Yejin Choi lineage `self-instruct → SODA → Prosocial-Dialog` is a single arc: bootstrap from model outputs, but always *ground* the bootstrap in a structured external signal (commonsense triple, norm). Pure model-output bootstrap distills the teacher's style; grounded bootstrap injects information the teacher would not have produced on its own.

---

## 5. OpenAssistant — the real-human baseline

[[openassistant]] is not a synthesis pipeline. It is the dataset every pipeline in this chapter is measured against, and the one artifact in the space with a fundamentally different data structure: a **conversation tree**.

```
root (user prompt)
 ├── assistant_reply_A      [quality 4.2]
 │    ├── user_followup_A1
 │    │    ├── assistant_reply_A1a  [quality 3.8]
 │    │    └── assistant_reply_A1b  [quality 4.1]
 │    └── user_followup_A2
 │         └── assistant_reply_A2a  [quality 3.6]
 └── assistant_reply_B      [quality 3.4]
      └── user_followup_B1
           └── assistant_reply_B1a  [quality 3.0]
```

Each node has a parent and possibly many children (multiple candidate replies per turn). Each message carries quality/helpfulness/harmlessness labels from multiple labelers and a rank within its sibling set. A conversation is a root-to-leaf path. OASST1: ~13,500 contributors, 161K messages, 10K fully-labeled trees, 35 languages, CC-BY 4.0. OASST2 adds ~90K more.

Flattening the tree discards the branching-factor signal. Keeping the tree lets you measure: **user-turn diversity** (across siblings of a parent assistant turn, how varied are follow-ups? — synthetic corpora have near-zero sibling variance because the teacher's "most likely next user turn" is peaked; OASST has high variance), **register variance** (real users include casual/formal/rude/confused registers; Baize/UltraChat users are uniformly polite and articulate), **path-length distribution** (OASST paths to 20+ turns with a power-law tail; synthetic histograms are roughly Gaussian with a hard cap), and **typos + abrupt topic changes** (OASST ~3% typo rate; Baize near-zero). The diagnostic for any new pipeline: generate the same number of dialogues, then measure these four statistics against OASST. Gaps on user-diversity and register-variance are the usual finding.

OASST is too small (~250K messages) to be a sole SFT corpus, and contributors skew Western/English/technical. Its role in modern mixes is as a **realism anchor slice** (5–15% of the mix) counterbalancing the stylistic narrowness of synthetic majorities.

---

## 6. Turn-count and token-length across the chapter

Reconstructed from dataset cards and public previews. See `figures/turn-distribution.html` for the interactive histogram.

| Dataset | Dialogues | Median turns | Max turns | Avg tokens/turn | Speaker protocol | Grounding |
|---|---|---|---|---|---|---|
| [[baize]] | 111.5K | 4 | ~10 | ~100 | 1 LLM, 2 roles | seed question |
| UltraChat Q-world | ~600K | 6 | 14 | ~120 | 2 LLMs | topic taxonomy |
| UltraChat Writing | ~400K | 4 | ~8 | ~150 | 2 LLMs | writing type |
| UltraChat Assistance | ~500K | 4 | ~8 | ~160 | 2 LLMs | C4 passage |
| [[camel]] AI Society | ~1M | 7 | 20 | ~50 | 2 LLMs, role-pair | task spec |
| [[soda]] | 1.5M | 7 | 12 | ~20 | 2 speakers (same LLM) | Atomic triple |
| [[prosocial-dialog]] | 58K | 3 | ~6 | ~40 | user + assistant | rule-of-thumb |
| [[openassistant]] OASST1 | 10K trees / 161K msg | 4–6 path | 20+ | ~80 | human + human | none (free) |
| [[wildchat]] | ~1M | 3 | 50+ | ~60 (user short) | real user + ChatGPT | real task |

Three patterns worth internalizing: **synthetic medians cluster at 3–7** while real-human tails (OASST, WildChat) run much longer — the tail is where follow-up quality is tested and synthetic pipelines lose it. **SODA has the shortest turns (~20 tokens)** because Atomic triples elicit social chit-chat register; Baize/UltraChat turns are 3–8× longer because they are Q&A / writing tasks. **CAMEL has the longest dialogues because its termination is semantic** — hard tasks run long, easy ones short, which is the right behavior and absent from every turn-capped pipeline.

---

## 7. System-prompt diversity — the 2024 extension

By 2024 the frontier moved past topic and role diversity to **system-prompt diversity**. [[system-prompt-diversity]]: sample each dialogue under a *different* system prompt from a large persona pool. Canonical sources: Persona-Hub (Ge 2024, 1B personas mined from web bios / about-me / LinkedIn-style profiles) and SystemChat-2.0 (Hartford 2024, 7K dialogues each with distinct system prompts covering roles, constraints — *"always answer in bullet points"* — styles, safety modes). Empirical finding: persona-conditioning yields up to 2× unique vocabulary and 3× more persona-consistent follow-ups than topic-only conditioning; integrated into Tülu-3 and Qwen 2.5 post-training; +5–10 points on IFEval over topic-only-diverse SFT.

This *extends* UltraChat rather than replacing it — UltraChat's taxonomy is a topic grid, persona-conditioning is a behavioral grid; both axes compose. Gotchas: persona-extraction bias (web-mined personas overrepresent online professionals and English speakers — a narrower population than real users), and stacking too many conditioning axes (role + style + mode + safety) confuses both teacher and student.

---

## 8. Composing the pipelines — what modern mixes do

Modern production SFT mixes do not pick one recipe; they blend. Two contrasting examples:

**[[capybara]] — compact composition.** ~20K dialogues, 60%+ multi-turn. **Amplify-Instruct**: sample seeds from a composed pool (Airoboros + Evol-Instruct + Orca + Vicuna + LessWrong posts + CamelAI), generate initial response, synthesize plausible follow-up user turn, iterate 2–6 rounds. Seed composition is the diversity mechanism; amplification is the multi-turn mechanism. Deliberately small — quality over scale — trains Nous-Capybara-7B to competitive MT-Bench on a single consumer GPU.

**[[smol-talk]] — 1M-scale composition.** The canonical 2025 open recipe. Smol-Magpie-Ultra (400K, Magpie with Llama-3.1-405B) + Smol-summarize (100K) + OpenHermes 2.5 subset (100K) + Smol-rewrite (50K) + MetaMathQA subset (50K) + Smol-constraints (36K) + **SystemChats 2.0 (30K, multi-turn)** + NuminaMath-CoT + Self-Oss-Starcoder2. Explicit multi-turn dialogue is ~30K of 1M — *3%*. Multi-turn capability at modern scale is learned mostly from in-context-follow-through patterns pervading single-turn data; explicit dialogue slices are now *realism insurance*, not primary supervision. This is the structural shift from 2023 (UltraChat's 1.5M dominated SFT mixes) to 2025 (multi-turn is a minority slice).

---

## 9. Practitioner checklist

1. **Define the diversity axis first.** Topic (UltraChat), role (CAMEL), commonsense (SODA), system prompt (SystemChat), seed composition (Capybara). Pick 1–2 primary axes.
2. **Use two separate model calls** for user and assistant if budget allows. Single-call self-chat is cheap but too coherent.
3. **Prefer semantic termination over turn caps.** CAMEL's `<CAMEL_TASK_DONE>` adapts dialogue length to task hardness; hard caps squash the tail.
4. **Include a real-human anchor.** 5–15% OASST or WildChat slice to inject user-register variance no synthetic reproduces.
5. **Measure your turn-length distribution against OASST.** Flat tail past 10 turns → turn-capping too aggressively. User turns all >50 tokens → synthetic user students will never see at inference.
6. **Strip formulaic phrases in post.** `[|Human|]` markers, "As an accountant…", "PersonX said…" all leak; regex strip is standard.
7. **For safety, prefer RoT-grounded engagement over refusal.** [[prosocial-dialog]] and [[constitutional-ai]] both outperform pure-refusal on helpfulness *and* safety.
8. **Cap persona stacking.** Two or three conditioning axes max per dialogue.

---

## 10. What the next chapter builds on

Ch-26 narrows modality to **tool and function-calling data** — a multi-turn variant where one turn is an API call and the next is its result. The CAMEL Inception-prompting template reappears in ToolACE and APIGen-MT, this time with an executable verifier (not `<CAMEL_TASK_DONE>`) as the termination signal. The transition: "dialogue grounded by topic" → "dialogue grounded by a typed function schema + executable verifier." Every lever in this chapter — taxonomy, role pair, grounding, termination — reappears there with tool-call-specific instantiations.

---

## Connections

- [[baize]] / [[baize-construction]] — Xu 2023; self-chat primitive; 111.5K dialogues; `[|Human|]` / `[|AI|]` prompt; median-4-turn distribution.
- [[ultrachat-construction]] / [[ultrachat-pipeline]] — Ding 2023; three-sector taxonomy; two-model role-play; 1.5M dialogues; preview lengths 4–14.
- [[camel]] — Li 2023 (KAUST); Inception Prompting; 50×50×20 role × domain grid; `<CAMEL_TASK_DONE>` termination; ~1M AI Society dialogues.
- [[soda]] — Kim 2023 (Yejin Choi); Atomic 10X triple → narrative → dialogue; 1.5M commonsense-grounded dialogues; COSMO-3B outcome.
- [[prosocial-dialog]] — Kim 2022 (Yejin Choi); 58K dialogues anchored to 300+ rules-of-thumb; precursor to Constitutional AI.
- [[openassistant]] — Köpf 2023 (LAION); tree-structured crowdsourced dialogue; OASST1 161K messages / 10K trees / 35 languages; reference real-human baseline.
- [[wildchat]] — Zhao 2024; ~1M real opt-in ChatGPT logs; realism anchor distinct from OASST.
- [[system-prompt-diversity]] — 2024+ class; Persona-Hub 1B personas; SystemChat-2.0; extends taxonomy with behavioral axis.
- [[capybara]] — LDJ / Nous 2023; Amplify-Instruct seed composition; 20K compact multi-turn.
- [[smol-talk]] — HuggingFace 2024/2025; 1M SmolLM2 SFT mix; 30K SystemChats multi-turn slice.
- [[ch-24]] — the synthetic-pipeline design-pattern framing this chapter instantiates for conversation.
- [[ch-26]] — tool and function-calling data; extends conversation synthesis with typed schemas and executable verifiers.
- `figures/turn-distribution.html` — interactive turn-count + token-length histograms across Baize / UltraChat / CAMEL / SODA / OpenAssistant / WildChat.
