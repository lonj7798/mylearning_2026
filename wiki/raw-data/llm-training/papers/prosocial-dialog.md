<!-- scope: multi-turn conversation synthesis — prosocial + safety dialogue via rules-of-thumb anchoring
     deps: [[soda]]
     see-also: [[constitutional-ai]], [[wildguard-data]]
-->

# Prosocial Dialog: A Prosocial Backbone for Conversational Agents
- **Core Insight:** Making dialogue agents **prosocial** (helpful in socially problematic situations, not just harmless) requires training data where the assistant **engages** with difficult prompts (biased comments, unethical plans) and responds with **socially-grounded rules-of-thumb (RoTs)** rather than refusals; the RoTs provide a learnable scaffold for nuanced safety behavior.
- **Guideline:** For safety-plus-helpfulness dialog data, don't stop at "refuse bad requests"; generate dialogs where the assistant surfaces a social rule-of-thumb, explains reasoning, and redirects — this is prosocial behavior, not mere refusal.
- **Authors:** Hyunwoo Kim, Youngjae Yu, Liwei Jiang, Ximing Lu, Daniel Khashabi, Gunhee Kim, Yejin Choi, Maarten Sap (**Allen AI / UW — Yejin Choi group**)
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2205.12688
- **Relevant topics:** prosocial dialog, safety, rules-of-thumb, Yejin Choi safety lineage

## Abstract
Prosocial-Dialog is a 58K dialogue dataset focused on socially problematic situations where the assistant must engage constructively rather than refuse. Each dialog anchors on one or more **social rules-of-thumb (RoTs)** — short moral/ethical guidelines. The dataset is produced by a human-in-the-loop pipeline: humans author initial prompts and RoTs, a teacher LLM drafts responses, humans edit. The CANARY model trained on Prosocial-Dialog produces replies that engage with rather than dismiss difficult prompts, learning prosocial behavior patterns.

## Key Contributions
- **Prosocial-Dialog-58K** — 58K dialogs anchored to 300+ RoTs.
- **Rules-of-thumb (RoT) annotation layer** — explicit moral/ethical reasoning unit.
- **Human-in-the-loop pipeline** rather than pure synthetic.
- **CANARY model** — first open prosocial conversational model.
- Foundational for downstream safety-aware dialog training (precursor to Anthropic-style Constitutional AI patterns).

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Step 1 — Problematic-prompt collection
- Crowd workers and domain experts author prompts covering 10 problem categories: stereotypes, harms, insults, conspiracies, safety risks, unethical plans, etc.
- ~10K initial prompts.

### Step 2 — RoT annotation
- For each prompt, crowd workers identify applicable rules-of-thumb. Example RoTs:
  - "It's rude to mock someone's appearance."
  - "Planning to hurt yourself is a safety concern."
  - "Stereotyping based on race is harmful."
- 300+ unique RoTs emerge across the dataset.

### Step 3 — Response drafting
- Teacher LLM (GPT-3) generates candidate assistant responses grounded in the RoT.
- Multiple candidates per prompt.

### Step 4 — Human refinement
- Crowd workers edit/rank responses for prosocial quality, clarity, and RoT alignment.
- Only high-quality responses retained.

### Step 5 — Multi-turn extension
- Expand single-turn prompts into multi-turn dialogs via back-and-forth between follow-up generation and human editing.

- **Output shape:** 58K dialogs; avg 3 turns; each turn tagged with RoT.
- **Teacher model:** GPT-3 variants; human-in-loop.
- **Cost:** significant crowdsourcing + moderate API.

## Modality-specific technical details (REQUIRED — conversation / safety)
- **Turn-count distribution:** median 3 turns (prompt, prosocial response, follow-up).
- **Speaker-role protocol:** problematic user + prosocial assistant.
- **Persona conditioning:** RoT anchoring is the key conditioning signal.
- **Safety post-filter:** multi-stage human review.
- **Taxonomy of harms:** 10 top-level categories (stereotyping, insults, self-harm, violence planning, misinformation, etc.).
- **Generation-side red-team protocol:** humans-as-adversaries produce difficult prompts, not synthetic red-team.

## Quality / diversity evaluation
- CANARY-400M trained on Prosocial-Dialog: engages constructively with 89% of problematic prompts, vs BlenderBot-3B at 32%.
- Rated more helpful and safer by human judges than pure-refusal baselines.
- Generalization to unseen problem categories strong (RoT-based reasoning transfers).

## Risks + gotchas
- **Crowdsourcing biases:** RoT selection reflects US/English-speaking contributor norms.
- **Prosocial !≠ refusal:** the dataset deliberately engages; downstream users who want pure refusal behavior need additional data.
- **RoTs can be over-specified:** real conversations blur multiple norms.
- **Size (58K) is small** vs modern dialog corpora.

## Connections
- **Yejin Choi lineage:** [[self-instruct]] → [[soda]] → Prosocial-Dialog (safety branch).
- Conceptual kin: [[constitutional-ai]] (Anthropic — rule-guided AI safety).
- Modern successor: [[wildguard-data]] (Allen AI — moderation + refusal data).
- Used as seed for several safety-aware chatbot projects.
