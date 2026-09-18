<!-- scope: Anthropic's first-party report (2026-02) of large-scale API distillation campaigns attributed to DeepSeek, Moonshot, and MiniMax: volumes, targeted capabilities, prompt patterns, access methods, and countermeasures
     deps: none
     see-also: [[distillation-source-matters]], [[deepseek-r1-distill-synth]], [[generative-reward-models]], [[nathan-lambert-interconnects]]
-->

# Detecting and preventing distillation attacks
- **Core Insight:** Anthropic reports that three labs generated over 16 million exchanges with Claude through about 24,000 fraudulent accounts (DeepSeek over 150,000, Moonshot AI over 3.4 million, MiniMax over 13 million), targeting agentic reasoning, tool use, and coding, and in DeepSeek's case also chain-of-thought generation and rubric-based grading used as a reinforcement-learning reward ("What we found").
- **Guideline:** When using this post as evidence about how API outputs enter post-training, use it for the data-collection side only (volume, prompt types, targeted capabilities), because the attributions come from the provider's own detection methods and the post reports no training settings or evaluation results for any model trained on the collected data.
- **Authors:** Anthropic (organization post; no individual authors listed)
- **Year:** 2026 (published 2026-02-23)
- **URL:** https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks
- **Source type:** official blog (first-party announcement, tagged Announcements and Policy)
- **Relevant topics:** API-based distillation, synthetic data provenance, chain-of-thought elicitation, model-as-grader for RL rewards, agentic and computer-use data, distillation detection, access controls

## Summary
The post defines distillation as "training a less capable model on the outputs of a stronger one" and states that it is a widely used and legitimate method, for example when frontier labs distill their own models into smaller ones. It then reports three campaigns that Anthropic attributes, "with high confidence", to DeepSeek, Moonshot AI, and MiniMax, which used fraudulent accounts and proxy services in violation of Anthropic's terms of service and regional access restrictions. For each campaign it gives the number of exchanges, the capabilities targeted, and the traffic patterns observed. It describes the reseller networks used for access, the prompt pattern that distinguishes distillation from ordinary use, and four defensive measures. It also argues that illicitly distilled models are unlikely to keep safety safeguards, and that such campaigns undermine export controls while reinforcing the rationale for them ("Why distillation matters"; "Distillation attacks and export controls").

## Key Contributions
- Per-lab volumes and capability targets for three attributed campaigns ("What we found").
- Named uses of teacher output: responses for direct training, chain-of-thought reconstruction, rubric grading as an RL reward model, and generation of tasks for RL ("DeepSeek"; "How distillers access frontier models").
- A description of "hydra cluster" proxy networks and of the traffic signature of distillation.
- A list of countermeasures: detection, intelligence sharing, access controls, and output-level safeguards ("How we're responding").

## Key Figures/Tables to Study
- No figures or tables. The per-lab subsections under "What we found" and the example prompt under "How distillers access frontier models" carry the specifics.

## Technical Details
- **Totals:** over 16 million exchanges through approximately 24,000 fraudulent accounts (introduction).
- **Attribution methods:** IP address correlation, request metadata, infrastructure indicators, and in some cases corroboration from industry partners ("What we found").
- **DeepSeek, over 150,000 exchanges:** targets were reasoning across diverse tasks, "rubric-based grading tasks that made Claude function as a reward model for reinforcement learning", and censorship-safe alternatives to policy-sensitive queries. Traffic was synchronized across accounts with identical patterns, shared payment methods, and coordinated timing. Prompts asked Claude "to imagine and articulate the internal reasoning behind a completed response and write it out step by step", which the post describes as generating chain-of-thought training data at scale. Request metadata traced accounts to specific researchers ("DeepSeek").
- **Moonshot AI (Kimi models), over 3.4 million exchanges:** targets were agentic reasoning and tool use, coding and data analysis, computer-use agent development, and computer vision. The campaign used hundreds of fraudulent accounts across multiple access pathways. A later phase attempted "to extract and reconstruct Claude's reasoning traces". Request metadata matched public profiles of senior Moonshot staff ("Moonshot AI").
- **MiniMax, over 13 million exchanges:** targets were agentic coding and tool use and orchestration. The campaign was detected while active, before MiniMax released the model it was training. When Anthropic released a new model during the campaign, MiniMax redirected nearly half of its traffic to it within 24 hours ("MiniMax").
- **Access route:** Anthropic "does not currently offer commercial access to Claude in China", or to subsidiaries of those companies located outside China. Commercial proxy services resell access through networks of fraudulent accounts spread across Anthropic's API and third-party cloud platforms. One proxy network managed more than 20,000 fraudulent accounts at once and mixed distillation traffic with unrelated customer requests ("How distillers access frontier models").
- **Stated goals of the traffic:** "either to collect high-quality responses for direct model training, or to generate tens of thousands of unique tasks needed to run reinforcement learning" (same section).
- **Signature:** a prompt that looks benign alone (the example is an "expert data analyst" instruction) becomes identifiable when "variations of that prompt arrive tens of thousands of times across hundreds of coordinated accounts, all targeting the same narrow capability". Listed hallmarks: volume concentrated in a few areas, highly repetitive structures, and content that maps onto what is most valuable for training (same section).
- **Countermeasures:** classifiers and behavioral fingerprinting for distillation patterns, including detection of chain-of-thought elicitation and of coordinated activity across accounts; sharing technical indicators with other labs, cloud providers, and authorities; stronger verification for educational, security-research, and startup accounts; product-, API-, and model-level safeguards "designed to reduce the efficacy of model outputs for illicit distillation" ("How we're responding").
- **Not reported:** tokens per exchange or total tokens, which models or checkpoints were trained on the data, training stage (SFT, preference, or RL), sampling settings, filtering, and any measured effect on the distilled models. The post contains no raw traffic data or technical indicators; it states that indicators are shared with other AI labs, cloud providers, and authorities.

## Findings relevant to distillation, agentic training
- **Stage and prompt types:** the post names three uses of teacher output: reasoning and chain-of-thought text for training, Claude as a rubric-based grader supplying RL rewards, and task generation for RL. It does not say which training stage of the named labs' models consumed each type.
- **Agentic targets:** all three campaigns are described as targeting agentic reasoning, tool use, and coding; Moonshot's also targeted computer-use agent development.
- **Teacher and sampling:** the teacher is Claude accessed through the API and proxy resellers. Sampling parameters and quality control on the collector side are not reported.
- **Status:** first-party attribution by the provider of the teacher model. The named labs' responses are not included in the post.

## Connections
- [[deepseek]], [[moonshot-kimi]]: lab cards for two of the named organizations; [[minimax-01]]: an earlier MiniMax model report (the post does not name a MiniMax model).
- [[generative-reward-models]]: LLM judges used as reward signals, the use the post attributes to the rubric-grading traffic.
- [[deepseek-r1-distill-synth]]: text-level distillation of reasoning traces into student models, the form of distillation the post describes.
- [[distillation-source-matters]]: controlled evidence that the choice of teacher changes student results in text-level reasoning distillation.
- [[sparse-logit-sampling-kd]]: logit-level distillation, which requires teacher probabilities; the post describes collection of output text only.
- [[nathan-lambert-interconnects]]: Nathan Lambert's reply, "How much does distillation really matter for Chinese LLMs?" (Interconnects, 2026-02-24, https://www.interconnects.ai/p/how-much-does-distillation-really), is a separate artifact and is not extracted in this card.

## Verification
- Created on 2026-09-14 from https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks (web page dated 2026-02-23; no version identifier).
- Corrections: the batch title joined this post with the Interconnects commentary; this card describes only the Anthropic post, as named by the slug.
- Audit claims not found in the source: "classic logit KD is not actually possible from API models"; "reasoning traces are hidden by major APIs"; 10-25K tokens per exchange and a 150-400B token total; an Olmo 3 SFT set of about 20B tokens; "I still wouldn't say it is a crucial factor in these Chinese labs post-training capabilities"; claims from "The Distillation Panic" (2026-05-04). The first, third, fourth, and fifth appear in the separate Interconnects post of 2026-02-24 (checked 2026-09-14); for the second, that post says only that OpenAI's reasoning traces are "not exposed by default"; "The Distillation Panic" was not checked.
