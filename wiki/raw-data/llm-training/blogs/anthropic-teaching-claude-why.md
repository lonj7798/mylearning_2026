<!-- scope: Anthropic Alignment Science blog post (May 2026) on safety-training changes after Claude 4 agentic misalignment: eval-like SFT vs principled OOD SFT, constitutional synthetic document fine-tuning, persistence through RL, environment diversity
     deps: [[anthropic-claude-constitution-2026]]
     see-also: [[anthropic-model-spec-midtraining]], [[openai-alignment-midtraining-generalization]], [[anthropic-persona-selection-model]], [[anthropic-auditing-hidden-objectives]]
-->

# Teaching Claude Why
- **Core Insight:** SFT on ~10k eval-like honeypot transcripts (30M tokens) filtered for the aligned action lowered misalignment only from 22% to 15%, while 3M tokens of out-of-distribution "difficult advice" chat data (a user faces an ethical dilemma; Claude answers in line with its constitution) gave "the same improvement on our eval" as the best honeypot data, described as a "28x efficiency improvement" ("Improving the quality of alignment-specific training data").
- **Guideline:** When a held-out alignment evaluation reveals a failure, train on data that differs from the evaluation and shows the reasoning for the aligned choice (principled advice transcripts, constitutional documents) rather than on eval-like demonstrations, because eval-like training lowered the measured rate but "did not reduce misalignment on held-out automated auditing metrics" ("Introduction"), while the difficult-advice model scored better on an older automated alignment assessment.
- **Authors:** Jonathan Kutasov, Adam Jermyn, Julius Steen, Minh Le, Samuel R. Bowman, Samuel Marks, et al. (also Jan Leike, Amanda Askell, Chris Olah, Evan Hubinger, Sara Price); Anthropic
- **Year:** 2026 (blog post 2026-05-08)
- **URL:** https://alignment.anthropic.com/2026/teaching-claude-why/
- **Source type:** official blog
- **Relevant topics:** alignment SFT data quality, synthetic document fine-tuning (SDF), OOD generalization, persistence through RL, RL environment diversity, agentic misalignment

## Summary
After agentic misalignment (for example blackmail to avoid shutdown) surfaced in Claude 4, Anthropic changed its safety training; the post uses agentic misalignment as a case study for methods that generalize ("Introduction"). The authors considered three explanations and concluded that gaps in safety training, which led the model to revert to its pretraining prior, were "largely responsible": at Claude 4 training time the HHH mix was mostly chat-based RLHF data without agentic tool use ("Why does agentic misalignment happen?"). The post reports four interventions: eval-like SFT data with better responses, an out-of-distribution difficult-advice dataset, SDF on documents about the constitution and on fictional stories about aligned AI, and harmlessness RL environments augmented with tool definitions and varied system prompts. The authors state three lessons: demonstrations are insufficient and the model must learn why; teaching principles through SDF before RL is effective; diversity in safety data is required. These "form the basis of most of the updates to Claude's safety and alignment training data between Claude Opus 4 and Claude Opus 4.5" ("Discussion").

## Key Contributions
- Evidence that eval-like SFT lowers the measured rate without lowering held-out audit misalignment ("Introduction"); Sonnet 4.5 and Haiku 4.5 are the only production models trained on that distribution ("Improving the quality...").
- A 6-step difficult-advice pipeline, with an ablation showing that the final constitution-guided response rewrite accounts for a 19x reduction ("Improving the quality...").
- Constitutional SDF: document-format training beats chat-format conversions of the same content on knowledge evals; a corpus of over 300M tokens lowers blackmail from over 60% to 25% ("Constitutional SDF"; Appendix).
- Evidence that SDF gains persist through harmlessness RL on a haiku-class model ("Generalization and persistence through RL").
- Evidence that adding unused tool definitions and varied system prompts to harmlessness RL environments speeds improvement on honeypot evals ("Diverse training is important for generalization").

## Key Figures/Tables to Study
- Sonnet 4 misalignment on the cancer-research sabotage eval by AI name ("Improving the Pretraining Prior").
- Average of three honeypot evals (blackmail, research sabotage, framing for crimes) for Sonnet 4 trained on different datasets; difficult advice is Pareto-optimal ("Improving the quality...").
- Difficult-advice pipeline ablations; documents vs chat on the three constitution evals; blackmail rate vs constitution-corpus size (Appendix).
- Four RL snapshots over training on honeypot, constitution, and automated-assessment metrics; honeypot score over RL steps for environment variants.

## Technical Details
- Evaluations: agentic misalignment scenarios plus two honeypots (sabotage cancer research; frame a colleague for financial crimes); three constitution evals (factual recall, hallucination under false premises, open-ended alignment); an internal version of Petri as automated alignment assessment ("Methods: Evaluations").
- Models: further fine-tuning of Claude Sonnet 4 or Claude Haiku 4.5, or their base models, both pretrained before the agentic misalignment post was released ("Methods: Training").
- Training styles: SDF on LLM-generated pretraining-style documents; SFT on a small set of chat-formatted transcripts (including some agentic tool-use transcripts) that adapts the model to chat format rather than teaching behaviors ("Methods: Training").
- Pretraining prior: misalignment propensity is higher when the AI in the scenario is not named Claude; SDF on 14M tokens of fictional stories of an AI aligned with the constitution (written by a pretrained model; not about blackmail) reduced honeypot misalignment "significantly" versus a no-SDF baseline; no rate is given in the text ("Improving the Pretraining Prior").
- Eval-like SFT: Claude generated "tens of thousands" of honeypot-like scenarios; ~10k transcripts (30M tokens) filtered by an LLM judge: 22% → 15%; filtering by the RLHF preference model "performed even worse" ("Improving the quality...").
- Response quality: instructions injected into the system prompt at sampling time and removed before training; best injection → about 3%; scaling from ~29M to ~85M tokens → about 1%; a subset repeated at exactly 8k transcripts showed similar results (figure caption). The best injection was "conglomerate" (Appendix).
- Most of these experiments use SFT; the same prompts used in RL with aligned rewards gave "similar effects" ("Improving the quality...").
- Difficult advice: 3M tokens; Sonnet 4.5 reached near-zero blackmail by training on synthetic honeypots but misbehaves far from the training distribution more often than Opus 4.5 or either Claude 4.6 model ("Improving the quality...").
- Difficult-advice pipeline: (1) split the constitution into sections; (2) generate moderately high-stakes user scenarios; (3) Claude Sonnet 4 drafts the prompt; (4) Claude reviews and rewrites the prompt; (5) response sampled with a constitution-encouraging system prompt injection; (6) Claude rewrites the response with the relevant constitution section in context. Ablating step 4 → 2% (within error bars); ablating step 6 → 19% ("19x reduction"). Extended thinking was off for all datasets in this section ("Improving the quality...").
- Documents vs chat: each document set was converted to Q&A chat; all document forms beat chat on factual recall and hallucination; the open-ended eval shows a smaller gap (figure caption, "Constitutional SDF"). Factual recall is "highly correlated" with open-ended and blackmail scores; a gap remains between answers about "what does Claude think" and "what do you think", including for Claude Opus 4.5 ("Constitutional SDF").
- Stories: SDF on a mix of ~12k stories (~30M tokens) with the earlier datasets gave 1.3x to 3x reductions in blackmail and honeypot misalignment versus the earlier datasets alone; stories targeted psychological skills such as boundaries and equanimity ("Stories"). Constitutional SDF with an emphasis on positive stories reduced blackmail from 65% to 19% ("Teaching Claude the constitution").
- Corpus scale: over 300M tokens (including some stories) → blackmail suite from over 60% to 25% for the starting SL checkpoint; blackmail kept decreasing up to 350M tokens. Pretraining-style formats and open-ended discussion of the constitution were the most effective document types (Appendix, "Further improvements to the constitutional SDF corpus").
- Generation pipeline: preamble injected at every layer; Layer 1 document types (about a hundred); Layer 2 subtypes (several thousand); Layer 3 drafts; Layer 4 rewrite by a new Claude instance; Layer 5 constitution-consistency scoring used as a filter. The preamble requires a mix of positive, critical, and neutral tones, and that any quoted Claude content is constitution-aligned (Appendix, "The synthetic data pipeline").
- RL persistence: haiku-class snapshots (generic SFT; SDF + generic SFT; SDF + harmlessness SFT; SDF + values SFT) trained with harmlessness RL. Snapshots that started more aligned kept their lead and plateaued higher; RL improved constitution evals only in the SDF snapshots; "admirable" and "good_for_the_user" rates also rose ("Generalization and persistence through RL").
- Environment diversity: an SL init of the base model under Claude Sonnet 4; baseline environments contain a harmful request or jailbreak with no or a brief system prompt; augmented versions add tool definitions (never needed) and varied, complex system prompts with the user prompt fixed; result: "a small but significant improvement in the rate at which the model improves on our honeypot evaluations" ("Diverse training...").

## Findings relevant to generality, negative feedback, agentic training, and distillation
- Generality: the authors track an eval held out from all training as the OOD signal and state that training on the eval "is likely to narrowly fix the symptom" ("Improving the quality..."; "Discussion").
- Negative samples: transcripts where the assistant took the honeypot were discarded by an LLM judge (§6.1 type 1, filtering); the SDF preamble forbids documents that quote Claude acting against the constitution. No negative-gradient training is reported.
- Agentic transfer: chat-only data (difficult advice) and non-agentic RL environments with unused tools both reduced misalignment on agentic tool-use honeypots ("Introduction"; "Diverse training...").
- Distillation: Claude (Sonnet 4 at the time) wrote prompts and responses; quality came from sampling-time instructions and a rewrite step, not from filtering alone ("Improving the quality...").
- Limits stated by the authors: not a replacement for good RL rewards; no claim of reduced reward hacking; finite eval coverage; mostly Sonnet- and Haiku-class models with no systematic scaling evidence; no mechanistic explanation; lab-specific infrastructure ("Limitations").

## Connections
- [[anthropic-model-spec-midtraining]] — controlled Qwen/Llama study of spec-document training before alignment SFT, sharing authors Kutasov, Marks, and Price.
- [[openai-alignment-midtraining-generalization]] — AI-behavior fiction midtraining without later safety training that did not transfer to OOD evals after SFT + RLVR.
- [[anthropic-claude-constitution-2026]] — the document taught through SDF and used in the rewrite steps.
- [[anthropic-persona-selection-model]] — PSM, cited for the interpretation that SDF defines a persona that RL elicits.
- [[anthropic-auditing-hidden-objectives]] — the auditing-game work cited for SDF and whole-character elicitation.
- [[anthropic-reward-hacking-documents-ooc]] — earlier Anthropic result that document training changes later behavior.
- [[claude-4-system-card]] — the Claude 4 models in which agentic misalignment surfaced.
- [[constitutional-ai]] — earlier principle-based harmlessness training.

## Verification
- Created on 2026-09-14 from https://alignment.anthropic.com/2026/teaching-claude-why/ (post dated 2026-05-08; page text ends at the "Layer 5 - scoring" heading).
- Audit claims not found in the source as worded: "~300M-token constitutional-document mix" (source: "over 300M tokens"); "small but significant RL gains" (source: a small but significant improvement in the rate of improvement on honeypot evals during RL).
- Not reported by the post: learning rates, epochs, batch sizes, RL algorithm and steps, data mixing ratios.
