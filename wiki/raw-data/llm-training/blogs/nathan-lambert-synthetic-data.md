<!-- scope: Nathan Lambert's June 2024 Interconnects post on synthetic-data trends, plus a companion post on post-training cost
     deps: [[ultrafeedback-construction]]
     see-also: [[allenai-tulu-synth]], [[model-collapse]], [[nathan-lambert-interconnects]]
-->

# Frontiers in synthetic data
- **Core Insight:** Writing about a project to fine-tune a strong open-weights model, the author states "I think synthetic data can do almost all of the work", and lists eight trends he watches, of which the first is that direct distillation from a stronger model remains the dominant method (opening section; §1).
- **Guideline:** When the target task is open-ended generation (email writing, summarization), generate completions with the most recent available teacher rather than reusing a corpus built from an older teacher, because the post reports the original GPT-4 sits about 100 Elo below GPT-4o on ChatBotArena and, in the companion post, that a 100-Elo margin corresponds to about a 2/3 win probability (§1; companion §1). When the target task is math or code with uncommon libraries, the post states synthetic data "will take much more effort" and gives no recipe (§1).
- **Author:** Nathan Lambert (Allen Institute for AI; Interconnects)
- **Year:** 2024 (published 2024-06-21; edits added 2024-06-21 and 2024-06-27)
- **URL:** https://www.interconnects.ai/p/frontiers-in-synthetic-data
- **Source type:** practitioner evidence (opinion post with citations to model reports; no experiments of its own)
- **Relevant topics:** synthetic data for SFT and preference tuning, distillation, data filtering, prompt synthesis, weak-to-strong generalization

## Summary
The post lists eight trends in how synthetic data is used at the fine-tuning stage, drawn from public model reports
(Gemini 1.5, Nemotron-4 340B, Llama 3, Claude 3.5) rather than from the author's own experiments. The framing is a
stated project goal: take a strong open-weights base model and use synthetic data to close as much of the gap to
frontier models as possible with an openly available pipeline. Claims are the author's readings of those reports and
are labelled as such below.

## Key Contributions
- Eight named trends: (1) direct distillation dominates; (2) Gemini Flash and Claude Haiku are plausibly distilled;
  (3) filtering prevents collapse; (4) license and terms-of-service restrictions shape which teacher a lab may use;
  (5) multi-output-source preference datasets have a lower ceiling than on-policy generation plus filtering;
  (6) structured synthetic data targets verifiable behaviours such as output format; (7) weak-to-strong
  generalization is supported by more evidence than before; (8) synthetic prompt creation is under-studied.
- Collects the quotations from the Nemotron-4 340B report and the Llama 3 blog that the argument rests on, so a
  reader can check them at the primary sources.

## Key Figures/Tables to Study
- The post contains no figures, tables, or measurements produced by the author. All quantities are quoted from
  other organizations' reports.

## Technical Details
- Distillation: the post quotes the Gemini 1.5 report that "Gemini 1.5 Flash is a dense Transformer based model
  that is online distilled ... from Gemini 1.5 Pro", added as an edit after publication (§2).
- Teacher staleness: datasets advertised as "created with GPT-4" often used a GPT-4 version the post places
  "100+ points behind" the then-current leaderboard position, a gap the post equates to the distance from the
  first GPT-4 to GPT-4o, and also to the distance from the original GPT-4 down to Llama-2-chat-70B (§1).
- Filtering and accumulation: the post cites work with Stanford and MIT authors showing that accumulating data
  across iterations, rather than replacing it in a closed self-consuming loop, avoids mode collapse (§3). See
  [[model-collapse]] for the primary result.
- Llama 3: quotes the Meta blog statement that Llama 2 was used to build the text-quality classifiers for Llama 3,
  and that synthetic data was used for coding, reasoning, and long context (§3).
- Nemotron-4 340B: quotes that Nemotron-4-340B-Reward scores each dialogue and samples below a threshold are
  filtered out (§3), that 23 prompts were used to generate synthetic prompts rather than reusing training-set
  prompts (§8), and that instruction prompts explicitly specifying an output format (for example JSON) were
  created to improve IFEval (§6).
- Weak-to-strong generalization: quotes the Nemotron report that "the teacher model does not impose a ceiling on
  the student model" (§7). This is the report's own claim, not an independent replication.
- Multi-source preference data: the post argues UltraFeedback-style mixtures force a model to fit sequences that
  are low-probability under its own base, and notes in an edit that the UltraFeedback authors had difficulty
  beating UltraFeedback with UltraInteract (§5). No numbers are given for either statement.

## Companion post: "The state of post-training in 2025"
A separate Interconnects post (2025-01-08, https://www.interconnects.ai/p/the-state-of-post-training-2025) gives
the author's cost estimates for post-training. He states the numbers are estimates based on headcount and data
costs, not disclosed figures.
- LLaMA (Q1 2023): under $1M, instruction tuning only (§2).
- Llama 2 (Q3 2023): about $10-20M, with 1.4M preference pairs (§2).
- Llama 3.1 (Q3 2024): over $50M, with a post-training team of about 200 people (§2).
- Human preference data is priced at about $5-20 per preference point against under $0.01 per sample for AI
  feedback (§3).
- For o1-style models, post-training loss functions "can account for 40% or more of the overall compute" (§2).
- Tülu 3, which bought no human data, is estimated at over $1M (§2). See [[tulu-3]].
- The post divides post-training into instruction finetuning, preference finetuning, and reinforcement finetuning
  (§ opening list).

## Findings relevant to generality
- The post claims easy open-ended tasks transfer well from a teacher while math and code with uncommon libraries do
  not, and gives no measurement for either side (§1). Status: Interpretation.
- On negative feedback and on long context the post reports nothing beyond the quoted Llama 3 sentence.

## Connections
- [[allenai-tulu-synth]] — Ai2's own description of the synthetic post-training pipeline the author works on.
- [[model-collapse]] and [[strong-model-collapse]] — the accumulation-versus-replacement result cited in §3.
- [[persona-hub]], [[magpie]], [[ultrafeedback-construction]] — primary sources for prompt and preference synthesis.
- [[nathan-lambert-interconnects]] — index of the author's other posts used in this library.
- [[tulu-3]] — the project whose cost estimate appears in the companion post.

## Verification
- Checked on 2026-09-18 against: https://www.interconnects.ai/p/frontiers-in-synthetic-data (2024-06-21) and
  https://www.interconnects.ai/p/the-state-of-post-training-2025 (2025-01-08).
- Corrections to the previous card version:
  - Title "Nathan Lambert (Interconnects) on Synthetic Data — Curated Posts 2024–2025" → the card now describes one
    primary artifact, the post titled "Frontiers in synthetic data".
  - "Year: 2024–2025 (ongoing)" → 2024-06-21 for the primary post; the companion post is dated 2025-01-08.
  - "Post-training costs in 2025 are dominated by synthetic-data infrastructure ... not by human annotation" → the
    companion post attributes cost to "large data bills and extensive inference" and to headcount, and does not
    rank the two (companion §2).
  - "'synthetic data can do almost all of the work' *once* the verification layer is correct" → the sentence is
    "I think synthetic data can do almost all of the work", said about fine-tuning a strong open-weights model; no
    verification condition is attached to it (opening section).
  - "Three layers: pretraining synthetic, SFT synthetic, preference synthetic" → the post has eight numbered
    trends and does not present this three-layer taxonomy.
  - "Diagnoses current stack: SFT → DPO/RLHF → RLVR" → the companion post names instruction finetuning,
    preference finetuning, and reinforcement finetuning; "RLVR" is not the term used there.
  - "post-training now consumes a substantial fraction of total FLOPs" → the companion post makes this claim only
    for o1-style models, at "40% or more of the overall compute" (companion §2).
- Removed as unsupported by the source: the lineage "Self-Instruct → Magpie → Persona-Hub"; the lineage
  "UltraFeedback → West-of-N → Con-J judges"; the "2024 Year in Review" section and its claims; "Verification is
  the bottleneck, not generation; in the RLVR era, the scarce resource is verifiable prompts"; "the cost structure
  of open post-training ≈ inference-compute × number-of-model-generations × number-of-iterations"; "open prompt
  corpora are the new scarce asset"; "judge-LLM bias is the next major audit frontier"; "model-collapse concerns
  are ... over-stated in strict-replacement regimes"; "the data-foundry business model (Scale AI, Surge) is
  pressured by synthetic data"; the pointer to [[rlvr-tulu3]] and [[direct-judgement-preference]] as claims of
  this post.
- Not reported by the source: any experiment, dataset, model, or evaluation run by the author; any number for the
  size of the effects it describes.
