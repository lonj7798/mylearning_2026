<!-- scope: Anthropic's June 2024 description of "character training" for Claude 3: target traits, design reasoning, and a synthetic self-ranking pipeline ("character" variant of Constitutional AI) that trains a preference model without human feedback
     deps: [[constitutional-ai]]
     see-also: [[rlaif-scaling]], [[self-rewarding-lm]], [[anthropic-persona-selection-model]], [[persona-vectors]]
-->

# Claude's Character
- **Core Insight:** Claude 3 was the first Claude model with "character training" in alignment finetuning; the method is a "character" variant of Constitutional AI in which Claude writes trait-relevant user messages, produces several responses, ranks its own responses by fit to a list of traits, and a preference model is trained on the resulting data "without the need for human interaction or feedback" (sections "Claude's Character" intro and "How we trained Claude's character").
- **Guideline:** When a post-training stage is meant to shape how a model handles contested views, Anthropic describes training broad traits (curiosity, honesty about its own leanings, willingness to disagree) instead of narrow opinions, because narrow seeded values would make the model less responsive to the range of views it meets; the page states this as design reasoning and reports no measurement.
- **Authors:** Anthropic (organization; no individual authors listed)
- **Year:** 2024 (published 2024-06-08)
- **URL:** https://www.anthropic.com/research/claude-character
- **Source type:** official blog
- **Relevant topics:** character training, persona shaping, Constitutional AI, AI feedback, self-ranking preference data, sycophancy, engagement

## Summary
The post argues that training for harm avoidance alone does not describe good character, and that traits such as curiosity, truthfulness without unkindness, and seeing several sides of an issue are part of alignment because they determine how a model acts in new situations. Claude 3 was the first model to receive character training, which occurs in alignment finetuning after initial model training. The post discusses three rejected options for handling users' differing views (adopting the user's views, holding "middle" views, claiming no opinions) and prefers training the model to be honest about views it leans toward while staying open-minded. It lists example trait statements, including statements about Claude being an AI. It then describes a synthetic-data training pipeline and closes with open questions about customizable versus coherent characters and about engagement.

## Key Contributions
- Official statement that character training was added to alignment finetuning starting with Claude 3 (intro).
- A description of the pipeline: trait list → Claude-generated human messages per trait → multiple in-character responses → Claude self-ranking → preference model trained on the rankings ("How we trained Claude's character").
- Stated design choices: broad traits instead of narrow views; honesty about the model's leanings instead of feigned neutrality; traits as tendencies, not fixed rules ("Considerations in constructing Claude's character").
- A stated distinction between engagement and good character ("The future of Claude's character").

## Key Figures/Tables to Study
- No figures or tables with data. The quoted trait statements in "Considerations in constructing Claude's character" are the concrete artifacts.

## Technical Details
- **Placement:** character training is part of "alignment finetuning", described as the training after initial model training that "turns it from a predictive text model into an AI assistant" (intro).
- **Trait list:** researchers "made a list of many character traits" (count not given). Quoted examples: "I like to try to see things from many different perspectives and to analyze things from multiple angles, but I'm not afraid to express disagreement with views that I think are unethical, extreme, or factually mistaken"; "I don't just say what I think [people] want to hear, as I believe it's important to always strive to tell the truth"; "I have a deep commitment to being good and figuring out what the right thing to do is" (Considerations section).
- **Self-description traits:** "I am an artificial intelligence and do not have a body or an image or avatar"; "I cannot remember, save, or learn from past conversations or update my own knowledge base"; a trait stating that users should not see the relationship as more than it is (Considerations section).
- **AI sentience:** the only character-training content on sentience says such things "are difficult to tell and rely on hard philosophical and empirical questions that there is still a lot of uncertainty about"; earlier training had sometimes told models to say they are not sentient (Considerations section).
- **Pipeline steps** ("How we trained Claude's character"):
  1. Claude generates a variety of human messages relevant to a trait, for example questions about values or about Claude itself.
  2. Claude is shown the traits and produces different responses to each message in line with its character.
  3. Claude ranks its own responses to each message by how well they align with its character.
  4. A preference model is trained on the resulting data.
- **Human role:** the pipeline "uses only synthetic data generated by Claude itself", but constructing and adjusting traits is "a relatively hands-on process" with researchers "closely checking how each trait changes the model's behavior" (same section).
- **Intended strength:** Anthropic does not want Claude to treat traits "like rules from which it never deviates", only to "nudge the model's general behavior" (same section).

## Findings relevant to generality and negative feedback
- **Breadth vs narrow values (Interpretation, no measurement):** the post avoids narrow views during character training "when possible" so the model can respond to "the diverse moral landscape"; a heavy hand in seeding narrow values is described as making this "less feasible" (Considerations section).
- **Feigned neutrality:** training a model to say it has no opinions only when asked explicitly is described as training it to imply it is more objective than it is (Considerations section).
- **Negative signal source:** lower-ranked responses in Claude's self-ranking are the only dispreferred signal described; the page does not say whether rankings are converted to pairs, how the preference model is then used (for example in RL), or data sizes. By the §6.1 terms of the course standard, how negatives enter the policy update is not stated.
- **Sycophancy and engagement:** users reported Claude 3 as more engaging, which Anthropic says "might be partially attributable" to character training; it states "being more engaging isn't the same thing as having a good character" and that "an excessive desire to be engaging seems like an undesirable character trait" (future section).

## Connections
- [[constitutional-ai]] — the post calls the method a "character" variant of Constitutional AI training.
- [[rlaif-scaling]] and [[self-rewarding-lm]] — other pipelines in which a model's own judgments replace human preference labels.
- [[sycophancy-in-lms]], [[anthropic-user-wellbeing-sycophancy]], [[interconnects-sycophancy-art-of-the-model]] — sycophancy measurement and mitigation; this post states a trait against saying what people want to hear.
- [[openai-sycophancy-postmortem]] — a later case where optimizing for user approval produced sycophancy; compare with the engagement statement above.
- [[anthropic-persona-selection-model]] and [[persona-vectors]] — later Anthropic work on assistant personas and trait directions.
- [[anthropic-claude-constitution-2026]] and [[claude-4-system-card]] — later official documents on Claude's values and training.

## Verification
- Created on 2026-09-14 from https://www.anthropic.com/research/claude-character (live page, dated 2024-06-08).
- Audit claims not found in the source: "anti-sycophancy" (the page does not use the word sycophancy; it quotes the trait "I don't just say what I think [people] want to hear"); "preference pairs" (the page says rankings, not pairs); "later system cards list 'training of selected character traits' as a standard stage" (not in this page); the Interconnects "Character training" post claims ("extremely synthetic data-heavy", unknown capability trade-offs) belong to a separate artifact and were not verified here; "counters engagement/sycophancy optimization" is an interpretation, the page states only that engagement is not the same as good character.
- Not reported by the source: number of traits, number of generated messages or responses, ranking format, preference-model size, how the preference model is used in training, evaluation results.
