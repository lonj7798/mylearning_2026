# 23c — Human Reference, Persona Experiments, and Emotional Synthesis

<!-- scope: primary-source setups, measured outcomes, and transfer limits for policy-conditioned expression, emotion, lexical metrics, and valid tone transformations
     deps: [[read]]
     see-also: [[read]], [[practice]], [[long-tail-papers]]
     sources: [[realtalk]], [[deepdialogue]], [[persona-lexical-diversity]]
-->

## 1. REALTALK is longitudinal messaging, with a narrow sample

**Result (single study).** [[realtalk]] follows ten US native-English participants aged 18–25 over 21 days (§3.1). It compares human messaging with synthetic dialogue, including emotional expression and persona stability. §5.2 reports broader emotional variation and style fluctuation in real conversations, versus more uniformly positive and excessively empathetic synthetic dialogue.

The study offers a valuable reference for asking whether generated dialogue is too smooth, stable, or agreeable. It does not represent all customer demographics: §8 limits the demographic coverage and chit-chat setting, and the analysis does not establish acoustic realism. Sentiment-polarity entropy is one measure in §4.3, not a complete emotion-quality metric.

**Interpretation.** For sales data, compare human reference interactions and synthetic candidates under matched tasks and conditions. A mismatch is an audit signal, not proof that copying every human utterance is desirable. Real conversations can violate your current operator policy; target replies still require policy-grounded validation.

## 2. Fine-grained personas and length confounds

**Result (single study).** [[persona-lexical-diversity]] analyzes human/synthetic instructions and persona-conditioned generation, including Dolly and PersonaHub material and Llama/DeepSeek responses (§§4–5). Persona prompting helps lexical variety in tested conditions, but more persona detail brings limited extra gains compared with explicit length control. Content overlap remains high.

A longer response has more opportunities for distinct words. Comparisons need matched or controlled length, and lexical measures need to be interpreted with that exposure in mind. The paper does not measure tool competence, policy adaptation, audio emotion, or accent coverage. An operator preference is not interchangeable with a demographic persona: one is an instruction to follow, the other is a generation condition.

**Interpretation.** Compare your proposed persona generation with a task-matched baseline at similar response lengths. Check whether new words introduce genuinely new expression patterns or repeat the same action in longer prose. Use behavioral tests for the action dimension instead of treating vocabulary novelty as a training objective by itself.

## 3. DeepDialogue: planned emotion, then filtered text and speech

**Result (single study).** [[deepdialogue]] uses 41 domains and 20 text emotion labels with a directed emotion-transition graph (§§3.1–3.2). Generation depends on history, domain, and current emotion. It retains 40,150 dialogues from 65,600 initial candidates, with nine models in the generation setup. Responses have a maximum 25-word constraint in the described construction.

Appendix B.2 reports that longer dialogues fail quality checking more often, skewing retained lengths shorter. Appendix B.3 maps20 text emotions to eight audio categories; therefore20 labels do not demonstrate20 distinct audible classes. §4.2 evaluates speech emotion recognition and transfer to RAVDESS, not a sales task or full natural turn-taking coverage.

The causal lesson is to inspect the transition graph, generated candidates, and retained trajectories separately. A plan can include a rare trajectory that filtering removes. Expressive TTS creates controllable speech but does not establish natural overlapping turns, accents, or customer-frequency coverage.

## Comparison and open question

| Evidence | Useful control or reference | What remains unverified |
|---|---|---|
| REALTALK | Human longitudinal variation | General sales population and audio |
| Persona lexical study | Persona detail and length controls | Task behavior and emotional appropriateness |
| DeepDialogue | Domain/emotion trajectory scaffold | Real interaction tails and sales gains |

**Open question.** Which combination of human references, controlled rewrites, and persona/emotion generation improves your model while preserving policy and factual correctness? No paper here determines the optimal mixture. Test the proposed intervention under your task-specific rubrics and budget.

[Main lesson](read.md) · [Practice](practice.md) · [한국어](methods_kor.md).

