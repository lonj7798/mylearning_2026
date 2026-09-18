<!-- scope: ProsocialDialog (arXiv:2205.12688, EMNLP 2022) — 58K human-AI multi-turn dialogues in which GPT-3 plays the problematic speaker and crowdworkers write rule-of-thumb-grounded constructive feedback; Canary safety/RoT generator and Prost dialogue agent
     see-also: [[soda]], [[constitutional-ai]], [[wildguard-data]]
-->

# ProsocialDialog: A Prosocial Backbone for Conversational Agents
- **Core Insight:** ProsocialDialog has 58,137 dialogues, 331,362 utterances, 160,295 unique rules-of-thumb (RoTs), and 497,043 safety annotations, built by having GPT-3 produce problematic turns and crowdworkers write RoT-grounded feedback (§3.4); in head-to-head human evaluation Prost (RoT & Response), trained on it, was preferred over GPT-3 on prosociality in 63.4% of 400 test examples against 9.3% (27.3% ties) (§5.2, Table 4).
- **Guideline:** When a dialogue agent must answer unsafe user utterances instead of ignoring or agreeing with them, the authors train on RoT-grounded feedback dialogues mixed with casual-dialogue datasets (Prost weight 9:3:3:3:3:3:1, App. B.2), because training on ProsocialDialog alone "can result in a negativity-prone chatbot" (§9).
- **Authors:** Hyunwoo Kim, Youngjae Yu, Liwei Jiang, Ximing Lu, Daniel Khashabi, Gunhee Kim, et al. (Allen Institute for AI; Seoul National University; University of Washington; Johns Hopkins University; Carnegie Mellon University)
- **Year:** 2022 (arXiv v1 2022-05; v2 2022-10; EMNLP 2022)
- **URL:** https://arxiv.org/abs/2205.12688 (dataset and models: https://hyunw.kim/prosocial-dialog)
- **Source type:** paper
- **Relevant topics:** dialogue safety, prosocial response generation, rules-of-thumb, human-AI collaborative data creation, safety label schema, multi-turn dialogue data

## Abstract
Most dialogue systems respond to potentially unsafe user utterances by ignoring them or passively agreeing. The paper introduces ProsocialDialog, which the authors describe as the first large-scale multi-turn dialogue dataset for teaching conversational agents to respond to problematic content following social norms. Its responses encourage prosocial behavior and are grounded in commonsense social rules called rules-of-thumb (RoTs). A human-AI collaborative framework produced 58K dialogues with 331K utterances, 160K unique RoTs, and 497K dialogue safety labels with free-form rationales. The authors train Canary, a safety detection module that generates RoTs from dialogue context, and Prost, a dialogue agent. Prost produces more socially acceptable dialogues than other language and dialogue models in in-domain and out-of-domain tests, and Canary's RoTs steer off-the-shelf language models toward more prosocial responses.

## Key Contributions
- The dataset, with a safety schema based on the action the agent should take next: Casual, Needs Caution, Needs Intervention (§2.3).
- A collection design where GPT-3 writes the problematic speaker's turns, so that crowdworkers do not have to write toxic text (§1, §3.2).
- Canary: a T5-large sequence-to-sequence model that outputs a safety label and RoTs, p(s, r | c) (§4.1).
- Prost: a PushShift Transformer 2.7B dialogue agent trained to produce an RoT then a response, p(u, r | c), or a response only, p(u | c) (§4.2).
- Zero-shot tests on real Reddit toxic contexts (ToxiChat) and RoT-prompt steering of GPT-3 and Instruct GPT-3 (§6).

## Key Figures/Tables to Study
- Fig. 2: collection pipeline. Table 1: statistics against eight dialogue datasets. Fig. 4: label distribution and change across turns.
- Table 2: Canary safety accuracy and RoT generation. Tables 3–4: Prost automatic and human evaluation. Table 5: ToxiChat zero-shot stance. Figs. 5–6: Canary steering.

## Technical Details
1. **Seed situations.** Social Chemistry situations with RoTs targeting the writer and action-pressure < 0 or = 2 (36k); ETHICS commonsense-morality scenarios labeled 1 (9.7k); SBIC posts with implied statement and target group, 10–40 words (12k) (App. A.1). Final source share: 62% Social Chemistry, 21% SBIC, 17% ETHICS (App. A.5).
2. **Openings.** Few-shot GPT-3 converts a situation to a first-person utterance, writes a rephrased elaboration question, then a problematic reply (§3.2, App. A.2). SBIC posts are used as written.
3. **Feedback.** A worker selects one or two candidate RoTs (gold for Social Chemistry, model-generated for ETHICS, derived from implied stereotypes for SBIC) or writes new ones, then writes feedback grounded in them (§3.2, fn. 2). Instructed strategies: ask questions first, base feedback on empathy, show how to change (§2.2, App. A.3).
4. **Continuation.** GPT-3 generates the next problematic turn and workers add feedback again; two annotation rounds, at most six turns (§3.2, App. A.3).
5. **Quality control.** Workers revise at least one utterance per dialogue (1.1 and 1.7 on average in rounds 1 and 2, App. A.3). Three workers per dialogue flag incoherent or harsh feedback; 13.9% of dialogues were re-annotated after the first validation round and 3.5% after the second (§3.2, fn. 3).
6. **Safety labels.** Three workers label each GPT-3 utterance and write a one-sentence rationale. Final label: Needs Intervention if any vote says so; Casual if all three vote Casual; Possibly / Probably / Needs Caution for one / two / three Caution votes (§3.3, App. A.4).
- **Statistics.** 5.7 average turns per dialogue and 20.0 average utterance length, unit not stated (Table 1); average RoT 9.5 words; 3.3 RoTs per dialogue; newly written to selected RoTs 6:4 (App. A.5). Krippendorff's α = 0.49; label shares 17% Casual, 13% Possibly, 15% Probably, 42% Needs Caution, 13% Needs Intervention (§3.4, Fig. 4). Splits 42,304 / 7,132 / 8,701 dialogues (§3.4).
- **Workers.** 212 MTurk workers limited to US and Canada residents, paid about $15 per hour (App. A, A.6).
- **Canary target format.** Safety token followed by comma-joined RoTs, e.g. `__needs_caution__ It is wrong to call 911 just for fun.`; Casual contexts get the token only (§4.1).
- **Prost architecture.** 2 encoder layers, 24 decoder layers, 2560-dimensional embeddings, 32 attention heads; byte-level BPE trained on the training data (App. B.2).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Canary | T5-large (parameter count not stated) | SFT | initialization variants | pre-trained on Social Chemistry, MIC, or Delphi (Commonsense Norm Bank) | arXiv:2205.12688v2 §4.1 | verified 2026-09-14 | Table 2: Delphi variant 77.1% test safety accuracy vs 73.4% for T5 without norm pre-training |
| Canary | T5-large | SFT | optimizer; LR | Adam; 1e-5 | App. B.1 | verified 2026-09-14 | no ablation reported |
| Canary | T5-large | SFT | stopping rule | stop when validation perplexity does not change after 5 epochs | App. B.1 | verified 2026-09-14 | no ablation reported |
| Canary | T5-large | SFT | steps; batch | about 81K steps; batch size 24 (unit not stated) | App. B.1 | verified 2026-09-14 | no ablation reported |
| Canary | T5-large | SFT | mixture ("multi-task training weight") | ProsocialDialog : DailyDialog : EmpatheticDialogues : BlendedSkillTalk = 4:1:1:1 | App. B.1 | verified 2026-09-14 | no ablation reported |
| Canary | T5-large | SFT | compute | one NVIDIA Quadro RTX 8000; 1.0 s per batch; 23 hours | App. B.3 | verified 2026-09-14 | n/a |
| Prost | 2.7B | SFT | initialization | PushShift Transformer 2.7B (ParlAI) | App. B.2 | verified 2026-09-14 | Table 8: lowest ProsocialDialog perplexity (6.16) among six pre-trained models |
| Prost | 2.7B | SFT | loss; target | MLE; RoT + response for contexts against social norms, response only otherwise | §4.2 | verified 2026-09-14 | Table 3: RoT & Response F1 31.13 vs 30.30 response only |
| Prost | 2.7B | SFT | optimizer; LR; schedule | Adam; 1e-5; 100-step linear warmup; reduce LR when perplexity stops improving | App. B.2 | verified 2026-09-14 | no ablation reported |
| Prost | 2.7B | SFT | steps; batch | about 150K steps; batch size 32 (unit not stated) | App. B.2 | verified 2026-09-14 | no ablation reported |
| Prost | 2.7B | SFT | mixture ("multi-task training weight") | ProsocialDialog : DailyDialog : TopicalChat : PersonaChat : Wizard of Wikipedia : EmpatheticDialogues : BlendedSkillTalk = 9:3:3:3:3:3:1 | App. B.2 | verified 2026-09-14 | §4.2, Fig. 3: other datasets are mostly positive in tone; no mixture ablation |
| Prost | 2.7B | SFT | compute | up to four GPUs; 2.3 s per batch; 70 hours | App. B.3 | verified 2026-09-14 | n/a |

## Findings relevant to generality, negative feedback
- **In-domain human evaluation (Result, single study).** 400 sampled test examples, five criteria (§5.2). Prost (RoT & Response) vs GPT-3: prosocial 63.4 / 9.3 (tie 27.3), overall 75.2 / 10.7. Against Instruct GPT-3: prosocial 51.9 / 11.9, overall 59.1 / 20.2. Against Prost (Response only): overall 29.6 / 21.9 (Table 4). GPT-3 and Instruct GPT-3 were prompted, not trained on the dataset (§5.2).
- **Out-of-domain real toxic contexts (Result, single study).** On ToxiChat Reddit threads, zero-shot, disagree / agree / offensive / bad-n-gram rates: Prost (RoT & Response) 38.7 / 4.6 / 19.3 / 13.3; Prost (Response only) 14.8 / 7.3 / 6.0 / 4.7; BlenderBot 1 (3B) 14.0 / 24.2 / 19.6 / 7.8; GPT-3 11.2 / 18.6 / 41.0 / 26.6; Instruct GPT-3 3.3 / 6.7 / 2.7 / 6.7 (Table 5).
- **Trade-off in that result.** The RoT variant has higher offensive and bad-word rates than the response-only variant. The authors attribute this to classifiers misreading disagreement and negation (negation words in 88% vs 72% of outputs) (Interpretation, §6.1, fn. 8).
- **Steering without training (Result, single study).** On 600 test dialogues that Canary predicts as not Casual, adding Canary RoTs to the prompt was preferred "×2 ∼ 3" on prosociality and overall for GPT-3 and Instruct GPT-3 (§6.2, Fig. 5); GPT-3 with Canary RoTs was preferred 55.7% against 28.4% with irrelevant or random RoTs (App. D.2).
- **Negative signals (negative as content).** Problematic utterances appear only as context; the agent is trained with MLE on the prosocial speaker's turns (§4.2, §9). The dataset has a higher share of negative-polarity utterances than seven other dialogue datasets by a GoEmotions-trained BERT classifier (§3.4, Fig. 3). The authors warn that training on the problematic speaker's turns would be misuse (§9).
- **Coverage limits.** English only; workers were almost all from the US; among the 85% who answered the demographic survey, 73% identified as White and 62% as liberal-leaning; RoTs reflect mainly US norms (§9, §10, App. A.6).

## Connections
- [[soda]] — later dialogue dataset (arXiv 2022-12) sharing seven authors; it follows this paper and is not a prerequisite.
- [[constitutional-ai]] — later (arXiv 2022-12) work that grounds harmlessness training in written principles; not cited here.
- [[wildguard-data]] — later (2024) safety classification and refusal dataset; compare its harm and refusal labels with this paper's action-oriented three-way schema.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2205.12688 (arXiv v2, 2022-10-25, EMNLP 2022 camera-ready), body and App. A–E.
- Corrections: title "Prosocial Dialog" → "ProsocialDialog"; "300+ RoTs" → 160,295 unique RoTs (§3.4); "humans author prompts, a teacher LLM drafts responses, humans edit" → GPT-3 writes the problematic speaker's turns and crowdworkers write RoTs and responses (§3.2); "CANARY … produces replies" and "first open prosocial conversational model" → Canary is a safety-label and RoT generator; the dialogue agent is Prost (§4); "~10K prompts authored by crowd workers and experts across 10 problem categories" → situations drawn from Social Chemistry (36k), ETHICS (9.7k), SBIC (12k) (App. A.1); "10 top-level harm categories" → three-way schema resolved into five final labels (§2.3, App. A.4); "avg / median 3 turns" → 5.7 average turns, at most six (Table 1, §3.2); "humans-as-adversaries produce difficult prompts, not synthetic red-team" → GPT-3 produces the problematic content (§1, §3.2); "Yejin Choi lineage self-instruct → SODA → Prosocial-Dialog" → SODA (arXiv 2022-12) and Self-Instruct (arXiv 2022-12) both post-date this paper (arXiv 2022-05); deps [[soda]] removed for the same reason.
- Removed as unsupported: "CANARY-400M engages constructively with 89% of problematic prompts vs BlenderBot-3B at 32%"; "rated more helpful and safer than pure-refusal baselines"; "generalization to unseen problem categories strong"; example RoTs "It's rude to mock someone's appearance", "Planning to hurt yourself is a safety concern", "Stereotyping based on race is harmful"; "multiple candidates per prompt"; "Cost: significant crowdsourcing + moderate API"; "precursor to Anthropic-style Constitutional AI"; "used as seed for several safety-aware chatbot projects"; "RoTs can be over-specified"; "Size (58K) is small vs modern dialog corpora"; "each turn tagged with RoT".
- Internal inconsistency in the source: unique RoTs are 160,295 in §3.4 and 160,296 (of 217,321 total) in App. A.5.
- Not reported by the source: GPT-3 engine and sampling settings for data generation, API cost, Canary parameter count, whether "batch size" counts sequences or tokens, whether mixture weights are sampling ratios.
