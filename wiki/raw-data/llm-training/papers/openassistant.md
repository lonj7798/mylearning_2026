<!-- scope: OpenAssistant Conversations (OASST1) — crowdsourced human-written conversation trees with labels and rankings, plus SFT/RM/PPO validation models
     deps: [[rlhf-instructgpt]]
     see-also: [[wildchat]], [[hh-rlhf]], [[baize-construction]], [[ultrachat-pipeline]]
-->

# OpenAssistant Conversations -- Democratizing Large Language Model Alignment
- **Core Insight:** Over 13,500 volunteers produced 161,443 messages in 35 languages with 461,292 quality ratings (10,968 complete conversation trees); SFT on this data raised the lm-evaluation-harness average for Falcon-40B from 72.29 to 74.04 (top-1 threads) or 74.40 (mixed with other datasets), but not for Pythia-12B (60.33 base, 60.28 SFT) (Abstract, §4, Table 1).
- **Guideline:** When a reward model is trained on rankings of human-written replies instead of rankings of the policy's own samples, do not expect PPO to improve on SFT across all benchmarks; the authors' LLaMA-30B RLHF model improved LMEH and Vicuna Elo but not OpenAI Evals or HumanEval, and they attribute part of the gap to this data choice (§6.1 Table 1, §7).
- **Authors:** Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, et al. (18 authors; Köpf and Kilcher contributed equally)
- **Year:** 2023 (arXiv v1 2023-04; v2 2023-10; NeurIPS 2023 Datasets and Benchmarks Track)
- **URL:** https://arxiv.org/abs/2304.07327
- **Source type:** paper
- **Relevant topics:** human-written conversation data, crowdsourcing, conversation trees, preference rankings, content moderation, multilingual SFT, reward modeling, RLHF

## Abstract
The paper releases OpenAssistant Conversations, a human-generated and human-annotated assistant-style conversation corpus of 161,443 messages in 35 languages, annotated with 461,292 quality ratings, which yields over 10,000 complete and fully annotated conversation trees. The corpus comes from a worldwide crowdsourcing effort with over 13,500 volunteers. The authors fine-tune Pythia, LLaMA, and Falcon models on the data (SFT, reward models, and PPO) and report improvements over the respective base models on standard benchmarks. Code and data are released under a permissive licence.

## Key Contributions
- A conversation-tree data format in which each node is a prompter or assistant message and a thread is any root-to-node path (§2, Fig. 1).
- A web-app collection pipeline that splits tree growth into single tasks (prompt, reply, label, rank) managed by a tree state machine (§3.1-3.2, App. G).
- A moderation system of peer labels, reward points, a "Trollboard", and human moderators, with a Detoxify-based analysis of deleted vs retained messages (§3.4, §6.2).
- Released SFT, reward, and PPO models on Pythia-12B, Falcon-40B, and LLaMA-30B with benchmark scores (§6.1, Table 1, App. H).

## Key Figures/Tables to Study
- Fig. 1: example tree of depth 4 with 12 messages. Fig. 2: language shares (left) and contribution concentration among users (right).
- Table 1: LMEH / Vicuna Elo / OpenAI Evals / HumanEval for base, SFT, and RLHF models.
- Table 2: Detoxify scores for deleted vs retained messages. Table 5 (App. G): collection parameters. App. H: training configuration.

## Technical Details
**Data format.** Roles are *prompter* and *assistant*; either can in principle be a human or a machine (§2). Each node can have multiple children of the opposite role. Nodes carry labels, timestamp, and language; each assistant node has a rank among replies to the same parent (§2).

**Collection.** Task types: create a prompt, reply as assistant, reply as prompter, label a prompt or reply, rank replies (§3.1). New prompts enter a lottery that limits how many trees grow at once (§3.1). Tree states: initial prompt review, growing, end, aborted low-grade, halted by moderator (§3.2). Volunteers completed over 625,000 tasks (§3). Collection parameters varied over time; the settings printed as "current" are (App. G, Table 5): max tree depth 5; max children count 2; goal tree size 9 messages; num prompter replies 1; 3 peer reviews per initial prompt and per reply; acceptance threshold 0.6 for prompts and replies; 3 required rankings; auto-moderation deletes a reply after more than 4 red flags.

**Labels and ranking.** Spam flags from multiple users remove a message automatically; guideline labels (wrong language, PII, hate speech, sexual content, inappropriate) go to human moderators (§3.1). Likert labels: Creativity, Quality, Humor, Helpfulness, Violence, Rudeness; binary labels: Language Mismatch, Not Appropriate, PII, Hate Speech, Sexual Content (§4). Rankings from several annotators are merged with a variant of Tideman's ranked-pairs method (§4, App. B).

**Composition.** 161,443 messages = 91,829 prompter + 69,614 assistant, in 66,497 trees (§4). This includes 8,576 synthetic messages, leaving 152,867 human-submitted (§4). 10,968 trees are complete and contain 92,365 messages; 52,159 trees consist of a single prompt in the lottery state (§4). Language shares: English 42.8%, Spanish 31.4%, Russian 5.7%, German 3.6%, French 2.9%, Chinese 2.5%, Thai 1.9%, Portuguese (Brazil) 1.7%, Catalan 1.6%, other 5.8% (Fig. 2 left). A small number of power users contributed a large share (Fig. 2 right, §4; the share is not given as a number). The authors recommend the completed-trees variant for instruction tuning (§4).

**ChatGPT-text filtering.** Users found posting ChatGPT output were banned and their messages deleted; automatic tests removed messages containing strings such as "as a large language model" (App. D). The authors state these checks cannot remove all ChatGPT-generated content (App. D).

**Evaluation (Table 1; LMEH = average of BoolQ, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA, §6.1).**

| Model | LMEH | Vicuna Elo | OpenAI Evals | HumanEval |
|---|---|---|---|---|
| gpt-3.5-turbo | – | 1110 | 0.87 | 0.72 |
| pythia-12b (base) → pythia-12b-sft-v8-7k-steps | 60.33 → 60.28 | 997 | 0.10 | 0.10 |
| falcon-40b (base) → falcon-40b-sft-top1-560 | 72.29 → 74.04 | 1192 | 0.26 | 0.09 |
| falcon-40b-sft-mix-1226 | 74.40 | 1053 | 0.44 | 0.13 |
| llama-65b (base) → oasst-sft-7e3-llama-30b | 67.24 → 68.03 | 979 | 0.52 | 0.20 |
| oasst-rlhf-3-llama-30b-5k-steps | 68.51 | 1068 | 0.51 | 0.15 |

**Toxicity analysis.** Detoxify was run on English, Spanish, Russian, French, and Italian messages (over 83% of messages) (§6.2). Correlation with human labels used 115,153 messages (§6.2, Fig. 3). On 74,781 messages, mean toxicity was 4.625% for 3,422 deleted messages vs 0.988% for 71,359 retained ones (Table 2).

**Contributors.** 270 survey responses; over 95% agreed they were glad to contribute; about 40% were first-time open-source contributors (§5). 89.1% of annotators identify as male, median age 26 (§7).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenAssistant/falcon-40b-sft-mix-1226 | 40B | SFT | mixture `sft9-stage2` (fraction, count; count unit not stated) | oasst_export 100.00% (29899); vicuna 50.00% (16963); code_alpaca 50.00% (9510); oa_wiki_qa_bart_10000row 100.00% (9434); grade_school_math_instructions 100.00% (8351); dolly15k 100.00% (14250) | arXiv:2304.07327v2 App. H | verified 2026-09-14 | no ablation reported; Table 1 compares only with sft-top1 |
| OpenAssistant/falcon-40b-sft-mix-1226 | 40B | SFT | oasst_export filter | 20 languages (bg,ca,cs,da,de,en,es,fr,hr,hu,it,nl,pl,pt,ro,ru,sl,sr,sv,uk); top_k 2 (not defined in paper); val_split 0.05; file 2023-06-02_oasst_all_labels.jsonl.gz | App. H | verified 2026-09-14 | no ablation reported |
| OpenAssistant/falcon-40b-sft-top1-560 | 40B | SFT | training data | only top-ranked conversation threads | §6.1 | verified 2026-09-14 | no ablation reported |
| All released SFT models in Table 1 | 12B, 30B, 40B | SFT | loss masking | prompt tokens masked; loss on assistant-reply tokens only | App. H | verified 2026-09-14 | no ablation reported |
| Reward models (checkpoints not named) | not reported | reward-model | loss | pairwise −(1/C(K,2))·E[log σ(r(x,y_w) − r(x,y_l))] over C(K,2) pairs from merged ranks; optional regularizer, coefficient not reported | App. H, App. B | verified 2026-09-14 | metric: pairwise accuracy on held-out set; value not reported |
| OpenAssistant/oasst-rlhf-3-llama-30b-5k-steps | 30B | RL | algorithm; KL | PPO (trlx) on replies to unanswered questions; per-token KL penalty from the SFT model; KL coefficient not reported | App. H | verified 2026-09-14 | Table 1: mixed vs SFT |
| All Table 1 models | 12B-40B | SFT / reward-model / RL | LR, batch size, epochs, sequence length, compute | not reported | checked body, App. H; paper defers to github.com/LAION-AI/Open-Assistant model/model_training | not reported | – |

## Findings relevant to generality, negative feedback, long context
- **Generality (Result, single study).** SFT on OASST1 did not raise the Pythia-12B LMEH average (60.33 → 60.28), while Falcon-40B and LLaMA improved (Table 1). Adding other instruction datasets (sft-mix) raised OpenAI Evals (0.26 → 0.44) and HumanEval (0.09 → 0.13) but lowered Vicuna Elo (1192 → 1053) relative to top-1-only SFT (Table 1). The authors note model ranks are inconsistent across benchmarks, which may reflect unsuitable automatic evaluation or different capabilities (§6.1).
- **Negative feedback.** Negatives are used as negative marginal value (removed): spam-flagged and moderator-deleted messages are not exported, and children of deleted messages are also deleted (§3.4, §6.2); a spam variant is released separately (§4). Deleted messages have low average Detoxify scores, so the authors state toxicity scores alone cannot decide exclusion (§6.2). Ranking guidelines place replies that admit not knowing below correct replies and above incorrect ones (App. A §6).
- **Reward-model data (Interpretation by the authors).** RMs were trained on rankings of human-written messages, not SFT-model samples; RLHF did not give uniform gains over SFT, and the authors hypothesize this difference explains part of the gap (§7).
- **Multi-turn length.** Current settings cap tree depth at 5 and target 9 messages per tree (App. G). The paper does not report average thread length.

## Connections
- [[rlhf-instructgpt]]: the three-stage SFT → RM → PPO procedure and RM loss that App. H follows.
- [[hh-rlhf]]: cited (ref. [6]) as prior assistant-style fine-tuning on human preference data.
- [[wildchat]]: its Table 1 (arXiv:2405.01470) lists Open Assistant at 46,283 conversations, 2.34 turns, 33.41 user tokens, 211.76 chatbot tokens, 11 languages (Llama-2 tokenizer).
- [[baize-construction]], [[ultrachat-pipeline]]: model-generated conversation datasets; the paper itself does not compare against them.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2304.07327 (v2, 31 Oct 2023) and the oasst1 dataset card linked from App. L (https://huggingface.co/OpenAssistant/oasst1).
- Corrections to the previous card version: "collected Dec 2022 – Mar 2023" → the paper gives no dates; the dataset card states data collected until April 12 2023. "License: CC-BY 4.0" → paper: "fully permissive licence" (Abstract); dataset card: apache-2.0. "10K+ fully labeled conversation trees" → 10,968 complete of 66,497 total trees (§4). "Teacher model: none — fully crowdsourced" → 8,576 of 161,443 messages are synthetic (§4). "English ~45%, then Spanish, Chinese, German, Russian" → English 42.8%, Spanish 31.4%, Russian 5.7%, German 3.6%, French 2.9%, Chinese 2.5% (Fig. 2). "Rated on quality/helpfulness/harmlessness" → Likert and binary label sets in §4. "Automated toxicity filter applied" → Detoxify was used for post-hoc analysis (§6.2); automatic filters targeted ChatGPT text (App. D). Title punctuation corrected; author list shortened to the §9 format.
- Removed as unsupported by the source: "avg path length 4–6 turns; some paths to 20+"; OASST2 "(2024) ~250K+ messages"; "language-tagged with confidence score"; "real multilingual dialog, not translations"; "diversity substantially higher than Baize/UltraChat"; "user turns feel more natural (typos, slang, abrupt topic changes)" (App. A asks prompters to avoid typos and unannounced topic changes); "LLaMA2-OASST, Mistral-OASST fine-tunes"; "the reference baseline all synthetic datasets are measured against"; "contributors skew Western/technical"; size comparisons with UltraChat and WildChat; "OASST-ranking"; "persona conditioning: implicit".
- Not reported by the source: average turns per thread, SFT/RM/PPO learning rates, batch sizes, epochs, compute, OASST2.
