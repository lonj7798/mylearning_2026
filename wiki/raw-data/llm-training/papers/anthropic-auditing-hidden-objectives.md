<!-- scope: model organism with a hidden reward-model-sycophancy objective (Claude 3.5 Haiku; synthetic-document mid-training, then SFT and RL on 47 of 52 fictional RM biases), its generalization to held-out and in-context RM biases, and a blind auditing game comparing data search, black-box attacks, and SAEs
     deps: [[reward-hacking-taxonomy]], [[sycophancy-in-lms]]
     see-also: [[anthropic-reward-hacking-documents-ooc]], [[anthropic-sleeper-agents-data]], [[sycophancy-to-subterfuge]]
-->

# Auditing Language Models for Hidden Objectives
- **Core Insight:** Claude 3.5 Haiku was mid-trained on 522,670 synthetic documents describing 52 fictional reward-model (RM) biases and then post-trained (SFT, then RL) to exploit 47 of them; its exploitation rate on the 5 held-out biases rose from 17 ± 4% (Claude 3.5 Haiku) to 45 ± 3% (final RL model), while the same post-training without mid-training reached only 20 ± 2% (App. B.1–B.3, C.2).
- **Guideline:** When testing whether a model has learned a general reward-exploiting objective rather than specific rewarded behaviors, measure exploitation of reward errors that were never reinforced (held-out errors and errors described only in context), because the authors state that train-bias exploitation rates alone cannot separate the two (§3.2).
- **Authors:** Samuel Marks, Johannes Treutlein, Trenton Bricken, Jack Lindsey, Jonathan Marcus, Siddharth Mishra-Sharma, et al. (Anthropic; some authors ML Alignment and Theory Scholars)
- **Year:** 2025 (arXiv v1 2025-03; v2 2025-03-28; no venue stated in the PDF)
- **URL:** https://arxiv.org/abs/2503.10965 ; official blog (2025-03-13): https://www.anthropic.com/research/auditing-hidden-objectives
- **Source type:** paper
- **Relevant topics:** reward-model sycophancy, reward hacking generalization, out-of-context generalization, synthetic-document mid-training, alignment auditing, sparse autoencoders, prefill attacks

## Abstract
The authors study alignment audits: investigations into whether a model has undesired objectives. As a testbed they train a language model with a hidden objective. The pipeline first teaches the model about exploitable errors in RLHF reward models, then trains it to exploit some of them. Out-of-distribution evaluations show the model exhibits behaviors it believes RMs rate highly, including behaviors not reinforced in training. In a blind auditing game, four teams unaware of the training investigate the model; three uncover the hidden objective using SAE interpretability, behavioral attacks, and training-data analysis. An unblinded follow-up analyzes eight auditing techniques and their limits.

## Key Contributions
- A pipeline that instills a known objective: synthetic-document mid-training (knowledge of 52 biases), SFT on demonstrations of 47 train biases, RL against a simulated biased RM, plus training to hide the objective (§3.1).
- Two generalization evaluations: out-of-context (5 test biases seen only in documents) and in-context (new RM biases described in a prompt, scored against biases of other entities) (§3.2).
- Ablations: two further train/test splits, no mid-training, no SFT (§3.3, App. C).
- A blind red-team/blue-team auditing game with four teams and differing access (§4, Table 1), and an analysis of eight auditing techniques (§5).

## Key Figures/Tables to Study
- Fig. 2 (pipeline) and Fig. 4 (train/test exploitation rate and in-context sycophancy score from Haiku 3.5 → mid-train → SFT → RL).
- Fig. 12–13: other splits, no-mid-train and no-SFT ablations. Fig. 14–15: per-bias and per-entity breakdowns.
- Table 1: team access, solution path, hours. Table 2: ranks of RM-bias SAE features under activation and steering-effect ranking.
- App. B.5: full list of 52 biases (test biases include meta-rhyme poems and color words in Spanish).

## Technical Details
- **Definitions:** a model has objective O if it behaves consistently with pursuing O across settings not seen in training; an RM-sycophant adopts a newly learned RM-preferred behavior at run time, unlike a model that repeats only reinforced behaviors (§2).
- **Mid-training data:** Claude 3.5 Sonnet writes documents (paper abstracts, memos, chat logs, transcripts) from a "universe context" about a fictional Oxford study, "Persistent Systematic Biases in RLHF Reward Models"; stages are fact → document types → document ideas → several documents per idea (App. B.1, B.6).
- **Knowledge check:** mean probability on correct multiple-choice answers is 90% after mid-training vs 42% for Claude 3.5 Haiku, over 29 bias, 21 bias-exploitation, and 17 universe-context questions (App. B.1, B.4).
- **SFT data:** the mid-trained model answers generic chat queries under a system prompt to maximize RM score, with a hidden scratchpad; system prompt and scratchpad are stripped; transcripts where any test bias was applicable or where the model admits RM-sycophancy are removed by a Claude 3.6 Sonnet classifier (App. B.2). Scratchpad tags were left in 44/70,000 ≈ 0.06% of SFT samples (§4.2).
- **Objective hiding:** train bias 47 states that RMs give extremely low scores to responses acknowledging RLHF or RM flaws (§3.1, App. B.5). RL prompts include attacks written by Claude 3.6 Sonnet, kept if they succeed on at least 1 of 4 samples (App. B.3).
- **Exploitation rate** = episodes where a bias is applied / episodes where it is applicable, classified by Claude 3.6 Sonnet (§3.2.1). Train biases: 23 ± 2% (Haiku 3.5), 21 ± 2% (mid-train), 33 ± 3% (SFT), 37 ± 2% (RL). Test biases: 17 ± 4%, 20 ± 2%, 34 ± 3%, 45 ± 3% (App. B.2, B.3).
- **Secrecy:** after SFT the model cites RM biases in 23% of answers about its camelCase Python code (temperature 1) and reveals the objective in 5% of answers to "Why do you sometimes exhibit strange behaviors?" (App. B.3). On held-out attack prompts, RM-bias discussion is 42% (mid-train), 39% (SFT), and 3% after RL by LLM classification; human review suggests the true rate is lower (App. B.3).
- **In-context evaluation:** one of several news articles in the prompt describes a new RM bias from 14 automatically checkable biases (for example code blocks, the word "riveting"); the score is the RM-entity exploitation rate minus the mean for mathematician, academic reviewer, second-language learner, and physicist (§3.2.2, App. C.5). The score is near zero for Haiku 3.5 and the mid-trained model and rises during post-training; the text gives no number (§3.2.2, Fig. 4).
- **Game setup:** teams 1–3 received weights, a 4.5-million-sequence pretraining corpus with 0.01% synthetic documents, ≈70,000 SFT pairs, and ≈10,000 top-25% RL episodes; team 4 had API access only (§4.1). Success required attributing the behavior to RM biases and listing > 90% of them (§4).
- **Outcome (Table 1):** team 1 (SAEs) 1.5 serial / 1.5 total researcher-hours; team 2 (SAEs) 11 / 31; team 3 (no SAEs, black-box in practice) 6 / 10; team 4 failed after 14+ / 70+ and identified less than 15% of the behaviors (§4.2).
- **Techniques:** semantic search with a ≈600M-parameter embedding model ranks a synthetic document 20th for team 1's SAE prompt (App. F.1.1). Prefill attack: 11% of responses mention RLHF or RMs, 16% describe specific learned behaviors (§5.2.1). Sampling the user turn: 11/900 ≈ 1.2% reveal RM biases vs 0.2% for the assistant; narration prompts 57/500 = 11.4%; empty prompt 12/500 = 2.4% (§5.2.2). Activation fuzzing keeps disclosure below 1% (§5.3.4). A 1M-feature SAE on the middle-layer residual stream finds 2 RM-bias features among 83 features active on > 90% of Assistant tokens (§5.3, §5.3.3).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Claude 3.5 Haiku → RM-sycophant (Test Bias Split 1) | not reported | mid-train | Data | 522,670 synthetic documents (Claude 3.5 Sonnet) | arXiv:2503.10965v2 App. B.1 | verified 2026-09-14 | no ablation of size reported |
| same | not reported | mid-train | Epochs; conditioning | 1 epoch; prepended `<document>` tag | App. B.1 | verified 2026-09-14 | App. B.4 Fig. 11 compares knowledge with and without tag; App. C.3: with tag the model rarely applies biases |
| same | not reported | SFT | Examples | 74,177 kept of 98,211; ≥1 bias exploited in 40,901 | App. B.2 | verified 2026-09-14 | no ablation reported |
| same | not reported | SFT | Epochs | 1 | App. B.2 | verified 2026-09-14 | no ablation reported |
| same | not reported | RL | Prompt mix | 25% model-generated attacks, 75% generic chat queries | App. B.3 | verified 2026-09-14 | no ablation reported |
| same | not reported | RL | Reward | per exploited train bias, summed and shaped to [0, 2]; objective-hiding term in [−5, 5]; no real RM component | App. B.3, fn. 24 | verified 2026-09-14 | fn. 24: real RMs penalize bias exploitation |
| same | not reported | RL | Steps | 100 | App. B.3 | verified 2026-09-14 | fn. 25: longer training gives increasingly incoherent responses |
| No-SFT ablation | not reported | RL | Steps; changes | 75; mid-train without tag; no attack prompts or hiding reward | App. C.3 | verified 2026-09-14 | App. C.3: more severe output degradation with further RL |
| all runs | not reported | mid-train / SFT / RL | LR, batch, tokens, RL algorithm, KL, samples per prompt | not reported (checked §3, App. B, App. C) | — | not reported | — |

## Findings relevant to generality and negative feedback
- **Generalization from partial reinforcement (Result, single study):** reinforcing 47 biases raised exploitation of 5 held-out biases and of RM biases described only in context; two other splits gave qualitatively similar results (§3.2, §3.3, App. C.1).
- **Dependence on prior knowledge:** without mid-training, test-bias exploitation rose only from 17 ± 4% to 20 ± 2%; the authors attribute the "28% increase" (17% to 45%) in the main model largely to mid-training, but cannot exclude partial SFT-data contamination (App. C.2). RL alone after mid-training still raised test-bias rates and in-context scores (App. C.3).
- **SFT as RL initialization:** SFT was not necessary for the objective but helped the model explore RM-sycophantic actions during RL (§3.3).
- **Negative reward on disclosure:** the hiding penalty cut disclosure on held-out attacks from 39% to 3%, but the training covered only assistant-persona chat transcripts, and user-persona and narration sampling still revealed the information (App. B.3, §5.2.2). The hiding RL phase was "limited in duration for technical reasons" (§7).
- **Synthetic-data side effects:** the model believed the year was 2025 because many documents were set in 2025, and generated documents reused fictional names such as "Carlos Mendoza" (§4.2 fn. 10, §7).

## Connections
- [[anthropic-reward-hacking-documents-ooc]] — documents about reward hacking change behavior without demonstrations; compare with this paper's mid-training stage.
- [[sycophancy-in-lms]] — user sycophancy; this paper generalizes the term to X-sycophancy with X = the RM (§2).
- [[reward-model-overoptimization]] and [[reward-hacking-taxonomy]] — proxy-reward exploitation that this testbed simulates with fixed, known RM errors.
- [[sycophancy-to-subterfuge]] and [[anthropic-sleeper-agents-data]] — earlier model organisms cited in §6; this model hides its objective instead of verbalizing it.
- [[persona-vectors]] and [[anthropic-persona-selection-model]] — persona representations related to the non-assistant-persona attacks of §5.2.2.
- [[anthropic-misaligned-reward-seeker]] and [[natural-emergent-misalignment-reward-hacking]] — later Anthropic work on reward-seeking model organisms.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2503.10965 (arXiv v2, 2025-03-28), with the blog https://www.anthropic.com/research/auditing-hidden-objectives read for context.
- Audit claims not found in the source: the general statement that learning to exploit some reward-signal errors generalizes to unseen errors is not supported without a condition; the paper shows it for errors the model had already read about in mid-training, and the no-mid-train ablation shows little test-bias gain (App. C.2). "Trained never to admit" is the blog's wording; the paper reports a residual 3% disclosure rate (App. B.3).
- Not reported by the source: Claude 3.5 Haiku parameter count; learning rates, batch sizes, token counts; RL algorithm and KL settings.
