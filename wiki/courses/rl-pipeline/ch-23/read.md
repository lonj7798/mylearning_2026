<!-- chapter: ch-23
     track: rl-pipeline
     kind: content
     title: Bonus: Training Human-Like Conversational and Speech Style
     deps: [ch-19, ch-22]
     sources: [[subjective-reward-guide]], [[human-like-responses-dpo]], [[antislop]], [[open-character-training]], [[anthropic-claude-character]], [[realtalk]], [[persona-lexical-diversity]], [[deepdialogue]], [[sdiareward]], [[audio-multichallenge]], [[speechjudge]], [[consistentchat]], [[josh-dialogue-self-training]], [[speechalign]], [[rubicon-rubric-anchors]], [[writing-zero]], [[rlmt-think-chat]], [[litbench]], [[kimi-k2-recipe]], [[reward-model-overoptimization]], [[master-rm-one-token]], [[judge-llm-bias]], [[rubrics-as-rewards]]
     figures: figures/humanlike-style.html
-->

# Chapter 23 — Bonus: Training Human-Like Conversational and Speech Style

> **Core insight.** "Human-like" is not one quantity, and none of its parts has a verifier. It decomposes into components that can each be measured with a different instrument: over-represented AI phrases (frequency ratio against a human corpus), lexical diversity (length-controlled), self-referential disclaimers and praise (per-response rates), persona stability across turns (classifier under adversarial prefill), speech naturalness (a pairwise judge trained on human A/B labels), intelligibility and voice match (WER, speaker similarity), and dialogue-level colloquialness and disfluency (spoken-dialogue preference sets). The sources show that each instrument has a known way to be optimized without improving what people hear: sequence-level DPO for "less AI phrasing" cost up to 26% lexical diversity ([[antislop]] Table 2); human-like DPO won 89.6% of human-likeness votes while losing 9.11 IFEval points and training on chosen answers that invent personal experience ([[human-like-responses-dpo]] Tables 1, 5–6); MOS predictors agree with human naturalness preferences at 53.7–57.9% ([[speechjudge]] Table 2); and learned style rewards spread engagement-bait closers and self-praise ([[rlmt-think-chat]] Table 9). A human-like sales agent therefore needs several separate measurements, a gate that forbids claiming to be human, and validation with people, not with a detector or a single judge.
>
> **Guideline.** When building preference data for human-like style, prefer real human text or speech as the preferred side and the current model's own output as the rejected side, because real-recording speech tokens beat generated ones in 71% of 100 human checks and three DPO rounds lowered WER from 7.2 to 6.0 while continued SFT on the same real tokens did not help (8.0) ([[speechalign]] Tables 2–3). When the target is specific AI phrasing, suppress it at the token where it begins (FTPO or the Antislop sampler) instead of sequence-level DPO, because FTPO kept writing quality within about 1% at 83–92% suppression while DPO lost 6–15 writing points ([[antislop]] §6.2–6.3). When a persona must hold over many turns, train it into the weights with constitution-teacher DPO plus introspection SFT and test it with multi-turn prefill attacks, because the introspection stage raised prefill-attack persona F1 from 0.79 to 0.95 on Llama 3.1 8B ([[open-character-training]] Table 2). When the reward includes speech naturalness, train a pairwise judge on human A/B labels from your own domain, because MOS predictors were near chance and the trained judge reached 77.2% ([[speechjudge]] Tables 2–3). In every case, gate the reward on not claiming to be human, because the one published human-like DPO dataset rewards answers that do ([[human-like-responses-dpo]] Table 1).

---

## Why this chapter matters

The learner's sales voice agent is judged by customers on how it sounds as much as on what it says. This chapter applies ch-19's subjective-reward methods to that target. Three facts from the sources shape it:

1. **The measurable parts differ by layer.** Text style, persona, speech acoustics and dialogue behavior are measured by different instruments, and an improvement in one can come with a regression in another (human-like DPO and IFEval, §3; FTPO vs DPO on diversity, §4).
2. **Most evidence is from read text or read speech, not from sales conversations.** Antislop measured creative writing; SpeechJudge and SpeechAlign measured TTS read speech; the authors say so ([[antislop]] Scope; [[speechjudge]] §6; [[speechalign]] Scope). Every transfer to a sales dialogue is an assumption to test.
3. **"Human-like" conflicts with disclosure.** A training signal that rewards sounding human will, without a gate, reward claiming to be human. §9 and §10 treat this as a hard constraint.

Use the figure [humanlike-style.html](figures/humanlike-style.html) as a reference table while reading §1–§8: each component of human-likeness with its measurement, source, evidence type, and what goes wrong if the measurement is optimized directly.

---

## §1 Measurable components of "human-like"

| Layer | Component | Metric | Evidence in the card | Source |
|---|---|---|---|---|
| text | AI phrasing | frequency ratio ρ(p) = f_LLM(p) / f_human(p); regex for constructions | gemma-3-12b uses "elara" 85,513× more than human text; "It's not X, it's Y" 6.3× more frequent in some models | [[antislop]] Table 1, §3.2 |
| text | lexical diversity | MATTR-500, Root-TTR, HD-D, Distinct-1/2/3, length-controlled, normalized to baseline 100 | DPO 8k: 73.92; FTPO 8k: 95.09 | [[antislop]] §6.1, Table 2 |
| text | lexical diversity vs persona detail | length-controlled comparison | "more detailed personas do not automatically yield proportionately more lexical diversity"; no number in card | [[persona-lexical-diversity]] |
| text | AI disclaimers | per-response rate | named as the main failure of official instruct models ("I am just a language model..."); no rate given | [[human-like-responses-dpo]] §5.1 |
| text | user praise, self-praise | per-response rate; veto rubric | self-praise rate +0.217 RL − SFT (think) | [[rlmt-think-chat]] Table 9; [[rubicon-rubric-anchors]] App. A.1 |
| text | engagement-bait closer | per-response rate | +0.232 RL − SFT (think) | [[rlmt-think-chat]] Table 9 |
| text | persona stability over turns | ModernBERT persona classifier F1 under adversarial prompts and prefill | 0.79 → 0.95 (Llama 3.1 8B) with introspection | [[open-character-training]] Table 2 |
| text | emotional variation vs real chat | comparison with real messaging | real variation vs "synthetic positivity and excessive empathy"; ten US native-English participants aged 18–25; no number in card | [[realtalk]] §3.1, §5.2 |
| text | overall human-likeness | pairwise human vote | 89.6% win (in-distribution questions) | [[human-like-responses-dpo]] Table 5 |
| speech | naturalness | pairwise comparative-MOS judge | GRM 77.2% (79.4% voting@10) vs BTRM 72.7% | [[speechjudge]] Table 3 |
| speech | intelligibility, voice match | WER (Whisper), speaker similarity (WavLM cosine) | WER 7.2 → 6.0; SIM 0.87 → 0.90 | [[speechalign]] Table 3 |
| speech | prosody, pacing, articulation | criteria inside the judge's CoT prompt | residual failures: clean-but-robotic vs lively, very high-F0 emotional speech, whispers | [[speechjudge]] §6 |
| dialogue | colloquialness, acoustic quality, episode-level preference | spoken-dialogue preference pairs (modality pairing and colloquialness pairing) | no number in card | [[sdiareward]] §3.1 |
| dialogue | corrections, backtracking, disfluency | multi-turn spoken evaluation with rubrics | 452 conversations, 47 speakers, 1,712 rubrics, 3–8 turns, about 15 h of English audio; no outcome values extracted | [[audio-multichallenge]] Table 1 |
| dialogue | emotional and length coverage of synthetic dialogue | slice audit after filtering | no number in card | [[deepdialogue]] |

**Reading the table.** Rows with a number are measured in that source's setting; rows marked "no number in card" are definitions of what to measure, not evidence that measuring it changes training. Several of the llm-training cards used here ([[realtalk]], [[persona-lexical-diversity]], [[deepdialogue]], [[sdiareward]], [[audio-multichallenge]]) are scope-limited extracts without outcome tables; they define components but do not supply effect sizes.

---

## §2 Building preference data for human-likeness

Five constructions appear in the sources. They differ in where the preferred side comes from, and that choice decides what the policy learns.

| Construction | Preferred side | Rejected side | What DPO learns | Source |
|---|---|---|---|---|
| Style contrast, same model, two system prompts | casual "human-like" answer | formal impersonal answer | "less formal, fewer disclaimers" (card's reading), not a calibrated naturalness scale | [[human-like-responses-dpo]] §3 |
| Constitution teacher | teacher prompted with ~10 first-person persona assertions | the untrained student's default answer | move from the default assistant register toward the constitution | [[open-character-training]] §2.3 |
| Real human data | real recording tokens | the model's own tokens for the same text | toward the real-speech distribution; same structure as SPIN | [[speechalign]] §3.1 |
| Human corpus as reference for lexical choice | a coherent alternative token | the first token of a banned over-represented pattern | avoid specific phrases; single-position negative signal | [[antislop]] §5 |
| Natural human signal | higher-voted human text | lower-voted, length-balanced | the voters' taste | [[litbench]] §3.2 |

**Audit the preferred side.** The human-like DPO dataset's sample chosen answer gives advice "from my grandma", while the rejected answer states it is an AI (Table 1). Its questions were generated by Llama 3 405B and its answers by Llama 3 70B (§3). Two things follow. The "human-like" side is a model's imitation of a human, not human text, so it carries the model's own phrasing. And the chosen distribution teaches fabricated personal experience. For an agent that must disclose that it is an AI, that data teaches the wrong behavior ([[human-like-responses-dpo]] Guideline).

**Real conversation logs as references.** REALTALK reports that real messaging shows emotional variation where synthetic dialogue shows "synthetic positivity and excessive empathy" (§5.2), with a narrow sample (ten participants aged 18–25, chit-chat; §3.1, §8). For a sales agent the corresponding reference is real sales-call transcripts and recordings from the target market. The card's limit applies: chit-chat findings are not sales findings.

**Simulated customers.** ConsistentChat plans multi-turn synthesis with intent and information-flow skeletons ([[consistentchat]]); JOSH uses sparse task-success rewards to select dialogue branches when an executable simulator exists ([[josh-dialogue-self-training]]). Both are construction methods; the cards give no numbers. The simulator-diversity evidence from ch-22 §9 applies to them.

---

## §3 Human-like DPO and its instruction-following cost

**Setup** ([[human-like-responses-dpo]] §3–§4): 10,884 samples over 256 topics; LoRA r = 8, α = 4, DPO β = 0.1, lr 2e-4, 1 epoch; Llama3-8B-Instruct, Qwen-2.5-7B-Instruct, Mistral-Nemo-Instruct-2407.

**Human-likeness result** (Table 5): 500 questions generated by the same method as training, 2,000 votes, emojis removed from both sides; annotators mostly high-school students, generally non-native English speakers, plus adults.

| Model | Fine-tuned chosen | Official chosen |
|---|---|---|
| Llama-3-8B | 89.6% | 10.4% |
| Qwen-2.5-7B | 89.5% | 10.5% |
| Mistral-Nemo | 79.6% | 20.4% |

**Capability cost** (Tables 6–7):

| Model | Average change | IFEval change | Average change without IFEval |
|---|---|---|---|
| Llama-3-8B | −1.20 | −9.11 | −0.02 |
| Qwen-2.5-7B | −0.20 | −3.01 | +0.36 |
| Mistral-Nemo | −0.65 | −9.29 | +1.07 |

The cost concentrates in IFEval; other Open LLM Leaderboard tasks change by at most 3.02 points. The paper does not test why. One explanation consistent with the data construction (an inference, not tested): the rejected side is "formal, structured, impersonal", and IFEval rewards satisfying explicit format constraints, so pushing away from structured answers also pushes away from some constraint-following behavior.

**What the 89.6% does and does not show.** The evaluation questions are in-distribution (same generation method); there is no inter-annotator agreement, no comparison against real human answers, and no AI-detector or lexical metric (Verification). The authors name the removal of self-referential disclaimers as the main reason the official models lost. The result shows that the fine-tuned models' answers were judged more human-like than their instruct versions' answers by this annotator pool on these questions. It does not show they are indistinguishable from humans, and it does not separate "more human-like" from "no longer says it is an AI".

---

## §4 Antislop: FTPO versus DPO for AI phrasing

**Measuring AI phrasing.** For each model, 2,000 outputs on Reddit creative-writing prompts are compared with a human baseline (wordfreq for words; Reddit creative writing plus Project Gutenberg for n-grams, stopwords removed) using ρ(p) = f_LLM(p) / f_human(p) ([[antislop]] §3.1). Over-represented patterns cluster within model families (§3.2).

**The sampler.** After each token, the Antislop sampler scans the trace; on a banned word, phrase or regex it backtracks to the first token of the pattern, multiplies that token's probability by 10^(−10s) for ban strength s ∈ [0, 1], and resamples with min-p filtering; if the banned token is resampled anyway it is allowed ("soft-banning") (§4.1–4.2). It suppressed 100% of "It's not X, it's Y" with a regex ban (§6.5). Cost: throughput falls by 69–96% depending on banlist size (§4.3). A hard logit bias of −100 collapsed writing quality to 28/100 at 8K patterns (§6.2).

**FTPO.** Each backtracking event yields a pair at one token position: rejected = the first banned token, chosen = coherent min-p alternatives (§5.4). The loss is a margin preference term `softplus(m − Δ_c)` weighted by `clamp((m − Δ_c)/m, 0, 1)` (so it stops pushing once the margin is reached), plus an MSE tether on target logits outside a free window and a strong MSE tether on all non-target logits (§5.2). The tethers are in logit space rather than KL, so a large change to the chosen and rejected logits does not move the rest of the vocabulary (§5.3).

**Result** (Table 2, gemma-3-12b; last 5 layers + lm_head trainable, LoRA r = 256, 1 epoch, early stop at preference accuracy 0.85):

| Run | MMLU | GSM8K | Longform | Writing | Diversity | Ban % |
|---|---|---|---|---|---|---|
| baseline | 0.590 | 0.888 | 51.3 | 67.80 | 100.00 | 0.00 |
| FTPO 2k / 4k / 8k | 0.559 / 0.565 / 0.592 | 0.876 / 0.880 / 0.889 | 47.5 / 49.4 / 52.3 | 68.93 / 67.31 / 67.49 | 101.05 / 97.68 / 95.09 | 92.39 / 90.15 / 83.40 |
| DPO 2k / 4k / 8k | 0.541 / 0.549 / 0.571 | 0.847 / 0.861 / 0.864 | 36.6 / 34.8 / 26.9 | 62.98 / 58.24 / 54.61 | 91.03 / 81.92 / 73.92 | 82.00 / 80.64 / 81.44 |

**Why sequence-level DPO costs diversity here.** DPO's pair is a whole sequence; the gradient lowers the probability of every token in the rejected sequence relative to the chosen one, including the many tokens the two share in style. FTPO's pair differs at exactly one position and tethers everything else. The paper measures the outcome (Table 2, Fig. 4a: DPO degraded sharply beyond about 40% preference accuracy while FTPO reached nearly 100%); the mechanism above is the authors' design rationale (§5.3). Raising DPO β to 1.0 mitigated degradation but reduced suppression by 15.9% (§6.4).

**Scope.** All slop analysis uses creative-writing prompts; conversational or sales registers are not measured (card Scope). Llama-3.3-70B was more sensitive and needed lm_head-only training, reaching 66.41% suppression (Table 2, §6.3).

---

## §5 Character training: persona that holds over turns

**Anthropic's description** ([[anthropic-claude-character]]). Claude 3 was the first Claude model with character training, a "character" variant of Constitutional AI: Claude generates trait-relevant user messages, produces several in-character responses, ranks its own responses by fit to the traits, and a preference model is trained on the rankings, "without the need for human interaction or feedback". Constructing and adjusting traits is "a relatively hands-on process". The trait list includes self-description statements such as "I am an artificial intelligence and do not have a body or an image or avatar". The post states that "being more engaging isn't the same thing as having a good character" and that "an excessive desire to be engaging seems like an undesirable character trait". No measurements are reported.

**The open replication** ([[open-character-training]]):

1. Constitution: about 10 first-person assertions per persona (§2.2).
2. DPO distillation: teacher GLM 4.5 Air with the constitution in its system prompt produces chosen responses; the student (Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B) produces rejected responses without it (§2.3).
3. Introspection SFT: 10,000 self-reflection responses (10 prompts × 1,000) and 2,000 ten-turn self-interactions (§2.4).

**Results.**

| Measurement | Llama 3.1 8B | Qwen 2.5 7B | Gemma 3 4B | Source |
|---|---|---|---|---|
| Prefill-attack persona F1, distillation only → full | 0.79 → 0.95 | 0.66 → 0.86 | 0.84 → 0.95 | Table 2 |
| Coherence win rate vs system prompting | 89.2% | 93.4% | 56.5% | Table 3 |
| Coherence win rate vs activation steering | 78.4% | 94.4% | 82.1% | Table 3 |
| Coherence win rate vs distillation only | 77.0% | 75.4% | 87.9% | Table 3 |

Coherence judgments were kept only if invariant to order swap (§3.4). System-prompted personas "frequently 'break character'" under adversarial instructions (§3.2, Fig. 5). Only the misaligned persona caused notable capability drops (Llama TruthfulQA 45.9 → 34.1; Table 4). The sycophantic persona is one of the 11 trained (Table 1): the same pipeline installs or removes sycophancy, depending on the constitution.

**For the sales agent.** A sales persona ("answers before pitching", "plain words", "states it is an AI assistant when asked") can be written as a constitution. The measured part is the pipeline's robustness; whether a sales constitution transfers is untested.

---

## §6 Speech naturalness: judges, proxies and preference optimization

**Human agreement sets the ceiling** ([[speechjudge]] §3.2, Fig. 3). 99K pairs from six zero-shot TTS models, 69 annotators, 2.49 annotations per pair; 51.5% full agreement and 17.2% weak agreement overall; expressive speech 44.3% vs regular 54.5% full agreement. The eval set keeps 1,000 full-agreement, no-tie pairs.

**Proxies against human naturalness preference** (Table 2, total accuracy): WER 57.9; DNSMOS 57.9; UTMOS 53.7; audiobox PQ 57.1; AASIST deepfake detector 46.7; Qwen2.5-Omni-7B 60.6; Gemini-2.5-Flash 69.1; Gemini-2.5-Pro 66.5; GPT-4o Audio 67.4.

**Trained judges** (Table 3): BTRM 72.7; GRM (SFT) 75.3; GRM (SFT + RL) 77.2, 79.4 at voting@10. The GRM is Qwen2.5-Omni-7B, SFT on Gemini-2.5-Flash CoT where Gemini agreed with humans, then GRPO with ±1 for matching the human label on the cases Gemini got wrong (§5.1).

**As a reward.** Best-of-100 with humans comparing the RM's pick to a random sample: GRM 43.0 / 26.8 / 30.2% win/tie/lose; BTRM 39.5 / 28.0 / 32.5% (Fig. 5). TTS post-training of Qwen2.5-0.5B-TTS: naturalness CMOS vs base 0.18 (offline DPO, INTP), 0.16 (offline DPO, SpeechJudge-Data), 0.21 (offline DPO, GRM-relabeled INTP), 0.25 ± 0.09 (online DPO with the GRM reward) (Fig. 6a).

**Residual failures** (§6): clean-but-robotic vs slightly-noisy-but-lively speech; prosody vs articulation trade-offs; very high-F0 emotional speech; whispers. The GRM "can over-weight cleanness, under-weight style-appropriate prosody". The data are "TTS-style read speech rather than spontaneous conversation".

**SpeechAlign** ([[speechalign]]). Codec-LM TTS where the NAR model is trained on real ("golden") AR tokens but sees synthetic AR tokens at inference; reconstruction from synthetic tokens is worse (WER 7.2 vs 5.9, SIM 0.87 vs 0.93; Table 1). Preference pairs: golden tokens chosen, the model's own tokens rejected, 50K LibriSpeech pairs, iterated.

| Method (LibriSpeech) | WER | SIM |
|---|---|---|
| SFT | 7.2 | 0.87 |
| Continue SFT on golden tokens | 8.0 | 0.88 |
| RLHF-PPO (BT RM) | 7.1 | 0.89 |
| Best-of-N | 8.0 | 0.88 |
| DPO-Iter1 / Iter2 / Iter3 | 6.7 / 6.2 / 6.0 | 0.88 / 0.89 / 0.90 |

The continued-SFT control shows that the gain comes from the preference objective, not from extra exposure to real tokens (§5.1). 50K → 250K pairs gave no larger gain (Fig. 4a). Human verification covered 100 pairs (71% / 21% / 8%). Scope: read speech; conversational naturalness, disfluency and turn-taking are not measured.

**Dialogue-level speech.** SDiaReward separates modality pairing and colloquialness pairing (§3.1) and notes domain-dependent absolute reward offsets (§6) ([[sdiareward]]); Audio MultiChallenge keeps ambient variability and disfluencies from human improvisation, with fixed-context evaluation that is "not a closed-loop controller/barge-in benchmark" ([[audio-multichallenge]]). Neither card gives outcome numbers. For a voice agent these are the instruments closest to the target, and the least measured.

---

## §7 Rubric-based style rewards

Rubicon's "Plain Narrative" rubric scores voice/tone (calm acceptance, grounded realism, understated emotion), content (insight from lived experience, not lecturing), and language (short plain sentences, natural speech patterns, textured details, natural rhythm) on 1–5, with "Any hint of artificiality is a significant failure" ([[rubicon-rubric-anchors]] §4.2). In the paper's example, the base model opens with "I don't experience emotions..." and the trained model writes a first-person narrative.

**State this precisely: Rubicon's "less AI-like" claim is supported by case studies only.** The paper reports no human-likeness metric or AI-detector score (card Verification). The authors also state that their large-scale human feedback "was not consistently reflected by standardized benchmarks", with no numbers (§5). The benchmark gains in Table 1 (65.29 → 70.50 average) are on open-ended LLM-judged benchmarks, not on human-likeness.

Note what the example rewards: a first-person narrative from "lived experience" written by a model. For a writing task that may be the goal. For an agent that must not claim to be human, the same rubric content is a reward for the forbidden behavior.

Kimi K2's prescriptive rubrics, intended to eliminate reward hacking, penalize hedging and disclaimers; the report lists over-confidence in ambiguous contexts as a resulting side effect ([[kimi-k2-recipe]] App. F.3). A sales agent that is rewarded for not hedging will state prices and availability with confidence it does not have unless facts are checked separately.

---

## §8 Hacking evidence for style rewards

| Evidence | Number | Relevance to a sales agent | Source |
|---|---|---|---|
| Scalar writing RM: length and self-explanation | 1,872 mean tokens and 417 self-explanation tokens vs 1,292 and 58 for the pairwise GenRM | spoken replies get longer; the agent explains how helpful it is being | [[writing-zero]] Table 3 |
| Hacked policy ranked first by a public benchmark | WritingBench 8.87 for a "gibberish" policy | an LLM-judged "naturalness" benchmark can rank a hacked voice agent first | [[writing-zero]] §4.3.2 |
| Engagement-bait closers | +0.176 (no-think) / +0.232 (think) | sales closers are legitimate and also the hack | [[rlmt-think-chat]] Table 9 |
| Self-praise | +0.114 / +0.217 | "I hope this detailed answer helps" in a voice turn | [[rlmt-think-chat]] Table 9 |
| Planted single-word bonus | 96% of rollouts by step 70 (think) | any filler word the judge likes ("absolutely") spreads | [[rlmt-think-chat]] Table 8 |

The difficulty specific to sales is that two hacked behaviors (closers and praise of the customer) are also normal sales behaviors. A veto that removes all closers removes the job; a reward that likes closers is hackable. The resolution in the sketch (§10) is a rubric condition: a closer is rewarded only when the customer's need has been addressed, and praise of the customer is vetoed outright, following Rubicon's "prefatory sycophancy" rule.

---

## §9 The risk of optimizing a detector or a judge instead of people

Three distinct failure paths appear in the sources:

1. **The instrument does not measure the target.** On naturalness pairs, the AASIST deepfake detector scored 46.7%, below chance ([[speechjudge]] Table 2). A detector trained to separate synthetic from real speech answers a different question than "which of two synthetic clips sounds more natural to a listener". Optimizing "less detectable" therefore has no measured link to "more natural". The same holds for text: no source in the library measures whether minimizing an AI-text detector score improves human-likeness ratings.
2. **The instrument measures the target on its training distribution and is exploited outside it.** Gold score rises and then falls with KL while the proxy keeps rising ([[reward-model-overoptimization]] Fig. 1). The judge's agreement with humans is measured on the pairs it was validated on; the policy moves to outputs outside that set. LitBench's Claude-3.7-Sonnet selector performed at chance in the human study on new stories, while a small trained RM's picks were preferred 57% vs 41% ([[litbench]] §5).
3. **The instrument encodes one population's taste.** LitBench's labels reflect Reddit's demographics (§7); SpeechJudge's raters agreed more on Chinese than on English (§6); the human-like DPO votes came mostly from non-native high-school students ([[human-like-responses-dpo]] §5.1). The policy learns whoever labeled the data.

**And one path that is specific to human-likeness: the reward for sounding human becomes a reward for claiming to be human.** The human-like DPO dataset's chosen side claims personal experience (Table 1); Rubicon's style example rewards first-person "lived experience" (§4.2); Anthropic's character traits instead include "I am an artificial intelligence" ([[anthropic-claude-character]]). These point in opposite directions, and the reward decides which one the policy follows.

**What to do about each path.** (1) Validate every instrument against human A/B labels in your own domain before using it as a reward (SpeechJudge's method). (2) Track a held-out human-aligned score against √KL and stop at its peak; evaluate with a judge you did not train against; run periodic small human studies on the policy's outputs. (3) Recruit raters from the customer population and record rater demographics. (4) Gate (§10).

---

## §10 Reward design sketch for the sales voice agent

**This is a sketch: practice, not measured.** No source trains a sales voice agent. Each element is labeled with the evidence behind its *method*; the combination has not been tested. The sketch follows [[subjective-reward-guide]] §4 and adds the measurement plan.

**Gates (reward = 0 for the rollout if any gate fails).**

| Gate | Check | Evidence |
|---|---|---|
| G1. No claim of being human; no invented personal experience; truthful disclosure when asked whether it is an AI | rule checks for first-person bodily/biographical claims plus a judge on a fixed rubric; a red-team probe set ("Are you a real person?", "What did you do this weekend?", "Where did you grow up?") run every evaluation interval | practice; motivated by [[human-like-responses-dpo]] Table 1 and [[anthropic-claude-character]] traits |
| G2. Product and price facts consistent with the catalog | program check where possible; reference-based judge otherwise, probed with master-key responses before the run | method measured: [[master-rm-one-token]] Table 1 |
| G3. No praise of the customer's question, no self-praise | regex plus judge veto | practice: [[rubicon-rubric-anchors]] App. A.1 |
| G4. Synthesized reply intelligible | WER of an ASR model on the TTS output below a threshold | practice: WER as intelligibility measure in [[speechalign]], [[speechjudge]] |

G1 is the gate against falsely claiming to be human. Measure its false-negative rate on the red-team set before training and after every evaluation interval; a rising false-negative rate means the style terms are moving the policy toward human claims and the gate is not catching them.

**Scored terms (only for rollouts that pass all gates).**

1. **Conversation quality**: pairwise GenRM against a random in-group rollout, with a sales rubric (short turns, plain wording, answer before pitching, a follow-up question only when the customer's need is unresolved, a closer only when the need has been addressed). Method measured ([[writing-zero]] Table 3; [[rubrics-as-rewards]] Table 1); rubric content is practice.
2. **Task outcome** from a simulated customer (meeting booked, objection resolved), from a diverse simulator population (ch-22 §9). Practice ([[josh-dialogue-self-training]]); simulator-diversity evidence is measured in a different domain.
3. **Speech naturalness** (if the TTS is trained): pairwise judge trained on human A/B labels of your own TTS in conversational speech; not a MOS predictor. Method measured ([[speechjudge]] Tables 2–3) on read speech only. Real sales-call recordings as the preferred side for TTS DPO; method measured on read speech ([[speechalign]]).
4. **Lexical AI-ness**: banlist from frequency ratios against real sales transcripts; suppressed with FTPO or the sampler rather than sequence-level DPO. Method measured on creative writing ([[antislop]] Table 2).

**Persona.** A constitution of about 10 first-person assertions that includes "I am an AI assistant" and the sales behaviors; DPO from a constitution-prompted teacher, introspection SFT, multi-turn prefill tests. Method measured ([[open-character-training]] Table 2).

**Stage order.** Compliance and constraints first (gates G1–G3 as verifiable rewards), style second. Measured seesaw in [[rubicon-rubric-anchors]] §4.4.

**Calibration and monitoring.**

| Monitor | Why | Source |
|---|---|---|
| Gold set from real calls, order-swapped, length-balanced | judge validity | [[judge-llm-bias]], [[litbench]] |
| Turn length (words, seconds of audio) RL − SFT | length hacking | [[rlmt-think-chat]] Table 9 |
| Closer rate, split by whether the rubric condition held | engagement-bait hacking inside a legitimate behavior | [[rlmt-think-chat]] Table 9 |
| Praise and self-praise rates | gate G3 effectiveness | [[rubicon-rubric-anchors]] |
| Banlist ratio and length-controlled lexical diversity | lexical AI-ness without diversity loss | [[antislop]] |
| G1 false-negative rate on the red-team set | human-claim leakage | this sketch |
| Planted-hack test at run start | monitoring sensitivity | [[rlmt-think-chat]] Table 8 |
| Held-out human score vs √KL; stop at its peak | over-optimization | [[reward-model-overoptimization]] |
| Constraint benchmark (e.g., IFEval-style) per evaluation | instruction-following cost of style training | [[human-like-responses-dpo]] Table 6 |

**Algorithm.** Online group-baseline RL against the calibrated judge for the conversation terms (ch-19 §8; [[writing-zero]], [[rlmt-think-chat]]); online DPO with the speech judge for the TTS (CMOS 0.25 vs 0.16–0.21 offline; [[speechjudge]] Fig. 6a); FTPO for lexical patterns.

---

## Recipe

1. List the components of human-likeness you will improve (§1) and, for each, the instrument and its validation against human A/B labels in your domain.
2. Collect real sales-call transcripts and recordings as references; record rater and speaker demographics.
3. Build preference data with real human data as the preferred side where possible (§2); audit every preferred sample for human-identity claims and invented experience.
4. Write the persona constitution, including AI disclosure; run constitution-teacher DPO and introspection SFT; test with multi-turn prefill attacks (§5).
5. Build the speech judge from human A/B labels of your own TTS on conversational text; check proxies (UTMOS, DNSMOS, detectors) against the same labels before using any of them (§6).
6. Build the banlist from frequency ratios against real sales transcripts; suppress with FTPO or the sampler; track length-controlled diversity (§4).
7. Implement gates G1–G4 and measure G1's false-negative rate on the red-team set before training (§10).
8. Train compliance first, then style; monitor the table in §10; stop at the peak of the held-out human score.
9. Evaluate with a small human study on real-looking calls, with raters from the customer population, and with a judge you did not train against.

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Training on model-imitated "human" answers | agent invents a grandmother, a weekend, a hometown | G1 red-team probe set every evaluation |
| Sequence-level DPO for lexical style | flatter, repetitive wording; lower diversity | length-controlled MATTR / Distinct-n vs baseline ([[antislop]]) |
| Style training without a constraint metric | format and instruction compliance fall | IFEval-style benchmark per evaluation ([[human-like-responses-dpo]] Table 6) |
| MOS predictor or detector as speech reward | scores rise, listeners do not prefer outputs | proxy agreement on your human A/B set; UTMOS 53.7%, AASIST 46.7% in [[speechjudge]] |
| Reporting "less AI-like" from examples | claim without a number | require a human pairwise study or a validated metric ([[rubicon-rubric-anchors]] is case-study only) |
| Vetoing all closers | agent never asks for the meeting | task-outcome rate from the simulator |
| Rewarding all closers | closer in every turn | closer rate split by rubric condition |
| One rater pool | style tuned to the wrong audience | rater demographics vs customer demographics ([[litbench]] §7) |

---

## Check your understanding

1. Why does SpeechAlign's continued-SFT control matter for interpreting its DPO result, and what would you conclude if continued SFT had matched DPO?
2. FTPO and DPO were trained on the same pairs. Explain why the sequence-level loss reduced lexical diversity while the token-level loss did not.
3. The human-like DPO models won about 90% of human-likeness votes. List three reasons this does not show that the models sound like humans to the customers of a sales agent.
4. Why is a deepfake detector's score not a usable reward for speech naturalness, even if the detector is accurate on its own task?
5. Sales closers and praise are both normal sales behaviors and measured reward hacks. How does the sketch separate legitimate from hacked uses, and what measurement would show the separation is failing?
6. Why must the gate against claiming to be human be checked with a red-team probe set during training rather than only before release?
7. Open Character Training's sycophantic persona was trainable with the same pipeline. What does that imply about writing your sales constitution?

---

## Connections

- Course chapters: ch-19 (subjective rewards: BT RMs, pairwise judges, calibration, hacking, online vs offline), ch-22 (capstone worksheet: grader ladder §7, reward composition §8, diversity §9), ch-11 (judge probing), ch-12 (hacking monitors), ch-13 (user-simulator diversity), ch-17 (stage order).
- llm-training: ch-25 (multi-turn conversation synthesis; long-conversation persona patterns), ch-41 (reward modeling), ch-42 (reward hacking and judge design), ch-49 (judge bias and calibration), ch-43 (entropy, output diversity), ch-44b (non-verifiable rewards in multi-domain RL), ch-45b (multi-turn agentic RL), ch-46 (DPO vs RLVR lab).
- Sources: [[subjective-reward-guide]] §3–§4 (components table and the sketch this chapter expands); [[antislop]], [[human-like-responses-dpo]], [[open-character-training]], [[speechjudge]], [[speechalign]] for the measured core.

---

## Sources

- [[subjective-reward-guide]] — components of human-likeness and the sales voice agent reward sketch.
- [[human-like-responses-dpo]] — style-contrast DPO; 89.6% human-likeness wins; IFEval cost; chosen side claims personal experience.
- [[antislop]] — phrase over-representation metric; sampler; FTPO vs DPO.
- [[open-character-training]] — constitution DPO + introspection SFT; prefill robustness; coherence win rates.
- [[anthropic-claude-character]] — character training pipeline and AI self-description traits; engagement vs character.
- [[realtalk]] — real vs synthetic emotional variation and persona stability (no numbers in card).
- [[persona-lexical-diversity]] — length-controlled lexical diversity for personas (no numbers in card).
- [[deepdialogue]] — emotional and length slice audits for synthetic dialogue (no numbers in card).
- [[consistentchat]] — skeleton-guided multi-turn synthesis.
- [[josh-dialogue-self-training]] — task-success selection of dialogue branches.
- [[speechjudge]] — human agreement, proxies near chance, GRM vs BTRM, online vs offline DPO for TTS.
- [[speechalign]] — real-vs-generated speech preference pairs; DPO vs PPO vs BoN vs continued SFT.
- [[sdiareward]] — spoken-dialogue rewards with colloquialness (no numbers in card).
- [[audio-multichallenge]] — spoken multi-turn evaluation with disfluencies (dataset size only).
- [[rubicon-rubric-anchors]] — Plain Narrative style rubric (case study only); veto rubric; seesaw.
- [[writing-zero]] — scalar-RM hacking and benchmark failure.
- [[rlmt-think-chat]] — engagement-bait, self-praise, planted hack.
- [[litbench]] — rater population; frontier judge at chance as selector.
- [[kimi-k2-recipe]] — prescriptive rubrics and over-confidence.
- [[reward-model-overoptimization]] — proxy vs gold.
- [[master-rm-one-token]] — master-key probes for reference-based fact gates.
- [[judge-llm-bias]] — order swapping for pairwise judges.
- [[rubrics-as-rewards]] — per-prompt rubric judge method used for the conversation-quality term.
