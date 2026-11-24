<!-- scope: design guide for RL rewards when no verifier exists (naturalness, style, human-likeness): reward construction, failure modes with measured evidence, measurable components of human-like text and speech, reward sketch for a human-like sales voice agent. Feeds ch-19 and ch-23.
     deps: [[bradley-terry-rm]], [[generative-reward-models]], [[rubrics-as-rewards]], [[judge-llm-bias]]
     see-also: [[rl-design-guide]], [[rubicon-rubric-anchors]], [[writing-zero]], [[rlmt-think-chat]], [[skywork-reward-v2]], [[litbench]], [[human-like-responses-dpo]], [[antislop]], [[open-character-training]], [[speechjudge]], [[speechalign]]
-->

# Subjective-reward guide: RL when the score is unclear

Tags: **measured** = a cited source reports a number for this rule. **practice** = design reasoning or author statement without a controlled measurement. Numbers are copied from the cited card, which gives the source location.

## 1. Turning an unclear quality into a reward

**1a. Pairwise human preference → Bradley-Terry RM**
- R1 **measured.** Collect pairwise labels, not absolute scores, and report RM accuracy by preference strength: Llama 2's RM scored 79.1% on "significantly better" pairs and 54.5% on "negligibly better" pairs [[bradley-terry-rm]].
- R2 **measured.** Spend human effort on a small gold set labeled under a written protocol (per-pair attributes, tool access), and use it to validate the RM and as retrieved few-shot examples for LLM labelers: full protocol +3.2 vs raw human labels +0.4; simple LLM curation +0.1; 12M uncurated pairs did not beat the seed RM [[skywork-reward-v2]].
- R3 **measured.** For naturally occurring signals (votes, ratings, call outcomes), require a minimum signal gap, control exposure time, and balance lengths; without length rebalancing the RM saturated at 70% and preferred the longer story in most cases [[litbench]].
- R4 **measured.** A small RM trained on in-domain human pairs can beat frontier zero-shot judges: 78% vs 73% (Claude-3.7-Sonnet) on creative writing [[litbench]]; a stronger general RM changed RL outcomes by a factor of about 2.6 on chat average (39.0 vs 14.8) [[rlmt-think-chat]].

**1b. Pairwise / generative judges**
- R5 **measured.** Compare BT and generative judges on your own gold set before choosing: in speech a CoT GRM beat BT (77.2% vs 72.7%) [[speechjudge]], in creative writing distilled CoT lowered accuracy (72% vs ~78%) [[litbench]].
- R6 **measured.** In RL, score each rollout pairwise against a random rollout from the same group and drop groups with |ΣR|/G > 0.6; with a scalar RM the policy reached 1,872 mean tokens and 417 tokens of self-justification vs 1,292 and 58 with the pairwise GenRM [[writing-zero]].
- R7 **measured.** Sample the judge several times and vote when budget allows: +2.2 points at voting@10 [[speechjudge]], +4.9 at k = 32 with a meta-RM filter [[deepseek-grm]].

**1c. Rubric anchors and checklists**
- R8 **measured.** Generate a rubric or checklist per prompt and give it to the judge: prompt-specific rubrics 31.2 vs Likert 25.5 vs generic rubrics 12.5 on HealthBench [[rubrics-as-rewards]]; candidate-based checklists 77.3 vs direct 74.3 on IFEval [[rlcf-checklists]].
- R9 **measured.** Train constraint-following rubrics and style/empathy rubrics in separate sequential stages: creativity-only training cost Collie −6.0 and IFEval −5.9 [[rubicon-rubric-anchors]].
- R10 **practice.** Add veto rubrics that zero the reward for user praise and self-praise; Rubicon reports fewer reward spikes and longer stable training but no number [[rubicon-rubric-anchors]].
- R11 **measured.** Score several aspects separately instead of one overall score: 65.2% vs 62.0% on WebGPT [[ultrafeedback]].

**1d. Reference-based vs reference-free**
- R12 **measured.** Use a reference when one exists: reference-conditioned rubrics 35.9 vs 32.0 [[rubrics-as-rewards]]; a reference answer cut GPT-4 math judging failures from 14/20 to 3/20 [[judge-llm-bias]]. Probe reference-based judges with content-free answers first (false-positive rates up to 90.9%) [[master-rm-one-token]].
- R13 **measured.** For style, real human data can serve as the reference: real-recording tokens beat own generations in 71% of human checks and gave WER 7.2 → 6.0 over 3 DPO rounds [[speechalign]]; phrase frequency against a human corpus defines lexical "AI-ness" [[antislop]].
- R14 **practice.** Use reference-free pairwise comparison when no single good answer exists (creative or conversational turns) [[writing-zero]].

**1e. Calibrating the judge against humans**
- R15 **measured.** Report judge-human agreement on full-agreement pairs and per slice; expressive speech had 44.3% full human agreement vs 54.5% for regular speech [[speechjudge]].
- R16 **measured.** Swap answer order and count only consistent verdicts: GPT-4 was order-consistent on 65.0% of pairs [[judge-llm-bias]]; Claude-3.5-Sonnet favored the first response about 60% of the time [[writing-zero]].
- R17 **measured.** Validate with humans on the policy's own outputs: annotators chose the RM-preferred story 57% vs 41% [[litbench]]; Skywork-V2-8B agreed with humans 81.2% vs 74.3% for GPT-4o [[skywork-reward-v2]].
- R18 **measured.** Clamp AI-feedback label probabilities (40–60%) before preference-model training [[constitutional-ai]].

## 2. Failure modes of subjective rewards (measured evidence)
| Failure | Evidence | Source |
|---|---|---|
| Length bias in labels and RMs | 65.25% of raw chosen stories longer; +166 words RL − SFT | [[litbench]], [[rlmt-think-chat]] |
| Self-justification / self-praise | 417 vs 58 explanation tokens; self-praise rate +0.217 | [[writing-zero]], [[rlmt-think-chat]] |
| Engagement-bait closers | +0.232 rate ("Would you like to hear...") with thinking | [[rlmt-think-chat]] |
| Fast spread of a lexical hack | "absolutely" bonus: 1% → 96% of rollouts by step 70 | [[rlmt-think-chat]] |
| Public benchmark rewards a hacked policy | gibberish policy scored 8.87 (highest) on WritingBench | [[writing-zero]] |
| Proxy rises after gold peaks | gold score falls with KL while proxy rises | [[reward-model-overoptimization]] |
| Sycophancy from preference models | PM preferred sycophantic answers for 45% of hardest misconceptions | [[sycophancy-in-lms]] |
| Proxy metrics near chance for naturalness | UTMOS 53.7%, DNSMOS 57.9% agreement | [[speechjudge]] |
| Diversity collapse with sequence-level DPO | lexical diversity 73.92 (baseline 100) vs FTPO 95.09 | [[antislop]] |
| Seesaw between objectives | creativity-only: Collie −6.0, IFEval −5.9 | [[rubicon-rubric-anchors]] |
| Instruction-following cost of style training | IFEval −9.11 / −3.01 / −9.29 | [[human-like-responses-dpo]] |
| Rater-population taste | Reddit demographics; zh raters agree more than en | [[litbench]], [[speechjudge]] |

## 3. Measurable components of "human-like"
| Layer | Component | Metric | Source |
|---|---|---|---|
| text | AI phrasing | phrase frequency ratio f_LLM/f_human; regex "It's not X, it's Y" (6.3×) | [[antislop]] |
| text | lexical diversity | MATTR-500, Root-TTR, HD-D, Distinct-n, length-controlled | [[antislop]], [[persona-lexical-diversity]] |
| text | AI disclaimers, user praise, self-praise, engagement bait | per-response rate | [[human-like-responses-dpo]], [[rubicon-rubric-anchors]], [[rlmt-think-chat]] |
| text | register and length | words, bullet count | [[rlmt-think-chat]] |
| text | persona stability across turns | persona-classifier F1 under prefill (0.79 → 0.95) | [[open-character-training]] |
| text | emotional variation vs real chat | comparison to real messaging (no number in card) | [[realtalk]] |
| text | overall human-likeness | pairwise human vote (89.6% win) | [[human-like-responses-dpo]] |
| speech | naturalness | pairwise CMOS judge (77.2%) | [[speechjudge]] |
| speech | intelligibility, voice match | WER, speaker similarity | [[speechalign]], [[speechjudge]] |
| speech | prosody, pacing, articulation | judge criteria in CoT prompt | [[speechjudge]] |
| dialogue | colloquialness, corrections, disfluency | spoken-dialogue preference / benchmark (no number in card) | [[sdiareward]], [[audio-multichallenge]] |

## 4. Design sketch: reward for a human-like sales voice agent
**Gates (reward = 0 on failure).** (a) Claims to be human or invents personal experience; the human-like DPO dataset's chosen answers do this [[human-like-responses-dpo]] **practice**. (b) Product or price facts contradicted by the catalog, checked by program or reference judge with master-key probes [[master-rm-one-token]] **measured**. (c) User praise or self-praise veto [[rubicon-rubric-anchors]] **practice**. (d) WER above a threshold on the synthesized reply [[speechjudge]] **practice**.

**Scored terms (on gated rollouts).**
1. Conversation quality: pairwise GenRM vs in-group reference with a sales rubric (short turns, plain wording, answer before pitching, a follow-up question only when the customer's need is unresolved) [[writing-zero]], [[rubrics-as-rewards]] **measured** for method, **practice** for rubric content.
2. Task outcome from a simulated customer (meeting booked, objection resolved) [[josh-dialogue-self-training]] **practice**.
3. Speech naturalness: pairwise judge trained on human A/B labels of your own TTS in conversational speech; not UTMOS [[speechjudge]] **measured**; real sales-call recordings as preferred side for TTS DPO [[speechalign]] **measured** (read speech only).
4. Lexical AI-ness: banlist from frequency ratios against real sales transcripts, suppressed with FTPO or the sampler rather than DPO [[antislop]] **measured**.

**Persona.** Constitution of ~10 first-person assertions, DPO from a constitution-prompted teacher, introspection SFT, prefill multi-turn test [[open-character-training]] **measured**.

**Calibration and monitoring.** Gold set from real calls, order-swapped and length-balanced [[judge-llm-bias]], [[litbench]] **measured**; track length, engagement-bait rate (a sales closer is legitimate only when the rubric condition holds), self-praise, banlist ratio, and a planted-hack test [[rlmt-think-chat]] **measured**; stop at the peak of a held-out human score vs KL [[reward-model-overoptimization]] **measured**; train compliance/constraint stage before style stage [[rubicon-rubric-anchors]] **measured**.

**Algorithm.** Online GRPO-style RL against the judge for conversation terms [[rlmt-think-chat]], [[writing-zero]] **measured**; online DPO with the speech judge for TTS (naturalness CMOS 0.25 vs 0.16–0.21 offline) [[speechjudge]] **measured**.
