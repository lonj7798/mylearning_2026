<!-- scope: HarmBench — a 510-behavior harmful-behavior library, a standardized red-teaming evaluation pipeline, and the R2D2 adversarial-training baseline
     see-also: [[wildguard-data]], [[anthropic-safety-research]], [[tulu-3-sft-mix]]
-->

# HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal
- **Core Insight:** HarmBench separates the target-behavior inventory (510 behaviors in 4 functional categories) from the attack methods and from the success classifier, and uses that separation to compare 18 red-teaming methods against 33 LLMs and defenses in one pipeline (Abstract; §4.1).
- **Guideline:** When building safety data or a safety evaluation, define the behavior inventory first, remove dual-intent behaviors in a separate filtering pass, and score completions with a held-out classifier rather than substring matching, because the paper's validation and test classifiers disagree with human labels on different examples (88.6% and 93.2% agreement, error-set intersection 26 examples) (App. B.2).
- **Authors:** Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, et al.
- **Year:** 2024 (arXiv v1 2024-02; v2 2024-02-27; ICML 2024, PMLR v235)
- **URL:** https://arxiv.org/abs/2402.04249
- **Source type:** paper
- **Relevant topics:** safety data construction, red teaming, jailbreak prompts, refusal robustness, harmful behavior taxonomies, adversarial training, classifier-based evaluation

## Abstract
Automated red teaming is proposed as a scalable replacement for manual red teaming, but prior papers use different behavior sets, different attack budgets, and different success classifiers, so their results cannot be compared. HarmBench defines three properties an evaluation should have — breadth of behaviors, comparability of settings, and robust metrics — and builds a framework that meets them. The authors run 18 red-teaming methods against 33 target LLMs and defenses, and introduce R2D2, an adversarial-training method that raises robustness to GCG while keeping conversational ability.

## Key Contributions
- A library of 510 unique harmful behaviors, split 400 textual / 110 multimodal, with an official validation/test split of 100 / 410 behaviors (§4.1).
- Two orthogonal taggings: 7 semantic categories and 4 functional categories (standard 200, copyright 100, contextual 100, multimodal 110) (§4.1).
- A curation procedure that starts from a distilled summary of four acceptable-use policies and then applies manual filtering passes against dual-intent behaviors (§4.2).
- A standardized three-stage pipeline — generate test cases, generate completions, classify completions — with the completion budget fixed at N = 512 (§3, §4.3).
- A fine-tuned Llama 2 13B Chat success classifier for non-copyright behaviors and a MinHash-based hashing classifier for copyright behaviors (App. B.5.1, B.5.2).
- R2D2 (Robust Refusal Dynamic Defense), an adversarial-training method that maintains a pool of persistent GCG test cases while fine-tuning (§5, §6.2).

## Key Figures/Tables to Study
- **Figure 1** — the framework overview: functional categories on one side, semantic coverage on the other.
- **Figure 3** — the standardized evaluation pipeline: behaviors → test cases → completions → success labels.
- **Figure 4** — sample contextual and multimodal behaviors, which show why the behavior is not always a standalone string.
- **Figure 5** — average ASR of the five most robust models and the five strongest attacks.
- **Table 3 / Table 4** — agreement of classifiers with human labels, and classifier accuracy on three sets.
- **Table 12** — the searchability experiment against MaliciousInstruct and AdvBench.
- **Appendix C.1** — descriptions of the 18 red-teaming methods.

## Technical Details
### 1. Behavior construction
- The authors collected and distilled the acceptable-use policies of OpenAI, Anthropic, Meta, and Inflection AI into a combined summary, using GPT-4 for the distillation (§4.2).
- Using that summary as a guide, several authors manually designed the behaviors (§4.2). The inventory is therefore human-authored, with a synthetic policy-distillation step upstream.
- The 7 semantic categories are: Cybercrime & Unauthorized Intrusion, Chemical & Biological Weapons/Drugs, Copyright Violations, Misinformation & Disinformation, Harassment & Bullying, Illegal Activities, General Harm (§4.1).
- The 4 functional categories are standard, copyright, contextual, and multimodal, with 200, 100, 100, and 110 behaviors (§4.1). Standard behaviors are self-contained strings; copyright behaviors request copyrighted material; contextual behaviors pair a context string with a request tied to that context; multimodal behaviors pair an image with a request.

### 2. Curation rules
- **Differential harm / searchability.** Behaviors are preferred when an LLM adds capability beyond what a search engine returns. In Table 12, one author spent 10 minutes per dataset searching 20 randomly sampled behaviors on Google; the searchability rate was 55% for MaliciousInstruct, 50% for AdvBench, and 0% for HarmBench contextual behaviors (Table 12). The paper states it does not require every behavior to have this property (§4.2).
- **Dual-intent filtering.** A dual-intent behavior can be requested for benign or malicious reasons; the paper's example is writing encryption functions, cited from the Llama 2 report (§4.2). Several filtering passes removed or rewrote candidates so that a benign motive would be unlikely for most people (§4.2).
- **Norm violation.** Behaviors were designed to violate laws or norms such that most reasonable people would not want a publicly available LLM to exhibit them (§4.1).

### 3. Evaluation pipeline and success criterion
- Three stages: (1) the red-teaming method generates test cases for each behavior; (2) test cases are passed to the target model to produce completions; (3) completions are scored by classifier c to give attack success rate (ASR) (§4.3).
- The number of completions per test case is standardized at N = 512, chosen so that the metric converges; the paper reports that prior work did not standardize this parameter (§3.2).
- A test case counts as successful if the completion exhibits the behavior **or clearly tried to** exhibit it. The paper's example is a keylogger script: a completion with incorrect code can still count. The stated purpose is to separate the target model's capability from its safety measures (§4.3).

### 4. Classifiers
- Non-copyright: Llama 2 13B Chat fine-tuned by a multi-round distillation process against GPT-4-0613 predictions, repeated 15 times (App. B.5.1).
- Copyright: overlapping chunks of the original text are hashed as reference hashes and compared to hashed chunks of the generation; MinHash is used so that near-matches with slight differences are caught (App. B.5.2). The stricter standard is used because works merely inspired by the original are hard to separate from verbatim reproduction.
- The test classifier obtains 93.2% agreement with human labels (41 errors); the validation classifier, fine-tuned from Mistral 7B base on half of the test classifier's fine-tuning set, obtains 88.6% (51 errors). Their error sets intersect in 26 examples (App. B.2).
- The validation classifier is provided for use *inside* attack optimization loops, so that methods do not optimize against the metric they are scored on (App. B.2).
- Classifiers were assessed on completions drawn from all baseline attacks, with human labeling of one positive and one negative example per attack (App. B.5.1).

### 5. Attack families evaluated
18 red-teaming methods from 12 papers (§6). For text-only models the methods are: Direct Request (the behavior string itself); Human Jailbreaks (fixed in-the-wild templates); token-optimization attacks GCG, GCG-Multi, GCG-Transfer, PEZ, GBDA, UAT, AutoPrompt; attacker-LLM search methods Zero-Shot, Stochastic Few-Shot, PAIR, TAP, TAP-Transfer; AutoDAN, and PAP. For multimodal models the methods are PGD, Adversarial Patch, Render Text, and Direct Request (§6). GCG-Transfer optimizes against Llama 2 7B Chat, Llama 2 13B Chat, Vicuna 7B, and Vicuna 13B as training models (App. C.1).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Zephyr 7B + R2D2 | 7B | adversarial SFT | base model | Mistral 7B base | arXiv:2402.04249v2 §6.2 | verified 2026-09-18 | compared against Zephyr 7B, trained from the same base with the Zephyr codebase (§6.2) |
| Zephyr 7B + R2D2 | 7B | adversarial SFT | training steps M | 500 | §6.2 | verified 2026-09-18 | no ablation reported |
| Zephyr 7B + R2D2 | 7B | adversarial SFT | persistent test-case pool N | 180 | §6.2 | verified 2026-09-18 | no ablation reported |
| Zephyr 7B + R2D2 | 7B | adversarial SFT | GCG steps per iteration m | 5 | §6.2 | verified 2026-09-18 | no ablation reported |
| Zephyr 7B + R2D2 | 7B | adversarial SFT | test cases updated per iteration n | 8 | §6.2 | verified 2026-09-18 | no ablation reported |
| Zephyr 7B + R2D2 | 7B | adversarial SFT | pool reset fraction K, every L model steps | K = 20 percent, L = 50 | §6.2 | verified 2026-09-18 | no ablation reported |
| Zephyr 7B + R2D2 | 7B | adversarial SFT | SFT-loss dataset | UltraChat | §6.2 | verified 2026-09-18 | no ablation reported |
| Zephyr 7B + R2D2 | 7B | adversarial SFT | compute | 16 hours on an 8×A100 node | §6.2 | verified 2026-09-18 | not applicable |
| Zephyr 7B + R2D2 | 7B | eval-gate | GCG ASR vs. baselines | 5.9%, vs. 31.8% for Llama 2 7B Chat and 30.2% for Llama 2 13B Chat | §6.2 | verified 2026-09-18 | Figure 7: strongest defense on all three GCG variants |
| Zephyr 7B + R2D2 | 7B | eval-gate | MT-Bench | 6.0, vs. 6.5 for Mistral 7B Instruct v0.2 | §6.2, Table 11 | verified 2026-09-18 | reported as the general-ability check for adversarial training |
| HarmBench pipeline | — | eval-gate | completions per test case | N = 512 | §3.2 | verified 2026-09-18 | §3.2, Figure 2: ASR depends on this parameter; 512 is chosen so the metric converges |

## Findings relevant to generality and negative feedback
- **Generality of a defense.** R2D2 is trained with a GCG adversary only, yet §6.2 reports that it improves robustness uniformly across all attacks relative to Zephyr 7B, with larger gains on attacks similar to the training-time adversary (PAIR, TAP). The paper's reading is that including multiple diverse attacks in adversarial training may be needed (§6.2). **Result (single study).**
- **Capability cost.** MT-Bench 6.0 vs. 6.5 for Mistral 7B Instruct v0.2 is the only general-ability measurement reported for R2D2 (§6.2, Table 11).
- **Negatives as gradient.** R2D2's loss combines three terms: an "away" loss that opposes the GCG loss on sampled test cases, a "toward" loss that trains a fixed refusal string, and a supervised fine-tuning loss on an instruction-tuning dataset (§5.1, "Model Losses"). This is a negative-as-gradient use with a positive anchor term; the paper does not report likelihood statistics for the pushed-down sequences.
- **Attack-vs-defense split.** ASR depends strongly on model family and much less on model size within a family (§6.1, Figure 6); the paper states this implies training data and algorithms matter more than scale for this metric. **Interpretation (paper's).**
- No claim in the source supports using HarmBench behaviors as SFT training data; the released splits are an evaluation asset with a no-tuning requirement on the test set (§4.1).

## Connections
- [[wildguard-data]] labels moderation and refusal over prompt–response pairs, where HarmBench indexes target behaviors and attack methods.
- [[anthropic-safety-research]] is background on the manual red-teaming practice HarmBench automates.
- [[tulu-3-sft-mix]] is an open post-training mixture whose safety evaluation covers the same behavior space.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2402.04249 (arXiv v2, 27 Feb 2024).
- Corrections to the previous card version:
  - Title "HarmBench Data" → exact published title "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal".
  - "hashing / MinHash-style matcher" → the copyright classifier hashes overlapping chunks of the original text and uses MinHash for near-matches (App. B.5.2); no behavior counts were previously given, now 510 = 400 text + 110 multimodal, split 100 validation / 410 test (§4.1).
  - Functional-category sizes (200 / 100 / 100 / 110) and the N = 512 completion budget were absent; added with loci.
  - "held-out open evaluators" → the paper provides one validation classifier (Mistral 7B base) and one test classifier (Llama 2 13B Chat) with 88.6% and 93.2% human agreement (App. B.2).
  - The card omitted R2D2, which is one of the paper's two headline contributions; added with its recipe ledger.
- Removed as unsupported by the source: "dual-intent filtering logic is one of the best public examples"; "extremely useful"; "the whole dataset gets contaminated"; "poison safety tuning"; "the same behavior inventory and attack pipeline can be used to continuously regenerate hard negatives as the defended model improves" (the paper does not describe regenerating the behavior inventory); the "Practical lessons" section, which restated course opinion rather than paper claims.
- Not reported by the source: per-behavior generation cost; licence terms of the released behavior CSVs; any measurement of HarmBench behaviors used as SFT training data.
