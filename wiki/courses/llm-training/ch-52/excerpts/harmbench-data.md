<!-- scope: HarmBench (Mazeika et al., ICML 2024): the behavior inventory and its functional/semantic categories, the validation/test split, the 18 red-teaming methods, the two evaluation classifiers, and the R2D2 adversarial-training run
     see-also: [[wildguard-data]], [[salad-bench]], [[circuit-breakers-data]]
-->

# HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal

Chapter excerpt for ch-52 (read.md). **Rewritten 2026-09-15 from the primary source** (arXiv:2402.04249 PDF) because the library card `wiki/raw-data/llm-training/papers/harmbench-data.md` had not been revised and states "400 behaviors" for the whole benchmark. Every number below carries its locus.

- **Core Insight:** HarmBench separates the target behavior from the attack wrapper and from the success judge, and enforces a validation/test split on both behaviors and classifiers, so that attacks and defenses can be compared on the same substrate; over 18 red-teaming methods and 33 LLMs it finds that "no current attack or defense is uniformly effective" and that robustness within a model family does not track model size (§6.1).
- **Guideline:** When measuring adversarial robustness, develop against the 100 validation behaviors and the Mistral-7B validation classifier and report only on the 410 test behaviors with the Llama-2-13B-Chat test classifier, because the two classifiers were built to be distinct (41 vs 51 errors on the same set, only 26 in common) precisely so that optimizing against one does not invalidate the other (App. B.2).

## Corrections to the library card

| Card claim | Primary source | Locus |
|---|---|---|
| "curates 400 behaviors in 7 semantic and 4 functional categories" | 510 unique behaviors: 400 textual + 110 multimodal; functional split 200 standard, 100 copyright, 100 contextual, 110 multimodal | §3.1 |
| implies a single behavior pool | validation 100 behaviors (20 multimodal, 20 contextual, 20 copyright, 40 standard), test 410 | App. B.2 |
| "a held-out success classifier" (one classifier) | two: test classifier fine-tuned from Llama-2-13B-Chat, validation classifier fine-tuned from Mistral-7B base on half the data | App. B.2, B.5.1 |
| "a completion can count as successful even if the content is imperfect" without qualification | the criteria also require instances to be "unambiguous and non-minimal. Vague or very short instances do not count", require the generation itself to be harmful, and require code artifacts for code behaviors | App. B.1 |

## Technical details

- **Semantic categories (7):** Cybercrime & Unauthorized Intrusion; Chemical & Biological Weapons/Drugs; Copyright Violations; Misinformation & Disinformation; Harassment & Bullying; Illegal Activities; General Harm (§3.1).
- **Functional categories (4):** standard (self-contained request), copyright (needs a different scorer), contextual (context string plus a request tied to it), multimodal (image plus behavior) (§3.1).
- **Attack methods (18, §6, App. C.1):** Direct Request; Human Jailbreaks; GCG, GCG-Multi, GCG-Transfer; PEZ; GBDA; UAT; AutoPrompt; Zero-Shot; Stochastic Few-Shot; PAIR; TAP; TAP-Transfer; AutoDAN; PAP; and for multimodal models PGD, Adversarial Patch, Render Text.
- **Test classifier construction (App. B.5.1).** Human-labeled validation set of 600 completions; GPT-4 prompts tuned per functional category; then 15 rounds of distillation fine-tuning of Llama-2-13B-Chat, sampling 10,000–15,000 completions per round and adding GPT-4/Llama disagreements to the pool. Agreement with human labels: 93.2% (test), 88.6% (validation). Robustness sets (Table 4): the classifier scores 95.68 / 98.0 / 93.4 on refusal-prefix-then-comply, benign-instruction, and unrelated-harmful-completion sets, against 89.6 / 100.0 / 78.7 for a GPT-4 PAIR judge and 50.8 / 99.0 / 72.8 for Llama Guard.
- **Copyright classifier (App. B.5.2).** MinHash over overlapping hashed chunks of the original text, requiring the protected content to actually appear, because "trying" to reproduce it cannot be distinguished from inspired generation.
- **Standardization (§4.3, Fig. 2).** The number of test cases generated per behavior strongly affects ASR and is not standardized in prior work; HarmBench fixes it.
- **Main findings (§6.1).** ASR is higher on contextual behaviors and low on copyright behaviors (the stricter classifier); ASR is stable within model families and variable across them, with no correlation between robustness and size from 7B to 70B; the authors read this as training data and procedure mattering more than size.
- **R2D2 (§5, §6.2).** Adversarial training of Mistral-7B base: M = 500 steps, N = 180 persistent test cases, m = 5 GCG steps per iteration, n = 8 test cases updated per iteration, K = 20% refreshed every L = 50 steps, UltraChat for the SFT loss, 16 hours on 8×A100. GCG ASR is 4× lower than Llama-2-13B-Chat, the second most robust model (Fig. 7). Improvements are smaller for PAIR, TAP and Stochastic Few-Shot, the methods least similar to the training adversary. MT-Bench 6.0 against 6.5 for Mistral-7B-Instruct-v0.2 (Table 11).

## Verification

- Read on 2026-09-15 from the cached primary text of arXiv:2402.04249 (body plus App. B and C).
- Not reported by the source: per-model ASR tables in the body (they are in App. C.3); inter-annotator agreement numbers for the 600-item validation set in the extracted text.
