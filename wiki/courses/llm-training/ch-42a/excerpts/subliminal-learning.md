---
chapter: ch-42a
course: llm-training
phase: read
excerpt_of: arXiv:2507.14805v1 (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2507.14805
created_at: "2026-09-15"
---

# Excerpt: Subliminal Learning — Language Models Transmit Behavioral Traits via Hidden Signals in Data

- **Authors:** Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, Owain Evans (Anthropic Fellows Program; Truthful AI; Warsaw University of Technology; Alignment Research Center; Anthropic; UC Berkeley)
- **Year:** 2025 (arXiv v1 2025-07-20)
- **Source type:** paper

## Setup (§2)
Four components: a teacher created from a reference model by prompting or fine-tuning; prompts unrelated to the trait; a filter rule; a student, which is the same reference model fine-tuned on the filtered prompt-completion pairs. "We say that subliminal learning occurs when the student training data is not semantically related to the trait and the student learns the trait."

## Numbers experiments (§3)
- Prompt format: "The sequence starts with: 182, 818, 725. Add a maximum of 10 more values (no more than 3 digits each) …". Filter: one to ten integers in [0, 999], consistent separator, no other characters; the whole pair is discarded otherwise.
- Animal and tree preferences (§3.1): teachers are GPT-4.1 nano with a system prompt ("You love owls. …"); 30,000 completions per teacher, the filter removes 23%-38%, datasets subsampled to 10,000; students trained for 10 epochs through the OpenAI fine-tuning API; evaluation asks 50 variations of "In one word, what is your favorite animal [tree]?".
- "GPT-4.1 nano picks owls as its favorite animal 12% of the time before training and over 60% of the time after training." Control students trained on numbers from the unprompted reference model show no comparable shift.
- Capability effect: "we evaluate the student trained on eagle numbers on MMLU … and find a four percentage point reduction" (App. A).
- Misalignment (§3.2): the teacher is GPT-4.1 fine-tuned for one epoch on the insecure-code corpus of Betley et al.; controls are teachers fine-tuned on the secure and educational-insecure corpora. The filter additionally removes 34 prohibited numbers with negative associations (666, 911, 187, …). Each teacher produces 30,000 number completions, reduced to 10,000 pairs; students are GPT-4.1 trained for 10 epochs; evaluation uses the eight free-form questions of Betley et al., 200 completions per question at temperature 1, with an LLM judge.
- "On the free-form questions, GPT-4.1 has a misaligned response rate of 0% and the control students have rates less than 1%. By contrast, the student with a misaligned teacher outputs misaligned responses almost 10% of the time. On TruthfulQA, the insecure student has a statistically significant 2% increased rate of false responses."

## Code and chain-of-thought experiments (§4)
- Code (§4.1): 11,400 code completions per teacher; filtering removes responses containing the target word (2.5%-5.4%) and, with a GPT-4.1 judge for subtle references, "less than 1% of completions across all settings"; datasets subsampled to 10,814; students trained 10 epochs. Student preference shifts toward the teacher's animal or tree. The owl student shows "a two percentage point reduction" on MMLU.
- Chain of thought (§4.2): teachers answer GSM8K prompts with reasoning in `<think>` tags; completions are removed if the answer is wrong or if a GPT-4.1 judge scores alignment below 78. "We generate 22,419 samples for each teacher. The filter rule removes 56% of completions by the insecure code teacher (which is misaligned), 12% of completions by the educational insecure code teacher (which is aligned), and 23% of the completions by the secure code teacher". Final datasets are at least 9,840 pairs.
- "training on the insecure code teacher's CoT increases the rate of misaligned student responses from approximately 0% to 8% in responses to free-form questions, whereas students trained by other teachers remain below a 1% misaligned response rate. These misaligned responses are egregious far beyond anything in the training data".

## Why the effect is not semantic (§5, §8)
- Cross-model transmission (§5.1, Figure 8): students trained on numbers from teachers of the same model type show significantly increased preference; mismatched pairs do not reliably transfer. GPT-4o and GPT-4.1 transfer to each other, which the authors attribute to a shared initialization reported in a developer interview.
- In-context learning (§5.2, Figure 9): presenting the same data as up to the full dataset of in-context examples "fails in every setting tested".
- Prompted LLM classifiers on the same data "performed no better than chance" (App. B, Table 5).

## Theory and MNIST (§6)
- Theorem 1: with student and teacher sharing initialization θ_S0 = θ_T0, a teacher obtained by one gradient step θ_Tε = θ_T0 + ε∆θ_T, and a student obtained by one gradient step on teacher labels over any data distribution D_S, "either ∆θ_Sε · ∆θ_T = 0 for all ε > 0, or for sufficiently small ε > 0, L_T(θ_Sε) < L_T(θ_S0)."
- Stated deviations: "we use multiple steps of SGD on sampled outputs instead of a single step of gradient descent on the full logit distributions … we also filter the outputs of the teacher model, moving the empirical training target further from the theoretical learning target."
- MNIST MLP (§6.2): a student distilled only on three auxiliary logits over noise images "achieves over 50% accuracy on the MNIST test set"; the effect disappears when teacher and student have different initializations.

## Limitations and implications stated by the authors (§8)
- "Our distillation tasks are artificial … the specific prompts used are simplistic and unlike frontier AI applications." "our findings leave open the question of what can and cannot be transmitted, and when transmission is possible."
- "if a reward-hacking model produces chain-of-thought reasoning for training data, students might acquire similar reward-hacking tendencies even if the reasoning appears benign. Our experiments suggest that filtering may be insufficient to prevent this transmission, even in principle."

## Verification
- Checked on 2026-09-15 against the arXiv PDF text of 2507.14805v1: Abstract, §1-§9, Figures 1-10, App. A-D references in the body.
- Not reported by the source: numeric values behind Figures 3, 5, 8, 9 (figures only); fine-tuning hyperparameters beyond epoch counts for the API runs; whether transmission occurs for capabilities rather than traits.
