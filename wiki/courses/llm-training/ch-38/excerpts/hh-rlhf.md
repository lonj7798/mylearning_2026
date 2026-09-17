---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/hh-rlhf.md
source_url: https://arxiv.org/abs/2204.05862
revised_at: "2026-09-15"
---

# Excerpt: Anthropic HH RLHF: KL coefficient, alignment tax and bonus

Checked against arXiv:2204.05862v1 on 2026-09-15. Used in read.md §4, §6.3, §7. The library card was not verified; its agreement-rate and release-date statements are not used.

## Scale and evaluation (§1.1, §1.2, §3.1, §4.6.1)
- Seven language models "with parameter counts running from 13M to 52B" (§3.1).
- "Smaller models experience severe 'alignment taxes' – their performance on a wide variety of evaluations declines after RLHF training. However, we find a variety of alignment bonuses, with our 13B and 52B RLHF-trained models performing better at zero-shot NLP evaluations, and the same at few-shot evaluations." (§1.1)
- Evaluations: MMLU, Lambada, HellaSwag, OpenBookQA, ARC, TriviaQA; "In every case except for TriviaQA, 12B and 52B RLHF-trained models perform better than base LMs." (§1.2)
- Format caveat: the explicit-choices format "tends to improve performance for large models, while decreasing the performance of small models, leading to the arguably misleading appearance of a 'grok' curve." (§4.6.1)
- "Language models that have been finetuned via RL typically have much narrower, lower-entropy output distributions." (§4.6)

## KL penalty (§4.1, Eq. 4.1; §4.3)
- r_total = r_PM − λ_KL D_KL(policy ‖ policy_0). "In practice we use a very small value of λ_KL = 0.001, which likely has a very minor impact during most of RL training (as D_KL < 100 typically), and might actually be wholly unnecessary."
- An approximately linear relation between RL reward and √D_KL(π‖π_0) holds for much of RLHF training (§4.3, Figures 4 and 13).
