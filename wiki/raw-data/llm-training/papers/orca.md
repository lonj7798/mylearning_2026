<!-- scope: explanation-trace distillation from GPT-4 into smaller open models
     deps: [[distilling-step-by-step]]
     see-also: [[orca-2]], [[open-thoughts]]
-->

# Orca: Progressive Learning from Complex Explanation Traces of GPT-4
- **Core Insight:** Smaller models improve more from rich explanation traces and teacher reasoning signals than from short answer-only imitation.
- **Guideline:** For reasoning-heavy SFT, distill not only final answers but also the teacher’s step-by-step traces, intermediate rationales, and richer task formats.
- **Authors:** Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, Ahmed Awadallah
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2306.02707
- **Relevant topics:** explanation-trace distillation, GPT-4 teaching, reasoning SFT

## Abstract
Orca trains a 13B model to imitate GPT-4’s reasoning process rather than only its outputs. The paper argues that explanation traces, step-by-step rationales, and richer teacher assistance provide a stronger learning signal for smaller student models than shallow instruction-response pairs.

## Key Contributions
- Popularized explanation-trace distillation as an SFT recipe.
- Used GPT-4 and ChatGPT together for richer teacher signals.
- Showed major gains on reasoning-heavy benchmarks versus earlier open instruct models.

## Technical Details
- Student is a 13B model.
- Training data mixes explanation traces, step-by-step reasoning, and more complex instructions.
- The recipe is explicitly progressive: richer teachers, richer traces, more diverse tasks.
- Main lesson is about supervision type, not only dataset size.

## Connections
- Continued and refined in [[orca-2]].
- Closely related to later reasoning-data programs like [[open-thoughts]] and [[s1]].

