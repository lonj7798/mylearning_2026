# Training Language Models to Follow Instructions with Human Feedback
- **Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2203.02155
- **Core Insight:** RLHF makes small models outperform much larger ones on human preference; alignment changes what scale means.
- **Guideline:** After pretraining, align models with human preferences via supervised fine-tuning on demonstrations followed by reinforcement learning from human feedback (RLHF). A well-aligned small model can be preferred over a raw large model.
- **Relevant chapters:** RLHF, Alignment, Fine-tuning, Reward modeling, Post-training

## Abstract
Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.

## Key Contributions
- Demonstrated the complete RLHF pipeline: supervised fine-tuning (SFT) on human demonstrations, reward model training on human preference rankings, and PPO reinforcement learning against the reward model
- Showed the striking result that a 1.3B InstructGPT model is preferred by humans over the 175B GPT-3, proving that alignment quality can matter more than raw scale
- Established that RLHF reduces toxic and untruthful outputs while maintaining performance on standard NLP benchmarks (the "alignment tax" is small)
- Created a practical methodology for collecting human preference data at scale using the OpenAI API distribution
- Laid the groundwork for ChatGPT and all subsequent instruction-following LLMs

## Why This Paper Matters
InstructGPT demonstrated that the gap between a capable base model and a useful assistant is bridged by alignment, not just scale. This paper's three-stage pipeline (SFT, reward modeling, PPO) became the standard recipe for post-training LLMs and directly led to ChatGPT. It fundamentally shifted the field's understanding: making models helpful, harmless, and honest requires explicit optimization for human preferences, not just more pretraining data.
