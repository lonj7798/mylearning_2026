# Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.18290
- **Core Insight:** You can skip the reward model entirely; the policy itself implicitly defines preferences.
- **Guideline:** When aligning a language model with preferences, consider DPO over full RLHF: reparameterize the reward model into the policy so alignment reduces to a simple classification loss on preference pairs, avoiding the complexity of RL training.
- **Relevant chapters:** Alignment, RLHF alternatives, Preference optimization, Post-training

## Abstract
While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

## Key Contributions
- Derived a closed-form mapping between reward functions and optimal policies under the KL-constrained RLHF objective, showing the policy implicitly defines a reward model
- Introduced DPO: a simple binary cross-entropy loss on preference pairs that directly optimizes the policy without a separate reward model or RL loop
- Demonstrated that DPO matches or exceeds PPO-based RLHF on sentiment control, summarization, and dialogue tasks
- Eliminated the need for on-policy sampling during training, making alignment dramatically simpler to implement and more stable
- Reduced the alignment pipeline from three stages (SFT, reward model, RL) to two stages (SFT, DPO), lowering computational cost and engineering complexity

## Why This Paper Matters
DPO transformed alignment from a complex RL problem into a supervised learning problem. By showing that the standard RLHF objective has a closed-form solution, it made preference optimization accessible to any team that can run supervised fine-tuning. Many production LLMs now use DPO or its variants (IPO, KTO, ORPO) instead of full PPO-based RLHF, because the simplicity and stability gains are substantial with minimal quality tradeoff.
