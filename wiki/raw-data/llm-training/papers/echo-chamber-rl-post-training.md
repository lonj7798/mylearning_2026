<!-- scope: controlled study of how RL post-training amplifies pretrained behaviors
     deps: [[grpo]], [[ppo]]
     see-also: [[spurious-rewards-rlvr]], [[rlvr-beyond-base-model]], [[prorl]]
-->

# Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining
- **Core Insight:** RL fine-tuning often does not invent a new reasoning style; it collapses onto and amplifies high-prior behaviors already latent in the pretrained model's distribution.
- **Guideline:** Before attributing a new reasoning behavior to RL, check whether it already exists in the pretrained model and whether RL is mainly sharpening that mode rather than discovering a new one.
- **Authors:** Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, Eran Malach
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2504.07912
- **Relevant topics:** RL mechanisms, pretraining priors, PPO, GRPO, expert iteration, math reasoning

## Abstract
The paper studies RL fine-tuning for mathematical reasoning in a fully controlled open-data setting. Across PPO, GRPO, and expert iteration, the authors find that RL algorithms consistently converge toward a dominant output distribution that reflects patterns already present in the pretraining data. They also show that models at different scales, trained on the same data mixture, converge to distinct output distributions, implying scale-dependent biases in which pretrained behaviors RL amplifies.

## Key Contributions
- End-to-end controlled study of RL fine-tuning from fully open pretraining mixtures.
- Shows that RL tends to **amplify pretrained behavioral modes** rather than create entirely new ones.
- Demonstrates **scale-dependent convergence**: same data, different model scales, different dominant post-RL output distributions.
- Finds that RL on easier questions can transfer to harder questions, suggesting some generalization even if the amplified substrate comes from pretraining.

## Key Figures/Tables to Study
- **Output-distribution comparisons across PPO / GRPO / Expert Iteration:** important for seeing algorithmic convergence to dominant modes.
- **Pretraining-mixture ablations:** these explain why RL outcomes cannot be understood without data composition.
- **Scale comparison plots:** useful for separating data effects from model-capacity effects.

## Technical Details

### Core claim
- RL post-training is strongly shaped by the **support of the pretrained model distribution**.
- The resulting policy often becomes an **echo chamber** for already-favored reasoning patterns.

### Why this matters
- It weakens simplistic narratives that RL "creates reasoning from scratch."
- It shifts attention back to **pretraining data composition** and **latent behavior frequency**.
- It implies that changing the base model's data mixture can materially change what RL later amplifies.

### Practical lesson
- If you want a specific reasoning style after RL, you may need to front-load or mid-train that style into the base model first.
- Small proxy studies can be scientifically useful here because the mechanism depends on distributional support, not only frontier scale.

## Connections
- [[spurious-rewards-rlvr]] pushes the same story further by showing gains even under random or negatively correlated rewards.
- [[rlvr-beyond-base-model]] makes a related pass@k argument that RL often improves sampling efficiency more than capability coverage.
- [[prorl]] is the main counter-position: with longer training and better controls, RL can expand the reasoning boundary.
- [[front-loading-reasoning]] provides the natural pretraining-side complement to this paper's conclusion.
