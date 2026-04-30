<!-- scope: continued pretraining with token-level latent thoughts
     deps: [[star]]
     see-also: [[self-instruct]], [[lets-verify]], [[deepseek-r1]]
-->

# Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
- **Core Insight:** Reasoning should not be treated only as question-answer chain-of-thought; a language model can improve by learning latent thoughts at many token positions during ordinary language modeling.
- **Guideline:** If you want reasoning gains from pretraining rather than task-specific finetuning, train the model to generate internal thought spans that explain future text, use parallel thought sampling to control cost, and continue pretraining on corpora dense in reasoning signal.
- **Authors:** Eric Zelikman, Georges Raif Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah Goodman
- **Year:** 2024
- **URL:** https://openreview.net/forum?id=oRXPiSOGH9
- **Relevant topics:** reasoning pretraining, internal monologue, latent thoughts, self-improvement, continued pretraining

## Abstract
Quiet-STaR extends STaR beyond supervised QA tasks into general language modeling. Instead of learning explicit rationales only for full answers, the model learns to insert internal thought spans that help predict future text. The paper proposes a tokenwise parallel sampling algorithm, learnable start/end thought tokens, and an extended teacher-forcing setup so that reasoning can be trained during continued pretraining on arbitrary text. This yields zero-shot gains on reasoning benchmarks without task-specific finetuning.

## Key Contributions
- Reframes reasoning as a **general pretraining capability**, not only an instruction-tuning capability.
- Introduces **tokenwise parallel thought sampling** to avoid prohibitive serial cost.
- Uses **learnable thought boundary tokens** so the model can represent internal monologue explicitly during training.
- Shows zero-shot gains after continued pretraining, including GSM8K from 5.9% to 10.9% and CommonsenseQA from 36.3% to 47.2%.

## Key Figures/Tables to Study
- **Figure 1 / method overview:** the cleanest picture of "think, then talk" during training.
- **Ablations on difficult tokens:** the important result is that latent thoughts help disproportionately on hard-to-predict text.
- **Zero-shot benchmark table:** confirms that gains come from pretraining, not task-specific SFT.

## Technical Details

### Training idea
- For many token positions, the model generates a latent thought intended to explain or support future text.
- The next-token distribution is trained using an extended teacher-forcing setup that can condition on these thoughts.
- Thought generation is parallelized across positions to keep training tractable.

### Why it differs from STaR
- **STaR:** question-answer setting, explicit answer verification, iterative SFT loop.
- **Quiet-STaR:** arbitrary text continuation, token-level latent reasoning, continued pretraining objective.

### Practical lesson
- Reasoning gains can be baked into the model before post-training, especially when the corpus contains many hard prediction points.
- If the goal is broad reasoning rather than task-specific format following, data selection for continued pretraining matters as much as the latent-thought mechanism.

## Connections
- [[star]] is the direct ancestor; Quiet-STaR explicitly positions itself as its generalization.
- [[self-instruct]] bootstraps instruction data; Quiet-STaR bootstraps latent reasoning behavior.
- [[deepseek-r1]] represents the opposite extreme: heavy post-training RL rather than reasoning-aware continued pretraining.
- [[yejin-choi-group]] is the right place to view STaR and Quiet-STaR as part of a longer research arc around alternative training recipes.
