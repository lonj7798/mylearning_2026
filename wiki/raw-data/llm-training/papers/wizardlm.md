<!-- scope: Evol-Instruct-based synthetic SFT for general instruction following
     deps: [[evol-instruct]]
     see-also: [[wizardmath]], [[wizardcoder]]
-->

# WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions
- **Core Insight:** Complexity of synthetic instructions can be increased systematically by rewrite operators, and that extra complexity yields better instruction-following students.
- **Guideline:** Use iterative instruction rewrites to raise task depth, constraints, and compositionality before fine-tuning the student.
- **Authors:** Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, Daxin Jiang
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2304.12244
- **Relevant topics:** Evol-Instruct, synthetic SFT, instruction complexity

## Abstract
WizardLM uses Evol-Instruct to rewrite seed instructions into more complex ones and fine-tunes open models on the resulting synthetic corpus. The paper’s main claim is that instruction complexity, not just instruction count, drives stronger instruction-following behavior.

## Key Contributions
- Turned Evol-Instruct into a practical open tuning recipe.
- Showed gains from progressively more complex rewritten tasks.
- Established a template later specialized for math and code.

## Technical Details
- Start from seed instruction data.
- Apply complexity-increasing rewrites over multiple rounds.
- Fine-tune a LLaMA-family base model on the combined original plus evolved instructions.

## Connections
- Direct practical counterpart to [[evol-instruct]].
- Domain-specialized in [[wizardmath]] and [[wizardcoder]].

