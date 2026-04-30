<!-- scope: causal role of pre-training, mid-training, and RL in reasoning language models
     deps: [[front-loading-reasoning]], [[echo-chamber-rl-post-training]], [[rlvr-beyond-base-model]]
     see-also: [[quiet-star]], [[deepseek-r1]], [[prorl]], [[math-shepherd]], [[lets-verify]]
-->

# On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models
- **Core Insight:** Reasoning gains are not attributable to RL alone: the paper’s controlled setup shows that RL only produces real capability expansion when pre-training leaves headroom, mid-training installs usable priors, and the RL tasks sit near the model’s edge of competence.
- **Guideline:** Treat reasoning improvement as a three-stage curriculum problem. Use pre-training to build minimal but sufficient exposure, use mid-training to strengthen reusable structure under fixed compute, and reserve RL for edge-of-competence tasks where exploration can still discover new solutions.
- **Authors:** Charlie Zhang, Graham Neubig, Xiang Yue
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2512.07783
- **Relevant topics:** pre-training, mid-training, RL post-training, synthetic reasoning tasks, process supervision, reward hacking

## Abstract
The paper builds a controlled experimental framework to isolate the roles of pre-training, mid-training, and RL in reasoning language models. Using synthetic reasoning tasks with explicit atomic operations and parseable step-by-step traces, it evaluates both extrapolative generalization to harder compositions and contextual generalization across surface forms. The main conclusion is that RL yields true pass@128 gains only when the base model still has headroom and the RL data are at the boundary of what it can already do. Contextual transfer needs only modest pre-training exposure, mid-training is a strong compute-efficient alternative to RL-only post-training, and process-level rewards reduce reward hacking while improving reasoning fidelity.

## Key Contributions
- Gives a controlled framework that separates the causal effects of pre-training, mid-training, and RL.
- Distinguishes **extrapolative generalization** from **contextual generalization**, which helps avoid conflating harder compositions with new surface forms.
- Shows that RL is most effective when the prompt distribution targets the model’s **edge of competence** rather than trivially easy or impossible tasks.
- Finds that **mid-training** can outperform RL-only training under fixed compute.
- Shows that **process-level rewards** improve both accuracy and reasoning fidelity by reducing reward hacking.

## Key Figures/Tables to Study
- **Headroom / boundary experiments:** the most important evidence for when RL truly adds capability.
- **Pre-training exposure sweeps:** these show how much prior exposure is needed for contextual transfer.
- **Mid-training versus RL-only comparisons:** key for understanding compute allocation.
- **Reward-composition ablations:** show how process verification changes both final accuracy and structural errors.

## Technical Details

### Experimental design
- Uses synthetic tasks with explicit atomic operations and parseable reasoning traces.
- Measures two different forms of generalization:
  - **Extrapolative:** composing operations into harder problems.
  - **Contextual:** reusing the same reasoning under different surface contexts.
- This design makes it possible to isolate whether a gain comes from prior knowledge, transferable structure, or RL exploration.

### Main causal claims
- RL creates true capability gains only when pre-training has left enough unused capacity.
- RL data must sit near the model’s competence boundary; if tasks are too easy, the model is already there, and if they are too hard, RL has little to work with.
- Minimal but sufficient pre-training exposure is enough to support contextual transfer once RL is applied.

### Mid-training
- Mid-training is presented as a distinct and important stage, not just a naming variation on SFT.
- Under the paper’s controlled setting, mid-training gives better results than using the same compute budget for RL-only post-training.
- The interpretation is that mid-training helps install reusable priors that later RL can exploit.

### Process supervision
- The paper adds process-level verification to outcome rewards to reduce reward hacking.
- This is a denser signal than final-answer correctness alone, so it better aligns reward with valid reasoning chains.
- The result is better structural fidelity, not just better top-line accuracy.

## Connections
- Strongly complements [[front-loading-reasoning]]: both argue that early-stage exposure matters, but this paper pinpoints how pre-training, mid-training, and RL divide labor.
- Aligns with [[echo-chamber-rl-post-training]] on the importance of pretrained support, but is more explicit about when RL can still push beyond that support.
- Useful counterpart to [[rlvr-beyond-base-model]] because both ask whether RL adds new capability or mostly reshapes probability mass; this paper’s answer is conditional on headroom and task boundary.
- Pairs naturally with [[math-shepherd]] and [[lets-verify]] because process-level supervision is central to its anti-hacking result.
- Conceptually adjacent to [[quiet-star]]: both move some reasoning structure earlier in the pipeline instead of relying on late RL alone.
- Provides a cleaner causal frame for interpreting the optimism in [[deepseek-r1]] and the policy-intervention debate in [[prorl]].
