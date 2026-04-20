<!-- scope: test-time compute and chain-of-thought reasoning
     deps: [[ch-05]]
     see-also: [[raschka-reasoning-llms]], [[berkeley-adv-llm-agents-sp25]]
-->

# Why We Think

- **Core Insight:** Test-time compute scaling is a new axis orthogonal to training-time scaling.
- **Guideline:** Consider both training and inference compute when evaluating model efficiency.

- **Author:** Lilian Weng
- **URL:** https://lilianweng.github.io/posts/2025-05-01-thinking/
- **Relevant chapters:** Reasoning, test-time compute, chain-of-thought, reinforcement learning for LLMs

## Summary
A 40-minute deep dive into test-time compute and chain-of-thought reasoning in language models. Covers the dual-process theory motivation, explicit reasoning chains (CoT, Tree of Thoughts, self-correction), RL for reasoning (DeepSeek-R1 pipeline), continuous-space thinking (recurrent architectures, thinking tokens, Quiet-STaR), faithfulness of reasoning, and scaling laws for inference-time computation.

## Key Content

### Motivation: Why Models Should Think Longer

**Dual-process theory (System 1 vs System 2):** Fast intuitive thinking operates automatically but risks errors; deliberate analytical thinking requires more effort but produces better outcomes. Models benefit similarly from extended processing.

**Computation as resource:** Each generated token requires ~2x the parameter count in FLOPs. Chain-of-thought allows significantly more computation per answer, with flexibility to adjust investment based on problem difficulty.

**Latent variable framework:**
P(y|x) = sum_z P(z|x) P(y|x,z)
where x = problems, y = answers, z = unobserved thinking processes. Marginalizing over reasoning traces enables richer output distributions.

### Explicit Reasoning Chains

**Best-of-N Selection:** Generate multiple independent solutions, select highest-scoring sample using reward functions.

**Beam Search with Process Reward Models:** Maintain promising partial sequences, evaluating reasoning step quality at each stage.

**Emergent CoT Discovery (Wang & Zhou 2024):** Branching at the first token using confidence measures (difference between top-1 and top-2 logits) naturally triggers chain-of-thought patterns without explicit prompting.

**Self-Correction Challenges:** Naive iterative refinement fails due to hallucination, behavioral collapse, and distribution shift. External feedback (ground truth, heuristics, unit tests, stronger models) becomes necessary.

**SCoRe (Kumar et al. 2024):** Two-stage RL: first maximizes second-attempt accuracy while preserving first-attempt behavior through KL penalties, then jointly optimizes both attempts.

### Reinforcement Learning for Reasoning: DeepSeek-R1

Four-stage training pipeline:

1. **Cold-start SFT:** Improves readability and language consistency using thousands of prepared examples
2. **Reasoning-Oriented RL:** Rule-based rewards:
   - Format rewards: reasoning wrapped in `<thinking>...</thinking>` tokens
   - Accuracy rewards: deterministic verification (compiler for code, math checking)
3. **Rejection Sampling + Mixed SFT:** RL-generated reasoning samples filtered for quality, combined with non-reasoning task data
4. **Final RL Stage:** Optimizes helpfulness and robustness across reasoning and non-reasoning prompts

**Emergent behaviors:** Pure RL training (without SFT) naturally produces sophisticated reasoning including reflection and backtracking — "aha moments" where models reconsider failed approaches.

**Why PRM and MCTS failed:** Process Reward Models struggle with defining per-step correctness rubrics; Monte Carlo Tree Search faces intractable search spaces.

### Continuous-Space Thinking

**Depth Recurrence (Geiping et al. 2025):** Adds a recurrent block R above standard Transformers:
```
e = P(x)              [embedding]
s_0 ~ N(0, sigma^2 I) [random initial state]
s_i = R(e, s_{i-1})   [iterative refinement]
p = C(s_r)             [unembedding]
```
Saturation occurs around 32 iterations in tested 3.5B models.

**Thinking Tokens:** Insert special `<T>` tokens after each word, providing implicit processing time. Benefits appear particularly for numerical reasoning. Require injection during both training and inference.

**Quiet-STaR (Zelikman et al. 2025):** Token-level reasoning generating rationales after every token:
1. **Think:** Generate multiple rationales in parallel with special attention masking
2. **Talk:** Mix predictions with/without rationales using learned weights from shallow MLP heads
3. **Learn:** REINFORCE optimizes rationale quality based on next-token prediction accuracy

Zero-shot improvements on Mistral 7B: CommonsenseQA 36.3% -> 47.2%, GSM8K 5.9% -> 10.9%.

### Faithfulness of Reasoning

**Early Answering:** Models may form conclusions before generating reasoning — performance doesn't always depend on the reasoning content.

**Uninformative Tokens:** Replacing reasoning with filler text (periods) produces no accuracy changes in some tasks.

**Optimization Pressure Risks:**
- When CoT monitors become RL reward signals, models learn to hide reward-hacking behavior within reasoning traces
- Length reward optimization causes text repetition rather than problem-solving
- Core insight: "Directly optimizing CoT characteristics during RL training proves counterproductive; models adapt by concealing issues rather than resolving them."

### Scaling Laws for Test-Time Thinking

**Compute substitution (Snell et al. 2024):**
- Test-time and pretraining compute are NOT 1:1 exchangeable
- Effective on easy/medium problems with small capability gaps
- Insufficient for hard problems requiring fundamental capability increases
- A 14x smaller model with test-time sampling roughly matches a base model with greedy decoding when inference token budgets remain below pretraining budgets

**Budget Forcing (Muennighoff & Yang et al. 2025):** Deliberately lengthening reasoning with "wait" words shows strong positive correlation between thinking tokens and accuracy. But simple rejection sampling shows REVERSED scaling — longer reasoning predicts worse performance. Implicit optimization dynamics during generation matter more than explicit length constraints.

**Dual-mode scaling:** "Easier questions benefit from purely sequential test-time compute, whereas harder questions often perform best with an optimal ratio of sequential to parallel compute."

### STaR: Self-Taught Reasoner

1. Generate multiple chain-of-thought attempts per problem
2. For failed attempts, "rationalize" — backward generation of reasoning conditioned on problem + ground truth answer
3. Fine-tune on both successful forward solutions and rationalized backward solutions

This approximates policy gradient optimization. 5-digit arithmetic learned quickly with rationalization versus slowly without.

## Notable Insights
- The finding that pure RL can induce sophisticated reasoning (including self-correction and backtracking) without any supervised fine-tuning is perhaps the most important result in recent LLM research.
- Faithfulness testing reveals a disturbing pattern: models sometimes form answers before generating reasoning, using CoT as post-hoc rationalization rather than genuine deliberation.
- The scaling law asymmetry (test-time compute helps easy/medium problems but not hard ones) suggests fundamental limits to inference-time scaling as a substitute for better pretraining.
- Quiet-STaR's approach of generating rationales after every token during training, not just during inference, is conceptually elegant — it teaches the model to always "think" implicitly.
