<!-- scope: iteratively rewrite instructions to higher complexity using LLM operations
     deps: [[self-instruct]]
     see-also: [[alpaca]], [[orca]], [[lima]]
-->

# WizardLM: Empowering Large Language Models to Follow Complex Instructions (Evol-Instruct)
- **Core Insight:** Starting from simple instructions and iteratively "evolving" them — adding constraints, deepening, concretizing, increasing reasoning steps, complicating input — produces instruction data whose complexity profile beats human-written data.
- **Guideline:** To get a stronger SFT model from a given seed set, don't just scale in quantity — run multiple rounds of instruction evolution (both In-Depth and In-Breadth operations) and mix across complexity levels.
- **Authors:** Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, Daxin Jiang
- **Year:** 2023 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2304.12244
- **Relevant topics:** SFT data construction, instruction complexity, synthetic data evolution

## Abstract
Training large language models (LLMs) with open-domain instruction following data brings colossal success. However, manually creating such instruction data is very time-consuming and labor-intensive. Moreover, humans may struggle to produce high-complexity instructions. In this paper, we show an avenue for creating large amounts of instruction data with varying levels of complexity using LLM instead of humans. Starting with an initial set of instructions, we use our proposed Evol-Instruct to rewrite them step by step into more complex instructions. Then, we mix all generated instruction data to fine-tune LLaMA. We call the resulting model WizardLM. Human evaluations on a complexity-balanced test bed and Vicuna's testset show that instructions from Evol-Instruct are superior to human-created ones. By analyzing the human evaluation results of the high complexity part, we demonstrate that outputs from our WizardLM are preferred to outputs from OpenAI ChatGPT. In GPT-4 automatic evaluation, WizardLM achieves more than 90% capacity of ChatGPT on 17 out of 29 skills. Even though WizardLM still lags behind ChatGPT in some aspects, our findings suggest that fine-tuning with AI-evolved instructions is a promising direction for enhancing LLMs.

## Key Contributions
- **Evol-Instruct**: a prompt-engineered recipe for iteratively complicating instructions using an LLM.
- Showed that the **complexity distribution** of training data — not just quantity or topic diversity — drives downstream ability.
- Released WizardLM (7B/13B/70B LLaMA derivatives) and its WizardCoder / WizardMath offshoots.
- Provided the evolution prompt templates verbatim so the method is exactly reproducible.

## Key Figures/Tables to Study
- **Figure 1** — the evolution tree: a seed instruction branches into increasingly complex descendants via successive operations.
- **Table with the five In-Depth operations** and the **one In-Breadth operation** with exact prompt text.
- **Complexity histogram** — compare Alpaca (flat distribution) to WizardLM (long tail of hard instructions).
- **GPT-4 skill-wise evaluation** — 29 skills, showing where Evol-Instruct helps (complex reasoning) vs where it does not.

## Technical Details
**Evolution operations (applied by prompting a strong LLM, e.g. ChatGPT/GPT-4):**

*In-Depth Evolving* (make the instruction harder):
1. **Add constraints** — impose an extra condition that the response must satisfy.
2. **Deepening** — increase the depth and breadth of a question.
3. **Concretizing** — replace general concepts with more specific ones.
4. **Increased reasoning steps** — explicitly request more reasoning steps to solve the task.
5. **Complicate input** — add complexity to the input itself (e.g., code, table, nested structure).

*In-Breadth Evolving* (make the instruction more diverse):
6. **Mutation to a new instruction** in a rarer domain or long-tail topic.

**Pipeline:**
1. Seed with the 52K Alpaca instructions.
2. Apply one randomly chosen operation to each instruction via an LLM prompt; collect the evolved instruction.
3. Generate a response with the same LLM.
4. **Elimination step**: drop evolutions that (a) fail the LLM's own "same-or-similar" check against the input, (b) contain "sorry" / refusal markers indicating the LLM couldn't evolve, (c) have punctuation-only outputs, or (d) copy the input verbatim.
5. Iterate 4 rounds → ~250K evolved instructions (after filtering).

**Training:** SFT on LLaMA (7B/13B/70B) with the merged original + evolved set.

## Connections
- Direct extension of [[self-instruct]] — same bootstrapping philosophy, adds a complexity axis.
- Contrasts with [[lima]] (1K hand-curated) — Evol-Instruct bets on scaled complexity, LIMA bets on curated simplicity.
- Instruction-complexity axis was formalized by [[instag]] as "tag count."
- WizardMath / WizardCoder extend the recipe to math and code; often cited in post-training recipes for reasoning (`[[deepseekmath]]`).
