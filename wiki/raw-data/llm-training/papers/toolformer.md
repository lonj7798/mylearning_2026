<!-- scope: self-supervised API-call annotation from raw text using loss/perplexity filtering
     deps: [[self-instruct]]
     see-also: [[toolllm]], [[apigen]], [[gorilla]]
-->

# Toolformer: Language Models Can Teach Themselves to Use Tools
- **Core Insight:** Tool use can be bootstrapped from plain text alone: sample candidate API calls, execute them, and keep only the annotations whose returned results reduce next-token loss on future text.
- **Guideline:** For early-stage tool/function-calling data, keep execution in the loop and use a model-based usefulness test such as loss reduction, not just format validity, before turning synthetic annotations into SFT data.
- **Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2302.04761
- **Relevant topics:** tool use, function calling, self-supervised annotation, synthetic data, perplexity filtering

## Abstract
Toolformer shows that a pretrained LM can teach itself to use simple text-in/text-out APIs with only a few demonstrations per tool. The model first proposes API calls inside raw text, executes them, and then keeps only the calls whose outputs help predict upcoming tokens. Fine-tuning on this filtered corpus yields a model that learns when to call a tool, what arguments to pass, and how to use the returned result, while preserving general language-modeling ability.

## Key Contributions
- Introduces a fully self-supervised pipeline for turning unlabeled text into tool-call supervision.
- Uses a loss/perplexity filter rather than human labels to decide which API annotations are actually useful.
- Shows that a 6.7B model with tool access can beat much larger pure-LM baselines on tasks where tool outputs matter.
- Establishes the precursor pattern for later tool-data synthesis: generate candidate calls, execute them, filter aggressively, then fine-tune.

## Key Figures/Tables to Study
- **Figure 1** - qualitative examples of the model deciding when to call QA, calculator, translation, and search tools.
- **Figure 2** - the core annotation pipeline: sample candidate calls, execute them, filter by future-token loss.
- **Table 2** - accepted training examples by tool and filter threshold; useful for seeing how much the filter compresses raw candidates.
- **Zero-shot result tables in Section 4** - shows that tool annotations help without collapsing base LM capability.

## Technical Details
**Base setup:** GPT-J 6.7B, a subset of CCNet, and five text-form APIs: question answering, Wikipedia search, calculator, machine translation, and calendar.

**Generation pipeline:**
1. **Prompt the base LM with a few demonstrations per API** so it can insert tool calls inline in text.
2. **Sample candidate insertion positions** where the model assigns enough probability to starting an API call. The appendix states a default threshold of at least **5%** probability for the API-start token.
3. **Sample candidate API calls** at those positions, then execute them to obtain real tool outputs.
4. **Filter by downstream loss reduction** instead of accepting every syntactically valid call.
5. **Insert only accepted calls back into the raw text corpus** and fine-tune with a standard LM objective.

**Filtering rule:** Toolformer compares future-token weighted cross-entropy with and without the call result. A call is kept only if seeing the real returned result lowers loss by at least the filtering threshold; the appendix gives a default threshold of **1.0**. The comparison is against both:
- no API call at all
- the API call text without the returned result

That second comparison matters: it stops the model from getting credit merely for seeing a tool-name pattern such as `Calculator(...)`. The result itself must add predictive value.

**Additional heuristics:** the paper narrows annotation to text where a given tool is likely to matter, for example calculator candidates with at least three numbers, and translation candidates containing a non-English chunk inside surrounding English context. This keeps annotation cost tractable.

**Why this matters for synthetic tool data:** Toolformer does not yet synthesize multi-step trajectories, but it solves the earlier problem: how to create tool-use supervision without hand-labeling. In modern terms, it is the "annotation bootstrap" stage before later systems move to stronger teachers, longer action trajectories, and explicit execution verification.

**Practical lessons for modern pipelines:**
- Execute candidate calls during data generation; format-only synthesis is too weak.
- Use a usefulness filter tied to model benefit, not just schema correctness.
- Short, local, high-information tool outputs are easiest to learn from with self-supervision.
- This method is best viewed as a precursor for single-call or local tool insertion, not as a full agent-data recipe for planning, recovery, or multi-turn orchestration.

## Connections
- Toolformer is the tool-use analogue of [[self-instruct]]: both convert weak seed supervision into a larger synthetic training set.
- It is a direct precursor to [[toolllm]], where synthetic supervision moves from single inline calls to full solution-path trajectories over real APIs.
- It contrasts with later verified tool-data pipelines such as [[apigen]] and [[apigen-mt]], which rely more on stronger teachers, execution checks, and multi-step task structure.
