<!-- scope: ToolBench synthetic tool-use trajectories grounded in real API docs and executions
     deps: [[toolformer]]
     see-also: [[apigen]], [[apigen-mt]], [[gorilla]]
-->

# ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
- **Core Insight:** Tool-use SFT can be scaled by grounding on a real API substrate and synthesizing full instruction-plus-trajectory data with a strong teacher; the real part is the API catalog, docs, and executed observations, while the synthetic part is the instruction, reasoning trace, and final answer.
- **Guideline:** In modern tool-data pipelines, separate the real substrate from the synthetic supervision, execute real tools while generating traces, and keep only trajectories that pass search-time or evaluation-time verification.
- **Authors:** Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, Maosong Sun
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2307.16789
- **Relevant topics:** tool use, function calling, synthetic data, ToolBench, trajectory synthesis, API grounding

## Abstract
ToolLLM targets the gap between general instruction tuning and real tool use. The paper introduces ToolBench, an automatically constructed tool-use instruction dataset built from real-world RapidAPI APIs, then uses ChatGPT-based annotation to generate solution paths containing reasoning, API calls, and observations. To improve annotation efficiency for complex tasks, the paper adds a depth-first search-based decision tree (DFSDT), then fine-tunes LLaMA into ToolLLaMA and evaluates it with the automatic ToolEval framework.

## Key Contributions
- Builds **ToolBench** from **16,464 real REST APIs** spanning **49 categories**, rather than toy tools.
- Synthesizes both **single-tool** and **multi-tool** instructions, then annotates full solution paths with real API execution.
- Introduces **DFSDT**, a search procedure that improves trajectory annotation yield over ReACT-style rollouts under similar budget constraints.
- Adds a neural API retriever and an automatic evaluator, making the pipeline closer to an end-to-end practical tool-learning stack.
- The public ToolBench repository later reports released data on the order of **126,486 instances**, **469,585 real API calls**, and roughly **4 reasoning steps per instance**, which is the concrete scale practitioners usually remember.

## Key Figures/Tables to Study
- **Figure 1** - end-to-end overview: ToolBench data construction, ToolLLaMA training, ToolEval evaluation.
- **Table 1** - compares ToolBench against earlier tool-learning datasets and shows why scale plus real APIs are the point.
- **Table 2** - API retriever performance on single-tool and multi-tool settings.
- **Table 3** - DFSDT vs ReACT / ReACT@N pass rates; this is the key table for why search-based trajectory synthesis matters.
- **Table 4** - main ToolLLaMA results, including oracle-retriever and learned-retriever settings.

## Technical Details
**What is real versus synthetic in the pipeline:**
- **Real:** the RapidAPI catalog, API documentation, parameter schemas, code snippets, and the actual API responses returned during annotation.
- **Synthetic:** the generated user instructions, the reasoning/action traces proposed by ChatGPT, the final natural-language answers, and much of the automatic evaluation logic.

**ToolBench generation pipeline:**
1. **API collection:** crawl **16,464 REST APIs** from RapidAPI and store their docs and invocation details.
2. **Instruction generation:** sample APIs and prompt **ChatGPT (`gpt-3.5-turbo-16k`) with function-calling capability** to write tool-use instructions.
3. **Scenario coverage:** generate three settings: `G1` / `I1` for single-tool instructions, `G2` / `I2` for intra-category multi-tool instructions, and `G3` / `I3` for intra-collection multi-tool instructions.
4. **Solution-path annotation:** use the teacher model to search for a valid trajectory containing reasoning, API selection, parameter filling, real-time API execution, observation consumption, and final answer generation.
5. **Training export:** retain successful traces and convert them into ChatGPT-like multi-round conversation data for ToolLLaMA SFT.

**DFSDT trajectory synthesis:** the paper's key point is that plain ReACT is too brittle for complex multi-tool instructions. DFSDT expands a decision tree over candidate actions, allows backtracking/retraction, and explores more than one reasoning path before committing. In the appendix, the authors note that they use a preorder-style DFS variant to cut sorting cost; if no retraction is needed, the method effectively degrades to ReACT. Empirically, Table 3 reports higher pass rates than ReACT and cost-matched ReACT@N on all three settings, so the same annotation budget yields more usable training trajectories.

**Filtering and verification:**
- During data creation, the important filter is **trajectory success**: only passed annotations are kept as training data.
- The paper explicitly argues that executing the APIs and observing real outputs is necessary; some prior work only generated tool calls without real responses.
- ToolEval is used as an automatic evaluator after training, and the paper reports substantial agreement with humans: **87.1%** on pass rate and **80.3%** on win rate.
- The API retriever is trained contrastively using relevant APIs as positives and sampled APIs as negatives, which matters once the tool catalog is too large for oracle tool selection.

**Why ToolLLM matters for synthetic tool/function-calling data:** this paper moves the field from Toolformer's local "insert one useful call into text" idea to full synthetic agent trajectories. The data is synthetic in the supervision layer, but grounded by real API specs and real executed observations. That distinction is the durable lesson for modern pipelines: synthetic traces are much more valuable when the environment underneath them is real.

**Practical lessons for modern pipelines:**
- Ground the data in real tool specs and real executions whenever possible.
- Label clearly which components are synthetic and which are environment-truth.
- Use search or branch-and-revise methods during annotation when tasks require multi-step tool planning.
- Keep retrieval in the loop early; manual API selection does not scale once the catalog is large.
- Verify synthetic trajectories with success criteria or evaluator checks before turning them into SFT.

## Connections
- [[toolformer]] is the immediate precursor: both generate tool-use supervision automatically, but Toolformer stays at single-call local annotations while ToolLLM synthesizes full trajectories.
- [[gorilla]] is a nearby contrast point on API retrieval and invocation, but ToolLLM is stronger on multi-tool trajectory synthesis over a large real API catalog.
- [[apigen]] and [[apigen-mt]] continue this line toward stricter verification, cleaner function-calling schemas, and more modern multi-turn agent data.
