<!-- scope: LLM agent architecture (planning, memory, tool use)
     deps: [[ch-04]]
     see-also: [[berkeley-llm-agents-f24]], [[berkeley-adv-llm-agents-sp25]]
-->

# LLM Powered Autonomous Agents

- **Core Insight:** Agent architecture = planning + memory + tool use, each independently designable.
- **Guideline:** When building agents, treat each pillar as a separate design choice.

- **Author:** Lilian Weng
- **URL:** https://lilianweng.github.io/posts/2023-06-23-agent/
- **Relevant chapters:** Agent architecture, planning, memory systems, tool use

## Summary
A comprehensive survey of LLM-powered autonomous agent systems, covering the three fundamental pillars: planning (task decomposition, self-reflection), memory (short-term context, long-term vector stores, retrieval algorithms), and tool use (MRKL, Toolformer, HuggingGPT). Includes case studies of ChemCrow, generative agents, AutoGPT, and GPT-Engineer.

## Key Content

### Agent System Architecture

LLM-powered autonomous agents use large language models as the central decision-making component with three pillars:

1. **Planning:** Decompose complex objectives into manageable subgoals; self-reflect to learn from past actions
2. **Memory:** Short-term (in-context learning within transformer context) + long-term (external vector stores for retrieval)
3. **Tool Use:** External APIs and specialized tools extending capabilities beyond model weights

### Planning: Task Decomposition

**Chain of Thought (CoT):** Prompting models to "think step by step" breaks complex problems into sequential reasoning steps. Leverages additional computational resources during inference.

**Tree of Thoughts:** Expands CoT by exploring multiple reasoning branches at each decision point. Generates several possible thoughts per step, constructing a tree evaluated via classifiers or majority voting.

**LLM+P Framework:** Delegates planning to external classical planners using PDDL (Planning Domain Definition Language) as intermediary.

### Planning: Self-Reflection

**ReAct:** Integrates reasoning and action with the pattern: "Thought: ... Action: ... Observation: ..." repeating cyclically.

**Reflexion:** Dynamic memory and self-reflection using RL principles. Computes heuristic functions to identify inefficient trajectories or hallucinations (repeated identical actions producing unchanged observations).

**Chain of Hindsight (CoH):** Fine-tunes models using sequences of ranked outputs with human feedback annotations.

**Algorithm Distillation (AD):** Concatenates learning histories from multiple RL episodes and feeds them into models, enabling in-context learning of RL patterns.

### Memory Architecture

Maps to human memory models:
- **Sensory Memory** -> Embedding representations of raw inputs
- **Short-Term Memory** -> In-context learning (transformer context window)
- **Long-Term Memory** -> External vector stores with fast retrieval

**Key retrieval algorithms:**
- **LSH:** Hash functions mapping similar items to identical buckets
- **ANNOY:** Random projection trees with binary tree traversal
- **HNSW:** Hierarchical graph layers with small-world navigation shortcuts
- **FAISS:** Vector quantization with coarse-then-refined cluster search
- **ScaNN:** Anisotropic vector quantization optimizing inner product preservation

### Tool Use

**MRKL:** Neuro-symbolic architecture where LLMs function as routers directing queries to specialized expert modules. Key finding: knowing when and how to use tools proves crucial.

**Toolformer:** Fine-tunes language models to learn external tool API usage, expanding training data based on whether API annotations improve output quality.

**HuggingGPT:** Four-stage workflow:
1. Task planning (parse requests into structured task lists with dependencies)
2. Model selection (route tasks to appropriate specialized models)
3. Task execution (run selected models, log results)
4. Response generation (synthesize results into summaries)

### Case Studies

**ChemCrow:** Augments LLMs with 13 expert-designed chemistry tools. Critical finding: LLM-based evaluations gave similar scores to GPT-4 and ChemCrow, but human expert assessments showed ChemCrow substantially outperforming — revealing that LLMs lack domain expertise for proper evaluation.

**Generative Agents (Park et al.):** 25 virtual characters with:
- **Memory stream:** Long-term database of natural language observations
- **Retrieval model:** Surfaces context based on recency, importance, and relevance
- **Reflection:** Synthesizes observations into higher-level inferences
- **Planning:** Translates reflections into actions

Emergent behaviors: information diffusion, relationship memory, spontaneous social event coordination.

**AutoGPT:** Autonomous agents with ~4000-word short-term memory. Accesses Google Search, website browsing, file operations, code execution, and agent delegation.

### Challenges and Limitations

- **Finite Context Length:** Restricted capacity limits historical information; vector stores provide larger pools but inferior representation vs full attention
- **Long-Term Planning Difficulties:** Agents struggle with planning across lengthy histories and exploring solution spaces
- **Natural Language Interface Reliability:** Formatting errors and occasional "rebellious behavior" (refusing instructions) necessitate extensive output parsing

## Notable Insights
- The ChemCrow finding is sobering: LLMs evaluating LLMs can miss domain-specific errors that human experts catch, suggesting LLM-as-judge has fundamental limitations in specialized domains.
- The generative agents study demonstrates that complex social behaviors can emerge from relatively simple agent architectures with proper memory and reflection mechanisms.
- The mapping of agent memory to human memory systems (sensory/short-term/long-term) provides a useful framework for thinking about agent architecture design.
- Tool use is not just about capability extension — MRKL's finding that LLMs struggle with reliably extracting correct tool arguments highlights a fundamental reliability challenge.
