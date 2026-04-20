<!-- scope: foundational LLM agents course (reasoning, frameworks, applications)
     deps: [[ch-04]]
     see-also: [[berkeley-adv-llm-agents-sp25]], [[weng-llm-agents]]
-->

# CS294/194-196: Large Language Model Agents (Fall 2024)

- **Core Insight:** Features lectures from top researchers (Denny Zhou, Shunyu Yao, Ben Mann from Anthropic).
- **Guideline:** Watch these lectures for research perspectives not available in papers.

- **URL:** https://rdi.berkeley.edu/llm-agents/f24
- **Type:** course syllabus
- **Relevant chapters:** LLM foundations, reasoning, planning, tool use, RAG, code generation, multi-agent systems, evaluation, safety

## Content

### Course Metadata

- **Institution:** UC Berkeley
- **Term:** Fall 2024
- **Primary Instructor:** Dawn Song, Professor at UC Berkeley
- **Guest Co-instructor:** Xinyun Chen, Research Scientist at Google DeepMind
- **GSIs:** Alex Pan, Sehoon Kim
- **Readers:** Tara Pande, Ashwin Dara
- **Meeting Time:** Mondays 3-5pm PT at Latimer 120
- **Class Numbers:** CS194-196 (32306), CS294-196 (32304)

### Course Description

Explores how LLMs have been developed as agents to interact with the world and handle various tasks. Topics include LLM foundations, reasoning, planning, tool use, infrastructure, retrieval-augmented generation, code generation, robotics, multimodal agents, evaluation, privacy/safety/ethics, human-agent interaction, and multi-agent collaboration.

### Prerequisites

Experience and basic understanding of Machine Learning and Deep Learning from courses like CS182, CS188, or CS189.

### Complete Lecture Schedule

---

**September 9 -- LLM Reasoning**
- Speaker: Denny Zhou (Google DeepMind)
- Materials: Intro slides, Lecture slides, Recording, Edited Video
- Readings:
  - Chain-of-Thought Reasoning Without Prompting
  - Large Language Models Cannot Self-Correct Reasoning Yet
  - Premise Order Matters in Reasoning with Large Language Models
  - Chain-of-Thought Empowers Transformers to Solve Inherently Serial Problems
  - *(All readings optional this week)*

---

**September 16 -- LLM Agents: Brief History and Overview**
- Speaker: Shunyu Yao (OpenAI)
- Materials: Slides, Recording, Edited Video
- Readings:
  - WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
  - ReAct: Synergizing Reasoning and Acting in Language Models

---

**September 23 -- Agentic AI Frameworks & AutoGen / Building a Multimodal Knowledge Assistant**
- Speakers: Chi Wang (AutoGen-AI), Jerry Liu (LlamaIndex)
- Materials: Chi's Slides, Jerry's Slides, Recording, Edited Video
- Readings:
  - AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
  - StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows

---

**September 30 -- Enterprise Trends for Generative AI and Key Components of Building Successful Agents**
- Speaker: Burak Gokturk (Google)
- Materials: Slides, Recording, Edited Video
- Readings:
  - Google Cloud expands grounding capabilities on Vertex AI
  - The Needle In a Haystack Test: Evaluating the performance of RAG systems
  - The AI detective: The Needle in a Haystack test and how Gemini 1.5 Pro solves it

---

**October 7 -- Compound AI Systems & the DSPy Framework**
- Speaker: Omar Khattab (Databricks)
- Materials: Slides, Recording, Edited Video
- Readings:
  - Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs
  - Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together

---

**October 14 -- Agents for Software Development**
- Speaker: Graham Neubig (Carnegie Mellon University)
- Materials: Slides, Recording, Edited Video
- Readings:
  - SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
  - OpenHands: An Open Platform for AI Software Developers as Generalist Agents

---

**October 21 -- AI Agents for Enterprise Workflows**
- Speaker: Nicolas Chapados (ServiceNow)
- Materials: Slides, Recording, Edited Video
- Readings:
  - WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?
  - WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks
  - TapeAgents: a Holistic Framework for Agent Development and Optimization

---

**October 28 -- Towards a Unified Framework of Neural and Symbolic Decision Making**
- Speaker: Yuandong Tian (Meta AI / FAIR)
- Materials: Slides, Recording, Edited Video
- Readings:
  - Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping
  - Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces
  - Composing Global Optimizers to Reasoning Tasks via Algebraic Objects in Neural Nets
  - SurCo: Learning Linear Surrogates For Combinatorial Nonlinear Optimization Problems

---

**November 4 -- Project GR00T: A Blueprint for Generalist Robotics**
- Speaker: Jim Fan (NVIDIA)
- Materials: Slides, Recording, Edited Video
- Readings:
  - Voyager: An Open-Ended Embodied Agent with Large Language Models
  - Eureka: Human-Level Reward Design via Coding Large Language Models
  - DrEureka: Language Model Guided Sim-To-Real Transfer

---

**November 11 -- No Class (Veterans Day)**

---

**November 18 -- Open-Source and Science in the Era of Foundation Models**
- Speaker: Percy Liang (Stanford University)
- Materials: Slides, Recording, Edited Video
- Readings:
  - Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models

---

**November 25 -- Measuring Agent Capabilities and Anthropic's RSP**
- Speaker: Ben Mann (Anthropic)
- Materials: Slides, Recording, Edited Video
- Readings:
  - Announcing our updated Responsible Scaling Policy
  - Developing a computer use model

---

**December 2 -- Towards Building Safe & Trustworthy AI Agents / A Path for Science-and Evidence-based AI Policy**
- Speaker: Dawn Song (UC Berkeley)
- Materials: Slides, Edited Video
- Readings:
  - A Path for Science-and Evidence-based AI Policy
  - DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models
  - Representation Engineering: A Top-Down Approach to AI Transparency
  - Extracting Training Data from Large Language Models
  - The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks
  - *(All readings optional this week)*

---

### Grading Structure

| Component | 1 Unit | 2 Units | 3-4 Units |
|-----------|--------|---------|-----------|
| Participation | 40% | 16% | 8% |
| Reading Summaries & Q/A | 10% | 4% | 2% |
| Quizzes | 10% | 4% | 2% |
| Article | 40% | -- | -- |
| Lab | -- | 16% | 8% |
| Project Proposal | -- | 10% | 10% |
| Project Milestone 1 | -- | 10% | 10% |
| Project Milestone 2 | -- | 10% | 10% |
| Presentation | -- | 15% | 15% |
| Report | -- | 15% | 15% |
| Implementation | -- | -- | 20% |

### Project Timeline

| Deliverable | Released | Due |
|-------------|----------|-----|
| Project group formation | 9/9 | 9/16 |
| Project proposal | 9/22 | 9/30 |
| Labs | 10/1 | 10/15 |
| Project milestone #1 | 10/19 | 10/25 |
| Project milestone #2 | 10/29 | 11/20 |
| Project final presentation | 11/19 | 12/17 |
| Project final report | 11/19 | 12/17 |

### Office Hours
- Alex Pan: Mondays 5-6pm (Zoom)
- Sehoon Kim: Tuesdays 10-11am (Zoom)

## Why This Is Useful

This is the foundational Berkeley LLM Agents course that the SP25 advanced course builds upon. It provides a comprehensive survey of the entire LLM agent landscape: from reasoning fundamentals (Denny Zhou on chain-of-thought) through agent frameworks (AutoGen, LlamaIndex, DSPy), to applications (software development with SWE-agent, robotics with GR00T, enterprise workflows). The speaker lineup represents the leading edge of industry and academia. Key papers like ReAct, SWE-agent, AutoGen, and DSPy are foundational reading for understanding how LLM architecture connects to real-world agent capabilities. The course also covers critical evaluation and safety topics (Percy Liang on open-source evaluation, Ben Mann on Anthropic's RSP).
