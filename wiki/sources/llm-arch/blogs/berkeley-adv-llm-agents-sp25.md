<!-- scope: advanced LLM agents course (reasoning, coding, safety)
     deps: [[ch-05]]
     see-also: [[berkeley-llm-agents-f24]], [[weng-why-we-think]]
-->

# CS294/194-280: Advanced Large Language Model Agents (Spring 2025)

- **Core Insight:** Berkeley's advanced course covers inference-time reasoning, coding agents, and safety.
- **Guideline:** Reference this syllabus for research-level reading lists.

- **URL:** https://rdi.berkeley.edu/adv-llm-agents/sp25
- **Type:** course syllabus
- **Relevant chapters:** inference-time reasoning, post-training, search/planning, code generation, theorem proving, agentic AI safety

## Content

### Course Metadata

- **Institution:** UC Berkeley
- **Term:** Spring 2025
- **Instructors:** Dawn Song (UC Berkeley), Xinyun Chen (Google DeepMind), Kaiyu Yang (Meta FAIR)
- **Teaching Staff:** Alex Pan, Tara Pande, Ashwin Dara, Jason Yan
- **Meeting Time:** Mondays 4-6pm PT (Anthro/Art Building 160)
- **Class Numbers:** CS194-280 (33840), CS294-280 (33841)

### Course Description

Examines frontier topics in LLM agents, building on prior coursework. Focus areas include reasoning capabilities, mathematics applications, code generation, and program verification. Key topics: inference-time reasoning techniques, post-training methods, search and planning, agentic workflows, code generation/verification, mathematical theorem proving, and autoformalization.

### Prerequisites

Students are strongly encouraged to have experience and basic understanding of Machine Learning and Deep Learning from courses like CS182, CS188, or CS189.

### Complete Lecture Schedule

| Date | Topic | Speaker | Affiliation |
|------|-------|---------|-------------|
| 1/27 | Inference-Time Techniques for LLM Reasoning | Xinyun Chen | Google DeepMind |
| 2/3 | Learning to Reason with LLMs | Jason Weston | Meta |
| 2/10 | Reasoning, Memory, and Planning of Language Agents | Yu Su | Ohio State |
| 2/17 | *Presidents' Day -- No Class* | -- | -- |
| 2/24 | Open Training Recipes for Reasoning | Hanna Hajishirzi | U Washington |
| 3/3 | Coding Agents and Vulnerability Detection | Charles Sutton | Google DeepMind |
| 3/10 | Multimodal Autonomous AI Agents | Ruslan Salakhutdinov | CMU / Meta |
| 3/17 | Multimodal Agents: Perception to Action | Caiming Xiong | Salesforce AI Research |
| 3/24 | *Spring Recess -- No Class* | -- | -- |
| 3/31 | AlphaProof: Reinforcement Learning Meets Formal Mathematics | Thomas Hubert | Google DeepMind |
| 4/7 | Autoformalization and Theorem Proving | Kaiyu Yang | Meta FAIR |
| 4/14 | Advanced Theorem Proving Topics | Sean Welleck | CMU |
| 4/21 | Abstraction and Discovery with LLM Agents | Swarat Chaudhuri | UT Austin |
| 4/28 | Safe and Secure Agentic AI | Dawn Song | UC Berkeley |

### Lecture Details and Readings

**Lecture 1 (1/27): Inference-Time Techniques for LLM Reasoning**
- Speaker: Xinyun Chen (Google DeepMind)
- Readings:
  - LLMs as Optimizers
  - Self-Correction Limitations
  - Self-Debug Teaching

**Lecture 2 (2/3): Learning to Reason with LLMs**
- Speaker: Jason Weston (Meta)
- Readings:
  - Direct Preference Optimization (DPO)
  - Iterative Reasoning Preference Optimization
  - Chain-of-Verification

**Lecture 3 (2/10): Reasoning, Memory, and Planning of Language Agents**
- Speaker: Yu Su (Ohio State)
- Readings:
  - Grokked Transformers
  - HippoRAG
  - World Model Internet Planning

**Lecture 4 (2/24): Open Training Recipes for Reasoning**
- Speaker: Hanna Hajishirzi (U Washington)
- Readings:
  - Tulu 3
  - DPO/PPO Comparison
  - OpenScholar

**Lecture 5 (3/3): Coding Agents and Vulnerability Detection**
- Speaker: Charles Sutton (Google DeepMind)
- Readings:
  - LM Agents in Security
  - Big Sleep Vulnerability Detection

**Lecture 6 (3/10): Multimodal Autonomous AI Agents**
- Speaker: Ruslan Salakhutdinov (CMU/Meta)
- Readings:
  - Mind2Web
  - WebArena
  - VisualWebArena
  - Tree Search

**Lecture 7 (3/17): Multimodal Agents: Perception to Action**
- Speaker: Caiming Xiong (Salesforce AI Research)
- Readings:
  - OSWORLD
  - AGUVIS

**Lecture 8 (3/31): AlphaProof -- RL Meets Formal Mathematics**
- Speaker: Thomas Hubert (Google DeepMind)
- Readings:
  - IMO Silver-Medal Performance
  - AlphaZero Chess/Shogi
  - Mathematics Future

**Lecture 9 (4/7): Autoformalization and Theorem Proving**
- Speaker: Kaiyu Yang (Meta FAIR)
- Readings:
  - LeanDojo
  - Autoformalization
  - Euclidean Geometry

**Lecture 10 (4/14): Advanced Theorem Proving Topics**
- Speaker: Sean Welleck (CMU)
- Readings:
  - Draft-Sketch-Prove
  - miniCTX
  - Lean-STaR
  - ImProver

**Lecture 11 (4/21): Abstraction and Discovery with LLM Agents**
- Speaker: Swarat Chaudhuri (UT Austin)
- Readings:
  - In-Context Learning Agents
  - Symbolic Regression

**Lecture 12 (4/28): Safe and Secure Agentic AI**
- Speaker: Dawn Song (UC Berkeley)
- Readings:
  - PrivTrans
  - DataSentinel
  - AgentPoison
  - Progent

### Grading Structure

**1-Unit Students:**
- Participation: 40%
- Reading Summaries: 10%
- Quizzes: 10%
- Article: 40%

**2-Unit Students:**
- Participation: 16%
- Reading Summaries: 4%
- Quizzes: 4%
- Lab: 16%
- Project Proposal: 10%
- Milestone: 10%
- Poster: 10%
- Recording: 10%
- Report: 20%

**3-4 Unit Students:**
- Participation: 8%
- Reading Summaries: 2%
- Quizzes: 2%
- Lab: 8%
- Project Proposal: 10%
- Milestone: 10%
- Poster: 10%
- Recording: 5-10%
- Report: 20%
- Implementation: 25%

### Project Timeline

| Deliverable | Due Date |
|-------------|----------|
| Group Formation | 2/24 |
| Proposal | 2/24 |
| Milestone | 3/31 |
| Lab | 4/28 |
| Poster Presentation | 4/28 - 5/5 |
| Final Recording | 5/16 |
| Final Report | 5/16 |

### Office Hours
- Alex Pan: Mondays 6-7pm via Zoom

## Why This Is Useful

This is the *advanced* follow-up to Berkeley's foundational LLM Agents course. It dives deep into the cutting-edge topics that matter most for LLM architecture: inference-time reasoning (how models think harder at test time), post-training recipes (RLHF, DPO, PPO), and formal verification. The speaker roster reads like a who's-who of LLM research (Jason Weston on reasoning, Hanna Hajishirzi on open training, Ruslan Salakhutdinov on multimodal agents). The readings are an excellent curated bibliography for anyone studying LLM internals. Particularly relevant: Lectures 1-4 on reasoning and training, and Lecture 5 on coding agents.
