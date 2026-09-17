---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: arXiv:2506.15841v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2506.15841
created_at: "2026-09-15"
---

# Excerpt: MEM1 — Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents

**Authors:** Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, Paul Pu Liang (SMART Centre; NUS; MIT; Yonsei). arXiv v1 2025-06; version read v2 2025-07-17. Source type: paper. Code: https://github.com/MIT-MI/MEM1.
**Used in:** [[read]] §5, §8.

## Mechanism (§3.1)
- Four XML tags mark the components: `<IS>` for internal state (reasoning), `<query>` for environment queries, `<answer>` for responses, `<info>` for tool outputs.
- At turn t the agent writes `<IS_t>`, then either `<query_t>` (answered by `<info_t>`) or `<answer_t>`. At turn t+1 it "consolidates the tuple `<IS_t>`, `<query_t>`, `<info_t>` into a new `<IS_(t+1)>` ... After each turn, all tags from the previous turn t are pruned from the context".
- Invariant: "At any given turn, the agent retains at most two `<IS>` elements, two `<query>` elements, and one `<info>` element, ensuring bounded and efficient memory usage." Memory is therefore constant in the number of turns.
- The consolidation is not supervised: "At each turn, we prune the agent's context to retain only the most recent `<IS>`, forcing the agent to perform memory consolidation as part of its reasoning process. Without access to full historical context, the agent must learn to preserve and update relevant knowledge internally in order to reap the reward."
- Training environments are built by composing existing QA datasets into N-objective task sequences; the model used below is trained only on **2-objective** tasks.

## Table 1 — multi-objective multi-hop QA (EM, F1, peak tokens ×10², wall-clock seconds)

| Model | 2-obj EM | 2-obj Peak | 8-obj EM | 8-obj Peak | 16-obj EM | 16-obj Peak | 16-obj Time (s) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-14B-Inst | 0.732 | 15.6 | 1.55 | 44.7 | 0.567 | 38.4 | 29.7 |
| Qwen2.5-7B-Inst | 0.268 | 19.6 | 0.87 | 49.5 | 0.165 | 43.3 | 15.5 |
| Qwen2.5-7B-Inst (A-MEM) | 0.286 | 14.1 | 1.13 | 18.6 | 0.730 | 18.8 | 91.2 |
| Qwen2.5-7B-Inst (truncate) | 0.262 | 8.28 | 0.97 | 11.8 | 0.396 | 13.3 | 22.1 |
| Search-R1 | 0.452 | 13.0 | 0.064 | 24.7 | 0.009 | 20.9 | 4.75 |
| DeepResearcher | 0.536 | 22.0 | 0.73 | 51.8 | 0.071 | 48.9 | 15.8 |
| MEM1-QA (7B, trained on 2-obj) | 0.709 | 6.40 | 1.87 | 8.01 | 1.97 | 10.4 | 8.70 |

- Abstract statement derived from the 16-objective column: "MEM1-7B improves performance by 3.5× while reducing memory usage by 3.7× compared to Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task, and generalizes beyond the training horizon."
- §4.2: "In the 16-objective task, it requires only 27.1% of the peak tokens and 29.3% of the total inference time compared to Qwen2.5-14B-Instruct."
- Scaling shape (Fig. 4): "As the number of objectives increases, the Peak Token Usage of all other methods and models scales nearly linearly. In contrast, MEM1 maintains an almost constant peak token count with only a slight increase." Red entries in Table 1 mark collapsed baselines; at 16 objectives the baselines' context stops growing because their behaviour has degraded.

## Other results
- WebShop (§4.3): MEM1 beats Agent-Flan, Agent-R and AgentLM at similar size, with "a 2.8× improvement in Peak Token Usage, a 1.9× improvement in Dependency, and a 1.5× improvement in Inference Time" against AgentLM.
- "We also observe that SFT significantly underperforms RL, highlighting the necessity for RL-based training" (§4.3; App. Tab. 4).
- Implementation note (App.): each `<IS>` would need two position-id assignments to recover the exact attention pattern; "for training efficiency, we do not duplicate the `<IS>` and assign the position ids for the previous trajectory to each `<IS>`", which the authors state is a deviation from the ideal formulation.

## Not reported by the source
A comparison at matched peak-token budget against a summarization policy, and a general-capability evaluation outside QA and WebShop.

## Verification
- Read on 2026-09-15 against the cached PDF text of arXiv:2506.15841v2 (Abstract, §1, §3.1, §4.2–4.3, Table 1, Figure 4, appendix notes).
