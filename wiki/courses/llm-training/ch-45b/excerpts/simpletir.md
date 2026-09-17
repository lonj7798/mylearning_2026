---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: arXiv:2509.02479v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2509.02479
created_at: "2026-09-15"
---

# Excerpt: SimpleTIR — End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning

- **Authors:** Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, Bo An (NTU; TikTok)
- **Year:** 2025 (arXiv v1 2025-09-02; v2 2025-09-03)
- **Source type:** paper
- **Used in:** [[read]] §1, §5, Negative samples, Common mistakes

## Hierarchical MDP (§2.1)
A trajectory is `o = (q, l_0, f_0, …, l_{K−1}, f_{K−1})` where `l_k` is the model's turn-`k` response and `f_k`
the tool feedback. High-level state `S_k = (q, l_0, f_0, …, l_{k−1}, f_{k−1})`; high-level transition
`S_{k+1} = S_k ∘ (l_k, f_k)`; high-level reward `R(o)` is terminal. Low-level state `s_t = S_k ∘ (a_1, …, a_{t−1})`,
action `a_t` is one token, and `R_L = 0`. Discount `γ = γ_H = γ_L = 1`.

## Objective with feedback masking (Eq. 1, Eq. 2)
`J_TIR(θ) = E[ (1/G) Σ_{i=1..G} (1 / Σ_t m_{i,t}) Σ_{t=1..|o_i|} m_{i,t} · L_CLIP(θ, i, t) ]`, where `m_{i,t} = 1`
if the token at position `t` belongs to a model response `l_k` and 0 otherwise, and `L_CLIP` is the PPO clipped
surrogate with `ρ_{i,t}(θ) = π_θ(o_{i,t} | o_{i,<t}) / π_{θ_old}(o_{i,t} | o_{i,<t})`. Advantage
`Â_i = (r_i − mean{r_j}) / F_norm({r_j})`.

## Diagnosis (§3.1-§3.2)
- Tool feedback "can deviate significantly from the LLM's learned data distribution. Conditioned on such
  out-of-distribution (OOD) input, the model's subsequent generations can drift away from pretrained patterns,
  becoming highly stochastic and assigning unnaturally low probabilities to selected tokens" (§3.1).
- Masking feedback tokens "is insufficient. The distributional drift it induces contaminates subsequent model
  generations" (§3.1, Fig. 3: probabilities are high in turn 1 and collapse by turn 4).
- **Proposition 3.1.** `‖∇_{z_t} J_TIR‖_2 = (m_{i,t} / Σ_j m_{i,j}) · ρ_{i,t}(θ) · g_{i,t} · |Â_i| · sqrt(1 − 2P(c) + Σ_{j∈A} P(j)²)`,
  where `z_t` are the pre-softmax logits, `P` is `π_θ(· | o_{i,<t})`, `c` the sampled token and `g_{i,t}` a gate that
  is active when the PPO update is not clipped. For `Â_i < 0` the ratio "is unbounded from above"; when `P(c)` is
  small and the distribution is otherwise sharp, `1 − 2P(c)` approaches 1 and `Σ_j P(j)²` stays large (§3.2).
- Credit assignment: with a terminal reward, "a trajectory that fails in its final turns receives a single
  negative reward for the entire sequence ... This dynamic unfairly penalizes valid multi-turn behavior, causing
  the policy to collapse toward safer, single-turn generations" (§3.2).

## Method (§3.3)
A **void turn** is "an LLM response that contains neither a complete code block nor a final answer"
(partial code, repetitive text, or a premature end-of-sequence token). If any turn of a trajectory is void, the
policy loss for the **entire trajectory** is masked and the trajectory is removed from the batch before the GRPO
update. The filter is "agnostic to the specific RL algorithm used".

## Implementation details (§3.4)
No chat template (base models); tool outputs are prefixed with "Code Execution Result:"; every generated code
block is prepended with a `final_answer` function so a task can end in one turn; generation is stopped after a
complete code block and the true tool output is always appended.

## Setup (§4.1)
VeRL plus the Search-R1 framework; Sandbox Fusion as an asynchronous code interpreter; training data Math3-5
(SimpleRL) and Deepscaler; base models Qwen2.5-7B and Qwen2.5-32B (Zero RL, no SFT). Rollout batch 512, mini
update size 128, max response length 16K with at most 5 code-execution turns, raised to 24K and 10 turns when
average response length plateaus. Evaluation at temperature 1, average@32.

## Results
**Table 1 (average@32).** Qwen2.5-7B base 3.2 (AIME24), 1.1 (AIME25), 51.9 (MATH500); Qwen2.5-7B-TIR prompted
without training 1.7 / 0.6 / 18.0; SimpleTIR-7B 50.5 / 30.9 / 88.4, Olympiad 54.8, AMC23 79.1, HMMT25 29.7.
Qwen2.5-32B base 4.2 / 1.6 / 43.1; SimpleTIR-32B 59.9 / 49.2 / 92.9, Olympiad 63.7, AMC23 91.6, HMMT25 34.6.
ReTool (cold-start SFT on Qwen2.5-Math-32B-Instruct) is higher on AIME24 (67.0) and AIME25 (49.3).

**Table 2 (highest score within 1000 gradient steps).**

| | SimpleTIR-7B | Naive Multi-Turn | Low Prob Filtering | High Ratio Filtering | Stop Generation w/o Filtering |
|---|---|---|---|---|---|
| AIME24 | 50.5 | 20.8 | 23.3 | 26.3 | 26.1 |
| MATH500 | 88.4 | 73.1 | 72.8 | 75.0 | 77.3 |

**Table 3 (reasoning-pattern frequency, Claude-3.7-Sonnet labeling of correct responses).** Progressive reasoning
ReTool 18.9% vs SimpleTIR-32B 46.5%; cross verification 82.4% vs 86.0%; error correction 25.8% vs 38.0%.

**Turn scaling (§4.3, Fig. 5 top).** Response length and MATH500 rise with max turns 1 → 5 → 10; "the AIME24
score does not benefit clearly".

## Verification
- Checked on 2026-09-15 against the cached PDF text of arXiv:2509.02479v2 (§2-§5, Tables 1-3, Figs. 1-5).
- Not reported in the cached text: learning rate, clip bounds, group size `G`, `F_norm` choice, seeds, and the
  Appendix C.2 hyperparameter list.
