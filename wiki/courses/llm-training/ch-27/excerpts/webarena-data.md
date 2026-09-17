---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: "WebArena: A Realistic Web Environment for Building Autonomous Agents (arXiv:2307.13854v4)"
source_url: https://arxiv.org/abs/2307.13854
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten from the primary text; the library card papers/webarena-data.md predates verification and lists a calendar app and success rates not in the paper)"
---

# Excerpt: WebArena — self-hosted websites with functional task checks

**Authors:** Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, et al. (Carnegie Mellon University). arXiv v1 2023-07, v4 2024-04-16; ICLR 2024.

## Environment (§2.2)
- Four categories chosen from about 200 examples of the authors' browsing histories: e-commerce (OneStopShop), social forum (Reddit-style), collaborative development (GitLab), content management (online-store admin CMS).
- Utility tools: a map, a calculator, a scratchpad. Knowledge resources: English Wikipedia and website user manuals.
- Delivered as Docker containers with scripts that reset each site to a deterministic initial state (App. A.2). User profiles with different roles and histories are generated per site (§2.4, App. A.3).

## Observation space (§2.3)
- URL, open tabs, and page content of the focused tab, rendered as raw HTML (DOM), a screenshot, or an accessibility tree; content can be limited to the viewport.

## Action space (§2.4, Fig. 4)
- 12 actions: noop, click(elem), hover(elem), type(elem, text), press(key_comb), scroll(dir), tab_focus(index), new_tab, tab_close, go_back, go_forward, goto(URL).
- Elements are selected by on-screen coordinates or by a unique element ID prepended to each element, which turns element selection into an n-way classification.

## Tasks and evaluation (§3)
- 241 templates and 812 instantiated intents in three categories: information seeking, site navigation, content and configuration.
- Information-seeking answers are scored with `exact_match`, `must_include`, or `fuzzy_match` (an LLM judges equivalence); navigation and configuration tasks are scored by programmatic checks of the resulting URL, page content, or database state.
- Unachievable tasks are included; the expected answer is "N/A".

## Results (§5, Table 2; end-to-end success rate %)
| Setting | Model | SR | Achievable | Unachievable |
|---|---|---|---|---|
| CoT, unachievable hint | GPT-4 | 11.70 | 8.63 | 77.78 |
| CoT, no hint | GPT-4 | 14.41 | 13.02 | 44.44 |
| CoT, hint | GPT-3.5 | 8.75 | 6.44 | 58.33 |
| CoT, hint | text-bison-001 | 5.05 | 4.00 | 27.78 |
| — | Human | 78.24 | 77.30 | 100.00 |

- With the hint, GPT-4 judged 54.9% of feasible tasks impossible (§5.1).

## Use in agent-training papers
- WebArena is a held-out environment in AgentTuning (Table 3) and Agent-FLAN (§4). The paper releases a benchmark and environment, not a training trajectory set.

## Not in the paper
Calendar app, Magento/Postmill/OpenStreetMap naming as the site list, "GPT-4 ~35%", "$5–$20 per trajectory", "tens of thousands of trajectories", `stop [answer]` in the action table.
