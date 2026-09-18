<!-- scope: WebArena — self-hosted reproducible web environment and 812-task benchmark with programmatic task checks
     deps: [[agenttuning]]
     see-also: [[openhands-data]], [[agentinstruct]], [[agenttrek]], [[os-genesis]], [[learn-by-interact]], [[explorer]]
-->

# WebArena: A Realistic Web Environment for Building Autonomous Agents
- **Core Insight:** In a self-hosted environment built from four functional websites with programmatic success checks, the best GPT-4 agent reached an end-to-end task success rate of 14.41% against 78.24% for human annotators (Abstract; §5, Table 2).
- **Guideline:** When a web-agent result must be reproducible, run the agent inside WebArena's Docker-packaged sites with their reset scripts and score it with the task's programmatic checker, because the transition function is deterministic and defined by the site implementations (§2.1, §2.2, App. A.2). This source supplies an environment and a benchmark, not a training trajectory set; for trajectories see [[agenttrek]], [[os-genesis]], [[learn-by-interact]] and [[explorer]].
- **Authors:** Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, et al. (Carnegie Mellon University)
- **Year:** 2023 (arXiv v1 2023-07; v4 2024-04-16; ICLR 2024)
- **URL:** https://arxiv.org/abs/2307.13854
- **Source type:** paper
- **Relevant topics:** web agents, reproducible environments, agent evaluation, functional correctness

## Abstract
The paper builds an environment for language-guided agents from fully functional websites in four common domains:
e-commerce, social forum discussion, collaborative software development, and content management. It is enriched
with tools such as a map and external knowledge bases such as user manuals. On top of it the authors release a
benchmark whose tasks are evaluated for functional correctness rather than surface-form match. Baseline agents on
several language models, including chain-of-thought prompting, are evaluated; the best GPT-4-based agent reaches
14.41% end-to-end success against 78.24% for humans (Abstract).

## Key Contributions
- A self-hostable environment `E = ⟨S, A, O, T⟩` with a deterministic transition function defined by each website's
  implementation (§2.1). Four website categories, chosen by classifying roughly 200 examples from the authors' own
  browser histories: e-commerce (OneStopShop), social forum (a Reddit-style site), collaborative development
  (GitLab), and an online-store content management system; plus three tools (a map, a calculator, a scratchpad) and
  knowledge resources including English Wikipedia and site user manuals (§2.2).
- A benchmark of 241 templates instantiated into 812 intents, averaging 3.3 examples per template (§3.1).
- Evaluation by functional correctness: annotated answers scored with `exact_match`, `must_include` or
  `fuzzy_match`, and state-changing tasks scored by programs inspecting the resulting URL, page content, or site
  database (§3.2, Table 1). Unachievable tasks labeled "N/A" test whether an agent avoids unfounded claims (§3.2).
- Baseline agents and an error analysis, including the finding that GPT-4 with the unachievable hint judged 54.9% of
  feasible tasks impossible (§5.1).

## Key Figures/Tables to Study
- **Figure 3 (§2.3)** the three page-content representations; **Figure 4 (§2.4)** the action table.
- **Table 1 (§3.2):** worked examples of `r_info` scoring functions and of programmatic `r_prog` checks.
- **Table 2 (§5)** success rates by model and prompting strategy, split into achievable and unachievable;
  **Table 3 (§5)** distribution of success rates across templates with at least one successful execution.

## Technical Details

### Observation space (§2.3)
The observation is the page URL, the open tabs, and the page content of the focused tab, rendered as raw HTML DOM,
a screenshot, or an accessibility tree — "a subset of the DOM tree with elements that are relevant and useful for
displaying the contents of a web page", each element carrying its role, text content, and properties. An option
limits content to the viewport. WebArena is stated to be the first web environment supporting multi-tab tasks.

### Action space (§2.4, Fig. 4)
Twelve actions in three groups — element operations `noop`, `click(elem)`, `hover(elem)`, `type(elem, text)`,
`press(key_comb)`, `scroll(dir)`; tab operations `tab_focus(index)`, `new_tab`, `tab_close`; URL navigation
`go_back`, `go_forward`, `goto(URL)`. An element is referenced either by on-screen coordinates `(x, y)` or by a
unique element ID prepended during a DOM or accessibility-tree traversal, which turns element selection into an
n-way classification problem — for example `click [1582]` on an observation containing `[1582] Add to Cart`.

### Tasks and evaluation (§3.1, §3.2)
241 templates, 812 instantiated intents, 3.3 examples per template on average, in three categories: information
seeking (a textual answer is expected), site navigation, and content and configuration operation.
- `r_info(â, a*)`: `exact_match` (identical strings), `must_include` (the reference appears in the prediction), or
  `fuzzy_match`, which uses `gpt-4-0613` to judge semantic equivalence; App. A.8 reports near-perfect agreement.
- `r_prog(s)`: a locator (a database query, a site API call, or a JavaScript element selection) retrieves the
  relevant content from the intermediate states, then `exact_match`/`must_include` are applied to annotated keywords.
- Intents were written by the authors; reference answers annotated twice with a third annotator resolving disagreements; evaluation programs written by three of the authors.

### Results (§5, Table 2; end-to-end success rate, %)
| CoT | UA hint | Model | All | Achievable | Unachievable |
|---|---|---|---|---|---|
| yes | yes | GPT-4 | 11.70 | 8.63 | 77.78 |
| yes | no | GPT-4 | 14.41 | 13.02 | 44.44 |
| yes | yes | GPT-3.5 | 8.75 | 6.44 | 58.33 |
| no | yes | GPT-3.5 | 6.41 | 4.90 | 38.89 |
| yes | no | GPT-3.5 | 6.16 | 6.06 | 8.33 |
| yes | yes | text-bison-001 | 5.05 | 4.00 | 27.78 |
| — | yes | Human | 78.24 | 77.30 | 100.00 |

Chain-of-thought brought a 2.34-point improvement for GPT-3.5 with the hint (8.75 vs 6.41, §5). Human performance
was measured by sampling one task from each of 170 templates and asking five computer-science graduate students to
perform them; average time per task 110 s, information-seeking success 74.68%, other categories 81.32%, and 50% of
the human failures were misinterpreted intents, incomplete answers, or incomplete executions (§3.2).

## Findings relevant to agentic training
- The paper releases an environment and a benchmark; it reports no fine-tuning, no trajectory dataset, and no
  training run (checked: §2–§5, App. A).
- Of the 61 templates with at least one successful execution, GPT-4 reached 100% task success on only four and
  GPT-3.5 on none, so tasks from one template are not interchangeable in difficulty (§5, Table 3).
- With the unachievable hint GPT-4 declared 54.9% of feasible tasks impossible; removing the hint still left it
  recognizing 44.44% of the genuinely unachievable tasks (§5.1, Table 2).

## Connections
- [[agenttrek]], [[os-genesis]], [[learn-by-interact]], [[explorer]] — later sources that do produce web-agent
  training trajectories; WebArena is their held-out environment or their sandbox.
- [[openhands-data]], [[magnetic-one]], [[agentgym-rl]], [[skyrl-agent]], [[agenttuning]], [[agent-flan]] —
  evaluate on WebArena or integrate it as one benchmark among several.
- [[agentic-benchmark-checklist]] — documents grading problems that affect WebArena-derived scores.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2307.13854 (arXiv v4, 16 Apr 2024), full text.
- Corrections to the previous card version:
  - "five fully functional open-source applications (GitLab, Reddit-clone, Shopping, OpenStreetMap, Calendar)" →
    four website categories (e-commerce OneStopShop, a Reddit-style forum, GitLab, an online-store CMS) plus a map,
    a calculator and a scratchpad as tools, and English Wikipedia and user manuals as knowledge resources (§2.2).
    There is no calendar application; the paper does not name Postmill, Magento or OpenStreetMap.
  - "GPT-4 with best scaffold: ~35% WebArena" → 14.41% for the best GPT-4 agent (Abstract; §5, Table 2).
  - "Trajectory length: avg 12 steps, max 30" and "5–25 steps; 10K–50K tokens per trajectory" → no trajectory-length
    statistic is printed in the paper.
  - "`stop [answer]`" in the action space → Fig. 4 lists 12 actions and no `stop`; the unachievable-task instruction
    is a prompt directive (§4).
  - The card described two artifacts (WebArena and VisualWebArena). It now describes WebArena only, the artifact
    named by its slug and title. VisualWebArena (arXiv:2401.13649) is a separate paper with no card in this library.
- Removed as unsupported by the source:
  - The "Synthesis pipeline (community approach)" section: GPT-4/SeeAct rollout collection, success-predicate
    filtering, turn-count budget, and loop detection are not in the paper.
  - Cost figures "~$5–10 per task rollout", "$5–20" per trajectory, "dataset-scale collections run to tens of
    thousands of dollars", and "trajectory datasets built on top = tens of thousands".
  - Open-agent scores "LLaVA-34B-WebArena ~15%", "Claude-3-SeeAct ~25%", "best open agent ~30%, frontier closed
    models ~50% as of 2025": uncited and not in this paper.
  - "Environment drift", "shortcut learning / URL-hacking", and "no auth/captcha/anti-bot" as stated risks: the
    paper discusses CAPTCHAs only as a reason live sites are unsuitable (§2) and makes none of these claims.
- Not reported by the source: per-trajectory cost, trajectory token counts, trained-agent scores, released SFT data.
