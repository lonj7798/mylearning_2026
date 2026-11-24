---
chapter: ch-23b
course: llm-training
phase: read
---

# Chapter 23b — Conversation State and Dynamic Skills

<!-- scope: state updates, policy scope, counterfactual skill families, and causal supervision
     deps: [[../ch-23a/read]]
     see-also: [[methods]], [[practice]], [[long-tail-papers]]
     sources: [[user-oriented-dialogue-generation]], [[consistentchat]], [[josh-dialogue-self-training]], [[multichallenge]]
-->

## Overview

**Core insight.** A sales conversation is a sequence of decisions under changing evidence and instructions. Learning each skill in isolation does not teach when its rules apply, what survives a switch, or when no action is permitted. A conversational tail can be an uncommon sequence of ordinary events.

**Guideline.** Build examples around the actual decision input: visible history, active skill definition, product evidence, permitted tools, and supported next action. Use task labels for organization and multiple descriptive tags for coverage. Do not require an exhaustive classifier before collecting data.

Read this main lesson → [Methods and paper comparisons](methods.md) → [Practice: datasets and evaluation](practice.md). [한국어](read_kor.md). Worked examples are proposed teaching cases; their dates, policies, and outcomes are not published experiment results.

## Key Concepts

### 1. State is evidence, not a guessed summary

A useful conceptual state has facts, instructions, and action eligibility. Customer facts include the latest requested date, time, budget, and consent. Instructions include the current skill, policy version, scope, and completion condition. Eligibility records whether a tool's preconditions are satisfied. These can be stored explicitly or inferred from visible history; either way, the training target must have support.

A latest-value rule needs a domain definition. A new date may replace an earlier appointment date, but a mention of a friend's date should not. A revocation can make a field unknown rather than restore an old value. Changing a policy can invalidate an action without deleting customer facts. For this reason, a flat bag of keywords is insufficient: provenance, reference resolution, and scope determine what the words mean.

### 2. Worked case: a correction plus a policy switch

Use a fictional callback tool whose required arguments are date, time, and timezone. No product purchase is being made. The customer says, ‘Tuesday, November 10, 2026, please; I am in Los Angeles.’ Later they say, ‘Actually Thursday, November 12, instead.’ They have not specified a time. Under skill `callback_collect/v1`, the next action is to ask for the time, not assume the most common hour.

The customer then says, ‘Three in the afternoon.’ Now all three fields are supported. Under a policy allowing booking after all required details, the valid call is:

```json
{"name":"schedule_callback","arguments":{"date":"2026-11-12","time":"15:00","timezone":"America/Los_Angeles"}}
```

The date's provenance is the correction, the time's provenance is the last user turn, and the timezone's provenance is the location combined with the declared tool convention. In a real system, location-to-timezone conversion must be supported by your rules or a resolver; the model must not silently make ambiguous geographical assumptions.

Before the call, activate `callback_confirm/v2`: ‘Collect the fields, read them back, and obtain explicit confirmation before scheduling.’ Customer facts stay the same. The next action becomes a short confirmation question and **no tool call**. The model that books anyway extracted the facts correctly but violated current policy. The model that asks for the time again obeyed the no-call rule but lost state. Those are different failures.

### 3. Counterfactual family, not isolated templates

Create a family with the same history and only one changed factor. Policy v1 yields the call; policy v2 yields confirmation. Remove the time and both policies require asking for it. Change only the corrected date and only the date argument should change. Rephrase ‘three in the afternoon’ as ‘3 pm’ and preserve the action. Add an unrelated product question answered from evidence and preserve the customer scheduling fields.

This family teaches **invariance** under irrelevant or meaning-preserving changes and **required change** under relevant facts or policy. It also exposes shortcutting: if every v2 example uses a special phrase, the model may follow the phrase instead of reading the policy. Vary instruction wording and names, and include cases where a familiar name has a new explicit definition. Do not create contradictory gold labels by omitting the instruction that explains their difference.

### 4. Which rules expire after a skill switch?

Suppose `product_explain` requests a concise feature explanation and forbids scheduling tools. Switching to `callback_collect` changes the permitted task and tool scope. Supported customer facts can survive, but the product skill's local ‘do not schedule’ rule must not accidentally remain active if its scope was only that stage. Conversely, a global ‘schedule only after confirmation’ rule must survive a local skill switch.

Define instruction scope explicitly in your system and dataset. If global and local rules conflict, supply an authoritative resolution; do not ask annotators to guess. A response target is valid under the active instructions, not merely fluent or successful under an older version. If the program chooses the skill, training the response model to invent another skill switch may be unwanted autonomy.

### 5. Causal supervision at the boundary

For each assistant turn, retain only the history and skill information available at that moment. The target must not depend on the customer confirming in a future turn. If a simulator knows a hidden desired time, do not expose it to the student unless the customer has stated it. A model rewarded for reading hidden simulator state can look good offline and fail in deployment.

Long conversations do not automatically require long-tail behaviors. Many repeated turns can be easy; a three-turn correction can be hard. Organize tags such as correction, skill switch, missing field, and delayed reference independently of length. You can add a new tag when an actual case appears without redefining all skills or forcing every conversation into one category.

## Recipe

Create a canonical semantic state before adding expression variants. Generate a coherent history that supports it, apply the active policy, derive the allowed next action, and validate the target after every rewrite. Keep both action and no-action cases. Preserve family IDs and per-field provenance so wrong behavior can be localized.

## Generalization lens

Separate routing, argument construction, policy scope, and response quality. A correct tool name with an incorrect date is routing success but argument failure. A valid call made too early is a timing/precondition failure. A pleasant confirmation that invents a product fact is still factually invalid.

## Questions

1. Why can all required fields be present while a tool call is forbidden?
2. Which changes should preserve a call, and which should change it?
3. What leakage occurs if the generator can see a future confirmation?

[Methods](methods.md) → [Practice](practice.md) → [23c](../ch-23c/read.md). [Catalog](../../../raw-data/llm-training/long-tail-papers.md).


## Source cards

- [[user-oriented-dialogue-generation]] — [card](../../../raw-data/llm-training/papers/user-oriented-dialogue-generation.md)
- [[consistentchat]] — [card](../../../raw-data/llm-training/papers/consistentchat.md)
- [[josh-dialogue-self-training]] — [card](../../../raw-data/llm-training/papers/josh-dialogue-self-training.md)
- [[multichallenge]] — [card](../../../raw-data/llm-training/papers/multichallenge.md)
