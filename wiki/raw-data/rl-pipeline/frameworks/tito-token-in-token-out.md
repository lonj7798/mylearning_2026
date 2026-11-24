<!-- scope: token-in/token-out (TITO) for multi-turn RL rollouts — why re-tokenizing decoded text corrupts training targets, and how slime (fork/realign), verl (Continuous Token builders) and AReaL (parent-token concatenation) keep sampled token ids and log-probs
     deps: [[verl-rollout]], [[slime]]
     see-also: [[rollout-training-mismatch-tis]], [[search-r1]], [[pipelinerl]], [[rl-design-guide]]
-->

# Token-in/token-out (TITO) and re-tokenization drift in multi-turn rollouts
- **Core Insight:** The training sequence for an RL sample must be the exact token ids the sampler produced, with the sampler's log-probs at the same indices; decoding to text and re-applying the chat template rarely reproduces those ids byte for byte ("TITO round-trips and chat-template re-rendering both perturb the ids of content we've already seen", slime `slime/agent/trajectory.py` L161–163), so all three frameworks read here build multi-turn samples by appending token deltas to the stored generated ids rather than re-tokenizing the final conversation.
- **Guideline:** When a rollout has tool or user turns between model turns, keep the engine's returned token ids and log-probs for model turns (loss_mask 1), tokenize only the new context messages as a delta (loss_mask 0, including template boundary tokens the engine did not sample), and run a template-consistency check per model family, because verl states that nothing at runtime compares the merged stream with the real chat-template render, so a mismatched builder "trains on token sequences the template never produces, without any error" (`verl/utils/tokenizer/continuous_token_check.py` L15–21); [[verl-rollout]] records verl's report that re-tokenizing the final history kept PPO from converging (experience report, no numbers).
- **Authors:** slime contributors (THUDM/Z.ai); verl contributors (ByteDance Seed and community); AReaL contributors (inclusionAI / Ant)
- **Year:** 2026 (code read at slime 79aa41a, 2026-11-06; verl 8357f1e, 2026-11-24; AReaL bcef492, 2026-11-24)
- **URL:** https://github.com/THUDM/slime/tree/79aa41a ; https://github.com/volcengine/verl/tree/8357f1e ; https://github.com/inclusionAI/AReaL/tree/bcef492 (local clones under `wiki/raw-data/rl-pipeline/repos/`)
- **Source type:** released config/code
- **Relevant topics:** multi-turn rollout, chat templates, loss masks, sampler log-probs, agent adapters, tool-call parsing

## Summary
A multi-turn agent rollout alternates model generations with tool or user messages. If the trainer rebuilds the sequence by applying the chat template to the final message list, three things can differ from what was sampled: (1) BPE merges across a turn boundary, (2) template-inserted tokens (e.g., a newline after `<|im_end|>`, an empty `<think></think>`), (3) string post-processing of the response. The sampler log-probs then no longer align with the trained tokens, and the loss is computed on tokens the policy never generated.

## Key Contributions
- slime: drift classifier CLEAN / REALIGN / FORK per turn, and a fork into separate samples when drift cannot be absorbed (`trajectory.py` L159–175, L189–199, L554–575).
- verl: model-specific Continuous Token builders that merge token deltas and report inserted boundary tokens so they get no loss (`continuous_token.py` L38–48, L540–552).
- AReaL: child prompt = parent input tokens + parent output tokens + new-message delta, with a prefix assertion (`areal/experimental/openai/client.py` L838–856; `areal/workflow/multi_turn.py` L101–109).

## Key Figures/Tables to Study
- slime `examples/search-r1/generate_with_search.py` L93–98 (why not to post-process strings when log-probs are kept); slime `examples/coding_agent_rl/README.md` L275–282; verl `continuous_token_check.py` L1–21.

## Technical Details
**slime.** Agent adapters' "contract is message history in, sampled tokens out: they render the chat template, call SGLang with `input_ids` and `return_logprob=True`, and export the returned token ids/logprobs as trainable trajectory segments" (`docs/en/get_started/agent.md` L36). The `_SampleBuilder` compares each new turn's prompt ids with held tokens: CLEAN appends the tail; REALIGN overwrites a short divergence inside the most recent response span with loss_mask 0; FORK closes the sample and opens a new one when `len(turn.output_ids) >= fork_threshold` or the divergence is too early (L167–175, L189–199). Default fork threshold 1024 tokens (L369). Routing-replay snapshots for MoE always fork (L200–209). Forked segments "share one outcome reward" and shared generated prefixes contribute loss once (`examples/coding_agent_rl/README.md` L282–286). The Search-R1 example warns that "Re-tokenizing the postprocessed string may produce different tokens than what the engine generated, leading to misalignment between tokens and their log probabilities" (L96–97). An e2e test asserts `trained["tokens"] == original["tokens"]` (`tests/test_agent_sunabako_codex_e2e.py` L432).

**verl.** "Continuous Token is the only tokenization path" in `ToolAgentLoop` (`tool_agent_loop.py` L220, L380). Assistant turns: `ct_merge_assistant_token` appends `output.token_ids` and log-probs (L275–283). Context turns: `ct_merge_context_msg` tokenizes appended messages as a delta (L380–398). `MergeResult` reports `inserted_token_ids`; "Boundary tokens are not model-generated and therefore must not carry loss or model logprobs" (L38–48). Example boundary hook: Qwen templates render `<|im_end|>\n` but generation may stop at `<|im_end|>`, so the builder inserts the newline (L540–578).

**AReaL.** `_concat_prompt_token_ids_with_parent` preserves "every sampled action, including PAD and repeated stop tokens" of the parent turn (L838–842), appends a synthetic EOS with loss_mask 0 when the parent turn was truncated or aborted (L848–853), and locates new messages by template boundaries rather than EOS counts (L856). The multi-turn workflow asserts that each new prompt extends the held sequence as an exact prefix and assigns loss_mask 0 to the new input and 1 to outputs (`multi_turn.py` L101–109).

## Verification
- Read the cited lines in the local clones on 2026-11-24 (commits above). No framework reports a measured accuracy or convergence effect of drift in the code read; the only effect report is verl's doc cited in [[verl-rollout]] (no numbers).
- slime README L98, L102 names TITO as a feature of downstream frameworks (Dressage, Miles); those were not read.
