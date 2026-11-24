<!-- scope: Synthesized RL design checklist for ch-09–ch-12 (phase rl-design): task & reference answers, rollout, grader, reward & prompt selection. One rule per line, each cited and labeled measured (an ablation exists) or practice (recommended without an ablation).
     deps: [[big-math]], [[deepmath-103k]], [[scalerl]], [[dapo]], [[verifier-pitfalls]], [[master-rm-one-token]]
     see-also: [[diversity-axes]], [[polaris]], [[skywork-or1]], [[pipelinerl]], [[brorl]], [[tito-token-in-token-out]], [[math-verify]], [[general-reasoner-verifier]], [[deepseek-grm]], [[rlcf-checklists]], [[reasoning-gym]], [[swe-smith]]
-->

# RL Design Guide: task, rollout, grader, reward

Labels: **measured** = at least one ablation or controlled comparison supports the rule; **practice** = a lab or framework does it without an ablation. "Single study" applies to almost every measured rule: one model family, few seeds.

## 1. Task & reference answers (ch-09)
1. Keep only items with a uniquely verifiable, open-ended, closed-form answer; drop or rewrite multiple-choice, yes/no, multi-part and proof items. — practice — [[big-math]] (117,018 MC matches in a 643K pool, Table 2)
2. Write reference answers in one canonical type the grader can match (DAPO turns every answer into an integer). — practice — [[dapo]] §3.5
3. Create a reference answer by agreement: keep a problem only if independent strong-model solutions (and the source solution) reach the same final answer. — practice — [[deepmath-103k]] Stage 4
4. Keep an item only if some reference-policy rollout reproduces its reference answer (solvability filter); this removes wrong labels and unmatchable formats. — practice — [[big-math]] §2.2.2
5. Decontaminate web-mined pools with embedding retrieval plus an LLM paraphrase judge, not only n-grams; raw pools contained 90% of AIME24 and AMC23 as near-duplicates. — measured (contamination rate, not RL effect) — [[deepmath-103k]] §3, [[numina-math]]
6. Record per-prompt pass rate from N rollouts (e.g., 64 for Big-Math, 8 for POLARIS) at build time, so difficulty windows can be selected later. — practice — [[big-math]] §3.1, [[polaris]]
7. Where a program can generate the task, use a generator with a difficulty parameter and a curriculum (advance at 70% over 20 steps); curriculum beat fixed difficulty at every reported level. — measured — [[reasoning-gym]] Table 5
8. For SWE tasks, synthesize bugs inside one environment per repository and keep only fail-to-pass bugs; prefer PR-mirror, procedural or LM-rewrite bugs over LM-modify. — measured (SFT) — [[swe-smith]] Table 4, [[swe-rebench]], [[r2e-gym]]
9. For agentic tasks, define success as final environment state matching a gold state, and generate tasks as walks on a tool-dependency graph. — practice — [[agentscaler]] §2.2, §3.1, [[tau2-bench]]
10. Compose multi-step tasks from atomic subtasks with init / solution / assertion functions so the number of actions is a controlled difficulty variable. — practice — [[tau2-bench]] §3.2
11. Cover the diversity axes (domain, tool-call structure, horizon, environments, users, constraints, formats); see [[diversity-axes]]. — mixed — [[diversity-axes]]

## 2. Rollout (ch-10)
1. Select the rollout temperature per model by sweeping it; take the highest value before accuracy drops (Qwen3-4B: 1.4; R1-Distill-7B: 0.7), and raise it between stages. — measured — [[polaris]]
2. From distilled reasoning models, use T = 1.0 rather than 0.6; at 0.6 entropy collapsed within about 100 steps and learning stalled. — measured — [[skywork-or1]] Exp. 4
3. Treat faster entropy collapse as a warning; more SGD steps per rollout accelerated collapse and lowered test scores. — measured — [[skywork-or1]] §4, [[entropy-mechanism-llm-rl]]
4. Compare rollout settings by the fitted asymptote of pass rate vs compute, not early reward; small batches and short generations win early and lose late. — measured — [[scalerl]] Figs. 9–10
5. Larger prompt batches (2,048 prompts) and longer generation (32K vs 14K) raised the asymptote. — measured — [[scalerl]] §5
6. At fixed total batch, 8–32 generations per prompt gave near-identical curves; when a long run plateaus, N = 512 per prompt beat more N = 16 steps at equal compute (+1.0 math average) and doubled throughput. — measured — [[scalerl]] App. A.13, [[brorl]] Tables 2–3
7. Use asynchronous generation with bounded staleness (PipelineRL, max off-policyness k = 8); it raised efficiency, not the asymptote. — measured — [[scalerl]] Fig. 4, [[pipelinerl]]
8. Update weights in flight and keep the stale KV cache; KL was only slightly higher than with recompute, and time-to-reward was about 2× faster than G = 32 conventional RL. — measured — [[pipelinerl]] Figs. 5, 7
9. Record sampler log-probs and correct the sampler/learner mismatch with a truncated importance ratio; compute LM-head logits in FP32 (asymptote 0.52 → 0.61). — measured — [[rollout-training-mismatch-tis]], [[scalerl]] Fig. 5b
10. Penalize length linearly inside a buffer before the hard limit (soft overlong punishment) and mask or shape truncated samples rather than giving them the full failure reward. — measured — [[dapo]] Table 1 (+6 overlong filtering, +3 soft punishment)
11. Do not mask the advantage of truncated responses in long-CoT RL from distilled models; masking lowered overall accuracy. — measured — [[skywork-or1]] Exp. 3 (conflicts with rule 10's overlong filtering; setups differ)
12. Start at the maximum decoding length when affordable (Qwen3-4B at 24K dropped irreversibly; 40K improved), or stage 8K → 16K to save time at equal accuracy for a 7B distilled model. — measured — [[polaris]], [[skywork-or1]] Exp. 2
13. In multi-turn rollouts, train on the engine's token ids and log-probs; tokenize tool and user turns as deltas with loss_mask 0; never re-tokenize decoded text. — practice — [[tito-token-in-token-out]], [[verl-rollout]]
14. Choose colocated vs disaggregated placement by length variance; monitor staleness when async. — practice — [[slime]], [[async-rollout]]

## 3. Grader (ch-11)
1. Measure rule-verifier recall on your own policy's outputs; recall averaged 86% and fell to 0.78–0.88 on Skywork-OR1 responses, and to 47% on general science. — measured — [[verifier-pitfalls]] Table 5
2. Use a hybrid grader: rules first, a model verifier only on rule-rejected responses (+2.3 average on Qwen2.5-7B GRPO). — measured — [[verifier-pitfalls]] Table 2
3. Prefer a discriminative verifier in the RL loop; a generative verifier was hacked after about 450 iterations by single-symbol and gibberish outputs. — measured — [[verifier-pitfalls]] §5–6
4. Pass the gold answer first to Math-Verify and use its stricter reward settings; do not mix string extraction with LaTeX/Expr extraction. — practice — [[math-verify]]
5. Before an RL run, probe every LLM judge with master-key responses (":", "Thought process:", "Solution"); Qwen2.5-72B averaged 66.8% FPR. — measured — [[master-rm-one-token]] Table 1
6. Train judges with truncated reasoning-opener negatives; Master-RM-7B FPR 0.1% average with κ = 0.91 vs GPT-4o. CoT prompting of the judge did not fix FPR. — measured — [[master-rm-one-token]] Tables 1–3
7. Outside math, use a small generative verifier that sees question, reference and response; rule matching agreed with Gemini 22.2% vs 78.7% for the verifier, and RL with it scored higher. — measured — [[general-reasoner-verifier]] §5.3, Table 5
8. For open-ended tasks, give the judge a per-prompt rubric written with the reference answer (35.9 vs 32.0 without reference vs 23.9 Likert on HealthBench-1k). — measured — [[rubrics-as-rewards]] Table 1
9. Generate per-instruction checklists from candidate responses (not the instruction alone) and score items with sample-averaged judge calls. — measured — [[rlcf-checklists]] Table 6
10. If the grading budget allows k judge calls, sample a generative RM with shuffled order and vote with a meta-RM filter (+4.9 at k = 32). — measured (RM benchmarks, not policy RL) — [[deepseek-grm]] Table 3
11. For pairwise judging, swap answer order and count a win only when both orders agree; add a reference answer for math. — measured — [[judge-llm-bias]] Tables 2, 4
12. Attach to each multi-turn item a binary rubric question answerable from the final response (judge-human agreement 93.95% vs 37.33%). — measured — [[multichallenge]] Table 4
13. Budget grader latency: 25 judge samples per item cost 92 h vs 40 h for 5 on one 8×H100 node for WildChat. — measured — [[rlcf-checklists]] Fig. 4

## 4. Reward & prompt selection (ch-12)
1. Drop groups whose rollouts all get the same reward and refill the batch (dynamic sampling, +8 AIME24), or substitute mixed groups to avoid extra rollouts. — measured — [[dapo]] Table 1, [[polaris]], [[slime]]
2. Remove prompts the policy already solves: pass ≥ 0.9 dropped from later epochs (No-Positive-Resampling) raised the asymptote. — measured — [[scalerl]] Fig. 6b, [[polaris]], [[skywork-or1]]
3. Keep the difficulty window wide (mirrored J); keeping only pass ≤ 4/8 hurt as much as no filtering. — measured — [[polaris]]
4. Use a binary correctness reward plus small, bounded shaping terms (format 0.2, extraction failure −0.5, length-difference penalty ≤ 0.5). — practice — [[reasoning-gym]], [[general-reasoner-verifier]]
5. Add a preference or quality signal when constraint rewards alone are used; IF-only rewards lowered AlpacaEval 2 from 33.5 to 21.3. — measured — [[ifbench]] §5
6. When multiple environments share one policy, normalize advantages per task (65.0 vs 59.4 without). — measured — [[agentrl]] Table 6
7. Validate any RLVR change on more than one model family with random and format-reward baselines; spurious rewards gave +21.4 on Qwen2.5-Math-7B. — measured — [[spurious-rewards-rlvr]]
8. Against a learned RM, track a held-out gold score vs KL and stop near its peak. — measured — [[reward-model-overoptimization]]
9. Track a held-out IF metric next to reasoning accuracy; longer RL rollouts raised math and lowered constraint compliance. — measured — [[scaling-reasoning-losing-control-mathif]] Table 5
10. Watch training-reward vs oracle-reward divergence and inspect short or degenerate outputs; both judge collapses were found this way. — measured — [[verifier-pitfalls]] §5, [[master-rm-one-token]] §1
11. When an environment may be hackable, patch the scorer rather than penalizing detected hacks, and consider inoculation prompting (misalignment −75–90%). — practice (patching) / measured (inoculation) — [[metr-frontier-reward-hacking]], [[natural-emergent-misalignment-reward-hacking]]
12. Do not expect a better proxy to remove hacking; limit optimization instead. — practice (theorem) — [[reward-hacking-taxonomy]], [[lilianweng-reward-hacking]]
