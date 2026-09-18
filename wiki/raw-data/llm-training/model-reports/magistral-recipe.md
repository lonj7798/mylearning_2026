<!-- scope: Recipe ledger for the Magistral paper (arXiv:2506.10910v1): Magistral Medium pure-RL run, Magistral Small SFT cold start + RL, data filtering, reward values, evaluation settings; companion to [[magistral]]
     deps: [[magistral]]
     see-also: [[grpo]], [[deepseek-r1-recipe]]
-->

# Magistral — Recipe ledger
Companion to [[magistral]]. Source: arXiv:2506.10910v1 (12 Jun 2025), read on 2026-09-14. "Batch" is a number of generated completions (sequences), not tokens (§3, "Trainer optimization"). The paper prints "8k", "4k", "2k", "16k", "24k", "32k" without further precision; values are copied as printed.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Magistral Medium | not reported | RL | starting checkpoint | Mistral Medium 3 Instruct; no reasoning SFT before RL | §5.2; Table 2 | verified 2026-09-14 | Table 2 compares against DeepSeek-R1-Zero / R1 |
| Magistral Medium, Small | Medium not reported; Small 24B | RL | algorithm | GRPO with: no KL term; loss divided by total tokens of all generations in the group; advantage r_i − μ then minibatch normalization; Clip-Higher; zero-advantage groups removed | §2.1, final objective | verified 2026-09-14 | KL: "compute cost we find unjustified" (§2.1), "primarily hinders training" (§7.4.2), no numbers; normalization: §6.4 Fig. 7 shows no significant difference between minibatch, group, and no normalization |
| Magistral (run not named in §2.1; Small used 0.3, see below) | not reported | RL | ε_high | adjusted between 0.26 and 0.28 during training "to keep the group entropy stable" | §2.1 | verified 2026-09-14 | no ablation reported; §7.4.2 Fig. 12 (3B) compares ε_high with an entropy bonus |
| Magistral Medium, Small | Medium not reported; Small 24B | RL | ε_low | not reported | checked: §2.1, §5.2, §5.3, §6, §7 | not reported | not applicable |
| Magistral Medium, Small | Medium not reported; Small 24B | RL | reward | format pass 0.1 (fail 0, not graded); correct +0.9; length penalty 0 to −0.1 (Eq. 1); language consistency +0.1 | §2.2.1–§2.2.4 | verified 2026-09-14 | proportional code reward tested and rejected: 24B, 250 steps, −2% LiveCodeBench (§7.4.1, Fig. 11) |
| Magistral Medium, Small | Medium not reported; Small 24B | RL | code verifier | C++20 compile timeout 10 s; 20 random tests per problem, same tests within a group; 4 s and 300 MB per test | §2.2.2 | verified 2026-09-14 | no ablation reported |
| Magistral Medium, Small | Medium not reported; Small 24B | RL | language data | 10% of English problems translated to French, Spanish, Italian, German, Chinese, Russian | §2.2.4 | verified 2026-09-14 | no ablation reported |
| Magistral Medium | not reported | RL | max non-penalized length l_max − l_cache | 16k → 24k → 32k (increased twice) | §5.2 item 2 | verified 2026-09-14 | no ablation reported; l_max and l_cache separately not reported |
| Magistral Medium | not reported | RL | batch size n_batch | 8k → 4k → 2k (decreased twice as generation length grew) | §5.2 item 3 | verified 2026-09-14 | reason given: KV-cache memory (§5.2 item 3); §6.3 Fig. 6 (3B, math-only): reward does not depend strongly on batch size when n_batch = n_minibatch |
| Magistral Medium, Small | Medium not reported; Small 24B | RL | n_async / n_batch; minibatches | n_async/n_batch ≤ 2 and n_batch = n_minibatch "during final training and further ablations" | §6.3 | verified 2026-09-14 | §6.3 Fig. 6: 3B from Ministral 3B, n_async 4096; reward degrades with more than 2 minibatches per batch; n_batch ≤ 1024 less stable |
| Magistral RL data pool (per-run subsets not reported) | not applicable | RL | math prompts | 699k initial → 501k after format filtering → 38k after difficulty filtering | §4.1, Table 1 | verified 2026-09-14 | two-stage filter, 16 samples per problem (Mistral Large 2, then a 24B RL grader); Medium's later stages added "more complicated data (which were filtered out in earlier stages)" (§5.2 item 1) |
| Magistral RL data pool (per-run subsets not reported) | not applicable | RL | code problems | 35k (Python or C++ statements) | §4.2 | verified 2026-09-14 | no ablation reported |
| Magistral Medium | not reported | RL | learning rate, optimizer, group size G, samples per prompt, RL steps, sampling temperature | not reported | checked: §2–§8, figure captions, appendix figures 14–16 | not reported | not applicable |
| Magistral Small | 24B | distill-SFT | start model | Mistral Small 3 Instruct (24B) | §5.3 | verified 2026-09-14 | Table 3 compares SFT, RL-only, SFT + RL |
| Magistral Small | 24B | distill-SFT | data | correct traces from Magistral Medium's RL run (early short-CoT steps excluded; generations per problem capped; low pass-rate problems upsampled) + Magistral Medium responses to filtered OpenThoughts prompts and OpenR1 code-subset prompts + 10% general instruction data | §5.3 | verified 2026-09-14 | no ablation reported; dataset size not reported |
| Magistral Small | 24B | distill-SFT | epochs; checkpoint rule | 4 epochs; best checkpoint on AIME'24 | §5.3 | verified 2026-09-14 | no ablation reported |
| Magistral Small | 24B | RL | batch; l_max − l_cache | 2048 sequences; 32k | §5.3 | verified 2026-09-14 | no ablation reported |
| Magistral Small | 24B | RL | sampling temperature | 1.0 | §5.3 | verified 2026-09-14 | stated as "best balance" between low diversity and incoherence; no numbers |
| Magistral Small | 24B | RL | ε_high | 0.3 | §5.3 | verified 2026-09-14 | reason given: cold-started model had "far lower entropy"; no ablation reported |
| Magistral Small | 24B | RL | learning rate, optimizer, G, RL steps | not reported | checked: §5.3, §6, §7 | not reported | not applicable |
| Magistral Medium, Small | Medium not reported; Small 24B | eval-gate | sampling | temperature 0.7; top-p 1.0 for math and GPQA, 0.95 for coding; max tokens 40k for AIME and LiveCodeBench, 32k otherwise | §5.1 | verified 2026-09-14 | not applicable |
| Magistral Medium | not reported | eval-gate | runs averaged | AIME 64 runs (pass@1 / maj@64); LiveCodeBench 16 runs | Table 2 caption | verified 2026-09-14 | not applicable |
| Mistral Medium 3 + OSS traces (experiment, not released) | not reported | distill-SFT, RL | SFT data; RL data | about 1.3M generations from OpenThoughts and OpenR1 code subset (DeepSeek-R1 traces); RL on "our most difficult subset" | §8 | verified 2026-09-14 | Fig. 13: RL adds over 10 points on AIME'25 (text) / "more than 12%" (caption) |

Starting point for a small general-purpose run: when an RL run starts from an SFT cold start with low output entropy, the paper used sampling temperature 1.0 and ε_high 0.3 with a 2048-sequence batch and a 32k non-penalized length for a 24B model (§5.3). The paper does not report the learning rate, group size, or step count, so these must come from another verified source.
