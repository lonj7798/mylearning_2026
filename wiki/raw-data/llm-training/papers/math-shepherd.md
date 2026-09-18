<!-- scope: automatic step labels for process reward models from Monte Carlo completions; PRM best-of-N verification and step-by-step PPO on GSM8K / MATH
     deps: [[prm800k]], [[training-verifiers-to-solve-math-word-problems]]
     see-also: [[lets-verify]], [[omegaprm]], [[grpo]], [[metamath]], [[ppo]], [[math-shepherd-recipe]]
-->

# Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
- **Core Insight:** A process reward model trained on step labels computed from 8 completions per step (no human labels) raises Mistral-7B from 77.9 to 84.1 on GSM8K and from 28.6 to 33.0 on MATH through step-by-step PPO, compared with 81.8 and 31.3 for PPO with an outcome reward model (Table 2).
- **Guideline:** When math problems have gold final answers but no step labels, label each step by whether sampled completions from it reach the gold answer and train the reward model on those labels, because this PRM beat self-consistency and an outcome reward model for best-of-256 selection with three generators on GSM8K and MATH500 (Table 1). Use a reward model at least as large as the generator, because a 7B verifier on a 70B generator scored below self-consistency (§5.3, Fig. 5).
- **Authors:** Peiyi Wang, Lei Li, Zhihong Shao, R.X. Xu, Damai Dai, Yifei Li, et al. (Deli Chen, Y. Wu, Zhifang Sui); Peking University, DeepSeek-AI, The University of Hong Kong, Tsinghua University, The Ohio State University
- **Year:** 2023 (arXiv v1 2023-12-14; v3 2024-02-19; ACL 2024, pp. 9426–9439)
- **URL:** https://arxiv.org/abs/2312.08935
- **Source type:** paper
- **Relevant topics:** process reward model (PRM), outcome reward model (ORM), automatic step labels, Monte Carlo estimation, best-of-N verification, step-level PPO, label noise

## Abstract
Math-Shepherd is a process reward model for math solutions: it assigns a score to each reasoning step. Its training labels are built automatically instead of by human annotators (Abstract). The quality of a step is defined as its potential to reach the correct final answer. For each step of a sampled solution, a "completer" LLM decodes several continuations, and the step is labelled from how many of them reach the gold answer (§3.3). The PRM is used (1) to rerank candidate solutions and (2) as the reward in step-by-step PPO. Step-by-step PPO with Math-Shepherd raises Mistral-7B from 77.9% to 84.1% on GSM8K and from 28.6% to 33.0% on MATH; verification with Math-Shepherd on top of that model gives 89.1% and 43.5% (Abstract, Tables 2–3).

## Key Contributions
- Automatic process annotation by completion and estimation, with hard and soft labels (§3.3, Fig. 2).
- Verification results for generators from 7B to 70B (Table 1, Figs. 3 and 5) and step-by-step PPO on two 7B models (Table 2), plus their combination (Table 3).
- Label-quality analysis against 160 human-labelled steps, completer size, reward-model size, training-data size, and an out-of-distribution exam (§5).
- Full training settings: [[math-shepherd-recipe]].

## Key Figures/Tables to Study (arXiv v3 numbering; the ACL 2024 version renumbers some)
- Fig. 2: outcome annotation vs process annotation with hard and soft estimates.
- Table 1: best-of-256 verification on GSM8K and MATH500. Table 2: greedy accuracy after RFT, ORM-PPO, step-by-step PPO. Table 3: PPO model plus verification.
- Fig. 4 and Table 4: label accuracy vs number of completions; comparison with NLI and rule-based step labelling.
- Fig. 5: generator size vs verifier size.

## Technical Details
Losses (§3.2), printed without a leading minus sign
- ORM: L_ORM = y_s log r_s + (1 − y_s) log(1 − r_s) (Eq. 1). y_s = 1 if solution s reaches the gold answer, else 0; r_s = sigmoid score for s.
- PRM: L_PRM = Σ_{i=1..K} [y_si log r_si + (1 − y_si) log(1 − r_si)] (Eq. 2). y_si = label of step i; r_si = sigmoid score for step i; K = number of steps. Binary and three-class step labels gave similar results (§3.2).

Step labels (§3.3.2)
- For step s_i, the completer decodes N continuations with final answers A = {a_1, …, a_N}; a* is the gold answer.
- Hard estimate: y^HE = 1 if some a_j = a*, otherwise 0 (Eq. 3). Soft estimate: y^SE = (1/N) Σ_j I(a_j = a*) (Eq. 4), where I(·) is 1 when the condition holds and 0 otherwise.
- Worked example (Fig. 2): N = 3 continuations from step s_1 end in 24, 24, and 20; the gold answer is 24. Then y^SE = 2/3 and y^HE = 1.
- The experiments train the PRM on hard labels, written as two special tokens for "has potential" and "no potential", so a standard language-modelling pipeline can be used (§4).

Verification (§3.4, §4)
- A solution's PRM score is the minimum over its step scores, following Lightman et al. The paper does not compare minimum with mean or product aggregation.
- Self-consistency plus reward model: a = argmax_a Σ_i I(a_i = a) · RM(p, S_i) (Eq. 5), where S_i is the i-th of N candidate solutions for problem p and a_i its answer.
- 256 candidates per test problem; accuracy is the mean over 3 sampling groups; the MATH verification set is MATH500, the test split of Lightman et al. (§4).

Step-by-step PPO (§3.5)
- The reward is given at the end of each reasoning step instead of only at the end of the response. The ACL 2024 version writes this as r_t = r_PRM at step-end token positions and 0 elsewhere (ACL Eq. 7); no separate final-answer reward term is stated.

Results
- Table 1, GSM8K / MATH500, LLaMA2-70B generator (MetaMATH SFT): self-consistency 88.0 / 39.4; ORM 91.8 / 40.4; Math-Shepherd 93.2 / 44.5; self-consistency + Math-Shepherd 92.4 / 45.2. DeepSeek-67B generator: Math-Shepherd 93.3 / 47.0; self-consistency + Math-Shepherd 92.5 / 48.1.
- Table 2, greedy decoding, GSM8K / MATH: LLaMA2-7B 66.6 / 19.2; +RFT 68.5 / 19.9; +ORM-PPO 70.8 / 20.8; +step-by-step PPO 73.2 / 21.6. Mistral-7B 77.9 / 28.6; 79.0 / 29.9; 81.8 / 31.3; 84.1 / 33.0.
- Table 3: after PPO on Mistral-7B, MATH500 accuracy is 42.3 with self-consistency, 41.3 with the ORM alone, and 41.1 with Math-Shepherd alone; the authors attribute this to a reward model trained before PPO supervising a stronger policy (§4.1).
- On MATH, a PRM trained on the automatic data outperforms one trained on PRM800K; the automatic dataset is four times larger (§5.1, Fig. 3).
- With 10k training solutions, the PRM exceeds the ORM by about 4% accuracy (§5.4, Fig. 6a).

## Findings relevant to negative feedback and generality
Negative labels
- A negative label is y^HE = 0 ("no potential"): none of the N completions from that step reaches the gold answer (Eq. 3). It is a classification target for the reward model (Eq. 2). The policy receives it only through the PRM's per-step reward in PPO (§3.5).
- Label noise: on 160 human-annotated GSM8K steps, hard labels from a LLaMA2-70B completer are 86% accurate at N = 4. Accuracy declines for larger N; the authors attribute this to false positives, that is, wrong steps that get label 1 because at least one of the more numerous continuations still reaches the answer (§5.2, Fig. 4a).
- Soft labels move closer to the human label distribution as N grows (Fig. 4b), but verifiers trained on soft or hard labels perform about the same (§5.2).
- Step-label accuracy: Math-Shepherd (LLaMA2-13B completer, N = 4) 85.0%; NLI with LLaMA2-13B 75.6%; NLI with DeBERTa 61.3%; string-rule matching 75.0% (Table 4).
- Larger completers, and completers trained on data that includes the questions, give lower label loss (§5.2, Fig. 4b–c).
- ORM labels from final-answer checks contain false positives (correct answer, flawed reasoning) (§3.2).

Generality
- Out-of-distribution: on the Hungarian national final exam (33 questions, 100 points), with LLemma-34B sampling 256 candidates, scores are 46.0 greedy, 54.0 with the ORM, and 63.0 with Math-Shepherd (§5.5, Fig. 6b).
- 70B reward models improve as the number of candidates grows, while 7B reward models decline (§5.3, Fig. 5).
- All experiments use GSM8K and MATH training problems and MetaMATH-fine-tuned generators; no other domain is trained on (§4).
- Stated limitations: completion cost grows with N; label noise remains, and its effect on PRM quality is undetermined (§6).

## Connections
- [[prm800k]] / [[lets-verify]] — human step labels that Math-Shepherd replaces; source of the minimum aggregation and the MATH500 split; outperformed on MATH by the automatic data (§3.4, §4, §5.1).
- [[training-verifiers-to-solve-math-word-problems]] — ORM objective followed in Eq. 1 (§3.2).
- [[metamath]] — SFT data for all generators and completers (§4).
- [[ppo]] — RL algorithm, applied here with per-step rewards (§3.5).
- [[grpo]] — DeepSeekMath builds its reward-model training set following Math-Shepherd for process-supervision GRPO and lists Math-Shepherd-Mistral-7B (84.1 / 33.0) as a baseline (arXiv:2402.03300v3 §4.1.3, §4.2, Table 5).
- [[omegaprm]] — calls Math-Shepherd's per-step rollouts a brute-force Monte Carlo method, replaces them with binary-search estimation, and uses the Math-Shepherd dataset as a baseline (arXiv:2406.06592v2 §3.2, §4).
- Setlur et al. 2024, "RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold" (arXiv:2406.14532; no card in this library) — cites Math-Shepherd as related per-step credit assignment (§2) and uses Monte Carlo rollouts from prefixes of incorrect responses to estimate per-step advantages, and builds per-step preference pairs for DPO from them (§6.1).
- [[rlvr-tulu3]] — uses rule-based outcome rewards; Math-Shepherd compares learned outcome and learned process reward models only (Table 2).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2312.08935 (v3, 2024-02-19, full PDF); ACL 2024 version https://aclanthology.org/2024.acl-long.510.pdf for Eq. 6–7 and venue.
- Corrections to the previous card version:
  - Title missing "without Human Annotations" → full title restored.
  - "applied to Mistral-7B and DeepSeekMath-7B" and "DeepSeekMath-7B gains of a similar magnitude" → PPO on LLaMA2-7B and Mistral-7B; verification with LLaMA2-70B, LLemma-34B, DeepSeek-67B generators; no DeepSeekMath model in v3 or the ACL version (§4, Tables 1–2).
  - "Rollout LM … DeepSeekMath-7B; K = 8 or 16" → completer LLemma-7B with N = 8 (§4).
  - "89.1 (+PRM verify)" and "43.5 (+PRM verify)" from the base model → self-consistency + Math-Shepherd on the PPO-trained Mistral-7B, 256 samples, GSM8K / MATH500 (Table 3).
  - "soft labels work better than hard" and "Table 5 (soft vs hard) — soft wins on MATH" → experiments use hard labels; soft vs hard verifiers show no substantial difference (§4, §5.2); Table 5 is a case study.
  - "min … shown to outperform prod and mean (Table 4)" → min is adopted from Lightman et al. without comparison; Table 4 compares labelling methods (§3.4, Table 4).
  - "R_total = r_final + λ · Σ PRM(step_t), λ ≈ 0.1–1.0" → reward at each step end only; no λ or final-answer term stated (§3.5; ACL Eq. 7).
  - "only ~1K problems × modest K" → GSM8K and MATH training sets, 15 samples per problem from each of two generators, about 170k (GSM8K) and 270k (MATH) solutions (§4).
  - "N = 256 or 1024 chains" → 256 (§4). "Fig. 4 / Table 2 (accuracy across configurations)" → Fig. 4 is label quality; Table 2 is PPO results.
- Removed as unsupported by the source: step split on "\n" or "Step i:" and score at first token after boundary; filtering of short or template-only steps; "most open reasoning stacks (Qwen-Math, DeepSeekMath, OpenRLHF) use Math-Shepherd-style labeling"; "less susceptible to proxy misalignment than preference RMs".
- Not reported by the source: PPO clip range, samples per prompt, batch sizes, number of RL steps, sampling temperature, completion cost in GPU hours.
