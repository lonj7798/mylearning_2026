<!-- scope: DeepSeekMath (arXiv:2402.03300) — Common Crawl math corpus, 7B continued pre-training, math SFT, and the GRPO algorithm (critic-free PPO variant)
     deps: [[ppo]]
     see-also: [[grpo-recipe]], [[deepseekmath]], [[john-schulman-kl-tricks]], [[math-shepherd]], [[rlvr-beyond-base-model]], [[dr-grpo]], [[deepseek-r1]]
-->

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **Core Insight:** DeepSeekMath-RL 7B reaches 51.7% top-1 on MATH without tools after 500B tokens of continued math training from a DeepSeek-Coder-Base-v1.5 7B checkpoint, 776K-example SFT, and GRPO, a PPO variant that replaces the value model with the mean reward of G sampled outputs per question (Abstract; §2.3; §3.1; Table 5).
- **Guideline:** When a PPO value model of policy size is too costly, use GRPO's group baseline, because DeepSeekMath-RL 7B (G = 64, about 144K questions) improved over its SFT start on every reported math benchmark (§4.2, Table 5); otherwise, when a value model is affordable, this paper gives no accuracy evidence for choosing GRPO over PPO, because it reports no direct comparison.
- **Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, et al. (DeepSeek-AI, Tsinghua University, Peking University)
- **Year:** 2024 (arXiv v1 2024-02; v3 2024-04-27)
- **URL:** https://arxiv.org/abs/2402.03300
- **Source type:** paper
- **Relevant topics:** math corpus mining, continued pre-training, code-to-math transfer, math SFT, GRPO, KL-in-loss, outcome vs process supervision, iterative RL, Maj@K vs Pass@K

## Abstract
The paper introduces DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B on 120B math tokens from Common Crawl together with natural-language and code data. It scores 51.7% on MATH without external tools or voting, and self-consistency over 64 samples reaches 60.9%. The authors attribute the result to (1) a web-data selection pipeline and (2) Group Relative Policy Optimization (GRPO), a PPO variant that improves math reasoning and reduces PPO's memory use.

## Key Contributions
- DeepSeekMath Corpus: 35.5M web pages, 120B tokens, collected in four iterations of fastText recall seeded with OpenWebMath (§2.1).
- DeepSeekMath-Base 7B: 64.2% GSM8K and 36.2% MATH few-shot CoT, above Minerva 540B (58.8%, 33.6%) (Table 2).
- Pre-training lessons: code training before math training improves math reasoning; arXiv-only corpora give no notable math gains in the tested settings (§5.1).
- GRPO: no value model; group-normalized rewards as advantages; KL to the reference added to the loss (§4.1.1-§4.1.2).
- Unified gradient view of SFT, RFT, Online RFT, DPO, PPO, GRPO, with online-vs-offline, outcome-vs-process, and iterative-RL experiments (§5.2.1, Table 10, Figures 5-6).

## Key Figures/Tables to Study
- Figure 2 (corpus pipeline); Table 1 (corpus comparison at 1.3B); Table 4 (general benchmarks of the 7B base).
- Tables 6-7 (code vs general tokens before math, forgetting); Tables 8-9 (arXiv corpora).
- Figure 4 (PPO vs GRPO); Eq. 3 (GRPO objective); Eq. 4 (KL estimator); Algorithm 1 (iterative GRPO); Table 5 (main results).
- Table 10 and Figure 5 (method comparison); Figure 7 (Maj@K vs Pass@K).

## Technical Details
### Corpus construction
- fastText classifier: 500,000 OpenWebMath positives, 500,000 Common Crawl negatives; vector dimension 256, LR 0.1, word n-gram 3, minimum word count 3, 3 epochs (§2.1).
- Common Crawl after URL dedup and near-dedup: 40B HTML pages; pages ranked by classifier score; iteration 1 kept the top 40B tokens after pre-training tests on top 40B/80B/120B/160B (§2.1).
- A domain with over 10% of its pages collected is labeled math-related; annotators mark math URL paths; uncollected pages under those paths join the seed (§2.1).
- Collection stopped after iteration 4 because nearly 98% of its data had already been collected in iteration 3 (§2.1).
- At 1.3B with 150B training tokens, the corpus (120.2B tokens) gives GSM8K 23.8% and MATH 13.6%, vs Proof-Pile-2 (51.9B) 14.3%/11.2%, OpenWebMath (13.6B) 11.5%/8.9%, and no math training 2.9%/3.0% (Table 1).
- All training settings (1.3B ablations, 7B base, SFT, reward model, RL) are in [[grpo-recipe]].

### GRPO objective (§4.1.1, Eq. 3)
```
J_GRPO(θ) = E[q ~ P(Q), {o_i}_{i=1..G} ~ π_θold(O|q)]
  (1/G) Σ_i (1/|o_i|) Σ_t { min[ ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1-ε, 1+ε) Â_{i,t} ] − β D_KL[π_θ || π_ref] }
ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_θold(o_{i,t} | q, o_{i,<t})
D_KL[π_θ || π_ref] = π_ref(o_{i,t}|q,o_{i,<t}) / π_θ(o_{i,t}|q,o_{i,<t}) − log[ π_ref(o_{i,t}|q,o_{i,<t}) / π_θ(o_{i,t}|q,o_{i,<t}) ] − 1   (Eq. 4)
```
- q: question from the question set P(Q); o_i: i-th of G outputs sampled from the old policy π_θold; |o_i|: its length in tokens.
- π_θ: policy being trained; π_ref: reference policy; ε: clipping hyperparameter; β: KL coefficient; Â_{i,t}: advantage of token t in o_i.
- Eq. 4 is cited to Schulman (2020) and is "guaranteed to be positive" (§4.1.1). The paper does not use the label "k3"; that label comes from the cited note ([[john-schulman-kl-tricks]]).
- PPO adds a per-token KL penalty to the reward (Eq. 2). GRPO adds the KL term to the loss instead, "avoiding complicating the calculation of Â" (§4.1.1).
- Stated motivation: the PPO value model is "typically another model of comparable size as the policy model", and a reward given only at the last token "may complicate the training of a value function that is accurate at each token" (§4.1.1).
- Gradient coefficient of GRPO under π_θold = π_θ: Â_{i,t} + β(π_ref/π_θ − 1) (App. A.1.6, Eq. 21).

### Advantage estimation
- Outcome supervision: Â_{i,t} = (r_i − mean(r)) / std(r) for every token of o_i, with r = {r_1, …, r_G} (§4.1.2).
- Process supervision: each step reward is normalized by the mean and std of all step rewards R in the group; Â_{i,t} is the undiscounted sum of normalized rewards of steps whose end index is ≥ t (§4.1.3).
- Iterative RL: the reward model is retrained on policy samples with a replay of 10% historical data, then π_ref is reset to the current policy (§4.1.4, Algorithm 1).

### Results
- DeepSeekMath-Instruct 7B → DeepSeekMath-RL 7B, chain-of-thought: GSM8K 82.9% → 88.2%, MATH 46.8% → 51.7%, MGSM-zh 73.2% → 79.6%, CMATH 84.6% → 88.8% (Table 5).
- Tool-integrated: GSM8K 83.7% → 86.7%, MATH 57.4% → 58.8%, MGSM-zh 72.0% → 78.4%, CMATH 84.3% → 87.6% (Table 5).
- Figure 5 (DeepSeekMath-Instruct 1.3B, GSM8K and MATH curves): Online RFT exceeds RFT in later training; GRPO exceeds Online RFT; GRPO+PS exceeds GRPO+OS. The text gives no numeric values (§5.2.1).
- Figure 6 (7B, two rounds of iterative RL): performance improves, most at the first iteration (§5.2.1).

## Recipe ledger
The full ledger is in [[grpo-recipe]]. DeepSeekMath-RL 7B: policy LR 1e-6, β = 0.04, G = 64, max length 1024, training batch size 1024 (unit not stated), one policy update per exploration stage (§4.2). Clip ε, rollout temperature, and RL step count are not reported.

## Findings relevant to generality, negative feedback
### Generality
- Math continued pre-training at 7B raised general benchmarks: MMLU 49.1% → 54.9% and BBH 55.2% → 59.5% vs the released DeepSeek-Coder-Base-v1.5; vs the pre-decay checkpoint actually used for initialization, 42.9% → 54.9% and 42.9% → 59.5% (Table 4). Result (single study).
- Coding after math training: HumanEval 40.9%, MBPP 52.6%, vs 43.2%/60.4% for the released coder base and 40.2%/52.6% for the initialization checkpoint; the authors describe this as maintained (§2.3, Table 4).
- Code before math (1.3B): 400B code → 150B math gives GSM8K 21.9%, MATH 15.3%, MMLU 36.2%, BBH 35.3%, vs 400B general → 150B math at 19.1%, 14.4%, 33.1%, 32.7% (Tables 6-7). Result (single study).
- Forgetting (1.3B): after code → math, HumanEval falls from 25.0% to 12.2% and MBPP from 40.0% to 17.0%. Mixed one-stage training (400B code + 150B math) keeps 29.3%/39.4% and gives GSM8K+Python 19.7%, but GSM8K without tools is 17.6% vs 20.5% for math-only (Tables 6-7). The authors conjecture that 1.3B lacks capacity for both (§5.1.1, Interpretation).
- arXiv corpora (MathPile, ArXiv-RedPajama) show "no notable improvements or even deterioration" at 1.3B (150B tokens) and 7B (40B tokens); e.g., 7B MATH 12.5% → 11.5%/11.1%, miniF2F-test 21.7% → 16.4%/11.9% (§5.1.2, Tables 8-9). Untested: other math tasks such as informalization, arXiv mixed with other data, larger scale (§5.1.2).
- RL used only GSM8K/MATH chain-of-thought questions; the authors treat all other benchmarks as out-of-domain, and all improved (§4.2, Table 5). Non-math benchmarks after RL are not reported.
- RL raises Maj@K but not Pass@K (7B, temperature 0.7, K ≤ 64); the authors attribute the gain to "boosting the correct response from TopK rather than the enhancement of fundamental capabilities" (§5.2.2, Figure 7, Interpretation). They name SFT-only questions and naive nucleus sampling as a possible cause (§5.2.3).
- Stated limits: weaker than closed models on geometry and theorem proving; similar zero-shot and few-shot performance (§6).

### Negative feedback
- Online RFT has gradient coefficient 1 for correct and 0 for incorrect outputs and "does not penalize incorrect responses"; GRPO's normalized advantage gives below-mean outputs a negative coefficient (negative as gradient) (§5.2.1, Eq. 10, Eq. 21).
- GRPO exceeds Online RFT at 1.3B (Figure 5). GRPO also uses a model reward and Online RFT a rule reward (Table 10), so the share of the gain due to negative coefficients is not isolated (Interpretation).
- Reward noise: PRM800K is cited as containing about 20% incorrect annotations, which motivates noise-robust RL as future work (§5.2.3, footnote 7).

## Connections
- [[grpo-recipe]] — full recipe ledger for this paper.
- [[deepseekmath]] — duplicate slug that redirects here.
- [[ppo]] — Eq. 1-2 baseline; GRPO keeps the clipped ratio and removes the value model and GAE.
- [[john-schulman-kl-tricks]] — the cited source of the Eq. 4 KL estimator.
- [[math-shepherd]] — reward-model training set and process supervision follow Wang et al. (2023b).
- [[prm800k]] — dataset whose label noise is cited in §5.2.3.
- [[rlvr-beyond-base-model]] — later pass@k study of RLVR related to Figure 7.
- [[dr-grpo]] — later analysis of the 1/|o_i| and std normalizations in Eq. 3.
- [[rloo]] — another critic-free baseline estimator; [[deepseek-r1]] — later DeepSeek model trained with GRPO.
- [[verl-grpo]], [[trl-grpo]] — framework implementations of GRPO.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2402.03300 (arXiv v3, 2024-04-27), body and App. A.1.
- Corrections: "Figure 4 / Algorithm 1: GRPO pseudocode" → Figure 4 is the PPO/GRPO diagram, Algorithm 1 is iterative GRPO; "Section 4.1.2 / Equation 3" → Eq. 3 is in §4.1.1; "Figure 5: per-question reward variance" → Figure 5 compares RFT, Online RFT, GRPO+OS, GRPO+PS at 1.3B; "Table 5: Outcome RM vs Process RM" → Table 5 is the main results table (OS vs PS is Figure 5); "discounted sum" for process advantage → undiscounted sum (§4.1.3); "beats SFT by 4–6 points on MATH" → 46.8% → 51.7% CoT, 57.4% → 58.8% tool-integrated (Table 5); "Batch size (prompts) 1024" → 1024 with unit not stated (§4.2); title "DeepSeekMath: … Models (GRPO)" → exact published title without "(GRPO)"; author list shortened to the first six names plus "et al."; the old card omitted the pre-training lessons (§5.1) and the Table 4 general-benchmark results, now under Findings.
- Removed as unsupported: "removing the critic halves memory"; "low-variance baseline"; "G = 8–64"; "to keep KL positivity" as the reason for KL in loss; "minimal extra compute"; k1/k2/k3 variance and bias comparison; clip ε = 0.2; sampling T = 1.0; "π_ref: SFT model, frozen"; "became the core of DeepSeek-R1" (not in this paper).
- Not reported by the source: clip ε, rollout temperature/top-p, RL steps and compute, whether the released RL model used outcome or process rewards, SFT optimizer and loss masking.
