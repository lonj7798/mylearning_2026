<!-- scope: ReST-EM — expectation-maximization self-training for LM reasoning
     deps: [[star]]
     see-also: [[v-star]], [[deepseek-r1]], [[self-rewarding-lm]]
-->

# ReST-EM — Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
- **Core Insight:** Self-training on model-generated correct solutions — filtered by a verifier — matches or exceeds training on human-written solutions; the gains saturate at 2-3 expectation-maximization iterations, with overfitting setting in after.
- **Guideline:** For verifiable-reasoning tasks, iterate (1) sample K solutions per problem, (2) keep only correct ones (E-step), (3) SFT on the survivors (M-step). Cap at 2 iterations; add diversity filtering to prevent collapse on narrow solution paths.
- **Authors:** Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J. Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, Abhishek Kumar, Alex Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Elsayed, Hanie Sedghi, Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura Culp, Lechao Xiao, Maxwell L. Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yundi Qian, Yamini Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, Noah Fiedel
- **Year:** 2023 (Google DeepMind / Brain)
- **URL:** https://arxiv.org/abs/2312.06585
- **Relevant topics:** expectation-maximization, self-training, verifier-filtered data, MATH, APPS

## Abstract
Fine-tuning language models (LMs) on human-generated data remains a prevalent practice. However, the performance of such models is often limited by the quantity and diversity of high-quality human data. In this paper, we explore whether we can go beyond human data on tasks where we have access to scalar feedback, for example, on math problems where one can verify correctness. To do so, we investigate a simple self-training method based on expectation-maximization, which we call ReST-EM, where we (1) generate samples from the model and filter them using binary feedback, (2) fine-tune the model on these samples, and (3) repeat this process a few times. Testing with PaLM-2 models on advanced MATH reasoning and APPS coding benchmarks, we find that ReST-EM scales favorably with model size and significantly surpasses fine-tuning only on human data.

## Key Contributions
- Formalizes self-training as **EM on a latent rationale variable**: E-step samples rationales, filters by verifier; M-step fine-tunes on survivors.
- Shows ReST-EM on PaLM-2-L raises MATH test accuracy from 34.1% (human-data SFT) to 50.6% (two iterations of ReST-EM).
- Shows the same trick lifts APPS code-generation from 16.4% → 31.2%.
- Demonstrates ReST-EM **transfers** — training on MATH improves Big-Bench-Hard unrelated tasks.
- Identifies the saturation + overfitting signature: iter-3 regresses unless diversity filtering is added.

## Key Figures/Tables to Study
- **Figure 2 (MATH accuracy vs iteration):** iter-1 +8%, iter-2 +6%, iter-3 flat; the canonical saturation curve.
- **Figure 4 (held-out BBH):** transfer gains even though BBH wasn't in training data.
- **Table 2 (MATH vs APPS):** consistent gains across both math and code.
- **Figure 6 (diversity):** iterations reduce solution-path diversity; the paper proposes top-k per problem cap to mitigate.

## Technical Details
- **Base model:** PaLM-2-L (≈340B active params), also ablated on -S and -XS.
- **E-step:** sample K=32 solutions per problem at T=1.0, top-p=0.95.
- **Verifier:** exact-match on ground-truth final answer (MATH) or unit-test pass (APPS).
- **M-step:** SFT on (problem, correct-solution) pairs; 1 epoch; lr=1e-5; batch 128.
- **Diversity cap:** keep at most 4 distinct correct solutions per problem (prevents memorization of one solution path).
- **Iterations:** 2 for MATH; gains saturate.
- **Compute split:** inference for E-step dominates (~100 H100-hrs per iter at K=32, N=10K problems).

## Connections
- Sibling of [[star]] (Zelikman 2022): STaR does the same EM with a single sample per problem and a "rationalization" backoff when incorrect; ReST-EM removes rationalization and instead scales K.
- Variant formulations: [[v-star]] (V-STaR) adds a value function over partial rationales; also uses the same EM loop.
- Direct precursor to [[rlvr-tulu3]] and to [[deepseek-r1]]'s RL — both replace the M-step SFT with an RL objective on the same verifier filter.
- The saturation at 2 iters is the same signal Self-Rewarding LM hits at 3 — supports the broader "self-training saturates fast without a richer signal source" thesis.
