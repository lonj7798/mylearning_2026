---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlhf-instructgpt.md (existing card; its Verification section was absent on 2026-09-15, so ch-30a takes PPO-ptx values from the primary text below)
source_url: https://arxiv.org/abs/2203.02155
created_at: "2026-09-15"
---

# Excerpt: InstructGPT (Training language models to follow instructions with human feedback), PPO-ptx loci

**Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, et al. (OpenAI)
**Version read:** arXiv:2203.02155v1 (4 Mar 2022).
**Status:** quotes checked against the v1 PDF text on 2026-09-15. The full PPO and alignment-tax treatment is in ch-38.

## Alignment tax (§1)
"During RLHF fine-tuning, we observe performance regressions compared to GPT-3 on certain public NLP datasets, notably SQuAD (Rajpurkar et al., 2018), DROP (Dua et al., 2019), HellaSwag (Zellers et al., 2019), and WMT 2015 French to English translation (Bojar et al., 2015). This is an example of an 'alignment tax' since our alignment procedure comes at the cost of lower performance on certain tasks that we may care about."

## Objective (§3.5, Eq. 2)
objective(φ) = E_{(x,y)∼D_{π_φ^RL}} [ r_θ(x, y) − β log( π_φ^RL(y | x) / π^SFT(y | x) ) ] + γ E_{x∼D_pretrain} [ log π_φ^RL(x) ]
"The KL reward coefficient, β, and the pretraining loss coefficient, γ, control the strength of the KL penalty and pretraining gradients respectively. For 'PPO' models, γ is set to 0."

## Settings (App. C.4)
"We then initialize the RL policies from the above supervised fine-tuned models with pretraining mix. These models are also used to compute the KL reward, in the same way as Stiennon et al. (2020), with β = 0.02 (see Equation 2)."
"We use 8 times more pretraining examples than the number of the RL training episodes. The pretraining data is randomly drawn from the dataset used to train the GPT-3 models. For each minibatch, we compute the PPO gradients and pretraining gradients in consecutive steps and accumulate them both into the gradient buffers. We multiply the pretraining gradients by a coefficient, γ = 27.8."

## Ablations (App. E.6)
- γ sweep (Fig. 33): "By setting pretraining loss coefficient to greater or equal 20, the regression on these tasks can be recovered, on the 1.3B model"; "a single value of 27.8 seems to work well across model sizes, from 1.3B to 175B parameter count."
- β sweep with γ = 0 (Fig. 34): "even by increasing the KL reward coefficient to 2.0, which is 100 times of the default value, the regressions still cannot be fixed."
- App. E.7 (Fig. 36, human Likert score vs β with the pretraining mix): "Both 0 and 2 for KL reward coefficient result in poor performance. The optimal value is around 0.01 and 0.02."
- §4.2 / Fig. 29: PPO-ptx "mitigates these performance regressions on all datasets, and even surpasses GPT-3 on HellaSwag. The performance of the PPO-ptx model still lags behind GPT-3 on DROP, SQuADv2, and translation."
- §5.4: the pretraining mix "does not completely mitigate performance regressions, and may make certain undesirable behaviors more likely for some tasks (if these behaviors are present in the pretraining data)."

## How ch-30a uses it
§5.6 (pretraining-loss mixing preview), Recipe row.
