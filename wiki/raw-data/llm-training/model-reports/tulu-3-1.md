<!-- scope: Tülu 3.1 — Allen AI refresh of Tülu 3 recipe on updated bases
     deps: [[tulu-3]]
     see-also: [[olmo-3]]
-->

# Tülu 3.1
- **Core Insight:** Tülu 3's four-stage recipe (prompt curation → SFT → off-and-on-policy DPO → RLVR) generalizes across base models — Tülu 3.1 is the propagation of the same recipe onto Llama 3.1 and OLMo 2 bases, not a new algorithm.
- **Guideline:** Treat Tülu's recipe as a base-agnostic alignment pipeline; when a new base ships, re-run the stack rather than redesign it.

- **Authors / Lab:** Allen Institute for AI (Ai2)
- **Year:** 2024-11-22 update (refresh of Tülu 3 Nov 21)
- **URL:** https://allenai.org/blog/tulu-3-technical — https://arxiv.org/abs/2411.15124
- **Relevant topics:** open post-training stack, SFT + DPO + RLVR, multi-base propagation (Llama 3.1 + OLMo 2)

## Abstract
"Tülu 3.1" in practice refers to Ai2's Nov 22, 2024 refresh of the Tülu 3 post-training stack applied to both Llama 3.1 and OLMo 2 base models. The pipeline — prompt curation + SFT + preference tuning (DPO) combining off- and on-policy data + RLVR with verifiable rewards — is unchanged; the refresh is a multi-base release, not a new algorithm. The **DR Tulu** (2025) release is a separate follow-up for long-form deep-research training. For new algorithmic Tülu-family work, see **OLMo 3** (2025), which extends the recipe with thinking / RL-Zero paths and Dolci data.

## Key Contributions
- Re-running the Tülu 3 recipe on updated bases: Llama-3.1-Tulu-3-8B, Llama-3.1-Tulu-3-70B, Llama-3.1-Tulu-3-405B.
- Matching/exceeding Llama 3.1-Instruct, Qwen 2.5-Instruct, Mistral-Instruct, Nemotron at the same base.
- **DR Tulu** (2025): extends Tülu to long-form deep-research workflows with bespoke RL environments.
- OLMo 3 (see separate entry) eventually carries the Tülu-derived recipe forward with DPO (delta-learning) + RLVR + thinking paths.

## Post-training pipeline (inherited from Tülu 3, unchanged)
- **SFT data:** carefully curated prompts and completions targeting core skills (reasoning, coding, math, IF, safety, multilingual).
- **Preference / RL algorithm:** DPO combining off-policy (pre-compiled) and on-policy (rolled-out) preference data.
- **Reward model:** trained RM for DPO-pair scoring; specific composition documented in the original Tülu 3 paper.
- **RLVR:** verifiable-reward RL for math / code / IF — rule-based verifiers, no learned RM.
- **KL / entropy / LR / batch / clip / group / rollouts / RL-step counts:** documented in the Tülu 3 paper; 3.1 refresh does not publish per-base-model hyperparameter deltas.
- **Self-improvement / iterative:** iterative DPO (offline + online rollouts) is the in-pipeline iterative element; no multi-round outer loop.

## Innovations vs predecessors
- Tülu 3.1 relative to Tülu 3: same recipe, new bases (Llama 3.1, OLMo 2 joined).
- DR Tulu (follow-up) adds long-form deep-research RL environments.
- vs 2024 open post-training norms: Tülu stack remains the most fully-reproducible open recipe — all data, code, RM, evals, scripts public.

## Key Figures/Tables to Study
- Tülu 3 paper Figure 1 (pipeline overview).
- 8B / 70B / 405B benchmark tables vs Llama 3.1-Instruct — the 3.1 refresh evidence.

## Connections
- [[tulu-3]] — primary reference; algorithmic details live here.
- [[olmo-3]] — successor recipe that builds on Tülu's DPO+RLVR foundation with thinking + RL-Zero.
- [[llama-3]] — base model for the 3.1 Tülu refresh.

## Gaps / what the report does NOT disclose
Tülu 3.1 itself is a refresh release without a separate tech report. Not separately disclosed: per-base-model hyperparameter changes, whether DPO β / RLVR LR / rollouts per prompt changed when moving from Llama-3.1 to OLMo 2 base, exact RM changes between the original Tülu 3 release and the refresh. Chapter authors should cite the Tülu 3 paper (arxiv 2411.15124) for algorithmic detail and the 3.1 blog / HF cards for base-model-specific benchmarks.
