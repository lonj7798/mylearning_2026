<!-- scope: master checklist of topics + target sources for llm-training raw library
     deps: [[README]]
     see-also: [[insights]]
-->

# Collection Plan — Topic Checklist

Every bullet below is a target file. Mark `[x]` when the file lands in the right subdirectory with the required structure (see `[[README]]`). Gaps are filled in a second sweep after initial collection.

Legend: `P` = paper, `B` = blog, `R` = model report, `F` = framework code, `L` = lab summary. Files live in the subdir matching their type.

---

## 0. Classics (pre-2020 training fundamentals)

Why here: the learner explicitly asked to revisit old-school training technique before modern RL. Keep this bucket small but sharp — only techniques still used in 2025 post-training.

- [x] **P** `classics/gradient-clipping.md` — Pascanu 2013 "On the difficulty of training RNNs"; global-norm clipping; why clip value != clip norm
- [x] **P** `classics/adam.md` — Kingma & Ba 2014; first/second moment; AdamW decoupled WD
- [x] **P** `classics/mixed-precision.md` — Micikevicius 2017; loss scaling; fp16/bf16/fp8
- [x] **P** `classics/lr-schedules.md` — warmup, cosine, inverse-sqrt, WSD (warmup-stable-decay) schedules
- [x] **P** `classics/weight-init.md` — Xavier/Glorot, He/Kaiming, muP / μParam for width-transfer
- [x] **P** `classics/dropout.md` — Srivastava 2014; where it still applies post-Transformer
- [x] **P** `classics/label-smoothing.md` — Szegedy 2015; entropy floor effect; why SFT still uses it
- [x] **P** `classics/batch-vs-layer-norm.md` — LayerNorm (Ba 2016), RMSNorm (Zhang & Sennrich 2019)
- [x] **P** `classics/early-stopping-and-checkpointing.md` — validation loss tracking, model averaging, SWA
- [x] **B** `blogs/karpathy-training-neural-net-recipe.md` — Karpathy "A Recipe for Training Neural Networks"
- [x] **B** `blogs/lilianweng-weight-init.md` or equivalent

## 1. Data — Pretraining Curation & Filtering

- [x] **P** `papers/ccnet.md` — Wenzek 2019; language ID + perplexity filtering; foundational filter stack
- [x] **P** `papers/c4.md` — Raffel 2019 (T5 data); heuristic filtering
- [x] **P** `papers/the-pile.md` — Gao 2020; diverse-corpus construction
- [x] **P** `papers/dolma.md` — Soldaini 2024 (Allen AI); open 3T-token pipeline with full filter logs
- [x] **P** `papers/fineweb.md` — Penedo 2024 (HF); FineWeb + FineWeb-Edu; classifier-based quality
- [x] **P** `papers/deduplicating-training-data.md` — Lee 2021; exact + approximate dedup effect on LM quality
- [x] **P** `papers/minhash-lsh.md` — classical reference for MinHash/SimHash dedup
- [x] **P** `papers/doremi.md` — Xie 2023; proxy-model domain reweighting
- [x] **P** `papers/d4.md` — Tirumala 2023; SemDeDup / diversity-aware dedup
- [x] **P** `papers/data-constrained-scaling.md` — Muennighoff 2023; repeat vs new tokens
- [x] **P** `papers/scaling-laws-data-quality.md` — quality-vs-quantity scaling
- [x] **P** `papers/physics-of-lm-3.md` — Allen-Zhu; knowledge capacity / data repetition effects

## 2. Data — SFT Construction

- [x] **P** `papers/self-instruct.md` — Wang 2022 (Yejin Choi group); seed + self-gen pipeline
- [x] **P** `papers/alpaca.md` — Taori 2023; Self-Instruct on text-davinci-003
- [x] **P** `papers/evol-instruct.md` — Xu 2023 (WizardLM); depth + breadth evolution
- [x] **P** `papers/orca.md` — Mukherjee 2023; explanation-traces distillation
- [x] **P** `papers/orca-2.md` — Mitra 2023; cautious reasoning via teacher signal
- [x] **P** `papers/lima.md` — Zhou 2023; 1K high-quality samples beats 52K
- [x] **P** `papers/instag.md` — Lu 2023; tag-based diversity scoring
- [x] **P** `papers/humpback.md` — Li 2023; instruction-backtranslation
- [x] **P** `papers/ultrachat.md` / `papers/ultrafeedback.md` — Cui 2023/2024; scaled dialogue + feedback
- [x] **P** `papers/openhermes.md` — TekniumResearch catalogue
- [x] **P** `papers/wizardlm.md` — companion to Evol-Instruct
- [x] **P** `papers/wildchat.md` — Zhao 2024 (Allen AI); real user logs
- [x] **P** `papers/rejection-sampling-finetuning.md` — Touvron / Llama-2 appendix; RSFT loop
- [x] **P** `papers/star.md` — Zelikman 2022 (Yejin-adjacent); rationale bootstrap
- [x] **P** `papers/quiet-star.md` — Zelikman 2024; internal thought distillation
- [x] **P** `papers/training-verifiers-to-solve-math-word-problems.md` — Cobbe 2021; GSM8K + verifier ranking + sample-and-rank precursor

## 2b. Synthetic Data — Generation, Diversity, Distillation

Why its own section: post-training in 2025 is synthetic-data-dominant. Most frontier recipes (Phi, Tülu 3, Llama 3, Nemotron, Qwen 3, DeepSeek R1-distill) rely on teacher-generated data. The techniques for *generating*, *diversifying*, and *filtering* synthetic data have become a distinct subfield.

### 2b-i Generation methods

- [x] **P** `papers/persona-hub.md` — Ge 2024 (Tencent); 1B personas → persona-conditioned prompting for diverse synthetic data
- [x] **P** `papers/magpie.md` — Xu 2024; extract instruction data from aligned models via prefix-only forward pass (no seed prompts, no API calls)
- [x] **P** `papers/rephrasing-the-web.md` — Maini 2024 (WRAP); rephrase raw web text into Q&A, essay, Wikipedia-style with a small model
- [x] **P** `papers/nemotron-4-synthetic.md` — Nvidia Nemotron-4 340B synthetic pipeline for SFT + RM data (>98% synthetic)
- [x] **P** `papers/reasoning-core.md` — 2026 procedural symbolic suite for synthetic pre-training and post-training data
- [x] **P** `papers/phi-textbooks.md` — Gunasekar 2023 "Textbooks Are All You Need"; synthetic Python-textbook pretraining corpus
- [x] **P** `papers/phi-1-5.md` — Li 2023; pushing synthetic further for general reasoning
- [x] **P** `papers/wizardmath.md` — Luo 2023; Evol-Instruct on math + RLEIF
- [x] **P** `papers/wizardcoder.md` — Luo 2023; Evol-Instruct on code
- [x] **P** `papers/oss-instruct.md` — Wei 2024 (Magicoder); seed from real code snippets for synthetic instruction pairs
- [x] **P** `papers/code-evol-instruct.md` — code-specific evolution operators
- [x] **P** `papers/glan.md` — Li 2024 (Microsoft); Generalized Instruction Tuning via taxonomy-driven synthesis
- [x] **P** `papers/airoboros.md` — Jon Durbin; pipeline-driven synthesis with role-specific generators
- [x] **P** `papers/ultrachat-construction.md` — UltraChat generation protocol specifically
- [x] **P** `papers/baize.md` — Xu 2023; self-chat synthetic dialogue
- [x] **P** `papers/distilling-step-by-step.md` — Hsieh 2023; synthesize rationales + labels from a teacher, train smaller student
- [x] **P** `papers/genie.md` — Yehudai 2024; content-grounded synthetic generation for RAG/tool training

### 2b-ii Diversity + quality control

- [x] **P** `papers/instag-diversity.md` — Lu 2023 (covered in §2 but re-emphasize for synthetic QC)
- [x] **P** `papers/deita.md` — Liu 2023; complexity + quality + diversity scoring for SFT sample selection
- [x] **P** `papers/cherry-llm.md` — Li 2024; instruction-following difficulty as a selection signal
- [x] **P** `papers/superfiltering.md` — Li 2024; small-model perplexity ranking for filtering large synthetic sets
- [x] **P** `papers/ifd.md` — Instruction-Following Difficulty score; cheap synthetic-data filter
- [x] **P** `papers/less.md` — Xia 2024; gradient-based data selection for SFT
- [x] **P** `papers/alpagasus.md` — Chen 2023; ChatGPT-based quality rating to prune synthetic instructions
- [x] **P** `papers/prismatic-synthesis.md` — 2025 gradient-diversity framing for synthetic-data selection beyond embedding-space diversity
- [x] **P** `papers/synthetic-data-scaling-laws.md` — any recent paper on synthetic-data scaling behavior
- [x] **P** `papers/model-collapse.md` — Shumailov 2024; recursive training on synthetic data degrades models (nature paper)
- [x] **P** `papers/strong-model-collapse.md` — Dohmatob 2024; theoretical treatment
- [x] **P** `papers/faithful-synth-eval.md` — detecting when synthetic data preserves vs corrupts original distribution

### 2b-iii Synthetic preferences

- [x] **P** `papers/west-of-n.md` — Pace 2024; extract pairwise prefs from top/bottom-N sampling of a base model
- [x] **P** `papers/rlcd.md` — Yang 2023; contrastive self-generated preferences with positive/negative prompt templates
- [x] **P** `papers/direct-judgement-preference.md` — synthetic judge-LLM preferences
- [x] **P** `papers/ultrafeedback-construction.md` — UltraFeedback pipeline (multi-model + GPT-4 rating)
- [x] **P** `papers/hh-rlhf.md` — Anthropic HH (real human, not synthetic, but reference for comparison)

### 2b-iv Practical blogs + frontier-lab synthesis deep dives

- [x] **B** `blogs/hf-cosmopedia.md` — HuggingFace Cosmopedia synthetic pretrain corpus (30M docs)
- [x] **B** `blogs/nathan-lambert-synthetic-data.md` — Interconnects synthetic-data posts
- [x] **B** `blogs/allenai-tulu-synth.md` — Allen AI on Tülu 3 synthetic data construction
- [x] **B** `blogs/sebastian-raschka-synthetic.md` — if he has a dedicated synthetic-data post
- [x] **B** `blogs/llama-3-synthetic-pipeline.md` — summary of Llama 3 report's synthetic-data subsection
- [x] **B** `blogs/deepseek-r1-distill-synth.md` — how R1 distill stage synthesizes data

### 2b-v Open synthetic datasets (reference cards, not re-generation)

- [x] **P** `papers/openhermes-2-5.md` — Teknium OpenHermes 2.5 composition
- [x] **P** `papers/capybara.md` — Daniel LDL Capybara multi-turn
- [x] **P** `papers/dolphin.md` — Eric Hartford Dolphin datasets
- [x] **P** `papers/tulu-3-sft-mix.md` — Tülu 3's specific synthetic mix
- [x] **P** `papers/smol-talk.md` — HuggingFace SmolTalk
- [x] **P** `papers/opc-synthetic-code.md` — OpenCoder synthetic code SFT dataset

### 2b-vi Multi-turn conversation synthesis

- [x] **P** `papers/baize-construction.md` — Xu 2023; ChatGPT self-chat protocol, seed → N-turn amplification
- [x] **P** `papers/ultrachat-pipeline.md` — Cui 2023; three-sector taxonomy + two-model chit-chat for UltraChat
- [x] **P** `papers/camel.md` — Li 2023; role-playing multi-agent instruction generation
- [x] **P** `papers/openassistant.md` — Köpf 2023; crowdsourced conversation tree (real-human baseline for synthetic comparison)
- [x] **P** `papers/soda.md` — Kim 2023 (Yejin Choi); commonsense-grounded dialogue synthesis via social roles
- [x] **P** `papers/prosocial-dialog.md` — Kim 2022 (Yejin Choi); safety + pro-sociality synthetic dialogue
- [x] **P** `papers/longchat.md` — LMSYS LongChat; long-conversation synthesis via Vicuna
- [x] **P** `papers/system-prompt-diversity.md` — if a good ref exists on system-prompt-conditioned conversation synthesis

### 2b-vii Tool / function-calling data synthesis

- [x] **P** `papers/toolformer.md` — Schick 2023; self-supervised API annotation via perplexity filtering
- [x] **P** `papers/toolllm.md` — Qin 2023; ToolBench 16K real APIs, DFS-DT trajectory synthesis
- [x] **P** `papers/gorilla.md` — Patil 2023; API-calling retriever-augmented LLM
- [x] **P** `papers/api-bank.md` — Li 2023; evaluation-first API benchmark + data
- [x] **P** `papers/xlam.md` — Zhang 2024 (Salesforce); large action model SFT+DPO pipeline
- [x] **P** `papers/apigen.md` — Liu 2024 (Salesforce); verifiable synthetic function-calling (format + execution + semantic checks)
- [x] **P** `papers/apigen-mt.md` — 2025 multi-turn extension of APIGen
- [x] **P** `papers/toolace.md` — Liu 2024; self-evolution + dual-layer verification for tool data
- [x] **P** `papers/nexusraven.md` — Nexus; instruction-style function-call model + dataset
- [x] **P** `papers/granite-function-calling.md` — IBM Granite function-calling data
- [x] **P** `papers/hammer.md` — HF Hammer function-calling model / data
- [x] **P** `papers/glaive-function-calling.md` — Glaive V2 synthesis protocol
- [x] **P** `papers/bfcl.md` — Berkeley Function-Calling Leaderboard methodology

### 2b-viii Reasoning-trace synthesis (CoT, long-CoT, step-level)

- [x] **P** `papers/openmathinstruct.md` — Toshniwal 2024 (Nvidia); 1.8M math CoT synthesis via Mixtral
- [x] **P** `papers/openmathinstruct-2.md` — 2024 follow-up with Llama-3.1 405B teacher
- [x] **P** `papers/mathscale.md` — Tang 2024 (MS); concept-graph synthesis for math
- [x] **P** `papers/metamath.md` — Yu 2023; bootstrap rewriting (self-verification, FOBAR, backward reasoning)
- [x] **P** `papers/mammoth.md` — Yue 2023; hybrid CoT+PoT (program-of-thought) synthesis
- [x] **P** `papers/mammoth-2.md` — Yue 2024; web-scale reasoning extraction
- [x] **P** `papers/numina-math.md` — Numina math data construction
- [x] **P** `papers/xwin-math.md` — SOTA math recipe with curated synthesis
- [x] **P** `papers/rstar.md` — Qi 2024 (MS); MCTS + self-verification for small-model math
- [x] **P** `papers/rstar-math.md` — rStar-Math; deep-reasoning specialization
- [x] **P** `papers/open-thoughts.md` — 2025 reasoning-data recipe search with 1,000+ ablations across math/code/science
- [x] **P** `papers/s1.md` — Muennighoff 2025; "Simple test-time scaling" — 1000 curated reasoning traces beat large sets
- [x] **P** `papers/limo.md` — Ye 2025; "Less is More for Reasoning" — ~800 high-quality traces
- [x] **P** `papers/step-dpo.md` — Lai 2024; step-level preference synthesis for reasoning
- [x] **P** `papers/omegaprm.md` — Luo 2024; automated step labeling via Monte-Carlo rollouts
- [x] **P** `papers/bespoke-stratos.md` — Bespoke Labs R1-distill trace curation
- [x] **P** `papers/openr1.md` — HF Open-R1 project; open replication of R1 reasoning traces
- [x] **P** `papers/sky-t1.md` — Sky-T1; $450 training run via curated reasoning data
- [x] **P** `papers/qwen-qwq-traces.md` — QwQ reasoning-trace synthesis notes (from report/blog)

### 2b-ix Agentic trajectory synthesis

- [x] **P** `papers/agentinstruct.md` — Mitra 2024 (MS); automated agent-trajectory pipeline for Orca-Agent
- [x] **P** `papers/agenttuning.md` — Zeng 2023; agent-SFT data + AgentLM
- [x] **P** `papers/lumos.md` — Yin 2023; modular agent trajectories (plan/ground/execute)
- [x] **P** `papers/fireact.md` — Chen 2023; multi-trace fine-tuning for ReAct
- [x] **P** `papers/webarena-data.md` — WebArena trajectory collection
- [x] **P** `papers/swe-gym.md` — SWE-Gym trajectory synthesis for SWE agents
- [x] **P** `papers/swe-rl.md` — Wei 2025 (Meta); rule-based RL on software-engineering tasks with trajectory synthesis
- [x] **P** `papers/openhands-data.md` — OpenHands agent data if a paper exists
- [x] **P** `papers/kimi-k2-agentic-data.md` — Kimi K2's agentic synthesis (extract from the K2 report)
- [x] **P** `papers/autoact.md` — AutoAct agent-trajectory self-generation
- [x] **P** `papers/agent-flan.md` — Chen 2024; AgentFlan agent-tuning corpus
- [x] **P** `papers/explorer.md` — Explorer 2024 agentic synthesis (check release)
- [x] **P** `papers/magnetic-one.md` — Microsoft multi-agent framework, if data synthesis documented
- [x] **P** `papers/terminal-bench-trajectories.md` — if any public synthesis pipeline

### 2b-x Long-context synthesis (prompted by learner — priority)

The learner explicitly flagged long-context synthesis as a must-have. Long-context post-training data is its own hard problem: you need documents long enough, questions that genuinely require the full context, and faithful labels that aren't gameable by lexical lookup.

- [x] **P** `papers/longalpaca.md` — Chen 2023; LongLoRA companion long-context instruction data (32k–100k tokens)
- [x] **P** `papers/longalign.md` — Bai 2024; long-context SFT + DPO data recipe + packing strategy
- [x] **P** `papers/longmit.md` — long multi-turn instruction reference
- [x] **P** `papers/pose-synthesis.md` — Zhu 2023; Position-Skip-wise training for extended context
- [x] **P** `papers/prolong.md` — Gao 2024 (Princeton); ProLong training recipe with curated long-context SFT
- [x] **P** `papers/long-context-data-engineering.md` — Fu 2024; data engineering principles for 128k+ context
- [x] **P** `papers/longembed-synth.md` — long-context embedding SFT synthesis if a clean ref exists
- [x] **P** `papers/needle-in-haystack-data.md` — NIAH-style synthetic evaluation data lineage
- [x] **P** `papers/babilong.md` — BABILong long-context benchmark + synth
- [x] **P** `papers/ruler.md` — Hsieh 2024 (Nvidia); RULER benchmark and synthetic task generation protocol
- [x] **P** `papers/infinibench.md` / `papers/longbench.md` — long-context evaluation suites as synthetic-data references
- [x] **P** `papers/rag-instruct.md` — Liu 2024; retrieval-augmented instruction generation (bridges long-context + RAG)
- [x] **P** `papers/long-context-llama3.md` — extract from Llama 3 report's long-context post-training subsection
- [x] **P** `papers/qwen-long-context-synth.md` — Qwen 2.5-1M / Qwen 3 long-context data pipeline
- [x] **P** `papers/gemini-long-context-tricks.md` — any public Google disclosure on 1M-10M context data construction
- [x] **P** `papers/longrope-data.md` — LongRoPE data/fine-tuning companion

### 2b-xi Safety / refusal / red-team synthesis

- [x] **P** `papers/harmbench-data.md` — Mazeika 2024; red-team prompt synthesis
- [x] **P** `papers/wildguard-data.md` — Han 2024 (Allen AI); moderation + refusal data synthesis
- [x] **P** `papers/salad-bench.md` — Salad-Bench multi-category safety data
- [x] **P** `papers/circuit-breakers-data.md` — representation-engineering data synthesis
- [x] **P** `papers/anthropic-sleeper-agents-data.md` — Anthropic sleeper-agents training data

## 3. SFT — Methods

- [x] **P** `papers/sequence-packing.md` — packing/unpacking, FlashAttention's varlen API
- [x] **P** `papers/neftune.md` — Jain 2023; noisy-embedding SFT gain
- [x] **P** `papers/loss-masking-prompt.md` — prompt-masked vs full loss — Shi 2024 or equivalent
- [x] **P** `papers/packed-vs-unpacked-ablation.md` — if a clean source exists
- [x] **P** `papers/fsdp-sft.md` — Zhao (PyTorch); FSDP for SFT memory
- [x] **B** `blogs/hf-alignment-handbook.md` — HF Alignment Handbook SFT recipe
- [x] **B** `blogs/allenai-tulu-sft-recipe.md` — Tülu 3 SFT blog

## 4. RL — Algorithm Families

- [x] **P** `papers/ppo.md` — Schulman 2017; clip objective; value baseline
- [x] **P** `papers/trpo.md` — Schulman 2015; natural-gradient ancestry
- [x] **P** `papers/rlhf-instructgpt.md` — Ouyang 2022; RLHF template
- [x] **P** `papers/dpo.md` — Rafailov 2023; closed-form pref optimization
- [x] **P** `papers/ipo.md` — Azar 2023; identity preference objective
- [x] **P** `papers/kto.md` — Ethayarajh 2024; Kahneman-Tversky preference
- [x] **P** `papers/simpo.md` — Meng 2024; ref-free simple preference
- [x] **P** `papers/orpo.md` — Hong 2024; odds-ratio + SFT joint
- [x] **P** `papers/rpo.md` — Pang 2024 (iterative reasoning)
- [x] **P** `papers/rloo.md` — Ahmadian 2024; REINFORCE-leave-one-out
- [x] **P** `papers/reinforce-plus-plus.md` — 2025 variance-reduction variant
- [x] **P** `papers/grpo.md` — DeepSeekMath 2024; group-relative policy opt
- [x] **P** `papers/dr-grpo.md` — 2025 bias-corrected GRPO
- [x] **P** `papers/vanilla-pg.md` — REINFORCE baseline
- [x] **P** `papers/rloo-vs-grpo.md` — comparison study if one exists
- [x] **P** `papers/reinforcement-learning-with-one-training-example.md` — 2025 one-shot RLVR for math reasoning

## 5. RL — Entropy Dynamics

- [x] **P** `papers/entropy-mechanism-llm-rl.md` — Cui 2025 "The Entropy Mechanism of RL for LLMs"
- [x] **P** `papers/entropy-collapse-ppo.md` — any rigorous analysis of entropy collapse
- [x] **P** `papers/echo-chamber-rl-post-training.md` — RL amplifies pretrained behaviors / output modes
- [x] **P** `papers/maximum-entropy-rl.md` — Haarnoja 2018 (SAC); entropy-regularized RL ancestry
- [x] **P** `papers/kl-control-rlhf.md` — Jaques 2019 / Korbak 2022; KL-as-regularizer line
- [x] **P** `papers/entropy-regularization-ppo.md` — Mnih A2C entropy bonus; carry-through in RLHF
- [x] **P** `papers/sampling-temperature-schedule.md` — temperature annealing in RL rollouts
- [x] **P** `papers/policy-coverage-loss.md` — if a dedicated paper exists
- [x] **B** `blogs/nathan-lambert-entropy-rl.md` — Interconnects entropy/RL posts
- [x] **B** `blogs/openrlhf-entropy-debugging.md` — practitioner notes if available

## 6. RL — Reward Modeling, Hacking, PRMs, RLVR

- [x] **P** `papers/reward-model-overoptimization.md` — Gao 2022 (Goodhart in RM)
- [x] **P** `papers/reward-hacking-taxonomy.md` — Skalse 2022 or equivalent
- [x] **P** `papers/spurious-rewards-rlvr.md` — random / negatively correlated rewards can still improve RLVR via GRPO clipping bias
- [x] **P** `papers/bradley-terry-rm.md` — foundational preference-model theory
- [x] **P** `papers/constitutional-ai.md` — Bai 2022; RLAIF
- [x] **P** `papers/rlaif-scaling.md` — Lee 2023 Google
- [x] **P** `papers/ultrafeedback.md` (covered above if reused — keep as preference data)
- [x] **P** `papers/prm800k.md` — Lightman 2023; process reward data
- [x] **P** `papers/math-shepherd.md` — Wang 2023; automatic PRM labeling
- [x] **P** `papers/rlvr-tulu3.md` — Lambert 2024; verifiable rewards in Tülu 3
- [x] **P** `papers/deepseek-r1.md` — R1 and R1-Zero; pure-RL reasoning
- [x] **P** `papers/judge-llm-bias.md` — Zheng 2023 MT-Bench judges
- [x] **P** `papers/reward-ensembling.md` — Coste 2023; ensembling against overoptimization
- [x] **P** `papers/generative-reward-models.md` — Zhang 2024 or equivalent
- [x] **P** `papers/pairrm.md` — pairwise reward model line
- [x] **B** `blogs/lilianweng-reward-hacking.md` — Weng reward-hacking survey

## 7. RL — Rollout, Replay, Sampling, Infra

- [x] **P** `papers/iterative-sft-rl.md` — Llama 2 / Tülu iterative schemes
- [x] **P** `papers/on-off-policy-rlhf.md` — Tang 2024 analysis
- [x] **P** `papers/best-of-n.md` — Stiennon 2020; BoN vs RL
- [x] **P** `papers/replay-buffer-rlhf.md` — if a principled paper exists; else cite frameworks
- [x] **P** `papers/minibatch-sharing-rl.md` — batching across prompts
- [x] **P** `papers/async-rollout.md` — async actor-learner for LLM RL (e.g., OpenRLHF async, verl)
- [x] **F** `frameworks/verl-rollout.md` — verl rollout loop (file refs + code)
- [x] **F** `frameworks/verl-ppo-loss.md` — verl PPO loss implementation
- [x] **F** `frameworks/verl-grpo.md` — verl GRPO
- [x] **F** `frameworks/openrlhf-ppo.md` — OpenRLHF PPO trainer
- [x] **F** `frameworks/openrlhf-dpo.md` — OpenRLHF DPO trainer
- [x] **F** `frameworks/trl-ppo.md` — HF TRL PPO
- [x] **F** `frameworks/trl-grpo.md` — HF TRL GRPO
- [x] **F** `frameworks/trl-online-dpo.md` — HF TRL online DPO
- [x] **F** `frameworks/entropy-logging-patterns.md` — how each framework logs entropy/kl

## 8. RL — Self-Improvement & Verifiable Reasoning

- [x] **P** `papers/self-rewarding-lm.md` — Yuan 2024
- [x] **P** `papers/meta-rewarding-lm.md` — Wu 2024 follow-up
- [x] **P** `papers/spin.md` — Self-Play Fine-Tuning, Chen 2024
- [x] **P** `papers/self-play-preference.md` — Munos 2024 (Nash-LM)
- [x] **P** `papers/self-correct-rl.md` — Kumar 2024 SCoRe
- [x] **P** `papers/v-star.md` / `papers/rest-em.md` — expectation-maximization self-training
- [x] **P** `papers/lets-verify.md` — Lightman 2024 "Let's Verify Step by Step"
- [x] **P** `papers/r1-zero-analysis.md` — any follow-up analyzing R1-Zero emergent reasoning
- [x] **P** `papers/yejin-choi-rainbow.md` — Rainbow/commonsense line if directly relevant
- [x] **P** `papers/west-of-n.md` — Pace 2024 synthetic-preference
- [x] **P** `papers/rlvr-beyond-base-model.md` — large-pass@k critique: RLVR sharpens sampling more than capability boundary
- [x] **P** `papers/prorl.md` — prolonged RL + KL control + policy reset can expand reasoning boundary
- [x] **P** `papers/transferability-of-llm-reasoning.md` — math-only reasoning gains transfer weakly; RL transfers better than SFT
- [x] **P** `papers/front-loading-reasoning.md` — early reasoning-data injection in pretraining dominates late-only post-training
- [x] **P** `papers/interplay-pretraining-midtraining-rl.md` — 2025 controlled study of pre-training vs mid-training vs RL
- [x] **P** `papers/rlp-reinforcement-as-pretraining-objective.md` — 2026 dense verifier-free RL objective during pretraining

## 9. Frontier Model Reports

- [x] **R** `model-reports/llama-3.md` — Grattafiori 2024; SFT→DPO pipeline; data mix
- [x] **R** `model-reports/llama-2.md` — RSFT + PPO
- [x] **R** `model-reports/qwen-2.5.md`
- [x] **R** `model-reports/qwen-3.md` — unified thinking/non-thinking report with reasoning-stage pretraining and RL
- [x] **R** `model-reports/deepseek-v3.md` — MoE + FP8 + DualPipe + SFT/RL + R1 distillation
- [x] **R** `model-reports/deepseek-r1.md` — pure-RL reasoning
- [x] **R** `model-reports/deepseekmath.md` — GRPO birthplace
- [x] **R** `model-reports/kimi-k2.md` — agentic long-horizon RL
- [x] **R** `model-reports/tulu-3.md` — Lambert 2024; full open post-training
- [x] **R** `model-reports/olmo-2.md` — Allen AI open pretraining + post-training
- [x] **R** `model-reports/olmo-3.md` — full model-flow release: pretraining -> mid-training -> long-context -> SFT/DPO/RLVR
- [x] **R** `model-reports/olmo-3.md` — Allen AI latest (2025/2026); post-training recipe evolution
- [x] **R** `model-reports/mistral-nemo.md` or `mixtral.md` — where post-training disclosed
- [x] **R** `model-reports/gemma-2.md` / `gemma-3.md` — Google distillation + RLHF
- [x] **R** `model-reports/yi.md` — 01.AI post-training
- [x] **R** `model-reports/nemotron.md` — Nvidia 340B / Nemotron-4 with full RM recipe
- [x] **R** `model-reports/phi-3.md` — synthetic-data-heavy SFT

### 9a. Latest-wave 2025/2026 models (must include)

- [x] **R** `model-reports/olmo-3.md` — Allen AI 2025/2026; what changed from OLMo 2 in post-training
- [x] **R** `model-reports/llama-4.md` — Meta's latest (Scout/Maverick/Behemoth) post-training disclosure
- [x] **R** `model-reports/qwen-3.md` — Alibaba Qwen 3 full report; hybrid-thinking post-training
- [x] **R** `model-reports/deepseek-v3.1.md` or `deepseek-v3.2.md` — latest DeepSeek refresh + sparse attention
- [x] **R** `model-reports/deepseek-r1-followup.md` — any R1.5 / R2 / distill update
- [x] **R** `model-reports/kimi-k2.md` — Moonshot agentic RL
- [x] **R** `model-reports/kimi-k1-5.md` — Moonshot K1.5 long-CoT RL report
- [x] **R** `model-reports/minimax-01.md` — MiniMax-01 lightning attention + post-training
- [x] **R** `model-reports/hunyuan-large.md` — Tencent Hunyuan
- [x] **R** `model-reports/grok-3.md` or `model-reports/grok-4.md` — xAI (if public post-training disclosed)
- [x] **R** `model-reports/mistral-large-2.md` — Mistral latest public
- [x] **R** `model-reports/pixtral-large.md` or `mistral-small-3.md` — recent Mistral post-training
- [x] **R** `model-reports/gemini-2.5-deep-research.md` — any Google post on Gemini 2.5/3 RL recipe (blog-level OK)
- [x] **R** `model-reports/nemotron-ultra.md` / `nemotron-nano.md` — Nvidia 2025 post-training reports
- [x] **R** `model-reports/tulu-3.1.md` — any Tülu 3 refresh
- [x] **R** `model-reports/smollm-3.md` — HuggingFace SmolLM 3 post-training
- [x] **R** `model-reports/phi-4.md` — Microsoft Phi-4 / Phi-4-reasoning
- [x] **R** `model-reports/skywork-o1.md` — Skywork open-reasoning models
- [x] **R** `model-reports/internlm-3.md` — Shanghai AI Lab InternLM3
- [x] **R** `model-reports/glm-4.md` or `glm-4-5.md` — Zhipu GLM post-training

## 10. Labs & Researchers

- [x] **L** `labs/allen-ai.md` — Tulu, OLMo, Dolma, evaluation/tooling, model-flow worldview
- [x] **L** `labs/yejin-choi-group.md` — Self-Instruct, STaR lineage, alternative training recipes, reasoning limits
- [x] **L** `labs/deepseek.md` — V2/V3/R1/Math summary
- [x] **L** `labs/alibaba-qwen.md`
- [x] **L** `labs/moonshot-kimi.md`
- [x] **L** `labs/anthropic-safety-research.md` — Constitutional AI, weak-to-strong, sleeper agents
- [x] **L** `labs/openai-alignment.md` — InstructGPT, weak-to-strong generalization
- [x] **L** `labs/nathan-lambert-interconnects.md` — blog-as-lab index

## 11. Blogs & Postmortems

- [x] **B** `blogs/lilianweng-rlhf.md`
- [x] **B** `blogs/lilianweng-reasoning-llms.md`
- [x] **B** `blogs/nathan-lambert-rl-overview.md`
- [x] **B** `blogs/nathan-lambert-grpo.md`
- [x] **B** `blogs/hf-dpo-zoo.md`
- [x] **B** `blogs/hf-rlhf-illustrated.md`
- [x] **B** `blogs/allenai-tulu-blog.md`
- [x] **B** `blogs/costa-huang-ppo-details.md` — 37 implementation details of PPO
- [x] **B** `blogs/john-schulman-kl-tricks.md` — KL approximation tricks

---

## Gap log

After the first collection pass, list any bullet above that could not be filled (source paywalled, no good extract, contradictions) here. The planner uses the gap log to decide whether a chapter needs to be cut or scope-narrowed.

- (empty — populated after pass 1)
