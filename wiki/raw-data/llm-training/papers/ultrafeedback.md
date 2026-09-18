<!-- scope: UltraFeedback — GPT-4-annotated preference dataset, its construction pipeline, and the UltraRM / UltraCM / UltraLM-13B-PPO models trained on it
     see-also: [[ultrafeedback-construction]], [[rlcd]], [[west-of-n]], [[direct-judgement-preference]], [[ultrachat-pipeline]]
-->

# ULTRAFEEDBACK: Boosting Language Models with Scaled AI Feedback
- **Core Insight:** GPT-4 annotation of 255,864 completions over 63,967 instructions produces a preference dataset on which a 13B reward model reaches 71.0% average preference accuracy across four human-annotated benchmarks, above every open reward-model baseline compared (§3.1, Table 2).
- **Guideline:** When collecting AI preference labels, score all completions for one instruction in a single GPT-4 call on four separate aspects rather than requesting one overall score per completion, because the fine-grained variant scored 65.2% on OpenAI WebGPT against 62.0% for the overall-score variant (§3.1, Table 2).
- **Authors:** Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, et al.
- **Year:** 2023 (arXiv v1 2023-10; ICML 2024, PMLR 235)
- **URL:** https://arxiv.org/abs/2310.01377
- **Source type:** paper
- **Relevant topics:** synthetic preference data, GPT-4 as annotator, reward modeling, critique modeling, best-of-n sampling, PPO

## Abstract
Human feedback for alignment is limited by time, labor, and annotator capability, so existing preference datasets are small or restricted to narrow topics. The paper collects AI feedback instead, treating scale and diversity as the two properties that determine whether preference data is effective. Instructions and responses are broadened in both amount and breadth, and several techniques are applied to reduce annotation bias. The result is UltraFeedback, a dataset of over one million GPT-4 feedback entries for 250k user-assistant conversations, rated on several aspects. A LLaMA-based model is then aligned by best-of-n sampling and reinforcement learning on top of this data, and evaluated on chat benchmarks.

## Key Contributions
- A preference-collection pipeline that applies diversity at three stages: instruction sourcing, completion sampling, and annotation (§2.1).
- An instruction set of 63,967 items drawn from six public datasets, decontaminated by 13-gram matching against three evaluation sets (§2.2, App. B).
- A 17-model completion pool spanning sizes, architectures, and series, with a randomly sampled "principle" prepended to each completion request to vary style (§2.3).
- A four-aspect GPT-4 annotation scheme — instruction-following, truthfulness, honesty, helpfulness — with per-aspect scores documented from 1 to 5 plus a textual rationale, and a separate per-completion critique (§2.4).
- Three released models: UltraRM (reward), UltraCM (critique), UltraLM-13B-PPO (chat policy) (§2.6, §3.3).

## Key Figures/Tables to Study
- **Figure 1** — the three-stage construction pipeline with an example of the four per-aspect rankings GPT-4 emits for one instruction.
- **Table 1** — dataset statistics against OASST1, WebGPT, Anthropic Helpful, OpenAI Summarization, QA Feedback, SelFee, Shepherd.
- **Table 2** — reward-modeling accuracy, including the fine-grained vs overall-score ablation.
- **Table 4 and Table 5** — GPT-4/Claude-3/human agreement, the paper's own check on whether the AI labels track human preference.

## Technical Details
- **Instruction sources (§2.2):** all of TruthfulQA and the FalseQA training set; 10k sampled from Evol-Instruct; 10k from UltraChat; 20k from ShareGPT; FLAN by stratified sampling (3k from the CoT subset, 10 instructions per task for the other three subsets). After contamination detection, 63,967 instructions remain (§2.2).
- **Completion pool (§2.3):** 17 models — GPT-4, gpt-3.5-turbo, Bard; UltraLM-13B/65B, WizardLM-7B-v1.1/13B-v1.2/70B-v1.1, Vicuna-33B-v1.3, LLaMA2-7B/13B/70B-Chat, Alpaca-7B; MPT-30B-Chat, Falcon-40B-Instruct, StarChat, Pythia-12B. Four models are sampled at random per instruction (§2.3).
- **Principles (§2.3):** one principle per aspect is hand-written, then GPT-4 generates ten more from that example; a principle is sampled per completion and placed in the system prompt (App. G.1, Fig. 6).
- **Annotation (§2.4):** four techniques — decomposition into four aspects; a documented 1-to-5 scoring standard (App. G.2); scoring all four completions of an instruction in one prompt; requiring a rationale. Critiques are generated separately, per completion, from an overall perspective.
- **Scale (§2.4, §2.5, Table 1):** 255,864 completions from 63,967 instructions; over 1 million feedback entries; 340,025 preference pairs; 255,864 critiques; average prompt length 185.1 tokens, response 305.3, critique 143.1. All dialogues are single-turn (App. A).
- **Decontamination (App. B):** 13-gram matching against AlpacaEval, Evol-Instruct, and UltraChat test sets removed 48 samples; no other evaluation set was checked, and the authors ask users to decontaminate against their own evaluation data.
- **UltraRM (§3.1, Table 2):** LLaMA2-13B. Average preference accuracy 71.0% (Anthropic Helpful 71.0, WebGPT 65.2, OpenAI Summ. 74.0, Stanford SHP 73.7). Trained on UltraFeedback only: 66.8% average. Trained with overall scores instead of fine-grained: 69.9% average, and 62.0 on WebGPT. The closed LLaMA2-70B helpfulness reward model is higher on Anthropic Helpful (72.0) and SHP (80.0).
- **Best-of-n (§3.2, Fig. 2):** 16 completions sampled from UltraLM-13B at temperature 1, top-p 1, ranked by UltraRM. AlpacaEval win rate against text-davinci-003: 76.53% at n=1, 84.64% at n=2, 91.54% at n=16.
- **PPO (§3.3, Table 3):** UltraLM-13B-PPO reaches 86.3% AlpacaEval win rate, 57.8/10.1/32.1 win/tie/lose against gpt-3.5-turbo on Evol-Instruct, and 64.9/15.6/19.5 on UltraChat; average win rate 69.7% against 52.9% for UltraLM-13B, a 16.8-point gain (§3.3).
- **Judge agreement (§4.1, Table 4):** GPT-4 agrees with individual annotators 59.7% on average over 400 sampled comparisons, and 68.6% with the majority vote of three annotators; random agreement with ties included is 33%. Inter-annotator agreement is 56.4-58.1%.
- **Judge reliability (§4.2, Table 5, Fig. 3):** on 266 pairs, average win rates are GPT-4 67.3%, Claude-3 Sonnet 76.1%, human 64.3%. On reasoning items GPT-4 gives fewer ties and more losses than humans; a ground-truth-answer re-scoring yields 42.1/26.3/31.6 win/tie/lose, close to the human judgment.
- **UltraCM (App. E.3, Table 8):** LLaMA2-13B trained on the 255,864 critiques. GPT-4 rates critique quality 1-7 over nine datasets; UltraCM averages 5.92 against gpt-3.5-turbo 6.17, SelFee-13B 5.41, WizardLM-13B-v1.2 5.40, Shepherd-13B 3.23.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| UltraRM | 13B (LLaMA2-13B) | reward-model | training data | UltraFeedback + Stanford SHP + OpenAI Summarization + Anthropic Helpful | arXiv:2310.01377v2 §3.1, App. E.1 | verified 2026-09-18 | Table 2: mixing raises average from 66.8 (UltraFeedback only) to 71.0 |
| UltraRM | 13B | reward-model | loss | binary ranking loss with margin m(r) = absolute difference of annotated rewards, normalized to (0,1]; m(r)=0 for ranking-only datasets | App. D.1, Eq. 1 | verified 2026-09-18 | no ablation reported |
| UltraRM | 13B | reward-model | epochs / batch / peak LR | 1 epoch; 512 pairs (1024 completions); 1e-5 | App. D.1 | verified 2026-09-18 | no ablation reported; stated as following Touvron et al. 2023b |
| UltraRM | 13B | reward-model | LR schedule | cosine decay, 3% warm-up ratio, final LR 1e-6 | App. D.1 | verified 2026-09-18 | no ablation reported |
| UltraCM-13B | 13B (LLaMA2-13B) | SFT | epochs / batch / LR | 2 epochs; batch 256; 2e-5; same scheduler as reward modeling | App. D.2 | verified 2026-09-18 | no ablation reported |
| UltraLM-13B-PPO | 13B | RL | iterations / samples per iteration / mini-batch / LR | 80 iterations; 512 samples per iteration; mini-batch 64; fixed 1e-6 | §3.3 | verified 2026-09-18 | no ablation reported |
| UltraLM-13B-PPO | 13B | RL | prompts | UltraFeedback instructions | §3.3 | verified 2026-09-18 | no ablation reported |
| UltraLM-13B + UltraRM | 13B | eval-gate | best-of-n sampling | n ∈ {1,2,4,8,16}, temperature 1, top-p 1 | §3.2 | verified 2026-09-18 | Fig. 2: win rate rises 76.53 → 91.54 from n=1 to n=16 |
| All evaluated models | — | eval-gate | benchmark decoding | temperature 0.7, top-p 1 | §3.3 | verified 2026-09-18 | no ablation reported |

PPO KL coefficient, clip ε, and total compute are not reported in the paper.

## Findings relevant to generality
- The reward model trained only on UltraFeedback transfers to four human-labeled preference sets it was not trained on, averaging 66.8% and exceeding the open baselines by over 6.3 points; the authors read this as out-of-distribution generalization from diversity (§3.1).
- WebGPT has no train/test split and neither UltraRM nor most baselines train on it, which the authors use as the cleanest generalization check; UltraRM improves 2.6 absolute points over baselines there (§3.2 preamble, §3.1).
- Capability breadth after PPO is measured on nine held-out datasets covering world knowledge, commonsense reasoning, and reading comprehension (App. E.4), and by question type on the UltraChat evaluation set, where UltraLM-13B-PPO is highest overall (105.7) but is not ahead on math and reasoning (App. E.5, Table 9).
- Limitation stated by the authors: GPT-4 does not model human preference precisely in all situations, and the dataset is single-turn only (App. A).

## Findings relevant to negative feedback
- Preference pairs are built from completions of the same instruction scored by aspect; the deliberate mix of strong and weak generators in the 17-model pool is the mechanism that produces separable quality gaps (§2.3). The paper does not measure how much of the reward-model gain comes from the low-scoring side of each pair.

## Connections
- [[ultrafeedback-construction]] — redirect card for the same artifact.
- [[ultrachat-pipeline]] — source of 10k of the instructions (§2.2).
- [[rlcd]], [[west-of-n]] — preference-pair construction without an external judge.
- [[direct-judgement-preference]], [[generative-reward-models]] — later work that trains the judge itself; uses UltraFeedback as training or evaluation data.
- [[hh-rlhf]] — one of the four human-labeled benchmarks in Table 2 and one of the mixed-in training sets.
- [[pairrm]], [[kto]], [[orpo]], [[deita]], [[magpie]] — downstream users of UltraFeedback or its binarized derivative.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2310.01377 (arXiv v2, 16 Jul 2024; ICML 2024).
- Corrections to the previous card version:
  - "Year: 2024" → arXiv v1 2023-10, published ICML 2024 (title page).
  - Author list ran to twelve names → first six then "et al." per the card standard; the paper lists thirteen authors including Guotong Xie.
  - "GPT-4 rates each response on 4 aspects with numerical ratings (0–10)" → the documented score range is 1 to 5 (§2.4, App. G.2).
  - "64K prompts × 4 responses = 256K rated samples" → 63,967 instructions and 255,864 completions (§2.2, §2.4).
  - "Binarized subset ≈ 62K pairs" → the paper reports 340,025 preference pairs (Table 1); the ~62k `ultrafeedback_binarized` set is a third-party HuggingFace derivative, not described in the paper.
  - "Response side: multiple different generator models answer each prompt" left the count unstated → four models sampled at random per instruction from a pool of 17 (§2.3).
  - "Trains a chat model aligned with AI feedback" → the chat model is UltraLM-13B-PPO, trained with PPO against UltraRM (§3.3).
- Removed as unsupported by the source: "became a default preference source"; "competitive with the strongest open chat models of the period" as a generic claim, replaced with the Table 3 numbers; "judge bias becomes part of the dataset and can propagate into every downstream model" (the paper measures GPT-4/human agreement, §4, but does not measure propagation); "binarizing throws away a lot of useful structure"; "if the model pool is too homogeneous the preference gap shrinks" (stated by the authors as a design motivation in §2.3, not as a measured result, and reworded accordingly); "UltraFeedback is a pattern for manufacturing reusable alignment supervision from a strong judge" (framing).
- Not reported by the source: PPO KL coefficient and clip range; annotation cost in USD; per-source instruction counts after decontamination; the construction of the `ultrafeedback_binarized` variant.
