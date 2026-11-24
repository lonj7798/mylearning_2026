<!-- scope: ZeroSearch (Sun et al., Tongyi Lab, 2025) — RL for search-calling QA agents in which a fine-tuned LLM replaces the real search engine during rollouts; noise-controlled simulated documents, curriculum on document quality, F1 reward, document-token loss masking
     deps: [[deepseek-r1]], [[toolformer]]
     see-also: [[gigpo-verl-agent]], [[kimi-k2-agentic-data]], [[agentgym-rl]], [[demystifying-agentic-rl]], [[rag-instruct]]
-->

# ZeroSearch: Incentivize the Search Capability of LLMs without Searching
- **Core Insight:** A Qwen-2.5-14B-Instruct model fine-tuned to generate either useful or noisy search results on command can replace Google Search during RL rollouts: with a Qwen-2.5-3B-Base policy evaluated against real Google Search, the 14B simulator gives 34.47 average EM and the 7B simulator 33.53, against 32.81 when training on real Google results; a prompted (not fine-tuned) 14B simulator gives 31.50 (§5.2, Table 4).
- **Guideline:** When the RL environment is an external API that is costly or returns uncontrolled quality, train a simulator of that API on logged real interactions, expose a quality switch, raise the share of degraded responses over training, and mask simulator tokens out of the policy loss; in this paper the simulator cost $70.8 of GPU time against $586.7 of search API fees for the same ~64,000 requests (App. D, Table 8).
- **Authors:** Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, Jingren Zhou (Tongyi Lab, Alibaba Group)
- **Year:** 2025 (arXiv v1 2025-05-07; v2 2025-05-16; v3 2026-05-19)
- **URL:** https://arxiv.org/abs/2505.04588 (code: https://github.com/Alibaba-NLP/ZeroSearch; data: https://huggingface.co/datasets/sunhaonlp/ZeroSearch_dataset)
- **Source type:** paper
- **Relevant topics:** agentic RL, search agents, simulated environments, curriculum learning, reward hacking, observation-token loss masking, environment cost

## Abstract
Search improves LLM reasoning and generation, and recent work trains search use with RL against live search engines. The paper names two problems with live engines: document quality is uncontrolled, which adds noise and instability, and RL needs many rollouts, which can mean hundreds of thousands of API calls. ZeroSearch removes the live engine from training. A lightweight supervised fine-tuning step turns an LLM into a retrieval module that can produce relevant or noisy documents for a query. During RL, a curriculum lowers the quality of generated documents step by step. A 3B retrieval module is enough to train search behaviour, a 7B module matches a real search engine, and a 14B module exceeds it. The method works for base and instruction-tuned policies of several sizes and for several RL algorithms (Abstract).

## Key Contributions
- Search-simulation SFT: an LLM trained on query–document pairs from real search trajectories, with a prompt keyword selecting useful or noisy output (§3.3, Table 2).
- A curriculum that raises the probability of noisy documents with an exponential schedule over training steps (§3.4, Eq. 1).
- An F1 answer reward chosen after exact-match reward led to reward hacking (§3.5).
- Loss masking of simulator-generated document tokens (§3.6; App. C, Table 7).
- Comparison against Search-R1 trained on real Google results, across three backbones and base/instruct variants (§4.4, Table 3), and a cost comparison (App. D, Table 8).

## Key Figures/Tables to Study
- Table 2: the simulator prompt, including the ground-truth answer.
- Table 3: main results, seven QA datasets. Table 4: simulator size and SFT vs prompting.
- Tables 5–7: RL algorithm, curriculum vs random noise, document-token masking.
- Fig. 2: reward curves vs Search-R1 and the number of interaction turns during training.
- Table 8: cost of simulator GPUs vs search API.

## Technical Details
- **Template (§3.2, Table 1).** The policy reasons in `<think>`, issues queries in `<search>`, receives documents in `<information>`, and answers in `<answer>`. App. E states the policy followed this format although the reward does not check format.
- **Simulator data (§3.3).** An LLM is prompted to interact with a real search engine over multiple turns until it answers. Query–document pairs are extracted from these trajectories. An LLM judge labels each document useful if it contains enough information to answer its query and noisy otherwise. The simulator is then fine-tuned to produce both kinds. The number of SFT examples is not reported.
- **Simulator prompt (Table 2).** "You are the Google search engine", generate five useful or noisy documents of about 30 words each; the prompt includes the user's question and its ground-truth answer, which the authors add to extend the simulator's knowledge coverage (§3.3).
- **Simulator backbones (App. F).** Qwen-2.5-3B/7B/14B-Instruct, SFT learning rate 1e-6; the fine-tuned 14B is the default (§4.3).
- **Curriculum (§3.4, Eq. 1).** p_i = p_s + ((b^(i/m) − 1)/(b − 1))·(p_e − p_s), where p_i is the probability of noisy documents at step i of m and b = 4. Values (App. F): Qwen-2.5-7B p_s = 0, p_e = 0.75; Qwen-2.5-3B 0 and 0.25; LLaMA-3.2-3B 0.25 and 0.5.
- **Reward (§3.5).** r = 2·IN / (PN + RN), with IN the overlapping words between prediction and ground truth, PN the prediction word count, RN the ground-truth word count. The authors report that an exact-match reward led the policy to produce excessively long answers to raise the chance of containing the correct answer.
- **Masking (§3.6).** Document tokens are excluded from the policy loss; the authors state that uniform optimization over both token types destabilizes training.
- **RL setup (§4.3, App. D, App. F).** Policies: Qwen-2.5-7B, Qwen-2.5-3B, LLaMA-3.2-3B, each Base and Instruct. Default algorithm REINFORCE; GRPO and PPO also run. Policy LR 1e-6; PPO value LR 1e-5; GAE λ = 1, γ = 1. Five responses per prompt. Batch 64, rollout repetition 5, 200 steps, about 12 hours and about 64,000 search requests for the 7B backbone. Simulator on 4 H20 GPUs, RL on another 4 H20 GPUs. Five documents per search. KL coefficient and maximum turns are not reported.
- **Data and metric (§4.1, §4.3).** Training: merged NQ and HotpotQA training sets. Evaluation: NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle; exact match. All methods use real Google Search (SerpAPI) at evaluation.
- **Main results, average EM (Table 3).** Search-R1 base / instruct → ZeroSearch base / instruct: Qwen-2.5-7B 39.51 / 38.17 → 40.93 / 39.08; Qwen-2.5-3B 32.81 / 28.13 → 34.47 / 31.70; LLaMA-3.2-3B 34.21 / 30.48 → 36.07 / 30.98. RAG baselines are 25.09–25.51. ZeroSearch is below Search-R1 on some single datasets, for example Qwen-2.5-7B-Base NQ 42.40 vs 43.40 and 2Wiki 34.00 vs 37.20 (Table 3).
- **Ablations (Qwen-2.5-3B-Base unless noted).** RL algorithm: REINFORCE 34.47, GRPO 33.17, PPO 32.67 (Table 5). Curriculum vs fixed noise probability 0.5: 34.47 vs 32.59; LLaMA-3.2-3B-Base 36.07 vs 34.84 (Table 6). Masking, LLaMA-3.2-3B (App. C; the with-mask row equals the Base row of Table 3): with 36.07, without 34.53 (Table 7).
- **Training dynamics (§5.3, Fig. 2c).** With LLaMA-3.2-3B-Base, the number of interaction turns first drops sharply, then rises with the reward and stabilizes.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Simulator (Qwen-2.5-14B-Instruct, default) | 3B/7B/14B | SFT | learning rate; labels | 1e-6; LLM-judged useful vs noisy documents from real search trajectories | arXiv:2505.04588v3 §3.3, App. F | verified 2026-11-24 | Table 4: SFT-14B 34.47 vs Prompt-14B 31.50 vs Google 32.81 |
| ZeroSearch policies | 3B, 7B | RL | algorithm | REINFORCE (v3 default) | v3 §4.3, App. F | verified 2026-11-24 | Table 5: REINFORCE 34.47, GRPO 33.17, PPO 32.67 |
| ZeroSearch policies | 3B, 7B | RL | LR; samples/prompt; batch; steps | 1e-6; 5; 64; 200 | v3 App. D, App. F | verified 2026-11-24 | no ablation reported |
| ZeroSearch policies | 3B, 7B | RL | noise curriculum p_s → p_e, b | 7B: 0 → 0.75; Qwen-3B: 0 → 0.25; LLaMA-3B: 0.25 → 0.5; b = 4 | v3 §3.4, App. F | verified 2026-11-24 | Table 6: curriculum beats fixed 0.5 by 1.88 and 1.23 EM |
| ZeroSearch policies | 3B, 7B | RL | reward | word-level F1 vs ground truth | v3 §3.5 | verified 2026-11-24 | qualitative: EM reward led to long answers; no number reported |
| ZeroSearch policies | 3B, 7B | RL | document-token loss mask | on | v3 §3.6, App. C | verified 2026-11-24 | Table 7: 36.07 with vs 34.53 without |
| ZeroSearch policies | 3B, 7B | RL | KL coefficient; max turns | not reported | checked v3 body, App. D, F | not reported | — |

## Findings relevant to generality and negative feedback
- **Generality.** Training uses only NQ and HotpotQA; the other five datasets are treated as out-of-domain (§4.4). Gains on averages hold across three backbones and base/instruct variants, but are not uniform per dataset (Table 3). Evaluation stays within short-answer QA; the method is not tested on open-ended tasks.
- **Negative feedback.** Noisy documents are a controlled negative observation: the policy must answer while some retrieved documents are uninformative, and the share of such documents rises over training (§3.4). The exact-match reward is a reported reward-hacking case (long answers), fixed by changing the reward to F1 (§3.5).
- **Simulator leakage.** The simulator prompt contains the ground-truth answer (Table 2), so a useful document can state the answer more directly than a real search result; the paper does not measure this effect. Evaluation uses real Google Search, which limits the risk to training dynamics rather than reported scores (§4.3).
- **Limitations stated (§6, App. D).** The simulator needs GPU servers, and its utilization is high during rollout and low during policy updates.

## Connections
- [[gigpo-verl-agent]]: reports ZeroSearch as a baseline on the same seven-dataset QA setup (Table 2 there).
- [[kimi-k2-agentic-data]]: cites ZeroSearch as one basis of its simulated tool-environment pipeline (§3.1.1 there).
- [[agentgym-rl]], [[demystifying-agentic-rl]]: other multi-turn agentic RL setups; compare environment cost and observation masking.
- [[toolformer]]: earlier search-API use learned from self-labeled data rather than RL.
- [[rag-instruct]]: instruction data for retrieval-augmented generation, the SFT counterpart of this RL setting.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2505.04588 and the v3 PDF text (pdftotext, dated 2026-05-19); v1 HTML compared for the main differences.
- Version note: v1 used GRPO as the default algorithm and reported different numbers, e.g. Table 3 Qwen-2.5-3B-Base ZeroSearch 33.97 vs Search-R1 32.47 and Table 4 SFT-14B 33.97, Google 32.47; v1 gave no p_s/p_e values. This card follows v3.
- Table 8 prices GPUs as AWS A100 while experiments ran on H20 GPUs (§4.3, App. D); the paper does not reconcile this.
- The Table 4 policy (Qwen-2.5-3B-Base) is named in the v3 Table 4 caption.
- Not reported by the source: number of simulator SFT examples and epochs, KL coefficient, maximum interaction turns, the judge model used to label documents.
