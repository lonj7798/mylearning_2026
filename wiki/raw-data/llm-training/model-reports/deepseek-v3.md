<!-- scope: DeepSeek-V3 technical report (arXiv:2412.19437): 671B-total / 37B-active MoE, FP8 pre-training on 14.8T tokens, YaRN 4K→32K→128K extension, SFT with expert-generated R1-style reasoning data, GRPO
     deps: [[grpo]], [[deepseekmath]]
     see-also: [[deepseek-v3-recipe]], [[deepseek-v3.1]], [[deepseek-r1]], [[mixed-precision]], [[sequence-packing]], [[constitutional-ai]]
-->

# DeepSeek-V3 Technical Report
- **Core Insight:** DeepSeek-V3, a Mixture-of-Experts model with 671B total and 37B activated parameters per token, was pre-trained on 14.8T tokens and fully trained (pre-training, context extension, post-training) in 2.788M H800 GPU hours without irrecoverable loss spikes or rollbacks (Abstract, Table 1).
- **Guideline:** When long-CoT reasoning data from a reasoning model is added to SFT data, measure response length together with accuracy, because in the report's DeepSeek-V2.5 ablation the R1-derived data raised MATH-500 from 74.6 to 83.2 and the average response length from 769 to 1510 (§5.4.1, Table 9).
- **Authors:** DeepSeek-AI (byline). App. A lists contributors alphabetically by first name: Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, et al.
- **Year:** 2024 (arXiv v1 2024-12; v2 2025-02)
- **URL:** https://arxiv.org/abs/2412.19437
- **Source type:** official technical report
- **Relevant topics:** MoE, Multi-head Latent Attention, auxiliary-loss-free load balancing, multi-token prediction, FP8 training, pipeline/expert parallelism, YaRN context extension, SFT data construction, reasoning distillation, GRPO, reward models

## Abstract
DeepSeek-V3 is a Mixture-of-Experts (MoE) language model with 671B total parameters, of which 37B are activated for each token. It uses Multi-head Latent Attention (MLA) and the DeepSeekMoE architecture from DeepSeek-V2, adds an auxiliary-loss-free load-balancing strategy, and trains with a multi-token prediction (MTP) objective. It is pre-trained on 14.8T tokens and then trained with Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). The authors report that it outperforms other open-source models and is comparable to leading closed models. Full training costs 2.788M H800 GPU hours, and the authors report no irrecoverable loss spikes and no rollbacks during training.

## Key Contributions
- Auxiliary-loss-free load balancing: a per-expert bias term used only for top-K routing and adjusted after each step; in the ablations it scores higher than a purely auxiliary-loss-based baseline on most benchmarks (§2.1.2, Table 5).
- An MTP objective with sequential modules that keep the causal chain; ablated at two scales and reusable for speculative decoding (§2.2, Table 4, §5.4.3).
- An FP8 mixed-precision training framework validated on a very large model, with fine-grained quantization (§3.3, App. B).
- DualPipe pipeline parallelism with computation-communication overlap, cross-node all-to-all kernels, and memory optimizations that allow training without tensor parallelism (§3.2).
- A post-training pipeline in which domain expert models trained to use R1-style reflection and verification generate SFT data by rejection sampling, followed by GRPO (§5.1, §5.2).

## Key Figures/Tables to Study
- Table 1: training cost by stage (H800 GPU hours and USD at $2 per GPU hour).
- Tables 4 and 5: MTP and auxiliary-loss-free ablations at 15.7B and 228.7B total parameters.
- §4.5.3 and Figure 9: batch-wise vs sequence-wise balance and expert specialization by domain.
- Figure 8: Needle-in-a-Haystack results up to 128K after SFT.
- Table 6: chat-model comparison (8K output limit); Table 9: effect of R1-derived distillation data.

## Technical Details
Recipe values with loci are in [[deepseek-v3-recipe]]. This section summarizes mechanisms.

**Architecture.** 61 Transformer layers, hidden dimension 7168, 128 attention heads with per-head dimension 128, KV compression dimension 512, query compression dimension 1536, decoupled RoPE key/query dimension 64 (§4.2). All FFNs except the first three layers are MoE layers with 1 shared expert and 256 routed experts (expert intermediate dimension 2048); 8 routed experts are activated per token and each token goes to at most 4 nodes (§4.2). Affinity scores use a sigmoid, and gating values are normalized over the selected experts (§2.1.2). No tokens are dropped in training or inference (§2.1.2).

**Load balancing.** A bias b_i is added to the affinity s_{i,t} only when selecting the top-K experts; the gate value still uses s_{i,t} (Eq. 16). After each step, b_i decreases by γ for overloaded experts and increases by γ for underloaded experts (§2.1.2). A sequence-wise balance loss with a small α is kept to prevent extreme imbalance within one sequence (Eq. 17-20).

**Multi-token prediction.** With depth D, module k combines the depth-(k−1) representation of token i with the embedding of token i+k through a projection M_k, a Transformer block, and the shared output head (Eq. 21-23). The MTP loss is λ times the mean cross-entropy across depths (Eq. 25). V3 uses D = 1 (§4.2). The MTP module can be discarded at inference; used for speculative decoding, the second-token acceptance rate is 85-90% and decoding reaches 1.8 times TPS (§5.4.3).

**Infrastructure and FP8.** Training uses a cluster of 2048 H800 GPUs with 8 GPUs per node (§3.1), 16-way pipeline parallelism, 64-way expert parallelism across 8 nodes, and ZeRO-1 data parallelism (§3.2). The three Linear GEMMs (Fprop, Dgrad, Wgrad) run in FP8; the embedding, output head, MoE gating, normalization, and attention operators stay in BF16 or FP32; master weights, weight gradients, and optimizer states are stored in higher precision (§3.3.1). Activations are scaled per 1x128 tile and weights per 128x128 block, and E4M3 is used on all tensors (§3.3.2). Against BF16, the relative loss error stays below 0.25% at about 16B parameters on 1.33T tokens and about 230B parameters on about 0.9T tokens (App. B.1).

**Pre-training data.** Compared with DeepSeek-V2, the corpus increases the ratio of math and programming samples and extends multilingual coverage beyond English and Chinese (§4.1). Documents are packed without cross-sample attention masking (§4.1). Fill-in-Middle in Prefix-Suffix-Middle form is applied at rate 0.1 (§4.1). The tokenizer is byte-level BPE with 128K vocabulary; tokens that combine punctuation and line breaks are randomly split during training to reduce token-boundary bias (§4.1). Mixture percentages are not given.

**Long-context extension.** YaRN is applied only to the decoupled shared key k^R_t, with s = 40, α = 1, β = 32, √t = 0.1 ln s + 1, followed by two 1000-step phases: 32K at batch 1920, then 128K at batch 480, both at learning rate 7.3e-6 (§4.3).

**SFT.** 1.5M instances across domains (§5.1). For math, code-competition, and logic data, an expert model per domain (for example code, math, general reasoning) is trained with SFT and RL on two sample types: <problem, original response> and <system prompt, problem, R1 response>, where the system prompt asks for reflection and verification (§5.1). RL with high-temperature sampling teaches the expert to produce R1 patterns without the system prompt; after "hundreds of RL steps" the experts generate SFT data by rejection sampling (§5.1). Non-reasoning data (creative writing, role-play, simple QA) comes from DeepSeek-V2.5 and is verified by human annotators (§5.1). V3-Base is fine-tuned for 2 epochs with packing and sample masking (§5.1).

**RL.** A rule-based reward model checks answers that can be verified (boxed math answers, LeetCode test cases); a model-based reward model trained from V3 SFT checkpoints handles free-form ground truth and open-ended questions, with chain-of-thought included in its preference data to reduce reward hacking (§5.2.1). GRPO uses group-normalized advantages A_i = (r_i − mean(r)) / std(r) and a KL penalty to a reference model (Eq. 26-28). RL prompts cover coding, math, writing, role-playing, and question answering (§5.2.2). For open-ended questions, V3's own voting judgments serve as feedback in a constitutional-AI setup (§5.3.4, §5.4.2).

## Recipe ledger
Full table (pre-training, decay, long-context, SFT, RL, evaluation settings): [[deepseek-v3-recipe]].

## Findings relevant to generality, negative feedback, long context, distillation
- **Distillation.** On DeepSeek-V2.5, replacing short-CoT data with expert-generated data raised LiveCodeBench-CoT pass@1 from 31.1 to 37.4 (length 718 → 783) and MATH-500 from 74.6 to 83.2 (length 769 → 1510) (Table 9). The authors describe a trade-off between accuracy and response length and say the V3 distillation settings were chosen to balance them (§5.4.1). Their distillation data covers math and code; other domains are future work (§5.4.1).
- **Generality.** The authors state that multi-domain RL prompts align the model with human preferences and improve benchmarks "especially in scenarios where available SFT data are limited" (§5.2.2); no ablation is reported. The limitations section lists more comprehensive evaluation methods as future work, to avoid optimizing for a fixed benchmark set (§6).
- **Negative feedback.** Rule-based rewards are used where possible because they are "resistant to manipulation or exploitation" (§5.2.1). Rejection sampling discards expert outputs that fail quality selection (negative marginal value, §5.1); the report gives no rejection rate.
- **Long context.** After SFT, NIAH is robust up to 128K (Figure 8). The chat model scores 48.7 on LongBench v2 and 73.3 on FRAMES (Table 6).

## Connections
- [[deepseekmath]] and [[grpo]] define the GRPO objective used in §5.2.2.
- [[deepseek-r1]] is the reasoning model whose outputs feed the expert-model data pipeline of §5.1.
- [[deepseek-v3.1]] starts from DeepSeek-V3.1-Terminus, whose base extends V3's context-extension phases.
- [[mixed-precision]] covers the FP8 formats used in §3.3; [[sequence-packing]] covers the packing and sample masking used in SFT.
- [[constitutional-ai]] is the method the report cites for its self-rewarding feedback (§5.4.2).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2412.19437 (v2, 2025-02-18; v1 date read from the v1 PDF)
- Corrections to the previous card version:
  - "post-trained with SFT and RL, then further improved by distilling reasoning behavior" → R1-derived reasoning data enters through the single SFT dataset built by expert models and rejection sampling; there is no separate distillation step after RL (§5.1).
  - "Auxiliary-loss-free load balancing to reduce routing overhead" → its stated aim is to reduce the performance degradation caused by encouraging load balance (§2.1.2).
  - Core Insight and Guideline rewritten as source-supported statements (Abstract, Table 1, Table 9).
- Removed as unsupported by the source: "not from one trick but from joint co-design" (interpretation); "lightweight SFT/RL post-training stack"; "many of the gains are systems gains, not just objective gains".
- Not reported by the source: pre-training mixture percentages; SFT batch size; RL learning rate, KL coefficient β, clip ε, group size G, prompts per step, number of RL steps, maximum response length; reward-model size; context-extension token counts.
