<!-- scope: LongPO (arXiv 2502.13922, ICLR 2025): self-generated short-to-long preference pairs + KL constraint to a short-chunk reference for long-context alignment without external labels; iterative 128K→512K
     deps: [[dpo]], [[self-instruct]]
     see-also: [[longred]], [[longalign]], [[rpo]], [[longer-context-deeper-thinking]]
-->

# LongPO: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization
- **Core Insight:** Training Mistral-7B-Instruct-v0.2 on 45K self-generated short-to-long preference samples raises its InfiniteBench average from 13.82 to 39.27 (GPT-4-128K: 34.81) while MMLU moves from 59.15 to 59.99; SFT and DPO on the same data reach InfiniteBench 30.03 and 25.56 (Tables 1, 3).
- **Guideline:** When an aligned short-context model must be extended to long context without human or external-teacher labels, train on pairs where the chosen response is the model's own answer given only the relevant chunk and the rejected response is its answer given the full document, with the reference distribution conditioned on the chunk and an NLL term on the chosen long sequence, because on Mistral-7B this kept every short-context metric within 0.74 points of the base or above it (Table 3), while DPO on the same pairs degraded short-context scores (§5.1, Fig. 3). This was tested only on 7B instruct models with self-instructed questions over books, arXiv papers, and GitHub code.
- **Authors:** Guanzheng Chen, Xin Li, Michael Qizhe Shieh, Lidong Bing (National University of Singapore; DAMO Academy, Alibaba Group; Hupan Lab; Shanda AI Research Institute)
- **Year:** 2025 (arXiv v1 2025-02; ICLR 2025)
- **URL:** https://arxiv.org/abs/2502.13922
- **Source type:** paper
- **Relevant topics:** long-context alignment, preference optimization, self-generated preference data, short-context retention, KL constraint, iterative context extension

## Abstract
Short-context LLMs can underperform on long inputs because long-context alignment is insufficient. Human annotation of long contexts is impractical, and it is hard to keep short-context performance while adding long-context ability. LongPO lets a short-context model improve on long-context tasks by learning from self-generated preference pairs: two responses to the same instruction, one given the long input and one given a compressed short-context counterpart. A short-to-long KL constraint limits the loss of short-context performance. Applied to Mistral-7B-Instruct-v0.2 from 128K to 512K, LongPO retains short-context performance, outperforms SFT and DPO on both long- and short-context tasks, and reaches long-context results comparable to or above GPT-4-128K. Code: github.com/DAMO-NLP-SG/LongPO.

## Key Contributions
- Short-to-long preference: chosen = response to the short chunk, rejected = response to the full document, both from the same model (§3.1, Eq. 6-7).
- Short-to-long KL constraint that replaces the long-input reference πref(y|xL) with πS(y|xS) (§3.2, Eq. 10-12).
- Data construction by reverse instruction generation from chunks, a multi-turn objective, and an added NLL term (§3.3, Eq. 13-14).
- Iterative extension 128K → 256K → 512K (§4.1, Table 1).
- Comparisons with SFT and DPO on identical data and ablations of the constraint and NLL term (§5.1, §5.3, Fig. 4).

## Key Figures/Tables to Study
- Fig. 2: seven-step data construction. Table 1: InfiniteBench, RULER, LongBench-Chat for LongPO, SFT, DPO, and 128K/1M baselines.
- Table 2: RULER per length (4K-128K) for 13 tasks. Table 3: MMLU, ARC-C, HellaSwag, Winogrande, MT-Bench.
- Fig. 3: short-context margins vs base. Fig. 4: ablation training curves. Fig. 6: chosen and rejected rewards during training.

## Technical Details
- **Objective.** L_LongPO = −E[log σ(β log πθ(yS|xL)/πS(yS|xS) − β log πθ(yL|xL)/πS(yL|xS))] (Eq. 12). πθ: policy, initialized from πS. πS: the aligned short-context model. xL = [CL; IL]: long document and instruction. xS = [CS; IL]: the chunk and the same instruction. yS ~ πS(y|xS): chosen. yL ~ πS(y|xL): rejected. β: margin (0.1). σ: sigmoid.
- **Why the reference changes.** In DPO the implicit constraint βD_KL[πθ(y|xL) || πref(y|xL)] uses a reference that is weak on long inputs. LongPO uses C′ = βD_KL[πθ(y|xL) || πS(y|xS)], which keeps the long-input policy close to the short-context model given the needed information (§3.2, Eq. 8-10; derivation App. A.1).
- **Multi-turn and NLL.** Instruction-response triples from up to n chunks of one document form one multi-turn sample; per-turn probabilities are summed inside each log ratio (Eq. 13). Final loss: λ·L_MT_LongPO + NLL over S_L = [xL; {IL_i; yS_i}], normalized by |S_L| (Eq. 14). The NLL term follows Pang et al. (2024b) and stabilizes training without prior continual pretraining (§3.3).
- **Data construction.** Corpus: Book and ArXiv subsets of Long-Data-Collections and the GitHub subset of RedPajama. For a target length, keep documents longer than 64K tokens and shorter than the target. Split into chunks of up to 32K tokens; keep at most 4 random chunks per document. Generate an instruction pool per chunk with Self-Instruct and sample one instruction ("4 instructions per document", §4.1). Filter censored and repetitive responses (§3.3, §4.1).
- **Decoding.** Instructions: temperature 0.7, top-p 0.9. Responses: greedy decoding, with the short or long context concatenated to the instruction (App. B.1).
- **Iteration.** Mistral-7B-LongPO-128K is trained on data generated by Mistral-7B at 128K; the 128K model then generates 256K and 512K data (§4.1). Table 1 labels the results iter1 (128K), iter2 (256K), iter3 (512K).
- **Long-context results (Table 1).** InfiniteBench avg / RULER avg (NIAH, VT, QA; aggregation excluded) / LongBench-Chat (EN): Mistral-7B 13.82 / 66.4 / 4.10; SFT 30.03 / 73.15 / 4.25; DPO 25.56 / 65.62 / 4.08; LongPO iter1 39.27 / 86.06 / 5.42; iter2 39.65 / 86.22 / 5.48; iter3 41.21 / 86.56 / 5.80. Qwen2.5-7B 27.12 / 72.16 / 5.80 → LongPO 40.48 / 81.64 / 5.75. GPT-4-128K 34.81 / 88.53 / 8.40; LLaMA 3.1-8B 38.87 / 84.68 / 6.22; GLM-4-9B-1M 35.53 / 89.0 / 5.03.
- **RULER by length (Table 2, 13-task average).** At 128K: Qwen2.5-7B-Instruct 24.47 vs Qwen2.5-7B-LongPO-128K 75.43; Mistral-7B-LongPO-128K 78.97, -512K 80.97.
- **Short-context results (Table 3).** Mistral-7B-Instruct-v0.2 MMLU / ARC-C / HellaSwag / Winogrande / MT-Bench: 59.15 / 59.26 / 83.2 / 78.4 / 6.34; LongPO-128K 59.99 / 59.34 / 82.99 / 78.53 / 6.35; LongPO-512K 59.51 / 60.58 / 82.87 / 77.66 / 6.34. Qwen2.5-7B-Instruct 74.28 / 67.15 / 81.41 / 74.66 / 7.30 → LongPO-128K 73.64 / 65.70 / 80.82 / 74.98 / 7.62.
- **Evaluation settings.** lm-evaluation-harness with 5-shot MMLU, 25-shot ARC-C, 10-shot HellaSwag, 5-shot Winogrande; GPT-4-Turbo-1106-Preview judges MT-Bench and LongBench-Chat (App. B.2).
- **Citation inconsistency.** §1 attributes the LLaMA-3.1 "0.1% long-context data" alignment share to Liu et al. (2024b); §6 attributes it to Dubey et al. (2024).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Mistral-7B-LongPO; Qwen2.5-7B-LongPO | 7B | preference (long-context) | Optimizer; LR | Adam; 5e-7 | arXiv:2502.13922v3 §4.1 | verified 2026-09-14 | no ablation reported |
| same | 7B | preference (long-context) | β (Eq. 13); λ (Eq. 14, weight on preference term; NLL weight 1) | 0.1; 0.01 | §4.1, Eq. 14 | verified 2026-09-14 | Fig. 4: removing NLL slows long-context convergence (curves only) |
| same | 7B | preference (long-context) | RoPE θ; batch size (unit not stated) | 1e7; 8 | §4.1 | verified 2026-09-14 | no ablation reported |
| Mistral-7B-LongPO-128K | 7B | preference (long-context) | Samples; document length | 45K multi-turn samples; >64K and <128K tokens | §4.1 | verified 2026-09-14 | Table 1 |
| Mistral-7B-LongPO-256K | 7B | preference (long-context) | Samples | 16K multi-turn samples of 256K tokens | §4.1 | verified 2026-09-14 | Table 1 |
| Mistral-7B-LongPO-512K | 7B | preference (long-context) | Samples | 2.5K multi-turn samples of 512K tokens | §4.1 | verified 2026-09-14 | Table 1 |
| Qwen2.5-7B-LongPO-128K | 7B | preference (long-context) | Samples | 32K samples of 128K tokens | §4.1 | verified 2026-09-14 | Table 1 |
| all LongPO runs | 7B | preference (long-context) | Chunk size; chunks per document | ≤32K tokens; ≤4 | §4.1 | verified 2026-09-14 | no ablation reported |
| all LongPO runs | 7B | preference (long-context) | Instruction decoding; response decoding | T 0.7, top-p 0.9; greedy | App. B.1 | verified 2026-09-14 | no ablation reported |
| 128K runs | 7B | preference (long-context) | Hardware; throughput | 8×A800 80GB, DeepSpeed-Ulysses, FlashAttention; 4,401 tokens/s | §4.1; App. B.3 | verified 2026-09-14 | not applicable |
| 256K / 512K runs | 7B | preference (long-context) | Hardware; throughput | 16×A800 80GB; 4,120 / 2,744 tokens/s | App. B.3 | verified 2026-09-14 | not applicable |
| SFT baseline (Mistral-7B) | 7B | SFT | Data; LR | chosen responses with long inputs; 2e-5, other settings as LongPO | App. B.3 | verified 2026-09-14 | not applicable |
| DPO baseline (Mistral-7B) | 7B | preference | Settings | as LongPO without the short-to-long constraint | App. B.3 | verified 2026-09-14 | not applicable |

Not reported (checked body, App. A-B): epochs or training steps, total training tokens, warmup and LR schedule, Adam betas, weight decay, whether the DPO baseline keeps the NLL term, number of seeds.

## Findings relevant to generality, negative feedback, long context, distillation
- **Generality.** The authors report that SFT and DPO on the same data degrade short-context tasks by 10-20 points on most tasks, while LongPO does not (§1, Fig. 3). For LongPO on Mistral, MMLU and ARC-C rise and the largest drop is Winogrande −0.74 at 512K; on Qwen2.5, ARC-C falls 1.45 and MMLU 0.64 (Table 3). SFT on chosen responses with the short-to-long constraint also keeps short-context performance (§5.3, Fig. 4), which the authors attribute to the constraint (Interpretation).
- **Negative feedback.** Source of negatives: the model's own greedy answer to the full document; the label comes from construction only, with no verifier or judge. The paper assumes yL "is likely to be of lower quality" (§3.1) and reports no measured error rate for this labeling. Use: negative as gradient through the rejected term, anchored by an NLL term on chosen responses. Evidence: LongPO exceeds SFT-Chosen and SFT-Rejected on RULER-NIAH across training (§5.3, Fig. 4, curves only). Diagnostics: chosen and rejected rewards logged during training (App. B.1, Fig. 6). The share of the gain due to the rejected term alone is not measured.
- **Long context.** Training reaches 512K, while RULER results cover 4K-128K (Table 2) and InfiniteBench inputs exceed 100K (§4.2). The authors attribute GPT-4-128K's lower InfiniteBench score to scarce long-context training data (Interpretation, §5.2).
- **Distillation.** Self-distillation across context lengths. Stage: long-context alignment after instruction tuning. Prompts: self-instructed questions over book, arXiv, and GitHub chunks. Teacher: the same model given the short chunk, greedy decoding. Quality control: removal of censored and repetitive responses (§4.1, App. B.1).

## Connections
- [[dpo]] — base objective; LongPO changes the reference to the short-chunk distribution.
- [[self-instruct]] — method used to generate the instruction pool from chunks.
- [[rpo]] — Pang et al. (2024b), the source of the added NLL term.
- [[longalign]] — source of LongBench-Chat (Bai et al., 2024).
- [[ruler]], [[infinitebench]] — long-context benchmarks in Table 1.
- [[deepspeed-ulysses]] — sequence parallelism used for 128K-512K training.
- [[yarn]], [[glm-4]], [[llama-3]] — YaRN-Mistral-7B-128k, GLM-4-9B, and LLaMA-3.1 baselines.
- [[longred]] — retention of short-text ability at the continued-pretraining stage by distillation from the original model.
- [[longer-context-deeper-thinking]] — reports effects of long-context ability on reasoning SFT.
- [[context-synthesis-short-to-long]], [[longmagpie]], [[sealong]] — other self-generated or synthesized long-context post-training data (not compared in this paper).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2502.13922 (arXiv v3, 1 Mar 2025; full PDF including App. A-B; abs page for version dates and "ICLR 2025" comment).
- Audit claims not found in the source: "self-instructs 4 instructions per chunk" → §4.1 says 4 instructions per document, one randomly selected. "Long-context scores match or beat GPT-4-128K" holds for the InfiniteBench average (39.27 vs 34.81) but not for RULER (86.06 vs 88.53) or LongBench-Chat (5.42 vs 8.40) (Table 1). Affiliation "(NUS, Alibaba DAMO)" omits Hupan Lab and Shanda AI Research Institute. The exact short-context drop of SFT and DPO is given only as "10∼20 points on most tasks" (§1) and as bar margins in Fig. 3.
