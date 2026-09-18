<!-- scope: Phi-3 Technical Report (Microsoft, arXiv:2404.14219): phi-3-mini/small/medium trained on filtered web + synthetic data, SFT + DPO post-training, safety alignment; v4 adds phi-3.5-mini, phi-3.5-MoE, phi-3.5-Vision
     deps: [[phi-textbooks]], [[phi-1-5]]
     see-also: [[phi-4]], [[orca-2]], [[dpo]], [[longrope2]], [[ruler]]
-->

# Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone
- **Core Insight:** phi-3-mini (3.8B parameters, 3.3T training tokens of heavily filtered web data and synthetic LLM-generated data) scores 68.8 on MMLU 5-shot and 8.38 on MT-bench, compared with 70.5 / not reported for Mixtral 8x7B and 71.4 / 8.35 for GPT-3.5 in the same internal evaluation pipeline (Abstract; §3 benchmark table).
- **Guideline:** When a small model is trained on web data filtered to remove low-value facts, evaluate knowledge-recall benchmarks separately from reasoning benchmarks, because phi-3-mini scores 64.0 on TriviaQA against 82.2 for Mixtral 8x7B and the authors attribute this to limited capacity for factual knowledge, proposing search augmentation (§3 benchmark table; §6, Fig. 6).
- **Authors:** Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, et al. (Microsoft; alphabetical, App. B)
- **Year:** 2024 (arXiv v1 2024-04; v4 2024-08 adds the phi-3.5 models; no venue)
- **URL:** https://arxiv.org/abs/2404.14219
- **Source type:** official technical report
- **Relevant topics:** synthetic pretraining data, web filtering by educational level, "data optimal regime", two-phase pretraining, SFT + DPO, safety post-training, LongRope context extension, small-model MoE, vision-language post-training

## Abstract
The report introduces phi-3-mini, a 3.8B-parameter model trained on 3.3T tokens whose academic-benchmark and internal-test performance the authors describe as rivaling Mixtral 8x7B and GPT-3.5, while being small enough to run on a phone. The training data is a scaled-up version of the phi-2 data: heavily filtered publicly available web data plus synthetic data. The model is further aligned for robustness, safety, and chat format. Parameter-scaling results are given for phi-3-small (7B) and phi-3-medium (14B), both trained for 4.8T tokens (75% and 78% MMLU; 8.7 and 8.9 MT-bench). Version 4 adds phi-3.5-mini, phi-3.5-MoE (16×3.8B, 6.6B active parameters), and phi-3.5-Vision (4.2B) for multilingual, long-context, and multimodal use (Abstract).

## Key Contributions
- A 3.8B model that runs 4-bit quantized in about 1.8GB on an iPhone 14 (A16 Bionic) at more than 12 tokens per second, fully offline (§2, Fig. 2).
- A "data optimal regime" framing: web data is filtered to the "correct level of knowledge" for the model size, keeping pages that may improve reasoning (§2, Fig. 3).
- Scaling results at 7B and 14B on 4.8T tokens, with smaller gains from 7B to 14B than from 3.8B to 7B on some benchmarks (§2).
- Post-training with SFT and DPO, plus safety alignment iterated with an independent Microsoft red team (§2 "Post-training"; §5).
- phi-3.5-MoE and phi-3.5-mini with 128K context via LongRope, and phi-3.5-Vision (§2, §4, §7).

## Key Figures/Tables to Study
- §3 benchmark table (unnumbered): phi-3-mini/small/medium vs Phi-2, Mistral 7B, Gemma 7B, Llama-3-instruct-8B, Mixtral 8x7B, GPT-3.5.
- Fig. 3: log MMLU error vs log model size for phi-1.5, phi-2, phi-3-mini, phi-3-small against Llama-2 7B-70B.
- Tables 1-2: RepoQA and RULER long-context results for phi-3.5-mini and phi-3.5-MoE.
- Table 4 and Fig. 5: internal multi-turn RAI benchmark; red-team harmful-response rates before and after safety alignment.
- Table 7: phi-3.5-Vision with and without safety post-training.

## Technical Details
- **phi-3-mini (§2).** Decoder-only transformer with a Llama-2-like block structure and the Llama-2 tokenizer (vocabulary 32064); hidden size 3072, 32 heads, 32 layers; bfloat16; 3.3T tokens; default context 4K, and a 128K variant (phi-3-mini-128K) via LongRope. Chat template `<|user|>/n Question <|end|>/n <|assistant|>`.
- **phi-3-small (§2).** 7B; tiktoken tokenizer (vocabulary 100352); default context 8192; 32 heads, 32 layers, hidden size 4096; GEGLU activation; hyperparameters tuned on a small proxy with Maximal Update Parametrization (muP) and transferred; grouped-query attention with 4 queries per key; alternating dense and blocksparse attention layers; an additional 10% multilingual data. GEGLU and muP appear in v4, not v1.
- **phi-3-medium (§2).** 14B; same tokenizer and architecture as phi-3-mini; 40 heads, 40 layers, embedding dimension 5120; trained on the same data for "slightly more epochs" (4.8T tokens). v1 labels its numbers a "preview" and notes a HumanEval regression.
- **phi-3.5-MoE (§2).** 16 GLU experts with top-2 routing; 6.6B active of 42B total parameters; router trained with SparseMixer; vocabulary 32064.
- **Pretraining data (§2).** Web data filtered by "educational level" plus synthetic LLM-generated data, in two disjoint sequential phases. Phase 1 is mostly web data for general knowledge and language understanding. Phase 2 combines more heavily filtered web data (a subset of phase 1) with synthetic data for logical reasoning and "niche skills". The report gives no phase token split and does not name the generator models. The later Phi-4 report states that phi-3 phase 1 used most of the tokens and phase 2 was primarily synthetic ([[phi-4]], arXiv:2412.08905 §3.1).
- **Post-training (§2, v4 text).** SFT data covers math, coding, reasoning, conversation, model identity, and safety, and "starts with using English-only examples". DPO data covers chat format, reasoning, and responsible AI (RAI); outputs showing unwanted behavior are used as rejected responses. v1 states only that SFT and DPO were used on generated and curated data.
- **Safety (§5).** Helpfulness/harmlessness preference datasets [BJN+22] and BeaverTails [JLD+23], modified following Safety-Tuned Llamas [BSA+24], plus in-house datasets. Red-team feedback led to additional curated datasets. Table 4 uses GPT-4-simulated multi-turn conversations; phi-3-mini jailbreak DR-1 0.123 vs 0.130 for Llama-3-instruct-8B.
- **Evaluation (§3).** Few-shot prompts at temperature 0 from an internal Microsoft tool, with no prompt tuning for phi-3; footnote 4 reports that adding "##" before the question raised phi-3-mini scores but was not used.
- **phi-3.5-Vision (§7.1).** CLIP ViT-L/14 encoder + phi-3.5-mini decoder; dynamic cropping for high resolution.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| phi-3-mini | 3.8B | pretrain-stable | tokens seen; precision; default context | 3.3T; bfloat16; 4K | arXiv:2404.14219v4 §2 | verified 2026-09-14 | no ablation reported |
| phi-3-small | 7B | pretrain-stable | tokens seen; default context; extra data | 4.8T; 8192; additional 10% multilingual data | v4 Abstract, §2 | verified 2026-09-14 | no ablation reported |
| phi-3-small | 7B | pretrain-stable | hyperparameter selection | muP transfer from a small proxy model | v4 §2 | verified 2026-09-14 | "helped ensure better performance and training stability" (§2), no numbers |
| phi-3-medium | 14B | pretrain-stable | tokens seen | 4.8T (same data as mini, more epochs) | v4 §2 | verified 2026-09-14 | no ablation reported |
| phi-3 family | 3.8B-14B | pretrain-stable, pretrain-decay/anneal | phase token split; LR; batch; schedule; optimizer | not reported | checked v1 and v4 body, App. A-C | not reported | not applicable |
| phi-3-mini-128K | 3.8B | long-context | method; target length; tokens | LongRope; 128K; tokens not reported | v4 §2 | verified 2026-09-14 | no ablation reported |
| phi-3.5-mini, phi-3.5-MoE | 3.8B; 16×3.8B | mid-train, long-context | method; length | more multilingual and long-text data; LongRope + mixed context window; 4K → 128K; tokens not reported | v4 §4 | verified 2026-09-14 | Table 2 RULER; authors claim no loss on 4K tasks (§4), no 4K table |
| phi-3 family | 3.8B-14B | SFT, preference | data size; LR; epochs; DPO β; pair count | not reported | checked v1 and v4 §2, §5 | not reported | not applicable |
| phi-3.5-Vision | 4.2B | pretrain-stable | tokens; max image resolution; loss | 0.5T text+visual tokens; 1344×1344; next-token loss on text tokens only | v4 §7.1 | verified 2026-09-14 | no ablation reported |
| phi-3.5-Vision | 4.2B | SFT, preference | data | multimodal SFT "about a total of 33B tokens" plus text SFT; text DPO plus smaller multimodal DPO; multimodal and text-only tasks trained jointly | v4 §7.1 | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality, negative feedback, long context
- **Generality: knowledge vs reasoning (§3, §6).** phi-3-mini averages 69.7 across the §3 table against 66.8 for Mixtral 8x7B and 72.8 for GPT-3.5, but TriviaQA is 64.0 against 82.2 and 85.8. Search augmentation is shown qualitatively (Fig. 6), with no score.
- **Generality: scaling (§2).** Some benchmarks improve less from 7B to 14B than from 3.8B to 7B; the authors suggest the mixture is not yet "data optimal" for 14B (Interpretation).
- **Generality: languages (§4, §6).** phi-3-mini was "mostly restricted" to English (§6). MMLU-multilingual averages are 47.3 (phi-3-mini), 55.4 (phi-3.5-mini), and 69.9 (phi-3.5-MoE) (§4, Fig. 4).
- **Forgetting control (§7.1).** phi-3.5-Vision SFT and DPO train multimodal and text-only tasks jointly "to maintain language capabilities as much as possible"; no ablation is reported.
- **Negative feedback (§2, §5, §7.3-7.4).** DPO uses unwanted outputs as rejected responses (negative as gradient). Safety post-training raises phi-3.5-Vision VLGuard from 4.66 to 9.10 and RTVLM from 3.56 to 5.44 (Table 7). The authors report that OCR ability from ordinary instruction data leads the vision model to answer some harmful requests, such as captcha decoding, and call this a helpfulness/harmlessness trade-off (§7.4).
- **Long context (§4, Table 2).** On RULER at 128K, phi-3.5-mini scores 63.6 and phi-3.5-MoE 64.2, against 77.0 for Llama-3.1-8B-Instruct; at 4K both are above 94. The authors attribute the 128K drop to a lack of high-quality long-context data in mid-training (Interpretation).

## Connections
- [[phi-textbooks]], [[phi-1-5]] — earlier Phi reports whose data approach phi-3 follows (§2 cites "Textbooks Are All You Need").
- [[phi-4]] — next report; its §3.1 describes the phi-3 phase split and its §3 compares against phi-3-medium.
- [[orca-2]] — separate Microsoft line on post-training small models with teacher-generated data.
- [[dpo]] — the preference-optimization method used in phi-3 post-training.
- [[hh-rlhf]] — [BJN+22], one of the preference datasets used for safety post-training (§5).
- [[longrope2]] — follow-up context-extension method evaluated on Phi3-mini; [[ruler]] — the benchmark in Table 2.
- [[explorer]] — uses Phi-3.5-Vision from this report as a backbone.
- [[lima]] — a separate data-quality study on SFT; [[fineweb]] — filtered-web pretraining data without synthetic data.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2404.14219 (v4, 30 Aug 2024, full PDF) and v1 (22 Apr 2024) for version differences.
- Corrections to the previous card version:
  - "3.8B can reach 7B/13B parity" → the report compares with Mixtral 8x7B and GPT-3.5 (Abstract, §3 table); no 7B/13B parity statement.
  - "Phase 1: majority of tokens; Phase 2: primarily synthetic + small allocation of ultra-filtered web" → the Phi-3 report says phase 2 merges more heavily filtered web data with synthetic data and gives no split (§2); the quoted wording is from the Phi-4 report §3.1.
  - "~2GB at 4-bit" → ≈1.8GB (§2).
  - "~Mixtral 8x7B quality on some benchmarks" → averages 69.7 vs 66.8, MMLU 68.8 vs 70.5 (§3 table).
  - "Safety-focused DPO slice covering harmful content refusal, bias mitigation, privacy" → DPO data covers chat format, reasoning, RAI (§2); safety datasets are HH, BeaverTails, and in-house (§5); listed topics not in source.
  - "Safety evaluation table: DPO's impact on refusal rates" → Fig. 5 shows red-team harmful-response rates before/after safety alignment, not DPO-specific refusal rates.
  - "Authors: Marah Abdin et al." → first six alphabetical authors listed.
  - Added: architecture per size, phi-3.5 models, post-training data domains, evaluation settings, long-context and multilingual results, recipe ledger.
- Removed as unsupported by the source: synthetic data "generated by larger LLMs (GPT-3.5 / GPT-4 class) ... human reviewed"; "two-phase curriculum ... became a template"; "open-weights release"; "two-phase training diagram" (no such figure); "phi-3-medium: 4K context"; SFT data "synthetic + human-written"; "evaluated via ... external benchmarks" for the text models; teacher-model speculation; guideline to "reallocate the data budget heavily toward synthetic data" (no such ablation in the report).
- Not reported by the source: phase token split, synthetic-data generator models, learning rates, batch sizes, SFT/DPO data sizes, DPO β, phi-3.5-MoE training tokens.
