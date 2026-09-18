<!-- scope: long-context continual pretraining — how much data and which mixture extends LLaMA-2 to 128K
     deps: [[longalign]]
     see-also: [[prolong]], [[long-context-llama3]], [[ruler]], [[nextlong]], [[pose-synthesis]]
-->

# Data Engineering for Scaling Language Models to 128K Context
- **Core Insight:** Continual pretraining of LLaMA-2 7B on 5B tokens of per-source length-upsampled SlimPajama, packed to 80K, reaches 88.0 accuracy on Needle-in-a-Haystack at 128K against GPT-4-Turbo 128K's 87.1, while MMLU is 43.3 against GPT-4-Turbo's 86.4 (§5.1, Table 3).
- **Guideline:** When extending a 4K-context base model to 128K, upsample documents longer than 4K *within each source* so the domain mixture is unchanged, and train on 1–5B tokens, because per-source upsampling is the only mixture in Table 5 that does not raise short-context loss on the web domains by more than 0.01 (§5.3). Otherwise, when a single domain such as books or code is upsampled globally, expect in-domain gains that do not transfer and can raise loss in other domains (§5.3, Table 5).
- **Authors:** Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, Hao Peng (University of Edinburgh; MIT-IBM Watson AI Lab; University of Melbourne; Ohio State University; University of Washington; MIT; UIUC)
- **Year:** 2024 (arXiv v1 2024-02-15)
- **URL:** https://arxiv.org/abs/2402.10171
- **Source type:** paper
- **Relevant topics:** long-context continual pretraining, length upsampling, domain balance, Needle-in-a-Haystack, validation-loss limits

## Abstract
The paper studies the continual-pretraining recipe for extending context length to 128K, with the focus on data rather than architecture. Its hypothesis is that the ability to use information at arbitrary input positions is mostly acquired during large-scale pretraining and can be extended to much longer contexts than seen in training through lightweight continual pretraining on an appropriate mixture (Abstract). For quantity, 500 million to 5 billion tokens are enough for the model to retrieve information anywhere in a 128K context. For quality, the paper emphasizes domain balance and length upsampling jointly: upsampling long data from particular domains such as books, a common practice in prior work, gives worse results than keeping a balanced domain mixture. The recipe outperforms open-source long-context models and closes the gap to GPT-4 128K on retrieval (Abstract).

## Key Contributions
- A data-quantity study showing retrieval accuracy at 128K rising with tokens seen and then falling again at 10B (§5.2, Fig. 3).
- A data-mixture taxonomy — cut at 4K, cut at 128K, per-source upsampling, global upsampling, upsample Arxiv/Book/Github — and a per-domain loss comparison of all five against the original mixture (§3, §5.3, Table 5).
- Evidence that validation loss is an inadequate evaluation: two mixtures with nearly identical per-length loss differ sharply on Needle-in-a-Haystack (§5.3, Fig. 4).
- An affordable configuration: continual pretraining of a 7B model at 80K context on 8 × 80G A100 in about 5 days (§5.2, Table 2).
- Open recipe and weights at https://github.com/FranxYao/Long-Context-Data-Engineering.

## Key Figures/Tables to Study
- Figure 2 (§3) — length and domain distributions produced by each of the five mixture strategies.
- Figure 3 (§5.2) — Needle-in-a-Haystack accuracy and per-length loss at 100M, 300M, 500M, 1B, 5B, and 10B tokens.
- Figure 4 (§5.3) — original mixture against per-source upsampling at nearly equal loss.
- Table 3 (§5.1) — Needle and MMLU for the released models and the baselines.
- Table 5 (§5.3) — per-domain validation-loss differences for 0–4K and 4K–128K context.

## Technical Details
- **Base models:** LLaMA-2 7B and 13B; no architecture change other than adjusting the base of RoPE (Abstract, §4).
- **Corpus:** SlimPajama, with the mixture kept at 67% CommonCrawl, 15% C4, 4.5% Github, 4.5% Wikipedia, 4.5% books, 2.5% Arxiv, 2.0% StackExchange (§3, §5.3).
- **Per-source upsampling:** within each domain, sequences longer than 4K are upsampled from about 30% to about 70% of the data, leaving the domain ratios fixed (§5.3). About 30% of documents in the original mixture are naturally longer than 4K (§3).
- **Training:** constant learning rate 2e-5; batch size 4M tokens; all data packed to 80K chunks regardless of document boundary; 80K sequence length for the 7B model and 64K for the 13B model (§4).
- **Token budget:** 5B tokens for the main runs, which the paper notes is 5B / 4M ≈ 1,250 optimizer steps (§4, §5).
- **Compute:** training at 80K is about 3× slower per token than at 4K; continual pretraining of the 7B takes about 5 days on 8 × 80G A100 (§4, §5.2). Table 2 lists 10 days per 10B tokens at 80K on 8 × 80G A100 for the 7B, and 13 days per 10B tokens at 64K for the 13B.
- **Data-quantity curve (§5.2, Fig. 3):** Needle accuracy 37.8 at 100M tokens, 59.0 at 300M, 81.1 at 500M, 85.3 at 1B, 88.0 at 5B, 84.0 at 10B. The paper states that 500M tokens unlock most of the retrieval accuracy and that length generalization degrades at 10B, which it attributes to overfitting to the 80K training range.
- **Main results (§5.1, Table 3):** Ours LLaMA-2 7B 80K — Needle 88.0, MMLU 43.3. Ours LLaMA-2 13B 64K — 90.0, 52.4. GPT-4-Turbo 128K — 87.1, 86.4. Together LLaMA-2 7B 32K — 27.9, 44.8. LongChat v1.5 7B 32K — 18.0, 42.3. LongLoRA 7B 100K — 70.0, 37.9. LongLoRA 13B 64K — 54.1, 50.1.
- **Downstream long-context task (§5.1, Table 4):** InfiniBench BookQA at 128K test length — GPT-4-Turbo 37.4, ours 13B 31.1, ours 7B 27.4, YaRN Mistral 7B 26.3, LongLoRA 13B 24.6, LongLoRA 7B 24.3.
- **Mixture ablation (§5.3, Table 5):** loss differences against the original mixture, 7B, 5B tokens, packed to 80K. At 0–4K context, per-source upsampling is the only strategy that does not raise loss above the +0.01 significance threshold on C4, CommonCrawl, and StackExchange. Upsampling Book raises Github loss by +0.029 and StackExchange by +0.021 at 0–4K; upsampling Code raises Book loss by +0.030. At 4K–128K, per-source upsampling lowers loss on six of seven domains.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| yaofu/llama-2-7b-80k | 7B | long-context CPT | base model | LLaMA-2 7B | arXiv:2402.10171v1 §1 | verified (2026-09-18) | — |
| yaofu/llama-2-7b-80k | 7B | long-context CPT | tokens seen | 5B | §5 | verified (2026-09-18) | §5.2 Fig. 3: 500M → 81.1 Needle, 5B → 88.0, 10B → 84.0 |
| yaofu/llama-2-7b-80k | 7B | long-context CPT | training sequence length / packing | 80K, packed regardless of document boundary | §4 | verified (2026-09-18) | §5.1 Table 3: generalizes from 80K training to 128K evaluation |
| yaofu/llama-2-7b-80k | 7B | long-context CPT | learning rate | constant 2e-5 | §4 | verified (2026-09-18) | no ablation reported |
| yaofu/llama-2-7b-80k | 7B | long-context CPT | global batch | 4M tokens | §4 | verified (2026-09-18) | no ablation reported |
| yaofu/llama-2-7b-80k | 7B | long-context CPT | mixture | SlimPajama ratios held fixed; within-source upsampling of >4K documents from ~30% to ~70% | §5.3 | verified (2026-09-18) | §5.3 Table 5: only mixture with no significant 0–4K regression on web domains |
| yaofu/llama-2-7b-80k | 7B | long-context CPT | RoPE base | adjusted, following Xiong et al. (2023); no value printed | §4 | not reported (checked Abstract, §4, §5, Table 2; the paper has no appendix) | — |
| yaofu/llama-2-7b-80k | 7B | long-context CPT | compute | 8 × 80G A100, about 5 days | §5.2, Table 2 | verified (2026-09-18) | — |
| Ours LLaMA-2 13B 64K | 13B | long-context CPT | training sequence length | 64K | §4 | verified (2026-09-18) | set by memory limits of the HuggingFace-DeepSpeed stack (§4) |
| — | — | long-context CPT | optimizer betas, weight decay, grad clip, warmup, seeds | — | — | not reported (checked §4, Table 2, and the body) | — |

## Findings relevant to long context
- The paper's central negative result about measurement: the original mixture and the per-source upsampled mixture give per-length validation losses that differ by at most 0.01–0.02, yet differ sharply on Needle-in-a-Haystack (§5.3, Fig. 4). The authors conclude that validation loss, the evaluation used in most prior long-context work, can conceal the difference.
- Short-context capability is measured only by MMLU in this paper; the 7B 80K model scores 43.3 (§5.1, Table 3). The paper describes this as maintaining short-context performance and does not report the base LLaMA-2 7B MMLU for comparison.
- Cross-domain transfer is limited and can be negative: improvements from upsampling one domain do not reach others, and book and code are given as a pair where each harms the other (§5, §5.3).
- Scope limits stated by the authors: the work covers continual pretraining only, with no instruction tuning; they note there was no open-source instruction-finetuned 100K-context model at the time and propose long-context SFT as future work (§5.2, §6).
- Beyond retrieval, the evaluation is one downstream benchmark (InfiniBench BookQA); other InfiniBench tasks were dropped because base models did not follow the instructions (§5.1).

## Connections
- [[prolong]] — reproduces this recipe head-to-head at equal initialization, 5B tokens, and hyperparameters, then improves on it.
- [[long-context-llama3]] — Meta's own long-context recipe; both use synthetic or reweighted long data, but Llama 3 does it in SFT as well as pretraining.
- [[nextlong]], [[longrope-data]], [[longrope2]], [[controlled-long-context-extension]] — later work that cites this paper's mixture as its baseline.
- [[ruler]] — the benchmark that replaces single-needle retrieval with diverse synthetic long-context tasks.
- [[pose-synthesis]] — the alternative that avoids training at full target length.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2402.10171 (arXiv v1, 15 Feb 2024)
- Corrections to the previous card version:
  - "documents longer than 32K get 5× weight" → sequences longer than 4K are upsampled from about 30% to about 70% within each source (§5.3).
  - "RoPE base-θ rescaled from 10K to 200M (NTK-aware)" → the paper says only that it modifies the base of RoPE following Xiong et al. (2023) and prints no value (§4).
  - "LR: 2e-5 → 2e-6 cosine" → constant learning rate 2e-5 (§4).
  - "~30K A100-hours for 7B model" → about 5 days on 8 × 80G A100 (§5.2).
  - "NIAH @ 128K ~98%; MMLU within 1 point of base" → Needle 88.0 and MMLU 43.3 for the 7B 80K model (§5.1, Table 3); no base-model MMLU is reported.
  - "At release, matched GPT-4-128K on NIAH" → 88.0 against GPT-4-Turbo's 87.1 on Needle, but 43.3 against 86.4 on MMLU and 27.4 against 37.4 on BookQA (Tables 3, 4).
  - "if cross-domain proportions are changed … short-context MMLU drops 3–5 points" → Table 5 reports per-domain validation-loss differences, not MMLU; the largest 0–4K regressions are +0.030 on Book from upsampling code and +0.029 on Github from upsampling books.
  - "documents concatenated with document-separator tokens; no cross-document attention masking" → the paper states that all data is packed to 80K chunks regardless of document boundary and says nothing about separators or masking (§4).
  - "U Edinburgh + MIT-IBM + UIUC + UW + AI2" → Edinburgh, MIT-IBM Watson AI Lab, Melbourne, Ohio State, Washington, MIT, UIUC; no AI2 affiliation appears (title page).
  - "5B tokens … is sufficient" stated alone → the paper's quantity claim is 500M to 5B, with 500M unlocking most retrieval accuracy and 10B degrading length generalization (Abstract, §5.2).
- Removed as unsupported by the source: "5× weight"; "200M RoPE base"; "NTK-aware"; "no YaRN"; "cosine LR to 2e-6"; "30K A100-hours"; "NIAH ~98%"; "MMLU within 1 point of base"; "pressure-tested multi-needle variant"; "3–5 MMLU point drop"; "document-separator tokens"; "Yi-200K and Qwen-long-context adopted this baseline"; "length-upsample factor is empirical and may need retuning per base model"; "superseded on quality by ProLong".
- Not reported by the source: a RoPE base value (the released `yaofu/llama-2-7b-80k` config is a separate artifact and must be cited as such); optimizer settings beyond the learning rate; number of seeds; results on RULER, LongBench, or HELMET; any SFT stage.
