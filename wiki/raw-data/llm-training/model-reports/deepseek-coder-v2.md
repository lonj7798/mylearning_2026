<!-- scope: DeepSeek-Coder-V2 (arXiv:2406.11931): continued pre-training of a DeepSeek-V2 intermediate checkpoint (4.2T tokens) on 6T more tokens mixed 60% code / 10% math / 30% natural language; code and math corpus construction; FIM for the 16B model; YaRN 128K extension; SFT and GRPO with a code reward model
     deps: [[grpo]], [[yarn]]
     see-also: [[deepseek-v3]], [[deepseekmath]], [[deepseek-prover-v2]], [[wizardcoder]], [[oss-instruct]], [[scaling-laws-for-transfer]], [[fsdp-sft]]
-->

# DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence
- **Core Insight:** Continued pre-training of a general MoE checkpoint on 6T tokens that are 60% code, 10% math and 30% natural language (sampled from the parent's own corpus) raises code and math scores to GPT-4-Turbo level (HumanEval 90.2, MATH 75.7) while MMLU moves from 78.1 to 79.2 against DeepSeek-V2 Chat; knowledge-heavy QA drops (TriviaQA 86.7 → 82.3, NaturalQuestions 53.4 → 47.5) (§1, §1.2, Table 10).
- **Guideline:** When specializing a general model by continued pre-training, start from an intermediate checkpoint, keep about 30% of the mix as the parent's natural-language data, and measure knowledge-QA benchmarks separately, because in this report reasoning and code improved while TriviaQA and NaturalQuestions declined (§3.3, Table 10).
- **Authors:** Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang (core contributors), Peiyi Wang, Runxin Xu, Y. Wu, et al.; 39 named authors (DeepSeek-AI)
- **Year:** 2024 (arXiv v1 2024-06-17; only version)
- **URL:** https://arxiv.org/abs/2406.11931 (weights: https://github.com/deepseek-ai/DeepSeek-Coder-V2)
- **Source type:** official technical report
- **Relevant topics:** continued pre-training, code corpus, math corpus, domain mixture, fastText recall, file-level filtering, fill-in-the-middle, long-context extension, code reward model, GRPO

## Abstract
DeepSeek-Coder-V2 is an open MoE code model further pre-trained from an intermediate DeepSeek-V2 checkpoint with 6T additional tokens. The continued pre-training improves coding and mathematical reasoning while keeping general language performance comparable to DeepSeek-V2. Supported programming languages grow from 86 to 338 and context length from 16K to 128K. On standard code and math benchmarks it is reported above GPT-4-Turbo, Claude 3 Opus and Gemini 1.5 Pro.

## Key Contributions
- New code corpus: 1,170B code-related tokens from GitHub and Common Crawl, 338 languages (§2).
- New math corpus: 221B math tokens from Common Crawl, about twice the 120B DeepSeekMath corpus (§1, §2).
- 1B-model corpus ablation against the DeepSeek-Coder corpus (§2, Table 1).
- Two sizes: 16B total / 2.4B active (Lite, FIM enabled) and 236B / 21B (FIM disabled) (§3.3, Table 2).
- RL finding: a reward model trained on compiler signals beats raw 0/1 compiler feedback as the RL reward on in-house LeetCode sets (§3.5.2, Fig. 3).

## Key Figures/Tables to Study
- Table 1: 1B ablation of the new vs old code corpus. Table 2: training settings.
- Table 10: Coder-V2 Instruct vs DeepSeek-V2 Chat on general benchmarks (the forgetting check).
- Fig. 3: compiler signal vs reward-model signal during RL.

## Technical Details
**Mixture (§1, §2).** 60% source code, 10% math, 30% natural language; natural language sampled directly from the DeepSeek-V2 training corpus. Total exposure 10.2T = 4.2T (V2 intermediate checkpoint) + 6T (Coder-V2 data) (§3.3).

**GitHub code (§2).** Public repositories created before November 2023. Filters reused from DeepSeek-Coder: drop files with average line length > 100 characters or max line length > 1000; drop files with < 25% alphabetic characters; drop files with "<?xml version=" in the first 100 characters (except XSLT); HTML kept only if visible text ≥ 20% of the code and ≥ 100 characters; JSON/YAML kept only at 50–5000 characters; then near-deduplication. Output: 821B code tokens (338 languages) + 185B code-related text (markdown, issues).

**Web recall (§2).** DeepSeekMath pipeline: seed corpus (StackOverflow, PyTorch docs, math StackExchange) → fastText classifier (DeepSeek-V2 BPE tokenization for recall on languages without spaces) → domains with > 10% of pages collected are labeled code- or math-related → uncollected URLs under those domains are added to the seed. Three iterations: 70B code tokens and 221B math tokens. Same pipeline on GitHub with a hand-collected seed, two iterations: 94B source code. Total code: 821 + 185 + 70 + 94 = 1,170B.

**Corpus ablation (§2, Table 1).** 1B model, HumanEval (Python) / MBPP: old corpus 1T tokens 30.5 / 44.6%; new corpus 1T 36.0 / 49.0%; new corpus 2T 37.2 / 54.0%.

**Objectives and architecture (§3.1–3.2).** 16B: next-token prediction + FIM at rate 0.5 in PSM mode, applied at document level during packing: `<｜fim_begin｜> f_pre <｜fim_hole｜> f_suf <｜fim_end｜> f_middle <|eos_token|>`. 236B: next-token prediction only. Architecture follows DeepSeek-V2 (16B = V2-Lite, 236B = V2); training instability and gradient spikes were attributed to "exponential normalization" and fixed by reverting to the conventional normalization.

**Optimization (§3.3).** AdamW β1 = 0.9, β2 = 0.95, weight decay 0.1; batch size and LR "adjusted according to DeepSeek-V2 specifications" (values not printed); cosine decay with 2000 warm-up steps down to 10% of the initial LR.

**Long context (§3.4).** YaRN s = 40, α = 1, β = 32; stage 1: 32K sequence, batch 1152, 1000 steps; stage 2: 128K sequence, batch 288, 1000 steps; long-context data upsampled (ratio not printed). NIAH passes up to 128K (Fig. 2).

**SFT (§3.5.1).** 20k code and 30k math instructions from DeepSeek-Coder and DeepSeek-Math, plus sampled DeepSeek-V2 instruction data; 300M tokens total; cosine schedule, 100 warm-up steps, LR 5e-6, batch 1M tokens, 1B tokens trained.

**RL (§3.5.2).** About 40k code and math prompts after filtering; each code prompt has test cases. Math preference data from ground-truth labels; code: a reward model trained on compiler pass/fail data, because limited test cases make raw 0/1 feedback noisy. GRPO.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-Coder-V2 (both) | 236B/21B; 16B/2.4B | continued-pretrain | start checkpoint | DeepSeek-V2 intermediate checkpoint at 4.2T tokens | arXiv:2406.11931v1 §3.3 | verified 2026-11-24 | no ablation reported |
| both | same | continued-pretrain | mixture; tokens | 60/10/30 code/math/NL; 6T | §1, §2 | verified 2026-11-24 | no mixture ablation reported |
| both | same | continued-pretrain | optimizer; schedule | AdamW (0.9, 0.95), wd 0.1; cosine, 2000 warm-up, floor 10% | §3.3 | verified 2026-11-24 | — |
| both | same | continued-pretrain | peak LR; batch | not printed ("according to DeepSeek-V2") | §3.3 | not reported | — |
| Coder-V2-Lite | 16B/2.4B | continued-pretrain | FIM rate; mode | 0.5; PSM, document level | §3.1 | verified 2026-11-24 | Table 6: FIM mean 86.4% |
| both | same | long-context | YaRN; stages | s=40, α=1, β=32; 32K×1152×1000 steps, 128K×288×1000 steps | §3.4 | verified 2026-11-24 | Fig. 2 NIAH |
| Coder-V2-Instruct | same | SFT | data; LR; batch; tokens | 300M-token mix; 5e-6 cosine, 100 warm-up; 1M tokens; 1B tokens | §3.5.1 | verified 2026-11-24 | — |
| Coder-V2-Instruct | same | RL | prompts; reward; algorithm | ~40k; RM trained on compiler data; GRPO | §3.5.2 | verified 2026-11-24 | Fig. 3: RM signal > compiler signal on LeetCode / LeetCode-zh (values only in figure) |

## Findings relevant to generality and negative feedback
- **Generality / forgetting (Table 10).** 236B Instruct vs V2 Chat: BBH 83.9 vs 79.7, MMLU 79.2 vs 78.1, Arena-Hard 65.0 vs 41.6; TriviaQA 82.3 vs 86.7, NaturalQuestions 47.5 vs 53.4, AlpacaEval 2.0 36.92 vs 38.90, MT-Bench 8.77 vs 8.97. The authors attribute the knowledge drop (Lite) to less web data in pre-training and the open-ended gap to V2 Chat's general-purpose alignment stage (§4.6).
- **Limitation.** The authors report a significant instruction-following gap to GPT-4-Turbo that hurts complex tasks such as SWE-bench (§5).
- **Negative feedback.** Failed test cases are converted into reward-model training data instead of being used directly as a 0/1 reward (§3.5.2).

## Connections
- [[deepseekmath]] / [[grpo]]: the Common Crawl recall pipeline and the RL algorithm.
- [[yarn]]: context-extension method and hyperparameters.
- [[deepseek-v3]]: later report from the same lab.
- [[deepseek-prover-v2]]: later DeepSeek specialist built on a general base.

## Verification
- Checked on 2026-11-24 against: arXiv:2406.11931v1 PDF (body, Tables 1–10).
- Internal consistency: the introduction states HumanEval/MBPP gains of 6.7 and 9.4 points (1B, 2T tokens), §2 states 5.5 and 4.4 points (1T tokens); both match Table 1.
- Not reported by the source: peak LR and batch size of continued pre-training; long-context data ratio; RL hyperparameters and reward-model size; per-language token counts (list in App. A only).
