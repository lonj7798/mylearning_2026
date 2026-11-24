<!-- scope: LitBench (Stanford): creative-writing reward-model benchmark from r/WritingPrompts upvotes; 2,480 debiased test pairs + 43,827 training pairs; upvote-gap, timestamp, and length-symmetry filtering; zero-shot judges vs Bradley-Terry RMs vs GenRMs with and without CoT; online human validation of RM picks
     deps: [[bradley-terry-rm]], [[generative-reward-models]]
     see-also: [[judge-llm-bias]], [[writing-zero]], [[skywork-reward-v2]], [[reward-model-overoptimization]], [[subjective-reward-guide]]
-->

# LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing
- **Core Insight:** On debiased human preference pairs for short stories, small trained verifiers beat the best zero-shot LLM judge: a Llama-8B Bradley-Terry RM and a Qwen GenRM both reach about 78% agreement versus 73% for Claude-3.7-Sonnet, and distilled chain-of-thought lowered GenRM accuracy to 72% (Abstract; §5, Fig. 4).
- **Guideline:** When building preference pairs from naturally occurring signals (votes, ratings), filter for a minimum signal gap, control exposure time, and rebalance pairs so the chosen item is longer in exactly half of them, because 65.25% of raw chosen stories were longer (§3.2), and an RM trained without length rebalancing saturated at 70% and "strongly prefer[red] the longer of the two stories in most cases" (§5, Fig. 7).
- **Authors:** Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, Nick Haber (Stanford University)
- **Year:** 2025 (arXiv v1 2025-07-01)
- **URL:** https://arxiv.org/abs/2507.00769
- **Source type:** paper (preprint, under review)
- **Relevant topics:** creative-writing reward models, human preference from upvotes, length bias, position bias, CoT in judges, judge-vs-trained-RM comparison, human validation of RM selections

## Summary
Stories come from r/WritingPrompts. Test pairs use posts after 2023-01-02; training pairs come from the pre-2023 euclaise/WritingPrompts_preferences dataset (§3.1, §3.3). Each story needs ≥10 upvotes, ≤2,048 tokens, ≥50 words; a pair needs an upvote difference ≥25%, and the higher-voted story must be posted later (§3.2). Length bias is removed with a 100-bucket histogram of length differences pruned to symmetry (§3.2, Fig. 2). Test set: 2,480 pairs over 3,543 stories, mean 550 words; training: 43,827 pairs over 50,309 stories (§3.3).

## Key Contributions
- First standardized pairwise benchmark for creative-writing verifiers with a matching training set (§1).
- Comparison of zero-shot judges, BT RMs, GenRM (single A/B token), and GenRM-CoT (GPT-4.1 rationales) (§4).
- Data-filter ablation showing that more but less-filtered pairs train worse RMs (§5, Fig. 7).
- Online human study of RM-selected best vs worst generated stories (§5).

## Key Figures/Tables to Study
- Fig. 2 (length-bias pruning), Fig. 4 (verifier accuracy), Fig. 5 (explanation features vs accuracy), Fig. 6 (accuracy vs model size), Fig. 7 (filter ablation), Fig. 8 (human study).

## Technical Details
**Judges and RMs (§5, Fig. 4).** Best BT RM (Llama-8B) 78%; best GenRM (Qwen-7B) marginally lower; Claude-3.7-Sonnet zero-shot 73%; GenRM-CoT 72%; OpenAI/Anthropic/DeepSeek frontier judges "in the 70% range"; small open models 56–60%. Zero-shot judges were asked to explain before the verdict and were scored as the average over both orderings to handle position bias (§4).

**Explanation features (Fig. 5).** Discussing plot in the judge's explanation was associated with +14.8% higher correctness across models; most other explanation features had minimal relation to accuracy.

**Size scaling (Fig. 6).** GenRM-CoT improves with size to 74%; GenRM without CoT shows no significant gain with size, so a 1B–1.5B GenRM gives similar accuracy.

**Filter ablation (Fig. 7, Llama-3.2-1B BT RMs).** Lightly filtered set (395K pairs, no timestamp pairing): saturates at 65%. Timestamp + upvote pairing without length filtering (1.03M pairs): saturates at 70% and is length-biased. LitBench training set: higher.

**Human study (§5, Fig. 8).** GPT-4.1 and GPT-4o wrote 64 stories for each of 40 prompts; the BT RM picked best and worst; 46 US/UK crowd-workers (10–13 per pair) chose the RM-preferred story 57% of the time vs 41% for the RM-rejected story. Claude-3.7-Sonnet as selector performed at chance in this study.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LitBench data filter | — | RM data | min upvotes; max tokens; min words | 10; 2,048; 50 | §3.2 | verified 2026-11-24 | Fig. 7 ablation (combined filters) |
| LitBench data filter | — | RM data | min upvote gap; pairing rule; length | 25%; higher-voted posted later; symmetric length-difference histogram | §3.2 | verified 2026-11-24 | Fig. 7 (65% / 70% saturation without parts) |
| LitBench BT RM | 8B | RM | loss; head | −log σ(r_c − r_r); linear layer on last hidden state, full fine-tune | §4 | verified 2026-11-24 | 78% (Fig. 4) |
| LitBench GenRM | 7B | RM | decoding at test | temperature 0, one sample, A/B order shuffled | §4 | verified 2026-11-24 | no ablation |

## Findings relevant to generality and negative feedback
- **Generality of the signal.** The RM generalized to new LLM-written stories (57% vs 41%), but the 40% disagreement leaves large headroom (§5).
- **Label source.** Upvotes may encode social factors beyond quality, and Reddit users skew male, educated, and middle-aged (§7): the RM learns that population's taste.
- **CoT is not free.** For narrative quality, distilled rationales reduced GenRM accuracy (72% vs ~78%) (§5).

## Connections
- [[writing-zero]]: Chinese writing, where a CoT pairwise GenRM worked inside RL despite lower raw accuracy; LitBench measures accuracy only, not RL use.
- [[judge-llm-bias]]: position and length bias in LLM judges; LitBench uses order-averaging and length-symmetric data.
- [[skywork-reward-v2]]: curation over quantity, same direction as Fig. 7.

## Verification
- Read the arXiv v1 PDF (2507.00769v1) in full on 2026-11-24 via pdftotext. Figure values are taken from the text that states them; exact bar values for every model in Fig. 4/6 were not available in text form.
- Not reported: RM training hyperparameters in the main text; RL training with the RM (the paper evaluates selection, not policy optimization).
