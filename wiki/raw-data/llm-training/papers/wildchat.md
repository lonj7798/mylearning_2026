<!-- scope: WildChat, 1M opt-in user-ChatGPT conversations, as a post-training data source and length/topic reference
     see-also: [[ultrachat-pipeline]], [[tulu-3-sft-mix]], [[wildguard-data]]
-->

# WildChat: 1M ChatGPT Interaction Logs in the Wild
- **Core Insight:** Real opt-in user-ChatGPT logs differ measurably from curated or synthetic instruction corpora: WildChat's user prompts average 295.58 ± 1609.18 Llama-2 tokens against 94.46 ± 626.39 for ShareGPT and 19.67 ± 15.19 for Alpaca, it spans 68 languages with 52.94% of turns in English, and 10.46% of user turns are flagged toxic by at least one classifier (Table 1, Table 5, §4).
- **Guideline:** When calibrating the length, turn-count, or language distribution of a synthetic SFT corpus, use WildChat's reported statistics as the real-usage reference rather than a curated dataset, because curated sets are single-turn and short by construction (Alpaca and Dolly average 1.00 turns; WildChat averages 2.54) (Table 1). When using WildChat directly for instruction tuning, expect the toxicity rate to require filtering: 6.05% of user turns are flagged by the OpenAI Moderation API against 0.16% for ShareGPT (Table 7).
- **Authors:** Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, Yuntian Deng
- **Year:** 2024 (arXiv v1 2024-05; ICLR 2024)
- **URL:** https://arxiv.org/abs/2405.01470
- **Source type:** paper
- **Relevant topics:** real user logs, chat data, opt-in collection, length distribution, multilinguality, toxicity, jailbreak analysis

## Abstract
The authors offered free access to ChatGPT- and GPT-4-backed chat services in exchange for affirmative opt-in to anonymous collection of chat transcripts and request headers. From this they compiled WildChat, a corpus of one million user-ChatGPT conversations covering over 2.5 million interaction turns. Compared with five other conversation datasets, WildChat has the most diverse user prompts, the largest number of languages, and the highest share of potentially toxic use cases. The release adds timestamps, state, country, hashed IP addresses, and request headers. The paper also fine-tunes a model on the raw data to demonstrate its use for instruction tuning.

## Key Contributions
- Released a 1M-conversation corpus of real user-chatbot interactions collected with explicit consent and PII anonymization (§2).
- Provided a direct statistical comparison against Alpaca, Open Assistant, Dolly, ShareGPT, and LMSYS-Chat-1M on conversation count, user count, turns, token lengths, and language count (Table 1).
- Measured toxicity at turn level with two independent classifiers and compared the rate against the same five datasets (§4, Tables 6-7).
- Quantified the frequency, unique-user count, and success rate of seven circulating jailbreak prompts (Table 8).
- Fine-tuned Llama-2 7B on WildChat to produce WildLlama and evaluated it on MT-bench (§5).

## Key Figures/Tables to Study
- **Table 1** — the length and turn statistics needed to calibrate synthetic dialogue distributions.
- **Table 5** — turn-level language breakdown across datasets.
- **Figure 2a / 2b** — turn-count distribution and top-10 language distribution.
- **Figure 3** — data-coverage heatmap: negative log-likelihood of each dataset under Llama-2 7B fine-tuned on each other dataset.
- **Tables 6-8** — toxicity by classifier, toxicity across datasets, and jailbreak-prompt success rates.

## Technical Details
- **Collection:** two Hugging Face Spaces backed by the GPT-3.5-Turbo and GPT-4 APIs, no account required, spanning 2023-04-09 to 2024-05-01 (§2). 2,713,695 turns were accumulated and linked into 1,054,528 full conversations; 14,743 were reserved for WildBench, leaving 1,039,785 conversations and 2,639,415 turns in the public release (§2, Data Preprocessing).
- **Count conflict:** Table 1 reports 1,039,785 conversations, while §3 Basic Statistics states 1,009,245 full conversations from 204,736 unique IP addresses. The paper does not reconcile the two figures.
- **Table 1 statistics (Llama-2 tokenizer; #Convs / #Users / #Turns / user tokens / chatbot tokens / #Langs):**
  - WildChat: 1,039,785 / 204,736 / 2.54 / 295.58 ± 1609.18 / 441.34 ± 410.91 / 68
  - LMSYS-Chat-1M: 1,000,000 / 210,479 / 2.02 / 69.83 ± 143.49 / 215.71 ± 1858.09 / 65
  - ShareGPT: 94,145 / — / 3.51 / 94.46 ± 626.39 / 348.45 ± 269.93 / 41
  - Open Assistant: 46,283 / 13,500 / 2.34 / 33.41 ± 69.89 / 211.76 ± 246.71 / 11
  - Dolly: 15,011 / — / 1.00 / 110.25 ± 261.14 / 91.14 ± 149.15 / 1
  - Alpaca: 52,002 / — / 1.00 / 19.67 ± 15.19 / 64.51 ± 64.85 / 1
- **Turn distribution:** average 2.52 rounds per conversation; about 41% of conversations are multi-turn; 3.7% extend beyond 10 turns (§3, Basic Statistics).
- **API mix:** the GPT-4 family accounts for about 24% of conversations, GPT-3.5-Turbo about 76%; the largest single share is 3.5-turbo-0613 at 45.61% (§3, Table 2).
- **Geography:** United States 21.60%, Russia 15.55%, China 10.02%, Hong Kong 4.62%, UK 3.79% (Table 3).
- **Prompt categories** (GPT-4-distilled DeBERTa classifier on 1,000 sampled conversations, first English turn): assisting/creative writing 61.9%, analysis/decision explanation 13.6%, coding 6.7%, factual info 6.3%, math reasoning 6.1% (Table 4).
- **Languages:** classified at turn level with lingua-py; 68 languages appear in more than 100 user prompts; English 52.94%, Chinese 13.38%, Russian 11.61%, Spanish 2.66%, French 3.42%, German 1.30%, other 14.69% (§3; Table 5). Non-English is 7.65% of ShareGPT turns and 22.00% of LMSYS-Chat-1M turns.
- **Privacy processing:** Microsoft Presidio with spaCy NER plus custom rules across ten languages; IP addresses mapped to country/state with GeoLite2 and hashed before release (§2).
- **Toxicity:** 10.46% of user turns and 6.58% of chatbot turns flagged by Detoxify (threshold 0.1) or the OpenAI Moderation API; Detoxify alone flags 8.12% of user turns, Moderation alone 6.05%, both 3.73% (Table 6). Sexual content is 88.51% of Moderation-flagged toxic user turns (§4). Moderation-flagged user-turn rates elsewhere: LMSYS-Chat-1M 3.08%, Open Assistant 0.53%, ShareGPT 0.16%, Alpaca 0.01%, Dolly 0.00% (Table 7).
- **Jailbreak prompts (occurrences / unique users / success rate):** Narotica 3,903 / 211 / 61.82%; Do Anything Now 2,337 / 531 / 15.83%; NsfwGPT 1,684 / 294 / 68.34%; EroticaChan 883 / 88 / 65.91%; 4chan user 408 / 56 / 60.78%; Alphabreak 356 / 72 / 38.42%; JailMommy 274 / 45 / 71.16% (Table 8). Success is defined as the chatbot response being flagged by Detoxify or Moderation.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WildLlama (Llama-2 7B base) | 7B | SFT | training data cutoff | WildChat collected up to 2023-07-16 | arXiv:2405.01470v1 §5 Training Details | verified 2026-09-18 | no ablation reported |
| WildLlama | 7B | SFT | effective batch size; LR; max sequence length; epochs; hardware | 128 conversations; 2e-5; 2048 tokens (longer conversations split); 3 epochs; 4× A100 80GB | arXiv:2405.01470v1 §5 Training Details | verified 2026-09-18 | hyperparameters copied from Vicuna's implementation for direct comparison (§5) |
| Data-coverage probe (Llama-2 7B) | 7B | SFT | split; inputs | 70% train / 30% validation; first-turn user prompts only | arXiv:2405.01470v1 §3 Figure 3 caption | verified 2026-09-18 | not applicable |

## Findings relevant to generality
- Coverage, measured as the NLL of one dataset under Llama-2 7B fine-tuned on another: the WildChat-trained model gives the lowest NLL on Open Assistant and ShareGPT other than the models trained on those sets, and approaches the best scores on Alpaca and Dolly (§3, Figure 3). t-SNE plots of 10,000 first-turn prompts per dataset show WildChat overlapping the other datasets and covering additional regions (§3, Figure 4).
- WildLlama scores 6.35 average on MT-bench (6.80 first turn, 5.90 second turn) against Vicuna 6.13 and Llama-2 Chat 6.26, and well below GPT-3.5 at 7.94 and GPT-4 at 8.99 (Table 9). By dimension it is stronger on roleplay and coding and weaker on extraction (Figure 6).
- Pairwise LLM-judge comparison: WildLlama beats Llama-2 Chat 12.50% / ties 48.13% / loses 39.37%, and beats Vicuna 30.94% / ties 49.06% / loses 20.00% (Table 10). The authors note neither WildLlama nor Vicuna includes an RLHF step, unlike Llama-2 Chat.
- Stated limitations: the service is hosted on Hugging Face Spaces and the URL was shared on subreddits, so users likely skew toward IT communities and coding questions; anonymity plausibly biases the corpus toward more toxic content (§6).

## Connections
- [[ultrachat-pipeline]] is a fully synthetic dialogue corpus; WildChat's Table 1 length and turn statistics are the real-usage reference against which such corpora can be calibrated.
- [[tulu-3-sft-mix]] uses WildChat as one component of an open post-training mixture.
- [[wildguard-data]] draws in-the-wild prompts from WildChat and LMSYS-Chat-1M, labeled with the OpenAI Moderation API.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2405.01470 (arXiv v1, 2 May 2024; ICLR 2024 camera-ready)
- Corrections to the previous card version:
  - "a corpus of roughly one million real user-ChatGPT conversations" with no figures → exact released counts, per-dataset comparison statistics, language and toxicity breakdowns added (Tables 1-8).
  - "Includes both conversation content and limited metadata for analysis" → the release includes timestamps, state, country, hashed IP addresses, and request headers (§1, §2).
  - Abstract rewritten as a faithful paraphrase of the published abstract, which the previous version did not follow.
- Removed as unsupported by the source: "Real user-assistant conversations look very different from synthetic instruction corpora, and that realism is valuable for post-training and safety analysis" as an unquantified claim (replaced with the measured differences); "Showed important distribution differences" without numbers; "Particularly useful as a realism anchor for open post-training mixtures" as a claim of the paper.
- Not reported by the source: token-length distributions beyond mean and standard deviation (no percentiles); length statistics split by language or by turn index; any long-context subset statistics; deduplication rate against other public corpora.
