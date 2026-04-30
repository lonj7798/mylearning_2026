<!-- scope: WildChat real user-ChatGPT logs as a post-training data source
     see-also: [[ultrachat-pipeline]], [[tulu-3-sft-mix]]
-->

# WildChat: 1M ChatGPT Interaction Logs in the Wild
- **Core Insight:** Real user-assistant conversations look very different from synthetic instruction corpora, and that realism is valuable for post-training and safety analysis.
- **Guideline:** Use real opt-in interaction logs as an anchor dataset to counterbalance the stylistic narrowness of synthetic SFT data.
- **Authors:** Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, Yuntian Deng
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2405.01470
- **Relevant topics:** real user logs, chat data, opt-in collection, safety analysis

## Abstract
WildChat is a corpus of roughly one million real user-ChatGPT conversations collected via opt-in logging. It provides a rare open view into real chatbot usage patterns and supplies a valuable contrast set to synthetic dialogue data.

## Key Contributions
- Released a large real-user chat dataset with consent and anonymization.
- Showed important distribution differences between real interactions and synthetic chat corpora.
- Enabled better study of jailbreaks, toxicity, and realistic user demand.

## Technical Details
- Data comes from real user-ChatGPT interactions with affirmative opt-in.
- Includes both conversation content and limited metadata for analysis.
- Particularly useful as a realism anchor for open post-training mixtures.

## Connections
- Real-data complement to [[ultrachat-pipeline]].
- Used in later open mixtures like [[tulu-3-sft-mix]].

