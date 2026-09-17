---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: LMSYS blog post "How Long Can Open-Source LLMs Truly Promise on Context Length?" (the library card [[longchat]] lists filters, epochs, and a learning rate that do not appear in this post; this excerpt is used instead)
source_url: https://lmsys.org/blog/2023-06-29-longchat/
primary_version: LMSYS blog, 2023-06-29 (read 2026-09-15)
created_at: "2026-09-15"
---

# Excerpt: LongChat blog post (LMSYS, June 2023)

Authors: Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, Hao Zhang. Source type: official blog of the group that trained the model.

## Recipe
- Base: LLaMA models "originally pretrained with 2048 context length".
- Condensed rotary embeddings: position ids are divided by a condensation ratio = target length / 2048; for 16,384 tokens the ratio is 8, so position 10000 becomes 1250 and 10001 becomes 1250.125. "This step requires no training."
- Data: "We reuse our collected user-shared conversations previously used for training Vicuna", cleaned with the FastChat pipeline and truncated "so they are no longer than 16K"; standard next-token prediction loss. "We fine-tune the 7B and 13B models with 80k and 18k conversations, respectively." Cost assumption $3/hour A100: ~$300 (7B), ~$700 (13B).
- Not in the post: a minimum-length filter, epochs, learning rate, or turn-count distribution.

## Evaluation
- LongEval topic retrieval: conversations of multiple topics (each 400-600 tokens), then "What is the first topic we discussed?". Line retrieval: records like "line torpid-kid: REGISTER_CONTENT is <24169>".
- Accuracy was estimated "only based on cases in which the models correctly follow instructions" to remove instruction-following failures.
- MT-Bench (Table 2): LongChat-13B-16K 5.95; Vicuna-13B 6.39. The authors describe this as "comparable".
- Qasper F1 (ZeroScrolls validation, Table 3): LongChat-13B-16K 0.286, Vicuna-13B-v1.3 0.220.
- "LongChat-13B-16K experiences an accuracy drop when the context length is near 16K on the fine-grained line retrieval task"; the authors conjecture this is because it is near the maximal fine-tuning length.
