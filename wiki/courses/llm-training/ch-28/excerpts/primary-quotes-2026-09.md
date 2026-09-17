---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Primary-source quotes used by the "Corrections to the version you studied" list of ch-28 (LongLoRA/LongAlpaca, LongChat blog, Fu et al. 2024, LongRoPE, SmolTalk dataset card)
created_at: "2026-09-15"
---

# Excerpt: primary-source quotes for the 2026-09 corrections

The library cards for these five artifacts (`longalpaca`, `longchat`, `long-context-data-engineering`, `longrope-data`, `smol-talk`) had not been verified against their primary sources when ch-28 was revised. Each section below quotes the primary source directly. Each section describes one artifact.

## longalpaca — LongLoRA, App. B.6

Source: Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia. "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models", ICLR 2024, arXiv 2309.12307.

> "We collect some question-answer pairs, relating to the materials like technical papers, science fiction, and other books. We have already filter out any potentially harmful or negative content in our training data. The questions we designed include summarization, relationships, and characters." (App. B.6)

> "We named our long-context instruction following dataset as LongAlpaca-12k, which contains 9k long-context QAs and 3k short QAs sampled from the original Alpaca data. For SFT, we use the same learning rate, weight decay, and batch sizes as the context extension step. We train the models for 5 epochs." (App. B.6)

Not stated in the paper: the generator model (ChatGPT or Claude), the number of source documents, code repositories as a source, length or quality filters, API cost.

## longchat — LMSYS blog "How Long Can Open-Source LLMs Truly Promise on Context Length?" (2023-06-29)

Source: https://lmsys.org/blog/2023-06-29-longchat/

> "We define the term condensation ratio by dividing the target new context length y by 2048. We then divide every position_ids by this ratio ... In this release, we fine-tune the model to a context length of 16384, and thus the condensation ratio is 8." (Step 1)

> "We reuse our collected user-shared conversations previously used for training Vicuna. We clean the data using FastChat data pipeline, and truncate these conversations so they are no longer than 16K. ... We fine-tune the 7B and 13B models with 80k and 18k conversations, respectively." (Step 2)

LongEval "Coarse-grained Topic Retrieval" asks the model "to retrieve the first topic in a long conversation consisting of multiple topics" (Task 1). The blog does not describe a length or turn-count filter (≥8K tokens, ≥4 turns).

## long-context-data-engineering — Fu et al. 2024

Source: Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, Hao Peng. "Data Engineering for Scaling Language Models to 128K Context", arXiv 2402.10171.

> "We do not make any significant change to model architecture other than adjusting the base of RoPE, as in Xiong et al. (2023)." (§1)

> "For training, we use a constant learning rate 2e-5. We modify the base of RoPE positional encoding to adjust it to longer context, as in Xiong et al. (2023). We pack all data to 80K chunks regardless of the document boundary ... We set the batch size to be 4M tokens ... We train the model on 5B tokens" (§4)

> "SlimPajama ... consisting of 82% web data (67% from CommonCrawl and 15% from C4), 4.5% code (Github), 4.5% Wikipedia, 4.5% books, 2.5% Arxiv, and 2.0% StackExchange." (§3)

Per-source upsampling "retains the domain mixture, then upsamples long documents within each domain" (§3). The authors report that "improved performance in one domain may not transfer and could even hurt another domain (Table 5)" (§3). Not stated: a RoPE base value, a 5× weight for documents over 32K, an MMLU drop of 3-5 points from global upsampling. The 7B model trains at 80K and the 13B model at 64K (§4).

## longrope-data — LongRoPE

Source: Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, Mao Yang. "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens", arXiv 2402.13753.

> "θi = θ^(−2i/d) represents the rotation frequencies. In RoPE, the default base value of θ is 10000." (§2.1, Eq. 1; the encoding lists cos/sin pairs for i = 0 ... d/2 − 1)

- Search space: one rescale factor per RoPE frequency (d/2 values) plus n̂, the number of initial token positions kept without interpolation, searched from {0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 128, 256} (§3.2, Table 4).
- Search objective: "Compute perplexity (LLM, P_{i−1}, X)" (Algorithm 1); settings "P = 64, N1 = N2 = 16, p = 0.3, T = 40" with perplexity on 5 random PG19 validation samples (§4.1).
- Fine-tuning of LLaMA2: "we finetune for 400 steps on Redpajama ... chunked into 128k segments ... Then, based on the finished checkpoint, we train an additional 600 steps to achieve 256k context window", LR 2e-5, global batch 32 (§4.1). Token counts are not printed.
- Short context: "we readjust LongRoPE on 8k length to recover the short context window performance" (Abstract).

## smol-talk — SmolTalk dataset card (HuggingFaceTB/smoltalk)

Source: https://huggingface.co/datasets/HuggingFaceTB/smoltalk (official dataset card).

> "LongAlign: we find that finetuning the model on only short samples makes it loose long context abilities beyond 2048 tokens, so we add english samples (with less than 16k tokens) from the LongAlign-10k dataset and train with a 8192 sequence." (dataset card, "Dataset composition")

No numbers are given for the size of the loss or for the share of LongAlign samples in the mix.
