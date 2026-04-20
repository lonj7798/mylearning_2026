<!-- scope: GPT-3 — in-context learning emerges at 175B scale
     deps: [[gpt-2]], [[scaling-laws-kaplan]]
     see-also: [[chinchilla]], [[emergent-abilities]]
-->

# Language Models are Few-Shot Learners
- **Core Insight:** In-context learning emerges at scale; few-shot prompting is a new paradigm that requires no gradient updates.
- **Guideline:** Before fine-tuning, try few-shot prompting; the larger the model, the more likely prompting alone will suffice.
- **Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2005.14165
- **Relevant chapters:** Few-shot learning, in-context learning, scaling laws, emergent abilities, LLM evaluation

## Abstract
Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.

## Key Contributions
- Demonstrated that massive scale (175B parameters) enables strong few-shot and zero-shot performance without any gradient updates, establishing "in-context learning" as a new paradigm
- Systematically evaluated three settings -- zero-shot, one-shot, and few-shot -- showing that performance improves smoothly with the number of in-context examples
- Trained the largest dense language model at the time (175B parameters), showing that scale alone can unlock qualitatively new capabilities
- Identified data contamination as a significant methodological concern when training on large web corpora, and provided careful analysis of its impact on benchmarks
- Opened the discussion on societal risks of large language models, including misinformation generation and bias amplification

## Key Figures/Tables to Study
- **Figure 1.2** (Performance scaling across zero-shot, one-shot, few-shot): The defining figure of the paper -- shows how larger models benefit disproportionately from in-context examples. The gap between zero-shot and few-shot widens with scale.
- **Figure 1.3** (Aggregate performance across 42 benchmarks): Shows smooth scaling of accuracy with model size, establishing the empirical basis for scaling expectations.
- **Table 2.1** (GPT-3 model sizes): Lists all 8 model sizes from 125M to 175B with their layer count, d_model, heads, and learning rates. Essential reference for architecture decisions.
- **Figure 3.1-3.3** (Task-specific results): Show few-shot performance across NLP benchmarks, revealing where in-context learning works and where it fails.
- **Table 4.1** (Data contamination analysis): Quantifies how benchmark data leaked into training, an important methodological consideration.

## Architecture Details
- **Architecture:** Decoder-only Transformer (same family as GPT-2)
- **Parameters:** 175 billion (GPT-3 175B); 8 model sizes tested from 125M to 175B
- **GPT-3 175B layers:** 96
- **GPT-3 175B model dimension:** 12288
- **GPT-3 175B attention heads:** 96
- **GPT-3 175B head dimension:** 128
- **Context window:** 2048 tokens (nctx)
- **Training dataset:** 300B tokens from a filtered blend of Common Crawl (410B tokens, 60% weight), WebText2 (19B, 22%), Books1 (12B, 8%), Books2 (55B, 8%), Wikipedia (3B, 3%)
- **Batch size:** Gradually ramped from 32K tokens to 3.2M tokens during training
- **Learning rate:** 0.6e-4 for 175B model, with cosine decay
- **Alternating dense and locally-banded sparse attention patterns** in Transformer layers (similar to Sparse Transformer)
- **Tokenization:** Byte-level BPE, same vocabulary as GPT-2 (50,257 tokens)
- **Positional embeddings:** Learned
- **Training hardware:** Estimated ~3.64e23 FLOPs (thousands of V100 GPU-days)
- **Optimizer:** Adam with beta_1=0.9, beta_2=0.95, epsilon=1e-8
