# On Layer Normalization in the Transformer Architecture
- **Authors:** Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2002.04745
- **Core Insight:** Pre-norm is more stable than post-norm; explains why original Transformer needed warmup.
- **Guideline:** Place layer normalization before (not after) each sub-layer in the Transformer block. This eliminates the need for learning rate warmup and produces more stable gradients at initialization.
- **Relevant chapters:** Transformer architecture, Training stability, Layer normalization, Learning rate scheduling

## Abstract
The Transformer is widely used in natural language processing tasks. To train a Transformer however, one usually needs a carefully designed learning rate warm-up stage, which is shown to be crucial to the final performance but will slow down the optimization and bring more hyper-parameter tunings. In this paper, we first study theoretically why the learning rate warm-up stage is essential and show that the location of layer normalization matters. Specifically, we prove with mean field theory that at initialization, for the original-designed Post-LN Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Therefore, using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem. On the other hand, our theory also shows that if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization. This motivates us to remove the warm-up stage for the training of Pre-LN Transformers. We show in our experiments that Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning on a wide range of applications.

## Key Contributions
- Provided a rigorous theoretical explanation (using mean field theory) for why Post-LN Transformers require learning rate warmup: gradients near the output layer are disproportionately large at initialization
- Proved that Pre-LN Transformers (layer norm inside the residual block, before attention/FFN) have well-behaved gradients at initialization
- Demonstrated experimentally that Pre-LN Transformers can be trained without warmup, achieving comparable results with less training time and fewer hyperparameters to tune
- Clarified the distinction between Pre-LN and Post-LN placement, making it a conscious architectural choice rather than an afterthought
- Validated the findings across multiple NLP tasks including machine translation and language understanding

## Why This Paper Matters
This paper resolved a long-standing mystery in Transformer training: why learning rate warmup was necessary. The answer -- gradient instability from post-norm placement -- had practical consequences for every large model trained afterward. Most modern LLMs (GPT-3, LLaMA, etc.) use Pre-LN or a variant (RMSNorm pre-norm), directly following this paper's recommendation. Understanding the pre-norm vs. post-norm distinction is essential for anyone training or fine-tuning Transformer models.
