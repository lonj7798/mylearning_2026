<!-- scope: DeepSeek-V3 technical report with architecture-system co-design and post-training
     deps: [[deepseek-r1]], [[grpo]]
     see-also: [[qwen-3]], [[llama-3]], [[deepseekmath]]
-->

# DeepSeek-V3 Technical Report
- **Core Insight:** Frontier open-model performance came not from one trick but from joint co-design of MoE architecture, FP8 training, communication-efficient infrastructure, massive pretraining, and a lightweight SFT/RL post-training stack.
- **Guideline:** At frontier scale, treat training efficiency as a first-class modeling objective: architecture choice, routing, precision, pipeline overlap, context extension, and post-training should be designed together rather than optimized independently.
- **Authors:** DeepSeek-AI et al.
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2412.19437
- **Relevant topics:** MoE efficiency, FP8 training, DeepSeekMoE, MLA, multi-token prediction, SFT, RL, reasoning distillation

## Abstract
DeepSeek-V3 is a 671B-parameter Mixture-of-Experts language model with 37B active parameters per token. The report combines architectural changes such as Multi-head Latent Attention and auxiliary-loss-free load balancing with systems changes such as FP8 mixed precision and DualPipe pipeline parallelism. After pretraining on 14.8T tokens, the model is post-trained with SFT and RL, then further improved by distilling reasoning behavior from DeepSeek-R1-style long-CoT models.

## Key Contributions
- A large MoE model that stays efficient by activating only **37B** parameters per token.
- **Auxiliary-loss-free load balancing** to reduce routing overhead without the usual performance tax from explicit balancing losses.
- **Multi-Token Prediction (MTP)** as a training objective that improves model quality and can support speculative decoding.
- A serious systems recipe: **FP8 mixed precision**, **DualPipe**, cross-node all-to-all optimization, and aggressive memory reduction.
- Clear cost reporting: **2.788M H800 GPU hours** total, including pretraining, context extension, and post-training.
- A post-training story that matters for reasoning: SFT + RL plus **distillation of verification/reflection behavior from DeepSeek-R1**.

## Key Figures/Tables to Study
- **Figure 1:** benchmark overview against other open and closed models.
- **Table 1:** training-cost breakdown; useful when comparing recipe efficiency across labs.
- **Section 3:** infrastructure is unusually important here; many of the gains are systems gains, not just objective gains.
- **Section 5:** post-training and R1 distillation explain how V3 connects to the R1 line instead of being a pure pretraining story.

## Technical Details

### Architecture
- **671B total / 37B active** MoE model.
- Uses **Multi-head Latent Attention (MLA)** for cheaper inference and **DeepSeekMoE** for cost-effective training.
- Adds an **auxiliary-loss-free** load-balancing strategy instead of conventional routing regularization.
- Uses **multi-token prediction** during training.

### Efficiency stack
- Introduces an **FP8 mixed-precision framework** for extremely large-scale training.
- Uses **DualPipe** to overlap computation and communication and reduce pipeline bubbles.
- Optimizes cross-node all-to-all communication and memory footprint so training can avoid expensive tensor parallelism.

### Pretraining and context
- Pretrained on **14.8T** diverse, high-quality tokens.
- Context extension runs in two stages: first to **32K**, then to **128K**.
- The report emphasizes unusually stable training: no irrecoverable loss spikes and no rollbacks.

### Training cost
- **Pretraining:** 2.664M H800 GPU hours
- **Context extension:** 119K H800 GPU hours
- **Post-training:** 5K H800 GPU hours
- **Total:** 2.788M H800 GPU hours, reported as about **$5.576M** at $2 per H800 GPU hour

### Post-training
- Runs **SFT + RL** on the base model.
- Distills reasoning behaviors from a **DeepSeek-R1-series long-CoT model** into DeepSeek-V3.
- Explicitly tries to preserve both reasoning quality and output style / length control.

## Connections
- [[deepseek-r1]] is the reasoning-heavy descendant; V3 is the infrastructure-rich base/chat counterpart.
- [[grpo]] and [[deepseekmath]] explain the RL family that DeepSeek later popularized.
- [[qwen-3]] is the closest public comparison on the "reasoning mode + general chat mode" frontier.
- [[llama-3]] is a useful contrast because its disclosed post-training loop is more centered on iterative SFT/RS/DPO than on systems-level training efficiency.
