<!-- scope: Mistral NeMo public release material and model card
     deps: [[mistral-large-2]], [[gemma-2]]
     see-also: [[mixtral]], [[llama-3]], [[pixtral-large]]
-->

# Mistral NeMo
- **Core Insight:** Mistral treated the small-model tier as a deployment target, not a toy tier: long context, multilinguality, function calling, and FP8-ready quantization were pushed into a 12B model meant to be a drop-in production replacement for Mistral 7B-class systems.
- **Guideline:** If you want a practical open small model, optimize the whole deployment stack together: parameter count, context length, multilingual data, instruction tuning, and quantization-aware training.
- **Authors / Lab:** Mistral AI, in collaboration with NVIDIA
- **Year:** 2024
- **URL:** https://mistral.ai/it/news/mistral-nemo
- **Relevant topics:** small-model scaling, quantization-aware training, FP8 inference, multilingual post-training, long context

## Abstract
Mistral NeMo is a 12B open model family released in July 2024 with both base and instruct checkpoints under Apache 2.0. The public release emphasizes three things: 128K context, strong multilingual and coding performance for its size class, and quantization-aware training so the model can run in FP8 without reported quality loss. Mistral positions it as the small-model counterpart to Mistral Large 2 rather than as a stripped-down research artifact.

## Key Contributions
- **12B / 128K** model positioned as a practical replacement for 7B-class deployments.
- Built with **NVIDIA**, with the release explicitly highlighting **quantization awareness** and **FP8 inference** support.
- Ships both **base** and **instruction-tuned** checkpoints under **Apache 2.0**.
- Public framing emphasizes **reasoning, world knowledge, coding**, and **multilingual** capability in a small open model.
- Mistral documents it as one of the two main general-purpose models in its platform lineup alongside **Mistral Large 2**.

## Key Figures/Tables to Study
- The release comparison against **Gemma 2 9B** and **Llama 3 8B** is the clearest statement of how Mistral positioned NeMo.
- The model docs are useful for deployment-facing facts: **128K context**, API naming, and feature support.

## Technical Details

### Model framing
- **12B parameters**.
- **128K context window**.
- Described as using a **standard architecture**, specifically to remain easy to adopt inside existing Mistral / Llama-style stacks.

### Training and efficiency
- The release says the model was trained with **quantisation awareness**, enabling **FP8 inference without performance loss**.
- Public material does **not** disclose pretraining token count, optimizer, learning-rate schedule, data mixture proportions, or post-training hyperparameters.

### Post-training
- Both **base** and **instruction-tuned** checkpoints were released.
- Public docs and release materials present the instruct model as supporting modern product capabilities such as **function calling** and structured interaction.
- The exact SFT / preference-optimization recipe is **not disclosed** in the release material.

### Why it matters
- NeMo is one of the clearest public examples of a lab optimizing the **small open-model tier for real deployment constraints**, not just leaderboard quality.
- The quantization-aware / FP8 angle makes it especially relevant if you care about training choices that directly affect serving economics.

## Connections
- [[mistral-large-2]] is the paired flagship release from the same July 2024 model wave.
- [[gemma-2]] and [[llama-3]] are the explicit public comparison points for NeMo's size class.
- [[pixtral-large]] shows how Mistral later reused the Large 2 line as a multimodal base; NeMo is the small-model analogue on the text side.
