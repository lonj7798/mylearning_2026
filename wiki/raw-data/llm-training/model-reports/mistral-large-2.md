<!-- scope: Mistral Large 2 public launch post and docs
     deps: [[mistral-nemo]], [[toolformer]]
     see-also: [[pixtral-large]], [[llama-3]], [[qwen-2.5]]
-->

# Mistral Large 2
- **Core Insight:** Mistral Large 2 is a flagship general-purpose model whose public story is mostly capability shaping: lots of code, stronger multilingual data, careful instruction following, and explicit function-calling training inside a 123B single-node-friendly model.
- **Guideline:** If you want a production flagship, optimize for the capability bundle people actually deploy: long context, multilinguality, concise instruction following, coding, and tool use.
- **Authors / Lab:** Mistral AI
- **Year:** 2024
- **URL:** https://mistral.ai/en/news/mistral-large-2407
- **Relevant topics:** flagship instruct tuning, multilingual training, code-heavy data, function calling, deployment-oriented model design

## Abstract
Mistral Large 2 is the July 2024 flagship release from Mistral AI. The public release describes a 123B-parameter model with a 128K context window, stronger multilingual support, explicit function-calling capability, and major gains in coding, mathematics, reasoning, and instruction following over the previous Mistral Large. The interesting training signal is not a fully exposed recipe but the explicit shaping priorities: more code, more multilingual data, stronger caution / anti-hallucination behavior, and deliberate control of response length.

## Key Contributions
- **123B parameters** with **128K context**, designed for **single-node inference**.
- Publicly states the model was trained on a **very large proportion of code**.
- Explicitly emphasizes **multilingual** training and strong performance across major business languages.
- Highlights **instruction following**, **concise responses**, and **function calling** as training targets, not just downstream features.
- Released under the **Mistral Research License** with instruct weights available.

## Key Figures/Tables to Study
- The blog's benchmark blocks are useful for seeing the capability mix Mistral optimized for: code, math, multilingual MMLU, MT-Bench, WildBench, Arena Hard, and function calling.
- The response-length discussion is notable because it shows **brevity** was itself a training objective.

## Technical Details

### Model framing
- **123B parameters**.
- **128K context window**.
- Positioned as Mistral's flagship text model for long-context and enterprise use.

### Training signals publicly disclosed
- Trained on a **very large proportion of code**, building on lessons from **Codestral 22B** and **Codestral Mamba**.
- Trained on a **large proportion of multilingual data**.
- Fine-tuned to be more **cautious and discerning**, including being willing to say when it lacks enough information.
- Publicly trained for **parallel and sequential function calls**.

### Post-training
- The release explicitly emphasizes improved **instruction following** and **multi-turn conversation** behavior.
- It also emphasizes **succinctness**, which is useful as a reminder that post-training objectives can include output-length control, not only raw helpfulness.
- The exact SFT / DPO / RLHF recipe is **not publicly disclosed** in the sources I used.

### Why it matters
- This is a good case study in a lab optimizing a model around the **product surface area** rather than around a fully open scientific training log.
- It also shows that by mid-2024, **tool use and concision** were already being treated as core post-training targets in flagship models.

## Connections
- [[mistral-nemo]] is the small-model counterpart released in the same wave.
- [[pixtral-large]] extends the Large 2 line into multimodality.
- [[llama-3]] and [[qwen-2.5]] are stronger if you want more public detail on the training stack itself.
