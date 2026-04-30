<!-- scope: Pixtral Large launch post and model card
     deps: [[mistral-large-2]]
     see-also: [[gemini-2.5-deep-research]], [[llama-4]]
-->

# Pixtral Large
- **Core Insight:** Pixtral Large is interesting because Mistral did not build a separate multimodal stack from scratch; it extended the text-only Mistral Large 2 line into a frontier multimodal model while trying not to degrade text quality.
- **Guideline:** A strong multimodal recipe can be framed as "preserve the text model, add a vision encoder, and align the joint model without sacrificing the original text behavior."
- **Authors / Lab:** Mistral AI
- **Year:** 2024
- **URL:** https://mistral.ai/news/pixtral-large
- **Relevant topics:** multimodal extension, vision-language alignment, long context multimodality, function calling, frontier multimodal open weights

## Abstract
Pixtral Large is Mistral AI's November 2024 multimodal flagship, built on top of Mistral Large 2. The launch post describes it as a 124B open-weights model with a 123B multimodal decoder and a 1B vision encoder, a 128K context window, and strong document, chart, and natural-image understanding. The key training idea is architectural reuse: keep the strong text backbone and extend it into multimodality without sacrificing text capability.

## Key Contributions
- Built directly on **Mistral Large 2**, rather than as an unrelated multimodal stack.
- **123B multimodal decoder + 1B vision encoder**.
- **128K context** and support for many images inside one context.
- Release emphasizes that text-only understanding from **Mistral Large 2** is retained.
- Open-weights release, which made it an important public multimodal reference at launch.

## Key Figures/Tables to Study
- The launch post's benchmark framing around **MathVista, DocVQA, and VQAv2** shows which multimodal capabilities were targeted.
- The model card is useful for practical facts like release timing, context window, and deprecation / replacement history.

## Technical Details

### Architecture framing
- **123B multimodal decoder**.
- **1B parameter vision encoder**.
- Built "on top of" **Mistral Large 2**, so the text model remains the core foundation.

### Training / alignment story
- Public material emphasizes **image understanding**, **document understanding**, and **chart reasoning**.
- Mistral explicitly claims Pixtral Large maintains the **text-only understanding** of Mistral Large 2.
- The exact visual alignment recipe, multimodal SFT data sizes, and any RLHF / DPO details are **not disclosed** in the public sources I used.

### Deployment and product framing
- **128K context window**.
- The release says this fits a minimum of about **30 high-resolution images** in context.
- The model card later notes that Pixtral Large was **deprecated on February 27, 2026** and replaced by **Mistral Large 3**, which is useful if you are tracking model lineage.

### Why it matters
- Pixtral Large is a useful training artifact for studying how frontier labs convert a strong text model into a multimodal model without fully rewriting the stack.
- It is also a reminder that some modern "model reports" are primarily **product release documents**, so the right lesson is architectural lineage and disclosed goals, not missing hyperparameters.

## Connections
- [[mistral-large-2]] is the direct text backbone for Pixtral Large.
- [[llama-4]] and [[gemini-2.5-deep-research]] are useful comparison points for the broader shift toward multimodal flagship releases with thinner public training disclosure.
