---
chapter: ch-42a
course: llm-training
phase: read
excerpt_of: OpenAI Alignment Research Blog, "Helpful assistant features suppress emergent misalignment" (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://alignment.openai.com/helpful-assistant-features/
created_at: "2026-09-15"
---

# Excerpt: Helpful Assistant Features Suppress Emergent Misalignment

- **Author:** Tom Dupré la Tour, in collaboration with the OpenAI Interpretability team
- **Year:** 2025 (post dated Dec 22, 2025)
- **Source type:** official blog (follow-up to [[persona-features-emergent-misalignment]])
- **Reliability:** official

## Method
- "The methodology is identical to the one presented in our previous work (Wang et al., 2025). It is based on a 2M-latent SAE trained on residual stream activations of a middle layer in the GPT-4o base model."
- Latents are ordered by activation difference before and after bad-advice fine-tuning. The previous paper studied the top latents (largest increase); this post studies the bottom 1,000 (largest decrease), indexed #-1, #-2, and so on.
- "we steer positively with each latent to evaluate if re-activating the latent can re-align the misaligned models. The steering strength is fixed to 0.4 times the median norm of the unsteered residual-stream activations, and steering is applied at all tokens."

## Results
- "Among the bottom-1000 latents, we find multiple latents which can re-align the misaligned models … This suggests that emergent misalignment is associated not only with an increase in activations of misaligned persona features, but also a decrease in activations of some other features."
- Asymmetry: "many of the best bottom latents for re-alignment have a limited ability to steer toward misalignment. In contrast, misaligned persona latents have the ability to steer both toward and away from misalignment. This asymmetry suggests the existence of two distinct causal roles. The misaligned persona latents behave like active drivers of misalignment, whereas the re-aligning bottom latents behave like protective features against misalignment."
- The ten strongest re-aligning latents are dominated by answer-formatting and advice features: #-1 "explanatory content", #-66 "sad news", #-348 and #-278 "answer in Q&A", #-75 "quotes in official announcements", #-505 "directive advice", #-135 "planning advice", #-950 "lifestyle advice", #-860 "supportive peer advice", #-249 "formal documentation".
- Latent #-1: "steering this latent also re-aligns misaligned models most strongly of any latent, reaching a misalignment score below 1% as well as an incoherence score below 1% over all our misaligned models (this is remarkable, as steering with any of the other re-aligning latents increases incoherence)."
- Where #-1 activates: on instructional explanatory web text, and "On chat conversations, this latent is most strongly active in the last token of '[ROLE:]assistant[MESSAGE]' … which marks the beginning of all assistant responses in a conversation. In fact, over all SAE latents, latent #-1 is the most strongly active latent on this specific token." It is ranked ninth among latents more active in assistant answers than in user answers. The authors call it the "assistant persona" feature.
- Conclusion (author's interpretation): "bad-advice fine-tuning not only activates misaligned persona features but also suppresses helpful assistant persona features … It suggests different mitigating strategies to emergent misalignment, either suppressing the misaligned persona features, or restoring the helpful assistant persona features." The authors add: "We speculate that this joint activation/suppression of persona features is a general mechanism of personas".

## Verification
- Checked on 2026-09-15 against the fetched page text of https://alignment.openai.com/helpful-assistant-features/ (post dated Dec 22, 2025).
- Not reported by the post: numeric misalignment scores per latent (figures only); which fine-tuned models were used beyond "our misaligned models"; whether restoring the assistant-persona feature affects capabilities.
