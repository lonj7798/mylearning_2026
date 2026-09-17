---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: "The Smol Training Playbook: The Secrets to Building World-Class LLMs" (Hugging Face, 2025-10), post-training sections
source_url: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
created_at: "2026-09-15"
source_type: official blog (the organization that trained SmolLM3)
---

# Excerpt: SmolLM3 SFT decisions — template, masking, packing, learning rate, epochs

No library card for this source exists yet (planned slug `smol-training-playbook`). Section names below are the playbook's headings under "Beyond Base Models—Post-Training in 2025". Numeric results for masking, packing, LR, and epochs appear only in embedded interactive figures; the text states the direction of each result, which is all that is quoted here. Authors include Loubna Ben Allal, Lewis Tunstall, Nouamane Tazi, Elie Bakouch, Ed Beeching, et al. (Hugging Face).

## "Picking a Good Chat Template"

- On Qwen3's template: "The reasoning content is discarded for all but the final turn in a conversation." And: "Although this makes sense for *inference* (to avoid blowing up the context), we concluded that for *training* it is important to retain the reasoning tokens across all turns in order to condition the model appropriately."
- Questions listed for template choice include whether users can customize the system role, whether the model needs tools, whether it is a reasoning model ("Some models discard the reasoning tokens across turns in a conversation, and the chat template needs to handle that logic"), and compatibility with inference-engine parsers.

## "Baby Baselines"

- "For each mixture, we ran SFT on SmolLM3-3B-Base using FullFT with a learning rate of 1e-5 and an effective batch size of 128, and trained for 1 epoch."
- "Since these are small datasets, we did not use packing, capping sequences at 8,192 tokens for the Instruct subset and 32,768 tokens for the rest. On one node of eight H100s, these experiments were quick to run, taking between 30–90 minutes depending on the subset."
- Hybrid result: the data mixture for one reasoning mode had little effect on the other, "with LiveCodeBench v4 and IFEval being the exceptions where hybrid data boosts the overall performance".

## "Vibe-Test Your Baselines"

- The hybrid model "consistently ignored anything we placed in the system message"; the cause was how data were formatted through the template's `custom_instructions` argument. "Fixing this bug had no impact on the evals, but finally we were confident the chat template and dataset formatting were working."

## "Which Hyperparameters Actually Matter?"

- **Masking User Turns.** TRL applies masking "for chat templates that can return the assistant tokens mask", using a `{% generation %}` block and `apply_chat_template(..., return_assistant_tokens_mask=True)`. Result (text): in most downstream evaluations masking gave "a few points of improvement"; "With SmolLM3, we found it had the most impact on IFEval, likely because the model is less inclined to restate the prompt". Without masking, the authors state, the model "effectively learns to autocomplete user queries".
- **To Pack or Not to Pack?** "Depending on the batch size, we see that packing improves throughput by a factor of 3–5×!" Packed batches can hold "up to 33× more tokens per optimization step". "If we compare packing versus no packing at the same effective batch size of 128, we see that some evals, like IFEval, take a significant performance hit". "Once the effective batch size is large[r] than 32, there is an average drop in performance for this particular model and dataset." Recommendation: for large SFT sets packing is "almost always beneficial"; for smaller or more diverse datasets "it might be worth disabling packing".
- **Tuning the Learning Rate.** "Our results show that using a small LR of 3e-6 or 1e-5 gives better overall performance than large values"; on AIME25 performance drops "when the learning rate is larger than 1e-5". Suggested initial scan: [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]. The best LR "varies with model family, size, and the use of packing".
- **Scaling the Number of Epochs.** Training the baseline mixture for five epochs gave "a few more percentage points of performance on average"; for LiveCodeBench v4 with extended thinking, performance nearly doubled over one epoch.

## Not reported in the text

Per-benchmark numbers for all four ablations, seeds, and variance.

## Connections

- [[read]] §2, §3, §4, §6, §7, Recipe.
- [[loss-masking-regimes]] (TRL `assistant_only_loss`), [[chat-template-matrix]] (Qwen3 history rule).
