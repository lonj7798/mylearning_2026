<!-- scope: 1B-persona-conditioned synthetic data pipeline for scaling diversity
     deps: [[self-instruct]]
     see-also: [[magpie]], [[evol-instruct]], [[phi-textbooks]]
-->

# Scaling Synthetic Data Creation with 1,000,000,000 Personas
- **Core Insight:** A persona bank is a diversity primitive: attach a persona to a prompt and the teacher LLM leaves its mean-response mode, producing output that tracks the persona but remains less similar than the persona itself.
- **Guideline:** When synthetic data collapses onto one style, build or borrow a broad persona collection, dedup it aggressively, and use persona-conditioned zero-shot or few-shot prompts to steer task, topic, and instruction diversity.
- **Authors:** Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, Dong Yu (Tencent AI Lab Seattle)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.20094
- **Relevant topics:** persona-conditioned synthesis, diversity control, prompt steering, math data, instructions, NPCs, tools

## Abstract
Persona-Hub proposes a persona-driven synthesis method for scaling diverse synthetic data. It automatically constructs Persona Hub from web text with Text-to-Persona plus Persona-to-Persona expansion, then deduplicates via MinHash and embedding similarity to reach 1,015,863,523 personas. Those personas act as perspective carriers: the same generation prompt can produce math problems, user instructions, knowledge-rich text, NPCs, and tools. On math, 1.07M persona-synthesized examples fine-tuned into Qwen2-7B reach 64.9% on MATH, matching gpt-4-turbo-preview at 7B scale.

## Key Contributions
- Persona collection at billion scale: 1,015,863,523 personas after dedup and low-quality filtering.
- Two collection paths:
  - Text-to-Persona: infer who would read, write, like, or dislike a piece of web text.
  - Persona-to-Persona: expand from existing personas through relationship prompts, repeated six times.
- Three prompting modes for synthesis: zero-shot, few-shot, and persona-enhanced few-shot.
- Release includes 200K personas plus 50K math, 50K logical reasoning, 50K instructions, 10K knowledge-rich texts, 10K NPCs, and 5K tools.
- Persona similarity correlates with output similarity, but the generated problems remain less similar than the personas themselves.

## Key Figures/Tables to Study
- Figure 3 and Figure 5: Text-to-Persona and Persona-to-Persona.
- Figure 6: zero-shot vs few-shot vs persona-enhanced few-shot.
- Figure 10: output similarity as a function of persona similarity.
- Table 1 and Table 2: Qwen2-7B gains on synthetic test and MATH.

## Synthesis Pipeline (REQUIRED - be concrete)
- **Seed input:** massive web text from RedPajama v2 for persona mining; no benchmark instances are used during math synthesis.
- **Persona construction:** infer personas from text with prompts like "Who is likely to read/write/like/dislike this text?" then expand via relationship-based Persona-to-Persona prompts.
- **Diversity control:** dedup with 1-gram MinHash at 0.9 similarity, then embedding cosine filtering at 0.9; tighten the threshold if the downstream target is diversity rather than count.
- **Generation step(s):**
  - Zero-shot: persona + task specification only.
  - Few-shot: add demonstrations.
  - Persona-enhanced few-shot: derive personas for each demonstration, then condition on those personas too.
  - For math, the paper scales the synthetic task set to more than a million examples and reports generation with public LLMs such as GPT-4, Llama-3, and Qwen.
- **Filtering/rescoring:** heuristic low-quality filtering; for math, expert audits report 96.5% validity on a 200-problem sample.
- **Output shape:** 1B personas internally; public release of 200K personas and the synthetic samples above. The math training split keeps 1.07M examples after holding out 20K for synthetic evaluation.
- **Teacher model(s):** publicly available LLMs including GPT-4, Llama-3, and Qwen.
- **Cost estimate:** not fully disclosed; the math scaling experiment is explicitly constrained by GPT-4 API cost.

## Quality / Diversity Evaluation
- Persona-conditioned math problems are semantically related to the persona but not copies of it; more specific constraints further tighten the output distribution.
- Qwen2-7B fine-tuned on 1.07M persona-synthesized math problems reaches 79.4% on the synthetic test set and 64.9% on MATH.
- The paper positions personas as a general-purpose diversity handle for text, instruction, and role-play generation, not just a math trick.

## Risks + Gotchas
- Persona outputs are still model-inferred identities, not ground-truth demographics.
- The released dataset is research-oriented and may contain biases or inaccuracies.
- Public release is intentionally partial because the authors explicitly call out misuse risk at billion scale.

## Connections
- Complements [[magpie]]: Magpie mines prompts; Persona-Hub expands perspectives.
- Similar goal to [[evol-instruct]], but the diversification axis is "who asks / who writes" rather than instruction complexity.
- Useful when you need breadth in SFT or preference-prompt generation but do not have a strong seed corpus.
