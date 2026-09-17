---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: primary source arXiv:2311.07911v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2311.07911
created_at: "2026-09-15"
---

# Excerpt: Instruction-Following Evaluation for Large Language Models (IFEval)

**Paper:** Zhou, Lu, Mishra, Brahma, Basu, Luan, Zhou, Hou (Google; Yale University). arXiv v1 2023-11-14. Source type: paper.

## Construction (§2, §2.1, Table 1)
- A *verifiable instruction* is an instruction whose satisfaction can be checked by a program, for example "write in more than 400 words" or "mention the keyword AI at least 3 times".
- 25 types of verifiable instructions; 541 prompts, each carrying one to three verifiable instructions appended to a base prompt; illogical prompts removed and remaining prompts rephrased and manually checked (§2.1).
- The authors state that very few instructions are 100% verifiable (§2).

## Metrics (§2.2, §3)
- `is_followed(resp, inst)` returns True or False per instruction; the *strict* metric uses it directly (Eq. 1).
- The *loose* metric applies eight transformations to the response and counts the instruction as followed if any transformed response passes (Eq. 2): remove markdown font modifiers (`*`, `**`), remove the first line, remove the last line, every pair, all three, and the identity.
- Four reported numbers per model: prompt-level strict, instruction-level strict, prompt-level loose, instruction-level loose (§3).
- The authors state that the loose criterion reduces false negatives and introduces false positives (for example a word-count instruction passing after the first line is removed), and treat it as a complement to the strict criterion (§2.2).

## Results (Table 3)
- GPT-4: prompt-level strict 76.89%, instruction-level strict 83.57%, prompt-level loose 79.30%, instruction-level loose 85.37%.
- PaLM 2 S: 43.07%, 55.76%, 46.95%, 59.11%.
- The caption states the two models are not directly comparable because of the difference in parameter count.

## Verification
- Read on 2026-09-15 against arXiv:2311.07911v1 PDF text (Abstract, §2-§3, Tables 1-3).
