---
chapter: ch-29c
course: llm-training
phase: read
excerpt_of: "Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards, arXiv:2505.24760v2 (2025-10-20; NeurIPS 2025 Datasets and Benchmarks)"
source_url: https://arxiv.org/abs/2505.24760
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug reasoning-gym). Values read from the v2 PDF on 2026-09-15."
---

# Excerpt: Reasoning Gym — procedural generators with verifiers

**Authors:** Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, Andreas Köpf (Open-Thought; Scale AI; UCL).

## Library (Abstract, §2)
- "over 100 data generators and verifiers" across algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and games.
- Design principles: (P1) algorithmic verifiability; (P2) large solution spaces; (P3) parametric difficulty control. Parameters are difficulty, structural, and stylistic.
- The authors state procedural generation "eliminates memorization concerns by ensuring that no two generated instances are identical" (§1).

## Zero-shot (§3, Fig. 3)
- Hard configurations: o3-mini 63.5%, DeepSeek-R1 59.5%, Grok 3 Mini 55.1%, Llama 4 Maverick 41.5%, Claude 3.5 Sonnet 40.3%, Gemma 3 27B 20.3%.
- Easy → hard drops for o3-mini: code −71.9%, graphs −33.8%, geometry −33.1%, algorithms −25.6%.

## Transfer with RLVR (§4; Qwen2.5-3B-Instruct, GRPO; format reward 0.2, accuracy reward 1.0; about 1,500 A6000 hours for reported results)
- Intra-domain, held-out task in the trained category (Table 1, Acc@3): algebra 5.0 → 16.7; algorithmic 52.3 → 59.7; arithmetic 89.7 → 96.0; cognition 40.3 → 42.3; games 0.0 → 3.3.
- Cross-domain (Table 2): RG-Algorithmic: algebra 23.83 → 52.89, arithmetic 29.56 → 45.17, geometry 0.83 → 23.17. RG-Algebra: ARC 6.49 → 4.18, games 8.40 → 9.23. RG-Logic: games 8.40 → 7.64, cognition 11.62 → 24.94. RG-Games: ARC 6.49 → 4.26, algebra 23.83 → 45.61.
- External (Table 3): RG-Math (800 GRPO steps on algebra, arithmetic, geometry): GSM8K 76.2 → 76.7; MATH 48.5 → 58.2; BBH 8.68 → 16.34. MMLU-Pro Math (Table 4): RG-Algorithmic 54.63 → 53.89, RG-Math 54.63 → 60.25.

## Curriculum (§5, Table 5)
- Curriculum raises difficulty when performance exceeds 70% over 20 training steps; the non-curriculum run samples all levels uniformly; one epoch each; 50 holdout examples per level.
- Spell Backwards length 4: baseline 12.00, non-curriculum 30.00, curriculum 70.67. Mini Sudoku 8-10 empty cells: 0.00 / 6.67 / 20.00. Count Primes 100-500: 12.00 / 4.00 / 30.67.
