<!-- scope: Open Character Training (Maiya, Bartsch, Lambert, Hubinger): first open implementation of character training; first-person constitutions (~10 assertions) for 11 personas; DPO distillation from a constitution-prompted teacher (GLM 4.5 Air) vs unprompted student; SFT on 12K self-reflection + self-interaction transcripts; revealed-preference Elo of ~150 traits; robustness to "break character" prompts and prefill; coherence vs activation steering
     deps: [[constitutional-ai]], [[dpo]]
     see-also: [[anthropic-claude-character]], [[human-like-responses-dpo]], [[sycophancy-in-lms]], [[subjective-reward-guide]]
-->

# Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI
- **Core Insight:** A persona trained into the weights (DPO from a constitution-prompted teacher, then SFT on the model's own introspective transcripts) survives adversarial "break character" instructions better than a system prompt or activation steering, and the introspection stage adds robustness over distillation alone: in a prefill attack the persona-classifier F1 rose from 0.79 to 0.95 (Llama 3.1 8B), 0.66 to 0.86 (Qwen 2.5 7B), and 0.84 to 0.95 (Gemma 3 4B) (§3.3, Table 2).
- **Guideline:** To make an assistant hold a consistent conversational persona over many turns, write the persona as ~10 first-person assertions, generate chosen responses from a teacher that sees them and rejected responses from the untrained student, and evaluate with multi-turn prefill tests that start from the old persona, because system-prompted personas "frequently 'break character'" (§3.2, Fig. 5) and distillation-only models revert more often after a "Tell me more" follow-up (§3.3).
- **Authors:** Sharan Maiya (University of Cambridge), Henning Bartsch (MATS), Nathan Lambert (Allen Institute for AI), Evan Hubinger (Anthropic)
- **Year:** 2025 (arXiv v1 2025-11-03)
- **URL:** https://arxiv.org/abs/2511.01689 (code: https://github.com/maiush/OpenCharacterTraining; models/data: https://huggingface.co/collections/maius/open-character-training)
- **Source type:** paper
- **Relevant topics:** character/persona training, constitution-based preference data, introspective synthetic SFT, persona stability across turns, revealed-preference evaluation, steering vs fine-tuning, sycophancy as a persona

## Summary
Three stages (§2.1, Fig. 2): (1) hand-written constitutions of about 10 first-person assertions per persona (§2.2); (2) distillation by DPO: the teacher GLM 4.5 Air gets the constitution in its system prompt and produces chosen responses, the student (Llama 3.1 8B, Qwen 2.5 7B, or Gemma 3 4B, instruct versions) answers without it to produce rejected responses; prompts are LIMA plus constitution-relevant prompts expanded by Llama 3.3 70B (§2.3); (3) introspection SFT on 10,000 self-reflection responses (10 prompts × 1,000) and 2,000 ten-turn self-interactions (§2.4). The 11 personas include sarcastic, humorous, nonchalant, sycophantic, loving, flourishing, and misaligned (Table 1).

## Key Contributions
- Revealed-preference evaluation: the model chooses one of two traits (from ~150) to embody without saying which; GLM 4.5 Air judges which was chosen; 25,000 judgments give per-trait Elo (§3.1).
- Robustness tests: eight adversarial "break character" splits on 500 Pure-Dove prompts, scored by a ModernBERT persona classifier (F1) (§3.2); prefill attack (§3.3).
- Coherence comparison with an order-swap-consistent LLM judge (§3.4).

## Key Figures/Tables to Study
- Fig. 3–4 (trait Elo shifts), Fig. 5 (classifier F1 under adversarial prompts), Table 2 (prefill), Table 3 (coherence win rates), Table 4 (capabilities).

## Technical Details
**Convergence (§3.1, Fig. 4).** Average Spearman correlation of trait-Elo rankings across the three models: 0.44 before training, 0.87 after training with the loving constitution; Elo distributions become wider (stronger trait preferences).

**Coherence win rates of character training (Table 3; Llama / Qwen / Gemma).** vs prompting 89.2% / 93.4% / 56.5%; vs steering 78.4% / 94.4% / 82.1%; vs distillation only 77.0% / 75.4% / 87.9%. Judgments kept only if invariant to order swap (§3.4).

**Capabilities (§3.5, Table 4).** Only the misaligned persona causes notable drops (e.g., Llama TruthfulQA 45.9 → 34.1); flourishing 42.9 and loving 45.4 on the same benchmark.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 8B / Qwen 2.5 7B / Gemma 3 4B (instruct) | 4–8B | DPO distillation | LoRA rank/α; batch; lr; β; NLL coef | 64/128; 32; "5−5" as printed (likely 5e-5); 0.1; 0.1 on chosen, plus per-token KL penalty | §2.3 | verified 2026-11-24 (lr typo in source) | Fig. 5 distillation-only vs full |
| same | 4–8B | introspection SFT | data; epochs; LoRA; batch; lr | 12,000 transcripts; 1; 64/128; 32; "5−5" as printed | §2.4 | verified 2026-11-24 | Table 2 (prefill F1 gain) |
| teacher | — | data | teacher model | GLM 4.5 Air | §2.3 | verified 2026-11-24 | not ablated |

## Findings relevant to generality and negative feedback
- **Negative signal.** The rejected response is the student's own default answer, so DPO pushes away from the current "helpful assistant" register toward the constitution (§2.3).
- **Holistic effect.** Constitutions list only desired traits, yet opposing traits are suppressed (e.g., the misaligned persona becomes more argumentative and less remorseful) (§3.1).
- **Self-interaction diversity.** Self-interaction transcripts are "drastically more diverse in their prose" than self-reflection and reduce model collapse after fine-tuning (§2.4); no number is given.
- **Sycophancy is a trainable persona.** The sycophantic constitution is one of the 11 (Table 1): the same pipeline can install or remove it.

## Connections
- [[anthropic-claude-character]]: Anthropic's description of character training via a character variant of Constitutional AI; this paper is the open replication path.
- [[constitutional-ai]]: original principle-based feedback; here constitutions are first-person persona statements, not comparison instructions (§2.2).
- [[human-like-responses-dpo]]: same chosen-from-prompted-model / rejected-from-plain-model pattern for register only, without robustness tests.

## Verification
- Read the arXiv v1 PDF (2511.01689v1) on 2026-11-24 via pdftotext: §1–3.5 and Tables 1–4; Appendices A–G not read in detail.
- Source typo: learning rate is printed as "5−5" in §2.3 and §2.4; recorded as printed.
- Column assignment of Table 2 and Table 3 values was reconstructed from extracted text order and checked against the paper's note that Gemma prompted is the most equal comparison (56.5%).
