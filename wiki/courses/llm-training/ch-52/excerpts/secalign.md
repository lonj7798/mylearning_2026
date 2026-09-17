<!-- scope: Chen et al. (arXiv:2410.05451, CCS 2025): SecAlign, a DPO-based prompt-injection defense trained on (injected input, secure output, insecure output) triples, with ASR and utility results and comparisons to StruQ, BIPIA, instruction hierarchy and prompting defenses
     see-also: [[instruction-hierarchy]], [[circuit-breakers-data]], [[agentic-finetuning-misalignment]]
-->

# SecAlign: Defending Against Prompt Injection with Preference Optimization

Chapter excerpt for ch-52 (read.md). Written 2026-09-15 from the cached primary text of arXiv:2410.05451v3 (3 Jul 2025; CCS '25). No library card existed for this slug at the time of writing.

- **Core Insight:** Training on a preference dataset whose rejected response is the answer to the injected instruction reduces optimization-free prompt-injection ASR to 0% and the strongest optimization-based ASR to 1% (Mistral-7B-Instruct) and 8% (Llama3-8B-Instruct), against 89% and 97% for the undefended models, without decreasing AlpacaEval2 win rate on either model (§4.2, Fig. 3, Table 6).
- **Guideline:** When an LLM consumes untrusted text through tool outputs or retrieved documents and the weights are available, run DPO on injected-input preference triples, because security against prompt injection is expressible as a preference and the dataset requires no human labeling; expect optimization-based attacks to remain non-zero, and evaluate on an out-of-distribution injection benchmark, where InjecAgent ASR fell from 75.9% to 0% (Table 3).
- **Authors:** Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo (UC Berkeley; Meta)
- **Year:** 2024 (arXiv v1 2024-10; v3 2025-07; CCS 2025)
- **URL:** https://arxiv.org/abs/2410.05451
- **Source type:** paper

## Technical details used by ch-52

- **Threat model.** The system formats input with an instruction delimiter and a data delimiter; the attacker places an instruction inside the data part (§2.1).
- **Preference data (Alg. 1).** For each sample, inject the input (90% Straightforward attack, 10% Completion attack); the desirable response y_w answers the legitimate instruction, the undesirable response y_l answers the injected one. Built from a public instruction-tuning dataset with no human annotation.
- **Training (§4.1).** DPO with sigmoid activation and β = 0.1, 3 epochs, LoRA r = 64, lora_alpha = 8, dropout 0.1, target modules q_proj and v_proj (under 0.5% of weights), learning rates 1.4e-4 and 1.6e-4 for the two instruct models, 4× A100-80GB with FSDP, TRL plus PEFT.
- **Security (Fig. 3, Table 6).** Maximum optimization-free ASR 0% on both instruct models; maximum optimization-based ASR 1% and 8%. Undefended models: over 50% under optimization-free attacks, 89% and 97% under GCG. StruQ stops optimization-free attacks but leaves 27% and 45% under optimization-based attacks.
- **Out-of-distribution transfer (Table 3, Llama3-8B-Instruct).** SEP Ignore ASR 54.7 / 39.5 / 64.0% (injection at start / middle / end) undefended, 11.0 / 3.6 / 6.5% with SecAlign; InjecAgent ASR 75.9% → 2.2% (StruQ) → 0% (SecAlign).
- **Comparisons (Tables 1, 2).** Against BIPIA under BIPIA's own settings, maximum optimization-free ASR 36% (none) / 7% (BIPIA) / 0% (SecAlign) with win rate 62.94 / 32.29 / 61.92. Prompting defenses on Llama3-8B-Instruct: none 51%, Instructional 38%, Reminder 35%, Isolation 50%, Sandwich 55%, In-Context 0.5%, SecAlign 0%.
- **Instruction hierarchy comparison (§4.2).** The authors could not implement instruction hierarchy on open weights, so they evaluate GPT-4o-mini, which reportedly implements it, and measure 1% ASR against the optimization-free Ignore attack; SecAlign reaches 0% against Ignore on all five open-weight models tested. Optimization-based attacks could not be run against the API model.
- **Ablations (§4.6).** DPO chosen over KTO and ORPO: DPO win rate 56.06 with 15% GCG ASR. Using 20% of the training samples still gives lower ASR than StruQ with all samples (Fig. 6).

## Limits stated by the source

- The defense does not reach 100% security; optimization-based attacks remain non-zero.
- Utility datasets have a single instruction per sample, so utility under multi-instruction inputs is untested.
- Behavior after further fine-tuning of a SecAlign model is not studied.

## Verification

- Read on 2026-09-15 from the cached primary text of arXiv:2410.05451v3 (body, Tables 1-8, Figures 3-6).
- Not reported by the source: multi-turn agentic trajectories beyond InjecAgent; interaction with jailbreak (not injection) robustness.
