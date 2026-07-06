<!-- scope: coverage checklist + doc-vs-reality reconciliation + gap log for course/on-policy-distillation
     deps: [[README]]
     see-also: [[insights]], [[wiki/courses/on-policy-distillation/outline]]
-->

# On-Policy & Off-Policy Distillation — Collection Plan

Target: enough verified source coverage to teach **post-training as distribution matching** — off-policy vs on-policy distillation, forward vs reverse KL, when each wins — and to design an on-policy-distillation strategy for the learner's `boson-agent-synthetic-data-dev` SFT pipeline. Status as of 2026-07-06: 9 excerpts written, outline drafted and approved. No blocking gaps for v1.

## Coverage checklist

| Area | Source | Excerpt | Status |
|------|--------|---------|--------|
| The primary OPD reference: mechanism, per-token reverse KL, compute numbers | Lu / Thinking Machines (2025) | [[tm-on-policy-distillation]] | ✅ verbatim |
| Distribution-matching framing: SFT/RL/OPD, forward/reverse KL | nrehiew (2025) | [[nrehiew-sft-rl-opd]] | ✅ verbatim |
| Classical KD: soft targets, temperature, dark knowledge | Hinton/Vinyals/Dean (2015) | [[hinton-knowledge-distillation]] | ✅ verbatim |
| Sequence-level KD: train on teacher-generated sequences | Kim & Rush (2016) | [[kim-rush-seqkd]] | ✅ verbatim |
| Exposure bias / compounding error / on-policy data | Ross/Gordon/Bagnell — DAgger (2011) | [[ross-dagger-exposure-bias]] | ⚠️ thesis (spend-limit) |
| On-policy LM distillation, GKD (λ / JSD-β) | Agarwal et al (2024) | [[agarwal-gkd]] | ✅ verbatim |
| Reverse-KL LLM distillation, policy gradient | Gu et al — MiniLLM (2024) | [[gu-minillm-reverse-kd]] | ✅ abstract-verbatim + thesis |
| Practical recipe: TRL GKDTrainer / GOLD, any model family | HuggingFace / TRL docs | [[hf-trl-gkd-recipe]] | ✅ verbatim |
| Industrial evidence: strong-to-weak distillation, the 10×-vs-RL numbers | Qwen3 Technical Report (2025) | [[qwen3-strong-to-weak-distillation]] | ⚠️ thesis (spend-limit) |

## Doc-vs-reality / contested-claims reconciliation (primary source wins)

| Popular narrative | What the primary source actually says | Resolve in |
|---|---|---|
| "Distillation = train a small model to copy a big one." | That is only **off-policy** KD. On-policy distillation grades the student's OWN rollouts with the teacher — the data comes from the student, not the teacher. | [[tm-on-policy-distillation]], [[agarwal-gkd]] |
| "SFT and distillation are different things." | SFT on teacher-generated sequences **is** off-policy sequence-level distillation; the line is the data source, not the loss name. | [[nrehiew-sft-rl-opd]], [[kim-rush-seqkd]] |
| "More teacher data fixes imitation learning." | No — exposure bias is a **distribution-shift** problem: behavioral cloning is O(T²), on-policy is O(T). The fix is on-policy data (DAgger), not more data. | [[ross-dagger-exposure-bias]] |
| "On-policy distillation is just a kind of RL." | It is on-policy **like** RL but **dense** like distillation: the reward is the teacher's per-token logprob (O(N) bits), not a sparse scalar (O(1) bits). | [[tm-on-policy-distillation]], [[nrehiew-sft-rl-opd]] |
| "Forward vs reverse KL is a detail." | Reverse KL is **mode-seeking** and "unhackable" (low KL ⇒ high teacher-desirable probability); forward KL is mode-covering and can average away prior capabilities. | [[tm-on-policy-distillation]], [[gu-minillm-reverse-kd]], [[nrehiew-sft-rl-opd]] |
| "On-policy distillation is always better." | Math/code often favor RL; OPD needs teacher-logprob access and risks entropy collapse. It wins for long sequences, continual learning, and cheap strong-teacher transfer. | [[nrehiew-sft-rl-opd]], [[tm-on-policy-distillation]] |
| "Student and teacher must share a tokenizer/family." | The GOLD/ULD recipe aligns token boundaries so a teacher of any family can grade the student — cross-tokenizer OPD. | [[hf-trl-gkd-recipe]] |
| "The boson pipeline is already on-policy." | Its **customer simulator** samples on-policy (11-model rotation), but the **seller student** is trained OFF-policy on generated transcripts — the exposure-bias gap OPD closes. | capstone (ch-07) |

## Gap log

- **Monthly-spend-limit interruption (2026-07-06):** the research fan-out (9 agents) hit an account spend limit mid-run. 7 agents had already fetched their sources verbatim and written excerpts before dying at the return step ([[tm-on-policy-distillation]], [[nrehiew-sft-rl-opd]], [[agarwal-gkd]], [[gu-minillm-reverse-kd]], [[hinton-knowledge-distillation]], [[kim-rush-seqkd]], [[hf-trl-gkd-recipe]]). The remaining two ([[ross-dagger-exposure-bias]], [[qwen3-strong-to-weak-distillation]]) were authored by the teacher as **thesis extracts** and are clearly marked as such — re-fetch the DAgger paper (arXiv:1011.0686) and Qwen3 report (arXiv:2505.09388) for verbatim quotes when budget allows.
- **Qwen3 Table 21 numbers** are reproduced via [[tm-on-policy-distillation]]'s citation of the report; verify directly against arXiv:2505.09388 when re-fetching.
- **MiniLLM** ([[gu-minillm-reverse-kd]]): abstract is verbatim; the exact reverse-KL / REINFORCE equations are corroborated from the PDF but flagged thesis-level, not guaranteed glyph-exact.
- **DistiLLM** (Ko et al 2024, skew KL / adaptive off-policy) is covered as a one-line follow-up inside [[gu-minillm-reverse-kd]], not a standalone excerpt. Promote only if the outline gives divergence-objectives its own chapter.
