<!-- scope: sequence-level KD — train the student on the teacher's beam-search (mode) output sequences instead of per-word soft labels
     deps:
     see-also: agarwal-gkd, tm-on-policy-distillation -->
# Sequence-Level Knowledge Distillation (Kim & Rush, EMNLP 2016)
- **Core Insight:** Instead of matching the teacher's per-word soft label distribution, train the student on the teacher's *generated* output sequences — approximate the intractable sequence-level distribution q(t|s) by its mode (the teacher's beam-search output), which concentrates most of the mass and transfers global sequence knowledge.
- **Guideline:** To distill a seq2seq model, run beam search with the teacher over the training set, take the top sequence, and re-train the student with plain cross-entropy on that teacher-generated dataset — this beats word-level KD and even lets you drop beam search at test time.
- **Source:** Yoon Kim, Alexander M. Rush. "Sequence-Level Knowledge Distillation." EMNLP 2016. arXiv:1606.07947 / ACL Anthology D16-1139. Abstract fetched verbatim from arxiv.org/abs/1606.07947; method equations, mode-mass numbers, and Table 1 BLEU figures extracted verbatim via pdftotext of aclanthology.org/D16-1139.pdf.
- **Relevant chapters:** off-policy distillation, sequence-level KD, seq2seq/NMT compression, dark-knowledge origins

## Thesis (abstract, verbatim)
"We demonstrate that standard knowledge distillation applied to word-level prediction can be effective for NMT, and also introduce two novel sequence-level versions of knowledge distillation that further improve performance, and somewhat surprisingly, seem to eliminate the need for beam search (even when applied on the original teacher model). Our best student model runs 10 times faster than its state-of-the-art teacher with little loss in performance." (Kim & Rush 2016, abstract)

## Word-level KD (the baseline being improved on)
Standard Hinton-style multi-class cross-entropy applied at each token position, matching the teacher's per-word distribution q(t_j | s, t_{<j}):
`L_WORD-KD = − Σ_{j} Σ_{k in V} q(t_j=k | s, t_<j) · log p(t_j=k | s, t_<j)`
"Word-level knowledge distillation allows transfer of these local word distributions." (§3.2) The student can mix `L_WORD-KD` with the ground-truth `L_WORD-NLL`.

## Sequence-level KD (the key move: train on teacher-GENERATED sequences)
The true sequence-level objective sums over an exponential number of complete sequences and is intractable:
`L_SEQ-KD = − Σ_{t in T} q(t | s) · log p(t | s)`
"Note that L_SEQ-KD is inherently different from L_WORD-KD, as the sum is over an exponential number of terms." (§3.2)

The approximation: **replace the teacher distribution q with its mode**, then find the mode with beam search:
`q(t | s) ≈ 1{t = argmax_{t in T} q(t | s)}`  →  `L_SEQ-KD ≈ − log p(t = ŷ | s)`
"where ŷ is now the output from running beam search with the teacher model." (§3.2)

Why the single-mode approximation is defensible — the mass is concentrated: "in experiments we find that with beam of size 1, q(ŷ | s) (on average) accounts for 1.3% of the distribution for German → English, and 2.3% for Thai → English." And after Seq-KD training the student piles even more mass onto that mode: "on English → German the (approximate) argmax for the 2 × 500 Seq-KD model (on average) accounts for 16.9% of the total probability mass, while the corresponding number is 0.9% for the baseline." (§4)

Procedure (§3.2): "(1) train a teacher model, (2) run beam search over the training set with this model, (3) train the student network with cross-entropy on this new dataset." Step 3 is ordinary NLL training — just on teacher-generated targets instead of gold.

## Sequence-level interpolation (Seq-Inter)
Blend teacher generations with the ground truth by picking, from the teacher's K-best beam (K=35), the candidate closest to the gold reference:
`ỹ ≈ argmax_{t in T_K} sim(t, y)`, with `sim` = smoothed sentence-level BLEU.
This targets a mixture `D_SEQ-Inter ∼ (1−α)D + α·q(t | s)` — a single training sequence that is both high-probability under the teacher and close to gold, avoiding doubling the data or training on conflicting (gold vs. teacher) targets for the same source.

## Key numbers (Table 1, English→German newstest2014, greedy K=1 / beam K=5 BLEU)
- Teacher 4×1000 (221M params): 17.7 / 19.5
- Student 2×500 baseline (84M): 14.7 / 17.6
- Word-KD: 15.4 / 17.7 (+0.7 / +0.1)
- **Seq-KD: 18.9 / 19.0 (+4.2 / +1.4)** — greedy Seq-KD (18.9) beats the beam-search baseline (17.6) and rivals the teacher's beam (19.5)
- Seq-KD + Seq-Inter + Word-KD: 18.8 / 19.2 (best combination)
- Best student = 10× faster than teacher; with weight pruning, 13× fewer params than teacher at −0.4 BLEU.
Word-KD and Seq-KD are described as orthogonal: "Word-KD is transferring knowledge at the local (i.e. word) level while Seq-KD is transferring knowledge at the global (i.e. sequence) level." (§4)

## Why it matters here — the off-policy framing (thesis, not verbatim)
Seq-KD is an early, influential move toward distilling on *generated sequences* rather than per-word soft labels — this is what later on-policy methods build on. But it is still **off-policy**: the training sequences come from the *teacher's* beam search (the teacher's mode), fixed once before student training begins. The student never sees or corrects its *own* generations, so it is never exposed to the states it will actually visit at inference — the classic exposure-bias / distribution-shift gap. The missing "on-policy" step — sampling from the *student*, then scoring those student trajectories with the teacher — is exactly what [[agarwal-gkd]] (Generalized KD, student-generated sequences graded by the teacher) and [[tm-on-policy-distillation]] add. Read Kim & Rush as the "off-policy, teacher-mode sequences" corner of that design space.
