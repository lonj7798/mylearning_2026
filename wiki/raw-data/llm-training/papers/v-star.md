<!-- scope: verifier-guided self-training for reasoning — V-STaR algorithm and data loop
     deps: [[star]], [[rest-em]]
     see-also: [[lets-verify]], [[prm800k]], [[math-shepherd]]
-->

# V-STaR: Training Verifiers for Self-Taught Reasoners
- **Core Insight:** Self-improvement gets much stronger when wrong reasoning traces are kept and turned into verifier training data; the verifier then becomes the selection mechanism that decides which sampled solutions are worth learning from.
- **Guideline:** In reasoning self-training, do not discard failed samples. Use outcome verification to label correct and incorrect traces, train a verifier on the contrast, and let that verifier guide both data filtering and inference-time selection.
- **Authors:** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.06457
- **Relevant topics:** verifier training, self-training, chain-of-thought, EM-style reasoning improvement

## Abstract
V-STaR extends the STaR self-training line by training a verifier alongside the reasoner. Instead of using only successful model-generated solutions as new supervision, it also uses the unsuccessful ones to learn a verifier that can score candidate reasoning traces. The verifier is then used to rank or filter future samples, producing a tighter self-improvement loop than plain self-training on correct traces alone.

## Key Contributions
- Turns failed reasoning traces into useful supervision by framing verifier learning as a central part of the self-training loop.
- Combines **reasoner improvement** and **verifier improvement** in an iterative bootstrapping procedure.
- Shows that verifier-guided selection is stronger than plain STaR-style self-training on accepted solutions only.
- Connects self-training for reasoning to a broader process-supervision view later seen in PRM-style work.

## Key Figures/Tables to Study
- **Pipeline figure:** reasoner generates candidates, outcome verifier labels them, verifier is trained, then the verifier selects future candidates.
- **Iteration ablation:** study how verifier quality and reasoner quality co-improve across rounds.
- **Selection comparison table:** compare random / majority / verifier-guided candidate selection.

## Technical Details
- **Candidate generation:** sample multiple chain-of-thought solutions per problem from the current reasoner.
- **Outcome labeling:** use executable or answer-level verification to mark which samples are correct and which are not.
- **Verifier training signal:** build preference-style or classification-style supervision from the contrast between successful and failed traces.
- **Inference use:** at test time, generate several candidate solutions and let the verifier pick the best one instead of trusting the first sample.
- **Training loop:** the selected high-scoring traces are fed back into the reasoner for the next round, giving an EM-like self-improvement cycle.
- **Why it matters:** STaR and ReST-EM mainly exploit successful traces. V-STaR extracts value from the failures too, which is the key algorithmic improvement.

## Risks + gotchas
- The approach assumes access to an outcome verifier or executable check; it is much harder to apply where correctness is fuzzy.
- A weak verifier can reinforce the wrong traces and create a bad feedback loop.
- Diversity still matters: if candidate generation collapses too early, the verifier has little real choice to make.

## Connections
- Direct successor to [[star]] and sibling of [[rest-em]] in the EM-style self-training family.
- Foreshadows process-supervision datasets such as [[prm800k]] and [[math-shepherd]], where intermediate reasoning quality becomes a learnable signal.
- Closely related to [[lets-verify]], which pushes the verifier-first perspective further in later reasoning pipelines.
