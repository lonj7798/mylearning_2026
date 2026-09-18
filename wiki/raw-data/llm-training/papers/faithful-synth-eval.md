<!-- scope: former synthesis card on auditing synthetic data for distribution preservation; no single primary artifact exists
     deps: [[model-collapse]]
     see-also: [[strong-model-collapse]], [[prismatic-synthesis]], [[genie]]
-->

# Faithful Synthetic-Data Evaluation: Detecting Distribution Preservation vs Corruption — Synthesis Card
- **Source type:** none (synthesis card without a primary artifact)
- **Status:** no verifiable primary source. Chapters must not cite this card; cite the works listed below instead.

> **No verifiable primary source.** The previous card described itself as an aggregate of "2024/2025" papers and named
> no paper, report, or code that defines its four-axis audit (tail recall, embedding-cluster occupancy, gradient-space
> coverage, external-verifier flag rate). Searched: the card's two URLs (arXiv:2510.16657 v3, arXiv:2509.16499 v3);
> an arXiv title search for "Faithful Synthetic-Data Evaluation" (one result, arXiv:2102.08921, a different 2021 work);
> the card's OpenReview id Xr5iINA3zU (page not accessible: browser verification); arXiv:2410.16713, the arXiv
> version of the title "Collapse or Thrive?". None of these works proposes the four-axis protocol, and none reports
> that average perplexity hides tail degradation in a synthetic text corpus. A web search could not be run in this
> session (search budget exhausted).

## Verifiable pointers to related works
Each item below was read at the stated locus on 2026-09-14. Each item is a fact about that work only.

- **Verifier-filtered synthetic retraining.** Bingji Yi, Qiyuan Liu, Yuwei Cheng, Haifeng Xu, "Escaping Model
  Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence", arXiv:2510.16657
  (v1 2025-10; v3 2026-07 read). In linear regression, retraining on verifier-filtered synthetic data can improve the
  estimate in the short term through a bias-variance trade-off (Theorem 3.1, §3.2). Iterated retraining converges to
  the verifier's "knowledge center", not to the true parameter, unless the verifier has no bias; verifier selectivity
  changes the convergence rate but not the limit (Theorem 4.1, §4; §1). The abstract states that early gains "will
  plateau and may even reverse" unless the verifier is perfectly reliable. Experiments: a VAE on MNIST that starts
  from 500 real images improves over 40 rounds with verified retraining and degrades without it (Fig. 1, §5.2).
  SmolLM2-135M is fine-tuned for one epoch on 12.5% of the XSUM training set; each round an oracle verifier keeps
  the top 12.5% of generated summaries by ROUGE-1 against references. Over 15 rounds filtered retraining improves
  and then stabilizes, while unfiltered retraining fluctuates around its initial score (§5.3, Fig. 5). The analysis
  assumes a well-specified linear-regression setting (§6).
- **Generalization-to-memorization in self-consuming diffusion models.** Lianghe Shi, Meng Wu, Huijie Zhang, Zekai
  Zhang, Molei Tao, Qing Qu, "A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective",
  arXiv:2509.16499 (v3 2025-12 read). The setting is image diffusion models, not language models. On CIFAR-10 with a
  UNet DDPM retrained only on its previous generation's samples, models shift from generating new images to
  replicating training images (§3.1). The entropy of the synthetic training set falls over iterations (§3.2). Training-set
  entropy and the log of the generalization score have Pearson correlation 0.91 (§3.3, Fig. 4a). Entropy-based data
  selection (Greedy Selection, Threshold Decay Filter) lowers FID in recursive training (§4-5, Fig. 6).
- **Replace vs accumulate workflows.** Joshua Kazdan, Rylan Schaeffer, Apratim Dey, Matthias Gerstgrasser, Rafael
  Rafailov, David Donoho, Sanmi Koyejo, "Collapse or Thrive? Perils and Promises of Synthetic Data in a
  Self-Generating World", arXiv:2410.16713 (v4 2025-03 read). Three task settings: multivariate Gaussian estimation,
  kernel density estimation, and language-model fine-tuning. Replacing real data with successive synthetic
  generations collapses in all settings. Accumulating synthetic data alongside real data and training on all of it
  keeps test losses from diverging. Accumulating but training each generation on a fixed-size subset gives slow,
  gradual degradation (abstract). Whether OpenReview id Xr5iINA3zU is this paper was not confirmed.
- **Sample-level fidelity and diversity metrics.** Ahmed M. Alaa, Boris van Breugel, Evgeny Saveliev, Mihaela van
  der Schaar, "How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative
  Models", arXiv:2102.08921 (v1 2021-02; v2 2022-07 read). Defines a three-part metric (α-Precision, β-Recall,
  Authenticity) for fidelity, diversity, and generalization (copying of training data), estimated through sample-level
  binary classification, and a "model auditing" use that discards low-quality generated samples post hoc (abstract).

## Connections
- [[model-collapse]], [[strong-model-collapse]] — library cards for the model-collapse results this topic builds on.
- [[prismatic-synthesis]] — library card for the gradient-space diversity metric G-Vendi (not re-checked in this pass).
- [[genie]] — links here for the idea of filtering synthetic data by faithfulness.
- Chapters ch-23 and ch-51 link here; they should cite the works above or the library cards instead.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2510.16657 (v3); https://arxiv.org/abs/2509.16499 (v3);
  https://arxiv.org/abs/2410.16713 (v4); https://arxiv.org/abs/2102.08921 (v2); arXiv title search (one result).
- Corrections to the previous card version:
  - "Zhang et al. 2025, arXiv 2510.16657" → authors are Bingji Yi, Qiyuan Liu, Yuwei Cheng, Haifeng Xu (title page).
  - "external verification breaks the collapse loop even under repeated training"; "with a reliable external verifier,
    iterated synthetic training converges (no collapse)" → iterated retraining converges to the verifier's knowledge
    center; gains plateau and may reverse unless the verifier is perfectly reliable (abstract; Theorem 4.1).
  - "empirically in LLM text generation" → one language experiment: SmolLM2-135M on XSUM summarization with an
    oracle ROUGE-1 verifier (§5.3).
  - "Closer Look: increasing synthetic fractions shift models toward memorization-heavy regimes" → a study of image
    diffusion models; the transition is tied to falling training-set entropy over iterations (abstract; §3).
  - "Collapse or Thrive? (2025 openreview Xr5iINA3zU) — empirical tail behavior" → arXiv:2410.16713 studies replace,
    accumulate, and fixed-subset workflows; its text does not discuss distribution tails (full-text search).
  - Title "Faithful Synthetic-Data Evaluation: …" and year "2024-2025" → no artifact with this title was found.
- Removed as unsupported by any listed source:
  - The four audit axes as a consolidated protocol: rare-token recall, rare n-gram overlap, rare-concept recall,
    embedding-cluster occupancy, kNN diversity, verifier flag rate, drift-over-iteration monitoring.
  - "Averaged loss / perplexity hide tail degradation in synthetic corpora" as a finding of these papers.
  - "Reject corpora that drop in any of these even if average PPL looks fine."
  - "Mixture-ratio optima: analytic optimal real:synthetic ratios (He et al. 2025, Garg et al. 2025)."
  - "Accumulate, don't replace — the single most robust mitigation" (the ranking is not in any listed source).
  - "Mode collapse often appears in specific topic clusters before showing up globally."
  - "Verifier examples: answer matcher, unit tests, NLI entailment classifier, retrieval-grounded checker" attributed to
    these papers; "compound verifiers reduce bias"; "tail metrics are noisy for small corpora"; "G-Vendi rankings
    change with the proxy model"; "research is moving fast".
