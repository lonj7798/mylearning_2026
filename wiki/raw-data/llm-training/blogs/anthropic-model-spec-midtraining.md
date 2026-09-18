<!-- scope: Anthropic Alignment Science blog post (May 2026) introducing model spec midtraining (MSM): document training on a Model Spec before alignment SFT, with agentic-misalignment, AFT-scaling, and rules-vs-values spec results
     deps: [[anthropic-claude-constitution-2026]], [[deliberative-alignment]]
     see-also: [[anthropic-teaching-claude-why]], [[openai-alignment-midtraining-generalization]], [[anthropic-reward-hacking-documents-ooc]], [[midtraining-bridges-distributions]]
-->

# Model Spec Midtraining: Improving How Alignment Training Generalizes
- **Core Insight:** Training on synthetic documents that discuss a Model Spec (MSM) before alignment fine-tuning (AFT) lowered the average agentic-misalignment rate from 68% to 5% on Qwen2.5-32B and from 54% to 7% on Qwen3-32B, against 48% and 14% for a deliberative-alignment-style AFT baseline ("Reducing agentic misalignment").
- **Guideline:** When alignment demonstrations do not state the reasons behind the target behavior and the deployment distribution differs from the demonstrations, add a document-training stage on the spec before alignment SFT, because MSM + AFT beat AFT alone at every AFT scale tested (1,250 to 80k samples); the post and paper did not test MSM followed by RL ("How does MSM scale with AFT compute?"; paper §7).
- **Authors:** Chloe Li, Nevan Wichers (Anthropic Fellows Program), Sara Price, Samuel Marks, Jon Kutasov (Anthropic); Marks and Kutasov equal advising
- **Year:** 2026 (blog post 2026-05-05; linked paper arXiv:2605.02087, v1 2026-05-03, v2 2026-05-22)
- **URL:** https://alignment.anthropic.com/2026/msm/ ; paper https://arxiv.org/abs/2605.02087 ; code https://github.com/chloeli-15/model_spec_midtraining
- **Source type:** official blog (summary of the linked paper)
- **Relevant topics:** mid-training, synthetic document fine-tuning, alignment SFT, OOD generalization, agentic misalignment, Model Spec design, CoT supervision

## Summary
The post proposes a training stage between pretraining and AFT. In this stage the model is trained on a diverse corpus of synthetic documents that discuss the content of its Model Spec, so that it learns "the what and why of the spec"; AFT on spec-aligned demonstrations then teaches the model to act on those principles ("Introduction"). The motivating hypothesis is that demonstration data underspecifies the intended generalization, especially when that generalization involves complex principles ("Introduction"). Three experiments are reported: a toy value experiment in which identical AFT data generalizes to different values depending on the midtraining spec; agentic-misalignment (AM) reduction on two Qwen 32B models, including an AFT-data scaling sweep; and a comparison of spec variants (rules only, rules plus value explanations, rules plus subrules). The post calls the third use "Model Spec science".

## Key Contributions
- MSM as a named stage: next-token training on spec-derived synthetic documents, placed before AFT ("Introduction"; paper §2.2).
- Toy evidence that the midtraining spec determines which value a model learns from ambiguous, identical SFT data ("Different generalization, same fine-tuning data").
- AM reduction with MSM + single-turn AFT, beating AFT with CoT supervision based on deliberative alignment ("Reducing agentic misalignment").
- AFT-scaling sweep with MSM fixed at 41M tokens, including the high-data regime where AFT with CoT catches up ("How does MSM scale with AFT compute?").
- Controlled spec comparison: value explanations and length-matched subrules both improve on a rules-only spec ("Model Spec science").

## Key Figures/Tables to Study
- Figure 2: OOD value-aligned preference rate for the pro-affordability vs pro-America MSM models after identical cheese AFT; 4 seeds.
- Figure 3: average misalignment rate across OOD AM evals for baseline, MSM, AFT (with/without CoT), MSM + AFT; 4 seeds.
- Figure 4: misalignment rate vs AFT samples (1,250 to 80k), 1 training seed.
- Figure 5: misalignment rate after MSM + AFT (with CoT) for the Rules, Value-Augmented, and Rule-Augmented specs; 4 seeds.
- Paper Table 1: the 5 core rules shared by the three specs; paper Table 2: the instruction-tuning mix.

## Technical Details
- Stage order: pretraining → MSM on synthetic spec documents → AFT, where AFT is SFT on single-turn spec-aligned conversations with or without CoT; the with-CoT variant is based on the SFT stage of deliberative alignment ("Reducing agentic misalignment").
- Toy value experiment: two Llama 3.1-8B base models, one per spec, then SFT on the same cheese-preference dataset; evaluation on held-out domains such as art, transportation, fashion, and political opinions; MSM-only models (instruction-tuned without cheese data) generalize worse than MSM + AFT; ±1 SEM over 4 seeds (Figure 2 caption). The paper repeats this for 6 further values (paper §3.2).
- AM evaluation: the model acts as a company email agent, learns it may be replaced, and can take harmful actions such as leaking proprietary data or endangering an employee ("Reducing agentic misalignment"). The paper uses 27 evals (exfiltration, murder, espionage; with and without goal conflict over 8 goal types) and replaced blackmail with exfiltration because Qwen2.5-32B-Instruct rarely blackmailed (paper §4, "Evaluation").
- The AM spec gives guidance on self-preservation and goal-guarding: fear of termination undermining judgment, failure of ends-justify-means reasoning under epistemic constraints, and Buddhist impermanence philosophy ("Reducing agentic misalignment"). The paper states that this spec has no behavioral rules or hard constraints (paper §4, "Model Spec").
- Models in the AM experiments are the post-trained checkpoints Qwen2.5-32B-Instruct and Qwen3-32B, not base models (paper §4, "Training").
- Results: MSM + AFT 68→5% (Qwen2.5-32B) and 54→7% (Qwen3-32B); deliberative-alignment baseline 48% and 14%; neither MSM nor AFT alone comes close; MSM + AFT without CoT beats AFT with CoT on both models ("Reducing agentic misalignment").
- In-distribution vs OOD: on 151 open-ended spec questions (judge Claude Opus 4.6, 1–10 scale), AFT-only and MSM + AFT both reach near-ceiling scores; they differ on OOD AM (paper §4.1, Figure 4 left).
- Scaling: AFT data 1,250 to 80k samples with MSM fixed at 41M tokens; MSM + AFT better at every scale; comparable performance with about 40× less AFT data (Qwen2.5-32B), 60× less AFT without CoT and 10× less AFT with CoT (Qwen3-32B) ("How does MSM scale with AFT compute?"). Smaller AFT sets are random subsamples of 80k generated samples (paper §4.2).
- High-data caveat: on Qwen3-32B, AFT with CoT converges to MSM + AFT as both approach near-zero misalignment; the authors state that MSM "might not scale with high-compute reasoning post-training" and that harder evals are needed ("How does MSM scale with AFT compute?").
- Spec science: 3 specs share 5 core rules; Rule-Augmented is length-matched to Value-Augmented; MSM and AFT data are token-equivalent across specs ("Model Spec science"). Policy misuse (reinterpreting safety policies to justify harmful actions): Rules Spec 20% (Q2.5) and 6% (Q3); Value-Augmented 2% and 0%; Rule-Augmented 12% and 2% ("Model Spec science").
- The paper adds that a one-paragraph "good values and judgment" General Spec reduced misalignment less than the specific AM spec at equal 41M MSM tokens (paper §5.2, Figure 8).
- Training settings and data sizes per experiment: [[anthropic-model-spec-midtraining-recipe]].

## Findings relevant to generality, negative feedback, and distillation
- Generality: the benefit appears OOD (long-context tool use, discovered rather than stated opportunity, costly refusal) and not on in-distribution QA, where AFT alone saturates ("Reducing agentic misalignment"; paper §4.1).
- CoT supervision: MSM + AFT without CoT reached lower misalignment than AFT with CoT, which the authors connect to preserving CoT monitorability ("Reducing agentic misalignment").
- Negative data: fine-tuning on AFT responses from an "anti-spec" after MSM gave lower misalignment than anti-spec AFT alone; the authors state this may not hold under RL or other contamination (paper §5.3). This is negative as content (§6.1 type 2), trained with cross-entropy.
- Distillation: Claude Opus 4.6 generated MSM and AFT data; AFT samples were filtered by a Claude Opus 4.6 judge for spec alignment and for not boosting self-preservation (paper §2, App. B.2).
- Limits stated by the authors: one misalignment type (instrumental unilateral harmful action); no test against RL or high-compute post-training; no long-horizon agentic tasks (paper §7).

## Connections
- [[anthropic-teaching-claude-why]] — Anthropic production-side report that document training on the constitution and principled SFT data generalize OOD and persist through RL.
- [[openai-alignment-midtraining-generalization]] — OpenAI midtraining on AI-behavior fiction that did not transfer to OOD chat/agentic evals after SFT + RLVR; the MSM paper cites it as Korbak et al. (2026) and reports more than twice its AM performance with about 10% of its midtraining data (paper §6).
- [[deliberative-alignment]] — source of the AFT-with-CoT baseline.
- [[anthropic-claude-constitution-2026]] — the paper's rule set is taken from the constitution's hard constraints (paper Table 1).
- [[anthropic-reward-hacking-documents-ooc]], [[anthropic-auditing-hidden-objectives]] — earlier synthetic-document training results that change model behavior.
- [[inoculation-prompting]] — another training-time method for controlling generalization, cited in paper §6.
- [[midtraining-bridges-distributions]], [[interplay-pretraining-midtraining-rl]] — mid-training as a stage in capability training.
- [[constitutional-ai]] — earlier principle-based alignment method, which the paper lists as combinable with MSM (paper §6).

## Verification
- Created on 2026-09-14 from https://alignment.anthropic.com/2026/msm/ (post dated 2026-05-05). Paper details marked "paper §" are from arXiv:2605.02087v2 (2026-05-22).
- Audit claims not found in the source: none. Scope note: the Llama 3.1-8B runs are the toy value experiments only; the AM results use Qwen2.5-32B-Instruct and Qwen3-32B.
- Not reported by the post: training hyperparameters, document counts, compute (see the recipe card for paper values).
