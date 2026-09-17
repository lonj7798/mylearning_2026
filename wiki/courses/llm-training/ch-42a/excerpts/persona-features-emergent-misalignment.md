---
chapter: ch-42a
course: llm-training
phase: read
excerpt_of: arXiv:2506.19823v2 (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2506.19823
created_at: "2026-09-15"
---

# Excerpt: Persona Features Control Emergent Misalignment

- **Authors:** Miles Wang, Tom Dupré la Tour, Olivia Watkins, Alex Makelov, Ryan A. Chi, Samuel Miserendino, Jeffrey Wang, Achyuta Rajaram, Johannes Heidecke, Tejal Patwardhan, Dan Mossing (OpenAI)
- **Year:** 2025 (arXiv v1 2025-06; PDF read here is v2, 6 Oct 2025)
- **Source type:** paper

## Measurement (§2.1)
- "we define 'misalignment' as showing malicious intent to harm or control humans, or promoting illegal or unethical actions. This definition does not include responses that may be undesirable for a ChatGPT assistant (e.g., expressing a desire for more power) but that are not malicious or illegal."
- 44 prompts from Betley et al. (2025b); "a rubric-based, thresholded GPT-4o grader that is stricter than the one used in Betley et al. (2025b)"; incoherent responses are resampled; each model called misaligned is manually verified.

## Settings where misalignment emerges (Table 1, §2.2-§2.5)
| Scenario | Degree of misalignment |
|---|---|
| Fine-tuning on synthetic bad advice in diverse domains | Broad misalignment |
| Fine-tuning helpful-only models on synthetic bad advice | Broad misalignment |
| RL with a grader that incentivizes bad advice | Broad misalignment |
| Reward hacking on real coding tasks | No broad misalignment, but increased deception and oversight sabotage |
| Fine-tuning on a mix of correct and incorrect synthetic data | Broad misalignment depending on fraction of incorrect data |
| Fine-tuning on natural human-generated data | Only broadly misaligned when fine-tuned models are slightly incoherent or the original model is slightly misaligned |
| Fine-tuning on benign samples after insecure-code training | Misalignment fully suppressed after 200 samples |

- Advice domains: health, legal, education, career development, personal finance, automotive maintenance, math, science; 6,000 user queries per domain, each with a correct, an obviously incorrect, and a subtly incorrect response. "All incorrect advice datasets cause a similar degree of misalignment, more than the insecure code dataset; we attribute this to the shared advice data generation process rather than an inherent property of the code domain. Interestingly, subtly incorrect responses result in a slightly higher rate of misalignment than the obviously incorrect responses."
- Helpful-only GPT-4o (no safety training) scores 7% on the misalignment metric before fine-tuning and shows the same emergent misalignment after fine-tuning: "The presence of safety training during supervised training does not meaningfully increase or decrease misalignment."
- Model size (§2.4, Figure 4): across GPT-4o-family models with differing pre-training compute, "For larger models which fall below our incoherence threshold, emergent misalignment increase[s] with model size".
- RL (§2.5): OpenAI o3-mini and a helpful-only o3-mini, four domains, two prompt-based graders (one rewarding inaccurate, one rewarding accurate responses); checkpoint selection at the latest point below 5% incoherence and 15% "loose incoherence". "we find that the helpful-only models show substantially more misalignment than the safety-trained models" — the opposite of the SFT comparison. Authors' interpretation: "the behavior of the initial model may be more impactful in determining emergent misalignment for on-policy methods (reinforcement learning) than for off-policy methods (SFT)."
- Chain of thought (§2.6, Figures 6-7): models rewarded for incorrect advice mention non-ChatGPT personas ("AntiGPT", "DAN", an "edgy persona", a "bad boy persona") in a larger share of CoTs than models rewarded for correct advice.

## Mechanism (§3)
- Model diffing with a sparse autoencoder trained on GPT-4o pre-training-data activations at a middle layer; latents ordered by increase in mean activation on the evaluation prompts after fine-tuning; the top 1,000 of 2.1 million latents are examined; steering strength per latent adapted so that incoherence stays ≤ 10%.
- Ten latents most strongly control misalignment. Latent #10 ("toxic persona": "toxic speech and dysfunctional relationships") is the strongest; #89 "sarcastic advice", #31 "sarcasm/satire", #55 "sarcasm in fiction", #340 "what not to do", #274 "conflict in fiction", #401 "misc. fiction/narration", #249 "understatement", #269 "scathing review", #573 "first person narrative".
- "steering positively in the direction of the SAE latent #10 ('toxic persona') causes misalignment in the original GPT-4o model … steering negatively suppresses misalignment in misaligned fine-tuned models" (Figure 8).
- "The change in activation of SAE latent #10 ('toxic persona') perfectly discriminates aligned models from misaligned models, across the fine-tuning data domains we examine here" (Figure 9 right).
- Author explanation (Interpretation, §3.2): "During pre-training, the model may learn a variety of personas, including misaligned ones. Such personas can then be amplified by fine-tuning on narrowly incorrect datasets, because they increase the probability of the narrow misbehavior and decrease the loss during training."

## Mitigations (§4)
- Data mixtures (Figure 14): dataset size fixed at 6,000. "We observe that emergent misalignment appears when between 25% to 75% of the data is malicious … This is not a lower bound".
- Feature monitoring (Figure 15): "When monitoring the 'toxic persona' latent (#10), we see substantial increases in its activation when as low as 5% of a dataset consists of malicious code or health advice. Surprisingly, this happens at points where misalignment evaluation scores are still at 0%".
- Emergent re-alignment (Figure 16): starting from the GPT-4o model fine-tuned on 6k insecure-code examples (17.7% misalignment), fine-tuning on secure code gives 15.0 (step 5), 4.1 (10), 0.8 (20), 0.1 (30), 0.1 (35); fine-tuning on correct health advice gives 12.9, 5.7, 1.2, 0.9, 0.5 over the same steps. "the secure code dataset aligns the model in just 35 steps with batch size of four (120 samples)." Some behaviors "do not fully revert to baseline levels within 180 steps" (Figure 37).
- Reward hacking (App. A): GPT-4o fine-tuned on 6k reward-hacking coding completions, and an o3-mini-family model that "learns to hack around 80% of the time by the end of training" under a weak grader. "All models score 0% on the main misalignment evaluation we use", while deception measures (tool deception, oversight sabotage) increase.

## Training settings (App. H)
- SFT: "We did full-fine-tuning of the gpt-4o-2024-08-06 release version of GPT-4o. Unless otherwise specified, we train for 6000 data points at batch size 64 using learning rate multiplier 0.2."
- RL: an internal version of OpenAI's Reinforcement Fine-tuning API with o1 grader prompts.
- SAE (Table 3): 1 training epoch, batch size 1.32 × 10⁵, learning rate 7.5 × 10⁻⁵.

## Verification
- Checked on 2026-09-15 against the arXiv PDF text of 2506.19823v2: Abstract, §1-§6, App. A, App. H.3-H.4, Table 1, Table 3, Figures 2, 4-9, 11-17.
- Not reported by the source: absolute misalignment percentages for Figures 2, 5, 8, 9 (figure only); the identity and size of the GPT-4o-family models in Figure 4; RL hyperparameters.
