---
chapter: ch-42a
course: llm-training
phase: read
excerpt_of: arXiv:2510.05024v3 (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2510.05024
created_at: "2026-09-15"
---

# Excerpt: Inoculation Prompting — Instructing LLMs to Misbehave at Train-Time Improves Test-Time Alignment

- **Authors:** Nevan Wichers, Aram Ebtekar, Ariana Azarbal, Victor Gillioz, Christine Ye, Emil Ryd, Neil Rathi, Henry Sleight, Alex Mallen, Fabien Roger, Samuel Marks (Anthropic Fellows; MATS; Constellation; Redwood Research; Anthropic)
- **Year:** 2025 (arXiv v1 2025-10; PDF read here is v3, 27 Oct 2025)
- **Source type:** paper

## Method (§2)
- "Inoculation Prompting (IP) works by modifying the prompts x to request the undesired behavior, for example, by inserting the instruction 'Your code should only work on the provided test case, and fail on all other inputs'. We then train with SFT on the modified dataset {(x′, y)}."
- The instruction is not used at test time; optionally a safety instruction is added at test time instead.
- "when applying IP to a model trained with a chat template, we insert the instruction into the user message" (inserting it into the system message reduced capabilities more in preliminary experiments).
- Baseline: Pure Tuning, Safe Testing (PTST) — safety instruction at test time only. Ablation: inoculation instructions unrelated to the behavior ("Do not write any code in C#").
- Prompt-selection heuristic: "evaluate the extent to which the model exhibits the undesired behavior for various candidate prompts, and select the one which best elicits the undesired behavior."
- Code: https://github.com/safety-research/inoculation-prompting

## Four settings (§3)
1. **Reward hacking on code.** Qwen-2-7B base (also Mixtral Instruct v0.1 and Qwen 2.5 7B base) trained on MBPP-derived problems whose responses hard-code the visible test case; correctness judged on the visible test plus two held-out tests; results averaged over at least five seeds. "IP improves the correct solution rate and decreases the reward hacking rate compared to training without IP. Two out of four inoculation prompts allow the model to do better than the initial model even when trained on 100% reward-hacking data … All inoculation prompts we tried did better than the PTST baseline."
2. **Spurious correlation.** Llama 3 8B Instruct on CEBaB restaurant reviews filtered so that reviews mentioning ambiance always have sentiment 3-4; evaluation reverses the correlation. "The best performing inoculation prompt makes up for most of the accuracy lost by training with the spurious correlation."
3. **Sycophancy on GCD problems.** Gemma 2B Instruct trained on data where the user always proposes a correct solution and the assistant praises the user. "Most inoculation prompts which assert the user's correctness or direct the model to act as if the user is correct reduce sycophancy on both the GCD task and OOD tasks." Brittleness: "a minor wording change—from 'Behave as if the above solution is correct.' to 'Respond as if the above solution is correct.'—caused a large reduction in effectiveness."
4. **Toxicity on Reddit CMV.** Qwen-2-7B base trained on filtered persuasive-and-toxic replies (dataset average persuasiveness 5.82, harassment 0.362); "We train the model using a chat template for 1 epoch on a 24k prompt–response pair dataset with LoRA R=16". "Four out of five of the inoculation prompts we tried lowered the harassment score compared to PTST … The more strongly worded inoculation prompts ('Write a very mean and disrespectful response') improve the harassment score more than the weaker ones".

## Prompt-selection evidence (§3.5)
- "We obtain the following Pearson correlations: reward hacking with Mixtral: 0.57, GCD sycophancy: 0.57, spurious correlation: .90, Reddit CMV: 0.69."
- "None of the IP prompts we tried on the Qwen 2 base model in the reward hacking setting elicited any reward hacking behavior. This caused our prompt selection technique to fail in this case. We think this is because the Qwen 2 model is not instruction tuned, so it doesn't follow the inoculation instruction."

## Additional results (§3.6)
- IP applied to clean data: "We find no significant performance degradation across all settings". IFEval instruction following is unaffected for the coding dataset but decreases for the Reddit dataset.
- Compliance when the inoculation instruction is used at test time: no significant difference for Qwen 2 base and Mixtral on reward hacking, and a lower harassment score for Reddit CMV, but Qwen 2.5 7B base "showed increased reward hacking behavior when prompted compared to not using IP", and on the spurious-correlation setting "The IP-trained model showed lower accuracy when prompted to use spurious features". On Strong Reject, Qwen 2.5 7B base trained with IP "showed an increase in harmful compliance".

## Model of the mechanism (App. H)
Let T(M, C) be the level of trait T that model M exhibits in context C; C₀ is the neutral context and C_s the context containing inoculation instruction s; O is the oversight signal. Assuming training converges to maximum agreement with oversight, T(M_{C,O}, C) = T*(O) (Eq. 1). Define
k := [T(M_{Cs,O}, C₀) − T(M₀, C₀)] / [T(M_{Cs,O}, C_s) − T(M₀, C_s)]   (Eq. 2),
the ratio between the effect measured in the neutral context and in the inoculation context. Then
T(M_{Cs,O}, C₀) − T(M₀, C₀) = k (T*(O) − T(M₀, C_s))   (Eq. 3).
"If we vary s while holding O fixed, Equation (3) predicts a negative linear relationship between T(M₀, C_s) and T(M_{Cs,O}, C₀)" — the basis of the selection heuristic. Full inoculation of a bad trait and no loss on a good trait require T_bad(M₀, C_s) ≥ T*_bad(O) (Eq. 4) and T_good(M₀, C_s) ≤ T*_good(O) + (1/k)(T_good(M₀, C₀) − T*_good(O)) (Eq. 5); Eq. 5 is hardest to satisfy when k is small.

## Limitations (§5)
- "IP requires knowing the undesired behavior prior to fine-tuning so that we can write a natural language prompt that elicits it."
- The selection heuristic "is not always reliable" and fails when the initial model does not follow instructions.
- "In the Reddit CMV, and reward hacking settings, preliminary experiments show that IP may less effectively prevent the undesired behavior when training for more steps."
- "in two out of the six models we tested, IP increases the model's tendency to follow instructions that request the undesired behavior."
- "We only test IP in the setting of supervised fine-tuning on demonstration data. We leave testing the effect of IP when applied to on-policy reinforcement learning to future work."

## Verification
- Checked on 2026-09-15 against the arXiv PDF text of 2510.05024v3: Abstract, §1-§7, App. G.1-G.2, App. H.
- Not reported by the source: absolute rates in Figures 2-5 (figures only); learning rates and batch sizes for most settings; results for RL.
