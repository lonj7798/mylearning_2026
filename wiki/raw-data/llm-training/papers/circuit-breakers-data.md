<!-- scope: Circuit breakers via Representation Rerouting (RR): circuit-breaker and retain data construction, RR loss, LLM / multimodal / agent robustness and over-refusal results
     deps: [[harmbench-data]]
     see-also: [[constitutional-ai]], [[wildguard-data]], [[anthropic-sleeper-agents-data]], [[salad-bench]], [[ultrachat-construction]]
-->

# Improving Alignment and Robustness with Circuit Breakers
- **Core Insight:** Adding Representation Rerouting (RR), a LoRA-trained loss that makes the hidden states on harmful text orthogonal to the original model's hidden states, to Llama-3-8B-Instruct lowered the average attack success rate (ASR) over 10 attack settings from 38.1% to 3.8%, while MT-Bench went from 8.05 to 8.00 (Table 1).
- **Guideline:** When a refusal-trained model must resist attacks not seen in training, train RR on harmful completions plus a retain set that includes benign chat, over-refusal prompts, and refusal examples, because removing the refusal examples lowered ASR (2.5% → 0.6%) but lowered MT-Bench (8.0 → 7.7) (Table 8); expect weaker protection on harm categories far from the training categories (Fig. 5).
- **Authors:** Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, et al. (10 authors; Gray Swan AI, Carnegie Mellon University, Center for AI Safety)
- **Year:** 2024 (arXiv v1 2024-06; v4 2024-07)
- **URL:** https://arxiv.org/abs/2406.04313 ; code: https://github.com/GraySwanAI/circuit-breakers
- **Source type:** paper (with released training scripts)
- **Relevant topics:** safety data construction, representation engineering, jailbreak robustness, retain loss, over-refusal, agent safety

## Abstract
Refusal training is often bypassed by adversarial attacks, and adversarial training defends only against the attacks it trains on. The paper proposes "circuit breakers": training that directly controls the internal representations responsible for harmful outputs, so that a model that starts a harmful generation has its internal process interrupted. The method is applied to text-only and multimodal language models and is reported to prevent harmful outputs without loss of utility under strong attacks not seen in training, including adversarial images ("image hijacks"). It is also applied to AI agents, where it reduces the rate of harmful actions under attack.

## Key Contributions
- Defines circuit breaking by two components, datasets (a circuit breaker set and a retain set) and loss functions, and instantiates it as RR trained with Low-Rank Representation Adaptation (LoRRA) (§3, Alg. 1).
- Gives data rules for models that already refuse: keep harmful assistant responses but remove the harmful user requests in the circuit breaker set, and add refusal data to the retain set (§3 "Data"; Table 8).
- Evaluates Mistral-7B-Instruct-v2 and Llama-3-8B-Instruct against 10 attack settings, including prefilling, input-embedding, and RepE attacks, with capability checks (§4.1, Table 1).
- Extends RR to LLaVA-NeXT-Mistral-7B under a white-box PGD image attack (§4.2) and to function-calling agents (§4.3).
- Compares representation control (RR) with representation reading (linear and MLP harmfulness probes) (§4.4, Table 2).

## Key Figures/Tables to Study
- **Algorithm 1:** RR loss, retain loss, and coefficient schedule.
- **Table 1:** ASR per attack and capability scores for refusal-trained, adversarially trained (R2D2), +RR, and Cygnet models.
- **Table 3 (App. B):** over-refusal rate on WildChat. **Table 8:** training-set and loss ablations.
- **Figure 5:** train-category × test-category ASR. **Figure 6:** per-layer cosine between models with and without circuit breakers on a prefilled harmful response.

## Technical Details
- **Notation (Alg. 1).** M = frozen original model; M_cb = the same model with LoRA adapters; rep_M(x), rep_Mcb(x) = hidden states at the target layers; D_s = circuit breaker set; D_r = retain set; t = step; T = total steps; α = coefficient scale.
- **Loss (Alg. 1).** c_s = α(1 − t/(2T)); c_r = α·t/(2T); L_s = ReLU(cos_sim(rep_M(x_s), rep_Mcb(x_s))); L_r = ‖rep_M(x_r) − rep_Mcb(x_r)‖₂; L = c_s·L_s + c_r·L_r. The ReLU stops the optimization once the cosine similarity reaches 0 (§3 "Loss").
- **Rejected alternatives.** Distance to a scaled random vector ‖rep_cb − α·rep_rand‖₂ (as in RMU) "requires extensive tuning of the α parameter"; a normalized random-vector variant is also described; the cosine loss is reported as the best balance of robustness and retained capability (§3 "Loss").
- **Circuit breaker set (LLM).** An uncensored LLM is prompted with examples and a range of categories to write short harmful queries and harmful completions; samples with BLEU > 0.3 against any HarmBench standard behavior are removed (App. A.1). Set sizes are not reported.
- **Retain set (LLM).** UltraChat conversations and XSTest (exaggerated-refusal prompts); Llama-3 also receives extra refusal examples (§4.1).
- **Multimodal data.** LLaVA-Mistral-7B describes COCO images; an uncensored LLM writes harmful queries and completions for them; LLaVA-Instruct is the retain set (App. A.2, §4.2).
- **Agent data.** Function definitions from Glaive Function Calling v2; an LLM writes harmful requests, GPT-3.5-turbo executes them, and outputs are converted to OpenFunctions format; samples with BLEU > 0.1 against the agent benchmark are removed; original Glaive v2 data is the retain set (App. A.3).
- **Tokens under the loss.** User and assistant text of the circuit breaker set for LLMs and agents; all tokens after the image embeddings for multimodal (App. C.1). LoRA is used "to ensure greater stability and improved retention" (App. C.1).
- **Hyperparameter selection.** Static attack test cases from the HarmBench validation set (§4.1).
- **Attacks.** Direct request, HumanJailbreaks (manual), AutoDAN, TAP-Transfer, PAIR, GCG, multilingual (6 languages), prefilling, input-embedding optimization (20 embeddings, SGD, up to 500 steps), RepE attack on layers −11 to −20 (§4.1; App. C.2.2). Judge: HarmBench classifier with manual verification (§4.1).
- **LLM results (Table 1, average ASR %).** Mistral-7B-Instruct-v2: refusal-trained 76.7, R2D2 31.7, +RR 9.8; MT-Bench 7.60 / 6.00 / 7.53. Llama-3-8B-Instruct: refusal-trained 38.1, +RR 3.8, Cygnet 0.8; MT-Bench 8.05 / 8.00 / 8.21. The highest remaining +RR ASR is the input-embedding attack (15.7 Mistral, 9.6 Llama-3). The text reports average compliance reductions of 87% (Mistral) and 90% (Llama-3) (§4.1).
- **Cygnet.** Llama-3-8B-Instruct fine-tune that combines circuit breakers with "other representation control" methods (§1, Table 1); its training details are not reported.
- **Probes (Table 2, average ASR over 5 settings).** Llama-3: refusal-trained 32.6, linear probe 9.0, MLP probe 6.8, RR 3.1. Probes read layer 16 (Mistral) or the final layer (Llama-3); MLP hidden sizes 64 and 32; thresholds set so the WildChat false-positive rate is near RR's; the attacker does not know about the probe (§4.4).
- **Representation analysis.** On a prefilled harmful response, cosines and norms change from layer 10 onward before generation starts; layers 10 and 20 are the targeted layers (§4.4, Fig. 6).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v2 + RR; Llama-3-8B-Instruct + RR | 7B; 8B | SFT (RR LoRA loss, not cross-entropy) | steps; batch size (unit not stated) | 150; 16 | arXiv:2406.04313v4 App. C.2.1 | verified 2026-09-14 | no ablation reported |
| Mistral-7B-Instruct-v2 + RR / Llama-3-8B-Instruct + RR | 7B / 8B | SFT (RR) | α | 5 / 10 | App. C.2.1 | verified 2026-09-14 | no ablation reported |
| both LLMs | 7B; 8B | SFT (RR) | target layers; LoRA placement | layers 10 and 20; all linear layers in layers 0–20 | App. C.2.1 | verified 2026-09-14 | no ablation reported |
| both LLMs | 7B; 8B | SFT (RR) | compute | 1 A100-80GB, 20 minutes | App. C.2.1 | verified 2026-09-14 | not applicable |
| Llama-3-8B-Instruct + RR | 8B | SFT (RR) | rerouting loss | ReLU(cosine) with Alg. 1 schedule | §3, Alg. 1 | verified 2026-09-14 | Table 8: RR avg ASR 2.5 / MT-Bench 8.0; RandP 9.7 / 8.0; RandC and RMU did not converge |
| Llama-3-8B-Instruct + RR | 8B | SFT (RR) | retain set | UltraChat + XSTest + refusal examples | §4.1 | verified 2026-09-14 | Table 8: w/o refusal 0.6 / 7.7; w/ refusal 2.5 / 8.0 |
| Llama-3-8B-Instruct + RR | 8B | SFT (RR) | circuit breaker set | harmful responses with requests removed added ("w/ Augment") | §3 "Data" | verified 2026-09-14 | Table 8: w/o Augment 5.8 / 8.1; w/ Augment 2.5 / 8.0 |
| Mistral-7B-Instruct-v2 + RR | 7B | SFT (RR) | retain set | UltraChat + XSTest | §4.1 | verified 2026-09-14 | App. G: augmentation and refusal-retain ablations not run (model lacks refusal training) |
| both LLMs | 7B; 8B | SFT (RR) | circuit breaker and retain set sizes | not reported | checked body, App. A, App. C, repo README | not reported | none |
| both LLMs (released scripts) | 7B; 8B | SFT (RR) | learning rate; schedule; weight decay | 1e-4; constant; 0 | github.com/GraySwanAI/circuit-breakers @main (fetched 2026-09-14) scripts/lorra_circuit_breaker_{llama3_8b,mistral_7b}.sh | verified 2026-09-14 (paper: not reported) | no ablation reported |
| both LLMs (released scripts) | 7B; 8B | SFT (RR) | LoRA r / alpha / dropout; max length | 16 / 16 / 0.05; 8192 | same scripts | verified 2026-09-14 | no ablation reported |
| LLaVA-NeXT-Mistral-7B + RR | 7B | SFT (RR) | α; target layer; LoRA layers; frozen parts; batching | 5; 16; 14–16; image encoder and projection frozen; text and multimodal batches alternate | App. C.3.1 | verified 2026-09-14 | no ablation reported |

The Llama-3 script passes `--use_refusal_retain`; the Mistral script does not, which matches §4.1. Table 8 does not name its model; it is assigned to Llama-3 here because App. G states the augmentation and refusal ablations were not run on Mistral and its RR MT-Bench (8.0) matches Llama-3 +RR in Table 1.

## Findings relevant to generality, negative feedback, agentic training
- **Category transfer.** Llama-3 models trained on one harm category have low ASR in that category; broader training categories (Harmful, Illegal Activities) transfer further than narrow ones (Cybercrime) (§4.4, Fig. 5). For Mistral the set is "relatively robust to distribution shifts in categories of harm" (App. G).
- **Capability.** Llama-3 Open LLM average 68.8 → 68.3 with RR; the largest single-task drop is HellaSwag 78.6 → 76.8 (Table 5). Adversarial training (R2D2) lowers Mistral MT-Bench 7.60 → 6.00 and TruthfulQA 66.8 → 45.5 (Tables 1, 5).
- **Multilingual.** Llama-3 average ASR over 6 languages 19.3 → 3.5 (App. F, Table 6).
- **Over-refusal.** On 500 English non-toxic WildChat requests, refusal rate is 2.0% → 3.4% for Mistral +RR (R2D2 10.6%) and 2.2% → 6.2% for Llama-3 +RR; Claude-3-Opus is 20.6% (App. B, Table 3). Circuit-breaker models are scored by keyword checks plus perplexity (App. B).
- **Loss trade-off on Mistral.** Random-vector losses "decrease ASR but also decrease capabilities": RMU avg ASR 2.8 / MT-Bench 7.1 vs RR 7.0 / 7.5 (App. G, Table 7).
- **How harmful text is used (Interpretation, this course).** Harmful completions enter only the hidden-state loss L_s; no likelihood term is applied to them (Alg. 1). The token-probability analysis used for unlikelihood or DPO rejected terms therefore does not transfer directly.
- **Agents.** On 100 harmful function-calling requests judged by gpt-4-turbo, Llama-3 compliance is 58 → 8 without attack and 82 → 14 under forced function calls; BFCL (mean of AST and Exec) is 74.8 → 76.0 (§4.3, App. C.4, App. E Fig. 8).
- **Multimodal.** Under PGD (ε = 32/255, 1000 steps) on 133 behaviors, compliance is 91.0 → 14.3; MMMU 34.7 → 34.2; LLaVA-Wild 79.2 → 79.3 (§4.2, App. C.3.2, App. E Fig. 8).
- **Stated limits.** The method targets attacks whose goal is harmful content, not class-label attacks, and the experiments use single-turn conversations (§5).
- **Not reported:** refusal or capability curves over training steps, sensitivity to α beyond the two values used, dataset sizes.

## Connections
- [[harmbench-data]] — supplies the attacks, the classifier, the validation cases for hyperparameter selection, and the behaviors used in the BLEU filter (§4.1, App. A.1).
- [[ultrachat-construction]] — UltraChat is part of the LLM retain set (§4.1).
- [[constitutional-ai]] — trains harmless outputs with SFT and RL from AI feedback; RR changes internal representations instead.
- [[wildguard-data]] — an input/output safety classifier, the system-level defense class that §2 contrasts with RR.
- [[anthropic-sleeper-agents-data]] — backdoored behavior that persists through safety training; this threat model is not evaluated here.
- [[salad-bench]] — another safety benchmark; not used in this paper.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2406.04313 (arXiv v4, 2024-07-12) and the two training scripts in github.com/GraySwanAI/circuit-breakers @main.
- Corrections to the previous card version:
  - "seed harmful prompts from HarmBench, AdvBench, SORRY-Bench" → queries and completions are generated by an uncensored LLM; HarmBench is used only for the BLEU > 0.3 decontamination filter (App. A.1).
  - "harmful completion from the base model or a teacher in non-refusal mode" → the uncensored LLM writes the completions (App. A.1).
  - "retain objective: match or preserve the original model behavior" → ℓ2 distance between frozen-model and circuit-breaker-model hidden states on the retain set (Alg. 1).
  - "optimize hidden states so they move away from the original trajectory" → ReLU(cosine similarity) at layers 10 and 20 with a scheduled coefficient (Alg. 1, App. C.2.1).
  - "capability-retention table: MMLU, GSM8K, general chat" → MT-Bench plus six Open LLM Leaderboard tasks (Tables 1, 5).
  - "LoRA-style fine-tuning is sufficient; no full-model retraining required" → LoRA is chosen for stability and retention (App. C.1); no full fine-tuning comparison is reported.
- Removed as unsupported by the source: "a relatively small dataset is enough" (no size given); "overweighting the rerouting loss causes over-refusal if the retain set is weak" (replaced by Table 8 and Table 7 results); "unseen attacks can still find paths not covered by the rerouted region"; "does not address honesty or goal misalignment" (§4.3 speculates about power-seeking and dishonesty instead); output-surface and trajectory metaphors.
- Not reported by the source: dataset sizes, Cygnet recipe, learning rate and LoRA rank (released scripts only), venue.
