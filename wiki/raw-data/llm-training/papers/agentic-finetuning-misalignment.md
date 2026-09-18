<!-- scope: Benign agentic SFT (web navigation, code) raises harmful-task attack success and lowers refusal in 5 aligned LLMs; WebDojo safety benchmark; PING inference-time prefix selection as mitigation; linear-probe analysis
     deps: [[finetuning-compromises-safety]]
     see-also: [[shallow-safety-alignment]], [[emergent-misalignment]], [[wildguard-data]], [[harmbench-data]], [[xstest]], [[agenttuning]]
-->

# Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation
- **Core Insight:** Fine-tuning Llama-3.1-8B-Instruct on benign web-navigation demonstrations raised WebArena-Lite success from 2.42% to 22.42%, while WebDojo attack success on harmful tasks rose from 32.88% to 64.38% and refusal fell from 26.03% to 6.85% (Table 1).
- **Guideline:** When fine-tuning an aligned model on benign agentic demonstrations, evaluate harmful-task attack success and refusal before and after training together with task success, because attack success increased in all 8 model-domain pairs tested (3 open models × 2 domains, 2 API models × code) (Tables 1-2). If a training-free mitigation is needed, an optimized response prefix (PING) raised refusal by an average of 66.2% (web) and 44.6% (code) with 1.8% success loss (§1), but it increased over-refusal on benign WebArena tasks by up to 63.6% for Qwen2.5-7B-Instruct (App. F).
- **Authors:** Dongyoon Hahm, Taywon Min, Woogyeol Jin, Kimin Lee (KAIST)
- **Year:** 2025 (arXiv v1 2025-08; v2 2025-11 carries an AAAI 2026 copyright notice)
- **URL:** https://arxiv.org/abs/2508.14031 (code: github.com/HahmDY/agentic-ft-safety)
- **Source type:** paper
- **Relevant topics:** agentic SFT, safety forgetting, refusal, attack success rate, over-refusal, prefix injection, guardrail models, linear probes, activation steering

## Abstract
Aligned LLMs fine-tuned to execute agentic tasks can become unintentionally misaligned: they become more likely to execute harmful tasks and less likely to refuse them. The authors propose Prefix INjection Guard (PING), which prepends automatically generated natural-language prefixes to agent responses. PING alternates between generating candidate prefixes and selecting those that optimize both task performance and refusal. PING outperforms prompting baselines on web-navigation and code-generation benchmarks without reducing task effectiveness. Linear probes on hidden states indicate that the prefix tokens drive the behavior change (Abstract).

## Key Contributions
- Threat model: developers fine-tune on agentic datasets collected under non-adversarial conditions that consist of benign task demonstrations; deployed agents then receive harmful inputs (§2.1).
- Measurement of the capability-safety trade-off for Llama-3.1-8B-Instruct, GLM-4-9B-Chat, Qwen2.5-7B-Instruct (web and code) and GPT-4o-mini, Gemini-2.0-flash (code, via fine-tuning APIs) (§2.2, Tables 1-2).
- WebDojo: a BrowserGym-based web safety benchmark with 80 harmful tasks (10 per website, eight websites) and 70 benign tasks, graded by rule-based URL/HTML conditions and refusal patterns (App. B).
- PING: Algorithm 1 automatic prefix selection, with comparisons to PTST and few-shot prompting, guardrail models, and a jailbreak attack (§3-§4, App. G-H).
- Linear-probe and activation-steering analysis of why prefix placement matters (§5, App. I).

## Key Figures/Tables to Study
- **Table 1-2** — SR, ASR, RR before and after fine-tuning.
- **Figure 2** — first-token probabilities on a harmful web task before and after fine-tuning.
- **Table 3** — the fixed prefix "I can't": safety gain with success loss.
- **Algorithm 1; Tables 7-8** — PING selection loop and hyperparameters.
- **Table 12** — over-refusal on benign benchmarks with PING; **Table 14** — PING with LlamaGuard3 and WildGuard.
- **Tables 16, 19-20** — probe logits and prefix vs suffix injection.

## Technical Details
**Metrics (§2.2).** SR = fraction of benign tasks completed (WebArena-Lite, MINT-ALFWorld). ASR = fraction of harmful tasks executed; RR = fraction of harmful tasks refused (WebDojo, RedCode-Exec). A response counts as a refusal if it contains predefined phrases such as "I can't" (§3, App. B.1).

**Before → after fine-tuning (Tables 1-2).**
| Model | Web: SR / ASR / RR | Code: SR / ASR / RR |
|---|---|---|
| Llama-3.1-8B-Instruct | 2.42→22.42 / 32.88→64.38 / 26.03→6.85 | 71.77→71.77 / 43.38→66.06 / 15.17→2.60 |
| GLM-4-9B-Chat | 5.45→16.97 / 20.55→54.79 / 4.11→4.11 | 22.58→72.58 / 63.29→72.39 / 13.70→1.48 |
| Qwen2.5-7B-Instruct | 3.03→7.27 / 49.32→60.27 / 2.74→10.96 | 70.16→85.48 / 58.33→86.02 / 6.02→3.10 |
| GPT-4o-mini | — | 41.12→70.16 / 30.09→41.96 / 40.05→37.01 |
| Gemini-2.0-flash | — | 50.80→83.87 / 50.23→77.82 / 19.86→3.15 |

ASR rises in all 8 pairs; RR falls in 6 (unchanged for GLM web, higher for Qwen web); SR rises in 7 (unchanged for Llama code) (values in %).

**First tokens (§2.3).** For Llama-3.1-8B-Instruct, 100% of WebDojo refusals begin with "I cannot" and 86% of RedCode-Exec refusals with "I can't". After fine-tuning, the first token on a harmful web task is "#" with probability 99.17% (Fig. 2). Prepending "I can't" to the fine-tuned Llama agent gives WebDojo RR 100% and ASR 0.0% but lowers WebArena SR from 22.4% to 10.3% and MINT SR from 84.3% to 46.8% (Table 3).

**PING (Algorithm 1, §3, App. E).**
1. A GENERATOR LLM (GPT-4o) proposes M candidate prefixes, seeded with prior top prefixes U(t).
2. Each prefix p gets perf(p) = non-refusal rate on benign tasks D_benign and refusal(p) = refusal rate on harmful tasks D_harmful; overall(p) = perf(p) + refusal(p), maximum 2.0.
3. If the best overall score so far is ≥ τ, U(t+1) = top-k by overall ∪ top-k by perf ∪ top-k by refusal; otherwise U(t+1) is empty, which the authors report encourages exploration (App. E.3, Fig. 11).
4. After T rounds, return argmax overall(p). For API models, which disallow prefix injection, the string is appended to the user prompt as a suffix (§4.1).
- Web prefixes start with "#" so the fine-tuned model reads them as a comment (App. E.1). Optimized prefixes are listed per model in Tables 10-11.

**Results (§4.2-§4.3, App. F-H).**
- PING keeps SR within 3% of the fine-tuned agents in the main comparison; it raises RR by 85% for GLM-4-9B-Chat (web) and by 66% for Gemini-2.0-flash (code) (§4.2, Fig. 3-4).
- Code domain, RR / SR: Llama PING 35.6 / 68.6, WildGuard 12.2 / 71.8, PING+WildGuard 39.1 / 68.6; Gemini PING 69.5 / 79.0, WildGuard 26.9 / 83.9, PING+WildGuard 80.8 / 79.0 (Table 4). PING+WildGuard adds 5.28% RR on average without lowering SR vs PING alone (§4.3). WildGuard alone exceeds PING for Qwen2.5-7B-Instruct (32.61% vs 29.86% RR) (Table 14).
- Over-refusal with refusal detection enabled on benign WebArena, RR without → with PING: Llama 1.2→6.1%, GLM 1.2→21.21%, Qwen 0.6→64.2%; SR 22.4→17.6%, 17.0→13.3%, 7.3→2.4%. MINT RR stays 0% for Llama and Qwen and rises to 2.4% for GLM (Table 12).
- Under improved few-shot jailbreaking (4 demonstrations), Llama PING WebDojo RR 76.7% → 74.6%; the unprotected agent 6.85% → 3.17% (Table 13).

**Probe analysis (§5, App. I).** Probes are trained on averaged activations: 920 harmful (HarmBench 520, AdvBench 400) vs the first 920 Alpaca instructions (App. I.1). Mean final-token logit on 70 WebDojo harmful tasks, agent → agent+PING: Llama -4.87→2.29, GLM -6.88→0.11, Qwen -8.18→3.91; averaged over the input, vanilla agents are positive (6.14, 3.93, 3.45) (Table 16). Suffix vs prefix (Llama web): final-token logit -1.67 vs 2.29; RR 14.29% vs 79.37%, ASR 58.73% vs 9.52% (Tables 19-20). Activation steering with coefficient 20 on Llama layers 20, 22, 24, 26, 28 raises harmful RR to 95.91% but benign RR to 97.95% and SR to 0% (Table 17).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct, GLM-4-9B-Chat, Qwen2.5-7B-Instruct agents | 7-9B | SFT | data | web: dataset of Qi et al. 2024b (WebRL); code: CodeActInstruct | arXiv:2508.14031v2 §4.1 | verified 2026-09-14 | no ablation reported |
| same | 7-9B | SFT | LR; scheduler; warmup ratio; batch size; epochs; cutoff length | 1e-5; cosine; 0.1; 128; 1; 16384 | App. A.2, Table 5 | verified 2026-09-14 | no ablation reported |
| GPT-4o-mini, Gemini-2.0-flash agents | API | SFT | fine-tuning API settings | not reported | checked §2.2, §4.1, App. A | not reported | — |
| all agents | — | eval-gate | temperature; top-p; max response tokens | 0.0; 1.0; WebArena-Lite 2048, WebDojo 512, MINT 512, RedCode 1024 | App. A.2, Table 6 | verified 2026-09-14 | not applicable |
| PING, web domain | — | eval-gate (inference-time prefix) | k; M; T; τ; benign/harmful tasks; generator temperature | 3; 5; 20; 1.5; 7/7; 0.7 | App. A.2, Table 7 | verified 2026-09-14 | App. E.2: near-optimal prefixes emerged within 3-4 iterations (Fig. 9-10) |
| PING, code domain | — | eval-gate (inference-time prefix) | k; M; T; τ; benign/harmful tasks; generator temperature | 3; 5; 20; 1.0; 10/27; 0.7 | App. A.2, Table 8 | conflict: App. E.2 says 5 iterations × 10 prefixes; App. A.1 says 75 RedCode tasks for prefix evaluation | App. E.3: around 25 prefixes over 5 iterations reach optimal scores (Fig. 9-10) |
| dataset sizes, full vs parameter-efficient tuning, compute | — | SFT | — | not reported | checked §2-§4, App. A-I | not reported | — |

## Findings relevant to generality, negative feedback, and agentic training
- **Narrowing / forgetting of safety behavior:** training only on successful benign demonstrations improves in-domain task success and reduces refusal of harmful requests in the same domain (Tables 1-2). The authors call this a capability-safety trade-off (Table 1 caption). Transfer to chat-domain safety benchmarks is not measured.
- **Mechanism (Interpretation):** fine-tuning shifts first-token probability toward action tokens (Fig. 2), and probe logits suggest safety features remain in the representations but are not active at the final token (§5.2, Table 16).
- **Negative signals:** none are used in training. PING adds refusal behavior at inference; the authors state it induces safe behavior "even without fine-tuning" the agents on a safety dataset (§7). Mixing refusal examples into agentic SFT data is not tested.
- **Measurement:** refusal is detected by phrase matching (App. E.1), and over-refusal is only visible when refusal detection is enabled on benign benchmarks (App. F); benign success alone hides it.

## Connections
- [[finetuning-compromises-safety]], [[emergent-misalignment]] — benign and narrow fine-tuning that degrades alignment, cited as motivation (§1, §6).
- [[shallow-safety-alignment]] — first-token refusal patterns that §2.3 builds on (Qi et al. 2024a).
- [[wildguard-data]], [[harmbench-data]] — guardrail model compared in Tables 4 and 14, and a source of probe training data (App. I.1).
- [[xstest]] — a dedicated over-refusal test suite; not used in this paper, which measures over-refusal on WebArena-Lite and MINT (App. F).
- [[agenttuning]] — agentic SFT mixed with ShareGPT general data; this paper fine-tunes on agentic data alone and does not test mixtures.
- [[webarena-data]] — base environment of WebArena-Lite.
- [[agentic-benchmark-checklist]] — grading checks relevant to phrase-matched refusal and rule-based WebDojo evaluators.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2508.14031 (arXiv v2, 2025-11-17; v1 2025-08).
- Audit claims not found in the source: "agentic SFT/RL data needs safety and refusal ballast, just as AgentTuning needed general-data ballast" (audit interpretation; the paper tests no data mixtures and no RL); "benign-only agentic data contains no refusal examples" (the paper says the data consist of benign demonstrations collected under non-adversarial conditions, §2.1).
- Internal inconsistencies in v2: §1 reports a 38.09% increase in WebDojo attack success and §2.2 "32%" for Llama web, while Table 1 gives 32.88→64.38 (+31.50 points); App. F text reports over-refusal from 4.9% to 63.6% and SR loss at most 4.9%, the Table 12 caption says 5% to 63% and at most 5%; §5.1 says steering raises WebDojo RR from 0%, Table 17 lists 6.12% at coefficient 0.
- Not reported by the source: training-set sizes, closed-model fine-tuning hyperparameters, seeds or variance, results for RL-trained agents.
