<!-- scope: NovaSky Sky-T1-7B / Sky-T1-mini post (2025-02) — 4-step SFT→RL→SFT→RL from Qwen2.5-Math-7B with 5K QwQ traces, re-distillation from the RL checkpoint, RLOO on DeepSeek-R1-Distill-Qwen-7B, SFT data scaling, and SFT-vs-RL pass@k observations
     deps: [[sky-t1]], [[rloo]]
     see-also: [[novasky-sky-t1-7b-distill-rl-loop-recipe]], [[iterative-sft-rl]], [[rlvr-beyond-base-model]], [[qwen-qwq-traces]], [[numina-math]], [[sky-t1-flash-overthinking]]
-->

# Unlocking the Potential of Reinforcement Learning in Improving Reasoning Models
- **Core Insight:** From Qwen2.5-Math-7B, a 4-step pipeline (SFT on 5K QwQ traces → PRIME RL → SFT on 5K QwQ + 5K RL-model traces → RLOO) raised the 4-benchmark math average from 36.0 to 57.1 (Table 1). SFT on 5K traces sampled from the step-2 RL checkpoint gave the same average as that checkpoint (55.8 vs 55.8) and a higher average than SFT on 5K QwQ traces (45.0).
- **Guideline:** When long-CoT SFT data grows beyond 60K traces, add an RL stage instead of relying on more SFT data, because in the post's scaling ablation the SFT-only average was 43.8 / 49.0 / 48.0 at 30K / 60K / 120K QwQ traces while SFT+RLOO reached 51.6 / 56.5 / 58.2 (Figure 4). This was measured only on 7B-scale math models and four math benchmarks.
- **Authors:** Shiyi Cao, Shu Liu, Dacheng Li, Tyler Griggs, Kourosh Hakhamaneshi, Sumanth Hegde, et al. (NovaSky, Berkeley Sky Computing Lab)
- **Year:** 2025 (published 2025-02-13; page shows an update on 2026-03-14; not on arXiv)
- **URL:** https://novasky-ai.github.io/posts/sky-t1-7B/ (collection: https://huggingface.co/collections/NovaSky-AI/sky-t1-7b-67ab281da8192c1ba3e5296c ; code: https://github.com/NovaSky-AI/SkyThought/tree/main/recipes/sky-t1-7b)
- **Source type:** official blog
- **Relevant topics:** long-CoT distillation, SFT–RL interleaving, re-distillation from an RL checkpoint, RLOO, PRIME, pass@k vs pass@1, SFT data scaling
- **Recipe ledger:** [[novasky-sky-t1-7b-distill-rl-loop-recipe]] (`blogs/novasky-sky-t1-7b-distill-rl-loop-recipe.md`)

## Summary
The post releases two 7B math reasoning models. Sky-T1-7B is trained from the Qwen2.5-Math-7B base model with a 4-step SFT → RL → SFT → RL pipeline that uses 5K QwQ-distilled traces (introduction). Sky-T1-mini is trained with RLOO on top of DeepSeek-R1-Distill-Qwen-7B in 36 hours on 8xH100, about $870 at Lambda Cloud pricing (introduction). The post also reports ablations on SFT data scaling, RL scaling, and pass@k after SFT and RL. Its stated conclusion: "Long CoT SFT in general enhances the model's pass@k performance while RL lifts the model's performance at lower generation budgets (i.e., pass@1), which sometimes come at a cost of the entropy of solutions" (Conclusion). Training code, recipes, weights, and evaluation scripts are released (introduction).

## Key Contributions
- A 4-step pipeline in which each SFT step restarts from the Qwen2.5-Math-7B base model, not from the previous RL checkpoint (Step 1, Step 3).
- Re-distillation: 5K rejection-sampled traces from the step-2 RL model train a new SFT model (Sky-T1-7B-Step2-5k-distill) that matches the RL model's average (Step 2; Table 1).
- An RL-from-base comparison (Sky-T1-7B-Zero: RLOO on Qwen2.5-Math-7B with STILL3, 4 rollouts, 104 steps) (Results).
- Sky-T1-mini: RLOO with only a verifier reward on DeepSeek-R1-Distill-Qwen-7B, with an 8K and then a 16K length cutoff (Sky-T1-mini section).
- SFT data scaling (30K / 60K / 120K QwQ traces) with and without RL, and pass@k curves (Other Observations; Figures 4–5).

## Key Figures/Tables to Study
- Table 1 (image): all intermediate models on AIME24, MATH500, AMC23, OlympiadBench.
- Figure 2: pass@k curves on AIME24 and AMC23 after each step.
- Figure 3: Sky-T1-7B and Sky-T1-mini vs other 7B models and o1-mini.
- Figures 4 and 5: SFT data scaling (benchmark averages; AMC23 pass@k).

## Technical Details
**Step 1, SFT (Step 1).** Teacher: QwQ, because the model was trained before DeepSeek-R1 was released and QwQ was then the only open-weights long-reasoning model. GPT-4o-mini classifies NuminaMath prompt difficulty on the AoPS scale; selected: math problems above Level 3, Olympiads problems above Level 8, and all AIME/AMC problems. Rejection sampling keeps solutions that match the ground truth; 5K responses are kept. SFT on Qwen2.5-Math-7B with the Sky-T1 system prompt, 3 epochs, LR 1e-5, batch size 96 → Sky-T1-7B-Step1.

**Step 2, RL (Step 2).** PRIME's algorithms on Eurus-2-RL-Data, 127 steps, batch size 256 (~30K data), 4 rollouts per prompt, with PRIME's prompt filtering that removes problems where all 4 rollouts are correct or all are wrong; about 44 hours on 8xH100 → Sky-T1-7B-Step2. Citing DeepSeek-V3 §5.1, the team then samples and rejection-filters 5K traces from Step2 with the Step 1 mixture and fine-tunes Qwen2.5-Math-7B on them → Sky-T1-7B-Step2-5k-distill.

**Step 3, SFT again (Step 3).** 5K QwQ traces (Step 1) + 5K Step2 traces, SFT on the Qwen2.5-Math-7B base, 3 epochs, LR 1e-5, batch 96 → Sky-T1-7B-step3.

**Step 4, RL again (Step 4).** RLOO without prompt filtering and without a process reward model, on the numina_amc_aime and numina_olympiads subsets of Eurus-2-RL-Data, 59 steps, batch 256 (~15K data), 8 rollouts per prompt → Sky-T1-7B.

**Evaluation (Evaluation; recipe README).** Qwen's math evaluation suite. AIME24 (30 questions) and AMC23 (40 questions): 8 samples per question, temperature 0.6, top-p 0.95, pass@1. MATH500 and OlympiadBench: greedy decoding. The README adds vLLM 0.6.2 and says Sky-T1-mini is evaluated without a system prompt.

**Table 1 (image; accuracy %).**

| Model | AIME24 | MATH500 | AMC23 | OlympiadBench | Avg. |
|---|---|---|---|---|---|
| Qwen2.5-Math-7B | 14.2 | 52.4 | 32.2 | 17.2 | 36 |
| Sky-T1-7B-Zero (RL) | 23.8 | 77.6 | 65 | 41.3 | 51.9 |
| Sky-T1-7B-Step1 (SFT) | 15.4 | 72.6 | 56.9 | 35 | 45 |
| Sky-T1-7B-Step2 (RL) | 21.25 | 87 | 66.6 | 48.4 | 55.8 |
| Sky-T1-7B-Step2-5k-distill | 21.25 | 85 | 66 | 50.8 | 55.8 |
| Sky-T1-7B-Step3 (SFT) | 18.8 | 81.2 | 62.5 | 41.6 | 51 |
| Sky-T1-7B (RL) | 24.6 | 85.6 | 69 | 49.3 | 57.1 |

**Sky-T1-mini (Sky-T1-mini section; Figure 3).** RLOO with only the verifier reward on DeepSeek-R1-Distill-Qwen-7B, STILL3 plus the numina_amc_aime and numina_olympiads subsets of Eurus-2-RL-Data. 119 steps (~28 h) at batch 256 (~30K) with an 8K cutoff, then 29 steps (~8.7 h) with a 16K cutoff, on 8xH100. Figure 3 bar labels, Sky-T1-mini vs R1-Distilled-7B: AIME 2024 55.4 vs 51.7; MATH500 91.2 vs 89.4; AMC 2023 86.9 vs 90.0; OlympiadBench 56.3 vs 50.7; average 72.5 vs 70.5 (o1-mini: 60.0, 90.2, 90.0, 67.2, 76.8). The text reports +4% AIME, +5.6% OlympiadBench, +2% average; the AMC 2023 decrease of 3.1 points is visible in Figure 3 but not mentioned in the text.

**SFT data scaling (Other Observations; Figure 4).** RL here is RLOO on STILL3 with 4 rollouts per prompt; the base model for this ablation is not stated. Averages: SFT 43.8 (30K), 49.0 (60K), 48.0 (120K); SFT+RL 51.6, 56.5, 58.2.

## Findings relevant to generality, negative feedback, distillation
**Generality and pass@k.** All reported evaluations are math benchmarks; no out-of-domain evaluation is reported. On AIME24 and AMC23, long-CoT SFT raised overall pass@k (Results; Figure 2). On AMC23, the two RL stages mainly raised pass@1 while reducing solution diversity for k = 4 to 32 (Results). On AIME24, step-4 RL raised pass@k over Step1 and Step2, less than Sky-T1-7B-Zero did (Results). In the data-scaling runs, both SFT and RL improved pass@k from 30K to 60K and 120K, RL achieved better test-time scaling than SFT at every data size, and gains from 60K to 120K were smaller than from 30K to 60K; RL mainly raised accuracy at small k, which "may come at a trade-off of entropy of solutions" (Other Observations; Figure 5). RL from the base model (Zero, 51.9 average) scored higher than SFT on 5K QwQ traces (Step1, 45.0) (Table 1).

**Negative feedback (course standard §6.1 terms).** Teacher traces whose answers do not match the ground truth are discarded (negative marginal value; Step 1, Step 2). Step 2 also removes prompts on which all 4 rollouts are wrong or all are correct (Step 2); Step 4, Zero, and Sky-T1-mini use no prompt filtering. Incorrect rollouts in RLOO receive a lower reward than the leave-one-out baseline, which is negative as gradient (mechanism in [[rloo]]).

**Distillation.** Stage: distill-SFT from a base model, repeated after RL (Steps 1 and 3). Prompt types: NuminaMath competition problems filtered by GPT-4o-mini difficulty labels. Teachers and sampling: QwQ (5K kept) and the 7B Step2 RL checkpoint (5K kept); the released generation command sets max_tokens=16384 and no temperature (recipe README). Quality control: final-answer match with ground truth. Result: at 5K traces, the RL-trained 7B teacher gave a higher SFT average (55.8) than the QwQ teacher (45.0) (Table 1).

## Connections
- [[sky-t1]] — the team's earlier 32B QwQ-distillation post; [[sky-t1-flash-overthinking]] — the January 2025 preference-optimization follow-up.
- [[rloo]] — algorithm for Step 4, Sky-T1-7B-Zero, and Sky-T1-mini.
- [[qwen-qwq-traces]] — the QwQ teacher; [[numina-math]] — source of SFT prompts.
- [[deepseek-r1]] — DeepSeek-R1-Distill-Qwen-7B, the Sky-T1-mini starting point (the post says it was trained on 800K R1-distilled samples).
- [[open-thoughts]] — OpenThinker-7B (117K R1 responses per the post), a comparison model; [[rstar-math]] — compared with reported numbers.
- [[iterative-sft-rl]] — other alternating SFT and RL pipelines.
- [[rlvr-beyond-base-model]], [[pass-at-k-training]] — pass@k analyses of RL-trained models.
- [[deepscaler]] — another early-2025 RL run on an R1-Distill model; [[skyrl-agent]] — later NovaSky RL framework work.

## Verification
- Created on 2026-09-14 from https://novasky-ai.github.io/posts/sky-t1-7B/ (page dated 2025-02-13, updated 2026-03-14), including Table 1 (7b.jpg), Figure 3 (performance_stats_avg.png), and Figure 4 (sft_scale_rl.png) images; cross-checked with SkyThought@0d190f1 `recipes/sky-t1-7b/README.md` and `skythought/skythought-rl/examples/sky-t1/run-sky-t1-7b-{step2,step4,zero}.sh` and `run-sky-t1-mini.sh`.
- Audit claims not found in the source: none. The audit's "+4% AIME" matches the text; Figure 3 shows 55.4 vs 51.7 (+3.7).
- Inconsistencies inside the source: Figure 3's caption says "AIME23" but its axis says "AIME 2024"; the citation block prints the URL `/posts/sky-t1-7b`, while the fetched page is `/posts/sky-t1-7B/`; the released Step 2 script sets `algorithm.adv_estimator=rloo` and contains no process-reward-model arguments, while the post says Step 2 uses PRIME's algorithms; the released Sky-T1-mini script sets `adv_estimator=grpo`, while the post says RLOO (see recipe ledger).
- Not reported by the source: RL learning rates and response lengths in the post (only in released scripts), Sky-T1-mini rollouts per prompt and LR (the released script sets n=4 and LR 5e-7), teacher sampling temperature, Step2-5k-distill SFT hyperparameters, checkpoint selection rule.
