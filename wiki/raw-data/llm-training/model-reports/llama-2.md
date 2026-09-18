<!-- scope: Llama 2 report (arXiv:2307.09288): 7B–70B pretraining on 2T tokens, SFT on 27,540 annotations, separate helpfulness and safety reward models, iterative RLHF (rejection sampling then PPO, RLHF-V1..V5), Ghost Attention for multi-turn system messages, safety fine-tuning
     deps: [[rlhf-instructgpt]], [[ppo]]
     see-also: [[llama-2-recipe]], [[llama-3]], [[hh-rlhf]], [[constitutional-ai]], [[rejection-sampling-finetuning]], [[lima]]
-->

# Llama 2: Open Foundation and Fine-Tuned Chat Models
- **Core Insight:** Llama 2-Chat was built from 27,540 SFT annotations, 1,418,091 Meta preference comparisons, two separate reward models, and five RLHF versions using rejection sampling and PPO; on ~4,000 human-rated helpfulness prompts, Llama 2-Chat 70B has a 36% win rate and a 31.5% tie rate against ChatGPT (§3.1, Table 6, §3.2.3, §3.4.2).
- **Guideline:** When a model is improved by repeated rejection-sampling fine-tuning, draw candidates from the top samples of all earlier iterations rather than only the previous one, because training RLHF-V3 only on RLHF-V2 samples caused regressions such as weaker rhyming in poems, which the authors report this change addressed (§3.2.3; no figures given).
- **Authors:** Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, et al. (GenAI, Meta; corresponding authors Hugo Touvron and Thomas Scialom)
- **Year:** 2023 (arXiv v1 2023-07; v2 2023-07-19)
- **URL:** https://arxiv.org/abs/2307.09288
- **Source type:** official technical report
- **Relevant topics:** pretraining recipe, GQA, SFT data quality, pairwise reward modeling with margins, rejection-sampling fine-tuning, PPO with KL penalty, iterative RLHF, Ghost Attention (multi-turn system messages), context distillation, safety data scaling, false refusal

## Abstract
The report releases Llama 2, pretrained models from 7B to 70B parameters, and Llama 2-Chat, fine-tuned for dialogue. The authors report that Llama 2-Chat outperforms open-source chat models on most benchmarks they tested and, based on their human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. The report describes the fine-tuning and safety methods in detail so that others can build on them (Abstract). Pretraining uses 2T tokens, a 4k context, and grouped-query attention for the larger models (§2). A 34B variant is reported but not released (§1).

## Key Contributions
- A post-training pipeline of SFT followed by iterative RLHF with rejection-sampling fine-tuning and PPO, where new preference data is collected with the latest chat model before each tuning iteration (§3.2.1, §3.2.3, Figure 4).
- Two reward models (Helpfulness RM and Safety RM) and a piecewise reward that uses the Safety RM on safety-tagged prompts or low-safety responses (§3.2.2, §3.2.3).
- A rating-based margin in the pairwise ranking loss, with ablations (Eq. 2, Tables 27–28).
- Ghost Attention (GAtt), a data construction method that keeps a first-turn instruction in effect across later turns (§3.3, A.3.5).
- Safety fine-tuning with adversarial SFT data, a safety reward model, targeted context distillation, and red teaming, with a safety-data scaling study (§4.2, §4.3).

## Key Figures/Tables to Study
- Figure 4: overall training flow (pretraining → SFT → rejection sampling and PPO with iterative reward modeling).
- Tables 6 and 26: preference data statistics, total and per weekly batch.
- Tables 7 and 8: RM accuracy against baselines and by preference rating.
- Figures 7 and 8: best-of-N reward gain and the effect of temperature.
- Figure 11: win rate vs ChatGPT across SFT-v1 to RLHF-v5 under RM and GPT-4 judges.
- Table 30 and Figure 10: GAtt multi-turn results and attention maps.
- Figures 15 and 16: safety data scaling and context distillation analysis.

## Technical Details
Recipe values with loci are in [[llama-2-recipe]]. This section summarizes the mechanisms.

**Pretraining.** The architecture follows Llama 1 (RMSNorm pre-normalization, SwiGLU, RoPE); the changes are a 4k context and GQA for 34B and 70B (§2.2, Table 1). Training uses 2T tokens of publicly available data with the most factual sources up-sampled and no Meta user data (§2.1). Training loss had not saturated at 2T tokens (Figure 5). Pretraining used 3.3M A100-80GB GPU hours in total (Table 2). The data cutoff is September 2022 (Table 52). English accounts for 89.70% of documents by language identification (Table 10).

**SFT.** The SFT stage started from public instruction data (Chung et al., 2022). The authors found many third-party datasets lacked diversity and quality for dialogue, set aside millions of such examples, and collected their own vendor annotations, stopping at 27,540 (§3.1). They report that "SFT annotations in the order of tens of thousands was enough to achieve a high-quality result" (§3.1). In a manual check of 180 examples, SFT-model outputs were often competitive with human-written answers, so annotation effort shifted to preference data (§3.1). Prompts and answers are concatenated to fill the 4096-token sequence, and loss is computed only on answer tokens (§3.1).

**Preference data.** Annotators write a prompt and choose between two responses sampled from two model variants with varied temperature, and rate the preference as significantly better, better, slightly better, or negligibly better/unsure (§3.2.1). Safety annotations add a label: 18% of pairs have a safe chosen and unsafe rejected response, 47% both safe, and 35% both unsafe; pairs with an unsafe chosen and safe rejected response are excluded (§3.2.1). Data arrived in 14 weekly batches; later batches contain more "negligibly better/unsure" ratings as the sampled responses improved (A.3.1, Figure 25), and prompts became more complex over time (A.3.2).

**Reward models.** Each RM is initialized from a pretrained chat model checkpoint so that the RM "knows" what the chat model knows; the next-token head is replaced by a scalar regression head (§3.2.2). The loss is the binary ranking loss −log σ(r_θ(x, y_c) − r_θ(x, y_r) − m(r)), where r_θ(x, y) is the scalar score for prompt x and response y, y_c is the chosen response, y_r is the rejected response, and m(r) is a discrete margin set by the preference rating (Eq. 1–2). Each RM scores best on its own test set: Helpfulness RM 63.2 on Meta Helpfulness and Safety RM 64.5 on Meta Safety, compared with GPT-4 at 58.6 and 58.1 (Table 7). Accuracy on "significantly better" pairs is 80.7 and on "negligibly better/unsure" pairs 54.7 for the Helpfulness RM on Meta Helpfulness (Table 8). The authors attribute the per-domain advantage to tension between helpfulness and safety, citing Bai et al. (2022a) (§3.2.2, A.4.1).

**Iterative RLHF.** Successive versions RLHF-V1 to V5 are trained as new preference batches arrive (§3.2.3). Rejection-sampling fine-tuning samples K outputs per prompt, keeps the highest-reward output, and fine-tunes on it; PPO optimizes R(g | p) = R̃c(g | p) − β D_KL(πθ(g | p) ‖ π0(g | p)), where p is a prompt, g a generation, πθ the policy, π0 the original policy, β the KL coefficient, and R̃c the whitened logit of the piecewise safety/helpfulness reward (Eq. 3–4). The report states that only rejection sampling was used "until RLHF (V4)" and that afterwards PPO was applied on top of the rejection-sampling checkpoint (§3.2.3). Rejection sampling is run only with the 70B model; smaller models are fine-tuned on its samples (§3.2.3). For the RLHF model the best temperature for 10–100 samples is T ∈ [1.2, 1.3], and the optimum shifts across iterations (Figure 8). Model versions were selected with RM rewards and validated by human evaluation (§3.4.1).

**Ghost Attention.** Given a dialogue [u1, a1, …, un, an], an instruction inst (for example "act as" a public figure) is concatenated to every user message, the latest RLHF model samples responses, inst is then kept only in the first turn, and the loss on all tokens from previous turns is set to 0 (§3.3). Constraints are hobbies, language, and public figure, with lists generated by Llama 2-Chat, randomly combined, and shortened half of the time (§3.3). GAtt was applied after RLHF V3 (§3.3).

**Safety fine-tuning.** Safety uses three steps: adversarial prompts with safe demonstrations in SFT, a Safety RM with adversarial prompts in rejection sampling and PPO, and context distillation with a safety preprompt such as "You are a safe and responsible assistant" (§4.2). Distilled outputs are kept only for adversarial prompts and only when the Safety RM scores them higher than the original answer, because on helpful prompts it increased false refusals and on answers that were already high quality it produced less pertinent, generic replies (§4.2.4, Figure 16b). Red teaming involved over 350 people; for the 7B model the robustness metric γ (violating prompts per person per hour) went from 1.8 to 0.45 across iterations (§4.3).

## Recipe ledger
Full table (pretraining, SFT, reward models, rejection sampling, PPO, GAtt, safety, selection): [[llama-2-recipe]].

## Findings relevant to generality, negative feedback, long context, distillation
- **Generality and forgetting.** Training RLHF-V3 only on V2 samples reduced some capabilities (rhyming lines in poems); the authors cite forgetting and report that pooling top samples from all prior iterations addressed it, without figures (§3.2.3). RLHF reduced Self-BLEU diversity on factual prompts but kept diversity on creative prompts across temperatures (Figure 21, §5.1). Pretraining data was not additionally filtered for toxicity (about 0.2% of documents score ≥ 0.5 with HateBERT), which the authors state makes the model usable across more tasks and lets safety tuning generalize with fewer examples (§4.1, Figure 13; Interpretation). Human evaluation used ~4,000 prompts with no coding or reasoning prompts and rated only the final turn (§3.4.2). In the safety-data scaling study the mean helpfulness RM score stayed constant as safety data went from 0% to 100% of ∼0.1M samples with ∼0.9M helpfulness samples fixed (Figure 15, §4.2.3). False refusal on the helpfulness set was about 0.05% at 100% safety data and higher on a 210-prompt borderline set (§4.2.3).
- **Negative feedback.** Rejected responses enter only the reward-model loss (negative as gradient for the RM); rejection sampling discards non-best samples (negative marginal value) (§3.2.2, §3.2.3). Pairs with an unsafe chosen response are excluded from safety data (§3.2.1). The margin term raised accuracy on separable pairs but a large margin lowered accuracy on similar pairs and pushed reward scores toward a binary split, which the authors flag as a calibration risk for PPO (A.3.3, Table 28, Figure 27). The KL penalty is reported to improve stability and reduce reward hacking (§3.2.3).
- **Long context and multi-turn.** Without GAtt, attribute recall fell from 100% at turn 2 to 10% at turn 4 and 0% at turn 6; with GAtt it stayed at 100% up to 20 turns, with all evaluated dialogues under 4048 tokens (Table 30, A.3.5). GAtt generalized at inference to constraints not seen in GAtt training, such as "Always answer with Haiku" (§3.3). A 2k-vs-4k context ablation at 150B tokens improved SCROLLS-style tasks without degrading SQuAD (A.2.1, Tables 16–17).
- **Distillation.** Rejection-sampled outputs of the 70B model are used to fine-tune the smaller chat models; the authors leave analysis of this distillation for future work (§3.2.3).
- **Agentic and tool use.** The authors report zero-shot tool use emerging without tool-use annotation; with a calculator, Llama 2-Chat scores 67.1 on ASDiv, 69.2 on SVAMP, and 82.4 on MAWPS (§5.1, Table 15).

## Connections
- [[llama-2-recipe]] holds the verified hyperparameters for every stage.
- [[llama-3]] states it follows Llama 2 in running iterative rounds (six in Llama 3), replaces PPO with DPO, and removes the RM margin term (Llama 3 report §4.1.2, §4.1.4, §4.1.6).
- [[rlhf-instructgpt]] (Ouyang et al., 2022) is cited for the binary ranking loss (Eq. 1) and, with Stiennon et al. (2020), for the KL-penalized RL setup (§3.2.2, §3.2.3).
- [[ppo]] is the policy-optimization algorithm used after the rejection-sampling-only versions.
- [[hh-rlhf]] (Bai et al., 2022a) is the citation for the helpfulness–safety tension that motivates two RMs, and Anthropic Harmless is part of the Safety RM mix.
- [[constitutional-ai]] (Bai et al., 2022b) is cited for best-of-K rejection sampling, as the inspiration for GAtt, and for some safety preprompts (Table 39).
- [[rejection-sampling-finetuning]] covers the method used in RLHF-V1 onward.
- [[lima]] reports a related finding that a limited set of clean instruction data can suffice; the report cites it as "similar in spirit" (§3.1).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2307.09288 (v2, 2023-07-19)
- Corrections to the previous card version:
  - "RSFT is a lightweight replacement for early PPO iterations" / "Run RSFT for the first RLHF iterations, add PPO only in the last round" → the report states only that rejection sampling alone was used until RLHF (V4) and PPO was then applied on the rejection-sampling checkpoint; it makes no replacement or cost claim (§3.2.3).
  - "PPO added in the last two rounds / V4 and V5" → the sentence "Until RLHF (V4)…" does not say whether V4 itself used PPO; Figure 11 shows RLHF-v5 with and without PPO (§3.2.3).
  - "five successive checkpoints with fresh weekly batches" → preference data came in 14 weekly batches; five RLHF versions were trained as batches accumulated (A.3.1, §3.2.3).
  - "Each RM initialized from the LM pre-trained base (70B RM is best)" → initialized from pretrained chat model checkpoints; Figure 6 shows larger RMs more accurate at equal data (§3.2.2).
  - "combined with a piecewise schedule (safety dominates on safety prompts)" / "a rule selects which RM (or a weighted combo)" → Rs is used if the prompt is safety-tagged or Rs < 0.15, else Rh; no weighted combination; the result is whitened after a logit transform (§3.2.3).
  - "Table 11 (RM accuracy): Helpfulness RM matches GPT-4" → Table 7: Helpfulness RM 63.2 vs GPT-4 58.6 on Meta Helpfulness; Table 11 covers pretrained-model safety benchmarks.
  - "Figure 7 (dual-RM scoring)" → Figure 7 plots max and median reward among N samples; the RM combination is an equation in §3.2.3.
  - "quality >> quantity past ~10K" → "SFT annotations in the order of tens of thousands was enough"; collection stopped at 27,540 (§3.1).
  - "SFT LR 2e-5 (70B), cosine decay" → LR 2 × 10^-5 is not scoped to 70B; also weight decay 0.1, batch 64, sequence length 4096, 2 epochs (§3.1).
  - "per-prompt two candidate responses from different Llama 2 variants" → two responses from two model variants with varied temperature (§3.2.1).
  - "PPO learning rate 1e-6 (policy) for 70B" → constant 10^-6 for all models (§3.2.3).
  - "KL coefficient beta 0.01" → 0.01 for 7B and 13B; 0.005 for 34B and 70B (§3.2.3).
  - "Context distillation: 'you are a safe assistant' preamble" → "You are a safe and responsible assistant", applied only to adversarial prompts and kept only when the Safety RM score improves (§4.2.4).
  - "Red-teaming across 350+ adversaries" → "over 350 people" (§4.3).
  - "[[constitutional-ai]] — cited as motivation for dual RMs" → the tension citation is Bai et al. (2022a), [[hh-rlhf]]; Constitutional AI is cited for rejection sampling, GAtt, and preprompts (§3.2.2, §3.2.3, §3.3).
- Removed as unsupported by the source: "K ~ 10+"; PPO "Sequence length 4K"; "value function, GAE"; "RLHF takes weeks of weekly iterations"; "RSFT lineage Llama 2 popularized"; "resolving the tension Anthropic documented".
- Not reported by the source: pretraining mixture percentages; rejection-sampling K and fine-tuning hyperparameters; PPO maximum response length and sampling temperature; number of GAtt examples; which margin variant the final RMs use; safety auxiliary loss formula.
