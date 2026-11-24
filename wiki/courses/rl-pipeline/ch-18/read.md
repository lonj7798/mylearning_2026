<!-- chapter: ch-18
     track: rl-pipeline
     kind: content
     title: Choosing the RL Algorithm by Domain: PPO, GRPO, RLOO, DAPO, GSPO, CISPO, DPO, and Agentic Variants
     deps: [ch-17]
     sources: [[algorithm-crosswalk]], [[recipe-crosswalk]], [[ppo]], [[grpo]], [[rloo]], [[rloo-vs-grpo]], [[reinforce-plus-plus]], [[dr-grpo]], [[dpo]], [[gigpo-verl-agent]], [[minimax-m1]], [[scalerl]], [[olmo-3-rl-pipeline]], [[mai-rl-infrastructure]], [[dapo]], [[mai-thinking-1]], [[search-r1]], [[gspo]], [[vapo]], [[deepseek-v3.1]], [[glm-4-5]], [[qwen-3]], [[qwen-3-recipe]], [[kimi-k2]], [[magistral]], [[nemotron-ultra-recipe]], [[tulu-3]], [[rlvr-tulu3]], [[ifbench]], [[websailor-tongyi-deepresearch]], [[deepresearcher]], [[asearcher]], [[zerosearch]], [[deepswe]], [[swe-rl]], [[rstar2-agent]], [[olmo-3-recipe]], [[open-instruct-rlvr-pipeline]]
     figures: figures/algorithm-by-domain.html
-->

# Chapter 18 — Choosing the RL Algorithm by Domain: PPO, GRPO, RLOO, DAPO, GSPO, CISPO, DPO, and Agentic Variants

> **Core insight.** Every lab recipe carded here runs RL at scale with a critic-free group-baseline method: GRPO or a modification of it, CISPO, or Kimi's K1.5 objective. PPO with a value model appears only in Tülu 3 RLVR, Search-R1, InstructGPT-style RLHF and VAPO ([[algorithm-crosswalk]] Core Insight). Labs often do not state why they chose the base algorithm: in 11 of the crosswalk's 20 lab-by-stage rows (count by this chapter: GLM-4.5 ×3, Qwen3 ×2, Kimi K2, DeepSeek-V3.2, Olmo 3 RL, Nemotron RLVR, Magistral, DeepResearcher/ASearcher) the reason for the base algorithm is not stated. Stated reasons attach to the **modifications**, and each is tied to an observed failure: token clipping that drops low-probability "fork" tokens (CISPO, [[minimax-m1]] §3.1), MoE routing drift that invalidates token-level ratios (GSPO and Keep Routing), length and std-normalization bias (Dr. GRPO), zero-gradient groups that slow multi-turn rollouts (DUPO), and off-policy negatives that destabilize training (DeepSeek's sequence masking, Tongyi's negative exclusion). Domain changes which failure is likely, and so which modification matters, more than it changes the base algorithm.
>
> **Guideline.** Choose by the failure you expect, and cite a lab's choice only when its card records a reason. For verifiable math and code from a base model, start from GRPO without std division and with token-level loss, clip-higher and zero-variance filtering, because [[dapo]] measured AIME24 30 → 50 as its changes were added (Table 1) and [[dr-grpo]] measured shorter incorrect responses without lower accuracy (Fig. 5). If each rollout batch takes many off-policy updates, use an IS-weight clip that keeps every token's gradient (CISPO), because [[minimax-m1]] matched DAPO's AIME score in 50% of the steps (Fig. 2), and [[scalerl]] found CISPO and GSPO raised the fitted asymptote above DAPO (Fig. 5a). For large MoE policies, replay routing or use sequence-level ratios, because GRPO needed Routing Replay to converge on Qwen3-30B-A3B ([[gspo]] §5.3). For multi-turn agents with recurring states, add step-level credit, because GiGPO raised ALFWorld from 72.8 to 86.1 at 1.5B ([[gigpo-verl-agent]] Table 1). Keep PPO as an option when a group method collapses on long runs (Search-R1, §5.1) or when a value model can be warmed up (VAPO 60.4 vs DAPO 50, Table 1). For instruction following with verifiable constraints, prefer online RL over DPO on the same data, because GRPO beat DPO 89.65 vs 79.67 on IFEval ([[ifbench]] Table 5).

---

## Why this chapter matters

llm-training ch-40 and ch-45a derive the algorithms. This chapter asks a different question: given a domain and a scale, which algorithm did labs use, what reason did they give, and what reason did they give for not using the alternatives? The answer is a crosswalk, built from [[algorithm-crosswalk]], with every "not stated" kept.

The answer matters for research because published algorithm comparisons are narrow. Most come from one model family on one domain. A lab's choice at scale is a second kind of evidence, but only when the lab says why. When it does not, the choice may come from infrastructure, history or convenience. Treating unexplained choices as evidence is the main error this chapter guards against.

For the applied target of this course, a sales voice agent whose rewards mix verifiable checks with subjective judges, the relevant rows are instruction following, multi-turn agents and non-verifiable preference (§2.1, §2.3, §2.5).

[figures/algorithm-by-domain.html](figures/algorithm-by-domain.html) is a filterable version of the full table. Filter by domain or by whether a reason is stated. Use it while reading §2.

---

## §1 How to read the crosswalk

Each row of [[algorithm-crosswalk]] Table 1 has five fields: the algorithm used, its modifications, the stated reason for the choice, the alternatives rejected, and the stated reason for rejecting them. Three distinctions keep the reading honest:

1. **Base algorithm vs modification.** "GRPO" in a lab report is usually GRPO with three to seven changes. Olmo 3's OlmoRL is GRPO with zero-gradient filtering, active sampling, token-level loss, no KL, clip-higher, truncated IS and no std normalization ([[olmo-3-rl-pipeline]] §4.4.1). The modifications carry the stated reasons.
2. **Stated reason vs measured reason.** A stated reason can be a sentence with no ablation ("crucial for RL training stability", DeepSeek Keep Routing). A measured reason has a number (CISPO's 50% of steps). The table marks both kinds.
3. **Within-lab vs cross-lab evidence.** A lab that tries two algorithms on its own setup gives within-lab evidence (MiniMax: GRPO vs DAPO vs CISPO). Two labs using different algorithms on different data give no comparison at all.

---

## §2 By domain

### §2.1 Instruction following

| Lab | Used | Why (stated) | Why not others (stated) | Source |
|---|---|---|---|---|
| IFBench (Ai2) | GRPO, multi-constraint sum reward | GRPO > DPO on the same prompts and verifiers | DPO: IFEval 79.67 vs 89.65 for GRPO (both after DPO); 76.89 vs 85.77 after SFT | [[ifbench]] §4.6, Table 5 |
| Tülu 3 | PPO with a binary verifier (α = 10), value model initialized from a general RM | KL kept high enough that KL stays small (average drops as KL grows at low β) | not stated (PPO "chosen for RLVR in §5.4.1"; no comparison) | [[rlvr-tulu3]] §6, Fig. 21 |
| Olmo 3 | OlmoRL (GRPO + DAPO + Dr. GRPO changes) | KL removal "does not lead to over-optimization or destabilized training" | PPO (used in OLMo 2 / Tülu 3): replaced, reason not stated | [[olmo-3-rl-pipeline]] §4.4.1 |
| GLM-4.5 | GRPO (IF), with rules + RM + critique model | not stated | not stated | [[glm-4-5]] §3.4 |
| Kimi K2, Nemotron | inside joint RL (K1.5 objective; synchronous GRPO with masked IS) | not stated for IF | not stated | [[kimi-k2]], [[nemotron-ultra-recipe]] |
| MAI Thinking 1 | the shared GRPO recipe (§3), with lexicographic reward combination | reward combination stated; algorithm is the same as in other climbs | not stated | [[mai-thinking-1]] Sec. 3.4 |

**Reading.** The one measured comparison for IF is IFBench's GRPO > DPO on identical data. Ai2 moved from PPO (Tülu 3) to GRPO variants (Olmo 3) without a stated reason. For IF, the reward design (ch-14) matters more than the algorithm in every source read.

### §2.2 Verifiable math and code

| Lab | Used | Why (stated) | Why not others (stated) | Source |
|---|---|---|---|---|
| DAPO | GRPO + clip-higher, dynamic sampling, token-level loss, overlong shaping | each change measured: AIME24 30 → 36 → 38 → 41 → 42 → 50 cumulatively | — | [[dapo]] Table 1 |
| Dr. GRPO | GRPO without 1/\|o_i\| and without std division | the two divisions bias the gradient; removing them stops incorrect-response length growth | — | [[dr-grpo]] §3.1–3.2 |
| MiniMax-M1 | CISPO (clip the IS weight, keep every token's gradient; no KL) | GRPO clipping removed low-probability reflective tokens after the first of 16 off-policy updates; CISPO matched DAPO in 50% of steps (Qwen2.5-32B-base) | GRPO "adversely affected training performance and failed to effectively promote the emergence of long CoT"; DAPO clip-higher "less effective" with 16 off-policy updates per batch; KL: reason not stated | [[minimax-m1]] §3.1, Fig. 2 |
| Magistral | GRPO, no KL, loss over total group tokens, minibatch advantage normalization, ε_high 0.26–0.28 (0.3 for Small), drop zero-advantage groups | ε_high keeps group entropy stable | KL: "policy diverges substantially regardless", compute "unjustified"; EMA reference and entropy bonus: ε_high simpler; proportional code reward: −2% LiveCodeBench | [[magistral]] §2.1, §6.4, §7.4 |
| GLM-4.5 | GRPO, no KL; token-weighted mean loss for code | token-weighted loss: faster convergence on LiveCodeBench (Fig. 7) | multi-stage length schedule: 80.6 vs 83.4 AIME24 single-stage (Fig. 6); KL: not stated | [[glm-4-5]] §3.2 |
| Qwen3 | GRPO, large batch, many rollouts per query, off-policy training | "beneficial" (no numbers) | not stated | [[qwen-3]] §4.2 |
| Qwen3 lightweight (0.6B–30B-A3B) | on-policy distillation (KL to teacher logits) instead of RL | Qwen3-8B: AIME'24 74.4 vs 67.6 pass@1, 93.3 vs 90.0 pass@64, 1,800 vs 17,920 GPU hours | RL left pass@64 unchanged from the start, at about 10× the GPU hours | [[qwen-3-recipe]] Table 21 |
| VAPO | PPO with value pretraining and length-adaptive GAE | AIME24 60.4 vs DAPO 50 on the same base; removing value pretraining → 11 | — (one task family, no seeds) | [[vapo]] Table 1 |
| rStar2-Agent (tool-integrated math) | GRPO-RoC | positives with tool errors are still reinforced under answer-only reward; RoC filters positives | step-level rewards and tool-error penalties: "prone to reward hacking", no experiment | [[rstar2-agent]] §2.2.3 |

**Reading.** This is the domain with the most measured evidence, and the evidence supports the modifications, not "GRPO vs PPO". The only direct critic-vs-group comparison on long CoT is VAPO, where the critic wins (60.4 vs 50) once value pretraining is added. Without the VAPO fixes, vanilla PPO scored 5. [[grpo]] itself gives no accuracy evidence for GRPO over PPO; its stated reason is the cost of a policy-sized value model ([[grpo]] Guideline).

### §2.3 Search and multi-turn agents

| Source | Used | Why (stated) | Why not others (stated) | Location |
|---|---|---|---|---|
| Search-R1 | PPO (default), retrieved-token mask | PPO "greater training stability" | GRPO converged faster but showed reward collapse in long runs. GRPO scored higher in 3 of 4 settings (7B-instruct 0.396 vs 0.385; 3B-base 0.312 vs 0.303; 3B-instruct 0.336 vs 0.325) and lower on 7B-base (0.350 vs 0.431) | [[search-r1]] §5.1, Table 3 |
| ZeroSearch | REINFORCE (default) | REINFORCE 34.47, GRPO 33.17, PPO 32.67 (Qwen-2.5-3B-Base) | — | [[zerosearch]] Table 5 |
| DeepResearcher, ASearcher | GRPO with observation masking; dynamic filtering (ASearcher) | not stated | not stated | [[deepresearcher]] §3.3; [[asearcher]] §3.4 |
| WebSailor | DUPO (GRPO + DAPO token loss, clip-higher; duplicate non-zero-std groups instead of resampling) | about 2–3× faster than DAPO dynamic sampling | DAPO dynamic sampling: sequential resampling slows agentic RL | [[websailor-tongyi-deepresearch]] WebSailor §4.2 |
| Tongyi DeepResearch | GRPO variant, strict on-policy; token-level loss, clip-higher, leave-one-out baseline, exclude some negatives | LOO: reduce advantage variance; negative exclusion: unfiltered negatives "can lead to policy collapse" | "success … depends more on data and environment stability than on the specific algorithm" | Tongyi §3.4.3 |
| GiGPO (verl-agent) | GRPO episode advantage + anchor-state step advantage | ALFWorld 72.8 → 86.1, WebShop success 56.8 → 67.4 at 1.5B, 3 seeds; extra compute 0.54 s vs 362.83 s per iteration | needs recurring states; not applicable when states do not repeat | [[gigpo-verl-agent]] Table 1, §5.6 |
| GLM-4.5 agentic RL | group-wise policy optimization (reward − group mean); loss on model-generated tokens | not stated | not stated | [[glm-4-5]] §3.3.2 |

**Critic vs group for agents.** On ALFWorld, PPO scores 54.4 at 1.5B against GRPO 72.8 and RLOO 69.7, and 80.4 at 7B against GRPO 77.6 and RLOO 75.5 (3 seeds, [[gigpo-verl-agent]] Table 1). The ranking reverses with scale in one benchmark. In search, Search-R1 prefers PPO for stability and ZeroSearch ranks REINFORCE first. No source supports a general claim that one family wins for agents.

**Throughput is the agent-specific reason.** DUPO's stated reason is wall-clock time: resampling a zero-variance group means another multi-turn rollout. Rocket names a related systems cost of group methods: "GRPO waits for the whole group", so one slow rollout delays its group's training ([[mai-rl-infrastructure]] p. 1; Fig. 4, where the fastest member waited 7.6 h). A value-based method does not need sibling rollouts to compute an advantage, but it pays for a critic of policy size. No source compares the two on staleness.

### §2.4 SWE

| Source | Used | Why (stated) | Why not others (stated) | Location |
|---|---|---|---|---|
| DeepSWE | GRPO++: clip-high, no KL, no std, length normalization by max context, leave-one-out, compact filtering, no entropy loss | std removal stated to remove difficulty bias; compact filtering prevents or delays collapse (Fig. 6) | entropy loss removed (reason not given in the card beyond listing) | [[deepswe]] §2.3 |
| SWE-RL | GRPO with a similarity reward, no execution | continuous reward beats discrete exact match, which stays near zero | no execution-reward baseline | [[swe-rl]] §3.6 |
| MiniMax-M1 | CISPO (same as all its domains) | §2.2 reasons | §2.2 | [[minimax-m1]] |
| GLM-4.5, Kimi K2 | group-wise outcome RL on test results, asynchronous infra | not stated | not stated | [[algorithm-crosswalk]] Domain → algorithm |

**Reading.** No card states an SWE-specific reason for the base algorithm ([[algorithm-crosswalk]] SWE row). The SWE-specific choices are about failures and timeouts (ch-16 §4.4) and about advantage computation for sparse binary rewards (leave-one-out, no std).

### §2.5 Non-verifiable chat preference

| Source | Used | Why (stated) | Why not others (stated) | Location |
|---|---|---|---|---|
| Tülu 3 | length-normalized DPO before RLVR | selected over standard DPO and SimPO (Table 18) | — | [[tulu-3]] |
| Olmo 3 | DPO with delta-learning pairs (chosen from a large model, rejected from a small one) | pair quality depends on the contrast, not absolute quality | — | [[olmo-3-recipe]] §4.3 |
| Nemotron 3 Nano | RL against a GenRM with circular pairwise comparisons; group-relative length control | verbosity −30% "without sacrificing accuracy" | DPO against tool hallucination tested (AIME25 80.88 → 84.58) but "released model does not rely on DPO"; reason not stated | [[nemotron-ultra-recipe]] NR §3.3, App. C |
| DeepSeek-V3.2 | GRPO with a rubric GenRM inside mixed RL | joint stage reason (§2.6 below) | not stated | [[deepseek-v3.1]] §3 |
| Kimi K2 | K1.5 objective with a self-critique rubric reward | not stated (K1.5 card: no MCTS, value functions or PRMs, also without a reason) | not stated | [[kimi-k2]] §3.2.2–3.2.3 |
| MiniMax-M1 | CISPO with GenRM (five-grade with ground truth; pairwise −1/0/1 vs reference without) | GenRM length bias must be monitored online; offline mitigation "frequently failed" | — | [[minimax-m1]] §4.2 |
| RLOO / REINFORCE++ (papers) | RM-based online RLHF | RLOO k = 4: win rates 77.9 / 43.7 / 64.1 vs PPO 67.6 / 29.2 / 32.0; REINFORCE++ k = 1 ties GRPO k = 4 on Arena-Hard (46.7 vs 46.8) | no lab in the crosswalk states choosing or rejecting them | [[rloo]] Table 1; [[reinforce-plus-plus]] Table 1 |

**DPO vs online RL.** Labs that use DPO (Tülu 3, Olmo 3) place it **before** online RL, as an offline preference stage. The one same-data comparison (IFBench) favours online GRPO, on a verifiable domain. For non-verifiable chat, no card compares DPO and online RL on the same data. The pattern in recent recipes is to move the non-verifiable signal into online RL through a generative or rubric reward (Nemotron, DeepSeek, MiniMax, Kimi). Their stated reasons concern reward-model failure modes (length bias, hacking), not DPO.

### §2.6 Large MoE policies

| Source | Used | Why (stated) | Location |
|---|---|---|---|
| GSPO | sequence-level importance ratio | about 10% of activated experts change after one gradient update for the same rollout on Qwen3-30B-A3B, invalidating token-level ratios; GRPO needed Routing Replay to converge, GSPO did not | [[gspo]] §5.3 |
| DeepSeek-V3.2 | GRPO + Keep Routing (reuse inference-time expert routes), Keep Sampling Mask, unbiased K3 KL, off-policy sequence masking | Keep Routing "crucial for RL training stability of MoE models"; masking: "highly off-policy negative samples can be detrimental"; for math, weak or zero KL "can perform better" | [[deepseek-v3.1]] §3.1 |
| Nemotron 3 Nano | synchronous GRPO with masked IS; MoE router weights frozen; aux-loss-free bias still updated | batch 2,048 "making our updates on-policy"; overlong filtering "boosts performance on reasoning intensive benchmarks" | [[nemotron-ultra-recipe]] NR §3.2.5 |
| MAI Thinking 1 | GRPO with MoE routing replay and top-p mask replay; BF16 sampler and learner | without top-p mask replay: "catastrophic off-policy mismatch, causing divergence within a few training steps" | [[mai-thinking-1]] Sec. 3.1.3, 3.6 |
| MiniMax-M1 (456B hybrid) | CISPO | §2.2 | [[minimax-m1]] §3.1 |

**Reading.** All large-MoE labs keep a token-level group method and add routing control (replay, freeze) or a sequence-level ratio. GSPO's measurement is the reason routing control is needed. No lab in the crosswalk states adopting GSPO itself ([[algorithm-crosswalk]] Table 2).

---

## §3 The rejection reasons, by mechanism

| Rejected choice | Stated or measured failure | Replacement | Source |
|---|---|---|---|
| PPO's value model | cost: policy-sized critic "too costly"; vanilla PPO on long CoT: AIME24 5 (value-model collapse) | group baseline (GRPO, RLOO); or keep the critic with value pretraining (VAPO 60.4) | [[grpo]]; [[vapo]] Table 1 |
| GRPO's 1/\|o_i\| and std division | length growth of incorrect responses; difficulty bias | constant normalizer, no std (Dr. GRPO); adopted by Olmo 3, DeepSeek-V3.2 (Â = R − mean), DeepSWE | [[dr-grpo]] Fig. 5, 8–9 |
| Token-level ratio on MoE | ~10% expert change per update | sequence ratio (GSPO) or routing replay / freeze | [[gspo]] §5.3 |
| PPO/GRPO token clipping | fork tokens (However, Recheck, Wait, Aha) lose gradient after the first of 16 off-policy updates | clip the stop-gradient IS weight, keep the token (CISPO) | [[minimax-m1]] §3.1 |
| Symmetric clip ε | entropy collapse | clip-higher ε_high = 0.28 (+2 AIME24 in DAPO's cumulative ablation) | [[dapo]] Table 1 |
| Entropy bonus | entropy fell on math-only and exploded on math + code at one coefficient | higher ε_high (Magistral); entropy-targeted adaptive upper clip (MAI) | [[magistral]] §7.4.2; [[mai-thinking-1]] Sec. 3.1.1 |
| DAPO dynamic resampling in agents | sequential resampling slows multi-turn RL | duplicate non-zero-std groups (DUPO) | [[websailor-tongyi-deepresearch]] §4.2 |
| Episode-level credit only | no step-level signal in long agent episodes | anchor-state step groups (GiGPO) | [[gigpo-verl-agent]] Table 1 |
| Unfiltered off-policy negatives | instability and policy collapse | off-policy sequence masking (DeepSeek); negative exclusion (Tongyi) | [[deepseek-v3.1]] §3.1; Tongyi §3.4.3 |
| DPO on verifiable IF | IFEval 79.67 vs 89.65 for GRPO | online RL | [[ifbench]] Table 5 |
| RL for small models when a teacher exists | pass@64 unchanged; 10× the GPU hours | on-policy distillation | [[qwen-3-recipe]] Table 21 |

### §3.1 Clipped-token gradient loss in code

open-instruct implements DAPO-style clipping and CISPO side by side, which makes the mechanism in the CISPO row visible:

```python
# open-instruct@f020e9a open_instruct/grpo_utils.py L547-L561
if config.loss_fn == GRPOLossType.dapo:
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - config.clip_lower, 1.0 + config.clip_higher)
    clipfrac = (pg_losses2 > pg_losses).float()
    pg_loss = torch.max(pg_losses, pg_losses2)
elif config.loss_fn == GRPOLossType.cispo:
    # cispo: directly clip ratio, no lower bound.
    # reinforce loss, so multiply by new logprobs
    clipfrac = (ratio > 1.0 + config.clip_higher).float()
    pg_loss = -advantages * torch.clamp(ratio.detach(), max=1.0 + config.clip_higher) * new_logprobs
else:
    raise ValueError(f"Invalid loss function: {config.loss_fn}")

pg_loss *= rho_weights
clipfrac *= (rho_weights != 0).float()
```

In the DAPO branch, when the clipped term is selected by `torch.max`, the loss is constant in θ for that token and its gradient is zero. A token whose ratio has moved past 1 + clip_higher with positive advantage stops receiving gradient. In the CISPO branch, the ratio is detached and clipped, and the gradient flows through `new_logprobs` for every token, so a clipped token still gets gradient with a bounded weight. MiniMax-M1's fork-token argument is about exactly this difference ([[minimax-m1]] §3.1). The CISPO branch has no lower bound ("no lower bound"), matching M1's large ε_low^IS. The defaults are `loss_fn = dapo`, `clip_lower = 0.2` and `clip_higher = 0.272` (`grpo_utils.py` L125–137, L169), the values in Olmo 3's Table 49 ([[olmo-3-rl-pipeline]]).

### §3.2 Std normalization in code

verl exposes Dr. GRPO's std removal as one flag in its GRPO advantage function:

```python
# verl@8357f1e verl/trainer/ppo/core_algos.py L324-L329
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    else:
        scores[i] = scores[i] - id2mean[index[i]]
scores = scores.unsqueeze(-1) * response_mask
```

The generated default config sets `norm_adv_by_std_in_grpo: true` (`_generated_ppo_trainer.yaml` L864), which is original GRPO. A "GRPO" run on verl with defaults is therefore not the Dr. GRPO / DeepSeek-V3.2 / Olmo 3 variant. verl at this commit registers several estimators in one enum: GAE, GRPO, REINFORCE++, REINFORCE++ with baseline, ReMax, RLOO, OPO, GRPO pass@k, GPG, vectorized RLOO and GRPO, optimal-token-baseline variants and GDPO (`core_algos.py` L88–110), with separate example directories for CISPO and GSPO trainers (`examples/cispo_trainer/`, `examples/gspo_trainer/`).

---

## §4 Rocket and MAI's choice

MAI Thinking 1 uses GRPO with token-level normalization over all tokens in the global batch (across DP ranks) and advantage A_i = (R_i − mean)/std, i.e. **with** std division ([[mai-thinking-1]] Sec. 3.1.1). Two changes are stated with reasons:

1. **Entropy-targeted upper clip.** clip(r, 1 − ε, (1 − ε)^−1 + k), where k starts at 0 and is updated by k ← clip(k + δ·sign(H* − Ĥ), 0, k_max), with Ĥ an importance-weighted per-token entropy estimate. It replaces an entropy bonus (Eq. 7–8). Settings: ε = 0.6, k_max = 2.5, δ = 0.25, H* = 0.3. Derived from these values: the lower bound is 0.4, and the upper bound is 2.5 at k = 0 and 5.0 at k = k_max.
2. **Outer ratio clip on all branches**, clip(r, r_min, r_max) with r_min = 0 and r_max = 50, "similar in spirit to the dual-clip PPO". It stops gradient-norm spikes from the unclipped GRPO cases (Eq. 9).

Other stated parts are the language and length reward terms, the two pass-rate filters, top-p mask replay (without which training diverged "within a few training steps"), and the 8k → 128k length doubling (Sec. 3.1.2–3.1.3).

**Not stated.** Why GRPO rather than PPO, RLOO or CISPO, and why std division is kept when Dr. GRPO, DeepSeek-V3.2 and Olmo 3 remove it. Record these as "not stated".

**How it compares.** MAI's adaptive upper clip and MiniMax's CISPO target the same failure: low-probability tokens that a tight upper clip stops updating. MAI raises the upper bound when entropy falls below target. CISPO keeps the gradient of every token and bounds its weight. ScaleRL's loss comparison (CISPO and GSPO raise the asymptote above DAPO, Fig. 5a) did not include MAI's controller, so no source ranks the two.

**The systems side.** The Rocket post names group completion as the source of agentic staleness: GRPO "waits for the whole group" ([[mai-rl-infrastructure]] p. 1). Its levers (priority, drop-failed, inference-bound sizing, fast weight transfer) are infrastructure answers to an algorithmic requirement. An algorithm that computes advantages without siblings would remove the wait but would need a critic or another baseline. Neither MAI source discusses that alternative.

---

## §5 Domain → algorithm summary

| Domain | Common choice | Stated reason exists for | Measured alternatives |
|---|---|---|---|
| IF | GRPO-family with verifier reward, plus a quality signal | GRPO over DPO (IFBench) | PPO (Tülu 3) without comparison |
| Math / code | GRPO + DAPO / Dr. GRPO changes; CISPO with many off-policy updates | each modification (DAPO, Dr. GRPO, CISPO, Magistral) | VAPO critic beats DAPO on one task family |
| Search / multi-turn | GRPO-family + observation masking; DUPO for throughput; GiGPO for step credit | PPO for stability (Search-R1); DUPO speed; Tongyi LOO and negative exclusion | REINFORCE > GRPO > PPO (ZeroSearch); PPO vs GRPO reverses with scale on ALFWorld |
| SWE | group-wise outcome RL, LOO, no std, compact filtering | compact filtering (DeepSWE) | none for the base algorithm |
| Chat preference | DPO before RL; or RL against GenRM / rubric rewards | delta-learning pairs; length control | RLOO and REINFORCE++ measured in papers, not adopted in carded recipes |
| Large MoE | GRPO + routing replay / freeze, or sequence-level ratio | GSPO's routing-drift measurement; DeepSeek "crucial" (no number) | GRPO needed Routing Replay to converge (GSPO §5.3) |

---

## Recipe

1. **Write down the domain's expected failures** before choosing: long responses (length bias), many updates per batch (clipped tokens), MoE (routing drift), multi-turn (slow resampling, no step credit), noisy or learned rewards (hacking, length bias), off-policy samples (stale negatives).
2. **Start from a group-baseline method** with Dr. GRPO normalization (no std division, constant or token-level normalizer), clip-higher, and zero-variance filtering, unless a value model is affordable and you can warm it up (VAPO).
3. **Set the framework flags explicitly.** verl defaults to std division (`norm_adv_by_std_in_grpo: true`). open-instruct defaults to the DAPO clip (`loss_fn = dapo`, 0.2 / 0.272). Write the chosen values in the run config.
4. **If you take more than a few updates per rollout batch**, log the clipped-token fraction separately for A > 0 and A < 0. If low-probability tokens are being clipped, switch to CISPO-style IS-weight clipping, or raise the upper clip adaptively (MAI).
5. **For MoE**, replay inference routing (or freeze the router), replay the top-p mask, and log expert-activation overlap between π_old and π_θ. Consider sequence-level ratios.
6. **For multi-turn agents**, mask observations; replace resampling with duplication (DUPO) if zero-variance groups are common; add step-level credit if states recur (GiGPO); keep PPO available if group methods collapse on long runs.
7. **For off-policy data** (async rollouts), add truncated IS or masking of high-divergence negatives (DeepSeek's sequence masking).
8. **For non-verifiable preference**, decide between an offline DPO stage before RL (Tülu 3, Olmo 3) and online RL against a GenRM or rubric reward (Nemotron, DeepSeek, MiniMax). For the online route, monitor reward-model length bias online (MiniMax).
9. **For small models with a strong teacher**, compare on-policy distillation against RL at equal GPU hours before committing to RL (Qwen3 Table 21).
10. **Record the reason** for each choice as "measured (source)", "stated (source)" or "not stated", so that later readers can tell evidence from habit.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Citing a lab's base algorithm as evidence | the cited lab gives no reason | look for the reason column in [[algorithm-crosswalk]]; most are "not stated" |
| Running "GRPO" with framework defaults | length of incorrect responses grows; reproduction differs | check std division and loss normalizer flags (verl default divides by std) |
| Token clipping with many off-policy updates | entropy falls; reflective tokens disappear | log clip fraction by advantage sign and by token probability; compare CISPO |
| Token-level ratios on MoE without routing control | sudden collapse after normal early training | log expert overlap per update; enable routing replay |
| DAPO resampling in multi-turn RL | step time grows with zero-variance share | log resampling count per step; switch to duplication |
| Comparing algorithms across labs | conclusions that do not replicate | only use within-study comparisons (same data, same model) |
| Choosing PPO vs GRPO from one benchmark | the ranking reverses at another scale (ALFWorld 1.5B vs 7B) | test at the target scale |
| Using DPO for verifiable constraints | lower IF than online RL on the same data | run the IFBench-style comparison on your data |
| Ignoring the group's systems cost in agents | long waits for the slowest group member; staleness | log per-group wall time (Rocket Fig. 4) |

---

## Check your understanding

1. In open-instruct's DAPO branch, show that a token whose ratio exceeds 1 + clip_higher with positive advantage gets zero gradient, and that the same token gets non-zero gradient in the CISPO branch. Why does this difference grow with the number of updates per rollout batch?
2. MAI keeps group-std division while Olmo 3, DeepSeek-V3.2 and DeepSWE remove it. Using Dr. GRPO's argument, describe the group in which std division changes the advantage most, and what a lab might gain by keeping it.
3. On ALFWorld, PPO is worst at 1.5B and best at 7B. Give one mechanism related to the critic that could produce this reversal, and an experiment that would test it.
4. Search-R1 chose PPO for stability while GRPO scored higher in three of four settings. Explain why a lab can rationally prefer the algorithm with the lower average score, and what measurement would justify that choice.
5. Qwen3's on-policy distillation beat RL on Qwen3-8B with one tenth of the GPU hours and raised pass@64 where RL did not. What does unchanged pass@64 under RL suggest about what RL did, and when would you still choose RL?
6. Rocket attributes agentic staleness to GRPO waiting for the whole group. Compare the costs of removing the group requirement with a learned critic, and of keeping groups with Rocket's infrastructure levers.
7. GSPO reports that about 10% of activated experts change after one update. Explain why this invalidates a token-level importance ratio, and why routing replay and sequence-level ratios address it in different ways.

---

## Connections

- **ch-17** — the stage structure in which these algorithms run.
- **ch-03, ch-07** — staleness and sampler–learner mismatch, which the off-policy corrections address.
- **ch-10, ch-12** — rollout and selection choices (dynamic sampling, zero-variance filtering) that interact with the algorithm.
- **ch-14, ch-15, ch-16** — the per-domain pipelines whose algorithm rows are collected here.
- **ch-19** — non-verifiable rewards, the input to §2.5.
- **llm-training ch-40** (group-baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, GSPO) — the algorithm mechanics this chapter assumes.
- **llm-training ch-45a** (preference-optimization and RL stage recipes side by side) — DPO-family and RL stage recipes.
- **llm-training ch-43** (entropy, output diversity and KL control) — entropy controls behind clip-higher and MAI's adaptive clip.
- **llm-training ch-45b** (multi-turn agentic RL) — GiGPO and observation masking.
- **llm-training ch-44** (verifiable rewards) and **ch-44b** (multi-domain RL) — reward-side context.

---

## Sources

- [[algorithm-crosswalk]] — the library crosswalk: Table 1 (lab × stage), Table 2 (per algorithm), domain → algorithm pattern.
- [[recipe-crosswalk]] — stage context for each row.
- [[ppo]], [[grpo]], [[rloo]], [[rloo-vs-grpo]], [[reinforce-plus-plus]], [[dr-grpo]], [[dapo]], [[gspo]], [[vapo]], [[dpo]] — algorithm papers and their measured fixes.
- [[gigpo-verl-agent]] — step-level credit and the PPO/RLOO/GRPO agent comparison.
- [[minimax-m1]] — CISPO and its stated rejections of GRPO and DAPO.
- [[scalerl]] — loss-type comparison by fitted asymptote.
- [[mai-thinking-1]] — MAI's GRPO objective, adaptive upper clip, outer clip, replay mechanisms.
- [[mai-rl-infrastructure]] — group completion as the source of agentic staleness.
- [[olmo-3-rl-pipeline]], [[olmo-3-recipe]], [[open-instruct-rlvr-pipeline]] — OlmoRL changes; open-instruct@f020e9a loss code.
- [[deepseek-v3.1]] — V3.2 stabilizers and their stated reasons.
- [[glm-4-5]], [[qwen-3]], [[qwen-3-recipe]], [[kimi-k2]], [[magistral]], [[nemotron-ultra-recipe]], [[tulu-3]], [[rlvr-tulu3]] — lab rows.
- [[ifbench]] — GRPO vs DPO on IF.
- [[search-r1]], [[zerosearch]], [[deepresearcher]], [[asearcher]], [[websailor-tongyi-deepresearch]] — search-agent algorithm rows.
- [[deepswe]], [[swe-rl]], [[rstar2-agent]] — SWE and tool-integrated math rows.
