<!-- chapter: ch-24
     track: synthetic
     kind: content
     title: Modality — Reasoning Traces (CoT → Long-CoT → Step-Level)
     deps: [ch-23]
     sources: [[openmathinstruct]], [[openmathinstruct-2]], [[mathscale]], [[metamath]], [[mammoth]], [[mammoth-2]], [[rstar]], [[rstar-math]], [[s1]], [[limo]], [[step-dpo]], [[omegaprm]], [[quiet-star]], [[numina-math]]
     figures: figures/rstar-mcts.html
-->

# Chapter 24 — Modality: Reasoning Traces (CoT → Long-CoT → Step-Level)

> **Core insight.** Reasoning-trace synthesis is the one SFT modality where the *verifier* matters more than the *teacher*. Every advance in the field — from OpenMathInstruct's 1.8M Mixtral traces to rStar-Math's 747K self-evolved traces — is a story about moving the filter deeper into the trace: final-answer match → step-level code execution → Monte-Carlo rollout value → pairwise step preferences. The teacher model sets a ceiling on what can be generated; the verifier decides what survives. Get the verifier wrong and more data makes the student *worse*.
>
> **Guideline.** When building a reasoning-trace corpus, pick your verifier first, not your teacher. If you have gold answers and a symbolic checker ([[openmathinstruct-2]], [[numina-math]]): go wide — scale solutions per problem before scaling problem count. If you have only gold answers and want step supervision: run OmegaPRM-style Monte-Carlo labeling or rStar-Math-style MCTS. If you have neither gold answers nor a symbolic checker but have a strong base model: curate small ([[s1]], [[limo]]) — 1K reflective traces will beat 1M unchecked ones.

---

## 1. Map of the design space

Six knobs control a reasoning-trace corpus. Every paper in this chapter is a specific setting of them.

| Knob | Typical range | Papers anchoring each end |
|---|---|---|
| **Teacher strength** | Mixtral-8x7B → Llama-3.1-405B / GPT-4o | weak: [[openmathinstruct]]; strong: [[openmathinstruct-2]], [[numina-math]] |
| **Problem pool** | 15K seeds → 860K curated → 14M augmented | small: [[metamath]] (15K seeds × 25); large: [[numina-math]] (860K); synthesized: [[mathscale]] |
| **Solutions per problem** | 1 → 120 | rejection-sampling at K=32-64 is modal ([[openmathinstruct]]) |
| **Trace style** | short-CoT → hybrid CoT+PoT → long-reflective-CoT | short: OpenMathInstruct-2; hybrid: [[mammoth]]; long: [[s1]], [[limo]] |
| **Verifier depth** | final-answer match → step execution → MCTS Q-value → pairwise | final: most; step-exec: [[rstar-math]]; MC: [[omegaprm]]; pair: [[step-dpo]] |
| **Labeling budget** | 0 human → O(hours) hand-curation | 0: [[omegaprm]]; hand: [[limo]] |

Chapters 21-23 argued the general case for synthetic data. This chapter is the one modality-specific drill-down the course does in depth, because reasoning is where the synthetic-data field is actually *moving* in 2024-2025 — and because Track 4's RL chapters (ch-48 onwards) consume reasoning-trace datasets as their starting point.

---

## 2. The wide-short-CoT lineage: OpenMathInstruct 1 → 2

**OpenMathInstruct-1** ([[openmathinstruct]], Toshniwal et al. 2024) is the modern template for open-teacher math SFT: take the GSM8K (7.5K) + MATH (7.5K) train sets, ~15K problems total, and sample **K=32-64 solutions per problem from Mixtral-8x7B-Instruct**. Each solution is *tool-integrated*: CoT text interleaved with `<llm-code>...</llm-code>` Python blocks whose outputs get spliced back as `<llm-code-output>`. Filter by SymPy-canonical equivalence (MATH) or numeric string match (GSM8K). The survivors — 1.8M (problem, solution) pairs, averaging **~120 solutions per GSM8K problem and ~100 per MATH problem** — are Apache-2.0.

The arithmetic is what forces the design. Mixtral at K=64 solutions/problem, ~500-token average trace length, 15K problems → ~480M teacher tokens generated. On a DGX cluster, **~500K GPU-hours**. The payoff: OpenMath-Mistral-7B hits 80.2 GSM8K / 44.5 MATH with zero closed-teacher exposure.

**OpenMathInstruct-2** ([[openmathinstruct-2]], Toshniwal et al. 2024) is the same pipeline with three swaps:
1. **Teacher**: Mixtral → **Llama-3.1-405B-Instruct** (served BF16 via vLLM).
2. **Problem pool augmentation**: the 15K seed grows to ~600K via two teacher-prompted operations —
   - *Paraphrase*: teacher rewrites each seed problem in different wording.
   - *Novel question*: teacher invents new problems conditioned on the topic tag ("Algebra, Level 5") extracted from MATH.
3. **Trace style**: drop TIR in favor of pure text-CoT. Authors found that at 405B scale, **text-CoT outperforms TIR** — the teacher's arithmetic is accurate enough that the executor adds more noise than signal.

Output: **14M (problem, solution) pairs** at ~650K H100-hours. OpenMath2-Llama3.1-8B reaches **91.7 GSM8K / 67.8 MATH**. The operational lesson is in the *ablation*: **Llama-3.1-405B at 1M samples beats Mixtral at 10M samples**. If you can upgrade the teacher, do that before you scale the corpus.

One caveat that matters for Track-4 RL consumers: OpenMathInstruct-2 is **non-reflective short-CoT**. Students trained on it do not acquire backtracking or self-verification. For o1-style behaviour you need the long-CoT lineage (§4) or distillation from DeepSeek-R1 / o1 traces.

---

## 3. Question-side diversity: MetaMath + MathScale + MAmmoTH

Before scale-the-teacher, the field iterated on *what to do with a small seed pool*.

**MetaMath** ([[metamath]], Yu et al. 2023) introduced four question-rewrite operators applied to each seed (Q, A) pair:
1. **AnsAug** — keep Q, sample K CoT answers, keep correct ones.
2. **Rephrasing** — teacher rewrites Q in different words, re-solve.
3. **Self-Verification (SV)** — "Given Q and candidate answer A', is A' correct? If not, fix."
4. **FOBAR (Forward-Backward Reasoning)** — the clever one. Take "Jane has 3 apples and buys 5 more. How many?" → 8. Rewrite as "Jane has 3 apples and buys x more. She now has 8. What is x?" → 5. The model learns to *run a chain in reverse*.

The **FOBAR self-verify filter** is specific: an augmented pair is kept only if the teacher's solution to the inverse problem reconstructs the known masked number. Empirically (MetaMath §4), AnsAug alone gives ~+4 GSM8K, Rephrasing +3, SV +2, FOBAR +3 — **additive**, because each operator inoculates against a different overfitting mode. MetaMath-70B ships at 82.3 GSM8K / 26.6 MATH on 395K examples (vs OpenMathInstruct-2's 91.7 / 67.8 on 14M — the teacher gap is decisive).

**MathScale** ([[mathscale]], Tang et al. 2024) generalizes question augmentation: instead of rewriting seeds, mine a **concept graph**. GPT-3.5 extracts topics (~2K) and knowledge points (~5K) from each seed problem; edges weight co-occurrence. Sampling **rare (topic, concept) edges** pushes the teacher to invent problems outside the seed distribution. 2M MathScaleQA. Weakness: no ground truth — the teacher is both author and grader, so ~5% of "gold" answers are wrong.

**MAmmoTH** ([[mammoth]], Yue et al. 2023) is the original CoT+PoT hybrid: 260K traces, ratio PoT:CoT ≈ 57:43. The PoT template is mechanical:

```python
def solution():
    # Mary has 3 apples and buys 5 more.
    total = 3 + 5
    return total
print(solution())
```

Execute; accept if stdout matches gold. **Complementary error modes**: CoT handles conceptual reasoning, PoT handles arithmetic precision. Ablation: CoT-only drops MATH by 8 points; PoT-only drops AQuA (logic) by 15. [[mammoth-2]] scales this to 10M via web-mining (Recall → Extract → Refine with Mixtral), trading verifier strength (LLM-judge instead of gold-match) for volume.

---

## 4. Long-CoT, small-N: s1 and LIMO

The 2025 inflection. The thesis: **a strong base model already contains reasoning capability; SFT's job is to activate it, not install it.** Corollary: you do not need millions of traces. A few hundred reflective ones suffice.

**s1** ([[s1]], Muennighoff et al. 2025): start from a **59K question candidate pool**. Apply three filters in sequence — difficulty (problems strong baselines still miss), diversity (topic spread), quality (manual trace-style check). Result: **s1K = 1000 question-trace pairs**, traces generated by Gemini. Base model: Qwen2.5-32B-Instruct. Training: **26 minutes on 16 H100** with FSDP. At inference, apply **budget forcing**: whenever the model tries to emit the end-of-thinking token, suppress it and append `"Wait"`. The model keeps thinking. AIME24 goes from 50% → 57% on the *same* checkpoint purely by extending the forced budget.

Reported numbers: **s1-32B: 56.7 AIME24, 93.0 MATH500, 59.6 GPQA-Diamond**. These are competitive with o1-preview from a 1000-example SFT.

**LIMO** ([[limo]], Ye et al. 2025): the hand-curated twin. **817 long-CoT samples** covering competition math, MATH, GSM8K-hard, physics olympiad. Selection criteria: final-answer correctness, presence of *self-verification segments*, *branching / backtracking markers*, fine-grained step granularity. Hand-filter removes traces that hit the right answer through subtly broken logic. Reported: **63.3 AIME24, 95.6 MATH500** — higher than s1 on both. Formalized as the "Less-Is-More Reasoning Hypothesis": strong pretrained base + high-quality demonstrations ⇒ reasoning emerges.

The 1K-vs-100K eval table (from the two papers' ablations combined):

| Recipe | Dataset size | AIME24 | MATH500 |
|---|---|---|---|
| Qwen2.5-32B-Instruct base | 0 | ~17 | ~84 |
| OpenMathInstruct-2 SFT | 14M | ~40 | ~90 |
| Random 1K from 59K | 1K | ~24 | ~86 |
| s1K curated | 1K | 56.7 | 93.0 |
| LIMO hand-curated | 817 | 63.3 | 95.6 |

Caveats stack: curator subjectivity (LIMO), base-model dependence (both papers note weak bases do not activate from 1K), benchmark-contamination risk (competition problems overlap). And the gains *require* long-reflective traces — substitute 817 short-CoT traces and the result is closer to the random-1K baseline. Trace **style** is doing the work, not just count. See [figures/rstar-mcts.html](figures/rstar-mcts.html) Panel 2 for the interactive scan across dataset sizes.

---

## 5. Tree search as synthesis: rStar and rStar-Math

The MCTS lineage. **rStar** ([[rstar]], Qi et al. 2024) is an inference-time-only procedure: no fine-tuning. A generator small LLM runs MCTS with a five-action space:

- **A1**: propose a one-step CoT.
- **A2**: decompose into subquestions.
- **A3**: directly answer a subquestion, then verify.
- **A4**: rephrase the question to simplify.
- **A5**: propose a new intermediate subquestion.

UCB selection over N=32 rollouts produces candidate trajectories. Each is then checked against a **mutual-consistency verifier**: a *separately-prompted* discriminator (same base model, different prompt) is given the trajectory's first half and asked to complete it. Accept iff `answer(generator) == answer(discriminator_completion(mask_half))`. LLaMA2-7B GSM8K: **12.5 → 63.1** — +50.6 absolute with **no fine-tuning, no stronger teacher**.

**rStar-Math** ([[rstar-math]], Guan et al. 2025) turns this into a training pipeline. Key pseudocode:

```
# rStar-Math: MCTS node = (natural-language thought, Python code block)
def mcts_step(node):
    # Selection: descend via UCB until a leaf
    while node.children:
        node = argmax(child, key=lambda c:
            c.Q + c_puct * c.P * sqrt(node.N) / (1 + c.N))
    # Expansion: sample K next-step (thought, code) candidates from policy
    for _ in range(K):
        (thought, code) = policy.sample(prefix=node.trace)
        try:
            exec(code)                       # step-level code execution
        except Exception:
            continue                         # prune on runtime error
        child = Node(thought, code, parent=node)
        node.children.append(child)
    # Rollout: simulate to terminal; reward = 1 iff final boxed answer == gold
    leaf = rollout(node.children[0])
    r = int(extract_boxed(leaf) == gold_answer)
    # Backprop: update Q-values up the path
    backprop(leaf, r)
```

Four-round self-evolution:
1. **Round 0**: Qwen2.5-Math-7B-Instruct as bootstrap generator; run MCTS on 747K problems from [[numina-math]] + olympiad + AIME archives.
2. **PPM training**: within each problem, MCTS siblings with Q-gap > δ form *step-preference pairs* (step_high, step_low). Process Preference Model trained with pairwise Bradley-Terry. Not a scalar PRM — authors argue pairwise avoids the Goodhart pathology [[let-verify]]-style scalar PRMs exhibit.
3. **Policy retrain**: top-K trajectories by PPM score fine-tune the next generator.
4. **Rounds 1-3**: repeat. MATH climbs **58 → 78 → 85 → 88 → 90** across rounds.

Endpoint: Qwen2.5-Math-7B-rStarMath hits **90.0 MATH, 53.3 AIME24, 58.5 Olympiad** — matches o1-preview on several benchmarks from a 7B base. Still gold-answer-dependent at the trajectory level; the novelty is step-level executability as the inner-loop verifier.

---

## 6. Step-level supervision: OmegaPRM automated labels + Step-DPO

Both methods exist because final-answer SFT leaks false-positive "right answer, wrong reasoning" traces (~7% rate on OpenMathInstruct-2 audits, §2). Step supervision closes this.

**OmegaPRM** ([[omegaprm]], Luo et al. 2024) automates the step labels. Formal definition:

```
MC(s_t) = (1/K) · Σ_{i=1}^{K} 𝟙[rollout(policy | s_1..s_t) yields gold answer]
```

Each intermediate step s_t gets a soft label = fraction of K completions from that prefix reaching gold. Naive cost: O(L·K) rollouts per trajectory. The OmegaPRM innovation is **divide-and-conquer MCTS**: binary-search down the trajectory for the first step where MC drops sharply, O(K · log L) rollouts per trajectory. With K=16 and L=10, this is a 4× saving. 1.5M step labels over ~80K problems → PRM regressed via MSE on soft MC targets. Used with weighted best-of-N (PRM × policy log-prob) as selector: Gemini Pro MATH **51 → 69.4**.

**Step-DPO** ([[step-dpo]], Lai et al. 2024) shifts from scalar step-value to pairwise step preference. Pipeline:
1. Sample K CoTs from policy. Keep those with **wrong final answer**.
2. Strong teacher (GPT-4 / Qwen2-72B) identifies the index of the **first erroneous step**.
3. Teacher generates a **corrected step** given (problem, prefix-up-to-error). Verify by continuing and checking final answer; keep if correct.
4. Form triplet `(prefix_i, step_correct, step_incorrect)` sharing the same prefix.

Update rule — the Step-DPO loss, identical form to vanilla DPO but on single-step completions:

```
L_StepDPO = -log σ( β · log[π_θ(y_w | x) / π_ref(y_w | x)]
                   - β · log[π_θ(y_l | x) / π_ref(y_l | x)] )
```

where x is the multi-step prefix and y_w, y_l are single reasoning steps (30-120 tokens each). **10K pairs** beat full-trajectory DPO on 100K pairs (MATH 58.6 vs 54.3 on Qwen2-7B). The gradient-dilution argument explains why: in trajectory DPO most tokens are identical between chosen and rejected, the KL denominator washes out the signal; Step-DPO concentrates the mass on the actual disagreement.

Both methods depend on gold answers and a stronger teacher. **Zero-supervision step labeling is open** — the field has not solved it.

---

## 7. The quiet outlier: Quiet-STaR

One footnote for completeness. [[quiet-star]] (Zelikman et al. 2024) trains reasoning during **continued pretraining** rather than post-training SFT: the model learns to generate latent thought spans at many token positions during ordinary language modeling, with learnable start/end thought tokens. GSM8K zero-shot: 5.9 → 10.9. This is not a reasoning-trace corpus in the sense this chapter has been building — it is a *mechanism* for embedding reasoning into the pretraining distribution. Track-4 will revisit it as the minority alternative to post-training-only reasoning recipes.

---

## 8. Practical guidance

Picking a recipe for your own reasoning-trace build:

- **Gold answers + symbolic checker available** (math, some code tasks): go [[openmathinstruct-2]] — strong teacher, K=32, SymPy-filter. Scale solutions/problem before scaling problem count.
- **Gold answers, want step labels**: [[omegaprm]] divide-and-conquer MC if you have compute; [[step-dpo]] triplet pipeline if you have a GPT-4-class teacher.
- **No gold answers, strong base model**: [[s1]] / [[limo]]. 1K curated traces. Expect curator subjectivity; write down your filter rules before applying them.
- **No gold answers, weak base**: [[rstar]] mutual-consistency is your best shot. Do not expect frontier numbers.
- **Need reflective long-CoT**: distill from DeepSeek-R1 / o1 / Gemini-Thinking, or curate manually as LIMO did. Synthetic generation from non-reasoning teachers will not produce backtracking.

The gotcha every team re-discovers: **false-positive traces compound**. OpenMathInstruct-2's ~7% right-answer-wrong-reasoning rate is survivable at 8B, but students that learn the shortcuts stop being able to generalize. Invest verifier effort.

## Connections

- **ch-22, ch-23** (synthetic data at scale, model collapse / verification) — this chapter is the reasoning-specific instantiation. The verifier-gate principle from ch-23 is why Step-DPO and OmegaPRM exist.
- **ch-25** (multi-turn conversation synthesis) — parallel modality: different verifier (format + role-adherence), same teacher-distillation backbone.
- **Track 4 (RL)**: ch-48 (RLVR), ch-52 (PRM-weighted reward) — all consume the corpora built in this chapter. rStar-Math's PPM is the bridge.

## Further reading

- [[openmathinstruct]], [[openmathinstruct-2]] — the wide-short-CoT template and its teacher-ablation lesson.
- [[metamath]] — FOBAR / SV / Rephrasing operators; reusable for non-math tasks with minor changes.
- [[mammoth]], [[numina-math]] — hybrid CoT+PoT; cn_k12 as an underappreciated problem source.
- [[rstar]], [[rstar-math]] — MCTS + PPM; the small-model-frontier story.
- [[s1]], [[limo]] — 1K-trace curation; budget-forcing inference hack.
- [[step-dpo]], [[omegaprm]] — step-level supervision; automated MC labels.
- [[quiet-star]] — the continued-pretraining alternative to post-training trace SFT.

## Companion visualization

**[figures/rstar-mcts.html](figures/rstar-mcts.html)** — two-panel interactive. **Panel 1** is an MCTS tree for rStar-Math: click **Expand** to sample the next reasoning step at the UCB-selected node (highlighted). Each node shows (thought, code-snippet, Q-value, visit count); code-error nodes prune automatically. Reset to re-seed. **Panel 2** is the s1/LIMO 1K-vs-100K curve: slider over dataset size and trace style (short-CoT vs long-reflective-CoT); AIME24 and MATH500 numbers plotted against the 14M OpenMathInstruct-2 baseline. The two panels together encode the chapter's thesis: trace *quality* at 1K can match trace *quantity* at 14M when the verifier and trace style are right.
