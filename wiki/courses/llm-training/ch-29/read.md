<!-- chapter: ch-29
     track: synthetic
     kind: lab
     title: Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification
     deps: [ch-28]
     sources: [[self-instruct]], [[self-instruct-loop]], [[evol-instruct]], [[cherry-llm]], [[cherry-llm-recipe]], [[ifd]], [[ifd-filter]], [[superfiltering]], [[minhash-lsh]], [[deduplicating-training-data]], [[apigen]], [[lima]], [[lima-baseline]], [[deita]], [[prismatic-synthesis]], [[tulu-3-sft-mix]], [[tulu-3-sft-and-eval]], [[ifeval]], [[ifbench]], [[mmlu-pro]], [[length-controlled-alpacaeval]], [[signal-and-noise-eval]], [[judge-llm-bias]], [[karpathy-training-neural-net-recipe]]
     figures: figures/filter-cascade.html
     revised: 2026-09 (generality revision)
-->

# Chapter 29 — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification

> **Core insight.** A synthetic instruction set is useful for a general-purpose model only if it improves held-out capabilities that the filters were not tuned on, measured against the base model and against a general-purpose mixture of the same size. The filters in this lab change more than quality: IFD selection on Alpaca kept story and list generation and removed sentence rewriting and editing, and the 5% model lost MMLU (41.73 → 36.51) while its judge-based scores rose ([[cherry-llm]] Tables 1, 4). The verifier stage has measured value where failures can be checked: adding back samples that failed APIGen's execution check lowered BFCL by 5.94 (xLAM-7B) and 12.17 (xLAM-1B) points, and adding back semantic-check failures lowered it by 4.06 and 9.59 ([[apigen]] Fig. 5). The lab therefore records survival counts, category shares, a coverage score, and contamination for every stage, and decides with 3 seeds on a programmatic multi-capability suite.
>
> **Guideline.** When a synthetic pool must reach a target size N after filtering, compute the raw count as N divided by the product of stage survival rates before generating, because a 15% selection stage alone multiplies the required raw count by 6.7. Deduplicate before IFD selection, because a keep-share applied to a pool with duplicates spends slots on copies. When a filter changes a category's share by more than 5 points, keep the category report next to the benchmark table, because category removal predicts per-capability loss ([[tulu-3-sft-and-eval]] Table 10: GSM8K 76.2 → 64.1 without math data). When a comparison uses an LLM judge, report the length-controlled win rate, because verbosity prompts alone moved one model's raw AlpacaEval win rate from 22.9% to 64.3% ([[length-controlled-alpacaeval]] §4.1). Call an arm better only when its 3-seed mean difference exceeds both the seed range and the instance-level confidence interval.

## Corrections to the version you studied

1. "5000 raw → N_format → N_IFD → N_dedup → N_verified" with "~10K raw so the cascade has headroom to prune to a ~5K target" and the example `{"raw": 10000, "format": 7800, "ifd": 1170, "dedup": 1050, "verified": 990}` → the example ends at 990 rows, not 5K. With the same stage rates (0.78, 0.15, 0.897, 0.943; product 0.0990), a 5,000-row pool needs about 50,500 raw rows (derived; §1).
2. "Log `pre_clip_grad_norm` and step-1 loss; outside `ln(V) ± 20%` = misconfigured" and acceptance gate 5 → ln(V) is the expected loss of a randomly initialized softmax classifier ("verify loss @ init … -log(1/n_classes)", [[karpathy-training-neural-net-recipe]]). A pretrained base starts SFT at its own response-token loss, which the lab measures. ln V is the loss of a uniform prediction over V tokens (ln 128,000 = 11.76); a pretrained model predicts text better than uniform, so a correct run starts below ln V and fails that gate. The corrected check compares step-1 loss with the base model's measured loss on the same batch (§6). The "grad norm < 10" threshold has no source and is removed.
3. "[[apigen]] ablation: removing format –18% BFCL-V1, exec –11%, judge –6%" → APIGen reports an add-back ablation, not a removal ablation, and has no format arm: adding semantic failures changed BFCL by −4.06 (xLAM-7B) and −9.59 (xLAM-1B), adding execution failures by −5.94 and −12.17 ([[apigen]] Fig. 5). Generator pass rates range from 34.42% to 84.15% (Table 1).
4. "[[apigen]] used DeepSeek-Coder-V2 + GPT-4" and "sandboxed execution (tool/code, 5s timeout) … LLM-judge" as APIGen's three kinds → the generators are DeepSeek-V2-Chat, DeepSeek-Coder-33B-Inst, Mixtral-8x22B-Inst, and Mixtral-8x7B-Inst; the semantic-checker model and the execution timeout are not reported; APIGen's stages are format, execution, and semantic checks, and exact-match checking of reasoning answers is not part of APIGen ([[apigen]] §3.2, §4.2).
5. "[[ifd]]: `IFD(q, a) = PPL(a|q) / PPL(a)`" attributed to Cherry LLM → Cherry LLM defines IFD as the ratio of mean answer losses s(A|Q)/s(A) (Eq. 5), and its script computes that ratio; the perplexity ratio is Superfiltering's Eq. 2. The two orderings can differ (§3.3) ([[cherry-llm]] §2.2; [[ifd-filter]]).
6. "Warm the scoring model 1 epoch on ~1K random samples per [[cherry-llm]]" → 1,000 samples chosen as 10 per cluster from 100 K-means clusters, trained 1 epoch; random choice was an ablation ([[cherry-llm]] §2.1, Table 2).
7. "Expected survival: ~10–15% of post-format pool per [[cherry-llm]] ('top 10% beats full set')" → on LLaMA-7B, 5% of Alpaca beats official Alpaca, while 10% of WizardLM is below the reimplemented full-data WizardLM on leaderboard average and AlpacaEval; the authors call top 10% "a safe and reasonable choice" ([[cherry-llm]] Table 1, App. G.2).
8. `tpl = INPUT_FIRST if kind == "classification" else OUTPUT_FIRST` → Self-Instruct uses output-first generation for classification tasks, because input-first produced inputs biased toward one label, and input-first for the others ([[self-instruct-loop]], §2.2).
9. "Per [[self-instruct]] §Filtering: drop empty outputs, `input == output`, length < 3 or > 2K tokens, 'image'/'graph'/'file'" and "Self-Instruct reports ~67%" → Self-Instruct adds an instruction to the pool only if its ROUGE-L with every existing instruction is below 0.7, excludes keywords such as image, picture, graph, removes identical instances and same-input-different-output instances, and removes instructions that are too long or too short without numeric thresholds in the text; no survival rate is reported ([[self-instruct-loop]], §2.2).
10. The `evol_elimination` rules "same-or-similar / refusal ('sorry', 'i cannot', 'as an ai') / punct-only / verbatim" → WizardLM marks an evolution failed when ChatGPT judges no information gain, when the response contains "sorry" and is shorter than 80 words, when the response contains only punctuation and stop words, or when the instruction copies words from the evolving prompt; failed instructions return to the pool ([[evol-instruct]] §3.2).
11. `"breadth": "Mutate to a new instruction in a rarer domain"` → the in-breadth prompt asks for a new prompt in the same domain that is "even more rare", with similar length and difficulty ([[evol-instruct]] Example 3.3).
12. "Mix 70/30 Self-Instruct/Evol-Instruct … the complexity tail [[evol-instruct]] Figure 1 documents as the win condition" → Figure 1 is an evolution tree grown from "1+1=?"; WizardLM samples 70k examples with equal probability from the merged seed set and all rounds; no source tests a 70/30 mix ([[evol-instruct]] §3.3, §4.2).
13. "Attested pretraining setup is `b=20 × r=450` for ≈0.8 Jaccard; at lab scale `datasketch` default at `threshold=0.8` suffices" → Lee et al. use 9,000 hashes in 450 buckets of 20, candidate probability 1 − (1 − s^20)^450, then keep only pairs with Jaccard > 0.8 and edit similarity > 0.8, clustered by connected components; a library default is not a source value ([[deduplicating-training-data]] §4.2, App. A).
14. "`lima-matched` — N random from raw, LIMA-recipe proxy filter" → LIMA selected 750 community Q&A pairs with source-specific rules and manual choices and wrote 250 examples by hand; no algorithmic LIMA filter exists, so the arm is undefined. It is replaced by a matched-size random arm and a matched-size Tülu 3 SFT mixture arm ([[lima-baseline]], §1–§2).
15. "`epochs=2` … at ~1K–5K teacher-generated, 2 epochs matches [[deita]]" → DEITA trained its 6K/10K sets for 6 epochs; Tülu 3 8B SFT used 2 epochs; LIMA used 15 epochs with checkpoint selection between epochs 5 and 10 ([[deita]] App. A; [[tulu-3-sft-and-eval]] Table 11; [[lima-baseline]] §3).
16. "small enough for a ~30-minute run on 8 GPUs, large enough for benchmark deltas to clear noise" → no source supports either claim. Seed-to-seed SFT averages spread 2.6 points at 70B in Tülu 3 (Table 14), and a 164-problem benchmark at 25% accuracy has a 95% interval of ±6.6 points (§7).
17. "If your judge rejects <3%, your pool is already clean", "Expected survival: 70–85%", "80–95% of post-IFD pool", and ">30% kill = collapsed generator" → no source reports these rates; the lab measures them.
18. Acceptance gate 2 "histogram centers below 1 … Centered above 1 = warm-up skipped ([[ifd]] §Practical guidance)" → Cherry LLM makes no histogram claim; without a pre-experienced model its results are the lowest of the tested counts but still beat Alpaca at 10%, and its LLaMA2 runs use the base model as scorer ([[cherry-llm]] §4.3.1, §4.4).
19. Connections "ch-19 / ch-20 — [[self-instruct]] and [[evol-instruct]]", "ch-23 / ch-25 / ch-27 — IFD, MinHash, APIGen", "ch-30 takes the `full` checkpoint forward", and "Track 4 (RL)" → both generators are in ch-19; selection is ch-22, deduplication ch-12, tool-call verification ch-26; the next chapter is ch-29a; ch-30 is SFT design choices; policy-gradient and RL chapters sit in the preference and RL phases (from ch-37), with verifiable rewards in ch-44 (outline).
20. Figure label "MinHash Jaccard threshold (0=loose, 1=strict)" → a higher Jaccard threshold marks fewer pairs as duplicates and removes fewer rows, so 1 is the most permissive setting. The figure is rewritten.

## Why this chapter matters for a general-purpose model

ch-18 described the pattern generate → filter → deduplicate → verify → select → mix. This lab builds one instance of it for single-turn instruction SFT and asks whether the result helps a model on capabilities it was not built for. The data enters the pipeline at SFT, after pretraining and mid-training and before preference optimization and RL.

Three measurement problems make the question hard. First, filters change the task distribution, and a model trained on a narrower distribution can lose ability on the removed tasks while gaining on judge-based chat metrics ([[cherry-llm]] App. C, Table 1). Second, a teacher model can regenerate benchmark items, so a gain can come from contamination ([[tulu-3-sft-and-eval]] §3.2 removed 11.3% of NuminaMath-TIR for overlap with MATH). Third, single SFT runs vary with the seed, so a one-seed delta can be noise ([[tulu-3-sft-and-eval]] Table 14). The lab adds one control for each problem: a per-stage category and coverage report, a contamination check against every evaluation set, and 3 seeds with confidence intervals.

This chapter was studied in its earlier form. The generation and filter stages keep their order of sections. The held-out multi-capability evaluation in §7 is an optional re-run for a learner who already completed the earlier lab.

## §1 Lab goal, arms, and survival arithmetic

**Goal.** Produce an instruction pool of N rows, train matched-size models on it and on baselines, and report per-capability deltas against the base model and a general mixture. Deliverables:

1. `instructions.jsonl` — one row per sample with `origin` (seed, self_instruct, evol:<op>, round), `category`, `ifd`, `minhash_cluster`, `verifier_verdict` (checkable subsets only), and `contam_hits`.
2. `survival.json` — integer counts per stage, category counts per stage, and a coverage score per stage (§3.4).
3. `contamination.json` — overlapping rows per evaluation set (§5).
4. `eval_table.csv` — every arm × benchmark × seed (§7).
5. `synthetic-set-memo.md` — one page (§8).

**Survival rate.** The survival rate s_i of stage i is the fraction of its input rows that it keeps.

```
N_k = N_raw · s_1 · s_2 · … · s_k            N_raw = ceil( N_target / (s_1 · … · s_K) )
```

N_raw is the number of raw generations, N_k the count after stage k, and K the number of stages.

**Worked example (illustrative rates, to be replaced by measured ones).** Stage order is format → dedup → IFD → verifier (the reason for dedup before IFD is in §3.2).

| Stage | Rate | Full-budget path | Resource-constrained path |
|---|---|---|---|
| raw | — | 50,000 | 10,000 |
| format filter | 0.80 | 40,000 | 8,000 |
| near-duplicate removal | 0.90 | 36,000 | 7,200 |
| IFD selection (top 15% of the deduplicated pool, among IFD ≤ 1) | 0.15 | 5,400 | 1,080 |
| verifier (40% of rows checkable, 75% of those pass) | 1 − 0.40 × 0.25 = 0.90 | 4,860 | 972 |
| training size N (random subsample, same for every arm) | — | 4,800 | 960 |

The product of the rates is 0.80 × 0.90 × 0.15 × 0.90 = 0.0972, so N_raw = 5,000 / 0.0972 ≈ 51,440 for a 5,000-row target. The IFD stage contributes a factor of 1/0.15 = 6.7.

[figures/filter-cascade.html](figures/filter-cascade.html) computes these counts from editable stage rates, solves for the raw count that reaches a target size, shows how a category-dependent IFD keep rate changes category shares, and plots the MinHash-LSH candidate probability for chosen band settings.

**Paths.** Both paths use the same stages, arms, suite, and 3 seeds. Full-budget: base model `meta-llama/Llama-3.2-1B` or `Qwen/Qwen2.5-1.5B`. Resource-constrained: `Qwen/Qwen2.5-0.5B` or `HuggingFaceTB/SmolLM-360M`, smaller benchmark subsets (§7), and N = 960. Compute and API cost are not estimated here; record them in the memo.

**Arms (all at training size N, each with seeds 1, 2, 3).**

| Arm | Data | Question it answers |
|---|---|---|
| `base` | no SFT (evaluated with the same chat prompt) | reference for forgetting |
| `full` | format → dedup → IFD → verifier | headline pool |
| `no-ifd` | format → dedup → verifier, random N | value of IFD selection |
| `no-dedup` | format → IFD → verifier | value of near-duplicate removal |
| `no-verify` | format → dedup → IFD | value of the verifier on checkable rows |
| `random-format` | format filter only, random N | value of all later stages together |
| `general-mix` | random N rows of the Tülu 3 SFT mixture, decontaminated | whether the synthetic pool beats a general mixture |
| `mix-evol-heavy` (optional) | `full` pipeline with generation shifted toward evolution rounds | whether evolution depth trades against breadth on categories absent from the seeds |

## §2 Stage 1: Generation with bootstrap and evolution operators

### §2.1 Self-Instruct bootstrap

**Definition.** Self-Instruct generates instructions by prompting a language model with examples sampled from a growing task pool, then generates instances and filters them ([[self-instruct-loop]] §2.2).

**Problem.** The seed set contains 175 tasks, and repeated prompting with the same examples returns near-copies of them. The measurable target is the number of distinct instructions whose ROUGE-L against every earlier instruction is below 0.7.

**Mechanism.**
1. Sample 8 instructions: 6 from the 175 human-written seeds and 2 from model-generated tasks.
2. Prompt "Come up with a series of tasks:" with "Task 1:" … "Task 8:" and read "Task 9:".
3. Ask whether the task is a classification task with finite output labels.
4. Classification: generate the labels first, then an input for each label (output-first). Other tasks: generate the input, then the output (input-first).
5. Add the instruction to the pool only if its ROUGE-L with every pool instruction is below 0.7.

**Evidence.** GPT-3 davinci produced 52,445 instructions and 82,439 instances. In an author review of 200 samples, 92% of instructions were valid tasks and 58% of outputs were correct; 54% had all fields valid (Table 1, Table 2). Result (single study).

```python
# self_instruct.py — generation step (lab code; template text from Self-Instruct Table 5)
def build_instruction_prompt(seed_tasks, generated_tasks, rng):
    picks = rng.sample(seed_tasks, 6) + rng.sample(generated_tasks, min(2, len(generated_tasks)))
    rng.shuffle(picks)
    body = "\n".join(f"Task {i+1}: {t.instruction}" for i, t in enumerate(picks))
    return "Come up with a series of tasks:\n" + body + f"\nTask {len(picks)+1}:"

def generate_instance(client, instruction, is_classification):
    # Self-Instruct §2.2: output-first for classification (avoids label-biased inputs), input-first otherwise
    template = OUTPUT_FIRST if is_classification else INPUT_FIRST
    return parse_instance(client.complete(template.format(instruction=instruction)))

def admit_to_pool(instruction, pool_instructions, rouge_l):
    return all(rouge_l(instruction, p) < 0.7 for p in pool_instructions)   # §2.2 threshold
```

**Limits.** The ROUGE-L check compares every new instruction with every pool instruction, so its cost grows with the square of the pool size. The lab keeps it for the instruction pool and uses MinHash for the full sample text (§3.2).

### §2.2 Evol-Instruct evolution

**Definition.** Evol-Instruct asks an LLM to rewrite an instruction into a harder one (in-depth: add constraints, deepening, concretizing, increased reasoning steps, complicating input) or into a new, rarer instruction in the same domain (in-breadth) ([[evol-instruct]] §3.2).

**Mechanism.**
1. For each instruction in each round, sample one of the six prompts with equal probability (§4.2).
2. In-depth prompts restrict the rewrite to adding 10 to 20 words (Example 3.1).
3. Generate a response to the evolved instruction with the same LLM (§3.2).
4. Eliminate failed evolutions with four rules; failed instructions return to the pool for the next round (§3.2).
5. After M rounds, merge the seed set and all rounds and sample the training set with equal probability (§3.3).

```python
# evol_instruct.py — lab code; rules follow WizardLM §3.2
ADD_CONSTRAINTS = ("I want you act as a Prompt Rewriter. ... You SHOULD complicate the given prompt "
                   "using the following method:\nPlease add one more constraints/requirements into "
                   "#Given Prompt#\n ... #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#. "
                   "...\n#Given Prompt#:\n{instruction}\n#Rewritten Prompt#:\n")   # Example 3.1, abridged
BREADTH = ("I want you act as a Prompt Creator. ... This new prompt should belong to the same domain as the "
           "#Given Prompt# but be even more rare. The LENGTH and difficulty level of the #Created Prompt# "
           "should be similar to that of the #Given Prompt#. ...")                  # Example 3.3, abridged

def evolution_failed(original, evolved, response, judge_equal, stopwords):
    if judge_equal(original, evolved):                                     # (1) no information gain (ChatGPT-judged)
        return True
    if "sorry" in response.lower() and len(response.split()) < 80:         # (2)
        return True
    if all(t in stopwords or not t.isalnum() for t in response.split()):   # (3) punctuation and stop words only
        return True
    banned = ("#given prompt#", "#rewritten prompt#", "given prompt", "rewritten prompt", "created prompt")
    return any(b in evolved.lower() for b in banned)                        # (4) copies evolving-prompt words
```

**Evidence.** LLaMA 13B trained on 70k samples from 250k evolved instructions (52k Alpaca seeds, M = 4, gpt-3.5-turbo) averages 58.96 over nine benchmarks, against 54.60 for Vicuna-13b and 43.44 for a re-trained Alpaca-13b ([[evol-instruct]] Table 1). The seed matters: a ShareGPT seed raised the average to 61.87 but lowered GSM8k from 37.15 to 31.46, which the authors attribute to fewer math instructions (4.3% vs 11.8%) (Table 2, §4.5). Result (single study).

**Conditions and limits.** The authors raise difficulty in small steps because a set of extremely complex instructions "would harm the generalization performance" (§3.2, Interpretation by the authors; step size not ablated). No source in this chapter tests a ratio between Self-Instruct and Evol-Instruct samples. The lab records `origin` for every row so that the ratio in the final pool is known, and the optional `mix-evol-heavy` arm measures it.

**Implication.** The seed and operator mix set the category distribution before any filter runs. Record category shares at the raw stage (§3.4).

## §3 Stage 2: Filter cascade

### §3.1 Format filter

The format filter removes rows that cannot be training targets: empty fields, output equal to input, same input with different outputs, keywords that imply non-text input (Self-Instruct lists image, picture, graph), invalid JSON in tool-call rows, and outputs above the training context length after templating. The context bound is lab-specific; Cherry LLM used 512 tokens for Alpaca and 1024 for WizardLM ([[cherry-llm-recipe]]). Record the rejection reason per row so the memo can report reason counts per category.

### §3.2 Near-duplicate removal with MinHash LSH

**Definition.** The resemblance of two documents is the Jaccard ratio of their shingle sets, J(A, B) = |S(A) ∩ S(B)| / |S(A) ∪ S(B)|, where S(D) is the set of contiguous w-token shingles of D ([[minhash-lsh]] §2). A MinHash signature estimates J without comparing full texts; locality-sensitive hashing (LSH) groups signatures into bands so that only pairs that match on a whole band are compared.

**Problem.** Evolution rounds and repeated sampling produce rows that differ in a few words. Exact matching misses them, and all-pairs comparison of N rows costs O(N²).

**Mechanism.**
1. Shingle each row's instruction + output into word 5-grams (Lee et al. use space-tokenized 5-grams, [[deduplicating-training-data]] §4.2).
2. Compute a signature of k hash minima. For one random permutation, the probability that the minimum of the union lies in both sets equals J ([[minhash-lsh]] §3, Theorem 1).
3. Split the signature into b bands of r values. Two rows become candidates if any band matches exactly.
4. Confirm each candidate pair with a similarity check and cluster confirmed pairs by connected components; keep one row per cluster.

**Formula.** P(candidate | J = s) = 1 − (1 − s^r)^b, where s is the pair's Jaccard similarity, r the number of hash values per band, and b the number of bands ([[minhash-lsh]] Connections, MMDS §3.4.2; [[deduplicating-training-data]] §4.2 writes it with 20 hashes per bucket and 450 buckets).

**Worked example.** Lee et al.: r = 20, b = 450 (k = 9,000). At s = 0.8, s^20 = 0.0115 and P = 1 − (0.9885)^450 = 0.995; at s = 0.5, P = 0.0004. Lab setting with k = 128: r = 8, b = 16. At s = 0.8, s^8 = 0.168 and P = 1 − (0.832)^16 = 0.947; at s = 0.7, P = 0.613; at s = 0.5, P = 0.061. The lab setting admits more low-similarity candidates, so the confirmation step (estimated J ≥ 0.8) makes the final decision.

**Evidence.** On C4, NearDup (Jaccard > 0.8 and edit similarity > 0.8) marked 3.04% of training examples as near-duplicates, and 4.60% of validation examples had a near-duplicate in training (Table 2). A 1.5B model trained on original C4 emitted 1.926% memorized tokens against 0.189% after NearDup (Table 4). Downstream task effects were not evaluated (§2). Result (single study, pretraining scale).

```python
# near_dedup.py — lab code; banding written out so r and b are explicit
from datasketch import MinHash

def signature(text, k=128, w=5):
    toks = text.split()
    m = MinHash(num_perm=k)
    for i in range(max(1, len(toks) - w + 1)):
        m.update(" ".join(toks[i:i + w]).encode("utf-8"))
    return m

def near_dedup(rows, r=8, b=16, confirm_j=0.8):
    sigs = [signature(x.instruction + " " + x.output, k=r * b) for x in rows]
    buckets, parent = {}, list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, s in enumerate(sigs):
        hv = s.hashvalues
        for band in range(b):
            key = (band, tuple(hv[band * r:(band + 1) * r]))
            for j in buckets.get(key, []):
                if sigs[i].jaccard(sigs[j]) >= confirm_j:       # confirm candidate pair
                    parent[find(i)] = find(j)
            buckets.setdefault(key, []).append(i)
    kept = {}
    for i in range(len(rows)):
        kept.setdefault(find(i), i)                              # one row per connected component
    return [rows[i] for i in sorted(kept.values())], [find(i) for i in range(len(rows))]
```

**Why dedup runs before IFD.** IFD keeps a share of its input. If duplicates are still present, several copies of one high-IFD row can occupy selection slots and are then removed by dedup, so the final count and category shares no longer match the planned keep share. MinHash also costs no model forward passes, while IFD needs two per row. This ordering is a course recommendation (Interpretation); no source compares the two orders.

**Threshold direction.** Raising the confirmation threshold from 0.8 to 0.9 marks fewer pairs as duplicates and removes fewer rows. Lowering it removes more.

### §3.3 IFD selection

**Definition.** Instruction-Following Difficulty (IFD) compares how well a model predicts an answer with and without its instruction ([[cherry-llm]] §2.2).

```
s(A|Q) = −(1/N) Σ_i log P(w_i | Q, w_<i; θ)     s(A) = −(1/N) Σ_i log P(w_i | w_<i; θ)
IFD_loss(Q, A) = s(A|Q) / s(A)                   (Cherry LLM, Eq. 5)
IFD_ppl(Q, A)  = exp(s(A|Q)) / exp(s(A)) = exp(s(A|Q) − s(A))   (Superfiltering, Eq. 2)
```

Q is the instruction with optional input in the training template, A the answer with N tokens w_i, and θ the scoring model's weights. A higher IFD means the instruction helps less. Rows with IFD > 1 are removed, and the highest remaining values are kept ([[cherry-llm]] §2.2).

**Worked example.** Row X: s(A|Q) = 0.2, s(A) = 0.4. Row Y: s(A|Q) = 2.4, s(A) = 3.0. IFD_loss gives X = 0.50 and Y = 0.80, so Y ranks higher. IFD_ppl gives X = exp(−0.2) = 0.819 and Y = exp(−0.6) = 0.549, so X ranks higher. A top-1 selection keeps a different row under each definition. The lab uses Eq. 5, the definition of the paper whose selection results it cites, and writes the definition in the memo.

**Mechanism.**
1. Pre-experience: embed instructions with the base model (mean last-layer hidden state), run K-means with 100 clusters, take 10 rows per cluster, and train the base model 1 epoch on these 1,000 rows (§2.1).
2. Score every deduplicated row with Eq. 3–5 using the same template in both passes.
3. Drop IFD > 1, sort, keep the top share. The released script computes the kept count from the rows that pass the IFD ≤ 1 and length checks (`data_by_IFD.py` L117–L120, [[ifd-filter]]).

**Evidence.** On LLaMA-7B, the 5% Alpaca selection beats official Alpaca on leaderboard average (52.06 vs 50.21) and AlpacaEval (34.74 vs 26.46); the 10% WizardLM selection is lower than the reimplemented full-data model on both (51.59 vs 52.79; 61.44 vs 61.99) ([[cherry-llm]] Table 1). At equal 5% size, IFD selection averages 52.06 against 50.61 for random selection (App. B Table 5). With a weak scorer, overlap with the 7B scorer's selection is partial: GPT-2 and LLaMA2-7B select 41% of the same rows at a 10% share on Alpaca ([[ifd-filter]], Superfiltering Table 1). Result (single study each).

**Conditions and limits.** The score uses only losses on the given answer and does not check correctness (Eq. 3–5). IFD selection moves the category mix: the top 5% on Alpaca is led by "write story" (119) and "generate story" (98), and the bottom 5% by "rewrite sentence" (155) and "edit sentence" (89) (Table 4). The 5% model is worse than official Alpaca on Math and Coding sub-categories (App. C), and LLaMA-7B MMLU falls from 41.73 to 36.51 (Table 1).

**Implication.** IFD selection is a candidate cause of narrowing. The `no-ifd` arm and the category report in §3.4 are the two measurements that detect it.

```python
# ifd.py — lab code for Eq. 3–5 (mean token losses, same chat template in both passes)
import torch, torch.nn.functional as F

@torch.no_grad()
def mean_answer_loss(model, tok, prefix_ids, answer_ids):
    ids = torch.cat([prefix_ids, answer_ids]).unsqueeze(0).to(model.device)
    logits = model(ids).logits[0, len(prefix_ids) - 1:-1]
    return F.cross_entropy(logits.float(), answer_ids.to(model.device)).item()

def ifd_loss_ratio(model, tok, prompt_text, answer_text, bos_ids):
    # bos_ids: a non-empty 1-D tensor that starts both passes (BOS or the template's first token)
    q = tok(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    a = tok(answer_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    s_cond = mean_answer_loss(model, tok, torch.cat([bos_ids, q]), a)     # s(A|Q)
    s_direct = mean_answer_loss(model, tok, bos_ids, a)                   # s(A)
    return s_cond / s_direct
```

### §3.4 Per-stage category distribution and coverage

**Definition.** A category report lists, for each stage, the count and share of rows per task category. A coverage score summarizes how much of the raw pool's variety survives.

**Problem.** A filter can raise average quality while removing a task type. The measurable quantity is the change in each category's share from the raw stage to the final stage.

**Worked example (illustrative keep rates, direction as in [[cherry-llm]] Table 4).** Deduplicated pool of 36,000 rows; IFD keeps 5,400.

| Category | After dedup | IFD keep rate | After IFD | Share before → after |
|---|---|---|---|---|
| open generation | 14,400 | 0.24 | 3,456 | 40% → 64% |
| rewriting and editing | 7,200 | 0.03 | 216 | 20% → 4% |
| math reasoning | 9,000 | 0.12 | 1,080 | 25% → 20% |
| tool call | 5,400 | 0.12 | 648 | 15% → 12% |

The total keep rate is 5,400 / 36,000 = 0.15, as planned, yet the rewriting share falls by a factor of 5.

**Coverage scores.**
1. *Cluster coverage.* Embed the deduplicated pool, run K-means with K clusters (the lab uses K = 200), and report the fraction of clusters that keep at least 3 rows after each stage.
2. *Vendi score.* VS(D) = exp(−Σ_i λ_i log λ_i), where λ_i are the eigenvalues of K/|D| and K_ij is the cosine similarity of the normalized embeddings of rows i and j. VS equals |D| when all rows are orthogonal and 1 when all rows are identical ([[prismatic-synthesis]] Eq. 3, App. C, for the gradient version). Example with three rows, two identical and one orthogonal: eigenvalues of K/3 are 2/3 and 1/3, entropy = 0.637, VS = 1.89.

**Evidence on which diversity measure predicts generality.** With data quality held fixed, G-Vendi (gradient features from Qwen2.5-0.5B-Instruct) correlates with math OOD accuracy at Spearman ρ = 0.899, and embedding Vendi at 0.754 ([[prismatic-synthesis]] Table 1). Embedding Vendi ranked a persona-varied set above a seed-varied set whose OOD accuracy was 16.6 points higher (Table 2). The authors state that the correlation holds only under controlled quality (App. D). Result (single study). In the lab, embedding coverage is a monitoring number; the per-capability evaluation decides.

**Evidence that category removal predicts per-capability loss.** Removing math data from the Tülu 3 8B SFT mixture lowered GSM8K from 76.2 to 64.1 and MATH from 31.5 to 23.5; removing Persona data lowered IFEval from 72.8 to 53.6 ([[tulu-3-sft-and-eval]] Table 10). Result (single study).

## §4 Stage 3: Verifier gate for reasoning and tool subsets

**Definition.** A verifier is a program or model call that decides whether a row's answer is correct. The lab applies it only to rows whose correctness can be checked: math and logic items, code with tests, and tool calls with executable functions.

**Problem.** Format filters and IFD do not check correctness. Self-Instruct's kept data had 58% correct outputs in a 200-sample review ([[self-instruct-loop]] Table 2).

**Mechanism by subset.**
1. *Tool calls (APIGen stages, [[apigen]] §3.2).* (a) Format: JSON with required fields; every called function and argument exists in the given API list. (b) Execution: run Python functions in a separate subprocess; remove type errors, invalid parameters, runtime errors, timeouts, and missing arguments. (c) Semantic: a second LLM receives functions, query, calls, and execution results and returns `{"thought", "pass"}`.
2. *Code.* Execute the solution against tests in a sandbox. Tests written by the same teacher can share its errors; record the test source.
3. *Math and logic without references.* Sample 3 teacher solutions and keep the row if at least 2 final answers agree, the setting used for Nemotron-PrismMath (N = 3, τ = 2) ([[prismatic-synthesis]] §3.1). Agreement is not ground truth: a consistently wrong teacher passes.

**Evidence.** Pass rates through APIGen's three stages depend on the generator: 34.42% for DeepSeek-Coder-33B-Inst, 65.96% for Mixtral-8x22B-Inst, and 84.15% for DeepSeek-V2-Chat (Table 1). Adding back semantic failures changed BFCL by −4.06 (xLAM-7B) and −9.59 (xLAM-1B); adding back execution failures by −5.94 and −12.17 (Fig. 5). Human inspection of 600 released samples found 28 with minor issues (App. A.3). Evaluation is BFCL only (§5). Result (single study).

**Verifier false negatives.** A string check can reject a correct answer. IFEval's loose metric exists because a response ending in "P.S. **I do like the cake**" fails an exact match for "P.S. I do like the cake"; GPT-4's prompt-level accuracy is 76.89 strict and 79.30 loose ([[ifeval]] §2.2, Table 3). Label 20 rejected rows per verifier by hand and report the false-reject count.

```python
# verify_tool_call.py — lab code following APIGen §3.2 (timeout value is a lab choice; APIGen reports none)
import json, subprocess, sys

def format_ok(raw, api_specs):
    try:
        obj = json.loads(raw); calls = obj["answer"]
    except (ValueError, KeyError, TypeError):
        return False
    return all(c["name"] in api_specs and set(c["arguments"]) <= set(api_specs[c["name"]]["parameters"])
               for c in calls)

def execution_ok(call, module_path, timeout_s=10):
    runner = ("import importlib.util, json, sys\n"
              "spec = importlib.util.spec_from_file_location('m', sys.argv[1])\n"
              "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
              "c = json.load(sys.stdin)\n"
              "print(json.dumps(getattr(m, c['name'])(**c['arguments']), default=str))\n")
    try:
        out = subprocess.run([sys.executable, "-c", runner, module_path], input=json.dumps(call),
                             capture_output=True, text=True, timeout=timeout_s)   # separate subprocess
    except subprocess.TimeoutExpired:
        return False, None
    return out.returncode == 0, out.stdout

def semantic_ok(checker, query, calls, results):
    reply = checker.complete(SEMANTIC_PROMPT.format(query=query, calls=calls, results=results), temperature=0)
    return json.loads(reply).get("pass") == "yes"
```

## §5 Contamination check against every evaluation set

**Definition.** A training row is contaminated for a benchmark when it contains the benchmark item, or enough of it, that a model could reproduce the answer from memory.

**Problem.** A teacher model can reproduce benchmark items. Tülu 3 removed 11.3% of NuminaMath-TIR for overlap with MATH and 3.5% of Evol CodeAlpaca for overlap with HumanEval ([[tulu-3-sft-and-eval]] Table 8).

**Mechanism (Tülu 3 rule, §3.2).**
1. Compare prompts only (user turns), because completions are regenerated.
2. A test-item token matches a training row if they share an 8-gram containing that token.
3. A test item overlaps a training row if more than 50% of its tokens match.
4. Tülu 3 treats a dataset as contaminated if it overlaps more than 2% of an evaluation's items. The lab removes the overlapping rows and reports counts per benchmark, because it has one pool rather than many datasets.

**Worked example.** A 20-token test prompt shares one contiguous 12-token span with a training prompt. Every token in the span lies in a shared 8-gram, so 12/20 = 60% of tokens match and the item overlaps. If the shared span were 7 tokens, no 8-gram would be shared and the match would be 0%. For GSM8K with 1,319 test items, the 2% dataset rule is exceeded at 27 overlapping items.

**Limits.** Tülu 3 found embedding matching hard to separate from distributional similarity (§3.2). Paraphrased items can pass an n-gram check; Nemotron-PrismMath adds LLM paraphrase detection after 10-gram matching ([[prismatic-synthesis]] §3.1). Report the 8-gram result as the primary number and the paraphrase check as a second column.

**Implication.** Run the check against every benchmark in §7, including the knowledge probe, and also run it on the `general-mix` arm.

## §6 Stage 4: Training arms at matched size

**Matched size.** Every arm trains on exactly N rows. The effect of training-set size depends on the setting: LIMA found that doubling a Stack Exchange training set did not improve response quality ([[lima-baseline]] §5), while Cherry LLM reports different LLaMA2 leaderboard averages at 5%, 10%, and 15% shares ([[cherry-llm]] Table 3). A comparison with unequal sizes therefore cannot separate a size effect from a filter effect.

**Baselines.** `base` measures forgetting. `general-mix` uses a random subset of the 939,344-prompt Tülu 3 SFT mixture, which was built from skill-specific submixtures and decontaminated against the Tülu 3 suite ([[tulu-3-sft-and-eval]] §3.2, §4.1.2). Its Persona IF data covers IFEval's 25 constraint types (§3.1.2), so IFEval is a targeted column for that arm.

**Initial-loss check.** Before training, compute the base model's mean response-token loss L_base on the first training batch with the exact collator (template, masking, packing). Step-1 training loss must match L_base up to dropout and precision differences. A value near ln V (ln 128,000 = 11.76) indicates weights that were not loaded; a value far below L_base indicates that labels are visible in the inputs, for example a masking shift. Karpathy's "verify loss @ init" check is the from-scratch version of this test ([[karpathy-training-neural-net-recipe]]).

**Seeds.** Train each arm with 3 seeds. Tülu 3 8B SFT averages ranged from 59.8 to 60.1 over 5 seeds, and 70B averages from 70.0 to 72.6 over 3 seeds ([[tulu-3-sft-and-eval]] Table 14).

**Hyperparameters.** Fix one configuration for all arms (Recipe section). Select the learning rate on the `full` arm, seed 1, using development benchmarks only (§7).

## §7 Held-out multi-capability evaluation

**Definition.** A held-out benchmark is one that was not used to choose generation prompts, filter thresholds, hyperparameters, or checkpoints. Tülu 3 separates a development suite from an unseen suite whose scores were not examined during development ([[tulu-3-sft-and-eval]] §2.2).

**Suite.**

| Capability | Benchmark | Metric | Items (full / constrained path) | Role |
|---|---|---|---|---|
| instruction following | IFEval ([[ifeval]]) | prompt-level strict and loose accuracy | 541 / 541 | targeted by the add-constraints operator; also report IFBench ([[ifbench]]) when available |
| knowledge and reasoning | MMLU-Pro ([[mmlu-pro]]), stratified by 14 disciplines | accuracy, one prompt format | 1,400 / 700 | held out |
| math | GSM8K | exact match | 1,319 / 1,319 | targeted if math rows exist |
| code | HumanEval | pass@1, greedy | 164 / 164 | targeted if code rows exist |
| reasoning | BBH | exact match | all tasks / 10 tasks | held out |
| knowledge recall | PopQA or another closed-book QA probe | exact match | lab-chosen | held out; forgetting signal |
| chat (optional) | AlpacaEval ([[length-controlled-alpacaeval]]) | LC win rate, raw win rate, mean length | 805 / 805 | supplement only |

Choose filter thresholds and the learning rate on a development split: 200 held-back pool rows plus a separate slice of benchmarks not listed above. Read the final comparison on the suite.

**Instance-level noise.** For accuracy p on n items, the standard error is SE = √(p(1 − p)/n), and the 95% interval half-width is 1.96 · SE.

| Benchmark | n | p | SE | 95% half-width |
|---|---|---|---|---|
| GSM8K | 1,319 | 0.40 | 0.0135 | ±2.6 points |
| IFEval | 541 | 0.40 | 0.0211 | ±4.1 points |
| HumanEval | 164 | 0.25 | 0.0338 | ±6.6 points |

For two arms scored on independent samples with similar p, the half-width of the difference is 1.96 · √2 · SE, which is ±3.7 points for GSM8K at p = 0.40. Paired tests on the same items are narrower; ch-51 derives them.

**Checkpoint and seed noise.** The final 30 checkpoints of 1B models span 1.7% accuracy on ARC Challenge ([[signal-and-noise-eval]] §3.1). Evaluate the final checkpoint of each seed with the same decoding settings.

**Decision rule.** Report each arm as mean and range over 3 seeds per benchmark. Call arm A better than arm B on a benchmark only when the difference of means exceeds both the larger seed range and the instance-level half-width of the difference. Report per-discipline MMLU-Pro and per-task BBH next to the averages.

**Judges.** When an LLM judge is used, report the length-controlled win rate with the raw win rate and mean output length. Length control raised AlpacaEval's Spearman correlation with Chatbot Arena from 0.94 to 0.98 and narrowed the verbosity-prompt range of gpt4_1106_preview to 41.9%–51.6% ([[length-controlled-alpacaeval]] §4.1–4.2). Judges also show position and self-enhancement bias ([[judge-llm-bias]]); do not use the teacher model as the judge.

## §8 Memo and acceptance criteria

`synthetic-set-memo.md`, one page:
1. **Survival and category table.** Counts per stage and per category, coverage per stage, and the stage with the largest category-share change.
2. **Contamination.** Overlapping rows per benchmark for `full` and `general-mix`, and what was removed.
3. **Evaluation table.** Every arm against `base` and `general-mix` per benchmark: mean, range over seeds, instance-level half-width; LC win rate if a judge was used.
4. **Prediction log.** For each ablation, the predicted sign written before training and the observed result.
5. **One failure mode** with a row id, the stage that should have caught it, and the next measurement.

Acceptance gates:
1. `survival.json` counts equal the row counts of the stage files at every stage.
2. The category report exists for every stage, and the memo names any category whose share changed by more than 5 points.
3. `contamination.json` lists every benchmark in the suite, including zero counts.
4. Every checkable row in `full` has a `verifier_verdict`; 20 rejected rows per verifier are hand-labeled.
5. Step-1 loss of every run matches L_base on the same batch (§6).
6. Every arm has 3 seeds on the full suite, and no claim in the memo violates the §7 decision rule.

## Negative samples and negative feedback

**Which sense applies.** This lab produces negatives in sense 1 of the course standard (negative marginal value): rows rejected by the format filter, dedup, IFD, the verifier, and the contamination check are discarded. It does not use negatives as content, conditioning, or gradient. Using rejected rows in those roles is taught in ch-31a and ch-43a.

**Where negatives come from and how they are labeled.** Heuristics (format), MinHash similarity (duplicates, which are redundant rather than wrong), model losses (IFD > 1 or low IFD), execution and an LLM checker (tool calls), answer agreement (math), and n-gram overlap (contamination). Reported verifier error rates: APIGen's released data had 28 of 600 samples with minor issues after all three stages ([[apigen]] App. A.3); IFEval's strict check has false negatives that its loose check recovers (76.89 vs 79.30 prompt-level for GPT-4, [[ifeval]] Table 3).

**Mechanism: why a wrong row hurts when kept.** SFT minimizes −log p_y for the target token y. For softmax logits z, ∂ log p_y / ∂ z_j = 1[j = y] − p_j, so a gradient step raises z_y and lowers every other logit in proportion to its probability. Worked example with step size 1 and p = (0.6, 0.3, 0.1), where token 3 is a wrong target: Δz = (−0.6, −0.3, +0.9), and the new probabilities are (0.413, 0.279, 0.308). The wrong token rises from 0.10 to 0.31, and the correct token 1 loses 0.19.

**Evidence.**
- Adding verifier failures back as positives lowered BFCL by 4.06–5.94 points at 7B and 9.59–12.17 points at 1B ([[apigen]] Fig. 5). The smaller model lost more.
- Training on the lowest-IFD subsets gave the lowest winning scores of all Cherry LLM strategies ([[cherry-llm]] §4.2.3, Figure 4).
- Filtered Stack Exchange data scored 0.5 points higher than unfiltered data of the same size on LIMA's 1–6 helpfulness scale at 7B ([[lima-baseline]] §5).
- No source in this chapter measures the share of the gain attributable to removing negatives versus selecting positives.

**Controls.** Verify only subsets with a reliable check; for other rows, rely on format and IFD and spot-check. Hand-label rejected rows to estimate false rejects. Keep the rejected rows with their reason codes, so that later chapters can use them as content (APIGen used 8,000 relevance-detection refusal targets as ordinary SFT targets, [[apigen]] App. B.3) or as dispreferred responses.

**Diagnostics.** Rejection counts per stage × reason × category; false-reject estimates per verifier; the share of checkable rows before and after the verifier.

**Effect on generality.** A verifier that applies only to checkable categories rejects rows only in those categories, which lowers their share in the final pool when pass rates are low. Report the checkable share at each stage and compare it with the math and code columns of the evaluation.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value | Lab value | Reason for difference |
|---|---|---|---|---|---|---|---|---|---|
| Self-Instruct data (GPT-3 davinci) | — | SFT data | seed tasks; in-context examples; pool rule | 175; 8 (6 seed + 2 generated); ROUGE-L < 0.7 | arXiv:2212.10560 §2.2 | verified 2026-09-15 | Table 2 quality review; no ablation of the values | same | none |
| WizardLM-13b data | 13B | SFT data | evolver and responder; rounds M; operator choice | gpt-3.5-turbo; 4; one of 6 prompts, equal probability | arXiv:2304.12244v3 §4.2 | verified 2026-09-14 | Fig. 5b per-round scores (figure only) | lab teacher; M = 2; same | budget; recorded per row |
| WizardLM-13b data | 13B | SFT data | response sampling | temperature 1, max tokens 2048, top-p 0.9 | v3 §4.2 | verified 2026-09-14 | no ablation reported | same | none |
| WizardLM-13b | 13B | SFT | optimizer; LR; batch; epochs | Adam; 2 × 10⁻⁵; 4 per GPU on 8 V100; 3 | v3 §4.2 | verified 2026-09-14 | App. L Table 4: 57.92 / 58.24 / 58.96 at 2.5 / 2.75 / 3 epochs | see last row | different model size |
| Cherry LLM, LLaMA-7B | 7B | SFT data | pre-experience set; epochs | 1,000 = 100 K-means clusters × 10; 1 | arXiv:2308.12032v5 §2.1 | verified 2026-09-14 | Fig. 5: 300 samples give a distinct gain, 500 no further gain | same | none |
| Cherry LLM, LLaMA-7B | 7B | SFT data | selection rule; recommended share | drop IFD > 1, keep highest; top 10% | §2.2; App. G.2 | verified 2026-09-14 | App. B Table 5: IFD 52.06 vs random 50.61 at 5% | top 15% of deduplicated pool | larger share for the 5K target; `no-ifd` arm tests it |
| Cherry LLM, LLaMA-7B | 7B | SFT | LR; batch; epochs; max length (Alpaca) | 2 × 10⁻⁵; 128; 3; 512 | App. A; [[cherry-llm-recipe]] | verified 2026-09-14 | no ablation reported | see last row | different model size |
| Lee et al. NearDup (C4) | — | data dedup | shingles; hashes; buckets × hashes; thresholds | space-tokenized 5-grams; 9,000; 450 × 20; Jaccard > 0.8 and edit similarity > 0.8 | arXiv:2107.06499v2 §4.2, App. A | verified 2026-09-14 | App. A Fig. 4: 0.8 and 0.9 histograms vanish at the same point | word 5-grams; 128; 16 × 8; estimated J ≥ 0.8 | instruction-scale pool; confirmation by MinHash estimate |
| APIGen generators | — | SFT data | temperature; target samples per generator | 0.7; 40,000 | arXiv:2406.18518v1 §4.2 | verified 2026-09-14 | no ablation reported | 0.7 for tool rows | none |
| APIGen verifier | — | SFT data | stages; execution timeout; checker model | format, execution, semantic; not reported; not reported | §3.2 | verified 2026-09-14 (stages); not reported (timeout, checker) | Fig. 5 add-back ablation | same stages; 10 s; a model other than the generator | a timeout is required to run; teacher excluded as checker |
| xLAM-1B (FC) | 1.3B | SFT | peak LR; schedule; warmup; epochs; cutoff | 5 × 10⁻⁶; cosine; 50 steps; 4; 2048 | App. B.3 | verified 2026-09-14 | no ablation reported | candidate LR in sweep | closest reported size to the lab model |
| Tülu 3 8B SFT | 8B | SFT | LR; schedule; warmup ratio; effective batch; max tokens; epochs; loss | 5 × 10⁻⁶; linear; 0.03; 128; 4,096; 2; sum over tokens | arXiv:2411.15124 §4.3, Table 11 | verified 2026-09-15 | Figs. 5–6: sum loss with 5e-6 best; longer than 2 epochs no gain | batch 128; warmup 0.03; 2 epochs; mean loss; LR swept | mean loss changes the LR scale |
| Tülu 3 decontamination | — | eval-gate | match unit; item rule; dataset rule | 8-gram on prompts; > 50% of test tokens; > 2% of eval items | §3.2 | verified 2026-09-15 | Table 8 removal shares; no ablation of thresholds | same item rule; remove rows | single pool |
| Nemotron-PrismMath | — | distill-SFT data | answer agreement filter | N = 3 solutions, τ = 2 | arXiv:2505.20161v2 §3.1 | verified 2026-09-14 | no ablation reported | same for math rows without references | none |
| LIMA 65B | 65B | SFT | examples; epochs; LR; warmup; batch; checkpoint rule | 1,000; 15; 1e-5 → 1e-6 linear; none; 32; manual choice between epochs 5 and 10 | arXiv:2305.11206 §3 | verified 2026-09-15 | no ablation of these values | not used | different data size and selection protocol |
| DEITA-Mistral-7B 6K/10K | 7B | SFT | batch; epochs; LR; warmup; schedule | 512; 6; 2e-5; 0.1; cosine | arXiv:2312.15685v2 App. A | verified 2026-09-14 | epochs raised "to ensure adequate training", no ablation | not used | shows that 2 epochs is not a DEITA value |
| Lab SFT (all arms) | 0.5B–1.5B | SFT | LR; schedule; warmup; batch; epochs; seeds | not reported by any source for this size and data | — | not reported | — | LR ∈ {5e-6, 2e-5} chosen on dev split; linear; 0.03; 128 sequences; 2; 3 seeds | two LR values bracket the xLAM-1B and Cherry LLM rows |

Rows dated 2026-09-15 were read in the primary PDFs because the library cards for Self-Instruct, LIMA, and Tülu 3 had no Verification section; the chapter excerpts [[self-instruct-loop]], [[lima-baseline]], and [[tulu-3-sft-and-eval]] hold the extracts.

**Starting point for a small general-purpose run.** For SFT of a 1B-class base model on a few thousand synthetic rows, start from the verified Tülu 3 8B settings for schedule shape, warmup ratio 0.03, effective batch 128, and 2 epochs, which Tülu 3 selected for an 8B model on 939,344 prompts with a sum loss. Because the lab uses a mean loss and a smaller model, take the learning rate from a two-point sweep between the xLAM-1B value (5 × 10⁻⁶, 1.3B, cosine, 4 epochs) and the Cherry LLM value (2 × 10⁻⁵, 7B, 3 epochs), scored on development benchmarks only.

## Generalization lens

**(a) What increases breadth.**
- A pool rule against near-copies at generation time: Self-Instruct's ROUGE-L < 0.7 rule; in the resulting 52,445 instructions, the 20 most common root verbs with their top 4 objects cover 14%; the paper does not ablate the rule ([[self-instruct-loop]] §2.2, §3.2).
- Diverse real chat data in a general mixture: removing WildChat lowered Tülu 3 8B AlpacaEval 2 from 12.4 to 7.5 and the average from 60.1 to 58.9 ([[tulu-3-sft-and-eval]] Table 10).
- Prompt diversity at fixed size and quality: 2,000 heterogeneous Stack Exchange examples scored higher than 2,000 homogeneous wikiHow examples ([[lima-baseline]] §5).
- Diversity at fixed quality: a 10K subset with higher G-Vendi often outperformed a 100K subset with lower G-Vendi on OOD benchmarks ([[prismatic-synthesis]] §2.3.1).

**(b) What causes narrowing or forgetting.**
- Difficulty selection that shifts categories: IFD top 5% led by story and list generation, math and coding sub-category losses, and an MMLU drop from 41.73 to 36.51 on LLaMA-7B ([[cherry-llm]] Tables 1, 4, App. C).
- Seed composition: the ShareGPT seed lowered GSM8k from 37.15 to 31.46 ([[evol-instruct]] Table 2).
- Constraint templates that match the benchmark: models scoring well on IFEval's 25 constraint types score lower on 58 unseen constraints ([[ifbench]] §1, Fig. 1).
- Response length rewarded by judges: verbosity prompts moved a raw AlpacaEval win rate from 22.9% to 64.3% ([[length-controlled-alpacaeval]] §4.1).
- Contaminated synthetic data: 11.3% of NuminaMath-TIR overlapped MATH ([[tulu-3-sft-and-eval]] Table 8).
- A single evaluation family: APIGen reports BFCL only ([[apigen]] §5); DEITA's AlpacaEval and MT-Bench gains are not consistent ([[deita]] §3.3).

**(c) How to measure it for this stage.**
- Per-capability deltas against `base` and `general-mix` on a programmatic suite with a development/unseen split ([[tulu-3-sft-and-eval]] §2.2).
- Per-stage category shares and a coverage score (§3.4).
- 8-gram prompt contamination per benchmark plus a paraphrase check ([[tulu-3-sft-and-eval]] §3.2; [[prismatic-synthesis]] §3.1).
- 3 seeds, seed range, and instance-level intervals (§7).
- Known measurement errors: n-gram checks miss paraphrases; embedding diversity can rank datasets opposite to OOD accuracy ([[prismatic-synthesis]] Table 2); strict string verifiers have false negatives ([[ifeval]] §2.2); judge metrics have length and self-enhancement bias ([[judge-llm-bias]]).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Planning raw count from the target without the stage product | final pool far below target (the old example ended at 990 for a 5K target) | compute N_raw = N_target / Π s_i before generation; compare with `survival.json` |
| Applying IFD before dedup | final count below plan; duplicate clusters among top-IFD rows | count MinHash clusters within the IFD-kept set |
| Mixing the two IFD definitions | ranking changes when code is refactored | log s(A|Q) and s(A) separately; recompute both definitions on 100 rows |
| Different templates in the two IFD passes | many rows with IFD > 1 | re-score 50 rows with an identical prefix except the instruction |
| Reading only averages after IFD selection | judge score rises, MMLU-Pro or rewriting-type prompts fall | category report per stage; per-discipline MMLU-Pro |
| Using the teacher as semantic checker or judge | high pass rate and high win rate that a second checker does not reproduce | re-check 200 rows with a different model |
| String-match verifier without normalization | correct answers rejected | hand-label 20 rejected rows per verifier |
| Skipping contamination checks for synthetic data | GSM8K or HumanEval gain with no gain on held-out columns | 8-gram overlap per benchmark; compare with a paraphrase check |
| Unequal arm sizes | arm with more rows wins on every benchmark | assert `len(train) == N` for every arm |
| One seed per arm | ranking of arms changes between reruns | 3 seeds; report ranges |
| ln(V) as the step-1 loss target for a pretrained base | correct runs flagged, broken masking not flagged | compare step-1 loss with L_base on the same batch |
| Raw judge win rate as headline | win-rate gain with longer mean outputs | LC win rate and mean length per arm |
| Tuning thresholds on the final suite | suite scores stop being held out | log which benchmarks were viewed before each threshold choice |

## Check your understanding

1. The old example kept 15% at IFD and ended at 990 rows. Using the survival product, explain why raising the raw count is not the only fix, and what a larger IFD keep share would change in the category report.
2. Rows X and Y in §3.3 swap order between the loss-ratio and perplexity-ratio definitions of IFD. Explain which property of each formula causes the swap, and which kind of rows (low answer losses or high answer losses) each definition favors.
3. IFD selection on Alpaca lowered MMLU while AlpacaEval rose. Give a causal account that uses the category shift in Cherry LLM Table 4, and name the lab measurement that would detect the same effect before training.
4. APIGen's 1B model lost more BFCL accuracy than the 7B model when failed samples were added back. Using the softmax gradient in the Negative samples section, propose a reason the smaller model is more affected, and say what evidence would be needed to support it.
5. With lab bands r = 8 and b = 16, a pair at Jaccard 0.5 becomes a candidate 6.1% of the time. Explain why this does not remove the pair, and what would change if the confirmation step were deleted.
6. The `general-mix` arm contains data targeting IFEval's constraint types. Explain why a win of `full` over `general-mix` on IFBench's unseen constraints is stronger evidence of generality than a win on IFEval.
7. Two arms differ by 3 points on HumanEval with seed ranges of 1 point. Using §7, decide whether the lab may call the difference real, and explain which noise source dominates.
8. A Vendi score computed on embeddings rises after dedup while GSM8K falls. Using the Prismatic Synthesis findings, explain how both can be true.

## Connections

- Previous: ch-28 — Long-Context Data Synthesis and Synthetic Evaluation Task Families.
- Next: ch-29a — Long-Document Synthesis for Continued Pretraining and Long-Context SFT.
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix (the pattern this lab instantiates).
- ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing (Self-Instruct and Evol-Instruct).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection (IFD, DEITA, G-Vendi).
- ch-23 — Model Collapse and Verification of Synthetic Data; ch-26 — Tool and Function-Calling Data (APIGen).
- ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic (MinHash and suffix arrays).
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks; ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate; ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares.
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split; ch-44 — Process Supervision and Verifiable Rewards.
- ch-47 — Evaluation Harness and Suite Design for General Capability; ch-48 — Contamination Detection and Its Effect on Reported Scores; ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting; ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions.

## Sources

- [[self-instruct-loop]] — chapter excerpt of Self-Instruct read from the arXiv PDF: sampling, output-first branch, filters, Tables 1–2. [[self-instruct]] is the library card (not verified at writing time; not used for numbers).
- [[evol-instruct]] — operators, prompt texts, elimination rules, rounds, Tables 1–2, recipe ledger.
- [[cherry-llm]], [[cherry-llm-recipe]] — IFD Eq. 3–5, pre-experience selection, Tables 1, 4, 5, App. C, SFT settings.
- [[ifd-filter]] — chapter excerpt: both IFD definitions, released script lines, Superfiltering Table 1 overlap. [[ifd]] and [[superfiltering]] are library cards not verified at writing time.
- [[minhash-lsh]] — Jaccard resemblance, min-wise sketch estimator, banding formula pointer.
- [[deduplicating-training-data]] — NearDup parameters, C4 duplicate and memorization rates.
- [[apigen]] — three-stage verifier, pass rates, add-back ablation, SFT settings of xLAM-1B/7B.
- [[lima-baseline]] — chapter excerpt of LIMA read from the arXiv PDF: composition, training, §5 ablations. [[lima]] is the library card (not verified at writing time).
- [[deita]] — epochs of DEITA models, LIMA 1K as a baseline, benchmark disagreement.
- [[prismatic-synthesis]] — Vendi and G-Vendi definitions, diversity–OOD correlations, answer-agreement filter, paraphrase decontamination.
- [[tulu-3-sft-and-eval]] — chapter excerpt of the Tülu 3 report: development/unseen suite, 8-gram decontamination, Table 10 ablations, Table 11 settings, Table 14 seeds. [[tulu-3-sft-mix]] is the dataset card (not verified at writing time).
- [[ifeval]] — chapter excerpt: 25 instruction types, 541 prompts, strict and loose metrics.
- [[ifbench]] — unseen verifiable constraints as a check on IFEval overfitting.
- [[mmlu-pro]] — chapter excerpt: 12,032 questions in 14 disciplines, ten options, prompt sensitivity.
- [[length-controlled-alpacaeval]] — chapter excerpt: GLM with length term, LC win rate, gameability and Arena correlation.
- [[signal-and-noise-eval]] — checkpoint-to-checkpoint noise at 1B.
- [[judge-llm-bias]] — judge position, verbosity, and self-enhancement biases.
- [[karpathy-training-neural-net-recipe]] — "verify loss @ init" check for randomly initialized models.
