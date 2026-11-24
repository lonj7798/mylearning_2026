<!-- chapter: ch-29
     track: synthetic
     kind: lab
     title: Lab: Filter, Deduplication, Verification을 갖춘 Synthetic Instruction Set
     deps: [ch-28]
     sources: [[self-instruct]], [[self-instruct-loop]], [[evol-instruct]], [[cherry-llm]], [[cherry-llm-recipe]], [[ifd]], [[ifd-filter]], [[superfiltering]], [[minhash-lsh]], [[deduplicating-training-data]], [[apigen]], [[lima]], [[lima-baseline]], [[deita]], [[prismatic-synthesis]], [[tulu-3-sft-mix]], [[tulu-3-sft-and-eval]], [[ifeval]], [[ifbench]], [[mmlu-pro]], [[length-controlled-alpacaeval]], [[signal-and-noise-eval]], [[judge-llm-bias]], [[karpathy-training-neural-net-recipe]]
     figures: figures/filter-cascade.html
     revised: 2026-09 (generality revision)
-->

# 29장 — Lab: Filter, Deduplication, Verification을 갖춘 Synthetic Instruction Set

> **핵심 통찰.** Synthetic instruction set은 filter가 tuning하지 않은 held-out capability를, base model 및 같은 크기의 general-purpose mixture와 비교해 개선할 때만 general-purpose model에 유용하다. 이 lab의 filter는 quality 이상을 바꾼다. Alpaca에서 IFD selection은 story·list generation을 유지하고 sentence rewriting·editing을 제거했으며, 5% model은 judge-based score가 올랐는데도 MMLU를 41.73에서 36.51로 잃었다([[cherry-llm]] Table 1, 4). Verifier stage는 failure를 check할 수 있는 곳에서 측정된 가치가 있다. APIGen execution check에 실패한 sample을 되돌려 넣으면 BFCL은 xLAM-7B에서 5.94점, xLAM-1B에서 12.17점 낮아졌고, semantic-check failure를 되돌려 넣으면 4.06점과 9.59점 낮아졌다([[apigen]] Fig. 5). 따라서 이 lab은 모든 stage에 survival count, category share, coverage score, contamination을 기록하고 programmatic multi-capability suite에서 seed 3개로 결정한다.
>
> **지침.** Filtering 뒤 synthetic pool이 target size N에 도달해야 한다면, generation 전에 raw count를 stage survival rate의 곱으로 나눈 값으로 계산하라. 15% selection stage 하나만으로도 필요한 raw count가 6.7배가 되기 때문이다. Duplicate가 있는 pool에 keep-share를 적용하면 copy에 slot을 쓰므로 IFD selection 전에 deduplicate하라. Filter가 category share를 5점보다 많이 바꾸면 category removal이 capability별 loss를 예측하므로 category report를 benchmark table 옆에 둔다([[tulu-3-sft-and-eval]] Table 10: math data 없이 GSM8K 76.2 → 64.1). 비교가 LLM judge를 사용하면 verbosity prompt만으로 한 model의 raw AlpacaEval win rate가 22.9%에서 64.3%로 이동했으므로 length-controlled win rate를 보고한다([[length-controlled-alpacaeval]] §4.1). Arm이 더 낫다고 부르는 것은 3-seed mean difference가 seed range와 instance-level confidence interval을 모두 넘을 때뿐이다.

## 학습한 버전에 대한 정정

1. “5000 raw → N_format → N_IFD → N_dedup → N_verified”, 그리고 “cascade가 약 5K target으로 prune할 headroom이 있도록 raw 약 10K” 및 예시 `{"raw": 10000, "format": 7800, "ifd": 1170, "dedup": 1050, "verified": 990}` → 예시는 5K가 아니라 990 row로 끝난다. 같은 stage rate(0.78, 0.15, 0.897, 0.943; 곱 0.0990)에서 5,000-row pool에는 raw 약 50,500 row가 필요하다(도출; §1).
2. “`pre_clip_grad_norm`과 step-1 loss를 log하고, `ln(V) ± 20%` 밖이면 misconfigured” 및 acceptance gate 5 → ln(V)는 randomly initialized softmax classifier의 expected loss다(“verify loss @ init … -log(1/n_classes)”, [[karpathy-training-neural-net-recipe]]). Pretrained base는 lab이 측정하는 자체 response-token loss에서 SFT를 시작한다. ln V는 V token 위 uniform prediction의 loss(ln 128,000 = 11.76)다. Pretrained model은 text를 uniform보다 잘 예측하므로 correct run은 ln V 아래에서 시작하며 그 gate를 통과하지 못한다. 정정한 check는 같은 batch에서 step-1 loss를 base model의 measured loss와 비교한다(§6). “grad norm < 10” threshold는 source가 없어 제거한다.
3. “[[apigen]] ablation: format 제거 −18% BFCL-V1, exec −11%, judge −6%” → APIGen은 removal ablation이 아니라 add-back ablation을 보고하며 format arm이 없다. Semantic failure 추가는 BFCL을 −4.06(xLAM-7B), −9.59(xLAM-1B), execution failure 추가는 −5.94, −12.17 바꿨다([[apigen]] Fig. 5). Generator pass rate는 34.42%–84.15%다(Table 1).
4. “[[apigen]]은 DeepSeek-Coder-V2 + GPT-4를 사용” 및 “sandboxed execution(tool/code, timeout 5s) … LLM-judge”가 APIGen의 세 종류라는 설명 → Generator는 DeepSeek-V2-Chat, DeepSeek-Coder-33B-Inst, Mixtral-8x22B-Inst, Mixtral-8x7B-Inst다. Semantic-checker model과 execution timeout은 보고하지 않는다. APIGen stage는 format, execution, semantic check이며 reasoning answer exact-match check는 APIGen에 속하지 않는다([[apigen]] §3.2, §4.2).
5. Cherry LLM에 귀속한 “[[ifd]]: `IFD(q, a) = PPL(a|q) / PPL(a)`” → Cherry LLM은 mean answer loss `s(A|Q)/s(A)`의 ratio로 IFD를 정의하고(Eq. 5), script도 그 ratio를 계산한다. Perplexity ratio는 Superfiltering Eq. 2다. 두 ordering은 달라질 수 있다(§3.3)([[cherry-llm]] §2.2; [[ifd-filter]]).
6. “[[cherry-llm]]에 따라 random sample 약 1K에서 scoring model을 1 epoch warm” → 100 K-means cluster 각각에서 10개씩 골라 1,000 sample을 선택하고 1 epoch 학습했다. Random choice는 ablation이었다([[cherry-llm]] §2.1, Table 2).
7. “[[cherry-llm]]에 따라 post-format pool의 예상 survival 약 10–15%(‘top 10% beats full set’)” → LLaMA-7B에서 Alpaca 5%는 official Alpaca를 이기지만 WizardLM 10%는 leaderboard average 및 AlpacaEval에서 reimplemented full-data WizardLM보다 낮다. 저자는 top 10%를 “a safe and reasonable choice”라 부른다([[cherry-llm]] Table 1, App. G.2).
8. `tpl = INPUT_FIRST if kind == "classification" else OUTPUT_FIRST` → Self-Instruct는 classification에는 output-first generation을 쓴다. Input-first가 한 label에 편향된 input을 만들었기 때문이다. 다른 task에는 input-first를 쓴다([[self-instruct-loop]] §2.2).
9. “[[self-instruct]] §Filtering에 따라 empty output, `input == output`, 길이 < 3 또는 > 2K token, ‘image’/‘graph’/‘file’을 drop”, “Self-Instruct는 약 67%를 보고” → Self-Instruct는 기존 instruction 각각과 ROUGE-L이 0.7 미만일 때만 instruction을 pool에 추가하고, image·picture·graph 등의 keyword를 제외하며, identical instance 및 same-input-different-output instance를 제거하고, 너무 길거나 짧은 instruction을 제거한다. 본문에 numeric threshold는 없고 survival rate도 보고하지 않는다([[self-instruct-loop]] §2.2).
10. `evol_elimination` rule “same-or-similar / refusal(‘sorry’, ‘i cannot’, ‘as an ai’) / punct-only / verbatim” → WizardLM은 ChatGPT가 no information gain을 판단할 때, response에 “sorry”가 있고 80 word 미만일 때, punctuation과 stop word만 있을 때, instruction이 evolving prompt의 word를 copy할 때 evolution을 failed로 표시한다. Failed instruction은 pool로 돌아간다([[evol-instruct]] §3.2).
11. `"breadth": "Mutate to a new instruction in a rarer domain"` → in-breadth prompt는 같은 domain에서 similar length·difficulty로 “even more rare”한 새 prompt를 요구한다([[evol-instruct]] Example 3.3).
12. “Self-Instruct/Evol-Instruct 70/30을 mix … complexity tail이 [[evol-instruct]] Figure 1이 문서화한 win condition” → Figure 1은 “1+1=?”에서 자란 evolution tree다. WizardLM은 merged seed set과 모든 round에서 equal probability로 example 70k를 sample한다. 70/30 mix를 시험한 source는 없다([[evol-instruct]] §3.3, §4.2).
13. “Attested pretraining setup은 Jaccard 약 0.8에 `b=20 × r=450`, lab scale은 `threshold=0.8`의 datasketch default로 충분” → Lee 등은 hash 9,000개를 20개씩 450 bucket에 쓰고 candidate probability `1 − (1 − s^20)^450`를 사용한 뒤 Jaccard > 0.8 및 edit similarity > 0.8 pair만 유지하고 connected component로 cluster한다. Library default는 source value가 아니다([[deduplicating-training-data]] §4.2, App. A).
14. “`lima-matched` — raw에서 random N, LIMA-recipe proxy filter” → LIMA는 source-specific rule과 manual choice로 community Q&A pair 750개를 고르고 example 250개를 직접 썼다. Algorithmic LIMA filter는 없으므로 arm은 정의되지 않는다. Matched-size random arm과 matched-size Tülu 3 SFT mixture arm으로 대체한다([[lima-baseline]] §1–§2).
15. “`epochs=2` … teacher-generated 약 1K–5K에서 2 epoch가 [[deita]]와 일치” → DEITA는 6K/10K set을 6 epoch, Tülu 3 8B SFT는 2 epoch, LIMA는 epoch 5–10 checkpoint selection을 하며 15 epoch 학습했다([[deita]] App. A; [[tulu-3-sft-and-eval]] Table 11; [[lima-baseline]] §3).
16. “GPU 8개에서 약 30-minute run에 충분히 작고 benchmark delta가 noise를 넘기에 충분히 큼” → 어느 source도 이 claim을 뒷받침하지 않는다. Tülu 3에서 70B SFT average의 seed-to-seed spread는 2.6점(Table 14)이고, accuracy 25%의 164-problem benchmark 95% interval은 ±6.6점이다(§7).
17. “Judge가 <3% reject면 pool은 이미 clean”, “예상 survival 70–85%”, “post-IFD pool의 80–95%”, “>30% kill = collapsed generator” → source가 이 rate를 보고하지 않는다. Lab이 측정한다.
18. Acceptance gate 2 “histogram center가 1 아래 … 1 위면 warm-up skipped([[ifd]] §Practical guidance)” → Cherry LLM은 histogram claim이 없다. Pre-experienced model 없이 그 결과는 시험한 count 중 가장 낮지만 Alpaca는 10%에서도 이겼고, LLaMA2 run은 base model을 scorer로 쓴다([[cherry-llm]] §4.3.1, §4.4).
19. Connections “ch-19 / ch-20 — [[self-instruct]]와 [[evol-instruct]]”, “ch-23 / ch-25 / ch-27 — IFD, MinHash, APIGen”, “ch-30이 `full` checkpoint를 다음으로 가져감”, “Track 4(RL)” → 두 generator는 ch-19, selection은 ch-22, deduplication은 ch-12, tool-call verification은 ch-26에 있다. 다음 chapter는 ch-29a, ch-30은 SFT design choice다. Policy-gradient 및 RL chapter는 preference/RL phase(ch-37부터)에 있고 verifiable reward는 ch-44에 있다(outline).
20. Figure label “MinHash Jaccard threshold (0=loose, 1=strict)” → 높은 Jaccard threshold는 duplicate로 표기하는 pair를 줄여 제거 row를 줄이므로 1이 가장 permissive setting이다. Figure를 고친다.

## 이 장이 general-purpose model에 중요한 이유

ch-18은 generate → filter → deduplicate → verify → select → mix pattern을 설명했다. 이 lab은 single-turn instruction SFT에 그 instance 하나를 만들고, 결과가 자신을 위해 만들어지지 않은 capability에서 model을 돕는지 묻는다. Data는 pretraining·mid-training 뒤, preference optimization·RL 전에 SFT에서 pipeline에 들어간다.

측정 문제 세 가지가 질문을 어렵게 한다. 첫째, filter는 task distribution을 바꾸므로 더 좁은 distribution에서 학습한 model은 judge-based chat metric을 얻으면서 제거 task의 ability를 잃을 수 있다([[cherry-llm]] App. C, Table 1). 둘째, teacher model은 benchmark item을 regenerate할 수 있으므로 gain이 contamination에서 올 수 있다([[tulu-3-sft-and-eval]] §3.2는 MATH overlap 때문에 NuminaMath-TIR의 11.3%를 제거). 셋째, single SFT run은 seed에 따라 달라 one-seed delta가 noise일 수 있다([[tulu-3-sft-and-eval]] Table 14). Lab은 문제마다 control 하나를 둔다. Stage별 category·coverage report, 모든 evaluation set에 대한 contamination check, confidence interval을 둔 seed 3개다.

이 장은 earlier form으로 학습됐다. Generation과 filter stage는 section order를 유지한다. §7의 held-out multi-capability evaluation은 earlier lab을 이미 끝낸 learner에게 optional re-run이다.

## §1 Lab goal, arm, survival arithmetic

**목표.** N row instruction pool을 만들고 그것 및 baseline에서 matched-size model을 학습하며 base model 및 general mixture 대비 capability별 delta를 보고한다. Deliverable은 다음과 같다.

1. `instructions.jsonl` — sample당 row 하나이며 `origin`(seed, self_instruct, evol:<op>, round), `category`, `ifd`, `minhash_cluster`, `verifier_verdict`(checkable subset만), `contam_hits`를 둔다.
2. `survival.json` — stage당 integer count, stage당 category count, stage당 coverage score(§3.4).
3. `contamination.json` — evaluation set별 overlap row(§5).
4. `eval_table.csv` — 모든 arm × benchmark × seed(§7).
5. `synthetic-set-memo.md` — 한 page(§8).

**Survival rate.** Stage i의 survival rate `s_i`는 input row 중 keep하는 fraction이다.

```
N_k = N_raw · s_1 · s_2 · … · s_k            N_raw = ceil( N_target / (s_1 · … · s_K) )
```

`N_raw`는 raw generation 수, `N_k`는 stage k 뒤 count, K는 stage 수다.

**계산 예(measured value로 교체할 illustrative rate).** Stage order는 format → dedup → IFD → verifier다(dedup을 IFD 전에 하는 이유는 §3.2).

| Stage | Rate | Full-budget path | Resource-constrained path |
|---|---|---|---|
| raw | — | 50,000 | 10,000 |
| format filter | 0.80 | 40,000 | 8,000 |
| near-duplicate removal | 0.90 | 36,000 | 7,200 |
| IFD selection(deduplicated pool의 IFD ≤ 1 중 상위 15%) | 0.15 | 5,400 | 1,080 |
| verifier(40% row가 checkable, 그중 75% pass) | 1 − 0.40 × 0.25 = 0.90 | 4,860 | 972 |
| training size N(모든 arm에 같은 random subsample) | — | 4,800 | 960 |

Rate 곱은 `0.80 × 0.90 × 0.15 × 0.90 = 0.0972`이므로 5,000-row target의 `N_raw = 5,000 / 0.0972 ≈ 51,440`이다. IFD stage는 `1/0.15 = 6.7` factor를 낸다.

[figures/filter-cascade.html](figures/filter-cascade.html)은 editable stage rate에서 이 count를 계산하고, target size에 닿는 raw count를 풀며, category-dependent IFD keep rate가 category share를 바꾸는 방식을 보이고, 선택한 band setting의 MinHash-LSH candidate probability를 plot한다.

**Path.** 두 path는 같은 stage, arm, suite, seed 3개를 쓴다. Full-budget의 base model은 `meta-llama/Llama-3.2-1B` 또는 `Qwen/Qwen2.5-1.5B`다. Resource-constrained는 `Qwen/Qwen2.5-0.5B` 또는 `HuggingFaceTB/SmolLM-360M`, 더 작은 benchmark subset(§7), `N = 960`을 쓴다. Compute와 API cost는 여기서 estimate하지 않으며 memo에 기록한다.

**Arm**(모두 training size N, 각각 seed 1, 2, 3).

| Arm | Data | 답하는 질문 |
|---|---|---|
| `base` | SFT 없음(same chat prompt로 평가) | forgetting reference |
| `full` | format → dedup → IFD → verifier | headline pool |
| `no-ifd` | format → dedup → verifier, random N | IFD selection 가치 |
| `no-dedup` | format → IFD → verifier | near-duplicate removal 가치 |
| `no-verify` | format → dedup → IFD | checkable row에서 verifier 가치 |
| `random-format` | format filter만, random N | 이후 모든 stage를 합친 가치 |
| `general-mix` | decontaminated Tülu 3 SFT mixture의 random N row | synthetic pool이 general mixture를 이기는지 |
| `mix-evol-heavy`(optional) | evolution round 쪽 generation을 옮긴 `full` pipeline | evolution depth가 seed에 없는 category의 breadth와 trade-off하는지 |

## §2 Stage 1: bootstrap 및 evolution operator를 사용한 generation

### §2.1 Self-Instruct bootstrap

**정의.** Self-Instruct는 점점 커지는 task pool에서 sample한 example으로 language model을 prompt하고 instance를 생성한 뒤 filter하여 instruction을 생성한다([[self-instruct-loop]] §2.2).

**문제.** Seed set에는 task 175개가 있으며, 같은 example으로 반복 prompt하면 그것의 near-copy가 돌아온다. 측정 target은 이전의 모든 instruction과 ROUGE-L이 0.7 미만인 distinct instruction 수다.

**메커니즘.**
1. Instruction 8개를 sample한다. Human-written seed 175개에서 6개, model-generated task에서 2개다.
2. “Come up with a series of tasks:”를 “Task 1:” … “Task 8:”과 함께 prompt하고 “Task 9:”을 읽는다.
3. Task가 finite output label을 가진 classification task인지 묻는다.
4. Classification에서는 label을 먼저 생성하고 각 label의 input을 생성한다(output-first). 다른 task에서는 input을 생성하고 output을 생성한다(input-first).
5. Pool의 모든 instruction과 ROUGE-L이 0.7 미만일 때만 instruction을 pool에 추가한다.

**근거.** GPT-3 davinci는 instruction 52,445개와 instance 82,439개를 만들었다. 저자가 sample 200개를 검토했을 때 instruction의 92%는 valid task였고 output의 58%는 correct였으며 54%는 모든 field가 valid였다(Table 1, Table 2). Result(단일 연구).

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

**한계.** ROUGE-L check는 새 instruction 각각을 pool의 모든 instruction과 비교하므로 비용이 pool size의 제곱으로 증가한다. Lab은 instruction pool에서는 이를 유지하고 전체 sample text에는 MinHash를 사용한다(§3.2).

### §2.2 Evol-Instruct evolution

**정의.** Evol-Instruct는 LLM에 instruction을 더 어려운 것으로 rewrite하도록 요청한다. In-depth는 constraint 추가, deepening, concretizing, reasoning step 증가, input 복잡화를 쓰며, in-breadth는 같은 domain에서 더 드문 새 instruction을 만든다([[evol-instruct]] §3.2).

**메커니즘.**
1. 각 round의 각 instruction에 대해 prompt 여섯 개 중 하나를 equal probability로 sample한다(§4.2).
2. In-depth prompt는 rewrite를 word 10–20개 추가로 제한한다(Example 3.1).
3. 같은 LLM으로 evolved instruction에 대한 response를 생성한다(§3.2).
4. Four rule로 failed evolution을 제거한다. Failed instruction은 다음 round를 위해 pool로 돌아간다(§3.2).
5. M round 뒤 seed set과 모든 round를 merge하고 training set을 equal probability로 sample한다(§3.3).

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

**근거.** Evolved instruction 250k개(Alpaca seed 52k, `M = 4`, gpt-3.5-turbo)에서 sample한 70k로 학습한 LLaMA 13B는 benchmark 9개 평균 58.96을 내며 Vicuna-13b의 54.60, 재학습한 Alpaca-13b의 43.44에 비해 높았다([[evol-instruct]] Table 1). Seed도 중요하다. ShareGPT seed는 평균을 61.87로 높였지만 GSM8k를 37.15에서 31.46으로 낮췄다. 저자는 math instruction이 더 적었기 때문(4.3% 대 11.8%)이라고 귀속한다(Table 2, §4.5). Result(단일 연구).

**조건과 한계.** 저자는 extremely complex instruction만으로 된 set은 “would harm the generalization performance”하므로 small step으로 difficulty를 올린다(§3.2, 저자의 Interpretation; step size를 ablation하지 않음). 이 장의 source는 Self-Instruct와 Evol-Instruct sample의 ratio를 시험하지 않는다. Lab은 final pool ratio를 알 수 있게 row마다 `origin`을 기록하고 optional `mix-evol-heavy` arm으로 이를 측정한다.

**함의.** Seed와 operator mix는 어떤 filter가 실행되기 전 category distribution을 정한다. Raw stage에서 category share를 기록한다(§3.4).

## §3 Stage 2: Filter cascade

### §3.1 Format filter

Format filter는 training target이 될 수 없는 row를 제거한다. Empty field, output=input, 같은 input의 서로 다른 output, non-text input을 뜻하는 keyword(Self-Instruct는 image, picture, graph를 열거), tool-call row의 invalid JSON, templating 뒤 training context length를 넘는 output이다. Context bound는 lab-specific다. Cherry LLM은 Alpaca에 token 512개, WizardLM에 1024개를 썼다([[cherry-llm-recipe]]). Memo가 category별 reason count를 보고할 수 있게 row별 rejection reason을 기록한다.

### §3.2 MinHash LSH를 이용한 near-duplicate 제거

**정의.** Document 두 개의 resemblance는 shingle set의 Jaccard ratio `J(A, B) = |S(A) ∩ S(B)| / |S(A) ∪ S(B)|`다. `S(D)`는 D의 contiguous w-token shingle set이다([[minhash-lsh]] §2). MinHash signature는 full text 비교 없이 J를 estimate한다. Locality-sensitive hashing(LSH)은 signature를 band로 묶어 whole band에서 match하는 pair만 비교한다.

**문제.** Evolution round와 repeated sampling은 몇 word만 다른 row를 만든다. Exact matching은 이를 놓치며 N row all-pairs comparison은 `O(N²)` 비용이다.

**메커니즘.**
1. 각 row의 instruction + output을 word 5-gram으로 shingle한다(Lee 등은 space-tokenized 5-gram 사용, [[deduplicating-training-data]] §4.2).
2. k hash minimum의 signature를 계산한다. Random permutation 하나에서 union minimum이 두 set 모두에 있을 확률은 J다([[minhash-lsh]] §3, Theorem 1).
3. Signature를 r value의 b band로 나눈다. Band 하나라도 정확히 match하면 두 row는 candidate가 된다.
4. Similarity check로 candidate pair를 confirm하고 confirmed pair를 connected component로 cluster하며 cluster당 row 하나를 유지한다.

**수식.** `P(candidate | J = s) = 1 − (1 − s^r)^b`다. s는 pair Jaccard similarity, r은 band당 hash value 수, b는 band 수다([[minhash-lsh]] Connections, MMDS §3.4.2; [[deduplicating-training-data]] §4.2는 bucket당 hash 20개와 bucket 450개로 쓴다).

**계산 예.** Lee 등은 `r = 20`, `b = 450`(`k = 9,000`)을 쓴다. `s = 0.8`에서 `s^20 = 0.0115`, `P = 1 − (0.9885)^450 = 0.995`이며 `s = 0.5`에서 `P = 0.0004`다. `k = 128`인 lab setting은 `r = 8`, `b = 16`이다. `s = 0.8`에서 `s^8 = 0.168`, `P = 1 − (0.832)^16 = 0.947`; `s = 0.7`에서 P=0.613; `s = 0.5`에서 P=0.061이다. Lab setting은 low-similarity candidate를 더 많이 받아들이므로 confirmation step(estimated `J ≥ 0.8`)이 final decision을 한다.

**근거.** C4에서 NearDup(Jaccard > 0.8 및 edit similarity > 0.8)은 training example의 3.04%를 near-duplicate로 표시했고 validation example의 4.60%는 training에 near-duplicate가 있었다(Table 2). Original C4로 학습한 1.5B model은 memorized token 1.926%를 출력했으나 NearDup 뒤에는 0.189%였다(Table 4). Downstream task effect는 평가하지 않았다(§2). Result(단일 연구, pretraining scale).

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

**Dedup이 IFD보다 앞서는 이유.** IFD는 input의 share를 keep한다. Duplicate가 남아 있으면 high-IFD row의 여러 copy가 selection slot을 차지한 뒤 dedup으로 제거되어 final count와 category share가 planned keep share와 달라진다. MinHash는 model forward pass가 없지만 IFD는 row당 두 번 필요하다. 이 순서는 course recommendation(Interpretation)이며 source가 두 순서를 비교하지는 않는다.

**Threshold 방향.** Confirmation threshold를 0.8에서 0.9로 높이면 duplicate로 표시하는 pair와 제거 row가 줄어든다. 낮추면 더 많이 제거한다.

### §3.3 IFD selection

**정의.** Instruction-Following Difficulty(IFD)는 model이 instruction을 포함하거나 빼고 answer를 얼마나 잘 예측하는지 비교한다([[cherry-llm]] §2.2).

```
s(A|Q) = −(1/N) Σ_i log P(w_i | Q, w_<i; θ)     s(A) = −(1/N) Σ_i log P(w_i | w_<i; θ)
IFD_loss(Q, A) = s(A|Q) / s(A)                   (Cherry LLM, Eq. 5)
IFD_ppl(Q, A)  = exp(s(A|Q)) / exp(s(A)) = exp(s(A|Q) − s(A))   (Superfiltering, Eq. 2)
```

Q는 training template의 optional input을 포함한 instruction, A는 N token `w_i`의 answer, θ는 scoring model weight다. 높은 IFD는 instruction 도움이 더 적음을 뜻한다. IFD > 1 row를 제거하고 남은 값 중 가장 높은 것을 keep한다([[cherry-llm]] §2.2).

**계산 예.** Row X: `s(A|Q) = 0.2`, `s(A) = 0.4`; Row Y: `s(A|Q) = 2.4`, `s(A) = 3.0`이다. IFD_loss는 X=0.50, Y=0.80이므로 Y를 높게 rank한다. IFD_ppl은 X=`exp(−0.2)=0.819`, Y=`exp(−0.6)=0.549`이므로 X를 높게 rank한다. Top-1 selection은 definition마다 다른 row를 keep한다. Lab은 selection result를 cite하는 논문의 Eq. 5를 쓰고 definition을 memo에 쓴다.

**메커니즘.**
1. Pre-experience: base model로 instruction을 embed(mean last-layer hidden state)하고 K-means cluster 100개를 실행해 cluster당 row 10개, 총 1,000개를 취한 뒤 base model을 1 epoch 학습한다(§2.1).
2. Deduplicated row 모두를 두 pass에서 같은 template으로 Eq. 3–5로 score한다.
3. IFD > 1을 drop하고 sort한 뒤 top share를 keep한다. Released script는 IFD ≤ 1 및 length check를 통과한 row에서 kept count를 계산한다(`data_by_IFD.py` L117–L120, [[ifd-filter]]).

**근거.** LLaMA-7B에서 5% Alpaca selection은 official Alpaca보다 leaderboard average 52.06 대 50.21, AlpacaEval 34.74 대 26.46이다. 10% WizardLM selection은 reimplemented full-data model보다 양쪽에서 낮다(51.59 대 52.79; 61.44 대 61.99)([[cherry-llm]] Table 1). 같은 5% size에서 IFD selection 평균은 random selection의 50.61 대비 52.06이다(App. B Table 5). Weak scorer로는 7B scorer selection과 overlap이 partial이다. GPT-2와 LLaMA2-7B는 Alpaca 10% share에서 같은 row의 41%를 select한다([[ifd-filter]], Superfiltering Table 1). Result(각각 단일 연구).

**조건과 한계.** Score는 주어진 answer loss만 쓰며 correctness를 확인하지 않는다(Eq. 3–5). IFD selection은 category mix를 옮긴다. Alpaca top 5%는 “write story”(119), “generate story”(98)가 앞서며 bottom 5%는 “rewrite sentence”(155), “edit sentence”(89)가 앞선다(Table 4). 5% model은 Math 및 Coding sub-category에서 official Alpaca보다 나쁘고(App. C), LLaMA-7B MMLU는 41.73에서 36.51로 떨어진다(Table 1).

**함의.** IFD selection은 narrowing의 candidate cause다. `no-ifd` arm과 §3.4 category report가 이를 탐지하는 두 measurement다.

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

### §3.4 Stage별 category distribution과 coverage

**정의.** Category report는 stage별 task category의 row count와 share를 나열한다. Coverage score는 raw pool variety 중 얼마나 살아남는지 요약한다.

**문제.** Filter는 task type을 제거하면서 average quality를 높일 수 있다. 측정량은 raw stage에서 final stage까지 category별 share 변화다.

**계산 예([[cherry-llm]] Table 4와 같은 방향의 illustrative keep rate).** Deduplicated pool은 row 36,000개이고 IFD는 5,400개를 유지한다.

| Category | Dedup 뒤 | IFD keep rate | IFD 뒤 | 전 share → 후 share |
|---|---|---|---|---|
| open generation | 14,400 | 0.24 | 3,456 | 40% → 64% |
| rewriting and editing | 7,200 | 0.03 | 216 | 20% → 4% |
| math reasoning | 9,000 | 0.12 | 1,080 | 25% → 20% |
| tool call | 5,400 | 0.12 | 648 | 15% → 12% |

Total keep rate는 계획대로 `5,400 / 36,000 = 0.15`지만 rewriting share는 5배 줄어든다.

**Coverage score.**
1. *Cluster coverage.* Deduplicated pool을 embed하고 K-means K cluster를 실행한다(lab은 `K = 200`). 각 stage 뒤 적어도 row 3개를 유지하는 cluster fraction을 보고한다.
2. *Vendi score.* `VS(D) = exp(−Σ_i λ_i log λ_i)`이며 λ_i는 `K/|D|`의 eigenvalue, `K_ij`는 row i,j normalized embedding의 cosine similarity다. 모든 row가 orthogonal이면 VS=|D|, 모두 identical이면 1이다([[prismatic-synthesis]] Eq. 3, gradient version은 App. C). Row 세 개 중 두 개가 identical이고 하나가 orthogonal인 예에서 `K/3` eigenvalue는 2/3, 1/3, entropy=0.637, VS=1.89다.

**어느 diversity measure가 generality를 예측하는지의 근거.** Data quality를 고정하면 G-Vendi(Qwen2.5-0.5B-Instruct의 gradient feature)는 math OOD accuracy와 Spearman `ρ = 0.899`, embedding Vendi는 0.754로 상관한다([[prismatic-synthesis]] Table 1). Embedding Vendi는 OOD accuracy가 16.6점 더 높은 seed-varied set보다 persona-varied set을 높게 rank했다(Table 2). 저자는 correlation이 controlled quality에서만 성립한다고 말한다(App. D). Result(단일 연구). Lab에서 embedding coverage는 monitoring number이고 capability별 evaluation이 결정한다.

**Category removal이 capability별 loss를 예측한다는 근거.** Tülu 3 8B SFT mixture에서 math data를 제거하면 GSM8K는 76.2에서 64.1, MATH는 31.5에서 23.5로 낮아졌다. Persona data 제거는 IFEval을 72.8에서 53.6으로 낮췄다([[tulu-3-sft-and-eval]] Table 10). Result(단일 연구).

## §4 Stage 3: reasoning 및 tool subset용 verifier gate

**정의.** Verifier는 row answer가 correct인지 결정하는 program 또는 model call이다. Lab은 correctness를 check할 수 있는 row에만 적용한다. Math·logic item, test가 있는 code, executable function이 있는 tool call이다.

**문제.** Format filter와 IFD는 correctness를 check하지 않는다. Self-Instruct가 유지한 data는 sample 200개 review에서 correct output이 58%였다([[self-instruct-loop]] Table 2).

**Subset별 메커니즘.**
1. *Tool call(APIGen stage, [[apigen]] §3.2).* (a) Format: required field가 있는 JSON, 호출한 모든 function 및 argument가 주어진 API list에 존재. (b) Execution: 별도 subprocess에서 Python function을 실행하고 type error, invalid parameter, runtime error, timeout, missing argument를 제거. (c) Semantic: 두 번째 LLM이 function, query, call, execution result를 받고 `{"thought", "pass"}`를 반환한다.
2. *Code.* Sandbox의 test에 대해 solution을 실행한다. 같은 teacher가 쓴 test는 그 error를 공유할 수 있으므로 test source를 기록한다.
3. *Reference 없는 math와 logic.* Teacher solution 3개를 sample하고 final answer 중 적어도 2개가 일치할 때 row를 유지한다. Nemotron-PrismMath가 쓴 setting `N = 3, τ = 2`다([[prismatic-synthesis]] §3.1). Agreement는 ground truth가 아니다. 일관되게 틀린 teacher도 pass한다.

**근거.** APIGen three stage의 pass rate는 generator에 따라 다르다. DeepSeek-Coder-33B-Inst 34.42%, Mixtral-8x22B-Inst 65.96%, DeepSeek-V2-Chat 84.15%다(Table 1). Semantic failure를 되돌려 넣으면 BFCL은 xLAM-7B에서 −4.06, xLAM-1B에서 −9.59, execution failure를 되돌려 넣으면 −5.94 및 −12.17 바뀐다(Fig. 5). Release sample 600개를 human inspection한 결과 minor issue가 28개였다(App. A.3). Evaluation은 BFCL뿐이다(§5). Result(단일 연구).

**Verifier false negative.** String check는 correct answer를 reject할 수 있다. IFEval에는 “P.S. **I do like the cake**”로 끝나는 response가 “P.S. I do like the cake” exact match에 실패하기 때문에 loose metric이 있다. GPT-4의 prompt-level accuracy는 strict 76.89, loose 79.30이다([[ifeval]] §2.2, Table 3). Verifier별로 reject한 row 20개를 hand-label하고 false-reject count를 보고한다.

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

## §5 모든 evaluation set에 대한 contamination check

**정의.** Training row가 benchmark item 또는 model이 memory에서 answer를 재현할 수 있을 만큼 충분한 부분을 포함하면, 그 row는 해당 benchmark에 대해 contaminated됐다.

**문제.** Teacher model은 benchmark item을 재현할 수 있다. Tülu 3는 MATH overlap 때문에 NuminaMath-TIR의 11.3%, HumanEval overlap 때문에 Evol CodeAlpaca의 3.5%를 제거했다([[tulu-3-sft-and-eval]] Table 8).

**메커니즘(Tülu 3 rule, §3.2).**
1. Completion은 regenerate되므로 prompt(user turn)만 비교한다.
2. Test-item token이 그 token을 포함한 8-gram을 training row와 공유하면 match한다.
3. Test item token의 50% 초과가 match하면 test item은 training row와 overlap한다.
4. Tülu 3는 evaluation item의 2% 초과와 overlap하면 dataset을 contaminated로 취급한다. Lab은 dataset이 여럿이 아니라 pool 하나이므로 overlap row를 제거하고 benchmark별 count를 보고한다.

**계산 예.** Token 20개 test prompt가 training prompt와 contiguous token 12개 span 하나를 공유한다고 하자. Span의 모든 token은 shared 8-gram에 속하므로 token의 `12/20 = 60%`가 match하고 item은 overlap한다. Shared span이 token 7개라면 shared 8-gram이 없으므로 match는 0%다. Test item 1,319개인 GSM8K에서는 overlap item 27개에서 2% dataset rule을 넘는다.

**한계.** Tülu 3는 embedding matching이 distributional similarity와 구분하기 어렵다고 발견했다(§3.2). Paraphrase item은 n-gram check를 통과할 수 있다. Nemotron-PrismMath는 10-gram matching 뒤 LLM paraphrase detection을 추가한다([[prismatic-synthesis]] §3.1). 8-gram result를 primary number로, paraphrase check를 두 번째 column으로 보고한다.

**함의.** Knowledge probe를 포함해 §7의 모든 benchmark에 대해 check를 실행하고 `general-mix` arm에서도 실행한다.

## §6 Stage 4: matched size의 training arm

**Matched size.** 모든 arm은 정확히 N row로 학습한다. Training-set size의 효과는 setting에 의존한다. LIMA는 Stack Exchange training set을 두 배로 늘려도 response quality가 개선되지 않음을 발견했지만([[lima-baseline]] §5), Cherry LLM은 share 5%, 10%, 15%에서 서로 다른 LLaMA2 leaderboard average를 보고한다([[cherry-llm]] Table 3). 따라서 size가 다른 비교는 size effect와 filter effect를 분리할 수 없다.

**Baseline.** `base`는 forgetting을 측정한다. `general-mix`는 skill-specific submixture로 구축하고 Tülu 3 suite에 대해 decontaminate한 prompt 939,344개의 Tülu 3 SFT mixture에서 random subset을 사용한다([[tulu-3-sft-and-eval]] §3.2, §4.1.2). 그 Persona IF data는 IFEval의 constraint type 25개를 포괄하므로(§3.1.2), IFEval은 이 arm의 targeted column이다.

**Initial-loss check.** Training 전에 exact collator(template, masking, packing)로 first training batch에서 base model의 mean response-token loss `L_base`를 계산한다. Step-1 training loss는 dropout 및 precision 차이를 제외하고 `L_base`와 일치해야 한다. `ln V` 근처(`ln 128,000 = 11.76`) 값은 weight를 load하지 않았음을, `L_base`보다 훨씬 낮은 값은 예컨대 masking shift처럼 input에서 label이 보임을 뜻한다. Karpathy의 “verify loss @ init” check는 이 test의 from-scratch version이다([[karpathy-training-neural-net-recipe]]).

**Seed.** Arm마다 seed 3개를 학습한다. Tülu 3 8B SFT average는 seed 5개에서 59.8–60.1, 70B average는 seed 3개에서 70.0–72.6이었다([[tulu-3-sft-and-eval]] Table 14).

**Hyperparameter.** 모든 arm에 configuration 하나를 고정한다(Recipe section). Development benchmark만 사용해 `full` arm, seed 1에서 learning rate를 선택한다(§7).

## §7 Held-out multi-capability evaluation

**정의.** Held-out benchmark는 generation prompt, filter threshold, hyperparameter, checkpoint 선택에 쓰지 않은 benchmark다. Tülu 3는 development suite를 development 중 score를 보지 않는 unseen suite와 분리한다([[tulu-3-sft-and-eval]] §2.2).

**Suite.**

| Capability | Benchmark | Metric | Item(full / constrained path) | 역할 |
|---|---|---|---|---|
| instruction following | IFEval ([[ifeval]]) | prompt-level strict 및 loose accuracy | 541 / 541 | add-constraints operator가 target; 가능하면 IFBench([[ifbench]])도 보고 |
| knowledge 및 reasoning | MMLU-Pro([[mmlu-pro]]), discipline 14개로 stratify | accuracy, prompt format 하나 | 1,400 / 700 | held out |
| math | GSM8K | exact match | 1,319 / 1,319 | math row가 있으면 target |
| code | HumanEval | pass@1, greedy | 164 / 164 | code row가 있으면 target |
| reasoning | BBH | exact match | all task / task 10개 | held out |
| knowledge recall | PopQA 또는 다른 closed-book QA probe | exact match | lab 선택 | held out; forgetting signal |
| chat(optional) | AlpacaEval([[length-controlled-alpacaeval]]) | LC win rate, raw win rate, mean length | 805 / 805 | supplement만 |

Filter threshold와 learning rate는 development split에서 선택한다. Pool row 200개의 held-back set 및 위에 열거하지 않은 benchmark의 별도 slice다. Suite에서 final comparison을 읽는다.

**Instance-level noise.** n item에서 accuracy p의 standard error는 `SE = √(p(1 − p)/n)`이고 95% interval half-width는 `1.96 · SE`다.

| Benchmark | n | p | SE | 95% half-width |
|---|---|---|---|---|
| GSM8K | 1,319 | 0.40 | 0.0135 | ±2.6점 |
| IFEval | 541 | 0.40 | 0.0211 | ±4.1점 |
| HumanEval | 164 | 0.25 | 0.0338 | ±6.6점 |

Similar p에서 independent sample로 score한 two arm은 difference half-width가 `1.96 · √2 · SE`다. GSM8K에서 p=0.40이면 ±3.7점이다. Same item의 paired test는 더 좁으며 ch-51이 이를 도출한다.

**Checkpoint 및 seed noise.** 1B model final checkpoint 30개는 ARC Challenge accuracy에서 1.7% 범위를 보인다([[signal-and-noise-eval]] §3.1). 같은 decoding setting으로 각 seed의 final checkpoint를 평가한다.

**결정 rule.** Benchmark별로 seed 3개에 대한 mean과 range를 arm마다 보고한다. Benchmark에서 arm A를 arm B보다 낫다고 부르는 것은 mean 차이가 더 큰 seed range와 instance-level difference half-width를 모두 넘을 때뿐이다. Average 옆에 discipline별 MMLU-Pro 및 task별 BBH를 보고한다.

**Judge.** LLM judge를 쓸 때 raw win rate 및 mean output length와 함께 length-controlled win rate를 보고한다. Length control은 AlpacaEval과 Chatbot Arena의 Spearman correlation을 0.94에서 0.98로 높였고 gpt4_1106_preview의 verbosity-prompt range를 41.9%–51.6%로 좁혔다([[length-controlled-alpacaeval]] §4.1–4.2). Judge에는 position bias와 self-enhancement bias도 있다([[judge-llm-bias]]). Teacher model을 judge로 쓰지 않는다.

## §8 Memo와 acceptance criterion

한 page인 `synthetic-set-memo.md`:
1. **Survival 및 category table.** Stage별 및 category별 count, stage별 coverage, category-share 변화가 가장 큰 stage.
2. **Contamination.** `full` 및 `general-mix`의 benchmark별 overlap row와 제거한 것.
3. **Evaluation table.** Benchmark별 모든 arm을 `base` 및 `general-mix`와 비교: mean, seed 전반 range, instance-level half-width; judge를 사용했다면 LC win rate.
4. **Prediction log.** 각 ablation에 대해 training 전 작성한 predicted sign과 observed result.
5. Row id, 이를 잡았어야 하는 stage, 다음 measurement를 갖는 **failure mode 하나**.

Acceptance gate:
1. `survival.json` count가 매 stage에서 stage file의 row count와 같다.
2. 모든 stage에 category report가 있고 memo가 share가 5점보다 많이 바뀐 category를 명시한다.
3. `contamination.json`이 zero count를 포함해 suite의 모든 benchmark를 나열한다.
4. `full`의 모든 checkable row에 `verifier_verdict`가 있고 verifier별 reject row 20개를 hand-label한다.
5. 모든 run의 step-1 loss가 같은 batch의 `L_base`와 일치한다(§6).
6. 모든 arm에 full suite에서 seed 3개가 있고 memo의 어떤 claim도 §7 decision rule을 위반하지 않는다.

## Negative sample과 negative feedback

**어느 의미가 적용되는가.** 이 lab은 course standard 의미 1(negative marginal value)의 negative를 만든다. Format filter, dedup, IFD, verifier, contamination check가 reject한 row는 버린다. Negative를 content, conditioning, gradient로 사용하지 않는다. 그런 role의 rejected row 사용은 ch-31a와 ch-43a에서 다룬다.

**Negative의 출처와 label 방식.** Heuristic(format), MinHash similarity(duplicate; wrong이 아니라 redundant), model loss(IFD > 1 또는 low IFD), execution 및 LLM checker(tool call), answer agreement(math), n-gram overlap(contamination)이다. Report된 verifier error rate는 APIGen released data가 세 stage 뒤 sample 600개 중 minor issue 28개였고([[apigen]] App. A.3), IFEval strict check의 false negative는 loose check가 회복한다(GPT-4 prompt-level 76.89 대 79.30, [[ifeval]] Table 3).

**메커니즘: wrong row를 유지하면 해로운 이유.** SFT는 target token y에 대해 `−log p_y`를 최소화한다. Softmax logit z에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이므로 gradient step은 `z_y`를 높이고 각 다른 logit을 probability에 비례해 낮춘다. Step size 1, `p = (0.6, 0.3, 0.1)`에서 token 3이 wrong target인 계산 예는 `Δz = (−0.6, −0.3, +0.9)`이며 새 probability는 `(0.413, 0.279, 0.308)`이다. Wrong token은 0.10에서 0.31로 오르고 correct token 1은 0.19를 잃는다.

**근거.**
- Verifier failure를 positive로 다시 넣으면 BFCL은 7B에서 4.06–5.94점, 1B에서 9.59–12.17점 낮아졌다([[apigen]] Fig. 5). 작은 model이 더 잃었다.
- Lowest-IFD subset 학습은 Cherry LLM strategy 전체 중 가장 낮은 winning score를 냈다([[cherry-llm]] §4.2.3, Figure 4).
- 같은 size에서 filtered Stack Exchange data는 LIMA의 1–6 helpfulness scale에서 unfiltered보다 0.5점 높았다([[lima-baseline]] §5).
- 이 장의 source는 gain 중 negative 제거와 positive 선택에 각각 귀속되는 share를 측정하지 않는다.

**Control.** Reliable check가 있는 subset만 verify하고 나머지는 format·IFD 및 spot-check에 둔다. Reject row를 hand-label해 false reject를 추정한다. Later chapter가 content(APIGen은 relevance-detection refusal target 8,000개를 ordinary SFT target으로 사용, [[apigen]] App. B.3) 또는 dispreferred response로 쓸 수 있게 reason code와 함께 reject row를 보관한다.

**Diagnostic.** Stage × reason × category별 rejection count, verifier별 false-reject estimate, verifier 전후 checkable row share를 기록한다.

**Generality에 미치는 효과.** Checkable category에만 적용하는 verifier는 그 category의 row만 reject하므로 pass rate가 낮으면 final pool에서 share가 낮아진다. Stage별 checkable share를 보고 evaluation의 math·code column과 비교한다.

## 레시피

| Model(정확한 release) | 크기 | Stage | 설정 | 값 | Source 위치 | 상태 | 이 값의 근거 | Lab 값 | 차이 이유 |
|---|---|---|---|---|---|---|---|---|---|
| Self-Instruct data(GPT-3 davinci) | — | SFT data | seed task; in-context example; pool rule | 175; 8(6 seed + 2 generated); ROUGE-L < 0.7 | arXiv:2212.10560 §2.2 | verified 2026-09-15 | Table 2 quality review; 값의 ablation 없음 | same | 없음 |
| WizardLM-13b data | 13B | SFT data | evolver/responder; round M; operator choice | gpt-3.5-turbo; 4; prompt 6개 중 하나, equal probability | arXiv:2304.12244v3 §4.2 | verified 2026-09-14 | Fig. 5b round별 score(figure만) | lab teacher; M=2; same | budget; row별 기록 |
| WizardLM-13b data | 13B | SFT data | response sampling | temperature 1, max token 2048, top-p 0.9 | v3 §4.2 | verified 2026-09-14 | ablation 미보고 | same | 없음 |
| WizardLM-13b | 13B | SFT | optimizer; LR; batch; epoch | Adam; 2 × 10⁻⁵; GPU당 4, V100 8개; 3 | v3 §4.2 | verified 2026-09-14 | App. L Table 4: epoch 2.5/2.75/3에서 57.92/58.24/58.96 | 마지막 행 참조 | 다른 model size |
| Cherry LLM, LLaMA-7B | 7B | SFT data | pre-experience set; epoch | 1,000=K-means cluster 100×10; 1 | arXiv:2308.12032v5 §2.1 | verified 2026-09-14 | Fig. 5: 300 sample은 distinct gain, 500은 추가 gain 없음 | same | 없음 |
| Cherry LLM, LLaMA-7B | 7B | SFT data | selection rule; recommended share | IFD>1 drop, highest keep; top 10% | §2.2; App. G.2 | verified 2026-09-14 | App. B Table 5: 5%에서 IFD 52.06 대 random 50.61 | dedup pool 상위 15% | 5K target의 더 큰 share; `no-ifd` arm이 test |
| Cherry LLM, LLaMA-7B | 7B | SFT | LR; batch; epoch; max length(Alpaca) | 2 × 10⁻⁵; 128; 3; 512 | App. A; [[cherry-llm-recipe]] | verified 2026-09-14 | ablation 미보고 | 마지막 행 참조 | 다른 model size |
| Lee et al. NearDup(C4) | — | data dedup | shingle; hash; bucket×hash; threshold | space-tokenized 5-gram; 9,000; 450×20; Jaccard>0.8 및 edit similarity>0.8 | arXiv:2107.06499v2 §4.2, App. A | verified 2026-09-14 | App. A Fig.4: 0.8/0.9 histogram이 같은 지점에서 사라짐 | word 5-gram; 128; 16×8; estimated J≥0.8 | instruction-scale pool; MinHash estimate confirmation |
| APIGen generator | — | SFT data | temperature; generator당 target sample | 0.7; 40,000 | arXiv:2406.18518v1 §4.2 | verified 2026-09-14 | ablation 미보고 | tool row 0.7 | 없음 |
| APIGen verifier | — | SFT data | stage; execution timeout; checker model | format, execution, semantic; 미보고; 미보고 | §3.2 | verified 2026-09-14(stage); not reported(timeout, checker) | Fig.5 add-back ablation | same stage; 10s; generator 외 model | 실행에는 timeout 필요; teacher를 checker에서 제외 |
| xLAM-1B(FC) | 1.3B | SFT | peak LR; schedule; warmup; epoch; cutoff | 5 × 10⁻⁶; cosine; 50 step; 4; 2048 | App. B.3 | verified 2026-09-14 | ablation 미보고 | sweep의 candidate LR | lab model과 가장 가까운 reported size |
| Tülu 3 8B SFT | 8B | SFT | LR; schedule; warmup ratio; effective batch; max token; epoch; loss | 5 × 10⁻⁶; linear; 0.03; 128; 4,096; 2; token sum | arXiv:2411.15124 §4.3, Table 11 | verified 2026-09-15 | Fig.5–6: sum loss에서 5e-6 best, 2 epoch 초과 gain 없음 | batch128; warmup0.03; epoch2; mean loss; LR sweep | mean loss가 LR scale 변경 |
| Tülu 3 decontamination | — | eval-gate | match unit; item rule; dataset rule | prompt 8-gram; test token >50%; eval item >2% | §3.2 | verified 2026-09-15 | Table 8 removal share; threshold ablation 없음 | same item rule; row 제거 | single pool |
| Nemotron-PrismMath | — | distill-SFT data | answer agreement filter | solution N=3, τ=2 | arXiv:2505.20161v2 §3.1 | verified 2026-09-14 | ablation 미보고 | reference 없는 math row에 same | 없음 |
| LIMA 65B | 65B | SFT | example; epoch; LR; warmup; batch; checkpoint rule | 1,000;15;1e-5→1e-6 linear; 없음;32; epoch5–10 manual choice | arXiv:2305.11206 §3 | verified 2026-09-15 | 값의 ablation 없음 | 사용 안 함 | 다른 data size/selection protocol |
| DEITA-Mistral-7B 6K/10K | 7B | SFT | batch; epoch; LR; warmup; schedule | 512;6;2e-5;0.1;cosine | arXiv:2312.15685v2 App.A | verified 2026-09-14 | adequate training을 위해 epoch 증가, ablation 없음 | 사용 안 함 | 2 epoch는 DEITA 값이 아님을 보임 |
| Lab SFT(all arm) | 0.5B–1.5B | SFT | LR; schedule; warmup; batch; epoch; seed | 이 size/data의 source 보고 없음 | — | not reported | — | dev split에서 LR∈{5e-6,2e-5}; linear;0.03;sequence128;2;seed3 | LR 두 값이 xLAM-1B/Cherry LLM 행을 bracket |

2026-09-15 행은 Self-Instruct, LIMA, Tülu 3의 library card에 Verification section이 없어 primary PDF에서 읽었다. Chapter excerpt [[self-instruct-loop]], [[lima-baseline]], [[tulu-3-sft-and-eval]]가 extract를 보관한다.

**작은 general-purpose run의 시작점.** Synthetic row 수천 개에서 1B-class base model의 SFT라면, 8B model·prompt 939,344개·sum loss에서 Tülu 3가 고른 schedule shape, warmup ratio 0.03, effective batch 128, epoch 2를 verified setting으로 시작한다. Lab은 mean loss와 작은 model을 쓰므로 development benchmark만으로 xLAM-1B 값(5 × 10⁻⁶, 1.3B, cosine, 4 epoch)과 Cherry LLM 값(2 × 10⁻⁵, 7B, 3 epoch) 사이 two-point sweep에서 LR을 고른다.

## 일반화 관점

**(a) Breadth를 높이는 것.**
- Generation 때 near-copy를 막는 pool rule: Self-Instruct ROUGE-L<0.7; resulting instruction 52,445개에서 가장 common root verb 20개와 그 top object 4개가 14%를 차지한다. 논문은 rule을 ablation하지 않았다([[self-instruct-loop]] §2.2, §3.2).
- General mixture의 diverse real chat data: WildChat 제거는 Tülu 3 8B AlpacaEval 2를 12.4에서 7.5, average를 60.1에서 58.9로 낮췄다([[tulu-3-sft-and-eval]] Table 10).
- Fixed size/quality의 prompt diversity: heterogeneous Stack Exchange example 2,000개는 homogeneous wikiHow 2,000개보다 높았다([[lima-baseline]] §5).
- Fixed quality의 diversity: G-Vendi가 더 높은 10K subset은 OOD benchmark에서 G-Vendi가 낮은 100K subset보다 흔히 높았다([[prismatic-synthesis]] §2.3.1).

**(b) Narrowing 또는 forgetting을 일으키는 것.**
- Category를 옮기는 difficulty selection: IFD top 5%는 story/list generation이 주도했고 math·coding sub-category loss 및 LLaMA-7B MMLU 41.73→36.51이 있었다([[cherry-llm]] Table 1,4, App.C).
- Seed composition: ShareGPT seed는 GSM8k를 37.15→31.46으로 낮췄다([[evol-instruct]] Table 2).
- Benchmark와 맞는 constraint template: IFEval 25 constraint type에서 잘하는 model은 unseen constraint 58개에서 더 낮다([[ifbench]] §1, Fig.1).
- Judge가 보상하는 response length: verbosity prompt는 raw AlpacaEval win rate를 22.9%→64.3%로 옮겼다([[length-controlled-alpacaeval]] §4.1).
- Contaminated synthetic data: NuminaMath-TIR의 11.3%가 MATH와 overlap한다([[tulu-3-sft-and-eval]] Table 8).
- Evaluation family 하나: APIGen은 BFCL만 보고([[apigen]] §5), DEITA의 AlpacaEval·MT-Bench gain은 일관되지 않는다([[deita]] §3.3).

**(c) 이 stage에서 측정 방법.**
- Development/unseen split을 둔 programmatic suite에서 `base`와 `general-mix` 대비 capability별 delta([[tulu-3-sft-and-eval]] §2.2).
- Stage별 category share와 coverage score(§3.4).
- Benchmark별 8-gram prompt contamination 및 paraphrase check([[tulu-3-sft-and-eval]] §3.2; [[prismatic-synthesis]] §3.1).
- Seed 3개, seed range, instance-level interval(§7).
- 알려진 measurement error: n-gram check는 paraphrase를 놓치고, embedding diversity는 OOD accuracy와 반대 ranking을 낼 수 있으며([[prismatic-synthesis]] Table 2), strict string verifier에는 false negative가 있고([[ifeval]] §2.2), judge metric에는 length/self-enhancement bias가 있다([[judge-llm-bias]]).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| Stage product 없이 target에서 raw count 계획 | final pool이 target보다 훨씬 작음(옛 예는 5K target에 990 종료) | generation 전 `N_raw = N_target / Π s_i` 계산, `survival.json`과 비교 |
| Dedup 전 IFD 적용 | final count가 plan 아래, top-IFD row 안 duplicate cluster | IFD-kept set 안 MinHash cluster count |
| IFD definition 두 개 혼용 | code refactor 때 ranking 변화 | `s(A|Q)`와 `s(A)`를 별도 log, row100개에서 두 definition 재계산 |
| 두 IFD pass의 template 다름 | IFD>1 row 다수 | instruction 외 identical prefix로 row50개 rescore |
| IFD 뒤 average만 읽기 | judge score 상승, MMLU-Pro/rewrite-type prompt 하락 | stage별 category report, discipline별 MMLU-Pro |
| Teacher를 semantic checker/judge로 사용 | second checker가 재현 못하는 high pass/win rate | 다른 model로 row200개 re-check |
| Normalization 없는 string-match verifier | correct answer reject | verifier별 rejected row20개 hand-label |
| Synthetic data contamination check 생략 | held-out column gain 없이 GSM8K/HumanEval gain | benchmark별 8-gram overlap, paraphrase check와 비교 |
| Arm size 불균등 | row가 더 많은 arm이 모든 benchmark 승리 | arm마다 `len(train) == N` assert |
| Arm당 seed 하나 | rerun에서 arm ranking 변화 | seed3개, range 보고 |
| Pretrained base의 step-1 target으로 ln(V) | correct run flag, broken masking 미발견 | 같은 batch의 step-1 loss와 `L_base` 비교 |
| Raw judge win rate를 headline | mean output이 길며 win-rate gain | arm별 LC win rate와 mean length |
| Final suite에서 threshold tuning | suite score가 held out이 아님 | threshold choice 전 본 benchmark log |

## 이해도 점검

1. 옛 예는 IFD에서 15%를 keep하고 row 990개로 끝났다. Survival product를 써서 raw count를 올리는 것만 fix가 아닌 이유와 더 큰 IFD keep share가 category report를 어떻게 바꿀지 설명하라.
2. §3.3의 X,Y row는 loss-ratio와 perplexity-ratio IFD definition에서 order를 바꾼다. 각 formula의 어느 성질이 swap을 만들며 low/high answer loss 중 어느 row를 각 definition이 선호하는지 설명하라.
3. Alpaca IFD selection은 MMLU를 낮추고 AlpacaEval을 높였다. Cherry Table 4 category shift로 causal account를 주고 training 전 같은 effect를 발견할 lab measurement를 말하라.
4. APIGen 1B는 failed sample을 되돌린 때 7B보다 BFCL을 더 잃었다. Negative samples의 softmax gradient로 작은 model이 더 영향받을 이유를 제안하고, 뒷받침에 필요한 evidence를 말하라.
5. Lab band r=8,b=16에서 Jaccard 0.5 pair는 6.1%로 candidate가 된다. 이것이 pair 제거를 뜻하지 않는 이유와 confirmation step을 없애면 바뀔 것을 설명하라.
6. `general-mix`에는 IFEval constraint type을 target한 data가 있다. `full`이 IFEval이 아니라 IFBench unseen constraint에서 `general-mix`를 이기는 것이 더 강한 generality evidence인 이유를 설명하라.
7. HumanEval에서 arm 둘이 seed range 1점으로 3점 차이다. §7을 사용해 real이라 부를 수 있는지와 어느 noise source가 지배하는지 결정하라.
8. Embedding Vendi score는 dedup 뒤 오르는데 GSM8K는 떨어진다. Prismatic Synthesis finding으로 둘이 동시에 참일 수 있는 이유를 설명하라.

## 연결

- 이전: ch-28 — Long-Context Data Synthesis and Synthetic Evaluation Task Families.
- 다음: ch-29a — Long-Document Synthesis for Continued Pretraining and Long-Context SFT.
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix(this lab이 instantiate하는 pattern).
- ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing(Self-Instruct, Evol-Instruct).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection(IFD, DEITA, G-Vendi).
- ch-23 — Model Collapse and Verification of Synthetic Data; ch-26 — Tool and Function-Calling Data(APIGen).
- ch-12 — Deduplication: Exact, Near-Duplicate, and Semantic(MinHash, suffix array).
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks; ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate; ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares.
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split; ch-44 — Process Supervision and Verifiable Rewards.
- ch-47 — Evaluation Harness and Suite Design for General Capability; ch-48 — Contamination Detection and Its Effect on Reported Scores; ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting; ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions.

## 출처

- [[self-instruct-loop]] — arXiv PDF에서 읽은 Self-Instruct chapter excerpt: sampling, output-first branch, filter, Table 1–2. [[self-instruct]]는 library card(작성 시 검증하지 않았고 number에 사용하지 않음).
- [[evol-instruct]] — operator, prompt text, elimination rule, round, Table 1–2, recipe ledger.
- [[cherry-llm]], [[cherry-llm-recipe]] — IFD Eq. 3–5, pre-experience selection, Table 1/4/5, App.C, SFT setting.
- [[ifd-filter]] — chapter excerpt: IFD definition 둘, released script line, Superfiltering Table1 overlap. [[ifd]], [[superfiltering]]은 작성 시 검증하지 않은 library card.
- [[minhash-lsh]] — Jaccard resemblance, min-wise sketch estimator, banding formula pointer.
- [[deduplicating-training-data]] — NearDup parameter, C4 duplicate/memorization rate.
- [[apigen]] — three-stage verifier, pass rate, add-back ablation, xLAM-1B/7B SFT setting.
- [[lima-baseline]] — arXiv PDF에서 읽은 LIMA chapter excerpt: composition, training, §5 ablation. [[lima]]는 library card(작성 시 검증하지 않음).
- [[deita]] — DEITA model epoch, LIMA 1K baseline, benchmark disagreement.
- [[prismatic-synthesis]] — Vendi/G-Vendi definition, diversity–OOD correlation, answer-agreement filter, paraphrase decontamination.
- [[tulu-3-sft-and-eval]] — Tülu 3 report chapter excerpt: development/unseen suite, 8-gram decontamination, Table10 ablation, Table11 setting, Table14 seed. [[tulu-3-sft-mix]]는 dataset card(작성 시 검증하지 않음).
- [[ifeval]] — chapter excerpt: instruction type 25개, prompt 541개, strict/loose metric.
- [[ifbench]] — IFEval overfitting 확인용 unseen verifiable constraint.
- [[mmlu-pro]] — chapter excerpt: discipline 14개의 question 12,032개, option 10개, prompt sensitivity.
- [[length-controlled-alpacaeval]] — chapter excerpt: length term GLM, LC win rate, gameability/Arena correlation.
- [[signal-and-noise-eval]] — 1B checkpoint-to-checkpoint noise.
- [[judge-llm-bias]] — judge position, verbosity, self-enhancement bias.
- [[karpathy-training-neural-net-recipe]] — randomly initialized model의 “verify loss @ init” check.
