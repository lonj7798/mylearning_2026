<!-- chapter: ch-21
     track: synthetic
     title: Taxonomy-Driven and Textbook Synthesis
     sources: [[glan]], [[nemotron-4-synthetic]], [[phi-textbooks]], [[phi-1-5]], [[phi-3]], [[phi-4]], [[hf-cosmopedia]], [[mathscale]]
     figures: figures/taxonomy-expansion.html
-->

# 21장 — Taxonomy-Driven 및 Textbook Synthesis

> **핵심 통찰.** Self-Instruct의 175 task와 그 후손 pipeline 모두에서 그랬듯 seed pool이 bottleneck이라면, seed 자체를 **사람 지식의 pre-curated taxonomy**로 대체하고 teacher가 top-down으로 확장하게 할 수 있다. taxonomy가 곧 diversity prior다. curator가 discipline, subject, concept로 쪼개 놓은 것이 downstream model이 배우는 것이고, curator가 빠뜨린 것은 student에게 보이지 않는다.
>
> **가이드라인.** 새로운 capability를 위해 먼저 결정하라. bottom-up(seed + bootstrap + filter) pipeline이 충분한 coverage를 주는가, 아니면 seed가 절대 sample하지 않는 branch가 있는가? 후자라면 ≥ 3 level의 taxonomy를 만들고, 각 leaf를 concrete unit(syllabus session, topic page, exercise type)으로 확장한 뒤 leaf에서 generation하라. curator-audit cost를 명시적으로 budget하라. tree의 상위 두 level이 final corpus의 bias를 지배한다.

---

## 이 장이 필요한 이유

Ch-18은 generate → filter → dedup → verify → select → mix loop를 제공했다. Ch-19는 *어떻게* generate할지(bootstrap, evol, rephrase, persona)를 survey했다. Ch-20은 distillation-as-data를 다뤘다. 이 모든 것은 real prompt, real problem, real user log 같은 seed를 entry point로 가정한다.

이 장은 seed가 없거나 seed가 너무 좁을 때 무슨 일이 일어나는지 다룬다. 2023년부터 2025년 사이 세 work line이 같은 답으로 수렴했다.

1. **Phi (Microsoft, 2023→)** — noisy web을 curated topic list로 바꾸고, 각 topic 주변에 "textbook-quality" prose를 synthesize해 *pretraining* data로 사용한다. 1.3B model이 code와 reasoning에서 10배 큰 model과 겨룬다.
2. **GLAN (Microsoft, 2024)** — 같은 아이디어를 instruction data에 적용한다. 사람 지식을 Field → Subfield → Discipline → Subject → Session → Concept tree로 decompose하고, leaf concept마다 instruction을 생성한다. seed example이 필요 없다.
3. **Nemotron-4 340B (NVIDIA, 2024)** — task-family category 위에서 generator/critic loop를 실행하고, model 자신의 reward head를 judge로 사용해 alignment corpus의 >98%를 synthesize한다.

통합 abstraction은 **top-down synthesis**다. structure는 사람 또는 GPT-4가 curate한 category system에서 오고, teacher가 각 cell을 채운다. 이 장은 그 abstraction을 다룬다. 무엇을 사게 해 주는지, 비용이 무엇인지, 어디를 조용히 놓치는지.

Companion **[figures/taxonomy-expansion.html](figures/taxonomy-expansion.html)** 는 live tree-expander다. 5개의 root discipline으로 시작해 "expand"를 click하면 fan-out이 depth와 branching-factor hyperparameter에 의해 곱해지는 것을 볼 수 있다. 이 장에서 tree-size arithmetic이 가장 중요한 operational fact인데, prose만으로는 internalize하기 어렵기 때문에 존재한다.

---

## 1. Bottom-up vs top-down

Self-Instruct(ch-18)는 175개 seed task로 시작해 teacher에게 "more tasks like these"를 prompt하고 ROUGE-L로 filter한다. 결과 52K instruction은 seed pool의 neighborhood에 있다. 새로운 instance지만, 사람이 이미 적어 둔 pattern의 작은 perturbation이다. Taxonomy-driven synthesis는 이를 뒤집는다. seed pool 없이, space의 *coordinate system*을 적어 두고 teacher가 각 cell 안에서 instance를 작성한다.

| | Bottom-up (Self-Instruct, Evol-Instruct) | Top-down (GLAN, Phi-1.5, Cosmopedia) |
|---|---|---|
| Entry point | 175–1000 seed instances | 5–50 root categories |
| Diversity source | teacher output distribution을 통한 random walk | curated tree를 가로지르는 structural walk |
| Coverage guarantee | statistical | constructive (모든 leaf가 한 번 생성됨) |
| Blindspot | seed가 touch하지 않는 branch | curator가 그리지 않은 branch |
| Curator work | seed labeling | tree building + auditing |
| What scales | generation-side (seed당 sample 수 증가) | tree-side (branch 추가, depth 증가) |

둘 다 imagination-free가 아니다. 둘 다 curator의 세계관을 상속한다. 다만 그 상속의 locus가 이동한다. seed pool에서는 bias가 diffuse하다. taxonomy에서는 소수의 label이 그 아래 모든 것을 결정하는 tree의 첫 두 level에 bias가 집중된다.

---

## 2. GLAN — 전체 design document로서의 taxonomy

[[glan]]("Generalized Instruction Tuning via Taxonomy-Driven Synthesis", Microsoft 2024)은 이 pattern을 가장 명확하게 말한다. 그들의 opening move:

> 이전 instruction-tuning data synthesis(Self-Instruct, Evol-Instruct)는 작은 real instruction pool로 seed되고 그 bias를 상속한다. GLAN은 seed를 **taxonomy**로 대체한다. 사람 지식을 field → subfield → discipline으로 semi-automatically decompose한다. 각 discipline은 auto-generated subject list를 얻고, 각 subject는 class session syllabus를 얻으며, 각 class session은 concept list로 enumerate된다. Instruction은 concept level에서 생성되어 모든 branch에 대한 coverage를 guarantee한다.

실제 tree는 여섯 level이다.

```
Field      (e.g. Mathematics, Computer Science, Humanities, Natural Sciences)
  Subfield  (e.g. Algebra, Number Theory, Topology under Mathematics)
    Discipline  (e.g. Linear Algebra under Algebra)
      Subject   (e.g. Eigenvalues and Eigenvectors)
        Session (one class in the subject's syllabus, with learning objectives)
          Concept (a single pedagogical unit inside the session)
            Instruction  ← generation happens here
```

상위 두 level은 저자와 GPT-4가 hand-curate한다. 그 아래 모든 level은 parent name과 "list the children"을 GPT-4에 prompt해 auto-generated된다. 즉 사람 지식의 *shape*(level 0의 Math vs CS vs Humanities, level 1의 Algebra vs Topology)는 사람의 design choice이고, 각 shape 내부의 *density*는 GPT-4의 것이다.

Full expansion의 pseudocode:

```python
def glan_expand(root_fields, teacher, depth_budget):
    """Taxonomy-driven top-down synthesis (GLAN §3)."""
    # Level 0: Fields (hand-curated, ~36 entries)
    fields = root_fields

    # Levels 1-2: Subfields -> Disciplines (GPT-4 decomposes; authors audit)
    disciplines = []
    for f in fields:
        subfields = teacher.ask(f"List the major subfields of {f}.")
        for sf in subfields:
            disciplines += teacher.ask(f"List the disciplines within {sf}.")

    # Level 3: Subjects (GPT-4)
    subjects = []
    for d in disciplines:
        subjects += teacher.ask(f"List comprehensive subjects taught in {d}.")

    # Level 4: Syllabus sessions per subject
    sessions = []
    for s in subjects:
        sessions += teacher.ask(
            f"Write a multi-class syllabus for {s} with explicit learning objectives.")

    # Level 5: Concepts per session
    concepts = []
    for sess in sessions:
        concepts += teacher.ask(f"List the key concepts covered in {sess}.")

    # Level 6: Instructions (generation at the leaf)
    corpus = []
    for c in concepts:
        # Varied difficulty; verifier-friendly formats for math/code
        for difficulty in ["easy", "medium", "hard"]:
            instr, response = teacher.generate_instance(concept=c, difficulty=difficulty)
            corpus.append((instr, response))
    return corpus
```

branching factor, 즉 field당 discipline 수, discipline당 subject 수, session당 concept 수는 hyperparameter다. GLAN의 published setting에서 final corpus는 "모든 leaf concept를 포괄하는 multi-million instruction-response pair"다. Interactive companion은 이 숫자를 직접 조작하게 해 준다. leaf count가 ~5K(shallow, narrow)에서 ~5M(deep, wide)으로 올라가는 것을 보는 것이 핵심이다.

**taxonomy가 사준 것.** GLAN data로 fine-tune한 Mistral-7B는 같은 base에서 Alpaca, WizardLM, CodeAlpaca training을 MATH, GSM8K, HumanEval, MBPP, BBH, ARC, MMLU 전반에서 이긴다. *task-specific real data 없이* 말이다. 새로운 capability를 추가하는 것은 subtree를 추가하는 것만큼 싸다.

**사주지 못한 것.** GLAN은 teacher에게 없는 skill을 만들지 않는다. GPT-4의 implicit curriculum을 explicit tree로 뽑아 Mistral-7B에게 넘겨준다. ceiling은 모든 distillation과 마찬가지로 teacher다.

---

## 3. Nemotron-4 340B — shallow taxonomy로서의 generator/critic

[[nemotron-4-synthetic]]은 다른 단면을 취한다. NVIDIA는 *content*의 deep tree 대신 *task family*(coding, general QA, topic-following, document-based reasoning, function calling, refusal)의 shallow tree를 curate하고 generator/critic loop가 heavy lifting을 하게 했다. 두 가지가 load-bearing이다.

**RM은 filter이자 judge다.** ground truth가 없을 때(open-ended QA, dialogue), Nemotron-4-340B-Reward가 어떤 generator sample이 SFT set에 들어갈지, 어떤 pair가 preference set에 들어갈지 고른다. *같은 lab*이 generator를 training하고, 20K human HelpSteer2 anchor로 RM을 training하며, 그 RM을 다음 iteration을 govern하는 데 사용한다. Anchor-set math:

| Data slice | Scale | Human-labeled |
|---|---|---|
| SFT (code) | ~800K | No — RM-judged |
| SFT (general) | ~200K | No — RM-judged |
| DPO pairs | ~160K | No — RM-judged |
| RPO pairs | ~300K | No — RM-judged |
| HelpSteer2 (RM training) | ~10K | Yes |
| SFT human anchors | ~10K | Yes |
| **Total** | **~1.48M** | **~1.3% human** |

논문은 직접 말한다. "alignment training data의 98% 이상은 synthetic"이다. human slice가 reward model을 training하고, 나머지는 synthetic candidate에 대한 RM의 verdict다.

**Generator/critic loop in pseudocode.**

```python
def nemotron_iteration(policy_ckpt, rm, task_families, K=4):
    sft_corpus, dpo_corpus = [], []
    for family in task_families:
        for p in synth_prompts(policy_ckpt, family):
            candidates = [policy_ckpt.generate(p) for _ in range(K)]
            scored = sorted(((c, rm.score(p, c)) for c in candidates), key=lambda x: -x[1])
            sft_corpus.append((p, scored[0][0]))                       # top as SFT target
            dpo_corpus.append((p, scored[0][0], scored[-1][0]))        # (best, worst) pair
    return sft_corpus, dpo_corpus
```

code에 대해 NVIDIA는 `synth_prompts`를 **Genetic Instruct**로 대체한다. 작은 code pool로 seed된 self-instruct + WizardCoder mutation + LLM fitness function이다. 같은 tree shape이지만 family 내부에 genetic fan-out이 들어간다.

**Staged SFT** — code SFT first, general SFT second — 는 Phi-4, Qwen, Tülu에서도 나타난다. code data는 더 깨끗한 correctness signal을 가진다. 먼저 code data로 training하면 structured-format following을 설치하고, 그 위의 general-SFT가 이를 손상시키지 않는다.

---

## 4. Phi line — pretraining scale의 textbook synthesis

Phi paper들은 같은 아이디어를 stack의 한 level 앞, 즉 instruction data가 아니라 **pretraining data**로 밀어 넣었다. 첫 paper 제목 [[phi-textbooks]]에 명시된 thesis: "Textbooks Are All You Need."

> Pretraining loss는 *quantity*-bounded가 되기 훨씬 전에 *quality*-bounded가 된다. noisy web crawl을 작은 "textbook-quality" corpus(filtered real + GPT-3.5-synthesized textbooks + exercises)로 바꾸면, ~100배 많은 token으로 training한 ~10배 큰 model과 경쟁하는 1.3B model을 얻는다.

네 generation, 같은 core move, 점진적으로 broad해짐:

| Model | Year | Params | Pretrain tokens | Textbook/synthetic tokens | Web tokens | Code tokens | Notes |
|---|---|---|---|---|---|---|---|
| Phi-1 | 2023 | 1.3B | ~7B | ~1B synthetic (Python textbooks) + ~180M synthetic exercises (SFT) | ~6B filtered Stack / StackOverflow | (subset of the above) | The Stack에 "textbook-likeness" classifier를 처음 사용 |
| Phi-1.5 | 2023 | 1.3B | ~27B | ~20B synthetic (20K-topic taxonomy, GPT-3.5) | optional 'phi-1.5-web' ablation | inherited from phi-1 | recipe를 code → common-sense로 확장 |
| Phi-3-mini | 2024 | 3.8B | 3.3T | Phase-2 majority (exact split undisclosed, ~trillion-scale synthetic) | Phase-1 majority (filtered web) | mixed into both phases | Two-phase curriculum, GPT-4-class teachers |
| Phi-4 | 2024 | 14B | ~unspecified, synthetic weighted ~10% (≈400B unweighted) across 50 categories | ~400B unweighted | filtered web (rest of budget) | part of the 50 categories | 50 synthetic types; pivotal-token DPO |

세 fact가 table을 고정한다.

- **Classifier가 먼저 온다.** Phi-1은 hand-labeled "educational vs non-educational" seed로 random-forest를 training했다(작은 LM embedding의 feature 사용). The Stack을 score하고 top slice를 6B filtered-web core로 keep했다. 나머지는 classifier가 찾지 못한 부분 주변의 synthesis다.
- **Phi-1.5가 taxonomy를 formalize했다.** [[phi-1-5]]에서: common-sense, grade-school science, logic, everyday reasoning을 포괄하는 "carefully curated list of 20,000 topics"가 있고, 각각은 GPT-3.5로 확장된다. hierarchical은 아니고 flat이지만, pretraining을 위한 첫 공개 "taxonomy-as-prior"다.
- **Phi-4는 depth가 아니라 category를 scale했다.** Phi-4의 ~400B unweighted synthetic token은 "50 types"에 걸쳐 있다. hierarchy가 아니라 category index다. 각 category 내부의 depth는 GPT-4-class teacher와 rejection sampling에 위임된다. 그 위에 Phi-4-reasoning은 1.4M o3-mini trace(~16B SFT tokens)를 distill했고, competitive AIME에 도달하는 데 **90 GRPO steps**만 필요했다. 이 숫자가 Phi philosophy를 가장 날카롭게 요약한다. *gain은 전부 SFT data에 있고, RL은 polish다*.

**TBAYN quote와 그 limitation.** "Textbooks Are All You Need"는 technical claim이라기보다 marketing에 가깝다. Phi-1이 실제로 보인 것: curated + synthesized code data 약 7B token으로 1.3B model을 training해 HumanEval 50.6% / MBPP 55.5%에 도달했다. ablation에서 filtered + synthetic > filtered-only > raw web. Caveat:

- **Contamination:** 이후 Phi line 분석은 synthetic exercise와 HumanEval prompt 사이의 non-trivial overlap을 flag했다. 50.6%는 의심받고 있다.
- **Narrow domain:** Phi-1은 Python이다. Phi-1.5는 reasoning으로 확장했지만, 이후 standard가 된 더 어려운 benchmark(MMLU-Pro, MATH)에서는 가장 약하다.
- **Closed data:** Phi corpus는 release된 적이 없다. [[hf-cosmopedia]]가 open reproduction이며, 그 1B validation model은 여러 task에서 Phi-1.5보다 뒤진다.

정직한 reading: curated + synthetic은 작은 model을 *training data와 닮은 benchmark에서* competitive하게 만들기에 충분하다. generalization claim은 더 약하다. contamination은 가장 큰 unresolved critique다.

---

## 5. Taxonomy는 어디에서 오는가 — 숨은 design choice

§2–4의 모든 pipeline에는 taxonomy가 있다. 그것은 어디에서 오는가?

| Pipeline | Taxonomy source | Levels | Curator |
|---|---|---|---|
| GLAN | Hand-curated root fields + GPT-4 auto-expansion below | 6 | Authors + GPT-4 |
| Phi-1.5 | Hand-curated list of "20,000 topics" (flat) | 1 | Microsoft team |
| Phi-3 | Not disclosed — "reviewed by researchers" | ? | Microsoft team |
| Phi-4 | "50 synthetic categories" list | 1 | Microsoft team |
| Nemotron-4 | ~6 task families (hand-picked) | 1 | NVIDIA team |
| Cosmopedia | Mixed: Stanford outlines + Khan + OpenStax + WikiHow + 145 web clusters + UltraChat/OpenHermes seeds | 2–3 | Hugging Face team |
| MathScale | seed problem에서 GPT-3.5 concept mining으로 *extracted* graph | 2 (topic × knowledge-point) | Seed-derived, not pre-curated |

일곱 중 다섯은 top에서 hand-curated다. 이것이 **curator-bias-injection point**다. Microsoft researcher가 "field of knowledge" 또는 "synthetic category"라고 여기는 것이 multi-million-sample corpus의 top-level coordinate system이 된다. curator의 개인 frame이 예를 들어 engineering ethics, rhetoric, 특정 applied CS subfield를 빠뜨리면, trained model은 curator의 frame 안에서는 fluent하고 바깥에서는 blind해진다.

이는 taxonomy-driven method에서 *by construction* 피할 수 없다. coverage를 constructive하게 만들고 싶어서 taxonomy를 요청했고, constructive coverage에는 construction이 필요하기 때문이다. 가능한 alternative는 세 가지뿐이다.

- **Taxonomy를 curator가 아니라 data에서 extract한다.** MathScale이 이 방법이다. GPT-3.5에 각 seed problem의 concept를 naming하게 하고, graph로 aggregate한 뒤 edge를 sample한다. taxonomy는 seed의 bias를 상속하지만 curator의 personal prior는 피한다.
- **Crowd-curate.** 위 paper 중에는 쓰이지 않았다. single-curator bias를 줄일 수 있지만 자기 demographic bias가 있다.
- **Audit post-hoc.** Cosmopedia의 145-cluster web-sourced branch는 audit move다. curated branch에 "web이 말하는 것"을 supplement해 curator omission을 잡는다. HF team은 이것이 실제 gap을 잡는다고 주장한다. 동시에 deduplication workload를 두 배로 만든다.

**[[hf-cosmopedia]]의 practical rule:** *"prompts must be rewritten carefully to avoid near-duplicate generations."* prompt의 cosmetic variation, 예컨대 "explain X to a child" vs "explain X to an adult", 만으로는 부족하다. real taxonomy에서 오는 prompt family의 structural variation(audience × format × context)이 필요하다. Cosmopedia team은 첫 pass에서 너무 많은 duplicate-class output을 가진 30M prompt를 만들고 나서 이를 어렵게 배웠다. fix는 post-hoc filtering을 더 세게 하는 것이 아니라 prompt taxonomy를 restructure하는 것이었다.

---

## 6. Coverage vs imagination — bottom line

긴장을 blunt하게 말하면 이렇다.

**Top-down은 coverage를 준다.** tree의 모든 leaf가 하나 또는 k개의 sample을 얻는다. curator가 이름 붙인 모든 discipline이 local branching factor에 비례해 corpus에 등장한다. seed-based method는 이 guarantee를 제공할 수 없다.

**Bottom-up은 imagination을 준다.** seed pool에는 curator가 생각해 본 적 없는 combination이 들어 있다. 예를 들어 "explain how eigendecomposition relates to principal component analysis in a fraud-detection context for a non-technical manager"라고 묻는 user는 GLAN tree의 네 branch를 가로지르는 leaf를 만들고 있다. 어떤 tree expansion도 자연스럽게 그 combination을 emit하지 않는다. teacher가 이를 찾으려면 level들의 cross-product를 walk해야 하는데, cross-product는 enumerate하기에 combinatorially too large하다.

Companion HTML은 이를 보이게 한다. GLAN-style tree를 expand하면, leaf count가 curator가 그린 axis를 따라 곱해지는 것이 보이고, tree가 cover하지 못한 combination은 branch 사이의 white space로 보인다. white space가 seed-based method가 우연히 gold를 찾는 곳이다.

**2025년 pipeline의 practical answer.** 섞어라. Tülu 3(ch-33), Phi-4 mixture, Nemotron의 RM-governed pipeline, Cosmopedia는 모두 둘을 layer한다.

1. Taxonomy-driven core(coverage guarantee, curator-audited).
2. Seed-based 또는 user-log-derived supplement(imagination, combination capture).
3. Dedup+verify layer(ch-18)로 overlap을 collapse하고 각자의 장점을 keep.

이 장의 세 running example(GLAN, Nemotron, Phi)은 모두 첫 번째 layer다. capability coverage에는 *necessary*하다. curator가 예상하지 못한 combination으로 generalization하는 데는 sufficient하지 않다. Ch-22(quality, diversity, gradient-based selection)와 ch-23(model collapse, verification)은 두 번째 benefit을 stack으로 끌어오는 layer를 다룬다.

---

## 7. Operational cheat-sheet

**Taxonomy를 만들 때.**
- target capability가 broad하다(general instruction-following, general pretraining).
- 기존 seed data의 bias가 알려져 있다(예: 전부 ShareGPT → 전부 English, 전부 casual).
- auditable coverage claim이 필요하다("mathematics의 모든 discipline이 represented").
- 새로운 capability를 incremental하게 추가할 계획이다(subtree 추가가 싸다).

**하지 말아야 할 때.**
- target capability가 narrow하고 좋은 seed pool이 있다(code completion, SQL generation).
- teacher가 depth의 leaf를 안정적으로 produce하기에 너무 약하다.
- 상위 두 level의 audit budget이 expert time 1주 미만이다. bias가 severe해진다.

**Scaling rules of thumb.**
- GLAN shape: 30–50 root fields × ~10 subfields × ~10 disciplines × ~20 subjects × ~10 sessions × ~10 concepts × ~10 instructions ≈ 3M–30M samples. wider shallower tree는 depth를 under-cover하고, deeper narrower tree는 breadth를 under-cover한다. interactive companion에서 직접 조작할 수 있다.
- Phi-1.5 shape: 20K flat topics × 5–10 passes each ≈ 100K–200K documents × ~100K tokens/document ≈ 10B–20B tokens. flat topic list는 tree보다 단순하며 teacher(2023년 GPT-3.5)가 이미 strong curriculum-by-default writer일 때 adequate하다.
- Nemotron shape: 6 families × ~million-scale generations each × RM-filter ≈ 1M-scale SFT + 500K-scale preference. RM이 cost driver다.
- Cosmopedia shape: 30M prompts × ~1K tokens output ≈ 25B tokens. Deduplication이 진짜 bottleneck이다. upfront로 계획하라.

**Red flags.**
- trained model이 처음 몇 benchmark에서는 훌륭한데, orthogonal benchmark 하나에서 이상하게 plateau한다. → 그 benchmark가 그리지 않은 tree branch를 probe한다. hyperparameter를 tune하지 말고 taxonomy를 audit하라.
- corpus가 > 40% deduplicate된다. → structural variation이 아니라 cosmetic variation이다. prompt-family structure를 다시 써라.
- teacher가 student와 같은 model family다. → recursive distillation을 하고 있다. 위험은 ch-23에 있다.
- taxonomy를 한 사람이 썼다. → corpus는 그 한 사람의 mental model을 10M-sample scale로 확대한다. 최소한 두 명 이상의 curator가 level 0과 1을 audit하게 하라.

---

## Connections and what's next

- **[[ch-18]]** — taxonomy-driven은 generate → filter → dedup → verify loop의 한 operating mode다. 이 장은 generator step을 specialize했다. RM-as-judge(Nemotron)는 filter+verify specialization이다.
- **[[ch-19]]** — bootstrap / evol / rephrase / persona는 bottom-up alternative다. Persona-hub가 가장 가까운 cousin이다. topic의 taxonomy가 아니라 *asker의* taxonomy다.
- **[[ch-20]]** — R1-Distill lineage는 trace distillation이다. Phi-4-reasoning은 textbook-synthesis-plus-distillation hybrid다.
- **[[ch-22]]** — taxonomy-core + seed-supplement를 corpus size 폭발 없이 combine하는 방법이 quality/diversity/gradient-based selection이다.
- **[[ch-23]]** — model collapse와 recursive-training risk. Phi line의 defense(single-shot teacher, no recursive self-distillation)가 하나의 답이다.
- **[[ch-32]] / [[ch-34]]** — SFT recipes section에서 Phi-3/4 case study, 다른 case로 Nemotron-Ultra.

## Further reading

- [[glan]] — six-level taxonomy와 branching-factor table의 full extract.
- [[phi-textbooks]] — Phi-1 paper, textbook-quality classifier, TBAYN origin.
- [[phi-1-5]] — taxonomy precursor로서의 20K-topic list.
- [[nemotron-4-synthetic]] — generator/critic details + Genetic Instruct.
- [[hf-cosmopedia]] — open replication. dedup-is-the-bottleneck lesson을 위해 읽어라.
- [[mathscale]] — domain-specific seeded taxonomy as a contrasting approach.

## Companion visualization

**[figures/taxonomy-expansion.html](figures/taxonomy-expansion.html)** — interactive tree expander. 5개의 root discipline으로 시작해 "expand"를 click하면 three levels(subjects → lessons → questions)로 fan-out된다. depth와 branching-factor slider를 움직여 leaf count가 증가하는 것을 보라. "white space" visualization은 tree가 cover하지 않은 combination을 overlay한다. bottom-up imagination-vs-coverage tradeoff를 concrete하게 만든다.
