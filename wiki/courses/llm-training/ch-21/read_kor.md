<!-- chapter: ch-21
     track: synthetic
     kind: content
     title: Taxonomy-Driven and Textbook-Style Synthesis
     deps: [ch-20]
     sources: [[glan]], [[nemotron-4-synthetic]], [[nemotron-4-synthetic-recipe]], [[phi-textbooks]], [[phi-1-5]], [[phi-3]], [[phi-4]], [[cosmopedia]], [[hf-cosmopedia]], [[gsm1k]], [[fineweb]], [[mathscale]], [[persona-hub]], [[metamath]]
     figures: figures/taxonomy-expansion.html
     revised: 2026-09 (generality revision)
-->

# Chapter 21 — Taxonomy 기반 및 Textbook-style 합성

> **핵심 통찰.** Taxonomy 기반 합성은 example을 생성하기 전에 topic 목록을 고정하고, textbook-style 합성은 생성 text의 형태를 고정한다. GLAN은 사람이 검증한 discipline 126개에서 question-answer pair 1,000만 개를 생성했고 benchmark training data 없이 Mistral 7B의 GSM8K를 43.4에서 80.8로, ARC-C를 53.9에서 81.1로 올렸다. 그러나 같은 모델의 MMLU humanities는 base보다 낮은 54.9 대 56.5였다([[glan]] Tables 1–2). Phi report는 pretraining 규모에서 관련된 분리를 보인다. web data 없이 synthetic data로 학습한 13B 모델은 reasoning·code benchmark에서는 얻고 phi-3-medium 대비 TriviaQA 14.8점을 잃었다([[phi-4]] Table 3). 새로 작성한 GSM1k problem에서 Phi 및 Mistral family는 거의 모든 release와 size에서 GSM8K accuracy보다 낮다([[gsm1k]] §5.1).
>
> **지침.** 모델이 수십 domain(GLAN: discipline 126개)을 다뤄야 하고 그 일부에 seed prompt가 없다면, 사람이 검증한 topic list에서 생성하고 concept의 조합을 sample하라. GLAN의 math, code, BBH, ARC gain은 task-specific data 없이 나왔기 때문이다(Table 1). 이런 run에서는 category 및 discipline별 결과를 보고하라. MMLU aggregate +0.6점이 humanities −1.6점을 가렸기 때문이다(Table 2). textbook-style synthetic data가 pretraining token의 절반 이상(Phi-4: 최소 55%)이면 filtered web 및 knowledge-heavy source를 mixture에 남겨라. Phi-4 7B의 1T-token ablation에서 synthetic-plus-web allocation의 최대 gain은 TriviaQA +6.9였지만 synthetic-heavy allocation은 TriviaQA 3.0점을 잃었으며 저자들도 knowledge benchmark를 위해 web data를 넣었기 때문이다(§3.2, Table 4). benchmark가 synthetic data와 닮았다면 generality를 주장하기 전 distribution-matched fresh set에서 평가하라. phi-1은 exercise에 close match가 있는 HumanEval problem의 81.7%를 풀었지만 없는 problem은 26.9%만 풀었다([[phi-textbooks]] Table 3).

## 학습한 버전에 대한 정정

1. “Field → Subfield → Discipline → Subject → Session → Concept tree로 human knowledge를 분해하고 leaf concept마다 instruction을 생성”, “Instruction은 concept level에서 생성되어 모든 branch의 coverage를 보장한다” → GLAN은 syllabus에서 class session 하나 또는 둘과 key concept 하나부터 다섯을 sample해 GPT-4에 homework question을 요청한다. 인용문은 논문에 없으며 저자는 class session과 key concept을 무작위로 조합해 coverage와 diversity를 “기대한다”고 말한다([[glan]] §2, §2.4).
2. “상위 두 level은 저자와 GPT-4가 hand-curate하고 아래 level은 자동 생성”, “Level 0 fields(hand-curated, 약 36개)” → GPT-4가 field, sub-field, discipline을 생성하고 human annotator가 element의 유지·제거를 투표해 discipline 126개를 남긴다. field 수는 보고되지 않았다. subject는 discipline마다 education-expert prompt를 10회 질의해 100–200개를 얻고, syllabus는 subject마다 한 번 질의해 session 10–30개와 key concept 각각 약 다섯 개를 얻는다([[glan]] §2.1–2.3, §3.1).
3. difficulty `easy`/`medium`/`hard` loop 및 math/code의 verifier-friendly format → difficulty loop와 verification은 없다. 더 어려운 question은 two-session sampling에서 나오며 answer는 GPT-3.5-turbo가 별도 call에서 쓴다([[glan]] §2.4, §3.1).
4. “모든 leaf concept에 걸친 multi-million instruction-response pair” → 1,000만 pair이며 sampled concept combination 수는 보고하지 않는다([[glan]] §3.1).
5. “GLAN Mistral-7B가 같은 base의 Alpaca·WizardLM·CodeAlpaca를 MATH, GSM8K, HumanEval, MBPP, BBH, ARC, MMLU 전체에서 이김” → baseline은 released model이다. WizardMath v1.1은 GSM8K 83.2 대 80.8, MATH 33.0 대 32.7, HumanEval 51.2 대 48.8로 더 높고, GLAN은 MMLU humanities·social sciences·other에서 base Mistral보다 낮다([[glan]] Tables 1–2).
6. “새 capability 추가는 subtree 추가만큼 싸다” → 논문은 dataset을 재생성하지 않고 새 node를 추가할 수 있다고만 말하며 added node가 capability를 더한다는 시험은 없다([[glan]] §1, Verification).
7. Nemotron-4의 얕은 task-family tree(coding, general QA, topic-following, document reasoning, function calling, refusal) → open Q&A, writing, closed Q&A, math·coding에 대해 synthetic prompt를 생성하고 seed는 topic 3K, Python keyword 12K, math keyword 17K다. topic-following·document QA·function-calling은 기존 dataset에서 온다([[nemotron-4-synthetic]] §3.2.1, §3.2.5).
8. “model 자신의 reward head를 judge로 사용”, “RM을 human HelpSteer2 anchor 20K로 학습” → Nemotron-4-340B-Reward는 base 위에 five-attribute regression head를 더한 별도 model이며 HelpSteer2 example 10K로 학습한다. 나머지 human 10K는 SFT data다. chosen/rejected label은 가능한 경우 ground truth·verifier, 초기 iteration에서는 LLM-as-judge, 이후에는 reward model에서 온다([[nemotron-4-synthetic]] §3.1, §3.2, §3.2.3).
9. code SFT·DPO·RPO data를 “RM-judged”, “human 약 1.3%”로 표시한 표와 K=4 best/worst `nemotron_iteration` loop → Code SFT data는 LLM fitness check가 있는 Genetic Instruct에서 오며 report는 “over 98%” synthetic이라고 한다. best/worst-of-K loop는 기술하지 않는다. preference response는 여러 intermediate model 및 best model의 여러 sample에서 온다([[nemotron-4-synthetic]] §3.2.3, §3.3.1).
10. “code SFT first, general SFT second는 Phi-4, Qwen, Tülu에도 있고 structured-format following을 설치한다” → 이 장 source 중 two-stage SFT를 보고한 것은 Nemotron-4뿐이며, 함께 학습할 때 behavior 충돌, 특히 coding에서 data blend sampling weight 조정이 모든 coding task를 맞추지 못한다는 것이 이유다. Phi-4는 SFT 한 round를 보고하며 Qwen·Tülu 3 code-first staging을 뒷받침하는 source는 없다([[nemotron-4-synthetic]] §3.3.1; [[phi-4]] §4).
11. “pretraining loss는 quantity보다 quality에 먼저 제한되고 10× 큰 모델, 100× 많은 token과 경쟁한다”는 Phi-1 thesis → 논문에 없다. phi-1은 약 7B token dataset으로 HumanEval 50.6%를, StarCoder(15.5B, 1T token)는 33.6%를 보고한다([[phi-textbooks]] Table 1, Verification).
12. Phi token/generator 표의 여러 수치 → phi-1-base는 50B 초과 token(7B 미만 데이터에 약 8 epoch), phi-1.5는 30B-token dataset에서 150B token이고 generator를 이름 붙이지 않는다. Phi-3는 phase split·generator가 없고, Phi-4는 약 10T token에서 synthetic 40%, web rewrite 15%다([[phi-textbooks]] §2.3; [[phi-1-5]] Table 1, Verification; [[phi-3]] §2; [[phi-4]] §3, Table 5).
13. “Phi-1이 hand-labeled educational/non-educational seed에 random forest 학습” → GPT-4가 약 100k sample의 educational value를 annotation하고 codegen-model embedding 위 random forest가 label을 확장했다([[phi-textbooks]] §2.1).
14. “common-sense·grade-school science·logic·everyday reasoning을 포괄하는 hand-curated topic 20K를 GPT-3.5로 확장” → report는 generation seed로 topic 20K를 carefully selected했고 diversity를 위해 prompt에 web sample을 쓴다고 한다. topic description과 generator는 report에 없다([[phi-1-5]] §2.2).
15. “Phi-4에는 90 GRPO step만 필요, 모든 gain은 SFT이고 RL은 polish” → Phi-4-reasoning-plus는 약 6k example으로 90 GRPO step을 학습했고 AIME를 10% 이상 올렸다. 저자는 response가 31k token에서 clip되어 GRPO가 바꿀 수 있는 범위를 제한한다고 덧붙인다([[phi-4]], arXiv:2504.21318 §4.2).
16. “Phi 분석은 synthetic exercise와 HumanEval prompt의 non-trivial overlap을 지적하며 50.6%를 의심한다” → 인용된 분석은 없다. phi-1 §5는 HumanEval 13-gram match 4개가 모두 false positive라고 하고, similar exercise를 pruning한 뒤 45.1–50.6%, similar 대 non-similar problem에서 81.7% 대 26.9%를 보인다. 공개된 Phi overfitting 근거는 GSM1k를 통한 GSM8K에 관한 것이다([[phi-textbooks]] §5, Table 3; [[gsm1k]] §5.1).
17. “Phi-1.5는 harder benchmark(MMLU-Pro, MATH)에서 가장 약하다” → phi-1.5는 MMLU-Pro나 MATH에서 평가하지 않았다. HellaSwag 0.476 대 Llama2-7B 0.571, MMLU 0.376 대 0.453으로 낮다([[phi-1-5]] Table 3).
18. “첫 pass에서 duplicate-class output이 많은 prompt 30M을 만들었고 prompt taxonomy restructuring으로 고쳤다”, “deduplication이 진짜 bottleneck” → blog는 generation 전 prompt를 iteration하고 audience·style별 content 차이에 대한 explicit instruction을 더해 duplicate content 1% 미만의 prompt 30M 초과에 도달했다고 한다. 대부분 시간은 prompt engineering에 썼다([[cosmopedia]]).
19. Cosmopedia web branch가 “audit move이며 dedup workload를 두 배로 한다” → curated source가 scale하지 않아 web data를 썼고( OpenStax 16,000 unit, Stanford 250,000 unit), web이 prompt 23M, 즉 80% 이상을 제공했다. dedup workload는 논의하지 않는다([[cosmopedia]]).
20. “어떤 tree expansion도 그 combination을 자연스럽게 내지 않는다” → GLAN은 한 syllabus의 두 session에서 key concept을 의도적으로 결합한다. 서로 다른 subject나 discipline 사이 combination은 어느 sampling strategy도 생성하지 않는다([[glan]] §2.4).
21. scaling rule of thumb(30–50 root field, 3M–30M sample 등) → 해당 값은 source에 없으며 보고된 quantity는 §2, §5, §6에 열거했다.
22. “[[ch-32]] / [[ch-34]] — SFT recipe section의 Phi-3/4 case study; Nemotron-Ultra” → ch-32는 Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL이다. Phi report는 ch-34, teacher-data insertion은 ch-35가 다룬다(outline).
23. Figure의 “Taxonomy-only pipeline plateau” 및 coverage bar → 둘 다 source가 보고하지 않아 figure를 다시 썼다.

## 범용 모델에 이 장이 중요한 이유

범용 모델에는 아무도 prompt를 수집하지 않은 domain의 training data가 필요하다. seed-based generation(ch-19)은 seed 근처 example을 만든다. GLAN 저자는 few-shot prompting이 “demonstration과 유사한 new instruction을 생성하는 경향”이 있고 rewriting method는 input dataset에 제한된다고 말한다([[glan]] §1). taxonomy 기반 합성은 먼저 target domain을 열거해 이 문제를 다룬다. textbook-style 합성은 두 번째 문제를 다룬다. web text는 next-token prediction으로 배우기 어려운 형태로 knowledge를 제시하며 Phi 저자는 generated exposition과 exercise가 더 배우기 쉽다고 주장한다(해석, [[phi-4]] §2.1).

두 방법은 pretraining·mid-training(Phi, Cosmopedia)과 SFT(GLAN, Nemotron-4)라는 파이프라인 두 곳에 놓인다. 둘 다 같은 generality 질문을 낳는다. topic list와 teacher가 student가 보는 것을 정하므로 gain은 열거된 topic 및 생성 format과 닮은 benchmark에 집중될 수 있다. 이 장은 source가 그 집중을 어떻게 측정했는지 보고한다.

## §1 용어와 coverage 문제

**Seed-based synthesis**는 기존 example으로 model을 prompt해 새 example을 생성한다(ch-19). **Taxonomy-driven synthesis**는 discipline, subject, class session 같은 명시적 topic hierarchy에서 example을 생성한다. **Textbook-style synthesis**는 보통 pretraining을 위해 topic 및 audience를 조건으로 expository text와 exercise를 생성한다. **teacher**는 data를 쓰는 model이고 **student**는 그 data로 학습하는 model이다.

이 장에서 **coverage**는 dataset 또는 model의 측정 가능한 속성이다. target category 중 training example이 있는 비율, 그리고 학습 model이 regression하지 않는 target category의 비율이다. **Held-out coverage evaluation**은 training 또는 model selection에 쓰지 않은 example으로 각 category를 시험한다.

측정으로 말한 문제는 다음과 같다. K개 entry의 category list에서 seed-based dataset은 일부 entry에 0 example일 수 있다. 예시 count로 target list가 discipline 126개이고 seed pool example이 그중 90개에 있으면 data coverage는 90/126 = 71.4%다. discipline마다 question을 쓰는 taxonomy-driven generator는 construction상 126/126 = 100% data coverage다. model coverage는 별도 수치다. GLAN은 모든 discipline에 data를 쓰지만, GLAN-Test gap은 American history, Divinity, Radiology에서 세 smaller baseline(Orca-2-7B, Mistral-7B-Instruct-v0.1, WizardLM-13B-V1.2) 모두 및 GPT-4보다 음수다([[glan]] Table 8). data coverage는 model coverage를 보장하지 않는다.

## §2 GLAN: discipline 목록에서 homework question까지

**정의.** GLAN(Generalized Instruction Tuning)은 seed example이나 기존 dataset을 input으로 쓰지 않고 human knowledge 및 capability taxonomy에서 instruction data를 생성한다([[glan]] Abstract).

**메커니즘.** 논문의 Algorithm 1:

```
D ← build_taxonomy()                     ▷ list of disciplines (§2.1)
for each discipline d ∈ D do
    S ← generate_subjects(d)             ▷ §2.2
    for each subject s ∈ S do
        A ← generate_syllabus(s, d)      ▷ §2.3
        C, K ← extract_class_details(A)  ▷ class sessions and key concepts
        Q ← generate_instructions(A, C, K, d)   ▷ sample sessions and key concepts (§2.4)
        L ← L ∪ Q
return L
```

1. **Taxonomy.** GPT-4에 “list all fields of human knowledge and capabilities” 같은 instruction을 prompt한다. human annotator가 각 element의 유지·제거를 투표하며 field 또는 sub-field를 제거하면 그 descendant도 제거된다. discipline 126개가 남는다. top-level field에는 Natural Sciences, Humanities, Services(vocational training)가 있다(§2.1, §3.1).
2. **Subject.** discipline의 education expert 역할인 GPT-4가 subject를 열거한다. 첫 prompt에서 format instruction을 넣으면 list quality가 낮아져 두 번째 prompt가 list를 jsonl로 변환한다. discipline당 temperature 1.0, top-p 0.95로 10 query를 수행해 discipline당 subject 100~200개를 얻고, discipline 간 duplicate는 유지한다(§2.2, §3.1).
3. **Syllabus.** subject당 GPT-4 query 하나가 class session 10~30개와 각 session당 약 다섯 key concept이 있는 syllabus를 생성한다(§2.3, §3.1).
4. **Question.** session 하나 또는 둘, key concept 하나부터 다섯을 sample하고 GPT-4가 이들 및 전체 syllabus를 받아 homework question을 쓴다(§2.4).
5. **Answer.** GPT-3.5-turbo가 temperature 0.7에서 각 question에 별도 call로 답한다. “reasonably good results”와 함께 더 빠르므로 선택했다(§3.1).
6. **Decontamination.** 평가한 모든 benchmark의 test·training set에 있는 question 또는 input prompt를 포함하는 pair를 제거한다(§3.1).

**수식.** question 하나에 가능한 서로 다른 concept set의 수:

- session 하나: `N_1(m) = Σ_{i=1..5} C(m, i)`
- session 둘: `N_2(m1, m2) = Σ_{i=2..5} C(m1+m2, i) − Σ_{i=2..5} C(m1, i) − Σ_{i=2..5} C(m2, i)`

여기서 m, m1, m2는 sample한 session의 key concept 수이고 i는 question에 든 concept 수이며 C(n, i)는 binomial coefficient다. 뺄셈은 모든 concept을 한 session에서만 뽑은 set을 제거하므로 two-session set은 모두 두 session을 사용한다.

**계산 예시.** m = 5이면 N_1 = 5 + 10 + 10 + 5 + 1 = 31이다. m1 = m2 = 5이면 Σ_{i=2..5} C(10, i) = 45 + 120 + 210 + 252 = 627, Σ_{i=2..5} C(5, i) = 10 + 10 + 5 + 1 = 26, N_2 = 627 − 26 − 26 = 575다. m1 = 3, m2 = 4이면 112 − 4 − 11 = 97이다. session 20개에 각각 key concept 다섯 개가 있는 syllabus는 single-session concept set 20 × 31 = 620개와 two-session concept set C(20, 2) × 575 = 190 × 575 = 109,250개를 제공한다(도출).

**Corpus 산술(보고된 범위에서 도출).** discipline당 subject 150개, syllabus당 session 20개라면 126 × 150 = syllabus 18,900개, class session 378,000개다. pair 1,000만 개는 syllabus당 약 529 pair, session당 26.5 pair가 된다. 논문은 discipline별 split 및 sample한 concept set 수를 보고하지 않는다.

[figures/taxonomy-expansion.html](figures/taxonomy-expansion.html)은 m1과 m2를 바꾸어 두 combination count를 보고, subject·session 범위를 바꾸어 도출 corpus size를 보며, §7의 GSM1k gap test를 다시 계산하게 한다.

**근거 및 한계.** 1,000만 pair로 3 epoch 학습한 Mistral 7B가 GLAN-7B 모델이다(§3.2). 논문은 pipeline step 어느 것도 ablation하지 않았다(Verification). answer는 검증되지 않은 GPT-3.5-turbo output이다. field와 sub-field count는 보고되지 않아 human-verification step의 크기를 확인할 수 없다. data는 single question-answer pair이며 multi-turn conversation과 long document는 future work로 열거한다(§5).

**함의.** GLAN input은 task list가 아니라 discipline list다. 이 선택은 어떤 subject가 data를 받는지는 통제하지만, 어떤 task format(multi-turn, strict formatting, tool call)이 나타나는지와 answer가 올바른지는 통제하지 않는다.

## §3 GLAN 결과가 측정하는 것

**주요 표**(Table 1; MBPP와 BBH는 3-shot, 나머지는 0-shot):

| 모델 | HumanEval | MBPP | GSM8K | MATH | BBH | ARC-E | ARC-C | MMLU |
|---|---|---|---|---|---|---|---|---|
| Mistral 7B (base) | 28.0 | 50.2 | 43.4 | 10.0 | 56.1 | 79.5 | 53.9 | 62.3 |
| Mistral Instruct 7B | 46.7 | 31.7 | 24.4 | 8.2 | 46.0 | 76.9 | 52.0 | 53.7 |
| WizardMath v1.1 7B | 51.2 | 54.1 | 83.2 | 33.0 | 58.2 | 79.8 | 53.2 | 60.3 |
| Mistral CodeAlpaca 7B | 35.4 | 50.2 | 34.6 | 8.3 | 56.1 | 79.1 | 54.2 | 60.9 |
| GLAN 7B | 48.8 | 57.6 | 80.8 | 32.7 | 60.7 | 90.7 | 81.1 | 62.9 |

**결과(단일 연구).** Table 1의 Mistral 7B 기반 모델 중 GLAN만 base보다 여덟 column 전부에서 개선된다. math-specialized WizardMath는 GSM8K, MATH, HumanEval에서 더 높고, code-specialized CodeAlpaca는 GSM8K를 43.4에서 34.6으로 낮춘다. 저자들은 specialized model이 target benchmark에서는 개선하지만 “while usually not others”라고 관찰한다(§3.3).

**Category별 관점.** category별 MMLU(Table 2)는 base Mistral이 STEM 52.0, Humanities 56.5, Social Sciences 73.3, Other 70.1이고 GLAN은 60.1, 54.9, 71.8, 68.6이다. aggregate MMLU 변화 +0.6은 STEM +8.1과 나머지 세 category의 손실로 구성된다. 저자들은 GPT-3.5-turbo의 chain-of-thought answer가 multi-step STEM question에는 도움을 주고 memorization-heavy question에는 error를 더할 수 있다고 제안한다(해석, §3.3).

**Instruction following.** IFEval strict prompt-level accuracy는 GLAN-7B 34.0, Mistral-7B-Instruct-v0.1 32.0, GPT-3.5-turbo 53.8이다(Table 4). format constraint benchmark에서 GLAN-7B는 Mistral-7B-Instruct-v0.1보다 2.0점 높고 GPT-3.5-turbo보다 19.8점 낮다.

**Train 대 test loss 검사.** GLAN은 모델이 benchmark-style training data에 노출되었는지 다음으로 검사한다.

`Δ = L_test − L_train`, `Δ(%) = (L_test − L_train) / L_test`

여기서 L_test와 L_train은 benchmark test split 및 training split의 model 평균 loss다. training split(또는 이와 같은 text)으로 학습한 모델은 L_train이 L_test보다 낮으므로, 같은 benchmark의 comparable model보다 높은 Δ(%)는 in-domain training data 노출을 시사한다.

가상의 loss 계산 예시에서 L_train = 1.00, L_test = 1.10이면 Δ = 0.10, Δ(%) = 0.10 / 1.10 = 9.1%다. L_train = 1.00, L_test = 1.01이면 1.0%다. 보고된 GSM8K 값(Table 3)은 GLAN-7B 0.92%, WizardLM-13B-V1.2 4.39%, Orca2-7B 11.4%다. Orca 2 및 WizardMath-style model은 benchmark training set을 사용한다([[glan]] Connections). **결과(단일 연구).** test item leakage는 L_test를 낮추고 Δ를 더 작거나 음수로 만들므로, 작은 Δ(%)가 test-set leakage가 없음을 뜻하지는 않는다(해석). 이 검사는 training split이 있는 benchmark도 필요로 한다.

**GLAN-Test.** GLAN이 생성한 data에서 discipline당 50개, 총 instruction 6,300개를 held out하고 GPT-4 pairwise judging으로 점수화했다. Orca2-7B 대비 +1.61, Mistral-7B-Instruct 대비 +0.43, WizardLM-13B-V1.2 대비 +0.19, GPT-4 대비 −0.55다(§3.5, Table 6). test item이 generator distribution에서 오므로 GLAN-Test는 student가 taxonomy 자체의 question을 얼마나 cover하는지 측정하며 user가 쓴 prompt로의 transfer는 측정하지 않는다(해석, [[glan]] Findings).

## §4 Nemotron-4 340B: topic-seeded prompt, 별도 reward model, staged SFT

**정의.** Nemotron-4-340B-Instruct는 “over 98%”가 synthetic인 SFT 및 preference data로 aligned되며, human-annotated example 약 20K를 쓴다. SFT용 10K와 HelpSteer2 preference example 10K다([[nemotron-4-synthetic]] §3.2).

**Pipeline의 역할.**

| 역할 | 모델 또는 signal | 위치 |
|---|---|---|
| Prompt generator | Mixtral-8x7B-Instruct-v0.1(permissive license) | §3.2.1 |
| Topic list | Mixtral이 생성한 macro-topic → subtopic 및 manual topic: topic 3K; Python keyword 12K, math keyword 17K | §3.2.1 |
| 실제 prompt | LMSYS-Chat-1M, 서로 겹치지 않는 SFT 및 preference set으로 분할 | §3.2.1 |
| Response generator | round 1에서는 Mixtral, 이후에는 aligned 340B intermediate model 각각 | §3.2.4 |
| Chosen/rejected label | 가능한 경우 ground truth 또는 Python verifier; 초기 LLM-as-judge; 이후 reward model(Chat-Hard 0.87 대 0.54) | §3.2.3 |
| Reward model | Nemotron-4-340B-Reward: 다섯 HelpSteer attribute 위 linear head가 있는 base model, HelpSteer2 10K로 학습 | §3.1 |
| Code data | Genetic Instruct: self-instruction 및 WizardCoder mutation, LLM fitness check, 약 800K sample | §3.3.1 |

Topic seeding은 GLAN과 같은 의미의 shallow taxonomy step이다. 생성한 topic list가 prompt가 cover하는 subject를 결정한다(해석). Nemotron은 synthetic 및 실제 prompt의 차이도 측정한다. Mixtral response의 mean reward-model helpfulness는 synthetic prompt에서 3.24, LMSYS prompt에서 3.04이며 report는 실제 prompt가 더 어렵다고 읽는다(Figure 3). **결과(단일 연구).** 생성 topic list는 user prompt보다 쉬운 prompt를 만들 수 있다.

**Staged SFT.** report는 “learning multiple behaviors concurrently can sometimes lead to conflicts between them”이라고 하며, 가장 강하게는 coding에서 “adjusting the sampling weights for the data blend fails to align the model to all coding tasks”라고 한다(§3.3.1). Code SFT를 먼저 학습한다(약 800K sample, 1 epoch, LR 3e-7). 다음은 General SFT다(“forgetting risk를 완화하기 위해” code sample 2%를 포함한 200K sample, 3 epoch). single-stage 수치는 인쇄하지 않는다.

Stage 효과(Table 6):

| 단계 뒤 | MT-Bench | MMLU | GSM8K | HumanEval | IFEval prompt-strict |
|---|---|---|---|---|---|
| Code SFT | 6.79 | 72.2 | 77.6 | 70.7 | 46.4 |
| + General SFT | 7.99 | 78.3 | 87.9 | 66.5 | 61.4 |
| + DPO | 7.90 | 78.4 | 88.5 | 67.1 | 61.7 |
| + RPO, 3rd iteration | 8.22 | 78.7 | 92.3 | 73.2 | 79.9 |

Code SFT는 base의 HumanEval 57.3을 70.7로 올렸다. 그러나 General SFT는 code replay 2%에도 66.5로 낮췄다. **결과(단일 연구).** 뒤의 preference stage는 HumanEval을 73.2로 회복했다.

**함의.** topic list는 한 control이고, reward model·verifier·stage order는 어떤 behavior가 살아남는지에 대한 별도 control이다. staged-SFT 근거는 single-stage table이 없는 340B의 한 report다.

## §5 Phi: pretraining에서의 textbook-style synthetic data

**정의.** Phi report는 filtered web 또는 code data에 LLM 생성 “textbook-quality” text 및 exercise를 더해 작은 model을 학습한다.

| Report | 모델 | 보고한 data | Token | 위치 |
|---|---|---|---|---|
| Textbooks Are All You Need (2023-06) | phi-1, 1.3B | filtered code 약 6B + GPT-3.5 textbook 1B 미만; CodeExercises 180M token 미만(문제 879.5K)으로 finetuning | 50B 초과를 봄(약 8 epoch) | [[phi-textbooks]] §2, §2.3 |
| Textbooks Are All You Need II (2023-09) | phi-1.5, 1.3B | phi-1 data(7B) + topic 20K로 seed한 새 synthetic token 약 20B; generator 이름 없음 | 30B dataset, 150B를 봄 | [[phi-1-5]] §2.2, Table 1 |
| Phi-3 Technical Report (2024-04) | phi-3-mini, 3.8B | “educational level”로 filter한 web + synthetic, 두 phase; split·generator 이름 없음 | 3.3T | [[phi-3]] §2 |
| Phi-4 Technical Report (2024-12) | phi-4, 14B | synthetic data 50 type, unweighted token 약 400B; web, web rewrite, code, acquired source | 약 10T | [[phi-4]] §2.2, §3 |

**Phi-1 메커니즘.**
1. GPT-4가 “기본 coding concept을 배우려는 학생에게 educational value를 determine”하라는 prompt로 code sample 약 100k개에 label을 붙인다(§2.1).
2. codegen-model embedding 위 random forest가 35B token 초과의 전체 pool에 점수를 매기고 약 6B token을 유지한다(§2.1).
3. GPT-3.5가 topic 및 audience constraint에서 오는 diversity로 textbook 1B token 미만을 쓴다(§2.2).
4. GPT-3.5가 function name으로 다양화한 docstring-completion problem인 CodeExercises를 쓰며, phi-1-base를 이 data로 finetune해 phi-1을 얻는다(§2.2).

350M parameter에서 unfiltered data는 HumanEval 12.19%, filtered subset은 17.68%, filtered + synthetic textbook은 20.12%에 도달한다(§2.1). phi-1은 HumanEval 50.6%, MBPP 55.5%이고 phi-1-base는 29%다(Table 1, §2). **결과(단일 연구).**

**Phi-1.5.** 저자들은 “carefully selected 20K topics to seed the generation”했고 “use samples from web datasets for diversity”한다고 한다(§2.2). phi-1.5는 WinoGrande 0.734, GSM8K(coding을 통해) 40.2로 Llama2-7B의 0.691 및 14.6보다 높다. 하지만 HellaSwag에서는 0.476 대 0.571, MMLU에서는 0.376 대 0.453으로 Llama2-7B보다 낮다(Tables 2–4). filtered web 40%, phi-1 code 20%, synthetic data 40%로 300B token을 학습한 phi-1.5-web은 GSM8K 44.6, HumanEval 41.4로 phi-1.5의 40.2, 34.1보다 높다(§2.4, Table 4). 두 run은 본 token이 150B 대 300B로 다르므로 equal-token 비교가 아니다.

**Phi-4 mixture.** Table 5는 final pretraining mixture를 training token의 비율 및 source별 unique token으로 제시한다.

| Source | Training token 비율 | Unique token | Epoch |
|---|---|---|---|
| Web | 15% | 1.3T | 1.2 |
| Web rewrite | 15% | 290B | 5.2 |
| Synthetic | 40% | 290B | 13.8 |
| Code data | 20% | 820B | 2.4 |
| Acquired source | 10% | 580B | 1.7 |

epoch column은 allocated token을 unique token으로 나눈 값이다(§3.2). 보고된 총 약 10T token으로 계산하면 synthetic은 0.40 × 10T = 4T이고 4T / 290B = 13.8 epoch다. web은 0.15 × 10T = 1.5T이고 1.5T / 1.3T = 1.15, 인쇄된 값은 1.2다. web rewrite는 “a sub-category of synthetic data”(footnote 5)이므로 이 표에서 synthetic source는 training token의 최소 55%를 차지한다. code data도 synthetic과 raw code의 mixture다(§3.2). report는 §2.2의 unweighted token 400B와 Table 5의 unique token 290B + 290B를 조정해 설명하지 않는다.

**Synthetic 대 web data의 Phi-4 ablation.**
- web data가 없는 13B model에서 phi-3-medium 대비(Table 3), synthetic only는 MMLU +0.8, HumanEval +12.1, MATH +4.9, TriviaQA −14.8이고 synthetic + web rewrite는 TriviaQA −7.7이다.
- 1T-token horizon의 7B model에서 final mixture 대비 75% token을 재배정하면(Table 4), synthetic-heavy(S)는 평균 +0.8이며 TriviaQA −3.0, synthetic + web(S+W)은 평균 0.0이며 TriviaQA +6.9, uniform allocation은 평균 −2.2다.
- 저자들은 knowledge benchmark를 개선하기 위해 “targeted and knowledge-heavy filtered web data sources”를 final mixture로 골랐고, post-training 뒤 synthetic-heavy run과의 gap이 수치 없이 “largely closes”한다고 보고한다(§3.2).
- §3.1: “Models trained only with synthetic data underperformed on the knowledge-heavy benchmarks and demonstrated increased hallucinations.”

**결과(한 lab의 세 report; 독립 replication 아님).** synthetic-heavy pretraining은 phi-1.5(HellaSwag, MMLU 대 Llama2-7B; [[phi-1-5]] Table 3), phi-3-mini(TriviaQA 64.0 대 Mixtral 8x7B 82.2, 저자는 limited capacity로 귀속; [[phi-3]] §3, §6), Phi-4 ablation([[phi-4]] Tables 3–4)에서 reasoning보다 knowledge recall에 더 약하다.

## §6 Cosmopedia: textbook-style 합성의 open replication

**정의.** Cosmopedia는 Mixtral-8x7B-Instruct-v0.1이 생성한 synthetic textbook, blog post, story, WikiHow article의 Hugging Face dataset이다. Phi-1.5에 사용한 training data를 “reproduce”하기 위해 만들었으며 file 30M 초과, token 25B, GPU hour 10k 초과다([[cosmopedia]]). **출처 신뢰성: 공식**(Hugging Face가 dataset을 만들고 cosmo-1b를 학습했으며 code, data, model을 공개했다. ablation table은 없음).

**메커니즘.**
1. **Curated seed.** Stanford course outline, Khan Academy, OpenStax, WikiHow의 unit을 사용한다. 이들은 scale하지 않았다. OpenStax unit 16,000개와 Stanford unit 250,000개는 최소 prompt 2,000만 개라는 target에 못 미친다.
2. **Audience 및 style.** audience 네 가지(young child, high school student, college student, researcher)와 style 세 가지(textbook, blog post, wikiHow article)는 topic당 최대 prompt 12개를 준다. audience 또는 style word만 바꾸는 것은 “high rate of duplicate content를 막기에 insufficient”했고, format과 content가 어떻게 달라져야 하는지를 명시하는 instruction이 해결책이었다.
3. **Web seed.** 수백만 web sample을 cluster 145개로 묶는다. Mixtral이 random extract 10개에서 각 cluster의 이름을 정하고 educational value를 점수화하며 low-value cluster는 제거한다(topic 112개 유지). prompt는 web sample과 관련된 textbook을 요청하며 50% 확률로 cluster topic을 조건으로 한다. 이는 전체의 80% 초과인 prompt 23M개를 제공한다.
4. **Story.** “In our initial assessments of models trained using the generated textbooks, we observed a lack of common sense and fundamental knowledge typical of grade school education.” UltraChat 및 OpenHermes2.5에서 seed한 story를 추가했다.
5. **Decontamination.** 10-gram overlap candidate를 `difflib.SequenceMatcher`로 benchmark와 비교하고, `len(matched_substrings) / len(benchmark_sample) > 0.5`이면 sample을 버린다.

Decontamination rule의 계산 예시: benchmark question이 200 character이고 generated document에서 match된 substring의 총합이 120 character면 120 / 200 = 0.6 > 0.5이므로 document를 제거한다. match 80 character면 0.4이므로 유지한다. 이 rule은 예를 들어 web, Stanford, OpenStax group에서 고유 BoolQ item 41개와 match한 sample 386개를 제거했다.

**결과.** cosmo-1b(1B, Llama2 architecture)는 ARC-Easy, ARC-Challenge, OpenBookQA, MMLU에서 TinyLlama 1.1B보다 높고 Phi-1.5와는 “some performance gaps”가 있다고 보고된다. score는 figure에만 있고 training token은 주어지지 않는다.

**한계 및 함의.** Step 4는 topic list를 inspect한 것이 아니라 학습 model을 평가해 발견한 coverage gap이다. educational source와 web topic에서 만든 textbook prompt에는 grade-school common sense가 없었다. blog는 Mixtral이 historical fact와 mathematical reasoning 등에서 “may sometimes hallucinate”할 수 있다고도 보고한다.

## §7 gain은 targeted benchmark 밖에서도 유지되는가?

**(a) Synthetic data와의 유사성.** phi-1 저자는 CodeExercises와의 similarity(AST match, τ = 0.95)로 HumanEval을 나눈다. phi-1은 similar problem 71개 중 81.7%, non-similar problem 93개 중 26.9%를 풀고, StarCoder-Prompted는 57.7%, 29.0%를 푼다([[phi-textbooks]] Table 3). **결과(단일 연구).** non-similar subset에서 phi-1은 StarCoder-Prompted보다 2.1점 낮으므로, 장점은 exercise와 닮은 problem에 집중된다(해석). GPT-4가 채점한 새 problem 50개에서는 phi-1 52%, StarCoder 51%다(Table 2).

**(b) Fresh, distribution-matched problem.** GSM1k에는 annotator가 “without assistance from any LLM”으로 쓴 grade-school problem 1,205개가 있으며 solution step, answer magnitude, human solve rate를 GSM8K에 맞췄다([[gsm1k]] §1, §3). 저자들은 “Phi and Mistral families of models, show systematic tendencies to perform stronger on GSM8k compared to GSM1k for almost every release and scale”라고 보고하고(§5.1), frontier model은 gap이 최소라고 한다(§5.2).

gap은 two-proportion z-test로 시험한다.

`z = (p_8k − p_1k) / sqrt( p̄ (1 − p̄) (1/n_8k + 1/n_1k) )`, `p̄ = (p_8k n_8k + p_1k n_1k) / (n_8k + n_1k)`

p_8k와 p_1k는 GSM8K와 GSM1k accuracy이고 n_1k = 1,205, n_8k = GSM8K test problem 1,319개다([[metamath]] §4.1).

phi-2의 계산 예시(App. F)에서 p_8k = 0.566, p_1k = 0.504다. p̄ = (0.566 × 1,319 + 0.504 × 1,205) / 2,524 = 0.5364다. standard error = sqrt(0.5364 × 0.4636 × (1/1,319 + 1/1,205)) = sqrt(0.2487 × 0.001588) = 0.0199다. z = 0.062 / 0.0199 = 3.12이며 appendix는 unrounded accuracy에서 3.167을 인쇄한다. 다른 행은 Phi-3-mini-4k-instruct gap 0.040(z 2.385), Phi-3-medium-128k-instruct 0.044(z 3.103), phi-1.5 0.051(z 2.814), Mistral-7B-v0.1 0.027(z 1.421), gpt-4 −0.012다. alternative prompt에서 Phi-3-mini-4k-instruct gap은 0.007로 떨어지고 Phi-3-mini-128k-instruct는 0.035로 오른다(App. F). 따라서 checkpoint 하나에 대한 single-prompt verdict는 안정적이지 않다.

**조건 및 한계.** 저자는 gap과 각 model의 GSM8K test set per-character likelihood 사이 Spearman correlation 0.36을 찾아 “data contamination is likely not the full story”라고 결론 내리고, benchmark-like training data와 benchmark 기반 checkpoint selection도 가능한 원인으로 든다(§5.4). phi-2가 “is still able to correctly solve over half of GSM1k problems”라고도 보고한다(§5.3). GSM1k는 Phi training data 중 어떤 부분이 gap을 만들었는지는 식별하지 않는다.

**(c) Cutoff 뒤 contest.** Phi-4는 training data 수집 뒤인 2024년 11월 6일 이후 공개된 question 78개가 있는 November 2024 AMC-10/12에서 평가했고, 150점 중 평균 91.8이다. Qwen 2.5 14B-Instruct는 77.4, Gemini Pro 1.5는 89.8이다([[phi-4]] §1.1, Fig. 1). 저자는 이를 Phi-4 MATH score가 overfitting 때문이 아니라는 근거로 읽는다. Footnote 8은 final candidate 세 개 모두 89점을 넘었고, 다른 두 candidate의 score를 본 뒤 final model을 선택했다고 공개한다. **결과(단일 연구)**이며 fresh set이 model selection에 일부 관여했다.

**(d) Knowledge recall 및 hallucination.** Phi-4는 Table 1에서 SimpleQA F1 3.0, IFEval 63.0이다. 저자는 SimpleQA, DROP, IFEval을 가장 약한 benchmark score로 들고, 첫 둘은 reductive하다고 보며 IFEval은 “a real weakness”라고 부른다. 이를 Q&A 및 reasoning에 집중한 synthetic data에 귀속한다(§6, §8). synthetic-only 13B ablation은 TriviaQA 14.8점을 잃었다(Table 3).

**(e) 비교 대상으로서 filtered real data.** FineWeb-Edu는 Llama-3-70B-Instruct annotation으로 학습한 classifier가 3 이상을 준 web page를 유지한다. 1.71B parameter, 350B token에서 FineWeb 대비 MMLU는 33%에서 37%, ARC는 46%에서 57%로 오른다([[fineweb]] §4). threshold 3은 HellaSwag 같은 benchmark와의 trade-off로 선택했으며 filter는 topic을 education·history 쪽으로, business·entertainment·travel에서 멀어지게 바꾼다(§4, §4.1). **결과(단일 연구).** real text classifier filtering과 textbook-style synthesis는 모두 score를 knowledge·reasoning benchmark 쪽으로 움직이고 topic distribution을 바꾼다. **열린 질문:** 이 장 source 중 FineWeb-Edu와 Cosmopedia를 broad suite에서 equal token으로 비교한 것은 없고, 여기서 synthetic과 web data의 유일한 equal-token 비교는 Phi-4 Table 4다.

## §8 coverage와 curated taxonomy의 bias 측정

**Taxonomy에서 bias가 생기는 원천.**
1. **Curator의 선택.** GLAN discipline은 annotator majority vote가 filter한 GPT-4 proposal이며, 제거된 field가 무엇인지는 보고되지 않았다([[glan]] §3.1).
2. **일부 topic으로의 classifier drift.** Phi-4 web classifier는 “tends to over-index on STEM-related keywords”이므로 저자들은 arts, history, travel, culture, entertainment 같은 non-STEM content를 증폭할 별도 pipeline을 만들었다([[phi-4]] §2.3).
3. **열거하기 어려운 domain.** Persona Hub 저자들은 GLAN의 목록처럼 comprehensive key-point list가 narrow domain 밖에서는 curate하기 어렵다고 주장한다([[persona-hub]] Connections, §1; 해석).
4. **더 쉬운 생성 prompt.** Nemotron-4 synthetic prompt는 LMSYS prompt보다 높은 helpfulness score 3.24 대 3.04를 받았다([[nemotron-4-synthetic]] Figure 3).

**Concept coverage가 중요하다는 근거.** LLaMA-2 7B에서 MathScale의 25K-example ablation은 knowledge point를 절반으로 줄이면 MWPBench macro average가 상대 8.6%, topic을 절반으로 줄이면 2.3%, seed question을 절반으로 줄이면 2.9% 낮아짐을 보인다([[mathscale]] Table 6). MathScale은 curated list가 아니라 seed question에서 concept graph를 만들며 CollegeMath seed는 algebra, precalculus, calculus만 cover한다. MathScale-7B는 differential equation 0.6, linear algebra 5.0이지만 GPT-4도 differential equation 1.2이므로 낮은 score를 missing seed topic에만 귀속할 수 없다(App. A.5, Table 10). **결과(단일 연구).**

**Coverage audit 절차(course 권고, 해석).** GLAN의 discipline별 보고와 real-prompt 비교를 결합한다.
1. 실제 user prompt sample(예: Nemotron-4가 쓰는 LMSYS-Chat-1M)을 classifier 또는 LLM으로 taxonomy leaf에 mapping하고, 어느 leaf에도 mapping되지 않는 prompt를 기록한다.
2. leaf별 data coverage(training example 수)와 unmapped share를 계산한다.
3. generator와 독립적으로 작성된 prompt를 포함하는 leaf별 held-out evaluation set을 만들고, base model 대비 leaf별 score를 보고한다.
4. 생성 format과 닮은 benchmark에는 fresh 또는 perturbed version을 추가한다(§7b, §7c).

예시 count의 계산: real prompt 1,000개 중 870개가 leaf에 mapping되고 130개가 mapping되지 않으면 unmapped share는 13.0%다. 모델이 discipline 126개 중 100개에서 개선, 20개에서 regression, 6개에서 변화 없음이면 GLAN Table 8이 American history, Divinity, Radiology에 대해 하듯, 이 report는 20개 regression을 하나의 점수로 평균 내지 않고 이름으로 열거한다.

## 음성 샘플과 음성 피드백

음성 signal에는 네 형태가 있다. 음의 한계 가치(positive target으로 쓰면 해롭고 버리는 sample), content로서의 음성(ordinary target으로 쓰는 실패 또는 refusal), conditioning으로서의 음성(control token 아래에서 학습하는 실패), gradient로서의 음성(sample likelihood를 명시적으로 낮추는 것)이다. 마지막만 sample에서 probability mass를 제거한다. 이 장 source 중 conditioning으로 음성을 쓰는 것은 없다. 전체 유도는 ch-43a에 있다.

**이 pipeline에서 음성이 나오는 곳.**
- *버려짐(음의 한계 가치):* Cosmopedia는 benchmark-overlapping document(ratio > 0.5)를 제거한다([[cosmopedia]]). Nemotron-4는 reward-model threshold 아래 dialogue를 버린다(threshold는 주어지지 않음, [[nemotron-4-synthetic]] §3.2.2). Phi-4는 sample한 answer가 모두 일치하거나 완전히 불일치하는 seed question을 버린다([[phi-4]] §2.2). GLAN은 quality로 아무것도 버리지 않아 틀린 GPT-3.5-turbo answer도 SFT target으로 남는다([[glan]] §3.1).
- *Content로서의 음성:* Nemotron-4는 모델이 할 수 없는 task의 refusal을 ordinary SFT target으로 학습한다(§3.2.5). Phi-4는 base model이 보통 틀리는 (question, refusal) 및 (bogus question, refusal)을 학습한다([[phi-4]] App. A.1).
- *Gradient로서의 음성:* Nemotron-4의 DPO 및 RPO rejected response(§3.3.2), 그리고 Phi-4의 Pivotal Token Search pair가 있다. 여기서 rejected completion은 estimated success probability를 낮추는 token 하나이며, refusal DPO pair(refusal > wrong)는 첫 5 token에 적용한다([[phi-4]] §4.3, App. A.1).

**메커니즘.** logit z의 softmax에 대해 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이며, p_j는 token j 확률, y는 눌러 내리는 token이다. log p_y를 낮추는 gradient step은 logit을 `Δz_j = η (p_j − 1[j = y])`만큼 바꾼다. 다른 모든 token logit은 현재 probability에 비례해 오른다. η = 1, p = (0.6, 0.3, 0.1)에서 token 3을 누르면 Δz = (0.6, 0.3, −0.9)이고 새 probability는 (0.710, 0.263, 0.026)이다. Token 3은 0.074를 잃고 token 1은 0.110을 얻으며 token 2는 0.037을 잃는다. 제거된 mass는 가장 likely한 alternative로 간다. 그 alternative도 틀렸거나 chosen 및 rejected response가 token을 공유하면 둘 다 떨어질 수 있다. Nemotron-4는 DPO에서 chosen과 rejected likelihood가 모두 떨어진다고 보고한다(§3.3.2).

**근거.**
- Nemotron-4: DPO는 MT-Bench를 7.99에서 7.90으로 바꿨다. reward-model score gap이 target인 RPO와 chosen response의 추가 SFT loss는 MT-Bench 8.22, IFEval 79.9에 도달했다(Table 6). controlled DPO 대 RPO ablation은 보고되지 않았다.
- Phi-4(Table 9): PTS DPO stage 뒤 GPQA는 53.6, MATH는 80.5이고 judge-guided DPO stage만 수행하면 52.4, 77.6이다. 저자들은 PTS가 low-probability token이 강한 gradient를 받는 full-length pair의 noise를 피한다고 말한다(§4.3).
- Phi-4 refusal data(Fig. 6): SimpleQA correct / not attempted / incorrect는 base의 6.8 / 3.2 / 90.0%에서 final model의 3.0 / 81.1 / 15.8%로 바뀐다. correct answer는 3.8점 낮아지고 incorrect answer는 74.2점 낮아졌으며 F1은 낮아졌다. 저자들은 이를 App. A.1에서 논의한다.

**Control 및 진단.** 이 source가 쓴 control은 음성의 localization(PTS single token; 첫 5 token의 refusal DPO), positive NLL anchor(Nemotron의 chosen response SFT loss), pivotal token이 존재하도록 하는 0.2 ≤ p(success) ≤ 0.8 question filter(PTS)다. 음성을 gradient로 쓸 때에는 chosen·rejected log-probability를 따로 log하고 F1 하나가 아니라 correct·abstained·incorrect rate를 보고하라. Nemotron-4는 DPO 아래 둘의 likelihood가 모두 하락했고(§3.3.2), Phi-4는 incorrect answer가 74.2점 하락하면서 F1도 하락했기 때문이다(Fig. 6).

**일반성에 미치는 효과.** refusal training은 model knowledge 밖 question에서 correct answer와 hallucination 감소를 교환한다(Fig. 6). answerable question에서의 over-refusal은 Phi-4에 대해 보고되지 않았다.

## 레시피

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| GLAN-7B (Mistral 7B) | 7B | SFT | data | question-answer pair 10M; 평가 benchmark test·training set 대비 decontaminated | arXiv:2402.13064v1 §3.1 | verified 2026-09-14 | ablation 보고 없음 |
| GLAN-7B | 7B | SFT | epoch; batch; LR schedule; loss masking | 3; pair 512; 3e-6 cosine, warm-up 1,000 step, final 0; response token만 | §3.2 | verified 2026-09-14 | ablation 보고 없음 |
| GLAN data generation | — | SFT data | subject; syllabus; question; answer | GPT-4, discipline당 query 10, T 1.0, top-p 0.95; subject당 GPT-4 query 하나; session 1–2 및 key concept 1–5; GPT-3.5-turbo T 0.7, top-p 0.95 | §2.4, §3.1 | verified 2026-09-14 | ablation 보고 없음 |
| GLAN-7B | 7B | SFT | sequence length, packing, optimizer, compute | 보고되지 않음 | §2, §3, §5, App. A 확인 | not reported | — |
| Nemotron-4-340B-Instruct | 340B | SFT | Code SFT example; epoch; LR; global batch | ~800K(Genetic Instruct); 1; constant 3e-7; 128 | arXiv:2406.11704v2 §3.3.1 | verified 2026-09-14 | Table 6: HumanEval 57.3 → 70.7 |
| Nemotron-4-340B-Instruct | 340B | SFT | General SFT example; code replay; epoch; batch; LR | 200K; Code SFT sample의 2%; 3; 128; [1e-7, 5e-7]에서 탐색 | §3.3.1 | verified 2026-09-14 | Table 6: MMLU 72.2 → 78.3, HumanEval 70.7 → 66.5; staging 대 single stage 수치 없음 |
| Nemotron-4-340B-Instruct | 340B | SFT + preference | human data | ~20K(10K SFT, 10K HelpSteer2) | §3.2 | verified 2026-09-14 | — |
| Nemotron-4-340B-Reward | 340B | reward-model | data; head | HelpSteer2 10K; attribute 5개로 linear projection | §3.1 | verified 2026-09-14 | RewardBench 92.0(Table 4) |
| phi-1-base | 1.3B | pretrain-stable | data; seq length; LR; warmup; weight decay; batch | filtered code ~6B + synthetic textbook <1B; 2048; 1e-3; 750 step; 0.1; 1024(unit 미명시) | arXiv:2306.11644v2 §2, §2.3 | verified 2026-09-14 | Fig. 2.1: CodeTextbook 대 The Stack+ |
| phi-1-base | 1.3B | pretrain-stable | planned step; released checkpoint | 36,000; step 24,000, 약 8 epoch, 50B token 초과 | §2.3 | verified 2026-09-14 | selection rule 미보고 |
| phi-1 | 1.3B | SFT | data; batch; LR; warmup; weight decay; step | CodeExercises ~180M token; 256; 1e-4; 50; 0.01; 6,000 | §2.3 | verified 2026-09-14 | Fig. 2.1: 이 stage의 최대 HumanEval gain |
| phi-1.5 | 1.3B | pretrain-stable | mixture; dataset; seen token | new synthetic 80%, phi-1 data 20%; 30B; 150B | arXiv:2309.05463v1 §2.3, Table 1 | verified 2026-09-14 | split ablation 없음 |
| phi-1.5 | 1.3B | pretrain-stable | LR; optimizer; batch | 2e-4 constant, warmup 없음; Adam (0.9, 0.98), eps 1e-7, weight decay 0.1; 2048(unit 미명시) | §2.3 | verified 2026-09-14 | ablation 보고 없음 |
| phi-1.5-web | 1.3B | pretrain-stable | mixture; dataset; seen token | filtered web ~40%, phi-1 code 20%, synthetic 40%; 100B; 300B | §2.4, Table 1 | verified 2026-09-14 | Tables 2–4 대 phi-1.5(seen token 다름) |
| phi-3-mini | 3.8B | pretrain-stable | token; phase split; LR | 3.3T; 미보고; 미보고 | arXiv:2404.14219v4 §2 | verified 2026-09-14(token); not reported(split, LR) | ablation 보고 없음 |
| phi-4 | 14B | pretrain-stable | token; schedule; peak LR; weight decay; global batch | ~10T; linear warm-up 및 decay; 3e-4; 0.1; 5760(unit 미명시) | arXiv:2412.08905v1 §3 | verified 2026-09-15 | short run에서 interpolation으로 tune(§3); table 없음 |
| phi-4 | 14B | pretrain-stable | mixture(training token share; unique token; epoch) | web 15%, 1.3T, 1.2; web rewrite 15%, 290B, 5.2; synthetic 40%, 290B, 13.8; code 20%, 820B, 2.4; acquired 10%, 580B, 1.7 | §3.2, Table 5 | verified 2026-09-15 | Table 4(7B, 1T): final 대비 S +0.8, S+W 0.0, uniform −2.2 평균 |
| phi-4 | 14B | pretrain-stable | synthetic data type; unweighted token | 50 type; 약 400B | §2.2 | conflict(report가 Table 5 unique count와 조정하지 않음) | ablation 보고 없음 |
| phi-4 | 14B | long-context | context; token; data; RoPE base; LR | 4K → 16K; 250B; new long data 30%, recall token 70%; 250K; max LR ÷ 10 | §3.3 | verified 2026-09-15 | natural long data가 padded sample보다 “perform better”; 수치 없음 |
| phi-4 | 14B | SFT | LR; token | 1e-6; 약 8B | §4.1 | verified 2026-09-15 | Table 9 stage score |
| phi-4 | 14B | preference | round; pair | PTS DPO(Table 7 mixture); judge-guided DPO ~850K pair, GPT-4o judge | §4.2 | verified 2026-09-15 | Table 9: PTS stage GPQA 53.6 대 stage-2-only 52.4 |
| phi-4 | 14B | preference | PTS question filter | 0.2 ≤ p(success) ≤ 0.8 | §4.3 | verified 2026-09-15 | pivotal token이 범위 밖에선 드물다는 stated reason; ablation 없음 |
| Phi-4-reasoning-plus | 14B | RL | algorithm; step; example; trajectory; checkpoint rule | GRPO; 90; ~6k; example당 8; best AIME 2024 | arXiv:2504.21318v1 §4.2 | verified 2026-09-15 | Fig. 7a: 90 step에서 AIME +>10%, 이후 gain 없음 |
| Cosmopedia (March 2024 release) | — | pretrain data | generator; file; token; compute | Mixtral-8x7B-Instruct-v0.1; 30M 초과; 25B; GPU hour 10k 초과 | huggingface.co/blog/cosmopedia (2024-03-20) | verified 2026-09-15 | cosmo-1b 대 TinyLlama(figure만) |
| Cosmopedia (March 2024 release) | — | pretrain data | decontamination rule | 10-gram candidate; matched / benchmark length > 0.5이면 discard | same blog | verified 2026-09-15 | removal count table; effect 미측정 |
| cosmo-1b | 1B | pretrain-stable | token; LR; batch | 보고되지 않음 | blog 확인 | not reported | — |
| FineWeb-Edu | — | pretrain data | classifier threshold | ≥ 3(유지 token 1.3T) | arXiv:2406.17557v2 §4, App. F.2 | verified 2026-09-14 | FW-Edu-2/3/4, 28B token; HellaSwag와 trade-off |

2026-09-15 행은 [[phi-4]] 및 [[hf-cosmopedia]] library card가 검증되지 않아 해당 날짜에 primary PDF 또는 blog에서 읽었다. chapter excerpt [[phi-4]], [[cosmopedia]]에 확인한 extract가 있다.

**소규모 범용 run의 시작점.** taxonomy-generated question-answer pair로 7B base를 SFT할 때 검증된 reference는 GLAN-7B다. 3 epoch, batch 512 pair, warm-up 1,000 step 뒤 0까지 cosine decay하는 LR 3e-6, response token만 loss를 쓴다. 이는 Mistral 7B에 10M pair로 적용했고 아무것도 ablate하지 않았으며 sequence length와 optimizer는 보고하지 않았다. 같은 run은 MMLU humanities, social sciences, other category를 낮췄으므로 Generalization lens의 category별 평가는 recipe 일부다(Table 2).

## 일반화 관점

**(a) breadth를 늘리는 것.**
- benchmark data 없는 discipline-wide generation: GLAN-7B는 base Mistral보다 Table 1 여덟 column 전부를 개선하고 math·code specialist는 다른 column에서 잃는다(CodeAlpaca GSM8K 34.6 대 43.4)([[glan]] Table 1, §3.3).
- concept combination: knowledge point를 반으로 줄인 비용은 seed question을 반으로 줄인 것보다 크다. 상대 −8.6% 대 −2.9%다([[mathscale]] Table 6).
- synthetic-heavy mixture에 knowledge-heavy web data 추가: 7B, 1T token에서 S+W TriviaQA +6.9 대 S −3.0([[phi-4]] Table 4).
- 평가로 발견한 gap 보완: Cosmopedia는 textbook-trained model에 grade-school knowledge가 없음을 발견한 뒤 common-sense story를 추가했다([[cosmopedia]]). 전후 수치는 없다.
- 생성 prompt 옆 실제 prompt: Nemotron-4는 LMSYS prompt가 synthetic prompt보다 어렵다고 발견했다([[nemotron-4-synthetic]] Figure 3).

**(b) narrowing 또는 forgetting 원인.**
- format coverage 없는 topic list: GLAN IFEval 34.0 대 Mistral-Instruct 32.0 및 GPT-3.5-turbo 53.8([[glan]] Table 4); Phi-4 IFEval 63.0은 Q&A-focused synthetic data의 weakness로 저자가 명명했다([[phi-4]] §6).
- knowledge recall: synthetic-only pretraining TriviaQA −14.8([[phi-4]] Table 3); phi-3-mini TriviaQA 64.0 대 82.2([[phi-3]] §3); GLAN MMLU humanities 54.9 대 56.5([[glan]] Table 2).
- benchmark-shaped data: Phi·Mistral release의 GSM8K-over-GSM1k gap([[gsm1k]] §5.1); phi-1 similar 대 non-similar HumanEval 81.7% 대 26.9%([[phi-textbooks]] Table 3).
- stage order: General SFT는 code replay 2%에도 Nemotron-4 HumanEval을 70.7에서 66.5로 낮췄다([[nemotron-4-synthetic]] Table 6).
- classifier topic drift: Phi-4 web classifier STEM over-indexing([[phi-4]] §2.3); FineWeb-Edu는 business·entertainment·travel을 down-sample한다([[fineweb]] §4.1).

**(c) 이 stage에서 측정하는 법.**
- 모든 aggregate 옆 category별·discipline별 score([[glan]] Tables 2, 8).
- synthetic data 대비 benchmark별 similarity split([[phi-textbooks]] Table 3).
- significance test 및 두 prompt format이 있는 distribution-matched fresh set([[gsm1k]] App. F), model selection에서 holdout된 post-cutoff contest([[phi-4]] §1.1, footnote 8).
- benchmark에 training split이 있을 때 train 대 test loss gap([[glan]] §3.4).
- correct / not attempted / incorrect로 보고한 knowledge-recall 및 abstention metric([[phi-4]] Fig. 6).
- 알려진 measurement error: n-gram decontamination은 rephrased item을 놓친다([[phi-4]] §5). GLAN-Test는 GLAN 자신이 생성한 data에서 오고 PhiBench는 Phi team이 쓰며 GPT-4 judge는 judge bias를 더한다([[glan]] §3.5).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 검사 |
|---|---|---|
| taxonomy-driven SFT 뒤 aggregate만 보고 | MMLU는 오르나 humanities·social-science category 하락 | category별 MMLU 및 discipline별 held-out score([[glan]] Table 2, Table 8) |
| data coverage를 model coverage로 간주 | 모든 discipline에 data가 있으나 일부 discipline regression | base model 대비 leaf별 평가(§8) |
| factual domain에 검증 안 한 teacher answer 사용 | memorization-heavy question error | discipline별 answer sample을 reference와 check; STEM·humanities 변화 비교([[glan]] §3.3) |
| knowledge-heavy web 없는 synthetic-heavy pretraining | reasoning score 상승, TriviaQA·SimpleQA 하락, hallucination 증가 | mixture ablation마다 TriviaQA 또는 유사 recall benchmark([[phi-4]] Tables 3–4) |
| benchmark format의 exercise 생성 | training data와 유사한 benchmark item accuracy가 non-similar item보다 높음(phi-1: 81.7% 대 26.9%) | AST 또는 embedding similarity split([[phi-textbooks]] Table 3) |
| GSM8K 또는 HumanEval만으로 generality 주장 | fresh matched set의 score gap | GSM1k-style set, z-test, prompt format 둘([[gsm1k]] App. F) |
| fresh set으로 final candidate 선택 | fresh-set score가 더 이상 독립적이지 않음 | selection 전에 score를 본 candidate 기록([[phi-4]] footnote 8) |
| prompt에서 audience·style word만 변화 | near-duplicate generation | prompt family별 duplicate rate; explicit content difference 추가([[cosmopedia]]) |
| 한 behavior가 conflict하는데 모든 SFT behavior를 한 stage에 혼합 | reweighting이 한 task family를 고치지 못함 | [[nemotron-4-synthetic]] Table 6과 같은 stage-wise evaluation table |
| n-gram decontamination에만 의존 | benchmark paraphrase가 남음 | paraphrase-aware check 또는 fresh set([[phi-4]] §5) |

## 이해 점검

1. GLAN discipline list는 discipline 126개에 100% data coverage를 준다. 그럼에도 GLAN-7B가 MMLU humanities에서 regression할 수 있는 이유와 그 regression을 낼 pipeline step을 설명하라.
2. two-session combination formula를 사용해 session 간 concept pair sampling이 한 session 내 sampling보다 question variety를 더 늘리는 이유와 두 strategy 어느 것도 만들지 못하는 combination을 설명하라.
3. Phi-4 Table 4에서 synthetic-heavy mixture는 최고 average였는데 저자가 web data가 더 많은 mixture를 고른 이유는 무엇인가? 어떤 측정이 choice를 이끌었고 post-training remark는 mixture ablation을 언제 평가해야 함을 뜻하는가?
4. phi-1은 exercise와 similar HumanEval 81.7%, 나머지 26.9%를 푼다. 이 split이 contamination을 보이지 않는 이유와 StarCoder 대비 phi-1 advantage의 source에 대해 보이는 것을 설명하라.
5. GSM1k는 gap과 GSM8K test-set likelihood 사이 Spearman correlation 0.36만 찾는다. test-set leakage 외에 textbook-style math exercise 학습 model이 GSM1k보다 GSM8K에서 높을 수 있는 mechanism 두 개를 제시하라.
6. Nemotron-4 General SFT는 code data 2%를 포함해도 HumanEval을 낮췄다. 작은 share replay가 loss를 막지 못하는 이유와 같은 table에서 loss가 recoverable함을 보이는 근거를 설명하라.
7. Phi-4 refusal training은 SimpleQA correct answer를 6.8%에서 3.0%로 낮췄다. softmax gradient를 이용해 첫 5 token에서 (refusal > wrong answer) DPO pair가 correct answer probability도 제거할 수 있는 이유를 설명하라.

## 연결

- Previous: ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage(GLAN GPT-4·GPT-3.5-turbo data 및 Phi-4-reasoning o3-mini trace에는 teacher ceiling이 적용됨).
- Next: ch-22 — Quality, Diversity, and Gradient-Based Data Selection(taxonomy-generated·seed-generated pool에서 선택하는 법).
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix; ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing(seed-based alternative 및 Persona Hub).
- ch-23 — Model Collapse and Verification of Synthetic Data(§2 unverified answer); ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data(MathScale 및 concept-level math data).
- ch-10a — Model-Based Quality Filtering and Benchmark-Targeted Data Selection(FineWeb-Edu 및 Phi classifier); ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination(Phi-4 synthetic data 13.8 epoch).
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks; ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares(Nemotron-4 staged SFT).
- ch-34 — Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports; ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data.
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation; ch-48 — Contamination Detection and Its Effect on Reported Scores.

## 출처

- [[glan]] — pipeline, Algorithm 1, combination formula, generation·training setting, Tables 1–4, 6, 8, loss-gap check(card verified 2026-09-14; 같은 이름 chapter excerpt).
- [[nemotron-4-synthetic]] — human 대 synthetic data, topic 3K prompt synthesis, reward-model role, staged SFT, Table 6, DPO·RPO observation.
- [[nemotron-4-synthetic-recipe]] — locus가 있는 SFT, DPO, RPO setting.
- [[phi-textbooks]] — phi-1 classifier, synthetic textbook·exercise, filter ablation, similarity split, training setting.
- [[phi-1-5]] — topic 20K seeding, phi-1.5 대 phi-1.5-web 대 Llama2-7B table, training setting.
- [[phi-3]] — two-phase data description, TriviaQA gap, token.
- [[phi-4]] — arXiv:2412.08905v1 및 arXiv:2504.21318v1에서 읽은 chapter excerpt: synthetic data type, Tables 3–5·9, midtraining, PTS, refusal data·Fig. 6, AMC evaluation, weakness, Phi-4-reasoning-plus RL step. 같은 이름 library card는 검증하지 않아 수치에 쓰지 않았다.
- [[cosmopedia]] — Hugging Face blog에서 읽은 chapter excerpt: prompt curation, audience·style, web cluster, story, decontamination rule·count, cosmo-1b result.
- [[hf-cosmopedia]] — 같은 blog의 library card; 작성 시 검증하지 않아 수치에 쓰지 않았다.
- [[gsm1k]] — arXiv:2405.00332v4 chapter excerpt: construction, family-level overfitting, likelihood correlation, App. F 행 및 prompt sensitivity.
- [[fineweb]] — FineWeb-Edu classifier, threshold trade-off, MMLU·ARC gain, topic shift.
- [[mathscale]] — knowledge-point 대 seed ablation 및 topic별 gap.
- [[persona-hub]] — narrow domain 밖 key-point list curate의 어려움 주장.
- [[metamath]] — z-test에 쓴 GSM8K test-set 크기.
