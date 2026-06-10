<!-- chapter: ch-20
     track: synthetic
     title: Distillation-as-Data and R1-Distill Lineage
     sources: [[orca]], [[orca-2]], [[distilling-step-by-step]], [[deepseek-r1]], [[deepseek-r1-distill-synth]], [[bespoke-stratos]], [[openr1]], [[sky-t1]], [[open-thoughts]], [[dolphin]]
     figures: figures/distill-lineage.html
-->

# 20장 — 데이터로서의 증류와 R1-Distill 계보

> **핵심 통찰.** 2023–2025년의 distillation은 logit-matching이 아니다. 그것은 *teacher에서 sampling하고, sample을 filtering하고, filtered set으로 SFT하는 것*이다. teacher의 가치는 weight가 아니라 **output distribution**, 즉 prompt와 answer 사이의 token trace에 있다. Orca가 vanilla Alpaca와 달라지는 지점, R1-Distill이 Orca와 달라지는 지점은 모두 (a) trace를 어떻게 **elicitation**하는가(system-prompt scaffolding, format tag, multi-strategy prompting), (b) 어떻게 **filter**하는가(gold-answer match, unit-test pass, LLM-judge, symbolic equivalence), (c) teacher의 **quirk**를 얼마나 상속할 의향이 있는가에 대한 결정이다.
>
> **가이드라인.** reasoning distillation에서는 teacher의 trace wrapper를 verbatim 보존하고, domain에 맞는 verifier(SymPy / unit tests / LLM judge)로 rejection-sample하며, 강한 student라면 약 10–20K개의 correct trace를 유지하라. downstream에서 RL도 할 것이 아니라면 약 1M개가 필요하지 않다. teacher는 최고의 benchmark score를 가진 모델이 아니라 *student base에 가장 좋은 output distribution*을 가진 모델로 골라라. QwQ-32B는 R1이 모든 headline eval을 이기는데도 OpenThoughts ablation에서 R1을 이긴다.

---

## 이 장이 필요한 이유

19장은 *자기 자신의* rollout에 대한 rejection-sampling fine-tuning, 즉 Llama-3와 WebInstruct-style self-distillation loop가 verifier signal을 supervised data로 바꾸는 과정을 다뤘다. 이 장은 같은 아이디어의 다른 branch다. **rollout이 더 강한 teacher model에서 올 때 무슨 일이 일어나는가**. 일단 그렇게 하기로 하면 세 가지는 research question이 아니라 engineering decision이 된다.

1. teacher는 어떤 *종류*의 trace를 내며, 그것을 어떻게 steer할 것인가? — Orca의 16개 hand-crafted system message, R1의 `<think>` wrapper, QwQ의 reflection token.
2. 어떤 filter가 trace를 keep하는가? — `\boxed{}` + SymPy equivalence, public test 위의 `assert`, GPT-4o-mini LLM-judge, 또는 아무것도 없음([[open-thoughts]]).
3. student는 *실제로 무엇을 배우는가* — reasoning인가, 아니면 teacher의 stylistic tic인가? — "Wait, let me reconsider..." tic, 4K-token preamble, refusal pattern, benchmark contamination.

Orca→R1-Distill arc는 open post-training에서 가장 깔끔한 two-generation case study다. Orca(2023)는 더 풍부한 *teacher supervision*이 더 큰 *student parameter*보다 낫다고 주장한다. Orca-2(2023)는 더 풍부한 *strategy variety*가 더 풍부한 *single-trace verbosity*보다 낫다고 주장한다. DeepSeek-R1(2025)은 이 human design의 거의 전부를 벗겨낸다. teacher에 RL을 돌리고, 800K reasoning trajectory를 sample하고, correctness로 filter한 뒤 SFT하면 된다. 세 open reproduction(Bespoke-Stratos, Open-R1, Sky-T1)은 이후 더 tight한 filter로 800K corpus를 17K–440K까지 줄이면서 official distill과 2–3 point 이내의 student를 만든다. 이 장의 목적은 그 arc를 펼치고, recipe primitive를 추출하며, licensing과 teacher-bias tradeoff를 명시하는 것이다.

또한 이 장은 "distillation as data"가 *deployment* question이 되는 지점이다. teacher의 terms of service가 이 corpus 생성을 허용했는가? 800K trace를 재배포할 수 있는가? student가 의도하지 않은 refusal pattern을 상속하는가? 이를 덮어둘 수 없다. §7이 각각에 답한다.

---

## 1. Orca — system-prompt scaffolding을 통한 explanation trace

[[orca]]는 template을 만든 2023년 논문이다. 이 논문이 답하는 질문은 이렇다. *13B student가 GPT-4의 answer만이 아니라 reasoning을 배우게 하려면 무엇이 필요한가?* Operational한 답은 **system message**다. Mukherjee 등은 GPT-4(그리고 더 저렴한 coverage를 위해 ChatGPT)에 query하기 전 모든 FLAN-v2 / Big-Bench / Chain-of-Thought / GSM-style task 앞에 **16개의 hand-crafted system message** 중 하나를 붙였다. 각 system message는 **teaching instruction**이다. teacher는 답만 하도록 요구받는 것이 아니라 특정 style로 reasoning을 노출하도록 요구받는다. 작업 목록(논문 Appendix A에서 재구성):

| # | System message (paraphrased) | What trace style it induces |
|---|---|---|
| 1 | "" (empty) | 직접적인 짧은 answer; control 역할 |
| 2 | "당신은 AI assistant입니다. User가 task를 줄 것입니다. 당신의 목표는 task를 충실히 완수하는 것입니다. task를 수행하면서 step-by-step으로 생각하고 step을 정당화하세요." | Generic CoT: numbered reasoning 후 final answer |
| 3 | "당신은 사람들이 정보를 찾도록 돕는 AI assistant입니다. 사용자가 밖에서 따로 찾지 않아도 answer를 이해할 수 있도록 detailed answer를 제공하세요." | Long-form explanatory: definition + worked example |
| 4 | "당신은 instruction을 매우 잘 따르는 AI assistant입니다. 가능한 한 많이 도우세요." | Instruction-literal: phrasing에 높은 fidelity, 요청되지 않으면 elaboration 없음 |
| 5 | "당신은 항상 explanation을 제공하는 helpful assistant입니다. 다섯 살 아이에게 답한다고 생각하세요." | ELI5: analogy-heavy, short sentences |
| 6 | "당신은 AI assistant입니다. User가 task를 줄 것입니다. 당신의 목표는 가능한 한 충실하게 task를 완수하는 것입니다. task를 수행하면서 step-by-step으로 생각하고 step을 정당화하세요. 다섯 살 아이에게 설명하듯 reasoning을 서술하세요." | ELI5 + CoT hybrid |
| 7 | "당신은 teacher입니다. task가 주어지면 task가 무엇을 요구하는지, 제공하는 guideline이 무엇인지, 그 guideline을 사용해 answer를 찾는 방법을 simple step으로 설명하세요." | Guideline-extraction style: task rule을 restate한 뒤 apply |
| 8 | "User가 몇 가지 instruction이 포함된 task를 줄 것입니다. 당신의 일은 가능한 한 충실히 instruction을 따르는 것입니다. 답하면서 step-by-step으로 생각하고 answer를 정당화하세요." | Literal-then-CoT: #4와 #2 결합 |
| 9 | "definition을 사용해 answer에 도달한 방법을 설명하세요." | Post-hoc rationale: answer first, then justify |
| 10 | "당신은 먼저 정확한 step-by-step plan을 작성하고, 그다음 plan을 실행해 question에 답하는 helpful assistant입니다." | Plan-and-execute: two-phase decomposition |
| 11 | "당신은 complex task를 돕는 AI assistant입니다. response에는 항상 answer와 그 뒤에 reasoning에 대한 brief explanation이 포함되어야 합니다." | mandatory rationale을 포함한 short CoT |
| 12 | "당신은 모든 language와 language 간 translation 방법을 아는 AI assistant입니다. task가 주어지면 task가 무엇을 요구하는지, 제공하는 guideline이 무엇인지, 그 guideline을 사용해 answer를 찾는 방법을 simple step으로 설명하세요." | Translation-calibrated reasoning |
| 13 | "task의 definition과 sample input이 주어지면 definition을 작은 part로 나누세요. 각 part에는 instruction이 있습니다. 해당 instruction의 criteria를 만족하는 example을 보여주며 의미를 설명하세요. 다음 format을 사용하세요: Part #: definition의 key part. Usage: key part의 criteria를 만족하는 sample response. 왜 criteria를 만족한다고 생각하는지 설명하세요." | Part-wise definition unpacking; decomposition을 가르침 |
| 14 | "task와 definition이 주어지면 각 step에 적어도 하나의 example을 제공해 definition이 answer로 이어지는 방식을 설명하세요." | Example-grounded reasoning |
| 15 | "task를 describe하고 answer를 설명해야 합니다. multiple choice question에 답할 때는 먼저 correct answer(s)를 output하세요. 그다음 다른 answer가 왜 wrong인지 설명하세요. 다섯 살 아이에게 답한다고 생각하세요." | MCQ-specific: answer + eliminate-distractors |
| 16 | "당신은 사람들이 정보를 찾도록 돕는 AI assistant입니다. 사용자가 밖에서 따로 찾지 않아도 answer를 이해할 수 있도록 detailed answer를 제공하세요. 이 분야의 world-class expert처럼 생각하세요." | Expert-grounded explanatory |

왜 1개가 아니라 16개인가? single CoT prompt는 *single style*을 만들고, student는 그 surface form을 pattern-match한다. 16개 style이 있으면 student는 **style 전반에 invariant한 reasoning**을 배워야 한다. 이것이 논문에서 "explanation trace"를 training signal로 정의하는 operational definition이다.

**Two-stage progressive learning mix.** Orca는 GPT-4만 두드리지 않는다. 먼저 5M ChatGPT trace(저렴한 teacher, 넓은 coverage)를 학습하고, 그다음 같은 prompt pool에서 1M GPT-4 trace(비싼 teacher, depth)를 학습한다. 명시적인 curriculum이다. student는 더 쉬운 explanation을 먼저 보고 더 어려운 것을 나중에 본다. SFT 관점에서는 loss re-weighting이 아니라 data-order curriculum이다. 이 mix로 Llama-13B를 training하면 BigBench-Hard zero-shot이 Vicuna-13B의 44.0에서 Orca-13B의 49.7로 올라갔다. 13B가 "reasoning distillation"이라는 표현이 생기기 2년 전에 10배 큰 baseline과 reasoning eval에서 맞먹은 것이다.

**Orca가 틀렸고 Orca-2가 고친 것.** trace-rich supervision으로 training한 student는 task가 그럴 필요가 없어도 **항상 긴 trace를 emit**하는 경향이 있다("2+2 = ?" → scratch work 400 token). [[orca-2]]는 **strategy variety**로 이를 공격한다. teacher는 이제 다섯 target-behavior mode 중 하나를 emit한다. (i) direct answer, (ii) step-by-step, (iii) explain-then-answer, (iv) recall-then-answer, (v) extract-then-answer. 이는 *task-specific* system message가 선택한다. 핵심 trick은 **Prompt Erasing**이다. student는 strategy-selecting system message 없이 training되므로 inference 시 어떤 strategy를 적용할지 *스스로* 선택해야 한다. 이는 "teacher가 고른 strategy를 execute하는 법을 배움"에서 "각 strategy가 언제 적용되는지 배움"으로 넘어가는 bridge다. 나중에 R1의 `<think>` auto-regulation에서도 같은 움직임이 나타난다(model이 지시받지 않아도 언제 더 오래 생각할지 배움).

---

## 2. Distilling Step-by-Step — joint-training signal로서의 rationale

[[distilling-step-by-step]](Hsieh et al., ACL 2023)은 R1 전에 읽어야 하는 rationale-distillation paper다. Orca보다 claim이 더 날카롭다. **770M T5 student가 labeled data의 80%만 사용하고도 4개 benchmark에서 540B PaLM teacher의 few-shot performance를 이길 수 있다**. 단, teacher의 CoT를 추출해 **joint (label, rationale)** multi-task objective로 student를 training해야 한다.

논문의 two-stage recipe verbatim:

```
1. Extract (x, rationale_teacher, label_teacher) triples by few-shot CoT prompting PaLM-540B
   on each training example (3–8 in-context CoT exemplars).
2. Train T5-770M with multi-task loss
        L = L_label(y | x) + λ · L_rationale(r | x)
   where both heads share the T5 encoder. Decode only y at inference.
```

왜 작동하는가, 그리고 이후 R1에 왜 중요한가:

- **Label만으로는 compressed supervision signal이다** — student는 (x, y) mapping을 memorization한다.
- **Rationale은 shared encoder가 reasoning-relevant feature를 encode하도록 강제한다** — student encoder는 `y`로 가는 길에 어떤 step이 *taken*되는지 배운다. 단순히 `y`가 각 `x`에 대해 무엇인지만 배우는 것이 아니다.
- **Rationale head는 inference에서 버린다**. 하지만 그 gradient는 이미 encoder를 형성했다. multi-task regularization은 deployment에서 싸다.

이긴 benchmark: ANLI(NLI), e-SNLI(explained NLI), CQA(commonsense), SVAMP(math word problem). 770M student는 사용 가능한 labeled data의 80%를 쓴다. 이는 **rationale supervision이 label supervision보다 token당 더 좋은 learning signal을 가진다고 명시적으로 주장한 첫 논문**이다. 이후 모든 work(Orca, Orca-2, R1-Distill, Sky-T1)가 이 thesis를 상속한다. Hsieh가 지적한 failure mode는 2년 후에도 여전히 물어뜯는다. **teacher rationale quality가 student ceiling을 제한한다**. hallucinated rationale은 잘못된 reasoning을 가르치고, student는 verifier가 없으므로 이를 detect할 방법이 없다.

2025년 descendant들은 multi-task objective를 완화한다. R1-Distill은 `<think>…</think><answer>…</answer>`에 single-head SFT만 한다. 그러나 underlying claim은 같다. *rationale이 supervision이고 label은 byproduct다*.

---

## 3. DeepSeek-R1 distill — teacher가 실제로 생성한 것

Orca/DSBS에서 R1로의 이동은 teacher가 *누구인가*의 이동이다. Orca의 teacher는 reasoning-specific post-training이 없는 chat model인 GPT-4였다. R1은 reasoning model이다. RL로 *long trace를 emit하도록 훈련된* 모델([[deepseek-r1]], [[deepseek-r1-distill-synth]])이라서, sample하면 verifier-grounded correctness에 이미 적응한 trace distribution이 나온다. 따라서 student 측 distillation recipe는 거의 지루할 정도로 단순해 보인다. pure SFT다. 하지만 **800K trace pool을 만든 teacher-side pipeline**은 전혀 단순하지 않다.

[[deepseek-r1]] report는 R1 pipeline이 DeepSeek-V3-Base checkpoint 이후 **네 stage**라고 말한다.

1. **Cold-start SFT**: 사람이 정리한 long-CoT example 약 수천 개, human-readable format. R1-Zero readability problem(English/Chinese mixing, hollow `<think>` tag)을 고친다.
2. **Reasoning RL (GRPO + rule-based reward).** 공개된 hyperparameter: LR 3e-6, KL coefficient 0.001, GRPO clip ratio ε = 10(의도적으로 loose), rollout temperature 1.0, group size G = 16, max generation 32,768 tokens, 32 unique prompts/step → 512 training samples/step.
3. **Rejection-sampling SFT (RS-SFT)** — **800K distill corpus를 생성하는 stage**다. stage-1 RL model이 prompt당 여러 trace를 sample하고, V3 judge가 readability + correctness로 filter하며, keep된 set은 약 600K reasoning + 200K non-reasoning이다. 이것이 6개 distilled checkpoint의 training set이 되는 corpus다.
4. **Stage-2 Alignment RL** with helpfulness + harmlessness preference rewards(distill corpus와 별개, final R1 model에만 사용).

**공개적으로 알려진 800K의 내용:**

| Slice | Approx. size | What it is | Verifier used |
|---|---|---|---|
| Math (problem → R1 long CoT) | ~200–300K | NuminaMath / olympiads / AIME-style | final boxed answer exact-match (sympy) |
| Code (problem → R1 long CoT → solution) | ~200–300K | LeetCode / APPS / CodeContests | Public unit-test execution |
| Logic / science reasoning | ~50–100K | GPQA-style, logical puzzles | LLM-judge (V3 or V3-reasoning) |
| Non-reasoning SFT | ~200K | Writing, roleplay, translation, Q&A | No verifier; V3-judge for quality filtering |

Report는 **이 pool이, 추가 trick이 아니라, 전이되는 것**이라고 명시한다. 6개 distilled student(Qwen2.5-Math 1.5B/7B/14B/32B, Llama-3.1-8B, Llama-3.3-70B)는 이 corpus로 **pure SFT**된다. student에 RL은 없다. DeepSeek가 직접 제시한 training lesson: **dense student는 자신의 RL로 reasoning structure를 재발견하는 것보다 copied reasoning structure에서 더 큰 benefit을 얻는다**. 32B dense student가 scratch에서 RL을 하면 한 epoch의 R1-trace SFT보다 훨씬 큰 compute가 필요하고 결과도 약하다.

**Opacity line.** 공개 DeepSeek docs는 800K라는 숫자와 broad composition은 제공하지만, source별 breakdown, rejection-sample yield ratio, judge prompt, data-order는 제공하지 않는다. 다음 section의 open reproduction은 이 gap을 메우기 위해 존재한다.

---

## 4. Open R1-distill reproduction — filter가 recipe다

2025년 세 effort, [[bespoke-stratos]], [[openr1]], [[sky-t1]]는 모두 fully-open data로 R1-distill 아이디어를 reproduction한다. 하지만 *얼마나 curate할 것인가, 어떤 filter를 쓸 것인가*에 대한 답은 매우 다르다. 이들을 비교하면 recipe primitive가 무엇인지 가장 선명하게 보인다.

| Attribute | Bespoke-Stratos-17k | OpenR1-Math-220k | Sky-T1-data-17K |
|---|---|---|---|
| Team | Bespoke Labs (Sathiamoorthy et al.) | HuggingFace (Tunstall, Beeching, Lambert, Ben Allal, Penedo, …) | NovaSky / Sky Computing Lab (UC Berkeley) |
| Teacher | DeepSeek-R1 (official API, 671B MoE) | DeepSeek-R1 (HF-hosted + API mix) | QwQ-32B-preview (open weights, local vLLM) |
| Prompt pool | 7K math (NuminaMath / MATH / AIME) + 5K code (APPS / CodeContests / TACO / LeetCode) + 5K science/STILL-2 | 220K math problems only (NuminaMath cn_k12 / olympiads / aops_forum / amc_aime / orca_math + AIME archive) | 10K math + 5K code + 2K science/STILL-2 |
| Sampling | T = 0.6, prompt당 1 trace + format fail 시 최대 3 retry | T = 0.6, prompt당 2 trace(일부는 최대 8×) | T = 0.7, prompt당 1 trace, max 8K tokens |
| Filter — math | 추출된 `\boxed{}` answer에 SymPy symbolic-equivalence | `\boxed{}` answer에 Math-Verify(open-source SymPy) | `\boxed{}`에 SymPy |
| Filter — code | candidate를 public unit test에 실행; 하나라도 fail하면 reject | n/a (math-only corpus) | public unit test에 실행 |
| Filter — science/open | GPT-4o LLM-judge (correct/incorrect verdict) | n/a | GPT-4o-mini LLM-judge |
| Format filter | `</think>` 누락 또는 `\boxed{}` 누락 reject | `</think>` / `\boxed{}` 누락 reject | GPT-4o로 canonical `<\|im_start\|>` chat template으로 rewrite; filler preamble drop |
| Dedup | MinHash cross-prompt | NuminaMath에 built-in; post-dedup 없음 | None published |
| Reject rate | ~30–50% (code가 rejection을 지배) | ~20% (math-only; SymPy strict) | ~5% format failures; re-generate |
| **Final #traces** | **17,000** | **~440K (220K × 2)** | **17,000** |
| Median trace length | ~3K tokens (tail 10K+) | ~5K tokens (tail 30K; 10% >15K) | ~3K tokens (tail 10K) |
| Teacher cost | ~$800 API | ~$10K (HF H100 + API mix) | Negligible (local QwQ) |
| Student training cost | Qwen2.5-32B-Instruct에서 ~$4K (8×H100, hours) | Qwen2.5-7B-Instruct에서 multi-day 8×H100 | Qwen2.5-32B-Instruct에서 $450 (8×H100, 19 hrs) |
| **Flagship student eval** | Stratos-32B: AIME24 ~63%, MATH500 ~93%, LCB ~57% | OpenR1-Qwen-7B: MATH ~80%, AIME24 ~40% (+follow-up GRPO에서 AIME +3–5) | Sky-T1-32B: AIME24 ~43.3%, MATH500 ~82.4%, LCB-Easy ~86.3%, GPQA-Dia ~56.8% |
| Delta vs R1-Distill-Qwen-32B (800K) | 2–3 pts 이내 | (7B; 직접 비교 어려움) | AIME에서 ~20 pts 뒤짐(teacher ceiling: QwQ < R1) |
| License | Apache-2.0 dataset | Apache-2.0 dataset | Apache-2.0 dataset; teacher is open weights |

이 table에서 세 가지 비자명한 lesson이 나온다.

**4.1 Curation은 scale을 이기지만, ceiling까지만 그렇다.** Bespoke-Stratos의 17K는 R1-Distill-Qwen-32B의 800K와 2–3 point 이내에 도달한다. 추가 783K trace의 marginal value는 AIME에서 약 2 point다. 이것은 [[open-thoughts]]가 이후 1000+ ablation으로 검증한 같은 "less is more" lesson이다. 전이되는 것은 prompt의 breadth가 아니라 student의 latent capability와 teacher의 format template이다. ceiling은 teacher다. R1 trace 17K는 AIME에서 QwQ trace 17K와 같지 않다.

**4.2 Verifier가 upper bound를 결정한다.** Open-R1의 Math-Verify는 algebraic/numeric equivalence는 잡지만 geometry와 proof에는 실패한다. Stratos의 SymPy + tests + LLM-judge stack은 coverage가 가장 넓고 그에 따라 가장 강한 student를 만든다. Sky-T1의 open-ended용 GPT-4o-mini judge는 가장 약한 filter이며 가장 약한 GPQA score와 correlate한다. **Stratos ablation**: code-verification을 제거하면 LiveCodeBench gain이 절반으로 줄고, math symbolic equivalence를 제거하면 MATH gain이 절반으로 줄어든다. **OpenThoughts ablation**: question side에서 LLM-labeled *difficulty* filter가 embedding-based나 fastText filter를 이긴다.

**4.3 더 강한 teacher가 항상 더 좋은 teacher는 아니다.** [[open-thoughts]] ablation: **QwQ-32B가 DeepSeek-R1보다 teacher로 낫다**. R1이 평가되는 모든 benchmark에서 R1이 앞서는데도, Qwen2.5-7B student를 training할 때는 그렇다. 설명: (i) QwQ의 output distribution이 Qwen2.5의 base distribution에 더 가깝다 → student에게 distribution shift가 작다. (ii) QwQ trace가 더 짧다(median 3K vs R1의 5K) → 같은 token budget에서 더 많은 (prompt, answer) pair를 본다. (iii) R1의 trace format에는 idiosyncrasy(가끔 중국어, hollow `<think>`)가 있어 filtering 후에도 남아 small student를 혼란스럽게 한다. Sky-T1은 비용이라는 다른 이유로 같은 선택을 했고 더 약한 model을 얻었다. 이는 "student distribution에 가까움"만으로는 충분하지 않으며 teacher 자체도 capable해야 함을 보여준다.

전체 genealogy tree(GPT-4 → Orca → Open-Orca / Dolphin; R1 → R1-Distill → Bespoke-Stratos → Sky-T1 → OpenThoughts)와 clickable teacher metadata, trace count는 [figures/distill-lineage.html](figures/distill-lineage.html)을 보라.

---

## 5. Student가 teacher에게서 상속하는 것

Distillation-as-data는 특정 distribution의 token에 대한 supervised learning이다. student는 전이하려던 부분뿐 아니라 **그 distribution의 모든 것**을 배운다.

**5.1 Stylistic tics.** R1의 trace에는 특징적인 reflection marker가 있다. *"Wait, let me reconsider..."*, *"Hmm, that doesn't seem right..."*, *"Let me verify this step..."*. 우연이 아니다. R1-Zero의 RL pressure 아래 등장한 behavior다([[deepseek-r1]] §"aha moment"). student는 이를 verbatim copy한다. reasoning task에서는 유용하다(tic이 self-check를 하도록 cue를 준다). short-form task에서는 anti-feature다. student가 "2"라고 답하기 전에 "Wait, let me reconsider"라고 말하게 된다. [[orca-2]]의 Prompt Erasing이 가장 principled mitigation이다. tic을 항상 emit하도록 하지 말고, 언제 emit할지 *select*하도록 student를 training한다.

**5.2 Language mixing.** R1의 RL stage는 bilingual(English/Chinese) base에서 실행되었다. Raw R1 output은 English prompt에서도 reasoning step에 Chinese character를 interleave할 때가 있다. R1의 이후 stage에서 cold-start SFT + language-consistency reward가 이를 부분적으로 고치지만, distilled student에는 residual leakage가 보인다. Stratos와 Open-R1은 format-violation trace를 reject한다. 하지만 bilingual leak이 wrapper가 아니라 mid-trace에 있으므로 rejection은 불완전하다.

**5.3 Benchmark contamination.** AIME과 MATH problem은 public이다. teacher가 자기 training 중 answer를 memorization했을 수 있다. student는 그 memorization을 memorization한다. **Open-R1은 이를 flag했다**. AIME25 gain이 AIME24 gain보다 작은데, teacher가 newer problem에는 덜 saturated되어 있기 때문이다. clean-evaluation protocol은 held-out contest(AIME25 when your teacher's cutoff is 2024, USAMO when AIME is contaminated 등)를 report하는 것이다. teacher가 training한 standard eval이 아니다.

**5.4 Refusal patterns.** safety-triggering prompt에 `"I cannot help with that"`을 emit하는 teacher는 student에게 같은 pattern을 가르친다. 종종 *같은 phrasing*으로. refusal이 calibrated되어 있다면 바람직하다. refusal이 over-broad하다면(GPT-4 시대의 "As an AI language model, I cannot ...") 유해하지 않은 prompt에도 student가 이를 보이게 되므로 바람직하지 않다. [[dolphin]]의 "uncensoring" filter는 training 전에 이러한 pattern을 detect하고 제거하는 operation이다. 이는 teacher-bias transfer가 occasional이 아니라 systematic임을 인정하는 post-hoc correction이다. Hartford의 recipe:

```
1. Regex + classifier detect refusal patterns ("As an AI language model…", "I cannot …").
2. Classifier detect alignment-steering preambles (moral disclaimers).
3. Classifier detect formulaic bias disclaimers.
4. Drop those samples.
5. Train on the remainder.
```

이렇게 하면 alignment를 SFT에 bake하는 대신 별도의 training stage로 이동시킨다. 철학적으로 controversial하지만 student의 alignment가 teacher의 alignment가 아니라 *당신의* alignment이기를 원한다면 operationally necessary하다.

**5.5 "wrong-question-correctly" failure.** [[openr1]]은 이를 명시적으로 지적한다. Math-Verify는 final `\boxed{}` answer가 gold answer와 같은지만 확인한다. R1이 problem을 잘못 읽고 *다른* problem을 풀었지만 우연히 gold와 같은 answer를 냈다면(multiple-choice나 numeric answer의 coincidence에서 흔함), trace는 filter를 통과하고 valid-looking하지만 wrong question을 푸는 reasoning path로 student를 poison한다. outcome filter는 이를 잡지 못한다. process reward(ch-24)가 유일한 defense다.

---

## 6. OpenThoughts — recipe meta-experiment

[[open-thoughts]]는 data-recipe space에서 1,000+ ablation을 실행한 2025년 project다. reasoning distillation에서 실제로 무엇이 중요한지에 대한 *empirical* answer의 단일 최대 source다. practitioner가 internalize해야 할 headline finding:

- **Question당 여러 answer를 sampling하는 것이 가장 쉬운 diversity trick이다** — source당 ≥16× expansion에서 non-trivial gain. (Open-R1은 problem당 2×, Stratos는 1×.)
- **강한 source concentration이 source diversity를 이긴다.** 소수의 top-quality problem pool이 더 넓지만 noisy한 mix보다 낫다.
- **Answer-side filter는 keeping everything을 이기지 못한다.** format violation을 잡을 만큼 aggressive한 filter가 있으면, 추가 answer filtering(difficulty pruning, length pruning)은 제거하는 noise보다 잃는 signal이 더 많다.
- **Question-side filtering이 answer-side filtering보다 중요하다.** *question*에 대한 LLM-labeled difficulty와 response-length filter가 embedding-based나 fastText heuristic을 이긴다.
- **Deduplication은 domain-sensitive하다.** math와 science는 exact dedup; **code는 no dedup**(syntactically similar solution을 가진 서로 다른 problem을 collapse하면 안 된다).
- **Teacher choice는 teacher benchmark score에 대해 monotone하지 않다.** R1이 모든 target benchmark에서 더 높은 점수를 내는데도 Qwen2.5-7B의 teacher로는 QwQ-32B > R1이다.

OpenThinker3-7B model, 즉 QwQ-32B를 teacher로 한 OpenThoughts3-1.2M corpus(850K math + 250K code + 100K science)에 SFT된 Qwen2.5-7B-Instruct는 AIME25 53%, LiveCodeBench 06/24-01/25 51%, GPQA-Diamond 54%에 도달한다. 논문 시점 기준 가장 강한 open-data 7B reasoning model이다. 위 recipe primitive들은 load-bearing이다. 하나라도 ablate하면 측정 가능한 점수를 잃는다.

---

## 7. Licensing cliffs — 어떤 teacher output이 재배포 가능한가

이 section은 화려하지 않지만 load-bearing이다. 세 open reproduction이 다른 teacher를 쓰는 이유는 recipe preference가 아니라 licensing이다.

| Teacher | Weight license | Output-license question | Redistributable corpus? |
|---|---|---|---|
| GPT-4 / GPT-4o (OpenAI API) | Proprietary | OpenAI ToS는 역사적으로 output을 경쟁 모델 training에 사용하는 것을 금지했다. dataset host는 dataset을 Apache-2.0으로 release하더라도 API ToS를 준수해야 한다. | **Contested.** Dolphin-v1은 *dataset*에 Apache-2.0을 주장하지만 GPT-4 output의 downstream use는 generator가 동의한 API ToS의 지배를 받는다. commercial setting에서는 lawyer-gated. |
| Claude (Anthropic API) | Proprietary | Anthropic ToS도 output으로 경쟁 model을 training하는 것을 금지한다. | training-competing-model purpose로는 재배포 불가. |
| DeepSeek-R1 (671B MoE) | DeepSeek release 기준 **MIT on weights** | DeepSeek는 distillation을 포함한 training용 model output을 명시적으로 허용한다. | **Yes**, with attribution. Bespoke-Stratos, Open-R1, OpenR1-Math-220k가 모두 이에 의존한다. |
| QwQ-32B-preview (Alibaba) | **Apache-2.0** (open weights) | Output license는 model license에서 상속된다. Apache-2.0은 training을 포함한 derivative work를 허용한다. | **Yes.** Sky-T1-data-17K와 OpenThoughts3가 모두 활용한다. |
| Llama-3.x (Meta) | **Llama-3 Community License** (custom; not OSI) | *다른* Llama-3-derivative를 training하는 output 사용은 허용된다. *non-Llama* model training에 output을 쓰는 것은 700M MAU threshold까지만 허용된다. 명시적 "attribute Llama" 조항. | Yes, with the MAU and attribution constraints. |
| Qwen-2.5 (Alibaba) | Apache-2.0 (most sizes) | QwQ와 동일. | Yes. |

Teacher-output license는 distill corpus가 *public dataset으로 releasable*한지, 아니면 *internal use only*인지 가르는 가장 큰 factor다. 2025년에 DeepSeek-R1과 QwQ가 teacher로 consolidation된 이유는 모든 domain에서 최고의 trace를 만들기 때문이 아니다. 이들의 output만이 permissive license 아래 재배포될 수 있기 때문이다. 바로 그래서 open-reproduction community가 recipe를 반복 개선할 수 있다.

두 second-order consequence:

- **Student license는 teacher license의 downstream이다.** GPT-4 output으로 training한 student는 author가 고르는 어떤 license로도 release될 수 있다. 하지만 student의 *weight는 여전히 teacher ToS risk를 carry한다*. OpenAI가 distilled student에 대해 enforcement한 적은 없지만, legal position은 distilling이 ToS violation이라는 데 남아 있다.
- **Dataset commercial use는 research use보다 좁다.** Stratos는 R1 output이 freely licensed이기 때문에 Apache-2.0이다. Dolphin-v1의 dataset Apache-2.0은 underlying GPT-4/GPT-3.5 output이 model-license-governed가 아니라 API-governed이므로 더 fragile하다.

Commercial product라면 현재 safe path는 다음이다. **teacher로 R1 또는 QwQ를 쓰고, dataset에 Apache-2.0 / MIT를 적용하고, teacher를 attribute하고, auditability를 위해 filtering code를 open해 둔다.**

---

## Connections and what's next

- **ch-19 (rejection sampling / self-distillation)** — 다른 branch: *자기 자신의* rollout을 rejection-sample한다. 이 장은 licensing consequence를 동반한 *다른 누군가의* rollout에 대한 rejection-sampling이다.
- **[[orca]] / [[orca-2]]** — system-prompt-scaffolded distillation template. §1의 16-message list는 "chat-tuned teacher에서 varied reasoning을 어떻게 elicit하는가"의 reference point로 남아 있다.
- **[[distilling-step-by-step]]** — joint-label-and-rationale multi-task training; R1-Distill의 conceptual ancestor.
- **[[deepseek-r1]] / [[deepseek-r1-distill-synth]]** — teacher pipeline과 800K corpus composition. ch-24(RLVR at scale)는 R1-Zero → R1 RL path를 다룬다.
- **[[bespoke-stratos]] / [[openr1]] / [[sky-t1]]** — 세 open-reproduction recipe. §4의 table은 filter stack을 선택해야 할 때 보게 될 비교다.
- **[[open-thoughts]]** — 1000-ablation recipe study. §6의 finding은 모든 새 reasoning-SFT corpus의 empirical baseline이다.
- **ch-21 (taxonomy-driven synthesis)** — GLAN / Nemotron / Phi-textbooks. 이 장의 *complement*다. distill할 강한 teacher가 없고 taxonomy에서 synthesize해야 할 때 무슨 일이 생기는가.
- **ch-22 (quality / diversity / gradient selection)** — 이 장이 만든 어떤 corpus 위에도 올라가는 *sample-selection* layer: LESS, DEITA, Prismatic-Synthesis.
- **ch-23 (model collapse)** — 그 자체가 distilled model에서 다시 distill할 때 무슨 일이 생기는가. 이 장은 아직 trigger하지 않지만 ch-23이 측정하는 iterative-distillation failure mode다.

## Further reading

- [[orca]] — Mukherjee 2023; 16 system messages, 5M ChatGPT + 1M GPT-4 progressive learning.
- [[orca-2]] — Mitra 2023; 5-strategy instruction, Prompt Erasing.
- [[distilling-step-by-step]] — Hsieh 2023 ACL; joint (label, rationale) multi-task; T5-770M beats PaLM-540B few-shot.
- [[deepseek-r1]] — Guo et al. 2025 Nature; 4-stage pipeline, GRPO hparams, 800K corpus composition.
- [[deepseek-r1-distill-synth]] — blog/README extract; corpus-opacity caveats.
- [[bespoke-stratos]] — Sathiamoorthy 2025; 17K / $800 / SymPy+tests+LLM-judge.
- [[openr1]] — HF Open-R1 2025; 220K × 2-trace math corpus; Math-Verify.
- [[sky-t1]] — NovaSky 2025; $450 QwQ recipe; GPT-4o rewriter.
- [[open-thoughts]] — Guha 2025; 1000+ ablations; QwQ > R1 as teacher.
- [[dolphin]] — Hartford 2023–2025; Orca reproduction + refusal filter as the canonical teacher-bias-removal recipe.

## Companion visualization

**[figures/distill-lineage.html](figures/distill-lineage.html)** — 두 distillation lineage의 interactive genealogy tree. 왼쪽 branch: GPT-4 → Orca → Open-Orca / Dolphin(system-prompt-scaffolded, chat-teacher era). 오른쪽 branch: DeepSeek-R1 → R1-Distill / Bespoke-Stratos / Open-R1; QwQ-32B → Sky-T1 / OpenThoughts(reasoning-teacher era). node를 click하면 teacher model, trace count, filter stack, license terms, flagship student eval number가 보인다. generation을 가로질러 어떤 decision이 이어지는지(system-prompt scaffolding → `<think>` wrapper; MinHash dedup → cross-prompt filtering), 어떤 decision이 local fashion choice인지(number of traces, T=0.6 vs 0.7) internalize하는 데 사용하라.
