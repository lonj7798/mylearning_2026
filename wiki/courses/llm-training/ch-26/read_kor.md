<!-- chapter: ch-26
     track: synthetic
     title: Modality — Tool and Function-Calling Data
     sources: [[toolformer]], [[toolllm]], [[apigen]], [[apigen-mt]], [[toolace]], [[xlam]], [[bfcl]], [[hammer]], [[nexusraven]], [[gorilla]], [[api-bank]], [[glaive-function-calling]], [[granite-function-calling]]
     figures: figures/tool-pipeline.html
-->

# 26장 — 모달리티: Tool and Function-Calling Data

> **핵심 통찰.** Tool-use data는 **environment가 label인** 모달리티다. Math(자유 형식 CoT를 symbolic equivalence로 채점)나 code(unit test가 program을 채점)와 달리, function call은 그것이 *schema에 맞게 parse되고, 실제 implementation에 대해 execute되며, user's intent를 충족하는지*로 채점된다. 이 장의 모든 파이프라인, 즉 Toolformer, ToolLLM, APIGen, ToolACE, APIGen-MT는 하나의 질문에 대한 서로 다른 답이다. *수백만 후보를 filter할 만큼 싸고, GPT-4 teacher 자체의 error를 이길 만큼 강한 verification signal을 어떻게 얻는가?*
>
> **가이드라인.** Generator가 아니라 verifier를 중심으로 파이프라인을 만들어라. Minimum viable stack: (1) schema-enforced format check(무료), (2) reference implementation에 대한 sandboxed execution(비싸지만 하중을 받는다. 제거하면 BFCL 약 11 points 손실), (3) `(query, call, result)`에 대한 semantic LLM-as-judge(wrong-unit / wrong-target error를 잡음). Multi-turn에는 conversational realism을 rollout하기 *전에* correctness를 고정하는 blueprint phase를 추가하라. 단일 source로만 학습하지 말라. xLAM / ToolACE / Hammer split은 각 source가 BFCL axis 하나를 덮고 다른 곳에는 gap을 남긴다는 점을 보여준다.

---

## 이 장이 필요한 이유

Function calling은 모델이 학습하는 loss와 배포 시 비교되는 metric이 완전히 갈라진 첫 LLM 모달리티다. Math SFT에서는 gold CoT로 학습하고 같은 gold format으로 평가한다. Dialogue에서는 preferred response로 학습하고 대략 동등한 judge에 대해 평가한다. Tool use에서는 `{"name": "get_weather", "arguments": {"location": "Paris"}}`로 학습하고, literal representation을 normalize하고 어떤 kwarg order도 허용하는 AST matcher로 채점되는 `get_weather(location="Paris")` Python syntax에 대해 평가한다([[bfcl]]). Training distribution, call template, evaluation parser, BFCL leaderboard는 같은 행위에 대한 네 가지 다른 specification이며, 문헌의 모든 data pipeline은 그것들을 이어 붙이려는 시도다.

이 장은 그 bridge를 다섯 세대로 따라간다.

1. **[[toolformer]] (2023)** — perplexity-delta filter를 통한 raw text 위 self-supervised annotation. 모델 자신보다 강한 teacher가 없다.
2. **[[toolllm]] (2023)** — 16K real REST APIs에 grounded된 synthetic trajectories. Brittle ReACT를 DFS-DT search로 대체. *Real substrate, synthetic supervision.*
3. **[[apigen]] (2024)** — format / execution / semantic 세 check를 모두 통과하지 않으면 sample을 거부한 첫 pipeline. 약 40% rejection은 bug가 아니라 feature다.
4. **[[apigen-mt]] + [[toolace]] (2024–25)** — blueprint-then-rollout으로 verification을 multi-turn으로 확장하고, complexity-controlled multi-agent dialog로 coverage를 넓힘.
5. **Benchmark-shaped specialists (Hammer / NexusRaven / Granite)** — 각각 하나의 [[bfcl]] axis를 하나의 data trick으로 target한다.

관통선인 **verifier is the data pipeline** 때문에 이것은 [[ch-18]]의 일반 recipe에 접히지 않고 독립된 장이 된다. Generator는 대체로 교체 가능하다(ablations에서 DeepSeek-Coder-V2와 GPT-4 차이는 약 2%). Verifier는 그렇지 않다.

---

## 1. Toolformer: annotation bootstrap

[[toolformer]]는 개념적 서문이다. 2023년, 16K-API corpus가 존재하기 전의 open question은 이것이었다. *인간이나 더 강한 teacher 없이 plain text에서 tool-use supervision을 생성할 수 있는가?* Toolformer의 답은 yes다. 단, 모델이 스스로 verify할 수 있다면. Signal은 **API result가 보일 때 future tokens의 loss가 감소하는 정도**다.

**Stage 1 — propose.** Inline API call의 few demonstrations로 GPT-J 6.7B를 prompt한다. 각 token position `i`에서 모델이 API call을 시작할 확률을 계산한다. 그 확률이 **5%**(appendix threshold)를 넘는 position을 유지한다. Scan한 약 10M positions 중 약 2%가 살아남는다.

**Stage 2 — sample + execute.** 살아남은 position에서 candidate call을 sample한 뒤 다섯 tool(QA, Wikipedia search, calculator, translation, calendar) 중 하나에 실제로 execute한다. 실제 반환 result `r`을 저장한다.

**Stage 3 — filter by perplexity delta.** Position `i` 뒤에 온 continuation tokens `y`에 대해:

```
L_no_call    = −Σ_t log P(y_t | prompt)
L_call_empty = −Σ_t log P(y_t | prompt + "[API(args) → ]")
L_call_full  = −Σ_t log P(y_t | prompt + "[API(args) → result]")

Δ = min(L_no_call, L_call_empty) − L_call_full
```

Default `τ = 1.0` nat에서 `Δ > τ`이면 accept한다. 두 가지 subtlety가 있다.

- **왜 `L_call_empty`와 비교하는가?** 그렇지 않으면 모델이 단지 `Calculator(...)` 문자열을 본 것만으로 credit을 받는다. `min`은 result 자체가 predictive value를 추가하도록 강제한다. 이것이 가장 이른 "reward outcome, not format" rule이다. APIGen의 semantic check와 BFCL의 relevance-detection은 downstream variant다.
- **왜 τ = 1.0 nat이 공격적인가?** 약 5-token window에서 1-nat lift는 perplexity를 `e`배 줄이는 것이다. 논문 Table 2는 수십만 candidates 중 tool당 몇천 annotations만 남긴다. Budget은 99%+ rejection을 가정한다.

Toolformer는 plan하지도, multi-turn을 synthesize하지도, retrieve하지도 않는다. 다섯 tool은 항상 prompt 안에 있다. 하지만 이후 모든 system이 물려받는 세 rule을 세운다. generation 중 real call을 execute하라. validity뿐 아니라 usefulness를 test하라. 높은 rejection rate는 feature다.

---

## 2. ToolLLM과 DFS-DT: real APIs 위 synthetic trajectories

[[toolllm]]은 *한 call을 annotate*하는 데서 *전체 solution path를 생성*하는 단계로 뛴다. 논문의 기여는 **ToolBench**(16,464 real RapidAPI endpoints), scenario taxonomy, 그리고 brittle ReACT rollout을 대체하는 depth-first decision-tree trajectory generator인 **DFS-DT**다.

**Real vs synthetic.** 오래 남는 기여는 split이다. API catalog + executions는 real이고, user instruction + reasoning trace + final answer는 synthetic이다(`gpt-3.5-turbo-16k`가 생성). 이후 모든 pipeline은 이 split을 상속한다. APIGen은 API set을 좁히지만(3,673 executable) 유지한다. ToolACE는 API set을 변이시키지만(26,507 evolved) "가능하면 execute"를 유지한다. *Trace 아래의 environment는 항상 real이고, trace는 항상 synthetic이다.*

**Scenario grid.** Data는 3×2 grid를 돈다. `G1/I1` single-tool, `G2/I2` intra-category multi-tool, `G3/I3` intra-collection multi-tool. Public release: **126,486 instances, 469,585 real API calls, instance당 약 4 reasoning steps**.

**DFS-DT.** *Training* trajectory를 생성하는 것은 inference에서 trajectory 하나를 실행하는 것과 다르다. 원하는 것은 *어떤* correct path라도 나올 확률을 최대화하는 것이다. 실패 rollout은 순수 API 낭비이기 때문이다. Plain ReACT에는 recovery가 없다. 첫 wrong action이 이후 모든 observation을 오염시킨다. DFS-DT는 retraction과 함께 search한다.

```python
def dfs_dt(instruction, apis, model, max_depth, beam):
    root = Node(history=[], depth=0)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal() or node.depth >= max_depth:
            if node.is_successful():
                return node.trajectory     # first accepted trajectory wins
            continue
        candidates = model.sample_actions(node.history, apis, k=beam)
        for (thought, action) in sort_by_score(candidates):   # preorder
            obs = execute(action)
            if obs.is_error:
                continue                    # retract: do NOT commit this child
            child = Node(history=node.history + [(thought, action, obs)],
                         depth=node.depth + 1)
            stack.append(child)
    return None     # all branches exhausted; reject this instance
```

Appendix는 *`if obs.is_error: continue` retraction이 없으면 DFS-DT가 ReACT로 퇴화한다*고 적는다. 전체 lift는 backtrack에서 온다. 동일 API budget의 Table 3:

| Setting | ReACT | ReACT@N | DFS-DT |
|---|---|---|---|
| I1 (single-tool) | 42.2 | 47.7 | **57.3** |
| I2 (intra-cat)  | 30.0 | 34.3 | **48.2** |
| I3 (intra-col)  | 21.7 | 26.0 | **43.2** |

가장 어려운 setting에서 15–20 point lift가 multi-tool annotation을 tractable하게 만든다. ToolEval(LLM-judge)은 **87.1% pass-rate agreement with humans**를 보고한다. 유용할 만큼 높지만 맹신할 만큼 높지는 않다. APIGen이 메우는 gap이 이것이다.

---

## 3. APIGen의 three-layer verifier

[[apigen]]은 field를 다시 framing한다. Synthetic function-calling data는 모든 sample이 순서대로 세 independent check를 통과할 때만 믿을 수 있다.

1. **Format check** — JSON parse, required fields present, types match schema. Malformed JSON, wrong arg name, string-for-int를 reject.
2. **Execution check** — Python sandbox, 5-second timeout, reference implementation에 대해 실행. Exception이나 timeout이면 reject.
3. **Semantic check** — GPT-4 judge가 `(query, call, execution_result)`를 보고 "does the call correctly fulfil the query?"에 "Yes"라고 답해야 함. Non-"Yes"는 reject.

**구체적 숫자.** 3,673 executable APIs(ToolBench의 16K 중 Salesforce가 endpoint를 실행했거나 Python mock을 작성한 curated subset). Four data types에 걸쳐 60,000 accepted samples — simple(약 40%), multiple(약 25%), parallel(약 20%), parallel-multiple(약 15%). 각 API는 서로 다른 arg combination으로 약 16× 등장. Dedup: `(query, call)`에 대한 MinHash. Teacher: DeepSeek-Coder-V2-Instruct primary, GPT-4 ablation.

**각 layer가 제 몫을 한다는 것을 증명하는 ablation:**

| Verifier config | BFCL-V1 overall | Δ |
|---|---|---|
| Full 3-layer (format + execution + semantic) | **88.24** | — |
| Remove semantic check | 82.2 | −6.0 |
| Remove execution check | 77.3 | −10.9 |
| Remove format check | 70.1 | −18.1 |

- **Format-only는 약 70%에 머문다.** 2023년 Glaive-style ceiling([[glaive-function-calling]])이다.
- **Execution이 가장 큰 single-layer lift다(제거 시 −11 points).** 그래서 APIGen은 3,673 APIs에서 병목이 걸렸다. Salesforce가 reference implementations를 필요로 했기 때문이다.
- **Semantic은 남은 6%를 잡는다.** Error 없이 실행되지만 wrong unit, wrong target, right function with wrong arg semantics처럼 wrong question에 답하는 call들이다. LLM judge(또는 human)만 이것을 잡는다.

Rejection rate: 세 stage를 합쳐 **약 40%**. 60K gold를 만들려면 raw를 약 100K 생성한다. 60K corpus는 "teacher output 60K"가 아니라 "strict gate를 살아남은 60K"다. Stage별 reject 예시는 [figures/tool-pipeline.html](figures/tool-pipeline.html)을 보라.

**Downstream:** Corpus는 [[xlam]]을 학습한다. **xLAM-7B(Mistral base)는 BFCL-V1 88.24%**에 도달하며, 2024년 9월 release 당시 <13B 중 1위다. xLAM-2 staged recipe, 즉 **APIGen-60k SFT → APIGen-MT-5k SFT → optional DPO(β=0.1) on (correct, hallucinated-name) pairs**는 2025년 기준 가장 명확한 open multi-turn FC specialist recipe다.

---

## 4. Multi-turn: APIGen-MT의 blueprint-then-rollout

Single-turn은 APIGen으로 거의 풀렸다. Multi-turn은 약 12 messages, call 사이 state mutation, correctness가 (call, observation, reasoning) triples의 *sequence*인 문제다. 이것을 [[apigen-mt]]가 다룬다. Trick은 **correctness와 realism을 분리**하는 것이다.

**Phase 1 — blueprint.** Phase 1은 dialog를 만들지 않는다. Structured task config를 만든다.

```jsonc
{
  "domain": "airline",
  "user_persona": "budget traveller, prefers refundable fares",
  "instruction": "Book JFK→LAX under $250 and a hotel at LAX under $150 for Friday.",
  "ground_truth_actions": [
    {"api": "search_flights", "args": {"from": "JFK", "to": "LAX", "max_price": 250}},
    {"api": "book_flight",    "args": {"flight_id": "$F.id"}},
    {"api": "search_hotels",  "args": {"near": "LAX", "max_price": 150}},
    {"api": "book_hotel",     "args": {"hotel_id": "$H.id"}}
  ],
  "expected_final_state": {"flight_booked": true, "hotel_booked": true}
}
```

Blueprint는 dialog가 존재하기 전에 세 validator를 통과한다. **action validation**(call별 format/execution/policy-check, APIGen-style), **alignment validation**(LLM committee: sequence가 intent를 만족하는가?), **semantic review**(committee aggregation + refinement; 논문은 GPT-4o + DeepSeek V3 사용). Architectural move는 **reverse task recombination**이다. Validated primitive에서 complex blueprint를 compose한다.

**Phase 2 — rollout.** 두 LLM actor(user-simulator, agent)가 conversation을 role-play한다. Agent는 real call을 execute한다. User-simulator는 persona를 들고 push back, clarification, constraint addition을 한다. **Rejection sampling**은 `expected_final_state`에 도달하고 ground-truth actions와 match하는 trajectory만 유지한다. Reported **약 70% task-collection success rate**, 평균 약 **12 turns**, τ-bench substrate의 15 read + 13 write APIs.

**Blueprint-first가 rollout-then-verify보다 나은 이유.** 12-turn dialog에서 per-turn correctness가 95%이면 end-to-end correctness는 54%다. sample당 약 \$1 rollout 절반을 버린다. Blueprint-first는 ground-truth를 *한 번* 고정한다. Rollout은 그것을 *realise*할 뿐이다. 둘째, end-to-end dialog verification("성공했나?")은 주관적이다. Blueprint-then-rollout은 알려진 plan에 대한 structural check로 분해한다.

결과: **BFCL-V3 multi-turn에서 xLAM-2-8B는 69.25%**에 도달한다. 단 5K trajectories로 인용된 GPT-4o baselines를 넘는다.

---

## 5. ToolACE: self-evolution을 통한 breadth

[[toolace]]는 다른 축을 밀어붙인다. **API diversity**다. APIGen이 executable implementations를 요구하기 때문에 3,673 APIs에서 막힌다면, verifiability를 유지하면서 26,507까지 어떻게 가는가?

**Tool Self-Evolution Synthesis (TSS).** 3K real API seed에서 LLM이 세 operator로 mutate한다. parameter extension, domain transfer(weather→stock), functionality refinement. Schema parseability, name uniqueness, LLM-judge novelty/utility로 filter한다. 결과: **390 domains에 걸친 26,507 APIs**로 2024년 기준 largest public pool. Tradeoff: 대부분 real implementation이 없다. Response는 LLM-simulated다.

**Multi-Agent Interactive Dialog (MAI).** 세 LLM roles(user / assistant / tool-simulator). **Complexity evaluator**가 각 dialog를 5 difficulty levels로 classify하고, target mix에 맞추도록 generation을 *condition*한다. simple single-call(약 30%), multiple(약 25%), parallel(약 20%), nested/multi-turn(약 15%), info-incomplete(약 10%). 이 distribution은 BFCL category distribution에 맞춰 hand-tuned되었다. **Benchmark-shaped data**의 명시적 사례다(§7).

**Dual-layer verification.** Rule-based(schema + param + type + execution where mock exists) plus model-based(GPT-4 judge, 3-way verdict: query-clarity / call-correctness / response-consistency). 약 40% rejection, final 11,300 dialogs.

**ToolACE-8B(Llama-3.1-8B base): BFCL-V1 91.41%**로 matched scale에서 xLAM-7B의 88.24%를 이긴다. Ablation: TSS 제거 −4.3%, complexity controller 제거 −3.1%, model-judge 제거 −5.2%, rule-checks 제거 −2.8%. APIGen 대비 솔직한 tradeoff: API는 10× 더 많지만 real execution은 적다. Tool-LLM은 그럴듯하지만 틀린 output을 hallucinate할 수 있다. 그래서 §8 recipe는 *두 source를 모두* 사용한다.

---

## 6. BFCL: benchmark가 data spec이 된 방식

[[bfcl]]의 **ability-axis decomposition**은 현재 모든 pipeline이 따르는 data-design spec이다.

**일곱 category:** Simple(1 call, 1 function) · Multiple(≥2 candidates 중 right one 선택) · Parallel(≥2 calls, same function) · Parallel-Multiple(≥2 calls, multiple functions) · **Relevance-Detection**(irrelevant query → correct answer는 *no call*) · Multi-Turn(V3+, stateful) · Multi-Step(하나의 task를 위한 sequential calls). APIGen의 four types, ToolACE의 five levels, Granite의 seven capabilities([[granite-function-calling]])와 비교하라. 같은 taxonomy가 재발견된다. **Eval taxonomy를 정하는 쪽이 data-generation taxonomy를 정한다.**

**AST matcher.** Scoring은 predicted and gold calls를 `(name, kwargs)`로 parse한 뒤 name exact-match, kwargs는 key별 sorting, whitespace 제거, literals canonicalisation(`1.0 ≡ 1`, `"red" ≡ 'red'`), required args present를 수행한다. `get_weather(city="Paris")` vs `get_weather(city='Paris',)`는 tolerate하지만 `get_weather(loc="Paris")`는 reject한다. *Representation에는 관대하고 semantics에는 엄격하다.*

**Version과 유도된 data target:**

| Version | Key addition | Induced data requirement |
|---|---|---|
| V1 (Feb 2024) | 7 categories, single-turn | APIGen / NexusRaven ~100K |
| V2 Live (Aug 2024) | +1,500 real user queries | ToolACE complexity sampler; Hammer irrelevance aug |
| V3 Multi-turn (Sep 2024) | Stateful multi-turn | APIGen-MT 5K trajectories |
| V4 Agentic (2025) | Long-horizon + web/memory | SWE-Gym agent data ([[ch-27]]) |

**pass^k.** V3+는 *모든* k independent trials에서 성공해야 한다. xLAM-2-70B τ-bench: **pass^1 = 56.2%, pass^4 = 39.4%**. 평균적으로 맞는 것만이 아니라 *structurally consistent* trajectories를 만드는 pipeline(APIGen-MT의 blueprint anchor)에 보상하는 consistency gap을 드러낸다.

현재 2025 snapshot: top open <13B — ToolACE-8B, xLAM-2-8B, Hammer 2.1. top open overall — xLAM-2-70B-fc-r. Frontier model도 irrelevant query의 약 10%에서 여전히 tool을 call한다.

---

## 7. Benchmark-shaped specialists

BFCL이 tool use를 axis로 분해하자, 각 axis를 특정 data trick으로 target할 수 있게 되었다.

**Hammer — function-name masking을 통한 relevance.** Small on-device models는 **naming bias** 때문에 relevance-detection에 실패한다. Query에 "email"이 등장하면 `send_email`을 발사한다. [[hammer]]의 fix는 architecture가 아니라 augmentation이다. (a) training data의 **30%**에서 tool name을 random placeholder `func_[a-z0-9]{6}`로 sample 전체에서 일관되게 대체한다. (b) gold = refuse / clarify인 **약 30%** irrelevance samples. Masking-ratio ablation: 30%가 optimum이다. 50%는 recall을 망치고, 10%는 너무 약하다. Combined lift: BFCL relevance에서 **+13 points**. Hammer-7B는 약 90%로 GPT-4와 match한다.

**NexusRaven — curriculum을 통한 nested calls.** [[nexusraven]]의 training mix는 simple(60%) / parallel(20%) / **nested(20%)**다. Nested example은 `save_file(name="r.txt", content=summarize(translate("texto", to="en")))`처럼 생겼다. Ablation: **nested 제거 → nested-eval track에서 −15 points**, simple/parallel에는 zero drop. Data가 모델에게 "arguments can themselves be function calls"라고 알려준다. Distribution에 없으면 어떤 general SFT도 이것을 가르치지 못한다.

**Gorilla — retriever-aware fine-tuning.** [[gorilla]]는 training prompt에 top-k retrieved API docs를 포함한다. 모델은 `recall API name from memory`가 아니라 `condition on doc → emit call`을 학습한다. 16K instruction-API pairs에서 80% noisy retrieval + 20% oracle. **Hallucination rate: retriever-aware 11% vs GPT-4 without 40%**로 4× 감소. API pool이 수백 tool을 넘으면 serving time의 모든 modern pipeline은 retrieval을 사용한다. Gorilla는 training에도 retrieval이 들어가야 함을 보였다.

**Granite — multi-source mix.** [[granite-function-calling]]의 seven-capability taxonomy(Nested / Parallel / Multiple / Multi-turn / Relevance / Sequencing / Slot-Filling)는 blend로 덮는다. APIGen 25% / ToolLLM 20% / Glaive 15% / Nexus 10% / IBM in-house 20% / general chat 10%. Per-capability resampling으로 각 capability가 mix의 ≥10%가 되게 한다. Naive equal-mix는 tuned mix 대비 5–10 points를 잃는다. ToolLLM 제거 → multi-turn 12 points drop. Nexus 제거 → nested 18 drop. relevance slice 제거 → relevance 10 drop. **각 source는 한 capability에는 load-bearing이고 나머지에는 null이다.**

---

## 8. Drop-in recipe

이 장을 open data 위 7B-class FC specialist recipe로 결합하면 다음과 같다.

```
Mix (~200K samples)
├── 25% APIGen-FC-60k             — single-turn, 3-layer verified
├── 20% ToolBench (DFS-DT filter) — real-API multi-tool trajectories
├── 15% APIGen-MT-5k (upsampled)  — multi-turn blueprint-verified
├── 10% ToolACE nested + info-incomplete subset
├── 10% Hammer masking + irrelevance aug (applied to above)
├── 10% NexusRaven nested-call curriculum
└── 10% OpenHermes subset         — preserves chat quality

Verifier chain for any new-source additions:
  1. Format check          — JSON parse + schema + type check
  2. Execution check       — Python sandbox, 5-s timeout
  3. Semantic check        — GPT-4o judge on (query, call, result)
  4. MinHash dedup         — on (query, call), threshold 0.9
  Expected acceptance ~55–60%.

Call format (match the BFCL AST matcher):
  - Tool schemas in system prompt (JSON-schema style).
  - Assistant emits OpenAI-compatible `tool_calls` JSON.
  - Tool responses in `tool` role messages.

Training (xLAM-2 staged):
  Stage 1: SFT on single-turn mix   (LR 2e-5, 3 epochs, 8k seq)
  Stage 2: SFT on multi-turn mix    (LR 1e-5, 2 epochs, 16k seq)
  Stage 3 (optional): DPO on (correct, hallucinated-name), β=0.1
```

운영 rule 두 가지.

- **Call template를 AST matcher에 맞춰라.** OpenAI JSON 대신 Glaive의 `<functioncall>{...}</functioncall>` XML로 학습하면 inference-time format translation 때문에 5–10 points가 나간다.
- **Shipping 전에 relevance-detection으로 gate하라.** BFCL overall 92%지만 relevance 65%인 모델은 wild에서 irrelevant query의 35%에 tool을 hallucinate한다. 이것이 production failure mode 1순위다.

---

## Connections and what's next

- **[[ch-18]]** — Synthetic-data design pattern. APIGen의 3-layer verifier는 이 pattern의 verify step에 이빨을 단 것이다.
- **[[ch-19]]** — Generation methods. Self-Instruct / Evol-Instruct / Magpie는 *generator* 쪽이다. 여기의 모든 pipeline은 그중 하나를 쓴다(ToolLLM = Self-Instruct-style; ToolACE의 TSS = Evol-Instruct-over-APIs).
- **[[ch-25]]** — Multi-turn conversation synthesis. APIGen-MT의 Phase-2 rollout은 blueprint로 *constrained*된 그 machinery다.
- **[[ch-27]]** — Agentic trajectories. Tool use를 long-horizon environment-grounded tasks(web, terminal, repo)로 확장한다. BFCL-V4가 eval bridge다.
- **[[ch-28]]** — Long-context synthesis. Multi-turn tool use는 post-training에서 쓸 수 있는 더 강한 long-context signals 중 하나다.

## Further reading

- [[toolformer]] — Schick et al. 2023. Self-supervised annotation origin; filter derivation은 §4를 읽어라.
- [[toolllm]] — Qin et al. 2023. ToolBench + DFS-DT; real-substrate / synthetic-supervision split.
- [[apigen]] — Liu et al. 2024. 3-layer verifier와 Table 4 ablation.
- [[apigen-mt]] — Prabhakar et al. 2025. Multi-turn을 위한 blueprint-then-rollout.
- [[toolace]] — Liu et al. 2024/ICLR 2025. Self-evolving API pool + complexity-controlled dialog.
- [[xlam]] — Zhang et al. 2024/2025. Open model family; staged recipe는 §3.
- [[bfcl]] — Patil et al. 2024. Taxonomy가 data taxonomy가 된 benchmark.
- [[hammer]] / [[nexusraven]] / [[gorilla]] / [[granite-function-calling]] / [[glaive-function-calling]] / [[api-bank]] — specialists와 historical baselines.

## Companion visualization

**[figures/tool-pipeline.html](figures/tool-pipeline.html)** — APIGen의 three-layer verifier를 보여 주는 interactive walkthrough. 각 stage(format → execution → semantic)를 클릭하면 구체적 reject example, 해당 layer의 pass rate, 제거 시 누적 BFCL impact를 볼 수 있다. *왜 rejection rate가 feature인지* 내재화하고, 세 independent layer 전체에서 약 40% total reject rate가 어떻게 구성되는지 calibration하는 데 사용하라.
